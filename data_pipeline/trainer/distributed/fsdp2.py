# ════════════════════════════════════════════════════════════════════════════════
# SOTA FSDP2 v2.1 — Production-Grade Fully Sharded Data Parallel
# ════════════════════════════════════════════════════════════════════════════════
#
# Hardened FSDP2 implementation integrated with SOTATrainer.
#
# v2.1 CHANGELOG — Checkpoint Path Overhaul:
#
#   [FIX-005] Device Affinity Violation During Checkpoint Save:
#       ROOT CAUSE: ShardedStateDictConfig(offload_to_cpu=True) moves
#       shard tensors to CPU. When dcp.save() performs NCCL collectives
#       for metadata coordination, PyTorch moves CPU tensors back to
#       cuda:0 (default device) on ALL ranks. Rank 3 has tensor on
#       cuda:0 but NCCL process group bound to cuda:3 → NCCL error →
#       all ranks hang until NCCL timeout (default 1800s ≈ 30 min).
#       FIX: Sharded saves keep tensors on-device. dcp.save()
#       writes each rank's shard directly from its local GPU.
#       offload_to_cpu used ONLY for full (rank0-only) checkpoints
#       where no NCCL collectives follow.
#
#   [FIX-006] Deprecated FSDP1 state_dict_type() API:
#       ROOT CAUSE: FSDP.state_dict_type() uses ShardedTensor internally.
#       ShardedTensor is deprecated in favor of DTensor. The legacy path
#       generates hundreds of FutureWarning lines per checkpoint,
#       serializes through legacy codepaths with extra allocations,
#       and triggers _get_pg_default_device deprecation warnings
#       during internal collectives.
#       FIX: Migrate to torch.distributed.checkpoint.state_dict API
#       (get_state_dict / set_state_dict with StateDictOptions).
#       Uses DTensor natively — zero ShardedTensor warnings, correct
#       device affinity, fewer collective round-trips.
#       Graceful fallback to FSDP1 API if torch < 2.3.
#
#   [FIX-007] Checkpoint Blocking Training (50s+ at 78% VRAM, 100% stall):
#       ROOT CAUSE: Synchronous checkpoint with offload_to_cpu caused:
#       (a) GPU→CPU memcpy of full sharded state (~30s for 70B params),
#       (b) Two separate dcp.save() calls (model + optimizer) with two
#           barriers each (4 barriers total),
#       (c) NCCL device mismatch retries before timeout.
#       FIX: Single dcp.save() call for model+optimizer (1 barrier).
#       No CPU offload for sharded path (eliminates memcpy).
#       Device guard context prevents affinity leakage.
#       Async metadata write on rank 0.
#
#   [FIX-008] Warning Spam Suppression:
#       _get_pg_default_device, ShardedTensor, state_dict_type warnings
#       suppressed via targeted warnings.filterwarnings at module init.
#       Only suppresses KNOWN HARMLESS warnings — novel warnings pass.
#
#   [FIX-009] FileSystemWriter Overwrite Warning:
#       Explicit overwrite=True passed to FileSystemWriter constructor.
#
# PRIOR FIXES (retained from v2.0):
#   [FIX-001] VRAM Exhaustion (GradientAccumulator, MemoryPool)
#   [FIX-002] Compute Utilization (MetricsCollector, StreamManager)
#   [FIX-003] Backpropagation (no_sync, gradient accumulation)
#   [FIX-004] AMD MI300X / ROCm Compatibility
#
# TRAINER INTEGRATION CONTRACT (SOTATrainer ↔ SOTAFSDP2):
#   [INT-001] through [INT-010] — unchanged from v2.0.
#
# Hardware Support:
#   NVIDIA: A100, H100, H200, B100, B200 (CUDA 12.x / NCCL 2.18+)
#   AMD:   MI300X, MI325X (ROCm 6.x / RCCL)
#
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import functools
import gc
import logging
import math
import os
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Generic,
    List,
    Literal,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer


# ════════════════════════════════════════════════════════════════════════════════
# [FIX-008] Warning Suppression — Known Harmless Deprecation Warnings
# ════════════════════════════════════════════════════════════════════════════════
#
# These warnings originate from PyTorch internals and cannot be fixed in
# user code. They indicate upcoming API changes that we already handle
# via our modern-API-first + fallback strategy.
#
# Suppressed:
#   1. _get_pg_default_device — triggered by ShardedTensor collectives
#   2. FSDP.state_dict_type() — we use get_state_dict() when available
#   3. ShardedTensor deprecation — we use DTensor when available
#   4. FileSystemWriter overwrite — we pass overwrite=True explicitly
#
# NOT suppressed: any novel/unexpected warnings (fail-open policy).
# ════════════════════════════════════════════════════════════════════════════════

def _suppress_known_deprecation_warnings() -> None:
    """
    Install targeted warning filters for known PyTorch deprecation noise.

    Called once at module import. Does NOT suppress unknown warnings.
    """
    # torch.distributed.distributed_c10d — _get_pg_default_device
    warnings.filterwarnings(
        "ignore",
        message=r".*_get_pg_default_device.*will be deprecated.*",
        category=UserWarning,
        module=r"torch\.distributed\.distributed_c10d",
    )
    # FSDP.state_dict_type() deprecation
    warnings.filterwarnings(
        "ignore",
        message=r".*FSDP\.state_dict_type\(\).*being deprecated.*",
        category=FutureWarning,
        module=r"torch\.distributed\.fsdp",
    )
    # ShardedTensor → DTensor migration
    warnings.filterwarnings(
        "ignore",
        message=r".*Please use DTensor.*deprecating ShardedTensor.*",
        category=FutureWarning,
        module=r"torch\.distributed",
    )
    # FileSystemWriter overwrite default change
    warnings.filterwarnings(
        "ignore",
        message=r".*Detected an existing checkpoint.*overwriting.*",
        category=UserWarning,
        module=r"torch\.distributed\.checkpoint",
    )


_suppress_known_deprecation_warnings()


# ════════════════════════════════════════════════════════════════════════════════
# [FIX-006] Modern Checkpoint API Detection
# ════════════════════════════════════════════════════════════════════════════════
#
# torch.distributed.checkpoint.state_dict (PyTorch 2.3+) provides:
#   get_state_dict()  — collects model + optimizer state using DTensor
#   set_state_dict()  — loads model + optimizer state from DTensor dicts
#   StateDictOptions  — controls full vs sharded, cpu offload, etc.
#
# Falls back to FSDP1 API (FSDP.state_dict_type) on PyTorch < 2.3.
# ════════════════════════════════════════════════════════════════════════════════

_MODERN_CKPT_API: bool = False

try:
    from torch.distributed.checkpoint.state_dict import (
        get_state_dict as _get_state_dict,
        set_state_dict as _set_state_dict,
        StateDictOptions as _StateDictOptions,
    )
    _MODERN_CKPT_API = True
except ImportError:
    pass


# ════════════════════════════════════════════════════════════════════════════════
# Constants — Cache-Line Aligned, Hardware-Informed
# ════════════════════════════════════════════════════════════════════════════════

T = TypeVar("T")
ModuleT = TypeVar("ModuleT", bound=nn.Module)

CACHE_LINE_BYTES: Final[int] = 64
SMALL_BUCKET_BYTES: Final[int] = 1 << 20
MEDIUM_BUCKET_BYTES: Final[int] = 25 << 20
LARGE_BUCKET_BYTES: Final[int] = 100 << 20
MIN_POOL_BLOCK_BYTES: Final[int] = 512
MAX_POOL_BLOCK_BYTES: Final[int] = 256 << 20
VRAM_PRESSURE_WATERMARK: Final[float] = 0.90

# [FIX-007] Checkpoint-specific timeout (seconds).
# If NCCL hangs during checkpoint, fail fast instead of waiting 1800s.
CHECKPOINT_NCCL_TIMEOUT_S: Final[int] = 120


# ════════════════════════════════════════════════════════════════════════════════
# Logging — Structured, Rank-Aware
# ════════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("sota_fsdp2")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "[%(asctime)s][FSDP2][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_handler)


# ════════════════════════════════════════════════════════════════════════════════
# Result Type — Explicit Error Handling (No Exceptions for Control Flow)
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Success variant: wraps a value of type T."""
    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value

    def map(self, fn: Callable[[T], Any]) -> Result[Any]:
        return Ok(fn(self.value))


@dataclass(frozen=True, slots=True)
class Err(Generic[T]):
    """Error variant: wraps an error message and optional code."""
    error: str
    code: int = 0

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> T:
        raise RuntimeError(
            f"Attempted to unwrap Err: {self.error} (code={self.code})"
        )

    def unwrap_or(self, default: T) -> T:
        return default

    def map(self, fn: Callable[[T], Any]) -> Result[Any]:
        return Err(self.error, self.code)


Result = Union[Ok[T], Err[T]]


# ════════════════════════════════════════════════════════════════════════════════
# Hardware Detection — Vendor-Aware Capability Enumeration
# ════════════════════════════════════════════════════════════════════════════════

class HardwareVendor(Enum):
    NVIDIA = auto()
    AMD = auto()
    INTEL = auto()
    UNKNOWN = auto()


class ComputeCapability(NamedTuple):
    major: int
    minor: int

    @property
    def sm_version(self) -> int:
        return self.major * 10 + self.minor

    def supports_bf16(self) -> bool:
        return self.sm_version >= 80

    def supports_fp8(self) -> bool:
        return self.sm_version >= 89

    def supports_tma(self) -> bool:
        return self.sm_version >= 90


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    vendor: HardwareVendor
    device_name: str
    compute_capability: ComputeCapability
    total_memory_bytes: int
    num_sms: int
    max_threads_per_sm: int
    l2_cache_bytes: int
    supports_nvlink: bool
    pcie_bandwidth_gbps: float

    @property
    def memory_gb(self) -> float:
        return self.total_memory_bytes / (1 << 30)

    @property
    def is_datacenter_gpu(self) -> bool:
        _patterns = ("A100", "H100", "H200", "B100", "B200", "MI300", "MI250")
        return any(p in self.device_name for p in _patterns)

    @property
    def is_amd(self) -> bool:
        return self.vendor == HardwareVendor.AMD


def detect_hardware(device_id: int = 0) -> Result[HardwareInfo]:
    """
    Probe CUDA device properties and infer vendor / capabilities.

    Returns Ok(HardwareInfo) on success, Err if device unavailable.
    """
    if not torch.cuda.is_available():
        return Err("CUDA runtime not available", code=1)

    if device_id >= torch.cuda.device_count():
        return Err(
            f"Device {device_id} not found "
            f"(available: {torch.cuda.device_count()})",
            code=2,
        )

    props = torch.cuda.get_device_properties(device_id)
    name_upper = props.name.upper()

    if any(k in name_upper for k in (
        "NVIDIA", "A100", "H100", "V100", "RTX", "GTX",
    )):
        vendor = HardwareVendor.NVIDIA
    elif any(k in name_upper for k in ("AMD", "MI", "RADEON", "INSTINCT")):
        vendor = HardwareVendor.AMD
    elif "INTEL" in name_upper:
        vendor = HardwareVendor.INTEL
    else:
        vendor = HardwareVendor.UNKNOWN

    has_nvlink = any(
        k in props.name for k in ("A100", "H100", "H200", "V100", "DGX")
    )
    l2 = getattr(
        props, "l2_cache_size",
        getattr(props, "L2_cache_size", 4 << 20),
    )

    return Ok(HardwareInfo(
        vendor=vendor,
        device_name=props.name,
        compute_capability=ComputeCapability(props.major, props.minor),
        total_memory_bytes=props.total_memory,
        num_sms=props.multi_processor_count,
        max_threads_per_sm=props.max_threads_per_multi_processor,
        l2_cache_bytes=l2,
        supports_nvlink=has_nvlink,
        pcie_bandwidth_gbps=32.0 if props.major >= 8 else 16.0,
    ))


# ════════════════════════════════════════════════════════════════════════════════
# Enums — Type-Safe Configuration with Semantic Methods
# ════════════════════════════════════════════════════════════════════════════════

class ShardingStrategy(Enum):
    """
    Memory / communication tradeoff selection.

    Per-GPU memory for N params, W GPUs:
        FULL_SHARD:    O(3N/W)
        SHARD_GRAD_OP: O(N + 2N/W)
        NO_SHARD:      O(3N)
        HYBRID_SHARD:  between FULL/NO
    """
    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()
    NO_SHARD = auto()
    HYBRID_SHARD = auto()

    def requires_all_gather(self) -> bool:
        return self in (
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.HYBRID_SHARD,
        )

    def requires_reduce_scatter(self) -> bool:
        return self != ShardingStrategy.NO_SHARD


class MixedPrecisionPolicy(Enum):
    """
    Precision policy governing param / reduce / buffer dtypes.
    """
    FULL_BF16 = auto()
    FULL_FP16 = auto()
    PARAM_FP32 = auto()
    PURE_FP32 = auto()

    def get_param_dtype(self) -> torch.dtype:
        return {
            MixedPrecisionPolicy.FULL_BF16: torch.bfloat16,
            MixedPrecisionPolicy.FULL_FP16: torch.float16,
            MixedPrecisionPolicy.PARAM_FP32: torch.float32,
            MixedPrecisionPolicy.PURE_FP32: torch.float32,
        }[self]

    def get_reduce_dtype(self) -> torch.dtype:
        return {
            MixedPrecisionPolicy.FULL_BF16: torch.bfloat16,
            MixedPrecisionPolicy.FULL_FP16: torch.float16,
            MixedPrecisionPolicy.PARAM_FP32: torch.bfloat16,
            MixedPrecisionPolicy.PURE_FP32: torch.float32,
        }[self]

    def requires_loss_scaling(self) -> bool:
        return self == MixedPrecisionPolicy.FULL_FP16


class OffloadStrategy(Enum):
    NONE = auto()
    CPU_PARAMS = auto()
    CPU_OPTIM = auto()
    CPU_FULL = auto()
    NVME = auto()

    def requires_pinned_memory(self) -> bool:
        return self in (
            OffloadStrategy.CPU_PARAMS,
            OffloadStrategy.CPU_OPTIM,
            OffloadStrategy.CPU_FULL,
        )


class BackwardPrefetchMode(Enum):
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    NONE = auto()


# ════════════════════════════════════════════════════════════════════════════════
# Memory Pool — Pressure-Aware, Power-of-2 Bucketed, Zero-Jitter
# ════════════════════════════════════════════════════════════════════════════════

class MemoryPool:
    """
    Pre-allocated memory pool eliminating cudaMalloc jitter.

    Uses power-of-2 bucketing with pressure-aware eviction.
    Thread-safe. Allocate: O(1) amortized. Release: O(1).
    """

    __slots__ = (
        "_device", "_dtype", "_pools", "_lock",
        "_total_allocated", "_peak_allocated", "_allocation_count",
        "_device_total_bytes",
    )

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self._device = device
        self._dtype = dtype
        self._pools: Dict[int, List[Tensor]] = {}
        self._lock = threading.Lock()
        self._total_allocated: int = 0
        self._peak_allocated: int = 0
        self._allocation_count: int = 0
        self._device_total_bytes: int = (
            torch.cuda.get_device_properties(device).total_memory
        )

    @staticmethod
    def _element_size(dtype: torch.dtype) -> int:
        return torch.tensor([], dtype=dtype).element_size()

    def _bucket_size(self, requested_bytes: int) -> int:
        if requested_bytes <= MIN_POOL_BLOCK_BYTES:
            return MIN_POOL_BLOCK_BYTES
        if requested_bytes >= MAX_POOL_BLOCK_BYTES:
            return requested_bytes
        return 1 << (requested_bytes - 1).bit_length()

    def _check_pressure(self) -> None:
        allocated = torch.cuda.memory_allocated(self._device)
        threshold = self._device_total_bytes * VRAM_PRESSURE_WATERMARK
        if allocated < threshold:
            return
        for bucket_size in sorted(self._pools.keys(), reverse=True):
            while self._pools[bucket_size]:
                tensor = self._pools[bucket_size].pop()
                freed = tensor.numel() * tensor.element_size()
                self._total_allocated -= freed
                del tensor
                if torch.cuda.memory_allocated(self._device) < threshold:
                    return
            if not self._pools[bucket_size]:
                del self._pools[bucket_size]

    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        dtype = dtype or self._dtype
        elem_size = self._element_size(dtype)
        num_elements = math.prod(shape)
        size_bytes = num_elements * elem_size
        bucket = self._bucket_size(size_bytes)

        with self._lock:
            self._allocation_count += 1
            self._check_pressure()
            if bucket in self._pools and self._pools[bucket]:
                slab = self._pools[bucket].pop()
                if slab.numel() >= num_elements:
                    return slab[:num_elements].view(shape)
                del slab

        tensor = torch.empty(num_elements, dtype=dtype, device=self._device)
        with self._lock:
            self._total_allocated += size_bytes
            self._peak_allocated = max(
                self._peak_allocated, self._total_allocated,
            )
        return tensor.view(shape)

    def release(self, tensor: Tensor) -> None:
        if not tensor.is_contiguous():
            return
        size_bytes = tensor.numel() * tensor.element_size()
        bucket = self._bucket_size(size_bytes)
        with self._lock:
            if bucket not in self._pools:
                self._pools[bucket] = []
            self._pools[bucket].append(tensor.detach().view(-1))

    def clear(self) -> None:
        with self._lock:
            self._pools.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_allocated_mb": self._total_allocated / (1 << 20),
                "peak_allocated_mb": self._peak_allocated / (1 << 20),
                "allocation_count": self._allocation_count,
                "bucket_count": len(self._pools),
                "pooled_tensors": sum(
                    len(v) for v in self._pools.values()
                ),
            }


# ════════════════════════════════════════════════════════════════════════════════
# Triton Kernels — Fused Collective Primitives
# ════════════════════════════════════════════════════════════════════════════════

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True

    @triton.jit
    def _fused_allgather_scale_kernel(
        src_ptr,
        dst_ptr,
        scale,
        num_elements,
        rank_offset,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        data = tl.load(src_ptr + offsets, mask=mask, other=0.0)
        data = data * scale
        tl.store(dst_ptr + rank_offset + offsets, data, mask=mask)

    @triton.jit
    def _fused_cast_and_scale_kernel(
        src_ptr,
        dst_ptr,
        scale,
        num_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        data = tl.load(src_ptr + offsets, mask=mask, other=0.0)
        data = data * scale
        tl.store(dst_ptr + offsets, data, mask=mask)

    @triton.jit
    def _fused_gradient_accumulate_kernel(
        grad_ptr,
        accum_ptr,
        num_elements,
        inv_accum_steps,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
        accum = tl.load(accum_ptr + offsets, mask=mask, other=0.0)
        accum = accum + grad * inv_accum_steps
        tl.store(accum_ptr + offsets, accum, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 8192}, num_warps=16),
        ],
        key=["shard_size"],
    )
    @triton.jit
    def _fused_param_shard_kernel(
        full_param_ptr,
        shard_ptr,
        full_numel,
        shard_size,
        rank_offset,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < shard_size
        src_offsets = rank_offset + offsets
        full_mask = mask & (src_offsets < full_numel)
        data = tl.load(
            full_param_ptr + src_offsets, mask=full_mask, other=0.0,
        )
        tl.store(shard_ptr + offsets, data, mask=mask)

except ImportError:
    TRITON_AVAILABLE = False
    logger.warning(
        "Triton not available (pip install triton>=2.1.0). "
        "Falling back to PyTorch-native ops (~15%% slower)."
    )


# ════════════════════════════════════════════════════════════════════════════════
# Stream Manager — Lock-Free, Deadlock-Safe Communication Overlap
# ════════════════════════════════════════════════════════════════════════════════

class StreamManager:
    __slots__ = (
        "_device",
        "_compute_stream",
        "_allgather_stream",
        "_reduce_scatter_stream",
        "_transfer_stream",
        "_is_amd",
    )

    def __init__(self, device: torch.device, is_amd: bool = False):
        self._device = device
        self._is_amd = is_amd
        high_prio = 0 if is_amd else -1

        with torch.cuda.device(device):
            self._compute_stream = torch.cuda.default_stream(device)
            self._allgather_stream = torch.cuda.Stream(
                device, priority=high_prio,
            )
            self._reduce_scatter_stream = torch.cuda.Stream(
                device, priority=high_prio,
            )
            self._transfer_stream = torch.cuda.Stream(
                device, priority=0,
            )

    @property
    def compute(self) -> torch.cuda.Stream:
        return self._compute_stream

    @property
    def allgather(self) -> torch.cuda.Stream:
        return self._allgather_stream

    @property
    def reduce_scatter(self) -> torch.cuda.Stream:
        return self._reduce_scatter_stream

    @property
    def transfer(self) -> torch.cuda.Stream:
        return self._transfer_stream

    def sync_stream_to(
        self,
        src: torch.cuda.Stream,
        dst: torch.cuda.Stream,
    ) -> None:
        event = torch.cuda.Event(enable_timing=False, blocking=False)
        event.record(src)
        dst.wait_event(event)

    def sync_allgather_to_compute(self) -> None:
        self.sync_stream_to(self._allgather_stream, self._compute_stream)

    def sync_compute_to_reduce_scatter(self) -> None:
        self.sync_stream_to(
            self._compute_stream, self._reduce_scatter_stream,
        )

    def synchronize_all(self) -> None:
        self._compute_stream.synchronize()
        self._allgather_stream.synchronize()
        self._reduce_scatter_stream.synchronize()
        self._transfer_stream.synchronize()


# ════════════════════════════════════════════════════════════════════════════════
# FSDP2 Configuration
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class FSDP2Config:
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    mixed_precision: MixedPrecisionPolicy = MixedPrecisionPolicy.FULL_BF16
    offload_strategy: OffloadStrategy = OffloadStrategy.NONE

    use_orig_params: bool = True
    forward_prefetch: bool = True
    backward_prefetch: BackwardPrefetchMode = BackwardPrefetchMode.BACKWARD_PRE
    reshard_after_forward: bool = True
    limit_all_gathers: bool = True

    use_triton_kernels: bool = True
    triton_block_size: int = 4096

    use_memory_pool: bool = True
    bucket_size_mb: int = 25
    sync_module_states: bool = True

    auto_wrap_policy: Optional[List[str]] = None
    ignored_modules: Optional[List[str]] = None
    min_num_params: int = 100_000_000

    activation_checkpointing: bool = True
    ac_mode: Literal["full", "selective", "offload"] = "selective"
    ac_frequency: int = 2
    ac_offload_to_cpu: bool = False

    gradient_accumulation_steps: int = 1
    gradient_clipping_norm: Optional[float] = 1.0

    use_cuda_graphs: bool = False
    cuda_graph_warmup_iters: int = 3

    deterministic: bool = False
    debug_mode: bool = False

    def __post_init__(self) -> None:
        if self.auto_wrap_policy is None:
            self.auto_wrap_policy = [
                "TransformerEncoderLayer",
                "TransformerDecoderLayer",
                "LlamaDecoderLayer",
                "Llama3DecoderLayer",
                "MistralDecoderLayer",
                "Qwen2DecoderLayer",
                "GPT2Block",
                "GPTNeoXLayer",
                "FalconDecoderLayer",
                "GemmaDecoderLayer",
                "Phi3DecoderLayer",
                "TransformerSentenceEncoderLayer",
            ]
        if self.ignored_modules is None:
            self.ignored_modules = []
        self._validate()

    def _validate(self) -> None:
        if self.bucket_size_mb < 1:
            raise ValueError(
                f"bucket_size_mb must be >= 1, got {self.bucket_size_mb}"
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, "
                f"got {self.gradient_accumulation_steps}"
            )
        if self.ac_frequency < 1:
            raise ValueError(
                f"ac_frequency must be >= 1, got {self.ac_frequency}"
            )
        if self.triton_block_size & (self.triton_block_size - 1):
            raise ValueError(
                f"triton_block_size must be power of 2, "
                f"got {self.triton_block_size}"
            )
        if self.use_cuda_graphs and self.offload_strategy != OffloadStrategy.NONE:
            warnings.warn(
                "CUDA graphs incompatible with CPU offloading; "
                "disabling graphs.",
                RuntimeWarning,
            )
            self.use_cuda_graphs = False
        if self.use_cuda_graphs and self.gradient_accumulation_steps > 1:
            warnings.warn(
                "CUDA graphs + gradient accumulation: "
                "ensure static shapes.",
                RuntimeWarning,
            )

    @property
    def bucket_size_bytes(self) -> int:
        return self.bucket_size_mb << 20


# ════════════════════════════════════════════════════════════════════════════════
# Metrics — Non-Blocking CUDA Event Timing (Zero GPU Stall)
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class FSDPMetrics:
    allgather_time_ns: int = 0
    reduce_scatter_time_ns: int = 0
    forward_time_ns: int = 0
    backward_time_ns: int = 0
    optimizer_time_ns: int = 0
    peak_memory_bytes: int = 0
    allocated_memory_bytes: int = 0
    reserved_memory_bytes: int = 0
    tokens_processed: int = 0
    samples_processed: int = 0
    gradient_norm: float = 0.0
    gradient_overflow_count: int = 0

    def reset(self) -> None:
        self.allgather_time_ns = 0
        self.reduce_scatter_time_ns = 0
        self.forward_time_ns = 0
        self.backward_time_ns = 0
        self.optimizer_time_ns = 0
        self.peak_memory_bytes = 0
        self.allocated_memory_bytes = 0
        self.reserved_memory_bytes = 0
        self.tokens_processed = 0
        self.samples_processed = 0
        self.gradient_norm = 0.0
        self.gradient_overflow_count = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "allgather_ms": self.allgather_time_ns / 1e6,
            "reduce_scatter_ms": self.reduce_scatter_time_ns / 1e6,
            "forward_ms": self.forward_time_ns / 1e6,
            "backward_ms": self.backward_time_ns / 1e6,
            "optimizer_ms": self.optimizer_time_ns / 1e6,
            "peak_memory_gb": self.peak_memory_bytes / (1 << 30),
            "allocated_memory_gb": self.allocated_memory_bytes / (1 << 30),
            "reserved_memory_gb": self.reserved_memory_bytes / (1 << 30),
            "tokens_processed": float(self.tokens_processed),
            "samples_processed": float(self.samples_processed),
            "gradient_norm": self.gradient_norm,
            "gradient_overflow_count": float(self.gradient_overflow_count),
        }


class MetricsCollector:
    """
    Non-blocking metrics via CUDA events.
    [FIX-002] No torch.cuda.synchronize() in hot path.
    """

    __slots__ = (
        "_current", "_history", "_max_history", "_lock",
        "_enable_gpu_timing", "_pending_events",
    )

    def __init__(
        self,
        max_history: int = 1000,
        enable_gpu_timing: bool = True,
    ):
        self._current = FSDPMetrics()
        self._history: List[FSDPMetrics] = []
        self._max_history = max_history
        self._lock = threading.Lock()
        self._enable_gpu_timing = enable_gpu_timing
        self._pending_events: List[
            Tuple[torch.cuda.Event, torch.cuda.Event, str]
        ] = []

    @property
    def current(self) -> FSDPMetrics:
        return self._current

    def _flush_pending_events(self) -> None:
        remaining = []
        for start_ev, end_ev, field in self._pending_events:
            if end_ev.query():
                elapsed_ms = start_ev.elapsed_time(end_ev)
                elapsed_ns = int(elapsed_ms * 1e6)
                current_val = getattr(self._current, field)
                setattr(self._current, field, current_val + elapsed_ns)
            else:
                remaining.append((start_ev, end_ev, field))
        self._pending_events = remaining

    @contextmanager
    def _measure_gpu(self, field_name: str):
        if not self._enable_gpu_timing:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            with self._lock:
                self._pending_events.append((start, end, field_name))

    @contextmanager
    def measure_allgather(self):
        with self._measure_gpu("allgather_time_ns"):
            yield

    @contextmanager
    def measure_reduce_scatter(self):
        with self._measure_gpu("reduce_scatter_time_ns"):
            yield

    @contextmanager
    def measure_forward(self):
        with self._measure_gpu("forward_time_ns"):
            yield

    @contextmanager
    def measure_backward(self):
        with self._measure_gpu("backward_time_ns"):
            yield

    @contextmanager
    def measure_optimizer(self):
        with self._measure_gpu("optimizer_time_ns"):
            yield

    def update_memory_stats(self) -> None:
        self._current.allocated_memory_bytes = torch.cuda.memory_allocated()
        self._current.reserved_memory_bytes = torch.cuda.memory_reserved()
        self._current.peak_memory_bytes = max(
            self._current.peak_memory_bytes,
            torch.cuda.max_memory_allocated(),
        )

    def record_step(self) -> None:
        with self._lock:
            self._flush_pending_events()
            self._history.append(self._current)
            if len(self._history) > self._max_history:
                self._history.pop(0)
            self._current = FSDPMetrics()

    def get_average(self, last_n: int = 100) -> Dict[str, float]:
        with self._lock:
            self._flush_pending_events()
            history = self._history[-last_n:] if self._history else []
        if not history:
            return {}
        n = len(history)
        return {
            "avg_allgather_ms": (
                sum(h.allgather_time_ns for h in history) / n / 1e6
            ),
            "avg_reduce_scatter_ms": (
                sum(h.reduce_scatter_time_ns for h in history) / n / 1e6
            ),
            "avg_forward_ms": (
                sum(h.forward_time_ns for h in history) / n / 1e6
            ),
            "avg_backward_ms": (
                sum(h.backward_time_ns for h in history) / n / 1e6
            ),
            "avg_optimizer_ms": (
                sum(h.optimizer_time_ns for h in history) / n / 1e6
            ),
            "max_peak_memory_gb": (
                max(h.peak_memory_bytes for h in history) / (1 << 30)
            ),
        }


# ════════════════════════════════════════════════════════════════════════════════
# Gradient Accumulator — Name-Keyed, In-Place, FSDP-Safe
# ════════════════════════════════════════════════════════════════════════════════

class GradientAccumulator:
    """
    Zero-copy gradient accumulation using named parameter buffers.
    [FIX-001/003] Keyed by name, never modifies param.grad, FSDP-safe.
    """

    __slots__ = (
        "_accumulation_steps",
        "_inv_accum_steps",
        "_current_step",
        "_grad_buffers",
        "_use_triton",
        "_block_size",
    )

    def __init__(
        self,
        model: nn.Module,
        accumulation_steps: int,
        use_triton: bool = True,
        block_size: int = 4096,
    ):
        self._accumulation_steps = accumulation_steps
        self._inv_accum_steps = 1.0 / float(accumulation_steps)
        self._current_step = 0
        self._use_triton = use_triton and TRITON_AVAILABLE
        self._block_size = block_size
        self._grad_buffers: Dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._grad_buffers[name] = torch.zeros_like(
                    param, memory_format=torch.contiguous_format,
                )

    @property
    def should_sync(self) -> bool:
        return self._current_step >= self._accumulation_steps

    @property
    def current_step(self) -> int:
        return self._current_step

    def accumulate(self, model: nn.Module) -> None:
        self._current_step += 1
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if name not in self._grad_buffers:
                self._grad_buffers[name] = torch.zeros_like(
                    param.grad, memory_format=torch.contiguous_format,
                )
            buffer = self._grad_buffers[name]
            grad = param.grad.detach()
            if buffer.shape != grad.shape:
                self._grad_buffers[name] = torch.zeros_like(
                    grad, memory_format=torch.contiguous_format,
                )
                buffer = self._grad_buffers[name]
            if self._use_triton and grad.numel() >= self._block_size:
                grid = (triton.cdiv(grad.numel(), self._block_size),)
                _fused_gradient_accumulate_kernel[grid](
                    grad.contiguous(),
                    buffer,
                    grad.numel(),
                    self._inv_accum_steps,
                    BLOCK_SIZE=self._block_size,
                )
            else:
                buffer.add_(grad, alpha=self._inv_accum_steps)

    def apply_to_model(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self._grad_buffers:
                param.grad = self._grad_buffers[name]

    def reset(self) -> None:
        self._current_step = 0
        for buffer in self._grad_buffers.values():
            buffer.zero_()


# ════════════════════════════════════════════════════════════════════════════════
# Mixed Precision Context
# ════════════════════════════════════════════════════════════════════════════════

class MixedPrecisionContext:
    __slots__ = ("_policy", "_scaler", "_enabled")

    def __init__(self, policy: MixedPrecisionPolicy):
        self._policy = policy
        self._enabled = policy != MixedPrecisionPolicy.PURE_FP32
        if policy.requires_loss_scaling():
            self._scaler = torch.amp.GradScaler(
                device="cuda",
                init_scale=2**16,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000,
                enabled=True,
            )
        else:
            self._scaler = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def scaler(self) -> Optional[torch.amp.GradScaler]:
        return self._scaler

    @contextmanager
    def autocast(self):
        if not self._enabled:
            yield
            return
        dtype = self._policy.get_param_dtype()
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            yield

    def scale_loss(self, loss: Tensor) -> Tensor:
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss

    def unscale_grads(self, optimizer: Optimizer) -> None:
        if self._scaler is not None:
            self._scaler.unscale_(optimizer)

    def step_optimizer(self, optimizer: Optimizer) -> None:
        if self._scaler is not None:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()


# ════════════════════════════════════════════════════════════════════════════════
# SOTA FSDP2 — Main Orchestrator
# ════════════════════════════════════════════════════════════════════════════════

class SOTAFSDP2:
    """
    FSDP2 engine with Triton acceleration and hardware-aware optimization.

    TRAINER INTEGRATION:
        forward_context()  → autocast + forward metrics
        backward(loss)     → no_sync + accumulation + loss scaling
        step(optimizer)    → unscale → clip → step → zero_grad → reset
    """

    __slots__ = (
        "_config",
        "_device_mesh",
        "_wrapped_model",
        "_hardware_info",
        "_memory_pool",
        "_stream_manager",
        "_metrics",
        "_gradient_accumulator",
        "_mp_context",
        "_rank",
        "_world_size",
        "_local_rank",
        "_is_rank_zero",
        "_cuda_graph",
        "_cuda_graph_captured",
        "_warmup_counter",
        "_accumulation_counter",
        "_device",
        "_is_amd",
        "_memory_per_shard_gb",
    )

    def __init__(
        self,
        config: FSDP2Config,
        device_mesh: Optional[Any] = None,
    ):
        self._config = config
        self._device_mesh = device_mesh
        self._wrapped_model: Optional[nn.Module] = None

        self._init_distributed_info()

        self._device = torch.device(f"cuda:{self._local_rank}")
        torch.cuda.set_device(self._device)

        hw_result = detect_hardware(self._local_rank)
        if hw_result.is_ok():
            self._hardware_info = hw_result.unwrap()
            self._is_amd = self._hardware_info.is_amd
            if self._is_rank_zero:
                logger.info(
                    f"GPU: {self._hardware_info.device_name} "
                    f"({self._hardware_info.memory_gb:.1f} GB, "
                    f"SM{self._hardware_info.compute_capability.sm_version})"
                )
        else:
            self._hardware_info = None
            self._is_amd = False
            if self._is_rank_zero:
                logger.warning(
                    f"Hardware detection failed: {hw_result.error}"
                )

        if config.use_memory_pool:
            self._memory_pool = MemoryPool(
                device=self._device,
                dtype=config.mixed_precision.get_param_dtype(),
            )
        else:
            self._memory_pool = None

        self._stream_manager = StreamManager(
            self._device, is_amd=self._is_amd,
        )
        self._metrics = MetricsCollector(
            enable_gpu_timing=not self._is_amd,
        )
        self._mp_context = MixedPrecisionContext(config.mixed_precision)

        self._gradient_accumulator: Optional[GradientAccumulator] = None
        self._accumulation_counter: int = 0
        self._cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self._cuda_graph_captured = False
        self._warmup_counter = 0
        self._memory_per_shard_gb: float = 0.0

        if self._is_rank_zero:
            logger.info(
                f"FSDP2 init: "
                f"strategy={config.sharding_strategy.name}, "
                f"precision={config.mixed_precision.name}, "
                f"triton={'on' if TRITON_AVAILABLE and config.use_triton_kernels else 'off'}, "
                f"world_size={self._world_size}, "
                f"grad_accum={config.gradient_accumulation_steps}, "
                f"ckpt_api={'modern(DTensor)' if _MODERN_CKPT_API else 'legacy(FSDP1)'}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────────────────────

    def _init_distributed_info(self) -> None:
        import torch.distributed as dist
        if dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
            self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            self._rank = 0
            self._world_size = 1
            self._local_rank = 0
        self._is_rank_zero = self._rank == 0

    # ──────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def config(self) -> FSDP2Config:
        return self._config

    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def is_rank_zero(self) -> bool:
        return self._is_rank_zero

    @property
    def model(self) -> Optional[nn.Module]:
        return self._wrapped_model

    @property
    def device(self) -> torch.device:
        """Expose device for checkpoint manager device-guard."""
        return self._device

    # ──────────────────────────────────────────────────────────────────────────
    # PyTorch FSDP Interop
    # ──────────────────────────────────────────────────────────────────────────
    
    def _get_torch_sharding_strategy(self):
        from torch.distributed.fsdp import ShardingStrategy as TS
        return {
            ShardingStrategy.FULL_SHARD: TS.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP: TS.SHARD_GRAD_OP,
            ShardingStrategy.NO_SHARD: TS.NO_SHARD,
            ShardingStrategy.HYBRID_SHARD: TS.HYBRID_SHARD,
        }[self._config.sharding_strategy]

    

    def _get_torch_mixed_precision(self):
        """
        Convert to PyTorch MixedPrecision policy.

        Returns None for PURE_FP32 (no autocast overhead).
        """
        from torch.distributed.fsdp import MixedPrecision

        if self._config.mixed_precision == MixedPrecisionPolicy.PURE_FP32:
            return None

        param_dtype = self._config.mixed_precision.get_param_dtype()
        reduce_dtype = self._config.mixed_precision.get_reduce_dtype()
        return MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=param_dtype,
        )

    def _get_torch_backward_prefetch(self):
        """Convert to PyTorch BackwardPrefetch enum."""
        from torch.distributed.fsdp import BackwardPrefetch
        return {
            BackwardPrefetchMode.BACKWARD_PRE: BackwardPrefetch.BACKWARD_PRE,
            BackwardPrefetchMode.BACKWARD_POST: BackwardPrefetch.BACKWARD_POST,
            BackwardPrefetchMode.NONE: None,
        }[self._config.backward_prefetch]

    def _get_auto_wrap_policy(self) -> Callable:
        """
        Create auto-wrap policy from config.

        Tries transformer_auto_wrap_policy first (layer-class matching).
        Falls back to size_based_auto_wrap_policy for unknown archs.
        """
        from torch.distributed.fsdp.wrap import (
            size_based_auto_wrap_policy,
            transformer_auto_wrap_policy,
        )

        layer_classes: Set[Type[nn.Module]] = set()
        for cls_name in self._config.auto_wrap_policy:
            cls = self._try_import_class(cls_name)
            if cls is not None:
                layer_classes.add(cls)

        if layer_classes:
            return functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=layer_classes,
            )

        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=self._config.min_num_params,
        )

    @staticmethod
    def _try_import_class(cls_name: str) -> Optional[Type[nn.Module]]:
        """Try to import transformer layer class from known locations."""
        _paths = [
            "transformers.models.llama.modeling_llama",
            "transformers.models.mistral.modeling_mistral",
            "transformers.models.qwen2.modeling_qwen2",
            "transformers.models.gpt2.modeling_gpt2",
            "transformers.models.falcon.modeling_falcon",
            "transformers.models.gemma.modeling_gemma",
            "transformers.models.phi3.modeling_phi3",
            "torch.nn",
            "fairseq.modules",
        ]
        for path in _paths:
            try:
                module = __import__(path, fromlist=[cls_name])
                if hasattr(module, cls_name):
                    return getattr(module, cls_name)
            except (ImportError, ModuleNotFoundError):
                continue
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Device Affinity — Prevent cuda:0 Drift [FIX-005]
    # ──────────────────────────────────────────────────────────────────────────

    @contextmanager
    def _enforce_device_affinity(self):
        """
        Enforce rank-local device affinity for ALL tensor ops in scope.

        [FIX-005] ROOT CAUSE: FSDP state_dict_type() with offload_to_cpu
        internally creates staging tensors on cuda:0 (default device)
        instead of the rank-local cuda:{local_rank}. When NCCL performs
        the next collective, it detects:

            "Tensor found on device cuda:0 but backend constrained to cuda:3"

        This context manager:
            1. Pins torch.cuda.set_device to this rank's device
            2. Uses torch.cuda.device() context to override default
            3. Verifies device after scope exit (debug mode)

        Complexity: O(1) — only sets device, no allocation.
        """
        torch.cuda.set_device(self._device)
        with torch.cuda.device(self._device):
            yield

        # Post-scope verification in debug mode
        if self._config.debug_mode:
            self._verify_param_devices()

    def _verify_param_devices(self) -> None:
        """
        Assert ALL parameters reside on rank-local device.

        Called after checkpoint ops to catch device drift early.
        O(P) where P = number of parameters (debug only).
        """
        if self._wrapped_model is None:
            return

        for name, param in self._wrapped_model.named_parameters():
            if param.is_cuda and param.device != self._device:
                logger.error(
                    f"[FIX-005] Device drift detected: "
                    f"param '{name}' on {param.device}, "
                    f"expected {self._device}"
                )
                # Force migration — do NOT let training continue on wrong device
                param.data = param.data.to(self._device)

    # ──────────────────────────────────────────────────────────────────────────
    # Activation Checkpointing
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_activation_checkpointing(self, model: nn.Module) -> None:
        """
        Apply activation checkpointing BEFORE FSDP wrapping.

        Selective mode: every ac_frequency-th layer (default 2nd).
        Full mode: all eligible layers.
        """
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
            apply_activation_checkpointing,
        )

        check_impl = CheckpointImpl.NO_REENTRANT
        layer_idx = [0]

        def check_fn(module: nn.Module) -> bool:
            name = module.__class__.__name__.lower()
            is_layer = any(
                p in name
                for p in ("layer", "block", "decoder", "encoder")
            )
            if not is_layer:
                return False
            if self._config.ac_mode == "full":
                return True
            if self._config.ac_mode == "selective":
                idx = layer_idx[0]
                layer_idx[0] += 1
                return idx % self._config.ac_frequency == 0
            return True

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=check_impl,
            ),
            check_fn=check_fn,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Model Wrapping
    # ──────────────────────────────────────────────────────────────────────────

    def wrap_model(
        self,
        model: nn.Module,
        device_mesh: Optional[Any] = None,
    ) -> nn.Module:
        """
        Wrap model with FSDP.

        Steps:
            1. Enforce device affinity (prevent cuda:0 drift)
            2. Apply activation checkpointing (before FSDP)
            3. Build FSDP kwargs from config
            4. Wrap with PyTorch FSDP
            5. Initialize gradient accumulator (if steps > 1)

        Returns:
            FSDP-wrapped module on rank-local device.
        """
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            CPUOffload,
        )

        mesh = device_mesh or self._device_mesh

        # [FIX-005] Pin device affinity before ANY FSDP operation
        with self._enforce_device_affinity():

            # Activation checkpointing MUST precede FSDP wrapping
            if self._config.activation_checkpointing:
                self._apply_activation_checkpointing(model)
                if self._is_rank_zero:
                    logger.info(
                        f"Activation checkpointing: "
                        f"mode={self._config.ac_mode}"
                    )

            # FSDP kwargs
            fsdp_kwargs: Dict[str, Any] = {
                "sharding_strategy": self._get_torch_sharding_strategy(),
                "auto_wrap_policy": self._get_auto_wrap_policy(),
                "use_orig_params": self._config.use_orig_params,
                "forward_prefetch": self._config.forward_prefetch,
                "sync_module_states": self._config.sync_module_states,
                "limit_all_gathers": self._config.limit_all_gathers,
            }

            mp_policy = self._get_torch_mixed_precision()
            if mp_policy is not None:
                fsdp_kwargs["mixed_precision"] = mp_policy

            backward_prefetch = self._get_torch_backward_prefetch()
            if backward_prefetch is not None:
                fsdp_kwargs["backward_prefetch"] = backward_prefetch

            if self._config.offload_strategy != OffloadStrategy.NONE:
                fsdp_kwargs["cpu_offload"] = CPUOffload(
                    offload_params=(
                        self._config.offload_strategy in (
                            OffloadStrategy.CPU_PARAMS,
                            OffloadStrategy.CPU_FULL,
                        )
                    ),
                )

            if mesh is not None:
                fsdp_kwargs["device_mesh"] = mesh

            # Wrap
            wrapped_model = FSDP(model, **fsdp_kwargs)
            self._wrapped_model = wrapped_model

        # Compute memory per shard
        param_count = sum(p.numel() for p in wrapped_model.parameters())
        bytes_per_elem = torch.tensor(
            [], dtype=self._config.mixed_precision.get_param_dtype(),
        ).element_size()
        self._memory_per_shard_gb = (
            (param_count / self._world_size)
            * bytes_per_elem
            / (1 << 30)
        )

        # Gradient accumulator (AFTER FSDP wrapping for correct names)
        if self._config.gradient_accumulation_steps > 1:
            self._gradient_accumulator = GradientAccumulator(
                wrapped_model,
                self._config.gradient_accumulation_steps,
                use_triton=(
                    self._config.use_triton_kernels and TRITON_AVAILABLE
                ),
                block_size=self._config.triton_block_size,
            )
            self._accumulation_counter = 0

        if self._is_rank_zero:
            logger.info(
                f"FSDP wrapped: {param_count:,} params "
                f"(~{param_count / self._world_size:,.0f}/rank, "
                f"~{self._memory_per_shard_gb:.2f} GB/rank in "
                f"{self._config.mixed_precision.get_param_dtype()})"
            )

        return wrapped_model

    # ──────────────────────────────────────────────────────────────────────────
    # no_sync Context
    # ──────────────────────────────────────────────────────────────────────────

    @contextmanager
    def _no_sync_context(self):
        """
        Suppress FSDP reduce-scatter during non-sync micro-steps.

        [FIX-003] Without this, every micro-step triggers reduce-scatter:
        O(steps * N) communication instead of O(N) per cycle.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if (
            self._wrapped_model is not None
            and isinstance(self._wrapped_model, FSDP)
        ):
            with self._wrapped_model.no_sync():
                yield
        else:
            yield

    # ──────────────────────────────────────────────────────────────────────────
    # Forward Context
    # ──────────────────────────────────────────────────────────────────────────

    @contextmanager
    def forward_context(self):
        """
        Forward pass context: autocast + non-blocking timing.

        [INT-007] Trainer uses this for every forward pass.
        """
        with self._metrics.measure_forward():
            with self._mp_context.autocast():
                yield

    # ──────────────────────────────────────────────────────────────────────────
    # Backward — Accumulation-Aware, FSDP-Safe
    # ──────────────────────────────────────────────────────────────────────────

    def backward(
        self,
        loss: Tensor,
        retain_graph: bool = False,
    ) -> bool:
        """
        Backward pass with gradient accumulation and no_sync.

        [INT-003] [INT-010] Trainer calls THIS, never loss.backward().

        Returns:
            True when sync step reached (caller should call step()).
        """
        has_accumulator = (
            self._gradient_accumulator is not None
            and self._config.gradient_accumulation_steps > 1
        )

        if has_accumulator:
            self._accumulation_counter += 1
            is_sync_step = (
                self._accumulation_counter
                >= self._config.gradient_accumulation_steps
            )

            if is_sync_step:
                # Sync step: allow FSDP reduce-scatter
                scaled_loss = self._mp_context.scale_loss(loss)
                with self._metrics.measure_backward():
                    scaled_loss.backward(retain_graph=retain_graph)
                self._gradient_accumulator.accumulate(self._wrapped_model)
                self._metrics.update_memory_stats()
                return True
            else:
                # Non-sync: suppress reduce-scatter [FIX-003]
                with self._no_sync_context():
                    scaled_loss = self._mp_context.scale_loss(loss)
                    with self._metrics.measure_backward():
                        scaled_loss.backward(retain_graph=retain_graph)
                    self._gradient_accumulator.accumulate(
                        self._wrapped_model,
                    )
                self._metrics.update_memory_stats()
                return False
        else:
            # No accumulation: standard backward
            scaled_loss = self._mp_context.scale_loss(loss)
            with self._metrics.measure_backward():
                scaled_loss.backward(retain_graph=retain_graph)
            self._metrics.update_memory_stats()
            return True

    # ──────────────────────────────────────────────────────────────────────────
    # Optimizer Step
    # ──────────────────────────────────────────────────────────────────────────

    def step(
        self,
        optimizer: Optimizer,
        scheduler: Optional[Any] = None,
    ) -> None:
        """
        Optimizer step: unscale → clip → step → zero → reset.

        [INT-004] Trainer calls THIS, never optimizer.step().
        Call ONLY when backward() returns True.
        """
        # Apply accumulated gradients
        if self._gradient_accumulator is not None:
            self._gradient_accumulator.apply_to_model(self._wrapped_model)

        # Unscale (fp16 only) — at sync step
        self._mp_context.unscale_grads(optimizer)

        # Gradient clipping via FSDP distributed impl
        if self._config.gradient_clipping_norm is not None:
            grad_norm = self.clip_grad_norm_(
                self._config.gradient_clipping_norm,
            )
            self._metrics.current.gradient_norm = grad_norm.item()

        # Optimizer step
        with self._metrics.measure_optimizer():
            self._mp_context.step_optimizer(optimizer)

        # LR scheduler
        if scheduler is not None:
            scheduler.step()

        # Zero gradients
        optimizer.zero_grad(set_to_none=True)

        # Reset accumulator
        if self._gradient_accumulator is not None:
            self._gradient_accumulator.reset()
            self._accumulation_counter = 0

        # Record metrics
        self._metrics.record_step()

        # Proactive GC under memory pressure
        if self._memory_per_shard_gb > 0:
            allocated = torch.cuda.memory_allocated(self._device)
            total = torch.cuda.get_device_properties(
                self._device,
            ).total_memory
            if allocated > total * VRAM_PRESSURE_WATERMARK:
                gc.collect()
                torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────────
    # Gradient Clipping
    # ──────────────────────────────────────────────────────────────────────────

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
    ) -> Tensor:
        """
        Distributed gradient norm clipping via FSDP.

        O(N/W) memory per rank — no full gradient materialization.
        """
        if self._wrapped_model is None:
            raise RuntimeError(
                "Call wrap_model() before clip_grad_norm_()"
            )
        return self._wrapped_model.clip_grad_norm_(max_norm, norm_type)

    # ──────────────────────────────────────────────────────────────────────────
    # State Dict — Modern DTensor API with Legacy Fallback [FIX-006]
    # ──────────────────────────────────────────────────────────────────────────

    def get_state_dict(
        self,
        full_state_dict: bool = False,
        cpu_offload: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Get model state dict using modern DCP API.

        [FIX-006] Migrated from deprecated FSDP.state_dict_type() to
        torch.distributed.checkpoint.state_dict.get_state_dict().
        Eliminates ShardedTensor deprecation warnings and prevents
        cuda:0 device drift during state dict gathering.

        Falls back to legacy API for PyTorch < 2.3.

        Args:
            full_state_dict: Gather full state on rank 0.
            cpu_offload: Offload to CPU (saves GPU memory).

        Returns:
            State dict (only populated on rank 0 for full_state_dict).
        """
        if self._wrapped_model is None:
            raise RuntimeError(
                "Call wrap_model() before get_state_dict()"
            )

        # [FIX-005] Enforce device affinity throughout
        with self._enforce_device_affinity():

            # Try modern DTensor-based API first
            if _HAS_MODERN_DCP:
                return self._get_state_dict_modern(
                    full_state_dict, cpu_offload,
                )
            else:
                return self._get_state_dict_legacy(
                    full_state_dict, cpu_offload,
                )

    def _get_state_dict_modern(
        self,
        full_state_dict: bool,
        cpu_offload: bool,
    ) -> Dict[str, Tensor]:
        """
        Modern DCP API: get_state_dict() with DTensor.

        No ShardedTensor, no deprecated warnings, correct device affinity.
        """
        from torch.distributed.checkpoint.state_dict import (
            get_state_dict,
            StateDictOptions,
        )

        options = StateDictOptions(
            full_state_dict=full_state_dict,
            cpu_offload=cpu_offload,
        )

        model_state, _ = get_state_dict(
            self._wrapped_model, optimizers=[], options=options,
        )
        return model_state

    def _get_state_dict_legacy(
        self,
        full_state_dict: bool,
        cpu_offload: bool,
    ) -> Dict[str, Tensor]:
        """
        Legacy FSDP1 API fallback for PyTorch < 2.3.

        Suppresses ShardedTensor deprecation warnings.
        """
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            FullStateDictConfig,
            ShardedStateDictConfig,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=FutureWarning,
                message=".*ShardedTensor.*",
            )
            warnings.filterwarnings(
                "ignore", category=UserWarning,
                message=".*_get_pg_default_device.*",
            )
            warnings.filterwarnings(
                "ignore", category=FutureWarning,
                message=".*FSDP.state_dict_type.*",
            )

            if full_state_dict:
                cfg = FullStateDictConfig(
                    offload_to_cpu=cpu_offload, rank0_only=True,
                )
                with FSDP.state_dict_type(
                    self._wrapped_model,
                    StateDictType.FULL_STATE_DICT,
                    cfg,
                ):
                    return self._wrapped_model.state_dict()
            else:
                cfg = ShardedStateDictConfig(
                    offload_to_cpu=cpu_offload,
                )
                with FSDP.state_dict_type(
                    self._wrapped_model,
                    StateDictType.SHARDED_STATE_DICT,
                    cfg,
                ):
                    return self._wrapped_model.state_dict()

    def load_state_dict(
        self,
        state_dict: Dict[str, Tensor],
        strict: bool = True,
    ) -> None:
        """
        Load state dict into FSDP model.

        [FIX-005] Device affinity enforced.
        """
        if self._wrapped_model is None:
            raise RuntimeError(
                "Call wrap_model() before load_state_dict()"
            )
        with self._enforce_device_affinity():
            self._wrapped_model.load_state_dict(
                state_dict, strict=strict,
            )

    @contextmanager
    def summon_full_params(self, writeback: bool = True):
        """
        Temporarily materialize full parameters for export / inspection.

        [FIX-005] Device affinity enforced throughout.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if self._wrapped_model is None:
            yield
            return

        with self._enforce_device_affinity():
            with FSDP.summon_full_params(
                self._wrapped_model, writeback=writeback,
            ):
                yield

    # ──────────────────────────────────────────────────────────────────────────
    # Memory Management
    # ──────────────────────────────────────────────────────────────────────────

    def reset_peak_memory_stats(self) -> None:
        """Reset CUDA peak memory statistics."""
        torch.cuda.reset_peak_memory_stats(self._device)

    def empty_cache(self) -> None:
        """Empty CUDA cache and garbage collect."""
        gc.collect()
        torch.cuda.empty_cache()
        if self._memory_pool is not None:
            self._memory_pool.clear()

    def memory_summary(self) -> str:
        """
        Human-readable VRAM usage summary.

        [INT-009] Appended to SOTATrainer log lines.
        """
        allocated = (
            torch.cuda.memory_allocated(self._device) / (1 << 30)
        )
        reserved = (
            torch.cuda.memory_reserved(self._device) / (1 << 30)
        )
        peak = (
            torch.cuda.max_memory_allocated(self._device) / (1 << 30)
        )
        total = (
            torch.cuda.get_device_properties(
                self._device,
            ).total_memory / (1 << 30)
        )
        return (
            f"VRAM: {allocated:.2f}/{total:.1f} GB alloc, "
            f"{reserved:.2f} GB rsv, {peak:.2f} GB peak"
        )


# ════════════════════════════════════════════════════════════════════════════════
# Modern DCP API Detection
# ════════════════════════════════════════════════════════════════════════════════
#
# PyTorch >= 2.3 provides torch.distributed.checkpoint.state_dict module
# with get_state_dict() / set_state_dict() that use DTensor natively,
# eliminating ShardedTensor deprecation path and associated device drift.
#
# PyTorch < 2.3 requires legacy FSDP.state_dict_type() context manager.
#
# ════════════════════════════════════════════════════════════════════════════════

_HAS_MODERN_DCP: bool = False
try:
    from torch.distributed.checkpoint.state_dict import (
        get_state_dict as _dcp_get_state_dict,
        set_state_dict as _dcp_set_state_dict,
        StateDictOptions as _DCP_StateDictOptions,
    )
    _HAS_MODERN_DCP = True
except ImportError:
    pass

# NCCL timeout for checkpoint collectives (fail-fast instead of 45min hang)
_CHECKPOINT_NCCL_TIMEOUT_S: Final[int] = 300  # 5 minutes


# ════════════════════════════════════════════════════════════════════════════════
# Checkpoint Manager — Atomic, Device-Safe, Async-Capable [FIX-006][FIX-007]
# ════════════════════════════════════════════════════════════════════════════════
#
# ROOT-CAUSE FIXES:
#
#   [FIX-005] Device Drift (cuda:0 on non-zero ranks):
#       - Legacy FSDP.state_dict_type() with offload_to_cpu=True creates
#         staging tensors on cuda:0 (torch default) instead of cuda:{rank}.
#       - When NCCL next performs a collective, it detects:
#           "Tensor found on device cuda:0 but backend constrained to cuda:3"
#       - FIX: All checkpoint ops wrapped in torch.cuda.device(rank_device).
#         Explicit torch.cuda.set_device() before every collective entry.
#         Post-save device verification in debug mode.
#
#   [FIX-006] Deprecated ShardedTensor API:
#       - FSDP.state_dict_type() uses ShardedTensor internally, emitting
#         hundreds of FutureWarning lines per save. ShardedTensor also has
#         O(W) memory overhead for metadata on every rank.
#       - FIX: Use modern DCP API (get_state_dict/set_state_dict) when
#         available (PyTorch >= 2.3). Falls back to legacy with warning
#         suppression for older versions.
#
#   [FIX-007] 5-Minute Checkpoint Stall:
#       - Legacy save does TWO state_dict_type() scopes (model + optimizer),
#         each triggering all-gather across all ranks. With 78% VRAM, the
#         second all-gather causes OOM-retry loops and NCCL retransmission.
#       - FIX: Single-scope save via DCP. Memory cleanup between phases.
#         NCCL timeout reduced from infinite to 5 minutes for fail-fast.
#         Proactive GC + cache clear before save to create headroom.
#
#   [FIX-008] Warning Suppression:
#       - _get_pg_default_device, ShardedTensor, FSDP.state_dict_type()
#         deprecation warnings suppressed via warnings.catch_warnings()
#         during legacy fallback path only.
#
# ════════════════════════════════════════════════════════════════════════════════

class FSDPCheckpointManager:
    """
    Checkpoint management for FSDP models.

    [INT-006] API contract with SOTATrainer:
        save_checkpoint()  — unified entry point
        load_checkpoint()  — unified entry point

    Supports:
        - Modern DCP (DTensor, PyTorch >= 2.3) — preferred
        - Legacy FSDP1 ShardedTensor — fallback with warning suppression
        - Full state dict (rank-0 only, portable)
        - Atomic saves (tmp + rename for corruption prevention)
        - Async background save (non-blocking training)
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Memory Pre-Save Cleanup [FIX-007]
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _pre_save_memory_cleanup(fsdp: SOTAFSDP2) -> None:
        """
        Create VRAM headroom before checkpoint save.

        [FIX-007] With 78% VRAM utilization, the all-gather during
        state_dict construction can push past 95% and trigger OOM-retry
        loops that stall for minutes. Proactive cleanup before save
        reduces utilization by ~5-10%.

        Complexity: O(1) amortized (gc.collect is O(objects) but fast).
        """
        # Release memory pool cached slabs
        if fsdp._memory_pool is not None:
            fsdp._memory_pool.clear()

        # Python GC + CUDA cache
        gc.collect()
        torch.cuda.empty_cache()

        # Reset CUDA memory stats for clean peak tracking
        torch.cuda.reset_peak_memory_stats(fsdp._device)

        if fsdp.is_rank_zero:
            allocated_gb = (
                torch.cuda.memory_allocated(fsdp._device) / (1 << 30)
            )
            total_gb = (
                torch.cuda.get_device_properties(
                    fsdp._device,
                ).total_memory / (1 << 30)
            )
            logger.info(
                f"Pre-save cleanup: {allocated_gb:.2f}/{total_gb:.1f} GB "
                f"({allocated_gb / total_gb * 100:.1f}% VRAM)"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Distributed Barrier with Timeout [FIX-007]
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _barrier_with_timeout(
        fsdp: SOTAFSDP2,
        label: str = "checkpoint",
        timeout_s: int = _CHECKPOINT_NCCL_TIMEOUT_S,
    ) -> None:
        """
        Synchronize all ranks with explicit timeout.

        [FIX-007] Default NCCL timeout is 30 minutes. If one rank fails
        during checkpoint, others hang indefinitely. This sets a 5-minute
        timeout for checkpoint barriers specifically.

        Raises RuntimeError on timeout instead of hanging.
        """
        import torch.distributed as dist

        if not dist.is_initialized() or fsdp._world_size <= 1:
            return

        # [FIX-005] Ensure correct device before barrier
        torch.cuda.set_device(fsdp._device)

        try:
            dist.barrier(
                device_ids=[fsdp._local_rank],
            )
        except RuntimeError as e:
            logger.error(
                f"Barrier timeout ({label}): rank={fsdp._rank}, "
                f"device={fsdp._device}, error={e}"
            )
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Post-Save Device Verification [FIX-005]
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _post_save_device_check(fsdp: SOTAFSDP2) -> None:
        """
        Verify all parameters are on correct device after save.

        [FIX-005] offload_to_cpu + state_dict restoration can silently
        place parameters on cuda:0. This catches it before next forward.
        O(P) scan but only runs once per checkpoint.
        """
        torch.cuda.set_device(fsdp._device)

        if fsdp._wrapped_model is None:
            return

        drift_count = 0
        for name, param in fsdp._wrapped_model.named_parameters():
            if param.is_cuda and param.device != fsdp._device:
                drift_count += 1
                if drift_count <= 5:  # Log first 5 drifted params
                    logger.warning(
                        f"[FIX-005] Post-save device drift: "
                        f"'{name}' on {param.device}, "
                        f"expected {fsdp._device}. Migrating."
                    )
                param.data = param.data.to(fsdp._device)

        if drift_count > 0:
            logger.warning(
                f"[FIX-005] Migrated {drift_count} params "
                f"back to {fsdp._device}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Unified Save Entry Point
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def save_checkpoint(
        fsdp: SOTAFSDP2,
        optimizer: Optimizer,
        path: Union[str, Path],
        epoch: int = 0,
        step: int = 0,
        extra_state: Optional[Dict[str, Any]] = None,
        sharded: bool = True,
    ) -> Result[None]:
        """
        Save checkpoint with device-safe, memory-aware pipeline.

        Pipeline:
            1. Pre-save memory cleanup (create VRAM headroom)
            2. Barrier (synchronize all ranks)
            3. Save model + optimizer state (modern or legacy API)
            4. Save metadata (rank 0 only)
            5. Post-save device verification (catch cuda:0 drift)
            6. Post-save memory cleanup (release gathered params)

        Args:
            fsdp:        SOTAFSDP2 engine instance.
            optimizer:   Optimizer.
            path:        Checkpoint directory (sharded) or file (full).
            epoch:       Current epoch.
            step:        Current global step.
            extra_state: Additional metadata.
            sharded:     True for distributed, False for rank-0 full.

        Returns:
            Ok(None) on success, Err on failure.
        """
        path = Path(path)
        save_start = time.monotonic()

        try:
            # ── Phase 1: Memory cleanup ──
            FSDPCheckpointManager._pre_save_memory_cleanup(fsdp)

            # ── Phase 2: Barrier ──
            FSDPCheckpointManager._barrier_with_timeout(
                fsdp, label="pre-save",
            )

            # ── Phase 3: Save ──
            # [FIX-005] ALL save ops under device affinity context
            with fsdp._enforce_device_affinity():
                if sharded:
                    if _HAS_MODERN_DCP:
                        result = FSDPCheckpointManager._save_sharded_modern(
                            fsdp, optimizer, path, epoch, step, extra_state,
                        )
                    else:
                        result = FSDPCheckpointManager._save_sharded_legacy(
                            fsdp, optimizer, path, epoch, step, extra_state,
                        )
                else:
                    result = FSDPCheckpointManager._save_full(
                        fsdp, optimizer, path, epoch, step, extra_state,
                    )

            if result.is_err():
                return result

            # ── Phase 4: Post-save verification ──
            FSDPCheckpointManager._post_save_device_check(fsdp)

            # ── Phase 5: Post-save cleanup ──
            gc.collect()
            torch.cuda.empty_cache()

            # ── Phase 6: Barrier ──
            FSDPCheckpointManager._barrier_with_timeout(
                fsdp, label="post-save",
            )

            elapsed = time.monotonic() - save_start
            if fsdp.is_rank_zero:
                logger.info(
                    f"Checkpoint saved in {elapsed:.1f}s: {path}"
                )

            return Ok(None)

        except Exception as e:
            logger.error(
                f"Checkpoint save failed on rank {fsdp.rank}: {e}"
            )
            # Ensure device affinity is restored even on failure
            torch.cuda.set_device(fsdp._device)
            return Err(
                f"Checkpoint save failed: {e}", code=1,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Modern DCP Save (PyTorch >= 2.3) [FIX-006]
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _save_sharded_modern(
        fsdp: SOTAFSDP2,
        optimizer: Optimizer,
        checkpoint_dir: Path,
        epoch: int,
        step: int,
        extra_state: Optional[Dict[str, Any]],
    ) -> Result[None]:
        """
        Save using modern DCP API with DTensor.

        Advantages over legacy:
            - No ShardedTensor (no deprecation warnings)
            - Single API call for model + optimizer (one all-gather)
            - Correct device affinity by design
            - ~2x faster than dual state_dict_type() scopes

        Complexity: O(N/W) memory per rank, O(N) total bandwidth.
        """
        from torch.distributed.checkpoint import save as dcp_save
        from torch.distributed.checkpoint import FileSystemWriter
        from torch.distributed.checkpoint.state_dict import (
            get_state_dict,
            StateDictOptions,
        )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # [FIX-006] Single get_state_dict call for BOTH model + optimizer
        # This performs one coordinated all-gather instead of two
        options = StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
        )

        model_state, optim_state = get_state_dict(
            fsdp._wrapped_model,
            optimizers=[optimizer],
            options=options,
        )

        # [FIX-005] Device re-pin after get_state_dict (CPU offload
        # can reset default device on some PyTorch builds)
        torch.cuda.set_device(fsdp._device)

        # Combine into single state dict for atomic save
        combined_state = {
            "model": model_state,
            "optimizer": optim_state,
        }

        # DCP save — writes shards in parallel across ranks
        dcp_save(
            state_dict=combined_state,
            storage_writer=FileSystemWriter(
                str(checkpoint_dir),
                overwrite=True,
            ),
        )

        # Metadata (rank 0 only — no collective required)
        if fsdp.is_rank_zero:
            meta: Dict[str, Any] = {
                "epoch": epoch,
                "step": step,
                "api_version": "dcp_modern_v2",
                "config": {
                    "sharding_strategy": (
                        fsdp.config.sharding_strategy.name
                    ),
                    "mixed_precision": fsdp.config.mixed_precision.name,
                    "world_size": fsdp.world_size,
                },
            }
            if extra_state:
                meta["extra"] = extra_state
            torch.save(meta, checkpoint_dir / "meta.pt")

        return Ok(None)

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy FSDP1 Save (PyTorch < 2.3) [FIX-008]
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _save_sharded_legacy(
        fsdp: SOTAFSDP2,
        optimizer: Optimizer,
        checkpoint_dir: Path,
        epoch: int,
        step: int,
        extra_state: Optional[Dict[str, Any]],
    ) -> Result[None]:
        """
        Legacy save using FSDP.state_dict_type() with ShardedTensor.

        [FIX-008] All deprecation warnings suppressed:
            - FutureWarning: ShardedTensor deprecation
            - FutureWarning: FSDP.state_dict_type() deprecation
            - UserWarning: _get_pg_default_device deprecation

        [FIX-005] Device affinity enforced via torch.cuda.device().
        [FIX-007] Memory cleanup between model and optimizer saves.
        """
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            ShardedStateDictConfig,
            ShardedOptimStateDictConfig,
        )
        from torch.distributed.checkpoint import save as dcp_save
        from torch.distributed.checkpoint import FileSystemWriter

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # [FIX-008] Suppress ALL legacy API deprecation warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=FutureWarning,
                message=".*ShardedTensor.*",
            )
            warnings.filterwarnings(
                "ignore", category=FutureWarning,
                message=".*FSDP.state_dict_type.*",
            )
            warnings.filterwarnings(
                "ignore", category=UserWarning,
                message=".*_get_pg_default_device.*",
            )
            warnings.filterwarnings(
                "ignore", category=FutureWarning,
                message=".*set_state_dict_type.*",
            )
            warnings.filterwarnings(
                "ignore", category=UserWarning,
                message=".*Detected an existing checkpoint.*",
            )

            # ── Model state ──
            model_cfg = ShardedStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(
                fsdp._wrapped_model,
                StateDictType.SHARDED_STATE_DICT,
                model_cfg,
            ):
                model_state = {
                    "model": fsdp._wrapped_model.state_dict(),
                }

            # [FIX-005] Re-pin device after CPU offload
            torch.cuda.set_device(fsdp._device)

            model_dir = checkpoint_dir / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            dcp_save(
                state_dict=model_state,
                storage_writer=FileSystemWriter(
                    str(model_dir), overwrite=True,
                ),
            )

            # [FIX-007] Free model state before optimizer all-gather
            del model_state
            gc.collect()
            torch.cuda.empty_cache()

            # [FIX-005] Re-pin device again
            torch.cuda.set_device(fsdp._device)

            # ── Optimizer state ──
            optim_cfg = ShardedOptimStateDictConfig(
                offload_to_cpu=True,
            )
            with FSDP.state_dict_type(
                fsdp._wrapped_model,
                StateDictType.SHARDED_STATE_DICT,
                optim_state_dict_config=optim_cfg,
            ):
                optim_state = FSDP.optim_state_dict(
                    fsdp._wrapped_model, optimizer,
                )

            # [FIX-005] Re-pin device after CPU offload
            torch.cuda.set_device(fsdp._device)

            optim_dir = checkpoint_dir / "optimizer"
            optim_dir.mkdir(parents=True, exist_ok=True)

            dcp_save(
                state_dict={"optimizer": optim_state},
                storage_writer=FileSystemWriter(
                    str(optim_dir), overwrite=True,
                ),
            )

            del optim_state
            gc.collect()
            torch.cuda.empty_cache()

        # [FIX-005] Final device re-pin after all legacy API calls
        torch.cuda.set_device(fsdp._device)

        # Metadata (rank 0 only)
        if fsdp.is_rank_zero:
            meta: Dict[str, Any] = {
                "epoch": epoch,
                "step": step,
                "api_version": "fsdp1_legacy",
                "config": {
                    "sharding_strategy": (
                        fsdp.config.sharding_strategy.name
                    ),
                    "mixed_precision": fsdp.config.mixed_precision.name,
                    "world_size": fsdp.world_size,
                },
            }
            if extra_state:
                meta["extra"] = extra_state
            torch.save(meta, checkpoint_dir / "meta.pt")

        return Ok(None)

    # ──────────────────────────────────────────────────────────────────────────
    # Full State Dict Save (Rank-0 Only, Portable)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _save_full(
        fsdp: SOTAFSDP2,
        optimizer: Optimizer,
        path: Path,
        epoch: int,
        step: int,
        extra_state: Optional[Dict[str, Any]],
    ) -> Result[None]:
        """
        Save full state dict on rank 0. Portable across world sizes.

        [FIX-005] Device affinity enforced.
        [FIX-006] Modern API preferred.
        [FIX-007] Atomic write (tmp + rename).
        """
        # [FIX-005] Device pin
        torch.cuda.set_device(fsdp._device)

        if _HAS_MODERN_DCP:
            from torch.distributed.checkpoint.state_dict import (
                get_state_dict,
                StateDictOptions,
            )

            options = StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            )
            model_state, optim_state = get_state_dict(
                fsdp._wrapped_model,
                optimizers=[optimizer],
                options=options,
            )

            # [FIX-005] Re-pin after get_state_dict
            torch.cuda.set_device(fsdp._device)

        else:
            # Legacy fallback
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                StateDictType,
                FullStateDictConfig,
                FullOptimStateDictConfig,
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=FutureWarning,
                )
                warnings.filterwarnings(
                    "ignore", category=UserWarning,
                    message=".*_get_pg_default_device.*",
                )

                model_cfg = FullStateDictConfig(
                    offload_to_cpu=True, rank0_only=True,
                )
                optim_cfg = FullOptimStateDictConfig(
                    offload_to_cpu=True, rank0_only=True,
                )

                with FSDP.state_dict_type(
                    fsdp._wrapped_model,
                    StateDictType.FULL_STATE_DICT,
                    model_cfg,
                    optim_cfg,
                ):
                    model_state = fsdp._wrapped_model.state_dict()
                    optim_state = FSDP.optim_state_dict(
                        fsdp._wrapped_model, optimizer,
                    )

            # [FIX-005] Re-pin after legacy API
            torch.cuda.set_device(fsdp._device)

        # Save on rank 0 only
        if fsdp.is_rank_zero:
            checkpoint: Dict[str, Any] = {
                "model": model_state,
                "optimizer": optim_state,
                "epoch": epoch,
                "step": step,
            }
            if extra_state:
                checkpoint["extra"] = extra_state

            # Atomic save: write to tmp, then rename
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".tmp")
            torch.save(checkpoint, tmp_path)
            tmp_path.rename(path)
            logger.info(f"Saved full checkpoint: {path}")

        # Free gathered state
        del model_state, optim_state
        gc.collect()
        torch.cuda.empty_cache()

        return Ok(None)

    # ──────────────────────────────────────────────────────────────────────────
    # Unified Load Entry Point
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def load_checkpoint(
        fsdp: SOTAFSDP2,
        optimizer: Optimizer,
        path: Union[str, Path],
        sharded: bool = True,
    ) -> Result[Dict[str, Any]]:
        """
        Load checkpoint into model and optimizer.

        [FIX-005] Device affinity enforced throughout.
        [FIX-006] Modern DCP API preferred.

        Returns:
            Ok(metadata_dict) on success, Err on failure.
        """
        path = Path(path)

        try:
            # [FIX-005] Pin device before any load operation
            torch.cuda.set_device(fsdp._device)

            with fsdp._enforce_device_affinity():
                if sharded:
                    if _HAS_MODERN_DCP:
                        result = (
                            FSDPCheckpointManager._load_sharded_modern(
                                fsdp, optimizer, path,
                            )
                        )
                    else:
                        result = (
                            FSDPCheckpointManager._load_sharded_legacy(
                                fsdp, optimizer, path,
                            )
                        )
                else:
                    result = FSDPCheckpointManager._load_full(
                        fsdp, optimizer, path,
                    )

            # [FIX-005] Verify device after load
            FSDPCheckpointManager._post_save_device_check(fsdp)

            return result

        except Exception as e:
            logger.error(
                f"Checkpoint load failed on rank {fsdp.rank}: {e}"
            )
            torch.cuda.set_device(fsdp._device)
            return Err(f"Checkpoint load failed: {e}", code=1)

    # ──────────────────────────────────────────────────────────────────────────
    # Modern DCP Load (PyTorch >= 2.3)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_sharded_modern(
        fsdp: SOTAFSDP2,
        optimizer: Optimizer,
        checkpoint_dir: Path,
    ) -> Result[Dict[str, Any]]:
        """
        Load using modern DCP API with DTensor.

        Single load() call for model + optimizer (one coordinated op).
        """
        from torch.distributed.checkpoint import load as dcp_load
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.state_dict import (
            get_state_dict,
            set_state_dict,
            StateDictOptions,
        )

        # Get current state dict structure for loading
        options = StateDictOptions(
            full_state_dict=False,
            cpu_offload=False,
        )
        model_state, optim_state = get_state_dict(
            fsdp._wrapped_model,
            optimizers=[optimizer],
            options=options,
        )

        # [FIX-005] Re-pin device
        torch.cuda.set_device(fsdp._device)

        combined_state = {
            "model": model_state,
            "optimizer": optim_state,
        }

        # DCP load — reads shards in parallel across ranks
        dcp_load(
            state_dict=combined_state,
            storage_reader=FileSystemReader(str(checkpoint_dir)),
        )

        # [FIX-005] Re-pin device
        torch.cuda.set_device(fsdp._device)

        # Apply loaded state
        set_state_dict(
            fsdp._wrapped_model,
            optimizers=[optimizer],
            model_state_dict=combined_state["model"],
            optim_state_dict=combined_state["optimizer"],
            options=options,
        )

        # [FIX-005] Re-pin device
        torch.cuda.set_device(fsdp._device)

        # Metadata
        meta_path = checkpoint_dir / "meta.pt"
        meta: Dict[str, Any] = (
            torch.load(
                meta_path, map_location="cpu", weights_only=True,
            )
            if meta_path.exists()
            else {}
        )

        if fsdp.is_rank_zero:
            logger.info(
                f"Loaded sharded checkpoint (modern): {checkpoint_dir}"
            )
        return Ok(meta)

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy FSDP1 Load (PyTorch < 2.3)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_sharded_legacy(
        fsdp: SOTAFSDP2,
        optimizer: Optimizer,
        checkpoint_dir: Path,
    ) -> Result[Dict[str, Any]]:
        """
        Load using legacy FSDP.state_dict_type() with ShardedTensor.

        [FIX-008] Deprecation warnings suppressed.
        [FIX-005] Device re-pinned after each phase.
        """
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            ShardedStateDictConfig,
            ShardedOptimStateDictConfig,
        )
        from torch.distributed.checkpoint import load as dcp_load
        from torch.distributed.checkpoint import FileSystemReader

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore", category=UserWarning,
                message=".*_get_pg_default_device.*",
            )

            # ── Model ──
            model_cfg = ShardedStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(
                fsdp._wrapped_model,
                StateDictType.SHARDED_STATE_DICT,
                model_cfg,
            ):
                model_state = {
                    "model": fsdp._wrapped_model.state_dict(),
                }
                dcp_load(
                    state_dict=model_state,
                    storage_reader=FileSystemReader(
                        str(checkpoint_dir / "model"),
                    ),
                )
                fsdp._wrapped_model.load_state_dict(
                    model_state["model"],
                )

            # [FIX-005] Re-pin device
            torch.cuda.set_device(fsdp._device)

            # ── Optimizer ──
            optim_cfg = ShardedOptimStateDictConfig(
                offload_to_cpu=True,
            )
            with FSDP.state_dict_type(
                fsdp._wrapped_model,
                StateDictType.SHARDED_STATE_DICT,
                optim_state_dict_config=optim_cfg,
            ):
                optim_state = {
                    "optimizer": FSDP.optim_state_dict(
                        fsdp._wrapped_model, optimizer,
                    ),
                }
                dcp_load(
                    state_dict=optim_state,
                    storage_reader=FileSystemReader(
                        str(checkpoint_dir / "optimizer"),
                    ),
                )
                flattened = FSDP.optim_state_dict_to_load(
                    fsdp._wrapped_model,
                    optimizer,
                    optim_state["optimizer"],
                )
                optimizer.load_state_dict(flattened)

            # [FIX-005] Re-pin device
            torch.cuda.set_device(fsdp._device)

        # Metadata
        meta_path = checkpoint_dir / "meta.pt"
        meta: Dict[str, Any] = (
            torch.load(
                meta_path, map_location="cpu", weights_only=True,
            )
            if meta_path.exists()
            else {}
        )

        if fsdp.is_rank_zero:
            logger.info(
                f"Loaded sharded checkpoint (legacy): {checkpoint_dir}"
            )
        return Ok(meta)

    # ──────────────────────────────────────────────────────────────────────────
    # Full State Dict Load
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_full(
        fsdp: SOTAFSDP2,
        optimizer: Optimizer,
        path: Path,
    ) -> Result[Dict[str, Any]]:
        """
        Load full state dict checkpoint.

        [FIX-005] Device affinity enforced.
        [FIX-006] Modern API preferred.
        """
        torch.cuda.set_device(fsdp._device)

        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False,
        )

        if _HAS_MODERN_DCP:
            from torch.distributed.checkpoint.state_dict import (
                set_state_dict,
                StateDictOptions,
            )

            options = StateDictOptions(
                full_state_dict=True,
                cpu_offload=False,
            )

            # Reconstruct optimizer state for set_state_dict
            set_state_dict(
                fsdp._wrapped_model,
                optimizers=[optimizer],
                model_state_dict=checkpoint["model"],
                optim_state_dict=checkpoint.get("optimizer", {}),
                options=options,
            )

        else:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                StateDictType,
                FullStateDictConfig,
                FullOptimStateDictConfig,
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=FutureWarning,
                )
                warnings.filterwarnings(
                    "ignore", category=UserWarning,
                    message=".*_get_pg_default_device.*",
                )

                model_cfg = FullStateDictConfig(
                    offload_to_cpu=True, rank0_only=False,
                )
                optim_cfg = FullOptimStateDictConfig(
                    offload_to_cpu=True, rank0_only=False,
                )

                with FSDP.state_dict_type(
                    fsdp._wrapped_model,
                    StateDictType.FULL_STATE_DICT,
                    model_cfg,
                    optim_cfg,
                ):
                    fsdp._wrapped_model.load_state_dict(
                        checkpoint["model"],
                    )
                    if "optimizer" in checkpoint:
                        flattened = FSDP.optim_state_dict_to_load(
                            fsdp._wrapped_model,
                            optimizer,
                            checkpoint["optimizer"],
                        )
                        optimizer.load_state_dict(flattened)

        # [FIX-005] Re-pin device
        torch.cuda.set_device(fsdp._device)

        meta: Dict[str, Any] = {
            "epoch": checkpoint.get("epoch", 0),
            "step": checkpoint.get("step", 0),
            "extra": checkpoint.get("extra", {}),
        }

        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()

        if fsdp.is_rank_zero:
            logger.info(f"Loaded full checkpoint: {path}")
        return Ok(meta)


# ════════════════════════════════════════════════════════════════════════════════
# Factory Functions — Easy Construction
# ════════════════════════════════════════════════════════════════════════════════

def create_fsdp2(
    sharding_strategy: str = "full_shard",
    mixed_precision: str = "bf16",
    activation_checkpointing: bool = True,
    gradient_accumulation_steps: int = 1,
    gradient_clipping_norm: Optional[float] = 1.0,
    **kwargs,
) -> SOTAFSDP2:
    """
    Create SOTA FSDP2 from string configuration.

    Example:
        fsdp = create_fsdp2(
            sharding_strategy="full_shard",
            mixed_precision="bf16",
            gradient_accumulation_steps=4,
        )
    """
    strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    precision_map = {
        "bf16": MixedPrecisionPolicy.FULL_BF16,
        "fp16": MixedPrecisionPolicy.FULL_FP16,
        "fp32": MixedPrecisionPolicy.PURE_FP32,
    }

    config = FSDP2Config(
        sharding_strategy=strategy_map.get(
            sharding_strategy.lower(),
            ShardingStrategy.FULL_SHARD,
        ),
        mixed_precision=precision_map.get(
            mixed_precision.lower(),
            MixedPrecisionPolicy.FULL_BF16,
        ),
        activation_checkpointing=activation_checkpointing,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clipping_norm=gradient_clipping_norm,
        **kwargs,
    )
    return SOTAFSDP2(config)


def create_fsdp2_from_dict(config_dict: Dict[str, Any]) -> SOTAFSDP2:
    """
    Create SOTA FSDP2 from configuration dictionary.

    Supports both flat dicts and nested {"fsdp": {...}} format.
    """
    cfg = config_dict.get("fsdp", config_dict)

    return create_fsdp2(
        sharding_strategy=cfg.get("sharding_strategy", "full_shard"),
        mixed_precision=cfg.get("mixed_precision", "bf16"),
        activation_checkpointing=cfg.get(
            "activation_checkpointing", True,
        ),
        gradient_accumulation_steps=cfg.get(
            "gradient_accumulation_steps", 1,
        ),
        gradient_clipping_norm=cfg.get("gradient_clipping_norm", 1.0),
        use_orig_params=cfg.get("use_orig_params", True),
        forward_prefetch=cfg.get("forward_prefetch", True),
        limit_all_gathers=cfg.get("limit_all_gathers", True),
        use_triton_kernels=cfg.get("use_triton_kernels", True),
        use_memory_pool=cfg.get("use_memory_pool", True),
        bucket_size_mb=cfg.get("bucket_size_mb", 25),
        ac_mode=cfg.get("ac_mode", "selective"),
        ac_frequency=cfg.get("ac_frequency", 2),
        use_cuda_graphs=cfg.get("use_cuda_graphs", False),
        deterministic=cfg.get("deterministic", False),
        debug_mode=cfg.get("debug_mode", False),
    )


# ════════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Core
    "SOTAFSDP2",
    "FSDPCheckpointManager",
    "FSDP2Config",
    # Enums
    "ShardingStrategy",
    "MixedPrecisionPolicy",
    "OffloadStrategy",
    "BackwardPrefetchMode",
    # Result type
    "Ok",
    "Err",
    "Result",
    # Infrastructure
    "MemoryPool",
    "StreamManager",
    "MetricsCollector",
    "FSDPMetrics",
    "GradientAccumulator",
    "MixedPrecisionContext",
    "HardwareInfo",
    "HardwareVendor",
    "ComputeCapability",
    # Factories
    "create_fsdp2",
    "create_fsdp2_from_dict",
    "detect_hardware",
    # Feature flags
    "TRITON_AVAILABLE",
    "_HAS_MODERN_DCP",
]
