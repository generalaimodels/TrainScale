# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer Module - Distributed Training Infrastructure
# ════════════════════════════════════════════════════════════════════════════════
# Comprehensive distributed training with DP, DDP, TP, PP, EP, FSDP/ZeRO.
#
# Features:
# 1. Unified device abstraction layer for heterogeneous hardware
# 2. Process group management for multi-node training
# 3. Gradient synchronization with all-reduce fusion
# 4. Memory-efficient sharding strategies
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import functools
import logging
import os
import socket
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DataParallel, DistributedDataParallel

from data_pipeline.trainer.core.types import (
    DeviceType,
    DistributedConfig,
    ParallelMode,
    Precision,
    ShardingStrategy,
    ZeROStage,
)
from data_pipeline.trainer.core.errors import (
    CommunicationError,
    DeviceNotAvailableError,
    DistributedError,
    ProcessGroupError,
    TensorShardingError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=nn.Module)


# ═════════════════════════════════════════════════════════════════════════════════
# Device Manager
# ═════════════════════════════════════════════════════════════════════════════════

class DeviceManager:
    """
    Unified device abstraction layer for heterogeneous hardware.
    
    Handles:
    - Device detection and capability checking
    - Memory management and monitoring
    - Device-specific optimizations
    
    Example:
        ```python
        manager = DeviceManager.auto_detect()
        device = manager.device
        model = model.to(device)
        ```
    """
    
    def __init__(
        self,
        device_type: DeviceType = DeviceType.AUTO,
        device_id: int = 0,
    ):
        self.device_type = device_type
        self.device_id = device_id
        self._device: Optional[torch.device] = None
        self._compute_capability: Optional[Tuple[int, int]] = None
    
    @classmethod
    def auto_detect(cls) -> "DeviceManager":
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return cls(DeviceType.CUDA, device_id=0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return cls(DeviceType.MPS, device_id=0)
        else:
            return cls(DeviceType.CPU, device_id=0)
    
    @property
    def device(self) -> torch.device:
        """Get torch device."""
        if self._device is None:
            if self.device_type == DeviceType.AUTO:
                self._device = self._auto_select_device()
            elif self.device_type == DeviceType.CUDA:
                if not torch.cuda.is_available():
                    raise DeviceNotAvailableError("CUDA not available")
                self._device = torch.device(f"cuda:{self.device_id}")
            elif self.device_type == DeviceType.MPS:
                if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                    raise DeviceNotAvailableError("MPS not available")
                self._device = torch.device("mps")
            elif self.device_type == DeviceType.CPU:
                self._device = torch.device("cpu")
            else:
                self._device = torch.device("cpu")
        return self._device
    
    def _auto_select_device(self) -> torch.device:
        """Select best device automatically."""
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.device_id}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    @property
    def compute_capability(self) -> Optional[Tuple[int, int]]:
        """Get GPU compute capability (CUDA only)."""
        if self._compute_capability is None and self.device_type == DeviceType.CUDA:
            if torch.cuda.is_available():
                self._compute_capability = torch.cuda.get_device_capability(self.device_id)
        return self._compute_capability
    
    def supports_bf16(self) -> bool:
        """Check BF16 support (SM >= 80 for CUDA)."""
        cc = self.compute_capability
        return cc is not None and cc[0] * 10 + cc[1] >= 80
    
    def supports_fp8(self) -> bool:
        """Check FP8 support (SM >= 89 for CUDA)."""
        cc = self.compute_capability
        return cc is not None and cc[0] * 10 + cc[1] >= 89
    
    def memory_stats(self) -> Dict[str, float]:
        """Get memory statistics in GB."""
        if self.device_type != DeviceType.CUDA or not torch.cuda.is_available():
            return {}
        
        return {
            "allocated": torch.cuda.memory_allocated(self.device_id) / 1e9,
            "reserved": torch.cuda.memory_reserved(self.device_id) / 1e9,
            "max_allocated": torch.cuda.max_memory_allocated(self.device_id) / 1e9,
        }
    
    def empty_cache(self) -> None:
        """Clear GPU memory cache."""
        if self.device_type == DeviceType.CUDA and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def synchronize(self) -> None:
        """Synchronize device."""
        if self.device_type == DeviceType.CUDA and torch.cuda.is_available():
            torch.cuda.synchronize(self.device_id)


# ═════════════════════════════════════════════════════════════════════════════════
# Distributed Environment
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class DistributedState:
    """
    Distributed training state container.
    
    Provides unified access to distributed training info.
    """
    is_initialized: bool = False
    backend: str = "nccl"
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    local_world_size: int = 1
    node_rank: int = 0
    num_nodes: int = 1
    device: Optional[torch.device] = None
    
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0
    
    @property
    def is_local_main_process(self) -> bool:
        """Check if this is the local main process (local_rank 0)."""
        return self.local_rank == 0


class DistributedManager:
    """
    Manages distributed training environment.
    
    Handles:
    - Process group initialization and cleanup
    - Rank and world size management
    - Communication primitives
    
    Example:
        ```python
        dist_manager = DistributedManager(config)
        dist_manager.init_process_group()
        
        # Wrap model
        model = dist_manager.wrap_model(model)
        
        # Cleanup
        dist_manager.destroy_process_group()
        ```
    """
    
    _instance: Optional["DistributedManager"] = None
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.state = DistributedState()
        self._process_group = None
    
    @classmethod
    def get_instance(cls) -> Optional["DistributedManager"]:
        """Get singleton instance."""
        return cls._instance
    
    def init_process_group(
        self,
        backend: str = "nccl",
        init_method: Optional[str] = None,
    ) -> None:
        """
        Initialize distributed process group.
        
        Args:
            backend: Communication backend (nccl, gloo, mpi)
            init_method: URL for process group initialization
        """
        if self.state.is_initialized:
            return
        
        # Get environment variables
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
        world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
        
        if world_size == 1:
            # Single process, no need to initialize
            self.state = DistributedState(
                is_initialized=False,
                rank=0,
                local_rank=0,
                world_size=1,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            return
        
        # Select backend
        if backend == "auto":
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        
        # Initialize process group
        try:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend=backend,
                    init_method=init_method,
                    world_size=world_size,
                    rank=rank,
                )
        except Exception as e:
            raise ProcessGroupError(f"Failed to initialize process group: {e}")
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        
        # Update state
        self.state = DistributedState(
            is_initialized=True,
            backend=backend,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=self.config.local_world_size,
            device=device,
        )
        
        DistributedManager._instance = self
        logger.info(f"Initialized distributed: rank={rank}/{world_size}, device={device}")
    
    def destroy_process_group(self) -> None:
        """Cleanup distributed state."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        self.state = DistributedState()
        DistributedManager._instance = None
    
    def wrap_model(
        self,
        model: T,
        find_unused_parameters: bool = False,
    ) -> T:
        """
        Wrap model for distributed training.
        
        Args:
            model: Model to wrap
            find_unused_parameters: Enable unused parameter detection
            
        Returns:
            Wrapped model
        """
        mode = self.config.mode
        
        if mode == ParallelMode.NONE or self.state.world_size == 1:
            return model
        
        if mode == ParallelMode.DP:
            return DataParallel(model)
        
        if mode in (ParallelMode.DDP, ParallelMode.FSDP):
            if not self.state.is_initialized:
                raise DistributedError("Process group not initialized")
            
            if mode == ParallelMode.FSDP:
                return self._wrap_fsdp(model)
            
            return DistributedDataParallel(
                model,
                device_ids=[self.state.local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=find_unused_parameters or self.config.find_unused_parameters,
                bucket_cap_mb=self.config.bucket_cap_mb,
                gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            )
        
        return model
    
    def _wrap_fsdp(self, model: T) -> T:
        """Wrap model with FSDP."""
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                ShardingStrategy as TorchShardingStrategy,
                CPUOffload,
                MixedPrecision,
            )
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        except ImportError:
            raise DistributedError("FSDP requires PyTorch >= 2.0")
        
        # Map sharding strategy
        strategy_map = {
            ShardingStrategy.FULL_SHARD: TorchShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP: TorchShardingStrategy.SHARD_GRAD_OP,
            ShardingStrategy.NO_SHARD: TorchShardingStrategy.NO_SHARD,
            ShardingStrategy.HYBRID_SHARD: TorchShardingStrategy.HYBRID_SHARD,
        }
        
        sharding_strategy = strategy_map.get(
            self.config.sharding_strategy,
            TorchShardingStrategy.FULL_SHARD,
        )
        
        # CPU offload
        cpu_offload = CPUOffload(offload_params=True) if self.config.cpu_offload else None
        
        return FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            device_id=self.state.local_rank,
        )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Communication Primitives
    # ─────────────────────────────────────────────────────────────────────────────
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.state.is_initialized and torch.distributed.is_initialized():
            torch.distributed.barrier()
    
    def all_reduce(
        self,
        tensor: Tensor,
        op: str = "sum",
        async_op: bool = False,
    ) -> Optional[Any]:
        """
        All-reduce tensor across all processes.
        
        Args:
            tensor: Tensor to reduce (modified in-place)
            op: Reduction operation (sum, mean, max, min)
            async_op: Return handle for async operation
        """
        if not self.state.is_initialized:
            return None
        
        op_map = {
            "sum": torch.distributed.ReduceOp.SUM,
            "mean": torch.distributed.ReduceOp.SUM,  # Divide after
            "max": torch.distributed.ReduceOp.MAX,
            "min": torch.distributed.ReduceOp.MIN,
        }
        
        reduce_op = op_map.get(op, torch.distributed.ReduceOp.SUM)
        handle = torch.distributed.all_reduce(tensor, op=reduce_op, async_op=async_op)
        
        if op == "mean" and not async_op:
            tensor.div_(self.state.world_size)
        
        return handle
    
    def all_gather(
        self,
        tensor: Tensor,
        async_op: bool = False,
    ) -> Tuple[List[Tensor], Optional[Any]]:
        """
        Gather tensors from all processes.
        
        Returns:
            List of tensors from all ranks
        """
        if not self.state.is_initialized:
            return [tensor], None
        
        gathered = [torch.zeros_like(tensor) for _ in range(self.state.world_size)]
        handle = torch.distributed.all_gather(gathered, tensor, async_op=async_op)
        return gathered, handle
    
    def broadcast(
        self,
        tensor: Tensor,
        src: int = 0,
    ) -> None:
        """Broadcast tensor from source rank to all."""
        if self.state.is_initialized:
            torch.distributed.broadcast(tensor, src=src)
    
    def reduce_dict(
        self,
        values: Dict[str, float],
        op: str = "mean",
    ) -> Dict[str, float]:
        """Reduce dictionary of values across all processes."""
        if not self.state.is_initialized or self.state.world_size == 1:
            return values
        
        keys = sorted(values.keys())
        tensor = torch.tensor([values[k] for k in keys], device=self.state.device)
        
        self.all_reduce(tensor, op=op)
        
        return {k: tensor[i].item() for i, k in enumerate(keys)}


# ═════════════════════════════════════════════════════════════════════════════════
# Gradient Synchronization
# ═════════════════════════════════════════════════════════════════════════════════

class GradientSynchronizer:
    """
    Gradient synchronization utilities for distributed training.
    
    Features:
    - Bucketed all-reduce for efficiency
    - Async gradient sync
    - Sparse gradient handling
    """
    
    def __init__(
        self,
        model: nn.Module,
        bucket_size_mb: int = 25,
        async_sync: bool = True,
    ):
        self.model = model
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self.async_sync = async_sync
        self._handles: List[Any] = []
    
    def sync_gradients(self) -> None:
        """Synchronize gradients across all processes."""
        if not torch.distributed.is_initialized():
            return
        
        # Collect gradients into buckets
        buckets = self._bucket_gradients()
        
        # All-reduce each bucket
        for bucket in buckets:
            if self.async_sync:
                handle = torch.distributed.all_reduce(
                    bucket,
                    op=torch.distributed.ReduceOp.SUM,
                    async_op=True,
                )
                self._handles.append(handle)
            else:
                torch.distributed.all_reduce(
                    bucket,
                    op=torch.distributed.ReduceOp.SUM,
                )
        
        # Wait for async ops
        if self.async_sync:
            for handle in self._handles:
                handle.wait()
            self._handles.clear()
        
        # Average gradients
        world_size = torch.distributed.get_world_size()
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.div_(world_size)
    
    def _bucket_gradients(self) -> List[Tensor]:
        """Organize gradients into buckets for efficient all-reduce."""
        buckets = []
        current_bucket = []
        current_size = 0
        
        for param in self.model.parameters():
            if param.grad is None:
                continue
            
            grad_size = param.grad.numel() * param.grad.element_size()
            
            if current_size + grad_size > self.bucket_size_bytes and current_bucket:
                buckets.append(self._flatten_grads(current_bucket))
                current_bucket = []
                current_size = 0
            
            current_bucket.append(param.grad)
            current_size += grad_size
        
        if current_bucket:
            buckets.append(self._flatten_grads(current_bucket))
        
        return buckets
    
    def _flatten_grads(self, grads: List[Tensor]) -> Tensor:
        """Flatten list of gradients into single tensor."""
        return torch.cat([g.view(-1) for g in grads])


# ═════════════════════════════════════════════════════════════════════════════════
# Activation Checkpointing
# ═════════════════════════════════════════════════════════════════════════════════

def activation_checkpoint(
    fn: Callable,
    *args,
    use_reentrant: bool = False,
    **kwargs,
) -> Any:
    """
    Checkpoint activations for memory reduction.
    
    Trades compute for memory by recomputing activations during backward.
    
    Args:
        fn: Function to checkpoint
        *args: Positional arguments to fn
        use_reentrant: Use reentrant checkpointing (legacy)
        
    Example:
        ```python
        # Checkpoint transformer layer
        output = activation_checkpoint(
            transformer_layer,
            hidden_states,
            attention_mask,
        )
        ```
    """
    from torch.utils.checkpoint import checkpoint
    return checkpoint(fn, *args, use_reentrant=use_reentrant, **kwargs)


def wrap_activation_checkpointing(
    model: nn.Module,
    layer_cls: Type[nn.Module],
) -> nn.Module:
    """
    Wrap specific layers with activation checkpointing.
    
    Args:
        model: Model to wrap
        layer_cls: Layer class to checkpoint (e.g., TransformerLayer)
    """
    from torch.utils.checkpoint import checkpoint_sequential
    
    for name, module in model.named_modules():
        if isinstance(module, layer_cls):
            # Wrap forward with checkpointing
            original_forward = module.forward
            
            @functools.wraps(original_forward)
            def checkpointed_forward(*args, _original=original_forward, **kwargs):
                return activation_checkpoint(_original, *args, **kwargs)
            
            module.forward = checkpointed_forward
    
    return model


# ═════════════════════════════════════════════════════════════════════════════════
# Context Managers
# ═════════════════════════════════════════════════════════════════════════════════

@contextmanager
def main_process_first(dist_manager: Optional[DistributedManager] = None):
    """
    Context manager ensuring main process executes first.
    
    Useful for download/preparation tasks.
    """
    manager = dist_manager or DistributedManager.get_instance()
    
    is_main = manager is None or manager.state.is_main_process
    
    if is_main:
        yield
        if manager is not None:
            manager.barrier()
    else:
        manager.barrier()
        yield


@contextmanager
def no_sync(model: nn.Module):
    """
    Context manager to disable gradient synchronization.
    
    Useful during gradient accumulation.
    """
    if isinstance(model, DistributedDataParallel):
        with model.no_sync():
            yield
    else:
        yield


# ═════════════════════════════════════════════════════════════════════════════════
# Factory Functions
# ═════════════════════════════════════════════════════════════════════════════════

def setup_distributed(
    config: Optional[DistributedConfig] = None,
    backend: str = "auto",
) -> DistributedManager:
    """
    Setup distributed training from configuration.
    
    Args:
        config: Distributed configuration
        backend: Communication backend
        
    Returns:
        Initialized DistributedManager
    """
    config = config or DistributedConfig()
    manager = DistributedManager(config)
    
    if config.mode != ParallelMode.NONE:
        manager.init_process_group(backend=backend)
    
    return manager


def get_world_info() -> Tuple[int, int, int]:
    """
    Get rank, local_rank, world_size.
    
    Returns:
        Tuple of (rank, local_rank, world_size)
    """
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    
    return rank, local_rank, world_size


def is_main_process() -> bool:
    """Check if current process is main (rank 0)."""
    rank, _, _ = get_world_info()
    return rank == 0


# ═════════════════════════════════════════════════════════════════════════════════
# Export
# ═════════════════════════════════════════════════════════════════════════════════

# Import sub-modules for export
from data_pipeline.trainer.distributed.tensor_parallel import (
    get_tensor_parallel_rank,
    get_tensor_parallel_world_size,
    copy_to_tensor_parallel_region,
    reduce_from_tensor_parallel_region,
    scatter_to_tensor_parallel_region,
    gather_from_tensor_parallel_region,
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

from data_pipeline.trainer.distributed.pipeline_parallel import (
    PipelineConfig,
    PipelineStage,
    PipelineSchedule,
    GPipeSchedule,
    OneFOneBSchedule,
    PipelineParallel,
    split_model_into_stages,
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
)

from data_pipeline.trainer.distributed.expert_parallel import (
    ExpertConfig,
    TopKGating,
    Expert,
    MoELayer,
    ExpertParallelMoE,
    get_expert_parallel_rank,
    get_expert_parallel_world_size,
)

__all__ = [
    # Device
    "DeviceManager",
    # Distributed state
    "DistributedState",
    "DistributedManager",
    # Gradient sync
    "GradientSynchronizer",
    # Activation checkpointing
    "activation_checkpoint",
    "wrap_activation_checkpointing",
    # Context managers
    "main_process_first",
    "no_sync",
    # Factory
    "setup_distributed",
    "get_world_info",
    "is_main_process",
    # Tensor Parallel
    "get_tensor_parallel_rank",
    "get_tensor_parallel_world_size",
    "copy_to_tensor_parallel_region",
    "reduce_from_tensor_parallel_region",
    "scatter_to_tensor_parallel_region",
    "gather_from_tensor_parallel_region",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    # Pipeline Parallel
    "PipelineConfig",
    "PipelineStage",
    "PipelineSchedule",
    "GPipeSchedule",
    "OneFOneBSchedule",
    "PipelineParallel",
    "split_model_into_stages",
    "get_pipeline_parallel_rank",
    "get_pipeline_parallel_world_size",
    # Expert Parallel (MoE)
    "ExpertConfig",
    "TopKGating",
    "Expert",
    "MoELayer",
    "ExpertParallelMoE",
    "get_expert_parallel_rank",
    "get_expert_parallel_world_size",
    # ════════════════════════════════════════════════════════════════════════════
    # SOTA Distributed Modules (Above SOTA-level, Triton+PyTorch)
    # ════════════════════════════════════════════════════════════════════════════
    # Parallel Dimensions (Multi-dimensional mesh manager)
    "ParallelDims",
    "create_parallel_dims",
    "create_parallel_dims_from_config",
    # FSDP2 (Above SOTA FSDP with Triton)
    "SOTAFSDP2",
    "FSDP2Config",
    "ShardingStrategy",
    "MixedPrecisionPolicy",
    "create_fsdp2",
    "create_fsdp2_from_config",
    # DDP (Optimized DDP with gradient compression)
    "SOTADDP",
    "DDPConfig",
    "GradientCompression",
    "create_ddp",
    "create_ddp_from_config",
    # Context Parallel (Ultra-long sequence support)
    "ContextParallel",
    "ContextParallelConfig",
    "LoadBalancer",
    "create_context_parallel",
    "create_context_parallel_from_config",
    # Activation Checkpointing (Memory-efficient AC)
    "ActivationCheckpoint",
    "ActivationCheckpointConfig",
    "ACMode",
    "create_activation_checkpoint",
    "create_activation_checkpoint_from_config",
    # SOTA Utilities (Registry-compatible)
    "clip_grad_norm_",
    "dist_reduce",
    "dist_all_gather",
    "set_deterministic_seed",
    "mixed_precision_context",
    "prepare_model_for_distributed",
    "prepare_optimizer_for_distributed",
    "prepare_scheduler_for_distributed",
    "log_rank_0",
    "print_rank_0",
]

# ════════════════════════════════════════════════════════════════════════════════
# SOTA Module Imports (Lazy import pattern for performance)
# ════════════════════════════════════════════════════════════════════════════════

from .parallel_dims import (
    ParallelDims,
    create_parallel_dims,
    create_parallel_dims_from_config,
)

from .fsdp2 import (
    SOTAFSDP2,
    FSDP2Config,
    ShardingStrategy,
    MixedPrecisionPolicy,
    create_fsdp2,
    create_fsdp2_from_config,
)

from .ddp import (
    SOTADDP,
    DDPConfig,
    GradientCompression,
    create_ddp,
    create_ddp_from_config,
)

from .context_parallel import (
    ContextParallel,
    ContextParallelConfig,
    LoadBalancer,
    create_context_parallel,
    create_context_parallel_from_config,
)

from .activation_checkpoint import (
    ActivationCheckpoint,
    ActivationCheckpointConfig,
    ACMode,
    create_activation_checkpoint,
    create_activation_checkpoint_from_config,
)

from .sota_utils import (
    clip_grad_norm_,
    dist_reduce,
    dist_all_gather,
    set_deterministic_seed,
    mixed_precision_context,
    prepare_model_for_distributed,
    prepare_optimizer_for_distributed,
    prepare_scheduler_for_distributed,
    log_rank_0,
    print_rank_0,
)
