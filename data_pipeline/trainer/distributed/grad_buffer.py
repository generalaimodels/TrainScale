# ════════════════════════════════════════════════════════════════════════════════
# CONTIGUOUS PARAM & GRAD BUFFER — PRODUCTION-GRADE MEGATRON-STYLE MEMORY LAYOUT
# ════════════════════════════════════════════════════════════════════════════════
# High-performance contiguous buffer system that collocates parameter and
# gradient tensors into flat, aligned allocations.  This is the foundation
# for every advanced Megatron-LM optimisation:
#
#   ● Distributed optimizer — reduce-scatter on contiguous shard views
#   ● Overlapped grad reduce — per-bucket async dispatch as grads become ready
#   ● Overlapped param gather — all-gather pipelined with forward compute
#   ● Coalesced collectives — multiple buckets in single NCCL launch via
#     torch.distributed._coalescing_manager
#   ● NCCL user-buffer registration — pin contiguous buffers in NCCL for
#     zero-copy kernel-launch transport (NVLink 4.0+ / IB NDR)
#   ● CPU offload & reload — release GPU memory between pipeline stages
#
# Memory Layout (distributed optimizer enabled):
# ┌───────────────────────────────────────────────────────────────────────────┐
# │  Bucket 0 (padded to DP-divisible)  │  Bucket 1  │  …  │  Bucket N     │
# │  ┌─────────────────────────────────┐ │            │     │               │
# │  │ param_k│pad│param_k-1│…│param_0│ │            │     │               │
# │  └─────────────────────────────────┘ │            │     │               │
# └───────────────────────────────────────────────────────────────────────────┘
#   Parameters are iterated in *reverse* registration order so that the first
#   bucket corresponds to the last layers → backward pass fills buckets in
#   natural order, enabling overlap of communication with remaining compute.
#
# Bucket Padding:
#   • Each bucket end is padded to lcm(dp_world_size, 128) for 256-byte
#     alignment of every shard (needed for TE cuBLAS compatibility).
#   • Optional pad to lcm(dp_world_size, 128, 2^16) when
#     pad_buckets_for_high_nccl_busbw is set, ensuring NCCL ring message
#     sizes are powers-of-2 divisible for peak bus bandwidth at high DP.
#   • Each param start is 128-byte aligned (64 elements × ≥16-bit dtype).
#
# Complexity:
#   Buffer allocation: O(N) where N = total param elements
#   Bucket formation:  O(P) where P = number of parameters
#   Grad sync:         O(N / dp_world_size) communication per rank
#   Param gather:      O(N / dp_world_size) communication per rank
#
# Reference: Megatron-LM core/distributed/param_and_grad_buffer
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
import math
import warnings
from contextlib import nullcontext
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple

import torch
import torch.distributed as dist

from .reduce_scatter_fp32 import reduce_scatter_with_fp32_accumulation

logger = logging.getLogger("sota_ddp.grad_buffer")

# ════════════════════════════════════════════════════════════════════════════════
# COMPILE-TIME CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

_PARAM_ALIGNMENT_ELEMS: Final[int] = 64     # 128-byte alignment for ≥16-bit dtypes
_BUCKET_ALIGNMENT_ELEMS: Final[int] = 128   # 256-byte alignment for bucket boundaries
_HIGH_BUSBW_POWER: Final[int] = 2 ** 16     # Ensures NCCL ring messages are 2^16-divisible

# ════════════════════════════════════════════════════════════════════════════════
# TORCH VERSION-AWARE COLLECTIVE DISPATCH
# ════════════════════════════════════════════════════════════════════════════════

try:
    # torch ≥ 1.13 renamed the internal APIs
    _dist_all_gather = torch.distributed.all_gather_into_tensor
    _dist_reduce_scatter = torch.distributed.reduce_scatter_tensor
except AttributeError:
    _dist_all_gather = torch.distributed._all_gather_base        # type: ignore[attr-defined]
    _dist_reduce_scatter = torch.distributed._reduce_scatter_base  # type: ignore[attr-defined]


# ════════════════════════════════════════════════════════════════════════════════
# BUFFER TYPE ENUM
# ════════════════════════════════════════════════════════════════════════════════

class BufferType(Enum):
    """Discriminator for views into the contiguous allocation."""
    PARAM = 1
    GRAD = 2


# ════════════════════════════════════════════════════════════════════════════════
# SHARD UTILITY
# ════════════════════════════════════════════════════════════════════════════════

def shard_buffer(
    buffer: torch.Tensor,
    data_parallel_world_size: int,
) -> List[torch.Tensor]:
    """
    Partition *buffer* into *data_parallel_world_size* equal shards.

    Used by the distributed optimizer to obtain per-rank views for
    reduce-scatter (grad sync) and all-gather (param sync).

    Args:
        buffer: 1-D contiguous tensor whose numel is divisible by
            *data_parallel_world_size*.
        data_parallel_world_size: Number of ranks in the data-parallel group.

    Returns:
        List of non-overlapping tensor views, one per rank.

    Complexity: O(dp_world_size) — creates lightweight views only.
    """
    assert buffer.numel() % data_parallel_world_size == 0, (
        f"buffer.numel()={buffer.numel()} not divisible by "
        f"data_parallel_world_size={data_parallel_world_size}"
    )
    shard_size = buffer.numel() // data_parallel_world_size
    return [
        buffer[r * shard_size : (r + 1) * shard_size]
        for r in range(data_parallel_world_size)
    ]


# ════════════════════════════════════════════════════════════════════════════════
# PARAM AND GRAD BUCKET
# ════════════════════════════════════════════════════════════════════════════════

class ParamAndGradBucket:
    """
    A contiguous slice of the global ParamAndGradBuffer that tracks a subset
    of model parameters and their gradients.

    Each bucket has its own ``param_data`` / ``grad_data`` view which is the
    unit of communication in all-reduce or reduce-scatter.

    Attributes:
        params_list: Ordered list of parameters in this bucket.
        params:      Set view for O(1) membership tests.
        param_data:  View into the buffer's param allocation (None when
                     distributed optimizer is disabled).
        grad_data:   View into the buffer's grad allocation.
        offset:      Start offset of this bucket within the full buffer.
        numel_unpadded: Actual number of parameter elements (before padding).
        gradient_scaling_factor: Pre-multiply applied before the collective
                     (encodes both 1/world_size averaging and MoE scaling).
        bucket_id:   Index within the parent buffer's bucket list.
        param_to_index: Maps each param to its (start, end) offset within
                     this bucket's view.
    """

    __slots__ = (
        "params_list",
        "params",
        "param_data",
        "grad_data",
        "offset",
        "numel_unpadded",
        "gradient_scaling_factor",
        "bucket_id",
        "param_to_index",
    )

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        param_data: Optional[torch.Tensor],
        grad_data: torch.Tensor,
        offset: int,
        numel_unpadded: int,
        gradient_scaling_factor: float,
        bucket_id: int,
    ) -> None:
        self.params_list = params
        self.params: Set[torch.nn.Parameter] = set(params)
        assert len(self.params_list) == len(self.params), "Duplicate params in bucket"

        self.param_data = param_data
        self.grad_data = grad_data
        self.offset = offset
        self.numel_unpadded = numel_unpadded
        self.gradient_scaling_factor = gradient_scaling_factor
        self.bucket_id = bucket_id

        # Build per-param index mapping
        self.param_to_index: Dict[torch.nn.Parameter, Tuple[int, int]] = {}
        cur = 0
        for p in params:
            self.param_to_index[p] = (cur, cur + p.numel())
            cur += p.numel()


# ════════════════════════════════════════════════════════════════════════════════
# PARAM AND GRAD BUCKET GROUP — COALESCED COLLECTIVE DISPATCH
# ════════════════════════════════════════════════════════════════════════════════

class ParamAndGradBucketGroup:
    """
    Aggregates multiple ``ParamAndGradBucket`` instances so that their NCCL
    communication kernels are coalesced into a single launch via
    ``torch.distributed._coalescing_manager``.

    This class manages:
      • **Grad sync** — reduce-scatter (distributed optimizer) or all-reduce
        dispatched asynchronously as soon as *all* params in the group have
        their grads ready (tracked via golden reference counts).
      • **Param sync** — all-gather of the local parameter shard after the
        optimizer step, pipelined with forward compute.
      • **Cached shard views** — avoids CPU overhead of re-slicing on every
        iteration.

    Lifecycle (per training step):
      1. ``reset()`` — clear bookkeeping for new iteration.
      2. Backward hooks call ``register_grad_ready(param)`` per param.
      3. When all params are ready → ``start_grad_sync()`` fires automatically
         (if ``overlap_grad_reduce`` is True).
      4. ``finish_grad_sync()`` — blocks until the collective completes (or
         fires synchronous collective if ``overlap_grad_reduce`` is False).
      5. After optimizer step → ``start_param_sync()`` / ``finish_param_sync()``
         to reconstruct full parameter tensors.

    Args:
        buckets: List of ``ParamAndGradBucket`` to aggregate.
        ddp_config: ``MegatronDDPConfig`` providing all knobs.
        collective_group: Process group for the collective
            (``intra_distributed_optimizer_instance_group`` if using dist-opt,
            ``data_parallel_group`` otherwise).
        collective_group_size: Size of *collective_group*.
    """

    __slots__ = (
        "buckets",
        "ddp_config",
        # Distributed-optimizer groups
        "intra_distributed_optimizer_instance_group",
        "intra_distributed_optimizer_instance_size",
        "intra_distributed_optimizer_instance_rank",
        # Non-distributed-optimizer group
        "data_parallel_group",
        # Bookkeeping
        "param_to_bucket",
        "params",
        "next_param_gather_bucket_group",
        # Grad-ready tracking
        "golden_per_param_grad_ready_counts",
        "per_param_grad_ready_counts",
        "is_last_microbatch",
        "is_first_batch",
        # Collective handles
        "param_gather_handle",
        "param_gather_dispatched",
        "grad_reduce_handle",
        # Cached shard views (avoid repeated CPU slicing)
        "cached_param_buffer_shard_list",
        "cached_grad_buffer_shard_list",
        # Communication stream (multi-instance DistOpt)
        "communication_stream",
    )

    def __init__(
        self,
        buckets: List[ParamAndGradBucket],
        ddp_config: Any,  # MegatronDDPConfig (forward ref to avoid circular)
        collective_group: dist.ProcessGroup,
        collective_group_size: int,
    ) -> None:
        self.buckets = buckets
        self.ddp_config = ddp_config

        # ── Assign collective groups ─────────────────────────────────────
        if getattr(self.ddp_config, "use_distributed_optimizer", False):
            self.intra_distributed_optimizer_instance_group = collective_group
            self.intra_distributed_optimizer_instance_size = collective_group_size
            self.intra_distributed_optimizer_instance_rank = collective_group.rank()
        else:
            self.data_parallel_group = collective_group

        # ── Param → bucket mapping ───────────────────────────────────────
        self.param_to_bucket: Dict[torch.nn.Parameter, ParamAndGradBucket] = {}
        self.params: Set[torch.nn.Parameter] = set()
        for bucket in self.buckets:
            for param in bucket.params_list:
                self.param_to_bucket[param] = bucket
                self.params.add(param)

        self.next_param_gather_bucket_group: Optional[ParamAndGradBucketGroup] = None

        # ── Grad-ready bookkeeping ───────────────────────────────────────
        # golden_per_param_grad_ready_counts: recorded after first batch so we
        # know exactly how many times each param's grad hook fires (handles
        # control-flow that revisits the same param).
        self.golden_per_param_grad_ready_counts: Dict[torch.nn.Parameter, int] = {}
        self.per_param_grad_ready_counts: Dict[torch.nn.Parameter, int] = {}
        self.is_last_microbatch: bool = True
        self.is_first_batch: bool = True

        # ── Collective handles ───────────────────────────────────────────
        self.param_gather_handle: Optional[Any] = None
        self.param_gather_dispatched: bool = False
        self.grad_reduce_handle: Optional[Any] = None

        # ── Cached shard views ───────────────────────────────────────────
        self.cached_param_buffer_shard_list: List[Optional[List[torch.Tensor]]] = [
            None
        ] * len(self.buckets)
        self.cached_grad_buffer_shard_list: List[Optional[List[torch.Tensor]]] = [
            None
        ] * len(self.buckets)

        # ── Communication stream (for multi-instance DistOpt overlap) ────
        self.communication_stream: Optional[torch.cuda.Stream] = None

        # ── Swap reduce-scatter implementation if FP32 accumulation ──────
        global _dist_reduce_scatter
        if getattr(self.ddp_config, "reduce_scatter_with_fp32_accumulation", False):
            _dist_reduce_scatter = reduce_scatter_with_fp32_accumulation
            logger.info(
                "Using reduce_scatter_with_fp32_accumulation as reduce-scatter "
                "implementation for numerical stability at high DP counts"
            )

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Clear iteration-local bookkeeping.  Called at the start of each
        training step.

        On the *first* batch, the golden reference counts are captured so
        that subsequent batches know exactly when all grads are ready.
        """
        if self.is_first_batch and len(self.per_param_grad_ready_counts) > 0:
            assert len(self.per_param_grad_ready_counts) == len(self.params)
            self.golden_per_param_grad_ready_counts = self.per_param_grad_ready_counts
            self.is_first_batch = False
        self.per_param_grad_ready_counts = {}
        self.is_last_microbatch = True

    # ── Gradient Health Checks ────────────────────────────────────────────

    def check_grads(
        self,
        check_for_nan_or_inf: bool = True,
        check_for_large: bool = False,
    ) -> None:
        """
        Validate gradient norms prior to the collective to catch NaN/Inf
        corruption before it propagates across ranks.

        Args:
            check_for_nan_or_inf: Abort if NaN or Inf detected in any bucket.
            check_for_large: Warn (non-fatal) on unexpectedly large norms.
        """
        for i, bucket in enumerate(self.buckets):
            grad_norm = bucket.grad_data.norm(p=2)
            if check_for_nan_or_inf:
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    raise RuntimeError(
                        f"NaN/Inf detected in local grad norm for bucket #{i} "
                        f"before data-parallel communication collective"
                    )
            if check_for_large and grad_norm.item() > 1e4:
                logger.warning(
                    f"Unexpectedly large grad norm ({grad_norm.item():.2f}) "
                    f"in bucket #{i}"
                )

    # ════════════════════════════════════════════════════════════════════════
    # PARAM ALL-GATHER (distributed optimizer only)
    # ════════════════════════════════════════════════════════════════════════

    def start_param_sync(self, force_sync: bool = False) -> None:
        """
        Dispatch all-gather to reconstruct full parameter tensors from
        the local optimizer shard.

        When ``overlap_param_gather`` is enabled, the collective runs
        asynchronously so forward compute on earlier layers can proceed
        while later layers' parameters are still being gathered.

        Args:
            force_sync: Force synchronous execution regardless of config.

        Complexity: O(N / dp_world_size) communication per rank.
        """
        assert getattr(self.ddp_config, "use_distributed_optimizer", False), (
            "start_param_sync requires use_distributed_optimizer=True"
        )

        if force_sync:
            if self.param_gather_handle is not None:
                self.param_gather_handle.wait()
                self.param_gather_handle = None
                return
        else:
            assert self.param_gather_handle is None

        async_op = (
            getattr(self.ddp_config, "overlap_param_gather", False)
            and not force_sync
        )

        # Coalesce all-gather kernels across buckets in this group
        with torch.distributed._coalescing_manager(
            self.intra_distributed_optimizer_instance_group, async_ops=async_op
        ) as cm:
            for idx, bucket in enumerate(self.buckets):
                if self.cached_param_buffer_shard_list[idx] is None:
                    self.cached_param_buffer_shard_list[idx] = shard_buffer(
                        bucket.param_data,
                        self.intra_distributed_optimizer_instance_size,
                    )
                local_view = self.cached_param_buffer_shard_list[idx][
                    self.intra_distributed_optimizer_instance_rank
                ]
                _dist_all_gather(
                    bucket.param_data,
                    local_view,
                    group=self.intra_distributed_optimizer_instance_group,
                    async_op=async_op,
                )

        if async_op:
            self.param_gather_handle = cm
        else:
            # Synchronous: coalescing_manager returns non-None even for sync
            # ops; normalise to None for consistency.
            self.param_gather_handle = None
        self.param_gather_dispatched = True

    def finish_param_sync(self, skip_next_bucket_dispatch: bool = False) -> None:
        """
        Wait for the outstanding param all-gather to complete, then dispatch
        the *next* bucket group's all-gather (pipelining with forward compute).

        Args:
            skip_next_bucket_dispatch: If True, do NOT auto-dispatch the next
                bucket group (useful for the last bucket in the chain).

        Raises:
            AssertionError: If called without ``use_distributed_optimizer`` and
                ``overlap_param_gather`` both enabled.
        """
        assert getattr(self.ddp_config, "use_distributed_optimizer", False)
        assert getattr(self.ddp_config, "overlap_param_gather", False)

        # Dispatch if not yet done (e.g. first AG bucket in first model chunk)
        if not self.param_gather_dispatched:
            self.start_param_sync()

        if self.param_gather_handle is not None:
            self.param_gather_handle.wait()
            self.param_gather_handle = None

            # Pipeline: dispatch next bucket group's AG immediately
            if (
                self.next_param_gather_bucket_group is not None
                and not skip_next_bucket_dispatch
            ):
                if self.next_param_gather_bucket_group.param_gather_dispatched:
                    warnings.warn(
                        "Next bucket group's param all-gather already dispatched.  "
                        "This may indicate a mismatch between parameter registration "
                        "order and forward-pass execution order, hurting overlap."
                    )
                else:
                    self.next_param_gather_bucket_group.start_param_sync()

    # ════════════════════════════════════════════════════════════════════════
    # GRADIENT REDUCE-SCATTER / ALL-REDUCE
    # ════════════════════════════════════════════════════════════════════════

    def start_grad_sync(self, force_all_reduce: bool = False) -> None:
        """
        Launch the data-parallel gradient collective.

        Behaviour depends on configuration:
          • **Distributed optimizer** → reduce-scatter (each rank keeps its shard).
          • **Standard DDP** → all-reduce (every rank has full gradients).
          • ``average_in_collective`` → use ReduceOp.AVG in NCCL to fuse
            the division into the collective.
          • ``overlap_grad_reduce`` → async dispatch (waited in
            ``finish_grad_sync``).

        Args:
            force_all_reduce: Override reduce-scatter with all-reduce
                (needed for special grads like shared embeddings).

        Complexity: O(N / dp_world_size) communication per rank.
        """
        if self.is_first_batch and self.grad_reduce_handle is not None:
            # No-op if first batch and collective already dispatched
            return

        assert self.grad_reduce_handle is None, (
            "Multiple outstanding grad-sync calls — previous collective not finished"
        )

        # ── Optional gradient health check ────────────────────────────────
        if getattr(self.ddp_config, "check_for_nan_in_grad", False):
            self.check_grads(check_for_nan_or_inf=True, check_for_large=False)

        # ── Pre-scale gradients ───────────────────────────────────────────
        for bucket in self.buckets:
            if bucket.gradient_scaling_factor != 1.0:
                bucket.grad_data *= bucket.gradient_scaling_factor

        # ── Decide reduce operation ───────────────────────────────────────
        reduce_op = dist.ReduceOp.SUM
        if getattr(self.ddp_config, "average_in_collective", False):
            reduce_op = dist.ReduceOp.AVG

        async_op = getattr(self.ddp_config, "overlap_grad_reduce", False)

        # Determine communication group
        use_dist_opt = getattr(self.ddp_config, "use_distributed_optimizer", False)
        if use_dist_opt:
            comm_group = self.intra_distributed_optimizer_instance_group
        else:
            comm_group = self.data_parallel_group

        # ── Coalesced collective dispatch ─────────────────────────────────
        grad_reduce_handle = None
        with torch.distributed._coalescing_manager(
            comm_group, async_ops=async_op
        ) as cm:
            for idx, bucket in enumerate(self.buckets):
                if use_dist_opt and not force_all_reduce:
                    # Reduce-scatter: each rank accumulates its shard
                    if self.cached_grad_buffer_shard_list[idx] is None:
                        self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                            bucket.grad_data,
                            self.intra_distributed_optimizer_instance_size,
                        )
                    local_view = self.cached_grad_buffer_shard_list[idx][
                        self.intra_distributed_optimizer_instance_rank
                    ]
                    grad_reduce_handle = _dist_reduce_scatter(
                        local_view,
                        bucket.grad_data,
                        op=reduce_op,
                        group=comm_group,
                        async_op=async_op,
                    )
                else:
                    # Standard all-reduce
                    dist.all_reduce(
                        bucket.grad_data,
                        op=reduce_op,
                        group=comm_group,
                        async_op=async_op,
                    )

        if async_op:
            rs_fp32 = getattr(
                self.ddp_config, "reduce_scatter_with_fp32_accumulation", False
            )
            if rs_fp32 and not force_all_reduce:
                # FP32 accumulation RS returns its own handle; coalescing
                # manager doesn't call our custom .wait() correctly.
                assert len(self.buckets) == 1, (
                    "Only 1 bucket supported with "
                    "reduce_scatter_with_fp32_accumulation=True"
                )
                assert grad_reduce_handle is not None
                self.grad_reduce_handle = grad_reduce_handle
            else:
                self.grad_reduce_handle = cm
        else:
            self.grad_reduce_handle = None

    def finish_grad_sync(self, force_all_reduce: bool = False) -> None:
        """
        Block until the gradient collective completes.

        If ``overlap_grad_reduce`` is False, this method issues a *synchronous*
        collective (start + finish in one call).

        Args:
            force_all_reduce: Pass through to ``start_grad_sync``.
        """
        self.param_gather_dispatched = False

        if not getattr(self.ddp_config, "overlap_grad_reduce", False):
            # Synchronous path
            self.start_grad_sync(force_all_reduce=force_all_reduce)
            return

        if self.is_first_batch:
            # First batch: golden counts not yet available, launch now
            self.start_grad_sync(force_all_reduce=force_all_reduce)

        assert self.grad_reduce_handle is not None, (
            f"Grad-sync communication not dispatched "
            f"({len(self.per_param_grad_ready_counts)}/{len(self.params)} "
            f"params have grad available)"
        )
        self.grad_reduce_handle.wait()
        self.grad_reduce_handle = None

    # ── Per-param grad-ready registration ─────────────────────────────────

    def register_grad_ready(
        self,
        param: torch.nn.Parameter,
        force_all_reduce: bool = False,
    ) -> None:
        """
        Register that *param*'s gradient is ready for synchronisation.

        Called from the backward post-hook.  When all params in this group
        have reported (matching the golden reference counts), the collective
        is automatically dispatched.

        Args:
            param: Parameter whose grad just became ready.
            force_all_reduce: Override reduce-scatter with all-reduce.
        """
        assert getattr(self.ddp_config, "overlap_grad_reduce", False), (
            "register_grad_ready() requires overlap_grad_reduce=True"
        )
        if self.is_last_microbatch:
            assert param in self.param_to_bucket, "Param not in this bucket group"
            if param not in self.per_param_grad_ready_counts:
                self.per_param_grad_ready_counts[param] = 0
            self.per_param_grad_ready_counts[param] += 1

            # Dispatch collective as soon as counts match the golden reference
            if not self.is_first_batch:
                if (
                    self.per_param_grad_ready_counts
                    == self.golden_per_param_grad_ready_counts
                ):
                    assert len(self.per_param_grad_ready_counts) == len(self.params)
                    self.start_grad_sync(force_all_reduce=force_all_reduce)


# ════════════════════════════════════════════════════════════════════════════════
# PARAM AND GRAD BUFFER — CONTIGUOUS MEMORY ALLOCATION
# ════════════════════════════════════════════════════════════════════════════════

class ParamAndGradBuffer:
    """
    Allocates a single contiguous GPU tensor and maps every parameter's
    ``.data`` and ``.main_grad`` into non-overlapping views within it.

    The buffer is subdivided into *buckets* (roughly ``bucket_size`` elements
    each).  Each bucket is the communication granularity — its ``grad_data``
    view is the tensor handed to NCCL.

    Key design decisions:
      • Parameters are walked in **reverse** registration order so that
        the first bucket corresponds to the *last* model layers.  This means
        backward fills buckets front-to-back, enabling the earliest buckets
        to launch their collective while later layers are still computing.
      • Shared-embedding parameters are forced into their own bucket so that
        the distributed optimizer partitions their state identically across
        all pipeline stages.
      • Bucket ends are padded to ``lcm(dp_world_size, 128)`` (or
        ``lcm(dp_world_size, 128, 2^16)`` with ``pad_buckets_for_high_nccl_busbw``)
        to guarantee 256-byte-aligned shards and high NCCL bus bandwidth.

    Args:
        ddp_config: ``MegatronDDPConfig`` instance.
        param_dtype: Storage dtype for parameters.
        grad_dtype: Storage dtype for gradients.
        params: Ordered list of model parameters.
        data_parallel_group: DP process group.
        bucket_size: Approximate number of elements per bucket.
        param_to_name: Param → name mapping for logging.
        gradient_scaling_factor: Pre-scale applied before the collective.
    """

    __slots__ = (
        "ddp_config",
        "params",
        "param_dtype",
        "grad_dtype",
        "data_parallel_group",
        "data_parallel_world_size",
        "gradient_scaling_factor",
        # Data storage
        "param_data",
        "grad_data",
        "numel",
        "numel_unpadded",
        # Buckets
        "buckets",
        "bucket_indices",
        "param_to_bucket",
        "param_index_map",
        # CPU offload state
        "_grad_data_size",
        "_param_data_size",
        "_param_data_cpu",
    )

    def __init__(
        self,
        ddp_config: Any,
        param_dtype: torch.dtype,
        grad_dtype: torch.dtype,
        params: List[torch.nn.Parameter],
        data_parallel_group: dist.ProcessGroup,
        bucket_size: int,
        param_to_name: Dict[torch.nn.Parameter, str],
        gradient_scaling_factor: float,
    ) -> None:
        self.ddp_config = ddp_config
        self.params = params
        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size: int = data_parallel_group.size()
        self.gradient_scaling_factor = gradient_scaling_factor

        # Validate uniqueness
        assert len(set(params)) == len(params), "Duplicate parameters detected"

        # ── Data structures ───────────────────────────────────────────────
        self.buckets: List[ParamAndGradBucket] = []
        self.param_to_bucket: Dict[torch.nn.Parameter, ParamAndGradBucket] = {}
        self.param_index_map: Dict[
            torch.nn.Parameter, Tuple[int, int, int]
        ] = {}  # param → (start, end, bucket_id)

        use_dist_opt = getattr(ddp_config, "use_distributed_optimizer", False)
        pad_high_busbw = getattr(ddp_config, "pad_buckets_for_high_nccl_busbw", False)

        # ── Helper: pad to divisor ────────────────────────────────────────
        def _pad(n: int, divisor: int) -> int:
            return int(math.ceil(n / divisor) * divisor)

        def _pad_bucket_end(end: int) -> int:
            if use_dist_opt:
                if pad_high_busbw:
                    divisor = math.lcm(
                        self.data_parallel_world_size,
                        _BUCKET_ALIGNMENT_ELEMS,
                        _HIGH_BUSBW_POWER,
                    )
                else:
                    divisor = math.lcm(
                        self.data_parallel_world_size, _BUCKET_ALIGNMENT_ELEMS
                    )
                return _pad(end, divisor)
            return end

        def _pad_param_start(start: int) -> int:
            if use_dist_opt:
                return _pad(start, _PARAM_ALIGNMENT_ELEMS)
            return start

        def _needs_own_bucket(param: torch.nn.Parameter) -> bool:
            """Shared embedding params need separate buckets for dist-opt."""
            return getattr(param, "shared_embedding", False) and use_dist_opt

        # ════════════════════════════════════════════════════════════════════
        # Phase 1: Compute bucket boundaries (reverse param order)
        # ════════════════════════════════════════════════════════════════════
        param_start_index = 0
        bucket_start_index = 0
        bucket_params: Set[torch.nn.Parameter] = set()
        self.bucket_indices: List[Tuple[int, int]] = []
        per_bucket_numel_unpadded: List[int] = []
        bucket_id = 0

        def _record_bucket(param_end: int) -> int:
            nonlocal bucket_start_index, bucket_params, bucket_id
            per_bucket_numel_unpadded.append(param_end - bucket_start_index)
            bucket_end = _pad_bucket_end(param_end)
            self.bucket_indices.append((bucket_start_index, bucket_end))
            bucket_start_index = bucket_end
            bucket_params = set()
            bucket_id += 1
            return bucket_end

        for param in params[::-1]:
            this_numel = param.data.nelement()
            param_start_index = _pad_param_start(param_start_index)

            # Force new bucket before this param if it needs isolation
            if _needs_own_bucket(param) and len(bucket_params) > 0:
                param_start_index = _record_bucket(param_start_index)

            param_end_index = param_start_index + this_numel
            self.param_index_map[param] = (
                param_start_index,
                param_end_index,
                bucket_id,
            )
            bucket_params.add(param)

            # Close bucket when size threshold reached or param needs isolation
            if (
                bucket_size is not None
                and (param_end_index - bucket_start_index) >= bucket_size
            ) or _needs_own_bucket(param):
                bucket_end = _record_bucket(param_end_index)
                param_start_index = bucket_end
            else:
                param_start_index = param_end_index

        # Final bucket for remaining params
        if len(bucket_params) > 0:
            bucket_end = _record_bucket(param_end_index)

        # ════════════════════════════════════════════════════════════════════
        # Phase 2: Allocate contiguous storage
        # ════════════════════════════════════════════════════════════════════
        self.numel: int = bucket_end
        self.numel_unpadded: int = sum(per_bucket_numel_unpadded)
        assert self.numel_unpadded <= self.numel

        if use_dist_opt:
            assert self.numel % self.data_parallel_world_size == 0, (
                f"Total buffer numel ({self.numel}) must be divisible by "
                f"dp_world_size ({self.data_parallel_world_size})"
            )

        self.param_data: Optional[torch.Tensor] = None
        if use_dist_opt:
            self.param_data = torch.zeros(
                self.numel,
                dtype=param_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        self.grad_data: torch.Tensor = torch.zeros(
            self.numel,
            dtype=grad_dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

        # CPU offload state
        self._grad_data_size: int = 0
        self._param_data_size: int = 0
        self._param_data_cpu: Optional[torch.Tensor] = None

        # ════════════════════════════════════════════════════════════════════
        # Phase 3: Map param.data and param.main_grad into buffer views
        # ════════════════════════════════════════════════════════════════════
        bucket_params_list: List[torch.nn.Parameter] = []
        bucket_start = 0
        cur_bucket_id = 0

        for param in params[::-1]:
            p_start, p_end, b_id = self.param_index_map[param]

            # Remap param.data into contiguous buffer
            if self.param_data is not None:
                new_data = self._get(param.data.shape, p_start, BufferType.PARAM)
                old_data = param.data
                param.data = new_data
                param.data.detach().copy_(old_data)
                del old_data

            # Assign main_grad to contiguous buffer view
            param.main_grad = self._get(  # type: ignore[attr-defined]
                param.data.shape, p_start, BufferType.GRAD
            )

            # Bucket boundary transition
            if b_id != cur_bucket_id:
                b_end = _pad_bucket_end(p_start)
                self.buckets.append(
                    self._make_bucket(
                        bucket_params_list,
                        bucket_start,
                        b_end,
                        per_bucket_numel_unpadded[cur_bucket_id],
                        cur_bucket_id,
                    )
                )
                bucket_start = b_end
                bucket_params_list = []
                cur_bucket_id = b_id
            bucket_params_list.append(param)

        # Final bucket
        if bucket_params_list:
            b_end = _pad_bucket_end(p_end)
            self.buckets.append(
                self._make_bucket(
                    bucket_params_list,
                    bucket_start,
                    b_end,
                    per_bucket_numel_unpadded[cur_bucket_id],
                    cur_bucket_id,
                )
            )

        # ── Logging ───────────────────────────────────────────────────────
        log_lines = [
            f"ParamAndGradBuffer: {len(self.buckets)} bucket(s), "
            f"{self.numel:,} total elements ({self.numel_unpadded:,} unpadded)"
        ]
        for i, bucket in enumerate(self.buckets):
            numel = sum(p.numel() for p in bucket.params)
            log_lines.append(
                f"  Bucket {i}: {numel:,} param elements, "
                f"{bucket.grad_data.nelement():,} padded size"
            )
            for p in bucket.params:
                log_lines.append(f"    {param_to_name.get(p, '<unknown>')}")
        logger.info("\n".join(log_lines))

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get(
        self,
        shape: torch.Size,
        start_index: int,
        buffer_type: BufferType,
    ) -> torch.Tensor:
        """Return a view with *shape* starting at *start_index*."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, "Requested tensor out of buffer range"
        if buffer_type == BufferType.PARAM:
            assert self.param_data is not None
            return self.param_data[start_index:end_index].view(shape)
        elif buffer_type == BufferType.GRAD:
            return self.grad_data[start_index:end_index].view(shape)
        raise ValueError(f"Unknown buffer type: {buffer_type}")

    def _make_bucket(
        self,
        bucket_params: List[torch.nn.Parameter],
        start: int,
        end: int,
        numel_unpadded: int,
        bucket_id: int,
    ) -> ParamAndGradBucket:
        """Create a new bucket and update param → bucket mapping."""
        use_dist_opt = getattr(self.ddp_config, "use_distributed_optimizer", False)
        if use_dist_opt:
            assert start % self.data_parallel_world_size == 0
            assert end % self.data_parallel_world_size == 0
        assert (start, end) == self.bucket_indices[bucket_id]

        p_data = None
        if self.param_data is not None:
            p_data = self._get(
                torch.Size([end - start]), start, BufferType.PARAM
            )
        g_data = self._get(
            torch.Size([end - start]), start, BufferType.GRAD
        )

        bucket = ParamAndGradBucket(
            params=bucket_params,
            param_data=p_data,
            grad_data=g_data,
            offset=start,
            numel_unpadded=numel_unpadded,
            gradient_scaling_factor=self.gradient_scaling_factor,
            bucket_id=bucket_id,
        )
        for p in bucket_params:
            assert p not in self.param_to_bucket
            self.param_to_bucket[p] = bucket
        return bucket

    # ── Public API ────────────────────────────────────────────────────────

    def scale_gradients(self, factor: float) -> None:
        """Scale entire gradient buffer by *factor*."""
        self.grad_data *= factor

    def reset(self) -> None:
        """Zero-fill the gradient buffer for the next accumulation cycle."""
        self.grad_data.zero_()

    # ── CPU Offload / Reload ──────────────────────────────────────────────

    def offload_to_cpu(
        self,
        move_params: bool = True,
        move_grads: bool = True,
    ) -> None:
        """
        Release GPU memory by shrinking storage to zero and (optionally)
        staging param data to pinned CPU memory.

        This is used between pipeline stages to free HBM for activations.

        Args:
            move_params: Offload parameter buffer to CPU.
            move_grads: Release gradient buffer storage.
        """
        if move_grads and self.grad_data is not None:
            if self.grad_data.storage().size() > 0:
                self._grad_data_size = self.grad_data.storage().size()
                self.grad_data.storage().resize_(0)

        if move_params and self.param_data is not None:
            if self.param_data.storage().size() > 0:
                self._param_data_size = self.param_data.storage().size()
                if self._param_data_cpu is not None:
                    self._param_data_cpu.copy_(self.param_data, non_blocking=True)
                else:
                    self._param_data_cpu = self.param_data.cpu().pin_memory()
                self.param_data.storage().resize_(0)

    def reload_from_cpu(
        self,
        move_params: bool = True,
        move_grads: bool = True,
    ) -> None:
        """
        Restore buffers from CPU / re-allocate GPU storage.

        Args:
            move_params: Reload parameter buffer from pinned CPU copy.
            move_grads: Re-allocate gradient buffer and zero-fill.
        """
        if (
            move_params
            and self.param_data is not None
            and self._param_data_cpu is not None
            and self.param_data.storage().size() == 0
        ):
            self.param_data.storage().resize_(self._param_data_size)
            self.param_data.copy_(self._param_data_cpu, non_blocking=True)

        if (
            move_grads
            and self.grad_data is not None
            and self._grad_data_size > 0
        ):
            self.grad_data.storage().resize_(self._grad_data_size)
            self.grad_data.zero_()
            self._grad_data_size = 0


# ════════════════════════════════════════════════════════════════════════════════
# BUCKET PARTITIONING — AUTOMATIC GROUPING STRATEGY
# ════════════════════════════════════════════════════════════════════════════════

def partition_buckets(
    buffers: List[ParamAndGradBuffer],
    force_single_bucket_group: bool = False,
) -> List[ParamAndGradBucketGroup]:
    """
    Regroup buckets from one or more ``ParamAndGradBuffer`` instances into
    ``ParamAndGradBucketGroup`` objects whose collectives are coalesced.

    Grouping strategy:
      1. **force_single_bucket_group**: All buckets across all buffers in one
         group (useful for very small models).
      2. **Default**: Each bucket gets its own group (one NCCL launch per
         bucket).

    Having fewer groups reduces NCCL launch overhead but limits overlap
    granularity.  The default (one-bucket-per-group) maximises overlap.

    Args:
        buffers: List of ``ParamAndGradBuffer`` (one per dtype in the model).
        force_single_bucket_group: Merge everything into one group.

    Returns:
        Ordered list of ``ParamAndGradBucketGroup``.

    Complexity: O(B) where B = total number of buckets across all buffers.
    """
    if not buffers:
        return []

    if force_single_bucket_group:
        all_buckets: List[ParamAndGradBucket] = []
        ddp_config = buffers[0].ddp_config
        dp_group = buffers[0].data_parallel_group
        dp_size = buffers[0].data_parallel_world_size
        for buf in buffers:
            assert buf.ddp_config is ddp_config
            assert buf.data_parallel_group is dp_group
            all_buckets.extend(buf.buckets)
        return [ParamAndGradBucketGroup(all_buckets, ddp_config, dp_group, dp_size)]

    # Default: one bucket per group
    groups: List[ParamAndGradBucketGroup] = []
    for buf in buffers:
        for bucket in buf.buckets:
            groups.append(
                ParamAndGradBucketGroup(
                    [bucket],
                    buf.ddp_config,
                    buf.data_parallel_group,
                    buf.data_parallel_world_size,
                )
            )
    return groups
