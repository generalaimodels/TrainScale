# ════════════════════════════════════════════════════════════════════════════════
# REDUCE-SCATTER WITH FP32 ACCUMULATION — NUMERICAL STABILITY AT SCALE
# ════════════════════════════════════════════════════════════════════════════════
# Production-grade reduce-scatter that transmits lower-precision values over the
# wire (BF16/FP16) but performs local accumulation in FP32 to prevent numerical
# drift at large data-parallel counts.
#
# Algorithm:
#   1. All-to-all: each rank sends its shard to every other rank
#      → total bytes on wire == standard ring reduce-scatter
#   2. Local FP32 accumulation: sum received shards in float32
#   3. Downcast result back to original dtype into output_tensor
#
# Why not standard reduce-scatter?
#   NCCL's built-in reduce-scatter accumulates in the wire dtype (BF16/FP16).
#   At DP ≥ 64 with BF16 the rounding error from 64 additions becomes visible
#   in gradient norms, hurting convergence.  This implementation keeps the wire
#   traffic identical but guarantees FP32-accurate final result.
#
# Complexity:
#   Communication: O(n) total bytes (same as ring RS)
#   Compute:       O(n/p) local FP32 reduction per rank
#   Memory:        O(n) temporary for all-to-all output buffer
#
# Reference: Megatron-LM core/distributed/reduce_scatter_with_fp32_accumulation
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.distributed as dist


class _ReduceScatterFP32WorkHandle:
    """
    Async work handle for reduce_scatter_with_fp32_accumulation.

    When the caller invokes .wait(), this handle:
      1. Waits for the all-to-all communication to complete.
      2. Reshapes the received tensor into (world_size, shard_size).
      3. Performs FP32 summation across the first dimension.
      4. Copies the downcasted result into the user's output_tensor.

    This two-phase design lets the communication overlap with other compute
    while the expensive FP32 reduction is deferred to the synchronisation point.
    """

    __slots__ = (
        "_all_to_all_handle",
        "_all_to_all_output",
        "_output_tensor",
        "_world_size",
    )

    def __init__(
        self,
        all_to_all_handle: Any,
        all_to_all_output: torch.Tensor,
        output_tensor: torch.Tensor,
        world_size: int,
    ) -> None:
        self._all_to_all_handle = all_to_all_handle
        self._all_to_all_output = all_to_all_output
        self._output_tensor = output_tensor
        self._world_size = world_size

    # ── Public API ────────────────────────────────────────────────────────────

    def wait(self) -> None:
        """Block until communication completes, then reduce in FP32."""
        # 1. Drain the NCCL stream
        if self._all_to_all_handle is not None:
            self._all_to_all_handle.wait()

        # 2. Reshape: (total_elements,) → (world_size, shard_size)
        #    and accumulate in float32 for numerical stability
        fp32_sum = torch.sum(
            self._all_to_all_output.view(self._world_size, -1),
            dim=0,
            dtype=torch.float32,
        )
        assert fp32_sum.dtype == torch.float32

        # 3. Downcast into caller's output tensor (in-place)
        self._output_tensor.copy_(fp32_sum)


def reduce_scatter_with_fp32_accumulation(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[_ReduceScatterFP32WorkHandle]:
    """
    Reduce-scatter with FP32 accumulation for numerical stability.

    Collects ``input_tensor`` shards from every rank via all-to-all (keeping
    the wire in the tensor's native dtype), then accumulates locally in FP32,
    then downcasts the result into ``output_tensor``.

    Args:
        output_tensor: Pre-allocated tensor to receive the reduced shard.
                       Shape must equal input_tensor.numel() // world_size.
        input_tensor:  Full gradient buffer to be reduce-scattered.
                       numel() must be divisible by world_size.
        op:            Only ``ReduceOp.SUM`` is supported.
        group:         Process group (``None`` → default world group).
        async_op:      If True, return a work handle whose ``.wait()``
                       performs the FP32 reduction.  If False, block here.

    Returns:
        ``_ReduceScatterFP32WorkHandle`` when ``async_op=True``,
        ``None`` otherwise.

    Raises:
        AssertionError: If ``op`` is not SUM or tensor sizes are invalid.
    """
    # ── Pre-condition checks ──────────────────────────────────────────────────
    assert op == dist.ReduceOp.SUM, (
        "reduce_scatter_with_fp32_accumulation only supports ReduceOp.SUM"
    )

    world_size = group.size() if group is not None else dist.get_world_size()
    assert input_tensor.numel() % world_size == 0, (
        f"input_tensor.numel()={input_tensor.numel()} must be divisible "
        f"by world_size={world_size}"
    )

    # ── All-to-all: collect shards from every rank ────────────────────────────
    # After this call each rank holds *all* shards that belong to its local
    # reduce-scatter output, arranged as (world_size, shard_size).
    all_to_all_output = torch.empty_like(input_tensor)
    handle = dist.all_to_all_single(
        output=all_to_all_output,
        input=input_tensor,
        group=group,
        async_op=async_op,
    )

    # ── Build work handle ─────────────────────────────────────────────────────
    work = _ReduceScatterFP32WorkHandle(
        all_to_all_handle=handle,
        all_to_all_output=all_to_all_output,
        output_tensor=output_tensor,
        world_size=world_size,
    )

    if async_op:
        return work

    # Synchronous path: reduce immediately
    work.wait()
    return None
