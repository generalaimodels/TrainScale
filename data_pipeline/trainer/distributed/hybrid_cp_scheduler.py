# ════════════════════════════════════════════════════════════════════════════════
# HYBRID CONTEXT-PARALLEL SCHEDULER — BALANCED WORKLOAD DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════════
# Production-grade scheduler for variable-length sequences in the DPxCP domain.
# Assigns sub-samples to GPU groups with roughly balanced workload, maximising
# utilization across all DP and CP ranks.
#
# Algorithm overview:
#   1. Estimate per-sequence workload: O(L²/CP) for attention-dominated models
#   2. Bucket sequences by required CP size (power-of-2 alignment)
#   3. Greedily assign sequences to GPU groups, balancing total workload
#   4. Redistribute idle GPUs by expanding CP size of assigned sequences
#   5. Trim overloaded groups to improve global balance
#
# Key features:
#   • Dynamic CP sizing based on sequence length and max_seq_len_per_rank
#   • Balance slack (δ) tolerance for early batch completion
#   • Empty GPU reclamation via recursive CP expansion
#   • Compatible with pipeline parallelism and virtual pipeline stages
#
# Complexity:
#   Scheduling: O(S log S) where S = number of sub-samples (sorting + heap ops)
#   Memory:     O(S) for bucket storage
#
# Reference: Megatron-LM core/pipeline_parallel/hybrid_cp_schedule
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from math import ceil, log2
from typing import (
    Callable,
    Deque,
    Dict,
    Final,
    List,
    Optional,
    Set,
    Tuple,
    TypeAlias,
)

import torch
import torch.distributed as dist

logger = logging.getLogger("sota_ddp.hybrid_cp_scheduler")

# ════════════════════════════════════════════════════════════════════════════════
# TYPE ALIASES
# ════════════════════════════════════════════════════════════════════════════════

SampleSeqLen: TypeAlias = Tuple[int, int]  # (sample_id, sequence_length)
MicroBatch: TypeAlias = List[int]           # List of sequence lengths per GPU
SampleIdList: TypeAlias = List[int]         # Sample IDs assigned to a GPU


# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HybridCPConfig:
    """Configuration for hybrid context-parallel scheduling.
    
    Attributes:
        max_seq_len_per_rank: Maximum sequence length each CP rank can handle.
            Sequences longer than this are split across multiple CP ranks.
        balance_slack: Acceptable imbalance ratio (e.g., 0.05 = 5% slack).
            Batches are considered balanced when max-min workload < δ * max.
        bucket_balance_eps: Target ε for bucket workload balance.
        strategy: Scheduling strategy ("dp" for data-parallel priority,
            "pp" for pipeline-parallel priority).
    """
    max_seq_len_per_rank: int = 8192
    balance_slack: float = 0.05
    bucket_balance_eps: float = 0.10
    strategy: str = "dp"  # "dp" or "pp"
    
    def __post_init__(self) -> None:
        assert self.max_seq_len_per_rank > 0, "max_seq_len_per_rank must be positive"
        assert 0.0 <= self.balance_slack <= 1.0, "balance_slack must be in [0, 1]"
        assert self.strategy in ("dp", "pp"), "strategy must be 'dp' or 'pp'"


# ════════════════════════════════════════════════════════════════════════════════
# BALANCED CP SCHEDULER
# ════════════════════════════════════════════════════════════════════════════════

class BalancedCPScheduler:
    """
    Scheduler for forming balanced groups of sub-samples across the DPxCP domain.
    
    Given a batch of variable-length sequences, this scheduler assigns each
    sequence to a subset of GPUs such that:
      1. Each sequence's CP size is power-of-2 aligned
      2. All GPUs have roughly equal total workload
      3. Communication barriers are minimised
    
    The scheduler operates in two phases:
      1. **Bucket formation**: Group sequences by required CP size
      2. **Greedy assignment**: Assign sequences to GPU groups, balancing load
    
    Example usage:
        scheduler = BalancedCPScheduler(
            max_seq_len_per_rank=4096,
            dp_cp_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
        )
        groups, sample_id_groups = scheduler.get_groups_and_subsamples(
            sample_id_seqlens=[(0, 2048), (1, 8192), (2, 1024)],
            config=config,
        )
    """
    
    __slots__ = (
        "max_seq_len_per_rank",
        "total_hdp_gpus",
        "num_subsamples",
        "num_subsamples_processed",
        "free_resources",
    )
    
    def __init__(
        self,
        max_seq_len_per_rank: int,
        dp_cp_group: dist.ProcessGroup,
    ) -> None:
        """
        Initialize the balanced CP scheduler.
        
        Args:
            max_seq_len_per_rank: Maximum sequence length per CP rank.
            dp_cp_group: Process group spanning DP × CP ranks.
        """
        self.max_seq_len_per_rank = max_seq_len_per_rank
        self.total_hdp_gpus = dp_cp_group.size()
        
        # Iteration-local bookkeeping
        self.num_subsamples = 0
        self.num_subsamples_processed = 0
        self.free_resources: List[int] = []
    
    # ── Workload estimation ────────────────────────────────────────────────
    
    @lru_cache(maxsize=256)
    def get_total_workload(
        self,
        seq_length: int,
        cp_size: Optional[int] = None,
    ) -> float:
        """
        Estimate the relative workload for a sub-sample.
        
        For attention-dominated transformer models, the workload scales as
        O(L²/CP) where L is sequence length and CP is context-parallel size.
        This estimate is used for load balancing, not accurate FLOPs calculation.
        
        Args:
            seq_length: Sequence length of the sub-sample.
            cp_size: Number of CP ranks sharing this sub-sample.
                If None, computed from gpus_needed().
        
        Returns:
            Estimated relative workload (dimensionless).
        
        Complexity: O(1) with caching.
        """
        if cp_size is None:
            cp_size = self.gpus_needed(seq_length)
        return (seq_length * seq_length) / cp_size
    
    @lru_cache(maxsize=256)
    def gpus_needed(self, seq_len: int) -> int:
        """
        Compute the number of GPUs (CP size) required for a sequence.
        
        The result is rounded up to the next power of 2 to match the
        available hybrid context-parallel process group sizes.
        
        Args:
            seq_len: Sequence length.
        
        Returns:
            Number of GPUs needed (power of 2).
        
        Complexity: O(1).
        """
        if seq_len <= self.max_seq_len_per_rank:
            return 1
        return max(1, 2 ** ceil(log2(seq_len / self.max_seq_len_per_rank)))
    
    # ── Bucket formation ───────────────────────────────────────────────────
    
    def make_buckets_equal(
        self,
        sample_seqlens: List[SampleSeqLen],
        compute_estimator: Callable[[int, Optional[int]], float],
    ) -> List[Deque[SampleSeqLen]]:
        """
        Create buckets of sequences with roughly equal total workload.
        
        The number of buckets equals the number of unique CP sizes needed.
        Each bucket contains sequences that can be processed together.
        
        Args:
            sample_seqlens: List of (sample_id, sequence_length) tuples.
            compute_estimator: Function (seq_len, cp_size) → workload.
        
        Returns:
            List of deques, each containing (sample_id, seq_len) tuples.
        
        Complexity: O(S) where S = number of samples.
        """
        if not sample_seqlens:
            return []
        
        # Determine k = number of unique CP sizes
        unique_cp_sizes = {self.gpus_needed(seq_len) for _, seq_len in sample_seqlens}
        k = len(unique_cp_sizes)
        
        # Compute total workload
        work_per_sample = []
        total_work = 0.0
        for _, seq_len in sample_seqlens:
            cp_size = self.gpus_needed(seq_len)
            w = compute_estimator(seq_len, cp_size)
            work_per_sample.append(w)
            total_work += w
        
        target = total_work / k
        
        # Greedy bucketing
        buckets: List[Deque[SampleSeqLen]] = []
        current_bucket: List[SampleSeqLen] = []
        current_work = 0.0
        remaining_buckets = k
        
        for i, (sample_id, seq_len) in enumerate(sample_seqlens):
            work = work_per_sample[i]
            projected = current_work + work
            
            # Close bucket if it's full or we need to save samples for remaining buckets
            remaining_samples = len(sample_seqlens) - i
            if current_bucket and (
                projected > target * 1.1 or remaining_samples <= remaining_buckets - len(buckets)
            ):
                buckets.append(deque(current_bucket))
                current_bucket = []
                current_work = 0.0
            
            current_bucket.append((sample_id, seq_len))
            current_work += work
        
        if current_bucket:
            buckets.append(deque(current_bucket))
        
        return buckets
    
    # ── Main scheduling algorithm ──────────────────────────────────────────
    
    def next_hdp_group(
        self,
        sample_seqlens: List[SampleSeqLen],
        compute_estimator: Callable[[int, Optional[int]], float],
        total_gpus: int,
        delta: float = 0.05,
        strategy: str = "dp",
    ) -> Tuple[
        List[MicroBatch],
        List[SampleSeqLen],
        List[float],
        List[SampleIdList],
    ]:
        """
        Form a balanced micro-batch assignment for the DPxCP domain.
        
        This is the core scheduling algorithm that assigns sequences to GPUs
        while maintaining workload balance across all ranks.
        
        Algorithm:
          1. Create workload-balanced buckets of sequences
          2. For each sequence, find the best GPU group:
             - Existing group of matching CP size with lowest load, OR
             - New group from free GPUs
          3. Check balance after CP size transitions
          4. Trim overloaded groups to improve balance
          5. Fill empty GPUs via CP expansion
        
        Args:
            sample_seqlens: List of (sample_id, sequence_length) tuples.
            compute_estimator: Workload estimation function.
            total_gpus: Total number of GPUs in the DPxCP domain.
            delta: Balance slack tolerance (default 5%).
            strategy: "dp" for data-parallel, "pp" for pipeline-parallel priority.
        
        Returns:
            Tuple of:
              - micro_batches: List of sequence lengths per GPU
              - leftovers: Unassigned (sample_id, seq_len) tuples
              - exec_times: Estimated execution time per GPU
              - sample_ids_per_gpu: Sample IDs assigned to each GPU
        
        Complexity: O(S log S) for S samples.
        """
        if not sample_seqlens:
            return (
                [[] for _ in range(total_gpus)],
                [],
                [0.0 for _ in range(total_gpus)],
                [[] for _ in range(total_gpus)],
            )
        
        buckets = self.make_buckets_equal(sample_seqlens, compute_estimator)
        
        # Initialize tracking structures
        micro_batches: List[MicroBatch] = [[] for _ in range(total_gpus)]
        exec_times: List[float] = [0.0 for _ in range(total_gpus)]
        sample_ids_per_gpu: List[SampleIdList] = [[] for _ in range(total_gpus)]
        
        gpu_group_id: List[Optional[int]] = [None] * total_gpus
        group_members: Dict[int, List[int]] = {}
        group_size: Dict[int, int] = {}
        next_gid = 0
        
        pp_cursor = 0
        prev_needed: Optional[int] = None
        check_balance = False
        
        while buckets:
            # Step 1: Select next sequence to place
            sample_seq_tuple: Optional[SampleSeqLen] = None
            bucket_idx: Optional[int] = None
            needed: Optional[int] = None
            
            scan_order = (
                range(len(buckets))
                if strategy == "dp"
                else [(pp_cursor + i) % len(buckets) for i in range(len(buckets))]
            )
            
            for idx in scan_order:
                if not buckets[idx]:
                    continue
                cand_tuple = buckets[idx][0]
                cand_seq_len = cand_tuple[1]
                needed = self.gpus_needed(cand_seq_len)
                
                # Check for existing group or free GPUs
                candidate_gids = [gid for gid, sz in group_size.items() if sz == needed]
                free_ranks = [r for r, gid in enumerate(gpu_group_id) if gid is None]
                
                if candidate_gids or len(free_ranks) >= needed:
                    sample_seq_tuple, bucket_idx = cand_tuple, idx
                    break
            
            if sample_seq_tuple is None:
                break
            
            if strategy == "pp":
                pp_cursor = (bucket_idx + 1) % len(buckets)
            
            sample_id, seq_len = sample_seq_tuple
            needed = self.gpus_needed(seq_len)
            
            if prev_needed is None:
                prev_needed = needed
            
            # Step 2: Find best GPU group
            candidate_gids = [gid for gid, sz in group_size.items() if sz == needed]
            if candidate_gids:
                best_gid, best_load = min(
                    (
                        (gid, max(exec_times[r] for r in group_members[gid]))
                        for gid in candidate_gids
                    ),
                    key=lambda t: t[1],
                )
            else:
                best_gid, best_load = None, float("inf")
            
            free_ranks = [r for r, gid in enumerate(gpu_group_id) if gid is None]
            if len(free_ranks) >= needed:
                free_sorted = sorted(free_ranks, key=lambda r: exec_times[r])
                new_members = free_sorted[:needed]
                new_load = exec_times[new_members[-1]]
                
                if new_load < best_load:
                    best_gid = None
                    chosen_members = new_members
                else:
                    chosen_members = group_members[best_gid]
            else:
                chosen_members = group_members[best_gid]
            
            # Step 3: Create new group if needed
            if best_gid is None:
                best_gid = next_gid
                next_gid += 1
                group_members[best_gid] = chosen_members
                group_size[best_gid] = needed
                for r in chosen_members:
                    gpu_group_id[r] = best_gid
            
            # Step 4: Assign sequence to all group members
            per_gpu_cost = compute_estimator(seq_len, needed)
            for r in chosen_members:
                micro_batches[r].append(seq_len)
                exec_times[r] += per_gpu_cost
                sample_ids_per_gpu[r].append(sample_id)
            
            buckets[bucket_idx].popleft()
            
            # Clean up empty buckets
            while buckets and not buckets[0]:
                buckets.pop(0)
                pp_cursor %= max(1, len(buckets))
            
            # Check balance on CP size transitions
            if needed < prev_needed:
                check_balance = True
            
            if (
                check_balance
                and buckets
                and max(exec_times) - min(exec_times) <= delta * max(exec_times)
            ):
                break
            
            prev_needed = needed
        
        # Gather leftovers
        leftovers: List[SampleSeqLen] = []
        for b in buckets:
            leftovers.extend(b)
        
        # Trim overloaded groups
        self._trim_overloaded(
            micro_batches,
            exec_times,
            sample_ids_per_gpu,
            group_members,
            group_size,
            compute_estimator,
            delta,
            leftovers,
        )
        
        # Fill empty GPUs
        self._fill_empty_gpus(
            micro_batches,
            exec_times,
            sample_ids_per_gpu,
            group_members,
            group_size,
            gpu_group_id,
            total_gpus,
        )
        
        return micro_batches, leftovers, exec_times, sample_ids_per_gpu
    
    def _trim_overloaded(
        self,
        micro_batches: List[MicroBatch],
        exec_times: List[float],
        sample_ids_per_gpu: List[SampleIdList],
        group_members: Dict[int, List[int]],
        group_size: Dict[int, int],
        compute_estimator: Callable[[int, Optional[int]], float],
        delta: float,
        leftovers: List[SampleSeqLen],
    ) -> None:
        """Iteratively remove sequences from overloaded groups to improve balance."""
        while True:
            cur_max = max(exec_times)
            cur_min = min(exec_times)
            cur_slack = cur_max - cur_min
            
            if cur_slack <= delta * cur_max or cur_min == 0:
                break
            
            max_r = exec_times.index(cur_max)
            gid = None
            for g, members in group_members.items():
                if max_r in members:
                    gid = g
                    break
            
            if gid is None or not micro_batches[max_r] or len(micro_batches[max_r]) <= 1:
                break
            
            members = group_members[gid]
            seq = micro_batches[max_r][-1]
            need = group_size[gid]
            per_gpu_cost = compute_estimator(seq, need)
            
            proj_times = exec_times[:]
            for r in members:
                proj_times[r] -= per_gpu_cost
            
            proj_slack = max(proj_times) - min(proj_times)
            
            if proj_slack < cur_slack:
                sample_id_to_remove = sample_ids_per_gpu[max_r][-1]
                for r in members:
                    micro_batches[r].pop()
                    exec_times[r] -= per_gpu_cost
                    sample_ids_per_gpu[r].pop()
                leftovers.append((sample_id_to_remove, seq))
            else:
                break
    
    def _fill_empty_gpus(
        self,
        micro_batches: List[MicroBatch],
        exec_times: List[float],
        sample_ids_per_gpu: List[SampleIdList],
        group_members: Dict[int, List[int]],
        group_size: Dict[int, int],
        gpu_group_id: List[Optional[int]],
        total_gpus: int,
    ) -> None:
        """Redistribute work by expanding CP sizes to fill empty GPUs."""
        iteration = 0
        max_iterations = total_gpus  # Prevent infinite loops
        
        while iteration < max_iterations:
            empty_gpus = [i for i in range(total_gpus) if not micro_batches[i]]
            if not empty_gpus:
                break
            
            if not group_size:
                logger.warning(
                    f"{len(empty_gpus)} GPUs have no work assigned. "
                    "Consider increasing 'max_seq_len_per_rank' or batch size."
                )
                break
            
            # Find smallest group that can be expanded
            min_group_size = min(group_size.values())
            next_power = min(min_group_size * 2, total_gpus)
            
            expanded = False
            for gid, size in list(group_size.items()):
                if size != min_group_size:
                    continue
                
                members = group_members[gid]
                needed_count = next_power - min_group_size
                
                if len(empty_gpus) < needed_count:
                    continue
                
                # Expand the group
                new_members = members + empty_gpus[:needed_count]
                group_members[gid] = new_members
                group_size[gid] = next_power
                
                # Copy work to new members
                for new_r in empty_gpus[:needed_count]:
                    micro_batches[new_r] = micro_batches[members[0]][:]
                    sample_ids_per_gpu[new_r] = sample_ids_per_gpu[members[0]][:]
                    exec_times[new_r] = self.get_total_workload(
                        micro_batches[members[0]][0], next_power
                    ) if micro_batches[members[0]] else 0.0
                    gpu_group_id[new_r] = gid
                
                expanded = True
                break
            
            if not expanded:
                break
            
            iteration += 1
    
    # ── High-level API ─────────────────────────────────────────────────────
    
    def get_groups_and_subsamples(
        self,
        sample_id_seqlens: List[SampleSeqLen],
        config: Optional[HybridCPConfig] = None,
    ) -> Tuple[List[List[MicroBatch]], List[List[SampleIdList]]]:
        """
        Recursively form balanced groups until all samples are assigned.
        
        Args:
            sample_id_seqlens: List of (sample_id, sequence_length) tuples.
            config: Scheduling configuration (optional).
        
        Returns:
            Tuple of:
              - groups: List of micro-batch assignments per group
              - sample_id_groups: List of sample ID assignments per group
        """
        if config is None:
            config = HybridCPConfig(max_seq_len_per_rank=self.max_seq_len_per_rank)
        
        groups: List[List[MicroBatch]] = []
        sample_id_groups: List[List[SampleIdList]] = []
        
        # Sort by sequence length (descending) for best packing
        remaining = sorted(sample_id_seqlens, key=lambda x: x[1], reverse=True)
        
        while remaining:
            mb, remaining, exec_times, sample_ids = self.next_hdp_group(
                remaining,
                self.get_total_workload,
                self.total_hdp_gpus,
                delta=config.balance_slack,
                strategy=config.strategy,
            )
            groups.append(mb)
            
            # Pad to total_hdp_gpus if needed
            while len(sample_ids) < self.total_hdp_gpus:
                sample_ids.append([])
            sample_id_groups.append(sample_ids)
        
        return groups, sample_id_groups


# ════════════════════════════════════════════════════════════════════════════════
# HYBRID CP FORWARD-BACKWARD EXECUTOR
# ════════════════════════════════════════════════════════════════════════════════

def hybrid_context_parallel_forward_backward(
    forward_step_func: Callable,
    data_iterator: Any,
    model: torch.nn.Module,
    num_microbatches: int,
    *,
    forward_data_store: List[Any],
    config: Any,
    collect_non_loss_data: bool = False,
    first_val_step: bool = False,
    forward_only: bool = False,
    no_sync_func: Callable,
    total_num_tokens: int = 0,
    model_type: Any = None,
    # Parallel state getters (dependency injection)
    get_data_parallel_rank: Optional[Callable[[], int]] = None,
    get_tensor_model_parallel_rank: Optional[Callable[[], int]] = None,
    get_tensor_model_parallel_src_rank: Optional[Callable[[], int]] = None,
    get_tensor_model_parallel_group: Optional[Callable[[], dist.ProcessGroup]] = None,
    get_data_parallel_group: Optional[Callable[[], dist.ProcessGroup]] = None,
) -> Tuple[List[Any], int]:
    """
    Execute forward/backward passes with hybrid context-parallel scheduling.
    
    This function orchestrates packed sample scheduling across the CP domain,
    determining:
      1. Number of micro-batches per CP rank
      2. Number of groups each CP rank executes
      3. Sub-samples per group for each CP rank
    
    Groups are defined by sets of samples that can run without barriers.
    Barriers are inserted between groups when GPU assignments change.
    
    Args:
        forward_step_func: User's forward step function.
        data_iterator: Data iterator yielding batches.
        model: Model to train.
        num_microbatches: Number of micro-batches in the global batch.
        forward_data_store: Storage for forward outputs.
        config: Training configuration.
        collect_non_loss_data: Whether to collect non-loss outputs.
        first_val_step: Whether this is the first validation step.
        forward_only: Whether to skip backward pass.
        no_sync_func: Context manager to disable gradient sync.
        total_num_tokens: Running token count.
        model_type: Model type identifier.
        get_*: Parallel state getter functions (for testing/DI).
    
    Returns:
        Tuple of (forward_data_store, total_num_tokens).
    
    Complexity: O(S × T) where S = samples, T = per-sample forward/backward time.
    """
    # This is a stub for the full implementation
    # The actual implementation requires deep integration with the training loop
    # and parallel state management
    
    logger.warning(
        "hybrid_context_parallel_forward_backward is a placeholder. "
        "Full implementation requires integration with megatron-core parallel_state."
    )
    
    return forward_data_store, total_num_tokens
