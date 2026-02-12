# ════════════════════════════════════════════════════════════════════════════════
# SOTA Training Metrics - Distributed Utilities
# ════════════════════════════════════════════════════════════════════════════════
# Utilities for synchronizing metrics across distributed ranks.
#
# Features:
#   - Single metric all-reduce (mean/sum)
#   - Batched metric reduction
#   - Metric buffer for efficient communication
#   - Rank-aware helpers
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor

# ════════════════════════════════════════════════════════════════════════════════
# Distributed Helpers
# ════════════════════════════════════════════════════════════════════════════════

def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return torch.distributed.is_initialized()


def get_world_size() -> int:
    """Get world size (1 if not distributed)."""
    if not is_distributed():
        return 1
    return torch.distributed.get_world_size()


def get_rank() -> int:
    """Get current rank (0 if not distributed)."""
    if not is_distributed():
        return 0
    return torch.distributed.get_rank()


def is_main_process() -> bool:
    """Check if current process is rank 0."""
    return get_rank() == 0


# ════════════════════════════════════════════════════════════════════════════════
# Single Metric Sync
# ════════════════════════════════════════════════════════════════════════════════

def sync_metric(
    value: Union[float, int, Tensor],
    operation: Literal["sum", "mean", "max", "min"] = "mean",
    process_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> float:
    """
    Synchronize a single metric across distributed ranks.
    
    Args:
        value: Metric value to sync
        operation: Reduction operation ("sum", "mean", "max", "min")
        process_group: Process group for reduction (None = default)
    
    Returns:
        Reduced metric value
    
    Example:
        >>> local_loss = 2.5
        >>> global_loss = sync_metric(local_loss, operation="mean")
    """
    if not is_distributed():
        return float(value)
    
    # Convert to tensor
    if isinstance(value, Tensor):
        tensor = value.detach().clone()
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
    else:
        tensor = torch.tensor([float(value)], dtype=torch.float64)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    
    # Reduce
    if operation == "sum":
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.SUM, group=process_group
        )
    elif operation == "mean":
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.SUM, group=process_group
        )
        tensor = tensor / get_world_size()
    elif operation == "max":
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MAX, group=process_group
        )
    elif operation == "min":
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MIN, group=process_group
        )
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return tensor.item()


# ════════════════════════════════════════════════════════════════════════════════
# Batched Metric Sync
# ════════════════════════════════════════════════════════════════════════════════

def sync_metrics(
    metrics: Dict[str, Union[float, int]],
    operation: Literal["sum", "mean"] = "mean",
    process_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> Dict[str, float]:
    """
    Synchronize a dictionary of metrics across distributed ranks.
    
    More efficient than calling sync_metric for each value since it
    batches all values into a single all-reduce operation.
    
    Args:
        metrics: Dictionary of metric name to value
        operation: Reduction operation ("sum" or "mean")
        process_group: Process group for reduction (None = default)
    
    Returns:
        Dictionary of reduced metric values
    
    Example:
        >>> local_metrics = {"loss": 2.5, "accuracy": 0.85, "tokens": 1024}
        >>> global_metrics = sync_metrics(local_metrics, operation="mean")
    """
    if not is_distributed():
        return {k: float(v) for k, v in metrics.items()}
    
    if not metrics:
        return {}
    
    # Pack into tensor
    keys = list(metrics.keys())
    values = [float(metrics[k]) for k in keys]
    tensor = torch.tensor(values, dtype=torch.float64)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    
    # Reduce
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=process_group)
    
    if operation == "mean":
        tensor = tensor / get_world_size()
    
    # Unpack
    result = {}
    for i, key in enumerate(keys):
        result[key] = tensor[i].item()
    
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Distributed Metric Buffer
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class DistributedMetricBuffer:
    """
    Buffer for efficient batched metric reduction.
    
    Accumulates metrics locally and reduces them in a single operation.
    
    Example:
        >>> buffer = DistributedMetricBuffer()
        >>> for step in range(10):
        ...     buffer.add("loss", loss_value)
        ...     buffer.add("accuracy", acc_value)
        >>> reduced = buffer.reduce_all()  # Single all-reduce for all metrics
    """
    
    operation: Literal["sum", "mean"] = "mean"
    _buffer: Dict[str, List[float]] = field(default_factory=dict, repr=False)
    
    def add(self, name: str, value: Union[float, int, Tensor]) -> None:
        """
        Add a metric value to the buffer.
        
        Args:
            name: Metric name
            value: Metric value
        """
        if isinstance(value, Tensor):
            value = value.detach().item()
        
        if name not in self._buffer:
            self._buffer[name] = []
        self._buffer[name].append(float(value))
    
    def clear(self) -> None:
        """Clear all buffered values."""
        self._buffer.clear()
    
    def local_means(self) -> Dict[str, float]:
        """
        Compute local means (no reduction).
        
        Returns:
            Dict of metric means
        """
        result = {}
        for name, values in self._buffer.items():
            if values:
                result[name] = sum(values) / len(values)
        return result
    
    def local_sums(self) -> Dict[str, float]:
        """
        Compute local sums (no reduction).
        
        Returns:
            Dict of metric sums
        """
        return {name: sum(values) for name, values in self._buffer.items() if values}
    
    def reduce_all(
        self,
        process_group: Optional["torch.distributed.ProcessGroup"] = None,
    ) -> Dict[str, float]:
        """
        Reduce all buffered metrics across ranks.
        
        Args:
            process_group: Process group for reduction
        
        Returns:
            Dict of reduced metric values
        """
        # First compute local aggregates
        if self.operation == "mean":
            local = self.local_means()
        else:
            local = self.local_sums()
        
        # Then reduce across ranks
        return sync_metrics(local, operation=self.operation, process_group=process_group)
    
    def counts(self) -> Dict[str, int]:
        """Get count of values for each metric."""
        return {name: len(values) for name, values in self._buffer.items()}


# ════════════════════════════════════════════════════════════════════════════════
# Exports
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    "is_distributed",
    "get_world_size",
    "get_rank",
    "is_main_process",
    "sync_metric",
    "sync_metrics",
    "DistributedMetricBuffer",
]
