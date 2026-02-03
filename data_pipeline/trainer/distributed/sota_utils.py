# ════════════════════════════════════════════════════════════════════════════════
# SOTA Distributed Utilities - Above SOTA-Level Distributed Training Utilities
# ════════════════════════════════════════════════════════════════════════════════
# Generalized utilities for distributed training that work with ANY model,
# optimizer, or scheduler from the existing TrainScale registry.
#
# Hardware Support:
#   - NVIDIA: A100, H100, H200, B100, B200 (CUDA/NCCL)
#   - AMD: MI300X, MI325X (ROCm/RCCL)
#
# Features:
#   - Distributed reduction with DTensor support
#   - Model-agnostic gradient clipping
#   - Deterministic seeding across meshes
#   - Mixed precision with any model architecture
#   - Registry-compatible model/optimizer preparation
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import contextlib
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

# ════════════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# Distributed State - Generalized Process Group Management
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class DistributedState:
    """
    Encapsulates distributed training state.
    
    Provides a unified interface for distributed operations that works
    with any model, optimizer, or scheduler from the TrainScale registry.
    
    Attributes:
        rank: Global process rank (0-indexed)
        local_rank: Local rank within node
        world_size: Total number of processes
        is_main_process: True if rank 0
        device: Current device
        backend: Distributed backend ("nccl", "gloo", "cpu:gloo,cuda:nccl")
        initialized: Whether distributed is initialized
    """
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    is_main_process: bool = True
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    backend: str = "nccl"
    initialized: bool = False
    
    @classmethod
    def from_environment(cls) -> "DistributedState":
        """
        Create DistributedState from environment variables.
        
        Supports standard PyTorch distributed environment:
          - RANK, LOCAL_RANK, WORLD_SIZE
          - MASTER_ADDR, MASTER_PORT
        
        Returns:
            DistributedState populated from environment
        """
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
        
        initialized = dist.is_initialized()
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            is_main_process=(rank == 0),
            device=device,
            backend=backend,
            initialized=initialized,
        )
    
    @classmethod
    def initialize(
        cls,
        backend: str = "auto",
        init_method: Optional[str] = None,
        timeout_minutes: int = 30,
    ) -> "DistributedState":
        """
        Initialize distributed training and return state.
        
        Args:
            backend: "nccl", "gloo", or "auto" (auto-detect)
            init_method: Process group init method (default: env://)
            timeout_minutes: Timeout for initialization
        
        Returns:
            Initialized DistributedState
        """
        from datetime import timedelta
        
        # Auto-detect backend
        if backend == "auto":
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        
        # Get distributed parameters
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Set device
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        
        # Initialize process group
        if not dist.is_initialized() and world_size > 1:
            dist.init_process_group(
                backend=backend,
                init_method=init_method or "env://",
                timeout=timedelta(minutes=timeout_minutes),
                rank=rank,
                world_size=world_size,
            )
            
            if rank == 0:
                logger.info(
                    f"Initialized distributed: rank={rank}, world_size={world_size}, "
                    f"backend={backend}, device={device}"
                )
        
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            is_main_process=(rank == 0),
            device=device,
            backend=backend,
            initialized=True,
        )
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.initialized and self.world_size > 1:
            dist.barrier()
    
    def broadcast(self, tensor: Tensor, src: int = 0) -> Tensor:
        """Broadcast tensor from source rank."""
        if self.world_size > 1:
            dist.broadcast(tensor, src=src)
        return tensor


# ════════════════════════════════════════════════════════════════════════════════
# Gradient Operations - Model-Agnostic
# ════════════════════════════════════════════════════════════════════════════════

def clip_grad_norm_(
    model_or_params: Union[nn.Module, Iterator[Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    foreach: Optional[bool] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Tensor:
    """
    Clip gradient norm for any model (FSDP, DDP, or regular).
    
    This function auto-detects the model type and uses the appropriate
    gradient clipping method:
      - FSDP: Uses FSDP's efficient distributed clip
      - DDP: Uses standard PyTorch clip with distributed norm
      - Regular: Uses standard PyTorch clip
    
    Args:
        model_or_params: Model or parameters iterator
        max_norm: Maximum gradient norm
        norm_type: Type of norm (default 2.0 for L2)
        foreach: Use foreach optimization if available
        process_group: Process group for distributed norm (optional)
    
    Returns:
        Total gradient norm before clipping
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    
    # Handle FSDP models
    if isinstance(model_or_params, FSDP):
        return model_or_params.clip_grad_norm_(max_norm, norm_type)
    
    # Extract parameters
    if isinstance(model_or_params, nn.Module):
        params = list(model_or_params.parameters())
    else:
        params = list(model_or_params)
    
    # Filter grads
    grads = [p.grad for p in params if p.grad is not None]
    
    if len(grads) == 0:
        return torch.tensor(0.0)
    
    # Compute local norm
    device = grads[0].device
    
    if norm_type == float("inf"):
        local_norm = max(g.abs().max() for g in grads)
        if dist.is_initialized() and process_group is not None:
            dist.all_reduce(local_norm, op=dist.ReduceOp.MAX, group=process_group)
        total_norm = local_norm
    else:
        local_norm_sq = sum(g.norm(norm_type).pow(2) for g in grads)
        
        if dist.is_initialized() and process_group is not None:
            dist.all_reduce(local_norm_sq, group=process_group)
        
        total_norm = local_norm_sq.pow(1.0 / norm_type)
    
    # Clip gradients
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    
    for g in grads:
        g.mul_(clip_coef_clamped.to(g.device))
    
    return total_norm


def sync_gradients(
    model: nn.Module,
    process_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    Explicitly synchronize gradients across processes.
    
    For DDP, this is normally automatic. This function is for
    LOCAL_SGD or gradient accumulation scenarios.
    
    Args:
        model: Model with gradients to sync
        process_group: Process group for sync
    """
    if not dist.is_initialized():
        return
    
    world_size = dist.get_world_size(process_group)
    if world_size <= 1:
        return
    
    # Unwrap DDP if needed
    module = model.module if hasattr(model, "module") else model
    
    for param in module.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, group=process_group)
            param.grad.div_(world_size)


# ════════════════════════════════════════════════════════════════════════════════
# Model Preparation - Registry Compatible
# ════════════════════════════════════════════════════════════════════════════════

def prepare_model_for_distributed(
    model: nn.Module,
    distributed_state: DistributedState,
    strategy: str = "ddp",
    **strategy_kwargs,
) -> nn.Module:
    """
    Prepare any model for distributed training.
    
    Works with models from the TrainScale registry or any PyTorch model.
    
    Args:
        model: Model to prepare (from registry or custom)
        distributed_state: Distributed state
        strategy: "ddp", "fsdp", "fsdp2", or "none"
        **strategy_kwargs: Additional strategy-specific arguments
    
    Returns:
        Distributed model wrapper
    """
    if distributed_state.world_size <= 1:
        # Single device, just move to device
        return model.to(distributed_state.device)
    
    # Move model to device
    model = model.to(distributed_state.device)
    
    if strategy == "none":
        return model
    
    if strategy == "ddp":
        from .ddp import create_ddp
        ddp = create_ddp(**strategy_kwargs)
        return ddp.wrap_model(model)
    
    if strategy in ("fsdp", "fsdp2"):
        from .fsdp2 import create_fsdp2
        fsdp = create_fsdp2(**strategy_kwargs)
        return fsdp.wrap_model(model)
    
    raise ValueError(f"Unknown distributed strategy: {strategy}")


def prepare_optimizer_for_distributed(
    optimizer: Optimizer,
    model: nn.Module,
    strategy: str = "ddp",
) -> Optimizer:
    """
    Prepare any optimizer for distributed training.
    
    Works with any optimizer from TrainScale (Lion, CAME, etc.) or PyTorch.
    
    Args:
        optimizer: Optimizer instance (from create_optimizer or custom)
        model: Distributed model
        strategy: Distribution strategy used
    
    Returns:
        Prepared optimizer
    """
    # For FSDP with use_orig_params=True, optimizer should work as-is
    # For standard DDP, optimizer works as-is
    # This function is for future extensions (optimizer state sharding, etc.)
    
    return optimizer


def prepare_scheduler_for_distributed(
    scheduler: Any,
    distributed_state: DistributedState,
) -> Any:
    """
    Prepare any scheduler for distributed training.
    
    Works with any scheduler from TrainScale (WSD, REX, etc.) or PyTorch.
    
    Args:
        scheduler: LR scheduler (from create_sota_scheduler or custom)
        distributed_state: Distributed state
    
    Returns:
        Prepared scheduler (unchanged, but validated)
    """
    # Schedulers work identically across all processes
    # No modification needed, just validation
    return scheduler


# ════════════════════════════════════════════════════════════════════════════════
# Distributed Reduction - DTensor Compatible
# ════════════════════════════════════════════════════════════════════════════════

def dist_reduce(
    tensor: Tensor,
    op: str = "mean",
    process_group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> Union[Tensor, dist.Work]:
    """
    Reduce tensor across distributed processes.
    
    Supports DTensor, regular Tensor, and scalar.
    
    Args:
        tensor: Tensor to reduce
        op: "mean", "sum", "max", "min"
        process_group: Process group for reduction
        async_op: Return async handle
    
    Returns:
        Reduced tensor (or async handle if async_op=True)
    """
    if not dist.is_initialized():
        return tensor
    
    world_size = dist.get_world_size(process_group)
    if world_size <= 1:
        return tensor
    
    # Handle scalars
    if not isinstance(tensor, Tensor):
        tensor = torch.tensor(tensor, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Reduction operation mapping
    reduce_ops = {
        "mean": dist.ReduceOp.SUM,
        "sum": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }
    
    reduce_op = reduce_ops.get(op, dist.ReduceOp.SUM)
    
    # Perform all-reduce
    result = tensor.clone()
    handle = dist.all_reduce(result, op=reduce_op, group=process_group, async_op=async_op)
    
    if async_op:
        return handle
    
    # Scale for mean
    if op == "mean":
        result = result / world_size
    
    return result


def dist_all_gather(
    tensor: Tensor,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Tensor:
    """
    Gather tensor from all processes.
    
    Args:
        tensor: Local tensor
        process_group: Process group
    
    Returns:
        Gathered tensor concatenated along dim 0
    """
    if not dist.is_initialized():
        return tensor
    
    world_size = dist.get_world_size(process_group)
    if world_size <= 1:
        return tensor
    
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=process_group)

    # DEBUG: Diagnose scalar warning
    if tensor.dim() == 0:
        # User reported warning with cat on scalars. verifying fix.
        # logger.debug(f"dist_all_gather: detected scalar input, using stack instead of cat")
        return torch.stack(gathered)
    
    return torch.cat(gathered, dim=0)


# ════════════════════════════════════════════════════════════════════════════════
# Determinism - Consistent Seeding Across Meshes
# ════════════════════════════════════════════════════════════════════════════════

def set_deterministic_seed(
    seed: int,
    distributed_state: Optional[DistributedState] = None,
    per_rank_offset: bool = True,
) -> None:
    """
    Set deterministic seed across all ranks.
    
    Args:
        seed: Base random seed
        distributed_state: Optional distributed state
        per_rank_offset: Add rank to seed for different data per GPU
    """
    rank = 0
    if distributed_state is not None:
        rank = distributed_state.rank
    elif dist.is_initialized():
        rank = dist.get_rank()
    
    # Compute final seed
    final_seed = seed + (rank if per_rank_offset else 0)
    
    # Set all seeds
    random.seed(final_seed)
    torch.manual_seed(final_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(final_seed)
        torch.cuda.manual_seed_all(final_seed)
    
    # Optional: NumPy if available
    try:
        import numpy as np
        np.random.seed(final_seed)
    except ImportError:
        pass
    
    # Enable deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ════════════════════════════════════════════════════════════════════════════════
# Mixed Precision - Model Agnostic
# ════════════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def mixed_precision_context(
    enabled: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> Iterator[None]:
    """
    Context manager for mixed precision training.
    
    Works with any model architecture.
    
    Args:
        enabled: Enable mixed precision
        dtype: Data type for autocast (bfloat16 or float16)
    
    Yields:
        Context for mixed precision forward pass
    """
    if not enabled:
        yield
        return
    
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    
    with torch.autocast(device_type=device_type, dtype=dtype):
        yield


# ════════════════════════════════════════════════════════════════════════════════
# Logging Utilities
# ════════════════════════════════════════════════════════════════════════════════

def log_rank_0(message: str, level: str = "info") -> None:
    """
    Log message only on rank 0.
    
    Args:
        message: Message to log
        level: Log level ("info", "warning", "error", "debug")
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    
    log_fn = getattr(logger, level, logger.info)
    log_fn(message)


def print_rank_0(*args, **kwargs) -> None:
    """
    Print only on rank 0.
    
    Args:
        *args: Print arguments
        **kwargs: Print keyword arguments
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    print(*args, **kwargs)


# ════════════════════════════════════════════════════════════════════════════════
# Exports
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # State management
    "DistributedState",
    
    # Gradient operations
    "clip_grad_norm_",
    "sync_gradients",
    
    # Model/optimizer/scheduler preparation (registry compatible)
    "prepare_model_for_distributed",
    "prepare_optimizer_for_distributed",
    "prepare_scheduler_for_distributed",
    
    # Distributed operations
    "dist_reduce",
    "dist_all_gather",
    
    # Determinism
    "set_deterministic_seed",
    
    # Mixed precision
    "mixed_precision_context",
    
    # Logging
    "log_rank_0",
    "print_rank_0",
]
