# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer Module - Learning Rate Schedulers
# ════════════════════════════════════════════════════════════════════════════════
# Above-SOTA LR schedulers with warmup integration.
#
# Features:
# 1. Unified warmup integration (linear, exponential, constant)
# 2. State serialization for checkpointing
# 3. Compatible with all optimizers (PyTorch and custom)
# 4. Step-based progression with sub-step precision
#
# Scheduler Types:
# - Cosine: Smooth annealing to min_lr (most popular)
# - CosineRestarts: SGDR with periodic restarts
# - Linear: Linear decay to end_lr
# - Polynomial: Power-law decay
# - InverseSqrt: 1/sqrt(step) decay (original Transformer)
# - OneCycle: Cyclical LR (fast.ai)
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from data_pipeline.trainer.core.types import (
    SchedulerConfig,
    SchedulerType,
    WarmupType,
)

# ═════════════════════════════════════════════════════════════════════════════════
# Base Scheduler
# ═════════════════════════════════════════════════════════════════════════════════

class BaseScheduler(abc.ABC):
    """
    Abstract base class for learning rate schedulers.
    
    Provides:
    - Unified warmup integration
    - Step-based progression
    - State serialization
    - Optional last_epoch tracking
    
    Subclasses must implement:
    - _get_lr_after_warmup(): Compute LR after warmup phase
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        *,
        warmup_steps: int = 0,
        warmup_type: WarmupType = WarmupType.LINEAR,
        warmup_start_lr: float = 0.0,
        min_lr_ratio: float = 0.0,
        last_step: int = -1,
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            num_training_steps: Total training steps
            warmup_steps: Number of warmup steps
            warmup_type: Warmup strategy (linear, exponential, constant)
            warmup_start_lr: Starting LR ratio during warmup (0 = start from 0)
            min_lr_ratio: Minimum LR as fraction of base LR
            last_step: Last completed step (-1 for fresh start)
        """
        self.optimizer = optimizer
        self.num_training_steps = num_training_steps
        self.warmup_steps = warmup_steps
        self.warmup_type = warmup_type
        self.warmup_start_lr = warmup_start_lr
        self.min_lr_ratio = min_lr_ratio
        
        # Store base learning rates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        
        # Step counter
        self._step_count = last_step + 1
        
        # Compute minimum LRs
        self.min_lrs = [lr * min_lr_ratio for lr in self.base_lrs]
        
        # Initialize LRs
        if last_step == -1:
            self._update_lrs()
    
    def _get_warmup_lr(self, base_lr: float, step: int) -> float:
        """
        Compute learning rate during warmup phase.
        
        Args:
            base_lr: Target (base) learning rate
            step: Current step (0-indexed)
            
        Returns:
            Learning rate for current step
        """
        if self.warmup_steps == 0 or step >= self.warmup_steps:
            return base_lr
        
        # Progress through warmup (0 to 1)
        progress = step / self.warmup_steps
        
        if self.warmup_type == WarmupType.LINEAR:
            # Linear interpolation from warmup_start_lr to base_lr
            start = base_lr * self.warmup_start_lr
            return start + (base_lr - start) * progress
        
        elif self.warmup_type == WarmupType.EXPONENTIAL:
            # Exponential growth
            # lr = base_lr * exp(-5 * (1 - progress))
            return base_lr * math.exp(-5.0 * (1.0 - progress))
        
        elif self.warmup_type == WarmupType.CONSTANT:
            # Constant warmup LR, then jump to base
            return base_lr * self.warmup_start_lr
        
        else:
            # Default to linear
            return base_lr * progress
    
    @abc.abstractmethod
    def _get_lr_after_warmup(
        self,
        base_lr: float,
        min_lr: float,
        step: int,
        decay_steps: int,
    ) -> float:
        """
        Compute learning rate after warmup phase.
        
        Args:
            base_lr: Base learning rate
            min_lr: Minimum learning rate floor
            step: Steps after warmup (0 = first step after warmup)
            decay_steps: Total decay steps (total - warmup)
            
        Returns:
            Learning rate for current step
        """
        pass
    
    def get_lr(self) -> List[float]:
        """
        Get current learning rates for all parameter groups.
        
        Returns:
            List of learning rates
        """
        lrs = []
        
        for base_lr, min_lr in zip(self.base_lrs, self.min_lrs):
            if self._step_count < self.warmup_steps:
                # Still in warmup
                lr = self._get_warmup_lr(base_lr, self._step_count)
            else:
                # After warmup
                decay_step = self._step_count - self.warmup_steps
                decay_steps = max(1, self.num_training_steps - self.warmup_steps)
                lr = self._get_lr_after_warmup(base_lr, min_lr, decay_step, decay_steps)
            
            lrs.append(lr)
        
        return lrs
    
    def _update_lrs(self) -> None:
        """Update optimizer learning rates."""
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr
    
    def step(self, step: Optional[int] = None) -> None:
        """
        Update learning rate.
        
        Args:
            step: Optional step override (for manual stepping)
        """
        if step is not None:
            self._step_count = step
        else:
            self._step_count += 1
        
        self._update_lrs()
    
    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state for checkpointing."""
        return {
            "step_count": self._step_count,
            "base_lrs": self.base_lrs,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        self._step_count = state_dict["step_count"]
        self.base_lrs = state_dict["base_lrs"]
        self._update_lrs()
    
    @property
    def last_step(self) -> int:
        """Return last completed step."""
        return self._step_count - 1
    
    def get_last_lr(self) -> List[float]:
        """
        Return last computed learning rates.
        
        Compatible with PyTorch LRScheduler interface.
        """
        return [group["lr"] for group in self.optimizer.param_groups]
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"warmup={self.warmup_steps}, "
            f"total={self.num_training_steps}, "
            f"step={self._step_count})"
        )


# ═════════════════════════════════════════════════════════════════════════════════
# Concrete Schedulers
# ═════════════════════════════════════════════════════════════════════════════════

class CosineScheduler(BaseScheduler):
    """
    Cosine annealing learning rate scheduler.
    
    LR follows half-cosine curve from base_lr to min_lr.
    Most popular scheduler for transformer training.
    
    Formula:
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * step / total))
    """
    
    def _get_lr_after_warmup(
        self,
        base_lr: float,
        min_lr: float,
        step: int,
        decay_steps: int,
    ) -> float:
        """Cosine annealing from base_lr to min_lr."""
        if step >= decay_steps:
            return min_lr
        
        progress = step / decay_steps
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_decay


class CosineRestartsScheduler(BaseScheduler):
    """
    Cosine annealing with warm restarts (SGDR).
    
    Periodically restarts the cosine schedule for improved exploration.
    
    Reference: SGDR: Stochastic Gradient Descent with Warm Restarts
    https://arxiv.org/abs/1608.03983
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        *,
        num_cycles: float = 1.0,
        cycle_decay: float = 1.0,
        restart_warmup_steps: int = 0,
        **kwargs,
    ):
        self.num_cycles = num_cycles
        self.cycle_decay = cycle_decay
        self.restart_warmup_steps = restart_warmup_steps
        super().__init__(optimizer, num_training_steps, **kwargs)
    
    def _get_lr_after_warmup(
        self,
        base_lr: float,
        min_lr: float,
        step: int,
        decay_steps: int,
    ) -> float:
        """Cosine with restarts."""
        if step >= decay_steps:
            return min_lr
        
        # Compute which cycle we're in
        progress = step / decay_steps
        cycle = math.floor(progress * self.num_cycles)
        cycle_progress = (progress * self.num_cycles) - cycle
        
        # Decay base_lr for subsequent cycles
        cycle_base_lr = base_lr * (self.cycle_decay ** cycle)
        
        # Cosine within cycle
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
        return min_lr + (cycle_base_lr - min_lr) * cosine_decay


class LinearScheduler(BaseScheduler):
    """
    Linear learning rate decay.
    
    LR decreases linearly from base_lr to min_lr.
    
    Formula:
        lr = base_lr - (base_lr - min_lr) * (step / total)
    """
    
    def _get_lr_after_warmup(
        self,
        base_lr: float,
        min_lr: float,
        step: int,
        decay_steps: int,
    ) -> float:
        """Linear decay from base_lr to min_lr."""
        if step >= decay_steps:
            return min_lr
        
        progress = step / decay_steps
        return base_lr - (base_lr - min_lr) * progress


class PolynomialScheduler(BaseScheduler):
    """
    Polynomial learning rate decay.
    
    LR follows power-law decay curve.
    
    Formula:
        lr = min_lr + (base_lr - min_lr) * (1 - step/total)^power
    
    Power values:
    - power=1.0: Linear decay
    - power=2.0: Quadratic decay (slower start, faster end)
    - power=0.5: Square root decay (faster start, slower end)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        *,
        power: float = 1.0,
        **kwargs,
    ):
        self.power = power
        super().__init__(optimizer, num_training_steps, **kwargs)
    
    def _get_lr_after_warmup(
        self,
        base_lr: float,
        min_lr: float,
        step: int,
        decay_steps: int,
    ) -> float:
        """Polynomial decay."""
        if step >= decay_steps:
            return min_lr
        
        progress = step / decay_steps
        decay = (1.0 - progress) ** self.power
        return min_lr + (base_lr - min_lr) * decay


class InverseSqrtScheduler(BaseScheduler):
    """
    Inverse square root learning rate decay.
    
    Original Transformer ("Attention is All You Need") schedule.
    
    Formula:
        lr = base_lr * sqrt(warmup_steps) / sqrt(step)
    
    Or equivalently after warmup:
        lr = base_lr / sqrt(step / warmup_steps)
    """
    
    def _get_lr_after_warmup(
        self,
        base_lr: float,
        min_lr: float,
        step: int,
        decay_steps: int,
    ) -> float:
        """Inverse square root decay."""
        # Use warmup_steps as the normalization factor
        warmup = max(1, self.warmup_steps)
        
        # Compute inverse sqrt decay
        effective_step = step + warmup
        decay = math.sqrt(warmup / effective_step)
        
        lr = base_lr * decay
        return max(lr, min_lr)


class ConstantScheduler(BaseScheduler):
    """
    Constant learning rate (with optional warmup).
    
    LR stays at base_lr after warmup.
    """
    
    def _get_lr_after_warmup(
        self,
        base_lr: float,
        min_lr: float,
        step: int,
        decay_steps: int,
    ) -> float:
        """Constant LR."""
        return base_lr


class OneCycleScheduler(BaseScheduler):
    """
    1Cycle learning rate policy.
    
    Fast.ai-style cyclical learning rate:
    1. Warmup phase: LR increases from min to max
    2. Decay phase: LR decreases from max to min (using cosine)
    
    Reference: Super-Convergence: Very Fast Training of Neural Networks
    https://arxiv.org/abs/1708.07120
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        *,
        max_lr_ratio: float = 10.0,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        pct_start: float = 0.3,
        **kwargs,
    ):
        # Override warmup to use pct_start
        warmup_steps = int(pct_start * num_training_steps)
        kwargs["warmup_steps"] = warmup_steps
        
        self.max_lr_ratio = max_lr_ratio
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.pct_start = pct_start
        
        super().__init__(optimizer, num_training_steps, **kwargs)
    
    def _get_warmup_lr(self, base_lr: float, step: int) -> float:
        """1Cycle warmup: linear increase to max."""
        if self.warmup_steps == 0 or step >= self.warmup_steps:
            return base_lr * self.max_lr_ratio
        
        # Start LR
        initial_lr = base_lr / self.div_factor
        max_lr = base_lr * self.max_lr_ratio
        
        progress = step / self.warmup_steps
        return initial_lr + (max_lr - initial_lr) * progress
    
    def _get_lr_after_warmup(
        self,
        base_lr: float,
        min_lr: float,
        step: int,
        decay_steps: int,
    ) -> float:
        """1Cycle decay: cosine from max to min."""
        max_lr = base_lr * self.max_lr_ratio
        final_lr = base_lr / self.final_div_factor
        
        if step >= decay_steps:
            return final_lr
        
        progress = step / decay_steps
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr + (max_lr - final_lr) * cosine_decay


# ═════════════════════════════════════════════════════════════════════════════════
# Factory Function
# ═════════════════════════════════════════════════════════════════════════════════

def create_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
    num_training_steps: int,
) -> BaseScheduler:
    """
    Create scheduler from configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        num_training_steps: Total training steps
        
    Returns:
        Configured scheduler instance
    """
    # Compute warmup steps
    warmup_steps = config.warmup_steps
    if warmup_steps == 0 and config.warmup_ratio > 0:
        warmup_steps = int(config.warmup_ratio * num_training_steps)
    
    common_kwargs = {
        "warmup_steps": warmup_steps,
        "warmup_type": config.warmup_type,
        "min_lr_ratio": config.min_lr_ratio,
    }
    
    scheduler_type = config.scheduler_type
    
    if scheduler_type == SchedulerType.CONSTANT:
        return ConstantScheduler(optimizer, num_training_steps, **common_kwargs)
    
    elif scheduler_type == SchedulerType.CONSTANT_WITH_WARMUP:
        return ConstantScheduler(optimizer, num_training_steps, **common_kwargs)
    
    elif scheduler_type == SchedulerType.LINEAR:
        return LinearScheduler(optimizer, num_training_steps, **common_kwargs)
    
    elif scheduler_type == SchedulerType.COSINE:
        return CosineScheduler(optimizer, num_training_steps, **common_kwargs)
    
    elif scheduler_type == SchedulerType.COSINE_RESTARTS:
        return CosineRestartsScheduler(
            optimizer, num_training_steps,
            num_cycles=config.num_cycles,
            restart_warmup_steps=config.restart_warmup_steps,
            **common_kwargs,
        )
    
    elif scheduler_type == SchedulerType.POLYNOMIAL:
        return PolynomialScheduler(
            optimizer, num_training_steps,
            power=config.power,
            **common_kwargs,
        )
    
    elif scheduler_type == SchedulerType.INVERSE_SQRT:
        return InverseSqrtScheduler(optimizer, num_training_steps, **common_kwargs)
    
    elif scheduler_type == SchedulerType.ONE_CYCLE:
        return OneCycleScheduler(optimizer, num_training_steps, **common_kwargs)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ═════════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═════════════════════════════════════════════════════════════════════════════════

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> CosineScheduler:
    """
    Create cosine scheduler with linear warmup.
    
    Most common scheduler for transformer fine-tuning.
    """
    return CosineScheduler(
        optimizer,
        num_training_steps,
        warmup_steps=num_warmup_steps,
        warmup_type=WarmupType.LINEAR,
        min_lr_ratio=min_lr_ratio,
    )


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LinearScheduler:
    """
    Create linear decay scheduler with warmup.
    
    Popular for BERT-style fine-tuning.
    """
    return LinearScheduler(
        optimizer,
        num_training_steps,
        warmup_steps=num_warmup_steps,
        warmup_type=WarmupType.LINEAR,
        min_lr_ratio=0.0,
    )


# ═════════════════════════════════════════════════════════════════════════════════
# Export
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Base
    "BaseScheduler",
    # Schedulers
    "CosineScheduler",
    "CosineRestartsScheduler",
    "LinearScheduler",
    "PolynomialScheduler",
    "InverseSqrtScheduler",
    "ConstantScheduler",
    "OneCycleScheduler",
    # Factory
    "create_scheduler",
    # Convenience
    "get_cosine_schedule_with_warmup",
    "get_linear_schedule_with_warmup",
]
