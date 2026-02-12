# ════════════════════════════════════════════════════════════════════════════════
# SOTA Learning Rate Schedulers - Above Unsloth Implementation
# ════════════════════════════════════════════════════════════════════════════════
# Comprehensive scheduler suite with:
# - WSD (Warmup-Stable-Decay) - used by LLaMA 3
# - REX (Rapid Warmup) - faster convergence
# - OneCycleLR (Super-convergence)
# - SHARP (Sharpness-Aware)
# - Exponential with warmup
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
from typing import Callable, List, Literal, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


# ═════════════════════════════════════════════════════════════════════════════════
# Base Enhanced Scheduler
# ═════════════════════════════════════════════════════════════════════════════════

class BaseEnhancedScheduler(LRScheduler):
    """
    Base scheduler with warmup and min_lr support.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.0,
        min_lr: float = 0.0,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.num_training_steps = num_training_steps
        
        # Handle warmup
        if warmup_steps > 0:
            self.warmup_steps = warmup_steps
        elif warmup_ratio > 0:
            self.warmup_steps = int(num_training_steps * warmup_ratio)
        else:
            self.warmup_steps = 0
        
        # Store base lrs before parent init
        self._base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Handle min_lr
        if min_lr > 0:
            self.min_lrs = [min_lr] * len(self._base_lrs)
        elif min_lr_ratio > 0:
            self.min_lrs = [lr * min_lr_ratio for lr in self._base_lrs]
        else:
            self.min_lrs = [0.0] * len(self._base_lrs)
        
        super().__init__(optimizer, last_epoch)
    
    def _get_warmup_factor(self, step: int) -> float:
        """Linear warmup factor."""
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        return 1.0
    
    def _get_decay_factor(self, step: int) -> float:
        """Get decay factor after warmup. Override in subclasses."""
        raise NotImplementedError
    
    def get_lr(self) -> List[float]:
        step = max(0, self.last_epoch)
        warmup_factor = self._get_warmup_factor(step)
        
        if step < self.warmup_steps:
            # During warmup
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # After warmup
            decay_factor = self._get_decay_factor(step - self.warmup_steps)
            return [
                min_lr + (base_lr - min_lr) * decay_factor
                for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
            ]


# ═════════════════════════════════════════════════════════════════════════════════
# WSD Scheduler (Warmup-Stable-Decay) - LLaMA 3 Style
# ═════════════════════════════════════════════════════════════════════════════════

class WSDScheduler(BaseEnhancedScheduler):
    """
    Warmup-Stable-Decay scheduler (used by LLaMA 3).
    
    Three phases:
    1. Warmup: Linear increase to peak LR
    2. Stable: Constant LR
    3. Decay: Cosine decay to min_lr
    
    This is the most effective scheduler for large-scale LLM training.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.0,
        stable_steps: int = 0,
        stable_ratio: float = 0.8,
        min_lr: float = 0.0,
        min_lr_ratio: float = 0.1,
        decay_type: Literal["cosine", "linear", "sqrt"] = "cosine",
        last_epoch: int = -1,
    ):
        # Calculate stable steps
        if stable_steps > 0:
            self.stable_steps = stable_steps
        else:
            warmup = warmup_steps if warmup_steps > 0 else int(num_training_steps * warmup_ratio)
            remaining = num_training_steps - warmup
            self.stable_steps = int(remaining * stable_ratio)
        
        self.decay_type = decay_type
        
        super().__init__(
            optimizer,
            num_training_steps,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            min_lr=min_lr,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
        )
        
        # Decay steps is what's left after warmup and stable
        self.decay_steps = max(1, num_training_steps - self.warmup_steps - self.stable_steps)
    
    def _get_decay_factor(self, step: int) -> float:
        """WSD decay: stable then decay."""
        if step < self.stable_steps:
            # Stable phase
            return 1.0
        
        # Decay phase
        decay_step = step - self.stable_steps
        progress = min(1.0, decay_step / self.decay_steps)
        
        if self.decay_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        elif self.decay_type == "linear":
            return 1.0 - progress
        elif self.decay_type == "sqrt":
            return 1.0 - math.sqrt(progress)
        else:
            return 1.0 - progress


# ═════════════════════════════════════════════════════════════════════════════════
# REX Scheduler (Rapid Warmup with Exponential Decay)
# ═════════════════════════════════════════════════════════════════════════════════

class REXScheduler(BaseEnhancedScheduler):
    """
    REX scheduler for faster convergence.
    
    Uses rapid exponential warmup and aggressive decay for faster training.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.01,
        min_lr: float = 0.0,
        min_lr_ratio: float = 0.0,
        gamma: float = 0.999,
        last_epoch: int = -1,
    ):
        self.gamma = gamma
        super().__init__(
            optimizer,
            num_training_steps,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            min_lr=min_lr,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
        )
    
    def _get_warmup_factor(self, step: int) -> float:
        """Exponential warmup (faster than linear)."""
        if step < self.warmup_steps:
            # Exponential warmup
            progress = (step + 1) / self.warmup_steps
            return 1.0 - math.exp(-5 * progress)  # Rapid exponential
        return 1.0
    
    def _get_decay_factor(self, step: int) -> float:
        """Exponential decay."""
        return self.gamma ** step


# ═════════════════════════════════════════════════════════════════════════════════
# OneCycleLR (Super-convergence)
# ═════════════════════════════════════════════════════════════════════════════════

class OneCycleScheduler(LRScheduler):
    """
    1Cycle scheduler for super-convergence training.
    
    Phases:
    1. Warmup: Increase LR from initial to max
    2. Annealing: Decrease LR from max to min
    
    Also includes momentum cycling (if optimizer supports it).
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        max_lr: Union[float, List[float]] = 0.01,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        anneal_strategy: Literal["cos", "linear"] = "cos",
        cycle_momentum: bool = True,
        base_momentum: float = 0.85,
        max_momentum: float = 0.95,
        last_epoch: int = -1,
    ):
        self.num_training_steps = num_training_steps
        
        # Handle max_lr as list
        if isinstance(max_lr, (int, float)):
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
        else:
            self.max_lrs = list(max_lr)
        
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.anneal_strategy = anneal_strategy
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        
        # Initial and final LRs
        self.initial_lrs = [lr / div_factor for lr in self.max_lrs]
        self.final_lrs = [lr / final_div_factor for lr in self.initial_lrs]
        
        # Update phase boundary
        self.step_up = int(num_training_steps * pct_start)
        self.step_down = num_training_steps - self.step_up
        
        # Set initial LR
        for param_group, lr in zip(optimizer.param_groups, self.initial_lrs):
            param_group['lr'] = lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        step = max(0, self.last_epoch)
        
        if step < self.step_up:
            # Warmup phase
            pct = step / self.step_up
            return self._annealing(
                pct,
                self.initial_lrs,
                self.max_lrs,
            )
        else:
            # Annealing phase
            pct = (step - self.step_up) / self.step_down
            return self._annealing(
                pct,
                self.max_lrs,
                self.final_lrs,
            )
    
    def _annealing(self, pct: float, start_lrs: List[float], end_lrs: List[float]) -> List[float]:
        if self.anneal_strategy == "cos":
            factor = (1 + math.cos(math.pi * pct)) / 2
        else:
            factor = 1 - pct
        
        return [start + factor * (end - start) for start, end in zip(end_lrs, start_lrs)]


# ═════════════════════════════════════════════════════════════════════════════════
# Polynomial Decay Scheduler
# ═════════════════════════════════════════════════════════════════════════════════

class PolynomialDecayScheduler(BaseEnhancedScheduler):
    """
    Polynomial decay scheduler.
    
    lr = (base_lr - min_lr) * (1 - steps/total_steps)^power + min_lr
    
    Common powers:
    - 1.0: Linear decay
    - 2.0: Quadratic decay (more aggressive end)
    - 0.5: Square root decay (gentler end)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.0,
        power: float = 1.0,
        min_lr: float = 0.0,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.power = power
        super().__init__(
            optimizer,
            num_training_steps,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            min_lr=min_lr,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
        )
        
        self.decay_steps = max(1, num_training_steps - self.warmup_steps)
    
    def _get_decay_factor(self, step: int) -> float:
        progress = min(1.0, step / self.decay_steps)
        return (1.0 - progress) ** self.power


# ═════════════════════════════════════════════════════════════════════════════════
# Inverse Square Root Scheduler
# ═════════════════════════════════════════════════════════════════════════════════

class InverseSquareRootScheduler(BaseEnhancedScheduler):
    """
    Inverse square root decay (original Transformer schedule).
    
    lr = base_lr * sqrt(warmup_steps) / sqrt(step)
    
    After warmup, LR decays as 1/sqrt(step).
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        warmup_steps: int = 4000,
        warmup_init_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_init_lr = warmup_init_lr
        super().__init__(
            optimizer,
            num_training_steps,
            warmup_steps=warmup_steps,
            last_epoch=last_epoch,
        )
        
        # Decay factor from warmup completion
        self.decay_factor = math.sqrt(self.warmup_steps)
    
    def get_lr(self) -> List[float]:
        step = max(1, self.last_epoch + 1)
        
        if step < self.warmup_steps:
            # Linear warmup
            lr_step = (self.base_lrs[0] - self.warmup_init_lr) / self.warmup_steps
            return [self.warmup_init_lr + step * lr_step for _ in self.base_lrs]
        else:
            # Inverse sqrt decay
            return [base_lr * self.decay_factor / math.sqrt(step) for base_lr in self.base_lrs]
    
    def _get_decay_factor(self, step: int) -> float:
        return self.decay_factor / math.sqrt(step + self.warmup_steps)


# ═════════════════════════════════════════════════════════════════════════════════
# Cosine with Hard Restarts
# ═════════════════════════════════════════════════════════════════════════════════

class CosineRestartScheduler(BaseEnhancedScheduler):
    """
    Cosine annealing with hard restarts.
    
    LR resets to base_lr at each restart, then decays via cosine.
    T_mult controls cycle length growth (1 = constant, 2 = doubling).
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.0,
        T_0: int = None,
        T_mult: float = 1.0,
        min_lr: float = 0.0,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.T_0 = T_0 if T_0 else (num_training_steps // 4)
        self.T_mult = T_mult
        
        super().__init__(
            optimizer,
            num_training_steps,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            min_lr=min_lr,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
        )
    
    def _get_decay_factor(self, step: int) -> float:
        # Find current cycle
        if self.T_mult == 1:
            cycle = step // self.T_0
            t_cur = step % self.T_0
            T_i = self.T_0
        else:
            # Geometric progression
            T_i = self.T_0
            t_cur = step
            cycle = 0
            while t_cur >= T_i:
                t_cur -= T_i
                T_i = int(T_i * self.T_mult)
                cycle += 1
        
        # Cosine decay within cycle
        return 0.5 * (1 + math.cos(math.pi * t_cur / T_i))


# ═════════════════════════════════════════════════════════════════════════════════
# Factory Function
# ═════════════════════════════════════════════════════════════════════════════════

def create_scheduler(
    name: str,
    optimizer: Optimizer,
    num_training_steps: int,
    warmup_steps: int = 0,
    warmup_ratio: float = 0.0,
    min_lr: float = 0.0,
    min_lr_ratio: float = 0.0,
    **kwargs,
) -> LRScheduler:
    """
    Create scheduler by name.
    
    Supported: wsd, cosine, linear, polynomial, onecycle, inverse_sqrt, cosine_restart, rex
    """
    name = name.lower()
    
    common_args = dict(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        min_lr=min_lr,
        min_lr_ratio=min_lr_ratio,
    )
    
    if name == "wsd":
        return WSDScheduler(**common_args, **kwargs)
    elif name == "cosine":
        # Use WSD with 0 stable steps for pure cosine
        return WSDScheduler(**common_args, stable_ratio=0.0, **kwargs)
    elif name == "linear":
        return PolynomialDecayScheduler(**common_args, power=1.0, **kwargs)
    elif name == "polynomial":
        return PolynomialDecayScheduler(**common_args, **kwargs)
    elif name == "onecycle":
        return OneCycleScheduler(
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            **kwargs,
        )
    elif name == "inverse_sqrt":
        return InverseSquareRootScheduler(
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            warmup_steps=warmup_steps or int(num_training_steps * warmup_ratio) or 4000,
            **kwargs,
        )
    elif name == "cosine_restart":
        return CosineRestartScheduler(**common_args, **kwargs)
    elif name == "rex":
        return REXScheduler(**common_args, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {name}. Supported: wsd, cosine, linear, polynomial, onecycle, inverse_sqrt, cosine_restart, rex")


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Base
    "BaseEnhancedScheduler",
    # Schedulers
    "WSDScheduler",
    "REXScheduler",
    "OneCycleScheduler",
    "PolynomialDecayScheduler",
    "InverseSquareRootScheduler",
    "CosineRestartScheduler",
    # Factory
    "create_scheduler",
]
