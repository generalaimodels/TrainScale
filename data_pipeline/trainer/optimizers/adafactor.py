# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer Module - AdaFactor Optimizer
# ════════════════════════════════════════════════════════════════════════════════
# Memory-efficient optimizer using factorized second moments.
# 
# Key Features:
# 1. ~50% memory reduction vs AdamW (no full second moment)
# 2. Row/column factorization for 2D+ tensors
# 3. Optional relative step size (no learning rate required)
# 4. Scale-invariant updates
#
# Memory Savings:
# - AdamW: 2x params (m, v)
# - AdaFactor: ~1.5x params (m, row_rms + col_rms)
#
# Reference: Adafactor: Adaptive Learning Rates with Sublinear Memory Cost
# https://arxiv.org/abs/1804.04235
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch import Tensor

from data_pipeline.trainer.optimizers.base import (
    BaseOptimizer,
    ParamGroup,
)

# ═════════════════════════════════════════════════════════════════════════════════
# AdaFactor Optimizer Implementation
# ═════════════════════════════════════════════════════════════════════════════════

class AdaFactor(BaseOptimizer):
    """
    AdaFactor optimizer with factorized second moments.
    
    AdaFactor achieves significant memory savings by factorizing
    the second moment matrix into row and column vectors for 2D+ tensors.
    
    Memory Comparison (for weight matrix of shape [d1, d2]):
    - AdamW: d1 * d2 (full matrix)
    - AdaFactor: d1 + d2 (row + column vectors)
    
    Key Innovations:
    1. Factorized second moment: v_rc = row_rms * col_rms
    2. Relative step size: lr = max(ε₂, RMS(θ)) * ρ_t
    3. Update clipping: prevents large updates
    
    Algorithm:
    ```
    row_rms = β₂ * row_rms + (1 - β₂) * mean(g², dim=-1)
    col_rms = β₂ * col_rms + (1 - β₂) * mean(g², dim=-2)
    v = outer(row_rms, col_rms) / mean(row_rms)  # Approximated v
    u = g / √v
    u = u / max(1, RMS(u) / d)  # Update clipping
    m = β₁ * m + (1 - β₁) * u   # Optional momentum
    θ = θ - lr * m
    ```
    
    Args:
        params: Parameters to optimize
        lr: Learning rate (None for relative step size mode)
        eps: Regularization constants (eps1, eps2) = (1e-30, 1e-3)
        clip_threshold: Threshold for update clipping (default: 1.0)
        decay_rate: Coefficient for second moment decay (default: -0.8)
        beta1: Coefficient for first moment (None = no momentum)
        weight_decay: Weight decay coefficient (default: 0.0)
        scale_parameter: Scale learning rate by RMS of weights (default: True)
        relative_step: Use relative step size (default: True)
        warmup_init: Use warmup initialization (default: False)
        use_triton: Enable Triton kernels (default: True)
        
    Example:
        ```python
        # Memory-efficient training (recommended for large models)
        optimizer = AdaFactor(
            model.parameters(),
            lr=None,  # Use relative step size
            relative_step=True,
            scale_parameter=True,
        )
        
        # Or with fixed learning rate
        optimizer = AdaFactor(
            model.parameters(),
            lr=1e-4,
            relative_step=False,
        )
        ```
    """
    
    def __init__(
        self,
        params: Iterator[Tensor],
        lr: Optional[float] = None,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        *,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
        use_triton: bool = True,
    ):
        # Validate hyperparameters
        if lr is not None and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if beta1 is not None and not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        
        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "beta1": beta1,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
        }
        
        super().__init__(params, defaults, use_triton=use_triton)
    
    @staticmethod
    def _get_lr(group: ParamGroup, state: Dict[str, Any]) -> float:
        """
        Compute learning rate for current step.
        
        With relative_step=True:
            lr = min(1/√step, RMS(θ)) * relative_decay
        """
        relative_step = group["relative_step"]
        warmup_init = group["warmup_init"]
        
        if group["lr"] is not None:
            return group["lr"]
        
        step = state["step"]
        
        # Base relative step size
        rel_step = 1.0 / math.sqrt(max(step, 1))
        
        if warmup_init:
            # Gradual warmup
            rel_step = min(rel_step, step / 10000.0)
        
        return rel_step
    
    @staticmethod
    def _get_rms(tensor: Tensor) -> float:
        """Compute root mean square of tensor."""
        return tensor.pow(2).mean().sqrt().item()
    
    @staticmethod
    def _rho(step: int, decay_rate: float) -> float:
        """
        Compute second moment decay rate for current step.
        
        ρ_t = min(1 - 1/(t+1), ρ)
        """
        return min(1.0 - math.pow(step + 1, decay_rate), 0.999)
    
    def _init_state(self, param: Tensor, group: ParamGroup) -> Dict[str, Any]:
        """
        Initialize optimizer state for parameter.
        
        For 2D+ tensors, allocates factorized row/column RMS.
        For 1D tensors, allocates full second moment.
        """
        state: Dict[str, Any] = {"step": 0}
        
        shape = param.shape
        
        # Factorization applies to 2D+ tensors with both dims > 1
        factored = len(shape) >= 2 and shape[-2] > 1 and shape[-1] > 1
        state["factored"] = factored
        
        if factored:
            # Factorized second moment: row and column RMS
            # For shape [d1, d2], allocate [d1] and [d2]
            state["exp_avg_sq_row"] = torch.zeros(
                shape[:-1], dtype=param.dtype, device=param.device
            )
            state["exp_avg_sq_col"] = torch.zeros(
                shape[:-2] + (shape[-1],), dtype=param.dtype, device=param.device
            )
        else:
            # Full second moment for 1D or small tensors
            state["exp_avg_sq"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
        
        # First moment (momentum), only if beta1 is set
        if group["beta1"] is not None:
            state["exp_avg"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
        
        return state
    
    def _step_param(
        self,
        param: Tensor,
        grad: Tensor,
        state: Dict[str, Any],
        group: ParamGroup,
    ) -> None:
        """
        Apply AdaFactor update to single parameter.
        
        Handles both factorized and non-factorized cases.
        """
        # Unpack hyperparameters
        eps1, eps2 = group["eps"]
        clip_threshold = group["clip_threshold"]
        decay_rate = group["decay_rate"]
        beta1 = group["beta1"]
        weight_decay = group["weight_decay"]
        scale_parameter = group["scale_parameter"]
        
        # Increment step
        state["step"] += 1
        step = state["step"]
        
        # Compute learning rate
        lr = self._get_lr(group, state)
        
        # Scale by parameter RMS
        if scale_parameter:
            param_rms = max(self._get_rms(param), eps2)
            lr = lr * param_rms
        
        # Second moment decay
        rho = self._rho(step, decay_rate)
        
        # Square of gradient
        grad_sq = grad.pow(2).add_(eps1)
        
        factored = state["factored"]
        
        if factored:
            # Factorized update
            exp_avg_sq_row = state["exp_avg_sq_row"]
            exp_avg_sq_col = state["exp_avg_sq_col"]
            
            # Update row RMS: mean over last dim
            exp_avg_sq_row.mul_(rho).add_(
                grad_sq.mean(dim=-1), alpha=1.0 - rho
            )
            
            # Update col RMS: mean over second-to-last dim
            exp_avg_sq_col.mul_(rho).add_(
                grad_sq.mean(dim=-2), alpha=1.0 - rho
            )
            
            # Reconstruct approximate second moment
            # v = outer(row, col) / mean(row)
            row_mean = exp_avg_sq_row.mean(dim=-1, keepdim=True)
            v = exp_avg_sq_row.unsqueeze(-1) * exp_avg_sq_col.unsqueeze(-2)
            v = v / row_mean.unsqueeze(-1).clamp(min=eps1)
        else:
            # Full second moment
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg_sq.mul_(rho).add_(grad_sq, alpha=1.0 - rho)
            v = exp_avg_sq
        
        # Compute update: u = g / √v
        u = grad / v.sqrt().clamp(min=eps1)
        
        # Update clipping
        u_rms = self._get_rms(u)
        d = max(1.0, u_rms / clip_threshold)
        u = u / d
        
        # First moment (momentum)
        if beta1 is not None:
            exp_avg = state["exp_avg"]
            exp_avg.mul_(beta1).add_(u, alpha=1.0 - beta1)
            u = exp_avg
        
        # Apply weight decay
        if weight_decay != 0:
            param.add_(param, alpha=-weight_decay * lr)
        
        # Apply update
        param.add_(u, alpha=-lr)
    
    def _step_fused(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        states: List[Dict[str, Any]],
        group: ParamGroup,
    ) -> None:
        """
        Apply fused AdaFactor update.
        
        Falls back to per-parameter updates due to varying shapes.
        """
        # AdaFactor has complex factorization logic that's harder to fuse
        # For now, use per-parameter updates
        for param, grad, state in zip(params, grads, states):
            self._step_param(param, grad, state, group)


# ═════════════════════════════════════════════════════════════════════════════════
# Factory Function
# ═════════════════════════════════════════════════════════════════════════════════

def create_adafactor(
    params: Iterator[Tensor],
    lr: Optional[float] = None,
    eps: Tuple[float, float] = (1e-30, 1e-3),
    clip_threshold: float = 1.0,
    decay_rate: float = -0.8,
    beta1: Optional[float] = None,
    weight_decay: float = 0.0,
    scale_parameter: bool = True,
    relative_step: bool = True,
    warmup_init: bool = False,
) -> AdaFactor:
    """
    Factory function for AdaFactor optimizer.
    
    Recommended settings:
    
    Memory-constrained (T5-style):
        lr=None, relative_step=True, scale_parameter=True
        
    With fixed LR:
        lr=1e-4, relative_step=False, scale_parameter=False
        
    With momentum:
        beta1=0.9
    """
    return AdaFactor(
        params,
        lr=lr,
        eps=eps,
        clip_threshold=clip_threshold,
        decay_rate=decay_rate,
        beta1=beta1,
        weight_decay=weight_decay,
        scale_parameter=scale_parameter,
        relative_step=relative_step,
        warmup_init=warmup_init,
    )


# ═════════════════════════════════════════════════════════════════════════════════
# Export
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    "AdaFactor",
    "create_adafactor",
]
