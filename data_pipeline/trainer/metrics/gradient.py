# ════════════════════════════════════════════════════════════════════════════════
# SOTA Training Metrics - Gradient Tracking
# ════════════════════════════════════════════════════════════════════════════════
# Gradient health monitoring: norm, variance, overflow detection, histograms.
#
# Features:
#   - Gradient norm tracking (L2 norm)
#   - Per-layer gradient statistics
#   - NaN/Inf detection for training stability
#   - Gradient histogram for debugging
#   - Distributed-aware aggregation
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

# ════════════════════════════════════════════════════════════════════════════════
# Gradient Tracker
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class GradientTracker:
    """
    SOTA gradient health monitoring for training stability.
    
    Tracks gradient norms, detects NaN/Inf values, and provides
    statistical analysis for debugging training issues.
    
    Example:
        >>> tracker = GradientTracker()
        >>> loss.backward()
        >>> tracker.update(model.parameters())
        >>> print(f"Grad Norm: {tracker.compute_norm():.4f}")
        >>> if tracker.has_nan_or_inf():
        ...     print("WARNING: Gradient overflow detected!")
    
    Attributes:
        max_norm: Maximum expected gradient norm (for anomaly detection)
        ema_decay: Decay factor for EMA of gradient norm
    """
    
    max_norm: float = 10.0
    ema_decay: float = 0.99
    
    # ───────────────────────────────────────────────────────────────────────────
    # Internal State
    # ───────────────────────────────────────────────────────────────────────────
    _total_norm: float = field(default=0.0, repr=False)
    _ema_norm: Optional[float] = field(default=None, repr=False)
    _step_count: int = field(default=0, repr=False)
    _has_nan: bool = field(default=False, repr=False)
    _has_inf: bool = field(default=False, repr=False)
    _param_count: int = field(default=0, repr=False)
    _grad_sum: float = field(default=0.0, repr=False)
    _grad_sq_sum: float = field(default=0.0, repr=False)
    _history: List[float] = field(default_factory=list, repr=False)
    _layer_norms: Dict[str, float] = field(default_factory=dict, repr=False)
    
    def __post_init__(self) -> None:
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulated values."""
        self._total_norm = 0.0
        self._ema_norm = None
        self._step_count = 0
        self._has_nan = False
        self._has_inf = False
        self._param_count = 0
        self._grad_sum = 0.0
        self._grad_sq_sum = 0.0
        self._history = []
        self._layer_norms = {}
    
    def update(
        self,
        parameters: Iterator[Parameter],
        named_parameters: Optional[Iterator[Tuple[str, Parameter]]] = None,
    ) -> float:
        """
        Update gradient statistics from model parameters.
        
        Args:
            parameters: Iterator of model parameters with gradients
            named_parameters: Optional named parameters for per-layer tracking
        
        Returns:
            Total gradient norm for this update
        """
        total_norm_sq = 0.0
        param_count = 0
        grad_sum = 0.0
        grad_sq_sum = 0.0
        has_nan = False
        has_inf = False
        
        # Use named_parameters if provided for layer tracking
        if named_parameters is not None:
            params = list(named_parameters)
        else:
            params = [(None, p) for p in parameters]
        
        layer_norms = {}
        
        for name, param in params:
            if param.grad is None:
                continue
            
            grad = param.grad.detach()
            
            # Check for NaN/Inf
            if torch.isnan(grad).any():
                has_nan = True
            if torch.isinf(grad).any():
                has_inf = True
            
            # Compute norm
            param_norm_sq = grad.pow(2).sum().item()
            total_norm_sq += param_norm_sq
            
            # Track per-layer norms
            if name is not None:
                layer_norms[name] = math.sqrt(param_norm_sq)
            
            # Statistics
            grad_sum += grad.sum().item()
            grad_sq_sum += param_norm_sq
            param_count += grad.numel()
        
        total_norm = math.sqrt(total_norm_sq)
        
        # Update state
        self._total_norm = total_norm
        self._has_nan = has_nan
        self._has_inf = has_inf
        self._param_count = param_count
        self._grad_sum = grad_sum
        self._grad_sq_sum = grad_sq_sum
        self._layer_norms = layer_norms
        self._step_count += 1
        
        # Update EMA
        if self._ema_norm is None:
            self._ema_norm = total_norm
        else:
            self._ema_norm = (
                self.ema_decay * self._ema_norm +
                (1 - self.ema_decay) * total_norm
            )
        
        # Store history (capped)
        if len(self._history) < 1000:
            self._history.append(total_norm)
        
        return total_norm
    
    def update_from_model(self, model: nn.Module) -> float:
        """
        Update gradient statistics from model.
        
        Args:
            model: PyTorch model with computed gradients
        
        Returns:
            Total gradient norm
        """
        return self.update(
            parameters=model.parameters(),
            named_parameters=model.named_parameters(),
        )
    
    def compute_norm(self) -> float:
        """
        Get the most recent gradient norm.
        
        Returns:
            L2 norm of gradients
        """
        return self._total_norm
    
    def compute_ema_norm(self) -> float:
        """
        Get EMA of gradient norm.
        
        Returns:
            Exponential moving average of gradient norm
        """
        return self._ema_norm if self._ema_norm is not None else 0.0
    
    def compute_mean(self) -> float:
        """
        Compute mean gradient value.
        
        Returns:
            Mean gradient across all parameters
        """
        if self._param_count == 0:
            return 0.0
        return self._grad_sum / self._param_count
    
    def compute_variance(self) -> float:
        """
        Compute gradient variance.
        
        Returns:
            Variance of gradient values
        """
        if self._param_count == 0:
            return 0.0
        mean = self.compute_mean()
        return (self._grad_sq_sum / self._param_count) - (mean ** 2)
    
    def compute_std(self) -> float:
        """
        Compute gradient standard deviation.
        
        Returns:
            Standard deviation of gradient values
        """
        return math.sqrt(max(0.0, self.compute_variance()))
    
    def has_nan_or_inf(self) -> bool:
        """
        Check if gradients contain NaN or Inf.
        
        Returns:
            True if any gradient contains NaN or Inf
        """
        return self._has_nan or self._has_inf
    
    def has_nan(self) -> bool:
        """Check for NaN in gradients."""
        return self._has_nan
    
    def has_inf(self) -> bool:
        """Check for Inf in gradients."""
        return self._has_inf
    
    def is_anomaly(self) -> bool:
        """
        Check if gradient norm exceeds expected maximum.
        
        Returns:
            True if norm > max_norm or NaN/Inf detected
        """
        return self.has_nan_or_inf() or self._total_norm > self.max_norm
    
    def get_layer_norms(self) -> Dict[str, float]:
        """
        Get per-layer gradient norms.
        
        Returns:
            Dict mapping layer name to gradient norm
        """
        return self._layer_norms.copy()
    
    def get_top_layers(self, k: int = 5) -> List[Tuple[str, float]]:
        """
        Get layers with largest gradient norms.
        
        Args:
            k: Number of top layers to return
        
        Returns:
            List of (layer_name, norm) tuples sorted by norm descending
        """
        sorted_layers = sorted(
            self._layer_norms.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_layers[:k]
    
    def histogram(self) -> Dict[str, float]:
        """
        Get gradient norm histogram statistics.
        
        Returns:
            Dict with p10, p25, p50, p75, p90, max values
        """
        if not self._history:
            return {"p10": 0, "p25": 0, "p50": 0, "p75": 0, "p90": 0, "max": 0}
        
        sorted_h = sorted(self._history)
        n = len(sorted_h)
        
        def percentile(p: float) -> float:
            idx = int(p * n / 100)
            return sorted_h[min(idx, n - 1)]
        
        return {
            "p10": percentile(10),
            "p25": percentile(25),
            "p50": percentile(50),
            "p75": percentile(75),
            "p90": percentile(90),
            "max": sorted_h[-1],
        }
    
    @property
    def step_count(self) -> int:
        """Get number of update calls."""
        return self._step_count
    
    def log_dict(self) -> Dict[str, float]:
        """
        Get metrics as dictionary for logging.
        
        Returns:
            Dict with gradient metrics
        """
        return {
            "gradient/norm": self._total_norm,
            "gradient/norm_ema": self.compute_ema_norm(),
            "gradient/mean": self.compute_mean(),
            "gradient/std": self.compute_std(),
            "gradient/has_nan": float(self._has_nan),
            "gradient/has_inf": float(self._has_inf),
        }
    
    def state_dict(self) -> Dict:
        """Get serializable state for checkpointing."""
        return {
            "ema_norm": self._ema_norm,
            "step_count": self._step_count,
            "history": self._history,
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """Load state from checkpoint."""
        self._ema_norm = state.get("ema_norm")
        self._step_count = state.get("step_count", 0)
        self._history = state.get("history", [])


# ════════════════════════════════════════════════════════════════════════════════
# Exports
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    "GradientTracker",
]
