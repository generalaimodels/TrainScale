# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer Module - Base Optimizer
# ════════════════════════════════════════════════════════════════════════════════
# Abstract base optimizer with Triton kernel fusion support.
# Designed for above-SOTA performance with hardware-agnostic execution.
#
# Key Features:
# - Unified interface for all optimizers
# - Memory-aligned state buffers for cache efficiency
# - Gradient clipping integration
# - Triton kernel dispatch for GPU acceleration
# - Full state serialization for checkpointing
#
# Complexity Analysis:
# - step(): O(params) time, O(1) additional space
# - state_dict()/load_state_dict(): O(params) time and space
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.nn as nn
from torch import Tensor

from data_pipeline.trainer.core.types import (
    CACHE_LINE_SIZE,
    GradientInfo,
    OptimizerConfig,
)
from data_pipeline.trainer.core.errors import (
    GradientOverflowError,
    OptimizationError,
)

# ═════════════════════════════════════════════════════════════════════════════════
# Type Definitions
# ═════════════════════════════════════════════════════════════════════════════════

ParamT = TypeVar("ParamT", bound=Tensor)
StateT = TypeVar("StateT", bound=Dict[str, Any])

# Parameter group type alias
ParamGroup = Dict[str, Any]

# State dictionary type alias
OptimizerState = Dict[str, Any]


# ═════════════════════════════════════════════════════════════════════════════════
# Gradient Clipping Utilities
# ═════════════════════════════════════════════════════════════════════════════════

def compute_grad_norm(
    parameters: Iterator[Tensor],
    norm_type: float = 2.0,
) -> Tuple[float, float, float]:
    """
    Compute gradient norms across all parameters.
    
    Time Complexity: O(total_params)
    Space Complexity: O(1) auxiliary
    
    Args:
        parameters: Iterator of parameters with gradients
        norm_type: Type of norm (1, 2, or inf)
        
    Returns:
        Tuple of (global_norm, max_norm, min_norm)
    """
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad.detach())
    
    if not grads:
        return 0.0, 0.0, 0.0
    
    if norm_type == float("inf"):
        norms = [g.abs().max().item() for g in grads]
        global_norm = max(norms)
    else:
        total_norm = torch.zeros(1, device=grads[0].device, dtype=grads[0].dtype)
        for g in grads:
            param_norm = g.norm(norm_type)
            total_norm += param_norm ** norm_type
        global_norm = total_norm.pow(1.0 / norm_type).item()
        norms = [g.norm(norm_type).item() for g in grads]
    
    return global_norm, max(norms) if norms else 0.0, min(norms) if norms else 0.0


def clip_grad_norm_(
    parameters: Iterator[Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = True,
) -> GradientInfo:
    """
    Clip gradients by global norm in-place.
    
    This is a fused implementation that computes norm and clips in
    minimal passes over parameters.
    
    Time Complexity: O(total_params)
    Space Complexity: O(num_params) for norm storage
    
    Args:
        parameters: Iterator of parameters with gradients
        max_norm: Maximum gradient norm
        norm_type: Type of norm (1, 2, or inf)
        error_if_nonfinite: Raise error if NaN/Inf detected
        
    Returns:
        GradientInfo with clipping statistics
        
    Raises:
        GradientOverflowError: If gradients contain NaN/Inf and error_if_nonfinite=True
    """
    params = list(parameters)
    grads = [p.grad for p in params if p.grad is not None]
    
    if not grads:
        return GradientInfo(
            global_norm=0.0,
            max_norm=0.0,
            min_norm=0.0,
            clipped=False,
            overflow=False,
        )
    
    device = grads[0].device
    dtype = grads[0].dtype
    
    # Compute total norm
    if norm_type == float("inf"):
        norms = torch.stack([g.abs().max().to(device) for g in grads])
        total_norm = norms.max()
    else:
        norms = torch.stack([g.norm(norm_type).to(device) for g in grads])
        total_norm = norms.norm(norm_type)
    
    # Check for NaN/Inf
    overflow = not torch.isfinite(total_norm).item()
    if overflow and error_if_nonfinite:
        raise GradientOverflowError(
            message="Gradient contains NaN or Inf",
            grad_norm=float("inf") if overflow else total_norm.item(),
            remediation="Reduce learning rate or enable gradient clipping"
        )
    
    # Clip if necessary
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    clipped = clip_coef_clamped.item() < 1.0
    
    if clipped:
        for g in grads:
            g.mul_(clip_coef_clamped.to(g.device, g.dtype))
    
    return GradientInfo(
        global_norm=total_norm.item(),
        max_norm=norms.max().item(),
        min_norm=norms.min().item(),
        clipped=clipped,
        overflow=overflow,
    )


def clip_grad_value_(
    parameters: Iterator[Tensor],
    clip_value: float,
) -> GradientInfo:
    """
    Clip gradients by value (element-wise) in-place.
    
    Time Complexity: O(total_params)
    Space Complexity: O(1)
    
    Args:
        parameters: Iterator of parameters with gradients
        clip_value: Maximum absolute gradient value
        
    Returns:
        GradientInfo with clipping statistics
    """
    params = list(parameters)
    grads = [p.grad for p in params if p.grad is not None]
    
    if not grads:
        return GradientInfo(
            global_norm=0.0,
            max_norm=0.0,
            min_norm=0.0,
            clipped=False,
            overflow=False,
        )
    
    max_val = 0.0
    min_val = float("inf")
    clipped = False
    
    for g in grads:
        max_elem = g.abs().max().item()
        min_elem = g.abs().min().item()
        max_val = max(max_val, max_elem)
        min_val = min(min_val, min_elem)
        
        if max_elem > clip_value:
            clipped = True
            g.clamp_(-clip_value, clip_value)
    
    # Compute global norm after clipping
    total_norm = torch.stack([g.norm(2.0) for g in grads]).norm(2.0).item()
    
    return GradientInfo(
        global_norm=total_norm,
        max_norm=max_val,
        min_norm=min_val,
        clipped=clipped,
        overflow=not math.isfinite(max_val),
    )


# ═════════════════════════════════════════════════════════════════════════════════
# Base Optimizer Interface
# ═════════════════════════════════════════════════════════════════════════════════

class BaseOptimizer(abc.ABC):
    """
    Abstract base optimizer for SOTA training.
    
    Provides unified interface for:
    - Parameter group management
    - Gradient clipping (norm and value)
    - State serialization
    - Triton kernel dispatch
    
    Subclasses must implement:
    - _init_state(): Initialize optimizer state for a parameter
    - _step_param(): Apply update to single parameter
    - _step_fused(): Apply fused update to parameter group (Triton)
    
    Memory Layout:
    State buffers are aligned to cache line boundaries (64 bytes)
    for optimal memory access patterns.
    """
    
    def __init__(
        self,
        params: Union[Iterator[Tensor], Iterator[ParamGroup]],
        defaults: Dict[str, Any],
        *,
        use_triton: bool = True,
    ):
        """
        Initialize optimizer.
        
        Args:
            params: Parameters or parameter groups to optimize
            defaults: Default hyperparameters for parameter groups
            use_triton: Enable Triton kernel fusion when available
        """
        self.defaults = defaults
        self.use_triton = use_triton and self._triton_available()
        
        # Convert params to parameter groups
        self.param_groups: List[ParamGroup] = []
        self._add_param_groups(params, defaults)
        
        # Optimizer state per parameter
        self.state: Dict[Tensor, Dict[str, Any]] = {}
        
        # Step counter
        self._step_count = 0
    
    def _add_param_groups(
        self,
        params: Union[Iterator[Tensor], Iterator[ParamGroup]],
        defaults: Dict[str, Any],
    ) -> None:
        """Add parameter groups from iterator."""
        params_list = list(params)
        
        if not params_list:
            raise OptimizationError(
                message="Optimizer received empty parameter list"
            )
        
        # Check if it's a list of parameter groups or raw parameters
        if isinstance(params_list[0], dict):
            # Already parameter groups
            for group in params_list:
                self._add_param_group(group)
        else:
            # Raw parameters, wrap in single group
            self._add_param_group({"params": params_list})
    
    def _add_param_group(self, param_group: ParamGroup) -> None:
        """Add a single parameter group."""
        params = param_group.get("params", [])
        if isinstance(params, Tensor):
            params = [params]
        
        # Validate parameters
        params = list(params)
        for p in params:
            if not isinstance(p, Tensor):
                raise OptimizationError(
                    message=f"Optimizer expected Tensor, got {type(p).__name__}"
                )
            if not p.requires_grad:
                raise OptimizationError(
                    message="Optimizer received parameter without requires_grad=True"
                )
        
        # Merge defaults with group-specific params
        group = {**self.defaults}
        for key, value in param_group.items():
            if key != "params":
                group[key] = value
        group["params"] = params
        
        self.param_groups.append(group)
    
    @abc.abstractmethod
    def _init_state(self, param: Tensor, group: ParamGroup) -> Dict[str, Any]:
        """
        Initialize optimizer state for a parameter.
        
        Called lazily on first update.
        
        Args:
            param: Parameter tensor
            group: Parameter group configuration
            
        Returns:
            Initial state dictionary for parameter
        """
        pass
    
    @abc.abstractmethod
    def _step_param(
        self,
        param: Tensor,
        grad: Tensor,
        state: Dict[str, Any],
        group: ParamGroup,
    ) -> None:
        """
        Apply optimizer update to single parameter.
        
        Pure PyTorch implementation for CPU/fallback.
        
        Args:
            param: Parameter tensor (modified in-place)
            grad: Gradient tensor
            state: Optimizer state for parameter (modified in-place)
            group: Parameter group configuration
        """
        pass
    
    def _step_fused(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        states: List[Dict[str, Any]],
        group: ParamGroup,
    ) -> None:
        """
        Apply fused optimizer update to parameter group.
        
        Default implementation falls back to per-parameter updates.
        Override in subclass for Triton kernel fusion.
        
        Args:
            params: List of parameter tensors
            grads: List of gradient tensors
            states: List of optimizer states
            group: Parameter group configuration
        """
        for param, grad, state in zip(params, grads, states):
            self._step_param(param, grad, state, group)
    
    def _triton_available(self) -> bool:
        """Check if Triton kernels are available."""
        try:
            import triton
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Zero out gradients for all parameters.
        
        Args:
            set_to_none: If True, set gradients to None (memory efficient)
                        If False, set gradients to zero tensor
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform single optimization step.
        
        Args:
            closure: Optional closure that reevaluates model and returns loss
            
        Returns:
            Loss value if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self._step_count += 1
        
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            states = []
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Lazy state initialization
                if p not in self.state:
                    self.state[p] = self._init_state(p, group)
                
                params_with_grad.append(p)
                grads.append(p.grad)
                states.append(self.state[p])
            
            if not params_with_grad:
                continue
            
            # Choose update path
            if self.use_triton and params_with_grad[0].is_cuda:
                self._step_fused(params_with_grad, grads, states, group)
            else:
                for param, grad, state in zip(params_with_grad, grads, states):
                    self._step_param(param, grad, state, group)
        
        return loss
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Return optimizer state as dictionary.
        
        Format compatible with torch.save().
        """
        # Pack state with parameter indices
        packed_state = {}
        param_to_idx = {}
        
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, p in enumerate(group["params"]):
                param_to_idx[id(p)] = (group_idx, param_idx)
                if p in self.state:
                    packed_state[(group_idx, param_idx)] = self.state[p]
        
        # Extract hyperparameters from groups
        param_groups = [
            {k: v for k, v in group.items() if k != "params"}
            for group in self.param_groups
        ]
        
        return {
            "state": packed_state,
            "param_groups": param_groups,
            "step_count": self._step_count,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load optimizer state from dictionary.
        
        Args:
            state_dict: State dictionary from state_dict()
        """
        # Restore step count
        self._step_count = state_dict.get("step_count", 0)
        
        # Restore parameter group hyperparameters
        saved_groups = state_dict.get("param_groups", [])
        if len(saved_groups) != len(self.param_groups):
            raise OptimizationError(
                message=f"Loaded state has {len(saved_groups)} param groups, "
                       f"but optimizer has {len(self.param_groups)}"
            )
        
        for group, saved in zip(self.param_groups, saved_groups):
            for key, val in saved.items():
                if key in group:
                    group[key] = val
        
        # Restore parameter states
        packed_state = state_dict.get("state", {})
        
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, p in enumerate(group["params"]):
                key = (group_idx, param_idx)
                if key in packed_state:
                    # Move tensors to parameter device
                    state = packed_state[key]
                    for k, v in state.items():
                        if isinstance(v, Tensor):
                            state[k] = v.to(p.device, p.dtype)
                    self.state[p] = state
    
    @property
    def step_count(self) -> int:
        """Current optimization step count."""
        return self._step_count
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"lr={self.defaults.get('lr', 'N/A')}, "
            f"param_groups={len(self.param_groups)}, "
            f"triton={self.use_triton})"
        )


# ═════════════════════════════════════════════════════════════════════════════════
# Export
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    "BaseOptimizer",
    "compute_grad_norm",
    "clip_grad_norm_",
    "clip_grad_value_",
    "ParamGroup",
    "OptimizerState",
]
