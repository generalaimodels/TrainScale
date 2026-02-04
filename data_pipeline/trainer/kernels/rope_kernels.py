# ════════════════════════════════════════════════════════════════════════════════
# SOTA RoPE Embedding Kernels - Above-Unsloth Implementation
# ════════════════════════════════════════════════════════════════════════════════
# Engineering: Premium AI Accelerator Kernel Lead Developer
# Target: Maximum throughput, numerical precision, zero memory overhead
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ═════════════════════════════════════════════════════════════════════════════════
# Hardware Detection & Triton Setup
# ═════════════════════════════════════════════════════════════════════════════════

_TRITON_AVAILABLE: bool = False
_DEVICE_PROPERTIES: Dict[int, Dict[str, Any]] = {}

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    triton = None
    tl = None


class RoPEScalingType(Enum):
    """RoPE scaling method enumeration."""
    NONE = auto()
    LINEAR = auto()       # Position Interpolation (PI)
    NTK = auto()          # NTK-aware scaling
    YARN = auto()         # YaRN scaling
    DYNAMIC_NTK = auto()  # Dynamic NTK
    LONGROPE = auto()     # LongRoPE scaling


@dataclass
class RoPEConfig:
    """Comprehensive RoPE configuration."""
    dim: int
    max_seq_len: int = 8192
    base: float = 10000.0
    
    # Scaling configuration
    scaling_type: RoPEScalingType = RoPEScalingType.NONE
    scaling_factor: float = 1.0
    
    # YaRN specific parameters
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float = 1.0
    yarn_mscale_all_dim: float = 0.0
    
    # NTK parameters
    ntk_alpha: Optional[float] = None
    
    # LongRoPE parameters
    longrope_short_factor: Optional[List[float]] = None
    longrope_long_factor: Optional[List[float]] = None
    
    # Hardware optimization
    use_fp32_precision: bool = True
    cache_freqs: bool = True
    
    # Kernel configuration
    block_size: int = 64
    num_warps: int = 4
    group_size: int = 4


@lru_cache(maxsize=8)
def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
    """Get compute capability with caching."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability(device_id)
    return (0, 0)


def supports_bf16(device_id: int = 0) -> bool:
    """Check BF16 support (Ampere+)."""
    major, _ = get_device_capability(device_id)
    return major >= 8


def get_optimal_block_size(head_dim: int) -> int:
    """Determine optimal block size based on head dimension."""
    if head_dim <= 64:
        return 64
    elif head_dim <= 128:
        return 128
    else:
        return 256


# ═════════════════════════════════════════════════════════════════════════════════
# RoPE Frequency Computation
# ═════════════════════════════════════════════════════════════════════════════════

class RoPEFrequencyCache:
    """
    Cached RoPE frequency computation with multiple scaling methods.
    
    Avoids recomputation across forward passes.
    """
    
    _cache: Dict[Tuple, Tuple[Tensor, Tensor]] = {}
    
    @classmethod
    def clear_cache(cls):
        """Clear frequency cache."""
        cls._cache.clear()
    
    @classmethod
    def get_freqs(
        cls,
        config: RoPEConfig,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        """Get or compute cached frequencies."""
        cache_key = (
            config.dim, config.base, config.scaling_type,
            config.scaling_factor, seq_len, device, dtype,
        )
        
        if cache_key not in cls._cache or not config.cache_freqs:
            cos, sin = compute_rope_frequencies(config, seq_len, device, dtype)
            if config.cache_freqs:
                cls._cache[cache_key] = (cos, sin)
            return cos, sin
        
        return cls._cache[cache_key]


def compute_rope_frequencies(
    config: RoPEConfig,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """
    Compute RoPE frequencies with comprehensive scaling support.
    
    Args:
        config: RoPE configuration
        seq_len: Sequence length
        device: Target device
        dtype: Target dtype
    
    Returns:
        cos: Cosine frequencies [seq_len, dim//2]
        sin: Sine frequencies [seq_len, dim//2]
    """
    dim = config.dim
    half_dim = dim // 2
    
    # Compute base inverse frequencies
    inv_freq = _compute_inv_freq(config, device)
    
    # Position indices
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    
    # Apply scaling to positions if needed
    if config.scaling_type == RoPEScalingType.LINEAR:
        t = t / config.scaling_factor
    
    # Outer product: [seq_len] x [dim//2] -> [seq_len, dim//2]
    freqs = torch.outer(t, inv_freq)
    
    # Apply YaRN mscale
    if config.scaling_type == RoPEScalingType.YARN:
        mscale = _compute_yarn_mscale(config)
        freqs = freqs * mscale
    
    # Compute cos/sin in FP32 for precision
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    
    # Convert to target dtype
    if dtype != torch.float32:
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    
    return cos, sin


def _compute_inv_freq(config: RoPEConfig, device: torch.device) -> Tensor:
    """Compute inverse frequencies with scaling."""
    dim = config.dim
    half_dim = dim // 2
    
    if config.scaling_type == RoPEScalingType.NONE:
        # Standard RoPE
        inv_freq = 1.0 / (config.base ** (
            torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
        ))
    
    elif config.scaling_type == RoPEScalingType.LINEAR:
        # Position Interpolation (PI)
        inv_freq = 1.0 / (config.base ** (
            torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
        ))
    
    elif config.scaling_type == RoPEScalingType.NTK:
        # NTK-aware scaling
        base = config.base * (
            config.scaling_factor ** (dim / (dim - 2))
        )
        inv_freq = 1.0 / (base ** (
            torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
        ))
    
    elif config.scaling_type == RoPEScalingType.DYNAMIC_NTK:
        # Dynamic NTK (adjusts based on sequence length)
        base = config.base * (
            (config.scaling_factor * config.max_seq_len / config.max_seq_len) - 
            (config.scaling_factor - 1)
        ) ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (
            torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
        ))
    
    elif config.scaling_type == RoPEScalingType.YARN:
        # YaRN scaling with frequency interpolation
        inv_freq = _compute_yarn_inv_freq(config, device)
    
    elif config.scaling_type == RoPEScalingType.LONGROPE:
        # LongRoPE with learned factors
        inv_freq = _compute_longrope_inv_freq(config, device)
    
    else:
        # Fallback to standard
        inv_freq = 1.0 / (config.base ** (
            torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
        ))
    
    return inv_freq


def _compute_yarn_inv_freq(config: RoPEConfig, device: torch.device) -> Tensor:
    """
    Compute YaRN inverse frequencies.
    
    YaRN uses frequency-dependent interpolation with smooth transitions.
    """
    dim = config.dim
    half_dim = dim // 2
    
    # Base inverse frequencies
    base_inv_freq = 1.0 / (config.base ** (
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
    ))
    
    # Compute wavelengths
    wavelengths = 2 * math.pi / base_inv_freq
    
    # Compute interpolation weights
    low_threshold = config.yarn_beta_fast
    high_threshold = config.yarn_beta_slow
    
    # Smooth interpolation between low and high frequencies
    ratio = config.max_seq_len / (2 * math.pi)
    low_freq = wavelengths / low_threshold
    high_freq = wavelengths / high_threshold
    
    # Ramp function for smooth transition
    ramp = (wavelengths - low_freq) / (high_freq - low_freq)
    ramp = torch.clamp(ramp, 0.0, 1.0)
    
    # Interpolated frequencies
    inv_freq = base_inv_freq * (1 - ramp) + (base_inv_freq / config.scaling_factor) * ramp
    
    return inv_freq


def _compute_yarn_mscale(config: RoPEConfig) -> float:
    """Compute YaRN magnitude scale."""
    if config.yarn_mscale_all_dim > 0:
        return config.yarn_mscale_all_dim
    
    if config.scaling_factor <= 1.0:
        return 1.0
    
    return 0.1 * math.log(config.scaling_factor) + 1.0


def _compute_longrope_inv_freq(config: RoPEConfig, device: torch.device) -> Tensor:
    """Compute LongRoPE inverse frequencies with learned factors."""
    dim = config.dim
    half_dim = dim // 2
    
    base_inv_freq = 1.0 / (config.base ** (
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
    ))
    
    if config.longrope_long_factor is not None:
        factors = torch.tensor(config.longrope_long_factor, device=device, dtype=torch.float32)
        factors = factors[:half_dim]
        inv_freq = base_inv_freq / factors
    else:
        inv_freq = base_inv_freq
    
    return inv_freq


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
    scaling_type: str = "none",
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """
    Precompute RoPE frequencies (cos, sin).
    
    High-level API for frequency computation.
    
    Args:
        dim: Head dimension
        max_seq_len: Maximum sequence length
        theta: RoPE base frequency
        scaling_factor: Scaling factor for extended context
        scaling_type: One of "none", "linear", "ntk", "yarn", "dynamic_ntk", "longrope"
        device: Target device
        dtype: Target dtype
        **kwargs: Additional scaling parameters
    
    Returns:
        cos: [max_seq_len, dim//2] cosine frequencies
        sin: [max_seq_len, dim//2] sine frequencies
    """
    # Map string to enum
    scaling_map = {
        "none": RoPEScalingType.NONE,
        "linear": RoPEScalingType.LINEAR,
        "pi": RoPEScalingType.LINEAR,
        "ntk": RoPEScalingType.NTK,
        "dynamic_ntk": RoPEScalingType.DYNAMIC_NTK,
        "yarn": RoPEScalingType.YARN,
        "longrope": RoPEScalingType.LONGROPE,
    }
    
    scaling_enum = scaling_map.get(scaling_type.lower(), RoPEScalingType.NONE)
    
    config = RoPEConfig(
        dim=dim,
        max_seq_len=max_seq_len,
        base=theta,
        scaling_type=scaling_enum,
        scaling_factor=scaling_factor,
        yarn_beta_fast=kwargs.get("yarn_beta_fast", 32.0),
        yarn_beta_slow=kwargs.get("yarn_beta_slow", 1.0),
        yarn_mscale=kwargs.get("yarn_mscale", 1.0),
        longrope_long_factor=kwargs.get("longrope_long_factor"),
        longrope_short_factor=kwargs.get("longrope_short_factor"),
    )
    
    if isinstance(device, str):
        device = torch.device(device)
    
    return compute_rope_frequencies(config, max_seq_len, device, dtype)


# ═════════════════════════════════════════════════════════════════════════════════
# Triton RoPE Kernels
# ═════════════════════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:

    @triton.jit
    def _rope_fwd_kernel(
        # Pointers
        Q_ptr, K_ptr,
        cos_ptr, sin_ptr,
        # Strides for Q
        Q_batch_stride, Q_seq_stride, Q_head_stride, Q_dim_stride,
        # Strides for K
        K_batch_stride, K_seq_stride, K_head_stride, K_dim_stride,
        # Frequency strides
        cos_seq_stride, cos_dim_stride,
        sin_seq_stride, sin_dim_stride,
        # Dimensions
        batch_size, seq_len, n_heads_Q, n_heads_K, head_dim,
        # Position offset (for KV-cache)
        position_offset,
        # Flags
        PROCESS_K: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        # Block sizes
        BLOCK_SEQ: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
    ):
        """
        Fused RoPE forward kernel for Q and K.
        
        Supports:
        - Grouped Query Attention (GQA)
        - Position offset for KV-cache inference
        - Interleaved vs sequential rotation
        - Batched processing
        """
        # Program IDs
        pid_batch = tl.program_id(0)
        pid_seq = tl.program_id(1)
        pid_head = tl.program_id(2)
        
        # Compute half dimension
        half_dim = head_dim // 2
        
        # Sequence offset handling
        seq_start = pid_seq * BLOCK_SEQ
        seq_offs = seq_start + tl.arange(0, BLOCK_SEQ)
        seq_mask = seq_offs < seq_len
        
        # Dimension offsets (first half)
        dim_offs = tl.arange(0, BLOCK_DIM)
        dim_mask = dim_offs < half_dim
        
        # Position for frequency lookup (with offset for KV-cache)
        pos_ids = seq_offs + position_offset
        
        # ─────────────────────────────────────────────────────────────────────
        # Load frequencies for all positions in block
        # ─────────────────────────────────────────────────────────────────────
        
        # For each position and dimension, load cos/sin
        # Shape: [BLOCK_SEQ, BLOCK_DIM]
        cos_ptrs = cos_ptr + pos_ids[:, None] * cos_seq_stride + dim_offs[None, :] * cos_dim_stride
        sin_ptrs = sin_ptr + pos_ids[:, None] * sin_seq_stride + dim_offs[None, :] * sin_dim_stride
        
        freq_mask = seq_mask[:, None] & dim_mask[None, :]
        
        cos_vals = tl.load(cos_ptrs, mask=freq_mask, other=1.0)
        sin_vals = tl.load(sin_ptrs, mask=freq_mask, other=0.0)
        
        # ─────────────────────────────────────────────────────────────────────
        # Process Q
        # ─────────────────────────────────────────────────────────────────────
        
        # Q base pointer for this batch and head
        q_base = Q_ptr + pid_batch * Q_batch_stride + pid_head * Q_head_stride
        
        if INTERLEAVED:
            # Interleaved layout: [x0, y0, x1, y1, ...]
            # Load pairs
            dim_offs_0 = dim_offs * 2
            dim_offs_1 = dim_offs * 2 + 1
            
            q_ptrs_0 = q_base + seq_offs[:, None] * Q_seq_stride + dim_offs_0[None, :] * Q_dim_stride
            q_ptrs_1 = q_base + seq_offs[:, None] * Q_seq_stride + dim_offs_1[None, :] * Q_dim_stride
            
            q0 = tl.load(q_ptrs_0, mask=freq_mask, other=0.0)
            q1 = tl.load(q_ptrs_1, mask=freq_mask, other=0.0)
            
            # Apply rotation
            q0_rot = q0 * cos_vals - q1 * sin_vals
            q1_rot = q1 * cos_vals + q0 * sin_vals
            
            tl.store(q_ptrs_0, q0_rot, mask=freq_mask)
            tl.store(q_ptrs_1, q1_rot, mask=freq_mask)
        else:
            # Sequential layout: [x0, x1, ..., y0, y1, ...]
            q_ptrs_first = q_base + seq_offs[:, None] * Q_seq_stride + dim_offs[None, :] * Q_dim_stride
            q_ptrs_second = q_base + seq_offs[:, None] * Q_seq_stride + (dim_offs[None, :] + half_dim) * Q_dim_stride
            
            q_first = tl.load(q_ptrs_first, mask=freq_mask, other=0.0)
            q_second = tl.load(q_ptrs_second, mask=freq_mask, other=0.0)
            
            # Apply rotation: q' = q_first * cos - q_second * sin
            #                 q'' = q_second * cos + q_first * sin
            q_first_rot = q_first * cos_vals - q_second * sin_vals
            q_second_rot = q_second * cos_vals + q_first * sin_vals
            
            tl.store(q_ptrs_first, q_first_rot, mask=freq_mask)
            tl.store(q_ptrs_second, q_second_rot, mask=freq_mask)
        
        # ─────────────────────────────────────────────────────────────────────
        # Process K (if within K heads for GQA)
        # ─────────────────────────────────────────────────────────────────────
        
        if PROCESS_K:
            if pid_head < n_heads_K:
                k_base = K_ptr + pid_batch * K_batch_stride + pid_head * K_head_stride
                
                if INTERLEAVED:
                    k_ptrs_0 = k_base + seq_offs[:, None] * K_seq_stride + dim_offs_0[None, :] * K_dim_stride
                    k_ptrs_1 = k_base + seq_offs[:, None] * K_seq_stride + dim_offs_1[None, :] * K_dim_stride
                    
                    k0 = tl.load(k_ptrs_0, mask=freq_mask, other=0.0)
                    k1 = tl.load(k_ptrs_1, mask=freq_mask, other=0.0)
                    
                    k0_rot = k0 * cos_vals - k1 * sin_vals
                    k1_rot = k1 * cos_vals + k0 * sin_vals
                    
                    tl.store(k_ptrs_0, k0_rot, mask=freq_mask)
                    tl.store(k_ptrs_1, k1_rot, mask=freq_mask)
                else:
                    k_ptrs_first = k_base + seq_offs[:, None] * K_seq_stride + dim_offs[None, :] * K_dim_stride
                    k_ptrs_second = k_base + seq_offs[:, None] * K_seq_stride + (dim_offs[None, :] + half_dim) * K_dim_stride
                    
                    k_first = tl.load(k_ptrs_first, mask=freq_mask, other=0.0)
                    k_second = tl.load(k_ptrs_second, mask=freq_mask, other=0.0)
                    
                    k_first_rot = k_first * cos_vals - k_second * sin_vals
                    k_second_rot = k_second * cos_vals + k_first * sin_vals
                    
                    tl.store(k_ptrs_first, k_first_rot, mask=freq_mask)
                    tl.store(k_ptrs_second, k_second_rot, mask=freq_mask)


    @triton.jit
    def _rope_bwd_kernel(
        # Pointers
        dQ_ptr, dK_ptr,
        cos_ptr, sin_ptr,
        # Strides for dQ
        dQ_batch_stride, dQ_seq_stride, dQ_head_stride, dQ_dim_stride,
        # Strides for dK
        dK_batch_stride, dK_seq_stride, dK_head_stride, dK_dim_stride,
        # Frequency strides
        cos_seq_stride, cos_dim_stride,
        sin_seq_stride, sin_dim_stride,
        # Dimensions
        batch_size, seq_len, n_heads_Q, n_heads_K, head_dim,
        # Position offset
        position_offset,
        # Flags
        PROCESS_K: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        # Block sizes
        BLOCK_SEQ: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
    ):
        """
        RoPE backward kernel.
        
        Backward is same as forward with negated sin (rotation in opposite direction).
        """
        pid_batch = tl.program_id(0)
        pid_seq = tl.program_id(1)
        pid_head = tl.program_id(2)
        
        half_dim = head_dim // 2
        
        seq_start = pid_seq * BLOCK_SEQ
        seq_offs = seq_start + tl.arange(0, BLOCK_SEQ)
        seq_mask = seq_offs < seq_len
        
        dim_offs = tl.arange(0, BLOCK_DIM)
        dim_mask = dim_offs < half_dim
        
        pos_ids = seq_offs + position_offset
        
        # Load frequencies
        cos_ptrs = cos_ptr + pos_ids[:, None] * cos_seq_stride + dim_offs[None, :] * cos_dim_stride
        sin_ptrs = sin_ptr + pos_ids[:, None] * sin_seq_stride + dim_offs[None, :] * sin_dim_stride
        
        freq_mask = seq_mask[:, None] & dim_mask[None, :]
        
        cos_vals = tl.load(cos_ptrs, mask=freq_mask, other=1.0)
        sin_vals = -tl.load(sin_ptrs, mask=freq_mask, other=0.0)  # Negate for backward!
        
        # Process dQ
        dq_base = dQ_ptr + pid_batch * dQ_batch_stride + pid_head * dQ_head_stride
        
        if INTERLEAVED:
            dim_offs_0 = dim_offs * 2
            dim_offs_1 = dim_offs * 2 + 1
            
            dq_ptrs_0 = dq_base + seq_offs[:, None] * dQ_seq_stride + dim_offs_0[None, :] * dQ_dim_stride
            dq_ptrs_1 = dq_base + seq_offs[:, None] * dQ_seq_stride + dim_offs_1[None, :] * dQ_dim_stride
            
            dq0 = tl.load(dq_ptrs_0, mask=freq_mask, other=0.0)
            dq1 = tl.load(dq_ptrs_1, mask=freq_mask, other=0.0)
            
            dq0_rot = dq0 * cos_vals - dq1 * sin_vals
            dq1_rot = dq1 * cos_vals + dq0 * sin_vals
            
            tl.store(dq_ptrs_0, dq0_rot, mask=freq_mask)
            tl.store(dq_ptrs_1, dq1_rot, mask=freq_mask)
        else:
            dq_ptrs_first = dq_base + seq_offs[:, None] * dQ_seq_stride + dim_offs[None, :] * dQ_dim_stride
            dq_ptrs_second = dq_base + seq_offs[:, None] * dQ_seq_stride + (dim_offs[None, :] + half_dim) * dQ_dim_stride
            
            dq_first = tl.load(dq_ptrs_first, mask=freq_mask, other=0.0)
            dq_second = tl.load(dq_ptrs_second, mask=freq_mask, other=0.0)
            
            dq_first_rot = dq_first * cos_vals - dq_second * sin_vals
            dq_second_rot = dq_second * cos_vals + dq_first * sin_vals
            
            tl.store(dq_ptrs_first, dq_first_rot, mask=freq_mask)
            tl.store(dq_ptrs_second, dq_second_rot, mask=freq_mask)
        
        # Process dK
        if PROCESS_K:
            if pid_head < n_heads_K:
                dk_base = dK_ptr + pid_batch * dK_batch_stride + pid_head * dK_head_stride
                
                if INTERLEAVED:
                    dk_ptrs_0 = dk_base + seq_offs[:, None] * dK_seq_stride + dim_offs_0[None, :] * dK_dim_stride
                    dk_ptrs_1 = dk_base + seq_offs[:, None] * dK_seq_stride + dim_offs_1[None, :] * dK_dim_stride
                    
                    dk0 = tl.load(dk_ptrs_0, mask=freq_mask, other=0.0)
                    dk1 = tl.load(dk_ptrs_1, mask=freq_mask, other=0.0)
                    
                    dk0_rot = dk0 * cos_vals - dk1 * sin_vals
                    dk1_rot = dk1 * cos_vals + dk0 * sin_vals
                    
                    tl.store(dk_ptrs_0, dk0_rot, mask=freq_mask)
                    tl.store(dk_ptrs_1, dk1_rot, mask=freq_mask)
                else:
                    dk_ptrs_first = dk_base + seq_offs[:, None] * dK_seq_stride + dim_offs[None, :] * dK_dim_stride
                    dk_ptrs_second = dk_base + seq_offs[:, None] * dK_seq_stride + (dim_offs[None, :] + half_dim) * dK_dim_stride
                    
                    dk_first = tl.load(dk_ptrs_first, mask=freq_mask, other=0.0)
                    dk_second = tl.load(dk_ptrs_second, mask=freq_mask, other=0.0)
                    
                    dk_first_rot = dk_first * cos_vals - dk_second * sin_vals
                    dk_second_rot = dk_second * cos_vals + dk_first * sin_vals
                    
                    tl.store(dk_ptrs_first, dk_first_rot, mask=freq_mask)
                    tl.store(dk_ptrs_second, dk_second_rot, mask=freq_mask)


    @triton.jit
    def _rope_single_tensor_kernel(
        # Pointer
        X_ptr,
        cos_ptr, sin_ptr,
        # Strides
        X_batch_stride, X_seq_stride, X_head_stride, X_dim_stride,
        cos_seq_stride, cos_dim_stride,
        sin_seq_stride, sin_dim_stride,
        # Dimensions
        batch_size, seq_len, n_heads, head_dim,
        # Position offset
        position_offset,
        # Flags
        BACKWARD: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        # Block sizes
        BLOCK_SEQ: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        """
        Single tensor RoPE kernel with head grouping.
        
        Processes GROUP_SIZE heads per program for improved efficiency.
        """
        pid_batch = tl.program_id(0)
        pid_seq = tl.program_id(1)
        pid_head_group = tl.program_id(2)
        
        half_dim = head_dim // 2
        
        seq_start = pid_seq * BLOCK_SEQ
        seq_offs = seq_start + tl.arange(0, BLOCK_SEQ)
        seq_mask = seq_offs < seq_len
        
        dim_offs = tl.arange(0, BLOCK_DIM)
        dim_mask = dim_offs < half_dim
        
        pos_ids = seq_offs + position_offset
        
        # Load frequencies
        cos_ptrs = cos_ptr + pos_ids[:, None] * cos_seq_stride + dim_offs[None, :] * cos_dim_stride
        sin_ptrs = sin_ptr + pos_ids[:, None] * sin_seq_stride + dim_offs[None, :] * sin_dim_stride
        
        freq_mask = seq_mask[:, None] & dim_mask[None, :]
        
        cos_vals = tl.load(cos_ptrs, mask=freq_mask, other=1.0)
        sin_vals = tl.load(sin_ptrs, mask=freq_mask, other=0.0)
        
        if BACKWARD:
            sin_vals = -sin_vals
        
        # Process GROUP_SIZE heads
        head_start = pid_head_group * GROUP_SIZE
        
        for h in range(GROUP_SIZE):
            head_idx = head_start + h
            if head_idx < n_heads:
                x_base = X_ptr + pid_batch * X_batch_stride + head_idx * X_head_stride
                
                if INTERLEAVED:
                    dim_offs_0 = dim_offs * 2
                    dim_offs_1 = dim_offs * 2 + 1
                    
                    x_ptrs_0 = x_base + seq_offs[:, None] * X_seq_stride + dim_offs_0[None, :] * X_dim_stride
                    x_ptrs_1 = x_base + seq_offs[:, None] * X_seq_stride + dim_offs_1[None, :] * X_dim_stride
                    
                    x0 = tl.load(x_ptrs_0, mask=freq_mask, other=0.0)
                    x1 = tl.load(x_ptrs_1, mask=freq_mask, other=0.0)
                    
                    x0_rot = x0 * cos_vals - x1 * sin_vals
                    x1_rot = x1 * cos_vals + x0 * sin_vals
                    
                    tl.store(x_ptrs_0, x0_rot, mask=freq_mask)
                    tl.store(x_ptrs_1, x1_rot, mask=freq_mask)
                else:
                    x_ptrs_first = x_base + seq_offs[:, None] * X_seq_stride + dim_offs[None, :] * X_dim_stride
                    x_ptrs_second = x_base + seq_offs[:, None] * X_seq_stride + (dim_offs[None, :] + half_dim) * X_dim_stride
                    
                    x_first = tl.load(x_ptrs_first, mask=freq_mask, other=0.0)
                    x_second = tl.load(x_ptrs_second, mask=freq_mask, other=0.0)
                    
                    x_first_rot = x_first * cos_vals - x_second * sin_vals
                    x_second_rot = x_second * cos_vals + x_first * sin_vals
                    
                    tl.store(x_ptrs_first, x_first_rot, mask=freq_mask)
                    tl.store(x_ptrs_second, x_second_rot, mask=freq_mask)


    @triton.jit
    def _rope_with_position_ids_kernel(
        # Pointers
        X_ptr,
        cos_ptr, sin_ptr,
        position_ids_ptr,
        # Strides
        X_batch_stride, X_seq_stride, X_head_stride, X_dim_stride,
        cos_seq_stride, cos_dim_stride,
        sin_seq_stride, sin_dim_stride,
        pos_batch_stride, pos_seq_stride,
        # Dimensions
        batch_size, seq_len, n_heads, head_dim,
        # Flags
        BACKWARD: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        # Block sizes
        BLOCK_DIM: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        """
        RoPE kernel with explicit position IDs.
        
        Supports non-contiguous position IDs for:
        - Prefix caching
        - Speculative decoding
        - Discontinuous sequences
        """
        pid_batch = tl.program_id(0)
        pid_seq = tl.program_id(1)
        pid_head_group = tl.program_id(2)
        
        half_dim = head_dim // 2
        
        dim_offs = tl.arange(0, BLOCK_DIM)
        dim_mask = dim_offs < half_dim
        
        # Load position ID for this sequence position
        pos_id = tl.load(position_ids_ptr + pid_batch * pos_batch_stride + pid_seq * pos_seq_stride)
        
        # Load frequencies for this position
        cos_vals = tl.load(cos_ptr + pos_id * cos_seq_stride + dim_offs * cos_dim_stride, mask=dim_mask, other=1.0)
        sin_vals = tl.load(sin_ptr + pos_id * sin_seq_stride + dim_offs * sin_dim_stride, mask=dim_mask, other=0.0)
        
        if BACKWARD:
            sin_vals = -sin_vals
        
        # Process GROUP_SIZE heads
        head_start = pid_head_group * GROUP_SIZE
        
        for h in range(GROUP_SIZE):
            head_idx = head_start + h
            if head_idx < n_heads:
                x_base = X_ptr + pid_batch * X_batch_stride + pid_seq * X_seq_stride + head_idx * X_head_stride
                
                if INTERLEAVED:
                    dim_offs_0 = dim_offs * 2
                    dim_offs_1 = dim_offs * 2 + 1
                    
                    x0 = tl.load(x_base + dim_offs_0 * X_dim_stride, mask=dim_mask, other=0.0)
                    x1 = tl.load(x_base + dim_offs_1 * X_dim_stride, mask=dim_mask, other=0.0)
                    
                    x0_rot = x0 * cos_vals - x1 * sin_vals
                    x1_rot = x1 * cos_vals + x0 * sin_vals
                    
                    tl.store(x_base + dim_offs_0 * X_dim_stride, x0_rot, mask=dim_mask)
                    tl.store(x_base + dim_offs_1 * X_dim_stride, x1_rot, mask=dim_mask)
                else:
                    x_first = tl.load(x_base + dim_offs * X_dim_stride, mask=dim_mask, other=0.0)
                    x_second = tl.load(x_base + (dim_offs + half_dim) * X_dim_stride, mask=dim_mask, other=0.0)
                    
                    x_first_rot = x_first * cos_vals - x_second * sin_vals
                    x_second_rot = x_second * cos_vals + x_first * sin_vals
                    
                    tl.store(x_base + dim_offs * X_dim_stride, x_first_rot, mask=dim_mask)
                    tl.store(x_base + (dim_offs + half_dim) * X_dim_stride, x_second_rot, mask=dim_mask)


# ═════════════════════════════════════════════════════════════════════════════════
# Kernel Launch Wrappers
# ═════════════════════════════════════════════════════════════════════════════════

def _get_rope_kernel_config(head_dim: int) -> Dict[str, int]:
    """Get optimal kernel configuration."""
    half_dim = head_dim // 2
    
    # Block size for dimension processing
    BLOCK_DIM = triton.next_power_of_2(half_dim) if _TRITON_AVAILABLE else 64
    BLOCK_DIM = min(BLOCK_DIM, 256)
    
    # Block size for sequence processing
    BLOCK_SEQ = 1 if head_dim >= 128 else 4
    
    # Number of warps
    if BLOCK_DIM <= 64:
        num_warps = 2
    elif BLOCK_DIM <= 128:
        num_warps = 4
    else:
        num_warps = 8
    
    return {
        "BLOCK_DIM": BLOCK_DIM,
        "BLOCK_SEQ": BLOCK_SEQ,
        "num_warps": num_warps,
        "GROUP_SIZE": 4,
    }


def _launch_rope_fused_qk_kernel(
    Q: Tensor,
    K: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_offset: int = 0,
    interleaved: bool = False,
) -> None:
    """Launch fused Q+K RoPE kernel."""
    batch_size, seq_len, n_heads_Q, head_dim = Q.shape
    n_heads_K = K.shape[2]
    
    config = _get_rope_kernel_config(head_dim)
    BLOCK_SEQ = config["BLOCK_SEQ"]
    BLOCK_DIM = config["BLOCK_DIM"]
    
    grid = (
        batch_size,
        triton.cdiv(seq_len, BLOCK_SEQ),
        n_heads_Q,
    )
    
    _rope_fwd_kernel[grid](
        Q, K,
        cos, sin,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        cos.stride(0), cos.stride(1) if cos.dim() > 1 else 1,
        sin.stride(0), sin.stride(1) if sin.dim() > 1 else 1,
        batch_size, seq_len, n_heads_Q, n_heads_K, head_dim,
        position_offset,
        PROCESS_K=True,
        INTERLEAVED=interleaved,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DIM=BLOCK_DIM,
        num_warps=config["num_warps"],
    )


def _launch_rope_single_kernel(
    X: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_offset: int = 0,
    backward: bool = False,
    interleaved: bool = False,
) -> None:
    """Launch single tensor RoPE kernel."""
    batch_size, seq_len, n_heads, head_dim = X.shape
    
    config = _get_rope_kernel_config(head_dim)
    BLOCK_SEQ = config["BLOCK_SEQ"]
    BLOCK_DIM = config["BLOCK_DIM"]
    GROUP_SIZE = config["GROUP_SIZE"]
    
    n_head_groups = triton.cdiv(n_heads, GROUP_SIZE)
    
    grid = (
        batch_size,
        triton.cdiv(seq_len, BLOCK_SEQ),
        n_head_groups,
    )
    
    _rope_single_tensor_kernel[grid](
        X,
        cos, sin,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        cos.stride(0), cos.stride(1) if cos.dim() > 1 else 1,
        sin.stride(0), sin.stride(1) if sin.dim() > 1 else 1,
        batch_size, seq_len, n_heads, head_dim,
        position_offset,
        BACKWARD=backward,
        INTERLEAVED=interleaved,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DIM=BLOCK_DIM,
        GROUP_SIZE=GROUP_SIZE,
        num_warps=config["num_warps"],
    )


def _launch_rope_with_position_ids_kernel(
    X: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Tensor,
    backward: bool = False,
    interleaved: bool = False,
) -> None:
    """Launch RoPE kernel with explicit position IDs."""
    batch_size, seq_len, n_heads, head_dim = X.shape
    
    config = _get_rope_kernel_config(head_dim)
    BLOCK_DIM = config["BLOCK_DIM"]
    GROUP_SIZE = config["GROUP_SIZE"]
    
    n_head_groups = triton.cdiv(n_heads, GROUP_SIZE)
    
    grid = (batch_size, seq_len, n_head_groups)
    
    _rope_with_position_ids_kernel[grid](
        X,
        cos, sin,
        position_ids,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        cos.stride(0), cos.stride(1) if cos.dim() > 1 else 1,
        sin.stride(0), sin.stride(1) if sin.dim() > 1 else 1,
        position_ids.stride(0), position_ids.stride(1) if position_ids.dim() > 1 else 0,
        batch_size, seq_len, n_heads, head_dim,
        BACKWARD=backward,
        INTERLEAVED=interleaved,
        BLOCK_DIM=BLOCK_DIM,
        GROUP_SIZE=GROUP_SIZE,
        num_warps=config["num_warps"],
    )


# ═════════════════════════════════════════════════════════════════════════════════
# PyTorch Fallback Implementations
# ═════════════════════════════════════════════════════════════════════════════════

def _rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_pytorch(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Optional[Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tensor:
    """PyTorch implementation of RoPE."""
    if position_ids is not None:
        # Gather cos/sin by position IDs
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    else:
        seq_len = x.shape[1]
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(unsqueeze_dim)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(unsqueeze_dim)
    
    # Expand cos/sin for broadcasting
    # x: [batch, seq, heads, dim]
    # cos/sin: [batch, 1, seq, dim//2] or [1, 1, seq, dim//2]
    
    half_dim = x.shape[-1] // 2
    x_first = x[..., :half_dim]
    x_second = x[..., half_dim:]
    
    x_rot_first = x_first * cos - x_second * sin
    x_rot_second = x_second * cos + x_first * sin
    
    return torch.cat([x_rot_first, x_rot_second], dim=-1)


def _apply_rope_interleaved_pytorch(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Optional[Tensor] = None,
) -> Tensor:
    """PyTorch implementation for interleaved RoPE layout."""
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]
    else:
        seq_len = x.shape[1]
        cos = cos[:seq_len]
        sin = sin[:seq_len]
    
    # Interleaved: [x0, y0, x1, y1, ...]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim//2]
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    x_even_rot = x_even * cos - x_odd * sin
    x_odd_rot = x_odd * cos + x_even * sin
    
    # Interleave back
    x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1).flatten(-2)
    
    return x_rot


# ═════════════════════════════════════════════════════════════════════════════════
# Autograd Functions
# ═════════════════════════════════════════════════════════════════════════════════

class RoPEFunction(torch.autograd.Function):
    """
    Custom autograd function for RoPE with Triton kernels.
    
    Supports inplace operation for zero memory overhead.
    """
    
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        position_ids: Optional[Tensor],
        position_offset: int,
        interleaved: bool,
        inplace: bool,
    ) -> Tensor:
        # Ensure contiguous
        if not inplace:
            x = x.clone()
        
        ctx.save_for_backward(cos, sin, position_ids)
        ctx.position_offset = position_offset
        ctx.interleaved = interleaved
        ctx.shape = x.shape
        
        if _TRITON_AVAILABLE and x.is_cuda:
            if position_ids is not None:
                _launch_rope_with_position_ids_kernel(
                    x, cos, sin, position_ids,
                    backward=False, interleaved=interleaved,
                )
            else:
                _launch_rope_single_kernel(
                    x, cos, sin, position_offset,
                    backward=False, interleaved=interleaved,
                )
        else:
            # PyTorch fallback
            if interleaved:
                x_rot = _apply_rope_interleaved_pytorch(x, cos, sin, position_ids)
            else:
                x_rot = _apply_rope_pytorch(x, cos, sin, position_ids)
            x.copy_(x_rot)
        
        return x
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None, None, None, None, None]:
        cos, sin, position_ids = ctx.saved_tensors
        
        # Clone gradient for inplace modification
        grad_input = grad_output.clone()
        
        if _TRITON_AVAILABLE and grad_input.is_cuda:
            if position_ids is not None:
                _launch_rope_with_position_ids_kernel(
                    grad_input, cos, sin, position_ids,
                    backward=True, interleaved=ctx.interleaved,
                )
            else:
                _launch_rope_single_kernel(
                    grad_input, cos, sin, ctx.position_offset,
                    backward=True, interleaved=ctx.interleaved,
                )
        else:
            # PyTorch fallback with negated sin
            neg_sin = -sin
            if ctx.interleaved:
                grad_rot = _apply_rope_interleaved_pytorch(grad_input, cos, neg_sin, position_ids)
            else:
                grad_rot = _apply_rope_pytorch(grad_input, cos, neg_sin, position_ids)
            grad_input.copy_(grad_rot)
        
        return grad_input, None, None, None, None, None, None


class FusedRoPEQKFunction(torch.autograd.Function):
    """
    Fused RoPE for Q and K tensors.
    
    Single kernel processes both tensors for maximum efficiency.
    """
    
    @staticmethod
    def forward(
        ctx,
        Q: Tensor,
        K: Tensor,
        cos: Tensor,
        sin: Tensor,
        position_offset: int,
        interleaved: bool,
    ) -> Tuple[Tensor, Tensor]:
        ctx.save_for_backward(cos, sin)
        ctx.position_offset = position_offset
        ctx.interleaved = interleaved
        ctx.Q_shape = Q.shape
        ctx.K_shape = K.shape
        
        if _TRITON_AVAILABLE and Q.is_cuda:
            _launch_rope_fused_qk_kernel(
                Q, K, cos, sin, position_offset, interleaved
            )
        else:
            # PyTorch fallback
            if interleaved:
                Q_rot = _apply_rope_interleaved_pytorch(Q, cos, sin)
                K_rot = _apply_rope_interleaved_pytorch(K, cos, sin)
            else:
                Q_rot = _apply_rope_pytorch(Q, cos, sin)
                K_rot = _apply_rope_pytorch(K, cos, sin)
            Q.copy_(Q_rot)
            K.copy_(K_rot)
        
        return Q, K
    
    @staticmethod
    def backward(ctx, dQ: Tensor, dK: Tensor) -> Tuple[Tensor, Tensor, None, None, None, None]:
        cos, sin = ctx.saved_tensors
        
        # Clone for inplace modification
        dQ = dQ.clone()
        dK = dK.clone()
        
        if _TRITON_AVAILABLE and dQ.is_cuda:
            # Use backward kernels (negate sin internally)
            batch_size, seq_len, n_heads_Q, head_dim = dQ.shape
            n_heads_K = dK.shape[2]
            
            config = _get_rope_kernel_config(head_dim)
            BLOCK_SEQ = config["BLOCK_SEQ"]
            BLOCK_DIM = config["BLOCK_DIM"]
            
            grid = (
                batch_size,
                triton.cdiv(seq_len, BLOCK_SEQ),
                n_heads_Q,
            )
            
            _rope_bwd_kernel[grid](
                dQ, dK,
                cos, sin,
                dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
                dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
                cos.stride(0), cos.stride(1) if cos.dim() > 1 else 1,
                sin.stride(0), sin.stride(1) if sin.dim() > 1 else 1,
                batch_size, seq_len, n_heads_Q, n_heads_K, head_dim,
                ctx.position_offset,
                PROCESS_K=True,
                INTERLEAVED=ctx.interleaved,
                BLOCK_SEQ=BLOCK_SEQ,
                BLOCK_DIM=BLOCK_DIM,
                num_warps=config["num_warps"],
            )
        else:
            neg_sin = -sin
            if ctx.interleaved:
                dQ_rot = _apply_rope_interleaved_pytorch(dQ, cos, neg_sin)
                dK_rot = _apply_rope_interleaved_pytorch(dK, cos, neg_sin)
            else:
                dQ_rot = _apply_rope_pytorch(dQ, cos, neg_sin)
                dK_rot = _apply_rope_pytorch(dK, cos, neg_sin)
            dQ.copy_(dQ_rot)
            dK.copy_(dK_rot)
        
        return dQ, dK, None, None, None, None


# Legacy compatibility class
class Fast_RoPE_Embedding(torch.autograd.Function):
    """
    Fast RoPE with custom backward pass (legacy interface).
    
    Maintained for backward compatibility with existing code.
    """
    
    @staticmethod
    def forward(ctx, Q: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        cos = cos.squeeze()
        sin = sin.squeeze()
        
        batch, seq_len, n_heads, head_dim = Q.shape
        
        ctx.save_for_backward(cos, sin)
        ctx.original_shape = (batch, seq_len, n_heads, head_dim)
        
        if _TRITON_AVAILABLE and Q.is_cuda:
            _launch_rope_single_kernel(
                Q, cos, sin,
                position_offset=0,
                backward=False,
                interleaved=False,
            )
        else:
            Q_rot = _apply_rope_pytorch(Q, cos, sin)
            Q.copy_(Q_rot)
        
        return Q
    
    @staticmethod
    def backward(ctx, dQ: Tensor) -> Tuple[Tensor, None, None]:
        cos, sin = ctx.saved_tensors
        batch, seq_len, n_heads, head_dim = ctx.original_shape
        
        dQ = dQ.clone()
        
        if _TRITON_AVAILABLE and dQ.is_cuda:
            _launch_rope_single_kernel(
                dQ, cos, sin,
                position_offset=0,
                backward=True,
                interleaved=False,
            )
        else:
            neg_sin = -sin
            dQ_rot = _apply_rope_pytorch(dQ, cos, neg_sin)
            dQ.copy_(dQ_rot)
        
        return dQ, None, None


# ═════════════════════════════════════════════════════════════════════════════════
# High-Level API Functions
# ═════════════════════════════════════════════════════════════════════════════════

def apply_rope(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Optional[Tensor] = None,
    position_offset: int = 0,
    interleaved: bool = False,
    inplace: bool = True,
) -> Tensor:
    """
    Apply Rotary Position Embedding to tensor.
    
    Args:
        x: Input tensor [batch, seq_len, n_heads, head_dim]
        cos: Cosine frequencies [max_seq_len, head_dim//2]
        sin: Sine frequencies [max_seq_len, head_dim//2]
        position_ids: Optional position indices [batch, seq_len]
        position_offset: Offset for positions (KV-cache inference)
        interleaved: Use interleaved rotation layout
        inplace: Modify tensor inplace (zero memory overhead)
    
    Returns:
        Rotated tensor (same as input if inplace=True)
    """
    return RoPEFunction.apply(
        x, cos.contiguous(), sin.contiguous(),
        position_ids, position_offset, interleaved, inplace,
    )


def apply_rope_qk(
    Q: Tensor,
    K: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_offset: int = 0,
    interleaved: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Apply RoPE to Q and K tensors with fused kernel.
    
    Args:
        Q: Query tensor [batch, seq_len, n_heads_q, head_dim]
        K: Key tensor [batch, seq_len, n_heads_k, head_dim]
        cos: Cosine frequencies [max_seq_len, head_dim//2]
        sin: Sine frequencies [max_seq_len, head_dim//2]
        position_offset: Offset for positions (KV-cache)
        interleaved: Use interleaved rotation layout
    
    Returns:
        Tuple of (rotated_Q, rotated_K)
    """
    return FusedRoPEQKFunction.apply(
        Q, K, cos.contiguous(), sin.contiguous(),
        position_offset, interleaved,
    )


def fast_rope_embedding(Q: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    Apply fast RoPE embedding with Triton acceleration.
    
    Legacy interface - use apply_rope for new code.
    """
    return Fast_RoPE_Embedding.apply(Q, cos, sin)


def inplace_rope_embedding(
    Q: Tensor,
    K: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_offset: int = 0,
) -> Tuple[Tensor, Tensor]:
    """
    Apply RoPE to Q and K inplace (fused kernel).
    
    Zero memory overhead - modifies tensors directly.
    
    Args:
        Q: Query tensor [batch, seq_len, n_heads_q, head_dim]
        K: Key tensor [batch, seq_len, n_heads_k, head_dim]
        cos: Cosine frequencies
        sin: Sine frequencies
        position_offset: Position offset for KV-cache
    
    Returns:
        Same Q and K tensors (modified inplace)
    """
    cos = cos.squeeze().contiguous()
    sin = sin.squeeze().contiguous()
    
    return apply_rope_qk(Q, K, cos, sin, position_offset, interleaved=False)


# ═════════════════════════════════════════════════════════════════════════════════
# RoPE Embedding Module
# ═════════════════════════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    """
    SOTA Rotary Position Embedding module.
    
    Features:
    - Cached frequency computation
    - Multiple scaling methods (PI, NTK, YaRN, LongRoPE)
    - Triton-accelerated application
    - Support for GQA/MQA
    - KV-cache position offset
    - Interleaved rotation support
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        scaling_type: str = "none",
        scaling_factor: float = 1.0,
        interleaved: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        self.interleaved = interleaved
        
        # Create config
        scaling_map = {
            "none": RoPEScalingType.NONE,
            "linear": RoPEScalingType.LINEAR,
            "pi": RoPEScalingType.LINEAR,
            "ntk": RoPEScalingType.NTK,
            "dynamic_ntk": RoPEScalingType.DYNAMIC_NTK,
            "yarn": RoPEScalingType.YARN,
            "longrope": RoPEScalingType.LONGROPE,
        }
        
        self.config = RoPEConfig(
            dim=dim,
            max_seq_len=max_seq_len,
            base=base,
            scaling_type=scaling_map.get(scaling_type.lower(), RoPEScalingType.NONE),
            scaling_factor=scaling_factor,
            **kwargs,
        )
        
        # Register frequencies as buffers
        cos, sin = compute_rope_frequencies(
            self.config, max_seq_len, torch.device("cpu"), torch.float32
        )
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        
        self._seq_len_cached = max_seq_len
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update frequency cache if needed."""
        if seq_len > self._seq_len_cached:
            cos, sin = compute_rope_frequencies(
                self.config, seq_len, device, dtype
            )
            self.cos_cached = cos
            self.sin_cached = sin
            self._seq_len_cached = seq_len
        elif self.cos_cached.device != device or self.cos_cached.dtype != dtype:
            self.cos_cached = self.cos_cached.to(device=device, dtype=dtype)
            self.sin_cached = self.sin_cached.to(device=device, dtype=dtype)
    
    def forward(
        self,
        x: Tensor,
        position_ids: Optional[Tensor] = None,
        position_offset: int = 0,
    ) -> Tensor:
        """
        Apply RoPE to input tensor.
        
        Args:
            x: [batch, seq_len, n_heads, head_dim]
            position_ids: Optional [batch, seq_len]
            position_offset: Offset for KV-cache
        
        Returns:
            Rotated tensor
        """
        seq_len = x.shape[1] + position_offset
        self._update_cache(seq_len, x.device, x.dtype)
        
        return apply_rope(
            x,
            self.cos_cached,
            self.sin_cached,
            position_ids=position_ids,
            position_offset=position_offset,
            interleaved=self.interleaved,
            inplace=True,
        )
    
    def forward_qk(
        self,
        Q: Tensor,
        K: Tensor,
        position_ids: Optional[Tensor] = None,
        position_offset: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply RoPE to Q and K tensors (fused).
        
        Args:
            Q: [batch, seq_len, n_heads_q, head_dim]
            K: [batch, seq_len, n_heads_k, head_dim]
            position_ids: Optional [batch, seq_len]
            position_offset: Offset for KV-cache
        
        Returns:
            Tuple of (rotated_Q, rotated_K)
        """
        seq_len = Q.shape[1] + position_offset
        self._update_cache(seq_len, Q.device, Q.dtype)
        
        return apply_rope_qk(
            Q, K,
            self.cos_cached,
            self.sin_cached,
            position_offset=position_offset,
            interleaved=self.interleaved,
        )
    
    def get_freqs(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        """Get cached frequencies up to seq_len."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


# ═════════════════════════════════════════════════════════════════════════════════
# Benchmark Utilities
# ═════════════════════════════════════════════════════════════════════════════════

def benchmark_rope(
    batch_size: int = 4,
    seq_len: int = 2048,
    n_heads: int = 32,
    head_dim: int = 128,
    num_warmup: int = 10,
    num_iters: int = 100,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    """
    Benchmark RoPE kernel performance.
    
    Returns timing statistics in milliseconds.
    """
    device = torch.device("cuda")
    
    # Create inputs
    Q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)
    
    cos, sin = precompute_freqs_cis(head_dim, seq_len, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(num_warmup):
        Q_clone = Q.clone()
        K_clone = K.clone()
        apply_rope_qk(Q_clone, K_clone, cos, sin)
    
    torch.cuda.synchronize()
    
    # Benchmark
    import time
    start = time.perf_counter()
    
    for _ in range(num_iters):
        Q_clone = Q.clone()
        K_clone = K.clone()
        apply_rope_qk(Q_clone, K_clone, cos, sin)
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed_ms = (end - start) * 1000 / num_iters
    
    # Compute bandwidth
    bytes_processed = (Q.numel() + K.numel()) * Q.element_size() * 2  # Read + Write
    bandwidth_gb = bytes_processed / (elapsed_ms * 1e6)
    
    return {
        "latency_ms": elapsed_ms,
        "bandwidth_gb_s": bandwidth_gb,
        "tokens_per_sec": (batch_size * seq_len) / (elapsed_ms / 1000),
    }


# ═════════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════════

ROPE_GROUP_SIZE = 4  # Process 4 heads per Triton program


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Configuration
    "RoPEConfig",
    "RoPEScalingType",
    
    # Frequency computation
    "precompute_freqs_cis",
    "compute_rope_frequencies",
    "RoPEFrequencyCache",
    
    # Autograd functions
    "RoPEFunction",
    "FusedRoPEQKFunction",
    "Fast_RoPE_Embedding",
    
    # High-level API
    "apply_rope",
    "apply_rope_qk",
    "fast_rope_embedding",
    "inplace_rope_embedding",
    
    # Module
    "RotaryEmbedding",
    
    # Benchmark
    "benchmark_rope",
    
    # Constants
    "ROPE_GROUP_SIZE",
]