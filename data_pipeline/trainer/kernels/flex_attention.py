# ════════════════════════════════════════════════════════════════════════════════
# SOTA Flex Attention Module
# ════════════════════════════════════════════════════════════════════════════════
# Unsloth-inspired Flex attention with:
# - Logit softcapping (Gemma 2 style)
# - Sliding window masks
# - Grouped Query Attention (GQA)
# - torch.compile optimization
#
# Supports torch 2.5+ Flex Attention API with fallback
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import functools
import math
import os
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ═════════════════════════════════════════════════════════════════════════════════
# Torch Compile Options
# ═════════════════════════════════════════════════════════════════════════════════

TORCH_COMPILE_OPTIONS = {
    "epilogue_fusion": True,
    "max_autotune": True,
    "shape_padding": True,
    "trace.enabled": os.environ.get("UNSLOTH_COMPILE_DEBUG", "0") == "1",
    "triton.cudagraphs": False,
}


# ═════════════════════════════════════════════════════════════════════════════════
# Flex Attention Availability Check
# ═════════════════════════════════════════════════════════════════════════════════

HAS_FLEX_ATTENTION = False

try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask as _create_block_mask,
    )
    _flex_attention = torch.compile(
        _flex_attention, dynamic=True, options=TORCH_COMPILE_OPTIONS
    )
    HAS_FLEX_ATTENTION = True
except ImportError:
    HAS_FLEX_ATTENTION = False


# ═════════════════════════════════════════════════════════════════════════════════
# Score Modification Functions
# ═════════════════════════════════════════════════════════════════════════════════

def generate_tanh_softcap(t: float) -> Callable:
    """
    Generate logit softcapping score modifier.
    
    Gemma 2 style: score = t * tanh(score / t)
    Prevents extreme attention weights.
    """
    def tanh_softcap(score, b, h, q_idx, kv_idx):
        return t * torch.tanh(score / t)
    return tanh_softcap


def generate_scale_modifier(scale: float) -> Callable:
    """Generate simple scaling score modifier."""
    def scale_mod(score, b, h, q_idx, kv_idx):
        return score * scale
    return scale_mod


# ═════════════════════════════════════════════════════════════════════════════════
# Mask Functions
# ═════════════════════════════════════════════════════════════════════════════════

def causal_masker(b, h, q_idx, kv_idx):
    """Standard causal attention mask."""
    return q_idx >= kv_idx


@functools.lru_cache(maxsize=32)
def sliding_window_masker(size: int = 4096) -> Callable:
    """
    Sliding window attention mask.
    
    Each token attends to at most `size` previous tokens.
    """
    def sliding_window(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        window_mask = q_idx - kv_idx <= size
        return causal_mask & window_mask
    return sliding_window


@functools.lru_cache(maxsize=32)
def create_block_mask(mask_fn: Callable, n: int = 128):
    """Create compiled block mask."""
    if HAS_FLEX_ATTENTION:
        return _create_block_mask(
            mask_fn,
            B=1, H=1, Q_LEN=n, KV_LEN=n,
            BLOCK_SIZE=128,
            _compile=True,
        )
    return None


# ═════════════════════════════════════════════════════════════════════════════════
# Flex Attention Creation
# ═════════════════════════════════════════════════════════════════════════════════

def create_flex_attention_causal_mask(max_seq_length: int = 8192):
    """Create causal attention block mask."""
    return create_block_mask(causal_masker, max_seq_length)


def create_flex_attention_sliding_window_mask(
    max_seq_length: int = 8192,
    sliding_window: int = 4096,
):
    """Create sliding window attention block mask."""
    masker = sliding_window_masker(sliding_window)
    return create_block_mask(masker, max_seq_length)


@functools.lru_cache(maxsize=32)
def get_flex_attention(s: float, t: float):
    """
    Get compiled flex attention function.
    
    Args:
        s: Query pre-attention scalar (for scaling)
        t: Logit softcapping value
    """
    if not HAS_FLEX_ATTENTION:
        return None
    
    scale = 1.0 / math.sqrt(s)
    score_mod = generate_tanh_softcap(t)
    
    return functools.partial(
        _flex_attention,
        score_mod=score_mod,
        scale=scale,
        enable_gqa=True,
    )


# ═════════════════════════════════════════════════════════════════════════════════
# Attention with Softcapping (Gemma 2 Style)
# ═════════════════════════════════════════════════════════════════════════════════

@torch.compile(fullgraph=True, dynamic=True, options=TORCH_COMPILE_OPTIONS)
def attention_softcapping_compiled(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    causal_mask: Tensor,
    s: float,  # query_pre_attn_scalar
    t: float,  # attn_logit_softcapping
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> Tensor:
    """
    Compiled attention with logit softcapping.
    
    For Gemma 2 and similar models.
    """
    bsz, _, q_len, _ = Q.shape
    n_groups = n_heads // n_kv_heads
    
    # Expand K, V for grouped query attention
    K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
    V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
    K = K.reshape(bsz, n_heads, q_len, head_dim)
    V = V.reshape(bsz, n_heads, q_len, head_dim)
    
    # Scale query
    Q = Q * (s ** -0.5)
    
    # Compute attention scores
    A = torch.matmul(Q, K.transpose(2, 3))
    
    # Apply softcapping: t * tanh(A / t)
    A = t * torch.tanh(A / t)
    
    # Apply causal mask
    A = A + causal_mask[:q_len, :q_len]
    
    # Softmax
    A = F.softmax(A, dim=-1, dtype=torch.float32).to(Q.dtype)
    
    # Weighted sum
    A = torch.matmul(A, V)
    
    # Reshape output
    A = A.transpose(1, 2).contiguous()
    A = A.reshape(bsz, q_len, n_heads * head_dim)
    
    return A


def slow_attention_softcapping(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    causal_mask: Tensor,
    config,  # Model config with attention params
    bsz: int,
    q_len: int,
) -> Tensor:
    """
    Attention with logit softcapping.
    
    Uses Flex Attention if available, otherwise compiled fallback.
    """
    n_heads = config.num_attention_heads
    head_dim = Q.shape[-1]
    n_kv_heads = config.num_key_value_heads
    s = config.query_pre_attn_scalar
    t = config.attn_logit_softcapping
    
    if HAS_FLEX_ATTENTION:
        fx = get_flex_attention(s, t)
        A = fx(query=Q, key=K, value=V, block_mask=causal_mask)
        A = A.transpose(1, 2).contiguous()
        A = A.reshape(bsz, q_len, n_heads * head_dim)
        return A
    
    return attention_softcapping_compiled(
        Q, K, V, causal_mask,
        s, t, n_heads, n_kv_heads, head_dim
    )


# ═════════════════════════════════════════════════════════════════════════════════
# Standard Scaled Dot-Product Attention (Optimized)
# ═════════════════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """
    Optimized scaled dot-product attention.
    
    Uses FlashAttention-2 backend when available.
    """
    return F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


# ═════════════════════════════════════════════════════════════════════════════════
# Inference Attention (No Compile for Speed)
# ═════════════════════════════════════════════════════════════════════════════════

def slow_inference_attention_softcapping(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    causal_mask: Tensor,
    config,
    bsz: int,
    q_len: int,
) -> Tensor:
    """
    Inference attention without torch.compile overhead.
    
    Faster for single-token generation.
    """
    n_heads = config.num_attention_heads
    head_dim = Q.shape[-1]
    n_kv_heads = config.num_key_value_heads
    n_groups = n_heads // n_kv_heads
    s = config.query_pre_attn_scalar
    t = config.attn_logit_softcapping
    
    # GQA expansion
    K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
    V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
    K = K.reshape(bsz, n_heads, q_len, head_dim)
    V = V.reshape(bsz, n_heads, q_len, head_dim)
    
    # Scale
    Q = Q * (s ** -0.5)
    
    # Attention scores
    A = torch.matmul(Q, K.transpose(2, 3))
    
    # In-place softcapping
    A = A / t
    torch.tanh(A, out=A)
    A = A * t
    A = A + causal_mask[:q_len, :q_len]
    
    # Softmax and output
    A = F.softmax(A, dim=-1, dtype=torch.float32).to(Q.dtype)
    A = torch.matmul(A, V)
    A = A.transpose(1, 2).contiguous()
    A = A.reshape(bsz, q_len, n_heads * head_dim)
    
    return A


# ═════════════════════════════════════════════════════════════════════════════════
# Causal Mask Creation
# ═════════════════════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=8)
def create_causal_mask(
    max_seq_len: int,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
) -> Tensor:
    """Create additive causal attention mask."""
    mask = torch.full((max_seq_len, max_seq_len), float('-inf'), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


@functools.lru_cache(maxsize=8)
def create_sliding_window_causal_mask(
    max_seq_len: int,
    window_size: int = 4096,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
) -> Tensor:
    """Create sliding window causal mask."""
    mask = torch.full((max_seq_len, max_seq_len), float('-inf'), dtype=dtype, device=device)
    
    for i in range(max_seq_len):
        # Can attend to tokens from max(0, i - window_size) to i
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = 0
    
    return mask


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Config
    "TORCH_COMPILE_OPTIONS",
    "HAS_FLEX_ATTENTION",
    # Score modifiers
    "generate_tanh_softcap",
    "generate_scale_modifier",
    # Mask functions
    "causal_masker",
    "sliding_window_masker",
    "create_block_mask",
    "create_flex_attention_causal_mask",
    "create_flex_attention_sliding_window_mask",
    # Flex attention
    "get_flex_attention",
    # Attention implementations
    "attention_softcapping_compiled",
    "slow_attention_softcapping",
    "slow_inference_attention_softcapping",
    "scaled_dot_product_attention",
    # Mask creation
    "create_causal_mask",
    "create_sliding_window_causal_mask",
]
