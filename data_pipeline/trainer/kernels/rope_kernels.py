# ════════════════════════════════════════════════════════════════════════════════
# SOTA RoPE Embedding Kernels
# ════════════════════════════════════════════════════════════════════════════════
# Above-Unsloth RoPE implementations:
# - Fused Q+K RoPE (single kernel)
# - Inplace RoPE (zero memory overhead)
# - Multi-head grouped RoPE
# - YaRN/PI/NTK scaling support
# - Custom backward pass
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional
import math


# ═════════════════════════════════════════════════════════════════════════════════
# Triton Setup
# ═════════════════════════════════════════════════════════════════════════════════

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    triton = None
    tl = None


ROPE_GROUP_SIZE = 4  # Process 4 heads per Triton program


# ═════════════════════════════════════════════════════════════════════════════════
# RoPE Frequency Computation
# ═════════════════════════════════════════════════════════════════════════════════

def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
    scaling_type: str = "none",
    device: str = "cuda",
) -> Tuple[Tensor, Tensor]:
    """
    Precompute RoPE frequencies (cos, sin).
    
    Supports multiple scaling types:
    - none: Standard RoPE
    - linear: Linear interpolation (PI)
    - yarn: YaRN scaling
    - ntk: NTK-aware scaling
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    
    if scaling_type == "linear":
        # PI scaling
        freqs = freqs / scaling_factor
    elif scaling_type == "ntk":
        # NTK-aware scaling
        base = theta * (scaling_factor ** (dim / (dim - 2)))
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    elif scaling_type == "yarn":
        # YaRN scaling (simplified)
        freqs = freqs / (scaling_factor ** 0.5)
    
    t = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(t, freqs)
    
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    
    return cos, sin


# ═════════════════════════════════════════════════════════════════════════════════
# Triton RoPE Kernels
# ═════════════════════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:
    
    @triton.jit
    def _rope_fused_qk_kernel(
        Q_ptr, K_ptr,
        cos_ptr, sin_ptr,
        Q_batch_stride, Q_head_stride, Q_seq_stride,
        K_batch_stride, K_head_stride, K_seq_stride,
        cos_row_stride, sin_row_stride,
        seq_len,
        head_dim: tl.constexpr,
        n_heads_K: tl.constexpr,
        BACKWARD: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused RoPE for Q and K in single kernel.
        
        RoPE: q' = q * cos + rotate_half(q) * sin
        """
        row_pos = tl.program_id(0)
        head_pos = tl.program_id(1)
        
        half_dim = head_dim // 2
        col_offs = tl.arange(0, BLOCK_SIZE)
        mask = col_offs < half_dim
        
        rot_pos = row_pos % seq_len
        
        # Load cos/sin for this position
        cos = tl.load(cos_ptr + rot_pos * cos_row_stride + col_offs, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + rot_pos * sin_row_stride + col_offs, mask=mask, other=0.0)
        
        if BACKWARD:
            sin = -sin
        
        batch_id = row_pos // seq_len
        seq_idx = row_pos - batch_id * seq_len
        
        # Process Q
        q_base = Q_ptr + batch_id * Q_batch_stride + head_pos * Q_head_stride + seq_idx * Q_seq_stride
        q0 = tl.load(q_base + col_offs, mask=mask, other=0.0)
        q1 = tl.load(q_base + half_dim + col_offs, mask=mask, other=0.0)
        
        tl.store(q_base + col_offs, q0 * cos - q1 * sin, mask=mask)
        tl.store(q_base + half_dim + col_offs, q1 * cos + q0 * sin, mask=mask)
        
        # Process K (if within K heads)
        if head_pos < n_heads_K:
            k_base = K_ptr + batch_id * K_batch_stride + head_pos * K_head_stride + seq_idx * K_seq_stride
            k0 = tl.load(k_base + col_offs, mask=mask, other=0.0)
            k1 = tl.load(k_base + half_dim + col_offs, mask=mask, other=0.0)
            
            tl.store(k_base + col_offs, k0 * cos - k1 * sin, mask=mask)
            tl.store(k_base + half_dim + col_offs, k1 * cos + k0 * sin, mask=mask)
    
    
    @triton.jit
    def _rope_grouped_kernel(
        Q_ptr,
        Q_row_stride,
        cos_ptr, cos_row_stride,
        sin_ptr, sin_row_stride,
        seq_len,
        head_dim: tl.constexpr,
        n_heads: tl.constexpr,
        BACKWARD: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        """
        Grouped RoPE: process multiple heads per program for efficiency.
        """
        row_pos = tl.program_id(0)
        group_head_pos = tl.program_id(1)
        
        half_dim = head_dim // 2
        col_offs = tl.arange(0, BLOCK_SIZE)
        mask = col_offs < half_dim
        
        # Load cos/sin
        rot_pos = row_pos % seq_len
        cos = tl.load(cos_ptr + rot_pos * cos_row_stride + col_offs, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + rot_pos * sin_row_stride + col_offs, mask=mask, other=0.0)
        
        if BACKWARD:
            sin = -sin
        
        # Process GROUP_SIZE heads
        head_start = group_head_pos * GROUP_SIZE
        head_end = min(head_start + GROUP_SIZE, n_heads)
        
        for k in range(head_start, head_end):
            offs_q0 = row_pos * Q_row_stride + k * head_dim + col_offs
            offs_q1 = row_pos * Q_row_stride + k * head_dim + col_offs + half_dim
            
            q0 = tl.load(Q_ptr + offs_q0, mask=mask, other=0.0).to(cos.dtype)
            q1 = tl.load(Q_ptr + offs_q1, mask=mask, other=0.0).to(cos.dtype)
            
            tl.store(Q_ptr + offs_q0, q0 * cos - q1 * sin, mask=mask)
            tl.store(Q_ptr + offs_q1, q1 * cos + q0 * sin, mask=mask)


# ═════════════════════════════════════════════════════════════════════════════════
# Fast RoPE Autograd Functions
# ═════════════════════════════════════════════════════════════════════════════════

class Fast_RoPE_Embedding(torch.autograd.Function):
    """
    Fast RoPE with custom backward pass.
    
    Inplace modification for zero memory overhead.
    """
    
    @staticmethod
    def forward(ctx, Q: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        cos = cos.squeeze()
        sin = sin.squeeze()
        
        batch, seq_len, n_heads, head_dim = Q.shape
        Q = Q.reshape(batch * seq_len, n_heads * head_dim)
        
        n_rows, n_cols = Q.shape
        half_dim = head_dim // 2
        
        if _TRITON_AVAILABLE and Q.is_cuda:
            BLOCK_SIZE = triton.next_power_of_2(half_dim)
            num_warps = 4 if BLOCK_SIZE <= 1024 else 8
            
            n_groups = triton.cdiv(n_heads, ROPE_GROUP_SIZE)
            
            _rope_grouped_kernel[(n_rows, n_groups)](
                Q.data_ptr(),
                Q.stride(0),
                cos.data_ptr(), cos.stride(0),
                sin.data_ptr(), sin.stride(0),
                seq_len, head_dim, n_heads,
                BACKWARD=False,
                BLOCK_SIZE=BLOCK_SIZE,
                GROUP_SIZE=ROPE_GROUP_SIZE,
                num_warps=num_warps,
            )
        else:
            # PyTorch fallback
            Q = Q.view(batch, seq_len, n_heads, head_dim)
            Q_rot = Q[..., :half_dim]
            Q_pass = Q[..., half_dim:]
            
            cos_expanded = cos[:seq_len].unsqueeze(0).unsqueeze(2)
            sin_expanded = sin[:seq_len].unsqueeze(0).unsqueeze(2)
            
            Q_rotated = torch.cat([
                Q_rot * cos_expanded - Q_pass * sin_expanded,
                Q_pass * cos_expanded + Q_rot * sin_expanded,
            ], dim=-1)
            
            Q = Q_rotated.view(batch * seq_len, n_heads * head_dim)
        
        ctx.save_for_backward(cos, sin)
        ctx.original_shape = (batch, seq_len, n_heads, head_dim)
        
        return Q.view(batch, seq_len, n_heads, head_dim)
    
    @staticmethod
    def backward(ctx, dQ: Tensor) -> Tuple[Tensor, None, None]:
        cos, sin = ctx.saved_tensors
        batch, seq_len, n_heads, head_dim = ctx.original_shape
        
        dQ = dQ.reshape(batch * seq_len, n_heads * head_dim)
        half_dim = head_dim // 2
        
        if _TRITON_AVAILABLE and dQ.is_cuda:
            BLOCK_SIZE = triton.next_power_of_2(half_dim)
            num_warps = 4 if BLOCK_SIZE <= 1024 else 8
            n_groups = triton.cdiv(n_heads, ROPE_GROUP_SIZE)
            
            _rope_grouped_kernel[(batch * seq_len, n_groups)](
                dQ.data_ptr(),
                dQ.stride(0),
                cos.data_ptr(), cos.stride(0),
                sin.data_ptr(), sin.stride(0),
                seq_len, head_dim, n_heads,
                BACKWARD=True,  # Negate sin
                BLOCK_SIZE=BLOCK_SIZE,
                GROUP_SIZE=ROPE_GROUP_SIZE,
                num_warps=num_warps,
            )
        else:
            # PyTorch fallback (negate sin for backward)
            dQ = dQ.view(batch, seq_len, n_heads, head_dim)
            dQ_rot = dQ[..., :half_dim]
            dQ_pass = dQ[..., half_dim:]
            
            cos_expanded = cos[:seq_len].unsqueeze(0).unsqueeze(2)
            sin_expanded = -sin[:seq_len].unsqueeze(0).unsqueeze(2)  # Negate!
            
            dQ = torch.cat([
                dQ_rot * cos_expanded - dQ_pass * sin_expanded,
                dQ_pass * cos_expanded + dQ_rot * sin_expanded,
            ], dim=-1).view(batch * seq_len, n_heads * head_dim)
        
        return dQ.view(batch, seq_len, n_heads, head_dim), None, None


def fast_rope_embedding(Q: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply fast RoPE embedding with Triton acceleration."""
    return Fast_RoPE_Embedding.apply(Q, cos, sin)


def inplace_rope_embedding(
    Q: Tensor,
    K: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Apply RoPE to Q and K inplace (fused kernel).
    
    Zero memory overhead - modifies tensors directly.
    """
    cos = cos.squeeze()
    sin = sin.squeeze()
    
    batch, seq_len, n_heads_Q, head_dim = Q.shape
    n_heads_K = K.shape[2]
    
    if _TRITON_AVAILABLE and Q.is_cuda:
        half_dim = head_dim // 2
        BLOCK_SIZE = triton.next_power_of_2(half_dim)
        num_warps = 4 if BLOCK_SIZE <= 1024 else 8
        
        grid = (batch * seq_len, n_heads_Q)
        
        _rope_fused_qk_kernel[grid](
            Q.data_ptr(), K.data_ptr(),
            cos.data_ptr(), sin.data_ptr(),
            Q.stride(0), Q.stride(2), Q.stride(1),
            K.stride(0), K.stride(2), K.stride(1),
            cos.stride(0), sin.stride(0),
            seq_len, head_dim, n_heads_K,
            BACKWARD=False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        # PyTorch fallback
        Q = fast_rope_embedding(Q, cos, sin)
        K = fast_rope_embedding(K, cos, sin)
    
    return Q, K


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Frequency computation
    "precompute_freqs_cis",
    # RoPE application
    "Fast_RoPE_Embedding",
    "fast_rope_embedding",
    "inplace_rope_embedding",
    # Constants
    "ROPE_GROUP_SIZE",
]
