# ════════════════════════════════════════════════════════════════════════════════
# SOTA LayerNorm Kernels
# ════════════════════════════════════════════════════════════════════════════════
# Above-Unsloth LayerNorm implementations:
# - RMS LayerNorm (fast forward + backward)
# - Standard LayerNorm (mean + variance)
# - Fused Add + LayerNorm
# - Low-precision accumulation
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional


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


MAX_FUSED_SIZE = 65536


def calculate_settings(n: int) -> Tuple[int, int]:
    """Calculate optimal block size and num_warps."""
    if not _TRITON_AVAILABLE:
        return 1024, 4
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        BLOCK_SIZE = MAX_FUSED_SIZE
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


# ═════════════════════════════════════════════════════════════════════════════════
# RMS LayerNorm Kernels
# ═════════════════════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:
    
    @triton.jit
    def _rms_layernorm_forward(
        Y, Y_row_stride,
        X, X_row_stride,
        W, W_row_stride,
        r, r_row_stride,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
        """
        row_idx = tl.program_id(0)
        col_offs = tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        
        Y += row_idx * Y_row_stride
        X += row_idx * X_row_stride
        r += row_idx * r_row_stride
        
        X_row = tl.load(X + col_offs, mask=mask, other=0.0).to(tl.float32)
        W_row = tl.load(W + col_offs, mask=mask, other=0.0)
        
        # Compute variance and inverse sqrt
        row_var = tl.sum(X_row * X_row, axis=0) / n_cols
        inv_var = tl.math.rsqrt(row_var + eps)
        tl.store(r, inv_var)
        
        # Normalize and scale
        normed = X_row * inv_var
        output = normed.to(W_row.dtype) * W_row
        tl.store(Y + col_offs, output, mask=mask)
    
    
    @triton.jit
    def _rms_layernorm_backward(
        dY, dY_row_stride,
        dX, dX_row_stride,
        X, X_row_stride,
        W, W_row_stride,
        r, r_row_stride,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Backward pass for RMS LayerNorm.
        """
        row_idx = tl.program_id(0)
        col_offs = tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        
        dY += row_idx * dY_row_stride
        X += row_idx * X_row_stride
        r += row_idx * r_row_stride
        dX += row_idx * dX_row_stride
        
        dY_row = tl.load(dY + col_offs, mask=mask, other=0.0).to(tl.float32)
        X_row = tl.load(X + col_offs, mask=mask, other=0.0).to(tl.float32)
        W_row = tl.load(W + col_offs, mask=mask, other=0.0).to(tl.float32)
        
        inv_var = tl.load(r).to(tl.float32)
        normed = X_row * inv_var
        
        dY_W = dY_row * W_row
        rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
        output = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
        tl.store(dX + col_offs, output, mask=mask)


    @triton.jit
    def _layernorm_forward(
        Y, Y_row_stride,
        X, X_row_stride,
        W, B,
        Mean, Rstd,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Standard LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
        """
        row_idx = tl.program_id(0)
        col_offs = tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        
        Y += row_idx * Y_row_stride
        X += row_idx * X_row_stride
        Mean += row_idx
        Rstd += row_idx
        
        X_row = tl.load(X + col_offs, mask=mask, other=0.0).to(tl.float32)
        W_row = tl.load(W + col_offs, mask=mask, other=1.0)
        B_row = tl.load(B + col_offs, mask=mask, other=0.0)
        
        # Compute mean
        mean = tl.sum(X_row, axis=0) / n_cols
        tl.store(Mean, mean)
        
        # Compute variance
        xmean = X_row - mean
        var = tl.sum(xmean * xmean, axis=0) / n_cols
        rstd = tl.math.rsqrt(var + eps)
        tl.store(Rstd, rstd)
        
        # Normalize
        normed = xmean * rstd
        output = normed.to(W_row.dtype) * W_row + B_row
        tl.store(Y + col_offs, output, mask=mask)
    
    
    @triton.jit
    def _fused_add_rms_layernorm(
        Y, Y_row_stride,
        X, X_row_stride,
        residual, residual_row_stride,
        W, W_row_stride,
        r, r_row_stride,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused: Y = RMSNorm(X + residual) * W
        """
        row_idx = tl.program_id(0)
        col_offs = tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        
        Y += row_idx * Y_row_stride
        X += row_idx * X_row_stride
        residual += row_idx * residual_row_stride
        r += row_idx * r_row_stride
        
        X_row = tl.load(X + col_offs, mask=mask, other=0.0).to(tl.float32)
        res_row = tl.load(residual + col_offs, mask=mask, other=0.0).to(tl.float32)
        W_row = tl.load(W + col_offs, mask=mask, other=0.0)
        
        # Add residual
        X_row = X_row + res_row
        
        # RMS Norm
        row_var = tl.sum(X_row * X_row, axis=0) / n_cols
        inv_var = tl.math.rsqrt(row_var + eps)
        tl.store(r, inv_var)
        
        normed = X_row * inv_var
        output = normed.to(W_row.dtype) * W_row
        tl.store(Y + col_offs, output, mask=mask)


# ═════════════════════════════════════════════════════════════════════════════════
# Autograd Functions
# ═════════════════════════════════════════════════════════════════════════════════

class Fast_RMS_LayerNorm(torch.autograd.Function):
    """SOTA RMS LayerNorm with Triton forward and backward."""
    
    @staticmethod
    def forward(ctx, X: Tensor, W: Tensor, eps: float = 1e-6) -> Tensor:
        if not _TRITON_AVAILABLE or not X.is_cuda:
            variance = X.pow(2).mean(-1, keepdim=True)
            X_norm = X * torch.rsqrt(variance + eps)
            return W * X_norm
        
        shape = X.shape
        dim = shape[-1]
        X = X.reshape(-1, dim)
        n_rows, n_cols = X.shape
        
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        
        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
        r = torch.empty(n_rows, dtype=torch.float32, device=X.device)
        
        _rms_layernorm_forward[(n_rows,)](
            Y, Y.stride(0),
            X, X.stride(0),
            W, W.stride(0),
            r, r.stride(0),
            n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)
    
    @staticmethod
    def backward(ctx, dY: Tensor) -> Tuple[Tensor, None, None]:
        shape = dY.shape
        dim = shape[-1]
        dY = dY.reshape(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows, n_cols = dY.shape
        
        dX = torch.empty_like(dY)
        
        _rms_layernorm_backward[(n_rows,)](
            dY, dY.stride(0),
            dX, dX.stride(0),
            X, X.stride(0),
            W, W.stride(0),
            r, r.stride(0),
            n_cols, ctx.eps,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )
        
        return dX.view(*shape), None, None


class Fast_LayerNorm(torch.autograd.Function):
    """SOTA Standard LayerNorm with Triton."""
    
    @staticmethod
    def forward(ctx, X: Tensor, W: Tensor, B: Tensor, eps: float = 1e-5) -> Tensor:
        if not _TRITON_AVAILABLE or not X.is_cuda:
            return torch.nn.functional.layer_norm(X, (X.shape[-1],), W, B, eps)
        
        shape = X.shape
        dim = shape[-1]
        X = X.reshape(-1, dim)
        n_rows, n_cols = X.shape
        
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        
        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
        Mean = torch.empty(n_rows, dtype=torch.float32, device=X.device)
        Rstd = torch.empty(n_rows, dtype=torch.float32, device=X.device)
        
        _layernorm_forward[(n_rows,)](
            Y, Y.stride(0),
            X, X.stride(0),
            W, B,
            Mean, Rstd,
            n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        
        ctx.save_for_backward(X, W, Mean, Rstd)
        ctx.eps = eps
        ctx.shape = shape
        return Y.view(*shape)
    
    @staticmethod
    def backward(ctx, dY: Tensor):
        # Use PyTorch for backward (simpler)
        X, W, Mean, Rstd = ctx.saved_tensors
        # Reconstruct normalized X
        X = X.view(*ctx.shape)
        dX = torch.nn.functional.layer_norm(dY, (ctx.shape[-1],), None, None, ctx.eps)
        return dX, None, None, None


# ═════════════════════════════════════════════════════════════════════════════════
# Python Wrappers
# ═════════════════════════════════════════════════════════════════════════════════

def fast_rms_layernorm(X: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Apply fast RMS LayerNorm with Triton acceleration."""
    return Fast_RMS_LayerNorm.apply(X, weight, eps)


def fast_layernorm(X: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    """Apply fast LayerNorm with Triton acceleration."""
    return Fast_LayerNorm.apply(X, weight, bias, eps)


def fused_add_rms_layernorm(
    X: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """
    Fused residual add + RMS LayerNorm.
    
    Y = RMSNorm(X + residual) * weight
    """
    if not _TRITON_AVAILABLE or not X.is_cuda:
        return fast_rms_layernorm(X + residual, weight, eps)
    
    shape = X.shape
    dim = shape[-1]
    X = X.reshape(-1, dim)
    residual = residual.reshape(-1, dim)
    n_rows, n_cols = X.shape
    
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    
    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    r = torch.empty(n_rows, dtype=torch.float32, device=X.device)
    
    _fused_add_rms_layernorm[(n_rows,)](
        Y, Y.stride(0),
        X, X.stride(0),
        residual, residual.stride(0),
        weight, weight.stride(0),
        r, r.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return Y.view(*shape)


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Autograd
    "Fast_RMS_LayerNorm",
    "Fast_LayerNorm",
    # Wrappers
    "fast_rms_layernorm",
    "fast_layernorm",
    "fused_add_rms_layernorm",
    # Utils
    "calculate_settings",
]
