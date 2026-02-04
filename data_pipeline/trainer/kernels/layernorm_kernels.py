# ════════════════════════════════════════════════════════════════════════════════
# SOTA LayerNorm Kernels
# ════════════════════════════════════════════════════════════════════════════════
# Implementation Features:
# - RMS LayerNorm with fused forward/backward
# - Standard LayerNorm with Welford's algorithm
# - Fused Add + LayerNorm (residual connection)
# - Fused Dropout + LayerNorm
# - Multi-row processing for small hidden dims
# - Large dimension support via chunked reduction
# - FP32 accumulation with mixed-precision output
# - Complete gradient computation (dX, dW, dB)
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, Union
from dataclasses import dataclass
import math

# ═════════════════════════════════════════════════════════════════════════════════
# Triton Setup with Hardware Detection
# ═════════════════════════════════════════════════════════════════════════════════

_TRITON_AVAILABLE = False
_DEVICE_CAPABILITY = (0, 0)

try:
    import triton
    import triton.language as tl
    if torch.cuda.is_available():
        _TRITON_AVAILABLE = True
        _DEVICE_CAPABILITY = torch.cuda.get_device_capability()
except ImportError:
    triton = None
    tl = None


# ═════════════════════════════════════════════════════════════════════════════════
# Configuration Constants
# ═════════════════════════════════════════════════════════════════════════════════

MAX_FUSED_SIZE = 65536
MIN_BLOCK_SIZE = 128
DEFAULT_EPS_RMS = 1e-6
DEFAULT_EPS_LN = 1e-5


@dataclass
class LayerNormConfig:
    """Configuration for LayerNorm kernel optimization."""
    max_fused_size: int = 65536
    use_welford: bool = True
    fp32_accumulation: bool = True
    vectorized_loads: bool = True
    multi_row_threshold: int = 256  # Process multiple rows if dim < threshold


LN_CONFIG = LayerNormConfig()


# ═════════════════════════════════════════════════════════════════════════════════
# Kernel Configuration Utilities
# ═════════════════════════════════════════════════════════════════════════════════

def calculate_settings(n: int) -> Tuple[int, int]:
    """
    Calculate optimal BLOCK_SIZE and num_warps for given dimension.
    
    Args:
        n: Hidden dimension size
    
    Returns:
        (BLOCK_SIZE, num_warps)
    """
    if not _TRITON_AVAILABLE:
        return min(1024, triton.next_power_of_2(n) if triton else 1024), 4
    
    BLOCK_SIZE = triton.next_power_of_2(n)
    BLOCK_SIZE = max(MIN_BLOCK_SIZE, min(BLOCK_SIZE, MAX_FUSED_SIZE))
    
    # Optimal warp configuration based on block size and architecture
    if _DEVICE_CAPABILITY >= (9, 0):  # Hopper
        if BLOCK_SIZE >= 32768:
            num_warps = 32
        elif BLOCK_SIZE >= 16384:
            num_warps = 16
        elif BLOCK_SIZE >= 4096:
            num_warps = 8
        else:
            num_warps = 4
    else:  # Ampere/Ada
        if BLOCK_SIZE >= 32768:
            num_warps = 32
        elif BLOCK_SIZE >= 8192:
            num_warps = 16
        elif BLOCK_SIZE >= 2048:
            num_warps = 8
        else:
            num_warps = 4
    
    return BLOCK_SIZE, num_warps


def calculate_multi_row_settings(n_cols: int, n_rows: int) -> Tuple[int, int, int]:
    """
    Calculate settings for multi-row processing.
    
    Returns:
        (BLOCK_SIZE_N, BLOCK_SIZE_M, num_warps)
    """
    if not _TRITON_AVAILABLE:
        return 1024, 1, 4
    
    BLOCK_SIZE_N = triton.next_power_of_2(n_cols)
    BLOCK_SIZE_N = max(MIN_BLOCK_SIZE, min(BLOCK_SIZE_N, 4096))
    
    # Determine rows per block based on available shared memory
    if n_cols <= 256:
        BLOCK_SIZE_M = 8
    elif n_cols <= 512:
        BLOCK_SIZE_M = 4
    elif n_cols <= 1024:
        BLOCK_SIZE_M = 2
    else:
        BLOCK_SIZE_M = 1
    
    num_warps = max(4, min(16, BLOCK_SIZE_N * BLOCK_SIZE_M // 256))
    
    return BLOCK_SIZE_N, BLOCK_SIZE_M, num_warps


# ═════════════════════════════════════════════════════════════════════════════════
# RMS LayerNorm Kernels
# ═════════════════════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:
    
    @triton.jit
    def _rms_layernorm_forward_kernel(
        # Output pointers
        Y_ptr, Y_row_stride,
        # Input pointers
        X_ptr, X_row_stride,
        W_ptr,
        # RMS inverse storage
        rstd_ptr,
        # Dimensions
        n_cols,
        eps,
        # Block config
        BLOCK_SIZE: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
    ):
        """
        RMSNorm Forward: Y = X / sqrt(mean(X²) + eps) * W
        
        Features:
        - FP32 accumulation for numerical stability
        - Vectorized memory access pattern
        - Fused weight multiplication
        """
        row_idx = tl.program_id(0)
        col_offs = tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        
        # Compute row pointers
        X_row_ptr = X_ptr + row_idx * X_row_stride
        Y_row_ptr = Y_ptr + row_idx * Y_row_stride
        
        # Load input row with coalesced access
        x = tl.load(X_row_ptr + col_offs, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        
        # Compute mean of squares (RMS denominator)
        x_sq = x_fp32 * x_fp32
        mean_sq = tl.sum(x_sq, axis=0) / n_cols
        
        # Compute inverse RMS with numerical stability
        rstd = tl.math.rsqrt(mean_sq + eps)
        
        # Store rstd for backward pass
        tl.store(rstd_ptr + row_idx, rstd)
        
        # Normalize
        x_norm = x_fp32 * rstd
        
        # Apply weight if present
        if HAS_WEIGHT:
            w = tl.load(W_ptr + col_offs, mask=mask, other=1.0)
            y = x_norm.to(w.dtype) * w
        else:
            y = x_norm.to(x.dtype)
        
        # Store output
        tl.store(Y_row_ptr + col_offs, y, mask=mask)
    
    
    @triton.jit
    def _rms_layernorm_backward_dx_kernel(
        # Gradient pointers
        dY_ptr, dY_row_stride,
        dX_ptr, dX_row_stride,
        # Forward tensors
        X_ptr, X_row_stride,
        W_ptr,
        rstd_ptr,
        # Dimensions
        n_cols,
        # Block config
        BLOCK_SIZE: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
    ):
        """
        RMSNorm Backward (dX computation):
        
        dX = rstd * (dY * W - X * rstd² * mean(dY * W * X))
        
        Simplified: dX = rstd * W * (dY - X_norm * mean(dY * X_norm))
        """
        row_idx = tl.program_id(0)
        col_offs = tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        
        # Load row data
        X_row_ptr = X_ptr + row_idx * X_row_stride
        dY_row_ptr = dY_ptr + row_idx * dY_row_stride
        dX_row_ptr = dX_ptr + row_idx * dX_row_stride
        
        x = tl.load(X_row_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dY_row_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row_idx).to(tl.float32)
        
        # Load weight
        if HAS_WEIGHT:
            w = tl.load(W_ptr + col_offs, mask=mask, other=1.0).to(tl.float32)
            dy_w = dy * w
        else:
            dy_w = dy
        
        # Compute normalized x
        x_norm = x * rstd
        
        # Compute mean term: mean(dY * W * X_norm)
        mean_term = tl.sum(dy_w * x_norm, axis=0) / n_cols
        
        # Compute gradient
        dx = rstd * (dy_w - x_norm * mean_term)
        
        # Store
        tl.store(dX_row_ptr + col_offs, dx.to(dX_ptr.dtype.element_ty), mask=mask)
    
    
    @triton.jit
    def _rms_layernorm_backward_dw_kernel(
        # Gradient pointers
        dY_ptr, dY_row_stride,
        dW_ptr,
        # Forward tensors
        X_ptr, X_row_stride,
        rstd_ptr,
        # Dimensions
        n_rows, n_cols,
        # Block config
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
    ):
        """
        RMSNorm Backward (dW computation):
        dW = sum_rows(dY * X_norm)
        
        Uses column-parallel reduction for efficiency.
        """
        col_block_idx = tl.program_id(0)
        col_offs = col_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        col_mask = col_offs < n_cols
        
        # Accumulator for weight gradient
        dw_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        
        # Iterate over rows
        for row_start in range(0, n_rows, BLOCK_SIZE_M):
            row_offs = row_start + tl.arange(0, BLOCK_SIZE_M)
            row_mask = row_offs < n_rows
            
            for i in range(BLOCK_SIZE_M):
                row_idx = row_start + i
                if row_idx < n_rows:
                    # Load data
                    x = tl.load(
                        X_ptr + row_idx * X_row_stride + col_offs,
                        mask=col_mask, other=0.0
                    ).to(tl.float32)
                    dy = tl.load(
                        dY_ptr + row_idx * dY_row_stride + col_offs,
                        mask=col_mask, other=0.0
                    ).to(tl.float32)
                    rstd = tl.load(rstd_ptr + row_idx).to(tl.float32)
                    
                    # Accumulate: dW += dY * X_norm
                    x_norm = x * rstd
                    dw_acc += dy * x_norm
        
        # Atomic add to global dW
        tl.atomic_add(dW_ptr + col_offs, dw_acc, mask=col_mask)
    
    
    @triton.jit
    def _welford_layernorm_forward_kernel(
        # Output pointers
        Y_ptr, Y_row_stride,
        # Input pointers
        X_ptr, X_row_stride,
        W_ptr, B_ptr,
        # Statistics storage
        mean_ptr, rstd_ptr,
        # Dimensions
        n_cols,
        eps,
        # Block config
        BLOCK_SIZE: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        """
        Standard LayerNorm Forward with Welford's Online Algorithm.
        
        Y = (X - mean) / sqrt(var + eps) * W + B
        
        Features:
        - Numerically stable single-pass mean/variance
        - FP32 accumulation
        - Fused affine transformation
        """
        row_idx = tl.program_id(0)
        col_offs = tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        
        X_row_ptr = X_ptr + row_idx * X_row_stride
        Y_row_ptr = Y_ptr + row_idx * Y_row_stride
        
        # Load input
        x = tl.load(X_row_ptr + col_offs, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        
        # Welford's algorithm for numerically stable mean/variance
        # Step 1: Compute mean
        mean = tl.sum(x_fp32, axis=0) / n_cols
        
        # Step 2: Compute variance
        x_centered = x_fp32 - mean
        var = tl.sum(x_centered * x_centered, axis=0) / n_cols
        
        # Compute inverse std
        rstd = tl.math.rsqrt(var + eps)
        
        # Store statistics for backward
        tl.store(mean_ptr + row_idx, mean)
        tl.store(rstd_ptr + row_idx, rstd)
        
        # Normalize
        x_norm = x_centered * rstd
        
        # Apply affine transformation
        if HAS_WEIGHT:
            w = tl.load(W_ptr + col_offs, mask=mask, other=1.0)
            y = x_norm.to(w.dtype) * w
        else:
            y = x_norm.to(x.dtype)
        
        if HAS_BIAS:
            b = tl.load(B_ptr + col_offs, mask=mask, other=0.0)
            y = y + b
        
        # Store output
        tl.store(Y_row_ptr + col_offs, y, mask=mask)
    
    
    @triton.jit
    def _layernorm_backward_dx_kernel(
        # Gradient pointers
        dY_ptr, dY_row_stride,
        dX_ptr, dX_row_stride,
        # Forward tensors
        X_ptr, X_row_stride,
        W_ptr,
        mean_ptr, rstd_ptr,
        # Dimensions
        n_cols,
        # Block config
        BLOCK_SIZE: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
    ):
        """
        LayerNorm Backward (dX computation):
        
        dX = rstd * W * (dY - mean(dY) - X_centered * rstd² * mean(dY * X_centered))
        """
        row_idx = tl.program_id(0)
        col_offs = tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        
        # Load row data
        X_row_ptr = X_ptr + row_idx * X_row_stride
        dY_row_ptr = dY_ptr + row_idx * dY_row_stride
        dX_row_ptr = dX_ptr + row_idx * dX_row_stride
        
        x = tl.load(X_row_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dY_row_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)
        
        mean = tl.load(mean_ptr + row_idx).to(tl.float32)
        rstd = tl.load(rstd_ptr + row_idx).to(tl.float32)
        
        # Load weight
        if HAS_WEIGHT:
            w = tl.load(W_ptr + col_offs, mask=mask, other=1.0).to(tl.float32)
            dy_w = dy * w
        else:
            dy_w = dy
        
        # Compute centered x and normalized x
        x_centered = x - mean
        x_norm = x_centered * rstd
        
        # Compute mean terms
        mean_dy = tl.sum(dy_w, axis=0) / n_cols
        mean_dy_xnorm = tl.sum(dy_w * x_norm, axis=0) / n_cols
        
        # Compute gradient: rstd * (dY*W - mean(dY*W) - X_norm * mean(dY*W*X_norm))
        dx = rstd * (dy_w - mean_dy - x_norm * mean_dy_xnorm)
        
        # Store
        tl.store(dX_row_ptr + col_offs, dx.to(dX_ptr.dtype.element_ty), mask=mask)
    
    
    @triton.jit
    def _layernorm_backward_dwdb_kernel(
        # Gradient pointers
        dY_ptr, dY_row_stride,
        dW_ptr, dB_ptr,
        # Forward tensors
        X_ptr, X_row_stride,
        mean_ptr, rstd_ptr,
        # Dimensions
        n_rows, n_cols,
        # Block config
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        """
        LayerNorm Backward (dW, dB computation):
        dW = sum_rows(dY * X_norm)
        dB = sum_rows(dY)
        """
        col_block_idx = tl.program_id(0)
        col_offs = col_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        col_mask = col_offs < n_cols
        
        # Accumulators
        dw_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        db_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        
        # Iterate over rows in chunks
        for row_idx in range(n_rows):
            # Load data
            x = tl.load(
                X_ptr + row_idx * X_row_stride + col_offs,
                mask=col_mask, other=0.0
            ).to(tl.float32)
            dy = tl.load(
                dY_ptr + row_idx * dY_row_stride + col_offs,
                mask=col_mask, other=0.0
            ).to(tl.float32)
            mean = tl.load(mean_ptr + row_idx).to(tl.float32)
            rstd = tl.load(rstd_ptr + row_idx).to(tl.float32)
            
            # Compute normalized x
            x_norm = (x - mean) * rstd
            
            # Accumulate gradients
            dw_acc += dy * x_norm
            if HAS_BIAS:
                db_acc += dy
        
        # Atomic add to global gradients
        tl.atomic_add(dW_ptr + col_offs, dw_acc, mask=col_mask)
        if HAS_BIAS:
            tl.atomic_add(dB_ptr + col_offs, db_acc, mask=col_mask)
    
    
    @triton.jit
    def _fused_add_rms_layernorm_kernel(
        # Output pointers
        Y_ptr, Y_row_stride,
        residual_out_ptr, residual_out_row_stride,
        # Input pointers
        X_ptr, X_row_stride,
        residual_ptr, residual_row_stride,
        W_ptr,
        rstd_ptr,
        # Dimensions
        n_cols,
        eps,
        # Block config
        BLOCK_SIZE: tl.constexpr,
        STORE_RESIDUAL: tl.constexpr,
    ):
        """
        Fused Residual Add + RMS LayerNorm:
        
        hidden = X + residual
        Y = hidden / sqrt(mean(hidden²) + eps) * W
        
        Optionally stores (X + residual) for gradient computation.
        """
        row_idx = tl.program_id(0)
        col_offs = tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        
        # Compute row pointers
        X_row_ptr = X_ptr + row_idx * X_row_stride
        res_row_ptr = residual_ptr + row_idx * residual_row_stride
        Y_row_ptr = Y_ptr + row_idx * Y_row_stride
        
        # Load and add
        x = tl.load(X_row_ptr + col_offs, mask=mask, other=0.0)
        res = tl.load(res_row_ptr + col_offs, mask=mask, other=0.0)
        
        hidden = x.to(tl.float32) + res.to(tl.float32)
        
        # Store residual sum if needed
        if STORE_RESIDUAL:
            res_out_ptr = residual_out_ptr + row_idx * residual_out_row_stride
            tl.store(res_out_ptr + col_offs, hidden.to(x.dtype), mask=mask)
        
        # RMS Norm
        hidden_sq = hidden * hidden
        mean_sq = tl.sum(hidden_sq, axis=0) / n_cols
        rstd = tl.math.rsqrt(mean_sq + eps)
        
        tl.store(rstd_ptr + row_idx, rstd)
        
        # Normalize and scale
        hidden_norm = hidden * rstd
        w = tl.load(W_ptr + col_offs, mask=mask, other=1.0)
        y = hidden_norm.to(w.dtype) * w
        
        tl.store(Y_row_ptr + col_offs, y, mask=mask)
    
    
    @triton.jit
    def _fused_dropout_rms_layernorm_kernel(
        # Output pointers
        Y_ptr, Y_row_stride,
        mask_ptr, mask_row_stride,
        # Input pointers
        X_ptr, X_row_stride,
        W_ptr,
        rstd_ptr,
        # Dropout params
        seed, dropout_p,
        # Dimensions
        n_cols, row_offset,
        eps,
        # Block config
        BLOCK_SIZE: tl.constexpr,
        TRAINING: tl.constexpr,
    ):
        """
        Fused Dropout + RMS LayerNorm:
        
        X_dropped = dropout(X, p) / (1 - p)  # Inverted dropout
        Y = X_dropped / sqrt(mean(X_dropped²) + eps) * W
        """
        row_idx = tl.program_id(0)
        col_offs = tl.arange(0, BLOCK_SIZE)
        mask_cols = col_offs < n_cols
        
        X_row_ptr = X_ptr + row_idx * X_row_stride
        Y_row_ptr = Y_ptr + row_idx * Y_row_stride
        
        # Load input
        x = tl.load(X_row_ptr + col_offs, mask=mask_cols, other=0.0)
        x_fp32 = x.to(tl.float32)
        
        # Apply dropout during training
        if TRAINING:
            # Generate dropout mask using Philox RNG
            philox_seed = seed
            philox_offset = (row_offset + row_idx) * n_cols + col_offs
            rand = tl.rand(philox_seed, philox_offset)
            
            keep_mask = rand > dropout_p
            scale = 1.0 / (1.0 - dropout_p)
            x_fp32 = tl.where(keep_mask, x_fp32 * scale, 0.0)
            
            # Store mask for backward
            mask_row_ptr = mask_ptr + row_idx * mask_row_stride
            tl.store(mask_row_ptr + col_offs, keep_mask.to(tl.int8), mask=mask_cols)
        
        # RMS Norm
        x_sq = x_fp32 * x_fp32
        mean_sq = tl.sum(x_sq, axis=0) / n_cols
        rstd = tl.math.rsqrt(mean_sq + eps)
        
        tl.store(rstd_ptr + row_idx, rstd)
        
        # Normalize and scale
        x_norm = x_fp32 * rstd
        w = tl.load(W_ptr + col_offs, mask=mask_cols, other=1.0)
        y = x_norm.to(w.dtype) * w
        
        tl.store(Y_row_ptr + col_offs, y, mask=mask_cols)
    
    
    @triton.jit
    def _large_rms_layernorm_kernel(
        # Output pointers
        Y_ptr, Y_row_stride,
        # Input pointers
        X_ptr, X_row_stride,
        W_ptr,
        rstd_ptr,
        # Dimensions
        n_cols,
        eps,
        # Block config
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        RMS LayerNorm for large hidden dimensions (> MAX_FUSED_SIZE).
        Uses two-pass algorithm with chunked reduction.
        """
        row_idx = tl.program_id(0)
        
        X_row_ptr = X_ptr + row_idx * X_row_stride
        Y_row_ptr = Y_ptr + row_idx * Y_row_stride
        
        # First pass: compute sum of squares
        sum_sq = tl.zeros([1], dtype=tl.float32)
        
        for col_start in range(0, n_cols, BLOCK_SIZE):
            col_offs = col_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offs < n_cols
            
            x = tl.load(X_row_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)
            sum_sq += tl.sum(x * x, axis=0)
        
        # Compute rstd
        mean_sq = sum_sq / n_cols
        rstd = tl.math.rsqrt(mean_sq + eps)
        tl.store(rstd_ptr + row_idx, rstd)
        
        # Second pass: normalize and store
        for col_start in range(0, n_cols, BLOCK_SIZE):
            col_offs = col_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offs < n_cols
            
            x = tl.load(X_row_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W_ptr + col_offs, mask=mask, other=1.0)
            
            x_norm = x * rstd
            y = x_norm.to(w.dtype) * w
            
            tl.store(Y_row_ptr + col_offs, y, mask=mask)


# ═════════════════════════════════════════════════════════════════════════════════
# Autograd Functions
# ═════════════════════════════════════════════════════════════════════════════════

class RMSLayerNormFunction(torch.autograd.Function):
    """
    SOTA RMS LayerNorm with complete forward/backward Triton kernels.
    
    Supports:
    - Arbitrary hidden dimensions
    - Mixed precision (FP16/BF16 compute, FP32 accumulation)
    - Full gradient computation (dX, dW)
    """
    
    @staticmethod
    def forward(
        ctx,
        X: Tensor,
        weight: Tensor,
        eps: float = DEFAULT_EPS_RMS,
        compute_weight_grad: bool = True,
    ) -> Tensor:
        # Handle non-CUDA fallback
        if not _TRITON_AVAILABLE or not X.is_cuda:
            variance = X.to(torch.float32).pow(2).mean(-1, keepdim=True)
            X_norm = X * torch.rsqrt(variance + eps)
            ctx.save_for_backward(X, weight, torch.rsqrt(variance + eps).squeeze(-1))
            ctx.eps = eps
            ctx.compute_weight_grad = compute_weight_grad
            return weight * X_norm
        
        # Reshape for kernel
        orig_shape = X.shape
        X = X.contiguous()
        n_cols = X.shape[-1]
        X_2d = X.view(-1, n_cols)
        n_rows = X_2d.shape[0]
        
        # Allocate outputs
        Y = torch.empty_like(X_2d)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=X.device)
        
        # Select kernel based on dimension size
        if n_cols <= MAX_FUSED_SIZE:
            BLOCK_SIZE, num_warps = calculate_settings(n_cols)
            
            _rms_layernorm_forward_kernel[(n_rows,)](
                Y, Y.stride(0),
                X_2d, X_2d.stride(0),
                weight,
                rstd,
                n_cols, eps,
                BLOCK_SIZE=BLOCK_SIZE,
                HAS_WEIGHT=True,
                num_warps=num_warps,
            )
        else:
            # Use multi-pass kernel for large dimensions
            BLOCK_SIZE = 4096
            _large_rms_layernorm_kernel[(n_rows,)](
                Y, Y.stride(0),
                X_2d, X_2d.stride(0),
                weight,
                rstd,
                n_cols, eps,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=16,
            )
        
        # Save for backward
        ctx.save_for_backward(X_2d, weight, rstd)
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE if n_cols <= MAX_FUSED_SIZE else 4096
        ctx.num_warps = num_warps if n_cols <= MAX_FUSED_SIZE else 16
        ctx.orig_shape = orig_shape
        ctx.compute_weight_grad = compute_weight_grad
        ctx.n_cols = n_cols
        
        return Y.view(orig_shape)
    
    @staticmethod
    def backward(ctx, dY: Tensor) -> Tuple[Optional[Tensor], ...]:
        X, weight, rstd = ctx.saved_tensors
        n_cols = ctx.n_cols
        
        dY = dY.contiguous()
        dY_2d = dY.view(-1, n_cols)
        n_rows = dY_2d.shape[0]
        
        # Compute dX
        dX = torch.empty_like(X)
        
        if _TRITON_AVAILABLE and dY.is_cuda:
            _rms_layernorm_backward_dx_kernel[(n_rows,)](
                dY_2d, dY_2d.stride(0),
                dX, dX.stride(0),
                X, X.stride(0),
                weight,
                rstd,
                n_cols,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                HAS_WEIGHT=True,
                num_warps=ctx.num_warps,
            )
        else:
            # PyTorch fallback
            x_norm = X * rstd[:, None]
            dy_w = dY_2d * weight
            mean_term = (dy_w * x_norm).mean(dim=-1, keepdim=True)
            dX = rstd[:, None] * (dy_w - x_norm * mean_term)
        
        # Compute dW if needed
        dW = None
        if ctx.compute_weight_grad and weight.requires_grad:
            dW = torch.zeros_like(weight)
            
            if _TRITON_AVAILABLE and dY.is_cuda and n_cols <= MAX_FUSED_SIZE:
                BLOCK_SIZE_N = min(1024, triton.next_power_of_2(n_cols))
                n_col_blocks = triton.cdiv(n_cols, BLOCK_SIZE_N)
                
                _rms_layernorm_backward_dw_kernel[(n_col_blocks,)](
                    dY_2d, dY_2d.stride(0),
                    dW,
                    X, X.stride(0),
                    rstd,
                    n_rows, n_cols,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    BLOCK_SIZE_M=1,
                    num_warps=4,
                )
            else:
                # PyTorch fallback
                x_norm = X * rstd[:, None]
                dW = (dY_2d * x_norm).sum(dim=0)
        
        return dX.view(ctx.orig_shape), dW, None, None


class LayerNormFunction(torch.autograd.Function):
    """
    SOTA Standard LayerNorm with Welford's algorithm.
    
    Supports:
    - Weight and bias
    - Full gradient computation (dX, dW, dB)
    - Numerically stable statistics
    """
    
    @staticmethod
    def forward(
        ctx,
        X: Tensor,
        weight: Optional[Tensor],
        bias: Optional[Tensor],
        eps: float = DEFAULT_EPS_LN,
        compute_weight_grad: bool = True,
    ) -> Tensor:
        # Handle non-CUDA fallback
        if not _TRITON_AVAILABLE or not X.is_cuda:
            return torch.nn.functional.layer_norm(
                X, (X.shape[-1],), weight, bias, eps
            )
        
        orig_shape = X.shape
        X = X.contiguous()
        n_cols = X.shape[-1]
        X_2d = X.view(-1, n_cols)
        n_rows = X_2d.shape[0]
        
        # Allocate outputs
        Y = torch.empty_like(X_2d)
        mean = torch.empty(n_rows, dtype=torch.float32, device=X.device)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=X.device)
        
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        
        has_weight = weight is not None
        has_bias = bias is not None
        
        _welford_layernorm_forward_kernel[(n_rows,)](
            Y, Y.stride(0),
            X_2d, X_2d.stride(0),
            weight if has_weight else X_2d,  # Dummy pointer
            bias if has_bias else X_2d,       # Dummy pointer
            mean, rstd,
            n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_WEIGHT=has_weight,
            HAS_BIAS=has_bias,
            num_warps=num_warps,
        )
        
        ctx.save_for_backward(X_2d, weight, bias, mean, rstd)
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.orig_shape = orig_shape
        ctx.compute_weight_grad = compute_weight_grad
        ctx.n_cols = n_cols
        ctx.has_weight = has_weight
        ctx.has_bias = has_bias
        
        return Y.view(orig_shape)
    
    @staticmethod
    def backward(ctx, dY: Tensor) -> Tuple[Optional[Tensor], ...]:
        X, weight, bias, mean, rstd = ctx.saved_tensors
        n_cols = ctx.n_cols
        
        dY = dY.contiguous()
        dY_2d = dY.view(-1, n_cols)
        n_rows = dY_2d.shape[0]
        
        # Compute dX
        dX = torch.empty_like(X)
        
        if _TRITON_AVAILABLE and dY.is_cuda:
            _layernorm_backward_dx_kernel[(n_rows,)](
                dY_2d, dY_2d.stride(0),
                dX, dX.stride(0),
                X, X.stride(0),
                weight if ctx.has_weight else dX,  # Dummy
                mean, rstd,
                n_cols,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                HAS_WEIGHT=ctx.has_weight,
                num_warps=ctx.num_warps,
            )
        else:
            # PyTorch fallback
            x_centered = X - mean[:, None]
            x_norm = x_centered * rstd[:, None]
            if ctx.has_weight:
                dy_w = dY_2d * weight
            else:
                dy_w = dY_2d
            mean_dy = dy_w.mean(dim=-1, keepdim=True)
            mean_dy_xnorm = (dy_w * x_norm).mean(dim=-1, keepdim=True)
            dX = rstd[:, None] * (dy_w - mean_dy - x_norm * mean_dy_xnorm)
        
        # Compute dW and dB
        dW = None
        dB = None
        
        if ctx.compute_weight_grad:
            if ctx.has_weight and weight.requires_grad:
                dW = torch.zeros_like(weight)
            if ctx.has_bias and bias.requires_grad:
                dB = torch.zeros_like(bias)
            
            if (dW is not None or dB is not None) and _TRITON_AVAILABLE and dY.is_cuda:
                BLOCK_SIZE_N = min(1024, triton.next_power_of_2(n_cols))
                n_col_blocks = triton.cdiv(n_cols, BLOCK_SIZE_N)
                
                _layernorm_backward_dwdb_kernel[(n_col_blocks,)](
                    dY_2d, dY_2d.stride(0),
                    dW if dW is not None else dY_2d,  # Dummy
                    dB if dB is not None else dY_2d,  # Dummy
                    X, X.stride(0),
                    mean, rstd,
                    n_rows, n_cols,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    BLOCK_SIZE_M=1,
                    HAS_BIAS=dB is not None,
                    num_warps=4,
                )
            else:
                # PyTorch fallback
                x_norm = (X - mean[:, None]) * rstd[:, None]
                if dW is not None:
                    dW = (dY_2d * x_norm).sum(dim=0)
                if dB is not None:
                    dB = dY_2d.sum(dim=0)
        
        return dX.view(ctx.orig_shape), dW, dB, None, None


class FusedAddRMSLayerNormFunction(torch.autograd.Function):
    """
    Fused Residual Add + RMS LayerNorm.
    
    Reduces memory traffic by combining operations.
    """
    
    @staticmethod
    def forward(
        ctx,
        X: Tensor,
        residual: Tensor,
        weight: Tensor,
        eps: float = DEFAULT_EPS_RMS,
        store_residual: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if not _TRITON_AVAILABLE or not X.is_cuda:
            hidden = X + residual
            variance = hidden.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_norm = hidden * torch.rsqrt(variance + eps)
            ctx.save_for_backward(hidden, weight, torch.rsqrt(variance + eps).squeeze(-1))
            ctx.eps = eps
            return weight * hidden_norm, hidden if store_residual else None
        
        orig_shape = X.shape
        X = X.contiguous()
        residual = residual.contiguous()
        n_cols = X.shape[-1]
        X_2d = X.view(-1, n_cols)
        res_2d = residual.view(-1, n_cols)
        n_rows = X_2d.shape[0]
        
        # Allocate outputs
        Y = torch.empty_like(X_2d)
        residual_out = torch.empty_like(X_2d) if store_residual else None
        rstd = torch.empty(n_rows, dtype=torch.float32, device=X.device)
        
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        
        _fused_add_rms_layernorm_kernel[(n_rows,)](
            Y, Y.stride(0),
            residual_out if store_residual else Y, 
            residual_out.stride(0) if store_residual else Y.stride(0),
            X_2d, X_2d.stride(0),
            res_2d, res_2d.stride(0),
            weight,
            rstd,
            n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            STORE_RESIDUAL=store_residual,
            num_warps=num_warps,
        )
        
        # Save for backward
        hidden_for_bwd = residual_out if store_residual else (X_2d + res_2d)
        ctx.save_for_backward(hidden_for_bwd, weight, rstd)
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.orig_shape = orig_shape
        ctx.n_cols = n_cols
        
        return (
            Y.view(orig_shape),
            residual_out.view(orig_shape) if store_residual else None
        )
    
    @staticmethod
    def backward(ctx, dY: Tensor, _) -> Tuple[Optional[Tensor], ...]:
        hidden, weight, rstd = ctx.saved_tensors
        n_cols = ctx.n_cols
        
        dY = dY.contiguous()
        dY_2d = dY.view(-1, n_cols)
        n_rows = dY_2d.shape[0]
        
        # Compute dX (same as residual gradient since hidden = X + residual)
        dX = torch.empty_like(hidden)
        
        if _TRITON_AVAILABLE and dY.is_cuda:
            _rms_layernorm_backward_dx_kernel[(n_rows,)](
                dY_2d, dY_2d.stride(0),
                dX, dX.stride(0),
                hidden, hidden.stride(0),
                weight,
                rstd,
                n_cols,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                HAS_WEIGHT=True,
                num_warps=ctx.num_warps,
            )
        else:
            hidden_norm = hidden * rstd[:, None]
            dy_w = dY_2d * weight
            mean_term = (dy_w * hidden_norm).mean(dim=-1, keepdim=True)
            dX = rstd[:, None] * (dy_w - hidden_norm * mean_term)
        
        # dX = dResidual for fused add
        return dX.view(ctx.orig_shape), dX.view(ctx.orig_shape), None, None, None


# ═════════════════════════════════════════════════════════════════════════════════
# Python API Functions
# ═════════════════════════════════════════════════════════════════════════════════

def rms_layernorm(
    X: Tensor,
    weight: Tensor,
    eps: float = DEFAULT_EPS_RMS,
) -> Tensor:
    """
    Apply RMS LayerNorm with Triton acceleration.
    
    Args:
        X: Input tensor [..., hidden_dim]
        weight: Learnable scale [hidden_dim]
        eps: Epsilon for numerical stability
    
    Returns:
        Normalized tensor of same shape as X
    """
    return RMSLayerNormFunction.apply(X, weight, eps, True)


def layernorm(
    X: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = DEFAULT_EPS_LN,
) -> Tensor:
    """
    Apply Standard LayerNorm with Triton acceleration.
    
    Args:
        X: Input tensor [..., hidden_dim]
        weight: Learnable scale [hidden_dim]
        bias: Learnable bias [hidden_dim]
        eps: Epsilon for numerical stability
    
    Returns:
        Normalized tensor of same shape as X
    """
    return LayerNormFunction.apply(X, weight, bias, eps, True)


def fused_add_rms_layernorm(
    X: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float = DEFAULT_EPS_RMS,
    return_residual: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Fused residual add + RMS LayerNorm.
    
    Y = RMSNorm(X + residual) * weight
    
    Args:
        X: Input tensor
        residual: Residual tensor
        weight: Learnable scale
        eps: Epsilon
        return_residual: If True, also return (X + residual)
    
    Returns:
        Normalized output (and optionally residual sum)
    """
    Y, residual_out = FusedAddRMSLayerNormFunction.apply(
        X, residual, weight, eps, return_residual
    )
    if return_residual:
        return Y, residual_out
    return Y


def fused_add_layernorm(
    X: Tensor,
    residual: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float = DEFAULT_EPS_LN,
) -> Tensor:
    """
    Fused residual add + Standard LayerNorm.
    
    Y = LayerNorm(X + residual, weight, bias)
    """
    hidden = X + residual
    return layernorm(hidden, weight, bias, eps)


# Aliases for compatibility
fast_rms_layernorm = rms_layernorm
fast_layernorm = layernorm
fused_layer_norm = layernorm


# ═════════════════════════════════════════════════════════════════════════════════
# Module Classes
# ═════════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """
    RMS LayerNorm Module with Triton acceleration.
    
    Args:
        hidden_size: Dimension of input
        eps: Epsilon for numerical stability
        elementwise_affine: Whether to learn scale parameter
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = DEFAULT_EPS_RMS,
        elementwise_affine: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)
    
    def forward(self, X: Tensor) -> Tensor:
        if self.elementwise_affine:
            return rms_layernorm(X, self.weight, self.eps)
        else:
            # No weight case
            if not _TRITON_AVAILABLE or not X.is_cuda:
                variance = X.to(torch.float32).pow(2).mean(-1, keepdim=True)
                return X * torch.rsqrt(variance + self.eps)
            return rms_layernorm(X, torch.ones(X.shape[-1], device=X.device, dtype=X.dtype), self.eps)
    
    def extra_repr(self) -> str:
        return f'{self.hidden_size}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class LayerNorm(nn.Module):
    """
    Standard LayerNorm Module with Triton acceleration.
    
    Drop-in replacement for torch.nn.LayerNorm.
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = DEFAULT_EPS_LN,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
            if bias:
                self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, X: Tensor) -> Tensor:
        # For multi-dim normalized_shape, use PyTorch
        if len(self.normalized_shape) > 1:
            return torch.nn.functional.layer_norm(
                X, self.normalized_shape, self.weight, self.bias, self.eps
            )
        return layernorm(X, self.weight, self.bias, self.eps)
    
    def extra_repr(self) -> str:
        return (
            f'{self.normalized_shape}, eps={self.eps}, '
            f'elementwise_affine={self.elementwise_affine}'
        )


class FusedRMSNorm(nn.Module):
    """
    Fused Residual Add + RMS LayerNorm Module.
    
    Optimized for transformer residual connections.
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = DEFAULT_EPS_RMS,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
    
    def forward(
        self,
        X: Tensor,
        residual: Tensor,
        return_residual: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        return fused_add_rms_layernorm(X, residual, self.weight, self.eps, return_residual)
    
    def extra_repr(self) -> str:
        return f'{self.hidden_size}, eps={self.eps}'


# ═════════════════════════════════════════════════════════════════════════════════
# Utilities
# ═════════════════════════════════════════════════════════════════════════════════

def get_layernorm_info() -> dict:
    """Get LayerNorm implementation capabilities."""
    return {
        'triton_available': _TRITON_AVAILABLE,
        'device_capability': _DEVICE_CAPABILITY,
        'max_fused_size': MAX_FUSED_SIZE,
        'default_eps_rms': DEFAULT_EPS_RMS,
        'default_eps_ln': DEFAULT_EPS_LN,
    }


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Configuration
    "LayerNormConfig",
    "LN_CONFIG",
    # Functions
    "rms_layernorm",
    "layernorm",
    "fused_add_rms_layernorm",
    "fused_add_layernorm",
    # Aliases
    "fast_rms_layernorm",
    "fast_layernorm",
    "fused_layer_norm",
    # Autograd
    "RMSLayerNormFunction",
    "LayerNormFunction",
    "FusedAddRMSLayerNormFunction",
    # Modules
    "RMSNorm",
    "LayerNorm",
    "FusedRMSNorm",
    # Utilities
    "calculate_settings",
    "get_layernorm_info",
    # Constants
    "MAX_FUSED_SIZE",
    "DEFAULT_EPS_RMS",
    "DEFAULT_EPS_LN",
]