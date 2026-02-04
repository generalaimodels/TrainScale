# ════════════════════════════════════════════════════════════════════════════════
# SOTA FP8 Training Kernels
# ════════════════════════════════════════════════════════════════════════════════
# Implementation Features:
# - Block-wise FP8 quantization (DeepSeek-V3 style)
# - Row-wise activation quantization with dynamic scaling
# - FP8 GEMM with fused scale computation
# - Custom backward with full gradient computation
# - Automatic hardware detection and fallback
# - E4M3/E5M2 format support with numerical stability
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, Any, Union, List
from dataclasses import dataclass
from enum import Enum
import math

# ═════════════════════════════════════════════════════════════════════════════════
# Triton Setup with Hardware Detection
# ═════════════════════════════════════════════════════════════════════════════════

_TRITON_AVAILABLE = False
_FP8_AVAILABLE = False
_DEVICE_CAPABILITY = (0, 0)

try:
    import triton
    import triton.language as tl
    if torch.cuda.is_available():
        _TRITON_AVAILABLE = True
        _DEVICE_CAPABILITY = torch.cuda.get_device_capability()
        _FP8_AVAILABLE = _DEVICE_CAPABILITY >= (8, 9)  # SM89+ (Ada/Hopper)
except ImportError:
    triton = None
    tl = None


def get_autotune_config():
    """Generate autotuning configurations for different GPU architectures."""
    if _DEVICE_CAPABILITY >= (9, 0):  # Hopper
        return [
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        ]
    elif _DEVICE_CAPABILITY >= (8, 0):  # Ampere/Ada
        return [
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        ]
    else:
        return [
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        ]


# ═════════════════════════════════════════════════════════════════════════════════
# FP8 Constants and Configuration
# ═════════════════════════════════════════════════════════════════════════════════

class FP8Format(Enum):
    E4M3 = "e4m3fn"  # Forward pass: higher precision
    E5M2 = "e5m2"    # Backward pass: higher dynamic range


@dataclass
class FP8Config:
    """Configuration for FP8 quantization parameters."""
    e4m3_max: float = 448.0
    e5m2_max: float = 57344.0
    block_size: int = 128
    eps: float = 1e-12
    amax_history_len: int = 16
    amax_compute_algo: str = "max"  # "max" or "most_recent"
    scale_update_interval: int = 1
    
    @property
    def forward_dtype(self) -> torch.dtype:
        return torch.float8_e4m3fn if _FP8_AVAILABLE else torch.bfloat16
    
    @property
    def backward_dtype(self) -> torch.dtype:
        return torch.float8_e5m2 if _FP8_AVAILABLE else torch.bfloat16


FP8_CONFIG = FP8Config()


# ═════════════════════════════════════════════════════════════════════════════════
# Scale Management for FP8 Training
# ═════════════════════════════════════════════════════════════════════════════════

class FP8ScaleManager:
    """
    Manages dynamic scaling factors for FP8 training stability.
    Implements delayed scaling with amax history tracking.
    """
    
    def __init__(
        self,
        config: FP8Config = FP8_CONFIG,
        device: torch.device = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amax_history: List[Tensor] = []
        self.scale: Optional[Tensor] = None
        self.scale_inv: Optional[Tensor] = None
        self._step = 0
    
    def compute_scale(
        self,
        amax: Tensor,
        fp8_max: float,
    ) -> Tuple[Tensor, Tensor]:
        """Compute scale and inverse scale from amax."""
        scale = (fp8_max / amax.clamp(min=self.config.eps)).to(torch.float32)
        scale_inv = amax / fp8_max
        return scale, scale_inv
    
    def update_amax(self, tensor: Tensor) -> Tensor:
        """Update amax history and compute current amax."""
        current_amax = tensor.abs().max().detach()
        self.amax_history.append(current_amax)
        
        if len(self.amax_history) > self.config.amax_history_len:
            self.amax_history.pop(0)
        
        if self.config.amax_compute_algo == "max":
            return torch.stack(self.amax_history).max()
        return current_amax
    
    def get_scale(
        self,
        tensor: Tensor,
        fp8_format: FP8Format = FP8Format.E4M3,
    ) -> Tuple[Tensor, Tensor]:
        """Get scaling factors for tensor quantization."""
        fp8_max = self.config.e4m3_max if fp8_format == FP8Format.E4M3 else self.config.e5m2_max
        amax = self.update_amax(tensor)
        return self.compute_scale(amax, fp8_max)


# ═════════════════════════════════════════════════════════════════════════════════
# Block-wise Quantization Kernel (DeepSeek-V3 Style)
# ═════════════════════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:
    
    @triton.jit
    def _block_quant_fp8_kernel(
        x_ptr,
        y_ptr,
        scale_ptr,
        M, N,
        stride_xm, stride_xn,
        stride_ym, stride_yn,
        stride_sm, stride_sn,
        fp8_max,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        DeepSeek-V3 style block-wise FP8 quantization.
        
        Algorithm:
        1. Load block from global memory
        2. Compute block-wise amax
        3. Compute scale = amax / fp8_max
        4. Quantize: y = x * (fp8_max / amax)
        5. Store quantized values and scales
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        # Block offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        # Boundary mask
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        
        # Load block to shared memory (coalesced access)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        
        # Compute block-wise absolute maximum (reduction)
        abs_x = tl.abs(x)
        block_amax = tl.max(abs_x)
        
        # Compute scale with numerical stability
        scale = block_amax / fp8_max
        scale = tl.where(scale < 1e-12, 1.0, scale)
        scale_inv = tl.where(block_amax < 1e-12, 1.0, fp8_max / block_amax)
        
        # Quantize with saturation
        y = x * scale_inv
        y = tl.maximum(tl.minimum(y, fp8_max), -fp8_max)
        
        # Store quantized block
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask)
        
        # Store block scale
        tl.store(scale_ptr + pid_m * stride_sm + pid_n * stride_sn, scale)
    
    
    @triton.jit
    def _block_dequant_fp8_kernel(
        y_ptr,
        scale_ptr,
        x_ptr,
        M, N,
        stride_ym, stride_yn,
        stride_sm, stride_sn,
        stride_xm, stride_xn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        Block-wise FP8 dequantization.
        
        x = y * scale (per-block)
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        
        # Load block scale
        scale = tl.load(scale_ptr + pid_m * stride_sm + pid_n * stride_sn)
        
        # Load quantized values
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        y = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)
        
        # Dequantize
        x = y * scale
        
        # Store result
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        tl.store(x_ptrs, x.to(x_ptr.dtype.element_ty), mask=mask)
    
    
    @triton.jit
    def _row_quant_fp8_kernel(
        x_ptr,
        y_ptr,
        scale_ptr,
        M, N,
        stride_xm, stride_xn,
        stride_ym, stride_yn,
        fp8_max,
        BLOCK_N: tl.constexpr,
    ):
        """
        Row-wise FP8 quantization for activations.
        
        Each row gets its own scale factor for better precision.
        """
        pid_m = tl.program_id(0)
        
        # Row offset
        offs_n = tl.arange(0, BLOCK_N)
        
        # Initialize max accumulator
        row_max = tl.zeros([1], dtype=tl.float32)
        
        # First pass: compute row maximum
        for start_n in range(0, N, BLOCK_N):
            offs = start_n + offs_n
            mask = offs < N
            x_ptrs = x_ptr + pid_m * stride_xm + offs * stride_xn
            x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
            row_max = tl.maximum(row_max, tl.max(tl.abs(x)))
        
        # Compute scale
        scale = row_max / fp8_max
        scale = tl.where(scale < 1e-12, 1.0, scale)
        scale_inv = tl.where(row_max < 1e-12, 1.0, fp8_max / row_max)
        
        # Second pass: quantize
        for start_n in range(0, N, BLOCK_N):
            offs = start_n + offs_n
            mask = offs < N
            x_ptrs = x_ptr + pid_m * stride_xm + offs * stride_xn
            x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
            
            y = x * scale_inv
            y = tl.maximum(tl.minimum(y, fp8_max), -fp8_max)
            
            y_ptrs = y_ptr + pid_m * stride_ym + offs * stride_yn
            tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask)
        
        # Store row scale
        tl.store(scale_ptr + pid_m, tl.max(scale))


# ═════════════════════════════════════════════════════════════════════════════════
# FP8 GEMM Kernel with Fused Scale
# ═════════════════════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:
    
    @triton.autotune(
        configs=get_autotune_config() if _TRITON_AVAILABLE else [],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def _fp8_gemm_scaled_kernel(
        # Matrix pointers
        A_ptr, B_ptr, C_ptr,
        # Scale pointers
        A_scale_ptr, B_scale_ptr,
        # Dimensions
        M, N, K,
        # Quantization block sizes
        quant_block_m, quant_block_k,
        quant_block_k_b, quant_block_n,
        # Strides
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_asm, stride_ask,
        stride_bsk, stride_bsn,
        # Accumulator dtype
        ACCUMULATOR_DTYPE: tl.constexpr,
        # Block sizes (autotuned)
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """
        FP8 GEMM with fused block-scale computation.
        
        C = (A * A_scale) @ (B * B_scale)
        
        Features:
        - Block-wise scale fusion during accumulation
        - FP32 accumulation for numerical stability
        - L2 cache optimization via grouped iteration
        """
        # Program ID and grid
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        
        # Grouped ordering for L2 locality
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        
        # Block base offsets
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        
        # Pointers for A and B tiles
        a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        
        # Scale pointer base
        a_scale_base = A_scale_ptr + (pid_m * BLOCK_SIZE_M // quant_block_m) * stride_asm
        b_scale_base = B_scale_ptr + (pid_n * BLOCK_SIZE_N // quant_block_n) * stride_bsn
        
        # FP32 accumulator
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        # Main GEMM loop with scale fusion
        for k in range(0, num_pid_k):
            k_start = k * BLOCK_SIZE_K
            k_remaining = K - k_start
            
            # Load A tile with mask
            a_mask = offs_k[None, :] < k_remaining
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            
            # Load B tile with mask
            b_mask = offs_k[:, None] < k_remaining
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            
            # Load scales for current K block
            k_scale_idx = k_start // quant_block_k
            a_scale = tl.load(a_scale_base + k_scale_idx * stride_ask)
            b_scale = tl.load(b_scale_base + k_scale_idx * stride_bsk)
            
            # Combined scale
            combined_scale = a_scale * b_scale
            
            # Accumulate with scale fusion
            accumulator += tl.dot(a, b).to(tl.float32) * combined_scale
            
            # Advance pointers
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        
        # Convert to output dtype and store
        c = accumulator.to(C_ptr.dtype.element_ty)
        
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)
    
    
    @triton.jit
    def _fp8_gemm_rowscale_kernel(
        # Matrix pointers
        A_ptr, B_ptr, C_ptr,
        # Row scales
        A_row_scale_ptr,
        B_col_scale_ptr,
        # Dimensions
        M, N, K,
        # Strides
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Block sizes
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """
        FP8 GEMM with row/column-wise scaling.
        
        C[m,n] = sum_k(A[m,k] * B[k,n]) * A_scale[m] * B_scale[n]
        """
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        
        # Grouped ordering
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        
        # Offsets
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        
        a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        
        # Load row/col scales
        a_scales = tl.load(A_row_scale_ptr + offs_am)
        b_scales = tl.load(B_col_scale_ptr + offs_bn)
        
        # Accumulator
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_remaining = K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
            accumulator += tl.dot(a, b).to(tl.float32)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        
        # Apply scales
        c = accumulator * a_scales[:, None] * b_scales[None, :]
        
        # Store
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c.to(C_ptr.dtype.element_ty), mask=c_mask)


# ═════════════════════════════════════════════════════════════════════════════════
# Python Wrappers with Fallback
# ═════════════════════════════════════════════════════════════════════════════════

def block_quantize_fp8(
    x: Tensor,
    block_size: int = 128,
    fp8_format: FP8Format = FP8Format.E4M3,
) -> Tuple[Tensor, Tensor]:
    """
    Block-wise FP8 quantization (DeepSeek-V3 style).
    
    Args:
        x: Input tensor [M, N] in FP32/BF16/FP16
        block_size: Quantization block size
        fp8_format: E4M3 or E5M2 format
    
    Returns:
        (quantized_tensor, block_scales)
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    M, N = x.shape
    fp8_max = FP8_CONFIG.e4m3_max if fp8_format == FP8Format.E4M3 else FP8_CONFIG.e5m2_max
    fp8_dtype = torch.float8_e4m3fn if fp8_format == FP8Format.E4M3 else torch.float8_e5m2
    
    if not _FP8_AVAILABLE:
        fp8_dtype = torch.bfloat16
    
    x = x.contiguous()
    n_blocks_m = triton.cdiv(M, block_size)
    n_blocks_n = triton.cdiv(N, block_size)
    
    if not _TRITON_AVAILABLE or not x.is_cuda:
        # PyTorch fallback
        pad_m = (block_size - M % block_size) % block_size
        pad_n = (block_size - N % block_size) % block_size
        x_padded = torch.nn.functional.pad(x, (0, pad_n, 0, pad_m))
        
        x_blocks = x_padded.view(n_blocks_m, block_size, n_blocks_n, block_size)
        x_blocks = x_blocks.permute(0, 2, 1, 3).contiguous()
        
        block_max = x_blocks.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=FP8_CONFIG.eps)
        scales = (block_max / fp8_max).squeeze(-1).squeeze(-1)
        
        x_quant = (x_blocks / scales[:, :, None, None]).clamp(-fp8_max, fp8_max)
        x_quant = x_quant.permute(0, 2, 1, 3).reshape(M + pad_m, N + pad_n)[:M, :N]
        
        return x_quant.to(fp8_dtype), scales.to(torch.float32)
    
    # Triton kernel
    y = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
    scales = torch.empty((n_blocks_m, n_blocks_n), dtype=torch.float32, device=x.device)
    
    grid = (n_blocks_m, n_blocks_n)
    _block_quant_fp8_kernel[grid](
        x, y, scales,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        scales.stride(0), scales.stride(1),
        fp8_max,
        BLOCK_M=block_size,
        BLOCK_N=block_size,
    )
    
    return y, scales


def block_dequantize_fp8(
    y: Tensor,
    scales: Tensor,
    block_size: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """
    Block-wise FP8 dequantization.
    
    Args:
        y: Quantized tensor [M, N]
        scales: Block scales [n_blocks_m, n_blocks_n]
        block_size: Quantization block size
        out_dtype: Output dtype
    
    Returns:
        Dequantized tensor [M, N]
    """
    M, N = y.shape
    n_blocks_m, n_blocks_n = scales.shape
    
    if not _TRITON_AVAILABLE or not y.is_cuda:
        # PyTorch fallback
        pad_m = (block_size - M % block_size) % block_size
        pad_n = (block_size - N % block_size) % block_size
        
        y_padded = torch.nn.functional.pad(y.to(out_dtype), (0, pad_n, 0, pad_m))
        y_blocks = y_padded.view(n_blocks_m, block_size, n_blocks_n, block_size)
        y_blocks = y_blocks.permute(0, 2, 1, 3).contiguous()
        
        x_blocks = y_blocks * scales[:, :, None, None].to(out_dtype)
        x = x_blocks.permute(0, 2, 1, 3).reshape(M + pad_m, N + pad_n)[:M, :N]
        
        return x.contiguous()
    
    # Triton kernel
    x = torch.empty((M, N), dtype=out_dtype, device=y.device)
    
    grid = (n_blocks_m, n_blocks_n)
    _block_dequant_fp8_kernel[grid](
        y, scales, x,
        M, N,
        y.stride(0), y.stride(1),
        scales.stride(0), scales.stride(1),
        x.stride(0), x.stride(1),
        BLOCK_M=block_size,
        BLOCK_N=block_size,
    )
    
    return x


def row_quantize_fp8(
    x: Tensor,
    fp8_format: FP8Format = FP8Format.E4M3,
) -> Tuple[Tensor, Tensor]:
    """
    Row-wise FP8 quantization for activations.
    
    Args:
        x: Input tensor [M, N]
        fp8_format: E4M3 or E5M2
    
    Returns:
        (quantized_tensor, row_scales)
    """
    M, N = x.shape
    fp8_max = FP8_CONFIG.e4m3_max if fp8_format == FP8Format.E4M3 else FP8_CONFIG.e5m2_max
    fp8_dtype = torch.float8_e4m3fn if fp8_format == FP8Format.E4M3 else torch.float8_e5m2
    
    if not _FP8_AVAILABLE:
        fp8_dtype = torch.bfloat16
    
    x = x.contiguous()
    
    if not _TRITON_AVAILABLE or not x.is_cuda:
        # PyTorch fallback
        row_max = x.abs().amax(dim=1, keepdim=True).clamp(min=FP8_CONFIG.eps)
        scales = row_max / fp8_max
        x_quant = (x / scales).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        return x_quant, scales.squeeze(1).to(torch.float32)
    
    # Triton kernel
    y = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
    scales = torch.empty(M, dtype=torch.float32, device=x.device)
    
    BLOCK_N = min(triton.next_power_of_2(N), 4096)
    grid = (M,)
    
    _row_quant_fp8_kernel[grid](
        x, y, scales,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        fp8_max,
        BLOCK_N=BLOCK_N,
    )
    
    return y, scales


def fp8_gemm_scaled(
    A: Tensor,
    A_scale: Tensor,
    B: Tensor,
    B_scale: Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    block_size: int = 128,
) -> Tensor:
    """
    FP8 GEMM with block-wise scale fusion.
    
    Args:
        A: FP8 matrix [M, K]
        A_scale: Block scales for A [n_blocks_m, n_blocks_k]
        B: FP8 matrix [K, N]
        B_scale: Block scales for B [n_blocks_k, n_blocks_n]
        out_dtype: Output dtype
        block_size: Quantization block size
    
    Returns:
        C = (A * A_scale) @ (B * B_scale) in out_dtype
    """
    M, K = A.shape
    K_, N = B.shape
    assert K == K_, f"Dimension mismatch: A({M},{K}) @ B({K_},{N})"
    
    if not _TRITON_AVAILABLE or not A.is_cuda:
        # PyTorch fallback
        A_deq = block_dequantize_fp8(A, A_scale, block_size, out_dtype)
        B_deq = block_dequantize_fp8(B, B_scale, block_size, out_dtype)
        return torch.matmul(A_deq, B_deq)
    
    # Triton kernel
    C = torch.empty((M, N), dtype=out_dtype, device=A.device)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    _fp8_gemm_scaled_kernel[grid](
        A, B, C,
        A_scale, B_scale,
        M, N, K,
        block_size, block_size,
        block_size, block_size,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        B_scale.stride(0), B_scale.stride(1),
        ACCUMULATOR_DTYPE=tl.float32,
    )
    
    return C


# ═════════════════════════════════════════════════════════════════════════════════
# FP8 Linear Autograd Function
# ═════════════════════════════════════════════════════════════════════════════════

class FP8BlockLinearFunction(torch.autograd.Function):
    """
    FP8 block-quantized linear with custom forward/backward.
    
    Forward: Y = XW^T + b
    - Quantize X with block-wise scaling
    - Use pre-quantized weights
    - Fused scale computation
    
    Backward:
    - dX = dY @ W
    - dW = dY^T @ X (accumulated in FP32)
    - db = sum(dY)
    """
    
    @staticmethod
    def forward(
        ctx,
        X: Tensor,
        weight_fp8: Tensor,
        weight_scale: Tensor,
        bias: Optional[Tensor],
        block_size: int,
        compute_weight_grad: bool,
    ) -> Tensor:
        # Store original shape
        orig_shape = X.shape
        X_2d = X.view(-1, X.size(-1))
        
        # Quantize activation (row-wise for efficiency)
        X_fp8, X_scale = row_quantize_fp8(X_2d, FP8Format.E4M3)
        
        # Transpose weight for GEMM: [out, in] -> [in, out]
        W_T = weight_fp8.T.contiguous()
        W_scale_T = weight_scale.T.contiguous()
        
        # Perform FP8 GEMM with row-wise X scaling
        M, K = X_2d.shape
        K_, N = W_T.shape
        
        if not _TRITON_AVAILABLE or not X.is_cuda:
            # Fallback
            X_deq = X_2d  # Already in full precision for fallback
            W_deq = block_dequantize_fp8(weight_fp8, weight_scale, block_size, X.dtype)
            output = torch.matmul(X_deq, W_deq.T)
        else:
            # Use row-scale GEMM
            output = torch.empty((M, N), dtype=X.dtype, device=X.device)
            
            # Column scales for weight
            W_col_scale = weight_scale.mean(dim=0)  # Approximate column scale
            
            grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
            
            _fp8_gemm_rowscale_kernel[grid](
                X_fp8, W_T, output,
                X_scale, W_col_scale,
                M, N, K,
                X_fp8.stride(0), X_fp8.stride(1),
                W_T.stride(0), W_T.stride(1),
                output.stride(0), output.stride(1),
                BLOCK_SIZE_M=64,
                BLOCK_SIZE_N=64,
                BLOCK_SIZE_K=32,
                GROUP_SIZE_M=8,
            )
        
        if bias is not None:
            output = output + bias
        
        # Save for backward
        ctx.save_for_backward(X_2d, weight_fp8, weight_scale, bias)
        ctx.block_size = block_size
        ctx.compute_weight_grad = compute_weight_grad
        ctx.orig_shape = orig_shape
        
        return output.view(*orig_shape[:-1], N)
    
    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor,
    ) -> Tuple[Optional[Tensor], ...]:
        X, weight_fp8, weight_scale, bias = ctx.saved_tensors
        block_size = ctx.block_size
        
        grad_output_2d = grad_output.view(-1, grad_output.size(-1))
        
        # Dequantize weight for gradient computation
        W_deq = block_dequantize_fp8(weight_fp8, weight_scale, block_size, grad_output.dtype)
        
        # Gradient w.r.t. input: dX = dY @ W
        grad_X = torch.matmul(grad_output_2d, W_deq)
        grad_X = grad_X.view(ctx.orig_shape)
        
        # Gradient w.r.t. weight: dW = dY^T @ X
        grad_weight = None
        if ctx.compute_weight_grad:
            # Accumulate in FP32 for stability
            grad_weight = torch.matmul(grad_output_2d.T.float(), X.float()).to(grad_output.dtype)
        
        # Gradient w.r.t. bias
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output_2d.sum(dim=0)
        
        return grad_X, grad_weight, None, grad_bias, None, None


def fp8_linear_forward(
    X: Tensor,
    weight_fp8: Tensor,
    weight_scale: Tensor,
    bias: Optional[Tensor] = None,
    block_size: int = 128,
    compute_weight_grad: bool = False,
) -> Tensor:
    """
    Apply FP8 block-quantized linear layer.
    
    Args:
        X: Input [*, in_features]
        weight_fp8: FP8 weight [out_features, in_features]
        weight_scale: Block scales [n_blocks_out, n_blocks_in]
        bias: Optional bias [out_features]
        block_size: Quantization block size
        compute_weight_grad: Whether to compute weight gradients
    
    Returns:
        Output [*, out_features]
    """
    return FP8BlockLinearFunction.apply(
        X, weight_fp8, weight_scale, bias, block_size, compute_weight_grad
    )


# ═════════════════════════════════════════════════════════════════════════════════
# FP8 Linear Module
# ═════════════════════════════════════════════════════════════════════════════════

class FP8Linear(nn.Module):
    """
    FP8 Block-Quantized Linear Layer.
    
    Features:
    - DeepSeek-V3 style block-wise quantization
    - Automatic weight quantization on first forward
    - Optional bias
    - Hardware fallback for non-FP8 GPUs
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 128,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.dtype = dtype
        
        # Initialize weight buffer (will be quantized on first forward)
        self.register_buffer(
            'weight_fp8',
            torch.empty((out_features, in_features), dtype=dtype, device=device),
        )
        self.register_buffer(
            'weight_scale',
            torch.ones((triton.cdiv(out_features, block_size), triton.cdiv(in_features, block_size)), device=device),
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        self._quantized = False
    
    def quantize_weight(self, weight: Tensor) -> None:
        """Quantize weight tensor to FP8."""
        self.weight_fp8, self.weight_scale = block_quantize_fp8(
            weight, self.block_size, FP8Format.E4M3
        )
        self._quantized = True
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        block_size: int = 128,
    ) -> 'FP8Linear':
        """Create FP8Linear from existing nn.Linear."""
        fp8_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            block_size=block_size,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        fp8_linear.quantize_weight(linear.weight.data)
        if linear.bias is not None:
            fp8_linear.bias.data.copy_(linear.bias.data)
        return fp8_linear
    
    def forward(self, X: Tensor) -> Tensor:
        if not self._quantized:
            raise RuntimeError("Weights not quantized. Call quantize_weight() first.")
        
        return fp8_linear_forward(
            X,
            self.weight_fp8,
            self.weight_scale,
            self.bias,
            self.block_size,
            compute_weight_grad=self.training,
        )
    
    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, block_size={self.block_size}, '
            f'quantized={self._quantized}'
        )


# ═════════════════════════════════════════════════════════════════════════════════
# Fast Dequantization Utilities
# ═════════════════════════════════════════════════════════════════════════════════

def fast_dequantize_4bit(
    weight: Tensor,
    quant_state: Any,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """
    Optimized 4-bit weight dequantization.
    
    Supports bitsandbytes and custom formats.
    """
    try:
        import bitsandbytes as bnb
        
        # Use bitsandbytes dequantization
        with torch.cuda.stream(torch.cuda.current_stream()):
            deq = bnb.functional.dequantize_4bit(
                weight,
                quant_state,
                blocksize=quant_state.blocksize if hasattr(quant_state, 'blocksize') else 64,
            )
        return deq.to(out_dtype)
    
    except ImportError:
        raise RuntimeError("bitsandbytes required for 4-bit dequantization")


def matmul_with_lora(
    X: Tensor,
    W: Tensor,
    W_quant_state: Optional[Any],
    lora_A: Optional[Tensor],
    lora_B: Optional[Tensor],
    scaling: float = 1.0,
) -> Tensor:
    """
    Fused matrix multiplication with LoRA.
    
    output = X @ W.T + scaling * (X @ A.T @ B.T)
    
    Handles quantized base weights automatically.
    """
    # Dequantize base weight if needed
    if W_quant_state is not None:
        W = fast_dequantize_4bit(W, W_quant_state, X.dtype)
    
    # Base computation
    output = torch.matmul(X, W.T)
    
    # LoRA computation (fused)
    if lora_A is not None and lora_B is not None:
        # X @ A.T @ B.T = (X @ A.T) @ B.T
        lora_out = torch.matmul(torch.matmul(X, lora_A.T), lora_B.T)
        output = output + scaling * lora_out
    
    return output


# ═════════════════════════════════════════════════════════════════════════════════
# Hardware Capability Detection
# ═════════════════════════════════════════════════════════════════════════════════

def get_fp8_capability() -> dict:
    """
    Detect FP8 hardware capabilities.
    
    Returns:
        Dictionary with capability information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'triton_available': _TRITON_AVAILABLE,
        'fp8_available': _FP8_AVAILABLE,
        'device_capability': _DEVICE_CAPABILITY,
        'device_name': None,
        'recommended_block_size': 128,
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name()
        
        # Adjust block size based on architecture
        if _DEVICE_CAPABILITY >= (9, 0):  # Hopper
            info['recommended_block_size'] = 128
        elif _DEVICE_CAPABILITY >= (8, 9):  # Ada
            info['recommended_block_size'] = 128
        elif _DEVICE_CAPABILITY >= (8, 0):  # Ampere
            info['recommended_block_size'] = 64
    
    return info


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Configuration
    "FP8Format",
    "FP8Config",
    "FP8_CONFIG",
    "FP8ScaleManager",
    # Block quantization
    "block_quantize_fp8",
    "block_dequantize_fp8",
    # Row quantization
    "row_quantize_fp8",
    # GEMM
    "fp8_gemm_scaled",
    # Linear layer
    "FP8BlockLinearFunction",
    "fp8_linear_forward",
    "FP8Linear",
    # Utilities
    "fast_dequantize_4bit",
    "matmul_with_lora",
    "get_fp8_capability",
    # Constants
    "_TRITON_AVAILABLE",
    "_FP8_AVAILABLE",
    "_DEVICE_CAPABILITY",
]