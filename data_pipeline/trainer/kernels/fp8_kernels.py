# ════════════════════════════════════════════════════════════════════════════════
# SOTA FP8 Training Kernels - Production Grade Implementation (Fixed)
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, Any, Union, List, Dict, NamedTuple
from dataclasses import dataclass
from enum import Enum, auto
import math
import warnings

# ═════════════════════════════════════════════════════════════════════════════════
# Hardware Detection & Triton Setup
# ═════════════════════════════════════════════════════════════════════════════════

_TRITON_AVAILABLE: bool = False
_FP8_HARDWARE_AVAILABLE: bool = False
_DEVICE_CAPABILITY: Tuple[int, int] = (0, 0)
_DEVICE_NAME: str = "cpu"

try:
    import triton
    import triton.language as tl
    
    if torch.cuda.is_available():
        _TRITON_AVAILABLE = True
        _DEVICE_CAPABILITY = torch.cuda.get_device_capability()
        _DEVICE_NAME = torch.cuda.get_device_name()
        _FP8_HARDWARE_AVAILABLE = _DEVICE_CAPABILITY >= (8, 9)
except ImportError:
    triton = None
    tl = None


# ═════════════════════════════════════════════════════════════════════════════════
# Autotuning Configurations
# ═════════════════════════════════════════════════════════════════════════════════

def _get_gemm_autotune_configs() -> List:
    """Architecture-specific GEMM autotuning configurations."""
    if not _TRITON_AVAILABLE:
        return []
    
    if _DEVICE_CAPABILITY >= (9, 0):  # Hopper
        return [
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        ]
    elif _DEVICE_CAPABILITY >= (8, 9):  # Ada
        return [
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        ]
    elif _DEVICE_CAPABILITY >= (8, 0):  # Ampere
        return [
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        ]
    else:
        return [
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        ]


# ═════════════════════════════════════════════════════════════════════════════════
# FP8 Format Specification
# ═════════════════════════════════════════════════════════════════════════════════

class FP8Format(Enum):
    E4M3 = auto()  # Higher precision: [-448, 448]
    E5M2 = auto()  # Higher range: [-57344, 57344]


class FP8Spec(NamedTuple):
    format: FP8Format
    max_val: float
    dtype: torch.dtype
    mantissa_bits: int
    exponent_bits: int


FP8_E4M3_SPEC = FP8Spec(
    format=FP8Format.E4M3,
    max_val=448.0,
    dtype=torch.float8_e4m3fn if _FP8_HARDWARE_AVAILABLE else torch.bfloat16,
    mantissa_bits=3,
    exponent_bits=4,
)

FP8_E5M2_SPEC = FP8Spec(
    format=FP8Format.E5M2,
    max_val=57344.0,
    dtype=torch.float8_e5m2 if _FP8_HARDWARE_AVAILABLE else torch.bfloat16,
    mantissa_bits=2,
    exponent_bits=5,
)


def get_fp8_spec(fp8_format: FP8Format) -> FP8Spec:
    return FP8_E4M3_SPEC if fp8_format == FP8Format.E4M3 else FP8_E5M2_SPEC


# ═════════════════════════════════════════════════════════════════════════════════
# FP8 Configuration
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class FP8Config:
    block_size: int = 128
    amax_history_len: int = 1024
    amax_compute_algo: str = "max"
    scale_update_interval: int = 1
    margin: float = 0.0
    eps: float = 1e-12
    forward_format: FP8Format = FP8Format.E4M3
    backward_format: FP8Format = FP8Format.E5M2
    
    def __post_init__(self) -> None:
        assert self.block_size > 0 and (self.block_size & (self.block_size - 1)) == 0
        assert self.amax_compute_algo in ("max", "moving_avg", "most_recent")
    
    @property
    def forward_spec(self) -> FP8Spec:
        return get_fp8_spec(self.forward_format)
    
    @property
    def backward_spec(self) -> FP8Spec:
        return get_fp8_spec(self.backward_format)


FP8_CONFIG = FP8Config()


# ═════════════════════════════════════════════════════════════════════════════════
# FP8 Scale Manager
# ═════════════════════════════════════════════════════════════════════════════════

class FP8TensorMeta:
    __slots__ = ('scale', 'scale_inv', 'amax', 'amax_history', 'fp8_spec', '_history_idx')
    
    def __init__(self, fp8_spec: FP8Spec, device: torch.device, history_len: int = 1024):
        self.fp8_spec = fp8_spec
        self.scale = torch.ones(1, device=device, dtype=torch.float32)
        self.scale_inv = torch.ones(1, device=device, dtype=torch.float32)
        self.amax = torch.zeros(1, device=device, dtype=torch.float32)
        self.amax_history = torch.zeros(history_len, device=device, dtype=torch.float32)
        self._history_idx = 0
    
    def update_amax(self, amax: Tensor) -> None:
        self.amax_history[self._history_idx] = amax.detach()
        self._history_idx = (self._history_idx + 1) % len(self.amax_history)
        self.amax = self.amax_history.max()
    
    def compute_scale(self, margin: float = 0.0) -> Tuple[Tensor, Tensor]:
        fp8_max = self.fp8_spec.max_val
        safe_amax = torch.clamp(self.amax, min=FP8_CONFIG.eps)
        scale = safe_amax / fp8_max * (2.0 ** margin)
        scale_inv = 1.0 / scale
        self.scale = scale
        self.scale_inv = scale_inv
        return scale, scale_inv


class FP8ScaleManager:
    def __init__(self, config: FP8Config = FP8_CONFIG, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._tensor_metas: Dict[str, FP8TensorMeta] = {}
        self._step_count = 0
    
    def get_meta(self, name: str, fp8_format: FP8Format) -> FP8TensorMeta:
        if name not in self._tensor_metas:
            self._tensor_metas[name] = FP8TensorMeta(
                get_fp8_spec(fp8_format), self.device, self.config.amax_history_len
            )
        return self._tensor_metas[name]
    
    def update_and_get_scale(self, tensor: Tensor, name: str, fp8_format: FP8Format) -> Tuple[Tensor, Tensor]:
        meta = self.get_meta(name, fp8_format)
        current_amax = tensor.abs().max().detach()
        meta.update_amax(current_amax)
        if self._step_count % self.config.scale_update_interval == 0:
            meta.compute_scale(self.config.margin)
        return meta.scale, meta.scale_inv
    
    def step(self) -> None:
        self._step_count += 1


# ═════════════════════════════════════════════════════════════════════════════════
# Triton Kernels: Block-wise Quantization (FIXED)
# ═════════════════════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:
    
    @triton.jit
    def _block_fp8_quantize_kernel(
        x_ptr,
        y_ptr,
        scale_ptr,
        M,
        N,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_scale_m,
        stride_scale_n,
        FP8_MAX: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Block-wise FP8 quantization kernel (DeepSeek-V3 style)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        mask_m = offs_m < M
        mask_n = offs_n < N
        mask = mask_m[:, None] & mask_n[None, :]
        
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        
        # Block-wise amax
        abs_x = tl.abs(x_fp32)
        block_amax = tl.max(abs_x)
        
        # Scale computation with stability
        safe_amax = tl.maximum(block_amax, EPS)
        scale = safe_amax / FP8_MAX
        scale_inv = FP8_MAX / safe_amax
        
        # Quantize with saturation
        y = x_fp32 * scale_inv
        y = tl.maximum(tl.minimum(y, FP8_MAX), -FP8_MAX)
        
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask)
        
        scale_ptr_block = scale_ptr + pid_m * stride_scale_m + pid_n * stride_scale_n
        tl.store(scale_ptr_block, scale)
    
    
    @triton.jit
    def _block_fp8_dequantize_kernel(
        y_ptr,
        scale_ptr,
        x_ptr,
        M,
        N,
        stride_ym,
        stride_yn,
        stride_scale_m,
        stride_scale_n,
        stride_xm,
        stride_xn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Block-wise FP8 dequantization kernel."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        
        scale = tl.load(scale_ptr + pid_m * stride_scale_m + pid_n * stride_scale_n)
        
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        y = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)
        
        x = y * scale
        
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        tl.store(x_ptrs, x.to(x_ptr.dtype.element_ty), mask=mask)


# ═════════════════════════════════════════════════════════════════════════════════
# Triton Kernels: Row-wise Quantization (FIXED - Critical Fix)
# ═════════════════════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:
    
    @triton.jit
    def _row_fp8_quantize_kernel(
        x_ptr,
        y_ptr,
        scale_ptr,
        M,
        N,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        FP8_MAX: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,  # Compile-time constant for loop bound
    ):
        """
        Row-wise FP8 quantization kernel (FIXED).
        
        Critical fixes:
        1. NUM_BLOCKS as tl.constexpr for loop unrolling
        2. Proper scalar handling for scale values
        3. Correct memory coalescing pattern
        """
        pid_m = tl.program_id(0)
        
        # Base pointers for this row
        x_row_base = x_ptr + pid_m * stride_xm
        y_row_base = y_ptr + pid_m * stride_ym
        
        # ═══════════════════════════════════════════════════════════════════════
        # Pass 1: Compute row-wise amax
        # ═══════════════════════════════════════════════════════════════════════
        row_amax = tl.full([1], value=EPS, dtype=tl.float32)
        
        for block_idx in tl.static_range(NUM_BLOCKS):
            block_start = block_idx * BLOCK_N
            offs_n = block_start + tl.arange(0, BLOCK_N)
            mask = offs_n < N
            
            x_ptrs = x_row_base + offs_n * stride_xn
            x_block = tl.load(x_ptrs, mask=mask, other=0.0)
            x_block_fp32 = x_block.to(tl.float32)
            
            block_max = tl.max(tl.abs(x_block_fp32))
            row_amax = tl.maximum(row_amax, block_max)
        
        # ═══════════════════════════════════════════════════════════════════════
        # Compute scale factors (scalar values)
        # ═══════════════════════════════════════════════════════════════════════
        # Extract scalar from [1] tensor
        row_amax_scalar = tl.max(row_amax)  # Reduce to scalar
        safe_amax = tl.maximum(row_amax_scalar, EPS)
        
        scale = safe_amax / FP8_MAX
        scale_inv = FP8_MAX / safe_amax
        
        # ═══════════════════════════════════════════════════════════════════════
        # Pass 2: Quantize with computed scale
        # ═══════════════════════════════════════════════════════════════════════
        for block_idx in tl.static_range(NUM_BLOCKS):
            block_start = block_idx * BLOCK_N
            offs_n = block_start + tl.arange(0, BLOCK_N)
            mask = offs_n < N
            
            x_ptrs = x_row_base + offs_n * stride_xn
            x_block = tl.load(x_ptrs, mask=mask, other=0.0)
            x_block_fp32 = x_block.to(tl.float32)
            
            # Quantize with saturation clipping
            y_block = x_block_fp32 * scale_inv
            y_block = tl.maximum(tl.minimum(y_block, FP8_MAX), -FP8_MAX)
            
            y_ptrs = y_row_base + offs_n * stride_yn
            tl.store(y_ptrs, y_block.to(y_ptr.dtype.element_ty), mask=mask)
        
        # Store row scale
        tl.store(scale_ptr + pid_m, scale)
    
    
    @triton.jit
    def _row_fp8_quantize_single_pass_kernel(
        x_ptr,
        y_ptr,
        scale_ptr,
        M,
        N,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        FP8_MAX: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        Optimized single-block row quantization for small N.
        
        When N <= BLOCK_N, processes entire row in single pass.
        """
        pid_m = tl.program_id(0)
        
        offs_n = tl.arange(0, BLOCK_N)
        mask = offs_n < N
        
        # Load entire row
        x_ptrs = x_ptr + pid_m * stride_xm + offs_n * stride_xn
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        
        # Compute row amax
        row_amax = tl.max(tl.abs(x_fp32))
        safe_amax = tl.maximum(row_amax, EPS)
        
        # Compute scale
        scale = safe_amax / FP8_MAX
        scale_inv = FP8_MAX / safe_amax
        
        # Quantize
        y = x_fp32 * scale_inv
        y = tl.maximum(tl.minimum(y, FP8_MAX), -FP8_MAX)
        
        # Store
        y_ptrs = y_ptr + pid_m * stride_ym + offs_n * stride_yn
        tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask)
        tl.store(scale_ptr + pid_m, scale)
    
    
    @triton.jit  
    def _row_fp8_dequantize_kernel(
        y_ptr,
        scale_ptr,
        x_ptr,
        M,
        N,
        stride_ym,
        stride_yn,
        stride_xm,
        stride_xn,
        BLOCK_N: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
    ):
        """Row-wise FP8 dequantization kernel."""
        pid_m = tl.program_id(0)
        
        scale = tl.load(scale_ptr + pid_m)
        
        y_row_base = y_ptr + pid_m * stride_ym
        x_row_base = x_ptr + pid_m * stride_xm
        
        for block_idx in tl.static_range(NUM_BLOCKS):
            block_start = block_idx * BLOCK_N
            offs_n = block_start + tl.arange(0, BLOCK_N)
            mask = offs_n < N
            
            y_ptrs = y_row_base + offs_n * stride_yn
            y = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)
            
            x = y * scale
            
            x_ptrs = x_row_base + offs_n * stride_xn
            tl.store(x_ptrs, x.to(x_ptr.dtype.element_ty), mask=mask)


# ═════════════════════════════════════════════════════════════════════════════════
# Triton Kernels: FP8 GEMM with Fused Scaling
# ═════════════════════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:
    
    @triton.autotune(
        configs=_get_gemm_autotune_configs() if _TRITON_AVAILABLE else [],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def _fp8_matmul_block_scaled_kernel(
        A_ptr, B_ptr, C_ptr,
        A_scale_ptr, B_scale_ptr,
        M, N, K,
        QUANT_BLOCK_M: tl.constexpr,
        QUANT_BLOCK_K: tl.constexpr,
        QUANT_BLOCK_K_B: tl.constexpr,
        QUANT_BLOCK_N: tl.constexpr,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_as_m, stride_as_k,
        stride_bs_k, stride_bs_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """FP8 GEMM with block-wise scale fusion."""
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        
        # L2 cache-optimized swizzled ordering
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        
        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        
        a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        num_k_tiles = tl.cdiv(K, BLOCK_K)
        for k_tile in range(0, num_k_tiles):
            k_offset = k_tile * BLOCK_K
            k_remaining = K - k_offset
            
            a_mask = offs_k[None, :] < k_remaining
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            
            b_mask = offs_k[:, None] < k_remaining
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            
            # Load scales
            a_scale_m_idx = pid_m * BLOCK_M // QUANT_BLOCK_M
            a_scale_k_idx = k_offset // QUANT_BLOCK_K
            a_scale = tl.load(A_scale_ptr + a_scale_m_idx * stride_as_m + a_scale_k_idx * stride_as_k)
            
            b_scale_k_idx = k_offset // QUANT_BLOCK_K_B
            b_scale_n_idx = pid_n * BLOCK_N // QUANT_BLOCK_N
            b_scale = tl.load(B_scale_ptr + b_scale_k_idx * stride_bs_k + b_scale_n_idx * stride_bs_n)
            
            combined_scale = a_scale * b_scale
            
            tile_result = tl.dot(a, b).to(tl.float32)
            acc += tile_result * combined_scale
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        
        tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)
    
    
    @triton.autotune(
        configs=_get_gemm_autotune_configs() if _TRITON_AVAILABLE else [],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def _fp8_matmul_row_scaled_kernel(
        A_ptr, B_ptr, C_ptr,
        A_row_scale_ptr,
        B_col_scale_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """FP8 GEMM with row/column-wise scaling."""
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        
        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        
        a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        
        # Load row/column scales
        a_scales = tl.load(A_row_scale_ptr + offs_am)
        b_scales = tl.load(B_col_scale_ptr + offs_bn)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        num_k_tiles = tl.cdiv(K, BLOCK_K)
        for k_tile in range(0, num_k_tiles):
            k_remaining = K - k_tile * BLOCK_K
            
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
            
            acc += tl.dot(a, b).to(tl.float32)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        
        # Apply scales
        c = acc * a_scales[:, None] * b_scales[None, :]
        
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        
        tl.store(c_ptrs, c.to(C_ptr.dtype.element_ty), mask=c_mask)


# ═════════════════════════════════════════════════════════════════════════════════
# Python API: Quantization Functions
# ═════════════════════════════════════════════════════════════════════════════════

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_power_of_2(n: int) -> int:
    """Return next power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def block_quantize_fp8(
    x: Tensor,
    block_size: int = 128,
    fp8_format: FP8Format = FP8Format.E4M3,
) -> Tuple[Tensor, Tensor]:
    """
    Block-wise FP8 quantization (DeepSeek-V3 style).
    
    Args:
        x: Input tensor [M, N]
        block_size: Quantization block size
        fp8_format: E4M3 or E5M2
    
    Returns:
        (quantized_tensor, block_scales)
    """
    if x.dim() != 2:
        raise ValueError(f"Expected 2D input, got {x.dim()}D")
    
    if not x.is_contiguous():
        x = x.contiguous()
    
    M, N = x.shape
    fp8_spec = get_fp8_spec(fp8_format)
    fp8_dtype = fp8_spec.dtype
    fp8_max = fp8_spec.max_val
    
    n_blocks_m = _ceil_div(M, block_size)
    n_blocks_n = _ceil_div(N, block_size)
    
    if _TRITON_AVAILABLE and x.is_cuda:
        y = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
        scales = torch.empty((n_blocks_m, n_blocks_n), dtype=torch.float32, device=x.device)
        
        grid = (n_blocks_m, n_blocks_n)
        
        _block_fp8_quantize_kernel[grid](
            x, y, scales,
            M, N,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            scales.stride(0), scales.stride(1),
            fp8_max,
            FP8_CONFIG.eps,
            BLOCK_M=block_size,
            BLOCK_N=block_size,
        )
        
        return y, scales
    
    # PyTorch fallback
    pad_m = (block_size - M % block_size) % block_size
    pad_n = (block_size - N % block_size) % block_size
    
    if pad_m > 0 or pad_n > 0:
        x = F.pad(x, (0, pad_n, 0, pad_m))
    
    M_padded, N_padded = x.shape
    
    x_blocks = x.view(n_blocks_m, block_size, n_blocks_n, block_size)
    x_blocks = x_blocks.permute(0, 2, 1, 3).contiguous()
    
    block_amax = x_blocks.abs().amax(dim=(-2, -1), keepdim=True)
    block_amax = block_amax.clamp(min=FP8_CONFIG.eps)
    
    scales = (block_amax / fp8_max).squeeze(-1).squeeze(-1)
    
    x_quant = (x_blocks / scales[:, :, None, None]).clamp(-fp8_max, fp8_max)
    
    x_quant = x_quant.permute(0, 2, 1, 3).reshape(M_padded, N_padded)
    x_quant = x_quant[:M, :N].contiguous()
    
    return x_quant.to(fp8_dtype), scales.to(torch.float32)


def block_dequantize_fp8(
    y: Tensor,
    scales: Tensor,
    block_size: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Block-wise FP8 dequantization."""
    M, N = y.shape
    n_blocks_m, n_blocks_n = scales.shape
    
    if _TRITON_AVAILABLE and y.is_cuda:
        x = torch.empty((M, N), dtype=out_dtype, device=y.device)
        
        grid = (n_blocks_m, n_blocks_n)
        
        _block_fp8_dequantize_kernel[grid](
            y, scales, x,
            M, N,
            y.stride(0), y.stride(1),
            scales.stride(0), scales.stride(1),
            x.stride(0), x.stride(1),
            BLOCK_M=block_size,
            BLOCK_N=block_size,
        )
        
        return x
    
    # PyTorch fallback
    pad_m = (block_size - M % block_size) % block_size
    pad_n = (block_size - N % block_size) % block_size
    
    y_padded = F.pad(y.to(out_dtype), (0, pad_n, 0, pad_m)) if (pad_m > 0 or pad_n > 0) else y.to(out_dtype)
    M_padded, N_padded = y_padded.shape
    
    y_blocks = y_padded.view(n_blocks_m, block_size, n_blocks_n, block_size)
    y_blocks = y_blocks.permute(0, 2, 1, 3).contiguous()
    
    x_blocks = y_blocks * scales[:, :, None, None].to(out_dtype)
    
    x = x_blocks.permute(0, 2, 1, 3).reshape(M_padded, N_padded)
    
    return x[:M, :N].contiguous()


def row_quantize_fp8(
    x: Tensor,
    fp8_format: FP8Format = FP8Format.E4M3,
) -> Tuple[Tensor, Tensor]:
    """
    Row-wise FP8 quantization for activations.
    
    Args:
        x: Input tensor [M, N] or [*, M, N]
        fp8_format: E4M3 or E5M2
    
    Returns:
        (quantized_tensor, row_scales)
    """
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.reshape(-1, x.size(-1))
    
    if not x.is_contiguous():
        x = x.contiguous()
    
    M, N = x.shape
    fp8_spec = get_fp8_spec(fp8_format)
    fp8_dtype = fp8_spec.dtype
    fp8_max = fp8_spec.max_val
    
    if _TRITON_AVAILABLE and x.is_cuda:
        y = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
        scales = torch.empty(M, dtype=torch.float32, device=x.device)
        
        # Determine optimal block size and compute NUM_BLOCKS at compile time
        BLOCK_N = min(_next_power_of_2(N), 4096)
        NUM_BLOCKS = _ceil_div(N, BLOCK_N)
        
        grid = (M,)
        
        if N <= BLOCK_N:
            # Use single-pass kernel for small N
            _row_fp8_quantize_single_pass_kernel[grid](
                x, y, scales,
                M, N,
                x.stride(0), x.stride(1),
                y.stride(0), y.stride(1),
                fp8_max,
                FP8_CONFIG.eps,
                BLOCK_N=BLOCK_N,
            )
        else:
            # Use multi-block kernel with compile-time NUM_BLOCKS
            _row_fp8_quantize_kernel[grid](
                x, y, scales,
                M, N,
                x.stride(0), x.stride(1),
                y.stride(0), y.stride(1),
                fp8_max,
                FP8_CONFIG.eps,
                BLOCK_N=BLOCK_N,
                NUM_BLOCKS=NUM_BLOCKS,
            )
        
        if len(orig_shape) > 2:
            y = y.view(*orig_shape)
            scales = scales.view(*orig_shape[:-1])
        
        return y, scales
    
    # PyTorch fallback
    row_amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=FP8_CONFIG.eps)
    scales = row_amax / fp8_max
    
    x_quant = (x / scales).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    
    if len(orig_shape) > 2:
        x_quant = x_quant.view(*orig_shape)
        scales = scales.view(*orig_shape[:-1])
    else:
        scales = scales.squeeze(-1)
    
    return x_quant, scales.to(torch.float32)


def row_dequantize_fp8(
    y: Tensor,
    scales: Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Row-wise FP8 dequantization."""
    if scales.dim() == 1:
        scales = scales.unsqueeze(-1)
    return (y.to(out_dtype) * scales.to(out_dtype))


# ═════════════════════════════════════════════════════════════════════════════════
# Python API: FP8 GEMM Functions
# ═════════════════════════════════════════════════════════════════════════════════

def fp8_matmul_block_scaled(
    A: Tensor,
    A_scale: Tensor,
    B: Tensor,
    B_scale: Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    block_size: int = 128,
) -> Tensor:
    """FP8 matrix multiplication with block-wise scaling."""
    M, K = A.shape
    K_B, N = B.shape
    
    if K != K_B:
        raise ValueError(f"Dimension mismatch: A[{M},{K}] @ B[{K_B},{N}]")
    
    if _TRITON_AVAILABLE and A.is_cuda:
        C = torch.empty((M, N), dtype=out_dtype, device=A.device)
        
        grid = lambda META: (
            _ceil_div(M, META['BLOCK_M']) * _ceil_div(N, META['BLOCK_N']),
        )
        
        _fp8_matmul_block_scaled_kernel[grid](
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
        )
        
        return C
    
    A_deq = block_dequantize_fp8(A, A_scale, block_size, out_dtype)
    B_deq = block_dequantize_fp8(B, B_scale, block_size, out_dtype)
    return torch.matmul(A_deq, B_deq)


def fp8_matmul_row_scaled(
    A: Tensor,
    A_scale: Tensor,
    B: Tensor,
    B_scale: Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """FP8 matrix multiplication with row/column scaling."""
    M, K = A.shape
    K_B, N = B.shape
    
    if K != K_B:
        raise ValueError(f"Dimension mismatch: A[{M},{K}] @ B[{K_B},{N}]")
    
    if _TRITON_AVAILABLE and A.is_cuda:
        C = torch.empty((M, N), dtype=out_dtype, device=A.device)
        
        grid = lambda META: (
            _ceil_div(M, META['BLOCK_M']) * _ceil_div(N, META['BLOCK_N']),
        )
        
        _fp8_matmul_row_scaled_kernel[grid](
            A, B, C,
            A_scale, B_scale,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )
        
        return C
    
    A_deq = row_dequantize_fp8(A, A_scale, out_dtype)
    B_deq = row_dequantize_fp8(B.T, B_scale, out_dtype).T
    return torch.matmul(A_deq, B_deq)


# ═════════════════════════════════════════════════════════════════════════════════
# FP8 Linear Autograd Function
# ═════════════════════════════════════════════════════════════════════════════════

class FP8LinearFunction(torch.autograd.Function):
    """Custom autograd for FP8 linear with E4M3 forward, E5M2 backward."""
    
    @staticmethod
    def forward(
        ctx,
        X: Tensor,
        weight_fp8: Tensor,
        weight_scale: Tensor,
        bias: Optional[Tensor],
        block_size: int,
    ) -> Tensor:
        orig_shape = X.shape
        X_2d = X.reshape(-1, X.size(-1))
        
        X_fp8, X_scale = row_quantize_fp8(X_2d, FP8Format.E4M3)
        
        n_blocks_k = weight_scale.size(1)
        W_col_scale = weight_scale.mean(dim=0)
        
        out_features = weight_fp8.size(0)
        W_col_scale_expanded = W_col_scale.repeat_interleave(block_size)[:out_features]
        
        W_T = weight_fp8.T.contiguous()
        
        if _TRITON_AVAILABLE and X.is_cuda:
            M, K = X_2d.shape
            N = out_features
            
            output = torch.empty((M, N), dtype=X.dtype, device=X.device)
            
            grid = lambda META: (
                _ceil_div(M, META['BLOCK_M']) * _ceil_div(N, META['BLOCK_N']),
            )
            
            _fp8_matmul_row_scaled_kernel[grid](
                X_fp8, W_T, output,
                X_scale, W_col_scale_expanded,
                M, N, K,
                X_fp8.stride(0), X_fp8.stride(1),
                W_T.stride(0), W_T.stride(1),
                output.stride(0), output.stride(1),
            )
        else:
            X_deq = row_dequantize_fp8(X_fp8, X_scale, X.dtype)
            W_deq = block_dequantize_fp8(weight_fp8, weight_scale, block_size, X.dtype)
            output = torch.matmul(X_deq, W_deq.T)
        
        if bias is not None:
            output = output + bias
        
        ctx.save_for_backward(X_2d, weight_fp8, weight_scale, bias)
        ctx.block_size = block_size
        ctx.orig_shape = orig_shape
        
        return output.view(*orig_shape[:-1], out_features)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        X, weight_fp8, weight_scale, bias = ctx.saved_tensors
        block_size = ctx.block_size
        orig_shape = ctx.orig_shape
        
        grad_output_2d = grad_output.reshape(-1, grad_output.size(-1))
        
        W = block_dequantize_fp8(weight_fp8, weight_scale, block_size, grad_output.dtype)
        
        grad_X = torch.matmul(grad_output_2d, W)
        grad_X = grad_X.view(orig_shape)
        
        grad_weight = torch.matmul(
            grad_output_2d.T.float(),
            X.float()
        ).to(grad_output.dtype)
        
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output_2d.sum(dim=0)
        
        return grad_X, grad_weight, None, grad_bias, None


def fp8_linear_forward(
    X: Tensor,
    weight_fp8: Tensor,
    weight_scale: Tensor,
    bias: Optional[Tensor] = None,
    block_size: int = 128,
) -> Tensor:
    """Functional interface for FP8 linear."""
    return FP8LinearFunction.apply(X, weight_fp8, weight_scale, bias, block_size)


# ═════════════════════════════════════════════════════════════════════════════════
# FP8 Linear Module
# ═════════════════════════════════════════════════════════════════════════════════

class FP8Linear(nn.Module):
    """FP8 Block-Quantized Linear Layer."""
    
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
        
        n_blocks_out = _ceil_div(out_features, block_size)
        n_blocks_in = _ceil_div(in_features, block_size)
        
        fp8_dtype = FP8_E4M3_SPEC.dtype
        
        self.register_buffer(
            'weight_fp8',
            torch.zeros((out_features, in_features), dtype=fp8_dtype, device=device),
        )
        self.register_buffer(
            'weight_scale',
            torch.ones((n_blocks_out, n_blocks_in), dtype=torch.float32, device=device),
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter('bias', None)
        
        self._is_quantized = False
    
    def quantize_weight(self, weight: Tensor) -> None:
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(f"Weight shape mismatch")
        
        weight_fp8, weight_scale = block_quantize_fp8(
            weight.to(self.dtype), self.block_size, FP8Format.E4M3
        )
        
        self.weight_fp8.copy_(weight_fp8)
        self.weight_scale.copy_(weight_scale)
        self._is_quantized = True
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, block_size: int = 128) -> 'FP8Linear':
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
        if not self._is_quantized:
            raise RuntimeError("Weights not quantized. Call quantize_weight() first.")
        
        return fp8_linear_forward(X, self.weight_fp8, self.weight_scale, self.bias, self.block_size)
    
    def extra_repr(self) -> str:
        return f'in={self.in_features}, out={self.out_features}, bs={self.block_size}, quant={self._is_quantized}'


# ═════════════════════════════════════════════════════════════════════════════════
# Utilities
# ═════════════════════════════════════════════════════════════════════════════════

def get_fp8_capability() -> Dict[str, Any]:
    """Get FP8 hardware capability information."""
    return {
        'cuda_available': torch.cuda.is_available(),
        'triton_available': _TRITON_AVAILABLE,
        'fp8_hardware': _FP8_HARDWARE_AVAILABLE,
        'device_capability': _DEVICE_CAPABILITY,
        'device_name': _DEVICE_NAME,
        'recommended_block_size': 128 if _DEVICE_CAPABILITY >= (8, 9) else 64,
    }


def convert_model_to_fp8(
    model: nn.Module,
    block_size: int = 128,
    exclude_layers: Optional[List[str]] = None,
) -> nn.Module:
    """Convert all Linear layers to FP8Linear."""
    exclude_layers = exclude_layers or []
    
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name not in exclude_layers:
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            parent = model.get_submodule(parent_name) if parent_name else model
            
            fp8_layer = FP8Linear.from_linear(module, block_size)
            setattr(parent, attr_name, fp8_layer)
    
    return model


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    '_TRITON_AVAILABLE', '_FP8_HARDWARE_AVAILABLE', '_DEVICE_CAPABILITY',
    'FP8Format', 'FP8Spec', 'FP8Config', 'FP8_CONFIG',
    'FP8TensorMeta', 'FP8ScaleManager',
    'block_quantize_fp8', 'block_dequantize_fp8',
    'row_quantize_fp8', 'row_dequantize_fp8',
    'fp8_matmul_block_scaled', 'fp8_matmul_row_scaled',
    'FP8LinearFunction', 'fp8_linear_forward', 'FP8Linear',
    'get_fp8_capability', 'convert_model_to_fp8',
]