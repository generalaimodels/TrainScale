# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer Module - Triton Kernels
# ════════════════════════════════════════════════════════════════════════════════
# Above-SOTA Triton kernel utilities and optimizations.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import logging
from typing import Optional, Tuple
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Check Triton availability
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass


def is_triton_available() -> bool:
    """Check if Triton is available."""
    return TRITON_AVAILABLE


def get_cuda_autotune_config():
    """Get autotuning configurations for CUDA kernels."""
    if not TRITON_AVAILABLE:
        return []
    return [
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ]


if TRITON_AVAILABLE:
    # ─────────────────────────────────────────────────────────────────────────────
    # Fused LayerNorm Kernel
    # ─────────────────────────────────────────────────────────────────────────────
    
    @triton.jit
    def _layer_norm_fwd_kernel(
        X, Y, W, B, Mean, Rstd,
        stride, N,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        
        # Compute mean
        mean = tl.sum(x, axis=0) / N
        
        # Compute variance
        x_centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(x_centered * x_centered, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        
        # Normalize
        x_hat = x_centered * rstd
        
        # Scale and shift
        w = tl.load(W + cols, mask=mask, other=1.0)
        b = tl.load(B + cols, mask=mask, other=0.0)
        y = x_hat * w + b
        
        tl.store(Y + row * stride + cols, y, mask=mask)
        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)
    
    
    @triton.jit
    def _layer_norm_bwd_kernel(
        DY, X, W, Mean, Rstd, DX, DW, DB,
        stride, N,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        dy = tl.load(DY + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        
        x_hat = (x - mean) * rstd
        
        # Gradient of weight and bias
        dw = dy * x_hat
        db = dy
        
        # Gradient of input
        dx_hat = dy * w
        dx = rstd * (dx_hat - tl.sum(dx_hat, axis=0) / N - 
                     x_hat * tl.sum(dx_hat * x_hat, axis=0) / N)
        
        tl.store(DX + row * stride + cols, dx, mask=mask)
        tl.atomic_add(DW + cols, dw, mask=mask)
        tl.atomic_add(DB + cols, db, mask=mask)
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Fused Softmax Kernel
    # ─────────────────────────────────────────────────────────────────────────────
    
    @triton.jit
    def _softmax_kernel(
        X, Y,
        stride,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        x = tl.load(X + row * stride + cols, mask=mask, other=float("-inf")).to(tl.float32)
        
        # Numerically stable softmax
        x_max = tl.max(x, axis=0)
        x_exp = tl.exp(x - x_max)
        x_sum = tl.sum(tl.where(mask, x_exp, 0.0), axis=0)
        y = x_exp / x_sum
        
        tl.store(Y + row * stride + cols, y, mask=mask)
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Fused GELU Kernel
    # ─────────────────────────────────────────────────────────────────────────────
    
    @triton.jit
    def _gelu_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        
        x = tl.load(X + offs, mask=mask).to(tl.float32)
        
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        cdf = 0.5 * (1.0 + tl.libdevice.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
        y = x * cdf
        
        tl.store(Y + offs, y, mask=mask)
    
    
    @triton.jit
    def _gelu_bwd_kernel(DY, X, DX, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        
        dy = tl.load(DY + offs, mask=mask).to(tl.float32)
        x = tl.load(X + offs, mask=mask).to(tl.float32)
        
        # Derivative of GELU
        tanh_in = 0.7978845608 * (x + 0.044715 * x * x * x)
        tanh_out = tl.libdevice.tanh(tanh_in)
        sech2 = 1.0 - tanh_out * tanh_out
        grad = 0.5 * (1 + tanh_out) + 0.5 * x * sech2 * 0.7978845608 * (1 + 3 * 0.044715 * x * x)
        
        dx = dy * grad
        tl.store(DX + offs, dx, mask=mask)


# ═════════════════════════════════════════════════════════════════════════════════
# PyTorch Wrappers
# ═════════════════════════════════════════════════════════════════════════════════

def fused_layer_norm(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: float = 1e-5,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Triton-fused layer normalization."""
    if not TRITON_AVAILABLE or not x.is_cuda:
        # Fallback to PyTorch
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        rstd = 1.0 / torch.sqrt(var + eps)
        y = (x - mean) * rstd * weight + bias
        return y, mean.squeeze(-1), rstd.squeeze(-1)
    
    M, N = x.shape[0], x.shape[-1]
    y = torch.empty_like(x)
    mean = torch.empty(M, device=x.device, dtype=torch.float32)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    _layer_norm_fwd_kernel[(M,)](
        x, y, weight, bias, mean, rstd,
        x.stride(-2), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y, mean, rstd


def fused_softmax(x: Tensor) -> Tensor:
    """Triton-fused softmax."""
    if not TRITON_AVAILABLE or not x.is_cuda:
        return torch.softmax(x, dim=-1)
    
    M, N = x.shape[0], x.shape[-1]
    y = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    _softmax_kernel[(M,)](x, y, x.stride(-2), N, BLOCK_SIZE=BLOCK_SIZE)
    
    return y


def fused_gelu(x: Tensor) -> Tensor:
    """Triton-fused GELU activation."""
    if not TRITON_AVAILABLE or not x.is_cuda:
        return torch.nn.functional.gelu(x)
    
    y = torch.empty_like(x)
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    _gelu_kernel[grid](x.view(-1), y.view(-1), N, BLOCK_SIZE=BLOCK_SIZE)
    
    return y.view_as(x)


# ═════════════════════════════════════════════════════════════════════════════════
# torch.compile Integration
# ═════════════════════════════════════════════════════════════════════════════════

def compile_model(
    model: torch.nn.Module,
    backend: str = "inductor",
    mode: str = "default",
    fullgraph: bool = False,
) -> torch.nn.Module:
    """
    Compile model with torch.compile for optimization.
    
    Args:
        model: Model to compile
        backend: Compilation backend (inductor, cudagraphs, etc.)
        mode: Optimization mode (default, reduce-overhead, max-autotune)
        fullgraph: Require full graph compilation
        
    Returns:
        Compiled model
    """
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not available (PyTorch < 2.0)")
        return model
    
    try:
        return torch.compile(
            model,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
        )
    except Exception as e:
        logger.warning(f"torch.compile failed: {e}, using eager mode")
        return model


# Import flash attention
from data_pipeline.trainer.kernels.flash_attention import (
    is_flash_attn_available,
    is_flash_attn_triton_available,
    flash_attention,
    naive_attention,
    FlashAttention,
    MultiHeadFlashAttention,
)


__all__ = [
    "is_triton_available",
    "fused_layer_norm",
    "fused_softmax",
    "fused_gelu",
    "compile_model",
    # Flash Attention
    "is_flash_attn_available",
    "is_flash_attn_triton_available",
    "flash_attention",
    "naive_attention",
    "FlashAttention",
    "MultiHeadFlashAttention",
]
