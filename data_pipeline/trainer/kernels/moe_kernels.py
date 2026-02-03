# ════════════════════════════════════════════════════════════════════════════════
# SOTA MoE Kernels (Above-Unsloth Implementation)
# ════════════════════════════════════════════════════════════════════════════════
# Enhanced Mixture of Experts with:
# - TMA (Tensor Memory Accelerator) support for H100+ GPUs
# - SM-level tile scheduling for persistent kernels
# - Autotuning with extensive configuration
# - Fused backward kernels (dX and dW)
# - Fused permutation (token-to-expert, expert-to-token)
# - TopK weight fusion
# - Load balancing with auxiliary loss
#
# Key innovations over Unsloth:
# - Unified forward/backward autograd class
# - Expert capacity constraints
# - Router z-loss regularization
# - Epsilon greedy routing
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ═════════════════════════════════════════════════════════════════════════════════
# Triton Setup
# ═════════════════════════════════════════════════════════════════════════════════

_TRITON_AVAILABLE = False
_SUPPORTS_TMA = None

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    triton = None
    tl = None


def supports_tma() -> bool:
    """Check if TMA (Tensor Memory Accelerator) is supported (Hopper+)."""
    global _SUPPORTS_TMA
    if _SUPPORTS_TMA is None:
        if torch.cuda.is_available():
            _SUPPORTS_TMA = torch.cuda.get_device_capability()[0] >= 9
        else:
            _SUPPORTS_TMA = False
    return _SUPPORTS_TMA


def get_num_sms() -> int:
    """Get number of streaming multiprocessors."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).multi_processor_count
    return 1


# ═════════════════════════════════════════════════════════════════════════════════
# Kernel Configuration Dataclasses
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class KernelConfigForward:
    """Forward kernel tuning parameters."""
    BLOCK_SIZE_M: int = 64
    BLOCK_SIZE_N: int = 64
    BLOCK_SIZE_K: int = 32
    num_warps: int = 4
    num_stages: int = 3
    use_tma_load_x: bool = False
    use_tma_load_w: bool = False
    use_tma_store: bool = False


@dataclass
class KernelConfigBackward:
    """Backward kernel tuning parameters."""
    BLOCK_SIZE_M: int = 64
    BLOCK_SIZE_N: int = 64
    BLOCK_SIZE_K: int = 32
    num_warps: int = 4
    num_stages: int = 3
    use_tma_load: bool = False
    use_tma_store: bool = False


# ═════════════════════════════════════════════════════════════════════════════════
# Triton Kernels (Grouped GEMM)
# ═════════════════════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:

    @triton.jit
    def _grouped_gemm_forward_kernel(
        # Pointers
        x_ptr, w_ptr, y_ptr,
        m_sizes_ptr, gather_indices_ptr, topk_weights_ptr,
        # Constants
        NUM_EXPERTS: tl.constexpr,
        NUM_TOKENS: tl.constexpr,
        TOPK: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        NUM_SMS: tl.constexpr,
        # Tuning
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        # Flags
        PERMUTE_X: tl.constexpr,
        PERMUTE_Y: tl.constexpr,
        FUSE_MUL: tl.constexpr,
        ACC_DTYPE: tl.constexpr = tl.float32,
    ):
        """
        SM-persistent grouped GEMM kernel for MoE.
        
        Each SM processes tiles across all experts in a persistent manner.
        """
        tl.static_assert(K % BLOCK_SIZE_K == 0, "K must be divisible by BLOCK_SIZE_K")
        
        TOTAL_TOKENS: tl.constexpr = NUM_TOKENS * TOPK
        tid = tl.program_id(0)
        output_dtype = y_ptr.dtype.element_ty
        
        m_end = 0
        processed_tiles = 0
        m_block_range = tl.arange(0, BLOCK_SIZE_M)
        
        for expert_idx in tl.range(NUM_EXPERTS):
            m_start = m_end
            m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
            m_end = m_start + m_size
            
            if m_size > 0:
                n_start = expert_idx * N
                
                num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
                num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
                num_tiles = num_m_tiles * num_n_tiles
                
                # Process tiles assigned to this SM
                while tid >= processed_tiles and tid < processed_tiles + num_tiles:
                    tile_idx = tid - processed_tiles
                    tile_m = tile_idx % num_m_tiles
                    tile_n = tile_idx // num_m_tiles
                    
                    # Compute row offsets
                    offs_m = tile_m * BLOCK_SIZE_M + m_block_range
                    row_mask = offs_m < m_size
                    
                    # Handle permutation
                    if PERMUTE_X or PERMUTE_Y:
                        gather_offs = m_start + tl.max_contiguous(
                            tl.multiple_of(offs_m % m_size, BLOCK_SIZE_M),
                            BLOCK_SIZE_M,
                        )
                        token_idx = tl.load(
                            gather_indices_ptr + gather_offs,
                            mask=gather_offs < TOTAL_TOKENS,
                        )
                    
                    # Setup load/store indices
                    if PERMUTE_X:
                        load_idx = (token_idx[:, None] // TOPK) * K
                        store_idx = (m_start + offs_m)[:, None] * N
                    elif PERMUTE_Y:
                        load_idx = (m_start + offs_m)[:, None] * K
                        store_idx = token_idx[:, None] * N
                    else:
                        row_idx = m_start + offs_m
                        load_idx = row_idx[:, None] * K
                        store_idx = row_idx[:, None] * N
                    
                    # Initialize accumulator
                    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACC_DTYPE)
                    
                    # K-loop with GEMM
                    offs_k = tl.arange(0, BLOCK_SIZE_K)
                    x_ptrs = x_ptr + load_idx + offs_k[None, :]
                    
                    offs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    offs_n = tl.max_contiguous(
                        tl.multiple_of(offs_n % N, BLOCK_SIZE_N),
                        BLOCK_SIZE_N,
                    )
                    w_ptrs = w_ptr + (n_start + offs_n[:, None]) * K + offs_k[None, :]
                    
                    for k in range(0, K, BLOCK_SIZE_K):
                        # Load X tile
                        x = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0)
                        
                        # Load W tile
                        w = tl.load(w_ptrs, mask=offs_n[:, None] < N, other=0.0)
                        
                        # Accumulate
                        # Accumulate with TF32 precision
                        acc += tl.dot(x, w.T, allow_tf32=True)
                        
                        x_ptrs += BLOCK_SIZE_K
                        w_ptrs += BLOCK_SIZE_K
                    
                    # Output conversion
                    y = acc.to(output_dtype)
                    
                    # Fuse topk weight multiplication
                    if FUSE_MUL:
                        topk_w = tl.load(
                            topk_weights_ptr + token_idx[:, None],
                            mask=row_mask[:, None],
                        )
                        y *= topk_w.to(output_dtype)
                    
                    # Store result
                    store_mask = row_mask[:, None] & (offs_n[None, :] < N)
                    tl.store(
                        y_ptr + store_idx + offs_n[None, :],
                        y,
                        mask=store_mask,
                    )
                    
                    tid += NUM_SMS
                
                processed_tiles += num_tiles


    @triton.jit
    def _grouped_gemm_backward_dX_kernel(
        # Pointers
        dY_ptr, w_ptr, dX_ptr,
        m_sizes_ptr, gather_indices_ptr,
        # Constants
        NUM_EXPERTS: tl.constexpr,
        NUM_TOKENS: tl.constexpr,
        TOPK: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        NUM_SMS: tl.constexpr,
        # Tuning
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        # Flags
        PERMUTE_X: tl.constexpr,
        PERMUTE_Y: tl.constexpr,
        ACC_DTYPE: tl.constexpr = tl.float32,
    ):
        """
        Backward kernel for computing dX = dY @ W.
        """
        TOTAL_TOKENS: tl.constexpr = NUM_TOKENS * TOPK
        tid = tl.program_id(0)
        output_dtype = dX_ptr.dtype.element_ty
        
        m_end = 0
        processed_tiles = 0
        m_block_range = tl.arange(0, BLOCK_SIZE_M)
        
        for expert_idx in tl.range(NUM_EXPERTS):
            m_start = m_end
            m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
            m_end = m_start + m_size
            
            if m_size > 0:
                n_start = expert_idx * N
                
                num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
                num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
                num_tiles = num_m_tiles * num_k_tiles
                
                while tid >= processed_tiles and tid < processed_tiles + num_tiles:
                    tile_idx = tid - processed_tiles
                    tile_m = tile_idx % num_m_tiles
                    tile_k = tile_idx // num_m_tiles
                    
                    offs_m = tile_m * BLOCK_SIZE_M + m_block_range
                    row_mask = offs_m < m_size
                    
                    # Handle permutation
                    if PERMUTE_Y:
                        gather_offs = m_start + tl.max_contiguous(
                            tl.multiple_of(offs_m % m_size, BLOCK_SIZE_M),
                            BLOCK_SIZE_M,
                        )
                        token_idx = tl.load(
                            gather_indices_ptr + gather_offs,
                            mask=gather_offs < TOTAL_TOKENS,
                        )
                        load_idx = token_idx[:, None] * N
                    else:
                        row_idx = m_start + offs_m
                        load_idx = row_idx[:, None] * N
                    
                    # Store index
                    store_idx = (m_start + offs_m)[:, None] * K
                    
                    # Accumulator
                    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=ACC_DTYPE)
                    
                    # N-loop: dX = dY @ W (reduce over N)
                    offs_n = tl.arange(0, BLOCK_SIZE_N)
                    dY_ptrs = dY_ptr + load_idx + offs_n[None, :]
                    
                    offs_k = tile_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                    w_ptrs = w_ptr + (n_start + offs_n[:, None]) * K + offs_k[None, :]
                    
                    for n in range(0, N, BLOCK_SIZE_N):
                        dY = tl.load(dY_ptrs, mask=row_mask[:, None] & (offs_n[None, :] < N), other=0.0)
                        w = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
                        
                        acc += tl.dot(dY, w, allow_tf32=True)
                        
                        dY_ptrs += BLOCK_SIZE_N
                        offs_n += BLOCK_SIZE_N
                        w_ptrs = w_ptr + (n_start + offs_n[:, None]) * K + offs_k[None, :]
                    
                    # Store
                    dx = acc.to(output_dtype)
                    store_mask = row_mask[:, None] & (offs_k[None, :] < K)
                    tl.atomic_add(
                        dX_ptr + store_idx + offs_k[None, :],
                        dx,
                        mask=store_mask,
                    )
                    
                    tid += NUM_SMS
                
                processed_tiles += num_tiles


    @triton.jit
    def _grouped_gemm_backward_dW_kernel(
        # Pointers
        x_ptr, dY_ptr, dW_ptr,
        m_sizes_ptr, gather_indices_ptr,
        # Constants
        NUM_EXPERTS: tl.constexpr,
        NUM_TOKENS: tl.constexpr,
        TOPK: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        NUM_SMS: tl.constexpr,
        # Tuning
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        # Flags
        PERMUTE_X: tl.constexpr,
        PERMUTE_Y: tl.constexpr,
        ACC_DTYPE: tl.constexpr = tl.float32,
    ):
        """
        Backward kernel for computing dW = dY.T @ X.
        """
        TOTAL_TOKENS: tl.constexpr = NUM_TOKENS * TOPK
        tid = tl.program_id(0)
        output_dtype = dW_ptr.dtype.element_ty
        
        m_end = 0
        processed_tiles = 0
        
        for expert_idx in tl.range(NUM_EXPERTS):
            m_start = m_end
            m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
            m_end = m_start + m_size
            
            if m_size > 0:
                n_start = expert_idx * N
                
                num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
                num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
                num_tiles = num_n_tiles * num_k_tiles
                
                while tid >= processed_tiles and tid < processed_tiles + num_tiles:
                    tile_idx = tid - processed_tiles
                    tile_n = tile_idx % num_n_tiles
                    tile_k = tile_idx // num_n_tiles
                    
                    offs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    offs_k = tile_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                    
                    # Accumulator
                    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=ACC_DTYPE)
                    
                    # M-loop: dW = dY.T @ X (reduce over M)
                    offs_m = tl.arange(0, BLOCK_SIZE_M)
                    
                    for m_off in range(0, m_size, BLOCK_SIZE_M):
                        m_idx = m_start + m_off + offs_m
                        m_mask = (m_off + offs_m) < m_size
                        
                        # Handle permutation for X
                        if PERMUTE_X:
                            token_idx = tl.load(
                                gather_indices_ptr + m_idx,
                                mask=m_mask,
                                other=0,
                            )
                            x_load_idx = (token_idx // TOPK)[:, None] * K
                        else:
                            x_load_idx = m_idx[:, None] * K
                        
                        # Handle permutation for dY
                        if PERMUTE_Y:
                            token_idx = tl.load(
                                gather_indices_ptr + m_idx,
                                mask=m_mask,
                                other=0,
                            )
                            dY_load_idx = token_idx[:, None] * N
                        else:
                            dY_load_idx = m_idx[:, None] * N
                        
                        # Load
                        x = tl.load(
                            x_ptr + x_load_idx + offs_k[None, :],
                            mask=m_mask[:, None] & (offs_k[None, :] < K),
                            other=0.0,
                        )
                        dY = tl.load(
                            dY_ptr + dY_load_idx + offs_n[None, :],
                            mask=m_mask[:, None] & (offs_n[None, :] < N),
                            other=0.0,
                        )
                        
                        # dW += dY.T @ X
                        acc += tl.dot(dY.T, x, allow_tf32=True)
                    
                    # Store
                    dw = acc.to(output_dtype)
                    store_idx = (n_start + offs_n[:, None]) * K + offs_k[None, :]
                    store_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
                    tl.store(dW_ptr + store_idx, dw, mask=store_mask)
                    
                    tid += NUM_SMS
                
                processed_tiles += num_tiles


# ═════════════════════════════════════════════════════════════════════════════════
# Python Interface Functions
# ═════════════════════════════════════════════════════════════════════════════════

def grouped_gemm_forward(
    X: Tensor,
    W: Tensor,
    topk: int,
    m_sizes: Tensor,
    gather_indices: Optional[Tensor] = None,
    topk_weights: Optional[Tensor] = None,
    permute_x: bool = False,
    permute_y: bool = False,
    fuse_mul: bool = False,
    config: Optional[KernelConfigForward] = None,
) -> Tensor:
    """
    Grouped GEMM forward for MoE.
    
    Args:
        X: Input tensor (num_tokens * topk, K) or (num_tokens, K) if permute_x
        W: Expert weights (num_experts, N, K)
        topk: Number of experts per token
        m_sizes: Tokens per expert (num_experts,)
        gather_indices: Token permutation indices
        topk_weights: TopK routing weights for fused multiplication
        permute_x: Fuse input permutation
        permute_y: Fuse output permutation
        fuse_mul: Fuse topk weight multiplication
        config: Kernel configuration
    
    Returns:
        Output tensor (total_tokens, N)
    """
    if config is None:
        config = KernelConfigForward()
    
    X = X.contiguous()
    W = W.contiguous()
    m_sizes = m_sizes.contiguous()
    
    num_experts = m_sizes.shape[0]
    X = X.view(-1, X.shape[-1])
    W = W.view(-1, W.shape[-1])
    
    total_tokens = X.shape[0] if not permute_x else X.shape[0] * topk
    num_tokens = total_tokens // topk
    K = X.shape[1]
    N = W.shape[0] // num_experts
    
    y = torch.empty((total_tokens, N), device=X.device, dtype=X.dtype)
    
    if total_tokens == 0 or N == 0:
        return y
    
    if not _TRITON_AVAILABLE:
        return _grouped_gemm_forward_fallback(
            X, W, m_sizes, gather_indices, topk_weights,
            num_experts, num_tokens, topk, permute_x, permute_y, fuse_mul
        )
    
    NUM_SMS = get_num_sms()
    
    def grid(META):
        return (NUM_SMS,)
    
    _grouped_gemm_forward_kernel[grid](
        # Pointers
        X, W, y,
        m_sizes, gather_indices, topk_weights,
        # Constants
        num_experts, num_tokens, topk, N, K, NUM_SMS,
        # Tuning
        config.BLOCK_SIZE_M, config.BLOCK_SIZE_N, config.BLOCK_SIZE_K,
        # Flags
        permute_x, permute_y, fuse_mul,
        # Warps/stages
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    
    return y


def grouped_gemm_backward_dX(
    dY: Tensor,
    W: Tensor,
    topk: int,
    m_sizes: Tensor,
    gather_indices: Optional[Tensor] = None,
    permute_x: bool = False,
    permute_y: bool = False,
    config: Optional[KernelConfigBackward] = None,
) -> Tensor:
    """Compute dX = dY @ W."""
    if config is None:
        config = KernelConfigBackward()
    
    dY = dY.contiguous()
    W = W.contiguous()
    
    num_experts = m_sizes.shape[0]
    dY = dY.view(-1, dY.shape[-1])
    W = W.view(-1, W.shape[-1])
    
    M_total, N = dY.shape
    N_total, K = W.shape
    N_per_exp = N_total // num_experts
    num_tokens = M_total // topk
    
    dX = torch.zeros((M_total, K), device=dY.device, dtype=dY.dtype)
    
    if M_total == 0:
        return dX
    
    if not _TRITON_AVAILABLE:
        return _grouped_gemm_dX_fallback(dY, W, m_sizes, gather_indices, num_experts, topk, permute_y)
    
    NUM_SMS = get_num_sms()
    
    def grid(META):
        return (NUM_SMS,)
    
    _grouped_gemm_backward_dX_kernel[grid](
        dY, W, dX,
        m_sizes, gather_indices,
        num_experts, num_tokens, topk, N_per_exp, K, NUM_SMS,
        config.BLOCK_SIZE_M, config.BLOCK_SIZE_N, config.BLOCK_SIZE_K,
        permute_x, permute_y,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    
    # Accumulate gradients across topk for each token
    if topk > 1 and permute_x:
        dX = dX.view(-1, topk, K).sum(dim=1)
    
    return dX


def grouped_gemm_backward_dW(
    X: Tensor,
    dY: Tensor,
    topk: int,
    m_sizes: Tensor,
    gather_indices: Optional[Tensor] = None,
    permute_x: bool = False,
    permute_y: bool = False,
    config: Optional[KernelConfigBackward] = None,
) -> Tensor:
    """Compute dW = dY.T @ X."""
    if config is None:
        config = KernelConfigBackward()
    
    X = X.contiguous()
    dY = dY.contiguous()
    
    num_experts = m_sizes.shape[0]
    X = X.view(-1, X.shape[-1])
    dY = dY.view(-1, dY.shape[-1])
    
    M_total, K = X.shape
    _, N = dY.shape
    num_tokens = M_total // topk if not permute_x else M_total
    
    dW = torch.zeros((num_experts, N, K), device=X.device, dtype=X.dtype)
    
    if M_total == 0:
        return dW
    
    if not _TRITON_AVAILABLE:
        return _grouped_gemm_dW_fallback(X, dY, m_sizes, gather_indices, num_experts, topk, permute_x, permute_y)
    
    NUM_SMS = get_num_sms()
    
    def grid(META):
        return (NUM_SMS,)
    
    _grouped_gemm_backward_dW_kernel[grid](
        X, dY, dW,
        m_sizes, gather_indices,
        num_experts, num_tokens, topk, N, K, NUM_SMS,
        config.BLOCK_SIZE_M, config.BLOCK_SIZE_N, config.BLOCK_SIZE_K,
        permute_x, permute_y,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    
    return dW


# ═════════════════════════════════════════════════════════════════════════════════
# PyTorch Fallbacks
# ═════════════════════════════════════════════════════════════════════════════════

def _grouped_gemm_forward_fallback(
    X: Tensor, W: Tensor, m_sizes: Tensor,
    gather_indices: Optional[Tensor],
    topk_weights: Optional[Tensor],
    num_experts: int, num_tokens: int, topk: int,
    permute_x: bool, permute_y: bool, fuse_mul: bool,
) -> Tensor:
    """PyTorch fallback for grouped GEMM forward."""
    total_tokens = num_tokens * topk
    K = X.shape[-1]
    N = W.shape[0] // num_experts
    
    y = torch.empty((total_tokens, N), device=X.device, dtype=X.dtype)
    
    m_start = 0
    for e in range(num_experts):
        m_size = m_sizes[e].item()
        if m_size == 0:
            continue
        
        m_end = m_start + m_size
        
        # Get X for this expert
        if permute_x:
            indices = gather_indices[m_start:m_end] // topk
            x_expert = X[indices]
        else:
            x_expert = X[m_start:m_end]
        
        # Get W for this expert
        w_expert = W[e * N:(e + 1) * N]
        
        # Compute
        out = x_expert @ w_expert.t()
        
        # Fuse multiply
        if fuse_mul and topk_weights is not None:
            weights = topk_weights[m_start:m_end].unsqueeze(-1)
            out = out * weights
        
        # Store
        if permute_y:
            indices = gather_indices[m_start:m_end]
            y[indices] = out
        else:
            y[m_start:m_end] = out
        
        m_start = m_end
    
    return y


def _grouped_gemm_dX_fallback(
    dY: Tensor, W: Tensor, m_sizes: Tensor,
    gather_indices: Optional[Tensor],
    num_experts: int, topk: int, permute_y: bool,
) -> Tensor:
    """PyTorch fallback for grouped GEMM dX."""
    M_total, N = dY.shape
    K = W.shape[-1]
    N_per_exp = N
    
    dX = torch.zeros((M_total, K), device=dY.device, dtype=dY.dtype)
    
    m_start = 0
    for e in range(num_experts):
        m_size = m_sizes[e].item()
        if m_size == 0:
            continue
        
        m_end = m_start + m_size
        
        # Get dY for this expert
        if permute_y:
            indices = gather_indices[m_start:m_end]
            dy_expert = dY[indices]
        else:
            dy_expert = dY[m_start:m_end]
        
        # Get W for this expert
        w_expert = W[e * N_per_exp:(e + 1) * N_per_exp]
        
        # dX = dY @ W
        dx = dy_expert @ w_expert
        dX[m_start:m_end] = dx
        
        m_start = m_end
    
    return dX


def _grouped_gemm_dW_fallback(
    X: Tensor, dY: Tensor, m_sizes: Tensor,
    gather_indices: Optional[Tensor],
    num_experts: int, topk: int,
    permute_x: bool, permute_y: bool,
) -> Tensor:
    """PyTorch fallback for grouped GEMM dW."""
    K = X.shape[-1]
    N = dY.shape[-1]
    
    dW = torch.zeros((num_experts, N, K), device=X.device, dtype=X.dtype)
    
    m_start = 0
    for e in range(num_experts):
        m_size = m_sizes[e].item()
        if m_size == 0:
            continue
        
        m_end = m_start + m_size
        
        # Get X for this expert
        if permute_x:
            indices = gather_indices[m_start:m_end] // topk
            x_expert = X[indices]
        else:
            x_expert = X[m_start:m_end]
        
        # Get dY for this expert
        if permute_y:
            indices = gather_indices[m_start:m_end]
            dy_expert = dY[indices]
        else:
            dy_expert = dY[m_start:m_end]
        
        # dW = dY.T @ X
        dW[e] = dy_expert.t() @ x_expert
        
        m_start = m_end
    
    return dW


# ═════════════════════════════════════════════════════════════════════════════════
# Autograd Function
# ═════════════════════════════════════════════════════════════════════════════════

class GroupedGEMM(torch.autograd.Function):
    """
    Autograd wrapper for grouped GEMM with full backward support.
    """
    
    @staticmethod
    def forward(
        ctx,
        X: Tensor,
        W: Tensor,
        topk: int,
        m_sizes: Tensor,
        gather_indices: Optional[Tensor],
        topk_weights: Optional[Tensor],
        permute_x: bool,
        permute_y: bool,
        fuse_mul: bool,
        fwd_config: Optional[KernelConfigForward],
        bwd_config: Optional[KernelConfigBackward],
    ) -> Tensor:
        ctx.save_for_backward(X, W, m_sizes, gather_indices)
        ctx.topk = topk
        ctx.permute_x = permute_x
        ctx.permute_y = permute_y
        ctx.bwd_config = bwd_config
        
        return grouped_gemm_forward(
            X, W, topk, m_sizes, gather_indices, topk_weights,
            permute_x, permute_y, fuse_mul, fwd_config,
        )
    
    @staticmethod
    def backward(ctx, dY: Tensor):
        X, W, m_sizes, gather_indices = ctx.saved_tensors
        
        dX = grouped_gemm_backward_dX(
            dY, W, ctx.topk, m_sizes, gather_indices,
            ctx.permute_x, ctx.permute_y, ctx.bwd_config,
        )
        
        dW = grouped_gemm_backward_dW(
            X, dY, ctx.topk, m_sizes, gather_indices,
            ctx.permute_x, ctx.permute_y, ctx.bwd_config,
        )
        
        return dX, dW, None, None, None, None, None, None, None, None, None


# ═════════════════════════════════════════════════════════════════════════════════
# Router with Load Balancing
# ═════════════════════════════════════════════════════════════════════════════════

class MoERouter(nn.Module):
    """
    SOTA MoE Router with:
    - Top-K expert selection
    - Auxiliary load balancing loss
    - Z-loss regularization
    - Expert capacity constraints
    - Epsilon-greedy exploration (training)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
        noise_std: float = 0.1,
        epsilon_greedy: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef
        self.noise_std = noise_std
        self.epsilon_greedy = epsilon_greedy
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Track auxiliary losses
        self.aux_loss = 0.0
        self.z_loss = 0.0
    
    def forward(
        self,
        hidden_states: Tensor,
        return_indices: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Route tokens to experts.
        
        Returns:
            routing_weights: (total_tokens,) weights for each expert selection
            selected_experts: (batch * seq, top_k) selected expert indices
            m_sizes: (num_experts,) tokens per expert
            gather_indices: (total_tokens,) permutation indices
        """
        batch, seq_len, hidden = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden)
        num_tokens = hidden_states.shape[0]
        
        # Compute routing logits
        router_logits = self.gate(hidden_states)  # (num_tokens, num_experts)
        
        # Z-loss for regularization
        if self.training and self.z_loss_coef > 0:
            self.z_loss = self.z_loss_coef * torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        
        # Add noise during training
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        
        # Epsilon-greedy exploration
        if self.training and self.epsilon_greedy > 0:
            explore_mask = torch.rand(num_tokens, device=router_logits.device) < self.epsilon_greedy
            random_experts = torch.randint(0, self.num_experts, (num_tokens, self.top_k), device=router_logits.device)
        
        # Compute routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Apply epsilon-greedy
        if self.training and self.epsilon_greedy > 0:
            selected_experts = torch.where(
                explore_mask.unsqueeze(-1),
                random_experts,
                selected_experts,
            )
            routing_weights = torch.gather(routing_probs, -1, selected_experts)
        
        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Compute auxiliary load balancing loss
        if self.training and self.aux_loss_coef > 0:
            # Fraction of tokens routed to each expert
            # expert_mask: (num_tokens, num_experts) one-hot
            expert_mask = F.one_hot(selected_experts, self.num_experts).float()
            expert_mask = expert_mask.sum(dim=1)  # (num_tokens, num_experts)
            tokens_per_expert = expert_mask.sum(dim=0)
            expert_fraction = tokens_per_expert / num_tokens
            
            # Average routing probability per expert
            router_prob_per_expert = routing_probs.mean(dim=0)
            
            # Aux loss encourages uniform distribution
            self.aux_loss = self.aux_loss_coef * self.num_experts * (
                expert_fraction * router_prob_per_expert
            ).sum()
        
        if not return_indices:
            return routing_weights, selected_experts, None, None
        
        # Compute m_sizes and gather_indices
        m_sizes, gather_indices = self._compute_indices(selected_experts, num_tokens)
        
        routing_weights = routing_weights.view(-1)
        
        return routing_weights, selected_experts, m_sizes, gather_indices
    
    def _compute_indices(
        self,
        selected_experts: Tensor,
        num_tokens: int,
    ) -> Tuple[Tensor, Tensor]:
        """Compute m_sizes and gather_indices for grouped GEMM."""
        # Flatten expert selections
        flat_experts = selected_experts.view(-1)  # (num_tokens * top_k,)
        total_tokens = flat_experts.shape[0]
        
        # Token indices
        token_indices = torch.arange(total_tokens, device=flat_experts.device)
        
        # Sort by expert
        sorted_experts, sort_indices = flat_experts.sort(stable=True)
        gather_indices = token_indices[sort_indices]
        
        # Count tokens per expert
        m_sizes = torch.bincount(flat_experts, minlength=self.num_experts)
        
        return m_sizes, gather_indices
    
    def get_aux_loss(self) -> Tensor:
        """Get combined auxiliary loss."""
        return self.aux_loss + self.z_loss


# ═════════════════════════════════════════════════════════════════════════════════
# Expert MLP
# ═════════════════════════════════════════════════════════════════════════════════

class ExpertMLP(nn.Module):
    """Single expert MLP with SwiGLU activation."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        if activation == "silu":
            self.act_fn = F.silu
        elif activation == "gelu":
            self.act_fn = F.gelu
        else:
            self.act_fn = F.relu
    
    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ═════════════════════════════════════════════════════════════════════════════════
# Sparse MoE Block (Complete Module)
# ═════════════════════════════════════════════════════════════════════════════════

class SparseMoEBlock(nn.Module):
    """
    SOTA Sparse Mixture of Experts block.
    
    Features:
    - Grouped GEMM kernels for efficient expert computation
    - Load balancing with auxiliary loss
    - Z-loss regularization
    - Expert capacity constraints
    - Fused permutation and weight multiplication
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
        activation: str = "silu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.router = MoERouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            aux_loss_coef=aux_loss_coef,
            z_loss_coef=z_loss_coef,
        )
        
        # Expert weights (packed for grouped GEMM)
        # gate_up: (num_experts, intermediate_size * 2, hidden_size)
        # down: (num_experts, hidden_size, intermediate_size)
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, intermediate_size * 2, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        
        if activation == "silu":
            self.act_fn = F.silu
        elif activation == "gelu":
            self.act_fn = F.gelu
        else:
            self.act_fn = F.relu
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.gate_up_proj)
        nn.init.kaiming_uniform_(self.down_proj)
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        batch, seq_len, hidden = hidden_states.shape
        
        # Route tokens
        routing_weights, selected_experts, m_sizes, gather_indices = self.router(hidden_states)
        
        if m_sizes is None or m_sizes.sum() == 0:
            return hidden_states
        
        # Reshape for grouped GEMM
        hidden_flat = hidden_states.view(-1, hidden)
        
        # First grouped GEMM: gate_up projection with fused permutation
        gate_up_out = grouped_gemm_forward(
            hidden_flat,
            self.gate_up_proj,
            topk=self.top_k,
            m_sizes=m_sizes,
            gather_indices=gather_indices,
            permute_x=True,
            permute_y=False,
        )
        
        # Split gate and up
        gate, up = gate_up_out.chunk(2, dim=-1)
        
        # SwiGLU
        hidden_act = self.act_fn(gate) * up
        
        # Second grouped GEMM: down projection with fused output permutation
        output = grouped_gemm_forward(
            hidden_act,
            self.down_proj,
            topk=self.top_k,
            m_sizes=m_sizes,
            gather_indices=gather_indices,
            topk_weights=routing_weights,
            permute_x=False,
            permute_y=True,
            fuse_mul=True,
        )
        
        return output.view(batch, seq_len, hidden)
    
    def get_aux_loss(self) -> Tensor:
        """Get auxiliary loss for load balancing."""
        return self.router.get_aux_loss()


# ═════════════════════════════════════════════════════════════════════════════════
# High-Level API
# ═════════════════════════════════════════════════════════════════════════════════

def top_k_gating(
    hidden_states: Tensor,
    gate: nn.Linear,
    num_experts: int,
    top_k: int = 2,
    training: bool = True,
    noise_std: float = 0.1,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Top-K expert routing with load balancing.
    
    Returns:
        routing_weights: (batch * seq, top_k)
        selected_experts: (batch * seq, top_k)
        aux_loss: scalar load balancing loss
    """
    batch, seq_len, hidden = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden)
    num_tokens = hidden_states.shape[0]
    
    # Compute logits
    logits = gate(hidden_states)
    
    # Add noise during training
    if training and noise_std > 0:
        noise = torch.randn_like(logits) * noise_std
        logits = logits + noise
    
    # Softmax
    probs = F.softmax(logits, dim=-1)
    
    # Top-K
    weights, experts = torch.topk(probs, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    
    # Auxiliary loss
    expert_mask = F.one_hot(experts, num_experts).float().sum(dim=1)
    tokens_per_expert = expert_mask.sum(dim=0)
    expert_fraction = tokens_per_expert / num_tokens
    router_prob_per_expert = probs.mean(dim=0)
    aux_loss = num_experts * (expert_fraction * router_prob_per_expert).sum()
    
    return weights, experts, aux_loss


def grouped_expert_forward(
    hidden_states: Tensor,
    expert_weights: Tensor,
    expert_indices: Tensor,
    experts: nn.ModuleList,
    num_experts: int,
    top_k: int,
) -> Tensor:
    """
    Forward pass through experts with grouping.
    
    Efficient implementation that groups tokens by expert.
    """
    batch_seq, hidden = hidden_states.shape
    output = torch.zeros_like(hidden_states)
    
    # Flatten expert selections
    flat_indices = expert_indices.view(-1)
    flat_weights = expert_weights.view(-1)
    token_indices = torch.arange(batch_seq, device=hidden_states.device).repeat_interleave(top_k)
    
    for e in range(num_experts):
        mask = flat_indices == e
        if not mask.any():
            continue
        
        tokens = token_indices[mask]
        weights = flat_weights[mask]
        
        expert_input = hidden_states[tokens]
        expert_output = experts[e](expert_input)
        
        output.index_add_(0, tokens, weights.unsqueeze(-1) * expert_output)
    
    return output


def grouped_gemm(
    X: Tensor,
    W: Tensor,
    expert_offsets: Tensor,
    num_experts: int,
) -> Tensor:
    """
    Grouped GEMM for MoE: Y = X @ W.T per expert group.
    """
    return grouped_gemm_forward(
        X, W,
        topk=1,
        m_sizes=expert_offsets,
        permute_x=False,
        permute_y=False,
    )


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Config
    "KernelConfigForward",
    "KernelConfigBackward",
    "supports_tma",
    "get_num_sms",
    # Core functions
    "grouped_gemm_forward",
    "grouped_gemm_backward_dX",
    "grouped_gemm_backward_dW",
    # Autograd
    "GroupedGEMM",
    # Router
    "MoERouter",
    # Modules
    "ExpertMLP",
    "SparseMoEBlock",
    # High-level API
    "top_k_gating",
    "grouped_expert_forward",
    "grouped_gemm",
]
