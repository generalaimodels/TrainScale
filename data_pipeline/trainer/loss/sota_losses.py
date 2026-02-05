# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SOTA Loss Functions Suite - Production-Grade Implementation
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Architecture: Zero-copy memory layout | Flash-style online softmax | Triton JIT compilation | Hardware-aware autotuning
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Loss Categories:
#   ├── Cross Entropy Variants    : Chunked, Focal, Poly, Label-Smoothed
#   ├── Knowledge Distillation    : Soft, Hard, Feature, Attention, Progressive
#   ├── Preference Optimization   : DPO, IPO, KTO, ORPO, SimPO, CPO, RSO
#   ├── Contrastive Learning      : InfoNCE, CLIP, Triplet, NT-Xent, SupCon
#   ├── MoE Regularization        : Z-Loss, Aux-Load-Balance, Expert-Entropy
#   ├── Reconstruction            : MSE, Huber, Charbonnier, Perceptual
#   ├── Ranking                   : Margin, Pairwise, ListMLE, LambdaRank
#   └── Specialized               : Focal, Dice, Tversky, GHM-C
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# §1: Constants & Type Definitions
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

T = TypeVar("T", bound=nn.Module)

# Cache line alignment for atomics
CACHE_LINE_BYTES: Final[int] = 64
# Warp size for CUDA kernels
WARP_SIZE: Final[int] = 32
# Maximum vocabulary chunk for memory efficiency
MAX_VOCAB_CHUNK: Final[int] = 65536
# Numerical stability epsilon
EPS: Final[float] = 1e-8
# Minimum safe log value
LOG_MIN: Final[float] = -100.0

# Reduction modes
ReductionMode = Literal["mean", "sum", "none", "batchmean"]

# Loss types for preference optimization
PreferenceLossType = Literal["sigmoid", "hinge", "ipo", "kto", "robust"]


class LossOutput(NamedTuple):
    """Structured loss output with metrics for logging/debugging."""
    loss: Tensor
    metrics: Dict[str, float]
    aux_loss: Optional[Tensor] = None


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Immutable configuration for loss functions with validation."""
    ignore_index: int = -100
    label_smoothing: float = 0.0
    reduction: ReductionMode = "mean"
    
    def __post_init__(self) -> None:
        # Validation using object.__setattr__ for frozen dataclass
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        if self.reduction not in ("mean", "sum", "none", "batchmean"):
            raise ValueError(f"Invalid reduction: {self.reduction}")


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# §2: Triton Backend - High-Performance Kernels
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

_TRITON_AVAILABLE: bool = False
_CUDA_AVAILABLE: bool = torch.cuda.is_available()

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = _CUDA_AVAILABLE and triton is not None
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


if _TRITON_AVAILABLE:
    
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Autotuning configurations for optimal hardware utilization
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    
    def _get_autotune_configs() -> List["triton.Config"]:
        """Generate autotuning configurations for cross-entropy kernel."""
        configs = []
        for block_v in [1024, 2048, 4096, 8192]:
            for num_warps in [4, 8, 16]:
                for num_stages in [2, 3, 4]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_V": block_v},
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
        return configs
    
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Online Softmax Cross-Entropy Kernel (Flash-style, numerically stable)
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    
    @triton.autotune(
        configs=_get_autotune_configs(),
        key=["vocab_size"],
    )
    @triton.jit
    def _fused_cross_entropy_fwd_kernel(
        logits_ptr,          # [N, V] input logits
        labels_ptr,          # [N] target labels
        loss_ptr,            # [N] per-sample loss output
        lse_ptr,             # [N] log-sum-exp for backward
        N: tl.constexpr,
        vocab_size: tl.constexpr,
        ignore_index: tl.constexpr,
        label_smoothing: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """
        Fused cross-entropy forward pass with online softmax algorithm.
        
        Memory complexity: O(BLOCK_V) vs O(V) for standard implementation.
        Numerical stability: Uses online algorithm to prevent overflow.
        
        Algorithm (Milakov & Gimelshein, Online normalizer calculation):
            m_i = max(m_{i-1}, max(x_i))
            d_i = d_{i-1} * exp(m_{i-1} - m_i) + sum(exp(x_i - m_i))
            logsumexp = m_n + log(d_n)
        """
        row_idx = tl.program_id(0)
        
        if row_idx >= N:
            return
        
        # Load target label
        label = tl.load(labels_ptr + row_idx)
        
        # Handle ignored indices
        if label == ignore_index:
            tl.store(loss_ptr + row_idx, 0.0)
            tl.store(lse_ptr + row_idx, 0.0)
            return
        
        # Online softmax: compute max and sum(exp) in single pass
        row_offset = row_idx * vocab_size
        
        # Initialize accumulators
        m_prev = float("-inf")  # Running maximum
        d_acc = 0.0             # Running sum of exp(x - m)
        
        # First pass: compute log-sum-exp with online algorithm
        for chunk_start in range(0, vocab_size, BLOCK_V):
            offs = chunk_start + tl.arange(0, BLOCK_V)
            mask = offs < vocab_size
            
            # Coalesced memory access
            logits = tl.load(logits_ptr + row_offset + offs, mask=mask, other=float("-inf"))
            
            # Online update
            m_curr = tl.max(logits)
            m_new = tl.where(m_curr > m_prev, m_curr, m_prev)
            
            # Rescale previous accumulator
            d_acc = d_acc * tl.exp(m_prev - m_new)
            # Add current chunk contribution
            d_acc += tl.sum(tl.exp(logits - m_new) * mask.to(tl.float32))
            
            m_prev = m_new
        
        # Final log-sum-exp
        lse = m_prev + tl.log(d_acc)
        tl.store(lse_ptr + row_idx, lse)
        
        # Load target logit (may be in any chunk)
        target_logit = tl.load(logits_ptr + row_offset + label)
        
        # Cross-entropy: -log(softmax) = -target + lse
        ce_loss = -target_logit + lse
        
        # Label smoothing: (1-α)*CE + α*uniform_loss
        if label_smoothing > 0.0:
            # Uniform loss = lse - mean(logits)
            # Second pass for mean (can be fused with first pass for production)
            logit_sum = 0.0
            for chunk_start in range(0, vocab_size, BLOCK_V):
                offs = chunk_start + tl.arange(0, BLOCK_V)
                mask = offs < vocab_size
                logits = tl.load(logits_ptr + row_offset + offs, mask=mask, other=0.0)
                logit_sum += tl.sum(logits * mask.to(tl.float32))
            
            smooth_loss = lse - logit_sum / vocab_size
            final_loss = (1.0 - label_smoothing) * ce_loss + label_smoothing * smooth_loss
        else:
            final_loss = ce_loss
        
        tl.store(loss_ptr + row_idx, final_loss)
    
    @triton.jit
    def _fused_cross_entropy_bwd_kernel(
        logits_ptr,          # [N, V] input logits
        labels_ptr,          # [N] target labels
        lse_ptr,             # [N] log-sum-exp from forward
        grad_output_ptr,     # [N] upstream gradient
        grad_logits_ptr,     # [N, V] output gradient
        N: tl.constexpr,
        vocab_size: tl.constexpr,
        ignore_index: tl.constexpr,
        label_smoothing: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """
        Fused cross-entropy backward pass.
        
        Gradient: d_loss/d_logits = softmax(logits) - one_hot(target)
        With label smoothing: (1-α)*grad + α/V
        """
        row_idx = tl.program_id(0)
        chunk_idx = tl.program_id(1)
        
        if row_idx >= N:
            return
        
        label = tl.load(labels_ptr + row_idx)
        
        # Handle ignored indices
        if label == ignore_index:
            offs = chunk_idx * BLOCK_V + tl.arange(0, BLOCK_V)
            mask = offs < vocab_size
            tl.store(grad_logits_ptr + row_idx * vocab_size + offs, 
                     tl.zeros([BLOCK_V], dtype=tl.float32), mask=mask)
            return
        
        lse = tl.load(lse_ptr + row_idx)
        grad_out = tl.load(grad_output_ptr + row_idx)
        
        # Process chunk
        offs = chunk_idx * BLOCK_V + tl.arange(0, BLOCK_V)
        mask = offs < vocab_size
        
        logits = tl.load(logits_ptr + row_idx * vocab_size + offs, mask=mask, other=0.0)
        
        # Softmax gradient: exp(logit - lse) = softmax
        probs = tl.exp(logits - lse)
        
        # Subtract 1 from target position
        is_target = (offs == label).to(tl.float32)
        grad = probs - is_target
        
        # Label smoothing adjustment
        if label_smoothing > 0.0:
            # Gradient becomes: (1-α)*(softmax - one_hot) + α*(softmax - 1/V)
            grad = (1.0 - label_smoothing) * grad + label_smoothing * (probs - 1.0 / vocab_size)
        
        # Scale by upstream gradient
        grad = grad * grad_out
        
        tl.store(grad_logits_ptr + row_idx * vocab_size + offs, grad, mask=mask)
    
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Contrastive Loss Kernel (InfoNCE with hard negative mining)
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8),
        ],
        key=["M", "N"],
    )
    @triton.jit
    def _contrastive_similarity_kernel(
        a_ptr,               # [M, D] first embeddings
        b_ptr,               # [N, D] second embeddings
        sim_ptr,             # [M, N] similarity output
        M: tl.constexpr,
        N: tl.constexpr,
        D: tl.constexpr,
        temperature: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        Compute scaled dot-product similarity matrix for contrastive loss.
        
        Output: sim[i, j] = (a[i] · b[j]) / temperature
        Optimized for L2-normalized inputs.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        mask_m = offs_m < M
        mask_n = offs_n < N
        
        # Accumulator for dot product
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        
        # Compute dot product in chunks over embedding dimension
        BLOCK_D: tl.constexpr = 64
        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D
            
            # Load embeddings
            a = tl.load(
                a_ptr + offs_m[:, None] * D + d_offs[None, :],
                mask=mask_m[:, None] & d_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptr + offs_n[:, None] * D + d_offs[None, :],
                mask=mask_n[:, None] & d_mask[None, :],
                other=0.0,
            )
            
            # Accumulate dot product
            acc += tl.dot(a, tl.trans(b))
        
        # Scale by temperature
        acc = acc / temperature
        
        # Store result
        tl.store(
            sim_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=mask_m[:, None] & mask_n[None, :],
        )
    
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Focal Loss Kernel (Class imbalance handling)
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    
    @triton.jit
    def _focal_loss_kernel(
        logits_ptr,          # [N, C] input logits
        labels_ptr,          # [N] target labels
        loss_ptr,            # [N] per-sample loss output
        N: tl.constexpr,
        num_classes: tl.constexpr,
        alpha: tl.constexpr,
        gamma: tl.constexpr,
        ignore_index: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """
        Focal loss for addressing class imbalance.
        
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        
        When γ=0, equivalent to weighted cross-entropy.
        Typical γ values: 0.5, 1.0, 2.0, 5.0
        """
        row_idx = tl.program_id(0)
        
        if row_idx >= N:
            return
        
        label = tl.load(labels_ptr + row_idx)
        
        if label == ignore_index:
            tl.store(loss_ptr + row_idx, 0.0)
            return
        
        row_offset = row_idx * num_classes
        
        # Online softmax for numerical stability
        m_prev = float("-inf")
        d_acc = 0.0
        
        for chunk_start in range(0, num_classes, BLOCK_C):
            offs = chunk_start + tl.arange(0, BLOCK_C)
            mask = offs < num_classes
            
            logits = tl.load(logits_ptr + row_offset + offs, mask=mask, other=float("-inf"))
            
            m_curr = tl.max(logits)
            m_new = tl.where(m_curr > m_prev, m_curr, m_prev)
            d_acc = d_acc * tl.exp(m_prev - m_new)
            d_acc += tl.sum(tl.exp(logits - m_new) * mask.to(tl.float32))
            m_prev = m_new
        
        lse = m_prev + tl.log(d_acc)
        
        # Target probability
        target_logit = tl.load(logits_ptr + row_offset + label)
        log_pt = target_logit - lse  # log(softmax[target])
        pt = tl.exp(log_pt)
        
        # Focal weight: (1 - pt)^gamma
        focal_weight = tl.pow(1.0 - pt, gamma)
        
        # Focal loss: -alpha * (1-pt)^gamma * log(pt)
        loss = -alpha * focal_weight * log_pt
        
        tl.store(loss_ptr + row_idx, loss)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# §3: Autograd Functions - Custom Backward Passes
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

class FusedCrossEntropyFunction(torch.autograd.Function):
    """
    Custom autograd function for fused cross-entropy with Triton kernels.
    
    Benefits:
    - Single kernel launch for forward pass
    - Fused softmax + log + nll in one pass
    - Memory-efficient: O(N) instead of O(N*V) for intermediate
    - Numerically stable via online softmax algorithm
    """
    
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        logits: Tensor,
        labels: Tensor,
        ignore_index: int,
        label_smoothing: float,
        reduction: str,
    ) -> Tensor:
        # Input validation
        assert logits.is_cuda and labels.is_cuda, "Inputs must be on CUDA"
        assert logits.is_contiguous() and labels.is_contiguous(), "Inputs must be contiguous"
        
        N, V = logits.shape
        
        # Allocate outputs
        loss = torch.empty(N, device=logits.device, dtype=torch.float32)
        lse = torch.empty(N, device=logits.device, dtype=torch.float32)
        
        # Kernel launch configuration
        grid = (N,)
        
        _fused_cross_entropy_fwd_kernel[grid](
            logits, labels, loss, lse,
            N=N, vocab_size=V,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        
        # Save for backward
        ctx.save_for_backward(logits, labels, lse)
        ctx.ignore_index = ignore_index
        ctx.label_smoothing = label_smoothing
        ctx.reduction = reduction
        ctx.N = N
        ctx.V = V
        
        # Apply reduction
        if reduction == "mean":
            valid_count = (labels != ignore_index).sum()
            return loss.sum() / valid_count.clamp(min=1)
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "batchmean":
            return loss.sum() / N
        else:
            return loss
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        logits, labels, lse = ctx.saved_tensors
        N, V = ctx.N, ctx.V
        
        # Allocate gradient
        grad_logits = torch.empty_like(logits)
        
        # Handle reduction gradient scaling
        if ctx.reduction == "mean":
            valid_count = (labels != ctx.ignore_index).sum()
            grad_output_scaled = grad_output / valid_count.clamp(min=1)
            grad_output_expanded = grad_output_scaled.expand(N)
        elif ctx.reduction == "sum":
            grad_output_expanded = grad_output.expand(N)
        elif ctx.reduction == "batchmean":
            grad_output_expanded = (grad_output / N).expand(N)
        else:
            grad_output_expanded = grad_output
        
        # Determine block size from autotuned config
        BLOCK_V = 4096  # Default, should match autotuned value
        num_chunks = (V + BLOCK_V - 1) // BLOCK_V
        
        grid = (N, num_chunks)
        
        _fused_cross_entropy_bwd_kernel[grid](
            logits, labels, lse, grad_output_expanded, grad_logits,
            N=N, vocab_size=V,
            ignore_index=ctx.ignore_index,
            label_smoothing=ctx.label_smoothing,
            BLOCK_V=BLOCK_V,
        )
        
        return grad_logits, None, None, None, None


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# §4: Core Loss Functions - Cross Entropy Family
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

class FusedCrossEntropyLoss(nn.Module):
    """
    Memory-efficient fused cross-entropy loss using Triton kernels.
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Standard CE: softmax → log → nll (3 kernels, O(N*V) memory)        │
    │  Fused CE: single kernel (1 kernel, O(N) memory)                    │
    │  Memory savings: ~3x for forward, ~2x for backward                  │
    └─────────────────────────────────────────────────────────────────────┘
    
    Falls back to PyTorch implementation when Triton unavailable.
    """
    
    __constants__ = ["ignore_index", "label_smoothing", "reduction"]
    
    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: ReductionMode = "mean",
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        # Fallback for non-CUDA or non-Triton environments
        self._fallback = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )
    
    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Unnormalized predictions [batch, seq, vocab] or [batch, vocab]
            labels: Target indices [batch, seq] or [batch]
        
        Returns:
            Loss tensor according to reduction mode
        """
        # Handle 3D logits (batch, seq, vocab)
        original_shape = logits.shape
        if logits.dim() == 3:
            batch, seq, vocab = logits.shape
            logits = logits.reshape(-1, vocab)
            labels = labels.reshape(-1)
        
        # Use Triton kernel if available and beneficial
        use_triton = (
            _TRITON_AVAILABLE
            and logits.is_cuda
            and logits.is_contiguous()
            and logits.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and logits.shape[1] >= 1024  # Overhead not worth it for small vocab
        )
        
        if use_triton:
            # Cast to float32 for numerical stability in kernel
            logits_fp32 = logits.float() if logits.dtype != torch.float32 else logits
            return FusedCrossEntropyFunction.apply(
                logits_fp32,
                labels,
                self.ignore_index,
                self.label_smoothing,
                self.reduction,
            )
        else:
            return self._fallback(logits, labels)
    
    def extra_repr(self) -> str:
        return (
            f"ignore_index={self.ignore_index}, "
            f"label_smoothing={self.label_smoothing}, "
            f"reduction='{self.reduction}'"
        )


class ChunkedCrossEntropyLoss(nn.Module):
    """
    Memory-efficient cross-entropy with vocabulary chunking.
    
    Algorithm:
    1. Split vocabulary into chunks of size `chunk_size`
    2. Compute log-sum-exp incrementally using online algorithm
    3. Accumulate loss without materializing full softmax
    
    Memory complexity: O(batch * seq * chunk_size) vs O(batch * seq * vocab)
    Recommended for vocab_size > 50k on memory-constrained GPUs.
    """
    
    __constants__ = ["ignore_index", "label_smoothing", "reduction", "chunk_size"]
    
    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: ReductionMode = "mean",
        chunk_size: int = 32768,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.chunk_size = chunk_size
    
    @torch.no_grad()
    def _compute_lse_chunked(self, logits: Tensor) -> Tensor:
        """
        Compute log-sum-exp in chunks using online algorithm.
        
        Maintains numerical stability by tracking running max.
        """
        N, V = logits.shape
        device = logits.device
        dtype = logits.dtype
        
        # Initialize accumulators
        max_val = torch.full((N,), float("-inf"), device=device, dtype=dtype)
        sum_exp = torch.zeros(N, device=device, dtype=dtype)
        
        for start in range(0, V, self.chunk_size):
            end = min(start + self.chunk_size, V)
            chunk = logits[:, start:end]
            
            # Chunk maximum
            chunk_max = chunk.max(dim=-1).values
            
            # Update running maximum
            new_max = torch.maximum(max_val, chunk_max)
            
            # Rescale previous sum
            sum_exp = sum_exp * torch.exp(max_val - new_max)
            
            # Add chunk contribution
            sum_exp = sum_exp + torch.exp(chunk - new_max.unsqueeze(-1)).sum(dim=-1)
            
            max_val = new_max
        
        # Final log-sum-exp
        return max_val + torch.log(sum_exp)
    
    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute chunked cross-entropy loss."""
        # Flatten if 3D
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)
        
        N, V = logits.shape
        
        # Compute log-sum-exp efficiently
        # Note: For training, we need gradients, so we compute properly
        if self.training:
            # Use gradient checkpointing style approach
            with torch.cuda.amp.autocast(enabled=False):
                logits_fp32 = logits.float()
                
                # Stable max computation
                max_val = logits_fp32.max(dim=-1, keepdim=True).values.detach()
                
                # Compute exp in chunks to save memory
                lse_parts = []
                for start in range(0, V, self.chunk_size):
                    end = min(start + self.chunk_size, V)
                    chunk_exp = torch.exp(logits_fp32[:, start:end] - max_val)
                    lse_parts.append(chunk_exp.sum(dim=-1, keepdim=True))
                
                total_exp = sum(lse_parts)
                lse = max_val.squeeze(-1) + torch.log(total_exp.squeeze(-1))
        else:
            lse = self._compute_lse_chunked(logits)
        
        # Gather target logits
        valid_mask = labels != self.ignore_index
        labels_clamped = labels.clamp(min=0)
        
        target_logits = torch.gather(
            logits, dim=1, index=labels_clamped.unsqueeze(-1)
        ).squeeze(-1)
        
        # Cross-entropy: -log(softmax) = -target + lse
        loss = -target_logits + lse
        
        # Label smoothing
        if self.label_smoothing > 0:
            # Compute mean logit efficiently
            mean_logit = logits.mean(dim=-1)
            smooth_loss = lse - mean_logit
            loss = (1 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss
        
        # Apply mask
        loss = torch.where(valid_mask, loss, torch.zeros_like(loss))
        
        # Reduction
        if self.reduction == "mean":
            return loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "batchmean":
            return loss.mean()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification.
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  γ (gamma): Focusing parameter                                      │
    │    γ = 0: Equivalent to weighted cross-entropy                      │
    │    γ = 1: Moderate down-weighting of easy examples                  │
    │    γ = 2: Strong down-weighting (recommended default)               │
    │    γ = 5: Very aggressive focusing on hard examples                 │
    │                                                                     │
    │  α (alpha): Class balancing weight                                  │
    │    Can be scalar or per-class tensor                                │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    __constants__ = ["gamma", "reduction", "ignore_index"]
    
    def __init__(
        self,
        alpha: Optional[Union[float, Tensor]] = None,
        gamma: float = 2.0,
        reduction: ReductionMode = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        # Handle alpha (class weights)
        if alpha is None:
            self.register_buffer("alpha", None)
        elif isinstance(alpha, (int, float)):
            self.register_buffer("alpha", torch.tensor([alpha]))
        else:
            self.register_buffer("alpha", alpha)
    
    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: [batch, num_classes] or [batch, seq, num_classes]
            labels: [batch] or [batch, seq]
        
        Returns:
            Focal loss value
        """
        # Flatten if needed
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)
        
        N, C = logits.shape
        
        # Compute log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # Gather target probabilities
        valid_mask = labels != self.ignore_index
        labels_clamped = labels.clamp(min=0)
        
        pt = torch.gather(probs, dim=1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)
        log_pt = torch.gather(log_probs, dim=1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)
        
        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt).pow(self.gamma)
        
        # Base loss: -log(pt)
        loss = -focal_weight * log_pt
        
        # Apply class weights (alpha)
        if self.alpha is not None:
            if self.alpha.numel() == 1:
                # Scalar alpha
                alpha_t = self.alpha
            else:
                # Per-class alpha
                alpha_t = torch.gather(self.alpha.expand(N, -1), dim=1, 
                                       index=labels_clamped.unsqueeze(-1)).squeeze(-1)
            loss = alpha_t * loss
        
        # Apply ignore mask
        loss = torch.where(valid_mask, loss, torch.zeros_like(loss))
        
        # Reduction
        if self.reduction == "mean":
            return loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class PolyLoss(nn.Module):
    """
    Poly Loss: A Polynomial Expansion Perspective of Classification Loss.
    
    Reference: "PolyLoss: A Polynomial Expansion Perspective of Classification Loss" (Leng et al., 2022)
    
    PolyLoss = CE + ε₁ * (1 - pₜ)
    
    The first-order polynomial expansion adds a focusing term similar to focal loss
    but with linear (instead of exponential) down-weighting.
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        reduction: ReductionMode = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute Poly-1 loss."""
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)
        
        # Cross-entropy component
        ce_loss = F.cross_entropy(
            logits, labels, 
            ignore_index=self.ignore_index,
            reduction="none"
        )
        
        # Poly component: ε * (1 - pt)
        probs = F.softmax(logits, dim=-1)
        valid_mask = labels != self.ignore_index
        labels_clamped = labels.clamp(min=0)
        
        pt = torch.gather(probs, dim=1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)
        poly_term = self.epsilon * (1 - pt)
        
        # Combined loss
        loss = ce_loss + poly_term
        loss = torch.where(valid_mask, loss, torch.zeros_like(loss))
        
        if self.reduction == "mean":
            return loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# §5: Knowledge Distillation Losses
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

class DistillationLoss(nn.Module):
    """
    Comprehensive knowledge distillation loss suite.
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Distillation Methods:                                              │
    │  ├── Soft: KL divergence between temperature-scaled logits         │
    │  ├── Hard: Cross-entropy with ground truth labels                  │
    │  ├── Feature: MSE/Cosine between intermediate representations      │
    │  └── Attention: Transfer attention patterns                        │
    │                                                                     │
    │  Combined Loss = (1-α)*hard_loss + α*T²*soft_loss                  │
    │                                                                     │
    │  Temperature (T): Higher = softer distributions                     │
    │    T = 1: Original distribution                                    │
    │    T = 2-5: Recommended for distillation                           │
    │    T > 10: Very soft, may lose discriminative info                 │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        reduction: ReductionMode = "mean",
        ignore_index: int = -100,
        distill_type: Literal["soft", "hard", "combined"] = "combined",
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.distill_type = distill_type
        
        self.hard_loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction,
        )
    
    def _soft_loss(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
    ) -> Tensor:
        """KL divergence between temperature-scaled distributions."""
        T = self.temperature
        
        # Temperature-scaled log-softmax
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        
        # KL divergence: sum(p * log(p/q)) = sum(p * log(p)) - sum(p * log(q))
        # = -entropy(p) - sum(p * log(q))
        # PyTorch KL: sum(p * (log(p) - log(q)))
        kl_div = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        )
        
        # Scale by T² (gradient magnitude correction)
        return kl_div * (T ** 2)
    
    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Optional[Tensor] = None,
    ) -> LossOutput:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model outputs [batch, seq, vocab]
            teacher_logits: Teacher model outputs [batch, seq, vocab]
            labels: Ground truth labels [batch, seq] (required for hard/combined)
        
        Returns:
            LossOutput with combined loss and component metrics
        """
        # Flatten for loss computation
        if student_logits.dim() == 3:
            B, S, V = student_logits.shape
            student_flat = student_logits.reshape(-1, V)
            teacher_flat = teacher_logits.reshape(-1, V)
            labels_flat = labels.reshape(-1) if labels is not None else None
        else:
            student_flat = student_logits
            teacher_flat = teacher_logits
            labels_flat = labels
        
        metrics: Dict[str, float] = {}
        
        # Soft loss (always computed)
        soft_loss = self._soft_loss(student_flat, teacher_flat)
        metrics["soft_loss"] = soft_loss.item()
        
        # Hard loss (if labels provided)
        if labels_flat is not None and self.distill_type in ("hard", "combined"):
            hard_loss = self.hard_loss_fn(student_flat, labels_flat)
            metrics["hard_loss"] = hard_loss.item()
        else:
            hard_loss = torch.tensor(0.0, device=student_logits.device)
        
        # Combine based on distill type
        if self.distill_type == "soft":
            combined = soft_loss
        elif self.distill_type == "hard":
            combined = hard_loss
        else:  # combined
            combined = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        metrics["combined_loss"] = combined.item()
        
        return LossOutput(loss=combined, metrics=metrics)


class FeatureDistillationLoss(nn.Module):
    """
    Feature-based knowledge distillation.
    
    Transfers knowledge through intermediate layer representations.
    Useful when student has different architecture than teacher.
    """
    
    def __init__(
        self,
        loss_type: Literal["mse", "cosine", "l1"] = "mse",
        normalize: bool = True,
        projection_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
        self.projection: Optional[nn.Module] = None
        
        if projection_dim is not None:
            # Lazy initialization of projection layer
            self._projection_dim = projection_dim
    
    def _ensure_projection(self, student_dim: int, teacher_dim: int) -> None:
        """Initialize projection layer if dimensions mismatch."""
        if student_dim != teacher_dim and self.projection is None:
            target_dim = getattr(self, "_projection_dim", teacher_dim)
            self.projection = nn.Linear(student_dim, target_dim, bias=False)
    
    def forward(
        self,
        student_features: Tensor,
        teacher_features: Tensor,
    ) -> Tensor:
        """
        Compute feature distillation loss.
        
        Args:
            student_features: [batch, seq, dim_s] or [batch, dim_s]
            teacher_features: [batch, seq, dim_t] or [batch, dim_t]
        
        Returns:
            Feature alignment loss
        """
        # Project student features if needed
        if self.projection is not None:
            student_features = self.projection(student_features)
        elif student_features.shape[-1] != teacher_features.shape[-1]:
            self._ensure_projection(
                student_features.shape[-1],
                teacher_features.shape[-1],
            )
            if self.projection is not None:
                self.projection = self.projection.to(student_features.device)
                student_features = self.projection(student_features)
        
        # Normalize if requested
        if self.normalize:
            student_features = F.normalize(student_features, p=2, dim=-1)
            teacher_features = F.normalize(teacher_features, p=2, dim=-1)
        
        # Compute loss
        if self.loss_type == "mse":
            return F.mse_loss(student_features, teacher_features.detach())
        elif self.loss_type == "cosine":
            # Cosine similarity loss: 1 - cos(s, t)
            cos_sim = F.cosine_similarity(student_features, teacher_features.detach(), dim=-1)
            return (1 - cos_sim).mean()
        elif self.loss_type == "l1":
            return F.l1_loss(student_features, teacher_features.detach())
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


class AttentionDistillationLoss(nn.Module):
    """
    Attention transfer loss for distilling attention patterns.
    
    Reference: "Paying More Attention to Attention" (Zagoruyko & Komodakis, 2017)
    """
    
    def __init__(
        self,
        loss_type: Literal["mse", "kl"] = "mse",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
    
    def forward(
        self,
        student_attention: Tensor,
        teacher_attention: Tensor,
    ) -> Tensor:
        """
        Compute attention distillation loss.
        
        Args:
            student_attention: [batch, heads, seq, seq]
            teacher_attention: [batch, heads, seq, seq]
        
        Returns:
            Attention alignment loss
        """
        if self.loss_type == "mse":
            return F.mse_loss(student_attention, teacher_attention.detach())
        elif self.loss_type == "kl":
            # Flatten attention for KL computation
            B, H, S1, S2 = student_attention.shape
            student_flat = student_attention.reshape(B * H * S1, S2)
            teacher_flat = teacher_attention.reshape(B * H * S1, S2)
            
            # Temperature scaling
            student_log = F.log_softmax(student_flat / self.temperature, dim=-1)
            teacher_prob = F.softmax(teacher_flat / self.temperature, dim=-1)
            
            return F.kl_div(student_log, teacher_prob.detach(), reduction="batchmean")
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# §6: Preference Optimization Losses (RLHF)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) loss.
    
    Reference: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023)
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  DPO bypasses reward modeling and PPO by directly optimizing       │
    │  the policy to satisfy human preferences.                          │
    │                                                                     │
    │  L_DPO = -log σ(β * (log π(y_w|x)/π_ref(y_w|x)                     │
    │                    - log π(y_l|x)/π_ref(y_l|x)))                   │
    │                                                                     │
    │  Variants:                                                          │
    │  ├── sigmoid: Standard DPO (default)                               │
    │  ├── hinge: Margin-based variant                                   │
    │  ├── ipo: Identity Preference Optimization                         │
    │  └── robust: Robust regression variant                             │
    │                                                                     │
    │  β (beta): Controls deviation from reference policy                 │
    │    β = 0.1-0.5: Recommended range                                  │
    │    Lower β: More exploration, may deviate from reference           │
    │    Higher β: Stays closer to reference, less alignment power       │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: PreferenceLossType = "sigmoid",
        reference_free: bool = False,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.reference_free = reference_free
    
    def forward(
        self,
        policy_chosen_logps: Tensor,
        policy_rejected_logps: Tensor,
        reference_chosen_logps: Optional[Tensor] = None,
        reference_rejected_logps: Optional[Tensor] = None,
    ) -> LossOutput:
        """
        Compute DPO loss.
        
        Args:
            policy_chosen_logps: Log probs of chosen responses [batch]
            policy_rejected_logps: Log probs of rejected responses [batch]
            reference_chosen_logps: Reference log probs for chosen [batch]
            reference_rejected_logps: Reference log probs for rejected [batch]
        
        Returns:
            LossOutput with loss, chosen/rejected rewards
        """
        device = policy_chosen_logps.device
        
        if self.reference_free:
            # Reference-free: use policy as implicit reference
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = torch.zeros_like(pi_logratios)
        else:
            if reference_chosen_logps is None or reference_rejected_logps is None:
                raise ValueError("Reference log probs required for non-reference-free DPO")
            
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        # Log-ratio difference: measures preference alignment
        logits = pi_logratios - ref_logratios
        
        # Compute loss based on variant
        if self.loss_type == "sigmoid":
            # Standard DPO: -log σ(β * logits)
            losses = -F.logsigmoid(self.beta * logits)
        
        elif self.loss_type == "hinge":
            # Hinge loss: max(0, 1 - β * logits)
            losses = F.relu(1 - self.beta * logits)
        
        elif self.loss_type == "ipo":
            # Identity Preference Optimization: (logits - 1/(2β))²
            # Reference: "A General Theoretical Paradigm to Understand Learning from Human Preferences"
            losses = (logits - 1 / (2 * self.beta)) ** 2
        
        elif self.loss_type == "kto":
            # Kahneman-Tversky Optimization (partial implementation)
            # Full KTO uses separate chosen/rejected losses
            losses = -F.logsigmoid(self.beta * logits)
        
        elif self.loss_type == "robust":
            # Robust regression variant (less sensitive to outliers)
            losses = F.smooth_l1_loss(
                logits,
                torch.ones_like(logits) / self.beta,
                reduction="none",
            )
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # Label smoothing: mix with reverse preference
        if self.label_smoothing > 0:
            smooth_losses = -F.logsigmoid(-self.beta * logits)
            losses = (1 - self.label_smoothing) * losses + self.label_smoothing * smooth_losses
        
        # Compute implicit rewards for logging
        if not self.reference_free:
            chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        else:
            chosen_rewards = self.beta * policy_chosen_logps
            rejected_rewards = self.beta * policy_rejected_logps
        
        metrics = {
            "loss": losses.mean().item(),
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
            "accuracy": (logits > 0).float().mean().item(),
        }
        
        return LossOutput(
            loss=losses.mean(),
            metrics=metrics,
        )


class KTOLoss(nn.Module):
    """
    Kahneman-Tversky Optimization (KTO) loss.
    
    Reference: "KTO: Model Alignment as Prospect Theoretic Optimization" (Ethayarajh et al., 2024)
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  KTO applies prospect theory to preference learning:               │
    │  - Asymmetric treatment of chosen vs rejected                      │
    │  - Loss aversion: rejected penalized more than chosen rewarded     │
    │  - Works with unpaired preference data                             │
    │                                                                     │
    │  L_KTO = (1-y) * λ_u * σ(β(r_θ(x,y) - z_ref))                      │
    │        + y * λ_d * σ(-β(r_θ(x,y) - z_ref))                         │
    │                                                                     │
    │  where z_ref = KL(π_θ || π_ref) baseline                           │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight
    
    def forward(
        self,
        policy_logps: Tensor,
        reference_logps: Tensor,
        is_desirable: Tensor,
        kl_baseline: Optional[Tensor] = None,
    ) -> LossOutput:
        """
        Compute KTO loss.
        
        Args:
            policy_logps: Policy log probs [batch]
            reference_logps: Reference log probs [batch]
            is_desirable: Binary mask, 1 for chosen, 0 for rejected [batch]
            kl_baseline: Optional KL baseline [batch] or scalar
        
        Returns:
            LossOutput with KTO loss
        """
        # Compute implicit reward
        reward = policy_logps - reference_logps
        
        # KL baseline (can be computed from reference or provided)
        if kl_baseline is None:
            # Use batch average as baseline
            kl_baseline = (policy_logps - reference_logps).mean().detach()
        
        # Centered reward
        centered_reward = reward - kl_baseline
        
        # Compute losses separately for desirable and undesirable
        # Desirable: want high reward, penalize low
        # Undesirable: want low reward, penalize high
        
        desirable_losses = -F.logsigmoid(self.beta * centered_reward)
        undesirable_losses = -F.logsigmoid(-self.beta * centered_reward)
        
        # Combine with asymmetric weights
        is_desirable_float = is_desirable.float()
        losses = (
            is_desirable_float * self.desirable_weight * desirable_losses
            + (1 - is_desirable_float) * self.undesirable_weight * undesirable_losses
        )
        
        metrics = {
            "loss": losses.mean().item(),
            "desirable_loss": (desirable_losses * is_desirable_float).sum().item() / is_desirable_float.sum().clamp(min=1).item(),
            "undesirable_loss": (undesirable_losses * (1 - is_desirable_float)).sum().item() / (1 - is_desirable_float).sum().clamp(min=1).item(),
            "mean_reward": reward.mean().item(),
        }
        
        return LossOutput(loss=losses.mean(), metrics=metrics)


class ORPOLoss(nn.Module):
    """
    Odds Ratio Preference Optimization (ORPO) loss.
    
    Reference: "ORPO: Monolithic Preference Optimization without Reference Model" (Hong et al., 2024)
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  ORPO combines SFT and preference alignment in single loss:        │
    │                                                                     │
    │  L_ORPO = L_NLL + λ * L_OR                                         │
    │                                                                     │
    │  L_OR = -log σ(log (p(y_w|x) / (1-p(y_w|x)))                       │
    │              / (p(y_l|x) / (1-p(y_l|x))))                          │
    │                                                                     │
    │  Benefits:                                                          │
    │  ├── No reference model needed (memory efficient)                  │
    │  ├── Single training phase (no SFT then RLHF)                      │
    │  └── Simpler hyperparameter tuning                                 │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.alpha = alpha
    
    def forward(
        self,
        policy_chosen_logps: Tensor,
        policy_rejected_logps: Tensor,
        chosen_nll_loss: Tensor,
    ) -> LossOutput:
        """
        Compute ORPO loss.
        
        Args:
            policy_chosen_logps: Log probs of chosen [batch]
            policy_rejected_logps: Log probs of rejected [batch]
            chosen_nll_loss: SFT loss on chosen responses
        
        Returns:
            LossOutput with total loss
        """
        # Convert log probs to probs for odds ratio
        # Using log-space for numerical stability:
        # log(odds(p)) = log(p / (1-p)) = log(p) - log(1-p)
        #              = logp - log(1 - exp(logp))
        #              ≈ logp - log(-logp) for small p (approximation)
        
        # More stable: log(p/(1-p)) = logp - log1p(-exp(logp))
        def log_odds(logp: Tensor) -> Tensor:
            """Compute log-odds from log-probability."""
            return logp - torch.log1p(-torch.exp(logp.clamp(max=-EPS)))
        
        chosen_log_odds = log_odds(policy_chosen_logps)
        rejected_log_odds = log_odds(policy_rejected_logps)
        
        # Log odds ratio
        log_odds_ratio = chosen_log_odds - rejected_log_odds
        
        # ORPO loss: -log(sigmoid(log_odds_ratio))
        orpo_loss = -F.logsigmoid(log_odds_ratio)
        
        # Combined loss
        total_loss = chosen_nll_loss + self.alpha * orpo_loss.mean()
        
        # Odds ratio for logging (sigmoid of log odds ratio)
        odds_ratio = torch.sigmoid(log_odds_ratio)
        
        metrics = {
            "total_loss": total_loss.item(),
            "nll_loss": chosen_nll_loss.item(),
            "orpo_loss": orpo_loss.mean().item(),
            "odds_ratio": odds_ratio.mean().item(),
            "accuracy": (log_odds_ratio > 0).float().mean().item(),
        }
        
        return LossOutput(loss=total_loss, metrics=metrics)


class SimPOLoss(nn.Module):
    """
    Simple Preference Optimization (SimPO) loss.
    
    Reference: "SimPO: Simple Preference Optimization with a Reference-Free Reward" (Meng et al., 2024)
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  SimPO simplifies DPO by:                                          │
    │  1. Removing reference model (like ORPO)                           │
    │  2. Length-normalizing rewards                                     │
    │  3. Adding target margin γ                                         │
    │                                                                     │
    │  L_SimPO = -log σ(β * (r_w/|y_w| - r_l/|y_l|) - γ)                │
    │                                                                     │
    │  Length normalization prevents bias toward shorter/longer responses │
    │  Margin γ encourages larger reward gaps                            │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        beta: float = 2.0,
        gamma: float = 0.5,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        policy_chosen_logps: Tensor,
        policy_rejected_logps: Tensor,
        chosen_length: Tensor,
        rejected_length: Tensor,
    ) -> LossOutput:
        """
        Compute SimPO loss.
        
        Args:
            policy_chosen_logps: Log probs of chosen [batch]
            policy_rejected_logps: Log probs of rejected [batch]
            chosen_length: Response lengths [batch]
            rejected_length: Response lengths [batch]
        
        Returns:
            LossOutput with SimPO loss
        """
        # Length-normalized rewards
        chosen_rewards = policy_chosen_logps / chosen_length.clamp(min=1)
        rejected_rewards = policy_rejected_logps / rejected_length.clamp(min=1)
        
        # SimPO logits with margin
        logits = self.beta * (chosen_rewards - rejected_rewards) - self.gamma
        
        # Loss
        losses = -F.logsigmoid(logits)
        
        # Label smoothing
        if self.label_smoothing > 0:
            smooth_losses = -F.logsigmoid(-logits)
            losses = (1 - self.label_smoothing) * losses + self.label_smoothing * smooth_losses
        
        metrics = {
            "loss": losses.mean().item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
            "accuracy": (logits > 0).float().mean().item(),
        }
        
        return LossOutput(loss=losses.mean(), metrics=metrics)


class CPOLoss(nn.Module):
    """
    Contrastive Preference Optimization (CPO) loss.
    
    Reference: "Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation" (Xu et al., 2024)
    
    Extends DPO with contrastive learning principles for multiple
    rejected responses per chosen response.
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        bc_coef: float = 0.0,  # Behavior cloning coefficient
    ) -> None:
        super().__init__()
        self.beta = beta
        self.bc_coef = bc_coef
    
    def forward(
        self,
        policy_chosen_logps: Tensor,
        policy_rejected_logps: Tensor,
        reference_chosen_logps: Tensor,
        reference_rejected_logps: Tensor,
        chosen_nll_loss: Optional[Tensor] = None,
    ) -> LossOutput:
        """
        Compute CPO loss.
        
        Args:
            policy_chosen_logps: [batch]
            policy_rejected_logps: [batch, num_rejected] or [batch]
            reference_chosen_logps: [batch]
            reference_rejected_logps: [batch, num_rejected] or [batch]
            chosen_nll_loss: Optional SFT component
        
        Returns:
            LossOutput with CPO loss
        """
        # Handle single vs multiple rejected
        if policy_rejected_logps.dim() == 1:
            policy_rejected_logps = policy_rejected_logps.unsqueeze(1)
            reference_rejected_logps = reference_rejected_logps.unsqueeze(1)
        
        batch_size, num_rejected = policy_rejected_logps.shape
        
        # Expand chosen for broadcasting
        policy_chosen = policy_chosen_logps.unsqueeze(1).expand(-1, num_rejected)
        reference_chosen = reference_chosen_logps.unsqueeze(1).expand(-1, num_rejected)
        
        # Log ratios
        pi_logratios = policy_chosen - policy_rejected_logps
        ref_logratios = reference_chosen - reference_rejected_logps
        
        logits = pi_logratios - ref_logratios
        
        # Contrastive loss (sum over rejected)
        losses = -F.logsigmoid(self.beta * logits).mean(dim=1)
        
        # Optional behavior cloning term
        if self.bc_coef > 0 and chosen_nll_loss is not None:
            losses = losses + self.bc_coef * chosen_nll_loss
        
        metrics = {
            "loss": losses.mean().item(),
            "accuracy": (logits > 0).float().mean().item(),
        }
        
        return LossOutput(loss=losses.mean(), metrics=metrics)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# §7: Contrastive Learning Losses
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for representation learning.
    
    Reference: "Representation Learning with Contrastive Predictive Coding" (van den Oord et al., 2018)
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  L_InfoNCE = -log(exp(sim(a,b+)/τ) / Σ exp(sim(a,b)/τ))           │
    │                                                                     │
    │  Core components:                                                   │
    │  ├── Anchor: Query representation                                  │
    │  ├── Positive: Matching representation                             │
    │  └── Negatives: Non-matching representations                       │
    │                                                                     │
    │  Temperature τ:                                                     │
    │    τ = 0.07: Common for vision (CLIP, SimCLR)                      │
    │    τ = 0.05: More aggressive separation                            │
    │    τ = 1.0: Softer, prevents collapse                              │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: ReductionMode = "mean",
        gather_distributed: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.gather_distributed = gather_distributed
    
    def forward(
        self,
        embeddings_a: Tensor,
        embeddings_b: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            embeddings_a: First view embeddings [batch, dim]
            embeddings_b: Second view embeddings [batch, dim]
            labels: Optional custom positive indices [batch]
        
        Returns:
            Contrastive loss value
        """
        batch_size = embeddings_a.shape[0]
        device = embeddings_a.device
        
        # L2 normalize embeddings
        embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)
        
        # Gather embeddings across GPUs for larger batch
        if self.gather_distributed and dist.is_initialized():
            embeddings_a = self._gather_all(embeddings_a)
            embeddings_b = self._gather_all(embeddings_b)
            batch_size = embeddings_a.shape[0]
        
        # Compute similarity matrix
        logits = embeddings_a @ embeddings_b.T / self.temperature
        
        # Positive pairs on diagonal
        if labels is None:
            labels = torch.arange(batch_size, device=device)
        
        # Symmetric loss
        loss_a = F.cross_entropy(logits, labels, reduction=self.reduction)
        loss_b = F.cross_entropy(logits.T, labels, reduction=self.reduction)
        
        return (loss_a + loss_b) / 2
    
    @staticmethod
    def _gather_all(tensor: Tensor) -> Tensor:
        """Gather tensors from all ranks."""
        if not dist.is_initialized():
            return tensor
        
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)


class CLIPLoss(nn.Module):
    """
    CLIP-style contrastive loss for multimodal learning.
    
    Reference: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  CLIP uses symmetric InfoNCE between image and text embeddings:    │
    │                                                                     │
    │  L_CLIP = (L_i2t + L_t2i) / 2                                      │
    │                                                                     │
    │  with learnable temperature: logit_scale = exp(log_scale)          │
    │  Clamped to prevent instability: [1, 100]                          │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        init_temperature: float = 0.07,
        learnable_temperature: bool = True,
        max_temperature: float = 100.0,
    ) -> None:
        super().__init__()
        self.max_temperature = max_temperature
        
        # Learnable log temperature (for numerical stability)
        log_temp = math.log(1 / init_temperature)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(log_temp))
        else:
            self.register_buffer("log_temperature", torch.tensor(log_temp))
    
    @property
    def temperature(self) -> Tensor:
        """Get current temperature value."""
        return self.log_temperature.exp().clamp(max=self.max_temperature)
    
    def forward(
        self,
        image_embeddings: Tensor,
        text_embeddings: Tensor,
    ) -> LossOutput:
        """
        Compute CLIP loss.
        
        Args:
            image_embeddings: [batch, dim]
            text_embeddings: [batch, dim]
        
        Returns:
            LossOutput with CLIP loss and metrics
        """
        batch_size = image_embeddings.shape[0]
        device = image_embeddings.device
        
        # Normalize
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Scaled similarity
        logits_per_image = self.temperature * (image_embeddings @ text_embeddings.T)
        logits_per_text = logits_per_image.T
        
        # Labels (diagonal = positive pairs)
        labels = torch.arange(batch_size, device=device)
        
        # Symmetric cross entropy
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i) / 2
        
        # Accuracy for monitoring
        with torch.no_grad():
            acc_i2t = (logits_per_image.argmax(dim=1) == labels).float().mean()
            acc_t2i = (logits_per_text.argmax(dim=1) == labels).float().mean()
        
        metrics = {
            "loss": loss.item(),
            "i2t_loss": loss_i2t.item(),
            "t2i_loss": loss_t2i.item(),
            "i2t_accuracy": acc_i2t.item(),
            "t2i_accuracy": acc_t2i.item(),
            "temperature": self.temperature.item(),
        }
        
        return LossOutput(loss=loss, metrics=metrics)


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning.
    
    Reference: "FaceNet: A Unified Embedding for Face Recognition and Clustering" (Schroff et al., 2015)
    
    L = max(0, d(a, p) - d(a, n) + margin)
    
    where d is distance metric (euclidean or cosine).
    """
    
    def __init__(
        self,
        margin: float = 0.3,
        distance: Literal["euclidean", "cosine"] = "euclidean",
        hard_mining: bool = False,
        reduction: ReductionMode = "mean",
    ) -> None:
        super().__init__()
        self.margin = margin
        self.distance = distance
        self.hard_mining = hard_mining
        self.reduction = reduction
    
    def _compute_distance(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute distance between embeddings."""
        if self.distance == "euclidean":
            return (x - y).pow(2).sum(dim=-1).sqrt()
        elif self.distance == "cosine":
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
            return 1 - (x * y).sum(dim=-1)
        else:
            raise ValueError(f"Unknown distance: {self.distance}")
    
    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings [batch, dim]
            positive: Positive embeddings [batch, dim]
            negative: Negative embeddings [batch, dim]
        
        Returns:
            Triplet loss value
        """
        pos_dist = self._compute_distance(anchor, positive)
        neg_dist = self._compute_distance(anchor, negative)
        
        losses = F.relu(pos_dist - neg_dist + self.margin)
        
        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        else:
            return losses


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    
    Reference: "A Simple Framework for Contrastive Learning of Visual Representations" (Chen et al., 2020)
    
    Used in SimCLR. Similar to InfoNCE but with explicit positive mask.
    """
    
    def __init__(
        self,
        temperature: float = 0.5,
        reduction: ReductionMode = "mean",
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Compute NT-Xent loss.
        
        Expects embeddings from two augmented views stacked:
        [view1_batch, view2_batch] of shape [2*batch, dim]
        
        Args:
            embeddings: Stacked embeddings [2*batch, dim]
        
        Returns:
            NT-Xent loss
        """
        batch_size = embeddings.shape[0] // 2
        device = embeddings.device
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # Similarity matrix
        sim = embeddings @ embeddings.T / self.temperature
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))
        
        # Positive pairs: (i, i+batch) and (i+batch, i)
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=device)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, device=device)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, device=device)
        
        # Log-softmax
        log_probs = F.log_softmax(sim, dim=1)
        
        # Extract positive pair log probs
        loss = -(log_probs * pos_mask).sum(dim=1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    
    Reference: "Supervised Contrastive Learning" (Khosla et al., 2020)
    
    Extends contrastive learning to leverage label information.
    Multiple positives per anchor (all same-class samples).
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        reduction: ReductionMode = "mean",
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.reduction = reduction
    
    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            embeddings: [batch, n_views, dim] or [batch, dim]
            labels: Class labels [batch]
            mask: Optional custom positive mask [batch, batch]
        
        Returns:
            SupCon loss
        """
        device = embeddings.device
        
        if embeddings.dim() == 3:
            # Multiple views: [batch, n_views, dim]
            batch_size, n_views, _ = embeddings.shape
            embeddings = embeddings.reshape(batch_size * n_views, -1)
            labels = labels.repeat(n_views)
        else:
            batch_size = embeddings.shape[0]
            n_views = 1
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # Compute similarity
        sim = embeddings @ embeddings.T / self.temperature
        
        # Mask self-contrast
        n_samples = embeddings.shape[0]
        self_mask = torch.eye(n_samples, device=device, dtype=torch.bool)
        
        # Positive mask: same label (excluding self)
        if mask is None:
            mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask = mask.float().masked_fill(self_mask, 0)
        
        # For numerical stability
        logits_max = sim.max(dim=1, keepdim=True).values.detach()
        logits = sim - logits_max
        
        # Log-sum-exp (denominator)
        exp_logits = torch.exp(logits).masked_fill(self_mask, 0)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + EPS)
        
        # Mean log-likelihood over positives
        mask_sum = mask.sum(dim=1).clamp(min=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask_sum
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# §8: MoE Regularization Losses
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

class ZLoss(nn.Module):
    """
    Z-Loss for MoE router regularization.
    
    Reference: "ST-MoE: Designing Stable and Transferable Sparse Expert Models" (Zoph et al., 2022)
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Z-Loss = coef * mean(log(sum(exp(router_logits))))²               │
    │                                                                     │
    │  Encourages router logits to remain small, preventing:             │
    │  ├── Numerical overflow in softmax                                 │
    │  ├── Gradient explosion                                            │
    │  └── Router saturation                                             │
    │                                                                     │
    │  Typical coefficient: 0.001 - 0.01                                 │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, z_loss_coef: float = 0.001) -> None:
        super().__init__()
        self.z_loss_coef = z_loss_coef
    
    def forward(self, router_logits: Tensor) -> Tensor:
        """
        Compute z-loss.
        
        Args:
            router_logits: [batch, seq, num_experts] or list of such tensors
        
        Returns:
            Z-loss value
        """
        if isinstance(router_logits, (list, tuple)):
            return sum(self.forward(rl) for rl in router_logits)
        
        # Log-sum-exp
        lse = torch.logsumexp(router_logits, dim=-1)
        
        # Squared LSE
        z_loss = lse.pow(2).mean()
        
        return self.z_loss_coef * z_loss


class LoadBalancingLoss(nn.Module):
    """
    Load balancing auxiliary loss for MoE models.
    
    Reference: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (Fedus et al., 2021)
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  L_aux = α * N * Σ_i (f_i * P_i)                                   │
    │                                                                     │
    │  where:                                                             │
    │  f_i = fraction of tokens routed to expert i                       │
    │  P_i = mean routing probability for expert i                       │
    │  N = number of experts                                             │
    │                                                                     │
    │  Minimized when tokens uniformly distributed across experts        │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        num_experts: int,
        aux_loss_coef: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.aux_loss_coef = aux_loss_coef
    
    def forward(
        self,
        router_probs: Tensor,
        expert_indices: Tensor,
    ) -> LossOutput:
        """
        Compute load balancing loss.
        
        Args:
            router_probs: Routing probabilities [batch*seq, num_experts]
            expert_indices: Selected expert indices [batch*seq, top_k]
        
        Returns:
            LossOutput with aux loss and load metrics
        """
        if router_probs.dim() == 3:
            router_probs = router_probs.reshape(-1, router_probs.shape[-1])
        if expert_indices.dim() == 3:
            expert_indices = expert_indices.reshape(-1, expert_indices.shape[-1])
        
        num_tokens = router_probs.shape[0]
        
        # Compute fraction of tokens per expert (f_i)
        one_hot = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        if one_hot.dim() == 3:
            one_hot = one_hot.sum(dim=1)  # Sum over top-k
        tokens_per_expert = one_hot.sum(dim=0)
        f = tokens_per_expert / num_tokens
        
        # Mean routing probability per expert (P_i)
        P = router_probs.mean(dim=0)
        
        # Auxiliary loss
        aux_loss = self.num_experts * (f * P).sum()
        
        # Compute load imbalance metrics
        load = tokens_per_expert
        ideal_load = num_tokens / self.num_experts
        load_imbalance = (load.max() - load.min()) / ideal_load
        
        metrics = {
            "aux_loss": (self.aux_loss_coef * aux_loss).item(),
            "load_imbalance": load_imbalance.item(),
            "max_load": load.max().item(),
            "min_load": load.min().item(),
        }
        
        return LossOutput(
            loss=self.aux_loss_coef * aux_loss,
            metrics=metrics,
        )


class ExpertEntropyLoss(nn.Module):
    """
    Expert entropy regularization for MoE.
    
    Encourages router to make confident (low-entropy) decisions.
    Prevents uniform/uncertain routing which wastes capacity.
    """
    
    def __init__(
        self,
        entropy_coef: float = 0.01,
        target_entropy: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.entropy_coef = entropy_coef
        self.target_entropy = target_entropy
    
    def forward(self, router_probs: Tensor) -> Tensor:
        """
        Compute entropy-based regularization.
        
        Args:
            router_probs: [batch, seq, num_experts]
        
        Returns:
            Entropy loss
        """
        # Compute entropy: -sum(p * log(p))
        entropy = -(router_probs * (router_probs + EPS).log()).sum(dim=-1)
        
        if self.target_entropy is not None:
            # Encourage entropy toward target
            loss = (entropy - self.target_entropy).pow(2).mean()
        else:
            # Minimize entropy (encourage confident routing)
            loss = entropy.mean()
        
        return self.entropy_coef * loss


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# §9: Reconstruction & Regression Losses
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

class HuberLoss(nn.Module):
    """
    Huber loss (smooth L1) for robust regression.
    
    Combines MSE for small errors and MAE for large errors.
    Less sensitive to outliers than MSE.
    
    L_δ(a) = 0.5 * a² if |a| ≤ δ else δ * (|a| - 0.5 * δ)
    """
    
    def __init__(
        self,
        delta: float = 1.0,
        reduction: ReductionMode = "mean",
    ) -> None:
        super().__init__()
        self.delta = delta
        self.reduction = reduction
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute Huber loss."""
        return F.huber_loss(pred, target, delta=self.delta, reduction=self.reduction)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss for robust regression (image restoration).
    
    L = sqrt((pred - target)² + ε²)
    
    Approximates L1 but is differentiable everywhere.
    """
    
    def __init__(
        self,
        epsilon: float = 1e-3,
        reduction: ReductionMode = "mean",
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute Charbonnier loss."""
        loss = torch.sqrt((pred - target).pow(2) + self.epsilon ** 2)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained network features.
    
    Reference: "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" (Johnson et al., 2016)
    
    Uses VGG or other networks' intermediate features to compare
    high-level structure rather than pixel-level differences.
    """
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        layer_weights: Optional[Dict[str, float]] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.layer_weights = layer_weights or {"default": 1.0}
        self.normalize = normalize
    
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted images [batch, channels, height, width]
            target: Target images [batch, channels, height, width]
        
        Returns:
            Perceptual loss
        """
        with torch.no_grad():
            target_features = self.feature_extractor(target)
        
        pred_features = self.feature_extractor(pred)
        
        # Handle single tensor or dict of features
        if isinstance(pred_features, Tensor):
            pred_features = {"default": pred_features}
            target_features = {"default": target_features}
        
        loss = torch.tensor(0.0, device=pred.device)
        
        for name, weight in self.layer_weights.items():
            if name in pred_features:
                pred_feat = pred_features[name]
                tgt_feat = target_features[name].detach()
                
                if self.normalize:
                    pred_feat = F.normalize(pred_feat, p=2, dim=1)
                    tgt_feat = F.normalize(tgt_feat, p=2, dim=1)
                
                loss = loss + weight * F.mse_loss(pred_feat, tgt_feat)
        
        return loss


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# §10: Ranking Losses
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

class MarginRankingLoss(nn.Module):
    """
    Margin ranking loss for pairwise ranking.
    
    L = max(0, -y * (x1 - x2) + margin)
    
    where y ∈ {-1, +1} indicates which input should be ranked higher.
    """
    
    def __init__(
        self,
        margin: float = 0.0,
        reduction: ReductionMode = "mean",
    ) -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        input1: Tensor,
        input2: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute margin ranking loss."""
        return F.margin_ranking_loss(
            input1, input2, target,
            margin=self.margin,
            reduction=self.reduction,
        )


class ListMLELoss(nn.Module):
    """
    ListMLE loss for listwise ranking.
    
    Reference: "Listwise Approach to Learning to Rank" (Xia et al., 2008)
    
    Optimizes the likelihood of the correct permutation.
    """
    
    def __init__(
        self,
        eps: float = 1e-10,
        reduction: ReductionMode = "mean",
    ) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(
        self,
        scores: Tensor,
        relevance: Tensor,
    ) -> Tensor:
        """
        Compute ListMLE loss.
        
        Args:
            scores: Predicted scores [batch, list_size]
            relevance: Ground truth relevance [batch, list_size]
        
        Returns:
            ListMLE loss
        """
        # Sort by true relevance (descending) to get ideal order
        _, perm = relevance.sort(descending=True, dim=1)
        
        # Apply permutation to scores
        sorted_scores = torch.gather(scores, dim=1, index=perm)
        
        # ListMLE: -log(prod_i softmax(remaining scores at position i))
        # = -sum_i (score[i] - logsumexp(scores[i:]))
        
        batch_size, list_size = sorted_scores.shape
        device = sorted_scores.device
        
        # Compute cumulative log-sum-exp from right
        # For efficiency, we compute logsumexp of suffix for each position
        cumsoftmax = torch.zeros(batch_size, list_size, device=device)
        
        # Reverse cumulative logsumexp
        rev_scores = sorted_scores.flip(dims=[1])
        
        # Compute logsumexp cumulatively
        running_lse = torch.full((batch_size,), float("-inf"), device=device)
        for i in range(list_size):
            running_lse = torch.logaddexp(running_lse, rev_scores[:, i])
            cumsoftmax[:, list_size - 1 - i] = running_lse
        
        # Loss: -sum(score - logsumexp(suffix))
        losses = -(sorted_scores - cumsoftmax).sum(dim=1)
        
        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        else:
            return losses


class LambdaRankLoss(nn.Module):
    """
    LambdaRank loss for learning to rank.
    
    Reference: "Learning to Rank with Nonsmooth Cost Functions" (Burges et al., 2006)
    
    Weights pairwise gradients by change in NDCG.
    """
    
    def __init__(
        self,
        sigma: float = 1.0,
        reduction: ReductionMode = "mean",
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.reduction = reduction
    
    def _compute_delta_ndcg(
        self,
        relevance: Tensor,
        ranks: Tensor,
    ) -> Tensor:
        """Compute |Δ NDCG| for swapping each pair."""
        batch_size, list_size = relevance.shape
        device = relevance.device
        
        # DCG gains: 2^rel - 1
        gains = 2.0 ** relevance - 1.0
        
        # Discounts: 1 / log2(rank + 1)
        discounts = 1.0 / torch.log2(ranks.float() + 2)
        
        # Ideal DCG
        sorted_gains, _ = gains.sort(descending=True, dim=1)
        ideal_discounts = 1.0 / torch.log2(torch.arange(1, list_size + 1, device=device).float() + 1)
        idcg = (sorted_gains * ideal_discounts).sum(dim=1, keepdim=True).clamp(min=1e-8)
        
        # Current DCG contribution per item
        current_contrib = gains * discounts
        
        # Delta NDCG for swapping i and j
        # |Δ NDCG|_{ij} = |gain_i * (disc_j - disc_i) + gain_j * (disc_i - disc_j)| / IDCG
        
        delta_ndcg = torch.zeros(batch_size, list_size, list_size, device=device)
        
        for i in range(list_size):
            for j in range(i + 1, list_size):
                delta = torch.abs(
                    gains[:, i] * (discounts[:, j] - discounts[:, i])
                    + gains[:, j] * (discounts[:, i] - discounts[:, j])
                )
                delta_ndcg[:, i, j] = delta.squeeze() / idcg.squeeze()
                delta_ndcg[:, j, i] = delta_ndcg[:, i, j]
        
        return delta_ndcg
    
    def forward(
        self,
        scores: Tensor,
        relevance: Tensor,
    ) -> Tensor:
        """
        Compute LambdaRank loss.
        
        Args:
            scores: Predicted scores [batch, list_size]
            relevance: Ground truth relevance [batch, list_size]
        
        Returns:
            LambdaRank loss
        """
        batch_size, list_size = scores.shape
        device = scores.device
        
        # Get current ranking
        _, ranks = scores.sort(descending=True, dim=1)
        ranks = ranks.argsort(dim=1)  # Position of each item
        
        # Compute delta NDCG weights
        delta_ndcg = self._compute_delta_ndcg(relevance, ranks)
        
        # Score differences
        score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # [batch, i, j]
        
        # Pairwise labels: 1 if rel_i > rel_j, -1 if rel_i < rel_j, 0 if equal
        rel_diff = relevance.unsqueeze(2) - relevance.unsqueeze(1)
        pairwise_labels = torch.sign(rel_diff)
        
        # RankNet loss weighted by |Δ NDCG|
        losses = delta_ndcg * torch.log1p(torch.exp(-self.sigma * pairwise_labels * score_diff))
        
        # Mask diagonal and average over valid pairs
        mask = torch.ones(list_size, list_size, device=device).triu(1)
        losses = losses * mask
                # Sum over valid pairs
        num_pairs = mask.sum()
        loss = losses.sum(dim=[1, 2]) / num_pairs.clamp(min=1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# §11: Specialized Segmentation & Detection Losses
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    
    Reference: "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" (Milletari et al., 2016)
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Dice = 2 * |A ∩ B| / (|A| + |B|)                                  │
    │  L_Dice = 1 - Dice                                                 │
    │                                                                     │
    │  Benefits:                                                          │
    │  ├── Handles class imbalance naturally                             │
    │  ├── Region-based metric (not pixel-independent)                   │
    │  └── Commonly used in medical imaging                              │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        reduction: ReductionMode = "mean",
        square_denominator: bool = False,
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.square_denominator = square_denominator
    
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: Predictions [batch, classes, ...] (after sigmoid/softmax)
            target: Targets [batch, classes, ...] (one-hot or soft labels)
        
        Returns:
            Dice loss
        """
        # Flatten spatial dimensions
        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
        target_flat = target.reshape(target.shape[0], target.shape[1], -1)
        
        # Intersection and cardinalities
        intersection = (pred_flat * target_flat).sum(dim=-1)
        
        if self.square_denominator:
            cardinality = pred_flat.pow(2).sum(dim=-1) + target_flat.pow(2).sum(dim=-1)
        else:
            cardinality = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1)
        
        # Dice coefficient per class
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Dice loss
        loss = 1.0 - dice
        
        # Average over classes
        loss = loss.mean(dim=1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalization of Dice for asymmetric FP/FN weighting.
    
    Reference: "Tversky loss function for image segmentation using 3D FCNN" (Salehi et al., 2017)
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Tversky = TP / (TP + α*FP + β*FN)                                 │
    │                                                                     │
    │  Special cases:                                                     │
    │  α = β = 0.5: Equivalent to Dice                                   │
    │  α = β = 1.0: Equivalent to Jaccard/IoU                            │
    │  α < β: Penalize false negatives more (recall-focused)             │
    │  α > β: Penalize false positives more (precision-focused)          │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        reduction: ReductionMode = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Compute Tversky loss.
        
        Args:
            pred: Predictions [batch, classes, ...]
            target: Targets [batch, classes, ...]
        
        Returns:
            Tversky loss
        """
        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
        target_flat = target.reshape(target.shape[0], target.shape[1], -1)
        
        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum(dim=-1)
        fp = (pred_flat * (1 - target_flat)).sum(dim=-1)
        fn = ((1 - pred_flat) * target_flat).sum(dim=-1)
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Loss
        loss = (1.0 - tversky).mean(dim=1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - Combines Tversky with focal weighting.
    
    Reference: "A Novel Focal Tversky loss function for Imbalanced Data" (Abraham & Khan, 2019)
    
    L = (1 - Tversky)^γ
    
    Higher γ focuses on hard examples (low Tversky regions).
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        smooth: float = 1.0,
        reduction: ReductionMode = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute Focal Tversky loss."""
        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
        target_flat = target.reshape(target.shape[0], target.shape[1], -1)
        
        tp = (pred_flat * target_flat).sum(dim=-1)
        fp = (pred_flat * (1 - target_flat)).sum(dim=-1)
        fn = ((1 - pred_flat) * target_flat).sum(dim=-1)
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Focal weighting
        loss = (1.0 - tversky).pow(self.gamma).mean(dim=1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class GHMCLoss(nn.Module):
    """
    Gradient Harmonizing Mechanism for Classification (GHM-C).
    
    Reference: "Gradient Harmonized Single-stage Detector" (Li et al., 2019)
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  GHM addresses gradient imbalance from easy/hard examples:         │
    │                                                                     │
    │  - Easy examples: |gradient| ≈ 0 (well-classified)                 │
    │  - Hard examples: |gradient| ≈ 1 (misclassified)                   │
    │  - Most examples cluster at extremes                               │
    │                                                                     │
    │  Solution: Weight by inverse gradient density                      │
    │  w_i = N / (GD(g_i) * M)                                           │
    │                                                                     │
    │  GD = gradient density in bin containing g_i                       │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        bins: int = 10,
        momentum: float = 0.0,
        reduction: ReductionMode = "mean",
    ) -> None:
        super().__init__()
        self.bins = bins
        self.momentum = momentum
        self.reduction = reduction
        
        # Bin edges for gradient magnitude
        edges = torch.linspace(0.0, 1.0, bins + 1)
        self.register_buffer("edges", edges)
        
        # Running average of bin counts (for momentum)
        if momentum > 0:
            self.register_buffer("acc_sum", torch.zeros(bins))
    
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute GHM-C loss.
        
        Args:
            pred: Logits [batch, num_classes, ...] or [batch, ...]
            target: Labels (same shape as pred for binary, or class indices)
            mask: Optional mask for valid samples
        
        Returns:
            GHM-weighted loss
        """
        # Convert to binary case if needed
        if pred.dim() > target.dim():
            # Multi-class: use softmax
            probs = F.softmax(pred, dim=1)
            # Gather target probabilities
            if target.dim() == pred.dim() - 1:
                target_probs = torch.gather(
                    probs, 1, target.unsqueeze(1)
                ).squeeze(1)
            else:
                target_probs = (probs * target).sum(dim=1)
        else:
            # Binary: use sigmoid
            target_probs = torch.where(
                target == 1,
                torch.sigmoid(pred),
                1 - torch.sigmoid(pred)
            )
        
        # Gradient magnitude: |p - 1| for positive, |p| for negative = 1 - p_target
        g = (1 - target_probs).detach()
        
        # Flatten
        g_flat = g.reshape(-1)
        n = g_flat.numel()
        
        if mask is not None:
            mask_flat = mask.reshape(-1).bool()
            g_flat = g_flat[mask_flat]
            n = mask_flat.sum().item()
        
        # Count samples in each bin
        bin_counts = torch.zeros(self.bins, device=pred.device)
        for i in range(self.bins):
            low, high = self.edges[i], self.edges[i + 1]
            if i == self.bins - 1:
                in_bin = (g_flat >= low) & (g_flat <= high)
            else:
                in_bin = (g_flat >= low) & (g_flat < high)
            bin_counts[i] = in_bin.sum()
        
        # Apply momentum
        if self.momentum > 0:
            self.acc_sum = self.momentum * self.acc_sum + (1 - self.momentum) * bin_counts
            bin_counts = self.acc_sum
        
        # Compute weights: N / (count * num_bins)
        # Avoid division by zero
        weights_per_bin = n / (bin_counts.clamp(min=1) * self.bins)
        
        # Assign weights to each sample
        weights = torch.zeros_like(g_flat)
        for i in range(self.bins):
            low, high = self.edges[i], self.edges[i + 1]
            if i == self.bins - 1:
                in_bin = (g_flat >= low) & (g_flat <= high)
            else:
                in_bin = (g_flat >= low) & (g_flat < high)
            weights[in_bin] = weights_per_bin[i]
        
        # Reshape weights back
        if mask is not None:
            full_weights = torch.zeros(g.numel(), device=pred.device)
            full_weights[mask_flat] = weights
            weights = full_weights.reshape(g.shape)
        else:
            weights = weights.reshape(g.shape)
        
        # Compute base cross-entropy loss
        if pred.dim() > target.dim():
            loss = F.cross_entropy(pred, target, reduction="none")
        else:
            loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
        
        # Apply GHM weights
        weighted_loss = loss * weights
        
        if self.reduction == "mean":
            if mask is not None:
                return weighted_loss.sum() / mask.sum().clamp(min=1)
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss


class IoULoss(nn.Module):
    """
    Intersection over Union (Jaccard) Loss.
    
    L = 1 - IoU = 1 - |A ∩ B| / |A ∪ B|
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        reduction: ReductionMode = "mean",
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute IoU loss."""
        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
        target_flat = target.reshape(target.shape[0], target.shape[1], -1)
        
        intersection = (pred_flat * target_flat).sum(dim=-1)
        union = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = (1.0 - iou).mean(dim=1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for segmentation.
    
    Reference: "Boundary loss for highly unbalanced segmentation" (Kervadec et al., 2019)
    
    Uses distance transform to weight predictions by distance to boundary.
    """
    
    def __init__(
        self,
        reduction: ReductionMode = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: Tensor,
        dist_map: Tensor,
    ) -> Tensor:
        """
        Compute boundary loss.
        
        Args:
            pred: Soft predictions [batch, classes, H, W]
            dist_map: Signed distance transform [batch, classes, H, W]
                      (precomputed from ground truth)
        
        Returns:
            Boundary loss
        """
        # Multiply prediction by distance map and sum
        loss = (pred * dist_map).sum(dim=[2, 3]).mean(dim=1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# §12: Loss Registry & Factory
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

class LossRegistry:
    """
    Registry for loss functions with factory pattern.
    
    Usage:
        registry = LossRegistry()
        loss_fn = registry.get("focal", gamma=2.0, alpha=0.25)
    """
    
    _losses: Dict[str, type] = {
        # Cross Entropy Family
        "cross_entropy": FusedCrossEntropyLoss,
        "ce": FusedCrossEntropyLoss,
        "chunked_ce": ChunkedCrossEntropyLoss,
        "focal": FocalLoss,
        "poly": PolyLoss,
        
        # Knowledge Distillation
        "distillation": DistillationLoss,
        "kd": DistillationLoss,
        "feature_distill": FeatureDistillationLoss,
        "attention_distill": AttentionDistillationLoss,
        
        # Preference Optimization
        "dpo": DPOLoss,
        "kto": KTOLoss,
        "orpo": ORPOLoss,
        "simpo": SimPOLoss,
        "cpo": CPOLoss,
        
        # Contrastive
        "infonce": InfoNCELoss,
        "clip": CLIPLoss,
        "triplet": TripletLoss,
        "ntxent": NTXentLoss,
        "supcon": SupConLoss,
        
        # MoE
        "z_loss": ZLoss,
        "load_balance": LoadBalancingLoss,
        "expert_entropy": ExpertEntropyLoss,
        
        # Reconstruction
        "huber": HuberLoss,
        "charbonnier": CharbonnierLoss,
        "perceptual": PerceptualLoss,
        
        # Ranking
        "margin_ranking": MarginRankingLoss,
        "listmle": ListMLELoss,
        "lambdarank": LambdaRankLoss,
        
        # Segmentation
        "dice": DiceLoss,
        "tversky": TverskyLoss,
        "focal_tversky": FocalTverskyLoss,
        "ghmc": GHMCLoss,
        "iou": IoULoss,
        "boundary": BoundaryLoss,
    }
    
    @classmethod
    def register(cls, name: str, loss_class: type) -> None:
        """Register a new loss function."""
        cls._losses[name.lower()] = loss_class
    
    @classmethod
    def get(cls, name: str, **kwargs: Any) -> nn.Module:
        """Get a loss function instance by name."""
        name_lower = name.lower()
        if name_lower not in cls._losses:
            available = ", ".join(sorted(cls._losses.keys()))
            raise ValueError(f"Unknown loss '{name}'. Available: {available}")
        return cls._losses[name_lower](**kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all available loss functions."""
        return sorted(cls._losses.keys())


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# §13: Composite & Multi-Task Losses
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

class CompositeLoss(nn.Module):
    """
    Weighted combination of multiple loss functions.
    
    Example:
        loss = CompositeLoss({
            'ce': (FusedCrossEntropyLoss(), 1.0),
            'focal': (FocalLoss(gamma=2.0), 0.5),
        })
    """
    
    def __init__(
        self,
        losses: Dict[str, Tuple[nn.Module, float]],
        normalize_weights: bool = False,
    ) -> None:
        super().__init__()
        self.loss_names = list(losses.keys())
        self.loss_modules = nn.ModuleDict({
            name: loss for name, (loss, _) in losses.items()
        })
        weights = {name: w for name, (_, w) in losses.items()}
        
        if normalize_weights:
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
        
        for name, w in weights.items():
            self.register_buffer(f"weight_{name}", torch.tensor(w))
    
    def forward(self, *args: Any, **kwargs: Any) -> LossOutput:
        """Compute weighted sum of all losses."""
        total_loss = torch.tensor(0.0, device=args[0].device if args else "cpu")
        metrics: Dict[str, float] = {}
        
        for name in self.loss_names:
            loss_fn = self.loss_modules[name]
            weight = getattr(self, f"weight_{name}")
            
            result = loss_fn(*args, **kwargs)
            
            if isinstance(result, LossOutput):
                loss_value = result.loss
                for k, v in result.metrics.items():
                    metrics[f"{name}/{k}"] = v
            else:
                loss_value = result
            
            weighted = weight * loss_value
            total_loss = total_loss + weighted
            metrics[f"{name}/loss"] = loss_value.item()
            metrics[f"{name}/weighted"] = weighted.item()
        
        metrics["total_loss"] = total_loss.item()
        
        return LossOutput(loss=total_loss, metrics=metrics)


class MultiTaskLoss(nn.Module):
    """
    Multi-task learning with uncertainty weighting.
    
    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., 2018)
    
    Learns task weights automatically via homoscedastic uncertainty.
    L = Σ_i (1/(2σ_i²)) * L_i + log(σ_i)
    """
    
    def __init__(
        self,
        task_names: List[str],
        loss_fns: List[nn.Module],
        learnable_weights: bool = True,
    ) -> None:
        super().__init__()
        assert len(task_names) == len(loss_fns)
        
        self.task_names = task_names
        self.loss_fns = nn.ModuleList(loss_fns)
        
        # Log variance parameters (log σ²)
        if learnable_weights:
            self.log_vars = nn.ParameterList([
                nn.Parameter(torch.zeros(1)) for _ in task_names
            ])
        else:
            for i in range(len(task_names)):
                self.register_buffer(f"log_var_{i}", torch.zeros(1))
    
    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor],
    ) -> LossOutput:
        """
        Compute multi-task loss with learned weights.
        
        Args:
            predictions: Dict mapping task name to predictions
            targets: Dict mapping task name to targets
        
        Returns:
            LossOutput with total loss and per-task metrics
        """
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        metrics: Dict[str, float] = {}
        
        for i, name in enumerate(self.task_names):
            if name not in predictions or name not in targets:
                continue
            
            # Compute task loss
            task_loss = self.loss_fns[i](predictions[name], targets[name])
            if isinstance(task_loss, LossOutput):
                task_loss = task_loss.loss
            
            # Get log variance
            if hasattr(self, "log_vars"):
                log_var = self.log_vars[i]
            else:
                log_var = getattr(self, f"log_var_{i}")
            
            # Uncertainty weighting: (1/(2σ²)) * L + log(σ) = (1/2) * exp(-log_var) * L + (1/2) * log_var
            precision = torch.exp(-log_var)
            weighted_loss = 0.5 * precision * task_loss + 0.5 * log_var
            
            total_loss = total_loss + weighted_loss.squeeze()
            
            metrics[f"{name}/loss"] = task_loss.item()
            metrics[f"{name}/weight"] = precision.item()
            metrics[f"{name}/log_var"] = log_var.item()
        
        metrics["total_loss"] = total_loss.item()
        
        return LossOutput(loss=total_loss, metrics=metrics)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# §14: Utilities & Helpers
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

def compute_log_probs(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
    per_token: bool = False,
) -> Tensor:
    """
    Compute log probabilities of target tokens.
    
    Utility for preference optimization losses.
    
    Args:
        logits: Model logits [batch, seq, vocab]
        labels: Target labels [batch, seq]
        ignore_index: Token to ignore
        per_token: If True, return per-token log probs; else sum per sequence
    
    Returns:
        Log probabilities [batch, seq] or [batch]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather target log probs
    labels_clamped = labels.clamp(min=0)
    target_log_probs = torch.gather(
        log_probs, dim=-1, index=labels_clamped.unsqueeze(-1)
    ).squeeze(-1)
    
    # Mask ignored tokens
    mask = labels != ignore_index
    target_log_probs = target_log_probs * mask.float()
    
    if per_token:
        return target_log_probs
    else:
        # Sum log probs per sequence
        return target_log_probs.sum(dim=-1)


def get_sequence_lengths(
    labels: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Compute number of valid tokens per sequence."""
    return (labels != ignore_index).sum(dim=-1)


@lru_cache(maxsize=32)
def create_label_smoothing_matrix(
    vocab_size: int,
    smoothing: float,
    device: torch.device,
) -> Tensor:
    """
    Create label smoothing distribution matrix (cached).
    
    For frequent vocab sizes, caches the smoothing pattern.
    """
    smooth_val = smoothing / vocab_size
    confidence = 1.0 - smoothing
    return torch.full(
        (vocab_size,), smooth_val, device=device
    ) + torch.eye(vocab_size, device=device) * (confidence - smooth_val)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Module Exports
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Configuration
    "LossConfig",
    "LossOutput",
    
    # Cross Entropy
    "FusedCrossEntropyLoss",
    "ChunkedCrossEntropyLoss",
    "FocalLoss",
    "PolyLoss",
    
    # Knowledge Distillation
    "DistillationLoss",
    "FeatureDistillationLoss",
    "AttentionDistillationLoss",
    
    # Preference Optimization
    "DPOLoss",
    "KTOLoss",
    "ORPOLoss",
    "SimPOLoss",
    "CPOLoss",
    
    # Contrastive
    "InfoNCELoss",
    "CLIPLoss",
    "TripletLoss",
    "NTXentLoss",
    "SupConLoss",
    
    # MoE
    "ZLoss",
    "LoadBalancingLoss",
    "ExpertEntropyLoss",
    
    # Reconstruction
    "HuberLoss",
    "CharbonnierLoss",
    "PerceptualLoss",
    
    # Ranking
    "MarginRankingLoss",
    "ListMLELoss",
    "LambdaRankLoss",
    
    # Segmentation
    "DiceLoss",
    "TverskyLoss",
    "FocalTverskyLoss",
    "GHMCLoss",
    "IoULoss",
    "BoundaryLoss",
    
    # Composite
    "CompositeLoss",
    "MultiTaskLoss",
    
    # Registry
    "LossRegistry",
    
    # Utilities
    "compute_log_probs",
    "get_sequence_lengths",
]
        
        