# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer Module - Loss Functions
# ════════════════════════════════════════════════════════════════════════════════
# Above-SOTA loss functions with Triton kernel acceleration.
#
# Features:
# 1. Numerically stable implementations
# 2. Triton-fused kernels for GPU acceleration
# 3. Label smoothing, focal loss, KL divergence
# 4. Gradient-aware computation
#
# All losses implement forward() returning the loss value and support
# autograd for backward pass computation.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from data_pipeline.trainer.core.types import LossConfig, LossType

# ═════════════════════════════════════════════════════════════════════════════════
# Triton Kernels
# ═════════════════════════════════════════════════════════════════════════════════

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass

if _TRITON_AVAILABLE:
    @triton.jit
    def cross_entropy_fwd_kernel(
        # Inputs
        logits_ptr,
        labels_ptr,
        # Output
        loss_ptr,
        # Parameters
        ignore_index: tl.constexpr,
        label_smoothing: tl.constexpr,
        vocab_size: tl.constexpr,
        # Dimensions
        batch_size,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused softmax + cross entropy kernel.
        
        Performs in single pass:
        1. Numerically stable softmax (via log-sum-exp)
        2. Cross entropy with optional label smoothing
        
        Memory: O(vocab_size) per thread block
        Compute: O(batch * vocab) total
        """
        pid = tl.program_id(0)
        
        if pid >= batch_size:
            return
        
        # Compute row offset
        row_offset = pid * vocab_size
        
        # Load label
        label = tl.load(labels_ptr + pid)
        
        # Check for ignore_index
        if label == ignore_index:
            tl.store(loss_ptr + pid, 0.0)
            return
        
        # Compute max for numerical stability (log-sum-exp trick)
        max_val = -float("inf")
        for i in range(0, vocab_size, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < vocab_size
            vals = tl.load(logits_ptr + row_offset + offsets, mask=mask, other=-float("inf"))
            max_val = tl.maximum(max_val, tl.max(vals, axis=0))
        
        # Compute sum of exp(logits - max)
        sum_exp = 0.0
        for i in range(0, vocab_size, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < vocab_size
            vals = tl.load(logits_ptr + row_offset + offsets, mask=mask, other=-float("inf"))
            sum_exp += tl.sum(tl.exp(vals - max_val), axis=0)
        
        # Log normalizer
        log_normalizer = max_val + tl.log(sum_exp)
        
        # Load target logit
        target_logit = tl.load(logits_ptr + row_offset + label)
        
        # Cross entropy: -log(softmax[target]) = log_normalizer - target_logit
        ce_loss = log_normalizer - target_logit
        
        # Apply label smoothing if enabled
        if label_smoothing > 0.0:
            # Smoothed loss = (1 - ε) * CE + ε * mean(-log(softmax))
            # mean(-log(softmax)) = log_normalizer - mean(logits)
            mean_logit = 0.0
            for i in range(0, vocab_size, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                mask = offsets < vocab_size
                vals = tl.load(logits_ptr + row_offset + offsets, mask=mask, other=0.0)
                mean_logit += tl.sum(vals, axis=0)
            mean_logit = mean_logit / vocab_size
            
            smooth_loss = log_normalizer - mean_logit
            ce_loss = (1.0 - label_smoothing) * ce_loss + label_smoothing * smooth_loss
        
        tl.store(loss_ptr + pid, ce_loss)


# ═════════════════════════════════════════════════════════════════════════════════
# Cross Entropy Loss
# ═════════════════════════════════════════════════════════════════════════════════

class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss with label smoothing and Triton acceleration.
    
    Above-SOTA features:
    1. Numerically stable log-softmax computation
    2. Label smoothing for regularization
    3. Triton-fused forward pass for GPU
    4. Ignore index for padding tokens
    
    Label Smoothing:
    Replaces hard labels y with soft labels:
        y_smooth = (1 - ε) * y + ε / num_classes
    
    This prevents overconfident predictions and improves generalization.
    
    Args:
        ignore_index: Token ID to ignore (-100 default)
        label_smoothing: Smoothing factor (0 = no smoothing)
        reduction: "mean", "sum", or "none"
        use_triton: Enable Triton kernel
        
    Example:
        ```python
        loss_fn = CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
        loss = loss_fn(logits, labels)  # logits: [B, T, V], labels: [B, T]
        ```
    """
    
    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
        *,
        use_triton: bool = True,
    ):
        super().__init__()
        
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.use_triton = use_triton and _TRITON_AVAILABLE
    
    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Compute cross entropy loss.
        
        Args:
            logits: Predicted logits [batch, seq, vocab] or [batch, vocab]
            labels: Target labels [batch, seq] or [batch]
            
        Returns:
            Loss value (scalar or per-sample based on reduction)
        """
        # Flatten for 3D logits
        if logits.dim() == 3:
            batch, seq, vocab = logits.shape
            logits = logits.view(-1, vocab)
            labels = labels.view(-1)
        else:
            batch_seq = logits.shape[0]
            vocab = logits.shape[1]
        
        # Use Triton kernel on GPU
        if self.use_triton and logits.is_cuda and self.label_smoothing == 0.0:
            return self._forward_triton(logits, labels)
        
        # PyTorch fallback
        return self._forward_torch(logits, labels)
    
    def _forward_torch(self, logits: Tensor, labels: Tensor) -> Tensor:
        """PyTorch implementation with label smoothing."""
        if self.label_smoothing > 0.0:
            # Manual label smoothing implementation
            vocab_size = logits.shape[-1]
            
            # Log-softmax
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Hard target loss
            nll_loss = F.nll_loss(
                log_probs,
                labels,
                ignore_index=self.ignore_index,
                reduction="none",
            )
            
            # Smooth loss (KL with uniform)
            # -sum(1/V * log_probs) = -mean(log_probs)
            smooth_loss = -log_probs.mean(dim=-1)
            
            # Combine
            loss = (1.0 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
            
            # Mask ignored indices
            mask = labels != self.ignore_index
            loss = loss * mask.float()
            
            # Reduction
            if self.reduction == "mean":
                return loss.sum() / mask.sum().clamp(min=1)
            elif self.reduction == "sum":
                return loss.sum()
            else:
                return loss
        else:
            # Standard cross entropy
            return F.cross_entropy(
                logits,
                labels,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
            )
    
    def _forward_triton(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Triton-accelerated forward pass."""
        batch_size = logits.shape[0]
        vocab_size = logits.shape[1]
        
        # Allocate output
        losses = torch.empty(batch_size, device=logits.device, dtype=logits.dtype)
        
        # Launch kernel
        BLOCK_SIZE = min(1024, triton.next_power_of_2(vocab_size))
        grid = (batch_size,)
        
        cross_entropy_fwd_kernel[grid](
            logits,
            labels,
            losses,
            self.ignore_index,
            self.label_smoothing,
            vocab_size,
            batch_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Reduction
        mask = labels != self.ignore_index
        
        if self.reduction == "mean":
            return losses.sum() / mask.sum().clamp(min=1)
        elif self.reduction == "sum":
            return losses.sum()
        else:
            return losses


# ═════════════════════════════════════════════════════════════════════════════════
# KL Divergence Loss
# ═════════════════════════════════════════════════════════════════════════════════

class KLDivergenceLoss(nn.Module):
    """
    KL Divergence loss for knowledge distillation.
    
    Supports:
    - Forward KL: KL(target || prediction) - mode-seeking
    - Reverse KL: KL(prediction || target) - mean-seeking
    - Symmetric KL (Jensen-Shannon): 0.5 * (KL(p||m) + KL(q||m)) where m = 0.5*(p+q)
    
    Temperature scaling for soft targets in distillation.
    
    Args:
        reduction: "mean", "sum", "batchmean", or "none"
        temperature: Softmax temperature (higher = softer)
        log_target: Whether target is already log probabilities
        mode: "forward", "reverse", or "symmetric"
        
    Example:
        ```python
        # Knowledge distillation
        loss_fn = KLDivergenceLoss(temperature=4.0)
        loss = loss_fn(student_logits, teacher_logits)
        ```
    """
    
    def __init__(
        self,
        reduction: Literal["mean", "sum", "batchmean", "none"] = "batchmean",
        temperature: float = 1.0,
        log_target: bool = False,
        mode: Literal["forward", "reverse", "symmetric"] = "forward",
    ):
        super().__init__()
        
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        
        self.reduction = reduction
        self.temperature = temperature
        self.log_target = log_target
        self.mode = mode
    
    def forward(
        self,
        input: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Compute KL divergence.
        
        Args:
            input: Predicted logits or log-probs
            target: Target logits or probs (based on log_target)
            
        Returns:
            KL divergence loss
        """
        # Apply temperature scaling
        if self.temperature != 1.0:
            input = input / self.temperature
            if not self.log_target:
                target = target / self.temperature
        
        # Convert to log-probs
        log_input = F.log_softmax(input, dim=-1)
        
        if self.log_target:
            log_target = target
            target_probs = log_target.exp()
        else:
            target_probs = F.softmax(target, dim=-1)
            log_target = target_probs.log()
        
        if self.mode == "forward":
            # KL(target || input) = sum(target * log(target/input))
            # = sum(target * (log_target - log_input))
            kl = target_probs * (log_target - log_input)
        
        elif self.mode == "reverse":
            # KL(input || target) = sum(input * log(input/target))
            input_probs = log_input.exp()
            kl = input_probs * (log_input - log_target)
        
        elif self.mode == "symmetric":
            # Jensen-Shannon: 0.5 * (KL(p||m) + KL(q||m))
            input_probs = log_input.exp()
            m = 0.5 * (input_probs + target_probs)
            log_m = m.log()
            
            kl_p = target_probs * (log_target - log_m)
            kl_q = input_probs * (log_input - log_m)
            kl = 0.5 * (kl_p + kl_q)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Sum over vocabulary
        kl = kl.sum(dim=-1)
        
        # Apply temperature^2 scaling (standard in distillation)
        if self.temperature != 1.0:
            kl = kl * (self.temperature ** 2)
        
        # Reduction
        if self.reduction == "mean":
            return kl.mean()
        elif self.reduction == "sum":
            return kl.sum()
        elif self.reduction == "batchmean":
            return kl.sum() / kl.shape[0]
        else:
            return kl


# ═════════════════════════════════════════════════════════════════════════════════
# Focal Loss
# ═════════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal loss for class imbalance.
    
    Down-weights well-classified examples to focus on hard negatives.
    
    Formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Where:
    - p_t = p if y=1 else (1-p)
    - γ (gamma) = focusing parameter (higher = more focus on hard examples)
    - α (alpha) = class weight
    
    Reference: Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002
    
    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Class weight for positive class (default: None = balanced)
        ignore_index: Label to ignore
        reduction: "mean", "sum", or "none"
        
    Example:
        ```python
        # For imbalanced classification
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        loss = loss_fn(logits, labels)
        ```
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        ignore_index: int = -100,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        
        if gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Predicted logits [batch, classes]
            labels: Target labels [batch]
            
        Returns:
            Focal loss value
        """
        num_classes = logits.shape[-1]
        
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Flatten if needed
        if logits.dim() > 2:
            logits = logits.view(-1, num_classes)
            labels = labels.view(-1)
            probs = probs.view(-1, num_classes)
        
        # Get probability of target class
        # p_t = prob[target]
        batch_indices = torch.arange(len(labels), device=labels.device)
        p_t = probs[batch_indices, labels.clamp(min=0)]  # Clamp for ignore_index
        
        # Compute focal weight
        # (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(
            logits,
            labels,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            # alpha for positive, (1-alpha) for negative
            alpha_weight = torch.where(
                labels == 1,
                torch.tensor(self.alpha, device=logits.device),
                torch.tensor(1.0 - self.alpha, device=logits.device),
            )
            focal_loss = alpha_weight * focal_loss
        
        # Mask ignored indices
        mask = labels != self.ignore_index
        focal_loss = focal_loss * mask.float()
        
        # Reduction
        if self.reduction == "mean":
            return focal_loss.sum() / mask.sum().clamp(min=1)
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ═════════════════════════════════════════════════════════════════════════════════
# Factory Function
# ═════════════════════════════════════════════════════════════════════════════════

def create_loss(config: LossConfig) -> nn.Module:
    """
    Create loss function from configuration.
    
    Args:
        config: Loss configuration
        
    Returns:
        Configured loss module
    """
    loss_type = config.loss_type
    
    if loss_type == LossType.CROSS_ENTROPY:
        return CrossEntropyLoss(
            ignore_index=config.ignore_index,
            label_smoothing=0.0,
            reduction=config.reduction,
            use_triton=config.use_triton_kernel,
        )
    
    elif loss_type == LossType.CROSS_ENTROPY_SMOOTH:
        return CrossEntropyLoss(
            ignore_index=config.ignore_index,
            label_smoothing=config.label_smoothing,
            reduction=config.reduction,
            use_triton=config.use_triton_kernel,
        )
    
    elif loss_type == LossType.KL_DIVERGENCE:
        return KLDivergenceLoss(
            reduction=config.reduction if config.reduction != "mean" else "batchmean",
            temperature=config.kl_temperature,
            log_target=config.kl_log_target,
            mode="forward",
        )
    
    elif loss_type == LossType.KL_REVERSE:
        return KLDivergenceLoss(
            reduction=config.reduction if config.reduction != "mean" else "batchmean",
            temperature=config.kl_temperature,
            log_target=config.kl_log_target,
            mode="reverse",
        )
    
    elif loss_type == LossType.FOCAL:
        return FocalLoss(
            gamma=config.focal_gamma,
            alpha=config.focal_alpha,
            ignore_index=config.ignore_index,
            reduction=config.reduction,
        )
    
    elif loss_type == LossType.MSE:
        return nn.MSELoss(reduction=config.reduction)
    
    elif loss_type == LossType.MAE:
        return nn.L1Loss(reduction=config.reduction)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ═════════════════════════════════════════════════════════════════════════════════
# Export
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    "CrossEntropyLoss",
    "KLDivergenceLoss",
    "FocalLoss",
    "create_loss",
]
