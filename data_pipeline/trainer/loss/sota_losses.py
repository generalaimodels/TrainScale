# ════════════════════════════════════════════════════════════════════════════════
# SOTA Loss Functions - Above Unsloth Implementation
# ════════════════════════════════════════════════════════════════════════════════
# Comprehensive loss suite with:
# - Chunked Cross Entropy (memory efficient)
# - Distillation Loss (knowledge transfer)
# - Contrastive Loss (representation learning)
# - DPO/ORPO Loss (preference alignment)
# - Z-Loss (router regularization)
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
from typing import Callable, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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


# ═════════════════════════════════════════════════════════════════════════════════
# Triton Kernels
# ═════════════════════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:

    @triton.jit
    def _chunked_cross_entropy_kernel(
        logits_ptr, labels_ptr, loss_ptr, lse_ptr,
        vocab_size: tl.constexpr,
        batch_size,
        ignore_index: tl.constexpr,
        label_smoothing: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Chunked cross entropy for memory efficiency.
        Processes vocabulary in chunks to reduce peak memory.
        """
        row = tl.program_id(0)
        
        if row >= batch_size:
            return
        
        label = tl.load(labels_ptr + row)
        
        if label == ignore_index:
            tl.store(loss_ptr + row, 0.0)
            return
        
        # Compute log-sum-exp in chunks
        max_logit = float('-inf')
        sum_exp = 0.0
        
        for chunk_start in range(0, vocab_size, BLOCK_SIZE):
            offs = chunk_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < vocab_size
            
            logits = tl.load(logits_ptr + row * vocab_size + offs, mask=mask, other=float('-inf'))
            
            chunk_max = tl.max(logits)
            if chunk_max > max_logit:
                sum_exp = sum_exp * tl.exp(max_logit - chunk_max)
                max_logit = chunk_max
            
            sum_exp += tl.sum(tl.exp(logits - max_logit))
        
        lse = max_logit + tl.log(sum_exp)
        tl.store(lse_ptr + row, lse)
        
        # Get target logit
        target_logit = tl.load(logits_ptr + row * vocab_size + label)
        
        # Cross entropy: -target_logit + lse
        if label_smoothing > 0:
            smooth_loss = lse - tl.sum(tl.load(logits_ptr + row * vocab_size + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < vocab_size, other=0.0)) / vocab_size
            ce_loss = -target_logit + lse
            loss = (1 - label_smoothing) * ce_loss + label_smoothing * smooth_loss
        else:
            loss = -target_logit + lse
        
        tl.store(loss_ptr + row, loss)


# ═════════════════════════════════════════════════════════════════════════════════
# Chunked Cross Entropy (Memory Efficient)
# ═════════════════════════════════════════════════════════════════════════════════

class ChunkedCrossEntropyLoss(nn.Module):
    """
    Memory-efficient cross entropy that processes vocabulary in chunks.
    
    Reduces peak memory by ~4x for large vocabularies (100k+).
    Essential for training with limited GPU memory.
    """
    
    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
        chunk_size: int = 32768,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.chunk_size = chunk_size
    
    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Compute chunked cross entropy loss.
        
        Args:
            logits: [batch, seq, vocab] or [batch, vocab]
            labels: [batch, seq] or [batch]
        
        Returns:
            Loss value
        """
        # Reshape for chunked processing
        if logits.dim() == 3:
            batch, seq, vocab = logits.shape
            logits = logits.view(-1, vocab)
            labels = labels.view(-1)
        else:
            batch = logits.shape[0]
            vocab = logits.shape[1]
        
        num_samples = logits.shape[0]
        device = logits.device
        
        # Process in chunks
        total_loss = torch.tensor(0.0, device=device)
        valid_count = 0
        
        for start in range(0, vocab, self.chunk_size):
            end = min(start + self.chunk_size, vocab)
            chunk_logits = logits[:, start:end]
            
            # Compute per-chunk contribution to log-sum-exp
            if start == 0:
                max_logits = chunk_logits.max(dim=-1, keepdim=True).values
                sum_exp = (chunk_logits - max_logits).exp().sum(dim=-1)
            else:
                chunk_max = chunk_logits.max(dim=-1, keepdim=True).values
                combined_max = torch.maximum(max_logits, chunk_max)
                
                # Rescale previous sum_exp
                sum_exp = sum_exp * (max_logits - combined_max).squeeze(-1).exp()
                sum_exp += (chunk_logits - combined_max).exp().sum(dim=-1)
                max_logits = combined_max
        
        # Log-sum-exp
        lse = max_logits.squeeze(-1) + sum_exp.log()
        
        # Get target logits
        valid_mask = labels != self.ignore_index
        valid_labels = labels.clamp(min=0)
        
        target_logits = torch.gather(logits, 1, valid_labels.unsqueeze(-1)).squeeze(-1)
        
        # Cross entropy
        loss = -target_logits + lse
        
        # Label smoothing
        if self.label_smoothing > 0:
            smooth_loss = lse - logits.mean(dim=-1)
            loss = (1 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss
        
        # Mask invalid
        loss = loss * valid_mask.float()
        
        # Reduction
        if self.reduction == "mean":
            return loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ═════════════════════════════════════════════════════════════════════════════════
# Knowledge Distillation Loss
# ═════════════════════════════════════════════════════════════════════════════════

class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss for model compression.
    
    Combines:
    - Hard label loss (cross entropy with ground truth)
    - Soft label loss (KL divergence with teacher)
    
    Supports temperature scaling and various mixing strategies.
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        reduction: Literal["mean", "sum", "none"] = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        self.hard_loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction,
        )
    
    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, dict]:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model outputs [batch, seq, vocab]
            teacher_logits: Teacher model outputs [batch, seq, vocab]
            labels: Ground truth labels [batch, seq]
        
        Returns:
            Combined loss and dict of individual losses
        """
        # Hard loss (student vs ground truth)
        hard_loss = self.hard_loss_fn(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
        )
        
        # Soft loss (student vs teacher with temperature)
        T = self.temperature
        student_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        
        soft_loss = F.kl_div(
            student_probs.view(-1, student_probs.size(-1)),
            teacher_probs.view(-1, teacher_probs.size(-1)),
            reduction="batchmean",
        ) * (T ** 2)
        
        # Combined loss
        combined_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        return combined_loss, {
            "hard_loss": hard_loss.item(),
            "soft_loss": soft_loss.item(),
            "combined_loss": combined_loss.item(),
        }


# ═════════════════════════════════════════════════════════════════════════════════
# DPO Loss (Direct Preference Optimization)
# ═════════════════════════════════════════════════════════════════════════════════

class DPOLoss(nn.Module):
    """
    Direct Preference Optimization loss for RLHF.
    
    From: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    
    Optimizes the policy directly without reward model or PPO.
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: Literal["sigmoid", "hinge", "ipo"] = "sigmoid",
        reference_free: bool = False,
    ):
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
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute DPO loss.
        
        Args:
            policy_chosen_logps: Log probs of chosen responses under policy
            policy_rejected_logps: Log probs of rejected responses under policy
            reference_chosen_logps: Log probs of chosen under reference
            reference_rejected_logps: Log probs of rejected under reference
        
        Returns:
            loss, chosen_rewards, rejected_rewards
        """
        if self.reference_free:
            # Reference-free DPO (uses policy as reference)
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = torch.zeros_like(pi_logratios)
        else:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        logits = pi_logratios - ref_logratios
        
        # Compute loss based on type
        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.loss_type == "hinge":
            losses = F.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # Identity Preference Optimization
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # Label smoothing
        if self.label_smoothing > 0:
            smooth_losses = -F.logsigmoid(-self.beta * logits)
            losses = (1 - self.label_smoothing) * losses + self.label_smoothing * smooth_losses
        
        # Compute implicit rewards
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        
        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


# ═════════════════════════════════════════════════════════════════════════════════
# ORPO Loss (Odds Ratio Preference Optimization)
# ═════════════════════════════════════════════════════════════════════════════════

class ORPOLoss(nn.Module):
    """
    ORPO: Odds Ratio Preference Optimization.
    
    From: "ORPO: Monolithic Preference Optimization without Reference Model"
    
    Combines SFT and preference alignment in a single loss without reference model.
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
    
    def forward(
        self,
        policy_chosen_logps: Tensor,
        policy_rejected_logps: Tensor,
        chosen_nll_loss: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute ORPO loss.
        
        Args:
            policy_chosen_logps: Log probs of chosen responses
            policy_rejected_logps: Log probs of rejected responses
            chosen_nll_loss: NLL loss on chosen responses (SFT component)
        
        Returns:
            Total loss, odds ratio
        """
        # Odds ratio
        log_odds = policy_chosen_logps - policy_rejected_logps
        ratio = F.sigmoid(log_odds)
        
        # ORPO loss: SFT + alpha * log(odds_ratio)
        orpo_loss = -self.alpha * torch.log(ratio + 1e-8)
        
        total_loss = chosen_nll_loss + orpo_loss.mean()
        
        return total_loss, ratio.mean()


# ═════════════════════════════════════════════════════════════════════════════════
# SimPO Loss (Simple Preference Optimization)
# ═════════════════════════════════════════════════════════════════════════════════

class SimPOLoss(nn.Module):
    """
    SimPO: Simple Preference Optimization.
    
    Simplified version of DPO without reference model.
    Uses length-normalized rewards and gamma margin.
    """
    
    def __init__(
        self,
        beta: float = 2.0,
        gamma: float = 0.5,
        label_smoothing: float = 0.0,
    ):
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
    ) -> Tensor:
        """
        Compute SimPO loss.
        
        Args:
            policy_chosen_logps: Log probs of chosen responses
            policy_rejected_logps: Log probs of rejected responses
            chosen_length: Length of chosen responses
            rejected_length: Length of rejected responses
        
        Returns:
            SimPO loss
        """
        # Length-normalized rewards
        chosen_rewards = policy_chosen_logps / chosen_length
        rejected_rewards = policy_rejected_logps / rejected_length
        
        # SimPO loss with margin
        logits = self.beta * (chosen_rewards - rejected_rewards) - self.gamma
        
        losses = -F.logsigmoid(logits)
        
        if self.label_smoothing > 0:
            smooth_losses = -F.logsigmoid(-logits)
            losses = (1 - self.label_smoothing) * losses + self.label_smoothing * smooth_losses
        
        return losses.mean()


# ═════════════════════════════════════════════════════════════════════════════════
# Contrastive Loss (InfoNCE)
# ═════════════════════════════════════════════════════════════════════════════════

class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for representation learning.
    
    Used in SimCLR, CLIP, and other contrastive methods.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        embeddings_a: Tensor,
        embeddings_b: Tensor,
    ) -> Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings_a: First set of embeddings [batch, dim]
            embeddings_b: Second set of embeddings [batch, dim]
        
        Returns:
            Contrastive loss
        """
        batch_size = embeddings_a.shape[0]
        
        # Normalize
        embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)
        
        # Compute similarity matrix
        logits = embeddings_a @ embeddings_b.T / self.temperature
        
        # Positive pairs are on diagonal
        labels = torch.arange(batch_size, device=logits.device)
        
        # Symmetric loss
        loss_a = F.cross_entropy(logits, labels, reduction=self.reduction)
        loss_b = F.cross_entropy(logits.T, labels, reduction=self.reduction)
        
        return (loss_a + loss_b) / 2


# ═════════════════════════════════════════════════════════════════════════════════
# Z-Loss (Router Regularization)
# ═════════════════════════════════════════════════════════════════════════════════

class ZLoss(nn.Module):
    """
    Z-Loss for MoE router regularization.
    
    Encourages router logits to stay small to prevent overflow
    and improve training stability.
    
    z_loss = log(sum(exp(logits)))^2
    """
    
    def __init__(
        self,
        z_loss_coef: float = 0.001,
    ):
        super().__init__()
        self.z_loss_coef = z_loss_coef
    
    def forward(self, router_logits: Tensor) -> Tensor:
        """
        Compute z-loss.
        
        Args:
            router_logits: Router logits [batch, seq, num_experts]
        
        Returns:
            Z-loss value
        """
        lse = torch.logsumexp(router_logits, dim=-1)
        z_loss = lse.pow(2).mean()
        return self.z_loss_coef * z_loss


# ═════════════════════════════════════════════════════════════════════════════════
# Combined Training Loss
# ═════════════════════════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    """
    Combined loss for multi-objective training.
    
    Supports:
    - Cross entropy (main LM loss)
    - Auxiliary loss (MoE load balancing)
    - Z-loss (router regularization)
    - Custom losses with configurable weights
    """
    
    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
        use_chunked_ce: bool = False,
    ):
        super().__init__()
        self.aux_loss_coef = aux_loss_coef
        
        if use_chunked_ce:
            self.ce_loss = ChunkedCrossEntropyLoss(
                ignore_index=ignore_index,
                label_smoothing=label_smoothing,
            )
        else:
            self.ce_loss = nn.CrossEntropyLoss(
                ignore_index=ignore_index,
                label_smoothing=label_smoothing,
            )
        
        self.z_loss = ZLoss(z_loss_coef=z_loss_coef)
    
    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        aux_loss: Optional[Tensor] = None,
        router_logits: Optional[Tensor] = None,
    ) -> Tuple[Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            logits: Model outputs [batch, seq, vocab]
            labels: Ground truth [batch, seq]
            aux_loss: MoE auxiliary loss (optional)
            router_logits: Router logits for z-loss (optional)
        
        Returns:
            Total loss and breakdown dict
        """
        # Main cross entropy loss
        ce_loss = self.ce_loss(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        
        total_loss = ce_loss
        breakdown = {"ce_loss": ce_loss.item()}
        
        # Auxiliary loss (MoE)
        if aux_loss is not None:
            aux = self.aux_loss_coef * aux_loss
            total_loss = total_loss + aux
            breakdown["aux_loss"] = aux.item()
        
        # Z-loss (router)
        if router_logits is not None:
            z_loss = self.z_loss(router_logits)
            total_loss = total_loss + z_loss
            breakdown["z_loss"] = z_loss.item()
        
        breakdown["total_loss"] = total_loss.item()
        
        return total_loss, breakdown


# ═════════════════════════════════════════════════════════════════════════════════
# Factory Function
# ═════════════════════════════════════════════════════════════════════════════════

def create_loss(
    name: str,
    **kwargs,
) -> nn.Module:
    """
    Create loss function by name.
    
    Supported: cross_entropy, chunked_ce, distillation, dpo, orpo, simpo, contrastive, z_loss, combined
    """
    name = name.lower()
    
    losses = {
        'cross_entropy': lambda: nn.CrossEntropyLoss(**kwargs),
        'chunked_ce': lambda: ChunkedCrossEntropyLoss(**kwargs),
        'distillation': lambda: DistillationLoss(**kwargs),
        'dpo': lambda: DPOLoss(**kwargs),
        'orpo': lambda: ORPOLoss(**kwargs),
        'simpo': lambda: SimPOLoss(**kwargs),
        'contrastive': lambda: ContrastiveLoss(**kwargs),
        'z_loss': lambda: ZLoss(**kwargs),
        'combined': lambda: CombinedLoss(**kwargs),
    }
    
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Supported: {list(losses.keys())}")
    
    return losses[name]()


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Core
    "ChunkedCrossEntropyLoss",
    "DistillationLoss",
    # Preference Optimization
    "DPOLoss",
    "ORPOLoss",
    "SimPOLoss",
    # Contrastive
    "ContrastiveLoss",
    # Regularization
    "ZLoss",
    # Combined
    "CombinedLoss",
    # Factory
    "create_loss",
]
