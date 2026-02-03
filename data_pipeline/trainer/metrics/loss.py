# ════════════════════════════════════════════════════════════════════════════════
# SOTA Training Metrics - Loss Tracking
# ════════════════════════════════════════════════════════════════════════════════
# Comprehensive loss tracking with EMA, running averages, and distributed sync.
#
# Features:
#   - Running average loss
#   - Exponential Moving Average (EMA)
#   - Perplexity computation (exp(loss))
#   - Token-weighted loss accumulation
#   - Distributed all-reduce support
#
# Hardware Support:
#   - NVIDIA: A100, H100, H200, B100, B200 (CUDA/NCCL)
#   - AMD: MI300X, MI325X (ROCm/RCCL)
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

# ════════════════════════════════════════════════════════════════════════════════
# Loss Tracker - Core Class
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class LossTracker:
    """
    SOTA loss tracking with exponential moving average and running averages.
    
    Designed for LLM training where losses are token-weighted and need
    proper aggregation across gradient accumulation steps and distributed ranks.
    
    Example:
        >>> tracker = LossTracker(ema_decay=0.99)
        >>> tracker.update(loss=2.5, num_tokens=512)
        >>> tracker.update(loss=2.3, num_tokens=480)
        >>> print(f"Avg: {tracker.compute_avg():.4f}, PPL: {tracker.compute_ppl():.2f}")
    
    Attributes:
        ema_decay: Decay factor for EMA (default: 0.99 for smooth curves)
        clip_loss: Maximum loss value to prevent exp overflow (default: 100.0)
    """
    
    ema_decay: float = 0.99
    clip_loss: float = 100.0
    
    # ───────────────────────────────────────────────────────────────────────────
    # Internal State
    # ───────────────────────────────────────────────────────────────────────────
    _total_loss: float = field(default=0.0, repr=False)
    _total_tokens: int = field(default=0, repr=False)
    _total_samples: int = field(default=0, repr=False)
    _ema_loss: Optional[float] = field(default=None, repr=False)
    _step_count: int = field(default=0, repr=False)
    _history: List[float] = field(default_factory=list, repr=False)
    _last_loss: float = field(default=0.0, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize internal state."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulated values."""
        self._total_loss = 0.0
        self._total_tokens = 0
        self._total_samples = 0
        self._ema_loss = None
        self._step_count = 0
        self._history = []
        self._last_loss = 0.0
    
    def update(
        self,
        loss: Union[float, Tensor],
        num_tokens: int = 1,
        num_samples: int = 1,
        accumulation_steps: int = 1,
    ) -> None:
        """
        Update loss tracker with new loss value.
        
        Args:
            loss: Loss value (scalar tensor or float)
            num_tokens: Number of tokens in batch (for token-weighted avg)
            num_samples: Number of samples in batch
            accumulation_steps: Gradient accumulation divisor applied to loss
        
        Note:
            If loss was already divided by accumulation_steps, pass 1.
        """
        # Convert tensor to float
        if isinstance(loss, Tensor):
            loss = loss.detach().item()
        
        # Undo accumulation division if needed
        actual_loss = loss * accumulation_steps
        
        # Update running totals (token-weighted)
        self._total_loss += actual_loss * num_tokens
        self._total_tokens += num_tokens
        self._total_samples += num_samples
        self._step_count += 1
        self._last_loss = actual_loss
        
        # Update EMA
        if self._ema_loss is None:
            self._ema_loss = actual_loss
        else:
            self._ema_loss = (
                self.ema_decay * self._ema_loss + 
                (1 - self.ema_decay) * actual_loss
            )
        
        # Store in history (capped at 1000 for memory)
        if len(self._history) < 1000:
            self._history.append(actual_loss)
    
    def update_from_logits(
        self,
        logits: Tensor,
        labels: Tensor,
        ignore_index: int = -100,
    ) -> float:
        """
        Compute and update loss from logits and labels.
        
        Args:
            logits: Model output logits [B, T, V] or [B*T, V]
            labels: Target labels [B, T] or [B*T]
            ignore_index: Label index to ignore (default: -100)
        
        Returns:
            Computed loss value
        """
        # Flatten if needed
        if logits.dim() == 3:
            # Shift for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            logits = shift_logits.view(-1, shift_logits.size(-1))
            labels = shift_labels.view(-1)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits, labels, 
            ignore_index=ignore_index, 
            reduction="sum"
        )
        
        # Count valid tokens
        num_tokens = (labels != ignore_index).sum().item()
        
        if num_tokens > 0:
            avg_loss = loss.item() / num_tokens
            # Ensure proper scaling for update
            self.update(avg_loss, num_tokens=num_tokens)
            return avg_loss
        
        return 0.0
    
    def compute_avg(self) -> float:
        """
        Compute token-weighted average loss.
        
        Returns:
            Average loss across all tokens
        """
        if self._total_tokens == 0:
            return 0.0
        return self._total_loss / self._total_tokens
    
    def compute_ema(self) -> float:
        """
        Get exponential moving average loss.
        
        Returns:
            EMA loss (0.0 if no updates)
        """
        return self._ema_loss if self._ema_loss is not None else 0.0
    
    def compute_ppl(self) -> float:
        """
        Compute perplexity from average loss.
        
        Returns:
            Perplexity (exp(avg_loss)), capped at exp(clip_loss)
        """
        avg_loss = self.compute_avg()
        clipped = min(avg_loss, self.clip_loss)
        return math.exp(clipped)
    
    def compute_ppl_ema(self) -> float:
        """
        Compute perplexity from EMA loss.
        
        Returns:
            Perplexity (exp(ema_loss)), capped at exp(clip_loss)
        """
        ema_loss = self.compute_ema()
        clipped = min(ema_loss, self.clip_loss)
        return math.exp(clipped)
    
    @property
    def last_loss(self) -> float:
        """Get the most recent loss value."""
        return self._last_loss
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens processed."""
        return self._total_tokens
    
    @property
    def total_samples(self) -> int:
        """Get total samples processed."""
        return self._total_samples
    
    @property
    def step_count(self) -> int:
        """Get number of update calls."""
        return self._step_count
    
    def get_history(self, last_n: Optional[int] = None) -> List[float]:
        """
        Get loss history.
        
        Args:
            last_n: Return only last N values (None = all)
        
        Returns:
            List of historical loss values
        """
        if last_n is None:
            return self._history.copy()
        return self._history[-last_n:]
    
    def compute_variance(self) -> float:
        """
        Compute variance of loss history.
        
        Returns:
            Loss variance (0.0 if insufficient data)
        """
        if len(self._history) < 2:
            return 0.0
        mean = sum(self._history) / len(self._history)
        variance = sum((x - mean) ** 2 for x in self._history) / len(self._history)
        return variance
    
    def sync(self, process_group: Optional["torch.distributed.ProcessGroup"] = None) -> None:
        """
        Synchronize loss across distributed ranks via all-reduce.
        
        Args:
            process_group: Process group for reduction (None = default)
        """
        if not torch.distributed.is_initialized():
            return
        
        # Pack values for single all-reduce
        buffer = torch.tensor(
            [self._total_loss, float(self._total_tokens), float(self._total_samples)],
            dtype=torch.float64,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        torch.distributed.all_reduce(
            buffer, 
            op=torch.distributed.ReduceOp.SUM,
            group=process_group,
        )
        
        self._total_loss = buffer[0].item()
        self._total_tokens = int(buffer[1].item())
        self._total_samples = int(buffer[2].item())
    
    def state_dict(self) -> Dict[str, Union[float, int, List[float]]]:
        """Get serializable state for checkpointing."""
        return {
            "total_loss": self._total_loss,
            "total_tokens": self._total_tokens,
            "total_samples": self._total_samples,
            "ema_loss": self._ema_loss,
            "step_count": self._step_count,
            "history": self._history,
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """Load state from checkpoint."""
        self._total_loss = state.get("total_loss", 0.0)
        self._total_tokens = state.get("total_tokens", 0)
        self._total_samples = state.get("total_samples", 0)
        self._ema_loss = state.get("ema_loss")
        self._step_count = state.get("step_count", 0)
        self._history = state.get("history", [])


# ════════════════════════════════════════════════════════════════════════════════
# Accuracy Tracker - Token-Level Accuracy
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class AccuracyTracker:
    """
    Token-level accuracy tracker for language modeling.
    
    Tracks correct predictions vs total predictions, properly ignoring
    masked tokens (label == -100).
    
    Example:
        >>> tracker = AccuracyTracker()
        >>> tracker.update_from_logits(logits, labels)
        >>> print(f"Accuracy: {tracker.compute():.2%}")
    """
    
    _correct: int = field(default=0, repr=False)
    _total: int = field(default=0, repr=False)
    
    def __post_init__(self) -> None:
        self.reset()
    
    def reset(self) -> None:
        """Reset counters."""
        self._correct = 0
        self._total = 0
    
    def update(self, correct: int, total: int) -> None:
        """
        Update with correct/total counts.
        
        Args:
            correct: Number of correct predictions
            total: Total number of predictions
        """
        self._correct += correct
        self._total += total
    
    def update_from_logits(
        self,
        logits: Tensor,
        labels: Tensor,
        ignore_index: int = -100,
    ) -> float:
        """
        Compute and update accuracy from logits and labels.
        
        Args:
            logits: Model output logits [B, T, V] or [B*T, V]
            labels: Target labels [B, T] or [B*T]
            ignore_index: Label index to ignore (default: -100)
        
        Returns:
            Step accuracy
        """
        # Flatten if needed
        if logits.dim() == 3:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            logits = shift_logits.view(-1, shift_logits.size(-1))
            labels = shift_labels.view(-1)
        
        # Get predictions
        preds = logits.argmax(dim=-1)
        
        # Mask ignored tokens
        mask = labels != ignore_index
        
        # Count correct
        correct = ((preds == labels) & mask).sum().item()
        total = mask.sum().item()
        
        self.update(int(correct), int(total))
        
        return correct / max(1, total)
    
    def compute(self) -> float:
        """
        Compute overall accuracy.
        
        Returns:
            Accuracy as float in [0, 1]
        """
        if self._total == 0:
            return 0.0
        return self._correct / self._total
    
    @property
    def correct(self) -> int:
        """Get correct count."""
        return self._correct
    
    @property
    def total(self) -> int:
        """Get total count."""
        return self._total
    
    def sync(self, process_group: Optional["torch.distributed.ProcessGroup"] = None) -> None:
        """Synchronize across distributed ranks."""
        if not torch.distributed.is_initialized():
            return
        
        buffer = torch.tensor(
            [self._correct, self._total],
            dtype=torch.int64,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        torch.distributed.all_reduce(
            buffer,
            op=torch.distributed.ReduceOp.SUM,
            group=process_group,
        )
        
        self._correct = buffer[0].item()
        self._total = buffer[1].item()


# ════════════════════════════════════════════════════════════════════════════════
# Exports
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    "LossTracker",
    "AccuracyTracker",
]
