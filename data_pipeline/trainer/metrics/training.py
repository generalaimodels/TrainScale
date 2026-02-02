# ════════════════════════════════════════════════════════════════════════════════
# SOTA Training Metrics - Unified Training Metrics Aggregator
# ════════════════════════════════════════════════════════════════════════════════
# Complete training metrics collection combining loss, accuracy, throughput,
# and gradient tracking with distributed synchronization.
#
# Features:
#   - Single update call for all metrics
#   - Automatic distributed reduction
#   - Logging-ready output (TensorBoard, WandB compatible)
#   - Checkpoint save/load support
#   - Per-epoch and per-step tracking
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .loss import LossTracker, AccuracyTracker
from .throughput import ThroughputTracker
from .gradient import GradientTracker

# ════════════════════════════════════════════════════════════════════════════════
# Training Metrics - Main Aggregator
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingMetrics:
    """
    Unified training metrics aggregator for LLM training.
    
    Combines loss, accuracy, throughput, and gradient tracking into a single
    interface with automatic distributed synchronization.
    
    Example:
        >>> metrics = TrainingMetrics(distributed=True)
        >>> metrics.start()
        >>> 
        >>> for step, batch in enumerate(dataloader):
        ...     outputs = model(**batch)
        ...     loss = outputs.loss
        ...     loss.backward()
        ...     
        ...     metrics.update_step(
        ...         loss=loss,
        ...         logits=outputs.logits,
        ...         labels=batch["labels"],
        ...         model=model,
        ...         batch_size=batch["input_ids"].size(0),
        ...         num_tokens=batch["input_ids"].numel(),
        ...     )
        ...     
        ...     if step % 10 == 0:
        ...         results = metrics.compute()
        ...         print(f"Step {step}: loss={results['loss']:.4f}, ppl={results['ppl']:.2f}")
    
    Attributes:
        distributed: Whether to sync metrics across ranks
        ema_decay: EMA decay for loss/gradient tracking
        log_prefix: Prefix for log dictionary keys
    """
    
    distributed: bool = False
    ema_decay: float = 0.99
    log_prefix: str = "train"
    model_params: Optional[int] = None  # For MFU calculation
    
    # ───────────────────────────────────────────────────────────────────────────
    # Sub-trackers
    # ───────────────────────────────────────────────────────────────────────────
    loss: LossTracker = field(default_factory=lambda: LossTracker())
    accuracy: AccuracyTracker = field(default_factory=lambda: AccuracyTracker())
    throughput: ThroughputTracker = field(default_factory=lambda: ThroughputTracker())
    gradient: GradientTracker = field(default_factory=lambda: GradientTracker())
    
    # ───────────────────────────────────────────────────────────────────────────
    # Step/Epoch Tracking
    # ───────────────────────────────────────────────────────────────────────────
    _global_step: int = field(default=0, repr=False)
    _epoch: int = field(default=0, repr=False)
    _epoch_step: int = field(default=0, repr=False)
    _lr: float = field(default=0.0, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize sub-trackers with shared EMA decay."""
        self.loss = LossTracker(ema_decay=self.ema_decay)
        self.accuracy = AccuracyTracker()
        self.throughput = ThroughputTracker()
        self.gradient = GradientTracker(ema_decay=self.ema_decay)
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics (call at epoch start)."""
        self.loss.reset()
        self.accuracy.reset()
        self.throughput.reset()
        self.gradient.reset()
        self._epoch_step = 0
    
    def reset_all(self) -> None:
        """Reset everything including global step."""
        self.reset()
        self._global_step = 0
        self._epoch = 0
    
    def start(self) -> None:
        """Start timing (call at training start)."""
        self.throughput.start()
    
    def new_epoch(self, epoch: int) -> None:
        """
        Signal start of new epoch.
        
        Args:
            epoch: Epoch number (0-indexed)
        """
        self._epoch = epoch
        self.reset()
    
    def update_step(
        self,
        loss: Union[float, Tensor],
        logits: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        model: Optional[nn.Module] = None,
        parameters: Optional[Any] = None,
        batch_size: int = 1,
        num_tokens: int = 0,
        lr: Optional[float] = None,
        grad_norm: Optional[float] = None,
        accumulation_steps: int = 1,
    ) -> None:
        """
        Update all metrics for a single training step.
        
        Args:
            loss: Loss value (tensor or float)
            logits: Model output logits [B, T, V] (optional, for accuracy)
            labels: Target labels [B, T] (optional, for accuracy)
            model: Model for gradient tracking (optional)
            parameters: Model parameters iterator (alternative to model)
            batch_size: Number of samples in batch
            num_tokens: Number of tokens in batch
            lr: Current learning rate (optional)
            grad_norm: Pre-computed gradient norm (optional)
            accumulation_steps: Gradient accumulation divisor
        """
        # Convert loss to float
        loss_val = loss.detach().item() if isinstance(loss, Tensor) else loss
        
        # Update loss tracker
        self.loss.update(
            loss=loss_val,
            num_tokens=max(num_tokens, 1),
            num_samples=batch_size,
            accumulation_steps=accumulation_steps,
        )
        
        # Update accuracy if logits/labels provided
        if logits is not None and labels is not None:
            self.accuracy.update_from_logits(logits, labels)
        
        # Update throughput
        self.throughput.update(samples=batch_size, tokens=num_tokens)
        
        # Update gradient tracker
        if grad_norm is not None:
            # Use pre-computed norm
            self.gradient._total_norm = grad_norm
            self.gradient._step_count += 1
            if self.gradient._ema_norm is None:
                self.gradient._ema_norm = grad_norm
            else:
                self.gradient._ema_norm = (
                    self.gradient.ema_decay * self.gradient._ema_norm +
                    (1 - self.gradient.ema_decay) * grad_norm
                )
        elif model is not None:
            self.gradient.update_from_model(model)
        elif parameters is not None:
            self.gradient.update(parameters)
        
        # Update learning rate
        if lr is not None:
            self._lr = lr
        
        # Increment step counters
        self._global_step += 1
        self._epoch_step += 1
    
    def sync(self, process_group: Optional["torch.distributed.ProcessGroup"] = None) -> None:
        """
        Synchronize all metrics across distributed ranks.
        
        Args:
            process_group: Process group for reduction (None = default)
        """
        if not self.distributed or not torch.distributed.is_initialized():
            return
        
        self.loss.sync(process_group)
        self.accuracy.sync(process_group)
        self.throughput.sync(process_group)
        # Note: Gradient norms are typically local, not synced
    
    def compute(self, sync: bool = True) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            sync: Whether to synchronize across ranks first
        
        Returns:
            Dict with all computed metrics
        """
        if sync:
            self.sync()
        
        avg_loss = self.loss.compute_avg()
        ema_loss = self.loss.compute_ema()
        ppl = self.loss.compute_ppl()
        acc = self.accuracy.compute()
        tokens_per_sec = self.throughput.tokens_per_sec()
        grad_norm = self.gradient.compute_norm()
        
        # Compute MFU if model params known
        mfu = 0.0
        if self.model_params is not None:
            mfu = self.throughput.compute_mfu(model_params=self.model_params)
        
        return {
            "loss": avg_loss,
            "loss_ema": ema_loss,
            "ppl": ppl,
            "accuracy": acc,
            "tokens_per_sec": tokens_per_sec,
            "samples_per_sec": self.throughput.samples_per_sec(),
            "grad_norm": grad_norm,
            "grad_norm_ema": self.gradient.compute_ema_norm(),
            "mfu": mfu,
            "lr": self._lr,
            "global_step": self._global_step,
            "epoch": self._epoch,
            "epoch_step": self._epoch_step,
            "total_tokens": self.throughput.total_tokens,
            "elapsed_seconds": self.throughput.elapsed_seconds(),
        }
    
    def log_dict(self, sync: bool = True) -> Dict[str, float]:
        """
        Get metrics as dictionary for logging (TensorBoard/WandB format).
        
        Args:
            sync: Whether to synchronize across ranks first
        
        Returns:
            Dict with prefixed metric names
        """
        metrics = self.compute(sync=sync)
        prefix = f"{self.log_prefix}/" if self.log_prefix else ""
        
        return {
            f"{prefix}loss": metrics["loss"],
            f"{prefix}loss_ema": metrics["loss_ema"],
            f"{prefix}ppl": metrics["ppl"],
            f"{prefix}accuracy": metrics["accuracy"],
            f"{prefix}tokens_per_sec": metrics["tokens_per_sec"],
            f"{prefix}samples_per_sec": metrics["samples_per_sec"],
            f"{prefix}grad_norm": metrics["grad_norm"],
            f"{prefix}mfu": metrics["mfu"],
            f"{prefix}lr": metrics["lr"],
            f"{prefix}step": float(metrics["global_step"]),
        }
    
    def format_log_string(self, sync: bool = True) -> str:
        """
        Format metrics as human-readable log string.
        
        Args:
            sync: Whether to synchronize across ranks first
        
        Returns:
            Formatted string for logging
        """
        m = self.compute(sync=sync)
        
        parts = [
            f"step={m['global_step']}",
            f"loss={m['loss']:.4f}",
            f"ppl={m['ppl']:.2f}",
            f"acc={m['accuracy']:.2%}",
            f"tok/s={m['tokens_per_sec']:,.0f}",
            f"grad={m['grad_norm']:.3f}",
            f"lr={m['lr']:.2e}",
        ]
        
        if m['mfu'] > 0:
            parts.append(f"mfu={m['mfu']:.1%}")
        
        return " | ".join(parts)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get serializable state for checkpointing."""
        return {
            "loss": self.loss.state_dict(),
            "throughput": self.throughput.state_dict(),
            "gradient": self.gradient.state_dict(),
            "global_step": self._global_step,
            "epoch": self._epoch,
            "lr": self._lr,
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        if "loss" in state:
            self.loss.load_state_dict(state["loss"])
        if "throughput" in state:
            self.throughput.load_state_dict(state["throughput"])
        if "gradient" in state:
            self.gradient.load_state_dict(state["gradient"])
        self._global_step = state.get("global_step", 0)
        self._epoch = state.get("epoch", 0)
        self._lr = state.get("lr", 0.0)


# ════════════════════════════════════════════════════════════════════════════════
# Convenience Factory
# ════════════════════════════════════════════════════════════════════════════════

def create_training_metrics(
    distributed: bool = False,
    model: Optional[nn.Module] = None,
    log_prefix: str = "train",
) -> TrainingMetrics:
    """
    Create TrainingMetrics with optional model parameter count.
    
    Args:
        distributed: Whether to sync across ranks
        model: Model for parameter counting (for MFU)
        log_prefix: Prefix for log keys
    
    Returns:
        Configured TrainingMetrics instance
    """
    model_params = None
    if model is not None:
        model_params = sum(p.numel() for p in model.parameters())
    
    return TrainingMetrics(
        distributed=distributed,
        log_prefix=log_prefix,
        model_params=model_params,
    )


# ════════════════════════════════════════════════════════════════════════════════
# Exports
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    "TrainingMetrics",
    "create_training_metrics",
]
