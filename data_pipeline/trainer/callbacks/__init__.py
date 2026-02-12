# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer Module - Callback System
# ════════════════════════════════════════════════════════════════════════════════
# Extensible callback framework for training customization.
#
# Features:
# 1. Event-driven architecture with before/after hooks
# 2. Built-in callbacks: EarlyStopping, Checkpointing, Logging
# 3. Custom callback support via inheritance
# 4. Priority-based callback ordering
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import abc
import json
import logging
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

import torch
from torch import Tensor

from data_pipeline.trainer.core.types import (
    CheckpointConfig,
    CheckpointStrategy,
    LoggingBackend,
    LoggingConfig,
    StepMetrics,
    TrainerEvent,
    TrainingState,
)

if TYPE_CHECKING:
    from data_pipeline.trainer.base import Trainer

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════════
# Callback Context
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class CallbackContext:
    """
    Context passed to callback hooks.
    
    Contains current training state and references to trainer components.
    """
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    loss: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0
    is_training: bool = True
    model: Optional[torch.nn.Module] = None
    optimizer: Optional[Any] = None
    scheduler: Optional[Any] = None
    
    # Control flags
    should_stop: bool = False
    skip_step: bool = False


# ═════════════════════════════════════════════════════════════════════════════════
# Base Callback
# ═════════════════════════════════════════════════════════════════════════════════

class Callback(abc.ABC):
    """
    Abstract base class for training callbacks.
    
    Callbacks can hook into various training events:
    - on_init_end: After trainer initialization
    - on_train_begin/end: Before/after training loop
    - on_epoch_begin/end: Before/after each epoch
    - on_step_begin/end: Before/after each training step
    - on_evaluate: After evaluation
    - on_save: Before checkpoint save
    - on_load: After checkpoint load
    - on_log: When logging is triggered
    
    Priority determines callback execution order (lower = earlier).
    """
    
    priority: int = 100  # Default priority
    
    def __init__(self, trainer: Optional["Trainer"] = None):
        """Initialize callback with optional trainer reference."""
        self.trainer = trainer
    
    def set_trainer(self, trainer: "Trainer") -> None:
        """Set trainer reference."""
        self.trainer = trainer
    
    def on_init_end(self, ctx: CallbackContext) -> None:
        """Called after trainer initialization."""
        pass
    
    def on_train_begin(self, ctx: CallbackContext) -> None:
        """Called before training starts."""
        pass
    
    def on_train_end(self, ctx: CallbackContext) -> None:
        """Called after training ends."""
        pass
    
    def on_epoch_begin(self, ctx: CallbackContext) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, ctx: CallbackContext) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_step_begin(self, ctx: CallbackContext) -> None:
        """Called before each training step."""
        pass
    
    def on_step_end(self, ctx: CallbackContext) -> None:
        """Called after each training step."""
        pass
    
    def on_evaluate(self, ctx: CallbackContext) -> None:
        """Called after evaluation."""
        pass
    
    def on_save(self, ctx: CallbackContext) -> None:
        """Called before saving checkpoint."""
        pass
    
    def on_load(self, ctx: CallbackContext) -> None:
        """Called after loading checkpoint."""
        pass
    
    def on_log(self, ctx: CallbackContext) -> None:
        """Called when logging is triggered."""
        pass


# ═════════════════════════════════════════════════════════════════════════════════
# Callback Handler
# ═════════════════════════════════════════════════════════════════════════════════

class CallbackHandler:
    """
    Manages a collection of callbacks.
    
    Handles callback execution order based on priority and
    provides convenience methods for triggering events.
    """
    
    def __init__(self, callbacks: Optional[Sequence[Callback]] = None):
        """Initialize with optional list of callbacks."""
        self.callbacks: List[Callback] = list(callbacks) if callbacks else []
        self._sorted = False
    
    def add_callback(self, callback: Callback) -> None:
        """Add a callback to the handler."""
        self.callbacks.append(callback)
        self._sorted = False
    
    def remove_callback(self, callback_type: Type[Callback]) -> None:
        """Remove all callbacks of a given type."""
        self.callbacks = [cb for cb in self.callbacks if not isinstance(cb, callback_type)]
    
    def _sort_callbacks(self) -> None:
        """Sort callbacks by priority."""
        if not self._sorted:
            self.callbacks.sort(key=lambda cb: cb.priority)
            self._sorted = True
    
    def set_trainer(self, trainer: "Trainer") -> None:
        """Set trainer reference for all callbacks."""
        for cb in self.callbacks:
            cb.set_trainer(trainer)
    
    def trigger(self, event: str, ctx: CallbackContext) -> CallbackContext:
        """
        Trigger an event on all callbacks.
        
        Args:
            event: Event name (e.g., "on_step_end")
            ctx: Callback context
            
        Returns:
            Updated context after all callbacks
        """
        self._sort_callbacks()
        
        for cb in self.callbacks:
            handler = getattr(cb, event, None)
            if handler is not None:
                handler(ctx)
                
                # Check for early stopping
                if ctx.should_stop:
                    break
        
        return ctx


# ═════════════════════════════════════════════════════════════════════════════════
# Built-in Callbacks
# ═════════════════════════════════════════════════════════════════════════════════

class EarlyStoppingCallback(Callback):
    """
    Early stopping callback based on monitored metric.
    
    Stops training if metric doesn't improve for patience epochs.
    
    Args:
        monitor: Metric name to monitor (e.g., "eval_loss")
        patience: Number of epochs without improvement before stopping
        min_delta: Minimum change to qualify as improvement
        mode: "min" (lower is better) or "max" (higher is better)
        restore_best: Restore model to best checkpoint
        
    Example:
        ```python
        early_stopping = EarlyStoppingCallback(
            monitor="eval_loss",
            patience=3,
            mode="min",
        )
        trainer.add_callback(early_stopping)
        ```
    """
    
    priority = 50  # Run after most callbacks
    
    def __init__(
        self,
        monitor: str = "eval_loss",
        patience: int = 3,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
        restore_best: bool = True,
    ):
        super().__init__()
        
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        # State
        self.best_value: Optional[float] = None
        self.best_epoch: int = 0
        self.wait: int = 0
        self.stopped_epoch: int = 0
        self.best_state: Optional[Dict[str, Any]] = None
    
    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self.best_value is None:
            return True
        
        if self.mode == "min":
            return current < self.best_value - self.min_delta
        else:
            return current > self.best_value + self.min_delta
    
    def on_epoch_end(self, ctx: CallbackContext) -> None:
        """Check for improvement and update state."""
        current = ctx.metrics.get(self.monitor)
        
        if current is None:
            logger.warning(f"EarlyStopping: metric '{self.monitor}' not found")
            return
        
        if self._is_improvement(current):
            self.best_value = current
            self.best_epoch = ctx.epoch
            self.wait = 0
            
            # Save best state
            if self.restore_best and ctx.model is not None:
                self.best_state = {
                    k: v.cpu().clone() for k, v in ctx.model.state_dict().items()
                }
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                ctx.should_stop = True
                self.stopped_epoch = ctx.epoch
                
                logger.info(
                    f"Early stopping triggered at epoch {ctx.epoch}. "
                    f"Best {self.monitor}: {self.best_value:.6f} at epoch {self.best_epoch}"
                )
    
    def on_train_end(self, ctx: CallbackContext) -> None:
        """Restore best model if requested."""
        if self.restore_best and self.best_state is not None and ctx.model is not None:
            ctx.model.load_state_dict(self.best_state)
            logger.info(f"Restored model to best epoch {self.best_epoch}")


class CheckpointCallback(Callback):
    """
    Checkpoint saving callback.
    
    Saves model checkpoints based on strategy (steps, epochs, best).
    
    Args:
        save_dir: Directory to save checkpoints
        config: Checkpoint configuration
        
    Example:
        ```python
        checkpoint = CheckpointCallback(
            save_dir="./checkpoints",
            config=CheckpointConfig(
                strategy=CheckpointStrategy.STEPS,
                save_steps=1000,
                save_total_limit=3,
            ),
        )
        ```
    """
    
    priority = 80  # Run late
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        config: CheckpointConfig,
    ):
        super().__init__()
        
        self.save_dir = Path(save_dir)
        self.config = config
        
        # State
        self.best_metric: Optional[float] = None
        self.saved_checkpoints: List[Path] = []
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _should_save(self, ctx: CallbackContext) -> bool:
        """Determine if checkpoint should be saved."""
        if self.config.strategy == CheckpointStrategy.STEPS:
            return ctx.step > 0 and ctx.step % self.config.save_steps == 0
        
        elif self.config.strategy == CheckpointStrategy.EPOCHS:
            # Called at epoch end
            return True
        
        elif self.config.strategy == CheckpointStrategy.BEST:
            monitor = self.config.metric_for_best
            current = ctx.metrics.get(monitor)
            
            if current is None:
                return False
            
            is_best = self.best_metric is None
            if not is_best:
                if self.config.greater_is_better:
                    is_best = current > self.best_metric
                else:
                    is_best = current < self.best_metric
            
            if is_best:
                self.best_metric = current
            
            return is_best
        
        return False
    
    def _save_checkpoint(self, ctx: CallbackContext, name: str) -> Path:
        """Save a checkpoint."""
        checkpoint_dir = self.save_dir / name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        if ctx.model is not None:
            model_path = checkpoint_dir / "model.pt"
            
            if self.config.save_safetensors:
                try:
                    from safetensors.torch import save_file
                    save_file(ctx.model.state_dict(), checkpoint_dir / "model.safetensors")
                except ImportError:
                    torch.save(ctx.model.state_dict(), model_path)
            else:
                torch.save(ctx.model.state_dict(), model_path)
        
        # Save optimizer
        if ctx.optimizer is not None:
            torch.save(ctx.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        
        # Save scheduler
        if ctx.scheduler is not None and hasattr(ctx.scheduler, "state_dict"):
            torch.save(ctx.scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        
        # Save training state
        state = {
            "epoch": ctx.epoch,
            "step": ctx.step,
            "total_steps": ctx.total_steps,
            "metrics": ctx.metrics,
            "best_metric": self.best_metric,
        }
        with open(checkpoint_dir / "trainer_state.json", "w") as f:
            json.dump(state, f, indent=2)
        
        return checkpoint_dir
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if limit exceeded."""
        if self.config.save_total_limit is None:
            return
        
        while len(self.saved_checkpoints) > self.config.save_total_limit:
            oldest = self.saved_checkpoints.pop(0)
            if oldest.exists():
                shutil.rmtree(oldest)
                logger.info(f"Removed old checkpoint: {oldest}")
    
    def on_step_end(self, ctx: CallbackContext) -> None:
        """Save checkpoint on step end if strategy is STEPS."""
        if self.config.strategy == CheckpointStrategy.STEPS:
            if self._should_save(ctx):
                name = f"checkpoint-{ctx.step}"
                path = self._save_checkpoint(ctx, name)
                self.saved_checkpoints.append(path)
                self._cleanup_old_checkpoints()
                logger.info(f"Saved checkpoint: {path}")
    
    def on_epoch_end(self, ctx: CallbackContext) -> None:
        """Save checkpoint on epoch end if strategy is EPOCHS or BEST."""
        if self.config.strategy in (CheckpointStrategy.EPOCHS, CheckpointStrategy.BEST):
            if self._should_save(ctx):
                name = f"checkpoint-epoch-{ctx.epoch}"
                path = self._save_checkpoint(ctx, name)
                self.saved_checkpoints.append(path)
                self._cleanup_old_checkpoints()
                logger.info(f"Saved checkpoint: {path}")


class LoggingCallback(Callback):
    """
    Training logging callback.
    
    Logs metrics to console, file, TensorBoard, and/or W&B.
    
    Args:
        config: Logging configuration
        
    Example:
        ```python
        logging = LoggingCallback(
            config=LoggingConfig(
                backends=[LoggingBackend.CONSOLE, LoggingBackend.TENSORBOARD],
                log_steps=10,
            ),
        )
        ```
    """
    
    priority = 90  # Run last
    
    def __init__(self, config: LoggingConfig):
        super().__init__()
        
        self.config = config
        self._writers: Dict[str, Any] = {}
        self._start_time: float = 0.0
        self._last_log_step: int = 0
        
        # Initialize backends
        self._init_backends()
    
    def _init_backends(self) -> None:
        """Initialize logging backends."""
        for backend in self.config.backends:
            if backend == LoggingBackend.TENSORBOARD:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self._writers["tensorboard"] = SummaryWriter(
                        log_dir=self.config.log_dir
                    )
                except ImportError:
                    logger.warning("TensorBoard not available")
            
            elif backend == LoggingBackend.WANDB:
                try:
                    import wandb
                    if not wandb.run:
                        wandb.init(
                            project=self.config.project_name,
                            name=self.config.run_name,
                            config=self.config.model_dump(),
                        )
                    self._writers["wandb"] = wandb
                except ImportError:
                    logger.warning("W&B not available")
    
    def on_train_begin(self, ctx: CallbackContext) -> None:
        """Record training start time."""
        self._start_time = time.time()
    
    def on_step_end(self, ctx: CallbackContext) -> None:
        """Log metrics at configured intervals."""
        if ctx.step - self._last_log_step < self.config.log_steps:
            return
        
        self._last_log_step = ctx.step
        
        # Compute throughput
        elapsed = time.time() - self._start_time
        steps_per_sec = ctx.step / elapsed if elapsed > 0 else 0
        
        # Build log message
        metrics = {
            "loss": ctx.loss,
            "learning_rate": ctx.learning_rate,
            "steps_per_sec": steps_per_sec,
            **ctx.metrics,
        }
        
        # Console logging
        if LoggingBackend.CONSOLE in self.config.backends:
            msg = f"Step {ctx.step}/{ctx.total_steps}"
            for k, v in metrics.items():
                if v is not None:
                    msg += f" | {k}: {v:.6f}"
            logger.info(msg)
        
        # TensorBoard logging
        if "tensorboard" in self._writers:
            for k, v in metrics.items():
                if v is not None:
                    self._writers["tensorboard"].add_scalar(k, v, ctx.step)
        
        # W&B logging
        if "wandb" in self._writers:
            self._writers["wandb"].log(
                {k: v for k, v in metrics.items() if v is not None},
                step=ctx.step,
            )
    
    def on_train_end(self, ctx: CallbackContext) -> None:
        """Close logging backends."""
        if "tensorboard" in self._writers:
            self._writers["tensorboard"].close()
        
        if "wandb" in self._writers:
            self._writers["wandb"].finish()


class ProgressCallback(Callback):
    """
    Training progress display callback (tqdm-style).
    
    Displays progress bar with metrics.
    """
    
    priority = 95  # Run very late
    
    def __init__(self, refresh_rate: int = 10):
        super().__init__()
        
        self.refresh_rate = refresh_rate
        self._pbar = None
    
    def on_train_begin(self, ctx: CallbackContext) -> None:
        """Initialize progress bar."""
        try:
            from tqdm import tqdm
            self._pbar = tqdm(
                total=ctx.total_steps,
                desc="Training",
                unit="step",
            )
        except ImportError:
            pass
    
    def on_step_end(self, ctx: CallbackContext) -> None:
        """Update progress bar."""
        if self._pbar is not None:
            self._pbar.update(1)
            
            if ctx.step % self.refresh_rate == 0:
                postfix = {"loss": f"{ctx.loss:.4f}" if ctx.loss else "N/A"}
                if ctx.learning_rate:
                    postfix["lr"] = f"{ctx.learning_rate:.2e}"
                self._pbar.set_postfix(postfix)
    
    def on_train_end(self, ctx: CallbackContext) -> None:
        """Close progress bar."""
        if self._pbar is not None:
            self._pbar.close()


# ═════════════════════════════════════════════════════════════════════════════════
# Export
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Context
    "CallbackContext",
    # Base
    "Callback",
    "CallbackHandler",
    # Built-in
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "LoggingCallback",
    "ProgressCallback",
]
