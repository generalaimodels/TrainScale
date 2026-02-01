# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer Module - Base Trainer
# ════════════════════════════════════════════════════════════════════════════════
# Above-SOTA trainer with automated training loop, mixed precision, 
# gradient accumulation, and distributed training support.
#
# Features:
# 1. Automatic batch accumulation for arbitrary effective batch sizes
# 2. Mixed precision training (FP16/BF16) with loss scaling
# 3. Gradient clipping (norm and value)
# 4. Distributed training support (DDP, FSDP)
# 5. Checkpointing with auto-resume
# 6. Comprehensive callback system
# 7. Integration with custom optimizers/schedulers/losses
#
# Architecture:
# - Uses TrainingArguments for YAML-based configuration
# - Compatible with data_pipeline for data loading
# - Extensible via callbacks and subclassing
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import gc
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from data_pipeline.trainer.core.types import (
    DeviceType,
    GradientInfo,
    Precision,
    StepMetrics,
    TrainingState,
)
from data_pipeline.trainer.core.config import TrainingArguments
from data_pipeline.trainer.core.errors import (
    ConfigurationError,
    GradientOverflowError,
    TrainingLoopError,
)
from data_pipeline.trainer.optimizers import (
    AdamW,
    LAMB,
    AdaFactor,
    clip_grad_norm_,
    clip_grad_value_,
)
from data_pipeline.trainer.schedulers import (
    create_scheduler,
    BaseScheduler,
)
from data_pipeline.trainer.loss import (
    CrossEntropyLoss,
    create_loss,
)
from data_pipeline.trainer.callbacks import (
    Callback,
    CallbackContext,
    CallbackHandler,
    CheckpointCallback,
    LoggingCallback,
    ProgressCallback,
)

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════════
# Type Aliases
# ═════════════════════════════════════════════════════════════════════════════════

BatchType = Dict[str, Tensor]
MetricsType = Dict[str, float]

# ═════════════════════════════════════════════════════════════════════════════════
# Training Output
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainOutput:
    """Output from training run."""
    global_step: int
    training_loss: float
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"TrainOutput(global_step={self.global_step}, "
            f"training_loss={self.training_loss:.6f}, "
            f"metrics={self.metrics})"
        )


@dataclass
class PredictionOutput:
    """Output from prediction run."""
    predictions: Tensor
    label_ids: Optional[Tensor] = None
    metrics: Dict[str, float] = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════════
# Base Trainer
# ═════════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Above-SOTA trainer for model training.
    
    Provides automated training loop with:
    - Mixed precision (FP16/BF16) with automatic loss scaling
    - Gradient accumulation for large effective batch sizes
    - Gradient clipping (norm and value)
    - Distributed training (DDP, FSDP)
    - Checkpointing with auto-resume
    - Comprehensive callback system
    
    Args:
        model: Model to train
        args: Training arguments (from YAML or code)
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        optimizers: Optional (optimizer, scheduler) tuple
        loss_fn: Optional loss function (auto-created from args)
        callbacks: Optional list of callbacks
        
    Example:
        ```python
        # Load config from YAML
        args = load_training_config("config.yaml")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
        )
        
        # Train
        result = trainer.train()
        print(f"Training loss: {result.training_loss}")
        ```
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        *,
        optimizers: Optional[Tuple[Any, Optional[BaseScheduler]]] = None,
        loss_fn: Optional[nn.Module] = None,
        callbacks: Optional[Sequence[Callback]] = None,
        compute_metrics: Optional[Callable[[Tensor, Tensor], MetricsType]] = None,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        
        # Device setup
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Precision setup
        self.scaler = self._setup_precision()
        
        # Optimizer and scheduler
        if optimizers is not None:
            self.optimizer, self.scheduler = optimizers
        else:
            self.optimizer = None
            self.scheduler = None
        
        # Loss function
        self.loss_fn = loss_fn or self._create_loss_fn()
        
        # Callbacks
        self.callback_handler = CallbackHandler(callbacks or [])
        self._add_default_callbacks()
        self.callback_handler.set_trainer(self)
        
        # Training state
        self.state = TrainingState()
        self.global_step = 0
        self._total_loss = 0.0
        
        # Context for callbacks
        self._ctx = CallbackContext()
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Setup Methods
    # ─────────────────────────────────────────────────────────────────────────────
    
    def _setup_device(self) -> torch.device:
        """Configure training device."""
        device_type = self.args.device
        
        if device_type == DeviceType.AUTO:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        
        return torch.device(device_type.value)
    
    def _setup_precision(self) -> Optional[torch.cuda.amp.GradScaler]:
        """Configure mixed precision and loss scaler."""
        precision = self.args.precision
        
        if precision in (Precision.MIXED_FP16, Precision.FP16):
            return torch.cuda.amp.GradScaler()
        
        # BF16 doesn't need loss scaling
        return None
    
    def _get_autocast_context(self):
        """Get autocast context for current precision."""
        precision = self.args.precision
        
        if precision == Precision.FP32:
            return torch.autocast(device_type=self.device.type, enabled=False)
        elif precision in (Precision.MIXED_FP16, Precision.FP16):
            return torch.autocast(device_type=self.device.type, dtype=torch.float16)
        elif precision in (Precision.MIXED_BF16, Precision.BF16):
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        else:
            return torch.autocast(device_type=self.device.type, enabled=False)
    
    def _create_optimizer(self) -> Any:
        """Create optimizer from configuration."""
        opt_config = self.args.optimizer
        
        params = self.model.parameters()
        
        if opt_config.optimizer_type.value == "adamw":
            return AdamW(
                params,
                lr=opt_config.learning_rate,
                betas=(opt_config.beta1, opt_config.beta2),
                eps=opt_config.epsilon,
                weight_decay=opt_config.weight_decay,
                use_triton=opt_config.use_triton_kernels,
            )
        elif opt_config.optimizer_type.value == "lamb":
            return LAMB(
                params,
                lr=opt_config.learning_rate,
                betas=(opt_config.beta1, opt_config.beta2),
                eps=opt_config.epsilon,
                weight_decay=opt_config.weight_decay,
                trust_ratio_clip=opt_config.lamb_trust_ratio_clip,
                use_triton=opt_config.use_triton_kernels,
            )
        elif opt_config.optimizer_type.value == "adafactor":
            return AdaFactor(
                params,
                lr=opt_config.learning_rate,
                weight_decay=opt_config.weight_decay,
                relative_step=opt_config.adafactor_relative_step,
                warmup_init=opt_config.adafactor_warmup_init,
            )
        else:
            # Fallback to standard AdamW
            return AdamW(
                params,
                lr=opt_config.learning_rate,
                weight_decay=opt_config.weight_decay,
            )
    
    def _create_scheduler(self, num_training_steps: int) -> BaseScheduler:
        """Create scheduler from configuration."""
        if self.optimizer is None:
            raise ConfigurationError("Optimizer must be created before scheduler")
        
        return create_scheduler(
            self.optimizer,
            self.args.scheduler,
            num_training_steps,
        )
    
    def _create_loss_fn(self) -> nn.Module:
        """Create loss function from configuration."""
        return create_loss(self.args.loss)
    
    def _add_default_callbacks(self) -> None:
        """Add default callbacks if enabled in config."""
        # Progress bar
        self.callback_handler.add_callback(ProgressCallback())
        
        # Logging
        if self.args.logging.backends:
            self.callback_handler.add_callback(LoggingCallback(self.args.logging))
        
        # Checkpointing
        if self.args.checkpoint.strategy.value != "no":
            self.callback_handler.add_callback(
                CheckpointCallback(self.args.output_dir, self.args.checkpoint)
            )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Data Loading
    # ─────────────────────────────────────────────────────────────────────────────
    
    def get_train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            raise ConfigurationError("Training dataset not provided")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )
    
    def get_eval_dataloader(self) -> DataLoader:
        """Create evaluation dataloader."""
        if self.eval_dataset is None:
            raise ConfigurationError("Evaluation dataset not provided")
        
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size or self.args.per_device_batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Training Loop
    # ─────────────────────────────────────────────────────────────────────────────
    
    def _prepare_inputs(self, batch: BatchType) -> BatchType:
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }
    
    def _compute_loss(self, model: nn.Module, inputs: BatchType) -> Tuple[Tensor, Tensor]:
        """
        Compute loss for a batch.
        
        Returns:
            Tuple of (loss, logits)
        """
        # Extract labels if present
        labels = inputs.pop("labels", None)
        
        # Forward pass
        outputs = model(**inputs)
        
        # Handle different output formats
        if isinstance(outputs, tuple):
            logits = outputs[0]
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Compute loss
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        else:
            # Model returned loss directly
            if isinstance(outputs, tuple) and len(outputs) > 1:
                loss = outputs[1] if isinstance(outputs[1], Tensor) else outputs[0]
            elif hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                raise TrainingLoopError("Model did not return loss and no labels provided")
        
        # Put labels back
        if labels is not None:
            inputs["labels"] = labels
        
        return loss, logits
    
    def _training_step(
        self,
        model: nn.Module,
        inputs: BatchType,
    ) -> float:
        """
        Perform single training step.
        
        Handles:
        - Mixed precision forward
        - Gradient accumulation
        - Loss scaling
        
        Returns:
            Loss value (float)
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Forward with autocast
        with self._get_autocast_context():
            loss, _ = self._compute_loss(model, inputs)
            
            # Scale for accumulation
            if self.args.optimizer.gradient_accumulation_steps > 1:
                loss = loss / self.args.optimizer.gradient_accumulation_steps
        
        # Backward with optional scaling
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.detach().item() * self.args.optimizer.gradient_accumulation_steps
    
    def _maybe_clip_gradients(self) -> Optional[GradientInfo]:
        """Clip gradients if configured."""
        max_norm = self.args.optimizer.max_grad_norm
        
        if max_norm is not None and max_norm > 0:
            if self.scaler is not None:
                # Unscale before clipping
                self.scaler.unscale_(self.optimizer)
            
            return clip_grad_norm_(
                self.model.parameters(),
                max_norm,
                error_if_nonfinite=True,
            )
        
        return None
    
    def _optimizer_step(self) -> None:
        """Perform optimizer step with optional scaling."""
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        if self.scheduler is not None:
            self.scheduler.step()
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> TrainOutput:
        """
        Run training loop.
        
        Args:
            resume_from_checkpoint: Optional checkpoint path to resume from
            
        Returns:
            TrainOutput with training results
        """
        # Get dataloader
        train_dataloader = self.get_train_dataloader()
        
        # Compute training steps
        num_update_steps_per_epoch = len(train_dataloader) // self.args.optimizer.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        
        if self.args.max_steps > 0:
            total_steps = self.args.max_steps
            num_epochs = math.ceil(total_steps / num_update_steps_per_epoch)
        else:
            total_steps = num_update_steps_per_epoch * self.args.num_epochs
            num_epochs = self.args.num_epochs
        
        # Create optimizer and scheduler if not provided
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        if self.scheduler is None:
            self.scheduler = self._create_scheduler(total_steps)
        
        # Set reproducibility
        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            random.seed(self.args.seed)
        
        # Initialize callback context
        self._ctx = CallbackContext(
            total_steps=total_steps,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        
        # Trigger train begin
        self.callback_handler.trigger("on_train_begin", self._ctx)
        
        logger.info(f"Starting training for {num_epochs} epochs, {total_steps} steps")
        
        self.global_step = 0
        self._total_loss = 0.0
        
        try:
            for epoch in range(num_epochs):
                self._ctx.epoch = epoch
                self._ctx.is_training = True
                
                self.callback_handler.trigger("on_epoch_begin", self._ctx)
                
                epoch_loss = self._train_epoch(train_dataloader, total_steps)
                
                self._ctx.loss = epoch_loss
                
                # Evaluation
                if self.eval_dataset is not None and self._should_evaluate():
                    eval_metrics = self.evaluate()
                    self._ctx.metrics.update(eval_metrics)
                
                self.callback_handler.trigger("on_epoch_end", self._ctx)
                
                # Check for early stopping
                if self._ctx.should_stop:
                    logger.info("Training stopped early")
                    break
                
                # Check max steps
                if self.global_step >= total_steps:
                    break
        
        finally:
            self.callback_handler.trigger("on_train_end", self._ctx)
        
        avg_loss = self._total_loss / max(1, self.global_step)
        
        return TrainOutput(
            global_step=self.global_step,
            training_loss=avg_loss,
            metrics=self._ctx.metrics,
        )
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        total_steps: int,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        step_count = 0
        
        accum_steps = self.args.optimizer.gradient_accumulation_steps
        
        for step, batch in enumerate(dataloader):
            self._ctx.step = self.global_step
            self.callback_handler.trigger("on_step_begin", self._ctx)
            
            if self._ctx.skip_step:
                self._ctx.skip_step = False
                continue
            
            # Training step
            loss = self._training_step(self.model, batch)
            epoch_loss += loss
            step_count += 1
            
            # Accumulation check
            if (step + 1) % accum_steps == 0:
                # Gradient clipping
                grad_info = self._maybe_clip_gradients()
                
                # Optimizer step
                self._optimizer_step()
                
                self.global_step += 1
                self._total_loss += loss
                
                # Update context
                self._ctx.step = self.global_step
                self._ctx.loss = loss
                self._ctx.learning_rate = self.scheduler.get_lr()[0] if self.scheduler else 0.0
                
                self.callback_handler.trigger("on_step_end", self._ctx)
                
                # Check max steps
                if self.global_step >= total_steps:
                    break
                
                # Check early stopping
                if self._ctx.should_stop:
                    break
        
        return epoch_loss / max(1, step_count)
    
    def _should_evaluate(self) -> bool:
        """Check if evaluation should run."""
        strategy = self.args.evaluation.strategy
        
        if strategy == "no":
            return False
        elif strategy == "epoch":
            return True
        elif strategy == "steps":
            return self.global_step % self.args.evaluation.eval_steps == 0
        
        return False
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────────
    
    @torch.no_grad()
    def evaluate(self) -> MetricsType:
        """
        Run evaluation loop.
        
        Returns:
            Dictionary of evaluation metrics
        """
        eval_dataloader = self.get_eval_dataloader()
        
        self.model.eval()
        self._ctx.is_training = False
        
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []
        
        for batch in eval_dataloader:
            inputs = self._prepare_inputs(batch)
            labels = inputs.get("labels")
            
            with self._get_autocast_context():
                loss, logits = self._compute_loss(self.model, inputs)
            
            total_loss += loss.item()
            num_batches += 1
            
            if labels is not None:
                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        metrics = {"eval_loss": total_loss / max(1, num_batches)}
        
        # Compute custom metrics
        if self.compute_metrics is not None and all_preds:
            preds = torch.cat(all_preds)
            labels = torch.cat(all_labels)
            custom_metrics = self.compute_metrics(preds, labels)
            metrics.update({f"eval_{k}": v for k, v in custom_metrics.items()})
        
        self.callback_handler.trigger("on_evaluate", self._ctx)
        
        return metrics
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────────────
    
    @torch.no_grad()
    def predict(self, dataset: Dataset) -> PredictionOutput:
        """
        Run prediction on dataset.
        
        Args:
            dataset: Dataset to predict on
            
        Returns:
            PredictionOutput with predictions
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size or self.args.per_device_batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
        )
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            inputs = self._prepare_inputs(batch)
            labels = inputs.pop("labels", None)
            
            with self._get_autocast_context():
                outputs = self.model(**inputs)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
            
            all_preds.append(logits.cpu())
            if labels is not None:
                all_labels.append(labels.cpu())
        
        predictions = torch.cat(all_preds)
        label_ids = torch.cat(all_labels) if all_labels else None
        
        metrics = {}
        if self.compute_metrics is not None and label_ids is not None:
            preds = predictions.argmax(dim=-1)
            metrics = self.compute_metrics(preds, label_ids)
        
        return PredictionOutput(
            predictions=predictions,
            label_ids=label_ids,
            metrics=metrics,
        )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ─────────────────────────────────────────────────────────────────────────────
    
    def save_model(self, output_dir: Optional[str] = None) -> None:
        """Save model to directory."""
        output_dir = Path(output_dir or self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_path = output_dir / "model.pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Save config
        config_path = output_dir / "training_args.json"
        with open(config_path, "w") as f:
            f.write(self.args.model_dump_json(indent=2))
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, checkpoint_dir: str) -> None:
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_dir)
        
        # Load model state
        model_path = checkpoint_path / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        logger.info(f"Model loaded from {checkpoint_dir}")


# ═════════════════════════════════════════════════════════════════════════════════
# Export
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    "Trainer",
    "TrainOutput",
    "PredictionOutput",
]
