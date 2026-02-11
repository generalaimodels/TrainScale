# ════════════════════════════════════════════════════════════════════════════════
# SOTA Unified Trainer v2.0 — Hardened FSDP2 Integration
# ════════════════════════════════════════════════════════════════════════════════
#
# End-to-end training orchestrator with YAML-driven configuration.
#
# ARCHITECTURE:
#   YAML Config → SOTAConfig → SOTATrainer → Model + Optimized Training
#
# INTEGRATION FIXES (v2.0):
#   [INT-001] FSDP2 wrapper now stored as self._fsdp2_engine and used for
#             backward(), step(), clip_grad_norm_(), checkpoint operations.
#   [INT-002] Trainer no longer manages its own GradScaler when FSDP2 is
#             active — FSDP2's MixedPrecisionContext handles all scaling.
#   [INT-003] Trainer no longer calls loss.backward() directly when FSDP2
#             is active — FSDP2.backward() handles no_sync, accumulation,
#             and loss scaling internally.
#   [INT-004] Trainer no longer calls optimizer.step() directly when FSDP2
#             is active — FSDP2.step() handles unscale→clip→step→zero_grad.
#   [INT-005] Config mapping from SOTAConfig → FSDP2Config is explicit
#             with proper type conversion (strings → enums).
#   [INT-006] Checkpoint manager calls use correct API names:
#             save_checkpoint() / load_checkpoint() (not save_sharded_*).
#   [INT-007] FSDP2's forward_context() is used instead of trainer's own
#             amp_context when FSDP2 is active.
#   [INT-008] gradient_accumulation_steps and gradient_clipping_norm are
#             forwarded to FSDP2Config so FSDP2 manages no_sync cycles
#             and distributed gradient clipping.
#   [INT-009] Memory pressure metrics from FSDP2's MetricsCollector are
#             logged alongside training metrics.
#   [INT-010] Removed double no_sync wrapping — FSDP2.backward() handles
#             no_sync internally; trainer's _get_no_sync_context is only
#             used for non-FSDP2 strategies (DDP, etc.).
#
# FEATURES:
#   • Full-finetuning, LoRA, QLoRA, FP8, Pretraining
#   • RL: GRPO, GSPO, DrGRPO, DAPO
#   • Triton kernels with manual backprop (0% accuracy loss)
#   • Export: GGUF, vLLM, SGLang, HuggingFace
#   • Multi-GPU: DDP, FSDP2 (hardened), DeepSpeed, ZBPP Pipeline, CP
#   • Hardware: NVIDIA V100+, AMD MI300X, Intel
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
import os
import datetime
import math
import time
import warnings
import random
import numpy as np
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

# ═════════════════════════════════════════════════════════════════════════════════
# Internal Imports — Distributed
# ═════════════════════════════════════════════════════════════════════════════════

from data_pipeline.trainer.distributed import (
    # DDP
    DDPInitializer,
    create_ddp_from_yaml,
    create_ddp_engine,
    SOTADDP,
    DDPConfig,
    # FSDP2 — [INT-001] import the class AND factory for proper lifecycle
    create_fsdp2_from_dict,
    create_fsdp2,
    SOTAFSDP2,
    FSDPCheckpointManager,
    FSDP2Config,
    ShardingStrategy,
    MixedPrecisionPolicy,
    BackwardPrefetchMode,
    OffloadStrategy,
    # Context Parallel
    ContextParallelEngine,
    create_context_parallel_engine,
    # Activation Checkpoint
    ActivationCheckpoint,
    create_activation_checkpoint,
    # Utilities
    DistributedState,
    is_main_process,
    get_world_info,
    set_deterministic_seed,
    log_rank_0,
    # SOTA: Gradient sync context (used for DDP only, NOT for FSDP2)
    no_sync,
)

# ═════════════════════════════════════════════════════════════════════════════════
# Internal Imports — Core Configuration
# ═════════════════════════════════════════════════════════════════════════════════

from data_pipeline.trainer.core.sota_config import (
    SOTAConfig,
    TrainingMode,
    Precision,
    OptimizerType,
    SchedulerType,
    LossType,
    RLAlgorithm,
    ExportFormat,
)

from data_pipeline.trainer.core.types import (
    LoggingConfig,
    LoggingBackend,
    CheckpointConfig,
    CheckpointStrategy,
)

# ═════════════════════════════════════════════════════════════════════════════════
# Internal Imports — Callbacks
# ═════════════════════════════════════════════════════════════════════════════════

from data_pipeline.trainer.callbacks import (
    Callback,
    CallbackHandler,
    CallbackContext,
    EarlyStoppingCallback,
    CheckpointCallback,
    LoggingCallback,
    ProgressCallback,
)

# ═════════════════════════════════════════════════════════════════════════════════
# Internal Imports — Metrics
# ═════════════════════════════════════════════════════════════════════════════════

from data_pipeline.trainer.metrics import (
    LossTracker,
    AccuracyTracker,
    ThroughputTracker,
    GradientTracker,
    TrainingMetrics,
    create_training_metrics,
    sync_metrics,
    is_distributed as metrics_is_distributed,
)

# ═════════════════════════════════════════════════════════════════════════════════
# Internal Imports — Loss Functions
# ═════════════════════════════════════════════════════════════════════════════════

from data_pipeline.trainer.loss import (
    CrossEntropyLoss,
    KLDivergenceLoss,
    FocalLoss,
    create_loss,
    ChunkedCrossEntropyLoss,
    FusedCrossEntropyLoss,
    DistillationLoss,
    DPOLoss,
    KTOLoss,
    ORPOLoss,
    SimPOLoss,
    CPOLoss,
    InfoNCELoss,
    CLIPLoss,
    ZLoss,
    LoadBalancingLoss,
    LossRegistry,
)

# ═════════════════════════════════════════════════════════════════════════════════
# Internal Imports — Optimizers
# ═════════════════════════════════════════════════════════════════════════════════

from data_pipeline.trainer.optimizers import (
    AdamW,
    LAMB,
    AdaFactor,
    Adam8bit,
    Lion,
    CAME,
    SophiaG,
    Prodigy,
    FusedAdamW,
    create_optimizer,
    clip_grad_norm_,
)

# ═════════════════════════════════════════════════════════════════════════════════
# Internal Imports — Schedulers
# ═════════════════════════════════════════════════════════════════════════════════

from data_pipeline.trainer.schedulers import (
    BaseScheduler,
    CosineScheduler,
    LinearScheduler,
    create_scheduler,
    WSDScheduler,
    REXScheduler,
    create_sota_scheduler,
)

# ═════════════════════════════════════════════════════════════════════════════════
# Internal Imports — Hub
# ═════════════════════════════════════════════════════════════════════════════════

from data_pipeline.trainer.hub import (
    HubManager,
    CheckpointManager,
    ModelSerializer,
    HubConfig,
    HubError,
    Ok,
    Err,
    get_hub_metrics,
)

# ═════════════════════════════════════════════════════════════════════════════════
# Internal Imports — Kernels
# ═════════════════════════════════════════════════════════════════════════════════

from data_pipeline.trainer.kernels import (
    is_triton_available,
    get_kernel_capabilities,
    is_flash_attn_available,
)

# ═════════════════════════════════════════════════════════════════════════════════
# Internal Imports — Registry
# ═════════════════════════════════════════════════════════════════════════════════

from data_pipeline.trainer.registry import (
    get_model_info,
    ModelInfo,
    register_model,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════════
# Training State
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingState:
    """Serializable training state for checkpoint resumption."""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float('inf')
    total_loss: float = 0.0
    samples_seen: int = 0
    patience_counter: int = 0


# ═════════════════════════════════════════════════════════════════════════════════
# FSDP2 Config Bridge — Explicit SOTAConfig → FSDP2Config Translation
# ═════════════════════════════════════════════════════════════════════════════════
#
# [INT-005] The original code passed a raw config dict to create_fsdp2_from_dict,
# but the factory expected flat keys like "sharding_strategy" = "full_shard",
# not nested dicts. The trainer was building {"distributed": {"use_orig_params": ...}}
# which the factory silently ignored → FSDP2 used defaults for everything.
#
# This bridge function performs explicit, type-safe translation with validation.
# ═════════════════════════════════════════════════════════════════════════════════

def _build_fsdp2_config(sota_config: SOTAConfig) -> FSDP2Config:
    """
    Translate SOTAConfig into FSDP2Config with explicit type mapping.

    Handles all the impedance mismatches between the two config systems:
        - SOTAConfig uses string enums; FSDP2Config uses Enum classes
        - SOTAConfig stores precision in hardware.precision; FSDP2 uses
          MixedPrecisionPolicy enum
        - SOTAConfig has gradient_accumulation_steps in training; FSDP2
          needs it for no_sync cycle management
        - SOTAConfig has gradient_clipping_norm in optimizer; FSDP2 needs
          it for distributed clip_grad_norm_

    Returns:
        Fully populated FSDP2Config ready for SOTAFSDP2 construction.
    """
    dist_cfg = sota_config.distributed
    hw_cfg = sota_config.hardware
    train_cfg = sota_config.training
    opt_cfg = sota_config.optimizer

    # ── Sharding Strategy ──
    sharding_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    fsdp_sharding_str = getattr(dist_cfg, "fsdp_sharding_strategy", "full_shard")
    sharding = sharding_map.get(fsdp_sharding_str.lower(), ShardingStrategy.FULL_SHARD)

    # ── Mixed Precision ──
    precision_map = {
        Precision.BF16: MixedPrecisionPolicy.FULL_BF16,
        Precision.FP16: MixedPrecisionPolicy.FULL_FP16,
        Precision.FP32: MixedPrecisionPolicy.PURE_FP32,
        Precision.FP8_E4M3: MixedPrecisionPolicy.FULL_BF16,  # FP8 compute, bf16 comms
        Precision.FP8_E5M2: MixedPrecisionPolicy.FULL_BF16,
    }
    mixed_prec = precision_map.get(hw_cfg.precision, MixedPrecisionPolicy.FULL_BF16)

    # ── Backward Prefetch ──
    prefetch_map = {
        "backward_pre": BackwardPrefetchMode.BACKWARD_PRE,
        "backward_post": BackwardPrefetchMode.BACKWARD_POST,
        "none": BackwardPrefetchMode.NONE,
    }
    prefetch_str = getattr(dist_cfg, "fsdp_backward_prefetch", "backward_pre")
    backward_prefetch = prefetch_map.get(prefetch_str.lower(), BackwardPrefetchMode.BACKWARD_PRE)

    # ── CPU Offload ──
    offload = OffloadStrategy.NONE
    if getattr(dist_cfg, "fsdp_cpu_offload", False):
        offload = OffloadStrategy.CPU_FULL

    # ── Activation Checkpointing ──
    ac_enabled = getattr(dist_cfg, "gradient_checkpointing", False)
    ac_mode = getattr(dist_cfg, "ac_mode", "selective")
    ac_freq = getattr(dist_cfg, "ac_frequency", 2)

    # ── Gradient Accumulation (from training config) ──
    grad_accum = getattr(train_cfg, "gradient_accumulation_steps", 1)

    # ── Gradient Clipping (from optimizer config) ──
    grad_clip = getattr(opt_cfg, "max_grad_norm", 1.0)
    if grad_clip <= 0:
        grad_clip = None  # FSDP2 interprets None as "no clipping"

    # ── use_orig_params (required for torch.compile + FSDP) ──
    use_orig = getattr(dist_cfg, "fsdp_use_orig_params", True)
    if hw_cfg.compile_model:
        use_orig = True  # Mandatory for torch.compile compatibility

    # ── Triton kernels ──
    use_triton = True
    if hasattr(sota_config, "kernels"):
        use_triton = sota_config.kernels.use_triton

    # ── Build FSDP2Config ──
    return FSDP2Config(
        sharding_strategy=sharding,
        mixed_precision=mixed_prec,
        offload_strategy=offload,
        use_orig_params=use_orig,
        forward_prefetch=getattr(dist_cfg, "fsdp_forward_prefetch", True),
        backward_prefetch=backward_prefetch,
        limit_all_gathers=getattr(dist_cfg, "fsdp_limit_all_gathers", True),
        use_triton_kernels=use_triton,
        bucket_size_mb=getattr(dist_cfg, "fsdp_bucket_size_mb", 25),
        sync_module_states=True,
        activation_checkpointing=ac_enabled,
        ac_mode=ac_mode,
        ac_frequency=ac_freq,
        gradient_accumulation_steps=grad_accum,
        gradient_clipping_norm=grad_clip,
        use_memory_pool=True,
        deterministic=getattr(train_cfg, "deterministic", False),
        debug_mode=getattr(train_cfg, "debug_mode", False),
    )


# ═════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer
# ═════════════════════════════════════════════════════════════════════════════════

class SOTATrainer:
    """
    SOTA Unified Trainer with hardened FSDP2 integration.

    When distributed_strategy is "fsdp"/"fsdp2"/"sota_fsdp":
        - self._fsdp2_engine holds the SOTAFSDP2 instance
        - backward() delegates to fsdp2_engine.backward() which handles
          no_sync, gradient accumulation, and loss scaling
        - step() delegates to fsdp2_engine.step() which handles
          unscale→clip→optimizer.step→zero_grad→reset
        - Trainer does NOT manage its own GradScaler
        - Trainer does NOT call loss.backward() directly
        - Checkpointing uses FSDPCheckpointManager via the engine

    For all other strategies (DDP, ZBPP, CP, single-GPU):
        - Trainer manages backward/step/scaler/no_sync directly
        - Original behavior preserved

    Example:
        >>> config = SOTAConfig.from_yaml("config.yaml")
        >>> trainer = SOTATrainer(config)
        >>> trainer.setup_model()
        >>> trainer.train(train_dataloader)
        >>> trainer.export()
    """

    def __init__(self, config: SOTAConfig):
        self.config = config
        self.state = TrainingState()

        # Components (initialized lazily)
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.loss_fn: Optional[nn.Module] = None
        self.scaler: Optional[torch.amp.GradScaler] = None

        # ─────────────────────────────────────────────────────────────────────
        # [INT-001] FSDP2 engine reference — set during setup_model()
        # When active, this object owns the training loop lifecycle:
        #   forward_context, backward, step, clip_grad_norm_, checkpointing
        # ─────────────────────────────────────────────────────────────────────
        self._fsdp2_engine: Optional[SOTAFSDP2] = None

        # Callback Handler
        self.callback_handler = CallbackHandler()
        self._setup_default_callbacks()

        # Training Metrics
        self.training_metrics: Optional[TrainingMetrics] = None
        self.loss_tracker: Optional[LossTracker] = None
        self.throughput_tracker: Optional[ThroughputTracker] = None
        self.gradient_tracker: Optional[GradientTracker] = None

        # Hub Integration
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.hub_manager: Optional[HubManager] = None
        self.model_serializer: Optional[ModelSerializer] = None

        # Kernel Capabilities
        self.kernel_capabilities = get_kernel_capabilities()

        # Distributed Initialization
        self.distributed_strategy = (
            config.distributed.strategy
            if hasattr(config.distributed, "strategy")
            else None
        )
        if self.distributed_strategy in ("ddp", "fsdp", "fsdp2", "sota_fsdp"):
            DDPInitializer.init_process_group()
            log_rank_0(
                f"Initialized distributed process group: {self.distributed_strategy}"
            )

        # Device setup
        self.device = self._setup_device()
        self.dtype = config.compute_dtype

        # [INT-002] Mixed precision context — only used for non-FSDP2 strategies
        # FSDP2 manages its own autocast via forward_context()
        self.amp_context = self._setup_amp_context()

        # Initialize metrics trackers
        self._setup_metrics()

        # Initialize hub components
        self._setup_hub()

        log_rank_0(
            f"SOTATrainer initialized: mode={config.training_mode.value}, "
            f"device={self.device}, strategy={self.distributed_strategy}"
        )

    # ═════════════════════════════════════════════════════════════════════════════
    # Properties — FSDP2 Awareness
    # ═════════════════════════════════════════════════════════════════════════════

    @property
    def _is_fsdp2_active(self) -> bool:
        """Check if FSDP2 engine is managing the training lifecycle."""
        return self._fsdp2_engine is not None

    # ═════════════════════════════════════════════════════════════════════════════
    # Device Setup
    # ═════════════════════════════════════════════════════════════════════════════

    def _setup_device(self) -> torch.device:
        """Setup compute device based on config."""
        hw = self.config.hardware

        if hw.device.value == "auto":
            if torch.cuda.is_available():
                # Use LOCAL_RANK for multi-GPU
                local_rank = int(os.environ.get("LOCAL_RANK", hw.device_id))
                device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(device)
                # Enable TF32 on Ampere+
                if hw.tf32 and torch.cuda.get_device_capability()[0] >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = torch.device("xpu")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(hw.device.value)

        return device

    def _setup_amp_context(self):
        """
        Setup automatic mixed precision context.

        [INT-002] This is ONLY used for non-FSDP2 strategies.
        When FSDP2 is active, fsdp2_engine.forward_context() provides autocast.
        """
        precision = self.config.hardware.precision

        if precision == Precision.FP16:
            # [INT-002] GradScaler only created for non-FSDP2 fp16
            # FSDP2 manages its own scaler via MixedPrecisionContext
            if self.distributed_strategy not in ("fsdp", "fsdp2", "sota_fsdp"):
                self.scaler = torch.amp.GradScaler(device="cuda")
            return lambda: torch.amp.autocast('cuda', dtype=torch.float16)
        elif precision == Precision.BF16:
            return lambda: torch.amp.autocast('cuda', dtype=torch.bfloat16)
        elif precision in (Precision.FP8_E4M3, Precision.FP8_E5M2):
            return lambda: torch.amp.autocast('cuda', dtype=torch.bfloat16)
        else:
            return lambda: nullcontext()

    def _setup_default_callbacks(self) -> None:
        """Setup default callbacks from callbacks sub-module."""
        train_cfg = self.config.training

        log_steps = getattr(train_cfg, 'logging_steps', 100)
        logging_config = LoggingConfig(
            backends=[LoggingBackend.CONSOLE],
            log_steps=log_steps,
            log_dir=getattr(train_cfg, 'logging_dir', 'logs'),
        )
        self.callback_handler.add_callback(LoggingCallback(config=logging_config))
        self.callback_handler.add_callback(ProgressCallback())

        if hasattr(train_cfg, 'save_steps') and train_cfg.save_steps > 0:
            ckpt_config = CheckpointConfig(
                strategy=CheckpointStrategy.STEPS,
                save_steps=train_cfg.save_steps,
                save_total_limit=getattr(train_cfg, 'save_total_limit', 3),
            )
            self.callback_handler.add_callback(CheckpointCallback(
                save_dir=getattr(train_cfg, 'output_dir', './checkpoints'),
                config=ckpt_config,
            ))

        if (
            hasattr(train_cfg, 'early_stopping_patience')
            and train_cfg.early_stopping_patience > 0
        ):
            self.callback_handler.add_callback(EarlyStoppingCallback(
                patience=train_cfg.early_stopping_patience,
                min_delta=getattr(train_cfg, 'early_stopping_threshold', 0.0),
            ))

    def _setup_metrics(self) -> None:
        """Setup training metrics from metrics sub-module."""
        try:
            self.training_metrics = create_training_metrics(
                distributed=metrics_is_distributed(),
                model=self.model,
            )
            self.loss_tracker = self.training_metrics.loss
            self.throughput_tracker = self.training_metrics.throughput
            self.gradient_tracker = self.training_metrics.gradient
        except Exception as e:
            logger.warning(f"Could not create unified TrainingMetrics: {e}")
            self.training_metrics = None
            self.loss_tracker = LossTracker(ema_decay=0.99)
            self.throughput_tracker = ThroughputTracker()
            self.gradient_tracker = GradientTracker()

    def _setup_hub(self) -> None:
        """Setup hub integration from hub sub-module."""
        export_cfg = self.config.export
        train_cfg = self.config.training

        self.hub_manager = None
        if hasattr(export_cfg, 'push_to_hub') and export_cfg.push_to_hub:
            try:
                hub_config = HubConfig(
                    token=getattr(export_cfg, 'hub_token', None),
                    repo_id=getattr(export_cfg, 'hub_model_id', None),
                    private=getattr(export_cfg, 'hub_private', False),
                )
                self.hub_manager = HubManager(config=hub_config)
            except Exception as e:
                logger.warning(f"Could not initialize HubManager: {e}")

        checkpoint_dir = getattr(train_cfg, 'output_dir', './checkpoints')
        max_checkpoints = getattr(train_cfg, 'save_total_limit', 3)

        try:
            self.checkpoint_manager = CheckpointManager(
                save_dir=Path(checkpoint_dir),
                hub_manager=self.hub_manager,
                max_checkpoints=max_checkpoints,
            )
        except Exception as e:
            logger.warning(f"Could not initialize CheckpointManager: {e}")
            self.checkpoint_manager = None

        self.model_serializer = ModelSerializer(
            shard_size_bytes=5 * 1024**3,
            prefer_safetensors=True,
        )

    # ═════════════════════════════════════════════════════════════════════════════
    # Numerical Stability — NaN/Inf Detection
    # ═════════════════════════════════════════════════════════════════════════════

    def _check_numerical_stability(
        self,
        loss: Tensor,
        grad_norm: Optional[float] = None,
        step: Optional[int] = None,
    ) -> bool:
        """
        Check for numerical stability issues.

        Returns True if stable, False if NaN/Inf detected.
        """
        step_info = f" at step {step}" if step is not None else ""

        if torch.isnan(loss) or torch.isinf(loss):
            log_rank_0(
                f"⚠️ NaN/Inf loss detected{step_info}: {loss.item():.4e}",
                level="warning",
            )
            return False

        if grad_norm is not None:
            if grad_norm != grad_norm:  # NaN check
                log_rank_0(f"⚠️ NaN gradient norm{step_info}", level="warning")
                return False
            if grad_norm > 1e6:
                log_rank_0(
                    f"⚠️ Exploding gradient{step_info}: {grad_norm:.2e}",
                    level="warning",
                )

        return True

    def _get_no_sync_context(self, should_sync: bool):
        """
        Get gradient sync context for gradient accumulation.

        [INT-010] This is ONLY used for DDP strategy.
        FSDP2 manages no_sync internally via its backward() method.
        """
        if should_sync:
            return nullcontext()
        return no_sync(self.model)

    # ═════════════════════════════════════════════════════════════════════════════
    # Model Setup
    # ═════════════════════════════════════════════════════════════════════════════

    def setup_model(
        self,
        model: Optional[nn.Module] = None,
        model_init_fn: Optional[Callable[[], nn.Module]] = None,
    ) -> nn.Module:
        """
        Setup model with optimizations and distributed wrapping.

        For FSDP2:
            1. Build FSDP2Config from SOTAConfig
            2. Create SOTAFSDP2 engine
            3. Wrap model (activation checkpointing applied inside)
            4. Store engine as self._fsdp2_engine
            5. Trainer delegates backward/step/checkpoint to engine

        For other strategies:
            - Original behavior preserved
        """
        if model is not None:
            self.model = model
        elif model_init_fn is not None:
            self.model = model_init_fn()
        else:
            self.model = self._load_model()

        # Apply training mode (LoRA, full-finetune, etc.)
        self.model = self._apply_training_mode(self.model)

        # Apply kernel optimizations
        self.model = self._apply_kernel_optimizations(self.model)

        # Move to device (MUST happen before FSDP/DDP wrapping)
        self.model = self.model.to(self.device)

        # ─────────────────────────────────────────────────────────────
        # Distributed Wrapping
        # ─────────────────────────────────────────────────────────────

        if self.distributed_strategy == "ddp":
            log_rank_0("Applying SOTA DDP wrapper...")

            dist_cfg = self.config.distributed
            ddp_params = {
                "gradient_as_bucket_view": dist_cfg.ddp_gradient_as_bucket_view,
                "static_graph": dist_cfg.ddp_static_graph,
                "find_unused_parameters": dist_cfg.ddp_find_unused_parameters,
                "broadcast_buffers": dist_cfg.ddp_broadcast_buffers,
            }
            ddp_params.update(dist_cfg.ddp_config)

            if hasattr(self.config, "kernels") and "use_triton_kernels" not in ddp_params:
                ddp_params["use_triton_kernels"] = self.config.kernels.use_triton

            try:
                self.model = (
                    create_ddp_engine(**ddp_params).unwrap().wrap_model(self.model)
                )
                log_rank_0(
                    f"  ✓ DDP applied (static_graph={ddp_params['static_graph']})"
                )
            except Exception as e:
                logger.error(f"DDP wrapping failed: {e}")
                raise

        elif self.distributed_strategy in ("fsdp", "fsdp2", "sota_fsdp"):
            # ═══════════════════════════════════════════════════════════════
            # [INT-001] FSDP2 — Proper lifecycle management
            #
            # The engine is created here and stored on self._fsdp2_engine.
            # From this point forward, the trainer delegates:
            #   - forward autocast  → engine.forward_context()
            #   - backward + accum  → engine.backward()
            #   - optimizer step    → engine.step()
            #   - grad clipping     → engine.clip_grad_norm_()
            #   - checkpointing     → FSDPCheckpointManager via engine
            #   - memory mgmt       → engine.empty_cache(), engine.memory_summary()
            # ═══════════════════════════════════════════════════════════════
            log_rank_0(
                f"Applying SOTA FSDP2 ({self.distributed_strategy})..."
            )

            # [INT-005] Build typed FSDP2Config from SOTAConfig
            fsdp2_config = _build_fsdp2_config(self.config)

            try:
                # Create engine
                self._fsdp2_engine = SOTAFSDP2(fsdp2_config)

                # Wrap model (activation checkpointing applied inside)
                self.model = self._fsdp2_engine.wrap_model(self.model)

                # [INT-002] Nullify trainer's scaler — FSDP2 owns scaling
                self.scaler = None

                log_rank_0(
                    f"  ✓ FSDP2 applied: "
                    f"strategy={fsdp2_config.sharding_strategy.name}, "
                    f"precision={fsdp2_config.mixed_precision.name}, "
                    f"grad_accum={fsdp2_config.gradient_accumulation_steps}, "
                    f"grad_clip={fsdp2_config.gradient_clipping_norm}"
                )
            except Exception as e:
                logger.error(f"FSDP2 wrapping failed: {e}")
                raise

        elif self.distributed_strategy == "pipeline_zbpp":
            log_rank_0("Applying SOTA ZBPP Pipeline wrapper...")
            from data_pipeline.trainer.distributed.zbpp import create_zbpp_pipeline

            num_stages = self.config.distributed.num_pipeline_stages
            num_microbatches = self.config.distributed.num_microbatches

            if num_microbatches < num_stages:
                logger.warning(
                    f"num_microbatches ({num_microbatches}) < num_stages ({num_stages}), "
                    f"adjusting to {num_stages}"
                )
                num_microbatches = num_stages

            try:
                self.model = create_zbpp_pipeline(
                    model=self.model,
                    num_stages=num_stages,
                    num_microbatches=num_microbatches,
                    memory_limit_gb=self.config.distributed.pipeline_memory_limit_gb,
                    lr=self.config.optimizer.learning_rate,
                    weight_decay=self.config.optimizer.weight_decay,
                    dtype=self.dtype,
                    rank=dist.get_rank() if dist.is_initialized() else 0,
                    world_size=dist.get_world_size() if dist.is_initialized() else 1,
                )
                self.optimizer = self.model._optimizer
                log_rank_0(
                    f"  ✓ ZBPP applied (stages={num_stages}, μB={num_microbatches})"
                )
            except Exception as e:
                logger.error(f"ZBPP wrapping failed: {e}")
                raise

        elif self.distributed_strategy == "context_parallel":
            log_rank_0("Applying Context Parallel wrapper...")

            cp_size = getattr(self.config.distributed, "context_parallel_size", 2)
            try:
                cp_engine = create_context_parallel_engine(
                    model=self.model,
                    cp_size=cp_size,
                    device=self.device,
                )
                self.model = cp_engine.wrap_model()
                log_rank_0(f"  ✓ Context Parallel applied (cp_size={cp_size})")
            except Exception as e:
                logger.error(f"Context Parallel wrapping failed: {e}")
                raise

        # ─────────────────────────────────────────────────────────────
        # Model Compilation (after wrapping)
        # ─────────────────────────────────────────────────────────────
        if self.config.hardware.compile_model:
            log_rank_0(
                f"Compiling model (mode={self.config.hardware.compile_mode})..."
            )
            self.model = torch.compile(
                self.model,
                mode=self.config.hardware.compile_mode,
            )

        return self.model

    def _load_model(self) -> nn.Module:
        """Load model from config."""
        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError:
            raise ImportError("transformers required for model loading")

        model_cfg = self.config.model
        quant_cfg = self.config.quantization

        bnb_config = None
        if quant_cfg.enabled:
            if quant_cfg.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=getattr(torch, quant_cfg.bnb_4bit_compute_dtype),
                    bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
                )
            elif quant_cfg.load_in_8bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=quant_cfg.llm_int8_threshold,
                )

        use_cache = not self.config.distributed.gradient_checkpointing

        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.name_or_path,
            revision=model_cfg.revision,
            trust_remote_code=model_cfg.trust_remote_code,
            dtype=model_cfg.torch_dtype if model_cfg.torch_dtype != "auto" else "auto",
            low_cpu_mem_usage=model_cfg.low_cpu_mem_usage,
            attn_implementation=model_cfg.attn_implementation,
            quantization_config=bnb_config,
            use_cache=use_cache,
        )

        return model

    def _apply_training_mode(self, model: nn.Module) -> nn.Module:
        """Apply training mode specific configurations."""
        mode = self.config.training_mode

        if mode in (TrainingMode.LORA, TrainingMode.QLORA):
            model = self._apply_lora(model)
        elif mode in (TrainingMode.FULL_FINETUNE, TrainingMode.PRETRAIN):
            for param in model.parameters():
                param.requires_grad = True

        # Gradient checkpointing
        # [INT-001] For FSDP2, activation checkpointing is applied INSIDE
        # fsdp2_engine.wrap_model(), so we skip it here for FSDP2 strategies.
        if (
            self.config.distributed.gradient_checkpointing
            and self.distributed_strategy not in ("fsdp", "fsdp2", "sota_fsdp")
        ):
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": self.config.distributed.gradient_checkpointing_use_reentrant,
                    }
                )

        return model

    def _apply_lora(self, model: nn.Module) -> nn.Module:
        """Apply LoRA/QLoRA to model."""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError:
            from data_pipeline.trainer.lora import apply_lora
            return apply_lora(model, self.config.lora)

        lora_cfg = self.config.lora

        if self.config.training_mode == TrainingMode.QLORA:
            model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            target_modules=lora_cfg.target_modules,
            bias=lora_cfg.bias,
            use_rslora=lora_cfg.use_rslora,
            use_dora=lora_cfg.use_dora,
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        return model

    # ═════════════════════════════════════════════════════════════════════════
    # Generalized Layer Detection (model-agnostic)
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _is_mlp_swiglu(module: nn.Module) -> bool:
        return (
            hasattr(module, 'gate_proj')
            and hasattr(module, 'up_proj')
            and hasattr(module, 'down_proj')
        )

    @staticmethod
    def _is_mlp_geglu(module: nn.Module) -> bool:
        return (
            (hasattr(module, 'wi_0') and hasattr(module, 'wi_1') and hasattr(module, 'wo'))
            or (hasattr(module, 'fc1') and hasattr(module, 'fc2') and hasattr(module, 'act'))
        )

    @staticmethod
    def _is_mlp_standard(module: nn.Module) -> bool:
        return (
            (hasattr(module, 'dense_h_to_4h') and hasattr(module, 'dense_4h_to_h'))
            or (hasattr(module, 'c_fc') and hasattr(module, 'c_proj'))
            or (hasattr(module, 'fc_in') and hasattr(module, 'fc_out'))
        )

    @staticmethod
    def _is_attention(module: nn.Module) -> bool:
        if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
            return True
        if hasattr(module, 'query') and hasattr(module, 'key') and hasattr(module, 'value'):
            return True
        if hasattr(module, 'q') and hasattr(module, 'k') and hasattr(module, 'v'):
            return True
        if hasattr(module, 'query_key_value'):
            return True
        if hasattr(module, 'Wqkv') or hasattr(module, 'qkv_proj'):
            return True
        if hasattr(module, 'c_attn'):
            return True
        if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
            return True
        return False

    @staticmethod
    def _is_layernorm(module: nn.Module) -> bool:
        import re
        cls_name = module.__class__.__name__
        return bool(re.match(
            r".*(?:RMS[_]?Norm|Layer[_]?Norm|GroupNorm|FusedNorm)",
            cls_name, re.IGNORECASE,
        ))

    @staticmethod
    def _is_rope_embedding(module: nn.Module) -> bool:
        import re
        cls_name = module.__class__.__name__
        if re.match(r".*Rotary.*Embed.*", cls_name, re.IGNORECASE):
            return True
        if hasattr(module, 'inv_freq') and hasattr(module, 'cos_cached'):
            return True
        if hasattr(module, 'inv_freq') and hasattr(module, 'max_seq_len_cached'):
            return True
        return False

    @staticmethod
    def _is_moe_layer(module: nn.Module) -> bool:
        import re
        cls_name = module.__class__.__name__
        if re.match(
            r".*(?:SparseMoe|MoE|MixtureOfExperts|BlockSparse|"
            r"ExpertLayer|SwitchTransformer|TopKGate).*",
            cls_name, re.IGNORECASE,
        ):
            return True
        if hasattr(module, 'experts') and (
            hasattr(module, 'gate') or hasattr(module, 'router')
            or hasattr(module, 'gating_network')
        ):
            return True
        return False

    @staticmethod
    def _has_lora_adapters(module: nn.Module) -> bool:
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            return True
        if hasattr(module, 'base_layer'):
            return True
        return False

    @staticmethod
    def _check_model_stability(model: nn.Module) -> None:
        """Pre-optimization sanity check: dtypes, NaN/Inf in weights."""
        try:
            dtypes = {}
            for name, param in model.named_parameters():
                dtype = str(param.dtype)
                dtypes[dtype] = dtypes.get(dtype, 0) + 1

            logger.info(f"Model Stability — Dtype Distribution: {dtypes}")

            if (
                'torch.float32' in dtypes
                and ('torch.float16' in dtypes or 'torch.bfloat16' in dtypes)
            ):
                logger.warning(
                    "⚠ Mixed dtypes (float32 + float16/bf16). "
                    "Ensure intended (e.g. norms/logits in fp32)."
                )

            has_nan = has_inf = False
            for i, (name, param) in enumerate(model.named_parameters()):
                if i > 5:
                    break
                if torch.isnan(param).any():
                    has_nan = True
                    logger.error(f"❌ NaN in weights: {name}")
                if torch.isinf(param).any():
                    has_inf = True
                    logger.error(f"❌ Inf in weights: {name}")

            if not has_nan and not has_inf:
                logger.info("✓ Weight sanity check passed")

        except Exception as e:
            logger.warning(f"Stability check failed: {e}")

    def _apply_kernel_optimizations(self, model: nn.Module) -> nn.Module:
        """
        Apply SOTA kernel optimizations (Triton, Flash, FP8, etc.).

        Model-agnostic: uses duck-typing for layer detection.
        10-phase optimization pipeline with graceful fallback.
        """
        import re

        kernel_cfg = self.config.kernels
        hw_cfg = self.config.hardware
        dist_cfg = self.config.distributed

        # Phase 0: Pre-optimization stability check
        self._check_model_stability(model)

        if not kernel_cfg.use_triton:
            logger.info("Triton disabled — skipping kernel optimizations")
            return model

        # Phase 1: Hardware capability detection
        capabilities = get_kernel_capabilities()
        triton_ok = capabilities.get("triton", False)
        flash_ok = capabilities.get("flash_attn", False)
        fp8_ok = capabilities.get("fp8", False)
        bf16_ok = capabilities.get("bf16", False)
        cuda_ok = capabilities.get("cuda", False)
        tma_ok = capabilities.get("tma", False)
        wgmma_ok = capabilities.get("wgmma", False)

        try:
            from data_pipeline.trainer.registry import detect_capabilities
            hw_caps = detect_capabilities()
        except (ImportError, Exception):
            hw_caps = {}

        logger.info(
            f"Kernel Capabilities: triton={triton_ok}, flash={flash_ok}, "
            f"fp8={fp8_ok}, bf16={bf16_ok}, tma={tma_ok}"
        )

        applied_phases = []

        # Pre-scan: build generalized layer map
        layer_map = {
            "mlp_swiglu": [], "mlp_geglu": [], "mlp_standard": [],
            "attention": [], "layernorm": [], "rope": [],
            "moe": [], "linear": [],
        }

        for name, module in model.named_modules():
            if self._is_mlp_swiglu(module):
                layer_map["mlp_swiglu"].append((name, module))
            elif self._is_mlp_geglu(module):
                layer_map["mlp_geglu"].append((name, module))
            elif self._is_mlp_standard(module):
                layer_map["mlp_standard"].append((name, module))
            if self._is_attention(module):
                layer_map["attention"].append((name, module))
            if self._is_layernorm(module):
                layer_map["layernorm"].append((name, module))
            if self._is_rope_embedding(module):
                layer_map["rope"].append((name, module))
            if self._is_moe_layer(module):
                layer_map["moe"].append((name, module))
            if isinstance(module, nn.Linear):
                layer_map["linear"].append((name, module))

        logger.info(
            f"Layer scan: mlp_swiglu={len(layer_map['mlp_swiglu'])}, "
            f"mlp_geglu={len(layer_map['mlp_geglu'])}, "
            f"mlp_std={len(layer_map['mlp_standard'])}, "
            f"attn={len(layer_map['attention'])}, "
            f"norm={len(layer_map['layernorm'])}, "
            f"rope={len(layer_map['rope'])}, "
            f"moe={len(layer_map['moe'])}, "
            f"linear={len(layer_map['linear'])}"
        )

        # Phase 2: Layer patching
        try:
            from data_pipeline.trainer.registry import KernelPatcher
            patcher = KernelPatcher(
                patch_layernorm=kernel_cfg.use_fused_rms_norm and triton_ok,
                patch_mlp=triton_ok,
                patch_attention=kernel_cfg.use_flash_attention and (flash_ok or triton_ok),
                patch_rope=kernel_cfg.use_fused_rope and triton_ok,
                patch_fp8=fp8_ok,
                verbose=True,
            )
            model = patcher.patch(model)
            applied_phases.append("layer_patching")
        except Exception as e:
            try:
                from data_pipeline.trainer.registry import patch_model
                model = patch_model(model)
                applied_phases.append("layer_patching_basic")
            except Exception as e2:
                logger.warning(f"Phase 2 — Layer patching failed: {e} → {e2}")

        # Phase 3: Fused cross-entropy
        if kernel_cfg.use_fused_cross_entropy and triton_ok:
            try:
                from data_pipeline.trainer.kernels import (
                    fast_cross_entropy_loss,
                    Fast_CrossEntropyLoss,
                )
                self._fused_cross_entropy_fn = fast_cross_entropy_loss
                self._fused_cross_entropy_cls = Fast_CrossEntropyLoss
                applied_phases.append("fused_cross_entropy")
            except Exception as e:
                logger.warning(f"Phase 3 — Fused CE failed: {e}")

        # Phase 4: Fused LoRA
        if kernel_cfg.use_fused_lora and triton_ok:
            try:
                from data_pipeline.trainer.kernels import (
                    LoRA_MLP, LoRA_QKV,
                    apply_lora_mlp_swiglu, apply_lora_qkv,
                    get_lora_parameters,
                )
                lora_patched = 0
                for name, module in layer_map["mlp_swiglu"]:
                    if self._has_lora_adapters(module.gate_proj):
                        gate_params = get_lora_parameters(module.gate_proj)
                        if gate_params[2] is not None:
                            module._fused_lora_fn = LoRA_MLP
                            module._lora_swiglu_fn = apply_lora_mlp_swiglu
                            module._uses_fused_lora = True
                            lora_patched += 1

                for name, module in layer_map["attention"]:
                    q_mod = (
                        getattr(module, 'q_proj', None)
                        or getattr(module, 'query', None)
                        or getattr(module, 'to_q', None)
                    )
                    if q_mod is not None and self._has_lora_adapters(q_mod):
                        q_params = get_lora_parameters(q_mod)
                        if q_params[2] is not None:
                            module._fused_lora_fn = LoRA_QKV
                            module._lora_qkv_fn = apply_lora_qkv
                            module._uses_fused_lora = True
                            lora_patched += 1

                if lora_patched > 0:
                    applied_phases.append("fused_lora")
                    logger.info(f"Phase 4 — Fused LoRA: {lora_patched} layers")
            except Exception as e:
                logger.warning(f"Phase 4 — Fused LoRA failed: {e}")

        # Phase 5: MoE kernels
        if kernel_cfg.use_moe_kernels and triton_ok:
            try:
                from data_pipeline.trainer.kernels import (
                    MoEKernelConfig, get_accelerator_arch,
                )
                arch = get_accelerator_arch()
                moe_config = MoEKernelConfig(
                    use_persistent_kernel=tma_ok,
                    l2_cache_above_first_wave=tma_ok,
                )
                for name, module in layer_map["moe"]:
                    module._moe_kernel_config = moe_config
                    module._accelerator_arch = arch
                    module._uses_triton_moe = True
                if layer_map["moe"]:
                    applied_phases.append("moe_kernels")
            except Exception as e:
                logger.warning(f"Phase 5 — MoE kernels failed: {e}")

        # Phase 6: FP8 quantization
        if fp8_ok and triton_ok and self.config.quantization.load_in_fp8:
            try:
                from data_pipeline.trainer.kernels import (
                    FP8Config, FP8Format, FP8ScaleManager,
                )
                fp8_cfg = FP8Config()
                fp8_layers = 0
                for name, module in layer_map["linear"]:
                    if not getattr(module, '_fp8_enabled', False):
                        module._fp8_config = fp8_cfg
                        module._fp8_format_fwd = FP8Format.E4M3
                        module._fp8_format_bwd = FP8Format.E5M2
                        module._fp8_enabled = True
                        module._fp8_scale_manager = FP8ScaleManager(config=fp8_cfg)
                        fp8_layers += 1
                if fp8_layers > 0:
                    applied_phases.append("fp8_quantization")
            except Exception as e:
                logger.warning(f"Phase 6 — FP8 failed: {e}")

        # Phase 7: Flex Attention
        if kernel_cfg.use_flash_attention and (flash_ok or triton_ok):
            try:
                from data_pipeline.trainer.kernels import (
                    FlexAttentionConfig, AttentionBackend, BackendCapabilities,
                )
                backend_caps = BackendCapabilities()
                attn_configured = 0
                for name, module in layer_map["attention"]:
                    flex_cfg = None
                    if hasattr(FlexAttentionConfig, 'PrecisionMode'):
                        precision = (
                            FlexAttentionConfig.PrecisionMode.FP8 if fp8_ok
                            else FlexAttentionConfig.PrecisionMode.BF16 if bf16_ok
                            else FlexAttentionConfig.PrecisionMode.FP16
                        )
                        flex_cfg = FlexAttentionConfig(causal=True, precision=precision)
                    if flex_cfg is not None:
                        module._flex_attention_backend = backend_caps.select_optimal_backend(flex_cfg)
                    elif flash_ok:
                        module._flex_attention_backend = AttentionBackend.FLASH_ATTENTION
                    elif triton_ok:
                        module._flex_attention_backend = AttentionBackend.TRITON_FUSED
                    else:
                        module._flex_attention_backend = AttentionBackend.SDPA_NATIVE
                    module._flex_attention_configured = True
                    attn_configured += 1
                if attn_configured > 0:
                    applied_phases.append("flex_attention")
            except Exception as e:
                logger.warning(f"Phase 7 — Flex Attention failed: {e}")

        # Phase 8: torch.compile (handled in setup_model after wrapping)
        # Phase 9: Distributed kernels
        if dist_cfg.enabled and cuda_ok:
            try:
                from data_pipeline.trainer.kernels import (
                    DistributedKernels,
                    DistributedConfig as KernelDistConfig,
                    DistributedBackend,
                )
                backend_val = None
                if hasattr(DistributedBackend, dist_cfg.backend.upper()):
                    backend_val = DistributedBackend(dist_cfg.backend)
                dist_kernel_config = KernelDistConfig(backend=backend_val)
                self._distributed_kernels = DistributedKernels(config=dist_kernel_config)
                applied_phases.append("distributed_kernels")
            except Exception as e:
                logger.warning(f"Phase 9 — Distributed kernels failed: {e}")

        # Phase 10: Summary
        self._kernel_optimization_phases = applied_phases
        self._kernel_capabilities = capabilities

        if applied_phases:
            logger.info(
                f"✓ Kernel optimizations ({len(applied_phases)}/10): "
                f"{', '.join(applied_phases)}"
            )
        else:
            logger.warning("⚠ No kernel optimizations applied")

        return model

    # ═════════════════════════════════════════════════════════════════════════════
    # Optimizer & Scheduler & Loss Setup
    # ═════════════════════════════════════════════════════════════════════════════

    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer based on config."""
        opt_cfg = self.config.optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt_type = opt_cfg.type

        if opt_type == OptimizerType.ADAMW:
            self.optimizer = torch.optim.AdamW(
                params, lr=opt_cfg.learning_rate,
                betas=opt_cfg.betas, eps=opt_cfg.eps,
                weight_decay=opt_cfg.weight_decay,
            )
        elif opt_type == OptimizerType.ADAM_8BIT:
            try:
                self.optimizer = Adam8bit(
                    params, lr=opt_cfg.learning_rate,
                    betas=opt_cfg.betas, eps=opt_cfg.eps,
                    weight_decay=opt_cfg.weight_decay,
                    percentile_clipping=opt_cfg.percentile_clipping,
                )
            except ImportError:
                logger.warning("8-bit Adam unavailable, using AdamW")
                self.optimizer = torch.optim.AdamW(params, lr=opt_cfg.learning_rate)
        elif opt_type == OptimizerType.LION:
            try:
                self.optimizer = Lion(
                    params, lr=opt_cfg.learning_rate,
                    betas=opt_cfg.lion_betas,
                    weight_decay=opt_cfg.weight_decay,
                )
            except ImportError:
                logger.warning("Lion unavailable, using AdamW")
                self.optimizer = torch.optim.AdamW(params, lr=opt_cfg.learning_rate)
        elif opt_type == OptimizerType.FUSED_ADAMW:
            try:
                self.optimizer = FusedAdamW(
                    params, lr=opt_cfg.learning_rate,
                    betas=opt_cfg.betas, eps=opt_cfg.eps,
                    weight_decay=opt_cfg.weight_decay,
                    max_grad_norm=opt_cfg.max_grad_norm,
                )
            except ImportError:
                self.optimizer = torch.optim.AdamW(params, lr=opt_cfg.learning_rate)
        else:
            self.optimizer = torch.optim.AdamW(
                params, lr=opt_cfg.learning_rate,
                weight_decay=opt_cfg.weight_decay,
            )

        return self.optimizer

    def setup_scheduler(self, num_training_steps: int) -> Any:
        """Setup learning rate scheduler."""
        sch_cfg = self.config.scheduler
        warmup_steps = sch_cfg.warmup_steps
        if warmup_steps == 0:
            warmup_steps = int(sch_cfg.warmup_ratio * num_training_steps)

        sch_type = sch_cfg.type

        if sch_type == SchedulerType.COSINE:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - warmup_steps,
                eta_min=self.config.optimizer.learning_rate * sch_cfg.min_lr_ratio,
            )
        elif sch_type == SchedulerType.WSD:
            try:
                self.scheduler = WSDScheduler(
                    self.optimizer,
                    num_training_steps=num_training_steps,
                    warmup_steps=warmup_steps,
                    stable_steps=int(sch_cfg.stable_ratio * num_training_steps),
                    min_lr_ratio=sch_cfg.min_lr_ratio,
                    decay_type=sch_cfg.decay_type,
                )
            except ImportError:
                from torch.optim.lr_scheduler import CosineAnnealingLR
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, T_max=num_training_steps,
                )
        elif sch_type == SchedulerType.LINEAR:
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                self.optimizer, start_factor=1.0,
                end_factor=sch_cfg.min_lr_ratio,
                total_iters=num_training_steps,
            )
        else:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=num_training_steps,
            )

        return self.scheduler

    def setup_loss(self) -> nn.Module:
        """Setup loss function."""
        loss_cfg = self.config.loss
        loss_type = loss_cfg.type

        if loss_type == LossType.CROSS_ENTROPY:
            self.loss_fn = nn.CrossEntropyLoss(
                ignore_index=loss_cfg.ignore_index,
                label_smoothing=loss_cfg.label_smoothing,
                reduction=loss_cfg.reduction,
            )
        elif loss_type == LossType.CHUNKED_CE:
            try:
                self.loss_fn = ChunkedCrossEntropyLoss(
                    ignore_index=loss_cfg.ignore_index,
                    label_smoothing=loss_cfg.label_smoothing,
                    reduction=loss_cfg.reduction,
                    chunk_size=loss_cfg.chunk_size,
                )
            except ImportError:
                self.loss_fn = nn.CrossEntropyLoss(ignore_index=loss_cfg.ignore_index)
        elif loss_type == LossType.FOCAL:
            try:
                self.loss_fn = FocalLoss(
                    gamma=loss_cfg.focal_gamma,
                    alpha=loss_cfg.focal_alpha,
                    ignore_index=loss_cfg.ignore_index,
                    reduction=loss_cfg.reduction,
                )
            except ImportError:
                self.loss_fn = nn.CrossEntropyLoss(ignore_index=loss_cfg.ignore_index)
        elif loss_type == LossType.DPO:
            self.loss_fn = DPOLoss(
                beta=loss_cfg.dpo_beta,
                label_smoothing=loss_cfg.label_smoothing,
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=loss_cfg.ignore_index)

        return self.loss_fn

    # ═════════════════════════════════════════════════════════════════════════════
    # Training Loop — FSDP2-Integrated
    # ═════════════════════════════════════════════════════════════════════════════

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        compute_metrics: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Run training loop.

        When FSDP2 is active:
            - Forward: uses fsdp2_engine.forward_context() for autocast + metrics
            - Backward: uses fsdp2_engine.backward() for no_sync + accumulation
            - Step: uses fsdp2_engine.step() for clip→step→zero_grad
            - Trainer does NOT manage scaler, no_sync, or grad accumulation

        When FSDP2 is NOT active (DDP, single-GPU, ZBPP, CP):
            - Original behavior with trainer-managed backward/step
        """
        # Setup components
        if self.optimizer is None:
            self.setup_optimizer()

        num_training_steps = (
            len(train_dataloader)
            // self.config.training.gradient_accumulation_steps
            * self.config.training.num_train_epochs
        )

        if self.scheduler is None:
            self.setup_scheduler(num_training_steps)

        if self.loss_fn is None:
            self.setup_loss()

        train_cfg = self.config.training
        grad_accum = train_cfg.gradient_accumulation_steps
        max_grad_norm = self.config.optimizer.max_grad_norm

        # Trackers
        loss_tracker = LossTracker(ema_decay=0.99)
        acc_tracker = AccuracyTracker()

        self.model.train()
        metrics: Dict[str, float] = {"train_loss": 0.0}
        self.start_time = time.time()

        if eval_dataloader is not None and (
            not dist.is_initialized() or dist.get_rank() == 0
        ):
            try:
                logger.info(f"Eval DataLoader: {len(eval_dataloader)} batches")
            except Exception:
                logger.info("Eval DataLoader ready")

        for epoch in range(train_cfg.num_train_epochs):
            self.state.epoch = epoch

            for step, batch in enumerate(train_dataloader):
                step_start_time = time.time()

                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                }

                # ═══════════════════════════════════════════════════════════
                # ZBPP Pipeline — Self-contained train_step
                # ═══════════════════════════════════════════════════════════
                if self.distributed_strategy == "pipeline_zbpp":
                    num_mb = self.config.distributed.num_microbatches
                    batch_size = batch["input_ids"].size(0)
                    if batch_size < num_mb:
                        num_mb = batch_size
                    mb_size = batch_size // num_mb
                    if mb_size == 0:
                        logger.warning(
                            f"Batch size {batch_size} too small for {num_mb} μB"
                        )
                        continue

                    micro_inputs = list(batch["input_ids"].split(mb_size))
                    micro_labels = list(batch["labels"].split(mb_size))

                    try:
                        outputs = self.model.train_step(
                            micro_batches=micro_inputs,
                            labels=micro_labels,
                            loss_fn=self.loss_fn,
                        )
                    except Exception as e:
                        logger.error(f"ZBPP train_step failed: {e}")
                        continue

                    loss = outputs.get("loss", torch.tensor(0.0))
                    if not self._check_numerical_stability(
                        loss, step=self.state.global_step
                    ):
                        logger.warning(f"ZBPP numerical issue at step {self.state.global_step}")

                    num_tokens = sum(
                        (labels != -100).sum().item() for labels in micro_labels
                    )
                    loss_tracker.update(loss, num_tokens=max(1, num_tokens))

                    if self.scheduler:
                        self.scheduler.step()

                    self.state.global_step += 1
                    continue

                # ═══════════════════════════════════════════════════════════
                # FSDP2 Path — Delegated to engine
                # [INT-003] [INT-004] [INT-007] [INT-010]
                # ═══════════════════════════════════════════════════════════
                if self._is_fsdp2_active:
                    # Forward with FSDP2's autocast + metrics
                    with self._fsdp2_engine.forward_context():
                        outputs = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask"),
                            labels=batch.get("labels"),
                        )

                        if hasattr(outputs, "loss") and outputs.loss is not None:
                            loss = outputs.loss
                        else:
                            logits = outputs.logits
                            labels = batch["labels"]
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            loss = self.loss_fn(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                            )

                    # Numerical stability
                    if not self._check_numerical_stability(
                        loss, step=self.state.global_step
                    ):
                        logger.warning(
                            f"Skipping step {self.state.global_step} "
                            f"(numerical instability)"
                        )
                        self.optimizer.zero_grad(set_to_none=True)
                        continue

                    # Scale for accumulation (FSDP2 handles the actual scaling
                    # for fp16 via its MixedPrecisionContext)
                    loss_for_backward = loss / grad_accum

                    # [INT-003] Delegate backward to FSDP2 engine
                    # Returns True on sync step (should call step())
                    is_sync_step = self._fsdp2_engine.backward(loss_for_backward)

                    # Update metrics
                    num_tokens = (batch.get("labels", batch["input_ids"]) != -100).sum().item()
                    loss_tracker.update(loss, num_tokens=max(1, num_tokens))
                    if hasattr(outputs, "logits"):
                        acc_tracker.update_from_logits(
                            outputs.logits, batch.get("labels"),
                        )

                    # [INT-004] Delegate step to FSDP2 engine
                    if is_sync_step:
                        self._fsdp2_engine.step(
                            self.optimizer,
                            scheduler=self.scheduler,
                        )
                        self.state.global_step += 1

                        # Logging
                        self._log_training_step(
                            loss_tracker, acc_tracker, num_tokens,
                            grad_accum, num_training_steps, epoch,
                            train_cfg, step_start_time,
                        )

                        # Evaluation
                        if (
                            eval_dataloader is not None
                            and train_cfg.eval_strategy == "steps"
                            and self.state.global_step % train_cfg.eval_steps == 0
                        ):
                            eval_metrics = self.evaluate(
                                eval_dataloader, compute_metrics
                            )
                            logger.info(f"Eval: {eval_metrics}")
                            if self._check_early_stopping(eval_metrics):
                                logger.info("Early stopping triggered.")
                                self._restore_best_model(train_cfg, metrics)
                                return metrics

                        # Checkpointing
                        if (
                            train_cfg.save_strategy == "steps"
                            and self.state.global_step % train_cfg.save_steps == 0
                        ):
                            self.save_checkpoint()

                    # Max steps check
                    if (
                        train_cfg.max_steps > 0
                        and self.state.global_step >= train_cfg.max_steps
                    ):
                        break

                    continue  # Skip non-FSDP2 backward/step logic

                # ═══════════════════════════════════════════════════════════
                # Standard Path — DDP / Single-GPU / Context Parallel
                # ═══════════════════════════════════════════════════════════
                with self.amp_context():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch.get("labels"),
                    )

                    if hasattr(outputs, "loss") and outputs.loss is not None:
                        loss = outputs.loss
                        if hasattr(outputs, "logits"):
                            num_tokens = (batch["labels"] != -100).sum().item()
                            loss_tracker.update(loss, num_tokens=num_tokens)
                            acc_tracker.update_from_logits(
                                outputs.logits, batch["labels"],
                            )
                    else:
                        logits = outputs.logits
                        labels = batch["labels"]
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = self.loss_fn(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        )
                        num_tokens = (labels != -100).sum().item()
                        loss_tracker.update(loss, num_tokens=num_tokens)
                        acc_tracker.update_from_logits(logits, labels)

                    loss_scaled = loss / grad_accum

                # Numerical stability
                if not self._check_numerical_stability(
                    loss, step=self.state.global_step
                ):
                    logger.warning(
                        f"Skipping step {self.state.global_step} "
                        f"(numerical instability)"
                    )
                    self.optimizer.zero_grad()
                    continue

                # [INT-010] no_sync for DDP only (FSDP2 handles internally)
                should_sync = (step + 1) % grad_accum == 0
                sync_context = self._get_no_sync_context(should_sync)

                with sync_context:
                    if self.scaler is not None:
                        self.scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()

                # Optimizer step
                if should_sync:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)

                    grad_norm = 0.0
                    if max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm,
                        )
                    elif hasattr(self.optimizer, "get_grad_norm"):
                        grad_norm = self.optimizer.get_grad_norm()

                    grad_norm_val = (
                        grad_norm
                        if isinstance(grad_norm, float)
                        else grad_norm.item()
                    )
                    if not self._check_numerical_stability(
                        loss, grad_norm=grad_norm_val, step=self.state.global_step
                    ):
                        logger.warning("Skipping optimizer step (gradient instability)")
                        self.optimizer.zero_grad()
                        self.state.global_step += 1
                        continue

                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.state.global_step += 1

                    # Logging
                    self._log_training_step(
                        loss_tracker, acc_tracker, num_tokens,
                        grad_accum, num_training_steps, epoch,
                        train_cfg, step_start_time,
                        grad_norm_override=grad_norm_val,
                    )

                    # Evaluation
                    if (
                        eval_dataloader is not None
                        and train_cfg.eval_strategy == "steps"
                        and self.state.global_step % train_cfg.eval_steps == 0
                    ):
                        eval_metrics = self.evaluate(
                            eval_dataloader, compute_metrics
                        )
                        logger.info(f"Eval: {eval_metrics}")
                        if self._check_early_stopping(eval_metrics):
                            logger.info("Early stopping triggered.")
                            self._restore_best_model(train_cfg, metrics)
                            return metrics

                    # Checkpointing
                    if (
                        train_cfg.save_strategy == "steps"
                        and self.state.global_step % train_cfg.save_steps == 0
                    ):
                        self.save_checkpoint()

                # Max steps
                if (
                    train_cfg.max_steps > 0
                    and self.state.global_step >= train_cfg.max_steps
                ):
                    break

            # End of epoch
            metrics["train_loss"] = loss_tracker.compute_avg()
            logger.info(
                f"Epoch {epoch + 1} | Loss: {metrics['train_loss']:.4f}"
            )

            if eval_dataloader is not None and train_cfg.eval_strategy == "epoch":
                eval_metrics = self.evaluate(eval_dataloader, compute_metrics)
                metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
                if self._check_early_stopping(eval_metrics):
                    logger.info("Early stopping triggered.")
                    break

            if train_cfg.save_strategy == "epoch":
                self.save_checkpoint()

        # End of training — restore best
        self._restore_best_model(train_cfg, metrics)
        return metrics

    # ═════════════════════════════════════════════════════════════════════════════
    # Logging Helper — Shared Between FSDP2 and Standard Paths
    # ═════════════════════════════════════════════════════════════════════════════

    def _log_training_step(
        self,
        loss_tracker: LossTracker,
        acc_tracker: AccuracyTracker,
        num_tokens: int,
        grad_accum: int,
        num_training_steps: int,
        epoch: int,
        train_cfg: Any,
        step_start_time: float,
        grad_norm_override: Optional[float] = None,
    ) -> None:
        """Unified logging for both FSDP2 and standard training paths."""
        is_main = not dist.is_initialized() or dist.get_rank() == 0

        if self.state.global_step % train_cfg.logging_steps != 0:
            return

        loss_val = loss_tracker.compute_ema()
        if loss_val == 0.0:
            loss_val = loss_tracker.compute_avg()

        acc = acc_tracker.compute()

        # [INT-009] Get gradient norm from FSDP2 metrics if available
        if self._is_fsdp2_active:
            grad_norm_val = self._fsdp2_engine.metrics.current.gradient_norm
        elif grad_norm_override is not None:
            grad_norm_val = grad_norm_override
        else:
            grad_norm_val = 0.0

        # Reduce metrics across ranks
        metrics_to_reduce = {
            "loss": loss_val,
            "acc": acc,
            "grad": grad_norm_val,
        }

        if dist.is_initialized():
            tensor_vals = torch.tensor(
                [metrics_to_reduce["loss"],
                 metrics_to_reduce["acc"],
                 metrics_to_reduce["grad"]],
                device=self.device,
            )
            dist.all_reduce(tensor_vals, op=dist.ReduceOp.AVG)
            metrics_to_reduce["loss"] = tensor_vals[0].item()
            metrics_to_reduce["acc"] = tensor_vals[1].item()
            metrics_to_reduce["grad"] = tensor_vals[2].item()

        if is_main:
            ppl = torch.exp(torch.tensor(metrics_to_reduce["loss"])).item()
            lr = self.scheduler.get_last_lr()[0]

            current_time = time.time()
            elapsed = current_time - self.start_time
            steps_done = self.state.global_step
            avg_time_per_step = elapsed / steps_done if steps_done > 0 else 0
            remaining_steps = num_training_steps - steps_done
            eta_seconds = int(avg_time_per_step * remaining_steps)
            eta_str = str(datetime.timedelta(seconds=eta_seconds))

            step_duration = current_time - step_start_time
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            tok_per_sec = int(
                num_tokens * grad_accum * world_size / max(step_duration, 1e-6)
            )

            ppl_str = f"PPL: {ppl:.2e}" if ppl > 1000 else f"PPL: {ppl:.2f}"

            # [INT-009] Append FSDP2 memory stats if available
            mem_str = ""
            if self._is_fsdp2_active:
                mem_str = f" | {self._fsdp2_engine.memory_summary()}"

            logger.info(
                f"Epoch {epoch+1}/{train_cfg.num_train_epochs} | "
                f"Step {steps_done}/{num_training_steps} | "
                f"Loss: {metrics_to_reduce['loss']:.4f} | "
                f"{ppl_str} | "
                f"Acc: {metrics_to_reduce['acc']:.2%} | "
                f"LR: {lr:.2e} | "
                f"Grad: {metrics_to_reduce['grad']:.2f} | "
                f"Tok/s: {tok_per_sec} | "
                f"ETA: {eta_str}{mem_str}"
            )
            acc_tracker.reset()

    def _restore_best_model(
        self,
        train_cfg: Any,
        metrics: Dict[str, float],
    ) -> None:
        """Restore best model checkpoint if available."""
        best_path = os.path.join(train_cfg.output_dir, "checkpoint-best")
        if os.path.exists(best_path):
            logger.info(f"Restoring best model from {best_path}...")
            self.load_checkpoint(best_path)
            metrics["best_metric"] = self.state.best_metric

    def _training_step(self, batch: Dict[str, Tensor]) -> Tensor:
        """Single training step (used by evaluate)."""
        batch = {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )

        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss

        logits = outputs.logits
        labels = batch["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss

    def evaluate(
        self,
        eval_dataloader: DataLoader,
        compute_metrics: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Run evaluation with distributed synchronization."""
        if dist.is_initialized():
            dist.barrier()

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # [INT-007] Use FSDP2's forward_context for autocast during eval
        eval_context = (
            self._fsdp2_engine.forward_context
            if self._is_fsdp2_active
            else self.amp_context
        )

        with torch.no_grad():
            for batch in eval_dataloader:
                with eval_context():
                    loss = self._training_step(batch)
                total_loss += loss.item()
                num_batches += 1

                if (
                    self.config.training.max_steps > 0
                    and self.config.training.max_steps < 100
                    and num_batches >= 5
                ):
                    break

        self.model.train()

        # Aggregate across ranks
        if dist.is_initialized():
            tensor_agg = torch.tensor(
                [total_loss, float(num_batches)], device=self.device,
            )
            dist.all_reduce(tensor_agg, op=dist.ReduceOp.SUM)
            global_total_loss = tensor_agg[0].item()
            global_num_batches = tensor_agg[1].item()
        else:
            global_total_loss = total_loss
            global_num_batches = num_batches

        avg_loss = (
            global_total_loss / global_num_batches
            if global_num_batches > 0
            else 0.0
        )

        eval_metrics = {"loss": avg_loss}
        try:
            eval_metrics["ppl"] = math.exp(avg_loss)
        except OverflowError:
            eval_metrics["ppl"] = float("inf")

        if compute_metrics is not None:
            eval_metrics.update(compute_metrics(self.model, eval_dataloader))

        return eval_metrics

    def _check_early_stopping(self, eval_metrics: Dict[str, float]) -> bool:
        """Check if training should stop early."""
        current_loss = eval_metrics.get("loss", float('inf'))
        patience = self.config.training.early_stopping_patience
        threshold = self.config.training.early_stopping_threshold

        if current_loss < (self.state.best_metric - threshold):
            self.state.best_metric = current_loss
            self.state.patience_counter = 0

            if not dist.is_initialized() or dist.get_rank() == 0:
                best_path = os.path.join(
                    self.config.training.output_dir, "checkpoint-best"
                )
                logger.info(
                    f"New best model (Loss: {current_loss:.4f}). "
                    f"Saving to {best_path}..."
                )
                self.save_checkpoint(path=best_path)
            return False

        self.state.patience_counter += 1
        logger.info(
            f"⏳ Early Stop: {self.state.patience_counter}/{patience} "
            f"(Best: {self.state.best_metric:.4f})"
        )

        return self.state.patience_counter >= patience

    # ═════════════════════════════════════════════════════════════════════════════
    # Checkpointing — FSDP2-Aware
    # ═════════════════════════════════════════════════════════════════════════════

    def save_checkpoint(self, path: Optional[str] = None):
        """
        Save training checkpoint for all distributed strategies.

        [INT-006] FSDP2 path uses FSDPCheckpointManager.save_checkpoint()
        (correct API, not the non-existent save_sharded_checkpoint).
        """
        if path is None:
            path = os.path.join(
                self.config.training.output_dir,
                f"checkpoint-{self.state.global_step}",
            )

        os.makedirs(path, exist_ok=True)

        # ── ZBPP Pipeline ──
        if self.distributed_strategy == "pipeline_zbpp":
            rank = dist.get_rank() if dist.is_initialized() else 0
            log_rank_0(f"Saving ZBPP checkpoint to {path}")

            if hasattr(self.model, 'save_stage_checkpoint'):
                self.model.save_stage_checkpoint(path)
            else:
                stage_path = os.path.join(path, f"stage_{rank}_model.pt")
                model_to_save = (
                    self.model.module
                    if hasattr(self.model, 'module')
                    else self.model
                )
                torch.save(model_to_save.state_dict(), stage_path)

            if hasattr(self.model, "_optimizer"):
                opt_path = os.path.join(path, f"stage_{rank}_optimizer.pt")
                torch.save(self.model._optimizer.state_dict(), opt_path)

            if rank == 0:
                self._save_trainer_state(path)

            if dist.is_initialized():
                dist.barrier()
            return

        # ── Context Parallel ──
        if self.distributed_strategy == "context_parallel" and dist.is_initialized():
            dist.barrier()

        # ── FSDP2 ──
        # [INT-006] Use FSDP2's checkpoint manager with correct API
        if self._is_fsdp2_active:
            log_rank_0(f"Saving FSDP2 checkpoint to {path}")
            result = FSDPCheckpointManager.save_checkpoint(
                fsdp=self._fsdp2_engine,
                optimizer=self.optimizer,
                path=path,
                epoch=self.state.epoch,
                step=self.state.global_step,
                extra_state={
                    "best_metric": self.state.best_metric,
                    "patience_counter": self.state.patience_counter,
                    "samples_seen": self.state.samples_seen,
                },
                sharded=True,
            )
            if result.is_err():
                logger.error(f"FSDP2 checkpoint save failed: {result.error}")
            else:
                log_rank_0(f"FSDP2 checkpoint saved: {path}")

            # Save scheduler and RNG state separately
            if self._fsdp2_engine.is_rank_zero:
                self._save_trainer_state(path)
            return

        # ── Fallback: check raw FSDP (shouldn't happen with proper setup) ──
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(self.model, FSDP):
            logger.warning(
                "Model is raw FSDP without FSDP2 engine. "
                "This shouldn't happen — check setup_model()."
            )
            # Best-effort save
            self._save_trainer_state(path)
            return

        # ── Standard DDP / Single GPU ──
        model_to_save = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            log_rank_0(f"Saving checkpoint to {path}")

            if hasattr(model_to_save, 'save_pretrained'):
                model_to_save.save_pretrained(path)
            else:
                torch.save(
                    model_to_save.state_dict(),
                    os.path.join(path, "model.pt"),
                )

            self._save_trainer_state(path)
            log_rank_0(f"Checkpoint saved: {path}")

    def _save_trainer_state(self, path: str):
        """Save trainer state, scheduler, RNG."""
        torch.save({
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "state": self.state,
            "config": self.config.to_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
        }, os.path.join(path, "trainer_state.pt"))

    def load_checkpoint(self, path: str):
        """
        Load training checkpoint for all distributed strategies.

        [INT-006] FSDP2 path uses FSDPCheckpointManager.load_checkpoint()
        (correct API).
        """
        log_rank_0(f"Loading checkpoint from {path}")

        # ── ZBPP ──
        if self.distributed_strategy == "pipeline_zbpp":
            rank = dist.get_rank() if dist.is_initialized() else 0
            if hasattr(self.model, 'load_stage_checkpoint'):
                self.model.load_stage_checkpoint(path)
            else:
                stage_path = os.path.join(path, f"stage_{rank}_model.pt")
                if os.path.exists(stage_path):
                    model_to_load = (
                        self.model.module
                        if hasattr(self.model, 'module')
                        else self.model
                    )
                    model_to_load.load_state_dict(
                        torch.load(stage_path, map_location=self.device)
                    )

            if hasattr(self.model, "_optimizer"):
                opt_path = os.path.join(path, f"stage_{rank}_optimizer.pt")
                if os.path.exists(opt_path):
                    self.model._optimizer.load_state_dict(
                        torch.load(opt_path, map_location=self.device)
                    )

            self._load_trainer_state(path)
            if dist.is_initialized():
                dist.barrier()
            return

        # ── FSDP2 ──
        if self._is_fsdp2_active:
            log_rank_0(f"Loading FSDP2 checkpoint from {path}")
            result = FSDPCheckpointManager.load_checkpoint(
                fsdp=self._fsdp2_engine,
                optimizer=self.optimizer,
                path=path,
                sharded=True,
            )
            if result.is_err():
                logger.error(f"FSDP2 checkpoint load failed: {result.error}")
            else:
                meta = result.unwrap()
                self.state.epoch = meta.get("epoch", self.state.epoch)
                self.state.global_step = meta.get("step", self.state.global_step)
                extra = meta.get("extra", {})
                self.state.best_metric = extra.get(
                    "best_metric", self.state.best_metric
                )
                self.state.patience_counter = extra.get(
                    "patience_counter", self.state.patience_counter
                )
                log_rank_0(f"FSDP2 checkpoint loaded: {path}")

            # Load scheduler and RNG
            self._load_trainer_state(path)
            return

        # ── Raw FSDP fallback ──
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(self.model, FSDP):
            logger.warning("Loading raw FSDP checkpoint without engine")
            self._load_trainer_state(path)
            return

        # ── Context Parallel ──
        if self.distributed_strategy == "context_parallel" and dist.is_initialized():
            dist.barrier()

        # ── Standard DDP / Single GPU ──
        self._load_trainer_state(path)

        if hasattr(self.model, 'load_adapter'):
            try:
                self.model.load_adapter(path, adapter_name="default")
            except Exception as e:
                logger.warning(f"load_adapter failed: {e}")
                try:
                    self.model.load_adapter(path)
                except Exception as e2:
                    logger.warning(f"Adapter load failed: {e2}")
        else:
            model_path = os.path.join(path, "model.pt")
            if os.path.exists(model_path):
                model_to_load = (
                    self.model.module
                    if hasattr(self.model, 'module')
                    else self.model
                )
                model_to_load.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )

        log_rank_0(f"Checkpoint loaded: {path}")

    def _load_trainer_state(self, path: str):
        """Load trainer state, scheduler, RNG."""
        state_path = os.path.join(path, "trainer_state.pt")
        if not os.path.exists(state_path):
            logger.warning(f"Trainer state not found: {state_path}")
            return

        checkpoint = torch.load(
            state_path, map_location="cpu", weights_only=False,
        )
        self.state = checkpoint.get("state", self.state)

        if self.optimizer and checkpoint.get("optimizer"):
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")

        if self.scheduler and checkpoint.get("scheduler"):
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            except Exception as e:
                logger.warning(f"Failed to load scheduler state: {e}")

        if "rng_state" in checkpoint:
            self._restore_rng_state(checkpoint["rng_state"])

    def _restore_rng_state(self, rng: Dict[str, Any]):
        """Restore RNG states for deterministic resumption."""
        try:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(
                rng["torch"].cpu()
                if hasattr(rng["torch"], 'cpu')
                else rng["torch"]
            )
            if rng.get("cuda") is not None and torch.cuda.is_available():
                cuda_states = rng["cuda"]
                if isinstance(cuda_states, list):
                    torch.cuda.set_rng_state_all([
                        t.cpu() if hasattr(t, 'cpu') else t
                        for t in cuda_states
                    ])
                else:
                    torch.cuda.set_rng_state(
                        cuda_states.cpu()
                        if hasattr(cuda_states, 'cpu')
                        else cuda_states
                    )
        except Exception as e:
            logger.warning(f"Failed to restore RNG state: {e}")

    # ═════════════════════════════════════════════════════════════════════════════
    # Export
    # ═════════════════════════════════════════════════════════════════════════════

    def export(self, tokenizer=None) -> str:
        """Export model based on config."""
        export_cfg = self.config.export
        if not export_cfg.enabled:
            logger.warning("Export not enabled")
            return ""

        from data_pipeline.trainer.export import (
            save_safetensors,
            export_to_gguf,
            push_to_hub,
            merge_lora_weights,
        )

        output_dir = export_cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # For FSDP2: materialize full params before export
        if self._is_fsdp2_active:
            log_rank_0("Gathering full parameters for export...")
            export_model = self.model
            export_context = self._fsdp2_engine.summon_full_params(writeback=False)
        else:
            export_model = (
                self.model.module
                if hasattr(self.model, "module")
                else self.model
            )
            export_context = nullcontext()

        with export_context:
            # Merge LoRA if needed
            if (
                export_cfg.merge_lora
                and self.config.training_mode in (TrainingMode.LORA, TrainingMode.QLORA)
            ):
                export_model = merge_lora_weights(export_model)

            if export_cfg.format == ExportFormat.SAFETENSORS:
                save_safetensors(export_model, output_dir)
            elif export_cfg.format in (
                ExportFormat.GGUF_Q4, ExportFormat.GGUF_Q5,
                ExportFormat.GGUF_Q8, ExportFormat.GGUF_F16,
            ):
                if tokenizer is None:
                    raise ValueError("Tokenizer required for GGUF export")
                export_to_gguf(
                    export_model, tokenizer, output_dir,
                    quantization=export_cfg.gguf_quantization,
                )

            if export_cfg.push_to_hub and export_cfg.hub_model_id:
                push_to_hub(
                    export_model, tokenizer,
                    export_cfg.hub_model_id,
                    token=export_cfg.hub_token,
                    private=export_cfg.hub_private,
                )

        logger.info(f"Model exported: {output_dir}")
        return output_dir


# ═════════════════════════════════════════════════════════════════════════════════
# Factory Function
# ═════════════════════════════════════════════════════════════════════════════════

def create_trainer(config_path: Union[str, Path]) -> SOTATrainer:
    """
    Create trainer from YAML config.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configured SOTATrainer instance
    """
    config = SOTAConfig.from_yaml(config_path)
    return SOTATrainer(config)


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    "SOTATrainer",
    "TrainingState",
    "create_trainer",
]
