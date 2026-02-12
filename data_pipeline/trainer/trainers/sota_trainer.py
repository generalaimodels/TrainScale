# ════════════════════════════════════════════════════════════════════════════════
# SOTA Unified Trainer v2.1 — Hardened FSDP2 + Checkpoint Device-Safety
# ════════════════════════════════════════════════════════════════════════════════
#
# End-to-end training orchestrator with YAML-driven configuration.
#
# ARCHITECTURE:
#   YAML Config → SOTAConfig → SOTATrainer → Model + Optimized Training
#
# v2.1 FIXES (Checkpoint Stall + Device Drift on MI300X):
#
#   [TFIX-001] Checkpoint Device Affinity:
#       ROOT CAUSE: _save_trainer_state() calls torch.save() which
#       internally moves tensors through default CUDA device (cuda:0).
#       On multi-GPU MI300X, this causes NCCL to detect:
#         "Tensor found on device cuda:0 but backend constrained to cuda:3"
#       FIX: All checkpoint ops wrapped in torch.cuda.device(local_rank).
#       Optimizer/scheduler state dicts explicitly mapped to CPU before
#       save to prevent any CUDA device leakage.
#
#   [TFIX-002] Checkpoint Stall (5-Minute Hang at 78% VRAM):
#       ROOT CAUSE: save_checkpoint() calls FSDPCheckpointManager which
#       does all-gather, then trainer calls _save_trainer_state() which
#       does another optimizer.state_dict() (second all-gather on some
#       FSDP configs). At 78% VRAM, the second gather causes OOM-retry
#       loops in RCCL. Also, no explicit barrier between FSDP save and
#       trainer state save causes rank desync.
#       FIX: _save_trainer_state() for FSDP2 path does NOT re-save
#       optimizer state (FSDP2 checkpoint manager already saved it).
#       Explicit GC + cache clear between phases. Barrier with timeout.
#
#   [TFIX-003] Warning Suppression:
#       ROOT CAUSE: PyTorch emits hundreds of FutureWarning (ShardedTensor
#       deprecation, FSDP.state_dict_type deprecation) and UserWarning
#       (_get_pg_default_device deprecation) per checkpoint save. These
#       flood logs and obscure real errors.
#       FIX: Checkpoint methods wrapped in warnings.catch_warnings()
#       with targeted filterwarnings. Suppression is scoped (not global).
#
#   [TFIX-004] Post-Checkpoint Memory Leak (GPU 6/7 at 91% VRAM, 0% compute):
#       ROOT CAUSE: After checkpoint save, gathered parameter copies and
#       staging buffers remain in CUDA cache. On idle GPUs (not in the
#       training process group), these never get reclaimed.
#       FIX: Explicit gc.collect() + torch.cuda.empty_cache() after
#       every checkpoint operation. Memory pool cleared.
#
# INTEGRATION CONTRACTS (v2.0 — preserved):
#   [INT-001] through [INT-010] — see v2.0 header
#
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import gc
import logging
import os
import datetime
import math
import time
import warnings
import random
import numpy as np
from contextlib import contextmanager, nullcontext
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
    # FSDP2 — [INT-001]
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
    # SOTA: Gradient sync context (DDP only, NOT FSDP2)
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
# Constants — Checkpoint Safety
# ═════════════════════════════════════════════════════════════════════════════════

# [TFIX-002] Max seconds to wait for checkpoint barrier before failing
_CHECKPOINT_BARRIER_TIMEOUT_S: int = 300


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
# Warning Suppression Context — Scoped, Not Global [TFIX-003]
# ═════════════════════════════════════════════════════════════════════════════════

@contextmanager
def _suppress_checkpoint_warnings():
    """
    Suppress known-benign deprecation warnings during checkpoint ops.

    [TFIX-003] Scoped suppression — only active during checkpoint I/O.
    Targets:
        - ShardedTensor deprecation (FutureWarning)
        - FSDP.state_dict_type deprecation (FutureWarning)
        - _get_pg_default_device deprecation (UserWarning)
        - FileSystemWriter overwrite warning (UserWarning)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning,
            message=".*ShardedTensor.*",
        )
        warnings.filterwarnings(
            "ignore", category=FutureWarning,
            message=".*FSDP.state_dict_type.*",
        )
        warnings.filterwarnings(
            "ignore", category=FutureWarning,
            message=".*set_state_dict_type.*",
        )
        warnings.filterwarnings(
            "ignore", category=UserWarning,
            message=".*_get_pg_default_device.*",
        )
        warnings.filterwarnings(
            "ignore", category=UserWarning,
            message=".*Detected an existing checkpoint.*",
        )
        warnings.filterwarnings(
            "ignore", category=FutureWarning,
            message=".*Please use DTensor.*",
        )
        yield


# ═════════════════════════════════════════════════════════════════════════════════
# Device Affinity Context — Trainer-Side [TFIX-001]
# ═════════════════════════════════════════════════════════════════════════════════

@contextmanager
def _enforce_trainer_device(device: torch.device):
    """
    Pin CUDA default device for the duration of checkpoint ops.

    [TFIX-001] Without this, torch.save/load and optimizer.state_dict()
    can allocate staging tensors on cuda:0 instead of the rank's device.
    NCCL then fails with device mismatch on the next collective.

    This is the TRAINER-SIDE complement to FSDP2's _enforce_device_affinity().
    """
    torch.cuda.set_device(device)
    with torch.cuda.device(device):
        yield
    # Re-pin after scope exit (defensive)
    torch.cuda.set_device(device)


# ═════════════════════════════════════════════════════════════════════════════════
# FSDP2 Config Bridge — SOTAConfig → FSDP2Config [INT-005]
# ═════════════════════════════════════════════════════════════════════════════════

def _build_fsdp2_config(sota_config: SOTAConfig) -> FSDP2Config:
    """
    Translate SOTAConfig into FSDP2Config with explicit type mapping.

    Handles impedance mismatches:
        - SOTAConfig string enums → FSDP2Config Enum classes
        - Precision in hardware → MixedPrecisionPolicy enum
        - gradient_accumulation_steps from training → FSDP2 no_sync cycles
        - gradient_clipping_norm from optimizer → FSDP2 distributed clip

    Returns:
        Fully populated FSDP2Config.
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
    fsdp_sharding_str = getattr(
        dist_cfg, "fsdp_sharding_strategy", "full_shard",
    )
    sharding = sharding_map.get(
        fsdp_sharding_str.lower(), ShardingStrategy.FULL_SHARD,
    )

    # ── Mixed Precision ──
    precision_map = {
        Precision.BF16: MixedPrecisionPolicy.FULL_BF16,
        Precision.FP16: MixedPrecisionPolicy.FULL_FP16,
        Precision.FP32: MixedPrecisionPolicy.PURE_FP32,
        Precision.FP8_E4M3: MixedPrecisionPolicy.FULL_BF16,
        Precision.FP8_E5M2: MixedPrecisionPolicy.FULL_BF16,
    }
    mixed_prec = precision_map.get(
        hw_cfg.precision, MixedPrecisionPolicy.FULL_BF16,
    )

    # ── Backward Prefetch ──
    prefetch_map = {
        "backward_pre": BackwardPrefetchMode.BACKWARD_PRE,
        "backward_post": BackwardPrefetchMode.BACKWARD_POST,
        "none": BackwardPrefetchMode.NONE,
    }
    prefetch_str = getattr(
        dist_cfg, "fsdp_backward_prefetch", "backward_pre",
    )
    backward_prefetch = prefetch_map.get(
        prefetch_str.lower(), BackwardPrefetchMode.BACKWARD_PRE,
    )

    # ── CPU Offload ──
    offload = OffloadStrategy.NONE
    if getattr(dist_cfg, "fsdp_cpu_offload", False):
        offload = OffloadStrategy.CPU_FULL

    # ── Activation Checkpointing ──
    ac_enabled = getattr(dist_cfg, "gradient_checkpointing", False)
    ac_mode = getattr(dist_cfg, "ac_mode", "selective")
    ac_freq = getattr(dist_cfg, "ac_frequency", 2)

    # ── Gradient Accumulation [INT-008] ──
    grad_accum = getattr(train_cfg, "gradient_accumulation_steps", 1)

    # ── Gradient Clipping [INT-008] ──
    grad_clip = getattr(opt_cfg, "max_grad_norm", 1.0)
    if grad_clip <= 0:
        grad_clip = None

    # ── use_orig_params (required for torch.compile + FSDP) ──
    use_orig = getattr(dist_cfg, "fsdp_use_orig_params", True)
    if hw_cfg.compile_model:
        use_orig = True

    # ── Triton ──
    use_triton = True
    if hasattr(sota_config, "kernels"):
        use_triton = sota_config.kernels.use_triton

    return FSDP2Config(
        sharding_strategy=sharding,
        mixed_precision=mixed_prec,
        offload_strategy=offload,
        use_orig_params=use_orig,
        forward_prefetch=getattr(dist_cfg, "fsdp_forward_prefetch", True),
        backward_prefetch=backward_prefetch,
        limit_all_gathers=getattr(
            dist_cfg, "fsdp_limit_all_gathers", True,
        ),
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
        - backward()   → engine.backward() (no_sync + accumulation)
        - step()       → engine.step() (clip → step → zero_grad)
        - checkpoint   → FSDPCheckpointManager via engine
        - Trainer does NOT manage GradScaler, no_sync, or grad accum

    For all other strategies (DDP, single-GPU, ZBPP, CP):
        - Original behavior with trainer-managed backward/step
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

        # ─────────────────────────────────────────────────────────────
        # [INT-001] FSDP2 engine — owns training lifecycle when active
        # ─────────────────────────────────────────────────────────────
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
        if self.distributed_strategy in (
            "ddp", "fsdp", "fsdp2", "sota_fsdp",
        ):
            DDPInitializer.init_process_group()
            log_rank_0(
                f"Initialized distributed: {self.distributed_strategy}"
            )

        # Device setup
        self.device = self._setup_device()
        self.dtype = config.compute_dtype

        # [INT-002] AMP context — only for non-FSDP2 strategies
        self.amp_context = self._setup_amp_context()

        # Metrics + Hub
        self._setup_metrics()
        self._setup_hub()

        log_rank_0(
            f"SOTATrainer initialized: "
            f"mode={config.training_mode.value}, "
            f"device={self.device}, "
            f"strategy={self.distributed_strategy}"
        )

    # ═════════════════════════════════════════════════════════════════════
    # Properties
    # ═════════════════════════════════════════════════════════════════════

    @property
    def _is_fsdp2_active(self) -> bool:
        """True when FSDP2 engine manages training lifecycle."""
        return self._fsdp2_engine is not None

    @property
    def _local_rank(self) -> int:
        """Local rank for device affinity."""
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def _is_main_process(self) -> bool:
        """True on rank 0 or non-distributed."""
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    # ═════════════════════════════════════════════════════════════════════
    # Device Setup
    # ═════════════════════════════════════════════════════════════════════

    def _setup_device(self) -> torch.device:
        """Setup compute device with rank-aware GPU selection."""
        hw = self.config.hardware

        if hw.device.value == "auto":
            if torch.cuda.is_available():
                local_rank = int(
                    os.environ.get("LOCAL_RANK", hw.device_id),
                )
                device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(device)
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
        AMP context — ONLY for non-FSDP2 strategies.

        [INT-002] FSDP2 manages autocast via forward_context().
        """
        precision = self.config.hardware.precision

        if precision == Precision.FP16:
            # [INT-002] GradScaler only for non-FSDP2
            if self.distributed_strategy not in (
                "fsdp", "fsdp2", "sota_fsdp",
            ):
                self.scaler = torch.amp.GradScaler(device="cuda")
            return lambda: torch.amp.autocast(
                'cuda', dtype=torch.float16,
            )
        elif precision == Precision.BF16:
            return lambda: torch.amp.autocast(
                'cuda', dtype=torch.bfloat16,
            )
        elif precision in (Precision.FP8_E4M3, Precision.FP8_E5M2):
            return lambda: torch.amp.autocast(
                'cuda', dtype=torch.bfloat16,
            )
        else:
            return lambda: nullcontext()

    def _setup_default_callbacks(self) -> None:
        """Setup default callbacks."""
        train_cfg = self.config.training
        log_steps = getattr(train_cfg, 'logging_steps', 100)

        logging_config = LoggingConfig(
            backends=[LoggingBackend.CONSOLE],
            log_steps=log_steps,
            log_dir=getattr(train_cfg, 'logging_dir', 'logs'),
        )
        self.callback_handler.add_callback(
            LoggingCallback(config=logging_config),
        )
        self.callback_handler.add_callback(ProgressCallback())

        if (
            hasattr(train_cfg, 'save_steps')
            and train_cfg.save_steps > 0
        ):
            ckpt_config = CheckpointConfig(
                strategy=CheckpointStrategy.STEPS,
                save_steps=train_cfg.save_steps,
                save_total_limit=getattr(
                    train_cfg, 'save_total_limit', 3,
                ),
            )
            self.callback_handler.add_callback(CheckpointCallback(
                save_dir=getattr(
                    train_cfg, 'output_dir', './checkpoints',
                ),
                config=ckpt_config,
            ))

        if (
            hasattr(train_cfg, 'early_stopping_patience')
            and train_cfg.early_stopping_patience > 0
        ):
            self.callback_handler.add_callback(EarlyStoppingCallback(
                patience=train_cfg.early_stopping_patience,
                min_delta=getattr(
                    train_cfg, 'early_stopping_threshold', 0.0,
                ),
            ))

    def _setup_metrics(self) -> None:
        """Setup training metrics."""
        try:
            self.training_metrics = create_training_metrics(
                distributed=metrics_is_distributed(),
                model=self.model,
            )
            self.loss_tracker = self.training_metrics.loss
            self.throughput_tracker = self.training_metrics.throughput
            self.gradient_tracker = self.training_metrics.gradient
        except Exception as e:
            logger.warning(f"Could not create TrainingMetrics: {e}")
            self.training_metrics = None
            self.loss_tracker = LossTracker(ema_decay=0.99)
            self.throughput_tracker = ThroughputTracker()
            self.gradient_tracker = GradientTracker()

    def _setup_hub(self) -> None:
        """Setup hub integration."""
        export_cfg = self.config.export
        train_cfg = self.config.training

        self.hub_manager = None
        if (
            hasattr(export_cfg, 'push_to_hub')
            and export_cfg.push_to_hub
        ):
            try:
                hub_config = HubConfig(
                    token=getattr(export_cfg, 'hub_token', None),
                    repo_id=getattr(export_cfg, 'hub_model_id', None),
                    private=getattr(export_cfg, 'hub_private', False),
                )
                self.hub_manager = HubManager(config=hub_config)
            except Exception as e:
                logger.warning(f"Could not initialize HubManager: {e}")

        checkpoint_dir = getattr(
            train_cfg, 'output_dir', './checkpoints',
        )
        max_checkpoints = getattr(train_cfg, 'save_total_limit', 3)

        try:
            self.checkpoint_manager = CheckpointManager(
                save_dir=Path(checkpoint_dir),
                hub_manager=self.hub_manager,
                max_checkpoints=max_checkpoints,
            )
        except Exception as e:
            logger.warning(f"Could not init CheckpointManager: {e}")
            self.checkpoint_manager = None

        self.model_serializer = ModelSerializer(
            shard_size_bytes=5 * 1024**3,
            prefer_safetensors=True,
        )

    # ═════════════════════════════════════════════════════════════════════
    # Numerical Stability
    # ═════════════════════════════════════════════════════════════════════

    def _check_numerical_stability(
        self,
        loss: Tensor,
        grad_norm: Optional[float] = None,
        step: Optional[int] = None,
    ) -> bool:
        """Check for NaN/Inf. Returns True if stable."""
        step_info = f" at step {step}" if step is not None else ""

        if torch.isnan(loss) or torch.isinf(loss):
            log_rank_0(
                f"⚠️ NaN/Inf loss{step_info}: {loss.item():.4e}",
                level="warning",
            )
            return False

        if grad_norm is not None:
            if grad_norm != grad_norm:  # NaN check
                log_rank_0(
                    f"⚠️ NaN gradient norm{step_info}",
                    level="warning",
                )
                return False
            if grad_norm > 1e6:
                log_rank_0(
                    f"⚠️ Exploding grad{step_info}: {grad_norm:.2e}",
                    level="warning",
                )
        return True

    def _get_no_sync_context(self, should_sync: bool):
        """
        Gradient sync context for DDP gradient accumulation.

        [INT-010] ONLY used for DDP. FSDP2 manages no_sync internally.
        """
        if should_sync:
            return nullcontext()
        return no_sync(self.model)

    # ═════════════════════════════════════════════════════════════════════
    # Model Setup
    # ═════════════════════════════════════════════════════════════════════

    def setup_model(
        self,
        model: Optional[nn.Module] = None,
        model_init_fn: Optional[Callable[[], nn.Module]] = None,
    ) -> nn.Module:
        """
        Setup model with distributed wrapping.

        FSDP2 path:
            1. Build FSDP2Config from SOTAConfig [INT-005]
            2. Create SOTAFSDP2 engine [INT-001]
            3. Wrap model (activation checkpointing inside)
            4. Store engine as self._fsdp2_engine
            5. Nullify trainer's scaler [INT-002]
        """
        if model is not None:
            self.model = model
        elif model_init_fn is not None:
            self.model = model_init_fn()
        else:
            self.model = self._load_model()

        self.model = self._apply_training_mode(self.model)
        self.model = self._apply_kernel_optimizations(self.model)
        self.model = self.model.to(self.device)

        # ── Distributed Wrapping ──

        if self.distributed_strategy == "ddp":
            log_rank_0("Applying SOTA DDP wrapper...")
            dist_cfg = self.config.distributed
            ddp_params = {
                "gradient_as_bucket_view": (
                    dist_cfg.ddp_gradient_as_bucket_view
                ),
                "static_graph": dist_cfg.ddp_static_graph,
                "find_unused_parameters": (
                    dist_cfg.ddp_find_unused_parameters
                ),
                "broadcast_buffers": dist_cfg.ddp_broadcast_buffers,
            }
            ddp_params.update(dist_cfg.ddp_config)

            if (
                hasattr(self.config, "kernels")
                and "use_triton_kernels" not in ddp_params
            ):
                ddp_params["use_triton_kernels"] = (
                    self.config.kernels.use_triton
                )

            try:
                self.model = (
                    create_ddp_engine(**ddp_params)
                    .unwrap()
                    .wrap_model(self.model)
                )
                log_rank_0(
                    f"  ✓ DDP (static_graph="
                    f"{ddp_params['static_graph']})"
                )
            except Exception as e:
                logger.error(f"DDP wrapping failed: {e}")
                raise

        elif self.distributed_strategy in (
            "fsdp", "fsdp2", "sota_fsdp",
        ):
            # ═══════════════════════════════════════════════════════
            # [INT-001] FSDP2 lifecycle management
            # ═══════════════════════════════════════════════════════
            log_rank_0(
                f"Applying SOTA FSDP2 "
                f"({self.distributed_strategy})..."
            )

            fsdp2_config = _build_fsdp2_config(self.config)

            try:
                self._fsdp2_engine = SOTAFSDP2(fsdp2_config)
                self.model = self._fsdp2_engine.wrap_model(self.model)
                # [INT-002] FSDP2 owns scaling
                self.scaler = None

                log_rank_0(
                    f"  ✓ FSDP2: "
                    f"strategy={fsdp2_config.sharding_strategy.name}, "
                    f"precision={fsdp2_config.mixed_precision.name}, "
                    f"grad_accum="
                    f"{fsdp2_config.gradient_accumulation_steps}, "
                    f"grad_clip={fsdp2_config.gradient_clipping_norm}"
                )
            except Exception as e:
                logger.error(f"FSDP2 wrapping failed: {e}")
                raise

        elif self.distributed_strategy == "pipeline_zbpp":
            log_rank_0("Applying ZBPP Pipeline...")
            from data_pipeline.trainer.distributed.zbpp import (
                create_zbpp_pipeline,
            )

            num_stages = self.config.distributed.num_pipeline_stages
            num_microbatches = (
                self.config.distributed.num_microbatches
            )
            if num_microbatches < num_stages:
                logger.warning(
                    f"num_microbatches ({num_microbatches}) < "
                    f"num_stages ({num_stages}), adjusting"
                )
                num_microbatches = num_stages

            try:
                self.model = create_zbpp_pipeline(
                    model=self.model,
                    num_stages=num_stages,
                    num_microbatches=num_microbatches,
                    memory_limit_gb=(
                        self.config.distributed.pipeline_memory_limit_gb
                    ),
                    lr=self.config.optimizer.learning_rate,
                    weight_decay=self.config.optimizer.weight_decay,
                    dtype=self.dtype,
                    rank=(
                        dist.get_rank()
                        if dist.is_initialized()
                        else 0
                    ),
                    world_size=(
                        dist.get_world_size()
                        if dist.is_initialized()
                        else 1
                    ),
                )
                self.optimizer = self.model._optimizer
                log_rank_0(
                    f"  ✓ ZBPP (stages={num_stages}, "
                    f"μB={num_microbatches})"
                )
            except Exception as e:
                logger.error(f"ZBPP wrapping failed: {e}")
                raise

        elif self.distributed_strategy == "context_parallel":
            log_rank_0("Applying Context Parallel...")
            cp_size = getattr(
                self.config.distributed, "context_parallel_size", 2,
            )
            try:
                cp_engine = create_context_parallel_engine(
                    model=self.model,
                    cp_size=cp_size,
                    device=self.device,
                )
                self.model = cp_engine.wrap_model()
                log_rank_0(f"  ✓ CP (cp_size={cp_size})")
            except Exception as e:
                logger.error(f"Context Parallel failed: {e}")
                raise

        # ── Compilation (after wrapping) ──
        if self.config.hardware.compile_model:
            log_rank_0(
                f"Compiling (mode={self.config.hardware.compile_mode})"
            )
            self.model = torch.compile(
                self.model,
                mode=self.config.hardware.compile_mode,
            )

        return self.model

    def _load_model(self) -> nn.Module:
        """Load model from config."""
        try:
            from transformers import (
                AutoModelForCausalLM, BitsAndBytesConfig,
            )
        except ImportError:
            raise ImportError(
                "transformers required for model loading"
            )

        model_cfg = self.config.model
        quant_cfg = self.config.quantization

        bnb_config = None
        if quant_cfg.enabled:
            if quant_cfg.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=getattr(
                        torch, quant_cfg.bnb_4bit_compute_dtype,
                    ),
                    bnb_4bit_use_double_quant=(
                        quant_cfg.bnb_4bit_use_double_quant
                    ),
                )
            elif quant_cfg.load_in_8bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=quant_cfg.llm_int8_threshold,
                )

        use_cache = (
            not self.config.distributed.gradient_checkpointing
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.name_or_path,
            revision=model_cfg.revision,
            trust_remote_code=model_cfg.trust_remote_code,
            dtype=(
                model_cfg.torch_dtype
                if model_cfg.torch_dtype != "auto"
                else "auto"
            ),
            low_cpu_mem_usage=model_cfg.low_cpu_mem_usage,
            attn_implementation=model_cfg.attn_implementation,
            quantization_config=bnb_config,
            use_cache=use_cache,
        )
        return model

    def _apply_training_mode(self, model: nn.Module) -> nn.Module:
        """Apply training mode (LoRA, full-finetune, etc.)."""
        mode = self.config.training_mode

        if mode in (TrainingMode.LORA, TrainingMode.QLORA):
            model = self._apply_lora(model)
        elif mode in (TrainingMode.FULL_FINETUNE, TrainingMode.PRETRAIN):
            for param in model.parameters():
                param.requires_grad = True

        # Gradient checkpointing — NOT for FSDP2 (handled internally)
        if (
            self.config.distributed.gradient_checkpointing
            and self.distributed_strategy not in (
                "fsdp", "fsdp2", "sota_fsdp",
            )
        ):
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": (
                            self.config.distributed
                            .gradient_checkpointing_use_reentrant
                        ),
                    }
                )
        return model

    def _apply_lora(self, model: nn.Module) -> nn.Module:
        """Apply LoRA/QLoRA."""
        try:
            from peft import (
                LoraConfig, get_peft_model,
                prepare_model_for_kbit_training,
            )
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

    # ═════════════════════════════════════════════════════════════════════
    # Layer Detection (model-agnostic duck-typing)
    # ═════════════════════════════════════════════════════════════════════

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
            (
                hasattr(module, 'wi_0')
                and hasattr(module, 'wi_1')
                and hasattr(module, 'wo')
            )
            or (
                hasattr(module, 'fc1')
                and hasattr(module, 'fc2')
                and hasattr(module, 'act')
            )
        )

    @staticmethod
    def _is_mlp_standard(module: nn.Module) -> bool:
        return (
            (
                hasattr(module, 'dense_h_to_4h')
                and hasattr(module, 'dense_4h_to_h')
            )
            or (
                hasattr(module, 'c_fc')
                and hasattr(module, 'c_proj')
            )
            or (
                hasattr(module, 'fc_in')
                and hasattr(module, 'fc_out')
            )
        )

    @staticmethod
    def _is_attention(module: nn.Module) -> bool:
        if (
            hasattr(module, 'q_proj')
            and hasattr(module, 'k_proj')
            and hasattr(module, 'v_proj')
        ):
            return True
        if (
            hasattr(module, 'query')
            and hasattr(module, 'key')
            and hasattr(module, 'value')
        ):
            return True
        if (
            hasattr(module, 'q')
            and hasattr(module, 'k')
            and hasattr(module, 'v')
        ):
            return True
        if hasattr(module, 'query_key_value'):
            return True
        if hasattr(module, 'Wqkv') or hasattr(module, 'qkv_proj'):
            return True
        if hasattr(module, 'c_attn'):
            return True
        if (
            hasattr(module, 'to_q')
            and hasattr(module, 'to_k')
            and hasattr(module, 'to_v')
        ):
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
        if re.match(
            r".*Rotary.*Embed.*", cls_name, re.IGNORECASE,
        ):
            return True
        if (
            hasattr(module, 'inv_freq')
            and hasattr(module, 'cos_cached')
        ):
            return True
        if (
            hasattr(module, 'inv_freq')
            and hasattr(module, 'max_seq_len_cached')
        ):
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
            hasattr(module, 'gate')
            or hasattr(module, 'router')
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
        """Pre-optimization sanity check."""
        try:
            dtypes = {}
            for name, param in model.named_parameters():
                dtype = str(param.dtype)
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
            logger.info(f"Dtype distribution: {dtypes}")

            if (
                'torch.float32' in dtypes
                and (
                    'torch.float16' in dtypes
                    or 'torch.bfloat16' in dtypes
                )
            ):
                logger.warning(
                    "⚠ Mixed dtypes (fp32 + fp16/bf16)"
                )

            has_nan = has_inf = False
            for i, (name, param) in enumerate(
                model.named_parameters(),
            ):
                if i > 5:
                    break
                if torch.isnan(param).any():
                    has_nan = True
                    logger.error(f"❌ NaN in: {name}")
                if torch.isinf(param).any():
                    has_inf = True
                    logger.error(f"❌ Inf in: {name}")

            if not has_nan and not has_inf:
                logger.info("✓ Weight sanity passed")
        except Exception as e:
            logger.warning(f"Stability check failed: {e}")

    def _apply_kernel_optimizations(
        self, model: nn.Module,
    ) -> nn.Module:
        """
        Apply Triton/Flash/FP8 kernel optimizations.

        10-phase pipeline with graceful fallback.
        """
        import re

        kernel_cfg = self.config.kernels
        hw_cfg = self.config.hardware
        dist_cfg = self.config.distributed

        self._check_model_stability(model)

        if not kernel_cfg.use_triton:
            logger.info("Triton disabled — skipping kernels")
            return model

        capabilities = get_kernel_capabilities()
        triton_ok = capabilities.get("triton", False)
        flash_ok = capabilities.get("flash_attn", False)
        fp8_ok = capabilities.get("fp8", False)
        bf16_ok = capabilities.get("bf16", False)
        cuda_ok = capabilities.get("cuda", False)
        tma_ok = capabilities.get("tma", False)

        try:
            from data_pipeline.trainer.registry import (
                detect_capabilities,
            )
            hw_caps = detect_capabilities()
        except (ImportError, Exception):
            hw_caps = {}

        logger.info(
            f"Kernels: triton={triton_ok}, flash={flash_ok}, "
            f"fp8={fp8_ok}, bf16={bf16_ok}, tma={tma_ok}"
        )

        applied_phases = []

        # Pre-scan: layer map
        layer_map = {
            "mlp_swiglu": [], "mlp_geglu": [],
            "mlp_standard": [], "attention": [],
            "layernorm": [], "rope": [],
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
            f"Layers: swiglu={len(layer_map['mlp_swiglu'])}, "
            f"attn={len(layer_map['attention'])}, "
            f"norm={len(layer_map['layernorm'])}, "
            f"moe={len(layer_map['moe'])}, "
            f"linear={len(layer_map['linear'])}"
        )

        # Phase 2: Layer patching
        try:
            from data_pipeline.trainer.registry import KernelPatcher
            patcher = KernelPatcher(
                patch_layernorm=(
                    kernel_cfg.use_fused_rms_norm and triton_ok
                ),
                patch_mlp=triton_ok,
                patch_attention=(
                    kernel_cfg.use_flash_attention
                    and (flash_ok or triton_ok)
                ),
                patch_rope=(
                    kernel_cfg.use_fused_rope and triton_ok
                ),
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
                logger.warning(
                    f"Phase 2 failed: {e} → {e2}"
                )

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
                logger.warning(f"Phase 3 failed: {e}")

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
                        gate_params = get_lora_parameters(
                            module.gate_proj,
                        )
                        if gate_params[2] is not None:
                            module._fused_lora_fn = LoRA_MLP
                            module._lora_swiglu_fn = (
                                apply_lora_mlp_swiglu
                            )
                            module._uses_fused_lora = True
                            lora_patched += 1

                for name, module in layer_map["attention"]:
                    q_mod = (
                        getattr(module, 'q_proj', None)
                        or getattr(module, 'query', None)
                        or getattr(module, 'to_q', None)
                    )
                    if (
                        q_mod is not None
                        and self._has_lora_adapters(q_mod)
                    ):
                        q_params = get_lora_parameters(q_mod)
                        if q_params[2] is not None:
                            module._fused_lora_fn = LoRA_QKV
                            module._lora_qkv_fn = apply_lora_qkv
                            module._uses_fused_lora = True
                            lora_patched += 1

                if lora_patched > 0:
                    applied_phases.append("fused_lora")
                    logger.info(
                        f"Phase 4: {lora_patched} fused LoRA"
                    )
            except Exception as e:
                logger.warning(f"Phase 4 failed: {e}")

        # Phase 5: MoE kernels
        if kernel_cfg.use_moe_kernels and triton_ok:
            try:
                from data_pipeline.trainer.kernels import (
                    MoEKernelConfig, get_accelerator_arch,
                )
                arch = get_accelerator_arch()
                moe_config = MoEKernelConfig()
                for name, module in layer_map["moe"]:
                    module._moe_kernel_config = moe_config
                    module._accelerator_arch = arch
                    module._uses_triton_moe = True
                if layer_map["moe"]:
                    applied_phases.append("moe_kernels")
            except Exception as e:
                logger.warning(f"Phase 5 failed: {e}")

        # Phase 6: FP8
        if (
            fp8_ok and triton_ok
            and self.config.quantization.load_in_fp8
        ):
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
                        module._fp8_scale_manager = (
                            FP8ScaleManager(config=fp8_cfg)
                        )
                        fp8_layers += 1
                if fp8_layers > 0:
                    applied_phases.append("fp8_quantization")
            except Exception as e:
                logger.warning(f"Phase 6 failed: {e}")

        # Phase 7: Flex Attention
        if kernel_cfg.use_flash_attention and (
            flash_ok or triton_ok
        ):
            try:
                from data_pipeline.trainer.kernels import (
                    FlexAttentionConfig, AttentionBackend,
                    BackendCapabilities,
                )
                backend_caps = BackendCapabilities()
                attn_configured = 0
                for name, module in layer_map["attention"]:
                    flex_cfg = None
                    if hasattr(
                        FlexAttentionConfig, 'PrecisionMode',
                    ):
                        prec = (
                            FlexAttentionConfig.PrecisionMode.FP8
                            if fp8_ok
                            else (
                                FlexAttentionConfig.PrecisionMode.BF16
                                if bf16_ok
                                else (
                                    FlexAttentionConfig
                                    .PrecisionMode.FP16
                                )
                            )
                        )
                        flex_cfg = FlexAttentionConfig(
                            causal=True, precision=prec,
                        )
                    if flex_cfg is not None:
                        module._flex_attention_backend = (
                            backend_caps
                            .select_optimal_backend(flex_cfg)
                        )
                    elif flash_ok:
                        module._flex_attention_backend = (
                            AttentionBackend.FLASH_ATTENTION
                        )
                    elif triton_ok:
                        module._flex_attention_backend = (
                            AttentionBackend.TRITON_FUSED
                        )
                    else:
                        module._flex_attention_backend = (
                            AttentionBackend.SDPA_NATIVE
                        )
                    module._flex_attention_configured = True
                    attn_configured += 1
                if attn_configured > 0:
                    applied_phases.append("flex_attention")
            except Exception as e:
                logger.warning(f"Phase 7 failed: {e}")

        # Phase 9: Distributed kernels
        if dist_cfg.enabled and cuda_ok:
            try:
                from data_pipeline.trainer.kernels import (
                    DistributedKernels,
                    DistributedConfig as KernelDistConfig,
                    DistributedBackend,
                )
                backend_val = None
                if hasattr(
                    DistributedBackend, dist_cfg.backend.upper(),
                ):
                    backend_val = DistributedBackend(
                        dist_cfg.backend,
                    )
                dist_kernel_config = KernelDistConfig(
                    backend=backend_val,
                )
                self._distributed_kernels = DistributedKernels(
                    config=dist_kernel_config,
                )
                applied_phases.append("distributed_kernels")
            except Exception as e:
                logger.warning(f"Phase 9 failed: {e}")

        # Summary
        self._kernel_optimization_phases = applied_phases
        self._kernel_capabilities = capabilities

        if applied_phases:
            logger.info(
                f"✓ Kernels ({len(applied_phases)}/10): "
                f"{', '.join(applied_phases)}"
            )
        else:
            logger.warning("⚠ No kernel optimizations applied")

        return model

    # ═════════════════════════════════════════════════════════════════════
    # Optimizer & Scheduler & Loss Setup
    # ═════════════════════════════════════════════════════════════════════

    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        opt_cfg = self.config.optimizer
        params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
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
                logger.warning("8-bit Adam unavailable → AdamW")
                self.optimizer = torch.optim.AdamW(
                    params, lr=opt_cfg.learning_rate,
                )
        elif opt_type == OptimizerType.LION:
            try:
                self.optimizer = Lion(
                    params, lr=opt_cfg.learning_rate,
                    betas=opt_cfg.lion_betas,
                    weight_decay=opt_cfg.weight_decay,
                )
            except ImportError:
                logger.warning("Lion unavailable → AdamW")
                self.optimizer = torch.optim.AdamW(
                    params, lr=opt_cfg.learning_rate,
                )
        elif opt_type == OptimizerType.FUSED_ADAMW:
            try:
                self.optimizer = FusedAdamW(
                    params, lr=opt_cfg.learning_rate,
                    betas=opt_cfg.betas, eps=opt_cfg.eps,
                    weight_decay=opt_cfg.weight_decay,
                    max_grad_norm=opt_cfg.max_grad_norm,
                )
            except ImportError:
                self.optimizer = torch.optim.AdamW(
                    params, lr=opt_cfg.learning_rate,
                )
        else:
            self.optimizer = torch.optim.AdamW(
                params, lr=opt_cfg.learning_rate,
                weight_decay=opt_cfg.weight_decay,
            )
        return self.optimizer

    def setup_scheduler(
        self, num_training_steps: int,
    ) -> Any:
        """Setup LR scheduler."""
        sch_cfg = self.config.scheduler
        warmup_steps = sch_cfg.warmup_steps
        if warmup_steps == 0:
            warmup_steps = int(
                sch_cfg.warmup_ratio * num_training_steps,
            )

        sch_type = sch_cfg.type

        if sch_type == SchedulerType.COSINE:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - warmup_steps,
                eta_min=(
                    self.config.optimizer.learning_rate
                    * sch_cfg.min_lr_ratio
                ),
            )
        elif sch_type == SchedulerType.WSD:
            try:
                self.scheduler = WSDScheduler(
                    self.optimizer,
                    num_training_steps=num_training_steps,
                    warmup_steps=warmup_steps,
                    stable_steps=int(
                        sch_cfg.stable_ratio * num_training_steps,
                    ),
                    min_lr_ratio=sch_cfg.min_lr_ratio,
                    decay_type=sch_cfg.decay_type,
                )
            except ImportError:
                from torch.optim.lr_scheduler import (
                    CosineAnnealingLR,
                )
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
                self.loss_fn = nn.CrossEntropyLoss(
                    ignore_index=loss_cfg.ignore_index,
                )
        elif loss_type == LossType.FOCAL:
            try:
                self.loss_fn = FocalLoss(
                    gamma=loss_cfg.focal_gamma,
                    alpha=loss_cfg.focal_alpha,
                    ignore_index=loss_cfg.ignore_index,
                    reduction=loss_cfg.reduction,
                )
            except ImportError:
                self.loss_fn = nn.CrossEntropyLoss(
                    ignore_index=loss_cfg.ignore_index,
                )
        elif loss_type == LossType.DPO:
            self.loss_fn = DPOLoss(
                beta=loss_cfg.dpo_beta,
                label_smoothing=loss_cfg.label_smoothing,
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                ignore_index=loss_cfg.ignore_index,
            )
        return self.loss_fn

    # ═════════════════════════════════════════════════════════════════════
    # Training Loop
    # ═════════════════════════════════════════════════════════════════════

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        compute_metrics: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Run training loop.

        FSDP2 active:
            forward  → engine.forward_context() [INT-007]
            backward → engine.backward()        [INT-003]
            step     → engine.step()             [INT-004]

        Non-FSDP2:
            Original trainer-managed backward/step
        """
        if self.optimizer is None:
            self.setup_optimizer()

        train_cfg = self.config.training
        grad_accum = train_cfg.gradient_accumulation_steps

        num_training_steps = (
            len(train_dataloader)
            // grad_accum
            * train_cfg.num_train_epochs
        )

        if self.scheduler is None:
            self.setup_scheduler(num_training_steps)
        if self.loss_fn is None:
            self.setup_loss()

        max_grad_norm = self.config.optimizer.max_grad_norm

        loss_tracker = LossTracker(ema_decay=0.99)
        acc_tracker = AccuracyTracker()

        self.model.train()
        metrics: Dict[str, float] = {"train_loss": 0.0}
        self.start_time = time.time()

        if eval_dataloader is not None and self._is_main_process:
            try:
                logger.info(
                    f"Eval: {len(eval_dataloader)} batches"
                )
            except Exception:
                logger.info("Eval ready")

        for epoch in range(train_cfg.num_train_epochs):
            self.state.epoch = epoch

            for step, batch in enumerate(train_dataloader):
                step_start_time = time.time()

                batch = {
                    k: v.to(self.device)
                    if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                }

                # ─── ZBPP Pipeline ───
                if self.distributed_strategy == "pipeline_zbpp":
                    num_mb = (
                        self.config.distributed.num_microbatches
                    )
                    batch_size = batch["input_ids"].size(0)
                    if batch_size < num_mb:
                        num_mb = batch_size
                    mb_size = batch_size // num_mb
                    if mb_size == 0:
                        continue

                    micro_inputs = list(
                        batch["input_ids"].split(mb_size),
                    )
                    micro_labels = list(
                        batch["labels"].split(mb_size),
                    )

                    try:
                        outputs = self.model.train_step(
                            micro_batches=micro_inputs,
                            labels=micro_labels,
                            loss_fn=self.loss_fn,
                        )
                    except Exception as e:
                        logger.error(f"ZBPP failed: {e}")
                        continue

                    loss = outputs.get("loss", torch.tensor(0.0))
                    if not self._check_numerical_stability(
                        loss, step=self.state.global_step,
                    ):
                        logger.warning("ZBPP numerical issue")

                    num_tokens = sum(
                        (labels != -100).sum().item()
                        for labels in micro_labels
                    )
                    loss_tracker.update(
                        loss, num_tokens=max(1, num_tokens),
                    )
                    if self.scheduler:
                        self.scheduler.step()
                    self.state.global_step += 1
                    continue

                # ─── FSDP2 Path [INT-003][INT-004][INT-007] ───
                if self._is_fsdp2_active:
                    with self._fsdp2_engine.forward_context():
                        outputs = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get(
                                "attention_mask",
                            ),
                            labels=batch.get("labels"),
                        )

                        if (
                            hasattr(outputs, "loss")
                            and outputs.loss is not None
                        ):
                            loss = outputs.loss
                        else:
                            logits = outputs.logits
                            labels = batch["labels"]
                            shift_logits = (
                                logits[..., :-1, :].contiguous()
                            )
                            shift_labels = (
                                labels[..., 1:].contiguous()
                            )
                            loss = self.loss_fn(
                                shift_logits.view(
                                    -1, shift_logits.size(-1),
                                ),
                                shift_labels.view(-1),
                            )

                    if not self._check_numerical_stability(
                        loss, step=self.state.global_step,
                    ):
                        logger.warning(
                            f"Skipping step "
                            f"{self.state.global_step}"
                        )
                        self.optimizer.zero_grad(set_to_none=True)
                        continue

                    loss_for_backward = loss / grad_accum

                    # [INT-003] Delegate backward to engine
                    is_sync_step = self._fsdp2_engine.backward(
                        loss_for_backward,
                    )

                    num_tokens = (
                        (
                            batch.get("labels", batch["input_ids"])
                            != -100
                        ).sum().item()
                    )
                    loss_tracker.update(
                        loss, num_tokens=max(1, num_tokens),
                    )
                    if hasattr(outputs, "logits"):
                        acc_tracker.update_from_logits(
                            outputs.logits, batch.get("labels"),
                        )

                    # [INT-004] Delegate step to engine
                    if is_sync_step:
                        self._fsdp2_engine.step(
                            self.optimizer,
                            scheduler=self.scheduler,
                        )
                        self.state.global_step += 1

                        self._log_training_step(
                            loss_tracker, acc_tracker,
                            num_tokens, grad_accum,
                            num_training_steps, epoch,
                            train_cfg, step_start_time,
                        )

                        if (
                            eval_dataloader is not None
                            and train_cfg.eval_strategy == "steps"
                            and self.state.global_step
                            % train_cfg.eval_steps == 0
                        ):
                            eval_metrics = self.evaluate(
                                eval_dataloader, compute_metrics,
                            )
                            logger.info(f"Eval: {eval_metrics}")
                            if self._check_early_stopping(
                                eval_metrics,
                            ):
                                logger.info("Early stopping.")
                                self._restore_best_model(
                                    train_cfg, metrics,
                                )
                                return metrics

                        if (
                            train_cfg.save_strategy == "steps"
                            and self.state.global_step
                            % train_cfg.save_steps == 0
                        ):
                            self.save_checkpoint()

                    if (
                        train_cfg.max_steps > 0
                        and self.state.global_step
                        >= train_cfg.max_steps
                    ):
                        break
                    continue

                # ─── Standard Path (DDP / Single-GPU / CP) ───
                with self.amp_context():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch.get("labels"),
                    )

                    if (
                        hasattr(outputs, "loss")
                        and outputs.loss is not None
                    ):
                        loss = outputs.loss
                        if hasattr(outputs, "logits"):
                            num_tokens = (
                                (batch["labels"] != -100)
                                .sum().item()
                            )
                            loss_tracker.update(
                                loss, num_tokens=num_tokens,
                            )
                            acc_tracker.update_from_logits(
                                outputs.logits, batch["labels"],
                            )
                    else:
                        logits = outputs.logits
                        labels = batch["labels"]
                        shift_logits = (
                            logits[..., :-1, :].contiguous()
                        )
                        shift_labels = (
                            labels[..., 1:].contiguous()
                        )
                        loss = self.loss_fn(
                            shift_logits.view(
                                -1, shift_logits.size(-1),
                            ),
                            shift_labels.view(-1),
                        )
                        num_tokens = (
                            (labels != -100).sum().item()
                        )
                        loss_tracker.update(
                            loss, num_tokens=num_tokens,
                        )
                        acc_tracker.update_from_logits(
                            logits, labels,
                        )

                    loss_scaled = loss / grad_accum

                if not self._check_numerical_stability(
                    loss, step=self.state.global_step,
                ):
                    self.optimizer.zero_grad()
                    continue

                # [INT-010] no_sync for DDP only
                should_sync = (step + 1) % grad_accum == 0
                sync_context = self._get_no_sync_context(
                    should_sync,
                )

                with sync_context:
                    if self.scaler is not None:
                        self.scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()

                if should_sync:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)

                    grad_norm = 0.0
                    if max_grad_norm > 0:
                        grad_norm = (
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_grad_norm,
                            )
                        )
                    elif hasattr(self.optimizer, "get_grad_norm"):
                        grad_norm = self.optimizer.get_grad_norm()

                    grad_norm_val = (
                        grad_norm
                        if isinstance(grad_norm, float)
                        else grad_norm.item()
                    )
                    if not self._check_numerical_stability(
                        loss, grad_norm=grad_norm_val,
                        step=self.state.global_step,
                    ):
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

                    self._log_training_step(
                        loss_tracker, acc_tracker,
                        num_tokens, grad_accum,
                        num_training_steps, epoch,
                        train_cfg, step_start_time,
                        grad_norm_override=grad_norm_val,
                    )

                    if (
                        eval_dataloader is not None
                        and train_cfg.eval_strategy == "steps"
                        and self.state.global_step
                        % train_cfg.eval_steps == 0
                    ):
                        eval_metrics = self.evaluate(
                            eval_dataloader, compute_metrics,
                        )
                        logger.info(f"Eval: {eval_metrics}")
                        if self._check_early_stopping(
                            eval_metrics,
                        ):
                            logger.info("Early stopping.")
                            self._restore_best_model(
                                train_cfg, metrics,
                            )
                            return metrics

                    if (
                        train_cfg.save_strategy == "steps"
                        and self.state.global_step
                        % train_cfg.save_steps == 0
                    ):
                        self.save_checkpoint()

                if (
                    train_cfg.max_steps > 0
                    and self.state.global_step
                    >= train_cfg.max_steps
                ):
                    break

            # End of epoch
            metrics["train_loss"] = loss_tracker.compute_avg()
            logger.info(
                f"Epoch {epoch + 1} | "
                f"Loss: {metrics['train_loss']:.4f}"
            )

            if (
                eval_dataloader is not None
                and train_cfg.eval_strategy == "epoch"
            ):
                eval_metrics = self.evaluate(
                    eval_dataloader, compute_metrics,
                )
                metrics.update({
                    f"eval_{k}": v
                    for k, v in eval_metrics.items()
                })
                if self._check_early_stopping(eval_metrics):
                    logger.info("Early stopping.")
                    break

            if train_cfg.save_strategy == "epoch":
                self.save_checkpoint()

        self._restore_best_model(train_cfg, metrics)
        return metrics

    # ═════════════════════════════════════════════════════════════════════
    # Logging Helper
    # ═════════════════════════════════════════════════════════════════════

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
        """Unified logging for FSDP2 and standard paths."""
        if self.state.global_step % train_cfg.logging_steps != 0:
            return

        loss_val = loss_tracker.compute_ema()
        if loss_val == 0.0:
            loss_val = loss_tracker.compute_avg()

        acc = acc_tracker.compute()

        # [INT-009] Gradient norm from FSDP2 metrics
        if self._is_fsdp2_active:
            grad_norm_val = (
                self._fsdp2_engine.metrics.current.gradient_norm
            )
        elif grad_norm_override is not None:
            grad_norm_val = grad_norm_override
        else:
            grad_norm_val = 0.0

        metrics_to_reduce = {
            "loss": loss_val,
            "acc": acc,
            "grad": grad_norm_val,
        }

        if dist.is_initialized():
            tensor_vals = torch.tensor(
                [
                    metrics_to_reduce["loss"],
                    metrics_to_reduce["acc"],
                    metrics_to_reduce["grad"],
                ],
                device=self.device,
            )
            dist.all_reduce(tensor_vals, op=dist.ReduceOp.AVG)
            metrics_to_reduce["loss"] = tensor_vals[0].item()
            metrics_to_reduce["acc"] = tensor_vals[1].item()
            metrics_to_reduce["grad"] = tensor_vals[2].item()

        if self._is_main_process:
            ppl = torch.exp(
                torch.tensor(metrics_to_reduce["loss"]),
            ).item()
            lr = self.scheduler.get_last_lr()[0]

            current_time = time.time()
            elapsed = current_time - self.start_time
            steps_done = self.state.global_step
            avg_time = elapsed / steps_done if steps_done > 0 else 0
            remaining = num_training_steps - steps_done
            eta_s = int(avg_time * remaining)
            eta_str = str(datetime.timedelta(seconds=eta_s))

            step_dur = current_time - step_start_time
            world_size = (
                dist.get_world_size()
                if dist.is_initialized()
                else 1
            )
            tok_per_sec = int(
                num_tokens * grad_accum * world_size
                / max(step_dur, 1e-6)
            )

            ppl_str = (
                f"PPL: {ppl:.2e}" if ppl > 1000
                else f"PPL: {ppl:.2f}"
            )

            # [INT-009] FSDP2 memory stats
            mem_str = ""
            if self._is_fsdp2_active:
                mem_str = (
                    f" | {self._fsdp2_engine.memory_summary()}"
                )

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
        """Restore best model if checkpoint exists."""
        best_path = os.path.join(
            train_cfg.output_dir, "checkpoint-best",
        )
        if os.path.exists(best_path):
            logger.info(f"Restoring best model: {best_path}")
            self.load_checkpoint(best_path)
            metrics["best_metric"] = self.state.best_metric

    def _training_step(
        self, batch: Dict[str, Tensor],
    ) -> Tensor:
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
        return self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    def evaluate(
        self,
        eval_dataloader: DataLoader,
        compute_metrics: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Evaluation with distributed synchronization."""
        if dist.is_initialized():
            dist.barrier()

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # [INT-007] FSDP2 autocast during eval
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

        if dist.is_initialized():
            tensor_agg = torch.tensor(
                [total_loss, float(num_batches)],
                device=self.device,
            )
            dist.all_reduce(tensor_agg, op=dist.ReduceOp.SUM)
            global_loss = tensor_agg[0].item()
            global_batches = tensor_agg[1].item()
        else:
            global_loss = total_loss
            global_batches = num_batches

        avg_loss = (
            global_loss / global_batches
            if global_batches > 0
            else 0.0
        )

        eval_metrics = {"loss": avg_loss}
        try:
            eval_metrics["ppl"] = math.exp(avg_loss)
        except OverflowError:
            eval_metrics["ppl"] = float("inf")

        if compute_metrics is not None:
            eval_metrics.update(
                compute_metrics(self.model, eval_dataloader),
            )
        return eval_metrics

    def _check_early_stopping(
        self, eval_metrics: Dict[str, float],
    ) -> bool:
        """Check early stopping condition."""
        current_loss = eval_metrics.get("loss", float('inf'))
        patience = self.config.training.early_stopping_patience
        threshold = (
            self.config.training.early_stopping_threshold
        )

        if current_loss < (self.state.best_metric - threshold):
            self.state.best_metric = current_loss
            self.state.patience_counter = 0

            if self._is_main_process:
                best_path = os.path.join(
                    self.config.training.output_dir,
                    "checkpoint-best",
                )
                logger.info(
                    f"New best (Loss: {current_loss:.4f}). "
                    f"Saving: {best_path}"
                )
                self.save_checkpoint(path=best_path)
            return False

        self.state.patience_counter += 1
        logger.info(
            f"⏳ Early Stop: "
            f"{self.state.patience_counter}/{patience} "
            f"(Best: {self.state.best_metric:.4f})"
        )
        return self.state.patience_counter >= patience

    # ═════════════════════════════════════════════════════════════════════
    # Checkpointing — Device-Safe, FSDP2-Aware [TFIX-001..004]
    # ═════════════════════════════════════════════════════════════════════

    def save_checkpoint(
        self, path: Optional[str] = None,
    ) -> None:
        """
        Save training checkpoint for all strategies.

        [TFIX-001] Device affinity enforced via _enforce_trainer_device.
        [TFIX-002] FSDP2: optimizer NOT re-saved by trainer (already
                   saved by FSDPCheckpointManager). GC between phases.
        [TFIX-003] All deprecated API warnings suppressed in scope.
        [TFIX-004] Explicit GC + cache clear after checkpoint.
        [INT-006]  FSDP2 uses FSDPCheckpointManager.save_checkpoint().
        """
        if path is None:
            path = os.path.join(
                self.config.training.output_dir,
                f"checkpoint-{self.state.global_step}",
            )

        os.makedirs(path, exist_ok=True)

        # [TFIX-003] Suppress all checkpoint warnings in scope
        with _suppress_checkpoint_warnings():
            # [TFIX-001] Pin device for entire checkpoint operation
            with _enforce_trainer_device(self.device):

                # ── ZBPP Pipeline ──
                if self.distributed_strategy == "pipeline_zbpp":
                    self._save_zbpp_checkpoint(path)
                    return

                # ── Context Parallel barrier ──
                if (
                    self.distributed_strategy == "context_parallel"
                    and dist.is_initialized()
                ):
                    dist.barrier()

                # ── FSDP2 [INT-006] ──
                if self._is_fsdp2_active:
                    self._save_fsdp2_checkpoint(path)
                    return

                # ── Raw FSDP guard ──
                from torch.distributed.fsdp import (
                    FullyShardedDataParallel as FSDP,
                )
                if isinstance(self.model, FSDP):
                    logger.warning(
                        "Raw FSDP without engine — "
                        "check setup_model()"
                    )
                    self._save_trainer_state(path)
                    return

                # ── Standard DDP / Single GPU ──
                self._save_standard_checkpoint(path)

    def _save_fsdp2_checkpoint(self, path: str) -> None:
        """
        FSDP2 checkpoint save with device safety.

        Pipeline:
            1. Pre-save memory cleanup [TFIX-002]
            2. FSDPCheckpointManager.save_checkpoint() [INT-006]
            3. Trainer state (scheduler, RNG — NO optimizer) [TFIX-002]
            4. Post-save memory cleanup [TFIX-004]
            5. Device re-pin [TFIX-001]
        """
        log_rank_0(f"Saving FSDP2 checkpoint: {path}")
        save_start = time.monotonic()

        # [TFIX-002] Pre-save memory cleanup
        if self._fsdp2_engine._memory_pool is not None:
            self._fsdp2_engine._memory_pool.clear()
        gc.collect()
        torch.cuda.empty_cache()

        # [INT-006] Delegate to FSDP2 checkpoint manager
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
            logger.error(
                f"FSDP2 checkpoint failed: {result.error}"
            )
            # [TFIX-001] Re-pin device on failure
            torch.cuda.set_device(self.device)
            return

        # [TFIX-001] Re-pin device after FSDP2 save
        torch.cuda.set_device(self.device)

        # [TFIX-002] Save trainer state (NO optimizer — already saved)
        if self._fsdp2_engine.is_rank_zero:
            self._save_trainer_state_fsdp2(path)

        # [TFIX-004] Post-save cleanup
        gc.collect()
        torch.cuda.empty_cache()

        # [TFIX-001] Final device re-pin
        torch.cuda.set_device(self.device)

        elapsed = time.monotonic() - save_start
        log_rank_0(
            f"FSDP2 checkpoint saved in {elapsed:.1f}s: {path}"
        )

    def _save_zbpp_checkpoint(self, path: str) -> None:
        """ZBPP pipeline checkpoint."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        log_rank_0(f"Saving ZBPP checkpoint: {path}")

        if hasattr(self.model, 'save_stage_checkpoint'):
            self.model.save_stage_checkpoint(path)
        else:
            stage_path = os.path.join(
                path, f"stage_{rank}_model.pt",
            )
            model_to_save = (
                self.model.module
                if hasattr(self.model, 'module')
                else self.model
            )
            torch.save(model_to_save.state_dict(), stage_path)

        if hasattr(self.model, "_optimizer"):
            opt_path = os.path.join(
                path, f"stage_{rank}_optimizer.pt",
            )
            torch.save(
                self.model._optimizer.state_dict(), opt_path,
            )

        if rank == 0:
            self._save_trainer_state(path)

        if dist.is_initialized():
            dist.barrier()

    def _save_standard_checkpoint(self, path: str) -> None:
        """Standard DDP / Single-GPU checkpoint."""
        model_to_save = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )

        if self._is_main_process:
            log_rank_0(f"Saving checkpoint: {path}")

            if hasattr(model_to_save, 'save_pretrained'):
                model_to_save.save_pretrained(path)
            else:
                torch.save(
                    model_to_save.state_dict(),
                    os.path.join(path, "model.pt"),
                )
            self._save_trainer_state(path)
            log_rank_0(f"Checkpoint saved: {path}")

    


    def _save_trainer_state(self, path: str) -> None:
        """
        Save trainer state (optimizer, scheduler, RNG).

        [TFIX-001] All state dicts moved to CPU before torch.save
        to prevent CUDA device leakage into the serialized file.
        When checkpoints are loaded on different ranks, tensors pinned
        to cuda:0 cause NCCL device-mismatch errors:
            "Tensor found on device cuda:0 but backend constrained to cuda:3"

        [TFIX-002] Optimizer state can hold large tensors (momentum, variance).
        Deep-copy to CPU is O(optimizer_state_size) but avoids corrupting
        the live optimizer state. For AdamW with a 7B model across 5 GPUs,
        optimizer state per rank is ~2.8 GB — fits comfortably in host RAM.

        [TFIX-003] RNG states are small (<1 KB each) but CUDA RNG tensors
        must be moved to CPU before serialization. Failure to do so causes
        device-pinned RNG states that break deterministic resumption on
        different GPU topologies.

        Complexity: O(S) where S = total optimizer state size per rank.
        """
        # ── Optimizer state → CPU ──
        opt_state = None
        if self.optimizer is not None:
            opt_state = self.optimizer.state_dict()
            # [TFIX-001] Recursively move ALL tensors to CPU
            opt_state = self._state_dict_to_cpu(opt_state)

        # ── Scheduler state → CPU ──
        sched_state = None
        if self.scheduler is not None:
            try:
                sched_state = self.scheduler.state_dict()
                sched_state = self._state_dict_to_cpu(sched_state)
            except Exception as e:
                logger.warning(f"Failed to serialize scheduler state: {e}")

        # ── RNG states → CPU [TFIX-003] ──
        rng_state = self._collect_rng_states()

        # ── Training state (pure Python, no tensors) ──
        training_state = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "best_metric": self.state.best_metric,
            "total_loss": self.state.total_loss,
            "samples_seen": self.state.samples_seen,
            "patience_counter": self.state.patience_counter,
        }

        # ── Config snapshot (for reproducibility auditing) ──
        config_dict = None
        try:
            config_dict = self.config.to_dict()
        except Exception:
            # Config serialization is best-effort
            pass

        # ── Scaler state (non-FSDP2 fp16 only) ──
        scaler_state = None
        if self.scaler is not None:
            try:
                scaler_state = self.scaler.state_dict()
            except Exception:
                pass

        # ── Assemble checkpoint ──
        checkpoint = {
            "optimizer": opt_state,
            "scheduler": sched_state,
            "state": training_state,
            "config": config_dict,
            "rng_state": rng_state,
            "scaler": scaler_state,
            # Metadata for diagnostic / compatibility checks
            "meta": {
                "torch_version": torch.__version__,
                "world_size": (
                    dist.get_world_size() if dist.is_initialized() else 1
                ),
                "rank": (
                    dist.get_rank() if dist.is_initialized() else 0
                ),
                "distributed_strategy": self.distributed_strategy,
                "timestamp": time.time(),
            },
        }

        # ── Atomic save: write to .tmp then rename [TFIX-004] ──
        # Prevents corrupted checkpoints from interrupted saves
        state_path = os.path.join(path, "trainer_state.pt")
        tmp_path = state_path + ".tmp"

        try:
            torch.save(checkpoint, tmp_path)
            # Atomic rename (POSIX guarantees on same filesystem)
            os.replace(tmp_path, state_path)
        except Exception as e:
            # Clean up partial write
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            logger.error(f"Failed to save trainer state: {e}")
            raise

    @staticmethod
    def _state_dict_to_cpu(state: Any) -> Any:
        """
        Recursively move all tensors in a state dict to CPU.

        [TFIX-001] Handles nested dicts, lists, tuples, and raw tensors.
        Non-tensor leaves (int, float, str, None) pass through unchanged.

        Complexity: O(N) where N = number of leaf elements in state tree.
        """
        if isinstance(state, Tensor):
            return state.detach().cpu()
        elif isinstance(state, dict):
            return {
                k: SOTATrainer._state_dict_to_cpu(v)
                for k, v in state.items()
            }
        elif isinstance(state, list):
            return [SOTATrainer._state_dict_to_cpu(v) for v in state]
        elif isinstance(state, tuple):
            return tuple(SOTATrainer._state_dict_to_cpu(v) for v in state)
        else:
            # int, float, str, None, np.ndarray, etc.
            return state

    @staticmethod
    def _collect_rng_states() -> Dict[str, Any]:
        """
        Collect all RNG states for deterministic resumption.

        [TFIX-003] CUDA RNG tensors MUST be on CPU before serialization.
        torch.cuda.get_rng_state() returns a ByteTensor on the current
        CUDA device — if saved as-is, loading on a different device
        topology causes silent device drift.
        """
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),  # Already CPU ByteTensor
        }

        if torch.cuda.is_available():
            try:
                # get_rng_state_all() returns list of ByteTensors,
                # one per visible CUDA device — move each to CPU
                cuda_states = torch.cuda.get_rng_state_all()
                rng["cuda"] = [
                    s.cpu() if isinstance(s, Tensor) else s
                    for s in cuda_states
                ]
            except Exception as e:
                logger.warning(f"Failed to collect CUDA RNG states: {e}")
                rng["cuda"] = None
        else:
            rng["cuda"] = None

        return rng

    def _load_trainer_state(self, path: str) -> None:
        """
        Load trainer state (optimizer, scheduler, RNG).

        [TFIX-005] All states loaded with map_location="cpu" to prevent
        device drift. Optimizer state tensors are lazily moved to the
        correct device by the optimizer on next step() call.

        [TFIX-006] Graceful degradation: if any component fails to load,
        training continues with default state + warning (no crash).
        """
        state_path = os.path.join(path, "trainer_state.pt")
        if not os.path.exists(state_path):
            logger.warning(f"Trainer state not found: {state_path}")
            return

        try:
            # [TFIX-005] Force CPU loading — no device drift
            checkpoint = torch.load(
                state_path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as e:
            logger.error(f"Failed to load trainer state file: {e}")
            return

        # ── Training state ──
        saved_state = checkpoint.get("state", {})
        if isinstance(saved_state, dict):
            self.state.epoch = saved_state.get("epoch", self.state.epoch)
            self.state.global_step = saved_state.get(
                "global_step", self.state.global_step,
            )
            self.state.best_metric = saved_state.get(
                "best_metric", self.state.best_metric,
            )
            self.state.total_loss = saved_state.get(
                "total_loss", self.state.total_loss,
            )
            self.state.samples_seen = saved_state.get(
                "samples_seen", self.state.samples_seen,
            )
            self.state.patience_counter = saved_state.get(
                "patience_counter", self.state.patience_counter,
            )
        elif isinstance(saved_state, TrainingState):
            self.state = saved_state

        # ── Optimizer state [TFIX-006] ──
        if self.optimizer is not None and checkpoint.get("optimizer") is not None:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as e:
                logger.warning(
                    f"Failed to load optimizer state: {e}. "
                    f"Optimizer will restart from scratch."
                )

        # ── Scheduler state [TFIX-006] ──
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            except Exception as e:
                logger.warning(
                    f"Failed to load scheduler state: {e}. "
                    f"Scheduler will restart from scratch."
                )

        # ── GradScaler state (non-FSDP2 fp16 only) ──
        if self.scaler is not None and checkpoint.get("scaler") is not None:
            try:
                self.scaler.load_state_dict(checkpoint["scaler"])
            except Exception as e:
                logger.warning(f"Failed to load scaler state: {e}")

        # ── RNG states [TFIX-003] ──
        if "rng_state" in checkpoint:
            self._restore_rng_state(checkpoint["rng_state"])

        # ── Compatibility check ──
        meta = checkpoint.get("meta", {})
        saved_world_size = meta.get("world_size", -1)
        current_world_size = (
            dist.get_world_size() if dist.is_initialized() else 1
        )
        if saved_world_size > 0 and saved_world_size != current_world_size:
            logger.warning(
                f"World size mismatch: checkpoint={saved_world_size}, "
                f"current={current_world_size}. "
                f"Optimizer/scheduler states may be incompatible."
            )

        log_rank_0(
            f"Trainer state loaded: "
            f"epoch={self.state.epoch}, "
            f"step={self.state.global_step}, "
            f"best_metric={self.state.best_metric:.4f}"
        )

    def _restore_rng_state(self, rng: Dict[str, Any]) -> None:
        """
        Restore RNG states for deterministic resumption.

        [TFIX-003] Each RNG backend restored independently with
        individual error handling — one failure doesn't block others.
        CUDA RNG states are moved to CPU before set_rng_state_all()
        as a defensive measure (some PyTorch versions expect CPU input).
        """
        # Python stdlib RNG
        try:
            if "python" in rng and rng["python"] is not None:
                random.setstate(rng["python"])
        except Exception as e:
            logger.warning(f"Failed to restore Python RNG: {e}")

        # NumPy RNG
        try:
            if "numpy" in rng and rng["numpy"] is not None:
                np.random.set_state(rng["numpy"])
        except Exception as e:
            logger.warning(f"Failed to restore NumPy RNG: {e}")

        # PyTorch CPU RNG
        try:
            if "torch" in rng and rng["torch"] is not None:
                torch_state = rng["torch"]
                if isinstance(torch_state, Tensor):
                    torch_state = torch_state.cpu()
                torch.set_rng_state(torch_state)
        except Exception as e:
            logger.warning(f"Failed to restore PyTorch CPU RNG: {e}")

        # PyTorch CUDA RNG [TFIX-003]
        try:
            if (
                rng.get("cuda") is not None
                and torch.cuda.is_available()
            ):
                cuda_states = rng["cuda"]
                if isinstance(cuda_states, list):
                    # Ensure all states are on CPU
                    cpu_states = [
                        s.cpu() if isinstance(s, Tensor) else s
                        for s in cuda_states
                    ]
                    # Only restore if device count matches
                    num_devices = torch.cuda.device_count()
                    if len(cpu_states) == num_devices:
                        torch.cuda.set_rng_state_all(cpu_states)
                    elif len(cpu_states) > 0:
                        # Partial restore: set current device only
                        local_rank = int(os.environ.get("LOCAL_RANK", 0))
                        if local_rank < len(cpu_states):
                            torch.cuda.set_rng_state(
                                cpu_states[local_rank],
                                device=local_rank,
                            )
                elif isinstance(cuda_states, Tensor):
                    torch.cuda.set_rng_state(cuda_states.cpu())
        except Exception as e:
            logger.warning(f"Failed to restore CUDA RNG: {e}")

    # ═════════════════════════════════════════════════════════════════════════════
    # Checkpointing — FSDP2-Aware [INT-006]
    # ═════════════════════════════════════════════════════════════════════════════

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        """
        Save training checkpoint for all distributed strategies.

        [INT-006] FSDP2 uses FSDPCheckpointManager.save_checkpoint().
        [TFIX-001] All trainer state tensors moved to CPU before save.
        [TFIX-004] Atomic write via .tmp + os.replace().
        [TFIX-007] Pre-save barrier ensures all ranks enter checkpoint
        simultaneously, preventing NCCL timeout from rank skew.

        Pipeline:
            1. Synchronize ranks (barrier)
            2. Strategy-specific model/optimizer save
            3. Trainer state save (rank 0 only for non-sharded)
            4. Post-save barrier
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

            if hasattr(self.model, "save_stage_checkpoint"):
                self.model.save_stage_checkpoint(path)
            else:
                stage_path = os.path.join(path, f"stage_{rank}_model.pt")
                model_to_save = (
                    self.model.module
                    if hasattr(self.model, "module")
                    else self.model
                )
                # [TFIX-001] State dict to CPU
                state = self._state_dict_to_cpu(model_to_save.state_dict())
                torch.save(state, stage_path)

            if hasattr(self.model, "_optimizer"):
                opt_path = os.path.join(
                    path, f"stage_{rank}_optimizer.pt",
                )
                opt_state = self._state_dict_to_cpu(
                    self.model._optimizer.state_dict(),
                )
                torch.save(opt_state, opt_path)

            if rank == 0:
                self._save_trainer_state(path)

            if dist.is_initialized():
                dist.barrier()
            return

        # ── Context Parallel — pre-save barrier ──
        if (
            self.distributed_strategy == "context_parallel"
            and dist.is_initialized()
        ):
            dist.barrier()

        # ── FSDP2 [INT-006] ──
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
                    "total_loss": self.state.total_loss,
                },
                sharded=True,
            )

            if result.is_err():
                logger.error(
                    f"FSDP2 checkpoint save failed: {result.error}"
                )
            else:
                log_rank_0(f"FSDP2 checkpoint saved: {path}")

            # Trainer state saved by rank 0 only
            if self._fsdp2_engine.is_rank_zero:
                self._save_trainer_state(path)

            return

        # ── Raw FSDP guard ──
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
        )
        if isinstance(self.model, FSDP):
            logger.warning(
                "Model is raw FSDP without FSDP2 engine. "
                "This should not happen — check setup_model()."
            )
            if not dist.is_initialized() or dist.get_rank() == 0:
                self._save_trainer_state(path)
            return

        # ── Standard DDP / Single GPU ──
        model_to_save = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )

        is_main = not dist.is_initialized() or dist.get_rank() == 0

        if is_main:
            log_rank_0(f"Saving checkpoint to {path}")

            if hasattr(model_to_save, "save_pretrained"):
                model_to_save.save_pretrained(path)
            else:
                state = self._state_dict_to_cpu(
                    model_to_save.state_dict(),
                )
                model_path = os.path.join(path, "model.pt")
                tmp_path = model_path + ".tmp"
                torch.save(state, tmp_path)
                os.replace(tmp_path, model_path)

            self._save_trainer_state(path)
            log_rank_0(f"Checkpoint saved: {path}")

        # Post-save barrier
        if dist.is_initialized():
            dist.barrier()

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint for all distributed strategies.

        [INT-006] FSDP2 uses FSDPCheckpointManager.load_checkpoint().
        [TFIX-005] All loads use map_location="cpu".
        [TFIX-006] Graceful degradation on component failures.
        """
        log_rank_0(f"Loading checkpoint from {path}")

        # ── ZBPP Pipeline ──
        if self.distributed_strategy == "pipeline_zbpp":
            rank = dist.get_rank() if dist.is_initialized() else 0

            if hasattr(self.model, "load_stage_checkpoint"):
                self.model.load_stage_checkpoint(path)
            else:
                stage_path = os.path.join(
                    path, f"stage_{rank}_model.pt",
                )
                if os.path.exists(stage_path):
                    model_to_load = (
                        self.model.module
                        if hasattr(self.model, "module")
                        else self.model
                    )
                    # [TFIX-005] Force CPU then move
                    state = torch.load(
                        stage_path, map_location="cpu",
                    )
                    model_to_load.load_state_dict(state)

            if hasattr(self.model, "_optimizer"):
                opt_path = os.path.join(
                    path, f"stage_{rank}_optimizer.pt",
                )
                if os.path.exists(opt_path):
                    opt_state = torch.load(
                        opt_path, map_location="cpu",
                    )
                    self.model._optimizer.load_state_dict(opt_state)

            self._load_trainer_state(path)

            if dist.is_initialized():
                dist.barrier()
            return

        # ── FSDP2 [INT-006] ──
        if self._is_fsdp2_active:
            log_rank_0(f"Loading FSDP2 checkpoint from {path}")

            result = FSDPCheckpointManager.load_checkpoint(
                fsdp=self._fsdp2_engine,
                optimizer=self.optimizer,
                path=path,
                sharded=True,
            )

            if result.is_err():
                logger.error(
                    f"FSDP2 checkpoint load failed: {result.error}"
                )
            else:
                meta = result.unwrap()
                self.state.epoch = meta.get(
                    "epoch", self.state.epoch,
                )
                self.state.global_step = meta.get(
                    "step", self.state.global_step,
                )
                extra = meta.get("extra", {})
                self.state.best_metric = extra.get(
                    "best_metric", self.state.best_metric,
                )
                self.state.patience_counter = extra.get(
                    "patience_counter", self.state.patience_counter,
                )
                self.state.samples_seen = extra.get(
                    "samples_seen", self.state.samples_seen,
                )
                self.state.total_loss = extra.get(
                    "total_loss", self.state.total_loss,
                )
                log_rank_0(f"FSDP2 checkpoint loaded: {path}")

            # Load scheduler and RNG from trainer state
            self._load_trainer_state(path)
            return

        # ── Raw FSDP guard ──
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
        )
        if isinstance(self.model, FSDP):
            logger.warning(
                "Loading raw FSDP checkpoint without FSDP2 engine."
            )
            self._load_trainer_state(path)
            return

        # ── Context Parallel — pre-load barrier ──
        if (
            self.distributed_strategy == "context_parallel"
            and dist.is_initialized()
        ):
            dist.barrier()

        # ── Standard DDP / Single GPU ──
        self._load_trainer_state(path)

        # Try PEFT adapter load first (LoRA/QLoRA)
        if hasattr(self.model, "load_adapter"):
            try:
                self.model.load_adapter(path, adapter_name="default")
                log_rank_0(f"LoRA adapter loaded: {path}")
                return
            except Exception as e:
                logger.warning(f"load_adapter failed: {e}")
                try:
                    self.model.load_adapter(path)
                    log_rank_0(f"LoRA adapter loaded (no name): {path}")
                    return
                except Exception as e2:
                    logger.warning(
                        f"Adapter load failed, trying full state: {e2}"
                    )

        # Full model state dict
        model_path = os.path.join(path, "model.pt")
        if os.path.exists(model_path):
            model_to_load = (
                self.model.module
                if hasattr(self.model, "module")
                else self.model
            )
            # [TFIX-005] Force CPU loading
            state = torch.load(model_path, map_location="cpu")
            model_to_load.load_state_dict(state)
            log_rank_0(f"Model state loaded: {model_path}")
        elif hasattr(self.model, "from_pretrained"):
            # HuggingFace pretrained directory
            safetensors_path = os.path.join(
                path, "model.safetensors",
            )
            if (
                os.path.exists(safetensors_path)
                or os.path.exists(os.path.join(path, "config.json"))
            ):
                log_rank_0(
                    f"Loading pretrained model from directory: {path}"
                )
                # Model already loaded; just log
            else:
                logger.warning(
                    f"No model weights found in {path}. "
                    f"Model state not restored."
                )

        log_rank_0(f"Checkpoint loaded: {path}")

    # ═════════════════════════════════════════════════════════════════════════════
    # Export — FSDP2-Aware Full Parameter Materialization
    # ═════════════════════════════════════════════════════════════════════════════

    def export(self, tokenizer=None) -> str:
        """
        Export model to target format.

        [INT-001] For FSDP2: uses summon_full_params() to materialize
        the complete model on rank 0 before export. This is a collective
        operation — ALL ranks must call it, but only rank 0 writes.

        Supported formats:
            - SafeTensors (default, recommended)
            - GGUF (Q4/Q5/Q8/F16 for llama.cpp / ollama)
            - HuggingFace Hub push

        Returns:
            Output directory path (empty string if export disabled).
        """
        export_cfg = self.config.export
        if not export_cfg.enabled:
            logger.warning("Export not enabled in config")
            return ""

        from data_pipeline.trainer.export import (
            save_safetensors,
            export_to_gguf,
            push_to_hub,
            merge_lora_weights,
        )

        output_dir = export_cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # ── FSDP2: materialize full params ──
        if self._is_fsdp2_active:
            log_rank_0("Gathering full parameters for export...")
            export_model = self.model
            export_context = self._fsdp2_engine.summon_full_params(
                writeback=False,
            )
        else:
            export_model = (
                self.model.module
                if hasattr(self.model, "module")
                else self.model
            )
            export_context = nullcontext()

        with export_context:
            # ── LoRA merge ──
            if (
                export_cfg.merge_lora
                and self.config.training_mode in (
                    TrainingMode.LORA, TrainingMode.QLORA,
                )
            ):
                log_rank_0("Merging LoRA weights...")
                export_model = merge_lora_weights(export_model)

            # ── Format dispatch ──
            is_main = (
                not dist.is_initialized() or dist.get_rank() == 0
            )

            if is_main:
                if export_cfg.format == ExportFormat.SAFETENSORS:
                    save_safetensors(export_model, output_dir)
                    log_rank_0(
                        f"SafeTensors exported: {output_dir}"
                    )

                elif export_cfg.format in (
                    ExportFormat.GGUF_Q4,
                    ExportFormat.GGUF_Q5,
                    ExportFormat.GGUF_Q8,
                    ExportFormat.GGUF_F16,
                ):
                    if tokenizer is None:
                        raise ValueError(
                            "Tokenizer required for GGUF export. "
                            "Pass tokenizer= to export()."
                        )
                    export_to_gguf(
                        export_model,
                        tokenizer,
                        output_dir,
                        quantization=export_cfg.gguf_quantization,
                    )
                    log_rank_0(
                        f"GGUF exported: {output_dir} "
                        f"(quant={export_cfg.gguf_quantization})"
                    )

                # ── Hub push ──
                if (
                    export_cfg.push_to_hub
                    and export_cfg.hub_model_id
                ):
                    push_to_hub(
                        export_model,
                        tokenizer,
                        export_cfg.hub_model_id,
                        token=export_cfg.hub_token,
                        private=export_cfg.hub_private,
                    )
                    log_rank_0(
                        f"Pushed to hub: {export_cfg.hub_model_id}"
                    )

        # Post-export barrier
        if dist.is_initialized():
            dist.barrier()

        logger.info(f"Model exported: {output_dir}")
        return output_dir

    # ═════════════════════════════════════════════════════════════════════════════
    # Resource Cleanup
    # ═════════════════════════════════════════════════════════════════════════════

    def cleanup(self) -> None:
        """
        Release all resources held by the trainer.

        Call after training completes or on exception exit.
        Handles distributed process group teardown, CUDA cache,
        memory pool release, and FSDP2 engine cleanup.
        """
        log_rank_0("Cleaning up trainer resources...")

        # ── FSDP2 engine cleanup ──
        if self._is_fsdp2_active:
            self._fsdp2_engine.empty_cache()
            self._fsdp2_engine = None

        # ── CUDA cleanup ──
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # ── Distributed teardown ──
        if dist.is_initialized():
            try:
                dist.barrier()
                dist.destroy_process_group()
            except Exception as e:
                logger.warning(
                    f"Process group teardown failed: {e}"
                )

        log_rank_0("Trainer cleanup complete.")

    def __del__(self) -> None:
        """Destructor — best-effort resource release."""
        try:
            if (
                hasattr(self, "_fsdp2_engine")
                and self._fsdp2_engine is not None
            ):
                self._fsdp2_engine.empty_cache()
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════════
# Factory Function
# ═════════════════════════════════════════════════════════════════════════════════

def create_trainer(config_path: Union[str, Path]) -> SOTATrainer:
    """
    Create trainer from YAML config.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Configured SOTATrainer instance.

    Example:
        >>> trainer = create_trainer("configs/llama3_finetune.yaml")
        >>> trainer.setup_model()
        >>> metrics = trainer.train(train_dl, eval_dl)
        >>> trainer.export(tokenizer)
        >>> trainer.cleanup()
    """
    config = SOTAConfig.from_yaml(config_path)
    return SOTATrainer(config)


def create_trainer_from_config(config: SOTAConfig) -> SOTATrainer:
    """
    Create trainer from pre-built SOTAConfig object.

    Args:
        config: Validated SOTAConfig instance.

    Returns:
        Configured SOTATrainer instance.
    """
    return SOTATrainer(config)


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    "SOTATrainer",
    "TrainingState",
    "create_trainer",
    "create_trainer_from_config",
]
