# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer Module - Core Types
# ════════════════════════════════════════════════════════════════════════════════
# Pydantic-based type definitions with YAML configuration support.
# Designed for above-SOTA training infrastructure with Triton+PyTorch.
#
# Design Principles:
# - Immutable configurations via frozen Pydantic models
# - Exhaustive enum coverage for all training options
# - Zero-cost type validation at configuration load time
# - Memory-aligned layouts for hot-path structures
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ─────────────────────────────────────────────────────────────────────────────────
# Type Variables - Generic Containers
# ─────────────────────────────────────────────────────────────────────────────────

T = TypeVar("T")
E = TypeVar("E")
ModelT = TypeVar("ModelT")
OptimizerT = TypeVar("OptimizerT")

# ─────────────────────────────────────────────────────────────────────────────────
# Constants - Cache Line Alignment & Hardware Specifications
# ─────────────────────────────────────────────────────────────────────────────────

CACHE_LINE_SIZE: Final[int] = 64          # bytes - x86/ARM L1 cache line
GPU_WARP_SIZE: Final[int] = 32            # threads - NVIDIA warp size
TPU_TILE_SIZE: Final[int] = 128           # elements - TPU matrix tile dimension
MAX_GRADIENT_NORM: Final[float] = 1.0     # default gradient clipping threshold
DEFAULT_WARMUP_RATIO: Final[float] = 0.1  # 10% warmup by default


# ═════════════════════════════════════════════════════════════════════════════════
# Section 1: Hardware & Device Abstraction
# ═════════════════════════════════════════════════════════════════════════════════

class DeviceType(str, enum.Enum):
    """
    Hardware device type enumeration.
    
    Rationale: Unified abstraction layer for heterogeneous hardware.
    Each device type maps to specific backend optimizations:
    - CPU: OpenMP threading, AVX-512/NEON vectorization
    - CUDA: NVIDIA GPUs, Triton kernels, cuBLAS
    - ROCM: AMD GPUs via HIP, MIOpen
    - MPS: Apple Silicon via Metal Performance Shaders
    - XLA: TPU via XLA compiler (torch_xla)
    - AUTO: Runtime detection with capability-sorted priority
    """
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    XLA = "xla"
    AUTO = "auto"


class ComputeCapability(BaseModel):
    """
    GPU compute capability specification.
    
    Used for Triton kernel dispatch and feature gating.
    Example: SM_80 (A100), SM_89 (RTX 4090), SM_90 (H100/B200).
    """
    model_config = ConfigDict(frozen=True)
    
    major: int = Field(ge=0, description="Major compute capability version")
    minor: int = Field(ge=0, description="Minor compute capability version")
    
    @property
    def sm_version(self) -> int:
        """Returns SM version as integer (e.g., 80 for SM_80)."""
        return self.major * 10 + self.minor
    
    def supports_bf16(self) -> bool:
        """BF16 requires SM >= 80 (Ampere+)."""
        return self.sm_version >= 80
    
    def supports_fp8(self) -> bool:
        """FP8 requires SM >= 89 (Ada Lovelace+)."""
        return self.sm_version >= 89
    
    def supports_flash_attention_v2(self) -> bool:
        """Flash Attention v2 optimized for SM >= 80."""
        return self.sm_version >= 80


class Precision(str, enum.Enum):
    """
    Training precision modes with automatic loss scaling.
    
    Memory vs Compute Trade-offs:
    - FP32: Full precision, 4 bytes/param, baseline accuracy
    - FP16: Half precision, 2 bytes/param, requires loss scaling
    - BF16: Brain float16, 2 bytes/param, no loss scaling needed (SM >= 80)
    - FP8: 1 byte/param, experimental, requires SM >= 89
    - MIXED_FP16: FP16 compute + FP32 master weights
    - MIXED_BF16: BF16 compute + FP32 master weights (recommended)
    """
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"   # Higher range, lower precision
    FP8_E5M2 = "fp8_e5m2"   # Gradient-optimized format
    MIXED_FP16 = "mixed_fp16"
    MIXED_BF16 = "mixed_bf16"
    AUTO = "auto"


# ═════════════════════════════════════════════════════════════════════════════════
# Section 2: Distributed Training Strategies
# ═════════════════════════════════════════════════════════════════════════════════

class ParallelMode(str, enum.Enum):
    """
    Distributed parallelism modes.
    
    Complexity Analysis (N=nodes, G=GPUs/node, P=pipeline stages, E=experts):
    - NONE: O(1) communication, single device
    - DP: O(params/G) all-reduce per step
    - DDP: O(params/G) bucketed all-reduce, overlapped with backward
    - TP: O(2*hidden/G) per layer (all-gather + reduce-scatter)
    - PP: O(micro_batch * activations/P) point-to-point per stage
    - EP: O(tokens * hidden / E) all-to-all routing
    - FSDP: O(params/G) reduce-scatter + all-gather, ZeRO-3 equivalent
    - HYBRID: Composition of above strategies
    """
    NONE = "none"
    DP = "dp"           # Data Parallel (legacy, replicated model)
    DDP = "ddp"         # Distributed Data Parallel (gradient bucketing)
    TP = "tp"           # Tensor Parallel (intra-layer sharding)
    PP = "pp"           # Pipeline Parallel (inter-layer partitioning)
    EP = "ep"           # Expert Parallel (MoE routing)
    SP = "sp"           # Sequence Parallel (long context)
    CP = "cp"           # Context Parallel (ring attention)
    FSDP = "fsdp"       # Fully Sharded Data Parallel (ZeRO-3)
    HYBRID = "hybrid"   # Multi-dimensional parallelism


class ZeROStage(int, enum.Enum):
    """
    ZeRO (Zero Redundancy Optimizer) memory optimization stages.
    
    Memory Reduction per GPU (vs DDP baseline):
    - STAGE_0: No sharding (1x memory)
    - STAGE_1: Optimizer state sharding (~4x reduction for Adam)
    - STAGE_2: + Gradient sharding (~8x reduction)
    - STAGE_3: + Parameter sharding (~Nx reduction, N=world_size)
    """
    STAGE_0 = 0
    STAGE_1 = 1
    STAGE_2 = 2
    STAGE_3 = 3


class ShardingStrategy(str, enum.Enum):
    """
    FSDP sharding strategies for parameter distribution.
    
    Trade-offs:
    - FULL_SHARD: Maximum memory savings, higher communication
    - SHARD_GRAD_OP: Optimizer + gradient sharding only
    - NO_SHARD: DDP-equivalent, parameters replicated
    - HYBRID_SHARD: Intra-node full shard, inter-node replicated
    """
    FULL_SHARD = "full_shard"
    SHARD_GRAD_OP = "shard_grad_op"
    NO_SHARD = "no_shard"
    HYBRID_SHARD = "hybrid_shard"


class DistributedConfig(BaseModel):
    """
    Comprehensive distributed training configuration.
    
    Loaded from YAML, validated via Pydantic.
    Supports multi-dimensional parallelism composition.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    # Primary parallelism mode
    mode: ParallelMode = Field(
        default=ParallelMode.NONE,
        description="Primary distributed training strategy"
    )
    
    # World topology
    world_size: int = Field(
        default=1, ge=1,
        description="Total number of processes across all nodes"
    )
    local_world_size: int = Field(
        default=1, ge=1,
        description="Number of processes per node (GPUs per machine)"
    )
    
    # Tensor Parallel configuration
    tp_size: int = Field(
        default=1, ge=1,
        description="Tensor parallel degree (splits within layers)"
    )
    
    # Pipeline Parallel configuration
    pp_size: int = Field(
        default=1, ge=1,
        description="Pipeline parallel degree (number of stages)"
    )
    pp_micro_batches: int = Field(
        default=1, ge=1,
        description="Number of micro-batches for pipeline interleaving"
    )
    
    # Expert Parallel configuration (MoE)
    ep_size: int = Field(
        default=1, ge=1,
        description="Expert parallel degree for MoE models"
    )
    
    # FSDP/ZeRO configuration
    zero_stage: ZeROStage = Field(
        default=ZeROStage.STAGE_0,
        description="ZeRO optimization stage"
    )
    sharding_strategy: ShardingStrategy = Field(
        default=ShardingStrategy.FULL_SHARD,
        description="FSDP parameter sharding strategy"
    )
    cpu_offload: bool = Field(
        default=False,
        description="Offload optimizer states and gradients to CPU"
    )
    nvme_offload: bool = Field(
        default=False,
        description="Offload to NVMe (requires cpu_offload=True)"
    )
    
    # Communication optimization
    bucket_cap_mb: int = Field(
        default=25, ge=1,
        description="Gradient bucket size for communication fusion (MB)"
    )
    find_unused_parameters: bool = Field(
        default=False,
        description="Enable unused parameter detection (overhead cost)"
    )
    gradient_as_bucket_view: bool = Field(
        default=True,
        description="Use gradient memory as bucket view (memory optimization)"
    )
    
    # Activation checkpointing
    activation_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing for memory reduction"
    )
    checkpoint_granularity: Literal["full", "selective"] = Field(
        default="selective",
        description="Checkpoint all layers or selected layers only"
    )
    
    @model_validator(mode="after")
    def validate_topology(self) -> "DistributedConfig":
        """Ensure parallelism degrees are consistent with world size."""
        expected = self.tp_size * self.pp_size * (self.world_size // (self.tp_size * self.pp_size))
        if self.world_size > 1 and self.mode != ParallelMode.NONE:
            # Validate that TP * PP divides world_size evenly for DP dimension
            dp_size = self.world_size // (self.tp_size * self.pp_size)
            if dp_size * self.tp_size * self.pp_size != self.world_size:
                raise ValueError(
                    f"world_size ({self.world_size}) must be divisible by "
                    f"tp_size * pp_size ({self.tp_size} * {self.pp_size})"
                )
        return self


# ═════════════════════════════════════════════════════════════════════════════════
# Section 3: Optimizer Configuration
# ═════════════════════════════════════════════════════════════════════════════════

class OptimizerType(str, enum.Enum):
    """
    Available optimizer implementations.
    
    All optimizers implemented from scratch with Triton kernel fusion.
    Selection criteria:
    - ADAMW: Default for most fine-tuning (decoupled weight decay)
    - LAMB: Large batch pretraining (layer-wise adaptive rates)
    - ADAFACTOR: Memory-constrained training (factorized moments)
    - LION: Sign-based updates (faster convergence, experimental)
    - SGD: Baseline, useful for specific architectures (ResNets)
    - ADAM_8BIT: Memory-efficient Adam with 8-bit quantized states
    """
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    LAMB = "lamb"
    ADAFACTOR = "adafactor"
    LION = "lion"
    ADAM_8BIT = "adam_8bit"
    SOPHIA = "sophia"        # Second-order optimizer
    SHAMPOO = "shampoo"      # Full-matrix preconditioning


class OptimizerConfig(BaseModel):
    """
    Optimizer hyperparameters with sensible SOTA defaults.
    
    All fields validated for numerical stability.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    optimizer_type: OptimizerType = Field(
        default=OptimizerType.ADAMW,
        description="Optimizer algorithm selection"
    )
    
    # Learning rate
    learning_rate: float = Field(
        default=1e-4, gt=0.0, le=1.0,
        description="Base learning rate"
    )
    
    # Adam family parameters
    beta1: float = Field(
        default=0.9, ge=0.0, lt=1.0,
        description="First moment exponential decay (momentum)"
    )
    beta2: float = Field(
        default=0.999, ge=0.0, lt=1.0,
        description="Second moment exponential decay (RMSprop)"
    )
    epsilon: float = Field(
        default=1e-8, gt=0.0,
        description="Numerical stability epsilon (denominator)"
    )
    
    # Regularization
    weight_decay: float = Field(
        default=0.01, ge=0.0,
        description="Decoupled L2 regularization coefficient"
    )
    
    # Gradient handling
    max_grad_norm: Optional[float] = Field(
        default=1.0, ge=0.0,
        description="Maximum gradient L2 norm for clipping (None=disabled)"
    )
    gradient_accumulation_steps: int = Field(
        default=1, ge=1,
        description="Number of steps to accumulate gradients before update"
    )
    
    # LAMB-specific
    lamb_trust_ratio_clip: Optional[float] = Field(
        default=10.0, ge=0.0,
        description="Maximum trust ratio for LAMB optimizer"
    )
    
    # AdaFactor-specific
    adafactor_relative_step: bool = Field(
        default=False,
        description="Use relative step size (no LR required)"
    )
    adafactor_warmup_init: bool = Field(
        default=False,
        description="Initialize with warmup (slow start)"
    )
    
    # Triton acceleration
    use_triton_kernels: bool = Field(
        default=True,
        description="Enable Triton-fused optimizer kernels"
    )
    fused_update: bool = Field(
        default=True,
        description="Fuse parameter update with gradient computation"
    )


# ═════════════════════════════════════════════════════════════════════════════════
# Section 4: Learning Rate Scheduler Configuration
# ═════════════════════════════════════════════════════════════════════════════════

class SchedulerType(str, enum.Enum):
    """
    Learning rate scheduler types.
    
    All schedulers support warmup integration.
    Selection criteria:
    - COSINE: Smooth decay to min_lr (default for most tasks)
    - COSINE_RESTARTS: SGDR-style with periodic restarts
    - LINEAR: Linear decay to end_lr
    - POLYNOMIAL: Power-law decay (power=2 for inverse square root)
    - CONSTANT: Fixed LR (useful for fine-tuning)
    - CONSTANT_WITH_WARMUP: Warmup then constant
    - INVERSE_SQRT: 1/sqrt(step) decay (transformer default)
    - ONE_CYCLE: Cyclical LR with momentum (fast.ai style)
    """
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_RESTARTS = "cosine_restarts"
    POLYNOMIAL = "polynomial"
    INVERSE_SQRT = "inverse_sqrt"
    ONE_CYCLE = "one_cycle"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


class WarmupType(str, enum.Enum):
    """
    Warmup strategy for learning rate ramp-up.
    
    Warmup prevents early training instability by gradually
    increasing LR from near-zero to target value.
    """
    NONE = "none"
    LINEAR = "linear"          # LR = base_lr * step / warmup_steps
    EXPONENTIAL = "exponential"  # LR = base_lr * (1 - exp(-step/tau))
    CONSTANT = "constant"        # LR = base_lr * warmup_ratio


class SchedulerConfig(BaseModel):
    """
    Learning rate scheduler configuration.
    
    Integrates warmup and main decay schedule.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    scheduler_type: SchedulerType = Field(
        default=SchedulerType.COSINE,
        description="LR decay schedule type"
    )
    
    # Warmup configuration
    warmup_type: WarmupType = Field(
        default=WarmupType.LINEAR,
        description="Warmup ramp-up strategy"
    )
    warmup_steps: int = Field(
        default=0, ge=0,
        description="Number of warmup steps (overrides warmup_ratio)"
    )
    warmup_ratio: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="Fraction of total steps for warmup"
    )
    
    # Decay parameters
    num_training_steps: int = Field(
        default=0, ge=0,
        description="Total training steps (set by trainer)"
    )
    min_lr_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Minimum LR as fraction of base LR"
    )
    
    # Cosine restarts (SGDR)
    num_cycles: float = Field(
        default=0.5, gt=0.0,
        description="Number of cosine cycles (0.5 = half cosine)"
    )
    restart_warmup_steps: int = Field(
        default=0, ge=0,
        description="Warmup steps after each restart"
    )
    
    # Polynomial decay
    power: float = Field(
        default=1.0, gt=0.0,
        description="Polynomial power (1=linear, 2=quadratic)"
    )
    
    # Reduce on plateau
    patience: int = Field(
        default=10, ge=1,
        description="Steps to wait before reducing LR"
    )
    factor: float = Field(
        default=0.1, gt=0.0, lt=1.0,
        description="LR reduction factor on plateau"
    )
    
    @model_validator(mode="after")
    def compute_warmup_steps(self) -> "SchedulerConfig":
        """Convert warmup_ratio to warmup_steps if needed."""
        if self.warmup_steps == 0 and self.warmup_ratio > 0 and self.num_training_steps > 0:
            # Mutable model for computed field
            object.__setattr__(
                self,
                "warmup_steps",
                int(self.warmup_ratio * self.num_training_steps)
            )
        return self


# ═════════════════════════════════════════════════════════════════════════════════
# Section 5: Loss Function Configuration
# ═════════════════════════════════════════════════════════════════════════════════

class LossType(str, enum.Enum):
    """
    Loss function types with Triton kernel acceleration.
    
    All losses implement backward pass gradients for autograd.
    """
    CROSS_ENTROPY = "cross_entropy"           # Standard classification
    CROSS_ENTROPY_SMOOTH = "cross_entropy_smooth"  # Label smoothing
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    KL_DIVERGENCE = "kl_divergence"           # Distribution matching
    KL_REVERSE = "kl_reverse"                 # Mode-seeking KL
    FOCAL = "focal"                           # Class imbalance
    CONTRASTIVE = "contrastive"               # Representation learning
    TRIPLET = "triplet"                       # Metric learning
    MSE = "mse"                               # Regression
    MAE = "mae"                               # Robust regression


class LossConfig(BaseModel):
    """
    Loss function configuration with smoothing and weighting.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    loss_type: LossType = Field(
        default=LossType.CROSS_ENTROPY,
        description="Primary loss function"
    )
    
    # Label smoothing (CrossEntropy)
    label_smoothing: float = Field(
        default=0.0, ge=0.0, lt=1.0,
        description="Label smoothing epsilon (0=no smoothing)"
    )
    
    # Ignore index for padding tokens
    ignore_index: int = Field(
        default=-100,
        description="Token ID to ignore in loss computation"
    )
    
    # Reduction mode
    reduction: Literal["mean", "sum", "none"] = Field(
        default="mean",
        description="Loss reduction across batch"
    )
    
    # Focal loss parameters
    focal_gamma: float = Field(
        default=2.0, ge=0.0,
        description="Focal loss focusing parameter"
    )
    focal_alpha: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Focal loss positive class weight"
    )
    
    # KL divergence
    kl_temperature: float = Field(
        default=1.0, gt=0.0,
        description="Temperature for softmax in KL div"
    )
    kl_log_target: bool = Field(
        default=False,
        description="Target already in log space"
    )
    
    # Triton acceleration
    use_triton_kernel: bool = Field(
        default=True,
        description="Use Triton fused loss kernel"
    )


# ═════════════════════════════════════════════════════════════════════════════════
# Section 6: Training State & Metrics
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True, frozen=True)
class TrainingState:
    """
    Immutable training state snapshot.
    
    Passed through training loop, checkpointed, restored.
    Uses slots for memory efficiency in hot path.
    
    Memory Layout:
    - 64-byte aligned for cache-line efficiency
    - Integer fields grouped for vectorized loads
    """
    global_step: int = 0
    epoch: int = 0
    epoch_step: int = 0
    total_steps: int = 0
    
    # Loss tracking (exponential moving average)
    running_loss: float = 0.0
    smoothing_factor: float = 0.99
    
    # Gradient statistics
    grad_norm: float = 0.0
    grad_scale: float = 1.0  # For mixed precision
    overflow_count: int = 0
    
    # Learning rate (for logging)
    current_lr: float = 0.0
    
    # Epoch progress
    samples_seen: int = 0
    tokens_seen: int = 0
    
    def with_update(self, **kwargs) -> "TrainingState":
        """Create new state with updated fields (immutable update)."""
        current = {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}
        current.update(kwargs)
        return TrainingState(**current)


@dataclass(slots=True, frozen=True)
class GradientInfo:
    """
    Gradient monitoring information for diagnostics.
    
    Tracked per-step for debugging vanishing/exploding gradients.
    """
    global_norm: float           # L2 norm across all parameters
    max_norm: float              # Maximum parameter gradient norm
    min_norm: float              # Minimum parameter gradient norm
    clipped: bool                # Whether clipping was applied
    overflow: bool               # NaN/Inf detected
    scale_factor: float = 1.0    # Applied gradient scale (AMP)
    
    # Per-layer statistics (optional, for debugging)
    layer_norms: Optional[Dict[str, float]] = None


@dataclass(slots=True, frozen=True)
class StepMetrics:
    """
    Metrics collected per training step.
    
    Aggregated and logged at configurable intervals.
    """
    loss: float
    lr: float
    grad_norm: float
    throughput: float            # tokens/second or samples/second
    gpu_memory_mb: float
    step_time_ms: float
    
    # Optional per-component losses
    component_losses: Optional[Dict[str, float]] = None


# ═════════════════════════════════════════════════════════════════════════════════
# Section 7: Callback Events
# ═════════════════════════════════════════════════════════════════════════════════

class TrainerEvent(str, enum.Enum):
    """
    Trainer lifecycle events for callback hooks.
    
    Ordered by occurrence in training loop.
    """
    # Initialization
    ON_INIT_END = "on_init_end"
    
    # Training lifecycle
    ON_TRAIN_BEGIN = "on_train_begin"
    ON_TRAIN_END = "on_train_end"
    
    # Epoch lifecycle
    ON_EPOCH_BEGIN = "on_epoch_begin"
    ON_EPOCH_END = "on_epoch_end"
    
    # Step lifecycle
    ON_STEP_BEGIN = "on_step_begin"
    ON_STEP_END = "on_step_end"
    
    # Gradient events
    ON_BEFORE_BACKWARD = "on_before_backward"
    ON_AFTER_BACKWARD = "on_after_backward"
    ON_BEFORE_OPTIMIZER_STEP = "on_before_optimizer_step"
    ON_AFTER_OPTIMIZER_STEP = "on_after_optimizer_step"
    
    # Evaluation
    ON_EVALUATE_BEGIN = "on_evaluate_begin"
    ON_EVALUATE_END = "on_evaluate_end"
    
    # Checkpointing
    ON_SAVE = "on_save"
    ON_LOAD = "on_load"
    
    # Logging
    ON_LOG = "on_log"
    
    # Prediction
    ON_PREDICT_BEGIN = "on_predict_begin"
    ON_PREDICT_END = "on_predict_end"


# ═════════════════════════════════════════════════════════════════════════════════
# Section 8: Checkpoint Configuration
# ═════════════════════════════════════════════════════════════════════════════════

class CheckpointStrategy(str, enum.Enum):
    """
    Checkpoint saving strategy.
    """
    STEPS = "steps"           # Save every N steps
    EPOCH = "epoch"           # Save at end of each epoch
    BEST = "best"             # Save only best model by metric
    BEST_AND_LAST = "best_and_last"  # Save best + most recent
    NO = "no"                 # No checkpointing


class CheckpointConfig(BaseModel):
    """
    Checkpoint saving and loading configuration.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    strategy: CheckpointStrategy = Field(
        default=CheckpointStrategy.BEST_AND_LAST,
        description="When to save checkpoints"
    )
    
    save_steps: int = Field(
        default=500, ge=1,
        description="Save checkpoint every N steps (if strategy=steps)"
    )
    
    save_total_limit: Optional[int] = Field(
        default=3, ge=1,
        description="Maximum checkpoints to retain (oldest deleted)"
    )
    
    # Best model tracking
    metric_for_best: str = Field(
        default="eval_loss",
        description="Metric to monitor for best model selection"
    )
    greater_is_better: bool = Field(
        default=False,
        description="Whether higher metric is better"
    )
    
    # Checkpoint format
    save_safetensors: bool = Field(
        default=True,
        description="Use safetensors format instead of pickle"
    )
    save_optimizer: bool = Field(
        default=True,
        description="Include optimizer state in checkpoint"
    )
    save_scheduler: bool = Field(
        default=True,
        description="Include scheduler state in checkpoint"
    )
    
    # Resume configuration
    resume_from_checkpoint: Optional[str] = Field(
        default=None,
        description="Path to checkpoint for resume"
    )


# ═════════════════════════════════════════════════════════════════════════════════
# Section 9: Logging Configuration
# ═════════════════════════════════════════════════════════════════════════════════

class LoggingBackend(str, enum.Enum):
    """
    Logging backend integration.
    """
    CONSOLE = "console"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    MLFLOW = "mlflow"
    COMET = "comet"


class LoggingConfig(BaseModel):
    """
    Logging and monitoring configuration.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    backends: List[LoggingBackend] = Field(
        default=[LoggingBackend.CONSOLE],
        description="Active logging backends"
    )
    
    log_steps: int = Field(
        default=10, ge=1,
        description="Log metrics every N steps"
    )
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Console log level"
    )
    
    # General logging
    log_dir: Optional[str] = Field(
        default=None,
        description="Directory for log files (TensorBoard, etc.)"
    )
    project_name: Optional[str] = Field(
        default=None,
        description="Project name for W&B/MLflow/Comet"
    )
    run_name: Optional[str] = Field(
        default=None,
        description="Run name for W&B/MLflow/Comet"
    )
    
    # WandB specific
    wandb_project: Optional[str] = Field(
        default=None,
        description="Weights & Biases project name"
    )
    wandb_run_name: Optional[str] = Field(
        default=None,
        description="WandB run name (auto-generated if None)"
    )
    
    # TensorBoard specific
    tensorboard_dir: Optional[str] = Field(
        default=None,
        description="TensorBoard log directory"
    )
    
    # Profiling
    enable_profiling: bool = Field(
        default=False,
        description="Enable PyTorch profiler"
    )
    profile_steps: Tuple[int, int] = Field(
        default=(10, 20),
        description="Step range for profiling (start, end)"
    )


# ═════════════════════════════════════════════════════════════════════════════════
# Section 10: Export Utilities
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Constants
    "CACHE_LINE_SIZE",
    "GPU_WARP_SIZE",
    "TPU_TILE_SIZE",
    "MAX_GRADIENT_NORM",
    "DEFAULT_WARMUP_RATIO",
    # Hardware
    "DeviceType",
    "ComputeCapability",
    "Precision",
    # Distributed
    "ParallelMode",
    "ZeROStage",
    "ShardingStrategy",
    "DistributedConfig",
    # Optimizer
    "OptimizerType",
    "OptimizerConfig",
    # Scheduler
    "SchedulerType",
    "WarmupType",
    "SchedulerConfig",
    # Loss
    "LossType",
    "LossConfig",
    # State
    "TrainingState",
    "GradientInfo",
    "StepMetrics",
    # Callbacks
    "TrainerEvent",
    # Checkpoint
    "CheckpointStrategy",
    "CheckpointConfig",
    # Logging
    "LoggingBackend",
    "LoggingConfig",
]
