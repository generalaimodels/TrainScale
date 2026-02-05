# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer Module
# ════════════════════════════════════════════════════════════════════════════════
# Above-SOTA training infrastructure compatible with data_pipeline.
#
# Modules:
# - core: Types, errors, configuration
# - optimizers: AdamW, LAMB, AdaFactor with Triton fusion
# - schedulers: Cosine, Linear, Polynomial, InverseSqrt, OneCycle
# - loss: CrossEntropy, KL Divergence, Focal Loss with Triton
# - callbacks: EarlyStopping, Checkpointing, Logging, Progress
# - base: Main Trainer class
# ════════════════════════════════════════════════════════════════════════════════

from data_pipeline.trainer.core import (
    # Types - Constants
    CACHE_LINE_SIZE,
    GPU_WARP_SIZE,
    TPU_TILE_SIZE,
    MAX_GRADIENT_NORM,
    DEFAULT_WARMUP_RATIO,
    # Types - Hardware
    DeviceType,
    ComputeCapability,
    Precision,
    # Types - Distributed
    ParallelMode,
    ZeROStage,
    ShardingStrategy,
    DistributedConfig,
    # Types - Optimizer
    OptimizerType,
    OptimizerConfig,
    # Types - Scheduler
    SchedulerType,
    WarmupType,
    SchedulerConfig,
    # Types - Loss
    LossType,
    LossConfig,
    # Types - State
    TrainingState,
    GradientInfo,
    StepMetrics,
    # Types - Callbacks
    TrainerEvent,
    # Types - Checkpoint
    CheckpointStrategy,
    CheckpointConfig,
    # Types - Logging
    LoggingBackend,
    LoggingConfig,
    # Errors
    TrainingError,
    ConfigurationError,
    YAMLParseError,
    SchemaValidationError,
    GradientError,
    GradientOverflowError,
    GradientUnderflowError,
    OptimizationError,
    CheckpointError,
    CheckpointSaveError,
    CheckpointLoadError,
    CheckpointVersionError,
    DistributedError,
    CommunicationError,
    ProcessGroupError,
    TensorShardingError,
    DeviceError,
    OutOfMemoryError,
    DeviceNotAvailableError,
    TrainingLoopError,
    EarlyStoppingError,
    DataLoadingError,
    ModelError,
    HubError,
    # Config
    TrainingArguments,
    load_training_config,
    load_training_config_from_dict,
    merge_configs,
    interpolate_env_vars,
)

from data_pipeline.trainer.optimizers import (
    BaseOptimizer,
    AdamW,
    LAMB,
    AdaFactor,
    create_adamw,
    create_lamb,
    create_adafactor,
    clip_grad_norm_,
    clip_grad_value_,
    compute_grad_norm,
)

from data_pipeline.trainer.schedulers import (
    BaseScheduler,
    CosineScheduler,
    CosineRestartsScheduler,
    LinearScheduler,
    PolynomialScheduler,
    InverseSqrtScheduler,
    ConstantScheduler,
    OneCycleScheduler,
    create_scheduler,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from data_pipeline.trainer.loss import (
    CrossEntropyLoss,
    KLDivergenceLoss,
    FocalLoss,
    create_loss,
)

from data_pipeline.trainer.callbacks import (
    Callback,
    CallbackContext,
    CallbackHandler,
    EarlyStoppingCallback,
    CheckpointCallback,
    LoggingCallback,
    ProgressCallback,
)

from data_pipeline.trainer.base import (
    Trainer,
    TrainOutput,
    PredictionOutput,
)

from data_pipeline.trainer.distributed import (
    DeviceManager,
    DistributedState,
    DistributedManager,
    GradientSynchronizer,
    activation_checkpoint,
    setup_distributed,
    get_world_info,
    is_main_process,
)

from data_pipeline.trainer.metrics import (
    Metric,
    Accuracy,
    F1Score,
    Perplexity,
    MetricCollection,
    compute_accuracy,
    compute_f1,
)

from data_pipeline.trainer.trainers import (
    PretrainingTrainer,
    FineTuningTrainer,
    Seq2SeqTrainer,
    TextGenerationTrainer,
)

from data_pipeline.trainer.hub import (
    HubManager,
    generate_model_card,
)

from data_pipeline.trainer.kernels import (
    is_triton_available,
    fused_layer_norm,
    fused_softmax,
    fused_gelu,
    compile_model,
)

# ════════════════════════════════════════════════════════════════════════════════
# SOTA Imports (Above Unsloth Level)
# ════════════════════════════════════════════════════════════════════════════════

# SOTA Optimizers
from data_pipeline.trainer.optimizers import (
    Adam8bit,
    Lion,
    CAME,
    SophiaG,
    Prodigy,
    FusedAdamW,
    create_optimizer,
)

# SOTA Schedulers
from data_pipeline.trainer.schedulers import (
    BaseEnhancedScheduler,
    WSDScheduler,
    REXScheduler,
    OneCycleSchedulerEnhanced,
    PolynomialDecayScheduler,
    InverseSquareRootScheduler,
    CosineRestartScheduler,
    create_sota_scheduler,
)

# SOTA Losses
from data_pipeline.trainer.loss import (
    ChunkedCrossEntropyLoss,
    DistillationLoss,
    DPOLoss,
    ORPOLoss,
    SimPOLoss,
    InfoNCELoss,
    ZLoss,
    CompositeLoss,
)

# SOTA Config
from data_pipeline.trainer.core.sota_config import (
    SOTAConfig,
    TrainingMode as SOTATrainingMode,
    Precision as SOTAPrecision,
    RLAlgorithm,
    ExportFormat,
    OptimizerType as SOTAOptimizerType,
    SchedulerType as SOTASchedulerType,
    LossType as SOTALossType,
    DeviceType as SOTADeviceType,
    ModelConfig,
    TokenizerConfig,
    LoRAConfig,
    QuantizationConfig as SOTAQuantizationConfig,
    RLConfig,
    ExportConfig,
    HardwareConfig,
    KernelConfig,
    DataConfig,
    TrainConfig,
    get_qlora_preset,
    get_full_finetune_preset,
    get_fp8_preset,
    get_rl_grpo_preset,
    get_dpo_preset,
)

# SOTA Trainer
from data_pipeline.trainer.trainers.sota_trainer import (
    SOTATrainer,
    TrainingState as SOTATrainingState,
    create_trainer,
)

__all__ = [
    # ─────────────────────────────────────────────────────────────
    # Core Types
    # ─────────────────────────────────────────────────────────────
    "CACHE_LINE_SIZE",
    "GPU_WARP_SIZE",
    "TPU_TILE_SIZE",
    "MAX_GRADIENT_NORM",
    "DEFAULT_WARMUP_RATIO",
    "DeviceType",
    "ComputeCapability",
    "Precision",
    "ParallelMode",
    "ZeROStage",
    "ShardingStrategy",
    "DistributedConfig",
    "OptimizerType",
    "OptimizerConfig",
    "SchedulerType",
    "WarmupType",
    "SchedulerConfig",
    "LossType",
    "LossConfig",
    "TrainingState",
    "GradientInfo",
    "StepMetrics",
    "TrainerEvent",
    "CheckpointStrategy",
    "CheckpointConfig",
    "LoggingBackend",
    "LoggingConfig",
    # ─────────────────────────────────────────────────────────────
    # Errors
    # ─────────────────────────────────────────────────────────────
    "TrainingError",
    "ConfigurationError",
    "YAMLParseError",
    "SchemaValidationError",
    "GradientError",
    "GradientOverflowError",
    "GradientUnderflowError",
    "OptimizationError",
    "CheckpointError",
    "CheckpointSaveError",
    "CheckpointLoadError",
    "CheckpointVersionError",
    "DistributedError",
    "CommunicationError",
    "ProcessGroupError",
    "TensorShardingError",
    "DeviceError",
    "OutOfMemoryError",
    "DeviceNotAvailableError",
    "TrainingLoopError",
    "EarlyStoppingError",
    "DataLoadingError",
    "ModelError",
    "HubError",
    # ─────────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────────
    "TrainingArguments",
    "load_training_config",
    "load_training_config_from_dict",
    "merge_configs",
    "interpolate_env_vars",
    # ─────────────────────────────────────────────────────────────
    # Optimizers
    # ─────────────────────────────────────────────────────────────
    "BaseOptimizer",
    "AdamW",
    "LAMB",
    "AdaFactor",
    "create_adamw",
    "create_lamb",
    "create_adafactor",
    "clip_grad_norm_",
    "clip_grad_value_",
    "compute_grad_norm",
    # ─────────────────────────────────────────────────────────────
    # Schedulers
    # ─────────────────────────────────────────────────────────────
    "BaseScheduler",
    "CosineScheduler",
    "CosineRestartsScheduler",
    "LinearScheduler",
    "PolynomialScheduler",
    "InverseSqrtScheduler",
    "ConstantScheduler",
    "OneCycleScheduler",
    "create_scheduler",
    "get_cosine_schedule_with_warmup",
    "get_linear_schedule_with_warmup",
    # ─────────────────────────────────────────────────────────────
    # Loss Functions
    # ─────────────────────────────────────────────────────────────
    "CrossEntropyLoss",
    "KLDivergenceLoss",
    "FocalLoss",
    "create_loss",
    # ─────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────
    "Callback",
    "CallbackContext",
    "CallbackHandler",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "LoggingCallback",
    "ProgressCallback",
    # ─────────────────────────────────────────────────────────────
    # Base Trainer
    # ─────────────────────────────────────────────────────────────
    "Trainer",
    "TrainOutput",
    "PredictionOutput",
    # ─────────────────────────────────────────────────────────────
    # Distributed
    # ─────────────────────────────────────────────────────────────
    "DeviceManager",
    "DistributedState",
    "DistributedManager",
    "GradientSynchronizer",
    "activation_checkpoint",
    "setup_distributed",
    "get_world_info",
    "is_main_process",
    # ─────────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────────
    "Metric",
    "Accuracy",
    "F1Score",
    "Perplexity",
    "MetricCollection",
    "compute_accuracy",
    "compute_f1",
    # ─────────────────────────────────────────────────────────────
    # Specialized Trainers
    # ─────────────────────────────────────────────────────────────
    "PretrainingTrainer",
    "FineTuningTrainer",
    "Seq2SeqTrainer",
    "TextGenerationTrainer",
    # ─────────────────────────────────────────────────────────────
    # Hub Integration
    # ─────────────────────────────────────────────────────────────
    "HubManager",
    "generate_model_card",
    # ─────────────────────────────────────────────────────────────
    # Kernels
    # ─────────────────────────────────────────────────────────────
    "is_triton_available",
    "fused_layer_norm",
    "fused_softmax",
    "fused_gelu",
    "compile_model",
    # ═════════════════════════════════════════════════════════════
    # SOTA (Above Unsloth Level)
    # ═════════════════════════════════════════════════════════════
    # SOTA Optimizers
    "Adam8bit",
    "Lion",
    "CAME",
    "SophiaG",
    "Prodigy",
    "FusedAdamW",
    "create_optimizer",
    # SOTA Schedulers
    "BaseEnhancedScheduler",
    "WSDScheduler",
    "REXScheduler",
    "OneCycleSchedulerEnhanced",
    "PolynomialDecayScheduler",
    "InverseSquareRootScheduler",
    "CosineRestartScheduler",
    "create_sota_scheduler",
    # SOTA Losses
    "ChunkedCrossEntropyLoss",
    "DistillationLoss",
    "DPOLoss",
    "ORPOLoss",
    "SimPOLoss",
    "InfoNCELoss",
    "ZLoss",
    "CompositeLoss",
    # SOTA Config
    "SOTAConfig",
    "SOTATrainingMode",
    "SOTAPrecision",
    "RLAlgorithm",
    "ExportFormat",
    "SOTAOptimizerType",
    "SOTASchedulerType",
    "SOTALossType",
    "SOTADeviceType",
    "ModelConfig",
    "TokenizerConfig",
    "LoRAConfig",
    "SOTAQuantizationConfig",
    "RLConfig",
    "ExportConfig",
    "HardwareConfig",
    "KernelConfig",
    "DataConfig",
    "TrainConfig",
    "get_qlora_preset",
    "get_full_finetune_preset",
    "get_fp8_preset",
    "get_rl_grpo_preset",
    "get_dpo_preset",
    # SOTA Trainer
    "SOTATrainer",
    "SOTATrainingState",
    "create_trainer",
]

