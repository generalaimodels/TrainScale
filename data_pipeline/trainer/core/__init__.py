# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer Module - Core Package
# ════════════════════════════════════════════════════════════════════════════════
# Core types, errors, and configuration for training infrastructure.
# ════════════════════════════════════════════════════════════════════════════════

from data_pipeline.trainer.core.types import (
    # Constants
    CACHE_LINE_SIZE,
    GPU_WARP_SIZE,
    TPU_TILE_SIZE,
    MAX_GRADIENT_NORM,
    DEFAULT_WARMUP_RATIO,
    # Hardware
    DeviceType,
    ComputeCapability,
    Precision,
    # Distributed
    ParallelMode,
    ZeROStage,
    ShardingStrategy,
    DistributedConfig,
    # Optimizer
    OptimizerType,
    OptimizerConfig,
    # Scheduler
    SchedulerType,
    WarmupType,
    SchedulerConfig,
    # Loss
    LossType,
    LossConfig,
    # State
    TrainingState,
    GradientInfo,
    StepMetrics,
    # Callbacks
    TrainerEvent,
    # Checkpoint
    CheckpointStrategy,
    CheckpointConfig,
    # Logging
    LoggingBackend,
    LoggingConfig,
)

from data_pipeline.trainer.core.errors import (
    # Base
    TrainingError,
    # Configuration
    ConfigurationError,
    YAMLParseError,
    SchemaValidationError,
    # Gradient
    GradientError,
    GradientOverflowError,
    GradientUnderflowError,
    OptimizationError,
    # Checkpoint
    CheckpointError,
    CheckpointSaveError,
    CheckpointLoadError,
    CheckpointVersionError,
    # Distributed
    DistributedError,
    CommunicationError,
    ProcessGroupError,
    TensorShardingError,
    # Device
    DeviceError,
    OutOfMemoryError,
    DeviceNotAvailableError,
    # Training loop
    TrainingLoopError,
    EarlyStoppingError,
    DataLoadingError,
    ModelError,
    # Hub
    HubError,
)

from data_pipeline.trainer.core.config import (
    TrainingArguments,
    load_training_config,
    load_training_config_from_dict,
    merge_configs,
    interpolate_env_vars,
)

__all__ = [
    # Types - Constants
    "CACHE_LINE_SIZE",
    "GPU_WARP_SIZE",
    "TPU_TILE_SIZE",
    "MAX_GRADIENT_NORM",
    "DEFAULT_WARMUP_RATIO",
    # Types - Hardware
    "DeviceType",
    "ComputeCapability",
    "Precision",
    # Types - Distributed
    "ParallelMode",
    "ZeROStage",
    "ShardingStrategy",
    "DistributedConfig",
    # Types - Optimizer
    "OptimizerType",
    "OptimizerConfig",
    # Types - Scheduler
    "SchedulerType",
    "WarmupType",
    "SchedulerConfig",
    # Types - Loss
    "LossType",
    "LossConfig",
    # Types - State
    "TrainingState",
    "GradientInfo",
    "StepMetrics",
    # Types - Callbacks
    "TrainerEvent",
    # Types - Checkpoint
    "CheckpointStrategy",
    "CheckpointConfig",
    # Types - Logging
    "LoggingBackend",
    "LoggingConfig",
    # Errors
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
    # Config
    "TrainingArguments",
    "load_training_config",
    "load_training_config_from_dict",
    "merge_configs",
    "interpolate_env_vars",
]
