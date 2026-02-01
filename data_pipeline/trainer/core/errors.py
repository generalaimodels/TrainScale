# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer Module - Error Hierarchy
# ════════════════════════════════════════════════════════════════════════════════
# Comprehensive training error types for precise diagnostics.
# All errors carry structured context for debugging.
#
# Design Principles:
# - Exception hierarchy mirrors training failure modes
# - Each error carries actionable remediation hints
# - Context dict for structured logging and telemetry
# - Chaining via __cause__ for root cause analysis
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


# ═════════════════════════════════════════════════════════════════════════════════
# Base Training Error
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingError(Exception):
    """
    Base exception for all training-related errors.
    
    All trainer exceptions inherit from this class.
    Carries structured context for debugging and logging.
    
    Attributes:
        message: Human-readable error description
        context: Structured key-value context for debugging
        cause: Original exception that caused this error
        remediation: Suggested fix or next steps
    """
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    cause: Optional[Exception] = None
    remediation: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Chain cause exception for traceback preservation."""
        super().__init__(self.message)
        if self.cause is not None:
            self.__cause__ = self.cause
    
    def __str__(self) -> str:
        """Format error with context for logging."""
        parts = [f"TrainingError: {self.message}"]
        
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"  Context: {ctx_str}")
        
        if self.remediation:
            parts.append(f"  Remediation: {self.remediation}")
        
        if self.cause:
            parts.append(f"  Caused by: {type(self.cause).__name__}: {self.cause}")
        
        return "\n".join(parts)
    
    def with_context(self, **kwargs: Any) -> "TrainingError":
        """Add additional context, returns self for chaining."""
        self.context.update(kwargs)
        return self


# ═════════════════════════════════════════════════════════════════════════════════
# Configuration Errors
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class ConfigurationError(TrainingError):
    """
    Error in training configuration (YAML or programmatic).
    
    Raised when:
    - YAML syntax is invalid
    - Required fields are missing
    - Field types don't match schema
    - Field values are out of valid range
    - Conflicting configuration options
    """
    field_path: Optional[str] = None
    expected: Optional[str] = None
    got: Optional[str] = None
    yaml_file: Optional[str] = None
    
    def __str__(self) -> str:
        parts = [f"ConfigurationError: {self.message}"]
        
        if self.yaml_file:
            parts.append(f"  File: {self.yaml_file}")
        
        if self.field_path:
            parts.append(f"  Field: {self.field_path}")
        
        if self.expected and self.got:
            parts.append(f"  Expected: {self.expected}")
            parts.append(f"  Got: {self.got}")
        
        if self.remediation:
            parts.append(f"  Remediation: {self.remediation}")
        
        return "\n".join(parts)


@dataclass
class YAMLParseError(ConfigurationError):
    """
    Error parsing YAML configuration file.
    
    Provides line/column info for syntax errors.
    """
    line: Optional[int] = None
    column: Optional[int] = None
    
    def __str__(self) -> str:
        location = ""
        if self.line is not None:
            location = f" at line {self.line}"
            if self.column is not None:
                location += f", column {self.column}"
        
        file_info = f" in {self.yaml_file}" if self.yaml_file else ""
        return f"YAMLParseError{file_info}{location}: {self.message}"


@dataclass
class SchemaValidationError(ConfigurationError):
    """
    Pydantic schema validation failed.
    
    Carries full validation error details.
    """
    validation_errors: Tuple[str, ...] = field(default_factory=tuple)
    
    def __str__(self) -> str:
        parts = [f"SchemaValidationError: {self.message}"]
        
        if self.yaml_file:
            parts.append(f"  File: {self.yaml_file}")
        
        for error in self.validation_errors:
            parts.append(f"  - {error}")
        
        return "\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════════
# Gradient & Optimization Errors
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class GradientError(TrainingError):
    """
    Base error for gradient-related issues.
    """
    step: Optional[int] = None
    
    def __str__(self) -> str:
        step_info = f" at step {self.step}" if self.step is not None else ""
        return f"GradientError{step_info}: {self.message}"


@dataclass
class GradientOverflowError(GradientError):
    """
    NaN or Inf detected in gradients.
    
    Common causes:
    - Learning rate too high
    - Loss scale overflow in mixed precision
    - Numerical instability in loss computation
    
    Recovery:
    - Reduce learning rate
    - Reduce loss scale factor
    - Enable gradient clipping
    """
    param_name: Optional[str] = None
    grad_norm: Optional[float] = None
    loss_scale: Optional[float] = None
    
    def __str__(self) -> str:
        parts = [f"GradientOverflowError at step {self.step}: {self.message}"]
        
        if self.param_name:
            parts.append(f"  Parameter: {self.param_name}")
        
        if self.grad_norm is not None:
            parts.append(f"  Gradient norm: {self.grad_norm}")
        
        if self.loss_scale is not None:
            parts.append(f"  Loss scale: {self.loss_scale}")
        
        parts.append("  Remediation: Reduce learning rate, enable gradient clipping, "
                    "or reduce loss scale factor")
        
        return "\n".join(parts)


@dataclass
class GradientUnderflowError(GradientError):
    """
    Gradient too small (vanishing gradients).
    
    Common causes:
    - Network too deep without skip connections
    - Activation functions saturating
    - Loss scale too low in mixed precision
    """
    param_name: Optional[str] = None
    grad_norm: Optional[float] = None
    
    def __str__(self) -> str:
        return (f"GradientUnderflowError at step {self.step}: {self.message}\n"
                f"  Remediation: Check for vanishing gradients, consider residual connections")


@dataclass
class OptimizationError(TrainingError):
    """
    Optimizer or scheduler error.
    
    Raised when:
    - Optimizer state is corrupted
    - Scheduler step count mismatch
    - Invalid optimizer configuration
    """
    optimizer_type: Optional[str] = None
    scheduler_type: Optional[str] = None
    
    def __str__(self) -> str:
        opt_info = f" [{self.optimizer_type}]" if self.optimizer_type else ""
        sched_info = f" with {self.scheduler_type}" if self.scheduler_type else ""
        return f"OptimizationError{opt_info}{sched_info}: {self.message}"


# ═════════════════════════════════════════════════════════════════════════════════
# Checkpoint Errors
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class CheckpointError(TrainingError):
    """
    Base error for checkpoint operations.
    """
    checkpoint_path: Optional[str] = None
    
    def __str__(self) -> str:
        path_info = f" [{self.checkpoint_path}]" if self.checkpoint_path else ""
        return f"CheckpointError{path_info}: {self.message}"


@dataclass
class CheckpointSaveError(CheckpointError):
    """
    Failed to save checkpoint.
    
    Causes:
    - Disk full
    - Permission denied
    - Serialization error (pickle/safetensors)
    """
    bytes_written: Optional[int] = None
    bytes_expected: Optional[int] = None
    
    def __str__(self) -> str:
        parts = [f"CheckpointSaveError: {self.message}"]
        
        if self.checkpoint_path:
            parts.append(f"  Path: {self.checkpoint_path}")
        
        if self.bytes_written is not None and self.bytes_expected is not None:
            parts.append(f"  Progress: {self.bytes_written}/{self.bytes_expected} bytes")
        
        return "\n".join(parts)


@dataclass
class CheckpointLoadError(CheckpointError):
    """
    Failed to load checkpoint.
    
    Causes:
    - File not found
    - File corrupted
    - State dict key mismatch
    - Version incompatibility
    """
    missing_keys: Tuple[str, ...] = field(default_factory=tuple)
    unexpected_keys: Tuple[str, ...] = field(default_factory=tuple)
    
    def __str__(self) -> str:
        parts = [f"CheckpointLoadError: {self.message}"]
        
        if self.checkpoint_path:
            parts.append(f"  Path: {self.checkpoint_path}")
        
        if self.missing_keys:
            parts.append(f"  Missing keys: {', '.join(self.missing_keys[:5])}")
            if len(self.missing_keys) > 5:
                parts.append(f"    ... and {len(self.missing_keys) - 5} more")
        
        if self.unexpected_keys:
            parts.append(f"  Unexpected keys: {', '.join(self.unexpected_keys[:5])}")
            if len(self.unexpected_keys) > 5:
                parts.append(f"    ... and {len(self.unexpected_keys) - 5} more")
        
        return "\n".join(parts)


@dataclass
class CheckpointVersionError(CheckpointError):
    """
    Checkpoint version incompatibility.
    """
    checkpoint_version: Optional[str] = None
    current_version: Optional[str] = None
    
    def __str__(self) -> str:
        return (f"CheckpointVersionError: {self.message}\n"
                f"  Checkpoint version: {self.checkpoint_version}\n"
                f"  Current version: {self.current_version}")


# ═════════════════════════════════════════════════════════════════════════════════
# Distributed Training Errors
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class DistributedError(TrainingError):
    """
    Base error for distributed training issues.
    """
    rank: Optional[int] = None
    world_size: Optional[int] = None
    
    def __str__(self) -> str:
        rank_info = ""
        if self.rank is not None and self.world_size is not None:
            rank_info = f" [rank {self.rank}/{self.world_size}]"
        return f"DistributedError{rank_info}: {self.message}"


@dataclass
class CommunicationError(DistributedError):
    """
    Collective communication failure.
    
    Causes:
    - Network timeout
    - NCCL error
    - Process group not initialized
    - Rank mismatch
    """
    operation: Optional[str] = None  # all_reduce, broadcast, etc.
    timeout_seconds: Optional[float] = None
    
    def __str__(self) -> str:
        op_info = f" during {self.operation}" if self.operation else ""
        return f"CommunicationError{op_info}: {self.message}"


@dataclass
class ProcessGroupError(DistributedError):
    """
    Process group initialization or operation error.
    """
    backend: Optional[str] = None  # nccl, gloo, mpi
    
    def __str__(self) -> str:
        backend_info = f" [{self.backend}]" if self.backend else ""
        return f"ProcessGroupError{backend_info}: {self.message}"


@dataclass
class TensorShardingError(DistributedError):
    """
    Tensor parallel sharding error.
    
    Causes:
    - Dimension not divisible by TP degree
    - Shape mismatch after sharding
    - Invalid gather/scatter configuration
    """
    tensor_shape: Optional[Tuple[int, ...]] = None
    tp_size: Optional[int] = None
    shard_dim: Optional[int] = None
    
    def __str__(self) -> str:
        parts = [f"TensorShardingError: {self.message}"]
        
        if self.tensor_shape:
            parts.append(f"  Tensor shape: {self.tensor_shape}")
        
        if self.tp_size and self.shard_dim is not None:
            parts.append(f"  TP size: {self.tp_size}, Shard dim: {self.shard_dim}")
        
        return "\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════════
# Device & Hardware Errors
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class DeviceError(TrainingError):
    """
    Hardware device error.
    """
    device_type: Optional[str] = None
    device_index: Optional[int] = None
    
    def __str__(self) -> str:
        device_info = ""
        if self.device_type:
            device_info = f" [{self.device_type}"
            if self.device_index is not None:
                device_info += f":{self.device_index}"
            device_info += "]"
        return f"DeviceError{device_info}: {self.message}"


@dataclass
class OutOfMemoryError(DeviceError):
    """
    GPU/CPU out of memory.
    
    Carries memory statistics for debugging.
    """
    allocated_mb: Optional[float] = None
    reserved_mb: Optional[float] = None
    total_mb: Optional[float] = None
    peak_mb: Optional[float] = None
    
    def __str__(self) -> str:
        parts = [f"OutOfMemoryError on {self.device_type}: {self.message}"]
        
        if self.allocated_mb is not None:
            parts.append(f"  Allocated: {self.allocated_mb:.1f} MB")
        
        if self.reserved_mb is not None:
            parts.append(f"  Reserved: {self.reserved_mb:.1f} MB")
        
        if self.total_mb is not None:
            parts.append(f"  Total: {self.total_mb:.1f} MB")
        
        if self.peak_mb is not None:
            parts.append(f"  Peak: {self.peak_mb:.1f} MB")
        
        parts.append("  Remediation: Reduce batch size, enable gradient checkpointing, "
                    "or use memory-efficient optimizer (AdaFactor)")
        
        return "\n".join(parts)


@dataclass
class DeviceNotAvailableError(DeviceError):
    """
    Requested device not available.
    """
    available_devices: Tuple[str, ...] = field(default_factory=tuple)
    
    def __str__(self) -> str:
        parts = [f"DeviceNotAvailableError: {self.message}"]
        
        if self.device_type:
            parts.append(f"  Requested: {self.device_type}")
        
        if self.available_devices:
            parts.append(f"  Available: {', '.join(self.available_devices)}")
        
        return "\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════════
# Training Loop Errors
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingLoopError(TrainingError):
    """
    Error during training loop execution.
    """
    step: Optional[int] = None
    epoch: Optional[int] = None
    
    def __str__(self) -> str:
        location = ""
        if self.epoch is not None:
            location = f" at epoch {self.epoch}"
        if self.step is not None:
            location += f", step {self.step}"
        return f"TrainingLoopError{location}: {self.message}"


@dataclass
class EarlyStoppingError(TrainingLoopError):
    """
    Early stopping triggered (not an actual error, used for control flow).
    """
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    patience: Optional[int] = None
    
    def __str__(self) -> str:
        return (f"EarlyStoppingTriggered: {self.message}\n"
                f"  Metric: {self.metric_name} = {self.metric_value}\n"
                f"  Patience exhausted after {self.patience} evaluations")


@dataclass
class DataLoadingError(TrainingError):
    """
    Error loading training data.
    
    Inherited from data_pipeline conventions.
    """
    dataset_name: Optional[str] = None
    split: Optional[str] = None
    batch_index: Optional[int] = None
    
    def __str__(self) -> str:
        parts = [f"DataLoadingError: {self.message}"]
        
        if self.dataset_name:
            parts.append(f"  Dataset: {self.dataset_name}")
        
        if self.split:
            parts.append(f"  Split: {self.split}")
        
        if self.batch_index is not None:
            parts.append(f"  Batch index: {self.batch_index}")
        
        return "\n".join(parts)


@dataclass
class ModelError(TrainingError):
    """
    Model-related error.
    """
    model_name: Optional[str] = None
    param_count: Optional[int] = None
    
    def __str__(self) -> str:
        model_info = f" [{self.model_name}]" if self.model_name else ""
        return f"ModelError{model_info}: {self.message}"


# ═════════════════════════════════════════════════════════════════════════════════
# Hub Integration Errors
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class HubError(TrainingError):
    """
    HuggingFace Hub operation error.
    """
    repo_id: Optional[str] = None
    operation: Optional[str] = None  # push, pull, create
    
    def __str__(self) -> str:
        parts = [f"HubError: {self.message}"]
        
        if self.repo_id:
            parts.append(f"  Repository: {self.repo_id}")
        
        if self.operation:
            parts.append(f"  Operation: {self.operation}")
        
        return "\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════════
# Export
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Base
    "TrainingError",
    # Configuration
    "ConfigurationError",
    "YAMLParseError",
    "SchemaValidationError",
    # Gradient
    "GradientError",
    "GradientOverflowError",
    "GradientUnderflowError",
    "OptimizationError",
    # Checkpoint
    "CheckpointError",
    "CheckpointSaveError",
    "CheckpointLoadError",
    "CheckpointVersionError",
    # Distributed
    "DistributedError",
    "CommunicationError",
    "ProcessGroupError",
    "TensorShardingError",
    # Device
    "DeviceError",
    "OutOfMemoryError",
    "DeviceNotAvailableError",
    # Training loop
    "TrainingLoopError",
    "EarlyStoppingError",
    "DataLoadingError",
    "ModelError",
    # Hub
    "HubError",
]
