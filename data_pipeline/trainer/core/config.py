# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer Module - YAML Configuration Loader
# ════════════════════════════════════════════════════════════════════════════════
# TrainingArguments and configuration management from YAML files.
# Integrates with data_pipeline's existing config loading pattern.
#
# Design Principles:
# - Single source of truth: YAML file defines all training parameters
# - Pydantic validation ensures type safety at load time
# - Sensible defaults allow minimal configuration
# - Environment variable interpolation for secrets
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from data_pipeline.trainer.core.errors import (
    ConfigurationError,
    SchemaValidationError,
    YAMLParseError,
)
from data_pipeline.trainer.core.types import (
    CheckpointConfig,
    DeviceType,
    DistributedConfig,
    LoggingConfig,
    LossConfig,
    OptimizerConfig,
    Precision,
    SchedulerConfig,
)


# ═════════════════════════════════════════════════════════════════════════════════
# Environment Variable Interpolation
# ═════════════════════════════════════════════════════════════════════════════════

ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def interpolate_env_vars(value: Any) -> Any:
    """
    Recursively interpolate environment variables in configuration values.
    
    Supports ${VAR_NAME} syntax with optional default: ${VAR_NAME:-default}
    
    Args:
        value: Configuration value (string, dict, list, or primitive)
        
    Returns:
        Value with environment variables substituted
        
    Example:
        "${HF_TOKEN}" -> actual token value
        "${BATCH_SIZE:-32}" -> "32" if BATCH_SIZE not set
    """
    if isinstance(value, str):
        def replace_env_var(match: re.Match) -> str:
            var_spec = match.group(1)
            
            # Check for default value syntax: VAR_NAME:-default
            if ":-" in var_spec:
                var_name, default = var_spec.split(":-", 1)
            else:
                var_name, default = var_spec, ""
            
            return os.environ.get(var_name.strip(), default)
        
        return ENV_VAR_PATTERN.sub(replace_env_var, value)
    
    elif isinstance(value, dict):
        return {k: interpolate_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [interpolate_env_vars(item) for item in value]
    
    return value


# ═════════════════════════════════════════════════════════════════════════════════
# Training Arguments - Master Configuration
# ═════════════════════════════════════════════════════════════════════════════════

class TrainingArguments(BaseModel):
    """
    Master training configuration loaded from YAML.
    
    This is the single source of truth for all training parameters.
    Integrates with data_pipeline's existing DataLoader configuration.
    
    Example YAML:
    ```yaml
    training:
      output_dir: ./outputs
      num_epochs: 3
      per_device_batch_size: 8
      
      optimizer:
        optimizer_type: adamw
        learning_rate: 2e-5
        weight_decay: 0.01
      
      scheduler:
        scheduler_type: cosine
        warmup_ratio: 0.1
      
      distributed:
        mode: ddp
        world_size: 4
    ```
    """
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Output & Organization
    # ─────────────────────────────────────────────────────────────────────────────
    
    output_dir: str = Field(
        default="./outputs",
        description="Directory for checkpoints, logs, and artifacts"
    )
    run_name: Optional[str] = Field(
        default=None,
        description="Experiment run name (auto-generated if None)"
    )
    overwrite_output_dir: bool = Field(
        default=False,
        description="Overwrite existing output directory"
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Training Duration
    # ─────────────────────────────────────────────────────────────────────────────
    
    num_epochs: int = Field(
        default=3, ge=1,
        description="Number of training epochs"
    )
    max_steps: int = Field(
        default=-1,
        description="Maximum training steps (-1 = use num_epochs)"
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Batch Size Configuration
    # ─────────────────────────────────────────────────────────────────────────────
    
    per_device_train_batch_size: int = Field(
        default=8, ge=1,
        description="Training batch size per device"
    )
    per_device_eval_batch_size: int = Field(
        default=8, ge=1,
        description="Evaluation batch size per device"
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Hardware & Precision
    # ─────────────────────────────────────────────────────────────────────────────
    
    device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Target device (auto-detected if AUTO)"
    )
    precision: Precision = Field(
        default=Precision.AUTO,
        description="Training precision mode"
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Nested Configuration Objects
    # ─────────────────────────────────────────────────────────────────────────────
    
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="Optimizer configuration"
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Learning rate scheduler configuration"
    )
    loss: LossConfig = Field(
        default_factory=LossConfig,
        description="Loss function configuration"
    )
    distributed: DistributedConfig = Field(
        default_factory=DistributedConfig,
        description="Distributed training configuration"
    )
    checkpoint: CheckpointConfig = Field(
        default_factory=CheckpointConfig,
        description="Checkpoint configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Evaluation Configuration
    # ─────────────────────────────────────────────────────────────────────────────
    
    evaluation_strategy: Literal["no", "steps", "epoch"] = Field(
        default="epoch",
        description="When to run evaluation"
    )
    eval_steps: int = Field(
        default=500, ge=1,
        description="Run evaluation every N steps (if strategy=steps)"
    )
    eval_delay: int = Field(
        default=0, ge=0,
        description="Number of epochs to wait before first evaluation"
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Performance Optimization
    # ─────────────────────────────────────────────────────────────────────────────
    
    dataloader_num_workers: int = Field(
        default=0, ge=0,
        description="Number of DataLoader worker processes"
    )
    dataloader_pin_memory: bool = Field(
        default=True,
        description="Pin memory for faster GPU transfer"
    )
    dataloader_prefetch_factor: int = Field(
        default=2, ge=1,
        description="Batches prefetched per worker"
    )
    
    # torch.compile integration
    torch_compile: bool = Field(
        default=False,
        description="Enable torch.compile for model acceleration"
    )
    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(
        default="default",
        description="torch.compile optimization mode"
    )
    torch_compile_backend: str = Field(
        default="inductor",
        description="torch.compile backend (inductor, cudagraphs, etc.)"
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Seed & Reproducibility
    # ─────────────────────────────────────────────────────────────────────────────
    
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    deterministic: bool = Field(
        default=False,
        description="Use deterministic algorithms (may reduce performance)"
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Hub Integration
    # ─────────────────────────────────────────────────────────────────────────────
    
    push_to_hub: bool = Field(
        default=False,
        description="Push model to HuggingFace Hub after training"
    )
    hub_model_id: Optional[str] = Field(
        default=None,
        description="Hub repository ID (user/model-name)"
    )
    hub_token: Optional[str] = Field(
        default=None,
        description="HuggingFace Hub token (use ${HF_TOKEN} for env var)"
    )
    hub_private_repo: bool = Field(
        default=False,
        description="Create private repository on Hub"
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Debug & Development
    # ─────────────────────────────────────────────────────────────────────────────
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode (extra logging, assertions)"
    )
    detect_anomaly: bool = Field(
        default=False,
        description="Enable autograd anomaly detection"
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Computed Properties
    # ─────────────────────────────────────────────────────────────────────────────
    
    @property
    def effective_batch_size(self) -> int:
        """
        Compute effective batch size including gradient accumulation
        and distributed processes.
        
        effective_batch = per_device * accumulation * world_size
        """
        return (
            self.per_device_train_batch_size
            * self.optimizer.gradient_accumulation_steps
            * self.distributed.world_size
        )
    
    @property
    def output_path(self) -> Path:
        """Return output directory as Path object."""
        return Path(self.output_dir)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Validators
    # ─────────────────────────────────────────────────────────────────────────────
    
    @model_validator(mode="after")
    def validate_arguments(self) -> "TrainingArguments":
        """Cross-field validation for configuration consistency."""
        # Validate output directory writability
        output_path = Path(self.output_dir)
        if output_path.exists() and not self.overwrite_output_dir:
            if any(output_path.iterdir()):
                # Not an error during loading, but log warning
                pass
        
        # Validate Hub configuration
        if self.push_to_hub and not self.hub_model_id:
            raise ValueError("hub_model_id is required when push_to_hub=True")
        
        return self


# ═════════════════════════════════════════════════════════════════════════════════
# YAML Loading Functions
# ═════════════════════════════════════════════════════════════════════════════════

def load_training_config(
    yaml_path: Union[str, Path],
    *,
    config_key: str = "training",
    interpolate_env: bool = True,
) -> TrainingArguments:
    """
    Load TrainingArguments from YAML file.
    
    Integrates with data_pipeline's existing configuration pattern.
    The training config can be in a dedicated training.yaml or as a
    section within a larger pipeline config.
    
    Args:
        yaml_path: Path to YAML configuration file
        config_key: Top-level key containing training config (default: "training")
        interpolate_env: Whether to substitute ${VAR} with environment variables
        
    Returns:
        Validated TrainingArguments instance
        
    Raises:
        YAMLParseError: If YAML syntax is invalid
        SchemaValidationError: If configuration doesn't match schema
        ConfigurationError: For other configuration issues
        
    Example:
        ```python
        # training.yaml:
        # training:
        #   output_dir: ./outputs
        #   num_epochs: 3
        
        args = load_training_config("training.yaml")
        ```
    """
    yaml_path = Path(yaml_path)
    
    # Check file exists
    if not yaml_path.exists():
        raise ConfigurationError(
            message=f"Configuration file not found: {yaml_path}",
            yaml_file=str(yaml_path),
            remediation="Ensure the YAML file exists at the specified path"
        )
    
    # Parse YAML
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        line = getattr(e, "problem_mark", None)
        raise YAMLParseError(
            message=f"Failed to parse YAML: {e}",
            yaml_file=str(yaml_path),
            line=line.line + 1 if line else None,
            column=line.column + 1 if line else None,
            cause=e
        )
    
    if raw_config is None:
        raise ConfigurationError(
            message="Empty configuration file",
            yaml_file=str(yaml_path),
            remediation="Add training configuration to the YAML file"
        )
    
    # Extract training section
    if config_key in raw_config:
        training_config = raw_config[config_key]
    else:
        # Assume entire file is training config
        training_config = raw_config
    
    # Environment variable interpolation
    if interpolate_env:
        training_config = interpolate_env_vars(training_config)
    
    # Validate with Pydantic
    try:
        return TrainingArguments.model_validate(training_config)
    except Exception as e:
        errors = []
        if hasattr(e, "errors"):
            for err in e.errors():
                loc = ".".join(str(x) for x in err.get("loc", []))
                msg = err.get("msg", "Unknown error")
                errors.append(f"{loc}: {msg}")
        
        raise SchemaValidationError(
            message=f"Training configuration validation failed",
            yaml_file=str(yaml_path),
            validation_errors=tuple(errors),
            cause=e
        )


def load_training_config_from_dict(
    config_dict: Dict[str, Any],
    *,
    interpolate_env: bool = True,
) -> TrainingArguments:
    """
    Create TrainingArguments from dictionary.
    
    Useful for programmatic configuration or testing.
    
    Args:
        config_dict: Configuration dictionary
        interpolate_env: Whether to substitute ${VAR} with environment variables
        
    Returns:
        Validated TrainingArguments instance
    """
    if interpolate_env:
        config_dict = interpolate_env_vars(config_dict)
    
    try:
        return TrainingArguments.model_validate(config_dict)
    except Exception as e:
        errors = []
        if hasattr(e, "errors"):
            for err in e.errors():
                loc = ".".join(str(x) for x in err.get("loc", []))
                msg = err.get("msg", "Unknown error")
                errors.append(f"{loc}: {msg}")
        
        raise SchemaValidationError(
            message=f"Training configuration validation failed",
            validation_errors=tuple(errors),
            cause=e
        )


def merge_configs(
    base: TrainingArguments,
    overrides: Dict[str, Any],
) -> TrainingArguments:
    """
    Merge override values into base configuration.
    
    Useful for CLI argument overrides on top of YAML config.
    
    Args:
        base: Base TrainingArguments instance
        overrides: Dictionary of override values (supports nested paths)
        
    Returns:
        New TrainingArguments with overrides applied
        
    Example:
        ```python
        base = load_training_config("config.yaml")
        args = merge_configs(base, {"num_epochs": 5, "optimizer.learning_rate": 1e-4})
        ```
    """
    base_dict = base.model_dump()
    
    for key, value in overrides.items():
        if "." in key:
            # Nested path: optimizer.learning_rate -> {"optimizer": {"learning_rate": ...}}
            parts = key.split(".")
            current = base_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            base_dict[key] = value
    
    return TrainingArguments.model_validate(base_dict)


# ═════════════════════════════════════════════════════════════════════════════════
# Export
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    "TrainingArguments",
    "load_training_config",
    "load_training_config_from_dict",
    "merge_configs",
    "interpolate_env_vars",
]
