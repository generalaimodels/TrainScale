# ════════════════════════════════════════════════════════════════════════════════
# Configuration Schema - Frozen Dataclasses with Validation
# ════════════════════════════════════════════════════════════════════════════════
# Immutable configuration objects with slots for cache-line optimization.
# Supports flexible stage-based splits and user-defined column mappings.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
from enum import Enum, auto

import yaml

from data_pipeline.core.types import Result, Ok, Err
from data_pipeline.core.errors import (
    ConfigurationError,
    YAMLParseError,
    SchemaValidationError,
)


# ─────────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────────

# Valid padding strategies
PADDING_STRATEGIES = frozenset({"max_length", "longest", "do_not_pad"})

# Valid prompt format types
PROMPT_FORMAT_TYPES = frozenset({"chat", "completion", "custom"})

# Valid tensor dtypes
TENSOR_DTYPES = frozenset({"long", "int", "float", "half", "bfloat16", "bool"})


# ─────────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────────

class TrainingStage(Enum):
    PRE_TRAINING = auto()
    FINE_TUNING = auto()
    POST_TRAINING_RL = auto()

class PreTrainingFormat(Enum):
    CAUSAL_LM = auto()
    FIM = auto()
    SPAN_CORRUPTION = auto()

class FineTuningFormat(Enum):
    CHAT = auto()
    COMPLETION = auto()
    INSTRUCTION = auto()
    CUSTOM = auto()

class RLAlgorithm(Enum):
    DPO = auto()
    PPO = auto()
    KTO = auto()
    ORPO = auto()
    GRPO = auto()
    REWARD_MODEL = auto()

class FIMMode(Enum):
    PSM = auto()  # Prefix-Suffix-Middle
    SPM = auto()  # Suffix-Prefix-Middle

class PackingStrategy(Enum):
    GREEDY = auto()   # Pack until context length (was FULL)
    SINGLE = auto()   # One document per sequence (pad rest)

class TruncationStrategy(Enum):
    NONE = auto()
    SIMPLE = auto()
    WORD_BOUNDARY = auto()
    SENTENCE_BOUNDARY = auto()
    CLAUSE_BOUNDARY = auto()
    SMART = auto()

class ContentDistributionMode(Enum):
    EQUAL = auto()
    PROPORTIONAL = auto()
    RATIO = auto()
    PRIORITY = auto()
    ADAPTIVE = auto()

class PaddingStrategy(Enum):
    DO_NOT_PAD = auto()
    LONGEST = auto()
    MAX_LENGTH = auto()
    BUCKET = auto()
    POWER_OF_TWO = auto()


# ─────────────────────────────────────────────────────────────────────────────────
# Split Configuration
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SplitSpec:
    """
    Single split specification.
    
    Attributes:
        name: Actual HF split name (e.g., "train", "validation")
        alias: Optional user-facing alias for this split
        sample_size: Optional limit on samples (for debugging/dev)
        shuffle: Whether to shuffle this split on load
        seed: Random seed for shuffling (deterministic)
    """
    name: str
    alias: Optional[str] = None
    sample_size: Optional[int] = None
    shuffle: bool = False
    seed: int = 42
    
    def __post_init__(self) -> None:
        # Validate name is non-empty
        if not self.name or not self.name.strip():
            raise ValueError("SplitSpec.name cannot be empty")
        # Validate sample_size if provided
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {self.sample_size}")


@dataclass(frozen=True, slots=True)
class StageConfig:
    """
    Stage configuration with multiple named splits.
    
    Supports user-defined split names within a stage.
    Common pattern: train, test, eval per stage.
    
    Attributes:
        train: Training split spec
        test: Test split spec
        eval: Evaluation split spec
        extras: Additional user-defined splits
    """
    train: Optional[SplitSpec] = None
    test: Optional[SplitSpec] = None
    eval: Optional[SplitSpec] = None
    extras: Dict[str, SplitSpec] = field(default_factory=dict)
    
    def get_split(self, name: str) -> Optional[SplitSpec]:
        """Get split by name, checking standard and extras."""
        if name == "train":
            return self.train
        if name == "test":
            return self.test
        if name == "eval":
            return self.eval
        return self.extras.get(name)
    
    def all_splits(self) -> Dict[str, SplitSpec]:
        """Get all splits as dictionary."""
        result: Dict[str, SplitSpec] = {}
        if self.train:
            result["train"] = self.train
        if self.test:
            result["test"] = self.test
        if self.eval:
            result["eval"] = self.eval
        result.update(self.extras)
        return result


# ─────────────────────────────────────────────────────────────────────────────────
# Dataset Configuration
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """
    Generalized dataset configuration.
    
    Supports:
    - Simple splits: {"train": SplitSpec, "test": SplitSpec}
    - Multi-stage: {"stage_1": StageConfig, "stage_2": StageConfig}
    
    Attributes:
        name: HuggingFace dataset name (e.g., "tatsu-lab/alpaca")
        config_name: Optional HF config/subset name
        streaming: Use IterableDataset for memory efficiency
        splits: Direct split mapping (simple mode)
        stages: Stage-based splits (multi-stage mode)
        column_mapping: Source column -> target column mapping
        columns: Columns to keep (empty = all)
        revision: Dataset revision/version
    """
    name: str
    config_name: Optional[str] = None
    streaming: bool = False
    splits: Optional[Dict[str, SplitSpec]] = None
    stages: Optional[Dict[str, StageConfig]] = None
    column_mapping: Dict[str, str] = field(default_factory=dict)
    columns: List[str] = field(default_factory=list)
    revision: Optional[str] = None
    
    def __post_init__(self) -> None:
        # Validate dataset name
        if not self.name or not self.name.strip():
            raise ValueError("DatasetConfig.name cannot be empty")
        # Validate mutual exclusivity
        if self.splits and self.stages:
            raise ValueError("Cannot specify both 'splits' and 'stages'")
    
    def is_staged(self) -> bool:
        """Check if using multi-stage configuration."""
        return self.stages is not None
    
    def get_split_spec(
        self, 
        split_name: str, 
        stage_name: Optional[str] = None
    ) -> Optional[SplitSpec]:
        """
        Get SplitSpec by name, optionally from a specific stage.
        
        Args:
            split_name: Name of split (e.g., "train")
            stage_name: Optional stage name (e.g., "stage_1")
            
        Returns:
            SplitSpec or None if not found
        """
        if self.stages and stage_name:
            stage = self.stages.get(stage_name)
            return stage.get_split(split_name) if stage else None
        if self.splits:
            return self.splits.get(split_name)
        return None


# ─────────────────────────────────────────────────────────────────────────────────
# Tokenizer Configuration
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    """
    Tokenizer specification with special token handling.
    
    Attributes:
        name_or_path: HF tokenizer name or local path
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate
        padding_side: Which side to pad ("left" or "right")
        truncation_side: Which side to truncate
        special_tokens: Special token definitions
        add_special_tokens: Whether to add special tokens on encode
    """
    name_or_path: str
    max_length: int = 2048
    padding: Literal["max_length", "longest", "do_not_pad"] = "max_length"
    truncation: bool = True
    padding_side: Literal["left", "right"] = "right"
    truncation_side: Literal["left", "right"] = "right"
    special_tokens: Dict[str, str] = field(default_factory=dict)
    add_special_tokens: bool = True
    
    def __post_init__(self) -> None:
        # Validate tokenizer name
        if not self.name_or_path or not self.name_or_path.strip():
            raise ValueError("TokenizerConfig.name_or_path cannot be empty")
        # Validate max_length
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        # Validate padding strategy
        if self.padding not in PADDING_STRATEGIES:
            raise ValueError(
                f"Invalid padding '{self.padding}', "
                f"must be one of {PADDING_STRATEGIES}"
            )


# ─────────────────────────────────────────────────────────────────────────────────
# Prompt Template Configuration
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """
    SDK-compatible prompt template specification.
    
    Supports:
    - HuggingFace chat templates
    - OpenAI message format
    - Custom Jinja2 templates
    
    Attributes:
        format_type: Template format ("chat", "completion", "custom")
        template: Custom Jinja2 template string
        system_message: System message for chat format
        user_template: User message template
        assistant_template: Assistant message template
        input_columns: Columns used as input (masked in labels)
        label_column: Column used as label
        mask_input: Whether to mask input tokens in labels
        add_bos: Add BOS token at start
        add_eos: Add EOS token at end
    """
    format_type: Literal["chat", "completion", "custom"] = "custom"
    template: Optional[str] = None
    system_message: Optional[str] = None
    user_template: Optional[str] = None
    assistant_template: Optional[str] = None
    input_columns: List[str] = field(default_factory=list)
    label_column: Optional[str] = None
    mask_input: bool = True
    add_bos: bool = True
    add_eos: bool = True
    
    def __post_init__(self) -> None:
        # Validate format type
        if self.format_type not in PROMPT_FORMAT_TYPES:
            raise ValueError(
                f"Invalid format_type '{self.format_type}', "
                f"must be one of {PROMPT_FORMAT_TYPES}"
            )
        # Validate template for custom type
        if self.format_type == "custom" and not self.template:
            raise ValueError("Custom format requires 'template' to be set")
    
    def get_required_columns(self) -> List[str]:
        """Get all columns required by this template."""
        columns = list(self.input_columns)
        if self.label_column and self.label_column not in columns:
            columns.append(self.label_column)
        return columns


# ─────────────────────────────────────────────────────────────────────────────────
# Prompt Engine Configuration
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class FIMConfig:
    enabled: bool = False
    rate: float = 0.5
    mode: FIMMode = FIMMode.PSM
    spm_rate: float = 0.5
    seed: int = 42

@dataclass(frozen=True, slots=True)
class PackingConfig:
    strategy: PackingStrategy = PackingStrategy.GREEDY
    use_vsl: bool = False  # Variable Sequence Length (unpadding)

@dataclass(frozen=True, slots=True)
class SpanCorruptionConfig:
    enabled: bool = False
    noise_density: float = 0.15
    mean_noise_span_length: float = 3.0

@dataclass(frozen=True, slots=True)
class PreTrainingConfig:
    format: PreTrainingFormat = PreTrainingFormat.CAUSAL_LM
    fim: FIMConfig = field(default_factory=FIMConfig)
    packing: PackingConfig = field(default_factory=PackingConfig)
    span_corruption: SpanCorruptionConfig = field(default_factory=SpanCorruptionConfig)
    
    # Data columns
    text_column: str = "text"
    
    # Tokenization settings
    add_bos: bool = True
    add_eos: bool = True

@dataclass(frozen=True, slots=True)
class MultiTurnConfig:
    enabled: bool = True
    max_turns: Optional[int] = None
    mask_user_turns: bool = True  # Only train on assistant turns

@dataclass(frozen=True, slots=True)
class FineTuningConfig:
    format: FineTuningFormat = FineTuningFormat.CHAT
    multiturn: MultiTurnConfig = field(default_factory=MultiTurnConfig)
    train_on_inputs: bool = False
    
    # Instruction format settings
    instruction_column: str = "instruction"
    input_column: str = "input"
    output_column: str = "output"
    
    # General generation settings
    mask_input: bool = True
    add_eos: bool = True

@dataclass(frozen=True, slots=True)
class RLColumnMapping:
    chosen: str = "chosen"
    rejected: str = "rejected"
    prompt: str = "prompt"
    response: str = "response"
    label: str = "label"       # For KTO (bool)
    weight: str = "weight"     # For KTO (float)
    reward: str = "reward"     # For numeric rewards

@dataclass(frozen=True, slots=True)
class RLConfig:
    algorithm: RLAlgorithm = RLAlgorithm.DPO
    mapping: RLColumnMapping = field(default_factory=RLColumnMapping)
    beta: float = 0.1
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None
    
    # Generation settings
    mask_prompt: bool = True
    system_message: Optional[str] = None
    add_bos: bool = True
    add_eos: bool = True

@dataclass(frozen=True, slots=True)
class PromptEngineConfig:
    """
    Unified configuration for all training stages.
    """
    stage: TrainingStage = TrainingStage.FINE_TUNING
    max_length: int = 2048
    # Stage-specific sub-configs.
    pretraining: PreTrainingConfig = field(default_factory=PreTrainingConfig)
    finetuning: FineTuningConfig = field(default_factory=FineTuningConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    # Legacy compatibility — forwarded to fine-tuning if present.
    template: Optional[PromptTemplate] = None
    # Column mapping override (src → dst).
    column_mapping: Dict[str, str] = field(default_factory=dict)
    # Per-column character limits (legacy LengthManager integration).
    per_column_limits: Dict[str, int] = field(default_factory=dict)
    # Truncation default.
    truncation_strategy: TruncationStrategy = TruncationStrategy.SMART


# ─────────────────────────────────────────────────────────────────────────────────
# Output Tensor Schema (Loss Alignment)
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class TensorSpec:
    """
    Specification for output tensor format.
    
    Ensures perfect compatibility with loss functions.
    
    Attributes:
        dtype: PyTorch dtype as string
        pad_value: Value used for padding (e.g., -100 for labels)
        requires_grad: Whether tensor requires gradient
    """
    dtype: str = "long"
    pad_value: Optional[int] = None
    requires_grad: bool = False
    
    def __post_init__(self) -> None:
        if self.dtype not in TENSOR_DTYPES:
            raise ValueError(
                f"Invalid dtype '{self.dtype}', "
                f"must be one of {TENSOR_DTYPES}"
            )


@dataclass(frozen=True, slots=True)
class OutputSchema:
    """
    Complete output schema for DataLoader batches.
    
    Standard outputs for language modeling:
    - input_ids: (batch, seq_len), torch.long
    - attention_mask: (batch, seq_len), torch.long
    - labels: (batch, seq_len), torch.long with -100 for ignored
    
    Attributes:
        input_ids: Spec for input token IDs
        attention_mask: Spec for attention mask
        labels: Spec for labels (with pad_value=-100)
        extras: Additional user-defined tensors
    """
    input_ids: TensorSpec = field(default_factory=lambda: TensorSpec("long"))
    attention_mask: TensorSpec = field(default_factory=lambda: TensorSpec("long", pad_value=0))
    labels: TensorSpec = field(default_factory=lambda: TensorSpec("long", pad_value=-100))
    extras: Dict[str, TensorSpec] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────────
# DataLoader Configuration
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class DataLoaderConfig:
    """
    PyTorch DataLoader configuration.
    
    Attributes:
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Pin memory for GPU transfer
        drop_last: Drop incomplete last batch
        shuffle: Shuffle data
        prefetch_factor: Batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
    """
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    shuffle: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")
        if self.prefetch_factor <= 0:
            raise ValueError(f"prefetch_factor must be positive, got {self.prefetch_factor}")


# ─────────────────────────────────────────────────────────────────────────────────
# Complete Pipeline Configuration
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """
    Complete pipeline configuration.
    
    Combines all sub-configs into single configuration object.
    Supports type-module pattern with version.
    
    Attributes:
        type: Module type identifier ("data_module")
        version: Configuration version
        dataset: Dataset configuration
        tokenizer: Tokenizer configuration
        prompt_template: Prompt template configuration
        output_schema: Output tensor schema
        dataloader: DataLoader configuration
    """
    type: str
    version: str
    dataset: DatasetConfig
    tokenizer: TokenizerConfig
    prompt_engine: PromptEngineConfig = field(default_factory=PromptEngineConfig)
    # Legacy field - kept for backward compatibility but mapped to proper PromptEngineConfig
    prompt_template: Optional[PromptTemplate] = None
    output_schema: OutputSchema = field(default_factory=OutputSchema)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    
    def __post_init__(self) -> None:
        if self.type != "data_module":
            raise ValueError(f"Invalid type '{self.type}', expected 'data_module'")


# ─────────────────────────────────────────────────────────────────────────────────
# YAML Parsing
# ─────────────────────────────────────────────────────────────────────────────────

def _parse_split_spec(data: Dict[str, Any], path: str) -> SplitSpec:
    """Parse SplitSpec from dict."""
    if not isinstance(data, dict):
        raise SchemaValidationError(
            message="Expected dict for SplitSpec",
            field_path=path,
            expected="dict",
            got=type(data).__name__
        )
    return SplitSpec(
        name=data.get("name", ""),
        alias=data.get("alias"),
        sample_size=data.get("sample_size"),
        shuffle=data.get("shuffle", False),
        seed=data.get("seed", 42),
    )


def _parse_stage_config(data: Dict[str, Any], path: str) -> StageConfig:
    """Parse StageConfig from dict."""
    train = _parse_split_spec(data["train"], f"{path}.train") if "train" in data else None
    test = _parse_split_spec(data["test"], f"{path}.test") if "test" in data else None
    eval_ = _parse_split_spec(data["eval"], f"{path}.eval") if "eval" in data else None
    
    # Parse extras (any other keys)
    extras = {}
    for key, value in data.items():
        if key not in ("train", "test", "eval") and isinstance(value, dict):
            extras[key] = _parse_split_spec(value, f"{path}.{key}")
    
    return StageConfig(train=train, test=test, eval=eval_, extras=extras)


def _parse_dataset_config(data: Dict[str, Any]) -> DatasetConfig:
    """Parse DatasetConfig from dict."""
    # Parse splits if present
    splits = None
    if "splits" in data:
        splits = {
            k: _parse_split_spec(v, f"dataset.splits.{k}")
            for k, v in data["splits"].items()
        }
    
    # Parse stages if present
    stages = None
    if "stages" in data:
        stages = {
            k: _parse_stage_config(v, f"dataset.stages.{k}")
            for k, v in data["stages"].items()
        }
    
    return DatasetConfig(
        name=data.get("name", ""),
        config_name=data.get("config_name"),
        streaming=data.get("streaming", False),
        splits=splits,
        stages=stages,
        column_mapping=data.get("column_mapping", {}),
        columns=data.get("columns", []),
        revision=data.get("revision"),
    )


def _parse_tokenizer_config(data: Dict[str, Any]) -> TokenizerConfig:
    """Parse TokenizerConfig from dict."""
    return TokenizerConfig(
        name_or_path=data.get("name_or_path", ""),
        max_length=data.get("max_length", 2048),
        padding=data.get("padding", "max_length"),
        truncation=data.get("truncation", True),
        padding_side=data.get("padding_side", "right"),
        truncation_side=data.get("truncation_side", "right"),
        special_tokens=data.get("special_tokens", {}),
        add_special_tokens=data.get("add_special_tokens", True),
    )


def _parse_prompt_engine_config(data: Dict[str, Any]) -> PromptEngineConfig:
    """Parse PromptEngineConfig from dict."""
    stage_str = data.get("stage", "FINE_TUNING").upper()
    try:
        stage = TrainingStage[stage_str]
    except KeyError:
        stage = TrainingStage.FINE_TUNING

    # Parse sub-configs (simplified for now, ideally full recursive parsing)
    # In a full implementation, we'd have _parse_pretraining_config, etc.
    # For now, we assume dicts are structured correctly or rely on default_factory
    
    trunc_str = data.get("truncation_strategy", "SMART").upper()
    try:
        trunc_strat = TruncationStrategy[trunc_str]
    except KeyError:
        trunc_strat = TruncationStrategy.SMART

    return PromptEngineConfig(
        stage=stage,
        max_length=data.get("max_length", 2048),
        column_mapping=data.get("column_mapping", {}),
        per_column_limits=data.get("per_column_limits", {}),
        truncation_strategy=trunc_strat,
        # TODO: Add recursive parsing for sub-configs if needed for strict validation
    )


def _parse_prompt_template(data: Dict[str, Any]) -> PromptTemplate:
    """Parse PromptTemplate from dict."""
    return PromptTemplate(
        format_type=data.get("format_type", "custom"),
        template=data.get("template"),
        system_message=data.get("system_message"),
        user_template=data.get("user_template"),
        assistant_template=data.get("assistant_template"),
        input_columns=data.get("input_columns", []),
        label_column=data.get("label_column"),
        mask_input=data.get("mask_input", True),
        add_bos=data.get("add_bos", True),
        add_eos=data.get("add_eos", True),
    )


def _parse_tensor_spec(data: Dict[str, Any]) -> TensorSpec:
    """Parse TensorSpec from dict."""
    return TensorSpec(
        dtype=data.get("dtype", "long"),
        pad_value=data.get("pad_value"),
        requires_grad=data.get("requires_grad", False),
    )


def _parse_output_schema(data: Dict[str, Any]) -> OutputSchema:
    """Parse OutputSchema from dict."""
    input_ids = _parse_tensor_spec(data.get("input_ids", {}))
    attention_mask = _parse_tensor_spec(data.get("attention_mask", {"pad_value": 0}))
    labels = _parse_tensor_spec(data.get("labels", {"pad_value": -100}))
    
    extras = {}
    for key, value in data.items():
        if key not in ("input_ids", "attention_mask", "labels"):
            extras[key] = _parse_tensor_spec(value)
    
    return OutputSchema(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        extras=extras,
    )


def _parse_dataloader_config(data: Dict[str, Any]) -> DataLoaderConfig:
    """Parse DataLoaderConfig from dict."""
    return DataLoaderConfig(
        batch_size=data.get("batch_size", 8),
        num_workers=data.get("num_workers", 4),
        pin_memory=data.get("pin_memory", True),
        drop_last=data.get("drop_last", False),
        shuffle=data.get("shuffle", True),
        prefetch_factor=data.get("prefetch_factor", 2),
        persistent_workers=data.get("persistent_workers", True),
    )


def load_config(path: Union[str, Path]) -> Result[PipelineConfig, ConfigurationError]:
    """
    Load and parse pipeline configuration from YAML file.
    
    Time Complexity: O(n) where n is config size
    Space Complexity: O(n)
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        Result containing PipelineConfig or ConfigurationError
    """
    path = Path(path)
    
    # Check file exists
    if not path.exists():
        return Err(ConfigurationError(
            message=f"Configuration file not found: {path}",
            context={"path": str(path)}
        ))
    
    # Parse YAML
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return Err(YAMLParseError(
            message="Failed to parse YAML",
            cause=e,
            line=getattr(e, "problem_mark", None) and e.problem_mark.line,
            column=getattr(e, "problem_mark", None) and e.problem_mark.column,
        ))
    
    if not isinstance(data, dict):
        return Err(SchemaValidationError(
            message="Root must be a mapping",
            expected="dict",
            got=type(data).__name__
        ))
    
    # Validate required top-level keys
    required_keys = {"type", "version", "dataset", "tokenizer"}
    # prompt_template is now optional if prompt_engine is present, or for legacy support

    missing = required_keys - set(data.keys())
    if missing:
        return Err(SchemaValidationError(
            message=f"Missing required keys: {missing}",
            context={"missing_keys": list(missing)}
        ))
    
    # Parse sub-configs
    try:
        config = PipelineConfig(
            type=data["type"],
            version=data["version"],
            dataset=_parse_dataset_config(data["dataset"]),
            tokenizer=_parse_tokenizer_config(data["tokenizer"]),
            prompt_engine=_parse_prompt_engine_config(data.get("prompt_engine", {})),
            prompt_template=_parse_prompt_template(data.get("prompt_template", {})),
            output_schema=_parse_output_schema(data.get("output_schema", {})),
            dataloader=_parse_dataloader_config(data.get("dataloader", {})),
        )
        return Ok(config)
    except (ValueError, KeyError, TypeError) as e:
        return Err(SchemaValidationError(
            message=f"Schema validation failed: {e}",
            cause=e
        ))
