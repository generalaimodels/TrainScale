# ════════════════════════════════════════════════════════════════════════════════
# Data Pipeline - SOTA HuggingFace to DataLoader Pipeline
# ════════════════════════════════════════════════════════════════════════════════
# A generalized, YAML-driven data preprocessing pipeline.
# Supports: auto-discovery, flexible stage-based splits, SDK-compatible templates.
# ════════════════════════════════════════════════════════════════════════════════

__version__ = "1.0.0"

# Core types and errors
from data_pipeline.core import (
    Ok,
    Err,
    Result,
    is_ok,
    is_err,
    unwrap,
    PipelineError,
    ConfigurationError,
    IntrospectionError,
    TokenizationError,
    DataLoadingError,
)

# Configuration
from data_pipeline.core.config_schema import (
    PipelineConfig,
    DatasetConfig,
    SplitSpec,
    StageConfig,
    TokenizerConfig,
    PromptTemplate,
    TensorSpec,
    OutputSchema,
    DataLoaderConfig,
    load_config,
)

# Introspection
from data_pipeline.introspection import (
    DatasetIntrospector,
    DatasetMetadata,
    discover_dataset,
    get_default_split,
    ColumnMapper,
    fuzzy_match_columns,
)

# Preprocessing
from data_pipeline.preprocessing import (
    TokenizerWrapper,
    create_tokenizer,
    wrap_tokenizer,
    PromptEngine,
    ProcessedExample,
    render_template,
    # Length management
    LengthManager,
    LengthManagerConfig,
    ColumnLengthConfig,
    TruncationStrategy,
    PaddingStrategy,
    create_length_manager,
    create_dynamic_collate_fn,
    smart_truncate,
    # SOTA content distribution
    TokenAwareContentDistributor,
    ContentDistributionMode,
    ColumnContentConfig,
)

# Data and DataLoader
from data_pipeline.data import (
    PreprocessedDataset,
    StreamingPreprocessedDataset,
    create_collate_fn,
    build_dataloader,
    DataLoaderBuilder,
)

# Main pipeline
from data_pipeline.pipeline import (
    DataPipeline,
    create_pipeline,
)

__all__ = [
    # Version
    "__version__",
    # Result types
    "Ok",
    "Err",
    "Result",
    "is_ok",
    "is_err",
    "unwrap",
    # Errors
    "PipelineError",
    "ConfigurationError",
    "IntrospectionError",
    "TokenizationError",
    "DataLoadingError",
    # Config
    "PipelineConfig",
    "DatasetConfig",
    "SplitSpec",
    "StageConfig",
    "TokenizerConfig",
    "PromptTemplate",
    "TensorSpec",
    "OutputSchema",
    "DataLoaderConfig",
    "load_config",
    # Introspection
    "DatasetIntrospector",
    "DatasetMetadata",
    "discover_dataset",
    "get_default_split",
    "ColumnMapper",
    "fuzzy_match_columns",
    # Preprocessing
    "TokenizerWrapper",
    "create_tokenizer",
    "wrap_tokenizer",
    "PromptEngine",
    "ProcessedExample",
    "render_template",
    # Length management
    "LengthManager",
    "LengthManagerConfig",
    "ColumnLengthConfig",
    "TruncationStrategy",
    "PaddingStrategy",
    "create_length_manager",
    "create_dynamic_collate_fn",
    "smart_truncate",
    # SOTA content distribution
    "TokenAwareContentDistributor",
    "ContentDistributionMode",
    "ColumnContentConfig",
    # Data
    "PreprocessedDataset",
    "StreamingPreprocessedDataset",
    "create_collate_fn",
    "build_dataloader",
    "DataLoaderBuilder",
    # Pipeline
    "DataPipeline",
    "create_pipeline",
]
