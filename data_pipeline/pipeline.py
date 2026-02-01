# ════════════════════════════════════════════════════════════════════════════════
# Pipeline Orchestrator - End-to-End Data Pipeline
# ════════════════════════════════════════════════════════════════════════════════
# Main entry point for the data preprocessing pipeline.
# Orchestrates: YAML config → HF dataset → preprocessing → DataLoader
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)

from torch.utils.data import DataLoader

from data_pipeline.core.types import Result, Ok, Err, is_ok, unwrap
from data_pipeline.core.errors import (
    PipelineError,
    ConfigurationError,
    DataLoadingError,
)
from data_pipeline.core.config_schema import (
    PipelineConfig,
    DatasetConfig,
    TokenizerConfig,
    PromptTemplate,
    DataLoaderConfig,
    OutputSchema,
    load_config,
)
from data_pipeline.introspection.introspector import (
    DatasetIntrospector,
    discover_dataset,
    DatasetMetadata,
)
from data_pipeline.introspection.column_mapper import (
    ColumnMapper,
    fuzzy_match_columns,
)
from data_pipeline.preprocessing.tokenization import (
    TokenizerWrapper,
    create_tokenizer,
    wrap_tokenizer,
)
from data_pipeline.preprocessing.prompt_engine import PromptEngine
from data_pipeline.data.dataset_wrappers import (
    PreprocessedDataset,
    StreamingPreprocessedDataset,
    create_preprocessed_dataset,
)
from data_pipeline.data.dataloader_factory import (
    build_dataloader,
    DataLoaderBuilder,
)

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset
    from datasets import IterableDataset as HFIterableDataset
    from transformers import PreTrainedTokenizer


# ─────────────────────────────────────────────────────────────────────────────────
# Pipeline Class
# ─────────────────────────────────────────────────────────────────────────────────

class DataPipeline:
    """
    End-to-end data preprocessing pipeline.
    
    Orchestrates:
    1. YAML configuration loading
    2. Dataset introspection and loading
    3. Tokenization setup
    4. Prompt templating
    5. DataLoader construction
    
    Thread-safe: stateless after initialization.
    
    Example:
        # From YAML config
        pipeline = DataPipeline.from_config("config.yaml")
        train_loader = pipeline.get_dataloader("train")
        
        # Programmatic configuration
        pipeline = DataPipeline(
            dataset_name="tatsu-lab/alpaca",
            tokenizer=my_tokenizer,
        )
        train_loader = pipeline.get_dataloader("train")
    """
    
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        dataset_config: Optional[DatasetConfig] = None,
        tokenizer: Optional[Any] = None,
        tokenizer_config: Optional[TokenizerConfig] = None,
        prompt_template: Optional[PromptTemplate] = None,
        output_schema: Optional[OutputSchema] = None,
        dataloader_config: Optional[DataLoaderConfig] = None,
        pipeline_config: Optional[PipelineConfig] = None,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        """
        Initialize data pipeline.
        
        Args:
            dataset_name: HuggingFace dataset identifier (shorthand)
            dataset_config: Full dataset configuration
            tokenizer: Pre-loaded tokenizer (OR use tokenizer_config)
            tokenizer_config: Configuration for loading tokenizer
            prompt_template: Template for formatting examples
            output_schema: Schema for output tensors
            dataloader_config: DataLoader configuration
            pipeline_config: Full pipeline configuration (overrides others)
            token: HuggingFace API token
            trust_remote_code: Trust remote code
        """
        self._token = token or os.environ.get("HF_TOKEN")
        self._trust_remote_code = trust_remote_code
        
        # Use pipeline_config if provided, else build from components
        if pipeline_config is not None:
            self._config = pipeline_config
        else:
            self._config = self._build_config(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                tokenizer_config=tokenizer_config,
                prompt_template=prompt_template,
                output_schema=output_schema,
                dataloader_config=dataloader_config,
            )
        
        # Initialize components lazily
        self._tokenizer_wrapper: Optional[TokenizerWrapper] = None
        self._prompt_engine: Optional[PromptEngine] = None
        self._introspector: Optional[DatasetIntrospector] = None
        self._metadata: Optional[DatasetMetadata] = None
        self._datasets: Dict[str, Any] = {}
        self._dataloaders: Dict[str, DataLoader] = {}
        
        # Use provided tokenizer if given
        if tokenizer is not None:
            self._tokenizer_wrapper = wrap_tokenizer(
                tokenizer,
                config=self._config.tokenizer if self._config else None,
            )
    
    def _build_config(
        self,
        dataset_name: Optional[str],
        dataset_config: Optional[DatasetConfig],
        tokenizer_config: Optional[TokenizerConfig],
        prompt_template: Optional[PromptTemplate],
        output_schema: Optional[OutputSchema],
        dataloader_config: Optional[DataLoaderConfig],
    ) -> PipelineConfig:
        """Build PipelineConfig from individual components."""
        return PipelineConfig(
            type="data_module",
            version="1.0",
            dataset=dataset_config or DatasetConfig(name=dataset_name or ""),
            tokenizer=tokenizer_config or TokenizerConfig(name_or_path=""),
            prompt=prompt_template,
            output_schema=output_schema,
            dataloader=dataloader_config,
        )
    
    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        token: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> Result["DataPipeline", ConfigurationError]:
        """
        Create pipeline from YAML configuration file.
        
        Args:
            config_path: Path to YAML config file
            token: Optional HuggingFace token
            trust_remote_code: Trust remote code
            
        Returns:
            Result containing DataPipeline or error
        """
        config_result = load_config(config_path)
        if isinstance(config_result, Err):
            return config_result
        
        config = unwrap(config_result)
        
        try:
            pipeline = cls(
                pipeline_config=config,
                token=token,
                trust_remote_code=trust_remote_code,
            )
            return Ok(pipeline)
        except Exception as e:
            return Err(ConfigurationError(
                message=f"Failed to create pipeline: {e}",
                config_path=str(config_path),
                cause=e,
            ))
    
    @property
    def config(self) -> PipelineConfig:
        """Get pipeline configuration."""
        return self._config
    
    def _ensure_tokenizer(self) -> Result[TokenizerWrapper, PipelineError]:
        """Ensure tokenizer is loaded."""
        if self._tokenizer_wrapper is not None:
            return Ok(self._tokenizer_wrapper)
        
        if not self._config.tokenizer or not self._config.tokenizer.name_or_path:
            return Err(ConfigurationError(
                message="No tokenizer configured. Provide tokenizer or tokenizer_config.",
            ))
        
        result = create_tokenizer(
            self._config.tokenizer,
            trust_remote_code=self._trust_remote_code,
            token=self._token,
        )
        
        if isinstance(result, Err):
            return result
        
        self._tokenizer_wrapper = unwrap(result)
        return Ok(self._tokenizer_wrapper)
    
    def _ensure_prompt_engine(self) -> Result[PromptEngine, PipelineError]:
        """Ensure prompt engine is initialized."""
        if self._prompt_engine is not None:
            return Ok(self._prompt_engine)
        
        tokenizer_result = self._ensure_tokenizer()
        if isinstance(tokenizer_result, Err):
            return tokenizer_result
        
        tokenizer = unwrap(tokenizer_result)
        
        # Use configured template or create default
        template = self._config.prompt or PromptTemplate()
        
        self._prompt_engine = PromptEngine(
            template=template,
            tokenizer=tokenizer,
        )
        
        return Ok(self._prompt_engine)
    
    def _ensure_introspector(self) -> DatasetIntrospector:
        """Ensure introspector is initialized."""
        if self._introspector is None:
            self._introspector = DatasetIntrospector(token=self._token)
        return self._introspector
    
    def discover(self) -> Result[DatasetMetadata, PipelineError]:
        """
        Discover dataset structure without downloading.
        
        Returns:
            Result containing DatasetMetadata or error
        """
        if self._metadata is not None:
            return Ok(self._metadata)
        
        introspector = self._ensure_introspector()
        result = introspector.discover(self._config.dataset.name)
        
        if isinstance(result, Err):
            return result
        
        self._metadata = unwrap(result)
        return Ok(self._metadata)
    
    def load_dataset(
        self,
        split: str,
        streaming: Optional[bool] = None,
    ) -> Result[Any, PipelineError]:
        """
        Load a specific split of the dataset.
        
        Args:
            split: Split name (e.g., "train", "test")
            streaming: Override config streaming setting
            
        Returns:
            Result containing HuggingFace Dataset or error
        """
        cache_key = f"{split}_{streaming}"
        if cache_key in self._datasets:
            return Ok(self._datasets[cache_key])
        
        try:
            from datasets import load_dataset
        except ImportError:
            return Err(DataLoadingError(
                message="datasets library not installed",
            ))
        
        use_streaming = streaming if streaming is not None else self._config.dataset.streaming
        
        try:
            ds = load_dataset(
                self._config.dataset.name,
                name=self._config.dataset.config_name,
                split=split,
                streaming=use_streaming,
                token=self._token,
                trust_remote_code=self._trust_remote_code,
            )
            
            self._datasets[cache_key] = ds
            return Ok(ds)
            
        except Exception as e:
            return Err(DataLoadingError(
                message=f"Failed to load dataset split '{split}': {e}",
                cause=e,
            ))
    
    def get_preprocessed_dataset(
        self,
        split: str,
        streaming: Optional[bool] = None,
    ) -> Result[Union[PreprocessedDataset, StreamingPreprocessedDataset], PipelineError]:
        """
        Get preprocessed dataset for a split.
        
        Args:
            split: Split name
            streaming: Override streaming setting
            
        Returns:
            Result containing preprocessed dataset or error
        """
        # Ensure prompt engine is ready
        engine_result = self._ensure_prompt_engine()
        if isinstance(engine_result, Err):
            return engine_result
        
        engine = unwrap(engine_result)
        
        # Load raw dataset
        ds_result = self.load_dataset(split, streaming)
        if isinstance(ds_result, Err):
            return ds_result
        
        hf_dataset = unwrap(ds_result)
        
        # Create preprocessed wrapper
        use_streaming = streaming if streaming is not None else self._config.dataset.streaming
        
        try:
            preprocessed = create_preprocessed_dataset(
                hf_dataset=hf_dataset,
                prompt_engine=engine,
                column_mapping=dict(self._config.dataset.column_mapping),
                streaming=use_streaming,
            )
            return Ok(preprocessed)
        except Exception as e:
            return Err(DataLoadingError(
                message=f"Failed to create preprocessed dataset: {e}",
                cause=e,
            ))
    
    def get_dataloader(
        self,
        split: str,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        num_workers: Optional[int] = None,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        **kwargs,
    ) -> Result[DataLoader, PipelineError]:
        """
        Get DataLoader for a split.
        
        Args:
            split: Split name
            batch_size: Override config batch size
            shuffle: Override shuffle setting
            num_workers: Override num workers
            distributed: Enable distributed mode
            rank: Process rank
            world_size: Total processes
            **kwargs: Additional DataLoader arguments
            
        Returns:
            Result containing DataLoader or error
        """
        # Get preprocessed dataset
        ds_result = self.get_preprocessed_dataset(split)
        if isinstance(ds_result, Err):
            return ds_result
        
        dataset = unwrap(ds_result)
        
        # Build DataLoader
        dl_config = self._config.dataloader or DataLoaderConfig()
        
        try:
            # Get tokenizer for padding config
            tokenizer_result = self._ensure_tokenizer()
            pad_token_id = 0
            if is_ok(tokenizer_result):
                pad_token_id = unwrap(tokenizer_result).pad_token_id or 0
            
            loader = build_dataloader(
                dataset=dataset,
                batch_size=batch_size or dl_config.batch_size,
                shuffle=shuffle if shuffle is not None else dl_config.shuffle,
                num_workers=num_workers or dl_config.num_workers,
                pin_memory=dl_config.pin_memory,
                drop_last=dl_config.drop_last,
                pad_token_id=pad_token_id,
                output_schema=self._config.output_schema,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
                **kwargs,
            )
            
            return Ok(loader)
            
        except Exception as e:
            return Err(DataLoadingError(
                message=f"Failed to create DataLoader: {e}",
                cause=e,
            ))
    
    def get_splits(self) -> List[str]:
        """
        Get available splits from configuration.
        
        Returns:
            List of split names
        """
        if self._config.dataset.splits:
            return list(self._config.dataset.splits.keys())
        
        if self._config.dataset.stages:
            # Collect all splits from all stages
            splits = set()
            for stage in self._config.dataset.stages.values():
                for split in stage.splits.keys():
                    splits.add(split)
            return sorted(splits)
        
        # Default splits
        return ["train", "test", "validation"]
    
    def get_stage_splits(self, stage: str) -> List[str]:
        """
        Get splits for a specific stage.
        
        Args:
            stage: Stage name (e.g., "stage_1")
            
        Returns:
            List of split names in that stage
        """
        if not self._config.dataset.stages:
            return []
        
        stage_config = self._config.dataset.stages.get(stage)
        if stage_config is None:
            return []
        
        return list(stage_config.splits.keys())


# ─────────────────────────────────────────────────────────────────────────────────
# Quick Builder Functions
# ─────────────────────────────────────────────────────────────────────────────────

def create_pipeline(
    dataset_name: str,
    tokenizer: Any,
    template: Optional[str] = None,
    input_columns: Optional[List[str]] = None,
    label_column: Optional[str] = None,
    max_length: int = 2048,
    batch_size: int = 8,
    streaming: bool = False,
) -> DataPipeline:
    """
    Create pipeline with minimal configuration.
    
    Convenience function for common use cases.
    
    Args:
        dataset_name: HuggingFace dataset name
        tokenizer: Pre-loaded tokenizer
        template: Optional Jinja2 template string
        input_columns: Columns to use as input
        label_column: Column to use as label
        max_length: Maximum sequence length
        batch_size: Batch size
        streaming: Use streaming mode
        
    Returns:
        Configured DataPipeline
    """
    prompt_template = PromptTemplate(
        template=template,
        input_columns=tuple(input_columns) if input_columns else (),
        label_column=label_column,
    )
    
    dataloader_config = DataLoaderConfig(
        batch_size=batch_size,
    )
    
    dataset_config = DatasetConfig(
        name=dataset_name,
        streaming=streaming,
    )
    
    return DataPipeline(
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        dataloader_config=dataloader_config,
    )
