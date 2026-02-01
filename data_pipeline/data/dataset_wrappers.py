# ════════════════════════════════════════════════════════════════════════════════
# Dataset Wrappers - Map-Style and Streaming Datasets
# ════════════════════════════════════════════════════════════════════════════════
# Wraps HuggingFace datasets with preprocessing applied.
# Supports both map-style (random access) and iterable (streaming) patterns.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import torch
from torch.utils.data import Dataset, IterableDataset

from data_pipeline.core.types import Result, Ok, Err, is_ok, unwrap
from data_pipeline.core.errors import DataLoadingError
from data_pipeline.preprocessing.prompt_engine import PromptEngine, ProcessedExample

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset
    from datasets import IterableDataset as HFIterableDataset


# ─────────────────────────────────────────────────────────────────────────────────
# Map-Style Preprocessed Dataset
# ─────────────────────────────────────────────────────────────────────────────────

class PreprocessedDataset(Dataset):
    """
    Map-style dataset with random access.
    
    Wraps a HuggingFace Dataset and applies preprocessing via PromptEngine.
    Supports caching of preprocessed examples.
    
    Attributes:
        hf_dataset: Underlying HuggingFace dataset
        prompt_engine: Engine for applying prompt templates
        cache_enabled: Whether to cache preprocessed examples
    
    Time Complexity:
        __getitem__: O(sequence_length) for preprocessing
        __len__: O(1)
    
    Space Complexity: O(dataset_size) if caching enabled
    """
    
    def __init__(
        self,
        hf_dataset: Any,  # HFDataset
        prompt_engine: PromptEngine,
        column_mapping: Optional[Dict[str, str]] = None,
        cache_enabled: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        """
        Initialize preprocessed dataset.
        
        Args:
            hf_dataset: HuggingFace Dataset instance
            prompt_engine: PromptEngine for tokenization/templating
            column_mapping: Optional column renaming (source -> target)
            cache_enabled: Cache preprocessed examples in memory
            transform: Optional additional transform function
        """
        super().__init__()
        self._dataset = hf_dataset
        self._engine = prompt_engine
        self._column_mapping = column_mapping or {}
        self._cache_enabled = cache_enabled
        self._transform = transform
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self._dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get preprocessed example at index.
        
        Args:
            idx: Example index
            
        Returns:
            Dict with input_ids, attention_mask, labels as tensors
            
        Raises:
            DataLoadingError: If preprocessing fails
        """
        # Check cache
        if self._cache_enabled and idx in self._cache:
            return self._cache[idx]
        
        # Get raw example
        raw_example = self._dataset[idx]
        
        # Apply column mapping
        example = self._apply_mapping(raw_example)
        
        # Apply optional transform
        if self._transform:
            example = self._transform(example)
        
        # Process through prompt engine
        result = self._engine.process(example)
        
        if isinstance(result, Err):
            raise DataLoadingError(
                message=f"Preprocessing failed for index {idx}",
                batch_index=idx,
                cause=result.error,
            )
        
        processed = unwrap(result)
        
        # Convert to tensors
        output = self._to_tensors(processed)
        
        # Cache if enabled
        if self._cache_enabled:
            self._cache[idx] = output
        
        return output
    
    def _apply_mapping(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Apply column name mapping."""
        if not self._column_mapping:
            return example
        
        result = {}
        for key, value in example.items():
            # Check if there's a target mapping
            for target, source in self._column_mapping.items():
                if key == source:
                    result[target] = value
                    break
            else:
                result[key] = value
        
        return result
    
    def _to_tensors(self, processed: ProcessedExample) -> Dict[str, torch.Tensor]:
        """Convert ProcessedExample to tensor dict."""
        return {
            "input_ids": torch.tensor(processed.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(processed.attention_mask, dtype=torch.long),
            "labels": torch.tensor(processed.labels, dtype=torch.long),
        }
    
    def clear_cache(self) -> None:
        """Clear the preprocessed example cache."""
        self._cache.clear()
    
    @property
    def features(self) -> Optional[Any]:
        """Get underlying dataset features."""
        return getattr(self._dataset, "features", None)
    
    def select(self, indices: List[int]) -> "PreprocessedDataset":
        """
        Create subset with selected indices.
        
        Args:
            indices: List of indices to select
            
        Returns:
            New PreprocessedDataset with subset
        """
        subset_dataset = self._dataset.select(indices)
        return PreprocessedDataset(
            hf_dataset=subset_dataset,
            prompt_engine=self._engine,
            column_mapping=self._column_mapping,
            cache_enabled=self._cache_enabled,
            transform=self._transform,
        )


# ─────────────────────────────────────────────────────────────────────────────────
# Streaming Preprocessed Dataset
# ─────────────────────────────────────────────────────────────────────────────────

class StreamingPreprocessedDataset(IterableDataset):
    """
    Streaming iterable dataset for large-scale data.
    
    Memory-efficient: processes examples on-the-fly without loading
    the entire dataset into memory.
    
    Attributes:
        hf_iterable: Underlying HuggingFace IterableDataset
        prompt_engine: Engine for applying prompt templates
    
    Time Complexity: O(sequence_length) per example
    Space Complexity: O(buffer_size) for prefetching
    """
    
    def __init__(
        self,
        hf_iterable: Any,  # HFIterableDataset
        prompt_engine: PromptEngine,
        column_mapping: Optional[Dict[str, str]] = None,
        buffer_size: int = 1000,
        skip_errors: bool = False,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        """
        Initialize streaming dataset.
        
        Args:
            hf_iterable: HuggingFace IterableDataset
            prompt_engine: PromptEngine for tokenization/templating
            column_mapping: Optional column renaming
            buffer_size: Buffer size for shuffling (if applicable)
            skip_errors: Skip examples that fail preprocessing
            transform: Optional additional transform
        """
        super().__init__()
        self._dataset = hf_iterable
        self._engine = prompt_engine
        self._column_mapping = column_mapping or {}
        self._buffer_size = buffer_size
        self._skip_errors = skip_errors
        self._transform = transform
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over preprocessed examples.
        
        Yields:
            Dict with input_ids, attention_mask, labels as tensors
        """
        worker_info = torch.utils.data.get_worker_info()
        
        # Get iterator - handle worker sharding if applicable
        if worker_info is not None:
            # Multi-worker: use worker-specific shard
            dataset = self._dataset.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id,
            )
        else:
            dataset = self._dataset
        
        for raw_example in dataset:
            # Apply column mapping
            example = self._apply_mapping(raw_example)
            
            # Apply optional transform
            if self._transform:
                try:
                    example = self._transform(example)
                except Exception as e:
                    if self._skip_errors:
                        continue
                    raise
            
            # Process through prompt engine
            result = self._engine.process(example)
            
            if isinstance(result, Err):
                if self._skip_errors:
                    continue
                raise DataLoadingError(
                    message="Streaming preprocessing failed",
                    cause=result.error,
                )
            
            processed = unwrap(result)
            yield self._to_tensors(processed)
    
    def _apply_mapping(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Apply column name mapping."""
        if not self._column_mapping:
            return example
        
        result = {}
        for key, value in example.items():
            for target, source in self._column_mapping.items():
                if key == source:
                    result[target] = value
                    break
            else:
                result[key] = value
        
        return result
    
    def _to_tensors(self, processed: ProcessedExample) -> Dict[str, torch.Tensor]:
        """Convert ProcessedExample to tensor dict."""
        return {
            "input_ids": torch.tensor(processed.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(processed.attention_mask, dtype=torch.long),
            "labels": torch.tensor(processed.labels, dtype=torch.long),
        }
    
    def shuffle(self, buffer_size: Optional[int] = None) -> "StreamingPreprocessedDataset":
        """
        Create shuffled version of dataset.
        
        Args:
            buffer_size: Buffer size for shuffle
            
        Returns:
            New StreamingPreprocessedDataset with shuffling
        """
        shuffled = self._dataset.shuffle(buffer_size=buffer_size or self._buffer_size)
        return StreamingPreprocessedDataset(
            hf_iterable=shuffled,
            prompt_engine=self._engine,
            column_mapping=self._column_mapping,
            buffer_size=buffer_size or self._buffer_size,
            skip_errors=self._skip_errors,
            transform=self._transform,
        )


# ─────────────────────────────────────────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────────────────────────────────────────

def create_preprocessed_dataset(
    hf_dataset: Any,
    prompt_engine: PromptEngine,
    column_mapping: Optional[Dict[str, str]] = None,
    streaming: bool = False,
    **kwargs,
) -> Union[PreprocessedDataset, StreamingPreprocessedDataset]:
    """
    Create appropriate preprocessed dataset wrapper.
    
    Args:
        hf_dataset: HuggingFace Dataset or IterableDataset
        prompt_engine: PromptEngine for preprocessing
        column_mapping: Optional column renaming
        streaming: Force streaming mode
        **kwargs: Additional arguments for dataset wrapper
        
    Returns:
        PreprocessedDataset or StreamingPreprocessedDataset
    """
    # Check if iterable dataset
    is_iterable = hasattr(hf_dataset, "__iter__") and not hasattr(hf_dataset, "__getitem__")
    
    if streaming or is_iterable:
        return StreamingPreprocessedDataset(
            hf_iterable=hf_dataset,
            prompt_engine=prompt_engine,
            column_mapping=column_mapping,
            **kwargs,
        )
    else:
        return PreprocessedDataset(
            hf_dataset=hf_dataset,
            prompt_engine=prompt_engine,
            column_mapping=column_mapping,
            **kwargs,
        )
