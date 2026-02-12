# ════════════════════════════════════════════════════════════════════════════════
# DataLoader Factory - Production-Ready DataLoader Construction
# ════════════════════════════════════════════════════════════════════════════════
# Builds optimized PyTorch DataLoaders with proper configuration.
# Supports distributed training, dynamic batching, and worker management.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

import torch
from torch import Tensor
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
    DistributedSampler,
    Sampler,
)

from data_pipeline.core.config_schema import DataLoaderConfig, OutputSchema
from data_pipeline.data.collate import create_collate_fn, DEFAULT_PAD_TOKEN_ID, DEFAULT_LABEL_PAD_ID
from data_pipeline.data.dataset_wrappers import PreprocessedDataset, StreamingPreprocessedDataset


# ─────────────────────────────────────────────────────────────────────────────────
# DataLoader Builder
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass
class DataLoaderBuilder:
    """
    Builder for constructing production-ready DataLoaders.
    
    Provides fluent API for configuration with sensible defaults.
    Ensures proper sampler selection and collation setup.
    
    Example:
        loader = (DataLoaderBuilder()
            .with_dataset(dataset)
            .with_batch_size(32)
            .with_distributed(rank=0, world_size=8)
            .build())
    """
    
    # Core configuration
    dataset: Optional[Union[Dataset, IterableDataset]] = None
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # Padding configuration
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID
    label_pad_token_id: int = DEFAULT_LABEL_PAD_ID
    padding_side: Literal["left", "right"] = "right"
    max_length: Optional[int] = None
    
    # Distributed configuration
    distributed: bool = False
    rank: int = 0
    world_size: int = 1
    
    # Custom components
    collate_fn: Optional[Callable] = None
    sampler: Optional[Sampler] = None
    output_schema: Optional[OutputSchema] = None
    
    # Random seed
    seed: int = 42
    
    def with_dataset(
        self, 
        dataset: Union[Dataset, IterableDataset]
    ) -> "DataLoaderBuilder":
        """Set the dataset."""
        self.dataset = dataset
        return self
    
    def with_batch_size(self, batch_size: int) -> "DataLoaderBuilder":
        """Set batch size."""
        self.batch_size = batch_size
        return self
    
    def with_shuffle(self, shuffle: bool) -> "DataLoaderBuilder":
        """Set shuffle mode."""
        self.shuffle = shuffle
        return self
    
    def with_num_workers(self, num_workers: int) -> "DataLoaderBuilder":
        """Set number of workers."""
        self.num_workers = num_workers
        return self
    
    def with_pin_memory(self, pin_memory: bool) -> "DataLoaderBuilder":
        """Set pin memory mode."""
        self.pin_memory = pin_memory
        return self
    
    def with_drop_last(self, drop_last: bool) -> "DataLoaderBuilder":
        """Set drop last mode."""
        self.drop_last = drop_last
        return self
    
    def with_padding(
        self,
        pad_token_id: int,
        label_pad_token_id: int = DEFAULT_LABEL_PAD_ID,
        padding_side: Literal["left", "right"] = "right",
        max_length: Optional[int] = None,
    ) -> "DataLoaderBuilder":
        """Configure padding behavior."""
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.padding_side = padding_side
        self.max_length = max_length
        return self
    
    def with_distributed(
        self,
        rank: int,
        world_size: int,
    ) -> "DataLoaderBuilder":
        """Enable distributed mode."""
        self.distributed = True
        self.rank = rank
        self.world_size = world_size
        return self
    
    def with_collate_fn(self, collate_fn: Callable) -> "DataLoaderBuilder":
        """Set custom collate function."""
        self.collate_fn = collate_fn
        return self
    
    def with_sampler(self, sampler: Sampler) -> "DataLoaderBuilder":
        """Set custom sampler."""
        self.sampler = sampler
        return self
    
    def with_output_schema(self, schema: OutputSchema) -> "DataLoaderBuilder":
        """Set output schema for collation."""
        self.output_schema = schema
        return self
    
    def with_config(self, config: DataLoaderConfig) -> "DataLoaderBuilder":
        """Configure from DataLoaderConfig."""
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        self.drop_last = config.drop_last
        self.shuffle = config.shuffle
        self.prefetch_factor = config.prefetch_factor
        self.persistent_workers = config.persistent_workers
        return self
    
    def with_seed(self, seed: int) -> "DataLoaderBuilder":
        """Set random seed."""
        self.seed = seed
        return self
    
    def _create_sampler(self) -> Optional[Sampler]:
        """Create appropriate sampler based on configuration."""
        if self.sampler is not None:
            return self.sampler
        
        # Iterable datasets don't use samplers
        if isinstance(self.dataset, IterableDataset):
            return None
        
        if self.distributed:
            return DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=self.shuffle,
                seed=self.seed,
                drop_last=self.drop_last,
            )
        
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            return RandomSampler(self.dataset, generator=generator)
        
        return SequentialSampler(self.dataset)
    
    def _create_collate_fn(self) -> Callable:
        """Create collate function if not provided."""
        if self.collate_fn is not None:
            return self.collate_fn
        
        return create_collate_fn(
            pad_token_id=self.pad_token_id,
            label_pad_token_id=self.label_pad_token_id,
            padding_side=self.padding_side,
            max_length=self.max_length,
            output_schema=self.output_schema,
        )
    
    def build(self) -> DataLoader:
        """
        Build the DataLoader.
        
        Returns:
            Configured PyTorch DataLoader
            
        Raises:
            ValueError: If dataset not set
        """
        if self.dataset is None:
            raise ValueError("Dataset must be set before building DataLoader")
        
        sampler = self._create_sampler()
        collate_fn = self._create_collate_fn()
        
        # Determine shuffle setting
        # When using sampler, DataLoader shuffle must be False
        shuffle = None if sampler is not None else self.shuffle
        
        # Adjust settings for iterable datasets
        is_iterable = isinstance(self.dataset, IterableDataset)
        
        # Build DataLoader kwargs
        kwargs = {
            "dataset": self.dataset,
            "batch_size": self.batch_size,
            "collate_fn": collate_fn,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
        }
        
        # Add sampler/shuffle only for map-style datasets
        if not is_iterable:
            if sampler is not None:
                kwargs["sampler"] = sampler
            else:
                kwargs["shuffle"] = self.shuffle
        
        # Add prefetch and persistent workers only if num_workers > 0
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = self.persistent_workers
        
        return DataLoader(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────────

def build_dataloader(
    dataset: Union[Dataset, IterableDataset, PreprocessedDataset, StreamingPreprocessedDataset],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
    label_pad_token_id: int = DEFAULT_LABEL_PAD_ID,
    padding_side: Literal["left", "right"] = "right",
    max_length: Optional[int] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
    collate_fn: Optional[Callable] = None,
    output_schema: Optional[OutputSchema] = None,
    config: Optional[DataLoaderConfig] = None,
) -> DataLoader:
    """
    Build production-ready DataLoader with sensible defaults.
    
    Convenience wrapper around DataLoaderBuilder.
    
    Args:
        dataset: PyTorch Dataset or IterableDataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Pin memory for GPU transfer
        drop_last: Drop incomplete last batch
        pad_token_id: Token ID for padding
        label_pad_token_id: Value for padding labels
        padding_side: Which side to pad
        max_length: Maximum sequence length
        distributed: Enable distributed mode
        rank: Process rank (for distributed)
        world_size: Total processes (for distributed)
        seed: Random seed
        collate_fn: Optional custom collate function
        output_schema: Optional output schema
        config: Optional DataLoaderConfig (overrides individual args)
        
    Returns:
        Configured DataLoader
    """
    builder = DataLoaderBuilder(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        pad_token_id=pad_token_id,
        label_pad_token_id=label_pad_token_id,
        padding_side=padding_side,
        max_length=max_length,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        seed=seed,
        collate_fn=collate_fn,
        output_schema=output_schema,
    )
    
    # Apply config if provided
    if config is not None:
        builder.with_config(config)
    
    return builder.build()


def build_distributed_dataloaders(
    dataset: Dataset,
    world_size: int,
    batch_size: int = 8,
    **kwargs,
) -> List[DataLoader]:
    """
    Build DataLoaders for all ranks in distributed training.
    
    Useful for testing distributed setups locally.
    
    Args:
        dataset: Source dataset
        world_size: Number of processes
        batch_size: Per-GPU batch size
        **kwargs: Additional arguments for build_dataloader
        
    Returns:
        List of DataLoaders, one per rank
    """
    loaders = []
    for rank in range(world_size):
        loader = build_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            distributed=True,
            rank=rank,
            world_size=world_size,
            **kwargs,
        )
        loaders.append(loader)
    return loaders
