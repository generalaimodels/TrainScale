# Data Module Documentation

The `data` module implements the high-performance data loading layer, bridging the gap between raw datasets and the model's training loop. It handles streaming, batching, caching, and dynamic tensors construction.

## 1. Dataset Wrappers (`data/dataset_wrappers.py`)

Unified interfaces for both map-style (random access) and iterable (streaming) datasets, ensuring consistent preprocessing application.

### `PreprocessedDataset`

Map-style wrapper for in-memory datasets.

**Features:**
- **Caching**: Stores preprocessed tensors in memory (`cache_enabled=True`).
- **Random Access**: O(1) access to any example (after preprocessing).
- **Subsetting**: Efficient `select()` method.

```python
dataset = PreprocessedDataset(
    hf_dataset=raw_ds,
    prompt_engine=engine,
    cache_enabled=True
)
```

### `StreamingPreprocessedDataset`

Iterable wrapper for large-scale or infinite datasets.

**Features:**
- **Zero-Copy**: Processes examples on-the-fly (`__iter__`).
- **Sharding**: Automatically handles multi-worker sharding.
- **Resiliency**: Optional `skip_errors` to ignore malformed examples.

```python
from torch.utils.data import DataLoader

dataset = StreamingPreprocessedDataset(
    hf_iterable=streaming_ds,
    prompt_engine=engine,
    buffer_size=1000  # For local shuffling
)
loader = DataLoader(dataset, num_workers=4)  # Auto-shards across workers
```

**Factory Function:**
`create_preprocessed_dataset(hf_dataset, prompt_engine, streaming=False, ...)` automatically selects the correct wrapper.

---

## 2. Dynamic Collation (`data/collate.py`)

Optimized batch construction that perfectly aligns with loss function requirements (e.g., masking padding tokens).

### `create_collate_fn`

Factory for creating a high-performance collate function.

**Arguments:**
- `pad_token_id` (int): Value for padding `input_ids`.
- `label_pad_token_id` (int): Value for padding `labels` (default: -100).
- `padding_side` (str): "active" side ("right" or "left").
- `max_length` (int, optional): Force fixed length (default: longest in batch).
- `return_tensors` (bool): If True, returns PyTorch tensors.

**Algorithms:**
- **Vectorized Padding**: Uses `torch.full` and slice assignment for O(batch * len) complexity.
- **Automatic Masking**: Ensures `labels` are masked where `attention_mask` is 0.
- **Type Safety**: Casts tensors to correct dtypes based on `OutputSchema`.

### Specialized Collators

- **`create_causal_lm_collate`**: Shifts labels right by 1 for autoregressive training.
- **`create_seq2seq_collate`**: Handles dual inputs (encoder `input_ids`, decoder `decoder_input_ids`).

---

## 3. DataLoader Factory (`data/dataloader_factory.py`)

A Fluent Builder API for constructing optimized PyTorch DataLoaders.

**Class: `DataLoaderBuilder`**

**Methods:**
- `with_dataset(dataset)`
- `with_batch_size(int)`
- `with_distributed(rank, world_size)`: Adds `DistributedSampler`.
- `with_workers(num, pin_memory=True)`
- `build() -> DataLoader`

**Usage:**

```python
loader = (DataLoaderBuilder()
    .with_dataset(train_ds)
    .with_batch_size(32)
    .with_workers(8)
    .with_distributed(rank=dist.rank, world_size=dist.world_size)
    .build())
```
