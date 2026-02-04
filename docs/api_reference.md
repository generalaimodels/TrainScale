# API Reference

This section documents the public API of the `data_pipeline` module.

## Core Core Classes

### `DataPipeline`

**Path**: `data_pipeline.pipeline.DataPipeline`

The main orchestrator class. It is thread-safe after initialization.

#### `__init__`
```python
def __init__(self, dataset_config=..., tokenizer=..., ...)
```
Initializes the pipeline manually.
- **Args**:
    - `dataset_config`: `DatasetConfig` object
    - `tokenizer`: Wrapper or HF tokenizer instance
    - ...
- **Use Case**: Advanced usage when not loading from YAML.

#### `from_config` (Class Method)
```python
@classmethod
def from_config(cls, config_path: str, token: str = None) -> Result[DataPipeline, ConfigurationError]
```
Factory method to create a pipeline from a YAML file.
- **Returns**: `Result` type containing the initialized pipeline or an error.

#### `get_dataloader`
```python
def get_dataloader(
    self,
    split: str,
    batch_size: int = None,
    shuffle: bool = None,
    distributed: bool = False,
    ...
) -> Result[DataLoader, PipelineError]
```
Constructs and returns a PyTorch `DataLoader`.
- **split**: The logical split name (e.g., "train").
- **distributed**: If `True`, automatically configured with `DistributedSampler`.
- **Returns**: `Result[DataLoader, PipelineError]`.

#### `discover`
```python
def discover(self) -> Result[DatasetMetadata, PipelineError]
```
Runs introspection on the dataset to find available splits and configs.

---

### `DatasetIntrospector`

**Path**: `data_pipeline.introspection.introspector`

Helper for zero-hardcoding dataset discovery.

#### `discover`
```python
def discover(self, dataset_id: str) -> Result[DatasetMetadata, IntrospectionError]
```
Fetches metadata from HuggingFace Hub without downloading the full dataset.
- **Returns**: `DatasetMetadata` containing configs, splits, and features.

---

### `PromptEngine`

**Path**: `data_pipeline.preprocessing.prompt_engine`

Handles text formatting and tokenization.

#### `process`
```python
def process(self, example: Dict[str, Any]) -> Result[ProcessedExample, TokenizationError]
```
Processes a single raw example into model inputs.
1.  Applies template (chat/custom).
2.  Truncates variable-length columns.
3.  Tokenizes text.
4.  Creats labels with input masking.

- **Returns**: `Result` with `input_ids`, `attention_mask`, `labels`.

---

### `DataLoaderBuilder`

**Path**: `data_pipeline.data.dataloader_factory`

Fluent builder pattern for creating DataLoaders.

```python
loader = (DataLoaderBuilder()
    .with_dataset(dataset)
    .with_batch_size(32)
    .with_distributed(rank=0, world_size=8)
    .with_padding(pad_token_id=0)
    .build())
```

---

## Type Definitions

### `Result[T, E]`

A generic type representing success (`Ok`) or failure (`Err`).

- `is_ok()`: `bool` - True if success.
- `is_err()`: `bool` - True if failure.
- `unwrap()`: `T` - Returns value or raises exception.
- `unwrap_err()`: `E` - Returns error value.

### `ProcessedExample`

Immutable dataclass for tokenized data.
```python
@dataclass
class ProcessedExample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    input_length: int
```

---

## Trainer Components

- [Trainer](trainer.md): Main `Trainer` class and configuration.
    - [Optimizers](trainer/optimizers.md): **SOTA** `Adam8bit`, `Lion`, `SophiaG`, `Prodigy`.
    - [Schedulers](trainer/schedulers.md): `WSD`, `CosineWithRestarts`.
    - [Loss Functions](trainer/loss.md): `FlashCrossEntropy`, `FocalLoss`, `DPO`, `ORPO`.
    - [Callbacks](trainer/callbacks.md): Event system and built-in callbacks.
    - [Metrics](trainer/metrics.md): Loss, throughput, and accuracy tracking.
    - [Distributed](trainer/distributed.md): DDP, FSDP-2, and Tensor Parallelism.
    - [Kernels](trainer/kernels.md): **Triton**-fused primitives for max speed.
