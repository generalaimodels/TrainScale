# Data Pipeline Documentation

## Overview

The `data_pipeline` module is a high-performance, production-grade system designed to orchestrate the entire lifecycle of data preprocessing for Large Language Model (LLM) training. It bridges the gap between raw HuggingFace datasets and optimized PyTorch DataLoaders, ensuring reliability, reproducibility, and efficiency.

Built with **zero-cost abstractions**, **robust error handling**, and **deterministic behavior**, this pipeline serves as the foundation for SOTA model training.

## Directory Structure

```text
data_pipeline/
├── core/                   # [Core] Fundamental types and configuration
│   ├── config_schema.py    # Configuration dataclasses (PipelineConfig, etc.)
│   ├── errors.py           # Structured error hierarchy
│   └── types.py            # Result type and generic utilities
├── data/                   # [Data] Loading, wrapping, and collation
│   ├── collate.py          # Dynamic batch construction strategies
│   ├── dataloader_factory.py # Builder for optimized PyTorch DataLoaders
│   └── dataset_wrappers.py # Map-style and Streaming wrappers
├── introspection/          # [Introspection] Auto-discovery mechanisms
│   ├── column_mapper.py    # Schema alignment and fuzzy matching
│   └── introspector.py     # Dataset metadata discovery
├── preprocessing/          # [Preprocessing] Transformation logic
│   ├── length_manager.py   # SOTA smart truncation and padding
│   ├── prompt_engine.py    # Template applications (Chat/Custom)
│   └── tokenization.py     # Tokenizer wrappers and encoding
├── trainer/                # [Trainer] SOTA Training Infrastructure
│   ├── base.py             # Main Trainer engine
│   ├── callbacks/          # Extensible callback system
│   ├── distributed/        # DDP/FSDP/Deepspeed integration
│   ├── loss/               # Loss functions with Triton kernels
│   ├── metrics/            # Training metrics (Perplexity, MFU)
│   ├── optimizers/         # SOTA Optimizers (Lion, Sophia, etc.)
│   ├── schedulers/         # Learning rate schedulers
│   └── training_config.py  # Unified training configuration
├── pipeline.py             # Main entry point class
└── __init__.py             # Value exports
```

## Architecture

The pipeline follows a modular architecture:

1.  **Configuration Layer (`core/config_schema.py`)**: Strongly-typed, immutable configuration objects (frozen dataclasses) that validate YAML inputs.
2.  **Introspection Layer (`introspection/`)**: Automatically discovers dataset structures (splits, columns, configs) without manual specification.
3.  **Preprocessing Layer (`preprocessing/`)**:
    *   **Prompt Engine**: Handles zero-shot/few-shot prompting, chat templates, and input masking.
    *   **Tokenization**: Wraps HF tokenizers with consistent padding/truncation logic.
    *   **Length Management**: Implements smart truncation strategies for variable-length inputs.
4.  **Data Layer (`data/`)**:
    *   **Dataset Wrappers**: Unified interface for memory-mapped and streaming datasets.
    *   **Collation**: Dynamic batch construction with padding and masking.
    *   **DataLoader Factory**: generic builder for highly optimized PyTorch DataLoaders.

## Key Features

- **YAML-Driven Control**: Define every aspect of your data flow in a readable YAML configuration.
- **Deep Introspection**: Auto-discovery of dataset capabilities to minimize boilerplate.
- **SOTA Prompting**: First-class support for `apply_chat_template`, OpenAI formats, and custom Jinja2 templates.
- **Robust Error Handling**: Uses Rust-style `Result` types (`Ok`/`Err`) to ensure no error goes unhandled.
- **Performance Optimized**:
    - Lazy initialization of components.
    - Zero-copy tensor operations where possible.
    - Native support for `IterableDataset` (streaming).

## Quick Start

### 1. Define Configuration (`pipeline_config.yaml`)

```yaml
type: "data_module"
version: "1.0"

dataset:
  name: "tatsu-lab/alpaca"
  columns: ["instruction", "input", "output"]

tokenizer:
  name_or_path: "meta-llama/Llama-2-7b-hf"
  max_length: 2048

prompt_template:
  format_type: "custom"
  template: |
    Below is an instruction that describes a task.
    
    ### Instruction:
    {instruction}
    
    ### Input:
    {input}
    
    ### Response:
    {output}
  input_columns: ["instruction", "input"]
  label_column: "output"

dataloader:
  batch_size: 4
  num_workers: 4
```

### 2. Initialize in Python

```python
from data_pipeline import DataPipeline

# Load pipeline from config
result = DataPipeline.from_config("pipeline_config.yaml")

if result.is_err():
    print(f"Configuration Error: {result.unwrap_err()}")
    exit(1)

pipeline = result.unwrap()

# Get a PyTorch DataLoader for the 'train' split
loader_result = pipeline.get_dataloader("train", shuffle=True)
train_loader = loader_result.unwrap()

# Iterate
for batch in train_loader:
    print(batch["input_ids"].shape)
    # torch.Size([4, 2048])
```

## Documentation Index

- [Configuration Guide](configuration.md): Detailed reference for YAML configuration options.
- [API Reference](api_reference.md): Comprehensive API documentation for classes and functions.
- [Core Module](core.md): Type system and error handling.
- [Data Module](data.md): Loading and batching internals.
- [Introspection Module](introspection.md): Discovery and mapping logic.
- [Preprocessing Module](preprocessing.md): Prompting and tokenization details.
- [Trainer Module](trainer.md): Training infrastructure guide.
