# Configuration Guide

This document details the configuration options available in the `data_pipeline` YAML schema. The configuration is strictly typed and validated upon loading.

## Root Structure

Top-level keys required for every configuration file.

| Key | Type | Description |
| :--- | :--- | :--- |
| `type` | `str` | Must be `"data_module"`. |
| `version` | `str` | Schema version (e.g., `"1.0"`). |
| `dataset` | `DatasetConfig` | Configuration for the source dataset. |
| `tokenizer` | `TokenizerConfig` | Configuration for tokenization. |
| `prompt_template` | `PromptTemplate` | Configuration for prompt formatting. |
| `dataloader` | `DataLoaderConfig` | Configuration for the PyTorch DataLoader. |
| `output_schema` | `OutputSchema` | (Optional) Schema for output tensors. |

---

## 1. Dataset Configuration (`dataset`)

Controls how the dataset is loaded from the HuggingFace Hub.

```yaml
dataset:
  name: "tatsu-lab/alpaca"
  config_name: "default"
  streaming: false
  columns: ["instruction", "input", "output"]
  splits:
    train:
      name: "train"
      sample_size: 1000
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `str` | **Required** | HuggingFace dataset identifier (e.g., `"org/repo"`). |
| `config_name` | `str` | `None` | Specific configuration name (subset) to load. |
| `streaming` | `bool` | `False` | If `True`, uses `IterableDataset` (lazy loading). |
| `columns` | `List[str]` | `[]` | List of columns to keep. Empty means keep all. |
| `splits` | `Dict` | `None` | Mapping of logical split names to `SplitSpec`. |
| `stages` | `Dict` | `None` | Advanced stage-based split configuration. |

### SplitSpec

Defines properties for a specific data split.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `str` | **Required** | The actual split name in the HF dataset (e.g., `"train"`). |
| `sample_size` | `int` | `None` | Limit the number of examples (useful for debugging). |
| `shuffle` | `bool` | `False` | Shuffle the data at the dataset level. |
| `seed` | `int` | `42` | Seed for shuffling. |

---

## 2. Tokenizer Configuration (`tokenizer`)

Controls the tokenizer loading and behavior.

```yaml
tokenizer:
  name_or_path: "meta-llama/Llama-2-7b-hf"
  max_length: 2048
  padding_side: "right"
  special_tokens:
    pad_token: "<pad>"
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name_or_path` | `str` | **Required** | HF model ID or local path to tokenizer. |
| `max_length` | `int` | `2048` | Maximum sequence length. |
| `padding` | `str` | `"max_length"` | Strategy: `"max_length"`, `"longest"`, `"do_not_pad"`. |
| `truncation` | `bool` | `True` | Whether to truncate sequences exceeding `max_length`. |
| `padding_side` | `str` | `"right"` | `"left"` or `"right"`. |
| `special_tokens` | `Dict` | `{}` | Map of special tokens (e.g., `pad_token`, `eos_token`) to override. |

---

## 3. Prompt Template Configuration (`prompt_template`)

Controls how raw data columns are formatted into the model input.

```yaml
prompt_template:
  format_type: "chat"
  input_columns: ["messages"]
  label_column: "response"
  mask_input: true
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `format_type` | `str` | `"custom"` | `"chat"`, `"completion"`, or `"custom"`. |
| `template` | `str` | `None` | Jinja2 template string (required for `"custom"`). |
| `input_columns` | `List[str]` | `[]` | Columns provided as input to the template. |
| `label_column` | `str` | `None` | Column containing the target/response. |
| `mask_input` | `bool` | `True` | If `True`, input tokens are masked (set to -100) in labels. |
| `add_bos` | `bool` | `True` | Prepend BOS token. |
| `add_eos` | `bool` | `True` | Append EOS token. |

---

## 4. DataLoader Configuration (`dataloader`)

Optimization settings for the PyTorch DataLoader.

```yaml
dataloader:
  batch_size: 8
  num_workers: 4
  pin_memory: true
  persistent_workers: true
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `batch_size` | `int` | `8` | Number of samples per batch. |
| `num_workers` | `int` | `4` | Number of subprocesses for data loading. |
| `pin_memory` | `bool` | `True` | Pin memory for faster host-to-device transfer. |
| `shuffle` | `bool` | `True` | Whether to shuffle the data every epoch. |
| `drop_last` | `bool` | `False` | Drop the last incomplete batch. |
| `prefetch_factor` | `int` | `2` | Number of batches loaded in advance by each worker. |
