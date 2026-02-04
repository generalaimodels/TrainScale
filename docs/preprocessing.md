# Preprocessing Module Documentation

The `preprocessing` module handles the transformation of raw text into model-ready tensors. It includes SOTA implementations for prompt engineering, smart truncation, and tokenization.

## 1. Prompt Engine (`preprocessing/prompt_engine.py`)

The `PromptEngine` is responsible for applying templates to raw examples, handling special tokens, and managing the overall formatting pipeline.

### Features
- **Template Support**: Support for `chat` (OpenAI format), `completion`, and `custom` (Jinja2) templates.
- **Label Masking**: Automatically masks input tokens (sets label to -100) for supervised fine-tuning.
- **Special Tokens**: Handles BOS/EOS token insertion.

### API

```python
engine = PromptEngine(
    template=prompt_template_config,
    tokenizer=tokenizer_wrapper,
    length_manager=length_manager
)

# Process a single example
result = engine.process({
    "instruction": "Hello", 
    "output": "World"
})
```

---

## 2. Length Manager (`preprocessing/length_manager.py`)

A SOTA system for managing variable-length sequences. Unlike simple truncation, this manager uses intelligent strategies to preserve the most important content.

### Strategies

- **Smart Truncation**: Tries to cut at sentence boundaries first, then word boundaries, and finally character limits.
- **Priority-Based Allocation**: Allocates token budget to high-priority columns (e.g., `labels` get priority over `inputs`).
- **Content Distribution**:
    - `EQUAL`: Split remaining budget equally.
    - `PROPORTIONAL`: Split based on original text length.
    - `ADAPTIVE`: Heuristic-based allocation.

### Configuration (`LengthManagerConfig`)

```python
config = LengthManagerConfig(
    total_max_length=2048,
    distribution_mode=ContentDistributionMode.ADAPTIVE,
    default_truncation=TruncationStrategy.SMART
)
```

### Padding Strategies

- `LONGEST`: Dynamic padding to the longest sequence in the batch (most efficient).
- `MAX_LENGTH`: Pad to fixed `max_length`.
- `BUCKET`: Pad to the nearest bucket size (128, 256, 512, etc.) to reduce compute waste.

---

## 3. Tokenization (`preprocessing/tokenization.py`)

Wrapper around HuggingFace tokenizers to ensure consistency and correct configuration.

### `TokenizerWrapper`

- Enforces `padding_side` (left/right).
- Manages `pad_token` logic (auto-adds if missing).
- Provides consistent `encode` / `encode_batch` interfaces.

### `TokenizationError`

Rich error reporting for tokenization failures, including context about input length and model limits.
