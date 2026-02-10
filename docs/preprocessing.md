# SOTA Preprocessing Module

The `preprocessing` module handles the transformation of raw text into model-ready tensors with "Beyond-SOTA" standards. It features a universal `PromptEngine` for all training stages (Pre-training to RL), a `LengthManager` with token-budget algebra, and a deterministic `TokenizerWrapper`.

## 1. Prompt Engine (`preprocessing/prompt_engine.py`)

The `PromptEngine` is the universal data formatter, constructing tokenised sequences for every stage of LLM development. It is designed to be **thread-safe** (stateless), **allocator-efficient** (zero-copy), and **result-oriented** (no exceptions for control flow).

### Supported Stages
*   **Pre-training**:
    *   **Causal LM**: Standard left-to-right modeling.
    *   **FIM (Fill-in-the-Middle)**: Support for code models (StarCoder/CodeLlama style) with `PSM` and `SPM` modes.
    *   **Packing**: Concatenates multiple documents into a single sequence with proper `position_ids` resets and `document_ids` masking.
*   **Fine-tuning (SFT)**:
    *   Support for **Chat** (OpenAI, ShareGPT), **Completion**, and **Instruction** formats.
    *   **Multi-turn Masking**: Can mask user/system prompts while training only on assistant responses.
    *   **Jinja2 Templates**: Full support for custom chat templates.
*   **Reinforcement Learning (RL)**:
    *   **DPO / ORPO**: Generates `(prompt, chosen, rejected)` triplets with shared prompt context.
    *   **GRPO**: Handles one prompt with $N$ candidate completions and scores.
    *   **PPO**: Tokenises queries for online generation.
    *   **KTO**: Binary (desirable/undesirable) feedback format.

### API Usage
```python
from data_pipeline.preprocessing.prompt_engine import PromptEngine
from data_pipeline.core.config_schema import PromptEngineConfig, TrainingStage

# Configure for Chat SFT
config = PromptEngineConfig(
    stage=TrainingStage.SFT,
    template="chatml",  # or custom Jinja2
    mask_prompt=True    # Mask user/system turns
)

engine = PromptEngine(config, tokenizer_wrapper, length_manager)

# Process a ShareGPT-style example
result = engine.process({
    "conversations": [
        {"from": "human", "value": "Explain quantum entanglement."},
        {"from": "gpt", "value": "It is a phenomenon where..."}
    ]
})

# Result is a ProcessedExample (input_ids, labels, attention_mask)
print(result.unwrap().input_ids)
```

---

## 2. Length Manager (`preprocessing/length_manager.py`)

The `LengthManager` goes beyond simple truncation, using **Token-Budget Algebra** to distribute capacity across variable-length columns. It ensures that critical content (like system prompts or labels) is preserved while optimizing the context window usage.

### Core Features
*   **TokenAwareContentDistributor**: An allocation engine that solves for the optimal distribution of tokens given column priorities and minimum requirements.
*   **Checked Arithmetic**: All budget computations use overflow-safe integer math (`checked_add`, `saturating_sub`).
*   **Smart Truncation Cascade**:
    1.  **Sentence Boundary**: Tries to cut at the end of a sentence (if >50% content remains).
    2.  **Clause Boundary**: Falls back to semicolons/commas (if >40% content remains).
    3.  **Word Boundary**: Cuts at whitespace.
    4.  **Hard Cut**: Final fallback to exact character limit.

### Configuration
```python
from data_pipeline.core.config_schema import LengthManagerConfig, ContentDistributionMode

config = LengthManagerConfig(
    total_max_length=4096,
    distribution_mode=ContentDistributionMode.ADAPTIVE, # Allocates based on content need
    bucket_boundaries=(128, 256, 512, 1024, 2048, 4096)  # For deterministic padding
)
```

---

## 3. Tokenizer Wrapper (`preprocessing/tokenization.py`)

A production-grade wrapper around HuggingFace tokenizers that solves common pitfalls like silent `None` propagation and nondeterministic special token handling.

### Key Guarantees
*   **No Silent Failures**: Special token IDs (`pad_token_id`, `eos_token_id`) are guaranteed to return valid integers, falling back to safe defaults (e.g., `eos_token` -> `0`) instead of `None`.
*   **Vocabulary Tracking**: Tracks if new special tokens were added, signaling the need for `model.resize_token_embeddings()`.
*   **Result[T, E]**: Factory methods return `Result` types, forcing callers to handle initialization errors explicitly.

### Example
```python
from data_pipeline.preprocessing.tokenization import create_tokenizer, TokenizerConfig

# Safe creation returns a Result
result = create_tokenizer(
    TokenizerConfig(name_or_path="meta-llama/Llama-3-8b")
)

if result.is_ok():
    tokenizer = result.unwrap()
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Pad ID (guaranteed): {tokenizer.pad_token_id}")
else:
    print(f"Error: {result.unwrap_err()}")
```
