# Core Module Documentation

The `core` module provides the foundational building blocks for the `data_pipeline`, enforcing strict type safety, deterministic error handling, and robust configuration management.

## 1. Type System (`core/types.py`)

The pipeline eschews standard Python exception-based control flow in favor of a Rust-inspired `Result` type. This ensures that errors are treated as values, enforcing exhaustive handling and explicit propagation.

### `Result[T, E]`

A disjoint union type representing either success (`Ok`) or failure (`Err`).

**Definition:**
```python
Result = Union[Ok[T], Err[E]]
```

#### Variants

- **`Ok(value: T)`**: Immutable dataclass containing the success value.
- **`Err(error: E)`**: Immutable dataclass containing the error information.

#### Operations

All operations are designed to be zero-cost at runtime while providing monadic error composition.

| Function | Signature | Description |
| :--- | :--- | :--- |
| `is_ok` | `Result -> bool` | O(1) check if variant is `Ok`. |
| `is_err` | `Result -> bool` | O(1) check if variant is `Err`. |
| `unwrap` | `Result -> T` | Returns value or raises `ValueError`. |
| `unwrap_err` | `Result -> E` | Returns error or raises `ValueError`. |
| `map_result` | `(T -> U) -> Result` | Functor map: transform success value. |
| `flat_map` | `(T -> Result[U, E]) -> Result` | Monadic bind: chain fallible operations. |
| `map_err` | `(E -> E') -> Result` | Transform error type. |

**Usage Example:**

```python
from data_pipeline.core.types import Result, Ok, Err, map_result

def safe_divide(a: int, b: int) -> Result[float, str]:
    if b == 0:
        return Err("Division by zero")
    return Ok(a / b)

result = safe_divide(10, 2)
if is_ok(result):
    print(unwrap(result))  # 5.0
```

---

## 2. Error Hierarchy (`core/errors.py`)

Exceptions are used only for fatal, unrecoverable states or when adapting legacy libraries. The hierarchy is structured to provide rich context for debugging.

**Base Class: `PipelineError`**
- Attributes: `message`, `context` (dict), `cause` (optional exception).
- Method: `with_context(**kwargs)` for context enrichment during propagation.

### Common Errors

| Error Class | Parent | Description |
| :--- | :--- | :--- |
| `ConfigurationError` | `PipelineError` | YAML parsing, schema validation, type mismatches. |
| `IntrospectionError` | `PipelineError` | Failures in dataset/split discovery on HF Hub. |
| `TokenizationError` | `PipelineError` | Input overflow, template rendering failure. |
| `DataLoadingError` | `PipelineError` | Batch collation, worker failures, shape mismatches. |

**Detailed/Specific Errors:**

- `DatasetNotFoundError` (Introspection)
- `ColumnNotFoundError` (Introspection)
- `TemplateRenderError` (Tokenization)
- `CollationError` (Data Loading)
- `ShapeError` (Data Loading)

---

## 3. Configuration (`core/config_schema.py`)

(See [Configuration Guide](configuration.md) for full schema details.)

The configuration system uses **strictly typed frozen dataclasses** to define the pipeline state.

**Key Features:**
- **Immutability**: Prevents side-effects and ensuring thread safety.
- **Validation**: Enforced at load time.
- **Serialization**: automated `to_dict` / `from_dict`.

### Core Schemas

- `PipelineConfig`: Root configuration.
- `DatasetConfig`: Source data definition.
- `TokenizerConfig`: Tokenization parameters.
- `PromptTemplate`: Jinja2/Chat format settings.
- `DataLoaderConfig`: PyTorch loader optimization.
