# Introspection Module Documentation

The `introspection` module enables **zero-hardcoding** and **auto-configuration** by discovering dataset structures dynamically. It handles schema alignment, split discovery, and metadata extraction from the HuggingFace Hub.

## 1. Dataset Introspector (`introspection/introspector.py`)

The `DatasetIntrospector` class discovers metadata without downloading the full dataset.

### Capabilities

- **Split Discovery**: Identifies available splits (train, validation, test) and their sizes.
- **Config Detection**: Lists available subsets (e.g., "en", "fr" for multilingual datasets).
- **Feature Extraction**: Analytics on column types and features.
- **Heuristic Selection**: Automatically picks the best config/split if not specified.

### API

```python
introspector = DatasetIntrospector(cache_dir="/tmp/cache")

# Discover metadata
result = introspector.discover("tatsu-lab/alpaca")

if result.is_ok():
    meta = result.unwrap()
    print(f"Default Config: {meta.default_config}")
    print(f"Splits: {list(meta.splits.keys())}")
```

---

## 2. Column Mapper (`introspection/column_mapper.py`)

Intelligent schema alignment using fuzzy matching and regex patterns. This allows the pipeline to adapt to datasets with non-standard column names (e.g., mapping "question" -> "instruction").

### matching Strategies (Priority Order)

1.  **Explicit Mapping**: User-provided overrides.
2.  **Exact Match**: Case-insensitive string equality.
3.  **Regex Patterns**: Matches common variations (e.g., `^user_?input$`).
4.  **Fuzzy Match**: Levenshtein distance (normalized similarity > threshold).

### Standard Schema

The mapper attempts to align columns to these standard targets:
- `instruction`: The task description.
- `input`: The context or input data.
- `output`: The target response/label.
- `label`: Classification targets.

### API

**`ColumnMapper` Class**

```python
mapper = ColumnMapper(min_fuzzy_score=0.8)

# Map source columns to target schema
mapping = mapper.map_columns(
    source_columns=["Question", "Context", "Answer"],
    target_schema=["instruction", "input", "output"]
)
# Result: {'instruction': 'Question', 'input': 'Context', 'output': 'Answer'}
```

**`fuzzy_match_columns` Function**
Convenience wrapper for one-off mapping.

```python
mapping_result = fuzzy_match_columns(
    source_cols, 
    ["instruction", "output"]
)
```

## 3. Complexity & Performance

- **Introspection**: Network-bound. Uses caching to minimize API calls (E-Tag based).
- **Mapping**: 
    - Time: O(M * N * L) where M=source cols, N=target cols, L=max string length.
    - Space: O(min(m, n)) for Levenshtein calculation.
- **Optimization**: Uses Wagner-Fischer algorithm with standard space optimization (2 rows) for edit distance.
