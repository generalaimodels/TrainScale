# ════════════════════════════════════════════════════════════════════════════════
# Custom Exceptions - Hierarchical Error Types
# ════════════════════════════════════════════════════════════════════════════════
# Structured exception hierarchy for the data pipeline.
# Each exception carries context for debugging and recovery.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────────
# Base Pipeline Error
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineError(Exception):
    """
    Base exception for all pipeline errors.
    
    Carries structured context for debugging.
    All pipeline exceptions inherit from this.
    """
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    cause: Optional[Exception] = None
    
    def __post_init__(self) -> None:
        super().__init__(self.message)
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            parts.append(f"[{ctx_str}]")
        if self.cause:
            parts.append(f"(caused by: {self.cause})")
        return " ".join(parts)
    
    def with_context(self, **kwargs: Any) -> PipelineError:
        """Add additional context and return self for chaining."""
        self.context.update(kwargs)
        return self


# ─────────────────────────────────────────────────────────────────────────────────
# Configuration Errors
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass
class ConfigurationError(PipelineError):
    """
    Error in YAML configuration parsing or validation.
    
    Raised when:
    - YAML syntax is invalid
    - Required fields are missing
    - Field types don't match schema
    - Field values are out of valid range
    """
    field_path: Optional[str] = None
    expected: Optional[str] = None
    got: Optional[str] = None
    
    def __str__(self) -> str:
        base = super().__str__()
        if self.field_path:
            base = f"[{self.field_path}] {base}"
        if self.expected and self.got:
            base += f" (expected {self.expected}, got {self.got})"
        return base


@dataclass
class YAMLParseError(ConfigurationError):
    """Error parsing YAML file."""
    line: Optional[int] = None
    column: Optional[int] = None
    
    def __str__(self) -> str:
        base = super().__str__()
        if self.line is not None:
            base += f" at line {self.line}"
            if self.column is not None:
                base += f", column {self.column}"
        return base


@dataclass  
class SchemaValidationError(ConfigurationError):
    """Schema validation failed."""
    pass


# ─────────────────────────────────────────────────────────────────────────────────
# Introspection Errors
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass
class IntrospectionError(PipelineError):
    """
    Error during dataset introspection.
    
    Raised when:
    - Dataset not found on Hub
    - Split/config doesn't exist
    - Schema discovery fails
    - Network/API errors
    """
    dataset_id: Optional[str] = None
    split: Optional[str] = None
    config: Optional[str] = None
    
    def __str__(self) -> str:
        base = super().__str__()
        parts = []
        if self.dataset_id:
            parts.append(f"dataset={self.dataset_id!r}")
        if self.split:
            parts.append(f"split={self.split!r}")
        if self.config:
            parts.append(f"config={self.config!r}")
        if parts:
            base += f" [{', '.join(parts)}]"
        return base


@dataclass
class DatasetNotFoundError(IntrospectionError):
    """Dataset does not exist on Hub."""
    pass


@dataclass
class SplitNotFoundError(IntrospectionError):
    """Split does not exist in dataset."""
    available_splits: tuple = field(default_factory=tuple)
    
    def __str__(self) -> str:
        base = super().__str__()
        if self.available_splits:
            base += f" (available: {', '.join(self.available_splits)})"
        return base


@dataclass
class ColumnNotFoundError(IntrospectionError):
    """Column does not exist in dataset."""
    column: Optional[str] = None
    available_columns: tuple = field(default_factory=tuple)
    
    def __str__(self) -> str:
        base = super().__str__()
        if self.column:
            base = f"Column {self.column!r} not found. {base}"
        if self.available_columns:
            base += f" (available: {', '.join(self.available_columns)})"
        return base


# ─────────────────────────────────────────────────────────────────────────────────
# Tokenization Errors
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenizationError(PipelineError):
    """
    Error during tokenization.
    
    Raised when:
    - Tokenizer not found/loadable
    - Input too long for max_length
    - Invalid special token configuration
    - Template rendering fails
    """
    tokenizer_name: Optional[str] = None
    input_length: Optional[int] = None
    max_length: Optional[int] = None
    
    def __str__(self) -> str:
        base = super().__str__()
        if self.tokenizer_name:
            base += f" [tokenizer={self.tokenizer_name!r}]"
        if self.input_length and self.max_length:
            base += f" (input_length={self.input_length}, max_length={self.max_length})"
        return base


@dataclass
class TemplateRenderError(TokenizationError):
    """Error rendering prompt template."""
    template: Optional[str] = None
    missing_vars: tuple = field(default_factory=tuple)
    
    def __str__(self) -> str:
        base = super().__str__()
        if self.missing_vars:
            base += f" (missing variables: {', '.join(self.missing_vars)})"
        return base


# ─────────────────────────────────────────────────────────────────────────────────
# Data Loading Errors
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass
class DataLoadingError(PipelineError):
    """
    Error during data loading or batching.
    
    Raised when:
    - DataLoader worker fails
    - Collation fails
    - Memory allocation fails
    - Invalid batch structure
    """
    batch_index: Optional[int] = None
    worker_id: Optional[int] = None
    
    def __str__(self) -> str:
        base = super().__str__()
        if self.batch_index is not None:
            base += f" [batch={self.batch_index}]"
        if self.worker_id is not None:
            base += f" [worker={self.worker_id}]"
        return base


@dataclass
class CollationError(DataLoadingError):
    """Error during batch collation."""
    tensor_name: Optional[str] = None
    shapes: tuple = field(default_factory=tuple)
    
    def __str__(self) -> str:
        base = super().__str__()
        if self.tensor_name:
            base += f" [tensor={self.tensor_name!r}]"
        if self.shapes:
            base += f" (shapes: {self.shapes})"
        return base


@dataclass
class ShapeError(DataLoadingError):
    """Tensor shape mismatch."""
    expected_shape: Optional[tuple] = None
    actual_shape: Optional[tuple] = None
    
    def __str__(self) -> str:
        base = super().__str__()
        if self.expected_shape and self.actual_shape:
            base += f" (expected {self.expected_shape}, got {self.actual_shape})"
        return base
