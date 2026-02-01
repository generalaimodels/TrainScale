# ════════════════════════════════════════════════════════════════════════════════
# Length Manager - SOTA Variable-Length Column Handling
# ════════════════════════════════════════════════════════════════════════════════
# Intelligent handling of columns with different text sizes.
# Provides: per-column limits, content-aware truncation, dynamic padding strategies.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)


# ─────────────────────────────────────────────────────────────────────────────────
# Configuration Types
# ─────────────────────────────────────────────────────────────────────────────────

class TruncationStrategy(Enum):
    """How to truncate text when exceeding limits."""
    NONE = auto()           # No truncation, may cause errors
    SIMPLE = auto()         # Hard cut at character/token limit
    WORD_BOUNDARY = auto()  # Truncate at word boundary
    SENTENCE_BOUNDARY = auto()  # Truncate at sentence boundary
    SMART = auto()          # Adaptive: prefer sentence > word > simple


class PaddingStrategy(Enum):
    """How to pad sequences in a batch."""
    DO_NOT_PAD = auto()     # No padding, variable lengths (for packing)
    LONGEST = auto()        # Pad to longest in batch (dynamic)
    MAX_LENGTH = auto()     # Pad to configured max_length (fixed)
    BUCKET = auto()         # Bucket by length, pad to bucket max


class ContentDistributionMode(Enum):
    """How to distribute token budget across columns."""
    EQUAL = auto()          # Split budget equally
    PROPORTIONAL = auto()   # Split based on original content lengths
    RATIO = auto()          # Split based on user-defined ratios
    PRIORITY = auto()       # Fill high-priority first, then remaining
    ADAPTIVE = auto()       # Smart: use content importance heuristics


@dataclass(frozen=True, slots=True)
class ColumnContentConfig:
    """
    Per-column content configuration for SOTA control.
    
    Attributes:
        token_ratio: Ratio of total token budget (0.0-1.0)
        min_tokens: Minimum tokens to preserve
        max_tokens: Maximum tokens for this column
        priority: Processing priority (lower = higher priority)
        truncation: Truncation strategy for this column
        is_input: True if input column, False if output/label
    """
    token_ratio: Optional[float] = None  # Ratio of total budget
    min_tokens: int = 0                  # Minimum guaranteed tokens
    max_tokens: Optional[int] = None     # Hard maximum tokens
    priority: int = 0                    # Lower = more important
    truncation: TruncationStrategy = TruncationStrategy.SMART
    is_input: bool = True                # Input vs output/label


@dataclass(frozen=True, slots=True)
class ColumnLengthConfig:
    """
    Per-column length configuration (legacy, for simple limits).
    
    Attributes:
        max_chars: Maximum characters (before tokenization)
        max_tokens: Maximum tokens (after tokenization, if specified)
        truncation: How to truncate if exceeded
        priority: Lower = higher priority when content must be cut
        preserve_ratio: Minimum ratio of content to preserve (0.0-1.0)
    """
    max_chars: Optional[int] = None
    max_tokens: Optional[int] = None
    truncation: TruncationStrategy = TruncationStrategy.SMART
    priority: int = 0
    preserve_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class LengthManagerConfig:
    """
    Global length management configuration.
    
    Attributes:
        total_max_length: Maximum total sequence length in tokens
        padding_strategy: How to pad batches
        distribution_mode: How to distribute tokens across columns
        column_ratios: Ratio-based token allocation
        column_configs: Detailed per-column configurations
        bucket_boundaries: Length boundaries for bucket padding
        special_tokens_budget: Reserved tokens for BOS/EOS/etc
        default_truncation: Default truncation strategy
    """
    total_max_length: int = 2048
    padding_strategy: PaddingStrategy = PaddingStrategy.LONGEST
    distribution_mode: ContentDistributionMode = ContentDistributionMode.PROPORTIONAL
    column_ratios: Dict[str, float] = field(default_factory=dict)  # e.g., {"instruction": 0.3, "output": 0.6}
    column_configs: Dict[str, ColumnContentConfig] = field(default_factory=dict)
    bucket_boundaries: Tuple[int, ...] = (128, 256, 512, 1024, 2048)
    special_tokens_budget: int = 10  # Reserve for BOS/EOS/special
    default_truncation: TruncationStrategy = TruncationStrategy.SMART
    # Legacy support
    per_column: Dict[str, ColumnLengthConfig] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────────
# Text Boundary Detection
# ─────────────────────────────────────────────────────────────────────────────────

# Sentence boundary pattern (handles common cases)
SENTENCE_END_PATTERN = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])|'  # Period/exclamation/question followed by capital
    r'(?<=[.!?])$|'             # End of string after sentence marker
    r'(?<=\n)\s*(?=\n)'         # Double newline (paragraph)
)

# Word boundary pattern
WORD_BOUNDARY_PATTERN = re.compile(r'\s+')


def find_sentence_boundary(text: str, max_chars: int) -> int:
    """
    Find the last sentence boundary before max_chars.
    
    Returns position to truncate at, or max_chars if no boundary found.
    
    Time Complexity: O(n) where n = max_chars
    Space Complexity: O(1)
    """
    if len(text) <= max_chars:
        return len(text)
    
    # Search in the region before max_chars
    search_region = text[:max_chars]
    
    # Find all sentence boundaries
    last_boundary = 0
    for match in SENTENCE_END_PATTERN.finditer(search_region):
        boundary_pos = match.end()
        if boundary_pos <= max_chars:
            last_boundary = boundary_pos
    
    # If no sentence boundary found, use max_chars
    if last_boundary == 0:
        return max_chars
    
    return last_boundary


def find_word_boundary(text: str, max_chars: int) -> int:
    """
    Find the last word boundary before max_chars.
    
    Time Complexity: O(n) where n = max_chars
    Space Complexity: O(1)
    """
    if len(text) <= max_chars:
        return len(text)
    
    # Search backwards from max_chars for whitespace
    search_start = max(0, max_chars - 50)  # Look back up to 50 chars
    search_region = text[search_start:max_chars]
    
    # Find last whitespace
    last_space = search_region.rfind(' ')
    if last_space == -1:
        last_space = search_region.rfind('\n')
    if last_space == -1:
        last_space = search_region.rfind('\t')
    
    if last_space != -1:
        return search_start + last_space
    
    return max_chars


def smart_truncate(
    text: str, 
    max_chars: int,
    strategy: TruncationStrategy = TruncationStrategy.SMART,
) -> str:
    """
    Truncate text using specified strategy.
    
    Args:
        text: Input text
        max_chars: Maximum characters
        strategy: Truncation strategy
        
    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text
    
    if strategy == TruncationStrategy.NONE:
        return text  # No truncation
    
    if strategy == TruncationStrategy.SIMPLE:
        return text[:max_chars]
    
    if strategy == TruncationStrategy.WORD_BOUNDARY:
        boundary = find_word_boundary(text, max_chars)
        return text[:boundary].rstrip()
    
    if strategy == TruncationStrategy.SENTENCE_BOUNDARY:
        boundary = find_sentence_boundary(text, max_chars)
        return text[:boundary].rstrip()
    
    # SMART: Try sentence first, then word, then simple
    sentence_boundary = find_sentence_boundary(text, max_chars)
    if sentence_boundary >= max_chars * 0.5:  # At least 50% content
        return text[:sentence_boundary].rstrip()
    
    word_boundary = find_word_boundary(text, max_chars)
    if word_boundary >= max_chars * 0.75:  # At least 75% content
        return text[:word_boundary].rstrip()
    
    return text[:max_chars]


# ─────────────────────────────────────────────────────────────────────────────────
# Token-Aware Content Distributor
# ─────────────────────────────────────────────────────────────────────────────────

class TokenAwareContentDistributor:
    """
    SOTA content distribution across variable-length columns.
    
    Instead of simple per-column limits, this distributes the total
    token budget intelligently based on:
    - User-defined ratios (e.g., instruction: 30%, output: 60%)
    - Proportional allocation based on original lengths
    - Priority-based allocation (high priority columns filled first)
    - Adaptive allocation with content importance heuristics
    
    Thread-safe: stateless after initialization.
    
    Time Complexity: O(n) for n columns
    Space Complexity: O(n)
    """
    
    def __init__(
        self,
        total_max_tokens: int = 2048,
        distribution_mode: ContentDistributionMode = ContentDistributionMode.PROPORTIONAL,
        column_ratios: Optional[Dict[str, float]] = None,
        column_configs: Optional[Dict[str, ColumnContentConfig]] = None,
        special_tokens_budget: int = 10,
        default_truncation: TruncationStrategy = TruncationStrategy.SMART,
        tokenizer: Optional[Any] = None,  # For accurate token counting
    ):
        """
        Initialize content distributor.
        
        Args:
            total_max_tokens: Total token budget for the sequence
            distribution_mode: How to allocate tokens across columns
            column_ratios: User-defined ratios {column: ratio}
            column_configs: Detailed column configurations
            special_tokens_budget: Reserved for BOS/EOS/special tokens
            default_truncation: Default truncation strategy
            tokenizer: Optional tokenizer for accurate counting
        """
        self._total_max = total_max_tokens
        self._mode = distribution_mode
        self._column_ratios = column_ratios or {}
        self._column_configs = column_configs or {}
        self._special_budget = special_tokens_budget
        self._default_truncation = default_truncation
        self._tokenizer = tokenizer
        
        # Effective budget (total - special tokens)
        self._available_budget = max(1, total_max_tokens - special_tokens_budget)
    
    @property
    def available_tokens(self) -> int:
        """Get available token budget (excluding special tokens)."""
        return self._available_budget
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Uses tokenizer if available, otherwise heuristic (~4 chars/token).
        """
        if not text:
            return 0
        
        if self._tokenizer is not None:
            try:
                if hasattr(self._tokenizer, 'encode'):
                    return len(self._tokenizer.encode(text, add_special_tokens=False))
                elif hasattr(self._tokenizer, '__call__'):
                    return len(self._tokenizer(text, add_special_tokens=False)['input_ids'])
            except Exception:
                pass
        
        # Fallback: heuristic estimation
        # Average ~4 characters per token for English
        return max(1, len(text) // 4)
    
    def _tokens_to_chars(self, tokens: int) -> int:
        """Convert token count to approximate character count."""
        return tokens * 4  # ~4 chars/token average
    
    def distribute(
        self,
        example: Dict[str, Any],
        text_columns: List[str],
        label_column: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Distribute token budget across columns and truncate accordingly.
        
        This is the main SOTA method for intelligent content control.
        
        Args:
            example: Input example with text columns
            text_columns: List of text column names
            label_column: Optional label column (gets priority)
            
        Returns:
            Modified example with truncated text
        """
        result = dict(example)
        
        # Get current content lengths
        column_texts = {}
        column_tokens = {}
        
        for col in text_columns:
            text = str(example.get(col, "")) if example.get(col) else ""
            column_texts[col] = text
            column_tokens[col] = self._estimate_tokens(text)
        
        # Allocate token budget based on mode
        allocations = self._allocate_tokens(column_tokens, label_column)
        
        # Apply truncation based on allocations
        for col, max_tokens in allocations.items():
            if col in column_texts:
                text = column_texts[col]
                current_tokens = column_tokens[col]
                
                if current_tokens > max_tokens:
                    # Need to truncate
                    max_chars = self._tokens_to_chars(max_tokens)
                    config = self._column_configs.get(col, ColumnContentConfig())
                    truncation = config.truncation if config else self._default_truncation
                    
                    result[col] = smart_truncate(text, max_chars, truncation)
                else:
                    result[col] = text
        
        return result
    
    def _allocate_tokens(
        self,
        column_tokens: Dict[str, int],
        label_column: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Allocate token budget to each column based on distribution mode.
        
        Returns dict of {column: max_tokens}.
        """
        total_current = sum(column_tokens.values())
        columns = list(column_tokens.keys())
        
        if self._mode == ContentDistributionMode.EQUAL:
            return self._allocate_equal(columns)
        
        elif self._mode == ContentDistributionMode.RATIO:
            return self._allocate_ratio(columns, column_tokens)
        
        elif self._mode == ContentDistributionMode.PROPORTIONAL:
            return self._allocate_proportional(column_tokens, total_current)
        
        elif self._mode == ContentDistributionMode.PRIORITY:
            return self._allocate_priority(column_tokens, label_column)
        
        else:  # ADAPTIVE
            return self._allocate_adaptive(column_tokens, label_column)
    
    def _allocate_equal(self, columns: List[str]) -> Dict[str, int]:
        """Equal split of budget across all columns."""
        n = len(columns)
        if n == 0:
            return {}
        
        per_column = self._available_budget // n
        return {col: per_column for col in columns}
    
    def _allocate_ratio(
        self,
        columns: List[str],
        column_tokens: Dict[str, int],
    ) -> Dict[str, int]:
        """Allocate based on user-defined ratios."""
        allocations = {}
        remaining = self._available_budget
        
        # First pass: allocate based on ratios
        for col in columns:
            ratio = self._column_ratios.get(col, 0.0)
            config = self._column_configs.get(col)
            
            if ratio > 0:
                allocation = int(self._available_budget * ratio)
            elif config and config.token_ratio:
                allocation = int(self._available_budget * config.token_ratio)
            else:
                allocation = 0  # Will be filled from remaining
            
            # Apply min/max constraints
            if config:
                if config.min_tokens:
                    allocation = max(allocation, config.min_tokens)
                if config.max_tokens:
                    allocation = min(allocation, config.max_tokens)
            
            allocations[col] = allocation
            remaining -= allocation
        
        # Second pass: distribute remaining to unallocated columns
        unallocated = [col for col in columns if allocations.get(col, 0) == 0]
        if unallocated and remaining > 0:
            per_column = remaining // len(unallocated)
            for col in unallocated:
                allocations[col] = per_column
        
        return allocations
    
    def _allocate_proportional(
        self,
        column_tokens: Dict[str, int],
        total_current: int,
    ) -> Dict[str, int]:
        """Allocate proportionally based on original content lengths."""
        if total_current == 0:
            return self._allocate_equal(list(column_tokens.keys()))
        
        # If total fits, no allocation needed
        if total_current <= self._available_budget:
            return {col: tokens for col, tokens in column_tokens.items()}
        
        # Scale down proportionally
        scale = self._available_budget / total_current
        
        allocations = {}
        for col, tokens in column_tokens.items():
            config = self._column_configs.get(col)
            allocation = int(tokens * scale)
            
            # Apply min/max constraints
            if config:
                if config.min_tokens:
                    allocation = max(allocation, config.min_tokens)
                if config.max_tokens:
                    allocation = min(allocation, config.max_tokens)
            
            allocations[col] = max(1, allocation)  # At least 1 token
        
        return allocations
    
    def _allocate_priority(
        self,
        column_tokens: Dict[str, int],
        label_column: Optional[str] = None,
    ) -> Dict[str, int]:
        """Allocate by priority: fill high-priority columns first."""
        allocations = {}
        remaining = self._available_budget
        
        # Sort by priority (lower = higher priority)
        columns_by_priority = sorted(
            column_tokens.keys(),
            key=lambda c: (
                0 if c == label_column else  # Label gets highest priority
                self._column_configs.get(c, ColumnContentConfig()).priority
            )
        )
        
        for col in columns_by_priority:
            tokens_needed = column_tokens[col]
            config = self._column_configs.get(col)
            
            # Allocate what we can
            allocation = min(tokens_needed, remaining)
            
            # Apply min constraint
            if config and config.min_tokens:
                allocation = max(allocation, min(config.min_tokens, remaining))
            
            # Apply max constraint
            if config and config.max_tokens:
                allocation = min(allocation, config.max_tokens)
            
            allocations[col] = allocation
            remaining -= allocation
            
            if remaining <= 0:
                break
        
        # Fill any remaining columns with 0
        for col in column_tokens:
            if col not in allocations:
                allocations[col] = 0
        
        return allocations
    
    def _allocate_adaptive(
        self,
        column_tokens: Dict[str, int],
        label_column: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Adaptive allocation with content importance heuristics.
        
        Prioritizes:
        1. Label/output column (needs full content for loss)
        2. Shorter columns (likely more dense/important)
        3. Input columns proportionally
        """
        total_current = sum(column_tokens.values())
        
        # If fits, no truncation needed
        if total_current <= self._available_budget:
            return {col: tokens for col, tokens in column_tokens.items()}
        
        allocations = {}
        remaining = self._available_budget
        
        # Step 1: Allocate label column (60% budget or actual, whichever is smaller)
        if label_column and label_column in column_tokens:
            label_tokens = column_tokens[label_column]
            label_max = int(self._available_budget * 0.6)
            allocations[label_column] = min(label_tokens, label_max)
            remaining -= allocations[label_column]
        
        # Step 2: Sort remaining by length (shorter = likely more important/dense)
        other_columns = [c for c in column_tokens if c != label_column]
        other_columns.sort(key=lambda c: column_tokens[c])
        
        # Step 3: Allocate remaining budget
        if other_columns:
            # Distribute remaining proportionally among input columns
            other_total = sum(column_tokens[c] for c in other_columns)
            
            for col in other_columns:
                if other_total > 0:
                    ratio = column_tokens[col] / other_total
                    allocation = int(remaining * ratio)
                else:
                    allocation = remaining // len(other_columns)
                
                allocations[col] = min(allocation, column_tokens[col])
        
        return allocations


# ─────────────────────────────────────────────────────────────────────────────────
# Length Manager Class
# ─────────────────────────────────────────────────────────────────────────────────

class LengthManager:
    """
    Manages variable-length columns with SOTA preprocessing.
    
    Provides:
    1. Token-aware content distribution across columns
    2. Content-aware truncation (sentence/word boundaries)
    3. Priority-based or ratio-based allocation
    4. Dynamic bucket-based padding
    
    Thread-safe: stateless after initialization.
    
    Time Complexity: O(total_text_length) per example
    Space Complexity: O(num_columns)
    """
    
    def __init__(self, config: LengthManagerConfig):
        """
        Initialize length manager.
        
        Args:
            config: Length management configuration
        """
        self._config = config
        self._column_configs = dict(config.per_column)
    
    @property
    def max_length(self) -> int:
        """Get total max sequence length."""
        return self._config.total_max_length
    
    @property
    def padding_strategy(self) -> PaddingStrategy:
        """Get padding strategy."""
        return self._config.padding_strategy
    
    def get_column_config(self, column: str) -> ColumnLengthConfig:
        """Get config for a specific column."""
        return self._column_configs.get(column, ColumnLengthConfig())
    
    def set_column_config(self, column: str, config: ColumnLengthConfig) -> None:
        """Set config for a specific column."""
        self._column_configs[column] = config
    
    def preprocess_text(
        self,
        text: str,
        column: Optional[str] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        """
        Preprocess a single text field.
        
        Args:
            text: Input text
            column: Column name (for per-column config)
            max_chars: Override max chars
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Get column-specific config
        col_config = self.get_column_config(column) if column else ColumnLengthConfig()
        
        # Determine max chars
        effective_max = max_chars or col_config.max_chars
        if effective_max is None:
            # Default: estimate based on total_max_length
            # Assume ~4 chars per token on average
            effective_max = self._config.total_max_length * 4
        
        # Apply truncation
        strategy = col_config.truncation or self._config.default_truncation
        return smart_truncate(text, effective_max, strategy)
    
    def preprocess_example(
        self,
        example: Dict[str, Any],
        text_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Preprocess all text columns in an example.
        
        Applies per-column limits and handles priority-based
        trimming when total content is too large.
        
        Args:
            example: Input example dict
            text_columns: Columns to process (all string columns if None)
            
        Returns:
            Preprocessed example
        """
        result = dict(example)
        
        # Identify text columns
        if text_columns is None:
            text_columns = [
                k for k, v in example.items() 
                if isinstance(v, str)
            ]
        
        # Phase 1: Apply per-column limits
        for col in text_columns:
            if col in result and isinstance(result[col], str):
                result[col] = self.preprocess_text(result[col], column=col)
        
        # Phase 2: Priority-based trimming if still too long
        # Estimate total length (rough approximation)
        total_chars = sum(
            len(str(result.get(col, ""))) 
            for col in text_columns
        )
        max_total_chars = self._config.total_max_length * 4  # ~4 chars/token
        
        if total_chars > max_total_chars:
            result = self._priority_trim(result, text_columns, max_total_chars)
        
        return result
    
    def _priority_trim(
        self,
        example: Dict[str, Any],
        text_columns: List[str],
        max_total_chars: int,
    ) -> Dict[str, Any]:
        """
        Trim lower-priority columns when total is too long.
        
        Columns with higher priority (lower number) are preserved.
        """
        result = dict(example)
        
        # Get columns with their priorities
        column_priorities = []
        for col in text_columns:
            if col in result and isinstance(result[col], str):
                config = self.get_column_config(col)
                column_priorities.append((col, config.priority, len(result[col])))
        
        # Sort by priority (descending = trim lowest priority first)
        column_priorities.sort(key=lambda x: -x[1])
        
        # Calculate current total
        current_total = sum(length for _, _, length in column_priorities)
        
        # Trim from lowest priority until within limit
        for col, priority, length in column_priorities:
            if current_total <= max_total_chars:
                break
            
            config = self.get_column_config(col)
            min_length = int(length * config.preserve_ratio)
            
            # Calculate how much to trim
            excess = current_total - max_total_chars
            trim_amount = min(excess, length - min_length)
            
            if trim_amount > 0:
                new_length = length - trim_amount
                result[col] = smart_truncate(
                    result[col], 
                    new_length,
                    config.truncation,
                )
                current_total -= trim_amount
        
        return result
    
    def get_bucket(self, length: int) -> int:
        """
        Get bucket size for a sequence length.
        
        Used for bucket padding strategy.
        
        Args:
            length: Sequence length
            
        Returns:
            Bucket size to pad to
        """
        for boundary in self._config.bucket_boundaries:
            if length <= boundary:
                return boundary
        return self._config.total_max_length
    
    def get_effective_max_length(
        self,
        batch_lengths: Optional[List[int]] = None,
    ) -> int:
        """
        Get effective max length based on padding strategy.
        
        Args:
            batch_lengths: Lengths of sequences in batch (for dynamic strategies)
            
        Returns:
            Effective max length for padding
        """
        strategy = self._config.padding_strategy
        
        if strategy == PaddingStrategy.MAX_LENGTH:
            return self._config.total_max_length
        
        if strategy == PaddingStrategy.DO_NOT_PAD:
            return 0  # No padding
        
        if batch_lengths is None:
            return self._config.total_max_length
        
        if strategy == PaddingStrategy.LONGEST:
            return max(batch_lengths)
        
        if strategy == PaddingStrategy.BUCKET:
            max_len = max(batch_lengths)
            return self.get_bucket(max_len)
        
        return self._config.total_max_length


# ─────────────────────────────────────────────────────────────────────────────────
# Dynamic Collate with Length Management
# ─────────────────────────────────────────────────────────────────────────────────

def create_dynamic_collate_fn(
    length_manager: LengthManager,
    pad_token_id: int = 0,
    label_pad_token_id: int = -100,
    padding_side: Literal["left", "right"] = "right",
) -> Callable:
    """
    Create collate function with dynamic length handling.
    
    Uses LengthManager's padding strategy to determine batch padding.
    
    Args:
        length_manager: Configured LengthManager
        pad_token_id: Token ID for padding
        label_pad_token_id: Label padding value
        padding_side: Which side to pad
        
    Returns:
        Collate function for DataLoader
    """
    import torch
    from torch import Tensor
    
    def pad_right(sequences: List[Tensor], pad_value: int, max_len: int) -> Tensor:
        batch_size = len(sequences)
        output = torch.full((batch_size, max_len), pad_value, dtype=sequences[0].dtype)
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_len)
            output[i, :length] = seq[:length]
        return output
    
    def pad_left(sequences: List[Tensor], pad_value: int, max_len: int) -> Tensor:
        batch_size = len(sequences)
        output = torch.full((batch_size, max_len), pad_value, dtype=sequences[0].dtype)
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_len)
            output[i, -length:] = seq[-length:]
        return output
    
    pad_fn = pad_left if padding_side == "left" else pad_right
    
    def collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        if not batch:
            return {}
        
        # Get batch lengths
        batch_lengths = [len(ex["input_ids"]) for ex in batch]
        
        # Determine effective max length using manager
        effective_max = length_manager.get_effective_max_length(batch_lengths)
        
        # Handle DO_NOT_PAD - use longest in batch
        if effective_max == 0:
            effective_max = max(batch_lengths)
        
        # Pad sequences
        input_ids = pad_fn(
            [ex["input_ids"] for ex in batch], 
            pad_token_id, 
            effective_max
        )
        attention_mask = pad_fn(
            [ex["attention_mask"] for ex in batch], 
            0, 
            effective_max
        )
        labels = pad_fn(
            [ex["labels"] for ex in batch], 
            label_pad_token_id, 
            effective_max
        )
        
        # Ensure labels are masked where attention is 0
        labels = labels.masked_fill(attention_mask == 0, label_pad_token_id)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    return collate_fn


# ─────────────────────────────────────────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────────────────────────────────────────

def create_length_manager(
    max_length: int = 2048,
    padding_strategy: Union[str, PaddingStrategy] = PaddingStrategy.LONGEST,
    default_truncation: Union[str, TruncationStrategy] = TruncationStrategy.SMART,
    per_column_limits: Optional[Dict[str, int]] = None,
    bucket_boundaries: Optional[Tuple[int, ...]] = None,
) -> LengthManager:
    """
    Create length manager with common configuration.
    
    Args:
        max_length: Maximum total sequence length
        padding_strategy: Padding strategy (or string name)
        default_truncation: Default truncation strategy (or string name)
        per_column_limits: Optional per-column character limits
        bucket_boundaries: Optional bucket boundaries for bucket padding
        
    Returns:
        Configured LengthManager
    """
    # Handle string inputs
    if isinstance(padding_strategy, str):
        padding_strategy = PaddingStrategy[padding_strategy.upper()]
    if isinstance(default_truncation, str):
        default_truncation = TruncationStrategy[default_truncation.upper()]
    
    # Build per-column configs
    per_column = {}
    if per_column_limits:
        for col, limit in per_column_limits.items():
            per_column[col] = ColumnLengthConfig(max_chars=limit)
    
    config = LengthManagerConfig(
        total_max_length=max_length,
        padding_strategy=padding_strategy,
        default_truncation=default_truncation,
        per_column=per_column,
        bucket_boundaries=bucket_boundaries or (128, 256, 512, 1024, 2048),
    )
    
    return LengthManager(config)
