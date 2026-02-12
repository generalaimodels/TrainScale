# ════════════════════════════════════════════════════════════════════════════════
# Column Mapper - Fuzzy Matching for Schema Alignment
# ════════════════════════════════════════════════════════════════════════════════
# Maps source dataset columns to target schema using fuzzy matching.
# Uses regex patterns and Levenshtein distance for intelligent mapping.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import (
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

from data_pipeline.core.types import Result, Ok, Err
from data_pipeline.core.errors import ColumnNotFoundError


# ─────────────────────────────────────────────────────────────────────────────────
# Constants - Common Column Name Patterns
# ─────────────────────────────────────────────────────────────────────────────────

# Patterns for instruction-following datasets
INSTRUCTION_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^instruction$", re.I),
    re.compile(r"^prompt$", re.I),
    re.compile(r"^query$", re.I),
    re.compile(r"^question$", re.I),
    re.compile(r"^user_input$", re.I),
    re.compile(r"^user_message$", re.I),
    re.compile(r"^human$", re.I),
    re.compile(r"^input_text$", re.I),
)

INPUT_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^input$", re.I),
    re.compile(r"^context$", re.I),
    re.compile(r"^passage$", re.I),
    re.compile(r"^document$", re.I),
    re.compile(r"^text$", re.I),
    re.compile(r"^source$", re.I),
)

OUTPUT_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^output$", re.I),
    re.compile(r"^response$", re.I),
    re.compile(r"^answer$", re.I),
    re.compile(r"^completion$", re.I),
    re.compile(r"^target$", re.I),
    re.compile(r"^assistant$", re.I),
    re.compile(r"^gpt$", re.I),
    re.compile(r"^model_output$", re.I),
    re.compile(r"^generated$", re.I),
)

LABEL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^label$", re.I),
    re.compile(r"^labels$", re.I),
    re.compile(r"^class$", re.I),
    re.compile(r"^category$", re.I),
    re.compile(r"^target$", re.I),
)

# Standard schema mapping (target -> patterns)
STANDARD_SCHEMA: Dict[str, Tuple[re.Pattern, ...]] = {
    "instruction": INSTRUCTION_PATTERNS,
    "input": INPUT_PATTERNS,
    "output": OUTPUT_PATTERNS,
    "label": LABEL_PATTERNS,
}


# ─────────────────────────────────────────────────────────────────────────────────
# Levenshtein Distance (Optimized)
# ─────────────────────────────────────────────────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein distance between two strings.
    
    Uses Wagner-Fischer algorithm with O(min(m,n)) space optimization.
    
    Time Complexity: O(m * n)
    Space Complexity: O(min(m, n))
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance between strings
    """
    # Ensure s1 is the shorter string for space optimization
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    
    m, n = len(s1), len(s2)
    
    # Early termination for empty strings
    if m == 0:
        return n
    if n == 0:
        return m
    
    # Use two rows for space optimization
    prev_row = list(range(m + 1))
    curr_row = [0] * (m + 1)
    
    for j in range(1, n + 1):
        curr_row[0] = j
        for i in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr_row[i] = min(
                prev_row[i] + 1,      # deletion
                curr_row[i - 1] + 1,  # insertion
                prev_row[i - 1] + cost  # substitution
            )
        prev_row, curr_row = curr_row, prev_row
    
    return prev_row[m]


def normalized_similarity(s1: str, s2: str) -> float:
    """
    Compute normalized similarity between strings.
    
    Returns value in [0, 1] where 1 means identical.
    
    Time Complexity: O(m * n)
    Space Complexity: O(min(m, n))
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity score in [0, 1]
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    distance = levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len)


# ─────────────────────────────────────────────────────────────────────────────────
# Column Mapping Result
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ColumnMatch:
    """
    Result of a column matching operation.
    
    Attributes:
        source: Source column name from dataset
        target: Target schema column name
        confidence: Match confidence score [0, 1]
        match_type: How the match was determined
    """
    source: str
    target: str
    confidence: float
    match_type: str  # "exact", "pattern", "fuzzy"


# ─────────────────────────────────────────────────────────────────────────────────
# Column Mapper Class
# ─────────────────────────────────────────────────────────────────────────────────

class ColumnMapper:
    """
    Maps source columns to target schema using multiple strategies.
    
    Strategy priority:
    1. Exact match (case-insensitive)
    2. Regex pattern match
    3. Fuzzy match (Levenshtein similarity)
    
    Time Complexity: O(source_cols * target_cols * max_col_length)
    Space Complexity: O(source_cols + target_cols)
    """
    
    def __init__(
        self,
        custom_patterns: Optional[Dict[str, Tuple[re.Pattern, ...]]] = None,
        min_fuzzy_score: float = 0.7,
    ):
        """
        Initialize column mapper.
        
        Args:
            custom_patterns: Additional pattern definitions
            min_fuzzy_score: Minimum similarity for fuzzy matching
        """
        self._patterns = dict(STANDARD_SCHEMA)
        if custom_patterns:
            self._patterns.update(custom_patterns)
        self._min_fuzzy_score = min_fuzzy_score
    
    def map_columns(
        self,
        source_columns: List[str],
        target_schema: List[str],
        explicit_mapping: Optional[Dict[str, str]] = None,
    ) -> Result[Dict[str, str], ColumnNotFoundError]:
        """
        Map source columns to target schema.
        
        Args:
            source_columns: Available columns in source dataset
            target_schema: Required target column names
            explicit_mapping: User-provided explicit mappings (highest priority)
            
        Returns:
            Result containing mapping dict or error
        """
        mapping: Dict[str, str] = {}
        unmapped_targets: Set[str] = set(target_schema)
        available_sources: Set[str] = set(source_columns)
        
        # Apply explicit mappings first
        if explicit_mapping:
            for target, source in explicit_mapping.items():
                if source in available_sources and target in unmapped_targets:
                    mapping[target] = source
                    unmapped_targets.discard(target)
                    available_sources.discard(source)
        
        # Map remaining using auto-detection
        for target in list(unmapped_targets):
            match = self._find_best_match(target, list(available_sources))
            if match:
                mapping[target] = match.source
                unmapped_targets.discard(target)
                available_sources.discard(match.source)
        
        # Check for missing required columns
        if unmapped_targets:
            return Err(ColumnNotFoundError(
                message=f"Could not map columns: {unmapped_targets}",
                available_columns=tuple(source_columns),
            ))
        
        return Ok(mapping)
    
    def _find_best_match(
        self,
        target: str,
        sources: List[str],
    ) -> Optional[ColumnMatch]:
        """Find best matching source column for target."""
        if not sources:
            return None
        
        # Strategy 1: Exact match (case-insensitive)
        target_lower = target.lower()
        for source in sources:
            if source.lower() == target_lower:
                return ColumnMatch(
                    source=source,
                    target=target,
                    confidence=1.0,
                    match_type="exact",
                )
        
        # Strategy 2: Pattern match
        patterns = self._patterns.get(target.lower(), ())
        for source in sources:
            for pattern in patterns:
                if pattern.match(source):
                    return ColumnMatch(
                        source=source,
                        target=target,
                        confidence=0.95,
                        match_type="pattern",
                    )
        
        # Strategy 3: Fuzzy match
        best_match: Optional[ColumnMatch] = None
        best_score = self._min_fuzzy_score
        
        for source in sources:
            score = normalized_similarity(source, target)
            if score > best_score:
                best_score = score
                best_match = ColumnMatch(
                    source=source,
                    target=target,
                    confidence=score,
                    match_type="fuzzy",
                )
        
        return best_match
    
    def suggest_mappings(
        self,
        source_columns: List[str],
        target_schema: Optional[List[str]] = None,
    ) -> List[ColumnMatch]:
        """
        Suggest column mappings without requiring targets.
        
        Useful for exploring unknown datasets.
        
        Args:
            source_columns: Available source columns
            target_schema: Optional target schema (uses standard if None)
            
        Returns:
            List of suggested column matches
        """
        targets = target_schema or list(STANDARD_SCHEMA.keys())
        suggestions = []
        
        for target in targets:
            match = self._find_best_match(target, source_columns)
            if match:
                suggestions.append(match)
        
        return suggestions


# ─────────────────────────────────────────────────────────────────────────────────
# Module-level Functions
# ─────────────────────────────────────────────────────────────────────────────────

def fuzzy_match_columns(
    source_columns: List[str],
    target_schema: List[str],
    explicit_mapping: Optional[Dict[str, str]] = None,
    min_score: float = 0.7,
) -> Result[Dict[str, str], ColumnNotFoundError]:
    """
    Convenience function for column mapping.
    
    Args:
        source_columns: Available columns in source
        target_schema: Required target columns
        explicit_mapping: User-provided mappings
        min_score: Minimum similarity for fuzzy matching
        
    Returns:
        Result containing mapping or error
    """
    mapper = ColumnMapper(min_fuzzy_score=min_score)
    return mapper.map_columns(source_columns, target_schema, explicit_mapping)
