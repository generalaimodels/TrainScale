# ════════════════════════════════════════════════════════════════════════════════
# Result Types - Rust-inspired Error Handling
# ════════════════════════════════════════════════════════════════════════════════
# Deterministic error handling without exceptions for control flow.
# Enforces exhaustive pattern matching via type system.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, Generic, Union, Callable, NoReturn

# ─────────────────────────────────────────────────────────────────────────────────
# Type Variables
# ─────────────────────────────────────────────────────────────────────────────────

T = TypeVar("T")  # Success type
E = TypeVar("E")  # Error type
U = TypeVar("U")  # Mapped success type


# ─────────────────────────────────────────────────────────────────────────────────
# Result Variants
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """
    Success variant of Result.
    
    Immutable, cache-line optimized via slots.
    Represents successful computation with value of type T.
    """
    value: T
    
    def __repr__(self) -> str:
        return f"Ok({self.value!r})"


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """
    Error variant of Result.
    
    Immutable, cache-line optimized via slots.
    Represents failed computation with error of type E.
    """
    error: E
    
    def __repr__(self) -> str:
        return f"Err({self.error!r})"


# ─────────────────────────────────────────────────────────────────────────────────
# Result Type Alias
# ─────────────────────────────────────────────────────────────────────────────────

Result = Union[Ok[T], Err[E]]


# ─────────────────────────────────────────────────────────────────────────────────
# Pattern Matching Utilities
# ─────────────────────────────────────────────────────────────────────────────────

def is_ok(result: Result[T, E]) -> bool:
    """
    Check if Result is Ok variant.
    
    Time Complexity: O(1)
    Space Complexity: O(1)
    
    Args:
        result: Result to check
        
    Returns:
        True if Ok, False if Err
    """
    return isinstance(result, Ok)


def is_err(result: Result[T, E]) -> bool:
    """
    Check if Result is Err variant.
    
    Time Complexity: O(1)
    Space Complexity: O(1)
    
    Args:
        result: Result to check
        
    Returns:
        True if Err, False if Ok
    """
    return isinstance(result, Err)


def unwrap(result: Result[T, E]) -> T:
    """
    Extract value from Ok, raise if Err.
    
    Use only when Ok is guaranteed (e.g., after is_ok check).
    Prefer pattern matching for production code.
    
    Time Complexity: O(1)
    Space Complexity: O(1)
    
    Args:
        result: Result to unwrap
        
    Returns:
        Value from Ok variant
        
    Raises:
        ValueError: If result is Err
    """
    if isinstance(result, Ok):
        return result.value
    raise ValueError(f"Called unwrap on Err: {result.error}")


def unwrap_err(result: Result[T, E]) -> E:
    """
    Extract error from Err, raise if Ok.
    
    Time Complexity: O(1)
    Space Complexity: O(1)
    
    Args:
        result: Result to unwrap
        
    Returns:
        Error from Err variant
        
    Raises:
        ValueError: If result is Ok
    """
    if isinstance(result, Err):
        return result.error
    raise ValueError(f"Called unwrap_err on Ok: {result.value}")


def unwrap_or(result: Result[T, E], default: T) -> T:
    """
    Extract value from Ok, or return default if Err.
    
    Time Complexity: O(1)
    Space Complexity: O(1)
    
    Args:
        result: Result to unwrap
        default: Default value if Err
        
    Returns:
        Value from Ok or default
    """
    return result.value if isinstance(result, Ok) else default


def unwrap_or_else(result: Result[T, E], f: Callable[[E], T]) -> T:
    """
    Extract value from Ok, or compute from error if Err.
    
    Lazy evaluation: f only called on Err.
    
    Time Complexity: O(1) + O(f)
    Space Complexity: O(1) + O(f)
    
    Args:
        result: Result to unwrap
        f: Function to compute default from error
        
    Returns:
        Value from Ok or f(error)
    """
    return result.value if isinstance(result, Ok) else f(result.error)


# ─────────────────────────────────────────────────────────────────────────────────
# Functor/Monad Operations
# ─────────────────────────────────────────────────────────────────────────────────

def map_result(result: Result[T, E], f: Callable[[T], U]) -> Result[U, E]:
    """
    Apply function to Ok value, pass through Err.
    
    Functor map operation.
    
    Time Complexity: O(1) + O(f)
    Space Complexity: O(1) + O(f)
    
    Args:
        result: Result to map
        f: Function to apply to Ok value
        
    Returns:
        Result with mapped value or original Err
    """
    if isinstance(result, Ok):
        return Ok(f(result.value))
    return result


def flat_map(result: Result[T, E], f: Callable[[T], Result[U, E]]) -> Result[U, E]:
    """
    Apply function returning Result to Ok value, flatten.
    
    Monad bind operation (>>=).
    
    Time Complexity: O(1) + O(f)
    Space Complexity: O(1) + O(f)
    
    Args:
        result: Result to flat map
        f: Function returning Result
        
    Returns:
        Flattened Result
    """
    if isinstance(result, Ok):
        return f(result.value)
    return result


def map_err(result: Result[T, E], f: Callable[[E], E]) -> Result[T, E]:
    """
    Apply function to Err value, pass through Ok.
    
    Time Complexity: O(1) + O(f)
    Space Complexity: O(1) + O(f)
    
    Args:
        result: Result to map error
        f: Function to apply to Err
        
    Returns:
        Result with mapped error or original Ok
    """
    if isinstance(result, Err):
        return Err(f(result.error))
    return result
