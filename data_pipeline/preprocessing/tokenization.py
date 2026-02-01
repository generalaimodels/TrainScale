# ════════════════════════════════════════════════════════════════════════════════
# Tokenizer Wrapper - Special Token Handling
# ════════════════════════════════════════════════════════════════════════════════
# Wraps HuggingFace tokenizers with configuration-driven special token setup.
# Provides consistent interface for encoding and decoding.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)

from data_pipeline.core.types import Result, Ok, Err
from data_pipeline.core.errors import TokenizationError
from data_pipeline.core.config_schema import TokenizerConfig

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


# ─────────────────────────────────────────────────────────────────────────────────
# Tokenizer Wrapper
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenizerWrapper:
    """
    Wrapper around HuggingFace tokenizer with configuration.
    
    Provides:
    - Consistent encoding interface
    - Special token management
    - Padding/truncation handling
    - Batch processing utilities
    
    Attributes:
        tokenizer: Underlying HuggingFace tokenizer
        config: Tokenizer configuration
    """
    tokenizer: Any  # PreTrainedTokenizer
    config: TokenizerConfig
    
    @property
    def pad_token_id(self) -> int:
        """Get pad token ID."""
        return self.tokenizer.pad_token_id or 0
    
    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        return self.tokenizer.eos_token_id or 0
    
    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID."""
        return self.tokenizer.bos_token_id or self.eos_token_id
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    def encode(
        self,
        text: str,
        add_special_tokens: Optional[bool] = None,
        max_length: Optional[int] = None,
        truncation: Optional[bool] = None,
        padding: Optional[str] = None,
    ) -> Dict[str, List[int]]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Override config setting
            max_length: Override config max_length
            truncation: Override config truncation
            padding: Override config padding
            
        Returns:
            Dict with input_ids and attention_mask
        """
        return self.tokenizer(
            text,
            add_special_tokens=add_special_tokens if add_special_tokens is not None else self.config.add_special_tokens,
            max_length=max_length or self.config.max_length,
            truncation=truncation if truncation is not None else self.config.truncation,
            padding=padding or self.config.padding,
            return_tensors=None,  # Return lists, not tensors
        )
    
    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: Optional[bool] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, List[List[int]]]:
        """
        Encode batch of texts.
        
        Args:
            texts: List of input texts
            add_special_tokens: Override config setting
            max_length: Override config max_length
            
        Returns:
            Dict with batched input_ids and attention_mask
        """
        return self.tokenizer(
            texts,
            add_special_tokens=add_special_tokens if add_special_tokens is not None else self.config.add_special_tokens,
            max_length=max_length or self.config.max_length,
            truncation=self.config.truncation,
            padding=self.config.padding,
            return_tensors=None,
        )
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_special_tokens_mask(
        self,
        token_ids: List[int],
        already_has_special_tokens: bool = True,
    ) -> List[int]:
        """
        Get mask indicating special token positions.
        
        Args:
            token_ids: Token IDs
            already_has_special_tokens: Whether input has special tokens
            
        Returns:
            Mask with 1 for special tokens, 0 otherwise
        """
        return self.tokenizer.get_special_tokens_mask(
            token_ids, 
            already_has_special_tokens=already_has_special_tokens
        )
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size


# ─────────────────────────────────────────────────────────────────────────────────
# Tokenizer Creation
# ─────────────────────────────────────────────────────────────────────────────────

def create_tokenizer(
    config: TokenizerConfig,
    trust_remote_code: bool = False,
    token: Optional[str] = None,
) -> Result[TokenizerWrapper, TokenizationError]:
    """
    Create tokenizer wrapper from configuration.
    
    Handles:
    - Loading from Hub or local path
    - Adding special tokens
    - Configuring padding/truncation sides
    
    Time Complexity: O(vocab_size) for special token setup
    Space Complexity: O(vocab_size)
    
    Args:
        config: Tokenizer configuration
        trust_remote_code: Trust remote code for custom tokenizers
        token: HuggingFace API token
        
    Returns:
        Result containing TokenizerWrapper or error
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return Err(TokenizationError(
            message="transformers library not installed",
            context={"install": "pip install transformers"}
        ))
    
    token = token or os.environ.get("HF_TOKEN")
    
    try:
        # Load base tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.name_or_path,
            trust_remote_code=trust_remote_code,
            token=token,
        )
        
        # Configure padding side
        tokenizer.padding_side = config.padding_side
        tokenizer.truncation_side = config.truncation_side
        
        # Add/configure special tokens
        special_tokens_to_add = {}
        
        for token_name, token_value in config.special_tokens.items():
            # Check if token already exists
            current = getattr(tokenizer, token_name, None)
            if current is None or current != token_value:
                special_tokens_to_add[token_name] = token_value
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            if "pad_token" not in special_tokens_to_add:
                # Use EOS as pad token if not specified
                if tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    special_tokens_to_add["pad_token"] = "<pad>"
        
        # Add special tokens if any
        if special_tokens_to_add:
            tokenizer.add_special_tokens(special_tokens_to_add)
        
        return Ok(TokenizerWrapper(tokenizer=tokenizer, config=config))
        
    except Exception as e:
        return Err(TokenizationError(
            message=f"Failed to load tokenizer: {e}",
            tokenizer_name=config.name_or_path,
            cause=e,
        ))


def wrap_tokenizer(
    tokenizer: Any,
    config: Optional[TokenizerConfig] = None,
    max_length: int = 2048,
    padding: Literal["max_length", "longest", "do_not_pad"] = "max_length",
    truncation: bool = True,
    padding_side: Literal["left", "right"] = "right",
) -> TokenizerWrapper:
    """
    Wrap an existing tokenizer with TokenizerWrapper.
    
    Use this when you already have a loaded tokenizer and want to use
    the pipeline without reloading it.
    
    Args:
        tokenizer: Pre-loaded HuggingFace tokenizer
        config: Optional TokenizerConfig (will be inferred if not provided)
        max_length: Maximum sequence length (used if config not provided)
        padding: Padding strategy (used if config not provided)
        truncation: Whether to truncate (used if config not provided)
        padding_side: Side to pad (used if config not provided)
        
    Returns:
        TokenizerWrapper wrapping the provided tokenizer
    """
    if config is None:
        # Infer config from tokenizer
        config = TokenizerConfig(
            name_or_path=getattr(tokenizer, "name_or_path", "unknown"),
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            padding_side=padding_side,
            truncation_side=getattr(tokenizer, "truncation_side", "right"),
            add_special_tokens=True,
        )
    
    # Apply config settings to tokenizer
    tokenizer.padding_side = config.padding_side
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = config.truncation_side
    
    # Ensure pad token exists
    ensure_tokenizer_padding(tokenizer)
    
    return TokenizerWrapper(tokenizer=tokenizer, config=config)


def ensure_tokenizer_padding(tokenizer: Any) -> None:
    """
    Ensure tokenizer has proper padding configuration.
    
    Sets pad_token to eos_token if not present.
    
    Args:
        tokenizer: HuggingFace tokenizer
    """
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # Add a new pad token as last resort
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
