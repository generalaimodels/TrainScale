# ════════════════════════════════════════════════════════════════════════════════
# Collate Functions - Loss-Aligned Batch Construction
# ════════════════════════════════════════════════════════════════════════════════
# Creates optimized collate functions for dynamic batching.
# Ensures perfect alignment with loss functions (CrossEntropyLoss ignore_index).
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

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

import torch
from torch import Tensor

from data_pipeline.core.config_schema import OutputSchema, TensorSpec


# ─────────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────────

# Default padding values aligned with loss functions
DEFAULT_PAD_TOKEN_ID: int = 0
DEFAULT_LABEL_PAD_ID: int = -100  # CrossEntropyLoss ignore_index

# Dtype mapping from string to torch dtype
DTYPE_MAP: Dict[str, torch.dtype] = {
    "long": torch.long,
    "int": torch.int,
    "float": torch.float,
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "bool": torch.bool,
}


# ─────────────────────────────────────────────────────────────────────────────────
# Padding Utilities
# ─────────────────────────────────────────────────────────────────────────────────

def pad_sequence_right(
    sequences: List[Tensor],
    padding_value: int = 0,
    max_length: Optional[int] = None,
) -> Tensor:
    """
    Pad sequences on the right side.
    
    Vectorized implementation using torch operations.
    
    Time Complexity: O(batch_size * max_length)
    Space Complexity: O(batch_size * max_length)
    
    Args:
        sequences: List of 1D tensors to pad
        padding_value: Value to use for padding
        max_length: Maximum length (uses longest sequence if None)
        
    Returns:
        Padded 2D tensor of shape (batch_size, max_length)
    """
    # Determine max length
    lengths = [len(seq) for seq in sequences]
    target_length = max_length or max(lengths)
    batch_size = len(sequences)
    
    # Preallocate output tensor
    output = torch.full(
        (batch_size, target_length),
        fill_value=padding_value,
        dtype=sequences[0].dtype,
        device=sequences[0].device,
    )
    
    # Fill in sequences
    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        actual_length = min(length, target_length)
        output[i, :actual_length] = seq[:actual_length]
    
    return output


def pad_sequence_left(
    sequences: List[Tensor],
    padding_value: int = 0,
    max_length: Optional[int] = None,
) -> Tensor:
    """
    Pad sequences on the left side.
    
    Useful for decoder-only models during inference.
    
    Time Complexity: O(batch_size * max_length)
    Space Complexity: O(batch_size * max_length)
    
    Args:
        sequences: List of 1D tensors to pad
        padding_value: Value to use for padding
        max_length: Maximum length (uses longest sequence if None)
        
    Returns:
        Padded 2D tensor of shape (batch_size, max_length)
    """
    lengths = [len(seq) for seq in sequences]
    target_length = max_length or max(lengths)
    batch_size = len(sequences)
    
    # Preallocate output tensor
    output = torch.full(
        (batch_size, target_length),
        fill_value=padding_value,
        dtype=sequences[0].dtype,
        device=sequences[0].device,
    )
    
    # Fill in sequences (right-aligned)
    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        actual_length = min(length, target_length)
        start_idx = target_length - actual_length
        output[i, start_idx:] = seq[-actual_length:]
    
    return output


# ─────────────────────────────────────────────────────────────────────────────────
# Collate Function Factory
# ─────────────────────────────────────────────────────────────────────────────────

def create_collate_fn(
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
    label_pad_token_id: int = DEFAULT_LABEL_PAD_ID,
    padding_side: Literal["left", "right"] = "right",
    max_length: Optional[int] = None,
    output_schema: Optional[OutputSchema] = None,
    return_tensors: bool = True,
) -> Callable[[List[Dict[str, Tensor]]], Dict[str, Tensor]]:
    """
    Create optimized collate function for DataLoader.
    
    Guarantees:
    1. labels[i] = -100 for all padding positions
    2. Tensor dtypes match loss function expectations
    3. Shapes are (batch, seq_len) for all sequence tensors
    
    Implementation:
    - Preallocates output tensors
    - Vectorized padding via torch operations
    - No Python loops in critical path
    
    Time Complexity: O(batch_size * max_sequence_length)
    Space Complexity: O(batch_size * max_sequence_length)
    
    Args:
        pad_token_id: Token ID for padding input_ids
        label_pad_token_id: Value for padding labels (-100 for CE loss)
        padding_side: Which side to pad ("left" or "right")
        max_length: Optional maximum sequence length
        output_schema: Optional OutputSchema for dtype configuration
        return_tensors: Return PyTorch tensors (vs lists)
        
    Returns:
        Collate function for use with DataLoader
    """
    # Select padding function
    pad_fn = pad_sequence_left if padding_side == "left" else pad_sequence_right
    
    # Get dtype for labels from schema if provided
    label_dtype = torch.long
    if output_schema and output_schema.labels:
        label_dtype = DTYPE_MAP.get(output_schema.labels.dtype, torch.long)
    
    def collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """
        Collate batch of examples into padded tensors.
        
        Args:
            batch: List of dicts with input_ids, attention_mask, labels
            
        Returns:
            Dict with batched and padded tensors
        """
        if not batch:
            return {}
        
        # Extract sequences
        input_ids_list = [ex["input_ids"] for ex in batch]
        attention_mask_list = [ex["attention_mask"] for ex in batch]
        labels_list = [ex["labels"] for ex in batch]
        
        # Pad sequences
        input_ids = pad_fn(input_ids_list, pad_token_id, max_length)
        attention_mask = pad_fn(attention_mask_list, 0, max_length)
        labels = pad_fn(labels_list, label_pad_token_id, max_length)
        
        # Ensure labels are properly masked for padding
        # Where attention_mask is 0, labels should be -100
        labels = labels.masked_fill(attention_mask == 0, label_pad_token_id)
        
        # Cast to expected dtype
        labels = labels.to(label_dtype)
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        # Handle any extra keys in batch
        extra_keys = set(batch[0].keys()) - {"input_ids", "attention_mask", "labels"}
        for key in extra_keys:
            values = [ex[key] for ex in batch]
            if isinstance(values[0], Tensor):
                # Determine padding value for this key
                pad_val = 0  # Default
                if output_schema and key in output_schema.extras:
                    pad_val = output_schema.extras[key].pad_value or 0
                result[key] = pad_fn(values, pad_val, max_length)
            else:
                # Non-tensor values, just list them
                result[key] = values
        
        return result
    
    return collate_fn


# ─────────────────────────────────────────────────────────────────────────────────
# Specialized Collate Functions
# ─────────────────────────────────────────────────────────────────────────────────

def create_causal_lm_collate(
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
    max_length: Optional[int] = None,
    padding_side: Literal["left", "right"] = "right",
) -> Callable[[List[Dict[str, Tensor]]], Dict[str, Tensor]]:
    """
    Create collate function optimized for causal language modeling.
    
    Labels are shifted right by 1 position (standard causal LM setup).
    
    Args:
        pad_token_id: Token ID for padding
        max_length: Maximum sequence length
        padding_side: Which side to pad
        
    Returns:
        Collate function for causal LM
    """
    base_collate = create_collate_fn(
        pad_token_id=pad_token_id,
        label_pad_token_id=DEFAULT_LABEL_PAD_ID,
        padding_side=padding_side,
        max_length=max_length,
    )
    
    def collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        result = base_collate(batch)
        
        # Shift labels right by 1 for causal LM
        # Labels[i] = input_ids[i+1], last position = -100
        labels = result["labels"]
        shifted_labels = torch.full_like(labels, DEFAULT_LABEL_PAD_ID)
        shifted_labels[:, :-1] = labels[:, 1:]
        result["labels"] = shifted_labels
        
        return result
    
    return collate_fn


def create_seq2seq_collate(
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
    decoder_start_token_id: int = 0,
    max_length: Optional[int] = None,
    max_target_length: Optional[int] = None,
) -> Callable[[List[Dict[str, Tensor]]], Dict[str, Tensor]]:
    """
    Create collate function for sequence-to-sequence models.
    
    Handles separate encoder and decoder inputs with proper padding.
    
    Args:
        pad_token_id: Token ID for padding
        decoder_start_token_id: Token to start decoder
        max_length: Maximum encoder sequence length
        max_target_length: Maximum decoder sequence length
        
    Returns:
        Collate function for seq2seq
    """
    def collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        # Encoder inputs
        input_ids_list = [ex["input_ids"] for ex in batch]
        attention_mask_list = [ex["attention_mask"] for ex in batch]
        
        input_ids = pad_sequence_right(input_ids_list, pad_token_id, max_length)
        attention_mask = pad_sequence_right(attention_mask_list, 0, max_length)
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        # Decoder inputs (if present)
        if "decoder_input_ids" in batch[0]:
            decoder_ids_list = [ex["decoder_input_ids"] for ex in batch]
            decoder_mask_list = [ex.get("decoder_attention_mask", torch.ones_like(ex["decoder_input_ids"])) for ex in batch]
            
            result["decoder_input_ids"] = pad_sequence_right(
                decoder_ids_list, pad_token_id, max_target_length
            )
            result["decoder_attention_mask"] = pad_sequence_right(
                decoder_mask_list, 0, max_target_length
            )
        
        # Labels
        if "labels" in batch[0]:
            labels_list = [ex["labels"] for ex in batch]
            result["labels"] = pad_sequence_right(
                labels_list, DEFAULT_LABEL_PAD_ID, max_target_length
            )
        
        return result
    
    return collate_fn
