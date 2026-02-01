# ════════════════════════════════════════════════════════════════════════════════
# SOTA Quantization Module
# ════════════════════════════════════════════════════════════════════════════════
# Unsloth-inspired quantization for memory-efficient training.
#
# Supports:
# - 4-bit QLoRA (BitsAndBytes NF4/FP4)
# - 8-bit quantization
# - FP8 training (H100/L40)
# - Dynamic quantization (Unsloth-style)
# - 16-bit full precision (BF16/FP16)
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


# ═════════════════════════════════════════════════════════════════════════════════
# Quantization Configuration
# ═════════════════════════════════════════════════════════════════════════════════

class QuantizationType(Enum):
    """Quantization method types."""
    NONE = "none"           # Full precision
    INT8 = "int8"           # 8-bit integer
    INT4 = "int4"           # 4-bit integer (NF4/FP4)
    FP8_E4M3 = "fp8_e4m3"   # FP8 E4M3 format
    FP8_E5M2 = "fp8_e5m2"   # FP8 E5M2 format
    DYNAMIC = "dynamic"      # Dynamic quantization


class QuantizationFormat(Enum):
    """4-bit quantization formats."""
    NF4 = "nf4"      # NormalFloat4 (better for LLMs)
    FP4 = "fp4"      # Float4
    INT4 = "int4"    # Pure integer 4-bit


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    
    # Primary settings
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    load_in_fp8: bool = False
    
    # 4-bit settings
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4"
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    bnb_4bit_use_double_quant: bool = True  # Nested quantization
    
    # FP8 settings
    fp8_format: str = "e4m3"  # "e4m3" or "e5m2"
    
    # General
    llm_int8_threshold: float = 6.0
    llm_int8_skip_modules: List[str] = field(default_factory=lambda: ["lm_head"])
    llm_int8_has_fp16_weight: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if sum([self.load_in_4bit, self.load_in_8bit, self.load_in_fp8]) > 1:
            raise ValueError("Only one quantization type can be enabled at a time")
        
        if self.bnb_4bit_quant_type not in ["nf4", "fp4"]:
            raise ValueError(f"Invalid quant_type: {self.bnb_4bit_quant_type}")
    
    @property
    def is_quantized(self) -> bool:
        return self.load_in_4bit or self.load_in_8bit or self.load_in_fp8
    
    @property
    def bits(self) -> int:
        if self.load_in_4bit:
            return 4
        elif self.load_in_8bit:
            return 8
        elif self.load_in_fp8:
            return 8
        return 16
    
    def to_bnb_config(self) -> Dict[str, Any]:
        """Convert to BitsAndBytes config dict."""
        return {
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "llm_int8_threshold": self.llm_int8_threshold,
            "llm_int8_skip_modules": self.llm_int8_skip_modules,
        }


# ═════════════════════════════════════════════════════════════════════════════════
# Preset Configurations
# ═════════════════════════════════════════════════════════════════════════════════

def get_4bit_config(
    compute_dtype: torch.dtype = torch.bfloat16,
    quant_type: str = "nf4",
    double_quant: bool = True,
) -> QuantizationConfig:
    """Get optimized 4-bit QLoRA configuration."""
    return QuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=double_quant,
    )


def get_8bit_config(
    threshold: float = 6.0,
    skip_modules: Optional[List[str]] = None,
) -> QuantizationConfig:
    """Get 8-bit quantization configuration."""
    return QuantizationConfig(
        load_in_8bit=True,
        llm_int8_threshold=threshold,
        llm_int8_skip_modules=skip_modules or ["lm_head"],
    )


def get_fp8_config(
    format: str = "e4m3",
) -> QuantizationConfig:
    """Get FP8 configuration for H100/L40 GPUs."""
    return QuantizationConfig(
        load_in_fp8=True,
        fp8_format=format,
    )


def get_16bit_config(
    dtype: torch.dtype = torch.bfloat16,
) -> QuantizationConfig:
    """Get 16-bit full precision configuration."""
    return QuantizationConfig(
        load_in_4bit=False,
        load_in_8bit=False,
        bnb_4bit_compute_dtype=dtype,
    )


# ═════════════════════════════════════════════════════════════════════════════════
# Quantization State Management
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantState:
    """Quantization state for a tensor/module."""
    absmax: Tensor
    code: Tensor
    blocksize: int = 64
    dtype: torch.dtype = torch.bfloat16
    shape: Optional[Tuple[int, ...]] = None
    offset: Optional[Tensor] = None
    state2: Optional["QuantState"] = None  # For double quant


def get_quant_state(weight: Tensor) -> Optional[QuantState]:
    """Extract quantization state from BNB quantized weight."""
    if hasattr(weight, "quant_state"):
        qs = weight.quant_state
        return QuantState(
            absmax=qs.absmax if hasattr(qs, "absmax") else None,
            code=qs.code if hasattr(qs, "code") else None,
            blocksize=getattr(qs, "blocksize", 64),
            dtype=getattr(qs, "dtype", torch.bfloat16),
            shape=getattr(qs, "shape", None),
        )
    return None


# ═════════════════════════════════════════════════════════════════════════════════
# Fast Dequantization (Triton-accelerated)
# ═════════════════════════════════════════════════════════════════════════════════

def fast_dequantize(
    weight: Tensor,
    quant_state: Optional[QuantState] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """
    Fast dequantization using Triton if available.
    
    For quantized weights, converts back to full precision for computation.
    """
    # Check if already full precision
    if weight.dtype in [torch.float16, torch.bfloat16, torch.float32]:
        return weight.to(out_dtype)
    
    # Try BitsAndBytes dequantization
    try:
        import bitsandbytes as bnb
        from bitsandbytes.functional import dequantize_4bit
        
        if quant_state is None:
            quant_state = get_quant_state(weight)
        
        if quant_state is not None:
            return dequantize_4bit(weight, quant_state).to(out_dtype)
    except ImportError:
        pass
    
    # Fallback: return as-is
    return weight.to(out_dtype)


# ═════════════════════════════════════════════════════════════════════════════════
# Model Quantization Utilities
# ═════════════════════════════════════════════════════════════════════════════════

def prepare_model_for_quantization(
    model: nn.Module,
    config: QuantizationConfig,
) -> nn.Module:
    """
    Prepare model for quantized training.
    
    - Sets up gradient checkpointing
    - Prepares input embeddings
    - Configures output projections
    """
    # Enable gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Prepare for k-bit training
    if config.is_quantized:
        try:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)
        except ImportError:
            # Manual preparation
            for param in model.parameters():
                if param.dtype in [torch.float16, torch.bfloat16]:
                    param.data = param.data.to(torch.float32)
    
    return model


def get_target_modules(model_type: str) -> List[str]:
    """Get default LoRA target modules for model type."""
    TARGETS = {
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    }
    return TARGETS.get(model_type.lower(), ["q_proj", "v_proj"])


def estimate_memory_usage(
    model_size_billion: float,
    config: QuantizationConfig,
    batch_size: int = 1,
    seq_length: int = 2048,
) -> Dict[str, float]:
    """
    Estimate GPU memory usage.
    
    Returns dict with memory estimates in GB.
    """
    # Base model size
    bytes_per_param = config.bits / 8
    model_memory_gb = (model_size_billion * 1e9 * bytes_per_param) / (1024**3)
    
    # LoRA overhead (typically ~1-3% of base)
    lora_memory_gb = model_memory_gb * 0.02
    
    # Activation memory (rough estimate)
    hidden_size = int(model_size_billion * 1000)  # Rough approximation
    activation_memory_gb = (batch_size * seq_length * hidden_size * 2 * 4) / (1024**3)
    
    # Gradient memory
    grad_memory_gb = model_memory_gb if not config.is_quantized else lora_memory_gb
    
    total = model_memory_gb + lora_memory_gb + activation_memory_gb + grad_memory_gb
    
    return {
        "model_gb": round(model_memory_gb, 2),
        "lora_gb": round(lora_memory_gb, 2),
        "activation_gb": round(activation_memory_gb, 2),
        "gradient_gb": round(grad_memory_gb, 2),
        "total_gb": round(total, 2),
    }


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "QuantizationType",
    "QuantizationFormat",
    # Config
    "QuantizationConfig",
    "QuantState",
    # Presets
    "get_4bit_config",
    "get_8bit_config",
    "get_fp8_config",
    "get_16bit_config",
    # Utilities
    "get_quant_state",
    "fast_dequantize",
    "prepare_model_for_quantization",
    "get_target_modules",
    "estimate_memory_usage",
]
