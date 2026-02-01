# ════════════════════════════════════════════════════════════════════════════════
# SOTA Model Registry
# ════════════════════════════════════════════════════════════════════════════════
# Unsloth-inspired model registry for automatic layer patching and optimization.
#
# Supports:
# - Llama, Qwen, Gemma, Mistral, Deepseek, Phi
# - 4-bit BNB, 16-bit, FP8, GGUF quantization
# - Auto-detection and patching of model layers
# - Multimodal and embedding models
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Type, Callable, Any

import torch
import torch.nn as nn


# ═════════════════════════════════════════════════════════════════════════════════
# Quantization Types
# ═════════════════════════════════════════════════════════════════════════════════

class QuantType(Enum):
    """Supported quantization types."""
    NONE = "none"           # Full precision (fp32/fp16/bf16)
    BNB_4BIT = "bnb-4bit"   # BitsAndBytes 4-bit
    BNB_8BIT = "bnb-8bit"   # BitsAndBytes 8-bit
    FP8 = "fp8"             # FP8 training
    GGUF = "gguf"           # GGUF format
    DYNAMIC = "dynamic"     # Unsloth dynamic quantization
    BF16 = "bf16"           # BF16 (DeepSeek V3 style)


class TrainingMode(Enum):
    """Training mode types."""
    FULL_FINETUNE = "full"      # Full parameter fine-tuning
    LORA = "lora"               # LoRA adapters
    QLORA = "qlora"             # QLoRA (4-bit + LoRA)
    PRETRAINING = "pretrain"    # Pretraining from scratch
    RL = "rl"                   # Reinforcement learning


# Quantization tags for HuggingFace paths
QUANT_TAG_MAP = {
    QuantType.NONE: None,
    QuantType.BNB_4BIT: "bnb-4bit",
    QuantType.BNB_8BIT: "bnb-8bit",
    QuantType.FP8: "fp8",
    QuantType.GGUF: "GGUF",
    QuantType.DYNAMIC: "unsloth-bnb-4bit",
    QuantType.BF16: "bf16",
}


# ═════════════════════════════════════════════════════════════════════════════════
# Model Info Classes
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelInfo:
    """Information about a registered model."""
    org: str                           # Organization (meta-llama, Qwen, google, etc.)
    base_name: str                     # Base model name (Llama, Qwen, Gemma)
    version: str                       # Version (3.1, 2.5, 2)
    size: str                          # Model size (1B, 7B, 70B)
    name: Optional[str] = None         # Full model name (auto-constructed)
    is_multimodal: bool = False        # Supports vision/audio
    instruct_tag: Optional[str] = None # Instruct/Chat variant
    quant_type: QuantType = QuantType.NONE
    description: Optional[str] = None
    
    # Layer configuration for patching
    attention_class: Optional[str] = None    # e.g., "LlamaAttention"
    mlp_class: Optional[str] = None          # e.g., "LlamaMLP"
    layernorm_class: Optional[str] = None    # e.g., "LlamaRMSNorm"
    rope_class: Optional[str] = None         # e.g., "LlamaRotaryEmbedding"
    
    def __post_init__(self):
        if self.name is None:
            self.name = self.construct_model_name(
                self.base_name, self.version, self.size,
                self.quant_type, self.instruct_tag
            )
    
    @classmethod
    def construct_model_name(
        cls,
        base_name: str,
        version: str,
        size: str,
        quant_type: QuantType = QuantType.NONE,
        instruct_tag: Optional[str] = None,
    ) -> str:
        """Construct full model name from components."""
        key = f"{base_name}-{version}-{size}B"
        if instruct_tag:
            key = f"{key}-{instruct_tag}"
        if quant_type != QuantType.NONE:
            tag = QUANT_TAG_MAP.get(quant_type)
            if tag:
                key = f"{key}-{tag}"
        return key
    
    @property
    def model_path(self) -> str:
        """Full HuggingFace model path."""
        return f"{self.org}/{self.name}"


@dataclass
class ModelMeta:
    """Metadata for registering a model family."""
    org: str                           # Organization
    base_name: str                     # Base model name
    model_version: str                 # Version string
    model_info_cls: Type[ModelInfo]    # ModelInfo subclass
    model_sizes: List[str] = field(default_factory=list)
    instruct_tags: List[Optional[str]] = field(default_factory=list)
    quant_types: List[QuantType] = field(default_factory=list)
    is_multimodal: bool = False
    
    # Layer class names for patching
    attention_class: Optional[str] = None
    mlp_class: Optional[str] = None
    layernorm_class: Optional[str] = None
    rope_class: Optional[str] = None


# ═════════════════════════════════════════════════════════════════════════════════
# Model Registry
# ═════════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: Dict[str, ModelInfo] = {}
LAYER_PATCHES: Dict[str, Callable] = {}  # Maps layer class names to patch functions


def register_model(
    model_info_cls: Type[ModelInfo],
    org: str,
    base_name: str,
    version: str,
    size: str,
    instruct_tag: Optional[str] = None,
    quant_type: QuantType = QuantType.NONE,
    is_multimodal: bool = False,
    name: Optional[str] = None,
    **kwargs,
) -> None:
    """Register a model in the global registry."""
    name = name or model_info_cls.construct_model_name(
        base_name, version, size, quant_type, instruct_tag
    )
    key = f"{org}/{name}"
    
    if key in MODEL_REGISTRY:
        warnings.warn(f"Model {key} already registered, skipping")
        return
    
    MODEL_REGISTRY[key] = model_info_cls(
        org=org,
        base_name=base_name,
        version=version,
        size=size,
        is_multimodal=is_multimodal,
        instruct_tag=instruct_tag,
        quant_type=quant_type,
        name=name,
        **kwargs,
    )


def register_models_from_meta(
    model_meta: ModelMeta,
    include_original: bool = False,
) -> None:
    """Register all model variants from ModelMeta."""
    for size in model_meta.model_sizes:
        for instruct_tag in model_meta.instruct_tags:
            for quant_type in model_meta.quant_types:
                # Register optimized version under "unsloth" org
                register_model(
                    model_info_cls=model_meta.model_info_cls,
                    org="unsloth",
                    base_name=model_meta.base_name,
                    version=model_meta.model_version,
                    size=size,
                    instruct_tag=instruct_tag,
                    quant_type=quant_type,
                    is_multimodal=model_meta.is_multimodal,
                    attention_class=model_meta.attention_class,
                    mlp_class=model_meta.mlp_class,
                    layernorm_class=model_meta.layernorm_class,
                    rope_class=model_meta.rope_class,
                )
            
            # Register original model if requested
            if include_original:
                register_model(
                    model_info_cls=model_meta.model_info_cls,
                    org=model_meta.org,
                    base_name=model_meta.base_name,
                    version=model_meta.model_version,
                    size=size,
                    instruct_tag=instruct_tag,
                    quant_type=QuantType.NONE,
                    is_multimodal=model_meta.is_multimodal,
                    attention_class=model_meta.attention_class,
                    mlp_class=model_meta.mlp_class,
                    layernorm_class=model_meta.layernorm_class,
                    rope_class=model_meta.rope_class,
                )


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """Get model info by ID (org/name)."""
    return MODEL_REGISTRY.get(model_id)


def search_models(
    base_name: Optional[str] = None,
    quant_type: Optional[QuantType] = None,
    is_multimodal: Optional[bool] = None,
) -> List[ModelInfo]:
    """Search registered models by criteria."""
    results = []
    for info in MODEL_REGISTRY.values():
        if base_name and info.base_name != base_name:
            continue
        if quant_type and info.quant_type != quant_type:
            continue
        if is_multimodal is not None and info.is_multimodal != is_multimodal:
            continue
        results.append(info)
    return results


# ═════════════════════════════════════════════════════════════════════════════════
# Layer Patching
# ═════════════════════════════════════════════════════════════════════════════════

def register_layer_patch(layer_class_name: str, patch_fn: Callable) -> None:
    """Register a patch function for a layer class."""
    LAYER_PATCHES[layer_class_name] = patch_fn


def patch_model(model: nn.Module, model_info: Optional[ModelInfo] = None) -> nn.Module:
    """
    Apply SOTA optimizations to a model's layers.
    
    Patches:
    - RMSNorm → Fast_RMS_LayerNorm
    - Attention → Flash Attention
    - MLP → SwiGLU/GeGLU kernels
    - RoPE → Fast RoPE
    """
    from data_pipeline.trainer.kernels.triton_kernels import (
        fast_rms_layernorm,
        swiglu_forward,
    )
    
    patched_layers = []
    
    for name, module in model.named_modules():
        module_class = module.__class__.__name__
        
        if module_class in LAYER_PATCHES:
            patch_fn = LAYER_PATCHES[module_class]
            patch_fn(module)
            patched_layers.append((name, module_class))
    
    if patched_layers:
        print(f"✓ Patched {len(patched_layers)} layers for SOTA performance")
    
    return model


def auto_patch_layernorm(module: nn.Module) -> None:
    """Patch RMSNorm forward to use Triton kernel."""
    from data_pipeline.trainer.kernels.triton_kernels import fast_rms_layernorm
    
    original_forward = module.forward
    weight = module.weight
    eps = getattr(module, 'eps', getattr(module, 'variance_epsilon', 1e-6))
    
    def patched_forward(hidden_states):
        return fast_rms_layernorm(hidden_states, weight, eps)
    
    module.forward = patched_forward


# Register default patches for all architectures
register_layer_patch("LlamaRMSNorm", auto_patch_layernorm)
register_layer_patch("Qwen2RMSNorm", auto_patch_layernorm)
register_layer_patch("Qwen3RMSNorm", auto_patch_layernorm)
register_layer_patch("GemmaRMSNorm", auto_patch_layernorm)
register_layer_patch("Gemma2RMSNorm", auto_patch_layernorm)
register_layer_patch("MistralRMSNorm", auto_patch_layernorm)
register_layer_patch("Phi3RMSNorm", auto_patch_layernorm)
register_layer_patch("Phi4RMSNorm", auto_patch_layernorm)
register_layer_patch("YiRMSNorm", auto_patch_layernorm)
register_layer_patch("FalconRMSNorm", auto_patch_layernorm)
register_layer_patch("FalconH1RMSNorm", auto_patch_layernorm)
register_layer_patch("CohereLayerNorm", auto_patch_layernorm)
register_layer_patch("GraniteRMSNorm", auto_patch_layernorm)
register_layer_patch("StarcoderLayerNorm", auto_patch_layernorm)
register_layer_patch("DeepseekRMSNorm", auto_patch_layernorm)
register_layer_patch("DeepseekV2RMSNorm", auto_patch_layernorm)
register_layer_patch("MixtralRMSNorm", auto_patch_layernorm)
register_layer_patch("DbrxLayerNorm", auto_patch_layernorm)
register_layer_patch("GrokRMSNorm", auto_patch_layernorm)
register_layer_patch("InternVLRMSNorm", auto_patch_layernorm)


# ═════════════════════════════════════════════════════════════════════════════════
# Import All Model Families (300+ models total)
# ═════════════════════════════════════════════════════════════════════════════════

try:
    from . import _llama      # Llama 2/3/3.1/3.2/3.3 (60 models)
    from . import _qwen       # Qwen 1/1.5/2/2.5/3 (50 models)
    from . import _gemma      # Gemma 1/2/3n (30 models)
    from . import _mistral    # Mistral 7B/Nemo/Large (25 models)
    from . import _phi        # Phi-3/Phi-4 (20 models)
    from . import _yi         # Yi-1/1.5/Coder (25 models)
    from . import _falcon     # Falcon 1/2/H1 (20 models)
    from . import _cohere_granite  # Command R, Aya, Granite (25 models)
    from . import _code_models     # StarCoder, CodeLlama, DeepSeek-Coder (40 models)
    from . import _moe_models      # Mixtral, DBRX, Grok, Qwen-MoE (25 models)
    from . import _vision_models   # LLaVA, Qwen-VL, Pixtral, InternVL (35 models)
except ImportError as e:
    import warnings
    warnings.warn(f"Some model families failed to import: {e}")


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "QuantType",
    "TrainingMode",
    "QUANT_TAG_MAP",
    # Classes
    "ModelInfo",
    "ModelMeta",
    # Registry
    "MODEL_REGISTRY",
    "register_model",
    "register_models_from_meta",
    "get_model_info",
    "search_models",
    # Patching
    "LAYER_PATCHES",
    "register_layer_patch",
    "patch_model",
    "auto_patch_layernorm",
]
