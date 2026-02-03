# ════════════════════════════════════════════════════════════════════════════════
# SOTA Compute Kernels Registry
# ════════════════════════════════════════════════════════════════════════════════
# Unified interface for high-performance training primitives.
#
# Optimization Hierarchy:
# 1. Custom Triton Kernels (Max Performance)
# 2. Torch Compile (Inductor)
# 3. Optimized CUDA Kernels (cuBLAS/cuDNN)
# 4. PyTorch Native Fallbacks
# ════════════════════════════════════════════════════════════════════════════════

from typing import Optional

# 1. Compiler & Infrastructure
from .compiler import compile_model
from .triton_kernels import is_triton_available

# 2. Normalization & Activations
from .layernorm_kernels import (
    fused_layer_norm,
    fast_rms_layernorm,
    Fast_RMS_LayerNorm,
    fused_add_rms_layernorm,
)
from .triton_kernels import (
    fused_softmax,
    fused_gelu,
    swiglu_forward,
    swiglu_backward,
    geglu_forward,
    fast_cross_entropy_loss,
)

# 3. Attention Mechanisms (Flash & Flex)
from .flash_attention import (
    flash_attention,
    FlashAttention,
    MultiHeadFlashAttention,
    is_flash_attn_available,
)
from .flex_attention import (
    attention_softcapping_compiled,
    slow_attention_softcapping,
    scaled_dot_product_attention,
    create_causal_mask,
    create_sliding_window_causal_mask,
)

# 4. RoPE Embeddings
from .rope_kernels import (
    fast_rope_embedding,
    Fast_RoPE_Embedding,
    inplace_rope_embedding,
    precompute_freqs_cis,
)

# 5. LoRA Primitives (Fused)
from .fast_lora_mlp import (
    matmul_lora,
    LoRA_MLP,
    LoRA_QKV,
    apply_lora_mlp_swiglu,
    apply_lora_qkv,
    get_lora_parameters,
)

# 6. Mixture of Experts (MoE)
from .moe_kernels import (
    GroupedGEMM,
    grouped_gemm_forward,
    grouped_gemm_backward_dX,
    grouped_gemm_backward_dW,
    supports_tma,
)

# 7. FP8 Quantization
from .fp8_kernels import (
    activation_quant,
    weight_dequant,
    FP8BlockLinear,
)

# 8. Distributed Primitives
from .distributed_kernels import (
    DistributedKernels,
    pinned_memory_transfer,
)

__all__ = [
    # Infrastructure
    "compile_model",
    "is_triton_available",
    "supports_tma",
    
    # Norms & Activations
    "fused_layer_norm",
    "fast_rms_layernorm",
    "Fast_RMS_LayerNorm",
    "fused_add_rms_layernorm",
    "fused_softmax",
    "fused_gelu",
    "swiglu_forward",
    "swiglu_backward",
    "geglu_forward",
    "fast_cross_entropy_loss",
    
    # Attention
    "flash_attention",
    "FlashAttention",
    "MultiHeadFlashAttention",
    "is_flash_attn_available",
    "attention_softcapping_compiled",
    "slow_attention_softcapping",
    "scaled_dot_product_attention",
    "create_causal_mask",
    "create_sliding_window_causal_mask",
    
    # RoPE
    "fast_rope_embedding",
    "Fast_RoPE_Embedding",
    "inplace_rope_embedding",
    "precompute_freqs_cis",
    
    # LoRA
    "matmul_lora",
    "LoRA_MLP",
    "LoRA_QKV",
    "apply_lora_mlp_swiglu",
    "apply_lora_qkv",
    "get_lora_parameters",
    
    # MoE
    "GroupedGEMM",
    "grouped_gemm_forward",
    "grouped_gemm_backward_dX",
    "grouped_gemm_backward_dW",
    
    # FP8
    "activation_quant",
    "weight_dequant",
    "FP8BlockLinear",
    
    # Distributed
    "DistributedKernels",
    "pinned_memory_transfer",
]
