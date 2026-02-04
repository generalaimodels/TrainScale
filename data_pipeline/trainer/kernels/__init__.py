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
from .compiler import (
    compile_model,
    CompilationMode,
    CompilationBackend,
    PrecisionMode,
    CompilationConfig,
    InductorConfig,
)
from .triton_kernels import (
    is_triton_available,
    Fast_CrossEntropyLoss,
)

# 2. Normalization & Activations
from .layernorm_kernels import (
    fused_layer_norm,
    fast_rms_layernorm,
    fused_add_rms_layernorm,
    LayerNormConfig,
)
from .triton_kernels import (
    fused_softmax,
    fused_gelu,
    swiglu_forward,
    geglu_forward,
    fast_cross_entropy_loss,
    Fast_RMS_LayerNorm,
    Fast_SwiGLU,
    Fast_GeGLU,
    TritonRMSNorm,
    TritonSwiGLU,
    TritonGeGLU,
)

# 3. Attention Mechanisms (Flash & Flex)
from .flash_attention import (
    flash_attention,
    FlashAttention,
    MultiHeadFlashAttention,
    is_flash_attn_available,
    FlashAttentionConfig,
    AttentionMaskType,
    AttentionOutput,
)
from .flex_attention import (
    attention_softcapping_compiled,
    slow_attention_softcapping,
    scaled_dot_product_attention,
    create_causal_mask,
    create_sliding_window_causal_mask,
    FlexAttentionConfig,
    AttentionBackend,
    BackendCapabilities,
)

# 4. RoPE Embeddings
from .rope_kernels import (
    fast_rope_embedding,
    Fast_RoPE_Embedding,
    inplace_rope_embedding,
    precompute_freqs_cis,
    RoPEConfig,
    RoPEScalingType,
    RoPEFrequencyCache,
)

# 5. LoRA Primitives (Fused)
from .fast_lora_mlp import (
    matmul_lora,
    LoRA_MLP,
    LoRA_QKV,
    apply_lora_mlp_swiglu,
    apply_lora_qkv,
    get_lora_parameters,
    torch_amp_custom_fwd,
    torch_amp_custom_bwd,
)

# 6. Mixture of Experts (MoE)
from .moe_kernels import (
    supports_tma,
    MoEKernelConfig,
    AcceleratorArch,
    get_accelerator_arch,
    supports_wgmma,
)

# 7. FP8 Quantization
from .fp8_kernels import (
    row_quantize_fp8,
    block_quantize_fp8,
    block_dequantize_fp8,
    fp8_matmul_block_scaled,
    fp8_matmul_row_scaled,
    FP8Linear,
    FP8Format,
    FP8Config,
    FP8ScaleManager,
)

# 8. Distributed Primitives
from .distributed_kernels import (
    DistributedKernels,
    pinned_memory_transfer,
    DistributedBackend,
    DistributedConfig,
    get_distributed_backend,
    get_world_info,
)

__all__ = [
    # Infrastructure
    "compile_model",
    "CompilationMode",
    "CompilationBackend",
    "PrecisionMode",
    "CompilationConfig",
    "InductorConfig",
    "is_triton_available",
    "supports_tma",
    "AcceleratorArch",
    "get_accelerator_arch",
    "supports_wgmma",
    
    # Norms & Activations
    "fused_layer_norm",
    "fast_rms_layernorm",
    "fused_add_rms_layernorm",
    "LayerNormConfig",
    "fused_softmax",
    "fused_gelu",
    "swiglu_forward",
    "geglu_forward",
    "fast_cross_entropy_loss",
    "Fast_CrossEntropyLoss",
    "Fast_RMS_LayerNorm",
    "Fast_SwiGLU",
    "Fast_GeGLU",
    "TritonRMSNorm",
    "TritonSwiGLU",
    "TritonGeGLU",
    
    # Attention
    "flash_attention",
    "FlashAttention",
    "FlashAttentionConfig",
    "AttentionMaskType",
    "AttentionOutput",
    "MultiHeadFlashAttention",
    "is_flash_attn_available",
    "attention_softcapping_compiled",
    "slow_attention_softcapping",
    "scaled_dot_product_attention",
    "create_causal_mask",
    "create_sliding_window_causal_mask",
    "FlexAttentionConfig",
    "AttentionBackend",
    "BackendCapabilities",
    
    # RoPE
    "fast_rope_embedding",
    "Fast_RoPE_Embedding",
    "inplace_rope_embedding",
    "precompute_freqs_cis",
    "RoPEConfig",
    "RoPEScalingType",
    "RoPEFrequencyCache",
    
    # LoRA
    "matmul_lora",
    "LoRA_MLP",
    "LoRA_QKV",
    "apply_lora_mlp_swiglu",
    "apply_lora_qkv",
    "get_lora_parameters",
    "torch_amp_custom_fwd",
    "torch_amp_custom_bwd",
    
    # MoE
    "MoEKernelConfig",
    
    # FP8
    "row_quantize_fp8",
    "block_quantize_fp8",
    "block_dequantize_fp8",
    "fp8_matmul_block_scaled",
    "fp8_matmul_row_scaled",
    "FP8Linear",
    "FP8Format",
    "FP8Config",
    "FP8ScaleManager",
    
    # Distributed
    "DistributedKernels",
    "pinned_memory_transfer",
    "DistributedBackend",
    "DistributedConfig",
    "get_distributed_backend",
    "get_world_info",
]
