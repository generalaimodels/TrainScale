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


# ════════════════════════════════════════════════════════════════════════════════
# Hardware Capability Detection
# ════════════════════════════════════════════════════════════════════════════════

from typing import Dict
import torch

def get_kernel_capabilities() -> Dict[str, bool]:
    """
    Return dict of available kernel capabilities for current hardware.
    
    O(1) complexity with cached device queries.
    
    Returns:
        Dictionary mapping capability names to availability booleans:
        - triton: Triton JIT compiler available
        - flash_attn: Flash Attention 2 library available
        - tma: Tensor Memory Accelerator (Hopper+)
        - wgmma: Warpgroup MMA (Hopper+)
        - fp8: FP8 hardware support (SM90+)
        - cuda: CUDA runtime available
        - bf16: BF16 hardware support (Ampere+)
    """
    cuda_available = torch.cuda.is_available()
    
    # Default capabilities when no CUDA
    if not cuda_available:
        return {
            "triton": False,
            "flash_attn": False,
            "tma": False,
            "wgmma": False,
            "fp8": False,
            "cuda": False,
            "bf16": False,
        }
    
    # Get compute capability
    major, minor = torch.cuda.get_device_capability()
    
    return {
        "triton": is_triton_available(),
        "flash_attn": is_flash_attn_available(),
        "tma": supports_tma(),
        "wgmma": supports_wgmma(),
        "fp8": major >= 9,  # SM90+ (Hopper)
        "cuda": True,
        "bf16": major >= 8,  # SM80+ (Ampere)
    }


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
    "get_kernel_capabilities",
    
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

