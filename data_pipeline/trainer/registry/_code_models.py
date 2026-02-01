# ════════════════════════════════════════════════════════════════════════════════
# Code Models Registry (StarCoder, CodeLlama, DeepSeek-Coder)
# ════════════════════════════════════════════════════════════════════════════════
# All major code generation models
# ════════════════════════════════════════════════════════════════════════════════

from . import ModelInfo, ModelMeta, QuantType, register_models_from_meta


# ═════════════════════════════════════════════════════════════════════════════════
# StarCoder Models
# ═════════════════════════════════════════════════════════════════════════════════

STARCODER_META = ModelMeta(
    org="bigcode",
    base_name="starcoder",
    model_version="1",
    model_info_cls=ModelInfo,
    model_sizes=["1", "3", "7", "15"],
    instruct_tags=[None, "instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    attention_class="StarcoderAttention",
    mlp_class="StarcoderMLP",
    layernorm_class="StarcoderLayerNorm",
)

STARCODER2_META = ModelMeta(
    org="bigcode",
    base_name="starcoder2",
    model_version="2",
    model_info_cls=ModelInfo,
    model_sizes=["3", "7", "15"],
    instruct_tags=[None, "instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    attention_class="Starcoder2Attention",
    mlp_class="Starcoder2MLP",
    layernorm_class="Starcoder2LayerNorm",
)

# ═════════════════════════════════════════════════════════════════════════════════
# CodeLlama Models
# ═════════════════════════════════════════════════════════════════════════════════

CODELLAMA_META = ModelMeta(
    org="meta-llama",
    base_name="CodeLlama",
    model_version="1",
    model_info_cls=ModelInfo,
    model_sizes=["7", "13", "34", "70"],
    instruct_tags=[None, "Instruct", "Python"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    attention_class="LlamaAttention",
    mlp_class="LlamaMLP",
    layernorm_class="LlamaRMSNorm",
    rope_class="LlamaRotaryEmbedding",
)

# ═════════════════════════════════════════════════════════════════════════════════
# DeepSeek-Coder Models
# ═════════════════════════════════════════════════════════════════════════════════

DEEPSEEK_CODER_META = ModelMeta(
    org="deepseek-ai",
    base_name="deepseek-coder",
    model_version="1.3",
    model_info_cls=ModelInfo,
    model_sizes=["1.3", "6.7", "33"],
    instruct_tags=[None, "instruct", "base"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    attention_class="DeepseekCoderAttention",
    mlp_class="DeepseekCoderMLP",
    layernorm_class="DeepseekCoderRMSNorm",
)

DEEPSEEK_CODER_V2_META = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek-Coder-V2",
    model_version="2",
    model_info_cls=ModelInfo,
    model_sizes=["16", "236"],
    instruct_tags=[None, "Instruct", "Lite-Instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8, QuantType.BF16],
    attention_class="DeepseekV2Attention",
    mlp_class="DeepseekV2MoE",
    layernorm_class="DeepseekV2RMSNorm",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Qwen-Coder Models
# ═════════════════════════════════════════════════════════════════════════════════

QWEN_CODER_META = ModelMeta(
    org="Qwen",
    base_name="Qwen2.5-Coder",
    model_version="2.5",
    model_info_cls=ModelInfo,
    model_sizes=["0.5", "1.5", "3", "7", "14", "32"],
    instruct_tags=[None, "Instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    attention_class="Qwen2Attention",
    mlp_class="Qwen2MLP",
    layernorm_class="Qwen2RMSNorm",
    rope_class="Qwen2RotaryEmbedding",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Register All Code Models
# ═════════════════════════════════════════════════════════════════════════════════

def register_code_models():
    """Register all code model variants."""
    register_models_from_meta(STARCODER_META, include_original=True)
    register_models_from_meta(STARCODER2_META, include_original=True)
    register_models_from_meta(CODELLAMA_META, include_original=True)
    register_models_from_meta(DEEPSEEK_CODER_META, include_original=True)
    register_models_from_meta(DEEPSEEK_CODER_V2_META, include_original=True)
    register_models_from_meta(QWEN_CODER_META, include_original=True)


# Auto-register on import
register_code_models()
