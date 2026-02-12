# ════════════════════════════════════════════════════════════════════════════════
# Yi Model Family Registry
# ════════════════════════════════════════════════════════════════════════════════
# 01.AI Yi models - all sizes and variants
# ════════════════════════════════════════════════════════════════════════════════

from . import ModelInfo, ModelMeta, QuantType, register_models_from_meta


# ═════════════════════════════════════════════════════════════════════════════════
# Yi-1 Models
# ═════════════════════════════════════════════════════════════════════════════════

YI_1_META = ModelMeta(
    org="01-ai",
    base_name="Yi",
    model_version="1",
    model_info_cls=ModelInfo,
    model_sizes=["6", "9", "34"],
    instruct_tags=[None, "Chat", "200K"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    attention_class="YiAttention",
    mlp_class="YiMLP",
    layernorm_class="YiRMSNorm",
    rope_class="YiRotaryEmbedding",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Yi-1.5 Models
# ═════════════════════════════════════════════════════════════════════════════════

YI_15_META = ModelMeta(
    org="01-ai",
    base_name="Yi-1.5",
    model_version="1.5",
    model_info_cls=ModelInfo,
    model_sizes=["6", "9", "34"],
    instruct_tags=[None, "Chat", "Chat-16K"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    attention_class="Yi15Attention",
    mlp_class="Yi15MLP",
    layernorm_class="Yi15RMSNorm",
    rope_class="Yi15RotaryEmbedding",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Yi-Coder Models
# ═════════════════════════════════════════════════════════════════════════════════

YI_CODER_META = ModelMeta(
    org="01-ai",
    base_name="Yi-Coder",
    model_version="1.5",
    model_info_cls=ModelInfo,
    model_sizes=["1.5", "9"],
    instruct_tags=[None, "Chat"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    attention_class="YiCoderAttention",
    mlp_class="YiCoderMLP",
    layernorm_class="YiCoderRMSNorm",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Register All Yi Models
# ═════════════════════════════════════════════════════════════════════════════════

def register_yi_models():
    """Register all Yi model variants."""
    register_models_from_meta(YI_1_META, include_original=True)
    register_models_from_meta(YI_15_META, include_original=True)
    register_models_from_meta(YI_CODER_META, include_original=True)


# Auto-register on import
register_yi_models()
