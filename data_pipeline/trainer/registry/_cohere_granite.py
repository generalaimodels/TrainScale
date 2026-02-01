# ════════════════════════════════════════════════════════════════════════════════
# Cohere & Granite Model Family Registry
# ════════════════════════════════════════════════════════════════════════════════
# Cohere Command R / Command R+ and IBM Granite models
# ════════════════════════════════════════════════════════════════════════════════

from . import ModelInfo, ModelMeta, QuantType, register_models_from_meta


# ═════════════════════════════════════════════════════════════════════════════════
# Cohere Command R Models
# ═════════════════════════════════════════════════════════════════════════════════

COMMAND_R_META = ModelMeta(
    org="CohereForAI",
    base_name="c4ai-command-r",
    model_version="1",
    model_info_cls=ModelInfo,
    model_sizes=["35", "104"],  # Command R and Command R+
    instruct_tags=[None, "v01"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    attention_class="CohereAttention",
    mlp_class="CohereMLP",
    layernorm_class="CohereLayerNorm",
    rope_class="CohereRotaryEmbedding",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Cohere Aya Models
# ═════════════════════════════════════════════════════════════════════════════════

AYA_META = ModelMeta(
    org="CohereForAI",
    base_name="aya",
    model_version="1",
    model_info_cls=ModelInfo,
    model_sizes=["8", "35"],  # Aya 8B, 35B
    instruct_tags=[None, "expanse"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    attention_class="AyaAttention",
    mlp_class="AyaMLP",
    layernorm_class="AyaRMSNorm",
)

# ═════════════════════════════════════════════════════════════════════════════════
# IBM Granite Models
# ═════════════════════════════════════════════════════════════════════════════════

GRANITE_META = ModelMeta(
    org="ibm-granite",
    base_name="granite",
    model_version="3.0",
    model_info_cls=ModelInfo,
    model_sizes=["1", "3", "8", "20"],
    instruct_tags=[None, "instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    attention_class="GraniteAttention",
    mlp_class="GraniteMLP",
    layernorm_class="GraniteRMSNorm",
    rope_class="GraniteRotaryEmbedding",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Granite Code Models
# ═════════════════════════════════════════════════════════════════════════════════

GRANITE_CODE_META = ModelMeta(
    org="ibm-granite",
    base_name="granite-code",
    model_version="3.0",
    model_info_cls=ModelInfo,
    model_sizes=["3", "8", "20", "34"],
    instruct_tags=[None, "instruct", "base"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    attention_class="GraniteCodeAttention",
    mlp_class="GraniteCodeMLP",
    layernorm_class="GraniteCodeRMSNorm",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Register All Cohere & Granite Models
# ═════════════════════════════════════════════════════════════════════════════════

def register_cohere_granite_models():
    """Register all Cohere and Granite model variants."""
    register_models_from_meta(COMMAND_R_META, include_original=True)
    register_models_from_meta(AYA_META, include_original=True)
    register_models_from_meta(GRANITE_META, include_original=True)
    register_models_from_meta(GRANITE_CODE_META, include_original=True)


# Auto-register on import
register_cohere_granite_models()
