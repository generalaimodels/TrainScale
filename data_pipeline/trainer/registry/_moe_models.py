# ════════════════════════════════════════════════════════════════════════════════
# Mixtral & MoE Model Family Registry
# ════════════════════════════════════════════════════════════════════════════════
# Mixture of Experts models: Mixtral, DBRX, Grok
# ════════════════════════════════════════════════════════════════════════════════

from . import ModelInfo, ModelMeta, QuantType, register_models_from_meta


# ═════════════════════════════════════════════════════════════════════════════════
# Mixtral Models
# ═════════════════════════════════════════════════════════════════════════════════

MIXTRAL_8x7B_META = ModelMeta(
    org="mistralai",
    base_name="Mixtral-8x7B",
    model_version="0.1",
    model_info_cls=ModelInfo,
    model_sizes=["8x7"],  # MOE size format
    instruct_tags=[None, "Instruct-v0.1"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    attention_class="MixtralAttention",
    mlp_class="MixtralSparseMoeBlock",
    layernorm_class="MixtralRMSNorm",
    rope_class="MixtralRotaryEmbedding",
)

MIXTRAL_8x22B_META = ModelMeta(
    org="mistralai",
    base_name="Mixtral-8x22B",
    model_version="0.1",
    model_info_cls=ModelInfo,
    model_sizes=["8x22"],  # MOE size format
    instruct_tags=[None, "Instruct-v0.1"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    attention_class="MixtralAttention",
    mlp_class="MixtralSparseMoeBlock",
    layernorm_class="MixtralRMSNorm",
    rope_class="MixtralRotaryEmbedding",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Qwen MoE Models
# ═════════════════════════════════════════════════════════════════════════════════

QWEN_MOE_META = ModelMeta(
    org="Qwen",
    base_name="Qwen1.5-MoE",
    model_version="1.5",
    model_info_cls=ModelInfo,
    model_sizes=["A2.7"],  # 2.7B active params
    instruct_tags=[None, "Chat"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    attention_class="Qwen2MoeAttention",
    mlp_class="Qwen2MoeSparseMoeBlock",
    layernorm_class="Qwen2MoeRMSNorm",
)

QWEN3_MOE_META = ModelMeta(
    org="Qwen",
    base_name="Qwen3-MoE",
    model_version="3",
    model_info_cls=ModelInfo,
    model_sizes=["A3", "A22"],  # Active params
    instruct_tags=[None, "Instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    attention_class="Qwen3MoeAttention",
    mlp_class="Qwen3MoeSparseMoeBlock",
    layernorm_class="Qwen3MoeRMSNorm",
)

# ═════════════════════════════════════════════════════════════════════════════════
# DBRX (Databricks)
# ═════════════════════════════════════════════════════════════════════════════════

DBRX_META = ModelMeta(
    org="databricks",
    base_name="dbrx",
    model_version="1",
    model_info_cls=ModelInfo,
    model_sizes=["132"],
    instruct_tags=[None, "instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    attention_class="DbrxAttention",
    mlp_class="DbrxFFN",
    layernorm_class="DbrxLayerNorm",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Grok (xAI)
# ═════════════════════════════════════════════════════════════════════════════════

GROK_META = ModelMeta(
    org="xai-org",
    base_name="grok",
    model_version="1",
    model_info_cls=ModelInfo,
    model_sizes=["314"],
    instruct_tags=[None],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    attention_class="GrokAttention",
    mlp_class="GrokMoE",
    layernorm_class="GrokRMSNorm",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Register All MoE Models
# ═════════════════════════════════════════════════════════════════════════════════

def register_moe_models():
    """Register all MoE model variants."""
    register_models_from_meta(MIXTRAL_8x7B_META, include_original=True)
    register_models_from_meta(MIXTRAL_8x22B_META, include_original=True)
    register_models_from_meta(QWEN_MOE_META, include_original=True)
    register_models_from_meta(QWEN3_MOE_META, include_original=True)
    register_models_from_meta(DBRX_META, include_original=True)
    register_models_from_meta(GROK_META, include_original=True)


# Auto-register on import
register_moe_models()
