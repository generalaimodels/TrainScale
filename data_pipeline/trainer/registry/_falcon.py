# ════════════════════════════════════════════════════════════════════════════════
# Falcon Model Family Registry
# ════════════════════════════════════════════════════════════════════════════════
# TII Falcon models including Falcon-H1 Mamba hybrid
# ════════════════════════════════════════════════════════════════════════════════

from . import ModelInfo, ModelMeta, QuantType, register_models_from_meta


# ═════════════════════════════════════════════════════════════════════════════════
# Falcon-1 Models
# ═════════════════════════════════════════════════════════════════════════════════

FALCON_1_META = ModelMeta(
    org="tiiuae",
    base_name="falcon",
    model_version="1",
    model_info_cls=ModelInfo,
    model_sizes=["7", "40", "180"],
    instruct_tags=[None, "instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    attention_class="FalconAttention",
    mlp_class="FalconMLP",
    layernorm_class="FalconLayerNorm",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Falcon-2 Models
# ═════════════════════════════════════════════════════════════════════════════════

FALCON_2_META = ModelMeta(
    org="tiiuae",
    base_name="falcon2",
    model_version="2",
    model_info_cls=ModelInfo,
    model_sizes=["11"],
    instruct_tags=[None, "instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    attention_class="Falcon2Attention",
    mlp_class="Falcon2MLP",
    layernorm_class="Falcon2RMSNorm",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Falcon-H1 (Mamba Hybrid)
# ═════════════════════════════════════════════════════════════════════════════════

FALCON_H1_META = ModelMeta(
    org="tiiuae",
    base_name="Falcon-H1",
    model_version="H1",
    model_info_cls=ModelInfo,
    model_sizes=["0.5", "1.5", "3", "7", "34"],
    instruct_tags=[None, "Instruct", "Deep-Instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    attention_class="FalconH1Attention",
    mlp_class="FalconH1MLP",
    layernorm_class="FalconH1RMSNorm",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Register All Falcon Models
# ═════════════════════════════════════════════════════════════════════════════════

def register_falcon_models():
    """Register all Falcon model variants."""
    register_models_from_meta(FALCON_1_META, include_original=True)
    register_models_from_meta(FALCON_2_META, include_original=True)
    register_models_from_meta(FALCON_H1_META, include_original=True)


# Auto-register on import
register_falcon_models()
