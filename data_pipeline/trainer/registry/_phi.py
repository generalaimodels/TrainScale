# ════════════════════════════════════════════════════════════════════════════════
# Phi Model Family Registry
# ════════════════════════════════════════════════════════════════════════════════
# Microsoft Phi models - all sizes and variants
# ════════════════════════════════════════════════════════════════════════════════

from . import ModelInfo, ModelMeta, QuantType, register_models_from_meta


# ═════════════════════════════════════════════════════════════════════════════════
# Phi-3 Models
# ═════════════════════════════════════════════════════════════════════════════════

PHI_3_META = ModelMeta(
    org="microsoft",
    base_name="Phi-3",
    model_version="3",
    model_info_cls=ModelInfo,
    model_sizes=["3.8", "7", "14"],
    instruct_tags=[None, "Instruct", "128k-Instruct", "4k-Instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    attention_class="Phi3Attention",
    mlp_class="Phi3MLP",
    layernorm_class="Phi3RMSNorm",
    rope_class="Phi3RotaryEmbedding",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Phi-4 Models
# ═════════════════════════════════════════════════════════════════════════════════

PHI_4_META = ModelMeta(
    org="microsoft",
    base_name="Phi-4",
    model_version="4",
    model_info_cls=ModelInfo,
    model_sizes=["14"],
    instruct_tags=[None, "Instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    attention_class="Phi4Attention",
    mlp_class="Phi4MLP",
    layernorm_class="Phi4RMSNorm",
    rope_class="Phi4RotaryEmbedding",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Register All Phi Models
# ═════════════════════════════════════════════════════════════════════════════════

def register_phi_models():
    """Register all Phi model variants."""
    register_models_from_meta(PHI_3_META, include_original=True)
    register_models_from_meta(PHI_4_META, include_original=True)


# Auto-register on import
register_phi_models()
