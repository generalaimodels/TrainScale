# ════════════════════════════════════════════════════════════════════════════════
# Mistral Model Registration
# ════════════════════════════════════════════════════════════════════════════════

from . import (
    ModelInfo, ModelMeta, QuantType, QUANT_TAG_MAP,
    register_models_from_meta,
)

_IS_MISTRAL_REGISTERED = False


class MistralModelInfo(ModelInfo):
    """Mistral-specific model info."""
    
    @classmethod
    def construct_model_name(
        cls,
        base_name: str,
        version: str,
        size: str,
        quant_type: QuantType = QuantType.NONE,
        instruct_tag: str = None,
    ) -> str:
        # Mistral format: Mistral-7B-v0.3
        if version:
            key = f"{base_name}-{size}B-v{version}"
        else:
            key = f"{base_name}-{size}B"
        if instruct_tag:
            key = f"{key}-{instruct_tag}"
        if quant_type != QuantType.NONE:
            tag = QUANT_TAG_MAP.get(quant_type)
            if tag:
                key = f"{key}-{tag}"
        return key


# Mistral 7B
MistralMeta_7B = ModelMeta(
    org="mistralai",
    base_name="Mistral",
    model_version="0.3",
    model_info_cls=MistralModelInfo,
    model_sizes=["7"],
    instruct_tags=[None, "Instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    is_multimodal=False,
    attention_class="MistralAttention",
    mlp_class="MistralMLP",
    layernorm_class="MistralRMSNorm",
    rope_class="MistralRotaryEmbedding",
)

# Mistral Small (latest efficient)
MistralMeta_Small = ModelMeta(
    org="mistralai",
    base_name="Mistral-Small",
    model_version="24.09",
    model_info_cls=MistralModelInfo,
    model_sizes=["22"],
    instruct_tags=[None, "Instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    is_multimodal=False,
    attention_class="MistralAttention",
    mlp_class="MistralMLP",
    layernorm_class="MistralRMSNorm",
    rope_class="MistralRotaryEmbedding",
)


def register_mistral_models(include_original: bool = False) -> None:
    """Register all Mistral model variants."""
    global _IS_MISTRAL_REGISTERED
    if _IS_MISTRAL_REGISTERED:
        return
    
    register_models_from_meta(MistralMeta_7B, include_original)
    register_models_from_meta(MistralMeta_Small, include_original)
    
    _IS_MISTRAL_REGISTERED = True
    print("✓ Registered Mistral 7B and Small models")
