# ════════════════════════════════════════════════════════════════════════════════
# Gemma Model Registration
# ════════════════════════════════════════════════════════════════════════════════

from . import (
    ModelInfo, ModelMeta, QuantType, QUANT_TAG_MAP,
    register_models_from_meta,
)

_IS_GEMMA_REGISTERED = False


class GemmaModelInfo(ModelInfo):
    """Gemma-specific model info with 256K vocab support."""
    
    @classmethod
    def construct_model_name(
        cls,
        base_name: str,
        version: str,
        size: str,
        quant_type: QuantType = QuantType.NONE,
        instruct_tag: str = None,
    ) -> str:
        # Gemma uses format: gemma-2-9b-it
        key = f"{base_name.lower()}-{version}-{size}b"
        if instruct_tag:
            key = f"{key}-{instruct_tag.lower()}"
        if quant_type != QuantType.NONE:
            tag = QUANT_TAG_MAP.get(quant_type)
            if tag:
                key = f"{key}-{tag}"
        return key


# Gemma 2 (256K vocab - requires chunked CE)
GemmaMeta_2 = ModelMeta(
    org="google",
    base_name="gemma",
    model_version="2",
    model_info_cls=GemmaModelInfo,
    model_sizes=["2", "9", "27"],
    instruct_tags=[None, "it"],  # "it" = instruction-tuned
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    is_multimodal=False,
    attention_class="GemmaAttention",
    mlp_class="GemmaMLP",
    layernorm_class="GemmaRMSNorm",
    rope_class="GemmaRotaryEmbedding",
)

# Gemma 3 (latest)
GemmaMeta_3 = ModelMeta(
    org="google",
    base_name="gemma",
    model_version="3",
    model_info_cls=GemmaModelInfo,
    model_sizes=["1", "4", "12", "27"],
    instruct_tags=[None, "it"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    is_multimodal=False,
    attention_class="Gemma3Attention",
    mlp_class="Gemma3MLP",
    layernorm_class="Gemma3RMSNorm",
    rope_class="Gemma3RotaryEmbedding",
)


def register_gemma_models(include_original: bool = False) -> None:
    """Register all Gemma model variants."""
    global _IS_GEMMA_REGISTERED
    if _IS_GEMMA_REGISTERED:
        return
    
    register_models_from_meta(GemmaMeta_2, include_original)
    register_models_from_meta(GemmaMeta_3, include_original)
    
    _IS_GEMMA_REGISTERED = True
    print("✓ Registered Gemma 2 and 3 models (256K vocab support)")
