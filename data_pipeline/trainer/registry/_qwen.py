# ════════════════════════════════════════════════════════════════════════════════
# Qwen Model Registration
# ════════════════════════════════════════════════════════════════════════════════

from . import (
    ModelInfo, ModelMeta, QuantType, QUANT_TAG_MAP,
    register_models_from_meta,
)

_IS_QWEN_REGISTERED = False


class QwenModelInfo(ModelInfo):
    """Qwen-specific model info."""
    
    @classmethod
    def construct_model_name(
        cls,
        base_name: str,
        version: str,
        size: str,
        quant_type: QuantType = QuantType.NONE,
        instruct_tag: str = None,
    ) -> str:
        # Qwen uses format: Qwen2.5-7B-Instruct
        key = f"{base_name}{version}-{size}B"
        if instruct_tag:
            key = f"{key}-{instruct_tag}"
        if quant_type != QuantType.NONE:
            tag = QUANT_TAG_MAP.get(quant_type)
            if tag:
                key = f"{key}-{tag}"
        return key


# Qwen 2.5
QwenMeta_2_5 = ModelMeta(
    org="Qwen",
    base_name="Qwen",
    model_version="2.5",
    model_info_cls=QwenModelInfo,
    model_sizes=["0.5", "1.5", "3", "7", "14", "32", "72"],
    instruct_tags=[None, "Instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    is_multimodal=False,
    attention_class="Qwen2Attention",
    mlp_class="Qwen2MLP",
    layernorm_class="Qwen2RMSNorm",
    rope_class="Qwen2RotaryEmbedding",
)

# Qwen 2.5 Coder
QwenMeta_2_5_Coder = ModelMeta(
    org="Qwen",
    base_name="Qwen",
    model_version="2.5-Coder",
    model_info_cls=QwenModelInfo,
    model_sizes=["1.5", "7", "14", "32"],
    instruct_tags=[None, "Instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    is_multimodal=False,
    attention_class="Qwen2Attention",
    mlp_class="Qwen2MLP",
    layernorm_class="Qwen2RMSNorm",
    rope_class="Qwen2RotaryEmbedding",
)

# QwQ (Reasoning)
QwQMeta = ModelMeta(
    org="Qwen",
    base_name="QwQ",
    model_version="",
    model_info_cls=QwenModelInfo,
    model_sizes=["32"],
    instruct_tags=["Preview"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT],
    is_multimodal=False,
    attention_class="Qwen2Attention",
    mlp_class="Qwen2MLP",
    layernorm_class="Qwen2RMSNorm",
    rope_class="Qwen2RotaryEmbedding",
)


def register_qwen_models(include_original: bool = False) -> None:
    """Register all Qwen model variants."""
    global _IS_QWEN_REGISTERED
    if _IS_QWEN_REGISTERED:
        return
    
    register_models_from_meta(QwenMeta_2_5, include_original)
    register_models_from_meta(QwenMeta_2_5_Coder, include_original)
    register_models_from_meta(QwQMeta, include_original)
    
    _IS_QWEN_REGISTERED = True
    print("✓ Registered Qwen 2.5, Coder, and QwQ models")
