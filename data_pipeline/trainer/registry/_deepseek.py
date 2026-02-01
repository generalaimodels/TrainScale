# ════════════════════════════════════════════════════════════════════════════════
# DeepSeek Model Registration
# ════════════════════════════════════════════════════════════════════════════════

from . import (
    ModelInfo, ModelMeta, QuantType, QUANT_TAG_MAP,
    register_models_from_meta,
)

_IS_DEEPSEEK_REGISTERED = False


class DeepSeekModelInfo(ModelInfo):
    """DeepSeek-specific model info."""
    
    @classmethod
    def construct_model_name(
        cls,
        base_name: str,
        version: str,
        size: str,
        quant_type: QuantType = QuantType.NONE,
        instruct_tag: str = None,
    ) -> str:
        # DeepSeek format: DeepSeek-V3, DeepSeek-R1
        key = f"{base_name}-{version}"
        if size != "default":
            key = f"{key}-{size}B"
        if instruct_tag:
            key = f"{key}-{instruct_tag}"
        if quant_type != QuantType.NONE:
            tag = QUANT_TAG_MAP.get(quant_type)
            if tag:
                key = f"{key}-{tag}"
        return key


# DeepSeek V3 (MoE architecture)
DeepSeekMeta_V3 = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek",
    model_version="V3",
    model_info_cls=DeepSeekModelInfo,
    model_sizes=["default"],  # 671B parameters
    instruct_tags=[None, "Chat"],
    quant_types=[QuantType.NONE, QuantType.BF16, QuantType.BNB_4BIT],
    is_multimodal=False,
    attention_class="DeepseekV3Attention",
    mlp_class="DeepseekV3MoE",
    layernorm_class="DeepseekV3RMSNorm",
    rope_class="DeepseekV3RotaryEmbedding",
)

# DeepSeek R1 (Reasoning)
DeepSeekMeta_R1 = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek",
    model_version="R1",
    model_info_cls=DeepSeekModelInfo,
    model_sizes=["default", "7", "14", "32", "70"],
    instruct_tags=[None, "Distill-Llama", "Distill-Qwen"],
    quant_types=[QuantType.NONE, QuantType.BF16, QuantType.BNB_4BIT],
    is_multimodal=False,
    attention_class="DeepseekV3Attention",
    mlp_class="DeepseekV3MoE",
    layernorm_class="DeepseekV3RMSNorm",
    rope_class="DeepseekV3RotaryEmbedding",
)


def register_deepseek_models(include_original: bool = False) -> None:
    """Register all DeepSeek model variants."""
    global _IS_DEEPSEEK_REGISTERED
    if _IS_DEEPSEEK_REGISTERED:
        return
    
    register_models_from_meta(DeepSeekMeta_V3, include_original)
    register_models_from_meta(DeepSeekMeta_R1, include_original)
    
    _IS_DEEPSEEK_REGISTERED = True
    print("✓ Registered DeepSeek V3 and R1 models")
