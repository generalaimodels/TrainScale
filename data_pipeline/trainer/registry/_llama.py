# ════════════════════════════════════════════════════════════════════════════════
# Llama Model Registration
# ════════════════════════════════════════════════════════════════════════════════

from . import (
    ModelInfo, ModelMeta, QuantType,
    register_models_from_meta,
)

_IS_LLAMA_REGISTERED = False


class LlamaModelInfo(ModelInfo):
    """Llama-specific model info."""
    
    @classmethod
    def construct_model_name(
        cls,
        base_name: str,
        version: str,
        size: str,
        quant_type: QuantType = QuantType.NONE,
        instruct_tag: str = None,
    ) -> str:
        key = f"{base_name}-{version}-{size}B"
        if instruct_tag:
            key = f"{key}-{instruct_tag}"
        if quant_type != QuantType.NONE:
            from . import QUANT_TAG_MAP
            tag = QUANT_TAG_MAP.get(quant_type)
            if tag:
                key = f"{key}-{tag}"
        return key


# Llama 3.1
LlamaMeta_3_1 = ModelMeta(
    org="meta-llama",
    base_name="Llama",
    model_version="3.1",
    model_info_cls=LlamaModelInfo,
    model_sizes=["8", "70", "405"],
    instruct_tags=[None, "Instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    is_multimodal=False,
    attention_class="LlamaAttention",
    mlp_class="LlamaMLP",
    layernorm_class="LlamaRMSNorm",
    rope_class="LlamaRotaryEmbedding",
)

# Llama 3.2
LlamaMeta_3_2 = ModelMeta(
    org="meta-llama",
    base_name="Llama",
    model_version="3.2",
    model_info_cls=LlamaModelInfo,
    model_sizes=["1", "3"],
    instruct_tags=[None, "Instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.GGUF],
    is_multimodal=False,
    attention_class="LlamaAttention",
    mlp_class="LlamaMLP",
    layernorm_class="LlamaRMSNorm",
    rope_class="LlamaRotaryEmbedding",
)

# Llama 3.2 Vision
LlamaMeta_3_2_Vision = ModelMeta(
    org="meta-llama",
    base_name="Llama",
    model_version="3.2",
    model_info_cls=LlamaModelInfo,
    model_sizes=["11", "90"],
    instruct_tags=["Vision", "Vision-Instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT],
    is_multimodal=True,
    attention_class="LlamaAttention",
    mlp_class="LlamaMLP",
    layernorm_class="LlamaRMSNorm",
    rope_class="LlamaRotaryEmbedding",
)


def register_llama_models(include_original: bool = False) -> None:
    """Register all Llama model variants."""
    global _IS_LLAMA_REGISTERED
    if _IS_LLAMA_REGISTERED:
        return
    
    register_models_from_meta(LlamaMeta_3_1, include_original)
    register_models_from_meta(LlamaMeta_3_2, include_original)
    register_models_from_meta(LlamaMeta_3_2_Vision, include_original)
    
    _IS_LLAMA_REGISTERED = True
    print("✓ Registered Llama 3.1, 3.2, and 3.2-Vision models")
