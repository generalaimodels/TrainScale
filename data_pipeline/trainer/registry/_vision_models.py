# ════════════════════════════════════════════════════════════════════════════════
# Vision & Multimodal Model Family Registry
# ════════════════════════════════════════════════════════════════════════════════
# Vision-Language models: LLaVA, Qwen-VL, Pixtral, InternVL
# ════════════════════════════════════════════════════════════════════════════════

from . import ModelInfo, ModelMeta, QuantType, register_models_from_meta


# ═════════════════════════════════════════════════════════════════════════════════
# LLaVA Models
# ═════════════════════════════════════════════════════════════════════════════════

LLAVA_META = ModelMeta(
    org="llava-hf",
    base_name="llava-1.5",
    model_version="1.5",
    model_info_cls=ModelInfo,
    model_sizes=["7", "13"],
    instruct_tags=[None, "hf"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT],
    is_multimodal=True,
    attention_class="LlavaAttention",
    mlp_class="LlavaMLP",
    layernorm_class="LlavaRMSNorm",
)

LLAVA_NEXT_META = ModelMeta(
    org="llava-hf",
    base_name="llava-v1.6",
    model_version="1.6",
    model_info_cls=ModelInfo,
    model_sizes=["7", "13", "34"],
    instruct_tags=[None, "vicuna", "mistral"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    is_multimodal=True,
    attention_class="LlavaNextAttention",
    mlp_class="LlavaNextMLP",
    layernorm_class="LlavaNextRMSNorm",
)

LLAVA_ONEVISION_META = ModelMeta(
    org="llava-hf",
    base_name="llava-onevision",
    model_version="0.5",
    model_info_cls=ModelInfo,
    model_sizes=["0.5", "7", "72"],
    instruct_tags=[None, "qwen2", "si"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    is_multimodal=True,
    attention_class="LlavaOnevisionAttention",
    mlp_class="LlavaOnevisionMLP",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Qwen-VL Models
# ═════════════════════════════════════════════════════════════════════════════════

QWEN_VL_META = ModelMeta(
    org="Qwen",
    base_name="Qwen2-VL",
    model_version="2",
    model_info_cls=ModelInfo,
    model_sizes=["2", "7", "72"],
    instruct_tags=[None, "Instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    is_multimodal=True,
    attention_class="Qwen2VLAttention",
    mlp_class="Qwen2VLMLP",
    layernorm_class="Qwen2VLRMSNorm",
)

QWEN25_VL_META = ModelMeta(
    org="Qwen",
    base_name="Qwen2.5-VL",
    model_version="2.5",
    model_info_cls=ModelInfo,
    model_sizes=["3", "7", "32", "72"],
    instruct_tags=[None, "Instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    is_multimodal=True,
    attention_class="Qwen25VLAttention",
    mlp_class="Qwen25VLMLP",
    layernorm_class="Qwen25VLRMSNorm",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Pixtral (Mistral Vision)
# ═════════════════════════════════════════════════════════════════════════════════

PIXTRAL_META = ModelMeta(
    org="mistralai",
    base_name="Pixtral",
    model_version="12B",
    model_info_cls=ModelInfo,
    model_sizes=["12"],
    instruct_tags=[None, "2409"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    is_multimodal=True,
    attention_class="PixtralAttention",
    mlp_class="PixtralMLP",
    layernorm_class="PixtralRMSNorm",
)

# ═════════════════════════════════════════════════════════════════════════════════
# InternVL Models
# ═════════════════════════════════════════════════════════════════════════════════

INTERNVL_META = ModelMeta(
    org="OpenGVLab",
    base_name="InternVL2",
    model_version="2",
    model_info_cls=ModelInfo,
    model_sizes=["1", "2", "8", "26", "40", "76"],
    instruct_tags=[None, "Instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC],
    is_multimodal=True,
    attention_class="InternVLAttention",
    mlp_class="InternVLMLP",
    layernorm_class="InternVLRMSNorm",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Llama 3.2 Vision
# ═════════════════════════════════════════════════════════════════════════════════

LLAMA32_VISION_META = ModelMeta(
    org="meta-llama",
    base_name="Llama-3.2-Vision",
    model_version="3.2",
    model_info_cls=ModelInfo,
    model_sizes=["11", "90"],
    instruct_tags=[None, "Instruct"],
    quant_types=[QuantType.NONE, QuantType.BNB_4BIT, QuantType.DYNAMIC, QuantType.FP8],
    is_multimodal=True,
    attention_class="Llama32VisionAttention",
    mlp_class="Llama32VisionMLP",
    layernorm_class="LlamaRMSNorm",
)

# ═════════════════════════════════════════════════════════════════════════════════
# Register All Vision Models
# ═════════════════════════════════════════════════════════════════════════════════

def register_vision_models():
    """Register all vision model variants."""
    register_models_from_meta(LLAVA_META, include_original=True)
    register_models_from_meta(LLAVA_NEXT_META, include_original=True)
    register_models_from_meta(LLAVA_ONEVISION_META, include_original=True)
    register_models_from_meta(QWEN_VL_META, include_original=True)
    register_models_from_meta(QWEN25_VL_META, include_original=True)
    register_models_from_meta(PIXTRAL_META, include_original=True)
    register_models_from_meta(INTERNVL_META, include_original=True)
    register_models_from_meta(LLAMA32_VISION_META, include_original=True)


# Auto-register on import
register_vision_models()
