# ════════════════════════════════════════════════════════════════════════════════
# SOTA Unified Configuration - Above Unsloth Level
# ════════════════════════════════════════════════════════════════════════════════
# Complete end-to-end training configuration controlled via YAML.
#
# FEATURES EXCEEDING UNSLOTH:
# ═══════════════════════════════════════
# Training Modes:
#   - Full fine-tuning (16-bit, bf16, fp16)
#   - LoRA/QLoRA (4-bit NF4, 8-bit)
#   - FP8 training (H100/L40)
#   - Pretraining from scratch
#
# RL Algorithms (80% VRAM reduction):
#   - GRPO, GSPO, DrGRPO, DAPO
#   - PPO, DPO, ORPO, SimPO
#
# Export Formats:
#   - GGUF (q4_k_m, q5_k_m, q8_0, f16)
#   - vLLM, SGLang
#   - HuggingFace Hub
#   - Safetensors
#
# Hardware:
#   - NVIDIA (CUDA 7.0+: V100, T4, RTX 20/30/40, A100, H100)
#   - AMD (ROCm)
#   - Intel (XPU)
#   - Multi-GPU DDP/FSDP
#
# Kernels:
#   - Triton-fused operations
#   - Manual backprop for 0% accuracy loss
#   - Flash Attention, RoPE, RMSNorm
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
import torch


# ═════════════════════════════════════════════════════════════════════════════════
# Enumerations
# ═════════════════════════════════════════════════════════════════════════════════

class TrainingMode(str, Enum):
    """Training mode types."""
    FULL_FINETUNE = "full"
    LORA = "lora"
    QLORA = "qlora"
    PRETRAIN = "pretrain"
    RL = "rl"
    DISTILL = "distill"


class Precision(str, Enum):
    """Precision types for training."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    INT4 = "int4"


class RLAlgorithm(str, Enum):
    """RL algorithms."""
    GRPO = "grpo"
    GSPO = "gspo"
    DRGRPO = "drgrpo"
    DAPO = "dapo"
    PPO = "ppo"
    DPO = "dpo"
    ORPO = "orpo"
    SIMPO = "simpo"


class ExportFormat(str, Enum):
    """Export formats."""
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    GGUF_Q4 = "gguf_q4_k_m"
    GGUF_Q5 = "gguf_q5_k_m"
    GGUF_Q8 = "gguf_q8_0"
    GGUF_F16 = "gguf_f16"
    VLLM = "vllm"
    SGLANG = "sglang"


class OptimizerType(str, Enum):
    """Optimizer types."""
    ADAMW = "adamw"
    ADAM_8BIT = "adam8bit"
    LION = "lion"
    CAME = "came"
    SOPHIA = "sophia"
    PRODIGY = "prodigy"
    FUSED_ADAMW = "fused_adamw"
    LAMB = "lamb"
    ADAFACTOR = "adafactor"


class SchedulerType(str, Enum):
    """Scheduler types."""
    COSINE = "cosine"
    WSD = "wsd"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    ONECYCLE = "onecycle"
    INVERSE_SQRT = "inverse_sqrt"
    COSINE_RESTART = "cosine_restart"
    REX = "rex"
    CONSTANT = "constant"


class LossType(str, Enum):
    """Loss types."""
    CROSS_ENTROPY = "cross_entropy"
    CHUNKED_CE = "chunked_ce"
    FOCAL = "focal"
    DPO = "dpo"
    ORPO = "orpo"
    SIMPO = "simpo"
    DISTILLATION = "distillation"
    CONTRASTIVE = "contrastive"


class DeviceType(str, Enum):
    """Device types."""
    AUTO = "auto"
    CUDA = "cuda"
    ROCM = "rocm"
    XPU = "xpu"
    MPS = "mps"
    CPU = "cpu"


# ═════════════════════════════════════════════════════════════════════════════════
# Sub-Configurations
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Model loading configuration."""
    name_or_path: str = ""
    revision: str = "main"
    trust_remote_code: bool = False
    torch_dtype: str = "auto"
    low_cpu_mem_usage: bool = True
    use_flash_attention_2: bool = True
    attn_implementation: str = "flash_attention_2"
    max_position_embeddings: Optional[int] = None


@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""
    name_or_path: Optional[str] = None
    use_fast: bool = True
    padding_side: str = "right"
    truncation_side: str = "right"
    max_length: int = 4096
    add_bos_token: bool = False
    add_eos_token: bool = True
    add_pad_token: bool = True


@dataclass
class LoRAConfig:
    """LoRA/QLoRA configuration."""
    enabled: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    modules_to_save: List[str] = field(default_factory=lambda: ["embed_tokens", "lm_head"])
    bias: str = "none"
    use_rslora: bool = True
    use_dora: bool = False
    init_lora_weights: Union[bool, str] = True
    rank_pattern: Dict[str, int] = field(default_factory=dict)



@dataclass
class QuantizationConfig:
    """Quantization configuration."""
    enabled: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    load_in_fp8: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    llm_int8_threshold: float = 6.0


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    # 8-bit specific
    percentile_clipping: float = 100.0
    block_wise: bool = True
    # Lion specific
    lion_betas: tuple = (0.9, 0.99)


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""
    type: SchedulerType = SchedulerType.COSINE
    warmup_steps: int = 0
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.1
    # WSD specific
    stable_ratio: float = 0.8
    decay_type: str = "cosine"
    # Polynomial specific
    power: float = 1.0
    # OneCycle specific
    max_lr_ratio: float = 10.0


@dataclass
class LossConfig:
    """Loss configuration."""
    type: LossType = LossType.CROSS_ENTROPY
    ignore_index: int = -100
    label_smoothing: float = 0.0
    reduction: str = "mean"
    # Chunked CE specific
    chunk_size: int = 32768
    # Focal specific
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    # DPO specific
    dpo_beta: float = 0.1
    # Distillation specific
    temperature: float = 2.0
    alpha: float = 0.5


@dataclass
class RLConfig:
    """RL training configuration."""
    enabled: bool = False
    algorithm: RLAlgorithm = RLAlgorithm.GRPO
    num_generations: int = 4
    temperature: float = 0.7
    kl_coef: float = 0.01
    gamma: float = 1.0
    clip_range: float = 0.2
    reward_clip: Optional[float] = 10.0
    use_length_penalty: bool = False
    length_penalty_alpha: float = 1.0
    use_ref_model: bool = True
    ref_model_sync_steps: int = 100
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.9
    # GRPO specific
    group_size: int = 8
    # DAPO specific
    dapo_delta: float = 0.1


@dataclass
class ExportConfig:
    """Model export configuration."""
    enabled: bool = False
    output_dir: str = "./exported_model"
    format: ExportFormat = ExportFormat.SAFETENSORS
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    hub_private: bool = False
    merge_lora: bool = True
    save_merged_16bit: bool = True
    gguf_quantization: str = "q4_k_m"


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    enabled: bool = False
    backend: str = "nccl"
    strategy: str = "ddp"
    world_size: int = 1
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    fsdp_config: Dict[str, Any] = field(default_factory=dict)
    ddp_config: Dict[str, Any] = field(default_factory=dict)
    deepspeed_config: Optional[str] = None
    # Pipeline Parallelism (ZBPP)
    num_pipeline_stages: int = 4
    num_microbatches: int = 8
    pipeline_memory_limit_gb: float = 4.0
    
    # ═════════════════════════════════════════════════════════════════════════
    # SOTA DDP Settings (config-driven, not hard-coded)
    # ═════════════════════════════════════════════════════════════════════════
    ddp_gradient_as_bucket_view: bool = True     # Memory efficiency: ~10-15% savings
    ddp_static_graph: bool = True                # Enables gradient sync optimization
    ddp_find_unused_parameters: bool = False     # Disable for performance if no unused params
    ddp_broadcast_buffers: bool = True           # Sync batch norm stats
    
    # ═════════════════════════════════════════════════════════════════════════
    # SOTA FSDP2 Settings (config-driven, not hard-coded)
    # ═════════════════════════════════════════════════════════════════════════
    fsdp_limit_all_gathers: bool = True          # Memory optimization
    fsdp_use_orig_params: bool = False           # Auto-set True if compile_model=True
    fsdp_reduce_dtype: str = "fp32"              # Numerical stability: always reduce in FP32
    fsdp_sharding_strategy: str = "full"         # full, shard_grad_op, no_shard
    fsdp_forward_prefetch: bool = True           # Performance optimization
    fsdp_backward_prefetch: str = "backward_pre" # backward_pre or backward_post
    
    # ═════════════════════════════════════════════════════════════════════════
    # Context Parallel Settings (for long sequences)
    # ═════════════════════════════════════════════════════════════════════════
    context_parallel_size: int = 2


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    device: DeviceType = DeviceType.AUTO
    device_id: int = 0
    precision: Precision = Precision.BF16
    tf32: bool = True
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


@dataclass
class KernelConfig:
    """Triton kernel configuration."""
    use_triton: bool = True
    use_flash_attention: bool = True
    use_fused_rope: bool = True
    use_fused_rms_norm: bool = True
    use_fused_cross_entropy: bool = True
    use_fused_lora: bool = True
    use_moe_kernels: bool = True
    autotune: bool = True
    # Memory optimization
    activation_checkpointing: bool = True
    memory_efficient_attention: bool = True


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    dataset_name: str = ""
    dataset_config: Optional[str] = None
    streaming: bool = False
    max_samples: Optional[int] = None
    train_split: str = "train"
    eval_split: Optional[str] = "validation"
    text_column: str = "text"
    max_seq_length: int = 4096
    packing: bool = True
    packing_efficiency: float = 0.95
    shuffle: bool = True
    seed: int = 42


@dataclass
class TrainConfig:
    """Core training configuration."""
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    logging_steps: int = 10
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01


# ═════════════════════════════════════════════════════════════════════════════════
# Master Configuration
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class SOTAConfig:
    """
    SOTA Unified Training Configuration.
    
    Above-Unsloth-level configuration with:
    - Full-finetuning, LoRA, QLoRA, FP8
    - All RL algorithms (GRPO, GSPO, DrGRPO, DAPO, PPO, DPO, ORPO)
    - GGUF, vLLM, SGLang, HuggingFace export
    - Multi-GPU distributed training
    - Triton kernel integration
    - 0% accuracy loss
    """
    # Meta
    config_version: str = "2.0"
    config_type: str = "sota_training"
    
    # Mode
    training_mode: TrainingMode = TrainingMode.QLORA
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    kernels: KernelConfig = field(default_factory=KernelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    
    def __post_init__(self):
        """Validate and auto-configure based on mode."""
        self._validate()
        self._auto_configure()
    
    def _validate(self):
        """Validate configuration consistency."""
        # LoRA/QLoRA needs LoRA config
        if self.training_mode in (TrainingMode.LORA, TrainingMode.QLORA):
            if not self.lora.enabled:
                self.lora.enabled = True
        
        # QLoRA needs quantization
        if self.training_mode == TrainingMode.QLORA:
            if not self.quantization.enabled:
                self.quantization.enabled = True
                self.quantization.load_in_4bit = True
        
        # RL mode needs RL config
        if self.training_mode == TrainingMode.RL:
            if not self.rl.enabled:
                self.rl.enabled = True
    
    def _auto_configure(self):
        """Auto-configure optimal settings based on hardware."""
        # Detect hardware capabilities
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability()
            
            # H100/L40 - Enable FP8
            if cc[0] >= 9:
                if self.hardware.precision == Precision.BF16:
                    pass  # FP8 optional for H100
                # self.kernels.use_flash_attention = True # Don't override user config
            
            # A100 - BF16 optimal
            elif cc[0] >= 8:
                self.hardware.precision = Precision.BF16
                self.hardware.tf32 = True
                # self.kernels.use_flash_attention = True # Don't override user config
            
            # V100/T4 - FP16
            elif cc[0] >= 7:
                if self.hardware.precision == Precision.BF16:
                    self.hardware.precision = Precision.FP16
                    warnings.warn("BF16 not supported on this GPU, falling back to FP16")
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SOTAConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SOTAConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        # Parse training mode
        if "training_mode" in d:
            config.training_mode = TrainingMode(d["training_mode"])
        
        # Parse sub-configs
        if "model" in d:
            config.model = ModelConfig(**d["model"])
        if "tokenizer" in d:
            config.tokenizer = TokenizerConfig(**d["tokenizer"])
        if "lora" in d:
            config.lora = LoRAConfig(**d["lora"])
        if "quantization" in d:
            config.quantization = QuantizationConfig(**d["quantization"])
        if "optimizer" in d:
            opt_d = d["optimizer"].copy()
            if "type" in opt_d:
                opt_d["type"] = OptimizerType(opt_d["type"])
            config.optimizer = OptimizerConfig(**opt_d)
        if "scheduler" in d:
            sch_d = d["scheduler"].copy()
            if "type" in sch_d:
                sch_d["type"] = SchedulerType(sch_d["type"])
            config.scheduler = SchedulerConfig(**sch_d)
        if "loss" in d:
            loss_d = d["loss"].copy()
            if "type" in loss_d:
                loss_d["type"] = LossType(loss_d["type"])
            config.loss = LossConfig(**loss_d)
        if "rl" in d:
            rl_d = d["rl"].copy()
            if "algorithm" in rl_d:
                rl_d["algorithm"] = RLAlgorithm(rl_d["algorithm"])
            config.rl = RLConfig(**rl_d)
        if "export" in d:
            exp_d = d["export"].copy()
            if "format" in exp_d:
                exp_d["format"] = ExportFormat(exp_d["format"])
            config.export = ExportConfig(**exp_d)
        if "distributed" in d:
            config.distributed = DistributedConfig(**d["distributed"])
        if "hardware" in d:
            hw_d = d["hardware"].copy()
            if "device" in hw_d:
                hw_d["device"] = DeviceType(hw_d["device"])
            if "precision" in hw_d:
                hw_d["precision"] = Precision(hw_d["precision"])
            config.hardware = HardwareConfig(**hw_d)
        if "kernels" in d:
            config.kernels = KernelConfig(**d["kernels"])
        if "data" in d:
            config.data = DataConfig(**d["data"])
        if "training" in d:
            config.training = TrainConfig(**d["training"])
        
        config._validate()
        config._auto_configure()
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def save_yaml(self, path: Union[str, Path]):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        d = self.to_dict()
        
        # Convert enums to strings
        def _convert_enums(obj):
            if isinstance(obj, dict):
                return {k: _convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_convert_enums(v) for v in obj]
            elif isinstance(obj, Enum):
                return obj.value
            return obj
        
        d = _convert_enums(d)
        
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)
    
    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size."""
        return (
            self.training.per_device_train_batch_size *
            self.training.gradient_accumulation_steps *
            max(1, self.distributed.world_size)
        )
    
    @property
    def compute_dtype(self) -> torch.dtype:
        """Get compute dtype based on precision setting."""
        dtype_map = {
            Precision.FP32: torch.float32,
            Precision.FP16: torch.float16,
            Precision.BF16: torch.bfloat16,
            Precision.FP8_E4M3: torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.bfloat16,
            Precision.FP8_E5M2: torch.float8_e5m2 if hasattr(torch, 'float8_e5m2') else torch.bfloat16,
            Precision.INT8: torch.int8,
            Precision.INT4: torch.int8,
        }
        return dtype_map.get(self.hardware.precision, torch.bfloat16)


# ═════════════════════════════════════════════════════════════════════════════════
# Preset Configurations
# ═════════════════════════════════════════════════════════════════════════════════

def get_qlora_preset(model_name: str, max_seq_length: int = 4096) -> SOTAConfig:
    """Get optimized QLoRA configuration."""
    config = SOTAConfig(training_mode=TrainingMode.QLORA)
    config.model.name_or_path = model_name
    config.data.max_seq_length = max_seq_length
    config.lora.enabled = True
    config.lora.r = 16
    config.lora.lora_alpha = 32
    config.quantization.enabled = True
    config.quantization.load_in_4bit = True
    config.quantization.bnb_4bit_quant_type = "nf4"
    config.quantization.bnb_4bit_use_double_quant = True
    return config


def get_full_finetune_preset(model_name: str, max_seq_length: int = 4096) -> SOTAConfig:
    """Get full fine-tuning configuration."""
    config = SOTAConfig(training_mode=TrainingMode.FULL_FINETUNE)
    config.model.name_or_path = model_name
    config.data.max_seq_length = max_seq_length
    config.lora.enabled = False
    config.quantization.enabled = False
    config.hardware.precision = Precision.BF16
    return config


def get_fp8_preset(model_name: str, max_seq_length: int = 4096) -> SOTAConfig:
    """Get FP8 training configuration (H100/L40)."""
    config = SOTAConfig(training_mode=TrainingMode.FULL_FINETUNE)
    config.model.name_or_path = model_name
    config.data.max_seq_length = max_seq_length
    config.hardware.precision = Precision.FP8_E4M3
    config.quantization.load_in_fp8 = True
    return config


def get_rl_grpo_preset(model_name: str) -> SOTAConfig:
    """Get GRPO RL training configuration."""
    config = SOTAConfig(training_mode=TrainingMode.RL)
    config.model.name_or_path = model_name
    config.rl.enabled = True
    config.rl.algorithm = RLAlgorithm.GRPO
    config.rl.num_generations = 4
    config.rl.group_size = 8
    return config


def get_dpo_preset(model_name: str) -> SOTAConfig:
    """Get DPO preference optimization configuration."""
    config = SOTAConfig(training_mode=TrainingMode.RL)
    config.model.name_or_path = model_name
    config.rl.enabled = True
    config.rl.algorithm = RLAlgorithm.DPO
    config.loss.type = LossType.DPO
    config.loss.dpo_beta = 0.1
    return config


def get_export_gguf_preset(output_dir: str, quant: str = "q4_k_m") -> ExportConfig:
    """Get GGUF export configuration."""
    return ExportConfig(
        enabled=True,
        output_dir=output_dir,
        format=ExportFormat.GGUF_Q4 if quant == "q4_k_m" else ExportFormat.GGUF_Q8,
        gguf_quantization=quant,
        merge_lora=True,
    )


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "TrainingMode",
    "Precision",
    "RLAlgorithm",
    "ExportFormat",
    "OptimizerType",
    "SchedulerType",
    "LossType",
    "DeviceType",
    # Configs
    "ModelConfig",
    "TokenizerConfig",
    "LoRAConfig",
    "QuantizationConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "LossConfig",
    "RLConfig",
    "ExportConfig",
    "DistributedConfig",
    "HardwareConfig",
    "KernelConfig",
    "DataConfig",
    "TrainConfig",
    "SOTAConfig",
    # Presets
    "get_qlora_preset",
    "get_full_finetune_preset",
    "get_fp8_preset",
    "get_rl_grpo_preset",
    "get_dpo_preset",
    "get_export_gguf_preset",
]
