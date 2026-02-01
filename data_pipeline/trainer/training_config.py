# ════════════════════════════════════════════════════════════════════════════════
# SOTA Training Configuration Module
# ════════════════════════════════════════════════════════════════════════════════
# Unified configuration for all training modes:
# - Full fine-tuning
# - Pretraining
# - LoRA/QLoRA
# - Quantization settings
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


class TrainingMode(Enum):
    """Training mode types."""
    FULL_FINETUNE = "full"
    LORA = "lora"
    QLORA = "qlora"
    PRETRAINING = "pretrain"


@dataclass
class TrainingConfig:
    """
    Unified SOTA training configuration.
    
    Supports all training modes with optimal defaults.
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # Core Settings
    # ─────────────────────────────────────────────────────────────────────────
    
    output_dir: str = "./output"
    model_name_or_path: str = ""
    training_mode: TrainingMode = TrainingMode.QLORA
    
    # ─────────────────────────────────────────────────────────────────────────
    # Training Hyperparameters
    # ─────────────────────────────────────────────────────────────────────────
    
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 = use epochs
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    
    max_seq_length: int = 2048
    
    # ─────────────────────────────────────────────────────────────────────────
    # Quantization Settings
    # ─────────────────────────────────────────────────────────────────────────
    
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    load_in_fp8: bool = False
    
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    
    # ─────────────────────────────────────────────────────────────────────────
    # LoRA Settings
    # ─────────────────────────────────────────────────────────────────────────
    
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_rslora: bool = False
    use_dora: bool = False
    
    # ─────────────────────────────────────────────────────────────────────────
    # Sequence Packing
    # ─────────────────────────────────────────────────────────────────────────
    
    pack_sequences: bool = True
    padding_free: bool = True
    
    # ─────────────────────────────────────────────────────────────────────────
    # Precision & Performance
    # ─────────────────────────────────────────────────────────────────────────
    
    fp16: bool = False
    bf16: bool = True
    tf32: bool = True
    
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    use_triton_kernels: bool = True
    
    # ─────────────────────────────────────────────────────────────────────────
    # Optimizer
    # ─────────────────────────────────────────────────────────────────────────
    
    optim: str = "adamw_8bit"  # adamw_8bit, adamw_torch, paged_adamw_8bit
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # Scheduling
    # ─────────────────────────────────────────────────────────────────────────
    
    lr_scheduler_type: str = "cosine"  # linear, cosine, cosine_with_restarts
    
    # ─────────────────────────────────────────────────────────────────────────
    # Logging & Saving
    # ─────────────────────────────────────────────────────────────────────────
    
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    report_to: str = "tensorboard"
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Seed
    # ─────────────────────────────────────────────────────────────────────────
    
    seed: int = 42
    data_seed: int = 42
    
    def __post_init__(self):
        """Validate and adjust settings."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Auto-adjust for training mode
        if self.training_mode == TrainingMode.QLORA:
            self.load_in_4bit = True
        elif self.training_mode == TrainingMode.FULL_FINETUNE:
            self.load_in_4bit = False
            self.load_in_8bit = False
        elif self.training_mode == TrainingMode.PRETRAINING:
            self.pack_sequences = True
            self.gradient_checkpointing = True
    
    @property 
    def compute_dtype(self) -> torch.dtype:
        """Get compute dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.bnb_4bit_compute_dtype, torch.bfloat16)
    
    @property
    def effective_batch_size(self) -> int:
        """Effective batch size with accumulation."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps
    
    def to_lora_config(self):
        """Convert to LoraConfig."""
        from .lora import LoraConfig
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            use_rslora=self.use_rslora,
            use_dora=self.use_dora,
        )
    
    def to_quant_config(self):
        """Convert to QuantizationConfig."""
        from .quantization import QuantizationConfig
        return QuantizationConfig(
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            load_in_fp8=self.load_in_fp8,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.compute_dtype,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, torch.dtype):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        if "training_mode" in d and isinstance(d["training_mode"], str):
            d["training_mode"] = TrainingMode(d["training_mode"])
        return cls(**d)


# ═════════════════════════════════════════════════════════════════════════════════
# Preset Configurations
# ═════════════════════════════════════════════════════════════════════════════════

def get_qlora_config(
    model_name: str,
    r: int = 16,
    max_seq_length: int = 2048,
) -> TrainingConfig:
    """Get optimized QLoRA configuration."""
    return TrainingConfig(
        model_name_or_path=model_name,
        training_mode=TrainingMode.QLORA,
        load_in_4bit=True,
        lora_r=r,
        lora_alpha=r * 2,
        max_seq_length=max_seq_length,
        pack_sequences=True,
        use_flash_attention=True,
    )


def get_full_finetune_config(
    model_name: str,
    max_seq_length: int = 2048,
) -> TrainingConfig:
    """Get full fine-tuning configuration."""
    return TrainingConfig(
        model_name_or_path=model_name,
        training_mode=TrainingMode.FULL_FINETUNE,
        load_in_4bit=False,
        bf16=True,
        max_seq_length=max_seq_length,
        gradient_checkpointing=True,
        learning_rate=1e-5,
    )


def get_pretraining_config(
    model_name: str,
    max_seq_length: int = 4096,
) -> TrainingConfig:
    """Get pretraining configuration."""
    return TrainingConfig(
        model_name_or_path=model_name,
        training_mode=TrainingMode.PRETRAINING,
        load_in_4bit=False,
        bf16=True,
        max_seq_length=max_seq_length,
        pack_sequences=True,
        gradient_checkpointing=True,
        learning_rate=3e-4,
        warmup_ratio=0.01,
    )


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    "TrainingMode",
    "TrainingConfig",
    "get_qlora_config",
    "get_full_finetune_config",
    "get_pretraining_config",
]
