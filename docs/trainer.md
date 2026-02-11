# SOTA Unified Trainer Documentation

The `trainer` module provides an **Above-Unsloth** training infrastructure designed for extreme performance, scalability, and ease of use. It unifies Full Finetuning, LoRA/QLoRA, FP8, and RL into a single engine.

## 1. SOTA Trainer (`trainer/trainers/sota_trainer.py`)

The `SOTATrainer` class is the central engine. It exceeds standard trainers by integrating:

- **All Training Modes**: Full Finetuning, LoRA, QLoRA, FP8, Pretraining.
- **Advanced RL**: GRPO, GSPO, DrGRPO, DAPO (80% VRAM reduction), PPO, DPO, ORPO.
- **Triton Kernels**: Manual backprop (0% accuracy loss), Flash Attention, Fused RoPE/RMSNorm.
- **Distributed**: Native SOTA DDP and FSDP2 integration.
- **Export**: Auto-export to GGUF (q4/q5/q8), vLLM, SGLang, and HuggingFace.

### Usage

```python
from data_pipeline.trainer.core.sota_config import SOTAConfig
from data_pipeline.trainer.trainers.sota_trainer import SOTATrainer

# 1. Load configuration (YAML or Presets)
config = SOTAConfig.from_yaml("sota_config.yaml")

# 2. Initialize Trainer
trainer = SOTATrainer(config)

# 3. Setup Model (applies LoRA/Quantization/Kernels)
trainer.setup_model()

# 4. Train
output = trainer.train()

# 5. Export
trainer.export()
```

---

## 2. Configuration (`trainer/core/sota_config.py`)

The `SOTAConfig` dataclass is the single source of truth, typically loaded from YAML.

### Key Sections
- **`training_mode`**: `full`, `lora`, `qlora`, `pretrain`, `rl`.
- **`hardware`**: Precision (`bf16`, `fp16`, `fp8`), device (`auto`, `cuda`, `rocm`).
- **`lora`**: Rank, Alpha, Target Modules (`gate_proj`, `up_proj`, etc.).
- **`rl`**: Algorithm (`grpo`, `dapo`), Group Size, Generations.
- **`kernels`**: Enable/disable Triton optimizations.

### Presets

Helper functions for instant SOTA configurations:

- `get_qlora_preset(model_name)`: 4-bit NF4 + Double Quant.
- `get_fp8_preset(model_name)`: H100/L40 optimized FP8 training.
- `get_rl_grpo_preset(model_name)`: DeepSeek-style GRPO RL.
- `get_dpo_preset(model_name)`: Direct Preference Optimization.

---

## 3. Distributed Training (`trainer/distributed/`)

The trainer automatically handles distributed setups with enhanced wrappers:

- **SOTA DDP**: Optimized with `static_graph=True` and gradient bucketing.
- **SOTA FSDP2**: config-driven Fully Sharded Data Parallel with mixed precision policies.
- **Context Parallel**: For ultra-long sequence training.

---

## 4. Kernels & Optimization (`trainer/kernels/`)

Automatically detects hardware and enables:
- **LCE**: Loss-Compute-Enable manual backprop for memory savings.
- **Fused Operators**: CrossEntropy, RMSNorm, RoPE, SwiGLU.
- **Quantization**: 4-bit / 8-bit / FP8 loading and training.
