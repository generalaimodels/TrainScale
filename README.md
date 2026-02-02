# TrainScale

> **A Scalable SOTA PyTorch Training Framework** — Above-Unsloth-level capabilities with 100% YAML-driven configuration.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ Features

### 🚀 End-to-End Pipeline
- **Zero Hardcoding** — All settings from YAML configuration
- **Auto-Discovery** — Automatic dataset introspection (splits, columns, schemas)
- **SOTA Preprocessing** — Token-aware content distribution, smart truncation
- **Production DataLoader** — Loss-aligned tensors ready for `model.forward()`

### 🎯 Training Modes
| Mode | Description | VRAM Usage |
|------|-------------|------------|
| **Full Fine-tuning** | 16-bit/32-bit training | 100% |
| **LoRA** | Low-rank adaptation | ~60% |
| **QLoRA** | 4-bit NF4 quantization | ~30% |
| **FP8** | H100/L40 optimized | ~50% |
| **Pretraining** | From scratch | 100% |

### ⚡ SOTA Optimizers
- `AdamW` — Standard with weight decay
- `Adam8bit` — 8-bit Adam with dynamic quantization
- `Lion` — Google's sign-momentum optimizer (2x faster)
- `CAME` — Communication-efficient distributed
- `SophiaG` — Second-order Hessian optimizer
- `Prodigy` — Adaptive learning rate (no LR tuning needed)

### 📊 SOTA Schedulers
- `Cosine` — Cosine annealing with warmup
- `WSD` — LLaMA-3 Warmup-Stable-Decay
- `REX` — Rapid warmup + exponential decay
- `OneCycle` — Super-convergence scheduler
- `Polynomial` — Configurable power decay
- `CosineRestart` — SGDR with warm restarts

### 🎲 RL Training (80% VRAM Reduction)
- **GRPO** — Group Relative Policy Optimization
- **DrGRPO** — Dynamic Reward GRPO
- **DAPO** — Decoupled Advantage Policy Optimization
- **PPO** — Proximal Policy Optimization
- **DPO/ORPO/SimPO** — Preference optimization

---

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/TrainScale.git
cd TrainScale

# Install dependencies
pip install -r requirements.txt

# Optional: Flash Attention 2
pip install flash-attn --no-build-isolation
```

---

## 🚀 Quick Start

### 1. Run E2E Pipeline Demo

```bash
# Quick test with small sample
python data_pipeline/examples/e2e_complete_demo.py \
    --config data_pipeline/examples/test_pipeline.yaml

# Full production run
python data_pipeline/examples/e2e_complete_demo.py \
    --config data_pipeline/examples/complete_pipeline.yaml
```

### 2. YAML Configuration

All settings are controlled via YAML — **no hardcoding**:

```yaml
# ═══════════════════════════════════════════════════════════════════════════
# Dataset Configuration
# ═══════════════════════════════════════════════════════════════════════════
dataset:
  name: "tatsu-lab/alpaca"
  splits:
    train:
      name: "train"
      sample_size: 1000  # null = all data
      shuffle: true
      seed: 42

# ═══════════════════════════════════════════════════════════════════════════
# Tokenizer Configuration
# ═══════════════════════════════════════════════════════════════════════════
tokenizer:
  name_or_path: "meta-llama/Llama-3.1-8B-Instruct"
  max_length: 4096
  padding: "max_length"
  truncation: true

# ═══════════════════════════════════════════════════════════════════════════
# Prompt Template (Jinja2 Supported)
# ═══════════════════════════════════════════════════════════════════════════
prompt_template:
  format_type: "custom"  # chat, completion, custom
  template: |
    ### Instruction:
    {{ instruction }}
    {% if input %}
    ### Input:
    {{ input }}
    {% endif %}
    ### Response:
    {{ output }}
  input_columns: ["instruction", "input"]
  label_column: "output"
  mask_input: true
```

---

## 🔧 Hardware & Token Settings

### Recommended Settings by GPU

| GPU | VRAM | Max Length | Batch Size | Precision | Mode |
|-----|------|------------|------------|-----------|------|
| **RTX 3090** | 24GB | 2048 | 4 | bf16 | QLoRA |
| **RTX 4090** | 24GB | 4096 | 4 | bf16 | QLoRA |
| **A100 40GB** | 40GB | 8192 | 8 | bf16 | LoRA |
| **A100 80GB** | 80GB | 8192 | 16 | bf16 | Full |
| **H100** | 80GB | 16384 | 32 | fp8 | Full |
| **L40** | 48GB | 8192 | 12 | fp8 | Full |

### YAML Hardware Configuration

```yaml
training:
  # ═══════════════════════════════════════════════════════════════════════════
  # Hardware Settings
  # ═══════════════════════════════════════════════════════════════════════════
  hardware:
    device: "auto"          # auto, cuda, cuda:0, cpu
    precision: "bf16"       # bf16, fp16, fp32, fp8
    tf32: true              # Enable TF32 (Ampere+)
    compile_model: false    # torch.compile (PyTorch 2.0+)
  
  # ═══════════════════════════════════════════════════════════════════════════
  # Memory Optimization
  # ═══════════════════════════════════════════════════════════════════════════
  kernels:
    use_triton: true              # Triton kernels
    use_flash_attention: true     # Flash Attention 2
    use_fused_cross_entropy: true # Fused CE loss
    activation_checkpointing: true # Gradient checkpointing
```

---

## 📊 Token & Sequence Settings

### Preprocessing Configuration

```yaml
preprocessing:
  # ─────────────────────────────────────────────────────────────────────────
  # Length Manager: Per-column limits and truncation
  # ─────────────────────────────────────────────────────────────────────────
  length_manager:
    enabled: true
    max_total_length: 4096        # Maximum sequence length
    padding_strategy: "longest"   # longest, max_length, do_not_pad, bucket
    truncation_strategy: "smart"  # smart, simple, word_boundary, sentence_boundary
    
    # Per-column character limits
    per_column_limits:
      instruction: 4000   # Max chars for instruction
      input: 4000         # Max chars for input
      output: 8000        # Max chars for output

  # ─────────────────────────────────────────────────────────────────────────
  # Content Distribution: Token-aware allocation
  # ─────────────────────────────────────────────────────────────────────────
  content_distribution:
    enabled: true
    mode: "proportional"  # equal, proportional, ratio, priority, adaptive
    column_ratios:
      instruction: 0.3
      input: 0.1
      output: 0.55
    special_tokens_budget: 10
```

### Truncation Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `smart` | Prefers sentence > word > simple | General purpose |
| `sentence_boundary` | Truncate at sentence end | Preserves semantics |
| `word_boundary` | Truncate at word boundary | Avoids mid-word cuts |
| `simple` | Hard cut at limit | Maximum content |

### Padding Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `longest` | Pad to longest in batch | Memory efficient |
| `max_length` | Pad to fixed max_length | Consistent shapes |
| `bucket` | Bucket by length (128,256,512...) | Best efficiency |
| `do_not_pad` | No padding (for packing) | Sequence packing |

---

## 🎮 CLI Arguments

```bash
python data_pipeline/examples/e2e_complete_demo.py \
    --config <path>         # YAML config file (required)
    --split <name>          # Split to use (default: train)
    --train                 # Enable training mode
    --output-dir <path>     # Output directory
```

### Example Commands

```bash
# Quick test (100 samples, GPT-2 tokenizer)
python data_pipeline/examples/e2e_complete_demo.py \
    --config data_pipeline/examples/test_pipeline.yaml

# Full training with Llama-3.1
python data_pipeline/examples/e2e_complete_demo.py \
    --config data_pipeline/examples/complete_pipeline.yaml \
    --train \
    --output-dir ./outputs

# Validation split only
python data_pipeline/examples/e2e_complete_demo.py \
    --config my_config.yaml \
    --split validation
```

---

## 📁 Project Structure

```
TrainScale/
├── data_pipeline/
│   ├── core/               # Config schema, types, errors
│   ├── data/               # DataLoader, collate functions
│   ├── introspection/      # Dataset discovery
│   ├── preprocessing/      # Tokenization, prompts, length management
│   ├── trainer/            # SOTA trainer, optimizers, schedulers
│   │   ├── core/           # SOTAConfig, base trainer
│   │   ├── optimizers/     # Adam8bit, Lion, CAME, SophiaG
│   │   ├── schedulers/     # WSD, REX, OneCycle, CosineRestart
│   │   ├── loss/           # Chunked CE, DPO, ORPO, SimPO
│   │   └── kernels/        # Triton kernels, Flash Attention
│   └── examples/           # Demo configs and scripts
├── requirements.txt
└── README.md
```

---

## 🧪 Verification

Run the test suite to verify installation:

```bash
# Test imports
python -c "from data_pipeline.trainer import SOTAConfig, SOTATrainer; print('✅ Imports OK')"

# Test E2E pipeline
python data_pipeline/examples/e2e_complete_demo.py \
    --config data_pipeline/examples/test_pipeline.yaml
```

Expected output:
```
============================================================
PIPELINE SUMMARY
============================================================
Config: data_pipeline/examples/test_pipeline.yaml
Discovered splits: ['train']
Discovered columns: ['instruction', 'input', 'output', 'text']

Batch Output:
  input_ids: torch.Size([4, 512]) (torch.int64)
  attention_mask: torch.Size([4, 512])
  labels: torch.Size([4, 512])

✅ Pipeline complete! Tensors are ready for model.forward()
```

---

## 📚 Documentation

- [Complete Pipeline YAML](data_pipeline/examples/complete_pipeline.yaml) — Full production config
- [Test Pipeline YAML](data_pipeline/examples/test_pipeline.yaml) — Quick testing config
- [SOTA Config Schema](data_pipeline/trainer/core/sota_config.py) — All training options

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## ⭐ Acknowledgments

TrainScale builds upon excellent open-source projects:
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

---

<p align="center">
  <b>TrainScale</b> — Train Smarter, Scale Faster 🚀
</p>
