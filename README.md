# TrainScale

> **A Scalable SOTA PyTorch Training Framework** — SOTA-level capabilities with 100% YAML-driven configuration.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 🌟 Why TrainScale?

TrainScale isn't just another training script. It's a **comprehensive, modular architecture** designed to solve the "last mile" problem in LLM training: **Data Engineering**.

Most frameworks treat data loading as an afterthought. TrainScale makes it a **first-class citizen** with SOTA preprocessing features usually found only in proprietary codebases (like flexible packing, token-aware distribution, and thorough dataset introspection).

### Key Differentiators
- **Zero Hardcoding**: Every aspect of the pipeline is controlled via YAML.
- **SOTA Data Pipeline**: Smart truncation, content-aware token distribution, and dynamic packing.
- **Rust-Inspired Reliability**: Uses `Result<T, E>` patterns for robust error handling.
- **Hardware Optimized**: Built-in support for Flash Attention 2, Triton kernels, and 8-bit optimizers.

---

## 🏗️ Architecture

The TrainScale pipeline operates in distinct, modular stages to ensure scalability and reproducibility.

```mermaid
graph LR
    A[YAML Configuration] --> B[Dataset Introspector]
    B --> C[Dataset Loader]
    C --> D[Prompt Engine]
    D --> E[Length Manager]
    E --> F[Tokenizer Wrapper]
    F --> G[DataLoader Builder]
    G --> H[SOTATrainer]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style H fill:#bfb,stroke:#333,stroke-width:2px
```

### Component Deep Dive

#### 1. Dataset Introspector
*   **Problem:** Hardcoding split names (`train`, `validation`) and columns (`text`, `input`) makes code brittle.
*   **Solution:** Automatically inspects HuggingFace datasets to discover available splits and columns, mapping them to a standardized schema defined in your YAML.

#### 2. Prompt Engine & Length Manager (SOTA)
*   **Problem:** Simple truncation cuts off important context; "max_length" is a blunt instrument.
*   **Solution:** 
    *   **Smart Truncation:** Respects sentence and word boundaries.
    *   **Content Distribution:** Allocates token budgets intelligently (e.g., "Give 60% to context, 40% to history").
    *   **Priority Trimming:** Drops least important columns first when context window is exceeded.

#### 3. SOTA Trainer
*   **Problem:** Training scripts are often monolithic and hard to extend.
*   **Solution:** A modular trainer supporting multiple backends (FSDP, DDP, QLoRA) and advanced features like:
    *   **Optimizers:** Adam8bit, Lion, SophiaG, Prodigy.
    *   **Schedulers:** Cosine, WSD (Warmup-Stable-Decay), REX.
    *   **Loss Functions:** Fused CrossEntropy, DPO, SimPO.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/generalaimodels/TrainScale.git
cd TrainScale

# Install dependencies
pip install -r requirements.txt

# Optional: Flash Attention 2 (Recommended for Ampere+)
pip install flash-attn --no-build-isolation
```

### 2. Run E2E Pipeline Demo

```bash
# Quick sanity check (100 samples, fast tokenizer)
python data_pipeline/examples/e2e_complete_demo.py \
    --config data_pipeline/examples/test_pipeline.yaml

# Full production training run
python data_pipeline/examples/e2e_complete_demo.py \
    --config data_pipeline/examples/complete_pipeline.yaml \
    --train \
    --output-dir ./outputs
```

---

## 🔧 Hardware & Performance

TrainScale is optimized for a wide range of hardware, from consumer GPUs to H100 clusters.

| GPU | VRAM | Mode | Max Context | Batch Size | Technique |
|-----|------|------|-------------|------------|-----------|
| **RTX 3090** | 24GB | QLoRA | 2048 | 4 | 4-bit NF4 + Gradient Checkpointing |
| **RTX 4090** | 24GB | QLoRA | 4096 | 4 | 4-bit NF4 + Flash Attn 2 |
| **A100 40GB** | 40GB | LoRA | 8192 | 8 | BF16 + Flash Attn 2 |
| **A100 80GB** | 80GB | Full | 8192 | 16 | BF16 + FSDP |
| **H100** | 80GB | Full | 16384 | 32 | FP8 + Transformer Engine |
| **Mac M1/M2** | Unified | MPS | 2048 | 1-2 | FP16 (Experimental) |

---

## 🛠️ Configuration Guide

### 1. Preprocessing (SOTA)

Control how your data is processed with granular detail:

```yaml
preprocessing:
  length_manager:
    enabled: true
    max_total_length: 4096
    truncation_strategy: "smart"  # smart, sentence_boundary, word_boundary
    
    # Precise control over character limits per column
    per_column_limits:
      instruction: 500
      input: 2000
      output: 1500

  content_distribution:
    enabled: true
    mode: "proportional" # or 'priority', 'ratio'
    column_ratios:
      instruction: 0.2
      input: 0.3
      output: 0.5
```

### 2. Training (SOTA)

Switch between training modes and hardware optimizations instantly:

```yaml
training:
  mode: "qlora" # full, lora, qlora
  
  hardware:
    precision: "bf16"
    compile_model: true  # torch.compile
  
  optimizer:
    type: "adamw_8bit"   # 75% memory saving over standard AdamW
    learning_rate: 2e-4
  
  scheduler:
    type: "wsd"          # Warmup-Stable-Decay (LLaMA-3 style)
```

---

## 🤝 How to Contribute

We welcome contributions! Whether you're fixing a bug, adding a new feature, or improving documentation, here's how you can help:

### Areas for Contribution
- [ ] **New Data Connectors:** Support for SQL, S3, or Arrow datasets.
- [ ] **Additional Kernels:** Implement optimized Triton kernels for new attention mechanisms.
- [ ] **Model Support:** Add configs for new architectures (Mistral, Gemma, Phi).
- [ ] **Benchmarks:** Run hardware benchmarks and update the README table.

### Development Standards
1.  **Type Hints**: All code must be fully type-hinted (`mypy` compliant).
2.  **Error Handling**: Use the `Result` type from `core/types.py` instead of raising raw exceptions where possible.
3.  **Config-First**: Avoid hardcoding. If a value might change, put it in the YAML schema.
4.  **Tests**: Add unit tests for new modules. Run existing tests before pushing.

### Submission Process
1.  Fork the repo.
2.  Create a branch: `git checkout -b feature/my-cool-feature`.
3.  Commit your changes.
4.  Push to your fork and submit a Pull Request.

---

## 🗺️ Roadmap

- **Phase 1: Foundation (Complete)** ✅
    - [x] End-to-end YAML pipeline
    - [x] SOTA preprocessing module
    - [x] QLoRA/LoRA support

- **Phase 2: Scale (In Progress)** 🚧
    - [ ] Multi-node FSDP training
    - [ ] DeepSpeed integration
    - [ ] Streaming dataset support for infinite datasets

- **Phase 3: Multimodal (Planned)** 🔮
    - [ ] Image/Video tokenization support
    - [ ] Audio processing pipeline

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>TrainScale</b> — Train Smarter, Scale Faster 🚀
</p>
