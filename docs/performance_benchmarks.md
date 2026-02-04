# SOTA Kernel Benchmarks

The `TrainScale` library includes a comprehensive benchmark suite to validate and measure the performance of its custom Triton and CUDA kernels against standard PyTorch implementations. This suite covers critical training primitives including normalization, attention, and quantization.

## Overview

The benchmark suite measures:
- **Latency (ms)**: Execution time for the kernel vs. baseline.
- **Speedup (x)**: How much faster the custom kernel is compared to the baseline.
- **Throughput**: Measured in GB/s (for memory-bound ops) or TFLOPS (for compute-bound ops).

### Benchmarked Kernels
- **RMS LayerNorm**: Triton-optimized RMS Normalization.
- **Fused GELU**: Memory-efficient GELU activation.
- **SwiGLU**: Fused Swish-Gated Linear Unit.
- **RoPE Embeddings**: Rotary Positional Embeddings with Triton.
- **Flash Attention**: Flash Attention v2 (via `flash_attn` or Triton implementation).
- **Cross Entropy Loss**: Fast, fused Cross Entropy Loss.
- **FP8 Quantization**: Block-wise and row-wise FP8 scaling and quantization.

## Prerequisites

- **GPU**: An NVIDIA GPU is required (Ampere or Hopper architecture recommended for FP8).
- **Triton**: The OpenAI Triton compiler (`triton`) must be installed for valid kernel benchmarks.
- **PyTorch**: Compatible PyTorch version with CUDA support.

> **Note**: If `triton` is not detected (e.g., on Windows without WSL/Triton support), the suite will use a mock fallback/robustness mode to ensure the script runs, but performance numbers will reflect the fallback implementation.

## Usage

To run the full benchmark suite:

```bash
python -m data_pipeline.trainer.kernels.benchmark_suite --device cuda
```

### Command Line Arguments

- `--device`: Specify the target device (default: `cuda`).

## Example Output

The benchmark suite outputs a tabular report summarizing the performance of each kernel.

```text
================================================================================
 FINAL BENCHMARK REPORT
================================================================================
+--------------------------------+------------------+-------------------+---------+--------+----------+-----------+
|             Kernel             | Our Latency (ms) | Base Latency (ms) | Speedup | Metric | Our Perf | Base Perf |
+--------------------------------+------------------+-------------------+---------+--------+----------+-----------+
|   RMSNorm (B=32, S=4096, D=8192)  |      0.1245      |       0.4521      |  3.63x  |  GB/s  | 2500.45  |   688.12  |
|        GELU (16384x4096)       |      0.0890      |       0.1560      |  1.75x  |  GB/s  | 1800.20  |  1028.50  |
|      FlashAttn (S=8192)        |      2.4500      |       8.1200      |  3.31x  | TFLOPS |  120.50  |   36.20   |
+--------------------------------+------------------+-------------------+---------+--------+----------+-----------+
================================================================================
```

## Troubleshooting

- **"Triton not available"**: Ensure you are running in a Linux environment or WSL2 with the `triton` package installed (`pip install triton`).
- **"CUDA not available"**: Ensure your machine has an NVIDIA GPU and PyTorch is installed with CUDA support (`torch.cuda.is_available()` returns `True`).
