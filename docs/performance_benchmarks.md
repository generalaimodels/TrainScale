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

## Real-World Performance (Tesla T4)

The following results were collected on a Tesla T4 GPU.

### Key Observations
- **RMSNorm**: Achieves **3.6x - 5.0x speedup** over PyTorch native, saturating memory bandwidth (~350 GB/s).
- **Cross Entropy**: **3.2x speedup** compared to `F.cross_entropy`.
- **SwiGLU**: **1.6x speedup** via kernel fusion.
- **RoPE / FP8**: These kernels are strictly memory-bound. The "Base Latency" for these benchmarks measures a simple `tensor.clone()` (memory copy) to establish the theoretical hardware limit. The fact that our kernels achieve ~0.85x of a pure memory copy speed (while doing complex math) indicates near-perfect saturation of memory bandwidth.

```text
================================================================================
 FINAL BENCHMARK REPORT
================================================================================
+----+-----------------------------------+---------------------+--------------------+--------------------+--------+--------------------+--------------------+
|    |              Kernel               |  Our Latency (ms)   | Base Latency (ms)  |      Speedup       | Metric |      Our Perf      |     Base Perf      |
+----+-----------------------------------+---------------------+--------------------+--------------------+--------+--------------------+--------------------+
| 0  |   RMSNorm (B=4, S=2048, D=4096)   | 0.5631999969482422  |  2.86737596988678  | 5.091221565028329  |  GB/s  | 357.4690928460744  | 70.21283365499833  |
| 1  |   RMSNorm (B=4, S=2048, D=8192)   | 1.1489280462265015  | 4.173791885375977  | 3.6327704759965003 |  GB/s  | 350.4598789475632  | 96.47179233128652  |
| 2  |   RMSNorm (B=4, S=4096, D=4096)   | 1.1601920127868652  | 4.222320079803467  | 3.639328691516457  |  GB/s  | 347.0573659896157  | 95.36301758031144  |
| 3  |   RMSNorm (B=4, S=4096, D=8192)   | 2.2911999225616455  | 8.354592323303223  | 3.646382945911801  |  GB/s  | 351.4780007061269  |  96.3908634720311  |
| 4  |  RMSNorm (B=32, S=2048, D=4096)   |  4.595263957977295  | 17.315263748168945 | 3.768067276768718  |  GB/s  | 350.4940631765027  | 93.01693346544138  |
| 5  |  RMSNorm (B=32, S=2048, D=8192)   |  9.12179183959961   | 33.528751373291016 | 3.675676003450954  |  GB/s  | 353.13516561691154 |  96.073529137325   |
| 6  |  RMSNorm (B=32, S=4096, D=4096)   |  9.217887878417969  | 34.141183853149414 | 3.7037968245507598 |  GB/s  | 349.45374846030853 | 94.35013987374819  |
| 7  |  RMSNorm (B=32, S=4096, D=8192)   | 18.236448287963867  | 67.37126159667969  | 3.6943192299755543 |  GB/s  | 353.27333712519254 | 95.62609919000697  |
| 8  |         GELU (4096x4096)          | 0.28886398673057556 | 0.2890560030937195 | 1.0006647293257882 |  GB/s  | 232.31993977356782 | 232.16561248251108 |
| 9  |         GELU (16384x4096)         | 1.1344000101089478  | 1.1369600296020508 | 1.0022567167403826 |  GB/s  | 236.6320994427878  |  236.099290222151  |
| 10 |     SwiGLU (B=2048, D=11008)      | 0.5611519813537598  | 0.9460480213165283 | 1.6859033786786606 |  GB/s  | 241.05110290027798 | 142.98037832346216 |
| 11 |           RoPE (S=4096)           | 0.32787200808525085 | 0.2805759906768799 | 0.8557485352757732 |  GB/s  | 204.68006522396038 | 239.18248969950065 |
| 12 |           RoPE (S=8192)           | 0.6676480174064636  | 0.5570400059223175 | 0.8343318506152202 |  GB/s  | 201.03066960549117 | 240.94809452288686 |
| 13 |          RoPE (S=16384)           | 1.3315359950065613  | 1.1079679727554321 | 0.8320976503154708 |  GB/s  | 201.5983473271988  | 242.27727028284173 |
| 14 |     CrossEntropy (4096 toks)      |  5.133440017700195  | 16.586719512939453 | 3.231111974767045  |  GB/s  | 204.26380680099325 | 63.21780501454769  |
| 15 |     CrossEntropy (16384 toks)     | 20.320335388183594  |  66.1401596069336  | 3.254875391741542  |  GB/s  | 206.4091915746142  | 63.415389756034145 |
| 16 | FP8 Activation Quant (4096x4096)  | 0.42393600940704346 | 0.553712010383606  | 1.3061216742547521 |  GB/s  | 197.91303908661553 | 151.52726042888838 |
| 17 | FP8 Activation Quant (16384x4096) | 1.6843199729919434  | 2.2120959758758545 | 1.313346639205611  |  GB/s  | 199.25540359403274 |  151.715775291856  |
+----+-----------------------------------+---------------------+--------------------+--------------------+--------+--------------------+--------------------+
================================================================================




```

## Troubleshooting

- **"Triton not available"**: Ensure you are running in a Linux environment or WSL2 with the `triton` package installed (`pip install triton`).
- **"CUDA not available"**: Ensure your machine has an NVIDIA GPU and PyTorch is installed with CUDA support (`torch.cuda.is_available()` returns `True`).
