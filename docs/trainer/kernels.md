# SOTA Kernel Suite

The `trainer.kernels` package provides high-performance custom kernels, primarily written in **Triton**, to accelerate key LLM operations beyond standard PyTorch implementations.

## 1. Normalization & Activations

### Fused RMS Norm (`Fast_RMS_LayerNorm`)

- **Optimization**: Fuses the root-mean-square calculation, division, and scaling into a single kernel.
- **Speedup**: ~4x faster than `torch.nn.RMSNorm`.
- **Numerical Stability**: Uses high-precision accumulation (fp32) even for fp16/bf16 inputs.

### SwiGLU / GeGLU

- **Optimization**: Fuses the gating mechanism (split, activation, multiplication) into one kernel.
- **Memory**: Avoids materializing the large intermediate tensor after the up-projection.

## 2. Attention Mechanisms

### Flash Attention v2

- **Wrapper**: Unified interface for Flash Attention with automatic fallback.
- **Features**: Supports causal masking, windowed attention, and variable sequence lengths.

### Sliding Window Attention

- **Optimization**: Efficiently computes checking only local neighbors.

## 3. Rotary Embeddings (RoPE)

### Fast RoPE (`Fast_RoPE_Embedding`)

- **Optimization**: Fuses the complex number multiplication and rotation.
- **In-Place**: Supports in-place modification to save memory.
- **Pre-Computation**: Efficiently manages `freqs_cis` cache.

## 4. LoRA Primitives

### Fused LoRA (`LoRA_MLP`, `LoRA_QKV`)

- **Optimization**: Fuses the low-rank adaptation ($W + BA$) directly into the matrix multiplication or applies the delta update without materializing the full weight matrix.
- **SwiGLU Integration**: `apply_lora_mlp_swiglu` fuses the LoRA projection + bias + activation.

## 5. Mixture of Experts (MoE)

### Grouped GEMM

- **Concept**: Performs multiple small matrix multiplications (for different experts) as a single batched kernel.
- **Benefit**: Essential for efficient high-throughput MoE training where batch sizes per expert vary.

## 6. FP8 Quantization

### FP8 Block Linear

- **Optimization**: Custom kernels for FP8 matrix multiplication using hardware support (H100/H200).
- **Quantization**: Block-wise quantization for higher accuracy.
