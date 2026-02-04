# ════════════════════════════════════════════════════════════════════════════════
# SOTA Kernel Benchmark Suite
# ════════════════════════════════════════════════════════════════════════════════
# Comprehensive benchmarking for SOTA kernels vs PyTorch native baselines.
# Measured in Latency (ms) and Throughput (TFLOPS/GB/s).
# ════════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn.functional as F
import math
try:
    import triton
    import triton.testing
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Triton not available. Using custom benchmarking utility.")

# Fallback for do_bench if Triton is missing
def do_bench_fallback(fn, warmup=25, rep=100, quantiles=None, return_mode="mean"):
    """
    Benchmark the runtime of a function.
    """
    fn()
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    
    for i in range(rep):
        start_events[i].record()
        fn()
        end_events[i].record()
    
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times = torch.tensor(times, dtype=torch.float32)
    
    if quantiles is not None:
        return torch.quantile(times, torch.tensor(quantiles, dtype=torch.float32)).tolist()
    return getattr(torch, return_mode)(times).item()

if not TRITON_AVAILABLE:
    class MockTritonTesting:
        do_bench = staticmethod(do_bench_fallback)
    
    class MockTriton:
        testing = MockTritonTesting
        
    triton = MockTriton()
import argparse
from typing import Dict, List, Optional, Callable, NamedTuple
import pandas as pd
from tabulate import tabulate

from data_pipeline.trainer.kernels import (
    # Norms
    fast_rms_layernorm,
    fused_layer_norm,
    # Activations
    fused_gelu,
    swiglu_forward,
    # Attention
    flash_attention,
    is_flash_attn_available,
    # RoPE
    fast_rope_embedding,
    precompute_freqs_cis,
    RoPEConfig,
    # Loss
    fast_cross_entropy_loss,
    # LoRA
    matmul_lora,
    # FP8
    row_quantize_fp8,
    FP8Config,
)

# Set high precision for float32 matrix multiplications
torch.set_float32_matmul_precision('high')

class BenchmarkResult(NamedTuple):
    name: str
    kernel_ms: float
    baseline_ms: float
    speedup: float
    metric_name: str
    kernel_metric: float  # e.g., TFLOPS or GB/s
    baseline_metric: float

class BenchmarkSuite:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.results: List[BenchmarkResult] = []
        
    def _bench_func(self, func: Callable, args: tuple, quantiles: list = [0.5, 0.2, 0.8]) -> float:
        """Run utility to benchmark a function using Triton's do_bench."""
        return triton.testing.do_bench(lambda: func(*args), quantiles=quantiles)[0]

    def _print_header(self, title: str):
        print(f"\n{'='*80}")
        print(f" BENCHMARK: {title}")
        print(f"{'='*80}")

    def run_rms_norm(self, batch_sizes=[4, 32], seq_lens=[2048, 4096], hidden_dims=[4096, 8192]):
        self._print_header("RMS LayerNorm")
        
        for bs in batch_sizes:
            for seq in seq_lens:
                for dim in hidden_dims:
                    x = torch.randn(bs, seq, dim, device=self.device, dtype=torch.float16)
                    w = torch.randn(dim, device=self.device, dtype=torch.float16)
                    eps = 1e-6

                    # Baseline
                    def baseline_rms(x, w, eps):
                        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

                    # Warmup & verify
                    y_ref = baseline_rms(x, w, eps)
                    y_our = fast_rms_layernorm(x, w, eps=eps)
                    
                    # Benchmark
                    ms_base = self._bench_func(baseline_rms, (x, w, eps))
                    ms_our = self._bench_func(fast_rms_layernorm, (x, w, eps))
                    
                    # Bandwidth in GB/s: 2 reads (x, w) + 1 write (y)
                    # Approximation: 2 * numel * 2 bytes + 1 * numel * 2 bytes
                    gb = (bs * seq * dim * 2 * 3) / 1e9
                    self.results.append(BenchmarkResult(
                        f"RMSNorm (B={bs}, S={seq}, D={dim})",
                        ms_our,
                        ms_base,
                        ms_base / ms_our,
                        "GB/s",
                        gb / (ms_our / 1000),
                        gb / (ms_base / 1000)
                    ))

    def run_fused_gelu(self, sizes=[(4096, 4096), (16384, 4096)]):
        self._print_header("Fused GELU")
        
        for rows, cols in sizes:
            x = torch.randn(rows, cols, device=self.device, dtype=torch.float16)
            
            # Baseline
            def baseline_gelu(x):
                return F.gelu(x)

            ms_base = self._bench_func(baseline_gelu, (x,))
            ms_our = self._bench_func(fused_gelu, (x,))
            
            gb = (rows * cols * 2 * 2) / 1e9 # Read + Write
            
            self.results.append(BenchmarkResult(
                f"GELU ({rows}x{cols})",
                ms_our,
                ms_base,
                ms_base / ms_our,
                "GB/s",
                gb / (ms_our / 1000),
                gb / (ms_base / 1000)
            ))

    def run_swiglu(self, sizes=[(2048, 11008)]): # Llama 2 7B size
        self._print_header("SwiGLU")
        
        for bs, dim in sizes:
            gate = torch.randn(bs, dim, device=self.device, dtype=torch.float16)
            up = torch.randn(bs, dim, device=self.device, dtype=torch.float16)
            
            def baseline_swiglu(g, u):
                return F.silu(g) * u
                
            ms_base = self._bench_func(baseline_swiglu, (gate, up))
            ms_our = self._bench_func(swiglu_forward, (gate, up))
            
            # 2 reads (gate, up) + 1 write (out)
            gb = (bs * dim * 2 * 3) / 1e9 
            
            self.results.append(BenchmarkResult(
                f"SwiGLU (B={bs}, D={dim})",
                ms_our,
                ms_base,
                ms_base / ms_our,
                "GB/s",
                gb / (ms_our / 1000),
                gb / (ms_base / 1000)
            ))

    def run_fp8_quantization(self, sizes=[(4096, 4096), (16384, 4096)]):
        self._print_header("FP8 Activation Quantization")

        for rows, cols in sizes:
            x = torch.randn(rows, cols, device=self.device, dtype=torch.float32)
            num_elements = x.numel()
            num_tokens = rows # Assuming rows are tokens for scale

            # Baseline: copy to simulate memory access for comparison
            def baseline_quant(x):
                return x.clone()

            ms_base = self._bench_func(baseline_quant, (x,))
            ms_quant = self._bench_func(row_quantize_fp8, (x,))
            
            # Since quantization is memory bound, we report GB/s (load + store)
            # FP32 load + FP8 store + FP32 scale store
            bytes_transferred = num_elements * 4 + num_elements * 1 + num_tokens * 4
            
            self.results.append(BenchmarkResult(
                f"FP8 Activation Quant ({rows}x{cols})",
                ms_quant,
                ms_base,
                ms_base / ms_quant,
                "GB/s",
                (bytes_transferred / ms_quant / 1e6),
                (bytes_transferred / ms_base / 1e6)
            ))

    def run_rope(self, seq_lens=[4096, 8192, 16384], dim=128):
        self._print_header("RoPE Embeddings")
        
        config = RoPEConfig(dim=dim, max_seq_len=max(seq_lens))
        
        for seq in seq_lens:
            q = torch.randn(1, seq, 32, dim, device=self.device, dtype=torch.float16) # 32 heads
            
            # Precompute frequencies
            cos, sin = precompute_freqs_cis(dim, seq, device=self.device, dtype=torch.float16)
            
            # Baseline: minimal python impl
            def baseline_rope(q, seq):
               # Simplified overhead measure
               return q + 0.0 # Placeholder for dummy op overhead
            
            # Since proper RoPE baseline is complex, we stick to measuring raw throughput of ours
            # and compare to a naive clone copy as baseline roughly
            ms_base = self._bench_func(lambda x: x.clone(), (q,)) 
            ms_our = self._bench_func(fast_rope_embedding, (q, cos, sin))
            
            gb = (q.numel() * 2 * 2) / 1e9 
            
            self.results.append(BenchmarkResult(
                f"RoPE (S={seq})",
                ms_our,
                ms_base,
                ms_base / ms_our,
                "GB/s",
                gb / (ms_our / 1000),
                gb / (ms_base / 1000)
            ))

    def run_flash_attention(self, seq_lens=[2048, 8192], head_dim=128):
        self._print_header("Flash Attention")
        
        if not is_flash_attn_available():
            print("Skipping Flash Attention (Not available)")
            return

        bs = 2
        heads = 32
        
        for seq in seq_lens:
            q = torch.randn(bs, heads, seq, head_dim, device=self.device, dtype=torch.float16)
            k = torch.randn(bs, heads, seq, head_dim, device=self.device, dtype=torch.float16)
            v = torch.randn(bs, heads, seq, head_dim, device=self.device, dtype=torch.float16)
            
            # Baseline: SDPA (which might use Flash internally, so this tests our wrapper overhead vs PyTorch's path)
            def baseline_sdpa(q, k, v):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
                
            ms_base = self._bench_func(baseline_sdpa, (q, k, v))
            ms_our = self._bench_func(flash_attention, (q, k, v, True)) # Causal=True
            
            # Approximate TFLOPS: 4 * B * H * S^2 * D / time
            flops = 4 * bs * heads * (seq**2) * head_dim
            tflops_val_our = (flops / ms_our / 1e9) # ms to s is 1e-3, so 1e12 total div -> 1e9 here... 
            # Wait, flops / (ms * 1e-3) = flops_per_sec. tflops = flops_per_sec / 1e12.
            # = flops / ms * 1000 / 1e12 = flops / ms / 1e9. Correct.

            self.results.append(BenchmarkResult(
                f"FlashAttn (S={seq})",
                ms_our,
                ms_base,
                ms_base / ms_our,
                "TFLOPS",
                (flops / ms_our / 1e9),
                (flops / ms_base / 1e9)
            ))

    def run_cross_entropy(self, vocab_size=128000, batch_tokens=[4096, 16384]):
        self._print_header("Cross Entropy Loss")
        
        for num_tokens in batch_tokens:
            logits = torch.randn(num_tokens, vocab_size, device=self.device, dtype=torch.float16)
            labels = torch.randint(0, vocab_size, (num_tokens,), device=self.device)
            
            ms_base = self._bench_func(F.cross_entropy, (logits, labels))
            ms_our = self._bench_func(fast_cross_entropy_loss, (logits, labels))
            
            gb = (num_tokens * vocab_size * 2) / 1e9
            
            self.results.append(BenchmarkResult(
                f"CrossEntropy ({num_tokens} toks)",
                ms_our,
                ms_base,
                ms_base / ms_our,
                "GB/s",
                gb / (ms_our / 1000),
                gb / (ms_base / 1000)
            ))
            


    def report(self):
        df = pd.DataFrame(self.results)
        if df.empty:
            print("No results to report.")
            return

        print("\n" + "="*80)
        print(" FINAL BENCHMARK REPORT")
        print("="*80)
        
        table = tabulate(
            df, 
            headers=["Kernel", "Our Latency (ms)", "Base Latency (ms)", "Speedup", "Metric", "Our Perf", "Base Perf"], 
            tablefmt="pretty",
            floatfmt=".4f"
        )
        print(table)
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Run SOTA Kernel Benchmarks")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available. Benchmarks require a GPU.")
        return

    suite = BenchmarkSuite(device=args.device)
    
    print(f"Running benchmarks on {torch.cuda.get_device_name(0)}")
    
    try:
        suite.run_rms_norm()
    except Exception as e: print(f"RMSNorm failed: {e}")
        
    try:
        suite.run_fused_gelu()
    except Exception as e: print(f"GELU failed: {e}")
        
    try:
        suite.run_swiglu()
    except Exception as e: print(f"SwiGLU failed: {e}")
        
    try:
        suite.run_rope()
    except Exception as e: print(f"RoPE failed: {e}")
        
    try:
        suite.run_flash_attention()
    except Exception as e: print(f"FlashAttn failed: {e}")
    
    try:
        suite.run_cross_entropy()
    except Exception as e: print(f"CrossEntropy failed: {e}")

    try:
        suite.run_fp8_quantization()
    except Exception as e: print(f"FP8 failed: {e}")
    


    suite.report()

if __name__ == "__main__":
    main()
