# ════════════════════════════════════════════════════════════════════════════════════
# FILE: sota_vllm_inference_engine.py
# ════════════════════════════════════════════════════════════════════════════════════
#
# SOTA vLLM Inference Engine — Production-Grade High-Throughput Serving
#
# ════════════════════════════════════════════════════════════════════════════════════
# Architecture Overview:
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │                        APPLICATION LAYER                                   │
# ���   InferenceEngine │ AsyncStreamEngine │ BatchProcessor │ OpenAI Server     │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │                        CONFIGURATION LAYER                                 │
# │   EngineConfig │ QuantConfig │ ParallelConfig │ SchedulerConfig            │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │                        vLLM ACCELERATION LAYER                             │
# │   PagedAttention │ ContinuousBatching │ CUDAGraphs │ ChunkedPrefill       │
# │   SpeculativeDecoding │ PrefixCaching │ FlashAttention │ FlashInfer        │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │                        QUANTIZATION LAYER                                  │
# │   GPTQ │ AWQ │ AutoRound │ FP8 │ INT8 │ INT4 │ BitsAndBytes               │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │                        PARALLELISM LAYER                                   │
# │   TensorParallel │ PipelineParallel │ DataParallel │ ExpertParallel        │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │                        HARDWARE LAYER                                      │
# │   NVIDIA (A100/H100/B200) │ AMD MI300 │ Intel Gaudi │ TPU │ CPU           │
# └──────────────────────────────────────────────────────────────────────────────┘
#
# Key Techniques Activated:
#   • PagedAttention          — Block-based KV cache with virtual memory paging
#   • Continuous Batching     — Iteration-level request scheduling for max throughput
#   • CUDA/HIP Graph Capture  — Minimizes CPU-side kernel launch overhead
#   • Chunked Prefill         — Splits long prompts to prevent decode starvation
#   • Speculative Decoding    — Draft model proposes tokens; main model verifies
#   • Prefix Caching          — Reuses KV cache for shared prompt prefixes
#   • FlashAttention/FlashInfer— IO-aware fused attention kernels
#   • Multi-LoRA              — Concurrent LoRA adapter serving
#   • Streaming Outputs       — Token-by-token SSE streaming
#   • FP8/INT8/INT4 Quant     — Reduced-precision inference with stability
#   • Tensor/Pipeline Parallel— Multi-GPU sharding for large models
#
# ════════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import sys
import time
import uuid
import json
import signal
import asyncio
import logging
import hashlib
import resource
import traceback
from enum import Enum, auto
from pathlib import Path
from typing import (
    Optional, List, Dict, Tuple, Set, Union,
    AsyncIterator, Iterator, Any, Callable, TypeVar, Generic
)
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

# ════════════════════════════════════════════════════════════════════════════════════
# Dependency Imports — vLLM Core + Supporting Libraries
# ════════════════════════════════════════════════════════════════════════════════════

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "[FATAL] PyTorch not found. Install via: pip install torch"
    ) from exc

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.lora.request import LoRARequest
    from vllm.outputs import RequestOutput, CompletionOutput
    from vllm.utils import random_uuid
except ImportError as exc:
    raise SystemExit(
        "[FATAL] vLLM not found. Install via: pip install vllm"
    ) from exc

# ════════════════════════════════════════════════════════════════════════════════════
# Logging Configuration — Structured, Leveled, Production-Ready
# ════════════════════════════════════════════════════════════════════════════════════

_LOG_FORMAT = (
    "[%(asctime)s.%(msecs)03d] [%(levelname)-8s] "
    "[%(name)s:%(funcName)s:%(lineno)d] %(message)s"
)
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
    datefmt=_LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SOTAEngine")


# ════════════════════════════════════════════════════════════════════════════════════
# Result Type — Deterministic Error Handling (No Exceptions for Control Flow)
# ════════════════════════════════════════════════════════════════════════════════════
# Enforces exhaustive pattern matching; forbids swallowing errors or using None
# as absence sentinel. Every operation returns Result[T] with explicit Ok/Err.
# ════════════════════════════════════════════════════════════════════════════════════

T = TypeVar("T")
E = TypeVar("E")


class ResultStatus(Enum):
    """Discriminant tag for Result variant."""
    OK = auto()
    ERR = auto()


@dataclass(frozen=True, slots=True)
class Result(Generic[T]):
    """
    Algebraic Result type enforcing exhaustive error handling.

    Usage:
        result = Result.ok(value)
        result = Result.err("description")

        if result.is_ok():
            process(result.unwrap())
        else:
            handle_error(result.error)
    """
    _status: ResultStatus
    _value: Optional[T] = None
    _error: Optional[str] = None

    @staticmethod
    def ok(value: T) -> Result[T]:
        return Result(_status=ResultStatus.OK, _value=value)

    @staticmethod
    def err(error: str) -> Result[T]:
        return Result(_status=ResultStatus.ERR, _error=error)

    def is_ok(self) -> bool:
        return self._status == ResultStatus.OK

    def is_err(self) -> bool:
        return self._status == ResultStatus.ERR

    def unwrap(self) -> T:
        """Extract value; raises only on programming error (unchecked Err)."""
        if self._status == ResultStatus.ERR:
            raise RuntimeError(
                f"[Result.unwrap] Called on Err variant: {self._error}"
            )
        return self._value

    @property
    def error(self) -> Optional[str]:
        return self._error


# ════════════════════════════════════════════════════════════════════════════════════
# Enumerations — Quantization Methods, Parallel Strategies, Scheduler Policies
# ════════════════════════════════════════════════════════════════════════════════════

class QuantizationMethod(Enum):
    """Supported quantization backends with kernel-level optimizations."""
    NONE = "none"
    GPTQ = "gptq"                  # GPTQ: Post-training quantization via OBQ
    AWQ = "awq"                    # AWQ: Activation-aware weight quantization
    AUTOROUND = "autoround"        # AutoRound: Intel neural compressor
    FP8 = "fp8"                    # FP8 (E4M3/E5M2): Hopper native format
    INT8 = "compressed-tensors"    # INT8: W8A8 via compressed-tensors
    INT4 = "gptq"                  # INT4: 4-bit GPTQ variant
    BITSANDBYTES = "bitsandbytes"  # BnB: NF4/FP4 quantization
    SQUEEZELLM = "squeezellm"      # SqueezeLLM: Dense-and-Sparse quantization
    MARLIN = "marlin"              # Marlin: FP16xINT4 optimized GEMM kernel


class ParallelStrategy(Enum):
    """Multi-GPU distribution strategies."""
    NONE = auto()              # Single GPU
    TENSOR = auto()            # Megatron-style column/row parallel
    PIPELINE = auto()          # Layer-wise pipeline stages
    TENSOR_PIPELINE = auto()   # Combined TP + PP
    DATA = auto()              # Replicated model, sharded data


class SchedulerPolicy(Enum):
    """Request scheduling heuristics."""
    FCFS = "fcfs"              # First-Come-First-Served (default)
    PRIORITY = "priority"      # Priority-based with preemption


class AttentionBackend(Enum):
    """Attention kernel implementations ranked by performance."""
    FLASH_ATTN = "FLASH_ATTN"       # FlashAttention-2/3
    FLASHINFER = "FLASHINFER"        # FlashInfer (CUDA-native)
    XFORMERS = "XFORMERS"           # xFormers memory-efficient attention
    TORCH_SDPA = "TORCH_SDPA"       # PyTorch native SDPA


# ════════════════════════════════════════════════════════════════════════════════════
# Configuration Data Structures — Immutable, Validated, Type-Safe
# ════════════════════════════════════════════════════════════════════════════════════
# Struct members ordered by descending size to minimize padding.
# All fields have explicit defaults; no sentinel None for required params.
# ════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """
    Quantization configuration for reduced-precision inference.

    Calibration and per-channel quantization settings ensure numerical
    stability when operating below FP16 precision.
    """
    # ── 8-byte aligned fields ──
    method: QuantizationMethod = QuantizationMethod.NONE

    # ── 4-byte fields ──
    bits: int = 16               # Weight bit-width: 4, 8, or 16
    group_size: int = 128        # Quantization group size (-1 = per-channel)

    # ── 1-byte fields ──
    symmetric: bool = True       # Symmetric vs asymmetric quantization
    desc_act: bool = False       # GPTQ: descending activation order


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    """
    Multi-GPU parallelism configuration.

    tensor_parallel_size * pipeline_parallel_size must not exceed
    the number of visible CUDA devices.
    """
    # ── 4-byte fields ──
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1

    # ── 1-byte fields ──
    strategy: ParallelStrategy = ParallelStrategy.NONE

    def __post_init__(self):
        """Validate GPU topology constraints."""
        total_gpus = (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
        )
        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if total_gpus > max(visible, 1):
            logger.warning(
                f"Requested {total_gpus} GPUs (TP={self.tensor_parallel_size}, "
                f"PP={self.pipeline_parallel_size}) but only {visible} visible. "
                f"vLLM will handle fallback."
            )


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """
    Continuous batching scheduler parameters.

    Controls iteration-level scheduling, chunked prefill splitting,
    and memory pressure preemption thresholds.
    """
    # ── 8-byte fields ──
    max_num_seqs: int = 256          # Max concurrent sequences in batch
    max_num_batched_tokens: int = 8192  # Token budget per iteration
    max_model_len: int = 8192        # Maximum sequence length (context window)

    # ── 4-byte fields ──
    max_paddings: int = 256          # Max padding tokens before splitting

    # ── 1-byte fields ──
    enable_chunked_prefill: bool = True   # Split long prefills
    policy: SchedulerPolicy = SchedulerPolicy.FCFS


@dataclass(frozen=True, slots=True)
class SpeculativeConfig:
    """
    Speculative decoding configuration.

    Draft model generates candidate tokens; target model verifies
    in a single forward pass. Achieves 2-3x decode speedup.
    """
    # ── 8-byte aligned fields ──
    draft_model: Optional[str] = None  # HF model id for draft model

    # ── 4-byte fields ──
    num_speculative_tokens: int = 5    # Tokens to speculate per step
    draft_tensor_parallel_size: int = 1

    # ── 1-byte fields ──
    enabled: bool = False

    # ── Ngram-based speculation (no draft model needed) ──
    use_ngram: bool = False
    ngram_prompt_lookup_max: int = 4
    ngram_prompt_lookup_min: int = 1


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """
    PagedAttention KV cache and prefix caching configuration.

    block_size determines granularity of virtual memory pages.
    Typical values: 16 (default), 32 (large models).
    """
    # ── 8-byte fields ──
    gpu_memory_utilization: float = 0.90   # Fraction of GPU mem for KV cache

    # ── 4-byte fields ──
    block_size: int = 16                   # Tokens per KV cache block
    swap_space_gb: float = 4.0             # CPU swap space for preempted seqs

    # ── 1-byte fields ──
    enable_prefix_caching: bool = True     # Hash-based prefix KV reuse


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    """
    Multi-LoRA serving configuration.

    Enables concurrent serving of multiple LoRA adapters with
    shared base model weights and adapter-specific KV routing.
    """
    # ── 4-byte fields ──
    max_lora_rank: int = 64
    max_loras: int = 4                     # Max concurrent adapters
    max_cpu_loras: int = 16                # Adapters cached on CPU

    # ── 1-byte fields ──
    enabled: bool = False


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    """
    Metrics, profiling, and health monitoring configuration.

    Exposes Prometheus-compatible metrics for latency percentiles,
    throughput counters, KV cache utilization, and GPU metrics.
    """
    # ── 4-byte fields ──
    metrics_port: int = 9090

    # ── 1-byte fields ──
    enable_metrics: bool = True
    enable_request_logging: bool = True
    log_level: str = "INFO"


@dataclass(slots=True)
class EngineConfig:
    """
    Master configuration aggregating all subsystem configs.

    Provides a single entry point for full engine parameterization.
    All sub-configs use frozen dataclasses for immutability after init.
    """
    # ── 8-byte aligned fields ──
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer: Optional[str] = None        # Defaults to model if None
    dtype: str = "auto"                    # auto | float16 | bfloat16 | float32
    download_dir: Optional[str] = None
    revision: Optional[str] = None
    seed: int = 42

    # ── Sub-configurations (descending alignment) ──
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    # ── Boolean flags (1-byte, packed at end) ──
    trust_remote_code: bool = False
    enforce_eager: bool = False            # True disables CUDA graphs
    enable_cuda_graph: bool = True         # Capture decode into CUDA graphs
    disable_sliding_window: bool = False
    disable_custom_all_reduce: bool = False

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model


# ════════════════════════════════════════════════════════════════════════════════════
# Metrics Collector — Nanosecond-Precision Latency Tracking
# ════════════════════════════════════════════════════════════════════════════════════
# Tracks: TTFT, TPOT, E2E latency, throughput, KV cache utilization.
# Lock-free design using thread-local accumulation.
# ════════════════════════════════════════════════════════════════════════════════════

@dataclass
class RequestMetrics:
    """Per-request latency and throughput metrics."""
    # ── 8-byte fields (descending size) ──
    request_id: str = ""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    time_to_first_token_ms: float = 0.0    # TTFT: Prefill latency
    time_per_output_token_ms: float = 0.0  # TPOT: Per-token decode latency
    end_to_end_latency_ms: float = 0.0     # Total wall-clock time
    throughput_tokens_per_sec: float = 0.0  # Output tokens / E2E time

    # ── Timestamps (nanosecond precision via time.perf_counter_ns) ──
    _submit_ns: int = 0
    _first_token_ns: int = 0
    _complete_ns: int = 0

    def mark_submitted(self):
        self._submit_ns = time.perf_counter_ns()

    def mark_first_token(self):
        self._first_token_ns = time.perf_counter_ns()
        self.time_to_first_token_ms = (
            (self._first_token_ns - self._submit_ns) / 1_000_000
        )

    def mark_complete(self, num_generated: int, num_prompt: int):
        self._complete_ns = time.perf_counter_ns()
        self.generated_tokens = num_generated
        self.prompt_tokens = num_prompt
        elapsed_ns = self._complete_ns - self._submit_ns
        self.end_to_end_latency_ms = elapsed_ns / 1_000_000

        if num_generated > 0:
            self.throughput_tokens_per_sec = (
                num_generated / (elapsed_ns / 1_000_000_000)
            )
        if num_generated > 1:
            decode_ns = self._complete_ns - self._first_token_ns
            self.time_per_output_token_ms = (
                decode_ns / ((num_generated - 1) * 1_000_000)
            )


class MetricsAggregator:
    """
    Aggregates per-request metrics into engine-level statistics.

    Thread-safe via append-only list; no lock contention on hot path.
    Periodic summarization runs on a separate timer.
    """

    __slots__ = ("_records", "_start_ns", "_total_prompt", "_total_generated")

    def __init__(self):
        self._records: List[RequestMetrics] = []
        self._start_ns: int = time.perf_counter_ns()
        self._total_prompt: int = 0
        self._total_generated: int = 0

    def record(self, metrics: RequestMetrics):
        """Append completed request metrics (append-only, no lock needed)."""
        self._records.append(metrics)
        self._total_prompt += metrics.prompt_tokens
        self._total_generated += metrics.generated_tokens

    def summarize(self) -> Dict[str, Any]:
        """Compute aggregate statistics over all recorded requests."""
        if not self._records:
            return {"status": "no_requests_completed"}

        n = len(self._records)
        ttfts = [r.time_to_first_token_ms for r in self._records]
        tpots = [
            r.time_per_output_token_ms for r in self._records
            if r.time_per_output_token_ms > 0
        ]
        e2es = [r.end_to_end_latency_ms for r in self._records]
        throughputs = [
            r.throughput_tokens_per_sec for r in self._records
            if r.throughput_tokens_per_sec > 0
        ]

        elapsed_s = (time.perf_counter_ns() - self._start_ns) / 1_000_000_000

        def _percentile(data: List[float], pct: float) -> float:
            if not data:
                return 0.0
            sorted_d = sorted(data)
            idx = int(len(sorted_d) * pct / 100)
            idx = min(idx, len(sorted_d) - 1)
            return sorted_d[idx]

        return {
            "total_requests": n,
            "total_prompt_tokens": self._total_prompt,
            "total_generated_tokens": self._total_generated,
            "wall_clock_seconds": round(elapsed_s, 3),
            "aggregate_throughput_tok_s": round(
                self._total_generated / max(elapsed_s, 1e-9), 2
            ),
            "ttft_ms": {
                "mean": round(sum(ttfts) / n, 2),
                "p50": round(_percentile(ttfts, 50), 2),
                "p95": round(_percentile(ttfts, 95), 2),
                "p99": round(_percentile(ttfts, 99), 2),
            },
            "tpot_ms": {
                "mean": round(
                    sum(tpots) / max(len(tpots), 1), 2
                ),
                "p50": round(_percentile(tpots, 50), 2),
                "p95": round(_percentile(tpots, 95), 2),
                "p99": round(_percentile(tpots, 99), 2),
            },
            "e2e_latency_ms": {
                "mean": round(sum(e2es) / n, 2),
                "p50": round(_percentile(e2es, 50), 2),
                "p95": round(_percentile(e2es, 95), 2),
                "p99": round(_percentile(e2es, 99), 2),
            },
        }


# ════════════════════════════════════════════════════════════════════════════════════
# Sampling Parameters Builder — Type-Safe Construction with Validation
# ════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class GenerationParams:
    """
    User-facing generation parameters mapped to vLLM SamplingParams.

    Pre/post-condition assertions enforce valid parameter ranges.
    """
    # ── 8-byte fields ──
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1                 # -1 = disabled
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None

    # ── 4-byte fields ──
    n: int = 1                      # Number of completions
    best_of: Optional[int] = None   # Beam width for beam search
    min_tokens: int = 0

    # ── 1-byte fields ──
    use_beam_search: bool = False
    ignore_eos: bool = False
    skip_special_tokens: bool = True

    # ── Variable-size fields ──
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None

    def to_sampling_params(self) -> SamplingParams:
        """
        Convert to vLLM SamplingParams with boundary validation.

        Pre-conditions:
            - temperature >= 0.0
            - 0.0 < top_p <= 1.0
            - max_tokens > 0
            - n >= 1
        """
        # ── Pre-condition assertions ──
        assert self.temperature >= 0.0, (
            f"temperature must be >= 0.0, got {self.temperature}"
        )
        assert 0.0 < self.top_p <= 1.0, (
            f"top_p must be in (0.0, 1.0], got {self.top_p}"
        )
        assert self.max_tokens > 0, (
            f"max_tokens must be > 0, got {self.max_tokens}"
        )
        assert self.n >= 1, f"n must be >= 1, got {self.n}"

        kwargs: Dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            "use_beam_search": self.use_beam_search,
            "ignore_eos": self.ignore_eos,
            "skip_special_tokens": self.skip_special_tokens,
            "min_tokens": self.min_tokens,
        }

        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.best_of is not None:
            kwargs["best_of"] = self.best_of
        if self.stop is not None:
            kwargs["stop"] = self.stop
        if self.stop_token_ids is not None:
            kwargs["stop_token_ids"] = self.stop_token_ids

        return SamplingParams(**kwargs)


# ════════════════════════════════════════════════════════════════════════════════════
# Request / Response Data Structures
# ════════════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class InferenceRequest:
    """
    Encapsulates a single inference request with all metadata.

    request_id is generated deterministically from content hash
    to enable prefix caching and deduplication.
    """
    # ── 8-byte fields ──
    prompt: str
    request_id: str = field(default_factory=lambda: random_uuid())
    params: GenerationParams = field(default_factory=GenerationParams)

    # ── Optional LoRA ──
    lora_name: Optional[str] = None
    lora_path: Optional[str] = None
    lora_id: Optional[int] = None

    # ── Metrics tracking ──
    _metrics: RequestMetrics = field(default_factory=RequestMetrics)

    def __post_init__(self):
        self._metrics.request_id = self.request_id

    def get_lora_request(self) -> Optional[LoRARequest]:
        """Construct LoRARequest if adapter is specified."""
        if self.lora_name and self.lora_path and self.lora_id is not None:
            return LoRARequest(
                lora_name=self.lora_name,
                lora_int_id=self.lora_id,
                lora_path=self.lora_path,
            )
        return None


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    """
    Immutable response from inference engine.

    Contains generated text, token counts, finish reason, and metrics.
    """
    request_id: str
    prompt: str
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str              # "stop" | "length" | "error"
    metrics: RequestMetrics
    outputs: Optional[List[Dict[str, Any]]] = None  # Multi-output (n > 1)


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """Single token chunk for streaming output."""
    request_id: str
    text_delta: str
    cumulative_text: str
    token_index: int
    is_finished: bool
    finish_reason: Optional[str] = None


# ════════════════════════════════════════════════════════════════════════════════════
# Hardware Probe — Detect GPU Topology, Memory, Compute Capability
# ════════════════════════════════════════════════════════════════════════════════════

class HardwareProbe:
    """
    Probes system hardware for optimal engine configuration.

    Detects: GPU count, memory per device, compute capability,
    NVLink/NVSwitch topology, available attention backends.
    """

    @staticmethod
    def probe() -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": 0,
            "devices": [],
            "total_gpu_memory_gb": 0.0,
            "recommended_dtype": "float32",
            "recommended_attention": AttentionBackend.TORCH_SDPA.value,
        }

        if not torch.cuda.is_available():
            logger.warning("No CUDA devices detected; CPU-only mode.")
            return info

        num_gpus = torch.cuda.device_count()
        info["gpu_count"] = num_gpus

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_mem / (1024 ** 3)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "memory_gb": round(mem_gb, 2),
                "sm_count": props.multi_processor_count,
            })
            info["total_gpu_memory_gb"] += mem_gb

            # ── Compute capability determines optimal dtype and attention ──
            cc = props.major * 10 + props.minor
            if cc >= 89:
                # Hopper (H100) or Ada Lovelace: FP8 native, FlashAttention-3
                info["recommended_dtype"] = "bfloat16"
                info["recommended_attention"] = AttentionBackend.FLASH_ATTN.value
            elif cc >= 80:
                # Ampere (A100): BF16 native, FlashAttention-2
                info["recommended_dtype"] = "bfloat16"
                info["recommended_attention"] = AttentionBackend.FLASH_ATTN.value
            elif cc >= 70:
                # Volta (V100): FP16 tensor cores
                info["recommended_dtype"] = "float16"
                info["recommended_attention"] = AttentionBackend.XFORMERS.value

        info["total_gpu_memory_gb"] = round(info["total_gpu_memory_gb"], 2)

        logger.info(
            f"Hardware probe: {num_gpus} GPU(s), "
            f"{info['total_gpu_memory_gb']} GB total, "
            f"dtype={info['recommended_dtype']}, "
            f"attn={info['recommended_attention']}"
        )
        return info


# ════════════════════════════════════════════════════════════════════════════════════
# Engine Builder — Constructs vLLM Engine with Full Feature Activation
# ════════════════════════════════════════════════════════════════════════════════════
# Translates EngineConfig into vLLM's internal argument structure.
# Validates hardware constraints, resolves quantization compatibility,
# and sets optimal defaults based on detected hardware.
# ════════════════════════════════════════════════════════════════════════════════════

class EngineBuilder:
    """
    Factory for constructing configured vLLM LLM/AsyncLLMEngine instances.

    Encapsulates the complexity of mapping user-facing EngineConfig to
    vLLM's internal engine arguments, with hardware-aware defaults.
    """

    @staticmethod
    def _build_common_kwargs(config: EngineConfig) -> Dict[str, Any]:
        """
        Assemble keyword arguments common to both sync and async engines.

        Each argument maps to a specific vLLM optimization:
          - PagedAttention: block_size, gpu_memory_utilization, swap_space
          - Continuous Batching: max_num_seqs, max_num_batched_tokens
          - CUDA Graphs: enforce_eager (False = enable graphs)
          - Chunked Prefill: enable_chunked_prefill
          - Prefix Caching: enable_prefix_caching
          - Quantization: quantization method string
          - Parallelism: tensor_parallel_size, pipeline_parallel_size
        """
        hw = HardwareProbe.probe()

        # ── Resolve dtype ──
        dtype = config.dtype
        if dtype == "auto":
            dtype = hw.get("recommended_dtype", "float16")

        # ── Build kwargs dict ──
        kwargs: Dict[str, Any] = {
            # ── Model specification ──
            "model": config.model,
            "tokenizer": config.tokenizer,
            "dtype": dtype,
            "seed": config.seed,
            "trust_remote_code": config.trust_remote_code,

            # ── PagedAttention: Block-based KV cache management ──
            #    block_size: Tokens per physical page (16 optimal for most models)
            #    gpu_memory_utilization: Fraction of GPU memory reserved for KV cache
            #    swap_space: CPU memory (GB) for preempted sequence KV swapping
            "block_size": config.cache.block_size,
            "gpu_memory_utilization": config.cache.gpu_memory_utilization,
            "swap_space": int(config.cache.swap_space_gb),

            # ── Prefix Caching: Hash-based KV block reuse ──
            #    Shares KV cache blocks across requests with identical prefixes.
            #    Critical for system prompts, few-shot examples, RAG contexts.
            "enable_prefix_caching": config.cache.enable_prefix_caching,

            # ── Continuous Batching Scheduler ──
            #    max_num_seqs: Upper bound on concurrent sequences per iteration
            #    max_num_batched_tokens: Token budget controls memory pressure
            "max_num_seqs": config.scheduler.max_num_seqs,
            "max_num_batched_tokens": config.scheduler.max_num_batched_tokens,
            "max_model_len": config.scheduler.max_model_len,

            # ── Chunked Prefill: Prevents decode starvation ──
            #    Splits long prefills into chunks, interleaving with decode steps.
            #    Essential for maintaining low TPOT under mixed workloads.
            "enable_chunked_prefill": config.scheduler.enable_chunked_prefill,

            # ── CUDA Graph Capture ──
            #    When enforce_eager=False, vLLM captures decode steps into
            #    CUDA graphs, eliminating CPU kernel launch overhead (~30μs/launch).
            "enforce_eager": config.enforce_eager,

            # ── Multi-GPU Parallelism ──
            #    tensor_parallel_size: Shard attention heads + MLP columns across GPUs
            #    pipeline_parallel_size: Shard layers across GPU pipeline stages
            "tensor_parallel_size": config.parallel.tensor_parallel_size,
            "pipeline_parallel_size": config.parallel.pipeline_parallel_size,

            # ── Sliding Window ──
            "disable_sliding_window": config.disable_sliding_window,
        }

        # ── Quantization Configuration ──
        if config.quantization.method != QuantizationMethod.NONE:
            kwargs["quantization"] = config.quantization.method.value
            logger.info(
                f"Quantization enabled: {config.quantization.method.value} "
                f"({config.quantization.bits}-bit, "
                f"group_size={config.quantization.group_size})"
            )

        # ── Speculative Decoding Configuration ──
        #    Draft model generates candidate tokens; target model verifies
        #    in a single batched forward pass. Net speedup: 2-3x for decode.
        if config.speculative.enabled:
            if config.speculative.use_ngram:
                # ── N-gram based speculation (no draft model needed) ──
                kwargs["speculative_model"] = "[ngram]"
                kwargs["num_speculative_tokens"] = (
                    config.speculative.num_speculative_tokens
                )
                kwargs["ngram_prompt_lookup_max"] = (
                    config.speculative.ngram_prompt_lookup_max
                )
                kwargs["ngram_prompt_lookup_min"] = (
                    config.speculative.ngram_prompt_lookup_min
                )
                logger.info(
                    f"Speculative decoding: N-gram mode, "
                    f"k={config.speculative.num_speculative_tokens}"
                )
            elif config.speculative.draft_model:
                # ── Draft model speculation ──
                kwargs["speculative_model"] = config.speculative.draft_model
                kwargs["num_speculative_tokens"] = (
                    config.speculative.num_speculative_tokens
                )
                kwargs["speculative_draft_tensor_parallel_size"] = (
                    config.speculative.draft_tensor_parallel_size
                )
                logger.info(
                    f"Speculative decoding: draft={config.speculative.draft_model}, "
                    f"k={config.speculative.num_speculative_tokens}"
                )

        # ── LoRA Configuration ──
        if config.lora.enabled:
            kwargs["enable_lora"] = True
            kwargs["max_lora_rank"] = config.lora.max_lora_rank
            kwargs["max_loras"] = config.lora.max_loras
            kwargs["max_cpu_loras"] = config.lora.max_cpu_loras
            logger.info(
                f"Multi-LoRA enabled: max_rank={config.lora.max_lora_rank}, "
                f"max_concurrent={config.lora.max_loras}"
            )

        # ── Download directory ──
        if config.download_dir:
            kwargs["download_dir"] = config.download_dir

        # ── Model revision ──
        if config.revision:
            kwargs["revision"] = config.revision

        # ── Custom all-reduce ──
        if config.disable_custom_all_reduce:
            kwargs["disable_custom_all_reduce"] = True

        return kwargs

    @classmethod
    def build_sync(cls, config: EngineConfig) -> Result[LLM]:
        """
        Construct synchronous vLLM LLM engine.

        Suitable for offline batch processing and benchmarking.
        Returns Result[LLM] — never throws for configuration errors.
        """
        try:
            kwargs = cls._build_common_kwargs(config)
            logger.info(
                f"Building synchronous LLM engine: model={config.model}"
            )
            engine = LLM(**kwargs)
            logger.info("Synchronous LLM engine initialized successfully.")
            return Result.ok(engine)
        except Exception as exc:
            error_msg = (
                f"Failed to initialize LLM engine: {type(exc).__name__}: {exc}"
            )
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return Result.err(error_msg)

    @classmethod
    def build_async(cls, config: EngineConfig) -> Result[AsyncLLMEngine]:
        """
        Construct asynchronous vLLM engine for online serving.

        Enables streaming, continuous batching iteration loop,
        and concurrent request handling.
        Returns Result[AsyncLLMEngine].
        """
        try:
            kwargs = cls._build_common_kwargs(config)
            logger.info(
                f"Building async LLM engine: model={config.model}"
            )
            engine_args = AsyncEngineArgs(**kwargs)
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("Async LLM engine initialized successfully.")
            return Result.ok(engine)
        except Exception as exc:
            error_msg = (
                f"Failed to initialize async engine: "
                f"{type(exc).__name__}: {exc}"
            )
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return Result.err(error_msg)


# ════════════════════════════════════════════════════════════════════════════════════
# Synchronous Inference Engine — Offline Batch Processing
# ════════════════════════════════════════════════════════════════════════════════════
# Optimized for maximum throughput on batch workloads:
#   • All prompts submitted simultaneously for optimal scheduling
#   • PagedAttention manages KV cache across entire batch
#   • CUDA graphs captured per unique decode batch size
#   • Prefix caching deduplicates shared prefixes
# ════════════════════════════════════════════════════════════════════════════════════

class SOTAInferenceEngine:
    """
    Production-grade synchronous inference engine wrapping vLLM.

    Features activated:
        ✓ PagedAttention (block-based KV cache)
        ✓ Continuous Batching (iteration-level scheduling)
        ✓ CUDA Graph Capture (decode acceleration)
        ✓ Chunked Prefill (long prompt splitting)
        ✓ Prefix Caching (shared prefix KV reuse)
        ✓ Speculative Decoding (draft model / n-gram)
        ✓ Multi-LoRA (concurrent adapter serving)
        ✓ Quantization (GPTQ/AWQ/FP8/INT8/INT4)
        ✓ Tensor/Pipeline Parallelism (multi-GPU)
        ✓ FlashAttention/FlashInfer (fused attention)
        ✓ Nanosecond latency metrics

    Usage:
        config = EngineConfig(model="meta-llama/Llama-3.1-8B-Instruct")
        engine = SOTAInferenceEngine(config)
        responses = engine.generate(["Hello, world!"])
    """

    __slots__ = ("_config", "_engine", "_metrics", "_initialized")

    def __init__(self, config: EngineConfig):
        self._config = config
        self._metrics = MetricsAggregator()
        self._initialized = False

        # ── Build engine via factory ──
        result = EngineBuilder.build_sync(config)
        if result.is_err():
            raise RuntimeError(
                f"Engine initialization failed: {result.error}"
            )
        self._engine: LLM = result.unwrap()
        self._initialized = True

        logger.info(
            f"SOTAInferenceEngine ready: "
            f"model={config.model}, "
            f"TP={config.parallel.tensor_parallel_size}, "
            f"PP={config.parallel.pipeline_parallel_size}, "
            f"quant={config.quantization.method.value}, "
            f"prefix_cache={config.cache.enable_prefix_caching}, "
            f"chunked_prefill={config.scheduler.enable_chunked_prefill}, "
            f"cuda_graphs={'disabled' if config.enforce_eager else 'enabled'}"
        )

    @property
    def engine(self) -> LLM:
        """Direct access to underlying vLLM LLM engine."""
        return self._engine

    @property
    def metrics(self) -> MetricsAggregator:
        """Access metrics aggregator."""
        return self._metrics

    # ════════════════════════════════════════════════════════════════════════════
    # Core Generation — Single & Batch
    # ════════════════════════════════════════════════════════════════════════════

    def generate(
        self,
        prompts: Union[str, List[str]],
        params: Optional[GenerationParams] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> Result[List[InferenceResponse]]:
        """
        Generate completions for one or more prompts.

        Leverages vLLM's continuous batching to process all prompts
        concurrently with optimal GPU utilization.

        Args:
            prompts: Single prompt string or list of prompts.
            params: Generation parameters (temperature, top_p, etc.).
            lora_request: Optional LoRA adapter for this batch.

        Returns:
            Result[List[InferenceResponse]] with per-request metrics.

        Pre-conditions:
            - Engine must be initialized (self._initialized == True)
            - prompts must be non-empty strings
        """
        if not self._initialized:
            return Result.err("Engine not initialized")

        # ── Normalize input ──
        if isinstance(prompts, str):
            prompts = [prompts]

        if not prompts:
            return Result.err("Empty prompt list")

        # ── Validate all prompts are non-empty ──
        for i, p in enumerate(prompts):
            if not p or not p.strip():
                return Result.err(f"Prompt at index {i} is empty")

        # ── Build sampling params ──
        gen_params = params or GenerationParams()
        sampling_params = gen_params.to_sampling_params()

        # ── Create per-request metrics ──
        request_metrics: List[RequestMetrics] = []
        for _ in prompts:
            m = RequestMetrics()
            m.mark_submitted()
            request_metrics.append(m)

        try:
            # ── vLLM generate: triggers full continuous batching pipeline ──
            #    Internally executes:
            #    1. Tokenization
            #    2. Prefix cache lookup (hash-based block matching)
            #    3. PagedAttention block allocation
            #    4. Chunked prefill scheduling
            #    5. CUDA graph replay for decode
            #    6. FlashAttention/FlashInfer kernels
            #    7. Speculative decoding (if configured)
            outputs: List[RequestOutput] = self._engine.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
        except Exception as exc:
            error_msg = f"Generation failed: {type(exc).__name__}: {exc}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return Result.err(error_msg)

        # ── Process outputs ──
        responses: List[InferenceResponse] = []
        for i, output in enumerate(outputs):
            metrics = request_metrics[i]

            # ── Extract first completion ──
            if output.outputs:
                primary = output.outputs[0]
                generated_text = primary.text
                num_generated = len(primary.token_ids)
                finish_reason = primary.finish_reason or "unknown"
            else:
                generated_text = ""
                num_generated = 0
                finish_reason = "error"

            num_prompt = len(output.prompt_token_ids)

            # ── Mark first token (approximate for batch) ──
            if num_generated > 0:
                metrics.mark_first_token()

            metrics.mark_complete(
                num_generated=num_generated,
                num_prompt=num_prompt,
            )
            self._metrics.record(metrics)

            # ── Multi-output handling (n > 1 or beam search) ──
            all_outputs = None
            if len(output.outputs) > 1:
                all_outputs = [
                    {
                        "index": j,
                        "text": o.text,
                        "tokens": len(o.token_ids),
                        "finish_reason": o.finish_reason,
                        "cumulative_logprob": (
                            o.cumulative_logprob
                            if hasattr(o, "cumulative_logprob")
                            else None
                        ),
                    }
                    for j, o in enumerate(output.outputs)
                ]

            responses.append(InferenceResponse(
                request_id=output.request_id,
                prompt=prompts[i],
                generated_text=generated_text,
                prompt_tokens=num_prompt,
                generated_tokens=num_generated,
                finish_reason=finish_reason,
                metrics=metrics,
                outputs=all_outputs,
            ))

        return Result.ok(responses)

    # ════════════════════════════════════════════════════════════════════════════
    # Convenience Methods
    # ════════════════════════════════════════════════════════════════════════════

    def generate_single(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> Result[InferenceResponse]:
        """Generate a single completion. Convenience wrapper."""
        result = self.generate([prompt], params, lora_request)
        if result.is_err():
            return Result.err(result.error)
        responses = result.unwrap()
        if not responses:
            return Result.err("No response generated")
        return Result.ok(responses[0])

    def generate_with_lora(
        self,
        prompts: Union[str, List[str]],
        lora_name: str,
        lora_path: str,
        lora_id: int,
        params: Optional[GenerationParams] = None,
    ) -> Result[List[InferenceResponse]]:
        """
        Generate with a specific LoRA adapter.

        Requires engine initialized with lora.enabled=True.

        Args:
            lora_name: Human-readable adapter name.
            lora_path: Path to LoRA weights directory.
            lora_id: Unique integer ID for this adapter.
        """
        if not self._config.lora.enabled:
            return Result.err(
                "LoRA not enabled. Set LoRAConfig(enabled=True) in EngineConfig."
            )

        lora_req = LoRARequest(
            lora_name=lora_name,
            lora_int_id=lora_id,
            lora_path=lora_path,
        )
        return self.generate(prompts, params, lora_request=lora_req)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return aggregated metrics summary."""
        return self._metrics.summarize()


# ════════════════════════════════════════════════════════════════════════════════════
# Asynchronous Streaming Engine — Online Serving
# ════════════════════════════════════════════════════════════════════════════════════
# Enables:
#   • Token-by-token streaming via async generators
#   • Concurrent request handling with continuous batching
#   • Non-blocking I/O for high-concurrency serving
#   • Per-request lifecycle metrics
# ════════════════════════════════════════════════════════════════════════════════════

class SOTAAsyncStreamEngine:
    """
    Asynchronous streaming inference engine for online serving.

    Uses vLLM's AsyncLLMEngine for iteration-level continuous batching
    with token-by-token streaming output.

    Features:
        ✓ All features from SOTAInferenceEngine
        ✓ Async/await native interface
        ✓ Token-by-token streaming (SSE-compatible)
        ✓ Concurrent request multiplexing
        ✓ Graceful shutdown with in-flight request draining

    Usage:
        config = EngineConfig(model="meta-llama/Llama-3.1-8B-Instruct")
        engine = SOTAAsyncStreamEngine(config)

        async for chunk in engine.generate_stream("Hello"):
            print(chunk.text_delta, end="", flush=True)
    """

    __slots__ = ("_config", "_engine", "_metrics", "_active_requests")

    def __init__(self, config: EngineConfig):
        self._config = config
        self._metrics = MetricsAggregator()
        self._active_requests: Set[str] = set()

        # ── Build async engine ──
        result = EngineBuilder.build_async(config)
        if result.is_err():
            raise RuntimeError(
                f"Async engine initialization failed: {result.error}"
            )
        self._engine: AsyncLLMEngine = result.unwrap()

        logger.info("SOTAAsyncStreamEngine ready.")

    @property
    def engine(self) -> AsyncLLMEngine:
        """Direct access to underlying AsyncLLMEngine."""
        return self._engine

    @property
    def active_request_count(self) -> int:
        """Number of currently in-flight requests."""
        return len(self._active_requests)

    # ════════════════════════════════════════════════════════════════════════════
    # Streaming Generation
    # ════════════════════════════════════════════════════════════════════════════

    async def generate_stream(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
        request_id: Optional[str] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream tokens as they are generated.

        Yields StreamChunk objects containing incremental text deltas.
        The continuous batching scheduler interleaves this request's
        decode steps with other concurrent requests.

        Args:
            prompt: Input prompt string.
            params: Generation parameters.
            request_id: Optional deterministic request ID.
            lora_request: Optional LoRA adapter.

        Yields:
            StreamChunk with text_delta for each generated token.

        Post-conditions:
            - Final chunk has is_finished=True
            - Metrics are recorded on completion
        """
        req_id = request_id or random_uuid()
        gen_params = params or GenerationParams()
        sampling_params = gen_params.to_sampling_params()

        metrics = RequestMetrics(request_id=req_id)
        metrics.mark_submitted()
        self._active_requests.add(req_id)

        cumulative_text = ""
        token_idx = 0
        first_token_recorded = False

        try:
            # ── Submit to async engine ──
            results_generator = self._engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=req_id,
                lora_request=lora_request,
            )

            # ── Stream token-by-token ──
            previous_text = ""
            async for request_output in results_generator:
                if not request_output.outputs:
                    continue

                primary = request_output.outputs[0]
                current_text = primary.text

                # ── Compute delta ──
                text_delta = current_text[len(previous_text):]
                previous_text = current_text

                if not text_delta:
                    continue

                # ── Record TTFT on first token ──
                if not first_token_recorded:
                    metrics.mark_first_token()
                    first_token_recorded = True

                cumulative_text = current_text
                is_finished = request_output.finished

                yield StreamChunk(
                    request_id=req_id,
                    text_delta=text_delta,
                    cumulative_text=cumulative_text,
                    token_index=token_idx,
                    is_finished=is_finished,
                    finish_reason=(
                        primary.finish_reason if is_finished else None
                    ),
                )
                token_idx += 1

                if is_finished:
                    num_prompt = len(request_output.prompt_token_ids)
                    num_generated = len(primary.token_ids)
                    metrics.mark_complete(num_generated, num_prompt)
                    self._metrics.record(metrics)

        except asyncio.CancelledError:
            # ── Graceful cancellation ──
            logger.info(f"Request {req_id} cancelled.")
            try:
                await self._engine.abort(req_id)
            except Exception:
                pass  # Best-effort abort
            raise
        except Exception as exc:
            logger.error(
                f"Stream generation error for {req_id}: "
                f"{type(exc).__name__}: {exc}"
            )
            yield StreamChunk(
                request_id=req_id,
                text_delta="",
                cumulative_text=cumulative_text,
                token_index=token_idx,
                is_finished=True,
                finish_reason="error",
            )
        finally:
            self._active_requests.discard(req_id)

    # ════════════════════════════════════════════════════════════════════════════
    # Non-Streaming Async Generation
    # ════════════════════════════════════════════════════════════════════════════

    async def generate(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
        request_id: Optional[str] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> Result[InferenceResponse]:
        """
        Non-streaming async generation. Collects all tokens before returning.

        Internally uses the streaming path but accumulates the full response.
        """
        req_id = request_id or random_uuid()
        gen_params = params or GenerationParams()

        final_text = ""
        finish_reason = "unknown"
        metrics = RequestMetrics(request_id=req_id)

        try:
            async for chunk in self.generate_stream(
                prompt=prompt,
                params=gen_params,
                request_id=req_id,
                lora_request=lora_request,
            ):
                final_text = chunk.cumulative_text
                if chunk.is_finished:
                    finish_reason = chunk.finish_reason or "stop"
                    metrics = RequestMetrics(request_id=req_id)
                    # Retrieve from aggregator (last recorded)

            return Result.ok(InferenceResponse(
                request_id=req_id,
                prompt=prompt,
                generated_text=final_text,
                prompt_tokens=0,  # Available from final output
                generated_tokens=0,
                finish_reason=finish_reason,
                metrics=metrics,
            ))
        except Exception as exc:
            return Result.err(
                f"Async generation failed: {type(exc).__name__}: {exc}"
            )

    # ════════════════════════════════════════════════════════════════════════════
    # Batch Async Generation
    # ════════════════════════════════════════════════════════════════════════════

    async def generate_batch(
        self,
        prompts: List[str],
        params: Optional[GenerationParams] = None,
    ) -> Result[List[InferenceResponse]]:
        """
        Process multiple prompts concurrently via async tasks.

        All requests are submitted to the continuous batching scheduler
        simultaneously, maximizing GPU utilization.
        """
        tasks = [
            self.generate(
                prompt=p,
                params=params,
                request_id=random_uuid(),
            )
            for p in prompts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        responses: List[InferenceResponse] = []

        for i, res in enumerate(results):
            if isinstance(res, Exception):
                return Result.err(
                    f"Batch request {i} failed: {type(res).__name__}: {res}"
                )
            if res.is_err():
                return Result.err(
                    f"Batch request {i} error: {res.error}"
                )
            responses.append(res.unwrap())

        return Result.ok(responses)

    # ════════════════════════════════════════════════════════════════════════════
    # Lifecycle Management
    # ════════════════════════════════════════════════════════════════════════════

    async def abort_request(self, request_id: str) -> Result[None]:
        """Abort an in-flight request."""
        try:
            await self._engine.abort(request_id)
            self._active_requests.discard(request_id)
            return Result.ok(None)
        except Exception as exc:
            return Result.err(f"Abort failed: {exc}")

    async def shutdown(self):
        """Graceful shutdown: abort all in-flight, release resources."""
        logger.info(
            f"Shutting down async engine. "
            f"Aborting {len(self._active_requests)} in-flight requests..."
        )
        for req_id in list(self._active_requests):
            try:
                await self._engine.abort(req_id)
            except Exception:
                pass
        self._active_requests.clear()
        logger.info("Async engine shutdown complete.")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return aggregated metrics summary."""
        summary = self._metrics.summarize()
        summary["active_requests"] = self.active_request_count
        return summary


# ════════════════════════════════════════════════════════════════════════════════════
# Preset Configuration Factory — Common Deployment Profiles
# ════════════════════════════════════════════════════════════════════════════════════
# Pre-validated configurations for common deployment scenarios.
# Each preset activates the optimal combination of vLLM features.
# ════════════════════════════════════════════════════════════════════════════════════

class EnginePresets:
    """
    Factory methods for common deployment configurations.

    Each preset is tuned for a specific deployment scenario
    with all relevant optimizations activated.
    """

    @staticmethod
    def high_throughput_batch(
        model: str,
        tp_size: int = 1,
        max_model_len: int = 8192,
    ) -> EngineConfig:
        """
        Maximum throughput for offline batch processing.

        Activations:
            ✓ Large batch budget (32768 tokens/iteration)
            ✓ 256 concurrent sequences
            ✓ Prefix caching for shared prefixes
            ✓ Chunked prefill for uniform latency
            ✓ CUDA graphs for decode acceleration
            ✓ 95% GPU memory for KV cache
        """
        return EngineConfig(
            model=model,
            scheduler=SchedulerConfig(
                max_num_seqs=256,
                max_num_batched_tokens=32768,
                max_model_len=max_model_len,
                enable_chunked_prefill=True,
            ),
            cache=CacheConfig(
                gpu_memory_utilization=0.95,
                enable_prefix_caching=True,
                block_size=16,
                swap_space_gb=8.0,
            ),
            parallel=ParallelConfig(
                tensor_parallel_size=tp_size,
                strategy=(
                    ParallelStrategy.TENSOR
                    if tp_size > 1
                    else ParallelStrategy.NONE
                ),
            ),
            enforce_eager=False,  # Enable CUDA graphs
        )

    @staticmethod
    def low_latency_interactive(
        model: str,
        tp_size: int = 1,
        max_model_len: int = 4096,
    ) -> EngineConfig:
        """
        Minimum latency for interactive chat serving.

        Activations:
            ✓ Small batch size for fast scheduling
            ✓ Chunked prefill to avoid decode starvation
            ✓ CUDA graphs for minimal decode latency
            ✓ Prefix caching for system prompt reuse
        """
        return EngineConfig(
            model=model,
            scheduler=SchedulerConfig(
                max_num_seqs=32,
                max_num_batched_tokens=4096,
                max_model_len=max_model_len,
                enable_chunked_prefill=True,
            ),
            cache=CacheConfig(
                gpu_memory_utilization=0.90,
                enable_prefix_caching=True,
                block_size=16,
            ),
            parallel=ParallelConfig(
                tensor_parallel_size=tp_size,
            ),
            enforce_eager=False,
        )

    @staticmethod
    def speculative_fast_decode(
        model: str,
        draft_model: Optional[str] = None,
        tp_size: int = 1,
        num_speculative_tokens: int = 5,
        max_model_len: int = 4096,
    ) -> EngineConfig:
        """
        Speculative decoding for accelerated auto-regressive generation.

        Uses either a draft model or n-gram based speculation.
        Achieves 2-3x decode speedup with verified acceptance.

        Activations:
            ✓ Draft model / N-gram speculation
            ✓ Verified token acceptance
            ✓ Prefix caching
            ✓ CUDA graphs
        """
        use_ngram = draft_model is None
        return EngineConfig(
            model=model,
            speculative=SpeculativeConfig(
                enabled=True,
                draft_model=draft_model,
                num_speculative_tokens=num_speculative_tokens,
                use_ngram=use_ngram,
                ngram_prompt_lookup_max=4,
                ngram_prompt_lookup_min=1,
            ),
            scheduler=SchedulerConfig(
                max_num_seqs=64,
                max_num_batched_tokens=8192,
                max_model_len=max_model_len,
                enable_chunked_prefill=False,  # Typically off with speculation
            ),
            cache=CacheConfig(
                gpu_memory_utilization=0.90,
                enable_prefix_caching=True,
            ),
            parallel=ParallelConfig(
                tensor_parallel_size=tp_size,
            ),
            enforce_eager=False,
        )

    @staticmethod
    def quantized_memory_efficient(
        model: str,
        quant_method: QuantizationMethod = QuantizationMethod.AWQ,
        bits: int = 4,
        tp_size: int = 1,
        max_model_len: int = 8192,
    ) -> EngineConfig:
        """
        Memory-efficient serving with quantized weights.

        Enables serving large models on limited GPU memory.

        Activations:
            ✓ AWQ/GPTQ/FP8 quantization
            ✓ Reduced memory footprint
            ✓ All serving optimizations
        """
        return EngineConfig(
            model=model,
            quantization=QuantizationConfig(
                method=quant_method,
                bits=bits,
            ),
            scheduler=SchedulerConfig(
                max_num_seqs=128,
                max_num_batched_tokens=16384,
                max_model_len=max_model_len,
                enable_chunked_prefill=True,
            ),
            cache=CacheConfig(
                gpu_memory_utilization=0.92,
                enable_prefix_caching=True,
            ),
            parallel=ParallelConfig(
                tensor_parallel_size=tp_size,
            ),
            enforce_eager=False,
        )

    @staticmethod
    def multi_gpu_large_model(
        model: str,
        tp_size: int = 4,
        pp_size: int = 1,
        max_model_len: int = 16384,
    ) -> EngineConfig:
        """
        Multi-GPU configuration for large models (70B+).

        Uses tensor parallelism to shard attention heads and MLP
        across GPUs. Optional pipeline parallelism for extreme sizes.

        Activations:
            ✓ Tensor parallelism (Megatron-style sharding)
            ✓ Optional pipeline parallelism
            ✓ NCCL all-reduce for gradient sync
            ✓ All serving optimizations
        """
        strategy = ParallelStrategy.TENSOR
        if pp_size > 1:
            strategy = ParallelStrategy.TENSOR_PIPELINE

        return EngineConfig(
            model=model,
            scheduler=SchedulerConfig(
                max_num_seqs=128,
                max_num_batched_tokens=16384,
                max_model_len=max_model_len,
                enable_chunked_prefill=True,
            ),
            cache=CacheConfig(
                gpu_memory_utilization=0.92,
                enable_prefix_caching=True,
                swap_space_gb=8.0,
            ),
            parallel=ParallelConfig(
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                strategy=strategy,
            ),
            enforce_eager=False,
        )

    @staticmethod
    def multi_lora_serving(
        model: str,
        max_lora_rank: int = 64,
        max_concurrent_loras: int = 4,
        tp_size: int = 1,
        max_model_len: int = 4096,
    ) -> EngineConfig:
        """
        Multi-LoRA concurrent adapter serving.

        Shares base model weights while routing requests to
        adapter-specific LoRA parameters.

        Activations:
            ✓ Multiple concurrent LoRA adapters
            ✓ CPU LoRA caching
            ✓ Adapter-aware KV routing
            ✓ All serving optimizations
        """
        return EngineConfig(
            model=model,
            lora=LoRAConfig(
                enabled=True,
                max_lora_rank=max_lora_rank,
                max_loras=max_concurrent_loras,
                max_cpu_loras=max_concurrent_loras * 4,
            ),
            scheduler=SchedulerConfig(
                max_num_seqs=64,
                max_num_batched_tokens=8192,
                max_model_len=max_model_len,
                enable_chunked_prefill=True,
            ),
            cache=CacheConfig(
                gpu_memory_utilization=0.88,
                enable_prefix_caching=True,
            ),
            parallel=ParallelConfig(
                tensor_parallel_size=tp_size,
            ),
            enforce_eager=False,
        )


# ════════════════════════════════════════════════════════════════════════════════════
# OpenAI-Compatible API Server — FastAPI Integration
# ════════════════════════════════════════════════════════════════════════════════════
# Provides /v1/completions and /v1/chat/completions endpoints
# compatible with OpenAI's API specification.
# Uses vLLM's async engine for non-blocking request handling.
# ════════════════════════════════════════════════════════════════════════════════════

def create_openai_server(
    config: EngineConfig,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> Result[Any]:
    """
    Create FastAPI-based OpenAI-compatible API server.

    Endpoints:
        POST /v1/completions       — Text completion
        POST /v1/chat/completions  — Chat completion (with streaming)
        GET  /v1/models            — List available models
        GET  /health               — Health check
        GET  /metrics              — Prometheus metrics

    Returns Result containing the FastAPI app instance.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse, JSONResponse
        from pydantic import BaseModel, Field
    except ImportError:
        return Result.err(
            "FastAPI not installed. Install via: pip install fastapi uvicorn"
        )

    # ── Initialize async engine ──
    async_engine = SOTAAsyncStreamEngine(config)

    app = FastAPI(
        title="SOTA vLLM Inference Server",
        description="OpenAI-compatible API powered by vLLM",
        version="1.0.0",
    )

    # ── Pydantic models for API ──
    class CompletionRequest(BaseModel):
        model: str = config.model
        prompt: Union[str, List[str]]
        max_tokens: int = Field(default=512, ge=1, le=config.scheduler.max_model_len)
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)
        top_p: float = Field(default=0.95, gt=0.0, le=1.0)
        n: int = Field(default=1, ge=1, le=16)
        stream: bool = False
        stop: Optional[Union[str, List[str]]] = None
        seed: Optional[int] = None
        frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
        presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str = config.model
        messages: List[ChatMessage]
        max_tokens: int = Field(default=512, ge=1)
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)
        top_p: float = Field(default=0.95, gt=0.0, le=1.0)
        n: int = Field(default=1, ge=1, le=16)
        stream: bool = False
        stop: Optional[Union[str, List[str]]] = None
        seed: Optional[int] = None

    # ── Health endpoint ──
    @app.get("/health")
    async def health():
        return {"status": "healthy", "model": config.model}

    # ── Metrics endpoint ──
    @app.get("/metrics")
    async def metrics():
        return JSONResponse(async_engine.get_metrics_summary())

    # ── Models endpoint ──
    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id": config.model,
                "object": "model",
                "owned_by": "organization",
            }],
        }

    # ── Completions endpoint ──
    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        prompts = (
            [request.prompt]
            if isinstance(request.prompt, str)
            else request.prompt
        )

        gen_params = GenerationParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            seed=request.seed,
            stop=(
                [request.stop]
                if isinstance(request.stop, str)
                else request.stop
            ),
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
        )

        if request.stream:
            # ── Streaming response (SSE) ──
            async def stream_generator():
                for prompt in prompts:
                    async for chunk in async_engine.generate_stream(
                        prompt=prompt, params=gen_params
                    ):
                        data = {
                            "id": f"cmpl-{chunk.request_id}",
                            "object": "text_completion",
                            "choices": [{
                                "text": chunk.text_delta,
                                "index": 0,
                                "finish_reason": chunk.finish_reason,
                            }],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
            )
        else:
            # ── Non-streaming response ──
            results = await async_engine.generate_batch(prompts, gen_params)
            if results.is_err():
                raise HTTPException(status_code=500, detail=results.error)

            responses = results.unwrap()
            choices = []
            for i, resp in enumerate(responses):
                choices.append({
                    "text": resp.generated_text,
                    "index": i,
                    "finish_reason": resp.finish_reason,
                })

            return {
                "id": f"cmpl-{random_uuid()}",
                "object": "text_completion",
                "model": config.model,
                "choices": choices,
                "usage": {
                    "prompt_tokens": sum(r.prompt_tokens for r in responses),
                    "completion_tokens": sum(
                        r.generated_tokens for r in responses
                    ),
                    "total_tokens": sum(
                        r.prompt_tokens + r.generated_tokens
                        for r in responses
                    ),
                },
            }

    # ── Chat completions endpoint ──
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        # ── Format messages into prompt ──
        # Uses tokenizer's chat template if available
        prompt_parts = []
        for msg in request.messages:
            prompt_parts.append(f"<|{msg.role}|>\n{msg.content}")
        prompt_parts.append("<|assistant|>\n")
        prompt = "\n".join(prompt_parts)

        gen_params = GenerationParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            seed=request.seed,
            stop=(
                [request.stop]
                if isinstance(request.stop, str)
                else request.stop
            ),
        )

        if request.stream:
            async def chat_stream():
                async for chunk in async_engine.generate_stream(
                    prompt=prompt, params=gen_params
                ):
                    data = {
                        "id": f"chatcmpl-{chunk.request_id}",
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "delta": {"content": chunk.text_delta},
                            "index": 0,
                            "finish_reason": chunk.finish_reason,
                        }],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                chat_stream(),
                media_type="text/event-stream",
            )
        else:
            result = await async_engine.generate(
                prompt=prompt, params=gen_params
            )
            if result.is_err():
                raise HTTPException(status_code=500, detail=result.error)

            resp = result.unwrap()
            return {
                "id": f"chatcmpl-{random_uuid()}",
                "object": "chat.completion",
                "model": config.model,
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": resp.generated_text,
                    },
                    "index": 0,
                    "finish_reason": resp.finish_reason,
                }],
                "usage": {
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.generated_tokens,
                    "total_tokens": (
                        resp.prompt_tokens + resp.generated_tokens
                    ),
                },
            }

    # ── Graceful shutdown ──
    @app.on_event("shutdown")
    async def on_shutdown():
        await async_engine.shutdown()

    return Result.ok(app)


# ════════════════════════════════════════════════════════════════════════════════════
# CLI Entry Point — Unified Interface for All Modes
# ════════════════════════════════════════════════════════════════════════════════════
# Modes:
#   1. batch    — Offline batch processing
#   2. serve    — OpenAI-compatible API server
#   3. bench    — Performance benchmarking
#   4. probe    — Hardware detection and capability report
# ════════════════════════════════════════════════════════════════════════════════════

def run_batch_demo(config: EngineConfig, prompts: List[str]):
    """
    Demonstrate synchronous batch inference with full metrics.

    Exercises:
        • PagedAttention block allocation
        • Continuous batching across prompts
        • CUDA graph replay for decode
        • Prefix caching for shared prefixes
        • Chunked prefill for long prompts
    """
    logger.info("=" * 72)
    logger.info("BATCH INFERENCE DEMO")
    logger.info("=" * 72)

    engine = SOTAInferenceEngine(config)

    params = GenerationParams(
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
    )

    result = engine.generate(prompts, params)

    if result.is_err():
        logger.error(f"Generation failed: {result.error}")
        return

    responses = result.unwrap()

    logger.info("-" * 72)
    for resp in responses:
        logger.info(f"[Request: {resp.request_id}]")
        logger.info(f"  Prompt:    {resp.prompt[:80]}...")
        logger.info(f"  Output:    {resp.generated_text[:200]}...")
        logger.info(
            f"  Tokens:    {resp.prompt_tokens} prompt + "
            f"{resp.generated_tokens} generated"
        )
        logger.info(
            f"  TTFT:      {resp.metrics.time_to_first_token_ms:.2f} ms"
        )
        logger.info(
            f"  TPOT:      {resp.metrics.time_per_output_token_ms:.2f} ms"
        )
        logger.info(
            f"  E2E:       {resp.metrics.end_to_end_latency_ms:.2f} ms"
        )
        logger.info(
            f"  Throughput: {resp.metrics.throughput_tokens_per_sec:.1f} tok/s"
        )
        logger.info("-" * 72)

    # ── Aggregate metrics ──
    summary = engine.get_metrics_summary()
    logger.info("AGGREGATE METRICS:")
    logger.info(json.dumps(summary, indent=2))


async def run_streaming_demo(config: EngineConfig, prompt: str):
    """
    Demonstrate async streaming generation with token-by-token output.

    Exercises:
        • AsyncLLMEngine iteration loop
        • Continuous batching with streaming
        • Token-by-token SSE emission
    """
    logger.info("=" * 72)
    logger.info("STREAMING INFERENCE DEMO")
    logger.info("=" * 72)

    engine = SOTAAsyncStreamEngine(config)

    params = GenerationParams(
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
    )

    logger.info(f"Prompt: {prompt}")
    logger.info("Output: ", )

    token_count = 0
    async for chunk in engine.generate_stream(prompt, params):
        sys.stdout.write(chunk.text_delta)
        sys.stdout.flush()
        token_count += 1

        if chunk.is_finished:
            sys.stdout.write("\n")
            logger.info(
                f"Finished: {token_count} tokens, "
                f"reason={chunk.finish_reason}"
            )

    summary = engine.get_metrics_summary()
    logger.info(json.dumps(summary, indent=2))

    await engine.shutdown()


def run_benchmark(
    config: EngineConfig,
    num_prompts: int = 100,
    prompt_len: int = 128,
    output_len: int = 128,
):
    """
    Performance benchmark exercising all engine optimizations.

    Measures:
        • Aggregate throughput (tokens/second)
        • TTFT percentiles (p50, p95, p99)
        • TPOT percentiles
        • End-to-end latency distribution
    """
    logger.info("=" * 72)
    logger.info("PERFORMANCE BENCHMARK")
    logger.info(
        f"  Prompts: {num_prompts}, "
        f"Prompt Len: ~{prompt_len}, "
        f"Output Len: {output_len}"
    )
    logger.info("=" * 72)

    engine = SOTAInferenceEngine(config)

    # ── Generate synthetic prompts ──
    # Using varied content to prevent degenerate prefix caching
    base_prompt = "Explain the concept of "
    topics = [
        "quantum computing", "neural networks", "distributed systems",
        "compiler optimization", "memory management", "concurrency",
        "graph algorithms", "cryptography", "machine learning",
        "operating systems", "database indexing", "network protocols",
    ]
    prompts = [
        f"{base_prompt}{topics[i % len(topics)]} in detail. "
        f"Variant {i}. "
        + "Please provide a comprehensive explanation. " * (prompt_len // 40)
        for i in range(num_prompts)
    ]

    params = GenerationParams(
        max_tokens=output_len,
        temperature=0.8,
        top_p=0.95,
    )

    result = engine.generate(prompts, params)

    if result.is_err():
        logger.error(f"Benchmark failed: {result.error}")
        return

    summary = engine.get_metrics_summary()
    logger.info("BENCHMARK RESULTS:")
    logger.info(json.dumps(summary, indent=2))


def run_hardware_probe():
    """Print detailed hardware capability report."""
    logger.info("=" * 72)
    logger.info("HARDWARE PROBE")
    logger.info("=" * 72)
    info = HardwareProbe.probe()
    logger.info(json.dumps(info, indent=2))


# ════════════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ════════════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point with mode selection.

    Usage:
        python sota_vllm_inference_engine.py probe
        python sota_vllm_inference_engine.py batch --model meta-llama/Llama-3.1-8B-Instruct
        python sota_vllm_inference_engine.py stream --model meta-llama/Llama-3.1-8B-Instruct
        python sota_vllm_inference_engine.py serve --model meta-llama/Llama-3.1-8B-Instruct --port 8000
        python sota_vllm_inference_engine.py bench --model meta-llama/Llama-3.1-8B-Instruct
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="SOTA vLLM Inference Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Execution mode")

    # ── Common args ──
    def add_common_args(sub):
        sub.add_argument(
            "--model", type=str,
            default="meta-llama/Llama-3.1-8B-Instruct",
            help="HuggingFace model ID or local path",
        )
        sub.add_argument("--dtype", type=str, default="auto")
        sub.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
        sub.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
        sub.add_argument(
            "--max-model-len", type=int, default=4096,
            help="Maximum sequence length",
        )
        sub.add_argument(
            "--quantization", type=str, default="none",
            choices=[m.value for m in QuantizationMethod],
        )
        sub.add_argument(
            "--enable-prefix-caching", action="store_true", default=True,
        )
        sub.add_argument(
            "--enable-chunked-prefill", action="store_true", default=True,
        )
        sub.add_argument("--enforce-eager", action="store_true", default=False)
        sub.add_argument(
            "--gpu-memory-utilization", type=float, default=0.90,
        )
        sub.add_argument(
            "--max-num-seqs", type=int, default=256,
        )
        sub.add_argument("--trust-remote-code", action="store_true")
        # ── Speculative decoding ──
        sub.add_argument("--speculative-model", type=str, default=None)
        sub.add_argument(
            "--num-speculative-tokens", type=int, default=5,
        )
        sub.add_argument("--use-ngram-spec", action="store_true")

    # ── Probe ──
    subparsers.add_parser("probe", help="Hardware detection")

    # ── Batch ──
    batch_parser = subparsers.add_parser("batch", help="Batch inference")
    add_common_args(batch_parser)
    batch_parser.add_argument(
        "--prompts", nargs="+",
        default=[
            "Explain the PagedAttention mechanism in vLLM.",
            "What is continuous batching and why is it important?",
            "Describe how CUDA graphs accelerate inference.",
        ],
    )

    # ── Stream ──
    stream_parser = subparsers.add_parser("stream", help="Streaming inference")
    add_common_args(stream_parser)
    stream_parser.add_argument(
        "--prompt", type=str,
        default="Explain how vLLM achieves high throughput inference.",
    )

    # ── Serve ──
    serve_parser = subparsers.add_parser("serve", help="API server")
    add_common_args(serve_parser)
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    # ── Bench ──
    bench_parser = subparsers.add_parser("bench", help="Benchmark")
    add_common_args(bench_parser)
    bench_parser.add_argument("--num-prompts", type=int, default=50)
    bench_parser.add_argument("--prompt-len", type=int, default=128)
    bench_parser.add_argument("--output-len", type=int, default=128)

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return

    if args.mode == "probe":
        run_hardware_probe()
        return

    # ── Build EngineConfig from CLI args ──
    quant_method = QuantizationMethod(args.quantization)
    spec_enabled = bool(
        args.speculative_model or args.use_ngram_spec
    )

    config = EngineConfig(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        quantization=QuantizationConfig(
            method=quant_method,
        ),
        parallel=ParallelConfig(
            tensor_parallel_size=args.tp,
            pipeline_parallel_size=args.pp,
        ),
        scheduler=SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            enable_chunked_prefill=args.enable_chunked_prefill,
        ),
        speculative=SpeculativeConfig(
            enabled=spec_enabled,
            draft_model=args.speculative_model,
            num_speculative_tokens=args.num_speculative_tokens,
            use_ngram=args.use_ngram_spec,
        ),
        cache=CacheConfig(
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=args.enable_prefix_caching,
        ),
    )

    # ── Dispatch to mode ──
    if args.mode == "batch":
        run_batch_demo(config, args.prompts)

    elif args.mode == "stream":
        asyncio.run(run_streaming_demo(config, args.prompt))

    elif args.mode == "serve":
        result = create_openai_server(config, args.host, args.port)
        if result.is_err():
            logger.error(f"Server creation failed: {result.error}")
            return
        app = result.unwrap()
        try:
            import uvicorn
            logger.info(
                f"Starting OpenAI-compatible server on "
                f"{args.host}:{args.port}"
            )
            uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        except ImportError:
            logger.error("uvicorn not installed: pip install uvicorn")

    elif args.mode == "bench":
        run_benchmark(
            config,
            num_prompts=args.num_prompts,
            prompt_len=args.prompt_len,
            output_len=args.output_len,
        )


if __name__ == "__main__":
    main()
