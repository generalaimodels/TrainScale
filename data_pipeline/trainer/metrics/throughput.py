# ════════════════════════════════════════════════════════════════════════════════
# SOTA Training Metrics - Throughput Tracking
# ════════════════════════════════════════════════════════════════════════════════
# Performance metrics for training: samples/sec, tokens/sec, MFU, TFLOPS.
#
# Features:
#   - Samples per second tracking
#   - Tokens per second (critical for LLM training)
#   - Model FLOPs Utilization (MFU)
#   - Hardware-aware TFLOPS estimation
#   - Distributed aggregation
#
# Hardware Support:
#   - NVIDIA: A100 (312 TFLOPS), H100 (989 TFLOPS), H200 (989 TFLOPS)
#   - AMD: MI300X (1307 TFLOPS), MI325X (1307 TFLOPS)
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor

# ════════════════════════════════════════════════════════════════════════════════
# Hardware Constants - Peak TFLOPS (BF16/FP16)
# ════════════════════════════════════════════════════════════════════════════════

PEAK_TFLOPS: Dict[str, float] = {
    # NVIDIA (BF16 Tensor Core)
    "a100": 312.0,
    "a100_80gb": 312.0,
    "h100": 989.0,
    "h100_sxm": 989.0,
    "h100_pcie": 756.0,
    "h200": 989.0,
    "b100": 1800.0,  # Estimated
    "b200": 2250.0,  # Estimated
    
    # AMD (BF16 Matrix Core)
    "mi250": 383.0,
    "mi250x": 383.0,
    "mi300x": 1307.0,
    "mi325x": 1307.0,  # Similar to MI300X
    
    # Fallbacks
    "unknown": 100.0,
    "cpu": 1.0,
}


def _detect_gpu_model() -> str:
    """Detect GPU model name for TFLOPS lookup."""
    if not torch.cuda.is_available():
        return "cpu"
    
    name = torch.cuda.get_device_name(0).lower()
    
    # NVIDIA detection
    if "h200" in name:
        return "h200"
    if "h100" in name:
        return "h100_sxm" if "sxm" in name else "h100"
    if "a100" in name:
        return "a100_80gb" if "80" in name else "a100"
    if "b200" in name:
        return "b200"
    if "b100" in name:
        return "b100"
    
    # AMD detection
    if "mi325" in name:
        return "mi325x"
    if "mi300" in name:
        return "mi300x"
    if "mi250" in name:
        return "mi250x" if "x" in name else "mi250"
    
    return "unknown"


def _get_peak_tflops(gpu_model: Optional[str] = None) -> float:
    """Get peak TFLOPS for GPU model."""
    if gpu_model is None:
        gpu_model = _detect_gpu_model()
    return PEAK_TFLOPS.get(gpu_model, PEAK_TFLOPS["unknown"])


# ════════════════════════════════════════════════════════════════════════════════
# Throughput Tracker
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class ThroughputTracker:
    """
    SOTA throughput tracking for training performance.
    
    Tracks samples/sec, tokens/sec, and computes Model FLOPs Utilization (MFU).
    
    Example:
        >>> tracker = ThroughputTracker()
        >>> tracker.start()
        >>> for batch in dataloader:
        ...     # training step
        ...     tracker.update(samples=batch_size, tokens=batch_size * seq_len)
        >>> print(f"Throughput: {tracker.tokens_per_sec():,.0f} tok/s")
        >>> print(f"MFU: {tracker.compute_mfu(flops_per_token=6e9):.1%}")
    
    Attributes:
        peak_tflops: Peak hardware TFLOPS (auto-detected if None)
        window_size: Number of steps for rolling average
    """
    
    peak_tflops: Optional[float] = None
    window_size: int = 100
    
    # ───────────────────────────────────────────────────────────────────────────
    # Internal State
    # ───────────────────────────────────────────────────────────────────────────
    _total_samples: int = field(default=0, repr=False)
    _total_tokens: int = field(default=0, repr=False)
    _total_flops: float = field(default=0.0, repr=False)
    _start_time: Optional[float] = field(default=None, repr=False)
    _last_time: Optional[float] = field(default=None, repr=False)
    _step_count: int = field(default=0, repr=False)
    _window_times: list = field(default_factory=list, repr=False)
    _window_tokens: list = field(default_factory=list, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize tracker."""
        if self.peak_tflops is None:
            self.peak_tflops = _get_peak_tflops()
        self.reset()
    
    def reset(self) -> None:
        """Reset all counters."""
        self._total_samples = 0
        self._total_tokens = 0
        self._total_flops = 0.0
        self._start_time = None
        self._last_time = None
        self._step_count = 0
        self._window_times = []
        self._window_tokens = []
    
    def start(self) -> None:
        """Start timing (call at beginning of training)."""
        self._start_time = time.perf_counter()
        self._last_time = self._start_time
    
    def update(
        self,
        samples: int = 0,
        tokens: int = 0,
        flops: float = 0.0,
    ) -> None:
        """
        Update throughput counters.
        
        Args:
            samples: Number of samples processed this step
            tokens: Number of tokens processed this step
            flops: FLOPs consumed this step (optional)
        """
        current_time = time.perf_counter()
        
        if self._start_time is None:
            self.start()
        
        # Update totals
        self._total_samples += samples
        self._total_tokens += tokens
        self._total_flops += flops
        self._step_count += 1
        
        # Update rolling window
        if self._last_time is not None:
            elapsed = current_time - self._last_time
            self._window_times.append(elapsed)
            self._window_tokens.append(tokens)
            
            # Trim to window size
            if len(self._window_times) > self.window_size:
                self._window_times.pop(0)
                self._window_tokens.pop(0)
        
        self._last_time = current_time
    
    def elapsed_seconds(self) -> float:
        """Get total elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time
    
    def samples_per_sec(self) -> float:
        """
        Compute samples per second.
        
        Returns:
            Samples/second throughput
        """
        elapsed = self.elapsed_seconds()
        if elapsed == 0:
            return 0.0
        return self._total_samples / elapsed
    
    def tokens_per_sec(self) -> float:
        """
        Compute tokens per second.
        
        Returns:
            Tokens/second throughput (primary LLM metric)
        """
        elapsed = self.elapsed_seconds()
        if elapsed == 0:
            return 0.0
        return self._total_tokens / elapsed
    
    def tokens_per_sec_window(self) -> float:
        """
        Compute tokens per second over rolling window.
        
        Returns:
            Tokens/second over last N steps (smoother)
        """
        if not self._window_times:
            return self.tokens_per_sec()
        
        total_time = sum(self._window_times)
        total_tokens = sum(self._window_tokens)
        
        if total_time == 0:
            return 0.0
        return total_tokens / total_time
    
    def compute_mfu(
        self,
        flops_per_token: Optional[float] = None,
        model_params: Optional[int] = None,
        seq_len: Optional[int] = None,
    ) -> float:
        """
        Compute Model FLOPs Utilization (MFU).
        
        MFU = Achieved TFLOPS / Peak TFLOPS
        
        Args:
            flops_per_token: FLOPs per token (if known)
            model_params: Model parameter count (for estimation)
            seq_len: Sequence length (for estimation)
        
        Returns:
            MFU as fraction in [0, 1]
        
        Note:
            For transformer: FLOPs/token ≈ 6 * params (forward + backward)
        """
        tokens_per_sec = self.tokens_per_sec()
        
        if tokens_per_sec == 0 or self.peak_tflops == 0:
            return 0.0
        
        # Estimate FLOPs per token if not provided
        if flops_per_token is None:
            if model_params is not None:
                # Standard transformer estimate: 6 * params (2 for forward, 4 for backward)
                flops_per_token = 6.0 * model_params
            else:
                return 0.0
        
        # Achieved TFLOPS
        achieved_tflops = (tokens_per_sec * flops_per_token) / 1e12
        
        # MFU
        return achieved_tflops / self.peak_tflops
    
    def compute_tflops(self, flops_per_token: float) -> float:
        """
        Compute achieved TFLOPS.
        
        Args:
            flops_per_token: FLOPs per token
        
        Returns:
            Achieved TFLOPS
        """
        tokens_per_sec = self.tokens_per_sec()
        return (tokens_per_sec * flops_per_token) / 1e12
    
    @property
    def total_samples(self) -> int:
        """Get total samples processed."""
        return self._total_samples
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens processed."""
        return self._total_tokens
    
    @property
    def step_count(self) -> int:
        """Get number of update calls."""
        return self._step_count
    
    def sync(self, process_group: Optional["torch.distributed.ProcessGroup"] = None) -> None:
        """
        Synchronize counters across distributed ranks.
        
        Args:
            process_group: Process group for reduction (None = default)
        """
        if not torch.distributed.is_initialized():
            return
        
        buffer = torch.tensor(
            [self._total_samples, self._total_tokens],
            dtype=torch.int64,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        torch.distributed.all_reduce(
            buffer,
            op=torch.distributed.ReduceOp.SUM,
            group=process_group,
        )
        
        self._total_samples = buffer[0].item()
        self._total_tokens = buffer[1].item()
    
    def log_dict(self) -> Dict[str, float]:
        """
        Get metrics as dictionary for logging.
        
        Returns:
            Dict with throughput metrics
        """
        return {
            "throughput/samples_per_sec": self.samples_per_sec(),
            "throughput/tokens_per_sec": self.tokens_per_sec(),
            "throughput/tokens_per_sec_window": self.tokens_per_sec_window(),
            "throughput/total_samples": float(self._total_samples),
            "throughput/total_tokens": float(self._total_tokens),
            "throughput/elapsed_seconds": self.elapsed_seconds(),
        }
    
    def state_dict(self) -> Dict:
        """Get serializable state for checkpointing."""
        return {
            "total_samples": self._total_samples,
            "total_tokens": self._total_tokens,
            "total_flops": self._total_flops,
            "step_count": self._step_count,
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """Load state from checkpoint."""
        self._total_samples = state.get("total_samples", 0)
        self._total_tokens = state.get("total_tokens", 0)
        self._total_flops = state.get("total_flops", 0.0)
        self._step_count = state.get("step_count", 0)


# ════════════════════════════════════════════════════════════════════════════════
# Exports
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    "ThroughputTracker",
    "PEAK_TFLOPS",
    "_get_peak_tflops",
    "_detect_gpu_model",
]
