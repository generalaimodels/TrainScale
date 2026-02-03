# ════════════════════════════════════════════════════════════════════════════════
# SOTA Model Compilation
# ════════════════════════════════════════════════════════════════════════════════
# Advanced PyTorch 2.0+ Compiler Configuration
#
# Features:
# - Max Autotune for Triton kernels
# - CUDAGraphs integration
# - Dynamic shape management
# - Joint graph optimization
# ════════════════════════════════════════════════════════════════════════════════

import torch
import os
from typing import Optional, Union, Callable

def compile_model(
    model: torch.nn.Module,
    fullgraph: bool = False,
    dynamic: bool = True,
    mode: str = "max-autotune",
    backend: str = "inductor",
) -> torch.nn.Module:
    """
    Compile a model using PyTorch 2.0+ SOTA compiler stack (Inductor).
    
    Args:
        model: PyTorch module to compile
        fullgraph: Whether to capture the full graph (no python fallbacks)
        dynamic: Support dynamic shapes (slightly slower but more flexible)
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
        backend: Backend compiler ('inductor', 'cudagraphs', etc.)
        
    Returns:
        Compiled PyTorch module
    """
    if os.name == 'nt':
        # Windows support for torch.compile is experimental/limited
        # We try to use it, but fallback gracefully or warn
        try:
            return torch.compile(model, mode=mode, fullgraph=fullgraph, dynamic=dynamic, backend=backend)
        except Exception as e:
            print(f"Warning: torch.compile failed on Windows: {e}")
            return model

    # Optimization flags for Inductor
    if mode == "max-autotune":
        torch._inductor.config.triton.cudagraphs = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True
    
    return torch.compile(
        model,
        mode=mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
        backend=backend,
    )

__all__ = ["compile_model"]
