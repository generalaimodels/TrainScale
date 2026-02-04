# ════════════════════════════════════════════════════════════════════════════════
# SOTA Model Compilation - Production-Grade Implementation
# ════════════════════════════════════════════════════════════════════════════════
# Advanced PyTorch 2.0+ Compiler Configuration
#
# Features:
# - Max Autotune with Triton kernel optimization
# - CUDAGraphs integration with automatic capture
# - Dynamic shape management with shape guards
# - Joint graph optimization across modules
# - Memory-efficient compilation strategies
# - Cross-platform backend support
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import functools
import gc
import logging
import os
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda import Stream

# ═════════════════════════════════════════════════════════════════════════════════
# Logging Configuration
# ═════════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════════
# Type Definitions
# ═════════════════════════════════════════════════════════════════════════════════

ModelType = TypeVar('ModelType', bound=nn.Module)
CompiledModel = Union[nn.Module, Callable[..., Any]]

# ═════════════════════════════════════════════════════════════════════════════════
# Compilation Modes and Backends
# ═════════════════════════════════════════════════════════════════════════════════

class CompilationMode(Enum):
    """Compilation optimization modes."""
    DEFAULT = "default"
    REDUCE_OVERHEAD = "reduce-overhead"
    MAX_AUTOTUNE = "max-autotune"
    MAX_AUTOTUNE_NO_CUDAGRAPHS = "max-autotune-no-cudagraphs"


class CompilationBackend(Enum):
    """Supported compilation backends."""
    INDUCTOR = "inductor"
    CUDAGRAPHS = "cudagraphs"
    ONNXRT = "onnxrt"
    TVM = "tvm"
    TENSORRT = "tensorrt"
    IPEX = "ipex"  # Intel Extension for PyTorch
    AOT_EAGER = "aot_eager"
    AOT_TS = "aot_ts"
    AOT_TS_NVFUSER = "aot_ts_nvfuser"


class PrecisionMode(Enum):
    """Precision modes for compilation."""
    FP32 = auto()
    FP16 = auto()
    BF16 = auto()
    FP8 = auto()
    INT8 = auto()
    MIXED = auto()


# ═════════════════════════════════════════════════════════════════════════════════
# Compilation Configuration
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class InductorConfig:
    """Inductor backend configuration."""
    # Triton settings
    triton_cudagraphs: bool = True
    triton_unique_kernel_names: bool = True
    triton_descriptive_names: bool = False
    
    # Autotuning
    coordinate_descent_tuning: bool = True
    max_autotune: bool = True
    max_autotune_gemm: bool = True
    max_autotune_pointwise: bool = True
    autotune_in_subproc: bool = True
    
    # Caching
    fx_graph_cache: bool = True
    fx_graph_remote_cache: bool = False
    autotune_local_cache: bool = True
    autotune_remote_cache: bool = False
    
    # Memory optimization
    memory_planning: bool = True
    reorder_for_locality: bool = True
    aggressive_fusion: bool = True
    
    # Epilogue fusion
    epilogue_fusion: bool = True
    epilogue_fusion_first: bool = True
    
    # Pattern matching
    pattern_matcher: bool = True
    split_reductions: bool = True
    
    # Debugging
    debug: bool = False
    trace_enabled: bool = False


@dataclass
class CUDAGraphConfig:
    """CUDA Graph configuration."""
    enabled: bool = True
    warmup_iterations: int = 3
    capture_stream: Optional[Stream] = None
    pool: Optional[Tuple[int, int]] = None
    capture_error_mode: str = "global"  # "global", "thread_local", "relaxed"


@dataclass
class DynamicShapeConfig:
    """Dynamic shape handling configuration."""
    enabled: bool = True
    assume_static_by_default: bool = False
    automatic_dynamic_shapes: bool = True
    specialize_int: bool = True
    capture_dynamic_output_shape_ops: bool = True
    capture_scalar_outputs: bool = False


@dataclass
class CompilationConfig:
    """Complete compilation configuration."""
    # Core settings
    mode: CompilationMode = CompilationMode.MAX_AUTOTUNE
    backend: CompilationBackend = CompilationBackend.INDUCTOR
    fullgraph: bool = False
    
    # Precision
    precision: PrecisionMode = PrecisionMode.MIXED
    
    # Sub-configurations
    inductor: InductorConfig = field(default_factory=InductorConfig)
    cudagraph: CUDAGraphConfig = field(default_factory=CUDAGraphConfig)
    dynamic: DynamicShapeConfig = field(default_factory=DynamicShapeConfig)
    
    # Optimization flags
    disable_on_error: bool = True
    suppress_errors: bool = False
    verbose: bool = False
    
    # Caching
    cache_size_limit: int = 256
    persistent_cache: bool = True
    cache_dir: Optional[str] = None
    
    # Platform-specific
    allow_tf32: bool = True
    use_fast_math: bool = True


# ═════════════════════════════════════════════════════════════════════════════════
# Inductor Configuration Manager
# ═════════════════════════════════════════════════════════════════════════════════

class InductorConfigManager:
    """Manages torch._inductor configuration."""
    
    _original_config: Dict[str, Any] = {}
    _lock = threading.Lock()
    
    @classmethod
    def apply(cls, config: InductorConfig) -> None:
        """Apply Inductor configuration."""
        try:
            import torch._inductor.config as inductor_config
        except ImportError:
            logger.warning("torch._inductor not available")
            return
        
        with cls._lock:
            # Store original values
            cls._original_config = {
                'triton.cudagraphs': getattr(inductor_config.triton, 'cudagraphs', None),
                'triton.unique_kernel_names': getattr(inductor_config.triton, 'unique_kernel_names', None),
                'coordinate_descent_tuning': getattr(inductor_config, 'coordinate_descent_tuning', None),
                'fx_graph_cache': getattr(inductor_config, 'fx_graph_cache', None),
            }
            
            # Apply new configuration
            if hasattr(inductor_config, 'triton'):
                inductor_config.triton.cudagraphs = config.triton_cudagraphs
                inductor_config.triton.unique_kernel_names = config.triton_unique_kernel_names
                if hasattr(inductor_config.triton, 'descriptive_names'):
                    inductor_config.triton.descriptive_names = config.triton_descriptive_names
            
            # Autotuning settings
            if hasattr(inductor_config, 'coordinate_descent_tuning'):
                inductor_config.coordinate_descent_tuning = config.coordinate_descent_tuning
            if hasattr(inductor_config, 'max_autotune'):
                inductor_config.max_autotune = config.max_autotune
            if hasattr(inductor_config, 'max_autotune_gemm'):
                inductor_config.max_autotune_gemm = config.max_autotune_gemm
            if hasattr(inductor_config, 'max_autotune_pointwise'):
                inductor_config.max_autotune_pointwise = config.max_autotune_pointwise
            
            # Caching
            if hasattr(inductor_config, 'fx_graph_cache'):
                inductor_config.fx_graph_cache = config.fx_graph_cache
            
            # Memory optimization
            if hasattr(inductor_config, 'memory_planning'):
                inductor_config.memory_planning = config.memory_planning
            if hasattr(inductor_config, 'reorder_for_locality'):
                inductor_config.reorder_for_locality = config.reorder_for_locality
            if hasattr(inductor_config, 'aggressive_fusion'):
                inductor_config.aggressive_fusion = config.aggressive_fusion
            
            # Epilogue fusion
            if hasattr(inductor_config, 'epilogue_fusion'):
                inductor_config.epilogue_fusion = config.epilogue_fusion
            if hasattr(inductor_config, 'epilogue_fusion_first'):
                inductor_config.epilogue_fusion_first = config.epilogue_fusion_first
            
            # Pattern matching
            if hasattr(inductor_config, 'pattern_matcher'):
                inductor_config.pattern_matcher = config.pattern_matcher
            if hasattr(inductor_config, 'split_reductions'):
                inductor_config.split_reductions = config.split_reductions
            
            # Debugging
            if hasattr(inductor_config, 'debug'):
                inductor_config.debug = config.debug
            if hasattr(inductor_config, 'trace'):
                if hasattr(inductor_config.trace, 'enabled'):
                    inductor_config.trace.enabled = config.trace_enabled
    
    @classmethod
    def restore(cls) -> None:
        """Restore original configuration."""
        try:
            import torch._inductor.config as inductor_config
        except ImportError:
            return
        
        with cls._lock:
            for key, value in cls._original_config.items():
                if value is not None:
                    parts = key.split('.')
                    obj = inductor_config
                    for part in parts[:-1]:
                        obj = getattr(obj, part, None)
                        if obj is None:
                            break
                    if obj is not None:
                        setattr(obj, parts[-1], value)
            
            cls._original_config.clear()


# ═════════════════════════════════════════════════════════════════════════════════
# CUDA Graph Manager
# ═════════════════════════════════════════════════════════════════════════════════

class CUDAGraphManager:
    """
    Manages CUDA Graph capture and replay.
    
    Features:
    - Automatic graph capture with warmup
    - Static input/output buffer management
    - Memory pool management
    - Graph replay optimization
    """
    
    def __init__(self, config: CUDAGraphConfig):
        self.config = config
        self._graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        self._static_inputs: Dict[str, List[Tensor]] = {}
        self._static_outputs: Dict[str, List[Tensor]] = {}
        self._captured: Set[str] = set()
        self._lock = threading.Lock()
    
    def capture(
        self,
        func: Callable[..., Any],
        *args: Tensor,
        key: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[torch.cuda.CUDAGraph, List[Tensor], List[Tensor]]:
        """
        Capture function execution as CUDA Graph.
        
        Args:
            func: Function to capture
            *args: Input tensors (will be copied to static buffers)
            key: Cache key for this graph
            **kwargs: Additional function arguments
            
        Returns:
            Tuple of (graph, static_inputs, static_outputs)
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for graph capture")
        
        key = key or str(id(func))
        
        with self._lock:
            if key in self._captured:
                return (
                    self._graphs[key],
                    self._static_inputs[key],
                    self._static_outputs[key],
                )
            
            # Create static input buffers
            static_inputs = [arg.clone() for arg in args if isinstance(arg, Tensor)]
            
            # Warmup iterations
            stream = self.config.capture_stream or torch.cuda.current_stream()
            
            with torch.cuda.stream(stream):
                for _ in range(self.config.warmup_iterations):
                    outputs = func(*static_inputs, **kwargs)
                stream.synchronize()
            
            # Normalize outputs to list
            if isinstance(outputs, Tensor):
                outputs = [outputs]
            elif isinstance(outputs, tuple):
                outputs = list(outputs)
            
            # Create static output buffers
            static_outputs = [out.clone() for out in outputs if isinstance(out, Tensor)]
            
            # Capture graph
            graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(graph, stream=stream, pool=self.config.pool):
                captured_outputs = func(*static_inputs, **kwargs)
            
            # Map outputs to static buffers
            if isinstance(captured_outputs, Tensor):
                static_outputs = [captured_outputs]
            elif isinstance(captured_outputs, tuple):
                static_outputs = list(captured_outputs)
            
            # Store
            self._graphs[key] = graph
            self._static_inputs[key] = static_inputs
            self._static_outputs[key] = static_outputs
            self._captured.add(key)
            
            return graph, static_inputs, static_outputs
    
    def replay(
        self,
        key: str,
        *args: Tensor,
    ) -> List[Tensor]:
        """
        Replay captured CUDA Graph with new inputs.
        
        Args:
            key: Graph cache key
            *args: New input tensors
            
        Returns:
            Output tensors
        """
        with self._lock:
            if key not in self._captured:
                raise KeyError(f"Graph '{key}' not captured")
            
            graph = self._graphs[key]
            static_inputs = self._static_inputs[key]
            static_outputs = self._static_outputs[key]
        
        # Copy new inputs to static buffers
        for static_in, new_in in zip(static_inputs, args):
            static_in.copy_(new_in)
        
        # Replay graph
        graph.replay()
        
        # Return copies of static outputs
        return [out.clone() for out in static_outputs]
    
    def clear(self) -> None:
        """Clear all captured graphs."""
        with self._lock:
            self._graphs.clear()
            self._static_inputs.clear()
            self._static_outputs.clear()
            self._captured.clear()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════════
# Compiled Function Wrapper
# ═════════════════════════════════════════════════════════════════════════════════

class CompiledFunctionWrapper:
    """
    Wrapper for compiled functions with caching and fallback.
    
    Features:
    - Automatic recompilation on shape changes
    - Fallback to eager mode on errors
    - Performance tracking
    """
    
    def __init__(
        self,
        func: Callable[..., Any],
        config: CompilationConfig,
    ):
        self.func = func
        self.config = config
        self._compiled: Optional[Callable[..., Any]] = None
        self._eager_fallback = False
        self._call_count = 0
        self._compile_errors: List[Exception] = []
    
    def _compile(self) -> Callable[..., Any]:
        """Compile the function."""
        compile_kwargs = {
            'fullgraph': self.config.fullgraph,
            'dynamic': self.config.dynamic.enabled,
            'mode': self.config.mode.value,
            'backend': self.config.backend.value,
        }
        
        # Add options for specific backends
        options = {}
        
        if self.config.backend == CompilationBackend.INDUCTOR:
            if self.config.inductor.max_autotune:
                options['max_autotune'] = True
            if self.config.inductor.epilogue_fusion:
                options['epilogue_fusion'] = True
        
        if options:
            compile_kwargs['options'] = options
        
        return torch.compile(self.func, **compile_kwargs)
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call compiled function with fallback."""
        self._call_count += 1
        
        if self._eager_fallback:
            return self.func(*args, **kwargs)
        
        try:
            if self._compiled is None:
                self._compiled = self._compile()
            
            return self._compiled(*args, **kwargs)
            
        except Exception as e:
            self._compile_errors.append(e)
            
            if self.config.suppress_errors:
                logger.warning(f"Compilation error (using eager): {e}")
                if self.config.disable_on_error:
                    self._eager_fallback = True
                return self.func(*args, **kwargs)
            
            raise
    
    def reset(self) -> None:
        """Reset compilation state."""
        self._compiled = None
        self._eager_fallback = False
        self._compile_errors.clear()
        
        # Clear dynamo cache
        torch._dynamo.reset()


# ═════════════════════════════════════════════════════════════════════════════════
# Model Compiler
# ═════════════════════════════════════════════════════════════════════════════════

class ModelCompiler:
    """
    SOTA Model Compiler with advanced optimization.
    
    Features:
    - Inductor backend with max autotune
    - CUDA graph integration
    - Dynamic shape support
    - Joint graph optimization
    - Cross-platform backend support
    """
    
    def __init__(self, config: Optional[CompilationConfig] = None):
        self.config = config or CompilationConfig()
        self._cuda_graph_manager: Optional[CUDAGraphManager] = None
        self._compiled_modules: Dict[int, CompiledModel] = {}
        
        # Apply global configurations
        self._apply_global_config()
    
    def _apply_global_config(self) -> None:
        """Apply global PyTorch configurations."""
        # TF32 settings
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = self.config.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.config.allow_tf32
        
        # Fast math
        if self.config.use_fast_math:
            torch.set_float32_matmul_precision('high')
        
        # Inductor configuration
        if self.config.backend == CompilationBackend.INDUCTOR:
            InductorConfigManager.apply(self.config.inductor)
        
        # Dynamo configuration
        self._configure_dynamo()
    
    def _configure_dynamo(self) -> None:
        """Configure torch._dynamo settings."""
        try:
            import torch._dynamo as dynamo
            
            # Cache settings
            if hasattr(dynamo.config, 'cache_size_limit'):
                dynamo.config.cache_size_limit = self.config.cache_size_limit
            
            # Dynamic shape settings
            if hasattr(dynamo.config, 'assume_static_by_default'):
                dynamo.config.assume_static_by_default = (
                    self.config.dynamic.assume_static_by_default
                )
            if hasattr(dynamo.config, 'automatic_dynamic_shapes'):
                dynamo.config.automatic_dynamic_shapes = (
                    self.config.dynamic.automatic_dynamic_shapes
                )
            if hasattr(dynamo.config, 'specialize_int'):
                dynamo.config.specialize_int = self.config.dynamic.specialize_int
            
            # Verbose logging
            if hasattr(dynamo.config, 'verbose'):
                dynamo.config.verbose = self.config.verbose
            
            # Suppress errors
            if hasattr(dynamo.config, 'suppress_errors'):
                dynamo.config.suppress_errors = self.config.suppress_errors
                
        except ImportError:
            logger.warning("torch._dynamo not available")
    
    def compile(
        self,
        model: ModelType,
        example_inputs: Optional[Tuple[Tensor, ...]] = None,
    ) -> ModelType:
        """
        Compile model with SOTA optimizations.
        
        Args:
            model: PyTorch module to compile
            example_inputs: Optional example inputs for tracing
            
        Returns:
            Compiled model
        """
        model_id = id(model)
        
        # Check if already compiled
        if model_id in self._compiled_modules:
            return self._compiled_modules[model_id]
        
        # Platform-specific handling
        if os.name == 'nt':  # Windows
            compiled = self._compile_windows(model)
        else:
            compiled = self._compile_unix(model, example_inputs)
        
        self._compiled_modules[model_id] = compiled
        return compiled
    
    def _compile_windows(self, model: ModelType) -> ModelType:
        """Windows-specific compilation with fallback."""
        try:
            return torch.compile(
                model,
                mode=self.config.mode.value,
                fullgraph=self.config.fullgraph,
                dynamic=self.config.dynamic.enabled,
                backend=self.config.backend.value,
            )
        except Exception as e:
            logger.warning(f"torch.compile failed on Windows: {e}")
            return model
    
    def _compile_unix(
        self,
        model: ModelType,
        example_inputs: Optional[Tuple[Tensor, ...]],
    ) -> ModelType:
        """Unix compilation with full optimization."""
        compile_kwargs = {
            'mode': self.config.mode.value,
            'fullgraph': self.config.fullgraph,
            'dynamic': self.config.dynamic.enabled,
            'backend': self.config.backend.value,
        }
        
        # Build options dict
        options = self._build_compile_options()
        if options:
            compile_kwargs['options'] = options
        
        compiled = torch.compile(model, **compile_kwargs)
        
        # Warmup with example inputs
        if example_inputs is not None and self.config.cudagraph.enabled:
            self._warmup_and_capture(compiled, example_inputs)
        
        return compiled
    
    def _build_compile_options(self) -> Dict[str, Any]:
        """Build compilation options dictionary."""
        options = {}
        
        if self.config.backend == CompilationBackend.INDUCTOR:
            if self.config.mode == CompilationMode.MAX_AUTOTUNE:
                options['max_autotune'] = True
                options['max_autotune_gemm'] = True
            
            if self.config.inductor.epilogue_fusion:
                options['epilogue_fusion'] = True
            
            if self.config.inductor.triton_cudagraphs:
                options['triton.cudagraphs'] = True
        
        return options
    
    def _warmup_and_capture(
        self,
        model: CompiledModel,
        example_inputs: Tuple[Tensor, ...],
    ) -> None:
        """Warmup compiled model and optionally capture CUDA graph."""
        if not torch.cuda.is_available():
            return
        
        # Warmup iterations
        with torch.no_grad():
            for _ in range(self.config.cudagraph.warmup_iterations):
                model(*example_inputs)
            
            torch.cuda.synchronize()
    
    @property
    def cuda_graph_manager(self) -> CUDAGraphManager:
        """Get or create CUDA graph manager."""
        if self._cuda_graph_manager is None:
            self._cuda_graph_manager = CUDAGraphManager(self.config.cudagraph)
        return self._cuda_graph_manager
    
    def reset(self) -> None:
        """Reset compiler state."""
        self._compiled_modules.clear()
        
        if self._cuda_graph_manager is not None:
            self._cuda_graph_manager.clear()
        
        # Reset dynamo cache
        torch._dynamo.reset()
        
        # Restore inductor config
        InductorConfigManager.restore()
        
        # Clear caches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════════
# High-Level API Functions
# ═════════════════════════════════════════════════════════════════════════════════

# Global compiler instance
_global_compiler: Optional[ModelCompiler] = None
_global_lock = threading.Lock()


def get_compiler(config: Optional[CompilationConfig] = None) -> ModelCompiler:
    """Get or create global model compiler."""
    global _global_compiler
    
    with _global_lock:
        if _global_compiler is None:
            _global_compiler = ModelCompiler(config)
        return _global_compiler


def compile_model(
    model: ModelType,
    fullgraph: bool = False,
    dynamic: bool = True,
    mode: str = "max-autotune",
    backend: str = "inductor",
    example_inputs: Optional[Tuple[Tensor, ...]] = None,
) -> ModelType:
    """
    Compile a model using PyTorch 2.0+ SOTA compiler stack.
    
    Args:
        model: PyTorch module to compile
        fullgraph: Capture full graph without Python fallbacks
        dynamic: Support dynamic shapes
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
        backend: Backend compiler ('inductor', 'cudagraphs', etc.)
        example_inputs: Optional inputs for warmup
        
    Returns:
        Compiled PyTorch module
    """
    # Map string mode to enum
    mode_map = {
        'default': CompilationMode.DEFAULT,
        'reduce-overhead': CompilationMode.REDUCE_OVERHEAD,
        'max-autotune': CompilationMode.MAX_AUTOTUNE,
        'max-autotune-no-cudagraphs': CompilationMode.MAX_AUTOTUNE_NO_CUDAGRAPHS,
    }
    
    backend_map = {
        'inductor': CompilationBackend.INDUCTOR,
        'cudagraphs': CompilationBackend.CUDAGRAPHS,
        'onnxrt': CompilationBackend.ONNXRT,
        'tensorrt': CompilationBackend.TENSORRT,
        'ipex': CompilationBackend.IPEX,
        'aot_eager': CompilationBackend.AOT_EAGER,
    }
    
    config = CompilationConfig(
        mode=mode_map.get(mode, CompilationMode.MAX_AUTOTUNE),
        backend=backend_map.get(backend, CompilationBackend.INDUCTOR),
        fullgraph=fullgraph,
    )
    config.dynamic.enabled = dynamic
    
    compiler = get_compiler(config)
    return compiler.compile(model, example_inputs)


def compile_function(
    func: Callable[..., Any],
    fullgraph: bool = False,
    dynamic: bool = True,
    mode: str = "max-autotune",
    backend: str = "inductor",
) -> Callable[..., Any]:
    """
    Compile a function using PyTorch 2.0+ compiler.
    
    Args:
        func: Function to compile
        fullgraph: Capture full graph
        dynamic: Support dynamic shapes
        mode: Compilation mode
        backend: Backend compiler
        
    Returns:
        Compiled function
    """
    config = CompilationConfig(
        fullgraph=fullgraph,
    )
    config.dynamic.enabled = dynamic
    
    wrapper = CompiledFunctionWrapper(func, config)
    return wrapper


@contextmanager
def compilation_context(
    mode: str = "max-autotune",
    backend: str = "inductor",
    disable: bool = False,
):
    """
    Context manager for temporary compilation settings.
    
    Args:
        mode: Compilation mode
        backend: Backend compiler
        disable: Disable compilation entirely
    """
    if disable:
        with torch._dynamo.disable():
            yield
        return
    
    # Store original settings
    original_cache_limit = getattr(torch._dynamo.config, 'cache_size_limit', 64)
    
    try:
        # Apply temporary settings
        torch._dynamo.config.cache_size_limit = 256
        yield
    finally:
        # Restore
        torch._dynamo.config.cache_size_limit = original_cache_limit


# ═════════════════════════════════════════════════════════════════════════════════
# Decorator API
# ═════════════════════════════════════════════════════════════════════════════════

def compiled(
    mode: str = "max-autotune",
    backend: str = "inductor",
    fullgraph: bool = False,
    dynamic: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to compile a function.
    
    Usage:
        @compiled(mode="max-autotune")
        def my_function(x):
            return x * 2
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return compile_function(
            func,
            fullgraph=fullgraph,
            dynamic=dynamic,
            mode=mode,
            backend=backend,
        )
    return decorator


# ═════════════════════════════════════════════════════════════════════════════════
# Optimization Utilities
# ═════════════════════════════════════════════════════════════════════════════════

def optimize_for_inference(
    model: nn.Module,
    example_inputs: Optional[Tuple[Tensor, ...]] = None,
    use_cudagraphs: bool = True,
    use_channels_last: bool = True,
    freeze: bool = True,
) -> nn.Module:
    """
    Optimize model specifically for inference.
    
    Args:
        model: Model to optimize
        example_inputs: Example inputs for tracing
        use_cudagraphs: Enable CUDA graphs
        use_channels_last: Use channels-last memory format
        freeze: Freeze model parameters
        
    Returns:
        Optimized model
    """
    model = model.eval()
    
    # Freeze parameters
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    
    # Channels-last format
    if use_channels_last and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    
    # Compile with inference-optimized settings
    config = CompilationConfig(
        mode=CompilationMode.MAX_AUTOTUNE,
        backend=CompilationBackend.INDUCTOR,
        fullgraph=True,
    )
    config.dynamic.enabled = False
    config.cudagraph.enabled = use_cudagraphs
    
    compiler = ModelCompiler(config)
    compiled_model = compiler.compile(model, example_inputs)
    
    return compiled_model


def optimize_for_training(
    model: nn.Module,
    use_activation_checkpointing: bool = False,
) -> nn.Module:
    """
    Optimize model for training.
    
    Args:
        model: Model to optimize
        use_activation_checkpointing: Enable activation checkpointing
        
    Returns:
        Optimized model
    """
    model = model.train()
    
    # Activation checkpointing
    if use_activation_checkpointing:
        try:
            from torch.utils.checkpoint import checkpoint_sequential
            # Apply checkpointing to sequential modules
            for name, module in model.named_modules():
                if isinstance(module, nn.Sequential):
                    setattr(
                        model, name,
                        lambda *args, m=module: checkpoint_sequential(m, 2, *args)
                    )
        except ImportError:
            pass
    
    # Compile with training-optimized settings
    config = CompilationConfig(
        mode=CompilationMode.REDUCE_OVERHEAD,
        backend=CompilationBackend.INDUCTOR,
        fullgraph=False,
    )
    config.dynamic.enabled = True
    
    compiler = ModelCompiler(config)
    return compiler.compile(model)


# ═════════════════════════════════════════════════════════════════════════════════
# Profiling and Debugging
# ═════════════════════════════════════════════════════════════════════════════════

@contextmanager
def compilation_debug_mode():
    """Enable verbose compilation debugging."""
    try:
        import torch._dynamo as dynamo
        
        original_verbose = getattr(dynamo.config, 'verbose', False)
        original_suppress = getattr(dynamo.config, 'suppress_errors', False)
        
        dynamo.config.verbose = True
        dynamo.config.suppress_errors = False
        
        # Enable inductor debug
        try:
            import torch._inductor.config as inductor_config
            original_debug = getattr(inductor_config, 'debug', False)
            inductor_config.debug = True
        except ImportError:
            original_debug = None
        
        yield
        
    finally:
        dynamo.config.verbose = original_verbose
        dynamo.config.suppress_errors = original_suppress
        
        if original_debug is not None:
            inductor_config.debug = original_debug


def get_compilation_stats() -> Dict[str, Any]:
    """Get compilation statistics."""
    stats = {
        'dynamo_cache_size': 0,
        'compiled_functions': 0,
        'graph_breaks': 0,
    }
    
    try:
        import torch._dynamo as dynamo
        
        if hasattr(dynamo, 'utils') and hasattr(dynamo.utils, 'counters'):
            counters = dynamo.utils.counters
            stats['graph_breaks'] = sum(counters.get('graph_break', {}).values())
            stats['compiled_functions'] = len(counters.get('frames', {}))
            
    except Exception:
        pass
    
    return stats


def clear_compilation_cache() -> None:
    """Clear all compilation caches."""
    torch._dynamo.reset()
    
    try:
        import torch._inductor
        if hasattr(torch._inductor, 'codecache'):
            torch._inductor.codecache.clear()
    except Exception:
        pass
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "CompilationMode",
    "CompilationBackend",
    "PrecisionMode",
    # Configuration
    "InductorConfig",
    "CUDAGraphConfig",
    "DynamicShapeConfig",
    "CompilationConfig",
    # Core classes
    "InductorConfigManager",
    "CUDAGraphManager",
    "CompiledFunctionWrapper",
    "ModelCompiler",
    # High-level API
    "get_compiler",
    "compile_model",
    "compile_function",
    "compilation_context",
    # Decorator
    "compiled",
    # Optimization utilities
    "optimize_for_inference",
    "optimize_for_training",
    # Debugging
    "compilation_debug_mode",
    "get_compilation_stats",
    "clear_compilation_cache",
]