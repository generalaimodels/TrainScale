### FSDP ARCHITECTURE & OPTIMIZATION PROTOCOL

**Objective:** Maximize VRAM efficiency and training throughput for Gigantic-scale models (>100B params) via Full Sharding of Data Parallelism resources.
**Context:** ZeRO-3 Implementation.
**Hardware:** NVIDIA H100 Cluster / AMD MI300.

---

### 1. MEMORY SEMANTICS & SHARDING STRATEGY
FSDP breaks the redundancy inherent in DDP (Distributed Data Parallel) by sharding all model states across the logical device mesh.

*   **Parameter Sharding ($P$):** Model weights split across $N$ ranks. Each GPU holds $1/N$ parameters.
*   **Gradient Sharding ($G$):** Gradients computed locally, synchronized via `ReduceScatter`, stored as $1/N$ shards.
*   **Optimizer State Sharding ($OS$):** Adam/AdamW moments (FP32) reside strictly on the owner rank ($1/N$).

**Memory Footprint Equation:**
$$ M_{per\_gpu} = \frac{M_{model} + M_{gradients} + M_{optimizer}}{N_{gpus}} + M_{activation} $$

### 2. EXECUTION PIPELINE & COMMUNICATION PRIMITIVES
Leverages NCCL collectives to reconstruct layers dynamically.

#### A. Forward Pass (Unshard -> Compute -> Reshard)
1.  **`AllGather`:** Local rank requests missing parameter shards for the current layer group (Transformer Block).
2.  **Compute:** FP16/BF16 GEMM execution on full layer parameters.
3.  **Release:** Immediately free non-local parameters to release VRAM.

#### B. Backward Pass (Unshard -> Compute -> Sync -> Reshard)
1.  **`AllGather`:** Re-fetch full parameters for gradient computation.
2.  **Compute:** Calculate gradients w.r.t input and weights.
3.  **`ReduceScatter`:** Average gradients across ranks; each rank stores only its specific shard of the gradient.
4.  **Release:** Discard full parameters and non-local gradients.

### 3. LATENCY HIDING & STREAM PIPELINING
To saturate HBM bandwidth and prevent SM stalls, communication must overlap with computation.

*   **Backward Prefetching:** Issue `AllGather` for Layer $L-1$ while computing gradients for Layer $L$.
*   **CUDA Streams:**
    *   *Stream 0:* Compute Kernels (GEMM, Softmax).
    *   *Stream 1:* NCCL `AllGather` (H2D/D2D).
    *   *Stream 2:* NCCL `ReduceScatter` (D2D).
*   **Limit All-Gathers:** Configure `limit_all_gathers=True` to throttle VRAM usage spikes during prefetching.

### 4. PRECISION ENGINEERING (Mixed Precision)
Maintain numerical stability while maximizing TFLOPS.

*   **Master Weights:** FP32 (High precision for Optimizer Step).
*   **Compute/Comm Buffers:** BF16 (High dynamic range, lower bandwidth).
*   **Gradient Accumulation:** FP32 (to prevent underflow before reduction).

**Optimization:** Utilize hardware casting kernels (e.g., Stochastic Rounding if necessary) during the parameter update phase.

### 5. IMPLEMENTATION SPECIFICATION (PyTorch Native)

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)

# 1. PRECISION POLICY
# Use BF16 for params/gradients to maximize throughput on Ampere/Hopper
# Keep Master Weights in FP32 for stability
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# 2. SHARDING STRATEGY
# FULL_SHARD: ZeRO-3 (Params, Grads, Opt State sharded)
# HYBRID_SHARD: Shard within node, Replicate across nodes (Lower latency)
sharding_strat = ShardingStrategy.FULL_SHARD

# 3. WRAPPING POLICY
# critical for VRAM management. Wrap at Transformer Block level.
# This ensures we only gather one block's weights at a time.
def get_wrapper(layer_class):
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_class},
    )

# 4. KERNEL INITIALIZATION
def engineer_fsdp_model(model: nn.Module, layer_cls):
    return FSDP(
        model,
        auto_wrap_policy=get_wrapper(layer_cls),
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strat,
        device_id=torch.cuda.current_device(),
        # 5. MEMORY OPTIMIZATION
        cpu_offload=CPUOffload(offload_params=False), # Set True if VRAM constrained
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE, # Maximize overlap
        limit_all_gathers=True, # Throttle VRAM usage
        use_orig_params=True # Required for torch.compile() fusion
    )
```

### 6. ADVANCED OPTIMIZATIONS FOR LEAD DEVELOPERS

1.  **Activation Checkpointing (AC):** Integrate FSDP with AC. Store only inputs to the wrapped layer; recompute forward pass during backward pass.
    *   *Trade-off:* Saves massive VRAM ($O(\sqrt{N})$), costs ~33% extra compute.
2.  **CPU Offloading:** Utilize pinned memory to offload parameters/optimizer states to system RAM when GPU VRAM is critical. High PCIe bandwidth dependency.
3.  **Orig Params (`use_orig_params=True`):** Essential for `torch.compile`. Allows the compiler to see the original tensor shapes for fusion, rather than flattened shards.
4.  **Custom All-Reduce Kernels:** For expert architectures (MoE), override default process groups to perform hierarchical reductions (intra-node NVLink -> inter-node InfiniBand).

### 7. VALIDATION & PROFILING
*   **Nsight Systems:** Verify `ncclKernel` overlaps with `gemm` kernels. Look for gaps in the compute stream indicating communication blocking.
*   **Memory Fragmentation:** Monitor caching allocator. If fragmentation is high, tune `limit_all_gathers` or adjust the wrapping block size.


----
# FSDP (Fully Sharded Data Parallel) — Technical Deep Dive

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         FSDP MEMORY LIFECYCLE                                  │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   FORWARD PASS                          BACKWARD PASS                          │
│   ────────────                          ─────────────                          │
│   ┌─────────────┐                       ┌─────────────┐                        │
│   │ All-Gather  │ ◄── Shard → Full      │ All-Gather  │ ◄── Params            │
│   │   Params    │                       │   Params    │                        │
│   └──────┬──────┘                       └──────┬──────┘                        │
│          │                                     │                               │
│          ▼                                     ▼                               │
│   ┌─────────────┐                       ┌─────────────┐                        │
│   │  Compute    │                       │  Compute    │                        │
│   │  Forward    │                       │  Gradients  │                        │
│   └──────┬──────┘                       └──────┬──────┘                        │
│          │                                     │                               │
│          ▼                                     ▼                               │
│   ┌─────────────┐                       ┌─────────────┐                        │
│   │   Discard   │ ◄── Free Memory       │Reduce-Scatter│ ◄── Gradient Shard   │
│   │ Full Params │                       │  Gradients  │                        │
│   └─────────────┘                       └──────┬──────┘                        │
│                                                │                               │
│                                                ▼                               │
│                                         ┌─────────────┐                        │
│                                         │   Discard   │                        │
│                                         │ Full Params │                        │
│                                         └─────────────┘                        │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Memory Footprint Comparison

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    MEMORY PER GPU (Model Size: M)                        │
├────────────────────┬─────────────────────────────────────────────────────┤
│ Strategy           │ Parameters │ Gradients │ Optimizer │ Total          │
├────────────────────┼────────────┼───────────┼───────────┼────────────────┤
│ DDP                │     M      │     M     │    2M     │   4M           │
│ ZeRO-1 (Opt Shard) │     M      │     M     │   2M/N    │   2M + 2M/N    │
│ ZeRO-2 (Grad+Opt)  │     M      │    M/N    │   2M/N    │   M + 3M/N     │
│ ZeRO-3 / FSDP      │    M/N     │    M/N    │   2M/N    │   4M/N         │
├────────────────────┴─────────────────────────────────────────────────────┤
│ N = Number of GPUs | Optimizer State assumes Adam (2x params)            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Core Sharding Strategies

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    ModuleWrapPolicy,
)

# ═══════════════════════════════════════════════════════════════════════════
# SHARDING STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

class FSDPShardingConfig:
    """
    Sharding strategy selection based on cluster topology and model size.
    """
    
    STRATEGIES = {
        # Full sharding: Maximum memory efficiency, highest communication
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,          # ZeRO-3 equivalent
        
        # Shard gradients + optimizer states only
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,    # ZeRO-2 equivalent
        
        # No sharding within node, full shard across nodes
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,      # Intra-node AllReduce
        
        # Hybrid with prefetch optimization
        "_HYBRID_SHARD_ZERO2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        
        # No sharding (DDP equivalent)
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    
    @staticmethod
    def select_strategy(
        model_params_billions: float,
        gpu_memory_gb: float,
        num_gpus: int,
        intra_node_bandwidth_gbps: float = 600,  # NVLink
        inter_node_bandwidth_gbps: float = 100,  # InfiniBand
    ) -> ShardingStrategy:
        """
        Optimal strategy selection based on hardware topology.
        """
        memory_per_param_bytes = 18  # FP16 param + FP16 grad + FP32 optimizer
        required_memory_gb = (model_params_billions * 1e9 * memory_per_param_bytes) / 1e9
        available_memory_gb = gpu_memory_gb * num_gpus * 0.85  # 85% utilization target
        
        sharding_ratio = required_memory_gb / available_memory_gb
        bandwidth_ratio = intra_node_bandwidth_gbps / inter_node_bandwidth_gbps
        
        if sharding_ratio < 0.25:
            return ShardingStrategy.NO_SHARD  # DDP sufficient
        elif sharding_ratio < 0.5 and bandwidth_ratio > 4:
            return ShardingStrategy.SHARD_GRAD_OP  # ZeRO-2
        elif bandwidth_ratio > 6:
            return ShardingStrategy.HYBRID_SHARD  # Minimize cross-node traffic
        else:
            return ShardingStrategy.FULL_SHARD  # Maximum sharding
```

---

## Mixed Precision Configuration

```python
from torch.distributed.fsdp import MixedPrecision
import torch

# ═══════════════════════════════════════════════════════════════════════════
# PRECISION POLICIES
# ═══════════════════════════════════════════════════════════════════════════

class FSDPPrecisionPolicies:
    """
    Precision configurations optimized for different hardware generations.
    """
    
    # Ampere/Hopper: BF16 compute, FP32 reduce for stability
    BF16_MIXED = MixedPrecision(
        param_dtype=torch.bfloat16,      # Parameters stored in BF16
        reduce_dtype=torch.float32,       # All-reduce in FP32 for precision
        buffer_dtype=torch.bfloat16,      # Buffers (BatchNorm stats) in BF16
        cast_forward_inputs=True,         # Cast inputs to param_dtype
    )
    
    # FP16 with loss scaling (legacy GPUs)
    FP16_MIXED = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float16,
        cast_forward_inputs=True,
    )
    
    # Hopper FP8 (requires TransformerEngine integration)
    FP8_COMPUTE = MixedPrecision(
        param_dtype=torch.bfloat16,       # Master weights in BF16
        reduce_dtype=torch.bfloat16,      # Communication in BF16
        buffer_dtype=torch.bfloat16,
    )
    
    # Full precision (debugging/validation)
    FP32_FULL = MixedPrecision(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    
    @classmethod
    def get_policy(cls, gpu_arch: str, model_type: str = "transformer"):
        """
        Hardware-aware precision selection.
        """
        if gpu_arch in ["hopper", "ada"]:
            return cls.BF16_MIXED  # Native BF16 tensor cores
        elif gpu_arch == "ampere":
            return cls.BF16_MIXED if model_type == "transformer" else cls.FP16_MIXED
        else:
            return cls.FP16_MIXED  # Volta/Turing
```

---

## Advanced Wrapping Policies

```python
import functools
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    lambda_auto_wrap_policy,
    ModuleWrapPolicy,
)

# ═══════════════════════════════════════════════════════════════════════════
# AUTO-WRAP POLICIES FOR OPTIMAL SHARDING GRANULARITY
# ═══════════════════════════════════════════════════════════════════════════

class FSDPWrapPolicies:
    """
    Wrapping policies control sharding granularity.
    Finer granularity = more memory savings, more communication.
    """
    
    @staticmethod
    def transformer_policy(transformer_layer_cls):
        """
        Wrap each transformer layer as FSDP unit.
        Optimal for decoder-only LLMs.
        """
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_layer_cls},
        )
    
    @staticmethod
    def size_based_policy(min_params: int = 100_000_000):
        """
        Wrap modules exceeding parameter threshold.
        100M params ≈ 200MB in FP16.
        """
        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=min_params,
        )
    
    @staticmethod
    def custom_lambda_policy(module_classes: set):
        """
        Custom lambda-based wrapping for hybrid architectures.
        """
        def policy_fn(module, recurse, nonwrapped_numel):
            if recurse:
                return True
            return type(module) in module_classes
        
        return functools.partial(lambda_auto_wrap_policy, lambda_fn=policy_fn)
    
    @staticmethod
    def moe_aware_policy(expert_cls, attention_cls):
        """
        Specialized wrapping for Mixture-of-Experts models.
        Wrap experts individually for load-balanced sharding.
        """
        return ModuleWrapPolicy({expert_cls, attention_cls})


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE: LLAMA-STYLE MODEL WRAPPING
# ═══════════════════════════════════════════════════════════════════════════

def wrap_llama_model(model, world_size: int):
    """
    Production FSDP configuration for LLaMA architecture.
    """
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    
    # Determine optimal strategy based on model size
    total_params = sum(p.numel() for p in model.parameters())
    params_per_gpu = total_params / world_size
    
    # 2B params/GPU threshold for HYBRID_SHARD
    if params_per_gpu > 2e9:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    else:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    
    fsdp_config = {
        "sharding_strategy": sharding_strategy,
        "mixed_precision": FSDPPrecisionPolicies.BF16_MIXED,
        "auto_wrap_policy": FSDPWrapPolicies.transformer_policy(LlamaDecoderLayer),
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "forward_prefetch": True,
        "use_orig_params": True,  # Required for torch.compile compatibility
        "limit_all_gathers": True,  # Memory-bound optimization
        "cpu_offload": None,  # Enable for extreme memory pressure
    }
    
    return FSDP(model, **fsdp_config)
```

---

## Communication Optimization

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    FSDP COMMUNICATION PATTERNS                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  FORWARD PASS:                                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                     │
│  │ Layer 0 │───▶│ Layer 1 │───▶│ Layer 2 │───▶│ Layer N │                     │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘                     │
│       │              │              │              │                           │
│  AllGather(W₀)  AllGather(W₁)  AllGather(W₂)  AllGather(Wₙ)                   │
│       ▼              ▼              ▼              ▼                           │
│   Compute        Compute        Compute        Compute                         │
│       ▼              ▼              ▼              ▼                           │
│    Free(W₀)      Free(W₁)      Free(W₂)      Free(Wₙ)                         │
│                                                                                │
│  BACKWARD PASS:                                                                │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                     │
│  │ Layer N │◀───│Layer N-1│◀───│Layer N-2│◀───│ Layer 0 │                     │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘                     │
│       │              │              │              │                           │
│  AllGather(Wₙ)  AllGather(Wₙ₋₁) AllGather(Wₙ₋₂) AllGather(W₀)                 │
│       ▼              ▼              ▼              ▼                           │
│  Compute ∇Wₙ   Compute ∇Wₙ₋₁   Compute ∇Wₙ₋₂  Compute ∇W₀                     │
│       ▼              ▼              ▼              ▼                           │
│ ReduceScatter  ReduceScatter   ReduceScatter  ReduceScatter                    │
│    (∇Wₙ)         (∇Wₙ₋₁)         (∇Wₙ₋₂)        (∇W₀)                         │
│                                                                                │
├────────────────────────────────────────────────────────────────────────────────┤
│  PREFETCH STRATEGIES:                                                          │
│                                                                                │
│  BACKWARD_PRE:   Prefetch layer i-1 during backward of layer i                │
│  BACKWARD_POST:  Prefetch layer i-1 after backward of layer i                 │
│  FORWARD_PREFETCH: Prefetch layer i+1 during forward of layer i               │
└────────────────────────────────────────────────────────────────────────────────┘
```

```python
# ═══════════════════════════════════════════════════════════════════════════
# COMMUNICATION-COMPUTE OVERLAP OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════

class FSDPCommunicationOptimizer:
    """
    Strategies for overlapping communication with computation.
    """
    
    @staticmethod
    def configure_prefetch(
        model_layers: int,
        compute_time_ms: float,
        allgather_time_ms: float,
    ) -> dict:
        """
        Select prefetch strategy based on compute/communication ratio.
        """
        ratio = compute_time_ms / allgather_time_ms
        
        if ratio > 2.0:
            # Compute-bound: aggressive prefetch won't cause memory spike
            return {
                "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
                "forward_prefetch": True,
            }
        elif ratio > 1.0:
            # Balanced: conservative prefetch
            return {
                "backward_prefetch": BackwardPrefetch.BACKWARD_POST,
                "forward_prefetch": True,
            }
        else:
            # Communication-bound: minimize memory overhead
            return {
                "backward_prefetch": BackwardPrefetch.BACKWARD_POST,
                "forward_prefetch": False,
            }
    
    @staticmethod
    def configure_bucketing(param_count: int, world_size: int) -> dict:
        """
        Configure AllGather bucketing for optimal bandwidth utilization.
        """
        shard_size = param_count / world_size
        
        # Target bucket size: 25MB for optimal NCCL performance
        target_bucket_bytes = 25 * 1024 * 1024
        param_bytes = 2  # FP16/BF16
        
        return {
            "all_gather_bucket_size": target_bucket_bytes // param_bytes,
            "reduce_scatter_bucket_size": target_bucket_bytes // param_bytes,
        }
```

---

## CPU Offloading Configuration

```python
from torch.distributed.fsdp import CPUOffload

# ═══════════════════════════════════════════════════════════════════════════
# CPU OFFLOAD STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

class FSDPOffloadConfig:
    """
    CPU/NVMe offloading for memory-constrained scenarios.
    """
    
    # Offload parameters to CPU when not in use
    CPU_OFFLOAD_PARAMS = CPUOffload(offload_params=True)
    
    # No offloading (default)
    NO_OFFLOAD = CPUOffload(offload_params=False)
    
    @staticmethod
    def configure_offload(
        gpu_memory_gb: float,
        model_memory_gb: float,
        cpu_memory_gb: float,
        pcie_bandwidth_gbps: float = 32,  # PCIe 4.0 x16
    ) -> CPUOffload:
        """
        Determine offload strategy based on memory hierarchy.
        """
        memory_ratio = model_memory_gb / gpu_memory_gb
        
        if memory_ratio < 0.7:
            # Sufficient GPU memory
            return CPUOffload(offload_params=False)
        elif cpu_memory_gb > model_memory_gb * 2:
            # CPU has capacity for full model + gradients
            return CPUOffload(offload_params=True)
        else:
            # Insufficient resources - consider NVMe offload via DeepSpeed
            raise MemoryError(
                f"Insufficient memory: GPU={gpu_memory_gb}GB, "
                f"CPU={cpu_memory_gb}GB, Model={model_memory_gb}GB"
            )
```

---

## State Management & Checkpointing

```python
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig,
    LocalStateDictConfig,
)
from torch.distributed.checkpoint import (
    save,
    load,
    FileSystemReader,
    FileSystemWriter,
)

# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINT STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

class FSDPCheckpointManager:
    """
    Efficient checkpointing strategies for FSDP models.
    """
    
    @staticmethod
    def save_full_checkpoint(
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        path: str,
        rank: int,
    ):
        """
        Gather full state dict on rank 0 (memory intensive but portable).
        """
        full_state_config = FullStateDictConfig(
            offload_to_cpu=True,      # Offload to CPU during gathering
            rank0_only=True,          # Only rank 0 saves
        )
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_config):
            state_dict = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
            
            if rank == 0:
                torch.save({
                    "model": state_dict,
                    "optimizer": optim_state,
                }, path)
    
    @staticmethod
    def save_sharded_checkpoint(
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: str,
    ):
        """
        Distributed sharded checkpoint (scalable, requires all ranks for loading).
        """
        sharded_config = ShardedStateDictConfig(
            offload_to_cpu=True,
        )
        
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_config):
            state_dict = {"model": model.state_dict()}
            optim_state = {"optimizer": FSDP.optim_state_dict(model, optimizer)}
            
            save(
                state_dict=state_dict,
                storage_writer=FileSystemWriter(checkpoint_dir),
            )
            save(
                state_dict=optim_state,
                storage_writer=FileSystemWriter(f"{checkpoint_dir}/optimizer"),
            )
    
    @staticmethod
    def load_sharded_checkpoint(
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: str,
    ):
        """
        Load distributed checkpoint across all ranks.
        """
        sharded_config = ShardedStateDictConfig(offload_to_cpu=True)
        
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_config):
            state_dict = {"model": model.state_dict()}
            load(
                state_dict=state_dict,
                storage_reader=FileSystemReader(checkpoint_dir),
            )
            model.load_state_dict(state_dict["model"])
            
            optim_state = {"optimizer": FSDP.optim_state_dict(model, optimizer)}
            load(
                state_dict=optim_state,
                storage_reader=FileSystemReader(f"{checkpoint_dir}/optimizer"),
            )
            FSDP.optim_state_dict_to_load(model, optimizer, optim_state["optimizer"])
```

---

## Production Training Loop

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.cuda.amp import GradScaler
from contextlib import nullcontext

# ═══════════════════════════════════════════════════════════════════════════
# PRODUCTION FSDP TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

class FSDPTrainer:
    """
    Production-grade FSDP training with gradient accumulation,
    mixed precision, and activation checkpointing.
    """
    
    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp_scaler: bool = False,  # Only for FP16, not BF16
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_accum_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.scaler = GradScaler() if use_amp_scaler else None
        
    def train_step(
        self,
        batch: dict,
        step: int,
    ) -> dict:
        """
        Single training step with gradient accumulation.
        """
        is_accumulating = (step + 1) % self.grad_accum_steps != 0
        
        # Context manager for gradient sync control
        sync_context = (
            self.model.no_sync() if is_accumulating else nullcontext()
        )
        
        with sync_context:
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss / self.grad_accum_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        
        # Optimizer step at accumulation boundary
        if not is_accumulating:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                
            # Gradient clipping with FSDP
            grad_norm = self.model.clip_grad_norm_(self.max_grad_norm)
            
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
                
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            return {
                "loss": loss.item() * self.grad_accum_steps,
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "lr": self.scheduler.get_last_lr()[0],
            }
        
        return {"loss": loss.item() * self.grad_accum_steps}


# ═══════════════════════════════════════════════════════════════════════════
# ACTIVATION CHECKPOINTING INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

def apply_fsdp_activation_checkpointing(
    model: FSDP,
    transformer_layer_cls: type,
    checkpoint_impl: CheckpointImpl = CheckpointImpl.NO_REENTRANT,
):
    """
    Apply activation checkpointing to transformer layers.
    Reduces memory by ~60% at cost of ~30% compute overhead.
    """
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=checkpoint_impl,
    )
    
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda module: isinstance(module, transformer_layer_cls),
    )
```

---

## Hybrid Parallelism Integration

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    FSDP + TENSOR PARALLELISM (2D PARALLELISM)                  │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   Node 0                              Node 1                                   │
│   ┌─────────────────────────┐         ┌─────────────────────────┐             │
│   │  TP Group (intra-node)  │         │  TP Group (intra-node)  │             │
│   │  ┌─────┐ ┌─────┐        │         │  ┌─────┐ ┌─────┐        │             │
│   │  │GPU 0│ │GPU 1│        │         │  │GPU 4│ │GPU 5│        │             │
│   │  │TP=0 │ │TP=1 │        │         │  │TP=0 │ │TP=1 │        │             │
│   │  └──┬──┘ └──┬──┘        │         │  └──┬──┘ └──┬──┘        │             │
│   │     │       │           │         │     │       │           │             │
│   │  ┌─────┐ ┌─────┐        │         │  ┌─────┐ ┌─────┐        │             │
│   │  │GPU 2│ │GPU 3│        │         │  │GPU 6│ │GPU 7│        │             │
│   │  │TP=0 │ │TP=1 │        │         │  │TP=0 │ │TP=1 │        │             │
│   │  └─────┘ └─────┘        │         │  └─────┘ └─────┘        │             │
│   └───────────┬─────────────┘         └───────────┬─────────────┘             │
│               │                                   │                            │
│               └───────────────┬───────────────────┘                            │
│                               │                                                │
│                      FSDP Group (inter-node)                                   │
│                      All-Gather / Reduce-Scatter                               │
│                                                                                │
├────────────────────────────────────────────────────────────────────────────────┤
│  TP: Column/Row parallel for attention/FFN (NVLink bandwidth)                  │
│  FSDP: Weight sharding across TP groups (InfiniBand bandwidth)                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

# ═══════════════════════════════════════════════════════════════════════════
# 2D PARALLELISM: FSDP + TENSOR PARALLELISM
# ═══════════════════════════════════════════════════════════════════════════

def configure_2d_parallelism(
    model: torch.nn.Module,
    tp_size: int,
    dp_size: int,
) -> FSDP:
    """
    Configure 2D parallelism with FSDP sharding over TP groups.
    """
    # Create 2D device mesh: (DP, TP)
    device_mesh = init_device_mesh(
        "cuda",
        (dp_size, tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    
    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp"]
    
    # Apply tensor parallelism to attention and FFN
    for layer in model.layers:
        # Column parallel for Q, K, V projections
        parallelize_module(
            layer.self_attn,
            tp_mesh,
            {
                "q_proj": ColwiseParallel(),
                "k_proj": ColwiseParallel(),
                "v_proj": ColwiseParallel(),
                "o_proj": RowwiseParallel(),
            },
        )
        
        # Column parallel for FFN gate/up, row parallel for down
        parallelize_module(
            layer.mlp,
            tp_mesh,
            {
                "gate_proj": ColwiseParallel(),
                "up_proj": ColwiseParallel(),
                "down_proj": RowwiseParallel(),
            },
        )
    
    # Wrap with FSDP over DP dimension
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        device_mesh=dp_mesh,
        mixed_precision=FSDPPrecisionPolicies.BF16_MIXED,
        use_orig_params=True,
    )
    
    return fsdp_model
```

---

## Performance Profiling

```python
import torch.profiler as profiler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# ═══════════════════════════════════════════════════════════════════════════
# FSDP PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════

class FSDPProfiler:
    """
    Profile FSDP communication and compute patterns.
    """
    
    @staticmethod
    def profile_training_step(
        model: FSDP,
        sample_batch: dict,
        output_dir: str = "./fsdp_profile",
    ):
        """
        Generate detailed trace for communication analysis.
        """
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            schedule=profiler.schedule(
                wait=1,
                warmup=2,
                active=3,
                repeat=1,
            ),
            on_trace_ready=profiler.tensorboard_trace_handler(output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            for _ in range(6):
                outputs = model(**sample_batch)
                loss = outputs.loss
                loss.backward()
                prof.step()
        
        return prof.key_averages().table(sort_by="cuda_time_total")
    
    @staticmethod
    def get_memory_stats(rank: int) -> dict:
        """
        Collect CUDA memory statistics per rank.
        """
        return {
            "rank": rank,
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        }
    
    @staticmethod
    def compute_mfu(
        model_params: int,
        batch_size: int,
        seq_len: int,
        step_time_s: float,
        gpu_flops: float = 312e12,  # H100 FP16 peak
    ) -> float:
        """
        Compute Model FLOPs Utilization (MFU).
        Assumes transformer: 6 * N * B * S FLOPs per forward+backward.
        """
        flops_per_step = 6 * model_params * batch_size * seq_len
        achieved_flops = flops_per_step / step_time_s
        mfu = achieved_flops / gpu_flops
        return mfu * 100  # Percentage
```

---

## Configuration Summary

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    FSDP CONFIGURATION DECISION MATRIX                          │
├─────────────────────────┬──────────────────────────────────────────────────────┤
│ Scenario                │ Recommended Configuration                            │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ Single Node (8 GPUs)    │ HYBRID_SHARD + BF16 + forward_prefetch               │
│ Multi-Node Homogeneous  │ FULL_SHARD + BACKWARD_PRE + limit_all_gathers        │
│ Memory Constrained      │ FULL_SHARD + CPU_OFFLOAD + activation_checkpointing  │
│ torch.compile           │ use_orig_params=True + NO_REENTRANT checkpointing    │
│ MoE Models              │ FULL_SHARD + custom wrap policy per expert           │
│ Very Long Sequences     │ HYBRID_SHARD + sequence_parallel (external)          │
├─────────────────────────┴──────────────────────────────────────────────────────┤
│                    COMMON PITFALLS                                             │
├────────────────────────────────────────────────────────────────────────────────┤
│ ✗ Wrapping too fine-grained → excessive AllGather operations                  │
│ ✗ Wrapping too coarse → insufficient memory savings                           │
│ ✗ Missing use_orig_params with torch.compile → compilation failures           │
│ ✗ FP16 without GradScaler → numerical instability                             │
│ ✗ Shared parameters without explicit handling → gradient corruption           │
│ ✗ Mixing FSDP with DDP in same model → undefined behavior                     │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Initialization Template

```python
# ═══════════════════════════════════════════════════════════════════════════
# PRODUCTION FSDP INITIALIZATION TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════

def init_fsdp_training(
    model_cls,
    model_config,
    world_size: int,
    local_rank: int,
):
    """
    Complete FSDP initialization for production training.
    """
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    
    # Create model on meta device to avoid CPU memory spike
    with torch.device("meta"):
        model = model_cls(model_config)
    
    # FSDP configuration
    fsdp_config = dict(
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
        auto_wrap_policy=transformer_auto_wrap_policy(
            transformer_layer_cls={model_config.layer_class}
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        use_orig_params=True,
        limit_all_gathers=True,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,  # Broadcast from rank 0
        param_init_fn=lambda m: m.to_empty(device=torch.cuda.current_device(), recurse=False),
    )
    
    # Wrap model
    model = FSDP(model, **fsdp_config)
    
    # Apply activation checkpointing
    apply_fsdp_activation_checkpointing(model, model_config.layer_class)
    
    # Optimizer with FSDP-aware parameter groups
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,  # CUDA-fused optimizer
    )
    
    return model, optimizer
```