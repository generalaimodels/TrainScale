### DISTRIBUTED DATA PARALLEL (DDP) ARCHITECTURE & OPTIMIZATION

**Objective:** Scale model training linearly across $N$ GPUs by replicating model states and partitioning input data, ensuring mathematical equivalence to local training via synchronized gradients.
**Paradigm:** Single-Program Multiple-Data (SPMD).
**Hardware:** Multi-GPU Nodes (NVLink/NVSwitch) & Multi-Node Clusters (InfiniBand/RoCE).

---

### 1. ARCHITECTURAL MECHANICS
DDP maintains a full copy of model parameters ($P$), optimizer states ($OS$), and gradients ($G$) on every rank.

*   **Forward Pass:** Each rank processes a unique micro-batch of data independent of other ranks.
*   **Backward Pass:** Gradients are computed locally.
*   **Synchronization Hook:** As gradients are generated (starting from the last layer), DDP triggers asynchronous communication hooks to synchronize $G$ across all ranks before the Optimizer Step.

### 2. COMMUNICATION OPTIMIZATION: BUCKETING & FUSION
To mitigate PCIe/Interconnect latency, DDP does not broadcast individual tensor gradients. It employs **Gradient Bucketing**.

*   **Mechanism:** Small tensors are fused into a contiguous flat buffer (Bucket).
*   **Thresholding:** Controlled by `bucket_cap_mb` (typically 25MB). Once a bucket is full, an async `AllReduce` call is dispatched.
*   **Latency Hiding:** By the time the backward pass reaches the first layer, the gradients for the last layers have already been synchronized (Overlap).

**NCCL Primitive:** `Ring-AllReduce` or `Tree-AllReduce`.
$$ G_{global} = \sum_{i=0}^{N} G_{rank\_i} $$

### 3. COMPUTE-COMMUNICATION OVERLAP STRATEGY
Maximizing SM (Streaming Multiprocessor) utilization requires masking communication overhead.

1.  **Stream 0 (Compute):** Executes Backward Kernel (GEMM/Element-wise).
2.  **Stream 1 (Comm):** Executes NCCL `AllReduce` on completed buckets.
3.  **Synchronization:** CPU thread orchestrates the launch. If computation is faster than bandwidth allows, stalls occur.

**Optimization Directive:**
*   Tune `bucket_cap_mb` based on model architecture.
    *   *Too Small:* High kernel launch overhead, latency bound.
    *   *Too Large:* Insufficient overlap, bandwidth bursts, blocking behavior.

### 4. PRECISION & MEMORY MANAGEMENT
*   **Mixed Precision:** Gradients are computed in BF16/FP16. `AllReduce` operations should occur in FP32 (via `fp32_reduce_scatter=False` if accumulation is needed) or BF16 for speed.
*   **Gradient Accumulation:** Perform $K$ local steps without synchronization. Triggers `no_sync` context manager to halt `AllReduce`, effectively increasing batch size and reducing communication cost by factor $K$.
*   **Static Graph:** Eliminate Python overhead by capturing the execution graph (CUDA Graphs) if topology is static.

### 5. IMPLEMENTATION SPECIFICATION (PyTorch Optimized)

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. PROCESS GROUP INITIALIZATION
# NCCL is mandatory for GPU backend
dist.init_process_group(backend="nccl")
local_rank = dist.get_rank() % torch.cuda.device_count()
torch.cuda.set_device(local_rank)

# 2. MODEL ARCHITECTURE
model = MyHighPerformanceTransformer().to(local_rank)

# 3. DDP CONSTRUCTOR OPTIMIZATION
# bucket_cap_mb: Tune based on layer sizes (25MB default, 50-100MB for LLMs)
# gradient_as_bucket_view: Reduces memory overhead by viewing gradients directly into buckets
# static_graph: Set True if architecture/control flow is constant (enables caching)
ddp_model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    bucket_cap_mb=50, 
    gradient_as_bucket_view=True,
    static_graph=False 
)

# 4. EXECUTION LOOP WITH GRADIENT ACCUMULATION
accumulation_steps = 4
optimizer.zero_grad()

for i, (inputs, labels) in enumerate(dataloader):
    # Context manager to prevent syncing until final accumulation step
    # Optimization: Reduces network traffic by factor of 'accumulation_steps'
    do_sync = (i + 1) % accumulation_steps == 0
    
    context = ddp_model if do_sync else ddp_model.no_sync()
    
    with context:
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
    
    if do_sync:
        optimizer.step()
        optimizer.zero_grad()
```

### 6. KERNEL LEAD DEVELOPER INSIGHTS

1.  **Bias Correction (Batch Norm):** Standard DDP computes BatchNorm statistics locally. This desynchronizes the model.
    *   *Solution:* Use `torch.nn.SyncBatchNorm` to synchronize running mean/variance across ranks. Increases communication overhead; strictly use only when necessary.
2.  **Buffer Reconstruction:** On the first iteration, DDP builds buckets. This is slow. Subsequent iterations are faster. Ensure benchmarking ignores `iter=0`.
3.  **Memory Imbalance:** DDP duplicates the model. For larger models exceeding single-GPU VRAM, strictly migrate to **FSDP** (ZeRO) or **Tensor Parallelism**. DDP is optimal *only* when $M_{model} \ll VRAM_{capacity}$.
4.  **Network Topology:** Ensure collective operations map to physical topology (NVLink within node, InfiniBand across nodes). Verify NCCL topology via `NCCL_DEBUG=INFO`.

### 7. VALIDATION PROTOCOL
*   **Throughput Metrics:** Measure Images/Sec or Tokens/Sec per GPU. Scaling efficiency should be >90% on NVLink.
*   **Determinism:** Verify loss curves match single-GPU execution (seed synchronization required).
*   **Nsight Systems:** Profile for "NCCL" gaps. If NCCL kernels are serialized after Compute kernels with no overlap, increase bucket size or check PCIe bandwidth saturation.


# Distributed Data Parallel (DDP) — Technical Deep Dive

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         DDP EXECUTION MODEL                                    │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   RANK 0                RANK 1                RANK 2                RANK 3     │
│   ┌─────────┐          ┌─────────┐          ┌─────────┐          ┌─────────┐  │
│   │ Model   │          │ Model   │          │ Model   │          │ Model   │  │
│   │ Replica │          │ Replica │          │ Replica │          │ Replica │  │
│   └────┬────┘          └────┬────┘          └────┬────┘          └────┬────┘  │
│        │                    │                    │                    │        │
│        ▼                    ▼                    ▼                    ▼        │
│   ┌─────────┐          ┌─────────┐          ┌─────────┐          ┌─────────┐  │
│   │ Batch 0 │          │ Batch 1 │          │ Batch 2 │          │ Batch 3 │  │
│   │ Forward │          │ Forward │          │ Forward │          │ Forward │  │
│   └────┬────┘          └────┬────┘          └────┬────┘          └────┬────┘  │
│        │                    │                    │                    │        │
│        ▼                    ▼                    ▼                    ▼        │
│   ┌─────────┐          ┌─────────┐          ┌─────────┐          ┌─────────┐  │
│   │Backward │          │Backward │          │Backward │          │Backward │  │
│   │   ∇W₀   │          │   ∇W₁   │          │   ∇W₂   │          │   ∇W₃   │  │
│   └────┬────┘          └────┬────┘          └────┬────┘          └────┬────┘  │
│        │                    │                    │                    │        │
│        └────────────────────┴────────────────────┴────────────────────┘        │
│                                    │                                           │
│                                    ▼                                           │
│                         ┌─────────────────────┐                                │
│                         │     ALL-REDUCE      │                                │
│                         │  ∇W = Σ(∇Wᵢ) / N    │                                │
│                         └──────────┬──────────┘                                │
│                                    │                                           │
│        ┌────────────────────┬──────┴──────┬────────────────────┐               │
│        ▼                    ▼             ▼                    ▼               │
│   ┌─────────┐          ┌─────────┐   ┌─────────┐          ┌─────────┐         │
│   │  W ←    │          │  W ←    │   │  W ←    │          │  W ←    │         │
│   │ W - η∇W │          │ W - η∇W │   │ W - η∇W │          │ W - η∇W │         │
│   └─────────┘          └─────────┘   └─────────┘          └─────────┘         │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Ring AllReduce Algorithm

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    RING ALLREDUCE COMMUNICATION PATTERN                        │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   PHASE 1: REDUCE-SCATTER (N-1 steps)                                         │
│   ─────────────────────────────────────                                        │
│                                                                                │
│   Step 0:          Step 1:          Step 2:          Final:                   │
│   ┌───┐           ┌───┐            ┌───┐            ┌───┐                     │
│   │ A │──────────▶│A+D│───────────▶│A+D │──────────▶│SUM│ Chunk 0            │
│   │ B │           │ B │            │+C+B│           │   │                     │
│   │ C │           │ C │            │    │           │   │                     │
│   │ D │           │ D │            │    │           │   │                     │
│   └─┬─┘           └─┬─┘            └─┬──┘           └───┘                     │
│     │               │                │                                         │
│     ▼               ▼                ▼                                         │
│   GPU 0 ──▶ GPU 1 ──▶ GPU 2 ──▶ GPU 3 ──▶ GPU 0                              │
│                                                                                │
│   Each GPU ends with fully reduced chunk i                                    │
│                                                                                │
│   PHASE 2: ALL-GATHER (N-1 steps)                                             │
│   ───────────────────────────────                                              │
│                                                                                │
│   Step 0:          Step 1:          Step 2:          Final:                   │
│   ┌───┐           ┌───┐            ┌───┐            ┌───┐                     │
│   │ S₀│──────────▶│S₀ │───────────▶│S₀ │──────────▶│S₀ │                     │
│   │   │           │S₃ │            │S₃ │           │S₁ │                     │
│   │   │           │   │            │S₂ │           │S₂ │                     │
│   │   │           │   │            │   │           │S₃ │                     │
│   └───┘           └───┘            └───┘           └───┘                     │
│                                                                                │
│   Communication Volume: 2(N-1)/N × Data Size ≈ 2× Data Size                   │
│   Bandwidth Optimal: Saturates all links simultaneously                        │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## DDP Core Implementation

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import ProcessGroup
import os

# ═══════════════════════════════════════════════════════════════════════════
# DDP INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

class DDPInitializer:
    """
    Production-grade DDP initialization with multi-backend support.
    """
    
    @staticmethod
    def init_process_group(
        backend: str = "nccl",
        init_method: str = "env://",
        world_size: int = None,
        rank: int = None,
        timeout_minutes: int = 30,
    ) -> ProcessGroup:
        """
        Initialize distributed process group.
        
        Backends:
        - nccl: GPU-to-GPU (optimal for NVIDIA)
        - gloo: CPU tensors, fallback
        - mpi: HPC environments
        """
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
        if rank is None:
            rank = int(os.environ.get("RANK", 0))
        
        # Configure NCCL environment for optimal performance
        os.environ.setdefault("NCCL_IB_DISABLE", "0")  # Enable InfiniBand
        os.environ.setdefault("NCCL_NET_GDR_LEVEL", "5")  # GPUDirect RDMA
        os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")  # NVLink preference
        
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=torch.distributed.default_pg_timeout 
                    if timeout_minutes == 30 
                    else torch.timedelta(minutes=timeout_minutes),
        )
        
        # Set device for current rank
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        
        return dist.group.WORLD
    
    @staticmethod
    def init_hybrid_backend():
        """
        Initialize with NCCL for GPU, Gloo for CPU operations.
        """
        dist.init_process_group(backend="nccl")
        
        # Create CPU process group for metadata operations
        cpu_group = dist.new_group(backend="gloo")
        
        return cpu_group


# ═══════════════════════════════════════════════════════════════════════════
# DDP MODEL WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

class DDPModelFactory:
    """
    Factory for creating optimized DDP model wrappers.
    """
    
    @staticmethod
    def wrap_model(
        model: torch.nn.Module,
        device_id: int,
        bucket_cap_mb: float = 25.0,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True,
        static_graph: bool = False,
        broadcast_buffers: bool = True,
    ) -> DDP:
        """
        Wrap model with optimized DDP configuration.
        
        Args:
            bucket_cap_mb: Gradient bucket size (25MB optimal for NCCL)
            find_unused_parameters: Enable for dynamic graphs (overhead)
            gradient_as_bucket_view: Reduce memory copies
            static_graph: Enable optimizations for fixed computation graphs
        """
        model = model.to(device_id)
        
        ddp_model = DDP(
            model,
            device_ids=[device_id],
            output_device=device_id,
            bucket_cap_mb=bucket_cap_mb,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
            broadcast_buffers=broadcast_buffers,
        )
        
        return ddp_model
    
    @staticmethod
    def configure_bucket_size(
        model_params_mb: float,
        gpu_count: int,
        network_bandwidth_gbps: float,
    ) -> float:
        """
        Compute optimal bucket size based on hardware topology.
        
        Larger buckets: Better bandwidth utilization
        Smaller buckets: Better compute-communication overlap
        """
        # Target: Balance latency and bandwidth
        # Optimal range: 10-50MB for most configurations
        
        if network_bandwidth_gbps >= 400:  # NVLink/NVSwitch
            return 25.0  # Default optimal
        elif network_bandwidth_gbps >= 100:  # InfiniBand HDR
            return 50.0  # Larger buckets for higher latency
        else:
            return 10.0  # Smaller buckets for low bandwidth
```

---

## Gradient Bucketing & Reduction

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    GRADIENT BUCKET REDUCTION TIMELINE                          │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  BACKWARD COMPUTATION          │ ALLREDUCE COMMUNICATION                       │
│  ──────────────────────        │ ────────────────────────                      │
│                                │                                               │
│  Layer N   ████████            │                                               │
│  Layer N-1 ░░░░████████        │                                               │
│  Layer N-2 ░░░░░░░░████████    │                                               │
│                                │                                               │
│  ──────── BUCKET 0 READY ──────┼──▶ ████████ AllReduce(Bucket0)               │
│                                │                                               │
│  Layer N-3 ░░░░░░░░░░░░████████│                                               │
│  Layer N-4 ░░░░░░░░░░░░░░░░████│███                                            │
│                                │                                               │
│  ──────── BUCKET 1 READY ──────┼──────────▶ ████████ AllReduce(Bucket1)       │
│                                │                                               │
│  Layer 1   ░░░░░░░░░░░░░░░░░░░░│░░░████████                                    │
│  Layer 0   ░░░░░░░░░░░░░░░░░░░░│░░░░░░░████████                                │
│                                │                                               │
│  ──────── BUCKET N READY ──────┼───────────────────▶ ████████ AllReduce(N)    │
│                                │                                               │
│  TIME ──────────────────────────────────────────────────────────────────────▶  │
│                                                                                │
│  KEY INSIGHT: Overlap backward computation with gradient communication         │
│  Buckets filled in REVERSE order (last layers compute gradients first)        │
└────────────────────────────────────────────────────────────────────────────────┘
```

```python
# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM GRADIENT HOOKS FOR PROFILING
# ═══════════════════════════════════════════════════════════════════════════

class GradientBucketProfiler:
    """
    Profile gradient bucket filling and AllReduce timing.
    """
    
    def __init__(self, ddp_model: DDP):
        self.ddp_model = ddp_model
        self.bucket_timings = []
        self.comm_timings = []
        
    def register_hooks(self):
        """
        Register communication hooks for profiling.
        """
        def timing_hook(state, bucket):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            fut = dist.all_reduce(bucket.buffer(), async_op=True)
            
            def callback(fut):
                end.record()
                torch.cuda.synchronize()
                self.comm_timings.append(start.elapsed_time(end))
            
            fut.then(callback)
            return fut
        
        self.ddp_model.register_comm_hook(
            state=None,
            hook=timing_hook,
        )
    
    def get_statistics(self) -> dict:
        """
        Return communication statistics.
        """
        if not self.comm_timings:
            return {}
        
        return {
            "mean_allreduce_ms": sum(self.comm_timings) / len(self.comm_timings),
            "max_allreduce_ms": max(self.comm_timings),
            "min_allreduce_ms": min(self.comm_timings),
            "total_comm_ms": sum(self.comm_timings),
            "bucket_count": len(self.comm_timings),
        }
```

---

## Communication Hooks

```python
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default_hooks
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
from torch.distributed.algorithms.ddp_comm_hooks import quantization_hooks

# ═══════════════════════════════════════════════════════════════════════════
# DDP COMMUNICATION HOOKS
# ═══════════════════════════════════════════════════════════════════════════

class DDPCommHooks:
    """
    Communication hooks for gradient compression and optimization.
    """
    
    @staticmethod
    def apply_fp16_compression(ddp_model: DDP):
        """
        Compress gradients to FP16 before AllReduce.
        Reduces communication by 50%.
        """
        ddp_model.register_comm_hook(
            state=dist.group.WORLD,
            hook=default_hooks.fp16_compress_hook,
        )
    
    @staticmethod
    def apply_bf16_compression(ddp_model: DDP):
        """
        BF16 compression for Ampere+ GPUs.
        Better numerical stability than FP16.
        """
        ddp_model.register_comm_hook(
            state=dist.group.WORLD,
            hook=default_hooks.bf16_compress_hook,
        )
    
    @staticmethod
    def apply_powersgd(
        ddp_model: DDP,
        matrix_approximation_rank: int = 4,
        start_powerSGD_iter: int = 1000,
    ):
        """
        PowerSGD low-rank gradient compression.
        Compression ratio: ~10x with minimal accuracy loss.
        
        Best for: Large models, limited bandwidth environments.
        """
        state = powerSGD.PowerSGDState(
            process_group=dist.group.WORLD,
            matrix_approximation_rank=matrix_approximation_rank,
            start_powerSGD_iter=start_powerSGD_iter,
            warm_start=True,
            use_error_feedback=True,
            min_compression_rate=2.0,
        )
        
        ddp_model.register_comm_hook(
            state=state,
            hook=powerSGD.powerSGD_hook,
        )
        
        return state
    
    @staticmethod
    def apply_batched_allreduce(ddp_model: DDP, batch_size: int = 4):
        """
        Batch multiple buckets into single AllReduce.
        Reduces launch overhead for small models.
        """
        ddp_model.register_comm_hook(
            state=dist.group.WORLD,
            hook=default_hooks.batched_all_reduce_hook,
        )


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM GRADIENT COMPRESSION HOOK
# ═══════════════════════════════════════════════════════════════════════════

class TopKCompressor:
    """
    Top-K sparsification with error feedback.
    Transmit only K largest gradient elements.
    """
    
    def __init__(self, compress_ratio: float = 0.01):
        self.compress_ratio = compress_ratio
        self.error_feedback = {}
    
    def __call__(self, state, bucket):
        tensor = bucket.buffer()
        device = tensor.device
        
        # Apply error feedback
        if id(bucket) in self.error_feedback:
            tensor.add_(self.error_feedback[id(bucket)])
        
        # Compute Top-K
        numel = tensor.numel()
        k = max(1, int(numel * self.compress_ratio))
        
        values, indices = torch.topk(tensor.abs().view(-1), k)
        
        # Store error for next iteration
        sparse_tensor = torch.zeros_like(tensor.view(-1))
        sparse_tensor.scatter_(0, indices, tensor.view(-1).gather(0, indices))
        self.error_feedback[id(bucket)] = tensor - sparse_tensor.view_as(tensor)
        
        # AllGather indices and values
        fut = dist.all_reduce(sparse_tensor, async_op=True)
        
        def decompress(fut):
            tensor.copy_(sparse_tensor.view_as(tensor))
            return tensor
        
        return fut.then(decompress)
```

---

## Multi-Node Configuration

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-NODE DDP TOPOLOGY                                     │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   NODE 0 (8 GPUs)                         NODE 1 (8 GPUs)                     │
│   ┌─────────────────────────────┐         ┌─────────────────────────────┐     │
│   │    NVSwitch Interconnect    │         │    NVSwitch Interconnect    │     │
│   │  ┌───┐ ┌───┐ ┌───┐ ┌───┐   │         │  ┌───┐ ┌───┐ ┌───┐ ┌───┐   │     │
│   │  │G0 │ │G1 │ │G2 │ │G3 │   │         │  │G8 │ │G9 │ │G10│ │G11│   │     │
│   │  └───┘ └───┘ └───┘ └───┘   │         │  └───┘ └───┘ └───┘ └───┘   │     │
│   │  ┌───┐ ┌───┐ ┌───┐ ┌───┐   │         │  ┌───┐ ┌───┐ ┌───┐ ┌───┐   │     │
│   │  │G4 │ │G5 │ │G6 │ │G7 │   │         │  │G12│ │G13│ │G14│ │G15│   │     │
│   │  └───┘ └───┘ └───┘ └───┘   │         │  └───┘ └───┘ └───┘ └───┘   │     │
│   └─────────────┬───────────────┘         └─────────────┬───────────────┘     │
│                 │                                       │                      │
│                 │     InfiniBand HDR (400 Gb/s)         │                      │
│                 └───────────────────────────────────────┘                      │
│                                                                                │
│   BANDWIDTH HIERARCHY:                                                         │
│   • Intra-GPU (HBM3):     3.35 TB/s                                           │
│   • Intra-Node (NVLink):  900 GB/s (H100)                                     │
│   • Inter-Node (IB HDR):  50 GB/s per link                                    │
│                                                                                │
│   OPTIMIZATION: Hierarchical AllReduce                                         │
│   1. Intra-node Reduce (NVLink) → 2. Inter-node AllReduce (IB)                │
└────────────────────────────────────────────────────────────────────────────────┘
```

```python
# ═══════════════════════════════════════════════════════════════════════════
# MULTI-NODE ENVIRONMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class MultiNodeConfig:
    """
    Configuration for multi-node DDP training.
    """
    
    @staticmethod
    def configure_nccl_environment():
        """
        Optimal NCCL settings for multi-node training.
        """
        nccl_config = {
            # Network settings
            "NCCL_IB_DISABLE": "0",           # Enable InfiniBand
            "NCCL_IB_HCA": "mlx5",            # Mellanox HCA
            "NCCL_IB_GID_INDEX": "3",         # RoCE v2
            "NCCL_NET_GDR_LEVEL": "5",        # GPUDirect RDMA
            "NCCL_NET_GDR_READ": "1",         # Enable GDR read
            
            # Topology settings
            "NCCL_TOPO_DUMP_FILE": "/tmp/nccl_topo.xml",
            "NCCL_GRAPH_DUMP_FILE": "/tmp/nccl_graph.xml",
            
            # Algorithm tuning
            "NCCL_ALGO": "Ring,Tree",         # Enable both algorithms
            "NCCL_PROTO": "Simple,LL128",     # Protocol selection
            
            # Buffer settings
            "NCCL_BUFFSIZE": "8388608",       # 8MB buffer
            "NCCL_NTHREADS": "512",           # Thread count
            
            # Debugging (disable in production)
            "NCCL_DEBUG": "WARN",
            "NCCL_DEBUG_SUBSYS": "INIT,NET",
        }
        
        for key, value in nccl_config.items():
            os.environ.setdefault(key, value)
    
    @staticmethod
    def create_hierarchical_groups(
        world_size: int,
        gpus_per_node: int,
    ) -> tuple:
        """
        Create process groups for hierarchical AllReduce.
        """
        rank = dist.get_rank()
        num_nodes = world_size // gpus_per_node
        
        # Intra-node group (local GPUs)
        local_ranks = list(range(
            (rank // gpus_per_node) * gpus_per_node,
            (rank // gpus_per_node + 1) * gpus_per_node
        ))
        local_group = dist.new_group(ranks=local_ranks)
        
        # Inter-node group (same local rank across nodes)
        cross_ranks = [i * gpus_per_node + (rank % gpus_per_node) 
                       for i in range(num_nodes)]
        cross_group = dist.new_group(ranks=cross_ranks)
        
        return local_group, cross_group


# ═══════════════════════════════════════════════════════════════════════════
# LAUNCH SCRIPTS
# ═══════════════════════════════════════════════════════════════════════════

"""
# Single Node Launch (torchrun)
torchrun --standalone --nproc_per_node=8 train.py

# Multi-Node Launch (2 nodes)
# Node 0:
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=node0_ip \
    --master_port=29500 \
    train.py

# Node 1:
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=1 \
    --master_addr=node0_ip \
    --master_port=29500 \
    train.py

# SLURM Launch
srun --nodes=2 --gpus-per-node=8 --ntasks-per-node=8 \
    python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py
"""
```

---

## Production Training Loop

```python
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext

# ═══════════════════════════════════════════════════════════════════════════
# DDP TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class DDPTrainer:
    """
    Production DDP training with gradient accumulation and mixed precision.
    """
    
    def __init__(
        self,
        model: DDP,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_dataloader: DataLoader,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        mixed_precision: bool = True,
        precision_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.grad_accum_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        self.device = next(model.parameters()).device
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Mixed precision setup
        self.use_amp = mixed_precision
        self.amp_dtype = precision_dtype
        self.scaler = torch.cuda.amp.GradScaler() if precision_dtype == torch.float16 else None
    
    def train_epoch(self, epoch: int) -> dict:
        """
        Execute single training epoch.
        """
        self.model.train()
        
        # Set epoch for DistributedSampler (ensures different shuffling per epoch)
        if hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(epoch)
        
        total_loss = 0.0
        num_steps = 0
        
        for step, batch in enumerate(self.train_dataloader):
            loss = self._training_step(batch, step)
            total_loss += loss
            num_steps += 1
        
        # Aggregate metrics across ranks
        avg_loss = self._reduce_metric(total_loss / num_steps)
        
        return {"loss": avg_loss, "epoch": epoch}
    
    def _training_step(self, batch: dict, step: int) -> float:
        """
        Single training step with gradient accumulation.
        """
        is_accumulating = (step + 1) % self.grad_accum_steps != 0
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Disable gradient sync during accumulation
        sync_context = self.model.no_sync() if is_accumulating else nullcontext()
        
        # Mixed precision context
        amp_context = (
            torch.cuda.amp.autocast(dtype=self.amp_dtype)
            if self.use_amp else nullcontext()
        )
        
        with sync_context, amp_context:
            outputs = self.model(**batch)
            loss = outputs.loss / self.grad_accum_steps
            
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        
        # Optimizer step at accumulation boundary
        if not is_accumulating:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )
            
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        
        return loss.item() * self.grad_accum_steps
    
    def _reduce_metric(self, value: float) -> float:
        """
        AllReduce scalar metric across ranks.
        """
        tensor = torch.tensor(value, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        return tensor.item()


# ═══════════════════════════════════════════════════════════════════════════
# DISTRIBUTED DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════

def create_distributed_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create DataLoader with DistributedSampler.
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=shuffle,
        drop_last=drop_last,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
```

---

## Static Graph Optimization

```python
# ═══════════════════════════════════════════════════════════════════════════
# STATIC GRAPH OPTIMIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

class StaticGraphDDP:
    """
    Enable DDP static graph optimizations for fixed computation graphs.
    
    Benefits:
    - Reuse AllReduce call graph
    - Pre-compute bucket assignments
    - Enable torch.compile compatibility
    """
    
    @staticmethod
    def wrap_model(
        model: torch.nn.Module,
        device_id: int,
    ) -> DDP:
        """
        Create DDP with static graph enabled.
        """
        ddp_model = DDP(
            model.to(device_id),
            device_ids=[device_id],
            static_graph=True,
            gradient_as_bucket_view=True,
        )
        
        return ddp_model
    
    @staticmethod
    def compile_with_ddp(
        ddp_model: DDP,
        backend: str = "inductor",
        mode: str = "reduce-overhead",
    ) -> DDP:
        """
        Apply torch.compile to DDP model.
        Requires static_graph=True.
        """
        # Compile the underlying module
        ddp_model._module = torch.compile(
            ddp_model.module,
            backend=backend,
            mode=mode,
            fullgraph=True,
        )
        
        return ddp_model


# ═══════════════════════════════════════════════════════════════════════════
# JOIN CONTEXT FOR UNEVEN INPUTS
# ═══════════════════════════════════════════════════════════════════════════

from torch.distributed.algorithms.join import Join

class UnevenInputHandler:
    """
    Handle uneven data distribution across ranks.
    Prevents deadlock when ranks have different batch counts.
    """
    
    @staticmethod
    def training_loop_with_join(
        ddp_model: DDP,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
    ):
        """
        Training loop that handles uneven inputs.
        """
        with Join([ddp_model]):
            for batch in dataloader:
                outputs = ddp_model(batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
```

---

## Performance Profiling

```python
import torch.profiler as profiler
from torch.profiler import tensorboard_trace_handler

# ═══════════════════════════════════════════════════════════════════════════
# DDP PERFORMANCE PROFILER
# ═══════════════════════════════════════════════════════════════════════════

class DDPProfiler:
    """
    Profile DDP communication and compute patterns.
    """
    
    @staticmethod
    def profile_training(
        train_fn,
        output_dir: str = "./ddp_profile",
        wait_steps: int = 1,
        warmup_steps: int = 2,
        active_steps: int = 3,
    ):
        """
        Generate profiling trace for analysis.
        """
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            schedule=profiler.schedule(
                wait=wait_steps,
                warmup=warmup_steps,
                active=active_steps,
                repeat=1,
            ),
            on_trace_ready=tensorboard_trace_handler(output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            for step in range(wait_steps + warmup_steps + active_steps):
                train_fn(step)
                prof.step()
        
        return prof
    
    @staticmethod
    def compute_metrics(
        model_params: int,
        batch_size: int,
        seq_len: int,
        step_time_s: float,
        world_size: int,
        gpu_peak_flops: float = 989e12,  # H100 BF16
    ) -> dict:
        """
        Compute training efficiency metrics.
        """
        # Global batch size
        global_batch = batch_size * world_size
        
        # Samples per second
        samples_per_second = global_batch / step_time_s
        
        # Tokens per second (for LLMs)
        tokens_per_second = samples_per_second * seq_len
        
        # Model FLOPs (transformer approximation)
        flops_per_sample = 6 * model_params * seq_len
        achieved_flops = flops_per_sample * samples_per_second
        
        # MFU (Model FLOPs Utilization)
        mfu = achieved_flops / (gpu_peak_flops * world_size) * 100
        
        return {
            "samples_per_second": samples_per_second,
            "tokens_per_second": tokens_per_second,
            "achieved_tflops": achieved_flops / 1e12,
            "mfu_percent": mfu,
            "step_time_ms": step_time_s * 1000,
        }
    
    @staticmethod
    def analyze_comm_overhead(
        ddp_model: DDP,
        sample_batch: dict,
        num_iterations: int = 10,
    ) -> dict:
        """
        Measure communication vs compute time.
        """
        torch.cuda.synchronize()
        
        # Profile without gradient sync
        ddp_model.require_backward_grad_sync = False
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iterations):
            output = ddp_model(**sample_batch)
            output.loss.backward()
        end.record()
        torch.cuda.synchronize()
        
        compute_time = start.elapsed_time(end) / num_iterations
        
        # Profile with gradient sync
        ddp_model.require_backward_grad_sync = True
        
        start.record()
        for _ in range(num_iterations):
            output = ddp_model(**sample_batch)
            output.loss.backward()
        end.record()
        torch.cuda.synchronize()
        
        total_time = start.elapsed_time(end) / num_iterations
        comm_time = total_time - compute_time
        
        return {
            "compute_ms": compute_time,
            "communication_ms": comm_time,
            "total_ms": total_time,
            "comm_overhead_percent": (comm_time / total_time) * 100,
            "compute_efficiency": (compute_time / total_time) * 100,
        }
```

---

## Checkpointing

```python
# ═══════════════════════════════════════════════════════════════════════════
# DDP CHECKPOINT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

class DDPCheckpointManager:
    """
    Efficient checkpoint save/load for DDP training.
    """
    
    @staticmethod
    def save_checkpoint(
        ddp_model: DDP,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        path: str,
        rank: int = 0,
    ):
        """
        Save checkpoint (only rank 0 saves to avoid redundancy).
        """
        if dist.get_rank() != rank:
            return
        
        checkpoint = {
            "model_state_dict": ddp_model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        }
        
        torch.save(checkpoint, path)
        
        # Barrier to ensure checkpoint is written before other ranks continue
        dist.barrier()
    
    @staticmethod
    def load_checkpoint(
        ddp_model: DDP,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        path: str,
        map_location: str = "cpu",
    ) -> int:
        """
        Load checkpoint on all ranks.
        """
        # All ranks load from same checkpoint
        checkpoint = torch.load(path, map_location=map_location)
        
        ddp_model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return checkpoint["epoch"]
    
    @staticmethod
    def save_sharded_checkpoint(
        ddp_model: DDP,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: str,
    ):
        """
        Save sharded checkpoint (each rank saves its shard).
        Useful for very large optimizers states.
        """
        rank = dist.get_rank()
        
        # Save model only from rank 0
        if rank == 0:
            torch.save(
                ddp_model.module.state_dict(),
                f"{checkpoint_dir}/model.pt"
            )
        
        # Each rank saves its optimizer shard
        torch.save(
            optimizer.state_dict(),
            f"{checkpoint_dir}/optimizer_rank{rank}.pt"
        )
        
        dist.barrier()
```

---

## DDP vs FSDP Comparison

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    DDP vs FSDP COMPARISON                                      │
├────────────────────────────┬───────────────────────┬───────────────────────────┤
│ Aspect                     │ DDP                   │ FSDP                      │
├────────────────────────────┼───────────────────────┼───────────────────────────┤
│ Model Replication          │ Full model per GPU    │ Sharded across GPUs       │
│ Memory per GPU             │ ~4M (params+grads+opt)│ ~4M/N                     │
│ Communication Pattern      │ AllReduce gradients   │ AllGather + ReduceScatter │
│ Communication Volume       │ 2× model size         │ 3× model size (worst)     │
│ Compute-Comm Overlap       │ Excellent             │ Good (with prefetch)      │
│ Implementation Complexity  │ Low                   │ Medium                    │
│ torch.compile Support      │ Native                │ Requires use_orig_params  │
│ Optimal Use Case           │ Model fits in GPU mem │ Large models              │
├────────────────────────────┴───────────────────────┴───────────────────────────┤
│ RECOMMENDATION:                                                                │
│ • Use DDP when: Model + gradients + optimizer fit in single GPU memory        │
│ • Use FSDP when: Memory-constrained or training very large models             │
│ • Hybrid: FSDP within node, DDP-like AllReduce across nodes                   │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Initialization Template

```python
# ═══════════════════════════════════════════════════════════════════════════
# PRODUCTION DDP TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """
    Complete DDP training script template.
    """
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    # Create model
    model = create_model().to(local_rank)
    
    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        bucket_cap_mb=25,
        gradient_as_bucket_view=True,
        static_graph=True,  # Enable if computation graph is fixed
    )
    
    # Optional: Apply communication hook
    model.register_comm_hook(
        state=dist.group.WORLD,
        hook=default_hooks.bf16_compress_hook,
    )
    
    # Optional: Compile model
    if static_graph:
        model = torch.compile(model, mode="reduce-overhead")
    
    # Create optimizer (after DDP wrap)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    
    # Create distributed dataloader
    dataloader = create_distributed_dataloader(dataset, batch_size=32)
    
    # Training loop
    trainer = DDPTrainer(model, optimizer, scheduler, dataloader)
    
    for epoch in range(num_epochs):
        metrics = trainer.train_epoch(epoch)
        
        if dist.get_rank() == 0:
            print(f"Epoch {epoch}: {metrics}")
            DDPCheckpointManager.save_checkpoint(model, optimizer, scheduler, epoch, "ckpt.pt")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```