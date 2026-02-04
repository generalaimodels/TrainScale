# Distributed Training Infrastructure

The `trainer.distributed` package provides a unified layer for scaling training from a single GPU to thousands of nodes.

## 1. Device Manager (`DeviceManager`)

Abstracts hardware heterogenity (CUDA, MPS, CPU) and provides auto-detection.

```python
from data_pipeline.trainer.distributed import DeviceManager

device_manager = DeviceManager.auto_detect()
model.to(device_manager.device)

print(f"BF16 Supported: {device_manager.supports_bf16()}")
print(f"FP8 Supported: {device_manager.supports_fp8()}")
```

## 2. Distributed Manager (`DistributedManager`)

Handles process group initialization, backend selection (NCCL/Gloo), and model wrapping.

### Usage

```python
from data_pipeline.trainer.distributed import setup_distributed, ParallelMode

config = DistributedConfig(
    mode=ParallelMode.FSDP,
    sharding_strategy=ShardingStrategy.FULL_SHARD
)

dist = setup_distributed(config)
model = dist.wrap_model(model)
```

## 3. Parallelism Modes

### Distributed Data Parallel (DDP)

- **Scenario**: Multi-GPU, model fits in single GPU memory.
- **Mechanism**: Replicates model on each GPU, synchronizes gradients via all-reduce.

### Fully Sharded Data Parallel (FSDP / ZeRO-3)

- **Scenario**: Massive models (7B+ parameters) that don't fit in one GPU.
- **Mechanism**: Shards optimizer states, gradients, and parameters across GPUs.
- **Sharding Strategies**:
    - `FULL_SHARD`: Max memory savings.
    - `HYBRID_SHARD`: Shards within node, replicates across nodes (good for large clusters).

### Tensor Parallelism (TP)

- **Scenario**: Model layers are too wide for a single GPU.
- **Mechanism**: Splits individual matrix multiplications across GPUs (Row/Column Parallel Linear).
- **Modules**: `ColumnParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding`.

## 4. Activation Checkpointing

Reduces memory usage by clearing intermediate activations during the forward pass and recomputing them during the backward pass.

```python
from data_pipeline.trainer.distributed import wrap_activation_checkpointing

model = wrap_activation_checkpointing(model, TransformerLayer)
```
