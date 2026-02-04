# Trainer Module Documentation

The `trainer` module provides an **Above-SOTA** training infrastructure designed for scalability, mixed precision, and ease of use. It abstracts the complexities of the training loop while exposing full control via callbacks and configuration.

## 1. SOTA Trainer (`trainer/base.py`)

The `Trainer` class is the central engine. It supports:
- **Mixed Precision**: Automatic FP16/BF16 scaling.
- **Gradient Accumulation**: Simulate large batch sizes.
- **Distributed Training**: Native DDP/FSDP integration.
- **Profiling**: Integrated throughput and loss monitoring.

### Usage

```python
from data_pipeline.trainer import Trainer, load_training_config

args = load_training_config("train_config.yaml")

trainer = Trainer(
    model=my_model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds
)

# Start training
output = trainer.train()
```

---

## 2. Configuration (`trainer/training_config.py`)

The `TrainingConfig` dataclass manages strict typing for all hyperparameters.

### Key Parameters
- `training_mode`: `FULL_FINETUNE`, `QLORA`, `PRETRAINING`.
- `precision`: `bf16`, `fp16`, `tf32`.
- `optimizer`: `adamw_8bit` (bitsandbytes), `lion`, `sophia`.
- `scheduler`: `cosine_with_restarts`, `linear`.

### Presets

Helper functions to get SOTA defaults:
- `get_qlora_config(model_name)`: Optimized for QLoRA.
- `get_pretraining_config(model_name)`: Optimized for throughput.

---

## 3. Callbacks System (`trainer/callbacks/`)

The trainer is highly extensible via the `Callback` interface.

### Built-in Callbacks
- `ProgressCallback`: TQDM progress bars.
- `LoggingCallback`: TensorBoard/WandB logging.
- `CheckpointCallback`: Model saving with rotation (keep last N).
- `EarlyStoppingCallback`: Stop based on metric improvement.

### Custom Callback

```python
class MyCallback(Callback):
    def on_step_end(self, ctx: CallbackContext):
        if ctx.loss < 0.1:
            print("Loss is low!")
```

---

## 4. Distributed Training (`trainer/distributed/`)

Auto-detects the environment (Torchrun, SLURM, MPI) and configures the process group.

- **`DeviceManager`**: Handles GPU/TPU/MPS selection.
- **`GradientSynchronizer`**: Manages all-reduce operations.
- **`FSDP Integration`**: Zero-config Fully Sharded Data Parallel.
