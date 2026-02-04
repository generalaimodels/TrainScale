# Callback System

The `trainer.callbacks` system allows deep customization of the training loop without modifying the core `Trainer` code.

## 1. Event System

Callbacks interact with the trainer via the `CallbackContext` object, which provides read/write access to the training state (loss, metrics, model, stopping flags).

### Events Hooks

| Hook | Description |
| :--- | :--- |
| `on_init_end` | Called after trainer initialization. |
| `on_train_begin` | Called before the training loop starts. |
| `on_step_begin` | Called before the forward/backward/optimizer step. |
| `on_step_end` | Called after the step completes (useful for logging). |
| `on_epoch_begin` | Called at the start of an epoch. |
| `on_epoch_end` | Called at the end of an epoch. |
| `on_save` | Called before a checkpoint is saved. |
| `on_load` | Called after a checkpoint is loaded. |

## 2. Built-in Callbacks

### Early Stopping (`EarlyStoppingCallback`)

Stops training when a metric stops improving.

```python
early_stop = EarlyStoppingCallback(
    monitor="eval_loss",
    patience=3,
    min_delta=0.001,
    mode="min",
    restore_best=True
)
```

### Checkpointing (`CheckpointCallback`)

Manages model saving policies.

- **Strategy**: Save by `step` or `epoch`.
- **Top-K**: Keep only the best $K$ checkpoints (`save_total_limit`).
- **Format**: Supports standard PyTorch `.pt` or HuggingFace `.safetensors`.

### Logging (`LoggingCallback`)

Unified interface for logging metrics to varying backends.

- **Backends**: `CONSOLE`, `TENSORBOARD`, `WANDB`.
- **Metrics**: Automatically logs throughput (steps/sec), loss, LR, and GPU stats.

### Progress (`ProgressCallback`)

Displays a rich TQDM progress bar with real-time metrics (loss, LR).

## 3. Creating Custom Callbacks

Inherit from `Callback` and implement desired hooks.

```python
from data_pipeline.trainer.callbacks import Callback, CallbackContext

class HaltOnNan(Callback):
    def on_step_end(self, ctx: CallbackContext):
        if ctx.loss != ctx.loss:  # NaN check
            print(f"NaN overflow detected at step {ctx.step}!")
            ctx.should_stop = True
```
