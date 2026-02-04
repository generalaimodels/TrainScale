# Metric System

The `trainer.metrics` package provides unified tracking of loss, accuracy, throughput, and gradients with automatic distributed synchronization.

## 1. Unified Aggregator (`TrainingMetrics`)

The `TrainingMetrics` class acts as the central hub, managing sub-trackers for specific domains. It handles distributed reduction (all-reduce) automatically.

### Usage

```python
from data_pipeline.trainer.metrics import TrainingMetrics

metrics = TrainingMetrics(distributed=True, model_params=7_000_000_000)

for step, batch in enumerate(dataloader):
    # ... training step ...
    metrics.update_step(
        loss=loss,
        logits=logits,
        labels=labels,
        batch_size=batch_size,
        num_tokens=num_tokens
    )
    
    if step % 100 == 0:
        logs = metrics.log_dict()
        # {'train/loss': 2.3, 'train/ppl': 9.97, 'train/mfu': 0.45, ...}
```

## 2. Loss Tracking (`LossTracker`)

- **Token-Weighted Averaging**: Correctly computes average loss across batches of varying sequence lengths.
- **EMA**: Tracks Exponential Moving Average of loss for smoother curves.
- **Perplexity**: Automatically calculates perplexity ($exp(loss)$).

## 3. Accuracy Tracking (`AccuracyTracker`)

- **Token-Level Accuracy**: Computes accuracy of next-token prediction.
- **Masking**: Properly ignores padding and masked tokens (`label == -100`).

## 4. Throughput & MFU (`ThroughputTracker`)

Tracks the speed of training and hardware utilization.

- **Tokens/sec**: The primary metric for LLM training speed.
- **MFU (Model FLOPs Utilization)**: The ratio of achieved TFLOPS to peak theoretical TFLOPS.
    - Uses hardware database to auto-detect peak TFLOPS for A100, H100, MI300X, etc.
    - Estimates FLOPs/token ($\approx 6N$) if not provided.

## 5. Gradient Tracking (`GradientTracker`)

Tracks gradient norms to monitor training stability and detect divergence (exploding gradients).
