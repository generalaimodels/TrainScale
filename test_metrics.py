#!/usr/bin/env python3
"""Quick test for SOTA metrics imports and basic functionality."""

from data_pipeline.trainer.metrics import (
    LossTracker,
    AccuracyTracker,
    ThroughputTracker,
    GradientTracker,
    TrainingMetrics,
    create_training_metrics,
    sync_metric,
    DistributedMetricBuffer,
    PEAK_TFLOPS,
    is_main_process,
)

print("=== All Imports Successful ===")

# Test LossTracker
loss = LossTracker()
loss.update(2.5, num_tokens=512)
loss.update(2.3, num_tokens=480)
print(f"LossTracker: avg={loss.compute_avg():.4f}, ppl={loss.compute_ppl():.2f}")

# Test ThroughputTracker
thru = ThroughputTracker()
thru.start()
thru.update(samples=4, tokens=4096)
print(f"ThroughputTracker: {thru.total_tokens} tokens tracked")

# Test TrainingMetrics
metrics = TrainingMetrics(distributed=False)
metrics.start()
metrics.update_step(loss=2.0, batch_size=4, num_tokens=2048, lr=1e-4)
result = metrics.compute(sync=False)
print(f"TrainingMetrics: loss={result['loss']:.4f}, ppl={result['ppl']:.2f}")

# Test PEAK_TFLOPS
print(f"PEAK_TFLOPS: {list(PEAK_TFLOPS.keys())[:5]}...")

print("\n=== All Tests Passed ===")
