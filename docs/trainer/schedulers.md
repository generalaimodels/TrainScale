# SOTA Schedulers

The `trainer.schedulers` package provides learning rate schedules crucial for LLM convergence.

## 1. Cosine with Warmup (`get_cosine_schedule_with_warmup`)

The standard schedule for LLM training.

- **Phase 1 (Warmup)**: Linearly increases LR from 0 to max_lr over `num_warmup_steps`.
- **Phase 2 (Decay)**: Decreases LR following a cosine curve to 0 (or `min_lr`).

```python
from data_pipeline.trainer.schedulers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=100, 
    num_training_steps=1000
)
```

## 2. Cosine with Restarts (`get_cosine_with_hard_restarts_schedule_with_warmup`)

Useful for longer training runs or when training gets stuck.

- **Algorithm**: Performs multiple cosine cycles. At the end of each cycle, the LR "restarts" (jumps back up), potentially allowing the model to escape local minima.
- **Parameters**: `num_cycles` determines the frequency of restarts.

## 3. Inverse Square Root (`get_inverse_sqrt_schedule`)

Common in Transformer pretraining (e.g., original Attention Is All You Need paper).

- **Formula**: $lr = decay\_factor / \sqrt{max(step, warmup\_steps)}$

## 4. Polynomial Decay (`get_polynomial_decay_schedule_with_warmup`)

Flexible decay control.

- **Power**: Controls the steepness. `power=1.0` is linear decay.
- **End LR**: Can decay to a non-zero value.

## 5. Constant with Warmup (`get_constant_schedule_with_warmup`)

Simple baseline. Warms up then stays flat.
