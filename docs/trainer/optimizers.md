# SOTA Optimizers

The `trainer.optimizers` package provides a suite of "Above-SOTA" optimization algorithms, many featuring custom **Triton kernels** for maximum throughput and memory efficiency.

## 1. 8-Bit Adam (`Adam8bit`)

Drastically reduces memory usage by storing momentum states in 8-bit precision instead of 32-bit.

- **Algorithm**: Maintains 32-bit master weights but quantizes `exp_avg` and `exp_avg_sq` to `int8` with dynamic block-wise scaling.
- **Benefits**: ~75% reduction in optimizer state memory, enabling larger model fine-tuning (e.g., Llama-70B on 48GB cards).

```python
from data_pipeline.trainer.optimizers import Adam8bit

optimizer = Adam8bit(
    model.parameters(),
    lr=2e-4,
    min_8bit_size=4096  # Only quantize tensors larger than this
)
```

## 2. Lion (`Lion`)

The "Evolved Sign Momentum" optimizer discovered by Google via symbolic program search.

- **Algorithm**: Uses the sign of the update rather than magnitude. Effectively tracks momentum of signs.
- **Benefits**: 
    - 2x fewer state variables than Adam (only tracks momentum, not variance).
    - Often converges faster and to better optima for Vision Transformers and LLMs.
    - Triton-fused implementation for speed.

```python
from data_pipeline.trainer.optimizers import Lion

optimizer = Lion(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.99)
)
```

## 3. Sophia (`SophiaG`)

Second-order optimization without the cost.

- **Algorithm**: Estimates the diagonal Hessian using a Gauss-Newton-Bartlett estimator. Normalizes updates by curvature.
- **Benefits**: faster convergence in terms of steps compared to Adam, especially for LLMs (2x speedup reported in paper).

```python
from data_pipeline.trainer.optimizers import SophiaG

optimizer = SophiaG(
    model.parameters(),
    lr=2e-4,
    betas=(0.965, 0.99),
    rho=0.04
)
```

## 4. Prodigy (`Prodigy`)

Adaptive learning rate optimization.

- **Algorithm**: Estimates the distance to the optimal solution ($D$) to dynamically adjust the step size.
- **Benefits**: Removes the need for painful learning rate tuning. Just set `lr=1.0`.

```python
from data_pipeline.trainer.optimizers import Prodigy

optimizer = Prodigy(
    model.parameters(),
    lr=1.0,  # Let Prodigy handle it
    d_coef=1.0
)
```

## 5. CAME (`CAME`)

Communication-Aware Momentum estimation for large-scale distributed training.

- **Algorithm**: Factorizes the second moment matrix into row and column ranks.
- **Benefits**: Reduces memory and communication overhead for massive models.

## 6. Fused AdamW (`FusedAdamW`)

Standard AdamW but rewritten in pure Triton.

- **Benefits**: 
    - Fuses the entire update step (momentum, variance, bias correction, weight decay, parameter update) into a single kernel launch.
    - Significantly reduced VRAM bandwidth usage compared to PyTorch native implementation.
