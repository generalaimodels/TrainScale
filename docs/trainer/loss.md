# SOTA Loss Functions

The `trainer.loss` package implements high-performance loss functions, often fused with Triton for maximum efficiency.

## 1. Flash Cross Entropy (`FlashCrossEntropy`)

Uses Triton to fuse the log-softmax and cross-entropy operations, avoiding the materialization of the large $(Batch \times Seq \times Vocab)$ logits tensor.

- **Complexity**: $O(N)$ memory instead of $O(N \times V)$.
- **Speed**: ~5-10x faster than `torch.nn.CrossEntropyLoss` for large vocabularies (e.g., Llama's 32k or 128k).
- **Z-Loss**: Optionally adds auxiliary loss ($log(z)^2$) to stabilize training of large models (PaLM style).

```python
from data_pipeline.trainer.loss import FlashCrossEntropy

criterion = FlashCrossEntropy(
    label_smoothing=0.1,
    z_loss_weight=1e-4
)
```

## 2. Focal Loss (`FocalLoss`)

Addresses class imbalance by down-weighting well-classified examples.

- **Formula**: $FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$
- **Use Case**: Useful when fine-tuning on datasets with rare critical tokens or imbalanced classification tasks.

## 3. Poly1Loss (`Poly1Loss`)

Polynomial expansion of Cross Entropy.

- **Concept**: Adjusts the gradient weighting of the leading polynomial term to emphasize different difficulty levels.
- **Reference**: "PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions".

## 4. Smoothed L1 Loss (`SmoothedL1`)

Robust regression loss (Huber Loss).

- **Use Case**: Value head training in RLHF (PPO/DPO) or regression tasks.
