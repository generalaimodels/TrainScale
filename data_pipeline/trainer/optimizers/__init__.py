# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer Module - Optimizers Package
# ════════════════════════════════════════════════════════════════════════════════
# Above-SOTA optimizers with Triton kernel fusion.
# ════════════════════════════════════════════════════════════════════════════════

from data_pipeline.trainer.optimizers.base import (
    BaseOptimizer,
    compute_grad_norm,
    clip_grad_norm_,
    clip_grad_value_,
    ParamGroup,
    OptimizerState,
)

from data_pipeline.trainer.optimizers.adamw import (
    AdamW,
    create_adamw,
)

from data_pipeline.trainer.optimizers.lamb import (
    LAMB,
    create_lamb,
)

from data_pipeline.trainer.optimizers.adafactor import (
    AdaFactor,
    create_adafactor,
)

# SOTA Optimizers (Above Unsloth)
from data_pipeline.trainer.optimizers.sota_optimizers import (
    Adam8bit,
    Lion,
    CAME,
    SophiaG,
    Prodigy,
    FusedAdamW,
    create_optimizer,
)

__all__ = [
    # Base
    "BaseOptimizer",
    "compute_grad_norm",
    "clip_grad_norm_",
    "clip_grad_value_",
    "ParamGroup",
    "OptimizerState",
    # AdamW
    "AdamW",
    "create_adamw",
    # LAMB
    "LAMB",
    "create_lamb",
    # AdaFactor
    "AdaFactor",
    "create_adafactor",
    # SOTA (Above Unsloth)
    "Adam8bit",
    "Lion",
    "CAME",
    "SophiaG",
    "Prodigy",
    "FusedAdamW",
    "create_optimizer",
]

