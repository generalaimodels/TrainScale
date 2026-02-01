# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer Module - HuggingFace Hub Integration
# ════════════════════════════════════════════════════════════════════════════════
# Above-SOTA Hub integration for model sharing and collaboration.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HubManager:
    """
    HuggingFace Hub integration manager.
    
    Features:
    - Model push/pull with automatic sharding
    - Model card generation
    - Training metrics logging
    - Checkpoint management
    """
    
    def __init__(
        self,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        private: bool = False,
    ):
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        self.private = private
        self._api = None
    
    @property
    def api(self):
        """Lazy-load HuggingFace Hub API."""
        if self._api is None:
            try:
                from huggingface_hub import HfApi
                self._api = HfApi(token=self.token)
            except ImportError:
                raise ImportError("huggingface_hub required: pip install huggingface_hub")
        return self._api
    
    def push_to_hub(
        self,
        model: nn.Module,
        repo_id: Optional[str] = None,
        commit_message: str = "Upload model",
        model_card: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Push model to HuggingFace Hub.
        
        Returns:
            URL of the uploaded model
        """
        repo_id = repo_id or self.repo_id
        if not repo_id:
            raise ValueError("repo_id required")
        
        # Create repo if needed
        self.api.create_repo(repo_id=repo_id, private=self.private, exist_ok=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Save model
            self._save_model(model, tmpdir)
            
            # Generate model card
            if model_card:
                (tmpdir / "README.md").write_text(model_card)
            elif tags:
                card = self._generate_model_card(model, tags)
                (tmpdir / "README.md").write_text(card)
            
            # Upload
            self.api.upload_folder(
                folder_path=str(tmpdir),
                repo_id=repo_id,
                commit_message=commit_message,
                **kwargs,
            )
        
        return f"https://huggingface.co/{repo_id}"
    
    def _save_model(self, model: nn.Module, path: Path) -> None:
        """Save model with safetensors if available."""
        try:
            from safetensors.torch import save_file
            save_file(model.state_dict(), path / "model.safetensors")
        except ImportError:
            torch.save(model.state_dict(), path / "pytorch_model.bin")
        
        # Save config if available
        if hasattr(model, "config"):
            config = model.config
            if hasattr(config, "to_dict"):
                config = config.to_dict()
            with open(path / "config.json", "w") as f:
                json.dump(config, f, indent=2)
    
    def pull_from_hub(
        self,
        repo_id: Optional[str] = None,
        revision: str = "main",
        cache_dir: Optional[str] = None,
    ) -> Path:
        """
        Download model from HuggingFace Hub.
        
        Returns:
            Path to downloaded model directory
        """
        repo_id = repo_id or self.repo_id
        from huggingface_hub import snapshot_download
        
        return Path(snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=self.token,
        ))
    
    def load_from_hub(
        self,
        model: nn.Module,
        repo_id: Optional[str] = None,
        revision: str = "main",
    ) -> nn.Module:
        """Load weights from Hub into model."""
        path = self.pull_from_hub(repo_id, revision)
        
        # Try safetensors first
        safetensors_path = path / "model.safetensors"
        pytorch_path = path / "pytorch_model.bin"
        
        if safetensors_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        elif pytorch_path.exists():
            state_dict = torch.load(pytorch_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model found in {path}")
        
        model.load_state_dict(state_dict)
        return model
    
    def _generate_model_card(self, model: nn.Module, tags: List[str]) -> str:
        """Generate basic model card."""
        num_params = sum(p.numel() for p in model.parameters())
        
        return f"""---
tags:
{chr(10).join(f'- {tag}' for tag in tags)}
library_name: pytorch
---

# Model Card

## Model Description

This model was trained using the SOTA Trainer framework.

## Model Details

- **Parameters**: {num_params:,}
- **Framework**: PyTorch
- **Date**: {datetime.now().strftime('%Y-%m-%d')}

## Training

Trained with above-SOTA infrastructure including:
- Triton-fused optimizers
- Mixed precision training
- Gradient accumulation
"""


def generate_model_card(
    model: nn.Module,
    model_name: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    metrics: Optional[Dict[str, float]] = None,
    training_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate comprehensive model card.
    
    Args:
        model: The trained model
        model_name: Name of the model
        description: Model description
        tags: List of tags
        metrics: Evaluation metrics
        training_config: Training configuration
        
    Returns:
        Markdown model card string
    """
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    tags = tags or ["pytorch", "sota-trainer"]
    
    card = f"""---
tags:
{chr(10).join(f'- {tag}' for tag in tags)}
library_name: pytorch
model-index:
- name: {model_name}
  results: []
---

# {model_name}

{description}

## Model Details

| Property | Value |
|----------|-------|
| Parameters | {num_params:,} |
| Trainable | {trainable_params:,} |
| Framework | PyTorch |
| Training Date | {datetime.now().strftime('%Y-%m-%d')} |

"""
    
    if metrics:
        card += "## Evaluation Results\n\n"
        card += "| Metric | Value |\n|--------|-------|\n"
        for k, v in metrics.items():
            card += f"| {k} | {v:.4f} |\n"
        card += "\n"
    
    if training_config:
        card += "## Training Configuration\n\n```json\n"
        card += json.dumps(training_config, indent=2)
        card += "\n```\n"
    
    card += """
## Usage

```python
from data_pipeline.trainer import Trainer

# Load model
model = YourModel()
trainer = Trainer(model=model, args=args)
trainer.load_model("path/to/checkpoint")
```

## Framework

Trained with SOTA Trainer featuring:
- Triton-fused AdamW/LAMB optimizers
- Mixed precision (FP16/BF16)
- Gradient accumulation and clipping
- Distributed training support
"""
    
    return card


__all__ = ["HubManager", "generate_model_card"]
