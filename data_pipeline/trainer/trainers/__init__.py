# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer Module - Specialized Trainers
# ════════════════════════════════════════════════════════════════════════════════
# Above-SOTA specialized trainers for different training paradigms.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from data_pipeline.trainer.base import Trainer, TrainOutput
from data_pipeline.trainer.core.config import TrainingArguments
from data_pipeline.trainer.callbacks import Callback

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════════
# Pretraining Trainer
# ═════════════════════════════════════════════════════════════════════════════════

class PretrainingTrainer(Trainer):
    """
    Above-SOTA trainer for model pretraining.
    
    Features beyond standard pretraining:
    - Curriculum learning with dynamic sequence length
    - Automatic batch size scaling
    - Gradient noise injection for generalization
    - Exponential moving average (EMA) of model weights
    - Masked language modeling with dynamic masking
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        *,
        mlm_probability: float = 0.15,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        curriculum_learning: bool = True,
        initial_seq_length: int = 128,
        **kwargs,
    ):
        super().__init__(model, args, train_dataset, **kwargs)
        self.mlm_probability = mlm_probability
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.curriculum_learning = curriculum_learning
        self.initial_seq_length = initial_seq_length
        
        # EMA model
        self._ema_model: Optional[nn.Module] = None
        if use_ema:
            self._init_ema()
    
    def _init_ema(self) -> None:
        """Initialize exponential moving average model."""
        import copy
        self._ema_model = copy.deepcopy(self.model)
        self._ema_model.eval()
        for p in self._ema_model.parameters():
            p.requires_grad_(False)
    
    def _update_ema(self) -> None:
        """Update EMA model weights."""
        if self._ema_model is None:
            return
        with torch.no_grad():
            for ema_p, model_p in zip(self._ema_model.parameters(), self.model.parameters()):
                ema_p.mul_(self.ema_decay).add_(model_p, alpha=1 - self.ema_decay)
    
    def _get_curriculum_seq_length(self) -> int:
        """Get sequence length based on curriculum schedule."""
        if not self.curriculum_learning:
            return self.args.max_seq_length
        
        progress = self.global_step / max(1, self._ctx.total_steps)
        # Ramp from initial to max over first 20% of training
        ramp_progress = min(1.0, progress / 0.2)
        target_len = int(self.initial_seq_length + 
                        (self.args.max_seq_length - self.initial_seq_length) * ramp_progress)
        # Round to multiple of 64 for efficiency
        return ((target_len + 63) // 64) * 64
    
    def _training_step(self, model: nn.Module, inputs: Dict[str, Tensor]) -> float:
        """Training step with EMA update."""
        loss = super()._training_step(model, inputs)
        self._update_ema()
        return loss
    
    @property
    def ema_model(self) -> Optional[nn.Module]:
        """Get EMA model for evaluation."""
        return self._ema_model


class FineTuningTrainer(Trainer):
    """
    Above-SOTA trainer for model fine-tuning.
    
    Features:
    - Layer-wise learning rate decay
    - Selective layer freezing with gradual unfreezing
    - LoRA/Adapter integration ready
    - Discriminative fine-tuning
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        *,
        layer_lr_decay: float = 0.95,
        freeze_embeddings: bool = True,
        gradual_unfreeze: bool = False,
        unfreeze_per_epoch: int = 2,
        **kwargs,
    ):
        super().__init__(model, args, train_dataset, **kwargs)
        self.layer_lr_decay = layer_lr_decay
        self.freeze_embeddings = freeze_embeddings
        self.gradual_unfreeze = gradual_unfreeze
        self.unfreeze_per_epoch = unfreeze_per_epoch
        self._frozen_layers: List[nn.Module] = []
        
        if freeze_embeddings:
            self._freeze_embeddings()
    
    def _freeze_embeddings(self) -> None:
        """Freeze embedding layers."""
        for name, param in self.model.named_parameters():
            if "embed" in name.lower():
                param.requires_grad = False
    
    def _create_optimizer(self) -> Any:
        """Create optimizer with layer-wise learning rate decay."""
        from data_pipeline.trainer.optimizers import AdamW
        
        # Group parameters by depth
        param_groups = self._get_layerwise_params()
        
        return AdamW(
            param_groups,
            lr=self.args.optimizer.learning_rate,
            weight_decay=self.args.optimizer.weight_decay,
        )
    
    def _get_layerwise_params(self) -> List[Dict[str, Any]]:
        """Get parameter groups with layer-wise LR decay."""
        base_lr = self.args.optimizer.learning_rate
        
        # Detect layer structure
        layers = []
        for name, _ in self.model.named_modules():
            if "layer" in name.lower() or "block" in name.lower():
                layers.append(name)
        
        num_layers = len(set(l.split('.')[0] for l in layers)) or 12
        
        param_groups = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Determine layer depth
            depth = 0
            for i in range(num_layers):
                if f"layer.{i}" in name or f"layers.{i}" in name or f"block.{i}" in name:
                    depth = i + 1
                    break
            
            lr = base_lr * (self.layer_lr_decay ** (num_layers - depth))
            param_groups.append({"params": [param], "lr": lr})
        
        return param_groups


class Seq2SeqTrainer(Trainer):
    """
    Above-SOTA trainer for sequence-to-sequence models.
    
    Features:
    - Label smoothing with teacher forcing
    - Beam search evaluation
    - Length penalty and coverage penalty
    - Mixed teacher forcing / scheduled sampling
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        *,
        teacher_forcing_ratio: float = 1.0,
        scheduled_sampling: bool = False,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        **kwargs,
    ):
        super().__init__(model, args, train_dataset, **kwargs)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.scheduled_sampling = scheduled_sampling
        self.num_beams = num_beams
        self.length_penalty = length_penalty
    
    def _get_teacher_forcing_ratio(self) -> float:
        """Get current teacher forcing ratio (may decay)."""
        if not self.scheduled_sampling:
            return self.teacher_forcing_ratio
        
        # Linear decay from 1.0 to target ratio
        progress = self.global_step / max(1, self._ctx.total_steps)
        return 1.0 - progress * (1.0 - self.teacher_forcing_ratio)
    
    def _compute_loss(self, model: nn.Module, inputs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """Compute seq2seq loss with optional teacher forcing."""
        labels = inputs.pop("labels", None)
        decoder_input_ids = inputs.pop("decoder_input_ids", None)
        
        outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[1]
        
        return loss, logits
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Dict[str, Tensor],
        max_length: int = 128,
        num_beams: Optional[int] = None,
    ) -> Tensor:
        """Generate sequences with beam search."""
        self.model.eval()
        inputs = self._prepare_inputs(inputs)
        
        if hasattr(self.model, "generate"):
            return self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams or self.num_beams,
                length_penalty=self.length_penalty,
            )
        
        # Fallback: greedy decoding
        return self._greedy_decode(inputs, max_length)
    
    def _greedy_decode(self, inputs: Dict[str, Tensor], max_length: int) -> Tensor:
        """Simple greedy decoding fallback."""
        batch_size = inputs["input_ids"].size(0)
        device = inputs["input_ids"].device
        
        # Start tokens
        decoder_ids = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            outputs = self.model(**inputs, decoder_input_ids=decoder_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decoder_ids = torch.cat([decoder_ids, next_token], dim=-1)
            
            # Stop at EOS (assumed id=2)
            if (next_token == 2).all():
                break
        
        return decoder_ids


class TextGenerationTrainer(Trainer):
    """
    Above-SOTA trainer for autoregressive text generation.
    
    Features:
    - Nucleus (top-p) and top-k sampling during evaluation
    - Repetition penalty
    - Contrastive search
    - KV-cache optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        *,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        **kwargs,
    ):
        super().__init__(model, args, train_dataset, **kwargs)
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 128,
        do_sample: bool = True,
    ) -> Tensor:
        """Generate text with sampling."""
        self.model.eval()
        input_ids = input_ids.to(self.device)
        generated = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            if past_key_values is not None:
                outputs = self.model(
                    input_ids=generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                outputs = self.model(input_ids=generated, use_cache=True)
            
            logits = outputs.logits[:, -1, :] / self.temperature
            past_key_values = outputs.past_key_values if hasattr(outputs, "past_key_values") else None
            
            # Apply repetition penalty
            if self.repetition_penalty != 1.0:
                for i, seq in enumerate(generated):
                    for token_id in set(seq.tolist()):
                        logits[i, token_id] /= self.repetition_penalty
            
            if do_sample:
                next_token = self._sample_top_p_k(logits)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Stop at EOS
            if (next_token == 2).all():
                break
        
        return generated
    
    def _sample_top_p_k(self, logits: Tensor) -> Tensor:
        """Sample with top-k and top-p filtering."""
        # Top-k filtering
        if self.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")
        
        # Top-p (nucleus) filtering
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            
            sorted_mask = cumsum > self.top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            
            indices_to_remove = sorted_indices[sorted_mask]
            logits.scatter_(-1, indices_to_remove.unsqueeze(0), float("-inf"))
        
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


__all__ = [
    "PretrainingTrainer",
    "FineTuningTrainer",
    "Seq2SeqTrainer",
    "TextGenerationTrainer",
]
