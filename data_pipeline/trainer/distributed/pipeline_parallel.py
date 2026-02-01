# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer - Pipeline Parallel Implementation
# ════════════════════════════════════════════════════════════════════════════════
# Above-SOTA Pipeline Parallelism with GPipe and 1F1B schedules.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Pipeline parallelism configuration."""
    num_stages: int = 4
    num_microbatches: int = 8
    schedule: str = "1f1b"  # "gpipe" or "1f1b"
    checkpoint_activations: bool = True


def get_pipeline_parallel_rank() -> int:
    """Get pipeline parallel rank."""
    if not torch.distributed.is_initialized():
        return 0
    # Assumes PP uses sequential ranks
    return torch.distributed.get_rank()


def get_pipeline_parallel_world_size() -> int:
    """Get pipeline parallel world size."""
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


class PipelineStage(nn.Module):
    """
    A single stage in the pipeline.
    
    Wraps a sequential portion of the model for pipeline execution.
    """
    
    def __init__(
        self,
        module: nn.Module,
        stage_id: int,
        num_stages: int,
        device: torch.device,
    ):
        super().__init__()
        self.module = module.to(device)
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.device = device
        
        self.is_first_stage = stage_id == 0
        self.is_last_stage = stage_id == num_stages - 1
    
    def forward(self, input_: Tensor) -> Tensor:
        return self.module(input_)


class PipelineSchedule:
    """
    Base class for pipeline schedules.
    
    Manages forward/backward execution order across microbatches.
    """
    
    def __init__(
        self,
        stages: List[PipelineStage],
        num_microbatches: int,
        loss_fn: Callable,
    ):
        self.stages = stages
        self.num_microbatches = num_microbatches
        self.loss_fn = loss_fn
        self.num_stages = len(stages)
    
    def run(self, input_batches: List[Tensor], target_batches: List[Tensor]) -> Tensor:
        """Execute pipeline schedule. Override in subclasses."""
        raise NotImplementedError


class GPipeSchedule(PipelineSchedule):
    """
    GPipe schedule: all forwards, then all backwards.
    
    Simple but has high memory footprint due to storing all activations.
    """
    
    def run(self, input_batches: List[Tensor], target_batches: List[Tensor]) -> Tensor:
        # Store activations for backward
        activations: List[List[Tensor]] = [[] for _ in range(self.num_stages)]
        outputs: List[Tensor] = []
        
        # Forward pass for all microbatches
        for mb_idx, input_batch in enumerate(input_batches):
            x = input_batch
            for stage_idx, stage in enumerate(self.stages):
                x = x.to(stage.device)
                if x.requires_grad:
                    activations[stage_idx].append(x.detach().requires_grad_(True))
                else:
                    activations[stage_idx].append(x)
                x = stage(x)
            outputs.append(x)
        
        # Compute losses
        total_loss = torch.tensor(0.0, device=outputs[0].device)
        losses = []
        for output, target in zip(outputs, target_batches):
            loss = self.loss_fn(output, target.to(output.device))
            losses.append(loss)
            total_loss = total_loss + loss
        
        # Backward pass for all microbatches (reverse order)
        for mb_idx in reversed(range(self.num_microbatches)):
            grad = torch.autograd.grad(losses[mb_idx], outputs[mb_idx])[0]
            
            for stage_idx in reversed(range(self.num_stages)):
                stage = self.stages[stage_idx]
                act = activations[stage_idx][mb_idx]
                
                # Recompute forward for gradient
                with torch.enable_grad():
                    act_req = act.requires_grad_(True)
                    out = stage(act_req)
                
                if stage_idx > 0:
                    grad = torch.autograd.grad(out, act_req, grad)[0]
        
        return total_loss / self.num_microbatches


class OneFOneBSchedule(PipelineSchedule):
    """
    1F1B (One Forward One Backward) schedule.
    
    Interleaves forward and backward passes for reduced memory.
    Also known as PipeDream-Flush schedule.
    """
    
    def run(self, input_batches: List[Tensor], target_batches: List[Tensor]) -> Tensor:
        num_warmup = min(self.num_stages - 1, self.num_microbatches)
        num_steady = self.num_microbatches - num_warmup
        
        activations: List[Tensor] = []
        losses: List[Tensor] = []
        total_loss = torch.tensor(0.0, device=self.stages[0].device)
        
        # Warmup phase: only forward passes
        for mb_idx in range(num_warmup):
            x = input_batches[mb_idx]
            for stage in self.stages:
                x = x.to(stage.device)
                x = stage(x)
            activations.append(x)
            
            loss = self.loss_fn(x, target_batches[mb_idx].to(x.device))
            losses.append(loss)
            total_loss = total_loss + loss
        
        # Steady state: 1F1B
        for mb_idx in range(num_steady):
            # Backward for oldest microbatch
            if activations:
                oldest_loss = losses.pop(0)
                oldest_act = activations.pop(0)
                oldest_loss.backward(retain_graph=False)
            
            # Forward for new microbatch
            x = input_batches[num_warmup + mb_idx]
            for stage in self.stages:
                x = x.to(stage.device)
                x = stage(x)
            activations.append(x)
            
            loss = self.loss_fn(x, target_batches[num_warmup + mb_idx].to(x.device))
            losses.append(loss)
            total_loss = total_loss + loss
        
        # Cooldown phase: only backward passes
        while losses:
            loss = losses.pop(0)
            loss.backward(retain_graph=False)
        
        return total_loss / self.num_microbatches


class PipelineParallel(nn.Module):
    """
    Pipeline parallel wrapper for models.
    
    Splits model into stages and executes with chosen schedule.
    
    Example:
        ```python
        # Split transformer into stages
        stages = split_model_into_stages(model, num_stages=4)
        
        pp_model = PipelineParallel(
            stages=stages,
            num_microbatches=8,
            schedule="1f1b",
        )
        
        loss = pp_model(input_batches, target_batches)
        ```
    """
    
    def __init__(
        self,
        stages: List[nn.Module],
        num_microbatches: int = 4,
        schedule: str = "1f1b",
        loss_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.num_stages = len(stages)
        self.num_microbatches = num_microbatches
        self.schedule_type = schedule
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        
        # Create pipeline stages
        devices = self._get_devices()
        self.pipeline_stages = nn.ModuleList([
            PipelineStage(stage, i, self.num_stages, devices[i])
            for i, stage in enumerate(stages)
        ])
        
        # Create schedule
        if schedule == "gpipe":
            self.schedule = GPipeSchedule(
                list(self.pipeline_stages), num_microbatches, self.loss_fn
            )
        else:
            self.schedule = OneFOneBSchedule(
                list(self.pipeline_stages), num_microbatches, self.loss_fn
            )
    
    def _get_devices(self) -> List[torch.device]:
        """Get devices for each stage."""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            return [torch.device(f"cuda:{i % num_gpus}") for i in range(self.num_stages)]
        return [torch.device("cpu")] * self.num_stages
    
    def forward(
        self,
        input_batches: List[Tensor],
        target_batches: List[Tensor],
    ) -> Tensor:
        """
        Execute pipeline parallel forward/backward.
        
        Args:
            input_batches: List of microbatch inputs
            target_batches: List of microbatch targets
            
        Returns:
            Average loss across microbatches
        """
        assert len(input_batches) == self.num_microbatches
        assert len(target_batches) == self.num_microbatches
        
        return self.schedule.run(input_batches, target_batches)


def split_model_into_stages(
    model: nn.Module,
    num_stages: int,
    split_points: Optional[List[str]] = None,
) -> List[nn.Module]:
    """
    Split a sequential model into pipeline stages.
    
    Args:
        model: Model to split (must be nn.Sequential or have named children)
        num_stages: Number of pipeline stages
        split_points: Optional layer names to split at
        
    Returns:
        List of nn.Sequential modules, one per stage
    """
    if isinstance(model, nn.Sequential):
        layers = list(model.children())
    else:
        layers = list(model.children())
    
    if not layers:
        raise ValueError("Model has no children to split")
    
    # Divide layers evenly
    layers_per_stage = max(1, len(layers) // num_stages)
    stages = []
    
    for i in range(num_stages):
        start = i * layers_per_stage
        if i == num_stages - 1:
            end = len(layers)
        else:
            end = start + layers_per_stage
        
        stage_layers = layers[start:end]
        if stage_layers:
            stages.append(nn.Sequential(*stage_layers))
    
    return stages


__all__ = [
    "PipelineConfig",
    "PipelineStage",
    "PipelineSchedule",
    "GPipeSchedule",
    "OneFOneBSchedule",
    "PipelineParallel",
    "split_model_into_stages",
    "get_pipeline_parallel_rank",
    "get_pipeline_parallel_world_size",
]
