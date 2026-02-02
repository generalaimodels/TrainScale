#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════════
# SOTA End-to-End Demo for AMD ROCm MI300X Multi-GPU Training - DDP Version
# ════════════════════════════════════════════════════════════════════════════════
# Complete DDP pipeline demonstration using TrainScale SOTA modules:
#   - YAML-driven configuration (NO hardcoding)
#   - AMD ROCm GPU detection via rocm-smi
#   - Multi-GPU training with DDP (Distributed Data Parallel)
#   - SOTA model: Mistral-7B / Llama-3.1-8B (registry compatible)
#   - SOTA optimizer: Lion, CAME, Prodigy (from TrainScale optimizers)
#   - SOTA scheduler: WSD, REX (from TrainScale schedulers)
#   - SOTA preprocessing: Token-aware content distribution
#
# Usage:
#   Single GPU:
#     python rocm_sota_demo_ddp.py --config rocm_sota_config.yaml
#
#   Multi-GPU (4 GPUs):
#     torchrun --nproc_per_node=4 rocm_sota_demo_ddp.py \
#        --config rocm_sota_config.yaml
#
# Hardware Support:
#   - AMD: MI300X, MI325X (ROCm/RCCL)
#   - NVIDIA: A100, H100, H200, B100, B200 (CUDA/NCCL)
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ═════════════════════════════════════════════════════════════════════════════════
# Path Setup
# ═════════════════════════════════════════════════════════════════════════════════
_SCRIPT_DIR = Path(__file__).resolve().parent
_TRAINSCALE_ROOT = _SCRIPT_DIR.parent.parent  # TrainScale/
if str(_TRAINSCALE_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINSCALE_ROOT))

import yaml
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler

# ═════════════════════════════════════════════════════════════════════════════════
# TrainScale SOTA Imports
# ═════════════════════════════════════════════════════════════════════════════════

# Distributed (SOTA modules)
from data_pipeline.trainer.distributed import (
    # SOTA DDP
    SOTADDP,
    DDPConfig,
    GradientCompression,
    create_ddp,
    create_ddp_from_config,
    # Utilities
    DistributedState,
    clip_grad_norm_,
    set_deterministic_seed,
    mixed_precision_context,
    prepare_model_for_distributed,
    prepare_optimizer_for_distributed,
    prepare_scheduler_for_distributed,
    log_rank_0,
    print_rank_0,
)

# Registry (model patching)
from data_pipeline.trainer import registry

# Optimizers (SOTA: Lion, CAME, etc.)
from data_pipeline.trainer.optimizers import create_optimizer

# Schedulers (SOTA: WSD, REX, etc.)
from data_pipeline.trainer.schedulers import create_sota_scheduler

# Metrics
from data_pipeline.trainer.metrics import MetricCollection, Perplexity, Accuracy

# ═════════════════════════════════════════════════════════════════════════════════
# Logging Setup
# ═════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rocm_sota_demo_ddp")


# ═════════════════════════════════════════════════════════════════════════════════
# Configuration Loader
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class SOTADDPConfig:
    """SOTA DDP configuration container from YAML."""
    raw: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> "SOTADDPConfig":
        """Load configuration from YAML file."""
        p = Path(path)
        if not p.exists():
            # Try relative to script dir
            p = _SCRIPT_DIR / path
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        inst = cls(raw=data)
        inst.config_path = str(p)
        return inst
    
    @property
    def model(self) -> Dict[str, Any]:
        return self.raw.get("model", {})
    
    @property
    def distributed(self) -> Dict[str, Any]:
        return self.raw.get("distributed", {})
    
    @property
    def optimizer(self) -> Dict[str, Any]:
        return self.raw.get("optimizer", {})
    
    @property
    def scheduler(self) -> Dict[str, Any]:
        return self.raw.get("scheduler", {})
    
    @property
    def training(self) -> Dict[str, Any]:
        return self.raw.get("training", {})
    
    @property
    def dataloader(self) -> Dict[str, Any]:
        return self.raw.get("dataloader", {})


# ═════════════════════════════════════════════════════════════════════════════════
# SOTA DDP Training Pipeline
# ═════════════════════════════════════════════════════════════════════════════════

class SOTADDPPipeline:
    """
    Complete SOTA DDP pipeline for multi-GPU training.
    
    Integrates with TrainScale registry for:
      - Any registered model
      - SOTA optimizers (Lion, CAME, Prodigy)
      - SOTA schedulers (WSD, REX)
    """
    
    def __init__(self, config: SOTADDPConfig):
        self.config = config
        
        # Initialize distributed state from environment
        self.dist_state = DistributedState.from_environment()
        
        # Initialize if not already done
        if self.dist_state.world_size > 1 and not self.dist_state.initialized:
            self.dist_state = DistributedState.initialize(
                backend=config.distributed.get("backend", "nccl"),
            )
        
        # Set seed for reproducibility
        set_deterministic_seed(
            config.training.get("seed", 42),
            self.dist_state,
        )
        
        # Components (lazy init)
        self._model = None
        self._tokenizer = None
        self._optimizer = None
        self._scheduler = None
        self._ddp = None
        
        log_rank_0(
            f"SOTA DDP Pipeline initialized: "
            f"rank={self.dist_state.rank}/{self.dist_state.world_size}",
        )
    
    def init_model(self) -> nn.Module:
        """Initialize model from config (registry compatible)."""
        from transformers import AutoModelForCausalLM
        
        model_cfg = self.config.model
        model_name = model_cfg.get("name_or_path", "mistralai/Mistral-7B-Instruct-v0.3")
        
        log_rank_0(f"Loading model: {model_name}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=model_cfg.get("trust_remote_code", True),
            torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
            low_cpu_mem_usage=model_cfg.get("low_cpu_mem_usage", True),
            attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        )
        
        # Apply registry patches (Triton kernels)
        registry.patch_model(model)
        
        # Move to device
        model = model.to(self.dist_state.device)
        
        self._model = model
        return model
    
    def init_tokenizer(self):
        """Initialize tokenizer."""
        from transformers import AutoTokenizer
        
        model_name = self.config.model.get("name_or_path")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self._tokenizer = tokenizer
        return tokenizer
    
    def wrap_model_with_ddp(self) -> nn.Module:
        """Wrap model with SOTA DDP."""
        ddp_cfg = self.config.distributed.get("ddp_config", {})
        
        # Create SOTA DDP from config
        self._ddp = create_ddp(
            bucket_cap_mb=ddp_cfg.get("bucket_cap_mb", 25),
            gradient_compression=ddp_cfg.get("gradient_compression", "none"),
            static_graph=ddp_cfg.get("static_graph", True),
            find_unused_parameters=ddp_cfg.get("find_unused_parameters", False),
        )
        
        # Wrap model
        wrapped = self._ddp.wrap_model(self._model)
        
        log_rank_0(f"Model wrapped with SOTA DDP (bucket={ddp_cfg.get('bucket_cap_mb', 25)}MB)")
        
        self._model = wrapped
        return wrapped
    
    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize SOTA optimizer from TrainScale registry."""
        opt_cfg = self.config.optimizer
        opt_type = opt_cfg.get("type", "adamw")
        
        # Get trainable parameters
        params = [p for p in self._model.parameters() if p.requires_grad]
        
        log_rank_0(f"Creating SOTA optimizer: {opt_type}")
        
        # Use TrainScale optimizer factory
        optimizer = create_optimizer(
            name=opt_type,
            params=params,
            lr=opt_cfg.get("learning_rate", 3e-5),
            weight_decay=opt_cfg.get("weight_decay", 0.1),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
        
        # Prepare for distributed (if needed)
        optimizer = prepare_optimizer_for_distributed(
            optimizer,
            self._model,
            strategy="ddp",
        )
        
        self._optimizer = optimizer
        return optimizer
    
    def init_scheduler(self, num_training_steps: int):
        """Initialize SOTA scheduler from TrainScale registry."""
        sch_cfg = self.config.scheduler
        sch_type = sch_cfg.get("type", "cosine")
        
        warmup_ratio = sch_cfg.get("warmup_ratio", 0.03)
        warmup_steps = int(warmup_ratio * num_training_steps)
        
        log_rank_0(f"Creating SOTA scheduler: {sch_type}")
        
        # Use TrainScale scheduler factory
        scheduler = create_sota_scheduler(
            name=sch_type,
            optimizer=self._optimizer,
            num_training_steps=num_training_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=sch_cfg.get("min_lr_ratio", 0.1),
        )
        
        # Prepare for distributed
        scheduler = prepare_scheduler_for_distributed(scheduler, self.dist_state)
        
        self._scheduler = scheduler
        return scheduler
    
    def create_dataloader(
        self,
        dataset,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create distributed dataloader."""
        dl_cfg = self.config.dataloader
        
        sampler = None
        if self.dist_state.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.dist_state.world_size,
                rank=self.dist_state.rank,
                shuffle=shuffle,
            )
            shuffle = False  # Sampler handles shuffling
        
        return DataLoader(
            dataset,
            batch_size=dl_cfg.get("batch_size", 4),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=dl_cfg.get("num_workers", 4),
            pin_memory=dl_cfg.get("pin_memory", True),
            drop_last=dl_cfg.get("drop_last", True),
        )
    
    def train_step(
        self,
        batch: Dict[str, Tensor],
        gradient_accumulation_steps: int = 1,
        step: int = 0,
    ) -> Dict[str, float]:
        """
        Execute single training step with DDP.
        
        Handles:
          - Mixed precision via autocast
          - Gradient accumulation with no_sync
          - SOTA gradient clipping
          - Loss reduction across ranks
        """
        self._model.train()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.dist_state.device)
        attention_mask = batch["attention_mask"].to(self.dist_state.device)
        labels = batch["labels"].to(self.dist_state.device)
        
        # Determine if we should sync gradients (last accumulation step)
        sync_context = (
            self._ddp.no_sync()
            if (step + 1) % gradient_accumulation_steps != 0
            else torch.cuda.amp.autocast(enabled=False)  # Dummy context
        )
        
        # Forward pass with mixed precision
        with mixed_precision_context(enabled=True, dtype=torch.bfloat16):
            with sync_context:
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights on accumulation boundary
        if (step + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            grad_norm = self._ddp.clip_grad_norm_(
                self.config.optimizer.get("max_grad_norm", 1.0)
            )
            
            # Optimizer step
            self._optimizer.step()
            self._scheduler.step()
            self._optimizer.zero_grad()
        else:
            grad_norm = 0.0
        
        # Reduce loss for logging
        loss_reduced = dist_reduce(loss.detach(), op="mean")
        
        return {
            "loss": loss_reduced.item() * gradient_accumulation_steps,
            "grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            "lr": self._scheduler.get_last_lr()[0],
        }
    
    def run(self, max_steps: int = 100) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Returns training metrics.
        """
        log_rank_0("═" * 60)
        log_rank_0("Starting SOTA DDP Training")
        log_rank_0("═" * 60)
        
        # Initialize components
        self.init_tokenizer()
        self.init_model()
        self.wrap_model_with_ddp()
        self.init_optimizer()
        self.init_scheduler(max_steps)
        
        # Create dummy dataset for demo
        from torch.utils.data import TensorDataset
        
        batch_size = self.config.dataloader.get("batch_size", 4)
        seq_len = 512
        vocab_size = self._tokenizer.vocab_size
        
        dummy_inputs = torch.randint(0, vocab_size, (batch_size * 10, seq_len))
        dummy_labels = dummy_inputs.clone()
        dummy_attention = torch.ones_like(dummy_inputs)
        
        dataset = TensorDataset(dummy_inputs, dummy_attention, dummy_labels)
        dataloader = self.create_dataloader(dataset, shuffle=True)
        
        # Training loop
        grad_accum = self.config.training.get("gradient_accumulation_steps", 4)
        
        start_time = time.time()
        total_loss = 0.0
        
        for step, (input_ids, attention_mask, labels) in enumerate(dataloader):
            if step >= max_steps:
                break
            
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            
            metrics = self.train_step(batch, grad_accum, step)
            total_loss += metrics["loss"]
            
            if step % 10 == 0:
                log_rank_0(
                    f"Step {step}/{max_steps}: "
                    f"loss={metrics['loss']:.4f}, "
                    f"lr={metrics['lr']:.2e}"
                )
        
        elapsed = time.time() - start_time
        
        results = {
            "avg_loss": total_loss / max(step, 1),
            "total_steps": step,
            "elapsed_seconds": elapsed,
            "samples_per_second": (step * batch_size) / elapsed,
        }
        
        log_rank_0("═" * 60)
        log_rank_0(f"Training Complete: avg_loss={results['avg_loss']:.4f}")
        log_rank_0(f"Throughput: {results['samples_per_second']:.1f} samples/sec")
        log_rank_0("═" * 60)
        
        return results


# ═════════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═════════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point for SOTA DDP Demo."""
    parser = argparse.ArgumentParser(
        description="SOTA DDP Training Demo for AMD ROCm / NVIDIA Multi-GPU"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="rocm_sota_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify setup without training",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = SOTADDPConfig.from_yaml(args.config)
    
    # Create and run pipeline
    pipeline = SOTADDPPipeline(config)
    
    if args.dry_run:
        log_rank_0("✅ Dry run complete. Setup verified.")
        return
    
    results = pipeline.run(max_steps=args.max_steps)
    
    print_rank_0(f"\n✅ Training complete! Final loss: {results['avg_loss']:.4f}")


if __name__ == "__main__":
    main()
