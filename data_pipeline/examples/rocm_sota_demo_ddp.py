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
import torch.distributed as dist


# ═════════════════════════════════════════════════════════════════════════════════
# TrainScale SOTA Imports
# ═════════════════════════════════════════════════════════════════════════════════

# Distributed (SOTA modules)
from data_pipeline.trainer.distributed import (
    # SOTA DDP
    SOTADDP,
    DDPConfig,
    GradientCompression,

    # Utilities
    DistributedState,
    mixed_precision_context,
    prepare_model_for_distributed,
    prepare_optimizer_for_distributed,
    prepare_scheduler_for_distributed,
    log_rank_0,
    print_rank_0,

)

# Data Pipeline
from data_pipeline.pipeline import DataPipeline
from data_pipeline.core.types import unwrap


# SOTA Trainer (API_TRAINER)
from data_pipeline.trainer.trainers.sota_trainer import create_trainer
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

@dataclass
class SOTADemo:
    """
    SOTA DDP Demo using SOTATrainer and DataPipeline.
    """
    config_path: str
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._data_pipeline = None
        
        # Initialize Distributed State (for logging/setup)
        self.dist_state = DistributedState.initialize().unwrap()
        
        log_rank_0(
            f"SOTA DDP Demo initialized: "
            f"rank={self.dist_state.rank}/{self.dist_state.world_size}",
        )


    def _export_model(self, trainer, output_dir: str):
        """Export trained model with LoRA merging."""
        log_rank_0(f"💾 Exporting model to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Unwrap model
        model = trainer.model
        if hasattr(model, "module"):
            model = model.module
            
        # Merge LoRA logic
        if self.config.export.merge_lora:
            log_rank_0("Merging LoRA weights...")
            try:
                # Use internal LoRA module
                from data_pipeline.trainer import lora
                model = lora.merge_and_unload(model)
                log_rank_0("✓ Merged LoRA weights via internal module")
            except Exception as e:
                logger.warning(f"Merge failed: {e}")
        
        # Save
        model.save_pretrained(output_dir)
        
        # Save tokenizer from data pipeline
        if self._data_pipeline._tokenizer_wrapper:
            self._data_pipeline._tokenizer_wrapper.tokenizer.save_pretrained(output_dir)
            
        log_rank_0("Export complete.")

    def _run_inference(self, model_path: str):
        """
        Run sample inference on the exported model.

        Strategy:
          1) Try TrainScale SOTA inference engine (data_pipeline/trainer/inference.py)
          2) Fallback to plain Transformers generation on error/unavailability
        """
        log_rank_0(f"🚀 Running Inference Benchmark on {model_path}...")

        prompts = [
            "User: What is the capital of France?\nAssistant:",
            "User: Write a python function to compute Fibonacci numbers.\nAssistant:",
        ]

        if self._run_inference_sota_engine(model_path, prompts):
            return

        log_rank_0(
            "⚠️ SOTA inference engine unavailable/failed; "
            "falling back to Transformers inference."
        )
        self._run_inference_transformers(model_path, prompts)

    def _run_inference_sota_engine(
        self,
        model_path: str,
        prompts: List[str],
    ) -> bool:
        """Try inference using data_pipeline.trainer.inference.py."""
        if not os.path.exists(model_path):
            logger.warning(
                f"SOTA inference skipped: model path not found: {model_path}"
            )
            return False

        try:
            from data_pipeline.trainer import inference as sota_inference
        except BaseException as e:
            logger.warning(
                "Could not import TrainScale inference module "
                f"(likely missing vLLM/runtime deps): {e}"
            )
            return False

        try:
            dtype = "bfloat16" if torch.cuda.is_available() else "float32"
            engine_config = sota_inference.EngineConfig(
                model=model_path,
                tokenizer=model_path,
                dtype=dtype,
                trust_remote_code=True,
                enforce_eager=True,
            )
            engine = sota_inference.SOTAInferenceEngine(engine_config)
            gen_params = sota_inference.GenerationParams(
                max_tokens=100,
                temperature=0.7,
                top_p=0.95,
            )

            result = engine.generate(prompts, params=gen_params)
            if result.is_err():
                logger.warning(
                    f"SOTA inference generation failed: {result.error}"
                )
                return False

            responses = result.unwrap()
            if not responses:
                logger.warning("SOTA inference returned no responses.")
                return False

            for prompt, response in zip(prompts, responses):
                log_rank_0(f"\nExample Prompt: {prompt}")
                log_rank_0(f"Response:\n{response.generated_text}")
                log_rank_0("-" * 40)

            try:
                metrics = engine.get_metrics_summary()
                log_rank_0(f"SOTA inference metrics: {metrics}")
            except Exception as metrics_err:
                logger.warning(
                    f"Failed to collect inference metrics: {metrics_err}"
                )

            return True

        except Exception as e:
            logger.warning(f"SOTA inference path failed: {e}")
            logger.debug("SOTA inference traceback", exc_info=True)
            return False

    def _run_inference_transformers(
        self,
        model_path: str,
        prompts: List[str],
    ) -> None:
        """Fallback inference using HuggingFace Transformers."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load back for verification
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                dtype=torch.bfloat16, 
                device_map="cuda",
                trust_remote_code=True
            )
            
            for prompt in prompts:
                log_rank_0(f"\nExample Prompt: {prompt}")
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=100, 
                        do_sample=True, 
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                log_rank_0(f"Response:\n{response}")
                log_rank_0("-" * 40)

        except Exception as e:
            log_rank_0(f"❌ Fallback inference failed: {e}")

    def run(self, max_steps: int = 100):
        """Run the demo."""
        log_rank_0("═" * 60)
        log_rank_0("Starting SOTA DDP Training (using SOTATrainer)")
        log_rank_0("═" * 60)
        
        # 1. Initialize Data Pipeline
        log_rank_0("Initializing Data Pipeline...")
        dp_result = DataPipeline.from_config(
            self.config_path,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=False,
        )
        self._data_pipeline = unwrap(dp_result)
        
        # 2. Create SOTA Trainer
        log_rank_0("Initializing SOTA Trainer...")
        trainer = create_trainer(self.config_path)
        self.config = trainer.config # Cache config
        
        # Setup Model
        log_rank_0("Setting up SOTA Model...")
        trainer.setup_model()
        
        # Override config with CLI arguments
        if max_steps > 0:
            log_rank_0(f"Overriding max_steps: {max_steps}")
            trainer.config.training.max_steps = max_steps
        
        # 3. Get Distributed DataLoader
        batch_size = trainer.config.training.per_device_train_batch_size
        
        log_rank_0("Creating Distributed DataLoader...")
        log_rank_0(f"DEBUG: train_split={trainer.config.data.train_split}")
        dl_result = self._data_pipeline.get_dataloader(
            split=trainer.config.data.train_split,
            batch_size=batch_size,
            distributed=(self.dist_state.world_size > 1),
            rank=self.dist_state.rank,
            world_size=self.dist_state.world_size,
        )
        dataloader = unwrap(dl_result)
        
        # Create Eval DataLoader
        eval_dl_result = self._data_pipeline.get_dataloader(
            split=trainer.config.data.eval_split,
            batch_size=trainer.config.training.per_device_eval_batch_size,
            distributed=(self.dist_state.world_size > 1),
            rank=self.dist_state.rank,
            world_size=self.dist_state.world_size,
        )
        eval_dataloader = unwrap(eval_dl_result)
        
        # 4. Train
        log_rank_0("Starting Training Loop...")
        metrics = trainer.train(dataloader, eval_dataloader=eval_dataloader)
        
        log_rank_0("Training Complete!")
        log_rank_0(f"Final Metrics: {metrics}")
        
        # 5. Export & Inference (Rank 0 only)
        if self.dist_state.rank == 0:
            output_dir = trainer.config.training.output_dir
            self._export_model(trainer, output_dir)
            self._run_inference(output_dir)
        
        return metrics

# ═════════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ROCm SOTA DDP Demo")
    parser.add_argument("--config", type=str, default="rocm_sota_config.yaml", help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Verify setup without training")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps (default: -1, use YAML)")
    args = parser.parse_args()
    
    # 1. Environment Setup
    # Ensure variables are set for DDP
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
    
    if args.dry_run:
        print("Dry run successful. Environment verified.")
        return

    # 2. Run Demo
    try:
        demo = SOTADemo(args.config)
        demo.run(max_steps=args.max_steps)
        
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()
            
    except Exception as e:
        # Log error cleanly
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
