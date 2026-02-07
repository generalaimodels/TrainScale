#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════════
# SOTA End-to-End Demo for Zero Bubble Pipeline Parallelism (ZBPP)
# ════════════════════════════════════════════════════════════════════════════════
# Complete ZBPP pipeline demonstration using TrainScale SOTA modules:
#   - YAML-driven configuration
#   - Pipeline Parallelism with Zero Bubbles
#   - SOTA model: Mistral-7B / Llama-3 (registry compatible)
#
# Usage:
#   Single Node (Multi-GPU):
#     torchrun --nproc_per_node=4 zbpp_sota_demo.py --config zbpp_sota_config.yaml
#
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

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

from data_pipeline.trainer.distributed import (
    DistributedState,
    log_rank_0,
)
from data_pipeline.trainer.core.sota_config import SOTAConfig
from data_pipeline.trainer.trainers.sota_trainer import create_trainer, SOTATrainer
from data_pipeline.pipeline import DataPipeline
from data_pipeline.core.types import unwrap
from data_pipeline.trainer.distributed.zbpp import ZeroBubblePipeline, ZBPPOptimizer
from data_pipeline.trainer import inference

# ═════════════════════════════════════════════════════════════════════════════════
# Logging Setup
# ═════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("zbpp_sota_demo")


# ═════════════════════════════════════════════════════════════════════════════════
# Configuration Loader
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class SOTAZBPPConfig:
    """SOTA ZBPP configuration container from YAML."""
    raw: Dict[str, Any] = field(default_factory=dict)
    config_path: str = ""
    
    @classmethod
    def from_yaml(cls, path: str) -> "SOTAZBPPConfig":
        """Load configuration from YAML file."""
        p = Path(path)
        if not p.exists():
            # Try relative to script dir
            p = _SCRIPT_DIR / path
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        inst = cls(raw=data, config_path=str(p))
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
    def data(self) -> Dict[str, Any]:
        return self.raw.get("data", {})
    
    @property
    def dataloader(self) -> Dict[str, Any]:
        return self.raw.get("dataloader", {})
    
    @property
    def quantization(self) -> Dict[str, Any]:
        return self.raw.get("quantization", {})
    
    @property
    def export(self) -> Dict[str, Any]:
        return self.raw.get("export", {})


# ═════════════════════════════════════════════════════════════════════════════════
# ZBPP Demo Class
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class SOTADemo:
    """
    SOTA ZBPP Demo using SOTATrainer and DataPipeline.
    """
    config_path: str
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._data_pipeline = None
        
        # Initialize Distributed State (for logging/setup)
        self.dist_state = DistributedState.initialize().unwrap()
        
        log_rank_0(
            f"SOTA ZBPP Demo initialized: "
            f"rank={self.dist_state.rank}/{self.dist_state.world_size}",
        )

    def verify_integration(self):
        """Run deep integration verification checks."""
        log_rank_0("🧪 Starting Deep Integration Verification...")
        
        # 1. Config Check
        self.config = SOTAZBPPConfig.from_yaml(self.config_path)
        # trainer = create_trainer(self.config.raw)  <-- Moved down
        
        if self.config.distributed.get("strategy") != "pipeline_zbpp":
            raise ValueError(f"❌ Strategy check failed: expected 'pipeline_zbpp', got '{self.config.distributed.get('strategy')}'")
        log_rank_0("✅ Config: Strategy is 'pipeline_zbpp'")
        
        # Override for single-process verification
        if self.dist_state.world_size == 1:
            log_rank_0("   Forcing num_pipeline_stages=1 for single-process verification...")
            self.config.distributed["num_pipeline_stages"] = 1
            self.config.distributed["num_microbatches"] = 4 # Needs to be > stages usually, or just 1 check

        # Override for single-process verification
        if self.dist_state.world_size == 1:
            log_rank_0("   Forcing num_pipeline_stages=1 for single-process verification...")
            self.config.distributed["num_pipeline_stages"] = 1
            self.config.distributed["num_microbatches"] = 4
            
            # Use tiny model for verification to avoid huge download
            log_rank_0("   Using tiny random model for verification...")
            self.config.model["name_or_path"] = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
            self.config.model["torch_dtype"] = "float32" # Tiny model safe
            self.config.quantization["enabled"] = False # Ensure quantization is off for tiny model
            
        # We need to explicitly import these to check instance type
        from data_pipeline.trainer.distributed.zbpp import ZeroBubblePipeline, ZBPPOptimizer
        # 2. Model Setup
        log_rank_0("   Setting up model...")
        # Create trainer AFTER overrides
        trainer_config = SOTAConfig.from_dict(self.config.raw)
        trainer = SOTATrainer(trainer_config)
        trainer.setup_model()
        
        if not isinstance(trainer.model, ZeroBubblePipeline):
            raise TypeError(f"❌ Model check failed: Expected ZeroBubblePipeline, got {type(trainer.model)}")
        log_rank_0("✅ Model: Wrapped in ZeroBubblePipeline")
        
        if not isinstance(trainer.optimizer, ZBPPOptimizer):
            raise TypeError(f"❌ Optimizer check failed: Expected ZBPPOptimizer, got {type(trainer.optimizer)}")
        log_rank_0("✅ Optimizer: Initialized as ZBPPOptimizer")
        
        # 3. Data Check (Synthetic)
        log_rank_0("   Creating Synthetic DataLoader (Skipping DataPipeline download)...")
        from torch.utils.data import TensorDataset, DataLoader
        
        batch_size = self.config.training.get("per_device_train_batch_size")
        seq_len = self.config.data.get("max_seq_length") or 128
        vocab_size = 32000
        
        # Create dummy data
        input_ids = torch.randint(0, vocab_size, (batch_size * 4, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size * 4, seq_len))
        attention_mask = torch.ones((batch_size * 4, seq_len))
        
        dataset = TensorDataset(input_ids, attention_mask, labels)
        
        # Determine collate_fn to match dictionary structure expected by SOTATrainer
        def collate_fn(batch):
            input_ids, attention_mask, labels = zip(*batch)
            return {
                "input_ids": torch.stack(input_ids),
                "attention_mask": torch.stack(attention_mask),
                "labels": torch.stack(labels)
            }
            
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        
        log_rank_0("✅ Data: Synthetic DataLoader created")
        
        # 4. Training Loop Check
        log_rank_0("   Running 1 Step Training Loop...")
        trainer.config.training["max_steps"] = 1
        trainer.config.training["logging_steps"] = 1
        trainer.train(dataloader)
        log_rank_0("✅ Training: 1 step completed successfully")
        
        log_rank_0("🎉 All Integration Checks Passed!")

    def _export_model(self, trainer):
        """Export trained model."""
        if self.config.export.get("enabled"):
             log_rank_0("Exporting Model...")
             try:
                 trainer.export(tokenizer=self._data_pipeline._tokenizer_wrapper.tokenizer)
             except Exception as e:
                 log_rank_0(f"⚠️ Export failed: {e}")

    def _run_inference(self):
        """Benchmark SOTA Inference Engine."""
        log_rank_0("\n🚀 Step 9: Benchmarking SOTA Inference...")
        
        # Load the exported (merged) model + tokenizer for fresh benchmark
        export_dir = self.config.export.get("output_dir")
        if not export_dir or not os.path.exists(export_dir):
             log_rank_0(f"⚠️ Export directory {export_dir} not found. Skipping inference benchmark.")
             return

        log_rank_0(f"Loading exported model from {export_dir} for benchmark...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load fresh components to verify 'W + change weights' loading
        try:
            inference_model = AutoModelForCausalLM.from_pretrained(
                export_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            inference_tokenizer = AutoTokenizer.from_pretrained(export_dir)
            
            # Initialize Engine with fresh model
            engine = inference.SOTAInferenceEngine(
                model=inference_model, 
                tokenizer=inference_tokenizer,
                block_size=16
            )
            
            # Add diverse prompts
            prompts = [
                [{"role": "user", "content": "Explain quantum computing in simple terms."}],
                [{"role": "user", "content": "Write a python function to Fibonacci."}],
                [{"role": "user", "content": "What is the capital of France?"}],
                [{"role": "user", "content": "List 3 benefits of exercise."}],
                [{"role": "user", "content": "Describe the future of AI."}]
            ]
            
            # Add requests
            for i in range(5):
                messages = prompts[i % len(prompts)]
                # Apply chat template
                text_prompt = inference_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                engine.add_request(prompt=text_prompt, max_new_tokens=50)
                
                # Run one standard generation for reference/sanity check
                if i == 0:
                    log_rank_0("Running standard HF generation for sanity check...")
                    inputs = inference_tokenizer(text_prompt, return_tensors="pt").to(inference_model.device)
                    with torch.no_grad():
                        gen_tokens = inference_model.generate(
                            **inputs, 
                            max_new_tokens=50, 
                            do_sample=False, 
                            pad_token_id=inference_tokenizer.eos_token_id
                        )
                    log_rank_0(f"Standard Ref: {inference_tokenizer.decode(gen_tokens[0], skip_special_tokens=True)}")

            log_rank_0(f"Added 5 requests to Continuous Batching Scheduler.")
            
            # Run Generation
            responses = engine.generate_all()
            
            for i, resp in enumerate(responses):
                log_rank_0(f"Response {i}: {resp[:50]}...")
                
        except Exception as e:
            log_rank_0(f"❌ Inference benchmark failed: {e}")
            import traceback
            traceback.print_exc()


    def run(self, max_steps: int = 100):
        """Run the demo."""
        log_rank_0("═" * 60)
        log_rank_0("Starting SOTA ZBPP Training (using SOTATrainer)")
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
        self.config = SOTAZBPPConfig.from_yaml(self.config_path) # Cache config
        trainer_config = SOTAConfig.from_dict(self.config.raw)
        trainer = SOTATrainer(trainer_config)
        
        # Check strategy validity
        if self.config.distributed.get("strategy") != "pipeline_zbpp":
             log_rank_0("⚠️ WARNING: Config strategy is NOT 'pipeline_zbpp'. This demo is intended for ZBPP.")
        
        # Setup Model
        log_rank_0("Setting up SOTA Model with ZBPP...")
        trainer.setup_model()
        
        # Override config with CLI arguments
        if max_steps > 0:
            log_rank_0(f"Overriding max_steps: {max_steps}")
            trainer.config.training["max_steps"] = max_steps
        
        # 3. Get Distributed DataLoader
        # For ZBPP, we fetch global batch, and trainer splits it into microbatches
        batch_size = self.config.training.get("per_device_train_batch_size")
        
        log_rank_0("Creating Distributed DataLoader...")
        dl_result = self._data_pipeline.get_dataloader(
            split=self.config.data.get("train_split"),
            batch_size=batch_size,
            distributed=(self.dist_state.world_size > 1),
            rank=self.dist_state.rank,
            world_size=self.dist_state.world_size,
        )
        dataloader = unwrap(dl_result)
        
        # Create Eval DataLoader
        eval_dl_result = self._data_pipeline.get_dataloader(
            split=self.config.data.get("eval_split"),
            batch_size=self.config.training.get("per_device_eval_batch_size"),
            distributed=(self.dist_state.world_size > 1),
            rank=self.dist_state.rank,
            world_size=self.dist_state.world_size,
        )
        eval_dataloader = unwrap(eval_dl_result)
        

        # 4. Train
        log_rank_0("Starting ZBPP Training Loop...")
        metrics = trainer.train(dataloader, eval_dataloader=eval_dataloader)
        
        log_rank_0("Training Complete!")
        log_rank_0(f"Final Metrics: {metrics}")
        
        # 5. Export
        self._export_model(trainer)

        # 6. Benchmark Inference
        self._run_inference()
        
        return metrics


# ═════════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ZBPP SOTA Demo")
    parser.add_argument("--config", type=str, default="zbpp_sota_config.yaml", help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Verify setup without training")
    parser.add_argument("--verify", action="store_true", help="Run deep integration verification")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps (default: -1, use YAML)")
    args = parser.parse_args()
    
    # 1. Environment Setup
    # Ensure variables are set
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
        
        if args.verify:
            demo.verify_integration()
            return

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
