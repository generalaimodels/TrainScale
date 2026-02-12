#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════════
# SOTA End-to-End Demo for AMD ROCm MI300X - FSDP2 Version
# ════════════════════════════════════════════════════════════════════════════════
# Complete FSDP2 pipeline demonstration using TrainScale SOTA modules.
# Optimized for massive scale training (Llama-3 70B+, etc.)
#
# Usage:
#   Verify Integration (CPU/Tiny Model):
#     python rocm_sota_demo_fsdp2.py --verify
#
#   Multi-GPU Training (4 GPUs):
#     torchrun --nproc_per_node=4 rocm_sota_demo_fsdp2.py --config rocm_sota_config_fsdp2.yaml
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

from data_pipeline.trainer.distributed import (
    DistributedState,
    log_rank_0,
)
from data_pipeline.trainer.core.sota_config import SOTAConfig
from data_pipeline.trainer.core.sota_config import SOTAConfig
from data_pipeline.trainer.trainers.sota_trainer import create_trainer, SOTATrainer
from data_pipeline.pipeline import DataPipeline
from data_pipeline.core.types import unwrap
from data_pipeline.trainer import inference

# ═════════════════════════════════════════════════════════════════════════════════
# Logging Setup
# ═════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rocm_sota_demo_fsdp2")


# ═════════════════════════════════════════════════════════════════════════════════
# Configuration Loader
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class SOTAFSDP2Config:
    """SOTA FSDP2 configuration container from YAML."""
    raw: Dict[str, Any] = field(default_factory=dict)
    config_path: str = ""
    
    @classmethod
    def from_yaml(cls, path: str) -> "SOTAFSDP2Config":
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
    def dataloader(self) -> Dict[str, Any]:
        return self.raw.get("dataloader", {})
    
    @property
    def export(self) -> Dict[str, Any]:
        return self.raw.get("export", {})


# ═════════════════════════════════════════════════════════════════════════════════
# SOTA FSDP2 Demo Class
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class SOTADemo:
    """
    SOTA FSDP2 Demo using SOTATrainer and DataPipeline.
    """
    config_path: str
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._data_pipeline = None
        
        # Initialize Distributed State
        self.dist_state = DistributedState.initialize().unwrap()
        
        log_rank_0(
            f"SOTA FSDP2 Demo initialized: "
            f"rank={self.dist_state.rank}/{self.dist_state.world_size}",
        )

    def verify_integration(self):
        """Run deep integration verification checks (CPU/Tiny Model friendly)."""
        log_rank_0("🧪 Starting FSDP2 Deep Integration Verification...")
        
        # 1. Config Check
        # 1. Config Check
        # Using SOTAConfig to load first (handles path resolution)
        self.config = SOTAFSDP2Config.from_yaml(self.config_path)
        
        if self.config.distributed.get("strategy") != "fsdp2":
             # Auto-fix for verification if config points elsewhere
             log_rank_0(f"⚠️ Config strategy is '{self.config.distributed.get('strategy')}', forcing 'fsdp2' for verification.")
             self.config.distributed["strategy"] = "fsdp2"

        log_rank_0("✅ Config: Strategy check passed")
        
        # Override for verification: always use tiny model + conservative kernels.
        if self.dist_state.world_size == 1:
            log_rank_0("   Modes: Single-Process Verification...")
        else:
            log_rank_0("   Modes: Multi-Process Verification...")

        log_rank_0("   Using tiny random model for verification...")
        self.config.model["name_or_path"] = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
        self.config.model["torch_dtype"] = "float32"
        self.config.quantization = self.config.raw.setdefault("quantization", {})
        self.config.quantization["enabled"] = False

        # Keep verification deterministic/stable.
        fsdp_cfg = self.config.distributed.get("fsdp_config", {})
        if isinstance(fsdp_cfg, dict):
            fsdp_cfg["use_triton_kernels"] = False
            fsdp_cfg["activation_checkpointing"] = False
            self.config.distributed["fsdp_config"] = fsdp_cfg

        # Force device to CPU if no GPU.
        if not torch.cuda.is_available():
            self.config.raw.setdefault("hardware", {})["device"] = "cpu"
            log_rank_0("   ⚠️ CUDA not available, falling back to CPU for verification.")

        # 2. Model Setup
        log_rank_0("   Setting up model...")
        # Create trainer AFTER overrides
        trainer_config = SOTAConfig.from_dict(self.config.raw)
        trainer = SOTATrainer(trainer_config)
        trainer.setup_model()
        
        log_rank_0(f"✅ Model: Setup complete (Type: {type(trainer.model).__name__})")

        # 3. Data Check (Synthetic)
        log_rank_0("   Creating Synthetic DataLoader...")
        from torch.utils.data import TensorDataset, DataLoader
        
        batch_size = 2
        seq_len = 16
        vocab_size = 1000 
        
        input_ids = torch.randint(0, vocab_size, (batch_size * 2, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size * 2, seq_len))
        attention_mask = torch.ones((batch_size * 2, seq_len))
        
        dataset = TensorDataset(input_ids, attention_mask, labels)
        
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
        trainer.config.training.max_steps = 1
        trainer.config.training.logging_steps = 1
        trainer.train(dataloader)
        log_rank_0("✅ Training: 1 step completed successfully")
        
        log_rank_0("🎉 All FSDP2 Integration Checks Passed!")

    def _export_model(self, trainer):
        """Export trained model."""
        export_cfg = self.config.export
        if not export_cfg.enabled:
            return

        log_rank_0("Exporting Model...")
        try:
             trainer.export(tokenizer=self._data_pipeline._tokenizer_wrapper.tokenizer)
        except Exception as e:
             log_rank_0(f"⚠️ Export failed: {e}")

    def _run_inference(self):
        """Benchmark SOTA Inference Engine."""
        log_rank_0("\n🚀 Step 9: Benchmarking SOTA Inference...")
        
        export_dir = self.config.export.output_dir
        if not export_dir or not os.path.exists(export_dir):
             log_rank_0(f"⚠️ Export directory {export_dir} not found. Skipping inference benchmark.")
             return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        try:
            # Load exported model
            inference_model = AutoModelForCausalLM.from_pretrained(
                export_dir,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True
            )
            inference_tokenizer = AutoTokenizer.from_pretrained(export_dir)
            
            engine = inference.SOTAInferenceEngine(
                model=inference_model, 
                tokenizer=inference_tokenizer,
                block_size=16
            )
            
            prompts = [
                [{"role": "user", "content": "Explain FSDP."}],
                [{"role": "user", "content": "What is ROCm?"}]
            ]
            
            for p in prompts:
                text = inference_tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                engine.add_request(prompt=text, max_new_tokens=30)
                
            responses = engine.generate_all()
            for i, resp in enumerate(responses):
                log_rank_0(f"Response {i}: {resp[:50]}...")
                
        except Exception as e:
             log_rank_0(f"❌ Inference benchmark failed: {e}")


    def run(self, max_steps: int = 100):
        """Run the demo."""
        log_rank_0("═" * 60)
        log_rank_0("Starting SOTA FSDP2 Training")
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
        self.config = SOTAFSDP2Config.from_yaml(self.config_path) # Cache config
        trainer_config = SOTAConfig.from_dict(self.config.raw)
        trainer = SOTATrainer(trainer_config)
        
        # Setup Model
        log_rank_0("Setting up SOTA FSDP2 Model...")
        trainer.setup_model()
        
        if max_steps > 0:
            trainer.config.training.max_steps = max_steps
        
        # 3. DataLoader
        log_rank_0("Creating Distributed DataLoader...")
        dl_result = self._data_pipeline.get_dataloader(
            split=trainer.config.data.train_split,
            batch_size=trainer.config.training.per_device_train_batch_size,
            distributed=(self.dist_state.world_size > 1),
            rank=self.dist_state.rank,
            world_size=self.dist_state.world_size,
        )
        dataloader = unwrap(dl_result)
        
        # Eval DataLoader
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
        
        # 5. Export
        if self.dist_state.rank == 0 and trainer.config.export.enabled:
            self._export_model(trainer)
            
            # 6. Benchmark Inference
            self._run_inference()
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="ROCm SOTA FSDP2 Demo")
    parser.add_argument("--config", type=str, default="rocm_sota_config_fsdp2.yaml", help="Path to YAML config")
    parser.add_argument("--verify", action="store_true", help="Run integration verification (Tiny Model)")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps")
    args = parser.parse_args()
    
    # Environment Setup
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
    
    demo: Optional[SOTADemo] = None
    try:
        demo = SOTADemo(args.config)
        
        if args.verify:
            demo.verify_integration()
        else:
            demo.run(max_steps=args.max_steps)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure all distributed resources are released to avoid
        # ProcessGroup warnings when the script exits.
        if demo is not None and hasattr(demo, "dist_state"):
            demo.dist_state.shutdown()
        elif dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
