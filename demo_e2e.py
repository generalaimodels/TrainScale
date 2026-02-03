#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════════
# SOTA End-to-End Demo for TrainScale
# ════════════════════════════════════════════════════════════════════════════════
# This script demonstrates the complete pipeline from data loading to inference
# using ONLY TrainScale's internal APIs.
#
# Features:
# - Data Pipeline Loading & Preprocessing
# - SOTA Training with LoRA, Lion Optimizer, and WSD Scheduler
# - Kernel Optimizations (RoPE, RMSNorm)
# - Inference with Continuous Batching & PagedAttention
# ════════════════════════════════════════════════════════════════════════════════

import os
import sys
import logging
import torch
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("demo_e2e")

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_TRAINSCALE_ROOT = _SCRIPT_DIR
if str(_TRAINSCALE_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINSCALE_ROOT))

# Imports from TrainScale packages
from data_pipeline.pipeline import DataPipeline
from data_pipeline.core.config_schema import PipelineConfig
from data_pipeline.trainer.core.sota_config import SOTAConfig, OptimizerType, SchedulerType, LossType
from data_pipeline.trainer.trainers.sota_trainer import SOTATrainer
from data_pipeline.trainer.inference import SOTAInferenceEngine
from transformers import AutoTokenizer

def main():
    logger.info("🚀 Starting TrainScale End-to-End Demo")
    
    # ═════════════════════════════════════════════════════════════════════════════
    # 1. Configuration
    # ═════════════════════════════════════════════════════════════════════════════
    # We define configurations programmatically for the demo, but they map to YAML structure
    
    # 1.1 Data Pipeline Config
    pipeline_config_dict = {
        "dataset": {
            "name": "tatsu-lab/alpaca",
            "split": "train[:100]", # Small subset for demo
            "columns": ["text"],
        },
        "preprocessing": {
            "max_length": 512,
            "truncation_strategy": "smart",
        },
        "dataloader": {
            "batch_size": 2,
            "num_workers": 0,
        }
    }
    
    # 1.2 Training Config (SOTAConfig)
    # Using internal configuration object
    train_config = SOTAConfig(
        model=SOTAConfig.ModelConfig(
            name_or_path="HuggingFaceTB/SmolLM-135M", # Small model for quick demo
            torch_dtype="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",
        ),
        optimizer=SOTAConfig.OptimizerConfig(
            type=OptimizerType.LION, # SOTA Optimizer
            learning_rate=1e-4,
            weight_decay=0.01,
        ),
        scheduler=SOTAConfig.SchedulerConfig(
            type=SchedulerType.WSD, # SOTA Scheduler
            warmup_steps=5,
            stable_ratio=0.8,
            min_lr_ratio=0.1,
        ),
        loss=SOTAConfig.LossConfig(
            type=LossType.CROSS_ENTROPY, # Will use fused if available
        ),
        training=SOTAConfig.TrainingConfig(
            num_train_epochs=1,
            gradient_accumulation_steps=1,
            logging_steps=1,
            output_dir="./demo_outputs",
            save_strategy="no",
        ),
        kernels=SOTAConfig.KernelConfig(
            use_triton=True,
            use_fused_rope=True,
            use_fused_rms_norm=True,
            use_fused_cross_entropy=True,
        ),
        # Using LoRA
        # Note: SOTAConfig might need explicit LoRA config or we handle it in trainer setup
    )
    
    # ═════════════════════════════════════════════════════════════════════════════
    # 2. Data Pipeline
    # ═════════════════════════════════════════════════════════════════════════════
    logger.info("📦 Initializing Data Pipeline...")
    
    # Write temp yaml for DataPipeline (since it loads from file usually)
    with open("temp_demo_config.yaml", "w") as f:
        yaml.dump(pipeline_config_dict, f)
        
    pipeline = DataPipeline("temp_demo_config.yaml")
    
    # Load and preprocess
    # Assuming pipeline.run() returns a dataloader or we step through it
    # pipeline.setup() -> pipeline.create_dataloader()
    
    # Let's use the components directly if pipeline.py is an orchestrator wrapper
    # Inspecting pipeline.py in previous turns: it has prepare_data(), create_dataloader()
    
    logger.info("   Loading and processing data...")
    pipeline.prepare_data()
    train_dataloader = pipeline.create_dataloader()
    
    logger.info(f"   Data ready: {len(train_dataloader)} batches")
    
    # ═════════════════════════════════════════════════════════════════════════════
    # 3. Training Setup
    # ═════════════════════════════════════════════════════════════════════════════
    logger.info("🏋️ Initializing SOTA Trainer...")
    
    trainer = SOTATrainer(config=train_config)
    
    # Setup Model
    model = trainer.setup_model()
    logger.info(f"   Model active: {type(model).__name__}")
    
    # Apply LoRA (Manual step if not auto-triggered by config, but let's check config)
    # SOTATrainer logic: setup_model usually loads base.
    # We might want to explicitly apply LoRA if we are doing PEFT
    from data_pipeline.trainer.lora import LoraConfig, apply_lora, print_trainable_parameters
    
    logger.info("   Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        init_lora_weights="gaussian"
    )
    model = apply_lora(model, lora_config)
    trainer.model = model # Update trainer's reference
    print_trainable_parameters(model)
    
    # ═════════════════════════════════════════════════════════════════════════════
    # 4. Training Loop
    # ═════════════════════════════════════════════════════════════════════════════
    logger.info("🔄 Starting Training Loop...")
    
    metrics = trainer.train(train_dataloader)
    
    logger.info(f"✅ Training Complete. Metrics: {metrics}")
    
    # ═════════════════════════════════════════════════════════════════════════════
    # 5. Inference
    # ═════════════════════════════════════════════════════════════════════════════
    logger.info("🤖 Starting Inference Engine...")
    
    # In a real scenario, we might merge adapters first
    # from data_pipeline.trainer.lora import merge_and_unload
    # model = merge_and_unload(model)
    
    tokenizer = AutoTokenizer.from_pretrained(train_config.model.name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    engine = SOTAInferenceEngine(model, tokenizer)
    
    prompts = [
        "The future of AI is",
        "Explain quantum computing in one sentence:",
    ]
    
    logger.info(f"   Processing {len(prompts)} prompts...")
    for p in prompts:
        engine.add_request(p, max_new_tokens=20)
        
    results = engine.generate_all()
    
    for prompt, output in zip(prompts, results):
        print(f"\n[Prompt]: {prompt}")
        print(f"[Output]: {output}")
        
    # Cleanup
    if os.path.exists("temp_demo_config.yaml"):
        os.remove("temp_demo_config.yaml")
        
    logger.info("✨ Demo Finished Successfully")

if __name__ == "__main__":
    main()
