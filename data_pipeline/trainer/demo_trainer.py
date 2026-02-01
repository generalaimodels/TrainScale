#!/usr/bin/env python3
"""
SOTA Trainer - End-to-End Demo Script

This demo showcases the complete SOTA trainer infrastructure:
1. Training configuration via YAML
2. Custom model training with mixed precision
3. Callbacks (progress, checkpointing, early stopping)
4. Metrics computation
5. Hub integration

Usage:
    python demo_trainer.py
    python demo_trainer.py --config training_config.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add parent to path for local testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.trainer import (
    # Trainer
    Trainer,
    PretrainingTrainer,
    FineTuningTrainer,
    TrainingArguments,
    load_training_config,
    # Optimizers
    AdamW,
    LAMB,
    # Schedulers
    CosineScheduler,
    get_cosine_schedule_with_warmup,
    # Loss
    CrossEntropyLoss,
    # Callbacks
    EarlyStoppingCallback,
    CheckpointCallback,
    LoggingCallback,
    ProgressCallback,
    # Metrics
    Accuracy,
    F1Score,
    MetricCollection,
    # Distributed
    DeviceManager,
    is_main_process,
    # Kernels
    is_triton_available,
    compile_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════════
# Demo Model
# ═════════════════════════════════════════════════════════════════════════════════

class DemoTransformer(nn.Module):
    """Simple transformer for demonstration."""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        num_classes: int = 10,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(512, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, input_ids, labels=None):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) + self.pos_encoding[:seq_len]
        x = self.encoder(x)
        logits = self.classifier(x[:, 0])  # CLS token
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        
        return {"loss": loss, "logits": logits}


# ═════════════════════════════════════════════════════════════════════════════════
# Demo Dataset
# ═════════════════════════════════════════════════════════════════════════════════

class DemoDataset(Dataset):
    """Random data for demonstration."""
    
    def __init__(self, size: int = 1000, seq_len: int = 64, vocab_size: int = 10000, num_classes: int = 10):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
            "labels": torch.randint(0, self.num_classes, ()),
        }


# ═════════════════════════════════════════════════════════════════════════════════
# Demo Functions
# ═════════════════════════════════════════════════════════════════════════════════

def demo_basic_training():
    """Demonstrate basic training loop."""
    logger.info("=" * 60)
    logger.info("DEMO: Basic Training")
    logger.info("=" * 60)
    
    # Create model and data
    model = DemoTransformer()
    train_dataset = DemoDataset(size=500)
    eval_dataset = DemoDataset(size=100)
    
    # Training arguments
    args = TrainingArguments(
        output_dir="./demo_output",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        max_grad_norm=1.0,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            ProgressCallback(),
            EarlyStoppingCallback(patience=3, metric="eval_loss"),
        ],
    )
    
    # Train
    logger.info("Starting training...")
    result = trainer.train()
    
    logger.info(f"Training complete!")
    logger.info(f"  Steps: {result.global_step}")
    logger.info(f"  Loss: {result.training_loss:.4f}")
    
    # Evaluate
    eval_result = trainer.evaluate()
    logger.info(f"Evaluation: {eval_result}")


def demo_pretraining():
    """Demonstrate pretraining with EMA and curriculum learning."""
    logger.info("=" * 60)
    logger.info("DEMO: Pretraining with EMA")
    logger.info("=" * 60)
    
    model = DemoTransformer()
    train_dataset = DemoDataset(size=300)
    
    args = TrainingArguments(
        output_dir="./demo_pretrain",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
    )
    
    trainer = PretrainingTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        use_ema=True,
        ema_decay=0.999,
        curriculum_learning=True,
    )
    
    result = trainer.train()
    logger.info(f"Pretraining complete! Loss: {result.training_loss:.4f}")
    
    # Access EMA model for inference
    if trainer.ema_model is not None:
        logger.info("EMA model available for inference")


def demo_finetuning():
    """Demonstrate fine-tuning with layer-wise LR decay."""
    logger.info("=" * 60)
    logger.info("DEMO: Fine-tuning with Layer-wise LR Decay")
    logger.info("=" * 60)
    
    model = DemoTransformer()
    train_dataset = DemoDataset(size=200)
    
    args = TrainingArguments(
        output_dir="./demo_finetune",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
    )
    
    trainer = FineTuningTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        layer_lr_decay=0.95,  # Lower layers get smaller LR
        freeze_embeddings=True,
    )
    
    result = trainer.train()
    logger.info(f"Fine-tuning complete! Loss: {result.training_loss:.4f}")


def demo_metrics():
    """Demonstrate metrics computation."""
    logger.info("=" * 60)
    logger.info("DEMO: Metrics Computation")
    logger.info("=" * 60)
    
    # Create metrics collection
    metrics = MetricCollection([
        Accuracy(),
        F1Score(average="macro"),
    ])
    
    # Simulate predictions
    for _ in range(5):
        preds = torch.randint(0, 10, (32, 10)).float()  # Logits
        targets = torch.randint(0, 10, (32,))
        metrics.update(preds, targets)
    
    # Compute
    results = metrics.compute()
    logger.info(f"Metrics: {results}")


def demo_kernels():
    """Demonstrate Triton kernel availability."""
    logger.info("=" * 60)
    logger.info("DEMO: Triton Kernels")
    logger.info("=" * 60)
    
    logger.info(f"Triton available: {is_triton_available()}")
    
    if torch.cuda.is_available():
        from data_pipeline.trainer.kernels import fused_softmax, fused_gelu
        
        x = torch.randn(32, 1024, device="cuda")
        
        # Fused operations
        y_softmax = fused_softmax(x)
        y_gelu = fused_gelu(x)
        
        logger.info(f"Fused softmax output shape: {y_softmax.shape}")
        logger.info(f"Fused GELU output shape: {y_gelu.shape}")
    else:
        logger.info("CUDA not available, skipping kernel demo")


def demo_torch_compile():
    """Demonstrate torch.compile integration."""
    logger.info("=" * 60)
    logger.info("DEMO: torch.compile Integration")
    logger.info("=" * 60)
    
    model = DemoTransformer()
    
    # Compile model
    compiled_model = compile_model(model, mode="default")
    
    # Test forward pass
    x = torch.randint(0, 10000, (2, 64))
    output = compiled_model(x)
    
    logger.info(f"Compiled model output: loss={output['loss']}, logits shape={output['logits'].shape}")


# ═════════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SOTA Trainer Demo")
    parser.add_argument("--demo", choices=["all", "basic", "pretrain", "finetune", "metrics", "kernels", "compile"],
                       default="all", help="Which demo to run")
    args = parser.parse_args()
    
    logger.info("SOTA Trainer Demo")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Triton available: {is_triton_available()}")
    
    demos = {
        "basic": demo_basic_training,
        "pretrain": demo_pretraining,
        "finetune": demo_finetuning,
        "metrics": demo_metrics,
        "kernels": demo_kernels,
        "compile": demo_torch_compile,
    }
    
    if args.demo == "all":
        for name, func in demos.items():
            try:
                func()
            except Exception as e:
                logger.error(f"Demo '{name}' failed: {e}")
    else:
        demos[args.demo]()
    
    logger.info("Demo complete!")


if __name__ == "__main__":
    main()
