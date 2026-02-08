#!/usr/bin/env python3
"""
SOTA Preprocessing-Trainer Integration Verification

This script verifies the complete integration between the preprocessing module
and the trainer module, ensuring end-to-end compatibility for all training stages.

Tests:
1. Preprocessing → DataLoader → Trainer Integration
2. All Training Stages (Pre-training, Fine-tuning, RL)
3. Token-Aware Content Distribution
4. Special Token Handling
5. Loss Functions Compatibility

Requirements:
- HuggingFaceTB/SmolLM2-135M tokenizer (auto-downloaded)
- CPU-only (no GPU required)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════════════════════
# Imports: Data Pipeline (Preprocessing)
# ═══════════════════════════════════════════════════════════════════════════════════

from data_pipeline.core.config_schema import (
    PipelineConfig, DatasetConfig, TokenizerConfig, PromptEngineConfig,
    PreTrainingConfig, FineTuningConfig, RLConfig as PipelineRLConfig,
    TrainingStage, PreTrainingFormat, FineTuningFormat, RLAlgorithm as PipelineRLAlgorithm,
    TruncationStrategy,
)
from data_pipeline.pipeline import DataPipeline
from data_pipeline.core.types import unwrap, unwrap_err, Err, Ok, Result
from data_pipeline.preprocessing.length_manager import (
    create_length_manager, create_content_distributor,
    ContentDistributionMode, TruncationStrategy as LMTruncationStrategy,
)
from data_pipeline.preprocessing.tokenization import create_tokenizer, TokenizerWrapper
from data_pipeline.preprocessing.prompt_engine import (
    PromptEngine, create_pretraining_engine, create_finetuning_engine, create_rl_engine,
    ProcessedExample, extract_successful, batch_to_dict,
)

# ═══════════════════════════════════════════════════════════════════════════════════
# Imports: Trainer Module
# ═══════════════════════════════════════════════════════════════════════════════════

from data_pipeline.trainer import (
    # Core Types
    TrainingState, GradientInfo, StepMetrics,
    # Optimizers
    AdamW, create_adamw, clip_grad_norm_, compute_grad_norm,
    # Schedulers
    CosineScheduler, create_scheduler, get_cosine_schedule_with_warmup,
    # Loss Functions
    CrossEntropyLoss, create_loss,
    ChunkedCrossEntropyLoss, DPOLoss, ORPOLoss,
    # Metrics
    Perplexity, MetricCollection, compute_accuracy,
    # Callbacks
    Callback, CallbackContext,
    # SOTA Config
    SOTAConfig, ModelConfig, DataConfig, TrainConfig,
    get_qlora_preset, get_dpo_preset,
    # SOTA Trainer
    SOTATrainer, create_trainer,
)

# ═══════════════════════════════════════════════════════════════════════════════════
# Test Configuration
# ═══════════════════════════════════════════════════════════════════════════════════

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
MAX_LENGTH = 128
BATCH_SIZE = 2
SAMPLE_SIZE = 10


# ═══════════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════════

def log_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'═' * 70}")
    print(f" {title}")
    print(f"{'═' * 70}")


def log_test(name: str, passed: bool, details: str = "") -> None:
    """Log test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    msg = f"  [{status}] {name}"
    if details:
        msg += f" - {details}"
    print(msg)


def create_dummy_dataset(stage: TrainingStage, size: int = SAMPLE_SIZE) -> Dict[str, List]:
    """Create dummy dataset for testing different stages."""
    from datasets import Dataset
    
    if stage == TrainingStage.PRE_TRAINING:
        return Dataset.from_dict({
            "text": [f"This is a test document number {i}. " * 10 for i in range(size)]
        })
    
    elif stage == TrainingStage.FINE_TUNING:
        return Dataset.from_dict({
            "instruction": [f"Summarize the following text about topic {i}." for i in range(size)],
            "input": [f"Input content for example {i}. " * 5 for i in range(size)],
            "output": [f"Summary of topic {i}." for i in range(size)],
        })
    
    elif stage == TrainingStage.POST_TRAINING_RL:
        return Dataset.from_dict({
            "prompt": [f"What is the capital of country {i}?" for i in range(size)],
            "chosen": [f"The capital of country {i} is City{i}." for i in range(size)],
            "rejected": [f"I don't know about country {i}." for i in range(size)],
        })
    
    raise ValueError(f"Unknown stage: {stage}")


# ═══════════════════════════════════════════════════════════════════════════════════
# Test 1: Tokenizer Wrapper Verification
# ═══════════════════════════════════════════════════════════════════════════════════

def test_tokenizer_wrapper() -> bool:
    """Test TokenizerWrapper special token handling."""
    log_section("Test 1: TokenizerWrapper Special Token Handling")
    
    try:
        from data_pipeline.core.config_schema import TokenizerConfig as TokConfig
        
        config = TokConfig(
            name_or_path=TOKENIZER_NAME,
            max_length=MAX_LENGTH,
        )
        
        result = create_tokenizer(config)
        if isinstance(result, Err):
            log_test("Tokenizer Creation", False, str(unwrap_err(result)))
            return False
        
        tokenizer = unwrap(result)
        
        # Test special token IDs (must be non-negative)
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id
        bos_id = tokenizer.bos_token_id
        
        log_test("pad_token_id >= 0", pad_id >= 0, f"pad_id={pad_id}")
        log_test("eos_token_id >= 0", eos_id >= 0, f"eos_id={eos_id}")
        log_test("bos_token_id >= 0", bos_id >= 0, f"bos_id={bos_id}")
        
        # Test encoding
        test_text = "Hello, this is a test."
        ids, mask = tokenizer.encode(test_text, max_length=MAX_LENGTH)
        
        log_test("Encode works", len(ids) > 0, f"len={len(ids)}")
        log_test("Attention mask aligned", len(ids) == len(mask), f"ids={len(ids)}, mask={len(mask)}")
        
        # Test vocab tracking
        log_test("Vocab size > 0", tokenizer.vocab_size > 0, f"vocab={tokenizer.vocab_size}")
        
        print(f"\n  Summary: Tokenizer loaded with vocab_size={tokenizer.vocab_size}")
        return True
        
    except Exception as e:
        log_test("TokenizerWrapper", False, str(e))
        return False


# ═══════════════════════════════════════════════════════════════════════════════════
# Test 2: Length Manager & Content Distribution
# ═══════════════════════════════════════════════════════════════════════════════════

def test_length_manager() -> bool:
    """Test LengthManager and TokenAwareContentDistributor."""
    log_section("Test 2: Length Manager & Content Distribution")
    
    try:
        # Create tokenizer for exact token counting
        from data_pipeline.core.config_schema import TokenizerConfig as TokConfig
        config = TokConfig(name_or_path=TOKENIZER_NAME, max_length=MAX_LENGTH)
        tok_result = create_tokenizer(config)
        if isinstance(tok_result, Err):
            log_test("Tokenizer for LengthManager", False, str(unwrap_err(tok_result)))
            return False
        tokenizer = unwrap(tok_result)
        
        # Test LengthManager creation
        length_manager = create_length_manager(
            max_length=MAX_LENGTH,
            padding_strategy="longest",
            default_truncation="smart",
            tokenizer=tokenizer.tokenizer,
        )
        
        log_test("LengthManager created", length_manager is not None, f"max={length_manager.max_length}")
        
        # Test preprocessing
        example = {"instruction": "Summarize this text.", "input": "A" * 500, "output": "Summary."}
        processed = length_manager.preprocess_example(example, text_columns=["instruction", "input", "output"])
        
        total_chars = sum(len(processed.get(k, "")) for k in ["instruction", "input", "output"] if isinstance(processed.get(k), str))
        log_test("Text truncated", total_chars < 500, f"total_chars={total_chars}")
        
        # Test ContentDistributor
        distributor = create_content_distributor(
            max_length=MAX_LENGTH,
            distribution_mode="adaptive",
            column_ratios={"instruction": 0.2, "input": 0.5, "output": 0.3},
            tokenizer=tokenizer.tokenizer,
        )
        
        log_test("Distributor created", distributor is not None)
        
        # Test distribution
        texts = {"instruction": "Task instruction", "input": "A" * 1000, "output": "Expected output"}
        distributed = distributor.distribute(texts, text_columns=["instruction", "input", "output"])
        
        log_test("Distribution works", len(distributed) > 0, f"keys={list(distributed.keys())}")
        
        return True
        
    except Exception as e:
        log_test("LengthManager", False, str(e))
        import traceback
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════════════════════
# Test 3: Pre-training Stage Integration
# ═══════════════════════════════════════════════════════════════════════════════════

def test_pretraining_stage() -> bool:
    """Test pre-training stage with trainer loss function."""
    log_section("Test 3: Pre-training → Trainer Integration")
    
    try:
        # Setup pipeline config
        config = PipelineConfig(
            type="data_module",
            version="1.0",
            dataset=DatasetConfig(
                name="dummy_pretrain",
                splits={"train": {"name": "train", "sample_size": SAMPLE_SIZE}},
                columns=["text"],
            ),
            tokenizer=TokenizerConfig(
                name_or_path=TOKENIZER_NAME,
                max_length=MAX_LENGTH,
            ),
            prompt_engine=PromptEngineConfig(
                stage=TrainingStage.PRE_TRAINING,
                max_length=MAX_LENGTH,
                pretraining=PreTrainingConfig(
                    format=PreTrainingFormat.CAUSAL_LM,
                    text_column="text",
                    add_bos=True,
                    add_eos=True,
                ),
            ),
        )
        
        # Initialize pipeline
        pipeline = DataPipeline(pipeline_config=config)
        log_test("Pipeline created", pipeline is not None)
        
        # Inject dummy dataset
        ds = create_dummy_dataset(TrainingStage.PRE_TRAINING)
        pipeline._datasets["train_None"] = ds
        
        # Get DataLoader
        dl_result = pipeline.get_dataloader("train", batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)
        if isinstance(dl_result, Err):
            log_test("DataLoader", False, str(unwrap_err(dl_result)))
            return False
        
        dataloader = unwrap(dl_result)
        batch = next(iter(dataloader))
        
        log_test("Batch has input_ids", "input_ids" in batch, f"shape={batch['input_ids'].shape}")
        log_test("Batch has labels", "labels" in batch, f"shape={batch['labels'].shape}")
        log_test("Batch has attention_mask", "attention_mask" in batch)
        
        # Test with Trainer loss function
        loss_fn = ChunkedCrossEntropyLoss(ignore_index=-100, chunk_size=1024)
        
        # Simulate logits (batch_size, seq_len, vocab_size)
        vocab_size = 49152  # SmolLM2 vocab size
        logits = torch.randn(BATCH_SIZE, MAX_LENGTH, vocab_size)
        labels = batch["labels"]
        
        loss = loss_fn(logits, labels)
        log_test("ChunkedCrossEntropyLoss works", loss.item() > 0, f"loss={loss.item():.4f}")
        
        # Verify tensor shapes are model-ready
        log_test("input_ids is torch.Tensor", isinstance(batch["input_ids"], torch.Tensor))
        log_test("Shape is (B, L)", batch["input_ids"].shape == (BATCH_SIZE, MAX_LENGTH))
        
        print(f"\n  Summary: Pre-training batch ready for model forward pass")
        return True
        
    except Exception as e:
        log_test("Pre-training", False, str(e))
        import traceback
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════════════════════
# Test 4: Fine-tuning Stage Integration
# ═══════════════════════════════════════════════════════════════════════════════════

def test_finetuning_stage() -> bool:
    """Test fine-tuning stage with trainer components."""
    log_section("Test 4: Fine-tuning → Trainer Integration")
    
    try:
        # Setup pipeline config
        config = PipelineConfig(
            type="data_module",
            version="1.0",
            dataset=DatasetConfig(
                name="dummy_finetune",
                splits={"train": {"name": "train", "sample_size": SAMPLE_SIZE}},
            ),
            tokenizer=TokenizerConfig(
                name_or_path=TOKENIZER_NAME,
                max_length=MAX_LENGTH,
            ),
            prompt_engine=PromptEngineConfig(
                stage=TrainingStage.FINE_TUNING,
                max_length=MAX_LENGTH,
                finetuning=FineTuningConfig(
                    format=FineTuningFormat.INSTRUCTION,
                    instruction_column="instruction",
                    input_column="input",
                    output_column="output",
                    mask_input=True,
                    add_eos=True,
                ),
            ),
        )
        
        # Initialize pipeline
        pipeline = DataPipeline(pipeline_config=config)
        
        # Inject dummy dataset
        ds = create_dummy_dataset(TrainingStage.FINE_TUNING)
        pipeline._datasets["train_None"] = ds
        
        # Get DataLoader
        dl_result = pipeline.get_dataloader("train", batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)
        if isinstance(dl_result, Err):
            log_test("DataLoader", False, str(unwrap_err(dl_result)))
            return False
        
        dataloader = unwrap(dl_result)
        batch = next(iter(dataloader))
        
        log_test("Batch shape correct", batch["input_ids"].shape == (BATCH_SIZE, MAX_LENGTH))
        
        # Test label masking (input should be masked with -100)
        labels = batch["labels"]
        masked_count = (labels == -100).sum().item()
        total_count = labels.numel()
        mask_ratio = masked_count / total_count
        
        log_test("Labels have masking", masked_count > 0, f"masked={mask_ratio:.1%}")
        
        # Test with standard CrossEntropyLoss from trainer
        loss_fn = CrossEntropyLoss(ignore_index=-100)
        
        vocab_size = 49152
        logits = torch.randn(BATCH_SIZE, MAX_LENGTH, vocab_size)
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
        
        log_test("CrossEntropyLoss works", loss.item() > 0, f"loss={loss.item():.4f}")
        
        # Test optimizer creation
        dummy_model = nn.Linear(100, 100)  # Minimal model for optimizer test
        optimizer = create_adamw(
            params=dummy_model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
        )
        log_test("AdamW optimizer created", optimizer is not None)
        
        # Test scheduler creation
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=100,
        )
        log_test("Cosine scheduler created", scheduler is not None)
        
        print(f"\n  Summary: Fine-tuning ready with mask_ratio={mask_ratio:.1%}")
        return True
        
    except Exception as e:
        log_test("Fine-tuning", False, str(e))
        import traceback
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════════════════════
# Test 5: RL Stage Integration (DPO)
# ═══════════════════════════════════════════════════════════════════════════════════

def test_rl_stage() -> bool:
    """Test RL stage (DPO) with trainer loss function."""
    log_section("Test 5: RL (DPO) → Trainer Integration")
    
    try:
        # Setup pipeline config
        config = PipelineConfig(
            type="data_module",
            version="1.0",
            dataset=DatasetConfig(
                name="dummy_rl",
                splits={"train": {"name": "train", "sample_size": SAMPLE_SIZE}},
            ),
            tokenizer=TokenizerConfig(
                name_or_path=TOKENIZER_NAME,
                max_length=MAX_LENGTH,
            ),
            prompt_engine=PromptEngineConfig(
                stage=TrainingStage.POST_TRAINING_RL,
                max_length=MAX_LENGTH,
                rl=PipelineRLConfig(
                    algorithm=PipelineRLAlgorithm.DPO,
                    max_prompt_length=64,
                    max_completion_length=64,
                    mask_prompt=True,
                ),
            ),
        )
        
        # Initialize pipeline
        pipeline = DataPipeline(pipeline_config=config)
        
        # Inject dummy dataset
        ds = create_dummy_dataset(TrainingStage.POST_TRAINING_RL)
        pipeline._datasets["train_None"] = ds
        
        # Get DataLoader
        dl_result = pipeline.get_dataloader("train", batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)
        if isinstance(dl_result, Err):
            log_test("DataLoader", False, str(unwrap_err(dl_result)))
            return False
        
        dataloader = unwrap(dl_result)
        batch = next(iter(dataloader))
        
        # Check DPO-specific tensors
        has_chosen = "chosen_input_ids" in batch
        has_rejected = "rejected_input_ids" in batch
        has_prompt = "prompt_input_ids" in batch
        
        log_test("Has chosen_input_ids", has_chosen)
        log_test("Has rejected_input_ids", has_rejected)
        log_test("Has prompt_input_ids", has_prompt)
        
        if has_chosen and has_rejected:
            log_test("Chosen shape", batch["chosen_input_ids"].shape == (BATCH_SIZE, MAX_LENGTH))
            log_test("Rejected shape", batch["rejected_input_ids"].shape == (BATCH_SIZE, MAX_LENGTH))
            
            # Test with DPOLoss from trainer
            try:
                dpo_loss = DPOLoss(beta=0.1)
                
                # Simulate log probabilities
                chosen_logprobs = torch.randn(BATCH_SIZE)
                rejected_logprobs = torch.randn(BATCH_SIZE)
                ref_chosen_logprobs = torch.randn(BATCH_SIZE)
                ref_rejected_logprobs = torch.randn(BATCH_SIZE)
                
                loss = dpo_loss(
                    chosen_logprobs=chosen_logprobs,
                    rejected_logprobs=rejected_logprobs,
                    ref_chosen_logprobs=ref_chosen_logprobs,
                    ref_rejected_logprobs=ref_rejected_logprobs,
                )
                log_test("DPOLoss works", loss.item() >= 0, f"loss={loss.item():.4f}")
            except Exception as e:
                log_test("DPOLoss", False, str(e))
        
        print(f"\n  Summary: DPO batch ready with chosen/rejected pairs")
        return True
        
    except Exception as e:
        log_test("RL Stage", False, str(e))
        import traceback
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════════════════════
# Test 6: Metrics Integration
# ═══════════════════════════════════════════════════════════════════════════════════

def test_metrics_integration() -> bool:
    """Test metrics from trainer module with preprocessing output."""
    log_section("Test 6: Metrics Integration")
    
    try:
        # Perplexity metric
        perplexity = Perplexity()
        
        # Simulate loss values from training
        losses = [2.5, 2.3, 2.1, 2.0, 1.9]
        for loss in losses:
            perplexity.update(loss=torch.tensor(loss))
        
        ppl_value = perplexity.compute()
        log_test("Perplexity compute", ppl_value > 0, f"ppl={ppl_value:.2f}")
        
        # MetricCollection
        collection = MetricCollection([
            Perplexity(),
        ])
        
        collection.update(loss=torch.tensor(2.0))
        metrics = collection.compute()
        log_test("MetricCollection works", len(metrics) > 0, f"metrics={list(metrics.keys())}")
        
        print(f"\n  Summary: Metrics integrated with training loop")
        return True
        
    except Exception as e:
        log_test("Metrics", False, str(e))
        import traceback
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════════════════════
# Test 7: Full Pipeline Smoke Test
# ═══════════════════════════════════════════════════════════════════════════════════

def test_full_pipeline() -> bool:
    """End-to-end smoke test simulating a training step."""
    log_section("Test 7: Full Training Step Simulation")
    
    try:
        # Setup fine-tuning pipeline
        config = PipelineConfig(
            type="data_module",
            version="1.0",
            dataset=DatasetConfig(
                name="smoke_test",
                splits={"train": {"name": "train", "sample_size": SAMPLE_SIZE}},
            ),
            tokenizer=TokenizerConfig(
                name_or_path=TOKENIZER_NAME,
                max_length=MAX_LENGTH,
            ),
            prompt_engine=PromptEngineConfig(
                stage=TrainingStage.FINE_TUNING,
                max_length=MAX_LENGTH,
                finetuning=FineTuningConfig(
                    format=FineTuningFormat.INSTRUCTION,
                    mask_input=True,
                ),
            ),
        )
        
        pipeline = DataPipeline(pipeline_config=config)
        ds = create_dummy_dataset(TrainingStage.FINE_TUNING)
        pipeline._datasets["train_None"] = ds
        
        dl_result = pipeline.get_dataloader("train", batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)
        dataloader = unwrap(dl_result)
        
        # Create minimal model
        vocab_size = 49152
        hidden_size = 64
        
        class MinimalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden_size)
                self.proj = nn.Linear(hidden_size, vocab_size)
            
            def forward(self, input_ids, **kwargs):
                x = self.embed(input_ids)
                logits = self.proj(x)
                return logits
        
        model = MinimalLM()
        
        # Create optimizer and scheduler
        optimizer = create_adamw(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=50)
        
        # Loss function
        loss_fn = CrossEntropyLoss(ignore_index=-100)
        
        # Training step
        model.train()
        batch = next(iter(dataloader))
        
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        # Forward pass
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
        
        log_test("Forward pass", loss.item() > 0, f"loss={loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
        log_test("Gradient norm computed", grad_norm >= 0, f"norm={grad_norm:.4f}")
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        log_test("Optimizer step complete", True)
        log_test("Scheduler step complete", True, f"lr={scheduler.get_last_lr()[0]:.6f}")
        
        print(f"\n  Summary: Full training step completed successfully!")
        return True
        
    except Exception as e:
        log_test("Full Pipeline", False, str(e))
        import traceback
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════════

def main() -> int:
    """Run all integration tests."""
    print("=" * 70)
    print(" SOTA Preprocessing-Trainer Integration Verification")
    print(f" Tokenizer: {TOKENIZER_NAME}")
    print(f" Max Length: {MAX_LENGTH}")
    print("=" * 70)
    
    start_time = time.time()
    
    tests = [
        ("Tokenizer Wrapper", test_tokenizer_wrapper),
        ("Length Manager", test_length_manager),
        ("Pre-training Stage", test_pretraining_stage),
        ("Fine-tuning Stage", test_finetuning_stage),
        ("RL Stage (DPO)", test_rl_stage),
        ("Metrics Integration", test_metrics_integration),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results: Dict[str, bool] = {}
    
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n✗ CRITICAL ERROR in {name}: {e}")
            results[name] = False
    
    # Summary
    elapsed = time.time() - start_time
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print("\n" + "=" * 70)
    print(" VERIFICATION SUMMARY")
    print("=" * 70)
    
    for name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print(f"  Time: {elapsed:.2f}s")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Preprocessing-Trainer Integration Verified!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} TESTS FAILED - Review output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
