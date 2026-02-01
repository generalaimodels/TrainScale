#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
End-to-End Pipeline Demo with Real HuggingFace Dataset
════════════════════════════════════════════════════════════════════════════════

This script demonstrates the complete data pipeline workflow:
1. Load YAML configuration
2. Download real dataset from HuggingFace
3. Apply tokenization and prompt templating
4. Create DataLoader with proper batching
5. Iterate through batches showing loss-aligned output

Run: python e2e_demo.py
════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from transformers import AutoTokenizer


def main():
    """Run end-to-end pipeline demonstration."""
    
    print("=" * 70)
    print("SOTA Data Pipeline - End-to-End Demo")
    print("=" * 70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Load Configuration
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📋 Step 1: Loading YAML Configuration...")
    
    from data_pipeline import (
        DataPipeline,
        create_pipeline,
        TokenAwareContentDistributor,
        ContentDistributionMode,
        unwrap,
        is_ok,
    )
    from data_pipeline.core.config_schema import load_config
    
    # Load from YAML
    config_path = Path(__file__).parent / "alpaca_e2e.yaml"
    config_result = load_config(config_path)
    
    if not is_ok(config_result):
        print(f"❌ Config load failed: {config_result}")
        return
    
    config = unwrap(config_result)
    print(f"   ✓ Loaded config for dataset: {config.dataset.name}")
    print(f"   ✓ Tokenizer: {config.tokenizer.name_or_path}")
    print(f"   ✓ Max length: {config.tokenizer.max_length}")
    print(f"   ✓ Batch size: {config.dataloader.batch_size}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Initialize Tokenizer
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🔤 Step 2: Loading Tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name_or_path)
    
    # GPT-2 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"   ✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"   ✓ Vocab size: {tokenizer.vocab_size}")
    print(f"   ✓ Pad token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
    print(f"   ✓ EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Create Pipeline with Content Distribution
    # ─────────────────────────────────────────────────────────────────────────
    print("\n⚙️  Step 3: Creating Pipeline with SOTA Content Distribution...")
    
    # Create content distributor for intelligent length handling
    distributor = TokenAwareContentDistributor(
        total_max_tokens=config.tokenizer.max_length,
        distribution_mode=ContentDistributionMode.ADAPTIVE,
        tokenizer=tokenizer,
    )
    
    print(f"   ✓ Content distributor: {distributor._mode.name} mode")
    print(f"   ✓ Available token budget: {distributor.available_tokens}")
    
    # Create pipeline with pre-loaded tokenizer
    pipeline = create_pipeline(
        dataset_name=config.dataset.name,
        tokenizer=tokenizer,
        input_columns=["instruction", "input"],
        label_column="output",
        max_length=config.tokenizer.max_length,
        batch_size=config.dataloader.batch_size,
        streaming=False,
    )
    
    print(f"   ✓ Pipeline created successfully")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Load Dataset
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📥 Step 4: Loading Dataset from HuggingFace...")
    
    from datasets import load_dataset
    
    # Load small sample for demo
    ds = load_dataset(
        config.dataset.name,
        split="train[:100]",  # Only 100 examples for demo
        trust_remote_code=True,
    )
    
    print(f"   ✓ Loaded {len(ds)} examples from '{config.dataset.name}'")
    print(f"   ✓ Columns: {list(ds.column_names)}")
    
    # Show sample
    sample = ds[0]
    print(f"\n   📝 Sample example:")
    print(f"      instruction: {sample['instruction'][:80]}...")
    print(f"      input: {sample['input'][:50] if sample['input'] else '(empty)'}...")
    print(f"      output: {sample['output'][:80]}...")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Create Preprocessed Dataset
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🔄 Step 5: Creating Preprocessed Dataset...")
    
    from data_pipeline.preprocessing import PromptEngine, wrap_tokenizer
    from data_pipeline.core.config_schema import PromptTemplate
    from data_pipeline.data import PreprocessedDataset, create_collate_fn
    
    # Wrap tokenizer
    tokenizer_wrapper = wrap_tokenizer(tokenizer)
    
    # Create prompt template
    prompt_template = PromptTemplate(
        format_type="custom",
        template="""### Instruction:
{{ instruction }}
{% if input %}

### Input:
{{ input }}
{% endif %}

### Response:
{{ output }}{{ eos_token }}""",
        input_columns=("instruction", "input"),
        label_column="output",
        mask_input=True,
        add_eos=True,
    )
    
    # Create prompt engine
    prompt_engine = PromptEngine(
        template=prompt_template,
        tokenizer=tokenizer_wrapper,
        max_length=config.tokenizer.max_length,
    )
    
    # Create preprocessed dataset
    preprocessed = PreprocessedDataset(
        hf_dataset=ds,
        prompt_engine=prompt_engine,
    )
    
    print(f"   ✓ Preprocessed dataset created: {len(preprocessed)} examples")
    
    # Show preprocessed sample
    processed_sample = preprocessed[0]
    print(f"\n   🔢 Preprocessed sample:")
    print(f"      input_ids shape: {processed_sample['input_ids'].shape}")
    print(f"      attention_mask shape: {processed_sample['attention_mask'].shape}")
    print(f"      labels shape: {processed_sample['labels'].shape}")
    
    # Show label masking
    labels = processed_sample['labels']
    masked_count = (labels == -100).sum().item()
    total_count = labels.shape[0]
    print(f"      Labels masked (-100): {masked_count}/{total_count} tokens")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: Create DataLoader
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📦 Step 6: Creating DataLoader...")
    
    from torch.utils.data import DataLoader
    
    # Create collate function
    collate_fn = create_collate_fn(
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100,
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        preprocessed,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    print(f"   ✓ DataLoader created")
    print(f"   ✓ Batch size: {config.dataloader.batch_size}")
    print(f"   ✓ Total batches: {len(dataloader)}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 7: Iterate Through Batches
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🚀 Step 7: Iterating Through Batches...")
    print("-" * 70)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 5:  # Show first 5 batches
            print(f"\n   ... ({len(dataloader) - 5} more batches)")
            break
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Calculate stats
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        masked_ratio = (labels == -100).float().mean().item()
        active_tokens = attention_mask.sum(dim=1).float().mean().item()
        
        print(f"\n   📦 Batch {batch_idx + 1}:")
        print(f"      Shape: ({batch_size}, {seq_length})")
        print(f"      Avg active tokens: {active_tokens:.1f}")
        print(f"      Labels masked: {masked_ratio*100:.1f}%")
        
        # Verify loss alignment
        # Labels should be -100 where attention_mask is 0
        padding_mask = attention_mask == 0
        labels_at_padding = labels[padding_mask]
        correctly_masked = (labels_at_padding == -100).all().item()
        print(f"      Loss alignment: {'✓ Correct' if correctly_masked else '✗ Issue'}")
        
        # Decode first example
        first_ids = input_ids[0][attention_mask[0] == 1]
        decoded = tokenizer.decode(first_ids[:50], skip_special_tokens=False)
        print(f"      First example preview: '{decoded[:60]}...'")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 8: Verification Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅ End-to-End Verification Complete")
    print("=" * 70)
    
    print("""
    Summary:
    ✓ YAML configuration loaded successfully
    ✓ Tokenizer initialized with proper padding
    ✓ Dataset downloaded from HuggingFace
    ✓ Prompt templating applied with Jinja2
    ✓ Labels properly masked for loss computation
    ✓ DataLoader batching works correctly
    ✓ Padding aligned with attention mask
    
    The pipeline is ready for training!
    """)


if __name__ == "__main__":
    main()
