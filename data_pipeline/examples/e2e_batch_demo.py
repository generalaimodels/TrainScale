#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
End-to-End Pipeline Demo with YAML Config - Model-Ready Tensors
════════════════════════════════════════════════════════════════════════════════

This script demonstrates:
1. Loading configuration from example_config.yaml
2. Real-time batch processing with HuggingFace dataset
3. Returning tensors ready for model.forward()

Run: python e2e_batch_demo.py
════════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yaml
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class Config:
    """Simple config loader from YAML."""
    dataset_name: str
    sample_size: int
    tokenizer_name: str
    max_length: int
    batch_size: int
    template: str
    input_columns: tuple
    label_column: str
    mask_input: bool
    # Hardware config (user-defined)
    device: str
    dtype: str
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # Hardware config with defaults
        hardware = data.get("hardware", {})
        
        return cls(
            dataset_name=data["dataset"]["name"],
            sample_size=data["dataset"]["splits"]["train"].get("sample_size", 50),
            tokenizer_name=data["tokenizer"]["name_or_path"],
            max_length=data["tokenizer"]["max_length"],
            batch_size=data["dataloader"]["batch_size"],
            template=data["prompt"]["template"],
            input_columns=tuple(data["prompt"]["input_columns"]),
            label_column=data["prompt"]["label_column"],
            mask_input=data["prompt"].get("mask_input", True),
            device=hardware.get("device", "auto"),
            dtype=hardware.get("dtype", "float32"),
        )
    
    def get_device(self) -> torch.device:
        """Get torch device based on config."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.device)
    
    def get_dtype(self) -> torch.dtype:
        """Get torch dtype based on config."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
        }
        return dtype_map.get(self.dtype, torch.float32)


def get_model_inputs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Extract model-ready inputs from batch.
    
    Returns dict that can be unpacked directly into model.forward(**inputs).
    """
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],  # For loss computation
    }


def main():
    print("=" * 70)
    print("🚀 SOTA Data Pipeline - YAML Config + Model-Ready Tensors")
    print("=" * 70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Load YAML Configuration
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📋 Step 1: Loading example_config.yaml...")
    
    config_path = Path(__file__).parent / "example_config.yaml"
    config = Config.from_yaml(config_path)
    
    # Get user-configured device and dtype
    device = config.get_device()
    dtype = config.get_dtype()
    
    print(f"   ✓ Dataset: {config.dataset_name}")
    print(f"   ✓ Sample size: {config.sample_size}")
    print(f"   ✓ Tokenizer: {config.tokenizer_name}")
    print(f"   ✓ Max length: {config.max_length}")
    print(f"   ✓ Batch size: {config.batch_size}")
    print(f"   ✓ Device: {device} (config: {config.device})")
    print(f"   ✓ Dtype: {dtype} (config: {config.dtype})")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Load Tokenizer
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🔤 Step 2: Loading Tokenizer...")
    
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"   ✓ Loaded: {tokenizer.__class__.__name__}")
    print(f"   ✓ Vocab: {tokenizer.vocab_size}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Download Dataset
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📥 Step 3: Downloading Dataset...")
    
    from datasets import load_dataset
    
    dataset = load_dataset(
        config.dataset_name,
        split=f"train[:{config.sample_size}]",
    )
    
    print(f"   ✓ Downloaded: {len(dataset)} examples")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Setup Pipeline Components
    # ─────────────────────────────────────────────────────────────────────────
    print("\n⚙️  Step 4: Setting up Pipeline...")
    
    from data_pipeline.preprocessing import PromptEngine, wrap_tokenizer
    from data_pipeline.core.config_schema import PromptTemplate
    from data_pipeline.data import PreprocessedDataset, create_collate_fn
    
    tokenizer_wrapper = wrap_tokenizer(tokenizer)
    
    # Create prompt template from YAML config
    template = PromptTemplate(
        format_type="custom",
        template=config.template,
        input_columns=config.input_columns,
        label_column=config.label_column,
        mask_input=config.mask_input,
        add_eos=True,
    )
    
    prompt_engine = PromptEngine(
        template=template,
        tokenizer=tokenizer_wrapper,
        max_length=config.max_length,
    )
    
    preprocessed = PreprocessedDataset(
        hf_dataset=dataset,
        prompt_engine=prompt_engine,
    )
    
    print(f"   ✓ Preprocessed: {len(preprocessed)} examples")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Create DataLoader
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📦 Step 5: Creating DataLoader...")
    
    collate_fn = create_collate_fn(
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100,
    )
    
    dataloader = DataLoader(
        preprocessed,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    print(f"   ✓ Total batches: {len(dataloader)}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: Iterate Batches - Show Model-Ready Tensors
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🔄 Real-Time Batch Processing → Model-Ready Tensors")
    print("=" * 70)
    
    total_tokens = 0
    total_loss_tokens = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Get model-ready inputs
        model_inputs = get_model_inputs(batch)
        
        # Move to user-configured device (hardware-dependent)
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)
        labels = model_inputs["labels"].to(device)
        
        # Stats
        batch_size, seq_len = input_ids.shape
        active_tokens = attention_mask.sum().item()
        loss_tokens = (labels != -100).sum().item()
        
        total_tokens += active_tokens
        total_loss_tokens += loss_tokens
        
        print(f"\n┌─ Batch {batch_idx + 1}/{len(dataloader)} ─────────────────────")
        print(f"│  📊 Tensor Shapes (ready for model.forward()):")
        print(f"│     input_ids:      {tuple(input_ids.shape)} dtype={input_ids.dtype} device={input_ids.device}")
        print(f"│     attention_mask: {tuple(attention_mask.shape)} dtype={attention_mask.dtype} device={attention_mask.device}")
        print(f"│     labels:         {tuple(labels.shape)} dtype={labels.dtype} device={labels.device}")
        print(f"│")
        print(f"│  📈 Stats:")
        print(f"│     Active tokens: {active_tokens}")
        print(f"│     Loss tokens:   {loss_tokens} ({loss_tokens/active_tokens*100:.1f}%)")
        print(f"│")
        print(f"│  🖥️  Hardware: device={device}, model_dtype={dtype}")
        print(f"│  ✓ Tensors ready: model(**batch) compatible")
        print(f"└────────────────────────────────────────────────")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 7: Summary with Model Usage Example
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅ Pipeline Complete - Model-Ready Output")
    print("=" * 70)
    print(f"""
    Pipeline Statistics:
    ─────────────────────
    • Config: example_config.yaml
    • Dataset: {config.dataset_name}
    • Batches: {len(dataloader)}
    • Total tokens: {total_tokens:,}
    • Loss tokens: {total_loss_tokens:,}
    
    Model Usage:
    ────────────
    ```python
    for batch in dataloader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],  # Auto-computes loss
        )
        loss = outputs.loss
        loss.backward()
    ```
    
    Tensor Format (per batch):
    ──────────────────────────
    • input_ids:      (batch_size, seq_len) torch.long
    • attention_mask: (batch_size, seq_len) torch.long
    • labels:         (batch_size, seq_len) torch.long
                      (-100 = masked, won't contribute to loss)
    
    🎉 Ready for Training!
    """)


if __name__ == "__main__":
    main()
