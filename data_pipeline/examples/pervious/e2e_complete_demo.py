#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════════
# Complete End-to-End Pipeline Demo
# ════════════════════════════════════════════════════════════════════════════════
# Uses ALL modules with YAML-only configuration:
#   - core: Config schema, types, errors
#   - introspection: Dataset discovery
#   - preprocessing: Prompt engine, tokenization
#   - data: DataLoader factory
#   - trainer: SOTA training
#
# ALL inputs from YAML - NO hardcoding.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ═════════════════════════════════════════════════════════════════════════════════
# Path Setup (for running as script)
# ═════════════════════════════════════════════════════════════════════════════════
# Add TrainScale root to path for data_pipeline imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_TRAINSCALE_ROOT = _SCRIPT_DIR.parent.parent  # TrainScale/
if str(_TRAINSCALE_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINSCALE_ROOT))

import yaml
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ═════════════════════════════════════════════════════════════════════════════════
# Logging Setup
# ═════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("e2e_pipeline")


# ═════════════════════════════════════════════════════════════════════════════════
# Unified Configuration Loader
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class UnifiedConfig:
    """
    Complete unified configuration from single YAML.
    
    Covers:
    - Dataset (name, splits, columns)
    - Introspection (auto-discovery settings)
    - Tokenizer (name, max_length, special tokens)
    - Preprocessing (prompt template, packing)
    - DataLoader (batch_size, workers, etc.)
    - Training (mode, optimizer, scheduler, hardware)
    - Export (format, push_to_hub)
    """
    # Raw data
    raw: Dict[str, Any] = field(default_factory=dict)
    
    # Meta
    type: str = "data_module"
    version: str = "2.0"
    
    @classmethod
    def from_yaml(cls, path: str) -> "UnifiedConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        config = cls(raw=data)
        config.type = data.get("type", "data_module")
        config.version = data.get("version", "2.0")
        return config
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Accessors
    # ─────────────────────────────────────────────────────────────────────────────
    
    @property
    def dataset(self) -> Dict[str, Any]:
        return self.raw.get("dataset", {})
    
    @property
    def introspection(self) -> Dict[str, Any]:
        return self.raw.get("introspection", {})
    
    @property
    def tokenizer(self) -> Dict[str, Any]:
        return self.raw.get("tokenizer", {})
    
    @property
    def prompt_template(self) -> Dict[str, Any]:
        return self.raw.get("prompt_template", {})
    
    @property
    def preprocessing(self) -> Dict[str, Any]:
        return self.raw.get("preprocessing", {})
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return self.raw.get("output_schema", {})
    
    @property
    def dataloader(self) -> Dict[str, Any]:
        return self.raw.get("dataloader", {})
    
    @property
    def training(self) -> Dict[str, Any]:
        return self.raw.get("training", {})
    
    @property
    def export(self) -> Dict[str, Any]:
        return self.raw.get("export", {})
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Convenience Methods
    # ─────────────────────────────────────────────────────────────────────────────
    
    def get_dataset_name(self) -> str:
        return self.dataset.get("name", "")
    
    def get_splits(self) -> Dict[str, Dict[str, Any]]:
        return self.dataset.get("splits", {})
    
    def get_column_mapping(self) -> Dict[str, str]:
        return self.dataset.get("column_mapping", {})
    
    def get_tokenizer_name(self) -> str:
        return self.tokenizer.get("name_or_path", "")
    
    def get_max_length(self) -> int:
        return self.tokenizer.get("max_length", 4096)
    
    def get_batch_size(self) -> int:
        return self.dataloader.get("batch_size", 4)


# ═════════════════════════════════════════════════════════════════════════════════
# Pipeline Orchestrator
# ═════════════════════════════════════════════════════════════════════════════════

class E2EPipeline:
    """
    Complete End-to-End Pipeline.
    
    Orchestrates:
    1. Config loading (YAML)
    2. Dataset introspection
    3. Dataset loading with discovered splits
    4. Tokenizer initialization
    5. Preprocessing (prompt engine)
    6. DataLoader creation
    7. Training (optional)
    8. Export (optional)
    """
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline from YAML config.
        
        Args:
            config_path: Path to complete_pipeline.yaml
        """
        logger.info(f"Loading config from: {config_path}")
        self.config = UnifiedConfig.from_yaml(config_path)
        
        # Components (initialized lazily)
        self._tokenizer = None
        self._dataset = None
        self._prompt_engine = None
        self._dataloader = None
        self._discovered_metadata = None
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Step 1: Introspection
    # ═════════════════════════════════════════════════════════════════════════════
    
    def introspect_dataset(self) -> Dict[str, Any]:
        """
        Discover dataset structure automatically.
        
        Returns:
            Discovered metadata including splits and columns
        """
        from data_pipeline.introspection import DatasetIntrospector
        from data_pipeline.core.types import is_err, is_ok, Ok, Err
        
        dataset_name = self.config.get_dataset_name()
        intro_cfg = self.config.introspection
        
        if not intro_cfg.get("enabled", True):
            logger.info("Introspection disabled, using config as-is")
            return {
                "splits": list(self.config.get_splits().keys()),
                "columns": intro_cfg.get("fallback_columns", []),
            }
        
        logger.info(f"Introspecting dataset: {dataset_name}")
        introspector = DatasetIntrospector()
        
        result = introspector.discover(
            dataset_name,
            trust_remote_code=intro_cfg.get("trust_remote_code", False),
        )
        
        if isinstance(result, Err):
            logger.warning(f"Introspection failed: {result.error}")
            return {
                "splits": list(self.config.get_splits().keys()),
                "columns": intro_cfg.get("fallback_columns", []),
            }
        
        metadata = result.value
        self._discovered_metadata = metadata
        
        # Extract discovered info
        discovered = {
            "dataset_id": metadata.dataset_id,
            "configs": [c.name for c in metadata.configs],
            "default_config": metadata.default_config,
            "splits": list(metadata.get_all_splits()),
            "columns": [f.name for f in metadata.features],
            "description": metadata.description,
        }
        
        logger.info(f"Discovered splits: {discovered['splits']}")
        logger.info(f"Discovered columns: {discovered['columns']}")
        
        return discovered
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Step 2: Load Dataset
    # ═════════════════════════════════════════════════════════════════════════════
    
    def load_dataset(self, split: str = "train") -> Any:
        """
        Load dataset split based on config.
        
        Args:
            split: Which split to load (from config's splits)
        
        Returns:
            HuggingFace Dataset
        """
        from datasets import load_dataset
        
        dataset_cfg = self.config.dataset
        split_cfg = self.config.get_splits().get(split, {})
        
        # Get actual HF split name
        hf_split_name = split_cfg.get("name", split)
        sample_size = split_cfg.get("sample_size")
        
        logger.info(f"Loading dataset: {dataset_cfg['name']} split={hf_split_name}")
        
        # Build split string with optional sample limit
        split_str = hf_split_name
        if sample_size:
            split_str = f"{hf_split_name}[:{sample_size}]"
        
        # Load from HF
        ds = load_dataset(
            dataset_cfg["name"],
            name=dataset_cfg.get("config_name"),
            split=split_str,
            streaming=dataset_cfg.get("streaming", False),
            revision=dataset_cfg.get("revision"),
            trust_remote_code=self.config.introspection.get("trust_remote_code", False),
        )
        
        # Shuffle if configured
        if split_cfg.get("shuffle", False):
            seed = split_cfg.get("seed", 42)
            ds = ds.shuffle(seed=seed)
        
        # Apply column mapping
        column_mapping = self.config.get_column_mapping()
        if column_mapping:
            logger.info(f"Applying column mapping: {column_mapping}")
            # Rename columns if needed
            for src, dst in column_mapping.items():
                if src in ds.column_names and src != dst:
                    ds = ds.rename_column(src, dst)
        
        # Filter columns if specified
        columns_to_keep = dataset_cfg.get("columns", [])
        if columns_to_keep:
            ds = ds.select_columns(columns_to_keep)
        
        self._dataset = ds
        logger.info(f"Loaded {len(ds) if hasattr(ds, '__len__') else 'streaming'} examples")
        
        return ds
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Step 3: Initialize Tokenizer
    # ═════════════════════════════════════════════════════════════════════════════
    
    def init_tokenizer(self):
        """
        Initialize tokenizer from config.
        
        Returns:
            TokenizerWrapper
        """
        from data_pipeline.preprocessing import TokenizerWrapper
        
        tok_cfg = self.config.tokenizer
        
        logger.info(f"Initializing tokenizer: {tok_cfg['name_or_path']}")
        
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            tok_cfg["name_or_path"],
            trust_remote_code=self.config.introspection.get("trust_remote_code", False),
        )
        
        # Set special tokens
        special_tokens = tok_cfg.get("special_tokens", {})
        if special_tokens:
            if "pad_token" in special_tokens and tokenizer.pad_token is None:
                tokenizer.pad_token = special_tokens["pad_token"]
            if "eos_token" in special_tokens:
                tokenizer.eos_token = special_tokens["eos_token"]
            if "bos_token" in special_tokens:
                tokenizer.bos_token = special_tokens["bos_token"]
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Set padding/truncation sides
        tokenizer.padding_side = tok_cfg.get("padding_side", "right")
        tokenizer.truncation_side = tok_cfg.get("truncation_side", "right")
        
        # Wrap in TokenizerWrapper using wrap_tokenizer helper
        from data_pipeline.preprocessing import wrap_tokenizer
        
        wrapper = wrap_tokenizer(
            tokenizer,
            max_length=tok_cfg.get("max_length", 4096),
            padding=tok_cfg.get("padding", "max_length"),
            truncation=tok_cfg.get("truncation", True),
            padding_side=tok_cfg.get("padding_side", "right"),
        )
        self._tokenizer = wrapper
        
        logger.info(f"Tokenizer initialized: vocab_size={tokenizer.vocab_size}")
        
        return wrapper
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Step 4: Initialize Prompt Engine
    # ═════════════════════════════════════════════════════════════════════════════
    # Step 4: Initialize Prompt Engine & Length Manager (SOTA)
    # ═════════════════════════════════════════════════════════════════════════════
    
    def init_prompt_engine(self):
        """
        Initialize prompt engine from config with SOTA preprocessing.
        
        Supports:
        - YAML-driven prompt templates (chat, completion, custom Jinja2)
        - Per-column length limits from config
        - Token-aware content distribution
        - Smart truncation (sentence/word boundaries)
        - Dynamic max_length handling
        
        Returns:
            PromptEngine
        """
        from data_pipeline.preprocessing import (
            PromptEngine,
            create_length_manager,
            TokenAwareContentDistributor,
            ContentDistributionMode,
        )
        from data_pipeline.core.config_schema import PromptTemplate
        
        if self._tokenizer is None:
            self.init_tokenizer()
        
        pt_cfg = self.config.prompt_template
        
        logger.info(f"Initializing prompt engine: format={pt_cfg.get('format_type', 'custom')}")
        
        # ─────────────────────────────────────────────────────────────────────
        # Build PromptTemplate from YAML config (SOTA: fully configurable)
        # ─────────────────────────────────────────────────────────────────────
        template = PromptTemplate(
            format_type=pt_cfg.get("format_type", "custom"),
            template=pt_cfg.get("template"),
            system_message=pt_cfg.get("system_message"),
            user_template=pt_cfg.get("user_template"),
            assistant_template=pt_cfg.get("assistant_template"),
            input_columns=pt_cfg.get("input_columns", []),
            label_column=pt_cfg.get("label_column"),
            mask_input=pt_cfg.get("mask_input", True),
            add_bos=pt_cfg.get("add_bos", True),
            add_eos=pt_cfg.get("add_eos", True),
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # Build LengthManager from YAML config (SOTA: content distribution)
        # ─────────────────────────────────────────────────────────────────────
        length_manager = None
        per_column_limits = None
        prep_cfg = self.config.preprocessing.get("length_manager", {})
        
        if prep_cfg.get("enabled", False):
            # Get max_length from tokenizer config or preprocessing config
            max_length = prep_cfg.get("max_total_length", self.config.get_max_length())
            
            # Per-column limits from YAML
            per_column_limits = prep_cfg.get("per_column_limits", {})
            
            # Truncation strategy from YAML
            truncation_strategy = prep_cfg.get("truncation_strategy", "smart")
            
            # Padding strategy from YAML
            padding_strategy = prep_cfg.get("padding_strategy", "longest")
            
            # Create LengthManager using factory function
            length_manager = create_length_manager(
                max_length=max_length,
                padding_strategy=padding_strategy,
                default_truncation=truncation_strategy,
                per_column_limits=per_column_limits,
            )
            
            logger.info(f"LengthManager initialized: max_length={max_length}, "
                       f"truncation={truncation_strategy}, per_column_limits={per_column_limits}")
        
        # ─────────────────────────────────────────────────────────────────────
        # Check for SOTA content distribution (advanced feature)
        # ─────────────────────────────────────────────────────────────────────
        content_distributor = None
        distrib_cfg = self.config.preprocessing.get("content_distribution", {})
        
        if distrib_cfg.get("enabled", False):
            mode_str = distrib_cfg.get("mode", "proportional").upper()
            mode = getattr(ContentDistributionMode, mode_str, ContentDistributionMode.PROPORTIONAL)
            
            content_distributor = TokenAwareContentDistributor(
                total_max_tokens=self.config.get_max_length(),
                distribution_mode=mode,
                column_ratios=distrib_cfg.get("column_ratios", {}),
                special_tokens_budget=distrib_cfg.get("special_tokens_budget", 10),
                tokenizer=self._tokenizer.tokenizer if self._tokenizer else None,
            )
            
            logger.info(f"TokenAwareContentDistributor initialized: mode={mode_str}")
        
        # ─────────────────────────────────────────────────────────────────────
        # Create PromptEngine with all SOTA features
        # ─────────────────────────────────────────────────────────────────────
        engine = PromptEngine(
            template=template,
            tokenizer=self._tokenizer,
            length_manager=length_manager,
            max_length=self.config.get_max_length(),
            per_column_limits=per_column_limits or pt_cfg.get("per_column_limits", {}),
        )
        
        self._prompt_engine = engine
        
        logger.info("Prompt engine initialized")
        
        return engine
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Step 5: Process Dataset
    # ═════════════════════════════════════════════════════════════════════════════
    
    def process_dataset(self, dataset=None) -> Dataset:
        """
        Process dataset through prompt engine.
        
        Args:
            dataset: Optional dataset (uses loaded dataset if None)
        
        Returns:
            Processed PyTorch Dataset
        """
        if dataset is None:
            dataset = self._dataset
        
        if self._prompt_engine is None:
            self.init_prompt_engine()
        
        logger.info("Processing dataset through prompt engine...")
        
        # Process each example
        def process_fn(example):
            from data_pipeline.core.types import Err
            result = self._prompt_engine.process(example)
            if isinstance(result, Err):
                logger.warning(f"Failed to process example: {result.error}")
                return None
            
            processed = result.value
            return {
                "input_ids": processed.input_ids,
                "attention_mask": processed.attention_mask,
                "labels": processed.labels,
            }
        
        # Map over dataset
        processed = dataset.map(
            process_fn,
            remove_columns=dataset.column_names,
            desc="Processing",
        )
        
        # Filter failed examples
        processed = processed.filter(lambda x: x is not None)
        
        logger.info(f"Processed {len(processed)} examples")
        
        return processed
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Step 6: Create DataLoader
    # ═════════════════════════════════════════════════════════════════════════════
    
    def create_dataloader(self, dataset=None) -> DataLoader:
        """
        Create DataLoader from config.
        
        Args:
            dataset: Optional processed dataset
        
        Returns:
            PyTorch DataLoader
        """
        from data_pipeline.data import DataLoaderBuilder
        from data_pipeline.core.config_schema import DataLoaderConfig, OutputSchema
        
        if dataset is None:
            dataset = self._dataset
        
        dl_cfg = self.config.dataloader
        
        logger.info(f"Creating DataLoader: batch_size={dl_cfg.get('batch_size', 4)}")
        
        config = DataLoaderConfig(
            batch_size=dl_cfg.get("batch_size", 4),
            num_workers=dl_cfg.get("num_workers", 4),
            pin_memory=dl_cfg.get("pin_memory", True),
            drop_last=dl_cfg.get("drop_last", False),
            shuffle=dl_cfg.get("shuffle", True),
            prefetch_factor=dl_cfg.get("prefetch_factor", 2),
            persistent_workers=dl_cfg.get("persistent_workers", True),
        )
        
        # Build output schema
        out_cfg = self.config.output_schema
        schema = OutputSchema()  # Uses defaults
        
        # Get pad token id
        pad_token_id = 0
        if self._tokenizer:
            pad_token_id = self._tokenizer.pad_token_id or 0
        
        builder = DataLoaderBuilder()
        dataloader = (
            builder
            .with_dataset(dataset)
            .with_config(config)
            .with_padding(pad_token_id=pad_token_id, label_pad_token_id=-100)
            .with_output_schema(schema)
            .build()
        )
        
        self._dataloader = dataloader
        
        logger.info(f"DataLoader created: {len(dataloader)} batches")
        
        return dataloader
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Step 7: Run Training (Optional)
    # ═════════════════════════════════════════════════════════════════════════════
    
    def train(self, dataloader=None) -> Dict[str, float]:
        """
        Run training based on config.
        
        Args:
            dataloader: Optional DataLoader
        
        Returns:
            Training metrics
        """
        from data_pipeline.trainer import SOTAConfig, SOTATrainer
        
        if dataloader is None:
            dataloader = self._dataloader
        
        train_cfg = self.config.training
        
        if not train_cfg:
            logger.info("No training config, skipping training")
            return {}
        
        logger.info(f"Training mode: {train_cfg.get('mode', 'qlora')}")
        
        # Build SOTA config from training section
        sota_config = SOTAConfig()
        
        # Apply training settings
        # (In full implementation, map all training config fields)
        
        # Create trainer
        trainer = SOTATrainer(sota_config)
        
        # Train
        metrics = trainer.train(dataloader)
        
        logger.info(f"Training complete: {metrics}")
        
        return metrics
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Full Pipeline Run
    # ═════════════════════════════════════════════════════════════════════════════
    
    def run(self, split: str = "train", train: bool = False):
        """
        Run the complete pipeline.
        
        Args:
            split: Which split to process
            train: Whether to run training
        
        Returns:
            Pipeline results
        """
        logger.info("=" * 60)
        logger.info("Starting E2E Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Introspect
        discovered = self.introspect_dataset()
        logger.info(f"Available splits: {discovered['splits']}")
        logger.info(f"Available columns: {discovered['columns']}")
        
        # Step 2: Load dataset
        dataset = self.load_dataset(split)
        
        # Step 3: Initialize tokenizer
        self.init_tokenizer()
        
        # Step 4: Initialize prompt engine
        self.init_prompt_engine()
        
        # Step 5: Process dataset
        processed = self.process_dataset(dataset)
        
        # Step 6: Create DataLoader
        dataloader = self.create_dataloader(processed)
        
        # Verify output
        logger.info("=" * 60)
        logger.info("Pipeline Output Verification")
        logger.info("=" * 60)
        
        batch = next(iter(dataloader))
        
        logger.info(f"Batch keys: {list(batch.keys())}")
        logger.info(f"input_ids shape: {batch['input_ids'].shape}")
        logger.info(f"attention_mask shape: {batch['attention_mask'].shape}")
        logger.info(f"labels shape: {batch['labels'].shape}")
        logger.info(f"input_ids dtype: {batch['input_ids'].dtype}")
        logger.info(f"labels dtype: {batch['labels'].dtype}")
        
        # Count valid labels
        valid_labels = (batch['labels'] != -100).sum().item()
        total_labels = batch['labels'].numel()
        logger.info(f"Valid labels: {valid_labels}/{total_labels} ({100*valid_labels/total_labels:.1f}%)")
        
        # Step 7: Train (optional)
        if train:
            metrics = self.train(dataloader)
            return {
                "discovered": discovered,
                "dataloader": dataloader,
                "batch_sample": batch,
                "metrics": metrics,
            }
        
        logger.info("=" * 60)
        logger.info("Pipeline Complete - Ready for Training")
        logger.info("=" * 60)
        
        return {
            "discovered": discovered,
            "dataloader": dataloader,
            "batch_sample": batch,
        }


# ═════════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═════════════════════════════════════════════════════════════════════════════════

def main():
    """Run the E2E pipeline demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="E2E Pipeline Demo")
    parser.add_argument(
        "--config",
        type=str,
        default="data_pipeline/examples/complete_pipeline.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which split to process",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training after data processing",
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = E2EPipeline(args.config)
    results = pipeline.run(split=args.split, train=args.train)
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"Discovered splits: {results['discovered']['splits']}")
    print(f"Discovered columns: {results['discovered']['columns']}")
    
    batch = results['batch_sample']
    print(f"\nBatch Output:")
    print(f"  input_ids: {batch['input_ids'].shape} ({batch['input_ids'].dtype})")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    
    print("\n✅ Pipeline complete! Tensors are ready for model.forward()")


if __name__ == "__main__":
    main()
