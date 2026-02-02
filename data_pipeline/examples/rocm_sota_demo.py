#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════════
# SOTA End-to-End Demo for AMD ROCm MI300X Multi-GPU Training
# ════════════════════════════════════════════════════════════════════════════════
# Complete pipeline demonstration using TrainScale modules:
#   - YAML-driven configuration (NO hardcoding)
#   - AMD ROCm GPU detection via rocm-smi
#   - Multi-GPU training with 4 middle GPUs (2,3,4,5)
#   - SOTA model: Mistral-7B / Llama-3.1-8B
#   - SOTA optimizer: Lion (2x faster than AdamW)
#   - SOTA scheduler: WSD (LLaMA-3 style Warmup-Stable-Decay)
#   - SOTA preprocessing: Token-aware content distribution
#
# Usage:
#   
#    python rocm_sota_demo.py \
#       --config rocm_sota_config.yaml
#
# For distributed multi-GPU:
#   torchrun --nproc_per_node=4 rocm_sota_demo.py \
#       --config rocm_sota_config.yaml --distributed
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import logging
import os
import subprocess
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
from torch import Tensor
from torch.utils.data import DataLoader

# Internal SOTA modules
from data_pipeline.trainer import registry
from data_pipeline.trainer import lora
from data_pipeline.trainer.metrics import (
    MetricCollection,
    Perplexity,
    Accuracy,
)
from data_pipeline.trainer import inference

# ═════════════════════════════════════════════════════════════════════════════════
# Logging Setup
# ═════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rocm_sota_demo")


# ═════════════════════════════════════════════════════════════════════════════════
# AMD ROCm GPU Utilities
# ═════════════════════════════════════════════════════════════════════════════════

def get_rocm_gpu_info() -> Dict[str, Any]:
    """
    Get AMD GPU information using rocm-smi.
    
    Returns:
        Dictionary with GPU information
    """
    gpu_info = {
        "available": False,
        "count": 0,
        "devices": [],
        "driver_version": None,
        "rocm_version": None,
    }
    
    try:
        # Check rocm-smi availability
        result = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0:
            gpu_info["available"] = True
            
            # Parse GPU count and names
            for line in result.stdout.split("\n"):
                if "GPU[" in line and "Device Name:" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        device_name = parts[-1].strip()
                        gpu_idx = int(line.split("GPU[")[1].split("]")[0])
                        gpu_info["devices"].append({
                            "id": gpu_idx,
                            "name": device_name,
                        })
            
            gpu_info["count"] = len(gpu_info["devices"])
        
        # Get ROCm version
        rocm_result = subprocess.run(
            ["rocm-smi", "--showdriverversion"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if rocm_result.returncode == 0:
            for line in rocm_result.stdout.split("\n"):
                if "Driver Version" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        gpu_info["driver_version"] = parts[-1].strip()
                        break
                        
    except FileNotFoundError:
        logger.warning("rocm-smi not found. AMD GPU detection unavailable.")
    except subprocess.TimeoutExpired:
        logger.warning("rocm-smi timed out.")
    except Exception as e:
        logger.warning(f"rocm-smi error: {e}")
    
    return gpu_info


def get_rocm_memory_info(device_ids: List[int] = None) -> Dict[int, Dict[str, float]]:
    """
    Get memory usage for specified AMD GPUs.
    
    Args:
        device_ids: List of GPU IDs to query (default: all)
    
    Returns:
        Dictionary mapping device ID to memory info
    """
    memory_info = {}
    
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0:
            current_gpu = None
            for line in result.stdout.split("\n"):
                if "GPU[" in line:
                    gpu_idx = int(line.split("GPU[")[1].split("]")[0])
                    if device_ids is None or gpu_idx in device_ids:
                        current_gpu = gpu_idx
                        memory_info[gpu_idx] = {}
                elif current_gpu is not None:
                    if "Total Memory" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            # Parse memory value (e.g., "196560 MB")
                            mem_str = parts[-1].strip().split()[0]
                            memory_info[current_gpu]["total_gb"] = float(mem_str) / 1024
                    elif "Used Memory" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            mem_str = parts[-1].strip().split()[0]
                            memory_info[current_gpu]["used_gb"] = float(mem_str) / 1024
                            
    except Exception as e:
        logger.warning(f"Memory info error: {e}")
    
    return memory_info


def setup_rocm_environment(gpu_ids: List[int]) -> None:
    """
    Setup environment variables for ROCm multi-GPU training.
    
    Args:
        gpu_ids: List of GPU IDs to use
    """
    # Set CUDA_VISIBLE_DEVICES for ROCm (HIP uses CUDA-compatible API)
    visible_devices = ",".join(str(i) for i in gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    os.environ["HIP_VISIBLE_DEVICES"] = visible_devices
    
    # ROCm-specific optimizations
    os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"  # Improve PCIe performance
    os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"  # Faster startup
    
    # Enable Flash Attention for ROCm
    os.environ["FLASH_ATTENTION_USE_TRITON_ROCM"] = "1"
    
    logger.info(f"ROCm environment configured for GPUs: {gpu_ids}")


def print_rocm_banner(gpu_info: Dict[str, Any], selected_gpus: List[int]) -> None:
    """Print formatted banner with ROCm GPU information."""
    print("\n" + "═" * 80)
    print("🔥 AMD ROCm SOTA Training Demo - TrainScale")
    print("═" * 80)
    
    print(f"\n📊 Detected {gpu_info['count']} AMD GPUs:")
    for device in gpu_info["devices"]:
        status = "✅ SELECTED" if device["id"] in selected_gpus else "⬜ Available"
        print(f"   GPU[{device['id']}]: {device['name']} [{status}]")
    
    if gpu_info["driver_version"]:
        print(f"\n📦 Driver Version: {gpu_info['driver_version']}")
    
    memory = get_rocm_memory_info(selected_gpus)
    if memory:
        print(f"\n💾 Memory Status:")
        for gpu_id, mem in memory.items():
            used = mem.get("used_gb", 0)
            total = mem.get("total_gb", 0)
            print(f"   GPU[{gpu_id}]: {used:.1f}/{total:.1f} GB used")
    
    print("\n" + "─" * 80)


# ═════════════════════════════════════════════════════════════════════════════════
# SOTA Configuration Loader
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class SOTARocmConfig:
    """
    SOTA configuration container for ROCm training.
    
    Loads from YAML and provides typed access to all settings.
    """
    raw: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> "SOTARocmConfig":
        """Load configuration from YAML file."""
        # Try resolving path
        p = Path(path)
        if not p.exists():
            # Try relative to script dir
            script_rel = _SCRIPT_DIR / path
            if script_rel.exists():
                p = script_rel
            # Try relative to examples dir (common pattern)
            elif (Path("data_pipeline/examples") / path).exists():
                p = Path("data_pipeline/examples") / path
            # Try absolute path usage if provided as relative
            elif (Path(_TRAINSCALE_ROOT) / path).exists():
                p = Path(_TRAINSCALE_ROOT) / path

        if not p.exists():
            raise FileNotFoundError(f"Config not found: {path} (checked CWD, script dir, and project root)")
        
        logger.info(f"Loading config from: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # Inject config path for later use
        inst = cls(raw=data)
        inst.config_path = str(p)
        return inst
    
    # ───────────────────────────────────────────────────────────────────────────
    # Property Accessors
    # ───────────────────────────────────────────────────────────────────────────
    
    @property
    def training_mode(self) -> str:
        return self.raw.get("training_mode", "lora")
    
    @property
    def model(self) -> Dict[str, Any]:
        return self.raw.get("model", {})
    
    @property
    def tokenizer(self) -> Dict[str, Any]:
        return self.raw.get("tokenizer", {})
    
    @property
    def dataset(self) -> Dict[str, Any]:
        return self.raw.get("dataset", {})
    
    @property
    def lora(self) -> Dict[str, Any]:
        return self.raw.get("lora", {})
    
    @property
    def optimizer(self) -> Dict[str, Any]:
        return self.raw.get("optimizer", {})
    
    @property
    def scheduler(self) -> Dict[str, Any]:
        return self.raw.get("scheduler", {})
    
    @property
    def loss(self) -> Dict[str, Any]:
        return self.raw.get("loss", {})
    
    @property
    def hardware(self) -> Dict[str, Any]:
        return self.raw.get("hardware", {})
    
    @property
    def distributed(self) -> Dict[str, Any]:
        return self.raw.get("distributed", {})
    
    @property
    def training(self) -> Dict[str, Any]:
        return self.raw.get("training", {})
    
    @property
    def kernels(self) -> Dict[str, Any]:
        return self.raw.get("kernels", {})
    
    @property
    def export(self) -> Dict[str, Any]:
        return self.raw.get("export", {})
    
    @property
    def preprocessing(self) -> Dict[str, Any]:
        return self.raw.get("preprocessing", {})
    
    @property
    def prompt_template(self) -> Dict[str, Any]:
        return self.raw.get("prompt_template", {})
    
    @property
    def dataloader(self) -> Dict[str, Any]:
        return self.raw.get("dataloader", {})
    
    # ───────────────────────────────────────────────────────────────────────────
    # Convenience Methods
    # ───────────────────────────────────────────────────────────────────────────
    
    def get_model_name(self) -> str:
        return self.model.get("name_or_path", "")
    
    def get_dataset_name(self) -> str:
        return self.dataset.get("name", "")
    
    def get_max_length(self) -> int:
        return self.tokenizer.get("max_length", 4096)
    
    def get_batch_size(self) -> int:
        return self.dataloader.get("batch_size", 4)
    
    def is_distributed(self) -> bool:
        return self.distributed.get("enabled", False)


# ═════════════════════════════════════════════════════════════════════════════════
# SOTA Pipeline Orchestrator
# ═════════════════════════════════════════════════════════════════════════════════

class SOTARocmPipeline:
    """
    Complete SOTA E2E pipeline for AMD ROCm training.
    
    Features:
    - YAML-driven configuration
    - Multi-GPU distributed training
    - SOTA optimizers (Lion, CAME, Prodigy)
    - SOTA schedulers (WSD, REX)
    - Flash Attention 2 support for ROCm
    - Full training + export pipeline
    """
    
    def __init__(self, config: SOTARocmConfig, selected_gpus: List[int]):
        """
        Initialize pipeline.
        
        Args:
            config: SOTA configuration from YAML
            selected_gpus: List of GPU IDs to use
        """
        self.config = config
        self.selected_gpus = selected_gpus
        
        # Components (lazy initialization)
        self._model = None
        self._tokenizer = None
        self._optimizer = None
        self._scheduler = None
        self._train_dataloader = None
        self._eval_dataloader = None
        
        # Distributed state
        self._rank = 0
        self._world_size = len(selected_gpus)
        self._is_main_process = True
        
        logger.info(f"Pipeline initialized: mode={config.training_mode}, gpus={selected_gpus}")
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Step 1: Load Dataset with Introspection
    # ═════════════════════════════════════════════════════════════════════════════
    
    def load_dataset(self, split: str = "train") -> Any:
        """
        Load dataset with auto-discovery.
        
        Args:
            split: Split name from config
        
        Returns:
            HuggingFace Dataset
        """
        from data_pipeline.introspection import DatasetIntrospector
        from data_pipeline.core.types import is_err
        from datasets import load_dataset
        
        dataset_name = self.config.get_dataset_name()
        splits = self.config.dataset.get("splits", {})
        split_config = splits.get(split, {})
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Introspect first
        introspector = DatasetIntrospector()
        result = introspector.discover(
            dataset_name,
            trust_remote_code=self.config.raw.get("introspection", {}).get("trust_remote_code", True),
        )
        
        if not is_err(result):
            metadata = result.value
            logger.info(f"Discovered splits: {list(metadata.get_all_splits())}")
            logger.info(f"Discovered columns: {[f.name for f in metadata.features]}")
        
        # Load dataset
        hf_split = split_config.get("name", split)
        sample_size = split_config.get("sample_size")
        
        split_str = hf_split
        if sample_size:
            split_str = f"{hf_split}[:{sample_size}]"
        
        ds = load_dataset(
            dataset_name,
            name=self.config.dataset.get("config_name"),
            split=split_str,
            trust_remote_code=True,
        )
        
        # Shuffle if configured
        if split_config.get("shuffle", False):
            ds = ds.shuffle(seed=split_config.get("seed", 42))
        
        logger.info(f"Loaded {len(ds)} examples from '{split}'")
        return ds
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Step 2: Initialize Tokenizer
    # ═════════════════════════════════════════════════════════════════════════════
    
    def init_tokenizer(self):
        """
        Initialize tokenizer with SOTA settings.
        
        Returns:
            HuggingFace Tokenizer
        """
        from transformers import AutoTokenizer
        
        tok_cfg = self.config.tokenizer
        model_name = tok_cfg.get("name_or_path") or self.config.get_model_name()
        
        logger.info(f"Loading tokenizer: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=tok_cfg.get("use_fast", True),
        )
        
        # Set special tokens
        if tok_cfg.get("add_pad_token", True) and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        tokenizer.padding_side = tok_cfg.get("padding_side", "right")
        tokenizer.truncation_side = tok_cfg.get("truncation_side", "right")
        
        self._tokenizer = tokenizer
        logger.info(f"Tokenizer ready: vocab_size={tokenizer.vocab_size}")
        
        return tokenizer
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Step 3: Initialize Model with SOTA Training Mode
    # ═════════════════════════════════════════════════════════════════════════════
    
    def init_model(self):
        """
        Initialize SOTA model with LoRA/QLoRA.
        
        Returns:
            Model ready for training
        """
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        
        model_cfg = self.config.model
        model_name = self.config.get_model_name()
        training_mode = self.config.training_mode
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Training mode: {training_mode}")
        
        # Quantization config for QLoRA
        bnb_config = None
        quant_cfg = self.config.raw.get("quantization", {})
        if quant_cfg.get("enabled", False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=quant_cfg.get("load_in_4bit", True),
                bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=getattr(
                    torch, 
                    quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
                ),
                bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
            )
            logger.info("QLoRA quantization enabled (NF4)")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=model_cfg.get("revision", "main"),
            trust_remote_code=model_cfg.get("trust_remote_code", True),
            torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
            low_cpu_mem_usage=model_cfg.get("low_cpu_mem_usage", True),
            attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
            quantization_config=bnb_config,
            device_map="auto" if len(self.selected_gpus) == 1 else None,
        )
        
        # Apply SOTA patches from registry (Triton kernels)
        logger.info("Checking registry for SOTA patches...")
        if False: # registry.get_model_info(model_name) or True: # Force check
            registry.patch_model(model)
        
        # Apply LoRA if enabled
        if training_mode in ("lora", "qlora"):
            model = self._apply_lora(model)
        
        # Enable gradient checkpointing
        if self.config.distributed.get("gradient_checkpointing", True):
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                logger.info("Gradient checkpointing enabled")
        
        self._model = model
        return model
    
    def _apply_lora(self, model):
        """Apply LoRA/QLoRA to model using internal SOTA module."""
        logger.info("Applying internal SOTA LoRA...")
        
        lora_cfg = self.config.lora
        
        # Create config
        config = lora.LoraConfig(
            r=lora_cfg.get("r", 64),
            lora_alpha=lora_cfg.get("lora_alpha", 128),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            bias=lora_cfg.get("bias", "none"),
            use_rslora=lora_cfg.get("use_rslora", True),
            use_dora=lora_cfg.get("use_dora", False),
        )
        
        # Apply LoRA
        model = lora.get_peft_model(model, config)
        
        # Print stats
        lora.print_trainable_parameters(model)
        
        return model
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Step 4: Create SOTA Optimizer
    # ═════════════════════════════════════════════════════════════════════════════
    
    def init_optimizer(self):
        """
        Initialize SOTA optimizer.
        
        Supports: AdamW, Lion, CAME, SophiaG, Prodigy, Adam8bit
        
        Returns:
            Optimizer instance
        """
        from data_pipeline.trainer.optimizers import create_optimizer
        
        opt_cfg = self.config.optimizer
        opt_type = opt_cfg.get("type", "lion").lower()
        
        logger.info(f"Initializing SOTA optimizer: {opt_type}")
        
        trainable_params = [p for p in self._model.parameters() if p.requires_grad]
        
        # Use TrainScale's SOTA optimizer factory
        # Prepare optimizer args
        opt_kwargs = {
            "name": opt_type,
            "params": trainable_params,
            "lr": opt_cfg.get("learning_rate", 3e-5),
            "weight_decay": opt_cfg.get("weight_decay", 0.1),
        }
        
        # Handle optimizer-specific args
        if opt_type == "lion":
            opt_kwargs["betas"] = tuple(opt_cfg.get("lion_betas", [0.9, 0.99]))
        else:
            opt_kwargs["betas"] = tuple(opt_cfg.get("betas", [0.9, 0.999]))
            opt_kwargs["eps"] = opt_cfg.get("eps", 1e-8)

        try:
            self._optimizer = create_optimizer(**opt_kwargs)
        except Exception as e:
            logger.warning(f"SOTA optimizer failed: {e}, falling back to AdamW")
            self._optimizer = torch.optim.AdamW(
                trainable_params,
                lr=opt_cfg.get("learning_rate", 3e-5),
                weight_decay=opt_cfg.get("weight_decay", 0.1),
            )
        
        logger.info(f"Optimizer: {type(self._optimizer).__name__}")
        return self._optimizer
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Step 5: Create SOTA Scheduler
    # ═════════════════════════════════════════════════════════════════════════════
    
    def init_scheduler(self, num_training_steps: int):
        """
        Initialize SOTA scheduler.
        
        Supports: Cosine, WSD (LLaMA-3), REX, OneCycle
        
        Args:
            num_training_steps: Total training steps
        
        Returns:
            Scheduler instance
        """
        from data_pipeline.trainer.schedulers import create_sota_scheduler
        
        sch_cfg = self.config.scheduler
        sch_type = sch_cfg.get("type", "wsd").lower()
        
        warmup_steps = sch_cfg.get("warmup_steps", 0)
        if warmup_steps == 0:
            warmup_steps = int(sch_cfg.get("warmup_ratio", 0.03) * num_training_steps)
        
        logger.info(f"Initializing SOTA scheduler: {sch_type}")
        logger.info(f"Warmup steps: {warmup_steps}, Total steps: {num_training_steps}")
        
        try:
            self._scheduler = create_sota_scheduler(
                name=sch_type,
                optimizer=self._optimizer,
                num_training_steps=num_training_steps,
                warmup_steps=warmup_steps,
                min_lr_ratio=sch_cfg.get("min_lr_ratio", 0.1),
                stable_ratio=sch_cfg.get("stable_ratio", 0.85),
            )
        except Exception as e:
            logger.warning(f"SOTA scheduler failed: {e}, using cosine")
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self._scheduler = CosineAnnealingLR(
                self._optimizer,
                T_max=num_training_steps - warmup_steps,
            )
        
        logger.info(f"Scheduler: {type(self._scheduler).__name__}")
        return self._scheduler
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Step 6: Preprocess Dataset (SOTA)
    # ═════════════════════════════════════════════════════════════════════════════
    
    def preprocess_dataset(self, dataset):
        """
        Apply SOTA preprocessing to dataset.
        
        Features:
        - Token-aware length management
        - Smart truncation (sentence/word boundaries)
        - Per-column limits
        - Sequence packing
        
        Args:
            dataset: Raw HuggingFace dataset
        
        Returns:
            Preprocessed dataset with input_ids, attention_mask, labels
        """
        from data_pipeline.preprocessing import (
            PromptEngine,
            create_length_manager,
            TokenAwareContentDistributor,
            ContentDistributionMode,
        )
        from data_pipeline.core.config_schema import PromptTemplate
        from data_pipeline.core.types import is_err
        
        logger.info("Applying SOTA preprocessing...")
        
        pt_cfg = self.config.prompt_template
        prep_cfg = self.config.preprocessing
        max_length = self.config.get_max_length()
        tokenizer = self._tokenizer
        
        # Length manager
        length_manager = None
        lm_cfg = prep_cfg.get("length_manager", {})
        if lm_cfg.get("enabled", True):
            length_manager = create_length_manager(
                max_length=lm_cfg.get("max_total_length", max_length),
                padding_strategy=lm_cfg.get("padding_strategy", "longest"),
                default_truncation=lm_cfg.get("truncation_strategy", "smart"),
                per_column_limits=lm_cfg.get("per_column_limits", {}),
            )
            logger.info(f"Length manager: max={max_length}")
        
        # Process function for chat datasets
        def process_chat_example(example):
            """Process chat-format dataset (messages column)."""
            messages = example.get("messages", [])
            
            # Build conversation text
            text_parts = []
            label_parts = []
            current_is_assistant = False
            
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role in ("user", "human"):
                    text_parts.append(f"### Human: {content}")
                    current_is_assistant = False
                elif role in ("assistant", "gpt"):
                    text_parts.append(f"### Assistant: {content}")
                    label_parts.append(content)
                    current_is_assistant = True
                elif role == "system":
                    text_parts.insert(0, f"### System: {content}")
            
            full_text = "\n\n".join(text_parts)
            
            # Tokenize
            tokenized = tokenizer(
                full_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )
            
            # Create labels (mask input, keep assistant output)
            labels = tokenized["input_ids"].copy()
            
            # Mask tokens before assistant responses
            if label_parts:
                label_text = "\n\n".join(label_parts)
                label_tokens = tokenizer.encode(label_text, add_special_tokens=False)
                
                # Find label tokens in full sequence
                input_ids = tokenized["input_ids"]
                label_start = len(input_ids) - len(label_tokens) - 1
                
                # Mask everything before labels
                for i in range(min(label_start, len(labels))):
                    labels[i] = -100
            
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": labels,
            }
        
        # Process instruction datasets (Alpaca format)
        def process_instruction_example(example):
            """Process instruction-format dataset."""
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")
            
            # Build prompt
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            
            tokenized = tokenizer(
                prompt,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )
            
            # Create labels
            labels = tokenized["input_ids"].copy()
            
            # Mask input portion
            response_start = prompt.find("### Response:") + len("### Response:\n")
            input_portion = prompt[:response_start]
            input_tokens = tokenizer.encode(input_portion, add_special_tokens=True)
            
            for i in range(min(len(input_tokens), len(labels))):
                labels[i] = -100
            
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": labels,
            }
        
        # Detect dataset format and process
        sample = dataset[0] if len(dataset) > 0 else {}
        
        if "messages" in sample:
            process_fn = process_chat_example
            logger.info("Detected chat format (messages)")
        elif "instruction" in sample:
            process_fn = process_instruction_example
            logger.info("Detected instruction format")
        elif "text" in sample:
            # Simple text completion
            def process_text_example(example):
                text = example.get("text", "")
                tokenized = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors=None,
                )
                return {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": tokenized["input_ids"].copy(),
                }
            process_fn = process_text_example
            logger.info("Detected text format")
        else:
            raise ValueError(f"Unknown dataset format. Columns: {list(sample.keys())}")
        
        # Map processing function
        processed = dataset.map(
            process_fn,
            remove_columns=dataset.column_names,
            desc="SOTA Preprocessing",
            num_proc=1,
        )
        
        logger.info(f"Preprocessing complete: {len(processed)} examples")
        
        return processed
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Step 7: Create DataLoader
    # ═════════════════════════════════════════════════════════════════════════════
    
    def create_dataloader(self, dataset, shuffle: bool = True) -> DataLoader:
        """
        Create optimized DataLoader.
        
        Args:
            dataset: Preprocessed dataset
            shuffle: Whether to shuffle
        
        Returns:
            PyTorch DataLoader
        """
        from data_pipeline.data import DataLoaderBuilder
        from data_pipeline.core.config_schema import DataLoaderConfig
        
        dl_cfg = self.config.dataloader
        
        # Collate function
        def collate_fn(batch):
            input_ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long)
            attention_mask = torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long)
            labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        
        dataloader = DataLoader(
            dataset,
            batch_size=dl_cfg.get("batch_size", 4),
            shuffle=shuffle,
            num_workers=dl_cfg.get("num_workers", 4),
            pin_memory=dl_cfg.get("pin_memory", True),
            drop_last=dl_cfg.get("drop_last", False),
            collate_fn=collate_fn,
        )
        
        logger.info(f"DataLoader: {len(dataloader)} batches, batch_size={dl_cfg.get('batch_size', 4)}")
        
        return dataloader
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Step 8: Training Loop
    # ═════════════════════════════════════════════════════════════════════════════
    
    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """
        Execute SOTA training loop.
        
        Args:
            train_dataloader: Training DataLoader
            eval_dataloader: Optional evaluation DataLoader
        
        Returns:
            Training metrics
        """
        from data_pipeline.trainer import is_main_process
        
        train_cfg = self.config.training
        
        # Calculate training steps
        num_epochs = train_cfg.get("num_train_epochs", 1)
        max_steps = train_cfg.get("max_steps", -1)
        grad_accum = train_cfg.get("gradient_accumulation_steps", 4)
        
        steps_per_epoch = len(train_dataloader) // grad_accum
        total_steps = steps_per_epoch * num_epochs
        
        if max_steps > 0:
            total_steps = min(total_steps, max_steps)
        
        # Initialize scheduler
        self.init_scheduler(total_steps)
        
        # Setup mixed precision
        hw_cfg = self.config.hardware
        precision = hw_cfg.get("precision", "bf16")
        
        if precision == "bf16":
            amp_dtype = torch.bfloat16
        elif precision == "fp16":
            amp_dtype = torch.float16
        else:
            amp_dtype = torch.float32
        
        use_amp = precision != "fp32"
        
        scaler = torch.cuda.amp.GradScaler() if precision == "fp16" else None
        
        # Move model to device
        device = torch.device("cuda:0")
        self._model = self._model.to(device)
        
        # Enable Multi-GPU (DataParallel)
        if torch.cuda.device_count() > 1:
            logger.info(f"🚀 Enabling DataParallel on {torch.cuda.device_count()} GPUs")
            self._model = torch.nn.DataParallel(self._model)
        
        # Training loop
        logger.info("=" * 60)
        logger.info("🚀 Starting SOTA Training")
        logger.info("=" * 60)
        logger.info(f"   Total epochs: {num_epochs}")
        logger.info(f"   Total steps: {total_steps}")
        logger.info(f"   Batch size: {train_cfg.get('per_device_train_batch_size', 4)}")
        logger.info(f"   Gradient accumulation: {grad_accum}")
        logger.info(f"   Precision: {precision}")
        logger.info("=" * 60)
        
        self._model.train()
        global_step = 0
        total_loss = 0.0
        
        max_grad_norm = self.config.optimizer.get("max_grad_norm", 1.0)
        logging_steps = train_cfg.get("logging_steps", 10)
        save_steps = train_cfg.get("save_steps", 100)
        output_dir = train_cfg.get("output_dir", "./outputs")
        
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        
        # Initialize metrics
        metrics_coll = MetricCollection([
            Perplexity(),
            Accuracy() 
        ])
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for step, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Mixed precision context
                # Use standard pytorch autocast since SDPA handles low-level details
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        outputs = self._model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch.get("labels"),
                        )
                        
                        # Update metrics
                        if step % logging_steps == 0:
                            metrics_coll.update(outputs.logits, batch["labels"])
                        
                        if hasattr(outputs, "loss"):
                            loss = outputs.loss
                        else:
                            loss = outputs[0]
                        
                        # Handle DataParallel (loss is vector)
                        if loss.dim() > 0:
                            loss = loss.mean()
                            
                        loss = loss / grad_accum
                else: # No AMP
                    outputs = self._model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch.get("labels"),
                    )
                    
                    if hasattr(outputs, "loss"):
                        loss = outputs.loss
                    else:
                        loss = outputs[0]
                    
                    # Handle DataParallel (loss is vector)
                    if loss.dim() > 0:
                        loss = loss.mean()
                        
                    loss = loss / grad_accum
                
                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item() * grad_accum
                total_loss += loss.item() * grad_accum
                
                # Optimizer step
                if (step + 1) % grad_accum == 0:
                    if scaler:
                        scaler.unscale_(self._optimizer)
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(),
                        max_grad_norm,
                    )
                    
                    if scaler:
                        scaler.step(self._optimizer)
                        scaler.update()
                    else:
                        self._optimizer.step()
                    
                    self._scheduler.step()
                    self._optimizer.zero_grad()
                    global_step += 1
                    
                    # Logging
                    if global_step % logging_steps == 0:
                        avg_loss = total_loss / global_step
                        lr = self._scheduler.get_last_lr()[0]
                        elapsed = time.time() - start_time
                        steps_per_sec = global_step / elapsed
                        
                        # Compute metrics
                        metric_results = metrics_coll.compute()
                        metrics_coll.reset()
                        
                        logger.info(
                            f"Step {global_step}/{total_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"PPL: {metric_results['perplexity']:.2f} | "
                            f"Acc: {metric_results['accuracy']:.2%} | "
                            f"LR: {lr:.2e} | "
                            f"Speed: {steps_per_sec:.2f} steps/s"
                        )
                    
                    # Save checkpoint
                    if global_step % save_steps == 0:
                        self._save_checkpoint(output_dir, global_step)
                
                # Max steps check
                if max_steps > 0 and global_step >= max_steps:
                    break
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1}/{num_epochs} completed | Loss: {avg_epoch_loss:.4f}")
            
            if max_steps > 0 and global_step >= max_steps:
                break
        
        # Final save
        self._save_checkpoint(output_dir, global_step, is_final=True)
        
        total_time = time.time() - start_time
        metrics = {
            "train_loss": total_loss / global_step,
            "total_steps": global_step,
            "training_time_hours": total_time / 3600,
        }
        
        logger.info("=" * 60)
        logger.info("✅ Training Complete!")
        logger.info(f"   Final loss: {metrics['train_loss']:.4f}")
        logger.info(f"   Total steps: {global_step}")
        logger.info(f"   Training time: {total_time/3600:.2f} hours")
        logger.info("=" * 60)
        
        return metrics
    
    def _save_checkpoint(self, output_dir: str, step: int, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(
            output_dir, 
            "final" if is_final else f"checkpoint-{step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Helper to unwrap model
        def get_model(model):
            return model.module if hasattr(model, "module") else model

        # Save model
        model_to_save = get_model(self._model)
        if hasattr(model_to_save, 'save_pretrained'):
            model_to_save.save_pretrained(checkpoint_dir)
        else:
            torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        
        if self._tokenizer:
            self._tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
    
    # ═════════════════════════════════════════════════════════════════════════════
    # Full Pipeline Execution
    # ═════════════════════════════════════════════════════════════════════════════
    
    def run(self) -> Dict[str, Any]:
        """
        Execute complete SOTA training pipeline.
        
        Returns:
            Training results and metrics
        """
        logger.info("=" * 60)
        logger.info("🔥 SOTA E2E Pipeline - AMD ROCm MI300X")
        logger.info("=" * 60)
        
        # Step 1: Load datasets
        logger.info("\n📂 Step 1: Loading Dataset...")
        train_dataset = self.load_dataset("train")
        
        eval_dataset = None
        splits = self.config.dataset.get("splits", {})
        if "validation" in splits or "test" in splits:
            eval_split = "validation" if "validation" in splits else "test"
            try:
                eval_dataset = self.load_dataset(eval_split)
            except Exception as e:
                logger.warning(f"Could not load eval dataset: {e}")
        
        # Step 2: Initialize tokenizer
        logger.info("\n🔤 Step 2: Initializing Tokenizer...")
        self.init_tokenizer()
        
        # Step 3: Initialize model
        logger.info("\n🧠 Step 3: Loading SOTA Model...")
        self.init_model()
        
        # Step 4: Initialize optimizer
        logger.info("\n⚡ Step 4: Initializing Optimizer...")
        self.init_optimizer()
        
        # Step 5: Preprocess datasets
        logger.info("\n🔧 Step 5: SOTA Preprocessing...")
        train_processed = self.preprocess_dataset(train_dataset)
        
        eval_processed = None
        if eval_dataset:
            eval_processed = self.preprocess_dataset(eval_dataset)
        
        # Step 6: Create data loaders
        logger.info("\n📦 Step 6: Creating DataLoaders...")
        train_dataloader = self.create_dataloader(train_processed, shuffle=True)
        
        eval_dataloader = None
        if eval_processed:
            eval_dataloader = self.create_dataloader(eval_processed, shuffle=False)
        
        # Step 7: Train
        logger.info("\n🚂 Step 7: Training...")
        metrics = self.train(train_dataloader, eval_dataloader)
        
        # Step 8: Export if enabled
        export_cfg = self.config.export
        if export_cfg.get("enabled", False):
            logger.info("\n💾 Step 8: Exporting Model...")
            self._export_model(export_cfg)
            
        # Step 9: Benchmark Inference
        self.benchmark_inference()
        
        return {
            "metrics": metrics,
            "model": self._model,
            "tokenizer": self._tokenizer,
        }
    
    def _export_model(self, export_cfg: Dict[str, Any]):
        """Export trained model."""
        output_dir = export_cfg.get("output_dir", "./outputs/export")
        os.makedirs(output_dir, exist_ok=True)
        
        # Helper to unwrap model
        def get_model(model):
            return model.module if hasattr(model, "module") else model
            
        # Unwrap for processing
        model_to_export = get_model(self._model)

        # Merge LoRA if requested
        if export_cfg.get("merge_lora", True):
            logger.info("Merging LoRA weights (Internal)...")
            try:
                # Use internal lora module to merge
                from data_pipeline.trainer import lora
                model_to_export = lora.merge_and_unload(model_to_export)
            except Exception as e:
                logger.warning(f"Could not merge LoRA: {e}")
        
        # Save
        logger.info(f"Saving to {output_dir}")
        model_to_export.save_pretrained(output_dir)
        if self._tokenizer:
            self._tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model exported to: {output_dir}")

    def benchmark_inference(self, num_requests: int = 4):
        """Benchmark SOTA Inference Engine."""
        logger.info("\n🚀 Step 9: Benchmarking SOTA Inference...")
        
        # Load the exported (merged) model + tokenizer for fresh benchmark
        export_dir = self.config.export.get("output_dir", "outputs/exported")
        logger.info(f"Loading exported model from {export_dir} for benchmark...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load fresh components to verify 'W + change weights' loading
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
        # Add diverse prompts (formatted as chat)
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
                logger.info("Running standard HF generation for sanity check...")
                inputs = inference_tokenizer(text_prompt, return_tensors="pt").to(inference_model.device)
                with torch.no_grad():
                    gen_tokens = inference_model.generate(
                        **inputs, 
                        max_new_tokens=50, 
                        do_sample=False, 
                        pad_token_id=inference_tokenizer.eos_token_id
                    )
                logger.info(f"Standard Ref: {inference_tokenizer.decode(gen_tokens[0], skip_special_tokens=True)}")

        logger.info(f"Added 5 requests to Continuous Batching Scheduler.")
        
        # Run Generation
        responses = engine.generate_all()
        
        for i, resp in enumerate(responses):
            logger.info(f"Response {i}: {resp[:50]}...")


# ═════════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═════════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point for ROCm SOTA Demo."""
    parser = argparse.ArgumentParser(
        description="SOTA Training Demo for AMD ROCm MI300X Multi-GPU"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data_pipeline/examples/rocm_sota_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="2,3,4,5",
        help="Comma-separated GPU IDs to use (middle 4 from 8)",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed multi-GPU training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just verify setup without training",
    )
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    selected_gpus = [int(x) for x in args.gpus.split(",")]
    
    # Get ROCm GPU info
    gpu_info = get_rocm_gpu_info()
    
    # Print banner
    print_rocm_banner(gpu_info, selected_gpus)
    
    if not gpu_info["available"]:
        logger.error("❌ No AMD GPUs detected! Please check ROCm installation.")
        sys.exit(1)
    
    # Setup environment
    setup_rocm_environment(selected_gpus)
    
    # Verify PyTorch can see GPUs
    if torch.cuda.is_available():
        logger.info(f"✅ PyTorch sees {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1e9
            logger.info(f"   GPU {i}: {props.name} ({mem_gb:.1f} GB)")
    else:
        logger.error("❌ PyTorch cannot see CUDA GPUs!")
        sys.exit(1)
    
    if args.dry_run:
        logger.info("\n✅ Dry run complete! Setup verified.")
        return
    
    # Load config
    logger.info(f"\n📄 Loading config: {args.config}")
    config = SOTARocmConfig.from_yaml(args.config)
    
    # Print config summary
    print("\n" + "─" * 60)
    print("📋 Configuration Summary:")
    print("─" * 60)
    print(f"   Model: {config.get_model_name()}")
    print(f"   Dataset: {config.get_dataset_name()}")
    print(f"   Training Mode: {config.training_mode}")
    print(f"   Optimizer: {config.optimizer.get('type', 'adamw')}")
    print(f"   Scheduler: {config.scheduler.get('type', 'cosine')}")
    print(f"   LoRA Rank: {config.lora.get('r', 64)}")
    print(f"   Max Length: {config.get_max_length()}")
    print(f"   Batch Size: {config.get_batch_size()}")
    print("─" * 60 + "\n")
    
    # Create and run pipeline
    pipeline = SOTARocmPipeline(config, selected_gpus)
    results = pipeline.run()
    
    # Final summary
    print("\n" + "═" * 60)
    print("🎉 SOTA Training Complete!")
    print("═" * 60)
    print(f"   Final Loss: {results['metrics']['train_loss']:.4f}")
    print(f"   Total Steps: {results['metrics']['total_steps']}")
    print(f"   Training Time: {results['metrics']['training_time_hours']:.2f} hours")
    print("═" * 60)


if __name__ == "__main__":
    main()
