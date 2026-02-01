# ════════════════════════════════════════════════════════════════════════════════
# SOTA Model Export Module
# ════════════════════════════════════════════════════════════════════════════════
# Export trained models to various formats:
# - GGUF (llama.cpp)
# - vLLM
# - SGLang
# - HuggingFace Hub
# - Merged weights (LoRA → full)
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import gc
import json
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


# ═════════════════════════════════════════════════════════════════════════════════
# Export Formats
# ═════════════════════════════════════════════════════════════════════════════════

class ExportFormat(Enum):
    """Supported export formats."""
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    GGUF = "gguf"
    GGUF_Q4_K_M = "q4_k_m"
    GGUF_Q5_K_M = "q5_k_m"
    GGUF_Q8_0 = "q8_0"
    GGUF_F16 = "f16"
    VLLM = "vllm"
    SGLANG = "sglang"


@dataclass
class ExportConfig:
    """Configuration for model export."""
    
    output_dir: str = "./exported_model"
    format: ExportFormat = ExportFormat.SAFETENSORS
    
    # GGUF settings
    gguf_quantization: str = "q4_k_m"
    
    # Hub settings
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    hub_private: bool = False
    
    # Merge settings
    merge_lora: bool = True
    save_merged_16bit: bool = True
    
    # vLLM/SGLang
    tensor_parallel_size: int = 1


# ═════════════════════════════════════════════════════════════════════════════════
# LoRA Merging
# ═════════════════════════════════════════════════════════════════════════════════

def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA adapters into base model weights.
    
    W_merged = W + scaling * (B @ A)
    """
    for name, module in model.named_modules():
        # Check for LoRA parameters
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            lora_A = module.lora_A
            lora_B = module.lora_B
            scaling = getattr(module, 'scaling', 1.0)
            
            if hasattr(lora_A, 'weight'):
                lora_A = lora_A.weight
            if hasattr(lora_B, 'weight'):
                lora_B = lora_B.weight
            
            if lora_A is not None and lora_B is not None:
                # Compute delta: scaling * B @ A
                delta = scaling * (lora_B @ lora_A)
                
                # Add to base weight
                if hasattr(module, 'weight'):
                    module.weight.data.add_(delta)
                    
                    # Clean up LoRA params
                    delattr(module, 'lora_A')
                    delattr(module, 'lora_B')
                    if hasattr(module, 'scaling'):
                        delattr(module, 'scaling')
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print("✓ Merged LoRA weights into base model")
    return model


def unmerge_lora_weights(
    model: nn.Module,
    lora_state_dict: Dict[str, torch.Tensor],
) -> nn.Module:
    """Reverse LoRA merge (for debugging/comparison)."""
    for name, module in model.named_modules():
        lora_a_key = f"{name}.lora_A.weight"
        lora_b_key = f"{name}.lora_B.weight"
        
        if lora_a_key in lora_state_dict and lora_b_key in lora_state_dict:
            lora_A = lora_state_dict[lora_a_key]
            lora_B = lora_state_dict[lora_b_key]
            scaling = lora_state_dict.get(f"{name}.scaling", 1.0)
            
            delta = scaling * (lora_B @ lora_A)
            
            if hasattr(module, 'weight'):
                module.weight.data.sub_(delta)
    
    return model


# ═════════════════════════════════════════════════════════════════════════════════
# Safetensors Export
# ═════════════════════════════════════════════════════════════════════════════════

def save_safetensors(
    model: nn.Module,
    output_dir: str,
    max_shard_size: str = "5GB",
) -> List[str]:
    """
    Save model in safetensors format.
    
    Supports sharding for large models.
    """
    from safetensors.torch import save_file
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    state_dict = model.state_dict()
    
    # Convert to float16 for efficient storage
    for key in state_dict:
        if state_dict[key].dtype == torch.float32:
            state_dict[key] = state_dict[key].half()
    
    # Single file for smaller models
    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    
    if total_size < 5 * 1024**3:  # < 5GB
        save_path = output_path / "model.safetensors"
        save_file(state_dict, save_path)
        return [str(save_path)]
    
    # Shard for larger models
    files = []
    current_shard = {}
    current_size = 0
    shard_idx = 0
    max_size = 5 * 1024**3
    
    for key, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        
        if current_size + tensor_size > max_size and current_shard:
            # Save current shard
            shard_path = output_path / f"model-{shard_idx:05d}.safetensors"
            save_file(current_shard, shard_path)
            files.append(str(shard_path))
            
            current_shard = {}
            current_size = 0
            shard_idx += 1
        
        current_shard[key] = tensor
        current_size += tensor_size
    
    # Save last shard
    if current_shard:
        shard_path = output_path / f"model-{shard_idx:05d}.safetensors"
        save_file(current_shard, shard_path)
        files.append(str(shard_path))
    
    print(f"✓ Saved {len(files)} safetensors shards to {output_dir}")
    return files


# ═════════════════════════════════════════════════════════════════════════════════
# GGUF Export
# ═════════════════════════════════════════════════════════════════════════════════

def export_to_gguf(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    quantization: str = "q4_k_m",
) -> str:
    """
    Export model to GGUF format for llama.cpp.
    
    Requires llama.cpp convert script.
    """
    import subprocess
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # First save as HF format
    hf_dir = output_path / "hf_temp"
    model.save_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)
    
    # Output GGUF path
    gguf_path = output_path / f"model-{quantization}.gguf"
    
    # Try llama.cpp convert
    try:
        convert_script = "convert_hf_to_gguf.py"
        
        cmd = [
            "python", convert_script,
            str(hf_dir),
            "--outfile", str(gguf_path),
            "--outtype", quantization,
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Exported to GGUF: {gguf_path}")
            shutil.rmtree(hf_dir)
            return str(gguf_path)
        else:
            print(f"⚠ GGUF conversion failed: {result.stderr}")
            
    except FileNotFoundError:
        print("⚠ llama.cpp convert script not found. Saving HF format instead.")
    
    return str(hf_dir)


# ═════════════════════════════════════════════════════════════════════════════════
# vLLM Export
# ═════════════════════════════════════════════════════════════════════════════════

def prepare_for_vllm(
    model_path: str,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
) -> Dict[str, Any]:
    """
    Prepare model configuration for vLLM deployment.
    
    Returns config dict for vLLM.LLM initialization.
    """
    config = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.9,
    }
    
    return config


def create_vllm_config(
    model_path: str,
    output_dir: str,
    **kwargs,
) -> str:
    """Create vLLM configuration file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = prepare_for_vllm(model_path, **kwargs)
    
    config_path = output_path / "vllm_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ vLLM config saved to {config_path}")
    return str(config_path)


# ═════════════════════════════════════════════════════════════════════════════════
# HuggingFace Hub
# ═════════════════════════════════════════════════════════════════════════════════

def push_to_hub(
    model: nn.Module,
    tokenizer,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload model",
) -> str:
    """
    Push model to HuggingFace Hub.
    
    Returns the hub URL.
    """
    from huggingface_hub import HfApi
    
    api = HfApi(token=token)
    
    # Create repo if needed
    try:
        api.create_repo(repo_id, private=private, exist_ok=True)
    except Exception as e:
        print(f"⚠ Repo creation warning: {e}")
    
    # Save and upload
    model.push_to_hub(repo_id, token=token, commit_message=commit_message)
    tokenizer.push_to_hub(repo_id, token=token)
    
    url = f"https://huggingface.co/{repo_id}"
    print(f"✓ Pushed to Hub: {url}")
    return url


# ═════════════════════════════════════════════════════════════════════════════════
# Unified Export Function
# ═════════════════════════════════════════════════════════════════════════════════

def export_model(
    model: nn.Module,
    tokenizer,
    config: ExportConfig,
) -> Dict[str, Any]:
    """
    Export model to specified format.
    
    Returns dict with export paths and info.
    """
    result = {
        "format": config.format.value,
        "output_dir": config.output_dir,
        "files": [],
    }
    
    # Merge LoRA if requested
    if config.merge_lora:
        model = merge_lora_weights(model)
    
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export based on format
    if config.format == ExportFormat.SAFETENSORS:
        result["files"] = save_safetensors(model, config.output_dir)
        
    elif config.format in [ExportFormat.GGUF, ExportFormat.GGUF_Q4_K_M, 
                           ExportFormat.GGUF_Q5_K_M, ExportFormat.GGUF_Q8_0,
                           ExportFormat.GGUF_F16]:
        quant = config.gguf_quantization
        if config.format == ExportFormat.GGUF_Q4_K_M:
            quant = "q4_k_m"
        elif config.format == ExportFormat.GGUF_Q5_K_M:
            quant = "q5_k_m"
        elif config.format == ExportFormat.GGUF_Q8_0:
            quant = "q8_0"
        elif config.format == ExportFormat.GGUF_F16:
            quant = "f16"
        
        gguf_path = export_to_gguf(model, tokenizer, config.output_dir, quant)
        result["files"] = [gguf_path]
        
    elif config.format == ExportFormat.VLLM:
        # Save as safetensors then create vLLM config
        result["files"] = save_safetensors(model, config.output_dir)
        vllm_config = create_vllm_config(
            config.output_dir, config.output_dir,
            tensor_parallel_size=config.tensor_parallel_size,
        )
        result["vllm_config"] = vllm_config
        
    elif config.format == ExportFormat.PYTORCH:
        # Save as PyTorch state dict
        torch_path = output_path / "pytorch_model.bin"
        torch.save(model.state_dict(), torch_path)
        result["files"] = [str(torch_path)]
    
    # Save tokenizer
    tokenizer.save_pretrained(config.output_dir)
    
    # Push to hub if requested
    if config.push_to_hub and config.hub_model_id:
        result["hub_url"] = push_to_hub(
            model, tokenizer,
            config.hub_model_id,
            token=config.hub_token,
            private=config.hub_private,
        )
    
    print(f"✓ Export complete: {config.format.value}")
    return result


# ═════════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "ExportFormat",
    # Config
    "ExportConfig",
    # LoRA merge
    "merge_lora_weights",
    "unmerge_lora_weights",
    # Format exports
    "save_safetensors",
    "export_to_gguf",
    "prepare_for_vllm",
    "create_vllm_config",
    "push_to_hub",
    # Unified
    "export_model",
]
