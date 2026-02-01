# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer - Tensor Parallel Implementation
# ════════════════════════════════════════════════════════════════════════════════
# Above-SOTA Tensor Parallelism for model sharding across GPUs.
# Implements Megatron-style column/row parallel linear layers.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import math
from typing import Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_tensor_parallel_rank() -> int:
    """Get tensor parallel rank."""
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def get_tensor_parallel_world_size() -> int:
    """Get tensor parallel world size."""
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Copy input to model parallel region (identity forward, all-reduce backward)."""
    
    @staticmethod
    def forward(ctx, input_: Tensor) -> Tensor:
        return input_
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(grad_output)
        return grad_output


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """Reduce from model parallel region (all-reduce forward, identity backward)."""
    
    @staticmethod
    def forward(ctx, input_: Tensor) -> Tensor:
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(input_)
        return input_
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Scatter input to model parallel region."""
    
    @staticmethod
    def forward(ctx, input_: Tensor, dim: int) -> Tensor:
        world_size = get_tensor_parallel_world_size()
        if world_size == 1:
            return input_
        rank = get_tensor_parallel_rank()
        chunks = input_.chunk(world_size, dim=dim)
        return chunks[rank].contiguous()
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        world_size = get_tensor_parallel_world_size()
        if world_size == 1:
            return grad_output, None
        gathered = [torch.zeros_like(grad_output) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, grad_output)
        return torch.cat(gathered, dim=-1), None


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather output from model parallel region."""
    
    @staticmethod
    def forward(ctx, input_: Tensor, dim: int) -> Tensor:
        ctx.dim = dim
        world_size = get_tensor_parallel_world_size()
        if world_size == 1:
            return input_
        gathered = [torch.zeros_like(input_) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, input_)
        return torch.cat(gathered, dim=dim)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        world_size = get_tensor_parallel_world_size()
        if world_size == 1:
            return grad_output, None
        rank = get_tensor_parallel_rank()
        chunks = grad_output.chunk(world_size, dim=ctx.dim)
        return chunks[rank].contiguous(), None


def copy_to_tensor_parallel_region(input_: Tensor) -> Tensor:
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_parallel_region(input_: Tensor) -> Tensor:
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_parallel_region(input_: Tensor, dim: int = -1) -> Tensor:
    return _ScatterToModelParallelRegion.apply(input_, dim)


def gather_from_tensor_parallel_region(input_: Tensor, dim: int = -1) -> Tensor:
    return _GatherFromModelParallelRegion.apply(input_, dim)


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism (Megatron-style).
    
    Splits weight matrix along output dimension across TP ranks.
    Y = XW where W is partitioned column-wise: W = [W1, W2, ..., Wn]
    
    Args:
        in_features: Input dimension
        out_features: Output dimension (will be divided by TP world size)
        bias: Include bias
        gather_output: Gather outputs across TP ranks
        init_method: Weight initialization function
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Optional[Callable] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        
        world_size = get_tensor_parallel_world_size()
        assert out_features % world_size == 0, f"out_features must be divisible by TP world size"
        
        self.out_features_per_partition = out_features // world_size
        
        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)
        
        self._init_weights(init_method)
    
    def _init_weights(self, init_method: Optional[Callable]) -> None:
        if init_method:
            init_method(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input_: Tensor) -> Tensor:
        # Copy input to all TP ranks
        input_parallel = copy_to_tensor_parallel_region(input_)
        
        # Local GEMM
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        
        if self.gather_output:
            output = gather_from_tensor_parallel_region(output_parallel, dim=-1)
            return output
        return output_parallel


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism (Megatron-style).
    
    Splits weight matrix along input dimension across TP ranks.
    Y = XW where W is partitioned row-wise.
    
    Args:
        in_features: Input dimension (will be divided by TP world size)
        out_features: Output dimension
        bias: Include bias
        input_is_parallel: Input is already partitioned
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        
        world_size = get_tensor_parallel_world_size()
        assert in_features % world_size == 0, f"in_features must be divisible by TP world size"
        
        self.in_features_per_partition = in_features // world_size
        
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input_: Tensor) -> Tensor:
        if not self.input_is_parallel:
            input_parallel = scatter_to_tensor_parallel_region(input_, dim=-1)
        else:
            input_parallel = input_
        
        # Local GEMM
        output_parallel = F.linear(input_parallel, self.weight)
        
        # All-reduce across TP ranks
        output = reduce_from_tensor_parallel_region(output_parallel)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer with vocabulary parallelism.
    
    Splits vocabulary across TP ranks for large vocab sizes.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        world_size = get_tensor_parallel_world_size()
        rank = get_tensor_parallel_rank()
        
        # Divide vocab
        assert num_embeddings % world_size == 0
        self.vocab_per_partition = num_embeddings // world_size
        self.vocab_start = rank * self.vocab_per_partition
        self.vocab_end = self.vocab_start + self.vocab_per_partition
        
        self.weight = nn.Parameter(torch.empty(self.vocab_per_partition, embedding_dim))
        nn.init.normal_(self.weight)
    
    def forward(self, input_: Tensor) -> Tensor:
        # Mask out-of-range indices
        input_mask = (input_ >= self.vocab_start) & (input_ < self.vocab_end)
        masked_input = (input_ - self.vocab_start) * input_mask.long()
        
        # Local embedding lookup
        output_parallel = F.embedding(masked_input, self.weight, self.padding_idx)
        output_parallel = output_parallel * input_mask.unsqueeze(-1).float()
        
        # All-reduce to combine
        output = reduce_from_tensor_parallel_region(output_parallel)
        return output


__all__ = [
    "get_tensor_parallel_rank",
    "get_tensor_parallel_world_size",
    "copy_to_tensor_parallel_region",
    "reduce_from_tensor_parallel_region", 
    "scatter_to_tensor_parallel_region",
    "gather_from_tensor_parallel_region",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
]
