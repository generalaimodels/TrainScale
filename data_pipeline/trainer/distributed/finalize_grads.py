# ════════════════════════════════════════════════════════════════════════════════
# GRADIENT FINALIZATION PIPELINE — CROSS-DIMENSION GRADIENT SYNCHRONISATION
# ════════════════════════════════════════════════════════════════════════════════
# After the per-bucket DP reduce-scatter / all-reduce completes, there are
# several categories of gradients that need *additional* synchronisation across
# other parallelism dimensions:
#
#   1. Shared word-embedding grads — all-reduce across first ↔ last PP stages
#      (the embedding weight is tied between input & output layers that live
#      on different pipeline stages).
#
#   2. Position-embedding grads — all-reduce across encoder & decoder stages
#      (when position embeddings are shared across pipeline stages).
#
#   3. LayerNorm / sequence-parallel grads — all-reduce across TP group
#      (sequence parallelism splits activations along the sequence dimension,
#      so LN grads must be summed across TP ranks).
#
#   4. Conditional-embedder grads — all-reduce across PP group
#      (for diffusion-style models where timestep/label embedders are
#      replicated on every PP stage).
#
#   5. Per-token loss normalisation — broadcast token count from last PP stage,
#      all-reduce across DP, then scale all grads by 1/num_tokens.
#
# The orchestration order matters for correctness:
#   finish_grad_sync → conditional_embedding → layernorm/TP →
#   word_embedding → position_embedding → per_token_scaling
#
# Complexity:
#   Each sub-step is O(G / group_size) communication where G = gradient size,
#   with coalesced flat-tensor packing for efficiency.
#
# Reference: Megatron-LM core/distributed/finalize_model_grads
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist

logger = logging.getLogger("sota_ddp.finalize_grads")


# ════════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ════════════════════════════════════════════════════════════════════════════════

def _get_main_grad_attr(param: torch.nn.Parameter) -> str:
    """Return the attribute name that holds the gradient.

    Megatron-style DDP assigns gradients to ``param.main_grad`` (a view into
    the contiguous gradient buffer) rather than the standard ``param.grad``.
    """
    return "main_grad" if hasattr(param, "main_grad") else "grad"


def _flatten_dense_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Flatten a list of dense tensors into a single contiguous 1-D tensor.

    This is the standard PyTorch utility used for coalesced communication.
    """
    return torch.cat([t.contiguous().view(-1) for t in tensors])


def _unflatten_dense_tensors(
    flat: torch.Tensor,
    reference: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Unflatten *flat* into views matching the shapes of *reference* tensors."""
    outputs: List[torch.Tensor] = []
    offset = 0
    for t in reference:
        numel = t.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(t))
        offset += numel
    return outputs


# ════════════════════════════════════════════════════════════════════════════════
# CONDITIONAL-EMBEDDING GRADIENT ALL-REDUCE (DiT / Diffusion models)
# ════════════════════════════════════════════════════════════════════════════════

def allreduce_conditional_embedding_grads(
    model: List[torch.nn.Module],
    pp_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    All-reduce gradients of conditional embedders across PP stages.

    Diffusion models (DiT) replicate timestep / FPS / label embedders on
    every PP stage.  Their gradients must be synchronised so parameters stay
    in lock-step.

    When virtual pipeline parallelism (VPP) is used, gradients from all local
    VPP chunks are first summed into the first chunk, then all-reduced across
    PP ranks, then broadcast back to other VPP chunks.

    Args:
        model: List of model chunks (one per VPP stage on this rank).
        pp_group: Pipeline-parallel process group.  If None, this step is
            skipped (single-stage pipeline).
    """
    if pp_group is None or pp_group.size() <= 1:
        return

    grads_dict: Dict[str, List[torch.Tensor]] = {}
    for model_chunk in model:
        named_params_fn = getattr(model_chunk, "named_parameters", None)
        if named_params_fn is None:
            continue
        for name, param in named_params_fn():
            if param.requires_grad and getattr(param, "pipeline_parallel", False):
                grad_attr = _get_main_grad_attr(param)
                grad = getattr(param, grad_attr, None)
                if grad is None:
                    continue
                if name in grads_dict:
                    # Sum into first VPP chunk's gradient
                    grads_dict[name][0].add_(grad)
                    grads_dict[name].append(grad)
                else:
                    grads_dict[name] = [grad]

    if not grads_dict:
        return

    # All-reduce the first-VPP-chunk gradients across PP ranks
    grads = [v[0] for v in grads_dict.values()]
    coalesced = _flatten_dense_tensors(grads)
    dist.all_reduce(coalesced, group=pp_group)
    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
        buf.copy_(synced)

    # Copy synchronised gradient back to other VPP chunks
    for grad_list in grads_dict.values():
        for grad in grad_list[1:]:
            grad.copy_(grad_list[0])


# ════════════════════════════════════════════════════════════════════════════════
# LAYERNORM / SEQUENCE-PARALLEL GRADIENT ALL-REDUCE
# ════════════════════════════════════════════════════════════════════════════════

def allreduce_layernorm_grads(
    model: List[torch.nn.Module],
    tp_group: Optional[dist.ProcessGroup] = None,
    sequence_parallel: bool = False,
    qk_layernorm: bool = False,
) -> None:
    """
    All-reduce gradients that are *not* reduced by the standard TP all-reduce.

    Two categories:
      1. **Sequence-parallel LayerNorm**: When sequence parallelism is active,
         LayerNorm parameters are replicated across TP ranks but their grads
         are computed on disjoint sequence chunks → must be summed.
      2. **QK LayerNorm**: ``q_layernorm`` and ``k_layernorm`` grads need
         explicit synchronisation across TP ranks.

    Uses coalesced flat-tensor packing for a single NCCL call.

    Args:
        model: List of model chunks.
        tp_group: Tensor-model-parallel process group.
        sequence_parallel: Whether sequence parallelism is active.
        qk_layernorm: Whether QK layernorm is active.
    """
    if tp_group is None or tp_group.size() <= 1:
        return

    grads: List[torch.Tensor] = []
    for model_chunk in model:
        named_params_fn = getattr(model_chunk, "named_parameters", None)
        if named_params_fn is None:
            continue
        for name, param in named_params_fn():
            if not param.requires_grad:
                continue

            is_seq_parallel = (
                sequence_parallel
                and getattr(param, "sequence_parallel", False)
            )
            is_qk_ln = qk_layernorm and (
                "q_layernorm" in name or "k_layernorm" in name
            )

            if is_seq_parallel or is_qk_ln:
                grad_attr = _get_main_grad_attr(param)
                grad = getattr(param, grad_attr, None)
                if grad is not None:
                    grads.append(grad.data)

    if not grads:
        return

    coalesced = _flatten_dense_tensors(grads)
    dist.all_reduce(coalesced, op=dist.ReduceOp.SUM, group=tp_group)
    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
        buf.copy_(synced)


# ════════════════════════════════════════════════════════════════════════════════
# WORD-EMBEDDING GRADIENT ALL-REDUCE
# ════════════════════════════════════════════════════════════════════════════════

def allreduce_word_embedding_grads(
    model: List[torch.nn.Module],
    embedding_group: Optional[dist.ProcessGroup] = None,
    pp_group: Optional[dist.ProcessGroup] = None,
    share_embeddings_and_output_weights: bool = True,
) -> None:
    """
    All-reduce shared word-embedding gradients across first ↔ last PP stages.

    When ``share_embeddings_and_output_weights`` is True, the input embedding
    and the output linear layer's weight are the same tensor.  Since these
    live on different pipeline stages, their gradients must be synchronised
    after the DP collective.

    Args:
        model: List of model chunks.
        embedding_group: Process group spanning the first and last PP stages.
        pp_group: Pipeline-parallel process group.
        share_embeddings_and_output_weights: Whether embeddings are tied.
    """
    if not share_embeddings_and_output_weights:
        return
    if embedding_group is None or embedding_group.size() <= 1:
        return

    # Identify the model chunk that holds the shared embedding
    # (first PP stage → model[0], last PP stage → model[-1])
    weight = _get_shared_embedding_weight(model)
    if weight is None:
        return

    grad_attr = _get_main_grad_attr(weight)
    grad = getattr(weight, grad_attr, None)
    if grad is None:
        return

    dist.all_reduce(grad, group=embedding_group)


def _get_shared_embedding_weight(
    model: List[torch.nn.Module],
) -> Optional[torch.nn.Parameter]:
    """Extract the shared embedding weight from the model.

    Looks for common attribute names used by Megatron-style and HuggingFace-style
    models for shared embedding weights.
    """
    for model_chunk in [model[0], model[-1]]:
        # Megatron-style
        getter = getattr(model_chunk, "shared_embedding_or_output_weight", None)
        if getter is not None:
            weight = getter()
            if weight is not None:
                return weight
        # Check direct attributes
        for attr in ("word_embeddings", "embed_tokens", "wte"):
            emb_module = getattr(model_chunk, attr, None)
            if emb_module is not None and hasattr(emb_module, "weight"):
                return emb_module.weight
    return None


# ════════════════════════════════════════════════════════════════════════════════
# POSITION-EMBEDDING GRADIENT ALL-REDUCE
# ════════════════════════════════════════════════════════════════════════════════

def allreduce_position_embedding_grads(
    model: List[torch.nn.Module],
    pos_embedding_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    All-reduce position-embedding gradients across encoder ↔ decoder stages.

    Only applicable when position embeddings are shared between encoder and
    decoder (e.g. T5-style models).

    Args:
        model: List of model chunks.
        pos_embedding_group: Process group for position embedding sync.
    """
    if pos_embedding_group is None or pos_embedding_group.size() <= 1:
        return

    for model_chunk in model:
        pos_emb = getattr(model_chunk, "position_embeddings", None)
        if pos_emb is None:
            continue
        weight = getattr(pos_emb, "weight", None)
        if weight is None:
            continue

        grad_attr = _get_main_grad_attr(weight)
        grad = getattr(weight, grad_attr, None)
        if grad is None:
            continue

        dist.all_reduce(grad, group=pos_embedding_group)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR — finalize_model_grads
# ════════════════════════════════════════════════════════════════════════════════

def finalize_model_grads(
    model: List[torch.nn.Module],
    num_tokens: Optional[torch.Tensor] = None,
    *,
    # Process groups (optional — callers may pass explicit groups)
    tp_group: Optional[dist.ProcessGroup] = None,
    pp_group: Optional[dist.ProcessGroup] = None,
    dp_group: Optional[dist.ProcessGroup] = None,
    embedding_group: Optional[dist.ProcessGroup] = None,
    pos_embedding_group: Optional[dist.ProcessGroup] = None,
    # Feature flags
    sequence_parallel: bool = False,
    qk_layernorm: bool = False,
    share_embeddings_and_output_weights: bool = True,
    force_all_reduce: bool = False,
) -> None:
    """
    Orchestrate the full gradient finalization pipeline.

    This is the single entry-point called at the end of each training step's
    backward pass, *before* the optimizer step.  It ensures that all gradient
    tensors are fully synchronised across every parallelism dimension.

    Execution order:
      1. **DP grad sync** — finish the per-bucket reduce-scatter / all-reduce
         that was overlapped with backward compute.
      2. **Conditional-embedding all-reduce** — sync DiT-style replicated
         embedder grads across PP stages.
      3. **LayerNorm / TP all-reduce** — sync sequence-parallel and QK-LN
         grads across TP ranks.
      4. **Word-embedding all-reduce** — sync tied embedding grads across
         first ↔ last PP stages.
      5. **Position-embedding all-reduce** — sync shared position-embedding
         grads across encoder ↔ decoder.
      6. **Per-token loss scaling** — broadcast num_tokens from last PP stage,
         all-reduce across DP, then scale all grads.

    Args:
        model: List of model chunks (pipeline / virtual-pipeline stages).
        num_tokens: Total non-padded token count for per-token loss normalisation.
            When provided, grads are scaled by ``1 / num_tokens`` globally.
        tp_group: Tensor-model-parallel group.
        pp_group: Pipeline-model-parallel group.
        dp_group: Data-parallel group (used for num_tokens all-reduce).
        embedding_group: Group spanning first + last PP stages for embeddings.
        pos_embedding_group: Group for position embedding sync.
        sequence_parallel: Whether sequence parallelism is active.
        qk_layernorm: Whether QK layernorm needs explicit sync.
        share_embeddings_and_output_weights: Whether embeddings are tied.
        force_all_reduce: Force all-reduce instead of reduce-scatter in DP sync.

    Complexity:
        Total communication = O(G/dp) + O(G_emb/pp) + O(G_ln/tp) + O(G_emb/emb)
        where G_* = gradient sizes for each category.
    """
    # ── Step 1: Finish DP grad sync ───────────────────────────────────────
    for model_chunk in model:
        finish_fn = getattr(model_chunk, "finish_grad_sync", None)
        if finish_fn is not None:
            finish_fn(force_all_reduce=force_all_reduce)

    # ── Step 2: Conditional-embedding all-reduce (DiT / diffusion) ────────
    allreduce_conditional_embedding_grads(model, pp_group=pp_group)

    # ── Step 3: LayerNorm / sequence-parallel / QK-LN all-reduce ──────────
    allreduce_layernorm_grads(
        model,
        tp_group=tp_group,
        sequence_parallel=sequence_parallel,
        qk_layernorm=qk_layernorm,
    )

    # ── Step 4: Word-embedding all-reduce ─────────────────────────────────
    allreduce_word_embedding_grads(
        model,
        embedding_group=embedding_group,
        pp_group=pp_group,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
    )

    # ── Step 5: Position-embedding all-reduce ─────────────────────────────
    allreduce_position_embedding_grads(
        model, pos_embedding_group=pos_embedding_group
    )

    # ── Step 6: Per-token loss normalisation ──────────────────────────────
    if num_tokens is not None and dp_group is not None:
        _scale_grads_by_token_count(model, num_tokens, pp_group, dp_group)


def _scale_grads_by_token_count(
    model: List[torch.nn.Module],
    num_tokens: torch.Tensor,
    pp_group: Optional[dist.ProcessGroup],
    dp_group: dist.ProcessGroup,
) -> None:
    """
    Normalise gradients by the global non-padded token count.

    The token count originates on the last PP stage (where the loss is
    computed).  It is broadcast to all PP ranks, then all-reduced across
    DP ranks to get the global count.

    Args:
        model: Model chunks.
        num_tokens: Token count tensor (scalar, on CUDA).
        pp_group: Pipeline-parallel group for broadcast.
        dp_group: Data-parallel group for all-reduce.
    """
    # Broadcast from last PP stage to all PP ranks
    if pp_group is not None and pp_group.size() > 1:
        last_rank = dist.get_process_group_ranks(pp_group)[-1]
        dist.broadcast(num_tokens, src=last_rank, group=pp_group)

    # All-reduce across DP ranks to get global token count
    dist.all_reduce(num_tokens, group=dp_group)

    # Scale all gradients
    if num_tokens.item() > 0:
        scaling = 1.0 / num_tokens.item()
        for model_chunk in model:
            scale_fn = getattr(model_chunk, "scale_gradients", None)
            if scale_fn is not None:
                scale_fn(scaling)
            else:
                # Fallback: scale each param's gradient directly
                for param in model_chunk.parameters():
                    if param.requires_grad:
                        grad_attr = _get_main_grad_attr(param)
                        grad = getattr(param, grad_attr, None)
                        if grad is not None:
                            grad.mul_(scaling)
