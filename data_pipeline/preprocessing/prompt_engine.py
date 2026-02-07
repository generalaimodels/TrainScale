# ════════════════════════════════════════════════════════════════════════════════
# Prompt Engine — Universal Training-Stage Data Formatter
# ════════════════════════════════════════════════════════════════════════════════
# Constructs tokenised sequences for every stage of LLM development:
#   • Pre-training   : Causal-LM, Fill-in-the-Middle (FIM), Document Packing
#   • Fine-tuning    : Chat (multi-turn per-turn masking), Completion,
#                      Instruction (Alpaca / Vicuna / ChatML), Custom Jinja2
#   • Post-training  : DPO, PPO, KTO, ORPO, GRPO, Reward-Model
#
# Design invariants
# ─────────────────
# 1.  Result[T, E] — every fallible call returns Ok | Err; zero exceptions
#     used for control flow.
# 2.  Thread-safe — all processors are stateless after __init__.
# 3.  O(sequence_length) per example for single-sequence formats.
# 4.  Cache-locality — frozen slotted dataclasses minimise pointer chasing.
# 5.  Backward-compatible — legacy PromptTemplate + TokenizerWrapper API
#     still works; new PromptEngineConfig unlocks all stages.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import bisect
import random
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
)

from data_pipeline.core.types import Result, Ok, Err
from data_pipeline.core.errors import TokenizationError, TemplateRenderError
from data_pipeline.core.types import Result, Ok, Err
from data_pipeline.core.errors import TokenizationError, TemplateRenderError
from data_pipeline.core.config_schema import (
    PromptTemplate,
    PromptEngineConfig,
    PreTrainingConfig,
    FineTuningConfig,
    RLConfig,
    TrainingStage,
    PreTrainingFormat,
    FineTuningFormat,
    RLAlgorithm,
    FIMMode,
    PackingStrategy,
    TruncationStrategy,
    FIMConfig,
    PackingConfig,
    SpanCorruptionConfig,
    MultiTurnConfig,
    RLColumnMapping,
)
from data_pipeline.preprocessing.tokenization import TokenizerWrapper
from data_pipeline.preprocessing.length_manager import TokenAwareContentDistributor

if TYPE_CHECKING:
    from jinja2 import Template as JinjaTemplate


# ═════════════════════════════════════════════════════════════════════════════════
# §1  Constants
# ═════════════════════════════════════════════════════════════════════════════════

# CrossEntropyLoss ignore_index — masks tokens that must not contribute to loss.
LABEL_PAD_TOKEN_ID: int = -100

# Simple {variable} pattern for Jinja2-free fallback rendering.
SIMPLE_VAR_PATTERN: re.Pattern[str] = re.compile(r"\{(\w+)\}")

# Sentence-boundary heuristic (handles ". ", "! ", "? ", paragraph breaks).
_SENTENCE_END: re.Pattern[str] = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$|(?<=\n)\s*(?=\n)"
)

# Default FIM special tokens (StarCoder / CodeLlama convention).
_DEFAULT_FIM_PREFIX: str = "<|fim_prefix|>"
_DEFAULT_FIM_SUFFIX: str = "<|fim_suffix|>"
_DEFAULT_FIM_MIDDLE: str = "<|fim_middle|>"

# Default role tags for fallback chat formatting when
# the underlying tokeniser has no apply_chat_template.
_FALLBACK_ROLE_TAG: str = "<|{role}|>"

# Minimum ratio of content to keep during smart sentence truncation.
_SENTENCE_MIN_RATIO: float = 0.50
_WORD_MIN_RATIO: float = 0.75


# ═════════════════════════════════════════════════════════════════════════════════
# §2  Enums — Training Stage, Format, Algorithm
# ═════════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════════
# §3  Configuration Dataclasses
# ═════════════════════════════════════════════════════════════════════════════════

# Configurations are now imported from data_pipeline.core.config_schema
# This section is intentionally left empty to maintain file structure compatibility
# while all logic uses the centralized schema.

# ═════════════════════════════════════════════════════════════════════════════════
# §4  Output Dataclasses
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ProcessedExample:
    """
    Standard training example (pre-training & SFT).

    Fields
    ------
    input_ids      : token IDs for the full sequence.
    attention_mask : 1 for real tokens, 0 for padding.
    labels         : loss targets (``-100`` for masked positions).
    input_length   : number of prompt / input tokens (masked region size).
    """
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    input_length: int

    # ── dict round-trip for datasets.Dataset.map() compatibility ──
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
            "input_length": self.input_length,
        }


@dataclass(frozen=True, slots=True)
class PackedExample:
    """
    Packed pre-training example (multiple documents concatenated).

    Additional field ``document_ids`` records which document each token
    belongs to, enabling per-document causal-mask isolation.
    ``position_ids`` resets at every document boundary so that RoPE
    positions are document-local.
    """
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    document_ids: List[int]
    position_ids: List[int]
    num_documents: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
            "document_ids": self.document_ids,
            "position_ids": self.position_ids,
            "num_documents": self.num_documents,
        }


@dataclass(frozen=True, slots=True)
class PreferencePairExample:
    """
    Preference-pair example for DPO / ORPO / Reward-Model.

    Prompt tokens are included in both chosen and rejected sequences
    so that the model sees identical context; their labels are masked.
    """
    prompt_input_ids: List[int]
    prompt_attention_mask: List[int]
    chosen_input_ids: List[int]
    chosen_attention_mask: List[int]
    chosen_labels: List[int]
    rejected_input_ids: List[int]
    rejected_attention_mask: List[int]
    rejected_labels: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_input_ids": self.prompt_input_ids,
            "prompt_attention_mask": self.prompt_attention_mask,
            "chosen_input_ids": self.chosen_input_ids,
            "chosen_attention_mask": self.chosen_attention_mask,
            "chosen_labels": self.chosen_labels,
            "rejected_input_ids": self.rejected_input_ids,
            "rejected_attention_mask": self.rejected_attention_mask,
            "rejected_labels": self.rejected_labels,
        }


@dataclass(frozen=True, slots=True)
class PPOExample:
    """
    PPO example — only the query (prompt) is tokenised here;
    the response is generated online by the policy model.
    """
    query_input_ids: List[int]
    query_attention_mask: List[int]
    query_length: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_input_ids": self.query_input_ids,
            "query_attention_mask": self.query_attention_mask,
            "query_length": self.query_length,
        }


@dataclass(frozen=True, slots=True)
class KTOExample:
    """
    KTO example — single (prompt, completion) pair with a binary
    desirability signal.
    """
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    prompt_length: int
    kto_tag: bool  # True = desirable, False = undesirable
    weight: float  # desirable_weight or undesirable_weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
            "prompt_length": self.prompt_length,
            "kto_tag": self.kto_tag,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class GRPOExample:
    """
    GRPO example — one prompt with *N* candidate completions and
    their associated scores / rankings.
    """
    prompt_input_ids: List[int]
    prompt_attention_mask: List[int]
    responses_input_ids: List[List[int]]
    responses_attention_mask: List[List[int]]
    responses_labels: List[List[int]]
    scores: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_input_ids": self.prompt_input_ids,
            "prompt_attention_mask": self.prompt_attention_mask,
            "responses_input_ids": self.responses_input_ids,
            "responses_attention_mask": self.responses_attention_mask,
            "responses_labels": self.responses_labels,
            "scores": self.scores,
        }


# Union of every output type — used as the generic return where the
# caller does not know the stage at compile time.
ProcessedOutput = Union[
    ProcessedExample,
    PackedExample,
    PreferencePairExample,
    PPOExample,
    KTOExample,
    GRPOExample,
]


# ═════════════════════════════════════════════════════════════════════════════════
# §5  Text Processing Utilities
# ═════════════════════════════════════════════════════════════════════════════════

def _find_sentence_boundary(text: str, max_chars: int) -> int:
    """
    Last sentence boundary ≤ *max_chars*.

    Time  O(max_chars)
    Space O(1)
    """
    if len(text) <= max_chars:
        return len(text)
    region = text[:max_chars]
    last: int = 0
    for m in _SENTENCE_END.finditer(region):
        pos = m.end()
        if pos <= max_chars:
            last = pos
    return last if last > 0 else max_chars


def _find_word_boundary(text: str, max_chars: int) -> int:
    """
    Last whitespace boundary ≤ *max_chars*.

    Time  O(1) — searches backward up to 100 chars.
    Space O(1)
    """
    if len(text) <= max_chars:
        return len(text)
    start = max(0, max_chars - 100)
    region = text[start:max_chars]
    for ch in (" ", "\n", "\t"):
        idx = region.rfind(ch)
        if idx != -1:
            return start + idx
    return max_chars


def smart_truncate(
    text: str,
    max_chars: int,
    strategy: TruncationStrategy = TruncationStrategy.SMART,
) -> str:
    """
    Truncate *text* to at most *max_chars* using the given *strategy*.

    Falls through sentence → word → simple when earlier boundaries
    do not retain enough content.
    """
    if not text or len(text) <= max_chars:
        return text
    if strategy is TruncationStrategy.NONE:
        return text
    if strategy is TruncationStrategy.SIMPLE:
        return text[:max_chars]
    if strategy is TruncationStrategy.WORD_BOUNDARY:
        return text[: _find_word_boundary(text, max_chars)].rstrip()
    if strategy is TruncationStrategy.SENTENCE_BOUNDARY:
        return text[: _find_sentence_boundary(text, max_chars)].rstrip()
    # SMART cascade
    sb = _find_sentence_boundary(text, max_chars)
    if sb >= max_chars * _SENTENCE_MIN_RATIO:
        return text[:sb].rstrip()
    wb = _find_word_boundary(text, max_chars)
    if wb >= max_chars * _WORD_MIN_RATIO:
        return text[:wb].rstrip()
    return text[:max_chars]


def _estimate_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """
    Estimate token count.  Uses *tokenizer* when available,
    otherwise falls back to ≈4 chars / token heuristic.
    """
    if not text:
        return 0
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, "encode"):
                ids = tokenizer.encode(text, add_special_tokens=False)
                return len(ids) if isinstance(ids, list) else len(ids["input_ids"])
            if hasattr(tokenizer, "__call__"):
                return len(tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:
            pass
    return max(1, len(text) // 4)


def _apply_column_mapping(
    example: Dict[str, Any],
    mapping: Dict[str, str],
) -> Dict[str, Any]:
    """
    Rename columns in *example* according to *mapping* (src→dst).

    Non-mapped keys are kept unchanged.  If both src and dst exist,
    dst takes precedence.
    """
    if not mapping:
        return example
    out = dict(example)
    for src, dst in mapping.items():
        if src in out and dst not in out:
            out[dst] = out.pop(src)
    return out


def _safe_str(value: Any) -> str:
    """Convert *value* to ``str``, mapping ``None`` to ``""``."""
    if value is None:
        return ""
    return str(value)


# ═════════════════════════════════════════════════════════════════════════════════
# §6  Jinja2 Template Utilities
# ═════════════════════════════════════════════════════════════════════════════════

def _compile_jinja(template_str: str) -> Optional[Any]:
    """
    Compile *template_str* as a Jinja2 ``Template``.

    Returns ``None`` when Jinja2 is not installed — the caller must
    fall back to :func:`_render_simple`.
    """
    try:
        from jinja2 import Template, StrictUndefined
        return Template(template_str, undefined=StrictUndefined)
    except ImportError:
        return None


def _render_simple(template: str, variables: Dict[str, Any]) -> str:
    """
    Render *template* using simple ``{variable}`` substitution.

    This is the Jinja2-free fallback.  Missing variables are left as-is
    so that the caller can detect them.
    """
    result = template
    for match in SIMPLE_VAR_PATTERN.finditer(template):
        var = match.group(1)
        if var in variables:
            result = result.replace(match.group(0), _safe_str(variables[var]))
    return result


def _find_missing_vars(template: str, variables: Dict[str, Any]) -> List[str]:
    """Return variable names present in *template* but absent from *variables*."""
    return [
        m.group(1)
        for m in SIMPLE_VAR_PATTERN.finditer(template)
        if m.group(1) not in variables
    ]


def render_template(
    template: str,
    variables: Dict[str, Any],
    use_jinja: bool = True,
) -> Result[str, TemplateRenderError]:
    """
    Render a template string with *variables*.

    Attempts Jinja2 first (unless *use_jinja* is ``False``), then falls
    back to simple ``{var}`` substitution.
    """
    try:
        if use_jinja:
            jt = _compile_jinja(template)
            if jt is not None:
                return Ok(jt.render(**variables))
        return Ok(_render_simple(template, variables))
    except Exception as exc:
        return Err(TemplateRenderError(
            message=f"Render failed: {exc}",
            template=template[:200],
            cause=exc,
        ))


# ═════════════════════════════════════════════════════════════════════════════════
# §7  Label-Mask Builder
# ═════════════════════════════════════════════════════════════════════════════════
# Centralises the logic for constructing ``labels`` from ``input_ids``
# with arbitrary masked regions.

class _LabelBuilder:
    """
    Stateless helper that constructs label arrays with masked regions.

    All methods are ``@staticmethod`` — no instance state is needed.
    """

    __slots__ = ()

    @staticmethod
    def full_causal(
        input_ids: List[int],
        attention_mask: List[int],
    ) -> List[int]:
        """
        All real tokens are labels (pre-training regime).

        Padding positions are masked with ``LABEL_PAD_TOKEN_ID``.
        """
        return [
            (tok if mask == 1 else LABEL_PAD_TOKEN_ID)
            for tok, mask in zip(input_ids, attention_mask)
        ]

    @staticmethod
    def mask_prefix(
        input_ids: List[int],
        attention_mask: List[int],
        prefix_length: int,
    ) -> List[int]:
        """
        Mask the first *prefix_length* tokens (SFT / RL prompt masking).

        Tokens beyond *prefix_length* whose ``attention_mask`` is 0 are
        also masked (padding).

        """
        n = len(input_ids)
        labels: List[int] = [LABEL_PAD_TOKEN_ID] * n
        for i in range(prefix_length, n):
            if attention_mask[i] == 1:
                labels[i] = input_ids[i]
        return labels

    @staticmethod
    def mask_ranges(
        input_ids: List[int],
        attention_mask: List[int],
        masked_ranges: List[Tuple[int, int]],
    ) -> List[int]:
        """
        Mask arbitrary ``[start, end)`` ranges — used for multi-turn
        per-role masking where non-contiguous assistant turns must be
        unmasked while everything else is masked.

        Tokens *outside* every masked range keep their original ID;
        tokens *inside* any range (or on padding) get ``LABEL_PAD_TOKEN_ID``.
        """
        n = len(input_ids)
        # Start with all tokens as labels, then carve out masked regions.
        labels: List[int] = list(input_ids)
        for start, end in masked_ranges:
            clamped_start = max(0, start)
            clamped_end = min(end, n)
            for i in range(clamped_start, clamped_end):
                labels[i] = LABEL_PAD_TOKEN_ID
        # Always mask padding positions.
        for i in range(n):
            if attention_mask[i] == 0:
                labels[i] = LABEL_PAD_TOKEN_ID
        return labels

    @staticmethod
    def from_turn_boundaries(
        input_ids: List[int],
        attention_mask: List[int],
        boundaries: List[Tuple[int, int, str]],
        train_roles: FrozenSet[str],
    ) -> List[int]:
        """
        Build labels from turn boundaries ``(start, end, role)``.

        Only tokens whose role is in *train_roles* contribute to the loss;
        all other positions are masked with ``LABEL_PAD_TOKEN_ID``.
        """
        n = len(input_ids)
        labels: List[int] = [LABEL_PAD_TOKEN_ID] * n
        for start, end, role in boundaries:
            if role in train_roles:
                for i in range(max(0, start), min(end, n)):
                    if attention_mask[i] == 1:
                        labels[i] = input_ids[i]
        return labels


# ═════════════════════════════════════════════════════════════════════════════════
# §8  Encoding Helpers
# ═════════════════════════════════════════════════════════════════════════════════
# Thin wrappers around ``TokenizerWrapper`` that standardise
# (input_ids, attention_mask) tuple returns and enforce deterministic
# truncation without padding for internal length calculations.

class _Encoder:
    """
    Stateless encoding helper bound to a ``TokenizerWrapper`` and
    a maximum sequence length.

    Every method returns plain ``(List[int], List[int])`` tuples —
    never tensors — so callers stay framework-agnostic.
    """

    __slots__ = ("_tok", "_raw", "_max_len")

    def __init__(self, tokenizer: TokenizerWrapper, max_length: int) -> None:
        self._tok: TokenizerWrapper = tokenizer
        self._raw: Any = tokenizer.tokenizer          # underlying HF tokenizer
        self._max_len: int = max_length

    # ── core encode ──────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: str = "do_not_pad",
    ) -> Tuple[List[int], List[int]]:
        """
        Encode *text* and return ``(input_ids, attention_mask)``.

        Default: no padding, truncation to *max_length* (or engine max).
        """
        enc = self._tok.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length or self._max_len,
            truncation=truncation,
            padding=padding,
        )
        return enc["input_ids"], enc["attention_mask"]

    def encode_padded(
        self,
        text: str,
        *,
        max_length: Optional[int] = None,
    ) -> Tuple[List[int], List[int]]:
        """Encode with ``max_length`` padding (for final output)."""
        enc = self._tok.encode(
            text,
            max_length=max_length or self._max_len,
            truncation=True,
            padding="max_length",
        )
        return enc["input_ids"], enc["attention_mask"]

    def token_count(self, text: str) -> int:
        """Return exact token count for *text* (no padding, no special)."""
        ids, _ = self.encode(text, add_special_tokens=False)
        return len(ids)

    # ── chat-template helpers ────────────────────────────────────────────

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        *,
        add_generation_prompt: bool = False,
    ) -> Optional[str]:
        """
        Apply HF chat template if available.

        Returns ``None`` when the tokeniser does not support
        ``apply_chat_template``.
        """
        if not hasattr(self._raw, "apply_chat_template"):
            return None
        try:
            return self._raw.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            return None

    # ── special-token accessors ──────────────────────────────────────────

    @property
    def eos_token(self) -> str:
        return self._raw.eos_token or ""

    @property
    def bos_token(self) -> str:
        return self._raw.bos_token or ""

    @property
    def pad_token_id(self) -> int:
        return self._tok.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._tok.eos_token_id


# ═════════════════════════════════════════════════════════════════════════════════
# §9  Message Normalisation
# ═════════════════════════════════════════════════════════════════════════════════
# Many open-source datasets ship messages in divergent schemas
# (ShareGPT, Alpaca, OpenAI, Vicuna, …).  This normaliser maps all
# of them to the canonical ``[{"role": str, "content": str}]`` format.

_SHAREGPT_ROLE_MAP: Dict[str, str] = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
}


def _normalise_messages(
    raw: Union[List[Dict[str, Any]], Any],
) -> List[Dict[str, str]]:
    """
    Convert heterogeneous message lists to ``[{"role", "content"}]``.

    Supported input schemas
    -----------------------
    * OpenAI: ``[{"role": "user", "content": "…"}]``
    * ShareGPT: ``[{"from": "human", "value": "…"}]``
    * String content with ``"text"`` key:
      ``[{"role": "user", "text": "…"}]``

    Unknown roles are mapped to ``"user"``.
    """
    if not raw or not isinstance(raw, list):
        return []

    normalised: List[Dict[str, str]] = []
    for msg in raw:
        if not isinstance(msg, dict):
            continue
        # --- role ---
        role = msg.get("role") or _SHAREGPT_ROLE_MAP.get(
            msg.get("from", ""), "user"
        )
        # --- content ---
        content = msg.get("content") or msg.get("value") or msg.get("text") or ""
        normalised.append({"role": str(role), "content": _safe_str(content)})
    return normalised


def _build_messages_from_columns(
    example: Dict[str, Any],
    input_columns: Sequence[str],
    label_column: Optional[str],
    system_message: Optional[str],
) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """
    Build a canonical message list from flat column-based examples.

    Returns ``(messages, assistant_content)`` where *assistant_content*
    is the label string (or ``None`` if absent).
    """
    messages: List[Dict[str, str]] = []
    assistant_content: Optional[str] = None

    if system_message:
        messages.append({"role": "system", "content": system_message})

    # Concatenate input columns into a single user message.
    user_parts: List[str] = []
    for col in input_columns:
        val = example.get(col)
        if val is not None:
            user_parts.append(_safe_str(val))
    if user_parts:
        messages.append({"role": "user", "content": "\n".join(user_parts)})

    # Assistant / label.
    if label_column and label_column in example:
        assistant_content = _safe_str(example[label_column])
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})

    return messages, assistant_content


def _fallback_format_messages(messages: List[Dict[str, str]]) -> str:
    """
    Format messages using simple role tags when the tokeniser has no
    ``apply_chat_template``.
    """
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{_FALLBACK_ROLE_TAG.format(role=role)}\n{content}")
    return "\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════════
# §10  Pre-Training Processor
# ═════════════════════════════════════════════════════════════════════════════════

class _PreTrainingProcessor:
    """
    Handles all pre-training data formats:
      • Causal LM
      • Fill-in-the-Middle (FIM)
      • T5-style span corruption
      • Document packing (batch-level)

    Thread-safe: stateless after ``__init__``.
    """

    __slots__ = ("_enc", "_cfg", "_max_len", "_rng")

    def __init__(
        self,
        encoder: _Encoder,
        config: PreTrainingConfig,
        max_length: int,
        seed: Optional[int] = None,
    ) -> None:
        self._enc = encoder
        self._cfg = config
        self._max_len = max_length
        # Per-processor RNG for FIM / span corruption reproducibility.
        self._rng = random.Random(
            seed if seed is not None else config.fim.seed
        )

    # ── dispatch ─────────────────────────────────────────────────────────

    def process(
        self, example: Dict[str, Any],
    ) -> Result[ProcessedExample, TokenizationError]:
        """Route to format-specific handler."""
        fmt = self._cfg.format
        try:
            if fmt is PreTrainingFormat.CAUSAL_LM:
                return self._causal_lm(example)
            if fmt is PreTrainingFormat.FILL_IN_MIDDLE:
                return self._fill_in_middle(example)
            if fmt is PreTrainingFormat.SPAN_CORRUPTION:
                return self._span_corruption(example)
            if fmt is PreTrainingFormat.PACKED_DOCUMENTS:
                # Single-document tokenisation; actual packing is batch-level.
                return self._causal_lm(example)
            return Err(TokenizationError(
                message=f"Unsupported pre-training format: {fmt}",
            ))
        except Exception as exc:
            return Err(TokenizationError(
                message=f"Pre-training processing failed: {exc}",
                cause=exc,
            ))

    # ── Causal LM ────────────────────────────────────────────────────────

    def _causal_lm(
        self, example: Dict[str, Any],
    ) -> Result[ProcessedExample, TokenizationError]:
        """
        Standard next-token prediction.  All non-padding tokens are labels.
        """
        text = _safe_str(example.get(self._cfg.text_column, ""))
        if not text:
            return Err(TokenizationError(
                message=f"Empty text in column '{self._cfg.text_column}'",
            ))

        # Optionally prepend BOS / append EOS.
        if self._cfg.add_bos:
            bos = self._enc.bos_token
            if bos and not text.startswith(bos):
                text = bos + text
        if self._cfg.add_eos:
            eos = self._enc.eos_token
            if eos and not text.endswith(eos):
                text = text + eos

        ids, mask = self._enc.encode(
            text,
            add_special_tokens=True,
            padding="max_length",
            max_length=self._max_len,
        )
        labels = _LabelBuilder.full_causal(ids, mask)
        return Ok(ProcessedExample(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
            input_length=0,
        ))

    # ── Fill-in-the-Middle ────────────────────────────────────────────────

    def _fill_in_middle(
        self, example: Dict[str, Any],
    ) -> Result[ProcessedExample, TokenizationError]:
        """
        FIM objective.  With probability ``fim_rate``, the document is
        split into (prefix, middle, suffix) and reordered as PSM or SPM.
        Otherwise falls back to standard causal LM.
        """
        text = _safe_str(example.get(self._cfg.text_column, ""))
        if not text:
            return Err(TokenizationError(
                message=f"Empty text for FIM in column '{self._cfg.text_column}'",
            ))

        fim = self._cfg.fim

        # Probabilistic FIM application.
        if self._rng.random() > fim.fim_rate:
            return self._causal_lm(example)

        # --- split point selection ---
        # Choose a random split creating prefix/middle/suffix.
        text_len = len(text)
        if text_len < 3:
            return self._causal_lm(example)

        max_mid = max(1, int(text_len * fim.max_middle_ratio))
        mid_start = self._rng.randint(0, text_len - 1)
        mid_len = self._rng.randint(1, min(max_mid, text_len - mid_start))
        mid_end = mid_start + mid_len

        prefix_str = text[:mid_start]
        middle_str = text[mid_start:mid_end]
        suffix_str = text[mid_end:]

        # --- assemble with FIM tokens ---
        if fim.mode is FIMMode.PSM:
            fim_text = (
                f"{fim.prefix_token}{prefix_str}"
                f"{fim.suffix_token}{suffix_str}"
                f"{fim.middle_token}{middle_str}"
            )
        else:  # SPM
            fim_text = (
                f"{fim.suffix_token}{suffix_str}"
                f"{fim.prefix_token}{prefix_str}"
                f"{fim.middle_token}{middle_str}"
            )

        if self._cfg.add_eos:
            fim_text += self._enc.eos_token

        ids, mask = self._enc.encode(
            fim_text,
            add_special_tokens=False,
            padding="max_length",
            max_length=self._max_len,
        )
        labels = _LabelBuilder.full_causal(ids, mask)
        return Ok(ProcessedExample(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
            input_length=0,
        ))

    # ── Span Corruption (T5-style, decoder-only adaptation) ──────────────

    def _span_corruption(
        self, example: Dict[str, Any],
    ) -> Result[ProcessedExample, TokenizationError]:
        """
        T5-style span corruption adapted for decoder-only models.

        Layout:  ``corrupted_input <sep> sentinel_0 span_0 sentinel_1 span_1 …``
        Labels mask the corrupted_input prefix so the model only trains
        on reconstructing the original spans.
        """
        text = _safe_str(example.get(self._cfg.text_column, ""))
        if not text:
            return Err(TokenizationError(
                message=f"Empty text for span corruption in '{self._cfg.text_column}'",
            ))

        sc = self._cfg.span_corruption
        # Tokenise raw text to operate at token level.
        raw_ids, _ = self._enc.encode(
            text, add_special_tokens=False, truncation=True,
        )
        n = len(raw_ids)
        if n < 2:
            return self._causal_lm(example)

        # Determine number of tokens to corrupt and span count.
        num_corrupted = max(1, int(n * sc.noise_density))
        num_spans = max(1, int(num_corrupted / sc.mean_span_len))

        # Sample span starting positions (sorted, non-overlapping).
        starts: List[int] = sorted(self._rng.sample(range(n), min(num_spans, n)))

        # Build spans using Poisson-distributed lengths.
        spans: List[Tuple[int, int]] = []
        occupied: List[bool] = [False] * n
        for s in starts:
            if occupied[s]:
                continue
            span_len = max(1, int(self._rng.expovariate(1.0 / sc.mean_span_len)))
            end = min(s + span_len, n)
            # Skip if overlaps existing span.
            if any(occupied[s:end]):
                continue
            for j in range(s, end):
                occupied[j] = True
            spans.append((s, end))
        spans.sort()

        if not spans:
            return self._causal_lm(example)

        # Build corrupted input (replace each span with a sentinel).
        corrupted: List[int] = []
        target: List[int] = []
        sentinel_idx = 0
        prev_end = 0

        for span_start, span_end in spans:
            if sentinel_idx >= sc.num_sentinels:
                break
            sentinel_id = sc.sentinel_start + sentinel_idx
            # Tokens before this span.
            corrupted.extend(raw_ids[prev_end:span_start])
            # Insert sentinel in corrupted input.
            corrupted.append(sentinel_id)
            # Target: sentinel then original span tokens.
            target.append(sentinel_id)
            target.extend(raw_ids[span_start:span_end])
            sentinel_idx += 1
            prev_end = span_end

        # Remaining tokens after last span.
        corrupted.extend(raw_ids[prev_end:])

        # Combine: corrupted_input + separator + target.
        separator: List[int] = [self._enc.eos_token_id] if self._enc.eos_token_id else []
        combined = corrupted + separator + target
        if self._cfg.add_eos and self._enc.eos_token_id:
            combined.append(self._enc.eos_token_id)

        # Truncate to max_length.
        combined = combined[: self._max_len]
        prefix_len = min(len(corrupted) + len(separator), len(combined))

        # Build attention mask (all 1s, pad later if needed).
        attn_mask = [1] * len(combined)
        # Pad to max_length.
        pad_amount = self._max_len - len(combined)
        if pad_amount > 0:
            combined.extend([self._enc.pad_token_id] * pad_amount)
            attn_mask.extend([0] * pad_amount)

        labels = _LabelBuilder.mask_prefix(combined, attn_mask, prefix_len)
        return Ok(ProcessedExample(
            input_ids=combined,
            attention_mask=attn_mask,
            labels=labels,
            input_length=prefix_len,
        ))

    # ── Document Packing (batch-level) ───────────────────────────────────

    def pack_documents(
        self,
        examples: List[Dict[str, Any]],
    ) -> Result[List[PackedExample], TokenizationError]:
        """
        Bin-pack multiple documents into fixed-length sequences.

        Returns a *list* of ``PackedExample`` — typically fewer items
        than the input list because several documents share one sequence.

        Strategy
        --------
        ``GREEDY`` : sequentially fill bins in order.
        ``FIRST_FIT_DECREASING`` : sort by length descending, assign
        each document to the first bin with sufficient remaining capacity.

        Time  O(N log N)  for first-fit-decreasing with binary search.
        Space O(total_tokens + N)
        """
        pc = self._cfg.packing
        try:
            # Step 1: tokenise every document.
            doc_tokens: List[List[int]] = []
            for ex in examples:
                text = _safe_str(ex.get(self._cfg.text_column, ""))
                if not text:
                    continue
                if self._cfg.add_eos:
                    text += self._enc.eos_token
                ids, _ = self._enc.encode(
                    text, add_special_tokens=False, truncation=False,
                )
                if ids:
                    doc_tokens.append(ids)

            if not doc_tokens:
                return Err(TokenizationError(
                    message="No non-empty documents to pack",
                ))

            sep: List[int] = (
                list(pc.separator_tokens)
                if pc.separator_tokens is not None
                else ([self._enc.eos_token_id] if self._enc.eos_token_id else [])
            )
            max_len = pc.max_packed_length

            # Step 2: bin-packing.
            if pc.strategy is PackingStrategy.FIRST_FIT_DECREASING:
                bins = self._ffd_pack(doc_tokens, sep, max_len)
            else:
                bins = self._greedy_pack(doc_tokens, sep, max_len)

            # Step 3: build PackedExamples.
            packed: List[PackedExample] = []
            for bin_docs in bins:
                result = self._build_packed(bin_docs, sep, max_len, pc)
                if result is not None:
                    packed.append(result)

            if not packed:
                return Err(TokenizationError(
                    message="Packing produced zero non-empty bins",
                ))

            return Ok(packed)

        except Exception as exc:
            return Err(TokenizationError(
                message=f"Document packing failed: {exc}", cause=exc,
            ))

    # ── packing algorithms ───────────────────────────────────────────────

    @staticmethod
    def _greedy_pack(
        docs: List[List[int]],
        sep: List[int],
        max_len: int,
    ) -> List[List[List[int]]]:
        """Greedy sequential bin-packing.  O(N)."""
        bins: List[List[List[int]]] = [[]]
        current_len = 0

        for doc in docs:
            needed = len(doc) + (len(sep) if bins[-1] else 0)
            if current_len + needed > max_len and bins[-1]:
                bins.append([])
                current_len = 0
                needed = len(doc)  # No separator for first doc in bin.
            # If single doc exceeds max_len, truncate it.
            if len(doc) > max_len:
                doc = doc[:max_len]
            bins[-1].append(doc)
            current_len += needed
        return bins

    @staticmethod
    def _ffd_pack(
        docs: List[List[int]],
        sep: List[int],
        max_len: int,
    ) -> List[List[List[int]]]:
        """
        First-Fit Decreasing bin-packing.

        Sort documents by length descending then greedily assign each
        to the first bin with enough remaining capacity.

        Uses ``bisect`` for O(N log N) bin lookup.
        """
        indexed = sorted(enumerate(docs), key=lambda t: -len(t[1]))
        bins: List[List[List[int]]] = []
        remaining: List[int] = []  # remaining capacity per bin

        for _, doc in indexed:
            doc_len = min(len(doc), max_len)
            placed = False
            for b_idx in range(len(bins)):
                sep_cost = len(sep) if bins[b_idx] else 0
                if remaining[b_idx] >= doc_len + sep_cost:
                    bins[b_idx].append(doc[:doc_len])
                    remaining[b_idx] -= doc_len + sep_cost
                    placed = True
                    break
            if not placed:
                bins.append([doc[:doc_len]])
                remaining.append(max_len - doc_len)
        return bins

    def _build_packed(
        self,
        bin_docs: List[List[int]],
        sep: List[int],
        max_len: int,
        pc: PackingConfig,
    ) -> Optional[PackedExample]:
        """Assemble one ``PackedExample`` from a bin of documents."""
        if not bin_docs:
            return None

        ids: List[int] = []
        doc_ids: List[int] = []
        pos_ids: List[int] = []
        num_docs = len(bin_docs)

        for d_idx, doc in enumerate(bin_docs):
            # Insert separator between documents.
            if d_idx > 0 and sep:
                ids.extend(sep)
                doc_ids.extend([d_idx - 1] * len(sep))
                pos_ids.extend(
                    range(pos_ids[-1] + 1, pos_ids[-1] + 1 + len(sep))
                    if pos_ids else range(len(sep))
                )
            # Append document tokens.
            start_pos = 0
            ids.extend(doc)
            doc_ids.extend([d_idx] * len(doc))
            pos_ids.extend(range(start_pos, start_pos + len(doc)))

        # Truncate to max_len.
        ids = ids[:max_len]
        doc_ids = doc_ids[:max_len]
        pos_ids = pos_ids[:max_len]
        seq_len = len(ids)

        # Check min fill ratio.
        if pc.drop_remainder and seq_len / max_len < pc.min_fill_ratio:
            return None

        # Pad to max_len.
        pad_amount = max_len - seq_len
        attn = [1] * seq_len + [0] * pad_amount
        labels = list(ids) + [LABEL_PAD_TOKEN_ID] * pad_amount
        ids = ids + [self._enc.pad_token_id] * pad_amount
        doc_ids = doc_ids + [-1] * pad_amount
        pos_ids = pos_ids + [0] * pad_amount

        return PackedExample(
            input_ids=ids,
            attention_mask=attn,
            labels=labels,
            document_ids=doc_ids,
            position_ids=pos_ids,
            num_documents=num_docs,
        )


# ═════════════════════════════════════════════════════════════════════════════════
# §11  Fine-Tuning Processor
# ═════════════════════════════════════════════════════════════════════════════════

class _FineTuningProcessor:
    """
    All supervised fine-tuning formats:
      • Chat         – uses ``apply_chat_template``; falls back gracefully.
      • Completion   – prompt + completion concatenation.
      • Instruction  – Alpaca / Vicuna templating.
      • Multi-turn   – per-turn role-based label masking.
      • Custom       – arbitrary Jinja2 template.

    Thread-safe: stateless after ``__init__``.
    """

    __slots__ = (
        "_enc", "_cfg", "_max_len", "_jinja_tmpl",
        "_legacy_template", "_length_manager", "_distributor",
    )

    def __init__(
        self,
        encoder: _Encoder,
        config: FineTuningConfig,
        max_length: int,
        legacy_template: Optional[PromptTemplate] = None,
        distributor: Optional[TokenAwareContentDistributor] = None,
    ) -> None:
        self._enc = encoder
        self._cfg = config
        self._max_len = max_length
        self._legacy_template = legacy_template
        self._distributor = distributor # distributor is now passed to processor
        self._jinja_tmpl: Optional[Any] = None

        # Pre-compile Jinja template for CUSTOM format.
        if legacy_template and legacy_template.template:
            self._jinja_tmpl = _compile_jinja(legacy_template.template)

    # ── dispatch ─────────────────────────────────────────────────────────

    def process(
        self, example: Dict[str, Any],
    ) -> Result[ProcessedExample, TokenizationError]:
        """Route to format-specific handler."""
        fmt = self._cfg.format
        try:
            if fmt is FineTuningFormat.CHAT:
                return self._chat(example)
            if fmt is FineTuningFormat.COMPLETION:
                return self._completion(example)
            if fmt is FineTuningFormat.INSTRUCTION:
                return self._instruction(example)
            if fmt is FineTuningFormat.MULTI_TURN:
                return self._multi_turn(example)
            if fmt is FineTuningFormat.CUSTOM:
                return self._custom(example)
            return Err(TokenizationError(
                message=f"Unsupported fine-tuning format: {fmt}",
            ))
        except Exception as exc:
            return Err(TokenizationError(
                message=f"Fine-tuning processing failed: {exc}",
                cause=exc,
            ))

    # ── Chat ─────────────────────────────────────────────────────────────

    def _chat(
        self, example: Dict[str, Any],
    ) -> Result[ProcessedExample, TokenizationError]:
        """
        Chat-format SFT.

        Supports:
          * Pre-built message lists (column contains ``[{role, content}]``).
          * Flat columns mapped to user / assistant messages.
          * ``apply_chat_template`` with automatic fallback formatting.

        Input (prompt) tokens are masked in labels when ``mask_input``
        is ``True``.
        """
        messages, assistant_content = self._resolve_messages(example)
        if not messages:
            return Err(TokenizationError(message="No messages to process"))

        # --- encode full conversation ---
        full_text = self._enc.apply_chat_template(
            messages, add_generation_prompt=False,
        )
        if full_text is None:
            full_text = _fallback_format_messages(messages)

        if self._cfg.add_eos and not full_text.endswith(self._enc.eos_token):
            full_text += self._enc.eos_token

        ids, mask = self._enc.encode(
            full_text,
            add_special_tokens=False,
            padding="max_length",
            max_length=self._max_len,
        )

        # --- input-length calculation for prompt masking ---
        input_length = 0
        if self._cfg.mask_input and assistant_content:
            input_msgs = [m for m in messages if m.get("role") != "assistant"]
            # Keep all messages up to (but not including) the last assistant turn.
            last_asst_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "assistant":
                    last_asst_idx = i
                    break
            if last_asst_idx >= 0:
                prompt_msgs = messages[:last_asst_idx]
                prompt_text = self._enc.apply_chat_template(
                    prompt_msgs, add_generation_prompt=True,
                )
                if prompt_text is None:
                    prompt_text = _fallback_format_messages(prompt_msgs)
                prompt_ids, _ = self._enc.encode(
                    prompt_text, add_special_tokens=False,
                )
                input_length = len(prompt_ids)

        labels = _LabelBuilder.mask_prefix(ids, mask, input_length)
        return Ok(ProcessedExample(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
            input_length=input_length,
        ))

    # ── Completion ───────────────────────────────────────────────────────

    def _completion(
        self, example: Dict[str, Any],
    ) -> Result[ProcessedExample, TokenizationError]:
        """
        Simple prompt + completion concatenation.

        The prompt portion is masked in labels so the model only learns
        to generate the completion.
        """
        prompt, completion = self._resolve_prompt_completion(example)
        if not prompt and not completion:
            return Err(TokenizationError(message="Empty prompt and completion"))

        prompt_ids, _ = self._enc.encode(prompt, add_special_tokens=True)
        input_length = len(prompt_ids)

        full_text = prompt + completion
        if self._cfg.add_eos:
            full_text += self._enc.eos_token

        ids, mask = self._enc.encode(
            full_text,
            add_special_tokens=True,
            padding="max_length",
            max_length=self._max_len,
        )
        labels = _LabelBuilder.mask_prefix(ids, mask, input_length) \
            if self._cfg.mask_input else _LabelBuilder.full_causal(ids, mask)

        return Ok(ProcessedExample(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
            input_length=input_length,
        ))

    # ── Instruction (Alpaca-style) ───────────────────────────────────────

    def _instruction(
        self, example: Dict[str, Any],
    ) -> Result[ProcessedExample, TokenizationError]:
        """
        Alpaca-format: ``instruction``, optional ``input``, ``output``.

        Template::

            ### Instruction:
            {instruction}

            ### Input:
            {input}

            ### Response:
            {output}
        """
        instruction = _safe_str(example.get(self._cfg.instruction_column, ""))
        inp = _safe_str(example.get(self._cfg.input_column, ""))
        output = _safe_str(example.get(self._cfg.output_column, ""))

        if not instruction and not output:
            return Err(TokenizationError(
                message="Both instruction and output are empty",
            ))

        # Build prompt.
        prompt_parts: List[str] = [f"### Instruction:\n{instruction}"]
        if inp:
            prompt_parts.append(f"\n\n### Input:\n{inp}")
        prompt_parts.append("\n\n### Response:\n")
        prompt_text = "".join(prompt_parts)

        prompt_ids, _ = self._enc.encode(prompt_text, add_special_tokens=True)
        input_length = len(prompt_ids)

        full_text = prompt_text + output
        if self._cfg.add_eos:
            full_text += self._enc.eos_token

        ids, mask = self._enc.encode(
            full_text,
            add_special_tokens=True,
            padding="max_length",
            max_length=self._max_len,
        )
        labels = _LabelBuilder.mask_prefix(ids, mask, input_length) \
            if self._cfg.mask_input else _LabelBuilder.full_causal(ids, mask)

        return Ok(ProcessedExample(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
            input_length=input_length,
        ))

    # ── Multi-Turn (per-turn loss masking) ───────────────────────────────

    def _multi_turn(
        self, example: Dict[str, Any],
    ) -> Result[ProcessedExample, TokenizationError]:
        """
        Multi-turn conversation with per-role label masking.

        Each turn's token boundaries are found by incremental encoding.
        Only turns whose role is in the *train set* (typically only
        ``assistant``) contribute to the loss.

        Time  O(T × L)  where T = number of turns, L = sequence length.
                         Acceptable for T < 50 which covers virtually all
                         real-world conversations.
        """
        messages = self._resolve_message_list(example)
        if not messages:
            return Err(TokenizationError(
                message="No messages for multi-turn processing",
            ))

        mt = self._cfg.multi_turn

        # Determine which roles are trainable.
        train_roles: set[str] = set()
        if mt.train_on_assistant_only:
            train_roles.add("assistant")
        else:
            if not mt.mask_system:
                train_roles.add("system")
            if not mt.mask_user:
                train_roles.add("user")
            if not mt.mask_tool:
                train_roles.add("tool")
            train_roles.add("assistant")  # Always train on assistant.

        frozen_roles: FrozenSet[str] = frozenset(train_roles)

        # --- find per-turn token boundaries via incremental encoding ---
        boundaries: List[Tuple[int, int, str]] = []
        prev_len = 0

        for k in range(1, len(messages) + 1):
            partial = messages[:k]
            text = self._enc.apply_chat_template(
                partial, add_generation_prompt=False,
            )
            if text is None:
                text = _fallback_format_messages(partial)
            tok_ids, _ = self._enc.encode(text, add_special_tokens=False)
            curr_len = len(tok_ids)
            role = messages[k - 1].get("role", "user")
            boundaries.append((prev_len, curr_len, role))
            prev_len = curr_len

        # --- encode full conversation ---
        full_text = self._enc.apply_chat_template(
            messages, add_generation_prompt=False,
        )
        if full_text is None:
            full_text = _fallback_format_messages(messages)
        if self._cfg.add_eos and not full_text.endswith(self._enc.eos_token):
            full_text += self._enc.eos_token

        ids, mask = self._enc.encode(
            full_text,
            add_special_tokens=False,
            padding="max_length",
            max_length=self._max_len,
        )

        labels = _LabelBuilder.from_turn_boundaries(
            ids, mask, boundaries, frozen_roles,
        )

        # input_length = total masked prefix (for metadata).
        first_train = next(
            (s for s, _, r in boundaries if r in frozen_roles), len(ids)
        )
        return Ok(ProcessedExample(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
            input_length=first_train,
        ))

    # ── Custom (Jinja2) ──────────────────────────────────────────────────

    def _custom(
        self, example: Dict[str, Any],
    ) -> Result[ProcessedExample, TokenizationError]:
        """
        Render an arbitrary Jinja2 template and tokenise the result.

        Falls back to simple ``{var}`` substitution when Jinja2 is not
        installed.
        """
        tmpl = self._legacy_template
        if tmpl is None or not tmpl.template:
            return Err(TemplateRenderError(
                message="No template string defined for custom format",
            ))

        variables: Dict[str, Any] = dict(example)
        variables["eos_token"] = self._enc.eos_token
        variables["bos_token"] = self._enc.bos_token

        # Render.
        try:
            if self._jinja_tmpl is not None:
                full_text = self._jinja_tmpl.render(**variables)
            else:
                full_text = _render_simple(tmpl.template, variables)
        except Exception as exc:
            missing = _find_missing_vars(tmpl.template, variables)
            return Err(TemplateRenderError(
                message=f"Template render failed: {exc}",
                template=tmpl.template[:200],
                missing_vars=tuple(missing),
                cause=exc,
            ))

        # Calculate input length.
        input_length = 0
        if tmpl.mask_input and tmpl.label_column:
            label_val = _safe_str(example.get(tmpl.label_column, ""))
            if label_val:
                label_pos = full_text.rfind(label_val)
                if label_pos > 0:
                    prefix_text = full_text[:label_pos]
                    prefix_ids, _ = self._enc.encode(
                        prefix_text, add_special_tokens=False,
                    )
                    input_length = len(prefix_ids)

        # BOS / EOS.
        if tmpl.add_bos:
            bos = self._enc.bos_token
            if bos and not full_text.startswith(bos):
                full_text = bos + full_text
        if tmpl.add_eos:
            eos = self._enc.eos_token
            if eos and not full_text.endswith(eos):
                full_text += eos

        ids, mask = self._enc.encode(
            full_text,
            add_special_tokens=False,
            padding="max_length",
            max_length=self._max_len,
        )
        labels = _LabelBuilder.mask_prefix(ids, mask, input_length) \
            if input_length > 0 else _LabelBuilder.full_causal(ids, mask)

        return Ok(ProcessedExample(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
            input_length=input_length,
        ))

    # ── helpers ──────────────────────────────────────────────────────────

    def _resolve_messages(
        self, example: Dict[str, Any],
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """
        Extract a canonical message list from *example*.

        Handles:
          * Legacy ``PromptTemplate`` with ``input_columns`` / ``label_column``.
          * Direct ``messages`` column containing a list of dicts.
          * Flat columns mapped to user / assistant.
        """
        # --- legacy PromptTemplate path ---
        if self._legacy_template is not None:
            lt = self._legacy_template
            # Check if the first input column is already a message list.
            if lt.input_columns:
                first_col = lt.input_columns[0]
                val = example.get(first_col)
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    msgs = _normalise_messages(val)
                    asst = None
                    if msgs and msgs[-1].get("role") == "assistant":
                        asst = msgs[-1].get("content")
                    return msgs, asst
            return _build_messages_from_columns(
                example,
                lt.input_columns or [],
                lt.label_column,
                lt.system_message or self._cfg.system_message,
            )

        # --- non-legacy: look for common column names ---
        for col in ("messages", "conversation", "conversations"):
            val = example.get(col)
            if isinstance(val, list):
                msgs = _normalise_messages(val)
                if msgs:
                    asst = None
                    if msgs[-1].get("role") == "assistant":
                        asst = msgs[-1].get("content")
                    return msgs, asst

        # Fallback: build from known columns.
        return _build_messages_from_columns(
            example,
            [self._cfg.instruction_column, self._cfg.input_column],
            self._cfg.output_column,
            self._cfg.system_message,
        )

    def _resolve_message_list(
        self, example: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Extract canonical message list (for multi-turn)."""
        msgs, _ = self._resolve_messages(example)
        return msgs

    def _resolve_prompt_completion(
        self, example: Dict[str, Any],
    ) -> Tuple[str, str]:
        """
        Extract ``(prompt, completion)`` for completion format.

        Uses legacy ``PromptTemplate`` if available, otherwise falls
        back to instruction / output columns.
        """
        if self._legacy_template is not None:
            lt = self._legacy_template
            parts: List[str] = []
            for col in (lt.input_columns or []):
                val = example.get(col)
                if val is not None:
                    parts.append(_safe_str(val))
            prompt = "\n".join(parts)
            completion = ""
            if lt.label_column and lt.label_column in example:
                completion = _safe_str(example[lt.label_column])
            return prompt, completion

        prompt = _safe_str(
            example.get(self._cfg.instruction_column, "")
        )
        inp = _safe_str(example.get(self._cfg.input_column, ""))
        if inp:
            prompt = prompt + "\n" + inp if prompt else inp
        completion = _safe_str(example.get(self._cfg.output_column, ""))
        return prompt, completion

# ═════════════════════════════════════════════════════════════════════════════════
# §12  Post-Training RL Processor
# ═════════════════════════════════════════════════════════════════════════════════

class _RLProcessor:
    """
    All reinforcement-learning / alignment algorithms:
      • DPO       — preference pair (chosen / rejected).
      • PPO       — query-only tokenisation (response generated online).
      • KTO       — single response with binary desirability signal.
      • ORPO      — preference pair with odds-ratio loss formulation.
      • GRPO      — one prompt, N candidate responses with scores.
      • Reward    — preference pair for Bradley-Terry reward modelling.

    Design notes
    ────────────
    • Every algorithm returns a *distinct* frozen dataclass so that
      downstream collators / trainers receive exactly the tensors they
      need — no unused fields, no runtime isinstance() branching.
    • Prompt encoding is shared across algorithms via ``_encode_prompt``.
    • Thread-safe: stateless after ``__init__``.

    Time  O(sequence_length) per example for single-pair algorithms.
          O(N × sequence_length) for GRPO with N candidate responses.
    Space O(sequence_length) per output tensor (no intermediate copies
          kept beyond the current example).
    """

    __slots__ = ("_enc", "_cfg", "_max_len")

    def __init__(
        self,
        encoder: _Encoder,
        config: RLConfig,
        max_length: int,
    ) -> None:
        self._enc = encoder
        self._cfg = config
        self._max_len = max_length

    # ── dispatch ─────────────────────────────────────────────────────────

    def process(
        self, example: Dict[str, Any],
    ) -> Result[ProcessedOutput, TokenizationError]:
        """Route to algorithm-specific handler."""
        algo = self._cfg.algorithm
        try:
            if algo is RLAlgorithm.DPO:
                return self._dpo(example)
            if algo is RLAlgorithm.PPO:
                return self._ppo(example)
            if algo is RLAlgorithm.KTO:
                return self._kto(example)
            if algo is RLAlgorithm.ORPO:
                return self._orpo(example)
            if algo is RLAlgorithm.GRPO:
                return self._grpo(example)
            if algo is RLAlgorithm.REWARD_MODEL:
                return self._reward_model(example)
            return Err(TokenizationError(
                message=f"Unsupported RL algorithm: {algo}",
            ))
        except Exception as exc:
            return Err(TokenizationError(
                message=f"RL processing failed ({algo.name}): {exc}",
                cause=exc,
            ))

    # ── shared helpers ───────────────────────────────────────────────────

    def _resolve_prompt_text(self, example: Dict[str, Any]) -> str:
        """
        Extract prompt text from *example*.

        Supports:
          • String column (``prompt``).
          • Message-list column (``messages`` / ``prompt`` as list of dicts)
            — formatted via ``apply_chat_template`` or fallback tags.
        """
        cols = self._cfg.mapping
        raw_prompt = example.get(cols.prompt)

        # ── message-list prompt ──────────────────────────────────────────
        if isinstance(raw_prompt, list):
            msgs = _normalise_messages(raw_prompt)
            if msgs:
                text = self._enc.apply_chat_template(
                    msgs, add_generation_prompt=True,
                )
                if text is not None:
                    return text
                return _fallback_format_messages(msgs)

        # ── string prompt ────────────────────────────────────────────────
        prompt_str = _safe_str(raw_prompt)

        # Optionally prepend system message.
        if self._cfg.system_message:
            sys_msgs: List[Dict[str, str]] = [
                {"role": "system", "content": self._cfg.system_message},
                {"role": "user", "content": prompt_str},
            ]
            text = self._enc.apply_chat_template(
                sys_msgs, add_generation_prompt=True,
            )
            if text is not None:
                return text
            return _fallback_format_messages(sys_msgs)

        return prompt_str

    def _resolve_response_text(
        self,
        response: Any,
    ) -> str:
        """
        Normalise a response that may be a string or a message-list
        (e.g. ``[{"role": "assistant", "content": "…"}]``).
        """
        if isinstance(response, list):
            msgs = _normalise_messages(response)
            if msgs:
                # Format only assistant turns.
                parts: List[str] = []
                for m in msgs:
                    if m.get("role") == "assistant":
                        parts.append(m.get("content", ""))
                if parts:
                    return "\n".join(parts)
                # Fallback: format all turns.
                text = self._enc.apply_chat_template(
                    msgs, add_generation_prompt=False,
                )
                return text if text is not None else _fallback_format_messages(msgs)
        return _safe_str(response)

    def _encode_prompt(
        self,
        prompt_text: str,
        *,
        max_length: Optional[int] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Encode prompt with BOS prepended if configured.

        Returns ``(input_ids, attention_mask)`` without padding.
        """
        if self._cfg.add_bos:
            bos = self._enc.bos_token
            if bos and not prompt_text.startswith(bos):
                prompt_text = bos + prompt_text

        effective_max = max_length or self._cfg.max_prompt_length or self._max_len
        return self._enc.encode(
            prompt_text,
            add_special_tokens=False,
            max_length=effective_max,
        )

    def _encode_response(
        self,
        response_text: str,
        *,
        max_length: Optional[int] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Encode response with EOS appended if configured.

        Returns ``(input_ids, attention_mask)`` without padding.
        """
        if self._cfg.add_eos:
            eos = self._enc.eos_token
            if eos and not response_text.endswith(eos):
                response_text += eos

        effective_max = max_length or self._cfg.max_completion_length or self._max_len
        return self._enc.encode(
            response_text,
            add_special_tokens=False,
            max_length=effective_max,
        )

    def _build_prompt_response_sequence(
        self,
        prompt_ids: List[int],
        response_ids: List[int],
        *,
        mask_prompt: bool = True,
    ) -> Tuple[List[int], List[int], List[int], int]:
        """
        Concatenate prompt + response, pad to ``_max_len``, build labels.

        Returns ``(input_ids, attention_mask, labels, prompt_length)``.

        The prompt portion of labels is filled with ``LABEL_PAD_TOKEN_ID``
        when *mask_prompt* is ``True`` so the loss ignores prompt tokens.
        """
        combined = (prompt_ids + response_ids)[: self._max_len]
        seq_len = len(combined)
        prompt_len = min(len(prompt_ids), seq_len)

        # Pad to max_length.
        pad_amount = self._max_len - seq_len
        attn = [1] * seq_len + [0] * pad_amount
        ids = combined + [self._enc.pad_token_id] * pad_amount

        if mask_prompt:
            labels = _LabelBuilder.mask_prefix(ids, attn, prompt_len)
        else:
            labels = _LabelBuilder.full_causal(ids, attn)

        return ids, attn, labels, prompt_len

    # ── DPO ──────────────────────────────────────────────────────────────

    def _dpo(
        self, example: Dict[str, Any],
    ) -> Result[PreferencePairExample, TokenizationError]:
        """
        Direct Preference Optimisation.

        Expects columns: ``prompt``, ``chosen``, ``rejected``.
        Each of *chosen* / *rejected* may be a string or a message list.

        Both chosen and rejected sequences include the prompt prefix
        so the model sees identical context; prompt labels are masked.
        """
        cols = self._cfg.mapping
        prompt_text = self._resolve_prompt_text(example)
        chosen_text = self._resolve_response_text(example.get(cols.chosen, ""))
        rejected_text = self._resolve_response_text(example.get(cols.rejected, ""))

        if not prompt_text:
            return Err(TokenizationError(message="DPO: empty prompt"))
        if not chosen_text and not rejected_text:
            return Err(TokenizationError(
                message="DPO: both chosen and rejected are empty",
            ))

        prompt_ids, _ = self._encode_prompt(prompt_text)
        chosen_ids, _ = self._encode_response(chosen_text)
        rejected_ids, _ = self._encode_response(rejected_text)

        c_ids, c_mask, c_labels, _ = self._build_prompt_response_sequence(
            prompt_ids, chosen_ids, mask_prompt=self._cfg.mask_prompt,
        )
        r_ids, r_mask, r_labels, _ = self._build_prompt_response_sequence(
            prompt_ids, rejected_ids, mask_prompt=self._cfg.mask_prompt,
        )

        # Prompt-only tensors (shared context for reference model).
        p_len = min(len(prompt_ids), self._max_len)
        p_pad = self._max_len - p_len
        p_ids = prompt_ids[:p_len] + [self._enc.pad_token_id] * p_pad
        p_mask = [1] * p_len + [0] * p_pad

        return Ok(PreferencePairExample(
            prompt_input_ids=p_ids,
            prompt_attention_mask=p_mask,
            chosen_input_ids=c_ids,
            chosen_attention_mask=c_mask,
            chosen_labels=c_labels,
            rejected_input_ids=r_ids,
            rejected_attention_mask=r_mask,
            rejected_labels=r_labels,
        ))

    # ── PPO ──────────────────────────────────────────────────────────────

    def _ppo(
        self, example: Dict[str, Any],
    ) -> Result[PPOExample, TokenizationError]:
        """
        Proximal Policy Optimisation.

        Only the query (prompt) is tokenised here; the policy model
        generates responses online during training.  The reward model
        scores them in a separate forward pass.

        Truncates / pads the prompt to ``max_prompt_length`` (or
        ``max_length``) so that there is room for generated tokens.
        """
        prompt_text = self._resolve_prompt_text(example)
        if not prompt_text:
            return Err(TokenizationError(message="PPO: empty prompt"))

        # Reserve space for generation.
        gen_budget = self._cfg.generate_max_new_tokens
        prompt_max = max(1, self._max_len - gen_budget)
        if self._cfg.max_prompt_length:
            prompt_max = min(prompt_max, self._cfg.max_prompt_length)

        prompt_ids, _ = self._encode_prompt(
            prompt_text, max_length=prompt_max,
        )
        q_len = len(prompt_ids)

        # Pad to prompt_max for batching.
        pad_amount = prompt_max - q_len
        q_ids = prompt_ids + [self._enc.pad_token_id] * pad_amount
        q_mask = [1] * q_len + [0] * pad_amount

        return Ok(PPOExample(
            query_input_ids=q_ids,
            query_attention_mask=q_mask,
            query_length=q_len,
        ))

    # ── KTO ──────────────────────────────────────────────────────────────

    def _kto(
        self, example: Dict[str, Any],
    ) -> Result[KTOExample, TokenizationError]:
        """
        Kahneman-Tversky Optimisation.

        Each example is a single (prompt, response) pair with a boolean
        *desirability* label.  The loss function up-weights desirable
        examples and down-weights undesirable ones according to
        configured weights.

        Expects columns: ``prompt``, ``response`` (or ``chosen``),
        ``label`` (bool or 0/1).
        """
        cols = self._cfg.columns
        prompt_text = self._resolve_prompt_text(example)
        if not prompt_text:
            return Err(TokenizationError(message="KTO: empty prompt"))

        # Response: try ``response`` first, fall back to ``chosen``.
        response_raw = example.get(cols.response) or example.get(cols.chosen, "")
        response_text = self._resolve_response_text(response_raw)

        # Desirability label.
        label_raw = example.get(cols.kto_label)
        if label_raw is None:
            # Heuristic: if ``chosen`` column present, it's desirable.
            kto_tag = cols.chosen in example and bool(example[cols.chosen])
        else:
            kto_tag = bool(label_raw)

        weight = (
            self._cfg.desirable_weight if kto_tag
            else self._cfg.undesirable_weight
        )

        prompt_ids, _ = self._encode_prompt(prompt_text)
        response_ids, _ = self._encode_response(response_text)

        ids, attn, labels, prompt_len = self._build_prompt_response_sequence(
            prompt_ids, response_ids, mask_prompt=self._cfg.mask_prompt,
        )

        return Ok(KTOExample(
            input_ids=ids,
            attention_mask=attn,
            labels=labels,
            prompt_length=prompt_len,
            kto_tag=kto_tag,
            weight=weight,
        ))

    # ── ORPO ─────────────────────────────────────────────────────────────

    def _orpo(
        self, example: Dict[str, Any],
    ) -> Result[PreferencePairExample, TokenizationError]:
        """
        Odds Ratio Preference Optimisation.

        Structurally identical to DPO at the data level — the
        difference is in the training loss.  We reuse the same
        ``PreferencePairExample`` output type so the trainer can
        compute the odds-ratio loss from chosen / rejected log-probs.
        """
        # ORPO shares the exact same tokenisation as DPO.
        return self._dpo(example)

    # ── GRPO ─────────────────────────────────────────────────────────────

    def _grpo(
        self, example: Dict[str, Any],
    ) -> Result[GRPOExample, TokenizationError]:
        """
        Group Relative Policy Optimisation.

        One prompt paired with *N* candidate responses, each scored.
        The policy is updated by comparing within-group relative
        advantages.

        Expects columns:
          • ``prompt``    — string or message list.
          • ``responses`` — list of strings (or message lists).
          • ``rankings``  — list of float scores aligned with responses.

        If ``responses`` is absent, falls back to ``chosen`` /
        ``rejected`` and synthesises a two-element group.
        """
        cols = self._cfg.columns
        prompt_text = self._resolve_prompt_text(example)
        if not prompt_text:
            return Err(TokenizationError(message="GRPO: empty prompt"))

        # --- resolve responses and scores ---
        raw_responses = example.get(cols.responses)
        raw_scores = example.get(cols.rankings)

        if isinstance(raw_responses, list) and raw_responses:
            responses: List[str] = [
                self._resolve_response_text(r) for r in raw_responses
            ]
            scores: List[float] = (
                [float(s) for s in raw_scores]
                if isinstance(raw_scores, list) and len(raw_scores) == len(responses)
                else [0.0] * len(responses)
            )
        else:
            # Fallback: build two-element group from chosen / rejected.
            chosen_text = self._resolve_response_text(
                example.get(cols.chosen, ""),
            )
            rejected_text = self._resolve_response_text(
                example.get(cols.rejected, ""),
            )
            if not chosen_text and not rejected_text:
                return Err(TokenizationError(
                    message="GRPO: no responses found",
                ))
            responses = []
            scores = []
            if chosen_text:
                responses.append(chosen_text)
                scores.append(1.0)
            if rejected_text:
                responses.append(rejected_text)
                scores.append(0.0)

        if not responses:
            return Err(TokenizationError(message="GRPO: empty response list"))

        # --- encode prompt ---
        prompt_ids, _ = self._encode_prompt(prompt_text)
        p_len = min(len(prompt_ids), self._max_len)
        p_pad = self._max_len - p_len
        p_ids = prompt_ids[:p_len] + [self._enc.pad_token_id] * p_pad
        p_mask = [1] * p_len + [0] * p_pad

        # --- encode each response ---
        all_resp_ids: List[List[int]] = []
        all_resp_mask: List[List[int]] = []
        all_resp_labels: List[List[int]] = []

        for resp_text in responses:
            resp_ids, _ = self._encode_response(resp_text)
            ids, attn, labels, _ = self._build_prompt_response_sequence(
                prompt_ids, resp_ids, mask_prompt=self._cfg.mask_prompt,
            )
            all_resp_ids.append(ids)
            all_resp_mask.append(attn)
            all_resp_labels.append(labels)

        return Ok(GRPOExample(
            prompt_input_ids=p_ids,
            prompt_attention_mask=p_mask,
            responses_input_ids=all_resp_ids,
            responses_attention_mask=all_resp_mask,
            responses_labels=all_resp_labels,
            scores=scores,
        ))

    # ── Reward Model ─────────────────────────────────────────────────────

    def _reward_model(
        self, example: Dict[str, Any],
    ) -> Result[PreferencePairExample, TokenizationError]:
        """
        Bradley-Terry reward model.

        Same data layout as DPO (chosen / rejected pairs); the trainer
        uses a scalar head instead of the DPO implicit-reward formulation.

        Prompt labels are *always* masked because the reward head
        typically reads from the last token of each sequence.
        """
        # Identical tokenisation to DPO — force prompt masking.
        cols = self._cfg.columns
        prompt_text = self._resolve_prompt_text(example)
        chosen_text = self._resolve_response_text(example.get(cols.chosen, ""))
        rejected_text = self._resolve_response_text(example.get(cols.rejected, ""))

        if not prompt_text:
            return Err(TokenizationError(message="RewardModel: empty prompt"))

        prompt_ids, _ = self._encode_prompt(prompt_text)
        chosen_ids, _ = self._encode_response(chosen_text)
        rejected_ids, _ = self._encode_response(rejected_text)

        c_ids, c_mask, c_labels, _ = self._build_prompt_response_sequence(
            prompt_ids, chosen_ids, mask_prompt=True,
        )
        r_ids, r_mask, r_labels, _ = self._build_prompt_response_sequence(
            prompt_ids, rejected_ids, mask_prompt=True,
        )

        p_len = min(len(prompt_ids), self._max_len)
        p_pad = self._max_len - p_len
        p_ids = prompt_ids[:p_len] + [self._enc.pad_token_id] * p_pad
        p_mask = [1] * p_len + [0] * p_pad

        return Ok(PreferencePairExample(
            prompt_input_ids=p_ids,
            prompt_attention_mask=p_mask,
            chosen_input_ids=c_ids,
            chosen_attention_mask=c_mask,
            chosen_labels=c_labels,
            rejected_input_ids=r_ids,
            rejected_attention_mask=r_mask,
            rejected_labels=r_labels,
        ))


# ═════════════════════════════════════════════════════════════════════════════════
# §13  Unified Prompt Engine
# ═════════════════════════════════════════════════════════════════════════════════
#
# Single entry-point that delegates to the stage-specific processor.
#
# Construction is O(1) (just wiring references); all heavy work happens
# lazily inside ``process()`` / ``process_batch()``.
#
# Thread-safe: all internal processors are stateless after init.
# ═════════════════════════════════════════════════════════════════════════════════

class PromptEngine:
    """
    Universal training-stage data formatter.

    Unified API
    ───────────
    ``process(example)``       → ``Result[ProcessedOutput, TokenizationError]``
    ``process_batch(examples)``→ ``List[Result[…]]``
    ``pack_documents(examples)``→ ``Result[List[PackedExample], …]``
        (pre-training packing only)

    Construction
    ────────────
    Option A — new-style ``PromptEngineConfig``::

        engine = PromptEngine.from_config(config, tokenizer_wrapper)

    Option B — legacy ``PromptTemplate`` (backward-compatible)::

        engine = PromptEngine.from_legacy(template, tokenizer_wrapper)

    Time  O(sequence_length) per single-sequence example.
    Space O(sequence_length) per output (no retained intermediates).
    """

    __slots__ = (
        "_config",
        "_encoder",
        "_pre_processor",
        "_ft_processor",
        "_rl_processor",
        "_length_manager",
        "_distributor",
    )

    # ── private constructor — use factory class-methods ──────────────────

    def __init__(
        self,
        config: PromptEngineConfig,
        tokenizer: TokenizerWrapper,
        distributor: Optional[TokenAwareContentDistributor] = None,
    ) -> None:
        self._config = config
        self._encoder = _Encoder(tokenizer, config.max_length)
        self._distributor = distributor
        self._length_manager = None # LengthManager is deprecated, distributor replaces it

        # Lazily-initialised processors (only the active stage is used).
        self._pre_processor: Optional[_PreTrainingProcessor] = None
        self._ft_processor: Optional[_FineTuningProcessor] = None
        self._rl_processor: Optional[_RLProcessor] = None

        stage = config.stage

        if stage is TrainingStage.PRE_TRAINING:
            self._pre_processor = _PreTrainingProcessor(
                encoder=self._encoder,
                config=config.pretraining,
                max_length=config.max_length,
            )

        elif stage is TrainingStage.FINE_TUNING:
            self._ft_processor = _FineTuningProcessor(
                encoder=self._encoder,
                config=config.finetuning,
                max_length=config.max_length,
                legacy_template=config.template,
                distributor=distributor,
            )

        elif stage is TrainingStage.POST_TRAINING_RL:
            self._rl_processor = _RLProcessor(
                encoder=self._encoder,
                config=config.rl,
                max_length=config.max_length,
            )

    # ── factory: from PromptEngineConfig ─────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: PromptEngineConfig,
        tokenizer: TokenizerWrapper,
        distributor: Optional[TokenAwareContentDistributor] = None,
    ) -> "PromptEngine":
        """
        Construct engine from a fully-specified ``PromptEngineConfig``.

        This is the recommended constructor for new code.
        """
        return cls(config, tokenizer, distributor)

    # ── factory: from legacy PromptTemplate (backward-compatible) ────────

    @classmethod
    def from_legacy(
        cls,
        template: PromptTemplate,
        tokenizer: TokenizerWrapper,
        length_manager: Optional[Any] = None, # Deprecated, kept for backward compatibility
        max_length: Optional[int] = None,
        per_column_limits: Optional[Dict[str, int]] = None,
        distributor: Optional[TokenAwareContentDistributor] = None,
    ) -> "PromptEngine":
        """
        Construct engine from a legacy ``PromptTemplate``.

        Maps the old template fields onto the new ``PromptEngineConfig``
        so that existing YAML configurations continue to work without
        modification.
        """
        # Infer fine-tuning format from template format_type.
        fmt_map: Dict[str, FineTuningFormat] = {
            "chat": FineTuningFormat.CHAT,
            "completion": FineTuningFormat.COMPLETION,
            "custom": FineTuningFormat.CUSTOM,
        }
        ft_format = fmt_map.get(
            getattr(template, "format_type", "chat"),
            FineTuningFormat.CHAT,
        )

        ft_config = FineTuningConfig(
            format=ft_format,
            system_message=getattr(template, "system_message", None),
            add_eos=getattr(template, "add_eos", True),
            add_bos=getattr(template, "add_bos", False),
            mask_input=getattr(template, "mask_input", True),
        )

        effective_max = max_length or tokenizer.config.max_length

        config = PromptEngineConfig(
            stage=TrainingStage.FINE_TUNING,
            max_length=effective_max,
            finetuning=ft_config,
            template=template,
            per_column_limits=per_column_limits or {},
        )

        # If length_manager is provided, it takes precedence over distributor for legacy.
        # However, the new __init__ only accepts distributor, so we'll pass distributor.
        # The internal _length_manager will be None.
        return cls(config, tokenizer, distributor)

    # ── public properties ────────────────────────────────────────────────

    @property
    def stage(self) -> TrainingStage:
        """Active training stage."""
        return self._config.stage

    @property
    def config(self) -> PromptEngineConfig:
        """Engine configuration (read-only)."""
        return self._config

    @property
    def max_length(self) -> int:
        """Maximum sequence length."""
        return self._config.max_length

    # ── preprocessing ────────────────────────────────────────────────────

    def _preprocess_example(
        self, example: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply column mapping, per-column character limits, and
        optional ``LengthManager`` preprocessing before tokenisation.

        Time  O(total_text_length)
        Space O(num_columns)
        """
        # Step 0: Apply TokenAwareContentDistributor if active
        if isinstance(example, dict):
            if self._distributor is not None:
                # Determine text columns to distribute over
                # For custom/legacy, this should come from template input_columns
                # For standardized formats (instruct, etc.), we can infer or fallback to all str
                text_columns = []
                if self._config.template and self._config.template.input_columns:
                    text_columns = self._config.template.input_columns
                elif self._config.stage is TrainingStage.FINE_TUNING and self._config.finetuning.format is FineTuningFormat.CHAT:
                    # For chat format, we typically distribute over 'content' in messages
                    # This is a simplification; a more robust solution might inspect the template
                    pass # Distributor should handle message lists directly
                else:
                    # Fallback: find all string columns
                    text_columns = [k for k, v in example.items() if isinstance(v, str)]

                # Apply distribution - this modifies the example in-place or returns new dict
                # distribute returns Dict[str, str] (truncated texts)
                truncated_texts = self._distributor.distribute(example, text_columns=text_columns)
                # Create a new dict to avoid modifying the original example if it's from a dataset
                example = dict(example)
                example.update(truncated_texts)

        # Step 1: column mapping (src → dst renaming).
        processed = _apply_column_mapping(example, self._config.column_mapping)

        # Step 2: LengthManager integration (deprecated, but kept for backward compatibility if passed via from_legacy).
        # Note: The new __init__ sets _length_manager to None.
        if self._length_manager is not None:
            try:
                processed = self._length_manager.preprocess_example(processed)
            except Exception:
                pass  # Graceful degradation — use untruncated text.

        # Step 3: per-column character limits (legacy).
        limits = self._config.per_column_limits
        if limits:
            for col, max_chars in limits.items():
                val = processed.get(col)
                if isinstance(val, str) and len(val) > max_chars:
                    processed[col] = smart_truncate(
                        val, max_chars, self._config.truncation_strategy,
                    )

        return processed

    # ── single-example processing ────────────────────────────────────────

    def process(
        self, example: Dict[str, Any],
    ) -> Result[ProcessedOutput, TokenizationError]:
        """
        Process a single example through the active training stage.

        Returns the stage-appropriate output dataclass wrapped in
        ``Ok``, or a ``TokenizationError`` wrapped in ``Err``.

        Thread-safe.

        Time  O(sequence_length)
        Space O(sequence_length)
        """
        try:
            preprocessed = self._preprocess_example(example)
        except Exception as exc:
            return Err(TokenizationError(
                message=f"Preprocessing failed: {exc}", cause=exc,
            ))

        stage = self._config.stage

        if stage is TrainingStage.PRE_TRAINING:
            if self._pre_processor is None:
                return Err(TokenizationError(
                    message="Pre-training processor not initialised",
                ))
            return self._pre_processor.process(preprocessed)

        if stage is TrainingStage.FINE_TUNING:
            if self._ft_processor is None:
                return Err(TokenizationError(
                    message="Fine-tuning processor not initialised",
                ))
            return self._ft_processor.process(preprocessed)

        if stage is TrainingStage.POST_TRAINING_RL:
            if self._rl_processor is None:
                return Err(TokenizationError(
                    message="RL processor not initialised",
                ))
            return self._rl_processor.process(preprocessed)

        return Err(TokenizationError(
            message=f"Unknown training stage: {stage}",
        ))

    # ── batch processing ─────────────────────────────────────────────────

    def process_batch(
        self,
        examples: List[Dict[str, Any]],
    ) -> List[Result[ProcessedOutput, TokenizationError]]:
        """
        Process a batch of examples.

        Note: Python GIL prevents true parallelism here.  Use
        ``DataLoader(num_workers=N)`` for multi-process acceleration.

        Time  O(B × sequence_length) where B = batch size.
        Space O(B × sequence_length)
        """
        return [self.process(ex) for ex in examples]

    # ── document packing (pre-training only) ─────────────────────────────

    def pack_documents(
        self,
        examples: List[Dict[str, Any]],
    ) -> Result[List[PackedExample], TokenizationError]:
        """
        Bin-pack multiple documents into fixed-length sequences.

        Only available when ``stage == PRE_TRAINING`` and ``format ==
        PACKED_DOCUMENTS``.  For other stages this returns an error.

        Time  O(N log N)  (first-fit-decreasing) or O(N) (greedy).
        Space O(total_tokens)
        """
        if self._config.stage is not TrainingStage.PRE_TRAINING:
            return Err(TokenizationError(
                message="pack_documents() is only available for PRE_TRAINING stage",
            ))
        if self._pre_processor is None:
            return Err(TokenizationError(
                message="Pre-training processor not initialised",
            ))

        preprocessed = [self._preprocess_example(ex) for ex in examples]
        return self._pre_processor.pack_documents(preprocessed)

    # ── introspection / debugging ────────────────────────────────────────

    def describe(self) -> Dict[str, Any]:
        """
        Return a human-readable summary of the engine configuration.

        Useful for logging / experiment tracking.
        """
        stage = self._config.stage
        info: Dict[str, Any] = {
            "stage": stage.name,
            "max_length": self._config.max_length,
            "truncation_strategy": self._config.truncation_strategy.name,
        }

        if stage is TrainingStage.PRE_TRAINING:
            pt = self._config.pretraining
            info["format"] = pt.format.name
            info["text_column"] = pt.text_column
            if pt.format is PreTrainingFormat.FILL_IN_MIDDLE:
                info["fim_rate"] = pt.fim.fim_rate
                info["fim_mode"] = pt.fim.mode.name
            if pt.format is PreTrainingFormat.PACKED_DOCUMENTS:
                info["packing_strategy"] = pt.packing.strategy.name
                info["max_packed_length"] = pt.packing.max_packed_length
            if pt.format is PreTrainingFormat.SPAN_CORRUPTION:
                info["noise_density"] = pt.span_corruption.noise_density
                info["mean_span_len"] = pt.span_corruption.mean_span_len

        elif stage is TrainingStage.FINE_TUNING:
            ft = self._config.finetuning
            info["format"] = ft.format.name
            info["mask_input"] = ft.mask_input
            if ft.format is FineTuningFormat.MULTI_TURN:
                info["train_on_assistant_only"] = ft.multi_turn.train_on_assistant_only

        elif stage is TrainingStage.POST_TRAINING_RL:
            rl = self._config.rl
            info["algorithm"] = rl.algorithm.name
            info["mask_prompt"] = rl.mask_prompt
            if rl.algorithm is RLAlgorithm.GRPO:
                info["num_generations"] = rl.num_generations
            if rl.algorithm is RLAlgorithm.KTO:
                info["desirable_weight"] = rl.desirable_weight
                info["undesirable_weight"] = rl.undesirable_weight

        return info


# ═════════════════════════════════════════════════════════════════════════════════
# §14  Factory Functions — Ergonomic Constructors
# ═════════════════════════════════════════════════════════════════════════════════
#
# These functions provide concise, intent-revealing constructors for the
# most common engine configurations.  They hide the nested dataclass
# wiring so that callers can express their intent in one call.
# ═════════════════════════════════════════════════════════════════════════════════

def create_pretraining_engine(
    tokenizer: TokenizerWrapper,
    *,
    format: Union[str, PreTrainingFormat] = PreTrainingFormat.CAUSAL_LM,
    text_column: str = "text",
    max_length: int = 2048,
    add_eos: bool = True,
    add_bos: bool = False,
    # FIM params.
    fim_rate: float = 0.5,
    fim_mode: Union[str, FIMMode] = FIMMode.PSM,
    fim_prefix_token: str = _DEFAULT_FIM_PREFIX,
    fim_suffix_token: str = _DEFAULT_FIM_SUFFIX,
    fim_middle_token: str = _DEFAULT_FIM_MIDDLE,
    # Packing params.
    packing_strategy: Union[str, PackingStrategy] = PackingStrategy.GREEDY,
    max_packed_length: Optional[int] = None,
    packing_separator_tokens: Optional[Tuple[int, ...]] = None,
    packing_track_document_ids: bool = True,
    # Span corruption params.
    noise_density: float = 0.15,
    mean_span_len: float = 3.0,
    sentinel_start: int = 32000,
    num_sentinels: int = 100,
    # Misc.
    seed: Optional[int] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    per_column_limits: Optional[Dict[str, int]] = None,
) -> PromptEngine:
    """
    Create a ``PromptEngine`` configured for pre-training.

    Supports causal LM, fill-in-the-middle, span corruption,
    and document packing — selectable via the *format* parameter.
    """
    if isinstance(format, str):
        format = PreTrainingFormat[format.upper()]
    if isinstance(fim_mode, str):
        fim_mode = FIMMode[fim_mode.upper()]
    if isinstance(packing_strategy, str):
        packing_strategy = PackingStrategy[packing_strategy.upper()]

    config = PromptEngineConfig(
        stage=TrainingStage.PRE_TRAINING,
        max_length=max_length,
        pretraining=PreTrainingConfig(
            format=format,
            text_column=text_column,
            add_eos=add_eos,
            add_bos=add_bos,
            fim=FIMConfig(
                fim_rate=fim_rate,
                mode=fim_mode,
                prefix_token=fim_prefix_token,
                suffix_token=fim_suffix_token,
                middle_token=fim_middle_token,
                seed=seed,
            ),
            packing=PackingConfig(
                strategy=packing_strategy,
                max_packed_length=max_packed_length or max_length,
                separator_tokens=packing_separator_tokens,
                track_document_ids=packing_track_document_ids,
            ),
            span_corruption=SpanCorruptionConfig(
                noise_density=noise_density,
                mean_span_len=mean_span_len,
                sentinel_start=sentinel_start,
                num_sentinels=num_sentinels,
            ),
        ),
        column_mapping=column_mapping or {},
        per_column_limits=per_column_limits or {},
    )
    return PromptEngine.from_config(config, tokenizer)


def create_finetuning_engine(
    tokenizer: TokenizerWrapper,
    *,
    format: Union[str, FineTuningFormat] = FineTuningFormat.CHAT,
    max_length: int = 2048,
    mask_input: bool = True,
    add_eos: bool = True,
    add_bos: bool = False,
    system_message: Optional[str] = None,
    # Instruction columns.
    instruction_column: str = "instruction",
    input_column: str = "input",
    output_column: str = "output",
    # Multi-turn config.
    train_on_assistant_only: bool = True,
    mask_system: bool = True,
    mask_user: bool = True,
    mask_tool: bool = True,
    # Legacy template.
    template: Optional[PromptTemplate] = None,
    # Misc.
    column_mapping: Optional[Dict[str, str]] = None,
    per_column_limits: Optional[Dict[str, int]] = None,
) -> PromptEngine:
    """
    Create a ``PromptEngine`` configured for supervised fine-tuning.

    Supports chat, completion, instruction, multi-turn, and custom
    Jinja2 formats — selectable via the *format* parameter.
    """
    if isinstance(format, str):
        format = FineTuningFormat[format.upper()]

    ft_config = FineTuningConfig(
        format=format,
        multi_turn=MultiTurnConfig(
            mask_system=mask_system,
            mask_user=mask_user,
            mask_tool=mask_tool,
            train_on_assistant_only=train_on_assistant_only,
        ),
        instruction_column=instruction_column,
        input_column=input_column,
        output_column=output_column,
        system_message=system_message,
        add_eos=add_eos,
        add_bos=add_bos,
        mask_input=mask_input,
    )

    config = PromptEngineConfig(
        stage=TrainingStage.FINE_TUNING,
        max_length=max_length,
        finetuning=ft_config,
        template=template,
        column_mapping=column_mapping or {},
        per_column_limits=per_column_limits or {},
    )
    return PromptEngine.from_config(config, tokenizer, length_manager)


def create_rl_engine(
    tokenizer: TokenizerWrapper,
    *,
    algorithm: Union[str, RLAlgorithm] = RLAlgorithm.DPO,
    max_length: int = 2048,
    mask_prompt: bool = True,
    add_eos: bool = True,
    add_bos: bool = False,
    system_message: Optional[str] = None,
    # Column names.
    prompt_column: str = "prompt",
    chosen_column: str = "chosen",
    rejected_column: str = "rejected",
    response_column: str = "response",
    reward_column: str = "reward",
    kto_label_column: str = "label",
    messages_column: str = "messages",
    responses_column: str = "responses",
    rankings_column: str = "rankings",
    # Length limits.
    max_prompt_length: Optional[int] = None,
    max_response_length: Optional[int] = None,
    # PPO.
    generate_max_new_tokens: int = 256,
    # KTO.
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    # GRPO.
    num_generations: int = 4,
    # Misc.
    column_mapping: Optional[Dict[str, str]] = None,
    per_column_limits: Optional[Dict[str, int]] = None,
) -> PromptEngine:
    """
    Create a ``PromptEngine`` configured for post-training RL / alignment.

    Supports DPO, PPO, KTO, ORPO, GRPO, and reward-model formats —
    selectable via the *algorithm* parameter.
    """
    if isinstance(algorithm, str):
        algorithm = RLAlgorithm[algorithm.upper()]

    rl_config = RLConfig(
        algorithm=algorithm,
        columns=RLColumnMapping(
            prompt=prompt_column,
            chosen=chosen_column,
            rejected=rejected_column,
            response=response_column,
            reward=reward_column,
            kto_label=kto_label_column,
            messages=messages_column,
            responses=responses_column,
            rankings=rankings_column,
        ),
        system_message=system_message,
        mask_prompt=mask_prompt,
        add_eos=add_eos,
        add_bos=add_bos,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        generate_max_new_tokens=generate_max_new_tokens,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
        num_generations=num_generations,
    )

    config = PromptEngineConfig(
        stage=TrainingStage.POST_TRAINING_RL,
        max_length=max_length,
        rl=rl_config,
        column_mapping=column_mapping or {},
        per_column_limits=per_column_limits or {},
    )
    return PromptEngine.from_config(config, tokenizer)


# ═════════════════════════════════════════════════════════════════════════════════
# §15  Batch Utilities — Collation & Result Extraction
# ═════════════════════════════════════════════════════════════════════════════════

def extract_successful(
    results: List[Result[ProcessedOutput, TokenizationError]],
) -> Tuple[List[ProcessedOutput], List[TokenizationError]]:
    """
    Partition a list of ``Result`` values into successes and failures.

    Returns ``(ok_values, err_values)`` so the caller can log errors
    and proceed with valid examples without silent data loss.

    Time  O(N)
    Space O(N)
    """
    oks: List[ProcessedOutput] = []
    errs: List[TokenizationError] = []
    for r in results:
        if isinstance(r, Ok):
            oks.append(r.value)
        elif isinstance(r, Err):
            errs.append(r.error)
    return oks, errs


def batch_to_dict(
    examples: List[ProcessedOutput],
) -> Dict[str, List[Any]]:
    """
    Convert a list of output dataclasses to a column-oriented dict
    suitable for ``datasets.Dataset.from_dict()``.

    Handles heterogeneous output types by taking the union of all
    fields and filling missing values with ``None``.

    Time  O(B × F) where B = batch size, F = num fields.
    Space O(B × F)
    """
    if not examples:
        return {}

    # Collect all field names across output types.
    all_keys: List[str] = []
    seen: set[str] = set()
    for ex in examples:
        d = ex.to_dict()
        for k in d:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    # Build column-oriented dict.
    result: Dict[str, List[Any]] = {k: [] for k in all_keys}
    for ex in examples:
        d = ex.to_dict()
        for k in all_keys:
            result[k].append(d.get(k))

    return result


def create_dataset_map_fn(
    engine: PromptEngine,
    *,
    drop_errors: bool = True,
) -> Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Create a mapping function compatible with ``datasets.Dataset.map()``.

    Returns ``None`` for failed examples when *drop_errors* is ``True``
    (use ``dataset.filter(lambda x: x is not None)`` afterwards).

    Usage::

        map_fn = create_dataset_map_fn(engine)
        processed = dataset.map(map_fn, remove_columns=dataset.column_names)
        processed = processed.filter(lambda x: x is not None)

    Time  O(sequence_length) per call.
    """
    def map_fn(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result = engine.process(example)
        if isinstance(result, Ok):
            return result.value.to_dict()
        if drop_errors:
            return None
        # Return empty dict so the row is not silently dropped.
        return {}

    return map_fn


# ═════════════════════════════════════════════════════════════════════════════════
# §16  Module Public API
# ═════════════════════════════════════════════════════════════════════════════════
#
# Explicit ``__all__`` for controlled namespace export.
# Only symbols listed here are part of the public contract.
# ═════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # ── Constants ────────────────────────────────────────────────────────
    "LABEL_PAD_TOKEN_ID",
    # ── Enums ────────────────────────────────────────────────────────────
    "TrainingStage",
    "PreTrainingFormat",
    "FineTuningFormat",
    "RLAlgorithm",
    "FIMMode",
    "PackingStrategy",
    "TruncationStrategy",
    # ── Configuration ────────────────────────────────────────────────────
    "FIMConfig",
    "PackingConfig",
    "SpanCorruptionConfig",
    "PreTrainingConfig",
    "MultiTurnConfig",
    "FineTuningConfig",
    "RLColumnMapping",
    "RLConfig",
    "PromptEngineConfig",
    # ── Output Types ─────────────────────────────────────────────────────
    "ProcessedExample",
    "PackedExample",
    "PreferencePairExample",
    "PPOExample",
    "KTOExample",
    "GRPOExample",
    "ProcessedOutput",
    # ── Engine ───────────────────────────────────────────────────────────
    "PromptEngine",
    # ── Factory Functions ────────────────────────────────────────────────
    "create_pretraining_engine",
    "create_finetuning_engine",
    "create_rl_engine",
    # ── Utilities ────────────────────────────────────────────────────────
    "render_template",
    "smart_truncate",
    "extract_successful",
    "batch_to_dict",
    "create_dataset_map_fn",
]