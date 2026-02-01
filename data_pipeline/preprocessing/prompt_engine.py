# ════════════════════════════════════════════════════════════════════════════════
# Prompt Engine - SDK-Compatible Templating
# ════════════════════════════════════════════════════════════════════════════════
# Constructs tokenized sequences from YAML-driven templates.
# Supports HuggingFace chat templates, OpenAI format, and custom Jinja2 templates.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)

from data_pipeline.core.types import Result, Ok, Err
from data_pipeline.core.errors import TokenizationError, TemplateRenderError
from data_pipeline.core.config_schema import PromptTemplate
from data_pipeline.preprocessing.tokenization import TokenizerWrapper

if TYPE_CHECKING:
    from jinja2 import Template


# ─────────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────────

# Label padding value for CrossEntropyLoss ignore_index
LABEL_PAD_TOKEN_ID: int = -100

# Simple variable pattern for non-Jinja templates
SIMPLE_VAR_PATTERN = re.compile(r"\{(\w+)\}")


# ─────────────────────────────────────────────────────────────────────────────────
# Template Output
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ProcessedExample:
    """
    Result of processing an example through the prompt engine.
    
    Attributes:
        input_ids: Token IDs for the full sequence
        attention_mask: Attention mask (1 for real tokens, 0 for padding)
        labels: Labels for loss computation (-100 for masked positions)
        input_length: Length of input portion (for input masking)
    """
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    input_length: int


# ─────────────────────────────────────────────────────────────────────────────────
# Prompt Engine
# ─────────────────────────────────────────────────────────────────────────────────

class PromptEngine:
    """
    Applies prompt templates with SDK compatibility.
    
    Supports:
    - transformers apply_chat_template()
    - OpenAI messages format
    - Custom Jinja2 templates
    - SOTA length management for variable-sized columns
    
    Thread-safe: stateless after initialization.
    
    Time Complexity: O(sequence_length) per example
    Space Complexity: O(sequence_length) per example
    """
    
    def __init__(
        self,
        template: PromptTemplate,
        tokenizer: TokenizerWrapper,
        length_manager: Optional[Any] = None,
        max_length: Optional[int] = None,
        per_column_limits: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize prompt engine.
        
        Args:
            template: Prompt template configuration
            tokenizer: Tokenizer wrapper
            length_manager: Optional LengthManager for SOTA preprocessing
            max_length: Optional max sequence length (creates default length manager)
            per_column_limits: Optional per-column character limits
        """
        self._template = template
        self._tokenizer = tokenizer
        self._length_manager = length_manager
        self._max_length = max_length or tokenizer.config.max_length
        self._per_column_limits = per_column_limits or {}
        self._jinja_template: Optional[Any] = None
        
        # Compile Jinja template if custom format
        if template.format_type == "custom" and template.template:
            self._jinja_template = self._compile_template(template.template)
    
    def _compile_template(self, template_str: str) -> Any:
        """
        Compile Jinja2 template.
        
        Falls back to simple string formatting if Jinja2 not available.
        """
        try:
            from jinja2 import Template, StrictUndefined
            return Template(template_str, undefined=StrictUndefined)
        except ImportError:
            # Fallback: use simple format
            return None
    
    def _preprocess_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess text columns before tokenization.
        
        Handles variable-length columns with intelligent truncation.
        Uses LengthManager if available, otherwise applies per-column limits.
        """
        # Use LengthManager if provided
        if self._length_manager is not None:
            text_columns = list(self._template.input_columns)
            if self._template.label_column:
                text_columns.append(self._template.label_column)
            return self._length_manager.preprocess_example(example, text_columns)
        
        # Fallback: Apply per-column limits manually
        if not self._per_column_limits:
            return example
        
        result = dict(example)
        for col, max_chars in self._per_column_limits.items():
            if col in result and isinstance(result[col], str):
                text = result[col]
                if len(text) > max_chars:
                    # Smart truncate at word boundary
                    truncated = text[:max_chars]
                    last_space = truncated.rfind(' ')
                    if last_space > max_chars * 0.5:
                        truncated = truncated[:last_space]
                    result[col] = truncated.rstrip()
        
        return result
    
    def process(
        self,
        example: Dict[str, Any],
    ) -> Result[ProcessedExample, TokenizationError]:
        """
        Process a single example through the template.
        
        Applies SOTA preprocessing for variable-length columns,
        then tokenizes with proper label masking.
        
        Args:
            example: Input example dict with column values
            
        Returns:
            Result containing ProcessedExample or error
        """
        try:
            # SOTA preprocessing for variable-sized columns
            preprocessed = self._preprocess_example(example)
            
            if self._template.format_type == "chat":
                return self._process_chat(preprocessed)
            elif self._template.format_type == "completion":
                return self._process_completion(preprocessed)
            else:  # custom
                return self._process_custom(preprocessed)
        except Exception as e:
            return Err(TokenizationError(
                message=f"Failed to process example: {e}",
                cause=e,
            ))
    
    def _process_chat(
        self,
        example: Dict[str, Any],
    ) -> Result[ProcessedExample, TokenizationError]:
        """
        Process example using chat template format.
        
        Constructs messages list and uses tokenizer's chat template.
        """
        messages = []
        
        # Add system message if present
        if self._template.system_message:
            messages.append({
                "role": "system",
                "content": self._template.system_message,
            })
        
        # Build user message from input columns
        user_content_parts = []
        for col in self._template.input_columns:
            if col in example and example[col]:
                user_content_parts.append(str(example[col]))
        
        user_content = "\n".join(user_content_parts)
        if user_content:
            messages.append({
                "role": "user",
                "content": user_content,
            })
        
        # Add assistant response if label column present
        assistant_content = ""
        if self._template.label_column and self._template.label_column in example:
            assistant_content = str(example[self._template.label_column])
            messages.append({
                "role": "assistant",
                "content": assistant_content,
            })
        
        # Use tokenizer's chat template
        tokenizer = self._tokenizer.tokenizer
        
        try:
            # Encode without assistant for input length calculation
            if hasattr(tokenizer, "apply_chat_template"):
                # Get input tokens (without assistant response)
                input_messages = messages[:-1] if assistant_content else messages
                input_text = tokenizer.apply_chat_template(
                    input_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                input_tokens = tokenizer.encode(input_text, add_special_tokens=False)
                input_length = len(input_tokens)
                
                # Get full sequence
                full_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                
                encoded = self._tokenizer.encode(full_text)
            else:
                # Fallback: simple concatenation
                return self._process_custom_fallback(example, messages)
            
        except Exception as e:
            return Err(TokenizationError(
                message=f"Chat template failed: {e}",
                cause=e,
            ))
        
        return self._build_output(encoded, input_length)
    
    def _process_completion(
        self,
        example: Dict[str, Any],
    ) -> Result[ProcessedExample, TokenizationError]:
        """
        Process example as completion format.
        
        Simple prompt + completion concatenation.
        """
        # Build prompt from input columns
        prompt_parts = []
        for col in self._template.input_columns:
            if col in example and example[col]:
                prompt_parts.append(str(example[col]))
        
        prompt = "\n".join(prompt_parts)
        
        # Get completion
        completion = ""
        if self._template.label_column and self._template.label_column in example:
            completion = str(example[self._template.label_column])
        
        # Encode prompt to get input length
        prompt_encoded = self._tokenizer.encode(prompt, padding="do_not_pad")
        input_length = len(prompt_encoded["input_ids"])
        
        # Encode full sequence
        full_text = prompt + completion
        if self._template.add_eos:
            full_text += self._tokenizer.tokenizer.eos_token or ""
        
        encoded = self._tokenizer.encode(full_text)
        
        return self._build_output(encoded, input_length)
    
    def _process_custom(
        self,
        example: Dict[str, Any],
    ) -> Result[ProcessedExample, TokenizationError]:
        """
        Process example using custom Jinja2 template.
        """
        if not self._template.template:
            return Err(TemplateRenderError(
                message="No template defined for custom format",
            ))
        
        # Prepare template variables
        variables = dict(example)
        variables["eos_token"] = self._tokenizer.tokenizer.eos_token or ""
        variables["bos_token"] = self._tokenizer.tokenizer.bos_token or ""
        variables["pad_token"] = self._tokenizer.tokenizer.pad_token or ""
        
        # Render template
        try:
            if self._jinja_template is not None:
                full_text = self._jinja_template.render(**variables)
            else:
                # Simple string formatting fallback
                full_text = self._render_simple(self._template.template, variables)
        except Exception as e:
            missing = self._find_missing_vars(self._template.template, variables)
            return Err(TemplateRenderError(
                message=f"Template render failed: {e}",
                template=self._template.template[:100],
                missing_vars=tuple(missing),
                cause=e,
            ))
        
        # Calculate input length (text before label)
        input_length = self._calculate_input_length(example, full_text)
        
        # Add BOS/EOS if configured
        if self._template.add_bos:
            bos = self._tokenizer.tokenizer.bos_token or ""
            if bos and not full_text.startswith(bos):
                full_text = bos + full_text
        
        if self._template.add_eos:
            eos = self._tokenizer.tokenizer.eos_token or ""
            if eos and not full_text.endswith(eos):
                full_text = full_text + eos
        
        # Encode
        encoded = self._tokenizer.encode(full_text)
        
        return self._build_output(encoded, input_length)
    
    def _render_simple(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Render template using simple {variable} substitution.
        
        Jinja2 fallback for environments without Jinja2 installed.
        """
        result = template
        for match in SIMPLE_VAR_PATTERN.finditer(template):
            var_name = match.group(1)
            if var_name in variables:
                value = variables[var_name]
                if value is not None:
                    result = result.replace(match.group(0), str(value))
                else:
                    result = result.replace(match.group(0), "")
        return result
    
    def _find_missing_vars(
        self, 
        template: str, 
        variables: Dict[str, Any]
    ) -> List[str]:
        """Find variables in template that are missing from provided variables."""
        missing = []
        for match in SIMPLE_VAR_PATTERN.finditer(template):
            var_name = match.group(1)
            if var_name not in variables:
                missing.append(var_name)
        return missing
    
    def _calculate_input_length(
        self,
        example: Dict[str, Any],
        full_text: str,
    ) -> int:
        """
        Calculate input token length (portion to mask in labels).
        
        Uses template structure to determine where input ends.
        """
        if not self._template.mask_input:
            return 0
        
        if not self._template.label_column:
            return 0
        
        label_value = example.get(self._template.label_column, "")
        if not label_value:
            return 0
        
        label_str = str(label_value)
        
        # Find where label starts in full text
        label_pos = full_text.rfind(label_str)
        if label_pos == -1:
            # Label not found in output, don't mask anything
            return 0
        
        # Get input text (before label)
        input_text = full_text[:label_pos]
        
        # Tokenize input to get length
        input_encoded = self._tokenizer.encode(input_text, padding="do_not_pad")
        return len(input_encoded["input_ids"])
    
    def _build_output(
        self,
        encoded: Dict[str, List[int]],
        input_length: int,
    ) -> Result[ProcessedExample, TokenizationError]:
        """
        Build ProcessedExample with proper label masking.
        
        Labels for input positions are set to -100 (ignored by loss).
        """
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        
        # Build labels
        if self._template.mask_input and input_length > 0:
            # Mask input portion with -100
            labels = [LABEL_PAD_TOKEN_ID] * input_length + input_ids[input_length:]
        else:
            # Use all tokens as labels
            labels = list(input_ids)
        
        # Mask padding in labels
        for i, mask in enumerate(attention_mask):
            if mask == 0:
                labels[i] = LABEL_PAD_TOKEN_ID
        
        return Ok(ProcessedExample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            input_length=input_length,
        ))
    
    def _process_custom_fallback(
        self,
        example: Dict[str, Any],
        messages: List[Dict[str, str]],
    ) -> Result[ProcessedExample, TokenizationError]:
        """
        Fallback for chat processing when apply_chat_template not available.
        """
        # Simple concatenation of messages
        parts = []
        input_parts = []
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
            
            # All but last message are input
            if i < len(messages) - 1:
                input_parts.append(f"<|{role}|>\n{content}")
        
        full_text = "\n".join(parts)
        input_text = "\n".join(input_parts)
        
        # Encode to get lengths
        input_encoded = self._tokenizer.encode(input_text, padding="do_not_pad")
        input_length = len(input_encoded["input_ids"])
        
        encoded = self._tokenizer.encode(full_text)
        
        return self._build_output(encoded, input_length)
    
    def process_batch(
        self,
        examples: List[Dict[str, Any]],
    ) -> List[Result[ProcessedExample, TokenizationError]]:
        """
        Process batch of examples.
        
        Note: Does not parallelize (Python GIL limitation).
        Use DataLoader workers for parallelization.
        
        Args:
            examples: List of example dicts
            
        Returns:
            List of results for each example
        """
        return [self.process(ex) for ex in examples]


# ─────────────────────────────────────────────────────────────────────────────────
# Module-level Functions
# ─────────────────────────────────────────────────────────────────────────────────

def render_template(
    template: str,
    variables: Dict[str, Any],
    use_jinja: bool = True,
) -> Result[str, TemplateRenderError]:
    """
    Render a template string with variables.
    
    Args:
        template: Template string (Jinja2 or simple {var} format)
        variables: Variable values
        use_jinja: Try Jinja2 first (fallback to simple)
        
    Returns:
        Result containing rendered string or error
    """
    try:
        if use_jinja:
            try:
                from jinja2 import Template, StrictUndefined
                jinja_tmpl = Template(template, undefined=StrictUndefined)
                return Ok(jinja_tmpl.render(**variables))
            except ImportError:
                pass
        
        # Simple fallback
        result = template
        for match in SIMPLE_VAR_PATTERN.finditer(template):
            var_name = match.group(1)
            if var_name in variables:
                value = variables[var_name]
                result = result.replace(match.group(0), str(value) if value else "")
        return Ok(result)
        
    except Exception as e:
        return Err(TemplateRenderError(
            message=f"Render failed: {e}",
            template=template[:100],
            cause=e,
        ))
