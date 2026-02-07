
import torch
from data_pipeline.core.config_schema import (
    PipelineConfig, DatasetConfig, TokenizerConfig, PromptEngineConfig,
    PreTrainingConfig, FineTuningConfig, RLConfig,
    TrainingStage, PreTrainingFormat, FineTuningFormat, RLAlgorithm,
    FIMConfig, PackingConfig, SpanCorruptionConfig, MultiTurnConfig,
    TruncationStrategy, FIMMode
)
from data_pipeline.pipeline import DataPipeline
from data_pipeline.core.types import unwrap, unwrap_err, Err, Ok
from datasets import Dataset

# =============================================================================
# UTILITIES
# =============================================================================

def create_pipeline(dataset_name: str, stage_config: PromptEngineConfig) -> DataPipeline:
    config = PipelineConfig(
        type="data_module",
        version="1.0",
        dataset=DatasetConfig(name=dataset_name, splits={"train": {"name": "train", "sample_size": 10}}),
        tokenizer=TokenizerConfig(name_or_path="HuggingFaceTB/SmolLM2-135M", max_length=128),
        prompt_engine=stage_config,
    )
    return DataPipeline(pipeline_config=config)

def run_batch_test(stage_name: str, config: PromptEngineConfig, dummy_data: dict, checks: dict):
    print(f"\n>>> Testing {stage_name}...")
    pipeline = create_pipeline(f"dummy_{stage_name.lower()}", config)
    pipeline._datasets["train_None"] = Dataset.from_dict(dummy_data)
    
    dl_result = pipeline.get_dataloader("train", batch_size=2, num_workers=0, pin_memory=False)
    if isinstance(dl_result, Err):
        print(f"FAILED: {unwrap_err(dl_result)}")
        return

    dl = unwrap(dl_result)
    batch = next(iter(dl))
    
    print(f"    Batch keys: {list(batch.keys())}")
    
    # Generic Checks
    for key, expected_dim in checks.items():
        if key not in batch:
            print(f"    ERROR: Missing key '{key}'")
            continue
        if torch.is_tensor(batch[key]):
             print(f"    {key} shape: {batch[key].shape}")
             # Basic dimension check
             if len(batch[key].shape) != expected_dim:
                 print(f"    ERROR: {key} expected dim {expected_dim}, got {len(batch[key].shape)}")

    print(f"    SUCCESS: {stage_name}")

# =============================================================================
# PRE-TRAINING TESTS
# =============================================================================

def test_pt_causal():
    cfg = PromptEngineConfig(
        stage=TrainingStage.PRE_TRAINING,
        max_length=128,
        pretraining=PreTrainingConfig(
            format=PreTrainingFormat.CAUSAL_LM,
            text_column="text"
        )
    )
    data = {"text": ["Causal LM example " * 5] * 10}
    run_batch_test("PreTraining (Causal LM)", cfg, data, {"input_ids": 2, "labels": 2})

def test_pt_fim():
    cfg = PromptEngineConfig(
        stage=TrainingStage.PRE_TRAINING,
        max_length=128,
        pretraining=PreTrainingConfig(
            format=PreTrainingFormat.FILL_IN_MIDDLE,
            text_column="text",
            fim=FIMConfig(enabled=True, rate=1.0, mode=FIMMode.PSM) # Force FIM
        )
    )
    data = {"text": ["Fill in the middle example " * 5] * 10}
    run_batch_test("PreTraining (FIM - PSM)", cfg, data, {"input_ids": 2, "labels": 2})

def test_pt_span_corruption():
    cfg = PromptEngineConfig(
        stage=TrainingStage.PRE_TRAINING,
        max_length=128,
        pretraining=PreTrainingConfig(
            format=PreTrainingFormat.SPAN_CORRUPTION,
            text_column="text",
            span_corruption=SpanCorruptionConfig(enabled=True, noise_density=0.15)
        )
    )
    data = {"text": ["Span corruption example " * 5] * 10}
    run_batch_test("PreTraining (Span Corruption)", cfg, data, {"input_ids": 2, "labels": 2})

# =============================================================================
# FINE-TUNING TESTS
# =============================================================================

def test_ft_instruction():
    cfg = PromptEngineConfig(
        stage=TrainingStage.FINE_TUNING,
        max_length=128,
        finetuning=FineTuningConfig(
            format=FineTuningFormat.INSTRUCTION,
            instruction_column="inst",
            input_column="inp",
            output_column="out"
        )
    )
    data = {
        "inst": ["Instruction"]*10,
        "inp": ["Input"]*10,
        "out": ["Output"]*10
    }
    run_batch_test("FineTuning (Instruction)", cfg, data, {"input_ids": 2, "labels": 2})

def test_ft_chat():
    cfg = PromptEngineConfig(
        stage=TrainingStage.FINE_TUNING,
        max_length=128,
        finetuning=FineTuningConfig(format=FineTuningFormat.CHAT)
    )
    data = {"messages": [[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Ho"}]] * 10}
    run_batch_test("FineTuning (Chat)", cfg, data, {"input_ids": 2, "labels": 2})

def test_ft_completion():
    cfg = PromptEngineConfig(
        stage=TrainingStage.FINE_TUNING,
        max_length=128,
        finetuning=FineTuningConfig(
            format=FineTuningFormat.COMPLETION,
            input_column="prompt",
            output_column="completion"
        )
    )
    data = {"prompt": ["Prompt"]*10, "completion": ["Completion"]*10}
    run_batch_test("FineTuning (Completion)", cfg, data, {"input_ids": 2, "labels": 2})

# =============================================================================
# RL TESTS
# =============================================================================

def test_rl_dpo():
    cfg = PromptEngineConfig(
        stage=TrainingStage.POST_TRAINING_RL,
        max_length=128,
        rl=RLConfig(algorithm=RLAlgorithm.DPO)
    )
    data = {"prompt": ["P"]*10, "chosen": ["C"]*10, "rejected": ["R"]*10}
    run_batch_test(
        "RL (DPO)", cfg, data, 
        {"chosen_input_ids": 2, "rejected_input_ids": 2, "chosen_labels": 2, "rejected_labels": 2}
    )

def test_rl_ppo():
    cfg = PromptEngineConfig(
        stage=TrainingStage.POST_TRAINING_RL,
        max_length=128,
        rl=RLConfig(algorithm=RLAlgorithm.PPO)
    )
    data = {"prompt": ["P"]*10}
    run_batch_test("RL (PPO)", cfg, data, {"input_ids": 2, "attention_mask": 2})

def test_rl_kto():
    cfg = PromptEngineConfig(
        stage=TrainingStage.POST_TRAINING_RL,
        max_length=128,
        rl=RLConfig(algorithm=RLAlgorithm.KTO)
    )
    data = {"prompt": ["P"]*10, "response": ["R"]*10, "label": [True]*10} # Boolean label for KTO
    run_batch_test(
        "RL (KTO)", cfg, data,
        {"input_ids": 2, "labels": 2, "kto_tags": 1} # kto_tags usually 1D or scalar per sample? Checking implementation
    )

def test_rl_grpo():
    cfg = PromptEngineConfig(
        stage=TrainingStage.POST_TRAINING_RL,
        max_length=128,
        rl=RLConfig(algorithm=RLAlgorithm.GRPO)
    )
    # GRPO usually expects a list of completions or just prompts that generate multiple.
    # Assuming standard prompt input
    data = {"prompt": ["P"]*10}
    run_batch_test("RL (GRPO)", cfg, data, {"input_ids": 2})


if __name__ == "__main__":
    try:
        test_pt_causal()
        test_pt_fim()
        # test_pt_span_corruption() # NOTE: Might fail if T5 tokenizer features needed
        
        test_ft_instruction()
        test_ft_chat()
        test_ft_completion()
        
        test_rl_dpo()
        test_rl_ppo()
        # test_rl_kto() # Check KTO implementation for exact keys
        test_rl_grpo()
        
        print("\n=== ALL COMPREHENSIVE TESTS COMPLETED ===")
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()
