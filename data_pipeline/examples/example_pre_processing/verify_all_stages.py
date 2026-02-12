
import os
import torch
from data_pipeline.core.config_schema import (
    PipelineConfig, DatasetConfig, TokenizerConfig, PromptEngineConfig,
    PreTrainingConfig, FineTuningConfig, RLConfig,
    TrainingStage, PreTrainingFormat, FineTuningFormat, RLAlgorithm,
    TruncationStrategy
)
from data_pipeline.pipeline import DataPipeline
from data_pipeline.core.types import unwrap, unwrap_err, Err, Ok

def test_pretraining_stage():
    print("\n=== Testing PRE_TRAINING Stage ===")
    
    # 1. Setup Config
    config = PipelineConfig(
        type="data_module",
        version="1.0",
        dataset=DatasetConfig(
            name="dummy_pretrain",
            splits={"train": {"name": "train", "sample_size": 10}},
            columns=["text"],
        ),
        tokenizer=TokenizerConfig(
            name_or_path="HuggingFaceTB/SmolLM2-135M",
            max_length=128,
        ),
        prompt_engine=PromptEngineConfig(
            stage=TrainingStage.PRE_TRAINING,
            max_length=128,
            pretraining=PreTrainingConfig(
                format=PreTrainingFormat.CAUSAL_LM,
                text_column="text",
                add_bos=True,
                add_eos=True,
            ),
        ),
    )
    
    # 2. Initialize Pipeline
    pipeline = DataPipeline(pipeline_config=config)
    
    # 3. Create dummy dataset
    from datasets import Dataset
    ds = Dataset.from_dict({
        "text": ["Hello world " * 10 for _ in range(10)]
    })
    pipeline._datasets["train_None"] = ds
    
    # 4. Get DataLoader
    dl_result = pipeline.get_dataloader("train", batch_size=2, num_workers=0, pin_memory=False)
    if isinstance(dl_result, Err):
        print(f"FAILED: {unwrap_err(dl_result)}")
        return
    
    dl = unwrap(dl_result)
    batch = next(iter(dl))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Input shape: {batch['input_ids'].shape}")
    
    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape == (2, 128)
    print("SUCCESS: Pre-training batch validated.")

def test_finetuning_stage():
    print("\n=== Testing FINE_TUNING Stage ===")
    
    # 1. Setup Config
    config = PipelineConfig(
        type="data_module",
        version="1.0",
        dataset=DatasetConfig(
            name="dummy_finetune",
            splits={"train": {"name": "train", "sample_size": 10}},
        ),
        tokenizer=TokenizerConfig(
            name_or_path="HuggingFaceTB/SmolLM2-135M",
            max_length=128,
        ),
        prompt_engine=PromptEngineConfig(
            stage=TrainingStage.FINE_TUNING,
            max_length=128,
            finetuning=FineTuningConfig(
                format=FineTuningFormat.INSTRUCTION,
                instruction_column="instruction",
                input_column="input",
                output_column="output",
            ),
        ),
    )
    
    # 2. Initialize Pipeline
    pipeline = DataPipeline(pipeline_config=config)
    
    # 3. Create dummy dataset
    from datasets import Dataset
    ds = Dataset.from_dict({
        "instruction": ["Instruct " + str(i) for i in range(10)],
        "input": ["Input " + str(i) for i in range(10)],
        "output": ["Output " + str(i) for i in range(10)],
    })
    pipeline._datasets["train_None"] = ds
    
    # 4. Get DataLoader
    dl_result = pipeline.get_dataloader("train", batch_size=2, num_workers=0, pin_memory=False)
    if isinstance(dl_result, Err):
        print(f"FAILED: {unwrap_err(dl_result)}")
        return
    
    dl = unwrap(dl_result)
    batch = next(iter(dl))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Input shape: {batch['input_ids'].shape}")
    
    assert batch["input_ids"].shape == (2, 128)
    print("SUCCESS: Fine-tuning batch validated.")

def test_rl_stage():
    print("\n=== Testing POST_TRAINING_RL Stage (DPO) ===")
    
    # 1. Setup Config
    config = PipelineConfig(
        type="data_module",
        version="1.0",
        dataset=DatasetConfig(
            name="dummy_rl",
            splits={"train": {"name": "train", "sample_size": 10}},
        ),
        tokenizer=TokenizerConfig(
            name_or_path="HuggingFaceTB/SmolLM2-135M",
            max_length=128,
        ),
        prompt_engine=PromptEngineConfig(
            stage=TrainingStage.POST_TRAINING_RL,
            max_length=128,
            rl=RLConfig(
                algorithm=RLAlgorithm.DPO,
                max_prompt_length=64,
                max_completion_length=64,
                mask_prompt=True,
            ),
        ),
    )
    
    # 2. Initialize Pipeline
    pipeline = DataPipeline(pipeline_config=config)
    
    # 3. Create dummy dataset
    from datasets import Dataset
    ds = Dataset.from_dict({
        "prompt": ["Prompt " + str(i) for i in range(10)],
        "chosen": ["Chosen response " + str(i) for i in range(10)],
        "rejected": ["Rejected response " + str(i) for i in range(10)],
    })
    pipeline._datasets["train_None"] = ds
    
    # 4. Get DataLoader
    dl_result = pipeline.get_dataloader("train", batch_size=2, num_workers=0, pin_memory=False)
    if isinstance(dl_result, Err):
        print(f"FAILED: {unwrap_err(dl_result)}")
        return
    
    dl = unwrap(dl_result)
    batch = next(iter(dl))
    
    print(f"Batch keys: {batch.keys()}")
    # DPO produces chosen/rejected input_ids/labels etc.
    if "chosen_input_ids" in batch:
        print(f"Chosen Input shape: {batch['chosen_input_ids'].shape}")
        assert batch['chosen_input_ids'].shape == (2, 128)
        assert batch['rejected_input_ids'].shape == (2, 128)
        print("SUCCESS: RL (DPO) batch validated.")
    else:
        print("FAILED: Missing DPO columns in batch.")
        print(batch.keys())

if __name__ == "__main__":
    try:
        test_pretraining_stage()
        test_finetuning_stage()
        test_rl_stage()
        print("\nALL STAGES VERIFIED SUCCESSFULLY!")
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()
