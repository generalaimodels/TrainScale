
import os
import torch
from datasets import Dataset
from data_pipeline.pipeline import DataPipeline, PipelineConfig
from data_pipeline.core.config_schema import (
    DatasetConfig, TokenizerConfig, PromptEngineConfig, 
    TrainingStage, FineTuningConfig, FineTuningFormat, 
    TruncationStrategy
)

# Mock dataset
data = {
    "instruction": ["Summarize this document: " * 5],
    "input": ["This is a very long document " * 100],
    "output": ["Configuration refactoring is complete."]
}
ds = Dataset.from_dict(data)

# Save mock dataset to disk for loading (or we can inject it if pipeline supported it, but pipeline expects loading from name)
# For simplicity, we'll just mock the tokenizer and engine directly or use a real one if available.
# Let's try to use minimal logic without external deps if possible, but pipeline needs transformers.
# We'll assume transformers is installed.

def test_pipeline():
    print("Testing Pipeline Configuration...")
    
    # 1. Create Config
    pe_config = PromptEngineConfig(
        stage=TrainingStage.FINE_TUNING,
        max_length=64, # Short length to force truncation
        finetuning=FineTuningConfig(format=FineTuningFormat.INSTRUCTION),
        truncation_strategy=TruncationStrategy.SMART
    )
    
    # We construct PipelineConfig manually
    pipeline_config = PipelineConfig(
        type="data_module",
        version="1.0",
        dataset=DatasetConfig(name="dummy"), # We won't actually load this if we inject
        tokenizer=TokenizerConfig(name_or_path="HuggingFaceTB/SmolLM2-135M"), # Modern SOTA-aligned small model
        prompt_engine=pe_config
    )
    
    # 2. Instantiate Pipeline
    # Note: we need a real tokenizer for this to work.
    try:
        from transformers import AutoTokenizer
        # Use a small, modern model instead of gpt2
        model_id = "HuggingFaceTB/SmolLM2-135M"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Skipping full pipeline test due to missing transformers/gpt2: {e}")
        return

    pipeline = DataPipeline(
        pipeline_config=pipeline_config,
        tokenizer=tokenizer
    )
    
    # 3. Inject dataset manually into cache to bypass loading
    # The pipeline.load_dataset method checks cache first.
    # cache_key is f"{split}_{streaming}" where streaming arg is None by default
    pipeline._datasets["train_None"] = ds
    
    # 4. Get DataLoader
    # Imports for Result handling
    from data_pipeline.core.types import Err, unwrap, unwrap_err

    print("Building DataLoader...")
    # num_workers=0 required for Windows debugging with local functions (collate_fn)
    dl_result = pipeline.get_dataloader("train", batch_size=1, num_workers=0, pin_memory=False)
    
    if isinstance(dl_result, Err):
        print(f"Error: {unwrap_err(dl_result)}")
        return

    dl = unwrap(dl_result)
    
    batch = next(iter(dl))
    input_ids = batch["input_ids"]
    
    print(f"Input shape: {input_ids.shape}")
    assert input_ids.shape[1] <= 64, f"Output exceeded max_length: {input_ids.shape[1]}"
    print("Verification Successful!")

if __name__ == "__main__":
    try:
        test_pipeline()
    except Exception as e:
        print(f"Test failed via exception: {e}")
        import traceback
        traceback.print_exc()
