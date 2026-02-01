"""
Example: Quick Start with Data Pipeline
========================================
Demonstrates using the pipeline with a pre-loaded tokenizer.
"""

from transformers import AutoTokenizer
from data_pipeline import (
    DataPipeline,
    create_pipeline,
    wrap_tokenizer,
    PromptTemplate,
    DatasetConfig,
    DataLoaderConfig,
    unwrap,
    is_ok,
)


def example_with_pre_loaded_tokenizer():
    """
    Example using a pre-loaded tokenizer.
    
    This is the recommended approach when you already have a tokenizer
    loaded (e.g., from a model you're training).
    """
    # Load your tokenizer however you want
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Create pipeline with pre-loaded tokenizer
    pipeline = create_pipeline(
        dataset_name="tatsu-lab/alpaca",
        tokenizer=tokenizer,
        input_columns=["instruction", "input"],
        label_column="output",
        max_length=512,
        batch_size=4,
        streaming=False,
    )
    
    # Get DataLoader
    result = pipeline.get_dataloader("train", shuffle=True)
    if is_ok(result):
        dataloader = unwrap(result)
        
        # Iterate
        for batch in dataloader:
            print(f"input_ids shape: {batch['input_ids'].shape}")
            print(f"labels shape: {batch['labels'].shape}")
            # Labels properly masked with -100 for input positions
            print(f"Labels sample (first 10): {batch['labels'][0, :10].tolist()}")
            break


def example_with_yaml_config():
    """
    Example loading pipeline from YAML configuration.
    """
    # Load from config file
    result = DataPipeline.from_config("example_config.yaml")
    
    if is_ok(result):
        pipeline = unwrap(result)
        
        # Discover dataset structure
        metadata_result = pipeline.discover()
        if is_ok(metadata_result):
            metadata = unwrap(metadata_result)
            print(f"Dataset: {metadata.dataset_id}")
            print(f"Configs: {[c.name for c in metadata.configs]}")
        
        # Get DataLoader
        loader_result = pipeline.get_dataloader("train")
        if is_ok(loader_result):
            loader = unwrap(loader_result)
            print(f"Batches: {len(loader)}")


def example_with_custom_template():
    """
    Example with custom Jinja2 template.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Define custom template
    template = PromptTemplate(
        format_type="custom",
        template="""### Instruction:
{{ instruction }}

{% if input %}### Input:
{{ input }}
{% endif %}

### Response:
{{ output }}{{ eos_token }}""",
        input_columns=("instruction", "input"),
        label_column="output",
        mask_input=True,  # Mask input in labels for loss
        add_eos=True,
    )
    
    pipeline = DataPipeline(
        dataset_config=DatasetConfig(name="tatsu-lab/alpaca"),
        tokenizer=tokenizer,
        prompt_template=template,
        dataloader_config=DataLoaderConfig(batch_size=8),
    )
    
    # Get DataLoader
    result = pipeline.get_dataloader("train")
    if is_ok(result):
        loader = unwrap(result)
        print(f"Created DataLoader with {len(loader)} batches")


def example_streaming():
    """
    Example with streaming (memory-efficient for large datasets).
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    pipeline = create_pipeline(
        dataset_name="HuggingFaceH4/ultrachat_200k",
        tokenizer=tokenizer,
        streaming=True,  # Stream! No full download
        batch_size=4,
    )
    
    # Get streaming DataLoader
    result = pipeline.get_dataloader("train")
    if is_ok(result):
        loader = unwrap(result)
        
        # Process first few batches
        for i, batch in enumerate(loader):
            if i >= 3:
                break
            print(f"Batch {i}: shape {batch['input_ids'].shape}")


def example_distributed():
    """
    Example for distributed training.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    pipeline = create_pipeline(
        dataset_name="tatsu-lab/alpaca",
        tokenizer=tokenizer,
        batch_size=4,
    )
    
    # Get distributed DataLoader (for rank 0 of 8 GPUs)
    result = pipeline.get_dataloader(
        "train",
        distributed=True,
        rank=0,
        world_size=8,
    )
    
    if is_ok(result):
        loader = unwrap(result)
        # Each loader only sees 1/8 of the data
        print(f"Rank 0 DataLoader ready")


if __name__ == "__main__":
    print("=" * 60)
    print("Data Pipeline Quick Start Examples")
    print("=" * 60)
    
    print("\n1. Pre-loaded Tokenizer Example")
    example_with_pre_loaded_tokenizer()
    
    print("\n2. YAML Config Example")
    example_with_yaml_config()
    
    print("\n3. Custom Template Example")
    example_with_custom_template()
