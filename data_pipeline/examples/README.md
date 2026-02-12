# Examples Test Commands

Run all commands from repository root (`TrainScale/`).

## 1. Activate environment

```bash
source /home/test1/miniconda3/etc/profile.d/conda.sh
conda activate cca_chatbot
```

Optional sanity check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## 2. DDP tests

DDP dry run (4 GPUs):

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 \
  data_pipeline/examples/rocm_sota_demo_ddp.py \
  --config data_pipeline/examples/rocm_sota_config.yaml \
  --dry-run
```

DDP smoke train (max 1 step):

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 \
  data_pipeline/examples/rocm_sota_demo_ddp.py \
  --config data_pipeline/examples/rocm_sota_config.yaml \
  --max-steps 1
```

## 3. FSDP2 test

FSDP2 verify (4 GPUs):

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 \
  data_pipeline/examples/rocm_sota_demo_fsdp2.py \
  --config data_pipeline/examples/rocm_sota_config_fsdp2.yaml \
  --verify
```

## 4. ZBPP test

ZBPP verify (4 GPUs):

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 \
  data_pipeline/examples/zbpp_sota_demo.py \
  --verify \
  --config data_pipeline/examples/zbpp_sota_config.yaml
```

## 5. Optional: save logs per mode

```bash
mkdir -p /tmp/trainscale_example_logs
```

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 \
  data_pipeline/examples/rocm_sota_demo_ddp.py \
  --config data_pipeline/examples/rocm_sota_config.yaml \
  --max-steps 1 \
  > /tmp/trainscale_example_logs/ddp.log 2>&1
```

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 \
  data_pipeline/examples/rocm_sota_demo_fsdp2.py \
  --config data_pipeline/examples/rocm_sota_config_fsdp2.yaml \
  --verify \
  > /tmp/trainscale_example_logs/fsdp2.log 2>&1
```

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 \
  data_pipeline/examples/zbpp_sota_demo.py \
  --verify \
  --config data_pipeline/examples/zbpp_sota_config.yaml \
  > /tmp/trainscale_example_logs/zbpp.log 2>&1
```
