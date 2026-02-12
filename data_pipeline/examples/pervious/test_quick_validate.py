#!/usr/bin/env python3
"""Quick validation of all SOTA trainer components on CPU."""

import torch
import torch.nn as nn
import sys
import os

# Ensure path is correct
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def main():
    print("=" * 60)
    print("  SOTA TRAINER - QUICK VALIDATION")
    print("=" * 60)
    
    # Import all main components
    from data_pipeline.trainer import (
        # Optimizers
        AdamW, LAMB, AdaFactor, clip_grad_norm_,
        # Schedulers
        CosineScheduler, LinearScheduler, OneCycleScheduler,
        # Loss
        CrossEntropyLoss, FocalLoss, KLDivergenceLoss,
        # Trainer
        Trainer, TrainingArguments,
        # Distributed
        DeviceManager, is_main_process,
    )
    from data_pipeline.trainer.metrics import Accuracy, F1Score, Perplexity, MetricCollection
    from data_pipeline.trainer.distributed import ColumnParallelLinear, MoELayer, TopKGating
    from data_pipeline.trainer.kernels import flash_attention, compile_model
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Optimizer
    print("\n[1] Testing AdamW Optimizer...")
    tests_total += 1
    try:
        model = nn.Linear(64, 10)
        opt = AdamW(model.parameters(), lr=1e-3)
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        print("    ✓ AdamW step successful")
        tests_passed += 1
    except Exception as e:
        print(f"    ✗ AdamW failed: {e}")
    
    # Test 2: LAMB Optimizer
    print("\n[2] Testing LAMB Optimizer...")
    tests_total += 1
    try:
        model = nn.Linear(64, 10)
        opt = LAMB(model.parameters(), lr=1e-3)
        loss = model(torch.randn(4, 64)).sum()
        loss.backward()
        opt.step()
        print("    ✓ LAMB step successful")
        tests_passed += 1
    except Exception as e:
        print(f"    ✗ LAMB failed: {e}")
    
    # Test 3: AdaFactor Optimizer
    print("\n[3] Testing AdaFactor Optimizer...")
    tests_total += 1
    try:
        model = nn.Linear(64, 10)
        opt = AdaFactor(model.parameters(), lr=1e-3)
        loss = model(torch.randn(4, 64)).sum()
        loss.backward()
        opt.step()
        print("    ✓ AdaFactor step successful")
        tests_passed += 1
    except Exception as e:
        print(f"    ✗ AdaFactor failed: {e}")
    
    # Test 4: Scheduler  
    print("\n[4] Testing CosineScheduler...")
    tests_total += 1
    try:
        model = nn.Linear(64, 10)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = CosineScheduler(opt, num_training_steps=100, warmup_steps=10)
        lrs = []
        for i in range(100):
            sched.step()
            if i % 25 == 0:
                lrs.append(sched.get_last_lr()[0])
        print(f"    ✓ LR schedule: {[f'{lr:.6f}' for lr in lrs]}")
        tests_passed += 1
    except Exception as e:
        print(f"    ✗ CosineScheduler failed: {e}")
    
    # Test 5: Loss Functions
    print("\n[5] Testing Loss Functions...")
    tests_total += 1
    try:
        logits = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))
        ce = CrossEntropyLoss(label_smoothing=0.1)
        focal = FocalLoss(gamma=2.0)
        ce_loss = ce(logits, targets).item()
        focal_loss = focal(logits, targets).item()
        print(f"    ✓ CE loss: {ce_loss:.4f}")
        print(f"    ✓ Focal loss: {focal_loss:.4f}")
        tests_passed += 1
    except Exception as e:
        print(f"    ✗ Loss functions failed: {e}")
    
    # Test 6: Metrics
    print("\n[6] Testing Metrics...")
    tests_total += 1
    try:
        preds = torch.randn(100, 5)
        targets = torch.randint(0, 5, (100,))
        collection = MetricCollection([Accuracy(), F1Score(average="macro")])
        collection.update(preds, targets)
        results = collection.compute()
        print(f"    ✓ Accuracy: {results['accuracy']:.2%}, F1: {results['f1']:.3f}")
        tests_passed += 1
    except Exception as e:
        print(f"    ✗ Metrics failed: {e}")
    
    # Test 7: Tensor Parallel
    print("\n[7] Testing Tensor Parallel Layer...")
    tests_total += 1
    try:
        col_lin = ColumnParallelLinear(64, 128)
        x = torch.randn(4, 64)
        y = col_lin(x)
        print(f"    ✓ ColumnParallelLinear: {tuple(x.shape)} -> {tuple(y.shape)}")
        tests_passed += 1
    except Exception as e:
        print(f"    ✗ Tensor Parallel failed: {e}")
    
    # Test 8: MoE Layer
    print("\n[8] Testing MoE Layer...")
    tests_total += 1
    try:
        moe = MoELayer(hidden_size=64, intermediate_size=256, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 64)
        out, aux_loss = moe(x)
        print(f"    ✓ MoE output: {tuple(out.shape)}, aux_loss: {aux_loss.item():.4f}")
        tests_passed += 1
    except Exception as e:
        print(f"    ✗ MoE Layer failed: {e}")
    
    # Test 9: Flash Attention (CPU fallback)
    print("\n[9] Testing Flash Attention (naive fallback)...")
    tests_total += 1
    try:
        q = torch.randn(2, 4, 16, 32)
        k, v = torch.randn_like(q), torch.randn_like(q)
        out = flash_attention(q, k, v, causal=True)
        print(f"    ✓ Flash attn output: {tuple(out.shape)}")
        tests_passed += 1
    except Exception as e:
        print(f"    ✗ Flash Attention failed: {e}")
    
    # Test 10: DeviceManager
    print("\n[10] Testing DeviceManager...")
    tests_total += 1
    try:
        dm = DeviceManager.auto_detect()
        print(f"    ✓ Detected device: {dm.device}")
        print(f"    ✓ Is main process: {is_main_process()}")
        tests_passed += 1
    except Exception as e:
        print(f"    ✗ DeviceManager failed: {e}")
    
    # Test 11: Gradient Clipping
    print("\n[11] Testing Gradient Clipping...")
    tests_total += 1
    try:
        model = nn.Linear(64, 10)
        loss = model(torch.randn(4, 64)).sum()
        loss.backward()
        norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        print(f"    ✓ Grad norm: {norm_before.item():.4f} -> {norm_after.item():.4f}")
        tests_passed += 1
    except Exception as e:
        print(f"    ✗ Gradient Clipping failed: {e}")
    
    # Test 12: torch.compile wrapper
    print("\n[12] Testing torch.compile wrapper...")
    tests_total += 1
    try:
        model = nn.Linear(64, 10)
        compiled = compile_model(model, mode="default")
        out = compiled(torch.randn(4, 64))
        print(f"    ✓ Compiled model output: {tuple(out.shape)}")
        tests_passed += 1
    except Exception as e:
        print(f"    ✗ torch.compile failed: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    pct = tests_passed / tests_total * 100
    if tests_passed == tests_total:
        print(f"  ✓ ALL {tests_passed}/{tests_total} TESTS PASSED ({pct:.0f}%)")
        print("  Ready for GPU testing on A100/B200/AMD!")
    else:
        print(f"  {tests_passed}/{tests_total} TESTS PASSED ({pct:.0f}%)")
    print("=" * 60)
    
    return 0 if tests_passed == tests_total else 1

if __name__ == "__main__":
    sys.exit(main())
