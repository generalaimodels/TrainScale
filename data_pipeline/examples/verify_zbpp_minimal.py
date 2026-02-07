
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

print("Importing zbpp...")
try:
    from data_pipeline.trainer.distributed.zbpp import ZeroBubblePipeline, ZBPPOptimizer, create_zbpp_pipeline
    print("✅ ZBPP imports successful")
except ImportError as e:
    print(f"❌ ZBPP import failed: {e}")
    sys.exit(1)

print("Checking classes...")
if not isinstance(ZeroBubblePipeline, type):
    print("❌ ZeroBubblePipeline is not a class")
else:
    print("✅ ZeroBubblePipeline is a class")

if not isinstance(ZBPPOptimizer, type):
    print("❌ ZBPPOptimizer is not a class")
else:
    print("✅ ZBPPOptimizer is a class")

print("Checking create_zbpp_pipeline...")
if not callable(create_zbpp_pipeline):
     print("❌ create_zbpp_pipeline is not callable")
else:
     print("✅ create_zbpp_pipeline is callable")

print("Minimal verification passed.")
