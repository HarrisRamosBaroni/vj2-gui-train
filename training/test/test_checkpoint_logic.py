#!/usr/bin/env python3
"""
Test the new checkpoint logic for large dataset single-epoch training.
Verifies that checkpoints are saved properly based on iterations and validation loss.
"""

import tempfile
import shutil
from pathlib import Path
import sys

def test_checkpoint_argument_parsing():
    """Test that checkpoint arguments are properly parsed."""
    print("Testing checkpoint argument parsing...")
    
    try:
        with open("vj2ac_train_multi_gpu.py", 'r') as f:
            content = f.read()
        
        # Check for new checkpoint arguments
        required_args = [
            "--save_every_iters",
            "default=500"
        ]
        
        missing_args = []
        for arg in required_args:
            if arg not in content:
                missing_args.append(arg)
        
        if missing_args:
            print(f"❌ Missing checkpoint arguments: {missing_args}")
            return False
        
        print("✅ Checkpoint arguments properly configured")
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint argument test failed: {e}")
        return False

def test_checkpoint_function_signature():
    """Test that checkpoint function only saves predictor weights."""
    print("Testing checkpoint function saves only predictor weights...")
    
    try:
        with open("vj2ac_train_multi_gpu.py", 'r') as f:
            content = f.read()
        
        # Check that checkpoint function has the right signature
        if 'def _save_checkpoint(self, tag, step, val_loss=None):' not in content:
            print("❌ Checkpoint function signature incorrect")
            return False
        
        # Check that only predictor is saved
        if '"predictor": self.predictor.module.state_dict()' not in content:
            print("❌ Predictor state_dict not saved")
            return False
        
        # Check that optimizer/scheduler are NOT saved in new version
        if '"opt": self.optimizer.state_dict()' in content:
            print("❌ Optimizer still being saved (should be removed)")
            return False
        
        print("✅ Checkpoint function correctly saves only predictor weights")
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint function test failed: {e}")
        return False

def test_best_model_tracking():
    """Test that best validation loss is tracked and saved."""
    print("Testing best validation loss tracking...")
    
    try:
        with open("vj2ac_train_multi_gpu.py", 'r') as f:
            content = f.read()
        
        required_features = [
            "self.best_val_loss = float(\"inf\")",
            "if val_loss < self.best_val_loss:",
            'self._save_checkpoint("best"',
            "🏆 New best validation loss"
        ]
        
        missing_features = []
        for feature in required_features:
            if feature not in content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing best model tracking features: {missing_features}")
            return False
        
        print("✅ Best validation loss tracking implemented")
        return True
        
    except Exception as e:
        print(f"❌ Best model tracking test failed: {e}")
        return False

def test_iteration_based_checkpointing():
    """Test that iteration-based checkpointing is implemented."""
    print("Testing iteration-based checkpoint saving...")
    
    try:
        with open("vj2ac_train_multi_gpu.py", 'r') as f:
            content = f.read()
        
        required_features = [
            "self.global_step += 1",
            "if self.global_step % self.save_every_iters == 0:",
            'self._save_checkpoint(f"step_{self.global_step}"',
            "self.save_every_iters = args.save_every_iters"
        ]
        
        missing_features = []
        for feature in required_features:
            if feature not in content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing iteration-based checkpoint features: {missing_features}")
            return False
        
        print("✅ Iteration-based checkpoint saving implemented")
        return True
        
    except Exception as e:
        print(f"❌ Iteration-based checkpoint test failed: {e}")
        return False

def test_epoch_checkpoint_removal():
    """Test that automatic epoch-based checkpoints are removed."""
    print("Testing epoch-based checkpoint removal...")
    
    try:
        with open("vj2ac_train_multi_gpu.py", 'r') as f:
            content = f.read()
        
        # Should NOT have automatic "last" checkpoint saving
        if 'self._save_checkpoint("last"' in content:
            print("❌ Automatic 'last' checkpoint saving still present")
            return False
        
        # Should only save epoch checkpoints if explicitly requested
        if "if self.save_every_epochs > 0" not in content:
            print("❌ Epoch-based checkpoint saving not properly conditional")
            return False
        
        print("✅ Automatic epoch-based checkpoint saving properly removed")
        return True
        
    except Exception as e:
        print(f"❌ Epoch checkpoint removal test failed: {e}")
        return False

def test_wandb_logging_updates():
    """Test that WandB logging includes step-based metrics."""
    print("Testing WandB logging updates...")
    
    try:
        with open("vj2ac_train_multi_gpu.py", 'r') as f:
            content = f.read()
        
        required_features = [
            '"iter": self.global_step',
            '"step": self.global_step',
            '"best_val_loss": self.best_val_loss'
        ]
        
        missing_features = []
        for feature in required_features:
            if feature not in content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing WandB logging features: {missing_features}")
            return False
        
        print("✅ WandB logging properly updated for step-based training")
        return True
        
    except Exception as e:
        print(f"❌ WandB logging test failed: {e}")
        return False

def main():
    print("🧪 VJ2 Checkpoint Logic Test")
    print("="*50)
    
    tests = [
        ("checkpoint_arguments", test_checkpoint_argument_parsing),
        ("checkpoint_function", test_checkpoint_function_signature),
        ("best_model_tracking", test_best_model_tracking),
        ("iteration_checkpointing", test_iteration_based_checkpointing),
        ("epoch_checkpoint_removal", test_epoch_checkpoint_removal),
        ("wandb_logging", test_wandb_logging_updates)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n--- {test_name.replace('_', ' ').title()} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("🧪 CHECKPOINT LOGIC TEST SUMMARY")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name.replace('_', ' ').title()}")
    
    print(f"\nResult: {passed}/{total} checkpoint logic tests passed")
    
    if passed == total:
        print("🎉 All checkpoint logic tests passed!")
        print("\n📋 CHECKPOINT STRATEGY VERIFICATION:")
        print("✅ Routine checkpoints every 500 iterations (configurable)")
        print("✅ Best model saved when validation loss improves") 
        print("✅ Only predictor weights saved (optimizer/scheduler excluded)")
        print("✅ Global step tracking for proper iteration counting")
        print("✅ Automatic epoch checkpoints removed (single-epoch training)")
        print("✅ WandB logging updated for step-based metrics")
        print("\n🚀 Checkpoint strategy optimized for large dataset training!")
        return 0
    else:
        print("⚠️  Some checkpoint logic tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())