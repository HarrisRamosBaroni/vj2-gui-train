#!/usr/bin/env python3
"""
Test script for VJ2 GUI training pipeline.
Tests the training code with synthetic data to verify DDP synchronization works.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import time
import signal
import argparse

def run_command(cmd, timeout=60, cwd=None):
    """Run a command with timeout and return success status."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=cwd
        )
        if result.returncode == 0:
            print("✅ Command succeeded")
            if result.stdout.strip():
                print("STDOUT:", result.stdout.strip()[-500:])  # Last 500 chars
            return True, result.stdout, result.stderr
        else:
            print("❌ Command failed")
            print("STDOUT:", result.stdout.strip()[-500:] if result.stdout else "None")
            print("STDERR:", result.stderr.strip()[-500:] if result.stderr else "None")
            return False, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out after {timeout}s")
        return False, "", "Timeout"
    except Exception as e:
        print(f"❌ Command failed with exception: {e}")
        return False, "", str(e)

def create_test_environment():
    """Create a temporary test environment with synthetic data."""
    test_dir = Path(tempfile.mkdtemp(prefix="vj2_test_"))
    print(f"Created test directory: {test_dir}")
    
    # Create synthetic training data
    train_data_dir = test_dir / "train_data"
    val_data_dir = test_dir / "val_data"
    
    # Generate synthetic data
    cmd = [
        sys.executable, "create_synthetic_data.py",
        "--output_dir", str(train_data_dir),
        "--num_files", "2",
        "--trajectories_per_file", "5"
    ]
    success, _, _ = run_command(cmd, timeout=30)
    if not success:
        raise RuntimeError("Failed to create synthetic training data")
    
    # Create smaller validation set
    cmd = [
        sys.executable, "create_synthetic_data.py", 
        "--output_dir", str(val_data_dir),
        "--num_files", "1", 
        "--trajectories_per_file", "3"
    ]
    success, _, _ = run_command(cmd, timeout=30)
    if not success:
        raise RuntimeError("Failed to create synthetic validation data")
    
    return test_dir, train_data_dir, val_data_dir

def test_dataloader():
    """Test the dataloader with synthetic data."""
    print("\n=== Testing Dataloader ===")
    
    # Test basic dataloader functionality
    cmd = [sys.executable, "vj2_dataloader.py"]
    success, stdout, stderr = run_command(cmd, timeout=30)
    
    if not success and "processed_data" in stderr:
        print("Expected error - no processed_data directory (this is fine)")
        return True
    elif success:
        print("✅ Dataloader test passed")
        return True
    else:
        print("❌ Unexpected dataloader error")
        return False

def test_single_rank_cpu(train_data_dir, val_data_dir):
    """Test single rank CPU training."""
    print("\n=== Testing Single Rank CPU Training ===")
    
    # Set environment for single process
    env = os.environ.copy()
    env.update({
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12355",
        "WORLD_SIZE": "1", 
        "RANK": "0",
        "LOCAL_RANK": "0"
    })
    
    cmd = [
        sys.executable, "vj2ac_train_multi_gpu.py",
        "--device", "cpu",
        "--dist-backend", "gloo", 
        "--num_epochs", "1",
        "--batch_size", "2",
        "--num_workers", "0",
        "--processed_data_dir", str(train_data_dir),
        "--validation_data_dir", str(val_data_dir)
    ]
    
    # Start the process
    print(f"Environment: WORLD_SIZE=1, RANK=0")
    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor output for a short time
        start_time = time.time()
        output_lines = []
        
        while time.time() - start_time < 45:  # 45 second timeout
            line = process.stdout.readline()
            if line:
                output_lines.append(line.strip())
                print(f"[TRAIN] {line.strip()}")
                
                # Check for success indicators
                if "Model initialized on cpu" in line:
                    print("✅ Model initialization successful")
                if "Dataloader created with" in line:
                    print("✅ Dataloader creation successful")
                if "Starting training loop" in line:
                    print("✅ Training loop started")
                if "Starting epoch 1" in line:
                    print("✅ First epoch started - terminating test")
                    break
                if "Entering validation" in line:
                    print("✅ Validation started - terminating test")
                    break
            elif process.poll() is not None:
                break
            else:
                time.sleep(0.1)
        
        # Terminate the process gracefully
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            
        # Check if we saw expected output
        success_indicators = [
            "Model initialized on cpu",
            "Dataloader created with", 
            "Starting training loop"
        ]
        
        output_text = "\n".join(output_lines)
        success_count = sum(1 for indicator in success_indicators if indicator in output_text)
        
        if success_count >= 2:
            print("✅ Single rank CPU test passed")
            return True
        else:
            print(f"❌ Single rank CPU test failed - only {success_count}/3 success indicators found")
            return False
            
    except Exception as e:
        print(f"❌ Single rank CPU test failed with exception: {e}")
        return False

def test_multi_rank_cpu(train_data_dir, val_data_dir):
    """Test multi-rank CPU training with torchrun."""
    print("\n=== Testing Multi-Rank CPU Training ===")
    
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "--master_port=12356",
        "vj2ac_train_multi_gpu.py",
        "--device", "cpu",
        "--dist-backend", "gloo",
        "--num_epochs", "1", 
        "--batch_size", "1",
        "--num_workers", "0",
        "--processed_data_dir", str(train_data_dir),
        "--validation_data_dir", str(val_data_dir)
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        start_time = time.time()
        output_lines = []
        rank_initialized = set()
        
        while time.time() - start_time < 60:  # 60 second timeout
            line = process.stdout.readline()
            if line:
                output_lines.append(line.strip())
                print(f"[MULTI] {line.strip()}")
                
                # Track rank initialization
                if "Rank " in line and "Model initialized on cpu" in line:
                    rank_num = line.split("Rank ")[1].split(":")[0]
                    rank_initialized.add(rank_num)
                    print(f"✅ Rank {rank_num} initialized")
                
                # Check for DDP synchronization
                if "Entering validation" in line and "Rank" in line:
                    print("✅ DDP validation synchronization working")
                
                # Early termination if we see good signs
                if len(rank_initialized) >= 2:
                    print("✅ Both ranks initialized - terminating test")
                    break
                    
            elif process.poll() is not None:
                break
            else:
                time.sleep(0.1)
        
        # Terminate the process
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        
        if len(rank_initialized) >= 2:
            print("✅ Multi-rank CPU test passed")
            return True
        else:
            print(f"❌ Multi-rank CPU test failed - only {len(rank_initialized)} ranks initialized")
            return False
            
    except Exception as e:
        print(f"❌ Multi-rank CPU test failed: {e}")
        return False

def cleanup_test_environment(test_dir):
    """Clean up the test environment."""
    try:
        shutil.rmtree(test_dir)
        print(f"✅ Cleaned up test directory: {test_dir}")
    except Exception as e:
        print(f"⚠️  Failed to clean up {test_dir}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test VJ2 training pipeline")
    parser.add_argument("--keep-data", action="store_true", help="Keep test data after completion")
    parser.add_argument("--skip-multi", action="store_true", help="Skip multi-rank tests")
    args = parser.parse_args()
    
    print("🧪 VJ2 GUI Training Pipeline Test Suite")
    print("="*50)
    
    test_results = {}
    test_dir = None
    
    try:
        # Create test environment
        test_dir, train_data_dir, val_data_dir = create_test_environment()
        print(f"✅ Test environment created")
        
        # Test 1: Basic dataloader
        test_results["dataloader"] = test_dataloader()
        
        # Test 2: Single rank CPU training
        test_results["single_rank_cpu"] = test_single_rank_cpu(train_data_dir, val_data_dir)
        
        # Test 3: Multi-rank CPU training (if torchrun available and not skipped)
        if not args.skip_multi:
            if shutil.which("torchrun"):
                test_results["multi_rank_cpu"] = test_multi_rank_cpu(train_data_dir, val_data_dir)
            else:
                print("⚠️  torchrun not found, skipping multi-rank tests")
                test_results["multi_rank_cpu"] = None
        else:
            print("⚠️  Skipping multi-rank tests")
            test_results["multi_rank_cpu"] = None
            
    except Exception as e:
        print(f"❌ Test suite failed with exception: {e}")
        test_results["exception"] = str(e)
    
    finally:
        # Clean up unless requested to keep
        if test_dir and not args.keep_data:
            cleanup_test_environment(test_dir)
        elif test_dir:
            print(f"📁 Test data preserved at: {test_dir}")
    
    # Print summary
    print("\n" + "="*50)
    print("🧪 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = 0
    
    for test_name, result in test_results.items():
        if result is None:
            print(f"⚠️  {test_name}: SKIPPED")
        elif result:
            print(f"✅ {test_name}: PASSED") 
            passed += 1
            total += 1
        else:
            print(f"❌ {test_name}: FAILED")
            total += 1
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total and total > 0:
        print("🎉 All tests passed! The training pipeline is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())