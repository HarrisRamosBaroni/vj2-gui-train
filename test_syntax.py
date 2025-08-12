#!/usr/bin/env python3
"""
Simple syntax and structure test for the training files.
Tests basic Python syntax without requiring PyTorch dependencies.
"""

import ast
import sys
from pathlib import Path

def test_python_syntax(file_path):
    """Test if a Python file has valid syntax."""
    print(f"Testing syntax of {file_path}")
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Try to parse the AST
        ast.parse(content)
        print(f"✅ {file_path} - Valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"❌ {file_path} - Syntax error: {e}")
        print(f"   Line {e.lineno}: {e.text.strip() if e.text else 'N/A'}")
        return False
    except Exception as e:
        print(f"❌ {file_path} - Error: {e}")
        return False

def test_file_structure():
    """Test that all expected files exist."""
    required_files = [
        "vj2ac_train_multi_gpu.py",
        "vj2_dataloader.py", 
        "create_synthetic_data.py",
        "config.py"
    ]
    
    missing_files = []
    for file_name in required_files:
        if not Path(file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def main():
    print("🧪 VJ2 Training Pipeline Syntax Test")
    print("="*50)
    
    # Test file structure
    structure_ok = test_file_structure()
    if not structure_ok:
        return 1
    
    # Test syntax of key files
    files_to_test = [
        "vj2ac_train_multi_gpu.py",
        "vj2_dataloader.py",
        "create_synthetic_data.py",
        "test_training.py"
    ]
    
    syntax_results = {}
    for file_path in files_to_test:
        if Path(file_path).exists():
            syntax_results[file_path] = test_python_syntax(file_path)
        else:
            print(f"⚠️  {file_path} not found, skipping")
            syntax_results[file_path] = None
    
    # Summary
    print("\n" + "="*50)
    print("🧪 SYNTAX TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = 0
    
    for file_path, result in syntax_results.items():
        if result is None:
            print(f"⚠️  {file_path}: SKIPPED")
        elif result:
            print(f"✅ {file_path}: VALID")
            passed += 1
            total += 1
        else:
            print(f"❌ {file_path}: INVALID")
            total += 1
    
    print(f"\nResult: {passed}/{total} files have valid syntax")
    
    if passed == total and total > 0:
        print("🎉 All files have valid Python syntax!")
        
        # Test synthetic data generation (no dependencies needed)
        print("\n📊 Testing synthetic data generation...")
        try:
            import tempfile
            import shutil
            
            test_dir = Path(tempfile.mkdtemp(prefix="syntax_test_"))
            exec_globals = {}
            
            with open("create_synthetic_data.py", 'r') as f:
                code = f.read()
                # Replace the __main__ block to prevent execution
                code = code.replace('if __name__ == "__main__":', 'if False:')
            
            exec(code, exec_globals)
            
            # Test the function
            exec_globals['create_synthetic_data'](str(test_dir), 1, 2)
            
            # Check if files were created
            npz_files = list(test_dir.glob("*.npz"))
            if npz_files:
                print("✅ Synthetic data generation works")
            else:
                print("❌ No data files generated")
            
            # Cleanup
            shutil.rmtree(test_dir)
            
        except Exception as e:
            print(f"❌ Synthetic data generation failed: {e}")
        
        return 0
    else:
        print("⚠️  Some files have syntax errors.")
        return 1

if __name__ == "__main__":
    sys.exit(main())