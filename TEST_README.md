# VJ2 GUI Training Pipeline Testing

This directory contains comprehensive testing scripts to validate the DDP training pipeline without requiring real datasets or GPU clusters.

## Test Scripts

### 1. `test_syntax.py` - Syntax Validation
Tests Python syntax validity of all core files.

```bash
python test_syntax.py
```

### 2. `test_logic.py` - Logic Validation  
Tests core logic, argument parsing, and DDP feature implementation.

```bash
python test_logic.py
```

### 3. `create_synthetic_data.py` - Synthetic Dataset Generator
Creates fake `.npz` files matching the expected training data format.

```bash
python create_synthetic_data.py --output_dir ./test_data --num_files 3 --trajectories_per_file 10
```

### 4. `test_training.py` - Full Pipeline Test (Requires PyTorch)
Comprehensive test that creates synthetic data and runs actual training with DDP.

```bash
# With PyTorch environment activated:
python test_training.py

# Skip multi-rank tests:
python test_training.py --skip-multi

# Keep test data after completion:
python test_training.py --keep-data
```

### 5. `test_checkpoint_logic.py` - Checkpoint Strategy Validation
Tests the new checkpoint strategy optimized for large dataset single-epoch training.

```bash
python test_checkpoint_logic.py
```

## Test Results Summary

✅ **All tests currently pass**, confirming:

### DDP Fixes Implemented:
- ✅ DDP desync elimination with proper barriers
- ✅ Validation runs on all ranks with rank-0-only logging
- ✅ Backend-agnostic initialization (NCCL/Gloo)
- ✅ Proper distributed sampling for both train/validation
- ✅ Sequence length validation for rollout horizon
- ✅ Comprehensive DDP debugging prints

### Training Features:
- ✅ Multi-GPU support with proper synchronization
- ✅ CPU-only testing capability with Gloo backend
- ✅ Proper device handling (CUDA/CPU)
- ✅ Distributed sampler epoch synchronization
- ✅ NCCL timeout protection (120s)

## Usage Examples

### Multi-GPU CUDA Training (Large Dataset):
```bash
torchrun --nproc_per_node=2 vj2ac_train_multi_gpu.py \
  --processed_data_dir ./train_data \
  --validation_data_dir ./val_data \
  --num_epochs 1 \
  --batch_size 4 \
  --save_every_iters 1000
```

### CPU Testing with Gloo:
```bash  
torchrun --nproc_per_node=2 vj2ac_train_multi_gpu.py \
  --device cpu \
  --dist-backend gloo \
  --processed_data_dir ./train_data \
  --validation_data_dir ./val_data \
  --num_epochs 1 \
  --batch_size 2 \
  --save_every_iters 100
```

### Single GPU Testing:
```bash
torchrun --nproc_per_node=1 vj2ac_train_multi_gpu.py \
  --processed_data_dir ./train_data \
  --validation_data_dir ./val_data \
  --save_every_iters 500
```

## Expected Debug Output

When running with DDP, you should see output like:
```
[DDP] Initializing with backend=nccl, device=cuda
[DDP] Rank 0/2, local_rank=0, device=cuda
[DDP] Rank 0: Model initialized on cuda:0
[DDP] Dataloader created with 10000 samples, batch_size=4
[DDP] Using DistributedSampler with 2 ranks
[DDP] Rank 0: Starting training loop for 1 epochs
[DDP] Starting epoch 1/1
[DDP] Rank 0: Entering validation at epoch 0, iter 100
[DDP] Rank 1: Entering validation at epoch 0, iter 100
🏆 New best validation loss: 0.123456
✅ Saved checkpoint → checkpoints/.../vjepa2_best.pt (val_loss: 0.123456)
✅ Saved checkpoint → checkpoints/.../vjepa2_step_500.pt (step: 500)
[DDP] Rank 0: Exiting validation at epoch 0, iter 100
[DDP] Rank 1: Exiting validation at epoch 0, iter 100
```

## New Checkpoint Strategy

The training now uses an optimized checkpoint strategy for large datasets:

### Checkpoint Types:
1. **Best Model** (`vjepa2_best.pt`): Saved when validation loss improves
2. **Routine Checkpoints** (`vjepa2_step_N.pt`): Saved every `--save_every_iters` (default 500)  
3. **Interrupted** (`vjepa2_interrupted.pt`): Saved on Ctrl+C
4. **Epoch Checkpoints** (`vjepa2_epoch_N.pt`): Only if `--save_every_epochs > 0`

### What's Saved:
- **Only predictor weights** (no optimizer/scheduler states)
- Global step counter 
- Validation loss (for best model)

This minimizes checkpoint size and I/O overhead for large-scale training.

## Troubleshooting

### If you see NCCL timeouts:
- Check that `NCCL_DEBUG=INFO` shows proper initialization
- Verify all ranks are synchronizing at the same validation steps
- Ensure `TORCH_NCCL_BLOCKING_WAIT=1` if needed for debugging

### If synthetic data generation fails:
- Check that `config.py` values are reasonable
- Verify write permissions in output directory
- Ensure sufficient disk space for generated `.npz` files

### If tests fail without PyTorch:
- Run `test_syntax.py` and `test_logic.py` first (no dependencies needed)
- Install PyTorch environment for full `test_training.py`
- Use `--skip-multi` flag if torchrun is unavailable

## Files Modified for DDP Fixes

1. **vj2ac_train_multi_gpu.py**: Main training script with DDP synchronization
2. **vj2_dataloader.py**: Added distributed sampling and DDP logging
3. **create_synthetic_data.py**: Synthetic data generator for testing
4. **test_*.py**: Comprehensive test suite for validation

The pipeline is now ready for production multi-GPU training with proper DDP synchronization!