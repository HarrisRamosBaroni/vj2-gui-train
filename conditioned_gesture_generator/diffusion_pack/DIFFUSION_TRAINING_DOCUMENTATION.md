# DDIM Diffusion Training - Complete Documentation

## 🎯 Overview

This documentation provides detailed explanations for all command-line arguments and training procedures for the DDIM (Denoising Diffusion Implicit Model) implementation for action sequence recreation.

## 📋 Command Structure

```bash
python train_diffusion_ddim.py [REQUIRED_ARGS] [OPTIONAL_ARGS]
```

## 🔧 Required Arguments

### `--data_dir`
**Type**: `str`  
**Required**: ✅  
**Purpose**: Path to directory containing training data files  
**Format**: Directory containing `*_actions.npy` files  
**Example**: `--data_dir diffusion_test_data/train`

**Details**: 
- Must contain `.npy` files with action sequences
- Files should be generated using `augmentation.generate_click_dataset`
- Each file contains flattened action sequences: `[N*seq_len, 3]` → reshaped to `[N, seq_len, 3]`
- Action format: `[x, y, press]` where x,y ∈ [0,1] and press ∈ {0,1}

### `--val_data_dir`
**Type**: `str`  
**Required**: ✅  
**Purpose**: Path to directory containing validation data files  
**Format**: Same as `--data_dir`  
**Example**: `--val_data_dir diffusion_test_data/val`

**Details**:
- Used for real-time evaluation during training
- Should be smaller than training set (typically 10-20% of training size)
- Same format as training data
- Metrics computed every epoch for immediate feedback

## ⚙️ Model Architecture Arguments

### `--d_model`
**Type**: `int`  
**Default**: `128`  
**Purpose**: Hidden dimension of the transformer model  
**Range**: `32-512` (recommended)  
**Example**: `--d_model 64`

**Details**:
- Controls model capacity and memory usage
- Smaller values (32-64): Faster training, less memory, may underfit
- Larger values (256-512): More capacity, slower training, may overfit
- Rule of thumb: Start small (64) and increase if underfitting

### `--num_layers`
**Type**: `int`  
**Default**: `4`  
**Purpose**: Number of transformer encoder layers  
**Range**: `2-12` (recommended)  
**Example**: `--num_layers 3`

**Details**:
- More layers = deeper model with more representational power
- Fewer layers (2-4): Faster training, good for simple patterns
- More layers (6-12): Better for complex patterns, risk of overfitting
- Each layer adds ~25% more parameters

### `--nhead`
**Type**: `int`  
**Default**: `4`  
**Purpose**: Number of attention heads in multi-head attention  
**Constraint**: Must divide `d_model` evenly  
**Example**: `--nhead 8`

**Details**:
- Each head focuses on different aspects of the sequence
- More heads = more diverse attention patterns
- Common combinations: d_model=64→nhead=4, d_model=128→nhead=8
- Must satisfy: `d_model % nhead == 0`

## 🎓 Training Arguments

### `--batch_size`
**Type**: `int`  
**Default**: `32`  
**Purpose**: Number of sequences processed simultaneously  
**Range**: `1-128` (depends on GPU memory)  
**Example**: `--batch_size 64`

**Details**:
- Larger batches: More stable gradients, better GPU utilization
- Smaller batches: Less memory, more noisy gradients, may train faster
- GPU memory constraints: batch_size × seq_len × d_model must fit
- CPU: Use small batches (2-8), GPU: Use larger batches (32-128)

### `--num_epochs`
**Type**: `int`  
**Default**: `50`  
**Purpose**: Number of complete passes through the training dataset  
**Example**: `--num_epochs 100`

**Details**:
- More epochs = more training time, better convergence
- Monitor validation loss to detect overfitting
- Early stopping recommended if validation loss plateaus
- Typical ranges: Testing (3-10), Full training (50-200)

### `--lr` (Learning Rate)
**Type**: `float`  
**Default**: `1e-4` (0.0001)  
**Purpose**: Step size for gradient descent optimization  
**Range**: `1e-5` to `1e-2`  
**Example**: `--lr 2e-4`

**Details**:
- Higher learning rates: Faster convergence, risk of instability
- Lower learning rates: Stable training, slower convergence
- Cosine annealing scheduler automatically reduces learning rate over time
- Recommended: 1e-4 for stable training, 2e-4 for faster convergence

## 🖥️ System Arguments

### `--device`
**Type**: `str`  
**Default**: `auto`  
**Choices**: `auto`, `cpu`, `cuda`  
**Purpose**: Hardware device for computation  
**Example**: `--device cuda`

**Details**:
- `auto`: Automatically detects and uses GPU if available, otherwise CPU
- `cpu`: Forces CPU usage (slower, but always available)
- `cuda`: Forces GPU usage (faster, requires NVIDIA GPU with CUDA)
- GPU provides 2-10x speedup depending on model size and batch size

### `--wandb_project`
**Type**: `str`  
**Default**: `action-diffusion-test`  
**Purpose**: Name of the Weights & Biases project for experiment tracking  
**Example**: `--wandb_project my-diffusion-experiment`

**Details**:
- Creates/uses existing WandB project for logging metrics
- All training metrics, visualizations, and model info logged here
- Useful for comparing different hyperparameters and runs
- Change name for different experiment categories

### `--save_dir`
**Type**: `str`  
**Default**: `checkpoints`  
**Purpose**: Directory to save model checkpoints  
**Example**: `--save_dir model_saves`

**Details**:
- Saves model state every 20 epochs and at the end of training
- Checkpoints include: model weights, optimizer state, scheduler state
- Enables resuming training from interruptions
- Directory created automatically if it doesn't exist

## 📊 Complete Command Examples with Explanations

### 1. Quick CPU Test (5 minutes)
```bash
python train_diffusion_ddim.py \
    --data_dir test_data_mini/train \      # Minimal training data (10 sequences)
    --val_data_dir test_data_mini/val \    # Minimal validation data (5 sequences)
    --batch_size 2 \                       # Small batch for CPU
    --num_epochs 3 \                       # Just 3 epochs for quick test
    --lr 1e-3 \                           # Higher learning rate for fast learning
    --d_model 32 \                        # Small model (fast, low memory)
    --num_layers 2 \                      # Minimal depth
    --nhead 2 \                           # Few attention heads (32/2=16 per head)
    --wandb_project ddim-mac-test \       # Separate project for testing
    --device cpu                          # Force CPU usage
```

**Purpose**: Validate all components work correctly before full training

### 2. GPU Test (5 minutes)
```bash
python train_diffusion_ddim.py \
    --data_dir test_data_mini/train \      # Same minimal data
    --val_data_dir test_data_mini/val \
    --batch_size 16 \                     # Larger batch for GPU efficiency
    --num_epochs 5 \                      # Slightly longer test
    --lr 2e-4 \                          # Standard learning rate
    --d_model 128 \                      # Medium model size
    --num_layers 4 \                     # Standard depth
    --nhead 4 \                          # More attention heads (128/4=32 per head)
    --wandb_project ddim-cuda-test \     # Separate GPU test project
    --device cuda                        # Force GPU usage
```

**Purpose**: Test GPU acceleration and memory usage before full training

### 3. Full Training (50 minutes)
```bash
python train_diffusion_ddim.py \
    --data_dir diffusion_test_data/train \  # Full training data (5000 sequences)
    --val_data_dir diffusion_test_data/val \ # Full validation data (1000 sequences)
    --batch_size 64 \                       # Large batch for efficiency
    --num_epochs 50 \                       # Full training duration
    --lr 2e-4 \                            # Balanced learning rate
    --d_model 64 \                         # Optimized model size (good performance/speed trade-off)
    --num_layers 3 \                       # Sufficient depth without overfitting
    --nhead 4 \                            # Good attention diversity (64/4=16 per head)
    --wandb_project action-ddim-test       # Main experiment project (device auto-detected)
```

**Purpose**: Main training run for evaluating DDIM as learned optimizer

## 📈 Expected Behavior and Metrics

### Training Progression
1. **Epochs 0-10**: Loss decreases rapidly, SNR improves from <5dB to >10dB
2. **Epochs 10-30**: Gradual improvement, variance ratios stabilize near 1.0
3. **Epochs 30-50**: Fine-tuning, metrics approach target values

### Key Metrics to Monitor

#### Reconstruction Quality
- `val/mse`: Mean Squared Error between original and reconstructed sequences
  - **Target**: <0.01 (good reconstruction)
  - **Interpretation**: Lower = better reconstruction accuracy

- `val/coord_mse`: MSE for X,Y coordinates specifically
  - **Target**: <0.005 (precise spatial accuracy)
  - **Interpretation**: How well the model preserves spatial information

- `val/press_accuracy`: Accuracy of touch/press state prediction
  - **Target**: >0.9 (90%+ accuracy)
  - **Interpretation**: Binary classification accuracy for press/no-press

#### Noise Level Analysis
- `val/mean_snr_db`: Signal-to-Noise Ratio in decibels
  - **Target**: >20dB (low reconstruction noise)
  - **Interpretation**: Higher = cleaner reconstruction, less noise

- `val/variance_ratio_x/y`: How well original variance is preserved
  - **Target**: 0.8-1.2 (maintains data characteristics)
  - **Interpretation**: 1.0 = perfect variance preservation

- `val/smoothness_ratio_x/y`: Temporal smoothness of reconstruction
  - **Target**: <2.0 (not too noisy temporally)
  - **Interpretation**: Lower = smoother temporal transitions

### Visual Validation
- `val/reconstructions_epoch_X`: Action sequence plots every 10 epochs
  - Shows original, noisy, and reconstructed action sequences
  - Check for pattern recognition and temporal coherence
  - Look for proper gesture shape preservation (clicks, swipes, etc.)

## 🚨 Troubleshooting Common Issues

### Memory Issues
**Error**: CUDA out of memory  
**Solutions**: 
- Reduce `--batch_size` (try 32, 16, 8, 4)
- Reduce `--d_model` (try 64, 32)
- Use `--device cpu` as fallback

### Training Issues
**Problem**: Loss not decreasing  
**Solutions**:
- Increase `--lr` to 2e-4 or 5e-4
- Check data loading (ensure files exist and have correct format)
- Reduce model size if overfitting

**Problem**: NaN losses  
**Solutions**:
- Reduce `--lr` to 1e-5
- Check input data for NaN values
- Enable gradient clipping (already implemented)

### Performance Issues
**Problem**: Training too slow  
**Solutions**:
- Use `--device cuda` for GPU acceleration
- Increase `--batch_size` for better GPU utilization
- Reduce `--num_epochs` for testing

## 🎯 Hyperparameter Recommendations

### For Testing (5-10 minutes)
```bash
--batch_size 4 --num_epochs 5 --lr 1e-3 --d_model 32 --num_layers 2
```

### For Development (20-30 minutes)
```bash
--batch_size 32 --num_epochs 20 --lr 2e-4 --d_model 64 --num_layers 3
```

### For Production (1+ hours)
```bash
--batch_size 64 --num_epochs 100 --lr 1e-4 --d_model 128 --num_layers 4
```

This documentation covers all aspects of training the DDIM diffusion model, from quick validation tests to full production training runs.