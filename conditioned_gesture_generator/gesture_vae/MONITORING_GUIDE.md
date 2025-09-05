# VAE Monitoring Guide

## Overview
Advanced monitoring for CNN-CNN VAE is **ENABLED BY DEFAULT** to provide comprehensive insights into training dynamics, detect issues early, and optimize model performance.

## Quick Start
```bash
# Monitoring is automatically enabled
python -m conditioned_gesture_generator.gesture_vae.train_cnn_cnn \
    --data_path resource/action_data

# To disable monitoring (not recommended)
python -m conditioned_gesture_generator.gesture_vae.train_cnn_cnn \
    --data_path resource/action_data \
    --disable_monitoring
```

## Key Metrics Tracked

### 1. **Posterior Collapse Detection**
- `monitor/kl_active_dimensions`: Number of latent dimensions being used
- `monitor/kl_collapsed_dimensions`: Dimensions with KL < 0.01
- `monitor/kl_collapse_ratio`: Percentage of collapsed dimensions
- **Why it matters**: Indicates if the model is using its full capacity

### 2. **KL Divergence Analysis**
- `monitor/kl_per_dim_mean/std/min/max`: Distribution of KL across dimensions
- **Visual**: Heatmap showing KL per dimension over time
- **Why it matters**: Identifies which latent dimensions encode information

### 3. **Reconstruction Quality**
- `monitor/recon_x_accuracy`: X-coordinate prediction accuracy
- `monitor/recon_y_accuracy`: Y-coordinate prediction accuracy  
- `monitor/recon_overall_accuracy`: Both coordinates correct
- `monitor/recon_x/y_top5_accuracy`: Top-5 accuracy
- `monitor/recon_x/y_confidence`: Model confidence scores
- `monitor/recon_x/y_entropy`: Prediction uncertainty
- **Why it matters**: Tracks reconstruction performance separately from generation

### 4. **Latent Space Health**
- `monitor/latent_mu_mean/std`: Encoder mean statistics
- `monitor/latent_sigma_mean/std/min/max`: Encoder variance statistics
- `monitor/latent_norm`: Average latent vector magnitude
- **Why it matters**: Ensures latent space is well-behaved

### 5. **Loss Balance**
- `monitor/loss_recon_ratio`: Proportion of loss from reconstruction
- `monitor/loss_kl_ratio`: Proportion of loss from KL
- `monitor/loss_balance_score`: How balanced the losses are (ideal ~0.5)
- **Why it matters**: Ensures both objectives contribute to training

### 6. **Gradient Flow**
- `monitor/grad_encoder_grad_norm`: Encoder gradient magnitude
- `monitor/grad_decoder_grad_norm`: Decoder gradient magnitude
- **Why it matters**: Detects vanishing/exploding gradients

## Monitoring Dashboard
With **every validation run** (default: every 1% of epoch), a comprehensive 9-panel visualization is created:

1. **KL Heatmap**: Shows KL divergence per latent dimension over time
2. **Active Dimensions**: Tracks non-collapsed dimensions
3. **Coordinate Accuracies**: X/Y reconstruction accuracy trends
4. **Mu Statistics**: Latent mean behavior
5. **Sigma Statistics**: Latent variance behavior
6. **Loss Balance**: Reconstruction vs KL contribution
7. **Gradient Norms**: Gradient flow through network
8. **Overall Accuracy**: Combined reconstruction performance
9. **Summary Metrics**: Key indicators at a glance

## Configuration Options

```bash
# Adjust monitoring frequency (tied to validation frequency)
--val_interval "0.5%"  # More frequent monitoring (every 0.5% of epoch)
--val_interval "5%"    # Less frequent monitoring (every 5% of epoch)

# Disable monitoring for faster training (not recommended)
--disable_monitoring
```

## Interpreting Common Patterns

### Healthy Training
- Active dimensions: 80-100% of latent size
- Balance score: 0.4-0.6
- Steady accuracy improvement
- Stable gradient norms (not increasing/decreasing exponentially)

### Posterior Collapse
- Active dimensions < 50% of latent size
- Very low KL divergence across many dimensions
- **Fix**: Use KL annealing (`--kl_anneal 20`)

### Overfitting
- Training accuracy high but validation low
- Increasing KL divergence
- **Fix**: Reduce model capacity or add regularization

### Underfitting
- Low reconstruction accuracy
- High entropy in predictions
- **Fix**: Increase model capacity or training time

## Benefits of Default Monitoring
1. **Early Problem Detection**: Catch issues before wasting compute
2. **Optimization Insights**: Understand what the model is learning
3. **Hyperparameter Tuning**: See immediate effects of changes
4. **Research Value**: Deep understanding of VAE behavior
5. **Minimal Overhead**: Efficient implementation (~5% training slowdown)

## Disabling Monitoring
While not recommended, you can disable monitoring if:
- Running final production training with known-good hyperparameters
- Extremely compute-constrained environment
- Debugging other components

```bash
python -m conditioned_gesture_generator.gesture_vae.train_cnn_cnn \
    --data_path resource/action_data \
    --disable_monitoring
```

## Tips
- Keep monitoring enabled during experimentation
- Check monitoring plots regularly (logged to wandb with each validation)
- Use `--val_interval "0.5%"` for more frequent monitoring during debugging
- Watch for posterior collapse in first few epochs
- Balance score should stabilize after KL annealing completes

## ✨ New Features Added

### Time Series Visualization (like no_int classifier)
✅ **X/Y vs Time plots** - Shows coordinate evolution over timesteps

✅ **Smart zero handling** - Ignores leading zeros, starts from first movement

✅ **2D + Time series combo** - Both trajectory paths and time series in one view

✅ **Comprehensive comparison** - Original, reconstructed, and sampled trajectories

### Enhanced Monitoring
✅ **Dashboard with every validation** (tied to validation, not arbitrary epochs)

✅ **Default 1% validation frequency** - Frequent monitoring without overhead

✅ **Zero configuration** - Advanced monitoring enabled by default
- Default 1% validation frequency provides good monitoring without excessive overhead