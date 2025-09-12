# VAE Decoder Evaluation Framework

This framework provides comprehensive tools for evaluating Variational Autoencoder (VAE) decoders in the gesture generation system. It includes model loading, visualization, interpolation analysis, and metrics calculation.

## Overview

The evaluation framework consists of three main components:

1. **Evaluation Dashboard** (`evaluate_decoder.py`) - Interactive visualization and analysis
2. **Elite Model Management** - Dual tracking of total loss vs CE-only elite models
3. **Model Source Preservation** (`save_model_with_source.py`) - Future-proof checkpoint saving

## Quick Start

```bash
# Basic evaluation with WandB dashboard
python -m conditioned_gesture_generator.gesture_vae.evaluate_decoder \
  --checkpoint checkpoints/elite_0050_score_1.234567.pt \
  --use_wandb \
  --wandb_project my_vae_eval

# Local evaluation (HTML files only)
python -m conditioned_gesture_generator.gesture_vae.evaluate_decoder \
  --checkpoint checkpoints/best_model.pt \
  --num_samples 20 \
  --output_dir eval_results
```

## Main Evaluation Script: `evaluate_decoder.py`

### Core Features

- **Dynamic Model Loading**: Automatically detects model architecture from checkpoints
- **Latent Space Sampling**: Generates random samples from N(0,I) distribution
- **Interactive Visualizations**: 2D trajectories, time series, interpolation grids
- **Latent Interpolation**: Linear and spherical interpolation between latent vectors
- **Trajectory Metrics**: Smoothness, length, spatial coverage analysis
- **WandB Integration**: Real-time dashboard with all plots and metrics
- **Zero Filtering**: Only plots active gesture segments (non-zero coordinates)

### Command Line Arguments

#### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--checkpoint` | string | **Required**. Path to model checkpoint file (e.g., `checkpoints/elite_0050_score_1.234567.pt`) |

#### Model Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_type` | choice | `None` | Model handling: `vae` (full model), `decoder_only` (extract decoder from VAE) |
| `--latent_dim` | int | `None` | Latent space dimension. Auto-detected if not specified |
| `--k_classes` | int | `3000` | Number of quantization classes for coordinate discretization |

#### Sampling Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_samples` | int | `10` | Number of random latent samples to generate |
| `--seed` | int | `None` | Random seed for reproducible results |

#### Interpolation Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_interpolations` | int | `3` | Number of interpolation experiments (pairs of latent vectors) |
| `--interp_steps` | int | `10` | Number of intermediate steps in each interpolation |
| `--interp_method` | choice | `linear` | Interpolation method: `linear` or `spherical` (slerp) |

#### Output Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | string | `decoder_eval_results` | Directory to save HTML plots and results |

#### WandB Integration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_wandb` | flag | `False` | Enable WandB logging for interactive dashboard |
| `--wandb_project` | string | `vae_decoder_eval` | WandB project name |
| `--wandb_run_name` | string | `None` | Custom run name (defaults to checkpoint filename) |

### Example Usage Scenarios

#### 1. Quick Model Evaluation
```bash
python -m conditioned_gesture_generator.gesture_vae.evaluate_decoder \
  --checkpoint checkpoints/best_model.pt \
  --num_samples 5 \
  --num_interpolations 2
```

#### 2. Comprehensive Analysis with WandB
```bash
python -m conditioned_gesture_generator.gesture_vae.evaluate_decoder \
  --checkpoint checkpoints/elite_0050_score_1.234567.pt \
  --num_samples 20 \
  --num_interpolations 10 \
  --interp_steps 15 \
  --interp_method spherical \
  --seed 42 \
  --use_wandb \
  --wandb_project gesture_vae_analysis \
  --wandb_run_name "elite_model_deep_eval"
```

#### 3. CE-Elite Model Comparison
```bash
# Total loss elite
python -m conditioned_gesture_generator.gesture_vae.evaluate_decoder \
  --checkpoint checkpoints/elite_0050_score_1.234567.pt \
  --use_wandb --wandb_run_name "total_loss_elite"

# CE-only elite
python -m conditioned_gesture_generator.gesture_vae.evaluate_decoder \
  --checkpoint checkpoints/elite_ce_0050_score_0.987654.pt \
  --use_wandb --wandb_run_name "ce_only_elite"
```

#### 4. Decoder-Only Analysis
```bash
python -m conditioned_gesture_generator.gesture_vae.evaluate_decoder \
  --checkpoint checkpoints/full_vae_model.pt \
  --model_type decoder_only \
  --latent_dim 128 \
  --num_samples 15
```

## Elite Model System

The training system now maintains two separate elite model rankings:

### Elite Model Types

1. **Total Loss Elites** (`elite_XXXX_score_Y.YYYYYY.pt`)
   - Based on combined VAE loss (CE + KL)
   - Optimizes for both reconstruction quality and latent space regularization
   - Best for general VAE performance

2. **CE-Only Elites** (`elite_ce_XXXX_score_Y.YYYYYY.pt`)
   - Based purely on reconstruction loss (Cross-Entropy)
   - Optimizes only for generation accuracy
   - Best for pure reconstruction quality

### Configuration

In training scripts, configure elite tracking:

```python
checkpoint_manager = CheckpointManager(
    checkpoint_dir="checkpoints",
    num_elite=3,      # Top 3 total loss models
    num_elite_ce=3    # Top 3 CE-only models
)
```

### Finding Elite Models

```python
# Get elite model information
total_elites = checkpoint_manager.get_elite_models_info()
ce_elites = checkpoint_manager.get_elite_models_ce_info()

# Get best models
best_total = checkpoint_manager.get_best_elite_model()
best_ce = checkpoint_manager.get_best_elite_model_ce()
```

## Model Source Preservation: `save_model_with_source.py`

Future-proof your checkpoints by embedding model source code:

### Update Existing Checkpoint

```bash
python -m conditioned_gesture_generator.gesture_vae.save_model_with_source \
  checkpoints/my_model.pt \
  --output checkpoints/my_model_with_source.pt
```

### In Training Code

```python
from save_model_with_source import save_checkpoint_with_source

# Save with embedded source code
save_checkpoint_with_source(
    checkpoint_path="checkpoints/model_with_source.pt",
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    train_metrics=train_metrics,
    val_metrics=val_metrics
)
```

## Visualization Outputs

### Generated Visualizations

1. **2D Trajectory Plots**
   - Interactive Plotly plots showing gesture paths
   - Filtered to show only active segments (non-zero coordinates)
   - Start/end markers, multi-segment support

2. **Time Series Plots**
   - X(t) and Y(t) coordinate evolution over time
   - Dual subplot layout for temporal analysis

3. **Interpolation Grids**
   - Multiple subplots showing smooth transitions
   - Color gradient from blue to red across interpolation steps
   - Supports both linear and spherical interpolation

4. **Interactive Dashboard** (with WandB)
   - Real-time metrics tracking
   - Comparative analysis across runs
   - Downloadable plots and data

### Calculated Metrics

- **Total Variation**: Trajectory smoothness measure
- **Trajectory Length**: Total path length
- **Bounding Box Area**: Spatial coverage
- **Coordinate Statistics**: Mean and standard deviation
- **Interpolation Smoothness**: Step-by-step transition analysis

## Advanced Usage

### Batch Evaluation

```bash
#!/bin/bash
# Evaluate multiple checkpoints
for checkpoint in checkpoints/elite_*.pt; do
    python -m conditioned_gesture_generator.gesture_vae.evaluate_decoder \
      --checkpoint "$checkpoint" \
      --use_wandb \
      --wandb_run_name "$(basename $checkpoint .pt)"
done
```

### Custom Metrics Integration

Extend the evaluation by modifying `calculate_metrics()` in `evaluate_decoder.py`:

```python
def calculate_metrics(coords: np.ndarray) -> Dict[str, float]:
    # Add custom metrics here
    metrics = {
        'total_variation': float(total_variation),
        'trajectory_length': float(trajectory_length),
        # ... existing metrics
        'custom_metric': float(my_custom_calculation(coords))
    }
    return metrics
```

## Troubleshooting

### Common Issues

1. **Checkpoint Loading Errors**
   - Ensure model architecture matches checkpoint
   - Use `--model_type decoder_only` for VAE checkpoints when evaluating decoder only
   - Check `--latent_dim` and `--k_classes` parameters

2. **Memory Issues**
   - Reduce `--num_samples` and `--num_interpolations`
   - Use smaller `--interp_steps`

3. **WandB Connection**
   - Login with `wandb login`
   - Check internet connection
   - Use local evaluation without `--use_wandb` as fallback

4. **Empty Visualizations**
   - Check if model generates valid coordinates
   - Verify quantization parameters match training setup
   - Inspect raw model outputs before visualization

### Performance Tips

- Use `--seed` for reproducible results
- Start with small `--num_samples` for initial testing
- Enable WandB for collaborative analysis
- Save HTML plots locally as backup with `--output_dir`

## Integration with Training

The evaluation framework seamlessly integrates with the training pipeline:

1. **During Training**: Elite models are automatically saved
2. **Post-Training**: Use evaluation framework to compare models
3. **Model Selection**: Compare total loss vs CE-only elites for different use cases
4. **Production**: Use best models based on your evaluation criteria

This comprehensive evaluation framework provides everything needed to analyze, compare, and select the best VAE decoder models for your gesture generation tasks.