# VVAE LAM Metrics Guide

## Metric Naming and Verification

This document verifies that all logged metrics accurately represent what their names indicate.

### Loss Metrics

#### Primary Optimization Losses (MSE-based)
- **`loss`**: Total training loss = `mse_loss * reconstruction_weight + vq_loss + rollout_loss * rollout_weight`
  - Uses **MSE** for reconstruction
  - **This is the loss used for backpropagation**

- **`recon_loss`**: Reconstruction loss using **MSE**
  - Formula: `(recons_vvae - targets).pow(2).mean()`
  - Same value as `mse_loss`

- **`mse_loss`**: Mean Squared Error reconstruction loss
  - Formula: `(recons_vvae - targets).pow(2).mean()`
  - Same value as `recon_loss`

- **`rollout_loss`**: Multi-step rollout loss using **MSE**
  - Formula: `(z_cur_vvae - z_true_vvae).pow(2).mean()` averaged over rollout steps
  - Only computed when `rollout_horizon > 1` and with probability `rollout_prob`

#### Monitoring Losses (MAE-based)
- **`loss_mae`**: Total loss with MAE = `mae_loss * reconstruction_weight + vq_loss + rollout_loss_mae * rollout_weight`
  - Uses **MAE** for reconstruction
  - **For monitoring only, NOT used for optimization**

- **`mae_loss`**: Mean Absolute Error reconstruction loss
  - Formula: `torch.abs(recons_vvae - targets).mean()`
  - **For monitoring only**

- **`rollout_loss_mae`**: Multi-step rollout loss using **MAE**
  - Formula: `torch.abs(z_cur_vvae_mae - z_true_vvae).mean()` averaged over rollout steps
  - **For monitoring only**

#### VQ-VAE Losses
- **`vq_loss`**: Total VQ-VAE quantization loss
  - Formula: `codebook_loss + commitment_loss`

- **`codebook_loss`**: Codebook learning loss (moves codebook toward encoder outputs)
  - Formula: `(z_q.detach() - z_e).pow(2).mean()`

- **`commitment_loss`**: Commitment loss (encourages encoder to commit to codes)
  - Formula: `commitment_weight * (z_q - z_e.detach()).pow(2).mean()`
  - Default `commitment_weight = 0.25`

### Action Sensitivity Metrics (Evaluation Only)

These metrics measure how much the decoder depends on action codes:

#### Distance-based Metrics
- **`action_sensitivity_l2`**: L2 distance improvement = `(d_rand - d_action).mean()`
  - Higher is better (decoder prefers true actions over random)

- **`action_sensitivity_l1`**: L1 distance improvement = `(d_rand_l1 - d_action_l1).mean()`

- **`action_sensitivity_mae`**: MAE improvement = `(d_rand_mae - d_action_mae).mean()`

- **`action_sensitivity_cos`**: Cosine distance improvement = `(d_cos_rand - d_cos_action).mean()`

#### DSNR (Decoder Signal-to-Noise Ratio)
- **`dsnr`**: Signal-to-noise ratio = `(d_rand - d_action) / (d_action_eps - d_action)`
  - Higher is better
  - Measures action sensitivity relative to perturbation noise

#### Raw Distances
- **`d_action_l2`**: L2 distance between prediction with true actions and target
  - **Uses MAE naming but actually L2**: Lower is better

- **`d_rand_l2`**: L2 distance between prediction with random actions and target

- **`d_action_eps_l2`**: L2 distance with perturbed actions (one code flipped)

- **`d_action_l1`**: L1 distance with true actions

- **`d_rand_l1`**: L1 distance with random actions

- **`d_action_mae`**: **TRUE MAE** with true actions (averaged over all dimensions)

- **`d_rand_mae`**: **TRUE MAE** with random actions

### Codebook Metrics
- **`codebook_usage`**: Number of unique codes used in current batch/epoch
  - Integer value (e.g., 8 out of 12 possible codes)

- **`codebook_unique_total`**: Total unique codes used across entire validation set
  - Only computed during validation

- **`codebook_usage_ratio`**: Fraction of codebook used = `codebook_unique_total / num_embeddings`
  - Range: [0, 1]

### Training Metrics
- **`lr`**: Current learning rate (after warmup schedule)

- **`kl_weight`**: KL annealing weight (not used in VQ-VAE, kept for compatibility)
  - Always 1.0 for VQ-VAE

- **`total_tflops`**: Cumulative TFLOPs consumed since training start

### WandB Logging Structure

#### Training Logs
- `train/loss` - Total MSE-based loss
- `train/loss_mae` - Total MAE-based loss (monitoring)
- `train/mse_loss` - Reconstruction MSE
- `train/mae_loss` - Reconstruction MAE (monitoring)
- `train/vq_loss` - VQ loss
- `train/codebook_loss` - Codebook component
- `train/commitment_loss` - Commitment component
- `train/rollout_loss` - Rollout MSE
- `train/rollout_loss_mae` - Rollout MAE (monitoring)
- `train/codebook_usage` - Unique codes per batch
- `train/lr` - Learning rate

#### Validation Logs
- `val/loss` - Total MSE-based loss
- `val/loss_mae` - Total MAE-based loss (monitoring)
- `val/mse_loss` - Reconstruction MSE
- `val/mae_loss` - Reconstruction MAE (monitoring)
- `val/vq_loss` - VQ loss
- `val/rollout_loss` - Rollout MSE
- `val/rollout_loss_mae` - Rollout MAE (monitoring)
- `val/action_sensitivity_l2` - L2 action sensitivity
- `val/action_sensitivity_l1` - L1 action sensitivity
- `val/action_sensitivity_mae` - MAE action sensitivity
- `val/action_sensitivity_cos` - Cosine action sensitivity
- `val/dsnr` - Decoder Signal-to-Noise Ratio
- `val/codebook_usage` - Unique codes per batch
- `val/codebook_unique_total` - Total unique codes in validation set
- `val/codebook_usage_ratio` - Fraction of codebook used
- `val/codebook_histogram` - Histogram of code usage

#### TFLOP-Indexed Logs (for plotting vs compute)
**Training:**
- `tflops/train_loss_mse` - MSE-based total loss
- `tflops/train_loss_mae` - MAE-based total loss (monitoring)
- `tflops/train_recon_mse` - Reconstruction MSE
- `tflops/train_recon_mae` - Reconstruction MAE (monitoring)
- `tflops/train_rollout_mse` - Rollout MSE
- `tflops/train_rollout_mae` - Rollout MAE (monitoring)
- `tflops/train_vq_loss` - VQ loss
- `tflops/x_axis` - TFLOPs (use as x-axis)

**Validation:**
- `tflops/val_loss_mse` - MSE-based total loss
- `tflops/val_loss_mae` - MAE-based total loss (monitoring)
- `tflops/val_recon_mse` - Reconstruction MSE
- `tflops/val_recon_mae` - Reconstruction MAE (monitoring)
- `tflops/val_rollout_mse` - Rollout MSE
- `tflops/val_rollout_mae` - Rollout MAE (monitoring)
- `tflops/val_vq_loss` - VQ loss
- `tflops/val_dsnr` - DSNR
- `tflops/x_axis` - TFLOPs (use as x-axis)

## Verification Checklist

### ✅ MSE Metrics
- [x] `loss` uses MSE reconstruction (line 302 in vqvae.py)
- [x] `recon_loss` = MSE (line 344 in vqvae.py)
- [x] `mse_loss` = MSE (line 295 in vqvae.py)
- [x] `rollout_loss` uses MSE (line 337 in vqvae.py)

### ✅ MAE Metrics
- [x] `loss_mae` uses MAE reconstruction (line 305 in vqvae.py)
- [x] `mae_loss` = MAE (line 299 in vqvae.py)
- [x] `rollout_loss_mae` uses MAE (line 354 in vqvae.py)

### ✅ Distance Metrics
- [x] `d_action_mae` uses MAE formula (line 606 in train_vvae_lam.py)
- [x] `d_rand_mae` uses MAE formula (line 607 in train_vvae_lam.py)
- [x] Other distance metrics use appropriate norms (L1, L2, cosine)

### ✅ WandB Logging
- [x] All MSE metrics logged with `_mse` suffix in tflops namespace
- [x] All MAE metrics logged with `_mae` suffix in tflops namespace
- [x] Standard namespace uses clear names (mse_loss, mae_loss)
- [x] Codebook histograms logged correctly

## Key Takeaways

1. **Optimization uses MSE**: The model is optimized with `loss` which uses MSE reconstruction
2. **MAE is monitored**: All MAE-based metrics (`loss_mae`, `mae_loss`, `rollout_loss_mae`) are for monitoring/comparison only
3. **Clear naming**: MSE metrics typically named `*_loss` or `*_mse`, MAE metrics named `mae_loss` or `*_mae`
4. **Both tracked**: You can compare MSE vs MAE performance in WandB by plotting both versions side-by-side
