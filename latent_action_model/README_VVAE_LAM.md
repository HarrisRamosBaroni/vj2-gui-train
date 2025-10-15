# VVAE Latent Action Model Training

Training pipeline for VVAELatentActionVQVAE using VVAE latent embeddings with VQ-VAE.

## Quick Start

### 1. Create Manifest

First, generate a manifest file to split your data into train/val/test sets (8:1:1 ratio):

```bash
python latent_action_model/create_vvae_manifest.py \
  /home/kevin/work/vj2-gui/output_h5 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42
```

This creates `output_h5/manifest.json` with random splits.

### 2. Test Dataloader (Optional)

Verify the dataloader works:

```bash
python latent_action_model/dataloader_vvae.py
```

### 3. Overfit Test

Test if the model can overfit to a single batch (sanity check):

```bash
python latent_action_model/train_vvae_lam.py \
  --data_dir /home/kevin/work/vj2-gui/output_h5 \
  --manifest_path /home/kevin/work/vj2-gui/output_h5/manifest.json \
  --test_mode \
  --num_epochs 1000 \
  --batch_size 4 \
  --learning_rate 1e-3
```

**Success criteria:** MSE < 0.01, MAE < 0.05

### 4. Full Training

Start full training:

```bash
python latent_action_model/train_vvae_lam.py \
  --data_dir /home/kevin/work/vj2-gui/output_h5 \
  --manifest_path /home/kevin/work/vj2-gui/output_h5/manifest.json \
  --batch_size 32 \
  --sequence_length 8 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --wandb_project vvae-lam
```

## Data Format

### Input: VVAE H5 Files
- **Location:** `/home/kevin/work/vj2-gui/output_h5/*.h5`
- **Format:** `[N_chunks, C=16, T_latent=2, H=64, W=64]`
- **Description:**
  - Each chunk = 1 second of video (8 frames)
  - 4× temporal compression (8 frames → 2 latent frames)
  - 8× spatial compression (512×512 → 64×64)
  - 16 latent channels

### Manifest File
- **Location:** `output_h5/manifest.json`
- **Contains:** Train/val/test splits
- **Default ratios:** 80% train, 10% val, 10% test

## Model Architecture

### VVAELatentActionVQVAE
- **Input adapter:** `VVAEtoLAMAdapter`
  - Conv2d: `[B, 16, 64, 64]` → `[B, 256, 16, 16]`
  - Flatten to patches: `[B, 256, 256]`

- **Core VQ-LAM:** VQ-VAE variant of LatentActionModel
  - `patch_dim=256` (reduced from 1024 for JEPA)
  - `num_patches=256`
  - VQ-VAE encoder/decoder transformers with 3-code quantization
  - Codebook vocabulary size: 12 (default, giving 12³ = 1,728 discrete actions)

- **Output adapter:** `LAMtoVVAEAdapter`
  - ConvTranspose2d: `[B, 256, 16, 16]` → `[B, 16, 64, 64]`

## Loss Configuration

### Primary Loss: MSE (Mean Squared Error)
- Changed from MAE for VVAE latents
- Better gradient signal for continuous latents

### Monitoring: Both MSE and MAE
- Training logs both metrics
- Validation logs both metrics
- WandB tracks both

### VQ-VAE Losses
- **Codebook loss:** Moves codebook toward encoder outputs
- **Commitment loss:** Encourages encoder to commit to codes
- Default commitment weight: 0.25
- No KL annealing (VQ-VAE uses discrete codes)

## Training Features

- **Warmup:** Linear LR warmup over `warmup_steps`
- **Gradient clipping:** Max norm = 1.0
- **Rollout training:** Optional multi-step decoder rollout
- **Test mode:** Single-batch overfitting for debugging
- **WandB logging:** Automatic tracking of all metrics
- **Checkpointing:** Save every N epochs

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Batch size |
| `sequence_length` | 8 | Latent frames per sequence (4 seconds) |
| `learning_rate` | 1e-4 | Base learning rate |
| `codebook_dim` | 128 | Dimension of each code embedding |
| `num_embeddings` | 12 | Codebook vocabulary size (N, giving N³ actions) |
| `embed_dim` | 512 | Transformer embedding dimension |
| `encoder_depth` | 3 | Encoder transformer layers |
| `decoder_depth` | 3 | Decoder transformer layers |
| `commitment_weight` | 0.25 | VQ-VAE commitment loss weight |
| `rollout_horizon` | 2 | Multi-step rollout length |

## Key Differences from JEPA LAM

1. **VQ-VAE instead of VAE:** Uses discrete codebook quantization
   - 3 codes per action for expressiveness
   - Codebook + commitment loss instead of KL divergence
   - No KL annealing needed

2. **Reduced patch dimension:** 256 vs 1024
   - Fewer parameters in projection layers
   - Faster training

3. **MSE loss:** Instead of MAE
   - Better for continuous VVAE latents

4. **Adapter layers:** CNN-based conversion
   - Efficient spatial transformation
   - Preserves information

5. **No normalization:** VVAE latents don't need layer norm

## File Structure

```
latent_action_model/
├── vqvae.py                     # Model definitions (includes VVAELatentActionVQVAE)
├── dataloader_vvae.py           # VVAE dataloader
├── train_vvae_lam.py            # Training script
├── create_vvae_manifest.py      # Manifest generation
└── README_VVAE_LAM.md           # This file
```

## Troubleshooting

### Overfit test fails
- Check data format: should be `[N_chunks, 16, 2, 64, 64]`
- Increase learning rate to 1e-3 for overfit test
- Increase num_epochs to 2000+

### Out of memory
- Reduce `batch_size`
- Reduce `sequence_length`
- Reduce `embed_dim`

### Poor validation performance
- Increase model capacity (`embed_dim`, `encoder_depth`)
- Adjust `commitment_weight` (try 0.1-0.5 for different trade-offs)
- Increase `num_embeddings` for larger codebook
- Enable rollout training for better multi-step prediction

## Monitoring

WandB tracks:
- `train/loss`, `train/mse_loss`, `train/mae_loss`, `train/vq_loss`
- `train/codebook_loss`, `train/commitment_loss`, `train/codebook_usage`
- `val/loss`, `val/mse_loss`, `val/mae_loss`, `val/vq_loss`
- `val/codebook_loss`, `val/commitment_loss`, `val/codebook_usage`
- `train/lr`, `train/grad_norm`
- `train/rollout_loss` (if enabled)
