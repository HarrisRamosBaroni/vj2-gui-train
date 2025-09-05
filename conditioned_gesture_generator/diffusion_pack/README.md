# 🎯 Diffusion Pack - DDIM Action Sequence Recreation

Complete implementation of DDIM (Denoising Diffusion Implicit Model) for VJ2-GUI action sequence recreation and planning evaluation.

## 📁 Package Contents

### Core Implementation
- `train_diffusion_ddim.py` - Complete DDIM training script with model, scheduler, and evaluation
- `DIFFUSION_TRAINING_DOCUMENTATION.md` - Detailed argument explanations and usage guide
- `RUN_DIFFUSION_TRAINING.md` - Quick start commands and expected results
- `diffusion_test_plan.md` - Full architecture documentation and implementation plan

### Data
- `test_data_mini/` - Minimal test dataset (10 train + 5 val sequences)
- `diffusion_test_data/` - Full training dataset (5K train + 1K val sequences)

## 🚀 Quick Start

### GPU Training on Real Data
```bash
cd diffusion_pack
python train_diffusion_ddim.py \
    --data_dir ../data_aug/train \
    --val_data_dir ../data_aug/val \
    --batch_size 64 \
    --num_epochs 50 \
    --lr 2e-4 \
    --d_model 64 \
    --num_layers 3 \
    --nhead 4 \
    --wandb_project action-ddim-real \
    --device cuda
```

### CPU Test (5 minutes)
```bash
cd diffusion_pack
python train_diffusion_ddim.py \
    --data_dir test_data_mini/train \
    --val_data_dir test_data_mini/val \
    --batch_size 2 \
    --num_epochs 3 \
    --lr 1e-3 \
    --d_model 32 \
    --num_layers 2 \
    --nhead 2 \
    --wandb_project ddim-test \
    --device cpu
```

## 🎯 Purpose

This package validates diffusion models as learned optimizers for the VJ2-GUI planner, replacing gradient descent with faster, more accurate DDIM sampling.

## 📊 Key Features

- **DDIM Sampling**: 10-50 steps vs DDPM's 1000 steps
- **Noise Analysis**: SNR, variance ratios, smoothness metrics
- **Real-time Evaluation**: Metrics computed every epoch
- **Visual Validation**: Action sequences + noise heatmaps every 10 epochs
- **WandB Integration**: Complete experiment tracking

## 📈 Expected Results

- **SNR >15dB**: Low reconstruction noise
- **Variance Ratio 0.5-2.0**: Preserves data characteristics  
- **MSE <0.02**: Good reconstruction quality
- **Training Speed**: ~1 minute per epoch on GPU

---

**Ready for immediate testing and integration into VJ2-GUI planner pipeline!**