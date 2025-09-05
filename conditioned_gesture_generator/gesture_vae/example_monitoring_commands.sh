#!/bin/bash
# Example commands for training CNN-CNN VAE
# Note: Advanced monitoring is ENABLED BY DEFAULT
# Monitoring dashboard created with each validation (default: every 1% of epoch)

# Basic training (monitoring enabled by default)
python -m conditioned_gesture_generator.gesture_vae.train_cnn_cnn \
    --data_path resource/action_data \
    --epochs 50

# Training with KL annealing (monitoring auto-detects posterior collapse)
python -m conditioned_gesture_generator.gesture_vae.train_cnn_cnn \
    --data_path resource/action_data \
    --beta 4.0 \
    --kl_anneal 20 \
    --log_interval 25

# High-capacity model with frequent validation/monitoring
python -m conditioned_gesture_generator.gesture_vae.train_cnn_cnn \
    --data_path resource/action_data \
    --d_latent 256 \
    --hidden_dim 512 \
    --k_classes 5000 \
    --val_interval "0.5%" \
    --vis_interval 500

# Debug mode with very frequent validation/monitoring
python -m conditioned_gesture_generator.gesture_vae.train_cnn_cnn \
    --data_path resource/action_data \
    --log_interval 10 \
    --val_interval "5%" \
    --epochs 10 \
    --batch_size 16

# Lightweight model for fast experiments
python -m conditioned_gesture_generator.gesture_vae.train_cnn_cnn \
    --data_path resource/action_data \
    --lightweight \
    --d_latent 64 \
    --hidden_dim 128

# Training WITHOUT monitoring (faster but less insight)
python -m conditioned_gesture_generator.gesture_vae.train_cnn_cnn \
    --data_path resource/action_data \
    --disable_monitoring \
    --epochs 50

# Production training with all features (monitoring enabled by default)
python -m conditioned_gesture_generator.gesture_vae.train_cnn_cnn \
    --data_path resource/action_data \
    --d_latent 128 \
    --hidden_dim 512 \
    --k_classes 3000 \
    --beta 1.0 \
    --kl_anneal 10 \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-3 \
    --val_interval "2%" \
    --vis_with_val \
    --save_every 5 \
    --project_name vae_production \
    --experiment_name vae_production_run