import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import wandb

from config import ACTION_DIM, ACTIONS_PER_BATCH
FEAT_DIM = ACTION_DIM
from conditioned_gesture_generator.basic_ae.transformer_ae import TinyTransformerAutoencoder, sphere_loss, total_loss



def main(args):
    """
    Main training function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing model...")
    model = TinyTransformerAutoencoder(
        FEAT_DIM=FEAT_DIM,
        MODEL_DIM=args.model_dim,
        SEQ_LEN=ACTIONS_PER_BATCH,
        LATENT_DIM=args.latent_dim
    ).to(device)

    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    actions = torch.zeros(args.horizon, ACTIONS_PER_BATCH, ACTION_DIM, device=device)

    with torch.no_grad():
        latent_noop, recon = model(actions)
    
    print(f"{recon.shape=}")
    print(f"{total_loss(recon, actions, latent_noop)=}")
    
    latent_noop_np = latent_noop.detach().cpu().numpy()    
    np.save(args.save_path, latent_noop_np)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinyTransformerAutoencoder on action blocks.")
    
    # --- Data & IO ---
    parser.add_argument("--save_path", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to a checkpoint file to resume training from.")
    
    # --- Model Hyperparameters ---
    parser.add_argument("--model_dim", type=int, default=64, help="The internal dimension of the transformer model.")
    parser.add_argument("--latent_dim", type=int, default=10, help="The dimension of the compressed latent representation.")

    # --- Script args ---
    parser.add_argument("--horizon", type=int, default=1, help="Number of action blocks to create.")
    args = parser.parse_args()
    main(args)
    
# python -m training.train_action_autoencoder --data_dir ../downloads/dense_action_train_train --validation_dir ../downloads/dense_action_train_val
