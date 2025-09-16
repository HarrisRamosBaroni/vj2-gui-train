"""
Autoregressive rollout and visualization for LatentActionVAE.

This script performs an autoregressive rollout on a given trajectory
using the ground truth actions and visualizes the predicted trajectory
using a JEPA decoder.
"""

import argparse
from pathlib import Path
import torch
import numpy as np

# Project modules
from latent_action_model.vae import LatentActionVAE, load_lam_config
from training.dataloader import init_preprocessed_data_loader
from gui_world_model.jepa_decoder.model import JEPADecoder
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import cv2


@torch.no_grad()
def layer_norm_patches(z: torch.Tensor) -> torch.Tensor:
    """Apply LayerNorm along the patch‐feature dimension (D)."""
    return F.layer_norm(z, (z.size(-1),))


def load_vae_model(ckpt_path: Path, config_path: Path, device: torch.device) -> LatentActionVAE:
    """Load LatentActionVAE from checkpoint produced by LAMTrainer."""
    config_dict = load_lam_config(config_path)
    model = LatentActionVAE(**config_dict).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded VAE checkpoint from {ckpt_path}")
    return model


def load_jepa_decoder(dec_ckpt: Path, device: torch.device) -> JEPADecoder:
    """Load JEPA decoder for RGB visualisation."""
    checkpoint = torch.load(dec_ckpt, map_location=device, weights_only=True)
    cfg = checkpoint.get("config", {})
    latent_dim = cfg.get("latent_dim", 1024)
    model = JEPADecoder(latent_dim=latent_dim, output_resolution=250).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded JEPA decoder from {dec_ckpt}")
    return model


def get_test_trajectory(
    data_dir: Path,
    trajectory_index: int,
    manifest_path: Path,
    split_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads a single test trajectory using the H5 dataloader."""
    loader, _ = init_preprocessed_data_loader(
        processed_data_dir=str(data_dir),
        batch_size=1,  # We only need one trajectory
        num_workers=0,
        manifest_path=str(manifest_path) if manifest_path else None,
        split_name=split_name,
    )
    
    dataset = loader.dataset
    if trajectory_index >= len(dataset):
        raise ValueError(
            f"Trajectory index {trajectory_index} is out of bounds for "
            f"dataset with {len(dataset)} trajectories."
        )

    embeddings, actions = dataset[trajectory_index]
    
    # Ensure actions are flattened, [T, A_dim]
    if actions.ndim > 2:
        actions = actions.view(actions.shape[0], -1)
        
    return embeddings.unsqueeze(0), actions.unsqueeze(0)


def perform_rollout(
    vae: LatentActionVAE,
    z_sequence: torch.Tensor,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs an autoregressive rollout using the VAE decoder.
    """
    B, T, N, D = z_sequence.shape
    device = z_sequence.device

    # 1. Get initial context and ground truth actions
    z_context = z_sequence[:, :context_len]
    num_rollout_steps = T - context_len

    # 2. Encode the full sequence to get the ground truth actions for the rollout period
    with torch.no_grad():
        mu, logvar = vae.encode(z_sequence)
        true_actions = vae.reparameterize(mu, logvar)  # [B, T-1, A]

    # The actions for rollout start from the transition out of the last context frame
    rollout_actions = true_actions[:, context_len - 1 : T - 1]

    # 3. Generate the sequence autoregressively
    with torch.no_grad():
        z_hat_sequence = vae.generate_sequence(
            z_init=z_context,
            num_steps=num_rollout_steps,
            action_latents=rollout_actions,
        )

    print(f"Rollout complete. Context: {context_len}, Steps: {num_rollout_steps}")
    return z_hat_sequence, rollout_actions


def visualise_rollout(
    decoder: JEPADecoder,
    z_true_sequence: torch.Tensor,
    z_hat_sequence: torch.Tensor,
    z_true_actions: torch.Tensor,
    context_len: int,
    save_dir: Path,
    traj_idx: int,
):
    """
    Renders and saves a grid of images comparing the true and predicted trajectories.
    """
    z_true_sequence = z_true_sequence.squeeze(0)  # [T, N, D]
    z_hat_sequence = z_hat_sequence.squeeze(0)    # [T, N, D]

    T = z_true_sequence.shape[0]
    rollout_len = T - context_len

    # Prepare latents for visualization
    # Row 1: Ground Truth Trajectory
    # Row 2: Predicted Trajectory (with context copied)
    latents_to_viz = []
    labels = []

    # Pad predicted rollout with context for alignment
    z_hat_rollout = z_hat_sequence[context_len:]
    z_pred_padded = torch.cat([z_true_sequence[:context_len], z_hat_rollout], dim=0)

    for t in range(T):
        latents_to_viz.append(z_true_sequence[t])
        labels.append(f"True_t{t}")

    for t in range(T):
        latents_to_viz.append(z_pred_padded[t])
        if t < context_len:
            labels.append(f"Context_t{t}")
        else:
            labels.append(f"Pred_t{t}")

    latents_to_viz = torch.stack(latents_to_viz, dim=0)

    # Decode latents into images
    imgs = decoder(latents_to_viz.to(decoder.final_conv[0].weight.device))
    grid = make_grid(imgs, nrow=T, normalize=True, value_range=(-1, 1))

    # Add labels to the grid
    grid_np = grid.permute(1, 2, 0).cpu().numpy().copy()
    grid_np = (grid_np * 255).astype(np.uint8)
    h, w, _ = grid_np.shape
    img_width = w // T
    img_height = h // 2

    for i, label in enumerate(labels):
        row = i // T
        col = i % T
        pos = (col * img_width + 5, row * img_height + 15)
        # Add black border for visibility
        cv2.putText(
            grid_np, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA
        )
        cv2.putText(
            grid_np, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
        )

    # Save the final image
    save_path = save_dir / f"rollout_traj_{traj_idx}_ctx_{context_len}.png"
    cv2.imwrite(str(save_path), cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR))
    print(f"Saved visualisation grid to {save_path}")


def main(args):
    """Main function to run the rollout and visualization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load models
    vae = load_vae_model(args.vae_ckpt, args.vae_config, device)
    jepa_decoder = load_jepa_decoder(args.decoder_ckpt, device)

    # 2. Load data
    # [B, T, N, D], [B, T, A_d] where B=1
    z_sequence, _ = get_test_trajectory(
        args.data_dir, args.trajectory_index, args.manifest, args.split
    )
    z_sequence = layer_norm_patches(z_sequence).to(device)

    for i in range(args.min_context_len, 8):
        # 3. Perform autoregressive rollout
        z_hat_sequence, z_true_actions = perform_rollout(
            vae, z_sequence, i
        )

        # 4. Visualize the results
        visualise_rollout(
            jepa_decoder,
            z_sequence,
            z_hat_sequence,
            z_true_actions,
            i,
            args.save_dir,
            args.trajectory_index,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Autoregressive rollout for LatentActionVAE.")
    
    # Arguments for models
    parser.add_argument("--vae_ckpt", type=Path, required=True, help="Path to VAE checkpoint")
    parser.add_argument("--vae_config", type=Path, required=True, help="Path to VAE config")
    parser.add_argument("--decoder_ckpt", type=Path, required=True, help="Path to JEPA decoder checkpoint")

    # Arguments for data
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing the test trajectory files (.h5).")
    parser.add_argument("--trajectory_index", type=int, default=0, help="Index of the trajectory to evaluate in the test directory.")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to the data manifest JSON file.")
    parser.add_argument("--split", type=str, default=None, help="Split name in the manifest (e.g., 'train', 'val').")

    # Arguments for rollout
    parser.add_argument("--min_context_len", type=int, default=1, help="Number of initial frames to use as context.")

    # Arguments for output
    parser.add_argument("--save_dir", type=Path, required=True, help="Directory to save the visualization.")

    args = parser.parse_args()
    
    main(args)