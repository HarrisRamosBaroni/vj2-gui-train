"""
Performs a systematic analysis of the latent action space of a VAE model.

This script implements the following analysis:
1.  **Rollout Analysis**: For a single starting state, it applies a large
    number of random latent actions to generate a distribution of resulting
    states. It calculates the L1 distance between the start state and each
    result, and plots a histogram of these distances.

2.  **Quantile Action Selection**: From the distance distribution, it identifies
    the latent actions that caused the minimum, lower-quartile, median,
    upper-quartile, and maximum change.

3.  **Generalization Test**: It applies these five "quantile" actions to a
    new set of starting states.

4.  **Visualization**: It decodes the initial and resulting states into images,
    displaying them in a labeled grid to visually assess if the quantile
    actions have a consistent semantic effect across different contexts.
"""
import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision.utils import make_grid
import wandb
from logging import getLogger
import matplotlib.pyplot as plt

# Project modules
from latent_action_model.vae import LatentActionVAE, load_lam_config
from training.dataloader import init_preprocessed_data_loader
from gui_world_model.jepa_decoder_diffusion.model import JEPADecoder, JEPADiffusionDecoder
from gui_world_model.utils.wandb_utils import init_wandb

logger = getLogger(__name__)

# --- Utility Functions (adapted from other test scripts) ---

@torch.no_grad()
def layer_norm_patches(z: torch.Tensor) -> torch.Tensor:
    """Apply LayerNorm along the patch-feature dimension (D)."""
    return F.layer_norm(z, (z.size(-1),))

def load_vae_model(ckpt_path: Path, config_path: Path, device: torch.device) -> LatentActionVAE:
    """Load LatentActionVAE from checkpoint."""
    config_dict = load_lam_config(config_path)
    model = LatentActionVAE(**config_dict).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded VAE checkpoint from {ckpt_path}")
    return model

def load_decoder(dec_ckpt: Path, device: torch.device, decoder_type: str) -> nn.Module:
    """Load a decoder model for RGB visualisation."""
    checkpoint = torch.load(dec_ckpt, map_location=device, weights_only=True)
    cfg = checkpoint.get("config", {})
    latent_dim = cfg.get("latent_dim", 1024)
    
    if decoder_type == "diffusion":
        model = JEPADiffusionDecoder(latent_dim=latent_dim, output_resolution=256).to(device)
        print("Loading JEPADiffusionDecoder...")
    else:
        model = JEPADecoder(latent_dim=latent_dim, output_resolution=250).to(device)
        print("Loading JEPADecoder...")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded {decoder_type} decoder from {dec_ckpt}")
    return model

# --- Core Analysis Logic ---

@torch.no_grad()
def find_quantile_actions(
    vae: LatentActionVAE,
    z_start: torch.Tensor,
    num_rollouts: int,
    output_dir: Path,
    run: wandb.wandb_sdk.wandb_run.Run,
) -> tuple[torch.Tensor, dict]:
    """
    Performs rollouts with random actions, analyzes the distribution of
    state changes, and returns actions corresponding to quantiles.
    """
    device = next(vae.parameters()).device
    print(f"Performing {num_rollouts} rollouts to find quantile actions...")

    # 1. Generate a large batch of random actions
    rand_actions = torch.randn(num_rollouts, vae.action_dim, device=device)

    # 2. Prepare batch for VAE decoder
    z_start_past_sequence = z_start.unsqueeze(0).unsqueeze(0)
    z_start_batch = z_start_past_sequence.expand(num_rollouts, -1, -1, -1)

    # 3. Get resulting states
    z_results = vae.decode(z_start_batch, rand_actions)

    # 4. Calculate L1 distances
    l1_distances = torch.mean(torch.abs(z_results - z_start.unsqueeze(0)), dim=(-1, -2))
    l1_distances_np = l1_distances.cpu().numpy()

    # 5. Log histogram of distances
    plt.figure(figsize=(10, 6))
    plt.hist(l1_distances_np, bins=50, alpha=0.75)
    plt.title("Distribution of L1 Distances from Start State after Random Actions")
    plt.xlabel("Mean L1 Distance")
    plt.ylabel("Frequency")
    plt.grid(True)
    hist_path = output_dir / "l1_distance_histogram.png"
    plt.savefig(hist_path)
    plt.close()
    if run:
        run.log({"l1_distance_histogram": wandb.Image(str(hist_path))})
    print(f"Saved L1 distance histogram to {hist_path}")

    # 6. Find actions for quantiles
    quantiles = {
        "min": 0.0,
        "lq": 0.25,
        "med": 0.5,
        "uq": 0.75,
        "max": 1.0
    }
    
    quantile_actions = {}
    quantile_vals = np.quantile(l1_distances_np, list(quantiles.values()))
    
    # For min/max, we take the absolute min/max. For others, we find the closest.
    indices = {
        "min": np.argmin(l1_distances_np),
        "lq": np.abs(l1_distances_np - quantile_vals[1]).argmin(),
        "med": np.abs(l1_distances_np - quantile_vals[2]).argmin(),
        "uq": np.abs(l1_distances_np - quantile_vals[3]).argmin(),
        "max": np.argmax(l1_distances_np),
    }

    print("Selected actions based on L1 distance:")
    for name, idx in indices.items():
        quantile_actions[name] = rand_actions[idx]
        print(f"  - {name.upper()}: L1 dist = {l1_distances_np[idx]:.4f} (index {idx})")

    return torch.stack(list(quantile_actions.values())), quantile_actions.keys()


@torch.no_grad()
def visualize_quantile_effects(
    vae: LatentActionVAE,
    decoder: nn.Module,
    z_start: torch.Tensor,
    quantile_actions: torch.Tensor,
    action_labels: list[str],
    output_dir: Path,
    run: wandb.wandb_sdk.wandb_run.Run,
    index: int,
    decoder_type: str,
    num_inference_steps: int,
):
    """
    Applies the selected quantile actions to a start state and visualizes the results.
    """
    device = next(vae.parameters()).device
    num_actions = quantile_actions.shape[0]

    # 1. Prepare batch for VAE decoder
    z_start_past_sequence = z_start.unsqueeze(0).unsqueeze(0)
    z_start_batch = z_start_past_sequence.expand(num_actions, -1, -1, -1)

    # 2. Get resulting states from quantile actions
    z_results = vae.decode(z_start_batch, quantile_actions)

    # 3. Calculate actual L1 distances for these specific results
    l1_distances = torch.mean(torch.abs(z_results - z_start.unsqueeze(0)), dim=(-1, -2))

    # 4. Combine for visualization
    latents_to_viz = torch.cat([z_start.unsqueeze(0), z_results], dim=0)
    labels = ["Start"] + [f"{name.upper()} (L1: {dist:.3f})" for name, dist in zip(action_labels, l1_distances)]

    # 5. Render images
    decoder_device = next(decoder.parameters()).device
    latents_to_viz = latents_to_viz.to(decoder_device)

    if decoder_type == "diffusion":
        imgs = decoder.sample_ddim(latents_to_viz, num_inference_steps=num_inference_steps)
    else:
        imgs = decoder(latents_to_viz)
        
    grid = make_grid(imgs, nrow=len(labels), normalize=True, value_range=(-1, 1))
    
    # 6. Add a black border for annotations
    grid = F.pad(grid, (0, 0, 0, 80), 'constant', 0)

    # 7. Create annotated image
    grid_np = grid.permute(1, 2, 0).cpu().numpy().copy()
    grid_np = (grid_np * 255).astype(np.uint8)
    h, w, _ = grid_np.shape
    img_width = w // len(labels)
    for i, label in enumerate(labels):
        pos = (i * img_width + 5, h - 40)
        cv2.putText(grid_np, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(grid_np, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
    # 8. Log and Save
    caption = f"Test state {index}: Effect of quantile actions."
    
    if run:
        run.log({f"test_state_{index}": wandb.Image(grid_np, caption=caption)})
        
    save_path = output_dir / f"test_state_{index}_quantile_effects.png"
    cv2.imwrite(str(save_path), cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to {save_path}")

# --- Main Execution ---

def parse_args():
    p = argparse.ArgumentParser(description="Analyze the latent action space of a VAE model.")
    
    # Models
    p.add_argument("--vae_ckpt", type=Path, required=True, help="Path to VAE checkpoint.")
    p.add_argument("--vae_config", type=Path, required=True, help="Path to VAE config.")
    p.add_argument("--decoder_ckpt", type=Path, required=True, help="Path to visualization decoder checkpoint.")
    p.add_argument("--decoder_type", type=str, default="original", choices=["original", "diffusion"], help="Type of decoder model.")
    p.add_argument("--num_inference_steps", type=int, default=50, help="DDIM inference steps for diffusion decoder.")

    # Data
    p.add_argument("--data_dir", type=Path, required=True, help="Directory with .h5 trajectory files.")
    p.add_argument("--manifest", type=Path, default=None, help="Path to data manifest JSON.")
    p.add_argument("--split", type=str, default="validation", help="Split name in manifest.")

    # Logic
    p.add_argument("--num_rollouts", type=int, default=1000, help="Number of random actions for initial distribution analysis.")
    p.add_argument("--num_test_states", type=int, default=5, help="Number of different start states to test quantile actions on.")

    # Output & Logging
    p.add_argument("--output_dir", type=Path, default=Path("action_space_analysis"), help="Directory to save result images.")
    p.add_argument("--wandb_project", type=str, default="action-space-analysis", help="Wandb project name for logging.")
    p.add_argument("--wandb_run_name", type=str, required=True, help="Wandb run name (name after lam model).")
    
    p.add_argument("--device", type=str, default="cuda", help="Device to use (cuda|cpu).")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Init W&B and Output Dir
    run = None
    if args.wandb_project:
        run = init_wandb(
            project_name=args.wandb_project,
            run_name=args.wandb_run_name,
            config=vars(args)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
        
    # 2. Load models
    vae = load_vae_model(args.vae_ckpt, args.vae_config, device)
    decoder = load_decoder(args.decoder_ckpt, device, args.decoder_type)
    
    # 3. Load data
    loader, _ = init_preprocessed_data_loader(
        processed_data_dir=str(args.data_dir),
        batch_size=1,
        num_workers=0,
        manifest_path=str(args.manifest) if args.manifest else None,
        split_name=args.split,
    )
    dataset = loader.dataset
    rng = random.Random(42)
    
    # --- Part 1: Find Quantile Actions ---
    # Get a fixed start state for the main analysis
    traj_idx = rng.randint(0, len(dataset) - 1)
    traj_emb, _ = dataset[traj_idx]
    frame_idx = rng.randint(0, traj_emb.shape[0] - 1)
    z_start_for_analysis = layer_norm_patches(traj_emb[frame_idx]).to(device)
    
    quantile_actions, action_labels = find_quantile_actions(
        vae=vae,
        z_start=z_start_for_analysis,
        num_rollouts=args.num_rollouts,
        output_dir=args.output_dir,
        run=run,
    )
    
    # --- Part 2: Test Quantile Actions on New States ---
    print(f"\nTesting quantile actions on {args.num_test_states} new start states...")
    
    # Get a list of unique random indices for test states
    test_state_indices = random.sample(range(len(dataset)), args.num_test_states)

    for i, traj_idx in enumerate(test_state_indices):
        traj_emb, _ = dataset[traj_idx]
        
        if traj_emb.shape[0] < 1:
            print(f"Skipping empty trajectory {traj_idx}")
            continue
            
        frame_idx = rng.randint(0, traj_emb.shape[0] - 1)
        z_start_test = layer_norm_patches(traj_emb[frame_idx]).to(device)
        
        visualize_quantile_effects(
            vae=vae,
            decoder=decoder,
            z_start=z_start_test,
            quantile_actions=quantile_actions,
            action_labels=action_labels,
            output_dir=args.output_dir,
            run=run,
            index=i,
            decoder_type=args.decoder_type,
            num_inference_steps=args.num_inference_steps,
        )
        
    # 6. Finish
    if run:
        run.finish()
    print("Done.")

if __name__ == "__main__":
    main()
