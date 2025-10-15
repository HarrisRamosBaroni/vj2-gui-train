
"""
Visualizes the effect of random actions on a set of starting states.

For a given number of starting states, this script applies several random
actions to each and visualizes the resulting predicted states, logging them
locally and to Weights & Biases.
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

# --- Core Visualization Logic ---

@torch.no_grad()
def visualize_and_log_effects(
    vae: LatentActionVAE,
    decoder: nn.Module,
    z_start: torch.Tensor,
    output_dir: Path,
    run: wandb.wandb_sdk.wandb_run.Run,
    index: int,
    num_random_actions: int,
    decoder_type: str,
    num_inference_steps: int,
):
    """
    Generates, visualizes, and logs the effect of random actions on a start state.
    """
    device = next(vae.parameters()).device
    
    # 1. Generate random actions
    rand_actions = torch.randn(num_random_actions, vae.action_dim, device=device)
    
    # 2. Prepare batch for VAE decoder
    # The VAE decoder expects a sequence history, we use a sequence of length 1.
    z_start_past_sequence = z_start.unsqueeze(0).unsqueeze(0)
    z_start_batch = z_start_past_sequence.expand(num_random_actions, -1, -1, -1)
    
    # 3. Get resulting states
    z_results = vae.decode(z_start_batch, rand_actions)
    
    # 4. Combine for visualization
    latents_to_viz = torch.cat([z_start.unsqueeze(0), z_results], dim=0)
    labels = ["Start State"] + [f"Result {k+1}" for k in range(num_random_actions)]
    
    # 5. Render images using the appropriate decoder
    decoder_device = next(decoder.parameters()).device
    latents_to_viz = latents_to_viz.to(decoder_device)

    if decoder_type == "diffusion":
        imgs = decoder.sample_ddim(latents_to_viz, num_inference_steps=num_inference_steps)
    else:
        imgs = decoder(latents_to_viz)
        
    grid = make_grid(imgs, nrow=len(labels), normalize=True, value_range=(-1, 1))
    
    # 6. Add a black border at the bottom for annotations
    grid = F.pad(grid, (0, 0, 0, 30), 'constant', 0) # 30px black padding at the bottom

    # 7. Create annotated image
    grid_np = grid.permute(1, 2, 0).cpu().numpy().copy()
    grid_np = (grid_np * 255).astype(np.uint8)
    h, w, _ = grid_np.shape
    img_width = w // len(labels)
    for i, label in enumerate(labels):
        # Position text inside the new black border
        pos = (i * img_width + 5, h - 15)
        cv2.putText(grid_np, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(grid_np, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
    # 8. Log and Save
    caption = f"Start state {index}: Effect of {num_random_actions} random actions."
    
    if run:
        run.log({f"start_state_{index}": wandb.Image(grid_np, caption=caption)})
        
    save_path = output_dir / f"start_state_{index}_effects.png"
    cv2.imwrite(str(save_path), cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to {save_path}")

# --- Main Execution ---

def parse_args():
    p = argparse.ArgumentParser(description="Visualize the effect of random actions on start states.")
    
    # Models
    p.add_argument("--vae_ckpt", type=Path, required=True, help="Path to VAE checkpoint.")
    p.add_argument("--vae_config", type=Path, required=True, help="Path to VAE config.")
    p.add_argument("--decoder_ckpt", type=Path, required=True, help="Path to visualization decoder checkpoint.")
    p.add_argument("--decoder_type", type=str, default="original", choices=["original", "diffusion"], help="Type of decoder model.")
    p.add_argument("--num_inference_steps", type=int, default=50, help="DDIM inference steps for diffusion decoder.")

    # Data
    p.add_argument("--data_dir", type=Path, required=True, help="Directory with .h5 trajectory files.")
    p.add_argument("--manifest", type=Path, default=None, help="Path to data manifest JSON.")
    p.add_argument("--split", type=str, default="val", help="Split name in manifest.")

    # Logic
    p.add_argument("--num_start_states", type=int, default=5, help="Number of different start states to visualize (i).")
    p.add_argument("--num_random_actions", type=int, default=4, help="Number of random actions to apply per start state (j).")

    # Output & Logging
    p.add_argument("--output_dir", type=Path, default=Path("action_effect_outputs"), help="Directory to save result images.")
    p.add_argument("--wandb_project", type=str, default="action-effects", help="Wandb project name for logging.")
    p.add_argument("--wandb_run_name", type=str, required=True, help="Wandb run name (name after lam model).")
    
    p.add_argument("--device", type=str, default="cuda", help="Device to use (cuda|cpu).")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Init W&B
    run = None
    if args.wandb_project:
        run = init_wandb(
            project_name=args.wandb_project,
            run_name=args.wandb_run_name,
            config=vars(args)
        )
        
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
    
    # 4. Prepare for loop
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42) # for reproducible sampling
    
    print(f"Generating visualizations for {args.num_start_states} start states...")
    
    # 5. Main loop
    for i in range(args.num_start_states):
        # Get a random start state from a random trajectory
        traj_idx = rng.randint(0, len(dataset) - 1)
        traj_emb, _ = dataset[traj_idx]
        
        if traj_emb.shape[0] < 1:
            print(f"Skipping empty trajectory {traj_idx}")
            continue
            
        frame_idx = rng.randint(0, traj_emb.shape[0] - 1)
        z_start = layer_norm_patches(traj_emb[frame_idx]).to(device)
        
        visualize_and_log_effects(
            vae=vae,
            decoder=decoder,
            z_start=z_start,
            output_dir=args.output_dir,
            run=run,
            index=i,
            num_random_actions=args.num_random_actions,
            decoder_type=args.decoder_type,
            num_inference_steps=args.num_inference_steps,
        )
        
    # 6. Finish
    if run:
        run.finish()
    print("Done.")

if __name__ == "__main__":
    main()
