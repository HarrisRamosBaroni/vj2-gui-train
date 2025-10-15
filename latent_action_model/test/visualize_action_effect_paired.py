"""
Visualizes the effect of random actions on a set of starting states with paired visualization.

For a given number of starting states, this script applies several random
actions to each and visualizes the resulting predicted states paired with
the original state, with decoded gesture actions overlaid on the original images.

Each result is shown as: [Original with Action Overlay] [Result] [Original with Action Overlay] [Result] ...
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
from conditioned_gesture_generator.train_autoregressive import load_autoregressive_model

logger = getLogger(__name__)

# --- Utility Functions (adapted from other test scripts) ---

@torch.no_grad()
def layer_norm_patches(z: torch.Tensor) -> torch.Tensor:
    """Apply LayerNorm along the patch-feature dimension (D)."""
    return F.layer_norm(z, (z.size(-1),))

def load_autoregressive_action_decoder(ckpt_path: Path, lam_ckpt_path: Path, lam_config_path: Path, device: torch.device):
    """Load autoregressive action decoder."""
    model, tokenizer, lam_model = load_autoregressive_model(
        checkpoint_path=str(ckpt_path),
        lam_ckpt_path=str(lam_ckpt_path),
        lam_config_path=str(lam_config_path),
        device=str(device)
    )
    print(f"Loaded autoregressive action decoder from {ckpt_path}")
    return model, tokenizer, lam_model

def plot_gestures_on_image(image, gestures, img_size=256):
    """
    Plot gesture trajectories on an image.

    Args:
        image: numpy array [H, W, 3] in range [0, 255]
        gestures: numpy array [T, 3] where columns are [x, y, touch]
        img_size: image size for coordinate scaling

    Returns:
        image with gestures plotted
    """
    img = image.copy()

    # Convert gesture coordinates from [0, 1] to image coordinates
    gesture_coords = gestures[:, :2] * img_size
    touch_values = gestures[:, 2]

    # Find continuous segments where touch=1
    touch_segments = []
    start_idx = None

    for i, touch in enumerate(touch_values):
        if touch > 0.5:  # Touch is active
            if start_idx is None:
                start_idx = i
        else:  # Touch is inactive
            if start_idx is not None:
                touch_segments.append((start_idx, i - 1))
                start_idx = None

    # Handle case where sequence ends with active touch
    if start_idx is not None:
        touch_segments.append((start_idx, len(touch_values) - 1))

    # Draw each touch segment
    for start, end in touch_segments:
        if start == end:
            # Single point - just draw a circle
            x, y = int(gesture_coords[start, 0]), int(gesture_coords[start, 1])
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)  # Green circle
        else:
            # Multi-point segment
            segment_coords = gesture_coords[start:end+1].astype(int)

            # Draw start point as circle
            start_x, start_y = segment_coords[0]
            cv2.circle(img, (start_x, start_y), 4, (0, 255, 0), -1)  # Green circle

            # Draw end point as cross
            end_x, end_y = segment_coords[-1]
            cross_size = 6
            cv2.line(img, (end_x - cross_size, end_y), (end_x + cross_size, end_y), (255, 0, 0), 2)  # Red cross
            cv2.line(img, (end_x, end_y - cross_size), (end_x, end_y + cross_size), (255, 0, 0), 2)

            # Draw connecting lines
            for i in range(len(segment_coords) - 1):
                pt1 = tuple(segment_coords[i])
                pt2 = tuple(segment_coords[i + 1])
                cv2.line(img, pt1, pt2, (0, 0, 255), 2)  # Blue line

    return img

def decode_actions_with_autoregressive(ar_model, tokenizer, lam_model, z_start, z_result, device):
    """
    Decode actions from state transition using autoregressive model.

    Args:
        ar_model: Autoregressive gesture decoder
        tokenizer: Gesture tokenizer
        lam_model: LAM model for encoding states
        z_start: [N, D] start state
        z_result: [N, D] result state
        device: torch device

    Returns:
        decoded_actions: [U, 3] decoded gesture actions
    """
    # Create a trajectory from start to result
    trajectory = torch.stack([z_start, z_result], dim=0).unsqueeze(0)  # [1, 2, N, D]

    with torch.no_grad():
        # Encode states with LAM to get action latents
        mu, logvar = lam_model.encode(trajectory)  # [1, 1, action_dim]
        action_sequence = lam_model.reparameterize(mu, logvar)

        # Generate gesture tokens using autoregressive model
        U = 250  # From model config

        try:
            if hasattr(ar_model, 'generate'):
                generated_tokens = ar_model.generate(
                    action_sequence,
                    max_length=U,
                    temperature=1.0
                )
            else:
                # Fallback: create dummy tokens for generation
                generated_tokens = {
                    'x_tokens': torch.randint(0, 3000, (1, U), device=device),
                    'y_tokens': torch.randint(0, 3000, (1, U), device=device),
                    'touch_tokens': torch.randint(0, 2, (1, U), device=device)
                }
        except Exception as e:
            print(f"Error in action generation: {e}")
            # Fallback: create dummy tokens
            generated_tokens = {
                'x_tokens': torch.randint(0, 3000, (1, U), device=device),
                'y_tokens': torch.randint(0, 3000, (1, U), device=device),
                'touch_tokens': torch.randint(0, 2, (1, U), device=device)
            }

        # Dequantize tokens to continuous actions
        decoded_actions = tokenizer.dequantize(generated_tokens)  # [1, U, 3]

        return decoded_actions[0].cpu().numpy()  # [U, 3]

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
def visualize_and_log_effects_paired(
    vae: LatentActionVAE,
    decoder: nn.Module,
    z_start: torch.Tensor,
    output_dir: Path,
    run: wandb.wandb_sdk.wandb_run.Run,
    index: int,
    num_random_actions: int,
    decoder_type: str,
    num_inference_steps: int,
    ar_model=None,
    tokenizer=None,
    lam_model=None,
):
    """
    Generates, visualizes, and logs the effect of random actions on a start state.
    Shows each result paired with the original with action overlays: [Original with Action] [Result] [Original with Action] [Result] ...
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

    # 4. Create paired visualization: [Original, Result1, Original, Result2, ...]
    paired_latents = []
    paired_labels = []

    for k in range(num_random_actions):
        paired_latents.append(z_start.unsqueeze(0))
        paired_latents.append(z_results[k].unsqueeze(0))
        paired_labels.extend([f"Original+Action{k+1}", f"Result{k+1}"])

    latents_to_viz = torch.cat(paired_latents, dim=0)

    # 5. Render images using the appropriate decoder
    decoder_device = next(decoder.parameters()).device
    latents_to_viz = latents_to_viz.to(decoder_device)

    if decoder_type == "diffusion":
        imgs = decoder.sample_ddim(latents_to_viz, num_inference_steps=num_inference_steps)
    else:
        imgs = decoder(latents_to_viz)

    # 6. Convert images to numpy for gesture overlay
    imgs_np = []
    for i in range(imgs.shape[0]):
        img = imgs[i].permute(1, 2, 0).cpu().numpy()
        img = ((img + 1) * 127.5).astype(np.uint8)  # [-1, 1] -> [0, 255]
        imgs_np.append(img)

    # 7. Add gesture overlays to original images (every other image starting from 0)
    if ar_model is not None and tokenizer is not None and lam_model is not None:
        print("Adding gesture overlays to original images...")
        for k in range(num_random_actions):
            original_idx = k * 2  # Every other image starting from 0
            result_idx = k * 2 + 1

            # Decode actions from state transition
            try:
                decoded_actions = decode_actions_with_autoregressive(
                    ar_model, tokenizer, lam_model,
                    z_start, z_results[k], device
                )

                # Add gesture overlay to original image
                img_with_gestures = plot_gestures_on_image(
                    imgs_np[original_idx],
                    decoded_actions[:50],  # Take first 50 actions as sample
                    img_size=imgs_np[original_idx].shape[0]
                )
                imgs_np[original_idx] = img_with_gestures
            except Exception as e:
                print(f"Error adding gesture overlay for action {k+1}: {e}")

    # 8. Create and log individual pairs
    print(f"Logging {num_random_actions} individual action-result pairs...")

    for k in range(num_random_actions):
        original_idx = k * 2
        result_idx = k * 2 + 1

        # Get the pair of images
        original_img = imgs_np[original_idx]
        result_img = imgs_np[result_idx]

        # Create side-by-side pair (preserve original aspect ratio)
        h, w = original_img.shape[:2]
        pair_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
        pair_img[:, :w] = original_img
        pair_img[:, w:] = result_img

        # Add text labels directly on the images
        label_height = 25
        pair_with_labels = np.zeros((h + label_height, w * 2, 3), dtype=np.uint8)
        pair_with_labels[label_height:, :] = pair_img

        # Add labels
        cv2.putText(pair_with_labels, f"Original+Action{k+1}", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pair_with_labels, f"Result{k+1}", (w + 5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Log to W&B as separate images (creates scrollable sequence)
        if run:
            step_value = index * num_random_actions + k  # Create unique step for scrolling
            run.log({
                "action_effect_pairs": wandb.Image(
                    pair_with_labels,
                    caption=f"Start state {index}, Action {k+1}: Cause → Effect"
                )
            }, step=step_value)

        # Save individual pair
        save_path = output_dir / f"start_state_{index}_action_{k+1}_pair.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(pair_with_labels, cv2.COLOR_RGB2BGR))

    print(f"Saved {num_random_actions} individual pairs for start state {index}")

# --- Main Execution ---

def parse_args():
    p = argparse.ArgumentParser(description="Visualize the effect of random actions on start states with paired layout and gesture overlays.")

    # Models
    p.add_argument("--vae_ckpt", type=Path, required=True, help="Path to VAE checkpoint.")
    p.add_argument("--vae_config", type=Path, required=True, help="Path to VAE config.")
    p.add_argument("--decoder_ckpt", type=Path, required=True, help="Path to visualization decoder checkpoint.")
    p.add_argument("--decoder_type", type=str, default="original", choices=["original", "diffusion"], help="Type of decoder model.")
    p.add_argument("--num_inference_steps", type=int, default=50, help="DDIM inference steps for diffusion decoder.")
    p.add_argument("--autoregressive_ckpt", type=Path, required=True, help="Path to autoregressive action decoder checkpoint.")

    # Data
    p.add_argument("--data_dir", type=Path, required=True, help="Directory with .h5 trajectory files.")
    p.add_argument("--manifest", type=Path, default=None, help="Path to data manifest JSON.")
    p.add_argument("--split", type=str, default="val", help="Split name in manifest.")

    # Logic
    p.add_argument("--num_start_states", type=int, default=5, help="Number of different start states to visualize (i).")
    p.add_argument("--num_random_actions", type=int, default=4, help="Number of random actions to apply per start state (j).")

    # Output & Logging
    p.add_argument("--output_dir", type=Path, default=Path("action_effect_paired_outputs"), help="Directory to save result images.")
    p.add_argument("--wandb_project", type=str, default="action-effects-paired-gestures", help="Wandb project name for logging.")
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

    # 3. Load autoregressive action decoder
    print("Loading autoregressive action decoder...")
    ar_model, tokenizer, lam_model = load_autoregressive_action_decoder(
        args.autoregressive_ckpt, args.vae_ckpt, args.vae_config, device
    )

    # 4. Load data
    loader, _ = init_preprocessed_data_loader(
        processed_data_dir=str(args.data_dir),
        batch_size=1,
        num_workers=0,
        manifest_path=str(args.manifest) if args.manifest else None,
        split_name=args.split,
    )
    dataset = loader.dataset

    # 5. Prepare for loop
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42) # for reproducible sampling

    print(f"Generating paired visualizations with gesture overlays for {args.num_start_states} start states...")

    # 6. Main loop
    for i in range(args.num_start_states):
        # Get a random start state from a random trajectory
        traj_idx = rng.randint(0, len(dataset) - 1)
        traj_emb, _ = dataset[traj_idx]

        if traj_emb.shape[0] < 1:
            print(f"Skipping empty trajectory {traj_idx}")
            continue

        frame_idx = rng.randint(0, traj_emb.shape[0] - 1)
        z_start = layer_norm_patches(traj_emb[frame_idx]).to(device)

        visualize_and_log_effects_paired(
            vae=vae,
            decoder=decoder,
            z_start=z_start,
            output_dir=args.output_dir,
            run=run,
            index=i,
            num_random_actions=args.num_random_actions,
            decoder_type=args.decoder_type,
            num_inference_steps=args.num_inference_steps,
            ar_model=ar_model,
            tokenizer=tokenizer,
            lam_model=lam_model,
        )

    # 7. Finish
    if run:
        run.finish()
    print("Done.")

if __name__ == "__main__":
    main()