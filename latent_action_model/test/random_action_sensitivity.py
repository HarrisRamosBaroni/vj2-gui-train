"""
Random-Action Sensitivity Analysis for LatentActionVAE.

Run offline to quantify how strongly the decoder is conditioned on the action
latent.  Optionally visualises predicted frames using a JEPA decoder.

See .plan/random_action_sensitivity_analysis.md for detailed specification.
"""
from pathlib import Path
import argparse
import json
import random
import math
import os
import h5py

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from logging import getLogger

# Project modules
from latent_action_model.vae import LatentActionVAE
from training.dataloader import init_preprocessed_data_loader
from gui_world_model.jepa_decoder.model import JEPADecoder  # optional – guarded

logger = getLogger(__name__)


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
@torch.no_grad()
def layer_norm_patches(z: torch.Tensor) -> torch.Tensor:
    """Apply LayerNorm along the patch‐feature dimension (D)."""
    return F.layer_norm(z, (z.size(-1),))


def load_vae_model(ckpt_path: Path, device: torch.device) -> LatentActionVAE:
    """Load LatentActionVAE from checkpoint produced by LAMTrainer."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # cfg = ckpt.get("config", None)

    # if cfg is None:
    #     raise RuntimeError(
    #         "Checkpoint missing 'config' key – cannot reconstruct model architecture."
    #     )

    model = LatentActionVAE(
        # latent_dim=cfg["latent_dim"],
        # action_dim=cfg["action_dim"],
        # kl_weight=cfg.get("kl_weight", 0.1),
        # Remaining args default to training defaults – engineers may edit here.
        latent_dim   = 1024,   # patch_dim
        action_dim   = 128,
        embed_dim    = 512,
        encoder_depth= 2,
        decoder_depth= 2,
        encoder_heads= 8,
        decoder_heads= 8,
        kl_weight    = 0.001,  # <- training value
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded VAE checkpoint from {ckpt_path}")
    return model


def load_jepa_decoder(dec_ckpt: Path, device: torch.device) -> JEPADecoder:
    """Load JEPA decoder for RGB visualisation."""
    checkpoint = torch.load(dec_ckpt, map_location=device, weights_only=True)
    cfg = checkpoint.get("config", {})
    latent_dim = cfg.get("latent_dim", 1024)
    model = JEPADecoder(latent_dim=latent_dim, output_resolution=250).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded JEPA decoder from {dec_ckpt}")
    return model


def save_histogram(values: np.ndarray, out_path: Path, title: str, xlabel: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=100, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved histogram to {out_path}")


def get_ground_truth_images(
    session_id: str,
    traj_idx: int,
    latent_idx: int,
    image_data_dir: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load the two ground-truth images corresponding to a single latent state."""
    image_h5_path = image_data_dir / f"{session_id}_images.h5"
    if not image_h5_path.exists():
        logger.warning(f"Could not find image h5 at {image_h5_path}")
        return None

    with h5py.File(image_h5_path, 'r') as f_img:
        try:
            image1 = f_img['images'][traj_idx, latent_idx * 2]
            image2 = f_img['images'][traj_idx, latent_idx * 2 + 1]
            return image1, image2
        except (IndexError, KeyError) as e:
            logger.warning(f"Could not load images for {session_id} traj {traj_idx} latent {latent_idx}: {e}")
            return None


def visualise_predictions(
    decoder: JEPADecoder,
    latents: torch.Tensor,
        out_path: Path,
        labels: list[str],
        ground_truth_imgs: list[torch.Tensor] = None,
    ):
    """
    Render RGB reconstructions for a series of latents and save grid.
    """
    from torchvision.utils import make_grid, save_image  # runtime import
    import cv2

    imgs = decoder(latents.to(decoder.final_conv[0].weight.device))  # [K, 3, H, W]
    
    # If ground truth images are provided, create a larger grid
    if ground_truth_imgs:
        # Pad the decoded images with a blank image to match the ground truth row
        blank_img = torch.zeros_like(imgs[0])
        imgs_padded = torch.cat([imgs, blank_img.unsqueeze(0)], dim=0)

        # Move all tensors to the same device before concatenation
        device = imgs_padded.device
        gt_tensors_on_device = [img.to(device) for img in ground_truth_imgs]

        # Create a two-row grid
        combined_imgs = torch.cat(gt_tensors_on_device + list(imgs_padded), dim=0)
        grid = make_grid(combined_imgs, nrow=4, normalize=True, value_range=(-1, 1))
        
        # Add top-row labels for ground truth
        full_labels = ["z_t (gt 1)", "z_t (gt 2)", "z_t+1 (gt 1)", "z_t+1 (gt 2)"] + labels
    else:
        grid = make_grid(imgs, nrow=imgs.shape[0], normalize=True, value_range=(-1, 1))
        full_labels = labels

    # Add labels
    grid_np = grid.permute(1, 2, 0).cpu().numpy().copy()
    grid_np = (grid_np * 255).astype(np.uint8)
    h, w, _ = grid_np.shape
    num_labels_per_row = len(full_labels) if not ground_truth_imgs else len(full_labels) // 2
    img_width = w // num_labels_per_row
    
    for i, label in enumerate(full_labels):
        row = i // num_labels_per_row
        col = i % num_labels_per_row
        
        # Position label at the bottom-left of the image cell
        pos_x = col * img_width + 10
        pos_y = (row + 1) * (h // 2) - 20 if ground_truth_imgs else h - 20

        # Adjust y position for second row of labels if there are two rows
        if ground_truth_imgs and row == 1:
            pos_y = h - 20

        cv2.putText(
            grid_np, label, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

    # Save with cv2
    cv2.imwrite(str(out_path), cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved visualisation grid to {out_path}")


# --------------------------------------------------------------------------- #
# Core analysis
# --------------------------------------------------------------------------- #
@torch.no_grad()
def analyse_random_action_sensitivity(
    vae: LatentActionVAE,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_initial: int,
    num_random: int,
    output_dir: Path,
    decoder: JEPADecoder = None,
    mode: str = "random_action",
    image_data_dir: Path = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    if mode == "visualize_transition":
        if decoder is None:
            raise ValueError("Decoder must be provided for 'visualize_transition' mode.")
        if image_data_dir is None:
            raise ValueError("Image data dir must be provided for 'visualize_transition' mode.")

        for idx in range(num_initial):
            # 1) sample trajectory and timestep
            rand_idx = rng.randint(0, len(data_loader.dataset) - 1)
            
            # Correctly get metadata from the dataset's trajectory_index
            file_path, traj_idx_in_session = data_loader.dataset.trajectory_index[rand_idx]
            session_id = Path(file_path).stem
            
            # Load the actual trajectory embedding
            traj_emb, _ = data_loader.dataset[rand_idx]
            traj_emb = layer_norm_patches(traj_emb).to(device)  # [T, N, D]
            T, N, D = traj_emb.shape
            if T < 3:
                continue  # Need at least 3 states for a t-1 -> t transition
            t = rng.randint(2, T - 1)  # Transition is from t-1 -> t

            # 2) Get necessary states
            z_t = traj_emb[t - 1]
            z_tplus1_actual = traj_emb[t]

            # The VAE needs the sequence up to t to predict the action for t-1 -> t
            z_sequence_for_action = traj_emb[:t].unsqueeze(0)

            # 3) Get true action from encoder
            mu, logvar = vae.encode(z_sequence_for_action)  # [1, t-1, A]
            a_true = vae.reparameterize(mu, logvar)[:, -1, :]  # Take the last action, corresponding to t-1 -> t

            # 4) Predict next state
            z_hat_tplus1 = vae.decode(z_t.unsqueeze(0).unsqueeze(0), a_true)[0]

            # 5) Load ground truth images
            gt_imgs = []
            # ground truth for z_t
            imgs_t = get_ground_truth_images(session_id, traj_idx_in_session, t - 1, image_data_dir)
            # ground truth for z_t+1
            imgs_tplus1 = get_ground_truth_images(session_id, traj_idx_in_session, t, image_data_dir)

            if imgs_t and imgs_tplus1:
                for img_np in imgs_t + imgs_tplus1:
                    img_torch = torch.from_numpy(img_np).float() / 127.5 - 1.0
                    gt_imgs.append(img_torch.permute(2, 0, 1))

            # 6) Visualise
            vis_path = output_dir / f"transition_{idx}.png"
            latents_to_viz = torch.stack([z_t, z_hat_tplus1, z_tplus1_actual], dim=0)
            labels = ["z_t (decoded)", "z_hat_t+1 (pred)", "z_t+1 (decoded)"]
            
            visualise_predictions(decoder, latents_to_viz, vis_path, labels, ground_truth_imgs=gt_imgs or None)

            if (idx + 1) % 10 == 0:
                logger.info(f"Generated {idx+1}/{num_initial} transition visualizations.")

    elif mode == "random_action":
        l1_deltas = []
        for idx in range(num_initial):
            # sample trajectory and timestep
            traj_emb, _ = data_loader.dataset[rng.randint(0, len(data_loader.dataset) - 1)]
            traj_emb = layer_norm_patches(traj_emb).to(device)
            T, N, D = traj_emb.shape
            if T < 2:
                continue
            t = rng.randint(1, T - 1)
            z_past = traj_emb[:t].unsqueeze(0)
            z_target = traj_emb[t]

            # get true action for anchor prediction
            mu, logvar = vae.encode(traj_emb.unsqueeze(0))
            a_true = vae.reparameterize(mu, logvar)[0, t - 1]

            # random action batch
            rand_actions = torch.randn(num_random, vae.action_dim, device=device)
            a_batch = torch.cat([a_true.unsqueeze(0), rand_actions], dim=0)
            z_past_batch = z_past.expand(a_batch.shape[0], -1, -1, -1)

            # decode
            preds = vae.decode(z_past_batch, a_batch)
            anchor_pred = preds[0]

            # compute L1 deltas
            deltas = torch.mean(torch.abs(preds[1:] - anchor_pred.unsqueeze(0)), dim=(-2, -1))
            l1_deltas.extend(deltas.cpu().numpy())

            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx+1}/{num_initial} anchor states for random action analysis.")
        
        l1_deltas = np.array(l1_deltas)
        save_histogram(
            l1_deltas,
            output_dir / "hist_l1.png",
            title="Random-Action Sensitivity (Mean L1 to Anchor)",
            xlabel="Mean |Δ| (patch latent space)",
        )
        metrics = {
            "mean": float(np.mean(l1_deltas)),
            "median": float(np.median(l1_deltas)),
            "std": float(np.std(l1_deltas)),
            "num_samples": int(len(l1_deltas)),
        }
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {output_dir/'metrics.json'}: {metrics}")
    else:
        raise ValueError(f"Unknown mode: {mode}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Random-Action Sensitivity Analysis")
    p.add_argument("--vae_ckpt", type=Path, required=True, help="Path to VAE checkpoint")
    p.add_argument("--data_dir", type=Path, required=True, help="Trajectory directory")
    p.add_argument(
        "--mode", type=str, default="random_action", choices=["random_action", "visualize_transition"],
        help="Analysis mode to run"
    )
    p.add_argument("--manifest", type=Path, default=None, help="Manifest JSON")
    p.add_argument("--split", type=str, default=None, help="Split name in manifest")
    p.add_argument("--num_initial", type=int, default=100, help="# anchor states or # examples")
    p.add_argument("--num_random", type=int, default=50, help="# random actions per anchor (for random_action mode)")
    p.add_argument("--decoder_ckpt", type=Path, default=None, help="JEPA decoder ckpt (required for visualize_transition)")
    p.add_argument("--image_data_dir", type=Path, default=None, help="Path to image_data directory for ground truth viz")
    p.add_argument("--output_dir", type=Path, default=Path("ras_outputs"), help="Where to save results")
    p.add_argument("--device", type=str, default="cuda", help="Device (cuda|cpu)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) load VAE
    vae = load_vae_model(args.vae_ckpt, device)

    # 2) data loader (batch_size=1 suffices for random sampling)
    loader, _ = init_preprocessed_data_loader(
        processed_data_dir=str(args.data_dir),
        batch_size=1,
        num_workers=0,
        manifest_path=str(args.manifest) if args.manifest else None,
        split_name=args.split,
    )

    # 3) optional decoder
    decoder = None
    if args.decoder_ckpt:
        decoder = load_jepa_decoder(args.decoder_ckpt, device)

    # 4) analysis
    analyse_random_action_sensitivity(
        vae,
        loader,
        device,
        num_initial=args.num_initial,
        num_random=args.num_random,
        output_dir=args.output_dir,
        decoder=decoder,
        mode=args.mode,
        image_data_dir=args.image_data_dir,
    )
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()