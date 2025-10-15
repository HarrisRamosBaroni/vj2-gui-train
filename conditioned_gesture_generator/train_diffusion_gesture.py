import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter threading issues
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm

from training.dataloader import init_preprocessed_data_loader
from src.utils.logging import get_logger
from conditioned_gesture_generator.diffusion_gesture_model import (
    GestureDiffusionModel, GestureDiffusionModelLAM, create_diffusion_schedule, DDIMSampler,
    save_gesture_diffusion_checkpoint
)


def visualize_gesture_sequences(original_gestures, generated_gestures, title_prefix="", max_samples=6):
    """Create comprehensive visualization with time series + 2D trajectory plots.

    Args:
        original_gestures: Ground truth gestures [B, L, 3]
        generated_gestures: Generated gestures [B, L, 3]
        title_prefix: Prefix for plot titles
        max_samples: Maximum number of sequences to plot
    Returns:
        matplotlib figure
    """
    # Convert to numpy
    if isinstance(original_gestures, torch.Tensor):
        original_gestures = original_gestures.detach().cpu().numpy()
    if isinstance(generated_gestures, torch.Tensor):
        generated_gestures = generated_gestures.detach().cpu().numpy()

    batch_size = min(max_samples, original_gestures.shape[0])

    # Reshape to get 2 frames of 250 timesteps each (1x250 per frame)
    orig_reshaped = original_gestures.reshape(batch_size, -1, 250, 3)  # [B, frames, 250, 3]
    gen_reshaped = generated_gestures.reshape(batch_size, -1, 250, 3)   # [B, frames, 250, 3]

    num_frames = min(2, orig_reshaped.shape[1])  # Use up to 2 frames

    # Create comprehensive plot layout
    # Top: Time series (2 rows x batch_size cols)
    # Bottom: 2D trajectories (2 rows x 4 cols for 4 plots each of train/val)
    fig = plt.figure(figsize=(16, 12))

    # Time series plots (top half)
    for i in range(batch_size):
        # Original time series (row 1)
        ax_orig = plt.subplot(4, batch_size, i + 1)
        orig_data = original_gestures[i]
        timesteps = np.arange(len(orig_data))

        ax_orig.plot(timesteps, orig_data[:, 0], 'b-', alpha=0.8, label='X', linewidth=1)
        ax_orig.plot(timesteps, orig_data[:, 1], 'g-', alpha=0.8, label='Y', linewidth=1)
        ax_orig.plot(timesteps, orig_data[:, 2], 'r-', alpha=0.8, label='Touch', linewidth=1)

        # Highlight touch regions
        touch_mask = orig_data[:, 2] > 0.5
        if np.any(touch_mask):
            ax_orig.fill_between(timesteps, 0, 1, where=touch_mask, alpha=0.2, color='red')

        ax_orig.set_title(f"{title_prefix} Orig {i+1}", fontsize=8)
        ax_orig.set_ylim(-0.1, 1.1)
        ax_orig.tick_params(labelsize=6)
        if i == 0:
            ax_orig.legend(fontsize=6)
            ax_orig.set_ylabel("Value", fontsize=7)
        ax_orig.grid(True, alpha=0.3)

        # Generated time series (row 2)
        ax_gen = plt.subplot(4, batch_size, batch_size + i + 1)
        gen_data = generated_gestures[i]

        ax_gen.plot(timesteps, gen_data[:, 0], 'b-', alpha=0.8, label='X', linewidth=1)
        ax_gen.plot(timesteps, gen_data[:, 1], 'g-', alpha=0.8, label='Y', linewidth=1)
        ax_gen.plot(timesteps, gen_data[:, 2], 'r-', alpha=0.8, label='Touch', linewidth=1)

        # Highlight touch regions
        touch_mask_gen = gen_data[:, 2] > 0.5
        if np.any(touch_mask_gen):
            ax_gen.fill_between(timesteps, 0, 1, where=touch_mask_gen, alpha=0.2, color='red')

        ax_gen.set_title(f"{title_prefix} Gen {i+1}", fontsize=8)
        ax_gen.set_ylim(-0.1, 1.1)
        ax_gen.tick_params(labelsize=6)
        ax_gen.set_xlabel("Timestep", fontsize=7)
        if i == 0:
            ax_gen.legend(fontsize=6)
            ax_gen.set_ylabel("Value", fontsize=7)
        ax_gen.grid(True, alpha=0.3)

    # 2D trajectory plots (bottom half) - 4 plots each for train/val
    plot_idx = 0
    for i in range(min(4, batch_size)):  # Up to 4 trajectory plots
        for frame_idx in range(num_frames):  # Up to 2 frames
            if plot_idx >= 8:  # Max 8 trajectory plots (4 per row)
                break

            # Calculate subplot position (rows 3-4, 8 columns total)
            row = 3 if plot_idx < 4 else 4
            col = (plot_idx % 4) + 1

            ax_traj = plt.subplot(4, 4, (row - 1) * 4 + col)

            # Extract single frame trajectory (250 timesteps)
            orig_frame = orig_reshaped[i, frame_idx]  # [250, 3]
            gen_frame = gen_reshaped[i, frame_idx]    # [250, 3]

            # Plot original trajectory - only when pen is down (p > 0.5)
            orig_pen_down = orig_frame[:, 2] > 0.5
            idx = 0
            first_orig_segment = True
            while idx < len(orig_frame):
                if orig_pen_down[idx]:
                    # Find continuous pen-down segment
                    start_idx = idx
                    while idx < len(orig_frame) and orig_pen_down[idx]:
                        idx += 1
                    end_idx = idx
                    if end_idx > start_idx + 1:  # Need at least 2 points to draw a line
                        ax_traj.plot(orig_frame[start_idx:end_idx, 0], orig_frame[start_idx:end_idx, 1],
                                   'b-', alpha=0.7, linewidth=1.5,
                                   label='Original' if first_orig_segment else "")
                        first_orig_segment = False
                else:
                    idx += 1

            # Plot generated trajectory - only when pen is down (p > 0.5)
            gen_pen_down = gen_frame[:, 2] > 0.5
            idx = 0
            first_gen_segment = True
            while idx < len(gen_frame):
                if gen_pen_down[idx]:
                    # Find continuous pen-down segment
                    start_idx = idx
                    while idx < len(gen_frame) and gen_pen_down[idx]:
                        idx += 1
                    end_idx = idx
                    if end_idx > start_idx + 1:  # Need at least 2 points to draw a line
                        ax_traj.plot(gen_frame[start_idx:end_idx, 0], gen_frame[start_idx:end_idx, 1],
                                   'r--', alpha=0.7, linewidth=1.5,
                                   label='Generated' if first_gen_segment else "")
                        first_gen_segment = False
                else:
                    idx += 1

            # Mark start/end points
            ax_traj.scatter(orig_frame[0, 0], orig_frame[0, 1], c='blue', s=30, marker='o', alpha=0.8)
            ax_traj.scatter(orig_frame[-1, 0], orig_frame[-1, 1], c='blue', s=30, marker='s', alpha=0.8)
            ax_traj.scatter(gen_frame[0, 0], gen_frame[0, 1], c='red', s=20, marker='o', alpha=0.8)
            ax_traj.scatter(gen_frame[-1, 0], gen_frame[-1, 1], c='red', s=20, marker='s', alpha=0.8)

            ax_traj.set_title(f"2D Traj S{i+1}F{frame_idx+1}", fontsize=8)
            ax_traj.set_xlabel("X", fontsize=7)
            ax_traj.set_ylabel("Y", fontsize=7)
            ax_traj.tick_params(labelsize=6)
            ax_traj.grid(True, alpha=0.3)
            # Cap canvas to [0, 1] range for proper visualization
            ax_traj.set_xlim(0, 1)
            ax_traj.set_ylim(0, 1)
            ax_traj.set_aspect('equal')

            if plot_idx == 0:
                ax_traj.legend(fontsize=6)

            plot_idx += 1

    plt.tight_layout(pad=1.0)
    return fig


def visualize_sparse_denoising_steps(trainer, gesture_sequences, latent_frames, sequence_length, max_samples=2):
    """Visualize the sparse 6-step denoising process showing intermediate states.

    Args:
        trainer: DiffusionTrainer instance
        gesture_sequences: Ground truth gestures [B, L, 3]
        latent_frames: Visual conditioning [B, sequence_length+1, N, D]
        sequence_length: Number of gesture segments
        max_samples: Number of samples to visualize (default 2)
    Returns:
        matplotlib figure showing denoising progression
    """
    trainer.model.eval()

    # Take only the first few samples to avoid clutter
    B = min(max_samples, gesture_sequences.shape[0])
    gesture_sequences = gesture_sequences[:B]
    latent_frames = latent_frames[:B]

    L = gesture_sequences.shape[1]
    device = trainer.device

    # Create sparse inference schedule
    inference_timesteps = trainer.create_sparse_inference_schedule(num_steps=6)

    # Manual DDIM sampling to capture intermediate states
    shape = (B, L, 3)

    # Start with pure noise for absolute coordinates, zero for pen channel
    x_t = torch.randn(shape, device=device)
    x_t[..., 2] = 0.0  # Zero out pen channel

    # Store intermediate states
    intermediate_states = [x_t.clone().cpu()]
    timestep_labels = [f"t={inference_timesteps[0]} (noise)"]

    # Create timestep schedule for 6 steps
    num_train_timesteps = len(trainer.diffusion_schedule['betas'])
    timesteps = torch.linspace(num_train_timesteps - 1, 0, 6, dtype=torch.long)
    timesteps = timesteps.to(device)

    # Manual denoising loop
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B)

            # Predict noise with classifier-free guidance
            if trainer.args.cfg_scale > 1.0 and latent_frames is not None:
                # Conditional prediction
                output_cond = trainer.model(x_t, t_batch, latent_frames, sequence_length)
                # Unconditional prediction
                output_uncond = trainer.model(x_t, t_batch, None, None)
                # Apply classifier-free guidance
                coord_noise_pred = output_cond['coordinates'] + trainer.args.cfg_scale * (output_cond['coordinates'] - output_uncond['coordinates'])
            else:
                output = trainer.model(x_t, t_batch, latent_frames, sequence_length)
                coord_noise_pred = output['coordinates']

            # DDIM update
            alpha_t = trainer.diffusion_schedule['alphas_cumprod'][t]
            if i < len(timesteps) - 1:
                alpha_t_prev = trainer.diffusion_schedule['alphas_cumprod'][timesteps[i + 1]]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)

            # Extract coordinates for DDIM update
            x_t_coords = x_t[..., :2]  # [B, L, 2]

            # Predict x_0 for coordinates
            x_0_pred_coords = (x_t_coords - torch.sqrt(1 - alpha_t) * coord_noise_pred) / torch.sqrt(alpha_t)

            # DDIM update formula
            if i < len(timesteps) - 1:
                eta = 0.0  # Deterministic
                sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
                noise = torch.randn_like(x_t_coords) if eta > 0 else 0

                x_t_coords_next = (torch.sqrt(alpha_t_prev) * x_0_pred_coords +
                                  torch.sqrt(1 - alpha_t_prev - sigma_t**2) * coord_noise_pred +
                                  sigma_t * noise)
            else:
                x_t_coords_next = x_0_pred_coords

            # Update x_t with new coordinates
            x_t = torch.cat([x_t_coords_next, x_t[..., 2:3]], dim=-1)

            # Store intermediate state
            intermediate_states.append(x_t.clone().cpu())
            timestep_labels.append(f"t={t.item()}")

    # Add final pen prediction
    clean_coords = x_t[..., :2]
    clean_coords_output = torch.cat([clean_coords, torch.zeros(B, L, 1, device=device)], dim=-1)
    dummy_t = torch.zeros(B, device=device)
    pen_logits = trainer.model.forward_pen_prediction(clean_coords_output, dummy_t, latent_frames, sequence_length)
    pen_probs = torch.sigmoid(pen_logits)
    final_output = torch.cat([clean_coords, pen_probs], dim=-1)

    intermediate_states.append(final_output.cpu())
    timestep_labels.append("Final (+ pen)")

    # Create visualization
    num_steps = len(intermediate_states)
    fig = plt.figure(figsize=(20, 4 * B))

    for sample_idx in range(B):
        for step_idx, (state, label) in enumerate(zip(intermediate_states, timestep_labels)):
            # 2D trajectory plot
            ax = plt.subplot(B, num_steps, sample_idx * num_steps + step_idx + 1)

            sample_data = state[sample_idx]  # [L, 3]

            if step_idx < len(intermediate_states) - 1:
                # For intermediate steps, just plot coordinates (no pen info)
                ax.scatter(sample_data[:, 0], sample_data[:, 1], c='blue', s=1, alpha=0.6)
                ax.set_title(f"S{sample_idx+1} {label}", fontsize=10)
            else:
                # For final step, plot with pen information
                coords = sample_data[:, :2]
                pen_down = sample_data[:, 2] > 0.5

                # Plot trajectory with pen down/up distinction
                if torch.any(pen_down):
                    pen_down_coords = coords[pen_down]
                    if len(pen_down_coords) > 1:
                        ax.plot(pen_down_coords[:, 0], pen_down_coords[:, 1], 'b-', alpha=0.8, linewidth=2, label='Pen down')

                ax.scatter(coords[:, 0], coords[:, 1], c=sample_data[:, 2], cmap='RdYlBu_r', s=2, alpha=0.8)
                ax.set_title(f"S{sample_idx+1} {label}", fontsize=10)
                if sample_idx == 0 and step_idx == len(intermediate_states) - 1:
                    ax.legend(fontsize=8)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

            if sample_idx == B - 1:
                ax.set_xlabel("X coordinate", fontsize=9)
            if step_idx == 0:
                ax.set_ylabel(f"Sample {sample_idx+1}\nY coordinate", fontsize=9)

    plt.suptitle(f"Sparse 6-Step Denoising Process (CFG={trainer.args.cfg_scale})", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    trainer.model.train()
    return fig


class DiffusionTrainer:
    """Trainer for gesture diffusion model."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.logger = get_logger()

        # Set seeds for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Visual embeddings come from VJepa preprocessing - no LAM model needed
        self.logger.info("Using preprocessed VJepa embeddings for conditioning")

        # --- Initialize diffusion model ---
        if getattr(args, 'lam', False):
            # LAM-conditioned model
            if not hasattr(args, 'lam_checkpoint') or not args.lam_checkpoint:
                raise ValueError("--lam_checkpoint required when using --lam flag")

            model_args = {
                "action_dim": getattr(args, 'action_dim', 128),
                "d_model": args.d_model,
                "channels": args.unet_channels,
                "time_embed_dim": args.time_embed_dim,
                "max_sequence_length": args.max_sequence_length,
                "cfg_dropout_prob": args.cfg_dropout_prob,
                "lam_checkpoint_path": args.lam_checkpoint,
            }
            self.model = GestureDiffusionModelLAM(**model_args).to(self.device)
            self.logger.info(f"LAM-conditioned diffusion model instantiated successfully")
            self.logger.info(f"Using LAM checkpoint: {args.lam_checkpoint}")
            self.is_lam_model = True
        else:
            # Visual-conditioned model (original)
            model_args = {
                "d_latent": args.d_latent,
                "d_model": args.d_model,
                "channels": args.unet_channels,  # Note: parameter name is 'channels' not 'unet_channels'
                "time_embed_dim": args.time_embed_dim,
                "max_sequence_length": args.max_sequence_length,
                "cfg_dropout_prob": args.cfg_dropout_prob,
            }
            self.model = GestureDiffusionModel(**model_args).to(self.device)
            self.logger.info("Visual-conditioned diffusion model instantiated successfully")
            self.is_lam_model = False

        # --- Create diffusion schedule ---
        self.diffusion_schedule = create_diffusion_schedule(
            num_timesteps=args.num_timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end
        )
        # Move schedule to device
        for key in self.diffusion_schedule:
            self.diffusion_schedule[key] = self.diffusion_schedule[key].to(self.device)

        # --- Initialize DDIM sampler ---
        self.sampler = DDIMSampler(self.model, self.diffusion_schedule, device=self.device)

        # --- Optimizer ---
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.num_epochs,
            eta_min=args.learning_rate * 0.1
        )

        # --- Dataloaders ---
        self.train_loader, _ = init_preprocessed_data_loader(
            processed_data_dir=args.processed_data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            manifest_path=args.manifest,
            split_name='train',
        )
        self.val_loader, _ = init_preprocessed_data_loader(
            processed_data_dir=args.processed_data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            manifest_path=args.manifest,
            split_name='validation',
        )

        # --- Checkpointing and Logging ---
        self.run_dir = Path(args.output_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}_diffusion"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.global_step = 0

        if args.wandb_project:
            run_name = args.wandb_run_name if args.wandb_run_name else self.run_dir.name
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
            wandb.watch(self.model, log="gradients", log_freq=100)
            self.logger.info(f"W&B initialized for run: {run_name}")

        # --- Resume from checkpoint if specified ---
        self.start_epoch = 0
        if args.resume_from:
            self.logger.info(f"Resuming training from checkpoint: {args.resume_from}")
            self._load_checkpoint(args.resume_from)

    def prepare_batch(self, sample):
        """Prepare a training batch with VJepa visual embeddings and variable sequence length."""
        visual_embeddings, ground_truth_actions = sample  # VJepa embeddings from preprocessing
        visual_embeddings = visual_embeddings.to(self.device, non_blocking=True)  # [B, T, N, D]
        ground_truth_actions = ground_truth_actions.to(self.device, non_blocking=True)  # [B, T-1, Traj, 3]

        B, T_minus_1, Traj, _ = ground_truth_actions.shape
        T = T_minus_1 + 1  # Total number of visual frames

        # Variable-length segment training: randomly sample sequence length from [1, N_max]
        # But use full length during overfit mode or validation
        if self.model.training and hasattr(self.args, 'max_sequence_length_frames') and not getattr(self.args, 'overfit', False):
            # N_max is the maximum number of gesture segments we want to use
            N_max = min(T_minus_1, self.args.max_sequence_length_frames)
            sequence_length = torch.randint(1, N_max + 1, (1,)).item()
        else:
            # Use full sequence during validation/inference/overfit
            sequence_length = T_minus_1

        # Extract the corresponding gesture sequences for the selected length
        # sequence_length refers to number of gesture segments (each Traj=250 steps)
        gesture_sequences = ground_truth_actions[:, :sequence_length].contiguous().view(B, sequence_length * Traj, 3)

        # Get latent frames for conditioning (need sequence_length + 1 frames)
        # We need frames [0, 1, ..., sequence_length] for conditioning gestures [0->1, 1->2, ..., (sequence_length-1)->sequence_length]
        latent_frames = visual_embeddings[:, :sequence_length + 1].contiguous()  # [B, sequence_length+1, N, D]

        # Apply classifier-free guidance dropout if training (but not during overfit)
        if self.model.training and not getattr(self.args, 'overfit', False):
            latent_frames = self.model.apply_cfg_dropout(latent_frames, training=True)

        return gesture_sequences, latent_frames, sequence_length

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add noise to clean gestures according to diffusion schedule.
        Absolute coordinate diffusion: we diffuse (x, y) directly and never the pen channel.

        Args:
            x_0: Clean gesture sequences [B, L, 3] with absolute coordinates
            t: Timesteps [B]
        Returns:
            x_t: Noisy gesture sequences [B, L, 3] (absolute coords noised, pen ZEROED)
            coordinate_noise: Added noise [B, L, 2] (for absolute coordinates)
            pen_state: Original pen state [B, L, 1] (for loss computation only)
        """
        # Split absolute coordinates and pen state
        abs_coords = x_0[..., :2]  # [B, L, 2], absolute coordinates
        pen_state = x_0[..., 2:3]  # [B, L, 1]

        # ---- Diffuse absolute coordinates directly ----
        coordinate_noise = torch.randn_like(abs_coords)

        sqrt_alphas_cumprod_t = self.diffusion_schedule['sqrt_alphas_cumprod'][t]
        sqrt_one_minus_alphas_cumprod_t = self.diffusion_schedule['sqrt_one_minus_alphas_cumprod'][t]

        # Broadcast to match coords shape
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None]

        # Apply noise to absolute coordinates
        noisy_coords = sqrt_alphas_cumprod_t * abs_coords + sqrt_one_minus_alphas_cumprod_t * coordinate_noise

        # pen channel is ZERO to avoid leakage
        zero_pen_channel = torch.zeros_like(pen_state)

        # x_t carries noisy absolute coordinates in the first two channels
        x_t = torch.cat([noisy_coords, zero_pen_channel], dim=-1)  # [B, L, 3]

        return x_t, coordinate_noise, pen_state

    def create_sparse_inference_schedule(self, num_steps: int = 6) -> list:
        """Create sparse inference schedule focused on final denoising steps.

        Args:
            num_steps: Number of inference steps (default 6)

        Returns:
            List of timesteps concentrated toward the end of denoising
        """
        total_timesteps = len(self.diffusion_schedule['betas'])

        # Create schedule concentrated toward final steps (lower timestep values)
        # Example: [950, 800, 600, 300, 100, 0] for 1000 total timesteps
        if num_steps == 6:
            # Hand-tuned schedule focusing on final steps
            ratios = [0.95, 0.80, 0.60, 0.30, 0.10, 0.0]
        else:
            # General case: quadratic schedule concentrated at the end
            ratios = [(1.0 - (i / (num_steps - 1))**2) for i in range(num_steps - 1)]
            ratios.append(0.0)  # Always end at 0

        timesteps = [int(ratio * (total_timesteps - 1)) for ratio in ratios]
        return timesteps

    @torch.no_grad()
    def compute_reconstruction_loss(self, gesture_sequences: torch.Tensor,
                                   latent_frames: torch.Tensor,
                                   sequence_length: int) -> Dict[str, float]:
        """Compute L1 reconstruction loss from multi-step denoising.

        Args:
            gesture_sequences: Ground truth gestures [B, L, 3]
            latent_frames: Visual conditioning [B, sequence_length+1, N, D]
            sequence_length: Number of gesture segments

        Returns:
            Dictionary with reconstruction loss components
        """
        B, L, _ = gesture_sequences.shape

        # Create sparse inference schedule (6 steps focused on final denoising)
        inference_timesteps = self.create_sparse_inference_schedule(num_steps=6)

        # Generate complete sequence using sparse DDIM sampling
        shape = (B, L, 3)
        generated_sequences = self.sampler.sample(
            shape=shape,
            latent_frames=latent_frames,
            sequence_length=sequence_length,
            num_inference_steps=6,  # Use exactly 6 steps
            cfg_scale=self.args.cfg_scale,
            eta=0.0  # Deterministic DDIM
        )

        # Split ground truth and generated into coordinates and pen states
        gt_coords = gesture_sequences[..., :2]  # [B, L, 2]
        gt_pen = gesture_sequences[..., 2:3]    # [B, L, 1]

        gen_coords = generated_sequences[..., :2]  # [B, L, 2]
        gen_pen = generated_sequences[..., 2:3]    # [B, L, 1]

        # Create temporal union mask: pen down in EITHER ground truth OR generated
        # This focuses loss on actual drawing regions while ignoring pen-up movements
        gt_pen_down = (gt_pen > 0.5).float()     # [B, L, 1]
        gen_pen_down = (gen_pen > 0.5).float()   # [B, L, 1]
        union_mask = torch.clamp(gt_pen_down + gen_pen_down, 0.0, 1.0)  # [B, L, 1]

        # L1 coordinate loss (masked by temporal union)
        coord_diff = torch.abs(gen_coords - gt_coords)  # [B, L, 2]
        masked_coord_loss = coord_diff * union_mask  # Apply mask to both x and y
        coord_l1_loss = masked_coord_loss.sum() / (union_mask.sum() + 1e-8)  # Normalize by valid regions

        # BCE pen state loss (unmasked) - binary classification loss
        pen_bce_loss = F.binary_cross_entropy(gen_pen, gt_pen, reduction='mean')

        # Combined reconstruction loss
        l_recon = coord_l1_loss + pen_bce_loss

        # Compute statistics for logging
        mask_coverage = union_mask.mean().item()  # Fraction of timesteps with pen down
        gt_pen_ratio = gt_pen_down.mean().item()
        gen_pen_ratio = gen_pen_down.mean().item()

        return {
            "l_recon": l_recon.item(),
            "coord_l1_loss": coord_l1_loss.item(),
            "pen_bce_loss": pen_bce_loss.item(),
            "mask_coverage": mask_coverage,
            "gt_pen_ratio": gt_pen_ratio,
            "gen_pen_ratio": gen_pen_ratio
        }

    def train_step(self, sample):
        """Single training step with two separate forward passes."""
        gesture_sequences, latent_frames, sequence_length = self.prepare_batch(sample)

        B, L, _ = gesture_sequences.shape

        # Sample random timesteps
        t = torch.randint(0, len(self.diffusion_schedule['betas']), (B,), device=self.device)

        # --- PASS 1: Diffusion training (coordinate denoising) ---
        # Add noise to gestures (only coordinates)
        x_t, coordinate_noise, pen_state = self.add_noise(gesture_sequences, t)

        # Forward pass with noisy coordinates - only use coordinate prediction
        model_output = self.model(x_t, t, latent_frames, sequence_length)

        # Compute coordinate loss only
        coord_loss = self.model.compute_coordinate_loss(model_output, coordinate_noise)

        # --- PASS 2: Clean pen prediction ---
        # Forward pass with clean coordinates for pen prediction
        pen_logits = self.model.forward_pen_prediction(gesture_sequences, t, latent_frames, sequence_length)

        # Compute pen loss
        pen_loss_dict = self.model.compute_pen_loss(pen_logits, pen_state)
        pen_loss = pen_loss_dict['pen_loss']
        pos_weight = pen_loss_dict['pos_weight']

        # --- PASS 3: Reconstruction Loss (optional) ---
        recon_weight = getattr(self.args, 'recon_weight', 0.0)
        l_recon = torch.tensor(0.0, device=self.device)
        recon_metrics = {}

        if recon_weight > 0.0:
            # Compute reconstruction loss through multi-step denoising
            self.model.eval()  # Set to eval mode for deterministic generation
            with torch.no_grad():
                recon_metrics = self.compute_reconstruction_loss(
                    gesture_sequences, latent_frames, sequence_length
                )
                l_recon = torch.tensor(recon_metrics['l_recon'], device=self.device)
            self.model.train()  # Switch back to training mode

        # --- Combined loss and backward pass ---
        coordinate_weight = getattr(self.args, 'coordinate_weight', 1.0)
        pen_weight = getattr(self.args, 'pen_weight', 1.0)
        total_loss = coordinate_weight * coord_loss + pen_weight * pen_loss + recon_weight * l_recon

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

        self.optimizer.step()
        self.global_step += 1

        # Log pen state class imbalance monitoring
        pen_state_mean = pen_state.mean().item()

        # Prepare return metrics
        metrics = {
            "loss": total_loss.item(),
            "coordinate_loss": coord_loss.item(),
            "pen_loss": pen_loss.item(),
            "pen_state_mean": pen_state_mean,
            "pos_weight": pos_weight.item(),
            "sequence_length": sequence_length
        }

        # Add reconstruction metrics if computed
        if recon_weight > 0.0:
            metrics.update({
                "l_recon": recon_metrics['l_recon'],
                "coord_l1_loss": recon_metrics['coord_l1_loss'],
                "pen_bce_loss": recon_metrics['pen_bce_loss'],
                "mask_coverage": recon_metrics['mask_coverage'],
                "gt_pen_ratio": recon_metrics['gt_pen_ratio'],
                "gen_pen_ratio": recon_metrics['gen_pen_ratio']
            })
        else:
            # Add zero placeholders when reconstruction is disabled
            metrics.update({
                "l_recon": 0.0,
                "coord_l1_loss": 0.0,
                "pen_bce_loss": 0.0,
                "mask_coverage": 0.0,
                "gt_pen_ratio": 0.0,
                "gen_pen_ratio": 0.0
            })

        return metrics

    @torch.no_grad()
    def validate(self):
        """Validation loop."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for sample in self.val_loader:
            gesture_sequences, latent_frames, sequence_length = self.prepare_batch(sample)

            B, L, _ = gesture_sequences.shape

            # Sample random timesteps
            t = torch.randint(0, len(self.diffusion_schedule['betas']), (B,), device=self.device)

            # --- PASS 1: Diffusion validation (coordinate denoising) ---
            # Add noise to gestures (only coordinates)
            x_t, coordinate_noise, pen_state = self.add_noise(gesture_sequences, t)

            # Forward pass with noisy coordinates - only use coordinate prediction
            model_output = self.model(x_t, t, latent_frames, sequence_length)

            # Compute coordinate loss only
            coord_loss = self.model.compute_coordinate_loss(model_output, coordinate_noise)

            # --- PASS 2: Clean pen prediction ---
            # Forward pass with clean coordinates for pen prediction
            pen_logits = self.model.forward_pen_prediction(gesture_sequences, t, latent_frames, sequence_length)

            # Compute pen loss
            pen_loss_dict = self.model.compute_pen_loss(pen_logits, pen_state)
            pen_loss = pen_loss_dict['pen_loss']

            # --- Reconstruction Loss (optional) ---
            recon_weight = getattr(self.args, 'recon_weight', 0.0)
            l_recon = torch.tensor(0.0, device=self.device)

            if recon_weight > 0.0:
                # Compute reconstruction loss for validation
                recon_metrics = self.compute_reconstruction_loss(
                    gesture_sequences, latent_frames, sequence_length
                )
                l_recon = torch.tensor(recon_metrics['l_recon'], device=self.device)

            # --- Combined loss ---
            coordinate_weight = getattr(self.args, 'coordinate_weight', 1.0)
            pen_weight = getattr(self.args, 'pen_weight', 1.0)
            batch_total_loss = coordinate_weight * coord_loss + pen_weight * pen_loss + recon_weight * l_recon

            total_loss += batch_total_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # Generate comprehensive validation samples
        if self.args.wandb_project:
            self.generate_validation_samples()

        self.model.train()
        return avg_loss

    @torch.no_grad()
    def generate_validation_samples(self, num_samples: int = 6,
                                   custom_embeddings: Optional[torch.Tensor] = None,
                                   custom_gestures: Optional[torch.Tensor] = None,
                                   sequence_length: Optional[int] = None,
                                   log_prefix: str = "val"):
        """Generate comprehensive samples with DDIM 50 steps showing both training and validation data."""
        self.model.eval()
        self.logger.info("Generating samples for visualization (training + validation)...")

        # Use custom embeddings if provided (for overfit mode)
        if custom_embeddings is not None:
            if custom_gestures is not None:
                # Use the real gesture data for overfit visualization
                train_samples = [(custom_embeddings, custom_gestures)]
                val_samples = [(custom_embeddings, custom_gestures)]  # Same sample for overfit
                self.logger.info(f"Using custom embeddings and REAL gestures for overfit test (showing the exact overfitted sample)")
            else:
                # Fallback to dummy gestures if no real gestures provided
                B, T = custom_embeddings.shape[:2]
                dummy_gestures = torch.zeros(B, T-1, 250, 3)  # Default trajectory length
                train_samples = [(custom_embeddings, dummy_gestures)]
                val_samples = [(custom_embeddings, dummy_gestures)]  # Same sample for overfit
                self.logger.info(f"Using custom embeddings with dummy gestures for overfit test")
        else:
            # Collect samples from both training and validation sets
            samples_per_set = num_samples // 2

            # Get training samples
            train_samples = []
            for i, sample in enumerate(self.train_loader):
                if len(train_samples) >= samples_per_set:
                    break
                train_samples.append(sample)

            # Get validation samples
            val_samples = []
            for i, sample in enumerate(self.val_loader):
                if len(val_samples) >= samples_per_set:
                    break
                val_samples.append(sample)

            self.logger.info(f"Collected {len(train_samples)} training samples and {len(val_samples)} validation samples")

        # Process both training and validation samples
        all_ground_truth = []
        all_generated = []
        all_conditioning_info = []
        all_sample_types = []  # Track whether sample is from train or val

        # Process training samples
        for batch_idx, sample in enumerate(train_samples):
            if len(all_ground_truth) >= num_samples:
                break

            # Prepare batch (this sets model to eval mode, so no CFG dropout)
            self.model.eval()
            gesture_sequences, latent_frames, batch_sequence_length = self.prepare_batch(sample)

            # Use custom sequence length if provided (for overfit mode)
            if sequence_length is not None:
                batch_sequence_length = sequence_length

            # Take samples up to our limit
            remaining_slots = num_samples - len(all_ground_truth)
            batch_size = min(gesture_sequences.shape[0], remaining_slots)

            gt_batch = gesture_sequences[:batch_size]
            latent_batch = latent_frames[:batch_size]

            # Generate using DDIM with 50 inference steps
            shape = (batch_size, gt_batch.shape[1], 3)  # Use same length as ground truth
            generated_batch = self.sampler.sample(
                shape=shape,
                latent_frames=latent_batch,
                sequence_length=batch_sequence_length,
                num_inference_steps=50,  # Always use 50 steps for validation
                cfg_scale=self.args.cfg_scale,
                eta=0.0  # Deterministic DDIM
            )

            # Store results
            all_ground_truth.append(gt_batch.cpu())
            all_generated.append(generated_batch.cpu())
            all_conditioning_info.extend([batch_sequence_length] * batch_size)
            all_sample_types.extend(['train'] * batch_size)

        # Process validation samples
        for batch_idx, sample in enumerate(val_samples):
            if len(all_ground_truth) >= num_samples:
                break

            # Prepare batch (this sets model to eval mode, so no CFG dropout)
            self.model.eval()
            gesture_sequences, latent_frames, batch_sequence_length = self.prepare_batch(sample)

            # Use custom sequence length if provided (for overfit mode)
            if sequence_length is not None:
                batch_sequence_length = sequence_length

            # Take samples up to our limit
            remaining_slots = num_samples - len(all_ground_truth)
            batch_size = min(gesture_sequences.shape[0], remaining_slots)

            gt_batch = gesture_sequences[:batch_size]
            latent_batch = latent_frames[:batch_size]

            # Generate using DDIM with 50 inference steps
            shape = (batch_size, gt_batch.shape[1], 3)  # Use same length as ground truth
            generated_batch = self.sampler.sample(
                shape=shape,
                latent_frames=latent_batch,
                sequence_length=batch_sequence_length,
                num_inference_steps=50,  # Always use 50 steps for validation
                cfg_scale=self.args.cfg_scale,
                eta=0.0  # Deterministic DDIM
            )

            # Store results
            all_ground_truth.append(gt_batch.cpu())
            all_generated.append(generated_batch.cpu())
            all_conditioning_info.extend([batch_sequence_length] * batch_size)
            all_sample_types.extend(['val'] * batch_size)

        if len(all_ground_truth) == 0:
            self.logger.warning("No validation samples generated")
            return

        # Concatenate all results
        ground_truth_tensor = torch.cat(all_ground_truth, dim=0)[:num_samples]
        generated_tensor = torch.cat(all_generated, dim=0)[:num_samples]

        # Create comprehensive visualization with sample type info
        if custom_embeddings is not None:
            title_prefix = f"Overfit Test - Exact Sample Being Trained (DDIM-50, Step {self.global_step})"
        else:
            title_prefix = f"Train+Val Samples (DDIM-50, Step {self.global_step})"

        fig = visualize_gesture_sequences(
            ground_truth_tensor,
            generated_tensor,
            title_prefix=title_prefix,
            max_samples=num_samples
        )

        # Generate sparse denoising visualization (only for first few samples)
        if len(all_ground_truth) > 0:
            # Use first batch for denoising visualization
            first_gt_batch = all_ground_truth[0][:2]  # Take first 2 samples
            first_latent_batch = None
            first_seq_length = all_conditioning_info[0] if all_conditioning_info else 2

            # Get the corresponding latent frames from the first sample
            if len(train_samples) > 0:
                first_sample = train_samples[0]
                _, first_latent_frames, _ = self.prepare_batch(first_sample)
                first_latent_batch = first_latent_frames[:2]  # Match sample count
            elif len(val_samples) > 0:
                first_sample = val_samples[0]
                _, first_latent_frames, _ = self.prepare_batch(first_sample)
                first_latent_batch = first_latent_frames[:2]  # Match sample count

            if first_latent_batch is not None:
                sparse_fig = visualize_sparse_denoising_steps(
                    self, first_gt_batch, first_latent_batch, first_seq_length, max_samples=2
                )
            else:
                sparse_fig = None
        else:
            sparse_fig = None

        # Log to wandb with additional metrics
        log_dict = {
            f"{log_prefix}_samples": wandb.Image(fig),
            f"{log_prefix}/num_generated_samples": len(generated_tensor),
            f"{log_prefix}/avg_sequence_length": np.mean(all_conditioning_info),
            "global_step": self.global_step
        }

        # Add sparse denoising visualization if available
        if sparse_fig is not None:
            log_dict[f"{log_prefix}_sparse_denoising"] = wandb.Image(sparse_fig)

        # Compute gesture statistics for validation
        gt_mean_x = ground_truth_tensor[..., 0].mean().item()
        gt_mean_y = ground_truth_tensor[..., 1].mean().item()
        gt_mean_touch = ground_truth_tensor[..., 2].mean().item()

        gen_mean_x = generated_tensor[..., 0].mean().item()
        gen_mean_y = generated_tensor[..., 1].mean().item()
        gen_mean_touch = generated_tensor[..., 2].mean().item()

        log_dict.update({
            f"{log_prefix}/gt_mean_x": gt_mean_x,
            f"{log_prefix}/gt_mean_y": gt_mean_y,
            f"{log_prefix}/gt_mean_touch": gt_mean_touch,
            f"{log_prefix}/gen_mean_x": gen_mean_x,
            f"{log_prefix}/gen_mean_y": gen_mean_y,
            f"{log_prefix}/gen_mean_touch": gen_mean_touch,
            f"{log_prefix}/mse_x": F.mse_loss(generated_tensor[..., 0], ground_truth_tensor[..., 0]).item(),
            f"{log_prefix}/mse_y": F.mse_loss(generated_tensor[..., 1], ground_truth_tensor[..., 1]).item(),
            f"{log_prefix}/mse_touch": F.mse_loss(generated_tensor[..., 2], ground_truth_tensor[..., 2]).item(),
        })

        # Add sample type information to logs
        if custom_embeddings is not None:
            log_dict[f"{log_prefix}/sample_info"] = "overfit_exact_sample"
            self.logger.info(f"Generated samples for OVERFIT TEST - showing the exact sample being trained")
        else:
            num_train = len([t for t in all_sample_types if t == 'train'])
            num_val = len([t for t in all_sample_types if t == 'val'])
            log_dict[f"{log_prefix}/num_train_samples"] = num_train
            log_dict[f"{log_prefix}/num_val_samples"] = num_val
            self.logger.info(f"Generated {len(generated_tensor)} samples: {num_train} train + {num_val} val with DDIM-50")

        wandb.log(log_dict)
        plt.close(fig)
        if sparse_fig is not None:
            plt.close(sparse_fig)
        self.model.train()

    def _save_checkpoint(self, tag: str, val_loss: float = None):
        """Save model checkpoint with proper versioning."""
        checkpoint_path = self.run_dir / f"diffusion_{tag}.pt"

        # Prepare trainer state
        trainer_state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "epoch": self.start_epoch,
            "diffusion_schedule": self.diffusion_schedule,
        }
        if val_loss is not None:
            trainer_state["val_loss"] = val_loss

        # Get LAM checkpoint path for LAM models
        lam_checkpoint_path = None
        if self.is_lam_model:
            lam_checkpoint_path = getattr(self.args, 'lam_checkpoint', None)

        try:
            # Use versioned checkpoint saving
            save_gesture_diffusion_checkpoint(
                model=self.model,
                checkpoint_path=str(checkpoint_path),
                trainer_state=trainer_state,
                lam_checkpoint_path=lam_checkpoint_path
            )
            self.logger.info(f"✅ Checkpoint saved to {checkpoint_path}")

            # Create latest symlink
            latest_symlink = self.run_dir / "latest.pt"
            if latest_symlink.exists():
                latest_symlink.unlink()
            latest_symlink.symlink_to(checkpoint_path.name)

        except Exception as e:
            self.logger.error(f"❌ Failed to save checkpoint: {e}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model state
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info(" Model state loaded")

            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.logger.info(" Optimizer state loaded")

            # Load scheduler state
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.logger.info(" Scheduler state loaded")

            # Load training state
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            self.global_step = checkpoint.get("global_step", 0)

            # Calculate start epoch from global step
            batches_per_epoch = len(self.train_loader)
            self.start_epoch = self.global_step // batches_per_epoch

            self.logger.info(f" Training state loaded:")
            self.logger.info(f"  - Best val loss: {self.best_val_loss:.4f}")
            self.logger.info(f"  - Global step: {self.global_step}")
            self.logger.info(f"  - Starting from epoch: {self.start_epoch}")

        except Exception as e:
            self.logger.error(f"L Failed to load checkpoint: {e}")
            raise

    def training_loop(self):
        """Main training loop."""
        if self.args.overfit:
            self.logger.info("Starting overfit test mode...")
            self.overfit_training_loop()
            return

        self.logger.info("Starting diffusion training loop...")

        # Run initial validation
        self.logger.info("=== Running initial evaluation before training ===")
        initial_val_loss = self.validate()
        self.logger.info(f"Initial validation loss: {initial_val_loss:.4f}")

        for epoch in range(self.start_epoch, self.args.num_epochs):
            self.model.train()

            epoch_loss = 0
            num_batches = 0

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")

            # Calculate half-epoch checkpoint
            total_batches = len(self.train_loader)
            half_epoch_checkpoint = total_batches // 2

            for i, sample in enumerate(progress_bar):
                loss_dict = self.train_step(sample)

                epoch_loss += loss_dict["loss"]
                num_batches += 1

                # Update progress bar
                if i % 10 == 0:
                    progress_bar.set_postfix({"loss": f"{loss_dict['loss']:.4f}"})

                # Log to wandb
                if self.args.wandb_project and i % 20 == 0:
                    log_dict = {
                        "train/total_loss": loss_dict["loss"],
                        "train/coordinate_loss": loss_dict["coordinate_loss"],
                        "train/pen_loss": loss_dict["pen_loss"],
                        "train/lr": self.optimizer.param_groups[0]['lr'],
                        "train/sequence_length": loss_dict["sequence_length"],
                        "global_step": self.global_step,
                        "epoch": epoch
                    }

                    # Add reconstruction metrics if enabled
                    recon_weight = getattr(self.args, 'recon_weight', 0.0)
                    if recon_weight > 0.0:
                        log_dict.update({
                            "train/l_recon": loss_dict["l_recon"],
                            "train/coord_l1_loss": loss_dict["coord_l1_loss"],
                            "train/pen_bce_loss": loss_dict["pen_bce_loss"],
                            "train/mask_coverage": loss_dict["mask_coverage"],
                            "train/gt_pen_ratio": loss_dict["gt_pen_ratio"],
                            "train/gen_pen_ratio": loss_dict["gen_pen_ratio"],
                            "train/recon_weight": recon_weight
                        })

                    # Log FiLM magnitude histograms (similar to autoregressive model)
                    try:
                        if (hasattr(self.model, 'last_film_gamma') and
                            self.model.last_film_gamma is not None):

                            # Log norms (L2 norm per position)
                            gamma_norms = self.model.last_gamma_norm.flatten().cpu().numpy()
                            beta_norms = self.model.last_beta_norm.flatten().cpu().numpy()
                            log_dict["train/film_gamma_norms"] = wandb.Histogram(gamma_norms)
                            log_dict["train/film_beta_norms"] = wandb.Histogram(beta_norms)

                            # Log raw magnitude distributions
                            gamma_magnitudes = self.model.last_film_gamma.flatten().cpu().numpy()
                            beta_magnitudes = self.model.last_film_beta.flatten().cpu().numpy()
                            log_dict["train/film_gamma_magnitudes"] = wandb.Histogram(gamma_magnitudes)
                            log_dict["train/film_beta_magnitudes"] = wandb.Histogram(beta_magnitudes)

                    except Exception as e:
                        # Silently skip if FiLM values aren't available
                        pass

                    wandb.log(log_dict)

                # Half-epoch validation and visualization
                if i == half_epoch_checkpoint:
                    self.logger.info(f"=== Half-epoch checkpoint: {epoch+1}.5 ===")

                    # Run validation
                    val_loss = self.validate()
                    self.logger.info(f"Half-epoch validation loss: {val_loss:.4f}")

                    # Log half-epoch validation
                    if self.args.wandb_project:
                        wandb.log({
                            "val_half/loss": val_loss,
                            "val_half/epoch": epoch + 0.5,
                            "global_step": self.global_step
                        })

                    # Save best model if this is the best validation loss so far
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.logger.info(f"🏆 New best validation loss at half-epoch: {val_loss:.4f}. Saving checkpoint.")
                        self._save_checkpoint("best", val_loss)

            # Update learning rate
            self.scheduler.step()

            # Log epoch summary
            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            self.logger.info(f"--- Epoch {epoch+1} Complete ---")
            self.logger.info(f"Average Train Loss: {avg_train_loss:.4f}")

            # End-of-epoch validation (every epoch now)
            self.logger.info(f"=== End-of-epoch validation: {epoch+1} ===")
            val_loss = self.validate()
            self.logger.info(f"End-of-epoch validation loss: {val_loss:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.logger.info(f"🏆 New best validation loss at end-of-epoch: {val_loss:.4f}. Saving checkpoint.")
                self._save_checkpoint("best", val_loss)

            # Log end-of-epoch validation to wandb
            if self.args.wandb_project:
                wandb.log({
                    "val_end/loss": val_loss,
                    "val_end/best_loss": self.best_val_loss,
                    "val_end/epoch": epoch + 1,
                    "global_step": self.global_step
                })

            # Save periodic checkpoint
            if (epoch + 1) % self.args.save_freq == 0:
                self._save_checkpoint(f"epoch_{epoch+1}")

        self.logger.info("Training completed!")

    def overfit_training_loop(self):
        """Overfit test mode: train on single sample repeatedly."""
        self.logger.info("=== OVERFIT TEST MODE ===")
        self.logger.info("Training on single sample repeatedly...")

        # Get first sample from train loader and take only one sample
        self.model.train()
        first_batch = next(iter(self.train_loader))
        embeddings, gestures = first_batch

        # Take only the first sample and use full sequence length
        single_embeddings = embeddings[:1]  # [1, T, N, D] - just one sample
        single_gestures = gestures[:1]       # [1, T-1, Traj, 3] - just one sample
        single_sample = (single_embeddings, single_gestures)

        # Log initial info
        self.logger.info(f"Overfitting on single sample:")
        self.logger.info(f"  Embeddings shape: {single_embeddings.shape}")
        self.logger.info(f"  Gestures shape: {single_gestures.shape}")
        self.logger.info(f"  Sequence length: {single_gestures.shape[1]} (full length)")

        # Training loop on single sample
        total_steps = 0
        for epoch in range(self.args.num_epochs):
            # Train on the same single sample repeatedly
            for step in range(100):  # 100 steps per epoch on same sample
                loss_dict = self.train_step(single_sample)
                total_steps += 1

                # Log every 50 steps
                if total_steps % 50 == 0:
                    self.logger.info(f"Step {total_steps}: Loss = {loss_dict['loss']:.6f}, "
                                   f"Coord = {loss_dict['coordinate_loss']:.6f}, "
                                   f"Pen = {loss_dict['pen_loss']:.6f}")

                # Log to wandb
                if self.args.wandb_project and total_steps % 10 == 0:
                    log_dict = {
                        "overfit/total_loss": loss_dict["loss"],
                        "overfit/coordinate_loss": loss_dict["coordinate_loss"],
                        "overfit/pen_loss": loss_dict["pen_loss"],
                        "overfit/lr": self.optimizer.param_groups[0]['lr'],
                        "overfit/sequence_length": loss_dict["sequence_length"],
                        "overfit/step": total_steps,
                        "overfit/epoch": epoch
                    }

                    # Add reconstruction metrics if enabled
                    recon_weight = getattr(self.args, 'recon_weight', 0.0)
                    if recon_weight > 0.0:
                        log_dict.update({
                            "overfit/l_recon": loss_dict["l_recon"],
                            "overfit/coord_l1_loss": loss_dict["coord_l1_loss"],
                            "overfit/pen_bce_loss": loss_dict["pen_bce_loss"],
                            "overfit/mask_coverage": loss_dict["mask_coverage"],
                            "overfit/gt_pen_ratio": loss_dict["gt_pen_ratio"],
                            "overfit/gen_pen_ratio": loss_dict["gen_pen_ratio"],
                            "overfit/recon_weight": recon_weight
                        })

                    wandb.log(log_dict)

            # Update learning rate
            self.scheduler.step()

            # Log epoch summary
            self.logger.info(f"=== Overfit Epoch {epoch+1}/{self.args.num_epochs} Complete ===")
            self.logger.info(f"Last loss: {loss_dict['loss']:.6f}")

            # Generate and log samples every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"=== Generating overfit samples at epoch {epoch+1} ===")
                self.model.eval()
                with torch.no_grad():
                    # Use the same single sample embeddings for generation
                    embeddings_device = single_embeddings.to(self.device)

                    # Generate with full sequence length (no variable length for overfit test)
                    full_seq_len = single_gestures.shape[1]  # Use the full gesture sequence length
                    self.logger.info(f"Generating with full sequence length: {full_seq_len}")

                    samples = self.generate_validation_samples(
                        num_samples=1,
                        custom_embeddings=embeddings_device,
                        custom_gestures=single_gestures,
                        sequence_length=full_seq_len,
                        log_prefix="overfit"  # Keep same name, use global_step for timeline
                    )
                self.model.train()  # Switch back to training mode

        self.logger.info("Overfit training completed!")


def get_args():
    parser = argparse.ArgumentParser(description="Train Gesture Diffusion Model")

    # Data args
    parser.add_argument("--processed_data_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/diffusion_gesture")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume training from")

    # Model args
    parser.add_argument("--d_latent", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--unet_channels", type=int, nargs='+', default=[64, 128, 256, 512])
    parser.add_argument("--time_embed_dim", type=int, default=128)
    parser.add_argument("--max_sequence_length", type=int, default=1000)
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.1)
    parser.add_argument("--max_sequence_length_frames", type=int, default=4,
                       help="Maximum number of latent frames to use (N_max). Training randomly samples from [1, N_max]")

    # Diffusion args
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.5)

    # VJepa embeddings are used from preprocessing - no LAM model needed

    # Training args
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Loss weighting
    parser.add_argument("--coordinate_weight", type=float, default=1.0,
                       help="Weight for coordinate MSE loss in L = L_MSE(x,y) + gamma * L_BCE(p)")
    parser.add_argument("--pen_weight", type=float, default=1.0,
                       help="Weight for pen state BCE loss (gamma in the formula)")
    parser.add_argument("--recon_weight", type=float, default=0.0,
                       help="Weight for reconstruction loss from multi-step denoising (coord L1 + pen BCE). 0 = disabled to save VRAM.")

    # Logging and evaluation
    parser.add_argument("--wandb_project", type=str, default="diffusion-gesture-generator")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")

    # Overfit test mode
    parser.add_argument("--overfit", action="store_true",
                       help="Overfit test mode: train on single sample repeatedly, log generation only at end")

    # LAM conditioning
    parser.add_argument("--lam", action="store_true",
                       help="Use LAM-conditioned model instead of visual-conditioned model")
    parser.add_argument("--lam_checkpoint", type=str, default=None,
                       help="Path to LAM model checkpoint (required when --lam is used)")
    parser.add_argument("--action_dim", type=int, default=128,
                       help="Dimension of LAM action tokens (default: 128)")

    return parser.parse_args()


def main():
    args = get_args()
    try:
        trainer = DiffusionTrainer(args)
        trainer.training_loop()
        print("Training finished successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise


if __name__ == "__main__":
    main()