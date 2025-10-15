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
from conditioned_gesture_generator.direct_gen_gesture_model import (
    DirectGestureModel
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


class DirectGenTrainer:
    """Trainer for direct gesture generation model."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.logger = get_logger()

        # Set seeds for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Visual embeddings come from VJepa preprocessing
        self.logger.info("Using preprocessed VJepa embeddings for direct gesture generation")

        # --- Initialize direct generation model ---
        model_args = {
            "d_latent": args.d_latent,
            "d_model": args.d_model,
            "channels": args.unet_channels,
            "max_segments": args.max_segments,
            "dropout": args.dropout,
        }
        self.model = DirectGestureModel(**model_args).to(self.device)
        self.logger.info(f"Direct generation model instantiated successfully with max_segments={args.max_segments}")
        self.logger.info(f"Target sequence length: {self.model.target_length} timesteps")

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
        self.run_dir = Path(args.output_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}_direct_gen"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.best_val_coord_loss = float("inf")
        self.best_val_pen_loss = float("inf")
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
        """Prepare a training batch with VJepa visual embeddings, variable length masking, and noise augmentation."""
        visual_embeddings, ground_truth_actions = sample  # VJepa embeddings from preprocessing
        visual_embeddings = visual_embeddings.to(self.device, non_blocking=True)  # [B, T, N, D]
        ground_truth_actions = ground_truth_actions.to(self.device, non_blocking=True)  # [B, T-1, Traj, 3]

        B, T_minus_1, Traj, _ = ground_truth_actions.shape
        max_segments = self.args.max_segments
        input_frames = max_segments + 1

        # Pad or crop visual embeddings to fixed input frame count
        if visual_embeddings.shape[1] >= input_frames:
            latent_frames = visual_embeddings[:, :input_frames].contiguous()  # [B, max_segments+1, N, D]
        else:
            # Pad with zeros if we have fewer frames
            padding_frames = input_frames - visual_embeddings.shape[1]
            padding = torch.zeros(B, padding_frames, visual_embeddings.shape[2], visual_embeddings.shape[3],
                                device=visual_embeddings.device, dtype=visual_embeddings.dtype)
            latent_frames = torch.cat([visual_embeddings, padding], dim=1)

        # Determine sequence length with new variable length logic
        available_segments = min(T_minus_1, max_segments)

        if self.model.training and not getattr(self.args, 'overfit', False):
            # Apply variable length masking with probability
            if torch.rand(1).item() < self.args.variable_length_prob:
                # Randomly truncate sequence length
                min_len = max(self.args.min_sequence_length, 1)
                max_len = available_segments
                sequence_length = torch.randint(min_len, max_len + 1, (1,)).item()

                # Zero out latent frames beyond sequence_length
                # latent_frames: [B, max_segments+1, N, D], need sequence_length+1 frames
                if sequence_length < max_segments:
                    latent_frames[:, sequence_length+1:] = 0.0
            else:
                sequence_length = available_segments
        else:
            sequence_length = available_segments

        # Apply Gaussian noise to last N frames of non-zero sequence (only during training)
        if self.model.training and not getattr(self.args, 'overfit', False):
            # Calculate noise standard deviation for target MAE
            # For Gaussian: E[|X|] = σ * sqrt(2/π), so σ = MAE / sqrt(2/π)
            import math
            noise_std = self.args.noise_mae_target / math.sqrt(2 / math.pi)

            # Determine which frames are eligible for noise (last N frames of non-zero sequence)
            non_zero_frames = sequence_length + 1  # +1 because we need N+1 latent frames for N segments
            noise_eligible_frames = min(self.args.noise_last_n_frames, non_zero_frames)

            if noise_eligible_frames > 0:
                # Apply noise to each eligible frame independently with probability
                start_frame = non_zero_frames - noise_eligible_frames
                for frame_idx in range(start_frame, non_zero_frames):
                    if torch.rand(1).item() < self.args.noise_prob:
                        # Add Gaussian noise to this frame
                        noise = torch.randn_like(latent_frames[:, frame_idx]) * noise_std
                        latent_frames[:, frame_idx] = latent_frames[:, frame_idx] + noise

        # Create target gesture sequences with fixed length (max_segments * 250)
        target_length = max_segments * 250
        gesture_sequences = torch.zeros(B, target_length, 3, device=ground_truth_actions.device, dtype=ground_truth_actions.dtype)

        # Fill in the available gesture data (only for non-zero sequence length)
        if sequence_length > 0:
            available_length = min(sequence_length * Traj, target_length)
            available_data = ground_truth_actions[:, :sequence_length].contiguous().view(B, sequence_length * Traj, 3)
            gesture_sequences[:, :available_length] = available_data[:, :available_length]

        # Note: Remaining positions in gesture_sequences stay as zeros (for truncated sequences)

        return gesture_sequences, latent_frames, sequence_length

    def train_step(self, sample):
        """Single training step for direct gesture generation."""
        gesture_sequences, latent_frames, sequence_length = self.prepare_batch(sample)

        # Forward pass: JEPA latents → gesture sequences
        predictions = self.model(latent_frames, sequence_length)

        # Compute loss with sequence length masking
        loss_dict = self.model.compute_loss(
            predictions, gesture_sequences, sequence_length,
            coordinate_weight=self.args.coordinate_weight,
            pen_weight=self.args.pen_weight,
            delta_weight=self.args.delta_weight
        )

        total_loss = loss_dict['total_loss']
        coord_loss = loss_dict['coordinate_loss']
        pen_loss = loss_dict['pen_loss']
        delta_loss = loss_dict['delta_loss']

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

        self.optimizer.step()
        self.global_step += 1

        # Prepare return metrics
        metrics = {
            "loss": total_loss.item(),
            "coordinate_loss": coord_loss.item(),
            "pen_loss": pen_loss.item(),
            "delta_loss": delta_loss.item(),
            "weighted_coordinate_loss": loss_dict['weighted_coordinate_loss'].item(),
            "weighted_pen_loss": loss_dict['weighted_pen_loss'].item(),
            "weighted_delta_loss": loss_dict['weighted_delta_loss'].item(),
            "target_pen_ratio": loss_dict['target_pen_ratio'],
            "pred_pen_ratio": loss_dict['pred_pen_ratio'],
            "mask_coverage": loss_dict['mask_coverage'],
            "delta_mask_coverage": loss_dict['delta_mask_coverage'],
            "sequence_length": sequence_length
        }

        return metrics

    @torch.no_grad()
    def validate(self):
        """Validation loop."""
        self.model.eval()
        total_loss = 0
        total_coord_loss = 0
        total_pen_loss = 0
        total_delta_loss = 0
        num_batches = 0

        for sample in self.val_loader:
            gesture_sequences, latent_frames, sequence_length = self.prepare_batch(sample)

            # Forward pass: JEPA latents → gesture sequences
            predictions = self.model(latent_frames, sequence_length)

            # Compute loss with sequence length masking
            loss_dict = self.model.compute_loss(
                predictions, gesture_sequences, sequence_length,
                coordinate_weight=self.args.coordinate_weight,
                pen_weight=self.args.pen_weight,
                delta_weight=self.args.delta_weight
            )

            total_loss += loss_dict['total_loss'].item()
            total_coord_loss += loss_dict['coordinate_loss'].item()
            total_pen_loss += loss_dict['pen_loss'].item()
            total_delta_loss += loss_dict['delta_loss'].item()
            num_batches += 1

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_coord_loss = total_coord_loss / num_batches
            avg_pen_loss = total_pen_loss / num_batches
            avg_delta_loss = total_delta_loss / num_batches
        else:
            avg_loss = avg_coord_loss = avg_pen_loss = avg_delta_loss = float('inf')

        # Log validation metrics
        if self.args.wandb_project:
            val_log_dict = {
                "val/total_loss": avg_loss,
                "val/coordinate_loss": avg_coord_loss,
                "val/pen_loss": avg_pen_loss,
                "val/delta_loss": avg_delta_loss,
                "global_step": self.global_step
            }
            wandb.log(val_log_dict)

        # Generate validation samples
        if self.args.wandb_project:
            self.generate_validation_samples()

        self.model.train()
        return {
            'total_loss': avg_loss,
            'coordinate_loss': avg_coord_loss,
            'pen_loss': avg_pen_loss,
            'delta_loss': avg_delta_loss
        }

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

            # Generate using direct generation model
            generated_batch = self.model.generate(
                latent_frames=latent_batch,
                sequence_length=batch_sequence_length
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

            # Generate using direct generation model
            generated_batch = self.model.generate(
                latent_frames=latent_batch,
                sequence_length=batch_sequence_length
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
            title_prefix = f"Overfit Test - Exact Sample Being Trained (Direct Gen, Step {self.global_step})"
        else:
            title_prefix = f"Train+Val Samples (Direct Gen, Step {self.global_step})"

        fig = visualize_gesture_sequences(
            ground_truth_tensor,
            generated_tensor,
            title_prefix=title_prefix,
            max_samples=num_samples
        )

        # Skip denoising visualization for direct generation model
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
            self.logger.info(f"Generated {len(generated_tensor)} samples: {num_train} train + {num_val} val with Direct Gen")

        wandb.log(log_dict)
        plt.close(fig)
        if sparse_fig is not None:
            plt.close(sparse_fig)
        self.model.train()

    def _save_checkpoint(self, tag: str, val_losses: Dict[str, float] = None):
        """Save model checkpoint."""
        checkpoint_path = self.run_dir / f"direct_gen_{tag}.pt"

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_coord_loss": self.best_val_coord_loss,
            "best_val_pen_loss": self.best_val_pen_loss,
            "global_step": self.global_step,
            "epoch": getattr(self, 'current_epoch', 0),
        }
        if val_losses is not None:
            checkpoint.update(val_losses)

        try:
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"🔄 Checkpoint saved to {checkpoint_path}")

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
            self.best_val_coord_loss = checkpoint.get("best_val_coord_loss", float("inf"))
            self.best_val_pen_loss = checkpoint.get("best_val_pen_loss", float("inf"))
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
        initial_val_losses = self.validate()
        self.logger.info(f"Initial validation - Total: {initial_val_losses['total_loss']:.4f}, "
                        f"Coord: {initial_val_losses['coordinate_loss']:.4f}, "
                        f"Pen: {initial_val_losses['pen_loss']:.4f}, "
                        f"Delta: {initial_val_losses['delta_loss']:.4f}")

        for epoch in range(self.start_epoch, self.args.num_epochs):
            self.current_epoch = epoch
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
                        "train/delta_loss": loss_dict["delta_loss"],
                        "train/weighted_coordinate_loss": loss_dict["weighted_coordinate_loss"],
                        "train/weighted_pen_loss": loss_dict["weighted_pen_loss"],
                        "train/weighted_delta_loss": loss_dict["weighted_delta_loss"],
                        "train/lr": self.optimizer.param_groups[0]['lr'],
                        "train/sequence_length": loss_dict["sequence_length"],
                        "train/mask_coverage": loss_dict["mask_coverage"],
                        "train/delta_mask_coverage": loss_dict["delta_mask_coverage"],
                        "train/target_pen_ratio": loss_dict["target_pen_ratio"],
                        "train/pred_pen_ratio": loss_dict["pred_pen_ratio"],
                        "train/coordinate_weight": self.args.coordinate_weight,
                        "train/pen_weight": self.args.pen_weight,
                        "train/delta_weight": self.args.delta_weight,
                        "train/variable_length_prob": self.args.variable_length_prob,
                        "train/noise_prob": self.args.noise_prob,
                        "train/noise_mae_target": self.args.noise_mae_target,
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
                    val_losses = self.validate()
                    total_loss = val_losses['total_loss']
                    coord_loss = val_losses['coordinate_loss']
                    pen_loss = val_losses['pen_loss']
                    delta_loss = val_losses['delta_loss']

                    self.logger.info(f"Half-epoch validation - Total: {total_loss:.4f}, "
                                   f"Coord: {coord_loss:.4f}, Pen: {pen_loss:.4f}, Delta: {delta_loss:.4f}")

                    # Log half-epoch validation
                    if self.args.wandb_project:
                        wandb.log({
                            "val_half/total_loss": total_loss,
                            "val_half/coordinate_loss": coord_loss,
                            "val_half/pen_loss": pen_loss,
                            "val_half/delta_loss": delta_loss,
                            "val_half/epoch": epoch + 0.5,
                            "global_step": self.global_step
                        })

                    # Check for new best losses and save checkpoints
                    saved_checkpoints = []
                    if total_loss < self.best_val_loss:
                        self.best_val_loss = total_loss
                        self.logger.info(f"🏆 New best total validation loss at half-epoch: {total_loss:.4f}")
                        self._save_checkpoint("best_total", val_losses)
                        saved_checkpoints.append("best_total")

                    if coord_loss < self.best_val_coord_loss:
                        self.best_val_coord_loss = coord_loss
                        self.logger.info(f"🏆 New best coordinate validation loss at half-epoch: {coord_loss:.4f}")
                        self._save_checkpoint("best_coord", val_losses)
                        saved_checkpoints.append("best_coord")

                    if pen_loss < self.best_val_pen_loss:
                        self.best_val_pen_loss = pen_loss
                        self.logger.info(f"🏆 New best pen validation loss at half-epoch: {pen_loss:.4f}")
                        self._save_checkpoint("best_pen", val_losses)
                        saved_checkpoints.append("best_pen")

                    if saved_checkpoints:
                        self.logger.info(f"Saved checkpoints: {', '.join(saved_checkpoints)}")

            # Update learning rate
            self.scheduler.step()

            # Log epoch summary
            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            self.logger.info(f"--- Epoch {epoch+1} Complete ---")
            self.logger.info(f"Average Train Loss: {avg_train_loss:.4f}")

            # End-of-epoch validation (every epoch now)
            self.logger.info(f"=== End-of-epoch validation: {epoch+1} ===")
            val_losses = self.validate()
            total_loss = val_losses['total_loss']
            coord_loss = val_losses['coordinate_loss']
            pen_loss = val_losses['pen_loss']
            delta_loss = val_losses['delta_loss']

            self.logger.info(f"End-of-epoch validation - Total: {total_loss:.4f}, "
                           f"Coord: {coord_loss:.4f}, Pen: {pen_loss:.4f}, Delta: {delta_loss:.4f}")

            # Check for new best losses and save checkpoints
            saved_checkpoints = []
            if total_loss < self.best_val_loss:
                self.best_val_loss = total_loss
                self.logger.info(f"🏆 New best total validation loss at end-of-epoch: {total_loss:.4f}")
                self._save_checkpoint("best_total", val_losses)
                saved_checkpoints.append("best_total")

            if coord_loss < self.best_val_coord_loss:
                self.best_val_coord_loss = coord_loss
                self.logger.info(f"🏆 New best coordinate validation loss at end-of-epoch: {coord_loss:.4f}")
                self._save_checkpoint("best_coord", val_losses)
                saved_checkpoints.append("best_coord")

            if pen_loss < self.best_val_pen_loss:
                self.best_val_pen_loss = pen_loss
                self.logger.info(f"🏆 New best pen validation loss at end-of-epoch: {pen_loss:.4f}")
                self._save_checkpoint("best_pen", val_losses)
                saved_checkpoints.append("best_pen")

            if saved_checkpoints:
                self.logger.info(f"Saved checkpoints: {', '.join(saved_checkpoints)}")

            # Log end-of-epoch validation to wandb
            if self.args.wandb_project:
                wandb.log({
                    "val_end/total_loss": total_loss,
                    "val_end/coordinate_loss": coord_loss,
                    "val_end/pen_loss": pen_loss,
                    "val_end/delta_loss": delta_loss,
                    "val_end/best_total_loss": self.best_val_loss,
                    "val_end/best_coord_loss": self.best_val_coord_loss,
                    "val_end/best_pen_loss": self.best_val_pen_loss,
                    "val_end/epoch": epoch + 1,
                    "global_step": self.global_step
                })

            # Save checkpoint every epoch
            self._save_checkpoint(f"epoch_{epoch+1:03d}", val_losses)
            self.logger.info(f"💾 Saved epoch checkpoint: epoch_{epoch+1:03d}")

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
        steps_per_epoch = 100
        half_epoch_checkpoint = steps_per_epoch // 2  # Visualization at step 50

        for epoch in range(self.args.num_epochs):
            # Train on the same single sample repeatedly
            for step in range(steps_per_epoch):  # 100 steps per epoch on same sample
                loss_dict = self.train_step(single_sample)
                total_steps += 1

                # Log every 50 steps
                if total_steps % 50 == 0:
                    self.logger.info(f"Step {total_steps}: Loss = {loss_dict['loss']:.6f}, "
                                   f"Coord = {loss_dict['coordinate_loss']:.6f}, "
                                   f"Pen = {loss_dict['pen_loss']:.6f}, "
                                   f"Delta = {loss_dict['delta_loss']:.6f}")

                # Log to wandb
                if self.args.wandb_project and total_steps % 10 == 0:
                    log_dict = {
                        "overfit/total_loss": loss_dict["loss"],
                        "overfit/coordinate_loss": loss_dict["coordinate_loss"],
                        "overfit/pen_loss": loss_dict["pen_loss"],
                        "overfit/delta_loss": loss_dict["delta_loss"],
                        "overfit/weighted_coordinate_loss": loss_dict["weighted_coordinate_loss"],
                        "overfit/weighted_pen_loss": loss_dict["weighted_pen_loss"],
                        "overfit/weighted_delta_loss": loss_dict["weighted_delta_loss"],
                        "overfit/lr": self.optimizer.param_groups[0]['lr'],
                        "overfit/sequence_length": loss_dict["sequence_length"],
                        "overfit/mask_coverage": loss_dict["mask_coverage"],
                        "overfit/delta_mask_coverage": loss_dict["delta_mask_coverage"],
                        "overfit/target_pen_ratio": loss_dict["target_pen_ratio"],
                        "overfit/pred_pen_ratio": loss_dict["pred_pen_ratio"],
                        "overfit/coordinate_weight": self.args.coordinate_weight,
                        "overfit/pen_weight": self.args.pen_weight,
                        "overfit/delta_weight": self.args.delta_weight,
                        "overfit/variable_length_prob": self.args.variable_length_prob,
                        "overfit/noise_prob": self.args.noise_prob,
                        "overfit/noise_mae_target": self.args.noise_mae_target,
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

                # Half-epoch visualization (2 per epoch)
                if step == half_epoch_checkpoint:
                    self.logger.info(f"=== Half-epoch overfit visualization: {epoch+1}.5 ===")
                    self.model.eval()
                    with torch.no_grad():
                        embeddings_device = single_embeddings.to(self.device)
                        full_seq_len = single_gestures.shape[1]

                        self.generate_validation_samples(
                            num_samples=1,
                            custom_embeddings=embeddings_device,
                            custom_gestures=single_gestures,
                            sequence_length=full_seq_len,
                            log_prefix="overfit_half"
                        )
                    self.model.train()

            # Update learning rate
            self.scheduler.step()

            # Log epoch summary
            self.logger.info(f"=== Overfit Epoch {epoch+1}/{self.args.num_epochs} Complete ===")
            self.logger.info(f"Last loss: {loss_dict['loss']:.6f}")

            # End-of-epoch visualization (2 per epoch)
            self.logger.info(f"=== End-of-epoch overfit visualization: {epoch+1} ===")
            self.model.eval()
            with torch.no_grad():
                embeddings_device = single_embeddings.to(self.device)
                full_seq_len = single_gestures.shape[1]

                self.generate_validation_samples(
                    num_samples=1,
                    custom_embeddings=embeddings_device,
                    custom_gestures=single_gestures,
                    sequence_length=full_seq_len,
                    log_prefix="overfit_end"
                )
            self.model.train()

        self.logger.info("Overfit training completed!")


def get_args():
    parser = argparse.ArgumentParser(description="Train Direct Gesture Generation Model")

    # Data args
    parser.add_argument("--processed_data_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/direct_gen_gesture")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume training from")

    # Model args
    parser.add_argument("--d_latent", type=int, default=1024,
                       help="JEPA latent dimension")
    parser.add_argument("--d_model", type=int, default=512,
                       help="Internal model dimension")
    parser.add_argument("--unet_channels", type=int, nargs='+', default=[64, 128, 256, 512],
                       help="U-Net channel progression")
    parser.add_argument("--max_segments", type=int, default=4,
                       help="Maximum number of gesture segments (fixed at training time)")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate for regularization")

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
                       help="Weight for coordinate MSE loss")
    parser.add_argument("--pen_weight", type=float, default=1.0,
                       help="Weight for pen state BCE loss")
    parser.add_argument("--delta_weight", type=float, default=1.0,
                       help="Weight for delta loss (frame-to-frame movement differences)")

    # Data augmentation args
    parser.add_argument("--variable_length_prob", type=float, default=0.5,
                       help="Probability of using variable sequence length (truncating sequences)")
    parser.add_argument("--min_sequence_length", type=int, default=1,
                       help="Minimum sequence length when using variable length")
    parser.add_argument("--noise_prob", type=float, default=0.3,
                       help="Probability of applying Gaussian noise to eligible frames")
    parser.add_argument("--noise_last_n_frames", type=int, default=2,
                       help="Number of last frames eligible for noise application")
    parser.add_argument("--noise_mae_target", type=float, default=0.35,
                       help="Target MAE for Gaussian noise (determines noise strength)")

    # Logging and evaluation
    parser.add_argument("--wandb_project", type=str, default="direct-gesture-generator")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")

    # Overfit test mode
    parser.add_argument("--overfit", action="store_true",
                       help="Overfit test mode: train on single sample repeatedly")

    return parser.parse_args()


def main():
    args = get_args()
    try:
        trainer = DirectGenTrainer(args)
        trainer.training_loop()
        print("Direct generation training finished successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise


if __name__ == "__main__":
    main()