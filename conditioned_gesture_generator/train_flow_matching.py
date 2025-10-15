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
from conditioned_gesture_generator.flow_matching_model import (
    GestureFlowModel, FlowSampler,
    save_gesture_flow_checkpoint, sample_rectified_flow_batch_joint
)


def visualize_jepa_reconstruction_heatmap(jepa_latents, recon_latents, title_prefix="JEPA Reconstruction", max_samples=6):
    """Create MAE heatmap comparing JEPA latents vs reconstructed latents.

    Args:
        jepa_latents: Original JEPA latents [B, T, N, D]
        recon_latents: Reconstructed latents [B, T, N, D]
        title_prefix: Title prefix for the plot
        max_samples: Maximum number of samples to visualize

    Returns:
        matplotlib figure with heatmaps
    """
    import matplotlib.pyplot as plt
    import numpy as np

    B, T, N, D = jepa_latents.shape
    batch_size = min(B, max_samples)

    # Compute MAE per token per timestep
    mae_per_token = torch.abs(jepa_latents - recon_latents).mean(dim=-1)  # [B, T, N]

    # Create subplots
    fig, axes = plt.subplots(batch_size, 1, figsize=(12, 3*batch_size))
    if batch_size == 1:
        axes = [axes]

    for b in range(batch_size):
        mae_map = mae_per_token[b].cpu().numpy()  # [T, N]

        im = axes[b].imshow(mae_map.T, aspect='auto', cmap='viridis',
                           interpolation='nearest')
        axes[b].set_title(f'{title_prefix} - Sample {b+1} - MAE Heatmap')
        axes[b].set_xlabel('Time Steps')
        axes[b].set_ylabel('JEPA Tokens')

        # Add colorbar
        plt.colorbar(im, ax=axes[b], label='Mean Absolute Error')

    plt.tight_layout()
    return fig


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

    # The new format is [B, L_gesture, 3] where L_gesture might be variable
    B, L_gesture, C = original_gestures.shape

    print(f"DEBUG: Input shapes - original: {original_gestures.shape}, generated: {generated_gestures.shape}")
    print(f"DEBUG: B={B}, L_gesture={L_gesture}, C={C}, batch_size={batch_size}")

    # First crop to batch_size to fix the shape mismatch
    original_gestures = original_gestures[:batch_size]  # [batch_size, L_gesture, 3]
    generated_gestures = generated_gestures[:batch_size]  # [batch_size, L_gesture, 3]
    print(f"DEBUG: After batch crop - original: {original_gestures.shape}, generated: {generated_gestures.shape}")

    # Use fixed chunk size of 250 for visualization consistency
    chunk_size = 250
    max_chunks = L_gesture // chunk_size
    num_chunks = min(4, max_chunks)  # Use up to 4 chunks (frames)

    print(f"DEBUG: max_chunks={max_chunks}, num_chunks={num_chunks}")

    if num_chunks == 0:
        # If sequence is shorter than chunk_size, use the full sequence as one chunk
        print(f"DEBUG: Sequence too short ({L_gesture} < {chunk_size}), using as single chunk")
        # Just use the first 250 timesteps or pad if shorter
        if L_gesture >= chunk_size:
            original_gestures = original_gestures[:, :chunk_size, :]
            generated_gestures = generated_gestures[:, :chunk_size, :]
        else:
            # Pad to chunk_size
            pad_length = chunk_size - L_gesture
            original_gestures = np.pad(original_gestures, ((0, 0), (0, pad_length), (0, 0)), 'constant')
            generated_gestures = np.pad(generated_gestures, ((0, 0), (0, pad_length), (0, 0)), 'constant')
        num_chunks = 1
        L_truncated = chunk_size
    else:
        # Truncate to fit exact chunks
        L_truncated = num_chunks * chunk_size
        print(f"DEBUG: Truncating from {L_gesture} to {L_truncated} timesteps")
        original_gestures = original_gestures[:, :L_truncated, :]  # Already cropped to batch_size above
        generated_gestures = generated_gestures[:, :L_truncated, :]

    print(f"DEBUG: After truncation - original: {original_gestures.shape}, generated: {generated_gestures.shape}")
    print(f"DEBUG: Total elements - original: {original_gestures.size}, generated: {generated_gestures.size}")
    print(f"DEBUG: Reshaping to ({batch_size}, {num_chunks}, {chunk_size}, 3) = {batch_size * num_chunks * chunk_size * 3} elements")

    # Verify shapes match before reshaping
    expected_size = batch_size * num_chunks * chunk_size * 3
    if original_gestures.size != expected_size:
        print(f"ERROR: Size mismatch! Have {original_gestures.size} elements, need {expected_size}")
        # Fallback: use first few timesteps only
        fallback_timesteps = min(L_gesture, chunk_size)
        original_gestures = original_gestures[:, :fallback_timesteps, :]
        generated_gestures = generated_gestures[:, :fallback_timesteps, :]
        num_chunks = 1
        chunk_size = fallback_timesteps
        print(f"DEBUG: Fallback - using {fallback_timesteps} timesteps as single chunk")

    orig_reshaped = original_gestures.reshape(batch_size, num_chunks, chunk_size, 3)  # [B, chunks, chunk_size, 3]
    gen_reshaped = generated_gestures.reshape(batch_size, num_chunks, chunk_size, 3)   # [B, chunks, chunk_size, 3]

    num_frames = min(2, num_chunks)  # Use up to 2 frames

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


# Note: visualize_sparse_flow_steps function removed due to API incompatibility
# with joint trajectory flow model (model.forward signature changed)


class FlowTrainer:
    """Trainer for gesture flow matching model."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.logger = get_logger()

        # Set seeds for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # --- Initialize state projection layer ---
        self.state_proj = nn.Linear(args.d_latent, args.d_state).to(self.device)
        self.state_norm = nn.LayerNorm(args.d_state).to(self.device)
        self.logger.info(f"State projection: {args.d_latent} -> {args.d_state} + LayerNorm")

        # --- Calculate joint feature dimensions ---
        d_action = 3 if args.include_pen_in_flow else 2

        # --- Compute action normalization statistics from actual dataset ---

        # --- Calculate joint feature dimensions ---
        in_feature_dim = args.d_state + d_action
        self.d_state = args.d_state
        self.d_action = d_action
        self.include_pen_in_flow = args.include_pen_in_flow

        # --- Initialize joint trajectory flow matching model ---
        model_args = {
            "in_feature_dim": in_feature_dim,
            "time_embed_dim": args.time_embed_dim,
            "channels": tuple(args.unet_channels),
            "max_sequence_length": args.max_sequence_length,
            "film_dim": args.film_dim if not args.no_film else None,
            "d_state": self.d_state,
            "use_film": not args.no_film
        }
        self.model = GestureFlowModel(**model_args).to(self.device)
        self.logger.info(f"Joint trajectory flow model instantiated successfully")
        self.logger.info(f"Input feature dim: {in_feature_dim} (state: {args.d_state}, action: {d_action})")

        # --- Initialize Flow sampler ---
        self.sampler = FlowSampler(self.model, device=self.device, method=self.args.ode_method)

        # --- Optimizer ---
        # Include all trainable parameters: model + state projection + state normalization
        all_params = list(self.model.parameters()) + \
                     list(self.state_proj.parameters()) + \
                     list(self.state_norm.parameters())

        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.num_epochs,
            eta_min=args.learning_rate * 0.999
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

        # --- Compute action normalization statistics from dataset ---
        action_mu, action_sigma = self._compute_dataset_action_stats()
        self.action_mu = torch.tensor(action_mu, device=self.device).view(1, 1, -1)
        self.action_sigma = torch.tensor(action_sigma, device=self.device).view(1, 1, -1)

        # --- Checkpointing and Logging ---
        self.run_dir = Path(args.output_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}_flow"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.global_step = 0

        # --- ADD: Training dynamics tracking ---
        self.loss_history = []
        self.val_loss_history = []
        self.gradient_norm_history = []
        self.state_action_ratio_history = []

        if args.wandb_project:
            run_name = args.wandb_run_name if args.wandb_run_name else self.run_dir.name
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
            wandb.watch(self.model, log="gradients", log_freq=100)
            self.logger.info(f"W&B initialized for run: {run_name}")

            # Log computed dataset statistics
            dataset_stats = {
                "dataset/action_mu_x": action_mu[0],
                "dataset/action_sigma_x": action_sigma[0],
                "dataset/action_mu_y": action_mu[1],
                "dataset/action_sigma_y": action_sigma[1],
            }
            if self.include_pen_in_flow:
                dataset_stats.update({
                    "dataset/action_mu_pen": action_mu[2],
                    "dataset/action_sigma_pen": action_sigma[2],
                })
            wandb.log(dataset_stats)
            self.logger.info(f"Logged dataset action statistics to wandb")

        # --- Resume from checkpoint if specified ---
        self.start_epoch = 0
        if args.resume_from:
            self.logger.info(f"Resuming training from checkpoint: {args.resume_from}")
            self._load_checkpoint(args.resume_from)

    def _norm_actions(self, a):
        """Normalize actions using fixed dataset statistics."""
        return (a - self.action_mu) / (self.action_sigma + 1e-6)

    def _denorm_actions(self, a):
        """Denormalize actions back to original units."""
        return a * (self.action_sigma + 1e-6) + self.action_mu

    def _compute_dataset_action_stats(self):
        """Compute mean and std of actions from the training dataset."""
        self.logger.info("Computing dataset action statistics...")

        all_actions = []
        sample_count = 0
        max_samples = 1000  # Limit samples for efficiency

        for batch_idx, sample in enumerate(self.train_loader):
            if sample_count >= max_samples:
                break

            visual_embeddings, ground_truth_actions = sample
            ground_truth_actions = ground_truth_actions.to(self.device)  # [B, T-1, Traj, 3]

            B, T_minus_1, Traj, _ = ground_truth_actions.shape

            # Reshape to [B*T*Traj, 3] and extract coordinates
            action_sequences = ground_truth_actions.contiguous().view(-1, 3)

            if self.include_pen_in_flow:
                action_coords = action_sequences  # [N, 3] - x, y, pen
            else:
                action_coords = action_sequences[..., :2]  # [N, 2] - x, y only

            all_actions.append(action_coords.cpu())
            sample_count += B

            if (batch_idx + 1) % 50 == 0:
                self.logger.info(f"Processed {batch_idx + 1} batches, {sample_count} samples")

        if not all_actions:
            raise ValueError("No action data found in training set")

        # Concatenate all actions and compute statistics
        all_actions = torch.cat(all_actions, dim=0)  # [N_total, d_action]

        action_mu = all_actions.mean(dim=0).tolist()  # Per-channel mean
        action_std = all_actions.std(dim=0).tolist()   # Per-channel std

        self.logger.info(f"Computed action statistics from {all_actions.shape[0]} action points:")
        self.logger.info(f"  Mean (μ): {action_mu}")
        self.logger.info(f"  Std (σ):  {action_std}")

        return action_mu, action_std

    def prepare_batch(self, sample):
        """Prepare a training batch by building joint state-action sequences."""
        visual_embeddings, ground_truth_actions = sample  # VJepa embeddings from preprocessing
        visual_embeddings = visual_embeddings.to(self.device, non_blocking=True)  # [B, T, N, D]
        ground_truth_actions = ground_truth_actions.to(self.device, non_blocking=True)  # [B, T-1, Traj, 3]

        # Apply layer normalization to visual embeddings (z-score normalization on D dimension)
        visual_embeddings = F.layer_norm(visual_embeddings, (visual_embeddings.size(-1),))

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

        # --- 1) Prepare raw latent frames for FiLM conditioning (matching diffusion model) ---
        # Get latent frames for sequence_length + 1 frames
        latent_frames = visual_embeddings[:, :sequence_length + 1].contiguous()  # [B, sequence_length+1, N, D]

        # Apply same pre-normalization as LAM expects (critical for consistency)
        latent_frames_normalized = F.layer_norm(latent_frames, (latent_frames.size(-1),))

        # No pooling for FiLM! Use raw patches for rich conditioning like diffusion model
        # latent_frames_normalized: [B, sequence_length+1, N=256, D=1024]

        # But still need pooled state tokens for joint sequence construction
        pooled = latent_frames_normalized.mean(dim=2)  # [B, F=sequence_length+1, D] - average over patches
        state_tokens = self.state_norm(self.state_proj(pooled))  # [B, F, d_state] - with layer norm

        # --- 2) Action tokens: gestures ---
        action_sequences = ground_truth_actions[:, :sequence_length].contiguous().view(B, sequence_length * Traj, 3)
        if self.include_pen_in_flow:
            action_tokens_raw = action_sequences  # [B, L, 3]
        else:
            action_tokens_raw = action_sequences[..., :2]  # [B, L, 2] - coordinates only

        # Apply fixed z-score normalization to action tokens
        action_tokens = self._norm_actions(action_tokens_raw)  # [B, L, d_action] - fixed normalization

        # --- 3) Diffuser-style per-timestep stacking: x_t = [s_t ; a_t] for all action steps ---
        # Broadcast states over their following 250 action steps
        states_for_actions = state_tokens[:, :-1, :]                                   # [B, sequence_length, d_state]
        state_per_step = states_for_actions.unsqueeze(2).expand(-1, -1, Traj, -1)     # [B, seq_len, 250, d_state]
        state_per_step = state_per_step.reshape(B, sequence_length * Traj, self.d_state)  # [B, L, d_state]

        # Concatenate per-timestep [s ; a]
        joint_seq = torch.cat([state_per_step, action_tokens], dim=-1)                # [B, L, d_state + d_action]

        # For dense training we don't need a type mask anymore
        type_mask = None

        # Pen targets aligned 1:1 with per-timestep actions
        gt_pen_aligned = action_sequences[:, :sequence_length * Traj, 2:3]            # [B, L, 1]

        # Clean joint equals GT (used by pen head pass)
        clean_joint = joint_seq.clone()

        # Store layout info (keep d_state/d_action for heads; state_positions no longer used)
        self.last_layout = {
            "d_state": self.d_state,
            "d_action": self.d_action,
            "sequence_length": sequence_length
        }

        # --- ADD: Store normalization statistics for monitoring ---
        with torch.no_grad():
            # State statistics after normalization
            state_stats = {
                "state_mean": state_per_step.mean().item(),
                "state_std": state_per_step.std().item(),
                "state_l2_norm": torch.norm(state_per_step, dim=-1).mean().item(),
                "state_max_abs": state_per_step.abs().max().item(),
                "state_min": state_per_step.min().item(),
                "state_max": state_per_step.max().item()
            }

            # Action statistics after normalization
            action_stats = {
                "action_mean": action_tokens.mean().item(),
                "action_std": action_tokens.std().item(),
                "action_l2_norm": torch.norm(action_tokens, dim=-1).mean().item(),
                "action_max_abs": action_tokens.abs().max().item(),
                "action_min": action_tokens.min().item(),
                "action_max": action_tokens.max().item()
            }

            # Joint sequence statistics
            joint_stats = {
                "joint_mean": joint_seq.mean().item(),
                "joint_std": joint_seq.std().item(),
                "joint_l2_norm": torch.norm(joint_seq, dim=-1).mean().item(),
                "joint_state_part_norm": torch.norm(joint_seq[..., :self.d_state], dim=-1).mean().item(),
                "joint_action_part_norm": torch.norm(joint_seq[..., self.d_state:], dim=-1).mean().item()
            }

            # Raw action statistics (before normalization)
            raw_action_stats = {
                "raw_action_mean": action_tokens_raw.mean().item(),
                "raw_action_std": action_tokens_raw.std().item(),
                "raw_action_l2_norm": torch.norm(action_tokens_raw, dim=-1).mean().item(),
                "raw_action_max_abs": action_tokens_raw.abs().max().item()
            }

            # Store for logging - safely handle different action dimensions
            normalization_stats = {
                **state_stats,
                **action_stats,
                **joint_stats,
                **raw_action_stats,
            }

            # Safely access normalization parameters based on actual dimensions
            if self.action_mu.shape[-1] >= 1:
                normalization_stats["normalization_mu_x"] = self.action_mu[0, 0, 0].item()
                normalization_stats["normalization_sigma_x"] = self.action_sigma[0, 0, 0].item()

            if self.action_mu.shape[-1] >= 2:
                normalization_stats["normalization_mu_y"] = self.action_mu[0, 0, 1].item()
                normalization_stats["normalization_sigma_y"] = self.action_sigma[0, 0, 1].item()

            if self.include_pen_in_flow and self.action_mu.shape[-1] >= 3:
                normalization_stats["normalization_mu_pen"] = self.action_mu[0, 0, 2].item()
                normalization_stats["normalization_sigma_pen"] = self.action_sigma[0, 0, 2].item()

            self.last_normalization_stats = normalization_stats

        return joint_seq, type_mask, clean_joint, gt_pen_aligned, latent_frames_normalized, action_sequences, sequence_length



    def train_step(self, sample):
        """Joint trajectory training step with flow + separate pen BCE passes."""
        joint_seq, type_mask, clean_joint, gt_pen_aligned, latent_frames, action_sequences, sequence_length = self.prepare_batch(sample)

        B, L_joint, C_joint = joint_seq.shape

        # --- PASS 1: Joint Rectified Flow training (velocity prediction) ---
        x_t, t, v_target = sample_rectified_flow_batch_joint(
            joint_seq,
            noisy=getattr(self.args, 'rf_noisy', False),
            sigma=getattr(self.args, 'rf_sigma', 0.1)
        )

        # Forward pass for joint flow - get velocity prediction
        model_output = self.model(x_t, t, latent_frames, sequence_length)
        v_pred = model_output['velocity']  # [B, L, d_state + d_action]

        # Dense velocity loss: supervise all channels at all timesteps
        vp_s, vt_s = v_pred[..., :self.d_state], v_target[..., :self.d_state]
        vp_a, vt_a = v_pred[...,  self.d_state:], v_target[...,  self.d_state:]
        state_mse  = F.mse_loss(vp_s, vt_s)
        action_mse = F.mse_loss(vp_a, vt_a)
        flow_loss  = self.args.state_weight * state_mse + (1 - self.args.state_weight) * action_mse
        flow_loss_dict = {"loss": flow_loss, "state_mse": state_mse, "action_mse": action_mse}

        # --- PASS 2: Clean pen prediction (now every row is an action row) ---
        dummy_t = torch.zeros(B, device=self.device)
        pen_logits = self.model.forward_pen_prediction(clean_joint, dummy_t, latent_frames, sequence_length)  # [B, L, 1]

        # (Option A) Simple BCE over all timesteps
        pen_loss = F.binary_cross_entropy_with_logits(
            pen_logits.squeeze(-1), gt_pen_aligned.squeeze(-1)
        )

        # --- Combined loss and backward pass ---
        flow_weight = getattr(self.args, 'velocity_weight', 1.0)
        pen_weight  = getattr(self.args, 'pen_weight', 1.0)
        total_loss  = flow_weight * flow_loss + pen_weight * pen_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

        self.optimizer.step()
        self.global_step += 1

        # --- ADD: Update training dynamics history ---
        self.loss_history.append(total_loss.item())
        if len(self.loss_history) > 1000:  # Keep last 1000 steps
            self.loss_history = self.loss_history[-1000:]

        # Prepare return metrics
        gt_pen_ratio = gt_pen_aligned.mean().item()

        # --- ADD: Compute additional flow statistics for monitoring ---
        with torch.no_grad():
            # Velocity prediction statistics
            v_pred_stats = {
                "v_pred_mean": v_pred.mean().item(),
                "v_pred_std": v_pred.std().item(),
                "v_pred_l2_norm": torch.norm(v_pred, dim=-1).mean().item(),
                "v_pred_state_norm": torch.norm(v_pred[..., :self.d_state], dim=-1).mean().item(),
                "v_pred_action_norm": torch.norm(v_pred[..., self.d_state:], dim=-1).mean().item()
            }

            # Velocity target statistics
            v_target_stats = {
                "v_target_mean": v_target.mean().item(),
                "v_target_std": v_target.std().item(),
                "v_target_l2_norm": torch.norm(v_target, dim=-1).mean().item(),
                "v_target_state_norm": torch.norm(v_target[..., :self.d_state], dim=-1).mean().item(),
                "v_target_action_norm": torch.norm(v_target[..., self.d_state:], dim=-1).mean().item()
            }

            # Flow input statistics (x_t)
            flow_input_stats = {
                "x_t_mean": x_t.mean().item(),
                "x_t_std": x_t.std().item(),
                "x_t_l2_norm": torch.norm(x_t, dim=-1).mean().item(),
                "t_mean": t.mean().item(),
                "t_std": t.std().item()
            }

            # State vs Action component analysis
            state_action_comparison = {
                "state_action_norm_ratio": v_pred_stats["v_pred_state_norm"] / (v_pred_stats["v_pred_action_norm"] + 1e-8),
                "state_action_loss_ratio": state_mse.item() / (action_mse.item() + 1e-8),
                "effective_state_weight": self.args.state_weight,
                "effective_action_weight": (1 - self.args.state_weight)
            }

        metrics = {
            "loss": total_loss.item(),
            "flow_loss": flow_loss.item(),
            "pen_loss": pen_loss.item(),
            "state_mse": state_mse.item(),
            "action_mse": action_mse.item(),
            "state_error": state_mse.item(),    # alias
            "action_error": action_mse.item(),  # alias
            "pen_state_mean": gt_pen_ratio,
            "sequence_length": sequence_length,
            "state_weight": self.args.state_weight,
            # --- ADD: Extended metrics ---
            **v_pred_stats,
            **v_target_stats,
            **flow_input_stats,
            **state_action_comparison
        }

        # Add normalization stats if available
        if hasattr(self, 'last_normalization_stats'):
            metrics.update({f"norm_{k}": v for k, v in self.last_normalization_stats.items()})

        return metrics

    def joint_to_gesture(self, x_joint):
        """Helper to extract gesture coords from per-timestep joint."""
        gen_xy_norm = x_joint[..., self.d_state:]      # [B, L, d_action]
        return self._denorm_actions(gen_xy_norm)


    def compute_jepa_reconstruction(self, visual_embeddings):
        """Compute JEPA latent reconstruction MAE for visualization.

        Since we only need latent MAE (not decoded images), we simulate reconstruction
        by comparing original latents with flow-processed latents.

        Args:
            visual_embeddings: VJepa embeddings [B, T, N, D]
        Returns:
            Tuple of (original_latents, reconstructed_latents, mse_loss)
        """
        with torch.no_grad():
            # Ensure visual_embeddings is on the same device as the model
            visual_embeddings = visual_embeddings.to(self.device)

            B, T, N, D = visual_embeddings.shape

            # Ensure JEPA embeddings are z-score normalized (should already be done in prepare_batch)
            visual_embeddings = F.layer_norm(visual_embeddings, (visual_embeddings.size(-1),))

            # Original JEPA latents
            original_latents = visual_embeddings.clone()

            # For reconstruction comparison, we'll use the flow-processed state tokens
            # This represents how well our joint trajectory flow preserves the JEPA information

            # Project through our state projection layer (like the flow model does)
            # Apply same pre-normalization as LAM expects (for consistency)
            visual_embeddings_normalized = F.layer_norm(visual_embeddings, (visual_embeddings.size(-1),))
            pooled = visual_embeddings_normalized.mean(dim=2)  # [B, T, D] - average over patches
            state_projected = self.state_norm(self.state_proj(pooled))  # [B, T, d_state] - with layer norm

            # "Reconstruct" by projecting back to original dimension
            # This simulates what a perfect reconstruction would look like
            reconstructed_pooled = F.linear(state_projected,
                                          self.state_proj.weight.T,
                                          None)  # [B, T, D]

            # Expand back to patch dimension (broadcast to all patches)
            reconstructed_latents = reconstructed_pooled.unsqueeze(2).expand(B, T, N, D)

            # Compute MSE loss between original and "reconstructed" latents
            mse_loss = F.mse_loss(original_latents, reconstructed_latents)

            return original_latents, reconstructed_latents, mse_loss

    @torch.no_grad()
    def validate(self):
        """Validation loop using joint trajectory flow."""
        self.model.eval()
        total_loss = 0
        total_flow_loss = 0
        total_state_mse = 0
        total_action_mse = 0
        total_pen_loss = 0
        total_sequence_length = 0
        num_batches = 0

        # --- ADD: Validation statistics accumulators ---
        val_stats_accum = {
            "state_l2_norms": [],
            "action_l2_norms": [],
            "v_pred_state_norms": [],
            "v_pred_action_norms": [],
            "state_action_norm_ratios": [],
            "state_action_loss_ratios": [],
            "state_stds": [],
            "action_stds": []
        }

        for sample in self.val_loader:
            # Use new joint trajectory API
            joint_seq, _, clean_joint, gt_pen_aligned, latent_frames, action_sequences, seq_len = self.prepare_batch(sample)

            # --- PASS 1: Flow validation (joint velocity prediction) ---
            x_t, t, v_tar = sample_rectified_flow_batch_joint(
                joint_seq,
                noisy=getattr(self.args, 'rf_noisy', False),
                sigma=getattr(self.args, 'rf_sigma', 0.1)
            )

            # Forward pass for joint flow
            out = self.model(x_t, t, latent_frames, seq_len)
            v_pred = out['velocity']

            # Dense velocity loss
            vp_s, vt_s = v_pred[..., :self.d_state], v_tar[..., :self.d_state]
            vp_a, vt_a = v_pred[...,  self.d_state:], v_tar[...,  self.d_state:]
            state_mse  = F.mse_loss(vp_s, vt_s)
            action_mse = F.mse_loss(vp_a, vt_a)
            flow_loss  = self.args.state_weight * state_mse + (1 - self.args.state_weight) * action_mse
            flow_loss_dict = {"loss": flow_loss, "state_mse": state_mse, "action_mse": action_mse}

            # Pen over all timesteps
            t_zero = torch.zeros(joint_seq.size(0), device=self.device)
            pen_logits = self.model.forward_pen_prediction(clean_joint, t_zero, latent_frames, seq_len)
            pen_loss = F.binary_cross_entropy_with_logits(
                pen_logits.squeeze(-1), gt_pen_aligned.squeeze(-1)
            )

            # --- Combined loss ---
            flow_weight = getattr(self.args, 'velocity_weight', 1.0)
            pen_weight  = getattr(self.args, 'pen_weight', 1.0)
            batch_total_loss = flow_weight * flow_loss + pen_weight * pen_loss

            total_loss += batch_total_loss.item()
            total_flow_loss += flow_loss.item()
            total_state_mse += state_mse.item()
            total_action_mse += action_mse.item()
            total_pen_loss += pen_loss.item()
            total_sequence_length += seq_len
            num_batches += 1

            # --- ADD: Collect validation statistics ---
            if hasattr(self, 'last_normalization_stats'):
                val_stats_accum["state_l2_norms"].append(self.last_normalization_stats.get("state_l2_norm", 0))
                val_stats_accum["action_l2_norms"].append(self.last_normalization_stats.get("action_l2_norm", 0))
                val_stats_accum["state_stds"].append(self.last_normalization_stats.get("state_std", 0))
                val_stats_accum["action_stds"].append(self.last_normalization_stats.get("action_std", 0))

            # Velocity statistics
            v_pred_state_norm = torch.norm(v_pred[..., :self.d_state], dim=-1).mean().item()
            v_pred_action_norm = torch.norm(v_pred[..., self.d_state:], dim=-1).mean().item()
            val_stats_accum["v_pred_state_norms"].append(v_pred_state_norm)
            val_stats_accum["v_pred_action_norms"].append(v_pred_action_norm)
            val_stats_accum["state_action_norm_ratios"].append(v_pred_state_norm / (v_pred_action_norm + 1e-8))
            val_stats_accum["state_action_loss_ratios"].append(state_mse.item() / (action_mse.item() + 1e-8))

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_flow_loss = total_flow_loss / num_batches if num_batches > 0 else float('inf')
        avg_state_mse = total_state_mse / num_batches if num_batches > 0 else float('inf')
        avg_action_mse = total_action_mse / num_batches if num_batches > 0 else float('inf')
        avg_pen_loss = total_pen_loss / num_batches if num_batches > 0 else float('inf')
        avg_sequence_length = total_sequence_length / num_batches if num_batches > 0 else 0

        # --- ADD: Comprehensive validation statistics ---
        if self.args.wandb_project:
            log_payload = {
                "val/total_loss": avg_loss,
                "val/flow_loss": avg_flow_loss,
                "val/pen_loss": avg_pen_loss,
                "val/state_error": avg_state_mse,
                "val/action_error": avg_action_mse,
                "val/sequence_length": avg_sequence_length,
                "global_step": self.global_step,
            }

            if val_stats_accum["state_l2_norms"]:
                val_stats = {}
                for key, values in val_stats_accum.items():
                    if values:  # Only log if we have data
                        val_stats[f"stats_val/{key}_mean"] = np.mean(values)
                        val_stats[f"stats_val/{key}_std"] = np.std(values)
                        val_stats[f"stats_val/{key}_min"] = np.min(values)
                        val_stats[f"stats_val/{key}_max"] = np.max(values)
                        val_stats[f"stats_val/{key}_cv"] = np.std(values) / (np.mean(values) + 1e-8)  # Coefficient of variation

                # Additional validation stability metrics
                if val_stats_accum["state_action_norm_ratios"]:
                    norm_ratios = val_stats_accum["state_action_norm_ratios"]
                    loss_ratios = val_stats_accum["state_action_loss_ratios"]

                    val_stats.update({
                        "stats_val/scale_consistency": 1.0 / (np.std(norm_ratios) + 1e-8),  # Higher = more consistent
                        "stats_val/loss_balance_consistency": 1.0 / (np.std(loss_ratios) + 1e-8),
                        "stats_val/norm_ratio_range": np.max(norm_ratios) - np.min(norm_ratios),
                        "stats_val/validation_stability": 1.0 / (np.std([avg_loss]) + 1e-8) if num_batches > 1 else 1.0,
                    })

                # Model convergence indicators
                val_stats.update({
                    "stats_val/avg_loss": avg_loss,
                    "stats_val/num_batches": num_batches,
                    "stats_val/loss_magnitude": abs(avg_loss),
                })

                log_payload.update(val_stats)

            wandb.log(log_payload)

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
        """Generate comprehensive samples with joint trajectory flow and JEPA reconstruction."""
        self.model.eval()
        self.logger.info("Generating samples with joint trajectory flow and JEPA reconstruction...")

        # Collect samples for visualization
        all_generated_gestures = []
        all_ground_truth_gestures = []
        all_sample_types = []
        all_conditioning_info = []
        all_visual_embeddings = []
        split_collections = {}

        def _add_split_sample(split_key, generated_tensor, ground_tensor, visual_tensor, seq_len, batch_size):
            all_generated_gestures.append(generated_tensor)
            all_ground_truth_gestures.append(ground_tensor)
            all_conditioning_info.extend([seq_len] * batch_size)
            all_visual_embeddings.append(visual_tensor)
            all_sample_types.extend([split_key] * batch_size)

            split_entry = split_collections.setdefault(split_key, {
                "generated": [],
                "ground": [],
                "visual": [],
                "sequence_lengths": []
            })
            split_entry["generated"].append(generated_tensor)
            split_entry["ground"].append(ground_tensor)
            split_entry["visual"].append(visual_tensor)
            split_entry["sequence_lengths"].extend([seq_len] * batch_size)

        if custom_embeddings is not None and custom_gestures is not None:
            # Overfit mode - use custom data
            B = custom_embeddings.shape[0]
            sample = (custom_embeddings, custom_gestures)
            visual_embeddings, gesture_sequences = sample

            # Prepare batch using joint trajectory API
            joint_seq, type_mask, clean_joint, gt_pen_aligned, latent_frames, action_sequences, seq_len = self.prepare_batch(sample)

            B, L, C = joint_seq.shape

            state_per_step = joint_seq[..., :self.d_state]   # [B, L, d_state]
            x_joint = self.sampler.sample(
                shape=(B, L, C),
                latent_frames=None,            # not using FiLM
                sequence_length=seq_len,
                state_per_step=state_per_step, # <-- inpaint states
                anchor_states=True,
                steps=50,
                d_state=self.d_state
            )

            # Per-timestep format: joint = [state ; action] at every step
            gen_xy_norm = x_joint[..., self.d_state:]          # [B, L, d_action]
            gen_xy = self._denorm_actions(gen_xy_norm)         # back to [0,1]

            # Pen prediction over all steps (clean joint is the GT joint we built)
            clean_joint_like = joint_seq.clone()               # [B, L, d_state + d_action]
            t_zero = torch.zeros(B, device=x_joint.device)
            pen_logits = self.model.forward_pen_prediction(clean_joint_like, t_zero, latent_frames, seq_len)
            gen_pen = torch.sigmoid(pen_logits)[..., :1]       # [B, L, 1]

            # Final gestures
            generated_gestures = torch.cat([gen_xy[..., :2], gen_pen], dim=-1)  # [B, L, 3]

            # Ground truth gestures (action sequences)
            gt_gestures = action_sequences[:, :generated_gestures.size(1), :]  # [B, L_gesture, 3]

            _add_split_sample(log_prefix, generated_gestures, gt_gestures, visual_embeddings, seq_len, B)

        else:
            # Normal validation mode - sample from both train and val loaders
            samples_per_loader = num_samples // 2
            loaders = [
                (self.train_loader, "train", samples_per_loader),
                (self.val_loader, "val", num_samples - samples_per_loader)
            ]

            for loader, split_name, n_samples in loaders:
                if n_samples == 0:
                    continue

                collected_samples = 0
                for sample in loader:
                    visual_embeddings, gesture_sequences = sample

                    # Prepare batch using joint trajectory API
                    joint_seq, type_mask, clean_joint, gt_pen_aligned, latent_frames, action_sequences, seq_len = self.prepare_batch(sample)

                    B, L_joint, C_joint = joint_seq.shape

                    state_per_step = joint_seq[..., :self.d_state]   # [B, L_joint, d_state]
                    x_joint = self.sampler.sample(
                        shape=(B, L_joint, C_joint),
                        latent_frames=None,             # not using FiLM
                        sequence_length=seq_len,
                        state_per_step=state_per_step,  # <-- inpaint states
                        anchor_states=True,
                        steps=50,
                        d_state=self.d_state
                    )

                    # Per-timestep: last d_action dims are the action coords
                    gen_xy_norm = x_joint[..., self.d_state:]              # [B, L, d_action]
                    gen_xy = self._denorm_actions(gen_xy_norm)             # [B, L, d_action] in [0,1]

                    # Pen prediction over all steps
                    clean_joint_like = joint_seq.clone()
                    t_zero = torch.zeros(B, device=x_joint.device)
                    pen_logits = self.model.forward_pen_prediction(clean_joint_like, t_zero, latent_frames, seq_len)
                    gen_pen = torch.sigmoid(pen_logits)[..., :1]           # [B, L, 1]

                    generated_gestures = torch.cat([gen_xy[..., :2], gen_pen], dim=-1)  # [B, L, 3]
                    gt_gestures = action_sequences[:, :generated_gestures.size(1), :]   # [B, L, 3]

                    _add_split_sample(split_name, generated_gestures, gt_gestures, visual_embeddings, seq_len, B)

                    collected_samples += B
                    if collected_samples >= n_samples:
                        break

        if not all_generated_gestures:
            self.logger.warning("No samples generated for visualization; skipping wandb logging.")
            self.model.train()
            return

        def build_visualization_payload(split_label,
                                         generated_list,
                                         ground_list,
                                         visual_list,
                                         seq_lengths):
            if not generated_list:
                return {}, []

            generated_tensor = torch.cat(generated_list, dim=0)
            ground_tensor = torch.cat(ground_list, dim=0)
            visual_tensor = torch.cat(visual_list, dim=0)

            figure_title = f"{(split_label or log_prefix).capitalize()} Samples"

            fig = visualize_gesture_sequences(
                ground_tensor.cpu().numpy(),
                generated_tensor.cpu().numpy(),
                title_prefix=figure_title,
                max_samples=min(6, len(generated_tensor))
            )

            original_latents, reconstructed_latents, mse_loss = self.compute_jepa_reconstruction(visual_tensor)
            jepa_fig = visualize_jepa_reconstruction_heatmap(
                original_latents.cpu(),
                reconstructed_latents.cpu(),
                title_prefix=f"{figure_title} JEPA State Reconstruction",
                max_samples=min(6, len(original_latents))
            )

            prefix = split_label or log_prefix
            image_key = f"{prefix}/gesture_samples"
            recon_key = f"{prefix}/jepa_reconstruction"

            avg_seq_len = float(np.mean(seq_lengths)) if seq_lengths else 0.0

            log_values = {
                image_key: wandb.Image(fig),
                recon_key: wandb.Image(jepa_fig),
                f"{prefix}/num_generated_samples": len(generated_tensor),
                f"{prefix}/avg_sequence_length": avg_seq_len,
                f"{prefix}/jepa_mse": mse_loss.item(),
                f"{prefix}/jepa_state_error": mse_loss.item(),
            }

            gt_mean = ground_tensor.mean(dim=(0, 1))
            gen_mean = generated_tensor.mean(dim=(0, 1))

            log_values.update({
                f"{prefix}/gt_mean_x": gt_mean[0].item(),
                f"{prefix}/gt_mean_y": gt_mean[1].item(),
                f"{prefix}/gt_mean_touch": gt_mean[2].item(),
                f"{prefix}/gen_mean_x": gen_mean[0].item(),
                f"{prefix}/gen_mean_y": gen_mean[1].item(),
                f"{prefix}/gen_mean_touch": gen_mean[2].item(),
            })

            action_error_x = F.mse_loss(generated_tensor[..., 0], ground_tensor[..., 0]).item()
            action_error_y = F.mse_loss(generated_tensor[..., 1], ground_tensor[..., 1]).item()
            action_error_pen = F.mse_loss(generated_tensor[..., 2], ground_tensor[..., 2]).item()
            action_error_overall = (action_error_x + action_error_y) / 2

            log_values.update({
                f"{prefix}/mse_x": action_error_x,
                f"{prefix}/mse_y": action_error_y,
                f"{prefix}/mse_touch": action_error_pen,
                f"{prefix}/action_coord_error": action_error_overall,
                f"{prefix}/action_pen_error": action_error_pen,
            })

            return log_values, [fig, jepa_fig]

        merged_log = {"global_step": self.global_step}
        figures_to_close = []

        for split_name, data in split_collections.items():
            split_log, split_figs = build_visualization_payload(
                split_label=split_name,
                generated_list=data["generated"],
                ground_list=data["ground"],
                visual_list=data["visual"],
                seq_lengths=data["sequence_lengths"],
            )
            merged_log.update(split_log)
            figures_to_close.extend(split_figs)

        if custom_embeddings is not None:
            merged_log[f"{log_prefix}/sample_info"] = "overfit_exact_sample"
            self.logger.info("Generated samples for OVERFIT TEST - showing the exact sample being trained")
        else:
            def _count_samples(split_key: str) -> int:
                if split_key not in split_collections:
                    return 0
                return sum(t.shape[0] for t in split_collections[split_key]["generated"])

            num_train = _count_samples("train")
            num_val = _count_samples("val")
            total_samples = sum(t.shape[0] for t in all_generated_gestures)
            if num_train:
                merged_log["train/num_samples"] = num_train
            if num_val:
                merged_log["val/num_samples"] = num_val
            self.logger.info(f"Generated {total_samples} samples: {num_train} train + {num_val} val with joint trajectory flow")

        wandb.log(merged_log)

        for fig in figures_to_close:
            plt.close(fig)
        self.model.train()

    def _save_checkpoint(self, tag: str, val_loss: float = None):
        """Save model checkpoint with proper versioning."""
        checkpoint_path = self.run_dir / f"flow_{tag}.pt"

        # Prepare trainer state
        trainer_state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "epoch": self.start_epoch,
        }
        if val_loss is not None:
            trainer_state["val_loss"] = val_loss

        # No LAM checkpoint needed for joint trajectory flow
        lam_checkpoint_path = None

        try:
            # Use versioned checkpoint saving
            save_gesture_flow_checkpoint(
                model=self.model,
                checkpoint_path=str(checkpoint_path),
                trainer_state=trainer_state
            )
            self.logger.info(f" Checkpoint saved to {checkpoint_path}")

            # Create latest symlink
            latest_symlink = self.run_dir / "latest.pt"
            if latest_symlink.exists():
                latest_symlink.unlink()
            latest_symlink.symlink_to(checkpoint_path.name)

        except Exception as e:
            self.logger.error(f"L Failed to save checkpoint: {e}")

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

        self.logger.info("Starting flow matching training loop...")

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
                        "train/flow_loss": loss_dict["flow_loss"],
                        "train/pen_loss": loss_dict["pen_loss"],
                        "train/state_error": loss_dict["state_error"],
                        "train/action_error": loss_dict["action_error"],
                        "train/lr": self.optimizer.param_groups[0]['lr'],
                        "train/sequence_length": loss_dict["sequence_length"],
                        "global_step": self.global_step,
                        "epoch": epoch
                    }

                    # --- ADD: Comprehensive statistics monitoring ---
                    stats_metrics = {
                        # === SCALE & NORMALIZATION ===
                        # Velocity prediction norms
                        "stats_train/velocity_pred_l2_norm": loss_dict.get("v_pred_l2_norm", 0),
                        "stats_train/velocity_pred_state_norm": loss_dict.get("v_pred_state_norm", 0),
                        "stats_train/velocity_pred_action_norm": loss_dict.get("v_pred_action_norm", 0),
                        "stats_train/velocity_target_l2_norm": loss_dict.get("v_target_l2_norm", 0),
                        "stats_train/velocity_target_state_norm": loss_dict.get("v_target_state_norm", 0),
                        "stats_train/velocity_target_action_norm": loss_dict.get("v_target_action_norm", 0),

                        # State vs Action analysis
                        "stats_train/state_action_norm_ratio": loss_dict.get("state_action_norm_ratio", 0),
                        "stats_train/state_action_loss_ratio": loss_dict.get("state_action_loss_ratio", 0),

                        # Flow input statistics
                        "stats_train/flow_input_l2_norm": loss_dict.get("x_t_l2_norm", 0),
                        "stats_train/flow_time_mean": loss_dict.get("t_mean", 0),

                        # Normalization effectiveness
                        "stats_train/norm_state_l2_norm": loss_dict.get("norm_state_l2_norm", 0),
                        "stats_train/norm_action_l2_norm": loss_dict.get("norm_action_l2_norm", 0),
                        "stats_train/norm_joint_state_part_norm": loss_dict.get("norm_joint_state_part_norm", 0),
                        "stats_train/norm_joint_action_part_norm": loss_dict.get("norm_joint_action_part_norm", 0),
                        "stats_train/norm_state_std": loss_dict.get("norm_state_std", 0),
                        "stats_train/norm_action_std": loss_dict.get("norm_action_std", 0),
                        "stats_train/norm_raw_action_l2_norm": loss_dict.get("norm_raw_action_l2_norm", 0),

                        # Normalization parameters
                        "stats_train/norm_mu_x": loss_dict.get("norm_normalization_mu_x", 0),
                        "stats_train/norm_sigma_x": loss_dict.get("norm_normalization_sigma_x", 0),
                        "stats_train/norm_mu_y": loss_dict.get("norm_normalization_mu_y", 0),
                        "stats_train/norm_sigma_y": loss_dict.get("norm_normalization_sigma_y", 0),

                        # === GRADIENT STATISTICS ===
                        "stats_train/grad_norm_total": 0,
                        "stats_train/grad_norm_model": 0,
                        "stats_train/grad_norm_state_proj": 0,
                        "stats_train/grad_max_abs": 0,

                        # === MODEL WEIGHT STATISTICS ===
                        "stats_train/model_weight_norm": 0,
                        "stats_train/state_proj_weight_norm": 0,

                        # === TRAINING DYNAMICS ===
                        "stats_train/velocity_prediction_error": abs(loss_dict.get("v_pred_l2_norm", 0) - loss_dict.get("v_target_l2_norm", 0)),
                        "stats_train/loss_balance_state_vs_action": loss_dict.get("state_mse", 0) / (loss_dict.get("action_mse", 0) + 1e-8),
                    }

                    # Add training history-based metrics
                    if len(self.loss_history) > 10:
                        recent_losses = self.loss_history[-10:]
                        stats_metrics.update({
                            "stats_train/loss_trend_10step": (recent_losses[-1] - recent_losses[0]) / len(recent_losses),
                            "stats_train/loss_variance_10step": np.var(recent_losses),
                            "stats_train/loss_smoothness": np.std(np.diff(recent_losses)),  # How smooth is training
                        })

                    if len(self.loss_history) > 100:
                        recent_losses_100 = self.loss_history[-100:]
                        stats_metrics.update({
                            "stats_train/loss_trend_100step": (recent_losses_100[-1] - recent_losses_100[0]) / len(recent_losses_100),
                            "stats_train/convergence_rate": abs(recent_losses_100[-1] - recent_losses_100[-10]) / 10,  # How fast we're converging
                        })

                    # Compute gradient statistics
                    try:
                        total_grad_norm = 0
                        model_grad_norm = 0
                        state_proj_grad_norm = 0
                        max_grad = 0

                        # Model gradients
                        for param in self.model.parameters():
                            if param.grad is not None:
                                param_norm = param.grad.data.norm(2)
                                model_grad_norm += param_norm.item() ** 2
                                max_grad = max(max_grad, param.grad.data.abs().max().item())
                        model_grad_norm = model_grad_norm ** 0.5

                        # State projection gradients
                        for param in self.state_proj.parameters():
                            if param.grad is not None:
                                param_norm = param.grad.data.norm(2)
                                state_proj_grad_norm += param_norm.item() ** 2
                        state_proj_grad_norm = state_proj_grad_norm ** 0.5

                        total_grad_norm = (model_grad_norm ** 2 + state_proj_grad_norm ** 2) ** 0.5

                        stats_metrics.update({
                            "stats_train/grad_norm_total": total_grad_norm,
                            "stats_train/grad_norm_model": model_grad_norm,
                            "stats_train/grad_norm_state_proj": state_proj_grad_norm,
                            "stats_train/grad_max_abs": max_grad,
                        })

                        # Track gradient norm history
                        self.gradient_norm_history.append(total_grad_norm)
                        if len(self.gradient_norm_history) > 1000:
                            self.gradient_norm_history = self.gradient_norm_history[-1000:]

                        # Gradient stability metrics
                        if len(self.gradient_norm_history) > 10:
                            recent_grads = self.gradient_norm_history[-10:]
                            stats_metrics.update({
                                "stats_train/grad_stability": 1.0 / (np.std(recent_grads) + 1e-8),
                                "stats_train/grad_trend": (recent_grads[-1] - recent_grads[0]) / len(recent_grads),
                            })

                    except:
                        pass  # Skip if gradients not available

                    # Compute model weight statistics
                    try:
                        model_weight_norm = sum(p.data.norm(2).item() ** 2 for p in self.model.parameters()) ** 0.5
                        state_proj_weight_norm = sum(p.data.norm(2).item() ** 2 for p in self.state_proj.parameters()) ** 0.5

                        stats_metrics.update({
                            "stats_train/model_weight_norm": model_weight_norm,
                            "stats_train/state_proj_weight_norm": state_proj_weight_norm,
                        })
                    except:
                        pass

                    log_dict.update(stats_metrics)

                    # Log FiLM magnitude histograms
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
                        self.logger.info(f"<� New best validation loss at half-epoch: {val_loss:.4f}. Saving checkpoint.")
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
                self.logger.info(f"<� New best validation loss at end-of-epoch: {val_loss:.4f}. Saving checkpoint.")
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
                                   f"Flow = {loss_dict['flow_loss']:.6f}, "
                                   f"Pen = {loss_dict['pen_loss']:.6f}, "
                                   f"State = {loss_dict['state_error']:.6f}, "
                                   f"Action = {loss_dict['action_error']:.6f}")

                # Log to wandb
                if self.args.wandb_project and total_steps % 10 == 0:
                    log_dict = {
                        "overfit/total_loss": loss_dict["loss"],
                        "overfit/flow_loss": loss_dict["flow_loss"],
                        "overfit/pen_loss": loss_dict["pen_loss"],
                        "overfit/state_error": loss_dict["state_error"],
                        "overfit/action_error": loss_dict["action_error"],
                        "overfit/lr": self.optimizer.param_groups[0]['lr'],
                        "overfit/sequence_length": loss_dict["sequence_length"],
                        "overfit/step": total_steps,
                        "overfit/epoch": epoch
                    }

                    # --- ADD: Comprehensive statistics for overfit mode ---
                    overfit_stats = {
                        # Scale & Normalization
                        "stats_overfit/velocity_pred_state_norm": loss_dict.get("v_pred_state_norm", 0),
                        "stats_overfit/velocity_pred_action_norm": loss_dict.get("v_pred_action_norm", 0),
                        "stats_overfit/state_action_norm_ratio": loss_dict.get("state_action_norm_ratio", 0),
                        "stats_overfit/state_action_loss_ratio": loss_dict.get("state_action_loss_ratio", 0),
                        "stats_overfit/norm_state_l2_norm": loss_dict.get("norm_state_l2_norm", 0),
                        "stats_overfit/norm_action_l2_norm": loss_dict.get("norm_action_l2_norm", 0),
                        "stats_overfit/norm_state_std": loss_dict.get("norm_state_std", 0),
                        "stats_overfit/norm_action_std": loss_dict.get("norm_action_std", 0),

                        # Training dynamics for single sample
                        "stats_overfit/velocity_prediction_error": abs(loss_dict.get("v_pred_l2_norm", 0) - loss_dict.get("v_target_l2_norm", 0)),
                        "stats_overfit/loss_balance_state_vs_action": loss_dict.get("state_mse", 0) / (loss_dict.get("action_mse", 0) + 1e-8),
                    }
                    log_dict.update(overfit_stats)

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
    parser = argparse.ArgumentParser(description="Train Gesture Flow Matching Model")

    # Data args
    parser.add_argument("--processed_data_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/flow_gesture")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume training from")

    # Model args
    parser.add_argument("--d_latent", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--unet_channels", type=int, nargs='+', default=[64, 128, 256, 512])
    parser.add_argument("--time_embed_dim", type=int, default=128)
    parser.add_argument("--max_sequence_length", type=int, default=1000)
    parser.add_argument("--max_sequence_length_frames", type=int, default=4,
                       help="Maximum number of latent frames to use (N_max). Training randomly samples from [1, N_max]")

    # Joint trajectory flow args
    parser.add_argument("--d_state", type=int, default=64,
                       help="State token dimension after pooling/projection")
    parser.add_argument("--include_pen_in_flow", action="store_true",
                       help="Include pen channel in joint flow (otherwise use 2D coordinates only)")
    parser.add_argument("--state_weight", type=float, default=0.5,
                       help="Loss weight for state tokens relative to action tokens")
    parser.add_argument("--no_film", action="store_true",
                       help="Disable FiLM conditioning (use unconditioned joint trajectory flow)")
    parser.add_argument("--film_dim", type=int, default=128,
                       help="FiLM conditioning dimension (ignored if --no_film)")


    # Flow matching args (replacing diffusion args)
    parser.add_argument("--rf_noisy", action="store_true",
                       help="Use noisy rectified flow g(t) smoothing")
    parser.add_argument("--rf_sigma", type=float, default=0.1,
                       help="Noise scale for g(t)=sigma*t*(1-t) if --rf_noisy")
    parser.add_argument("--flow_steps", type=int, default=20,
                       help="ODE steps for inference (Euler/Heun)")
    parser.add_argument("--ode_method", type=str, default="heun", choices=["euler","heun"],
                       help="ODE integration method")

    # Training args
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Loss weighting
    parser.add_argument("--velocity_weight", type=float, default=1.0,
                       help="Weight for velocity MSE loss in L = L_MSE(v) + gamma * L_BCE(p)")
    parser.add_argument("--pen_weight", type=float, default=1.0,
                       help="Weight for pen state BCE loss (gamma in the formula)")

    # Logging and evaluation
    parser.add_argument("--wandb_project", type=str, default="flow-gesture-generator")
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
        trainer = FlowTrainer(args)
        trainer.training_loop()
        print("Training finished successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise


if __name__ == "__main__":
    main()
