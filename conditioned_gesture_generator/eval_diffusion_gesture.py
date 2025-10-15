import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import h5py

from conditioned_gesture_generator.diffusion_gesture_model import (
    GestureDiffusionModel, create_diffusion_schedule, DDIMSampler
)
from training.dataloader import init_preprocessed_data_loader
from gui_world_model.encoder import VJEPA2Wrapper

logger = logging.getLogger(__name__)


class GestureDiffusionEvaluator:
    """Evaluator for Gesture Diffusion Model."""

    def __init__(
        self,
        model_path: str,
        device: str = None,
        use_wandb: bool = True,
        wandb_project: str = "gesture-diffusion-eval",
        wandb_run_name: str = None,
        num_inference_steps: int = 50,
        cfg_scale: float = 7.5,
        cfg_alpha: float = 0.0
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_wandb = use_wandb
        self.num_inference_steps = num_inference_steps
        self.cfg_scale = cfg_scale
        self.cfg_alpha = cfg_alpha

        # Load diffusion model
        self.model, self.diffusion_schedule, self.sampler = self._load_model(model_path)

        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name or f"eval_{self._get_timestamp()}",
                config={
                    "model_path": model_path,
                    "device": self.device,
                    "model_type": "gesture_diffusion",
                    "num_inference_steps": self.num_inference_steps,
                    "cfg_scale": self.cfg_scale,
                    "cfg_alpha": self.cfg_alpha
                }
            )

    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _load_model(self, model_path: str) -> Tuple[GestureDiffusionModel, Dict, DDIMSampler]:
        """Load gesture diffusion model from checkpoint."""
        logger.info(f"Loading gesture diffusion model from {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract model config from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Try to infer from model state dict or use defaults
            logger.warning("No config found in checkpoint, using default parameters")
            config = {}

        # Model parameters with defaults
        d_latent = config.get('d_latent', 1024)
        d_model = config.get('d_model', 512)
        channels = config.get('unet_channels', [64, 128, 256, 512])
        time_embed_dim = config.get('time_embed_dim', 128)
        max_sequence_length = config.get('max_sequence_length', 1000)
        cfg_dropout_prob = config.get('cfg_dropout_prob', 0.1)

        # Create model
        model = GestureDiffusionModel(
            d_latent=d_latent,
            d_model=d_model,
            channels=tuple(channels),
            time_embed_dim=time_embed_dim,
            max_sequence_length=max_sequence_length,
            cfg_dropout_prob=cfg_dropout_prob
        )

        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device).eval()

        # Create diffusion schedule
        num_timesteps = config.get('num_timesteps', 1000)
        beta_start = config.get('beta_start', 1e-4)
        beta_end = config.get('beta_end', 0.02)

        diffusion_schedule = create_diffusion_schedule(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end
        )

        # Move schedule to device
        for key in diffusion_schedule:
            diffusion_schedule[key] = diffusion_schedule[key].to(self.device)

        # Create sampler
        sampler = DDIMSampler(model, diffusion_schedule, device=self.device)

        logger.info(f"Gesture diffusion model loaded successfully on {self.device}")
        return model, diffusion_schedule, sampler

    def _load_corresponding_images(self, session_id: str, traj_idx: int,
                                 sequence_length: int, image_dir: str) -> Optional[torch.Tensor]:
        """Load corresponding images for a trajectory sequence."""
        image_path = Path(image_dir) / f"{session_id}_images.h5"

        if not image_path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return None

        try:
            with h5py.File(image_path, 'r') as f:
                if 'images' not in f:
                    logger.warning(f"No 'images' key in {image_path}")
                    return None

                # For a sequence of `sequence_length` actions, we have `sequence_length + 1` states.
                # Each state s_i is composed of frames (f_{2i}, f_{2i+1}).
                # The gesture a_i occurs between f_{2i+1} and f_{2i+2}.
                # We load all frames for the relevant states to allow flexible visualization.
                num_frames_to_load = (sequence_length + 1) * 2

                available_frames = f['images'].shape[1]
                if num_frames_to_load > available_frames:
                    logger.warning(f"Requested {num_frames_to_load} frames, but only {available_frames} available. "
                                   f"Loading all available frames for {session_id}, trajectory {traj_idx}.")
                    num_frames_to_load = available_frames

                images = []
                for i in range(num_frames_to_load):
                    img = f['images'][traj_idx, i]  # [H, W, C]
                    images.append(torch.from_numpy(img))

                if images:
                    return torch.stack(images)  # [num_frames, H, W, C]
                else:
                    return None

        except Exception as e:
            logger.warning(f"Error loading images from {image_path}: {e}")
            return None

    def _create_trajectory_visualization(self, images: torch.Tensor, generated_gestures: torch.Tensor,
                                       ground_truth_gestures: torch.Tensor, sequence_length: int,
                                       session_id: str, traj_idx: int, split_type: str) -> plt.Figure:
        """Create comprehensive trajectory visualization showing image→gesture→next_image progression.

        As per user specification, this visualization provides a high-level view of the transition
        s_i -> s_{i+1}. It shows the gesture plotted on the first frame of the start state (f_{2i})
        and the resulting last frame of the end state (f_{2i+3}).

        Visualization shows: f_{2i} → gesture a_i → f_{2i+3} for each step i.
        """
        # images: [num_frames, H, W, C] - f_0, f_1, f_2, ...
        # generated_gestures: [sequence_length*250, 3] (flattened)
        # ground_truth_gestures: [sequence_length*250, 3] (flattened)

        # Reshape gestures to [sequence_length, 250, 3]
        gen_gestures = generated_gestures.view(sequence_length, 250, 3)
        gt_gestures = ground_truth_gestures.view(sequence_length, 250, 3)

        # Create grid layout: 4 rows x sequence_length columns
        # Row 1: Source images (f_0, f_2, f_4...)
        # Row 2: Source images with generated gestures overlaid
        # Row 3: Source images with ground truth gestures overlaid
        # Row 4: Target images (f_3, f_5, f_7...)
        fig, axes = plt.subplots(4, sequence_length, figsize=(4 * sequence_length, 16))

        if sequence_length == 1:
            axes = axes.reshape(4, 1)

        for seq_idx in range(sequence_length):
            # For action a_i (seq_idx), visualize f_{2i} -> a_i -> f_{2i+3}
            source_image_idx = 2 * seq_idx
            target_image_idx = 2 * seq_idx + 3

            if target_image_idx >= len(images):
                logger.warning(f"Not enough images for step {seq_idx} in trajectory {session_id}_{traj_idx}. "
                               f"Need index {target_image_idx}, have {len(images)} images. Skipping this step.")
                for row in range(4):
                    axes[row, seq_idx].axis('off')
                continue

            source_image = images[source_image_idx]
            target_image = images[target_image_idx]
            gen_gesture = gen_gestures[seq_idx]
            gt_gesture = gt_gestures[seq_idx]

            # Convert source image to displayable format
            if source_image.dim() == 3 and source_image.shape[-1] == 3:
                source_np = source_image.cpu().numpy()
            else:
                source_np = source_image.permute(1, 2, 0).cpu().numpy()

            # Convert target image to displayable format
            if target_image.dim() == 3 and target_image.shape[-1] == 3:
                target_np = target_image.cpu().numpy()
            else:
                target_np = target_image.permute(1, 2, 0).cpu().numpy()

            # Normalize images to [0, 1]
            if source_np.max() > 1.0:
                source_np = source_np / 255.0
            elif source_np.min() < 0:
                source_np = (source_np + 1.0) / 2.0

            if target_np.max() > 1.0:
                target_np = target_np / 255.0
            elif target_np.min() < 0:
                target_np = (target_np + 1.0) / 2.0

            # Row 1: Source image (f_{2i})
            ax1 = axes[0, seq_idx]
            ax1.imshow(source_np)
            ax1.set_title(f"Frame {source_image_idx} (Start of s_{seq_idx})", fontsize=10)
            ax1.axis('off')

            # Row 2: Source image with generated gesture overlay
            ax2 = axes[1, seq_idx]
            ax2.imshow(source_np)
            self._plot_gesture_overlay(ax2, gen_gesture, source_np.shape[:2], color='red', label='Generated')
            ax2.set_title(f"Generated Gesture on Frame {source_image_idx}", fontsize=10)
            ax2.axis('off')

            # Row 3: Source image with ground truth gesture overlay
            ax3 = axes[2, seq_idx]
            ax3.imshow(source_np)
            self._plot_gesture_overlay(ax3, gt_gesture, source_np.shape[:2], color='blue', label='Ground Truth')
            ax3.set_title(f"GT Gesture on Frame {source_image_idx}", fontsize=10)
            ax3.axis('off')

            # Row 4: Target image (f_{2i+3})
            ax4 = axes[3, seq_idx]
            ax4.imshow(target_np)
            ax4.set_title(f"Frame {target_image_idx} (End of s_{seq_idx + 1})", fontsize=10)
            ax4.axis('off')

        plt.suptitle(f"Trajectory Progression - {split_type.upper()} - {session_id}_{traj_idx}", fontsize=14)
        plt.tight_layout()
        return fig

    def _plot_gesture_overlay(self, ax, gesture: torch.Tensor, img_shape: tuple,
                            color: str = 'red', label: str = '') -> None:
        """Plot gesture overlay on axis, only showing pen-down segments."""
        # Extract coordinates and pen state
        coords = gesture[..., :2].cpu().numpy()  # [L, 2]
        pen_state = gesture[..., 2].cpu().numpy()  # [L]

        # Convert to pixel coordinates
        coords_px = self._denormalize_coordinates(torch.from_numpy(coords), img_shape[0]).numpy()

        # Plot only pen-down segments (p > 0.5)
        pen_down_mask = pen_state > 0.5

        if np.any(pen_down_mask):
            # Find continuous pen-down segments
            pen_changes = np.diff(np.concatenate(([False], pen_down_mask, [False])).astype(int))
            starts = np.where(pen_changes == 1)[0]
            ends = np.where(pen_changes == -1)[0]

            # Plot each segment
            for i, (start, end) in enumerate(zip(starts, ends)):
                if end > start + 1:  # Need at least 2 points
                    segment_coords = coords_px[start:end]
                    ax.plot(segment_coords[:, 0], segment_coords[:, 1],
                           color=color, linewidth=2, alpha=0.8,
                           label=label if i == 0 else "")

            # Mark start and end points
            if len(starts) > 0:
                start_pt = coords_px[starts[0]]
                ax.scatter(start_pt[0], start_pt[1], c='green', s=80, marker='o',
                          alpha=0.9, edgecolors='black', linewidth=1)

            if len(ends) > 0:
                end_pt = coords_px[ends[-1] - 1]  # ends is exclusive
                ax.scatter(end_pt[0], end_pt[1], c='red', s=80, marker='s',
                          alpha=0.9, edgecolors='black', linewidth=1)

        ax.set_xlim(0, img_shape[1])
        ax.set_ylim(img_shape[0], 0)  # Flip y-axis for image coordinates

    def _denormalize_coordinates(self, coords: torch.Tensor, img_size: int = 256) -> torch.Tensor:
        """Convert normalized coordinates [0,1] to pixel coordinates."""
        return coords * img_size

    def _plot_gesture_on_image(self, image: torch.Tensor, gesture: torch.Tensor,
                              title: str = "", ax=None) -> plt.Figure:
        """Plot gesture trajectory on image, only showing pen-down segments."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        else:
            fig = ax.figure

        # Convert image to displayable format
        if image.dim() == 3 and image.shape[0] == 3:
            # CHW format, convert to HWC
            img_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            # Already HWC
            img_np = image.cpu().numpy()

        # Ensure image is in [0, 1] range
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        elif img_np.min() < 0:
            img_np = (img_np + 1.0) / 2.0  # Convert from [-1,1] to [0,1]

        ax.imshow(img_np)

        # Extract coordinates and pen state
        coords = gesture[..., :2].cpu().numpy()  # [L, 2]
        pen_state = gesture[..., 2].cpu().numpy()  # [L]

        # Convert to pixel coordinates
        coords_px = self._denormalize_coordinates(torch.from_numpy(coords), img_np.shape[0]).numpy()

        # Plot only pen-down segments (p > 0.5)
        pen_down_mask = pen_state > 0.5

        if np.any(pen_down_mask):
            # Find continuous pen-down segments
            pen_changes = np.diff(np.concatenate(([False], pen_down_mask, [False])).astype(int))
            starts = np.where(pen_changes == 1)[0]
            ends = np.where(pen_changes == -1)[0]

            # Plot each segment
            for start, end in zip(starts, ends):
                if end > start + 1:  # Need at least 2 points
                    segment_coords = coords_px[start:end]
                    ax.plot(segment_coords[:, 0], segment_coords[:, 1],
                           'r-', linewidth=2, alpha=0.8)

            # Mark start and end points
            if len(starts) > 0:
                start_pt = coords_px[starts[0]]
                ax.scatter(start_pt[0], start_pt[1], c='green', s=100, marker='o',
                          alpha=0.9, edgecolors='black', linewidth=2, label='Start')

            if len(ends) > 0:
                end_pt = coords_px[ends[-1] - 1]  # ends is exclusive
                ax.scatter(end_pt[0], end_pt[1], c='red', s=100, marker='s',
                          alpha=0.9, edgecolors='black', linewidth=2, label='End')

            ax.legend()

        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, img_np.shape[1])
        ax.set_ylim(img_np.shape[0], 0)  # Flip y-axis for image coordinates
        ax.grid(True, alpha=0.3)

        return fig

    def _create_sequence_visualization(self, images: torch.Tensor, gestures: torch.Tensor,
                                     sequence_length: int, sample_idx: int = 0) -> plt.Figure:
        """Create visualization showing image→gesture→next_image progression."""
        # images: [sequence_length+1, H, W, C]
        # gestures: [sequence_length*250, 3] (flattened) or [sequence_length, 250, 3]

        if gestures.dim() == 2:
            # Reshape from [sequence_length*250, 3] to [sequence_length, 250, 3]
            gestures = gestures.view(sequence_length, 250, 3)

        # Create grid layout: sequence_length rows, 3 columns (image, gesture, next_image)
        fig, axes = plt.subplots(sequence_length, 3, figsize=(15, 5 * sequence_length))
        if sequence_length == 1:
            axes = axes.reshape(1, -1)

        for seq_idx in range(sequence_length):
            current_image = images[seq_idx]  # [H, W, C]
            next_image = images[seq_idx + 1]  # [H, W, C]
            gesture = gestures[seq_idx]  # [250, 3]

            # Column 1: Current image
            ax1 = axes[seq_idx, 0]
            if current_image.dim() == 3 and current_image.shape[-1] == 3:
                img_np = current_image.cpu().numpy()
            else:
                img_np = current_image.permute(1, 2, 0).cpu().numpy()

            if img_np.max() > 1.0:
                img_np = img_np / 255.0
            elif img_np.min() < 0:
                img_np = (img_np + 1.0) / 2.0

            ax1.imshow(img_np)
            ax1.set_title(f"Frame {seq_idx}", fontsize=10)
            ax1.axis('off')

            # Column 2: Gesture on current image
            ax2 = axes[seq_idx, 1]
            self._plot_gesture_on_image(current_image, gesture,
                                      f"Gesture {seq_idx}", ax=ax2)

            # Column 3: Next image
            ax3 = axes[seq_idx, 2]
            if next_image.dim() == 3 and next_image.shape[-1] == 3:
                img_np = next_image.cpu().numpy()
            else:
                img_np = next_image.permute(1, 2, 0).cpu().numpy()

            if img_np.max() > 1.0:
                img_np = img_np / 255.0
            elif img_np.min() < 0:
                img_np = (img_np + 1.0) / 2.0

            ax3.imshow(img_np)
            ax3.set_title(f"Frame {seq_idx + 1}", fontsize=10)
            ax3.axis('off')

        plt.suptitle(f"Gesture Sequence Visualization - Sample {sample_idx}", fontsize=14)
        plt.tight_layout()
        return fig

    def _generate_with_cfg_modes(self, shape: Tuple[int, int, int],
                                latent_frames: torch.Tensor,
                                sequence_length: int) -> Dict[str, torch.Tensor]:
        """Generate gestures with different CFG modes based on cfg_alpha."""
        results = {}

        if self.cfg_alpha == 0.0:
            # Only conditioned generation
            generated = self.sampler.sample(
                shape=shape,
                latent_frames=latent_frames,
                sequence_length=sequence_length,
                num_inference_steps=self.num_inference_steps,
                cfg_scale=1.0,  # No guidance
                eta=0.0
            )
            results['conditioned'] = generated
        else:
            # Generate all three variants
            # 1. Conditioned only
            generated_cond = self.sampler.sample(
                shape=shape,
                latent_frames=latent_frames,
                sequence_length=sequence_length,
                num_inference_steps=self.num_inference_steps,
                cfg_scale=1.0,  # No guidance
                eta=0.0
            )
            results['conditioned'] = generated_cond

            # 2. Unconditional only
            generated_uncond = self.sampler.sample(
                shape=shape,
                latent_frames=None,  # No conditioning
                sequence_length=None,  # No sequence length for unconditional
                num_inference_steps=self.num_inference_steps,
                cfg_scale=1.0,  # No guidance
                eta=0.0
            )
            results['unconditioned'] = generated_uncond

            # 3. CFG guided (using cfg_alpha as the guidance strength)
            generated_cfg = self.sampler.sample(
                shape=shape,
                latent_frames=latent_frames,
                sequence_length=sequence_length,
                num_inference_steps=self.num_inference_steps,
                cfg_scale=1.0 + self.cfg_alpha,  # Convert cfg_alpha to cfg_scale format
                eta=0.0
            )
            results['cfg'] = generated_cfg

        return results

    def _load_trajectory_data(self, processed_data_dir: str, manifest_path: str,
                             split_name: str, n_trajectories: int = 20) -> List[Tuple]:
        """Load specific trajectories from dataset."""
        import json

        # Load manifest to get file list
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        if split_name not in manifest['splits']:
            raise ValueError(f"Split '{split_name}' not found in manifest")

        file_list = manifest['splits'][split_name][:n_trajectories]  # Take first n_trajectories files

        trajectories = []
        processed_dir = Path(processed_data_dir)

        for filename in file_list:
            h5_path = processed_dir / filename
            if not h5_path.exists():
                logger.warning(f"File not found: {h5_path}")
                continue

            try:
                with h5py.File(h5_path, 'r') as f:
                    embeddings = f['embeddings'][:]  # [n_traj, T, N, D]
                    actions = f['actions'][:]        # [n_traj, T-1, Traj, 3]

                    # Extract session_id from filename (remove .h5 extension)
                    session_id = h5_path.stem

                    # Add each trajectory in this file
                    for traj_idx in range(embeddings.shape[0]):
                        trajectories.append((
                            session_id,
                            traj_idx,
                            embeddings[traj_idx],  # [T, N, D]
                            actions[traj_idx]      # [T-1, Traj, 3]
                        ))

                        if len(trajectories) >= n_trajectories:
                            break

            except Exception as e:
                logger.warning(f"Error loading {h5_path}: {e}")
                continue

        logger.info(f"Loaded {len(trajectories)} trajectories from {split_name} split")
        return trajectories

    def evaluate_dataset_trajectories(
        self,
        processed_data_dir: str,
        manifest_path: str,
        image_dir: str,
        n_train_trajectories: int = 20,
        n_val_trajectories: int = 20,
        batch_size: int = 4
    ) -> Dict[str, float]:
        """
        Evaluate complete trajectories, showing full sequence progression.

        Args:
            processed_data_dir: Directory containing preprocessed latent data
            manifest_path: Path to manifest file for data splits
            image_dir: Directory containing image h5 files
            n_train_trajectories: Number of training trajectories to visualize
            n_val_trajectories: Number of validation trajectories to visualize
            batch_size: Batch size for GPU efficiency

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {n_train_trajectories} training + {n_val_trajectories} validation trajectories...")

        # Load training trajectories
        train_trajectories = self._load_trajectory_data(
            processed_data_dir, manifest_path, 'train', n_train_trajectories
        )

        # Load validation trajectories
        val_trajectories = self._load_trajectory_data(
            processed_data_dir, manifest_path, 'validation', n_val_trajectories
        )

        all_trajectories = [('train', traj) for traj in train_trajectories] + \
                          [('val', traj) for traj in val_trajectories]

        all_losses = []

        with torch.no_grad():
            # Process trajectories in batches for GPU efficiency
            for batch_start in tqdm(range(0, len(all_trajectories), batch_size), desc="Processing trajectory batches"):
                batch_trajectories = all_trajectories[batch_start:batch_start + batch_size]

                # Group trajectories by sequence length for efficient batching
                batch_by_length = {}
                for split_type, (session_id, traj_idx, embeddings, actions) in batch_trajectories:
                    seq_len = actions.shape[0]  # T-1
                    if seq_len not in batch_by_length:
                        batch_by_length[seq_len] = []
                    batch_by_length[seq_len].append((split_type, session_id, traj_idx, embeddings, actions))

                # Process each length group as a batch
                for sequence_length, traj_group in batch_by_length.items():
                    if not traj_group:
                        continue

                    # Stack trajectories into batch
                    batch_embeddings = []
                    batch_actions = []
                    batch_metadata = []

                    for split_type, session_id, traj_idx, embeddings, actions in traj_group:
                        batch_embeddings.append(torch.from_numpy(embeddings).float())
                        batch_actions.append(torch.from_numpy(actions).float())
                        batch_metadata.append((split_type, session_id, traj_idx))

                    # Stack into batch tensors
                    visual_embeddings = torch.stack(batch_embeddings).to(self.device)  # [B, T, N, D]
                    ground_truth_actions = torch.stack(batch_actions).to(self.device)  # [B, T-1, Traj, 3]

                    B, T_minus_1, Traj, _ = ground_truth_actions.shape

                    # Prepare data for generation
                    gesture_sequences = ground_truth_actions.contiguous().view(B, sequence_length * Traj, 3)
                    latent_frames = visual_embeddings[:, :sequence_length + 1].contiguous()

                    # Generate gestures using the diffusion model with different CFG modes
                    shape = (B, sequence_length * Traj, 3)
                    generation_results = self._generate_with_cfg_modes(
                        shape=shape,
                        latent_frames=latent_frames,
                        sequence_length=sequence_length
                    )

                    # Process each trajectory in the batch for visualization and loss computation
                    for i, (split_type, session_id, traj_idx) in enumerate(batch_metadata):
                        # Load corresponding images
                        images = self._load_corresponding_images(session_id, traj_idx, sequence_length, image_dir)

                        # Create visualizations for each generation mode
                        for mode_name, generated_sequences in generation_results.items():
                            if images is not None:
                                # Create trajectory visualization
                                fig = self._create_trajectory_visualization(
                                    images=images,  # [T, H, W, C]
                                    generated_gestures=generated_sequences[i],  # [sequence_length*250, 3]
                                    ground_truth_gestures=gesture_sequences[i],  # [sequence_length*250, 3]
                                    sequence_length=sequence_length,
                                    session_id=session_id,
                                    traj_idx=traj_idx,
                                    split_type=split_type
                                )

                                if self.use_wandb:
                                    wandb.log({
                                        f"{mode_name}/{split_type}_trajectories/trajectory_{session_id}_{traj_idx}": wandb.Image(fig)
                                    })

                                plt.close(fig)

                            # Compute losses for the primary mode (conditioned or cfg)
                            if mode_name in ['conditioned', 'cfg']:
                                gt = gesture_sequences[i]
                                gen = generated_sequences[i]

                                # Compute coordinate L1 loss (only pen-down regions)
                                gt_coords = gt[..., :2]
                                gen_coords = gen[..., :2]
                                gt_pen = gt[..., 2:3]
                                gen_pen = gen[..., 2:3]

                                # Union mask (pen down in either GT or generated)
                                gt_pen_down = (gt_pen > 0.5).float()
                                gen_pen_down = (gen_pen > 0.5).float()
                                union_mask = torch.clamp(gt_pen_down + gen_pen_down, 0.0, 1.0)

                                # Masked coordinate loss
                                coord_diff = torch.abs(gen_coords - gt_coords)
                                masked_coord_loss = coord_diff * union_mask
                                coord_l1_loss = masked_coord_loss.sum() / (union_mask.sum() + 1e-8)

                                # Pen BCE loss
                                pen_bce_loss = F.binary_cross_entropy(gen_pen, gt_pen, reduction='mean')

                                # Combined loss
                                total_loss = coord_l1_loss + pen_bce_loss

                                # Only add loss once per trajectory (use the main generation mode)
                                if (self.cfg_alpha == 0.0 and mode_name == 'conditioned') or \
                                   (self.cfg_alpha != 0.0 and mode_name == 'cfg'):
                                    all_losses.append(total_loss.item())

        # Compute statistics
        losses = np.array(all_losses)
        metrics = {
            'mean_loss': float(np.mean(losses)),
            'std_loss': float(np.std(losses)),
            'min_loss': float(np.min(losses)),
            'max_loss': float(np.max(losses)),
            'median_loss': float(np.median(losses)),
            'n_trajectories': len(all_trajectories)
        }

        # Log metrics
        if self.use_wandb:
            wandb.log({
                "trajectory_eval/metrics": metrics,
                "trajectory_eval/loss_distribution": wandb.Histogram(losses)
            })

        logger.info(f"Trajectory evaluation complete. Mean loss: {metrics['mean_loss']:.4f}")
        return metrics

    def evaluate_arbitrary_images(
        self,
        image_folder: str,
        batch_size: int = 4,
        sequence_length: int = 2
    ) -> Dict[str, float]:
        """
        Evaluate on arbitrary images using V-JEPA2 encoder.

        Args:
            image_folder: Directory containing images
            batch_size: Batch size for processing
            sequence_length: Number of gesture segments to generate

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating arbitrary images from {image_folder}...")

        # Initialize V-JEPA2 encoder
        vjepa_encoder = VJEPA2Wrapper(device=self.device, num_frames=16, image_size=256)

        # Find all image files
        image_folder = Path(image_folder)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(image_folder.glob(f'*{ext}'))
            image_paths.extend(image_folder.glob(f'*{ext.upper()}'))

        if not image_paths:
            raise ValueError(f"No images found in {image_folder}")

        logger.info(f"Found {len(image_paths)} images")

        # Group images into sequences
        # For simplicity, take consecutive images as sequences
        sequences = []
        for i in range(0, len(image_paths), sequence_length + 1):
            if i + sequence_length < len(image_paths):
                sequences.append(image_paths[i:i + sequence_length + 1])

        logger.info(f"Created {len(sequences)} image sequences")

        images_processed = 0

        # Process sequences in batches
        for seq_batch_start in tqdm(range(0, len(sequences), batch_size), desc="Processing image sequences"):
            seq_batch = sequences[seq_batch_start:seq_batch_start + batch_size]

            batch_latents = []
            batch_images = []

            for seq_paths in seq_batch:
                # Load and process images in sequence
                sequence_images = []
                sequence_frames = []

                for img_path in seq_paths:
                    try:
                        # Load image
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue

                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_resized = cv2.resize(img_rgb, (256, 256))
                        sequence_images.append(img_resized)

                        # Create frames for V-JEPA2 (2 frames per latent)
                        frames = np.stack([img_resized] * 2)  # [2, H, W, C]
                        for frame in frames:
                            frame_tensor = vjepa_encoder.transform(frame)
                            sequence_frames.append(frame_tensor)

                    except Exception as e:
                        logger.warning(f"Failed to process {img_path}: {e}")
                        continue

                if len(sequence_images) == sequence_length + 1:
                    # Encode sequence with V-JEPA2
                    video_tensor = torch.stack(sequence_frames).unsqueeze(0)  # [1, T*2, C, H, W]
                    video_tensor = video_tensor.to(self.device)

                    with torch.no_grad():
                        latents = vjepa_encoder(video_tensor)  # [1, T, N, D]

                    batch_latents.append(latents[0])  # [T, N, D]
                    batch_images.append(torch.stack([torch.from_numpy(img) for img in sequence_images]))

            if not batch_latents:
                continue

            # Stack batch
            latent_frames = torch.stack(batch_latents).to(self.device)  # [B, T, N, D]
            image_sequences = torch.stack(batch_images)  # [B, T, H, W, C]

            # Generate gestures with different CFG modes
            B = latent_frames.shape[0]
            shape = (B, sequence_length * 250, 3)  # 250 timesteps per segment

            with torch.no_grad():
                generation_results = self._generate_with_cfg_modes(
                    shape=shape,
                    latent_frames=latent_frames,
                    sequence_length=sequence_length
                )

            # Create visualizations for each mode
            for i in range(B):
                if images_processed >= 10:  # Limit number of visualizations
                    break

                for mode_name, generated_sequences in generation_results.items():
                    fig = self._create_sequence_visualization(
                        image_sequences[i],  # [T, H, W, C]
                        generated_sequences[i],  # [sequence_length*250, 3]
                        sequence_length,
                        sample_idx=images_processed
                    )

                    if self.use_wandb:
                        wandb.log({
                            f"{mode_name}/arbitrary_eval/sequence_{images_processed}": wandb.Image(fig)
                        })

                    plt.close(fig)

                images_processed += 1

        metrics = {
            'sequences_processed': images_processed,
            'total_sequences': len(sequences)
        }

        logger.info(f"Arbitrary images evaluation complete. Processed {images_processed} sequences")
        return metrics

    def run_full_evaluation(
        self,
        processed_data_dir: str = None,
        manifest_path: str = None,
        image_dir: str = None,
        arbitrary_image_folder: str = None,
        n_train_trajectories: int = 20,
        n_val_trajectories: int = 20,
        batch_size: int = 4
    ):
        """Run complete evaluation pipeline."""
        results = {}

        # Part 1: Trajectory evaluation
        if processed_data_dir and manifest_path and image_dir:
            logger.info("Running trajectory evaluation...")
            try:
                trajectory_metrics = self.evaluate_dataset_trajectories(
                    processed_data_dir=processed_data_dir,
                    manifest_path=manifest_path,
                    image_dir=image_dir,
                    n_train_trajectories=n_train_trajectories,
                    n_val_trajectories=n_val_trajectories,
                    batch_size=batch_size
                )
                results['trajectories'] = trajectory_metrics
            except Exception as e:
                logger.error(f"Trajectory evaluation failed: {e}")
                results['trajectories'] = None
        else:
            logger.info("Skipping trajectory evaluation (missing processed_data_dir, manifest_path, or image_dir)")
            results['trajectories'] = None

        # Part 2: Arbitrary images evaluation
        if arbitrary_image_folder:
            logger.info("Running arbitrary images evaluation...")
            try:
                arbitrary_metrics = self.evaluate_arbitrary_images(
                    image_folder=arbitrary_image_folder,
                    batch_size=batch_size
                )
                results['arbitrary'] = arbitrary_metrics
            except Exception as e:
                logger.error(f"Arbitrary images evaluation failed: {e}")
                results['arbitrary'] = None
        else:
            logger.info("Skipping arbitrary images evaluation (arbitrary_image_folder not provided)")
            results['arbitrary'] = None

        # Log summary
        if self.use_wandb:
            wandb.log({"evaluation_summary": results})

        logger.info(f"Evaluation complete! Results: {json.dumps(results, indent=2)}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gesture Diffusion Model")

    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")

    # Dataset evaluation arguments (optional)
    parser.add_argument("--processed_data_dir", type=str, default=None,
                       help="Directory containing preprocessed latent data (optional)")
    parser.add_argument("--manifest_path", type=str, default=None,
                       help="Path to manifest file for data splits (optional)")
    parser.add_argument("--image_dir", type=str, default=None,
                       help="Directory containing image h5 files (required for trajectory visualization)")
    parser.add_argument("--n_train_trajectories", type=int, default=20,
                       help="Number of training trajectories to evaluate")
    parser.add_argument("--n_val_trajectories", type=int, default=20,
                       help="Number of validation trajectories to evaluate")

    # Arbitrary images evaluation arguments (optional)
    parser.add_argument("--arbitrary_image_folder", type=str, default=None,
                       help="Folder containing arbitrary images to evaluate (optional)")

    # General arguments
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu), auto-detect if not specified")

    # Diffusion sampling arguments
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of DDIM inference steps")
    parser.add_argument("--cfg_scale", type=float, default=7.5,
                       help="Classifier-free guidance scale (legacy)")
    parser.add_argument("--cfg_alpha", type=float, default=0.0,
                       help="CFG alpha parameter. If 0, use conditioned only. If >0, generate conditioned/unconditioned/cfg variants")

    # WandB arguments
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="gesture-diffusion-eval",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")

    args = parser.parse_args()

    # Validate arguments
    has_trajectory_eval = args.processed_data_dir and args.manifest_path and args.image_dir
    has_arbitrary_eval = args.arbitrary_image_folder
    if not has_trajectory_eval and not has_arbitrary_eval:
        parser.error("Must provide either (--processed_data_dir AND --manifest_path AND --image_dir) OR --arbitrary_image_folder for evaluation")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create evaluator
    evaluator = GestureDiffusionEvaluator(
        model_path=args.model_path,
        device=args.device,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        cfg_alpha=args.cfg_alpha
    )

    # Run evaluation
    results = evaluator.run_full_evaluation(
        processed_data_dir=args.processed_data_dir,
        manifest_path=args.manifest_path,
        image_dir=args.image_dir,
        arbitrary_image_folder=args.arbitrary_image_folder,
        n_train_trajectories=args.n_train_trajectories,
        n_val_trajectories=args.n_val_trajectories,
        batch_size=args.batch_size
    )

    # Save results
    output_file = f"gesture_eval_results_{evaluator._get_timestamp()}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()