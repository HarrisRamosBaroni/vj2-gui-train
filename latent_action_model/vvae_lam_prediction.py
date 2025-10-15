"""
latent_action_model/vvae_lam_prediction.py

Evaluate VVAE LAM on test data with WandB logging:
1. Ground truth action reconstruction
2. Shuffled action application
3. Cross-context action transfer (5 actions × 5 contexts)
4. Multi-step rollout (5 steps × 5 initial states)

All results logged to WandB with frame-by-frame comparisons.
Each VVAE latent frame decodes to 4 video frames (4x temporal compression).
"""

import torch
import argparse
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import wandb
import sys
import os
import re
from typing import Dict, List, Tuple
import cv2

# Add vvae directory to Python path for imports
vvae_dir = Path(__file__).parent.parent / "vvae"
if str(vvae_dir) not in sys.path:
    sys.path.insert(0, str(vvae_dir))

from latent_action_model.vqvae import VVAELatentActionVQVAE, load_model_from_config
from latent_action_model.dataloader_vvae import VVAELAMDataset
from vvae.utils.common_utils import instantiate_from_config
from omegaconf import OmegaConf


def compute_metrics(original: torch.Tensor, predicted: torch.Tensor) -> Dict[str, float]:
    """
    Compute reconstruction metrics.

    Args:
        original: [B, C, H, W] or [B, C, T, H, W]
        predicted: Same shape as original

    Returns:
        Dictionary with mse, mae, psnr
    """
    mse = torch.mean((original - predicted) ** 2).item()
    mae = torch.mean(torch.abs(original - predicted)).item()

    # PSNR: assuming pixel values in [-1, 1] range
    # Convert to [0, 1] range for PSNR calculation
    original_01 = (original + 1) / 2
    predicted_01 = (predicted + 1) / 2
    mse_01 = torch.mean((original_01 - predicted_01) ** 2).item()
    psnr = 10 * np.log10(1.0 / (mse_01 + 1e-10))

    return {
        'mse': mse,
        'mae': mae,
        'psnr': psnr
    }


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy image for visualization.

    Args:
        tensor: [C, H, W] in range [-1, 1]

    Returns:
        numpy array [H, W, C] in range [0, 255] uint8
    """
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return img


def save_video_from_frames(frames: torch.Tensor, output_path: str, fps: int = 8):
    """
    Save video from frame tensor.

    Args:
        frames: [T, 3, H, W] tensor in range [-1, 1]
        output_path: Path to save video
        fps: Frames per second
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    T, C, H, W = frames.shape

    # Convert to numpy [T, H, W, C] in range [0, 255]
    frames_np = []
    for t in range(T):
        frame = tensor_to_image(frames[t])
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames_np.append(frame_bgr)

    # Write video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    for frame in frames_np:
        out.write(frame)

    out.release()
    print(f"Saved video to {output_path}")


class VVAELAMPredictor:
    """Evaluate VVAE LAM on test data with WandB logging."""

    def __init__(
        self,
        lam_checkpoint: str,
        lam_config: str,
        vvae_config: str,
        test_data_dir: str,
        manifest_path: str,
        device: str = "cuda",
        use_wandb: bool = True,
        wandb_project: str = "vvae-lam-prediction",
        wandb_run_name: str = None
    ):
        """
        Args:
            lam_checkpoint: Path to trained LAM checkpoint
            lam_config: Path to LAM model_config.json
            vvae_config: Path to VVAE config (e.g., config_4z.yaml)
            test_data_dir: Directory with VVAE-encoded h5 files
            manifest_path: Manifest JSON with test split
            device: Device to run on
            use_wandb: Whether to use WandB logging
            wandb_project: WandB project name
            wandb_run_name: Optional WandB run name
        """
        self.device = device
        self.use_wandb = use_wandb

        # Extract timestamp from LAM checkpoint path for output directory
        checkpoint_path = Path(lam_checkpoint)
        self.timestamp = None
        # Try to extract timestamp from path like "checkpoints/lam_20251007_184150/..."
        match = re.search(r'lam_(\d{8}_\d{6})', str(checkpoint_path))
        if match:
            self.timestamp = match.group(1)
        else:
            # Fallback to current timestamp
            from datetime import datetime
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize WandB
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    'lam_checkpoint': lam_checkpoint,
                    'lam_config': lam_config,
                    'vvae_config': vvae_config,
                    'test_data_dir': test_data_dir,
                    'timestamp': self.timestamp
                }
            )

        # Load LAM model
        print(f"Loading LAM from {lam_checkpoint}...")
        self.lam = load_model_from_config(lam_config, lam_checkpoint, device)
        self.lam.eval()

        # Load VVAE decoder for visualization
        print(f"Loading VVAE from {vvae_config}...")
        vvae_cfg = OmegaConf.load(vvae_config)
        self.vvae = instantiate_from_config(vvae_cfg.model).to(device)
        self.vvae.eval()

        # Get temporal compression factor from config
        self.temporal_compression = vvae_cfg.model.params.ppconfig.temporal_scale_factor
        print(f"VVAE temporal compression: {self.temporal_compression}x")
        print(f"Each latent frame decodes to {self.temporal_compression} video frames")

        # Create all three datasets to compare overfitting
        print("Loading train dataset...")
        self.train_dataset = VVAELAMDataset(
            data_dir=test_data_dir,
            manifest_path=manifest_path,
            split='train',
            sequence_length=8,  # Need sufficient length for rollouts
            stride=4  # Less overlap for diverse samples
        )
        print(f"Train dataset: {len(self.train_dataset)} sequences")

        print("Loading validation dataset...")
        self.val_dataset = VVAELAMDataset(
            data_dir=test_data_dir,
            manifest_path=manifest_path,
            split='validation',
            sequence_length=8,
            stride=4
        )
        print(f"Validation dataset: {len(self.val_dataset)} sequences")

        print("Loading test dataset...")
        self.test_dataset = VVAELAMDataset(
            data_dir=test_data_dir,
            manifest_path=manifest_path,
            split='test',
            sequence_length=8,
            stride=4
        )
        print(f"Test dataset: {len(self.test_dataset)} sequences")
        print("\n⚠️  MEMORY OPTIMIZATION ENABLED:")
        print("  - Processing batch size = 1")
        print("  - Decoding one latent frame at a time")
        print("  - Moving results to CPU immediately")
        print("  - Clearing GPU cache frequently")

        # Freeze models
        for param in self.lam.parameters():
            param.requires_grad = False
        for param in self.vvae.parameters():
            param.requires_grad = False

    def encode_actions(self, z_sequence_vvae: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode VVAE sequence to discrete action codes.

        Args:
            z_sequence_vvae: [B, T, C=16, H=64, W=64]

        Returns:
            indices: [B, T-1, 3] - discrete action codes
            z_q: [B, T-1, 3, codebook_dim] - quantized embeddings
        """
        with torch.no_grad():
            mu, logvar = self.lam.encode(z_sequence_vvae)
            z_e = self.lam.lam.reparameterize(mu, logvar)
            z_q, indices, _, _, _ = self.lam.lam.quantize(z_e)
        return indices, z_q

    def apply_action(self, z_past_vvae: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """
        Apply action codes to predict next frame.

        Args:
            z_past_vvae: [B, T_past, C=16, H=64, W=64]
            codes: [B, T_past, 3, codebook_dim]

        Returns:
            z_next_pred: [B, C=16, H=64, W=64]
        """
        with torch.no_grad():
            z_next_pred = self.lam.decode(z_past_vvae, codes)
        return z_next_pred

    def decode_latent_to_frames(self, z_vvae: torch.Tensor) -> torch.Tensor:
        """
        Decode VVAE latent to video frames with minimal VRAM usage.
        Processes one latent frame at a time.

        Args:
            z_vvae: [1, C=16, H=64, W=64] - single latent frame (batch=1)

        Returns:
            frames: [1, temporal_compression, C=3, H_full, W_full]
                   e.g., [1, 4, 3, 512, 512] for 4x compression
        """
        # IMPORTANT: Only batch size 1 for minimal VRAM
        assert z_vvae.shape[0] == 1, f"Expected batch size 1, got {z_vvae.shape[0]}"

        # Add temporal dimension: [1, C, H, W] -> [1, C, 1, H, W]
        z_vvae_with_time = z_vvae.unsqueeze(2)  # [1, 16, 1, 64, 64]

        with torch.no_grad():
            video = self.vvae.decode(z_vvae_with_time)  # [1, 3, temporal_compression, H, W]

        # Rearrange to [1, T, C, H, W]
        video = video.permute(0, 2, 1, 3, 4)  # [1, temporal_compression, 3, H, W]

        return video

    def experiment_0_full_video_rollout(
        self,
        max_rollout_steps: int = 30,
        num_videos: int = 1,
        output_dir: str = "vvae_lam_results"
    ):
        """
        Experiment 0: Full-length video rollout from single initial state.
        Creates videos with ground truth actions and random actions.

        Samples videos from all three splits to examine overfitting:
        - 20% from train (model has seen during training)
        - 10% from validation (model has seen during validation)
        - 70% from test (model has never seen)

        Args:
            max_rollout_steps: Maximum rollout steps (latent frames)
            num_videos: Number of video pairs to generate
            output_dir: Base output directory
        """
        print("\n" + "="*60)
        print("EXPERIMENT 0: Full-Length Video Rollout")
        print(f"Generating {num_videos} video pair(s)")
        print("Distribution: 20% train, 10% val, 70% test")
        print("="*60)

        # Create timestamped output directory
        output_dir = Path(output_dir) / f"exp0_full_rollout_{self.timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Log explanation to WandB
        if self.use_wandb:
            explanation = """
# Experiment 0: Full-Length Video Rollout

## What This Tests:
Generate **complete videos** from a single starting frame using autoregressive rollout.

## Split Distribution (to examine overfitting):
- **20% from TRAIN**: Data model was trained on (may be overfitted)
- **10% from VALIDATION**: Data used for validation during training
- **70% from TEST**: Data model has never seen (true generalization)

## Two Conditions:
1. **Ground Truth Actions**: Use actions from the original sequence
2. **Random Actions**: Use randomly sampled actions from other sequences

## How It Works:
- Start from single initial frame
- Autoregressively predict next frames using LAM
- Decode all predicted latents to create full video
- Compare ground truth action rollout vs random action rollout

## What to Look For:
- **Train split**: Should have lowest error (model has seen this data)
- **Val split**: Should have medium error
- **Test split**: Should have highest error (but hopefully still good)
- Compare coherence, smoothness, and semantic consistency across splits
"""
            wandb.log({"exp0_full_rollout/explanation": wandb.Html(f"<pre>{explanation}</pre>")})

        # Calculate split distribution (20% train, 10% val, 70% test)
        num_train = max(1, int(num_videos * 0.20))
        num_val = max(1, int(num_videos * 0.10))
        num_test = max(1, num_videos - num_train - num_val)  # Remainder goes to test

        print(f"\nSplit distribution:")
        print(f"  Train: {num_train} videos ({num_train/num_videos*100:.1f}%)")
        print(f"  Val:   {num_val} videos ({num_val/num_videos*100:.1f}%)")
        print(f"  Test:  {num_test} videos ({num_test/num_videos*100:.1f}%)")

        # Generate videos from each split
        video_idx = 0

        # Train split
        for i in range(num_train):
            print(f"\n{'='*60}")
            print(f"Generating Video {video_idx + 1}/{num_videos} [TRAIN SPLIT]")
            print(f"{'='*60}")
            self._generate_single_video_rollout(
                sample_idx=i,
                max_rollout_steps=max_rollout_steps,
                output_dir=output_dir,
                video_idx=video_idx,
                dataset=self.train_dataset,
                split_name='train'
            )
            video_idx += 1

        # Validation split
        for i in range(num_val):
            print(f"\n{'='*60}")
            print(f"Generating Video {video_idx + 1}/{num_videos} [VAL SPLIT]")
            print(f"{'='*60}")
            self._generate_single_video_rollout(
                sample_idx=i,
                max_rollout_steps=max_rollout_steps,
                output_dir=output_dir,
                video_idx=video_idx,
                dataset=self.val_dataset,
                split_name='val'
            )
            video_idx += 1

        # Test split
        for i in range(num_test):
            print(f"\n{'='*60}")
            print(f"Generating Video {video_idx + 1}/{num_videos} [TEST SPLIT]")
            print(f"{'='*60}")
            self._generate_single_video_rollout(
                sample_idx=i,
                max_rollout_steps=max_rollout_steps,
                output_dir=output_dir,
                video_idx=video_idx,
                dataset=self.test_dataset,
                split_name='test'
            )
            video_idx += 1

    def _generate_single_video_rollout(
        self,
        sample_idx: int,
        max_rollout_steps: int,
        output_dir: Path,
        video_idx: int,
        dataset=None,
        split_name: str = 'test'
    ):
        """Generate a single video rollout pair.

        Args:
            sample_idx: Index in the dataset
            max_rollout_steps: Maximum number of rollout steps
            output_dir: Output directory
            video_idx: Video index for naming
            dataset: Dataset to sample from (defaults to test_dataset)
            split_name: Name of the split ('train', 'val', or 'test')
        """
        # Use provided dataset or default to test
        if dataset is None:
            dataset = self.test_dataset
            split_name = 'test'

        # Get a sample with sufficient length
        sample = dataset[sample_idx]
        z_sequence = sample['sequence'].unsqueeze(0).to(self.device)  # [1, T, 16, 64, 64]
        T = z_sequence.shape[1]

        actual_rollout_steps = min(max_rollout_steps, T - 1)
        print(f"Using sample {sample_idx} from {split_name.upper()} split with {T} latent frames")
        print(f"Rollout steps: {actual_rollout_steps}")

        # Encode all ground truth actions from this sequence
        indices_gt, z_q_gt = self.encode_actions(z_sequence)  # [1, T-1, 3, codebook_dim]

        # Sample random actions from OTHER sequences (never from current sequence)
        # Sample from the same split to be fair
        print(f"Sampling random actions from other sequences in {split_name} split...")
        random_actions = []
        max_samples = min(len(dataset), 100)

        for _ in range(actual_rollout_steps):
            # Pick a random sequence that's NOT the current one
            while True:
                rand_sample_idx = np.random.randint(0, max_samples)
                if rand_sample_idx != sample_idx:
                    break

            rand_sample = dataset[rand_sample_idx]
            rand_z_seq = rand_sample['sequence'].unsqueeze(0).to(self.device)
            _, rand_z_q = self.encode_actions(rand_z_seq)
            # Take a random action from this sequence
            rand_t = np.random.randint(0, rand_z_q.shape[1])
            random_actions.append(rand_z_q[:, rand_t:rand_t+1])  # [1, 1, 3, codebook_dim]

        # Initial state
        z_init = z_sequence[:, :1]  # [1, 1, 16, 64, 64]

        print("\n1. Rollout with GROUND TRUTH actions...")
        z_rollout_gt = self._rollout_with_actions(z_init, z_q_gt, actual_rollout_steps)

        print("2. Rollout with RANDOM actions...")
        z_rollout_random = self._rollout_with_actions(z_init, random_actions, actual_rollout_steps)

        # Decode all latents to video frames
        print("\nDecoding to video frames...")

        # Ground truth frames
        all_gt_frames = []
        for t in range(actual_rollout_steps):
            frames = self.decode_latent_to_frames(z_rollout_gt[t:t+1])  # [1, temporal_compression, 3, H, W]
            all_gt_frames.append(frames[0].cpu())
            torch.cuda.empty_cache()
        gt_video_frames = torch.cat(all_gt_frames, dim=0)  # [T*temporal_compression, 3, H, W]

        # Random action frames
        all_random_frames = []
        for t in range(actual_rollout_steps):
            frames = self.decode_latent_to_frames(z_rollout_random[t:t+1])
            all_random_frames.append(frames[0].cpu())
            torch.cuda.empty_cache()
        random_video_frames = torch.cat(all_random_frames, dim=0)  # [T*temporal_compression, 3, H, W]

        # Original ground truth video for comparison
        all_orig_frames = []
        for t in range(1, actual_rollout_steps + 1):
            # Extract single latent frame: [1, 16, 64, 64]
            z_single = z_sequence[:, t]  # Remove temporal dimension
            frames = self.decode_latent_to_frames(z_single)
            all_orig_frames.append(frames[0].cpu())
            torch.cuda.empty_cache()
        orig_video_frames = torch.cat(all_orig_frames, dim=0)

        # Save videos with video index and split name
        fps = 8  # 8 fps for video playback
        save_video_from_frames(orig_video_frames, output_dir / f"video_{video_idx:03d}_{split_name}_original.mp4", fps=fps)
        save_video_from_frames(gt_video_frames, output_dir / f"video_{video_idx:03d}_{split_name}_rollout_gt_actions.mp4", fps=fps)
        save_video_from_frames(random_video_frames, output_dir / f"video_{video_idx:03d}_{split_name}_rollout_random_actions.mp4", fps=fps)

        # Compute metrics
        metrics_gt = compute_metrics(orig_video_frames, gt_video_frames)
        metrics_random = compute_metrics(orig_video_frames, random_video_frames)

        # Log to WandB with split information
        if self.use_wandb:
            wandb.log({
                f"exp0_full_rollout/{split_name}/video_{video_idx:03d}_original": wandb.Video(str(output_dir / f"video_{video_idx:03d}_{split_name}_original.mp4"), fps=fps, format="mp4"),
                f"exp0_full_rollout/{split_name}/video_{video_idx:03d}_gt_actions": wandb.Video(str(output_dir / f"video_{video_idx:03d}_{split_name}_rollout_gt_actions.mp4"), fps=fps, format="mp4"),
                f"exp0_full_rollout/{split_name}/video_{video_idx:03d}_random_actions": wandb.Video(str(output_dir / f"video_{video_idx:03d}_{split_name}_rollout_random_actions.mp4"), fps=fps, format="mp4"),
                f"exp0_full_rollout/{split_name}/video_{video_idx:03d}_gt_mse": metrics_gt['mse'],
                f"exp0_full_rollout/{split_name}/video_{video_idx:03d}_gt_psnr": metrics_gt['psnr'],
                f"exp0_full_rollout/{split_name}/video_{video_idx:03d}_random_mse": metrics_random['mse'],
                f"exp0_full_rollout/{split_name}/video_{video_idx:03d}_random_psnr": metrics_random['psnr'],
                # Also log aggregate metrics per split
                f"exp0_full_rollout/{split_name}/avg_gt_mse": metrics_gt['mse'],
                f"exp0_full_rollout/{split_name}/avg_gt_psnr": metrics_gt['psnr'],
                f"exp0_full_rollout/{split_name}/avg_random_mse": metrics_random['mse'],
                f"exp0_full_rollout/{split_name}/avg_random_psnr": metrics_random['psnr'],
            })

        print(f"\n✓ Video {video_idx} [{split_name.upper()}] saved to {output_dir}")
        print(f"  - Original: {actual_rollout_steps * self.temporal_compression} frames")
        print(f"  - GT Actions MSE: {metrics_gt['mse']:.4f}, PSNR: {metrics_gt['psnr']:.2f}dB")
        print(f"  - Random Actions MSE: {metrics_random['mse']:.4f}, PSNR: {metrics_random['psnr']:.2f}dB")

    def _rollout_with_actions(self, z_init: torch.Tensor, actions, num_steps: int) -> torch.Tensor:
        """
        Helper function to rollout predictions with given actions.

        Args:
            z_init: Initial latent frame [1, 1, 16, 64, 64]
            actions: Either [1, T, 3, codebook_dim] tensor or list of [1, 1, 3, codebook_dim] tensors
            num_steps: Number of rollout steps

        Returns:
            Predicted latents [num_steps, 16, 64, 64]
        """
        z_current = z_init
        codes_accumulated = []
        rollout_predictions = []

        for step in range(num_steps):
            # Get action for this step
            if isinstance(actions, list):
                action = actions[step]
            else:
                action = actions[:, step:step+1]

            codes_accumulated.append(action)
            codes_so_far = torch.cat(codes_accumulated, dim=1)

            # Predict next frame
            z_next_pred = self.apply_action(z_current, codes_so_far)
            rollout_predictions.append(z_next_pred)

            # Update context
            z_current = torch.cat([z_current, z_next_pred.unsqueeze(1)], dim=1)

        # Stack predictions: [num_steps, 16, 64, 64]
        return torch.stack(rollout_predictions, dim=0).squeeze(1)

    def log_frame_comparison(
        self,
        original_frames: torch.Tensor,
        predicted_frames: torch.Tensor,
        prefix: str,
        sample_idx: int,
        metrics: Dict[str, float] = None
    ):
        """
        Log frame-by-frame comparison to WandB.

        Args:
            original_frames: [N, 3, H, W] - N frames
            predicted_frames: [N, 3, H, W] - N frames
            prefix: Logging prefix (e.g., "exp1")
            sample_idx: Sample index
            metrics: Optional metrics dict
        """
        if not self.use_wandb:
            return

        N = original_frames.shape[0]

        # Create side-by-side comparisons for each frame
        wandb_images = []
        for i in range(N):
            orig_img = tensor_to_image(original_frames[i])
            pred_img = tensor_to_image(predicted_frames[i])

            # Create side-by-side image
            combined = np.concatenate([orig_img, pred_img], axis=1)

            caption = f"Frame {i+1}: Original (left) vs Predicted (right)"
            if metrics:
                caption += f" | MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, PSNR: {metrics['psnr']:.2f}dB"

            wandb_images.append(wandb.Image(combined, caption=caption))

        # Log to WandB
        log_dict = {f"{prefix}/sample_{sample_idx}_frames": wandb_images}
        if metrics:
            log_dict.update({f"{prefix}/sample_{sample_idx}_{k}": v for k, v in metrics.items()})
        wandb.log(log_dict)

    def experiment_1_ground_truth_reconstruction(
        self,
        num_samples: int = 5,
        output_dir: str = "vvae_lam_results"
    ):
        """
        Experiment 1: Use ground truth actions to reconstruct next frames.
        For each sample, reconstruct T-1 latent frames → (T-1) * temporal_compression video frames.
        """
        print("\n" + "="*60)
        print("EXPERIMENT 1: Ground Truth Action Reconstruction")
        print("="*60)

        # Log explanation to WandB
        if self.use_wandb:
            explanation = """
# Experiment 1: Ground Truth Action Reconstruction

## What This Tests:
Can the LAM accurately reconstruct frames when given **ground truth actions**?

## How to Read:
- **LEFT**: Original/ground truth frames from test dataset
- **RIGHT**: LAM-reconstructed frames using ground truth actions

## Process:
1. Extract ground truth actions from test sequence (LAM encoder)
2. Use true past latents + ground truth actions → predict next latent (LAM decoder)
3. Decode both original and predicted latents to video frames (VVAE decoder)

## What Good Results Look Like:
- **Low MSE/MAE**: Frames look similar (pixel-level accuracy)
- **High PSNR**: >25dB is good, >30dB is excellent
- LEFT and RIGHT should look **nearly identical**

## What This Tells You:
If reconstruction is good, the LAM has learned meaningful action representations that can accurately predict state transitions.
If reconstruction is poor, the action encoding/decoding is not working properly even with perfect actions.
"""
            wandb.log({"exp1_ground_truth/explanation": wandb.Html(f"<pre>{explanation}</pre>")})

        output_dir = Path(output_dir) / "exp1_ground_truth"
        output_dir.mkdir(parents=True, exist_ok=True)

        all_metrics = {'mse': [], 'mae': [], 'psnr': []}

        for i in tqdm(range(num_samples), desc="Exp 1"):
            # Get test sample
            sample = self.test_dataset[i]
            z_sequence = sample['sequence'].unsqueeze(0).to(self.device)  # [1, T, 16, 64, 64]
            T = z_sequence.shape[1]

            # Encode actions
            indices, z_q = self.encode_actions(z_sequence)

            # Reconstruct all frames using LAM
            reconstructed_latents = []
            for t in range(1, T):
                z_past = z_sequence[:, :t]  # [1, t, 16, 64, 64]
                codes_past = z_q[:, :t]  # [1, t, 3, codebook_dim]
                z_next_pred = self.apply_action(z_past, codes_past)  # [1, 16, 64, 64]
                reconstructed_latents.append(z_next_pred)

            # Stack: [T-1, 16, 64, 64]
            z_reconstructed = torch.stack(reconstructed_latents, dim=0).squeeze(1)
            z_original = z_sequence[0, 1:]  # [T-1, 16, 64, 64]

            # Decode each latent frame to video frames ONE AT A TIME for minimal VRAM
            # Each latent → temporal_compression frames
            all_original_frames = []
            all_predicted_frames = []

            for t in range(T - 1):
                # Decode original latent frame (one at a time)
                orig_frames = self.decode_latent_to_frames(z_original[t:t+1])  # [1, temporal_compression, 3, H, W]
                all_original_frames.append(orig_frames[0].cpu())  # Move to CPU immediately: [temporal_compression, 3, H, W]

                # Clear GPU cache
                torch.cuda.empty_cache()

                # Decode predicted latent frame (one at a time)
                pred_frames = self.decode_latent_to_frames(z_reconstructed[t:t+1])  # [1, temporal_compression, 3, H, W]
                all_predicted_frames.append(pred_frames[0].cpu())  # Move to CPU immediately

                # Clear GPU cache
                torch.cuda.empty_cache()

            # Concatenate all frames on CPU: [(T-1)*temporal_compression, 3, H, W]
            original_frames = torch.cat(all_original_frames, dim=0)
            predicted_frames = torch.cat(all_predicted_frames, dim=0)

            # Compute metrics
            metrics = compute_metrics(original_frames, predicted_frames)
            for k in all_metrics:
                all_metrics[k].append(metrics[k])

            # Log to WandB
            self.log_frame_comparison(
                original_frames,
                predicted_frames,
                prefix="exp1_ground_truth",
                sample_idx=i,
                metrics=metrics
            )

            print(f"  Sample {i+1}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, PSNR={metrics['psnr']:.2f}dB")

        # Log summary statistics
        if self.use_wandb:
            wandb.log({
                "exp1_ground_truth/avg_mse": np.mean(all_metrics['mse']),
                "exp1_ground_truth/avg_mae": np.mean(all_metrics['mae']),
                "exp1_ground_truth/avg_psnr": np.mean(all_metrics['psnr']),
            })

        print(f"\nExp 1 Summary:")
        print(f"  Avg MSE:  {np.mean(all_metrics['mse']):.4f} ± {np.std(all_metrics['mse']):.4f}")
        print(f"  Avg MAE:  {np.mean(all_metrics['mae']):.4f} ± {np.std(all_metrics['mae']):.4f}")
        print(f"  Avg PSNR: {np.mean(all_metrics['psnr']):.2f} ± {np.std(all_metrics['psnr']):.2f} dB")

    def experiment_2_shuffled_actions(
        self,
        num_samples: int = 5,
        output_dir: str = "vvae_lam_results"
    ):
        """
        Experiment 2: Shuffle action codes and apply them.
        Tests if actions are context-independent or produce coherent outputs.
        """
        print("\n" + "="*60)
        print("EXPERIMENT 2: Shuffled Action Application")
        print("="*60)

        # Log explanation to WandB
        if self.use_wandb:
            explanation = """
# Experiment 2: Shuffled Action Application

## What This Tests:
What happens when we apply **random/shuffled actions** instead of the correct sequence?

## How to Read:
- **LEFT**: Original frames (correct action sequence)
- **RIGHT**: Reconstructed frames using **shuffled actions** (wrong temporal order)

## Process:
1. Extract ground truth actions from test sequence
2. **Randomly shuffle** the action sequence
3. Apply shuffled actions to reconstruct frames
4. Compare with original

## What to Look For:
- **Similar to Exp 1**: Actions might be somewhat context-independent
- **Very different/incoherent**: Actions are highly context-dependent
- **Structured but wrong**: Actions encode meaningful motions but wrong timing

## What This Tells You:
- How sensitive the LAM is to action ordering
- Whether actions are "absolute" motions or "relative" to context
- If the model can still produce coherent frames with wrong action timing
"""
            wandb.log({"exp2_shuffled_actions/explanation": wandb.Html(f"<pre>{explanation}</pre>")})

        output_dir = Path(output_dir) / "exp2_shuffled_actions"
        output_dir.mkdir(parents=True, exist_ok=True)

        all_metrics = {'mse': [], 'mae': [], 'psnr': []}

        for i in tqdm(range(num_samples), desc="Exp 2"):
            sample = self.test_dataset[i]
            z_sequence = sample['sequence'].unsqueeze(0).to(self.device)
            T = z_sequence.shape[1]

            # Encode actions
            indices, z_q = self.encode_actions(z_sequence)

            # Shuffle action codes randomly
            shuffle_indices = torch.randperm(T - 1)
            z_q_shuffled = z_q[:, shuffle_indices]  # [1, T-1, 3, codebook_dim]

            # Apply shuffled actions
            reconstructed_latents = []
            for t in range(1, T):
                z_past = z_sequence[:, :t]
                codes_past = z_q_shuffled[:, :t]
                z_next_pred = self.apply_action(z_past, codes_past)
                reconstructed_latents.append(z_next_pred)

            z_reconstructed = torch.stack(reconstructed_latents, dim=0).squeeze(1)
            z_original = z_sequence[0, 1:]

            # Decode frames ONE AT A TIME for minimal VRAM
            all_original_frames = []
            all_predicted_frames = []

            for t in range(T - 1):
                # Decode one at a time, move to CPU immediately
                orig_frames = self.decode_latent_to_frames(z_original[t:t+1])
                all_original_frames.append(orig_frames[0].cpu())
                torch.cuda.empty_cache()

                pred_frames = self.decode_latent_to_frames(z_reconstructed[t:t+1])
                all_predicted_frames.append(pred_frames[0].cpu())
                torch.cuda.empty_cache()

            # Concatenate on CPU
            original_frames = torch.cat(all_original_frames, dim=0)
            predicted_frames = torch.cat(all_predicted_frames, dim=0)

            # Compute metrics (expect worse than exp1)
            metrics = compute_metrics(original_frames, predicted_frames)
            for k in all_metrics:
                all_metrics[k].append(metrics[k])

            # Log to WandB
            self.log_frame_comparison(
                original_frames,
                predicted_frames,
                prefix="exp2_shuffled_actions",
                sample_idx=i,
                metrics=metrics
            )

            print(f"  Sample {i+1}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, PSNR={metrics['psnr']:.2f}dB")

        # Log summary
        if self.use_wandb:
            wandb.log({
                "exp2_shuffled_actions/avg_mse": np.mean(all_metrics['mse']),
                "exp2_shuffled_actions/avg_mae": np.mean(all_metrics['mae']),
                "exp2_shuffled_actions/avg_psnr": np.mean(all_metrics['psnr']),
            })

        print(f"\nExp 2 Summary:")
        print(f"  Avg MSE:  {np.mean(all_metrics['mse']):.4f} ± {np.std(all_metrics['mse']):.4f}")
        print(f"  Avg MAE:  {np.mean(all_metrics['mae']):.4f} ± {np.std(all_metrics['mae']):.4f}")
        print(f"  Avg PSNR: {np.mean(all_metrics['psnr']):.2f} ± {np.std(all_metrics['psnr']):.2f} dB")

    def experiment_3_cross_context_actions(
        self,
        num_actions: int = 5,
        num_contexts: int = 5,
        output_dir: str = "vvae_lam_results"
    ):
        """
        Experiment 3: Apply each of N actions on M different contexts.
        Creates N×M grid showing how same action affects different states.
        """
        print("\n" + "="*60)
        print("EXPERIMENT 3: Cross-Context Action Transfer")
        print(f"Testing {num_actions} actions × {num_contexts} contexts")
        print("="*60)

        # Log explanation to WandB
        if self.use_wandb:
            explanation = """
# Experiment 3: Cross-Context Action Transfer

## What This Tests:
Does the **same action** produce **similar effects** on **different states**?

## How to Read the Grid:
- **Top Row**: Original context frames (different starting states)
- **Rows below**: Same action applied to each context
  - Row 1 = Action 1 applied to all 5 contexts
  - Row 2 = Action 2 applied to all 5 contexts
  - etc.

## What to Look For:

### Context-Independent Actions:
- Same action produces **similar visual changes** across different contexts
- E.g., "swipe right" moves everything rightward regardless of what's on screen
- Rows show consistent motion patterns

### Context-Dependent Actions:
- Same action produces **different results** depending on context
- E.g., action adapts to screen content
- Rows show varied transformations

## What This Tells You:
- Whether actions encode **absolute motions** (context-free) or **conditional motions** (context-aware)
- Action transferability across different states
- If the model has learned general-purpose action primitives
"""
            wandb.log({"exp3_cross_context/explanation": wandb.Html(f"<pre>{explanation}</pre>")})

        output_dir = Path(output_dir) / "exp3_cross_context"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sample N action sequences
        print("Extracting actions...")
        action_samples = []
        for i in range(num_actions):
            sample = self.test_dataset[i]
            z_seq = sample['sequence'].unsqueeze(0).to(self.device)
            _, z_q = self.encode_actions(z_seq)
            # Take first action (t=0->1 transition)
            action_samples.append(z_q[:, 0])  # [1, 3, codebook_dim]

        # Sample M context sequences
        print("Extracting contexts...")
        context_samples = []
        for i in range(num_actions, num_actions + num_contexts):
            sample = self.test_dataset[i]
            z_seq = sample['sequence'].unsqueeze(0).to(self.device)
            # Take first latent frame as context
            context_samples.append(z_seq[:, 0])  # [1, 16, 64, 64]

        # Apply each action on each context ONE AT A TIME for minimal VRAM
        print("Applying actions to contexts...")
        results_grid = []  # [num_actions, num_contexts, temporal_compression, 3, H, W]

        for action_idx, action in enumerate(tqdm(action_samples, desc="Actions")):
            action_row = []
            for context_idx, context in enumerate(context_samples):
                # Apply action to context
                z_past = context.unsqueeze(1)  # [1, 1, 16, 64, 64]
                codes = action.unsqueeze(1)  # [1, 1, 3, codebook_dim]
                z_next = self.apply_action(z_past, codes)  # [1, 16, 64, 64]

                # Decode to frames ONE AT A TIME
                frames = self.decode_latent_to_frames(z_next)  # [1, temporal_compression, 3, H, W]
                action_row.append(frames[0].cpu())  # Move to CPU: [temporal_compression, 3, H, W]

                # Clear GPU cache
                torch.cuda.empty_cache()

            results_grid.append(torch.stack(action_row))  # [num_contexts, temporal_compression, 3, H, W]

        results_grid = torch.stack(results_grid)  # [num_actions, num_contexts, temporal_compression, 3, H, W]

        # Also decode the original contexts ONE AT A TIME
        context_frames_list = []
        for context in context_samples:
            frames = self.decode_latent_to_frames(context)  # [1, temporal_compression, 3, H, W]
            context_frames_list.append(frames[0].cpu())  # Move to CPU
            torch.cuda.empty_cache()
        context_frames = torch.stack(context_frames_list)  # [num_contexts, temporal_compression, 3, H, W]

        # Log grid to WandB - one image per temporal frame
        if self.use_wandb:
            for t in range(self.temporal_compression):
                # Create grid for this temporal frame
                grid_images = []

                # First row: original contexts
                context_row = []
                for j in range(num_contexts):
                    frame = context_frames[j, t]  # [3, H, W]
                    context_row.append(tensor_to_image(frame))

                # Remaining rows: action applications
                for i in range(num_actions):
                    action_row_imgs = []
                    for j in range(num_contexts):
                        frame = results_grid[i, j, t]  # [3, H, W]
                        action_row_imgs.append(tensor_to_image(frame))
                    grid_images.append(action_row_imgs)

                # Create combined grid visualization
                fig, axes = plt.subplots(num_actions + 1, num_contexts, figsize=(num_contexts * 2, (num_actions + 1) * 2))
                if num_contexts == 1:
                    axes = axes.reshape(-1, 1)

                # Plot contexts
                for j in range(num_contexts):
                    axes[0, j].imshow(context_row[j])
                    axes[0, j].set_title(f"Context {j+1}", fontsize=10)
                    axes[0, j].axis('off')

                # Plot action results
                for i in range(num_actions):
                    for j in range(num_contexts):
                        axes[i+1, j].imshow(grid_images[i][j])
                        if j == 0:
                            axes[i+1, j].set_ylabel(f"Action {i+1}", fontsize=10)
                        axes[i+1, j].axis('off')

                plt.suptitle(f"Cross-Context Grid (Frame {t+1}/{self.temporal_compression})", fontsize=12)
                plt.tight_layout()

                # Save and log to WandB
                save_path = output_dir / f"grid_frame_{t+1}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')

                wandb.log({f"exp3_cross_context/grid_frame_{t+1}": wandb.Image(str(save_path))})
                plt.close()

        print(f"✓ Cross-context grid saved ({self.temporal_compression} frames)")

    def experiment_4_multistep_rollout(
        self,
        num_samples: int = 5,
        rollout_steps: int = 5,
        output_dir: str = "vvae_lam_results"
    ):
        """
        Experiment 4: Multi-step autoregressive rollout.
        For N different initial states, rollout K steps using ground truth actions.
        Tests compounding error and long-horizon prediction.

        Args:
            num_samples: Number of different initial states
            rollout_steps: Number of prediction steps (latent frames)
            output_dir: Output directory
        """
        print("\n" + "="*60)
        print("EXPERIMENT 4: Multi-Step Autoregressive Rollout")
        print(f"Testing {num_samples} samples × {rollout_steps} rollout steps")
        print("="*60)

        # Log explanation to WandB
        if self.use_wandb:
            explanation = """
# Experiment 4: Multi-Step Autoregressive Rollout

## What This Tests:
How well can the LAM **predict multiple steps into the future** using its own predictions?

## The Process (Autoregressive):
- Step 1: Start with real frame z[0] + action[0] → predict z[1]'
- Step 2: Use **predicted** z[1]' + action[1] → predict z[2]'
- Step 3: Use **predicted** z[1]', z[2]' + action[2] → predict z[3]'
- ... (errors compound over time)

## How to Read:
- **LEFT**: Ground truth frames
- **RIGHT**: LAM predictions (using its own past predictions)
- View results for **Step 1, Step 2, Step 3, Step 4, Step 5**

## What to Look For:

### Good Rollout:
- Errors increase **slowly** over steps
- Predictions stay coherent even at Step 5
- PSNR decreases gradually (e.g., 30→28→26→24→22 dB)

### Poor Rollout:
- Errors **compound rapidly** (divergence)
- Predictions become blurry or incoherent
- PSNR drops sharply (e.g., 30→25→15→10→5 dB)

## What This Tells You:
- Long-horizon prediction capability
- How quickly errors accumulate (important for planning)
- Model stability in autoregressive mode
- Whether the model can be used for multi-step planning

## Key Metrics:
Check the step-wise summary table showing how MSE/MAE/PSNR degrade over rollout steps.
"""
            wandb.log({"exp4_rollout/explanation": wandb.Html(f"<pre>{explanation}</pre>")})

        output_dir = Path(output_dir) / "exp4_multistep_rollout"
        output_dir.mkdir(parents=True, exist_ok=True)

        all_step_metrics = {step: {'mse': [], 'mae': [], 'psnr': []} for step in range(1, rollout_steps + 1)}

        for sample_idx in tqdm(range(num_samples), desc="Exp 4"):
            # Get test sample with sufficient length
            sample = self.test_dataset[sample_idx]
            z_sequence = sample['sequence'].unsqueeze(0).to(self.device)  # [1, T, 16, 64, 64]
            T = z_sequence.shape[1]

            if T < rollout_steps + 2:
                print(f"  Sample {sample_idx} too short ({T} frames), skipping")
                continue

            # Encode all ground truth actions
            indices, z_q = self.encode_actions(z_sequence)  # [1, T-1, 3, codebook_dim]

            # Initial context: first frame
            z_init = z_sequence[:, :1]  # [1, 1, 16, 64, 64]

            # Rollout K steps autoregressively
            z_current = z_init
            codes_accumulated = []  # Accumulate all actions used
            rollout_predictions = []

            for step in range(rollout_steps):
                # Get ground truth action for this step
                action = z_q[:, step:step+1]  # [1, 1, 3, codebook_dim]
                codes_accumulated.append(action)

                # Stack all actions so far: [1, step+1, 3, codebook_dim]
                codes_so_far = torch.cat(codes_accumulated, dim=1)

                # Predict next latent frame
                # z_current has step+1 frames, codes_so_far has step+1 actions
                z_next_pred = self.apply_action(z_current, codes_so_far)  # [1, 16, 64, 64]
                rollout_predictions.append(z_next_pred)

                # Update context: append prediction
                z_current = torch.cat([z_current, z_next_pred.unsqueeze(1)], dim=1)  # [1, step+2, 16, 64, 64]

            # Stack predictions: [rollout_steps, 16, 64, 64]
            z_predicted_rollout = torch.stack(rollout_predictions, dim=0).squeeze(1)

            # Ground truth: frames 1 to rollout_steps
            z_ground_truth = z_sequence[0, 1:rollout_steps+1]  # [rollout_steps, 16, 64, 64]

            # Decode and visualize step-by-step, ONE AT A TIME for minimal VRAM
            for step in range(rollout_steps):
                # Decode this step (one at a time)
                gt_frames = self.decode_latent_to_frames(z_ground_truth[step:step+1])  # [1, temporal_compression, 3, H, W]
                gt_frames = gt_frames[0].cpu()  # Move to CPU: [temporal_compression, 3, H, W]
                torch.cuda.empty_cache()

                pred_frames = self.decode_latent_to_frames(z_predicted_rollout[step:step+1])  # [1, temporal_compression, 3, H, W]
                pred_frames = pred_frames[0].cpu()  # Move to CPU
                torch.cuda.empty_cache()

                # Compute metrics for this step
                step_metrics = compute_metrics(gt_frames, pred_frames)
                for k in all_step_metrics[step + 1]:
                    all_step_metrics[step + 1][k].append(step_metrics[k])

                # Log to WandB
                self.log_frame_comparison(
                    gt_frames,
                    pred_frames,
                    prefix=f"exp4_rollout/sample_{sample_idx}/step_{step+1}",
                    sample_idx=sample_idx,
                    metrics=step_metrics
                )

            print(f"  Sample {sample_idx}: Rollout complete ({rollout_steps} steps)")

        # Log step-wise summary statistics
        if self.use_wandb:
            for step in range(1, rollout_steps + 1):
                if all_step_metrics[step]['mse']:  # Check if we have data
                    wandb.log({
                        f"exp4_rollout/step_{step}/avg_mse": np.mean(all_step_metrics[step]['mse']),
                        f"exp4_rollout/step_{step}/avg_mae": np.mean(all_step_metrics[step]['mae']),
                        f"exp4_rollout/step_{step}/avg_psnr": np.mean(all_step_metrics[step]['psnr']),
                    })

        # Print summary
        print(f"\nExp 4 Summary (Error vs Rollout Step):")
        print(f"{'Step':<6} {'MSE':<12} {'MAE':<12} {'PSNR (dB)':<12}")
        print("-" * 48)
        for step in range(1, rollout_steps + 1):
            if all_step_metrics[step]['mse']:
                mse_mean = np.mean(all_step_metrics[step]['mse'])
                mae_mean = np.mean(all_step_metrics[step]['mae'])
                psnr_mean = np.mean(all_step_metrics[step]['psnr'])
                print(f"{step:<6} {mse_mean:<12.4f} {mae_mean:<12.4f} {psnr_mean:<12.2f}")

    def run_all_experiments(
        self,
        num_samples: int = 5,
        num_actions: int = 5,
        num_contexts: int = 5,
        rollout_steps: int = 5,
        max_video_rollout_steps: int = 30,
        num_videos: int = 1,
        video_only: bool = False,
        output_dir: str = "vvae_lam_results"
    ):
        """Run all experiments including full video rollout."""

        # Experiment 0: Full video rollout (runs first)
        self.experiment_0_full_video_rollout(max_video_rollout_steps, num_videos, output_dir)

        # If video_only mode, skip other experiments
        if video_only:
            print("\n" + "="*60)
            print("VIDEO ONLY MODE - Skipping other experiments")
            print(f"Results saved to: {output_dir}")
            if self.use_wandb:
                print(f"WandB dashboard: {wandb.run.url}")
            print("="*60)
            if self.use_wandb:
                wandb.finish()
            return

        # Main experiments
        self.experiment_1_ground_truth_reconstruction(num_samples, output_dir)
        self.experiment_2_shuffled_actions(num_samples, output_dir)
        self.experiment_3_cross_context_actions(num_actions, num_contexts, output_dir)
        self.experiment_4_multistep_rollout(num_samples, rollout_steps, output_dir)

        print("\n" + "="*60)
        print("ALL EXPERIMENTS COMPLETE!")
        print(f"Results saved to: {output_dir}")
        if self.use_wandb:
            print(f"WandB dashboard: {wandb.run.url}")
        print("="*60)

        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="VVAE LAM Prediction and Visualization")
    parser.add_argument("--lam_checkpoint", type=str, required=True, help="Path to LAM checkpoint (.pt)")
    parser.add_argument("--lam_config", type=str, required=True, help="Path to LAM model_config.json")
    parser.add_argument("--vvae_config", type=str, default="vvae/configs/config_16z.yaml", help="VVAE config (use 16z for LAM)")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Test data directory (h5 files)")
    parser.add_argument("--manifest_path", type=str, required=True, help="Manifest JSON path")

    # Experiment parameters
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples for exp 1, 2, 4")
    parser.add_argument("--num_actions", type=int, default=5, help="Number of actions for exp 3")
    parser.add_argument("--num_contexts", type=int, default=5, help="Number of contexts for exp 3")
    parser.add_argument("--rollout_steps", type=int, default=5, help="Rollout steps for exp 4")
    parser.add_argument("--max_video_rollout_steps", type=int, default=30, help="Max rollout steps for exp 0 video")
    parser.add_argument("--num_videos", type=int, default=1, help="Number of video pairs to generate in exp 0")
    parser.add_argument("--video_only", action="store_true", help="Only run video generation (exp 0), skip other experiments")

    parser.add_argument("--output_dir", type=str, default="vvae_lam_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    # WandB parameters
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="vvae-lam-prediction", help="WandB project")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")

    args = parser.parse_args()

    predictor = VVAELAMPredictor(
        lam_checkpoint=args.lam_checkpoint,
        lam_config=args.lam_config,
        vvae_config=args.vvae_config,
        test_data_dir=args.test_data_dir,
        manifest_path=args.manifest_path,
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )

    predictor.run_all_experiments(
        num_samples=args.num_samples,
        num_actions=args.num_actions,
        num_contexts=args.num_contexts,
        rollout_steps=args.rollout_steps,
        max_video_rollout_steps=args.max_video_rollout_steps,
        num_videos=args.num_videos,
        video_only=args.video_only,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
