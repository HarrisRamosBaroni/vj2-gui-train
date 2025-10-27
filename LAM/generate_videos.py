"""
Video Generation and Reconstruction Script

This script loads a trained World Model checkpoint and generates video reconstructions
by:
1. Loading trajectories from the VVAE latent dataset
2. Performing forward passes through the World Model to predict next frames
3. Decoding the predicted latent frames using the VVAE decoder
4. Saving the reconstructed videos and computing quantitative metrics
"""

import argparse
import logging
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf
import torchvision.io

from LAM.training import load_model
from LAM.world_model import WorldModel
from LAM.dataloader_vvae import create_vvae_dataloaders

# Add VVAE directory to Python path so that config references like 'src.models.*' can be resolved
# This is necessary because VVAE configs expect to be run from within the vvae/ directory
VVAE_DIR = Path(__file__).parent.parent / "vvae"
if str(VVAE_DIR) not in sys.path:
    sys.path.insert(0, str(VVAE_DIR))

from vvae.utils.common_utils import instantiate_from_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_vvae_model(config_path: str, device: str):
    """
    Loads the VVAE model from its OmegaConf config file.
    Adapted from vvae/example.py.

    Args:
        config_path: Path to the VVAE config yaml file.
        device: Device to load the model on.

    Returns:
        The loaded and frozen VVAE model in evaluation mode.
    """
    logger.info(f"Loading VVAE model from config: {config_path}")
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    logger.info("VVAE model loaded successfully and frozen.")
    return model


def save_video_tensor(video_tensor: torch.Tensor, path: Path, fps: int = 10):
    """
    Saves a single video tensor to a .mp4 file.

    Args:
        video_tensor: Video tensor of shape [C, T, H, W] with values in [-1, 1].
        path: The output file path.
        fps: Frames per second for the output video.
    """
    # Denormalize from [-1, 1] to [0, 255] and convert to uint8
    video_tensor = ((video_tensor.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)

    # Permute from [C, T, H, W] to the expected [T, H, W, C] for write_video
    video_tensor = video_tensor.permute(1, 2, 3, 0)

    # Write the video file
    torchvision.io.write_video(str(path), video_tensor.cpu(), fps=fps)


def decode_sequence_memory_efficiently(vvae_model, latent_sequence: torch.Tensor) -> torch.Tensor:
    """
    Decodes a sequence of VVAE latents frame by frame to conserve memory.

    Args:
        vvae_model: The loaded VVAE model.
        latent_sequence: Tensor of shape [1, C, T, H, W].

    Returns:
        The decoded video tensor of shape [1, C_out, T_full, H_out, W_out].
    """
    assert latent_sequence.shape[0] == 1, "Only batch size 1 is supported for memory-efficient decoding."
    _, _, T, _, _ = latent_sequence.shape

    video_chunks = []
    for t in range(T):
        # Process one latent time step at a time.
        # Shape of latent_frame: [1, C, 1, H, W]
        latent_frame = latent_sequence[:, :, t:t+1, :, :]
        with torch.no_grad():
            # The VVAE decoder expects a temporal dimension.
            # Output shape: [1, C_out, temporal_compression, H_out, W_out]
            video_chunk = vvae_model.decode(latent_frame)
        video_chunks.append(video_chunk)

    # Concatenate along the temporal dimension (dim=2) to form the full video.
    full_video = torch.cat(video_chunks, dim=2)
    return full_video


def autoregressive_rollout(
    model: WorldModel,
    h_sequence: torch.Tensor,
) -> tuple:
    """
    Perform autoregressive rollout with real actions and world embedding.

    Uses the built-in autoregressive_rollout method from DynamicsPredictor.
    - Extract real actions from ground truth sequence (from action encoder)
    - Extract real world embedding from ground truth sequence (from world encoder)
    - Start with first frame as context
    - Call built-in autoregressive rollout

    Args:
        model: The WorldModel instance
        h_sequence: Ground truth sequence [B, T, C, H, W]

    Returns:
        pred_sequence: Predicted sequence [B, T, C, H, W] where predictions are rolled out autoregressively
        per_step_mse: List of MSE values for each prediction step [T-1]
    """
    B, T, C, H, W = h_sequence.shape
    device = h_sequence.device

    # =========================================================================
    # Step 1: Extract real actions and world embedding from GT sequence
    # =========================================================================
    # Tokenize full ground truth sequence
    tokens_gt, _, _ = model.tokenizer(h_sequence)  # [B*T, d_model, out_h, out_w]

    # Extract real actions (what actions are present in the GT sequence)
    action_codes, _, _, _ = model.action_encoder(tokens_gt, B, T - 1)  # [B, T-1, d_code_a]

    # Extract real world embedding (world hypothesis from GT sequence)
    world_emb, _, _, _ = model.world_encoder(tokens_gt, B, T)  # [B, d_code_h]

    # =========================================================================
    # Step 2: Use built-in autoregressive rollout
    # =========================================================================
    # Start with first frame as context
    context = h_sequence[:, 0:1, :, :, :]  # [B, 1, C, H, W]

    # Rollout for T-1 steps (to generate frames 1 to T-1)
    predictions = model.dynamics_predictor.autoregressive_rollout(
        context=context,
        action_sequence=action_codes,
        world_emb=world_emb,
        tokenizer=model.tokenizer,
        detokenizer=model.detokenizer
    )  # [B, T-1, C, H, W]

    # =========================================================================
    # Step 3: Combine initial frame with predictions
    # =========================================================================
    # pred_sequence: [B, T, C, H, W] where frame 0 is GT, frames 1:T are predictions
    pred_sequence = torch.cat([context, predictions], dim=1)  # [B, T, C, H, W]

    # =========================================================================
    # Step 4: Compute per-step MSE
    # =========================================================================
    per_step_mse = []
    for step in range(1, T):
        step_mse = F.mse_loss(pred_sequence[:, step], h_sequence[:, step])
        per_step_mse.append(step_mse.item())

    return pred_sequence, per_step_mse


def autoregressive_rollout_shared_action(
    model: WorldModel,
    h_sequence: torch.Tensor,
) -> tuple:
    """
    Perform autoregressive rollout with SHARED actions across the batch.

    Pick one random action sequence from the batch and replicate it for all sequences.
    This tests: "How does the same action affect different starting contexts?"

    Uses the built-in autoregressive_rollout method from DynamicsPredictor.

    Args:
        model: The WorldModel instance
        h_sequence: Ground truth sequence [B, T, C, H, W]

    Returns:
        pred_sequence: Predicted sequence [B, T, C, H, W] where all sequences use the same actions
        per_step_mse: List of MSE values for each prediction step [T-1]
        shared_action_idx: The batch index whose action was used (for logging)
    """
    B, T, C, H, W = h_sequence.shape
    device = h_sequence.device

    # =========================================================================
    # Step 1: Extract real actions and world embedding from GT sequence
    # =========================================================================
    # Tokenize full ground truth sequence
    tokens_gt, _, _ = model.tokenizer(h_sequence)  # [B*T, d_model, out_h, out_w]

    # Extract real actions (what actions are present in the GT sequence)
    action_codes, _, _, _ = model.action_encoder(tokens_gt, B, T - 1)  # [B, T-1, d_code_a]

    # Extract real world embedding (world hypothesis from GT sequence)
    world_emb, _, _, _ = model.world_encoder(tokens_gt, B, T)  # [B, d_code_h]

    # =========================================================================
    # Step 2: Pick one random action sequence and replicate for entire batch
    # =========================================================================
    shared_action_idx = torch.randint(0, B, (1,)).item()  # Random index in batch
    shared_action = action_codes[shared_action_idx:shared_action_idx+1, :, :]  # [1, T-1, d_code_a]
    action_codes_shared = shared_action.repeat(B, 1, 1)  # [B, T-1, d_code_a] - all same

    # =========================================================================
    # Step 3: Use built-in autoregressive rollout with shared actions
    # =========================================================================
    # Start with first frame as context
    context = h_sequence[:, 0:1, :, :, :]  # [B, 1, C, H, W]

    # Rollout for T-1 steps with shared actions
    predictions = model.dynamics_predictor.autoregressive_rollout(
        context=context,
        action_sequence=action_codes_shared,  # Using shared actions!
        world_emb=world_emb,
        tokenizer=model.tokenizer,
        detokenizer=model.detokenizer
    )  # [B, T-1, C, H, W]

    # =========================================================================
    # Step 4: Combine initial frame with predictions
    # =========================================================================
    pred_sequence = torch.cat([context, predictions], dim=1)  # [B, T, C, H, W]

    # =========================================================================
    # Step 5: Compute per-step MSE
    # =========================================================================
    per_step_mse = []
    for step in range(1, T):
        step_mse = F.mse_loss(pred_sequence[:, step], h_sequence[:, step])
        per_step_mse.append(step_mse.item())

    return pred_sequence, per_step_mse, shared_action_idx


def autoregressive_rollout_random_action(
    model: WorldModel,
    h_sequence: torch.Tensor,
) -> tuple:
    """
    Perform autoregressive rollout with RANDOM actions sampled from codebook.

    Sample random action codes from the action encoder's RVQ codebook.
    This tests: "What happens with completely random (but valid) actions?"

    Uses the built-in autoregressive_rollout method from DynamicsPredictor.

    Args:
        model: The WorldModel instance
        h_sequence: Ground truth sequence [B, T, C, H, W]

    Returns:
        pred_sequence: Predicted sequence [B, T, C, H, W] where all sequences use random actions
        per_step_mse: List of MSE values for each prediction step [T-1]
    """
    B, T, C, H, W = h_sequence.shape
    device = h_sequence.device

    # =========================================================================
    # Step 1: Tokenize and extract world embedding from GT sequence
    # =========================================================================
    # Tokenize full ground truth sequence
    tokens_gt, _, _ = model.tokenizer(h_sequence)  # [B*T, d_model, out_h, out_w]

    # Extract real world embedding (world hypothesis from GT sequence)
    world_emb, _, _, _ = model.world_encoder(tokens_gt, B, T)  # [B, d_code_h]

    # =========================================================================
    # Step 2: Sample random actions from RVQ codebook
    # =========================================================================
    # Get RVQ parameters
    num_levels = model.action_encoder.rvq.num_levels
    d_code_a = model.action_encoder.rvq.d_code

    # Initialize random action codes: [B, T-1, d_code_a]
    action_codes_random = torch.zeros(B, T - 1, d_code_a, device=device)

    # Sample from each RVQ level and sum (residual quantization)
    for level_idx in range(num_levels):
        # Get codebook for this level: [codebook_size, d_code_a]
        codebook = model.action_encoder.rvq._get_codebook(level_idx)
        codebook_size = model.action_encoder.rvq.codebook_sizes[level_idx]

        # Sample random indices: [B, T-1]
        random_idx = torch.randint(0, codebook_size, (B, T - 1), device=device)

        # Look up embeddings: [B, T-1, d_code_a]
        level_codes = F.embedding(random_idx, codebook)

        # Add to total (RVQ sums across levels)
        action_codes_random += level_codes

    # =========================================================================
    # Step 3: Use built-in autoregressive rollout with random actions
    # =========================================================================
    # Start with first frame as context
    context = h_sequence[:, 0:1, :, :, :]  # [B, 1, C, H, W]

    # Rollout for T-1 steps with random actions
    predictions = model.dynamics_predictor.autoregressive_rollout(
        context=context,
        action_sequence=action_codes_random,  # Using random actions!
        world_emb=world_emb,
        tokenizer=model.tokenizer,
        detokenizer=model.detokenizer
    )  # [B, T-1, C, H, W]

    # =========================================================================
    # Step 4: Combine initial frame with predictions
    # =========================================================================
    pred_sequence = torch.cat([context, predictions], dim=1)  # [B, T, C, H, W]

    # =========================================================================
    # Step 5: Compute per-step MSE
    # =========================================================================
    per_step_mse = []
    for step in range(1, T):
        step_mse = F.mse_loss(pred_sequence[:, step], h_sequence[:, step])
        per_step_mse.append(step_mse.item())

    return pred_sequence, per_step_mse


def compute_metrics(
    model: WorldModel,
    h_sequence: torch.Tensor,
    pred_frames_tf: torch.Tensor,
    pred_frames_rollout: torch.Tensor,
    per_step_mse_rollout: list,
    rvq_losses: dict,
) -> dict:
    """
    Compute quantitative metrics for both teacher forcing and rollout predictions.

    Args:
        model: The WorldModel instance
        h_sequence: Ground truth latent sequence [B, T, C, H, W]
        pred_frames_tf: Teacher forcing predicted frames [B, T-1, C, H, W]
        pred_frames_rollout: Rollout predicted frames [B, T, C, H, W] (includes GT frame 0)
        per_step_mse_rollout: List of per-step MSE values from rollout
        rvq_losses: Dictionary of RVQ losses from the teacher forcing pass

    Returns:
        Dictionary of computed metrics
    """
    # Teacher forcing reconstruction MSE (per description.md: GT[0:T-1] in → PRED[1:T] out)
    # Direct comparison, no slicing needed!
    recon_mse_tf = F.mse_loss(pred_frames_tf, h_sequence[:, 1:]).item()

    # Rollout reconstruction MSE (overall)
    # rollout includes GT frame 0, so compare frames [1:T] with GT[1:T]
    recon_mse_rollout = F.mse_loss(pred_frames_rollout[:, 1:], h_sequence[:, 1:]).item()

    # Total loss (matching training objective) - using TF predictions
    total_loss = (
        recon_mse_tf +
        rvq_losses['action_commit_loss'].item() +
        rvq_losses['action_codebook_loss'].item() +
        rvq_losses['world_commit_loss'].item() +
        rvq_losses['world_codebook_loss'].item()
    )

    metrics = {
        # Overall losses
        'total_loss': total_loss,
        'recon_mse_tf': recon_mse_tf,
        'recon_mse_rollout': recon_mse_rollout,
        # RVQ losses (from TF pass)
        'action_commit_loss': rvq_losses['action_commit_loss'].item(),
        'action_codebook_loss': rvq_losses['action_codebook_loss'].item(),
        'world_commit_loss': rvq_losses['world_commit_loss'].item(),
        'world_codebook_loss': rvq_losses['world_codebook_loss'].item(),
        # Per-step rollout MSE
        'per_step_mse_rollout': per_step_mse_rollout,
    }

    return metrics


def generate_videos(
    checkpoint_path: str,
    vvae_config_path: str,
    data_dir: str,
    manifest_path: str,
    split: str = 'val',
    num_videos: int = 5,
    batch_size: int = 8,
    sequence_length: int = 8,
    output_dir: str = './output/video_reconstructions',
    device: str = 'cuda',
    num_workers: int = 4,
    fps: int = 10,
):
    """
    Generate video reconstructions from the trained World Model.

    For each trajectory, generates 3 videos:
    - Ground truth (gt)
    - Teacher forcing prediction (tf)
    - Autoregressive rollout prediction (rollout)

    Args:
        checkpoint_path: Path to the trained World Model checkpoint (.pt file)
        vvae_config_path: Path to the VVAE config file
        data_dir: Path to directory containing VVAE HDF5 files
        manifest_path: Path to manifest JSON file
        split: Which split to use ('train', 'val', or 'test')
        num_videos: Number of video triplets to generate and save
        batch_size: Batch size for metric computation
        sequence_length: Number of frames in sequence
        output_dir: Directory to save generated videos
        device: Device to use ('cuda' or 'cpu')
        num_workers: Number of worker processes for data loading
        fps: Frames per second for output videos
    """
    logger.info("=" * 80)
    logger.info("VIDEO GENERATION FROM WORLD MODEL")
    logger.info("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    # =========================================================================
    # Step 1: Load World Model from checkpoint
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOADING WORLD MODEL")
    logger.info("=" * 80)

    world_model, optimizer, checkpoint_info = load_model(
        checkpoint_path=checkpoint_path,
        device=device
    )
    world_model.eval()

    logger.info(f"Loaded checkpoint info:")
    logger.info(f"  Epoch: {checkpoint_info['epoch']}")
    logger.info(f"  Global step: {checkpoint_info['global_step']}")
    logger.info(f"  Best val recon MSE: {checkpoint_info['best_val_recon_mse']:.6f}")
    logger.info(f"  Best val total loss: {checkpoint_info['best_val_total_loss']:.6f}")

    # =========================================================================
    # Step 2: Load VVAE model
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: LOADING VVAE DECODER")
    logger.info("=" * 80)

    vvae_model = load_vvae_model(vvae_config_path, device)

    # =========================================================================
    # Step 3: Create dataloaders
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: LOADING DATA")
    logger.info("=" * 80)
    logger.info(f"Using split: {split}")

    train_loader, val_loader, test_loader = create_vvae_dataloaders(
        data_dir=data_dir,
        manifest_path=manifest_path,
        batch_size=batch_size,
        sequence_length=sequence_length,
        stride_train=1,
        num_workers=num_workers,
        ddp=False
    )

    # Select the appropriate dataloader
    if split == 'train':
        dataloader = train_loader
    elif split == 'val':
        dataloader = val_loader
    elif split == 'test':
        dataloader = test_loader
        if test_loader is None:
            logger.error("Test split requested but test_loader is None. Using validation set instead.")
            dataloader = val_loader
            split = 'val'
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

    logger.info(f"Selected {split} set with {len(dataloader)} batches")

    # =========================================================================
    # Step 4: Generate video quintuplets (GT, TF, Rollout, Shared-Action, Random-Action)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 4: GENERATING {num_videos} VIDEO QUINTUPLETS")
    logger.info("=" * 80)
    logger.info("Each quintuplet consists of: Ground Truth, Teacher Forcing, Autoregressive Rollout, Shared-Action Rollout, Random-Action Rollout")

    video_count = 0
    video_metrics = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if video_count >= num_videos:
                break

            h_sequence = batch['sequence'].to(device)  # [B, T, C=16, H=64, W=64]
            B = h_sequence.shape[0]

            # Teacher forcing pass
            output_tf = world_model(h_sequence)
            pred_frames_tf = output_tf['pred_frames']  # [B, T, 16, 64, 64]
            rvq_losses = output_tf['losses']

            # Autoregressive rollout pass (for entire batch)
            pred_frames_rollout, _ = autoregressive_rollout(world_model, h_sequence)

            # ============ ADD SHARED ACTION ROLLOUT HERE ============
            # Shared-action rollout: All sequences use the same action
            pred_frames_shared, _, shared_action_idx = autoregressive_rollout_shared_action(world_model, h_sequence)
            logger.info(f"  Batch {batch_idx}: Using actions from sequence {shared_action_idx} for all sequences in batch")

            # ============ ADD RANDOM ACTION ROLLOUT HERE ============
            # Random-action rollout: All sequences use random actions sampled from codebook
            pred_frames_random, _ = autoregressive_rollout_random_action(world_model, h_sequence)
            logger.info(f"  Batch {batch_idx}: Using random actions sampled from codebook")

            # Process each sequence in the batch
            for i in range(B):
                if video_count >= num_videos:
                    break

                # Get ground truth and predictions for this sequence
                gt_latent = h_sequence[i:i+1]  # [1, T, 16, 64, 64]
                pred_latent_tf = pred_frames_tf[i:i+1]  # [1, T-1, 16, 64, 64] (predictions for frames 1 to T)
                pred_latent_rollout = pred_frames_rollout[i:i+1]  # [1, T, 16, 64, 64] (includes GT frame 0)
                pred_latent_shared = pred_frames_shared[i:i+1]  # [1, T, 16, 64, 64]
                pred_latent_random = pred_frames_random[i:i+1]  # [1, T, 16, 64, 64]

                # Compute per-sequence rollout for metrics
                _, per_step_mse_rollout_seq = autoregressive_rollout(world_model, gt_latent)

                # Compute metrics for this sequence
                seq_metrics = compute_metrics(
                    model=world_model,
                    h_sequence=gt_latent,
                    pred_frames_tf=pred_latent_tf,
                    pred_frames_rollout=pred_latent_rollout,
                    per_step_mse_rollout=per_step_mse_rollout_seq,
                    rvq_losses={k: v[i:i+1] if v.dim() > 0 else v for k, v in rvq_losses.items()}
                )

                logger.info(f"\nVideo Quintuplet {video_count + 1}/{num_videos}:")
                logger.info(f"  Batch {batch_idx}, Sequence {i}")
                logger.info(f"  Shared-action rollout uses actions from sequence {shared_action_idx}")
                logger.info(f"  Random-action rollout uses randomly sampled actions from codebook")
                logger.info(f"  Metrics:")
                logger.info(f"    Total Loss: {seq_metrics['total_loss']:.6f}")
                logger.info(f"    Recon MSE (TF): {seq_metrics['recon_mse_tf']:.6f}")
                logger.info(f"    Recon MSE (Rollout): {seq_metrics['recon_mse_rollout']:.6f}")
                logger.info(f"    Action Commit: {seq_metrics['action_commit_loss']:.6f}")
                logger.info(f"    Action Codebook: {seq_metrics['action_codebook_loss']:.6f}")
                logger.info(f"    World Commit: {seq_metrics['world_commit_loss']:.6f}")
                logger.info(f"    World Codebook: {seq_metrics['world_codebook_loss']:.6f}")

                # Log per-step rollout MSE
                per_step_str = ", ".join([f"{mse:.6f}" for mse in seq_metrics['per_step_mse_rollout']])
                logger.info(f"    Rollout per-step MSE: [{per_step_str}]")

                video_metrics.append(seq_metrics)

                # Reshape latents for VVAE decoder
                # From [1, T, C, H, W] to [1, C, T, H, W]
                gt_latent_vvae = gt_latent.permute(0, 2, 1, 3, 4)
                pred_latent_tf_vvae = pred_latent_tf.permute(0, 2, 1, 3, 4)
                pred_latent_rollout_vvae = pred_latent_rollout.permute(0, 2, 1, 3, 4)
                pred_latent_shared_vvae = pred_latent_shared.permute(0, 2, 1, 3, 4)
                pred_latent_random_vvae = pred_latent_random.permute(0, 2, 1, 3, 4)

                # Decode ground truth
                logger.info(f"  Decoding ground truth video...")
                gt_video = decode_sequence_memory_efficiently(vvae_model, gt_latent_vvae)
                # gt_video shape: [1, C_out, T_full, H_out, W_out]

                # Decode teacher forcing prediction
                logger.info(f"  Decoding teacher forcing video...")
                pred_video_tf = decode_sequence_memory_efficiently(vvae_model, pred_latent_tf_vvae)

                # Decode rollout prediction
                logger.info(f"  Decoding rollout video...")
                pred_video_rollout = decode_sequence_memory_efficiently(vvae_model, pred_latent_rollout_vvae)

                # ============ ADD SHARED ACTION DECODING HERE ============
                # Decode shared-action rollout prediction
                logger.info(f"  Decoding shared-action rollout video...")
                pred_video_shared = decode_sequence_memory_efficiently(vvae_model, pred_latent_shared_vvae)

                # ============ ADD RANDOM ACTION DECODING HERE ============
                # Decode random-action rollout prediction
                logger.info(f"  Decoding random-action rollout video...")
                pred_video_random = decode_sequence_memory_efficiently(vvae_model, pred_latent_random_vvae)

                # Remove batch dimension: [C, T, H, W]
                gt_video = gt_video[0]
                pred_video_tf = pred_video_tf[0]
                pred_video_rollout = pred_video_rollout[0]
                pred_video_shared = pred_video_shared[0]
                pred_video_random = pred_video_random[0]

                # Save videos
                gt_path = output_path / f"video_{video_count:04d}_gt.mp4"
                tf_path = output_path / f"video_{video_count:04d}_pred_tf.mp4"
                rollout_path = output_path / f"video_{video_count:04d}_pred_rollout.mp4"
                shared_path = output_path / f"video_{video_count:04d}_pred_shared_action_rollout.mp4"
                random_path = output_path / f"video_{video_count:04d}_pred_random_action_rollout.mp4"

                logger.info(f"  Saving ground truth to: {gt_path}")
                save_video_tensor(gt_video, gt_path, fps=fps)

                logger.info(f"  Saving teacher forcing to: {tf_path}")
                save_video_tensor(pred_video_tf, tf_path, fps=fps)

                logger.info(f"  Saving rollout to: {rollout_path}")
                save_video_tensor(pred_video_rollout, rollout_path, fps=fps)

                # ============ ADD SHARED ACTION SAVING HERE ============
                logger.info(f"  Saving shared-action rollout to: {shared_path}")
                save_video_tensor(pred_video_shared, shared_path, fps=fps)

                # ============ ADD RANDOM ACTION SAVING HERE ============
                logger.info(f"  Saving random-action rollout to: {random_path}")
                save_video_tensor(pred_video_random, random_path, fps=fps)

                video_count += 1

    # =========================================================================
    # Step 5: Compute aggregate metrics on remaining data
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: COMPUTING AGGREGATE METRICS")
    logger.info("=" * 80)

    all_metrics = []
    batch_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            h_sequence = batch['sequence'].to(device)  # [B, T, C=16, H=64, W=64]

            # Teacher forcing pass
            output = world_model(h_sequence)
            pred_frames_tf = output['pred_frames']
            rvq_losses = output['losses']

            # Autoregressive rollout pass
            pred_frames_rollout, per_step_mse_rollout = autoregressive_rollout(world_model, h_sequence)

            # Compute batch metrics
            batch_metrics = compute_metrics(
                model=world_model,
                h_sequence=h_sequence,
                pred_frames_tf=pred_frames_tf,
                pred_frames_rollout=pred_frames_rollout,
                per_step_mse_rollout=per_step_mse_rollout,
                rvq_losses=rvq_losses
            )

            all_metrics.append(batch_metrics)
            batch_count += 1

    # Compute averages (excluding per_step_mse_rollout which is a list)
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if key == 'per_step_mse_rollout':
            # Average per-step MSE across all batches
            # all_metrics[i]['per_step_mse_rollout'] is a list of length T-1
            T_steps = len(all_metrics[0]['per_step_mse_rollout'])
            avg_per_step = [
                sum(m['per_step_mse_rollout'][t] for m in all_metrics) / len(all_metrics)
                for t in range(T_steps)
            ]
            avg_metrics[key] = avg_per_step
        else:
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    logger.info(f"\nAggregate Metrics over {batch_count} batches ({split} set):")
    logger.info(f"  Total Loss: {avg_metrics['total_loss']:.6f}")
    logger.info(f"  Recon MSE (TF): {avg_metrics['recon_mse_tf']:.6f}")
    logger.info(f"  Recon MSE (Rollout): {avg_metrics['recon_mse_rollout']:.6f}")
    logger.info(f"  Action Commit Loss: {avg_metrics['action_commit_loss']:.6f}")
    logger.info(f"  Action Codebook Loss: {avg_metrics['action_codebook_loss']:.6f}")
    logger.info(f"  World Commit Loss: {avg_metrics['world_commit_loss']:.6f}")
    logger.info(f"  World Codebook Loss: {avg_metrics['world_codebook_loss']:.6f}")

    # Log per-step rollout MSE
    per_step_str = ", ".join([f"{mse:.6f}" for mse in avg_metrics['per_step_mse_rollout']])
    logger.info(f"  Rollout per-step MSE (avg): [{per_step_str}]")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("VIDEO GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Generated {video_count} video quintuplets ({video_count * 5} total videos)")
    logger.info(f"  - {video_count} ground truth videos")
    logger.info(f"  - {video_count} teacher forcing videos")
    logger.info(f"  - {video_count} autoregressive rollout videos")
    logger.info(f"  - {video_count} shared-action rollout videos")
    logger.info(f"  - {video_count} random-action rollout videos")
    logger.info(f"Computed metrics on {batch_count} batches")
    logger.info(f"Output directory: {output_path}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Generate videos from trained World Model')

    # Model arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to trained World Model checkpoint (.pt file)')
    parser.add_argument('--vvae_config_path', type=str, required=True,
                        help='Path to VVAE config file')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing VVAE HDF5 files')
    parser.add_argument('--manifest_path', type=str, required=True,
                        help='Path to manifest JSON file')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Which data split to use (default: val)')

    # Generation arguments
    parser.add_argument('--num_videos', type=int, default=5,
                        help='Number of video triplets to generate (default: 5)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for metric computation (default: 8)')
    parser.add_argument('--sequence_length', type=int, default=8,
                        help='Number of frames in sequence (default: 8)')
    parser.add_argument('--output_dir', type=str, default='./output/video_reconstructions',
                        help='Directory to save generated videos (default: ./output/video_reconstructions)')

    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading (default: 4)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for output videos (default: 10)')

    args = parser.parse_args()

    # Run video generation
    generate_videos(
        checkpoint_path=args.checkpoint_path,
        vvae_config_path=args.vvae_config_path,
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        split=args.split,
        num_videos=args.num_videos,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        output_dir=args.output_dir,
        device=args.device,
        num_workers=args.num_workers,
        fps=args.fps,
    )


if __name__ == '__main__':
    main()
