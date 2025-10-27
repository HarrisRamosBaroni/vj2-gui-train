"""
Overfit Test for World Model

Tests if the model can overfit to a single datapoint by training on it repeatedly.
This verifies that the model has sufficient capacity and the training loop works correctly.

From description.md:
- Use the same data loading method as training: h5 and manifest
- Take a single datapoint from the dataset
- Log all the losses
- steps default to 1000, run over one datapoint with steps many times to see if the model overfits
"""

import argparse
import os
import torch
import torch.nn.functional as F
from pathlib import Path
import logging
import wandb
from datetime import datetime

from LAM.world_model import WorldModel
from LAM.dataloader_vvae import create_vvae_dataloaders

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_rollout_losses(model, h_sequence, weights=(1.0, 0.8, 0.5)):
    """
    Compute Teacher Forcing (TF) loss and Rollout losses.

    Rollout strategy:
    1. Teacher Forcing (TF): Use ground truth frames as input, predict next frame
    2. Rollout 1: Use TF predictions to replace frames [1:T], predict again
    3. Rollout 2: Use Rollout 1 predictions to replace frames [1:T], predict again

    This simulates autoregressive generation while still using parallel computation
    via teacher forcing with causal masking.

    Args:
        model: WorldModel instance
        h_sequence: [B, T, 16, 64, 64] - ground truth VVAE latent sequence
        weights: Tuple of (weight_tf, weight_roll1, weight_roll2) for loss weighting

    Returns:
        Dictionary containing:
            - total_recon_loss: Weighted sum of TF + Roll1 + Roll2 reconstruction losses
            - tf_loss: Teacher forcing reconstruction loss
            - rollout1_loss: Rollout 1 step reconstruction loss
            - rollout2_loss: Rollout 2 step reconstruction loss
            - action_commit_loss: Action encoder RVQ commitment loss
            - action_codebook_loss: Action encoder RVQ codebook loss
            - world_commit_loss: World encoder RVQ commitment loss
            - world_codebook_loss: World encoder RVQ codebook loss
    """
    B, T, C, H, W = h_sequence.shape
    weight_tf, weight_roll1, weight_roll2 = weights

    # =========================================================================
    # Step 1: Teacher Forcing (TF) - Use ground truth frames
    # =========================================================================
    output_tf = model(h_sequence)
    pred_frames_tf = output_tf['pred_frames']  # [B, T-1, 16, 64, 64] (predicting frames 1 to T)

    # Compute TF reconstruction loss (per description.md: GT[0:T-1] in → PRED[1:T] out)
    # Direct comparison, no slicing needed!
    tf_loss = F.mse_loss(pred_frames_tf, h_sequence[:, 1:])

    # Extract RVQ losses (only computed once, shared across all rollouts)
    action_commit_loss = output_tf['losses']['action_commit_loss']
    action_codebook_loss = output_tf['losses']['action_codebook_loss']
    world_commit_loss = output_tf['losses']['world_commit_loss']
    world_codebook_loss = output_tf['losses']['world_codebook_loss']

    # =========================================================================
    # Step 2: Rollout 1 - Use TF predictions as context, predict again
    # =========================================================================
    # Replace frames [1:T] with TF predictions, keep frame 0 as ground truth
    # pred_frames_tf already has shape [B, T-1, 16, 64, 64]
    h_rollout1 = torch.cat([
        h_sequence[:, :1, :, :, :],  # Keep first frame [B, 1, 16, 64, 64]
        pred_frames_tf.detach()  # T-1 predictions [B, T-1, 16, 64, 64]
    ], dim=1)  # [B, T, 16, 64, 64]

    output_roll1 = model(h_rollout1)
    pred_frames_roll1 = output_roll1['pred_frames']  # [B, T-1, 16, 64, 64]

    # Compute Rollout 1 reconstruction loss (no slicing needed!)
    rollout1_loss = F.mse_loss(pred_frames_roll1, h_sequence[:, 1:])

    # =========================================================================
    # Step 3: Rollout 2 - Use Rollout 1 predictions as context, predict again
    # =========================================================================
    # Replace frames [1:T] with Rollout 1 predictions, keep frame 0 as ground truth
    # pred_frames_roll1 already has shape [B, T-1, 16, 64, 64]
    h_rollout2 = torch.cat([
        h_sequence[:, :1, :, :, :],  # Keep first frame [B, 1, 16, 64, 64]
        pred_frames_roll1.detach()  # T-1 predictions [B, T-1, 16, 64, 64]
    ], dim=1)  # [B, T, 16, 64, 64]

    output_roll2 = model(h_rollout2)
    pred_frames_roll2 = output_roll2['pred_frames']  # [B, T-1, 16, 64, 64]

    # Compute Rollout 2 reconstruction loss (no slicing needed!)
    rollout2_loss = F.mse_loss(pred_frames_roll2, h_sequence[:, 1:])

    # =========================================================================
    # Step 4: Combine losses with weights
    # =========================================================================
    total_recon_loss = (
        weight_tf * tf_loss +
        weight_roll1 * rollout1_loss +
        weight_roll2 * rollout2_loss
    )

    return {
        'total_recon_loss': total_recon_loss,
        'tf_loss': tf_loss,
        'rollout1_loss': rollout1_loss,
        'rollout2_loss': rollout2_loss,
        'action_commit_loss': action_commit_loss,
        'action_codebook_loss': action_codebook_loss,
        'world_commit_loss': world_commit_loss,
        'world_codebook_loss': world_codebook_loss,
    }


def overfit_test(
    data_dir: str,
    manifest_path: str,
    steps: int = 1000,
    sequence_length: int = 8,
    lr: float = 1e-4,
    device: str = 'cuda',
    log_interval: int = 10,
    # Model hyperparameters
    d_model: int = 256,
    d_code_a: int = 128,
    d_code_h: int = 128,
    num_lvq_levels_a: int = 3,
    num_lvq_levels_h: int = 3,
    codebook_sizes_a: tuple = (256, 256, 256),
    codebook_sizes_h: tuple = (256, 256, 256),
    # Loss weights
    rollout_weights: tuple = (1.0, 0.8, 0.5),
):
    """
    Overfit test: Train model on a single datapoint to verify capacity and training loop.

    Args:
        data_dir: Path to directory containing VVAE HDF5 files
        manifest_path: Path to manifest JSON file
        steps: Number of training steps (default 1000)
        sequence_length: Number of frames in sequence (default 8)
        lr: Learning rate (default 1e-4)
        device: Device to use ('cuda' or 'cpu')
        log_interval: Log every N steps (default 10)
        d_model: Feature dimension after tokenization (default 256)
        d_code_a: Action code dimension (default 128)
        d_code_h: World hypothesis dimension (default 128)
        num_lvq_levels_a: Number of RVQ levels for action encoder (default 3)
        num_lvq_levels_h: Number of RVQ levels for world encoder (default 3)
        codebook_sizes_a: Tuple of codebook sizes per level for actions (default (256, 256, 256))
        codebook_sizes_h: Tuple of codebook sizes per level for world (default (256, 256, 256))
        rollout_weights: Tuple of (weight_tf, weight_roll1, weight_roll2) (default (1.0, 0.8, 0.5))
    """
    logger.info("=" * 80)
    logger.info("OVERFIT TEST - Training on a single datapoint")
    logger.info("=" * 80)

    # =========================================================================
    # Initialize Wandb
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_project = "latent_hypothesis_action_model"
    wandb_run_name = f"overfit_test_{timestamp}"

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            'steps': steps,
            'sequence_length': sequence_length,
            'lr': lr,
            'd_model': d_model,
            'd_code_a': d_code_a,
            'd_code_h': d_code_h,
            'num_lvq_levels_a': num_lvq_levels_a,
            'num_lvq_levels_h': num_lvq_levels_h,
            'codebook_sizes_a': codebook_sizes_a,
            'codebook_sizes_h': codebook_sizes_h,
            'rollout_weights': rollout_weights,
        }
    )
    logger.info(f"Wandb initialized: project={wandb_project}, run={wandb_run_name}")

    # =========================================================================
    # Step 1: Load a single datapoint
    # =========================================================================
    logger.info(f"Loading data from {data_dir}")
    logger.info(f"Manifest: {manifest_path}")

    train_loader, _, _ = create_vvae_dataloaders(
        data_dir=data_dir,
        manifest_path=manifest_path,
        batch_size=1,  # Single datapoint
        sequence_length=sequence_length,
        stride_train=1,
        num_workers=0,  # No multiprocessing for single datapoint
        ddp=False
    )

    # Get the first (and only) datapoint we'll train on
    single_datapoint = next(iter(train_loader))
    h_sequence = single_datapoint['sequence'].to(device)  # [1, T, 16, 64, 64]

    B, T, C, H, W = h_sequence.shape
    logger.info(f"Loaded single datapoint: shape={h_sequence.shape}")
    logger.info(f"  Batch size: {B}")
    logger.info(f"  Sequence length: {T}")
    logger.info(f"  Channels: {C}, Height: {H}, Width: {W}")
    logger.info(f"  Value range: [{h_sequence.min():.4f}, {h_sequence.max():.4f}]")

    # =========================================================================
    # Step 2: Initialize model
    # =========================================================================
    logger.info("\nInitializing WorldModel...")
    logger.info(f"  d_model: {d_model}")
    logger.info(f"  d_code_a: {d_code_a}, num_lvq_levels_a: {num_lvq_levels_a}, codebook_sizes_a: {codebook_sizes_a}")
    logger.info(f"  d_code_h: {d_code_h}, num_lvq_levels_h: {num_lvq_levels_h}, codebook_sizes_h: {codebook_sizes_h}")

    model = WorldModel(
        d_model=d_model,
        d_code_a=d_code_a,
        d_code_h=d_code_h,
        num_lvq_levels_a=num_lvq_levels_a,
        num_lvq_levels_h=num_lvq_levels_h,
        codebook_sizes_a=codebook_sizes_a,
        codebook_sizes_h=codebook_sizes_h,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    # =========================================================================
    # Step 3: Setup optimizer
    # =========================================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logger.info(f"\nOptimizer: Adam(lr={lr})")
    logger.info(f"Rollout weights: TF={rollout_weights[0]}, Roll1={rollout_weights[1]}, Roll2={rollout_weights[2]}")

    # =========================================================================
    # Step 4: Training loop
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(f"Starting overfit training for {steps} steps...")
    logger.info("=" * 80)
    logger.info(f"{'Step':>6} | {'Total':>10} | {'TF':>10} | {'Roll1':>10} | {'Roll2':>10} | "
                f"{'A_Commit':>10} | {'A_Code':>10} | {'W_Commit':>10} | {'W_Code':>10}")
    logger.info("-" * 80)

    model.train()

    for step in range(steps):
        optimizer.zero_grad()

        # Compute losses with rollout
        losses = compute_rollout_losses(model, h_sequence, weights=rollout_weights)

        # Total loss = reconstruction losses + RVQ losses
        total_loss = (
            losses['total_recon_loss'] +
            losses['action_commit_loss'] +
            losses['action_codebook_loss'] +
            losses['world_commit_loss'] +
            losses['world_codebook_loss']
        )

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Logging
        if (step + 1) % log_interval == 0 or step == 0:
            logger.info(
                f"{step+1:6d} | "
                f"{total_loss.item():10.6f} | "
                f"{losses['tf_loss'].item():10.6f} | "
                f"{losses['rollout1_loss'].item():10.6f} | "
                f"{losses['rollout2_loss'].item():10.6f} | "
                f"{losses['action_commit_loss'].item():10.6f} | "
                f"{losses['action_codebook_loss'].item():10.6f} | "
                f"{losses['world_commit_loss'].item():10.6f} | "
                f"{losses['world_codebook_loss'].item():10.6f}"
            )

            # Wandb logging
            wandb.log({
                'Overfit/total_loss': total_loss.item(),
                'Overfit/tf_loss': losses['tf_loss'].item(),
                'Overfit/rollout1_loss': losses['rollout1_loss'].item(),
                'Overfit/rollout2_loss': losses['rollout2_loss'].item(),
                'Overfit/action_commit_loss': losses['action_commit_loss'].item(),
                'Overfit/action_codebook_loss': losses['action_codebook_loss'].item(),
                'Overfit/world_commit_loss': losses['world_commit_loss'].item(),
                'Overfit/world_codebook_loss': losses['world_codebook_loss'].item(),
            }, step=step + 1)

    # =========================================================================
    # Step 5: Final evaluation
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 80)

    model.eval()
    with torch.no_grad():
        losses = compute_rollout_losses(model, h_sequence, weights=rollout_weights)
        total_loss = (
            losses['total_recon_loss'] +
            losses['action_commit_loss'] +
            losses['action_codebook_loss'] +
            losses['world_commit_loss'] +
            losses['world_codebook_loss']
        )

    logger.info(f"Final Total Loss:       {total_loss.item():.6f}")
    logger.info(f"Final TF Loss:          {losses['tf_loss'].item():.6f}")
    logger.info(f"Final Rollout 1 Loss:   {losses['rollout1_loss'].item():.6f}")
    logger.info(f"Final Rollout 2 Loss:   {losses['rollout2_loss'].item():.6f}")
    logger.info(f"Final Action Commit:    {losses['action_commit_loss'].item():.6f}")
    logger.info(f"Final Action Codebook:  {losses['action_codebook_loss'].item():.6f}")
    logger.info(f"Final World Commit:     {losses['world_commit_loss'].item():.6f}")
    logger.info(f"Final World Codebook:   {losses['world_codebook_loss'].item():.6f}")

    # Log final results to wandb
    wandb.log({
        'Overfit/final_total_loss': total_loss.item(),
        'Overfit/final_tf_loss': losses['tf_loss'].item(),
        'Overfit/final_rollout1_loss': losses['rollout1_loss'].item(),
        'Overfit/final_rollout2_loss': losses['rollout2_loss'].item(),
    }, step=steps)

    logger.info("\n" + "=" * 80)
    logger.info("OVERFIT TEST COMPLETE")
    logger.info("=" * 80)

    # Check if model successfully overfitted
    if losses['tf_loss'].item() < 0.01:
        logger.info("SUCCESS: Model successfully overfitted to single datapoint (TF loss < 0.01)")
    else:
        logger.warning(f"WARNING: Model may not have fully overfitted (TF loss = {losses['tf_loss'].item():.6f})")
        logger.warning("Consider: increasing steps, adjusting learning rate, or checking model capacity")

    # Finish wandb run
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Overfit test for World Model')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing VVAE HDF5 files')
    parser.add_argument('--manifest_path', type=str, required=True,
                        help='Path to manifest JSON file')

    # Training arguments
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of training steps (default: 1000)')
    parser.add_argument('--sequence_length', type=int, default=8,
                        help='Number of frames in sequence (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N steps (default: 10)')

    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=256,
                        help='Feature dimension after tokenization (default: 256)')
    parser.add_argument('--d_code_a', type=int, default=128,
                        help='Action code dimension (default: 128)')
    parser.add_argument('--d_code_h', type=int, default=128,
                        help='World hypothesis dimension (default: 128)')
    parser.add_argument('--num_lvq_levels_a', type=int, default=3,
                        help='Number of RVQ levels for action encoder (default: 3)')
    parser.add_argument('--num_lvq_levels_h', type=int, default=3,
                        help='Number of RVQ levels for world encoder (default: 3)')

    args = parser.parse_args()

    # Run overfit test
    overfit_test(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        steps=args.steps,
        sequence_length=args.sequence_length,
        lr=args.lr,
        device=args.device,
        log_interval=args.log_interval,
        d_model=args.d_model,
        d_code_a=args.d_code_a,
        d_code_h=args.d_code_h,
        num_lvq_levels_a=args.num_lvq_levels_a,
        num_lvq_levels_h=args.num_lvq_levels_h,
    )


if __name__ == '__main__':
    main()
