"""
Test trained world model:
1. Load model from checkpoint.
2. Load a manifest and dataset path.
3. Accept --train_pct, --val_pct, --test_pct .e.g. 0.5 for 50% of a dataset.
4. Run forward pass in batches for a given sequence length (see training code args).
5. Run evaluations (tf_mse, real rollout error, diagnal attention, codebook usage, and dPSNR) store value for each batch.
6. wandb: log distribution using wandb dist. Rollout expects an axis to show MSE on a rollout step.

Regarding evaluations to run:
    - tf_mse, diagnal attention, codebook usage, and dPSNR can all refer to the training.py
    - real rollout refers to running the model using a context of 1 frame, and autoregressively generate given real actions (from action encoder), and real world latent (from world encoder).
"""

import argparse
import os
import torch
import torch.nn.functional as F
from pathlib import Path
import logging
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Import from LAM modules
from LAM.training import load_model
from LAM.dataloader_vvae import VVAEDataset, create_dataloaders
from LAM.utils import codebook_usage, action_sensitivity_dsnr, diagonal_attention_score
from LAM.world_model import WorldModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Metric Functions
# =============================================================================

def compute_teacher_forcing_mse(model: WorldModel, h_sequence: torch.Tensor):
    """
    Compute teacher forcing MSE - standard forward pass with ground truth.

    Args:
        model: WorldModel instance
        h_sequence: [B, T, C=16, H=64, W=64] - ground truth sequence

    Returns:
        dict: {
            'tf_mse': float - reconstruction MSE
            'rvq_losses': dict - RVQ losses (for reference)
        }
    """
    # Run forward pass (simple teacher forcing, no rollout)
    output = model(h_sequence)
    pred_frames = output['pred_frames']  # [B, T-1, 16, 64, 64]
    rvq_losses = output['losses']

    # Compute reconstruction MSE (per description.md: GT[0:T-1] in → PRED[1:T] out)
    # Direct comparison, no slicing needed!
    recon_mse = F.mse_loss(pred_frames, h_sequence[:, 1:])

    return {
        'tf_mse': recon_mse.item(),
        'rvq_losses': rvq_losses
    }


def compute_real_rollout_error(model: WorldModel, h_sequence: torch.Tensor):
    """
    Compute autoregressive rollout error with real actions and world embedding.

    This is the most complex evaluation:
    - Start with context of 1 frame (frame 0)
    - Extract real actions from action encoder (using full GT sequence)
    - Extract real world embedding from world encoder (using full GT sequence)
    - Autoregressively predict frames 1, 2, ..., T-1
    - Compute MSE at each rollout step

    Args:
        model: WorldModel instance
        h_sequence: [B, T, C=16, H=64, W=64] - ground truth sequence

    Returns:
        dict: {
            'rollout_mse_per_step': list[float] - MSE at each step [step_1, step_2, ..., step_T-1]
            'rollout_mse_avg': float - average MSE across all steps
        }
    """
    B, T, C, H, W = h_sequence.shape
    device = h_sequence.device

    # =========================================================================
    # Step 1: Extract real actions and world embedding from GT sequence
    # =========================================================================
    # Tokenize full ground truth sequence
    tokens_gt, out_h, out_w = model.tokenizer(h_sequence)  # [B*T, d_model, out_h, out_w]

    # Extract real actions (what actions are present in the GT sequence)
    action_codes, _, _, _ = model.action_encoder(tokens_gt, B, T - 1)  # [B, T-1, d_code_a]

    # Extract real world embedding (world hypothesis from GT sequence)
    world_emb, _, _, _ = model.world_encoder(tokens_gt, B, T)  # [B, d_code_h]

    # =========================================================================
    # Step 2: Initialize autoregressive rollout
    # =========================================================================
    # Create a copy of GT tokens - we'll progressively replace with predictions
    # Start with GT frame 0, rest will be replaced during rollout
    rollout_tokens = tokens_gt.clone()

    # Track MSE at each prediction step
    mse_per_step = []

    # =========================================================================
    # Step 3: Autoregressive generation loop
    # =========================================================================
    for step in range(1, T):  # Predict frames 1, 2, ..., T-1
        # Predict using current context (frame 0 is GT, frames 1..step-1 are predictions)
        # The dynamics predictor uses causal masking, so frame 'step' will only
        # see frames 0..step-1 even though we pass the full sequence
        pred_tokens = model.dynamics_predictor(
            rollout_tokens,
            action_codes,
            world_emb,
            B,
            T
        )

        # Extract predicted tokens for frame 'step'
        # pred_tokens shape: [B*T, d_model, out_h, out_w]
        # We want tokens for frame 'step': indices [B*step : B*(step+1)]
        frame_start_idx = B * step
        frame_end_idx = B * (step + 1)
        pred_frame_tokens = pred_tokens[frame_start_idx:frame_end_idx, :, :, :]  # [B, d_model, out_h, out_w]

        # Decode predicted tokens to get the predicted frame
        pred_frame = model.detokenizer(pred_frame_tokens)  # [B, C, H, W]

        # Compute MSE against ground truth frame at step 'step'
        gt_frame = h_sequence[:, step, :, :, :]  # [B, C, H, W]
        step_mse = F.mse_loss(pred_frame, gt_frame)
        mse_per_step.append(step_mse.item())

        # CRITICAL: Replace tokens for frame 'step' with predicted tokens
        # This ensures the next prediction uses the predicted frame as context
        # (simulating real autoregressive generation where we don't have GT)
        rollout_tokens[frame_start_idx:frame_end_idx, :, :, :] = pred_frame_tokens.detach()

    # =========================================================================
    # Step 4: Compute average MSE
    # =========================================================================
    avg_mse = sum(mse_per_step) / len(mse_per_step)

    return {
        'rollout_mse_per_step': mse_per_step,
        'rollout_mse_avg': avg_mse,
    }


def compute_diagonal_attention(model: WorldModel, h_sequence: torch.Tensor):
    """
    Compute diagonal attention score for Action Encoder.

    Measures how well the action encoder focuses on main and upper diagonal
    (shifted causal masking with diagonal=2).

    Args:
        model: WorldModel instance
        h_sequence: [B, T, C=16, H=64, W=64] - ground truth sequence

    Returns:
        dict: {
            'diagonal_attention_per_block': list[float] - score for each transformer block
        }
    """
    B, T, C, H, W = h_sequence.shape

    # Tokenize sequence
    tokens, _, _ = model.tokenizer(h_sequence)

    # Run action_encoder with return_attention=True
    _, _, _, _, attention_weights = model.action_encoder(
        tokens, B, T - 1, return_attention=True
    )

    # attention_weights is a list of lists: [num_blocks][num_heads]
    # Each element is [B, t*out_h*out_w, t*out_h*out_w]
    block_diagonal_scores = []

    for block_attn in attention_weights:
        # block_attn is a list of attention from each head (one per patch scale)
        # For simplicity, use the first head's attention
        # Shape: [B, T*patches, T*patches]
        if len(block_attn) > 0:
            attn_first_head = block_attn[0]  # [B, T*patches, T*patches]

            # Average over batch dimension
            attn_avg = attn_first_head.mean(dim=0)  # [T*patches, T*patches]

            # Extract temporal dimension from spatial-temporal tokens
            # attn_avg: [T*patches, T*patches]
            num_patches = attn_avg.shape[0] // (T - 1)  # T-1 for action encoder

            # Average over spatial patches to get temporal attention [T, T]
            attn_temporal = attn_avg.view(T - 1, num_patches, T - 1, num_patches)
            attn_temporal = attn_temporal.mean(dim=(1, 3))  # [T-1, T-1]

            # Compute diagonal score
            score = diagonal_attention_score(attn_temporal.unsqueeze(0))  # Add head dimension
            block_diagonal_scores.append(score)

    return {
        'diagonal_attention_per_block': block_diagonal_scores
    }


def compute_codebook_usage(model: WorldModel, h_sequence: torch.Tensor):
    """
    Compute codebook usage for Action Encoder and World Encoder.

    Measures what percentage of codebook entries are used in this batch.

    Args:
        model: WorldModel instance
        h_sequence: [B, T, C=16, H=64, W=64] - ground truth sequence

    Returns:
        dict: {
            'action_usage': torch.Tensor - [num_lvq_levels] - unique codes used per level
            'world_usage': torch.Tensor - [num_lvq_levels] - unique codes used per level
            'action_usage_pct': list[float] - percentage per level
            'world_usage_pct': list[float] - percentage per level
        }
    """
    # Run forward pass to get indices
    output = model(h_sequence)

    # Extract indices
    action_indices = output['action_indices']  # [B, T-1, num_lvq_levels]
    world_indices = output['world_indices']    # [B, num_lvq_levels]

    # Use codebook_usage() function from utils.py
    action_usage = codebook_usage(action_indices)  # [num_lvq_levels]
    world_usage = codebook_usage(world_indices.unsqueeze(1))  # Add T dim, then [num_lvq_levels]

    # Get codebook sizes from model
    codebook_sizes_a = model.action_encoder.rvq.codebook_sizes
    codebook_sizes_h = model.world_encoder.rvq.codebook_sizes

    # Convert to percentages
    action_usage_pct = [action_usage[i].item() / codebook_sizes_a[i] * 100 for i in range(len(action_usage))]
    world_usage_pct = [world_usage[i].item() / codebook_sizes_h[i] * 100 for i in range(len(world_usage))]

    return {
        'action_usage': action_usage,
        'world_usage': world_usage,
        'action_usage_pct': action_usage_pct,
        'world_usage_pct': world_usage_pct,
    }


def compute_action_sensitivity(model: WorldModel, h_sequence: torch.Tensor):
    """
    Compute action sensitivity metric (dPSNR).

    Measures how sensitive the model is to correct vs random actions.

    Args:
        model: WorldModel instance
        h_sequence: [B, T, C=16, H=64, W=64] - ground truth sequence

    Returns:
        dict: {
            'dsnr': float - action sensitivity metric
            'psnr_seq': float - PSNR with correct actions
            'psnr_rand': float - PSNR with random actions
        }
    """
    # Call the existing function from utils.py
    dsnr, psnr_seq, psnr_rand = action_sensitivity_dsnr(model, h_sequence)

    return {
        'dsnr': dsnr,
        'psnr_seq': psnr_seq,
        'psnr_rand': psnr_rand,
    }


def test_model(
    checkpoint_path: str,
    data_dir: str,
    manifest_path: str,
    split: str = 'val',
    split_pct: float = 1.0,
    batch_size: int = 16,
    sequence_length: int = 16,
    device: str = 'cuda',
    num_workers: int = 4,
    use_wandb: bool = True,
    wandb_project: str = 'world_model_test',
    wandb_run_name: str = None,
    wandb_entity: str = None,
    seed: int = 42,
):
    """
    Test a trained world model on a dataset split.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        data_dir: Directory containing h5 files
        manifest_path: Path to manifest file
        split: Which split to test on ['train', 'val', 'test']
        split_pct: Percentage of split to use (0-1, default: 1.0 = 100%)
        batch_size: Batch size for evaluation
        sequence_length: Sequence length to use
        device: Device to run on ('cuda' or 'cpu')
        num_workers: Number of data loader workers
        use_wandb: Whether to log to wandb
        wandb_project: Wandb project name
        wandb_run_name: Wandb run name (auto-generated if None)
        wandb_entity: Wandb entity/team name
        seed: Random seed for reproducibility
    """
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    logger.info("=" * 80)
    logger.info("WORLD MODEL TESTING")
    logger.info("=" * 80)

    # =========================================================================
    # Step 1: Load Model from Checkpoint
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 1: LOADING MODEL")
    logger.info(f"{'='*80}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, optimizer, checkpoint_info = load_model(checkpoint_path, device=device)
    model.eval()  # Set to evaluation mode

    logger.info(f"\nModel loaded successfully!")
    logger.info(f"Checkpoint info:")
    logger.info(f"  Epoch: {checkpoint_info['epoch']}")
    logger.info(f"  Global step: {checkpoint_info['global_step']}")
    logger.info(f"  Best val recon MSE: {checkpoint_info['best_val_recon_mse']:.6f}")
    logger.info(f"  Best val total loss: {checkpoint_info['best_val_total_loss']:.6f}")

    # =========================================================================
    # Step 2: Setup Wandb (if enabled)
    # =========================================================================
    if use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("Wandb not available. Disabling wandb logging.")
            use_wandb = False
        else:
            logger.info(f"\n{'='*80}")
            logger.info("STEP 2: INITIALIZING WANDB")
            logger.info(f"{'='*80}")

            # Auto-generate run name if not provided
            if wandb_run_name is None:
                checkpoint_name = Path(checkpoint_path).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                wandb_run_name = f"test_{checkpoint_name}_{split}_{timestamp}"

            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                entity=wandb_entity,
                config={
                    'checkpoint_path': checkpoint_path,
                    'checkpoint_epoch': checkpoint_info['epoch'],
                    'checkpoint_step': checkpoint_info['global_step'],
                    'split': split,
                    'split_pct': split_pct,
                    'batch_size': batch_size,
                    'sequence_length': sequence_length,
                    'device': device,
                    'seed': seed,
                }
            )
            logger.info(f"Wandb initialized: {wandb_project}/{wandb_run_name}")

    # =========================================================================
    # Step 3: Load Dataset
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 3: LOADING DATASET")
    logger.info(f"{'='*80}")

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Manifest path: {manifest_path}")
    logger.info(f"Split: {split}")
    logger.info(f"Split percentage: {split_pct*100:.1f}%")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Sequence length: {sequence_length}")

    # Create data loaders
    # Note: We'll use the appropriate split and sample only split_pct of it
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        manifest_path=manifest_path,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_workers=num_workers,
        shuffle=False,  # Don't shuffle for testing
    )

    # Select the appropriate loader based on split
    if split == 'train':
        test_loader = train_loader
    elif split == 'val':
        test_loader = val_loader
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")

    total_batches = len(test_loader)
    num_test_batches = max(1, int(total_batches * split_pct))

    logger.info(f"\nDataset info:")
    logger.info(f"  Total batches in {split} split: {total_batches}")
    logger.info(f"  Testing on {num_test_batches} batches ({split_pct*100:.1f}%)")
    logger.info(f"  Total samples to test: ~{num_test_batches * batch_size}")

    # =========================================================================
    # Step 4: Run Evaluations
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 4: RUNNING EVALUATIONS")
    logger.info(f"{'='*80}")

    logger.info("\nEvaluations to run:")
    logger.info("  1. Teacher forcing MSE (tf_mse)")
    logger.info("  2. Real rollout error (autoregressive generation)")
    logger.info("  3. Diagonal attention score")
    logger.info("  4. Codebook usage")
    logger.info("  5. Action sensitivity (dPSNR)")

    # Results accumulators for all batches
    all_tf_mse = []
    all_rollout_mse_per_step = []  # List of lists: [[step1, step2, ...], [step1, step2, ...], ...]
    all_diagonal_attention = []  # List of lists: [[block1, block2, ...], [block1, block2, ...], ...]
    all_action_usage = []  # Per-batch usage
    all_world_usage = []  # Per-batch usage
    all_action_indices = []  # For global usage computation
    all_world_indices = []  # For global usage computation
    all_dsnr = []
    all_psnr_seq = []
    all_psnr_rand = []

    # Evaluation loop
    logger.info(f"\nProcessing {num_test_batches} batches...")
    batch_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Stop after processing enough batches
            if batch_count >= num_test_batches:
                break

            h_sequence = batch['sequence'].to(device)  # [B, T, 16, 64, 64]
            batch_count += 1

            # Progress logging every 10 batches
            if batch_count % 10 == 0 or batch_count == 1:
                logger.info(f"Processing batch {batch_count}/{num_test_batches}...")

            # =================================================================
            # Metric 1: Teacher Forcing MSE
            # =================================================================
            try:
                tf_metrics = compute_teacher_forcing_mse(model, h_sequence)
                all_tf_mse.append(tf_metrics['tf_mse'])
            except NotImplementedError:
                if batch_count == 1:
                    logger.warning("  [SKIPPED] Teacher forcing MSE - not yet implemented")

            # =================================================================
            # Metric 2: Real Rollout Error
            # =================================================================
            try:
                rollout_metrics = compute_real_rollout_error(model, h_sequence)
                all_rollout_mse_per_step.append(rollout_metrics['rollout_mse_per_step'])
            except NotImplementedError:
                if batch_count == 1:
                    logger.warning("  [SKIPPED] Real rollout error - not yet implemented")

            # =================================================================
            # Metric 3: Diagonal Attention
            # =================================================================
            try:
                diag_metrics = compute_diagonal_attention(model, h_sequence)
                all_diagonal_attention.append(diag_metrics['diagonal_attention_per_block'])
            except NotImplementedError:
                if batch_count == 1:
                    logger.warning("  [SKIPPED] Diagonal attention - not yet implemented")

            # =================================================================
            # Metric 4: Codebook Usage
            # =================================================================
            try:
                usage_metrics = compute_codebook_usage(model, h_sequence)
                all_action_usage.append(usage_metrics['action_usage'])
                all_world_usage.append(usage_metrics['world_usage'])
                # Also store indices for global usage computation
                output = model(h_sequence)
                all_action_indices.append(output['action_indices'].cpu())  # [B, T-1, 3]
                all_world_indices.append(output['world_indices'].cpu())    # [B, 3]
            except NotImplementedError:
                if batch_count == 1:
                    logger.warning("  [SKIPPED] Codebook usage - not yet implemented")

            # =================================================================
            # Metric 5: Action Sensitivity (dPSNR)
            # =================================================================
            try:
                sensitivity_metrics = compute_action_sensitivity(model, h_sequence)
                all_dsnr.append(sensitivity_metrics['dsnr'])
                all_psnr_seq.append(sensitivity_metrics['psnr_seq'])
                all_psnr_rand.append(sensitivity_metrics['psnr_rand'])
            except NotImplementedError:
                if batch_count == 1:
                    logger.warning("  [SKIPPED] Action sensitivity - not yet implemented")

    logger.info(f"\nProcessed {batch_count} batches total.")

    # =========================================================================
    # Step 5: Compute Aggregate Statistics and Log to Wandb
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 5: COMPUTING AGGREGATE STATISTICS")
    logger.info(f"{'='*80}")

    # Teacher Forcing MSE
    if len(all_tf_mse) > 0:
        avg_tf_mse = sum(all_tf_mse) / len(all_tf_mse)
        logger.info(f"\nTeacher Forcing MSE:")
        logger.info(f"  Average: {avg_tf_mse:.6f}")
        logger.info(f"  Min: {min(all_tf_mse):.6f}")
        logger.info(f"  Max: {max(all_tf_mse):.6f}")

        if use_wandb:
            # Log individual batch values
            for i, mse in enumerate(all_tf_mse):
                wandb.log({'Test/tf_mse': mse}, step=i)
            # Log summary statistics
            wandb.log({
                'Test_Summary/tf_mse_mean': avg_tf_mse,
                'Test_Summary/tf_mse_min': min(all_tf_mse),
                'Test_Summary/tf_mse_max': max(all_tf_mse),
            })

    # Real Rollout Error
    if len(all_rollout_mse_per_step) > 0:
        # all_rollout_mse_per_step is a list of lists: [[step1, step2, ...], ...]
        # We need to compute average MSE per step across all batches
        num_steps = len(all_rollout_mse_per_step[0])
        avg_mse_per_step = []
        for step_idx in range(num_steps):
            step_mses = [batch_mses[step_idx] for batch_mses in all_rollout_mse_per_step]
            avg_mse_per_step.append(sum(step_mses) / len(step_mses))

        logger.info(f"\nReal Rollout Error:")
        logger.info(f"  Average MSE per step:")
        for step_idx, mse in enumerate(avg_mse_per_step):
            logger.info(f"    Step {step_idx + 1}: {mse:.6f}")
        overall_avg = sum(avg_mse_per_step) / len(avg_mse_per_step)
        logger.info(f"  Overall average: {overall_avg:.6f}")

        if use_wandb:
            # Log MSE per rollout step
            for step_idx, mse in enumerate(avg_mse_per_step):
                wandb.log({f'Test_Rollout/step_{step_idx + 1}_mse': mse})
            wandb.log({'Test_Summary/rollout_mse_overall': overall_avg})

    # Diagonal Attention
    if len(all_diagonal_attention) > 0:
        # all_diagonal_attention is a list of lists: [[block1, block2, block3], ...]
        num_blocks = len(all_diagonal_attention[0])
        avg_attention_per_block = []
        for block_idx in range(num_blocks):
            block_scores = [batch_scores[block_idx] for batch_scores in all_diagonal_attention]
            avg_attention_per_block.append(sum(block_scores) / len(block_scores))

        logger.info(f"\nDiagonal Attention:")
        for block_idx, score in enumerate(avg_attention_per_block):
            logger.info(f"  Block {block_idx + 1}: {score:.4f}")

        if use_wandb:
            for block_idx, score in enumerate(avg_attention_per_block):
                wandb.log({f'Test_DiagonalAttention/block_{block_idx + 1}': score})

    # Codebook Usage
    if len(all_action_usage) > 0:
        # 1. Per-batch average: Stack and average
        action_usage_stacked = torch.stack(all_action_usage)  # [num_batches, num_levels]
        world_usage_stacked = torch.stack(all_world_usage)
        avg_action_usage_per_batch = action_usage_stacked.float().mean(dim=0)  # [num_levels]
        avg_world_usage_per_batch = world_usage_stacked.float().mean(dim=0)

        # 2. Overall/global: Unique codes across ALL test batches
        action_indices_all = torch.cat(all_action_indices, dim=0)  # [total_samples, T-1, 3]
        world_indices_all = torch.cat(all_world_indices, dim=0)    # [total_samples, 3]

        # Compute global unique codes for each level
        global_action_usage = []
        for level_idx in range(action_indices_all.shape[2]):  # num_levels
            indices_level = action_indices_all[:, :, level_idx].flatten()
            unique_codes = torch.unique(indices_level).numel()
            global_action_usage.append(unique_codes)

        global_world_usage = []
        for level_idx in range(world_indices_all.shape[1]):  # num_levels
            indices_level = world_indices_all[:, level_idx].flatten()
            unique_codes = torch.unique(indices_level).numel()
            global_world_usage.append(unique_codes)

        logger.info(f"\nCodebook Usage:")
        logger.info(f"  Action Encoder (per-batch average unique codes):")
        for level_idx, usage in enumerate(avg_action_usage_per_batch):
            logger.info(f"    Level {level_idx + 1}: {usage:.1f}")
        logger.info(f"  Action Encoder (overall unique codes across all batches):")
        for level_idx, usage in enumerate(global_action_usage):
            logger.info(f"    Level {level_idx + 1}: {usage}")
        logger.info(f"  World Encoder (per-batch average unique codes):")
        for level_idx, usage in enumerate(avg_world_usage_per_batch):
            logger.info(f"    Level {level_idx + 1}: {usage:.1f}")
        logger.info(f"  World Encoder (overall unique codes across all batches):")
        for level_idx, usage in enumerate(global_world_usage):
            logger.info(f"    Level {level_idx + 1}: {usage}")

        if use_wandb:
            for level_idx, usage in enumerate(avg_action_usage_per_batch):
                wandb.log({f'Test_CodebookUsage/action_per_batch_level_{level_idx + 1}': usage.item()})
            for level_idx, usage in enumerate(global_action_usage):
                wandb.log({f'Test_CodebookUsage/action_overall_level_{level_idx + 1}': usage})
            for level_idx, usage in enumerate(avg_world_usage_per_batch):
                wandb.log({f'Test_CodebookUsage/world_per_batch_level_{level_idx + 1}': usage.item()})
            for level_idx, usage in enumerate(global_world_usage):
                wandb.log({f'Test_CodebookUsage/world_overall_level_{level_idx + 1}': usage})

    # Action Sensitivity (dPSNR)
    if len(all_dsnr) > 0:
        avg_dsnr = sum(all_dsnr) / len(all_dsnr)
        avg_psnr_seq = sum(all_psnr_seq) / len(all_psnr_seq)
        avg_psnr_rand = sum(all_psnr_rand) / len(all_psnr_rand)

        logger.info(f"\nAction Sensitivity:")
        logger.info(f"  dPSNR: {avg_dsnr:.4f}")
        logger.info(f"  PSNR_seq: {avg_psnr_seq:.4f}")
        logger.info(f"  PSNR_rand: {avg_psnr_rand:.4f}")

        if use_wandb:
            wandb.log({
                'Test_Summary/dsnr': avg_dsnr,
                'Test_Summary/psnr_seq': avg_psnr_seq,
                'Test_Summary/psnr_rand': avg_psnr_rand,
            })

    # =========================================================================
    # Cleanup
    # =========================================================================
    if use_wandb:
        wandb.finish()
        logger.info("\nWandb run finished.")

    logger.info("\n" + "=" * 80)
    logger.info("TESTING COMPLETE")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Test trained World Model')

    # Model checkpoint
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing h5 files')
    parser.add_argument('--manifest_path', type=str, required=True,
                        help='Path to manifest file')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                        help='Which split to test on (default: val)')
    parser.add_argument('--split_pct', type=float, default=1.0,
                        help='Percentage of split to use, 0-1 (default: 1.0 = 100%%)')

    # Batch processing
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation (default: 16)')
    parser.add_argument('--sequence_length', type=int, default=16,
                        help='Sequence length to use (default: 16)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu (default: cuda)')

    # Wandb
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='world_model_test',
                        help='Wandb project name (default: world_model_test)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity/team name (default: None)')

    # Seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Validate arguments
    if args.split_pct <= 0 or args.split_pct > 1:
        raise ValueError(f"split_pct must be between 0 and 1, got {args.split_pct}")

    # Run testing
    test_model(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        split=args.split,
        split_pct=args.split_pct,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=args.device,
        num_workers=args.num_workers,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
