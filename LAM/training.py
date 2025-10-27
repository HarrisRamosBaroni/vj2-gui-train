"""
World Model Training Script

Implementation steps:
    1. Data loader. ✓
    2. Training Loop and Validation Loop. ✓
    3. Weight saving mechanism. ✓
    4. Training tricks implementation: Input dropouts, rollout simulation. ✓
    5. Basic Logging on wandb. ✓
    6. Further validation set loggings on wandb: ✓ (codebook usage, dSNR, diagonal attention)
    7. Overall revision, adding commandline arguments. ✓

ALL STEPS COMPLETE!
"""

import argparse
import os
import random
import torch
import torch.nn.functional as F
from pathlib import Path
import logging
from datetime import datetime
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from LAM.world_model import WorldModel
from LAM.dataloader_vvae import create_vvae_dataloaders
from LAM.utils import codebook_usage, action_sensitivity_dsnr, world_sensitivity_dsnr, diagonal_attention_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_model(
    checkpoint_path: str,
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_recon_mse: float,
    best_val_total_loss: float,
    model_config: dict,
):
    """
    Save model checkpoint with all necessary information to fully reconstruct the model.

    Args:
        checkpoint_path: Full path to save the checkpoint
        model: WorldModel instance
        optimizer: Optimizer instance
        epoch: Current epoch number
        global_step: Current global step
        best_val_recon_mse: Best validation reconstruction MSE so far
        best_val_total_loss: Best validation total loss so far
        model_config: Dictionary containing model hyperparameters
    """
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_recon_mse': best_val_recon_mse,
        'best_val_total_loss': best_val_total_loss,
        'model_config': model_config,
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"  Saved checkpoint: {checkpoint_path}")


def load_model(
    checkpoint_path: str,
    device: str = 'cuda',
):
    """
    Load model checkpoint and reconstruct the WorldModel.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to

    Returns:
        model: Reconstructed WorldModel instance
        optimizer: Optimizer with loaded state
        checkpoint_info: Dictionary containing epoch, global_step, and best metrics
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct model from saved config
    model_config = checkpoint['model_config']
    model = WorldModel(
        d_model=model_config['d_model'],
        d_code_a=model_config['d_code_a'],
        d_code_h=model_config['d_code_h'],
        num_lvq_levels_a=model_config['num_lvq_levels_a'],
        num_lvq_levels_h=model_config['num_lvq_levels_h'],
        codebook_sizes_a=tuple(model_config['codebook_sizes_a']),
        codebook_sizes_h=tuple(model_config['codebook_sizes_h']),
        use_random_temporal_pe=model_config.get('use_random_temporal_pe', False),  # Backward compatibility
        max_pe_offset=model_config.get('max_pe_offset', 120),  # Backward compatibility
    ).to(device)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Reconstruct optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Prepare checkpoint info
    checkpoint_info = {
        'epoch': checkpoint['epoch'],
        'global_step': checkpoint['global_step'],
        'best_val_recon_mse': checkpoint['best_val_recon_mse'],
        'best_val_total_loss': checkpoint['best_val_total_loss'],
    }

    logger.info(f"  Loaded model from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
    logger.info(f"  Best val recon MSE: {checkpoint['best_val_recon_mse']:.6f}")
    logger.info(f"  Best val total loss: {checkpoint['best_val_total_loss']:.6f}")

    return model, optimizer, checkpoint_info


def save_training_config(config_path: str, config: dict):
    """
    Save complete training configuration to JSON file.

    Args:
        config_path: Path to save config.json
        config: Dictionary containing all training hyperparameters
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"  Saved training config: {config_path}")


def load_training_config(config_path: str) -> dict:
    """
    Load training configuration from JSON file.

    Args:
        config_path: Path to config.json

    Returns:
        config: Dictionary containing all training hyperparameters
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info(f"  Loaded training config from: {config_path}")
    return config


def apply_token_masking(tokens: torch.Tensor, B: int, T: int, mask_prob: float = 0.1) -> torch.Tensor:
    """
    Apply random frame masking to tokens post tokenizer.

    This should be applied to world encoder and dynamics predictor inputs,
    but NOT to action encoder (per description.md:85).

    Args:
        tokens: Tokenized sequence [B*T, d_model, H', W']
        B: Batch size
        T: Sequence length
        mask_prob: Probability of masking a frame (default: 0.1)

    Returns:
        masked_tokens: Tokens with random frames set to zero
    """
    _, d_model, H_prime, W_prime = tokens.shape

    # Create Bernoulli mask for each frame in the batch
    # Shape: [B, T, 1, 1, 1] to broadcast over d_model, H', W'
    mask = torch.bernoulli(torch.full((B, T, 1, 1, 1), 1 - mask_prob, device=tokens.device))

    # Reshape mask to [B*T, 1, 1, 1] to match tokens
    mask = mask.view(B * T, 1, 1, 1)

    # Apply mask (multiply by mask, so masked frames become 0)
    masked_tokens = tokens * mask

    return masked_tokens


def compute_rollout_loss(
    model: WorldModel,
    h_sequence: torch.Tensor,
    rollout_steps: int = 4,
    weights: tuple = (1.0, 0.8, 0.5, 0.3, 0.1),
    mask_prob: float = 0.0,
    ablate_action: bool = False,
    ablate_world: bool = False,
) -> dict:
    """
    Compute teacher forcing loss and rollout losses with context replacement.

    The predictor uses previous predictions as context (input) to simulate rollout.
    Action codes and world embedding are extracted ONCE from GT and remain fixed.
    Only the dynamics predictor receives updated context during rollout.

    Token masking (per description.md:85):
    - Applied POST tokenization to world encoder and dynamics predictor
    - NOT applied to action encoder

    Args:
        model: WorldModel instance
        h_sequence: Ground truth sequence [B, T, C=16, H=64, W=64]
        rollout_steps: Number of rollout steps (default: 4)
        weights: Tuple of weights for (TF, Roll1, Roll2, Roll3, Roll4) losses (default: (1.0, 0.8, 0.5, 0.3, 0.1))
        mask_prob: Probability of masking tokens (default: 0.0 = no masking)
        ablate_action: If True, zero out all action codes (default: False)
        ablate_world: If True, zero out world embedding (default: False)

    Returns:
        Dictionary containing:
            - 'total_recon_loss': Weighted sum of all reconstruction losses
            - 'tf_loss': Teacher forcing loss
            - 'rollout_losses': List of rollout losses for each step
            - 'rvq_losses': Dictionary of RVQ losses from teacher forcing pass
    """
    B, T, C, H, W = h_sequence.shape
    device = h_sequence.device

    # =========================================================================
    # Step 1: Teacher Forcing with token masking
    # =========================================================================
    # Tokenize GT sequence once
    tokens_gt, out_h, out_w = model.tokenizer(h_sequence)  # [B*T, d_model, out_h, out_w]

    # Apply masking for world encoder and dynamics predictor (but NOT action encoder)
    if mask_prob > 0:
        tokens_gt_masked = apply_token_masking(tokens_gt, B, T, mask_prob)
    else:
        tokens_gt_masked = tokens_gt

    # Extract action codes from UNMASKED GT tokens
    action_codes, action_indices, action_commit_loss, action_codebook_loss = \
        model.action_encoder(tokens_gt, B, T - 1)  # [B, T-1, d_code_a]

    # Extract world embedding from MASKED GT tokens
    world_emb, world_indices, world_commit_loss, world_codebook_loss = \
        model.world_encoder(tokens_gt_masked, B, T)  # [B, d_code_h]

    # Apply ablation: zero out action codes and/or world embedding if requested
    if ablate_action:
        action_codes = torch.zeros_like(action_codes)
    if ablate_world:
        world_emb = torch.zeros_like(world_emb)

    # Per description.md lines 40,46: Drop last frame, pass GT[0:T-1] to predictor
    # This gives us perfect alignment: T-1 frames + T-1 actions → T-1 predictions
    tokens_gt_masked_input = tokens_gt_masked[:B * (T - 1)]  # [B*(T-1), d_model, out_h, out_w]

    # Run dynamics predictor with MASKED tokens (first T-1 frames only)
    pred_tokens_tf = model.dynamics_predictor(
        tokens_gt_masked_input,
        action_codes,
        world_emb,
        B,
        T - 1  # Pass T-1 as temporal dimension
    )

    # Detokenize to get predicted frames
    pred_frames_tf = model.detokenizer(pred_tokens_tf)  # [B*(T-1), C, H, W]
    pred_frames_tf = pred_frames_tf.view(B, T - 1, C, H, W)  # [B, T-1, C, H, W]

    # Compute teacher forcing reconstruction loss
    # Per description.md: GT[0:T-1] in → PRED[1:T] out, compare with GT[1:T]
    # No slicing needed! Direct comparison: [B,C,D] predictions vs [B,C,D] ground truth
    tf_loss = F.mse_loss(pred_frames_tf, h_sequence[:, 1:])

    # Collect RVQ losses
    rvq_losses = {
        'action_commit_loss': action_commit_loss,
        'action_codebook_loss': action_codebook_loss,
        'world_commit_loss': world_commit_loss,
        'world_codebook_loss': world_codebook_loss,
    }

    # =========================================================================
    # Step 2: Rollout Simulation (only predictor uses updated context)
    # =========================================================================
    # Action codes and world embedding are FIXED from GT (already extracted above)
    rollout_losses = []

    # Start with ground truth as initial context
    context = h_sequence.clone()

    for step in range(rollout_steps):
        # Tokenize current context (which contains predicted frames from previous step)
        context_tokens, _, _ = model.tokenizer(context)  # [B*T, d_model, out_h, out_w]

        # Apply masking to context tokens before passing to dynamics predictor
        if mask_prob > 0:
            context_tokens_masked = apply_token_masking(context_tokens, B, T, mask_prob)
        else:
            context_tokens_masked = context_tokens

        # Per description.md: Drop last frame, pass first T-1 frames to predictor
        context_tokens_masked_input = context_tokens_masked[:B * (T - 1)]  # [B*(T-1), d_model, out_h, out_w]

        # Run dynamics predictor with FIXED action codes and world embedding
        pred_tokens = model.dynamics_predictor(
            context_tokens_masked_input,
            action_codes,  # Fixed from GT, shape [B, T-1, d_code_a]
            world_emb,     # Fixed from GT
            B,
            T - 1  # Pass T-1 as temporal dimension
        )

        # Detokenize to get predicted frames
        pred_frames_rollout = model.detokenizer(pred_tokens)  # [B*(T-1), C, H, W]
        pred_frames_rollout = pred_frames_rollout.view(B, T - 1, C, H, W)  # [B, T-1, C, H, W]

        # Compute rollout loss for this step with redundancy masking
        # Per description.md lines 94-113: "apply a mask from 0 to k-1 (inclusive)"
        # For rollout step k, mask positions [0:k] to avoid counting redundant predictions
        # Only compute loss on NEW predictions at positions [k+1:]
        mask_end_idx = step + 1  # step=0 masks [0:1], step=1 masks [0:2], etc.

        if mask_end_idx < pred_frames_rollout.shape[1]:
            # Compute loss only on non-masked positions [mask_end_idx:]
            rollout_loss = F.mse_loss(
                pred_frames_rollout[:, mask_end_idx:],    # New predictions at [k+1:]
                h_sequence[:, mask_end_idx + 1:]           # Corresponding GT frames
            )
        else:
            # All positions are masked (shouldn't happen with normal rollout_steps)
            rollout_loss = torch.tensor(0.0, device=device, requires_grad=True)

        rollout_losses.append(rollout_loss)

        # Replace context frames [1:T] with predictions for next rollout step
        # Keep frame 0 as ground truth (initial condition)
        # pred_frames_rollout shape: [B, T-1, C, H, W]
        # context[:, 1:] shape: [B, T-1, C, H, W]
        # Perfect match! Direct replacement
        context = context.clone()
        context[:, 1:] = pred_frames_rollout.detach()  # Detach to avoid backprop through multiple rollout steps

    # =========================================================================
    # Step 3: Compute weighted total reconstruction loss
    # =========================================================================
    # Combine teacher forcing + rollout losses with weights
    total_recon_loss = weights[0] * tf_loss
    for i, rollout_loss in enumerate(rollout_losses):
        total_recon_loss += weights[i + 1] * rollout_loss

    return {
        'total_recon_loss': total_recon_loss,
        'tf_loss': tf_loss,
        'rollout_losses': rollout_losses,
        'rvq_losses': rvq_losses,
    }


def create_dataloaders(
    data_dir: str,
    manifest_path: str,
    batch_size: int,
    sequence_length: int,
    stride_train: int = 1,
    num_workers: int = 4,
    ddp: bool = False
):
    """
    Create training and validation dataloaders using the same mechanism as overfit.py.

    Args:
        data_dir: Path to directory containing VVAE HDF5 files
        manifest_path: Path to manifest JSON file
        batch_size: Batch size for training
        sequence_length: Number of frames in each sequence
        stride_train: Stride for sampling frames in training (default: 1)
        num_workers: Number of worker processes for data loading (default: 4)
        ddp: Whether to use DistributedDataParallel (default: False)

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader (optional, may be None)
    """
    logger.info("=" * 80)
    logger.info("STEP 1: DATA LOADING")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Manifest path: {manifest_path}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Sequence length: {sequence_length}")
    logger.info(f"Stride (train): {stride_train}")
    logger.info(f"Num workers: {num_workers}")
    logger.info(f"DDP enabled: {ddp}")

    # Create dataloaders using the same function as overfit.py
    train_loader, val_loader, test_loader = create_vvae_dataloaders(
        data_dir=data_dir,
        manifest_path=manifest_path,
        batch_size=batch_size,
        sequence_length=sequence_length,
        stride_train=stride_train,
        num_workers=num_workers,
        ddp=ddp
    )

    # Log dataset information
    logger.info(f"\nDataset loaded successfully:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader) if val_loader is not None else 0}")
    logger.info(f"  Test batches: {len(test_loader) if test_loader is not None else 0}")

    # Sample a batch to verify data format
    sample_batch = next(iter(train_loader))
    sample_sequence = sample_batch['sequence']
    logger.info(f"\nSample batch info:")
    logger.info(f"  Keys: {list(sample_batch.keys())}")
    logger.info(f"  Sequence shape: {sample_sequence.shape}")
    logger.info(f"  Expected format: [B={batch_size}, T={sequence_length}, C=16, H=64, W=64]")
    logger.info(f"  Value range: [{sample_sequence.min():.4f}, {sample_sequence.max():.4f}]")

    # Verify shape matches expected format
    B, T, C, H, W = sample_sequence.shape
    assert C == 16 and H == 64 and W == 64, \
        f"Unexpected data shape: expected [B, T, 16, 64, 64], got [B={B}, T={T}, C={C}, H={H}, W={W}]"

    logger.info("✓ Data loading complete!\n")

    return train_loader, val_loader, test_loader


def train(
    data_dir: str,
    manifest_path: str,
    batch_size: int = 8,
    sequence_length: int = 8,
    epochs: int = 100,
    lr: float = 1e-4,
    device: str = 'cuda',
    num_workers: int = 4,
    # Model hyperparameters
    d_model: int = 256,
    d_code_a: int = 128,
    d_code_h: int = 128,
    num_lvq_levels_a: int = 3,
    num_lvq_levels_h: int = 6,
    codebook_sizes_a: tuple = (12, 64, 256),
    codebook_sizes_h: tuple = (12, 24, 48, 256, 256, 256),
    num_encoder_blocks: int = 3,
    num_decoder_blocks: int = 3,
    # Training tricks (Step 4)
    use_frame_masking: bool = True,
    frame_mask_prob: float = 0.1,
    use_rollout: bool = True,
    rollout_steps: int = 4,
    rollout_weights: tuple = (1.0, 0.8, 0.5, 0.3, 0.1),
    # Wandb logging (Step 5)
    use_wandb: bool = True,
    wandb_project: str = "world_model",
    wandb_run_name: str = None,
    wandb_entity: str = None,
    # Validation frequency
    val_frequency: float = 0.1,
    val_size_percent: float = 1.0,
    # Codebook EMA decay
    codebook_ema_decay_init: float = 0.8,
    # Dead code reinitialization
    reinit_dead_codes_interval: int = 1000,
    dead_code_threshold: float = 0.01,
    # Positional embedding - random temporal PE offset for length extrapolation
    use_random_temporal_pe: bool = False,
    max_pe_offset: int = 120,
    # Ablation flags
    ablate_action: bool = False,
    ablate_world: bool = False,
    # Resume training
    resume_from: str = None,
):
    """
    Main training function for World Model.

    Args:
        data_dir: Path to directory containing VVAE HDF5 files
        manifest_path: Path to manifest JSON file
        batch_size: Batch size for training (default: 8)
        sequence_length: Number of frames in sequence (default: 8)
        epochs: Number of training epochs (default: 100)
        lr: Learning rate (default: 1e-4)
        device: Device to use ('cuda' or 'cpu')
        num_workers: Number of worker processes for data loading (default: 4)
        d_model: Feature dimension after tokenization (default: 256)
        d_code_a: Action code dimension (default: 128)
        d_code_h: World hypothesis dimension (default: 128)
        num_lvq_levels_a: Number of RVQ levels for action encoder (inferred from len(codebook_sizes_a))
        num_lvq_levels_h: Number of RVQ levels for world encoder (inferred from len(codebook_sizes_h))
        codebook_sizes_a: Tuple of codebook sizes per level for actions (default: (12, 64, 256))
        codebook_sizes_h: Tuple of codebook sizes per level for world (default: (12, 24, 48, 256, 256, 256))
        num_encoder_blocks: Number of ST-Transformer blocks for encoders (action & world) (default: 3)
        num_decoder_blocks: Number of ST-Transformer blocks for predictor/decoder (dynamics) (default: 3)
        use_frame_masking: Enable random frame masking (default: True)
        frame_mask_prob: Probability of masking a frame (default: 0.1)
        use_rollout: Enable rollout simulation (default: True)
        rollout_steps: Number of rollout steps (default: 4)
        rollout_weights: Weights for (TF, Roll1, Roll2, Roll3, Roll4) losses (default: (1.0, 0.8, 0.5, 0.3, 0.1))
        use_wandb: Enable wandb logging (default: True)
        wandb_project: Wandb project name (default: "world_model")
        wandb_run_name: Wandb run name (default: None, auto-generated)
        wandb_entity: Wandb entity/team name (default: None)
        val_frequency: Validation frequency as fraction of epoch (default: 0.1 = every 10%)
        val_size_percent: Fraction of validation set to use per validation (default: 1.0 = 100%)
        codebook_ema_decay_init: Initial EMA decay for codebook updates (default: 0.8, linearly increases to 0.99)
        reinit_dead_codes_interval: Steps between dead code reinitialization (default: 1000, 0=disabled)
        dead_code_threshold: Cluster size threshold for dead codes (default: 0.01)
        use_random_temporal_pe: Enable random temporal PE offset for length extrapolation (default: False)
        max_pe_offset: Maximum random offset for temporal PE when use_random_temporal_pe=True (default: 120)
        ablate_action: Ablation: Zero out all action codes (default: False)
        ablate_world: Ablation: Zero out world embedding (default: False)
        resume_from: Path to checkpoint file to resume training from (default: None)
    """
    logger.info("=" * 80)
    logger.info("WORLD MODEL TRAINING")
    logger.info("=" * 80)

    # =========================================================================
    # Handle config loading and CLI overrides if resuming
    # =========================================================================
    if resume_from is not None:
        logger.info("\n" + "=" * 80)
        logger.info("LOADING CONFIG FROM CHECKPOINT")
        logger.info("=" * 80)

        # Parse resume_from path (could be .pt file or directory)
        resume_path = Path(resume_from)
        if resume_path.is_file() and resume_path.suffix == '.pt':
            # User provided .pt file, use its directory
            checkpoint_dir_resume = resume_path.parent
            checkpoint_file = resume_path
        elif resume_path.is_dir():
            # User provided directory, find latest checkpoint
            checkpoint_dir_resume = resume_path
            # Look for best checkpoints first, then epoch checkpoints
            checkpoint_candidates = list(checkpoint_dir_resume.glob('best_*.pt')) + \
                                   sorted(checkpoint_dir_resume.glob('epoch_*.pt'),
                                         key=lambda x: int(x.stem.split('_')[1]), reverse=True)
            if not checkpoint_candidates:
                raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir_resume}")
            checkpoint_file = checkpoint_candidates[0]
            logger.info(f"Using checkpoint: {checkpoint_file}")
        else:
            raise ValueError(f"Invalid --resume_from path: {resume_from}")

        # Load training config
        config_path = checkpoint_dir_resume / 'training_config.json'
        if config_path.exists():
            loaded_config = load_training_config(str(config_path))

            # Track which parameters are being overridden
            overrides = []

            # Apply loaded config, but allow CLI overrides
            # We check each parameter and see if it differs from the loaded config
            original_values = {
                'batch_size': batch_size, 'sequence_length': sequence_length,
                'epochs': epochs, 'lr': lr, 'num_workers': num_workers,
                'd_model': d_model, 'd_code_a': d_code_a, 'd_code_h': d_code_h,
                'num_lvq_levels_a': num_lvq_levels_a, 'num_lvq_levels_h': num_lvq_levels_h,
                'codebook_sizes_a': codebook_sizes_a, 'codebook_sizes_h': codebook_sizes_h,
                'num_encoder_blocks': num_encoder_blocks, 'num_decoder_blocks': num_decoder_blocks,
                'use_frame_masking': use_frame_masking, 'frame_mask_prob': frame_mask_prob,
                'use_rollout': use_rollout, 'rollout_steps': rollout_steps,
                'rollout_weights': rollout_weights,
                'val_frequency': val_frequency, 'val_size_percent': val_size_percent,
                'codebook_ema_decay_init': codebook_ema_decay_init,
                'reinit_dead_codes_interval': reinit_dead_codes_interval,
                'dead_code_threshold': dead_code_threshold,
                'ablate_action': ablate_action, 'ablate_world': ablate_world,
            }

            # Override with loaded config, but keep CLI values if they differ from defaults
            for key, cli_value in original_values.items():
                if key in loaded_config:
                    loaded_value = loaded_config[key]
                    # Convert tuple back from list (JSON doesn't preserve tuples)
                    if isinstance(loaded_value, list) and isinstance(cli_value, tuple):
                        loaded_value = tuple(loaded_value)

                    # If CLI value differs from loaded config, it's an override
                    if cli_value != loaded_value:
                        overrides.append(f"{key}: {loaded_value} -> {cli_value}")
                    else:
                        # Use loaded config value
                        if key == 'batch_size': batch_size = loaded_value
                        elif key == 'sequence_length': sequence_length = loaded_value
                        elif key == 'epochs': epochs = loaded_value
                        elif key == 'lr': lr = loaded_value
                        elif key == 'num_workers': num_workers = loaded_value
                        elif key == 'd_model': d_model = loaded_value
                        elif key == 'd_code_a': d_code_a = loaded_value
                        elif key == 'd_code_h': d_code_h = loaded_value
                        # Note: num_lvq_levels_a and num_lvq_levels_h are ignored - always inferred from codebook_sizes
                        elif key == 'codebook_sizes_a':
                            codebook_sizes_a = loaded_value
                            num_lvq_levels_a = len(codebook_sizes_a)  # Recalculate from codebook sizes
                        elif key == 'codebook_sizes_h':
                            codebook_sizes_h = loaded_value
                            num_lvq_levels_h = len(codebook_sizes_h)  # Recalculate from codebook sizes
                        elif key == 'num_encoder_blocks': num_encoder_blocks = loaded_value
                        elif key == 'num_decoder_blocks': num_decoder_blocks = loaded_value
                        elif key == 'use_frame_masking': use_frame_masking = loaded_value
                        elif key == 'frame_mask_prob': frame_mask_prob = loaded_value
                        elif key == 'use_rollout': use_rollout = loaded_value
                        elif key == 'rollout_steps': rollout_steps = loaded_value
                        elif key == 'rollout_weights': rollout_weights = loaded_value
                        elif key == 'val_frequency': val_frequency = loaded_value
                        elif key == 'val_size_percent': val_size_percent = loaded_value
                        elif key == 'codebook_ema_decay_init': codebook_ema_decay_init = loaded_value
                        elif key == 'reinit_dead_codes_interval': reinit_dead_codes_interval = loaded_value
                        elif key == 'dead_code_threshold': dead_code_threshold = loaded_value
                        elif key == 'use_random_temporal_pe': use_random_temporal_pe = loaded_value
                        elif key == 'max_pe_offset': max_pe_offset = loaded_value
                        elif key == 'ablate_action': ablate_action = loaded_value
                        elif key == 'ablate_world': ablate_world = loaded_value

            if overrides:
                logger.info("\nCLI Overrides detected:")
                for override in overrides:
                    logger.info(f"  {override}")
            else:
                logger.info("\nNo CLI overrides - using all parameters from checkpoint config")

            # Store the checkpoint file path for later loading
            resume_from = str(checkpoint_file)
        else:
            logger.warning(f"Config file not found at {config_path}, using CLI parameters")

    # =========================================================================
    # STEP 5: Initialize Wandb
    # =========================================================================
    if use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("Wandb logging requested but wandb not available. Disabling wandb.")
            use_wandb = False
        else:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 5: WANDB INITIALIZATION")
            logger.info("=" * 80)

            # Create wandb config
            wandb_config = {
                # Training params
                'batch_size': batch_size,
                'sequence_length': sequence_length,
                'epochs': epochs,
                'lr': lr,
                'val_frequency': val_frequency,
                'val_size_percent': val_size_percent,
                # Model params
                'd_model': d_model,
                'd_code_a': d_code_a,
                'd_code_h': d_code_h,
                'num_lvq_levels_a': num_lvq_levels_a,
                'num_lvq_levels_h': num_lvq_levels_h,
                'codebook_sizes_a': codebook_sizes_a,
                'codebook_sizes_h': codebook_sizes_h,
                # Training tricks
                'use_frame_masking': use_frame_masking,
                'frame_mask_prob': frame_mask_prob,
                'use_rollout': use_rollout,
                'rollout_steps': rollout_steps,
                'rollout_weights': rollout_weights,
                # Positional embedding
                'use_random_temporal_pe': use_random_temporal_pe,
                'max_pe_offset': max_pe_offset,
            }

            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                entity=wandb_entity,
                config=wandb_config,
            )
            logger.info(f"Wandb initialized:")
            logger.info(f"  Project: {wandb_project}")
            logger.info(f"  Run name: {wandb_run_name or 'auto-generated'}")
            if wandb_entity:
                logger.info(f"  Entity: {wandb_entity}")

    # =========================================================================
    # STEP 1: Create dataloaders
    # =========================================================================
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        manifest_path=manifest_path,
        batch_size=batch_size,
        sequence_length=sequence_length,
        stride_train=1,
        num_workers=num_workers,
        ddp=False
    )

    # =========================================================================
    # STEP 2: Initialize model
    # =========================================================================
    logger.info("=" * 80)
    logger.info("STEP 2: MODEL INITIALIZATION")
    logger.info("=" * 80)

    model = WorldModel(
        d_model=d_model,
        d_code_a=d_code_a,
        d_code_h=d_code_h,
        num_lvq_levels_a=num_lvq_levels_a,
        num_lvq_levels_h=num_lvq_levels_h,
        codebook_sizes_a=codebook_sizes_a,
        codebook_sizes_h=codebook_sizes_h,
        num_encoder_blocks=num_encoder_blocks,
        num_decoder_blocks=num_decoder_blocks,
        decay=codebook_ema_decay_init,
        use_random_temporal_pe=use_random_temporal_pe,
        max_pe_offset=max_pe_offset,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized:")
    logger.info(f"  d_model: {d_model}")
    logger.info(f"  d_code_a: {d_code_a}, num_lvq_levels_a: {num_lvq_levels_a}, codebook_sizes_a: {codebook_sizes_a}")
    logger.info(f"  d_code_h: {d_code_h}, num_lvq_levels_h: {num_lvq_levels_h}, codebook_sizes_h: {codebook_sizes_h}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Device: {device}")

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logger.info(f"  Optimizer: Adam(lr={lr})")

    # =========================================================================
    # Load checkpoint if resuming
    # =========================================================================
    start_epoch = 0
    global_step = 0
    best_val_recon_mse = float('inf')
    best_val_total_loss = float('inf')

    if resume_from is not None:
        logger.info("\n" + "=" * 80)
        logger.info("RESUMING FROM CHECKPOINT")
        logger.info("=" * 80)
        logger.info(f"Loading checkpoint from: {resume_from}")

        # Load checkpoint (this reconstructs model and optimizer from saved state)
        model_loaded, optimizer_loaded, checkpoint_info = load_model(
            checkpoint_path=resume_from,
            device=device
        )

        # Override model and optimizer with loaded versions
        model = model_loaded
        optimizer = optimizer_loaded

        # Restore training state
        start_epoch = checkpoint_info['epoch'] + 1  # Continue from next epoch
        global_step = checkpoint_info['global_step']
        best_val_recon_mse = checkpoint_info['best_val_recon_mse']
        best_val_total_loss = checkpoint_info['best_val_total_loss']

        logger.info(f"Resumed training state:")
        logger.info(f"  Starting from epoch: {start_epoch}")
        logger.info(f"  Global step: {global_step}")
        logger.info(f"  Best val recon MSE: {best_val_recon_mse:.6f}")
        logger.info(f"  Best val total loss: {best_val_total_loss:.6f}")

        # Note: model_config from checkpoint is used by load_model() internally
        # The checkpoint's hyperparameters take precedence over CLI args
        logger.info("Note: Model architecture loaded from checkpoint (checkpoint config takes precedence)")

    # =========================================================================
    # STEP 3: Setup checkpoint directory
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: CHECKPOINT SETUP")
    logger.info("=" * 80)

    # Create checkpoint directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(f"world_model_{timestamp}")
    checkpoint_dir.mkdir(exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Store model config for checkpointing (used internally by load_model)
    model_config = {
        'd_model': d_model,
        'd_code_a': d_code_a,
        'd_code_h': d_code_h,
        'num_lvq_levels_a': num_lvq_levels_a,
        'num_lvq_levels_h': num_lvq_levels_h,
        'codebook_sizes_a': codebook_sizes_a,
        'codebook_sizes_h': codebook_sizes_h,
        'lr': lr,
        'use_random_temporal_pe': use_random_temporal_pe,
        'max_pe_offset': max_pe_offset,
    }

    # Save complete training configuration to JSON
    training_config = {
        # Data parameters
        'batch_size': batch_size,
        'sequence_length': sequence_length,
        'num_workers': num_workers,
        # Training parameters
        'epochs': epochs,
        'lr': lr,
        # Model architecture
        'd_model': d_model,
        'd_code_a': d_code_a,
        'd_code_h': d_code_h,
        'num_lvq_levels_a': num_lvq_levels_a,
        'num_lvq_levels_h': num_lvq_levels_h,
        'codebook_sizes_a': list(codebook_sizes_a),  # Convert tuple to list for JSON
        'codebook_sizes_h': list(codebook_sizes_h),
        'num_encoder_blocks': num_encoder_blocks,
        'num_decoder_blocks': num_decoder_blocks,
        # Training tricks
        'use_frame_masking': use_frame_masking,
        'frame_mask_prob': frame_mask_prob,
        'use_rollout': use_rollout,
        'rollout_steps': rollout_steps,
        'rollout_weights': list(rollout_weights),
        # Validation
        'val_frequency': val_frequency,
        'val_size_percent': val_size_percent,
        # Codebook EMA
        'codebook_ema_decay_init': codebook_ema_decay_init,
        # Dead code reinitialization
        'reinit_dead_codes_interval': reinit_dead_codes_interval,
        'dead_code_threshold': dead_code_threshold,
        # Ablation flags
        'ablate_action': ablate_action,
        'ablate_world': ablate_world,
    }

    # Save training config to checkpoint directory
    config_path = checkpoint_dir / 'training_config.json'
    save_training_config(str(config_path), training_config)
    logger.info(f"Training configuration saved to: {config_path}")

    # Note: best_val_recon_mse and best_val_total_loss are initialized above
    # (either from checkpoint or as float('inf'))
    logger.info(f"Tracking best validation metrics:")
    logger.info(f"  - Best reconstruction MSE (current: {best_val_recon_mse:.6f})")
    logger.info(f"  - Best total loss (current: {best_val_total_loss:.6f})")

    # =========================================================================
    # STEP 4: Training Tricks Configuration
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: TRAINING TRICKS")
    logger.info("=" * 80)
    logger.info(f"Frame masking: {'Enabled' if use_frame_masking else 'Disabled'}")
    if use_frame_masking:
        logger.info(f"  - Mask probability: {frame_mask_prob}")
    logger.info(f"Rollout simulation: {'Enabled' if use_rollout else 'Disabled'}")
    if use_rollout:
        logger.info(f"  - Rollout steps: {rollout_steps}")
        logger.info(f"  - Loss weights (TF, Roll1, Roll2, Roll3, Roll4): {rollout_weights}")

    logger.info(f"Ablation - Zero out actions: {'YES (action encoder not used)' if ablate_action else 'No'}")
    logger.info(f"Ablation - Zero out world: {'YES (world encoder not used)' if ablate_world else 'No'}")

    # =========================================================================
    # STEP 2: Training and Validation Loops
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: TRAINING & VALIDATION LOOPS")
    logger.info("=" * 80)
    logger.info("Loss components:")
    if use_rollout:
        logger.info("  - Reconstruction: Weighted(TF + Rollout1 + Rollout2 + Rollout3 + Rollout4)")
    else:
        logger.info("  - Reconstruction: MSE(pred_frames, ground_truth)")
    logger.info("  - Action RVQ: Commitment (for training)")
    logger.info("  - World RVQ: Commitment (for training)")
    logger.info("  - Codebook losses: Monitoring only (codebook updated via EMA, not gradients)")

    # Calculate validation points based on val_frequency
    total_batches = len(train_loader)
    num_validations = int(1.0 / val_frequency)
    val_points = [0]  # Always validate at the beginning (0%)
    val_points.extend([int(total_batches * val_frequency * i) for i in range(1, num_validations + 1)])
    # Ensure we validate at the end of epoch
    if total_batches not in val_points:
        val_points.append(total_batches)
    # Remove duplicates and sort
    val_points = sorted(list(set(val_points)))

    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING CONFIGURATION")
    logger.info(f"{'='*80}")
    logger.info(f"Training batches per epoch: {total_batches}")
    logger.info(f"Validation frequency: Every {val_frequency*100:.1f}% of epoch ({len(val_points)} times per epoch)")
    logger.info(f"Validation at batch indices: {val_points}")

    # Calculate validation set size
    total_val_batches = len(val_loader)
    num_val_samples = max(1, int(total_val_batches * val_size_percent))
    logger.info(f"Validation batches per run: {num_val_samples} ({val_size_percent*100:.1f}% of {total_val_batches} total validation batches)")
    logger.info(f"Progress prints: Every 100 training batches")
    if reinit_dead_codes_interval > 0:
        logger.info(f"Dead code reinitialization: Every {reinit_dead_codes_interval} steps (threshold: {dead_code_threshold})")
    else:
        logger.info(f"Dead code reinitialization: Disabled")

    # Training loop
    if resume_from is not None:
        logger.info(f"\nResuming training from epoch {start_epoch} to {epochs}...")
    else:
        logger.info(f"\nStarting training for {epochs} epochs...")
    logger.info("=" * 80)

    # Note: global_step is initialized above (either from checkpoint or as 0)

    for epoch in range(start_epoch, epochs):
        logger.info(f"\n{'='*80}")
        logger.info(f"EPOCH {epoch + 1}/{epochs}")
        logger.info(f"{'='*80}")

        # =====================================================================
        # Update EMA decay scheduler (linear from initial to 0.99)
        # =====================================================================
        progress = epoch / max(epochs - 1, 1)  # 0 to 1 over training
        current_decay = codebook_ema_decay_init + (0.99 - codebook_ema_decay_init) * progress
        model.update_codebook_ema_decay(current_decay)
        logger.info(f"Codebook EMA decay: {current_decay:.4f}")

        if use_wandb:
            wandb.log({'Training/codebook_ema_decay': current_decay}, step=global_step)

        # =====================================================================
        # Training phase
        # =====================================================================
        model.train()
        epoch_train_losses = []

        for batch_idx, batch in enumerate(train_loader):
            # Get data
            h_sequence = batch['sequence'].to(device)  # [B, T, 16, 64, 64]

            # Forward pass with training tricks
            optimizer.zero_grad()

            # Determine mask probability
            mask_prob_current = frame_mask_prob if use_frame_masking else 0.0

            if use_rollout:
                # Use rollout simulation with token masking
                loss_dict = compute_rollout_loss(
                    model=model,
                    h_sequence=h_sequence,
                    rollout_steps=rollout_steps,
                    weights=rollout_weights,
                    mask_prob=mask_prob_current,
                    ablate_action=ablate_action,
                    ablate_world=ablate_world,
                )
                recon_loss = loss_dict['total_recon_loss']
                tf_loss = loss_dict['tf_loss']
                rollout_losses = loss_dict['rollout_losses']
                rvq_losses = loss_dict['rvq_losses']
            else:
                # Simple teacher forcing with token masking
                B, T, C, H, W = h_sequence.shape

                # Tokenize
                tokens, _, _ = model.tokenizer(h_sequence)

                # Apply masking for world encoder and dynamics predictor
                if mask_prob_current > 0:
                    tokens_masked = apply_token_masking(tokens, B, T, mask_prob_current)
                else:
                    tokens_masked = tokens

                # Forward through components
                action_codes, action_indices, action_commit_loss, action_codebook_loss = \
                    model.action_encoder(tokens, B, T - 1)  # Unmasked tokens
                world_emb, world_indices, world_commit_loss, world_codebook_loss = \
                    model.world_encoder(tokens_masked, B, T)  # Masked tokens

                # Apply ablation: zero out action codes and/or world embedding if requested
                if ablate_action:
                    action_codes = torch.zeros_like(action_codes)
                if ablate_world:
                    world_emb = torch.zeros_like(world_emb)

                # Per description.md: Drop last frame, pass GT[0:T-1] to predictor
                tokens_masked_input = tokens_masked[:B * (T - 1)]  # [B*(T-1), d_model, out_h, out_w]

                pred_tokens = model.dynamics_predictor(tokens_masked_input, action_codes, world_emb, B, T - 1)

                # Detokenize
                pred_frames = model.detokenizer(pred_tokens).view(B, T - 1, C, H, W)

                # Compute loss (with proper shifting per description.md)
                # GT[0:T-1] in → PRED[1:T] out, compare with GT[1:T]
                # No slicing needed! Direct comparison
                recon_loss = F.mse_loss(pred_frames, h_sequence[:, 1:])
                tf_loss = recon_loss
                rollout_losses = []

                # Collect RVQ losses
                rvq_losses = {
                    'action_commit_loss': action_commit_loss,
                    'action_codebook_loss': action_codebook_loss,
                    'world_commit_loss': world_commit_loss,
                    'world_codebook_loss': world_codebook_loss,
                }

            # Total loss = reconstruction + RVQ commitment losses
            # Note: Codebook losses are for monitoring only (codebook updated via EMA, not gradients)
            total_loss = (
                recon_loss +
                rvq_losses['action_commit_loss'] +
                rvq_losses['world_commit_loss']
            )

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # =====================================================================
            # Dead code reinitialization (periodic maintenance)
            # =====================================================================
            if reinit_dead_codes_interval > 0 and global_step % reinit_dead_codes_interval == 0 and global_step > 0:
                logger.info(f"\n[Step {global_step}] Reinitializing dead codebook entries...")
                # Reinitialize for both action and world encoders
                action_reinit_stats = model.action_encoder.rvq.reinitialize_dead_codes(dead_threshold=dead_code_threshold)
                world_reinit_stats = model.world_encoder.rvq.reinitialize_dead_codes(dead_threshold=dead_code_threshold)

                # Log statistics
                action_total = sum(action_reinit_stats.values())
                world_total = sum(world_reinit_stats.values())
                logger.info(f"  Action encoder: {action_total} codes reinitialized {action_reinit_stats}")
                logger.info(f"  World encoder: {world_total} codes reinitialized {world_reinit_stats}")

                # Log to wandb if enabled
                if use_wandb:
                    reinit_log = {
                        'Codebook_Maintenance/action_codes_reinitialized': action_total,
                        'Codebook_Maintenance/world_codes_reinitialized': world_total,
                    }
                    # Add per-level stats
                    for level, count in action_reinit_stats.items():
                        reinit_log[f'Codebook_Maintenance/action_{level}_reinitialized'] = count
                    for level, count in world_reinit_stats.items():
                        reinit_log[f'Codebook_Maintenance/world_{level}_reinitialized'] = count
                    wandb.log(reinit_log, step=global_step)

            # Store losses for epoch average
            loss_record = {
                'total': total_loss.item(),
                'recon': recon_loss.item(),
                'tf_loss': tf_loss.item(),
                'action_commit': rvq_losses['action_commit_loss'].item(),
                'action_codebook': rvq_losses['action_codebook_loss'].item(),
                'world_commit': rvq_losses['world_commit_loss'].item(),
                'world_codebook': rvq_losses['world_codebook_loss'].item(),
            }
            # Add rollout losses if enabled
            for i, r_loss in enumerate(rollout_losses):
                loss_record[f'rollout_{i+1}'] = r_loss.item()

            epoch_train_losses.append(loss_record)

            # =====================================================================
            # Wandb logging (training)
            # =====================================================================
            if use_wandb:
                wandb_log = {
                    'Train_Total/total_loss': total_loss.item(),
                    'Train_Total/recon_loss': recon_loss.item(),
                    'Train_Dynamics_Predictor/tf_loss': tf_loss.item(),
                    'Train_Action_Encoder/commitment_loss': rvq_losses['action_commit_loss'].item(),
                    'Train_Action_Encoder/codebook_loss': rvq_losses['action_codebook_loss'].item(),
                    'Train_World_Encoder/commitment_loss': rvq_losses['world_commit_loss'].item(),
                    'Train_World_Encoder/codebook_loss': rvq_losses['world_codebook_loss'].item(),
                    'global_step': global_step,
                }
                # Add rollout losses if enabled
                for i, r_loss in enumerate(rollout_losses):
                    wandb_log[f'Train_Dynamics_Predictor/rollout_{i+1}'] = r_loss.item()

                wandb.log(wandb_log, step=global_step)

            global_step += 1

            # =====================================================================
            # Progress print every 100 batches
            # =====================================================================
            if (batch_idx + 1) % 100 == 0:
                progress_pct = ((batch_idx + 1) / total_batches) * 100
                logger.info(f"[Epoch {epoch + 1}/{epochs}] Batch {batch_idx + 1}/{total_batches} ({progress_pct:.1f}%) - "
                           f"Loss: {total_loss.item():.6f} (Recon: {recon_loss.item():.6f}, "
                           f"Action: {(rvq_losses['action_commit_loss'] + rvq_losses['action_codebook_loss']).item():.6f}, "
                           f"World: {(rvq_losses['world_commit_loss'] + rvq_losses['world_codebook_loss']).item():.6f})")

            # =====================================================================
            # Validation phase (at specified frequency, including 0%)
            # =====================================================================
            if batch_idx in val_points:
                progress = (batch_idx / total_batches) * 100 if batch_idx > 0 else 0
                logger.info(f"\n--- Validation at {progress:.0f}% of Epoch {epoch + 1} (batch {batch_idx}/{total_batches}) ---")

                model.eval()
                val_losses = []

                # Advanced metrics accumulators
                val_action_usage = []  # Codebook usage per batch for action encoder
                val_world_usage = []   # Codebook usage per batch for world encoder
                val_action_indices_all = []  # All action indices for global usage
                val_world_indices_all = []   # All world indices for global usage

                # Action sensitivity metrics
                val_dsnr_action_scores = []   # Action sensitivity dSNR
                val_psnr_seq_action_scores = []  # PSNR with correct actions
                val_psnr_rand_action_scores = []  # PSNR with random actions

                # World sensitivity metrics
                val_dsnr_world_scores = []   # World sensitivity dSNR
                val_psnr_seq_world_scores = []  # PSNR with correct world embedding
                val_psnr_rand_world_scores = []  # PSNR with random world embedding

                val_diagonal_attn = []  # Diagonal attention scores per block

                # Randomly sample validation batches
                total_val_batches = len(val_loader)
                num_val_samples = max(1, int(total_val_batches * val_size_percent))

                # Use itertools.islice to efficiently skip unsampled batches
                # This prevents loading all batches when we only need a few
                with torch.no_grad():
                    val_batch_count = 0
                    for val_batch_idx, val_batch in enumerate(val_loader):
                        # Early termination: stop after processing enough samples
                        if val_batch_count >= num_val_samples:
                            break

                        h_sequence_val = val_batch['sequence'].to(device)
                        val_batch_count += 1

                        # NOTE: No frame masking during validation (evaluate on clean data)

                        # Extract batch dimensions (needed for metrics later)
                        B_val, T_val, C_val, H_val, W_val = h_sequence_val.shape

                        # Forward pass (NO masking during validation)
                        if use_rollout:
                            # Use rollout simulation for validation too (no masking)
                            loss_dict_val = compute_rollout_loss(
                                model=model,
                                h_sequence=h_sequence_val,
                                rollout_steps=rollout_steps,
                                weights=rollout_weights,
                                mask_prob=0.0,  # No masking during validation
                                ablate_action=ablate_action,
                                ablate_world=ablate_world,
                            )
                            recon_loss_val = loss_dict_val['total_recon_loss']
                            tf_loss_val = loss_dict_val['tf_loss']
                            rollout_losses_val = loss_dict_val['rollout_losses']
                            rvq_losses_val = loss_dict_val['rvq_losses']
                        else:
                            # Simple teacher forcing (no masking)
                            # Tokenize
                            tokens_val, _, _ = model.tokenizer(h_sequence_val)

                            # No masking during validation - use unmasked tokens for all components
                            action_codes_val, _, action_commit_loss_val, action_codebook_loss_val = \
                                model.action_encoder(tokens_val, B_val, T_val - 1)
                            world_emb_val, _, world_commit_loss_val, world_codebook_loss_val = \
                                model.world_encoder(tokens_val, B_val, T_val)

                            # Apply ablation: zero out action codes and/or world embedding if requested
                            if ablate_action:
                                action_codes_val = torch.zeros_like(action_codes_val)
                            if ablate_world:
                                world_emb_val = torch.zeros_like(world_emb_val)

                            # Per description.md: Drop last frame, pass GT[0:T-1] to predictor
                            tokens_val_input = tokens_val[:B_val * (T_val - 1)]  # [B*(T-1), d_model, out_h, out_w]

                            pred_tokens_val = model.dynamics_predictor(tokens_val_input, action_codes_val, world_emb_val, B_val, T_val - 1)

                            # Detokenize
                            pred_frames_val = model.detokenizer(pred_tokens_val).view(B_val, T_val - 1, C_val, H_val, W_val)

                            # Compute loss (with proper shifting per description.md)
                            # GT[0:T-1] in → PRED[1:T] out, compare with GT[1:T]
                            # No slicing needed! Direct comparison
                            recon_loss_val = F.mse_loss(pred_frames_val, h_sequence_val[:, 1:])
                            tf_loss_val = recon_loss_val
                            rollout_losses_val = []

                            # Collect RVQ losses
                            rvq_losses_val = {
                                'action_commit_loss': action_commit_loss_val,
                                'action_codebook_loss': action_codebook_loss_val,
                                'world_commit_loss': world_commit_loss_val,
                                'world_codebook_loss': world_codebook_loss_val,
                            }

                        # Compute total loss (codebook losses for monitoring only)
                        total_loss_val = (
                            recon_loss_val +
                            rvq_losses_val['action_commit_loss'] +
                            rvq_losses_val['world_commit_loss']
                        )

                        # Record losses
                        loss_record_val = {
                            'total': total_loss_val.item(),
                            'recon': recon_loss_val.item(),
                            'tf_loss': tf_loss_val.item(),
                            'action_commit': rvq_losses_val['action_commit_loss'].item(),
                            'action_codebook': rvq_losses_val['action_codebook_loss'].item(),
                            'world_commit': rvq_losses_val['world_commit_loss'].item(),
                            'world_codebook': rvq_losses_val['world_codebook_loss'].item(),
                        }
                        # Add rollout losses if enabled
                        for i, r_loss in enumerate(rollout_losses_val):
                            loss_record_val[f'rollout_{i+1}'] = r_loss.item()

                        val_losses.append(loss_record_val)

                        # =====================================================================
                        # Compute advanced validation metrics (Step 6)
                        # =====================================================================
                        # Get indices for codebook usage metrics
                        # We need to run a full forward pass to get indices (no masking for validation)
                        if use_rollout:
                            # Run a simple forward pass to get indices (reuse tokenization)
                            B_idx, T_idx = B_val, T_val
                            tokens_idx, _, _ = model.tokenizer(h_sequence_val)

                            # Get action and world indices
                            _, action_indices, _, _ = model.action_encoder(tokens_idx, B_idx, T_idx - 1)
                            _, world_indices, _, _ = model.world_encoder(tokens_idx, B_idx, T_idx)
                        else:
                            # Already computed during forward pass - need to save them
                            # We need to extract indices from the forward pass above
                            # Re-run encoder forwards to get indices (they're cheap)
                            _, action_indices, _, _ = model.action_encoder(tokens_val, B_val, T_val - 1)
                            _, world_indices, _, _ = model.world_encoder(tokens_val, B_val, T_val)

                        # 1. Codebook usage for Action Encoder
                        action_usage = codebook_usage(action_indices)  # [num_lvq_levels]
                        val_action_usage.append(action_usage)
                        val_action_indices_all.append(action_indices.cpu())  # Store for global usage

                        # 2. Codebook usage for World Encoder
                        world_usage = codebook_usage(world_indices.unsqueeze(1))  # Add T dim, then [num_lvq_levels]
                        val_world_usage.append(world_usage)
                        val_world_indices_all.append(world_indices.cpu())  # Store for global usage

                        # 3. Action sensitivity dSNR
                        dsnr_action, psnr_seq_action, psnr_rand_action = action_sensitivity_dsnr(model, h_sequence_val)
                        val_dsnr_action_scores.append(dsnr_action)
                        val_psnr_seq_action_scores.append(psnr_seq_action)
                        val_psnr_rand_action_scores.append(psnr_rand_action)

                        # 3.5. World sensitivity dSNR
                        try:
                            dsnr_world, psnr_seq_world, psnr_rand_world = world_sensitivity_dsnr(model, h_sequence_val)
                            val_dsnr_world_scores.append(dsnr_world)
                            val_psnr_seq_world_scores.append(psnr_seq_world)
                            val_psnr_rand_world_scores.append(psnr_rand_world)
                        except Exception as e:
                            logger.warning(f"World sensitivity dSNR computation failed: {e}")

                        # 4. Diagonal attention score (Action Encoder only)
                        # Get attention weights from action encoder
                        tokens, _, _ = model.tokenizer(h_sequence_val)
                        _, _, _, _, attention_weights = model.action_encoder(
                            tokens, h_sequence_val.shape[0], h_sequence_val.shape[1] - 1, return_attention=True
                        )
                        # attention_weights is a list of lists: [num_blocks][num_heads]
                        # Each element is [B, t*out_h*out_w, t*out_h*out_w]
                        # Compute diagonal score for each block
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
                                # We need to reshape back to temporal structure
                                # attn_avg: [T*patches, T*patches]
                                # Assuming patches are ordered temporally first
                                B_val = h_sequence_val.shape[0]
                                T_val = h_sequence_val.shape[1]
                                num_patches = attn_avg.shape[0] // T_val

                                # Average over spatial patches to get temporal attention [T, T]
                                attn_temporal = attn_avg.view(T_val, num_patches, T_val, num_patches)
                                attn_temporal = attn_temporal.mean(dim=(1, 3))  # [T, T]

                                # Compute diagonal score
                                score = diagonal_attention_score(attn_temporal.unsqueeze(0))  # Add head dimension
                                block_diagonal_scores.append(score)

                        if len(block_diagonal_scores) > 0:
                            val_diagonal_attn.append(block_diagonal_scores)

                # =====================================================================
                # Autoregressive Rollout Evaluation (Step 6.5)
                # =====================================================================
                # Sample a small batch (4 sequences) for autoregressive generation test
                # This tests true autoregressive capability where predictions are used as context
                autoregressive_rollout_batch_size = 4
                autoregressive_rollout_length = 8  # Generate 8 frames (with 1 context frame = 9 total)

                # Initialize variables for wandb logging
                per_step_mse_autoregressive = None
                total_mse_autoregressive = None

                # Get a batch for rollout testing
                # We'll sample from the validation loader
                rollout_test_batch = None
                for val_rollout_batch in val_loader:
                    rollout_test_batch = val_rollout_batch
                    break  # Take first batch

                if rollout_test_batch is not None:
                    h_sequence_rollout = rollout_test_batch['sequence'].to(device)

                    # Only use first 4 sequences (or less if batch is smaller)
                    actual_batch_size = min(autoregressive_rollout_batch_size, h_sequence_rollout.shape[0])
                    h_sequence_rollout = h_sequence_rollout[:actual_batch_size]  # [4, T, 16, 64, 64]

                    # Ensure we have at least 9 frames (1 context + 8 generated)
                    if h_sequence_rollout.shape[1] >= 9:
                        B_rollout = h_sequence_rollout.shape[0]

                        # Extract initial context (first frame only)
                        context_rollout = h_sequence_rollout[:, 0:1, :, :, :]  # [B, 1, 16, 64, 64]

                        # Extract ground truth for comparison (frames 1-8, which we'll try to generate)
                        gt_rollout = h_sequence_rollout[:, 1:9, :, :, :]  # [B, 8, 16, 64, 64]

                        # Extract actions and world embedding from ground truth
                        # Tokenize full GT sequence to extract codes
                        # Note: .contiguous() needed because slicing can create non-contiguous tensors
                        tokens_rollout_full, _, _ = model.tokenizer(h_sequence_rollout[:, :9, :, :, :].contiguous())

                        # Extract action codes from GT (for frames 1-8, we need actions 0-7)
                        action_codes_rollout, _, _, _ = model.action_encoder(tokens_rollout_full, B_rollout, 8)  # [B, 8, d_code_a]

                        # Extract world embedding from GT
                        world_emb_rollout, _, _, _ = model.world_encoder(tokens_rollout_full, B_rollout, 9)  # [B, d_code_h]

                        # Apply ablation: zero out action codes and/or world embedding if requested
                        if ablate_action:
                            action_codes_rollout = torch.zeros_like(action_codes_rollout)
                        if ablate_world:
                            world_emb_rollout = torch.zeros_like(world_emb_rollout)

                        # Run autoregressive rollout
                        with torch.no_grad():
                            predictions_rollout = model.dynamics_predictor.autoregressive_rollout(
                                context=context_rollout,
                                action_sequence=action_codes_rollout,
                                world_emb=world_emb_rollout,
                                tokenizer=model.tokenizer,
                                detokenizer=model.detokenizer
                            )  # [B, 8, 16, 64, 64]

                        # Compute per-step MSE for each of the 8 generated frames
                        per_step_mse_autoregressive = []
                        for step in range(autoregressive_rollout_length):
                            step_mse = F.mse_loss(
                                predictions_rollout[:, step, :, :, :],
                                gt_rollout[:, step, :, :, :]
                            ).item()
                            per_step_mse_autoregressive.append(step_mse)

                        # Compute total MSE across all frames
                        total_mse_autoregressive = F.mse_loss(predictions_rollout, gt_rollout).item()

                        logger.info(f"  Autoregressive Rollout Test (4 sequences, 1 context + 8 generated):")
                        logger.info(f"    Total MSE: {total_mse_autoregressive:.6f}")
                        per_step_str = ", ".join([f"{mse:.6f}" for mse in per_step_mse_autoregressive])
                        logger.info(f"    Per-step MSE: [{per_step_str}]")

                # Log number of validation batches processed
                logger.info(f"Processed {len(val_losses)} validation batches (sampled from {total_val_batches} total)")

                # Compute validation averages
                val_avg = {
                    'total': sum(l['total'] for l in val_losses) / len(val_losses),
                    'recon': sum(l['recon'] for l in val_losses) / len(val_losses),
                    'tf_loss': sum(l['tf_loss'] for l in val_losses) / len(val_losses),
                    'action_commit': sum(l['action_commit'] for l in val_losses) / len(val_losses),
                    'action_codebook': sum(l['action_codebook'] for l in val_losses) / len(val_losses),
                    'world_commit': sum(l['world_commit'] for l in val_losses) / len(val_losses),
                    'world_codebook': sum(l['world_codebook'] for l in val_losses) / len(val_losses),
                }
                # Add rollout averages if enabled
                if use_rollout:
                    for i in range(rollout_steps):
                        val_avg[f'rollout_{i+1}'] = sum(l[f'rollout_{i+1}'] for l in val_losses) / len(val_losses)

                # =====================================================================
                # Compute advanced metrics averages
                # =====================================================================
                # 1. Per-batch average: Average codebook usage across validation batches
                # Stack: List[Tensor[num_levels]] -> Tensor[num_batches, num_levels]
                action_usage_stacked = torch.stack(val_action_usage)  # [num_batches, num_lvq_levels_a]
                world_usage_stacked = torch.stack(val_world_usage)    # [num_batches, num_lvq_levels_h]

                # Average across batches (per-batch metric)
                avg_action_usage_per_batch = action_usage_stacked.float().mean(dim=0)  # [num_lvq_levels_a]
                avg_world_usage_per_batch = world_usage_stacked.float().mean(dim=0)    # [num_lvq_levels_h]

                # 2. Overall/global: Unique codes across ALL validation batches
                # Concatenate all indices: List[Tensor[B, T-1, num_levels]] -> Tensor[total_samples, T-1, num_levels]
                action_indices_all = torch.cat(val_action_indices_all, dim=0)  # [total_samples, T-1, num_lvq_levels_a]
                world_indices_all = torch.cat(val_world_indices_all, dim=0)    # [total_samples, num_lvq_levels_h]

                # Compute global unique codes for each level
                global_action_usage = []
                for level_idx in range(num_lvq_levels_a):  # Dynamic: action encoder levels
                    indices_level = action_indices_all[:, :, level_idx].flatten()  # [total_samples * T-1]
                    unique_codes = torch.unique(indices_level).numel()
                    global_action_usage.append(unique_codes)
                global_action_usage = torch.tensor(global_action_usage)  # [num_lvq_levels_a]

                global_world_usage = []
                for level_idx in range(num_lvq_levels_h):  # Dynamic: world encoder levels
                    indices_level = world_indices_all[:, level_idx].flatten()  # [total_samples]
                    unique_codes = torch.unique(indices_level).numel()
                    global_world_usage.append(unique_codes)
                global_world_usage = torch.tensor(global_world_usage)  # [num_lvq_levels_h]

                # Average action sensitivity metrics
                avg_dsnr_action = sum(val_dsnr_action_scores) / len(val_dsnr_action_scores) if len(val_dsnr_action_scores) > 0 else 0.0
                avg_psnr_seq_action = sum(val_psnr_seq_action_scores) / len(val_psnr_seq_action_scores) if len(val_psnr_seq_action_scores) > 0 else 0.0
                avg_psnr_rand_action = sum(val_psnr_rand_action_scores) / len(val_psnr_rand_action_scores) if len(val_psnr_rand_action_scores) > 0 else 0.0

                # Average world sensitivity metrics
                if len(val_dsnr_world_scores) > 0:
                    avg_dsnr_world = sum(val_dsnr_world_scores) / len(val_dsnr_world_scores)
                    avg_psnr_seq_world = sum(val_psnr_seq_world_scores) / len(val_psnr_seq_world_scores)
                    avg_psnr_rand_world = sum(val_psnr_rand_world_scores) / len(val_psnr_rand_world_scores)
                else:
                    logger.warning("No world sensitivity scores collected - world dSNR/PSNR metrics will not be logged")
                    avg_dsnr_world = 0.0
                    avg_psnr_seq_world = 0.0
                    avg_psnr_rand_world = 0.0

                # Average diagonal attention scores per block
                # val_diagonal_attn: List[List[float]] - [num_batches][num_blocks]
                avg_diagonal_attn_per_block = []
                if len(val_diagonal_attn) > 0:
                    num_blocks = len(val_diagonal_attn[0])
                    for block_idx in range(num_blocks):
                        block_scores = [batch_scores[block_idx] for batch_scores in val_diagonal_attn if block_idx < len(batch_scores)]
                        if len(block_scores) > 0:
                            avg_diagonal_attn_per_block.append(sum(block_scores) / len(block_scores))

                # Log validation results
                log_msg = f"Val Loss - Total: {val_avg['total']:.6f}, Recon: {val_avg['recon']:.6f}"
                if use_rollout:
                    log_msg += f", TF: {val_avg['tf_loss']:.6f}"
                    for i in range(rollout_steps):
                        log_msg += f", Roll{i+1}: {val_avg[f'rollout_{i+1}']:.6f}"
                log_msg += (f", A_Commit: {val_avg['action_commit']:.6f}, A_Code: {val_avg['action_codebook']:.6f}, "
                           f"W_Commit: {val_avg['world_commit']:.6f}, W_Code: {val_avg['world_codebook']:.6f}")
                logger.info(log_msg)

                # Log advanced metrics (convert to percentages)
                action_usage_per_batch_pct = [avg_action_usage_per_batch[i] / codebook_sizes_a[i] * 100 for i in range(num_lvq_levels_a)]
                world_usage_per_batch_pct = [avg_world_usage_per_batch[i] / codebook_sizes_h[i] * 100 for i in range(num_lvq_levels_h)]
                action_usage_overall_pct = [global_action_usage[i] / codebook_sizes_a[i] * 100 for i in range(num_lvq_levels_a)]
                world_usage_overall_pct = [global_world_usage[i] / codebook_sizes_h[i] * 100 for i in range(num_lvq_levels_h)]

                logger.info(f"Val Advanced Metrics:")
                # Dynamic logging based on number of RVQ levels
                action_per_batch_str = "/".join([f"{action_usage_per_batch_pct[i]:.1f}%" for i in range(num_lvq_levels_a)])
                action_overall_str = "/".join([f"{action_usage_overall_pct[i]:.1f}%" for i in range(num_lvq_levels_a)])
                world_per_batch_str = "/".join([f"{world_usage_per_batch_pct[i]:.1f}%" for i in range(num_lvq_levels_h)])
                world_overall_str = "/".join([f"{world_usage_overall_pct[i]:.1f}%" for i in range(num_lvq_levels_h)])

                action_levels_str = "/".join([f"L{i+1}" for i in range(num_lvq_levels_a)])
                world_levels_str = "/".join([f"L{i+1}" for i in range(num_lvq_levels_h)])

                logger.info(f"  Action Codebook Usage Per-Batch ({action_levels_str}): {action_per_batch_str}")
                logger.info(f"  Action Codebook Usage Overall ({action_levels_str}): {action_overall_str}")
                logger.info(f"  World Codebook Usage Per-Batch ({world_levels_str}): {world_per_batch_str}")
                logger.info(f"  World Codebook Usage Overall ({world_levels_str}): {world_overall_str}")
                logger.info(f"  Action Sensitivity - dPSNR: {avg_dsnr_action:.4f}, PSNR_seq: {avg_psnr_seq_action:.4f}, PSNR_rand: {avg_psnr_rand_action:.4f}")
                logger.info(f"  World Sensitivity - dPSNR: {avg_dsnr_world:.4f}, PSNR_seq: {avg_psnr_seq_world:.4f}, PSNR_rand: {avg_psnr_rand_world:.4f}")
                if len(avg_diagonal_attn_per_block) > 0:
                    diagonal_str = "/".join([f"{score:.4f}" for score in avg_diagonal_attn_per_block])
                    logger.info(f"  Diagonal Attention (Block 1/2/3): {diagonal_str}")

                # =====================================================================
                # Wandb logging (validation)
                # =====================================================================
                if use_wandb:
                    wandb_val_log = {
                        'Val_Total/total_loss': val_avg['total'],
                        'Val_Total/recon_loss': val_avg['recon'],
                        'Val_Dynamics_Predictor/tf_loss': val_avg['tf_loss'],
                        'Val_Action_Encoder/commitment_loss': val_avg['action_commit'],
                        'Val_Action_Encoder/codebook_loss': val_avg['action_codebook'],
                        'Val_World_Encoder/commitment_loss': val_avg['world_commit'],
                        'Val_World_Encoder/codebook_loss': val_avg['world_codebook'],
                        'epoch': epoch + (batch_idx + 1) / total_batches,  # Fractional epoch
                    }
                    # Add rollout losses if enabled
                    if use_rollout:
                        for i in range(rollout_steps):
                            wandb_val_log[f'Val_Dynamics_Predictor/rollout_{i+1}'] = val_avg[f'rollout_{i+1}']

                    # Add advanced metrics (Step 6) - convert to percentages
                    # Dynamic logging for all RVQ levels
                    advanced_metrics = {}

                    # Per-batch metrics (average unique codes per batch)
                    for i in range(num_lvq_levels_a):
                        advanced_metrics[f'Val_Action_Encoder/codebook_usage_per_batch_level{i+1}_pct'] = action_usage_per_batch_pct[i]
                    for i in range(num_lvq_levels_h):
                        advanced_metrics[f'Val_World_Encoder/codebook_usage_per_batch_level{i+1}_pct'] = world_usage_per_batch_pct[i]

                    # Overall metrics (unique codes across all validation batches)
                    for i in range(num_lvq_levels_a):
                        advanced_metrics[f'Val_Action_Encoder/codebook_usage_overall_level{i+1}_pct'] = action_usage_overall_pct[i]
                    for i in range(num_lvq_levels_h):
                        advanced_metrics[f'Val_World_Encoder/codebook_usage_overall_level{i+1}_pct'] = world_usage_overall_pct[i]

                    # Action sensitivity metrics - logged under Val_Action_Encoder/
                    advanced_metrics.update({
                        'Val_Action_Encoder/dsnr': avg_dsnr_action,
                        'Val_Action_Encoder/psnr_seq': avg_psnr_seq_action,
                        'Val_Action_Encoder/psnr_rand': avg_psnr_rand_action,
                    })

                    # World sensitivity metrics - logged under Val_World_Encoder/
                    # Only log if we actually computed these metrics (avoid logging zeros)
                    if len(val_dsnr_world_scores) > 0:
                        advanced_metrics.update({
                            'Val_World_Encoder/dsnr': avg_dsnr_world,
                            'Val_World_Encoder/psnr_seq': avg_psnr_seq_world,
                            'Val_World_Encoder/psnr_rand': avg_psnr_rand_world,
                        })

                    wandb_val_log.update(advanced_metrics)

                    # Add diagonal attention scores per block
                    for block_idx, score in enumerate(avg_diagonal_attn_per_block):
                        wandb_val_log[f'Val_Action_Encoder/diagonal_attention_block{block_idx+1}'] = score

                    # Add autoregressive rollout metrics (Step 6.5)
                    if total_mse_autoregressive is not None:
                        # Log total MSE
                        wandb_val_log['Val_Dynamics_Predictor/autoregressive_total_mse'] = total_mse_autoregressive

                        # Log per-step MSE (creates a nice line plot in wandb with all 8 steps)
                        if per_step_mse_autoregressive is not None:
                            for step_idx, step_mse in enumerate(per_step_mse_autoregressive):
                                wandb_val_log[f'Val_Dynamics_Predictor/autoregressive_step{step_idx+1}_mse'] = step_mse

                    wandb.log(wandb_val_log, step=global_step)

                # =====================================================================
                # Save checkpoints if new best
                # =====================================================================
                # Check if we have a new best reconstruction MSE
                if val_avg['recon'] < best_val_recon_mse:
                    best_val_recon_mse = val_avg['recon']
                    checkpoint_path = checkpoint_dir / "best_recon_mse.pt"
                    logger.info(f"  New best reconstruction MSE: {best_val_recon_mse:.6f}")
                    save_model(
                        checkpoint_path=str(checkpoint_path),
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_recon_mse=best_val_recon_mse,
                        best_val_total_loss=best_val_total_loss,
                        model_config=model_config,
                    )

                # Check if we have a new best total loss
                if val_avg['total'] < best_val_total_loss:
                    best_val_total_loss = val_avg['total']
                    checkpoint_path = checkpoint_dir / "best_total_loss.pt"
                    logger.info(f"  New best total loss: {best_val_total_loss:.6f}")
                    save_model(
                        checkpoint_path=str(checkpoint_path),
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_recon_mse=best_val_recon_mse,
                        best_val_total_loss=best_val_total_loss,
                        model_config=model_config,
                    )

                model.train()  # Back to training mode

        # =====================================================================
        # Epoch summary
        # =====================================================================
        train_avg = {
            'total': sum(l['total'] for l in epoch_train_losses) / len(epoch_train_losses),
            'recon': sum(l['recon'] for l in epoch_train_losses) / len(epoch_train_losses),
            'tf_loss': sum(l['tf_loss'] for l in epoch_train_losses) / len(epoch_train_losses),
            'action_commit': sum(l['action_commit'] for l in epoch_train_losses) / len(epoch_train_losses),
            'action_codebook': sum(l['action_codebook'] for l in epoch_train_losses) / len(epoch_train_losses),
            'world_commit': sum(l['world_commit'] for l in epoch_train_losses) / len(epoch_train_losses),
            'world_codebook': sum(l['world_codebook'] for l in epoch_train_losses) / len(epoch_train_losses),
        }
        # Add rollout averages if enabled
        if use_rollout:
            for i in range(rollout_steps):
                train_avg[f'rollout_{i+1}'] = sum(l[f'rollout_{i+1}'] for l in epoch_train_losses) / len(epoch_train_losses)

        logger.info(f"\nEpoch {epoch + 1} Summary:")
        log_msg_train = f"Train Loss - Total: {train_avg['total']:.6f}, Recon: {train_avg['recon']:.6f}"
        if use_rollout:
            log_msg_train += f", TF: {train_avg['tf_loss']:.6f}"
            for i in range(rollout_steps):
                log_msg_train += f", Roll{i+1}: {train_avg[f'rollout_{i+1}']:.6f}"
        log_msg_train += (f", A_Commit: {train_avg['action_commit']:.6f}, A_Code: {train_avg['action_codebook']:.6f}, "
                         f"W_Commit: {train_avg['world_commit']:.6f}, W_Code: {train_avg['world_codebook']:.6f}")
        logger.info(log_msg_train)

        # =====================================================================
        # Save epoch checkpoint
        # =====================================================================
        checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
        logger.info(f"\nSaving epoch checkpoint:")
        save_model(
            checkpoint_path=str(checkpoint_path),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            best_val_recon_mse=best_val_recon_mse,
            best_val_total_loss=best_val_total_loss,
            model_config=model_config,
        )

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info("Steps completed:")
    logger.info("  ✓ Step 1: Data loading")
    logger.info("  ✓ Step 2: Training & Validation loops")
    logger.info("  ✓ Step 3: Weight saving mechanism")
    logger.info("  ✓ Step 4: Training tricks (frame masking + rollout simulation)")
    logger.info("  ✓ Step 5: Wandb logging (basic scalar metrics)")
    logger.info("  ✓ Step 6: Advanced validation metrics (codebook usage + dSNR + diagonal attention)")
    logger.info(f"\nCheckpoints saved in: {checkpoint_dir}")
    logger.info(f"  - best_recon_mse.pt (Val Recon MSE: {best_val_recon_mse:.6f})")
    logger.info(f"  - best_total_loss.pt (Val Total Loss: {best_val_total_loss:.6f})")
    logger.info(f"  - epoch_*.pt (epoch checkpoints)")
    logger.info("=" * 80)

    # Finish wandb run
    if use_wandb:
        wandb.finish()
        logger.info("Wandb run finished.")


def main():
    parser = argparse.ArgumentParser(description='World Model Training')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing VVAE HDF5 files')
    parser.add_argument('--manifest_path', type=str, required=True,
                        help='Path to manifest JSON file')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--sequence_length', type=int, default=8,
                        help='Number of frames in sequence (default: 8)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading (default: 4)')

    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=256,
                        help='Feature dimension after tokenization (default: 256)')
    parser.add_argument('--d_code_a', type=int, default=128,
                        help='Action code dimension (default: 128)')
    parser.add_argument('--d_code_h', type=int, default=128,
                        help='World hypothesis dimension (default: 128)')
    parser.add_argument('--codebook_sizes_a', type=int, nargs='+', default=[12, 64, 256],
                        help='Codebook sizes per RVQ level for action encoder. Number of values determines number of RVQ levels. '
                             '(default: 12 64 256 = 3 levels). Example: --codebook_sizes_a 8 64 64 64')
    parser.add_argument('--codebook_sizes_h', type=int, nargs='+', default=[12, 24, 48, 256, 256, 256],
                        help='Codebook sizes per RVQ level for world encoder. Number of values determines number of RVQ levels. '
                             '(default: 12 24 48 256 256 256 = 6 levels). Example: --codebook_sizes_h 256 256 256 256')
    parser.add_argument('--num_encoder_blocks', type=int, default=3,
                        help='Number of ST-Transformer blocks for encoders (action & world) (default: 3)')
    parser.add_argument('--num_decoder_blocks', type=int, default=3,
                        help='Number of ST-Transformer blocks for dynamics predictor (default: 3)')

    # Training tricks (Step 4)
    parser.add_argument('--no_frame_masking', action='store_true',
                        help='Disable random frame masking (default: enabled)')
    parser.add_argument('--frame_mask_prob', type=float, default=0.1,
                        help='Probability of masking a frame (default: 0.1)')
    parser.add_argument('--no_rollout', action='store_true',
                        help='Disable rollout simulation (default: enabled)')
    parser.add_argument('--rollout_steps', type=int, default=4,
                        help='Number of rollout steps (default: 4)')

    # Wandb logging (Step 5)
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging (default: enabled)')
    parser.add_argument('--wandb_project', type=str, default='world_model',
                        help='Wandb project name (default: world_model)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity/team name (default: None)')

    # Validation frequency
    parser.add_argument('--val_frequency', type=float, default=0.1,
                        help='Validation frequency as fraction of epoch (default: 0.1 = every 10%%)')
    parser.add_argument('--val_size_percent', type=float, default=1.0,
                        help='Fraction of validation set to use per validation run (default: 1.0 = 100%%)')

    # Codebook EMA decay scheduler
    parser.add_argument('--codebook_ema_decay', type=float, default=0.8,
                        help='Initial EMA decay for codebook updates (default: 0.8, linearly increases to 0.99)')

    # Dead code reinitialization
    parser.add_argument('--reinit_dead_codes_interval', type=int, default=1000,
                        help='Steps between dead code reinitialization (default: 1000, 0=disabled)')
    parser.add_argument('--dead_code_threshold', type=float, default=0.01,
                        help='Cluster size threshold for identifying dead codes (default: 0.01)')

    # Positional embedding - random temporal PE offset for length extrapolation
    parser.add_argument('--use_random_temporal_pe', action='store_true',
                        help='Enable random temporal PE offset during training for length extrapolation (default: disabled)')
    parser.add_argument('--max_pe_offset', type=int, default=120,
                        help='Maximum random offset for temporal PE when use_random_temporal_pe is enabled (default: 120)')

    # Ablation study flags
    parser.add_argument('--no_action', action='store_true',
                        help='Ablation: Zero out all action codes (measure action contribution)')
    parser.add_argument('--no_world', action='store_true',
                        help='Ablation: Zero out world embedding (measure world encoder contribution)')

    # Resume training
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint directory or .pt file to resume from. '
                             'Loads training_config.json from checkpoint dir. '
                             'CLI args override loaded config. (default: None)')

    args = parser.parse_args()

    # Convert codebook sizes from list to tuple and infer number of RVQ levels
    codebook_sizes_a = tuple(args.codebook_sizes_a)
    codebook_sizes_h = tuple(args.codebook_sizes_h)

    # Infer number of RVQ levels from codebook sizes
    num_lvq_levels_a = len(codebook_sizes_a)
    num_lvq_levels_h = len(codebook_sizes_h)

    print(f"Inferred RVQ levels: action={num_lvq_levels_a}, world={num_lvq_levels_h}")

    # Run training
    train(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
        d_model=args.d_model,
        d_code_a=args.d_code_a,
        d_code_h=args.d_code_h,
        num_lvq_levels_a=num_lvq_levels_a,
        num_lvq_levels_h=num_lvq_levels_h,
        codebook_sizes_a=codebook_sizes_a,
        codebook_sizes_h=codebook_sizes_h,
        num_encoder_blocks=args.num_encoder_blocks,
        num_decoder_blocks=args.num_decoder_blocks,
        use_frame_masking=not args.no_frame_masking,
        frame_mask_prob=args.frame_mask_prob,
        use_rollout=not args.no_rollout,
        rollout_steps=args.rollout_steps,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        val_frequency=args.val_frequency,
        val_size_percent=args.val_size_percent,
        codebook_ema_decay_init=args.codebook_ema_decay,
        reinit_dead_codes_interval=args.reinit_dead_codes_interval,
        dead_code_threshold=args.dead_code_threshold,
        use_random_temporal_pe=args.use_random_temporal_pe,
        max_pe_offset=args.max_pe_offset,
        ablate_action=args.no_action,
        ablate_world=args.no_world,
        resume_from=args.resume_from,
    )


if __name__ == '__main__':
    main()
