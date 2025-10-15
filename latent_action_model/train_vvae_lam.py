import os
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
# Removed TensorBoard - using only WandB
import numpy as np
from tqdm import tqdm
import wandb

# Try to import FLOP counting libraries (install with: pip install ptflops fvcore)
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False

try:
    from fvcore.nn import FlopCountMode, flop_count
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

from latent_action_model.vqvae import VVAELatentActionVQVAE
from latent_action_model.dataloader_vvae import create_vvae_dataloaders


logger = logging.getLogger(__name__)


def _ddp_mean(x: float, device) -> float:
    """Average a scalar across all DDP ranks."""
    if not dist.is_initialized():
        return x
    t = torch.tensor([x], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def calculate_model_flops(model, input_shape=(8, 16, 64, 64), device="cuda"):
    """
    Calculate FLOPs for the VVAE LAM model.

    Args:
        model: VVAE LAM model
        input_shape: Input tensor shape (T, C, H, W) for VVAE sequence
        device: Device to run calculation on

    Returns:
        forward_flops: FLOPs for forward pass
        backward_flops: Approximate FLOPs for backward pass (3x forward)
    """
    model.eval()

    # Create dummy input batch
    T, C, H, W = input_shape
    B = 1  # Use batch size 1 for FLOP calculation
    dummy_input = torch.randn(B, T, C, H, W).to(device)
    
    forward_flops = 0
    
    if FVCORE_AVAILABLE:
        try:
            # Use fvcore for more accurate counting
            with flop_count(model, inputs=(dummy_input,), mode=FlopCountMode.TABLE) as fc:
                _ = model(dummy_input)
            forward_flops = fc.get_total_flops()
            logger.info(f"FLOPs calculated using fvcore: {forward_flops:,}")
        except Exception as e:
            logger.warning(f"fvcore FLOP counting failed: {e}")
            forward_flops = 0
    
    if forward_flops == 0 and PTFLOPS_AVAILABLE:
        try:
            # Fallback to ptflops
            # ptflops expects input shape without batch dimension
            input_size = (T, C, H, W)  # VVAE format
            macs, params = get_model_complexity_info(
                model,
                input_size,
                print_per_layer_stat=False,
                as_strings=False
            )
            forward_flops = 2 * macs  # MACs to FLOPs (multiply-add = 2 ops)
            logger.info(f"FLOPs calculated using ptflops: {forward_flops:,}")
        except Exception as e:
            logger.warning(f"ptflops FLOP counting failed: {e}")
            forward_flops = 0

    if forward_flops == 0:
        # Manual estimation as fallback
        logger.warning("Using manual FLOP estimation (less accurate)")

        # Rough estimation based on transformer architecture
        # For VVAE LAM: model.lam has encoder/decoder, not model directly
        try:
            embed_dim = model.lam.encoder.embed_dim
            encoder_blocks = len(model.lam.encoder.blocks)
            decoder_blocks = len(model.lam.decoder.blocks)
        except:
            # Fallback values if structure is different
            embed_dim = 512
            encoder_blocks = 3
            decoder_blocks = 3

        # Get action/code dimension (works for both VAE and VQ-VAE)
        action_dim = getattr(model, 'codebook_dim', 128) * 3

        # VVAE adapts C=16, H=64, W=64 -> N=256 patches of D=256 dim
        N = 256  # num_patches after adapter
        D = 256  # patch_dim after adapter
        seq_len = T

        # Adapter FLOPs (CNN operations)
        # vvae_to_lam: Conv2d 16->256, output size 16x16
        adapter_flops = T * 16 * 256 * 16 * 16 * 9  # Conv with 3x3 kernel (approx)

        # Encoder FLOPs (approximate)
        encoder_flops = 0
        encoder_flops += T * N * D * embed_dim  # Patch projection
        encoder_flops += encoder_blocks * (T * N) * embed_dim * embed_dim * 4  # Attention + MLP
        encoder_flops += embed_dim * action_dim * 2  # Output heads

        # Decoder FLOPs (approximate)
        decoder_flops = 0
        decoder_flops += action_dim * embed_dim  # Action projection
        decoder_flops += decoder_blocks * N * embed_dim * embed_dim * 4  # Cross-attention + MLP
        decoder_flops += N * embed_dim * D  # Output projection

        # Reverse adapter FLOPs (CNN operations)
        reverse_adapter_flops = (T-1) * 256 * 16 * 16 * 16 * 9  # ConvTranspose (approx)

        # Loss computation (approximate) - MSE on VVAE latents
        loss_flops = (T-1) * C * H * W * 2  # MSE loss + VQ loss

        forward_flops = adapter_flops + encoder_flops + decoder_flops + reverse_adapter_flops + loss_flops
        logger.info(f"FLOPs estimated manually: {forward_flops:,}")
    
    # Backward pass is approximately 2x forward pass
    backward_flops = 2 * forward_flops
    total_flops = forward_flops + backward_flops
    
    return forward_flops, backward_flops, total_flops


class VVAELAMTrainer:
    """
    Trainer for VVAE Latent Action Model (VQ-VAE variant).
    """

    def __init__(
        self,
        model: VVAELatentActionVQVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        kl_annealing_steps: int = 10000,
        kl_min_weight: float = 0.0,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        save_every: int = 5,
        eval_every: int = 1,
        mixed_precision: bool = True,
        use_wandb: bool = True,
        wandb_project: str = "lam-vae",
        wandb_run_name: Optional[str] = None,
        test_mode: bool = False,
        use_scheduler: bool = False,
        min_lr_ratio: float = 0.1,
        rollout_horizon: int = 2,
        rollout_weight: float = 1.0,
        rollout_prob: float = 1.0,
        detach_rollout_first: bool = True,
        anchor_strategy: str = "random",
        validation_fraction: float = 1.0
    ):
        """
        Args:
            model: LAM VAE model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            test_loader: Optional test dataloader
            device: Device to train on
            learning_rate: Base learning rate
            warmup_steps: Number of warmup steps for learning rate
            max_grad_norm: Maximum gradient norm for clipping
            kl_annealing_steps: Steps for KL weight annealing
            kl_min_weight: Minimum KL weight during annealing
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
            save_every: Save checkpoint every N epochs
            eval_every: Evaluate every N epochs
            mixed_precision: Use mixed precision training
            use_scheduler: Enable manual LR scheduling with warmup and cosine annealing
            min_lr_ratio: Minimum LR as ratio of base LR (e.g., 0.1 = 10% of base)
            rollout_horizon: Number of steps for rollout (1 = no rollout; 2-3 typical)
            rollout_weight: Weight for rollout loss component
            rollout_prob: Probability of computing rollout loss (0-1)
            detach_rollout_first: Whether to detach first predicted state before rollout
            anchor_strategy: Rollout anchor strategy ("random" or "last")
            validation_fraction: Fraction of validation set to evaluate (0-1, default 1.0)
        """
        self.device = device
        self.validation_fraction = validation_fraction
        self.use_ddp = dist.is_initialized()
        
        if self.use_ddp:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.model = DDP(model.to(device), device_ids=[local_rank])
        else:
            self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Optimizer and scheduler
        model_for_optim = self.model.module if self.use_ddp else self.model
        self.optimizer = optim.AdamW(
            model_for_optim.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Training settings
        self.base_lr = learning_rate  # Store base LR to fix compounding decay bug
        self.use_scheduler = use_scheduler
        self.min_lr_ratio = min_lr_ratio
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.kl_annealing_steps = kl_annealing_steps
        self.kl_min_weight = kl_min_weight
        # VQ-VAE doesn't use kl_weight, set to 0
        self.kl_max_weight = 0.0

        # Rollout parameters
        self.rollout_horizon = rollout_horizon
        self.rollout_weight = rollout_weight
        self.rollout_prob = rollout_prob
        self.detach_rollout_first = detach_rollout_first
        self.anchor_strategy = anchor_strategy
        
        # Checkpointing
        # Create timestamped checkpoint directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Path(checkpoint_dir) / f"lam_{timestamp}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model config immediately
        self._save_model_config(model_for_optim)
        self.save_every = save_every
        self.eval_every = eval_every
        
        # Logging (WandB only)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision
        self.mixed_precision = mixed_precision
        if mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_recon_loss = float('inf')  # Track best reconstruction loss
        
        # Validation frequency (every half epoch for VQ-VAE)
        self.steps_per_epoch = len(train_loader)
        self.val_frequency = max(1, self.steps_per_epoch // 2)  # Validate 2 times per epoch (every half epoch)
        
        # Total training steps will be set when training starts
        self.total_training_steps = None
        
        # TFLOP tracking
        self.flops_per_step = None
        self.total_tflops = 0.0  # Cumulative TFLOPs consumed
        
        # WandB initialization
        self.test_mode = test_mode
        self.fixed_batch = None  # Will store the single batch in test mode
        
        self.use_wandb = use_wandb
        
        # In DDP mode, wandb should only be initialized on the main process (rank 0)
        is_main_process = not self.use_ddp or dist.get_rank() == 0
        
        if self.use_wandb and is_main_process:
            run_name = wandb_run_name or (f"lam_test_mode_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if test_mode else None)
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    **model_for_optim.config,  # Include all model config (works for both VAE and VQ-VAE)
                    'learning_rate': learning_rate,
                    'batch_size': train_loader.batch_size,
                    'warmup_steps': warmup_steps,
                    'kl_annealing_steps': kl_annealing_steps,
                    'mixed_precision': mixed_precision,
                    'test_mode': test_mode,
                    'rollout_horizon': rollout_horizon,
                    'rollout_weight': rollout_weight,
                    'rollout_prob': rollout_prob,
                    'detach_rollout_first': detach_rollout_first,
                    'anchor_strategy': anchor_strategy
                }
            )
            wandb.watch(model_for_optim, log='gradients', log_freq=500)
        elif self.use_wandb and not is_main_process:
            # Disable wandb on non-main processes to prevent duplicate runs
            self.use_wandb = False
        
        # In test mode, grab and store a single batch
        if self.test_mode:
            logger.info("=" * 60)
            logger.info("TEST MODE ACTIVATED: Using only 1 batch for overfitting test")
            logger.info("Test mode defaults:")
            logger.info("  - VQ loss only (no KL divergence)")
            logger.info("  - Dropout: 0.0 (disabled)")
            logger.info("  - Validation: Disabled")
            logger.info("=" * 60)
            
            # Override settings for test mode
            self.kl_weight = 0.0
            self.kl_min_weight = 0.0
            self.kl_max_weight = 0.0
            model_for_optim.kl_weight = 0.0
            
            # Disable dropout in the model
            self._disable_dropout()
            
            self.fixed_batch = next(iter(train_loader))
            # Log batch info
            logger.info(f"Fixed batch shapes:")
            for key, val in self.fixed_batch.items():
                logger.info(f"  {key}: {val.shape}")
        
    def _disable_dropout(self):
        """Disable all dropout in the model for test mode."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        logger.info("Disabled all dropout layers for test mode")
    
    def _save_model_config(self, model: VVAELatentActionVQVAE):
        """Save model configuration to JSON file."""
        config_path = self.checkpoint_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(model.config, f, indent=2)
        logger.info(f"Saved model config to {config_path}")
    
    def _initialize_flop_tracking(self):
        """Initialize FLOP tracking for compute efficiency analysis."""
        if self.flops_per_step is not None:
            return  # Already initialized
            
        logger.info("Initializing FLOP tracking...")
        
        # Get typical input shape from the dataloader
        sample_batch = next(iter(self.train_loader))
        sequence_shape = sample_batch['sequence'].shape  # [B, T, C, H, W]
        input_shape = sequence_shape[1:]  # (T, C, H, W) - remove batch dimension
        
        try:
            # In DDP, the model is on a specific device, get it from a parameter
            device = next(self.model.parameters()).device
            forward_flops, backward_flops, total_flops = calculate_model_flops(
                self.model.module if self.use_ddp else self.model, # unwrap for FLOPs
                input_shape=input_shape,
                device=device
            )
            
            # Scale by batch size (FLOPs scale linearly with batch size)
            batch_size = self.train_loader.batch_size
            self.flops_per_step = total_flops * batch_size
            
            # Convert to TFLOPs per step
            self.tflops_per_step = self.flops_per_step / 1e12
            
            logger.info(f"FLOP tracking initialized:")
            logger.info(f"  Forward pass: {forward_flops:,} FLOPs")
            logger.info(f"  Backward pass: {backward_flops:,} FLOPs") 
            logger.info(f"  Total per sample: {total_flops:,} FLOPs")
            logger.info(f"  Total per batch (B={batch_size}): {self.flops_per_step:,} FLOPs")
            logger.info(f"  TFLOPs per training step: {self.tflops_per_step:.6f}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FLOP tracking: {e}")
            self.flops_per_step = 0
            self.tflops_per_step = 0
    
    def get_lr(self) -> float:
        """Get current learning rate with optional warmup and cosine annealing."""
        # If scheduler is disabled, just return the base learning rate
        if not self.use_scheduler:
            return self.base_lr
        
        # Manual scheduling is enabled
        if self.global_step < self.warmup_steps:
            # Linear warmup from min_lr to base_lr
            warmup_progress = self.global_step / max(1, self.warmup_steps)
            return self.base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * warmup_progress)
        else:
            # Cosine annealing after warmup, with minimum LR
            # Use total training steps if available, otherwise use a default
            total_steps = self.total_training_steps if self.total_training_steps else 100_000
            progress = (self.global_step - self.warmup_steps) / max(1, (total_steps - self.warmup_steps))
            progress = min(1.0, progress)  # Clamp to [0, 1]
            
            # Cosine annealing from 1.0 to min_lr_ratio
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            lr_factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor
            
            return self.base_lr * lr_factor
    
    def get_kl_weight(self) -> float:
        """Get current KL weight with annealing."""
        # In test mode, always return 0
        if self.test_mode:
            return 0.0
        
        # Use total training steps if available, otherwise fall back to kl_annealing_steps
        total_steps = self.total_training_steps if self.total_training_steps is not None else self.kl_annealing_steps
        
        if self.global_step < total_steps:
            # Linear annealing from min to max over the entire training duration
            progress = self.global_step / total_steps
            return self.kl_min_weight + (self.kl_max_weight - self.kl_min_weight) * progress
        else:
            return self.kl_max_weight
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        # In DDP, self.device is the local rank's device, which is correct
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        # No layer normalization needed for VVAE latents
        
        # Get current KL weight
        kl_weight = self.get_kl_weight()
        
        # Forward pass with mixed precision
        if self.mixed_precision:
            with torch.amp.autocast('cuda'):
                loss_dict = (self.model.module if self.use_ddp else self.model).compute_loss(
                    batch,
                    beta_schedule=1.0,  # Not used in VQ-VAE
                    rollout_horizon=self.rollout_horizon,
                    rollout_weight=self.rollout_weight,
                    rollout_prob=self.rollout_prob,
                    detach_rollout_first=self.detach_rollout_first,
                    anchor_strategy=self.anchor_strategy
                )
                loss = loss_dict['loss']
        else:
            loss_dict = (self.model.module if self.use_ddp else self.model).compute_loss(
                batch,
                beta_schedule=1.0,  # Not used in VQ-VAE
                rollout_horizon=self.rollout_horizon,
                rollout_weight=self.rollout_weight,
                rollout_prob=self.rollout_prob,
                detach_rollout_first=self.detach_rollout_first,
                anchor_strategy=self.anchor_strategy
            )
            loss = loss_dict['loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # Update learning rate (only if scheduler is enabled)
        if self.use_scheduler:
            new_lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        
        self.global_step += 1
        
        # Update TFLOP tracking
        if hasattr(self, 'tflops_per_step') and self.tflops_per_step > 0:
            self.total_tflops += self.tflops_per_step
        
        # Return metrics with clear naming
        metrics = {
            'loss': loss.item(),                                        # Total loss (MSE-based, used for optimization)
            'loss_mae': loss_dict['loss_mae'].item(),                   # Total loss (MAE-based, monitoring)
            'recon_loss': loss_dict['recon_loss'].item(),               # Reconstruction MSE
            'mse_loss': loss_dict['mse_loss'].item(),                   # MSE (same as recon_loss)
            'mae_loss': loss_dict['mae_loss'].item(),                   # MAE (monitoring)
            'vq_loss': loss_dict['vq_loss'].item(),                     # VQ loss
            'codebook_loss': loss_dict['codebook_loss'].item(),         # Codebook loss component
            'commitment_loss': loss_dict['commitment_loss'].item(),     # Commitment loss component
            'rollout_loss': loss_dict['rollout_loss'].item(),           # Rollout MSE
            'rollout_loss_mae': loss_dict['rollout_loss_mae'].item(),   # Rollout MAE (monitoring)
            'kl_weight': kl_weight,
            'lr': self.get_lr(),
            'total_tflops': self.total_tflops,
            'codebook_usage': loss_dict.get('codebook_usage', 0),
            'indices': loss_dict.get('indices', None)
        }

        return metrics
    
    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single evaluation step with action sensitivity check."""
        self.model.eval()
        
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        # No layer normalization needed for VVAE latents
        
        # Forward pass for standard losses with rollout parameters
        loss_dict = (self.model.module if self.use_ddp else self.model).compute_loss(
            batch,
            rollout_horizon=self.rollout_horizon,
            rollout_weight=self.rollout_weight,
            rollout_prob=self.rollout_prob,
            detach_rollout_first=self.detach_rollout_first,
            anchor_strategy=self.anchor_strategy
        )
        
        # Action sensitivity check for VVAE format
        z_sequence = batch['sequence']  # [B, T, C=16, H=64, W=64]
        B, T, C, H, W = z_sequence.shape
        z_target = z_sequence[:, -1]  # [B, C, H, W] - target is the last frame

        # Get true action codes from encoder
        model_ref = self.model.module if self.use_ddp else self.model
        mu, logvar = model_ref.encode(z_sequence)  # [B, T-1, 3*codebook_dim]
        z_e = model_ref.lam.reparameterize(mu, logvar)  # [B, T-1, 3*codebook_dim]
        z_q, indices, _, _, _ = model_ref.lam.quantize(z_e)  # z_q: [B, T-1, 3, codebook_dim]

        # For decoding the final frame, we need all T-1 past frames (z_0 to z_{T-2})
        # and all T-1 code embeddings (for transitions 0->1, 1->2, ..., T-2->T-1)
        z_past = z_sequence[:, :-1]  # [B, T-1, C, H, W] - frames 0 to T-2
        codes_past = z_q  # [B, T-1, 3, codebook_dim]

        # Generate random code indices for comparison
        rand_indices = torch.randint(0, model_ref.lam.num_embeddings, (B, T-1, 3), device=z_sequence.device)
        codes_rand = model_ref.lam.codebook(rand_indices)  # [B, T-1, 3, codebook_dim]

        # Generate perturbed codes (flip one random code per sequence)
        codes_perturbed = codes_past.clone()
        for i in range(B):
            # Randomly select which code to flip (out of T-1 transitions * 3 codes each)
            flip_idx = torch.randint(0, (T-1) * 3, (1,)).item()
            t_idx = flip_idx // 3
            code_idx = flip_idx % 3
            new_code_idx = torch.randint(0, model_ref.lam.num_embeddings, (1,), device=z_sequence.device)
            codes_perturbed[i, t_idx, code_idx] = model_ref.lam.codebook(new_code_idx).squeeze(0)

        # Decode with true codes - note: decode expects [B, T-1, C, H, W] and returns [B, C, H, W]
        z_pred_true = model_ref.decode(z_past, codes_past)  # [B, C, H, W]

        # Decode with random codes
        z_pred_rand = model_ref.decode(z_past, codes_rand)  # [B, C, H, W]

        # Decode with perturbed codes
        z_pred_perturbed = model_ref.decode(z_past, codes_perturbed)  # [B, C, H, W]

        # Compute distances (L2 norm in VVAE latent space)
        # Flatten VVAE latents for distance computation: [B, C*H*W]
        z_target_flat = z_target.reshape(B, -1)
        z_pred_true_flat = z_pred_true.reshape(B, -1)
        z_pred_rand_flat = z_pred_rand.reshape(B, -1)
        z_pred_perturbed_flat = z_pred_perturbed.reshape(B, -1)
        
        # L2 distances
        d_action = torch.norm(z_pred_true_flat - z_target_flat, dim=1)  # [B] (renamed from d_true)
        d_rand = torch.norm(z_pred_rand_flat - z_target_flat, dim=1)  # [B]
        d_action_eps = torch.norm(z_pred_perturbed_flat - z_target_flat, dim=1)  # [B]
        
        # L1 distances
        d_action_l1 = torch.norm(z_pred_true_flat - z_target_flat, p=1, dim=1)  # [B]
        d_rand_l1 = torch.norm(z_pred_rand_flat - z_target_flat, p=1, dim=1)  # [B]
        
        # MAE (Mean Absolute Error) - averaged over all dimensions
        num_elements = z_pred_true_flat.shape[1]  # C*H*W for VVAE
        d_action_mae = torch.abs(z_pred_true_flat - z_target_flat).sum(dim=1) / num_elements  # [B]
        d_rand_mae = torch.abs(z_pred_rand_flat - z_target_flat).sum(dim=1) / num_elements  # [B]
        
        # Action sensitivity: Δd = d_rand - d_action
        # Positive values indicate decoder uses action meaningfully
        action_sensitivity = (d_rand - d_action).mean().item()
        action_sensitivity_l1 = (d_rand_l1 - d_action_l1).mean().item()
        action_sensitivity_mae = (d_rand_mae - d_action_mae).mean().item()
        
        # DSNR: Decoder Signal-to-Noise Ratio
        # DSNR = (d_rand - d_action) / (d_action_eps - d_action)
        # Higher values indicate better action sensitivity relative to noise
        denominator = d_action_eps - d_action
        # Avoid division by zero by adding small epsilon
        dsnr = (d_rand - d_action) / (denominator + 1e-8)
        dsnr_mean = dsnr.mean().item()
        
        # Also compute cosine distance version for additional insight
        # Cosine distance = 1 - cosine_similarity
        cos_sim_true = torch.nn.functional.cosine_similarity(z_pred_true_flat, z_target_flat, dim=1)
        cos_sim_rand = torch.nn.functional.cosine_similarity(z_pred_rand_flat, z_target_flat, dim=1)
        d_cos_action = 1 - cos_sim_true  # [B]
        d_cos_rand = 1 - cos_sim_rand  # [B]
        action_sensitivity_cos = (d_cos_rand - d_cos_action).mean().item()

        # Get codebook usage and indices for histogram
        codebook_usage = loss_dict.get('codebook_usage', 0)
        indices = loss_dict.get('indices', None)  # [B, T-1, 3]

        return {
            'loss': loss_dict['loss'].item(),                           # Total loss (MSE-based)
            'loss_mae': loss_dict['loss_mae'].item(),                   # Total loss (MAE-based, monitoring)
            'recon_loss': loss_dict['recon_loss'].item(),               # Reconstruction MSE
            'mse_loss': loss_dict['mse_loss'].item(),                   # MSE (same as recon_loss)
            'mae_loss': loss_dict['mae_loss'].item(),                   # MAE (monitoring)
            'vq_loss': loss_dict['vq_loss'].item(),                     # VQ loss
            'codebook_loss': loss_dict['codebook_loss'].item(),         # Codebook loss component
            'commitment_loss': loss_dict['commitment_loss'].item(),     # Commitment loss component
            'rollout_loss': loss_dict['rollout_loss'].item(),           # Rollout MSE
            'rollout_loss_mae': loss_dict['rollout_loss_mae'].item(),   # Rollout MAE (monitoring)
            'action_sensitivity_l2': action_sensitivity,                # Action sensitivity (L2 distance)
            'action_sensitivity_l1': action_sensitivity_l1,             # Action sensitivity (L1 distance)
            'action_sensitivity_mae': action_sensitivity_mae,           # Action sensitivity (MAE)
            'action_sensitivity_cos': action_sensitivity_cos,           # Action sensitivity (cosine)
            'dsnr': dsnr_mean,                                          # Decoder Signal-to-Noise Ratio
            'd_action_l2': d_action.mean().item(),                      # L2 distance with true actions
            'd_rand_l2': d_rand.mean().item(),                          # L2 distance with random actions
            'd_action_eps_l2': d_action_eps.mean().item(),              # L2 distance with perturbed actions
            'd_action_l1': d_action_l1.mean().item(),                   # L1 distance with true actions
            'd_rand_l1': d_rand_l1.mean().item(),                       # L1 distance with random actions
            'd_action_mae': d_action_mae.mean().item(),                 # MAE with true actions
            'd_rand_mae': d_rand_mae.mean().item(),                     # MAE with random actions
            'codebook_usage': codebook_usage,                           # Unique codes used
            'indices': indices,                                         # Codebook indices for histogram
            'total_tflops': self.total_tflops                           # Cumulative TFLOPs
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_metrics = {}
        all_train_indices = []  # Collect indices for histogram in test/overfit mode

        if self.test_mode:
            # Test mode: use only the fixed batch multiple times
            num_steps = 100  # Fixed number of steps per epoch in test mode
            logger.info(f"Test mode: Running {num_steps} steps with the same batch")

            progress_bar = tqdm(range(num_steps), desc=f"Test Epoch {self.epoch} (Fixed Batch)")

            for step in progress_bar:
                metrics = self.train_step(self.fixed_batch)

                # Collect indices for histogram
                if 'indices' in metrics and metrics['indices'] is not None:
                    all_train_indices.append(metrics['indices'].cpu())

                # Update epoch metrics (skip non-numeric keys)
                for key, value in metrics.items():
                    if key == 'indices':  # Skip indices
                        continue
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    epoch_metrics[key] += value

                # Update progress bar
                if step % 10 == 0:
                    postfix = {
                        'loss': f"{metrics['loss']:.4f}",
                        'recon': f"{metrics['recon_loss']:.4f}",
                        'vq': f"{metrics['vq_loss']:.4f}",
                        'rollout': f"{metrics['rollout_loss']:.4f}"
                    }
                    if 'codebook_usage' in metrics:
                        postfix['cb_usage'] = f"{int(metrics['codebook_usage'])}"
                    progress_bar.set_postfix(postfix)

                # Log to wandb more frequently in test mode
                if self.global_step % 5 == 0 and self.use_wandb:
                    log_dict = {f'train_test/{k}': v for k, v in metrics.items() if k != 'indices'}
                    # Add histogram every 20 steps
                    if step % 20 == 0 and all_train_indices:
                        indices_tensor = torch.cat(all_train_indices[-5:], dim=0)  # Last 5 batches
                        indices_flat = indices_tensor.flatten().numpy()
                        model_ref = self.model.module if self.use_ddp else self.model
                        log_dict['train_test/codebook_histogram'] = wandb.Histogram(indices_flat, num_bins=model_ref.num_embeddings)
                    wandb.log(log_dict, step=self.global_step)
            
            # Average metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= num_steps
                
        else:
            # Normal mode: iterate through all batches
            num_batches = len(self.train_loader)
            update_frequency = max(1, num_batches // 20)  # Update progress 20 times per epoch
            
            progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {self.epoch}")
            
            for batch_idx, batch in enumerate(progress_bar):
                metrics = self.train_step(batch)
                
                # Update epoch metrics (initialize keys dynamically)
                for key, value in metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    epoch_metrics[key] += value
                
                # Update progress bar only periodically to reduce lag
                if batch_idx % update_frequency == 0 or batch_idx == num_batches - 1:
                    postfix = {
                        'loss': f"{metrics['loss']:.4f}",
                        'recon': f"{metrics['recon_loss']:.4f}",
                        'vq': f"{metrics['vq_loss']:.4f}",
                        'rollout': f"{metrics['rollout_loss']:.4f}"
                    }
                    if 'mse_loss' in metrics:
                        postfix['mse'] = f"{metrics['mse_loss']:.4f}"
                    progress_bar.set_postfix(postfix)
                
                # Log to wandb (rank 0 only)
                if self.global_step % 20 == 0 and self.use_wandb and (not self.use_ddp or dist.get_rank() == 0):
                    # Filter out non-scalar metrics (indices)
                    scalar_metrics = {k: v for k, v in metrics.items() if k != 'indices'}
                    wandb.log({f'train/{k}': v for k, v in scalar_metrics.items()}, step=self.global_step)

                    # Also log training loss vs TFLOPs for plotting
                    if 'total_tflops' in metrics and metrics['total_tflops'] > 0:
                        wandb.log({
                            # MSE-based metrics (used for optimization)
                            'tflops/train_loss_mse': metrics['loss'],
                            'tflops/train_recon_mse': metrics['recon_loss'],
                            'tflops/train_rollout_mse': metrics['rollout_loss'],
                            # MAE-based metrics (monitoring)
                            'tflops/train_loss_mae': metrics['loss_mae'],
                            'tflops/train_recon_mae': metrics['mae_loss'],
                            'tflops/train_rollout_mae': metrics['rollout_loss_mae'],
                            # Shared metrics
                            'tflops/train_vq_loss': metrics['vq_loss'],
                            'tflops/x_axis': metrics['total_tflops']
                        }, step=self.global_step)
                
                # Validation every half epoch (skip in test mode)
                if not self.test_mode and self.global_step % self.val_frequency == 0 and self.global_step > 0:
                    val_metrics = self.evaluate(self.val_loader, "val")
                    logger.info(f"Step {self.global_step} - Val metrics: {val_metrics}")
                    
                    # Check for elite models
                    self._check_elite_models(val_metrics)
            
            # Average metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str = "val") -> Dict[str, float]:
        """Evaluate on a dataset or a random subset of it."""
        self.model.eval()

        epoch_metrics = {}
        all_indices = []  # Collect all codebook indices for histogram
        num_batches = len(loader)

        # Determine which batches to evaluate
        if self.validation_fraction < 1.0:
            num_batches_to_eval = max(1, int(num_batches * self.validation_fraction))
            logger.info(f"Evaluating {num_batches_to_eval}/{num_batches} batches ({self.validation_fraction*100:.0f}%) from {split}")

            # Create a subset of indices to sample
            dataset_size = len(loader.dataset)
            samples_to_eval = max(1, int(dataset_size * self.validation_fraction))

            # Randomly sample dataset indices (not batch indices)
            sampled_indices = np.random.choice(dataset_size, samples_to_eval, replace=False)

            # Create new dataloader with subset sampler
            from torch.utils.data import SubsetRandomSampler
            subset_sampler = SubsetRandomSampler(sampled_indices)

            subset_loader = DataLoader(
                loader.dataset,
                batch_size=loader.batch_size,
                sampler=subset_sampler,
                num_workers=0,  # Use 0 workers for subset to avoid deadlock
                pin_memory=loader.pin_memory
            )

            actual_loader = subset_loader
            num_batches_to_eval = len(subset_loader)
        else:
            actual_loader = loader
            num_batches_to_eval = num_batches

        update_frequency = max(1, num_batches_to_eval // 10)  # Update progress 10 times during evaluation

        # Create manual progress bar
        progress_bar = tqdm(total=num_batches_to_eval, desc=f"Evaluating {split} ({num_batches_to_eval} batches)")

        evaluated_count = 0
        for batch_idx, batch in enumerate(actual_loader):
            evaluated_count += 1
            metrics = self.eval_step(batch)

            # Collect codebook indices for histogram
            if 'indices' in metrics and metrics['indices'] is not None:
                all_indices.append(metrics['indices'].cpu())

            # Update epoch metrics (initialize keys dynamically, skip non-numeric keys)
            for key, value in metrics.items():
                if key == 'indices':  # Skip indices, we're collecting them separately
                    continue
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value

            # Update progress bar only periodically to reduce lag
            if evaluated_count % update_frequency == 0 or evaluated_count == num_batches_to_eval:
                postfix = {
                    'loss': f"{metrics['loss']:.4f}",
                    'recon': f"{metrics['recon_loss']:.4f}",
                    'vq': f"{metrics['vq_loss']:.4f}",
                    'rollout': f"{metrics['rollout_loss']:.4f}"
                }
                if 'mse_loss' in metrics:
                    postfix['mse'] = f"{metrics['mse_loss']:.4f}"
                if 'action_sensitivity_l2' in metrics:
                    postfix['act_sens'] = f"{metrics['action_sensitivity_l2']:.3f}"
                if 'codebook_usage' in metrics:
                    postfix['cb_usage'] = f"{int(metrics['codebook_usage'])}"
                progress_bar.set_postfix(postfix)

            progress_bar.update(1)

        progress_bar.close()

        # Average metrics over the number of batches actually evaluated
        for key in epoch_metrics:
            epoch_metrics[key] /= evaluated_count
            
        # Average across DDP ranks
        if self.use_ddp:
            for key in epoch_metrics:
                epoch_metrics[key] = _ddp_mean(epoch_metrics[key], self.device)

        # Create codebook histogram and compute total unique codes if we have indices
        codebook_histogram = None
        if all_indices and self.use_wandb and (not self.use_ddp or dist.get_rank() == 0):
            # Concatenate all indices: [total_batches*B, T-1, 3]
            all_indices_tensor = torch.cat(all_indices, dim=0)  # [N, T-1, 3]
            # Flatten to get all code usage: [N*(T-1)*3]
            indices_flat = all_indices_tensor.flatten().numpy()

            # Get codebook size from model
            model_ref = self.model.module if self.use_ddp else self.model
            codebook_size = model_ref.num_embeddings

            # Compute total unique codes used across entire validation run
            total_unique_codes = len(np.unique(indices_flat))
            epoch_metrics['codebook_unique_total'] = total_unique_codes
            epoch_metrics['codebook_usage_ratio'] = total_unique_codes / codebook_size

            # Create histogram
            codebook_histogram = wandb.Histogram(indices_flat, num_bins=codebook_size)

        # Log to wandb with TFLOP tracking for plots (rank 0 only)
        if self.use_wandb and (not self.use_ddp or dist.get_rank() == 0):
            log_dict = {f'{split}/{k}': v for k, v in epoch_metrics.items()}

            # Add codebook histogram
            if codebook_histogram is not None:
                log_dict[f'{split}/codebook_histogram'] = codebook_histogram
            
            # Add TFLOP-indexed metrics for validation to create plots with TFLOPs on x-axis
            if split == 'val' and 'total_tflops' in epoch_metrics:
                total_tflops = epoch_metrics['total_tflops']
                # MSE-based metrics
                val_loss_mse = epoch_metrics['loss']
                recon_mse = epoch_metrics['recon_loss']
                rollout_mse = epoch_metrics.get('rollout_loss', 0.0)
                # MAE-based metrics
                val_loss_mae = epoch_metrics.get('loss_mae', 0.0)
                recon_mae = epoch_metrics.get('mae_loss', 0.0)
                rollout_mae = epoch_metrics.get('rollout_loss_mae', 0.0)
                # Shared metrics
                vq_loss = epoch_metrics['vq_loss']
                dsnr = epoch_metrics.get('dsnr', 0.0)

                # Log metrics with TFLOPs as x-axis (using custom x-axis in wandb)
                # These will create plots with TFLOPs on x-axis when viewed in WandB
                if total_tflops > 0:
                    wandb.log({
                        # MSE-based metrics (used for optimization)
                        'tflops/val_loss_mse': val_loss_mse,
                        'tflops/val_recon_mse': recon_mse,
                        'tflops/val_rollout_mse': rollout_mse,
                        # MAE-based metrics (monitoring)
                        'tflops/val_loss_mae': val_loss_mae,
                        'tflops/val_recon_mae': recon_mae,
                        'tflops/val_rollout_mae': rollout_mae,
                        # Shared metrics
                        'tflops/val_vq_loss': vq_loss,
                        'tflops/val_dsnr': dsnr,
                        'tflops/x_axis': total_tflops  # This serves as the x-axis value
                    }, step=self.global_step)

                    # Efficiency metrics (using MSE-based loss)
                    log_dict[f'{split}/loss_per_tflop'] = val_loss_mse / total_tflops
                    log_dict[f'{split}/recon_per_tflop'] = recon_mse / total_tflops
            
            wandb.log(log_dict, step=self.global_step)
        
        return epoch_metrics
    
    def _check_elite_models(self, val_metrics: Dict[str, float]):
        """Check and save elite models based on validation metrics."""
        # Elite model with lowest total loss
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            self.save_checkpoint(checkpoint_type='elite_total')
            logger.info(f"New elite model (total loss): {self.best_val_loss:.4f}")
            if self.use_wandb:
                wandb.log({'best_val_loss': self.best_val_loss}, step=self.global_step)
        
        # Elite model with lowest reconstruction loss
        if val_metrics['recon_loss'] < self.best_recon_loss:
            self.best_recon_loss = val_metrics['recon_loss']
            self.save_checkpoint(checkpoint_type='elite_recon')
            logger.info(f"New elite model (recon loss): {self.best_recon_loss:.4f}")
            if self.use_wandb:
                wandb.log({'best_recon_loss': self.best_recon_loss}, step=self.global_step)
    
    def save_checkpoint(self, is_best: bool = False, checkpoint_type: str = 'regular'):
        """Save model checkpoint."""
        if self.use_ddp and dist.get_rank() != 0:
            return
            
        model_for_save = self.model.module if self.use_ddp else self.model
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_for_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'scaler': self.scaler.state_dict() if self.mixed_precision else None,
            'config': model_for_save.config  # Use model's config dict (works for both VAE and VQ-VAE)
        }
        
        # Save based on checkpoint type
        if checkpoint_type == 'elite_total':
            path = self.checkpoint_dir / "elite_total_loss.pt"
            torch.save(checkpoint, path)
            logger.info(f"Saved elite model (total loss) to {path}")
        elif checkpoint_type == 'elite_recon':
            path = self.checkpoint_dir / "elite_recon_loss.pt"
            torch.save(checkpoint, path)
            logger.info(f"Saved elite model (recon loss) to {path}")
        elif checkpoint_type == 'regular':
            # Always save epoch checkpoint
            epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch}_step_{self.global_step}.pt"
            torch.save(checkpoint, epoch_path)
            logger.info(f"Saved checkpoint to {epoch_path}")
            
            # Save best model (backward compatibility)
            if is_best:
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                logger.info(f"Saved best model to {best_path}")
        
        # Always save latest checkpoint (for easy resuming)
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Log to wandb
        if self.use_wandb and checkpoint_type != 'regular':
            wandb.save(str(path))
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if self.use_ddp:
            dist.barrier() # Ensure all processes are ready before loading

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model_to_load = self.model.module if self.use_ddp else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.mixed_precision and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
            self.scaler.load_state_dict(checkpoint['scaler'])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.epoch}, step {self.global_step}")
    
    def train(self, num_epochs: int):
        """Main training loop."""
        # Set total training steps for KL annealing (reaches max at final epoch)
        self.total_training_steps = num_epochs * self.steps_per_epoch
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
        logger.info(f"Learning rate scheduler: {'ENABLED' if self.use_scheduler else 'DISABLED'}")
        if self.use_scheduler:
            logger.info(f"  LR schedule: {self.base_lr} -> {self.base_lr * self.min_lr_ratio} (min={self.min_lr_ratio * 100:.0f}% of base)")
        else:
            logger.info(f"  Fixed learning rate: {self.base_lr}")
        logger.info(f"KL weight will reach max ({self.kl_max_weight}) at step {self.total_training_steps}")
        
        # Initialize FLOP tracking for compute efficiency analysis
        self._initialize_flop_tracking()
        
        # Initial validation run to establish baseline (skip in test mode)
        if not self.test_mode:
            logger.info("Running initial validation...")
            initial_val_metrics = self.evaluate(self.val_loader, "val")
            logger.info(f"Initial validation metrics: {initial_val_metrics}")
        else:
            logger.info("Test mode: Skipping initial validation")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            if self.test_mode:
                # Enhanced logging for test mode - check if model is overfitting
                logger.info("=" * 50)
                logger.info(f"TEST MODE - Epoch {epoch} Results:")
                logger.info(f"  Total Loss (MSE):     {train_metrics['loss']:.6f}")
                logger.info(f"  Total Loss (MAE):     {train_metrics['loss_mae']:.6f}")
                logger.info(f"  Recon MSE:            {train_metrics['mse_loss']:.6f}")
                logger.info(f"  Recon MAE:            {train_metrics['mae_loss']:.6f}")
                logger.info(f"  VQ Loss:              {train_metrics['vq_loss']:.6f}")
                logger.info(f"  Rollout MSE:          {train_metrics['rollout_loss']:.6f}")
                logger.info(f"  Rollout MAE:          {train_metrics['rollout_loss_mae']:.6f}")
                logger.info(f"  Codebook Usage:       {train_metrics.get('codebook_usage', 'N/A')}")
                logger.info(f"  Learning Rate:        {train_metrics['lr']:.6f}")
                
                # Check if loss is decreasing (good sign)
                if epoch > 0 and hasattr(self, 'prev_loss'):
                    loss_change = train_metrics['loss'] - self.prev_loss
                    if loss_change < -1e-6:
                        logger.info(f"  ✅ Loss DECREASED by {-loss_change:.6f} (Good!)")
                    elif loss_change > 1e-6:
                        logger.info(f"  ❌ Loss INCREASED by {loss_change:.6f} (Bad!)")
                    else:
                        logger.info(f"  ⚠️  Loss UNCHANGED (±{abs(loss_change):.6f})")
                
                self.prev_loss = train_metrics['loss']
                
                # Gradient sanity check on first epoch
                if epoch == 0 and self.global_step >= 10:  # After 10 steps
                    logger.info("\n=== Gradient Sanity Check ===")
                    no_grad_params = []
                    # For VVAE LAM, decoder is at model.lam.decoder
                    decoder = self.model.lam.decoder if hasattr(self.model, 'lam') else self.model.decoder
                    for n, p in decoder.named_parameters():
                        if p.grad is None:
                            no_grad_params.append(n)
                        elif p.grad.norm().item() < 1e-8:
                            logger.info(f"  ⚠️  Very small gradient: {n} (norm={p.grad.norm().item():.2e})")

                    if no_grad_params:
                        logger.warning("  ⚠️  Parameters with NO gradients:")
                        for n in no_grad_params:
                            logger.warning(f"    - {n}")
                    else:
                        logger.info("  ✅ All decoder parameters have gradients!")

                    # Check key components
                    key_params = ['query_embed', 'cross_attn', 'output_proj']
                    for key in key_params:
                        has_grad = any(key in n for n, p in decoder.named_parameters() if p.grad is not None and p.grad.norm().item() > 1e-8)
                        if has_grad:
                            logger.info(f"  ✅ {key} components have gradients")
                        else:
                            logger.warning(f"  ❌ {key} components missing gradients!")
                
                logger.info("=" * 50)
            else:
                logger.info(f"Epoch {epoch} - Train metrics: {train_metrics}")
            
            # Skip all evaluation in test mode
            if not self.test_mode:
                # Evaluate
                if (epoch + 1) % self.eval_every == 0:
                    val_metrics = self.evaluate(self.val_loader, "val")
                    logger.info(f"Epoch {epoch} - Val metrics: {val_metrics}")
                    
                    # Check if best model
                    is_best = val_metrics['loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics['loss']
                        logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
                    
                    # Test if available
                    if self.test_loader is not None:
                        test_metrics = self.evaluate(self.test_loader, "test")
                        logger.info(f"Epoch {epoch} - Test metrics: {test_metrics}")
                else:
                    is_best = False
                
                # Save checkpoint every epoch in normal mode
                self.save_checkpoint(is_best, checkpoint_type='regular')
            else:
                # In test mode, only save checkpoint at the end
                if epoch == num_epochs - 1:
                    self.save_checkpoint(is_best=False, checkpoint_type='regular')
                    logger.info(f"Test mode complete - saved final checkpoint")
        
        logger.info("Training completed!")
        if self.use_wandb:
            wandb.finish()


def train_with_variable_context(
    data_dir: str,
    manifest_path: str,
    num_epochs: int = 100,
    batch_size: int = 32,
    context_length: int = 20,
    device: str = "cuda",
    is_ddp: bool = False,
    **kwargs
):
    """
    Train LAM with fixed context length.
    Causal masking handles variable-length contexts automatically.

    Args:
        data_dir: Path to dataset directory
        manifest_path: Path to manifest file
        num_epochs: Total number of epochs
        batch_size: Batch size
        context_length: Fixed context length (model uses causal masking for variable lengths)
        device: Device to train on
        is_ddp: Whether distributed training is enabled
        **kwargs: Additional arguments for model and trainer
            resume_from: Path to checkpoint to resume from (optional)
    """

    # Initialize VVAE VQ-VAE model with adapters
    model = VVAELatentActionVQVAE(
        codebook_dim=kwargs.get('codebook_dim', 128),
        num_embeddings=kwargs.get('num_embeddings', 12),
        embed_dim=kwargs.get('embed_dim', 512),
        encoder_depth=kwargs.get('encoder_depth', 3),
        decoder_depth=kwargs.get('decoder_depth', 3),
        encoder_heads=kwargs.get('encoder_heads', 8),
        decoder_heads=kwargs.get('decoder_heads', 8),
        mlp_ratio=kwargs.get('mlp_ratio', 4.0),
        drop_rate=kwargs.get('drop_rate', 0.0),
        attn_drop_rate=kwargs.get('attn_drop_rate', 0.0),
        commitment_weight=kwargs.get('commitment_weight', 0.25),
        reconstruction_weight=kwargs.get('reconstruction_weight', 1.0),
        max_seq_len=context_length
    )

    # Create dataloaders with fixed context length
    train_loader, val_loader, test_loader = create_vvae_dataloaders(
        data_dir=data_dir,
        manifest_path=manifest_path,
        batch_size=batch_size,
        sequence_length=context_length,
        num_workers=kwargs.get('num_workers', 8),
        ddp=is_ddp
    )
    
    # Initialize trainer
    trainer = VVAELAMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=kwargs.get('learning_rate', 1e-4),
        kl_annealing_steps=kwargs.get('kl_annealing_steps', 10000),
        mixed_precision=kwargs.get('mixed_precision', True),
        test_mode=kwargs.get('test_mode', False),
        use_scheduler=kwargs.get('use_scheduler', False),
        min_lr_ratio=kwargs.get('min_lr_ratio', 0.1),
        rollout_horizon=kwargs.get('rollout_horizon', 2),
        rollout_weight=kwargs.get('rollout_weight', 1.0),
        rollout_prob=kwargs.get('rollout_prob', 1.0),
        detach_rollout_first=kwargs.get('detach_rollout_first', True),
        anchor_strategy=kwargs.get('anchor_strategy', 'random'),
        validation_fraction=kwargs.get('validation_fraction', 1.0),
        use_wandb=kwargs.get('use_wandb', False),
        wandb_project=kwargs.get('project_name', 'vvae-lam'),
        wandb_run_name=kwargs.get('run_name', None)
    )

    # Load checkpoint if resuming
    resume_from = kwargs.get('resume_from', None)
    if resume_from is not None:
        logger.info(f"Resuming training from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)
        logger.info(f"Resumed from epoch {trainer.epoch}, step {trainer.global_step}")

    # Initial validation run to establish baseline (skip in test mode or when resuming)
    if not kwargs.get('test_mode', False) and resume_from is None:
        logger.info("Running initial validation...")
        initial_val_metrics = trainer.evaluate(trainer.val_loader, "val")
        logger.info(f"Initial validation metrics: {initial_val_metrics}")
    elif resume_from is not None:
        logger.info("Resuming from checkpoint: Skipping initial validation")
    else:
        logger.info("Test mode: Skipping initial validation")
    
    # Set total training steps for KL annealing (reaches max at final epoch)
    trainer.total_training_steps = num_epochs * len(train_loader)
    logger.info(f"KL weight will reach max ({trainer.kl_max_weight}) at step {trainer.total_training_steps}")

    # Initialize FLOP tracking for compute efficiency analysis
    trainer._initialize_flop_tracking()

    # Determine starting epoch for training loop
    start_epoch = trainer.epoch + 1 if resume_from is not None else 0
    if resume_from is not None:
        logger.info(f"Continuing training from epoch {start_epoch} (checkpoint was at epoch {trainer.epoch})")

    if start_epoch >= num_epochs:
        logger.info(f"Training already completed for {num_epochs} epochs. Exiting.")
        return trainer.model

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Set epoch for distributed sampler
        if is_ddp:
            train_sampler = train_loader.sampler if hasattr(train_loader, 'sampler') else None
            val_sampler = val_loader.sampler if val_loader and hasattr(val_loader, 'sampler') else None
            test_sampler = test_loader.sampler if test_loader and hasattr(test_loader, 'sampler') else None

            if train_sampler:
                train_sampler.set_epoch(epoch)
            if val_sampler:
                val_sampler.set_epoch(epoch)
            if test_sampler:
                test_sampler.set_epoch(epoch)

        # Train for one epoch
        trainer.epoch = epoch
        train_metrics = trainer.train_epoch()
        logger.info(f"Epoch {epoch} - Train: {train_metrics}")
        
        # Skip evaluation in test mode
        if not kwargs.get('test_mode', False):
            # Evaluate
            if (epoch + 1) % trainer.eval_every == 0:
                val_metrics = trainer.evaluate(trainer.val_loader, "val")
                logger.info(f"Epoch {epoch} - Val: {val_metrics}")

                # Synchronize before saving checkpoint
                if is_ddp:
                    dist.barrier()
                
                # Save checkpoint
                is_best = val_metrics['loss'] < trainer.best_val_loss
                if is_best:
                    trainer.best_val_loss = val_metrics['loss']
                
                if (epoch + 1) % trainer.save_every == 0 or is_best:
                    trainer.save_checkpoint(is_best)
        else:
            # In test mode, only save at the end
            if epoch == num_epochs - 1:
                trainer.save_checkpoint(is_best=False)
                logger.info("Test mode complete - saved final checkpoint")
    
    if hasattr(trainer, 'writer'):
        trainer.writer.close()
    return trainer.model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Latent Action Model VAE")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to manifest file")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--test_mode", action="store_true", help="Test mode: use only 1 batch for debugging/overfitting test")
    parser.add_argument("--context_length", type=int, default=20, help="Context length (model uses causal masking for variable lengths)")

    # DDP arguments
    # local_rank is deprecated, but we keep it for backward compatibility.
    # torchrun uses the LOCAL_RANK env var.
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for DistributedDataParallel (deprecated)')
    parser.add_argument("--dist_backend", default=None, help="nccl|gloo")

    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")

    # VQ-VAE specific arguments
    parser.add_argument("--codebook_dim", type=int, default=128, help="Dimension of each codebook entry")
    parser.add_argument("--num_embeddings", type=int, default=12, help="Number of codebook entries (vocabulary size)")
    parser.add_argument("--commitment_weight", type=float, default=0.25, help="Commitment loss weight (beta)")
    parser.add_argument("--reconstruction_weight", type=float, default=1.0, help="Reconstruction loss weight")

    # WandB arguments
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--project_name", type=str, default="vvae-lam", help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")

    # Scheduler
    parser.add_argument("--scheduler", action="store_true", help="Enable manual LR scheduling with warmup and cosine annealing")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum LR as ratio of base LR (default: 0.1 = 10%% of base)")

    # Rollout loss arguments
    parser.add_argument("--rollout_horizon", type=int, default=2, help="Number of steps for rollout (1 = no rollout; 2-3 typical)")
    parser.add_argument("--rollout_weight", type=float, default=1.0, help="Weight for rollout loss component")
    parser.add_argument("--rollout_prob", type=float, default=1.0, help="Probability of computing rollout loss (0-1)")
    parser.add_argument("--detach_rollout_first", action="store_true", help="Detach first predicted state before rollout for gradient stability")
    parser.add_argument("--anchor_strategy", type=str, default="random", choices=["random", "last"], help="Rollout anchor strategy")
    # Because the encoder outputs T−1 actions, a rollout spanning rollout_horizon steps requires T ≥ rollout_horizon + 1 frames. With --rollout_horizon 2 you need sequence length T ≥ 3 (so setting --min_context 3 enables rollouts immediately). The model skips the rollout branch when that condition fails, producing rollout_loss = 0.

    # Model architecture arguments (latent_dim removed for VVAE - fixed at 256 by adapter)
    parser.add_argument("--encoder_depth", type=int, default=3, help="Number of encoder transformer layers")
    parser.add_argument("--decoder_depth", type=int, default=3, help="Number of decoder transformer layers")
    parser.add_argument("--encoder_heads", type=int, default=8, help="Number of encoder attention heads")
    parser.add_argument("--decoder_heads", type=int, default=8, help="Number of decoder attention heads")
    parser.add_argument("--embed_dim", type=int, default=512, help="Transformer embedding dimension")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP hidden dimension multiplier")
    parser.add_argument("--drop_rate", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--attn_drop_rate", type=float, default=0.0, help="Attention dropout rate")

    # Validation arguments
    parser.add_argument("--validation_fraction", type=float, default=1.0, help="Fraction of validation set to evaluate each time (0-1, e.g., 0.35 for 35%%)")

    # Resume training arguments
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from (e.g., checkpoints/lam_20251013_121224/latest.pt)")

    args = parser.parse_args()

    # --- DDP Initialization ---
    # torchrun provides the necessary environment variables for DDP.
    is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if is_ddp:
        # The LOCAL_RANK env var is set by torchrun.
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend=args.dist_backend or ('nccl' if torch.cuda.is_available() else 'gloo'))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        # Set the device for the current process.
        args.device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

    # Setup logging (show rank if DDP)
    log_format = '%(asctime)s - %(name)s'
    if is_ddp:
        log_format += f' - RANK {dist.get_rank()}'
    log_format += ' - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=logging.INFO,
        format=log_format
    )
    
    # Override settings in test mode
    if args.test_mode:
        print("TEST MODE: Overriding settings")
        args.drop_rate = 0.0
        args.attn_drop_rate = 0.0
        print(f"  - Dropout rates set to 0.0")

    print(f"Using context length: {args.context_length} (causal masking handles variable lengths)")

    # Train model
    try:
        model = train_with_variable_context(
            data_dir=args.data_dir,
            manifest_path=args.manifest_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
            mixed_precision=args.mixed_precision,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
            test_mode=args.test_mode,
            use_scheduler=args.scheduler,
            min_lr_ratio=args.min_lr_ratio,
            # Rollout loss parameters
            rollout_horizon=args.rollout_horizon,
            rollout_weight=args.rollout_weight,
            rollout_prob=args.rollout_prob,
            detach_rollout_first=args.detach_rollout_first,
            anchor_strategy=args.anchor_strategy,
            # Model architecture parameters (no latent_dim for VVAE)
            encoder_depth=args.encoder_depth,
            decoder_depth=args.decoder_depth,
            encoder_heads=args.encoder_heads,
            decoder_heads=args.decoder_heads,
            embed_dim=args.embed_dim,
            mlp_ratio=args.mlp_ratio,
            drop_rate=args.drop_rate,
            attn_drop_rate=args.attn_drop_rate,
            # VQ-VAE specific parameters
            codebook_dim=args.codebook_dim,
            num_embeddings=args.num_embeddings,
            commitment_weight=args.commitment_weight,
            reconstruction_weight=args.reconstruction_weight,
            # Validation parameters
            validation_fraction=args.validation_fraction,
            # Resume parameters
            resume_from=args.resume_from,
            # WandB parameters
            use_wandb=args.use_wandb,
            project_name=args.project_name,
            run_name=args.run_name,
            is_ddp=is_ddp
        )
    finally:
        if is_ddp:
            dist.destroy_process_group()
