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

from .vae import LatentActionVAE
from .dataloader import create_dataloaders, LAMDataset


logger = logging.getLogger(__name__)


def _ddp_mean(x: float, device) -> float:
    """Average a scalar across all DDP ranks."""
    if not dist.is_initialized():
        return x
    t = torch.tensor([x], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def calculate_model_flops(model, input_shape=(8, 256, 1024), device="cuda"):
    """
    Calculate FLOPs for the LAM VAE model.
    
    Args:
        model: LAM VAE model
        input_shape: Input tensor shape (T, N, D) for sequence
        device: Device to run calculation on
    
    Returns:
        forward_flops: FLOPs for forward pass
        backward_flops: Approximate FLOPs for backward pass (3x forward)
    """
    model.eval()
    
    # Create dummy input batch
    T, N, D = input_shape
    B = 1  # Use batch size 1 for FLOP calculation
    dummy_input = torch.randn(B, T, N, D).to(device)
    
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
            input_size = (T, N, D)
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
        embed_dim = model.encoder.embed_dim
        action_dim = model.action_dim
        num_patches = N
        seq_len = T
        
        # Encoder FLOPs (approximate)
        encoder_flops = 0
        encoder_flops += T * N * D * embed_dim  # Patch projection
        encoder_flops += len(model.encoder.blocks) * (T * N) * embed_dim * embed_dim * 4  # Attention + MLP
        encoder_flops += embed_dim * action_dim * 2  # Output heads
        
        # Decoder FLOPs (approximate) 
        decoder_flops = 0
        decoder_flops += action_dim * embed_dim  # Action projection
        decoder_flops += len(model.decoder.blocks) * N * embed_dim * embed_dim * 4  # Cross-attention + MLP
        decoder_flops += N * embed_dim * D  # Output projection
        
        # Loss computation (approximate)
        loss_flops = N * D * 2  # L1 loss + KL loss
        
        forward_flops = encoder_flops + decoder_flops + loss_flops
        logger.info(f"FLOPs estimated manually: {forward_flops:,}")
    
    # Backward pass is approximately 2x forward pass
    backward_flops = 2 * forward_flops
    total_flops = forward_flops + backward_flops
    
    return forward_flops, backward_flops, total_flops


class LAMTrainer:
    """
    Trainer for Latent Action Model VAE with variable context length support.
    """
    
    def __init__(
        self,
        model: LatentActionVAE,
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
        anchor_strategy: str = "random"
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
        """
        self.device = device
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
        self.kl_max_weight = model_for_optim.kl_weight

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
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_recon_loss = float('inf')  # Track best reconstruction loss
        
        # Validation frequency (every 5% of training steps per epoch - twice as frequent)
        self.steps_per_epoch = len(train_loader)
        self.val_frequency = max(1, self.steps_per_epoch // 20)  # Validate 20 times per epoch
        
        # Total training steps will be set when training starts
        self.total_training_steps = None
        
        # TFLOP tracking
        self.flops_per_step = None
        self.total_tflops = 0.0  # Cumulative TFLOPs consumed
        
        # WandB initialization
        self.test_mode = test_mode
        self.fixed_batch = None  # Will store the single batch in test mode
        
        self.use_wandb = use_wandb
        if use_wandb and (not self.use_ddp or dist.get_rank() == 0):
            run_name = wandb_run_name or (f"lam_test_mode_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if test_mode else None)
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    'latent_dim': model_for_optim.latent_dim,
                    'action_dim': model_for_optim.action_dim,
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
        
        # In test mode, grab and store a single batch
        if self.test_mode:
            logger.info("=" * 60)
            logger.info("TEST MODE ACTIVATED: Using only 1 batch for overfitting test")
            logger.info("Test mode defaults:")
            logger.info("  - KL weight: 0.0 (disabled)")
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
    
    def _save_model_config(self, model: LatentActionVAE):
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
        sequence_shape = sample_batch['sequence'].shape  # [B, T, N, D]
        input_shape = sequence_shape[1:]  # (T, N, D) - remove batch dimension
        
        try:
            forward_flops, backward_flops, total_flops = calculate_model_flops(
                self.model, 
                input_shape=input_shape, 
                device=self.device
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
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Apply layer normalization to input sequences (z-score normalization on D=1024)
        # This matches the preprocessing used by VJ2 predictor models
        batch['sequence'] = F.layer_norm(batch['sequence'], (batch['sequence'].size(-1),))
        if 'next' in batch:
            batch['next'] = F.layer_norm(batch['next'], (batch['next'].size(-1),))
        
        # Note: Additional layer normalization is also applied inside the model after patch_proj
        # for stability (on embed_dim=512). This double normalization is intentional.
        
        # Get current KL weight
        kl_weight = self.get_kl_weight()
        
        # Forward pass with mixed precision
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                loss_dict = self.model.compute_loss(
                    batch,
                    beta_schedule=kl_weight,
                    rollout_horizon=self.rollout_horizon,
                    rollout_weight=self.rollout_weight,
                    rollout_prob=self.rollout_prob,
                    detach_rollout_first=self.detach_rollout_first,
                    anchor_strategy=self.anchor_strategy
                )
                loss = loss_dict['loss']
        else:
            loss_dict = self.model.compute_loss(
                batch,
                beta_schedule=kl_weight,
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
        
        # Return metrics
        metrics = {
            'loss': loss.item(),
            'recon_loss': loss_dict['recon_loss'].item(),
            'kl_loss': loss_dict['kl_loss'].item(),
            'rollout_loss': loss_dict['rollout_loss'].item(),
            'kl_weight': kl_weight,
            'lr': self.get_lr(),
            'total_tflops': self.total_tflops  # Include cumulative TFLOPs
        }

        # Add MSE loss if available
        if 'mse_loss' in loss_dict:
            metrics['mse_loss'] = loss_dict['mse_loss'].item()

        return metrics
    
    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single evaluation step with action sensitivity check."""
        self.model.eval()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Apply layer normalization to input sequences (z-score normalization on D=1024)
        # This matches the preprocessing used by VJ2 predictor models
        batch['sequence'] = F.layer_norm(batch['sequence'], (batch['sequence'].size(-1),))
        if 'next' in batch:
            batch['next'] = F.layer_norm(batch['next'], (batch['next'].size(-1),))
        
        # Note: Additional layer normalization is also applied inside the model after patch_proj
        # for stability (on embed_dim=512). This double normalization is intentional.
        
        # Forward pass for standard losses with rollout parameters
        loss_dict = self.model.compute_loss(
            batch,
            rollout_horizon=self.rollout_horizon,
            rollout_weight=self.rollout_weight,
            rollout_prob=self.rollout_prob,
            detach_rollout_first=self.detach_rollout_first,
            anchor_strategy=self.anchor_strategy
        )
        
        # Action sensitivity check
        z_sequence = batch['sequence']  # [B, T, N, D]
        B, T, N, D = z_sequence.shape
        z_past = z_sequence[:, :-1, :, :]  # [B, T-1, N, D]
        z_target = z_sequence[:, -1, :, :]  # [B, N, D]
        
        # Get true action latents from encoder (now per-transition)
        mu, logvar = self.model.encode(z_sequence)  # [B, T-1, A] 
        a_true_seq = self.model.reparameterize(mu, logvar)  # [B, T-1, A]
        
        # For eval, we want the action for the final transition (z_{T-1} -> z_T)
        a_true = a_true_seq[:, -1, :]  # [B, A] - last action in sequence
        
        # Generate random action latent from prior N(0, I)
        a_rand = torch.randn_like(a_true)  # [B, A]
        
        # Generate slightly perturbed action for DSNR calculation
        epsilon = 0.01  # Small perturbation
        a_perturbed = a_true + epsilon * torch.randn_like(a_true)  # [B, A]
        
        # Decode with true action
        z_pred_true = self.model.decode(z_past, a_true)  # [B, N, D]
        
        # Decode with random action  
        z_pred_rand = self.model.decode(z_past, a_rand)  # [B, N, D]
        
        # Decode with perturbed action
        z_pred_perturbed = self.model.decode(z_past, a_perturbed)  # [B, N, D]
        
        # Compute distances (L2 norm in patch space)
        # Flatten patches for distance computation: [B, N*D]
        z_target_flat = z_target.view(B, -1)
        z_pred_true_flat = z_pred_true.view(B, -1)
        z_pred_rand_flat = z_pred_rand.view(B, -1)
        z_pred_perturbed_flat = z_pred_perturbed.view(B, -1)
        
        # L2 distances
        d_action = torch.norm(z_pred_true_flat - z_target_flat, dim=1)  # [B] (renamed from d_true)
        d_rand = torch.norm(z_pred_rand_flat - z_target_flat, dim=1)  # [B]
        d_action_eps = torch.norm(z_pred_perturbed_flat - z_target_flat, dim=1)  # [B]
        
        # L1 distances
        d_action_l1 = torch.norm(z_pred_true_flat - z_target_flat, p=1, dim=1)  # [B]
        d_rand_l1 = torch.norm(z_pred_rand_flat - z_target_flat, p=1, dim=1)  # [B]
        
        # MAE (Mean Absolute Error) - averaged over all dimensions
        num_elements = z_pred_true_flat.shape[1]  # N*D
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
        
        return {
            'loss': loss_dict['loss'].item(),
            'recon_loss': loss_dict['recon_loss'].item(),
            'mse_loss': loss_dict['mse_loss'].item(),
            'kl_loss': loss_dict['kl_loss'].item(),
            'rollout_loss': loss_dict['rollout_loss'].item(),
            'action_sensitivity_l2': action_sensitivity,
            'action_sensitivity_l1': action_sensitivity_l1,
            'action_sensitivity_mae': action_sensitivity_mae,
            'action_sensitivity_cos': action_sensitivity_cos,
            'dsnr': dsnr_mean,
            'd_action_l2': d_action.mean().item(),
            'd_rand_l2': d_rand.mean().item(),
            'd_action_eps_l2': d_action_eps.mean().item(),
            'd_action_l1': d_action_l1.mean().item(),
            'd_rand_l1': d_rand_l1.mean().item(),
            'd_action_mae': d_action_mae.mean().item(),
            'd_rand_mae': d_rand_mae.mean().item(),
            'total_tflops': self.total_tflops  # Include cumulative TFLOPs
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_metrics = {}
        
        if self.test_mode:
            # Test mode: use only the fixed batch multiple times
            num_steps = 100  # Fixed number of steps per epoch in test mode
            logger.info(f"Test mode: Running {num_steps} steps with the same batch")
            
            progress_bar = tqdm(range(num_steps), desc=f"Test Epoch {self.epoch} (Fixed Batch)")
            
            for step in progress_bar:
                metrics = self.train_step(self.fixed_batch)
                
                # Update epoch metrics
                for key, value in metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    epoch_metrics[key] += value
                
                # Update progress bar
                if step % 10 == 0:
                    postfix = {
                        'loss': f"{metrics['loss']:.4f}",
                        'recon': f"{metrics['recon_loss']:.4f}",
                        'kl': f"{metrics['kl_loss']:.4f}",
                        'rollout': f"{metrics['rollout_loss']:.4f}",
                        'kl_w': f"{metrics['kl_weight']:.3f}"
                    }
                    progress_bar.set_postfix(postfix)
                
                # Log to wandb more frequently in test mode
                if self.global_step % 5 == 0 and self.use_wandb:
                    wandb.log({f'train_test/{k}': v for k, v in metrics.items()}, step=self.global_step)
            
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
                        'kl': f"{metrics['kl_loss']:.4f}",
                        'rollout': f"{metrics['rollout_loss']:.4f}",
                        'kl_w': f"{metrics['kl_weight']:.3f}"
                    }
                    if 'mse_loss' in metrics:
                        postfix['mse'] = f"{metrics['mse_loss']:.4f}"
                    progress_bar.set_postfix(postfix)
                
                # Log to wandb (rank 0 only)
                if self.global_step % 20 == 0 and self.use_wandb and (not self.use_ddp or dist.get_rank() == 0):
                    wandb.log({f'train/{k}': v for k, v in metrics.items()}, step=self.global_step)
                    
                    # Also log training loss vs TFLOPs for plotting
                    if 'total_tflops' in metrics and metrics['total_tflops'] > 0:
                        wandb.log({
                            'tflops/train_loss': metrics['loss'],
                            'tflops/train_recon_loss': metrics['recon_loss'],
                            'tflops/train_kl_loss': metrics['kl_loss'],
                            'tflops/train_rollout_loss': metrics['rollout_loss'],
                            'tflops/x_axis': metrics['total_tflops']
                        }, step=self.global_step)
                
                # Validation at 10% intervals (skip in test mode)
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
        """Evaluate on a dataset."""
        self.model.eval()
        
        epoch_metrics = {}
        num_batches = len(loader)
        update_frequency = max(1, num_batches // 10)  # Update progress 10 times during evaluation
        
        progress_bar = tqdm(loader, desc=f"Evaluating {split}")
        
        for batch_idx, batch in enumerate(progress_bar):
            metrics = self.eval_step(batch)
            
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
                    'kl': f"{metrics['kl_loss']:.4f}",
                    'rollout': f"{metrics['rollout_loss']:.4f}"
                }
                if 'mse_loss' in metrics:
                    postfix['mse'] = f"{metrics['mse_loss']:.4f}"
                if 'action_sensitivity_l2' in metrics:
                    postfix['act_sens'] = f"{metrics['action_sensitivity_l2']:.3f}"
                progress_bar.set_postfix(postfix)
        
        # Average metrics
        num_batches = len(loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            
        # Average across DDP ranks
        if self.use_ddp:
            for key in epoch_metrics:
                epoch_metrics[key] = _ddp_mean(epoch_metrics[key], self.device)
        
        # Log to wandb with TFLOP tracking for plots (rank 0 only)
        if self.use_wandb and (not self.use_ddp or dist.get_rank() == 0):
            log_dict = {f'{split}/{k}': v for k, v in epoch_metrics.items()}
            
            # Add TFLOP-indexed metrics for validation to create plots with TFLOPs on x-axis
            if split == 'val' and 'total_tflops' in epoch_metrics:
                total_tflops = epoch_metrics['total_tflops']
                val_loss = epoch_metrics['loss']
                recon_loss = epoch_metrics['recon_loss']
                kl_loss = epoch_metrics['kl_loss']
                dsnr = epoch_metrics.get('dsnr', 0.0)
                
                # Log metrics with TFLOPs as x-axis (using custom x-axis in wandb)
                # These will create plots with TFLOPs on x-axis when viewed in WandB
                if total_tflops > 0:
                    # Primary metrics vs TFLOPs
                    rollout_loss = epoch_metrics.get('rollout_loss', 0.0)
                    wandb.log({
                        'tflops/val_loss': val_loss,
                        'tflops/val_recon_loss': recon_loss,
                        'tflops/val_kl_loss': kl_loss,
                        'tflops/val_rollout_loss': rollout_loss,
                        'tflops/val_dsnr': dsnr,
                        'tflops/x_axis': total_tflops  # This serves as the x-axis value
                    }, step=self.global_step)
                    
                    # Efficiency metrics
                    log_dict[f'{split}/loss_per_tflop'] = val_loss / total_tflops
                    log_dict[f'{split}/recon_per_tflop'] = recon_loss / total_tflops
            
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
            'config': {
                'latent_dim': model_for_save.latent_dim,
                'action_dim': model_for_save.action_dim,
                'kl_weight': model_for_save.kl_weight
            }
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
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
                logger.info(f"  Total Loss:    {train_metrics['loss']:.6f}")
                logger.info(f"  Recon Loss:    {train_metrics['recon_loss']:.6f}")
                logger.info(f"  KL Loss:       {train_metrics['kl_loss']:.6f}")
                logger.info(f"  MSE Loss:      {train_metrics.get('mse_loss', 'N/A')}")
                logger.info(f"  KL Weight:     {train_metrics['kl_weight']:.4f}")
                logger.info(f"  Learning Rate: {train_metrics['lr']:.6f}")
                
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
                    for n, p in self.model.decoder.named_parameters():
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
                        has_grad = any(key in n for n, p in self.model.decoder.named_parameters() if p.grad is not None and p.grad.norm().item() > 1e-8)
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
    min_context_length: int = 4,
    max_context_length: int = 20,
    context_schedule: str = "linear",
    device: str = "cuda",
    is_ddp: bool = False,
    **kwargs
):
    """
    Train LAM with variable context length curriculum.
    
    Args:
        data_dir: Path to dataset directory
        manifest_path: Path to manifest file
        num_epochs: Total number of epochs
        batch_size: Batch size
        min_context_length: Minimum context length to start with
        max_context_length: Maximum context length to reach
        context_schedule: How to increase context ("linear", "exponential", "step")
        device: Device to train on
        is_ddp: Whether distributed training is enabled
        **kwargs: Additional arguments for model and trainer
    """
    
    def get_context_length(epoch: int) -> int:
        """Get context length for current epoch based on schedule."""
        progress = epoch / num_epochs
        
        if context_schedule == "linear":
            # Linear increase
            length = min_context_length + (max_context_length - min_context_length) * progress
        elif context_schedule == "exponential":
            # Exponential increase
            length = min_context_length * ((max_context_length / min_context_length) ** progress)
        elif context_schedule == "step":
            # Step increase every 25% of training
            steps = [0.25, 0.5, 0.75, 1.0]
            for step in steps:
                if progress <= step:
                    length = min_context_length + (max_context_length - min_context_length) * step
                    break
        else:
            length = max_context_length
        
        return int(length)
    
    # Initialize model with patch-based architecture
    model = LatentActionVAE(
        latent_dim=kwargs.get('latent_dim', 1024),  # D = patch dimension = 1024
        action_dim=kwargs.get('action_dim', 128),
        embed_dim=kwargs.get('embed_dim', 512),
        encoder_depth=kwargs.get('encoder_depth', 3),
        decoder_depth=kwargs.get('decoder_depth', 3),
        encoder_heads=kwargs.get('encoder_heads', 8),
        decoder_heads=kwargs.get('decoder_heads', 8),
        mlp_ratio=kwargs.get('mlp_ratio', 4.0),
        drop_rate=kwargs.get('drop_rate', 0.0),
        attn_drop_rate=kwargs.get('attn_drop_rate', 0.0),
        kl_weight=kwargs.get('kl_weight', 0.1)
    )
    
    # Create initial dataloaders with minimum context length
    current_context = min_context_length
    (train_loader, val_loader, test_loader), (train_sampler, val_sampler, test_sampler) = create_dataloaders(
        data_dir=data_dir,
        manifest_path=manifest_path,
        batch_size=batch_size,
        sequence_length=current_context,
        num_workers=kwargs.get('num_workers', 8),
        return_samplers=True,
        ddp=is_ddp
    )
    
    # Initialize trainer
    trainer = LAMTrainer(
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
        anchor_strategy=kwargs.get('anchor_strategy', 'random')
    )
    
    # Initial validation run to establish baseline (skip in test mode)
    if not kwargs.get('test_mode', False):
        logger.info("Running initial validation...")
        initial_val_metrics = trainer.evaluate(trainer.val_loader, "val")
        logger.info(f"Initial validation metrics: {initial_val_metrics}")
    else:
        logger.info("Test mode: Skipping initial validation")
    
    # Set total training steps for KL annealing (reaches max at final epoch)  
    trainer.total_training_steps = num_epochs * len(train_loader)
    logger.info(f"KL weight will reach max ({trainer.kl_max_weight}) at step {trainer.total_training_steps}")
    
    # Initialize FLOP tracking for compute efficiency analysis
    trainer._initialize_flop_tracking()
    
    # Training loop with curriculum
    for epoch in range(num_epochs):
        # Skip context length updates in test mode
        if not kwargs.get('test_mode', False):
            # Update context length based on schedule
            new_context = get_context_length(epoch)
            
            # Recreate dataloaders if context changed
            if new_context != current_context:
                logger.info(f"Updating context length from {current_context} to {new_context}")
                current_context = new_context
                
                (train_loader, val_loader, test_loader), (train_sampler, val_sampler, test_sampler) = create_dataloaders(
                    data_dir=data_dir,
                    manifest_path=manifest_path,
                    batch_size=batch_size,
                    sequence_length=current_context,
                    num_workers=kwargs.get('num_workers', 8),
                    return_samplers=True,
                    ddp=is_ddp
                )
                
                # Update trainer's dataloaders
                trainer.train_loader = train_loader
                trainer.val_loader = val_loader
                trainer.test_loader = test_loader
                
                # Update total training steps since loader size changed
                remaining_epochs = num_epochs - epoch
                trainer.total_training_steps = trainer.global_step + (remaining_epochs * len(train_loader))
                logger.info(f"Updated KL annealing: will reach max at step {trainer.total_training_steps}")
        
        # Set epoch for distributed sampler
        if is_ddp:
            if train_sampler:
                train_sampler.set_epoch(epoch)
            if val_sampler:
                val_sampler.set_epoch(epoch)
            if test_sampler:
                test_sampler.set_epoch(epoch)
        
        # Train for one epoch
        trainer.epoch = epoch
        train_metrics = trainer.train_epoch()
        logger.info(f"Epoch {epoch} (context={current_context}) - Train: {train_metrics}")
        
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
    parser.add_argument("--min_context", type=int, default=4, help="Minimum context length")
    parser.add_argument("--max_context", type=int, default=20, help="Maximum context length")
    parser.add_argument("--context_schedule", type=str, default="linear", 
                       choices=["linear", "exponential", "step"], help="Context length schedule")

    # DDP arguments
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for DistributedDataParallel')
    parser.add_argument("--dist_backend", default=None, help="nccl|gloo")

    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--kl_weight", type=float, default=0.1, help="KL divergence weight")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--scheduler", action="store_true", help="Enable manual LR scheduling with warmup and cosine annealing")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum LR as ratio of base LR (default: 0.1 = 10%% of base)")

    # Rollout loss arguments
    parser.add_argument("--rollout_horizon", type=int, default=2, help="Number of steps for rollout (1 = no rollout; 2-3 typical)")
    parser.add_argument("--rollout_weight", type=float, default=1.0, help="Weight for rollout loss component")
    parser.add_argument("--rollout_prob", type=float, default=1.0, help="Probability of computing rollout loss (0-1)")
    parser.add_argument("--detach_rollout_first", action="store_true", help="Detach first predicted state before rollout for gradient stability")
    parser.add_argument("--anchor_strategy", type=str, default="random", choices=["random", "last"], help="Rollout anchor strategy")

    # Model architecture arguments
    parser.add_argument("--latent_dim", type=int, default=1024, help="Patch token dimension (default: 1024)")
    parser.add_argument("--encoder_depth", type=int, default=3, help="Number of encoder transformer layers")
    parser.add_argument("--decoder_depth", type=int, default=3, help="Number of decoder transformer layers") 
    parser.add_argument("--encoder_heads", type=int, default=8, help="Number of encoder attention heads")
    parser.add_argument("--decoder_heads", type=int, default=8, help="Number of decoder attention heads")
    parser.add_argument("--embed_dim", type=int, default=512, help="Transformer embedding dimension")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP hidden dimension multiplier")
    parser.add_argument("--drop_rate", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--attn_drop_rate", type=float, default=0.0, help="Attention dropout rate")
    
    args = parser.parse_args()

    # --- DDP Initialization ---
    is_ddp = args.local_rank != -1
    if is_ddp:
        dist.init_process_group(backend=args.dist_backend or ('nccl' if torch.cuda.is_available() else 'gloo'))
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
        args.device = f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu'

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
        args.kl_weight = 0.0
        args.drop_rate = 0.0
        args.attn_drop_rate = 0.0
        print(f"  - KL weight set to 0.0")
        print(f"  - Dropout rates set to 0.0")
    
    # Train model
    try:
        model = train_with_variable_context(
            data_dir=args.data_dir,
            manifest_path=args.manifest_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            min_context_length=args.min_context,
            max_context_length=args.max_context,
            context_schedule=args.context_schedule,
            device=args.device,
            mixed_precision=args.mixed_precision,
            learning_rate=args.learning_rate,
            kl_weight=args.kl_weight,
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
            # Model architecture parameters
            latent_dim=args.latent_dim,
            encoder_depth=args.encoder_depth,
            decoder_depth=args.decoder_depth,
            encoder_heads=args.encoder_heads,
            decoder_heads=args.decoder_heads,
            embed_dim=args.embed_dim,
            mlp_ratio=args.mlp_ratio,
            drop_rate=args.drop_rate,
            attn_drop_rate=args.attn_drop_rate,
            is_ddp=is_ddp
        )
    finally:
        if is_ddp:
            dist.destroy_process_group()
