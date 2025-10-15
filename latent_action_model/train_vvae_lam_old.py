"""
Training script for VVAE Latent Action Model.
Simplified training loop for VVAELatentActionVAE with MSE loss.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from latent_action_model.vqvae import VVAELatentActionVQVAE
from latent_action_model.dataloader_vvae import create_vvae_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VVAELAMTrainer:
    """Trainer for VVAE Latent Action Model (VQ-VAE variant)."""

    def __init__(
        self,
        model: VVAELatentActionVQVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        kl_annealing_steps: int = 10000,
        kl_min_weight: float = 0.0,
        kl_max_weight: float = 1.0,
        checkpoint_dir: str = "./checkpoints_vvae_lam",
        save_every: int = 5,
        eval_every: int = 1,
        use_wandb: bool = True,
        wandb_project: str = "vvae-lam",
        wandb_run_name: str = None,
        rollout_horizon: int = 2,
        rollout_weight: float = 1.0,
        rollout_prob: float = 1.0,
        detach_rollout_first: bool = True,
        anchor_strategy: str = "random",
        test_mode: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.test_mode = test_mode

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.kl_annealing_steps = kl_annealing_steps
        self.kl_min_weight = kl_min_weight
        self.kl_max_weight = kl_max_weight

        # Rollout parameters
        self.rollout_horizon = rollout_horizon
        self.rollout_weight = rollout_weight
        self.rollout_prob = rollout_prob
        self.detach_rollout_first = detach_rollout_first
        self.anchor_strategy = anchor_strategy

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.eval_every = eval_every

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01)

        # Tracking
        self.current_step = 0
        self.current_epoch = 0
        self.use_wandb = use_wandb

        # Test mode setup
        if self.test_mode:
            logger.info("=" * 60)
            logger.info("TEST MODE ACTIVATED: Using only 1 batch for overfitting test")
            logger.info("Test mode overrides:")
            logger.info("  - Rollout: disabled")
            logger.info("  - Validation: disabled")
            logger.info("=" * 60)

            # Override settings for test mode
            self.rollout_horizon = 1
            self.rollout_weight = 0.0
            self.rollout_prob = 0.0

            # Store fixed batch
            self.fixed_batch = next(iter(train_loader))
            logger.info(f"Fixed batch shape: {self.fixed_batch['sequence'].shape}")

        if use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name or f"vvae_lam_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "learning_rate": learning_rate,
                    "warmup_steps": warmup_steps,
                    "kl_annealing_steps": kl_annealing_steps,
                    "rollout_horizon": rollout_horizon,
                    "rollout_weight": rollout_weight,
                    "test_mode": test_mode,
                    "model_config": model.config,
                }
            )

    def get_lr_multiplier(self):
        """Warmup learning rate schedule."""
        if self.current_step < self.warmup_steps:
            return self.current_step / max(1, self.warmup_steps)
        return 1.0

    def get_kl_weight(self):
        """KL annealing schedule."""
        if self.current_step < self.kl_annealing_steps:
            progress = self.current_step / self.kl_annealing_steps
            return self.kl_min_weight + (self.kl_max_weight - self.kl_min_weight) * progress
        return self.kl_max_weight

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            "loss": 0.0,
            "recon_loss": 0.0,
            "mae_loss": 0.0,
            "mse_loss": 0.0,
            "vq_loss": 0.0,
            "codebook_loss": 0.0,
            "commitment_loss": 0.0,
            "rollout_loss": 0.0,
            "codebook_usage": 0.0,
        }
        num_batches = 0

        # Use fixed batch in test mode, otherwise iterate over dataloader
        if self.test_mode:
            batches = [self.fixed_batch]
            pbar = tqdm(batches, desc=f"Epoch {self.current_epoch} (TEST MODE - 1 batch)")
        else:
            batches = self.train_loader
            pbar = tqdm(batches, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass with rollout
            kl_weight = self.get_kl_weight()
            losses = self.model.compute_loss(
                batch,
                beta_schedule=kl_weight,
                rollout_horizon=self.rollout_horizon,
                rollout_weight=self.rollout_weight,
                rollout_prob=self.rollout_prob,
                detach_rollout_first=self.detach_rollout_first,
                anchor_strategy=self.anchor_strategy
            )

            loss = losses["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Optimizer step with warmup
            lr_mult = self.get_lr_multiplier()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate * lr_mult

            self.optimizer.step()

            # Accumulate metrics
            for key in epoch_metrics:
                if isinstance(losses[key], torch.Tensor):
                    epoch_metrics[key] += losses[key].item()
                else:
                    epoch_metrics[key] += losses[key]  # Already a scalar (e.g., codebook_usage)
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['loss'].item():.4f}",
                "mse": f"{losses['mse_loss'].item():.4f}",
                "mae": f"{losses['mae_loss'].item():.4f}",
                "vq": f"{losses['vq_loss'].item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "train/loss": losses["loss"].item(),
                    "train/recon_loss": losses["recon_loss"].item(),
                    "train/mae_loss": losses["mae_loss"].item(),
                    "train/mse_loss": losses["mse_loss"].item(),
                    "train/vq_loss": losses["vq_loss"].item(),
                    "train/codebook_loss": losses["codebook_loss"].item(),
                    "train/commitment_loss": losses["commitment_loss"].item(),
                    "train/rollout_loss": losses["rollout_loss"].item(),
                    "train/codebook_usage": losses["codebook_usage"],  # Already an int
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm.item(),
                    "step": self.current_step,
                })

            self.current_step += 1

        # Average epoch metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics

    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        self.model.eval()
        val_metrics = {
            "loss": 0.0,
            "recon_loss": 0.0,
            "mae_loss": 0.0,
            "mse_loss": 0.0,
            "vq_loss": 0.0,
            "codebook_loss": 0.0,
            "commitment_loss": 0.0,
            "rollout_loss": 0.0,
            "codebook_usage": 0.0,
        }
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass (no rollout during validation for speed)
            kl_weight = self.get_kl_weight()
            losses = self.model.compute_loss(
                batch,
                beta_schedule=kl_weight,
                rollout_horizon=1,  # Disable rollout for validation
                rollout_weight=0.0,
                rollout_prob=0.0
            )

            for key in val_metrics:
                if isinstance(losses[key], torch.Tensor):
                    val_metrics[key] += losses[key].item()
                else:
                    val_metrics[key] += losses[key]  # Already a scalar (e.g., codebook_usage)
            num_batches += 1

        # Average validation metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "val/loss": val_metrics["loss"],
                "val/recon_loss": val_metrics["recon_loss"],
                "val/mae_loss": val_metrics["mae_loss"],
                "val/mse_loss": val_metrics["mse_loss"],
                "val/vq_loss": val_metrics["vq_loss"],
                "val/codebook_loss": val_metrics["codebook_loss"],
                "val/commitment_loss": val_metrics["commitment_loss"],
                "val/codebook_usage": val_metrics["codebook_usage"],
                "epoch": self.current_epoch,
            })

        return val_metrics

    def save_checkpoint(self, filename: str = None):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        torch.save({
            "epoch": self.current_epoch,
            "step": self.current_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config,
        }, checkpoint_path)

        # Also save config as JSON
        config_path = self.checkpoint_dir / "model_config.json"
        with open(config_path, "w") as f:
            json.dump(self.model.config, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def train(self, num_epochs: int):
        """Main training loop."""
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                       f"MSE: {train_metrics['mse_loss']:.4f}, MAE: {train_metrics['mae_loss']:.4f}")

            # Validate (skip in test mode)
            if self.val_loader and (epoch + 1) % self.eval_every == 0 and not self.test_mode:
                val_metrics = self.validate()
                logger.info(f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}, "
                           f"MSE: {val_metrics['mse_loss']:.4f}, MAE: {val_metrics['mae_loss']:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint()

        # Save final checkpoint
        self.save_checkpoint("final_checkpoint.pt")
        logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train VVAE Latent Action Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to VVAE h5 files")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to manifest JSON file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--sequence_length", type=int, default=8, help="Sequence length in latent frames")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--codebook_dim", type=int, default=128, help="Codebook dimension")
    parser.add_argument("--num_embeddings", type=int, default=12, help="Codebook size")
    parser.add_argument("--embed_dim", type=int, default=512, help="Transformer embedding dimension")
    parser.add_argument("--encoder_depth", type=int, default=3, help="Encoder depth")
    parser.add_argument("--decoder_depth", type=int, default=3, help="Decoder depth")
    parser.add_argument("--commitment_weight", type=float, default=0.25, help="VQ-VAE commitment weight")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_vvae_lam", help="Checkpoint directory")
    parser.add_argument("--wandb_project", type=str, default="vvae-lam", help="Weights & Biases project name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--rollout_horizon", type=int, default=2, help="Rollout horizon")
    parser.add_argument("--rollout_weight", type=float, default=1.0, help="Rollout loss weight")
    parser.add_argument("--test_mode", action="store_true", help="Test mode: use only 1 batch for overfitting test")

    args = parser.parse_args()

    # Create model
    logger.info("Creating VVAELatentActionVQVAE model...")
    model = VVAELatentActionVQVAE(
        codebook_dim=args.codebook_dim,
        num_embeddings=args.num_embeddings,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        encoder_heads=8,
        decoder_heads=8,
        commitment_weight=args.commitment_weight,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_vvae_dataloaders(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        stride_train=1,
        stride_val=2,
        num_workers=8
    )

    # Create trainer
    trainer = VVAELAMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        rollout_horizon=args.rollout_horizon,
        rollout_weight=args.rollout_weight,
        test_mode=args.test_mode,
    )

    # Train
    if args.test_mode:
        logger.info("\n" + "="*60)
        logger.info("RUNNING OVERFIT TEST")
        logger.info("="*60)
        logger.info("Target: MSE and MAE should decrease to near zero")
        logger.info("Success criteria: MSE < 0.01, MAE < 0.05")
        logger.info("="*60 + "\n")

    trainer.train(num_epochs=args.num_epochs)


if __name__ == "__main__":
    main()
