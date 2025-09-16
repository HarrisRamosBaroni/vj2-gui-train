import argparse
import torch
import torch.nn as nn
import wandb
import time
from pathlib import Path
from typing import Dict, Any

from .spline_decoder import SplineDecoder
from .gru_decoder import GRUDecoder
from latent_action_model.vae import LatentActionVAE, load_lam_config
from training.dataloader import init_preprocessed_data_loader
from src.utils.logging import get_logger
from .validators.decoder_visualizer import DecoderVisualizer

# Model registry to select decoder via CLI
DECODER_MODEL_REGISTRY = {
    "spline": SplineDecoder,
    "gru": GRUDecoder,
}

class CompositeLoss(nn.Module):
    """Computes the composite loss for the action decoder."""
    def __init__(self, alpha_grad: float, beta_tv: float):
        super().__init__()
        self.alpha_grad = alpha_grad
        self.beta_tv = beta_tv
        self.recon_loss_fn = nn.HuberLoss()

    def forward(self, pred_traj: torch.Tensor, target_traj: torch.Tensor):
        # Huber loss on all dimensions
        recon_loss = self.recon_loss_fn(pred_traj, target_traj)

        # Gradient matching loss on x, y dimensions
        pred_grad = torch.diff(pred_traj[..., :2], dim=1)
        target_grad = torch.diff(target_traj[..., :2], dim=1)
        grad_loss = nn.functional.mse_loss(pred_grad, target_grad)

        # Total Variation loss on touch state dimension
        tv_loss = torch.mean(torch.abs(torch.diff(pred_traj[..., 2], dim=1)))
        
        total_loss = recon_loss + self.alpha_grad * grad_loss + self.beta_tv * tv_loss
        return total_loss, recon_loss, grad_loss, tv_loss

class DecoderTrainer:
    """Encapsulates the training and validation logic for the action decoder."""
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        
        self.logger = get_logger()
        
        # --- Models ---
        # Load pretrained LAM
        print(f"Loading LAM from checkpoint: {args.lam_ckpt_path}")
        lam_checkpoint = torch.load(args.lam_ckpt_path, map_location=self.device, weights_only=True)
        lam_config = load_lam_config(args.lam_config)
        # lam_config = lam_checkpoint['config']
        # lam_config = {
        #     "latent_dim": 1024,   # patch_dim
        #     "action_dim": 128,
        #     "embed_dim": 512,
        #     "encoder_depth": 2,
        #     "decoder_depth": 2,
        #     "encoder_heads": 8,
        #     "decoder_heads": 8,
        #     "kl_weight": 0.001,  # <- training value
        # }
        self.lam_model = LatentActionVAE(**lam_config).to(self.device)
        self.lam_model.load_state_dict(lam_checkpoint['model_state_dict'])
        self.lam_model.eval()
        print("LAM model loaded successfully.")
        
        # Instantiate the selected decoder model
        model_class = DECODER_MODEL_REGISTRY[args.decoder_model_type]
        
        # Prepare model arguments
        model_args = {
            "z_dim": lam_config["action_dim"],
            "T_steps": 250,
        }
        if args.decoder_model_type == "spline":
            model_args.update({
                "k_points": args.spline_k_points,
                "hidden_dim": args.spline_hidden_dim,
            })
        elif args.decoder_model_type == "gru":
            model_args.update({
                "hidden_dim": args.gru_hidden_dim,
                "n_layers": args.gru_n_layers,
            })
            
        self.decoder = model_class(**model_args).to(self.device)
        print(f"Decoder model ({args.decoder_model_type}) instantiated successfully.")

        # --- Loss and Optimizer ---
        self.loss_fn = CompositeLoss(args.alpha_grad_loss, args.beta_tv_loss).to(self.device)
        self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=args.learning_rate)
        
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

        # --- Checkpointing and Logging ---
        self.run_dir = Path(args.output_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.decoder_model_type}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.validation_cycle_count = 0
        
        if args.wandb_project:
            run_name = args.wandb_run_name if args.wandb_run_name else self.run_dir.name
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
            wandb.watch(self.decoder, log="all", log_freq=100)
            self.logger.info(f"W&B initialized for run: {run_name}")

        self.visualizer = None
        if args.visualization_freq > 0:
            self.visualizer = DecoderVisualizer(frequency=args.visualization_freq)

    def train_step(self, sample):
        visual_embeddings, ground_truth_actions = sample
        visual_embeddings = visual_embeddings.to(self.device, non_blocking=True)
        ground_truth_actions = ground_truth_actions.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad()
        
        with torch.no_grad():
            # The LAM expects per-transition actions, so we get T-1 actions for T frames.
            # We are interested in the action that leads to the final state of a sequence.
            # Here, we assume the dataloader provides sequences where the last action is the one to be decoded.
            mu, logvar = self.lam_model.encode(visual_embeddings)
            
            # We take the latent corresponding to the last transition in the sequence.
            z = self.lam_model.reparameterize(mu, logvar)[:, -1, :]
            
        pred_traj = self.decoder(z)
        
        # Ensure target trajectory has the same length as prediction
        T_pred = pred_traj.shape[1]
        if ground_truth_actions.shape[1] < T_pred:
            raise ValueError("Ground truth action sequence is shorter than predicted trajectory.")
        
        # For simplicity, we match from the start of the sequence.
        target_traj = ground_truth_actions[:, :T_pred, :]
        
        loss, recon, grad, tv = self.loss_fn(pred_traj, target_traj)
        
        loss.backward()
        self.optimizer.step()
        
        if self.args.wandb_project:
            wandb.log({
                "train/loss": loss.item(),
                "train/recon_loss": recon.item(),
                "train/grad_loss": grad.item(),
                "train/tv_loss": tv.item(),
                "global_step": self.global_step,
            })
            
        return loss.item(), recon.item(), grad.item(), tv.item()

    def validate(self):
        self.decoder.eval()
        total_loss, total_recon, total_grad, total_tv = 0, 0, 0, 0
        
        with torch.no_grad():
            for sample in self.val_loader:
                visual_embeddings, ground_truth_actions = sample
                visual_embeddings = visual_embeddings.to(self.device, non_blocking=True)
                ground_truth_actions = ground_truth_actions.to(self.device, non_blocking=True)
                
                mu, logvar = self.lam_model.encode(visual_embeddings)
                z = self.lam_model.reparameterize(mu, logvar)[:, -1, :]
                
                pred_traj = self.decoder(z)
                
                T_pred = pred_traj.shape[1]
                target_traj = ground_truth_actions[:, :T_pred, :]

                loss, recon, grad, tv = self.loss_fn(pred_traj, target_traj)
                
                total_loss += loss.item()
                total_recon += recon.item()
                total_grad += grad.item()
                total_tv += tv.item()

        num_batches = len(self.val_loader) if self.val_loader else 1
        avg_loss = total_loss / num_batches
        
        if self.args.wandb_project:
            wandb.log({
                "val/loss": avg_loss,
                "val/recon_loss": total_recon / num_batches,
                "val/grad_loss": total_grad / num_batches,
                "val/tv_loss": total_tv / num_batches,
                "global_step": self.global_step,
            })
        
        self.validation_cycle_count += 1
        if self.visualizer and self.visualizer.should_run(self.validation_cycle_count):
            self.logger.info("Running DecoderVisualizer...")
            viz_metrics = self.visualizer.run(
                model=self.decoder,
                validation_loader=self.val_loader,
                lam_model=self.lam_model,
                device=self.device,
                global_step=self.global_step
            )
            if self.args.wandb_project and viz_metrics:
                wandb.log(viz_metrics)
        
        self.decoder.train()
        return avg_loss

    def training_loop(self):
        self.logger.info("Starting training loop...")
        for epoch in range(self.args.num_epochs):
            self.decoder.train()
            
            for i, sample in enumerate(self.train_loader):
                self.global_step += 1
                loss, recon, grad, tv = self.train_step(sample)
                
                if i % 20 == 0: # Log every 20 steps
                    self.logger.info(f"[Epoch {epoch+1}/{self.args.num_epochs} | Step {self.global_step}] "
                                     f"Loss: {loss:.4f} | Recon: {recon:.4f} | Grad: {grad:.4f} | TV: {tv:.4f}")

            val_loss = self.validate()
            self.logger.info(f"--- Epoch {epoch+1} Validation --- Avg Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.logger.info(f"🏆 New best validation loss: {val_loss:.4f}. Saving checkpoint.")
                self._save_checkpoint("best", val_loss)
            
            # Save a checkpoint periodically
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(f"epoch_{epoch+1}", val_loss)
    
    def _save_checkpoint(self, tag: str, val_loss: float = None):
        """Saves a checkpoint of the model and optimizer states."""
        
        checkpoint_path = self.run_dir / f"decoder_{tag}.pt"
        
        # Get model config and state
        model_config = self.decoder.get_config()
        model_state = self.decoder.state_dict()
        
        # Prepare checkpoint dictionary
        checkpoint = {
            "config": model_config,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
        }
        if val_loss is not None:
            checkpoint["val_loss"] = val_loss

        try:
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"💾 Checkpoint saved to {checkpoint_path}")

            # Also create/update a 'latest.pt' symlink for convenience
            latest_symlink = self.run_dir / "latest.pt"
            if latest_symlink.exists():
                latest_symlink.unlink()
            latest_symlink.symlink_to(checkpoint_path.name)

        except Exception as e:
            self.logger.error(f"❌ Failed to save checkpoint: {e}")

def get_args():
    parser = argparse.ArgumentParser(description="Train a Latent Action Decoder")
    # ... (arguments as defined before, keeping them here)
    parser.add_argument("--processed_data_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--lam_ckpt_path", type=str, required=True)
    parser.add_argument("--lam_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="decoder_checkpoints")
    parser.add_argument("--decoder_model_type", type=str, required=True, choices=["spline", "gru"])
    parser.add_argument("--spline_k_points", type=int, default=12)
    parser.add_argument("--spline_hidden_dim", type=int, default=512)
    parser.add_argument("--gru_hidden_dim", type=int, default=512)
    parser.add_argument("--gru_n_layers", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--alpha_grad_loss", type=float, default=0.5)
    parser.add_argument("--beta_tv_loss", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_project", type=str, default="latent-action-decoder")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--visualization_freq", type=int, default=5, help="Run visualizer every N validation cycles. Set to 0 to disable.")
    return parser.parse_args()

def main():
    args = get_args()
    try:
        trainer = DecoderTrainer(args)
        trainer.training_loop()
        print("Training finished successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        # Optionally, re-raise or handle cleanup
        raise

if __name__ == "__main__":
    main()
