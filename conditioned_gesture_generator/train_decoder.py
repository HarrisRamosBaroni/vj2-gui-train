import argparse
import torch
import torch.nn as nn
import wandb
import time
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from .latent_action_decoder.spline_decoder import SplineDecoder
from .latent_action_decoder.gru_decoder import GRUDecoder
from .latent_action_decoder.hybrid_gated_decoder import HybridGatedDecoder
from .latent_action_decoder.cnn_quantizing_decoder_lightweight import CNNQuantizingDecoderLightweight
from .latent_action_decoder.cnn_quantizing_decoder import CNNQuantizingDecoder
from .latent_action_decoder.cvae import ActionCVAE
from .latent_action_decoder.dtw_cvae_decoder import DTWCVAEDecoder
from .latent_action_decoder.samplewise_dtw_cvae_decoder import SamplewiseDTWCVAEDecoder
from latent_action_model.vae import LatentActionVAE, load_lam_config
from training.dataloader import init_preprocessed_data_loader
from src.utils.logging import get_logger
from .validators.decoder_visualizer import DecoderVisualizer

# Model registry to select decoder via CLI
DECODER_MODEL_REGISTRY = {
    "spline": SplineDecoder,
    "gru": GRUDecoder,
    "hybrid_gated": HybridGatedDecoder,
    "cnn_quantizing_lightweight": CNNQuantizingDecoderLightweight,
    "cnn_quantizing": CNNQuantizingDecoder,
    "cvae": ActionCVAE,
    "dtw_cvae": DTWCVAEDecoder,
    "samplewise_dtw_cvae": SamplewiseDTWCVAEDecoder,
}

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
        elif args.decoder_model_type == "hybrid_gated":
            model_args.update({
                "k_points": args.hybrid_k_points,
                "hidden_dim": args.hybrid_hidden_dim,
                "spline_interpolator": args.hybrid_spline_interpolator,
                "loss_config": {
                    "touch_pos_weight": args.hybrid_touch_pos_weight,
                }
            })
        elif args.decoder_model_type == "cnn_quantizing_lightweight":
            model_args.update({
                "k_classes": args.cnn_k_classes,
                "hidden_dim": args.cnn_hidden_dim,
            })
        elif args.decoder_model_type == "cnn_quantizing":
            model_args.update({
                "k_classes": args.cnn_k_classes,
                "hidden_dim": args.cnn_hidden_dim,
                "dropout": args.cnn_dropout,
            })
        elif args.decoder_model_type in ["dtw_cvae", "samplewise_dtw_cvae"]:
            loss_cfg = {
                "w_kl": args.dtw_cvae_w_kl,
                "w_aux": args.dtw_cvae_w_aux,
                "w_trans": args.dtw_cvae_w_trans,
                "kl_free_bits": args.dtw_cvae_kl_free_bits,
                "dist_gamma": args.dtw_cvae_dist_gamma,
                "focal_gamma": args.dtw_cvae_focal_gamma,
                "focal_alpha": args.dtw_cvae_focal_alpha,
                "sdtw_gamma": args.dtw_cvae_sdtw_gamma,
            }
            if args.decoder_model_type == "samplewise_dtw_cvae":
                loss_cfg["w_recon"] = args.dtw_cvae_w_recon

            model_args.update({
                "s_dim": lam_config["latent_dim"],
                "style_dim": args.dtw_cvae_style_dim,
                "K_ctrl_pts": args.dtw_cvae_K_ctrl_pts,
                "nhead": args.dtw_cvae_nhead,
                "num_decoder_layers": args.dtw_cvae_num_decoder_layers,
                "num_s_tokens": 256, # Hardcoded to match legacy model
                "loss_cfg": loss_cfg
            })

        if args.decoder_model_type == "cvae":
            encoder_cfg = {
                "action_dim": 3,
                "z_dim": lam_config["action_dim"],
                "s_dim": lam_config["latent_dim"], # This is the patch dim
                "style_dim": args.cvae_style_dim,
                "model_dim": args.cvae_model_dim,
                "nhead": args.cvae_nhead,
                "num_layers": args.cvae_encoder_layers,
                "num_s_tokens": 256, # Number of visual patches, assumed to be 256
            }
            decoder_cfg = {
                "z_dim": lam_config["action_dim"],
                "s_dim": lam_config["latent_dim"],
                "style_dim": args.cvae_style_dim,
                "model_dim": args.cvae_model_dim,
                "nhead": args.cvae_nhead,
                "num_layers": args.cvae_decoder_layers,
                "T": 250,
            }
            loss_cfg = {
                "kl_weight": args.cvae_kl_weight,
                "touch_pos_weight": args.cvae_touch_pos_weight,
            }
            # CVAE has a different signature, so we overwrite model_args completely
            model_args = {
                "encoder_cfg": encoder_cfg,
                "decoder_cfg": decoder_cfg,
                "loss_cfg": loss_cfg,
            }
            
        self.decoder = model_class(**model_args).to(self.device)
        print(f"Decoder model ({args.decoder_model_type}) instantiated successfully.")

        # --- Loss and Optimizer ---
        self.loss_fn = self.decoder.get_loss_function().to(self.device)
        self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=args.learning_rate)
        print(f"Loss function {self.loss_fn.__class__.__name__} instantiated successfully.")
        
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
        
        self.test_mode = args.test_mode
        self.fixed_batch = None
        
        if self.test_mode:
            self.logger.info("=" * 60)
            self.logger.info("TEST MODE ACTIVATED: Using only 1 batch for overfitting test")
            self.logger.info("Test mode defaults:")
            self.logger.info("  - Dropout: 0.0 (disabled)")
            self.logger.info("  - Validation: Disabled")
            self.logger.info("=" * 60)
            
            # Disable dropout in the model
            self._disable_dropout()
            
            # Store a single batch for repeated training
            self.fixed_batch = next(iter(self.train_loader))
            # Log batch info
            self.logger.info(f"Fixed batch shapes:")
            visual_embeddings, ground_truth_actions = self.fixed_batch
            self.logger.info(f"  visual_embeddings: {visual_embeddings.shape}")
            self.logger.info(f"  ground_truth_actions: {ground_truth_actions.shape}")

    def _disable_dropout(self):
        """Disable all dropout in the model for test mode."""
        for module in self.decoder.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.logger.info("Disabled all dropout layers for test mode")

    def train_step(self, sample):
        visual_embeddings, ground_truth_actions = sample
        visual_embeddings = visual_embeddings.to(self.device, non_blocking=True) # (B, T, N, D)
        ground_truth_actions = ground_truth_actions.to(self.device, non_blocking=True) # (B, T-1, Traj, A)

        # Select the T-1 states corresponding to the T-1 actions
        num_actions = ground_truth_actions.size(1)
        state_embeddings = visual_embeddings[:, :num_actions, :, :]
        
        target_traj = ground_truth_actions.reshape(-1, ground_truth_actions.size(2), ground_truth_actions.size(3))
        
        self.optimizer.zero_grad()
        
        with torch.no_grad():
            # The LAM requires T states to produce T-1 actions, so we pass the full visual_embeddings
            mu, logvar = self.lam_model.encode(visual_embeddings)
            z = self.lam_model.reparameterize(mu, logvar)
            z = z.reshape(-1, z.size(2))  # [B*(T-1), A]
            
        # Prepare kwargs for models that need more than just z
        model_kwargs = {}

        if getattr(self.decoder, 'forward_requires_action', False):
            model_kwargs['A'] = target_traj
        
        if getattr(self.decoder, 'forward_requires_state', False):
            # Use the same sliced and flattened states for 's'
            s = state_embeddings.reshape(-1, state_embeddings.size(2), state_embeddings.size(3))
            model_kwargs['s'] = s
            
        # --- KL Annealing for DTW-CVAE ---
        if self.args.decoder_model_type in ["dtw_cvae", "samplewise_dtw_cvae"]:
            # Calculate current KL weight based on linear schedule
            base_w_kl = self.args.dtw_cvae_w_kl
            anneal_ratio = min(1.0, self.global_step / self.args.kl_anneal_steps)
            current_w_kl = base_w_kl * anneal_ratio
            
            # Update the w_kl attribute in the loss function instance
            self.loss_fn.w_kl = current_w_kl
            
            # Log the dynamic weight to wandb if enabled
            if self.args.wandb_project:
                wandb.log({"train/dynamic_w_kl": current_w_kl, "global_step": self.global_step})

        model_output = self.decoder(z, **model_kwargs)
        
        loss_components = self.loss_fn(model_output, target_traj)
        loss = loss_components["total_loss"]
        
        loss.backward()
        self.optimizer.step()
        
        if self.args.wandb_project:
            log_dict = {f"train/{k}": v.item() + 1000 for k, v in loss_components.items()}
            # Add a large offset to loss values for logging to avoid W&B plotting issues with negative starting values.
            # The '_boosted' suffix indicates that the true value is (logged_value - 1000).
            log_dict = {f"train/{k}_boosted": v.item() + 1000 for k, v in loss_components.items()}
            log_dict["global_step"] = self.global_step
            wandb.log(log_dict)
            
        return loss_components

    def validate(self):
        self.decoder.eval()
        total_loss = 0
        
        with torch.no_grad():
            for sample in self.val_loader:
                visual_embeddings, ground_truth_actions = sample
                visual_embeddings = visual_embeddings.to(self.device, non_blocking=True)
                ground_truth_actions = ground_truth_actions.to(self.device, non_blocking=True)

                num_actions = ground_truth_actions.size(1)
                state_embeddings = visual_embeddings[:, :num_actions, :, :]

                target_traj = ground_truth_actions.reshape(-1, ground_truth_actions.size(2), ground_truth_actions.size(3))
                
                # The LAM requires T states to produce T-1 actions, so we pass the full visual_embeddings
                mu, logvar = self.lam_model.encode(visual_embeddings)
                z = self.lam_model.reparameterize(mu, logvar)
                z = z.reshape(-1, z.size(2))
                
                model_kwargs = {}
                if getattr(self.decoder, 'forward_requires_action', False):
                    model_kwargs['A'] = target_traj
                if getattr(self.decoder, 'forward_requires_state', False):
                    s = state_embeddings.reshape(-1, state_embeddings.size(2), state_embeddings.size(3))
                    model_kwargs['s'] = s

                model_output = self.decoder(z, **model_kwargs)

                loss_components = self.loss_fn(model_output, target_traj)
                
                total_loss += loss_components["total_loss"].item()
                # Note: Accumulating other losses would require more complex handling
                # if loss components are not guaranteed to exist across all models.
                # For now, we focus on the total loss for validation reporting here.

        num_batches = len(self.val_loader) if self.val_loader else 1
        avg_loss = total_loss / num_batches
        
        # In a real scenario, you would dynamically log all components from loss_components
        # similar to the training loop, but averaged over the validation set.
        if self.args.wandb_project:
            wandb.log({
                "val/loss": avg_loss,
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

    def _run_test_mode_visualization(self, epoch: int):
        """Runs the visualizer on the fixed batch in test mode."""
        if not self.visualizer:
            return

        self.logger.info(f"Running DecoderVisualizer in test mode for epoch {epoch}...")
        
        # The visualizer expects a dataloader, so we wrap the fixed batch in a simple list.
        # This is a bit of a hack, but avoids modifying the visualizer's interface.
        test_loader_mock = [self.fixed_batch]
        
        viz_metrics = self.visualizer.run(
            model=self.decoder,
            validation_loader=test_loader_mock,
            lam_model=self.lam_model,
            device=self.device,
            global_step=self.global_step
        )
        
        if self.args.wandb_project and viz_metrics:
            # Add a 'test/' prefix to distinguish from validation plots
            test_viz_metrics = {f"test/{k.split('/')[-1]}": v for k, v in viz_metrics.items()}
            test_viz_metrics["global_step"] = self.global_step
            wandb.log(test_viz_metrics)
        
        self.logger.info("Test mode visualization complete.")

    def _gradient_sanity_check(self):
        """Check gradients to ensure they're flowing properly."""
        self.logger.info("\n=== Gradient Sanity Check ===")
        no_grad_params = []
        for n, p in self.decoder.named_parameters():
            if p.grad is None:
                no_grad_params.append(n)
            elif p.grad.norm().item() < 1e-8:
                self.logger.info(f"  ⚠️  Very small gradient: {n} (norm={p.grad.norm().item():.2e})")
        
        if no_grad_params:
            self.logger.warning("  ⚠️  Parameters with NO gradients:")
            for n in no_grad_params:
                self.logger.warning(f"    - {n}")
        else:
            self.logger.info("  ✅ All parameters have gradients!")

    def training_loop(self):
        self.logger.info("Starting training loop...")
        for epoch in range(self.args.num_epochs):
            self.decoder.train()

            if self.test_mode:
                # Test mode: use only the fixed batch multiple times
                num_steps = 100  # Fixed number of steps per epoch in test mode
                self.logger.info(f"Test mode: Running {num_steps} steps with the same batch")
                
                progress_bar = tqdm(range(num_steps), desc=f"Test Epoch {epoch+1} (Fixed Batch)")
                
                for step in progress_bar:
                    self.global_step += 1
                    loss_components = self.train_step(self.fixed_batch)
                    
                    # Update progress bar
                    if step % 10 == 0:
                        postfix = {k: f"{v.item():.4f}" for k, v in loss_components.items()}
                        progress_bar.set_postfix(postfix)

                # Enhanced logging for test mode
                self.logger.info("=" * 50)
                self.logger.info(f"TEST MODE - Epoch {epoch} Results:")
                for k, v in loss_components.items():
                    self.logger.info(f"  {k}: {v.item():.6f}")

                # Check if loss is decreasing
                if epoch > 0 and hasattr(self, 'prev_loss'):
                    loss_change = loss_components["total_loss"].item() - self.prev_loss
                    if loss_change < -1e-6:
                        self.logger.info(f"  ✅ Loss DECREASED by {-loss_change:.6f} (Good!)")
                    elif loss_change > 1e-6:
                        self.logger.info(f"  ❌ Loss INCREASED by {loss_change:.6f} (Bad!)")
                    else:
                        self.logger.info(f"  ⚠️  Loss UNCHANGED (±{abs(loss_change):.6f})")
                
                self.prev_loss = loss_components["total_loss"].item()

                # Gradient sanity check on first epoch
                if epoch == 0:
                    self._gradient_sanity_check()
                
                # Run visualizer periodically in test mode
                if self.args.test_visualization_freq > 0 and (epoch + 1) % self.args.test_visualization_freq == 0:
                    self._run_test_mode_visualization(epoch)

            else:
                for i, sample in enumerate(self.train_loader):
                    self.global_step += 1
                    loss_components = self.train_step(sample)
                    
                    if i % 20 == 0: # Log every 20 steps
                        log_str = " | ".join([f"{k}: {v.item():.4f}" for k, v in loss_components.items()])
                        self.logger.info(f"[Epoch {epoch+1}/{self.args.num_epochs} | Step {self.global_step}] {log_str}")

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
    parser.add_argument("--processed_data_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--lam_ckpt_path", type=str, required=True)
    parser.add_argument("--lam_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/action_decoder")
    parser.add_argument("--decoder_model_type", type=str, required=True, choices=list(DECODER_MODEL_REGISTRY.keys()))
    parser.add_argument("--test_mode", action="store_true", help="Test mode: use only 1 batch for debugging/overfitting test")
    
    # Spline args
    parser.add_argument("--spline_k_points", type=int, default=12)
    parser.add_argument("--spline_hidden_dim", type=int, default=512)

    # GRU args
    parser.add_argument("--gru_hidden_dim", type=int, default=512)
    parser.add_argument("--gru_n_layers", type=int, default=2)

    # HybridGated args
    parser.add_argument("--hybrid_k_points", type=int, default=16)
    parser.add_argument("--hybrid_hidden_dim", type=int, default=512)
    parser.add_argument("--hybrid_touch_pos_weight", type=float, default=10.0, help="Weight for positive class in touch BCE loss.")
    parser.add_argument("--hybrid_spline_interpolator", type=str, default="linear", help="'linear' or 'cubic' spline.")

    # CNNQuantizing args
    parser.add_argument("--cnn_k_classes", type=int, default=3000)
    parser.add_argument("--cnn_hidden_dim", type=int, default=128)
    parser.add_argument("--cnn_dropout", type=float, default=0.1)

    # CVAE args
    parser.add_argument("--cvae_style_dim", type=int, default=32)
    parser.add_argument("--cvae_model_dim", type=int, default=256)
    parser.add_argument("--cvae_nhead", type=int, default=4)
    parser.add_argument("--cvae_encoder_layers", type=int, default=4)
    parser.add_argument("--cvae_decoder_layers", type=int, default=6)
    parser.add_argument("--cvae_kl_weight", type=float, default=1.0)
    parser.add_argument("--cvae_touch_pos_weight", type=float, default=10.0)

    # DTW-CVAE args
    parser.add_argument("--dtw_cvae_style_dim", type=int, default=64, help="Style vector dimensionality for DTW-CVAE.")
    parser.add_argument("--dtw_cvae_K_ctrl_pts", type=int, default=32, help="Number of control points for DTW-CVAE generator.")
    parser.add_argument("--dtw_cvae_nhead", type=int, default=8, help="Number of attention heads for DTW-CVAE generator.")
    parser.add_argument("--dtw_cvae_num_decoder_layers", type=int, default=3, help="Number of decoder layers for DTW-CVAE generator.")
    # Loss args
    parser.add_argument("--dtw_cvae_w_recon", type=float, default=1.0, help="Weight for sample-wise reconstruction loss (for samplewise_dtw_cvae).")
    parser.add_argument("--dtw_cvae_w_kl", type=float, default=1.0, help="Weight for KL divergence loss.")
    parser.add_argument("--dtw_cvae_w_aux", type=float, default=0.1, help="Weight for auxiliary transition count loss.")
    parser.add_argument("--dtw_cvae_w_trans", type=float, default=0.5, help="Weight for predicted transition count regularizer.")
    parser.add_argument("--dtw_cvae_kl_free_bits", type=float, default=0.5, help="Free bits for KL divergence.")
    parser.add_argument("--dtw_cvae_dist_gamma", type=float, default=1.0, help="Gamma for distance function in Soft-DTW.")
    parser.add_argument("--dtw_cvae_focal_gamma", type=float, default=2.0, help="Gamma for focal loss in Soft-DTW.")
    parser.add_argument("--dtw_cvae_focal_alpha", type=float, default=0.6, help="Alpha for focal loss in Soft-DTW.")
    parser.add_argument("--dtw_cvae_sdtw_gamma", type=float, default=0.1, help="Gamma for Soft-DTW.")
    parser.add_argument("--kl_anneal_steps", type=int, default=20000, help="Number of steps for KL annealing schedule.")

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
    parser.add_argument("--test_visualization_freq", type=int, default=5, help="Run visualizer every N epochs in test mode. Set to 0 to disable.")
    return parser.parse_args()

def main():
    args = get_args()
    if args.test_mode:
        print("Enabling autograd anomaly detection for test mode.")
        torch.autograd.set_detect_anomaly(True)
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
