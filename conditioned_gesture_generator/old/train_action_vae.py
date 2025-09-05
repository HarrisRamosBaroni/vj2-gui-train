"""
Action VAE Training Script for Gesture Prediction

This script trains a Variational Autoencoder for gesture prediction using discrete
class labels on normalized screen coordinates with comprehensive evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
import sys
import os
from datetime import datetime

# Import our VAE implementation
from .action_vae import (
    ActionVAE, ActionVAETrainer, CoordinateQuantizer, 
    compute_vae_loss, decode_predictions, create_model_and_trainer
)

# ============================================================================
# DATASET
# ============================================================================

class VAEActionDataset(Dataset):
    """Dataset for training VAE models on action sequences."""
    
    def __init__(self, data_dir, seq_len=250, test_mode=False):
        self.seq_len = seq_len
        self.test_mode = test_mode
        
        if test_mode:
            # Generate synthetic test data
            num_sequences = 100
            self.sequences = self._generate_test_data(num_sequences, seq_len)
            print(f"Generated {num_sequences} synthetic action sequences of length {seq_len} for testing")
        else:
            # Load real data
            self.data_dir = Path(data_dir)
            
            # Load all action files
            self.action_files = list(self.data_dir.glob("*_actions.npy"))
            if not self.action_files:
                raise ValueError(f"No '_actions.npy' files found in {self.data_dir}")
                
            self.sequences = []
            
            for file_path in self.action_files:
                data = np.load(file_path)
                # Reshape from flattened to sequences: [N*seq_len, 3] -> [N, seq_len, 3]
                num_sequences = data.shape[0] // seq_len
                if num_sequences > 0:
                    reshaped = data[:num_sequences * seq_len].reshape(num_sequences, seq_len, 3)
                    self.sequences.extend(reshaped)
            
            self.sequences = np.array(self.sequences, dtype=np.float32)
            
            # Normalize coordinates to [0, 1] range
            self.sequences[:, :, :2] = np.clip(self.sequences[:, :, :2], 0.0, 1.0)
            self.sequences[:, :, 2] = np.clip(self.sequences[:, :, 2], 0.0, 1.0)
            
            print(f"Loaded {len(self.sequences)} action sequences of length {seq_len}")
    
    def _generate_test_data(self, num_sequences, seq_len):
        """Generate synthetic test data with realistic gesture patterns."""
        sequences = []
        
        for _ in range(num_sequences):
            sequence = np.zeros((seq_len, 3), dtype=np.float32)
            
            # Generate random gesture patterns
            num_gestures = np.random.randint(1, 5)  # 1-4 gestures per sequence
            
            for _ in range(num_gestures):
                # Random gesture start and length
                start_t = np.random.randint(0, seq_len - 10)
                length = np.random.randint(5, min(20, seq_len - start_t))
                
                # Random start and end positions
                start_x, start_y = np.random.random(2)
                end_x, end_y = np.random.random(2)
                
                # Interpolate coordinates
                for i, t in enumerate(range(start_t, start_t + length)):
                    alpha = i / (length - 1) if length > 1 else 0
                    sequence[t, 0] = start_x + alpha * (end_x - start_x)
                    sequence[t, 1] = start_y + alpha * (end_y - start_y)
                    sequence[t, 2] = 1.0  # Press active during gesture
            
            # Add some noise
            sequence[:, :2] += np.random.normal(0, 0.01, (seq_len, 2))
            sequence[:, :2] = np.clip(sequence[:, :2], 0.0, 1.0)
            
            sequences.append(sequence)
        
        return np.array(sequences, dtype=np.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx])

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_vae_reconstruction_metrics(original, reconstructed, mu, logvar, quantizer):
    """Compute detailed VAE reconstruction metrics."""
    
    # Basic reconstruction metrics
    mse = torch.mean((original - reconstructed) ** 2)
    mae = torch.mean(torch.abs(original - reconstructed))
    
    # Channel-specific metrics
    coord_mse = torch.mean((original[:, :, :2] - reconstructed[:, :, :2]) ** 2)
    press_mse = torch.mean((original[:, :, 2] - reconstructed[:, :, 2]) ** 2)
    
    # Coordinate classification accuracy
    orig_coords = original[:, :, :2]  # [B, T, 2]
    recon_coords = reconstructed[:, :, :2]  # [B, T, 2]
    
    # Quantize both for accuracy computation
    orig_quantized = quantizer.quantize(orig_coords)  # [B, T, 2]
    recon_quantized = quantizer.quantize(recon_coords)  # [B, T, 2]
    
    x_accuracy = (orig_quantized[:, :, 0] == recon_quantized[:, :, 0]).float().mean()
    y_accuracy = (orig_quantized[:, :, 1] == recon_quantized[:, :, 1]).float().mean()
    
    # Press classification accuracy
    press_accuracy = ((original[:, :, 2] > 0.5) == (reconstructed[:, :, 2] > 0.5)).float().mean()
    
    # Action-specific metrics (only during active periods)
    press_mask = (original[:, :, 2] > 0.5) | (reconstructed[:, :, 2] > 0.5)
    if press_mask.any():
        active_coord_mse = torch.mean(((original - reconstructed)[:, :, :2] ** 2)[press_mask])
    else:
        active_coord_mse = torch.tensor(0.0, device=original.device)
    
    # Latent space metrics
    latent_mean_norm = torch.mean(torch.norm(mu, dim=1))
    latent_std_mean = torch.mean(torch.exp(0.5 * logvar).mean(dim=1))
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    # Posterior collapse detection
    active_dims = (torch.exp(0.5 * logvar).mean(dim=0) > 0.1).sum().float()
    
    return {
        'mse': mse.item(),
        'mae': mae.item(), 
        'coord_mse': coord_mse.item(),
        'press_mse': press_mse.item(),
        'active_coord_mse': active_coord_mse.item(),
        'x_accuracy': x_accuracy.item(),
        'y_accuracy': y_accuracy.item(),
        'press_accuracy': press_accuracy.item(),
        'latent_mean_norm': latent_mean_norm.item(),
        'latent_std_mean': latent_std_mean.item(),
        'kl_divergence': kl_div.item(),
        'active_latent_dims': active_dims.item(),
    }

def compute_generation_metrics(generated_sequences):
    """Compute metrics for generated sequences."""
    
    # Diversity metrics
    coords = generated_sequences[:, :, :2]  # [B, T, 2]
    press = generated_sequences[:, :, 2]    # [B, T]
    
    # Coordinate diversity
    coord_std = torch.std(coords.reshape(-1, 2), dim=0).mean()
    
    # Press activity statistics
    press_rate = (press > 0.5).float().mean()
    
    # Smoothness (how much coordinates change between timesteps)
    coord_diff = torch.diff(coords, dim=1)  # [B, T-1, 2]
    smoothness = torch.mean(torch.norm(coord_diff, dim=2))
    
    # Sequence-level diversity (pairwise distances between sequences)
    if generated_sequences.shape[0] > 1:
        seq_flat = coords.reshape(coords.shape[0], -1)  # [B, T*2]
        pairwise_dists = torch.pdist(seq_flat)
        sequence_diversity = pairwise_dists.mean()
    else:
        sequence_diversity = torch.tensor(0.0)
    
    return {
        'coord_std': coord_std.item(),
        'press_rate': press_rate.item(), 
        'smoothness': smoothness.item(),
        'sequence_diversity': sequence_diversity.item(),
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_action_sequences(original, reconstructed, title_prefix="", max_samples=2):
    """Create comprehensive visualization plots for WandB logging."""
    
    batch_size = min(max_samples, original.shape[0])
    fig, axes = plt.subplots(batch_size, 2, figsize=(12, 4 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        for j, (data, label) in enumerate([(original[i], 'Original'), 
                                          (reconstructed[i], 'Reconstructed')]):
            ax = axes[i, j]
            
            # Plot each channel
            timesteps = np.arange(len(data))
            ax.plot(timesteps, data[:, 0], 'b-', alpha=0.7, label='X')
            ax.plot(timesteps, data[:, 1], 'g-', alpha=0.7, label='Y')
            ax.plot(timesteps, data[:, 2], 'r-', alpha=0.7, label='Press')
            
            # Highlight active regions
            active_mask = data[:, 2] > 0.5
            if np.any(active_mask):
                ax.fill_between(timesteps, 0, 1, where=active_mask, alpha=0.2, color='red')
            
            ax.set_title(f"{title_prefix} {label} (Sample {i+1})")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    return fig

def visualize_generated_sequences(generated, title_prefix="", max_samples=4):
    """Visualize generated sequences."""
    
    batch_size = min(max_samples, generated.shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(batch_size):
        ax = axes[i]
        data = generated[i]
        
        # Plot each channel
        timesteps = np.arange(len(data))
        ax.plot(timesteps, data[:, 0], 'b-', alpha=0.7, label='X')
        ax.plot(timesteps, data[:, 1], 'g-', alpha=0.7, label='Y')
        ax.plot(timesteps, data[:, 2], 'r-', alpha=0.7, label='Press')
        
        # Highlight active regions
        active_mask = data[:, 2] > 0.5
        if np.any(active_mask):
            ax.fill_between(timesteps, 0, 1, where=active_mask, alpha=0.2, color='red')
        
        ax.set_title(f"{title_prefix} Generated Sample {i+1}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    return fig

def visualize_latent_space(mu, title_prefix="", max_points=500):
    """Visualize latent space distribution."""
    
    # Sample random subset for visualization
    if mu.shape[0] > max_points:
        indices = torch.randperm(mu.shape[0])[:max_points]
        mu_sample = mu[indices]
    else:
        mu_sample = mu
    
    latent_dim = mu_sample.shape[1]
    
    if latent_dim >= 3:
        # 3D scatter plot for first 3 dimensions
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            mu_sample[:, 0], mu_sample[:, 1], mu_sample[:, 2],
            c=np.arange(len(mu_sample)), cmap='viridis', alpha=0.6, s=20
        )
        
        ax.set_xlabel('Latent Dim 0')
        ax.set_ylabel('Latent Dim 1')  
        ax.set_zlabel('Latent Dim 2')
        ax.set_title(f'{title_prefix} Latent Space (3D)')
        
        plt.colorbar(scatter, ax=ax, label='Sample Index')
        
    elif latent_dim >= 2:
        # 2D scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        scatter = ax.scatter(
            mu_sample[:, 0], mu_sample[:, 1],
            c=np.arange(len(mu_sample)), cmap='viridis', alpha=0.6, s=20
        )
        
        ax.set_xlabel('Latent Dim 0')
        ax.set_ylabel('Latent Dim 1')
        ax.set_title(f'{title_prefix} Latent Space (2D)')
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='Sample Index')
        
    else:
        # 1D histogram
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        ax.hist(mu_sample[:, 0].cpu().numpy(), bins=50, alpha=0.7, density=True)
        ax.set_xlabel('Latent Dim 0')
        ax.set_ylabel('Density')
        ax.set_title(f'{title_prefix} Latent Space (1D)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def visualize_classification_heatmap(x_logits, y_logits, p_logits, target_coords, target_press, quantizer, title_prefix="", max_samples=2):
    """Visualize classification predictions vs targets."""
    
    batch_size = min(max_samples, x_logits.shape[0])
    seq_len = x_logits.shape[1]
    
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # X coordinate predictions
        x_probs = F.softmax(x_logits[i], dim=-1)  # [T, num_classes]
        x_pred_classes = torch.argmax(x_probs, dim=-1)  # [T]
        x_target_classes = quantizer.quantize(target_coords[i:i+1, :, 0:1]).squeeze()  # [T]
        
        ax = axes[i, 0]
        ax.plot(x_pred_classes.cpu().numpy(), 'b-', label='Predicted', alpha=0.7)
        ax.plot(x_target_classes.cpu().numpy(), 'r-', label='Target', alpha=0.7)
        ax.set_title(f"{title_prefix} X Classes (Sample {i+1})")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Class Index")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Y coordinate predictions  
        y_probs = F.softmax(y_logits[i], dim=-1)  # [T, num_classes]
        y_pred_classes = torch.argmax(y_probs, dim=-1)  # [T]
        y_target_classes = quantizer.quantize(target_coords[i:i+1, :, 1:2]).squeeze()  # [T]
        
        ax = axes[i, 1]
        ax.plot(y_pred_classes.cpu().numpy(), 'g-', label='Predicted', alpha=0.7)
        ax.plot(y_target_classes.cpu().numpy(), 'r-', label='Target', alpha=0.7)
        ax.set_title(f"{title_prefix} Y Classes (Sample {i+1})")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Class Index")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Press predictions
        p_probs = torch.sigmoid(p_logits[i].squeeze(-1))  # [T]
        p_target = target_press[i]  # [T]
        
        ax = axes[i, 2]
        ax.plot(p_probs.cpu().numpy(), 'orange', label='Predicted Prob', alpha=0.7)
        ax.plot(p_target.cpu().numpy(), 'r-', label='Target', alpha=0.7)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        ax.set_title(f"{title_prefix} Press Predictions (Sample {i+1})")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Probability")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    return fig

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_vae_model(
    model,
    trainer,
    quantizer,
    train_loader, 
    val_loader,
    num_epochs=100,
    lr=1e-4,
    device='cuda',
    checkpoint_dir='checkpoints',  # Now expects full directory path
    kl_annealing_epochs=50,
    val_interval=200,
    vis_interval=2000,
    w_x=1.0,
    w_y=1.0,
    w_p=0.5,
    w_kl=1.0,
):
    """Training loop with comprehensive logging. Logs training metrics every step."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # checkpoint_dir is already created in main()
    
    step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {
            'x_loss': 0.0, 'y_loss': 0.0, 'p_loss': 0.0, 'kl_loss': 0.0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            
            # Forward pass
            mu, logvar, x_logits, y_logits, p_logits = model(batch)
            
            # Compute loss with KL annealing
            w_kl_annealed = w_kl * (min(1.0, epoch / kl_annealing_epochs) if kl_annealing_epochs > 0 else 1.0)
            
            loss, loss_dict = compute_vae_loss(
                x_logits, y_logits, p_logits,
                batch[:, :, :2], batch[:, :, 2],
                mu, logvar, quantizer,
                w_x=w_x, w_y=w_y, w_p=w_p, w_kl=w_kl_annealed
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            for key in epoch_metrics:
                if key + '_loss' in loss_dict:
                    epoch_metrics[key] += loss_dict[key + '_loss']
                elif key in loss_dict:
                    epoch_metrics[key] += loss_dict[key]
            
            step += 1
            
            # Log training metrics every step
            log_dict = {
                "train/loss": loss.item(),
                "train/epoch": epoch + batch_idx / len(train_loader),
                "train/lr": optimizer.param_groups[0]['lr'],
                "train/kl_weight": w_kl_annealed,
                "step": step
            }
            log_dict.update({f"train/{k}": v for k, v in loss_dict.items()})
            wandb.log(log_dict)
            
            # Run validation at specified interval
            if step % val_interval == 0:
                model.eval()
                val_loss = 0.0
                val_metrics = []
                
                with torch.no_grad():
                    for val_batch_idx, val_batch in enumerate(val_loader):
                        if val_batch_idx >= 3:  # Only process first 3 batches for speed
                            break
                            
                        val_batch = val_batch.to(device)
                        
                        # Forward pass
                        mu_val, logvar_val, x_logits_val, y_logits_val, p_logits_val = model(val_batch)
                        
                        # Compute loss
                        loss_val, loss_dict_val = compute_vae_loss(
                            x_logits_val, y_logits_val, p_logits_val,
                            val_batch[:, :, :2], val_batch[:, :, 2],
                            mu_val, logvar_val, quantizer,
                            w_x=w_x, w_y=w_y, w_p=w_p, w_kl=w_kl_annealed
                        )
                        val_loss += loss_val.item()
                        
                        # Decode predictions for evaluation
                        reconstructed = decode_predictions(
                            x_logits_val, y_logits_val, p_logits_val, quantizer
                        )
                        
                        # Compute comprehensive metrics
                        metrics = compute_vae_reconstruction_metrics(
                            val_batch, reconstructed, mu_val, logvar_val, quantizer
                        )
                        val_metrics.append(metrics)
                        
                        # Create visualizations at specified interval
                        if step % vis_interval == 0 and val_batch_idx == 0:
                            # Action sequence reconstruction visualization
                            fig1 = visualize_action_sequences(
                                val_batch.cpu().numpy(), 
                                reconstructed.cpu().numpy(),
                                title_prefix=f"Step {step}",
                                max_samples=2
                            )
                            wandb.log({"val/reconstructions": wandb.Image(fig1), "step": step})
                            plt.close(fig1)
                            
                            # Classification predictions visualization
                            fig2 = visualize_classification_heatmap(
                                x_logits_val, y_logits_val, p_logits_val,
                                val_batch[:, :, :2], val_batch[:, :, 2],
                                quantizer, title_prefix=f"Step {step}",
                                max_samples=2
                            )
                            wandb.log({"val/predictions": wandb.Image(fig2), "step": step})
                            plt.close(fig2)
                            
                            # Latent space visualization
                            fig3 = visualize_latent_space(
                                mu_val.cpu(), title_prefix=f"Step {step}",
                                max_points=200
                            )
                            wandb.log({"val/latent_space": wandb.Image(fig3), "step": step})
                            plt.close(fig3)
                            
                            # Generate samples
                            generated = trainer.generate_samples(num_samples=4, temperature=1.0)
                            gen_metrics = compute_generation_metrics(generated)
                            
                            # Generated sequences visualization
                            fig4 = visualize_generated_sequences(
                                generated.numpy(), title_prefix=f"Step {step}",
                                max_samples=4
                            )
                            wandb.log({"val/generated": wandb.Image(fig4), "step": step})
                            plt.close(fig4)
                            
                            # Log generation metrics
                            gen_log = {f"val/gen_{k}": v for k, v in gen_metrics.items()}
                            gen_log["step"] = step
                            wandb.log(gen_log)
                
                # Average validation metrics
                avg_val_loss = val_loss / min(3, len(val_loader))
                if val_metrics:
                    avg_metrics = {}
                    for key in val_metrics[0].keys():
                        avg_metrics[f"val/{key}"] = np.mean([m[key] for m in val_metrics])
                    
                    # Log all metrics
                    avg_metrics["val/loss"] = avg_val_loss
                    avg_metrics["step"] = step
                    wandb.log(avg_metrics)
                    
                    # Print key metrics
                    print(f"Step {step}: Val={avg_val_loss:.4f}, "
                          f"MSE={avg_metrics['val/mse']:.4f}, "
                          f"XAcc={avg_metrics['val/x_accuracy']:.3f}, "
                          f"YAcc={avg_metrics['val/y_accuracy']:.3f}, "
                          f"KL={avg_metrics['val/kl_divergence']:.3f}")
                
                # Save elite model at every validation run (overwrites previous)
                # Track if this is the best model so far
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                
                # Always save current model as elite (overwrites previous elite)
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                    'best_loss': best_val_loss,
                    'is_best': avg_val_loss <= best_val_loss,
                    'config': {
                        'latent_dim': model.latent_dim,
                        'model_dim': model.model_dim,
                        'sequence_length': model.sequence_length,
                        'num_classes': model.num_classes,
                        'num_heads': model.num_heads,
                    }
                }, Path(checkpoint_dir) / 'elite_model.pth')
                
                model.train()  # Switch back to training mode
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler_lr.step()
        
        # Log epoch training loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_log = {
            "train/epoch_loss": avg_epoch_loss,
            "train/epoch": epoch
        }
        for key in epoch_metrics:
            epoch_log[f"train/epoch_{key}"] = epoch_metrics[key] / len(train_loader)
        wandb.log(epoch_log)
        
        print(f"Epoch {epoch}: Train={avg_epoch_loss:.4f}")
        
        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler_lr.state_dict(),
            'loss': avg_epoch_loss,
            'config': {
                'latent_dim': model.latent_dim,
                'model_dim': model.model_dim,
                'sequence_length': model.sequence_length,
                'num_classes': model.num_classes,
                'num_heads': model.num_heads,
            }
        }
        torch.save(checkpoint, Path(checkpoint_dir) / f'epoch_{epoch:03d}.pth')

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Action VAE for gesture prediction')
    parser.add_argument('--data_dir', type=str, help='Directory containing training data')
    parser.add_argument('--val_data_dir', type=str, help='Directory containing validation data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--model_dim', type=int, default=256)
    parser.add_argument('--sequence_length', type=int, default=250)
    parser.add_argument('--num_classes', type=int, default=3000)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--kl_annealing_epochs', type=int, default=50)
    parser.add_argument('--wandb_project', type=str, default='action-vae-training')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--val_interval', type=int, default=200, 
                       help='Validation interval in steps')
    parser.add_argument('--vis_interval', type=int, default=2000, 
                       help='Visualization interval in steps')
    parser.add_argument('--num_workers', type=int, default=0, 
                       help='Number of worker processes for data loading (0=single-threaded)')
    parser.add_argument('--w_x', type=float, default=1.0,
                       help='Weight for X coordinate classification loss')
    parser.add_argument('--w_y', type=float, default=1.0,
                       help='Weight for Y coordinate classification loss') 
    parser.add_argument('--w_p', type=float, default=0.5,
                       help='Weight for press classification loss')
    parser.add_argument('--w_kl', type=float, default=1.0,
                       help='Weight for KL divergence loss (before annealing)')
    parser.add_argument('--test', action='store_true', 
                       help='Run in test mode with synthetic data (no real data files needed)')
    args = parser.parse_args()
    
    # Test mode validation
    if args.test:
        print("Running in TEST MODE with synthetic data")
        # Override settings for faster testing
        if args.num_epochs > 3:
            print(f"Reducing epochs from {args.num_epochs} to 3 for test mode")
            args.num_epochs = 3
        if args.batch_size > 8:
            print(f"Reducing batch size from {args.batch_size} to 8 for test mode")
            args.batch_size = 8
        if args.sequence_length > 50:
            print(f"Reducing sequence length from {args.sequence_length} to 50 for test mode")
            args.sequence_length = 50
        if args.latent_dim > 16:
            print(f"Reducing latent dim from {args.latent_dim} to 16 for test mode")
            args.latent_dim = 16
        if args.model_dim > 64:
            print(f"Reducing model dim from {args.model_dim} to 64 for test mode")
            args.model_dim = 64
        if args.num_encoder_layers > 2:
            print(f"Reducing encoder layers from {args.num_encoder_layers} to 2 for test mode")
            args.num_encoder_layers = 2
        if args.num_decoder_layers > 2:
            print(f"Reducing decoder layers from {args.num_decoder_layers} to 2 for test mode")
            args.num_decoder_layers = 2
        args.kl_annealing_epochs = 1
        args.wandb_project = 'action-vae-test'
        # Shorter intervals for test mode
        if args.val_interval > 50:
            print(f"Reducing val_interval from {args.val_interval} to 50 for test mode")
            args.val_interval = 50
        if args.vis_interval > 100:
            print(f"Reducing vis_interval from {args.vis_interval} to 100 for test mode")
            args.vis_interval = 100
    else:
        # Validate required arguments for real training
        if not args.data_dir:
            parser.error("--data_dir is required when not in test mode")
        if not args.val_data_dir:
            parser.error("--val_data_dir is required when not in test mode")
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"transformer_ld{args.latent_dim}_md{args.model_dim}_el{args.num_encoder_layers}_dl{args.num_decoder_layers}"
    if args.test:
        run_name = f"TEST_{run_name}"
    
    # Create run-specific checkpoint directory
    run_dir_name = f"run_{timestamp}_{run_name}"
    checkpoint_dir = Path(args.save_dir) / run_dir_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        name=run_name
    )
    
    # Create datasets
    train_dataset = VAEActionDataset(
        args.data_dir if not args.test else None, 
        seq_len=args.sequence_length, 
        test_mode=args.test
    )
    val_dataset = VAEActionDataset(
        args.val_data_dir if not args.test else None, 
        seq_len=args.sequence_length, 
        test_mode=args.test
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Model configuration
    model_config = {
        'input_dim': 3,
        'latent_dim': args.latent_dim,
        'model_dim': args.model_dim,
        'sequence_length': args.sequence_length,
        'num_classes': args.num_classes,
        'num_heads': args.num_heads,
        'num_encoder_layers': args.num_encoder_layers,
        'num_decoder_layers': args.num_decoder_layers,
    }
    
    # Create model directly with all parameters
    model = ActionVAE(
        input_dim=3,
        latent_dim=args.latent_dim,
        model_dim=args.model_dim,
        sequence_length=args.sequence_length,
        num_classes=args.num_classes,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
    ).to(device)
    
    quantizer = CoordinateQuantizer(num_classes=args.num_classes)
    trainer = ActionVAETrainer(model, quantizer, device=device)
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    wandb.log({"model/num_parameters": num_params})
    print(f"Model has {num_params:,} parameters")
    
    # Log SDPA backend info for debugging
    try:
        from gui_world_model.utils.modules import get_sdpa_info
        sdpa_info = get_sdpa_info()
        print(f"SDPA Backend: {sdpa_info['context_manager']}")
        if sdpa_info['backends']:
            print(f"Available backends: {sdpa_info['backends']}")
        wandb.log({
            "system/sdpa_backend": sdpa_info['context_manager'],
            "system/pytorch_version": sdpa_info['pytorch_version'],
        })
        if sdpa_info['backends']:
            wandb.log({"system/sdpa_backends": str(sdpa_info['backends'])})
    except Exception as e:
        print(f"Could not get SDPA info: {e}")
    
    if args.test:
        print("\n=== TEST MODE SUMMARY ===")
        print(f"Epochs: {args.num_epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Sequence length: {args.sequence_length}")
        print(f"Model parameters: {num_params:,}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print("========================\n")
    
    # Train model
    train_vae_model(
        model=model,
        trainer=trainer,
        quantizer=quantizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device,
        checkpoint_dir=checkpoint_dir,
        kl_annealing_epochs=args.kl_annealing_epochs,
        val_interval=args.val_interval,
        vis_interval=args.vis_interval,
        w_x=args.w_x,
        w_y=args.w_y,
        w_p=args.w_p,
        w_kl=args.w_kl,
    )

if __name__ == "__main__":
    main()

# ============================================================================
# TRAINING COMMANDS - Copy and run these commands
# ============================================================================

"""
# 0. TEST MODE - NO DATA NEEDED (2 minutes)
# Run with synthetic data to test model, training, and visualization
python -m conditioned_gesture_generator.train_action_vae --test

# Test with specific parameters
python -m conditioned_gesture_generator.train_action_vae --test \
    --device cpu \
    --wandb_project vae-test-run

# 1. GENERATE TRAINING DATA (if needed)
python -m data_collection.augmentation.generate_click_dataset \
    --output_dir vae_test_data/train \
    --num_sequences 5000 \
    --action_type mixed \
    --min_delta_t_pct 0.02 \
    --max_delta_t_pct 0.3

python -m data_collection.augmentation.generate_click_dataset \
    --output_dir vae_test_data/val \
    --num_sequences 1000 \
    --action_type mixed \
    --min_delta_t_pct 0.02 \
    --max_delta_t_pct 0.3

# 2. QUICK REAL DATA TEST (10 minutes)
python -m conditioned_gesture_generator.train_action_vae \
    --data_dir test_data_mini/train \
    --val_data_dir test_data_mini/val \
    --batch_size 8 \
    --num_epochs 3 \
    --lr 1e-3 \
    --latent_dim 16 \
    --model_dim 64 \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --sequence_length 100 \
    --kl_annealing_epochs 1 \
    --wandb_project vae-real-test \
    --device auto

# 3. CUDA GPU TEST (15 minutes)
python -m conditioned_gesture_generator.train_action_vae \
    --data_dir test_data_mini/train \
    --val_data_dir test_data_mini/val \
    --batch_size 16 \
    --num_epochs 10 \
    --lr 2e-4 \
    --latent_dim 32 \
    --model_dim 128 \
    --num_encoder_layers 4 \
    --num_decoder_layers 4 \
    --sequence_length 250 \
    --kl_annealing_epochs 5 \
    --wandb_project vae-cuda-test \
    --device cuda

# 4. FULL TRAINING (2-3 hours)
python -m conditioned_gesture_generator.train_action_vae \
    --data_dir vae_test_data/train \
    --val_data_dir vae_test_data/val \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 1e-4 \
    --latent_dim 64 \
    --model_dim 256 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --sequence_length 250 \
    --kl_annealing_epochs 50 \
    --wandb_project action-vae-full

# EXPECTED RESULTS:
# Test Mode: Model runs, plots generated, synthetic data works (~2 min)
# Real Data Test: Model ~30K params, loss decreases, KL annealing works (~10 min) 
# CUDA Test: GPU memory <2GB, 5-10x speedup, accuracy >60% (~15 min)
# Full Training: X/Y accuracy >80%, Press accuracy >90%, KL ~1-5, MSE <0.01 (~2-3 hours)

# WANDB METRICS TO MONITOR:
# - val/mse (target: <0.01)
# - val/x_accuracy, val/y_accuracy (target: >0.8)  
# - val/press_accuracy (target: >0.9)
# - val/kl_divergence (target: 1-5)
# - val/active_latent_dims (target: >50% of latent_dim)
# - val/reconstructions (visual quality every 2000 steps)
# - val/latent_space (clustering visualization every 2000 steps)
# - val/generated (sample quality every 2000 steps)
"""