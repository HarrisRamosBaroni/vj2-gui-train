"""
DDIM Diffusion Model Training for Action Sequence Recreation
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
from tqdm import tqdm
import time

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position."""
    
    def __init__(self, d_model, max_len=300):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TimestepEmbedding(nn.Module):
    """Embedding for diffusion timestep t."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, timesteps):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class MinimalDiffusionTransformer(nn.Module):
    """Minimalistic transformer-based DDIM for action sequence generation."""
    
    def __init__(
        self,
        action_dim=3,           # [x, y, press] action space
        seq_len=250,            # Action sequence length
        d_model=128,            # Model dimension
        nhead=4,                # Attention heads
        num_layers=4,           # Transformer layers
        dim_feedforward=256,    # FFN dimension
        dropout=0.1,            # Dropout rate
        max_timesteps=1000,     # Maximum diffusion timesteps
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Input embedding: project actions to model dimension
        self.action_embedding = nn.Linear(action_dim, d_model)
        
        # Positional encoding for sequence position
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        
        # Timestep embedding for diffusion step
        self.timestep_embedding = TimestepEmbedding(d_model)
        
        # Core transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection: predict noise
        self.output_projection = nn.Linear(d_model, action_dim)
        
        # Layer norm for stable training
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x_t, t):
        """
        Forward pass for denoising.
        
        Args:
            x_t: Noisy action sequence [B, seq_len, action_dim]
            t: Timestep [B]
            
        Returns:
            predicted_noise: [B, seq_len, action_dim]
        """
        batch_size, seq_len, _ = x_t.shape
        
        # Embed actions to model dimension
        x = self.action_embedding(x_t)  # [B, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Add timestep embedding (broadcast across sequence)
        t_emb = self.timestep_embedding(t)  # [B, d_model]
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, seq_len, d_model]
        x = x + t_emb
        
        # Apply layer norm
        x = self.layer_norm(x)
        
        # Transformer encoding
        x = self.transformer(x)  # [B, seq_len, d_model]
        
        # Project to noise space
        predicted_noise = self.output_projection(x)  # [B, seq_len, action_dim]
        
        return predicted_noise

class DDIMScheduler:
    """DDIM (Denoising Diffusion Implicit Model) scheduler for faster sampling."""
    
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        self.num_timesteps = num_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Pre-compute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def add_noise(self, x_0, t, noise=None):
        """Add noise according to forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Move scheduler tensors to same device as input
        device = x_0.device
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
            
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise, noise
    
    @torch.no_grad()
    def ddim_step(self, model, x_t, t, t_prev, eta=0.0):
        """Single DDIM denoising step with deterministic (eta=0) or stochastic (eta>0) sampling."""
        
        # Move scheduler tensors to same device as input
        device = x_t.device
        alphas_cumprod = self.alphas_cumprod.to(device)
        
        # Get alpha values
        alpha_t = alphas_cumprod[t][:, None, None]
        alpha_prev = alphas_cumprod[t_prev][:, None, None] if t_prev.min() >= 0 else torch.ones_like(alpha_t)
        
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_prev = torch.sqrt(alpha_prev)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        
        # Predict noise
        predicted_noise = model(x_t, t)
        
        # Predict x_0
        pred_x0 = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1.0 - alpha_prev - eta**2 * (1.0 - alpha_t)) * predicted_noise
        
        # Random noise for stochastic sampling
        noise = torch.randn_like(x_t) if eta > 0.0 else torch.zeros_like(x_t)
        
        # DDIM step
        x_prev = sqrt_alpha_prev * pred_x0 + dir_xt + eta * torch.sqrt(1.0 - alpha_t) * noise
        
        return x_prev, pred_x0
    
    @torch.no_grad()
    def sample(self, model, shape, device, num_inference_steps=50, eta=0.0):
        """Fast DDIM sampling with fewer steps."""
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Create timestep schedule (fewer steps than training)
        timesteps = torch.linspace(self.num_timesteps-1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        # Reverse diffusion with DDIM steps
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i+1] if i < len(timesteps)-1 else torch.tensor(-1, device=device)
            
            t_batch = t.expand(batch_size).to(device)
            t_prev_batch = t_prev.expand(batch_size).to(device) if t_prev >= 0 else torch.full((batch_size,), -1, device=device)
            
            x, pred_x0 = self.ddim_step(model, x, t_batch, t_prev_batch, eta=eta)
            
        return x
    
    def get_variance(self, x):
        """Compute variance across sequence and batch dimensions."""
        # Variance across time steps for each sample and channel
        temporal_var = torch.var(x, dim=1)  # [B, 3]
        # Mean across batch
        return torch.mean(temporal_var, dim=0)  # [3] - variance for each channel

# ============================================================================
# DATASET
# ============================================================================

class DiffusionActionDataset(Dataset):
    """Dataset for training diffusion models on action sequences."""
    
    def __init__(self, data_dir, seq_len=250):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        
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
        print(f"Loaded {len(self.sequences)} action sequences of length {seq_len}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx])

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_reconstruction_metrics(original, reconstructed, scheduler=None):
    """Compute detailed reconstruction metrics including noise level analysis."""
    
    # Overall metrics
    mse = torch.mean((original - reconstructed) ** 2)
    mae = torch.mean(torch.abs(original - reconstructed))
    
    # Channel-specific metrics
    coord_mse = torch.mean((original[:, :, :2] - reconstructed[:, :, :2]) ** 2)
    press_mse = torch.mean((original[:, :, 2] - reconstructed[:, :, 2]) ** 2)
    
    # Action-specific metrics (only during active periods)
    press_mask = (original[:, :, 2] > 0.5) | (reconstructed[:, :, 2] > 0.5)
    device = original.device
    if press_mask.any():
        active_coord_mse = torch.mean(((original - reconstructed)[:, :, :2] ** 2)[press_mask])
        press_accuracy = torch.mean((press_mask == (reconstructed[:, :, 2] > 0.5)).float())
    else:
        active_coord_mse = torch.tensor(0.0, device=device)
        press_accuracy = torch.tensor(1.0, device=device)
    
    # NOISE LEVEL ANALYSIS - Key addition for diffusion evaluation
    orig_variance = scheduler.get_variance(original) if scheduler else torch.var(original, dim=1).mean(0)
    recon_variance = scheduler.get_variance(reconstructed) if scheduler else torch.var(reconstructed, dim=1).mean(0)
    
    # Signal-to-noise ratio analysis
    signal_power = torch.mean(original ** 2, dim=(1, 2))  # [B]
    noise_power = torch.mean((original - reconstructed) ** 2, dim=(1, 2))  # [B]
    snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    
    # Variance ratio (how much variance is preserved)
    variance_ratio = recon_variance / (orig_variance + 1e-8)
    
    # Smoothness analysis (how noisy is the reconstruction)
    def compute_smoothness(x):
        """Compute smoothness as variance of first differences."""
        diff = x[:, 1:] - x[:, :-1]  # [B, seq_len-1, 3]
        return torch.var(diff, dim=1).mean(0)  # [3]
    
    orig_smoothness = compute_smoothness(original)
    recon_smoothness = compute_smoothness(reconstructed)
    smoothness_ratio = recon_smoothness / (orig_smoothness + 1e-8)
    
    return {
        'mse': mse.item(),
        'mae': mae.item(), 
        'coord_mse': coord_mse.item(),
        'press_mse': press_mse.item(),
        'active_coord_mse': active_coord_mse.item(),
        'press_accuracy': press_accuracy.item(),
        
        # Noise level metrics
        'orig_variance_x': orig_variance[0].item(),
        'orig_variance_y': orig_variance[1].item(), 
        'orig_variance_p': orig_variance[2].item(),
        'recon_variance_x': recon_variance[0].item(),
        'recon_variance_y': recon_variance[1].item(),
        'recon_variance_p': recon_variance[2].item(),
        'variance_ratio_x': variance_ratio[0].item(),
        'variance_ratio_y': variance_ratio[1].item(),
        'variance_ratio_p': variance_ratio[2].item(),
        'mean_snr_db': snr_db.mean().item(),
        'smoothness_ratio_x': smoothness_ratio[0].item(),
        'smoothness_ratio_y': smoothness_ratio[1].item(),
        'smoothness_ratio_p': smoothness_ratio[2].item(),
    }

def visualize_action_sequences(original, reconstructed, noisy, title_prefix="", max_samples=2):
    """Create comprehensive visualization plots for WandB logging."""
    
    batch_size = min(max_samples, original.shape[0])
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 4 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        for j, (data, label) in enumerate([(original[i], 'Original'), 
                                          (noisy[i], 'Noisy'), 
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

def visualize_noise_comparison(ground_truth_noise, predicted_noise, title_prefix="", max_samples=2):
    """Create side-by-side heatmap comparison of ground truth vs predicted noise."""
    
    batch_size = min(max_samples, ground_truth_noise.shape[0])
    fig, axes = plt.subplots(batch_size, 2, figsize=(12, 4 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    # Find global min/max for consistent color scale
    vmin = min(ground_truth_noise.min(), predicted_noise.min())
    vmax = max(ground_truth_noise.max(), predicted_noise.max())
    
    for i in range(batch_size):
        # Ground truth noise heatmap
        gt_noise = ground_truth_noise[i].T  # [3, seq_len] for better visualization
        axes[i, 0].imshow(gt_noise, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"{title_prefix} Ground Truth Noise (Sample {i+1})")
        axes[i, 0].set_xlabel("Timestep")
        axes[i, 0].set_ylabel("Channel")
        axes[i, 0].set_yticks([0, 1, 2])
        axes[i, 0].set_yticklabels(['X', 'Y', 'Press'])
        
        # Predicted noise heatmap
        pred_noise = predicted_noise[i].T  # [3, seq_len]
        axes[i, 1].imshow(pred_noise, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"{title_prefix} Predicted Noise (Sample {i+1})")
        axes[i, 1].set_xlabel("Timestep")
        axes[i, 1].set_ylabel("Channel")
        axes[i, 1].set_yticks([0, 1, 2])
        axes[i, 1].set_yticklabels(['X', 'Y', 'Press'])
    
    plt.tight_layout()
    return fig

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_diffusion_model(
    model, 
    scheduler, 
    train_loader, 
    val_loader,
    num_epochs=100,
    lr=1e-4,
    device='cuda',
    log_interval=100,
    save_dir='checkpoints'
):
    """Training loop with comprehensive logging."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    Path(save_dir).mkdir(exist_ok=True)
    
    step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, x_0 in enumerate(pbar):
            x_0 = x_0.to(device)
            batch_size = x_0.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
            
            # Add noise
            x_t, noise = scheduler.add_noise(x_0, t)
            
            # Predict noise
            predicted_noise = model(x_t, t)
            
            # Compute loss
            loss = nn.MSELoss()(predicted_noise, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            step += 1
            
            # Log training metrics
            if step % log_interval == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/epoch": epoch + batch_idx / len(train_loader),
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "step": step
                })
            
            # Run validation every 200 steps
            if step % 200 == 0:
                model.eval()
                val_loss = 0.0
                val_metrics = []
                
                with torch.no_grad():
                    for val_batch_idx, x_0_val in enumerate(val_loader):
                        if val_batch_idx >= 3:  # Only process first 3 batches for speed
                            break
                            
                        x_0_val = x_0_val.to(device)
                        val_batch_size = x_0_val.shape[0]
                        
                        # Validation loss (same as training)
                        t_val = torch.randint(0, scheduler.num_timesteps, (val_batch_size,), device=device)
                        x_t_val, noise_val = scheduler.add_noise(x_0_val, t_val)
                        predicted_noise_val = model(x_t_val, t_val)
                        loss_val = nn.MSELoss()(predicted_noise_val, noise_val)
                        val_loss += loss_val.item()
                        
                        # Fast DDIM sampling for evaluation (10 steps for speed)
                        sampled = scheduler.sample(model, x_0_val.shape, device, num_inference_steps=10)
                        
                        # Compute comprehensive metrics including noise analysis
                        metrics = compute_reconstruction_metrics(x_0_val, sampled, scheduler)
                        val_metrics.append(metrics)
                        
                        # Create visualizations every 2000 steps (every 10th validation)
                        if step % 2000 == 0 and val_batch_idx == 0:
                            # Add some noise for visualization
                            t_vis = torch.full((val_batch_size,), scheduler.num_timesteps // 2, device=device)
                            x_t_vis, noise_vis = scheduler.add_noise(x_0_val, t_vis)
                            
                            # Get predicted noise for the same timestep
                            predicted_noise_vis = model(x_t_vis, t_vis)
                            
                            # Action sequence reconstruction visualization - use consistent key for indexable plots
                            fig1 = visualize_action_sequences(
                                x_0_val.cpu().numpy(), 
                                sampled.cpu().numpy(),
                                x_t_vis.cpu().numpy(),
                                title_prefix=f"Step {step}",
                                max_samples=2
                            )
                            wandb.log({"val/reconstructions": wandb.Image(fig1), "step": step})
                            plt.close(fig1)
                            
                            # Ground truth vs predicted noise heatmap comparison - use consistent key for indexable plots
                            fig2 = visualize_noise_comparison(
                                noise_vis.cpu().numpy(),
                                predicted_noise_vis.cpu().numpy(), 
                                title_prefix=f"Step {step}",
                                max_samples=2
                            )
                            wandb.log({"val/noise_comparison": wandb.Image(fig2), "step": step})
                            plt.close(fig2)
                
                # Average validation metrics with noise analysis
                avg_val_loss = val_loss / min(3, len(val_loader))  # Average over batches processed
                if val_metrics:
                    avg_metrics = {}
                    for key in val_metrics[0].keys():
                        avg_metrics[f"val/{key}"] = np.mean([m[key] for m in val_metrics])
                    
                    # Log all metrics including noise analysis with step for indexable plots
                    avg_metrics["val/loss"] = avg_val_loss
                    avg_metrics["step"] = step
                    wandb.log(avg_metrics)
                    
                    # Print key noise metrics for immediate feedback
                    print(f"Step {step}: Val={avg_val_loss:.4f}, "
                          f"SNR={avg_metrics['val/mean_snr_db']:.1f}dB, "
                          f"VarRatio=({avg_metrics['val/variance_ratio_x']:.3f},{avg_metrics['val/variance_ratio_y']:.3f})")
                
                model.train()  # Switch back to training mode
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler_lr.step()
        
        # Log epoch training loss
        wandb.log({
            "train/epoch_loss": epoch_loss/len(train_loader),
            "train/epoch": epoch
        })
        
        print(f"Epoch {epoch}: Train={epoch_loss/len(train_loader):.4f}")
        
        # Save checkpoint
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler_lr.state_dict(),
                'loss': epoch_loss / len(train_loader)
            }
            torch.save(checkpoint, Path(save_dir) / f'diffusion_checkpoint_epoch_{epoch}.pth')

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--val_data_dir', type=str, required=True, help='Directory containing validation data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--wandb_project', type=str, default='action-diffusion-test')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        name=f"ddim_d{args.d_model}_l{args.num_layers}_h{args.nhead}"
    )
    
    # Create datasets
    train_dataset = DiffusionActionDataset(args.data_dir)
    val_dataset = DiffusionActionDataset(args.val_data_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model and scheduler
    model = MinimalDiffusionTransformer(
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead
    ).to(device)
    
    scheduler = DDIMScheduler()
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    wandb.log({"model/num_parameters": num_params})
    print(f"Model has {num_params:,} parameters")
    
    # Train model
    train_diffusion_model(
        model=model,
        scheduler=scheduler, 
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()

# ============================================================================
# TRAINING COMMANDS - Copy and run these commands
# ============================================================================

"""
# 1. GENERATE FULL TRAINING DATA
python -m augmentation.generate_click_dataset \
    --output_dir diffusion_test_data/train \
    --num_sequences 5000 \
    --action_type mixed \
    --min_delta_t_pct 0.02 \
    --max_delta_t_pct 0.3

python -m augmentation.generate_click_dataset \
    --output_dir diffusion_test_data/val \
    --num_sequences 1000 \
    --action_type mixed \
    --min_delta_t_pct 0.02 \
    --max_delta_t_pct 0.3

# 2. MAC CPU TEST (5 minutes)
python train_diffusion_ddim.py \
    --data_dir test_data_mini/train \
    --val_data_dir test_data_mini/val \
    --batch_size 2 \
    --num_epochs 3 \
    --lr 1e-3 \
    --d_model 32 \
    --num_layers 2 \
    --nhead 2 \
    --wandb_project ddim-mac-test \
    --device cpu

# 3. CUDA GPU TEST (5 minutes) - if GPU available
python train_diffusion_ddim.py \
    --data_dir test_data_mini/train \
    --val_data_dir test_data_mini/val \
    --batch_size 16 \
    --num_epochs 5 \
    --lr 2e-4 \
    --d_model 128 \
    --num_layers 4 \
    --nhead 4 \
    --wandb_project ddim-cuda-test \
    --device cuda

# 4. FULL TRAINING (50 minutes)
python train_diffusion_ddim.py \
    --data_dir diffusion_test_data/train \
    --val_data_dir diffusion_test_data/val \
    --batch_size 64 \
    --num_epochs 50 \
    --lr 2e-4 \
    --d_model 64 \
    --num_layers 3 \
    --nhead 4 \
    --wandb_project action-ddim-test

# EXPECTED RESULTS:
# Mac CPU Test: Model ~1K-5K params, loss decreases, SNR >10dB
# CUDA Test: GPU memory <1GB, 2-5x speedup, DDIM <1s for small batches  
# Full Training: SNR >15dB, VarRatio 0.5-2.0, MSE <0.02

# WANDB METRICS TO MONITOR:
# - val/mean_snr_db (target: >20dB)
# - val/variance_ratio_x/y (target: 0.8-1.2) 
# - val/smoothness_ratio_x/y (target: <2.0)
# - val/mse (target: <0.01)
# - val/reconstructions_epoch_X (action sequence visualization every 10 epochs)
# - val/noise_comparison_epoch_X (ground truth vs predicted noise heatmaps every 10 epochs)
"""
