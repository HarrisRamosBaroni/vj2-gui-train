# Diffusion Model Test Plan: Action Sequence Recreation (DDIM)

## Overview

This document outlines a comprehensive testing plan for a transformer-based DDIM (Denoising Diffusion Implicit Model) to evaluate the ability of diffusion models to recreate detailed action sequences. The goal is to validate diffusion models as learned optimizers and implement them within the next hour for immediate testing.

## Objectives

1. **Validate Diffusion Capability**: Test if diffusion models can accurately recreate complex action sequences (clicks, swipes, gestures)
2. **Architecture Optimization**: Find the minimal transformer architecture that achieves good reconstruction quality
3. **Training Pipeline**: Establish robust training procedures with comprehensive monitoring
4. **Baseline Establishment**: Create performance benchmarks for future planner integration

## Dataset Generation Strategy

### Leveraging Augmentation Code

Based on the existing `augmentation/` module, we will generate synthetic datasets with controlled complexity:

#### 1. **Basic Action Types**
```python
# Use existing augmentation functions
from augmentation.click_aug import generate_augmented_dataset

# Generate datasets of increasing complexity
datasets = [
    # Simple clicks only
    {"action_type": "click", "num_sequences": 5000, "name": "clicks_only"},
    
    # Swipes only (diagonal movement)  
    {"action_type": "swipe", "num_sequences": 5000, "name": "swipes_only"},
    
    # Mixed actions (clicks + horizontal + vertical + diagonal)
    {"action_type": "mixed", "num_sequences": 10000, "name": "mixed_actions"}
]
```

#### 2. **Complexity Progression**
```python
complexity_levels = [
    # Level 1: Short actions, simple patterns
    {"min_delta_t": 5, "max_delta_t": 15, "name": "simple"},
    
    # Level 2: Medium actions, moderate complexity  
    {"min_delta_t": 10, "max_delta_t": 40, "name": "medium"},
    
    # Level 3: Long actions, complex patterns
    {"min_delta_t": 20, "max_delta_t": 80, "name": "complex"}
]
```

#### 3. **Evaluation Datasets**
```python
# Generate specific test patterns for evaluation
eval_patterns = [
    # Precise reconstruction tests
    {"pattern": "center_click", "x": 0.5, "y": 0.5, "delta_t": 10},
    {"pattern": "corner_clicks", "positions": [(0.1,0.1), (0.9,0.9), (0.1,0.9), (0.9,0.1)]},
    {"pattern": "horizontal_swipe", "start": (0.2, 0.5), "end": (0.8, 0.5)},
    {"pattern": "diagonal_swipe", "start": (0.1, 0.1), "end": (0.9, 0.9)},
    {"pattern": "circle_approximation", "center": (0.5, 0.5), "radius": 0.3}
]
```

## Minimalistic Transformer-Based DDPM Architecture

### Core Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class MinimalDiffusionTransformer(nn.Module):
    """
    Minimalistic transformer-based DDPM for action sequence generation.
    
    Key Design Principles:
    - Small parameter count for fast training/inference
    - Focus on temporal modeling over spatial attention
    - Simple conditioning mechanism
    """
    
    def __init__(
        self,
        action_dim=3,           # [x, y, press] action space
        seq_len=250,            # Action sequence length
        d_model=128,            # Model dimension (small for minimal design)
        nhead=4,                # Attention heads (small for minimal design)  
        num_layers=4,           # Transformer layers (minimal depth)
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
            
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise, noise
    
    @torch.no_grad()
    def ddim_step(self, model, x_t, t, t_prev, eta=0.0):
        """Single DDIM denoising step with deterministic (eta=0) or stochastic (eta>0) sampling."""
        
        # Get alpha values
        alpha_t = self.alphas_cumprod[t][:, None, None]
        alpha_prev = self.alphas_cumprod[t_prev][:, None, None] if t_prev >= 0 else torch.ones_like(alpha_t)
        
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
            
            t_batch = t.expand(batch_size)
            t_prev_batch = t_prev.expand(batch_size) if t_prev >= 0 else torch.full((batch_size,), -1, device=device)
            
            x, pred_x0 = self.ddim_step(model, x, t_batch, t_prev_batch, eta=eta)
            
        return x
    
    def get_variance(self, x):
        """Compute variance across sequence and batch dimensions."""
        # Variance across time steps for each sample and channel
        temporal_var = torch.var(x, dim=1)  # [B, 3]
        # Mean across batch
        return torch.mean(temporal_var, dim=0)  # [3] - variance for each channel
```

## Training Pipeline

### Dataset Preparation

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

class DiffusionActionDataset(Dataset):
    """Dataset for training diffusion models on action sequences."""
    
    def __init__(self, data_dir, seq_len=250):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        
        # Load all action files
        self.action_files = list(self.data_dir.glob("*_actions.npy"))
        self.sequences = []
        
        for file_path in self.action_files:
            data = np.load(file_path)
            # Reshape from flattened to sequences: [N*seq_len, 3] -> [N, seq_len, 3]
            num_sequences = data.shape[0] // seq_len
            reshaped = data[:num_sequences * seq_len].reshape(num_sequences, seq_len, 3)
            self.sequences.extend(reshaped)
            
        self.sequences = np.array(self.sequences, dtype=np.float32)
        print(f"Loaded {len(self.sequences)} action sequences of length {seq_len}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx])

# Generate training data using augmentation
def prepare_diffusion_data():
    """Generate datasets for diffusion training."""
    
    from augmentation.click_aug import generate_augmented_dataset
    
    # Generate training datasets
    datasets = [
        {"output_dir": "diffusion_data/train_clicks", 
         "num_sequences": 5000, "action_type": "click", "min_delta_t": 5, "max_delta_t": 30},
        {"output_dir": "diffusion_data/train_swipes", 
         "num_sequences": 5000, "action_type": "swipe", "min_delta_t": 10, "max_delta_t": 50},
        {"output_dir": "diffusion_data/train_mixed", 
         "num_sequences": 10000, "action_type": "mixed", "min_delta_t": 5, "max_delta_t": 40}
    ]
    
    for config in datasets:
        print(f"Generating {config['output_dir']}...")
        generate_augmented_dataset(**config)
    
    # Generate validation/test sets (smaller, specific patterns)
    generate_augmented_dataset(
        output_dir="diffusion_data/val_mixed",
        num_sequences=1000,
        action_type="mixed",
        min_delta_t=5,
        max_delta_t=40
    )
```

### Training Script with WandB Integration

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def visualize_action_sequences(original, reconstructed, noisy, title_prefix="", max_samples=4):
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
    if press_mask.any():
        active_coord_mse = torch.mean(((original - reconstructed)[:, :, :2] ** 2)[press_mask])
        press_accuracy = torch.mean((press_mask == (reconstructed[:, :, 2] > 0.5)).float())
    else:
        active_coord_mse = torch.tensor(0.0)
        press_accuracy = torch.tensor(1.0)
    
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
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler_lr.step()
        
        # REAL-TIME EVALUATION DURING TRAINING - Every epoch for immediate feedback
        model.eval()
        val_loss = 0.0
        val_metrics = []
        
        with torch.no_grad():
            for batch_idx, x_0 in enumerate(val_loader):
                if batch_idx >= 3:  # Only process first 3 batches for speed
                    break
                    
                x_0 = x_0.to(device)
                batch_size = x_0.shape[0]
                
                # Validation loss (same as training)
                t = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
                x_t, noise = scheduler.add_noise(x_0, t)
                predicted_noise = model(x_t, t)
                loss = nn.MSELoss()(predicted_noise, noise)
                val_loss += loss.item()
                
                # Fast DDIM sampling for evaluation (10 steps for speed)
                sampled = scheduler.sample(model, x_0.shape, device, num_inference_steps=10)
                
                # Compute comprehensive metrics including noise analysis
                metrics = compute_reconstruction_metrics(x_0, sampled, scheduler)
                val_metrics.append(metrics)
                
                # Create visualizations every 10 epochs
                if epoch % 10 == 0 and batch_idx == 0:
                    # Add some noise for visualization
                    t_vis = torch.full((batch_size,), scheduler.num_timesteps // 2, device=device)
                    x_t_vis, _ = scheduler.add_noise(x_0, t_vis)
                    
                    fig = visualize_action_sequences(
                        x_0.cpu().numpy(), 
                        sampled.cpu().numpy(),
                        x_t_vis.cpu().numpy(),
                        title_prefix=f"Epoch {epoch}",
                        max_samples=2
                    )
                    wandb.log({f"val/reconstructions_epoch_{epoch}": wandb.Image(fig)})
                    plt.close(fig)
            
        # Average validation metrics with noise analysis
        avg_val_loss = val_loss / min(3, len(val_loader))  # Average over batches processed
        if val_metrics:
            avg_metrics = {}
            for key in val_metrics[0].keys():
                avg_metrics[f"val/{key}"] = np.mean([m[key] for m in val_metrics])
            
            # Log all metrics including noise analysis
            wandb.log(avg_metrics)
            
            # Print key noise metrics for immediate feedback
            print(f"Epoch {epoch}: Train={epoch_loss/len(train_loader):.4f}, Val={avg_val_loss:.4f}, "
                  f"SNR={avg_metrics['val/mean_snr_db']:.1f}dB, "
                  f"VarRatio=({avg_metrics['val/variance_ratio_x']:.3f},{avg_metrics['val/variance_ratio_y']:.3f})")
        
        wandb.log({
            "val/loss": avg_val_loss,
            "train/epoch_loss": epoch_loss/len(train_loader),
            "val/epoch": epoch
        })
        
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--val_data_dir', type=str, required=True, help='Directory containing validation data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--wandb_project', type=str, default='action-diffusion-test')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        name=f"diffusion_d{args.d_model}_l{args.num_layers}_h{args.nhead}"
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = DiffusionActionDataset(args.data_dir)
    val_dataset = DiffusionActionDataset(args.val_data_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model and scheduler
    model = MinimalDiffusionTransformer(
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead
    ).to(device)
    
    scheduler = DDPMScheduler()
    
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
```

## Comprehensive Testing & Evaluation

### 1. **Quantitative Metrics**

```python
def comprehensive_evaluation(model, scheduler, test_loader, device):
    """Evaluate model on comprehensive metrics."""
    
    model.eval()
    results = {
        'reconstruction_quality': [],
        'action_timing_accuracy': [],
        'spatial_accuracy': [],
        'gesture_type_preservation': []
    }
    
    with torch.no_grad():
        for x_0 in test_loader:
            x_0 = x_0.to(device)
            
            # Sample reconstructions
            reconstructed = scheduler.sample(model, x_0.shape, device)
            
            # 1. Basic reconstruction quality
            mse = torch.mean((x_0 - reconstructed) ** 2)
            results['reconstruction_quality'].append(mse.item())
            
            # 2. Action timing accuracy
            orig_active = (x_0[:, :, 2] > 0.5).float()
            recon_active = (reconstructed[:, :, 2] > 0.5).float()
            timing_iou = compute_temporal_iou(orig_active, recon_active)
            results['action_timing_accuracy'].append(timing_iou)
            
            # 3. Spatial accuracy (during active periods)
            spatial_acc = compute_spatial_accuracy(x_0, reconstructed)
            results['spatial_accuracy'].append(spatial_acc)
            
            # 4. Gesture type preservation
            gesture_acc = compute_gesture_type_accuracy(x_0, reconstructed)
            results['gesture_type_preservation'].append(gesture_acc)
    
    # Aggregate results
    final_results = {}
    for key, values in results.items():
        final_results[f'{key}_mean'] = np.mean(values)
        final_results[f'{key}_std'] = np.std(values)
    
    return final_results

def compute_temporal_iou(orig_active, recon_active):
    """Compute IoU of active time periods."""
    intersection = torch.sum(orig_active * recon_active, dim=1)
    union = torch.sum((orig_active + recon_active) > 0, dim=1).float()
    iou = intersection / (union + 1e-8)
    return torch.mean(iou).item()

def compute_spatial_accuracy(original, reconstructed, threshold=0.1):
    """Compute spatial accuracy during active periods."""
    active_mask = (original[:, :, 2] > 0.5) | (reconstructed[:, :, 2] > 0.5)
    
    if not active_mask.any():
        return 1.0
    
    coord_diff = torch.norm(original[:, :, :2] - reconstructed[:, :, :2], dim=2)
    accurate = (coord_diff < threshold)[active_mask]
    return torch.mean(accurate.float()).item()

def compute_gesture_type_accuracy(original, reconstructed):
    """Classify and compare gesture types."""
    
    def classify_gesture(seq):
        active_indices = torch.where(seq[:, 2] > 0.5)[0]
        if len(active_indices) == 0:
            return 'none'
        
        start_pos = seq[active_indices[0], :2]
        end_pos = seq[active_indices[-1], :2]
        movement = torch.norm(end_pos - start_pos)
        
        if movement < 0.05:
            return 'click'
        else:
            dx = abs(end_pos[0] - start_pos[0])
            dy = abs(end_pos[1] - start_pos[1])
            
            if dx > 2 * dy:
                return 'horizontal'
            elif dy > 2 * dx:
                return 'vertical'
            else:
                return 'diagonal'
    
    batch_size = original.shape[0]
    correct = 0
    
    for i in range(batch_size):
        orig_type = classify_gesture(original[i])
        recon_type = classify_gesture(reconstructed[i])
        if orig_type == recon_type:
            correct += 1
    
    return correct / batch_size
```

### 2. **Qualitative Analysis**

```python
def create_detailed_analysis_plots():
    """Generate comprehensive analysis visualizations."""
    
    # 1. Reconstruction quality across different action types
    # 2. Timestep-by-timestep error analysis
    # 3. Noise level vs reconstruction quality
    # 4. Model size vs performance tradeoffs
    # 5. Training convergence analysis
    
    pass  # Implementation would create detailed matplotlib plots
```

## WandB Monitoring & Tracking

### Enhanced Logging Strategy

```python
# Extend the training logging from VAE reference
wandb_config = {
    "architecture": {
        "model_type": "MinimalDiffusionTransformer",
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "nhead": args.nhead,
        "parameter_count": num_params
    },
    "training": {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_epochs": args.num_epochs,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR"
    },
    "diffusion": {
        "num_timesteps": 1000,
        "beta_start": 1e-4,
        "beta_end": 2e-2,
        "noise_schedule": "linear"
    },
    "data": {
        "sequence_length": 250,
        "action_dim": 3,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset)
    }
}

# Enhanced WandB logging
logging_metrics = [
    # Training metrics
    "train/loss", "train/grad_norm", "train/lr",
    
    # Validation metrics
    "val/loss", "val/mse", "val/mae", 
    "val/coord_mse", "val/press_mse",
    "val/active_coord_mse", "val/press_accuracy",
    
    # Reconstruction quality
    "val/temporal_iou", "val/spatial_accuracy", 
    "val/gesture_type_accuracy",
    
    # Model diagnostics
    "model/weight_norm", "model/activation_stats",
    
    # Sampling quality
    "sampling/time_per_sample", "sampling/noise_reduction",
]

# Custom WandB tables for detailed analysis
reconstruction_table = wandb.Table(columns=[
    "epoch", "sample_id", "original_type", "reconstructed_type", 
    "timing_accuracy", "spatial_accuracy", "overall_quality"
])
```

## Local Testing Plans

### Mac CPU Test Plan (5 minutes)
**Purpose**: Validate all components work locally before full training

```bash
# Step 1: Generate minimal test data (30 seconds)
python -m augmentation.generate_click_dataset \
    --output_dir test_data_mini/train \
    --num_sequences 10 \
    --sequences_per_file 10 \
    --action_type mixed

python -m augmentation.generate_click_dataset \
    --output_dir test_data_mini/val \
    --num_sequences 5 \
    --sequences_per_file 5 \
    --action_type mixed

# Step 2: Test model creation and forward pass (1 minute)
python -c "
import torch
from train_diffusion_ddim import MinimalDiffusionTransformer, DDIMScheduler

# Test model instantiation
model = MinimalDiffusionTransformer(d_model=32, num_layers=2, nhead=2)
scheduler = DDIMScheduler(num_timesteps=100)
print(f'Model created: {sum(p.numel() for p in model.parameters())} params')

# Test forward pass
x = torch.randn(2, 250, 3)  # 2 sequences
t = torch.randint(0, 100, (2,))
noise_pred = model(x, t)
print(f'Forward pass: {x.shape} -> {noise_pred.shape}')

# Test DDIM sampling
sampled = scheduler.sample(model, (1, 250, 3), 'cpu', num_inference_steps=5)
print(f'DDIM sampling: {sampled.shape}, range: [{sampled.min():.3f}, {sampled.max():.3f}]')
"

# Step 3: Test training loop (2 minutes)
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

# Step 4: Test inference metrics (1 minute)
python -c "
import torch
import numpy as np
from train_diffusion_ddim import compute_reconstruction_metrics, DDIMScheduler

# Create dummy data
original = torch.rand(3, 250, 3) * 0.8 + 0.1  # [0.1, 0.9] range
reconstructed = original + torch.randn_like(original) * 0.05  # Add small noise
scheduler = DDIMScheduler()

# Test all metrics
metrics = compute_reconstruction_metrics(original, reconstructed, scheduler)
for k, v in metrics.items():
    print(f'{k}: {v:.4f}')
"

# Expected Mac CPU results:
# - Model creation: ~1K-5K parameters
# - Forward pass: no errors, correct shapes
# - Training: loss decreases over 3 epochs
# - Metrics: SNR > 20dB, variance ratios near 1.0
```

### CUDA GPU Test Plan (5 minutes)
**Purpose**: Validate GPU acceleration and memory efficiency

```bash
# Step 1: Same minimal data as Mac test (reuse if available)
# [Use same test_data_mini from Mac test]

# Step 2: Test GPU model and memory (1 minute)
python -c "
import torch
from train_diffusion_ddim import MinimalDiffusionTransformer, DDIMScheduler

if not torch.cuda.is_available():
    print('❌ CUDA not available')
    exit(1)

device = torch.device('cuda')
print(f'✅ Using GPU: {torch.cuda.get_device_name()}')

# Test model on GPU
model = MinimalDiffusionTransformer(d_model=128, num_layers=4, nhead=4).to(device)
scheduler = DDIMScheduler()
print(f'Model GPU memory: {torch.cuda.memory_allocated()/1e6:.1f}MB')

# Test forward pass with larger batch
x = torch.randn(16, 250, 3, device=device)
t = torch.randint(0, 1000, (16,), device=device)
noise_pred = model(x, t)
print(f'Forward pass memory: {torch.cuda.memory_allocated()/1e6:.1f}MB')

# Test DDIM sampling speed
import time
start = time.time()
sampled = scheduler.sample(model, (8, 250, 3), device, num_inference_steps=20)
elapsed = time.time() - start
print(f'DDIM sampling: {elapsed:.2f}s for 8 sequences (20 steps)')
torch.cuda.empty_cache()
"

# Step 3: Test GPU training speed (2 minutes)
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

# Step 4: Test GPU memory scaling (1 minute)
python -c "
import torch
from train_diffusion_ddim import MinimalDiffusionTransformer

device = torch.device('cuda')
model = MinimalDiffusionTransformer(d_model=128, num_layers=4, nhead=4).to(device)

# Test different batch sizes
for batch_size in [4, 8, 16, 32]:
    try:
        x = torch.randn(batch_size, 250, 3, device=device)
        t = torch.randint(0, 1000, (batch_size,), device=device)
        _ = model(x, t)
        mem_mb = torch.cuda.memory_allocated() / 1e6
        print(f'Batch {batch_size}: {mem_mb:.1f}MB')
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f'Batch {batch_size}: OOM - {str(e)[:50]}...')
        break
"

# Expected CUDA results:
# - GPU memory usage < 1GB for training
# - DDIM sampling < 0.5s for 8 sequences  
# - Training 2-5x faster than CPU
# - Batch size 16-32 should work on most GPUs
```

### Test Success Criteria

**Mac CPU Test (Pass/Fail):**
- [ ] Model instantiation without errors
- [ ] Forward pass produces correct shapes
- [ ] Training loss decreases over 3 epochs
- [ ] All metrics compute without errors
- [ ] SNR metrics show reasonable values (>10dB)

**CUDA GPU Test (Pass/Fail):**
- [ ] CUDA detection and model moves to GPU
- [ ] GPU memory usage reasonable (<1GB)
- [ ] Forward pass 2x+ faster than CPU
- [ ] Training completes without OOM errors
- [ ] DDIM sampling <1s for small batches

## 1-Hour Implementation Timeline

### Minutes 0-10: Local Testing
```bash
# Run Mac test first (5 min)
# If GPU available, run CUDA test (5 min)
# Fix any issues found before proceeding
```

### Minutes 10-20: Setup & Data Generation
```bash
# Create directory and generate training data
mkdir -p diffusion_test_data/{train,val}
cd /Users/xyle/Research/vj2-gui

# Generate training data (5K mixed actions)
python -m augmentation.generate_click_dataset \
    --output_dir diffusion_test_data/train \
    --num_sequences 5000 \
    --action_type mixed \
    --min_delta_t_pct 0.02 \
    --max_delta_t_pct 0.3

# Generate validation data (1K mixed actions)  
python -m augmentation.generate_click_dataset \
    --output_dir diffusion_test_data/val \
    --num_sequences 1000 \
    --action_type mixed \
    --min_delta_t_pct 0.02 \
    --max_delta_t_pct 0.3
```

### Minutes 15-30: Model Implementation
```bash
# Create diffusion model file
touch train_diffusion_ddim.py

# Copy MinimalDiffusionTransformer + DDIMScheduler code from plan
# Add dataset loader and training loop
```

### Minutes 30-45: Training Launch
```bash
# Start training with minimal config for fast iteration
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
```

### Minutes 45-60: Real-time Analysis
- Monitor WandB metrics live:
  - `val/mean_snr_db` (target: >20dB)
  - `val/variance_ratio_x/y` (target: 0.8-1.2)
  - `val/smoothness_ratio_x/y` (target: <2.0)
  - `val/mse` (target: <0.01)
- Adjust hyperparameters if needed
- Evaluate first results and plan next steps

### Key Success Metrics (1 Hour):
- [ ] **Model trains without errors**
- [ ] **Training loss decreases consistently** 
- [ ] **SNR > 15dB** (shows low reconstruction noise)
- [ ] **Variance ratio 0.5-2.0** (preserves data variance reasonably)
- [ ] **MSE < 0.02** (acceptable reconstruction quality)
- [ ] **Visualization shows recognizable patterns**

## Success Criteria

### Minimum Viable Results
- **Reconstruction MSE < 0.01** for simple click actions
- **Temporal IoU > 0.8** for action timing preservation  
- **Spatial accuracy > 0.9** within 0.1 normalized coordinate threshold
- **Training convergence** within 100 epochs

### Optimal Results
- **Reconstruction MSE < 0.005** across all action types
- **Temporal IoU > 0.9** for complex gesture sequences
- **Gesture type accuracy > 0.95** for classification preservation
- **Fast inference** < 10ms per sequence on GPU

### Comparison Baselines
- Direct regression (no diffusion): Should outperform on reconstruction quality
- VAE reconstruction: Should match or exceed reconstruction fidelity
- Random sampling: Should significantly outperform on all metrics

## Expected Outcomes

1. **Technical Validation**: Proof that diffusion models can accurately recreate detailed action sequences
2. **Architecture Insights**: Optimal transformer configuration for action modeling  
3. **Training Best Practices**: Robust training procedures and hyperparameters
4. **Performance Benchmarks**: Quantitative baselines for planner integration
5. **Implementation Ready**: Tested, documented code ready for full VJ2-GUI integration

This comprehensive test plan provides a solid foundation for validating diffusion models as learned optimizers, with extensive monitoring, evaluation, and analysis to ensure reliable performance before integration into the full VJ2-GUI planning pipeline.