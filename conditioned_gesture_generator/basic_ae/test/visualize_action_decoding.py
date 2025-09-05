import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
from scipy.interpolate import griddata

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conditioned_gesture_generator.basic_ae.transformer_ae import TinyTransformerAutoencoder, normalize_zscore, load_checkpoint

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = './checkpoints_autoenc/best.pt'
NPZ_FILE = './trajectory_data/20250729_181208.npz'
SEQ_LEN = 250
FEAT_DIM = 3
MODEL_DIM = 32
LATENT_DIM = 3

def load_model():
    """Load the trained autoencoder model"""
    model = TinyTransformerAutoencoder(
        FEAT_DIM=FEAT_DIM, 
        MODEL_DIM=MODEL_DIM, 
        SEQ_LEN=SEQ_LEN, 
        LATENT_DIM=LATENT_DIM
    ).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        try:
            # First check what keys are available
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            print(f"Checkpoint keys: {list(ckpt.keys())}")
            
            # Try different possible keys for the model state
            if 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
            elif 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            elif isinstance(ckpt, dict) and len(ckpt.keys()) == 1:
                # Sometimes the entire checkpoint is just the model state
                key = list(ckpt.keys())[0]
                model.load_state_dict(ckpt[key])
            else:
                # Try to load the checkpoint directly as model state
                model.load_state_dict(ckpt)
            
            print(f"Loaded model from {CHECKPOINT_PATH}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print(f"Using random weights instead")
    else:
        print(f"Warning: Checkpoint {CHECKPOINT_PATH} not found, using random weights")
    
    model.eval()
    return model

def load_action_data():
    """Load action data from NPZ file"""
    data = np.load(NPZ_FILE)
    print(f"NPZ Keys: {list(data.keys())}")
    
    actions = data['actions']  # Shape: (18, 7, 250, 3)
    print(f"Actions shape: {actions.shape}")
    print(f"Actions range - x: [{actions[:,:,:,0].min():.3f}, {actions[:,:,:,0].max():.3f}]")
    print(f"Actions range - y: [{actions[:,:,:,1].min():.3f}, {actions[:,:,:,1].max():.3f}]")
    print(f"Actions range - type: [{actions[:,:,:,2].min():.3f}, {actions[:,:,:,2].max():.3f}]")
    
    # Reshape to get individual sequences: (18*7, 250, 3)
    actions_flat = actions.reshape(-1, SEQ_LEN, FEAT_DIM)
    
    # Filter out sequences that are all zeros (no action)
    non_zero_mask = np.any(actions_flat.reshape(actions_flat.shape[0], -1) != 0, axis=1)
    actions_filtered = actions_flat[non_zero_mask]
    
    print(f"Total sequences: {actions_flat.shape[0]}")
    print(f"Non-zero sequences: {actions_filtered.shape[0]}")
    
    return torch.from_numpy(actions_filtered).float()

def plot_trajectory_line(ax, x_coords, y_coords, p_values, title, alpha=0.7):
    """Plot trajectory as connected line with points colored by p-value"""
    # Filter out zero coordinates (no action)
    non_zero_mask = (x_coords != 0) | (y_coords != 0) | (p_values != 0)
    x_nz = x_coords[non_zero_mask]
    y_nz = y_coords[non_zero_mask]
    p_nz = p_values[non_zero_mask]
    
    if len(x_nz) == 0:
        ax.set_title(f'{title} (No Data)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return
    
    # Plot connected line in gray
    ax.plot(x_nz, y_nz, '-', color='gray', linewidth=1, alpha=0.5)
    
    # Plot points colored by p-value
    scatter = ax.scatter(x_nz, y_nz, c=p_nz, cmap='viridis', 
                        s=30, alpha=alpha, vmin=0, vmax=1, edgecolors='black', linewidth=0.5)
    
    ax.set_title(title)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    return scatter

def visualize_overlay_comparison(original, reconstructed, seq_idx, save_dir):
    """Plot original and reconstructed overlaid for direct comparison"""
    original_np = original.detach().cpu().numpy()
    reconstructed_np = reconstructed.detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Action Sequence {seq_idx}: Overlay Comparison', fontsize=16)
    
    dimensions = ['X Position', 'Y Position', 'P-value (Pressure)']
    
    for i, dim_name in enumerate(dimensions):
        axes[i].plot(original_np[:, i], 'b-', linewidth=2, label='Original', alpha=0.7)
        axes[i].plot(reconstructed_np[:, i], 'r--', linewidth=2, label='Reconstructed', alpha=0.7)
        axes[i].set_title(f'{dim_name}')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'seq_{seq_idx}_overlay.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved overlay plot to {save_path}")

def visualize_trajectory_plots(original, reconstructed, seq_idx, save_dir):
    """Create 2D trajectory line plots for original vs reconstructed"""
    original_np = original.detach().cpu().numpy()
    reconstructed_np = reconstructed.detach().cpu().numpy()
    
    # Plot side-by-side trajectories
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Action Sequence {seq_idx}: Trajectory Comparison (P-value as Color)', fontsize=16)
    
    # Original trajectory
    scatter1 = plot_trajectory_line(axes[0], original_np[:, 0], original_np[:, 1], 
                                   original_np[:, 2], 'Original Trajectory')
    
    # Reconstructed trajectory  
    scatter2 = plot_trajectory_line(axes[1], reconstructed_np[:, 0], reconstructed_np[:, 1], 
                                   reconstructed_np[:, 2], 'Reconstructed Trajectory')
    
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'seq_{seq_idx}_trajectory_plots.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory plots to {save_path}")

def compute_reconstruction_metrics(original, reconstructed):
    """Compute reconstruction quality metrics"""
    mse = torch.mean((original - reconstructed) ** 2)
    mae = torch.mean(torch.abs(original - reconstructed))
    
    # Per-dimension metrics
    mse_per_dim = torch.mean((original - reconstructed) ** 2, dim=0)
    mae_per_dim = torch.mean(torch.abs(original - reconstructed), dim=0)
    
    return {
        'mse_total': mse.item(),
        'mae_total': mae.item(),
        'mse_x': mse_per_dim[0].item(),
        'mse_y': mse_per_dim[1].item(),
        'mse_action_type': mse_per_dim[2].item(),
        'mae_x': mae_per_dim[0].item(),
        'mae_y': mae_per_dim[1].item(),
        'mae_action_type': mae_per_dim[2].item(),
    }

def main():
    """Main visualization function"""
    parser = argparse.ArgumentParser(description='Visualize action autoencoder reconstruction')
    parser.add_argument('--save_dir', type=str, default='action_recon', 
                       help='Directory to save visualization plots (default: action_recon)')
    args = parser.parse_args()
    
    # Create save directory
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving plots to: {save_dir}")
    
    print("Loading model...")
    model = load_model()
    
    print("Loading action data...")
    actions = load_action_data()
    
    # Plot all sequences
    num_sequences = actions.shape[0]
    selected_indices = range(num_sequences)
    
    print(f"\nProcessing all {num_sequences} action sequences...")
    
    all_metrics = []
    
    # Set matplotlib to not show plots
    plt.ioff()
    
    with torch.no_grad():
        for i, seq_idx in enumerate(selected_indices):
            action_seq = actions[seq_idx:seq_idx+1].to(DEVICE)  # Shape: (1, 250, 3)
            
            # Use the sequence as-is (already normalized to [0,1] in NPZ)
            input_seq = action_seq
            
            # Encode and decode
            latent, reconstructed = model(input_seq)
            
            # Move back to CPU for visualization
            original_cpu = input_seq.squeeze(0).cpu()
            reconstructed_cpu = reconstructed.squeeze(0).cpu()
            latent_cpu = latent.squeeze(0).cpu()
            
            print(f"\nSequence {seq_idx}:")
            print(f"  Latent vector: [{latent_cpu[0]:.3f}, {latent_cpu[1]:.3f}, {latent_cpu[2]:.3f}]")
            print(f"  Latent norm: {torch.norm(latent_cpu):.3f}")
            
            # Compute metrics
            metrics = compute_reconstruction_metrics(original_cpu, reconstructed_cpu)
            all_metrics.append(metrics)
            
            print(f"  MSE: {metrics['mse_total']:.6f}")
            print(f"  MAE: {metrics['mae_total']:.6f}")
            print(f"  Per-dim MSE - X: {metrics['mse_x']:.6f}, Y: {metrics['mse_y']:.6f}, P: {metrics['mse_action_type']:.6f}")
            
            # Create visualizations
            visualize_overlay_comparison(original_cpu, reconstructed_cpu, seq_idx, save_dir)
            visualize_trajectory_plots(original_cpu, reconstructed_cpu, seq_idx, save_dir)
    
    # Summary statistics
    print("\n" + "="*50)
    print("RECONSTRUCTION SUMMARY")
    print("="*50)
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        std_metrics = np.std([m[key] for m in all_metrics])
        print(f"{key:20s}: {avg_metrics[key]:.6f} ± {std_metrics:.6f}")
    
    print(f"\nAll plots saved to: {save_dir}")

if __name__ == "__main__":
    main()