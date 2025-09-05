#!/usr/bin/env python3
"""
Test script for CNN Gesture Classifier.
- Loads gesture data from train_subset directory
- Tests CNN through the dataset 
- Plots original vs reconstruction side by side
- Creates t-SNE visualization of latent vectors
- Logs results to wandb with interactive plots
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
from tqdm import tqdm
import wandb

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the exact same dataset class used during training
from conditioned_gesture_generator.train_cnn_gesture_classifier import StreamingGestureDataset

# Dynamic imports for different model types - will be set at runtime
CNNGestureClassifier = None
CNNGestureClassifierTrainer = None
CoordinateQuantizer = None
create_model_and_trainer = None
decode_predictions = None

# Handle t-SNE import
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    print("⚠️  t-SNE not available, dimensionality reduction will be skipped")


def load_model_modules(model_type: str):
    """Dynamically import model modules based on type."""
    global CNNGestureClassifier, CNNGestureClassifierTrainer, CoordinateQuantizer
    global create_model_and_trainer, decode_predictions
    
    if model_type == 'interpolation':
        from conditioned_gesture_generator.cnn_gesture_classifier import (
            CNNGestureClassifier, 
            CNNGestureClassifierTrainer,
            CoordinateQuantizer,
            create_model_and_trainer,
            decode_predictions
        )
    elif model_type == 'no_int':
        from conditioned_gesture_generator.cnn_gesture_classifier_no_int import (
            CNNGestureClassifier, 
            CNNGestureClassifierTrainer,
            CoordinateQuantizer,
            create_model_and_trainer,
            decode_predictions
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: 'interpolation', 'no_int'")
    
    print(f"✅ Loaded {model_type} model modules")


# Old custom GestureDataset class removed - now using StreamingGestureDataset from training


def plot_gesture_comparison(original: torch.Tensor, reconstruction: torch.Tensor, 
                          title: str = "Original vs Reconstruction") -> plt.Figure:
    """
    Plot original vs reconstructed gesture side by side.
    
    Args:
        original: [T, 2] original gesture coordinates
        reconstruction: [T, 2] reconstructed gesture coordinates
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Convert to numpy for plotting
    orig_np = original.detach().cpu().numpy()
    recon_np = reconstruction.detach().cpu().numpy()
    
    # Plot original
    ax1.plot(orig_np[:, 0], orig_np[:, 1], 'b-o', markersize=3, linewidth=2, alpha=0.7)
    ax1.set_title('Original Gesture')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Plot reconstruction
    ax2.plot(recon_np[:, 0], recon_np[:, 1], 'r-s', markersize=3, linewidth=2, alpha=0.7)
    ax2.set_title('Reconstructed Gesture')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def reduce_dimensionality_tsne(latent_vectors: List[np.ndarray], n_components: int = 2) -> List[np.ndarray]:
    """
    Reduce latent vectors to 2D using t-SNE (adapted from latent_clustering.py).
    """
    if not latent_vectors or not TSNE_AVAILABLE:
        print("⚠️  t-SNE not available, returning zero vectors")
        return [np.zeros(n_components) for _ in latent_vectors]
    
    try:
        points_array = np.vstack(latent_vectors)
        
        # Pre-reduction for high dimensional data
        if points_array.shape[1] > 50:
            print(f"📉 Pre-reducing {points_array.shape[1]} dimensions with PCA before t-SNE")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50, random_state=42)
            points_array = pca.fit_transform(points_array)
        
        # Apply t-SNE
        perplexity = min(30, len(latent_vectors) - 1)
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
        reduced_points = tsne.fit_transform(points_array)
        
        return [reduced_points[i] for i in range(len(reduced_points))]
        
    except Exception as e:
        print(f"⚠️  t-SNE failed ({e}), returning zero vectors")
        return [np.zeros(n_components) for _ in latent_vectors]


def test_cnn_classifier(data_dir: str, max_files: int = 100, batch_size: int = 16, 
                       device: str = 'cuda', wandb_project: str = 'cnn_gesture_test',
                       use_wandb: bool = True, checkpoint_path: Optional[str] = None,
                       model_type: str = 'interpolation'):
    """
    Test CNN gesture classifier on dataset.
    
    Args:
        data_dir: Directory containing action files
        max_files: Maximum number of files to process
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu')
        wandb_project: Wandb project name
        use_wandb: Whether to log to wandb
        checkpoint_path: Path to trained model checkpoint (optional)
        model_type: Type of model to test ('interpolation' or 'no_int')
    """
    # Load the appropriate model modules
    load_model_modules(model_type)
    
    print(f"🚀 Starting CNN Gesture Classifier Test")
    print(f"   Model type: {model_type}")
    print(f"   Data directory: {data_dir}")
    print(f"   Max files: {max_files}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")
    print(f"   Checkpoint: {checkpoint_path or 'None (random weights)'}")
    print(f"   Wandb project: {wandb_project}")
    print(f"   Wandb logging: {'Enabled' if use_wandb else 'Disabled'}")
    
    # Initialize wandb
    if use_wandb:
        wandb.init(project=wandb_project, config={
            'model_type': model_type,
            'max_files': max_files,
            'batch_size': batch_size,
            'device': device,
            'checkpoint_path': checkpoint_path,
            'sequence_length': 250,
            'latent_dim': 128,
            'num_classes': 3000
        })
    
    # Create dataset using exact same class as training
    print("📂 Loading dataset...")
    dataset = StreamingGestureDataset(
        data_dir, 
        seq_len=250, 
        test_mode=False, 
        cache_size=min(1000, max_files * 100) if max_files else 1000
    )
    
    if len(dataset) == 0:
        print("❌ No data files found!")
        return
    
    # Apply max_files limit after dataset creation
    if max_files is not None:
        # Estimate sequences per file (gesture_data has ~1000 sequences per file)
        estimated_sequences_per_file = 1000
        max_sequences = max_files * estimated_sequences_per_file
        
        if len(dataset) > max_sequences:
            print(f"📊 Limiting dataset: {len(dataset)} sequences → {max_sequences} sequences ({max_files} files)")
            # Create a subset by modifying the dataset's total_sequences
            dataset.total_sequences = min(dataset.total_sequences, max_sequences)
        else:
            print(f"📊 Dataset size: {len(dataset)} sequences from available files")
    
    # Load checkpoint first to get model architecture
    model_config = {
        'sequence_length': 250,
        'latent_dim': 128,
        'num_classes': 3000,
        'encoder_channels': [64, 128, 256, 512],  # Default
        'decoder_channels': [512, 256, 128, 64],  # Default
    }
    
    if checkpoint_path:
        print(f"📥 Loading checkpoint to extract architecture: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Extract architecture from checkpoint config if available
            if 'config' in checkpoint:
                config = checkpoint['config']
                print(f"   Found config in checkpoint: {config}")
                
                # Update model config with checkpoint values
                model_config.update({
                    'sequence_length': config.get('sequence_length', 250),
                    'latent_dim': config.get('latent_dim', 128),
                    'num_classes': config.get('num_classes', 3000),
                    'encoder_channels': config.get('encoder_channels', [64, 128, 256, 512]),
                    'decoder_channels': config.get('decoder_channels', [512, 256, 128, 64]),
                })
            else:
                print("   No config found in checkpoint, trying to infer from state_dict...")
                # Try to infer architecture from state dict shapes
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                # Count encoder layers
                encoder_layers = []
                layer_idx = 0
                while f'encoder_layers.{layer_idx}.0.weight' in state_dict:
                    weight_shape = state_dict[f'encoder_layers.{layer_idx}.0.weight'].shape
                    out_channels = weight_shape[0]
                    encoder_layers.append(out_channels)
                    layer_idx += 1
                
                if encoder_layers:
                    model_config['encoder_channels'] = encoder_layers
                    model_config['decoder_channels'] = encoder_layers[::-1]  # Reverse
                    print(f"   Inferred encoder channels: {encoder_layers}")
                
                # Try to infer latent_dim
                if 'encoder_mlp.2.weight' in state_dict:
                    latent_dim = state_dict['encoder_mlp.2.weight'].shape[0]
                    model_config['latent_dim'] = latent_dim
                    print(f"   Inferred latent_dim: {latent_dim}")
                elif 'encoder_mlp.1.weight' in state_dict:
                    latent_dim = state_dict['encoder_mlp.1.weight'].shape[0] 
                    model_config['latent_dim'] = latent_dim
                    print(f"   Inferred latent_dim: {latent_dim}")
                
        except Exception as e:
            print(f"   ⚠️  Could not load checkpoint for architecture: {e}")
            print(f"   Using default architecture")
    
    # Create model with correct architecture
    print("🤖 Creating CNN model with architecture:")
    for key, value in model_config.items():
        print(f"   {key}: {value}")
    
    model = CNNGestureClassifier(
        input_dim=2,
        sequence_length=model_config['sequence_length'],
        latent_dim=model_config['latent_dim'],
        num_classes=model_config['num_classes'],
        encoder_channels=model_config['encoder_channels'],
        decoder_channels=model_config['decoder_channels'],
    ).to(device)
    
    quantizer = CoordinateQuantizer(num_classes=model_config['num_classes'])
    trainer = CNNGestureClassifierTrainer(model, quantizer, device=device)
    
    # Now load checkpoint with matching architecture
    if checkpoint_path:
        print(f"📥 Loading checkpoint weights...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'unknown')
                loss = checkpoint.get('loss', 'unknown')
                print(f"   ✅ Loaded model from epoch {epoch} with loss {loss}")
            else:
                model.load_state_dict(checkpoint)
                print(f"   ✅ Loaded model state dict")
                
        except Exception as e:
            print(f"   ❌ Failed to load checkpoint: {e}")
            print(f"   🔄 Continuing with random weights...")
            import traceback
            traceback.print_exc()
    else:
        print("   ⚠️  No checkpoint provided, using random weights!")
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Process dataset in batches
    print("🔄 Processing gestures...")
    
    all_latents = []
    all_reconstructions = []
    all_originals = []
    all_filenames = []
    all_losses = []
    
    # Process in batches
    for start_idx in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_gestures = []
        batch_filenames = []
        
        # Load batch using StreamingGestureDataset (returns [T, 3] tensors directly)
        for idx in range(start_idx, end_idx):
            if idx >= len(dataset):
                break
            gesture_tensor = dataset[idx]  # Already a torch tensor [T, 3] with [x,y,p]
            batch_gestures.append(gesture_tensor)
            batch_filenames.append(f"gesture_{idx}")  # Simple filename for tracking
        
        if not batch_gestures:
            continue
            
        # Stack into batch tensor
        try:
            batch_tensor = torch.stack(batch_gestures).to(device)  # [B, T, 3]
        except Exception as e:
            print(f"   Error stacking tensors: {e}")
            print(f"   Individual gesture shapes: {[g.shape for g in batch_gestures[:3]]}")
            continue
        
        # Validate tensor shape [B, T, 3] where last dim is [x,y,p]
        if len(batch_tensor.shape) != 3 or batch_tensor.shape[2] != 3:
            print(f"   Error: Expected shape [B, T, 3], got {batch_tensor.shape}")
            continue
        
        # Use the exact working method: manual forward pass + decode_predictions
        with torch.no_grad():
            # Extract x,y coordinates for model input
            batch_tensor_input = batch_tensor[:, :, :2]  # Ensure only x,y coordinates
            
            # Forward pass through model
            latents, logits = model(batch_tensor_input)
            
            # Use decode_predictions directly (this is what works!)
            reconstructions = decode_predictions(logits, quantizer)
            
            # Compute reconstruction loss (for analysis)
            batch_losses = []
            for i in range(len(batch_tensor_input)):
                orig = batch_tensor_input[i].cpu()
                recon = reconstructions[i].cpu()
                loss = F.mse_loss(orig, recon).item()
                batch_losses.append(loss)
        
        # Store results
        all_latents.extend([latent.cpu().numpy() for latent in latents])
        all_reconstructions.extend([recon.cpu() for recon in reconstructions])
        all_originals.extend([orig.cpu() for orig in batch_tensor])
        all_filenames.extend(batch_filenames)
        all_losses.extend(batch_losses)
    
    print(f"✅ Processed {len(all_latents)} gestures")
    print(f"   Average reconstruction MSE: {np.mean(all_losses):.6f}")
    print(f"   Min reconstruction MSE: {np.min(all_losses):.6f}")
    print(f"   Max reconstruction MSE: {np.max(all_losses):.6f}")
    
    # Test generation from random latents (like training script)
    print("🎲 Testing generation from random latents...")
    num_generation_samples = min(20, len(all_latents))
    random_latent = torch.randn(num_generation_samples, model.latent_dim, device=device)
    generated_sequences = trainer.generate_from_latent(random_latent)
    print(f"   Generated {len(generated_sequences)} sequences from random latents")
    
    # Apply t-SNE to latent vectors
    print("📉 Computing t-SNE visualization...")
    tsne_coords = reduce_dimensionality_tsne(all_latents)
    
    # Create visualizations
    print("📊 Creating visualizations...")
    
    # 1. Loss histogram (for general analysis)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(all_losses, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title('Distribution of Reconstruction MSE Loss')
    ax.set_xlabel('MSE Loss')
    ax.set_ylabel('Count')
    ax.axvline(np.mean(all_losses), color='red', linestyle='--', label=f'Mean: {np.mean(all_losses):.6f}')
    ax.legend()
    
    if use_wandb:
        wandb.log({"loss_histogram": wandb.Image(fig)})
    plt.close()
    
    # 2. Create interactive wandb table with gesture plots for hover visualization
    if use_wandb:
        # Limit visualization to avoid memory/time issues
        max_visualizations = min(200, len(all_originals))
        print(f"📋 Creating wandb table with gesture plots ({max_visualizations}/{len(all_originals)} gestures)...")
        
        # Sample indices if we have too many gestures
        if len(all_originals) > max_visualizations:
            vis_indices = np.random.choice(len(all_originals), max_visualizations, replace=False)
            vis_indices = sorted(vis_indices)  # Keep order for consistency
        else:
            vis_indices = range(len(all_originals))
        
        # Create gesture images for wandb (2D trajectory + time series plots)
        gesture_images = []
        print("   Creating plots:", end=" ")
        
        for plot_idx, i in enumerate(vis_indices):
            if plot_idx % 20 == 0:  # Progress indicator
                print(f"{plot_idx}/{max_visualizations}", end=" ", flush=True)
            # Create 2x2 subplot layout: 2D trajectories (top) + time series (bottom)
            fig_small, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=80)
            
            orig = all_originals[i].numpy()
            recon = all_reconstructions[i].numpy()
            time_steps = np.arange(len(orig))
            
            # Top row: 2D trajectory plots
            # Original 2D trajectory
            axes[0, 0].plot(orig[:, 0], orig[:, 1], 'b-o', markersize=1.5, linewidth=1.5, alpha=0.8)
            axes[0, 0].set_title('Original Trajectory', fontsize=10)
            axes[0, 0].set_xlim([0, 1])
            axes[0, 0].set_ylim([0, 1])
            axes[0, 0].set_aspect('equal')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xlabel('X', fontsize=9)
            axes[0, 0].set_ylabel('Y', fontsize=9)
            axes[0, 0].tick_params(labelsize=8)
            
            # Reconstructed 2D trajectory
            axes[0, 1].plot(recon[:, 0], recon[:, 1], 'r-s', markersize=1.5, linewidth=1.5, alpha=0.8)
            axes[0, 1].set_title('Reconstructed Trajectory', fontsize=10)
            axes[0, 1].set_xlim([0, 1])
            axes[0, 1].set_ylim([0, 1])
            axes[0, 1].set_aspect('equal')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xlabel('X', fontsize=9)
            axes[0, 1].set_ylabel('Y', fontsize=9)
            axes[0, 1].tick_params(labelsize=8)
            
            # Bottom row: Time series plots
            # X coordinate time series
            axes[1, 0].plot(time_steps, orig[:, 0], 'b-', linewidth=2, alpha=0.8, label='Original X')
            axes[1, 0].plot(time_steps, recon[:, 0], 'r--', linewidth=2, alpha=0.8, label='Recon X')
            axes[1, 0].set_title('X Coordinate vs Time', fontsize=10)
            axes[1, 0].set_xlabel('Time Step', fontsize=9)
            axes[1, 0].set_ylabel('X Position', fontsize=9)
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend(fontsize=8)
            axes[1, 0].tick_params(labelsize=8)
            
            # Y coordinate time series
            axes[1, 1].plot(time_steps, orig[:, 1], 'b-', linewidth=2, alpha=0.8, label='Original Y')
            axes[1, 1].plot(time_steps, recon[:, 1], 'r--', linewidth=2, alpha=0.8, label='Recon Y')
            axes[1, 1].set_title('Y Coordinate vs Time', fontsize=10)
            axes[1, 1].set_xlabel('Time Step', fontsize=9)
            axes[1, 1].set_ylabel('Y Position', fontsize=9)
            axes[1, 1].set_ylim([0, 1])
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend(fontsize=8)
            axes[1, 1].tick_params(labelsize=8)
            
            plt.tight_layout()
            gesture_images.append(wandb.Image(fig_small, caption=f"MSE: {all_losses[i]:.6f}"))
            plt.close()
        
        print(" Done!")
        
        # Create table data for interactive scatter plot (only for visualized gestures)
        table_data = []
        for plot_idx, i in enumerate(vis_indices):
            table_data.append([
                all_filenames[i],
                gesture_images[plot_idx],  # Use plot_idx for gesture_images indexing
                f"{all_losses[i]:.6f}",
                f"{tsne_coords[i][0]:.3f}" if tsne_coords else "0.000",
                f"{tsne_coords[i][1]:.3f}" if tsne_coords else "0.000",
                f"{np.linalg.norm(all_latents[i]):.3f}",  # Latent vector magnitude
            ])
        
        table = wandb.Table(
            columns=[
                "filename", 
                "gesture_plot", 
                "reconstruction_mse", 
                "tsne_x", 
                "tsne_y",
                "latent_magnitude"
            ],
            data=table_data
        )
        wandb.log({"gesture_analysis_table": table})
        
        # 3. Interactive t-SNE scatter plot linked to table
        if tsne_coords:
            wandb.log({
                "interactive_tsne_scatter": wandb.plot.scatter(
                    table, "tsne_x", "tsne_y",
                    title=f"Interactive t-SNE of Gesture Latent Vectors ({max_visualizations} samples)"
                )
            })
        
        print(f"✅ Wandb table created with {len(table_data)} gesture visualizations")
        
        # 4. Add generation visualization (like training script)
        print("🎲 Creating generation visualizations...")
        
        # Create generation plots similar to training script
        gen_fig, gen_axes = plt.subplots(2, min(5, len(generated_sequences)), figsize=(15, 6))
        if len(generated_sequences) == 1:
            gen_axes = gen_axes.reshape(-1, 1)
        elif len(generated_sequences) < 5:
            gen_axes = gen_axes[:, :len(generated_sequences)]
        
        for i in range(min(5, len(generated_sequences))):
            gen_seq = generated_sequences[i].numpy()
            time_steps = np.arange(len(gen_seq))
            
            # Top: 2D trajectory
            gen_axes[0, i].plot(gen_seq[:, 0], gen_seq[:, 1], 'g-o', markersize=1, linewidth=1.5, alpha=0.8)
            gen_axes[0, i].set_title(f'Generated {i+1} - 2D', fontsize=10)
            gen_axes[0, i].set_xlim([0, 1])
            gen_axes[0, i].set_ylim([0, 1])
            gen_axes[0, i].set_aspect('equal')
            gen_axes[0, i].grid(True, alpha=0.3)
            
            # Bottom: Time series
            gen_axes[1, i].plot(time_steps, gen_seq[:, 0], 'g-', label='X', alpha=0.8)
            gen_axes[1, i].plot(time_steps, gen_seq[:, 1], 'g--', label='Y', alpha=0.8)
            gen_axes[1, i].set_title(f'Generated {i+1} - Time', fontsize=10)
            gen_axes[1, i].set_ylim([0, 1])
            gen_axes[1, i].legend(fontsize=8)
            gen_axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        wandb.log({"generated_sequences": wandb.Image(gen_fig)})
        plt.close()
        
        # Log generation metrics (simple diversity measures)
        gen_coords = np.array([seq.numpy() for seq in generated_sequences])  # [N, T, 2]
        gen_diversity = np.std(gen_coords.reshape(-1, 2), axis=0).mean()  # Coordinate diversity
        gen_smoothness = np.mean([np.mean(np.abs(np.diff(seq, axis=0))) for seq in gen_coords])  # Movement smoothness
        
        wandb.log({
            "generation/coord_diversity": gen_diversity,
            "generation/smoothness": gen_smoothness,
            "generation/num_samples": len(generated_sequences)
        })
    
    print("🎉 Analysis completed successfully!")
    
    if use_wandb:
        # Log summary metrics
        wandb.log({
            "total_gestures": len(all_latents),
            "avg_reconstruction_mse": np.mean(all_losses),
            "std_reconstruction_mse": np.std(all_losses),
            "min_reconstruction_mse": np.min(all_losses),
            "max_reconstruction_mse": np.max(all_losses)
        })
        
        wandb.finish()
    
    return {
        'latents': all_latents,
        'reconstructions': all_reconstructions,
        'originals': all_originals,
        'filenames': all_filenames,
        'losses': all_losses,
        'tsne_coords': tsne_coords
    }


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Test CNN Gesture Classifier on dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test interpolation model (default) with default settings
  python test_cnn_gesture_classifier.py --data-dir resource/dataset/final_data/train_subset
  
  # Test no_int model with custom parameters
  python test_cnn_gesture_classifier.py --data-dir resource/dataset/final_data/train_subset --model-type no_int --max-files 50 --batch-size 32
  
  # Test with checkpoint and no wandb logging
  python test_cnn_gesture_classifier.py --data-dir resource/dataset/final_data/train_subset --checkpoint path/to/model.pth --model-type interpolation --no-wandb
        """
    )
    
    parser.add_argument(
        '--data-dir', 
        type=str, 
        required=True,
        help='Directory containing *_act.npy files'
    )
    
    parser.add_argument(
        '--max-files', 
        type=int, 
        default=100,
        help='Maximum number of files to process (default: 100)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=16,
        help='Batch size for processing (default: 16)'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    
    parser.add_argument(
        '--project', 
        type=str, 
        default='cnn_gesture_test',
        help='Wandb project name (default: cnn_gesture_test)'
    )
    
    parser.add_argument(
        '--no-wandb', 
        action='store_true',
        help='Skip wandb logging'
    )
    
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default=None,
        help='Path to trained model checkpoint (.pth file). If not provided, uses random weights.'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='interpolation',
        choices=['interpolation', 'no_int'],
        help='Type of CNN model to test (default: interpolation)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_dir):
        parser.error(f"Data directory does not exist: {args.data_dir}")
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Run test
    try:
        results = test_cnn_classifier(
            data_dir=args.data_dir,
            max_files=args.max_files,
            batch_size=args.batch_size,
            device=args.device,
            wandb_project=args.project,
            use_wandb=not args.no_wandb,
            checkpoint_path=args.checkpoint,
            model_type=args.model_type
        )
        
        print("\n📈 Test Summary:")
        print(f"   Processed gestures: {len(results['latents'])}")
        print(f"   Average MSE loss: {np.mean(results['losses']):.6f}")
        print(f"   Latent space dim: {len(results['latents'][0]) if results['latents'] else 0}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()