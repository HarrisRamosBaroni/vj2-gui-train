"""
Sample Generation Script for Action VAE Models

This script loads a trained VAE checkpoint, generates N samples from the latent space,
computes comprehensive statistics, and saves all results to a timestamped directory.

Usage:
python -m conditioned_gesture_generator.generate_samples \
    --checkpoint ./checkpoints/elite_vae_cnn_model.pth \
    --num_samples 100 \
    --dir results

Output directory format: {dir}_{timestamp}_{checkpoint_name}
Example: results_20240829_143022_elite_vae_cnn_model/
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import argparse
import json
from pathlib import Path
from datetime import datetime
import time
import sys
import os

# Import both VAE implementations
from .action_vae import (
    ActionVAE, ActionVAETrainer, CoordinateQuantizer,
    decode_predictions as decode_predictions_transformer
)
from .action_vae_cnn import (
    ActionVAECNN, ActionVAECNNTrainer, CoordinateQuantizer as CoordinateQuantizerCNN,
    decode_predictions as decode_predictions_cnn
)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """
    Load model from checkpoint and determine model type.
    
    Returns:
        model: Loaded model
        trainer: Corresponding trainer
        quantizer: Coordinate quantizer
        config: Model configuration
        model_type: 'transformer' or 'cnn'
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Determine model type based on config keys
    if 'encoder_channels' in config:
        model_type = 'cnn'
        model = ActionVAECNN(
            input_dim=3,
            latent_dim=config['latent_dim'],
            sequence_length=config['sequence_length'],
            num_classes=config['num_classes'],
            encoder_channels=config['encoder_channels'],
            decoder_channels=config['decoder_channels'],
        ).to(device)
        quantizer = CoordinateQuantizerCNN(num_classes=config['num_classes'])
        trainer = ActionVAECNNTrainer(model, quantizer, device=device)
        decode_predictions = decode_predictions_cnn
    else:
        model_type = 'transformer'
        model = ActionVAE(
            input_dim=3,
            latent_dim=config['latent_dim'],
            model_dim=config['model_dim'],
            sequence_length=config['sequence_length'],
            num_classes=config['num_classes'],
            num_heads=config['num_heads'],
        ).to(device)
        quantizer = CoordinateQuantizer(num_classes=config['num_classes'])
        trainer = ActionVAETrainer(model, quantizer, device=device)
        decode_predictions = decode_predictions_transformer
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, trainer, quantizer, config, model_type, decode_predictions

def generate_samples(trainer, num_samples: int, temperatures: list = [0.5, 1.0, 1.5]):
    """Generate samples at different temperatures."""
    samples_dict = {}
    
    for temp in temperatures:
        print(f"Generating {num_samples} samples at temperature {temp}...")
        samples = trainer.generate_samples(num_samples=num_samples, temperature=temp)
        samples_dict[f'temp_{temp}'] = samples.numpy()
    
    return samples_dict

def compute_comprehensive_statistics(samples_dict: dict):
    """Compute comprehensive statistics for generated samples."""
    stats = {}
    
    for temp_key, samples in samples_dict.items():
        temp = float(temp_key.split('_')[1])
        sample_stats = {}
        
        # Basic shape info
        num_samples, seq_len, num_dims = samples.shape
        sample_stats['num_samples'] = num_samples
        sample_stats['sequence_length'] = seq_len
        sample_stats['temperature'] = temp
        
        # Coordinate statistics (x, y)
        coords = samples[:, :, :2]  # [N, T, 2]
        sample_stats['coord_mean'] = coords.mean(axis=(0, 1)).tolist()
        sample_stats['coord_std'] = coords.std(axis=(0, 1)).tolist()
        sample_stats['coord_min'] = coords.min(axis=(0, 1)).tolist()
        sample_stats['coord_max'] = coords.max(axis=(0, 1)).tolist()
        
        # Press statistics
        press = samples[:, :, 2]  # [N, T]
        sample_stats['press_rate'] = float((press > 0.5).mean())
        sample_stats['press_std'] = float(press.std())
        sample_stats['avg_press_duration'] = compute_avg_press_duration(press)
        
        # Diversity metrics
        sample_stats['pairwise_distance'] = compute_pairwise_diversity(coords)
        sample_stats['temporal_smoothness'] = compute_temporal_smoothness(coords)
        
        # Gesture detection
        gesture_stats = detect_gestures(samples)
        sample_stats.update(gesture_stats)
        
        stats[temp_key] = sample_stats
    
    return stats

def compute_avg_press_duration(press: np.ndarray):
    """Compute average duration of press events."""
    durations = []
    
    for seq in press:
        in_press = False
        current_duration = 0
        
        for p in seq:
            if p > 0.5:  # Press active
                if not in_press:
                    in_press = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:  # Press inactive
                if in_press:
                    durations.append(current_duration)
                    in_press = False
                    current_duration = 0
        
        # Handle case where sequence ends with press
        if in_press:
            durations.append(current_duration)
    
    return float(np.mean(durations)) if durations else 0.0

def compute_pairwise_diversity(coords: np.ndarray):
    """Compute average pairwise distance between sequences."""
    num_samples = coords.shape[0]
    if num_samples < 2:
        return 0.0
    
    # Flatten sequences for distance computation
    coords_flat = coords.reshape(num_samples, -1)  # [N, T*2]
    
    total_distance = 0.0
    num_pairs = 0
    
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            dist = np.linalg.norm(coords_flat[i] - coords_flat[j])
            total_distance += dist
            num_pairs += 1
    
    return float(total_distance / num_pairs) if num_pairs > 0 else 0.0

def compute_temporal_smoothness(coords: np.ndarray):
    """Compute temporal smoothness (how much coordinates change between timesteps)."""
    # Compute differences between consecutive timesteps
    coord_diffs = np.diff(coords, axis=1)  # [N, T-1, 2]
    # Compute L2 norm of differences
    diff_norms = np.linalg.norm(coord_diffs, axis=2)  # [N, T-1]
    # Return average smoothness
    return float(diff_norms.mean())

def detect_gestures(samples: np.ndarray):
    """Detect and analyze gesture patterns in samples."""
    gesture_stats = {
        'avg_gestures_per_sequence': 0.0,
        'avg_gesture_length': 0.0,
        'gesture_coverage': 0.0,  # Fraction of time spent in gestures
    }
    
    all_gesture_lengths = []
    total_sequences = samples.shape[0]
    total_gestures = 0
    total_gesture_time = 0
    total_time = samples.shape[0] * samples.shape[1]
    
    for seq in samples:
        press = seq[:, 2]
        gesture_lengths = []
        
        in_gesture = False
        current_length = 0
        
        for p in press:
            if p > 0.5:  # In gesture
                if not in_gesture:
                    in_gesture = True
                    current_length = 1
                else:
                    current_length += 1
                total_gesture_time += 1
            else:  # Not in gesture
                if in_gesture:
                    gesture_lengths.append(current_length)
                    in_gesture = False
                    current_length = 0
        
        # Handle case where sequence ends with gesture
        if in_gesture:
            gesture_lengths.append(current_length)
        
        all_gesture_lengths.extend(gesture_lengths)
        total_gestures += len(gesture_lengths)
    
    gesture_stats['avg_gestures_per_sequence'] = total_gestures / total_sequences
    gesture_stats['avg_gesture_length'] = np.mean(all_gesture_lengths) if all_gesture_lengths else 0.0
    gesture_stats['gesture_coverage'] = total_gesture_time / total_time
    
    return gesture_stats

def create_visualizations(samples_dict: dict, stats: dict, output_dir: Path, num_samples: int):
    """Create comprehensive visualizations."""
    
    # 1. Sample sequences visualization
    create_sequence_plots(samples_dict, output_dir)
    
    # 2. Statistical comparison plots
    create_temperature_comparison_plots(stats, output_dir)
    
    # 3. Distribution plots
    create_distribution_plots(samples_dict, output_dir)
    
    # 4. Trajectory plots
    create_trajectory_plots(samples_dict, output_dir, num_samples)
    
    # 5. Gesture analysis plots
    create_gesture_analysis_plots(samples_dict, output_dir)
    
    # 6. Individual signal plots and 2D trajectories
    create_individual_plots(samples_dict, output_dir, num_samples)

def create_sequence_plots(samples_dict: dict, output_dir: Path):
    """Create time series plots of sample sequences."""
    for temp_key, samples in samples_dict.items():
        temp = temp_key.split('_')[1]
        
        # Plot first 6 sequences
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i in range(min(6, samples.shape[0])):
            ax = axes[i]
            seq = samples[i]
            timesteps = np.arange(len(seq))
            
            ax.plot(timesteps, seq[:, 0], 'b-', alpha=0.7, label='X', linewidth=1.5)
            ax.plot(timesteps, seq[:, 1], 'g-', alpha=0.7, label='Y', linewidth=1.5)
            ax.plot(timesteps, seq[:, 2], 'r-', alpha=0.7, label='Press', linewidth=1.5)
            
            # Highlight gesture regions
            press_mask = seq[:, 2] > 0.5
            if np.any(press_mask):
                ax.fill_between(timesteps, 0, 1, where=press_mask, alpha=0.2, color='red')
            
            ax.set_title(f'Sample {i+1} (T={temp})')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'sequences_temp_{temp}.png', dpi=150, bbox_inches='tight')
        plt.close()

def create_temperature_comparison_plots(stats: dict, output_dir: Path):
    """Create plots comparing statistics across temperatures."""
    temperatures = [stats[key]['temperature'] for key in stats.keys()]
    
    # Prepare data for plotting
    metrics = {
        'Press Rate': [stats[key]['press_rate'] for key in stats.keys()],
        'Pairwise Diversity': [stats[key]['pairwise_distance'] for key in stats.keys()],
        'Temporal Smoothness': [stats[key]['temporal_smoothness'] for key in stats.keys()],
        'Avg Gestures/Seq': [stats[key]['avg_gestures_per_sequence'] for key in stats.keys()],
        'Gesture Coverage': [stats[key]['gesture_coverage'] for key in stats.keys()],
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[i]
        ax.plot(temperatures, values, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Temperature')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs Temperature')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'temperature_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_distribution_plots(samples_dict: dict, output_dir: Path):
    """Create distribution plots for coordinates and press values."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for i, (temp_key, samples) in enumerate(samples_dict.items()):
        temp = temp_key.split('_')[1]
        
        # X coordinate distribution
        ax = axes[0, i]
        ax.hist(samples[:, :, 0].flatten(), bins=50, alpha=0.7, density=True)
        ax.set_title(f'X Distribution (T={temp})')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        # Y coordinate distribution
        ax = axes[1, i]
        ax.hist(samples[:, :, 1].flatten(), bins=50, alpha=0.7, density=True)
        ax.set_title(f'Y Distribution (T={temp})')
        ax.set_xlabel('Y Coordinate')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_trajectory_plots(samples_dict: dict, output_dir: Path, num_samples: int):
    """Create 2D trajectory plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (temp_key, samples) in enumerate(samples_dict.items()):
        temp = temp_key.split('_')[1]
        ax = axes[i]
        
        # Plot all trajectories up to num_samples (with reasonable alpha for many samples)
        num_to_plot = min(num_samples, samples.shape[0])
        base_alpha_active = max(0.1, min(0.7, 20.0 / num_to_plot))  # Adjust alpha based on sample count
        base_alpha_inactive = max(0.05, min(0.3, 10.0 / num_to_plot))
        
        for j in range(num_to_plot):
            seq = samples[j]
            coords = seq[:, :2]
            press = seq[:, 2]
            
            # Color trajectory by press state
            for k in range(len(coords) - 1):
                color = 'red' if press[k] > 0.5 else 'blue'
                alpha = base_alpha_active if press[k] > 0.5 else base_alpha_inactive
                ax.plot(coords[k:k+2, 0], coords[k:k+2, 1], 
                       color=color, alpha=alpha, linewidth=1)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'Trajectories (T={temp})')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_gesture_analysis_plots(samples_dict: dict, output_dir: Path):
    """Create gesture-specific analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Collect gesture lengths across all temperatures
    all_gesture_lengths = {}
    press_durations = {}
    
    for temp_key, samples in samples_dict.items():
        temp = temp_key.split('_')[1]
        gesture_lengths = []
        
        for seq in samples:
            press = seq[:, 2]
            current_length = 0
            in_gesture = False
            
            for p in press:
                if p > 0.5:
                    if not in_gesture:
                        in_gesture = True
                        current_length = 1
                    else:
                        current_length += 1
                else:
                    if in_gesture:
                        gesture_lengths.append(current_length)
                        in_gesture = False
                        current_length = 0
            
            if in_gesture:
                gesture_lengths.append(current_length)
        
        all_gesture_lengths[temp] = gesture_lengths
        press_durations[temp] = [seq[:, 2].sum() for seq in samples]
    
    # Plot gesture length distributions
    ax = axes[0, 0]
    for temp, lengths in all_gesture_lengths.items():
        if lengths:
            ax.hist(lengths, bins=20, alpha=0.6, label=f'T={temp}', density=True)
    ax.set_xlabel('Gesture Length (timesteps)')
    ax.set_ylabel('Density')
    ax.set_title('Gesture Length Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot press duration per sequence
    ax = axes[0, 1]
    for temp, durations in press_durations.items():
        ax.hist(durations, bins=20, alpha=0.6, label=f'T={temp}', density=True)
    ax.set_xlabel('Total Press Duration per Sequence')
    ax.set_ylabel('Density')
    ax.set_title('Press Duration Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot press rate over time (averaged across sequences)
    ax = axes[1, 0]
    for temp_key, samples in samples_dict.items():
        temp = temp_key.split('_')[1]
        press_over_time = samples[:, :, 2].mean(axis=0)  # Average over sequences
        timesteps = np.arange(len(press_over_time))
        ax.plot(timesteps, press_over_time, label=f'T={temp}', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Average Press Rate')
    ax.set_title('Press Rate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot coordinate variance over time
    ax = axes[1, 1]
    for temp_key, samples in samples_dict.items():
        temp = temp_key.split('_')[1]
        coord_var = samples[:, :, :2].var(axis=0).mean(axis=1)  # Variance over samples, avg over x,y
        timesteps = np.arange(len(coord_var))
        ax.plot(timesteps, coord_var, label=f'T={temp}', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Coordinate Variance')
    ax.set_title('Coordinate Variance Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gesture_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_individual_plots(samples_dict: dict, output_dir: Path, num_samples: int):
    """Create individual signal plots and press-gated 2D trajectory plots."""
    
    # Create subdirectories
    signals_dir = output_dir / 'individual_signals'
    trajectories_dir = output_dir / 'individual_trajectories'
    signals_dir.mkdir(exist_ok=True)
    trajectories_dir.mkdir(exist_ok=True)
    
    press_threshold = 0.5
    
    for temp_key, samples in samples_dict.items():
        temp = temp_key.split('_')[1]
        
        # Create signal plots for all samples up to num_samples
        num_plots = min(num_samples, samples.shape[0])
        
        for i in range(num_plots):
            seq = samples[i]
            timesteps = np.arange(len(seq))
            
            # Individual signal plots (x, y, p vs time)
            fig, axes = plt.subplots(3, 1, figsize=(12, 9))
            
            # X signal
            axes[0].plot(timesteps, seq[:, 0], 'b-', linewidth=2, alpha=0.8)
            axes[0].set_title(f'X Coordinate vs Time (Sample {i+1}, T={temp})')
            axes[0].set_ylabel('X Coordinate')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(-0.05, 1.05)
            
            # Y signal
            axes[1].plot(timesteps, seq[:, 1], 'g-', linewidth=2, alpha=0.8)
            axes[1].set_title(f'Y Coordinate vs Time (Sample {i+1}, T={temp})')
            axes[1].set_ylabel('Y Coordinate')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(-0.05, 1.05)
            
            # Press signal
            axes[2].plot(timesteps, seq[:, 2], 'r-', linewidth=2, alpha=0.8)
            axes[2].axhline(y=press_threshold, color='gray', linestyle='--', alpha=0.7, label=f'Threshold ({press_threshold})')
            axes[2].fill_between(timesteps, 0, seq[:, 2], where=(seq[:, 2] > press_threshold), 
                               alpha=0.3, color='red', label='Active Press')
            axes[2].set_title(f'Press Signal vs Time (Sample {i+1}, T={temp})')
            axes[2].set_xlabel('Timestep')
            axes[2].set_ylabel('Press Value')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim(-0.05, 1.05)
            
            plt.tight_layout()
            plt.savefig(signals_dir / f'signals_sample_{i+1:03d}_temp_{temp}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # 2D trajectory plot with press-gated drawing
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
            coords = seq[:, :2]
            press = seq[:, 2]
            
            # Plot trajectory only when press is active
            active_segments = []
            current_segment = []
            
            for j in range(len(coords)):
                if press[j] > press_threshold:
                    current_segment.append(coords[j])
                else:
                    if len(current_segment) > 1:
                        active_segments.append(np.array(current_segment))
                    current_segment = []
            
            # Don't forget the last segment
            if len(current_segment) > 1:
                active_segments.append(np.array(current_segment))
            
            # Plot each active segment
            for segment_idx, segment in enumerate(active_segments):
                if len(segment) > 1:
                    ax.plot(segment[:, 0], segment[:, 1], 'r-', linewidth=2, alpha=0.8)
                    # Mark start and end points
                    ax.scatter(segment[0, 0], segment[0, 1], c='green', s=50, marker='o', 
                             label='Start' if segment_idx == 0 else '', alpha=0.8)
                    ax.scatter(segment[-1, 0], segment[-1, 1], c='red', s=50, marker='s', 
                             label='End' if segment_idx == 0 else '', alpha=0.8)
            
            # Plot all points (active and inactive) with different colors
            active_mask = press > press_threshold
            if np.any(active_mask):
                ax.scatter(coords[active_mask, 0], coords[active_mask, 1], 
                          c='red', s=10, alpha=0.6, label='Active Points')
            if np.any(~active_mask):
                ax.scatter(coords[~active_mask, 0], coords[~active_mask, 1], 
                          c='lightgray', s=5, alpha=0.4, label='Inactive Points')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_title(f'2D Trajectory (Press-Gated)\nSample {i+1}, T={temp}')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add canvas border
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2))
            
            plt.tight_layout()
            plt.savefig(trajectories_dir / f'trajectory_sample_{i+1:03d}_temp_{temp}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"  - Individual signal plots saved to: {signals_dir}")
    print(f"  - Individual trajectory plots saved to: {trajectories_dir}")

def save_samples_and_stats(samples_dict: dict, stats: dict, config: dict, output_dir: Path):
    """Save raw samples and statistics to files."""
    
    # Save samples as numpy arrays
    samples_dir = output_dir / 'samples'
    samples_dir.mkdir(exist_ok=True)
    
    for temp_key, samples in samples_dict.items():
        np.save(samples_dir / f'samples_{temp_key}.npy', samples)
    
    # Save statistics as JSON
    with open(output_dir / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save model configuration
    with open(output_dir / 'model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create summary report
    create_summary_report(stats, config, output_dir)

def create_summary_report(stats: dict, config: dict, output_dir: Path):
    """Create a human-readable summary report."""
    
    report_lines = []
    report_lines.append("# Sample Generation Report")
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Model info
    report_lines.append("## Model Configuration")
    report_lines.append(f"- Latent Dimension: {config['latent_dim']}")
    report_lines.append(f"- Sequence Length: {config['sequence_length']}")
    report_lines.append(f"- Number of Classes: {config['num_classes']}")
    
    if 'encoder_channels' in config:
        report_lines.append(f"- Model Type: CNN")
        report_lines.append(f"- Encoder Channels: {config['encoder_channels']}")
        report_lines.append(f"- Decoder Channels: {config['decoder_channels']}")
    else:
        report_lines.append(f"- Model Type: Transformer")
        report_lines.append(f"- Model Dimension: {config['model_dim']}")
        report_lines.append(f"- Number of Heads: {config['num_heads']}")
    
    report_lines.append("")
    
    # Statistics summary
    report_lines.append("## Generation Statistics")
    report_lines.append("")
    
    for temp_key, temp_stats in stats.items():
        temp = temp_stats['temperature']
        report_lines.append(f"### Temperature {temp}")
        report_lines.append(f"- Samples Generated: {temp_stats['num_samples']}")
        report_lines.append(f"- Press Rate: {temp_stats['press_rate']:.3f}")
        report_lines.append(f"- Pairwise Diversity: {temp_stats['pairwise_distance']:.3f}")
        report_lines.append(f"- Temporal Smoothness: {temp_stats['temporal_smoothness']:.3f}")
        report_lines.append(f"- Avg Gestures per Sequence: {temp_stats['avg_gestures_per_sequence']:.2f}")
        report_lines.append(f"- Gesture Coverage: {temp_stats['gesture_coverage']:.3f}")
        report_lines.append(f"- Avg Gesture Length: {temp_stats['avg_gesture_length']:.2f}")
        report_lines.append("")
    
    # Write report
    with open(output_dir / 'REPORT.md', 'w') as f:
        f.write('\n'.join(report_lines))

def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained VAE checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--num_samples', '-N', type=int, default=100,
                       help='Number of samples to generate')
    parser.add_argument('--dir', type=str, default='samples',
                       help='Base directory name for output')
    parser.add_argument('--temperatures', type=str, default='0.5,1.0,1.5',
                       help='Comma-separated list of sampling temperatures')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for generation')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Parse temperatures
    temperatures = [float(t.strip()) for t in args.temperatures.split(',')]
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_name = Path(args.checkpoint).stem
    output_dir_name = f"{args.dir}_{timestamp}_{checkpoint_name}"
    output_dir = Path(output_dir_name)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {args.num_samples}")
    print(f"Temperatures: {temperatures}")
    
    # Load model
    print("\nLoading checkpoint...")
    model, trainer, quantizer, config, model_type, decode_predictions = load_checkpoint(
        args.checkpoint, device=str(device)
    )
    print(f"Loaded {model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Generate samples
    print("\nGenerating samples...")
    start_time = time.time()
    samples_dict = generate_samples(trainer, args.num_samples, temperatures)
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_comprehensive_statistics(samples_dict)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(samples_dict, stats, output_dir, args.num_samples)
    
    # Save results
    print("\nSaving results...")
    save_samples_and_stats(samples_dict, stats, config, output_dir)
    
    print(f"\n✅ Generation complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Generated {sum(len(samples) for samples in samples_dict.values())} total samples")
    print(f"⏱️  Total time: {time.time() - start_time:.2f} seconds")
    
    # Print quick summary
    print(f"\n📈 Quick Summary:")
    for temp_key, temp_stats in stats.items():
        temp = temp_stats['temperature']
        print(f"  T={temp}: Press Rate={temp_stats['press_rate']:.3f}, "
              f"Diversity={temp_stats['pairwise_distance']:.3f}, "
              f"Gestures/Seq={temp_stats['avg_gestures_per_sequence']:.1f}")

if __name__ == "__main__":
    main()

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
# Basic usage with CNN checkpoint
python -m conditioned_gesture_generator.generate_samples \
    --checkpoint ./checkpoints/run_20240829_143022_cnn_ld64_ec64-128-256_dc256-128-64/elite_model.pth \
    --num_samples 100 \
    --dir results

# Or use specific epoch checkpoint
python -m conditioned_gesture_generator.generate_samples \
    --checkpoint ./checkpoints/run_20240829_143022_cnn_ld64_ec64-128-256_dc256-128-64/epoch_050.pth \
    --num_samples 100 \
    --dir results

# Generate many samples at different temperatures
python -m conditioned_gesture_generator.generate_samples \
    --checkpoint ./checkpoints/run_20240829_143022_cnn_ld64_ec64-128-256_dc256-128-64/elite_model.pth \
    --num_samples 500 \
    --temperatures 0.1,0.5,1.0,1.5,2.0 \
    --dir high_temp_analysis

# Use transformer checkpoint
python -m conditioned_gesture_generator.generate_samples \
    --checkpoint ./checkpoints/run_20240829_144000_transformer_ld64_md256_el6_dl6/elite_model.pth \
    --num_samples 200 \
    --dir transformer_samples

# CPU-only generation
python -m conditioned_gesture_generator.generate_samples \
    --checkpoint ./checkpoints/run_20240829_143022_cnn_ld64_ec64-128-256_dc256-128-64/elite_model.pth \
    --num_samples 50 \
    --dir cpu_samples \
    --device cpu

# Output directory examples:
# results_20240829_143022_elite_model/
# high_temp_analysis_20240829_143500_elite_model/
# transformer_samples_20240829_144000_elite_model/

# Generated files:
# ├── samples/
# │   ├── samples_temp_0.5.npy
# │   ├── samples_temp_1.0.npy
# │   └── samples_temp_1.5.npy
# ├── individual_signals/
# │   ├── signals_sample_001_temp_0.5.png
# │   ├── signals_sample_002_temp_0.5.png
# │   └── ... (x, y, p vs time plots for each sample)
# ├── individual_trajectories/
# │   ├── trajectory_sample_001_temp_0.5.png
# │   ├── trajectory_sample_002_temp_0.5.png
# │   └── ... (2D press-gated trajectory plots)
# ├── sequences_temp_0.5.png
# ├── sequences_temp_1.0.png
# ├── sequences_temp_1.5.png
# ├── temperature_comparison.png
# ├── distributions.png
# ├── trajectories.png
# ├── gesture_analysis.png
# ├── statistics.json
# ├── model_config.json
# └── REPORT.md
"""