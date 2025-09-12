#!/usr/bin/env python3
"""
Evaluation script for VAE decoders with dynamic model loading from checkpoints.
Loads the exact model class that was saved with the checkpoint to ensure compatibility.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Try importing wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available - logging will be disabled")


# Import reusable modules
from conditioned_gesture_generator.gesture_vae.utils import CoordinateQuantizer, ExperimentLogger


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device, model_type: Optional[str] = None) -> Tuple[nn.Module, Dict]:
    """
    Load model from checkpoint using CheckpointManager.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        model_type: Optional model type hint ('vae', 'decoder_only')
        
    Returns:
        model: Loaded model instance or decoder
        checkpoint: Full checkpoint dictionary
    """
    from conditioned_gesture_generator.gesture_vae.utils import CheckpointManager
    
    # Load model using CheckpointManager
    model = CheckpointManager.load_model_from_config(checkpoint_path, device=str(device))
    
    # Load checkpoint to return it as well
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # If it's a VAE but we only want the decoder, extract it
    if model_type == 'decoder_only' and hasattr(model, 'decoder'):
        print("Extracting decoder from VAE model")
        return model.decoder, checkpoint
    
    return model, checkpoint


def sample_latent(num_samples: int, latent_dim: int, device: torch.device, seed: Optional[int] = None) -> torch.Tensor:
    """
    Sample random latent vectors from standard normal distribution.
    
    Args:
        num_samples: Number of samples to generate
        latent_dim: Dimension of latent space
        device: Device to generate samples on
        seed: Optional random seed for reproducibility
        
    Returns:
        z: Sampled latent vectors [num_samples, latent_dim]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    z = torch.randn(num_samples, latent_dim, device=device)
    return z




def plot_trajectory(coords: np.ndarray, title: str = "Generated Gesture", save_path: Optional[Path] = None) -> go.Figure:
    """
    Plot 2D trajectory of gesture using Visualizer class.
    
    Args:
        coords: Coordinates array [B, T, 2] or [T, 2]
        title: Plot title
        save_path: Optional path to save plot
        
    Returns:
        Plotly figure object
    """
    if coords.ndim == 3:
        # If batch dimension, plot first sample
        coords = coords[0]
    
    # Create a temporary logger to use its trajectory splitting method
    logger = ExperimentLogger(project="temp", enabled=False)
    
    fig = go.Figure()
    
    # Split trajectory into segments filtering out zero coordinates
    segments = logger._split_trajectory_at_zeros(coords)
    
    if not segments:
        # No valid segments, create empty plot
        fig.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='markers',
            name='No valid trajectory',
            marker=dict(color='gray', size=5)
        ))
    else:
        # Plot each segment
        for seg_idx, segment in enumerate(segments):
            if len(segment) > 1:
                # Plot as connected line
                fig.add_trace(go.Scatter(
                    x=segment[:, 0],
                    y=segment[:, 1],
                    mode='lines+markers',
                    name='Trajectory' if seg_idx == 0 else None,
                    showlegend=(seg_idx == 0),
                    line=dict(color='blue', width=2),
                    marker=dict(size=3),
                    legendgroup='trajectory'
                ))
            elif len(segment) == 1:
                # Single point
                fig.add_trace(go.Scatter(
                    x=[segment[0, 0]],
                    y=[segment[0, 1]],
                    mode='markers',
                    showlegend=False,
                    marker=dict(color='blue', size=4)
                ))
        
        # Add start marker (first point of first segment)
        if segments and len(segments[0]) > 0:
            first_point = segments[0][0]
            fig.add_trace(go.Scatter(
                x=[first_point[0]],
                y=[first_point[1]],
                mode='markers',
                name='Start',
                marker=dict(color='green', size=10, symbol='circle')
            ))
        
        # Add end marker (last point of last segment)
        if segments and len(segments[-1]) > 0:
            last_point = segments[-1][-1]
            fig.add_trace(go.Scatter(
                x=[last_point[0]],
                y=[last_point[1]],
                mode='markers',
                name='End',
                marker=dict(color='red', size=10, symbol='square')
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=600,
        height=600,
        showlegend=True
    )
    
    if save_path:
        fig.write_html(str(save_path))
    
    return fig


def plot_time_series(coords: np.ndarray, title: str = "Gesture Time Series") -> go.Figure:
    """
    Plot time series of x and y coordinates.
    
    Args:
        coords: Coordinates array [B, T, 2] or [T, 2]
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    if coords.ndim == 3:
        coords = coords[0]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("X Coordinate", "Y Coordinate"),
        shared_xaxes=True
    )
    
    timesteps = np.arange(coords.shape[0])
    
    # X coordinate
    fig.add_trace(
        go.Scatter(x=timesteps, y=coords[:, 0], name='X', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Y coordinate
    fig.add_trace(
        go.Scatter(x=timesteps, y=coords[:, 1], name='Y', line=dict(color='red')),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Timestep", row=2, col=1)
    fig.update_yaxes(title_text="Value", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Value", range=[0, 1], row=2, col=1)
    
    fig.update_layout(
        title=title,
        height=600,
        showlegend=False
    )
    
    return fig


def interpolate_latent(z1: torch.Tensor, z2: torch.Tensor, num_steps: int = 10, 
                       method: str = 'linear') -> torch.Tensor:
    """
    Interpolate between two latent vectors.
    
    Args:
        z1: First latent vector [latent_dim]
        z2: Second latent vector [latent_dim]
        num_steps: Number of interpolation steps
        method: Interpolation method ('linear' or 'spherical')
        
    Returns:
        z_interp: Interpolated latent vectors [num_steps, latent_dim]
    """
    alphas = torch.linspace(0, 1, num_steps, device=z1.device)
    
    if method == 'linear':
        # Linear interpolation
        z_interp = []
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            z_interp.append(z)
        z_interp = torch.stack(z_interp)
    
    elif method == 'spherical':
        # Spherical interpolation (slerp)
        # Normalize vectors
        z1_norm = z1 / z1.norm()
        z2_norm = z2 / z2.norm()
        
        # Compute angle between vectors
        dot = (z1_norm * z2_norm).sum()
        theta = torch.acos(torch.clamp(dot, -1, 1))
        
        z_interp = []
        for alpha in alphas:
            if theta < 1e-6:
                # Vectors are parallel, use linear interpolation
                z = (1 - alpha) * z1 + alpha * z2
            else:
                # Spherical interpolation
                z = (torch.sin((1 - alpha) * theta) * z1 + torch.sin(alpha * theta) * z2) / torch.sin(theta)
            z_interp.append(z)
        z_interp = torch.stack(z_interp)
    
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    return z_interp


def plot_interpolation_grid(coords_list: List[np.ndarray], title: str = "Latent Interpolation") -> go.Figure:
    """
    Plot grid of interpolated trajectories using Visualizer class.
    
    Args:
        coords_list: List of coordinate arrays [T, 2]
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    num_interp = len(coords_list)
    cols = min(5, num_interp)
    rows = (num_interp + cols - 1) // cols
    
    # Create a temporary logger to use its trajectory splitting method
    logger = ExperimentLogger(project="temp", enabled=False)
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Step {i+1}" for i in range(num_interp)],
        specs=[[{'type': 'scatter'} for _ in range(cols)] for _ in range(rows)]
    )
    
    for idx, coords in enumerate(coords_list):
        row = idx // cols + 1
        col = idx % cols + 1
        
        # Color gradient from blue to red
        color_val = idx / (num_interp - 1) if num_interp > 1 else 0
        color = f'rgb({int(255*color_val)}, 0, {int(255*(1-color_val))})'
        
        # Split trajectory into segments filtering out zero coordinates
        segments = logger._split_trajectory_at_zeros(coords)
        
        # Plot each segment
        for seg_idx, segment in enumerate(segments):
            if len(segment) > 1:
                # Plot as connected line
                fig.add_trace(
                    go.Scatter(
                        x=segment[:, 0],
                        y=segment[:, 1],
                        mode='lines',
                        line=dict(color=color, width=2),
                        showlegend=False
                    ),
                    row=row, col=col
                )
            elif len(segment) == 1:
                # Single point
                fig.add_trace(
                    go.Scatter(
                        x=[segment[0, 0]],
                        y=[segment[0, 1]],
                        mode='markers',
                        marker=dict(color=color, size=4),
                        showlegend=False
                    ),
                    row=row, col=col
                )
    
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    
    fig.update_layout(
        title=title,
        height=200 * rows,
        width=200 * cols
    )
    
    return fig


def calculate_metrics(coords: np.ndarray) -> Dict[str, float]:
    """
    Calculate trajectory metrics.
    
    Args:
        coords: Coordinates array [B, T, 2] or [T, 2]
        
    Returns:
        Dictionary of metrics
    """
    if coords.ndim == 3:
        # Calculate metrics for each sample and average
        metrics_list = [calculate_metrics(coords[i]) for i in range(coords.shape[0])]
        return {
            key: np.mean([m[key] for m in metrics_list])
            for key in metrics_list[0].keys()
        }
    
    # Calculate total variation (smoothness)
    diffs = np.diff(coords, axis=0)
    total_variation = np.sum(np.linalg.norm(diffs, axis=1))
    
    # Calculate trajectory length
    trajectory_length = np.sum(np.linalg.norm(diffs, axis=1))
    
    # Calculate bounding box area (spatial coverage)
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    bbox_area = np.prod(max_coords - min_coords)
    
    # Calculate mean and std of coordinates
    mean_x, mean_y = np.mean(coords, axis=0)
    std_x, std_y = np.std(coords, axis=0)
    
    return {
        'total_variation': float(total_variation),
        'trajectory_length': float(trajectory_length),
        'bbox_area': float(bbox_area),
        'mean_x': float(mean_x),
        'mean_y': float(mean_y),
        'std_x': float(std_x),
        'std_y': float(std_y)
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb if available
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"eval_{Path(args.checkpoint).stem}",
            config=vars(args)
        )
    
    # Load model from checkpoint
    print(f"Loading model from {args.checkpoint}")
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device, model_type=args.model_type)
    
    # Determine latent dimension (prioritize model attributes over CLI args)
    if hasattr(model, 'd_latent'):
        latent_dim = model.d_latent
        if args.latent_dim and args.latent_dim != latent_dim:
            print(f"Warning: CLI latent_dim ({args.latent_dim}) overridden by model config ({latent_dim})")
    elif hasattr(model, 'latent_dim'):
        latent_dim = model.latent_dim
        if args.latent_dim and args.latent_dim != latent_dim:
            print(f"Warning: CLI latent_dim ({args.latent_dim}) overridden by model config ({latent_dim})")
    elif args.latent_dim:
        latent_dim = args.latent_dim
    else:
        raise ValueError("Cannot determine latent dimension. Please specify --latent_dim")
    
    print(f"Latent dimension: {latent_dim}")
    
    # Determine number of classes for quantizer (prioritize model attributes)
    if hasattr(model, 'k_classes'):
        k_classes = model.k_classes
        if args.k_classes != 3000 and args.k_classes != k_classes:  # 3000 is default
            print(f"Warning: CLI k_classes ({args.k_classes}) overridden by model config ({k_classes})")
    elif hasattr(model, 'num_classes'):
        k_classes = model.num_classes
        if args.k_classes != 3000 and args.k_classes != k_classes:
            print(f"Warning: CLI k_classes ({args.k_classes}) overridden by model config ({k_classes})")
    else:
        k_classes = args.k_classes
        print(f"Warning: Could not determine k_classes from model, using CLI value: {k_classes}")
    
    quantizer = CoordinateQuantizer(num_classes=k_classes)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample random latents
    print(f"Sampling {args.num_samples} random latent vectors")
    z_samples = sample_latent(args.num_samples, latent_dim, device, seed=args.seed)
    
    # Generate trajectories
    print("Generating trajectories from random samples")
    with torch.no_grad():
        if hasattr(model, 'decode'):
            logits = model.decode(z_samples)
        elif hasattr(model, 'decoder'):
            logits = model.decoder(z_samples)
        else:
            logits = model(z_samples)
        
        coords = quantizer.decode_to_trajectory(logits)
        coords_np = coords.cpu().numpy()
    
    # Plot samples
    for i in range(min(args.num_samples, 5)):  # Plot first 5 samples
        fig_traj = plot_trajectory(
            coords_np[i],
            title=f"Sample {i+1}",
            save_path=output_dir / f"trajectory_{i+1}.html"
        )
        
        fig_ts = plot_time_series(
            coords_np[i],
            title=f"Time Series - Sample {i+1}"
        )
        
        if WANDB_AVAILABLE and args.use_wandb:
            wandb.log({
                f"trajectory_{i+1}": wandb.Plotly(fig_traj),
                f"time_series_{i+1}": wandb.Plotly(fig_ts)
            })
    
    # Calculate and log metrics
    metrics = calculate_metrics(coords_np)
    print("\nTrajectory Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.log(metrics)
    
    # Latent interpolation
    if args.num_interpolations > 0:
        print(f"\nPerforming {args.num_interpolations} latent interpolations")
        
        for interp_idx in range(args.num_interpolations):
            # Sample two random latents
            z1 = sample_latent(1, latent_dim, device, seed=args.seed + interp_idx * 2 if args.seed else None)[0]
            z2 = sample_latent(1, latent_dim, device, seed=args.seed + interp_idx * 2 + 1 if args.seed else None)[0]
            
            # Interpolate
            z_interp = interpolate_latent(z1, z2, num_steps=args.interp_steps, method=args.interp_method)
            
            # Generate trajectories for interpolated latents
            with torch.no_grad():
                if hasattr(model, 'decode'):
                    logits_interp = model.decode(z_interp)
                elif hasattr(model, 'decoder'):
                    logits_interp = model.decoder(z_interp)
                else:
                    logits_interp = model(z_interp)
                
                coords_interp = quantizer.decode_to_trajectory(logits_interp)
                coords_interp_np = coords_interp.cpu().numpy()
            
            # Plot interpolation grid
            coords_list = [coords_interp_np[j] for j in range(coords_interp_np.shape[0])]
            fig_interp = plot_interpolation_grid(
                coords_list,
                title=f"Interpolation {interp_idx+1}"
            )
            
            # Save plot
            fig_interp.write_html(str(output_dir / f"interpolation_{interp_idx+1}.html"))
            
            if WANDB_AVAILABLE and args.use_wandb:
                wandb.log({f"interpolation_{interp_idx+1}": wandb.Plotly(fig_interp)})
            
            # Calculate smoothness of interpolation
            interp_metrics = []
            for j in range(len(coords_list) - 1):
                diff = np.mean(np.linalg.norm(coords_list[j+1] - coords_list[j]))
                interp_metrics.append(diff)
            
            avg_smoothness = np.mean(interp_metrics)
            std_smoothness = np.std(interp_metrics)
            
            print(f"  Interpolation {interp_idx+1} - Avg step diff: {avg_smoothness:.4f} ± {std_smoothness:.4f}")
            
            if WANDB_AVAILABLE and args.use_wandb:
                wandb.log({
                    f"interp_{interp_idx+1}_smoothness_mean": avg_smoothness,
                    f"interp_{interp_idx+1}_smoothness_std": std_smoothness
                })
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")
    
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VAE decoder with dynamic model loading")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    
    # Optional model parameters
    parser.add_argument("--model_type", type=str, default=None,
                        choices=["vae", "decoder_only"],
                        help="Model type hint (useful for VAE models when you only want decoder)")
    parser.add_argument("--latent_dim", type=int, default=None,
                        help="Latent dimension (will try to infer from model if not specified)")
    parser.add_argument("--k_classes", type=int, default=3000,
                        help="Number of quantization classes")
    
    # Sampling parameters
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of random samples to generate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    # Interpolation parameters
    parser.add_argument("--num_interpolations", type=int, default=3,
                        help="Number of interpolation experiments")
    parser.add_argument("--interp_steps", type=int, default=10,
                        help="Number of interpolation steps")
    parser.add_argument("--interp_method", type=str, default="linear",
                        choices=["linear", "spherical"],
                        help="Interpolation method")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="decoder_eval_results",
                        help="Directory to save results")
    
    # WandB parameters
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="vae_decoder_eval",
                        help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run name (defaults to checkpoint name)")
    
    args = parser.parse_args()
    main(args)