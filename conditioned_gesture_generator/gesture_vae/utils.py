import torch
import torch.nn as nn
import numpy as np
import os
import shutil
import signal
import sys
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Optional, Dict, Any, List, Tuple, Union
from torch.utils.data import Dataset, random_split
import warnings

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available - logging will be disabled")

# Quantization code
class CoordinateQuantizer:
    """Utilities for quantizing continuous coordinates to discrete classes."""
    
    def __init__(self, num_classes: int = 3000):
        self.num_classes = num_classes
    
    def quantize(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Quantize continuous coordinates to discrete class indices.
        
        Args:
            coords: Continuous coordinates [B, T, 2] in range [0, 1]
            
        Returns:
            class_indices: Discrete class indices [B, T, 2] in range [0, num_classes-1]
        """
        # Clamp to valid range and quantize
        coords_clamped = torch.clamp(coords, 0.0, 1.0)
        class_indices = (coords_clamped * self.num_classes).long()
        class_indices = torch.clamp(class_indices, 0, self.num_classes - 1)
        return class_indices
    
    def dequantize(self, class_indices: torch.Tensor) -> torch.Tensor:
        """
        Dequantize discrete class indices to continuous coordinates.
        
        Args:
            class_indices: Discrete class indices [B, T, 2]
            
        Returns:
            coords: Continuous coordinates [B, T, 2] in range [0, 1]
        """
        coords = class_indices.float() / self.num_classes
        return coords
    
    def argmax_decode(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Decode logits to coordinates via argmax.
        
        Args:
            logits: Classification logits [B, T, num_classes] or [B, T, 2, num_classes]
            
        Returns:
            coords: Decoded coordinates [B, T] or [B, T, 2] in range [0, 1]
        """
        class_indices = torch.argmax(logits, dim=-1)
        coords = class_indices.float() / self.num_classes
        return coords
    
    def decode_to_trajectory(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Decode logits to full 2D trajectory coordinates.
        Now handles separate x/y coordinate predictions.
        
        Args:
            logits: Classification logits [B, T, 2, num_classes] for x/y coordinates
            
        Returns:
            coords: Full coordinate trajectories [B, T, 2] in range [0, 1]
        """
        # Check logits shape to handle both old and new formats
        if len(logits.shape) == 3:  # Old format [B, T, num_classes] - single coordinate
            # Decode x-coordinates only
            x_coords = self.argmax_decode(logits)  # [B, T]
            # Set y-coordinates to center (0.5)
            batch_size, seq_len = x_coords.shape
            y_coords = torch.full_like(x_coords, 0.5)  # [B, T]
            # Stack to create [B, T, 2]
            coords = torch.stack([x_coords, y_coords], dim=-1)
        else:  # New format [B, T, 2, num_classes] - separate x/y
            # Decode both x and y coordinates using argmax
            coords = self.argmax_decode(logits)  # [B, T, 2]
        
        return coords


import os
import shutil
import signal
import json
import torch
import wandb
# matplotlib removed - using plotly instead
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List


class CheckpointManager:
    """
    Handles model checkpointing, source code saving, and graceful interruption handling.
    """
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5, 
                 save_source: bool = True, num_elite: int = 3):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of regular checkpoints to keep
            save_source: Whether to copy source code to checkpoint directory
            num_elite: Number of elite models to keep based on validation score
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.save_source = save_source
        self.num_elite = num_elite
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Track checkpoints for cleanup
        self.checkpoint_files = []
        
        # Elite model tracking: list of (score, filepath, metadata) tuples
        # Lower scores are better (loss values)
        self.elite_models = []
        
        # Setup graceful interruption handling
        self._setup_signal_handler()
        self._training_state = None
        
        # Copy source code if requested
        if save_source:
            self._copy_source_code()
            
        # Load existing elite models if resuming
        self._load_existing_elite_models()
    
    def _setup_signal_handler(self):
        """Setup signal handler for graceful interruption (Ctrl+C)."""
        def signal_handler(signum, frame):
            print("\nReceived interrupt signal. Saving checkpoint before exit...")
            if self._training_state is not None:
                self.save_final_checkpoint()
            print("Final checkpoint saved. Exiting...")
            exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _copy_source_code(self):
        """Copy source code files to checkpoint directory for reproducibility."""
        source_dir = os.path.join(self.checkpoint_dir, 'source_code')
        os.makedirs(source_dir, exist_ok=True)
        
        # Get the gesture_vae module directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Copy all Python files from the module
        for filename in os.listdir(current_dir):
            if filename.endswith('.py'):
                src_path = os.path.join(current_dir, filename)
                dst_path = os.path.join(source_dir, filename)
                shutil.copy2(src_path, dst_path)
        
        print(f"Source code copied to {source_dir}")
    
    def register_training_state(self, model, optimizer, epoch, train_metrics, val_metrics=None):
        """Register current training state for emergency saving."""
        self._training_state = {
            'model': model,
            'optimizer': optimizer,
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_checkpoint(self, model, optimizer, epoch: int, train_metrics: Dict[str, float], 
                       val_metrics: Optional[Dict[str, float]] = None, 
                       is_best: bool = False, tag: str = None, scheduler=None, 
                       step_counter: int = 0, best_val_loss: float = None,
                       steps_per_epoch: int = None) -> str:
        """
        Save training checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            is_best: Whether this is the best model so far
            tag: Optional tag for checkpoint filename
            scheduler: Optional learning rate scheduler to save
            step_counter: Current training step counter
            best_val_loss: Best validation loss so far
            steps_per_epoch: Number of steps per epoch (for resume estimation)
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'timestamp': datetime.now().isoformat(),
            'step_counter': step_counter,
            'best_val_loss': best_val_loss,
            'steps_per_epoch': steps_per_epoch,
            'model_config': {
                'd_latent': getattr(model, 'd_latent', None),
                'k_classes': getattr(model, 'k_classes', None),
            }
        }
        
        # Add scheduler state if provided
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Determine filename
        if tag:
            filename = f'checkpoint_{tag}_epoch_{epoch}.pt'
        else:
            filename = f'checkpoint_epoch_{epoch}.pt'
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Track for cleanup
        if not is_best and not tag:
            self.checkpoint_files.append(checkpoint_path)
            self._cleanup_old_checkpoints()
        
        # Save best model separately
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"New best model saved: {best_path}")
        
        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def save_final_checkpoint(self):
        """Save final checkpoint when training is interrupted."""
        if self._training_state is None:
            print("No training state registered for final checkpoint.")
            return
        
        state = self._training_state
        final_path = os.path.join(self.checkpoint_dir, 'final_checkpoint.pt')
        
        checkpoint = {
            'epoch': state['epoch'],
            'model_state_dict': state['model'].state_dict(),
            'optimizer_state_dict': state['optimizer'].state_dict(),
            'train_metrics': state['train_metrics'],
            'val_metrics': state['val_metrics'],
            'timestamp': state['timestamp'],
            'interrupted': True,
            'model_config': {
                'd_latent': getattr(state['model'], 'd_latent', None),
                'k_classes': getattr(state['model'], 'k_classes', None),
            }
        }
        
        torch.save(checkpoint, final_path)
        print(f"Final checkpoint saved: {final_path}")
    
    def load_checkpoint(self, checkpoint_path: str, model, optimizer=None):
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            
        Returns:
            Dictionary with checkpoint information
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}, Timestamp: {checkpoint.get('timestamp', 'N/A')}")
        
        return checkpoint
    
    def load_checkpoint_for_training_resume(self, checkpoint_path: str, model, optimizer, 
                                          scheduler=None, logger=None):
        """
        Load checkpoint for resuming training with full state restoration.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Optional learning rate scheduler to restore
            logger: Optional logger for experiment continuity
            
        Returns:
            Dict with resume information: {
                'start_epoch': int,
                'best_val_loss': float,
                'step_counter': int,
                'train_metrics': dict,
                'val_metrics': dict,
                'model_config': dict
            }
        """
        import os
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Extract training resume information
        start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        best_val_loss = float('inf')  # Reset best validation loss (or load if stored)
        
        # Try to extract additional training state if available
        train_metrics = checkpoint.get('train_metrics', {})
        val_metrics = checkpoint.get('val_metrics', {})
        model_config = checkpoint.get('model_config', {})
        
        # Handle best validation loss if available
        if val_metrics and 'total_loss' in val_metrics:
            best_val_loss = val_metrics['total_loss']
        elif 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        
        # Get step counter from checkpoint or estimate based on epoch
        step_counter = checkpoint.get('step_counter', 0)
        
        # If step_counter is 0 or not available, estimate from epoch
        if step_counter == 0 and 'steps_per_epoch' in checkpoint:
            step_counter = checkpoint['epoch'] * checkpoint['steps_per_epoch']
        elif step_counter == 0:
            print(f"Warning: Step counter not found in checkpoint, will start from 0")
        
        # Handle scheduler state if provided and available
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"Scheduler state restored")
            except Exception as e:
                print(f"Warning: Could not restore scheduler state: {e}")
        
        # Update scheduler to correct step if we don't have stored state
        if scheduler is not None and 'scheduler_state_dict' not in checkpoint:
            # Fast-forward scheduler to the correct epoch
            for _ in range(checkpoint['epoch']):
                scheduler.step()
            print(f"Scheduler fast-forwarded to epoch {checkpoint['epoch']}")
        
        print(f"Training will resume from epoch {start_epoch}")
        print(f"Previous epoch: {checkpoint['epoch']}")
        print(f"Previous train metrics: {train_metrics}")
        if val_metrics:
            print(f"Previous val metrics: {val_metrics}")
            print(f"Best validation loss: {best_val_loss:.4f}")
        
        resume_info = {
            'start_epoch': start_epoch,
            'best_val_loss': best_val_loss,
            'step_counter': step_counter,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model_config': model_config,
            'checkpoint_epoch': checkpoint['epoch'],
            'checkpoint_timestamp': checkpoint.get('timestamp', 'N/A')
        }
        
        return resume_info
    
    def _load_existing_elite_models(self):
        """Load existing elite models from checkpoint directory when resuming."""
        import glob
        
        # Look for elite checkpoint files
        elite_pattern = os.path.join(self.checkpoint_dir, 'elite_*.pt')
        elite_files = glob.glob(elite_pattern)
        
        for elite_file in elite_files:
            try:
                # Load checkpoint to get score
                checkpoint = torch.load(elite_file, map_location='cpu')
                if 'val_metrics' in checkpoint and checkpoint['val_metrics']:
                    score = checkpoint['val_metrics'].get('total_loss', float('inf'))
                    metadata = {
                        'epoch': checkpoint.get('epoch', 0),
                        'timestamp': checkpoint.get('timestamp', 'unknown'),
                        'train_metrics': checkpoint.get('train_metrics', {}),
                        'val_metrics': checkpoint.get('val_metrics', {})
                    }
                    self.elite_models.append((score, elite_file, metadata))
            except Exception as e:
                print(f"Warning: Could not load elite checkpoint {elite_file}: {e}")
        
        # Sort by score (lower is better)
        self.elite_models.sort(key=lambda x: x[0])
        
        if self.elite_models:
            print(f"Loaded {len(self.elite_models)} existing elite models:")
            for i, (score, filepath, metadata) in enumerate(self.elite_models):
                basename = os.path.basename(filepath)
                epoch = metadata.get('epoch', 'unknown')
                print(f"  {i+1}. {basename} - Score: {score:.6f} (Epoch {epoch})")
    
    def save_elite_model(self, model, optimizer, epoch: int, val_score: float,
                        train_metrics: Dict[str, float], val_metrics: Dict[str, float],
                        scheduler=None, step_counter: int = 0, steps_per_epoch: int = None):
        """
        Save model if it qualifies as elite based on validation score.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            val_score: Validation score (lower is better)
            train_metrics: Training metrics
            val_metrics: Validation metrics
            scheduler: Optional scheduler
            step_counter: Current step
            steps_per_epoch: Steps per epoch
        
        Returns:
            bool: True if model was saved as elite, False otherwise
        """
        # Check if this model qualifies as elite
        if self._is_elite(val_score):
            # Create elite checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'timestamp': datetime.now().isoformat(),
                'step_counter': step_counter,
                'best_val_loss': val_score,
                'steps_per_epoch': steps_per_epoch,
                'elite_rank': len([x for x in self.elite_models if x[0] < val_score]) + 1,
                'model_config': {
                    'd_latent': getattr(model, 'd_latent', None),
                    'k_classes': getattr(model, 'k_classes', None),
                }
            }
            
            # Add scheduler state if provided
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            # Generate elite filename
            elite_filename = f'elite_{epoch:04d}_score_{val_score:.6f}.pt'
            elite_path = os.path.join(self.checkpoint_dir, elite_filename)
            
            # Save checkpoint
            torch.save(checkpoint, elite_path)
            
            # Update elite tracking
            metadata = {
                'epoch': epoch,
                'timestamp': checkpoint['timestamp'],
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            
            self.elite_models.append((val_score, elite_path, metadata))
            self.elite_models.sort(key=lambda x: x[0])  # Sort by score
            
            # Remove excess elite models
            self._cleanup_elite_models()
            
            # Determine rank
            current_rank = next(i for i, (score, _, _) in enumerate(self.elite_models) 
                              if score == val_score) + 1
            
            print(f"🏆 Elite model saved! Rank: {current_rank}/{len(self.elite_models)}")
            print(f"   Score: {val_score:.6f}, Epoch: {epoch}")
            print(f"   File: {os.path.basename(elite_path)}")
            
            return True
        
        return False
    
    def _is_elite(self, val_score: float) -> bool:
        """
        Check if a validation score qualifies as elite.
        
        Args:
            val_score: Validation score to check
            
        Returns:
            bool: True if score qualifies as elite
        """
        if len(self.elite_models) < self.num_elite:
            return True
        
        # Check if score is better than the worst elite
        worst_elite_score = self.elite_models[-1][0]
        return val_score < worst_elite_score
    
    def _cleanup_elite_models(self):
        """Remove excess elite models, keeping only the best num_elite."""
        if len(self.elite_models) > self.num_elite:
            # Remove worst models
            models_to_remove = self.elite_models[self.num_elite:]
            
            for score, filepath, metadata in models_to_remove:
                try:
                    os.remove(filepath)
                    print(f"Removed elite checkpoint: {os.path.basename(filepath)} (score: {score:.6f})")
                except OSError as e:
                    print(f"Warning: Could not remove {filepath}: {e}")
            
            # Keep only the best models
            self.elite_models = self.elite_models[:self.num_elite]
    
    def get_elite_models_info(self) -> List[Dict]:
        """
        Get information about current elite models.
        
        Returns:
            List of dictionaries with elite model information
        """
        elite_info = []
        for rank, (score, filepath, metadata) in enumerate(self.elite_models, 1):
            info = {
                'rank': rank,
                'score': score,
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'epoch': metadata.get('epoch', 'unknown'),
                'timestamp': metadata.get('timestamp', 'unknown'),
                'exists': os.path.exists(filepath)
            }
            elite_info.append(info)
        
        return elite_info
    
    def get_best_elite_model(self) -> Optional[str]:
        """
        Get the path to the best elite model.
        
        Returns:
            Path to best elite model or None if no elite models exist
        """
        if self.elite_models:
            return self.elite_models[0][1]  # First element has the best score
        return None
    
    def find_latest_checkpoint(self, checkpoint_dir: str = None) -> str:
        """
        Find the latest checkpoint in the checkpoint directory.
        
        Args:
            checkpoint_dir: Directory to search (defaults to self.checkpoint_dir)
            
        Returns:
            Path to the latest checkpoint file
            
        Raises:
            FileNotFoundError: If no checkpoints found
        """
        import glob
        
        search_dir = checkpoint_dir or self.checkpoint_dir
        
        # Look for different checkpoint patterns
        patterns = [
            os.path.join(search_dir, 'checkpoint_epoch_*.pt'),
            os.path.join(search_dir, 'best_model.pt'),
            os.path.join(search_dir, 'final_checkpoint.pt')
        ]
        
        all_checkpoints = []
        for pattern in patterns:
            all_checkpoints.extend(glob.glob(pattern))
        
        if not all_checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {search_dir}")
        
        # Sort by modification time and return the most recent
        latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
        
        print(f"Found {len(all_checkpoints)} checkpoints in {search_dir}")
        print(f"Latest checkpoint: {latest_checkpoint}")
        
        return latest_checkpoint
    
    def list_available_checkpoints(self, checkpoint_dir: str = None) -> List[str]:
        """
        List all available checkpoints with their details.
        
        Args:
            checkpoint_dir: Directory to search (defaults to self.checkpoint_dir)
            
        Returns:
            List of checkpoint paths sorted by modification time (newest first)
        """
        import glob
        
        search_dir = checkpoint_dir or self.checkpoint_dir
        
        patterns = [
            os.path.join(search_dir, 'checkpoint_epoch_*.pt'),
            os.path.join(search_dir, 'best_model.pt'),
            os.path.join(search_dir, 'final_checkpoint.pt')
        ]
        
        all_checkpoints = []
        for pattern in patterns:
            all_checkpoints.extend(glob.glob(pattern))
        
        if not all_checkpoints:
            return []
        
        # Sort by modification time (newest first)
        all_checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        print(f"Available checkpoints in {search_dir}:")
        for i, checkpoint in enumerate(all_checkpoints):
            try:
                # Try to load checkpoint info
                ckpt = torch.load(checkpoint, map_location='cpu')
                epoch = ckpt.get('epoch', 'N/A')
                timestamp = ckpt.get('timestamp', 'N/A')
                print(f"  {i+1}. {os.path.basename(checkpoint)} - Epoch {epoch} - {timestamp}")
            except Exception as e:
                print(f"  {i+1}. {os.path.basename(checkpoint)} - Error loading: {e}")
        
        return all_checkpoints
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        if len(self.checkpoint_files) > self.max_checkpoints:
            # Sort by modification time
            self.checkpoint_files.sort(key=os.path.getmtime)
            
            # Remove oldest checkpoints
            while len(self.checkpoint_files) > self.max_checkpoints:
                old_checkpoint = self.checkpoint_files.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    print(f"Removed old checkpoint: {old_checkpoint}")


class ExperimentLogger:
    """
    Handles experiment logging with wandb integration.
    """
    
    def __init__(self, project: str, experiment_name: str = None, 
                 config: Dict[str, Any] = None, enabled: bool = True):
        """
        Args:
            project: wandb project name
            experiment_name: Name for this experiment run
            config: Configuration dictionary to log
            enabled: Whether to enable logging (useful for debugging)
        """
        self.enabled = enabled
        self.project = project
        self.experiment_name = experiment_name
        
        if self.enabled:
            # Initialize wandb
            wandb.init(
                project=project,
                name=experiment_name,
                config=config,
                reinit=True
            )
            print(f"Initialized wandb logging for project: {project}")
        else:
            print("Logging disabled")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None, prefix: str = None):
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Step number (epoch, iteration, etc.)
            prefix: Optional prefix for metric names
        """
        if not self.enabled:
            return
        
        # Add prefix if specified
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Log to wandb
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    
    def log_model_samples(self, samples: torch.Tensor, step: int, 
                         sample_type: str = "samples"):
        """
        Log model samples as images.
        
        Args:
            samples: Generated samples [B, 250, 2] or visualization images
            step: Step number
            sample_type: Type of samples (for labeling)
        """
        if not self.enabled:
            return
        
        if len(samples.shape) == 3 and samples.shape[-1] == 2:
            # Convert coordinate sequences to trajectory plots
            fig = self._plot_trajectories(samples[:8])  # Plot first 8 samples
            wandb.log({f"{sample_type}_trajectories": fig}, step=step)  # Plotly figures work directly with wandb
        elif len(samples.shape) == 4:  # Assume BHWC or BCHW image format
            # Assume it's already an image
            wandb.log({sample_type: wandb.Image(samples)}, step=step)
        else:
            # For other formats, create a simple plot
            print(f"Warning: Unsupported sample shape {samples.shape} for logging")
    
    def log_val_compare_plots(self, original: torch.Tensor, reconstructed: torch.Tensor,
                             sampled: torch.Tensor, step: int):
        """
        Log individual trajectory comparison plots under val_compare.
        Creates separate 2D trajectory and time series plots for each sample.
        
        Args:
            original: Original coordinate sequences [B, 250, 2]
            reconstructed: Reconstructed coordinate sequences [B, 250, 2] 
            sampled: Sampled coordinate sequences [B, 250, 2]
            step: Step number
        """
        if not self.enabled:
            return
            
        num_samples = min(20, original.shape[0])
        log_data = {}
        
        for i in range(num_samples):
            # Create 2D trajectory comparison plot
            fig_2d = self._create_single_trajectory_comparison(
                original[i], reconstructed[i], sample_idx=i+1
            )
            log_data[f"val_compare/recon_{i+1}"] = fig_2d
            
            # Create time series comparison plot
            fig_ts = self._create_single_time_series_comparison(
                original[i], reconstructed[i], sample_idx=i+1
            )
            log_data[f"val_compare/recon_time_series_{i+1}"] = fig_ts
        
        wandb.log(log_data, step=step)
    
    def _create_single_trajectory_comparison(self, original: torch.Tensor, reconstructed: torch.Tensor, sample_idx: int):
        """Create a single 2D trajectory comparison plot."""
        fig = go.Figure()
        
        # Convert to numpy
        orig_np = original.detach().cpu().numpy()
        recon_np = reconstructed.detach().cpu().numpy()
        
        # Plot original trajectory segments
        orig_segments = self._split_trajectory_at_zeros(orig_np)
        for seg_idx, segment in enumerate(orig_segments):
            if len(segment) > 1:
                fig.add_trace(go.Scatter(
                    x=segment[:, 0], y=segment[:, 1],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Original' if seg_idx == 0 else None,
                    showlegend=(seg_idx == 0),
                    legendgroup='original'
                ))
            elif len(segment) == 1:
                fig.add_trace(go.Scatter(
                    x=[segment[0, 0]], y=[segment[0, 1]],
                    mode='markers',
                    marker=dict(color='blue', size=4),
                    showlegend=False
                ))
        
        # Plot reconstructed trajectory segments
        recon_segments = self._split_trajectory_at_zeros(recon_np)
        for seg_idx, segment in enumerate(recon_segments):
            if len(segment) > 1:
                fig.add_trace(go.Scatter(
                    x=segment[:, 0], y=segment[:, 1],
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    name='Reconstructed' if seg_idx == 0 else None,
                    showlegend=(seg_idx == 0),
                    legendgroup='reconstructed'
                ))
            elif len(segment) == 1:
                fig.add_trace(go.Scatter(
                    x=[segment[0, 0]], y=[segment[0, 1]],
                    mode='markers',
                    marker=dict(color='red', size=4),
                    showlegend=False
                ))
        
        # Add start/end markers for original
        if orig_segments:
            first_segment = orig_segments[0]
            last_segment = orig_segments[-1]
            
            fig.add_trace(go.Scatter(
                x=[first_segment[0, 0]], y=[first_segment[0, 1]],
                mode='markers',
                marker=dict(color='darkgreen', size=8, symbol='circle'),
                name='Start',
                legendgroup='start'
            ))
            
            fig.add_trace(go.Scatter(
                x=[last_segment[-1, 0]], y=[last_segment[-1, 1]],
                mode='markers',
                marker=dict(color='darkred', size=8, symbol='x'),
                name='End',
                legendgroup='end'
            ))
        
        fig.update_layout(
            title=f"2D Trajectory Comparison - Sample {sample_idx}",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1], scaleanchor="x", scaleratio=1),
            width=600,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _create_single_time_series_comparison(self, original: torch.Tensor, reconstructed: torch.Tensor, sample_idx: int):
        """Create a single time series comparison plot."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['X Coordinate vs Time', 'Y Coordinate vs Time'],
            vertical_spacing=0.1
        )
        
        # Convert to numpy
        orig_np = original.detach().cpu().numpy()
        recon_np = reconstructed.detach().cpu().numpy()
        
        # Find gesture start (ignore leading zeros)
        orig_start = self._find_gesture_start(orig_np)
        recon_start = self._find_gesture_start(recon_np)
        start_idx = min(orig_start, recon_start)
        time_steps = np.arange(start_idx, len(orig_np))
        
        # X coordinate time series
        fig.add_trace(go.Scatter(
            x=time_steps, y=orig_np[start_idx:, 0],
            mode='lines', name='Original X', line=dict(color='blue', width=2),
            showlegend=True
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=time_steps, y=recon_np[start_idx:, 0],
            mode='lines', name='Reconstructed X', line=dict(color='red', width=2, dash='dash'),
            showlegend=True
        ), row=1, col=1)
        
        # Y coordinate time series
        fig.add_trace(go.Scatter(
            x=time_steps, y=orig_np[start_idx:, 1],
            mode='lines', name='Original Y', line=dict(color='green', width=2),
            showlegend=True
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=time_steps, y=recon_np[start_idx:, 1],
            mode='lines', name='Reconstructed Y', line=dict(color='orange', width=2, dash='dash'),
            showlegend=True
        ), row=2, col=1)
        
        fig.update_layout(
            title=f"Time Series Comparison - Sample {sample_idx}",
            width=800,
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time Step", range=[0, 250])
        fig.update_yaxes(title_text="Position", range=[0, 1])
        
        return fig
    
    def log_loss_histogram(self, losses: List[float], step: int, loss_type: str = "reconstruction"):
        """
        Log histogram of losses using Plotly for interactivity.
        
        Args:
            losses: List of loss values
            step: Step number
            loss_type: Type of loss (for title/naming)
        """
        if not self.enabled or not losses:
            return
        
        mean_loss = np.mean(losses)
        
        # Create interactive histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=losses,
            nbinsx=50,
            name=f'{loss_type.title()} Loss',
            opacity=0.7,
            marker_color='lightblue',
            marker_line_color='black',
            marker_line_width=1
        ))
        
        # Add mean line
        fig.add_vline(
            x=mean_loss, 
            line_dash='dash', 
            line_color='red',
            line_width=2,
            annotation_text=f'Mean: {mean_loss:.6f}',
            annotation_position="top right"
        )
        
        fig.update_layout(
            title=f'{loss_type.title()} Loss Distribution',
            xaxis_title=f'{loss_type.title()} Loss',
            yaxis_title='Count',
            showlegend=True,
            width=1200,
            height=800
        )
        
        wandb.log({f"{loss_type}_loss_histogram": fig}, step=step)
    
    def log_summary_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log summary metrics (like in no_int classifier).
        
        Args:
            metrics: Dictionary of summary metrics
            step: Step number
        """
        if not self.enabled:
            return
        
        # Add step to metrics and log
        summary_metrics = {f"summary/{k}": v for k, v in metrics.items()}
        summary_metrics["step"] = step
        wandb.log(summary_metrics)
    
    def log_model_architecture(self, model, model_summary_text=None):
        """Log model architecture and parameter count."""
        if not self.enabled:
            return
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.log({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params
        })
        
        # Log detailed model summary if provided
        if model_summary_text:
            wandb.log({"model/architecture_summary": wandb.Html(f"<pre>{model_summary_text}</pre>")})
        else:
            wandb.log({"model/architecture": str(model)})
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        if not self.enabled:
            return
        wandb.config.update(hparams)
    
    def save_artifact(self, file_path: str, name: str, type: str = "model"):
        """
        Save file as wandb artifact.
        
        Args:
            file_path: Path to file
            name: Artifact name
            type: Artifact type
        """
        if not self.enabled:
            return
        
        artifact = wandb.Artifact(name=name, type=type)
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)
    
    def _plot_trajectories(self, trajectories: torch.Tensor):
        """Plot coordinate trajectories using Plotly."""
        n_trajectories = min(8, len(trajectories))
        
        # Create subplots in 2x4 grid
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=[f'Sample {i+1}' for i in range(n_trajectories)],
            specs=[[{'type': 'xy'} for _ in range(4)] for _ in range(2)]
        )
        
        for i, traj in enumerate(trajectories):
            if i >= 8:
                break
            
            row = i // 4 + 1
            col = i % 4 + 1
            
            traj_np = traj.detach().cpu().numpy()
            # Split trajectory into segments at pen lifts (zeros)
            segments = self._split_trajectory_at_zeros(traj_np)
            
            if len(segments) > 0:
                # Plot each continuous segment separately
                for seg_idx, segment in enumerate(segments):
                    if len(segment) > 1:
                        fig.add_trace(go.Scatter(
                            x=segment[:, 0], y=segment[:, 1],
                            mode='lines',
                            line=dict(color='blue', width=2),
                            name='Trajectory' if i == 0 and seg_idx == 0 else None,
                            showlegend=(i == 0 and seg_idx == 0),
                            legendgroup='trajectory'
                        ), row=row, col=col)
                    elif len(segment) == 1:
                        fig.add_trace(go.Scatter(
                            x=[segment[0, 0]], y=[segment[0, 1]],
                            mode='markers',
                            marker=dict(color='blue', size=5),
                            showlegend=False
                        ), row=row, col=col)
                
                # Mark overall start and end
                first_segment = segments[0]
                last_segment = segments[-1]
                
                fig.add_trace(go.Scatter(
                    x=[first_segment[0, 0]], y=[first_segment[0, 1]],
                    mode='markers',
                    marker=dict(color='green', size=8, symbol='circle'),
                    name='Start' if i == 0 else None,
                    showlegend=(i == 0),
                    legendgroup='start'
                ), row=row, col=col)
                
                fig.add_trace(go.Scatter(
                    x=[last_segment[-1, 0]], y=[last_segment[-1, 1]],
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='x'),
                    name='End' if i == 0 else None,
                    showlegend=(i == 0),
                    legendgroup='end'
                ), row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title="Trajectory Samples",
            height=800,
            width=1600,
            showlegend=True
        )
        
        # Update all axes
        fig.update_xaxes(range=[0, 1], title_text="X")
        fig.update_yaxes(range=[0, 1], title_text="Y", scaleanchor="x", scaleratio=1)
        
        return fig
    
    def _find_gesture_start(self, coords):
        """Find the first non-zero point for both x and y coordinates."""
        x_nonzero = np.where(coords[:, 0] != 0)[0]
        y_nonzero = np.where(coords[:, 1] != 0)[0]
        
        if len(x_nonzero) == 0 and len(y_nonzero) == 0:
            return 0  # All zeros, start from beginning
        elif len(x_nonzero) == 0:
            return y_nonzero[0]
        elif len(y_nonzero) == 0:
            return x_nonzero[0]
        else:
            return min(x_nonzero[0], y_nonzero[0])
    
    def _split_trajectory_at_zeros(self, coords):
        """
        Split trajectory into continuous segments, breaking at zero coordinates (pen lifts).
        
        Args:
            coords: numpy array of shape [T, 2] with x,y coordinates
            
        Returns:
            List of trajectory segments, each a numpy array [segment_length, 2]
        """
        segments = []
        current_segment = []
        
        for i, point in enumerate(coords):
            # Check if point is a pen lift (both x and y are zero)
            if point[0] == 0 and point[1] == 0:
                # End current segment if it has points
                if len(current_segment) > 0:
                    segments.append(np.array(current_segment))
                    current_segment = []
            else:
                # Add non-zero point to current segment
                current_segment.append(point)
        
        # Add final segment if it has points
        if len(current_segment) > 0:
            segments.append(np.array(current_segment))
        
        return segments
    
    def _plot_trajectory_comparison(self, original: torch.Tensor, reconstructed: torch.Tensor, 
                                   sampled: torch.Tensor):
        """Plot comparison with 2D trajectories showing both original and reconstructed using Plotly.
        Organizes 20 samples into 4 subplots with 5 samples each."""
        n_samples = min(20, original.shape[0])
        
        # Create 4 subplots (2x2 grid), each showing 5 samples
        rows, cols = 2, 2
        samples_per_subplot = 5
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f'Samples {i*samples_per_subplot+1}-{min((i+1)*samples_per_subplot, n_samples)}' 
                          for i in range(rows*cols)],
            specs=[[{'type': 'xy'} for _ in range(cols)] for _ in range(rows)],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Define colors for original and reconstructed
        orig_color = 'blue'
        recon_color = 'red'
        start_color = 'darkgreen'
        end_color = 'darkred'
        
        for sample_idx in range(n_samples):
            # Determine which subplot this sample belongs to
            subplot_idx = sample_idx // samples_per_subplot
            row = (subplot_idx // cols) + 1
            col = (subplot_idx % cols) + 1
            
            # Plot original trajectory
            orig_traj = original[sample_idx].detach().cpu().numpy()
            segments = self._split_trajectory_at_zeros(orig_traj)
            
            if len(segments) > 0:
                for seg_idx, segment in enumerate(segments):
                    if len(segment) > 1:
                        fig.add_trace(go.Scatter(
                            x=segment[:, 0], y=segment[:, 1],
                            mode='lines',
                            line=dict(color=orig_color, width=2),
                            name='Original' if sample_idx == 0 and seg_idx == 0 else None,
                            showlegend=(sample_idx == 0 and seg_idx == 0),
                            legendgroup='original'
                        ), row=row, col=col)
                    elif len(segment) == 1:
                        fig.add_trace(go.Scatter(
                            x=[segment[0, 0]], y=[segment[0, 1]],
                            mode='markers',
                            marker=dict(color=orig_color, size=4),
                            showlegend=False
                        ), row=row, col=col)
                
                # Mark start and end for original
                first_segment = segments[0]
                last_segment = segments[-1]
                
                fig.add_trace(go.Scatter(
                    x=[first_segment[0, 0]], y=[first_segment[0, 1]],
                    mode='markers',
                    marker=dict(color=start_color, size=6, symbol='circle'),
                    name='Start' if sample_idx == 0 else None,
                    showlegend=(sample_idx == 0),
                    legendgroup='start'
                ), row=row, col=col)
                
                fig.add_trace(go.Scatter(
                    x=[last_segment[-1, 0]], y=[last_segment[-1, 1]],
                    mode='markers',
                    marker=dict(color=end_color, size=6, symbol='x'),
                    name='End' if sample_idx == 0 else None,
                    showlegend=(sample_idx == 0),
                    legendgroup='end'
                ), row=row, col=col)
            
            # Plot reconstructed trajectory
            recon_traj = reconstructed[sample_idx].detach().cpu().numpy()
            segments = self._split_trajectory_at_zeros(recon_traj)
            
            if len(segments) > 0:
                for seg_idx, segment in enumerate(segments):
                    if len(segment) > 1:
                        fig.add_trace(go.Scatter(
                            x=segment[:, 0], y=segment[:, 1],
                            mode='lines',
                            line=dict(color=recon_color, width=2, dash='dash'),
                            name='Reconstructed' if sample_idx == 0 and seg_idx == 0 else None,
                            showlegend=(sample_idx == 0 and seg_idx == 0),
                            legendgroup='reconstructed'
                        ), row=row, col=col)
                    elif len(segment) == 1:
                        fig.add_trace(go.Scatter(
                            x=[segment[0, 0]], y=[segment[0, 1]],
                            mode='markers',
                            marker=dict(color=recon_color, size=4),
                            showlegend=False
                        ), row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title="Original vs Reconstructed Trajectories (20 samples in 4 groups)",
            height=800,
            width=1600,
            showlegend=True
        )
        
        # Update all axes
        fig.update_xaxes(range=[0, 1], title_text="X Position")
        fig.update_yaxes(range=[0, 1], title_text="Y Position", scaleanchor="x", scaleratio=1)
        
        return fig
    
    def finish(self):
        """Finish logging session."""
        if self.enabled:
            wandb.finish()


class Visualizer:
    """
    Handles visualization tasks that complement wandb logging.
    """
    
    def __init__(self, save_dir: str = None):
        """
        Args:
            save_dir: Directory to save visualization files
        """
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_curves(self, train_history: List[Dict], val_history: List[Dict] = None, 
                           save_path: str = None):
        """
        Plot training curves using Plotly.
        
        Args:
            train_history: List of training metrics per epoch
            val_history: List of validation metrics per epoch
            save_path: Optional path to save plot
        """
        epochs = list(range(1, len(train_history) + 1))
        
        # Extract metrics
        metrics = ['total_loss', 'recon_loss', 'kl_loss']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[metric.replace('_', ' ').title() + ' Over Time' for metric in metrics]
        )
        
        colors = {'train': 'blue', 'val': 'red'}
        
        for i, metric in enumerate(metrics):
            # Training curves
            train_values = [h[metric] for h in train_history]
            fig.add_trace(go.Scatter(
                x=epochs, y=train_values,
                mode='lines',
                name=f'Train {metric}',
                line=dict(color=colors['train'], width=2),
                showlegend=(i == 0),
                legendgroup='train'
            ), row=1, col=i+1)
            
            # Validation curves
            if val_history:
                val_values = [h[metric] for h in val_history]
                fig.add_trace(go.Scatter(
                    x=epochs, y=val_values,
                    mode='lines',
                    name=f'Val {metric}',
                    line=dict(color=colors['val'], width=2),
                    showlegend=(i == 0),
                    legendgroup='val'
                ), row=1, col=i+1)
        
        fig.update_layout(
            title="Training Progress",
            height=600,
            width=1600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Loss Value")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_latent_space(self, model, data_loader, device, max_samples=1000, 
                              save_path: str = None):
        """
        Visualize latent space using t-SNE or PCA with Plotly.
        
        Args:
            model: Trained model
            data_loader: Data loader
            device: Device
            max_samples: Maximum samples to include
            save_path: Optional save path
        """
        model.eval()
        latents = []
        
        with torch.no_grad():
            sample_count = 0
            for batch in data_loader:
                if sample_count >= max_samples:
                    break
                
                batch = batch.to(device)
                mu, _ = model.encode(batch)
                latents.append(mu.cpu().numpy())
                sample_count += batch.shape[0]
        
        latents = np.concatenate(latents, axis=0)[:max_samples]
        
        # Use PCA for 2D visualization if latent dim > 2
        if latents.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            latents_2d = pca.fit_transform(latents)
            explained_variance = pca.explained_variance_ratio_
            title = f'Latent Space Visualization (PCA: {explained_variance[0]:.1%} + {explained_variance[1]:.1%} variance)'
        else:
            latents_2d = latents
            title = 'Latent Space Visualization'
        
        # Create interactive scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=latents_2d[:, 0],
            y=latents_2d[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=np.arange(len(latents_2d)),
                colorscale='Viridis',
                opacity=0.6,
                showscale=True,
                colorbar=dict(title="Sample Index")
            ),
            text=[f'Sample {i}' for i in range(len(latents_2d))],
            hovertemplate='<b>%{text}</b><br>Dim 1: %{x:.3f}<br>Dim 2: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Latent Dimension 1',
            yaxis_title='Latent Dimension 2',
            width=1200,
            height=800,
            hovermode='closest'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_reconstructions(self, model, data_batch, quantizer, device, 
                                 num_samples=4, save_path: str = None):
        """
        Visualize original vs reconstructed sequences using Plotly.
        
        Args:
            model: Trained model
            data_batch: Batch of input data
            quantizer: Coordinate quantizer
            device: Device
            num_samples: Number of samples to visualize
            save_path: Optional save path
        """
        model.eval()
        
        with torch.no_grad():
            data_batch = data_batch[:num_samples].to(device)
            
            # Get reconstructions
            outputs = model(data_batch)
            logits = outputs['logits']
            
            # Decode logits back to coordinates
            pred_classes = torch.argmax(logits, dim=-1)  # [B, 250]
            reconstructed = quantizer.dequantize(pred_classes.unsqueeze(-1))  # [B, 250, 1]
            
            # Prepare for plotting (assuming x-coordinate only for now)
            original = data_batch.cpu().numpy()
            reconstructed = reconstructed.cpu().numpy()
        
        # Create subplots: 2 rows (original, reconstructed) x num_samples columns
        subplot_titles = [f'Original {i+1}' for i in range(num_samples)] + \
                        [f'Reconstructed {i+1}' for i in range(num_samples)]
        
        fig = make_subplots(
            rows=2, cols=num_samples,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1
        )
        
        time_steps = np.arange(original.shape[1])
        
        for i in range(num_samples):
            # Original
            fig.add_trace(go.Scatter(
                x=time_steps, y=original[i, :, 0],
                mode='lines', name='Original X', line=dict(color='blue', width=2),
                showlegend=(i == 0), legendgroup='orig_x'
            ), row=1, col=i+1)
            
            fig.add_trace(go.Scatter(
                x=time_steps, y=original[i, :, 1],
                mode='lines', name='Original Y', line=dict(color='green', width=2),
                showlegend=(i == 0), legendgroup='orig_y'
            ), row=1, col=i+1)
            
            # Reconstructed
            fig.add_trace(go.Scatter(
                x=time_steps, y=reconstructed[i, :, 0],
                mode='lines', name='Reconstructed X', line=dict(color='red', width=2, dash='dash'),
                showlegend=(i == 0), legendgroup='recon_x'
            ), row=2, col=i+1)
            
            fig.add_trace(go.Scatter(
                x=time_steps, y=original[i, :, 1],
                mode='lines', name='Original Y', line=dict(color='green', width=2, dash='dot'),
                showlegend=False, legendgroup='orig_y'
            ), row=2, col=i+1)
        
        fig.update_layout(
            title="Original vs Reconstructed Sequences",
            height=800,
            width=400 * num_samples,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time Step")
        fig.update_yaxes(title_text="Position", range=[0, 1])
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_logit_heatmap(self, logits, target_coords, quantizer, 
                               title_prefix="", max_samples=3, save_path: str = None):
        """
        Visualize logit distributions as interactive heatmaps using Plotly.
        
        Args:
            logits: Model logits [B, T, k_classes] 
            target_coords: Target coordinates [B, T, 2]
            quantizer: Coordinate quantizer
            title_prefix: Title prefix for plots
            max_samples: Number of samples to visualize
            save_path: Optional save path
        """
        import torch.nn.functional as F
        
        batch_size = min(max_samples, logits.shape[0])
        seq_len = logits.shape[1]
        num_classes = logits.shape[-1]
        
        # Create subplots for X and Y coordinate heatmaps
        subplot_titles = []
        for i in range(batch_size):
            subplot_titles.extend([f"{title_prefix} X Logit Distribution (Sample {i+1})",
                                 f"{title_prefix} Y Logit Distribution (Sample {i+1})"])
        
        fig = make_subplots(
            rows=batch_size, cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.05
        )
        
        for i in range(batch_size):
            # Handle different logit formats
            if len(logits.shape) == 4:  # [B, T, 2, k_classes] 
                x_logits = logits[i, :, 0, :]  # [T, num_classes]
                y_logits = logits[i, :, 1, :]  # [T, num_classes]
            else:  # [B, T, k_classes] - assume single coordinate
                x_logits = logits[i, :, :]  # [T, num_classes]
                y_logits = x_logits  # Use same for both
            
            x_probs = F.softmax(x_logits, dim=-1)  # [T, num_classes]
            x_target_classes = quantizer.quantize(target_coords[i:i+1, :, 0:1]).squeeze()  # [T]
            
            # X coordinate heatmap (log scale)
            x_probs_vis = x_probs.cpu().numpy()
            x_probs_log = np.log(x_probs_vis + 1e-8)
            
            fig.add_trace(go.Heatmap(
                z=x_probs_log.T,  # Transpose so classes are on Y-axis
                x=list(range(seq_len)),
                y=list(range(num_classes)),
                colorscale='Viridis',
                name=f'X Logits Sample {i+1}',
                showscale=(i == 0),
                colorbar=dict(title="Log Probability") if i == 0 else None,
                hovertemplate='Time: %{x}<br>Class: %{y}<br>Log Prob: %{z:.3f}<extra></extra>'
            ), row=i+1, col=1)
            
            # Overlay target trajectory
            fig.add_trace(go.Scatter(
                x=list(range(seq_len)),
                y=x_target_classes.cpu().numpy(),
                mode='lines',
                line=dict(color='red', width=3),
                name='X Target' if i == 0 else None,
                showlegend=(i == 0),
                legendgroup='x_target'
            ), row=i+1, col=1)
            
            # Y coordinate heatmap
            y_probs = F.softmax(y_logits, dim=-1)  # [T, num_classes]
            y_target_classes = quantizer.quantize(target_coords[i:i+1, :, 1:2]).squeeze()  # [T]
            
            y_probs_vis = y_probs.cpu().numpy()
            y_probs_log = np.log(y_probs_vis + 1e-8)
            
            fig.add_trace(go.Heatmap(
                z=y_probs_log.T,  # Transpose so classes are on Y-axis
                x=list(range(seq_len)),
                y=list(range(num_classes)),
                colorscale='Plasma',
                name=f'Y Logits Sample {i+1}',
                showscale=False,
                hovertemplate='Time: %{x}<br>Class: %{y}<br>Log Prob: %{z:.3f}<extra></extra>'
            ), row=i+1, col=2)
            
            # Overlay target trajectory
            fig.add_trace(go.Scatter(
                x=list(range(seq_len)),
                y=y_target_classes.cpu().numpy(),
                mode='lines',
                line=dict(color='red', width=3),
                name='Y Target' if i == 0 else None,
                showlegend=(i == 0),
                legendgroup='y_target'
            ), row=i+1, col=2)
        
        fig.update_layout(
            title="Logit Probability Heatmaps",
            height=400 * batch_size,
            width=1600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Class Index")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_wandb_logit_heatmaps(self, logits, target_coords, quantizer, max_samples=2):
        """
        Create wandb-compatible heatmap data for logit distributions using Plotly.
        
        Args:
            logits: Model logits [B, T, k_classes] or [B, T, 2, k_classes]
            target_coords: Target coordinates [B, T, 2]
            quantizer: Coordinate quantizer
            max_samples: Number of samples to create
            
        Returns:
            Dictionary of Plotly Figure objects
        """
        import torch.nn.functional as F
        
        batch_size = min(max_samples, logits.shape[0])
        seq_len = logits.shape[1]
        num_classes = logits.shape[-1]
        
        log_data = {}
        
        for i in range(batch_size):
            # Handle different logit formats
            if len(logits.shape) == 4:  # [B, T, 2, k_classes]
                x_logits = logits[i, :, 0, :]  # [T, num_classes]
                y_logits = logits[i, :, 1, :]  # [T, num_classes]
            else:  # [B, T, k_classes] - single coordinate
                x_logits = logits[i, :, :]
                y_logits = x_logits
            
            # X coordinate heatmap data
            x_probs = F.softmax(x_logits, dim=-1)  # [T, num_classes]
            x_target_classes = quantizer.quantize(target_coords[i:i+1, :, 0:1]).squeeze()  # [T]
            
            x_probs_np = x_probs.cpu().numpy()  # [250, num_classes]
            x_heatmap_data = x_probs_np.T  # [num_classes, 250]
            
            # Log scale version
            x_probs_log = np.log(x_heatmap_data + 1e-10)
            
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=x_probs_log,
                x=list(range(seq_len)),
                y=list(range(num_classes)),
                colorscale='Viridis',
                hovertemplate='Time: %{x}<br>Class: %{y}<br>Log Prob: %{z:.3f}<extra></extra>',
                colorbar=dict(title="Log Probability")
            ))
            
            # Overlay target trajectory
            fig.add_trace(go.Scatter(
                x=list(range(seq_len)),
                y=x_target_classes.cpu().numpy(),
                mode='lines',
                line=dict(color='red', width=3),
                name='Target'
            ))
            
            fig.update_layout(
                title=f"X Logit Probabilities Sample {i+1} (Log Scale)",
                xaxis_title="Timestep",
                yaxis_title="Class Index",
                width=1200,
                height=800
            )
            
            log_data[f"logits_x_log_sample_{i+1}"] = fig
            
            # Linear scale version
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=x_heatmap_data,
                x=list(range(seq_len)),
                y=list(range(num_classes)),
                colorscale='Viridis',
                hovertemplate='Time: %{x}<br>Class: %{y}<br>Prob: %{z:.6f}<extra></extra>',
                colorbar=dict(title="Probability")
            ))
            
            # Overlay target trajectory
            fig.add_trace(go.Scatter(
                x=list(range(seq_len)),
                y=x_target_classes.cpu().numpy(),
                mode='lines',
                line=dict(color='red', width=3),
                name='Target'
            ))
            
            fig.update_layout(
                title=f"X Logit Probabilities Sample {i+1} (Linear Scale)",
                xaxis_title="Timestep",
                yaxis_title="Class Index",
                width=1200,
                height=800
            )
            
            log_data[f"logits_x_prob_sample_{i+1}"] = fig
            
            # Y coordinate heatmap data  
            y_probs = F.softmax(y_logits, dim=-1)  # [T, num_classes]
            y_target_classes = quantizer.quantize(target_coords[i:i+1, :, 1:2]).squeeze()  # [T]
            
            y_probs_np = y_probs.cpu().numpy()  # [250, num_classes]
            y_heatmap_data = y_probs_np.T  # [num_classes, 250]
            
            # Log scale version
            y_probs_log = np.log(y_heatmap_data + 1e-10)
            
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=y_probs_log,
                x=list(range(seq_len)),
                y=list(range(num_classes)),
                colorscale='Plasma',
                hovertemplate='Time: %{x}<br>Class: %{y}<br>Log Prob: %{z:.3f}<extra></extra>',
                colorbar=dict(title="Log Probability")
            ))
            
            # Overlay target trajectory
            fig.add_trace(go.Scatter(
                x=list(range(seq_len)),
                y=y_target_classes.cpu().numpy(),
                mode='lines',
                line=dict(color='red', width=3),
                name='Target'
            ))
            
            fig.update_layout(
                title=f"Y Logit Probabilities Sample {i+1} (Log Scale)",
                xaxis_title="Timestep",
                yaxis_title="Class Index",
                width=1200,
                height=800
            )
            
            log_data[f"logits_y_log_sample_{i+1}"] = fig
            
            # Linear scale version
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=y_heatmap_data,
                x=list(range(seq_len)),
                y=list(range(num_classes)),
                colorscale='Plasma',
                hovertemplate='Time: %{x}<br>Class: %{y}<br>Prob: %{z:.6f}<extra></extra>',
                colorbar=dict(title="Probability")
            ))
            
            # Overlay target trajectory
            fig.add_trace(go.Scatter(
                x=list(range(seq_len)),
                y=y_target_classes.cpu().numpy(),
                mode='lines',
                line=dict(color='red', width=3),
                name='Target'
            ))
            
            fig.update_layout(
                title=f"Y Logit Probabilities Sample {i+1} (Linear Scale)",
                xaxis_title="Timestep",
                yaxis_title="Class Index",
                width=1200,
                height=800
            )
            
            log_data[f"logits_y_prob_sample_{i+1}"] = fig
        
        return log_data
    
    def visualize_classification_predictions(self, logits, target_coords, quantizer,
                                           title_prefix="", max_samples=10, save_path: str = None):
        """
        Visualize classification predictions vs targets as simple line plots.
        
        Args:
            logits: Model logits [B, T, k_classes] or [B, T, 2, k_classes]
            target_coords: Target coordinates [B, T, 2]
            quantizer: Coordinate quantizer
            title_prefix: Title prefix
            max_samples: Number of samples to show
            save_path: Optional save path
        """
        import torch.nn.functional as F
        
        batch_size = min(max_samples, logits.shape[0])
        seq_len = logits.shape[1]
        
        # Create subplot titles
        subplot_titles = []
        for i in range(batch_size):
            subplot_titles.extend([f"{title_prefix} X Classes (Sample {i+1})",
                                 f"{title_prefix} Y Classes (Sample {i+1})"])
        
        fig = make_subplots(
            rows=batch_size, cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.05
        )
        
        time_steps = list(range(seq_len))
        
        for i in range(batch_size):
            # Handle different logit formats
            if len(logits.shape) == 4:  # [B, T, 2, k_classes]
                x_logits = logits[i, :, 0, :]  # [T, num_classes]
                y_logits = logits[i, :, 1, :]  # [T, num_classes]
            else:  # [B, T, k_classes] - single coordinate
                x_logits = logits[i, :, :]
                y_logits = x_logits
            
            # X coordinate predictions
            x_probs = F.softmax(x_logits, dim=-1)  # [T, num_classes]
            x_pred_classes = torch.argmax(x_probs, dim=-1)  # [T]
            x_target_classes = quantizer.quantize(target_coords[i:i+1, :, 0:1]).squeeze()  # [T]
            
            fig.add_trace(go.Scatter(
                x=time_steps, y=x_pred_classes.cpu().numpy(),
                mode='lines', name='X Predicted', line=dict(color='blue', width=2),
                showlegend=(i == 0), legendgroup='x_pred'
            ), row=i+1, col=1)
            
            fig.add_trace(go.Scatter(
                x=time_steps, y=x_target_classes.cpu().numpy(),
                mode='lines', name='X Target', line=dict(color='red', width=2),
                showlegend=(i == 0), legendgroup='x_target'
            ), row=i+1, col=1)
            
            # Y coordinate predictions  
            y_probs = F.softmax(y_logits, dim=-1)  # [T, num_classes]
            y_pred_classes = torch.argmax(y_probs, dim=-1)  # [T]
            y_target_classes = quantizer.quantize(target_coords[i:i+1, :, 1:2]).squeeze()  # [T]
            
            fig.add_trace(go.Scatter(
                x=time_steps, y=y_pred_classes.cpu().numpy(),
                mode='lines', name='Y Predicted', line=dict(color='green', width=2),
                showlegend=(i == 0), legendgroup='y_pred'
            ), row=i+1, col=2)
            
            fig.add_trace(go.Scatter(
                x=time_steps, y=y_target_classes.cpu().numpy(),
                mode='lines', name='Y Target', line=dict(color='red', width=2),
                showlegend=(i == 0), legendgroup='y_target'
            ), row=i+1, col=2)
        
        fig.update_layout(
            title="Classification Predictions vs Targets",
            height=400 * batch_size,
            width=1400,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Class Index")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


def summarize_model_architecture(model, input_shape=(250, 2), device='cpu'):
    """
    Create detailed model architecture summary with layer information.
    
    Args:
        model: PyTorch model
        input_shape: Expected input shape (excluding batch dimension)
        device: Device for computation
        
    Returns:
        str: Formatted architecture summary
    """
    model.eval()
    summary_lines = []
    
    # Model overview
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary_lines.append("=" * 80)
    summary_lines.append(f"MODEL ARCHITECTURE SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Model Type: {model.__class__.__name__}")
    summary_lines.append(f"Input Shape: {input_shape}")
    summary_lines.append(f"Total Parameters: {total_params:,}")
    summary_lines.append(f"Trainable Parameters: {trainable_params:,}")
    summary_lines.append(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    summary_lines.append("")
    
    # Model configuration
    if hasattr(model, 'd_latent'):
        summary_lines.append(f"Latent Dimension: {model.d_latent}")
    if hasattr(model, 'k_classes'):
        summary_lines.append(f"Number of Classes: {model.k_classes}")
    
    # Encoder details
    if hasattr(model, 'encoder'):
        summary_lines.append("\nENCODER ARCHITECTURE:")
        summary_lines.append("-" * 40)
        encoder = model.encoder
        if hasattr(encoder, '__class__'):
            summary_lines.append(f"Type: {encoder.__class__.__name__}")
        
        # Count encoder parameters
        encoder_params = sum(p.numel() for p in encoder.parameters())
        summary_lines.append(f"Parameters: {encoder_params:,}")
        
        # Try to get layer details
        try:
            for name, module in encoder.named_children():
                if hasattr(module, '__len__') and len(list(module.children())) > 0:
                    # Sequential module
                    summary_lines.append(f"  {name}:")
                    for i, layer in enumerate(module):
                        layer_params = sum(p.numel() for p in layer.parameters())
                        summary_lines.append(f"    {i}: {layer} [{layer_params:,} params]")
                else:
                    # Single layer
                    layer_params = sum(p.numel() for p in module.parameters())
                    summary_lines.append(f"  {name}: {module} [{layer_params:,} params]")
        except:
            summary_lines.append(f"  {encoder}")
    
    # Decoder details
    if hasattr(model, 'decoder'):
        summary_lines.append("\nDECODER ARCHITECTURE:")
        summary_lines.append("-" * 40)
        decoder = model.decoder
        if hasattr(decoder, '__class__'):
            summary_lines.append(f"Type: {decoder.__class__.__name__}")
        
        # Count decoder parameters
        decoder_params = sum(p.numel() for p in decoder.parameters())
        summary_lines.append(f"Parameters: {decoder_params:,}")
        
        # Try to get layer details
        try:
            for name, module in decoder.named_children():
                if hasattr(module, '__len__') and len(list(module.children())) > 0:
                    # Sequential module
                    summary_lines.append(f"  {name}:")
                    for i, layer in enumerate(module):
                        layer_params = sum(p.numel() for p in layer.parameters())
                        summary_lines.append(f"    {i}: {layer} [{layer_params:,} params]")
                else:
                    # Single layer
                    layer_params = sum(p.numel() for p in module.parameters())
                    summary_lines.append(f"  {name}: {module} [{layer_params:,} params]")
        except:
            summary_lines.append(f"  {decoder}")
    
    # Forward pass analysis with dummy input
    try:
        summary_lines.append("\nFORWARD PASS ANALYSIS:")
        summary_lines.append("-" * 40)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape).to(device)
            model.to(device)
            
            if hasattr(model, 'forward'):
                outputs = model(dummy_input)
                
                if isinstance(outputs, dict):
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor):
                            summary_lines.append(f"  {key}: {tuple(value.shape)}")
                elif isinstance(outputs, torch.Tensor):
                    summary_lines.append(f"  output: {tuple(outputs.shape)}")
                elif isinstance(outputs, (tuple, list)):
                    for i, output in enumerate(outputs):
                        if isinstance(output, torch.Tensor):
                            summary_lines.append(f"  output_{i}: {tuple(output.shape)}")
                            
    except Exception as e:
        summary_lines.append(f"  Forward pass analysis failed: {str(e)}")
    
    # Memory analysis
    try:
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        summary_lines.append(f"\nMODEL SIZE: {model_size_mb:.2f} MB")
    except:
        pass
    
    summary_lines.append("=" * 80)
    
    return "\n".join(summary_lines)


def parse_validation_interval(val_interval_str, steps_per_epoch):
    """
    Parse validation interval from string format.
    
    Args:
        val_interval_str: Validation interval ('5%', '10%', or '100' for steps)
        steps_per_epoch: Number of steps per epoch
        
    Returns:
        int: Validation interval in steps
    """
    if val_interval_str.endswith('%'):
        # Percentage of epoch
        percentage = float(val_interval_str[:-1]) / 100.0
        if percentage <= 0 or percentage > 100:
            raise ValueError(f"Percentage must be between 0-100, got {val_interval_str}")
        val_steps = max(1, int(steps_per_epoch * percentage))
        print(f"Validation every {val_interval_str} of epoch = {val_steps} steps")
        return val_steps
    else:
        # Direct step count
        try:
            val_steps = int(val_interval_str)
            if val_steps <= 0:
                raise ValueError(f"Validation interval must be positive, got {val_steps}")
            print(f"Validation every {val_steps} steps")
            return val_steps
        except ValueError:
            raise ValueError(f"Invalid validation interval format: {val_interval_str}")


def generate_reconstructed_trajectories(model, data_batch: torch.Tensor, 
                                       quantizer, device: str) -> torch.Tensor:
    """
    Generate reconstructed trajectories by encoding and decoding through the VAE.
    
    Args:
        model: Trained VAE model
        data_batch: Original data batch [B, 250, 2]
        quantizer: Coordinate quantizer for converting logits to coordinates
        device: Device to run inference on
        
    Returns:
        reconstructed_coords: Reconstructed coordinate sequences [B, 250, 2]
    """
    model.eval()
    with torch.no_grad():
        data_batch = data_batch.to(device)
        
        # Encode to get latent representation (use mean, no sampling)
        mu, log_sigma = model.encode(data_batch)
        
        # Decode from latent space
        logits = model.decode(mu)  # [B, 250, k_classes]
        
        # Convert logits to coordinates
        reconstructed_coords = quantizer.decode_to_trajectory(logits)  # [B, 250, 2]
        
    return reconstructed_coords


def generate_sampled_trajectories(model, num_samples: int, quantizer, 
                                device: str) -> torch.Tensor:
    """
    Generate new trajectories by sampling from the latent space.
    
    Args:
        model: Trained VAE model
        num_samples: Number of trajectories to generate
        quantizer: Coordinate quantizer for converting logits to coordinates
        device: Device to run inference on
        
    Returns:
        sampled_coords: Generated coordinate sequences [num_samples, 250, 2]
    """
    model.eval()
    with torch.no_grad():
        # Sample from prior N(0, I)
        z = torch.randn(num_samples, model.d_latent, device=device)
        
        # Decode from latent space
        logits = model.decode(z)  # [num_samples, 250, k_classes]
        
        # Convert logits to coordinates
        sampled_coords = quantizer.decode_to_trajectory(logits)  # [num_samples, 250, 2]
        
    return sampled_coords


def create_train_val_split(dataset, val_split=0.2, seed=42):
    """
    Split a dataset into training and validation sets.
    
    Args:
        dataset: PyTorch dataset
        val_split: Fraction for validation (0.0-1.0)  
        seed: Random seed for reproducible splits
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    from torch.utils.data import Subset, random_split
    import torch
    
    if val_split <= 0.0 or val_split >= 1.0:
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}")
    
    # Set seed for reproducible splits
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    print(f"Splitting dataset: {train_size:,} train, {val_size:,} val ({val_split:.1%})")
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    return train_dataset, val_dataset


def generate_experiment_name(encoder_name, decoder_name, args, timestamp=None):
    """
    Generate unique experiment name from encoder, decoder, args, and timestamp.
    
    Args:
        encoder_name: Name of encoder (e.g., "cnn", "transformer")  
        decoder_name: Name of decoder (e.g., "cnn", "transformer")
        args: Argument namespace or dictionary with training parameters
        timestamp: Optional timestamp string (generated if None)
        
    Returns:
        str: Unique experiment name
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Convert args to dict if needed
    if hasattr(args, '__dict__'):
        args_dict = vars(args)
    else:
        args_dict = args
    
    # Extract key parameters
    d_latent = args_dict.get('d_latent', 128)
    k_classes = args_dict.get('k_classes', 3000)
    batch_size = args_dict.get('batch_size', 32)
    lr = args_dict.get('lr', 1e-3)
    beta = args_dict.get('beta', 1.0)
    lightweight = args_dict.get('lightweight', False)
    beta_warmup = args_dict.get('beta_warmup', False)
    
    # Build name components
    components = []
    
    # Encoder-Decoder architecture
    components.append(f"enc-{encoder_name}")
    components.append(f"dec-{decoder_name}")
    
    # Model variant
    if lightweight:
        components.append("lite")
    
    # Key hyperparameters
    components.append(f"ld{d_latent}")
    components.append(f"kc{k_classes}")
    components.append(f"bs{batch_size}")
    components.append(f"lr{lr}")
    
    # VAE-specific parameters
    if beta != 1.0:
        components.append(f"b{beta}")
    if beta_warmup:
        components.append("warmup")
    
    # Add timestamp
    components.append(timestamp)
    
    return "_".join(components)


# Model Checkpointing saving code: Save model code (encoder, decoder and model) and weights, and ONNX.
