"""
CNN Gesture Classifier Training Script

This script trains a CNN-based classifier for gesture prediction using discrete
class labels on normalized screen coordinates [x, y] ignoring pressure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Deque, Dict, List, Optional, Tuple
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
import sys
import os
from datetime import datetime
import h5py
from collections import deque

# Import our CNN Gesture Classifier implementation
from .cnn_gesture_classifier_no_int import (
    CNNGestureClassifier, CNNGestureClassifierTrainer, CoordinateQuantizer, 
    compute_classification_loss, decode_predictions, compute_delta_class, create_model_and_trainer
)
# ============================================================================
# DATASET
# ============================================================================

class StreamingGestureDataset(Dataset):
    """Streaming dataset for training gesture classifier models with on-demand loading.
    
    Supports multiple data directory formats:
    - gesture_data format: *_actions.npy files with shape [timesteps, 3]
    - trajectory format: *_act.npy files with shape [7, 250, 3] (action blocks)
    """
    
    def __init__(self, data_dirs, seq_len=250, test_mode=False, cache_size=1000):
        self.seq_len = seq_len
        self.test_mode = test_mode
        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []
        
        # Support both single directory and list of directories
        if isinstance(data_dirs, (str, Path)):
            self.data_dirs = [Path(data_dirs)]
        else:
            self.data_dirs = [Path(d) for d in data_dirs]
        
        if test_mode:
            # Generate synthetic test data (small and fast)
            self.total_sequences = 100
            self.file_indices = []  # Not used in test mode
            print(f"Initialized synthetic dataset with {self.total_sequences} sequences of length {seq_len}")
        else:
            # Streaming setup - only scan files, don't load data
            import time
            start_time = time.time()
            
            # Collect files from all data directories
            self.action_files = []
            self.file_formats = []  # Track format for each file
            
            for data_dir in self.data_dirs:
                if not data_dir.exists():
                    print(f"Warning: Directory {data_dir} does not exist, skipping...")
                    continue
                    
                # Try both file patterns with recursive search
                actions_files = list(data_dir.glob("**/*_actions.npy"))
                act_files = list(data_dir.glob("**/*_act.npy"))
                
                # Add files with format tracking
                for f in actions_files:
                    self.action_files.append(f)
                    self.file_formats.append('gesture_data')  # Flat format
                    
                for f in act_files:
                    self.action_files.append(f) 
                    self.file_formats.append('trajectory')  # Action blocks format
                    
                print(f"Found {len(actions_files)} gesture_data files and {len(act_files)} trajectory files in {data_dir}")
            
            if not self.action_files:
                raise ValueError(f"No '_actions.npy' or '_act.npy' files found in any of {self.data_dirs} (searched recursively)")
            
            # Build index of (file_idx, sequence_idx) without loading data
            self.file_indices = []
            self.total_sequences = 0
            
            print(f"Scanning {len(self.action_files)} files for sequence indexing...")
            for file_idx, file_path in enumerate(self.action_files):
                file_format = self.file_formats[file_idx]
                
                # Only load file header to get shape - much faster than full load
                with open(file_path, 'rb') as f:
                    # Read numpy header to get shape without loading data
                    try:
                        header = np.lib.format.read_magic(f)
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                        
                        num_sequences = self._calculate_num_sequences(shape, file_format)
                        
                        if num_sequences > 0:
                            for seq_idx in range(num_sequences):
                                self.file_indices.append((file_idx, seq_idx, file_format))
                            self.total_sequences += num_sequences
                    except:
                        # Fallback: quick load just to get shape
                        data = np.load(file_path)
                        
                        num_sequences = self._calculate_num_sequences(data.shape, file_format)
                        
                        if num_sequences > 0:
                            for seq_idx in range(num_sequences):
                                self.file_indices.append((file_idx, seq_idx, file_format))
                            self.total_sequences += num_sequences
                        del data  # Free memory immediately
            
            setup_time = time.time() - start_time
            print(f"Dataset setup complete in {setup_time:.2f}s: {self.total_sequences} sequences from {len(self.action_files)} files")
            print(f"Streaming mode enabled with cache size: {cache_size}")
            
            # Print format summary
            gesture_data_count = sum(1 for fmt in self.file_formats if fmt == 'gesture_data')
            trajectory_count = sum(1 for fmt in self.file_formats if fmt == 'trajectory')
            print(f"Data formats: {gesture_data_count} gesture_data files, {trajectory_count} trajectory files")
    
    def _calculate_num_sequences(self, shape, file_format):
        """Calculate number of sequences based on data shape and format."""
        if file_format == 'gesture_data':
            # gesture_data format: either [total_timesteps, 3] or [num_sequences, seq_len, 3]
            if len(shape) == 3 and shape[1] == self.seq_len:
                # Pre-segmented format: [num_sequences, seq_len, 3]
                return shape[0]
            elif len(shape) == 2:
                # Flat format: [total_timesteps, 3] - divide by seq_len
                return shape[0] // self.seq_len
            else:
                return 0
        elif file_format == 'trajectory':
            # Trajectory format: [7, 250, 3] action blocks
            if len(shape) == 3 and shape[0] == 7 and shape[1] == 250:
                # Each trajectory file contains 7 action blocks, we flatten them into sequences
                return 7  # Each action block becomes a sequence 
            else:
                return 0
        else:
            return 0
    
    def _generate_test_sequence(self, idx):
        """Generate a single synthetic test sequence on-demand."""
        np.random.seed(idx)  # Deterministic generation
        sequence = np.zeros((self.seq_len, 3), dtype=np.float32)
        
        # Generate random gesture patterns
        num_gestures = np.random.randint(1, 5)
        
        for _ in range(num_gestures):
            start_t = np.random.randint(0, self.seq_len - 10)
            length = np.random.randint(5, min(20, self.seq_len - start_t))
            
            start_x, start_y = np.random.random(2)
            end_x, end_y = np.random.random(2)
            
            for i, t in enumerate(range(start_t, start_t + length)):
                alpha = i / (length - 1) if length > 1 else 0
                sequence[t, 0] = start_x + alpha * (end_x - start_x)
                sequence[t, 1] = start_y + alpha * (end_y - start_y)
                sequence[t, 2] = 1.0
        
        # Add noise
        sequence[:, :2] += np.random.normal(0, 0.01, (self.seq_len, 2))
        sequence[:, :2] = np.clip(sequence[:, :2], 0.0, 1.0)
        
        return sequence
    
    def _load_sequence(self, file_idx, seq_idx, file_format):
        """Load a specific sequence from file on-demand."""
        cache_key = (file_idx, seq_idx, file_format)
        
        # Check cache first
        if cache_key in self.cache:
            # Move to end of cache order (LRU)
            self.cache_order.remove(cache_key)
            self.cache_order.append(cache_key)
            return self.cache[cache_key]
        
        # Load from file
        file_path = self.action_files[file_idx]
        
        # Use memory mapping for efficient partial loading
        data_mmap = np.load(file_path, mmap_mode='r')
        
        # Handle different data formats
        if file_format == 'gesture_data':
            if len(data_mmap.shape) == 3 and data_mmap.shape[1] == self.seq_len:
                # Pre-segmented format: [num_sequences, seq_len, 3] - directly index sequence
                sequence = np.array(data_mmap[seq_idx], dtype=np.float32)
            elif len(data_mmap.shape) == 2:
                # Flat format: [total_timesteps, 3] - extract slice
                start_idx = seq_idx * self.seq_len
                end_idx = start_idx + self.seq_len
                sequence = np.array(data_mmap[start_idx:end_idx], dtype=np.float32)
            else:
                raise ValueError(f"Unsupported gesture_data format: {data_mmap.shape}")
        elif file_format == 'trajectory':
            # Trajectory format: [7, 250, 3] - extract specific action block
            if len(data_mmap.shape) == 3 and data_mmap.shape[0] == 7 and data_mmap.shape[1] == 250:
                sequence = np.array(data_mmap[seq_idx], dtype=np.float32)  # seq_idx corresponds to action block index
            else:
                raise ValueError(f"Unsupported trajectory format: {data_mmap.shape}")
        else:
            raise ValueError(f"Unknown file format: {file_format}")
        
        # Normalize coordinates
        sequence[:, :2] = np.clip(sequence[:, :2], 0.0, 1.0)
        sequence[:, 2] = np.clip(sequence[:, 2], 0.0, 1.0)
        
        # Add to cache with LRU eviction
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            oldest_key = self.cache_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[cache_key] = sequence
        self.cache_order.append(cache_key)
        
        return sequence
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        if self.test_mode:
            sequence = self._generate_test_sequence(idx)
        else:
            file_idx, seq_idx, file_format = self.file_indices[idx]
            sequence = self._load_sequence(file_idx, seq_idx, file_format)
        
        return torch.from_numpy(sequence)


class GestureDataset(StreamingGestureDataset):
    """Legacy wrapper for backward compatibility."""

    def __init__(self, data_dirs, seq_len=250, test_mode=False):
        # Use streaming dataset with moderate cache size
        cache_size = 500 if not test_mode else 0
        super().__init__(data_dirs, seq_len, test_mode, cache_size)

        if not test_mode:
            print(f"Using streaming dataset (legacy mode) - cache size: {cache_size}")


class OverfitDataset(Dataset):
    """Dataset that always returns the same single sample for overfitting tests."""

    def __init__(self, sample_data, num_iterations):
        """
        Args:
            sample_data: Single sample tensor [T, 3] or [T, 2]
            num_iterations: Number of iterations to simulate
        """
        self.sample_data = sample_data
        self.num_iterations = num_iterations

    def __len__(self):
        return self.num_iterations

    def __getitem__(self, idx):
        return self.sample_data.clone()


class H5ActionSequenceDataset(Dataset):
    """Expose individual 250-step gesture sequences from HDF5 files."""

    def __init__(
        self,
        processed_data_dir,
        manifest_path=None,
        split='train',
        max_open_files: int = 32,
    ):
        self.data_dir = Path(processed_data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Processed data directory {self.data_dir} does not exist")

        if max_open_files <= 0:
            raise ValueError("max_open_files must be a positive integer")

        if manifest_path and split:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            if 'splits' not in manifest:
                raise ValueError(f"Manifest at {manifest_path} missing 'splits' key")
            if split not in manifest['splits']:
                raise ValueError(
                    f"Split '{split}' not found in manifest; available splits: {list(manifest['splits'].keys())}"
                )
            relative_files = manifest['splits'][split]
        else:
            # Fallback: load all *.h5 files in the directory
            relative_files = [str(path.name) for path in sorted(self.data_dir.glob('*.h5'))]

        self.files: List[Path] = []
        self.file_metadata: List[Dict] = []
        self.segment_index: List[Tuple[int, Tuple[int, Optional[int]]]] = []
        self.sequence_length: Optional[int] = None
        self.num_channels: Optional[int] = None
        self._file_handles: Dict[int, Dict[Path, h5py.File]] = {}
        self._handle_order: Dict[int, Deque[Path]] = {}
        self.max_open_files = max_open_files

        for rel_path in relative_files:
            file_path = self.data_dir / rel_path
            if not file_path.exists():
                raise FileNotFoundError(f"Referenced file {rel_path} not found in {self.data_dir}")

            with h5py.File(file_path, 'r') as f:
                if 'actions' not in f:
                    raise KeyError(f"File {file_path} missing 'actions' dataset")
                actions_ds = f['actions']
                if actions_ds.ndim == 4:
                    n_primary, n_segments, seq_len, channels = actions_ds.shape
                    entry_kind = 'grid'
                elif actions_ds.ndim == 3:
                    n_primary, seq_len, channels = actions_ds.shape
                    n_segments = 1
                    entry_kind = 'flat'
                else:
                    raise ValueError(
                        f"Unsupported 'actions' dataset shape {actions_ds.shape} in {file_path};"
                        " expected 3D or 4D"
                    )

            if self.sequence_length is None:
                self.sequence_length = seq_len
                self.num_channels = channels
            else:
                if seq_len != self.sequence_length or channels != self.num_channels:
                    raise ValueError(
                        f"Inconsistent action shape detected. Expected ({self.sequence_length}, {self.num_channels})"
                        f" but found ({seq_len}, {channels}) in {file_path}"
                    )

            file_id = len(self.files)
            self.files.append(file_path)
            self.file_metadata.append(
                {
                    'kind': entry_kind,
                    'shape': (n_primary, n_segments, seq_len, channels),
                }
            )

            if entry_kind == 'grid':
                for primary_idx in range(n_primary):
                    for segment_idx in range(n_segments):
                        self.segment_index.append((file_id, (primary_idx, segment_idx)))
            else:  # flat layout already stores segments individually along axis 0
                for flat_idx in range(n_primary):
                    self.segment_index.append((file_id, (flat_idx, None)))

        if not self.segment_index:
            raise ValueError("No action segments found in the provided HDF5 data")

    def __len__(self):
        return len(self.segment_index)

    def _get_file_handle(self, file_path: Path) -> h5py.File:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0

        handles = self._file_handles.setdefault(worker_id, {})
        order = self._handle_order.setdefault(worker_id, deque())

        if file_path in handles:
            try:
                order.remove(file_path)
            except ValueError:
                pass  # Handle missing bookkeeping gracefully
            order.append(file_path)
            return handles[file_path]

        if len(handles) >= self.max_open_files and order:
            stale_path = order.popleft()
            stale_handle = handles.pop(stale_path, None)
            if stale_handle is not None:
                stale_handle.close()

        handles[file_path] = h5py.File(file_path, 'r')
        order.append(file_path)
        return handles[file_path]

    def __getitem__(self, idx):
        file_id, entry = self.segment_index[idx]
        file_path = self.files[file_id]
        metadata = self.file_metadata[file_id]
        h5_file = self._get_file_handle(file_path)

        if metadata['kind'] == 'grid':
            primary_idx, segment_idx = entry
            action_np = h5_file['actions'][primary_idx, segment_idx]
        else:
            flat_idx, _ = entry
            action_np = h5_file['actions'][flat_idx]

        action_tensor = torch.from_numpy(np.array(action_np, copy=True)).float()
        return action_tensor

    def _close_all_handles(self) -> None:
        for worker_handles in self._file_handles.values():
            for handle in worker_handles.values():
                try:
                    handle.close()
                except Exception:
                    pass
        self._file_handles.clear()
        self._handle_order.clear()

    def __del__(self):
        try:
            self._close_all_handles()
        except Exception:
            pass

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_classifier_reconstruction_metrics(original, reconstructed, latent, quantizer):
    """Compute detailed classifier reconstruction metrics."""
    
    # Basic reconstruction metrics (only for x, y coordinates)
    coords_orig = original[:, :, :2]  # [B, T, 2]
    coords_recon = reconstructed[:, :, :2] if reconstructed.shape[2] >= 2 else reconstructed  # [B, T, 2]
    
    mse = torch.mean((coords_orig - coords_recon) ** 2)
    mae = torch.mean(torch.abs(coords_orig - coords_recon))
    
    # Coordinate classification accuracy
    # Quantize both for accuracy computation
    orig_quantized = quantizer.quantize(coords_orig)  # [B, T, 2]
    recon_quantized = quantizer.quantize(coords_recon)  # [B, T, 2]
    
    x_accuracy = (orig_quantized[:, :, 0] == recon_quantized[:, :, 0]).float().mean()
    y_accuracy = (orig_quantized[:, :, 1] == recon_quantized[:, :, 1]).float().mean()
    
    # Action-specific metrics (only during active periods if pressure is available)
    if original.shape[2] > 2:
        press_mask = original[:, :, 2] > 0.5
        if press_mask.any():
            active_coord_mse = torch.mean(((coords_orig - coords_recon) ** 2)[press_mask])
        else:
            active_coord_mse = torch.tensor(0.0, device=original.device)
    else:
        active_coord_mse = torch.tensor(0.0, device=original.device)
    
    # Latent space metrics
    latent_mean_norm = torch.mean(torch.norm(latent, dim=1))
    latent_std = torch.std(latent, dim=0).mean()
    
    return {
        'mse': mse.item(),
        'mae': mae.item(), 
        'coord_mse': mse.item(),  # Same as mse for coordinate-only data
        'active_coord_mse': active_coord_mse.item(),
        'x_accuracy': x_accuracy.item(),
        'y_accuracy': y_accuracy.item(),
        'latent_mean_norm': latent_mean_norm.item(),
        'latent_std': latent_std.item(),
    }

def compute_generation_metrics(generated_sequences):
    """Compute metrics for generated sequences."""
    
    # Diversity metrics
    coords = generated_sequences[:, :, :2]  # [B, T, 2]
    
    # Coordinate diversity
    coord_std = torch.std(coords.reshape(-1, 2), dim=0).mean()
    
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
        'smoothness': smoothness.item(),
        'sequence_diversity': sequence_diversity.item(),
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_action_sequences(original, reconstructed, title_prefix="", max_samples=10):
    """Create comprehensive visualization plots for WandB logging."""
    
    batch_size = min(max_samples, original.shape[0])
    fig, axes = plt.subplots(batch_size, 2, figsize=(12, 4 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        for j, (data, label) in enumerate([(original[i], 'Original'), 
                                          (reconstructed[i], 'Reconstructed')]):
            ax = axes[i, j]
            
            # Plot x, y coordinates
            timesteps = np.arange(len(data))
            ax.plot(timesteps, data[:, 0], 'b-', alpha=0.7, label='X')
            ax.plot(timesteps, data[:, 1], 'g-', alpha=0.7, label='Y')
            
            # Plot pressure if available
            if data.shape[1] > 2:
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

def visualize_generated_sequences(generated, title_prefix="", max_samples=10):
    """Visualize generated sequences."""
    
    batch_size = min(max_samples, generated.shape[0])
    # Create a grid layout for up to 10 samples
    nrows = min(3, (batch_size + 2) // 3)  # 3 rows max
    ncols = min(4, batch_size if batch_size <= 4 else 4)  # 4 cols max
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    
    # Handle single subplot case
    if batch_size == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()
    
    for i in range(batch_size):
        ax = axes[i]
        data = generated[i]
        
        # Plot coordinates
        timesteps = np.arange(len(data))
        ax.plot(timesteps, data[:, 0], 'b-', alpha=0.7, label='X')
        ax.plot(timesteps, data[:, 1], 'g-', alpha=0.7, label='Y')
        
        ax.set_title(f"{title_prefix} Generated Sample {i+1}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
    
    # Hide unused subplots
    for i in range(batch_size, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def visualize_latent_space(latent, title_prefix="", max_points=500):
    """Visualize latent space distribution."""
    
    # Sample random subset for visualization
    if latent.shape[0] > max_points:
        indices = torch.randperm(latent.shape[0])[:max_points]
        latent_sample = latent[indices]
    else:
        latent_sample = latent
    
    latent_dim = latent_sample.shape[1]
    
    if latent_dim >= 3:
        # 3D scatter plot for first 3 dimensions
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            latent_sample[:, 0], latent_sample[:, 1], latent_sample[:, 2],
            c=np.arange(len(latent_sample)), cmap='viridis', alpha=0.6, s=20
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
            latent_sample[:, 0], latent_sample[:, 1],
            c=np.arange(len(latent_sample)), cmap='viridis', alpha=0.6, s=20
        )
        
        ax.set_xlabel('Latent Dim 0')
        ax.set_ylabel('Latent Dim 1')
        ax.set_title(f'{title_prefix} Latent Space (2D)')
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='Sample Index')
        
    else:
        # 1D histogram
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        ax.hist(latent_sample[:, 0].cpu().numpy(), bins=50, alpha=0.7, density=True)
        ax.set_xlabel('Latent Dim 0')
        ax.set_ylabel('Density')
        ax.set_title(f'{title_prefix} Latent Space (1D)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def visualize_classification_heatmap(logits, target_coords, quantizer, title_prefix="", max_samples=10):
    """Visualize classification predictions vs targets."""
    
    batch_size = min(max_samples, logits.shape[0])
    seq_len = logits.shape[1]
    
    fig, axes = plt.subplots(batch_size, 2, figsize=(12, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # X coordinate predictions
        x_logits = logits[i, :, 0, :]  # [T, num_classes]
        x_probs = F.softmax(x_logits, dim=-1)  # [T, num_classes]
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
        y_logits = logits[i, :, 1, :]  # [T, num_classes]
        y_probs = F.softmax(y_logits, dim=-1)  # [T, num_classes]
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
    
    plt.tight_layout()
    return fig

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_classifier_model(
    model,
    trainer,
    quantizer,
    train_loader, 
    val_loader,
    num_epochs=100,
    lr=1e-4,
    device='cuda',
    checkpoint_dir='checkpoints',
    val_interval=200,
    vis_interval=2000,
    start_epoch=0,
    resume_checkpoint=None,
    w_smooth=0.0,
):
    """Training loop with comprehensive logging. Logs training metrics every step."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min = 0.01 * lr)
    
    step = 0
    best_val_loss = float('inf')
    
    # Restore optimizer and scheduler state if resuming
    if resume_checkpoint is not None:
        if 'optimizer_state_dict' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
            print("Restored optimizer state")
        if 'scheduler_state_dict' in resume_checkpoint:
            scheduler_lr.load_state_dict(resume_checkpoint['scheduler_state_dict'])
            print("Restored scheduler state")
        step = resume_checkpoint.get('step', 0)
        best_val_loss = resume_checkpoint.get('best_loss', float('inf'))
        print(f"Restored training state: step={step}, best_val_loss={best_val_loss:.4f}")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {
            'x_loss': 0.0, 'y_loss': 0.0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            
            # Extract x, y coordinates (ignore pressure p)
            input_coords = batch[:, :, :2]  # [B, T, 2]
            
            # Forward pass
            latent, logits = model(input_coords)
            
            # Compute loss with delta smoothness term
            loss, loss_dict = compute_classification_loss(
                logits, input_coords, quantizer,
                w_smooth=w_smooth,
                current_step=step,
                steps_per_epoch=len(train_loader)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            for key in epoch_metrics:
                if key in loss_dict:
                    epoch_metrics[key] += loss_dict[key]
            
            step += 1
            
            # Log training metrics every step
            log_dict = {
                "train/loss": loss.item(),
                "train/epoch": epoch + batch_idx / len(train_loader),
                "train/lr": optimizer.param_groups[0]['lr'],
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
                        
                        # Extract x, y coordinates
                        input_coords = val_batch[:, :, :2]
                        
                        # Forward pass
                        latent_val, logits_val = model(input_coords)
                        
                        # Compute loss
                        loss_val, loss_dict_val = compute_classification_loss(
                            logits_val, input_coords, quantizer,
                            w_smooth=w_smooth,
                            current_step=step,
                            steps_per_epoch=len(train_loader)
                        )
                        val_loss += loss_val.item()
                        
                        # Decode predictions for evaluation
                        reconstructed = decode_predictions(logits_val, quantizer)
                        
                        # Compute comprehensive metrics
                        metrics = compute_classifier_reconstruction_metrics(
                            val_batch, reconstructed, latent_val, quantizer
                        )
                        val_metrics.append(metrics)
                        
                        # Create visualizations at specified interval
                        if step % vis_interval == 0 and val_batch_idx == 0:
                            # Action sequence reconstruction visualization
                            fig1 = visualize_action_sequences(
                                val_batch.cpu().numpy(), 
                                reconstructed.cpu().numpy(),
                                title_prefix=f"Step {step}",
                                max_samples=10
                            )
                            wandb.log({"val/reconstructions": wandb.Image(fig1), "step": step})
                            plt.close(fig1)
                            
                            # Classification predictions visualization
                            fig2 = visualize_classification_heatmap(
                                logits_val, input_coords, quantizer, 
                                title_prefix=f"Step {step}",
                                max_samples=10
                            )
                            wandb.log({"val/predictions": wandb.Image(fig2), "step": step})
                            plt.close(fig2)
                            
                            # Latent space visualization
                            fig3 = visualize_latent_space(
                                latent_val.cpu(), title_prefix=f"Step {step}",
                                max_points=200
                            )
                            wandb.log({"val/latent_space": wandb.Image(fig3), "step": step})
                            plt.close(fig3)
                            
                            # Generate samples from random latent vectors
                            random_latent = torch.randn(10, model.latent_dim, device=device)
                            generated = trainer.generate_from_latent(random_latent)
                            gen_metrics = compute_generation_metrics(generated)
                            
                            # Generated sequences visualization
                            fig4 = visualize_generated_sequences(
                                generated.numpy(), title_prefix=f"Step {step}",
                                max_samples=10
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
                          f"YAcc={avg_metrics['val/y_accuracy']:.3f}")
                
                # Save best model
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
                        'sequence_length': model.sequence_length,
                        'num_classes': model.num_classes,
                        'encoder_channels': model.encoder_channels,
                        'decoder_channels': model.decoder_channels,
                        'use_tfm_decoder': model.use_tfm_decoder,
                        'tfm_d_model': model.tfm_d_model,
                        'tfm_num_layers': model.tfm_num_layers,
                        'tfm_num_heads': model.tfm_num_heads,
                        'tfm_mlp_ratio': model.tfm_mlp_ratio,
                        'tfm_dropout': model.tfm_dropout,
                        'tfm_attn_dropout': model.tfm_attn_dropout,
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
                'sequence_length': model.sequence_length,
                'num_classes': model.num_classes,
                'encoder_channels': model.encoder_channels,
                'decoder_channels': model.decoder_channels,
                'use_tfm_decoder': model.use_tfm_decoder,
                'tfm_d_model': model.tfm_d_model,
                'tfm_num_layers': model.tfm_num_layers,
                'tfm_num_heads': model.tfm_num_heads,
                'tfm_mlp_ratio': model.tfm_mlp_ratio,
                'tfm_dropout': model.tfm_dropout,
                'tfm_attn_dropout': model.tfm_attn_dropout,
            }
        }
        torch.save(checkpoint, Path(checkpoint_dir) / f'epoch_{epoch:03d}.pth')

# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint without needing architecture args."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} missing config information")
    
    config = checkpoint['config']
    
    # Create model with saved configuration
    from .cnn_gesture_classifier_no_int import CNNGestureClassifier, CoordinateQuantizer, CNNGestureClassifierTrainer
    
    model = CNNGestureClassifier(
        input_dim=2,  # x, y coordinates
        sequence_length=config['sequence_length'],
        latent_dim=config['latent_dim'],
        num_classes=config['num_classes'],
        encoder_channels=config['encoder_channels'],
        decoder_channels=config['decoder_channels'],
        kernel_size=config.get('kernel_size', 5),  # Default kernel size
        use_tfm_decoder=config.get('use_tfm_decoder', False),
        tfm_d_model=config.get('tfm_d_model', 512),
        tfm_num_layers=config.get('tfm_num_layers', 6),
        tfm_num_heads=config.get('tfm_num_heads', 8),
        tfm_mlp_ratio=config.get('tfm_mlp_ratio', 4.0),
        tfm_dropout=config.get('tfm_dropout', 0.1),
        tfm_attn_dropout=config.get('tfm_attn_dropout', 0.1),
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create quantizer and trainer
    quantizer = CoordinateQuantizer(num_classes=config['num_classes'])
    trainer = CNNGestureClassifierTrainer(model, quantizer, device=device)
    
    print(f"Loaded model from {checkpoint_path}:")
    print(f"  - Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  - Step: {checkpoint.get('step', 'unknown')}")
    print(f"  - Loss: {checkpoint.get('loss', 'unknown'):.4f}")
    print(f"  - Architecture: latent_dim={config['latent_dim']}, num_classes={config['num_classes']}")
    
    return model, trainer, quantizer, checkpoint

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train CNN Gesture Classifier')
    parser.add_argument('--data_dir', type=str, action='append', help='Directory containing training data (can be specified multiple times)')
    parser.add_argument('--val_data_dir', type=str, action='append', help='Directory containing validation data (can be specified multiple times)')
    parser.add_argument('--processed_data_dir', type=str, help='Directory containing preprocessed gesture trajectories in HDF5 format')
    parser.add_argument('--processed_val_dir', type=str, help='Optional directory for validation trajectories (defaults to processed_data_dir)')
    parser.add_argument('--manifest', type=str, help='Path to manifest JSON describing train/val/test splits for HDF5 datasets')
    parser.add_argument('--train_split', type=str, default='train', help="Split name to use from manifest for training (default: 'train')")
    parser.add_argument('--val_split', type=str, default='validation', help="Split name to use from manifest for validation (default: 'validation')")
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training from')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--sequence_length', type=int, default=250)
    parser.add_argument('--num_classes', type=int, default=3000)
    parser.add_argument('--encoder_channels', type=str, default='64,128,256,512', 
                       help='Comma-separated list of encoder channel sizes')
    parser.add_argument('--decoder_channels', type=str, default='512,256,128,64',
                       help='Comma-separated list of decoder channel sizes') 
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--wandb_project', type=str, default='cnn-gesture-classifier-training')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--val_interval', type=int, default=200, 
                       help='Validation interval in steps')
    parser.add_argument('--vis_interval', type=int, default=2000, 
                       help='Visualization interval in steps')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of worker processes for data loading (0=single-threaded, 4-8 recommended for fast training)')
    parser.add_argument('--cache_size', type=int, default=2000,
                       help='Cache size for streaming dataset (number of sequences to keep in memory)')
    parser.add_argument('--streaming', action='store_true', default=True,
                       help='Use streaming dataset (default: True, set --no-streaming to disable)')
    parser.add_argument('--no-streaming', dest='streaming', action='store_false',
                       help='Disable streaming dataset (load all data into memory at startup)')
    parser.add_argument('--test', action='store_true', 
                       help='Run in test mode with synthetic data (no real data files needed)')
    parser.add_argument('--w_smooth', type=float, default=0.0,
                       help='Weight for delta smoothness loss (default: 0.0, disabled)')
    parser.add_argument('--use_tfm_decoder', action='store_true',
                       help='Use Transformer decoder instead of CNN decoder')
    parser.add_argument('--tfm_d_model', type=int, default=512,
                       help='Transformer model dimension (default: 512)')
    parser.add_argument('--tfm_num_layers', type=int, default=6,
                       help='Number of transformer layers (default: 6)')
    parser.add_argument('--tfm_num_heads', type=int, default=8,
                       help='Number of attention heads (default: 8)')
    parser.add_argument('--tfm_mlp_ratio', type=float, default=4.0,
                       help='Transformer MLP ratio (default: 4.0)')
    parser.add_argument('--tfm_dropout', type=float, default=0.1,
                       help='Transformer dropout rate (default: 0.1)')
    parser.add_argument('--tfm_attn_dropout', type=float, default=0.1,
                       help='Transformer attention dropout rate (default: 0.1)')
    parser.add_argument('--overfit', action='store_true',
                       help='Run overfit test on single sample for (epochs * 50) iterations')
    args = parser.parse_args()
    
    # Convert single directories to lists for backward compatibility
    if args.data_dir is None:
        args.data_dir = []
    if args.val_data_dir is None:
        args.val_data_dir = []

    using_h5 = args.processed_data_dir is not None
    if args.processed_data_dir and args.processed_val_dir is None:
        args.processed_val_dir = args.processed_data_dir

    if using_h5 and (args.data_dir or args.val_data_dir):
        parser.error("Specify either HDF5 inputs (--processed_data_dir/--manifest) or raw directories (--data_dir/--val_data_dir), not both")
    
    # Handle checkpoint resumption
    resume_checkpoint = None
    if args.resume:
        if not Path(args.resume).exists():
            parser.error(f"Checkpoint file {args.resume} does not exist")
        print(f"Resuming training from checkpoint: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location='cpu')
        
        # Override architecture args with checkpoint config if available
        if 'config' in resume_checkpoint:
            config = resume_checkpoint['config']
            args.latent_dim = config['latent_dim']
            args.sequence_length = config['sequence_length']
            args.num_classes = config['num_classes']
            encoder_channels = config['encoder_channels']
            decoder_channels = config['decoder_channels']
            args.kernel_size = config.get('kernel_size', args.kernel_size)
            print("Using architecture from checkpoint config")
        else:
            # Parse channel lists from args
            encoder_channels = [int(x) for x in args.encoder_channels.split(',')]
            decoder_channels = [int(x) for x in args.decoder_channels.split(',')]
    else:
        # Parse channel lists from args
        encoder_channels = [int(x) for x in args.encoder_channels.split(',')]
        decoder_channels = [int(x) for x in args.decoder_channels.split(',')]
    
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
        if args.latent_dim > 32:
            print(f"Reducing latent dim from {args.latent_dim} to 32 for test mode")
            args.latent_dim = 32
        if len(encoder_channels) > 2:
            print(f"Reducing encoder channels from {encoder_channels} to [32, 64] for test mode")
            encoder_channels = [32, 64]
        if len(decoder_channels) > 2:
            print(f"Reducing decoder channels from {decoder_channels} to [64, 32] for test mode")
            decoder_channels = [64, 32]
        args.wandb_project = 'cnn-gesture-classifier-test'
        # Shorter intervals for test mode
        if args.val_interval > 50:
            print(f"Reducing val_interval from {args.val_interval} to 50 for test mode")
            args.val_interval = 50
        if args.vis_interval > 100:
            print(f"Reducing vis_interval from {args.vis_interval} to 100 for test mode")
            args.vis_interval = 100

    # Overfit mode validation
    if args.overfit:
        print("Running in OVERFIT MODE with single sample")
        # Override settings for overfitting test
        total_iterations = args.num_epochs * 50
        print(f"Total iterations: {total_iterations} (epochs {args.num_epochs} * 50)")
        args.batch_size = 1  # Always use batch size 1 for overfit
        args.wandb_project = f"{args.wandb_project}-overfit"
        # More frequent monitoring for overfit
        if args.val_interval > 10:
            print(f"Reducing val_interval from {args.val_interval} to 10 for overfit mode")
            args.val_interval = 10
        if args.vis_interval > 20:
            print(f"Reducing vis_interval from {args.vis_interval} to 20 for overfit mode")
            args.vis_interval = 20
    else:
        if using_h5:
            if not args.processed_data_dir:
                parser.error("--processed_data_dir is required when using HDF5 inputs")
            if not Path(args.processed_data_dir).exists():
                parser.error(f"Processed data directory {args.processed_data_dir} does not exist")
            if args.processed_val_dir and not Path(args.processed_val_dir).exists():
                parser.error(f"Validation processed directory {args.processed_val_dir} does not exist")
            if args.manifest and not Path(args.manifest).exists():
                parser.error(f"Manifest file {args.manifest} does not exist")
        else:
            # Validate required arguments for real training
            if not args.data_dir:
                # Try default directories if none specified
                default_dirs = [
                    'resource/gesture_data/train',
                    'resource/dataset/final_data/train_subset'
                ]
                existing_dirs = [d for d in default_dirs if Path(d).exists()]
                if existing_dirs:
                    args.data_dir = existing_dirs
                    print(f"No --data_dir specified, using default directories: {args.data_dir}")
                else:
                    parser.error("--data_dir is required when not in test mode and no HDF5 inputs are provided")
                
            if not args.val_data_dir:
                # Try default validation directories
                default_val_dirs = [
                    'resource/gesture_data/val',
                    'resource/dataset/final_data/val_subset'
                ]
                existing_val_dirs = [d for d in default_val_dirs if Path(d).exists()]
                if existing_val_dirs:
                    args.val_data_dir = existing_val_dirs
                    print(f"No --val_data_dir specified, using default directories: {args.val_data_dir}")
                else:
                    parser.error("--val_data_dir is required when not in test mode and no HDF5 inputs are provided")
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    decoder_type = "TFM" if args.use_tfm_decoder else "CNN"
    run_name = f"{decoder_type.lower()}_ld{args.latent_dim}_ec{'-'.join(map(str, encoder_channels))}_dc{'-'.join(map(str, decoder_channels))}"
    if args.use_tfm_decoder:
        run_name += f"_tfm{args.tfm_d_model}_{args.tfm_num_layers}L_{args.tfm_num_heads}H"
    if args.test:
        run_name = f"TEST_{run_name}"
    elif args.overfit:
        run_name = f"OVERFIT_{run_name}"
    
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
    
    # Choose dataset type based on provided inputs
    if args.overfit:
        print("Creating OVERFIT dataset from single sample")
        # Create a single sample for overfitting
        np.random.seed(42)  # Deterministic sample
        sample_data = np.zeros((args.sequence_length, 3), dtype=np.float32)

        # Generate a simple deterministic gesture pattern
        num_gestures = 2
        for g in range(num_gestures):
            start_t = g * (args.sequence_length // num_gestures)
            length = args.sequence_length // num_gestures - 10

            start_x, start_y = 0.2 + g * 0.3, 0.3 + g * 0.2  # Different start points
            end_x, end_y = 0.7 - g * 0.2, 0.8 - g * 0.3    # Different end points

            for i, t in enumerate(range(start_t, start_t + length)):
                if t < args.sequence_length:
                    alpha = i / (length - 1) if length > 1 else 0
                    sample_data[t, 0] = start_x + alpha * (end_x - start_x)
                    sample_data[t, 1] = start_y + alpha * (end_y - start_y)
                    sample_data[t, 2] = 1.0  # Pressure on

        sample_tensor = torch.from_numpy(sample_data)
        total_iterations = args.num_epochs * 50

        train_dataset = OverfitDataset(sample_tensor, total_iterations)
        val_dataset = OverfitDataset(sample_tensor, 10)  # Small validation set

        print(f"Overfit sample shape: {sample_tensor.shape}")
        print(f"Training iterations: {total_iterations}, Validation samples: 10")

    elif using_h5 and not args.test:
        print("Using HDF5 action sequence dataset")
        train_dataset = H5ActionSequenceDataset(
            processed_data_dir=args.processed_data_dir,
            manifest_path=args.manifest,
            split=args.train_split
        )
        val_dataset = H5ActionSequenceDataset(
            processed_data_dir=args.processed_val_dir,
            manifest_path=args.manifest,
            split=args.val_split
        )
        if train_dataset.sequence_length and args.sequence_length != train_dataset.sequence_length:
            print(
                f"Adjusting sequence_length from {args.sequence_length} to "
                f"{train_dataset.sequence_length} based on HDF5 data"
            )
            args.sequence_length = train_dataset.sequence_length
    elif args.streaming and not args.test:
        print("Using STREAMING dataset for fast startup")
        train_cache_size = args.cache_size
        val_cache_size = max(500, args.cache_size // 4)  # Smaller cache for validation

        print(f"Creating streaming datasets - Train cache: {train_cache_size}, Val cache: {val_cache_size}")

        train_dataset = StreamingGestureDataset(
            args.data_dir,
            seq_len=args.sequence_length,
            test_mode=False,
            cache_size=train_cache_size
        )
        val_dataset = StreamingGestureDataset(
            args.val_data_dir,
            seq_len=args.sequence_length,
            test_mode=False,
            cache_size=val_cache_size
        )
    else:
        print("Using LEGACY dataset (loads all data into memory at startup)")
        train_dataset = GestureDataset(
            args.data_dir if not args.test else None,
            seq_len=args.sequence_length,
            test_mode=args.test
        )
        val_dataset = GestureDataset(
            args.val_data_dir if not args.test else None,
            seq_len=args.sequence_length,
            test_mode=args.test
        )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Create or load CNN Gesture Classifier model 
    if resume_checkpoint is not None:
        # Load model from checkpoint
        model, trainer, quantizer, _ = load_model_from_checkpoint(args.resume, device)
    else:
        # Create new model
        model = CNNGestureClassifier(
            input_dim=2,  # Only x, y coordinates
            sequence_length=args.sequence_length,
            latent_dim=args.latent_dim,
            num_classes=args.num_classes,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            kernel_size=args.kernel_size,
            use_tfm_decoder=args.use_tfm_decoder,
            tfm_d_model=args.tfm_d_model,
            tfm_num_layers=args.tfm_num_layers,
            tfm_num_heads=args.tfm_num_heads,
            tfm_mlp_ratio=args.tfm_mlp_ratio,
            tfm_dropout=args.tfm_dropout,
            tfm_attn_dropout=args.tfm_attn_dropout,
        ).to(device)
        
        quantizer = CoordinateQuantizer(num_classes=args.num_classes)
        trainer = CNNGestureClassifierTrainer(model, quantizer, device=device)
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    wandb.log({"model/num_parameters": num_params})
    print(f"CNN Gesture Classifier has {num_params:,} parameters")
    
    if args.test:
        print("\n=== CNN GESTURE CLASSIFIER TEST MODE SUMMARY ===")
        print(f"Epochs: {args.num_epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Sequence length: {args.sequence_length}")
        print(f"Decoder type: {'Transformer' if args.use_tfm_decoder else 'CNN'}")
        if args.use_tfm_decoder:
            print(f"TFM d_model: {args.tfm_d_model}, layers: {args.tfm_num_layers}, heads: {args.tfm_num_heads}")
        print(f"Encoder channels: {encoder_channels}")
        print(f"Decoder channels: {decoder_channels}")
        print(f"Model parameters: {num_params:,}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print("================================================\n")
    elif args.overfit:
        print("\n=== OVERFIT MODE SUMMARY ===")
        print(f"Total iterations: {len(train_dataset)}")
        print(f"Batch size: {args.batch_size}")
        print(f"Sequence length: {args.sequence_length}")
        print(f"Decoder type: {'Transformer' if args.use_tfm_decoder else 'CNN'}")
        if args.use_tfm_decoder:
            print(f"TFM d_model: {args.tfm_d_model}, layers: {args.tfm_num_layers}, heads: {args.tfm_num_heads}")
        print(f"Model parameters: {num_params:,}")
        print(f"Expected loss: Should approach 0 (perfect overfitting)")
        print("=============================\n")
    
    # Prepare training parameters
    start_epoch = 0
    if resume_checkpoint is not None:
        start_epoch = resume_checkpoint.get('epoch', 0) + 1  # Start from next epoch
        print(f"Resuming training from epoch {start_epoch}")
    
    # Train model
    train_classifier_model(
        model=model,
        trainer=trainer,
        quantizer=quantizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device,
        checkpoint_dir=checkpoint_dir,
        val_interval=args.val_interval,
        vis_interval=args.vis_interval,
        start_epoch=start_epoch,
        resume_checkpoint=resume_checkpoint,
        w_smooth=args.w_smooth,
    )

if __name__ == "__main__":
    main()

# ============================================================================
# TRAINING COMMANDS - Copy and run these commands
# ============================================================================

"""
# 0. TEST MODE - NO DATA NEEDED (2 minutes)
# Run with synthetic data to test CNN gesture classifier model, training, and visualization
python -m conditioned_gesture_generator.train_cnn_gesture_classifier --test

# Test with specific CNN parameters
python -m conditioned_gesture_generator.train_cnn_gesture_classifier --test \
    --device cpu \
    --encoder_channels 32,64 \
    --decoder_channels 64,32 \
    --wandb_project cnn-gesture-classifier-test-run

# 0a. MULTIPLE DATA SOURCES - Uses both gesture_data and trajectory data automatically
# The script now supports multiple data directories and formats:
# - resource/gesture_data/train (batch_*_actions.npy files)
# - resource/dataset/final_data/train_subset (*_act.npy files) 
python -m conditioned_gesture_generator.train_cnn_gesture_classifier --test

# 0b. RESUME FROM CHECKPOINT - Continue training from saved model
# python -m conditioned_gesture_generator.train_cnn_gesture_classifier \
#     --resume checkpoints/run_20241231_120000_cnn_model/elite_model.pth

# 1. QUICK CNN TEST (10 minutes) - Auto-detects both data directories
python -m conditioned_gesture_generator.train_cnn_gesture_classifier \
    --batch_size 8 \
    --num_epochs 3 \
    --lr 1e-3 \
    --latent_dim 64 \
    --encoder_channels 32,64,128 \
    --decoder_channels 128,64 \
    --sequence_length 250 \
    --wandb_project cnn-gesture-classifier-real-test \
    --device auto

# 1a. EXPLICIT MULTIPLE DATA DIRECTORIES 
# You can specify multiple data directories explicitly:
python -m conditioned_gesture_generator.train_cnn_gesture_classifier \
    --data_dir resource/gesture_data/train \
    --data_dir resource/dataset/final_data/train_subset \
    --val_data_dir resource/gesture_data/val \
    --val_data_dir resource/dataset/final_data/val_subset \
    --batch_size 8 --num_epochs 3 --device auto

# 2. CUDA CNN TEST (15 minutes)
python -m conditioned_gesture_generator.train_cnn_gesture_classifier \
    --data_dir resource/gesture_data/train \
    --val_data_dir resource/gesture_data/val \
    --batch_size 16 \
    --num_epochs 10 \
    --lr 2e-4 \
    --latent_dim 128 \
    --encoder_channels 64,128,256 \
    --decoder_channels 256,128,64 \
    --sequence_length 250 \
    --wandb_project cnn-gesture-classifier-cuda-test \
    --device cuda

# 3. FULL CNN TRAINING (2-3 hours) - Uses all available data automatically
python -m conditioned_gesture_generator.train_cnn_gesture_classifier \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 1e-4 \
    --latent_dim 128 \
    --encoder_channels 64,128,256,512 \
    --decoder_channels 512,256,128,64 \
    --sequence_length 250 \
    --wandb_project cnn-gesture-classifier-full

# 3a. RESUME FULL TRAINING FROM CHECKPOINT
python -m conditioned_gesture_generator.train_cnn_gesture_classifier \
    --resume checkpoints/run_20241231_120000_cnn_model/elite_model.pth \
    --num_epochs 200 \
    --wandb_project cnn-gesture-classifier-resumed

# EXPECTED RESULTS:
# Test Mode: CNN model runs, plots generated, synthetic data works (~2 min)
# Real Data Test: CNN model ~200K params, loss decreases, coordinate classification works (~10 min) 
# CUDA Test: GPU memory efficient, good reconstruction accuracy >70% (~15 min)
# Full Training: X/Y accuracy >80%, MSE <0.01 (~2-3 hours)
# Resume Training: Continues from saved epoch with preserved optimizer state

# NEW FEATURES:
# - Multi-format data loading: Supports both gesture_data (*_actions.npy) and trajectory (*_act.npy) formats
# - Auto data discovery: Finds data in resource/gesture_data/train and resource/dataset/final_data/train_subset automatically  
# - Multiple data directories: Use --data_dir multiple times to specify different sources
# - Checkpoint resuming: Use --resume path/to/checkpoint.pth to continue training without specifying architecture
# - Format mixing: Can train on both flat action streams and trajectory action blocks in the same run

# WANDB METRICS TO MONITOR:
# - val/mse (target: <0.01)
# - val/x_accuracy, val/y_accuracy (target: >0.8)
# - val/reconstructions (visual quality every vis_interval steps)
# - val/latent_space (clustering visualization)
# - val/generated (sample quality from random latent vectors)

# DATA FORMAT SUPPORT:
# - gesture_data format: [timesteps, 3] or [num_sequences, seq_len, 3] from stage 1 processing
# - trajectory format: [7, 250, 3] action blocks from full preprocessing pipeline
# - Mixed datasets: Can combine both formats in single training run for maximum data utilization

# NEW FEATURES:
# 1. TRANSFORMER DECODER:
# - Use --use_tfm_decoder flag to use transformer-based decoder instead of CNN decoder
# - Additional TFM parameters: --tfm_d_model, --tfm_num_layers, --tfm_num_heads, etc.
# - Model config automatically saves decoder type for checkpoint loading

# 2. OVERFIT TEST:
# - Use --overfit flag to run overfitting test on single sample
# - Runs for (epochs * 50) iterations on the same deterministic sample
# - Expected behavior: Loss should approach 0, demonstrating perfect memorization

# TRANSFORMER DECODER EXAMPLES:

# 0c. TEST WITH TRANSFORMER DECODER (2 minutes)
python -m conditioned_gesture_generator.train_cnn_gesture_classifier_no_int --test --use_tfm_decoder

# 0d. OVERFIT TEST WITH CNN DECODER (5 minutes)
python -m conditioned_gesture_generator.train_cnn_gesture_classifier_no_int --overfit --num_epochs 10

# 0e. OVERFIT TEST WITH TRANSFORMER DECODER (5 minutes)
python -m conditioned_gesture_generator.train_cnn_gesture_classifier_no_int --overfit --num_epochs 10 --use_tfm_decoder

# 3b. QUICK TRAINING WITH TRANSFORMER DECODER (15 minutes)
python -m conditioned_gesture_generator.train_cnn_gesture_classifier_no_int \\
    --batch_size 8 \\
    --num_epochs 3 \\
    --lr 1e-3 \\
    --latent_dim 64 \\
    --use_tfm_decoder \\
    --tfm_d_model 256 \\
    --tfm_num_layers 4 \\
    --tfm_num_heads 4 \\
    --wandb_project cnn-tfm-gesture-classifier-test

# 3c. FULL TRAINING WITH TRANSFORMER DECODER (3-4 hours)
python -m conditioned_gesture_generator.train_cnn_gesture_classifier_no_int \\
    --batch_size 16 \\
    --num_epochs 100 \\
    --lr 5e-5 \\
    --latent_dim 128 \\
    --use_tfm_decoder \\
    --tfm_d_model 512 \\
    --tfm_num_layers 6 \\
    --tfm_num_heads 8 \\
    --wandb_project cnn-tfm-gesture-classifier-full

# 3d. RESUME TFM TRAINING FROM CHECKPOINT
python -m conditioned_gesture_generator.train_cnn_gesture_classifier_no_int \\
    --resume checkpoints/run_20241231_120000_tfm_model/elite_model.pth \\
    --num_epochs 200 \\
    --wandb_project cnn-tfm-gesture-classifier-resumed

# EXPECTED RESULTS FOR NEW FEATURES:
# Transformer Decoder: Generally better reconstruction quality, more parameters
# Overfit Test: Loss should go to ~0.001 or lower, perfect reconstruction of single sample
# Config Saving: Checkpoints include decoder type, can resume TFM or CNN models correctly
"""
