import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GestureDataset(Dataset):
    """
    Dataset for loading gesture action data from .npy files.
    Assumes data structure with train/val folders containing batch_*.npy files.
    """
    
    def __init__(self, data_path, sequence_length=None, normalize=True):
        """
        Args:
            data_path: Path to folder containing .npy batch files
            sequence_length: If specified, clips sequences to this length
            normalize: Whether to normalize coordinates to [0, 1] range
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        # Find all .npy files in the directory
        self.file_paths = sorted(glob.glob(os.path.join(data_path, "*.npy")))
        if not self.file_paths:
            raise ValueError(f"No .npy files found in {data_path}")
        
        # Load first file to get data info
        sample_data = np.load(self.file_paths[0])
        self.raw_data_shape = sample_data.shape
        
        # Data is stored as (N, 3) but we need (N//sequence_length, sequence_length, 3) sequences
        if sequence_length is None:
            raise ValueError("sequence_length must be specified for this data format")
            
        self.points_per_file = self.raw_data_shape[0]
        self.sequences_per_file = self.points_per_file // sequence_length
        self.total_sequences = len(self.file_paths) * self.sequences_per_file
        
        print(f"Found {len(self.file_paths)} batch files")
        print(f"Each batch shape: {self.raw_data_shape}")
        print(f"Sequences per file: {self.sequences_per_file} (length {sequence_length})")
        print(f"Total sequences: {self.total_sequences}")
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        # Determine which file and which sequence within that file
        file_idx = idx // self.sequences_per_file
        seq_idx = idx % self.sequences_per_file
        
        # Load the appropriate file
        data = np.load(self.file_paths[file_idx])  # Shape: (N, 3)
        
        # Extract the sequence starting at the correct position
        start_idx = seq_idx * self.sequence_length
        end_idx = start_idx + self.sequence_length
        sequence = data[start_idx:end_idx]  # Shape: (sequence_length, 3)
        
        # Extract coordinate data (assuming first 2 columns are x, y coordinates)
        coords = sequence[:, :2]  # Shape: (sequence_length, 2)
        
        # Normalize coordinates to [0, 1] range if requested
        if self.normalize:
            # Assuming coordinates are in some range, normalize to [0, 1]
            # You may need to adjust this based on your coordinate system
            coords = np.clip(coords, 0, 1)
        
        return torch.FloatTensor(coords)


class StreamingGestureDataLoader:
    """
    Streaming data loader that can handle train/val splits from separate folders.
    Provides efficient batching and streaming during training.
    """
    
    def __init__(self, train_path=None, val_path=None, batch_size=32, 
                 sequence_length=None, normalize=True, num_workers=4, 
                 shuffle=True, pin_memory=True):
        """
        Args:
            train_path: Path to training data folder
            val_path: Path to validation data folder  
            batch_size: Batch size for data loading
            sequence_length: Max sequence length (clips longer sequences)
            normalize: Whether to normalize coordinates to [0, 1]
            num_workers: Number of worker processes for data loading
            shuffle: Whether to shuffle training data
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        
        # Create datasets
        self.train_dataset = None
        self.val_dataset = None
        
        if train_path and os.path.exists(train_path):
            self.train_dataset = GestureDataset(
                train_path, sequence_length, normalize
            )
        
        if val_path and os.path.exists(val_path):
            self.val_dataset = GestureDataset(
                val_path, sequence_length, normalize
            )
    
    def get_train_loader(self):
        """Returns training data loader."""
        if self.train_dataset is None:
            raise ValueError("No training dataset available")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True  # Drop incomplete batches for consistent training
        )
    
    def get_val_loader(self):
        """Returns validation data loader."""
        if self.val_dataset is None:
            raise ValueError("No validation dataset available")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def get_data_info(self):
        """Returns information about the loaded datasets."""
        info = {}
        
        if self.train_dataset:
            info['train_size'] = len(self.train_dataset)
            info['train_shape'] = self.train_dataset.data_shape
        
        if self.val_dataset:
            info['val_size'] = len(self.val_dataset)
            info['val_shape'] = self.val_dataset.data_shape
            
        return info


# Example usage functions
def create_data_loaders(data_root, batch_size=32, sequence_length=None, 
                       normalize=True, num_workers=4):
    """
    Convenience function to create train/val data loaders.
    
    Args:
        data_root: Root path containing 'train/' and 'val/' subfolders
        batch_size: Batch size for training
        sequence_length: Max sequence length
        normalize: Whether to normalize coordinates
        num_workers: Number of data loading workers
    
    Returns:
        dict: Dictionary with 'train' and 'val' DataLoaders
    """
    train_path = os.path.join(data_root, 'train')
    val_path = os.path.join(data_root, 'val')
    
    loader = StreamingGestureDataLoader(
        train_path=train_path if os.path.exists(train_path) else None,
        val_path=val_path if os.path.exists(val_path) else None,
        batch_size=batch_size,
        sequence_length=sequence_length,
        normalize=normalize,
        num_workers=num_workers
    )
    
    loaders = {}
    try:
        loaders['train'] = loader.get_train_loader()
    except ValueError:
        print("Warning: No training data found")
    
    try:
        loaders['val'] = loader.get_val_loader()
    except ValueError:
        print("Warning: No validation data found")
    
    return loaders, loader.get_data_info()