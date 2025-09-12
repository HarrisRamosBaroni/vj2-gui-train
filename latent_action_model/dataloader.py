import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

logger = logging.getLogger(__name__)


class LAMDataset(Dataset):
    """
    Dataset for Latent Action Model training.
    Loads pre-encoded latent sequences from HDF5 files.
    
    Uses the same efficient loading strategy as train_v2.py:
    - No in-memory caching (prevents OOM)
    - Worker-aware file handle management
    - Lazy loading with trajectory indexing
    """
    
    def __init__(
        self,
        data_dir: str,
        manifest_path: str,
        split: str = 'train',
        sequence_length: int = 16,
        stride: int = 1
    ):
        """
        Args:
            data_dir: Path to directory containing HDF5 files
            manifest_path: Path to manifest JSON file with train/val/test splits
            split: One of 'train', 'validation', or 'test'
            sequence_length: Number of frames in each sequence
            stride: Stride between sequences (for data augmentation)
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.split = split
        
        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Get file list for this split
        if split not in manifest['splits']:
            raise ValueError(f"Split '{split}' not found in manifest")
        
        self.file_list = manifest['splits'][split]
        
        # Build index of all sequences
        self._build_sequence_index()
        
        # File handles for each worker (similar to H5TrajectoryDataset)
        self._file_handles = {}
    
    def _build_sequence_index(self):
        """Build an index of all valid sequences across all files."""
        self.sequence_index = []
        
        for file_name in self.file_list:
            file_path = self.data_dir / file_name
            if not file_path.exists():
                logger.warning(f"File {file_path} not found, skipping")
                continue
                
            # Quick check of file length
            with h5py.File(file_path, 'r') as f:
                # Check for both 'latents' and 'embeddings' keys
                if 'latents' in f:
                    data = f['latents']
                elif 'embeddings' in f:
                    data = f['embeddings']
                else:
                    logger.warning(f"Neither 'latents' nor 'embeddings' found in {file_path}, skipping")
                    continue
                    
                # Handle different data shapes
                if len(data.shape) == 4:  # [num_clips, T, N, D] format
                    # Flatten temporal dimension
                    num_clips, T, N, D = data.shape
                    num_frames = num_clips * T  # Total temporal steps
                    shape_info = (num_clips, T, N, D)
                else:
                    num_frames = data.shape[0]
                    shape_info = None
                
                # Generate all valid sequence start indices
                for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
                    self.sequence_index.append({
                        'file': str(file_path),  # Store full path for file handle management
                        'file_name': file_name,
                        'start': start_idx,
                        'length': self.sequence_length,
                        'shape_info': shape_info
                    })
        
        logger.info(f"Built index with {len(self.sequence_index)} sequences from {len(self.file_list)} files")
    
    def _get_file_handle(self, file_path: str):
        """Get file handle for current worker (similar to H5TrajectoryDataset)."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        
        if worker_id not in self._file_handles:
            self._file_handles[worker_id] = {}
        
        if file_path not in self._file_handles[worker_id]:
            self._file_handles[worker_id][file_path] = h5py.File(file_path, 'r')
            
        return self._file_handles[worker_id][file_path]
    
    def __len__(self) -> int:
        return len(self.sequence_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary containing:
            - 'sequence': Full sequence [z_1, ..., z_T] of shape [T, N, D]
            - 'past': Past sequence [z_1, ..., z_{T-1}] of shape [T-1, N, D]
            - 'next': Next latent z_T of shape [N, D]
        
        Note: Now directly loads from file without caching to prevent OOM.
        """
        seq_info = self.sequence_index[idx]
        
        # Get file handle for this worker
        h5_file = self._get_file_handle(seq_info['file'])
        
        # Load the specific sequence from file
        start = seq_info['start']
        end = start + seq_info['length']
        
        # Check for both 'latents' and 'embeddings' keys
        if 'latents' in h5_file:
            data_key = 'latents'
        elif 'embeddings' in h5_file:
            data_key = 'embeddings'
        else:
            raise KeyError(f"Neither 'latents' nor 'embeddings' found in file")
        
        # Handle different data shapes
        if seq_info['shape_info'] is not None:
            # Data is in [num_clips, T, N, D] format, need to reshape
            num_clips, T, N, D = seq_info['shape_info']
            
            # Calculate which clips and frames we need
            clip_start = start // T
            frame_start = start % T
            clip_end = (end - 1) // T + 1
            
            # Load required clips
            data = h5_file[data_key][clip_start:clip_end]
            
            # Reshape to [total_frames, N, D]
            data = data.reshape(-1, N, D)
            
            # Extract exact sequence
            local_start = frame_start
            local_end = local_start + seq_info['length']
            sequence = data[local_start:local_end]
        else:
            # Data is already in correct format
            sequence = h5_file[data_key][start:end]
        
        # Convert to torch tensors (copy to ensure data ownership)
        sequence = torch.from_numpy(sequence.copy()).float()
        
        # Split into past and next
        past = sequence[:-1]  # [T-1, N, D]
        next_latent = sequence[-1]  # [N, D]
        
        return {
            'sequence': sequence,  # Full sequence for encoder
            'past': past,  # Past sequence for decoder
            'next': next_latent  # Target for reconstruction
        }


def create_dataloaders(
    data_dir: str,
    manifest_path: str,
    batch_size: int = 32,
    sequence_length: int = 16,
    stride_train: int = 1,
    stride_val: int = 4,
    num_workers: int = 8,
    pin_memory: bool = True,
    return_samplers: bool = False
):
    """
    Create train, validation, and test dataloaders.
    
    Uses the same approach as train_v2.py with DistributedSampler support for DDP.
    
    Args:
        data_dir: Path to directory containing HDF5 files
        manifest_path: Path to manifest JSON file
        batch_size: Batch size for training
        sequence_length: Number of frames in each sequence
        stride_train: Stride for training sequences (smaller = more augmentation)
        stride_val: Stride for validation/test sequences
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for GPU transfer
        return_samplers: If True, also return the samplers for DDP epoch setting
    
    Returns:
        If return_samplers is False:
            Tuple of (train_loader, val_loader, test_loader)
        If return_samplers is True:
            Tuple of ((train_loader, val_loader, test_loader), (train_sampler, val_sampler, test_sampler))
        val_loader and test_loader may be None if splits don't exist
    """
    # Check if DDP is initialized
    use_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
    
    # Create train dataset
    train_dataset = LAMDataset(
        data_dir=data_dir,
        manifest_path=manifest_path,
        split='train',
        sequence_length=sequence_length,
        stride=stride_train
    )
    
    # Create train sampler
    train_sampler = None
    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, 
            shuffle=True
        )
    
    # Create train dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=train_sampler,
        drop_last=True
    )
    
    # Try to create validation dataset
    val_loader = None
    try:
        val_dataset = LAMDataset(
            data_dir=data_dir,
            manifest_path=manifest_path,
            split='validation',
            sequence_length=sequence_length,
            stride=stride_val
        )
        
        val_sampler = None
        if use_ddp:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                shuffle=False
            )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=val_sampler,
            drop_last=False
        )
    except (ValueError, KeyError) as e:
        logger.warning(f"Could not create validation dataloader: {e}")
    
    # Try to create test dataset
    test_loader = None
    try:
        test_dataset = LAMDataset(
            data_dir=data_dir,
            manifest_path=manifest_path,
            split='test',
            sequence_length=sequence_length,
            stride=stride_val
        )
        
        test_sampler = None
        if use_ddp:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                shuffle=False
            )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=test_sampler,
            drop_last=False
        )
    except (ValueError, KeyError) as e:
        logger.warning(f"Could not create test dataloader: {e}")
    
    if use_ddp and torch.distributed.get_rank() == 0:
        logger.info(f"[DDP] Created dataloaders with DistributedSampler")
        logger.info(f"[DDP] Train: {len(train_dataset)} samples")
        if val_loader:
            logger.info(f"[DDP] Val: {len(val_dataset)} samples")
        if test_loader:
            logger.info(f"[DDP] Test: {len(test_dataset)} samples")
    
    if return_samplers:
        # Return loaders and samplers for epoch setting in DDP
        return (train_loader, val_loader, test_loader), (train_sampler, val_sampler if val_loader else None, test_sampler if test_loader else None)
    else:
        return train_loader, val_loader, test_loader


def init_lam_data_loader(
    data_dir: str,
    manifest_path: str,
    batch_size: int,
    split: str = 'train',
    sequence_length: int = 16,
    stride: int = 1,
    num_workers: int = 8
):
    """
    Initialize a single LAM dataloader for a specific split.
    Similar to init_preprocessed_data_loader in train_v2.py.
    
    Args:
        data_dir: Path to directory containing HDF5 files
        manifest_path: Path to manifest JSON file
        batch_size: Batch size
        split: One of 'train', 'validation', or 'test'
        sequence_length: Number of frames in each sequence
        stride: Stride between sequences
        num_workers: Number of dataloader workers
    
    Returns:
        Tuple of (dataloader, sampler) where sampler is None if not using DDP
    """
    # Check if DDP is initialized
    use_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
    
    # Create dataset
    dataset = LAMDataset(
        data_dir=data_dir,
        manifest_path=manifest_path,
        split=split,
        sequence_length=sequence_length,
        stride=stride
    )
    
    # Create sampler if using DDP
    sampler = None
    if use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=(split == 'train')
        )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=(split == 'train')
    )
    
    if use_ddp and torch.distributed.get_rank() == 0:
        logger.info(f"[DDP] LAM dataloader created for {split} split with {len(dataset)} samples")
        if sampler:
            logger.info(f"[DDP] Using DistributedSampler with {torch.distributed.get_world_size()} ranks")
    else:
        logger.info(f"LAM dataloader created for {split} split with {len(dataset)} samples")
    
    return dataloader, sampler


if __name__ == "__main__":
    # Test the dataloader
    data_dir = "/home/kevin/work/vj2-gui/resource/dataset/final_data/train_all/mother"
    manifest_path = "/home/kevin/work/vj2-gui/resource/dataset/final_data/train_all/mother/manifest/FiLM_1.json"
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        manifest_path=manifest_path,
        batch_size=4,
        sequence_length=8,
        num_workers=0  # Set to 0 for debugging
    )
    
    # Test loading a batch
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get one batch
    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    print(f"  sequence shape: {batch['sequence'].shape}")
    print(f"  past shape: {batch['past'].shape}")
    print(f"  next shape: {batch['next'].shape}")