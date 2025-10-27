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


class VVAELAMDataset(Dataset):
    """
    Dataset for VVAE Latent Action Model training.
    Loads pre-encoded VVAE latent sequences from HDF5 files.

    VVAE format: [T, C=16, T_latent=1, H=64, W=64]
    - T_latent=1 to avoid temporal leakage (previous T_latent=2 was a bug)
    - Each frame is independently encoded without looking ahead
    - Returns sequences in format [T, C, H, W] for VVAELatentActionVAE

    Uses the same efficient loading strategy as LAMDataset:
    - No in-memory caching (prevents OOM)
    - Worker-aware file handle management
    - Lazy loading with trajectory indexing
    """

    def __init__(
        self,
        data_dir: str,
        manifest_path: str,
        split: str = 'train',
        sequence_length: int = 8,  # Number of latent frames (not chunks)
        stride: int = 1
    ):
        """
        Args:
            data_dir: Path to directory containing VVAE HDF5 files (e.g., output_h5/)
            manifest_path: Path to manifest JSON file with train/val/test splits
            split: One of 'train', 'validation', or 'test'
            sequence_length: Number of latent frames in each sequence (default 8 = 4 seconds of video)
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
            raise ValueError(f"Split '{split}' not found in manifest. Available: {list(manifest['splits'].keys())}")

        file_names = manifest['splits'][split]
        if not file_names:
            raise ValueError(f"No files in {split} split")

        self.file_list = [self.data_dir / fname for fname in file_names]

        # Build index of all sequences
        self._build_sequence_index()

        # File handles for each worker
        self._file_handles = {}

    def _build_sequence_index(self):
        """Build an index of all valid sequences across all files."""
        self.sequence_index = []

        for file_path in self.file_list:
            if not file_path.exists():
                logger.warning(f"File {file_path} not found, skipping")
                continue

            # Check file contents
            with h5py.File(file_path, 'r') as f:
                if 'embeddings' not in f:
                    logger.warning(f"'embeddings' not found in {file_path}, skipping")
                    continue

                data = f['embeddings']
                # Expected shape: [T, C=16, T_latent=1, H=64, W=64] (new correct format)
                # Also support: [N_chunks, C=16, T_latent=2, H=64, W=64] (old buggy format for backward compatibility)
                if len(data.shape) != 5:
                    logger.warning(f"Unexpected shape {data.shape} in {file_path}, skipping")
                    continue

                N_chunks, C, T_latent, H, W = data.shape

                # Validate T_latent is 1 or 2
                if T_latent not in [1, 2]:
                    logger.warning(f"Unexpected T_latent={T_latent} in {file_path}, expected 1 or 2, skipping")
                    continue

                # Flatten temporal dimension: N_chunks chunks × T_latent frames per chunk
                total_frames = N_chunks * T_latent

                # Generate all valid sequence start indices
                for start_idx in range(0, total_frames - self.sequence_length + 1, self.stride):
                    self.sequence_index.append({
                        'file': str(file_path),
                        'start': start_idx,
                        'length': self.sequence_length,
                        'chunk_info': (N_chunks, C, T_latent, H, W)
                    })

        logger.info(f"{self.split}: Built index with {len(self.sequence_index)} sequences from {len(self.file_list)} files")

    def _get_file_handle(self, file_path: str):
        """Get file handle for current worker."""
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
            - 'sequence': Full sequence [T, C, H, W] where T=sequence_length, C=16, H=W=64

        The VVAELatentActionVAE wrapper expects this format.
        """
        seq_info = self.sequence_index[idx]

        # Get file handle for this worker
        h5_file = self._get_file_handle(seq_info['file'])

        # Load the specific sequence
        start = seq_info['start']
        end = start + seq_info['length']

        N_chunks, C, T_latent, H, W = seq_info['chunk_info']

        # Calculate which chunks we need
        chunk_start = start // T_latent
        frame_start_in_chunk = start % T_latent
        chunk_end = (end - 1) // T_latent + 1

        # Load required chunks: [num_chunks_needed, C, T_latent, H, W]
        data = h5_file['embeddings'][chunk_start:chunk_end]

        # Reshape to flatten temporal dimension: [num_chunks_needed * T_latent, C, H, W]
        # We need to transpose from [num_chunks, C, T_latent, H, W] to [num_chunks, T_latent, C, H, W]
        # then reshape to [num_chunks * T_latent, C, H, W]
        num_chunks_loaded = data.shape[0]
        data = np.transpose(data, (0, 2, 1, 3, 4))  # [num_chunks, T_latent, C, H, W]
        data = data.reshape(-1, C, H, W)  # [num_chunks * T_latent, C, H, W]

        # Extract exact sequence
        local_start = frame_start_in_chunk
        local_end = local_start + seq_info['length']
        sequence = data[local_start:local_end]  # [T, C, H, W]

        # Convert to torch tensor
        sequence = torch.from_numpy(sequence.copy()).float()

        return {
            'sequence': sequence,  # [T, C=16, H=64, W=64]
        }


def create_vvae_dataloaders(
    data_dir: str,
    manifest_path: str,
    batch_size: int = 32,
    sequence_length: int = 8,
    stride_train: int = 1,
    stride_val: int = 2,
    num_workers: int = 8,
    pin_memory: bool = True,
    ddp: bool = False
):
    """
    Create train, validation, and test dataloaders for VVAE LAM training.

    Args:
        data_dir: Path to directory containing VVAE HDF5 files (e.g., output_h5/)
        manifest_path: Path to manifest JSON file
        batch_size: Batch size for training
        sequence_length: Number of latent frames in each sequence
        stride_train: Stride for training sequences (smaller = more augmentation)
        stride_val: Stride for validation/test sequences
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for GPU transfer
        ddp: Whether to use DistributedDataParallel

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        val_loader and test_loader may be None if splits are empty
    """
    use_ddp = ddp

    # Create train dataset
    train_dataset = VVAELAMDataset(
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

    # Create validation dataset
    val_loader = None
    val_sampler = None
    try:
        val_dataset = VVAELAMDataset(
            data_dir=data_dir,
            manifest_path=manifest_path,
            split='validation',
            sequence_length=sequence_length,
            stride=stride_val
        )

        if len(val_dataset) > 0:
            if use_ddp:
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_dataset,
                    shuffle=True
                )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=(val_sampler is None),
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=val_sampler,
                drop_last=False
            )
    except Exception as e:
        logger.warning(f"Could not create validation dataloader: {e}")

    # Create test dataset
    test_loader = None
    test_sampler = None
    try:
        test_dataset = VVAELAMDataset(
            data_dir=data_dir,
            manifest_path=manifest_path,
            split='test',
            sequence_length=sequence_length,
            stride=stride_val
        )

        if len(test_dataset) > 0:
            if use_ddp:
                test_sampler = torch.utils.data.distributed.DistributedSampler(
                    test_dataset,
                    shuffle=True
                )

            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=(test_sampler is None),
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=test_sampler,
                drop_last=False
            )
    except Exception as e:
        logger.warning(f"Could not create test dataloader: {e}")

    if use_ddp and torch.distributed.get_rank() == 0:
        logger.info(f"[DDP] Created VVAE dataloaders with DistributedSampler")
        logger.info(f"[DDP] Train: {len(train_dataset)} sequences")
        if val_loader:
            logger.info(f"[DDP] Val: {len(val_dataset)} sequences")
        if test_loader:
            logger.info(f"[DDP] Test: {len(test_dataset)} sequences")
    else:
        logger.info(f"Created VVAE dataloaders")
        logger.info(f"Train: {len(train_dataset)} sequences")
        if val_loader:
            logger.info(f"Val: {len(val_dataset)} sequences")
        if test_loader:
            logger.info(f"Test: {len(test_dataset)} sequences")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the VVAE dataloader
    import sys

    data_dir = "/home/kevin/work/vj2-gui/output_h5"
    manifest_path = "/home/kevin/work/vj2-gui/output_h5/manifest.json"

    print("Testing VVAE dataloader...")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_vvae_dataloaders(
        data_dir=data_dir,
        manifest_path=manifest_path,
        batch_size=4,
        sequence_length=8,
        num_workers=0  # Set to 0 for debugging
    )

    # Test loading a batch
    print(f"\nTrain batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")

    # Get one batch
    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    print(f"  sequence shape: {batch['sequence'].shape}")  # Should be [B, T, C=16, H=64, W=64]
    print(f"  Expected: [B=4, T=8, C=16, H=64, W=64]")

    # Verify values are reasonable
    print(f"\nData statistics:")
    print(f"  Min: {batch['sequence'].min():.4f}")
    print(f"  Max: {batch['sequence'].max():.4f}")
    print(f"  Mean: {batch['sequence'].mean():.4f}")
    print(f"  Std: {batch['sequence'].std():.4f}")
