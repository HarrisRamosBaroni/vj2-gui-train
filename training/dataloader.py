# vj2_dataloader.py - Loads pre-processed (State, Action) pairs
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from logging import getLogger
import os
import h5py
import json

logger = getLogger(__name__)

# ---------------------------------------------------------------------------

class PreprocessedGUIAgentDataset(Dataset):
    """
    A PyTorch Dataset for loading pre-processed trajectories from individual
    .npy files.

    This dataset reads trajectory data where each trajectory consists of a pair
    of files: one for embeddings (`*_emb.npy`) and one for actions (`*_act.npy`).
    It uses memory-mapping for efficient data loading.

    Returns a single trajectory of (embeddings, actions).

    Parameters
    ----------
    processed_data_dir: str
        Directory containing the pre-processed .npy trajectory files.
    """
    def __init__(self, processed_data_dir: str):
        self.data_dir = Path(processed_data_dir)
        self.file_list = sorted(self.data_dir.glob("*_emb.npy"))
        
        if not self.file_list:
            raise RuntimeError(f"No '*_emb.npy' files found in {self.data_dir}. "
                               f"Have you run the migration script `migrate_npz_to_npy.py`?")
        
        self.total_trajectories = len(self.file_list)
        logger.info(f"Found {self.total_trajectories} trajectories from .npy files.")

    def __len__(self):
        return self.total_trajectories

    def __getitem__(self, idx: int):
        if not 0 <= idx < self.total_trajectories:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.total_trajectories} trajectories.")

        emb_path = self.file_list[idx]
        act_path = emb_path.with_name(emb_path.name.replace('_emb.npy', '_act.npy'))

        if not act_path.exists():
            # This check is important for data integrity.
            raise FileNotFoundError(f"Action file not found for embedding: {emb_path}\n"
                                    f"Expected at: {act_path}")
        
        embeddings = np.load(emb_path, mmap_mode='r')
        actions = np.load(act_path, mmap_mode='r')

        return torch.from_numpy(embeddings.copy()).float(), torch.from_numpy(actions.copy()).float()

class H5TrajectoryDataset(Dataset):
    """
    A PyTorch Dataset for loading pre-processed trajectories from HDF5 files.
    """
    def __init__(self, processed_data_dir: str, file_whitelist: list = None):
        self.data_dir = Path(processed_data_dir)
        
        if file_whitelist is None:
            self.h5_files = sorted(self.data_dir.glob("*.h5"))
        else:
            self.h5_files = sorted([self.data_dir / f for f in file_whitelist if (self.data_dir / f).exists()])

        if not self.h5_files:
            if file_whitelist is not None:
                logger.warning(f"No '.h5' files from the manifest whitelist were found in {self.data_dir}.")
            raise RuntimeError(f"No usable '.h5' files found in {self.data_dir}.")

        self._file_handles = {}
        self.trajectory_index = []
        
        total_trajectories = 0
        for file_path in self.h5_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    num_in_file = f['embeddings'].shape[0]
                    for i in range(num_in_file):
                        self.trajectory_index.append((str(file_path), i))
                    total_trajectories += num_in_file
            except Exception as e:
                logger.error(f"Could not read or index file {file_path}: {e}")

        self.total_trajectories = total_trajectories
        logger.info(f"Found {self.total_trajectories} trajectories across {len(self.h5_files)} HDF5 files.")

    def __len__(self):
        return self.total_trajectories

    def _get_file_handle(self, file_path):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        
        if worker_id not in self._file_handles:
            self._file_handles[worker_id] = {}
        
        if file_path not in self._file_handles[worker_id]:
            self._file_handles[worker_id][file_path] = h5py.File(file_path, 'r')
            
        return self._file_handles[worker_id][file_path]

    def __getitem__(self, idx):
        file_path, local_idx = self.trajectory_index[idx]
        h5_file = self._get_file_handle(file_path)
        
        embeddings = h5_file['embeddings'][local_idx]
        actions = h5_file['actions'][local_idx]
        
        return torch.from_numpy(embeddings).float(), torch.from_numpy(actions).float()

# ---------------------------------------------------------------------------

def init_preprocessed_data_loader(processed_data_dir: str, batch_size: int, num_workers=0, manifest_path: str = None, split_name: str = None):
    """Initializes the DataLoader for the pre-processed dataset."""
    data_path = Path(processed_data_dir)
    
    if any(data_path.glob("*.h5")):
        logger.info("HDF5 files detected. Using H5TrajectoryDataset.")
        file_whitelist = None
        if manifest_path and split_name:
            logger.info(f"Loading '{split_name}' split from manifest: {manifest_path}")
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            if split_name not in manifest['splits']:
                raise ValueError(f"Split '{split_name}' not found in manifest file. Available splits: {list(manifest['splits'].keys())}")
            file_whitelist = manifest['splits'][split_name]
        
        dataset = H5TrajectoryDataset(processed_data_dir=processed_data_dir, file_whitelist=file_whitelist)
    elif any(data_path.glob("*_emb.npy")):
        logger.info(".npy files detected. Using PreprocessedGUIAgentDataset.")
        dataset = PreprocessedGUIAgentDataset(processed_data_dir=processed_data_dir)
    else:
        raise FileNotFoundError(f"No trajectory data (.h5 or .npy) found in '{processed_data_dir}'")

    use_sampler = torch.distributed.is_available() and torch.distributed.is_initialized()
    
    sampler = None
    if use_sampler:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler
    )
    
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        if rank == 0:
            logger.info(f"[DDP] Dataloader created with {len(dataset)} samples, batch_size={batch_size}")
            if sampler:
                logger.info(f"[DDP] Using DistributedSampler with {world_size} ranks")
    else:
        logger.info("Preprocessed GUI Agent data loader created.")
    return data_loader, sampler

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    # This block is for testing the data loader.
    # It assumes you have already run `process_raw_data.py` and have data
    # in the './processed_data' directory.
    
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing PreprocessedGUIAgentDataset...")

    processed_dir = "./processed_data"
    
    if not Path(processed_dir).exists() or not any(Path(processed_dir).iterdir()):
         logger.warning(f"'{processed_dir}' is empty or doesn't exist.")
         logger.warning("Please run process_raw_data.py to generate test data.")
    else:
        try:
            test_loader, _ = init_preprocessed_data_loader(
                processed_data_dir=processed_dir,
                batch_size=4
            )

            # Fetch one batch
            sample_embeddings, sample_actions = next(iter(test_loader))

            print("\n--- DataLoader Test ---")
            print(f"Batch embeddings shape: {sample_embeddings.shape}")
            print(f"Batch actions shape:      {sample_actions.shape}")
            print("--- Test Successful ---")
            
        except Exception as e:
            print(f"\n--- Test Failed ---")
            print(f"An error occurred during the test: {e}")