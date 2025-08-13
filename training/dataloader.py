# vj2_dataloader.py - Loads pre-processed (State, Action) pairs
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from logging import getLogger
import os

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
        
        # Get a list of all embedding files, which defines the dataset size.
        self.file_list = sorted(self.data_dir.glob("*_emb.npy"))
        
        if not self.file_list:
            raise RuntimeError(f"No '*_emb.npy' files found in {self.data_dir}. "
                               f"Have you run the migration script `migrate_npz_to_npy.py`?")
        
        self.total_trajectories = len(self.file_list)
        logger.info(f"Found {self.total_trajectories} trajectories.")

    def __len__(self):
        return self.total_trajectories

    def __getitem__(self, idx: int):
        if not 0 <= idx < self.total_trajectories:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.total_trajectories} trajectories.")

        emb_path = self.file_list[idx]
        # Construct the action file path by replacing the suffix.
        act_path = emb_path.with_name(emb_path.name.replace('_emb.npy', '_act.npy'))

        if not act_path.exists():
            # This check is important for data integrity.
            raise FileNotFoundError(f"Action file not found for embedding: {emb_path}\n"
                                    f"Expected at: {act_path}")
        
        # Load the arrays using memory-mapping for efficiency.
        # This avoids loading the entire file into RAM.
        embeddings = np.load(emb_path, mmap_mode='r')
        actions = np.load(act_path, mmap_mode='r')

        return torch.from_numpy(embeddings).float(), torch.from_numpy(actions).float()

# ---------------------------------------------------------------------------

def init_preprocessed_data_loader(processed_data_dir: str, batch_size: int, num_workers=0):
    """Initializes the DataLoader for the pre-processed dataset."""
    dataset = PreprocessedGUIAgentDataset(
        processed_data_dir=processed_data_dir
    )
    
    # Check for distributed training environment
    use_sampler = torch.distributed.is_available() and torch.distributed.is_initialized()
    
    sampler = None
    if use_sampler:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None), # Shuffle only if not using a distributed sampler
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler
    )
    
    # DDP sanity prints
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