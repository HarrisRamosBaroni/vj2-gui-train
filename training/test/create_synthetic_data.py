#!/usr/bin/env python3
"""
Create synthetic data for testing the VJ2 GUI training pipeline.
This script generates fake .npz files that match the expected format.
"""

import numpy as np
from pathlib import Path
import argparse
from config import ROLLOUT_HORIZON, OBSERVATIONS_PER_WINDOW

def create_synthetic_data(output_dir: str, num_files: int = 3, trajectories_per_file: int = 10):
    """Create synthetic training data that matches the expected format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating synthetic data in {output_path}")
    print(f"ROLLOUT_HORIZON: {ROLLOUT_HORIZON}")
    print(f"OBSERVATIONS_PER_WINDOW: {OBSERVATIONS_PER_WINDOW}")
    
    # Dimensions based on VJ2 config
    embedding_dim = 768  # Typical transformer embedding size
    action_blocks = 7    # From the original code
    action_dim_per_block = 32
    
    # Ensure sequence length is sufficient for rollout horizon
    sequence_length = max(OBSERVATIONS_PER_WINDOW, ROLLOUT_HORIZON + 2)
    
    for file_idx in range(num_files):
        # Generate synthetic embeddings and actions
        embeddings = np.random.randn(
            trajectories_per_file, 
            sequence_length, 
            embedding_dim
        ).astype(np.float32)
        
        actions = np.random.randn(
            trajectories_per_file,
            sequence_length - 1,  # Actions are typically one step shorter
            action_blocks,
            action_dim_per_block
        ).astype(np.float32)
        
        # Make actions realistic (bounded between -1 and 1 for GUI actions)
        actions = np.tanh(actions)
        
        file_path = output_path / f"synthetic_data_{file_idx:03d}.npz"
        np.savez(file_path, embeddings=embeddings, actions=actions)
        
        print(f"Created {file_path}")
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Actions shape: {actions.shape}")
    
    print(f"\nSynthetic dataset created with:")
    print(f"  {num_files} files")
    print(f"  {trajectories_per_file} trajectories per file")
    print(f"  {num_files * trajectories_per_file} total trajectories")
    print(f"  Sequence length: {sequence_length}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic data for testing")
    parser.add_argument("--output_dir", default="./test_data", help="Output directory for synthetic data")
    parser.add_argument("--num_files", type=int, default=3, help="Number of .npz files to create")
    parser.add_argument("--trajectories_per_file", type=int, default=10, help="Trajectories per file")
    
    args = parser.parse_args()
    create_synthetic_data(args.output_dir, args.num_files, args.trajectories_per_file)