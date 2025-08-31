import argparse
import numpy as np
import os
from pathlib import Path

def migrate_npz_to_npy(input_dir, output_dir, cleanup):
    """
    Converts a dataset from large .npz files to individual .npy files per trajectory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for npz_file in input_path.glob('*.npz'):
        print(f"Processing {npz_file}...")
        try:
            data = np.load(npz_file)
            embeddings = data['embeddings']
            actions = data['actions']
            num_trajectories = len(embeddings)
            filename_stem = npz_file.stem

            for i in range(num_trajectories):
                embedding_trajectory = embeddings[i]
                action_trajectory = actions[i]
                
                emb_filename = f"{filename_stem}_traj_{i:05d}_emb.npy"
                act_filename = f"{filename_stem}_traj_{i:05d}_act.npy"

                np.save(output_path / emb_filename, embedding_trajectory)
                np.save(output_path / act_filename, action_trajectory)

            print(f"Successfully converted {npz_file} into {num_trajectories} trajectories.")

            if cleanup:
                os.remove(npz_file)
                print(f"Removed original file: {npz_file}")

        except Exception as e:
            print(f"Could not process {npz_file}. Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Migrate .npz dataset to per-trajectory .npy files.")
    parser.add_argument('--input-dir', type=str, required=True, help="Path to the directory containing the source .npz files.")
    parser.add_argument('--output-dir', type=str, required=True, help="Path to the directory where the new .npy files will be saved.")
    parser.add_argument('--cleanup', action='store_true', help="If set, delete the original .npz file after it has been successfully converted.")
    
    args = parser.parse_args()
    
    migrate_npz_to_npy(args.input_dir, args.output_dir, args.cleanup)