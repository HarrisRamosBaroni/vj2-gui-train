"""
Create manifest file for VVAE LAM training.
Generates train/val/test splits from VVAE h5 files.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import h5py
from datetime import datetime


def create_vvae_manifest(
    data_dir: str,
    output_path: str = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Create manifest JSON file with train/val/test splits.

    Args:
        data_dir: Directory containing VVAE h5 files
        output_path: Path to save manifest JSON (default: data_dir/manifest.json)
        train_ratio: Fraction of files for training (default: 0.8)
        val_ratio: Fraction of files for validation (default: 0.1)
        test_ratio: Fraction of files for test (default: 0.1)
        seed: Random seed for reproducible splits
    """
    data_dir = Path(data_dir)

    if output_path is None:
        output_path = data_dir / "manifest.json"
    else:
        output_path = Path(output_path)

    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    print("="*60)
    print("Creating VVAE LAM Manifest")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Output path: {output_path}")
    print(f"Split ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
    print(f"Random seed: {seed}\n")

    # Get all h5 files
    all_files = sorted(list(data_dir.glob("*.h5")))
    if not all_files:
        raise ValueError(f"No .h5 files found in {data_dir}")

    print(f"Found {len(all_files)} h5 files")

    # Validate files and collect metadata
    valid_files = []
    total_sequences = 0

    print("\nValidating files...")
    for file_path in all_files:
        try:
            with h5py.File(file_path, 'r') as f:
                if 'embeddings' not in f:
                    print(f"  ⚠️  Skipping {file_path.name}: no 'embeddings' key")
                    continue

                data = f['embeddings']
                if len(data.shape) != 5:
                    print(f"  ⚠️  Skipping {file_path.name}: unexpected shape {data.shape}")
                    continue

                N_chunks, C, T_latent, H, W = data.shape

                # Expected: C=16, T_latent=2, H=64, W=64
                if C != 16 or T_latent != 2 or H != 64 or W != 64:
                    print(f"  ⚠️  Skipping {file_path.name}: unexpected dimensions (C={C}, T={T_latent}, H={H}, W={W})")
                    continue

                num_latent_frames = N_chunks * T_latent
                valid_files.append({
                    'filename': file_path.name,
                    'num_chunks': N_chunks,
                    'num_latent_frames': num_latent_frames,
                    'shape': list(data.shape)
                })
                total_sequences += num_latent_frames

        except Exception as e:
            print(f"  ⚠️  Skipping {file_path.name}: error reading file ({e})")
            continue

    if not valid_files:
        raise ValueError("No valid h5 files found!")

    print(f"\n✓ Validated {len(valid_files)} files")
    print(f"  Total latent frames: {total_sequences:,}")

    # Create random splits
    np.random.seed(seed)
    indices = np.random.permutation(len(valid_files))

    n_train = int(len(valid_files) * train_ratio)
    n_val = int(len(valid_files) * val_ratio)
    n_test = len(valid_files) - n_train - n_val  # Remaining go to test

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]

    # Create splits
    splits = {
        'train': [valid_files[i]['filename'] for i in train_indices],
        'validation': [valid_files[i]['filename'] for i in val_indices],
        'test': [valid_files[i]['filename'] for i in test_indices]
    }

    # Calculate statistics per split
    split_stats = {}
    for split_name, file_list in splits.items():
        files_in_split = [f for f in valid_files if f['filename'] in file_list]
        total_frames = sum(f['num_latent_frames'] for f in files_in_split)
        total_chunks = sum(f['num_chunks'] for f in files_in_split)

        split_stats[split_name] = {
            'num_files': len(file_list),
            'num_chunks': total_chunks,
            'num_latent_frames': total_frames
        }

    # Create manifest
    manifest = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'data_directory': str(data_dir),
            'total_files': len(valid_files),
            'seed': seed,
            'split_ratios': {
                'train': train_ratio,
                'validation': val_ratio,
                'test': test_ratio
            }
        },
        'data_format': {
            'shape': '[N_chunks, C=16, T_latent=2, H=64, W=64]',
            'description': 'VVAE latent embeddings',
            'temporal_compression': '4x (8 frames -> 2 latent frames)',
            'spatial_resolution': '64x64',
            'channels': 16
        },
        'splits': splits,
        'statistics': split_stats
    }

    # Save manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "="*60)
    print("MANIFEST CREATED SUCCESSFULLY")
    print("="*60)
    print(f"\nSplit Statistics:")
    for split_name in ['train', 'validation', 'test']:
        stats = split_stats[split_name]
        print(f"\n{split_name.upper()}:")
        print(f"  Files: {stats['num_files']}")
        print(f"  Chunks: {stats['num_chunks']:,}")
        print(f"  Latent frames: {stats['num_latent_frames']:,}")

    print(f"\nManifest saved to: {output_path}")
    print("="*60)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Create VVAE LAM manifest file")
    parser.add_argument("data_dir", type=str, help="Directory containing VVAE h5 files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for manifest.json (default: data_dir/manifest.json)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Training set ratio (default: 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="Validation set ratio (default: 0.1)")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                       help="Test set ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits (default: 42)")

    args = parser.parse_args()

    create_vvae_manifest(
        data_dir=args.data_dir,
        output_path=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
