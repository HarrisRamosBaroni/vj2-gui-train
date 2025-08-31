import argparse
import json
from pathlib import Path
import random
import datetime

def generate_split_manifest(data_dir: str, output_dir: str, name: str, ratios: list, seed: int):
    """
    Scans a directory for .h5 files and generates a JSON manifest file
    with train, validation, and test splits.
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
        
    output_path.mkdir(parents=True, exist_ok=True)

    all_files = sorted([f.name for f in data_path.glob("*.h5")])
    if not all_files:
        print(f"Warning: No '.h5' files found in {data_path}")
        return

    # Ensure split ratios sum to 1.0
    if sum(ratios) != 1.0:
        raise ValueError(f"Split ratios must sum to 1.0. Got: {ratios} (sum: {sum(ratios)})")

    # Shuffle files deterministically
    random.Random(seed).shuffle(all_files)
    
    num_total = len(all_files)
    num_train = int(num_total * ratios[0])
    num_val = int(num_total * ratios[1])
    
    train_files = all_files[:num_train]
    val_files = all_files[num_train : num_train + num_val]
    test_files = all_files[num_train + num_val :]

    manifest = {
        "version": 1,
        "description": "Dataset split manifest for VJ2-GUI experiments.",
        "metadata": {
            "source_directory": str(data_path),
            "generation_date_utc": datetime.datetime.utcnow().isoformat(),
            "seed": seed,
            "split_ratios": {
                "train": ratios[0],
                "validation": ratios[1],
                "test": ratios[2]
            },
            "file_counts": {
                "total": num_total,
                "train": len(train_files),
                "validation": len(val_files),
                "test": len(test_files)
            }
        },
        "splits": {
            "train": train_files,
            "validation": val_files,
            "test": test_files
        }
    }
    
    manifest_file = output_path / f"{name}.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Successfully generated manifest: {manifest_file}")
    print(f"  - Total files: {num_total}")
    print(f"  - Train: {len(train_files)} | Validation: {len(val_files)} | Test: {len(test_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a JSON manifest for dataset splitting.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to the mother directory containing .h5 data files.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory where the manifest file will be saved.")
    parser.add_argument("--name", type=str, default="manifest",
                        help="The base name for the output manifest file (e.g., 'experiment_A').")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        help="A list of three floats representing the train, validation, and test split ratios.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling to ensure reproducible splits.")
    
    args = parser.parse_args()
    
    generate_split_manifest(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        name=args.name,
        ratios=args.ratios,
        seed=args.seed
    )