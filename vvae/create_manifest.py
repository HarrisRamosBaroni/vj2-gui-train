import argparse
import os
from pathlib import Path
import numpy as np
import sys

def find_video_files(input_dir):
    """Recursively find all video files in a directory."""
    video_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) and not f.startswith('._'):
                video_files.append(os.path.abspath(os.path.join(root, f)))
    return sorted(video_files)

def create_manifest():
    parser = argparse.ArgumentParser(
        description="Create a processing manifest for distributed video preprocessing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', type=str, help='Directory containing video files to process.')
    parser.add_argument('manifest_dir', type=str, help='Directory to store the manifest files (todo, in_progress, done, failed).')
    parser.add_argument('--num-batches', type=int, default=100, help='Number of batches to split the dataset into.')
    parser.add_argument('--videos-per-batch', type=int, default=None, help='Number of videos per batch. Overrides --num-batches if set.')

    args = parser.parse_args()

    # Find all video files
    print(f"Scanning for video files in {args.input_dir}...")
    video_paths = find_video_files(args.input_dir)

    if not video_paths:
        print("Error: No video files found.")
        sys.exit(1)

    print(f"Found {len(video_paths)} video files.")

    # Determine batching strategy
    if args.videos_per_batch:
        num_batches = int(np.ceil(len(video_paths) / args.videos_per_batch))
        print(f"Splitting into {num_batches} batches of up to {args.videos_per_batch} videos each.")
        batches = np.array_split(video_paths, num_batches)
    else:
        num_batches = min(args.num_batches, len(video_paths))
        print(f"Splitting into {num_batches} batches.")
        batches = np.array_split(video_paths, num_batches)

    # Create manifest directories
    manifest_path = Path(args.manifest_dir)
    todo_path = manifest_path / 'todo'
    in_progress_path = manifest_path / 'in_progress'
    done_path = manifest_path / 'done'
    failed_path = manifest_path / 'failed'

    for p in [manifest_path, todo_path, in_progress_path, done_path, failed_path]:
        p.mkdir(parents=True, exist_ok=True)

    # Check if 'todo' directory is already populated
    if any(f.is_file() for f in todo_path.iterdir()):
        user_input = input(f"The directory {todo_path} is not empty. Overwrite? (y/n): ").lower()
        if user_input != 'y':
            print("Aborting.")
            sys.exit(0)
        # Clean up old batch files
        for f in todo_path.glob('*.txt'):
            f.unlink()

    # Write batch files
    num_digits = len(str(len(batches)))
    for i, batch in enumerate(batches):
        if batch.size == 0:
            continue
        batch_filename = f"batch_{str(i+1).zfill(num_digits)}.txt"
        batch_filepath = todo_path / batch_filename
        with open(batch_filepath, 'w') as f:
            for video_path in batch:
                f.write(f"{video_path}\n")

    print(f"\nSuccessfully created {len(batches)} batch files in {todo_path}")
    print("Manifest generation complete.")

if __name__ == "__main__":
    create_manifest()
