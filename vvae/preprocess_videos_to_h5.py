"""
Preprocess videos to h5 files with VideoVAE+ embeddings.

Converts video files to h5 format containing only embeddings (no actions).
Each h5 file contains embeddings for all chunks from one video.
"""

import torch
import h5py
import argparse
import os
import sys
import json
from pathlib import Path
import numpy as np
from datetime import datetime
import time
import yaml

# Add vvae directory to path
vvae_dir = Path(__file__).parent
if str(vvae_dir) not in sys.path:
    sys.path.insert(0, str(vvae_dir))

from experiment import load_model, chunk_video, get_memory_usage
from vvae.video_utils import load_video_cfr

def calculate_motion_energies(embeddings, device):
    """
    Calculates motion energies from embeddings.

    Args:
        embeddings: numpy array of shape (N_chunks, C, T_latent, H_latent, W_latent)
        device: cuda/cpu

    Returns:
        A numpy array of motion energy values for one video.
    """
    if embeddings.shape[0] == 0:
        return np.array([])

    num_chunks, C, T_latent, H, W = embeddings.shape
    latents = torch.from_numpy(embeddings).view(num_chunks * T_latent, C, H, W)
    latents = latents.to(device)

    # Calculate velocity and then motion energy (magnitude)
    with torch.no_grad():
        velocities = latents[1:] - latents[:-1]
        # Calculate norm over all dimensions except time
        motion_energies = torch.norm(velocities.view(velocities.shape[0], -1), dim=1)

    return motion_energies.cpu().numpy()

def process_video_to_embeddings(video_path, model, device, chunk_size, fps=8, max_duration=None, target_size=(512, 512)):
    """
    Process a single video and extract embeddings.

    Args:
        video_path: Path to video file
        model: VideoVAE+ model
        device: cuda/cpu
        chunk_size: Number of frames per chunk
        fps: Target FPS
        max_duration: Max video duration (seconds)
        target_size: Target resolution (H, W)

    Returns:
        embeddings: numpy array of shape (N_chunks, C, T_latent, H_latent, W_latent)
    """
    # Load video
    video_array = load_video_cfr(video_path, target_fps=fps, max_duration=max_duration, target_size=target_size)

    if len(video_array) == 0:
        return None, None, None

    # Convert to tensor
    video_tensor = torch.from_numpy(video_array).float()
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (3, T, H, W)
    video_tensor = video_tensor.unsqueeze(0)  # (1, 3, T, H, W)
    video_tensor = (video_tensor / 255.0 - 0.5) * 2  # Normalize to [-1, 1]

    # Chunk video
    chunked, chunk_metadata = chunk_video(video_tensor, chunk_size)
    num_chunks = chunked.shape[0]

    # Extract embeddings for all chunks
    embeddings_list = []

    with torch.no_grad():
        for i in range(num_chunks):
            chunk = chunked[i:i+1].to(device)

            # Encode to get latent
            latent, posterior = model.encode(chunk, sample_posterior=False)

            # Store on CPU
            embeddings_list.append(latent.cpu().numpy())

            # Clear GPU cache
            if device == 'cuda':
                torch.cuda.empty_cache()

    # Stack all embeddings: (N_chunks, C, T_latent, H_latent, W_latent)
    embeddings = np.concatenate(embeddings_list, axis=0)

    # Calculate motion energies
    motion_energies = calculate_motion_energies(embeddings, device)
    total_motion_energy = np.sum(motion_energies)

    return embeddings, motion_energies, total_motion_energy

def process_batch_files(video_files, model, device, chunk_size, args, output_dir):
    successful_in_batch = 0
    failed_in_batch = 0
    skipped_in_batch = 0
    
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        output_name = os.path.splitext(video_name)[0] + '.h5'
        output_path = os.path.join(output_dir, output_name)

        print(f"  Processing: {video_name}")

        if os.path.exists(output_path):
            print(f"  -> Skipped (already exists)")
            skipped_in_batch += 1
            continue

        try:
            start_time = time.time()
            embeddings, motion_energies, total_motion_energy = process_video_to_embeddings(
                video_path,
                model,
                device,
                chunk_size,
                fps=args.fps,
                max_duration=args.max_duration,
                target_size=(args.size, args.size)
            )

            if embeddings is None:
                print(f"  -> Failed to load")
                failed_in_batch += 1
                continue

            with h5py.File(output_path, 'w') as f:
                f.create_dataset('embeddings', data=embeddings, compression='gzip', compression_opts=4)
                f.create_dataset('motion_energy', data=motion_energies, compression='gzip', compression_opts=4)
                f.create_dataset('total_motion_energy', data=total_motion_energy)
                f.attrs['video_name'] = video_name
                f.attrs['fps'] = args.fps
                f.attrs['resolution'] = args.size
                f.attrs['num_chunks'] = embeddings.shape[0]
                f.attrs['embedding_shape'] = str(embeddings.shape)
            
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"  -> Success ({embeddings.shape[0]} chunks) in {processing_time:.2f}s")
            successful_in_batch += 1

        except Exception as e:
            print(f"  -> Error: {e}")
            failed_in_batch += 1
            continue
            
    return successful_in_batch, failed_in_batch, skipped_in_batch

def main():
    parser = argparse.ArgumentParser(description='Preprocess videos to h5 files with VideoVAE+ embeddings')
    parser.add_argument('output_dir', type=str, help='Output directory for h5 files')
    parser.add_argument('--input', type=str, help='Input video file or directory')
    parser.add_argument('--manifest_dir', type=str, default=None, help='Path to manifest directory. If provided, worker mode is activated.')
    parser.add_argument('--fps', type=int, default=8, help='Target FPS (default: 8)')
    parser.add_argument('--chunk-size', type=int, default=None,
                        help='Number of frames per chunk. Defaults to model temporal downsampling factor.')
    parser.add_argument('--max-duration', type=int, default=None,
                        help='Maximum video duration in seconds (default: None = process all)')
    parser.add_argument('--size', type=int, default=512,
                        help='Target resolution (square, default: 512)')
    parser.add_argument('--batch', action='store_true',
                        help='Process all videos in input directory')

    args = parser.parse_args()
    
    # Validate arguments
    if args.manifest_dir is None and args.input is None:
        parser.error("Either --input or --manifest_dir must be provided.")

    if args.manifest_dir and args.input:
        parser.error("Cannot use --input and --manifest_dir simultaneously.")

    if args.manifest_dir and args.batch:
        print("Warning: --batch is ignored when --manifest_dir is used.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    successful = 0
    failed = 0
    skipped = 0
    total_processed_videos = 0

    # --- Model & Chunk Size ---
    model_config_path = 'vvae/configs/config_16z.yaml'

    # Load model config
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    # Determine chunk size
    chunk_size = args.chunk_size
    temporal_scale_factor = model_config['model']['params']['ppconfig']['temporal_scale_factor']

    if chunk_size is None:
        chunk_size = temporal_scale_factor
        print(f"Using default chunk size from config: {chunk_size} (temporal scale factor)")
    else:
        print(f"Using specified chunk size: {chunk_size}")

    if chunk_size % temporal_scale_factor != 0:
        print(f"Warning: chunk_size ({chunk_size}) is not divisible by the model's temporal scale factor ({temporal_scale_factor}). "
              f"This may lead to unexpected behavior or errors.")


    # Load model
    print("Loading VideoVAE+ model...")
    model = load_model(model_config_path, device=device)
    print("Model loaded!\n")

    if args.manifest_dir:
        # Worker mode
        print("Running in worker mode.")
        manifest_path = Path(args.manifest_dir)
        todo_path = manifest_path / 'todo'
        in_progress_path = manifest_path / 'in_progress'
        done_path = manifest_path / 'done'
        failed_path = manifest_path / 'failed'

        while True:
            # Find a batch to claim
            todo_files = sorted(list(todo_path.glob('*.txt')))
            if not todo_files:
                print("No more batches to process. Exiting.")
                break

            claimed_batch_path = None
            for batch_file in todo_files:
                try:
                    target_path = in_progress_path / batch_file.name
                    os.rename(batch_file, target_path)
                    claimed_batch_path = target_path
                    print(f"\nClaimed batch: {batch_file.name}")
                    break
                except FileNotFoundError:
                    # Another worker grabbed it first
                    continue

            if not claimed_batch_path:
                # All files were grabbed by others during the loop
                time.sleep(5)  # Wait before checking again
                continue

            # Process the claimed batch
            with open(claimed_batch_path, 'r') as f:
                video_files_in_batch = [line.strip() for line in f if line.strip()]

            successful_in_batch, failed_in_batch, skipped_in_batch = process_batch_files(video_files_in_batch, model, device, chunk_size, args, args.output_dir)

            # Aggregate counts
            successful += successful_in_batch
            failed += failed_in_batch
            skipped += skipped_in_batch
            total_processed_videos += len(video_files_in_batch)

            # Move batch to final state
            if failed_in_batch > 0:
                final_path = failed_path / claimed_batch_path.name
                print(f"Batch failed. Moving to: {final_path}")
                os.rename(claimed_batch_path, final_path)
            else:
                final_path = done_path / claimed_batch_path.name
                print(f"Batch completed successfully. Moving to: {final_path}")
                os.rename(claimed_batch_path, final_path)

    else:
        # Original single-threaded mode
        # Get list of videos to process
        if args.batch:
            if not os.path.isdir(args.input):
                print(f"Error: --batch specified but {args.input} is not a directory")
                return
            # Recursively find all video files in subdirectories
            video_files = []
            for root, dirs, files in os.walk(args.input):
                for f in files:
                    if f.endswith(('.mp4', '.avi', '.mov')) and not f.startswith('._'):
                        video_files.append(os.path.join(root, f))
            video_files = sorted(video_files)
            if not video_files:
                print(f"No video files found in {args.input}")
                return
        else:
            if not os.path.isfile(args.input):
                print(f"Error: {args.input} is not a file")
                return
            video_files = [args.input]

        print(f"Processing {len(video_files)} video(s)...\n")
        successful, failed, skipped = process_batch_files(video_files, model, device, chunk_size, args, args.output_dir)
        total_processed_videos = len(video_files)
        # Summary
        print(f"\n{'='*60}")
        print("PREPROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total videos: {total_processed_videos}")
        print(f"Successful: {successful}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {failed}")
        print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")

    # Save dataset info config
    if args.manifest_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        config_path = os.path.join(args.output_dir, f'dataset_info_{timestamp}_pid{pid}.json')
        print(f"\nWorker summary saved to: {config_path}")
    else:
        config_path = os.path.join(args.output_dir, 'dataset_info.json')

    # Dynamic embedding params
    latent_channels = model_config['model']['params']['ddconfig']['z_channels']
    spatial_compression_factor = 2 ** (len(model_config['model']['params']['ddconfig']['ch_mult']) - 1)
    latent_size = args.size // spatial_compression_factor
    output_frames_per_chunk = chunk_size // temporal_scale_factor

    output_shape_str = f"[1, {latent_channels}, {output_frames_per_chunk}, {latent_size}, {latent_size}]"
    h5_shape_str = f"(N_chunks, {latent_channels}, {output_frames_per_chunk}, {latent_size}, {latent_size})"

    dataset_config = {
        "preprocessing_info": {
            "model": "VideoVAE+ (sota-4-16z)",
            "config_file": model_config_path,
            "preprocessing_date": datetime.now().isoformat(),
            "total_videos_processed": successful,
            "total_videos_failed": failed,
            "input_directory": args.input if not args.manifest_dir else "N/A (worker mode)",
        },
        "video_parameters": {
            "fps": args.fps,
            "target_resolution": [args.size, args.size],
            "max_duration_seconds": args.max_duration,
            "chunk_size_frames": chunk_size,
            "chunk_duration_seconds": chunk_size / args.fps
        },
        "embedding_parameters": {
            "latent_channels": latent_channels,
            "temporal_compression_factor": temporal_scale_factor,
            "spatial_compression_factor": spatial_compression_factor,
            "embedding_shape_per_chunk": output_shape_str,
            "input_frames_per_chunk": chunk_size,
            "output_frames_per_chunk": output_frames_per_chunk
        },
        "compression_info": {
            "input_shape_per_chunk": f"[1, 3, {chunk_size}, {args.size}, {args.size}]",
            "output_shape_per_chunk": output_shape_str,
            "values_per_input_chunk": 3 * chunk_size * args.size * args.size,
            "values_per_output_chunk": latent_channels * output_frames_per_chunk * latent_size * latent_size
        },
        "h5_structure": {
            "datasets": {
                "embeddings": {
                    "description": "VideoVAE+ latent embeddings",
                    "shape": h5_shape_str,
                    "dtype": "float32",
                    "compression": "gzip (level 4)"
                },
                "motion_energy": {
                    "description": "Temporal velocity of latent states (z_t+1 - z_t), as a per-frame energy value.",
                    "shape": "(N_chunks * T_latent - 1,)",
                    "dtype": "float32",
                    "compression": "gzip (level 4)"
                },
                "total_motion_energy": {
                    "description": "Sum of all motion energy values for the video.",
                    "shape": "(1,)",
                    "dtype": "float32",
                    "compression": "None"
                }
            },
            "attributes": {
                "video_name": "Original video filename",
                "fps": "Target FPS",
                "resolution": "Target resolution (square)",
                "num_chunks": "Number of chunks processed",
                "embedding_shape": "Shape of embeddings array"
            }
        },
        "notes": [
            "Each h5 file corresponds to one video",
            f"Videos are chunked into segments of {chunk_size} frames",
            "Each chunk is encoded independently",
            f"Temporal compression: {chunk_size} frames -> {output_frames_per_chunk} latent frames ({temporal_scale_factor}x)",
            f"Spatial compression: {args.size}x{args.size} -> {latent_size}x{latent_size} ({spatial_compression_factor}x per dimension)",
            "No actions dataset included (embeddings only)"
        ]
    }

    with open(config_path, 'w') as f:
        json.dump(dataset_config, f, indent=2)

    print(f"\nDataset info saved to: {config_path}")


if __name__ == "__main__":
    main()