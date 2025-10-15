"""
Experiment script to test chunked encoding/decoding with VideoVAE+.

Tests:
1. Encoding/Decoding time with frame chunking
2. Shape transformations at each stage
3. Video reconstruction quality with chunking
"""

import torch
import time
import sys
import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from vvae.utils.common_utils import instantiate_from_config
from vvae.video_utils import load_video_cfr, save_video
import numpy as np
import psutil

# Add vvae directory to path so config imports work
vvae_dir = Path(__file__).parent
if str(vvae_dir) not in sys.path:
    sys.path.insert(0, str(vvae_dir))


def get_memory_usage():
    """Get current RAM usage in GB."""
    process = psutil.Process()
    ram_gb = process.memory_info().rss / (1024 ** 3)
    return ram_gb


def load_model(config_path='vvae/configs/config_16z.yaml', device='cuda'):
    """Load the VideoVAE+ model from config."""
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    model = model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


def chunk_video(video_tensor, chunk_size):
    """
    Chunk video tensor along temporal dimension for batch processing.

    Transforms [B, C, T, H, W] -> [B*num_chunks, C, chunk_size, H, W]

    Args:
        video_tensor: Input video tensor [B, C, T, H, W]
        chunk_size: Number of frames per chunk

    Returns:
        chunked: Chunked tensor [B*num_chunks, C, chunk_size, H, W]
        metadata: Dict with chunking info (B, num_chunks, padding_needed, original_T)
    """
    B, C, T, H, W = video_tensor.shape

    # Calculate padding
    num_chunks = T // chunk_size
    remaining_frames = T % chunk_size
    padding_needed = 0
    original_T = T

    if remaining_frames != 0:
        padding_needed = chunk_size - remaining_frames
        padding = video_tensor[:, :, -1:, :, :].repeat(1, 1, padding_needed, 1, 1)
        video_tensor = torch.cat([video_tensor, padding], dim=2)
        T = video_tensor.shape[2]
        num_chunks = T // chunk_size

    # Reshape: [B, C, T, H, W] -> [B*num_chunks, C, chunk_size, H, W]
    chunked = video_tensor.reshape(B, C, num_chunks, chunk_size, H, W)
    chunked = chunked.permute(0, 2, 1, 3, 4, 5)  # [B, num_chunks, C, chunk_size, H, W]
    chunked = chunked.reshape(B * num_chunks, C, chunk_size, H, W)

    metadata = {
        'B': B,
        'num_chunks': num_chunks,
        'padding_needed': padding_needed,
        'original_T': original_T
    }

    return chunked, metadata


def unchunk_video(chunked_tensor, metadata):
    """
    Reshape chunked video back to original format.

    Transforms [B*num_chunks, C, chunk_T, H, W] -> [B, C, T, H, W]

    Args:
        chunked_tensor: Chunked tensor [B*num_chunks, C, chunk_T, H, W]
        metadata: Dict from chunk_video with chunking info

    Returns:
        video_tensor: Unchunked tensor [B, C, T, H, W]
    """
    B = metadata['B']
    num_chunks = metadata['num_chunks']
    padding_needed = metadata['padding_needed']

    recon_B, recon_C, recon_T, recon_H, recon_W = chunked_tensor.shape

    # [B*num_chunks, C, chunk_T, H, W] -> [B, C, T, H, W]
    video = chunked_tensor.reshape(B, num_chunks, recon_C, recon_T, recon_H, recon_W)
    video = video.permute(0, 2, 1, 3, 4, 5)  # [B, C, num_chunks, chunk_T, H, W]
    video = video.reshape(B, recon_C, num_chunks * recon_T, recon_H, recon_W)

    # Remove padding if it was added
    if padding_needed != 0:
        original_T = metadata['original_T']
        video = video[:, :, :original_T, :, :]

    return video


def main():
    parser = argparse.ArgumentParser(description='Video VAE experiment with chunked encoding/decoding')
    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('--config', type=str, default='vvae/configs/config_16z.yaml',
                        help='Path to model config file (default: config_16z.yaml)')
    parser.add_argument('--chunks', type=str, default='all',
                        help='Number of chunks to process (1-N or "all", default: all)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Number of chunks to process in parallel (default: 1, try 4-8 for 32GB VRAM)')
    parser.add_argument('--fps', type=int, default=8, help='Target FPS (default: 8)')
    parser.add_argument('--max-duration', type=int, default=12,
                        help='Maximum video duration in seconds (default: 12)')
    parser.add_argument('--size', type=int, default=512,
                        help='Target resolution (square, default: 512)')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Configuration
    video_path = args.video_path
    target_fps = args.fps
    chunk_size = target_fps  # 1 second chunks
    max_duration = args.max_duration  # seconds
    target_size = (args.size, args.size)  # (height, width)

    print("="*60)
    print("EXPERIMENT: Chunked Video Encoding/Decoding")
    print("="*60)

    # Initial RAM usage
    initial_ram = get_memory_usage()
    print(f"\nInitial RAM usage: {initial_ram:.2f} GB")

    # Load video
    print(f"\n[1] Loading video (max {max_duration}s, resized to {target_size[0]}x{target_size[1]})...")
    video_array = load_video_cfr(video_path, target_fps=target_fps, max_duration=max_duration, target_size=target_size)
    after_load_ram = get_memory_usage()
    print(f"    Loaded video shape: {video_array.shape}")  # (N, H, W, 3)
    print(f"    RAM usage: {after_load_ram:.2f} GB (Δ {after_load_ram - initial_ram:.2f} GB)")

    # Convert to tensor and normalize
    # video_array is (N, H, W, 3), need (B, C, T, H, W)
    video_tensor = torch.from_numpy(video_array).float()
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (3, T, H, W)
    video_tensor = video_tensor.unsqueeze(0)  # (1, 3, T, H, W)
    video_tensor = (video_tensor / 255.0 - 0.5) * 2  # Normalize to [-1, 1]

    B, C, T, H, W = video_tensor.shape
    print(f"    Input tensor shape: {video_tensor.shape}")
    print(f"    B={B}, C={C}, T={T}, H={H}, W={W}")

    # Chunk video
    print(f"\n[2] Chunking video...")
    chunked, chunk_metadata = chunk_video(video_tensor, chunk_size)
    after_chunk_ram = get_memory_usage()

    if chunk_metadata['padding_needed'] > 0:
        print(f"    Padded {chunk_metadata['padding_needed']} frames")

    total_chunks = chunk_metadata['num_chunks']
    print(f"    Total chunks available: {total_chunks} x {chunk_size} frames")
    print(f"    RAM usage: {after_chunk_ram:.2f} GB (Δ {after_chunk_ram - after_load_ram:.2f} GB)")

    # Determine how many chunks to process
    if args.chunks.lower() == 'all':
        num_chunks_to_process = total_chunks
    else:
        num_chunks_to_process = min(int(args.chunks), total_chunks)

    batch_size = args.batch_size
    print(f"    Processing {num_chunks_to_process} out of {total_chunks} chunks")
    print(f"    Batch size: {batch_size} chunks")
    print(f"    Chunked input shape: {chunked.shape}")
    print(f"    [B*Frame_Block, C, fps, H, W] = [{chunked.shape[0]}, {C}, {chunk_size}, {H}, {W}]")

    # Load model
    print(f"\n[3] Loading model from {args.config}...")
    model = load_model(args.config, device=device)
    after_model_ram = get_memory_usage()
    print(f"    Model loaded successfully!")
    print(f"    RAM usage: {after_model_ram:.2f} GB (Δ {after_model_ram - after_chunk_ram:.2f} GB)")

    # Process chunks in batches
    mode_str = "in batches" if batch_size > 1 else "sequentially"
    print(f"\n[4-5] Processing {num_chunks_to_process} chunks {mode_str}...")
    reconstructed_chunks = []
    total_encoding_time = 0
    total_decoding_time = 0

    # Track VRAM usage
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        initial_vram = torch.cuda.memory_allocated() / (1024 ** 3)

    with torch.no_grad():
        for batch_start in range(0, num_chunks_to_process, batch_size):
            batch_end = min(batch_start + batch_size, num_chunks_to_process)
            batch_actual_size = batch_end - batch_start

            if batch_size > 1:
                print(f"    Processing batch {batch_start+1}-{batch_end}/{num_chunks_to_process}...", end=' ')
            else:
                print(f"    Processing chunk {batch_start+1}/{num_chunks_to_process}...", end=' ')

            # Get batch of chunks [batch_size, C, chunk_size, H, W]
            chunk_batch = chunked[batch_start:batch_end].to(device)

            # Encode
            start_time = time.time()
            latent, posterior = model.encode(chunk_batch, sample_posterior=False)
            encoding_time = time.time() - start_time
            total_encoding_time += encoding_time

            # Print latent shape on first batch
            if batch_start == 0:
                print(f"\n    Latent shape: {latent.shape} (compressed from {chunk_batch.shape})")
                print(f"    ", end='')

            # Decode
            start_time = time.time()
            reconstructed_batch = model.decode(latent)
            decoding_time = time.time() - start_time
            total_decoding_time += decoding_time

            # Store on CPU to save GPU memory
            reconstructed_chunks.append(reconstructed_batch.cpu())

            # Report timing and VRAM
            avg_enc = encoding_time / batch_actual_size
            avg_dec = decoding_time / batch_actual_size
            print(f"✓ (enc: {encoding_time:.3f}s, dec: {decoding_time:.3f}s", end='')
            if batch_size > 1:
                print(f", avg/chunk: {avg_enc+avg_dec:.3f}s", end='')
            if device == 'cuda':
                current_vram = torch.cuda.memory_allocated() / (1024 ** 3)
                peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f", VRAM: {current_vram:.2f}GB, peak: {peak_vram:.2f}GB", end='')
            print(")")

            # Clear GPU cache
            if device == 'cuda':
                torch.cuda.empty_cache()

    # Concatenate all reconstructed chunks
    reconstructed_chunked = torch.cat(reconstructed_chunks, dim=0)
    after_processing_ram = get_memory_usage()

    total_time = total_encoding_time + total_decoding_time
    print(f"\n    ✓ All chunks processed!")
    print(f"    Reconstructed chunked shape: {reconstructed_chunked.shape}")
    print(f"    TOTAL TIME: {total_time:.4f} seconds")
    print(f"    (Encoding: {total_encoding_time:.4f}s, Decoding: {total_decoding_time:.4f}s)")
    print(f"    RAM usage: {after_processing_ram:.2f} GB (Δ {after_processing_ram - after_model_ram:.2f} GB)")

    # Update metadata for partial processing
    if num_chunks_to_process < total_chunks:
        chunk_metadata['num_chunks'] = num_chunks_to_process
        chunk_metadata['original_T'] = num_chunks_to_process * chunk_size
        if chunk_metadata['padding_needed'] > 0 and num_chunks_to_process == total_chunks:
            chunk_metadata['original_T'] -= chunk_metadata['padding_needed']
        else:
            chunk_metadata['padding_needed'] = 0

    # Reshape back to full video
    print(f"\n[6] Unchunking video...")
    reconstructed = unchunk_video(reconstructed_chunked, chunk_metadata)

    print(f"    Unchunked output: {reconstructed.shape}")
    print(f"    Expected: [{chunk_metadata['B']}, {C}, {chunk_metadata['original_T']}, {H}, {W}]")

    # Get original video matching the reconstructed portion
    # Slice to match the number of frames we actually processed
    original_video = video_tensor[:, :, :chunk_metadata['original_T'], :, :]

    if reconstructed.shape == original_video.shape:
        print(f"    ✓ Shape matches original input!")
    else:
        print(f"    ✗ WARNING: Shape mismatch!")
        print(f"       Got:      {reconstructed.shape}")
        print(f"       Expected: {original_video.shape}")

    # Save reconstructed video
    output_dir = 'reconstructed'
    os.makedirs(output_dir, exist_ok=True)
    video_filename = os.path.basename(video_path)
    output_filename = video_filename.replace('.mp4', '_reconstructed.mp4')
    output_path = os.path.join(output_dir, output_filename)
    print(f"\n[7] Saving reconstructed video to: {output_path}")
    save_video(reconstructed, output_path, fps=target_fps)

    # Calculate reconstruction error
    mse = torch.mean((original_video.to(device) - reconstructed.to(device)) ** 2).item()
    print(f"\n[8] Reconstruction MSE: {mse:.6f}")

    # Summary
    final_ram = get_memory_usage()
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Input shape:              {original_video.shape}")
    print(f"Chunks processed:         {num_chunks_to_process}/{total_chunks}")
    print(f"Batch size:               {batch_size}")
    print(f"Reconstructed chunks:     {reconstructed_chunked.shape}")
    print(f"Final output shape:       {reconstructed.shape}")
    print(f"\nTiming:")
    print(f"  Encoding:  {total_encoding_time:.4f}s")
    print(f"  Decoding:  {total_decoding_time:.4f}s")
    print(f"  Total:     {total_time:.4f}s")
    print(f"  Avg per chunk: {total_time/num_chunks_to_process:.4f}s")
    print(f"\nMemory usage:")
    print(f"  Initial:       {initial_ram:.2f} GB")
    print(f"  After load:    {after_load_ram:.2f} GB (Δ {after_load_ram - initial_ram:.2f} GB)")
    print(f"  After chunk:   {after_chunk_ram:.2f} GB (Δ {after_chunk_ram - after_load_ram:.2f} GB)")
    print(f"  After model:   {after_model_ram:.2f} GB (Δ {after_model_ram - after_chunk_ram:.2f} GB)")
    print(f"  After process: {after_processing_ram:.2f} GB (Δ {after_processing_ram - after_model_ram:.2f} GB)")
    print(f"  Final:         {final_ram:.2f} GB")
    print(f"  Peak delta:    {final_ram - initial_ram:.2f} GB")
    print(f"\nReconstruction MSE: {mse:.6f}")
    print(f"Output saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
