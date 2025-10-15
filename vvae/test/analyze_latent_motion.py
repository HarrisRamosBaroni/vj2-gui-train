"""
Analyzes the motion content of a dataset of VVAE latent embeddings.

This script processes a directory of .h5 files, each containing video embeddings,
to quantify the amount of "stillness" or low-motion segments.

It performs the following steps:
1.  Loads latent embeddings from all .h5 files in a given directory.
2.  Calculates the temporal velocity of latent states (z_{t+1} - z_t).
3.  Computes the magnitude (Euclidean norm) of these velocity vectors, referred
    to as "motion energy."
4.  Generates and logs plots to visualize the distribution of this motion energy:
    - A global histogram of motion energy across the entire dataset.
    - A Cumulative Distribution Function (CDF) to aid in threshold selection.
    - Time-series plots of motion energy for individual example videos.

Results can be saved locally and/or logged to Weights & Biases (wandb).
"""
import argparse
import glob
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path
import wandb

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze motion in VVAE latent embeddings.")
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Directory containing the .h5 embedding files.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optional: Directory to save plot images locally.")
    parser.add_argument("--wandb-project", type=str, default="latent-motion-analysis",
                        help="Optional: Weights & Biases project name to log results.")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Optional: Name for the wandb run.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for computation (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--num-examples", type=int, default=3,
                        help="Number of example videos to plot time series for.")
    return parser.parse_args()

def init_wandb_run(project, name, config):
    """Initializes a wandb run."""
    if not project:
        return None
    run_name = name or f"motion_analysis_{Path(config.dataset_dir).name}"
    return wandb.init(project=project, name=run_name, config=config)

def get_motion_energies(h5_path, device):
    """
    Loads embeddings from an h5 file and computes motion energies.

    Returns:
        A numpy array of motion energy values for one video.
    """
    with h5py.File(h5_path, 'r') as f:
        if 'embeddings' not in f:
            return np.array([])
        # embeddings = f['embeddings'][:]
        motion_energies = f['motion_energy'][:]

    # # Reshape from (N_chunks, C, T_latent, H, W) to a single time series
    # # (N_chunks * T_latent, C, H, W)
    # num_chunks, C, T_latent, H, W = embeddings.shape
    # if num_chunks == 0:
    #     return np.array([])
    
    # latents = torch.from_numpy(embeddings).view(num_chunks * T_latent, C, H, W)
    # latents = latents.to(device)

    # # Calculate velocity and then motion energy (magnitude)
    # with torch.no_grad():
    #     velocities = latents[1:] - latents[:-1]
    #     # Calculate norm over all dimensions except time
    #     motion_energies = torch.norm(velocities.view(velocities.shape[0], -1), dim=1)

    # return motion_energies.cpu().numpy()
    return motion_energies # .cpu().numpy()

def plot_histogram(energies, output_path=None, run=None):
    """Plots and logs the global motion energy histogram."""
    plt.figure(figsize=(12, 7))
    plt.hist(energies, bins=100, alpha=0.75, color='royalblue')
    plt.yscale('log')
    plt.title('Global Motion Energy Distribution (Log Scale)')
    plt.xlabel('Motion Energy (L2 Norm of z_t+1 - z_t)')
    plt.ylabel('Frequency (Log)')
    plt.grid(True, which="both", ls="--")
    
    if output_path:
        plt.savefig(output_path)
    if run:
        run.log({"1_global_motion_histogram": wandb.Image(plt)})
    plt.close()

def plot_cdf(energies, output_path=None, run=None):
    """Plots and logs the motion energy CDF."""
    plt.figure(figsize=(12, 7))
    sorted_energies = np.sort(energies)
    yvals = np.arange(len(sorted_energies)) / float(len(sorted_energies) - 1)
    plt.plot(sorted_energies, yvals)
    plt.title('Cumulative Distribution Function (CDF) of Motion Energy')
    plt.xlabel('Motion Energy')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    
    # Add percentile lines
    percentiles = [10, 25, 50, 75, 90]
    p_values = np.percentile(energies, percentiles)
    for p, val in zip(percentiles, p_values):
        plt.axvline(x=val, linestyle='--', alpha=0.6, label=f'{p}th percentile ({val:.4f})')
    plt.legend()

    if output_path:
        plt.savefig(output_path)
    if run:
        run.log({"2_motion_energy_cdf": wandb.Image(plt)})
    plt.close()

def plot_time_series(energies, video_name, output_path=None, run=None):
    """Plots and logs motion energy time series for one video."""
    plt.figure(figsize=(15, 5))
    plt.plot(energies)
    plt.title(f'Motion Energy over Time for: {video_name}')
    plt.xlabel('Latent Frame Index')
    plt.ylabel('Motion Energy')
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
    if run:
        run.log({f"3_timeseries/{video_name}": wandb.Image(plt)})
    plt.close()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run = init_wandb_run(args.wandb_project, args.wandb_run_name, args)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(glob.glob(str(args.dataset_dir / "*.h5")))
    if not h5_files:
        print(f"Error: No .h5 files found in {args.dataset_dir}")
        return

    print(f"Found {len(h5_files)} .h5 files to analyze.")

    all_motion_energies = []
    for i, h5_path in enumerate(h5_files):
        video_name = Path(h5_path).stem
        print(f"[{i+1}/{len(h5_files)}] Processing {video_name}...")
        
        energies = get_motion_energies(h5_path, device)
        if energies.size == 0:
            print(f"  -> Skipped (no embeddings found or empty).")
            continue
            
        all_motion_energies.append(energies)

        # Plot time series for the first few examples
        if i < args.num_examples:
            path = args.output_dir / f"timeseries_{video_name}.png" if args.output_dir else None
            plot_time_series(energies, video_name, path, run)

    if not all_motion_energies:
        print("No valid embedding data found to analyze.")
        if run: run.finish()
        return

    global_energies = np.concatenate(all_motion_energies)
    print(f"Analyzed {len(global_energies)} total latent frame transitions.")

    # Generate and save/log global plots
    print("Generating global plots...")
    hist_path = args.output_dir / "global_motion_histogram.png" if args.output_dir else None
    plot_histogram(global_energies, hist_path, run)

    cdf_path = args.output_dir / "motion_energy_cdf.png" if args.output_dir else None
    plot_cdf(global_energies, cdf_path, run)
    
    # Print summary statistics
    print("--- Motion Energy Statistics ---")
    percentiles = [10, 25, 50, 75, 90, 99]
    p_values = np.percentile(global_energies, percentiles)
    for p, val in zip(percentiles, p_values):
        print(f"{p:2}th percentile: {val:.6f}")
    
    # Suggest a threshold
    threshold = p_values[1] # 25th percentile
    still_frames = np.sum(global_energies < threshold)
    still_percentage = (still_frames / len(global_energies)) * 100
    print(f"Based on the 25th percentile threshold ({threshold:.6f}), "
          f"approximately {still_percentage:.2f}% of the dataset can be "
          "considered 'low-motion' or 'still'.")
    print("---")

    if run:
        # Log summary stats to wandb
        summary = {f"percentile_{p}": v for p, v in zip(percentiles, p_values)}
        summary["stillness_threshold_p25"] = threshold
        summary["stillness_percentage_p25"] = still_percentage
        run.summary.update(summary)
        run.finish()

    print("Analysis complete.")
    if args.output_dir:
        print(f"Plots saved to {args.output_dir}")
    if args.wandb_project:
        print(f"Results logged to wandb project: {args.wandb_project}")

if __name__ == "__main__":
    main()
