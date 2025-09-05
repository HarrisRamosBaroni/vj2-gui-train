#!/usr/bin/env python3
"""
plot_reconstructions.py
Repurpose of the training-time plotting: loads a checkpointed TinyTransformerAutoencoder,
runs it over a directory of *_actions.npy files, and saves original vs reconstructed plots.

Example:
python plot_reconstructions.py \
  --data_dir ../downloads/dense_action_train_val \
  --checkpoint checkpoints/action_ae_run_20250101_120000/best.pt \
  --output_dir recon_plots \
  --stride 128 \
  --max_dims 6 \
  --samples_per_file 12
"""

import argparse
import os
from pathlib import Path
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt
import csv

# --- Project-specific imports you already have ---
from config import ACTION_DIM, ACTIONS_PER_BATCH
FEAT_DIM = ACTION_DIM
from conditioned_gesture_generator.basic_ae.transformer_ae import TinyTransformerAutoencoder  # sphere_loss/total_loss not needed here

# --------------------------- Dataset ---------------------------

class ActionBlockDataset:
    """
    Minimal dataset for sliding windows over *_actions.npy files.
    Yields (file_path, start_idx, action_block_tensor).
    """
    def __init__(self, data_dir, stride=ACTIONS_PER_BATCH):
        self.data_dir = Path(data_dir)
        self.stride = stride
        self.file_paths = sorted(self.data_dir.glob("*_actions.npy"))
        if not self.file_paths:
            raise ValueError(f"No '*_actions.npy' files found in {self.data_dir}")

        # Build (file_idx, start_idx) index
        self.index = []
        self.file_lengths = []
        for fidx, fp in enumerate(self.file_paths):
            arr = np.load(fp, mmap_mode="r")
            total = arr.shape[0]
            self.file_lengths.append(total)
            max_start = total - ACTIONS_PER_BATCH
            if max_start >= 0:
                starts = list(range(0, max_start + 1, self.stride))
                self.index.extend([(fidx, s) for s in starts])

    def __len__(self):
        return len(self.index)

    def __iter__(self):
        for fidx, start in self.index:
            fp = self.file_paths[fidx]
            arr = np.load(fp, mmap_mode="r")
            block_np = np.array(arr[start:start + ACTIONS_PER_BATCH])
            yield fp, start, torch.from_numpy(block_np).float()

# ------------------------ Plotting utils -----------------------

def _plot_action_seq_pair(orig: torch.Tensor, recon: torch.Tensor, title: str = "", max_dims: int = 6):
    """
    orig/recon: [SEQ_LEN, ACTION_DIM] (CPU tensors).
    Returns a matplotlib Figure overlaying orig vs recon for up to max_dims dims.
    """
    orig = orig.detach().cpu().numpy()
    recon = recon.detach().cpu().numpy()
    seq_len, action_dim = orig.shape
    dims = min(action_dim, max_dims)

    fig, axes = plt.subplots(dims, 1, figsize=(10, 1.6 * dims), sharex=True)
    if dims == 1:
        axes = [axes]

    t = np.arange(seq_len)
    for d, ax in enumerate(axes):
        ax.plot(t, orig[:, d], linewidth=1.5, label="original")
        ax.plot(t, recon[:, d], linewidth=1.2, linestyle="--", label="reconstructed")
        ax.set_ylabel(f"dim {d}")
        ax.grid(alpha=0.3)
    axes[0].set_title(title)
    axes[-1].set_xlabel("time step")
    axes[0].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    return fig

# --------------------------- Main logic ------------------------

def load_model(checkpoint_path: Path, model_dim: int, latent_dim: int, device: torch.device):
    model = TinyTransformerAutoencoder(
        FEAT_DIM=FEAT_DIM,
        MODEL_DIM=model_dim,
        SEQ_LEN=ACTIONS_PER_BATCH,
        LATENT_DIM=latent_dim
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    # Some checkpoints store under 'model_state_dict'; support both direct and dict forms.
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model

def plot_full_reconstruction_from_chunks(
    file_path: Path,
    model,
    device: torch.device,
    max_dims: int = 6,
    drop_last: bool = True,
    pad_remainder: bool = False,
    pad_mode: str = "edge",
):
    """
    Reconstructs a whole *_actions.npy file by chunking into windows of ACTIONS_PER_BATCH,
    feeding each chunk through the autoencoder, concatenating the reconstructions, and
    plotting original vs reconstructed over the full timeline.

    Args:
        file_path: path to the .npy file (shape [T, ACTION_DIM])
        model: trained TinyTransformerAutoencoder (SEQ_LEN == ACTIONS_PER_BATCH)
        device: 'cpu' or 'cuda'
        max_dims: number of action dimensions to plot
        drop_last: if True, ignore the remainder (< ACTIONS_PER_BATCH) at the end
        pad_remainder: if True and remainder exists, pad the last partial chunk up to
                       ACTIONS_PER_BATCH before reconstruction, then crop back
        pad_mode: numpy pad mode, e.g. 'edge', 'constant', 'reflect' (used only if pad_remainder)

    Returns:
        fig: matplotlib Figure
        recon_all: np.ndarray of reconstructed trajectory [T_used, ACTION_DIM]
        orig_all:  np.ndarray of original trajectory [T_used, ACTION_DIM]
    """
    data = np.load(file_path)  # [T, ACTION_DIM]
    if data.ndim != 2 or data.shape[1] != ACTION_DIM:
        raise ValueError(f"{file_path} must have shape [T, {ACTION_DIM}], got {data.shape}")

    T = data.shape[0]
    L = ACTIONS_PER_BATCH
    full_chunks = T // L
    remainder = T % L

    recon_pieces = []
    orig_pieces = []

    # Process full chunks
    with torch.no_grad():
        for i in range(full_chunks):
            start = i * L
            end = start + L
            chunk_np = data[start:end]  # [L, D]
            chunk = torch.from_numpy(chunk_np).float().unsqueeze(0).to(device)  # [1, L, D]
            _, recon = model(chunk)  # [1, L, D]
            recon_pieces.append(recon.squeeze(0).cpu().numpy())
            orig_pieces.append(chunk_np)

        # Handle remainder
        if remainder > 0:
            start = full_chunks * L
            tail_np = data[start:]  # [R, D]

            if pad_remainder:
                pad_len = L - remainder
                # pad at the end along time dimension
                if pad_mode == "constant":
                    last_row = np.zeros((1, tail_np.shape[1]), dtype=tail_np.dtype)
                else:
                    last_row = None  # not used, but kept for clarity

                padded = np.pad(
                    tail_np,
                    pad_width=((0, L - remainder), (0, 0)),
                    mode=pad_mode
                )  # -> [L, D]

                chunk = torch.from_numpy(padded).float().unsqueeze(0).to(device)
                _, recon = model(chunk)
                recon_tail = recon.squeeze(0).cpu().numpy()[:remainder]  # crop back to R
                recon_pieces.append(recon_tail)
                orig_pieces.append(tail_np)

            elif not drop_last:
                # If not dropping and not padding, we can't feed partial length to the model.
                # Default to padding with 'edge' to keep behavior sane.
                padded = np.pad(
                    tail_np,
                    pad_width=((0, L - remainder), (0, 0)),
                    mode="edge"
                )
                chunk = torch.from_numpy(padded).float().unsqueeze(0).to(device)
                _, recon = model(chunk)
                recon_tail = recon.squeeze(0).cpu().numpy()[:remainder]
                recon_pieces.append(recon_tail)
                orig_pieces.append(tail_np)
            # else: drop_last=True -> ignore tail

    # Concatenate
    if len(orig_pieces) == 0:
        raise ValueError("No usable data chunks (file shorter than ACTIONS_PER_BATCH and drop_last=True).")

    orig_all = np.concatenate(orig_pieces, axis=0)   # [T_used, D]
    recon_all = np.concatenate(recon_pieces, axis=0) # [T_used, D]
    T_used, D = orig_all.shape
    dims = min(D, max_dims)

    # Plot
    fig, axes = plt.subplots(dims, 1, figsize=(12, 1.6 * dims), sharex=True)
    if dims == 1:
        axes = [axes]
    t = np.arange(T_used)
    for d, ax in enumerate(axes):
        ax.plot(t, orig_all[:, d], linewidth=1.4, label="original")
        ax.plot(t, recon_all[:, d], linewidth=1.1, linestyle="--", label="reconstructed")
        ax.set_ylabel(f"dim {d}")
        ax.grid(alpha=0.3)
    axes[0].set_title(f"Full reconstruction (chunked) · {file_path.stem}")
    axes[-1].set_xlabel("time step")
    axes[0].legend(loc="upper right", frameon=False)
    fig.tight_layout()

    return fig, recon_all, orig_all



def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean((a - b) ** 2).item()

def main():
    parser = argparse.ArgumentParser(description="Plot original vs reconstructed action sequences.")
    parser.add_argument("--data_dir", required=True, type=str, help="Path with *_actions.npy files.")
    parser.add_argument("--load_checkpoint", required=True, type=str, help="Path to model checkpoint (.pt).")
    parser.add_argument("--output_dir", type=str, default="recon_plots", help="Where to save PNGs and CSV.")
    parser.add_argument("--stride", type=int, default=ACTIONS_PER_BATCH, help="Sliding window stride.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="'cuda' or 'cpu'. Default picks CUDA if available.")
    parser.add_argument("--model_dim", type=int, default=256, help="Transformer model dim (must match training).")
    parser.add_argument("--latent_dim", type=int, default=20, help="Latent dim (must match training).")
    parser.add_argument("--max_dims", type=int, default=6, help="Max action dims to plot per panel.")
    parser.add_argument("--samples_per_file", type=int, default=12,
                        help="Max number of windows to plot per file (evenly spaced). Use -1 for all.")
    parser.add_argument("--save_csv", action="store_true",
                        help="If set, writes a CSV with per-sample MSEs.")
    args = parser.parse_args()
    
    # import wandb 
    # wandb.init(project="autoencoder_recon", name="plot_og_recon")

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(Path(args.load_checkpoint), args.model_dim, args.latent_dim, device)

    # Build dataset index
    ds = ActionBlockDataset(args.data_dir, stride=args.stride)

    # To select evenly spaced samples per file, first group indices by file
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fig = plot_full_reconstruction_from_chunks(Path("./practnpy/20250729_181208_actions.npy"),
                                model, device, max_dims=6)
    fig[0].savefig("full_reconstruction.png", dpi=150)
    plt.close(fig[0])

    # by_file = {}
    # for idx, (fidx, start) in enumerate(ds.index):
    #     by_file.setdefault(fidx, []).append((idx, start))

    # csv_rows = []
    
    # total_plots = 0

    # with torch.no_grad():
    #     for fidx, idx_list in by_file.items():
    #         fp = ds.file_paths[fidx]
    #         # Determine which indices to plot for this file
    #         if args.samples_per_file == -1 or args.samples_per_file >= len(idx_list):
    #             chosen = idx_list
    #         else:
    #             # Even spacing
    #             n = len(idx_list)
    #             k = args.samples_per_file
    #             positions = np.linspace(0, n - 1, num=k, dtype=int)
    #             chosen = [idx_list[p] for p in positions]

    #         # Process chosen windows
    #         images = []
    #         for rank, (global_idx, start) in enumerate(chosen):
    #             # Fetch the sample (reuse dataset iteration logic directly)
    #             arr = np.load(fp, mmap_mode="r")
    #             block_np = np.array(arr[start:start + ACTIONS_PER_BATCH])
    #             batch = torch.from_numpy(block_np).float().to(device)  # [SEQ_LEN, ACTION_DIM]
    #             batch = batch.unsqueeze(0)  # [1, SEQ_LEN, ACTION_DIM]

    #             # Forward pass
    #             latent, recon = model(batch)  # shapes: [1, latent_dim], [1, SEQ_LEN, ACTION_DIM]

    #             # Compute MSE for log
    #             cur_mse = mse(recon.squeeze(0), batch.squeeze(0))

    #             # Plot and save
    #             fig = _plot_action_seq_pair(
    #                 orig=batch.squeeze(0).cpu(),
    #                 recon=recon.squeeze(0).cpu(),
    #                 title=f"{fp.stem} | start={start} | MSE={cur_mse:.6f}",
    #                 max_dims=args.max_dims
    #             )
    #             # File name pattern
    #             safe_stem = fp.stem
    #             out_path = out_dir / f"{safe_stem}_start{start:07d}_idx{global_idx:07d}.png"
    #             fig.savefig(out_path, dpi=150, bbox_inches="tight")
    #             images.append(wandb.Image(fig))
    #             plt.close(fig)
    #             total_plots += 1

    #             if args.save_csv:
    #                 csv_rows.append({
    #                     "file": str(fp),
    #                     "start_index": start,
    #                     "global_index": global_idx,
    #                     "mse": f"{cur_mse:.8f}"
    #                 })
                
                    
    #         wandb.log({f"recon": images})

    # if args.save_csv:
    #     csv_path = out_dir / "reconstruction_metrics.csv"
    #     with open(csv_path, "w", newline="") as f:
    #         writer = csv.DictWriter(f, fieldnames=["file", "start_index", "global_index", "mse"])
    #         writer.writeheader()
    #         writer.writerows(csv_rows)

    # print(f"Saved {total_plots} plot(s) to: {out_dir}")
    # if args.save_csv:
    #     print(f"Wrote CSV metrics to: {csv_path}")

if __name__ == "__main__":
    main()
