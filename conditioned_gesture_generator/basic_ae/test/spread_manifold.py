#!/usr/bin/env python3
"""
cosine_distribution_all.py

- Load *_actions.npy files from a directory
- Window them, encode with TinyTransformerAutoencoder
- Compute cosine similarities (off-diagonal only)
- Pool all values across all files
- Plot a single overall distribution (histogram + fitted Normal curve)
"""

import argparse, math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ACTION_DIM, ACTIONS_PER_BATCH
FEAT_DIM = ACTION_DIM
from conditioned_gesture_generator.basic_ae.transformer_ae import TinyTransformerAutoencoder


def load_model(checkpoint, model_dim, latent_dim, device):
    model = TinyTransformerAutoencoder(
        FEAT_DIM=FEAT_DIM,
        MODEL_DIM=model_dim,
        SEQ_LEN=ACTIONS_PER_BATCH,
        LATENT_DIM=latent_dim
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def window_data(file_path, stride):
    arr = np.load(file_path, mmap_mode="r")
    seqs = []
    max_start = arr.shape[0] - ACTIONS_PER_BATCH
    if max_start < 0:
        return []
    for start in range(0, max_start + 1, stride):
        seqs.append(np.array(arr[start:start + ACTIONS_PER_BATCH]))
    return seqs


def encode_sequences(model, sequences, device):
    Z = []
    with torch.no_grad():
        for block in sequences:
            x = torch.from_numpy(block).float().unsqueeze(0).to(device)
            z, _ = model(x)
            Z.append(z.squeeze(0).cpu().numpy())
    Z = np.stack(Z, axis=0)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    return Z


def get_offdiag_cosines(Z):
    if len(Z) < 2:
        return np.array([])
    S = Z @ Z.T
    iu = np.triu_indices(len(Z), k=1) 
    
    return S[iu]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="cosine_all_out")
    ap.add_argument("--stride", type=int, default=ACTIONS_PER_BATCH)
    ap.add_argument("--model_dim", type=int, default=256)
    ap.add_argument("--latent_dim", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    outdir = Path(args.save_dir); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_model(Path(args.checkpoint), args.model_dim, args.latent_dim, device)

    files = sorted(Path(args.data_dir).glob("*_actions.npy"))
    if not files:
        print(f"No *_actions.npy files found in {args.data_dir}")
        return

    all_vals = []
    min_val, min_pair = 1.0, None

    for fp in files:
        print(f"Processing {fp.name} ...")
        seqs = window_data(fp, stride=args.stride)
        if len(seqs) < 2:
            continue
        Z = encode_sequences(model, seqs, device)
        vals = get_offdiag_cosines(Z)
        if vals.size == 0:
            continue
        all_vals.append(vals)

        # track global minimum
        vmin = vals.min()
        if vmin < min_val:
            min_val = vmin
            min_pair = (fp.name, vmin)

    if not all_vals:
        print("No cosine values computed.")
        return

    all_vals = np.concatenate(all_vals)
    mu, sigma = all_vals.mean(), all_vals.std(ddof=1)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.hist(all_vals, bins=100, density=True, alpha=0.6, color="steelblue")

    # Fit Gaussian curve
    xs = np.linspace(all_vals.min(), all_vals.max(), 400)
    pdf = (1.0 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
    plt.plot(xs, pdf, "r-", linewidth=2)

    plt.title("Overall cosine similarity distribution (all files pooled)")
    plt.xlabel("cosine similarity")
    plt.ylabel("density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "cosine_distribution_all.png", dpi=180)
    plt.close()

    np.save(outdir / "cosine_values_all.npy", all_vals)

    print(f"Saved histogram + values to {outdir.resolve()}")
    print(f"mean={mu:.4f}, sigma={sigma:.4f}, global_min={min_val:.4f} from {min_pair}")


if __name__ == "__main__":
    main()
