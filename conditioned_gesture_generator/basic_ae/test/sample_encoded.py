

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# project specifics
from config import ACTION_DIM, ACTIONS_PER_BATCH
FEAT_DIM = ACTION_DIM
from conditioned_gesture_generator.basic_ae.transformer_ae import TinyTransformerAutoencoder


# -------------------- Model utils --------------------

def load_model(checkpoint: Path, model_dim: int, latent_dim: int, device: torch.device):
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

@torch.no_grad()
def decode_from_latent(model, z: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Try common decode APIs:
      - model.decode(z)
      - model.decoder(z)
      - model.decode(z, seq_len)
    Returns: reconstructed sequence [B, SEQ_LEN, ACTION_DIM] on CPU.
    """
    z = z.to(device)
    # latent = model.to_latent(z[:, 0, :])  # Grab [CLS]

    # Decode
    decoded_seed = model.latent_to_model(z).unsqueeze(1).repeat(1, 250, 1)
    decoded = model.decoder(decoded_seed + model.pos_embed)
    output = model.output_proj(decoded)
    return output


# -------------------- Sampling --------------------

def sample_uniform_sphere(n: int, d: int, device: torch.device) -> torch.Tensor:
    z = torch.randn(n, d, device=device)
    z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
    return z

def sample_gaussian(n: int, d: int, device: torch.device) -> torch.Tensor:
    # effectively same as uniform on sphere after normalization
    return sample_uniform_sphere(n, d, device)

def window_data(file_path: Path, stride: int):
    arr = np.load(file_path, mmap_mode="r")
    seqs = []
    max_start = arr.shape[0] - ACTIONS_PER_BATCH
    if max_start < 0:
        return []
    for start in range(0, max_start + 1, stride):
        seqs.append(np.array(arr[start:start + ACTIONS_PER_BATCH]))
    return seqs

@torch.no_grad()
def estimate_latent_center(model, sequences, device: torch.device, max_windows: int = 256) -> torch.Tensor:
    """
    Encode up to max_windows sequences, average their unit-normalized latents,
    then renormalize to the sphere. Gives a 'typical' direction on the manifold.
    """
    take = min(len(sequences), max_windows)
    Z = []
    for i in range(take):
        x = torch.from_numpy(sequences[i]).float().unsqueeze(0).to(device)
        z, _ = model(x)
        z = z.squeeze(0)
        z = z / (z.norm() + 1e-8)
        Z.append(z)
    if not Z:
        raise ValueError("No sequences available to estimate latent center.")
    m = torch.stack(Z, dim=0).mean(dim=0)
    m = m / (m.norm() + 1e-8)
    return m  # [latent_dim]

def sample_nearby(center: torch.Tensor, n: int, eps: float, device: torch.device) -> torch.Tensor:
    """
    Sample around 'center' on the sphere: center + eps * noise, then renormalize.
    eps ~ 0.1 gives small local variations.
    """
    d = center.numel()
    noise = torch.randn(n, d, device=device)
    z = center.unsqueeze(0) + eps * noise
    z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
    return z


# -------------------- Plotting --------------------

def plot_sequences(batch_seq: torch.Tensor, out_path: Path, max_dims: int = 6):
    """
    batch_seq: [B, SEQ_LEN, ACTION_DIM] CPU tensor
    Save a grid of line plots.
    """
    B = batch_seq.shape[0]
    cols = min(4, B)
    rows = (B + cols - 1) // cols
    plt.figure(figsize=(4*cols, 2.6*rows))
    for i in range(B):
        plt.subplot(rows, cols, i+1)
        seq = batch_seq[i].numpy()
        t = np.arange(seq.shape[0])
        for d in range(min(seq.shape[1], max_dims)):
            plt.plot(t, seq[:, d], linewidth=1)
        plt.title(f"sample {i}")
        plt.xlabel("t")
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="synth_out")
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--sampling", type=str, default="uniform_sphere",
                    choices=["uniform_sphere", "gaussian", "nearby"])
    ap.add_argument("--nearby_eps", type=float, default=0.1,
                    help="Noise scale for 'nearby' sampling on the sphere")
    ap.add_argument("--data_file", type=str, default=None,
                    help="Optional *_actions.npy, only needed for 'nearby' center estimation")
    ap.add_argument("--stride", type=int, default=ACTIONS_PER_BATCH)
    ap.add_argument("--model_dim", type=int, default=256)
    ap.add_argument("--latent_dim", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    outdir = Path(args.save_dir); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load model
    model = load_model(Path(args.checkpoint), args.model_dim, args.latent_dim, device)

    # Sample latents
    if args.sampling in ("uniform_sphere", "gaussian"):
        sampler = sample_uniform_sphere if args.sampling == "uniform_sphere" else sample_gaussian
        z = sampler(args.num_samples, args.latent_dim, device)
        print(z.shape)
    else:
        if args.data_file is None:
            raise ValueError("Provide --data_file for 'nearby' sampling.")
        seqs = window_data(Path(args.data_file), stride=args.stride)
        if len(seqs) == 0:
            raise ValueError("No windows from data_file to estimate center.")
        center = estimate_latent_center(model, seqs, device)
        z = sample_nearby(center, args.num_samples, eps=args.nearby_eps, device=device)

    # Decode and plot
    recon = decode_from_latent(model, z, device)  # [B, SEQ_LEN, ACTION_DIM] on CPU
    np.save(outdir / "synthetic_actions.npy", recon.numpy())
    plot_sequences(recon, outdir / "synthetic_actions.png")

    print(f"Saved {recon.shape[0]} decoded samples to {outdir.resolve()}/synthetic_actions.npy")
    print(f"Preview plot written to {outdir.resolve()}/synthetic_actions.png")

if __name__ == "__main__":
    main()
