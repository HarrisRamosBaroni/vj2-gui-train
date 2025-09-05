#!/usr/bin/env python3
"""
encode_and_save_reconstructions.py

- Load *_actions.npy files from a directory
- Window each into [ACTIONS_PER_BATCH, ACTION_DIM] with --stride
- Run model forward: z, recon = model(x)
- Save ONLY the decoded outputs (reconstructions) per file:
    <stem>_recon.npy   -> [N_windows, ACTIONS_PER_BATCH, ACTION_DIM] (float32, memmap-backed)
    <stem>_starts.npy  -> [N_windows] start indices
"""

import argparse
from pathlib import Path
import numpy as np
import torch

from config import ACTION_DIM, ACTIONS_PER_BATCH
FEAT_DIM = ACTION_DIM
from conditioned_gesture_generator.basic_ae.transformer_ae import TinyTransformerAutoencoder


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

def ema_filter(action_sequence, alpha=0.0, window_size_pct=0.1):
    """
    Applies differentiable exponential moving average filtering to action sequences with windowed memory.
    
    Args:
        action_sequence: [B, SEQ_LEN, ACTION_DIM] tensor of action sequences
        alpha: float, smoothing factor (0.0 = no smoothing, 0 < alpha < 1). Higher alpha = more smoothing
        window_size_pct: float, window size as percentage of sequence length (default 0.1 = 10%)
        
    Returns:
        Filtered action sequence of same shape [B, SEQ_LEN, ACTION_DIM]
    """
    # If alpha is 0, no smoothing needed - return original sequence
    if alpha == 0.0:
        return action_sequence
    
    B, T, D = action_sequence.shape
    
    # Calculate window size (minimum 1, maximum T)
    window_size = max(1, min(T, int(T * window_size_pct)))
    
    # Initialize output tensor
    filtered = torch.zeros_like(action_sequence)
    
    # Initialize EMA state with first timestep
    ema_state = action_sequence[:, 0, :]  # [B, ACTION_DIM]
    filtered[:, 0, :] = ema_state
    
    # Apply EMA filter across time dimension with windowed reset
    for t in range(1, T):
        # Reset EMA state if we've exceeded the window size
        if t % window_size == 0:
            ema_state = action_sequence[:, t, :]
        else:
            # EMA update: state = alpha * state + (1 - alpha) * new_value
            ema_state = alpha * ema_state + (1 - alpha) * action_sequence[:, t, :]
        
        filtered[:, t, :] = ema_state
    
    return filtered


def window_starts(total_len: int, stride: int):
    max_start = total_len - ACTIONS_PER_BATCH
    if max_start < 0:
        return []
    return list(range(0, max_start + 1, stride))


from numpy.lib.format import open_memmap

@torch.no_grad()
def save_reconstructions_for_file(model,
                                  file_path: Path,
                                  stride: int,
                                  batch_size: int,
                                  device: torch.device,
                                  out_dir: Path):
    """
    Creates:
      <stem>_recon.npy  (float32 .npy with header) [N, ACTIONS_PER_BATCH, ACTION_DIM]
      <stem>_starts.npy (int64)                    [N]
    """
    arr = np.load(file_path, mmap_mode="r")
    starts = window_starts(arr.shape[0], stride)
    stem = file_path.stem

    if not starts:
        # write empty but valid .npy files
        empty_recon = open_memmap(out_dir / f"{stem}_recon.npy",
                                  mode="w+",
                                  dtype=np.float32,
                                  shape=(0, ACTIONS_PER_BATCH, ACTION_DIM))
        del empty_recon
        np.save(out_dir / f"{stem}_starts.npy", np.array([], dtype=np.int64))
        print(f"  (no windows) wrote empty recon & starts for {stem}")
        return

    N = len(starts)
    recon_mm = open_memmap(out_dir / f"{stem}_recon.npy",
                           mode="w+",
                           dtype=np.float32,
                           shape=(N, ACTIONS_PER_BATCH, ACTION_DIM))

    write_pos = 0
    for i in range(0, N, batch_size):
        batch_starts = starts[i:i + batch_size]
        blocks = []
        for s in batch_starts:
            block = np.asarray(arr[s:s + ACTIONS_PER_BATCH], dtype=np.float32)
            if block.shape != (ACTIONS_PER_BATCH, ACTION_DIM):
                raise ValueError(f"Unexpected block shape {block.shape} from {file_path} at start {s}")
            blocks.append(torch.from_numpy(block))
        x = torch.stack(blocks, dim=0).to(device)      # [B,SEQ,D]
        _, recon = model(x)                             # [B,SEQ,D]
        recon = ema_filter(recon,alpha = 0.3, window_size_pct=0.1)
        recon_np = recon.detach().cpu().numpy().astype(np.float32)

        recon_mm[write_pos:write_pos + recon_np.shape[0], :, :] = recon_np
        write_pos += recon_np.shape[0]

    # flush header + data
    del recon_mm

    np.save(out_dir / f"{stem}_starts.npy", np.asarray(starts, dtype=np.int64))
    print(f"  → saved recon: {stem}_recon.npy shape=({N},{ACTIONS_PER_BATCH},{ACTION_DIM})")
    print(f"  → saved starts: {stem}_starts.npy")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Directory with *_actions.npy")
    ap.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint (.pt)")
    ap.add_argument("--save_dir", type=str, default="reconstructed_outputs")
    ap.add_argument("--stride", type=int, default=ACTIONS_PER_BATCH)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--model_dim", type=int, default=256)
    ap.add_argument("--latent_dim", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = load_model(Path(args.checkpoint), args.model_dim, args.latent_dim, device)

    files = sorted(data_dir.glob("*_actions.npy"))
    if not files:
        print(f"No '*_actions.npy' files found in {data_dir}")
        return

    for fp in files:
        print(f"Processing {fp.name} ...")
        save_reconstructions_for_file(model, fp, stride=args.stride,
                                      batch_size=args.batch_size, device=device, out_dir=out_dir)

    print("Done.")

if __name__ == "__main__":
    main()
