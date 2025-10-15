"""
Diffusor-style Rectified Flow Trainer (joint states+actions)
------------------------------------------------------------

This script adapts your original training loop to a Diffuser-aligned
**rectified flow** setup that predicts a velocity field over a *joint* trajectory
(State slots + Action slots), while preserving:

- your dataloaders (init_preprocessed_data_loader)
- your normalization (JEPA pooled → Linear(d_latent→d_state) → LayerNorm; actions z-score)
- logging/visualization (W&B images + metrics)
- automatic checkpointing (uses save_gesture_flow_checkpoint if available)
- progress bars
- overfit test mode

Key changes vs. original:
- swaps the model to a Temporal U-Net that **predicts velocity** v_θ(x_t, t)
  for **rectified flow** training (linear flow from noise→data),
  with masked losses for state vs action slots.
- sampling integrates the ODE with Euler/Heun, with **state anchoring (inpainting)**
  at each step (pin pooled-state slots every step) — mirrors Diffuser’s line-10.
- keeps pen as part of action channels by default; optional separate BCE head
  is provided (disabled by default to keep the step light). Visualization expects
  pen in [0,1], so we clamp with sigmoid at eval time.

Run example (same flags as before, just a new filename):

python train_diffusor_style_flow.py \
  --processed_data_dir resource/dataset/final_data/train_all/mother \
  --manifest resource/dataset/final_data/train_all/mother/manifest/FiLM_1.json \
  --batch_size 24 --learning_rate 1e-4 --num_epochs 20 \
  --d_state 64 --max_sequence_length_frames 4 \
  --wandb_project flow-gesture-generator

"""

from __future__ import annotations
import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------
# External project imports
# -------------------------
from training.dataloader import init_preprocessed_data_loader
from src.utils.logging import get_logger

# Optional checkpoint helper from your project
try:
    from conditioned_gesture_generator.flow_matching_model_old import (
        save_gesture_flow_checkpoint as _save_ckpt_external,
    )
except Exception:
    _save_ckpt_external = None


# ================================
# Visualization helpers (unchanged)
# ================================

def visualize_jepa_reconstruction_heatmap(jepa_latents, recon_latents, title_prefix="JEPA Reconstruction", max_samples=6):
    import numpy as np
    B, T, N, D = jepa_latents.shape
    batch_size = min(B, max_samples)
    mae_per_token = torch.abs(jepa_latents - recon_latents).mean(dim=-1)  # [B,T,N]

    fig, axes = plt.subplots(batch_size, 1, figsize=(12, 3 * batch_size))
    if batch_size == 1:
        axes = [axes]
    for b in range(batch_size):
        mae_map = mae_per_token[b].cpu().numpy()  # [T,N]
        im = axes[b].imshow(mae_map.T, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[b].set_title(f'{title_prefix} - Sample {b+1} - MAE Heatmap')
        axes[b].set_xlabel('Time Steps'); axes[b].set_ylabel('JEPA Tokens')
        plt.colorbar(im, ax=axes[b], label='Mean Absolute Error')
    plt.tight_layout();
    return fig


def visualize_gesture_sequences(original_gestures, generated_gestures, title_prefix="", max_samples=6):
    # Accept torch or numpy
    if isinstance(original_gestures, torch.Tensor):
        original_gestures = original_gestures.detach().cpu().numpy()
    if isinstance(generated_gestures, torch.Tensor):
        generated_gestures = generated_gestures.detach().cpu().numpy()

    batch_size = min(max_samples, original_gestures.shape[0])
    B, L_gesture, C = original_gestures.shape

    original_gestures = original_gestures[:batch_size]
    generated_gestures = generated_gestures[:batch_size]

    # Use adaptive chunk size based on sequence length
    suggested_chunk_size = 250
    if L_gesture < suggested_chunk_size:
        chunk_size = L_gesture
        num_chunks = 1
    elif L_gesture < suggested_chunk_size * 2:
        chunk_size = L_gesture // 2
        num_chunks = 2
    else:
        chunk_size = suggested_chunk_size
        max_chunks = L_gesture // chunk_size
        num_chunks = min(4, max_chunks)

    # Always truncate to exact multiples to avoid reshape errors
    L_truncated = num_chunks * chunk_size
    original_gestures = original_gestures[:, :L_truncated, :]
    generated_gestures = generated_gestures[:, :L_truncated, :]

    try:
        orig_reshaped = original_gestures.reshape(batch_size, num_chunks, chunk_size, 3)
        gen_reshaped = generated_gestures.reshape(batch_size, num_chunks, chunk_size, 3)
    except ValueError as e:
        # Fallback: use smaller chunks
        chunk_size = L_gesture // 4 if L_gesture >= 4 else L_gesture
        num_chunks = L_gesture // chunk_size
        L_truncated = num_chunks * chunk_size
        original_gestures = original_gestures[:, :L_truncated, :]
        generated_gestures = generated_gestures[:, :L_truncated, :]
        orig_reshaped = original_gestures.reshape(batch_size, num_chunks, chunk_size, 3)
        gen_reshaped = generated_gestures.reshape(batch_size, num_chunks, chunk_size, 3)
    num_frames = min(2, num_chunks)

    fig = plt.figure(figsize=(16, 12))
    for i in range(batch_size):
        # Original time-series
        ax_orig = plt.subplot(4, batch_size, i + 1)
        orig_data = original_gestures[i]
        t = np.arange(len(orig_data))
        ax_orig.plot(t, orig_data[:,0], '-', alpha=0.8, label='X', linewidth=1)
        ax_orig.plot(t, orig_data[:,1], '-', alpha=0.8, label='Y', linewidth=1)
        ax_orig.plot(t, orig_data[:,2], '-', alpha=0.8, label='Touch', linewidth=1)
        touch_mask = orig_data[:,2] > 0.5
        if np.any(touch_mask):
            ax_orig.fill_between(t, 0, 1, where=touch_mask, alpha=0.2)
        ax_orig.set_title(f"{title_prefix} Orig {i+1}", fontsize=8)
        ax_orig.set_ylim(-0.1, 1.1); ax_orig.tick_params(labelsize=6); ax_orig.grid(True, alpha=0.3)
        if i==0: ax_orig.legend(fontsize=6); ax_orig.set_ylabel("Value", fontsize=7)

        # Generated time-series
        ax_gen = plt.subplot(4, batch_size, batch_size + i + 1)
        gen_data = generated_gestures[i]
        ax_gen.plot(t, gen_data[:,0], '-', alpha=0.8, label='X', linewidth=1)
        ax_gen.plot(t, gen_data[:,1], '-', alpha=0.8, label='Y', linewidth=1)
        ax_gen.plot(t, gen_data[:,2], '-', alpha=0.8, label='Touch', linewidth=1)
        touch_mask_gen = gen_data[:,2] > 0.5
        if np.any(touch_mask_gen):
            ax_gen.fill_between(t, 0, 1, where=touch_mask_gen, alpha=0.2)
        ax_gen.set_title(f"{title_prefix} Gen {i+1}", fontsize=8)
        ax_gen.set_ylim(-0.1, 1.1); ax_gen.tick_params(labelsize=6); ax_gen.set_xlabel("Timestep", fontsize=7); ax_gen.grid(True, alpha=0.3)
        if i==0: ax_gen.legend(fontsize=6); ax_gen.set_ylabel("Value", fontsize=7)

    # 2D trajectories
    plot_idx = 0
    for i in range(min(4, batch_size)):
        for frame_idx in range(num_frames):
            if plot_idx >= 8: break
            row = 3 if plot_idx < 4 else 4
            col = (plot_idx % 4) + 1
            ax_traj = plt.subplot(4,4,(row-1)*4 + col)
            orig_frame = orig_reshaped[i, frame_idx]
            gen_frame  = gen_reshaped[i, frame_idx]
            # segments pen-down only
            def plot_segments(ax, frame, style, lbl):
                pen_down = frame[:,2] > 0.5
                idx=0; first=True
                while idx < len(frame):
                    if pen_down[idx]:
                        s = idx
                        while idx < len(frame) and pen_down[idx]: idx+=1
                        e = idx
                        if e>s+1:
                            ax.plot(frame[s:e,0], frame[s:e,1], style, alpha=0.7, linewidth=1.5, label=lbl if first else "")
                            first=False
                    else:
                        idx+=1
            plot_segments(ax_traj, orig_frame, '-', 'Original')
            plot_segments(ax_traj, gen_frame,  '--', 'Generated')
            ax_traj.scatter(orig_frame[0,0], orig_frame[0,1], s=30, marker='o', alpha=0.8)
            ax_traj.scatter(orig_frame[-1,0], orig_frame[-1,1], s=30, marker='s', alpha=0.8)
            ax_traj.scatter(gen_frame[0,0], gen_frame[0,1], s=20, marker='o', alpha=0.8)
            ax_traj.scatter(gen_frame[-1,0], gen_frame[-1,1], s=20, marker='s', alpha=0.8)
            ax_traj.set_title(f"2D Traj S{i+1}F{frame_idx+1}", fontsize=8)
            ax_traj.set_xlim(0,1); ax_traj.set_ylim(0,1); ax_traj.set_aspect('equal'); ax_traj.grid(True, alpha=0.3)
            if plot_idx==0: ax_traj.legend(fontsize=6)
            plot_idx+=1
    plt.tight_layout(pad=1.0)
    return fig


# =====================
# Model (Rectified Flow)
# =====================

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1))
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*2), nn.Mish(), nn.Linear(dim*2, dim))
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim()==2: t = t.squeeze(-1)
        args = t.float().unsqueeze(-1) * self.freqs.to(t.device).unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, emb.new_zeros(emb.size(0), 1)], dim=-1)
        return self.mlp(emb)

class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.act = nn.Mish()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t_emb):
        h = self.act(self.norm1(x)); h = self.conv1(h)
        h = h + self.t_proj(t_emb)[:, :, None]
        h = self.act(self.norm2(h)); h = self.conv2(h)
        return h + self.skip(x)

class Downsample1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        # Handle very short sequences: if length <= 2, use adaptive pooling instead
        if x.size(-1) <= 2:
            # Use adaptive average pooling to downsample by 2x
            target_len = max(1, x.size(-1) // 2)
            pooled = F.adaptive_avg_pool1d(x, target_len)
            return pooled
        return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.tconv = nn.ConvTranspose1d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        # For very short sequences, use interpolation
        if x.size(-1) <= 2:
            # Use nearest neighbor interpolation to upsample by 2x
            target_len = x.size(-1) * 2
            upsampled = F.interpolate(x, size=target_len, mode='nearest')
            return upsampled
        return self.tconv(x)

class TemporalUNet(nn.Module):
    def __init__(self, in_ch: int, base_ch: int = 64, ch_mult=(1,2,4,4), num_res_blocks: int = 2, t_dim: int = 128, groups: int = 8):
        super().__init__()
        self.time_mlp = SinusoidalTimestepEmbedding(t_dim)
        self.input = nn.Conv1d(in_ch, base_ch, 7, padding=3)
        downs=[]; ch=base_ch; self.down_channels=[ch]
        for mult in ch_mult:
            out = base_ch * mult
            for _ in range(num_res_blocks):
                downs.append(ResBlock1D(ch, out, t_dim, groups)); ch = out; self.down_channels.append(ch)
            downs.append(Downsample1D(ch))
        self.down = nn.ModuleList(downs)
        self.mid1 = ResBlock1D(ch, ch, t_dim, groups)
        self.mid2 = ResBlock1D(ch, ch, t_dim, groups)
        ups=[]
        for mult in reversed(ch_mult):
            out = base_ch * mult
            ups.append(Upsample1D(ch))
            for _ in range(num_res_blocks):
                ups.append(ResBlock1D(ch + out, out, t_dim, groups)); ch = out
        self.up = nn.ModuleList(ups)
        self.out = nn.Sequential(nn.GroupNorm(groups, ch), nn.Mish(), nn.Conv1d(ch, in_ch, 3, padding=1))
    def forward(self, x: torch.Tensor, t_cont: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C)
        x = x.transpose(1,2)  # (B,C,T)
        t_emb = self.time_mlp(t_cont)
        h = self.input(x); skips=[]
        for m in self.down:
            if isinstance(m, ResBlock1D):
                h = m(h, t_emb); skips.append(h)
            else:
                h = m(h)
        h = self.mid1(h, t_emb); h = self.mid2(h, t_emb)
        for m in self.up:
            if isinstance(m, Upsample1D):
                h = m(h)
                if len(skips)>0:
                    tgt = skips[-1]
                    if h.size(-1) != tgt.size(-1):
                        diff = tgt.size(-1) - h.size(-1)
                        if diff>0: h = F.pad(h, (0, diff))
                        else: h = h[..., :tgt.size(-1)]
            else:
                skip = skips.pop(); h = torch.cat([h, skip], dim=1); h = m(h, t_emb)
        out = self.out(h).transpose(1,2)  # (B,T,C)
        return out


class RFJointModel(nn.Module):
    """Rectified-Flow joint model: predicts velocity over joint [state|action] channels.
    Includes separate pen head for pen state prediction based on x,y coordinates.
    """
    def __init__(self, in_ch: int, base_ch: int = 64, t_dim: int = 128, ch_mult=(1,2,4,4), num_res_blocks: int = 2,
                 use_pen_head: bool = True, d_state: int = 64):
        super().__init__()
        self.unet = TemporalUNet(in_ch, base_ch, ch_mult, num_res_blocks, t_dim)
        self.use_pen_head = use_pen_head
        self.d_state = d_state

        if use_pen_head:
            # Pen head takes x,y coordinates (2D) and predicts pen state
            self.pen_head = nn.Sequential(
                nn.Linear(2, 32),
                nn.Mish(),
                nn.Linear(32, 16),
                nn.Mish(),
                nn.Linear(16, 1)  # Output pen logits
            )

    def forward_velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t: (B, T_joint, C); C = d_state + d_action
        return self.unet(x_t, t)  # (B,T,C) velocity field

    def forward_pen_logits(self, xy_coords: torch.Tensor) -> torch.Tensor:
        """Predict pen state from x,y coordinates.
        Args:
            xy_coords: (B, T, 2) x,y coordinates
        Returns:
            pen_logits: (B, T, 1) pen state logits
        """
        if not self.use_pen_head:
            raise RuntimeError("pen_head disabled; instantiate with use_pen_head=True")
        return self.pen_head(xy_coords)  # (B,T,2) -> (B,T,1)


# ============================
# Rectified Flow: train + samp
# ============================

@dataclass
class RFSchedule:
    steps: int = 50
    def t_grid(self, device) -> torch.Tensor:
        # increasing t from 0 -> 1
        return torch.linspace(0.0, 1.0, self.steps, device=device)


def rf_pair_linear(x0: torch.Tensor, x1: torch.Tensor, noisy: bool = False, sigma: float = 0.0):
    """Build (x_t, t, v_target) for rectified flow with linear path.
    v* = x1 - x0; x_t = (1 - t) x0 + t x1.
    Optionally perturb the path with small Gaussian noise.
    """
    B = x0.size(0)
    t = torch.rand(B, device=x0.device)  # U[0,1]
    v_target = x1 - x0
    x_t = (1.0 - t.view(B,1,1)) * x0 + t.view(B,1,1) * x1
    if noisy and sigma > 0:
        x_t = x_t + sigma * torch.randn_like(x_t) * t.view(B,1,1) * (1 - t.view(B,1,1))
    return x_t, t, v_target


@torch.no_grad()
def rf_sample(model: RFJointModel, shape: Tuple[int,int,int], state_positions: List[int], d_state: int,
              steps: int = 50, method: str = "heun", state_anchor: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Integrate ODE dx/dt = v_θ(x,t) from t=0 to 1 starting from noise.
    - state anchoring: if provided (B,F,d_state) pooled tokens, they will be injected at the
      designated state positions at each step into the state slice.
    """
    device = next(model.parameters()).device
    B, T, C = shape
    x = torch.randn(B, T, C, device=device)
    t_grid = torch.linspace(0.0, 1.0, steps, device=device)

    for i in range(steps-1):
        t = t_grid[i].expand(B)
        v = model.forward_velocity(x, t)  # (B,T,C)
        dt = (t_grid[i+1] - t_grid[i]).item()
        if method == "euler":
            x_next = x + dt * v
        else:  # Heun (predictor-corrector)
            x_pred = x + dt * v
            v_pred = model.forward_velocity(x_pred, t_grid[i+1].expand(B))
            x_next = x + 0.5 * dt * (v + v_pred)

        # State anchoring (hard inpainting at each step)
        if state_anchor is not None:
            # state_anchor: (B, F, d_state) mapped to positions
            for si, pos in enumerate(state_positions):
                x_next[:, pos, :d_state] = state_anchor[:, si, :]
        x = x_next
    return x


# ===================
# Trainer (diffusor-RF)
# ===================

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.logger = get_logger()

        # Seeds
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # State projection + LN (JEPA pooled → d_state)
        self.state_proj = nn.Linear(args.d_latent, args.d_state).to(self.device)
        self.state_norm = nn.LayerNorm(args.d_state).to(self.device)
        self.logger.info(f"State projection: {args.d_latent} -> {args.d_state} + LayerNorm")

        # Action dims
        self.include_pen = args.include_pen_in_flow
        self.d_action = 3 if self.include_pen else 2
        self.d_state = args.d_state
        # The model will work with joint sequences that have varying action dimensions
        # We'll dynamically calculate the input dimension based on sequence length
        in_feature_dim = self.d_state + self.d_action  # Just state + single action step for now

        # Model (Rectified Flow)
        use_pen_head = not args.disable_pen_head  # Enable by default, disable with flag

        self.logger.info(f"Input feature dim: {in_feature_dim} (state {self.d_state} + action {self.d_action})")
        self.logger.info(f"Pen head: {'enabled' if use_pen_head else 'disabled'}")
        self.model = RFJointModel(in_ch=in_feature_dim, base_ch=args.base_ch, t_dim=args.time_embed_dim,
                                  ch_mult=tuple(args.unet_channels), num_res_blocks=args.num_res_blocks,
                                  use_pen_head=use_pen_head, d_state=self.d_state).to(self.device)

        # Optimizer (model + state proj/LN)
        all_params = list(self.model.parameters()) + list(self.state_proj.parameters()) + list(self.state_norm.parameters())
        self.optimizer = torch.optim.AdamW(all_params, lr=args.learning_rate, betas=(0.9,0.999), weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_epochs, eta_min=args.learning_rate*0.1)

        # Data
        self.train_loader, _ = init_preprocessed_data_loader(
            processed_data_dir=args.processed_data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
            manifest_path=args.manifest, split_name='train')
        self.val_loader, _ = init_preprocessed_data_loader(
            processed_data_dir=args.processed_data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
            manifest_path=args.manifest, split_name='validation')

        # Action stats from training set
        mu, sigma = self._compute_dataset_action_stats()
        self.action_mu = torch.tensor(mu, device=self.device).view(1,1,-1)
        self.action_sigma = torch.tensor(sigma, device=self.device).view(1,1,-1)

        # Checkpoint/logging
        self.run_dir = Path(args.output_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}_rf"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float('inf'); self.global_step = 0

        # W&B
        if args.wandb_project:
            import wandb
            run_name = args.wandb_run_name if args.wandb_run_name else self.run_dir.name
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
            wandb.watch(self.model, log="gradients", log_freq=100)
            ds = {"dataset/action_mu": mu, "dataset/action_sigma": sigma}
            if self.include_pen:
                ds.update({"dataset/action_mu_pen": mu[2], "dataset/action_sigma_pen": sigma[2]})
            wandb.log(ds)

        # Resume
        self.start_epoch = 0
        if args.resume_from:
            self._load_checkpoint(args.resume_from)

    # ---- Stats / norms ----
    def _norm_actions(self, a): return (a - self.action_mu) / (self.action_sigma + 1e-6)
    def _denorm_actions(self, a): return a * (self.action_sigma + 1e-6) + self.action_mu

    def _compute_dataset_action_stats(self):
        self.logger.info("Computing dataset action statistics...")
        all_actions=[]; sample_count=0; max_samples=1000
        for batch_idx, sample in enumerate(self.train_loader):
            if sample_count >= max_samples: break
            _, gt_actions = sample
            gt_actions = gt_actions.to(self.device)  # [B, T-1, 250, 3]
            B, Tm1, Traj, _ = gt_actions.shape
            acts = gt_actions.contiguous().view(-1, 3)
            acts = acts if self.include_pen else acts[...,:2]
            all_actions.append(acts.cpu()); sample_count += B
        if not all_actions: raise ValueError("No action data found in training set")
        all_actions = torch.cat(all_actions, dim=0)
        mu = all_actions.mean(dim=0).tolist(); std = all_actions.std(dim=0).tolist()
        self.logger.info(f"Computed action stats from {all_actions.shape[0]} points: mu={mu}, sigma={std}")
        return mu, std

    # ---- Batch prep: build joint sequence ----
    def prepare_batch(self, sample):
        visual_embeddings, gt_actions = sample
        visual_embeddings = visual_embeddings.to(self.device, non_blocking=True)  # [B,T,N,D]
        gt_actions = gt_actions.to(self.device, non_blocking=True)                # [B,T-1,250,3]

        # z-score JEPA tokens over D
        visual_embeddings = F.layer_norm(visual_embeddings, (visual_embeddings.size(-1),))

        B, Tm1, Traj, _ = gt_actions.shape; T = Tm1 + 1
        # For training: random sequence length in [1, N_max] (unless overfit/val)
        if self.model.training and hasattr(self.args, 'max_sequence_length_frames') and not getattr(self.args, 'overfit', False):
            N_max = min(Tm1, self.args.max_sequence_length_frames)
            seq_len = torch.randint(1, N_max+1, (1,)).item()
        else:
            seq_len = Tm1

        latent_frames = visual_embeddings[:, :seq_len+1].contiguous()  # [B, F=seq_len+1, N, D]
        latent_frames = F.layer_norm(latent_frames, (latent_frames.size(-1),))

        # Pooled state tokens → proj → LN
        pooled = latent_frames.mean(dim=2)  # [B,F,D_lat]
        state_tokens = self.state_norm(self.state_proj(pooled))  # [B,F,d_state]

        # Action tokens (simple flatten segments)
        action_seq = gt_actions[:, :seq_len].contiguous().view(B, seq_len*Traj, 3)  # [B, seq_len*250, 3]
        action_tokens_raw = action_seq if self.include_pen else action_seq[...,:2]  # Keep or drop pen
        action_tokens = self._norm_actions(action_tokens_raw)  # Normalize

        # Interleave: [s0][a0(flat)]...[sN]
        num_states = state_tokens.shape[1]  # F frames
        L_actions = action_tokens.shape[1]   # seq_len * 250
        joint_len = num_states + L_actions
        joint_dim = self.d_state + self.d_action
        joint_seq = torch.zeros(B, joint_len, joint_dim, device=self.device)
        type_mask = torch.zeros(B, joint_len, 1, device=self.device)

        state_positions = []
        pos = 0
        for i in range(num_states):
            # Place state
            joint_seq[:, pos, :self.d_state] = state_tokens[:, i]
            type_mask[:, pos, 0] = 1.0  # state mask
            state_positions.append(pos)

            if i < num_states - 1:  # Not the last state
                # Place actions between states
                actions_per_segment = Traj  # 250
                a_start = i * actions_per_segment
                a_end = (i + 1) * actions_per_segment
                joint_seq[:, pos+1:pos+1+actions_per_segment, self.d_state:] = action_tokens[:, a_start:a_end]
                pos += 1 + actions_per_segment
            else:
                pos += 1

        layout = {"state_positions": state_positions, "sequence_length": seq_len, "traj": Traj}
        return joint_seq, type_mask, latent_frames, state_tokens, layout, action_seq

    # ---- Loss (masked MSE on velocity) ----
    @staticmethod
    def masked_mse(pred, target, mask):
        # pred/target: (B,T,C), mask: (B,T,1)
        se = (pred - target) ** 2
        return (se * mask).sum() / (mask.sum() * pred.size(-1) + 1e-8)

    def compute_joint_velocity_loss(self, v_pred, v_target, type_mask, state_weight: float):
        # Slice channels
        Ds = self.d_state
        m_state = type_mask
        m_action = 1.0 - type_mask
        loss_s = self.masked_mse(v_pred[...,:Ds], v_target[...,:Ds], m_state)
        loss_a = self.masked_mse(v_pred[...,Ds:], v_target[...,Ds:], m_action)
        total = state_weight * loss_s + loss_a
        return {"loss": total, "state_mse": loss_s, "action_mse": loss_a}

    # ---- Training step ----
    def train_step(self, sample):
        joint_seq, type_mask, latent_frames, state_tokens, layout, action_seq = self.prepare_batch(sample)
        B, T_joint, C_joint = joint_seq.shape

        # Build RF pair: x0 ~ N(0,1), x1 = joint_seq
        x0 = torch.randn_like(joint_seq)
        x_t, t, v_target = rf_pair_linear(x0, joint_seq, noisy=self.args.rf_noisy, sigma=self.args.rf_sigma)

        # 1. Predict velocity (flow matching loss)
        v_pred = self.model.forward_velocity(x_t, t)
        flow_loss_dict = self.compute_joint_velocity_loss(v_pred, v_target, type_mask, state_weight=self.args.state_weight)
        flow_loss = flow_loss_dict["loss"]

        # 2. Pen head training (if enabled) - use GT x,y coordinates
        pen_loss = torch.tensor(0.0, device=joint_seq.device)
        if self.model.use_pen_head:
            # Extract action positions from joint sequence
            action_mask = (1.0 - type_mask).bool().squeeze(-1)  # [B, T_joint] - True for action positions

            # Simple extraction: action slots have x,y,pen directly
            xy_batches, sample_ids = [], []
            for b in range(joint_seq.size(0)):
                action_positions = joint_seq[b, action_mask[b], self.d_state:self.d_state+2]  # [N_actions, 2] - x,y only
                if action_positions.size(0) > 0:
                    xy_batches.append(action_positions)
                    sample_ids.append(b)

            if xy_batches:
                # Find minimum length to pad/truncate to same size
                min_points = min([xy.size(0) for xy in xy_batches])
                gt_joint_xy = torch.stack([xy[:min_points] for xy in xy_batches], dim=0)  # [B', min_points, 2]

                # Get corresponding pen ground truth from action_seq
                seq_len = layout['sequence_length']
                traj = layout['traj']
                gt_pen_flat = action_seq[:, :, 2].contiguous().view(B, seq_len * traj)  # [B, seq_len*250]
                gt_pen_states = gt_pen_flat[sample_ids, :min_points]  # [B', min_points]

                # Predict pen states from GT x,y coordinates
                pen_logits = self.model.forward_pen_logits(gt_joint_xy)  # [B', min_points, 1]

                # Compute BCE loss
                pen_loss = F.binary_cross_entropy_with_logits(
                    pen_logits.squeeze(-1), gt_pen_states, reduction='mean'
                )

        # 3. Combined loss
        pen_weight = getattr(self.args, 'pen_weight', 1.0)
        total_loss = flow_loss + pen_weight * pen_loss

        self.optimizer.zero_grad(); total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.optimizer.step(); self.global_step += 1

        metrics = {
            "loss": total_loss.item(),
            "flow_loss": flow_loss.item(),
            "pen_loss": pen_loss.item(),
            "state_mse": flow_loss_dict['state_mse'].item(),
            "action_mse": flow_loss_dict['action_mse'].item(),
            "sequence_length": layout['sequence_length'],
            "state_weight": self.args.state_weight,
        }
        return metrics

    # ---- Validation ----
    @torch.no_grad()
    def validate(self):
        self.model.eval(); total=0; n=0
        for sample in self.val_loader:
            joint_seq, type_mask, latent_frames, state_tokens, layout, action_seq = self.prepare_batch(sample)
            B = joint_seq.size(0)
            x0 = torch.randn_like(joint_seq)
            x_t, t, v_target = rf_pair_linear(x0, joint_seq, noisy=self.args.rf_noisy, sigma=self.args.rf_sigma)

            # Flow loss
            v_pred = self.model.forward_velocity(x_t, t)
            flow_loss_dict = self.compute_joint_velocity_loss(v_pred, v_target, type_mask, state_weight=self.args.state_weight)
            flow_loss = flow_loss_dict['loss']

            # Pen loss (if enabled)
            pen_loss = torch.tensor(0.0, device=joint_seq.device)
            if self.model.use_pen_head:
                action_mask = (1.0 - type_mask).bool().squeeze(-1)  # [B, T_joint]

                # Simple extraction: action slots have x,y,pen directly
                xy_batches, sample_ids = [], []
                for b in range(joint_seq.size(0)):
                    action_positions = joint_seq[b, action_mask[b], self.d_state:self.d_state+2]  # [N_actions, 2] - x,y only
                    if action_positions.size(0) > 0:
                        xy_batches.append(action_positions)
                        sample_ids.append(b)

                if xy_batches:
                    min_points = min([xy.size(0) for xy in xy_batches])
                    gt_joint_xy = torch.stack([xy[:min_points] for xy in xy_batches], dim=0)  # [B', min_points, 2]

                    seq_len = layout['sequence_length']
                    traj = layout['traj']
                    gt_pen_flat = action_seq[:, :, 2].contiguous().view(B, seq_len * traj)  # [B, seq_len*250]
                    gt_pen_states = gt_pen_flat[sample_ids, :min_points]  # [B', min_points]

                    pen_logits = self.model.forward_pen_logits(gt_joint_xy)  # [B', min_points, 1]
                    pen_loss = F.binary_cross_entropy_with_logits(
                        pen_logits.squeeze(-1), gt_pen_states, reduction='mean'
                    )

            pen_weight = getattr(self.args, 'pen_weight', 1.0)
            batch_total_loss = flow_loss + pen_weight * pen_loss
            total += batch_total_loss.item(); n += 1
        return total / max(n,1)

    # ---- Generation helpers ----
    def extract_actions_from_joint(self, x_joint: torch.Tensor, state_positions: List[int]) -> torch.Tensor:
        """Return (B, L_actions, d_action) by taking all non-state positions' action slice.
        """
        B, T, C = x_joint.shape
        mask = torch.ones(T, dtype=torch.bool, device=x_joint.device)
        mask[state_positions] = False
        actions = x_joint[:, mask, self.d_state:]  # [B, L_actions, d_action]
        return actions

    @torch.no_grad()
    def generate_samples(self, loader, split_name: str, num_batches: int = 1):
        import wandb
        self.model.eval()
        all_gen=[]; all_gt=[]; all_emb=[]; seq_lens=[]
        batches_done = 0
        for sample in loader:
            joint_seq, type_mask, latent_frames, state_tokens, layout, action_seq = self.prepare_batch(sample)
            B, T_joint, C_joint = joint_seq.shape
            state_positions = layout['state_positions']

            # State anchor = the pooled states we built
            gen_joint = rf_sample(self.model, (B,T_joint,C_joint), state_positions, self.d_state,
                                  steps=self.args.flow_steps, method=self.args.ode_method,
                                  state_anchor=state_tokens)

            # Extract action slice - simple flattened actions
            gen_actions_norm = self.extract_actions_from_joint(gen_joint, state_positions)  # [B, L_actions, d_action]
            gen_actions_xy = self._denorm_actions(gen_actions_norm)  # back to [0,1] scale

            # Generate pen predictions using pen head on generated x,y coordinates
            if self.model.use_pen_head:
                pen_logits = self.model.forward_pen_logits(gen_actions_xy)  # [B, L_gesture*K, 1]
                pen_probs = torch.sigmoid(pen_logits)  # Convert to probabilities
                gen_actions = torch.cat([gen_actions_xy, pen_probs], dim=-1)  # [B, L_gesture*K, 3]
            else:
                # Fallback: add dummy pen channel (zeros) for visualization compatibility
                dummy_pen = torch.zeros(gen_actions_xy.shape[0], gen_actions_xy.shape[1], 1, device=gen_actions_xy.device)
                gen_actions = torch.cat([gen_actions_xy, dummy_pen], dim=-1)

            # GT flattened gesture for same horizon
            Lg = gen_actions.size(1)
            gt_gestures = action_seq[:, :Lg, :]  # Match length

            all_gen.append(gen_actions.cpu()); all_gt.append(gt_gestures.cpu()); all_emb.append(latent_frames.cpu())
            seq_lens.extend([layout['sequence_length']] * B)
            batches_done += 1
            if batches_done >= num_batches: break

        gen_tensor = torch.cat(all_gen, dim=0)
        gt_tensor = torch.cat(all_gt, dim=0)
        emb_tensor = torch.cat(all_emb, dim=0)

        fig = visualize_gesture_sequences(gt_tensor, gen_tensor, title_prefix=f"{split_name.title()} Samples", max_samples=min(6, len(gen_tensor)))

        # JEPA reconstruction viz (project back via state_proj^T)
        pooled = emb_tensor.mean(dim=2)                              # [B,F,D_lat]
        state_proj = self.state_norm(self.state_proj(pooled.to(self.device))).cpu()  # [B,F,d_state]
        recon_pooled = F.linear(state_proj, self.state_proj.weight.T.cpu(), None)    # [B,F,D_lat]
        recon_latents = recon_pooled.unsqueeze(2).expand_as(emb_tensor)
        jepa_fig = visualize_jepa_reconstruction_heatmap(emb_tensor, recon_latents, title_prefix=f"{split_name.title()} JEPA State Reconstruction", max_samples=min(6, len(emb_tensor)))

        log = {
            f"{split_name}_gesture_samples": wandb.Image(fig),
            f"{split_name}_jepa_reconstruction": wandb.Image(jepa_fig),
            f"{split_name}/avg_sequence_length": float(np.mean(seq_lens)),
            "global_step": self.global_step,
        }
        # Action errors
        mse_x = F.mse_loss(gen_tensor[...,0], gt_tensor[...,0]).item()
        mse_y = F.mse_loss(gen_tensor[...,1], gt_tensor[...,1]).item()
        mse_pen = F.mse_loss(gen_tensor[...,2], gt_tensor[...,2]).item() if self.include_pen else 0.0
        log.update({f"{split_name}/mse_x": mse_x, f"{split_name}/mse_y": mse_y, f"{split_name}/mse_touch": mse_pen,
                    f"{split_name}/action_coord_error": 0.5*(mse_x+mse_y), f"{split_name}/action_pen_error": mse_pen,
                    f"{split_name}/gen_mean_x": gen_tensor[...,0].mean().item(),
                    f"{split_name}/gen_mean_y": gen_tensor[...,1].mean().item(),
                    f"{split_name}/gen_mean_touch": gen_tensor[...,2].mean().item() if self.include_pen else 0.0,
                    f"{split_name}/gt_mean_x": gt_tensor[...,0].mean().item(),
                    f"{split_name}/gt_mean_y": gt_tensor[...,1].mean().item(),
                    f"{split_name}/gt_mean_touch": gt_tensor[...,2].mean().item() if self.include_pen else 0.0})
        import wandb
        wandb.log(log)
        plt.close(fig); plt.close(jepa_fig)
        self.model.train()

    # ---- Train loop ----
    def training_loop(self):
        if self.args.overfit:
            self.overfit_training_loop(); return

        self.logger.info("Starting rectified-flow training loop (Diffuser-aligned)...")
        self.logger.info("=== Pre-training validation ===")
        init_val = self.validate(); self.logger.info(f"Initial val loss: {init_val:.4f}")

        for epoch in range(self.start_epoch, self.args.num_epochs):
            self.model.train(); epoch_loss=0; n=0
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
            total_batches = len(self.train_loader); half_ckpt = total_batches // 2

            for i, sample in enumerate(progress):
                loss = self.train_step(sample)
                epoch_loss += loss['loss']; n += 1
                if i % 10 == 0:
                    progress.set_postfix({"loss": f"{loss['loss']:.4f}"})
                if self.args.wandb_project and i % 20 == 0:
                    import wandb
                    log_dict = {
                        "train/total_loss": loss['loss'],
                        "train/flow_loss": loss['flow_loss'],
                        "train/pen_loss": loss['pen_loss'],
                        "train/state_error": loss['state_mse'],
                        "train/action_error": loss['action_mse'],
                        "train/lr": self.optimizer.param_groups[0]['lr'],
                        "train/sequence_length": loss['sequence_length'],
                        "global_step": self.global_step,
                        "epoch": epoch,
                    }
                    wandb.log(log_dict)
                if i == half_ckpt:
                    self.logger.info(f"=== Half-epoch checkpoint: {epoch+1}.5 ===")
                    val_loss = self.validate(); self.logger.info(f"Half-epoch val: {val_loss:.4f}")
                    if self.args.wandb_project:
                        import wandb
                        wandb.log({"val_half/loss": val_loss, "val_half/epoch": epoch+0.5, "global_step": self.global_step})
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.logger.info(f"< New best (half): {val_loss:.4f}. Saving checkpoint.")
                        self._save_checkpoint("best", val_loss)

            self.scheduler.step()
            avg_train = epoch_loss / max(n,1)
            self.logger.info(f"--- Epoch {epoch+1} complete --- Avg Train Loss: {avg_train:.4f}")

            self.logger.info(f"=== End-of-epoch validation ===")
            val_loss = self.validate(); self.logger.info(f"Val loss: {val_loss:.4f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.logger.info(f"< New best (end): {val_loss:.4f}. Saving checkpoint.")
                self._save_checkpoint("best", val_loss)
            if self.args.wandb_project:
                import wandb
                wandb.log({"val_end/loss": val_loss, "val_end/best_loss": self.best_val_loss, "val_end/epoch": epoch+1, "global_step": self.global_step})

            if (epoch + 1) % self.args.save_freq == 0:
                self._save_checkpoint(f"epoch_{epoch+1}")

            # Sample a few batches from train/val for visualization each epoch
            if self.args.wandb_project:
                self.generate_samples(self.train_loader, split_name="train", num_batches=1)
                self.generate_samples(self.val_loader, split_name="val", num_batches=1)

        self.logger.info("Training completed!")

    # ---- Overfit loop ----
    def overfit_training_loop(self):
        self.logger.info("=== OVERFIT TEST MODE ===")
        self.model.train()
        first_batch = next(iter(self.train_loader))
        single_embeddings, single_gestures = first_batch
        single_embeddings = single_embeddings[:1]; single_gestures = single_gestures[:1]
        single_sample = (single_embeddings, single_gestures)
        total_steps=0
        for epoch in range(self.args.num_epochs):
            for step in range(100):
                loss = self.train_step(single_sample); total_steps+=1
                if total_steps % 50 == 0:
                    self.logger.info(f"Step {total_steps}: Loss={loss['loss']:.6f} Flow={loss['flow_loss']:.6f} Pen={loss['pen_loss']:.6f} State={loss['state_mse']:.6f} Action={loss['action_mse']:.6f}")
                if self.args.wandb_project and total_steps % 10 == 0:
                    import wandb
                    wandb.log({
                        "overfit/total_loss": loss['loss'],
                        "overfit/flow_loss": loss['flow_loss'],
                        "overfit/pen_loss": loss['pen_loss'],
                        "overfit/state_error": loss['state_mse'],
                        "overfit/action_error": loss['action_mse'],
                        "overfit/lr": self.optimizer.param_groups[0]['lr'],
                        "overfit/step": total_steps, "overfit/epoch": epoch
                    })
            self.scheduler.step()
            self.logger.info(f"=== Overfit Epoch {epoch+1}/{self.args.num_epochs} Complete === Last loss: {loss['loss']:.6f}")
            if (epoch+1) % 10 == 0 and self.args.wandb_project:
                # generate
                self.generate_samples(self.train_loader, split_name="overfit", num_batches=1)
        self.logger.info("Overfit training completed!")

    # ---- Checkpointing ----
    def _save_checkpoint(self, tag: str, val_loss: float = None):
        ckpt_path = self.run_dir / f"rf_{tag}.pt"
        trainer_state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "epoch": self.start_epoch,
        }
        payload = {
            "model_state_dict": self.model.state_dict(),
            "state_proj": self.state_proj.state_dict(),
            "state_norm": self.state_norm.state_dict(),
            "trainer_state": trainer_state,
        }
        try:
            if _save_ckpt_external is not None:
                _save_ckpt_external(model=self.model, checkpoint_path=str(ckpt_path), trainer_state=trainer_state)
            else:
                torch.save(payload, ckpt_path)
            # latest symlink
            latest = self.run_dir / "latest.pt"
            if latest.exists(): latest.unlink()
            latest.symlink_to(ckpt_path.name)
            self.logger.info(f"✓ Checkpoint saved to {ckpt_path}")
        except Exception as e:
            self.logger.error(f"✗ Failed to save checkpoint: {e}")

    def _load_checkpoint(self, path: str):
        try:
            ckpt = torch.load(path, map_location=self.device)
            if "model_state_dict" in ckpt:  # our local format
                self.model.load_state_dict(ckpt["model_state_dict"])  # type: ignore
                self.state_proj.load_state_dict(ckpt["state_proj"])   # type: ignore
                self.state_norm.load_state_dict(ckpt["state_norm"])   # type: ignore
                tr = ckpt.get("trainer_state", {})
                self.optimizer.load_state_dict(tr.get("optimizer_state_dict", self.optimizer.state_dict()))
                self.scheduler.load_state_dict(tr.get("scheduler_state_dict", self.scheduler.state_dict()))
                self.best_val_loss = tr.get("best_val_loss", self.best_val_loss)
                self.global_step = tr.get("global_step", self.global_step)
                self.start_epoch = tr.get("epoch", self.start_epoch)
            else:  # external project format
                self.model.load_state_dict(ckpt["model_state_dict"])  # type: ignore
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])  # type: ignore
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])  # type: ignore
                self.best_val_loss = ckpt.get("best_val_loss", float('inf'))
                self.global_step = ckpt.get("global_step", 0)
                batches = len(self.train_loader)
                self.start_epoch = self.global_step // max(batches,1)
            self.logger.info("✓ Checkpoint loaded")
        except Exception as e:
            self.logger.error(f"✗ Failed to load checkpoint: {e}")
            raise


# =====================
# Args & entry point
# =====================

def get_args():
    p = argparse.ArgumentParser(description="Train Diffusor-style Rectified Flow (joint states+actions)")
    # Data
    p.add_argument("--processed_data_dir", type=str, required=True)
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="checkpoints/flow_gesture")
    p.add_argument("--resume_from", type=str, default=None)
    # Model
    p.add_argument("--d_latent", type=int, default=1024)
    p.add_argument("--d_state", type=int, default=64)
    p.add_argument("--include_pen_in_flow", action="store_true")
    p.add_argument("--state_weight", type=float, default=0.5)
    p.add_argument("--micro_pe_dim", type=int, default=8, help="Micro-time positional encoding dimensions (must be even)")
    p.add_argument("--base_ch", type=int, default=96)
    p.add_argument("--unet_channels", type=int, nargs='+', default=[1,2,4,4])
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--time_embed_dim", type=int, default=128)
    p.add_argument("--disable_pen_head", action="store_true", help="Disable pen head (pen head enabled by default)")
    p.add_argument("--max_sequence_length_frames", type=int, default=4, help="Max sequence length in frames (minimum 2 for conv stability)")
    # RF / ODE
    p.add_argument("--rf_noisy", action="store_true")
    p.add_argument("--rf_sigma", type=float, default=0.1)
    p.add_argument("--flow_steps", type=int, default=50)
    p.add_argument("--ode_method", type=str, default="heun", choices=["euler","heun"])
    # Train
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--pen_weight", type=float, default=1.0, help="Weight for pen BCE loss")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # Logging / eval
    p.add_argument("--wandb_project", type=str, default="flow-gesture-generator")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--save_freq", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda")
    # Overfit
    p.add_argument("--overfit", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    try:
        trainer = Trainer(args)
        trainer.training_loop()
        print("Training finished successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise


if __name__ == "__main__":
    main()
