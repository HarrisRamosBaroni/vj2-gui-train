import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, Union
import os
from pathlib import Path

class SinusoidalPosEnc1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "pos_enc dim must be even"
        self.dim = dim

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: [B, L] (int or float)
        if positions.dim() == 1:
            positions = positions[None, :]
        B, L = positions.shape
        device = positions.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / (half - 1))
        )  # [half]
        angles = positions.float().unsqueeze(-1) * freqs  # [B,L,half]
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B,L,dim]
        return pe

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for flow matching timesteps."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal embeddings for timesteps.

        Args:
            timesteps: Tensor of shape [B] containing float32 values in [0,1]
        Returns:
            embeddings: Tensor of shape [B, embed_dim]
        """
        if timesteps.dim() == 2:
            timesteps = timesteps.squeeze(-1)

        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)

        return emb






class AttentionBlock(nn.Module):
    """Self-attention block for bottleneck."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with self-attention.

        Args:
            x: Input tensor [B, C, L]
        Returns:
            Output tensor [B, C, L]
        """
        # x: [B,C,L] -> transpose to [B,L,C]
        x_in = x.transpose(1, 2)
        h = self.norm(x_in)
        h, _ = self.attn(h, h, h)
        return (h + x_in).transpose(1, 2)


class FiLMCrossAttention(nn.Module):
    """FiLM conditioning via cross-attention that returns per-frame FiLM parameters."""

    def __init__(self, d_model: int, d_latent: int = 1024, nhead: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.nhead = nhead

        # Learned query (used per frame)
        self.gesture_query = nn.Parameter(torch.randn(1, 1, d_model))

        # Project latent patches to key/value
        self.latent_key_proj = nn.Linear(d_latent, d_model)
        self.latent_value_proj = nn.Linear(d_latent, d_model)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )

        # Map attended features -> FiLM params (film_dim == d_model here)
        self.film_scale = nn.Linear(d_model, d_model)
        self.film_shift = nn.Linear(d_model, d_model)

        nn.init.normal_(self.film_scale.weight, std=0.02)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.normal_(self.film_shift.weight, std=0.02)
        nn.init.zeros_(self.film_shift.bias)

    def forward(
        self,
        latent_frames: torch.Tensor,
        sequence_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_frames: [B, F_max+1, N, D] (F includes context frame)
            sequence_length: use first (sequence_length + 1) frames if provided
        Returns:
            gamma: [B, F, d_model]
            beta:  [B, F, d_model]
        """
        B, F_all, N, D = latent_frames.shape

        # Mask to actual frames (context + sequence_length)
        if sequence_length is not None:
            num_frames = sequence_length + 1
            latent_frames = latent_frames[:, :num_frames]
        else:
            num_frames = F_all

        # Do attention per frame: reshape (B,F,N,D) -> (B*F, N, D)
        frames_flat = latent_frames.reshape(B * num_frames, N, D)
        keys   = self.latent_key_proj(frames_flat)           # [B*F, N, d_model]
        values = self.latent_value_proj(frames_flat)         # [B*F, N, d_model]

        # Repeat the learned query per frame in the batch
        query = self.gesture_query.expand(B * num_frames, -1, -1)  # [B*F, 1, d_model]

        attended, _ = self.multihead_attn(query=query, key=keys, value=values)  # [B*F, 1, d_model]
        attended = attended.squeeze(1).reshape(B, num_frames, -1)               # [B, F, d_model]

        gamma = self.film_scale(attended)                    # [B, F, d_model]
        beta  = self.film_shift(attended)                    # [B, F, d_model]
        return gamma, beta


class FiLMResidualBlock(nn.Module):
    """Residual block with time embedding + per-step FiLM ([B,L,film_dim])."""
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, film_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_channels)

        # Map FiLM dim -> out_channels, applied per position
        self.film_scale = nn.Linear(film_dim, out_channels)
        self.film_shift = nn.Linear(film_dim, out_channels)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,                 # [B, C_in, L]
        t_emb: torch.Tensor,             # [B, time_dim]
        gamma_steps: Optional[torch.Tensor],  # [B, L, film_dim] or None
        beta_steps: Optional[torch.Tensor]    # [B, L, film_dim] or None
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # add time embedding
        h = h + self.time_proj(t_emb)[:, :, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)  # [B, C_out, L]

        # per-step FiLM if provided
        if gamma_steps is not None and beta_steps is not None:
            # [B,L,film_dim] -> [B,L,C_out] -> [B,C_out,L]
            g = self.film_scale(gamma_steps).transpose(1, 2)
            b = self.film_shift(beta_steps).transpose(1, 2)
            h = g * h + b

        return h + self.residual(x)


class ResidualBlock(nn.Module):
    """Standard residual block with time embedding (no FiLM)."""
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: [B,C,L], t_emb: [B,time_dim]
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = h + self.time_proj(t_emb)[:, :, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.residual(x)


class GestureFlowUNet(nn.Module):
    """U-Net for joint trajectory velocity prediction with FiLM state conditioning."""
    def __init__(self, in_feature_dim: int, time_embed_dim: int = 128,
                 film_dim: int = 128, channels: tuple = (64, 128, 256, 512), use_film: bool = True,
                 d_state: int = 64, d_action: int = 2):
        super().__init__()
        self.use_film = use_film
        self.d_state = d_state
        self.d_action = d_action

        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )

        self.use_posenc = True
        self.pos_dim = 128  # tune 64–256 if you like
        self.pos_enc = SinusoidalPosEnc1D(self.pos_dim)
        self.pos_proj = nn.Linear(self.pos_dim, channels[0])

        self.input_proj = nn.Conv1d(in_feature_dim, channels[0], 7, padding=3)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(channels) - 1):
            if use_film:
                self.enc_blocks.append(FiLMResidualBlock(channels[i], channels[i+1], time_embed_dim, film_dim))
            else:
                self.enc_blocks.append(ResidualBlock(channels[i], channels[i+1], time_embed_dim))
            self.downs.append(nn.Conv1d(channels[i+1], channels[i+1], 4, stride=2, padding=1))

        # Bottleneck
        if use_film:
            self.mid_block1 = FiLMResidualBlock(channels[-1], channels[-1], time_embed_dim, film_dim)
            self.mid_block2 = FiLMResidualBlock(channels[-1], channels[-1], time_embed_dim, film_dim)
        else:
            self.mid_block1 = ResidualBlock(channels[-1], channels[-1], time_embed_dim)
            self.mid_block2 = ResidualBlock(channels[-1], channels[-1], time_embed_dim)
        self.attn = AttentionBlock(channels[-1])

        # Decoder
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        rev_channels = list(reversed(channels))
        for i in range(len(rev_channels) - 1):
            self.ups.append(nn.ConvTranspose1d(rev_channels[i], rev_channels[i+1], 4, stride=2, padding=1))
            if use_film:
                self.dec_blocks.append(FiLMResidualBlock(rev_channels[i] + rev_channels[i+1], rev_channels[i+1],
                                                         time_embed_dim, film_dim))
            else:
                self.dec_blocks.append(ResidualBlock(rev_channels[i] + rev_channels[i+1], rev_channels[i+1],
                                                     time_embed_dim))

        # Separate heads for gesture and state prediction
        self.gesture_head = nn.Conv1d(channels[0], d_action, 3, padding=1)
        self.state_head = nn.Conv1d(channels[0], d_state, 3, padding=1)
        self.pen_head = nn.Sequential(
            nn.Conv1d(channels[0], 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(32, 1, 3, padding=1)  # logits
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                gamma: torch.Tensor = None, beta: torch.Tensor = None,
                sequence_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        # x: [B, L, C_in]
        t = t.float()
        t_emb = self.time_embed(t)

        x = x.transpose(1, 2)            # [B, C_in, L]
        x = self.input_proj(x)           # [B, C0, L]

        # We’ll compute a consistent frame repetition plan `reps` and use it
        # both for positional encodings and FiLM expansion.
        B, C0, L = x.shape
        reps = None
        if sequence_length is not None and sequence_length > 0 and self.use_film and (gamma is not None):
            # Partition steps across the N action segments (t = 0..N-1), not across N+1 frames.
            num_segments = sequence_length  # number of action segments
            base = L // num_segments
            rem  = L - base * num_segments
            reps = torch.full((num_segments,), base, device=x.device, dtype=torch.long)
            if rem > 0:
                reps[-1] += rem  # dump leftovers into the last action segment

        # ----- Positional encodings with the same frame partitioning -----
        if self.use_posenc:
            if reps is not None:
                # Build frame_idx by repeating [0..F-1] with `reps`
                frame_ids = torch.arange(reps.numel(), device=x.device)
                frame_idx = torch.repeat_interleave(frame_ids, reps)           # [L]
                frame_idx = frame_idx.unsqueeze(0).expand(B, -1)               # [B,L]

                # Build local_idx: within each frame, 0..reps[f]-1, concatenated
                local_chunks = [torch.arange(int(r), device=x.device) for r in reps.tolist()]
                local_idx = torch.cat(local_chunks, dim=0).unsqueeze(0).expand(B, -1)  # [B,L]
            else:
                # Fallback: absolute positions only
                frame_idx = torch.zeros(B, L, device=x.device)  # unused
                local_idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)

            pe = self.pos_enc(local_idx)  # local timing bias
            if reps is not None:
                pe = pe + self.pos_enc(frame_idx)  # add coarse frame id bias
            pe = self.pos_proj(pe).transpose(1, 2)  # [B,C0,L]
            x = x + pe

        # ----- Per-frame FiLM -> per-step FiLM aligned to action segments -----
        gamma_steps = beta_steps = None
        if self.use_film and (gamma is not None) and (beta is not None):
            if reps is not None:
                # gamma/beta are per FRAME for [0..N] frames. Build action-aligned FiLM by
                # combining the pair (s_t, s_{t+1}) for each action segment t in [0..N-1].
                gamma_t   = gamma[:, :-1, :]   # [B, N, D]      uses s_t
                gamma_tp1 = gamma[:, 1:,  :]   # [B, N, D]      uses s_{t+1}
                beta_t    = beta[:, :-1,  :]
                beta_tp1  = beta[:, 1:,   :]

                # Minimal, parameter-free combine; you can swap for a small Linear if desired.
                gamma_action = 0.5 * (gamma_t + gamma_tp1)   # [B, N, D]
                beta_action  = 0.5 * (beta_t  + beta_tp1)    # [B, N, D]

                # Now expand each action segment t across its Traj steps using reps (length N).
                gamma_steps = torch.repeat_interleave(gamma_action, reps, dim=1)  # [B, L, D]
                beta_steps  = torch.repeat_interleave(beta_action,  reps, dim=1)  # [B, L, D]
            else:
                # Fallback: broadcast the mean
                g_avg = gamma.mean(dim=1, keepdim=True)
                b_avg = beta.mean(dim=1, keepdim=True)
                gamma_steps = g_avg.expand(-1, L, -1)
                beta_steps  = b_avg.expand(-1, L, -1)

        # Helper: resize FiLM to a target length (for every UNet scale)
        def _resize_film(film_steps: Optional[torch.Tensor], L_target: int) -> Optional[torch.Tensor]:
            if film_steps is None or film_steps.shape[1] == L_target:
                return film_steps
            # [B,L,F] -> [B,F,L] for interpolation on length dim, mode=nearest keeps alignment to frames
            return torch.nn.functional.interpolate(
                film_steps.transpose(1, 2),
                size=L_target,
                mode="nearest"
            ).transpose(1, 2)

        skips = []
        for block, down in zip(self.enc_blocks, self.downs):
            if self.use_film:
                L_cur = x.shape[2]
                g_cur = _resize_film(gamma_steps, L_cur)
                b_cur = _resize_film(beta_steps,  L_cur)
                x = block(x, t_emb, g_cur, b_cur)
            else:
                x = block(x, t_emb)
            skips.append(x)
            x = down(x)

        # Bottleneck
        if self.use_film:
            L_cur = x.shape[2]
            g_cur = _resize_film(gamma_steps, L_cur)
            b_cur = _resize_film(beta_steps,  L_cur)
            x = self.mid_block1(x, t_emb, g_cur, b_cur)
            x = self.attn(x)
            # length may change slightly after attn norm/pads, re-pull length
            L_cur = x.shape[2]
            g_cur = _resize_film(gamma_steps, L_cur)
            b_cur = _resize_film(beta_steps,  L_cur)
            x = self.mid_block2(x, t_emb, g_cur, b_cur)
        else:
            x = self.mid_block1(x, t_emb)
            x = self.attn(x)
            x = self.mid_block2(x, t_emb)

        # Decoder
        for up, block, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            x = up(x)
            if x.shape[2] != skip.shape[2]:
                if x.shape[2] > skip.shape[2]:
                    x = x[:, :, :skip.shape[2]]
                else:
                    x = F.pad(x, (0, skip.shape[2] - x.shape[2]))
            x = torch.cat([x, skip], dim=1)
            if self.use_film:
                L_cur = x.shape[2]
                g_cur = _resize_film(gamma_steps, L_cur)
                b_cur = _resize_film(beta_steps,  L_cur)
                x = block(x, t_emb, g_cur, b_cur)
            else:
                x = block(x, t_emb)

        gesture_v = self.gesture_head(x).transpose(1, 2)  # [B, L, d_action]
        state_v = self.state_head(x).transpose(1, 2)      # [B, L, d_state]
        pen_logits = self.pen_head(x).transpose(1, 2)     # [B, L, 1]
        return {"gesture_velocity": gesture_v, "state_velocity": state_v, "pen_logits": pen_logits}


class GestureFlowModel(nn.Module):
    """Joint trajectory flow matching model for states and actions."""

    def __init__(self, in_feature_dim: int, time_embed_dim: int = 128,
                 channels: tuple = None, max_sequence_length: int = 1000,
                 film_dim: int = 128, d_state: int = 64, use_film: bool = True):
        super().__init__()
        self.in_feature_dim = in_feature_dim
        self.max_sequence_length = max_sequence_length
        self.d_state = d_state
        self.d_action = in_feature_dim - d_state  # Calculate d_action from total dim
        self.use_film = use_film

        if channels is None:
            channels = (64, 128, 256, 512)

        # FiLM from raw latent frames (only if enabled) - matching diffusion model
        if use_film:
            self.film_cross_attn = FiLMCrossAttention(d_model=film_dim, d_latent=1024)
        else:
            self.film_cross_attn = None

        # UNet backbone (FiLM-aware or regular)
        self.unet = GestureFlowUNet(
            in_feature_dim=in_feature_dim,
            time_embed_dim=time_embed_dim,
            film_dim=film_dim if use_film else None,
            channels=channels,
            use_film=use_film,
            d_state=d_state,
            d_action=self.d_action
        )

        # For optional logging
        self.last_film_gamma = None
        self.last_film_beta = None
        self.last_gamma_norm = None
        self.last_beta_norm = None

    def _gamma_beta(self, latent_frames: torch.Tensor, sequence_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_film:
            return None, None
        gamma, beta = self.film_cross_attn(latent_frames, sequence_length)  # [B,F,film_dim]
        # store for histograms if you want
        with torch.no_grad():
            self.last_film_gamma = gamma.detach().clone()
            self.last_film_beta = beta.detach().clone()
            self.last_gamma_norm = torch.norm(gamma, dim=-1)  # [B,1]
            self.last_beta_norm = torch.norm(beta, dim=-1)    # [B,1]
        return gamma, beta

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                latent_frames: torch.Tensor = None, sequence_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """x_t is the joint sequence [B,L,C_joint]; latent_frames are raw JEPA patches [B,F,N,D] (ignored if no FiLM)."""
        if self.use_film:
            gamma, beta = self._gamma_beta(latent_frames, sequence_length)
            unet_output = self.unet(x_t, t, gamma, beta, sequence_length=sequence_length)
        else:
            unet_output = self.unet(x_t, t, sequence_length=sequence_length)

        # Concatenate separate head outputs for backward compatibility
        # Joint velocity: [state_velocity, gesture_velocity] -> [B, L, d_state + d_action]
        velocity = torch.cat([unet_output['state_velocity'], unet_output['gesture_velocity']], dim=-1)

        return {
            'velocity': velocity,
            'pen_logits': unet_output['pen_logits'],
            # Also return separate components for potential future use
            'state_velocity': unet_output['state_velocity'],
            'gesture_velocity': unet_output['gesture_velocity']
        }

    def forward_pen_prediction(self, clean_joint: torch.Tensor, t: torch.Tensor,
                               latent_frames: torch.Tensor = None, sequence_length: Optional[int] = None) -> torch.Tensor:
        if self.use_film:
            gamma, beta = self._gamma_beta(latent_frames, sequence_length)
            out = self.unet(clean_joint, t, gamma, beta, sequence_length=sequence_length)
        else:
            out = self.unet(clean_joint, t, sequence_length=sequence_length)
        return out["pen_logits"]


class FlowSampler:
    """ODE sampler for joint trajectory flow matching with Euler/Heun integration."""

    def __init__(self, model, device="cuda", method="heun"):
        self.model = model
        self.device = device
        self.method = method

    @torch.no_grad()
    def sample(self, shape, latent_frames=None, sequence_length=None,
               state_per_step=None,    # [B, L, d_state]
               anchor_states=True, steps=20, d_state=None):
        B, L_joint, C_joint = shape
        device = self.device

        # Full sample starts from noise (actions + states)
        x = torch.randn(B, L_joint, C_joint, device=device)
        dt = 1.0 / steps

        # If we’re inpainting states, draw a FIXED base noise x0_state once per sample.
        if anchor_states and (state_per_step is not None) and (d_state is not None):
            x0_state = torch.randn(B, L_joint, d_state, device=device)

        for k in range(steps):
            t0 = k / steps
            t1 = (k + 1) / steps  # time *after* this update

            t0_vec = torch.full((B,), t0, device=device)
            v = self.model(x, t0_vec, latent_frames, sequence_length)['velocity']

            if self.method == "euler" or k == steps - 1:
                x = x + dt * v
            else:
                x_euler = x + dt * v
                t1_vec = torch.full((B,), t1, device=device)
                v1 = self.model(x_euler, t1_vec, latent_frames, sequence_length)['velocity']
                x = x + 0.5 * dt * (v + v1)

            # === DIFFUSER-STYLE INPAINTING FOR STATES (rectified flow path, avoid distribution shift) ===
            if anchor_states and (state_per_step is not None) and (d_state is not None):
                # Rebuild the state slice at the current time t1 using the same forward mix used in training.
                t1_ = torch.full((B, 1, 1), t1, device=device)   # [B,1,1]
                x_state_t = (1.0 - t1_) * x0_state + t1_ * state_per_step  # [B,L,d_state]
                # Replace only the state channels
                x[:, :, :d_state] = x_state_t

        return x





# Model Loading and Versioning Utilities

def load_gesture_flow_model(checkpoint_path: str):
    """
    Load joint trajectory flow matching model from checkpoint.

    Args:
        checkpoint_path: Path to the flow model checkpoint

    Returns:
        Loaded GestureFlowModel instance
    """
    return GestureFlowModel.from_checkpoint(checkpoint_path)


def save_gesture_flow_checkpoint(model, checkpoint_path: str,
                                trainer_state: Optional[Dict] = None):
    """
    Save joint trajectory flow matching model checkpoint.

    Args:
        model: GestureFlowModel to save
        checkpoint_path: Path to save checkpoint
        trainer_state: Optional trainer state dict
    """
    model_type = "gesture_flow_joint"
    model_version = "3.0"
    conditioning_type = "joint_trajectory"

    # Store model config
    model_config = {
        "in_feature_dim": model.in_feature_dim,
        "time_embed_dim": 128,  # Default
        "channels": (64, 128, 256, 512),  # Default from model
        "max_sequence_length": model.max_sequence_length,
        "d_state": model.d_state,
        "film_dim": model.film_cross_attn.d_model if model.use_film else None,
        "use_film": model.use_film,
        "include_pen_in_flow": getattr(model, "include_pen_in_flow", False)
    }

    # Create checkpoint dict
    checkpoint = {
        # Model versioning
        "model_type": model_type,
        "model_version": model_version,
        "conditioning_type": conditioning_type,

        # Model state
        "model_state_dict": model.state_dict(),
        "model_config": model_config,

        # Trainer state (if provided)
        **(trainer_state or {})
    }

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved {model_type} v{model_version} to {checkpoint_path}")


# Add from_checkpoint method to GestureFlowModel
def _add_from_checkpoint_method():
    """Add from_checkpoint class method to GestureFlowModel"""

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str):
        """Load joint trajectory flow model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Create model with saved hyperparameters
        model_config = checkpoint.get("model_config", {})
        model = cls(
            in_feature_dim=model_config.get("in_feature_dim", 66),
            time_embed_dim=model_config.get("time_embed_dim", 128),
            channels=tuple(model_config.get("channels", (64, 128, 256, 512))),
            max_sequence_length=model_config.get("max_sequence_length", 1000),
            film_dim=model_config.get("film_dim", 128),
            d_state=model_config.get("d_state", 64),
            use_film=model_config.get("use_film", True),   # <- add this
        )

        # Load model state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        return model

    # Add method to class
    GestureFlowModel.from_checkpoint = from_checkpoint

# Apply the method addition
_add_from_checkpoint_method()


def sample_rectified_flow_batch_joint(x1_joint: torch.Tensor,
                                       noisy: bool = False,
                                       sigma: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample rectified flow training batch for joint trajectory.

    Args:
        x1_joint: [B, L_joint, C_joint] joint trajectory (states + actions)
    Returns:
        x_t: [B, L_joint, C_joint] -> joint trajectory at time t
        t:   [B]                   -> float32 in [0,1]
        v_tar: [B, L_joint, C_joint] -> target velocity field
    """
    device = x1_joint.device
    B, L, C = x1_joint.shape
    x0 = torch.randn_like(x1_joint)                 # base Gaussian
    t = torch.rand(B, device=device)                # uniform [0,1]
    t_ = t[:, None, None]                           # [B,1,1]

    if not noisy:
        x_t = (1 - t_) * x0 + t_ * x1_joint
        v_tar = (x1_joint - x0)                     # constant in t
    else:
        eps = torch.randn_like(x1_joint)
        g = sigma * t_ * (1 - t_)                   # smoothing
        gp = sigma * (1 - 2 * t_)
        x_t = (1 - t_) * x0 + t_ * x1_joint + g * eps
        v_tar = (x1_joint - x0) + gp * eps

    return x_t, t, v_tar
