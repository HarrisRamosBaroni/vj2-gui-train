"""
Diffuser-style Flow Model (Temporal U-Net, joint states+actions, ε-prediction)
-------------------------------------------------------------------------------

This module implements a Diffuser-aligned model for gesture planning with pooled
JEPA states and action trajectories as channels in a single temporal trajectory
tensor τ. The network performs ε-prediction over all channels jointly and
supports inpainting-style hard constraints and soft guidance (product of
experts) during sampling.

Key features
- Input as a 2-D trajectory τ ∈ ℝ^{B×T×(D_s+D_a)} (time-major inside the model)
- States = pooled JEPA vectors (D_s); Actions = gesture (x, y, pen) (D_a)
- Temporal U-Net backbone with GroupNorm + Mish + timestep embedding
- Cosine noise schedule; training with ε-MSE objective
- Boundary-only supervision for states via masks; full supervision for actions
- Inpainting constraints for start/goal/waypoints at every reverse step
- Soft guidance hooks to implement product-of-experts (PoE)
- Variable horizon (fully convolutional in time)

No training script is included by request.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utilities: time embeddings, activations
# -----------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """Standard sinusoidal embedding for diffusion timestep index t (integer or float).

    Args:
        dim: embedding dimension
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        self.register_buffer("freqs", torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1)
        ), persistent=False)
        self.out = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.Mish(), nn.Linear(dim * 2, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, num_steps] or normalized [0,1]
        if t.dim() == 2:
            t = t.squeeze(-1)
        args = t.float().unsqueeze(-1) * self.freqs.to(t.device).unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, emb.new_zeros(emb.size(0), 1)], dim=-1)
        return self.out(emb)


# -----------------------------------------------------------------------------
# Temporal U-Net (Diffuser-style)
# -----------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.Mish()
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T); t_emb: (B, t_dim)
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        # inject timestep
        h = h + self.t_proj(t_emb)[:, :, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.tconv = nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tconv(x)


class TemporalUNet(nn.Module):
    """Temporal U-Net that maps noisy trajectories to predicted ε over channels.

    Args:
        in_ch: input channels (D_s + D_a)
        base_ch: width of first stage
        ch_mult: channel multipliers per down block
        num_res_blocks: residual blocks per stage
        t_dim: timestep embedding dim
        groups: GroupNorm groups
    """
    def __init__(
        self,
        in_ch: int,
        base_ch: int = 64,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        t_dim: int = 128,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.time_mlp = SinusoidalTimestepEmbedding(t_dim)

        # input conv
        self.input = nn.Conv1d(in_ch, base_ch, kernel_size=7, padding=3)

        # down path
        downs: List[nn.Module] = []
        ch = base_ch
        self.down_channels = [ch]
        for mult in ch_mult:
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                downs.append(ResBlock1D(ch, out_ch, t_dim, groups))
                ch = out_ch
                self.down_channels.append(ch)
            downs.append(Downsample1D(ch))
        self.down = nn.ModuleList(downs)

        # bottleneck
        self.mid1 = ResBlock1D(ch, ch, t_dim, groups)
        self.mid2 = ResBlock1D(ch, ch, t_dim, groups)

        # up path (mirror)
        ups: List[nn.Module] = []
        for mult in reversed(ch_mult):
            out_ch = base_ch * mult
            ups.append(Upsample1D(ch))
            for _ in range(num_res_blocks):
                ups.append(ResBlock1D(ch + out_ch, out_ch, t_dim, groups))
                ch = out_ch
        self.up = nn.ModuleList(ups)

        # output head
        self.out = nn.Sequential(
            nn.GroupNorm(groups, ch), nn.Mish(), nn.Conv1d(ch, in_ch, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, t_cont: torch.Tensor) -> torch.Tensor:
        """Forward.
        Args:
            x: (B, T, C)
            t_cont: (B,) continuous or integer timestep for diffusion
        Returns:
            eps: (B, T, C) predicted noise
        """
        x = x.transpose(1, 2)  # (B, C, T)
        t_emb = self.time_mlp(t_cont)

        h = self.input(x)
        skips = []

        for m in self.down:
            if isinstance(m, ResBlock1D):
                h = m(h, t_emb)
                skips.append(h)
            else:
                h = m(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for m in self.up:
            if isinstance(m, Upsample1D):
                h = m(h)
                # align with next skip length if odd
                if len(skips) > 0:
                    tgt = skips[-1]
                    if h.size(-1) != tgt.size(-1):
                        diff = tgt.size(-1) - h.size(-1)
                        if diff > 0:
                            h = F.pad(h, (0, diff))
                        else:
                            h = h[..., :tgt.size(-1)]
            else:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = m(h, t_emb)

        out = self.out(h).transpose(1, 2)  # (B, T, C)
        return out


# -----------------------------------------------------------------------------
# Diffusion schedule (cosine) and helpers
# -----------------------------------------------------------------------------

@dataclass
class CosineSchedule:
    num_steps: int = 1000
    s: float = 0.008  # per Nichol & Dhariwal

    def _alphas_cumprod(self, device) -> torch.Tensor:
        steps = self.num_steps
        t = torch.linspace(0, steps, steps + 1, device=device) / steps
        fn = lambda u: torch.cos((u + self.s) / (1 + self.s) * math.pi / 2) ** 2
        alphas_bar = fn(t) / fn(torch.tensor(0.0, device=device))
        return alphas_bar.clamp(min=1e-5, max=1.0)

    def buffers(self, device) -> Dict[str, torch.Tensor]:
        ab = self._alphas_cumprod(device)  # (S+1,)
        beta = 1 - (ab[1:] / ab[:-1]).clamp(min=1e-5, max=1.0)
        alpha = 1.0 - beta
        return {
            "alpha": alpha,                     # (S,)
            "alpha_bar": ab[1:],                # (S,)
            "alpha_bar_prev": ab[:-1],         # (S,)
            "beta": beta,
        }

    # q(x_t|x_0)
    def add_noise(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer timesteps in [0, S-1]; gather alpha_bar[t]
        a = alpha_bar.index_select(0, t.long()).view(-1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * eps


# -----------------------------------------------------------------------------
# Core model: joints states+actions, ε-prediction, masks, sampling, guidance
# -----------------------------------------------------------------------------

class DiffuserStyleFlowModel(nn.Module):
    """Joint denoiser for pooled JEPA states + gesture actions (Diffuser-aligned).

    I/O contracts
    ------------
    Clean trajectory τ0: shape (B, T, C) with C = D_s + D_a.
      - channel slice [ :D_s]  = states (pooled JEPA per micro-step via hold)
      - channel slice [D_s: ]  = actions (x, y, pen), normalized
    Boundary masks:
      - M_s: (B, T, 1) with 1 at frame-boundary micro-steps (state GT known), else 0
      - M_a: (B, T, 1) usually all ones

    Training forward returns ε̂ over all channels and utility loss functions.
    Sampling supports inpainting constraints and PoE-style guidance.
    """

    def __init__(
        self,
        d_state: int,                # D_s (pooled JEPA dim you choose, e.g., 1024 or projected)
        d_action: int,               # D_a (e.g., 3 for [x,y,pen])
        base_channels: int = 64,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        time_embed_dim: int = 128,
        num_diffusion_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.d_state = d_state
        self.d_action = d_action
        self.channels = d_state + d_action

        self.unet = TemporalUNet(
            in_ch=self.channels,
            base_ch=base_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            t_dim=time_embed_dim,
        )
        self.schedule = CosineSchedule(num_steps=num_diffusion_steps)

    # ------------------------------
    # Trajectory building utilities
    # ------------------------------
    @staticmethod
    def hold_state_over_microsteps(pooled_states: torch.Tensor, K: int, T: Optional[int] = None) -> torch.Tensor:
        """Repeat each frame state over K micro-steps (piecewise-constant hold).
        Args:
            pooled_states: (B, F, D_s)
            K: steps per gesture segment
            T: optional explicit T; otherwise T = (F-1)*K
        Returns:
            S: (B, T, D_s)
        """
        B, F, Ds = pooled_states.shape
        if T is None:
            T = (F - 1) * K
        # Use frames 0..F-2 each repeated K times
        S = pooled_states[:, :-1, :].unsqueeze(2).expand(B, F - 1, K, Ds).reshape(B, T, Ds)
        return S

    @staticmethod
    def pack_trajectory(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Concatenate states and actions along channel dim.
        Args:
            states:  (B, T, D_s)
            actions: (B, T, D_a)
        Returns: τ (B, T, D_s + D_a)
        """
        return torch.cat([states, actions], dim=-1)

    @staticmethod
    def make_boundary_mask(T: int, K: int, device: torch.device, B: int = 1) -> torch.Tensor:
        """Mask with ones at frame boundaries: t ∈ {0, K, 2K, ...}.
        Returns: (B, T, 1)
        """
        m = torch.zeros(T, 1, device=device)
        m[::K] = 1.0
        return m.unsqueeze(0).expand(B, T, 1)

    # ------------------------------
    # Forward / training (ε-prediction)
    # ------------------------------
    def forward(self, x_noisy: torch.Tensor, t_index: torch.Tensor) -> torch.Tensor:
        """Predict ε over all channels.
        Args:
            x_noisy: (B, T, C)
            t_index: (B,) scalar diffusion steps in [0, S-1]
        Returns:
            eps_pred: (B, T, C)
        """
        return self.unet(x_noisy, t_index)

    def loss_eps_mse(
        self,
        eps_pred: torch.Tensor,
        eps_true: torch.Tensor,
        m_state: Optional[torch.Tensor] = None,
        m_action: Optional[torch.Tensor] = None,
        w_state: float = 1.0,
        w_action: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute masked ε-MSE for state and action channel slices.
        Args:
            eps_pred/eps_true: (B, T, C)
            m_state/m_action:  (B, T, 1) masks; if None, use all ones
        Returns:
            dict with total loss and components
        """
        B, T, C = eps_pred.shape
        Ds = self.d_state
        Da = self.d_action
        if m_state is None:
            m_state = eps_pred.new_ones(B, T, 1)
        if m_action is None:
            m_action = eps_pred.new_ones(B, T, 1)
        se_s = (eps_pred[..., :Ds] - eps_true[..., :Ds]) ** 2
        se_a = (eps_pred[..., Ds:] - eps_true[..., Ds:]) ** 2
        loss_s = (se_s * m_state).sum() / (m_state.sum() * max(Ds, 1) + 1e-8)
        loss_a = (se_a * m_action).sum() / (m_action.sum() * max(Da, 1) + 1e-8)
        total = w_state * loss_s + w_action * loss_a
        return {"loss": total, "loss_state": loss_s, "loss_action": loss_a}

    # ------------------------------
    # Noise helpers for training
    # ------------------------------
    def q_sample(self, x0: torch.Tensor, t_index: torch.Tensor, eps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward diffusion q(x_t|x_0). Returns x_t and the noise used.
        Args:
            x0: (B, T, C)
            t_index: (B,) in [0, S-1]
        """
        device = x0.device
        buf = self.schedule.buffers(device)
        if eps is None:
            eps = torch.randn_like(x0)
        x_t = self.schedule.add_noise(x0, eps, t_index, buf["alpha_bar"])  # (B, T, C)
        return x_t, eps

    # ------------------------------
    # Sampling with inpainting + PoE guidance (DDIM-like)
    # ------------------------------
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int],
        steps: int,
        eta: float = 0.0,
        constraints: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        guidance_fns: Optional[List[Callable[[torch.Tensor, int], torch.Tensor]]] = None,
        # constraints: list of (mask, value) pairs, each (B, T, C) where mask ∈ {0,1}
        # guidance_fns: each returns ∂J/∂x with shape (B, T, C) for current estimate
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        B, T, C = shape
        x = torch.randn(B, T, C, device=device)

        # choose uniform indices over [0, S-1]
        S = self.schedule.num_steps
        # DDIM schedule: integer indices with equal stride
        t_seq = torch.linspace(S - 1, 0, steps, device=device).long()
        buf = self.schedule.buffers(device)
        alpha_bar = buf["alpha_bar"]

        for i, t in enumerate(t_seq):
            t_b = t.expand(B)
            eps_pred = self.forward(x, t_b)

            # DDIM params
            a_t = alpha_bar[t]
            a_prev = alpha_bar[t_seq[i + 1]] if i + 1 < len(t_seq) else alpha_bar[0]
            sqrt_a_t = a_t.sqrt()
            sqrt_one_minus_a_t = (1 - a_t).sqrt()
            # predict x0
            x0_pred = (x - sqrt_one_minus_a_t * eps_pred) / (sqrt_a_t + 1e-8)

            # guidance (PoE): x0 ← x0 + γ * Σ ∇J(x0)
            if guidance_fns:
                grad_sum = torch.zeros_like(x0_pred)
                for gfn in guidance_fns:
                    grad_sum = grad_sum + gfn(x0_pred, int(t))
                # simple step size (could be scheduled)
                gamma = 1.0
                x0_pred = x0_pred + gamma * grad_sum

            # compute x_{t-1} (DDIM)
            dir_xt = (a_prev.sqrt()) * x0_pred
            if eta == 0.0:
                x = dir_xt + (1 - a_prev).sqrt() * eps_pred
            else:
                # stochasticity per DDIM
                sigma_t = eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).sqrt()
                noise = torch.randn_like(x)
                x = dir_xt + sigma_t * noise + ((1 - a_prev - sigma_t**2).clamp(min=0.0)).sqrt() * eps_pred

            # inpainting constraints (hard): overwrite masked positions each step
            if constraints:
                for mask, value in constraints:
                    # mask/value are (B, T, C)
                    x = x * (1 - mask) + value * mask

        return x

    # ------------------------------
    # Convenience: build constraints for start/goal/waypoints
    # ------------------------------
    @staticmethod
    def constraint_tensor(
        x_like: torch.Tensor,
        positions: List[int],
        values: torch.Tensor,
        channels: Optional[slice] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (mask, value) tensors for inpainting.
        Args:
            x_like: (B, T, C) reference for shape/device
            positions: list of time indices to constrain
            values: (B, len(positions), C') values to set
            channels: optional slice of channels to constrain (e.g., slice(0, D_s) for states)
        Returns:
            (mask, value) both (B, T, C)
        """
        B, T, C = x_like.shape
        mask = torch.zeros(B, T, C, device=x_like.device)
        val = torch.zeros(B, T, C, device=x_like.device)
        if channels is None:
            ch_slice = slice(0, C)
            Cprime = C
        else:
            ch_slice = channels
            Cprime = (channels.stop - channels.start)
        assert values.shape[1] == len(positions)
        assert values.shape[-1] == Cprime
        for j, t in enumerate(positions):
            mask[:, t, ch_slice] = 1.0
            val[:, t, ch_slice] = values[:, j, :]
        return mask, val


# -----------------------------------------------------------------------------
# Example usage (pseudocode, not executed here):
# -----------------------------------------------------------------------------
# model = DiffuserStyleFlowModel(d_state=1024, d_action=3)
#
# # Build τ0 from pooled states and actions
# S = model.hold_state_over_microsteps(pooled_states, K=250)  # (B, T, 1024)
# A = norm_actions(actions)                                   # (B, T, 3)
# tau0 = model.pack_trajectory(S, A)                          # (B, T, 1027)
# M_s = model.make_boundary_mask(T=tau0.size(1), K=250, device=tau0.device, B=tau0.size(0))
# M_a = torch.ones_like(M_s)
#
# # Training step (ε-MSE):
# t = torch.randint(0, model.schedule.num_steps, (B,), device=tau0.device)
# x_t, eps = model.q_sample(tau0, t)
# eps_pred = model(x_t, t)
# losses = model.loss_eps_mse(eps_pred, eps, m_state=M_s, m_action=M_a)
#
# # Sampling with start-state inpainting:
# x_shape = (B, T, 1027)
# start_vals = S[:, :1, :]  # or measured JEPA pooled state at t=0
# mask0, val0 = model.constraint_tensor(
#     x_like=torch.zeros(*x_shape, device=S.device),
#     positions=[0],
#     values=start_vals,                 # (B, 1, 1024)
#     channels=slice(0, 1024),          # constrain only state channels
# )
# x_gen = model.sample(x_shape, steps=50, constraints=[(mask0, val0)], guidance_fns=None)
# -----------------------------------------------------------------------------
