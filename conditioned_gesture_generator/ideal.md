let’s rewrite your backbone so that:

* It’s a **true U-Net** (encoder–decoder pyramid with down/upsampling + skip connections).
* Includes a **bottleneck with attention** (like the paper).
* Keeps your **FiLM conditioning** (γ/β modulation from latent frames).
* Still **strips pen channel during diffusion** (coordinates diffused, pen predicted from shared features).

Here’s a clean modified implementation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


# -------------------
# Positional embedding
# -------------------
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() == 2:
            timesteps = timesteps.squeeze(-1)

        half_dim = self.embed_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb_scale)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


# -------------------
# FiLM cross-attention
# -------------------
class FiLMCrossAttention(nn.Module):
    def __init__(self, d_model: int, d_latent: int = 1024, nhead: int = 8):
        super().__init__()
        self.gesture_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.key_proj = nn.Linear(d_latent, d_model)
        self.value_proj = nn.Linear(d_latent, d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.scale_proj = nn.Linear(d_model, d_model)
        self.shift_proj = nn.Linear(d_model, d_model)

    def forward(self, latent_frames: torch.Tensor,
                sequence_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_plus_1, N_patches, D = latent_frames.shape
        if sequence_length is not None:
            latent_frames = latent_frames[:, :sequence_length + 1]
        seq_len = latent_frames.shape[1]
        latent_flat = latent_frames.reshape(B, seq_len * N_patches, D)

        keys = self.key_proj(latent_flat)
        values = self.value_proj(latent_flat)
        query = self.gesture_query.expand(B, -1, -1)

        attended, _ = self.attn(query, keys, values)
        gamma = self.scale_proj(attended)  # [B,1,d_model]
        beta = self.shift_proj(attended)   # [B,1,d_model]
        return gamma, beta


# -------------------
# Basic UNet blocks
# -------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, film_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_channels)
        self.film_scale = nn.Linear(film_dim, out_channels)
        self.film_shift = nn.Linear(film_dim, out_channels)

        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb, gamma, beta):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = h + self.time_proj(t_emb)[:, :, None]
        h = self.norm2(h)
        h = F.silu(h)

        gamma = self.film_scale(gamma.squeeze(1))[:, :, None]
        beta = self.film_shift(beta.squeeze(1))[:, :, None]
        h = self.conv2(h)
        h = gamma * h + beta

        return h + self.residual(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: [B,C,L] → transpose to [B,L,C]
        x_in = x.transpose(1, 2)
        h = self.norm(x_in)
        h, _ = self.attn(h, h, h)
        return (h + x_in).transpose(1, 2)


# -------------------
# Full U-Net backbone
# -------------------
class GestureDiffusionUNet(nn.Module):
    def __init__(self, coordinate_dim=2, d_model=512, time_embed_dim=128, channels=(64,128,256,512)):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim*2),
            nn.SiLU(),
            nn.Linear(time_embed_dim*2, time_embed_dim)
        )

        self.input_proj = nn.Conv1d(coordinate_dim, channels[0], 7, padding=3)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(channels)-1):
            self.enc_blocks.append(ResidualBlock(channels[i], channels[i+1], time_embed_dim, d_model))
            self.downs.append(nn.Conv1d(channels[i+1], channels[i+1], 4, stride=2, padding=1))

        # Bottleneck
        self.mid_block1 = ResidualBlock(channels[-1], channels[-1], time_embed_dim, d_model)
        self.attn = AttentionBlock(channels[-1])
        self.mid_block2 = ResidualBlock(channels[-1], channels[-1], time_embed_dim, d_model)

        # Decoder
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        rev_channels = list(reversed(channels))
        for i in range(len(rev_channels)-1):
            self.ups.append(nn.ConvTranspose1d(rev_channels[i], rev_channels[i+1], 4, stride=2, padding=1))
            self.dec_blocks.append(ResidualBlock(rev_channels[i], rev_channels[i+1], time_embed_dim, d_model))

        # Heads
        self.coord_head = nn.Conv1d(channels[0], 2, 3, padding=1)
        self.pen_head = nn.Sequential(
            nn.Conv1d(channels[0], 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, t, gamma, beta):
        t_emb = self.time_embed(t)

        # Input
        x = x[..., :2].transpose(1,2)   # only (x,y)
        x = self.input_proj(x)

        # Encoder
        skips = []
        for block, down in zip(self.enc_blocks, self.downs):
            x = block(x, t_emb, gamma, beta)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.mid_block1(x, t_emb, gamma, beta)
        x = self.attn(x)
        x = self.mid_block2(x, t_emb, gamma, beta)

        # Decoder
        for up, block, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb, gamma, beta)

        # Heads
        coords = self.coord_head(x).transpose(1,2)
        pen = self.pen_head(x).transpose(1,2)
        return {"coordinates": coords, "pen_state": pen}
```

---

### 🔑 Changes vs. your old model

* Now a **true U-Net**:

  * Encoder: residual + downsampling.
  * Bottleneck: residual + attention.
  * Decoder: upsampling + skip connections.
* **FiLM conditioning**: still applied at every residual block.
* **Pen stripped**: input always ignores pen channel, only `(x,y)` are diffused.
* **Heads**: coordinate noise head + sigmoid pen head.

---

