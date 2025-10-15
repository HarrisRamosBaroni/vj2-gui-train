import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, Union
import os
from pathlib import Path


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal embeddings for timesteps.

        Args:
            timesteps: Tensor of shape [B] or [B, 1]
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


class FiLMCrossAttention(nn.Module):
    """FiLM conditioning via cross-attention to latent frames with variable sequence length support."""

    def __init__(self, d_model: int, d_latent: int = 1024, nhead: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.nhead = nhead

        # Learned query for gesture context
        self.gesture_query = nn.Parameter(torch.randn(1, 1, d_model))

        # Project latent frames to key/value space
        self.latent_key_proj = nn.Linear(d_latent, d_model)
        self.latent_value_proj = nn.Linear(d_latent, d_model)

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )

        # FiLM parameter extraction
        self.film_scale = nn.Linear(d_model, d_model)
        self.film_shift = nn.Linear(d_model, d_model)

        # Initialize with small weights
        nn.init.normal_(self.film_scale.weight, std=0.02)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.normal_(self.film_shift.weight, std=0.02)
        nn.init.zeros_(self.film_shift.bias)

    def forward(self, latent_frames: torch.Tensor,
                sequence_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate FiLM parameters from latent frames with variable length masking.

        Args:
            latent_frames: Tensor of shape [B, N_max+1, N_patches, D] where N_max is max sequence length
            sequence_length: Actual sequence length to use (1 to N_max). If None, use all frames.
        Returns:
            gamma: Scale parameters [B, 1, d_model]
            beta: Shift parameters [B, 1, d_model]
        """
        B, N_max_plus_1, N_patches, D = latent_frames.shape

        # Apply sequence length masking if specified
        if sequence_length is not None:
            # Use only the first sequence_length+1 frames (includes context frame)
            actual_frames = sequence_length + 1
            latent_frames = latent_frames[:, :actual_frames]  # [B, actual_frames, N_patches, D]

        # Get actual sequence length after masking
        seq_len = latent_frames.shape[1]

        # Flatten patches for each frame: [B, seq_len*N_patches, D]
        latent_flat = latent_frames.view(B, seq_len * N_patches, D)

        # Project to key/value space
        keys = self.latent_key_proj(latent_flat)  # [B, seq_len*N_patches, d_model]
        values = self.latent_value_proj(latent_flat)  # [B, seq_len*N_patches, d_model]

        # Expand query for batch
        query = self.gesture_query.expand(B, -1, -1)  # [B, 1, d_model]

        # Cross-attention
        attended, _ = self.multihead_attn(query, keys, values)  # [B, 1, d_model]

        # Extract FiLM parameters
        gamma = self.film_scale(attended)  # [B, 1, d_model]
        beta = self.film_shift(attended)  # [B, 1, d_model]

        return gamma, beta


class FiLMActionAttention(nn.Module):
    """FiLM conditioning via cross-attention to LAM action tokens with variable sequence length support."""

    def __init__(self, d_model: int, action_dim: int = 128, nhead: int = 8):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.nhead = nhead

        # Learned query for gesture context (same as visual version)
        self.gesture_query = nn.Parameter(torch.randn(1, 1, d_model))

        # Project action tokens to key/value space
        self.action_key_proj = nn.Linear(action_dim, d_model)
        self.action_value_proj = nn.Linear(action_dim, d_model)

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )

        # FiLM parameter extraction
        self.film_scale = nn.Linear(d_model, d_model)
        self.film_shift = nn.Linear(d_model, d_model)

        # Initialize with small weights
        nn.init.normal_(self.film_scale.weight, std=0.02)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.normal_(self.film_shift.weight, std=0.02)
        nn.init.zeros_(self.film_shift.bias)

    def forward(self, action_tokens: torch.Tensor,
                sequence_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate FiLM parameters from LAM action tokens with variable length masking.

        Args:
            action_tokens: LAM action tokens [B, N_max, action_dim] where N_max is max number of actions
            sequence_length: Actual sequence length to use (1 to N_max). If None, use all tokens.
        Returns:
            gamma: Scale parameters [B, 1, d_model]
            beta: Shift parameters [B, 1, d_model]
        """
        B, N_max, A = action_tokens.shape

        # Apply sequence length masking if specified
        if sequence_length is not None:
            # Use only the first sequence_length action tokens
            action_tokens = action_tokens[:, :sequence_length]  # [B, sequence_length, A]

        # Get actual sequence length after masking
        seq_len = action_tokens.shape[1]

        # Project to key/value space
        keys = self.action_key_proj(action_tokens)  # [B, seq_len, d_model]
        values = self.action_value_proj(action_tokens)  # [B, seq_len, d_model]

        # Expand query for batch
        query = self.gesture_query.expand(B, -1, -1)  # [B, 1, d_model]

        # Cross-attention
        attended, _ = self.multihead_attn(query, keys, values)  # [B, 1, d_model]

        # Extract FiLM parameters
        gamma = self.film_scale(attended)  # [B, 1, d_model]
        beta = self.film_shift(attended)  # [B, 1, d_model]

        return gamma, beta


class ResidualBlock(nn.Module):
    """Residual block with FiLM conditioning and time embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, film_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_channels)
        self.film_scale = nn.Linear(film_dim, out_channels)
        self.film_shift = nn.Linear(film_dim, out_channels)

        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Forward pass with FiLM conditioning.

        Args:
            x: Input tensor [B, C, L]
            t_emb: Time embedding [B, time_dim]
            gamma: FiLM scale [B, 1, film_dim]
            beta: FiLM shift [B, 1, film_dim]
        Returns:
            Output tensor [B, out_channels, L]
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = h + self.time_proj(t_emb)[:, :, None]
        h = self.norm2(h)
        h = F.silu(h)

        gamma_proj = self.film_scale(gamma.squeeze(1))[:, :, None]
        beta_proj = self.film_shift(beta.squeeze(1))[:, :, None]
        h = self.conv2(h)
        h = gamma_proj * h + beta_proj

        return h + self.residual(x)


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
        # x: [B,C,L] → transpose to [B,L,C]
        x_in = x.transpose(1, 2)
        h = self.norm(x_in)
        h, _ = self.attn(h, h, h)
        return (h + x_in).transpose(1, 2)


class GestureDiffusionUNet(nn.Module):
    """True U-Net for gesture sequence denoising with encoder-decoder pyramid."""

    def __init__(self, coordinate_dim: int = 2, d_model: int = 512,
                 time_embed_dim: int = 128, channels: tuple = (64, 128, 256, 512)):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )

        self.input_proj = nn.Conv1d(coordinate_dim, channels[0], 7, padding=3)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(channels) - 1):
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
        for i in range(len(rev_channels) - 1):
            self.ups.append(nn.ConvTranspose1d(rev_channels[i], rev_channels[i+1], 4, stride=2, padding=1))
            # Skip connections double the input channels for decoder blocks
            self.dec_blocks.append(ResidualBlock(rev_channels[i] + rev_channels[i+1], rev_channels[i+1], time_embed_dim, d_model))

        # Output heads
        self.coord_head = nn.Conv1d(channels[0], 2, 3, padding=1)
        self.pen_head = nn.Sequential(
            nn.Conv1d(channels[0], 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(32, 1, 3, padding=1)
            # No activation - output logits for BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                gamma: torch.Tensor, beta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of true U-Net with encoder-decoder pyramid.

        Args:
            x: Noisy gesture sequence [B, L, 3] (pen channel ignored)
            t: Timestep [B] or [B, 1]
            gamma: FiLM scale parameters [B, 1, d_model]
            beta: FiLM shift parameters [B, 1, d_model]
        Returns:
            Dict containing:
                'coordinates': Predicted noise for x,y coordinates [B, L, 2]
                'pen_state': Predicted pen state probabilities [B, L, 1]
        """
        t_emb = self.time_embed(t)

        # Input - only use (x,y) coordinates
        x = x[..., :2].transpose(1, 2)   # [B, 2, L]
        x = self.input_proj(x)           # [B, channels[0], L]

        # Encoder with skip connections
        skips = []
        for block, down in zip(self.enc_blocks, self.downs):
            x = block(x, t_emb, gamma, beta)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.mid_block1(x, t_emb, gamma, beta)
        x = self.attn(x)
        x = self.mid_block2(x, t_emb, gamma, beta)

        # Decoder with skip connections
        for up, block, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            x = up(x)
            # Handle size mismatch between upsampled x and skip connection
            if x.shape[2] != skip.shape[2]:
                # Crop or pad to match the skip connection size
                if x.shape[2] > skip.shape[2]:
                    x = x[:, :, :skip.shape[2]]  # Crop x to match skip
                else:
                    # Pad x to match skip (should rarely happen with stride=2)
                    padding = skip.shape[2] - x.shape[2]
                    x = F.pad(x, (0, padding))
            x = torch.cat([x, skip], dim=1)  # Concatenate skip connection
            x = block(x, t_emb, gamma, beta)

        # Output heads
        coords = self.coord_head(x).transpose(1, 2)    # [B, L, 2] - delta coordinates
        pen = self.pen_head(x).transpose(1, 2)         # [B, L, 1] - logits (no sigmoid)

        return {"coordinates": coords, "pen_state": pen}


class GestureDiffusionModel(nn.Module):
    """Complete gesture diffusion model with FiLM conditioning and classifier-free guidance."""

    def __init__(self, d_latent: int = 1024, d_model: int = 512,
                 channels: tuple = None, time_embed_dim: int = 128,
                 max_sequence_length: int = 1000, cfg_dropout_prob: float = 0.1):
        super().__init__()
        self.d_latent = d_latent
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.cfg_dropout_prob = cfg_dropout_prob

        # FiLM cross-attention conditioning
        self.film_cross_attn = FiLMCrossAttention(d_model, d_latent)

        # U-Net backbone - only processes coordinates
        if channels is None:
            channels = (64, 128, 256, 512)
        self.unet = GestureDiffusionUNet(
            coordinate_dim=2,  # Only x,y coordinates, pen channel completely ignored
            d_model=d_model,
            time_embed_dim=time_embed_dim,
            channels=channels
        )

        # Store d_latent for null conditioning creation
        self.d_latent = d_latent

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                latent_frames: Optional[torch.Tensor] = None,
                sequence_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of diffusion model.

        Args:
            x_t: Noisy gesture sequence [B, L, 3]
            t: Diffusion timestep [B] or [B, 1]
            latent_frames: Conditioning latent frames [B, N_max+1, N_patches, D] or None for unconditional
            sequence_length: Actual sequence length to use (1 to N_max). If None, use all frames.
        Returns:
            Dict containing:
                'coordinates': Predicted noise for x,y coordinates [B, L, 2]
                'pen_state': Predicted pen state probabilities [B, L, 1]
        """
        B, L, _ = x_t.shape

        # Handle null conditioning for classifier-free guidance
        if latent_frames is None:
            # Use null conditioning - create zeros with default patch count
            N_patches = 256  # Default patch count
            latent_frames = torch.zeros(B, 1, N_patches, self.d_latent,
                                      device=x_t.device, dtype=x_t.dtype)
            sequence_length = None  # No masking for null conditioning

        # Generate FiLM parameters from latent frames with sequence length masking
        film_gamma, film_beta = self.film_cross_attn(latent_frames, sequence_length)

        # Store FiLM values for histogram logging (similar to autoregressive model)
        self._store_film_values(film_gamma, film_beta)

        # Apply U-Net denoising with two-branch output
        model_output = self.unet(x_t, t, film_gamma, film_beta)

        return model_output

    def forward_pen_prediction(self, clean_gestures: torch.Tensor, t: torch.Tensor,
                              latent_frames: Optional[torch.Tensor] = None,
                              sequence_length: Optional[int] = None) -> torch.Tensor:
        """Forward pass using clean coordinates for pen state prediction only.

        Args:
            clean_gestures: Clean gesture sequence [B, L, 3] with ground truth coordinates
            t: Diffusion timestep [B] or [B, 1] (for time embedding consistency)
            latent_frames: Conditioning latent frames [B, N_max+1, N_patches, D] or None
            sequence_length: Actual sequence length to use
        Returns:
            Predicted pen state logits [B, L, 1]
        """
        B, L, _ = clean_gestures.shape

        # Handle null conditioning for classifier-free guidance
        if latent_frames is None:
            # Use null conditioning - create zeros with default patch count
            N_patches = 256  # Default patch count
            latent_frames = torch.zeros(B, 1, N_patches, self.d_latent,
                                      device=clean_gestures.device, dtype=clean_gestures.dtype)
            sequence_length = None  # No masking for null conditioning

        # Generate FiLM parameters from latent frames with sequence length masking
        film_gamma, film_beta = self.film_cross_attn(latent_frames, sequence_length)

        # Store FiLM values for histogram logging (similar to autoregressive model)
        self._store_film_values(film_gamma, film_beta)

        # Use absolute coordinates directly (no conversion to deltas)
        clean_coords = clean_gestures[..., :2]  # [B, L, 2] - absolute coordinates

        # Create input with absolute coords and zero pen channel
        zero_pen_channel = torch.zeros_like(clean_gestures[..., 2:3])
        clean_input = torch.cat([clean_coords, zero_pen_channel], dim=-1)  # [B, L, 3]

        # Apply U-Net denoising - we only care about pen head output
        model_output = self.unet(clean_input, t, film_gamma, film_beta)

        # Return only pen state predictions
        return model_output['pen_state']  # [B, L, 1] - logits

    def _store_film_values(self, gamma: torch.Tensor, beta: torch.Tensor):
        """Store FiLM values for histogram logging (similar to autoregressive model).

        Args:
            gamma: FiLM scale parameters [B, 1, d_model]
            beta: FiLM shift parameters [B, 1, d_model]
        """
        with torch.no_grad():
            # Store raw FiLM values for magnitude distribution logging
            self.last_film_gamma = gamma.detach().clone()
            self.last_film_beta = beta.detach().clone()

            # Compute norms (L2 norm per position)
            self.last_gamma_norm = torch.norm(gamma, dim=-1, keepdim=False)  # [B, 1]
            self.last_beta_norm = torch.norm(beta, dim=-1, keepdim=False)    # [B, 1]

    def apply_cfg_dropout(self, latent_frames: torch.Tensor,
                         training: bool = True) -> torch.Tensor:
        """Apply classifier-free guidance dropout during training.

        Args:
            latent_frames: Conditioning latent frames [B, N+1, N_patches, D]
            training: Whether in training mode
        Returns:
            latent_frames with some samples replaced by null conditioning
        """
        if not training:
            return latent_frames

        B, N_plus_1, N_patches, D = latent_frames.shape

        # Create dropout mask
        dropout_mask = torch.rand(B, device=latent_frames.device) < self.cfg_dropout_prob

        # Create null conditioning with correct patch count
        null_conditioning = torch.zeros_like(latent_frames)
        latent_frames = torch.where(
            dropout_mask[:, None, None, None],
            null_conditioning,
            latent_frames
        )

        return latent_frames

    def preprocess_gestures_for_training(self, gestures: torch.Tensor) -> torch.Tensor:
        """Preprocess gestures for training (now using absolute coordinates).

        Args:
            gestures: Tensor of shape [B, L, 3] with absolute coordinates (x, y, p)
        Returns:
            gestures unchanged: [B, L, 3] with absolute coordinates
        """
        # No conversion needed - we use absolute coordinates directly
        return gestures

    def compute_coordinate_loss(self, model_output: Dict[str, torch.Tensor],
                              target_coordinates: torch.Tensor) -> torch.Tensor:
        """Compute coordinate loss (MSE for diffusion denoising).

        Args:
            model_output: Dict with 'coordinates' [B, L, 2]
            target_coordinates: Ground truth coordinate noise [B, L, 2]
        Returns:
            Coordinate MSE loss
        """
        return F.mse_loss(model_output['coordinates'], target_coordinates)

    def compute_pen_loss(self, pen_logits: torch.Tensor,
                        target_pen_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute pen state loss (BCEWithLogits with class imbalance handling).

        Args:
            pen_logits: Predicted pen state logits [B, L, 1]
            target_pen_state: Ground truth pen state [B, L, 1]
        Returns:
            Dict with pen loss and pos_weight
        """
        # Compute pos_weight = neg_samples / pos_samples per batch
        pen_targets_flat = target_pen_state.view(-1)  # [B*L]
        pos_samples = torch.sum(pen_targets_flat)
        neg_samples = len(pen_targets_flat) - pos_samples

        if pos_samples > 0:
            pos_weight = neg_samples / pos_samples
        else:
            pos_weight = torch.tensor(1.0, device=target_pen_state.device)

        pen_logits_flat = pen_logits.view(-1)  # [B*L]
        pen_loss = F.binary_cross_entropy_with_logits(
            pen_logits_flat, pen_targets_flat, pos_weight=pos_weight
        )

        return {
            'pen_loss': pen_loss,
            'pos_weight': pos_weight
        }

    def compute_combined_loss(self, model_output: Dict[str, torch.Tensor],
                            target_coordinates: torch.Tensor, target_pen_state: torch.Tensor,
                            coordinate_weight: float = 1.0, pen_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """Compute combined loss for coordinates (MSE) and pen state (BCEWithLogits).
        DEPRECATED: Use compute_coordinate_loss and compute_pen_loss separately.

        Args:
            model_output: Dict with 'coordinates' [B, L, 2] and 'pen_state' [B, L, 1] (logits)
            target_coordinates: Ground truth coordinate noise [B, L, 2]
            target_pen_state: Ground truth pen state [B, L, 1]
            coordinate_weight: Weight for coordinate MSE loss
            pen_weight: Weight for pen state BCE loss

        Returns:
            Dict with loss components
        """
        # Coordinate loss (MSE for diffusion denoising)
        coord_loss = self.compute_coordinate_loss(model_output, target_coordinates)

        # Pen state loss
        pen_loss_dict = self.compute_pen_loss(model_output['pen_state'], target_pen_state)
        pen_loss = pen_loss_dict['pen_loss']
        pos_weight = pen_loss_dict['pos_weight']

        # Combined loss
        total_loss = coordinate_weight * coord_loss + pen_weight * pen_loss

        return {
            'total_loss': total_loss,
            'coordinate_loss': coord_loss,
            'pen_loss': pen_loss,
            'coordinate_weight': coordinate_weight,
            'pen_weight': pen_weight,
            'pos_weight': pos_weight
        }


class GestureDiffusionModelLAM(nn.Module):
    """LAM-conditioned gesture diffusion model with action token FiLM conditioning."""

    def __init__(self, action_dim: int = 128, d_model: int = 512,
                 channels: tuple = None, time_embed_dim: int = 128,
                 max_sequence_length: int = 1000, cfg_dropout_prob: float = 0.1,
                 lam_checkpoint_path: Optional[str] = None):
        super().__init__()
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.cfg_dropout_prob = cfg_dropout_prob
        self.lam_checkpoint_path = lam_checkpoint_path

        # Load LAM model for action token generation
        self.lam_model = self._load_lam_model(lam_checkpoint_path)

        # FiLM action-attention conditioning
        self.film_action_attn = FiLMActionAttention(d_model, action_dim)

        # U-Net backbone - same as visual version
        if channels is None:
            channels = (64, 128, 256, 512)
        self.unet = GestureDiffusionUNet(
            coordinate_dim=2,  # Only x,y coordinates, pen channel completely ignored
            d_model=d_model,
            time_embed_dim=time_embed_dim,
            channels=channels
        )

        # Store action_dim for null conditioning creation
        self.action_dim = action_dim

    def _load_lam_model(self, checkpoint_path: Optional[str]):
        """Load pre-trained LAM model for action token generation."""
        if checkpoint_path is None:
            # Create a dummy LAM model for initialization (will need to be loaded later)
            from latent_action_model.vae import LatentActionVAE
            lam_model = LatentActionVAE(
                latent_dim=1024,  # VJepa patch dimension
                action_dim=self.action_dim,
                embed_dim=512,
                encoder_depth=3,
                decoder_depth=3
            )
            lam_model.eval()
            # Freeze parameters
            for param in lam_model.parameters():
                param.requires_grad = False
            return lam_model

        # Load from checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"LAM checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Import here to avoid circular dependency
        from latent_action_model.vae import LatentActionVAE

        # Extract model config from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']

            # Infer architecture from state_dict if config is incomplete
            state_dict = checkpoint['model_state_dict']

            # Count encoder/decoder layers by examining state_dict keys
            encoder_blocks = [k for k in state_dict.keys() if k.startswith('encoder.blocks.')]
            decoder_blocks = [k for k in state_dict.keys() if k.startswith('decoder.blocks.')]

            inferred_encoder_depth = max([int(k.split('.')[2]) for k in encoder_blocks]) + 1 if encoder_blocks else 3
            inferred_decoder_depth = max([int(k.split('.')[2]) for k in decoder_blocks]) + 1 if decoder_blocks else 3

            lam_model = LatentActionVAE(
                latent_dim=config.get('patch_dim', config.get('latent_dim', 1024)),
                action_dim=config.get('action_dim', self.action_dim),
                embed_dim=config.get('embed_dim', 512),
                encoder_depth=config.get('encoder_depth', inferred_encoder_depth),
                decoder_depth=config.get('decoder_depth', inferred_decoder_depth),
                encoder_heads=config.get('encoder_heads', 8),
                decoder_heads=config.get('decoder_heads', 8),
                mlp_ratio=config.get('mlp_ratio', 4.0),
                drop_rate=config.get('drop_rate', 0.0),
                attn_drop_rate=config.get('attn_drop_rate', 0.0),
                kl_weight=config.get('kl_weight', 1.0)
            )

            print(f"Inferred LAM architecture: encoder_depth={inferred_encoder_depth}, decoder_depth={inferred_decoder_depth}")
        else:
            # Fallback: infer from state_dict only
            state_dict = checkpoint['model_state_dict']
            encoder_blocks = [k for k in state_dict.keys() if k.startswith('encoder.blocks.')]
            decoder_blocks = [k for k in state_dict.keys() if k.startswith('decoder.blocks.')]

            encoder_depth = max([int(k.split('.')[2]) for k in encoder_blocks]) + 1 if encoder_blocks else 3
            decoder_depth = max([int(k.split('.')[2]) for k in decoder_blocks]) + 1 if decoder_blocks else 3

            lam_model = LatentActionVAE(
                latent_dim=1024,
                action_dim=self.action_dim,
                embed_dim=512,
                encoder_depth=encoder_depth,
                decoder_depth=decoder_depth
            )

            print(f"No config found, inferred LAM architecture: encoder_depth={encoder_depth}, decoder_depth={decoder_depth}")

        # Load state dict
        lam_model.load_state_dict(checkpoint['model_state_dict'])
        lam_model.eval()

        # Freeze LAM parameters
        for param in lam_model.parameters():
            param.requires_grad = False

        print(f"Loaded LAM model from {checkpoint_path}")
        return lam_model

    def load_lam_checkpoint(self, checkpoint_path: str):
        """Load LAM checkpoint after model creation."""
        self.lam_model = self._load_lam_model(checkpoint_path)
        self.lam_checkpoint_path = checkpoint_path

    @torch.no_grad()
    def encode_visual_to_actions(self, latent_frames: torch.Tensor,
                                sequence_length: Optional[int] = None) -> torch.Tensor:
        """Convert visual frames to action tokens using LAM encoder.

        Args:
            latent_frames: Visual frames [B, N_max+1, N_patches, D]
            sequence_length: Actual sequence length (1 to N_max). If None, use all frames.
        Returns:
            action_tokens: LAM action tokens [B, N_max, action_dim]
        """
        B, N_max_plus_1, N_patches, D = latent_frames.shape

        # Apply sequence length masking if specified
        if sequence_length is not None:
            # Use only the first sequence_length+1 frames
            actual_frames = sequence_length + 1
            latent_frames = latent_frames[:, :actual_frames]  # [B, actual_frames, N_patches, D]

        # LAM encoder expects [B, T, N, D] format
        # latent_frames is already in correct format

        # Generate action tokens using LAM encoder
        mu, logvar = self.lam_model.encode(latent_frames)  # [B, T-1, action_dim]
        action_tokens = self.lam_model.reparameterize(mu, logvar)  # [B, T-1, action_dim]

        # Pad to max sequence length if needed for consistent tensor sizes
        if sequence_length is not None and sequence_length < N_max_plus_1 - 1:
            # Pad to N_max actions
            N_max = N_max_plus_1 - 1  # Maximum number of actions
            current_actions = action_tokens.shape[1]  # T-1 = sequence_length
            if current_actions < N_max:
                padding = torch.zeros(B, N_max - current_actions, self.action_dim,
                                    device=action_tokens.device, dtype=action_tokens.dtype)
                action_tokens = torch.cat([action_tokens, padding], dim=1)  # [B, N_max, action_dim]

        return action_tokens

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                latent_frames: Optional[torch.Tensor] = None,
                sequence_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of LAM-conditioned diffusion model.

        Args:
            x_t: Noisy gesture sequence [B, L, 3]
            t: Diffusion timestep [B] or [B, 1]
            latent_frames: Conditioning visual frames [B, N_max+1, N_patches, D] or None for unconditional
            sequence_length: Actual sequence length to use (1 to N_max). If None, use all frames.
        Returns:
            Dict containing:
                'coordinates': Predicted noise for x,y coordinates [B, L, 2]
                'pen_state': Predicted pen state probabilities [B, L, 1]
        """
        B, L, _ = x_t.shape

        # Handle null conditioning for classifier-free guidance
        if latent_frames is None:
            # Use null conditioning - create zeros with max action sequence length
            N_max = self.max_sequence_length // 250 if hasattr(self, 'max_sequence_length') else 4
            action_tokens = torch.zeros(B, N_max, self.action_dim,
                                      device=x_t.device, dtype=x_t.dtype)
            actual_sequence_length = None
        else:
            # Convert visual frames to action tokens using LAM
            action_tokens = self.encode_visual_to_actions(latent_frames, sequence_length)
            actual_sequence_length = sequence_length

        # Generate FiLM parameters from action tokens
        film_gamma, film_beta = self.film_action_attn(action_tokens, actual_sequence_length)

        # Store FiLM values for histogram logging
        self._store_film_values(film_gamma, film_beta)

        # Apply U-Net denoising with action-conditioned FiLM
        model_output = self.unet(x_t, t, film_gamma, film_beta)

        return model_output

    def forward_pen_prediction(self, clean_gestures: torch.Tensor, t: torch.Tensor,
                              latent_frames: Optional[torch.Tensor] = None,
                              sequence_length: Optional[int] = None) -> torch.Tensor:
        """Forward pass using clean coordinates for pen state prediction only.

        Args:
            clean_gestures: Clean gesture sequence [B, L, 3] with ground truth coordinates
            t: Diffusion timestep [B] or [B, 1] (for time embedding consistency)
            latent_frames: Conditioning visual frames [B, N_max+1, N_patches, D] or None
            sequence_length: Actual sequence length to use
        Returns:
            Predicted pen state logits [B, L, 1]
        """
        B, L, _ = clean_gestures.shape

        # Handle null conditioning for classifier-free guidance
        if latent_frames is None:
            # Use null conditioning - create zeros
            N_max = self.max_sequence_length // 250 if hasattr(self, 'max_sequence_length') else 4
            action_tokens = torch.zeros(B, N_max, self.action_dim,
                                      device=clean_gestures.device, dtype=clean_gestures.dtype)
            actual_sequence_length = None
        else:
            # Convert visual frames to action tokens using LAM
            action_tokens = self.encode_visual_to_actions(latent_frames, sequence_length)
            actual_sequence_length = sequence_length

        # Generate FiLM parameters from action tokens
        film_gamma, film_beta = self.film_action_attn(action_tokens, actual_sequence_length)

        # Store FiLM values for histogram logging
        self._store_film_values(film_gamma, film_beta)

        # Use absolute coordinates directly (no conversion to deltas)
        clean_coords = clean_gestures[..., :2]  # [B, L, 2] - absolute coordinates

        # Create input with absolute coords and zero pen channel
        zero_pen_channel = torch.zeros_like(clean_gestures[..., 2:3])
        clean_input = torch.cat([clean_coords, zero_pen_channel], dim=-1)  # [B, L, 3]

        # Apply U-Net denoising - we only care about pen head output
        model_output = self.unet(clean_input, t, film_gamma, film_beta)

        # Return only pen state predictions
        return model_output['pen_state']  # [B, L, 1] - logits

    def _store_film_values(self, gamma: torch.Tensor, beta: torch.Tensor):
        """Store FiLM values for histogram logging (same as visual version)."""
        with torch.no_grad():
            # Store raw FiLM values for magnitude distribution logging
            self.last_film_gamma = gamma.detach().clone()
            self.last_film_beta = beta.detach().clone()

            # Compute norms (L2 norm per position)
            self.last_gamma_norm = torch.norm(gamma, dim=-1, keepdim=False)  # [B, 1]
            self.last_beta_norm = torch.norm(beta, dim=-1, keepdim=False)    # [B, 1]

    def apply_cfg_dropout(self, latent_frames: torch.Tensor,
                         training: bool = True) -> torch.Tensor:
        """Apply classifier-free guidance dropout during training.
        Note: This works on visual frames, action token dropout happens in forward()."""
        if not training:
            return latent_frames

        B, N_plus_1, N_patches, D = latent_frames.shape

        # Create dropout mask
        dropout_mask = torch.rand(B, device=latent_frames.device) < self.cfg_dropout_prob

        # Create null conditioning with correct patch count
        null_conditioning = torch.zeros_like(latent_frames)
        latent_frames = torch.where(
            dropout_mask[:, None, None, None],
            null_conditioning,
            latent_frames
        )

        return latent_frames

    def preprocess_gestures_for_training(self, gestures: torch.Tensor) -> torch.Tensor:
        """Preprocess gestures for training (same as visual version)."""
        # No conversion needed - we use absolute coordinates directly
        return gestures

    def compute_coordinate_loss(self, model_output: Dict[str, torch.Tensor],
                              target_coordinates: torch.Tensor) -> torch.Tensor:
        """Compute coordinate loss (same as visual version)."""
        return F.mse_loss(model_output['coordinates'], target_coordinates)

    def compute_pen_loss(self, pen_logits: torch.Tensor,
                        target_pen_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute pen state loss (same as visual version)."""
        # Compute pos_weight = neg_samples / pos_samples per batch
        pen_targets_flat = target_pen_state.view(-1)  # [B*L]
        pos_samples = torch.sum(pen_targets_flat)
        neg_samples = len(pen_targets_flat) - pos_samples

        if pos_samples > 0:
            pos_weight = neg_samples / pos_samples
        else:
            pos_weight = torch.tensor(1.0, device=target_pen_state.device)

        pen_logits_flat = pen_logits.view(-1)  # [B*L]
        pen_loss = F.binary_cross_entropy_with_logits(
            pen_logits_flat, pen_targets_flat, pos_weight=pos_weight
        )

        return {
            'pen_loss': pen_loss,
            'pos_weight': pos_weight
        }

    def compute_combined_loss(self, model_output: Dict[str, torch.Tensor],
                            target_coordinates: torch.Tensor, target_pen_state: torch.Tensor,
                            coordinate_weight: float = 1.0, pen_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """Compute combined loss (same as visual version)."""
        # Coordinate loss (MSE for diffusion denoising)
        coord_loss = self.compute_coordinate_loss(model_output, target_coordinates)

        # Pen state loss
        pen_loss_dict = self.compute_pen_loss(model_output['pen_state'], target_pen_state)
        pen_loss = pen_loss_dict['pen_loss']
        pos_weight = pen_loss_dict['pos_weight']

        # Combined loss
        total_loss = coordinate_weight * coord_loss + pen_weight * pen_loss

        return {
            'total_loss': total_loss,
            'coordinate_loss': coord_loss,
            'pen_loss': pen_loss,
            'coordinate_weight': coordinate_weight,
            'pen_weight': pen_weight,
            'pos_weight': pos_weight
        }

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, lam_checkpoint_path: Optional[str] = None):
        """Load LAM-conditioned model from checkpoint with auto-detection."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract model config
        model_type = checkpoint.get("model_type", "unknown")
        if model_type != "gesture_diffusion_lam":
            raise ValueError(f"Expected LAM model, got {model_type}")

        # Get LAM checkpoint path from saved config or override
        saved_lam_path = checkpoint.get("lam_checkpoint_path")
        actual_lam_path = lam_checkpoint_path or saved_lam_path

        # Create model with saved hyperparameters
        model_config = checkpoint.get("model_config", {})
        model = cls(
            action_dim=model_config.get("action_dim", 128),
            d_model=model_config.get("d_model", 512),
            channels=model_config.get("channels", (64, 128, 256, 512)),
            time_embed_dim=model_config.get("time_embed_dim", 128),
            max_sequence_length=model_config.get("max_sequence_length", 1000),
            cfg_dropout_prob=model_config.get("cfg_dropout_prob", 0.1),
            lam_checkpoint_path=actual_lam_path
        )

        # Load model state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        return model


# Delta conversion functions - DEPRECATED (now using absolute coordinates)
# Kept for reference but no longer used in training or generation

# def convert_to_deltas(gestures: torch.Tensor) -> torch.Tensor:
#     """Convert absolute coordinates to frame-local deltas.
#     DEPRECATED: We now use absolute coordinates directly.
#     """
#     coords = gestures[..., :2]  # [B, L, 2] - absolute coordinates
#     deltas = torch.zeros_like(coords)
#     deltas[:, 1:] = coords[:, 1:] - coords[:, :-1]  # Δx_t = x_t - x_{t-1}
#     deltas[:, 0] = 0.0  # First delta is zero: Δx_0 = 0, Δy_0 = 0
#
#     # Replace coordinates with deltas, keep pen state unchanged
#     result = gestures.clone()
#     result[..., :2] = deltas
#     return result
#
#
# def reconstruct_coordinates(delta_gestures: torch.Tensor,
#                           start_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
#     """Reconstruct absolute coordinates from delta gestures via cumulative sum.
#     DEPRECATED: We now use absolute coordinates directly.
#     """
#     B, L, _ = delta_gestures.shape
#     device = delta_gestures.device
#
#     deltas = delta_gestures[..., :2]  # [B, L, 2]
#     pen_states = delta_gestures[..., 2:3]  # [B, L, 1]
#
#     # Inverse scaling from training (scale back up by 10)
#     deltas = deltas * 10.0
#
#     # Cumulative sum to get absolute coordinates
#     coords = torch.cumsum(deltas, dim=1)  # [B, L, 2]
#
#     # Add starting positions if provided
#     if start_positions is not None:
#         coords = coords + start_positions[:, None, :]  # Broadcast [B, 1, 2]
#
#     # Combine coordinates and pen states
#     result = torch.cat([coords, pen_states], dim=-1)  # [B, L, 3]
#     return result


def create_diffusion_schedule(num_timesteps: int = 1000,
                            beta_start: float = 1e-4,
                            beta_end: float = 0.02) -> Dict[str, torch.Tensor]:
    """Create diffusion schedule with precomputed values for efficient sampling.

    Args:
        num_timesteps: Number of diffusion timesteps
        beta_start: Starting noise schedule value
        beta_end: Ending noise schedule value
    Returns:
        Dictionary containing diffusion schedule tensors
    """
    # Linear beta schedule
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - alphas_cumprod),
        'sqrt_recip_alphas': torch.sqrt(1.0 / alphas),
        'posterior_variance': betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
    }


class DDIMSampler:
    """DDIM sampler for fast deterministic sampling."""

    def __init__(self, model: GestureDiffusionModel, schedule: Dict[str, torch.Tensor],
                 device: str = "cuda"):
        self.model = model
        self.schedule = schedule
        self.device = device

        # Move schedule to device
        for key in self.schedule:
            self.schedule[key] = self.schedule[key].to(device)

    @torch.no_grad()
    def sample(self, shape: Tuple[int, int, int],
               latent_frames: Optional[torch.Tensor] = None,
               sequence_length: Optional[int] = None,
               num_inference_steps: int = 50,
               cfg_scale: float = 7.5,
               eta: float = 0.0) -> torch.Tensor:
        """Sample gesture sequences using DDIM with absolute coordinate approach.

        Args:
            shape: Output shape [B, L, 3]
            latent_frames: Conditioning latent frames [B, N_max+1, N_patches, D]
            sequence_length: Actual sequence length to use (1 to N_max). If None, use all frames.
            num_inference_steps: Number of denoising steps
            cfg_scale: Classifier-free guidance scale
            eta: DDIM eta parameter (0.0 = deterministic)
        Returns:
            Generated gesture sequences [B, L, 3] with absolute coordinates
        """
        B, L, C = shape
        device = self.device

        # Create timestep schedule
        num_train_timesteps = len(self.schedule['betas'])
        timesteps = torch.linspace(num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long)
        timesteps = timesteps.to(device)

        # Start with pure noise for absolute coordinates, zero for pen channel
        x_t = torch.randn(shape, device=device)
        x_t[..., 2] = 0.0  # Zero out pen channel to match training
        # Note: x_t now represents noisy absolute coordinates

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B)

            # Predict noise with classifier-free guidance (coordinates only)
            if cfg_scale > 1.0 and latent_frames is not None:
                # Conditional prediction
                output_cond = self.model(x_t, t_batch, latent_frames, sequence_length)

                # Unconditional prediction
                output_uncond = self.model(x_t, t_batch, None, None)

                # Apply classifier-free guidance to coordinates only (diffusion part)
                coord_noise_pred = output_cond['coordinates'] + cfg_scale * (output_cond['coordinates'] - output_uncond['coordinates'])
            else:
                # Standard prediction
                output = self.model(x_t, t_batch, latent_frames, sequence_length)
                coord_noise_pred = output['coordinates']
                # Note: pen_state prediction is ignored during DDIM loop

            # DDIM update (only for coordinates, since pen state is not diffused)
            alpha_t = self.schedule['alphas_cumprod'][t]
            if i < len(timesteps) - 1:
                alpha_t_prev = self.schedule['alphas_cumprod'][timesteps[i + 1]]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)

            # Extract coordinates for DDIM update
            x_t_coords = x_t[..., :2]  # [B, L, 2] - only x,y coordinates

            # Predict x_0 for coordinates only
            x_0_pred_coords = (x_t_coords - torch.sqrt(1 - alpha_t) * coord_noise_pred) / torch.sqrt(alpha_t)

            # DDIM update formula for coordinates
            if i < len(timesteps) - 1:
                sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
                noise = torch.randn_like(x_t_coords) if eta > 0 else 0

                x_t_coords_next = (torch.sqrt(alpha_t_prev) * x_0_pred_coords +
                                  torch.sqrt(1 - alpha_t_prev - sigma_t**2) * coord_noise_pred +
                                  sigma_t * noise)
            else:
                x_t_coords_next = x_0_pred_coords

            # Update x_t with new coordinates (keep original pen channel as placeholder)
            x_t = torch.cat([x_t_coords_next, x_t[..., 2:3]], dim=-1)  # [B, L, 3]

        # At this point, x_t contains clean absolute coordinates and pen channel placeholder
        clean_coords = x_t[..., :2]  # [B, L, 2] - denoised absolute coordinates

        # Create temporary output with coordinates (no cumsum needed for absolute coords)
        clean_coords_output = torch.cat([clean_coords, torch.zeros(B, L, 1, device=device)], dim=-1)  # [B, L, 3]

        # --- CLEAN PEN PREDICTION PASS ---
        # Now do a separate forward pass with clean reconstructed coordinates to predict pen state
        # Use a dummy timestep (0) since we're not doing diffusion for pen prediction
        dummy_t = torch.zeros(B, device=device, dtype=torch.long)

        # Predict pen state using clean coordinates (matching training approach)
        pen_logits = self.model.forward_pen_prediction(
            clean_coords_output, dummy_t, latent_frames, sequence_length
        )  # [B, L, 1]

        # Apply sigmoid to logits before thresholding
        pen_probs = torch.sigmoid(pen_logits)  # [B, L, 1] - convert logits to probabilities
        thresholded_pen_state = (pen_probs > 0.5).float()  # [B, L, 1] - threshold at 0.5

        # Combine clean coordinates with predicted pen state
        final_output = torch.cat([clean_coords, thresholded_pen_state], dim=-1)  # [B, L, 3]

        return final_output


# Model Loading and Versioning Utilities

def load_gesture_diffusion_model(checkpoint_path: str, lam_checkpoint_path: Optional[str] = None):
    """
    Auto-detect and load the correct gesture diffusion model version.

    Args:
        checkpoint_path: Path to the diffusion model checkpoint
        lam_checkpoint_path: Path to LAM model checkpoint (for LAM models)

    Returns:
        Loaded model (either GestureDiffusionModel or GestureDiffusionModelLAM)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint metadata
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Detect model type
    model_type = checkpoint.get("model_type", "gesture_diffusion_visual")  # backward compatibility
    model_version = checkpoint.get("model_version", "1.0")

    print(f"Loading {model_type} v{model_version} from {checkpoint_path}")

    if model_type == "gesture_diffusion_lam":
        # Load LAM-conditioned model
        return GestureDiffusionModelLAM.from_checkpoint(checkpoint_path, lam_checkpoint_path)
    else:
        # Load visual-conditioned model (original)
        return GestureDiffusionModel.from_checkpoint(checkpoint_path)


def save_gesture_diffusion_checkpoint(model, checkpoint_path: str,
                                    trainer_state: Optional[Dict] = None,
                                    lam_checkpoint_path: Optional[str] = None):
    """
    Save gesture diffusion model checkpoint with proper versioning.

    Args:
        model: Model to save (GestureDiffusionModel or GestureDiffusionModelLAM)
        checkpoint_path: Path to save checkpoint
        trainer_state: Optional trainer state dict
        lam_checkpoint_path: Path to LAM checkpoint (for LAM models)
    """
    # Determine model type and version
    if isinstance(model, GestureDiffusionModelLAM):
        model_type = "gesture_diffusion_lam"
        model_version = "2.0"
        conditioning_type = "lam_actions"

        # Store model config
        model_config = {
            "action_dim": model.action_dim,
            "d_model": model.d_model,
            "channels": (64, 128, 256, 512),  # Default from model
            "time_embed_dim": 128,  # Default
            "max_sequence_length": model.max_sequence_length,
            "cfg_dropout_prob": model.cfg_dropout_prob
        }

    else:  # GestureDiffusionModel
        model_type = "gesture_diffusion_visual"
        model_version = "1.0"
        conditioning_type = "visual_frames"

        # Store model config
        model_config = {
            "d_latent": model.d_latent,
            "d_model": model.d_model,
            "channels": (64, 128, 256, 512),  # Default from model
            "time_embed_dim": 128,  # Default
            "max_sequence_length": model.max_sequence_length,
            "cfg_dropout_prob": model.cfg_dropout_prob
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

        # LAM-specific
        "lam_checkpoint_path": lam_checkpoint_path,

        # Trainer state (if provided)
        **(trainer_state or {})
    }

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved {model_type} v{model_version} to {checkpoint_path}")


# Add from_checkpoint method to original model for consistency
def _add_from_checkpoint_to_original():
    """Add from_checkpoint class method to original GestureDiffusionModel"""

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str):
        """Load visual-conditioned model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract model config
        model_type = checkpoint.get("model_type", "gesture_diffusion_visual")
        if model_type == "gesture_diffusion_lam":
            raise ValueError(f"Use GestureDiffusionModelLAM.from_checkpoint() for LAM models")

        # Create model with saved hyperparameters
        model_config = checkpoint.get("model_config", {})
        model = cls(
            d_latent=model_config.get("d_latent", 1024),
            d_model=model_config.get("d_model", 512),
            channels=model_config.get("channels", (64, 128, 256, 512)),
            time_embed_dim=model_config.get("time_embed_dim", 128),
            max_sequence_length=model_config.get("max_sequence_length", 1000),
            cfg_dropout_prob=model_config.get("cfg_dropout_prob", 0.1)
        )

        # Load model state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        return model

    # Add method to class
    GestureDiffusionModel.from_checkpoint = from_checkpoint

