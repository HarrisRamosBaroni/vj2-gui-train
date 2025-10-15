import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, Union


class JEPAToSequenceProcessor(nn.Module):
    """Convert JEPA latent frames to initial U-Net features via self-attention."""

    def __init__(self, d_latent: int = 1024, d_model: int = 512, max_segments: int = 4,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_latent = d_latent
        self.d_model = d_model
        self.max_segments = max_segments
        self.target_length = max_segments * 250  # Fixed sequence length
        self.input_frames = max_segments + 1     # e.g., 5 frames for 4 segments

        # Self-attention for processing JEPA frames
        self.jepa_self_attn = nn.MultiheadAttention(
            embed_dim=d_latent,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.jepa_norm = nn.LayerNorm(d_latent)

        # Learned positional queries for target sequence
        self.sequence_queries = nn.Parameter(torch.randn(self.target_length, d_latent))
        nn.init.normal_(self.sequence_queries, std=0.02)

        # Cross-attention to extract sequence features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_latent,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_latent)

        # Project to U-Net feature space
        self.feature_proj = nn.Sequential(
            nn.Linear(d_latent, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        # Positional encoding for sequence
        self.pos_encoding = nn.Parameter(torch.randn(1, self.target_length, d_model))
        nn.init.normal_(self.pos_encoding, std=0.02)

    def forward(self, latent_frames: torch.Tensor, sequence_length: Optional[int] = None) -> torch.Tensor:
        """Convert JEPA latent frames to sequence features.

        Args:
            latent_frames: [B, max_segments+1, N_patches, D_latent]
            sequence_length: Active sequence length (1 to max_segments)
        Returns:
            Sequence features: [B, max_segments*250, d_model]
        """
        B, N_frames, N_patches, D = latent_frames.shape
        assert N_frames == self.input_frames, f"Expected {self.input_frames} frames, got {N_frames}"

        # Apply sequence length masking if specified
        if sequence_length is not None:
            # Use only the first sequence_length+1 frames
            actual_frames = min(sequence_length + 1, N_frames)
            active_frames = latent_frames[:, :actual_frames]  # [B, actual_frames, N_patches, D]

            # Create attention mask for unused frames
            frame_mask = torch.ones(B, N_frames, device=latent_frames.device, dtype=torch.bool)
            frame_mask[:, actual_frames:] = False
        else:
            active_frames = latent_frames
            frame_mask = None

        # Flatten JEPA frames: [B, N_frames * N_patches, D]
        jepa_flat = latent_frames.view(B, N_frames * N_patches, D)

        # Self-attention on JEPA features
        jepa_attended, _ = self.jepa_self_attn(jepa_flat, jepa_flat, jepa_flat)
        jepa_features = self.jepa_norm(jepa_attended + jepa_flat)  # [B, N_frames * N_patches, D]

        # Expand sequence queries for batch
        queries = self.sequence_queries.unsqueeze(0).expand(B, -1, -1)  # [B, target_length, D]

        # Cross-attention: sequence queries attend to JEPA features
        sequence_features, _ = self.cross_attn(queries, jepa_features, jepa_features)
        sequence_features = self.cross_norm(sequence_features + queries)  # [B, target_length, D]

        # Project to U-Net feature space
        unet_features = self.feature_proj(sequence_features)  # [B, target_length, d_model]

        # Add positional encoding
        unet_features = unet_features + self.pos_encoding

        return unet_features  # [B, max_segments*250, d_model]


class FiLMSkipBridge(nn.Module):
    """FiLM conditioning applied to skip connections in U-Net."""

    def __init__(self, skip_channels: int, d_latent: int = 1024, nhead: int = 8):
        super().__init__()
        self.skip_channels = skip_channels
        self.d_latent = d_latent

        # Learned query for this specific skip level
        self.skip_query = nn.Parameter(torch.randn(1, 1, skip_channels))

        # Project latent frames to key/value space
        self.latent_key_proj = nn.Linear(d_latent, skip_channels)
        self.latent_value_proj = nn.Linear(d_latent, skip_channels)

        # Multi-head attention (ensure at least 1 head)
        num_heads = max(1, min(nhead, skip_channels // 64))
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=skip_channels,
            num_heads=num_heads,
            batch_first=True
        )

        # FiLM parameter extraction
        self.film_scale = nn.Linear(skip_channels, skip_channels)
        self.film_shift = nn.Linear(skip_channels, skip_channels)

        # Initialize with small weights
        nn.init.normal_(self.film_scale.weight, std=0.02)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.normal_(self.film_shift.weight, std=0.02)
        nn.init.zeros_(self.film_shift.bias)

    def forward(self, skip_features: torch.Tensor, latent_frames: torch.Tensor,
                sequence_length: Optional[int] = None) -> torch.Tensor:
        """Apply FiLM conditioning to skip connection features.

        Args:
            skip_features: Skip connection features [B, C, L]
            latent_frames: JEPA latent frames [B, N_max+1, N_patches, D]
            sequence_length: Active sequence length
        Returns:
            FiLM-conditioned skip features [B, C, L]
        """
        B, C, L = skip_features.shape
        N_frames, N_patches, D = latent_frames.shape[1:]

        # Apply sequence length masking to latent frames
        if sequence_length is not None:
            actual_frames = min(sequence_length + 1, N_frames)
            active_latents = latent_frames[:, :actual_frames]
        else:
            active_latents = latent_frames

        # Flatten latent frames: [B, seq_len*N_patches, D]
        seq_len = active_latents.shape[1]
        latent_flat = active_latents.view(B, seq_len * N_patches, D)

        # Project to key/value space
        keys = self.latent_key_proj(latent_flat)  # [B, seq_len*N_patches, C]
        values = self.latent_value_proj(latent_flat)  # [B, seq_len*N_patches, C]

        # Expand query for batch
        query = self.skip_query.expand(B, -1, -1)  # [B, 1, C]

        # Cross-attention to extract conditioning
        attended, _ = self.multihead_attn(query, keys, values)  # [B, 1, C]

        # Extract FiLM parameters
        gamma = self.film_scale(attended)  # [B, 1, C]
        beta = self.film_shift(attended)   # [B, 1, C]

        # Apply FiLM conditioning: gamma * x + beta
        # skip_features: [B, C, L], gamma/beta: [B, 1, C]
        gamma = gamma.transpose(1, 2)  # [B, C, 1]
        beta = beta.transpose(1, 2)    # [B, C, 1]

        conditioned_features = gamma * skip_features + beta

        return conditioned_features


class CleanResidualBlock(nn.Module):
    """Clean residual block without time embedding or FiLM conditioning."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without conditioning.

        Args:
            x: Input tensor [B, C, L]
        Returns:
            Output tensor [B, out_channels, L]
        """
        h = self.norm1(x)
        h = F.gelu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.conv2(h)

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


class DirectGenUNet(nn.Module):
    """Clean U-Net for direct gesture generation with FiLM-conditioned skip connections."""

    def __init__(self, input_dim: int = 512, channels: tuple = (64, 128, 256, 512),
                 d_latent: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.channels = channels

        # Input projection from sequence features to first U-Net channel
        self.input_proj = nn.Conv1d(input_dim, channels[0], 7, padding=3)

        # Encoder (clean, no conditioning)
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.enc_blocks.append(CleanResidualBlock(channels[i], channels[i+1], dropout))
            self.downs.append(nn.Conv1d(channels[i+1], channels[i+1], 4, stride=2, padding=1))

        # Bottleneck
        self.mid_block1 = CleanResidualBlock(channels[-1], channels[-1], dropout)
        self.attn = AttentionBlock(channels[-1])
        self.mid_block2 = CleanResidualBlock(channels[-1], channels[-1], dropout)

        # FiLM conditioning for skip connections
        # Skip connections are saved after each encoder block, so they have output channels
        # Encoder blocks transform: 64→128, 128→256, 256→512
        # Skips saved have channels: [128, 256, 512] (output channels of encoder blocks)
        # Decoder consumes in reverse order: [512, 256, 128]
        self.skip_bridges = nn.ModuleList()
        encoder_output_channels = channels[1:]  # [128, 256, 512]
        for ch in reversed(encoder_output_channels):  # [512, 256, 128] to match decoder order
            self.skip_bridges.append(FiLMSkipBridge(ch, d_latent))

        # Decoder (clean, no conditioning)
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        rev_channels = list(reversed(channels))
        for i in range(len(rev_channels) - 1):
            self.ups.append(nn.ConvTranspose1d(rev_channels[i], rev_channels[i+1], 4, stride=2, padding=1))
            # Skip connections double the input channels for decoder blocks
            self.dec_blocks.append(CleanResidualBlock(rev_channels[i] + rev_channels[i+1], rev_channels[i+1], dropout))

        # Output heads
        self.coord_head = nn.Sequential(
            nn.Conv1d(channels[0], 32, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 2, 3, padding=1)  # Direct coordinate prediction
        )
        self.pen_head = nn.Sequential(
            nn.Conv1d(channels[0], 32, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 1, 3, padding=1)
            # No activation - output logits for BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor, latent_frames: torch.Tensor,
                sequence_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of clean U-Net with FiLM-conditioned skip connections.

        Args:
            x: Input sequence features [B, L, input_dim]
            latent_frames: JEPA latent frames [B, N_max+1, N_patches, D_latent]
            sequence_length: Active sequence length
        Returns:
            Dict containing:
                'coordinates': Predicted absolute coordinates [B, L, 2]
                'pen_state': Predicted pen state logits [B, L, 1]
        """
        # Convert to Conv1d format: [B, L, C] → [B, C, L]
        x = x.transpose(1, 2)  # [B, input_dim, L]
        x = self.input_proj(x)  # [B, channels[0], L]

        # Encoder with skip connections (no conditioning here)
        skips = []
        for block, down in zip(self.enc_blocks, self.downs):
            x = block(x)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.mid_block1(x)
        x = self.attn(x)
        x = self.mid_block2(x)

        # Decoder with FiLM-conditioned skip connections
        for i, (up, block, skip, skip_bridge) in enumerate(zip(self.ups, self.dec_blocks, reversed(skips), self.skip_bridges)):
            x = up(x)

            # Handle size mismatch between upsampled x and skip connection
            if x.shape[2] != skip.shape[2]:
                if x.shape[2] > skip.shape[2]:
                    x = x[:, :, :skip.shape[2]]  # Crop x to match skip
                else:
                    # Pad x to match skip
                    padding = skip.shape[2] - x.shape[2]
                    x = F.pad(x, (0, padding))


            # Apply FiLM conditioning to skip connection
            conditioned_skip = skip_bridge(skip, latent_frames, sequence_length)

            # Concatenate and process
            x = torch.cat([x, conditioned_skip], dim=1)
            x = block(x)

        # Output heads
        coords = self.coord_head(x).transpose(1, 2)    # [B, L, 2] - absolute coordinates
        pen = self.pen_head(x).transpose(1, 2)         # [B, L, 1] - logits

        return {"coordinates": coords, "pen_state": pen}


class DirectGestureModel(nn.Module):
    """Complete direct gesture generation model with JEPA conditioning."""

    def __init__(self, d_latent: int = 1024, d_model: int = 512,
                 channels: tuple = None, max_segments: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.d_latent = d_latent
        self.d_model = d_model
        self.max_segments = max_segments
        self.target_length = max_segments * 250

        # JEPA latent to sequence processor
        self.jepa_processor = JEPAToSequenceProcessor(
            d_latent=d_latent,
            d_model=d_model,
            max_segments=max_segments,
            dropout=dropout
        )

        # U-Net backbone
        if channels is None:
            channels = (64, 128, 256, 512)
        self.unet = DirectGenUNet(
            input_dim=d_model,
            channels=channels,
            d_latent=d_latent,
            dropout=dropout
        )

    def forward(self, latent_frames: torch.Tensor, sequence_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for direct gesture generation.

        Args:
            latent_frames: JEPA latent frames [B, max_segments+1, N_patches, D_latent]
            sequence_length: Active sequence length (1 to max_segments)
        Returns:
            Dict containing:
                'coordinates': Predicted absolute coordinates [B, max_segments*250, 2]
                'pen_state': Predicted pen state logits [B, max_segments*250, 1]
        """
        # Convert JEPA latents to sequence features
        sequence_features = self.jepa_processor(latent_frames, sequence_length)  # [B, L, d_model]

        # Generate gestures through U-Net
        output = self.unet(sequence_features, latent_frames, sequence_length)

        return output

    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor,
                    sequence_length: Optional[int] = None, coordinate_weight: float = 1.0,
                    pen_weight: float = 1.0, delta_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """Compute direct generation loss with optional sequence length masking.

        Args:
            predictions: Model predictions {'coordinates': [B, L, 2], 'pen_state': [B, L, 1]}
            targets: Ground truth gestures [B, L, 3] (x, y, pen)
            sequence_length: Active sequence length for masking
        Returns:
            Dict with loss components
        """
        pred_coords = predictions['coordinates']  # [B, L, 2]
        pred_pen = predictions['pen_state']       # [B, L, 1]

        target_coords = targets[..., :2]  # [B, L, 2]
        target_pen = targets[..., 2:3]    # [B, L, 1]

        # Create sequence mask if length is specified
        if sequence_length is not None:
            mask_length = sequence_length * 250
            mask = torch.zeros_like(targets[..., 0])  # [B, L]
            mask[:, :mask_length] = 1.0
            coord_mask = mask.unsqueeze(-1)  # [B, L, 1]
            pen_mask = mask.unsqueeze(-1)    # [B, L, 1]
        else:
            coord_mask = torch.ones_like(target_coords[..., :1])  # [B, L, 1]
            pen_mask = torch.ones_like(target_pen)                # [B, L, 1]

        # Coordinate loss (MSE with masking)
        coord_loss = F.mse_loss(pred_coords * coord_mask, target_coords * coord_mask, reduction='sum')
        coord_loss = coord_loss / coord_mask.sum().clamp(min=1)

        # Pen loss (BCE with masking)
        pen_loss = F.binary_cross_entropy_with_logits(
            pred_pen * pen_mask, target_pen * pen_mask, reduction='sum'
        )
        pen_loss = pen_loss / pen_mask.sum().clamp(min=1)

        # Delta loss: Compare frame-to-frame differences in union of pen-down regions
        delta_loss = torch.tensor(0.0, device=pred_coords.device)
        delta_mask_coverage = 0.0

        if pred_coords.shape[1] > 1:  # Need at least 2 frames for deltas
            # Compute deltas (frame-to-frame differences)
            target_deltas = target_coords[:, 1:] - target_coords[:, :-1]  # [B, L-1, 2]
            pred_deltas = pred_coords[:, 1:] - pred_coords[:, :-1]        # [B, L-1, 2]

            # Get pen states for union mask
            target_pen_binary = (target_pen > 0.5).float()  # [B, L, 1]
            pred_pen_probs = torch.sigmoid(pred_pen)
            pred_pen_binary = (pred_pen_probs > 0.5).float()  # [B, L, 1]

            # Create union mask for delta positions (use both current and next frame)
            target_pen_union_curr = target_pen_binary[:, :-1]  # [B, L-1, 1] - current frame
            target_pen_union_next = target_pen_binary[:, 1:]   # [B, L-1, 1] - next frame
            pred_pen_union_curr = pred_pen_binary[:, :-1]      # [B, L-1, 1] - current frame
            pred_pen_union_next = pred_pen_binary[:, 1:]       # [B, L-1, 1] - next frame

            # Union: pen down in either target or prediction, in either current or next frame
            union_mask = torch.clamp(
                target_pen_union_curr + target_pen_union_next +
                pred_pen_union_curr + pred_pen_union_next,
                max=1.0
            )  # [B, L-1, 1]

            # Apply sequence length mask to deltas if specified
            if sequence_length is not None:
                delta_seq_mask_length = mask_length - 1  # L-1 for deltas
                delta_seq_mask = torch.zeros_like(union_mask[..., 0])  # [B, L-1]
                if delta_seq_mask_length > 0:
                    delta_seq_mask[:, :delta_seq_mask_length] = 1.0
                delta_seq_mask = delta_seq_mask.unsqueeze(-1)  # [B, L-1, 1]
                union_mask = union_mask * delta_seq_mask

            # Compute delta loss only in union regions
            if union_mask.sum() > 0:
                delta_loss = F.mse_loss(
                    pred_deltas * union_mask,
                    target_deltas * union_mask,
                    reduction='sum'
                )
                delta_loss = delta_loss / union_mask.sum().clamp(min=1)
                delta_mask_coverage = union_mask.mean().item()

        # Compute pen state statistics
        with torch.no_grad():
            pred_pen_probs = torch.sigmoid(pred_pen)
            target_pen_mean = (target_pen * pen_mask).sum() / pen_mask.sum().clamp(min=1)
            pred_pen_mean = (pred_pen_probs * pen_mask).sum() / pen_mask.sum().clamp(min=1)

        # Apply weights to individual loss components
        weighted_coord_loss = coordinate_weight * coord_loss
        weighted_pen_loss = pen_weight * pen_loss
        weighted_delta_loss = delta_weight * delta_loss
        total_loss = weighted_coord_loss + weighted_pen_loss + weighted_delta_loss

        return {
            'coordinate_loss': coord_loss,
            'pen_loss': pen_loss,
            'delta_loss': delta_loss,
            'weighted_coordinate_loss': weighted_coord_loss,
            'weighted_pen_loss': weighted_pen_loss,
            'weighted_delta_loss': weighted_delta_loss,
            'total_loss': total_loss,
            'target_pen_ratio': target_pen_mean.item(),
            'pred_pen_ratio': pred_pen_mean.item(),
            'mask_coverage': coord_mask.mean().item(),
            'delta_mask_coverage': delta_mask_coverage
        }

    def generate(self, latent_frames: torch.Tensor, sequence_length: Optional[int] = None) -> torch.Tensor:
        """Generate gesture sequences from JEPA latent frames.

        Args:
            latent_frames: JEPA latent frames [B, max_segments+1, N_patches, D_latent]
            sequence_length: Active sequence length
        Returns:
            Generated gestures [B, L, 3] with coordinates and pen states
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(latent_frames, sequence_length)

            coords = predictions['coordinates']  # [B, L, 2]
            pen_logits = predictions['pen_state']  # [B, L, 1]
            pen_probs = torch.sigmoid(pen_logits)

            # Combine coordinates and pen states
            gestures = torch.cat([coords, pen_probs], dim=-1)  # [B, L, 3]

            return gestures


# Legacy classes (keep for compatibility but not used)
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
