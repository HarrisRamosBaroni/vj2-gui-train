import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import h5py

from latent_action_model.vae import LatentActionVAE, load_model_from_config

logger = logging.getLogger(__name__)


class LAMAttentionVisualizer:
    """Visualize attention patterns in LAM encoder."""

    def __init__(
        self,
        model_path: str,
        config_path: str = None,
        device: str = None,
        use_wandb: bool = True,
        wandb_project: str = "lam-attention-vis",
        wandb_run_name: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_wandb = use_wandb

        # Load LAM model
        self.model = self._load_model(model_path, config_path)

        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name or f"attention_vis_{self._get_timestamp()}",
                config={
                    "model_path": model_path,
                    "device": self.device
                }
            )

    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _load_model(self, model_path: str, config_path: str = None) -> LatentActionVAE:
        """Load LAM model from checkpoint."""
        logger.info(f"Loading LAM model from {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Start with checkpoint config if available, otherwise use defaults
        config = checkpoint.get('config', {
            'latent_dim': 1024,
            'action_dim': 128,
            'embed_dim': 512,
            'encoder_depth': 3,
            'decoder_depth': 3,
            'encoder_heads': 8,
            'decoder_heads': 8
        })

        # Override with YAML config if available
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Extract parameters from wandb YAML format
            args_dict = {}
            if '_wandb' in yaml_config and 'value' in yaml_config['_wandb'] and 'e' in yaml_config['_wandb']['value']:
                # Get the run info (first entry)
                e_dict = yaml_config['_wandb']['value']['e']
                run_info = next(iter(e_dict.values()))
                args = run_info['args']

                # Parse args into dict
                for i in range(0, len(args), 2):
                    if i + 1 < len(args):
                        key = args[i].lstrip('-')
                        value = args[i + 1]
                        args_dict[key] = value

            # Update config with YAML values
            config.update({
                'latent_dim': yaml_config.get('latent_dim', {}).get('value', config.get('latent_dim', 1024)),
                'action_dim': yaml_config.get('action_dim', {}).get('value', config.get('action_dim', 128)),
                'embed_dim': int(args_dict.get('embed_dim', config.get('embed_dim', 512))),
                'encoder_depth': int(args_dict.get('encoder_depth', config.get('encoder_depth', 16))),
                'decoder_depth': int(args_dict.get('decoder_depth', config.get('decoder_depth', 16))),
                'encoder_heads': int(args_dict.get('encoder_heads', config.get('encoder_heads', 16))),
                'decoder_heads': int(args_dict.get('decoder_heads', config.get('decoder_heads', 16))),
                'kl_weight': float(args_dict.get('kl_weight', config.get('kl_weight', 0.0005)))
            })

            logger.info(f"Loaded config from YAML: {config}")
            logger.info(f"Args dict from YAML: {args_dict}")
        else:
            logger.info(f"Using checkpoint config: {config}")

        model = LatentActionVAE(**config)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(self.device)

        model.eval()
        logger.info(f"LAM model loaded successfully on {self.device}")
        return model

    def _load_corresponding_images(self, session_id: str, traj_idx: int,
                                 T_jepa: int, image_dir: str) -> Optional[torch.Tensor]:
        """
        Load corresponding images for a JEPA trajectory sequence.

        Args:
            session_id: Session identifier
            traj_idx: Trajectory index within session
            T_jepa: Number of JEPA embeddings (e.g., 8)
            image_dir: Directory containing image H5 files

        Returns:
            Tensor of shape [T_jepa, H, W, C] containing representative frames
            for each JEPA embedding (using the first frame of each tublet)
        """
        image_path = Path(image_dir) / f"{session_id}_images.h5"

        if not image_path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return None

        try:
            with h5py.File(image_path, 'r') as f:
                if 'images' not in f:
                    logger.warning(f"No 'images' key in {image_path}")
                    return None

                # JEPA tublet-to-frame mapping: 2 frames per tublet
                # T_jepa embeddings represent 2*T_jepa real frames
                num_real_frames = T_jepa * 2
                available_frames = f['images'].shape[1]

                if num_real_frames > available_frames:
                    logger.warning(f"Requested {num_real_frames} real frames for {T_jepa} JEPA embeddings, "
                                   f"but only {available_frames} available.")
                    num_real_frames = available_frames

                images = []
                for jepa_idx in range(T_jepa):
                    # Each JEPA embedding corresponds to frames [2*jepa_idx, 2*jepa_idx+1]
                    # Use the first frame of each tublet as representative
                    frame_idx = 2 * jepa_idx

                    if frame_idx < num_real_frames:
                        img = f['images'][traj_idx, frame_idx]  # [H, W, C]
                        images.append(torch.from_numpy(img))
                    else:
                        # If we run out of frames, duplicate the last available frame
                        logger.warning(f"Frame {frame_idx} not available, using last frame")
                        if images:
                            images.append(images[-1].clone())

                if images:
                    return torch.stack(images)  # [T_jepa, H, W, C]
                else:
                    return None

        except Exception as e:
            logger.warning(f"Error loading images from {image_path}: {e}")
            return None

    def _load_trajectory_data(self, processed_data_dir: str, manifest_path: str,
                             split_name: str, n_trajectories: int = 20) -> List[Tuple]:
        """Load specific trajectories from dataset."""
        # Load manifest to get file list
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        if split_name not in manifest['splits']:
            raise ValueError(f"Split '{split_name}' not found in manifest")

        file_list = manifest['splits'][split_name][:n_trajectories]
        trajectories = []
        processed_dir = Path(processed_data_dir)

        for filename in file_list:
            h5_path = processed_dir / filename
            if not h5_path.exists():
                logger.warning(f"File not found: {h5_path}")
                continue

            try:
                with h5py.File(h5_path, 'r') as f:
                    embeddings = f['embeddings'][:]  # [n_traj, T, N, D]

                    session_id = h5_path.stem

                    for traj_idx in range(embeddings.shape[0]):
                        trajectories.append((
                            session_id,
                            traj_idx,
                            embeddings[traj_idx]  # [T, N, D]
                        ))

                        if len(trajectories) >= n_trajectories:
                            break

            except Exception as e:
                logger.warning(f"Error loading {h5_path}: {e}")
                continue

        logger.info(f"Loaded {len(trajectories)} trajectories from {split_name} split")
        return trajectories

    def _extract_attention_weights(self, z_sequence: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention weights from LAM encoder by hooking into transformer blocks."""
        attention_weights = []

        def attention_hook(module, input, output):
            # For nn.MultiheadAttention, we need to capture the attention weights
            # We'll modify the forward pass to return attention weights
            pass

        # Register hooks on attention modules
        hooks = []
        for i, block in enumerate(self.model.encoder.blocks):
            # We need to manually compute attention to get weights
            # Let's override the forward method temporarily
            pass

        # Since nn.MultiheadAttention doesn't easily expose attention weights,
        # let's compute attention manually
        B, T, N, D = z_sequence.shape

        # Apply z-score normalization as expected by LAM
        z_normalized = F.layer_norm(z_sequence, (z_sequence.size(-1),))

        # Forward through projection and positional embeddings (copied from LAMEncoder)
        x = self.model.encoder.patch_proj(z_normalized)
        x = F.layer_norm(x, (self.model.encoder.embed_dim,))

        # Add positional embeddings
        spatial_pos = self.model.encoder.spatial_pos_embed.unsqueeze(0).unsqueeze(0)
        x = x + spatial_pos.to(x.device)

        temporal_pos = self.model.encoder.temporal_pos_embed[:T].unsqueeze(0).unsqueeze(2)
        x = x + temporal_pos.to(x.device)

        # Reshape to sequence: [B, T*N, embed_dim]
        x = x.reshape(B, T * N, self.model.encoder.embed_dim)

        # Manually compute attention for each block
        block_attentions = []
        for block in self.model.encoder.blocks:
            # Pre-norm
            y = block.norm1(x)

            # Manual attention computation
            embed_dim = y.size(-1)
            num_heads = block.attn.num_heads
            head_dim = embed_dim // num_heads
            scaling = head_dim ** -0.5

            # Get Q, K, V from the linear layers
            q = block.attn.in_proj_weight[:embed_dim] @ y.transpose(-2, -1)  # [B, embed_dim, seq_len]
            k = block.attn.in_proj_weight[embed_dim:2*embed_dim] @ y.transpose(-2, -1)
            v = block.attn.in_proj_weight[2*embed_dim:] @ y.transpose(-2, -1)

            # Reshape for multi-head attention
            q = q.transpose(-2, -1).reshape(B, T*N, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, seq_len, head_dim]
            k = k.transpose(-2, -1).reshape(B, T*N, num_heads, head_dim).transpose(1, 2)
            v = v.transpose(-2, -1).reshape(B, T*N, num_heads, head_dim).transpose(1, 2)

            # Compute attention scores
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scaling  # [B, num_heads, seq_len, seq_len]
            attn_weights = F.softmax(attn_weights, dim=-1)

            block_attentions.append(attn_weights.detach())

            # Apply attention and continue forward pass
            attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, seq_len, head_dim]
            attn_output = attn_output.transpose(1, 2).reshape(B, T*N, embed_dim)
            attn_output = block.attn.out_proj(attn_output)

            # Residual connection
            x = x + attn_output

            # MLP block
            y = block.norm2(x)
            y = block.mlp(y)
            x = x + y

        return block_attentions

    def _compute_last_frame_attention(self, attention_weights: List[torch.Tensor],
                                    T: int, N: int) -> torch.Tensor:
        """Extract and average attention from last frame patches to all previous patches."""
        # attention_weights: List of [B, num_heads, seq_len, seq_len] where seq_len = T*N

        # Indices for last frame patches
        last_frame_start = (T - 1) * N
        last_frame_end = T * N

        # Average across all blocks and heads
        avg_attention = []
        for attn in attention_weights:
            # Extract last frame as queries: [B, num_heads, N, T*N]
            last_frame_attn = attn[:, :, last_frame_start:last_frame_end, :]

            # Average across heads: [B, N, T*N]
            last_frame_attn = last_frame_attn.mean(dim=1)

            avg_attention.append(last_frame_attn)

        # Average across blocks: [B, N, T*N]
        final_attention = torch.stack(avg_attention).mean(dim=0)

        # Exclude attention to self (last frame)
        # Only consider attention to previous frames: [B, N, (T-1)*N]
        prev_frames_attention = final_attention[:, :, :last_frame_start]

        # Average across query patches (all patches in last frame): [B, (T-1)*N]
        avg_query_attention = prev_frames_attention.mean(dim=1)

        return avg_query_attention

    def _extract_decoder_attention_weights(self, z_past: torch.Tensor, action_latent: torch.Tensor) -> List[torch.Tensor]:
        """Extract cross-attention weights from LAM decoder by manually computing attention."""
        B, T_past, N, D = z_past.shape

        # Apply z-score normalization as expected by LAM
        z_normalized = F.layer_norm(z_past, (z_past.size(-1),))

        # === Prepare Memory (same as decoder forward) ===
        z_embed = self.model.decoder.patch_proj(z_normalized)
        z_embed = F.layer_norm(z_embed, (self.model.decoder.embed_dim,))

        # Add positional embeddings
        spatial_pos = self.model.decoder.spatial_pos_embed.unsqueeze(0).unsqueeze(0)
        z_embed = z_embed + spatial_pos.to(z_embed.device)

        temporal_pos = self.model.decoder.temporal_pos_embed[:T_past].unsqueeze(0).unsqueeze(2)
        z_embed = z_embed + temporal_pos.to(z_embed.device)

        # Add action conditioning
        action_embed = self.model.decoder.action_proj(action_latent)
        action_embed = F.layer_norm(action_embed, (self.model.decoder.embed_dim,))
        action_embed = action_embed.unsqueeze(1).unsqueeze(1)
        z_embed = z_embed + action_embed

        # Flatten to sequence for memory
        memory = z_embed.reshape(B, T_past * N, self.model.decoder.embed_dim)

        # === Prepare Queries ===
        queries = self.model.decoder.query_embed.expand(B, -1, -1)
        query_pos = self.model.decoder.query_pos_embed.unsqueeze(0)
        queries = queries + query_pos
        queries = queries + action_embed.squeeze(1)

        # === Manual Cross-Attention Computation ===
        block_cross_attentions = []
        for block in self.model.decoder.blocks:
            # Self-attention first (we could visualize this too, but focus on cross-attention)
            y = block.norm1(queries)
            y, _ = block.self_attn(y, y, y)
            queries = queries + y

            # Cross-attention: queries attend to memory
            y = block.norm2(queries)

            # Manual cross-attention computation to extract weights
            embed_dim = y.size(-1)
            num_heads = block.cross_attn.num_heads
            head_dim = embed_dim // num_heads
            scaling = head_dim ** -0.5

            # Get Q, K, V from the cross-attention linear layers
            q = block.cross_attn.in_proj_weight[:embed_dim] @ y.transpose(-2, -1)  # [B, embed_dim, N]
            k = block.cross_attn.in_proj_weight[embed_dim:2*embed_dim] @ memory.transpose(-2, -1)
            v = block.cross_attn.in_proj_weight[2*embed_dim:] @ memory.transpose(-2, -1)

            # Reshape for multi-head attention
            q = q.transpose(-2, -1).reshape(B, N, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
            k = k.transpose(-2, -1).reshape(B, T_past*N, num_heads, head_dim).transpose(1, 2)
            v = v.transpose(-2, -1).reshape(B, T_past*N, num_heads, head_dim).transpose(1, 2)

            # Compute cross-attention scores: queries attend to memory
            cross_attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scaling  # [B, num_heads, N, T_past*N]
            cross_attn_weights = F.softmax(cross_attn_weights, dim=-1)

            block_cross_attentions.append(cross_attn_weights.detach())

            # Apply cross-attention and continue
            cross_attn_output = torch.matmul(cross_attn_weights, v)  # [B, num_heads, N, head_dim]
            cross_attn_output = cross_attn_output.transpose(1, 2).reshape(B, N, embed_dim)
            cross_attn_output = block.cross_attn.out_proj(cross_attn_output)
            queries = queries + cross_attn_output

            # MLP block
            y = block.norm3(queries)
            y = block.mlp(y)
            queries = queries + y

        return block_cross_attentions

    def _compute_decoder_attention_summary(self, cross_attention_weights: List[torch.Tensor],
                                         T_past: int, N: int) -> torch.Tensor:
        """Compute summary of decoder cross-attention across blocks and heads."""
        # cross_attention_weights: List of [B, num_heads, N, T_past*N]

        # Average across all blocks and heads
        avg_attention = []
        for attn in cross_attention_weights:
            # Average across heads: [B, N, T_past*N]
            avg_attn = attn.mean(dim=1)
            avg_attention.append(avg_attn)

        # Average across blocks: [B, N, T_past*N]
        final_attention = torch.stack(avg_attention).mean(dim=0)

        return final_attention

    def _visualize_decoder_attention_on_images(self, decoder_attention: torch.Tensor,
                                             past_images: torch.Tensor, target_image: torch.Tensor,
                                             T_past: int, N: int, session_id: str, traj_idx: int) -> plt.Figure:
        """
        Visualize decoder cross-attention: how query patches attend to past visual patches.

        Args:
            decoder_attention: [B, N, T_past*N] - cross-attention from queries to memory
            past_images: [T_past, H, W, C] - representative frames for past JEPA embeddings (frames 0 to T-2)
            target_image: [H, W, C] - the target frame being predicted (frame T-1)
            T_past: Number of past JEPA embeddings (T-1)
            N: Number of patches per frame
        """
        B = decoder_attention.shape[0]
        grid_size = int(np.sqrt(N))

        # Take first batch element
        attn = decoder_attention[0].cpu().numpy()  # [N, T_past*N]

        # For visualization, average attention across all query patches to see overall pattern
        avg_attn_per_frame = []
        for t in range(T_past):
            # Extract attention to frame t: [N, N] (queries to patches in frame t)
            frame_attn = attn[:, t*N:(t+1)*N]  # [N_queries, N_memory]
            # Average across query patches to get attention to each memory patch
            avg_frame_attn = frame_attn.mean(axis=0)  # [N_memory]
            avg_attn_per_frame.append(avg_frame_attn)

        # Create visualization with T_past+1 columns (past frames + target frame)
        fig, axes = plt.subplots(3, T_past + 1, figsize=(4*(T_past + 1), 12))
        if T_past == 0:  # Edge case
            axes = axes.reshape(3, 1)

        # Show past frames and their attention
        for t in range(T_past):
            # Top row: Past frames
            ax_img = axes[0, t]

            if t < len(past_images):
                img = past_images[t]

                # Convert to displayable format
                if img.dim() == 3 and img.shape[-1] == 3:
                    img_np = img.cpu().numpy()
                else:
                    img_np = img.permute(1, 2, 0).cpu().numpy()

                # Normalize to [0, 1]
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
                elif img_np.min() < 0:
                    img_np = (img_np + 1.0) / 2.0

                ax_img.imshow(img_np)
                ax_img.set_title(f"Past JEPA {t}\n(Frames {t*2}-{t*2+1})", fontsize=10)
                ax_img.axis('off')

            # Middle row: Cross-attention from queries to this past frame
            ax_attn = axes[1, t]

            # Get attention to this frame
            frame_attn = avg_attn_per_frame[t]  # [N]
            attn_map = frame_attn.reshape(grid_size, grid_size)

            # Show attention heatmap
            im = ax_attn.imshow(attn_map, cmap='hot', alpha=0.7)
            ax_attn.set_title(f"Query Attention to Past {t}", fontsize=10)

            # Add colorbar
            plt.colorbar(im, ax=ax_attn, fraction=0.046, pad=0.04)

            # Overlay on image if available
            if t < len(past_images):
                img_size = img_np.shape[:2]
                attn_resized = cv2.resize(attn_map, (img_size[1], img_size[0]))
                ax_attn.imshow(img_np, alpha=0.3)
                ax_attn.imshow(attn_resized, cmap='hot', alpha=0.7)

            # Bottom row: Empty for past frames
            ax_bottom = axes[2, t]
            ax_bottom.axis('off')

        # Last column: Show target frame being predicted
        ax_target = axes[0, T_past]

        # Convert target image to displayable format
        if target_image.dim() == 3 and target_image.shape[-1] == 3:
            target_np = target_image.cpu().numpy()
        else:
            target_np = target_image.permute(1, 2, 0).cpu().numpy()

        # Normalize to [0, 1]
        if target_np.max() > 1.0:
            target_np = target_np / 255.0
        elif target_np.min() < 0:
            target_np = (target_np + 1.0) / 2.0

        ax_target.imshow(target_np)
        ax_target.set_title(f"TARGET\nJEPA {T_past} (Frames {T_past*2}-{T_past*2+1})\nBEING PREDICTED", fontsize=10, fontweight='bold', color='red')
        ax_target.axis('off')

        # Middle row for target: Show explanation
        ax_target_mid = axes[1, T_past]
        ax_target_mid.text(0.5, 0.5, 'DECODER\nTARGET\n\n(Not visible to\nquery tokens)',
                          horizontalalignment='center', verticalalignment='center',
                          transform=ax_target_mid.transAxes, fontsize=12, fontweight='bold', color='red')
        ax_target_mid.axis('off')

        # Bottom row for target: Empty
        ax_target_bottom = axes[2, T_past]
        ax_target_bottom.axis('off')

        plt.suptitle(f'LAM Decoder Cross-Attention - {session_id}_{traj_idx}\n'
                    f'Query Tokens (Predicting Frame {T_past}) Attend to Past Frames (0-{T_past-1}) + Action',
                    fontsize=14)
        plt.tight_layout()

        return fig

    def _visualize_attention_on_images(self, attention_weights: torch.Tensor,
                                     images: torch.Tensor, T: int, N: int,
                                     session_id: str, traj_idx: int) -> plt.Figure:
        """
        Visualize attention weights overlaid on corresponding images.

        Args:
            attention_weights: [B, (T-1)*N] - attention from last JEPA embedding to all previous patches
            images: [T, H, W, C] - representative frames for each JEPA embedding
            T: Number of JEPA embeddings (temporal dimension)
            N: Number of patches per frame (256 = 16x16)
            session_id: Session identifier
            traj_idx: Trajectory index
        """
        B = attention_weights.shape[0]
        grid_size = int(np.sqrt(N))  # Assuming square grid (16x16 = 256)

        # Take first batch element
        attn = attention_weights[0].cpu().numpy()  # [(T-1)*N]

        # Reshape to [T-1, N] to separate by JEPA embedding
        attn = attn.reshape(T-1, N)

        # Create visualization with proper JEPA frame mapping
        fig, axes = plt.subplots(2, T, figsize=(4*T, 8))
        if T == 1:
            axes = axes.reshape(2, 1)

        for jepa_idx in range(T):
            # Top row: Representative frames for each JEPA embedding
            ax_img = axes[0, jepa_idx]

            if jepa_idx < len(images):
                img = images[jepa_idx]  # Representative frame for this JEPA embedding

                # Convert to displayable format
                if img.dim() == 3 and img.shape[-1] == 3:
                    img_np = img.cpu().numpy()
                else:
                    img_np = img.permute(1, 2, 0).cpu().numpy()

                # Normalize to [0, 1]
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
                elif img_np.min() < 0:
                    img_np = (img_np + 1.0) / 2.0

                ax_img.imshow(img_np)

                # Calculate real frame range for this JEPA embedding
                real_frame_start = jepa_idx * 2
                real_frame_end = jepa_idx * 2 + 1
                ax_img.set_title(f"JEPA {jepa_idx}\n(Frames {real_frame_start}-{real_frame_end})", fontsize=10)
                ax_img.axis('off')
            else:
                ax_img.axis('off')
                ax_img.set_title(f"JEPA {jepa_idx}\n(unavailable)", fontsize=10)

            # Bottom row: Attention visualization
            ax_attn = axes[1, jepa_idx]

            if jepa_idx < T-1:  # Only show attention for previous JEPA embeddings
                # Get attention for this JEPA embedding
                jepa_attn = attn[jepa_idx]  # [N]

                # Reshape to spatial grid
                attn_map = jepa_attn.reshape(grid_size, grid_size)

                # Show attention heatmap
                im = ax_attn.imshow(attn_map, cmap='hot', alpha=0.7)
                ax_attn.set_title(f"Attention from Last JEPA", fontsize=10)

                # Add colorbar
                plt.colorbar(im, ax=ax_attn, fraction=0.046, pad=0.04)

                # Overlay on original image if available
                if jepa_idx < len(images):
                    # Resize attention map to image size
                    img_size = img_np.shape[:2]
                    attn_resized = cv2.resize(attn_map, (img_size[1], img_size[0]))
                    ax_attn.imshow(img_np, alpha=0.3)
                    ax_attn.imshow(attn_resized, cmap='hot', alpha=0.7)
            else:
                # Last JEPA embedding - show as query
                ax_attn.text(0.5, 0.5, 'QUERY\n(Last JEPA Embedding)',
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax_attn.transAxes, fontsize=12, fontweight='bold')
                ax_attn.axis('off')

        plt.suptitle(f'LAM Encoder Attention - {session_id}_{traj_idx}\n'
                    f'How Last JEPA Embedding (frames {2*(T-1)}-{2*(T-1)+1}) Attends to Previous Embeddings',
                    fontsize=14)
        plt.tight_layout()

        return fig

    def visualize_trajectories(
        self,
        processed_data_dir: str,
        manifest_path: str,
        image_dir: str,
        n_trajectories: int = 10,
        split_name: str = 'validation'
    ) -> Dict[str, float]:
        """
        Visualize attention patterns for N trajectories.

        Args:
            processed_data_dir: Directory containing preprocessed latent data
            manifest_path: Path to manifest file for data splits
            image_dir: Directory containing image h5 files
            n_trajectories: Number of trajectories to visualize
            split_name: Which data split to use

        Returns:
            Dictionary with processing metrics
        """
        logger.info(f"Visualizing attention for {n_trajectories} trajectories from {split_name} split...")

        # Load trajectory data
        trajectories = self._load_trajectory_data(
            processed_data_dir, manifest_path, split_name, n_trajectories
        )

        processed_count = 0

        with torch.no_grad():
            for session_id, traj_idx, embeddings in tqdm(trajectories, desc="Processing trajectories"):
                try:
                    # Convert to tensor and add batch dimension
                    z_sequence = torch.from_numpy(embeddings).float().unsqueeze(0).to(self.device)  # [1, T, N, D]
                    T, N, D = embeddings.shape

                    if T < 2:  # Need at least 2 frames
                        logger.warning(f"Trajectory {session_id}_{traj_idx} has only {T} frames, skipping")
                        continue

                    # === ENCODER ATTENTION ===
                    # Extract encoder attention weights
                    encoder_attention_weights = self._extract_attention_weights(z_sequence)
                    # Compute last frame attention
                    last_frame_attention = self._compute_last_frame_attention(encoder_attention_weights, T, N)

                    # === DECODER ATTENTION ===
                    # Simulate decoder forward pass: predict last frame from previous frames + action
                    if T >= 2:  # Need at least 2 frames for decoder (past frames + frame to predict)
                        # Use frames 0 to T-2 as past context (first T-1 frames)
                        z_past = z_sequence[:, :-1]  # [1, T-1, N, D] - frames 0 to T-2
                        # Target is the last frame: z_sequence[:, -1]  # [1, N, D] - frame T-1

                        # Get the action for the transition from (T-2) -> (T-1)
                        mu, logvar = self.model.encode(z_sequence)  # [1, T-1, A] - all transitions
                        action_latent = self.model.reparameterize(mu[:, -1:], logvar[:, -1:])  # [1, A] - last transition action

                        # Extract decoder cross-attention weights
                        decoder_attention_weights = self._extract_decoder_attention_weights(z_past, action_latent.squeeze(1))
                        decoder_attention_summary = self._compute_decoder_attention_summary(decoder_attention_weights, T-1, N)

                    # Load corresponding images
                    images = self._load_corresponding_images(session_id, traj_idx, T, image_dir)

                    if images is not None:
                        # Create encoder visualization
                        encoder_fig = self._visualize_attention_on_images(
                            last_frame_attention, images, T, N, session_id, traj_idx
                        )

                        # Log encoder attention to wandb
                        if self.use_wandb:
                            wandb.log({
                                f"encoder_attention/{split_name}/trajectory_{session_id}_{traj_idx}": wandb.Image(encoder_fig)
                            })

                        plt.close(encoder_fig)

                        # Create decoder visualization if we have enough frames
                        if T >= 2:
                            # Use images for past frames only (frames 0 to T-2, which are T-1 frames)
                            past_images = images[:-1]  # [T-1, H, W, C] - frames 0 to T-2
                            target_image = images[-1]   # [H, W, C] - frame T-1 (what decoder is trying to predict)

                            decoder_fig = self._visualize_decoder_attention_on_images(
                                decoder_attention_summary, past_images, target_image, T-1, N, session_id, traj_idx
                            )

                            # Log decoder attention to wandb
                            if self.use_wandb:
                                wandb.log({
                                    f"decoder_attention/{split_name}/trajectory_{session_id}_{traj_idx}": wandb.Image(decoder_fig)
                                })

                            plt.close(decoder_fig)

                        processed_count += 1

                except Exception as e:
                    logger.error(f"Error processing trajectory {session_id}_{traj_idx}: {e}")
                    continue

        metrics = {
            'processed_trajectories': processed_count,
            'total_trajectories': len(trajectories)
        }

        logger.info(f"Attention visualization complete. Processed {processed_count}/{len(trajectories)} trajectories")
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Visualize LAM Encoder Attention Patterns")

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to LAM model checkpoint")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Path to model config file (optional)")

    # Data arguments
    parser.add_argument("--processed_data_dir", type=str, required=True,
                       help="Directory containing preprocessed latent data")
    parser.add_argument("--manifest_path", type=str, required=True,
                       help="Path to manifest file for data splits")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Directory containing image h5 files")

    # Processing arguments
    parser.add_argument("--n_trajectories", type=int, default=10,
                       help="Number of trajectories to visualize")
    parser.add_argument("--split_name", type=str, default="validation",
                       help="Data split to use (train/validation/test)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu), auto-detect if not specified")

    # Wandb arguments
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="lam-attention-vis",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create visualizer
    visualizer = LAMAttentionVisualizer(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )

    # Run visualization
    results = visualizer.visualize_trajectories(
        processed_data_dir=args.processed_data_dir,
        manifest_path=args.manifest_path,
        image_dir=args.image_dir,
        n_trajectories=args.n_trajectories,
        split_name=args.split_name
    )

    # Save results
    output_file = f"attention_vis_results_{visualizer._get_timestamp()}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()