import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import logging
logger = logging.getLogger(__name__)
import argparse
import math

import torch
import torch.nn as nn
import torchvision.transforms as T

from gui_world_model.utils.modules import ACBlock as Block
from gui_world_model.utils.modules import build_action_block_causal_attention_mask
from gui_world_model.utils.modules import CrossAttentionBlock
from src.utils.tensors import trunc_normal_
from gui_world_model.encoder import VJEPA2Wrapper
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import (
    ACTION_DIM,
    ACTIONS_PER_BATCH,
    OBSERVATIONS_PER_WINDOW
)

class VJ2GUIPredictor(nn.Module):
    """
    Cross-Attention Action-Conditioned Predictor for GUI control using V-JEPA 2 architecture.
    
    This model implements a two-stage pipeline for action conditioning:
    1. Cross-attention stage: Actions are injected into visual tokens via cross-attention
    2. Self-attention stage: Action-conditioned visual tokens are processed through transformer blocks
    
    Architecture Overview:
    ┌─────────────────────────────┐
    │       z_in: latent tokens   │    ← from V-JEPA
    │       [B, T, H·W, D]        │
    └────────────┬────────────────┘
                 │
                 ▼
         predictor_embed (Linear)
                 │
                 ▼
         z_proj: [B, T, H·W, D]
                 ▼
           reshape to merge time
                 ▼
         z_proj: [B·T, H·W, D]
                 │

    ┌─────────────────────────────┐
    │      a_in: raw action bins  │
    │      [B, T, A_total]        │
    └────────────┬────────────────┘
                 ▼
        reshape → [B, T, M, A]
                 ▼
       action_encoder (Linear)
                 ▼
         a_proj: [B, T, M, D]
                 ▼
           reshape time
                 ▼
         a_proj: [B·T, M, D]
                 │

    ┌────────────────────────────────────────────┐
    │ CrossAttentionBlock × C                    │  ← injects action into z
    │   z_proj ← CrossAttention(z_proj, a_proj)  │
    │   z_proj ← MLP(norm(z_proj))               │
    │   shape: [B·T, H·W, D]                     │
    └────────────────────────────────────────────┘
                 │
       reshape to restore time
                 ▼
       z_ca_out: [B, T, H·W, D]
                 ▼
       flatten spatial+time
                 ▼
           x: [B, T·H·W, D]

    ┌────────────────────────────────────┐
    │ predictor_blocks (ACBlock × L)     │ ← Self-attention
    │   Each block:                      │
    │     x ← SelfAttention(norm(x))     │
    │     x ← MLP(norm(x))               │
    └────────────────────────────────────┘
                 │
                 ▼
         predictor_norm (LN)
                 ▼
         predictor_proj (Linear)
                 ▼
         ẑ_out: [B, T, H·W, D]

    Input Tensors:
        - z (State): A tensor of shape [B, T, N, D] representing the visual state.
            - B: Batch size of video clips.
            - T: Number of time steps (typically frames // tubelet_size).
            - N: Number of visual tokens per frame (e.g., 16x16 = 256).
            - D: Embedding dimension of each token (e.g., 1024).
        - actions (Action): A tensor of shape [B, T, A_total] representing the actions.
            - A_total: The flattened dimension of one "Action Batch"
                      (ACTIONS_PER_BATCH * ACTION_DIM = 250 * 3 = 750).

    Output Tensor:
        - Predicted visual tokens of shape [B, T, N, D].
        
    Key Improvements over Concatenation-based Approach:
        - Multiple action tokens (M=8) preserve more action granularity
        - Cross-attention enables direct action-to-vision conditioning
        - Separates action injection from temporal/spatial modeling
        - More structured and interpretable action influence
    """

    def __init__(
        self,
        img_size=(256, 256),
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        embed_dim=1024,
        predictor_embed_dim=1024,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        use_rope=True,
        action_embed_dim=ACTIONS_PER_BATCH * ACTION_DIM,  # 250*3=750
        num_action_tokens=8,  # Split actions into M=8 tokens
        cross_attn_depth=3,   # Number of cross-attention blocks
        cross_attn_drop_path_rate=0.0,  # Drop path rate for cross-attention
    ):
        super().__init__()

        # Store hyperparameters for saving/loading
        self._config = {
            "img_size": img_size,
            "patch_size": patch_size,
            "num_frames": num_frames,
            "tubelet_size": tubelet_size,
            "embed_dim": embed_dim,
            "predictor_embed_dim": predictor_embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "qkv_bias": qkv_bias,
            "qk_scale": qk_scale,
            "drop_rate": drop_rate,
            "attn_drop_rate": attn_drop_rate,
            "drop_path_rate": drop_path_rate,
            "norm_layer": norm_layer,
            "init_std": init_std,
            "use_silu": use_silu,
            "wide_silu": wide_silu,
            "use_activation_checkpointing": use_activation_checkpointing,
            "use_rope": use_rope,
            "action_embed_dim": action_embed_dim,
            "num_action_tokens": num_action_tokens,
            "cross_attn_depth": cross_attn_depth,
            "cross_attn_drop_path_rate": cross_attn_drop_path_rate,
        }

        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing
        self.num_action_tokens = num_action_tokens
        self.cross_attn_depth = cross_attn_depth

        # Token embeddings
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        
        # Action encoding: Split actions into multiple tokens
        # Calculate the actual dimensions per token accounting for potential padding
        self.action_embed_dim = action_embed_dim
        # If action_embed_dim doesn't divide evenly by num_action_tokens, we'll need padding
        if action_embed_dim % num_action_tokens != 0:
            padding_needed = num_action_tokens - (action_embed_dim % num_action_tokens)
            padded_action_dim = action_embed_dim + padding_needed
        else:
            padded_action_dim = action_embed_dim
        self.action_dim_per_token = padded_action_dim // num_action_tokens
        self.action_encoder = nn.Linear(self.action_dim_per_token, predictor_embed_dim)

        # Cross-attention blocks for action injection
        cross_dpr = [x.item() for x in torch.linspace(0, cross_attn_drop_path_rate, cross_attn_depth)]
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                act_layer=nn.SiLU if use_silu else nn.GELU,
                norm_layer=norm_layer,
            )
            for i in range(cross_attn_depth)
        ])

        # Self-attention blocks for temporal/spatial modeling
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList([
            Block(
                use_rope=use_rope,
                grid_size=self.grid_height,
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.SiLU if use_silu else nn.GELU,
                wide_silu=wide_silu,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)

        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        # Build causal attention mask for self-attention (no action tokens)
        self.attn_mask = build_action_block_causal_attention_mask(
            self.num_frames // self.tubelet_size,
            self.grid_height,
            self.grid_width,
            0,                  # ← action_tokens = 0 (no action tokens in self-attention)
        )
        logger.debug(f"{self.attn_mask.shape=}")
        print(f"{self.attn_mask.shape=}")  # [2048, 2048] where 2048 = T*H*W = 8*16*16

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    # ───────────────────────────────────────────────────────────────────────
    def forward(self, z, actions):
        """
        Cross-Attention Action-Conditioned Forward Pass
        
        Two-stage pipeline:
        1. Cross-attention: Inject actions into visual tokens via cross-attention
        2. Self-attention: Process action-conditioned visual tokens through transformer
        
        Args:
            z:       Visual state tensor [B, T, N, D] where N = H*W
            actions: Action tensor [B, T, A_total] where A_total = 750
            
        Returns:
            Predicted visual tokens [B, T, N, D]
        """
        B, T, N, D = z.shape
        H, W = self.grid_height, self.grid_width
        if N != H * W:
            raise ValueError(f"Mismatch in spatial token count: {N} != {H}*{W}")

        # ═════════ STAGE 1: CROSS-ATTENTION ACTION INJECTION ═════════
        
        # 1. Embed visual tokens
        z_proj = self.predictor_embed(z)  # [B, T, H*W, D]
        logger.debug(f"z_proj.shape: {z_proj.shape}")
        
        # 2. Split actions into multiple tokens and encode each
        # Reshape: [B, T, A_total] → [B, T, M, A_per_token]
        A_total = actions.shape[-1]
        M = self.num_action_tokens
        
        # Apply the same padding logic as in __init__
        if A_total % M != 0:
            pad_size = M - (A_total % M)
            actions = torch.cat([actions, torch.zeros(B, T, pad_size, device=actions.device, dtype=actions.dtype)], dim=-1)
            A_total = actions.shape[-1]
        
        A_per_token = A_total // M
        assert A_per_token == self.action_dim_per_token, f"Action dimension mismatch: {A_per_token} != {self.action_dim_per_token}"
        
        a_split = actions.view(B, T, M, A_per_token)  # [B, T, M, A_per_token]
        logger.debug(f"a_split.shape: {a_split.shape}")
        
        # Encode each action token: [B, T, M, A_per_token] → [B, T, M, D]
        a_proj = self.action_encoder(a_split)  # [B, T, M, D]
        logger.debug(f"a_proj.shape: {a_proj.shape}")
        
        # 3. Reshape for cross-attention: merge time dimension
        z_flat = z_proj.view(B * T, H * W, D)  # [B*T, H*W, D]
        a_flat = a_proj.view(B * T, M, D)      # [B*T, M, D]
        logger.debug(f"z_flat.shape: {z_flat.shape}")
        logger.debug(f"a_flat.shape: {a_flat.shape}")
        
        # 4. Apply cross-attention blocks: visual tokens attend to action tokens
        z_cross = z_flat
        for cross_blk in self.cross_attn_blocks:
            z_cross = cross_blk(z_cross, a_flat)  # z_cross queries, a_flat keys/values
        
        logger.debug(f"z_cross.shape after cross-attention: {z_cross.shape}")
        
        # 5. Reshape back to include time dimension
        z_conditioned = z_cross.view(B, T, H * W, D)  # [B, T, H*W, D]
        
        # ═════════ STAGE 2: SELF-ATTENTION PROCESSING ═════════
        
        # 6. Flatten spatial and time for self-attention (no action tokens)
        x = z_conditioned.flatten(1, 2)  # [B, T*H*W, D]
        logger.debug(f"x.shape for self-attention: {x.shape}")
        
        # 7. Apply self-attention blocks with causal masking (action_tokens=0)
        attn_mask = self.attn_mask[:x.size(1), :x.size(1)].to(z.device, non_blocking=True)
        logger.debug(f"attn_mask.shape: {attn_mask.shape}")
        
        for blk in self.predictor_blocks:
            x = blk(x, mask=None, attn_mask=attn_mask,
                    T=T, H=H, W=W, action_tokens=0)  # ← 0 action tokens
        
        # 8. Reshape back to [B, T, H*W, D] for output
        x = x.view(B, T, H * W, D)
        
        # 9. Final normalization and projection
        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        
        logger.debug(f"Final output shape: {x.shape}")
        return x

    def get_config(self):
        """Returns the model's initialization configuration."""
        return self._config


def load_predictor_model(model_path, device):
    """Loads the VJ2GUIPredictor model from a checkpoint."""
    print(f"Loading predictor from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model_config = None
    if "predictor_config" in checkpoint:
        model_config = checkpoint["predictor_config"]
        # Ensure norm_layer is correctly referenced if it's a string
        if "norm_layer" in model_config and isinstance(model_config["norm_layer"], str):
            if model_config["norm_layer"] == "nn.LayerNorm":
                model_config["norm_layer"] = torch.nn.LayerNorm
        model = VJ2GUIPredictor(**model_config).to(device)
    else:
        # Fallback for older checkpoints without config, assume default depth
        print("⚠️ Checkpoint does not contain model configuration. Using default depth=24, num_frames=OBSERVATIONS_PER_WINDOW, tubelet_size=2")
        model = VJ2GUIPredictor(depth=24, num_frames=OBSERVATIONS_PER_WINDOW, tubelet_size=2).to(device)

    # Handle various checkpoint formats (DDP, simple, etc.)
    if "predictor" in checkpoint:
        state_dict = checkpoint["predictor"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Remove `module.` prefix if present from DDP training
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    model.eval()
    model.requires_grad_(False)
    print("✅ Predictor loaded successfully.")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='info', choices=['debug', 'info', 'warning', 'error', 'critical'])
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))    

    encoder = VJEPA2Wrapper(num_frames=16)
    predictor = VJ2GUIPredictor(num_frames=16).to(encoder.device)

    # video_tensor = encoder.from_video("videos/test1.mp4", fps=4)

    from torchvision import transforms
    from device_control.screen_capture import capture_screen
    NUM_CONTEXT_FRAMES = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])


    buffer = []
    while len(buffer) < NUM_CONTEXT_FRAMES:
        frame = capture_screen()
        tensor = preprocess(frame).to(DEVICE)
        buffer.append(tensor)
    video_tensor = torch.stack(buffer, dim=0).unsqueeze(0)
    print(f"{video_tensor.shape=}")

    z_all = encoder(video_tensor)  # [1, 16, 256, 1024]
    B, T, N, D = z_all.shape # T = 8 (reduced from 16 to 8 by tubulet = 2)
    action_dim = ACTIONS_PER_BATCH * ACTION_DIM
    actions = torch.randn(B, T, action_dim).to(z_all.device)

    print(f"Visual tokens shape      (z): {z_all.shape}")
    print(f"Action tensor shape      (a): {actions.shape}")

    z_pred = predictor(z_all, actions)
    print(f"Predicted tokens shape: {z_pred.shape}")
