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
    Action-Conditioned Predictor for GUI control using V-JEPA 2 architecture.
    Takes a sequence of visual tokens (State) and a corresponding sequence of
    action batches (Action) to predict the next sequence of visual tokens.
  
    Input Tensors:
        - z (State): A tensor of shape [B, T, N, D] representing the visual state.
            - B: Batch size of video clips.
            - T: Number of time steps (typically frames // tubelet_size).
            - N: Number of visual tokens per frame (e.g., 14x14 = 196).
            - D: Embedding dimension of each token (e.g., 1024).
        - actions (Action): A tensor of shape [B, T, A] representing the actions.
            - A: The flattened dimension of one "Action Batch"
                 (ACTIONS_PER_BATCH * ACTION_DIM).

    Output Tensor:
        - Predicted visual tokens of shape [B, T, N, D].
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
        action_embed_dim=ACTIONS_PER_BATCH * ACTION_DIM  # 250*3=750
    ):
        super().__init__()

        self._config = self.capture_init_args(locals())

        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing

        # Token embeddings
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim)

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

        # Build causal attention mask
        self.attn_mask = build_action_block_causal_attention_mask(
            self.num_frames // self.tubelet_size,
            self.grid_height,
            self.grid_width,
            1,                  # ← action_tokens = 1
        )
        logger.debug(f"{self.attn_mask.shape=}")
        print(f"{self.attn_mask.shape=}")  # [2056, 2056] where 2056 = T*(H*W+1) = 8*(16*16+1)

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
        Processes the visual and action tokens to predict the next visual state.
        - z:       State tensor [B, T, N, D]
        - actions: Action tensor [B, T, A]
        """
        B, T, N, D = z.shape
        H, W = self.grid_height, self.grid_width
        if N != H * W:
            raise ValueError(f"Mismatch in spatial token count: {N} != {H}*{W}")

        z = self.predictor_embed(z)                        # [B, T, N, D]
        a = self.action_encoder(actions).unsqueeze(2)      # [B, T, 1, D]

        # z = z.view(B, T, H * W, D)  # TODO: unnecessary? since N == H*W anyway...
        logger.debug(f"z.shape: {z.shape}")
        logger.debug(f"a.shape: {a.shape}")
        x = torch.cat([a, z], dim=2).flatten(1, 2)         # [B, T*(1+N), D]
        logger.debug(f"{x.shape=}")  # [B, T*(H*W+1), D] = [1, 2056, 1024]

        # Should be using mask with shape [:x.size(1), :x.size(1)] = [2056, 2056]
        attn_mask = self.attn_mask[:x.size(1), :x.size(1)].to(z.device, non_blocking=True)  # build_action_block_causal_attention_mask(T, H, W, 1).to(z.device)
        logger.debug(f"{attn_mask.shape=}")  # [2056, 2056] where 2056 = T*(H*W+1) = 8*(16*16+1)
        for blk in self.predictor_blocks:
            x = blk(x, mask=None, attn_mask=attn_mask,
                    T=T, H=H, W=W, action_tokens=1)        # ← 1 token

        x = x.view(B, T, 1 + H * W, D)
        x = x[:, :, 1:, :].reshape(B, T, H * W, D)         # strip action token for prediction

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        
        return x

    def get_config(self):
        """Returns the model's initialization configuration."""
        return self._config

    def capture_init_args(self, local_vars):
        # Captures constructor arguments for saving/loading.
        # Exclude 'self' and '__class__'
        return {k: v for k, v in local_vars.items() if k not in ['self', '__class__']}


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
