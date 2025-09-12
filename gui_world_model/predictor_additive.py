import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import logging
logger = logging.getLogger(__name__)
import argparse
import math

import torch
import torch.nn as nn
import torchvision.transforms as T

from gui_world_model.utils.modules import ACBlock as Block, build_action_block_causal_attention_mask
from src.utils.tensors import trunc_normal_
from gui_world_model.encoder import VJEPA2Wrapper

from config import (
    ACTION_DIM,
    ACTIONS_PER_BATCH,
    OBSERVATIONS_PER_WINDOW
)

class VJ2GUIPredictorAdditive(nn.Module):
    """
    Action-Conditioned Predictor where actions are treated as additive embeddings.
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
        action_embed_dim=ACTIONS_PER_BATCH * ACTION_DIM
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
            add_tokens=0
        )

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

    def forward(self, z, actions):
        """
        - z:       State tensor [B, T, N, D]
        - actions: Action tensor [B, T, A]
        """
        B, T, N, D = z.shape
        H, W = self.grid_height, self.grid_width
        if N != H * W:
            raise ValueError(f"Mismatch in spatial token count: {N} != {H}*{W}")

        z = self.predictor_embed(z)
        a = self.action_encoder(actions).unsqueeze(2)

        x = z + a
        x = x.flatten(1, 2)

        attn_mask = self.attn_mask[:x.size(1), :x.size(1)].to(z.device, non_blocking=True)
        for blk in self.predictor_blocks:
            x = blk(x, mask=None, attn_mask=attn_mask, T=T, H=H, W=W, action_tokens=0)

        x = x.view(B, T, N, -1)

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        
        return x

    def get_config(self):
        return self._config

    def capture_init_args(self, local_vars):
        return {k: v for k, v in local_vars.items() if k not in ['self', '__class__']}