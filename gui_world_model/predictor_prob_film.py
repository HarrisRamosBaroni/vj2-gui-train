import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import logging
logger = logging.getLogger(__name__)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from gui_world_model.utils.modules import ACBlock, ACFilmBlock, build_action_block_causal_attention_mask
from src.utils.tensors import trunc_normal_
from gui_world_model.utils.film import ActionBiGRUEncoder, FiLMHeads, apply_film
from gui_world_model.predictor_film import VJ2GUIPredictorFiLM
from config import ACTION_DIM

class VJ2GUIPredictorProbFiLM(VJ2GUIPredictorFiLM):
    """
    Probabilistic FiLM predictor that outputs (mu, b) per latent dimension.
    b is constrained to be positive via softplus.
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
        # FiLM specific parameters
        film_d_a=32,
        film_d_c=128,
        film_layers_modulated=6,
        film_clamp_gamma_alpha=None,
    ):
        # Reuse parent's initialization to keep behaviour identical.
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_std=init_std,
            use_silu=use_silu,
            wide_silu=wide_silu,
            use_activation_checkpointing=use_activation_checkpointing,
            use_rope=use_rope,
            film_d_a=film_d_a,
            film_d_c=film_d_c,
            film_layers_modulated=film_layers_modulated,
            film_clamp_gamma_alpha=film_clamp_gamma_alpha,
        )

        # Replace projection to output twice the latent dimension (mu and raw_b)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim * 2)
        trunc_normal_(self.predictor_proj.weight, std=self.init_std)
        if self.predictor_proj.bias is not None:
            nn.init.constant_(self.predictor_proj.bias, 0.0)

    def forward(self, z, actions):
        """
        Similar to parent forward but returns (mu, b) where b = softplus(raw_b) + eps.
        """
        B, T, N, D = z.shape
        H, W = self.grid_height, self.grid_width
        
        z_embed = self.predictor_embed(z)
        action_context = self.action_encoder(actions)
        film_params_per_layer = self.film_heads(action_context, self.film_clamp_gamma_alpha)
        
        x = z_embed.view(B, T * N, -1)
        attn_mask = self.attn_mask[:x.size(1), :x.size(1)].to(z.device, non_blocking=True)
        
        num_total_layers = len(self.predictor_blocks)
        first_modulated_layer_idx = num_total_layers - self.film_layers_modulated
        
        for i, blk in enumerate(self.predictor_blocks):
            if isinstance(blk, ACFilmBlock):
                layer_film_params = film_params_per_layer[i - first_modulated_layer_idx]
                
                gamma_att_expanded = layer_film_params['gamma_att'].repeat(1, 1, N, 1).view(B, T * N, -1)
                beta_att_expanded = layer_film_params['beta_att'].repeat(1, 1, N, 1).view(B, T * N, -1)
                gamma_ff_expanded = layer_film_params['gamma_ff'].repeat(1, 1, N, 1).view(B, T * N, -1)
                beta_ff_expanded = layer_film_params['beta_ff'].repeat(1, 1, N, 1).view(B, T * N, -1)

                pass_params = {
                    'gamma_att': gamma_att_expanded, 'beta_att': beta_att_expanded,
                    'gamma_ff': gamma_ff_expanded, 'beta_ff': beta_ff_expanded,
                }
                x = blk(x, film_params=pass_params, mask=None, attn_mask=attn_mask, T=T, H=H, W=W)
            else:
                x = blk(x, mask=None, attn_mask=attn_mask, T=T, H=H, W=W, action_tokens=0)

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)  # Now shape [B, T*N, 2*D]
        x = x.view(B, T, N, -1)     # [B, T, N, 2*D]

        # Split into mu and raw_b
        mu = x[..., :D]
        raw_b = x[..., D:]
        b = F.softplus(raw_b) + 1e-6

        return mu, b

    def actions_formatter(self, actions):
        return actions