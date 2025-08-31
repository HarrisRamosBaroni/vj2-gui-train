import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import logging
logger = logging.getLogger(__name__)
import math
import torch
import torch.nn as nn

from gui_world_model.utils.modules import ACBlock, ACFilmBlock, build_action_block_causal_attention_mask
from src.utils.tensors import trunc_normal_
from gui_world_model.utils.film import ActionBiGRUEncoder, FiLMHeads, apply_film
from config import ACTION_DIM

class VJ2GUIPredictorFiLM(nn.Module):
    """
    FiLM-Conditioned Predictor for GUI control.
    Action information is injected via FiLM modulation of intermediate transformer activations.
    This version does NOT use a concatenated action token.
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
        super().__init__()
        self._config = self.capture_init_args(locals())

        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing
        self.film_layers_modulated = film_layers_modulated
        self.film_clamp_gamma_alpha = film_clamp_gamma_alpha

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        
        self.action_encoder = ActionBiGRUEncoder(
            action_dim=ACTION_DIM,
            embedding_dim=film_d_a,
            context_dim=film_d_c * 2  # Bidirectional
        )
        
        self.film_heads = FiLMHeads(
            context_dim=film_d_c * 2,
            model_dim=predictor_embed_dim,
            num_modulated_layers=film_layers_modulated
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList()
        num_standard_layers = depth - film_layers_modulated
        
        # Standard blocks
        for i in range(num_standard_layers):
            self.predictor_blocks.append(ACBlock(
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
            ))

        # FiLM-modulated blocks
        for i in range(num_standard_layers, depth):
            self.predictor_blocks.append(ACFilmBlock(
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
            ))

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)

        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        self.attn_mask = build_action_block_causal_attention_mask(
            self.num_frames // self.tubelet_size,
            self.grid_height,
            self.grid_width,
            add_tokens=0, # No action tokens in the sequence
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
        for layer_id, layer in enumerate(self.predictor_blocks, 1):
            rescale(layer.attn.proj.weight.data, layer_id)
            rescale(layer.mlp.fc2.weight.data, layer_id)

    def forward(self, z, actions):
        B, T, N, D = z.shape
        H, W = self.grid_height, self.grid_width
        
        z_embed = self.predictor_embed(z)
        
        # NOTE: Original VJEPA flattens tokens for the transformer forward pass.
        # FiLM params are generated per-timestep, so we need to handle shapes carefully.
        # Action context has shape [B, T, D_ctx]. FiLM heads expand it to [B, T, 1, D_m].
        action_context = self.action_encoder(actions)
        film_params_per_layer = self.film_heads(action_context, self.film_clamp_gamma_alpha)
        
        # The transformer blocks expect [B, SeqLen, Dim]
        x = z_embed.view(B, T * N, -1)

        # attn_mask = self.attn_mask.to(z.device, non_blocking=True)
        attn_mask = self.attn_mask[:x.size(1), :x.size(1)].to(z.device, non_blocking=True)
        
        num_total_layers = len(self.predictor_blocks)
        first_modulated_layer_idx = num_total_layers - self.film_layers_modulated
        
        for i, blk in enumerate(self.predictor_blocks):
            if isinstance(blk, ACFilmBlock):
                # For FiLM blocks, we need to provide the parameters.
                # The FiLM parameters have shape [B, T, 1, D_m].
                # The block's input `x` is [B, T*N, D_m]. We need to align them.
                # We will reshape x inside the block, apply FiLM, then flatten back.
                layer_film_params = film_params_per_layer[i - first_modulated_layer_idx]
                
                # Reshape params to match flattened token sequence
                # Gamma/Beta shape: [B, T, 1, D_m] -> [B, T, N, D_m] -> [B, T*N, D_m]
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
                 # Standard block forward pass
                x = blk(x, mask=None, attn_mask=attn_mask, T=T, H=H, W=W, action_tokens=0)

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        x = x.view(B, T, N, -1)
        
        return x

    def actions_formatter(self, actions):
        """The FiLM predictor requires the original 4D action tensor [B, T, L, 3]."""
        return actions

    def get_config(self):
        return self._config

    def capture_init_args(self, local_vars):
        # Captures constructor arguments for saving/loading.
        # Exclude 'self' and '__class__'
        return {k: v for k, v in local_vars.items() if k not in ['self', '__class__']}