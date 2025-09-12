import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import logging
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import torch.nn.functional as F

from gui_world_model.predictor import VJ2GUIPredictor
from src.utils.tensors import trunc_normal_

class VJ2GUIPredictorProb(VJ2GUIPredictor):
    """
    Probabilistic predictor that outputs (mu, b) per latent dimension,
    where b is constrained to be positive via softplus.
    This version inherits the standard action-token concatenation mechanism.
    """

    def __init__(self, embed_dim=1024, predictor_embed_dim=1024, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, predictor_embed_dim=predictor_embed_dim, *args, **kwargs)

        # Overwrite the final projection layer to output mu and raw_b
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim * 2)
        trunc_normal_(self.predictor_proj.weight, std=self.init_std)
        if self.predictor_proj.bias is not None:
            nn.init.constant_(self.predictor_proj.bias, 0.0)

    def forward(self, z, actions):
        """
        Processes visual and action tokens to predict a distribution over the next visual state.
        - z:       State tensor [B, T, N, D]
        - actions: Action tensor [B, T, A]

        Returns:
        - mu: Mean of the predicted distribution [B, T, N, D]
        - b: Scale parameter of the Laplace distribution [B, T, N, D]
        """
        B, T, N, D = z.shape
        H, W = self.grid_height, self.grid_width
        
        # Embed visual and action tokens
        z_embed = self.predictor_embed(z)
        a_embed = self.action_encoder(actions).unsqueeze(2)

        # Concatenate action and visual tokens
        x = torch.cat([a_embed, z_embed], dim=2).flatten(1, 2)

        # Apply transformer blocks
        attn_mask = self.attn_mask[:x.size(1), :x.size(1)].to(z.device, non_blocking=True)
        for blk in self.predictor_blocks:
            x = blk(x, mask=None, attn_mask=attn_mask, T=T, H=H, W=W, action_tokens=1)

        # Reshape and strip action token
        x = x.view(B, T, 1 + H * W, -1)
        x = x[:, :, 1:, :]

        # Final normalization and projection
        x = self.predictor_norm(x)
        delta = self.predictor_proj(x)  # Shape: [B, T, N, 2*D]

        # Split into mu and raw_b for the distribution
        delta_mu = delta[..., :D]
        delta_raw_b = delta[..., D:]
        
        # The final prediction is the input `z` plus the predicted residual `delta_mu`
        mu = z + delta_mu
        b = F.softplus(delta_raw_b) + 1e-6  # b is the scale of the residual

        return mu, b