import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

from .losses import CVAELoss


# --- Helper Modules ---

class PositionalEncoding(nn.Module):
    """Adds positional information to a sequence of tokens."""
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [B, L, D]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

# --- Core CVAE Components ---

class CVAEEncoder(nn.Module):
    """
    Infers the posterior distribution q(z_style | A, z, s).
    This network takes the ground truth action sequence and conditioning
    variables to produce parameters for the style latent space.
    """
    def __init__(self, action_dim=3, z_dim=128, s_dim=1024, style_dim=32, model_dim=256, nhead=4, num_layers=4, num_s_tokens=256):
        super().__init__()
        self.style_dim = style_dim

        # Input embeddings
        self.action_embed = nn.Linear(action_dim, model_dim)
        self.z_embed = nn.Linear(z_dim, model_dim)
        self.s_proj = nn.Linear(s_dim, model_dim)
        self.s_pos_embed = nn.Parameter(torch.randn(1, num_s_tokens, model_dim)) # Learned positional embedding for s
        self.pos_encoder = PositionalEncoding(model_dim) # Sinusoidal for the action sequence

        # Transformer to process the sequence
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads for distribution parameters
        self.fc_mu = nn.Linear(model_dim, style_dim)
        self.fc_logvar = nn.Linear(model_dim, style_dim)

    def forward(self, A, z, s):
        # A: (B, T, 3), z: (B, 128), s: (B, 256, 1024)
        
        # 1. Embed inputs and create a single sequence for the transformer
        a_tokens = self.pos_encoder(self.action_embed(A))       # (B, 250, model_dim)
        z_token = self.z_embed(z).unsqueeze(1)                   # (B, 1, model_dim)
        s_tokens = self.s_proj(s) + self.s_pos_embed             # (B, 256, model_dim)

        # Concatenate all tokens to form the input sequence
        # The model will learn to associate the action with the context
        full_sequence = torch.cat([z_token, s_tokens, a_tokens], dim=1)

        # 2. Process through Transformer
        encoded_sequence = self.transformer_encoder(full_sequence)

        # 3. Pool the output to a single vector (using the first token's output)
        pooled_output = encoded_sequence[:, 0]

        # 4. Compute posterior parameters
        mu = self.fc_mu(pooled_output)
        logvar = self.fc_logvar(pooled_output)

        return mu, logvar

class CVAEDecoder(nn.Module):
    """
    Generates an action sequence A from p(A | z, s, z_style).
    This network takes the conditioning variables and a sampled style vector
    to produce the full action trajectory.
    """
    def __init__(self, z_dim=128, s_dim=1024, style_dim=32, model_dim=512, nhead=8, num_layers=6, T=250):
        super().__init__()
        
        # 1. Input Conditioning Module
        self.z_embed = nn.Linear(z_dim, model_dim)
        self.style_embed = nn.Linear(style_dim, model_dim)
        self.s_proj = nn.Linear(s_dim, model_dim)
        self.s_pos_embed = nn.Parameter(torch.randn(1, 256, model_dim))

        # 2. Sequence Generation Core
        self.time_queries = nn.Parameter(torch.randn(1, T, model_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 3. Dual Output Heads
        self.touch_head = nn.Linear(model_dim, 1)
        self.xy_head = nn.Sequential(
            nn.Linear(model_dim, 2),
            nn.Sigmoid()
        )

    def forward(self, z, s, z_style):
        # z: (B, 128), s: (B, 256, 1024), z_style: (B, 32)
        
        # Condition inputs
        z_token = self.z_embed(z).unsqueeze(1)
        style_token = self.style_embed(z_style).unsqueeze(1)
        s_tokens = self.s_proj(s) + self.s_pos_embed
        
        # Memory for the transformer decoder
        context_memory = torch.cat([z_token, style_token, s_tokens], dim=1)

        # Expand time queries for the batch
        batch_queries = self.time_queries.expand(z.shape[0], -1, -1)

        # Pass through transformer
        output_sequence = self.transformer_decoder(tgt=batch_queries, memory=context_memory)

        # Get predictions from dual heads
        touch_logits = self.touch_head(output_sequence)
        xy_pred = self.xy_head(output_sequence)

        return {
            "touch_logits": touch_logits,
            "xy_pred": xy_pred
        }

# --- Top-Level CVAE Model ---

class ActionCVAE(nn.Module):
    """
    A wrapper for the full Conditional VAE model, combining the
    encoder and decoder and handling the reparameterization trick.
    """
    def __init__(self, encoder_cfg: dict, decoder_cfg: dict, loss_cfg: dict):
        super().__init__()
        self.encoder = CVAEEncoder(**encoder_cfg)
        self.decoder = CVAEDecoder(**decoder_cfg)
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.loss_cfg = loss_cfg

    def get_config(self) -> Dict[str, Any]:
        """Required: Return model configuration for checkpointing."""
        return {
            "encoder_cfg": self.encoder_cfg,
            "decoder_cfg": self.decoder_cfg,
            "loss_cfg": self.loss_cfg,
        }

    def get_loss_function(self) -> CVAELoss:
        """Required: Return appropriate loss function."""
        return CVAELoss(**self.loss_cfg)

    @property
    def forward_requires_action(self) -> bool:
        """Declares that the forward pass requires the ground truth action `A`."""
        return True

    @property
    def forward_requires_state(self) -> bool:
        """Declares that the forward pass requires the world model state `s`."""
        return True

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for training.
        Accepts `z` and a flexible set of keyword arguments, which must
        include the ground truth action `A` and world model state `s`.
        """
        # A: (B, T, 3), z: (B, 128), s: (B, 256, 1024)
        A = kwargs['A']
        s = kwargs['s']
        
        # 1. Get posterior distribution from the encoder
        mu, logvar = self.encoder(A, z, s)
        
        # 2. Sample z_style using the reparameterization trick
        z_style = self.reparameterize(mu, logvar)
        
        # 3. Decode to reconstruct the action
        recon_A_outputs = self.decoder(z, s, z_style)
        
        # 4. Assemble final action trajectory for visualization/evaluation
        # This must be part of the output dict as per the system contract.
        with torch.no_grad():
            touch_pred = torch.sigmoid(recon_A_outputs["touch_logits"])
            xy_pred = recon_A_outputs["xy_pred"]
            action_trajectory = torch.cat([xy_pred, touch_pred], dim=-1)

        return {
            "recon_A_outputs": recon_A_outputs,
            "mu": mu,
            "logvar": logvar,
            "action_trajectory": action_trajectory, # Required by trainer contract
        }

    @torch.no_grad()
    def sample(self, z, s):
        # This method is for inference/generation
        # z: (B, 128), s: (B, 256, 1024)
        
        # Sample z_style from the prior N(0, I)
        B = z.shape[0]
        device = z.device
        style_dim = self.encoder.style_dim
        z_style = torch.randn(B, style_dim, device=device)
        
        # Generate action from the decoder
        generated_A_outputs = self.decoder(z, s, z_style)
        
        # Assemble final action trajectory
        touch_pred = torch.sigmoid(generated_A_outputs["touch_logits"])
        xy_pred = generated_A_outputs["xy_pred"]
        action_trajectory = torch.cat([xy_pred, touch_pred], dim=-1)
        
        return {
            "recon_A_outputs": generated_A_outputs,
            "action_trajectory": action_trajectory,
        }
