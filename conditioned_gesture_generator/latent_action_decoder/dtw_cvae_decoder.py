
import torch
import torch.nn as nn
from typing import Dict, Any

from ..losses.dtw_cvae_loss import DTWCVAELoss

class StyleEncoder(nn.Module):
    """
    Encodes a ground truth gesture into a latent style vector and predicts
    the number of touch transitions.
    """
    def __init__(self, input_dim, hidden_dim, style_dim, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          bidirectional=True, batch_first=True)
        
        # Output layers for style distribution
        self.fc_mu = nn.Linear(hidden_dim * 2, style_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, style_dim)
        
        # Auxiliary output for transition count prediction
        self.fc_aux = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        _, h_n = self.rnn(x)
        # Concatenate final hidden states from both directions
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        aux_trans_count = self.fc_aux(h_n)
        
        return mu, logvar, aux_trans_count

class GestureGenerator(nn.Module):
    """
    Generates a gesture trajectory from latent action, state, and style vectors.
    Uses a Transformer-based architecture to produce control points.
    """
    def __init__(self, z_dim, s_dim, style_dim, T_steps, K_ctrl_pts=32, nhead=8, num_decoder_layers=3, num_s_tokens=256):
        super().__init__()
        self.T_steps = T_steps
        self.K_ctrl_pts = K_ctrl_pts
        
        # --- Conditioning Context --- #
        self.z_proj = nn.Linear(z_dim, 512)
        self.s_proj = nn.Linear(s_dim, 512)
        self.s_pos_embed = nn.Parameter(torch.randn(1, num_s_tokens, 512))

        # --- Control Point Generation --- #
        self.style_proj = nn.Linear(style_dim, 512)
        self.ctrl_pt_pos_emb = nn.Parameter(torch.randn(1, K_ctrl_pts, 512))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.final_norm = nn.LayerNorm(512)
        self.output_head = nn.Linear(512, 3) # x, y, touch

        # Initialize output_head bias to prevent model collapse to (0,0,0).
        # We initialize x,y logits to 0.5 (center of screen) and the touch logit
        # to a large negative number to default to 'touch off'.
        with torch.no_grad():
            self.output_head.bias.data.copy_(torch.tensor([0.5, 0.5, -4.0]))

    def forward(self, z, s, z_style):
        # Create the context memory for the cross-attention
        z_token = self.z_proj(z).unsqueeze(1)
        s_tokens = self.s_proj(s) + self.s_pos_embed
        context = torch.cat([z_token, s_tokens], dim=1)
        
        # Initialize control points from style
        ctrl_pts = self.style_proj(z_style).unsqueeze(1).repeat(1, self.K_ctrl_pts, 1)
        ctrl_pts += self.ctrl_pt_pos_emb
        
        # Transformer decoding
        decoded_ctrl_pts = self.transformer_decoder(ctrl_pts, memory=context)
        
        # --- DEBUG PRINTS ---
        if torch.is_tensor(decoded_ctrl_pts):
            print(f"\n--- GESTURE GENERATOR DEBUG ---")
            print(f"  decoded_ctrl_pts | min: {decoded_ctrl_pts.min():.4f}, max: {decoded_ctrl_pts.max():.4f}, mean: {decoded_ctrl_pts.mean():.4f}")
        # --- END DEBUG ---
        # Apply layernorm
        norm_decoded_ctrl_pts = self.final_norm(decoded_ctrl_pts) 
        # Project to final control points
        # output_ctrl_pts = self.output_head(decoded_ctrl_pts)
        output_ctrl_pts = self.output_head(norm_decoded_ctrl_pts)
        
        # --- DEBUG PRINTS ---
        if torch.is_tensor(output_ctrl_pts):
            print(f"  output_ctrl_pts (logits) | min: {output_ctrl_pts.min():.4f}, max: {output_ctrl_pts.max():.4f}, mean: {output_ctrl_pts.mean():.4f}")
            print(f"--- END GESTURE GENERATOR DEBUG ---\n")
        # --- END DEBUG ---
        
        # Clamp xy and apply sigmoid to touch
        clamped_coords = torch.clamp(output_ctrl_pts[..., :2], 0.0, 1.0)
        touch_sigmoid = torch.sigmoid(output_ctrl_pts[..., 2]).unsqueeze(-1)
        output_ctrl_pts = torch.cat([clamped_coords, touch_sigmoid], dim=-1)

        # --- Interpolation Layer --- #
        return self.interpolate_to_trajectory(output_ctrl_pts)

    def interpolate_to_trajectory(self, ctrl_pts):
        B, K, D = ctrl_pts.shape
        T = self.T_steps
        device = ctrl_pts.device

        # Create a tensor of timesteps from 0.0 to 1.0
        t_points = torch.linspace(0, 1, T, device=device)
        
        # Scale t to be in range [0, K-1]
        t = t_points * (K - 1)
        
        # Find the two control points to interpolate between
        idx0 = torch.floor(t).long().clamp(0, K - 2) # Shape (T,)
        idx1 = idx0 + 1 # Shape (T,)
        
        # Get the fractional part for interpolation
        w1 = (t - idx0.to(t.dtype)).unsqueeze(-1) # Shape (T, 1)
        w0 = 1.0 - w1
        
        # Gather control points
        p0 = ctrl_pts[:, idx0, :] # Shape (B, T, D)
        p1 = ctrl_pts[:, idx1, :] # Shape (B, T, D)
        
        # --- Linear interpolation for x, y (dims 0, 1) ---
        interp_coords = w0 * p0[..., :2] + w1 * p1[..., :2]
        
        # --- Zero-order hold for touch (dim 2) ---
        # Use the value of the first (earlier) control point
        touch_hold = p0[..., 2].unsqueeze(-1)

        return torch.cat([interp_coords, touch_hold], dim=-1)

class DTWCVAEDecoder(nn.Module):
    """
    Main model class for the DTW-CVAE Gesture Decoder.
    Follows the contract from doc_latent_action_decoder_system.md
    """
    def __init__(self, z_dim: int, T_steps: int, s_dim: int = 1024, style_dim: int = 64, loss_cfg: Dict[str, Any] = {}, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.T_steps = T_steps
        self.s_dim = s_dim
        self.style_dim = style_dim
        self.loss_cfg = loss_cfg
        self.kwargs = kwargs

        self.style_encoder = StyleEncoder(input_dim=3, hidden_dim=256, style_dim=style_dim)
        self.generator = GestureGenerator(z_dim, s_dim, style_dim, T_steps, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        return {
            "z_dim": self.z_dim,
            "T_steps": self.T_steps,
            "s_dim": self.s_dim,
            "style_dim": self.style_dim,
            "loss_cfg": self.loss_cfg,
            **self.kwargs
        }

    def get_loss_function(self):
        return DTWCVAELoss(**self.loss_cfg, use_cuda=torch.cuda.is_available())

    @property
    def forward_requires_action(self) -> bool:
        return True

    @property
    def forward_requires_state(self) -> bool:
        return True

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # Get ground truth action `A` and state `s` from kwargs
        A = kwargs['A']
        s = kwargs['s']

        # 1. Encode style from ground truth gesture
        style_mu, style_logvar, pred_trans_count = self.style_encoder(A)

        # 2. Sample style vector using reparameterization trick
        z_style = self.reparameterize(style_mu, style_logvar)

        # 3. Generate trajectory
        action_trajectory = self.generator(z, s, z_style)

        return {
            "action_trajectory": action_trajectory,
            "style_mu": style_mu,
            "style_logvar": style_logvar,
            "predicted_transition_count": pred_trans_count,
        }

    @torch.no_grad()
    def sample(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        s = kwargs['s']
        B = z.shape[0]
        device = z.device

        # Sample style from the prior N(0, I)
        z_style = torch.randn(B, self.style_dim, device=device)

        # Generate trajectory
        action_trajectory = self.generator(z, s, z_style)

        return {"action_trajectory": action_trajectory}
