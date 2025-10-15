import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .losses import QuantizingLoss

def _create_latent_expansion(d_latent, hidden_dim, output_size, dropout=0.1):
    """
    Faithful re-implementation of the original latent expansion module.
    Architecture: Linear(d_latent → 2*hidden) → GELU → Linear(2*hidden → output_size)
    """
    return nn.Sequential(
        nn.Linear(d_latent, 2 * hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(2 * hidden_dim, output_size)
    )

class CNNQuantizingDecoder(nn.Module):
    """
    A corrected, faithful implementation of the legacy gesture_vae.CNNDecoder.

    This version corrects two main issues:
    1. It is a 1:1 architectural match to the original, including the GELU
       activation, BatchNorm1d layers, and final refinement convolution.
    2. It eliminates the information-losing cropping step by using a final
       ConvTranspose1d layer to reach the exact target sequence length.
    """

    def __init__(self, z_dim: int, T_steps: int, k_classes: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.z_dim = z_dim
        self.T_steps = T_steps
        self.k_classes = k_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Use the same channel progression as the legacy model
        decoder_channels = [hidden_dim, hidden_dim * 3 // 4, hidden_dim // 2, hidden_dim // 4]
        self.decoder_channels = decoder_channels
        
        # Decoder starts from a sequence of length 32
        self.decoder_initial_length = 32
        
        # 1. Initial latent expansion (matches legacy)
        mlp_output_size = decoder_channels[0] * self.decoder_initial_length
        self.latent_to_features = _create_latent_expansion(z_dim, hidden_dim, mlp_output_size, dropout)
        
        # 2. Upsampling layers (matches legacy)
        self.decoder_layers = nn.ModuleList()
        in_channels = decoder_channels[0]
        
        # Upsample: 32 -> 64 -> 128
        for i, out_channels in enumerate(decoder_channels[1:-1]): # Stop before the last channel
            self.decoder_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels
        
        # 3. Final ConvTranspose1d layer to go from 128 to 250 (NO CROPPING)
        # This is the key correction.
        # Kernel size and padding are calculated to produce the target length.
        # L_out = (L_in - 1) * stride - 2*padding + kernel
        # 250 = (128 - 1) * 2 - 2*padding + kernel => 250 = 254 - 2p + k
        # k - 2p = -4. Choose k=4, p=4.
        final_in_channels = decoder_channels[-2]
        final_out_channels = decoder_channels[-1]
        self.final_upsample = nn.ConvTranspose1d(
            final_in_channels, final_out_channels,
            kernel_size=4, stride=2, padding=4
        )
        self.final_bn_relu = nn.Sequential(
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # 4. Final refinement layer (matches legacy)
        self.final_conv = nn.Conv1d(
            final_out_channels, final_out_channels, 
            kernel_size=3, stride=1, padding=1
        )
        
        # 5. Classification heads (matches legacy)
        self.x_classifier = nn.Conv1d(final_out_channels, k_classes, kernel_size=1)
        self.y_classifier = nn.Conv1d(final_out_channels, k_classes, kernel_size=1)

    def get_loss_function(self) -> nn.Module:
        return QuantizingLoss(k_classes=self.k_classes)

    def get_config(self) -> Dict[str, Any]:
        return {
            "z_dim": self.z_dim, "T_steps": self.T_steps, "k_classes": self.k_classes,
            "hidden_dim": self.hidden_dim, "dropout": self.dropout,
        }
    
    def _dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Converts class indices back to coordinate values in [0, 1]."""
        return (indices.float() / (self.k_classes - 1))  #  * 2.0 - 1.0
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = z.shape[0]

        features = self.latent_to_features(z)
        features = features.view(B, self.decoder_channels[0], self.decoder_initial_length)
        
        for layer in self.decoder_layers:
            features = layer(features)

        # Apply final upsample layer to get near the target length
        features = self.final_upsample(features)[:, :, :self.T_steps] # Final size might be 250 or 251, trim just in case.
        features = self.final_bn_relu(features)

        features = self.final_conv(features)

        # Apply classification heads
        x_logits = self.x_classifier(features).transpose(1, 2)
        y_logits = self.y_classifier(features).transpose(1, 2)
        xy_logits = torch.stack([x_logits, y_logits], dim=2)
        
        # --- Create final trajectory for visualization ---
        with torch.no_grad():
            pred_indices = torch.argmax(xy_logits, dim=-1)
            # Dequantize: indices -> [0, 1]
            xy_coords = self._dequantize(pred_indices) # Shape: (B, T, 2)
            # xy_coords = (pred_indices.float() / (self.k_classes - 1))  # * 2.0 - 1.0
            # Infer touch state
            touch_gate = (pred_indices != 0).any(dim=-1, keepdim=True).float()  # TODO: does this define touch value in each 250 of the timesteps or uniformly gives all 250 one value?
            action_trajectory = torch.cat([xy_coords, touch_gate], dim=-1)

        return {
            "xy_logits": xy_logits,
            "action_trajectory": action_trajectory,
        }