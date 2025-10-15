import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .losses import QuantizingLoss

class CNNQuantizingDecoderLightweight(nn.Module):
    """
    A CNN-based decoder that predicts quantized (x, y) coordinates.
    This model is adapted from the legacy `LightweightCNNDecoder`. It takes a
    latent vector 'z' and outputs classification logits for the (x, y) dimensions.
    The touch state is inferred downstream: non-zero coordinates imply touch.
    """

    def __init__(self, z_dim: int, T_steps: int, k_classes: int, hidden_dim: int):
        super().__init__()
        self.z_dim = z_dim
        self.T_steps = T_steps
        self.k_classes = k_classes
        self.hidden_dim = hidden_dim

        # Decoder starts with a dense layer and reshapes into a small sequence
        self.initial_length = 32  # Start with a small sequence and upsample
        self.initial_channels = hidden_dim

        self.fc_expand = nn.Sequential(
            nn.Linear(z_dim, self.initial_channels * self.initial_length),
            nn.ReLU(inplace=True)
        )

        # Upsampling layers using stable Upsample + Conv1d
        self.deconv_layers = nn.Sequential(
            # 32 -> 64
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 64 -> 128
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 128 -> 256
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 8, kernel_size=3, padding=1),
        )

        # Separate 1x1 conv classification heads for x and y
        self.x_classifier = nn.Conv1d(hidden_dim // 8, k_classes, kernel_size=1)
        self.y_classifier = nn.Conv1d(hidden_dim // 8, k_classes, kernel_size=1)

    def get_loss_function(self) -> nn.Module:
        """Returns the appropriate loss function for this model."""
        return QuantizingLoss(k_classes=self.k_classes)

    def get_config(self) -> Dict[str, Any]:
        """Serializes model configuration for checkpointing."""
        return {
            "z_dim": self.z_dim,
            "T_steps": self.T_steps,
            "k_classes": self.k_classes,
            "hidden_dim": self.hidden_dim,
        }

    def _dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Converts class indices back to coordinate values in [-1, 1]."""
        return (indices.float() / (self.k_classes - 1))  #  * 2.0 - 1.0

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = z.shape[0]

        features = self.fc_expand(z)
        features = features.view(B, self.initial_channels, self.initial_length)
        features = self.deconv_layers(features) # Shape: (B, hidden/8, 256)

        # Crop from 256 to the target sequence length T_steps
        if features.shape[-1] > self.T_steps:
            start_idx = (features.shape[-1] - self.T_steps) // 2
            features = features[:, :, start_idx:start_idx + self.T_steps]
        elif features.shape[-1] < self.T_steps:
            pad_amount = self.T_steps - features.shape[-1]
            features = F.pad(features, (0, pad_amount), mode='replicate')

        # Apply classification heads
        x_logits = self.x_classifier(features)  # Shape: (B, k_classes, T)
        y_logits = self.y_classifier(features)  # Shape: (B, k_classes, T)

        # Transpose and stack to get final logits shape: (B, T, 2, k_classes)
        x_logits = x_logits.transpose(1, 2)
        y_logits = y_logits.transpose(1, 2)
        xy_logits = torch.stack([x_logits, y_logits], dim=2)

        # --- Create final trajectory for visualization ---
        with torch.no_grad():
            # Get the predicted class index for each coordinate
            pred_indices = torch.argmax(xy_logits, dim=-1) # Shape: (B, T, 2)
            
            # Dequantize to get coordinate values
            xy_coords = self._dequantize(pred_indices) # Shape: (B, T, 2)

            # Infer touch state: touch is active if either x or y is not near zero.
            # We check if the class index is not 0 (assuming class 0 maps to coord 0).
            touch_gate = (pred_indices != 0).any(dim=-1, keepdim=True).float() # Shape: (B, T, 1)

            # Concatenate to form the final action
            action_trajectory = torch.cat([xy_coords, touch_gate], dim=-1)

        return {
            "xy_logits": xy_logits,
            "action_trajectory": action_trajectory,
        }