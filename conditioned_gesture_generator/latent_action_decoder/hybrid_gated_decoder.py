import torch
import torch.nn as nn
from typing import Dict, Any

from .losses import HybridGatedLoss
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

def cubic_spline_interpolator(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Evaluates a natural cubic spline at given time points.
    
    Args:
        control_points: Tensor of shape (B, K, D) where K is num control points
                        and D is the dimension of the action.
        t: Tensor of shape (T,) with time points to evaluate.
        
    Returns:
        Tensor of shape (B, T, D) with the evaluated spline.
    """
    B, K, D = control_points.shape
    
    # Generate time points for control points, assuming they are equidistant
    t_control = torch.linspace(0.0, 1.0, K, device=control_points.device)
    
    # Transpose control_points to (B, D, K) for spline generation
    # then back to (B, K, D) for evaluation
    coeffs = natural_cubic_spline_coeffs(t_control, control_points)
    spline = NaturalCubicSpline(coeffs)
    
    # Evaluate the spline at the desired time points
    evaluated_points = spline.evaluate(t.to(control_points.device))
    return evaluated_points

# A simple spline utility, can be replaced with a more advanced one if needed.
# For now, we'll use linear interpolation for simplicity and speed.
def linear_spline_interpolator(control_points: torch.Tensor, t_points: torch.Tensor) -> torch.Tensor:
    """
    Performs linear interpolation between control points.
    
    Args:
        control_points: Shape (B, K, Dims)
        t_points: Shape (T) -> tensor of timesteps from 0.0 to 1.0
        
    Returns:
        Interpolated points: Shape (B, T, Dims)
    """
    B, K, Dims = control_points.shape
    T = t_points.shape[0]
    
    # Add batch dimension to t_points
    t = t_points.view(1, T).expand(B, -1) # (B, T)
    
    # Scale t to be in range [0, K-1]
    t = t * (K - 1)
    
    # Find the two control points to interpolate between
    idx0 = torch.floor(t).long().clamp(0, K - 2) # (B, T)
    idx1 = idx0 + 1 # (B, T)
    
    # Get the fractional part for interpolation
    w1 = (t - idx0).unsqueeze(-1) # (B, T, 1)
    w0 = 1.0 - w1
    
    # Gather control points
    # Unsqueeze idx to match dims of control_points for gather
    p0 = control_points.gather(1, idx0.unsqueeze(-1).expand(-1, -1, Dims)) # (B, T, Dims)
    p1 = control_points.gather(1, idx1.unsqueeze(-1).expand(-1, -1, Dims)) # (B, T, Dims)
    
    # Linearly interpolate
    interpolated = w0 * p0 + w1 * p1
    
    return interpolated


class HybridGatedDecoder(nn.Module):
    """
    Decodes a latent vector 'z' into a structured action trajectory.
    - Touch State: Treated as a per-timestep binary classification problem.
    - (x, y) Coords: Treated as a continuous regression problem, modeled via splines.
    """
    # def __init__(self, z_dim: int, T_steps: int, k_points: int, hidden_dim: int, loss_config: Dict[str, Any] = None, spline_interpolator: str = "linear"):
    def __init__(self, 
                z_dim: int, 
                T_steps: int, 
                k_points: int, 
                hidden_dim: int, 
                loss_config: Dict[str, Any] = None, 
                spline_interpolator: str = "linear",
                touch_conv_channels: int = 16,
                touch_conv_kernel_size: int = 7):
        super().__init__()
        self.z_dim = z_dim
        self.T_steps = T_steps
        self.k_points = k_points
        self.hidden_dim = hidden_dim
        self.loss_config = loss_config if loss_config is not None else {}
        self.spline_interpolator = None
        if spline_interpolator == "linear":
            self.spline_interpolator = linear_spline_interpolator
        elif spline_interpolator == "cubic":
            self.spline_interpolator = cubic_spline_interpolator
        self.touch_conv_channels = touch_conv_channels
        self.touch_conv_kernel_size = touch_conv_kernel_size

        # --- Shared Body ---
        self.shared_body = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # --- Touch Head (Classifier) ---
        # Predicts T logits directly
        # self.touch_head = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim // 2, T_steps),
        # )

        self.touch_head_input = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, T_steps * self.touch_conv_channels)
        )
        
        conv_padding = (self.touch_conv_kernel_size - 1) // 2

        self.touch_head_convs = nn.Sequential(
            nn.Conv1d(
                in_channels=self.touch_conv_channels,
                out_channels=self.touch_conv_channels,
                kernel_size=self.touch_conv_kernel_size,
                padding=conv_padding,
                padding_mode='reflect'
            ),
            nn.BatchNorm1d(self.touch_conv_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.touch_conv_channels,
                out_channels=self.touch_conv_channels,
                kernel_size=self.touch_conv_kernel_size,
                padding=conv_padding,
                padding_mode='reflect'
            ),
            nn.BatchNorm1d(self.touch_conv_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.touch_conv_channels,
                out_channels=1,
                kernel_size=1
            )
        )

        # --- Coordinate Head (Regressor) ---
        # Predicts K control points for a 2D spline
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, k_points * 2), # k_points for x, k_points for y
        )
            
        # Create a persistent buffer for the timesteps to avoid recreating it
        self.register_buffer('t_points', torch.linspace(0.0, 1.0, T_steps))

    def get_loss_function(self):
        """Returns the appropriate loss function for this model."""
        return HybridGatedLoss(**self.loss_config)

    def get_config(self):
        """Serializes model configuration for checkpointing."""
        return {
            "z_dim": self.z_dim,
            "T_steps": self.T_steps,
            "k_points": self.k_points,
            "hidden_dim": self.hidden_dim,
            "loss_config": self.loss_config,
            "spline_interpolator": self.spline_interpolator,
            "touch_conv_channels": self.touch_conv_channels,
            "touch_conv_kernel_size": self.touch_conv_kernel_size,
        }

    # def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
    #     B = z.size(0)

    #     # 1. Pass through shared body
    #     shared_embedding = self.shared_body(z) # (B, hidden_dim)

    #     # 2. Touch Head -> predicts T logits
    #     touch_logits = self.touch_head(shared_embedding).view(B, self.T_steps, 1) # (B, T, 1)

    #     # 3. Coordinate Head -> predicts spline control points
    #     control_points_flat = self.coord_head(shared_embedding)
    #     control_points = control_points_flat.view(B, self.k_points, 2) # (B, K, 2)
        
    #     # 4. Interpolate spline to get full (x, y) trajectory
    #     xy_pred = self.spline_interpolator(control_points, self.t_points) # (B, T, 2)
    #     xy_pred = torch.clamp(xy_pred, 0.0, 1.0)  # Hard clipping approach
    #     # xy_pred = torch.sigmoid(xy_pred)  # Smooth sigmoid approach
        
    #     # --- Gating for the final trajectory ---
    #     # This part is for inference/visualization, not used in the loss calculation itself
    #     with torch.no_grad():
    #         touch_probs = torch.sigmoid(touch_logits)
    #         touch_gate = (touch_probs > 0.5).float()
            
    #         # Zero out xy coordinates where touch is not active
    #         gated_xy = xy_pred * touch_gate
            
    #         # Concatenate to form the final action
    #         # action_trajectory = torch.cat([gated_xy, touch_gate], dim=-1) # (B, T, 3)
    #         action_trajectory = torch.cat([xy_pred, touch_gate], dim=-1) # (B, T, 3)

    #     return {
    #         "touch_logits": touch_logits,
    #         "xy_pred": xy_pred,
    #         "action_trajectory": action_trajectory,
    #     }
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = z.size(0)

        shared_embedding = self.shared_body(z)

        touch_features = self.touch_head_input(shared_embedding)
        touch_features_seq = touch_features.view(B, self.touch_conv_channels, self.T_steps)
        touch_logits_seq = self.touch_head_convs(touch_features_seq)
        touch_logits = touch_logits_seq.permute(0, 2, 1)

        control_points_flat = self.coord_head(shared_embedding)
        control_points = control_points_flat.view(B, self.k_points, 2)
        
        xy_pred = self.spline_interpolator(control_points, self.t_points)
        
        with torch.no_grad():
            touch_probs = torch.sigmoid(touch_logits)
            touch_gate = (touch_probs > 0.5).float()
            
            action_trajectory = torch.cat([xy_pred, touch_gate], dim=-1)

        return {
            "touch_logits": touch_logits,
            "xy_pred": xy_pred,
            "action_trajectory": action_trajectory,
        }
