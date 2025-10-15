import torch
import torch.nn as nn
from typing import Dict, Any
from torchcubicspline import(
    NaturalCubicSpline,
    natural_cubic_spline_coeffs,
    # hermite_cubic_spline_coeffs,
)

def evaluate_spline(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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

class SplineDecoder(nn.Module):
    """
    Decodes a latent vector 'z' into an action trajectory by predicting
    the control points of a spline.
    """
    def __init__(self, z_dim: int, k_points: int, hidden_dim: int, T_steps: int):
        super().__init__()
        self.z_dim = z_dim
        self.k_points = k_points
        self.hidden_dim = hidden_dim
        self.T_steps = T_steps
        self.output_dim = 3 # (x, y, touch_state)

        self.mlp = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.k_points * self.output_dim)
        )

        # Time points to evaluate spline at, created once
        self.register_buffer('t_eval', torch.linspace(0.0, 1.0, self.T_steps))

    def get_config(self) -> Dict[str, Any]:
        """Returns the model's configuration for serialization."""
        return {
            "z_dim": self.z_dim,
            "k_points": self.k_points,
            "hidden_dim": self.hidden_dim,
            "T_steps": self.T_steps,
        }

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to decode the latent vector.
        
        Args:
            z: Latent vector of shape (B, z_dim).
        
        Returns:
            Action trajectory of shape (B, T_steps, 3).
        """
        if z.dim() != 2 or z.shape[1] != self.z_dim:
            raise ValueError(f"Input z must have shape (B, {self.z_dim}), but got {z.shape}")

        B = z.size(0)
        
        # Predict control points from the latent vector
        control_points = self.mlp(z).view(B, self.k_points, self.output_dim)
        
        # Evaluate the spline at the pre-defined time points
        trajectory = evaluate_spline(control_points, self.t_eval)
        
        return trajectory