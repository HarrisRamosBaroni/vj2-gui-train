import torch
import torch.nn as nn
from typing import Dict, Any

class GRUDecoder(nn.Module):
    """
    Decodes a latent vector 'z' into an action trajectory using a GRU.
    The latent vector initializes the GRU's hidden state.
    """
    def __init__(self, z_dim: int, hidden_dim: int, n_layers: int, T_steps: int):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.T_steps = T_steps
        self.output_dim = 3 # (x, y, touch_state)

        # A learnable start token for the sequence
        self.start_token = nn.Parameter(torch.randn(1, 1, self.output_dim))

        # Project latent z to the initial hidden state for the GRU
        self.z_to_h0 = nn.Linear(self.z_dim, self.n_layers * self.hidden_dim)

        # The GRU layer
        self.gru = nn.GRU(
            input_size=self.output_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True
        )

        # Output layer to map hidden state to action dimensions
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def get_config(self) -> Dict[str, Any]:
        """Returns the model's configuration for serialization."""
        return {
            "z_dim": self.z_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
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
        
        # 1. Project z to initial hidden state h0
        # The view reshapes it to (num_layers, batch_size, hidden_size)
        h0 = self.z_to_h0(z).view(self.n_layers, B, self.hidden_dim)
        
        # 2. Prepare the input sequence for the GRU
        # We use a learned start token and feed it for T_steps.
        # The GRU will generate the sequence based on the evolving hidden state.
        gru_input = self.start_token.expand(B, self.T_steps, -1)
        
        # 3. Unroll the GRU
        # gru_output will have shape (B, T_steps, hidden_dim)
        gru_output, _ = self.gru(gru_input, h0)
        
        # 4. Map the hidden states to the output action dimensions
        # output will have shape (B, T_steps, 3)
        trajectory = self.out(gru_output)
        
        return trajectory