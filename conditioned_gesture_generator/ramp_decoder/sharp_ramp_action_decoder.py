import numpy as np
import torch

from config import ACTIONS_PER_BATCH

class ZtoSSharp:
    """Non-differentiable sharp step 2D curve generator using numpy operations."""
    def __init__(self, T=ACTIONS_PER_BATCH): # ACTIONS_PER_BATCH = 250
        self.T = T
        self.t = np.linspace(0, 1, T)
    
    def __call__(self, z):
        if torch.is_tensor(z):
            z_np = z.cpu().detach().numpy()
            device = z.device
        else:
            z_np = z
            device = torch.device('cpu')
        
        batch_size = z_np.shape[0]
        curves = np.zeros((batch_size, self.T, 3)) # [B, T, 3] for (x, y, pressure)
        
        for b in range(batch_size):
            X0, Y0, R0, X1, Y1, R1 = z_np[b]
            
            if R0 > R1:
                R0, R1 = R1, R0
                X0, Y0, X1, Y1 = X1, Y1, X0, Y0
            
            # Initialize curve with zero pressure
            curve = np.zeros((self.T, 3))
            
            start_idx = max(0, min(self.T - 1, int(R0 * (self.T - 1))))
            end_idx = max(start_idx, min(self.T - 1, int(R1 * (self.T - 1))))
            
            if start_idx < self.T and start_idx <= end_idx:
                # Set pressure to 1 during the gesture
                curve[start_idx:end_idx + 1, 2] = 1.0

                if start_idx == end_idx:
                    # Handle single-point touch
                    curve[start_idx, 0] = X0
                    curve[start_idx, 1] = Y0
                else:
                    # Create linear ramp for x and y
                    ramp_length = end_idx - start_idx + 1
                    x_ramp = np.linspace(X0, X1, ramp_length)
                    y_ramp = np.linspace(Y0, Y1, ramp_length)
                    curve[start_idx:end_idx + 1, 0] = x_ramp
                    curve[start_idx:end_idx + 1, 1] = y_ramp
            
            curves[b] = curve
        
        return torch.tensor(curves, dtype=torch.float32, device=device)
