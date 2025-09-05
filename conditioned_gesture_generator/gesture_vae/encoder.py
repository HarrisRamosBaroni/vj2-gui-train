
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    CNN-based encoder matching the CNN gesture classifier architecture.
    Takes input of shape [B, 250, 2] and outputs mu and sigma for VAE latent space.
    Uses dilated 1D convolutions preserving sequence length.
    """
    
    def __init__(self, d_latent=128, hidden_dim=512, dropout=0.1):
        """
        Args:
            d_latent: Dimension of the latent space
            hidden_dim: Hidden dimension for intermediate layers (matches decoder_channels[0])
            dropout: Dropout probability
        """
        super().__init__()
        self.d_latent = d_latent
        self.hidden_dim = hidden_dim
        
        # CNN Encoder layers with dilated convolutions (stride=1, preserving length)
        # Matches CNN gesture classifier exactly
        encoder_channels = [64, 128, 256, 512]
        kernel_size = 5
        
        self.encoder_layers = nn.ModuleList()
        in_channels = 2  # input_dim (x, y coordinates)
        
        for i, out_channels in enumerate(encoder_channels):
            dilation = 2 ** i  # Increasing dilation: 1, 2, 4, 8
            padding = (kernel_size - 1) * dilation // 2  # Maintain same output length
            self.encoder_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, 
                         padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels
        
        # Calculate flattened dimension (sequence length preserved)
        self.encoded_length = 250  # stride=1 preserves length
        self.flattened_dim = encoder_channels[-1] * self.encoded_length  # 512 * 250 = 128,000
        
        # MLP after CNN encoding - matches CNN gesture classifier
        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.flattened_dim, hidden_dim),  # 128,000 → 512
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_latent)  # 512 → 128 (common representation)
        )
        
        # Layer normalization for latent vector
        self.latent_norm = nn.LayerNorm(d_latent)
        
        # Separate heads for mu and log_sigma (from normalized latent)
        self.fc_mu = nn.Linear(d_latent, d_latent)
        self.fc_log_sigma = nn.Linear(d_latent, d_latent)
        
    def forward(self, x):
        """
        Forward pass through the encoder matching CNN gesture classifier.
        
        Args:
            x: Input tensor of shape [B, 250, 2]
            
        Returns:
            mu: Mean of latent distribution [B, d_latent]
            log_sigma: Log standard deviation of latent distribution [B, d_latent]
        """
        B, T, C = x.shape
        
        # Transpose for Conv1d: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)  # [B, 2, 250]
        
        # Encode through CNN layers (dilated convolutions)
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Flatten: [B, 512, 250] -> [B, 128000]
        x = x.view(B, -1)
        
        # Pass through MLP to get common latent representation
        latent = self.encoder_mlp(x)  # [B, d_latent]
        
        # Apply layer normalization
        latent = self.latent_norm(latent)  # [B, d_latent]
        
        # Split into mu and log_sigma from normalized latent
        mu = self.fc_mu(latent)
        log_sigma = self.fc_log_sigma(latent)
        
        return mu, log_sigma
    
    def encode(self, x):
        """
        Encode input to latent parameters.
        Alias for forward pass.
        """
        return self.forward(x)


class LightweightCNNEncoder(nn.Module):
    """
    Lightweight CNN encoder with fewer parameters.
    Alternative encoder for faster training or smaller models.
    """
    
    def __init__(self, d_latent=128, hidden_dim=128):
        super().__init__()
        self.d_latent = d_latent
        self.hidden_dim = hidden_dim
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=3, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=7, stride=3, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )
        
        # 250 -> 84 -> 28 -> 14
        self.conv_output_size = 64 * 14
        
        self.fc_mu = nn.Linear(self.conv_output_size, d_latent)
        self.fc_log_sigma = nn.Linear(self.conv_output_size, d_latent)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.transpose(1, 2)  # [B, 2, 250]
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)
        
        mu = self.fc_mu(x)
        log_sigma = self.fc_log_sigma(x)
        
        return mu, log_sigma
