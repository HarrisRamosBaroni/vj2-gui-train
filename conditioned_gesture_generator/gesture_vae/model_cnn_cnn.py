import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import CNNEncoder
from .decoder import CNNDecoder


class CNNCNNVAE(nn.Module):
    """
    VAE model combining CNN encoder and CNN decoder.
    Implements reparameterization trick for differentiable sampling.
    """
    
    def __init__(self, d_latent=128, k_classes=3000, encoder_hidden_dim=256, decoder_hidden_dim=256,
                 decoder_channels=None, conv_kernel=3, conv_stride=2, use_output_padding=False):
        """
        Args:
            d_latent: Dimension of the latent space
            k_classes: Number of quantization classes for coordinate prediction
            encoder_hidden_dim: Hidden dimension for encoder
            decoder_hidden_dim: Hidden dimension for decoder
            decoder_channels: Custom channel progression for decoder
            conv_kernel: Kernel size for ConvTranspose1d layers
            conv_stride: Stride for ConvTranspose1d layers
            use_output_padding: Whether to use output_padding in ConvTranspose1d
        """
        super().__init__()
        self.d_latent = d_latent
        self.k_classes = k_classes
        
        # Initialize encoder and decoder
        self.encoder = CNNEncoder(d_latent=d_latent, hidden_dim=encoder_hidden_dim)
        self.decoder = CNNDecoder(
            d_latent=d_latent, 
            k_classes=k_classes, 
            hidden_dim=decoder_hidden_dim,
            decoder_channels=decoder_channels,
            conv_kernel=conv_kernel,
            conv_stride=conv_stride,
            use_output_padding=use_output_padding
        )
        
    def reparameterize(self, mu, log_sigma):
        """
        Reparameterization trick for differentiable sampling.
        
        Args:
            mu: Mean of latent distribution [B, d_latent]
            log_sigma: Log standard deviation of latent distribution [B, d_latent]
            
        Returns:
            z: Sampled latent vector [B, d_latent]
        """
        if self.training:
            # Sample from N(0, 1)
            eps = torch.randn_like(mu)
            # Reparameterize: z = mu + sigma * eps
            sigma = torch.exp(log_sigma)
            z = mu + sigma * eps
        else:
            # During inference, use mean
            z = mu
        return z
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Args:
            x: Input gesture sequences [B, 250, 2]
            
        Returns:
            dict: Dictionary containing:
                - logits: Classification logits [B, 250, 2, k_classes]
                - mu: Latent mean [B, d_latent]
                - log_sigma: Latent log std [B, d_latent]
                - z: Sampled latent vector [B, d_latent]
        """
        # Encode
        mu, log_sigma = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, log_sigma)
        
        # Decode
        logits = self.decoder(z)
        
        return {
            'logits': logits,
            'mu': mu,
            'log_sigma': log_sigma,
            'z': z
        }
    
    def encode(self, x):
        """
        Encode input to latent parameters.
        
        Args:
            x: Input gesture sequences [B, 250, 2]
            
        Returns:
            tuple: (mu, log_sigma)
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode latent vector to classification logits.
        
        Args:
            z: Latent vector [B, d_latent]
            
        Returns:
            logits: Classification logits [B, 250, 2, k_classes]
        """
        return self.decoder(z)
    
    def sample(self, num_samples=1, device='cpu'):
        """
        Generate samples from the model by sampling from prior.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            logits: Generated classification logits [num_samples, 250, 2, k_classes]
        """
        self.eval()
        with torch.no_grad():
            # Sample from prior N(0, I)
            z = torch.randn(num_samples, self.d_latent, device=device)
            
            # Decode
            logits = self.decoder(z)
            
        return logits
    
    def reconstruct(self, x):
        """
        Reconstruct input sequences.
        
        Args:
            x: Input gesture sequences [B, 250, 2]
            
        Returns:
            logits: Reconstructed classification logits [B, 250, 2, k_classes]
        """
        self.eval()
        with torch.no_grad():
            # Encode to mean (no sampling)
            mu, _ = self.encoder(x)
            
            # Decode
            logits = self.decoder(mu)
            
        return logits
    
    def get_latent_representation(self, x):
        """
        Get latent representation of input.
        
        Args:
            x: Input gesture sequences [B, 250, 2]
            
        Returns:
            z: Latent representation [B, d_latent]
        """
        self.eval()
        with torch.no_grad():
            mu, log_sigma = self.encoder(x)
            z = self.reparameterize(mu, log_sigma)
        return z


class LightweightCNNCNNVAE(nn.Module):
    """
    Lightweight version of CNN-CNN VAE for faster training.
    """
    
    def __init__(self, d_latent=128, k_classes=3000, hidden_dim=128):
        super().__init__()
        from .encoder import LightweightCNNEncoder
        from .decoder import LightweightCNNDecoder
        
        self.d_latent = d_latent
        self.k_classes = k_classes
        
        self.encoder = LightweightCNNEncoder(d_latent=d_latent, hidden_dim=hidden_dim)
        self.decoder = LightweightCNNDecoder(d_latent=d_latent, k_classes=k_classes, hidden_dim=hidden_dim)
    
    def reparameterize(self, mu, log_sigma):
        if self.training:
            eps = torch.randn_like(mu)
            sigma = torch.exp(log_sigma)
            z = mu + sigma * eps
        else:
            z = mu
        return z
    
    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        z = self.reparameterize(mu, log_sigma)
        logits = self.decoder(z)
        
        return {
            'logits': logits,
            'mu': mu,
            'log_sigma': log_sigma,
            'z': z
        }