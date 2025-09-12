
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride, padding,
                output_padding=0  # assuming clean doubling, as in your design
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        # For the shortcut, we need to handle both channel and spatial dimension changes
        # Use interpolation for spatial upsampling to ensure exact size match
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.Upsample(scale_factor=stride, mode='linear', align_corners=False)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


def create_latent_expansion(d_latent, hidden_dim, output_size, dropout=0.1):
    """
    Create non-linear latent expansion module.
    
    Architecture: Linear(d_latent → 2*hidden) → GELU → Linear(2*hidden → output_size)
    
    Args:
        d_latent: Input latent dimension
        hidden_dim: Hidden dimension for intermediate expansion  
        output_size: Final output size
        dropout: Dropout probability
        
    Returns:
        nn.Sequential module for latent expansion
    """
    return nn.Sequential(
        nn.Linear(d_latent, 2 * hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(2 * hidden_dim, output_size)
    )


class SeededCNNDecoder(nn.Module):
    """
    Seeded CNN decoder that generates gestures from a tiled latent seed pattern.
    
    The decoder is called "Seeded" because it creates a tiled pattern by repeating
    the latent vector across all timesteps, acting as a seed that is then modulated
    by positional encoding and refined through dilated convolutions.
    
    Takes latent vector [B, d_latent] and outputs classification logits [B, 250, 2, k_classes].
    Uses repeat (tiling) + positional encoding + dilated conv stack with axis-specific heads.
    """
    
    def __init__(self, d_latent=124, k_classes=5000, feature_dim=256, num_layers=8):
        """
        Args:
            d_latent: Dimension of the latent space (default: 124)
            k_classes: Number of quantization classes for coordinate prediction (default: 5000)
            feature_dim: Hidden feature dimension (default: 256)
            num_layers: Number of dilated conv layers (default: 8)
        """
        super().__init__()
        self.d_latent = d_latent
        self.k_classes = k_classes
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.sequence_length = 250
        
        # Linear projection from latent to features
        self.latent_proj = nn.Linear(d_latent, feature_dim)
        
        # Learned positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, feature_dim, self.sequence_length) * 0.02)
        
        # Dilated conv stack with exponentially increasing dilation
        self.conv_stack = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i  # 1, 2, 4, 8, 16, 32, 64, 128
            self.conv_stack.append(nn.Sequential(
                nn.Conv1d(feature_dim, feature_dim, kernel_size=3, dilation=dilation, padding=dilation),
                nn.GroupNorm(1, feature_dim),
                nn.GELU()
            ))
        
        # Axis embeddings for x and y
        self.axis_embedding = nn.Parameter(torch.randn(2, feature_dim) * 0.02)
        
        # Output heads for x and y predictions
        self.x_head = nn.Conv1d(feature_dim, k_classes, kernel_size=1)
        self.y_head = nn.Conv1d(feature_dim, k_classes, kernel_size=1)
        
    def forward(self, z):
        """
        Forward pass through the decoder.
        
        Args:
            z: Latent vector of shape [B, d_latent]
            
        Returns:
            logits: Classification logits of shape [B, 250, 2, k_classes]
        """
        B = z.shape[0]
        
        # Project latent to features [B, feature_dim]
        features = self.latent_proj(z)
        
        # Repeat along time dimension [B, feature_dim, 250]
        features = features.unsqueeze(-1).expand(-1, -1, self.sequence_length)
        
        # Add positional encoding
        features = features + self.positional_encoding
        
        # Apply dilated conv stack with residual connections
        for conv_layer in self.conv_stack:
            features = features + conv_layer(features)
        
        # Create axis-specific features by adding axis embeddings
        # features: [B, feature_dim, 250]
        # axis_embedding: [2, feature_dim]
        
        # Process x-axis
        x_features = features + self.axis_embedding[0].unsqueeze(-1)  # [B, feature_dim, 250]
        x_logits = self.x_head(x_features)  # [B, k_classes, 250]
        
        # Process y-axis
        y_features = features + self.axis_embedding[1].unsqueeze(-1)  # [B, feature_dim, 250]
        y_logits = self.y_head(y_features)  # [B, k_classes, 250]
        
        # Transpose to [B, 250, k_classes] and stack for [B, 250, 2, k_classes]
        x_logits = x_logits.transpose(1, 2)  # [B, 250, k_classes]
        y_logits = y_logits.transpose(1, 2)  # [B, 250, k_classes]
        
        logits = torch.stack([x_logits, y_logits], dim=2)  # [B, 250, 2, k_classes]
        
        return logits
    
    def decode(self, z):
        """
        Decode latent vector to classification logits.
        Alias for forward pass.
        """
        return self.forward(z)


class CNNDecoder(nn.Module):
    """
    CNN-based decoder with configurable architecture.
    Takes latent vector of shape [B, d_latent] and outputs classification logits [B, 250, 2, k_classes].
    Uses MLP + CNN upscaling with separate x/y classification heads.
    """
    
    def __init__(self, d_latent=128, k_classes=3000, hidden_dim=512, dropout=0.1, 
                 decoder_channels=None, conv_kernel=3, conv_stride=2, use_output_padding=False):
        """
        Args:
            d_latent: Dimension of the latent space
            k_classes: Number of quantization classes for coordinate prediction
            hidden_dim: Hidden dimension for intermediate layers (decoder_channels[0])
            dropout: Dropout probability
            decoder_channels: Custom channel progression (if None, uses slower halving)
            conv_kernel: Kernel size for ConvTranspose1d layers (default: 3)
            conv_stride: Stride for ConvTranspose1d layers (default: 2)
            use_output_padding: Whether to use output_padding in ConvTranspose1d (default: False)
        """
        super().__init__()
        self.d_latent = d_latent
        self.k_classes = k_classes
        self.sequence_length = 250
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.use_output_padding = use_output_padding
        
        # Slower channel progression: [512, 384, 256, 128] instead of [512, 256, 128, 64]
        if decoder_channels is None:
            decoder_channels = [hidden_dim, hidden_dim*3//4, hidden_dim//2, hidden_dim//4]
        self.decoder_channels = decoder_channels
        
        # Calculate initial length for decoder CNN - start small and upsample
        # Use 32 to get clean doubling: 32 → 64 → 128 → 256, then crop to 250
        num_decoder_layers = len(decoder_channels) - 1  # 3 transpose conv layers
        self.decoder_initial_length = 32
        
        # MLP Decoder start - non-linear expansion: Linear(d_latent → 2*hidden) → GELU → Linear(2*hidden → hidden * L0)
        output_size = decoder_channels[0] * self.decoder_initial_length
        self.latent_to_features = create_latent_expansion(d_latent, hidden_dim, output_size, dropout)
        
        # CNN Decoder layers using Upsample + Conv1d for stable upsampling
        self.decoder_layers = nn.ModuleList()
        in_channels = decoder_channels[0]
        
        for i, out_channels in enumerate(decoder_channels[1:]):
            self.decoder_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels
        
        # Final refinement layer (no upsampling, just feature processing)
        self.final_conv = nn.Conv1d(
            decoder_channels[-1], decoder_channels[-1], 
            kernel_size=3, stride=1, padding=1
        )
        
        # Simple classification heads - direct projection to classes
        final_channels = decoder_channels[-1]
        
        # Simple 1x1 convolution for classification
        self.x_classifier = nn.Conv1d(final_channels, k_classes, kernel_size=1)
        self.y_classifier = nn.Conv1d(final_channels, k_classes, kernel_size=1)
        
    def forward(self, z):
        """
        Forward pass through the decoder matching CNN gesture classifier exactly.
        
        Args:
            z: Latent vector of shape [B, d_latent]
            
        Returns:
            logits: Classification logits of shape [B, 250, 2, k_classes]
        """
        B = z.shape[0]
        
        # Project latent to feature map
        features = self.latent_to_features(z)  # [B, channels * initial_length]
        features = features.view(B, self.decoder_channels[0], self.decoder_initial_length)  # [B, C, L]
        
        # Upsample through transposed CNN layers
        for layer in self.decoder_layers:
            features = layer(features)
        
        # Final refinement
        features = self.final_conv(features)  # [B, final_channels, current_length]
        
        # Ensure exactly 250 timesteps (handle any length mismatch)
        if features.shape[2] != self.sequence_length:
            if features.shape[2] > self.sequence_length:
                # Crop from center if too long
                start_idx = (features.shape[2] - self.sequence_length) // 2
                features = features[:, :, start_idx:start_idx + self.sequence_length]
            else:
                # Pad if too short
                pad_amount = self.sequence_length - features.shape[2]
                features = F.pad(features, (0, pad_amount), mode='replicate')
        
        # Apply classification heads
        x_logits = self.x_classifier(features)  # [B, num_classes, T]
        y_logits = self.y_classifier(features)  # [B, num_classes, T]
        
        # Transpose to [B, T, num_classes] and stack for [B, T, 2, num_classes]
        x_logits = x_logits.transpose(1, 2)  # [B, T, num_classes]
        y_logits = y_logits.transpose(1, 2)  # [B, T, num_classes]
        
        logits = torch.stack([x_logits, y_logits], dim=2)  # [B, T, 2, num_classes]
        
        return logits
    
    def decode(self, z):
        """
        Decode latent vector to classification logits.
        Alias for forward pass.
        """
        return self.forward(z)
    
    def _init_weights(self):
        """Initialize weights following best practices."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


class LightweightCNNDecoder(nn.Module):
    """
    Lightweight CNN decoder with fewer parameters.
    Alternative decoder for faster training or smaller models.
    Outputs separate x/y coordinate predictions like the main decoder.
    """
    
    def __init__(self, d_latent=128, k_classes=3000, hidden_dim=128):
        super().__init__()
        self.d_latent = d_latent
        self.k_classes = k_classes
        self.hidden_dim = hidden_dim
        self.sequence_length = 250
        
        # Use clean doubling: 32 → 64 → 128 → 256, then crop to 250
        self.initial_length = 32
        self.initial_channels = hidden_dim
        
        self.fc_expand = nn.Sequential(
            nn.Linear(d_latent, self.initial_channels * self.initial_length),
            nn.ReLU(inplace=True)
        )
        
        self.deconv_layers = nn.Sequential(
            # 32 → 64 (using Upsample + Conv1d for stable upsampling)
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 64 → 128
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(hidden_dim//2, hidden_dim//4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 128 → 256
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(hidden_dim//4, hidden_dim//8, kernel_size=3, padding=1),
        )
        
        # Separate classification heads for x and y
        self.x_classifier = nn.Conv1d(hidden_dim//8, k_classes, kernel_size=1)
        self.y_classifier = nn.Conv1d(hidden_dim//8, k_classes, kernel_size=1)
        
    def forward(self, z):
        B = z.shape[0]
        
        features = self.fc_expand(z)
        features = features.view(B, self.initial_channels, self.initial_length)
        features = self.deconv_layers(features)
        
        # Crop from 256 to 250 to avoid resampling artifacts
        if features.shape[-1] > self.sequence_length:
            start_idx = (features.shape[-1] - self.sequence_length) // 2
            features = features[:, :, start_idx:start_idx + self.sequence_length]
        elif features.shape[-1] < self.sequence_length:
            pad_amount = self.sequence_length - features.shape[-1]
            features = F.pad(features, (0, pad_amount), mode='replicate')
        
        # Apply classification heads
        x_logits = self.x_classifier(features)  # [B, num_classes, T]
        y_logits = self.y_classifier(features)  # [B, num_classes, T]
        
        # Transpose to [B, T, num_classes] and stack for [B, T, 2, num_classes]
        x_logits = x_logits.transpose(1, 2)  # [B, T, num_classes]
        y_logits = y_logits.transpose(1, 2)  # [B, T, num_classes]
        
        logits = torch.stack([x_logits, y_logits], dim=2)  # [B, T, 2, num_classes]
        
        return logits
