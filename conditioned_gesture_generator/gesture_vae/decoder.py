
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        # CNN Decoder layers (ConvTranspose1d with configurable kernel/stride)
        self.decoder_layers = nn.ModuleList()
        in_channels = decoder_channels[0]
        
        for i, out_channels in enumerate(decoder_channels[1:]):
            # Use kernel=4, stride=2, padding=1 for clean doubling without checkerboard artifacts
            # This ensures even kernel coverage: each output position gets exactly 2 kernel hits
            padding = 1
            output_padding = 0  # Not needed with proper kernel/stride/padding combo
            
            self.decoder_layers.append(nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels, out_channels, 
                    kernel_size=conv_kernel, 
                    stride=conv_stride, 
                    padding=padding,
                    output_padding=output_padding
                ),
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
            # 32 → 64 (clean doubling with kernel=4, stride=2, padding=1)
            nn.ConvTranspose1d(hidden_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 64 → 128
            nn.ConvTranspose1d(hidden_dim//2, hidden_dim//4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 128 → 256
            nn.ConvTranspose1d(hidden_dim//4, hidden_dim//8, kernel_size=4, stride=2, padding=1),
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
