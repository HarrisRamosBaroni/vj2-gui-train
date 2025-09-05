import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class CNNGestureClassifier(nn.Module):
    """
    CNN-based gesture classifier for time series data [T=250, 2].
    
    Architecture:
    - CNN Encoder: 1D CNN layers -> flatten -> MLP -> LayerNorm -> latent vector
    - MLP Decoder: latent -> MLP -> separate x/y classification heads [T, 2, 3000]
    """
    
    def __init__(
        self,
        input_dim: int = 2,  # x, y coordinates (ignoring pressure)
        sequence_length: int = 250,
        latent_dim: int = 128,
        num_classes: int = 3000,
        encoder_channels: list = [64, 128, 256, 512],
        decoder_channels: list = [512, 256, 128, 64],
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.kernel_size = kernel_size
        
        # CNN Encoder layers with dilated convolutions (stride=1, increasing dilation)
        self.encoder_layers = nn.ModuleList()
        in_channels = input_dim
        
        for i, out_channels in enumerate(encoder_channels):
            dilation = 2 ** i  # Increasing dilation: 1, 2, 4, 8, ...
            padding = (kernel_size - 1) * dilation // 2  # Maintain same output length
            self.encoder_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, 
                         padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels
        
        # Calculate the size after convolutions (stride=1 preserves length)
        self.encoded_length = sequence_length
        
        # Flatten dimension
        self.flattened_dim = encoder_channels[-1] * self.encoded_length
        
        # MLP after CNN encoding
        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.flattened_dim, decoder_channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(decoder_channels[0], latent_dim)
        )
        
        # Layer normalization for latent vector
        self.latent_norm = nn.LayerNorm(latent_dim)
        
        # MLP Decoder start - project latent to initial feature map
        # Calculate initial length for decoder CNN - start small and upsample
        # We need to calculate the starting size to reach sequence_length after upsampling
        num_decoder_layers = len(decoder_channels) - 1
        self.decoder_initial_length = max(1, sequence_length // (2 ** num_decoder_layers))
        
        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, decoder_channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(decoder_channels[0], decoder_channels[0] * self.decoder_initial_length)
        )
        
        # CNN Decoder layers (ConvTranspose1d with stride=2, kernel=4)
        self.decoder_layers = nn.ModuleList()
        in_channels = decoder_channels[0]
        
        for i, out_channels in enumerate(decoder_channels[1:]):
            self.decoder_layers.append(nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels, out_channels, 
                    kernel_size=4, 
                    stride=2, 
                    padding=1,  # (kernel_size - stride) // 2 = (4-2)//2 = 1
                    output_padding=1  # Ensures exact output size
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
        
        # Classification heads - separate for x and y coordinates
        # Each outputs [B, num_classes, T] which will be transposed to [B, T, num_classes]
        self.x_classifier = nn.Conv1d(decoder_channels[-1], num_classes, kernel_size=1)
        self.y_classifier = nn.Conv1d(decoder_channels[-1], num_classes, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following best practices."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode gesture sequence to latent vector using CNN + MLP.
        
        Args:
            x: Input gesture sequence [B, T, 2] (x, y coordinates only)
            
        Returns:
            latent: Normalized latent vector [B, latent_dim]
        """
        B, T, C = x.shape
        
        # Transpose for Conv1d: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)  # [B, 2, T]
        
        # Encode through CNN layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Flatten
        x = x.view(B, -1)  # [B, flattened_dim]
        
        # Pass through MLP
        latent = self.encoder_mlp(x)  # [B, latent_dim]
        
        # Apply layer normalization
        latent = self.latent_norm(latent)  # [B, latent_dim]
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to classification logits using MLP + CNN upscaling.
        
        Args:
            latent: Latent vector [B, latent_dim]
            
        Returns:
            logits: Classification logits [B, T, 2, num_classes]
        """
        B = latent.shape[0]
        
        # Project latent to feature map
        features = self.latent_to_features(latent)  # [B, channels * initial_length]
        features = features.view(B, self.decoder_channels[0], self.decoder_initial_length)  # [B, C, L]
        
        # Upsample through transposed CNN layers
        for layer in self.decoder_layers:
            features = layer(features)
        
        # Final refinement
        features = self.final_conv(features)  # [B, final_channels, current_length]
        
        # Ensure exact sequence length match
        if features.shape[2] != self.sequence_length:
            features = F.interpolate(
                features, size=self.sequence_length, 
                mode='linear', align_corners=False
            )
        
        # Apply classification heads
        x_logits = self.x_classifier(features)  # [B, num_classes, T]
        y_logits = self.y_classifier(features)  # [B, num_classes, T]
        
        # Transpose to [B, T, num_classes] and stack for [B, T, 2, num_classes]
        x_logits = x_logits.transpose(1, 2)  # [B, T, num_classes]
        y_logits = y_logits.transpose(1, 2)  # [B, T, num_classes]
        
        logits = torch.stack([x_logits, y_logits], dim=2)  # [B, T, 2, num_classes]
        
        return logits
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            x: Input gesture sequence [B, T, 2] (x, y coordinates only)
            
        Returns:
            latent: Normalized latent vector [B, latent_dim]
            logits: Classification logits [B, T, 2, num_classes]
        """
        # Encode
        latent = self.encode(x)
        
        # Decode
        logits = self.decode(latent)
        
        return latent, logits


class CoordinateQuantizer:
    """Utilities for quantizing continuous coordinates to discrete classes."""
    
    def __init__(self, num_classes: int = 3000):
        self.num_classes = num_classes
    
    def quantize(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Quantize continuous coordinates to discrete class indices.
        
        Args:
            coords: Continuous coordinates [B, T, 2] in range [0, 1]
            
        Returns:
            class_indices: Discrete class indices [B, T, 2] in range [0, num_classes-1]
        """
        # Clamp to valid range and quantize
        coords_clamped = torch.clamp(coords, 0.0, 1.0)
        class_indices = (coords_clamped * self.num_classes).long()
        class_indices = torch.clamp(class_indices, 0, self.num_classes - 1)
        return class_indices
    
    def dequantize(self, class_indices: torch.Tensor) -> torch.Tensor:
        """
        Dequantize discrete class indices to continuous coordinates.
        
        Args:
            class_indices: Discrete class indices [B, T, 2]
            
        Returns:
            coords: Continuous coordinates [B, T, 2] in range [0, 1]
        """
        coords = class_indices.float() / self.num_classes
        return coords
    
    def argmax_decode(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Decode logits to coordinates via argmax.
        
        Args:
            logits: Classification logits [B, T, num_classes] or [B, T, 2, num_classes]
            
        Returns:
            coords: Decoded coordinates [B, T] or [B, T, 2] in range [0, 1]
        """
        class_indices = torch.argmax(logits, dim=-1)
        coords = class_indices.float() / self.num_classes
        return coords


def compute_classification_loss(
    logits: torch.Tensor,
    target_coords: torch.Tensor,
    quantizer: CoordinateQuantizer,
    w_smooth: float = 0.0,
    current_step: int = 0,
    steps_per_epoch: int = 1,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute cross-entropy classification loss for coordinate prediction with delta smoothness term.
    
    Args:
        logits: Classification logits [B, T, 2, num_classes]
        target_coords: Target coordinates [B, T, 2] in range [0, 1]
        quantizer: Coordinate quantizer
        w_smooth: Weight for delta smoothness loss (linearly annealed)
        current_step: Current training step
        steps_per_epoch: Number of steps per epoch for annealing
        
    Returns:
        total_loss: Combined cross-entropy loss with delta term
        loss_dict: Dictionary of individual losses
    """
    B, T = target_coords.shape[:2]
    
    # Quantize target coordinates
    target_classes = quantizer.quantize(target_coords)  # [B, T, 2]
    
    # Split logits for x and y coordinates
    x_logits = logits[:, :, 0, :]  # [B, T, num_classes]
    y_logits = logits[:, :, 1, :]  # [B, T, num_classes]
    
    # Flatten targets for cross-entropy
    target_x_classes = target_classes[:, :, 0].flatten()  # [B*T]
    target_y_classes = target_classes[:, :, 1].flatten()  # [B*T]
    
    # Compute cross-entropy losses for coordinates
    x_loss = F.cross_entropy(x_logits.reshape(-1, x_logits.size(-1)), target_x_classes)
    y_loss = F.cross_entropy(y_logits.reshape(-1, y_logits.size(-1)), target_y_classes)
    
    # Base coordinate loss
    coord_loss = x_loss + y_loss
    
    # Delta smoothness loss
    delta_loss = torch.tensor(0.0, device=logits.device)
    current_w_smooth = 0.0
    
    if w_smooth > 0.0 and T > 1:
        # Linear annealing: 0 to w_smooth over first epoch
        annealing_progress = min(1.0, current_step / max(1, steps_per_epoch))
        current_w_smooth = w_smooth * annealing_progress
        
        # Compute ground truth delta and quantize it
        target_delta = target_coords[:, 1:, :] - target_coords[:, :-1, :]  # [B, T-1, 2]
        
        # Shift delta to positive range for classification
        # Map delta from [-1, 1] to [0, 1] range: (delta + 1) / 2
        target_delta_shifted = (target_delta + 1.0) / 2.0  # [B, T-1, 2]
        target_delta_shifted = torch.clamp(target_delta_shifted, 0.0, 1.0)
        
        # Quantize delta to classes
        target_delta_classes = quantizer.quantize(target_delta_shifted)  # [B, T-1, 2]
        
        # Compute predicted delta from logits
        predicted_delta = compute_delta_class(logits, quantizer)  # [B, T-1, 2]
        
        # Shift predicted delta to same range
        predicted_delta_shifted = (predicted_delta + 1.0) / 2.0
        predicted_delta_shifted = torch.clamp(predicted_delta_shifted, 0.0, 1.0)
        
        # Quantize predicted delta to get logits for classification
        # We need to compute logits for delta classification
        # Use a simple approach: compute "pseudo-logits" from predicted delta
        predicted_delta_classes = quantizer.quantize(predicted_delta_shifted)  # [B, T-1, 2]
        
        # For proper CE loss, we need actual logits for delta predictions
        # Compute delta logits by taking differences of coordinate logits
        # This approximates the delta distribution
        coord_probs = F.softmax(logits, dim=-1)  # [B, T, 2, num_classes]
        delta_logits_x = coord_probs[:, 1:, 0, :] - coord_probs[:, :-1, 0, :]  # [B, T-1, num_classes]
        delta_logits_y = coord_probs[:, 1:, 1, :] - coord_probs[:, :-1, 1, :]  # [B, T-1, num_classes]
        
        # Shift delta probabilities to positive range and renormalize
        delta_logits_x_shifted = delta_logits_x + 1.0/quantizer.num_classes  # Add small positive bias
        delta_logits_y_shifted = delta_logits_y + 1.0/quantizer.num_classes
        
        # Convert back to logits (this is approximate but differentiable)
        delta_logits_x_final = torch.log(F.softmax(delta_logits_x_shifted, dim=-1) + 1e-8)
        delta_logits_y_final = torch.log(F.softmax(delta_logits_y_shifted, dim=-1) + 1e-8)
        
        # Compute delta classification loss
        target_delta_x_classes = target_delta_classes[:, :, 0].flatten()  # [B*(T-1)]
        target_delta_y_classes = target_delta_classes[:, :, 1].flatten()  # [B*(T-1)]
        
        delta_x_loss = F.cross_entropy(delta_logits_x_final.reshape(-1, delta_logits_x_final.size(-1)), target_delta_x_classes)
        delta_y_loss = F.cross_entropy(delta_logits_y_final.reshape(-1, delta_logits_y_final.size(-1)), target_delta_y_classes)
        
        delta_loss = delta_x_loss + delta_y_loss
    
    # Combined loss
    total_loss = coord_loss + current_w_smooth * delta_loss
    
    loss_dict = {
        'total_loss': total_loss.item(),
        'x_loss': x_loss.item(),
        'y_loss': y_loss.item(),
        'coord_loss': coord_loss.item(),
        'delta_loss': delta_loss.item(),
        'current_w_smooth': current_w_smooth,
    }
    
    return total_loss, loss_dict


def compute_delta_class(
    logits: torch.Tensor,
    quantizer: CoordinateQuantizer,
) -> torch.Tensor:
    """
    Compute delta (differences) of class predictions using probability-weighted means.
    This preserves smoothness by computing expected values before taking differences.
    
    Args:
        logits: Classification logits [B, T, 2, num_classes]
        quantizer: Coordinate quantizer
        
    Returns:
        delta_coords: Delta coordinates [B, T-1, 2]
    """
    B, T, _, num_classes = logits.shape
    
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)  # [B, T, 2, num_classes]
    
    # Create class indices tensor
    class_indices = torch.arange(num_classes, dtype=torch.float32, device=logits.device)  # [num_classes]
    
    # Compute expected class values (probability-weighted mean)
    # probs: [B, T, 2, num_classes], class_indices: [num_classes]
    expected_classes = torch.sum(probs * class_indices, dim=-1)  # [B, T, 2]
    
    # Convert expected class indices to coordinates
    expected_coords = expected_classes / quantizer.num_classes  # [B, T, 2]
    
    # Compute delta (differences between consecutive timesteps)
    delta_coords = expected_coords[:, 1:, :] - expected_coords[:, :-1, :]  # [B, T-1, 2]
    
    return delta_coords


def decode_predictions(
    logits: torch.Tensor,
    quantizer: CoordinateQuantizer,
) -> torch.Tensor:
    """
    Decode model predictions to gesture sequences.
    
    Args:
        logits: Classification logits [B, T, 2, num_classes]
        quantizer: Coordinate quantizer
        
    Returns:
        predictions: Decoded gesture sequences [B, T, 2]
    """
    # Decode coordinates using argmax
    coords = quantizer.argmax_decode(logits)  # [B, T, 2]
    
    return coords


class CNNGestureClassifierTrainer:
    """Training utilities for CNN Gesture Classifier."""
    
    def __init__(
        self,
        model: CNNGestureClassifier,
        quantizer: CoordinateQuantizer,
        device: str = 'cuda',
    ):
        self.model = model
        self.quantizer = quantizer
        self.device = device
        
    def train_step(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        w_smooth: float = 0.0,
        current_step: int = 0,
        steps_per_epoch: int = 1,
    ) -> dict:
        """
        Perform one training step.
        
        Args:
            batch: Gesture batch [B, T, 3] (we'll use only x, y)
            optimizer: Optimizer
            
        Returns:
            loss_dict: Dictionary of losses
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Move to device
        batch = batch.to(self.device)
        
        # Extract x, y coordinates (ignore pressure p)
        input_coords = batch[:, :, :2]  # [B, T, 2]
        
        # Forward pass
        latent, logits = self.model(input_coords)
        
        # Compute loss
        loss, loss_dict = compute_classification_loss(
            logits, input_coords, self.quantizer,
            w_smooth=w_smooth,
            current_step=current_step,
            steps_per_epoch=steps_per_epoch
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss_dict
    
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
        w_smooth: float = 0.0,
        current_step: int = 0,
        steps_per_epoch: int = 1,
    ) -> dict:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            avg_loss_dict: Average losses over validation set
        """
        self.model.eval()
        
        total_losses = {
            'total_loss': 0.0,
            'x_loss': 0.0,
            'y_loss': 0.0,
        }
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Extract x, y coordinates
                input_coords = batch[:, :, :2]
                
                # Forward pass
                latent, logits = self.model(input_coords)
                
                # Compute loss
                _, loss_dict = compute_classification_loss(
                    logits, input_coords, self.quantizer,
                    w_smooth=w_smooth,
                    current_step=current_step,
                    steps_per_epoch=steps_per_epoch
                )
                
                # Accumulate losses
                for key in total_losses:
                    total_losses[key] += loss_dict[key]
                num_batches += 1
        
        # Average losses
        avg_loss_dict = {
            key: total_losses[key] / num_batches 
            for key in total_losses
        }
        
        return avg_loss_dict
    
    def generate_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Generate gesture sequences from latent vectors.
        
        Args:
            latent: Latent vectors [B, latent_dim]
            
        Returns:
            predictions: Generated gesture sequences [B, T, 2]
        """
        self.model.eval()
        
        with torch.no_grad():
            latent = latent.to(self.device)
            
            # Decode latent to logits
            logits = self.model.decode(latent)
            
            # Convert to gesture sequences
            predictions = decode_predictions(logits, self.quantizer)
            
        return predictions.cpu()
    
    def reconstruct(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct gesture sequences (encode then decode).
        
        Args:
            batch: Input gesture sequences [B, T, 2/3] (will use only x, y)
            
        Returns:
            reconstructions: Reconstructed sequences [B, T, 2]
        """
        self.model.eval()
        
        with torch.no_grad():
            batch = batch.to(self.device)
            
            # Extract x, y coordinates
            input_coords = batch[:, :, :2]
            
            # Forward pass
            latent, logits = self.model(input_coords)
            
            # Convert to gesture sequences
            reconstructions = decode_predictions(logits, self.quantizer)
            
        return reconstructions.cpu()
    
    def encode_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Encode batch to latent vectors.
        
        Args:
            batch: Input gesture sequences [B, T, 2/3] (will use only x, y)
            
        Returns:
            latents: Latent vectors [B, latent_dim]
        """
        self.model.eval()
        
        with torch.no_grad():
            batch = batch.to(self.device)
            
            # Extract x, y coordinates
            input_coords = batch[:, :, :2]
            
            # Encode
            latents = self.model.encode(input_coords)
            
        return latents.cpu()
    
    def compute_delta_predictions(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute delta (smoothed) predictions from input batch.
        
        Args:
            batch: Input gesture sequences [B, T, 2/3] (will use only x, y)
            
        Returns:
            delta_coords: Delta coordinates [B, T-1, 2]
        """
        self.model.eval()
        
        with torch.no_grad():
            batch = batch.to(self.device)
            
            # Extract x, y coordinates
            input_coords = batch[:, :, :2]
            
            # Forward pass
            latent, logits = self.model(input_coords)
            
            # Compute delta coordinates
            delta_coords = compute_delta_class(logits, self.quantizer)
            
        return delta_coords.cpu()


def create_model_and_trainer(
    input_dim: int = 2,
    sequence_length: int = 250,
    latent_dim: int = 128,
    num_classes: int = 3000,
    encoder_channels: list = [64, 128, 256, 512],
    decoder_channels: list = [512, 256, 128, 64],
    device: str = 'cuda',
) -> Tuple[CNNGestureClassifier, CNNGestureClassifierTrainer, CoordinateQuantizer]:
    """
    Factory function to create CNN model, trainer, and quantizer.
    
    Returns:
        model: CNNGestureClassifier model
        trainer: CNNGestureClassifierTrainer
        quantizer: CoordinateQuantizer
    """
    # Create components
    model = CNNGestureClassifier(
        input_dim=input_dim,
        sequence_length=sequence_length,
        latent_dim=latent_dim,
        num_classes=num_classes,
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
    ).to(device)
    
    quantizer = CoordinateQuantizer(num_classes=num_classes)
    trainer = CNNGestureClassifierTrainer(model, quantizer, device=device)
    
    return model, trainer, quantizer


def recommend_architecture(sequence_length: int, target_latent_dim: int = None):
    """
    Recommend optimal CNN architecture for given sequence length.
    
    Args:
        sequence_length: Target sequence length
        target_latent_dim: Desired latent dimension (optional)
        
    Returns:
        dict: Recommended architecture parameters
    """
    import math
    
    # Calculate optimal depth (aim for 4-8 feature maps at bottleneck)
    min_bottleneck = 4
    max_bottleneck = 16
    
    # Find optimal number of encoder layers
    optimal_depth = max(2, int(math.log2(sequence_length // max_bottleneck)))
    max_depth = int(math.log2(sequence_length // min_bottleneck))
    
    # Recommend channel progression
    base_channels = 64
    encoder_channels = [base_channels * (2 ** i) for i in range(optimal_depth)]
    
    # Calculate actual bottleneck size
    bottleneck_size = sequence_length // (2 ** optimal_depth)
    
    # Recommend latent dimension
    if target_latent_dim is None:
        # Use rule of thumb: 1/8 to 1/16 of input dimensionality
        input_dim = sequence_length * 2  # x, y coordinates
        recommended_latent = max(32, min(256, input_dim // 8))
    else:
        recommended_latent = target_latent_dim
    
    return {
        'encoder_channels': encoder_channels,
        'latent_dim': recommended_latent,
        'optimal_depth': optimal_depth,
        'bottleneck_size': bottleneck_size,
        'max_depth': max_depth,
        'notes': f"Sequence {sequence_length} -> bottleneck {bottleneck_size} with {optimal_depth} layers"
    }


if __name__ == "__main__":
    # Test the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model, trainer, quantizer = create_model_and_trainer(
        sequence_length=250,
        latent_dim=128,
        device=device
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with sample data
    batch_size = 4
    sample_input = torch.randn(batch_size, 250, 2)  # [B, T, 2]
    
    # Forward pass
    latent, logits = model(sample_input.to(device))
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Logits shape: {logits.shape}")
    
    # Test loss computation
    loss, loss_dict = compute_classification_loss(logits, sample_input, quantizer)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
    
    # Test reconstruction
    reconstruction = trainer.reconstruct(sample_input)
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Test delta computation
    delta_coords = trainer.compute_delta_predictions(sample_input)
    print(f"Delta coordinates shape: {delta_coords.shape}")
    
    # Test compute_delta_class function directly
    delta_direct = compute_delta_class(logits, quantizer)
    print(f"Direct delta computation shape: {delta_direct.shape}")
    
    # Verify delta computation preserves smoothness
    # Compare delta of original vs delta of reconstructed
    original_delta = sample_input[:, 1:, :2] - sample_input[:, :-1, :2]
    reconstruction_delta = reconstruction[:, 1:, :] - reconstruction[:, :-1, :]
    delta_mse = torch.mean((original_delta - reconstruction_delta) ** 2)
    delta_computed_mse = torch.mean((original_delta - delta_coords) ** 2)
    
    print(f"Original vs reconstruction delta MSE: {delta_mse.item():.6f}")
    print(f"Original vs computed delta MSE: {delta_computed_mse.item():.6f}")
    
    # Recommend architecture for sequence length 250
    arch_rec = recommend_architecture(250)
    print(f"Architecture recommendation: {arch_rec}")