import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gui_world_model.utils.modules import Block, MLP


class ActionVAE(nn.Module):
    """
    Variational Autoencoder for gesture prediction using discrete class labels.
    
    Architecture:
    - Encoder: Transformer with ACRoPEAttention -> latent distribution (μ, logvar)
    - Decoder: Latent upsampler -> Transformer -> 3000-class x/y + binary press classifiers
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        latent_dim: int = 64,
        model_dim: int = 256,
        sequence_length: int = 100,
        num_classes: int = 3000,
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dropout: float = 0.1,
        grid_size: int = 16,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.num_heads = num_heads
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, model_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, sequence_length, model_dim))
        
        # CLS token for sequence summarization
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        
        # Encoder layers with standard attention
        self.encoder_layers = nn.ModuleList([
            Block(
                dim=model_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                drop=dropout,
                attn_drop=dropout,
                use_rope=False,  # Use standard attention for sequences
                is_causal=False,  # Encoder sees full sequence
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Encoder normalization
        self.encoder_norm = nn.LayerNorm(model_dim)
        
        # Latent distribution heads
        self.mu_head = nn.Linear(model_dim, latent_dim)
        self.logvar_head = nn.Linear(model_dim, latent_dim)
        
        # Latent upsampler
        self.latent_proj = nn.Linear(latent_dim, model_dim)
        
        # Latent conditioning token
        self.latent_token = nn.Parameter(torch.randn(1, 1, model_dim))
        
        # Decoder layers with causal attention
        self.decoder_layers = nn.ModuleList([
            Block(
                dim=model_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                drop=dropout,
                attn_drop=dropout,
                use_rope=False,  # Use standard attention for sequences
                is_causal=True,   # Decoder is autoregressive
            )
            for _ in range(num_decoder_layers)
        ])
        
        # Decoder normalization
        self.decoder_norm = nn.LayerNorm(model_dim)
        
        # Output classifiers
        self.x_classifier = nn.Linear(model_dim, num_classes)
        self.y_classifier = nn.Linear(model_dim, num_classes)
        self.p_classifier = nn.Linear(model_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following best practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode gesture sequence to latent distribution.
        
        Args:
            x: Input gesture sequence [B, T, 3]
            
        Returns:
            mu: Mean of latent distribution [B, latent_dim]
            logvar: Log variance of latent distribution [B, latent_dim]
        """
        B, T, _ = x.shape
        
        # Project input and add positional embedding
        x = self.input_proj(x) + self.pos_embed[:, :T, :]  # [B, T, model_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, model_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, model_dim]
        
        # Encode through transformer layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        x = self.encoder_norm(x)
        
        # Extract CLS token representation
        cls_repr = x[:, 0, :]  # [B, model_dim]
        
        # Compute latent distribution parameters
        mu = self.mu_head(cls_repr)
        logvar = self.logvar_head(cls_repr)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean [B, latent_dim]
            logvar: Log variance [B, latent_dim]
            
        Returns:
            z: Sampled latent variable [B, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent variable to gesture sequence.
        
        Args:
            z: Latent variable [B, latent_dim]
            
        Returns:
            output: Decoded gesture logits [B, T, num_classes*2 + 1]
        """
        B = z.shape[0]
        T = self.sequence_length
        
        # Project latent to model dimension
        latent_proj = self.latent_proj(z)  # [B, model_dim]
        
        # Create latent conditioning token
        latent_tokens = self.latent_token.expand(B, -1, -1)  # [B, 1, model_dim]
        latent_tokens = latent_tokens + latent_proj.unsqueeze(1)
        
        # Create decoder input with positional embeddings
        decoder_input = self.pos_embed[:, :T, :].expand(B, -1, -1)  # [B, T, model_dim]
        
        # Concatenate latent token with sequence
        x = torch.cat([latent_tokens, decoder_input], dim=1)  # [B, T+1, model_dim]
        
        # Decode through transformer layers
        for layer in self.decoder_layers:
            x = layer(x)
        
        x = self.decoder_norm(x)
        
        # Extract sequence tokens (skip latent token)
        sequence_tokens = x[:, 1:, :]  # [B, T, model_dim]
        
        # Apply classifiers
        x_logits = self.x_classifier(sequence_tokens)  # [B, T, num_classes]
        y_logits = self.y_classifier(sequence_tokens)  # [B, T, num_classes]
        p_logits = self.p_classifier(sequence_tokens)  # [B, T, 1]
        
        return x_logits, y_logits, p_logits
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            x: Input gesture sequence [B, T, 3]
            
        Returns:
            mu: Latent mean [B, latent_dim]
            logvar: Latent log variance [B, latent_dim]
            x_logits: X coordinate logits [B, T, num_classes]
            y_logits: Y coordinate logits [B, T, num_classes]
            p_logits: Press logits [B, T, 1]
        """
        # Encode
        mu, logvar = self.encode(x)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_logits, y_logits, p_logits = self.decode(z)
        
        return mu, logvar, x_logits, y_logits, p_logits


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
            logits: Classification logits [B, T, num_classes]
            
        Returns:
            coords: Decoded coordinates [B, T] in range [0, 1]
        """
        class_indices = torch.argmax(logits, dim=-1)
        coords = class_indices.float() / self.num_classes
        return coords


def compute_vae_loss(
    x_logits: torch.Tensor,
    y_logits: torch.Tensor, 
    p_logits: torch.Tensor,
    target_coords: torch.Tensor,
    target_press: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    quantizer: CoordinateQuantizer,
    w_x: float = 1.0,
    w_y: float = 1.0,
    w_p: float = 0.5,
    w_kl: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute VAE loss with coordinate classification and KL regularization.
    
    Args:
        x_logits: X coordinate logits [B, T, num_classes]
        y_logits: Y coordinate logits [B, T, num_classes]  
        p_logits: Press logits [B, T, 1]
        target_coords: Target coordinates [B, T, 2] in range [0, 1]
        target_press: Target press states [B, T] in {0, 1}
        mu: Latent mean [B, latent_dim]
        logvar: Latent log variance [B, latent_dim]
        quantizer: Coordinate quantizer
        w_x, w_y, w_p, w_kl: Loss weights
        
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary of individual losses
    """
    B, T = target_coords.shape[:2]
    
    # Quantize target coordinates
    target_classes = quantizer.quantize(target_coords)  # [B, T, 2]
    target_x_classes = target_classes[:, :, 0].flatten()  # [B*T]
    target_y_classes = target_classes[:, :, 1].flatten()  # [B*T]
    
    # Coordinate classification losses
    x_loss = F.cross_entropy(x_logits.reshape(-1, x_logits.size(-1)), target_x_classes)
    y_loss = F.cross_entropy(y_logits.reshape(-1, y_logits.size(-1)), target_y_classes)
    
    # Press classification loss
    p_loss = F.binary_cross_entropy_with_logits(
        p_logits.squeeze(-1).flatten(), 
        target_press.flatten().float()
    )
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    # Combined loss
    total_loss = w_x * x_loss + w_y * y_loss + w_p * p_loss + w_kl * kl_loss
    
    loss_dict = {
        'total_loss': total_loss.item(),
        'x_loss': x_loss.item(),
        'y_loss': y_loss.item(), 
        'p_loss': p_loss.item(),
        'kl_loss': kl_loss.item(),
    }
    
    return total_loss, loss_dict


def decode_predictions(
    x_logits: torch.Tensor,
    y_logits: torch.Tensor,
    p_logits: torch.Tensor,
    quantizer: CoordinateQuantizer,
    press_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Decode model predictions to gesture sequences.
    
    Args:
        x_logits: X coordinate logits [B, T, num_classes]
        y_logits: Y coordinate logits [B, T, num_classes]
        p_logits: Press logits [B, T, 1]
        quantizer: Coordinate quantizer
        press_threshold: Threshold for press prediction
        
    Returns:
        predictions: Decoded gesture sequences [B, T, 3]
    """
    # Decode coordinates
    x_coords = quantizer.argmax_decode(x_logits)  # [B, T]
    y_coords = quantizer.argmax_decode(y_logits)  # [B, T]
    
    # Decode press states
    p_probs = torch.sigmoid(p_logits.squeeze(-1))  # [B, T]
    p_preds = (p_probs > press_threshold).float()  # [B, T]
    
    # Combine predictions
    predictions = torch.stack([x_coords, y_coords, p_preds], dim=-1)  # [B, T, 3]
    
    return predictions


class ActionVAETrainer:
    """Training utilities for ActionVAE."""
    
    def __init__(
        self,
        model: ActionVAE,
        quantizer: CoordinateQuantizer,
        device: str = 'cuda',
        kl_annealing_epochs: int = 50,
    ):
        self.model = model
        self.quantizer = quantizer
        self.device = device
        self.kl_annealing_epochs = kl_annealing_epochs
        
    def get_kl_weight(self, epoch: int) -> float:
        """Get KL annealing weight for current epoch."""
        if epoch >= self.kl_annealing_epochs:
            return 1.0
        return epoch / self.kl_annealing_epochs
    
    def train_step(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        w_x: float = 1.0,
        w_y: float = 1.0,
        w_p: float = 0.5,
    ) -> dict:
        """
        Perform one training step.
        
        Args:
            batch: Gesture batch [B, T, 3]
            optimizer: Optimizer
            epoch: Current epoch for KL annealing
            w_x, w_y, w_p: Loss weights
            
        Returns:
            loss_dict: Dictionary of losses
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Move to device
        batch = batch.to(self.device)
        
        # Split input and targets
        input_coords = batch[:, :, :2]  # [B, T, 2]
        input_press = batch[:, :, 2]    # [B, T]
        
        # Forward pass
        mu, logvar, x_logits, y_logits, p_logits = self.model(batch)
        
        # Compute loss with KL annealing
        w_kl = self.get_kl_weight(epoch)
        loss, loss_dict = compute_vae_loss(
            x_logits, y_logits, p_logits,
            input_coords, input_press,
            mu, logvar, self.quantizer,
            w_x=w_x, w_y=w_y, w_p=w_p, w_kl=w_kl
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Add KL weight to loss dict
        loss_dict['kl_weight'] = w_kl
        
        return loss_dict
    
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
        epoch: int,
        w_x: float = 1.0,
        w_y: float = 1.0, 
        w_p: float = 0.5,
    ) -> dict:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch for KL annealing
            w_x, w_y, w_p: Loss weights
            
        Returns:
            avg_loss_dict: Average losses over validation set
        """
        self.model.eval()
        
        total_losses = {
            'total_loss': 0.0,
            'x_loss': 0.0,
            'y_loss': 0.0,
            'p_loss': 0.0,
            'kl_loss': 0.0,
        }
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Split input and targets
                input_coords = batch[:, :, :2]
                input_press = batch[:, :, 2]
                
                # Forward pass
                mu, logvar, x_logits, y_logits, p_logits = self.model(batch)
                
                # Compute loss
                w_kl = self.get_kl_weight(epoch)
                _, loss_dict = compute_vae_loss(
                    x_logits, y_logits, p_logits,
                    input_coords, input_press,
                    mu, logvar, self.quantizer,
                    w_x=w_x, w_y=w_y, w_p=w_p, w_kl=w_kl
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
        avg_loss_dict['kl_weight'] = self.get_kl_weight(epoch)
        
        return avg_loss_dict
    
    def generate_samples(
        self,
        num_samples: int = 10,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate gesture samples from the model.
        
        Args:
            num_samples: Number of samples to generate
            temperature: Sampling temperature for latent space
            
        Returns:
            samples: Generated gesture sequences [num_samples, T, 3]
        """
        self.model.eval()
        
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(
                num_samples, self.model.latent_dim,
                device=self.device
            ) * temperature
            
            # Decode to gesture sequences
            x_logits, y_logits, p_logits = self.model.decode(z)
            
            # Convert to gesture sequences
            samples = decode_predictions(
                x_logits, y_logits, p_logits, self.quantizer
            )
            
        return samples.cpu()
    
    def reconstruct(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct gesture sequences (encode then decode).
        
        Args:
            batch: Input gesture sequences [B, T, 3]
            
        Returns:
            reconstructions: Reconstructed sequences [B, T, 3]
        """
        self.model.eval()
        
        with torch.no_grad():
            batch = batch.to(self.device)
            
            # Encode
            mu, logvar = self.model.encode(batch)
            
            # Use mean for reconstruction (no sampling)
            z = mu
            
            # Decode
            x_logits, y_logits, p_logits = self.model.decode(z)
            
            # Convert to gesture sequences
            reconstructions = decode_predictions(
                x_logits, y_logits, p_logits, self.quantizer
            )
            
        return reconstructions.cpu()
    
    def interpolate(
        self,
        start_sequence: torch.Tensor,
        end_sequence: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Interpolate between two gesture sequences in latent space.
        
        Args:
            start_sequence: Start sequence [1, T, 3]
            end_sequence: End sequence [1, T, 3]
            num_steps: Number of interpolation steps
            
        Returns:
            interpolations: Interpolated sequences [num_steps, T, 3]
        """
        self.model.eval()
        
        with torch.no_grad():
            start_sequence = start_sequence.to(self.device)
            end_sequence = end_sequence.to(self.device)
            
            # Encode both sequences
            start_mu, _ = self.model.encode(start_sequence)
            end_mu, _ = self.model.encode(end_sequence)
            
            # Create interpolation weights
            alphas = torch.linspace(0, 1, num_steps, device=self.device)
            
            interpolations = []
            for alpha in alphas:
                # Interpolate in latent space
                z_interp = (1 - alpha) * start_mu + alpha * end_mu
                
                # Decode
                x_logits, y_logits, p_logits = self.model.decode(z_interp)
                
                # Convert to gesture sequence
                interp_seq = decode_predictions(
                    x_logits, y_logits, p_logits, self.quantizer
                )
                
                interpolations.append(interp_seq)
            
            interpolations = torch.cat(interpolations, dim=0)
            
        return interpolations.cpu()


def create_model_and_trainer(
    input_dim: int = 3,
    latent_dim: int = 64,
    model_dim: int = 256,
    sequence_length: int = 100,
    num_classes: int = 3000,
    device: str = 'cuda',
) -> Tuple[ActionVAE, ActionVAETrainer, CoordinateQuantizer]:
    """
    Factory function to create model, trainer, and quantizer.
    
    Returns:
        model: ActionVAE model
        trainer: ActionVAETrainer
        quantizer: CoordinateQuantizer
    """
    # Create components
    model = ActionVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        model_dim=model_dim,
        sequence_length=sequence_length,
        num_classes=num_classes,
    ).to(device)
    
    quantizer = CoordinateQuantizer(num_classes=num_classes)
    trainer = ActionVAETrainer(model, quantizer, device=device)
    
    return model, trainer, quantizer
