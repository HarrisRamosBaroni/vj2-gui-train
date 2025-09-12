import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .modules import LAMEncoder, LAMDecoder


class LatentActionVAE(nn.Module):
    """
    Latent Action Model VAE
    Combines encoder and decoder for learning action representations from latent sequences.
    """
    
    def __init__(
        self,
        latent_dim: int = 1024,
        action_dim: int = 128,
        embed_dim: int = 512,
        encoder_depth: int = 3,
        decoder_depth: int = 3,
        encoder_heads: int = 8,
        decoder_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        init_std: float = 0.02,
        kl_weight: float = 1.0,
        reconstruction_weight: float = 1.0
    ):
        """
        Args:
            latent_dim: Dimension of latent vectors from frozen encoder
            action_dim: Dimension of action latent space
            embed_dim: Dimension of transformer embeddings
            encoder_depth: Number of transformer blocks in encoder
            decoder_depth: Number of transformer blocks in decoder
            encoder_heads: Number of attention heads in encoder
            decoder_heads: Number of attention heads in decoder
            mlp_ratio: MLP hidden dim multiplier
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            init_std: Standard deviation for weight initialization
            kl_weight: Weight for KL divergence loss (beta in beta-VAE)
            reconstruction_weight: Weight for reconstruction loss
        """
        super().__init__()
        
        # Store hyperparameters for config saving
        self.config = {
            'patch_dim': latent_dim,  # Keep as latent_dim for backward compatibility but it's patch_dim
            'action_dim': action_dim,
            'embed_dim': embed_dim,
            'encoder_depth': encoder_depth,
            'decoder_depth': decoder_depth,
            'encoder_heads': encoder_heads,
            'decoder_heads': decoder_heads,
            'mlp_ratio': mlp_ratio,
            'drop_rate': drop_rate,
            'attn_drop_rate': attn_drop_rate,
            'init_std': init_std,
            'kl_weight': kl_weight,
            'reconstruction_weight': reconstruction_weight
        }
        
        # Store for backward compatibility
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.kl_weight = kl_weight
        self.reconstruction_weight = reconstruction_weight
        
        # Initialize encoder and decoder with new patch-based parameters
        self.encoder = LAMEncoder(
            patch_dim=latent_dim,  # 1024 - dimension of each patch token
            num_patches=256,       # 16x16 = 256 patches per frame
            action_dim=action_dim,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            init_std=init_std
        )
        
        self.decoder = LAMDecoder(
            patch_dim=latent_dim,  # 1024 - dimension of each patch token
            num_patches=256,       # 16x16 = 256 patches per frame  
            action_dim=action_dim,
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            init_std=init_std
        )
    
    def encode(self, z_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a sequence of patch tokens into per-transition action distribution parameters.
        
        Args:
            z_sequence: Sequence of patch tokens [z_1, ..., z_T], shape [B, T, N, D]
                        where N=256 patches, D=1024 patch dimension
        
        Returns:
            mu: Action means for each transition [B, T-1, A]
            logvar: Action log-variances for each transition [B, T-1, A]
        """
        return self.encoder(z_sequence)
    
    def decode(self, z_past: torch.Tensor, action_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode past patch sequence and action latent to predict next frame patches.
        
        Args:
            z_past: Past patch sequence [z_1, ..., z_{T-1}], shape [B, T-1, N, D]
                    where N=256 patches, D=1024 patch dimension
            action_latent: Action latent vector, shape [B, A]
        
        Returns:
            z_next_pred: Predicted next frame patches z_T, shape [B, N, D]
        """
        return self.decoder(z_past, action_latent)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from Gaussian.
        
        Args:
            mu: Mean of distribution [B, T-1, A] or [B, A]
            logvar: Log variance of distribution [B, T-1, A] or [B, A]
        
        Returns:
            Sampled action latent [B, T-1, A] or [B, A]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, return mean
            return mu
    
    def forward(
        self, 
        z_sequence: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VAE with per-transition actions (Genie-style).
        
        Args:
            z_sequence: Full sequence [z_1, ..., z_T], shape [B, T, N, D]
                        where N=256 patches, D=1024 patch dimension
                        
                        IMPORTANT: Input should be pre-normalized using:
                        F.layer_norm(z_sequence, (z_sequence.size(-1),))
                        This ensures compatibility with VJ2 predictor models.
            return_components: Whether to return individual loss components
        
        Returns:
            Dictionary containing:
                - 'reconstructions': Predicted frames [B, T-1, N, D]
                - 'mu': Action means [B, T-1, A]
                - 'logvar': Action log-variances [B, T-1, A]
                - 'action_latents': Sampled actions [B, T-1, A]
                - 'loss': Total VAE loss (if training)
                - 'recon_loss': Reconstruction loss component (if return_components)
                - 'kl_loss': KL divergence component (if return_components)
        """
        B, T, N, D = z_sequence.shape
        
        # Encode full sequence to get per-transition action distributions
        mu, logvar = self.encode(z_sequence)  # [B, T-1, A]
        
        # Clamp logvar for numerical stability
        logvar = logvar.clamp(-5, 5)
        
        # Sample action latents using reparameterization trick
        action_latents = self.reparameterize(mu, logvar)  # [B, T-1, A]
        
        # Decode each transition step-by-step
        reconstructions = []
        for t in range(T - 1):
            # Get past frames up to time t
            z_past = z_sequence[:, :t+1, :, :]  # [B, t+1, N, D]
            
            # Get action for transition from t to t+1
            action_t = action_latents[:, t, :]  # [B, A]
            
            # Predict next frame
            z_next_pred = self.decode(z_past, action_t)  # [B, N, D]
            reconstructions.append(z_next_pred)
        
        # Stack reconstructions
        reconstructions = torch.stack(reconstructions, dim=1)  # [B, T-1, N, D]
        
        # Prepare output dictionary
        output = {
            'reconstructions': reconstructions,
            'mu': mu,
            'logvar': logvar,
            'action_latents': action_latents
        }
        
        # Compute losses if in training mode
        if self.training or return_components:
            # Target frames are all frames except the first
            z_targets = z_sequence[:, 1:, :, :]  # [B, T-1, N, D]
            
            # Per-patch reconstruction loss (L1) for stable training
            # First compute element-wise L1, then average per patch, then across patches/batch
            recon = F.l1_loss(reconstructions, z_targets, reduction='none')  # [B, T-1, N, D]
            recon = recon.mean(dim=-1)  # Average over D dimension (per patch) -> [B, T-1, N]
            recon_loss = recon.mean()   # Average over batch, time, and patches
            
            # MSE loss for monitoring (computed similarly with per-patch normalization)
            mse = F.mse_loss(reconstructions, z_targets, reduction='none')  # [B, T-1, N, D]
            mse = mse.mean(dim=-1)  # Average per patch -> [B, T-1, N]
            mse_loss = mse.mean()    # Average over batch, time, and patches
            
            # KL divergence loss for each transition
            # KL(q(a|z) || p(a)) where p(a) = N(0, I)
            # mu, logvar shape: [B, T-1, A]
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # [B, T-1]
            kl_loss = kl_loss.mean()  # Average over batch and time
            
            # Total loss
            loss = self.reconstruction_weight * recon_loss + self.kl_weight * kl_loss
            
            output['loss'] = loss
            output['recon_loss'] = recon_loss
            output['mse_loss'] = mse_loss  
            output['kl_loss'] = kl_loss
        
        return output
    
    def generate(
        self, 
        z_past: torch.Tensor, 
        action_latent: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate next frame patches given past sequence.
        
        Args:
            z_past: Past patch sequence [z_1, ..., z_{T-1}], shape [B, T-1, N, D]
                    where N=256 patches, D=1024 patch dimension
                    
                    IMPORTANT: Input should be pre-normalized using:
                    F.layer_norm(z_past, (z_past.size(-1),))
                    This ensures compatibility with VJ2 predictor models.
            action_latent: Optional action latent [B, A]. If None, samples from prior N(0, I)
            temperature: Temperature for sampling (higher = more random)
        
        Returns:
            Predicted next frame patches [B, N, D]
        """
        B = z_past.shape[0]
        
        if action_latent is None:
            # Sample from prior
            action_latent = torch.randn(B, self.action_dim, device=z_past.device)
            action_latent = action_latent * temperature
        
        with torch.no_grad():
            z_next_pred = self.decode(z_past, action_latent)
        
        return z_next_pred
    
    def generate_sequence(
        self,
        z_init: torch.Tensor,
        num_steps: int,
        action_latents: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate a sequence of frames autoregressively.
        
        Args:
            z_init: Initial frame(s) [B, T_init, N, D]
            num_steps: Number of frames to generate
            action_latents: Optional action sequence [B, num_steps, A]
            temperature: Temperature for sampling
        
        Returns:
            Generated sequence [B, T_init + num_steps, N, D]
        """
        B = z_init.shape[0]
        device = z_init.device
        
        # Start with initial frames
        z_sequence = [z_init]
        
        for step in range(num_steps):
            # Get all frames generated so far
            z_past = torch.cat(z_sequence, dim=1)  # [B, T_past, N, D]
            
            # Get or sample action for this step
            if action_latents is not None:
                action = action_latents[:, step, :]  # [B, A]
            else:
                action = torch.randn(B, self.action_dim, device=device) * temperature
            
            # Generate next frame
            with torch.no_grad():
                z_next = self.decode(z_past, action)  # [B, N, D]
                z_sequence.append(z_next.unsqueeze(1))  # Add time dimension
        
        return torch.cat(z_sequence, dim=1)  # [B, T_init + num_steps, N, D]
    
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        beta_schedule: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss for a batch of data.
        
        Args:
            batch: Dictionary containing 'sequence', 'past', and 'next' tensors
            beta_schedule: Optional beta value for KL weight annealing
        
        Returns:
            Dictionary with loss values
        """
        z_sequence = batch['sequence']  # [B, T, N, D] patch format
        z_target = batch['next']  # [B, N, D] patch format
        
        # Forward pass
        output = self.forward(z_sequence, return_components=True)
        
        # Apply beta schedule if provided
        if beta_schedule is not None:
            kl_weight = beta_schedule
        else:
            kl_weight = self.kl_weight
        
        # Recompute total loss with scheduled beta
        total_loss = (
            self.reconstruction_weight * output['recon_loss'] + 
            kl_weight * output['kl_loss']
        )
        
        return {
            'loss': total_loss,
            'recon_loss': output['recon_loss'],
            'mse_loss': output['mse_loss'],
            'kl_loss': output['kl_loss'],
            'kl_weight': kl_weight
        }


if __name__ == "__main__":
    # Test the VAE model with Genie-style per-transition actions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model for overfit test
    model = LatentActionVAE(
        latent_dim=1024,
        action_dim=128,
        embed_dim=512,
        encoder_depth=3,
        decoder_depth=3,
        encoder_heads=8,
        decoder_heads=8,
        kl_weight=0.0,  # Turn off KL for overfit test
    ).to(device)
    
    # Create dummy data with correct shape [B, T, N, D]
    B, T, N, D = 2, 8, 256, 1024
    z_sequence = torch.randn(B, T, N, D, device=device)
    
    # Set to train mode for testing
    model.train()
    
    # Test forward pass
    output = model(z_sequence, return_components=True)
    
    print("VAE Output (Genie-style per-transition actions):")
    print(f"  Input shape: {z_sequence.shape} [B={B}, T={T}, N={N}, D={D}]")
    print(f"  Reconstructions shape: {output['reconstructions'].shape} [B, T-1, N, D]")
    print(f"  Mu shape: {output['mu'].shape} [B, T-1, A]")
    print(f"  Logvar shape: {output['logvar'].shape} [B, T-1, A]")
    print(f"  Action latents shape: {output['action_latents'].shape} [B, T-1, A]")
    print(f"  Total loss: {output['loss'].item():.4f}")
    print(f"  Recon loss: {output['recon_loss'].item():.4f}")
    print(f"  KL loss: {output['kl_loss'].item():.4f}")
    
    # Test single-step generation
    z_past = z_sequence[:, :3]  # Use first 3 frames as context
    action_for_next = output['action_latents'][:, 2, :]  # Action for transition from frame 2 to 3
    z_next_gen = model.generate(z_past, action_latent=action_for_next)
    print(f"\nGenerated next frame shape: {z_next_gen.shape} [B, N, D]")
    
    # Test sequence generation
    z_init = z_sequence[:, :1]  # Start with first frame
    num_steps = 5
    generated_seq = model.generate_sequence(z_init, num_steps, temperature=0.5)
    print(f"\nGenerated sequence shape: {generated_seq.shape} [B, {1 + num_steps}, N, D]")
    
    # Gradient sanity check
    print("\n=== Gradient Sanity Check ===")
    loss = output['loss']
    loss.backward()
    
    print("\nEncoder gradient check:")
    for n, p in list(model.encoder.named_parameters())[:5]:  # Check first 5 params
        if p.grad is None:
            print(f"  ⚠️  No gradient: {n}")
        else:
            grad_norm = p.grad.norm().item()
            print(f"  ✓ {n}: grad_norm={grad_norm:.6f}")
    
    print("\nDecoder gradient check:")
    for n, p in list(model.decoder.named_parameters())[:5]:  # Check first 5 params
        if p.grad is None:
            print(f"  ⚠️  No gradient: {n}")
        else:
            grad_norm = p.grad.norm().item()
            print(f"  ✓ {n}: grad_norm={grad_norm:.6f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


def load_model_from_config(config_path: str, checkpoint_path: str, device: str = "cuda") -> LatentActionVAE:
    """
    Load a model from saved config and checkpoint.
    
    Args:
        config_path: Path to model_config.json
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    import json
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Handle backward compatibility for config parameter names
    if 'patch_dim' in config:
        # New format with patch_dim
        model = LatentActionVAE(latent_dim=config['patch_dim'], **{k: v for k, v in config.items() if k != 'patch_dim'})
    else:
        # Old format or direct mapping
        model = LatentActionVAE(**config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model