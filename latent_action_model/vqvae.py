import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .modules import LAMEncoder, LAMDecoder
from .batched_decoder import BatchedVQLAMDecoder
import math

class VVAEtoLAMAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 256, kernel_size=4, stride=4, padding=0)
        self.norm = nn.BatchNorm2d(256)
    def forward(self, z_vvae):
        # z_vvae: [B, 16, 64, 64]
        x = self.conv(z_vvae)        # [B, 256, 16, 16]
        return self.norm(x)

class LAMtoVVAEAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(256, 16, kernel_size=4, stride=4, padding=0)
    def forward(self, z_lam):
        # z_lam: [B, 256, 16, 16]
        return self.deconv(z_lam)    # [B, 16, 64, 64]


class VVAELatentActionVQVAE(nn.Module):
    """
    Latent Action Model VQ-VAE adapted for VVAE latents.
    Wraps VQLatentActionVAE with adapters to handle VVAE format [B, T, C, H, W].
    """

    def __init__(
        self,
        codebook_dim: int = 128,
        num_embeddings: int = 12,
        embed_dim: int = 512,
        encoder_depth: int = 3,
        decoder_depth: int = 3,
        encoder_heads: int = 8,
        decoder_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        init_std: float = 0.02,
        commitment_weight: float = 0.25,
        reconstruction_weight: float = 1.0,
        max_seq_len: int = 20
    ):
        """
        Args:
            codebook_dim: Dimension of each code embedding
            num_embeddings: Size of codebook vocabulary (N)
            embed_dim: Dimension of transformer embeddings
            encoder_depth: Number of transformer blocks in encoder
            decoder_depth: Number of transformer blocks in decoder
            encoder_heads: Number of attention heads in encoder
            decoder_heads: Number of attention heads in decoder
            mlp_ratio: MLP hidden dim multiplier
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            init_std: Standard deviation for weight initialization
            commitment_weight: Weight for commitment loss (beta)
            reconstruction_weight: Weight for reconstruction loss
            max_seq_len: Maximum sequence length for positional embeddings
        """
        super().__init__()

        # Store hyperparameters for config saving
        self.config = {
            'model_type': 'vvae_lam',
            'codebook_dim': codebook_dim,
            'num_embeddings': num_embeddings,
            'embed_dim': embed_dim,
            'encoder_depth': encoder_depth,
            'decoder_depth': decoder_depth,
            'encoder_heads': encoder_heads,
            'decoder_heads': decoder_heads,
            'mlp_ratio': mlp_ratio,
            'drop_rate': drop_rate,
            'attn_drop_rate': attn_drop_rate,
            'init_std': init_std,
            'commitment_weight': commitment_weight,
            'reconstruction_weight': reconstruction_weight,
            'max_seq_len': max_seq_len
        }

        self.codebook_dim = codebook_dim
        self.num_embeddings = num_embeddings
        self.commitment_weight = commitment_weight
        self.reconstruction_weight = reconstruction_weight

        # Adapters for VVAE <-> LAM format conversion
        self.vvae_to_lam = VVAEtoLAMAdapter()
        self.lam_to_vvae = LAMtoVVAEAdapter()

        # Core VQ-LAM with reduced patch_dim for VVAE
        self.lam = VQLatentActionVAE(
            latent_dim=256,  # Reduced from 1024 to match adapter output
            codebook_dim=codebook_dim,
            num_embeddings=num_embeddings,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            encoder_heads=encoder_heads,
            decoder_heads=decoder_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            init_std=init_std,
            commitment_weight=commitment_weight,
            reconstruction_weight=reconstruction_weight,
            max_seq_len=max_seq_len
        )

    def vvae_to_patches(self, z_vvae: torch.Tensor) -> torch.Tensor:
        """
        Convert VVAE latent format to LAM patch format.

        Args:
            z_vvae: VVAE latent [B, T, C=16, H=64, W=64]

        Returns:
            z_patches: LAM patch format [B, T, N=256, D=256]
        """
        B, T, C, H, W = z_vvae.shape
        assert C == 16 and H == 64 and W == 64, f"Expected [B, T, 16, 64, 64], got {z_vvae.shape}"

        # Process each frame through adapter
        z_patches_list = []
        for t in range(T):
            z_frame = z_vvae[:, t]  # [B, 16, 64, 64]
            z_adapted = self.vvae_to_lam(z_frame)  # [B, 256, 16, 16]

            # Reshape to patch format: [B, 256, 16, 16] -> [B, 256, 256]
            z_flat = z_adapted.flatten(2, 3)  # [B, 256, 256]
            z_patches = z_flat.transpose(1, 2)  # [B, 256, 256]
            z_patches_list.append(z_patches)

        # Stack along time dimension
        z_patches = torch.stack(z_patches_list, dim=1)  # [B, T, 256, 256]
        return z_patches

    def patches_to_vvae(self, z_patches: torch.Tensor) -> torch.Tensor:
        """
        Convert LAM patch format back to VVAE latent format.

        Args:
            z_patches: LAM patch format [B, T, N=256, D=256]

        Returns:
            z_vvae: VVAE latent [B, T, C=16, H=64, W=64]
        """
        B, T, N, D = z_patches.shape
        assert N == 256 and D == 256, f"Expected [B, T, 256, 256], got {z_patches.shape}"

        # Process each frame through inverse adapter
        z_vvae_list = []
        for t in range(T):
            z_frame = z_patches[:, t]  # [B, 256, 256]

            # Reshape to spatial format: [B, 256, 256] -> [B, 256, 16, 16]
            z_spatial = z_frame.transpose(1, 2)  # [B, 256, 256]
            z_spatial = z_spatial.reshape(B, 256, 16, 16)  # [B, 256, 16, 16]

            # Apply inverse adapter
            z_vvae_frame = self.lam_to_vvae(z_spatial)  # [B, 16, 64, 64]
            z_vvae_list.append(z_vvae_frame)

        # Stack along time dimension
        z_vvae = torch.stack(z_vvae_list, dim=1)  # [B, T, 16, 64, 64]
        return z_vvae

    def encode(self, z_sequence_vvae: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode VVAE sequence to continuous code distribution parameters.

        Args:
            z_sequence_vvae: VVAE latent sequence [B, T, 16, 64, 64]

        Returns:
            mu: Continuous code means [B, T-1, 3*codebook_dim]
            logvar: Continuous code log-variances [B, T-1, 3*codebook_dim]
        """
        z_patches = self.vvae_to_patches(z_sequence_vvae)
        return self.lam.encode(z_patches)

    def decode(self, z_past_vvae: torch.Tensor, code_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Decode past VVAE sequence and code embeddings to predict next frame.

        Args:
            z_past_vvae: Past VVAE sequence [B, t+1, 16, 64, 64]
            code_embeddings: Code embeddings [B, t+1, 3, codebook_dim]

        Returns:
            z_next_pred_vvae: Predicted next frame [B, 16, 64, 64]
        """
        z_past_patches = self.vvae_to_patches(z_past_vvae)
        z_next_patches = self.lam.decode(z_past_patches, code_embeddings)  # [B, 256, 256]

        # Convert single frame back to VVAE format
        z_next_patches = z_next_patches.unsqueeze(1)  # [B, 1, 256, 256]
        z_next_vvae = self.patches_to_vvae(z_next_patches)  # [B, 1, 16, 64, 64]
        return z_next_vvae.squeeze(1)  # [B, 16, 64, 64]

    def forward(
        self,
        z_sequence_vvae: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VVAE-adapted VQ-LAM.

        Args:
            z_sequence_vvae: VVAE sequence [B, T, 16, 64, 64]
            return_components: Whether to return individual loss components

        Returns:
            Dictionary containing:
                - 'reconstructions': Predicted frames [B, T-1, 16, 64, 64]
                - 'indices': Codebook indices [B, T-1, 3]
                - 'loss': Total VQ-VAE loss (if training)
                - 'recon_loss': Reconstruction loss component
                - 'vq_loss': VQ loss component
                - 'codebook_usage': Number of unique codes used
        """
        # Convert to patch format
        z_patches = self.vvae_to_patches(z_sequence_vvae)

        # Run through core VQ-LAM
        output = self.lam.forward(z_patches, return_components=return_components)

        # Convert reconstructions back to VVAE format
        recon_patches = output['reconstructions']  # [B, T-1, 256, 256]
        recon_vvae = self.patches_to_vvae(recon_patches)  # [B, T-1, 16, 64, 64]
        output['reconstructions'] = recon_vvae

        return output

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        beta_schedule: float = 1.0,  # Not used in VQ-VAE, kept for compatibility
        rollout_horizon: int = 2,
        rollout_weight: float = 1.0,
        rollout_prob: float = 1.0,
        detach_rollout_first: bool = True,
        anchor_strategy: str = "random"
    ) -> Dict[str, torch.Tensor]:
        """
        VQ-VAE compute loss with VVAE sequences using MSE (instead of MAE).

        Args:
            batch: Dict with 'sequence' key containing [B, T, 16, 64, 64] tensor
            beta_schedule: Unused (VQ-VAE doesn't have KL), kept for compatibility
            rollout_horizon: Number of steps to roll out
            rollout_weight: Weight for rollout loss
            rollout_prob: Probability of computing rollout
            detach_rollout_first: Whether to detach first predicted state
            anchor_strategy: "random" or "last" for rollout anchor selection

        Returns:
            Dict with 'loss', 'recon_loss', 'mae_loss', 'mse_loss', 'vq_loss', 'rollout_loss', 'codebook_usage', 'indices'
        """
        z_sequence_vvae = batch["sequence"]  # [B, T, 16, 64, 64]
        B, T, C, H, W = z_sequence_vvae.shape
        assert T >= 2, "Need at least two frames"

        # Convert to patch format
        z_patches = self.vvae_to_patches(z_sequence_vvae)  # [B, T, 256, 256]

        # ----- 1) Encode and quantize from GT sequence -----
        mu, logvar = self.lam.encode(z_patches)  # [B, T-1, 3*codebook_dim]
        logvar = logvar.clamp(-5, 5)
        z_e = self.lam.reparameterize(mu, logvar)  # [B, T-1, 3*codebook_dim]

        # Quantize to discrete codes
        z_q, indices, vq_loss, codebook_loss, commitment_loss = self.lam.quantize(z_e)  # z_q: [B, T-1, 3, codebook_dim]

        # ----- 2) Teacher-forced next-frame predictions for ALL steps -----
        # Use batched decoder for parallel processing (causal masking)
        recons_patches = self.lam.batched_decoder(z_patches[:, :-1], z_q)  # [B, T-1, 256, 256]

        # Convert reconstructions back to VVAE format
        recons_vvae = self.patches_to_vvae(recons_patches)  # [B, T-1, 16, 64, 64]

        # Reconstruction loss using MSE (not MAE!)
        targets = z_sequence_vvae[:, 1:]  # [B, T-1, 16, 64, 64]

        # MSE as primary reconstruction loss
        mse_per_pixel = (recons_vvae - targets).pow(2)  # [B, T-1, 16, 64, 64]
        mse_loss = mse_per_pixel.mean()  # Average over all dimensions

        # MAE for monitoring only
        mae_per_pixel = torch.abs(recons_vvae - targets)  # [B, T-1, 16, 64, 64]
        mae_loss = mae_per_pixel.mean()  # Average over all dimensions

        # Base loss: MSE + VQ (not KL!)
        total_loss = self.reconstruction_weight * mse_loss + vq_loss

        # Also compute MAE-based total loss for comparison/monitoring
        total_loss_mae = self.reconstruction_weight * mae_loss + vq_loss

        # ----- 3) Optional decoder-only rollout -----
        rollout_loss = torch.zeros((), device=z_sequence_vvae.device)
        rollout_loss_mae = torch.zeros((), device=z_sequence_vvae.device)
        if rollout_horizon > 1 and torch.rand(()) < rollout_prob:
            max_start = (T - 1) - rollout_horizon
            if max_start >= 0:
                if anchor_strategy == "random":
                    t0 = int(torch.randint(low=0, high=max_start + 1, size=(1,)).item())
                else:
                    t0 = max_start

                # Get first predicted state in patch format
                z_cur_patches = recons_patches[:, t0]  # [B, 256, 256]
                if detach_rollout_first:
                    z_cur_patches = z_cur_patches.detach()

                # Roll forward K-1 steps
                z_hist_patches = z_patches[:, :t0+1]  # [B, t0+1, 256, 256]
                acc = 0.0
                count = 0
                for k in range(1, rollout_horizon):
                    z_hist_patches = torch.cat([z_hist_patches, z_cur_patches.unsqueeze(1)], dim=1)
                    codes_k = z_q[:, :t0+1+k, :, :]  # Use GT codes
                    z_cur_patches = self.lam.decode(z_hist_patches, codes_k)

                    # Convert to VVAE format for comparison
                    z_cur_vvae = self.patches_to_vvae(z_cur_patches.unsqueeze(1)).squeeze(1)  # [B, 16, 64, 64]
                    z_true_vvae = z_sequence_vvae[:, t0 + 1 + k]  # [B, 16, 64, 64]

                    # Use MSE for rollout loss (primary)
                    acc += (z_cur_vvae - z_true_vvae).pow(2).mean()
                    count += 1
                rollout_loss = acc / max(1, count)

                # Also compute MAE version for rollout
                acc_mae = 0.0
                z_cur_patches_mae = recons_patches[:, t0]  # [B, 256, 256]
                if detach_rollout_first:
                    z_cur_patches_mae = z_cur_patches_mae.detach()
                z_hist_patches_mae = z_patches[:, :t0+1]  # [B, t0+1, 256, 256]
                for k in range(1, rollout_horizon):
                    z_hist_patches_mae = torch.cat([z_hist_patches_mae, z_cur_patches_mae.unsqueeze(1)], dim=1)
                    codes_k = z_q[:, :t0+1+k, :, :]
                    z_cur_patches_mae = self.lam.decode(z_hist_patches_mae, codes_k)
                    z_cur_vvae_mae = self.patches_to_vvae(z_cur_patches_mae.unsqueeze(1)).squeeze(1)
                    z_true_vvae = z_sequence_vvae[:, t0 + 1 + k]
                    acc_mae += torch.abs(z_cur_vvae_mae - z_true_vvae).mean()
                rollout_loss_mae = acc_mae / max(1, count)

                total_loss = total_loss + rollout_weight * rollout_loss
                total_loss_mae = total_loss_mae + rollout_weight * rollout_loss_mae

        # Codebook usage statistics
        unique_codes = torch.unique(indices).numel()

        return {
            "loss": total_loss,                    # Total loss using MSE (for optimization)
            "loss_mae": total_loss_mae,            # Total loss using MAE (for monitoring)
            "recon_loss": mse_loss,                # Reconstruction loss is MSE
            "mae_loss": mae_loss,                  # MAE reconstruction (monitoring)
            "mse_loss": mse_loss,                  # MSE reconstruction (same as recon_loss)
            "vq_loss": vq_loss,                    # VQ loss (codebook + commitment)
            "codebook_loss": codebook_loss,        # Codebook loss component
            "commitment_loss": commitment_loss,    # Commitment loss component
            "rollout_loss": rollout_loss,          # Rollout loss using MSE
            "rollout_loss_mae": rollout_loss_mae,  # Rollout loss using MAE (monitoring)
            "codebook_usage": unique_codes,        # Number of unique codes used
            "indices": indices,                    # [B, T-1, 3] for histogram
        }

    def generate(
        self,
        z_past_vvae: torch.Tensor,
        code_indices: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate next frame given past VVAE sequence.

        Args:
            z_past_vvae: Past VVAE sequence [B, T-1, 16, 64, 64]
            code_indices: Optional code indices [B, 3]. If None, samples randomly
            temperature: Temperature for sampling (unused for VQ-VAE)

        Returns:
            z_next_pred_vvae: Predicted next frame [B, 16, 64, 64]
        """
        z_past_patches = self.vvae_to_patches(z_past_vvae)
        z_next_patches = self.lam.generate(z_past_patches, code_indices, temperature)

        # Convert back to VVAE format
        z_next_patches = z_next_patches.unsqueeze(1)  # [B, 1, 256, 256]
        z_next_vvae = self.patches_to_vvae(z_next_patches)  # [B, 1, 16, 64, 64]
        return z_next_vvae.squeeze(1)  # [B, 16, 64, 64]

    def generate_sequence(
        self,
        z_init_vvae: torch.Tensor,
        num_steps: int,
        code_indices: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate a sequence of VVAE frames autoregressively.

        Args:
            z_init_vvae: Initial frame(s) [B, T_init, 16, 64, 64]
            num_steps: Number of frames to generate
            code_indices: Optional code sequence [B, num_steps, 3]
            temperature: Temperature for sampling (unused for VQ-VAE)

        Returns:
            Generated VVAE sequence [B, T_init + num_steps, 16, 64, 64]
        """
        z_init_patches = self.vvae_to_patches(z_init_vvae)
        z_seq_patches = self.lam.generate_sequence(z_init_patches, num_steps, code_indices, temperature)

        # Convert back to VVAE format
        z_seq_vvae = self.patches_to_vvae(z_seq_patches)
        return z_seq_vvae


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

            # Per-patch reconstruction loss (L1)
            recon = F.l1_loss(reconstructions, z_targets, reduction='none')  # [B, T-1, N, D]
            recon = recon.mean(dim=-1)  # Average over D dimension (per patch) -> [B, T-1, N]

            # --- Motion energy weighting ---
            # Compute latent energy of ground-truth transitions
            energy = (z_sequence[:, 1:] - z_sequence[:, :-1]).pow(2).mean(dim=[2, 3])  # [B, T-1]
            mean_energy = energy.mean() + 1e-8
            weights = (energy / mean_energy).detach()  # [B, T-1], stop grad through weights
            # Optional smoothing if you have spikes:
            weights = torch.tanh(weights / 2.0) + 0.1

            # Apply weights when reducing across time/batch
            recon_loss = (recon.mean(dim=-1) * weights).mean()

            # MSE loss for monitoring (weighted as well)
            mse = F.mse_loss(reconstructions, z_targets, reduction='none')  # [B, T-1, N, D]
            mse = mse.mean(dim=-1)  # Average per patch -> [B, T-1, N]
            mse_loss = (mse.mean(dim=-1) * weights).mean()
            
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
        beta_schedule: float = 1.0,
        rollout_horizon: int = 2,          # 1 ⇒ no rollout; 2–3 is typical
        rollout_weight: float = 1.0,       # scale for rollout loss
        rollout_prob: float = 1.0,         # compute rollout with this probability
        detach_rollout_first: bool = True, # detach first predicted state before rolling
        anchor_strategy: str = "random"      # "last" or "random"
    ) -> Dict[str, torch.Tensor]:
        """
        Decoder-only rollout:
          - Encoder ALWAYS sees GT z-seq once to produce a_seq (no iterative encoding).
          - Decoder rolls forward from its OWN prediction, but actions stay GT.
          - Adds only (rollout_horizon - 1) extra decode calls per batch (anchor shared).
        """
        z_sequence = batch["sequence"]  # [B,T,N,D], already layer-normed by trainer
        B, T, N, D = z_sequence.shape
        assert T >= 2, "Need at least two frames"

        # ----- 1) Encode actions ONCE from GT sequence (NO rollout on encoder) -----
        mu, logvar = self.encode(z_sequence)             # [B, T-1, A]
        logvar = logvar.clamp(-5, 5)
        a_seq = self.reparameterize(mu, logvar)          # [B, T-1, A]

        # ----- 2) Teacher-forced next-frame predictions for ALL steps -----
        # Reuse your existing per-step decode loop (keeps behavior identical).
        recons = []
        for t in range(T - 1):
            z_past = z_sequence[:, :t+1]                 # [B, t+1, N, D]
            a_t = a_seq[:, t]                            # [B, A]
            z_next_pred = self.decode(z_past, a_t)       # [B, N, D]
            recons.append(z_next_pred)
        recons = torch.stack(recons, dim=1)              # [B, T-1, N, D]

        # Reconstruction & MSE (teacher-forced) over ALL steps
        targets = z_sequence[:, 1:]                      # [B, T-1, N, D]
        recon = torch.abs(recons - targets).mean(dim=-1) # per-patch MAE
        recon_loss = recon.mean()
        mse = (recons - targets).pow(2).mean(dim=-1)
        mse_loss = mse.mean()

        # KL over transitions (use scheduler-provided weight)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)  # [B, T-1]
        kl_loss = kl.mean()

        # Base loss (no rollout yet)
        total_loss = self.reconstruction_weight * recon_loss + beta_schedule * kl_loss

        # ----- 3) Optional decoder-only rollout starting from ONE anchor -----
        rollout_loss = torch.zeros((), device=z_sequence.device)
        if rollout_horizon > 1 and torch.rand(()) < rollout_prob:
            # Valid anchors: t in [0, T-1-rollout_horizon]
            max_start = (T - 1) - rollout_horizon
            if max_start >= 0:
                if anchor_strategy == "random":
                    t0 = int(torch.randint(low=0, high=max_start + 1, size=(1,)).item())
                else:
                    t0 = max_start  # "last": start as far right as possible

                # Step 1 (already computed via teacher forcing):
                # prediction for z_{t0+1} is recons[:, t0]
                z_cur = recons[:, t0]                     # [B, N, D]
                if detach_rollout_first:
                    z_cur = z_cur.detach()               # stabilize gradients

                # Roll forward K-1 steps using GT actions but predicted states
                # Extra decodes: (rollout_horizon - 1)
                z_hist = z_sequence[:, :t0+1]             # [B, t0+1, N, D]
                acc = 0.0
                count = 0
                for k in range(1, rollout_horizon):
                    # History grows by 1 predicted state each step
                    z_hist = torch.cat([z_hist, z_cur.unsqueeze(1)], dim=1)  # [B, t0+1+k, N, D]
                    a_k = a_seq[:, t0 + k]                                   # GT action for this step
                    z_cur = self.decode(z_hist, a_k)                         # predict next state
                    z_true = z_sequence[:, t0 + 1 + k]                       # GT state
                    acc += torch.abs(z_cur - z_true).mean()                  # MAE
                    count += 1
                rollout_loss = acc / max(1, count)

                total_loss = total_loss + rollout_weight * rollout_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "mse_loss": mse_loss,
            "kl_loss": kl_loss,
            "rollout_loss": rollout_loss,
        }



class VQLatentActionVAE(nn.Module):
    """
    VQ-VAE variant of Latent Action Model
    Uses discrete codebook with 3 codes per action for N^3 expressiveness.
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        codebook_dim: int = 128,
        num_embeddings: int = 512,
        embed_dim: int = 512,
        encoder_depth: int = 3,
        decoder_depth: int = 3,
        encoder_heads: int = 8,
        decoder_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        init_std: float = 0.02,
        commitment_weight: float = 0.25,
        reconstruction_weight: float = 1.0,
        max_seq_len: int = 20
    ):
        """
        Args:
            latent_dim: Dimension of latent vectors from frozen encoder
            codebook_dim: Dimension of each code embedding
            num_embeddings: Size of codebook vocabulary (N)
            embed_dim: Dimension of transformer embeddings
            encoder_depth: Number of transformer blocks in encoder
            decoder_depth: Number of transformer blocks in decoder
            encoder_heads: Number of attention heads in encoder
            decoder_heads: Number of attention heads in decoder
            mlp_ratio: MLP hidden dim multiplier
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            init_std: Standard deviation for weight initialization
            commitment_weight: Weight for commitment loss (beta)
            reconstruction_weight: Weight for reconstruction loss
            max_seq_len: Maximum sequence length for positional embeddings
        """
        super().__init__()

        # Store hyperparameters for config saving
        self.config = {
            'model_type': 'vqvae',
            'patch_dim': latent_dim,
            'codebook_dim': codebook_dim,
            'num_embeddings': num_embeddings,
            'embed_dim': embed_dim,
            'encoder_depth': encoder_depth,
            'decoder_depth': decoder_depth,
            'encoder_heads': encoder_heads,
            'decoder_heads': decoder_heads,
            'mlp_ratio': mlp_ratio,
            'drop_rate': drop_rate,
            'attn_drop_rate': attn_drop_rate,
            'init_std': init_std,
            'commitment_weight': commitment_weight,
            'reconstruction_weight': reconstruction_weight,
            'max_seq_len': max_seq_len
        }

        # Store for backward compatibility
        self.latent_dim = latent_dim
        self.codebook_dim = codebook_dim
        self.num_embeddings = num_embeddings
        self.commitment_weight = commitment_weight
        self.reconstruction_weight = reconstruction_weight

        # Initialize encoder and decoder with VQ parameters
        self.encoder = VQLAMEncoder(
            patch_dim=latent_dim,
            num_patches=256,
            codebook_dim=codebook_dim,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            init_std=init_std,
            max_seq_len=max_seq_len
        )

        self.decoder = VQLAMDecoder(
            patch_dim=latent_dim,
            num_patches=256,
            codebook_dim=codebook_dim,
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            init_std=init_std,
            max_seq_len=max_seq_len
        )

        # Batched decoder for parallel teacher forcing during training
        self.batched_decoder = BatchedVQLAMDecoder(
            patch_dim=latent_dim,
            num_patches=256,
            codebook_dim=codebook_dim,
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            init_std=init_std,
            max_seq_len=max_seq_len
        )

        # Codebook: vocabulary of size N
        self.codebook = nn.Embedding(num_embeddings, codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def quantize(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous codes to discrete codebook entries.

        Args:
            z_e: Continuous codes from encoder [B, T-1, 3*codebook_dim]

        Returns:
            z_q: Quantized codes [B, T-1, 3, codebook_dim]
            indices: Codebook indices [B, T-1, 3]
            vq_loss: Combined codebook + commitment loss
        """
        B, T_minus_1, _ = z_e.shape

        # Reshape to separate 3 codes: [B, T-1, 3*codebook_dim] -> [B, T-1, 3, codebook_dim]
        z_e = z_e.reshape(B, T_minus_1, 3, self.codebook_dim)

        # Flatten for batch processing: [B, T-1, 3, codebook_dim] -> [B*(T-1)*3, codebook_dim]
        z_e_flat = z_e.reshape(-1, self.codebook_dim)

        # Compute distances to all codebook entries
        # ||z_e - e||^2 = ||z_e||^2 + ||e||^2 - 2*z_e·e
        z_e_sq = torch.sum(z_e_flat ** 2, dim=1, keepdim=True)  # [B*(T-1)*3, 1]
        codebook_sq = torch.sum(self.codebook.weight ** 2, dim=1)  # [num_embeddings]
        distances = z_e_sq + codebook_sq - 2 * torch.matmul(z_e_flat, self.codebook.weight.t())
        # distances: [B*(T-1)*3, num_embeddings]

        # Find nearest codebook entries
        indices_flat = torch.argmin(distances, dim=1)  # [B*(T-1)*3]

        # Lookup quantized codes
        z_q_flat = self.codebook(indices_flat)  # [B*(T-1)*3, codebook_dim]

        # Reshape back
        z_q = z_q_flat.reshape(B, T_minus_1, 3, self.codebook_dim)
        indices = indices_flat.reshape(B, T_minus_1, 3)

        # VQ losses
        # Codebook loss: ||sg[z_e] - z_q||^2 (moves codebook toward encoder outputs)
        codebook_loss = F.mse_loss(z_q, z_e.detach())

        # Commitment loss: ||z_e - sg[z_q]||^2 (encourages encoder to commit)
        commitment_loss = F.mse_loss(z_e, z_q.detach())

        vq_loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator: copy gradients from z_q to z_e
        z_q = z_e + (z_q - z_e).detach()

        return z_q, indices, vq_loss, codebook_loss, commitment_loss

    def encode(self, z_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequence and return continuous codes (before quantization).

        Args:
            z_sequence: Sequence of patch tokens [B, T, N, D]

        Returns:
            mu: Continuous codes [B, T-1, 3*codebook_dim]
            logvar: Log-variances [B, T-1, 3*codebook_dim]
        """
        return self.encoder(z_sequence)

    def decode(self, z_past: torch.Tensor, code_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Decode past patches and code embeddings to predict next frame.

        Args:
            z_past: Past patch sequence [B, T-1, N, D]
            code_embeddings: Code embeddings [B, T-1, 3, codebook_dim]

        Returns:
            z_next_pred: Predicted next frame patches [B, N, D]
        """
        return self.decoder(z_past, code_embeddings)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling continuous codes.

        Args:
            mu: Mean [B, T-1, 3*codebook_dim]
            logvar: Log variance [B, T-1, 3*codebook_dim]

        Returns:
            Sampled continuous codes [B, T-1, 3*codebook_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(
        self,
        z_sequence: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VQ-VAE.

        Args:
            z_sequence: Full sequence [z_1, ..., z_T], shape [B, T, N, D]
            return_components: Whether to return individual loss components

        Returns:
            Dictionary containing:
                - 'reconstructions': Predicted frames [B, T-1, N, D]
                - 'indices': Codebook indices [B, T-1, 3]
                - 'loss': Total loss (if training)
                - 'recon_loss': Reconstruction loss component
                - 'vq_loss': VQ loss component
                - 'codebook_usage': Number of unique codes used
        """
        B, T, N, D = z_sequence.shape

        # Encode full sequence to get continuous codes
        mu, logvar = self.encode(z_sequence)  # [B, T-1, 3*codebook_dim]

        # Clamp logvar for numerical stability
        logvar = logvar.clamp(-5, 5)

        # Sample continuous codes
        z_e = self.reparameterize(mu, logvar)  # [B, T-1, 3*codebook_dim]

        # Quantize to discrete codes
        z_q, indices, vq_loss, codebook_loss, commitment_loss = self.quantize(z_e)  # z_q: [B, T-1, 3, codebook_dim]

        # Decode each transition step-by-step
        reconstructions = []
        for t in range(T - 1):
            # Get past frames up to time t
            z_past = z_sequence[:, :t+1, :, :]  # [B, t+1, N, D]

            # Get codes for this transition (all 3 codes from steps 0 to t)
            codes_past = z_q[:, :t+1, :, :]  # [B, t+1, 3, codebook_dim]

            # Predict next frame
            z_next_pred = self.decode(z_past, codes_past)  # [B, N, D]
            reconstructions.append(z_next_pred)

        # Stack reconstructions
        reconstructions = torch.stack(reconstructions, dim=1)  # [B, T-1, N, D]

        # Prepare output dictionary
        output = {
            'reconstructions': reconstructions,
            'indices': indices,
            'mu': mu,
            'logvar': logvar
        }

        # Compute losses if in training mode
        if self.training or return_components:
            # Target frames are all frames except the first
            z_targets = z_sequence[:, 1:, :, :]  # [B, T-1, N, D]

            # Per-patch reconstruction loss (L1)
            recon = F.l1_loss(reconstructions, z_targets, reduction='none')  # [B, T-1, N, D]
            recon = recon.mean(dim=-1)  # Average over D dimension -> [B, T-1, N]
            recon_loss = recon.mean()   # Average over batch, time, and patches

            # MSE loss for monitoring
            mse = F.mse_loss(reconstructions, z_targets, reduction='none')
            mse = mse.mean(dim=-1)
            mse_loss = mse.mean()

            # Total loss
            loss = self.reconstruction_weight * recon_loss + vq_loss

            # Codebook usage statistics
            unique_codes = torch.unique(indices).numel()

            output['loss'] = loss
            output['recon_loss'] = recon_loss
            output['mse_loss'] = mse_loss
            output['vq_loss'] = vq_loss
            output['codebook_loss'] = codebook_loss
            output['commitment_loss'] = commitment_loss
            output['codebook_usage'] = unique_codes

        return output

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        beta_schedule: float = 1.0,  # Not used in VQ-VAE, kept for compatibility
        rollout_horizon: int = 2,
        rollout_weight: float = 1.0,
        rollout_prob: float = 1.0,
        detach_rollout_first: bool = True,
        anchor_strategy: str = "random"
    ) -> Dict[str, torch.Tensor]:
        """
        VQ-VAE compute loss with optional rollout.
        Similar to VAE's compute_loss but uses VQ losses instead of KL.

        Args:
            batch: Dict with 'sequence' key containing [B, T, N, D] tensor
            beta_schedule: Unused (VQ-VAE doesn't have KL), kept for compatibility
            rollout_horizon: Number of steps to roll out
            rollout_weight: Weight for rollout loss
            rollout_prob: Probability of computing rollout
            detach_rollout_first: Whether to detach first predicted state
            anchor_strategy: "random" or "last" for rollout anchor selection

        Returns:
            Dict with 'loss', 'recon_loss', 'vq_loss', 'rollout_loss', 'codebook_usage', 'indices'
        """
        z_sequence = batch["sequence"]  # [B,T,N,D], already layer-normed by trainer
        B, T, N, D = z_sequence.shape
        assert T >= 2, "Need at least two frames"

        # ----- 1) Encode actions ONCE from GT sequence -----
        mu, logvar = self.encode(z_sequence)  # [B, T-1, 3*codebook_dim]
        logvar = logvar.clamp(-5, 5)
        z_e = self.reparameterize(mu, logvar)  # [B, T-1, 3*codebook_dim]

        # ----- 2) Quantize to discrete codes -----
        z_q, indices, vq_loss, codebook_loss, commitment_loss = self.quantize(z_e)  # z_q: [B, T-1, 3, codebook_dim]

        # ----- 3) Teacher-forced next-frame predictions for ALL steps -----
        # Use batched decoder for parallel processing (causal masking)
        recons = self.batched_decoder(z_sequence[:, :-1], z_q)  # [B, T-1, N, D]

        # Reconstruction loss (MAE)
        targets = z_sequence[:, 1:]  # [B, T-1, N, D]
        recon = torch.abs(recons - targets).mean(dim=-1)  # per-patch MAE
        recon_loss = recon.mean()

        # MSE for monitoring
        mse = (recons - targets).pow(2).mean(dim=-1)
        mse_loss = mse.mean()

        # Total base loss
        total_loss = self.reconstruction_weight * recon_loss + vq_loss

        # ----- 4) Optional decoder-only rollout -----
        rollout_loss = torch.zeros((), device=z_sequence.device)
        if rollout_horizon > 1 and torch.rand(()) < rollout_prob:
            max_start = (T - 1) - rollout_horizon
            if max_start >= 0:
                if anchor_strategy == "random":
                    t0 = int(torch.randint(low=0, high=max_start + 1, size=(1,)).item())
                else:
                    t0 = max_start

                z_cur = recons[:, t0]  # [B, N, D]
                if detach_rollout_first:
                    z_cur = z_cur.detach()

                z_hist = z_sequence[:, :t0+1]  # [B, t0+1, N, D]
                acc = 0.0
                count = 0
                for k in range(1, rollout_horizon):
                    z_hist = torch.cat([z_hist, z_cur.unsqueeze(1)], dim=1)
                    codes_k = z_q[:, :t0+1+k, :, :]  # Use GT codes
                    z_cur = self.decode(z_hist, codes_k)
                    z_true = z_sequence[:, t0 + 1 + k]
                    acc += torch.abs(z_cur - z_true).mean()
                    count += 1
                rollout_loss = acc / max(1, count)
                total_loss = total_loss + rollout_weight * rollout_loss

        # Codebook usage statistics
        unique_codes = torch.unique(indices).numel()

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "mse_loss": mse_loss,
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "rollout_loss": rollout_loss,
            "codebook_usage": unique_codes,
            "indices": indices,  # [B, T-1, 3] for monitoring
        }

    def generate(
        self,
        z_past: torch.Tensor,
        code_indices: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate next frame given past sequence.

        Args:
            z_past: Past patch sequence [B, T-1, N, D]
            code_indices: Optional code indices [B, 3]. If None, samples randomly
            temperature: Temperature for sampling (not used for discrete codes)

        Returns:
            Predicted next frame patches [B, N, D]
        """
        B = z_past.shape[0]
        T_past = z_past.shape[1]

        if code_indices is None:
            # Sample random code indices
            code_indices = torch.randint(0, self.num_embeddings, (B, 3), device=z_past.device)

        # Lookup code embeddings
        code_embeddings = self.codebook(code_indices)  # [B, 3, codebook_dim]
        code_embeddings = code_embeddings.unsqueeze(1)  # [B, 1, 3, codebook_dim]

        # Need to provide all past codes, but we only have the last one
        # For generation, we'll use a simplified approach
        with torch.no_grad():
            # For now, assume we're generating from scratch with just the new codes
            # This is a simplification - in practice you'd track all past codes
            z_next_pred = self.decode(z_past, code_embeddings.squeeze(1).unsqueeze(1).expand(-1, T_past, -1, -1))

        return z_next_pred


class VQLAMEncoder(nn.Module):
    """
    VQ Latent Action Model Encoder
    Encodes a sequence of patch token latents into per-transition action codes (3 codes per transition).
    For each consecutive pair (z_t, z_{t+1}), predicts 3*codebook_dim continuous values.
    """

    def __init__(
        self,
        patch_dim=1024,      # D: patch embedding dimension from V-JEPA
        num_patches=256,     # N: number of patches per frame (H*W)
        codebook_dim=128,    # Dimension of each code embedding
        embed_dim=512,
        depth=3,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        init_std=0.02,
        max_seq_len=20,      # Maximum temporal sequence length
        patch_size=16        # For spatial position calculation
    ):
        """
        Args:
            patch_dim: Dimension of each patch token from V-JEPA (D=1024)
            num_patches: Number of patches per frame (N=256 for 16x16 grid)
            codebook_dim: Dimension of each code embedding (will output 3*codebook_dim)
            embed_dim: Dimension of transformer embeddings
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim multiplier
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            init_std: Standard deviation for weight initialization
            max_seq_len: Maximum temporal sequence length
            patch_size: Patch size for spatial position calculation
        """
        super().__init__()

        self.patch_dim = patch_dim
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.codebook_dim = codebook_dim

        # Calculate spatial grid size (assuming square)
        self.grid_size = int(math.sqrt(num_patches))  # 16x16 = 256 patches

        # Patch embedding projection
        self.patch_proj = nn.Linear(patch_dim, embed_dim)

        # No CLS token - we predict actions per timestep

        # Temporal positional embeddings (per frame)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(max_seq_len, embed_dim))

        # Spatial positional embeddings (per patch position in frame)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(num_patches, embed_dim))

        # Transformer blocks (copied from LAMEncoder)
        from .modules import TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop_rate,
                drop=drop_rate
            )
            for _ in range(depth)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Frame-level aggregation: pool patches to get frame representation
        self.frame_pool = nn.Linear(embed_dim, embed_dim)

        # Output heads for 3 codes per transition (mu and logvar for each)
        self.mu_head = nn.Linear(embed_dim, 3 * codebook_dim)
        self.logvar_head = nn.Linear(embed_dim, 3 * codebook_dim)

        # Initialize weights
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        for layer_id, block in enumerate(self.blocks):
            layer_scale = math.sqrt(2.0 * (layer_id + 1))
            if hasattr(block.attn, 'out_proj'):
                block.attn.out_proj.weight.data.div_(layer_scale)
            if isinstance(block.mlp, nn.Sequential):
                block.mlp[-2].weight.data.div_(layer_scale)

    def _create_shifted_causal_mask(self, T: int, N: int, device: torch.device) -> torch.Tensor:
        """
        Create shifted causal mask for Genie-style encoding.
        When encoding action for transition t→(t+1), allow seeing frames [0, ..., t+1].

        Args:
            T: Number of frames in sequence
            N: Number of patches per frame
            device: Device for mask tensor

        Returns:
            mask: [T*N, T*N] attention mask where -inf blocks attention
        """
        # Create frame-level mask: frame t can see frames [0, 1, ..., t+1]
        # Standard causal: diagonal=1 means frame t sees [0, ..., t]
        # Shifted causal: diagonal=2 means frame t sees [0, ..., t+1]
        frame_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=device),
            diagonal=2  # Shifted by 1: allow seeing one frame into the future
        )  # [T, T]

        # Expand to patch level: each frame becomes N×N block
        mask = frame_mask.repeat_interleave(N, dim=0).repeat_interleave(N, dim=1)
        # [T*N, T*N]

        return mask

    def forward(self, z_sequence):
        """
        Args:
            z_sequence: Sequence of patch tokens [B, T, N, D] where:
                B = batch size
                T = temporal sequence length
                N = number of patches per frame (256)
                D = patch embedding dimension (1024)

        Returns:
            mu: Continuous codes for each transition, shape [B, T-1, 3*codebook_dim]
            logvar: Log-variances for each transition, shape [B, T-1, 3*codebook_dim]
        """
        B, T, N, D = z_sequence.shape
        assert N == self.num_patches, f"Expected {self.num_patches} patches, got {N}"
        assert D == self.patch_dim, f"Expected patch dim {self.patch_dim}, got {D}"

        # Project patch tokens to embedding dimension
        # [B, T, N, D] -> [B, T, N, embed_dim]
        x = self.patch_proj(z_sequence)

        # Layer normalize after projection (helps with stability)
        x = F.layer_norm(x, (self.embed_dim,))

        # Add spatial positional embeddings (same for all frames)
        spatial_pos = self.spatial_pos_embed.unsqueeze(0).unsqueeze(0)
        x = x + spatial_pos.to(x.device)   # [B, T, N, embed_dim]

        # Add temporal positional embeddings (same for all patches in a frame)
        temporal_pos = self.temporal_pos_embed[:T].unsqueeze(0).unsqueeze(2)
        x = x + temporal_pos.to(x.device)   # [B, T, N, embed_dim]

        # Reshape to sequence: [B, T*N, embed_dim]
        x = x.reshape(B, T * N, self.embed_dim)

        # Create shifted causal mask for Genie-style encoding
        # Position t can see frames [0, ..., t+1] to encode action t→(t+1)
        causal_mask = self._create_shifted_causal_mask(T, N, x.device)

        # Apply transformer blocks with shifted causal masking
        for block in self.blocks:
            x = block(x, mask=causal_mask)

        # Normalize output
        x = self.norm(x)  # [B, T*N, embed_dim]

        # Reshape back to frame structure
        x = x.reshape(B, T, N, self.embed_dim)  # [B, T, N, embed_dim]

        # Pool patches within each frame to get frame-level representations
        # Mean pooling over spatial patches
        frame_features = x.mean(dim=2)  # [B, T, embed_dim]

        # Apply frame pooling layer for better representation
        frame_features = self.frame_pool(frame_features)  # [B, T, embed_dim]

        # For each transition (z_t, z_{t+1}), we use the representation at timestep t
        # This gives us T-1 transition representations
        transition_features = frame_features[:, :-1, :]  # [B, T-1, embed_dim]

        # Output 3*codebook_dim values for each transition
        mu = self.mu_head(transition_features)       # [B, T-1, 3*codebook_dim]
        logvar = self.logvar_head(transition_features)  # [B, T-1, 3*codebook_dim]

        return mu, logvar


class VQLAMDecoder(nn.Module):
    """
    VQ Latent Action Model Decoder with Interleaved Sequence Processing
    Processes interleaved sequence: [state_1, code_{1,1}, code_{1,2}, code_{1,3}, state_2, ...]
    Uses shared positional encoding for each (state, code, code, code) group.
    """

    def __init__(
        self,
        patch_dim=1024,
        num_patches=256,
        codebook_dim=128,
        embed_dim=512,
        depth=3,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        init_std=0.02,
        max_seq_len=20
    ):
        """
        Args:
            patch_dim: Dimension of each patch token (D=1024)
            num_patches: Number of patches per frame (N=256)
            codebook_dim: Dimension of code embeddings
            embed_dim: Dimension of transformer embeddings
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim multiplier
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            init_std: Standard deviation for weight initialization
            max_seq_len: Maximum temporal sequence length
        """
        super().__init__()

        self.patch_dim = patch_dim
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.codebook_dim = codebook_dim

        # Input projections for interleaved sequence
        self.patch_proj = nn.Linear(patch_dim, embed_dim)  # For state patches
        self.code_proj = nn.Linear(codebook_dim, embed_dim)  # For code embeddings

        # Learnable query embeddings for next frame patches
        self.query_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Positional embeddings
        # For interleaved sequence: max_seq_len * 4 tokens (state + 3 codes per timestep)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(max_seq_len, embed_dim))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(num_patches, embed_dim))
        # Separate spatial positional embeddings for queries (target frame)
        self.query_pos_embed = nn.Parameter(torch.zeros(num_patches, embed_dim))

        # Cross-attention decoder blocks (copied from LAMDecoder)
        from .modules import CrossAttentionBlock
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop_rate,
                drop=drop_rate
            )
            for _ in range(depth)
        ])

        # Final normalization and output projection
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, patch_dim)

        # Initialize weights
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        # Initialize embeddings
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.query_pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        for layer_id, block in enumerate(self.blocks):
            layer_scale = math.sqrt(2.0 * (layer_id + 1))
            # For CrossAttentionBlock, we have self_attn and cross_attn
            if hasattr(block, 'self_attn') and hasattr(block.self_attn, 'out_proj'):
                block.self_attn.out_proj.weight.data.div_(layer_scale)
            if hasattr(block, 'cross_attn') and hasattr(block.cross_attn, 'out_proj'):
                block.cross_attn.out_proj.weight.data.div_(layer_scale)
            if isinstance(block.mlp, nn.Sequential):
                block.mlp[-2].weight.data.div_(layer_scale)

    def forward(self, z_past, code_embeddings):
        """
        Args:
            z_past: Past patch sequence [B, T-1, N, D] where T-1 is past frames
            code_embeddings: Code embeddings for each transition [B, T-1, 3, codebook_dim]
                            3 codes per transition

        Returns:
            z_next_pred: Predicted next frame patches [B, N, D]
        """
        B, T_past, N, D = z_past.shape
        assert N == self.num_patches, f"Expected {self.num_patches} patches, got {N}"
        assert D == self.patch_dim, f"Expected patch dim {self.patch_dim}, got {D}"

        # === Build Interleaved Memory Sequence ===
        # Structure: [state_1_patches, code_1_1, code_1_2, code_1_3, state_2_patches, code_2_1, ...]

        # Project past patches to embedding dimension
        # [B, T_past, N, D] -> [B, T_past, N, embed_dim]
        z_embed = self.patch_proj(z_past)
        z_embed = F.layer_norm(z_embed, (self.embed_dim,))

        # Add spatial positional embeddings to past patches
        spatial_pos = self.spatial_pos_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, N, embed_dim]
        z_embed = z_embed + spatial_pos.to(z_embed.device)

        # Project code embeddings to embedding dimension
        # [B, T_past, 3, codebook_dim] -> [B, T_past, 3, embed_dim]
        code_embed = self.code_proj(code_embeddings)
        code_embed = F.layer_norm(code_embed, (self.embed_dim,))

        # Build interleaved sequence
        memory_tokens = []
        for t in range(T_past):
            # Add temporal position to state patches
            temporal_pos = self.temporal_pos_embed[t:t+1].unsqueeze(1)  # [1, 1, embed_dim]
            state_t = z_embed[:, t, :, :] + temporal_pos  # [B, N, embed_dim]

            # Flatten state patches to sequence
            memory_tokens.append(state_t)  # [B, N, embed_dim]

            # Add 3 code tokens with SAME temporal position
            for code_idx in range(3):
                code_t = code_embed[:, t, code_idx, :].unsqueeze(1)  # [B, 1, embed_dim]
                code_t = code_t + temporal_pos  # Share position with state_t
                memory_tokens.append(code_t)  # [B, 1, embed_dim]

        # Concatenate all memory tokens
        # Total length: T_past * (N + 3) tokens
        memory = torch.cat(memory_tokens, dim=1)  # [B, T_past*(N+3), embed_dim]

        # === Prepare Queries: Learnable tokens for next frame patches ===

        # Initialize query tokens with learnable embeddings
        queries = self.query_embed.expand(B, -1, -1)  # [B, N, embed_dim]

        # Add positional embeddings for target frame patches
        query_pos = self.query_pos_embed.unsqueeze(0)  # [1, N, embed_dim]
        queries = queries + query_pos  # [B, N, embed_dim]

        # === Cross-Attention Decoding ===

        # Apply cross-attention blocks
        # Each block: queries attend to memory (interleaved states + codes)
        for block in self.blocks:
            queries = block(queries, memory)

        # === Output Prediction ===

        # Final normalization
        queries = self.norm(queries)  # [B, N, embed_dim]

        # Project each query to patch space
        z_next_pred = self.output_proj(queries)  # [B, N, D]

        return z_next_pred



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

    # =====================================================
    # VQ-VAE Test
    # =====================================================
    print("\n" + "="*60)
    print("VQ-VAE Test")
    print("="*60)

    # Initialize VQ-VAE model
    vq_model = VQLatentActionVAE(
        latent_dim=1024,
        codebook_dim=128,
        num_embeddings=512,
        embed_dim=512,
        encoder_depth=3,
        decoder_depth=3,
        encoder_heads=8,
        decoder_heads=8,
        commitment_weight=0.25,
    ).to(device)

    # Create dummy data with correct shape [B, T, N, D]
    B, T, N, D = 2, 8, 256, 1024
    z_sequence_vq = torch.randn(B, T, N, D, device=device)

    # Set to train mode for testing
    vq_model.train()

    # Test forward pass
    output_vq = vq_model(z_sequence_vq, return_components=True)

    print("\nVQ-VAE Output:")
    print(f"  Input shape: {z_sequence_vq.shape} [B={B}, T={T}, N={N}, D={D}]")
    print(f"  Reconstructions shape: {output_vq['reconstructions'].shape} [B, T-1, N, D]")
    print(f"  Indices shape: {output_vq['indices'].shape} [B, T-1, 3] (3 codes per transition)")
    print(f"  Mu shape: {output_vq['mu'].shape} [B, T-1, 3*codebook_dim]")
    print(f"  Total loss: {output_vq['loss'].item():.4f}")
    print(f"  Recon loss: {output_vq['recon_loss'].item():.4f}")
    print(f"  VQ loss: {output_vq['vq_loss'].item():.4f}")
    print(f"  Codebook usage: {output_vq['codebook_usage']}/{vq_model.num_embeddings} unique codes")

    # Show code distribution
    print(f"\n  Code indices sample (first transition, first batch):")
    print(f"    {output_vq['indices'][0, 0].cpu().tolist()}")
    print(f"  Vocabulary size: {vq_model.num_embeddings}")
    print(f"  Possible combinations: {vq_model.num_embeddings}^3 = {vq_model.num_embeddings**3:,}")

    # Test generation
    z_past_vq = z_sequence_vq[:, :3]  # Use first 3 frames as context
    code_indices_test = torch.tensor([[42, 100, 255]], device=device).expand(B, -1)  # [B, 3]
    z_next_gen_vq = vq_model.generate(z_past_vq, code_indices=code_indices_test)
    print(f"\n  Generated next frame shape: {z_next_gen_vq.shape} [B, N, D]")

    # Gradient sanity check for VQ-VAE
    print("\n=== VQ-VAE Gradient Sanity Check ===")
    loss_vq = output_vq['loss']
    vq_model.zero_grad()
    loss_vq.backward()

    print("\nVQ Encoder gradient check:")
    for n, p in list(vq_model.encoder.named_parameters())[:5]:
        if p.grad is None:
            print(f"  ⚠️  No gradient: {n}")
        else:
            grad_norm = p.grad.norm().item()
            print(f"  ✓ {n}: grad_norm={grad_norm:.6f}")

    print("\nVQ Decoder gradient check:")
    for n, p in list(vq_model.decoder.named_parameters())[:5]:
        if p.grad is None:
            print(f"  ⚠️  No gradient: {n}")
        else:
            grad_norm = p.grad.norm().item()
            print(f"  ✓ {n}: grad_norm={grad_norm:.6f}")

    print("\nCodebook gradient check:")
    if vq_model.codebook.weight.grad is None:
        print(f"  ⚠️  No gradient: codebook.weight")
    else:
        grad_norm = vq_model.codebook.weight.grad.norm().item()
        print(f"  ✓ codebook.weight: grad_norm={grad_norm:.6f}")

    # Count VQ-VAE parameters
    vq_total_params = sum(p.numel() for p in vq_model.parameters())
    vq_trainable_params = sum(p.numel() for p in vq_model.parameters() if p.requires_grad)
    codebook_params = vq_model.codebook.weight.numel()
    print(f"\nVQ-VAE Total parameters: {vq_total_params:,}")
    print(f"VQ-VAE Trainable parameters: {vq_trainable_params:,}")
    print(f"Codebook parameters: {codebook_params:,} ({vq_model.num_embeddings} codes × {vq_model.codebook_dim} dim)")

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)

def load_lam_config(path: str) -> dict:
    import yaml
    # Load YAML
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Get the dynamic run info (the first entry under "_wandb -> value -> e")
    e_dict = data["_wandb"]["value"]["e"]
    run_info = next(iter(e_dict.values()))  # works even if key is random

    # Parse args into a dict
    args = run_info["args"]
    arg_dict = {args[i].lstrip("-"): args[i+1] for i in range(0, len(args), 2)}

    # Build lam_config
    lam_config = {
        "latent_dim": data["latent_dim"]["value"],
        "action_dim": data["action_dim"]["value"],
        "embed_dim": int(arg_dict["embed_dim"]),
        "encoder_depth": int(arg_dict["encoder_depth"]),
        "decoder_depth": int(arg_dict["decoder_depth"]),
        "encoder_heads": int(arg_dict["encoder_heads"]),
        "decoder_heads": int(arg_dict["decoder_heads"]),
        "kl_weight": float(arg_dict["kl_weight"]),
    }
    return lam_config

def load_model_from_config(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """
    Load a model from saved config and checkpoint.
    Supports LatentActionVAE, VQLatentActionVAE, and VVAELatentActionVQVAE.

    Args:
        config_path: Path to model_config.json
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model on

    Returns:
        Loaded model (LatentActionVAE, VQLatentActionVAE, or VVAELatentActionVQVAE)
    """
    import json

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Check model type
    model_type = config.get('model_type', 'vae')

    if model_type == 'vvae_lam':
        # Load VVAE LAM model (VQ-VAE variant)
        # Remove model_type from config before passing to constructor
        config_copy = {k: v for k, v in config.items() if k != 'model_type'}
        model = VVAELatentActionVQVAE(**config_copy)
    elif model_type == 'vqvae':
        # Load VQ-VAE model
        # Handle backward compatibility for parameter names
        if 'patch_dim' in config:
            latent_dim = config.pop('patch_dim')
        else:
            latent_dim = config.get('latent_dim', 1024)

        # Remove model_type from config before passing to constructor
        config_copy = {k: v for k, v in config.items() if k != 'model_type'}
        model = VQLatentActionVAE(latent_dim=latent_dim, **config_copy)
    else:
        # Load regular VAE model
        # Handle backward compatibility for config parameter names
        if 'patch_dim' in config:
            # New format with patch_dim
            model = LatentActionVAE(latent_dim=config['patch_dim'], **{k: v for k, v in config.items() if k not in ['patch_dim', 'model_type']})
        else:
            # Old format or direct mapping
            model = LatentActionVAE(**{k: v for k, v in config.items() if k != 'model_type'})

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    return model
