import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    """
    Combined VAE loss function with classification reconstruction loss and KL divergence.
    """
    
    def __init__(self, k_classes=3000, beta=1.0, reduction='mean'):
        """
        Args:
            k_classes: Number of quantization classes
            beta: Weight for KL divergence term (beta-VAE)
            reduction: Reduction method for losses ('mean', 'sum', 'none')
        """
        super().__init__()
        self.k_classes = k_classes
        self.beta = beta
        self.reduction = reduction
        
        # Classification loss for reconstruction
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(self, logits, targets, mu, log_sigma):
        """
        Compute VAE loss.
        
        Args:
            logits: Predicted classification logits [B, 250, 2, k_classes]
            targets: Target class indices [B, 250, 2]
            mu: Latent mean [B, d_latent]
            log_sigma: Latent log std [B, d_latent]
            
        Returns:
            dict: Dictionary containing total loss and components
        """
        batch_size = logits.shape[0]
        
        # Reconstruction loss (Cross-Entropy) for separate x, y coordinates
        # Split logits for x and y coordinates
        x_logits = logits[:, :, 0, :]  # [B, 250, k_classes]
        y_logits = logits[:, :, 1, :]  # [B, 250, k_classes]
        
        # Flatten targets for cross-entropy
        target_x = targets[:, :, 0].flatten()  # [B*250]
        target_y = targets[:, :, 1].flatten()  # [B*250]
        
        # Compute cross-entropy losses for coordinates
        x_loss = self.ce_loss(x_logits.reshape(-1, self.k_classes), target_x)
        y_loss = self.ce_loss(y_logits.reshape(-1, self.k_classes), target_y)
        
        # Total reconstruction loss
        recon_loss = x_loss + y_loss
        
        # KL divergence loss
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.sum(1 + 2*log_sigma - mu.pow(2) - (2*log_sigma).exp(), dim=1)
        
        if self.reduction == 'mean':
            kl_loss = kl_loss.mean()
        elif self.reduction == 'sum':
            kl_loss = kl_loss.sum()
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'recon_x_loss': x_loss,
            'recon_y_loss': y_loss,
            'kl_loss': kl_loss,
            'beta': self.beta
        }


class ClassificationLoss(nn.Module):
    """
    Pure classification loss for coordinate prediction.
    Can be used without VAE components for ablation studies.
    """
    
    def __init__(self, k_classes=3000, reduction='mean'):
        super().__init__()
        self.k_classes = k_classes
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(self, logits, targets):
        """
        Compute classification loss.
        
        Args:
            logits: Predicted classification logits [B, 250, 2, k_classes]
            targets: Target class indices [B, 250, 2]
            
        Returns:
            loss: Classification loss
        """
        # Split logits for x and y coordinates
        x_logits = logits[:, :, 0, :]  # [B, 250, k_classes]
        y_logits = logits[:, :, 1, :]  # [B, 250, k_classes]
        
        # Flatten targets for cross-entropy
        target_x = targets[:, :, 0].flatten()  # [B*250]
        target_y = targets[:, :, 1].flatten()  # [B*250]
        
        # Compute and combine losses
        x_loss = self.ce_loss(x_logits.reshape(-1, self.k_classes), target_x)
        y_loss = self.ce_loss(y_logits.reshape(-1, self.k_classes), target_y)
        
        return x_loss + y_loss


class BetaVAELoss(VAELoss):
    """
    Beta-VAE loss with configurable beta scheduling.
    """
    
    def __init__(self, k_classes=3000, beta_start=0.0, beta_end=1.0, 
                 beta_warmup_steps=1000, reduction='mean'):
        super().__init__(k_classes=k_classes, beta=beta_start, reduction=reduction)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_warmup_steps = beta_warmup_steps
        self.current_step = 0
    
    def update_beta(self, step):
        """Update beta value based on training step."""
        self.current_step = step
        if step < self.beta_warmup_steps:
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * (step / self.beta_warmup_steps)
        else:
            self.beta = self.beta_end
    
    def forward(self, logits, targets, mu, log_sigma):
        """Forward pass with current beta value."""
        result = super().forward(logits, targets, mu, log_sigma)
        result['beta'] = self.beta
        result['step'] = self.current_step
        return result


class KLAnnealingVAELoss(VAELoss):
    """
    VAE loss with KL annealing based on epochs (linear growth from 0).
    """
    
    def __init__(self, k_classes=3000, beta_max=1.0, anneal_epochs=10, reduction='mean'):
        super().__init__(k_classes=k_classes, beta=0.0, reduction=reduction)
        self.beta_max = beta_max
        self.anneal_epochs = anneal_epochs
        self.current_epoch = 0
    
    def update_beta(self, epoch):
        """Update beta value based on training epoch."""
        self.current_epoch = epoch
        if epoch < self.anneal_epochs:
            self.beta = (epoch / self.anneal_epochs) * self.beta_max
        else:
            self.beta = self.beta_max
    
    def forward(self, logits, targets, mu, log_sigma):
        """Forward pass with current beta value."""
        result = super().forward(logits, targets, mu, log_sigma)
        result['beta'] = self.beta
        result['epoch'] = self.current_epoch
        return result


def create_loss_function(loss_type='vae', **kwargs):
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss ('vae', 'classification', 'beta_vae', 'kl_anneal')
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function instance
    """
    if loss_type == 'vae':
        return VAELoss(**kwargs)
    elif loss_type == 'classification':
        return ClassificationLoss(**kwargs)
    elif loss_type == 'beta_vae':
        return BetaVAELoss(**kwargs)
    elif loss_type == 'kl_anneal':
        return KLAnnealingVAELoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")