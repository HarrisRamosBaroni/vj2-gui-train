from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseDecoderLoss(nn.Module):
    """
    Abstract base class for decoder loss functions.
    Ensures a consistent interface for the trainer.
    """
    def forward(self, model_output: Dict[str, torch.Tensor], ground_truth: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculates losses based on model output and ground truth.

        Args:
            model_output (Dict[str, torch.Tensor]): The raw output from the decoder model.
            ground_truth (torch.Tensor): The ground truth action trajectory, shape (B, T, 3).

        Returns:
            A dictionary of all loss components, which must include 'total_loss'.
            e.g., {"total_loss": ..., "recon_loss": ..., "touch_bce_loss": ...}
        """
        raise NotImplementedError

class HybridGatedLoss(BaseDecoderLoss):
    """
    Computes the composite loss for the HybridGatedDecoder.
    - BCE loss for the touch state (classification).
    - Masked Huber loss for the (x, y) coordinates (regression).
    """
    def __init__(self, xy_loss_weight: float = 1.0, touch_pos_weight: float = 1.0):
        super().__init__()
        self.xy_loss_weight = xy_loss_weight
        self.huber_loss = nn.HuberLoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(touch_pos_weight))

    def forward(self, model_output: Dict[str, torch.Tensor], ground_truth: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Unpack model predictions
        touch_logits = model_output["touch_logits"] # Shape (B, T, 1)
        xy_pred = model_output["xy_pred"]           # Shape (B, T, 2)

        # Unpack ground truth
        gt_xy = ground_truth[..., :2]
        gt_touch = ground_truth[..., 2].unsqueeze(-1) # Ensure shape (B, T, 1)

        # 1. Touch Loss (BCE)
        touch_loss = self.bce_loss(touch_logits, gt_touch)

        # 2. Masked Coordinate Loss (Huber)
        mask = (gt_touch > 0.5).float()
        
        # Calculate Huber loss for each element, then apply mask
        unmasked_xy_loss = self.huber_loss(xy_pred, gt_xy) # Shape (B, T, 2)
        masked_xy_loss = unmasked_xy_loss * mask
        
        # Normalize only by the number of active timesteps to avoid skew
        num_active_timesteps = mask.sum()
        if num_active_timesteps > 0:
            xy_loss = masked_xy_loss.sum() / num_active_timesteps
        else:
            xy_loss = torch.tensor(0.0, device=xy_pred.device) # No touch in batch

        # 3. Total Loss
        total_loss = touch_loss + self.xy_loss_weight * xy_loss

        return {
            "total_loss": total_loss,
            "touch_bce_loss": touch_loss,
            "xy_huber_loss": xy_loss,
        }

class CVAELoss(BaseDecoderLoss):
    """
    Computes the loss for the ActionCVAE model.
    This is the evidence lower bound (ELBO), which consists of:
    1. Reconstruction Loss: How well the decoder reconstructs the action.
    2. KL Divergence: A regularizer that pushes the posterior distribution
       q(z_style | A, z, s) to be close to the prior p(z_style) = N(0, I).
    """
    def __init__(self, xy_loss_weight: float = 1.0, touch_pos_weight: float = 1.0, kl_weight: float = 1.0):
        super().__init__()
        self.xy_loss_weight = xy_loss_weight
        self.kl_weight = kl_weight
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(touch_pos_weight))

    def forward(self, model_output: Dict[str, torch.Tensor], ground_truth: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Unpack model predictions
        recon_outputs = model_output["recon_A_outputs"]
        mu = model_output["mu"]
        logvar = model_output["logvar"]

        touch_logits = recon_outputs["touch_logits"] # Shape (B, T, 1)
        xy_pred = recon_outputs["xy_pred"]           # Shape (B, T, 2)

        # Unpack ground truth
        gt_xy = ground_truth[..., :2]
        gt_touch = ground_truth[..., 2].unsqueeze(-1) # Ensure shape (B, T, 1)

        # 1. Reconstruction Loss
        # 1a. Touch Loss (BCE)
        touch_loss = self.bce_loss(touch_logits, gt_touch)

        # 1b. Masked Coordinate Loss (MSE)
        mask = (gt_touch > 0.5).float()
        unmasked_xy_loss = self.mse_loss(xy_pred, gt_xy) # Shape (B, T, 2)
        masked_xy_loss = unmasked_xy_loss * mask

        num_active_timesteps = mask.sum()
        if num_active_timesteps > 0:
            xy_loss = masked_xy_loss.sum() / num_active_timesteps
        else:
            xy_loss = torch.tensor(0.0, device=xy_pred.device)

        recon_loss = touch_loss + self.xy_loss_weight * xy_loss

        # 2. KL Divergence
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalize by batch size
        kl_div /= mu.shape[0]

        # 3. Total Loss (ELBO)
        total_loss = recon_loss + self.kl_weight * kl_div

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "touch_bce_loss": touch_loss,
            "xy_mse_loss": xy_loss,
            "kl_divergence": kl_div,
        }


class QuantizingLoss(BaseDecoderLoss):
    """
    Computes the loss for a quantizing (classification-based) decoder.
    - Cross-Entropy loss for the quantized (x, y) coordinates.
    """
    def __init__(self, k_classes: int):
        super().__init__()
        self.k_classes = k_classes
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, model_output: Dict[str, torch.Tensor], ground_truth: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Unpack model predictions: shape (B, T, 2, K)
        xy_logits = model_output["xy_logits"]
        B, T, _, K = xy_logits.shape

        # Unpack and quantize ground truth
        gt_xy = ground_truth[..., :2] # Shape (B, T, 2)
        # Assuming values are in [-1, 1], scale to [0, K-1]
        # gt_xy_quantized = ((gt_xy * 0.5 + 0.5) * (self.k_classes - 1)).long()
        # Scale from [0, 1] to [0, K-1]
        gt_xy_quantized = (gt_xy * (self.k_classes - 1)).long()
        gt_xy_quantized = torch.clamp(gt_xy_quantized, 0, self.k_classes - 1)

        # Reshape for CrossEntropyLoss which expects (N, C, ...)
        # We treat each coordinate (x and y) as a separate classification task
        # a) Permute to (B, 2, K, T)
        xy_logits_permuted = xy_logits.permute(0, 2, 3, 1)
        # b) Reshape to (B * 2 * T, K)
        xy_logits_reshaped = xy_logits_permuted.reshape(-1, K)
        
        # Reshape ground truth to (B * T * 2) -> (N)
        gt_xy_quantized_reshaped = gt_xy_quantized.permute(0, 2, 1).reshape(-1)

        total_loss = self.ce_loss(xy_logits_reshaped, gt_xy_quantized_reshaped)

        return {
            "total_loss": total_loss,
            "xy_ce_loss": total_loss,
        }