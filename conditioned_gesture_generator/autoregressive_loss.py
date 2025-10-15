import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class FactorizedAutoregressiveLoss(nn.Module):
    """
    Loss function for factorized autoregressive gesture generation.
    Computes separate cross-entropy losses for x, y, and touch components,
    plus optional delta-consistency regularization on x and y.
    """

    def __init__(self, ignore_index: int = 0, delta_loss_weight: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.delta_loss_weight = delta_loss_weight

    def forward(self, model_output: Dict[str, torch.Tensor], ground_truth: torch.Tensor = None,
                action_sequence: torch.Tensor = None, model: torch.nn.Module = None) -> Dict[str, torch.Tensor]:
        # Extract components from model output
        x_logits = model_output["x_logits"]      # [B, L, x_classes]
        y_logits = model_output["y_logits"]      # [B, L, y_classes]
        touch_logits = model_output["touch_logits"]  # [B, L, touch_classes]

        target_x = model_output["target_x"]      # [B, L]
        target_y = model_output["target_y"]      # [B, L]
        target_touch = model_output["target_touch"]  # [B, L]

        B, L = target_x.shape
        Cx = x_logits.size(-1)
        Cy = y_logits.size(-1)

        # --- Mask BOS token after t=0 to prevent zero collapse ---
        # Create a copy to avoid modifying the input
        x_logits_masked = x_logits.clone()
        y_logits_masked = y_logits.clone()
        touch_logits_masked = touch_logits.clone()

        # Set BOS logits to -inf for all positions after t=0
        if L > 1:
            x_logits_masked[:, 1:, 0] = float('-inf')
            y_logits_masked[:, 1:, 0] = float('-inf')
            touch_logits_masked[:, 1:, 0] = float('-inf')

        # --- Cross-entropy losses ---
        loss_x = F.cross_entropy(
            x_logits_masked.view(-1, Cx),
            target_x.view(-1),
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        loss_y = F.cross_entropy(
            y_logits_masked.view(-1, Cy),
            target_y.view(-1),
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        loss_touch = F.cross_entropy(
            touch_logits_masked.view(-1, touch_logits_masked.size(-1)),
            target_touch.view(-1),
            ignore_index=self.ignore_index,
            reduction='mean'
        )

        total_loss = loss_x + loss_y + loss_touch

        # --- Delta consistency loss ---
        loss_delta = torch.tensor(0.0, device=x_logits.device)
        if self.delta_loss_weight > 0.0:
            # Differentiable expected index from softmax
            idx_x = torch.arange(Cx, device=x_logits.device).float()
            idx_y = torch.arange(Cy, device=y_logits.device).float()
            prob_x = F.softmax(x_logits_masked, dim=-1)
            prob_y = F.softmax(y_logits_masked, dim=-1)
            pred_x = (prob_x * idx_x).sum(dim=-1)  # [B, L]
            pred_y = (prob_y * idx_y).sum(dim=-1)  # [B, L]

            # Ground truth as float
            true_x = target_x.float()
            true_y = target_y.float()

            # Compute deltas
            delta_pred_x = (pred_x[:, 1:] - pred_x[:, :-1]) / (Cx - 1)
            delta_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]) / (Cy - 1)
            delta_true_x = (true_x[:, 1:] - true_x[:, :-1]) / (Cx - 1)
            delta_true_y = (true_y[:, 1:] - true_y[:, :-1]) / (Cy - 1)

            # MSE
            loss_delta_x = F.mse_loss(delta_pred_x, delta_true_x)
            loss_delta_y = F.mse_loss(delta_pred_y, delta_true_y)
            loss_delta = 0.5 * (loss_delta_x + loss_delta_y)

            # Add to total
            total_loss = total_loss + self.delta_loss_weight * loss_delta

        # --- Metrics ---
        with torch.no_grad():
            pred_x = torch.argmax(x_logits_masked, dim=-1)
            pred_y = torch.argmax(y_logits_masked, dim=-1)
            pred_touch = torch.argmax(touch_logits_masked, dim=-1)

            valid_mask_x = (target_x != self.ignore_index)
            valid_mask_y = (target_y != self.ignore_index)
            valid_mask_touch = (target_touch != self.ignore_index)

            acc_x = ((pred_x == target_x) & valid_mask_x).float().sum() / valid_mask_x.float().sum().clamp(min=1)
            acc_y = ((pred_y == target_y) & valid_mask_y).float().sum() / valid_mask_y.float().sum().clamp(min=1)
            acc_touch = ((pred_touch == target_touch) & valid_mask_touch).float().sum() / valid_mask_touch.float().sum().clamp(min=1)

            combined_correct = (pred_x == target_x) & (pred_y == target_y) & (pred_touch == target_touch)
            valid_mask_combined = valid_mask_x & valid_mask_y & valid_mask_touch
            acc_combined = (combined_correct & valid_mask_combined).float().sum() / valid_mask_combined.float().sum().clamp(min=1)

            perplexity_x = torch.exp(loss_x)
            perplexity_y = torch.exp(loss_y)
            perplexity_touch = torch.exp(loss_touch)
            perplexity_total = torch.exp(total_loss)

        return {
            "total_loss": total_loss,
            "loss_x": loss_x,
            "loss_y": loss_y,
            "loss_touch": loss_touch,
            "loss_delta": loss_delta,
            "accuracy_x": acc_x,
            "accuracy_y": acc_y,
            "accuracy_touch": acc_touch,
            "accuracy_combined": acc_combined,
            "perplexity_x": perplexity_x,
            "perplexity_y": perplexity_y,
            "perplexity_touch": perplexity_touch,
            "perplexity_total": perplexity_total,
        }

