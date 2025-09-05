import torch
import torch.nn as nn
import torch.nn.functional as F

class MAELoss(nn.Module):
    """Mean absolute error loss wrapper (L1)."""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        # pred is a tensor; target is a tensor of the same shape
        loss = torch.abs(pred - target)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class LaplaceNLL(nn.Module):
    """
    Laplace Negative Log-Likelihood loss.

    Expects model output to be a tuple: (mu, b)
    where:
      - mu: predicted location, same shape as target
      - b: predicted scale (positive). Prefer using softplus in the model to ensure positivity.
    Loss per element: |target - mu| / b + log(b)
    """
    def __init__(self, eps=1e-6, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        if not isinstance(pred, (tuple, list)) or len(pred) != 2:
            raise ValueError("LaplaceNLL expects model output to be (mu, b)")
        mu, b = pred
        # Ensure numerical stability
        b = b.clamp(min=self.eps)
        loss = torch.abs(target - mu) / b + torch.log(b)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss