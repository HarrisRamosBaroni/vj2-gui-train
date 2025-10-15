import torch
import torch.nn as nn
from typing import Dict

# Correcting the import path based on file hierarchy
from ..latent_action_decoder.soft_dtw_cuda import SoftDTW, manhattan_only_dist_func

class SamplewiseDTWCVAELoss(nn.Module):
    """
    Loss function for a CVAE model that uses a sample-wise Soft-DTW reconstruction
    loss. This provides a more direct and stable training signal than distribution-
    matching losses like NAG.

    The total loss combines:
    1. Sample-wise Soft-DTW reconstruction loss.
    2. KL divergence on the style latent space.
    3. Auxiliary loss for predicting the number of touch transitions.
    4. Regularizer to enforce consistency between predicted touch and its transition count.
    """
    def __init__(self,
                 w_kl: float = 1.0,
                 w_recon: float = 1.0,
                 w_aux: float = 0.1,
                 w_trans: float = 0.5,
                 kl_free_bits: float = 0.5,
                 dist_gamma: float = 1.0,
                 focal_gamma: float = 2.0,
                 focal_alpha: float = 0.6,
                 sdtw_gamma: float = 0.1,
                 use_cuda: bool = True):
        super().__init__()
        self.w_kl = w_kl
        self.w_recon = w_recon
        self.w_aux = w_aux
        self.w_trans = w_trans
        self.kl_free_bits = kl_free_bits
        self.dist_gamma = dist_gamma
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        self.sdtw = SoftDTW(
            use_cuda=use_cuda,
            gamma=sdtw_gamma,
            normalize=True,  # Ensure non-negative distance
            bandwidth=150,
            dist_func=self.pointwise_dist_func
        )

    def pointwise_dist_func(self, x, y):
        """Point-wise distance for Soft-DTW."""
        # DEBUG: Using ultra-simple distance function for debugging.
        return manhattan_only_dist_func(x, y)

    def _calculate_samplewise_sdtw_loss(self, gen_batch, real_batch):
        """
        Calculates a sample-wise Soft-DTW loss between the generated batch
        and the real batch.
        """
        sdtw_matrix = self.sdtw(gen_batch, real_batch)
        # The diagonal of the resulting matrix contains the sdtw
        # between each generated sample and its corresponding ground truth.
        recon_loss = torch.mean(torch.diag(sdtw_matrix))
        return recon_loss

    def _calculate_kl_loss(self, mu, logvar):
        """KL divergence with free bits."""
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(torch.clamp(kl_div - self.kl_free_bits, min=0.))
        return kl_loss

    def _calculate_transition_count(self, touch_signal, soft=True):
        """Calculates the number of touch state transitions."""
        if soft:
            sharpened = torch.sigmoid((touch_signal - 0.5) * 10)
            return torch.sum(torch.abs(sharpened[:, 1:] - sharpened[:, :-1]), dim=1)
        else:
            return torch.sum(torch.abs(touch_signal[:, 1:] - touch_signal[:, :-1]), dim=1)

    def forward(self, model_output: Dict[str, torch.Tensor],
                ground_truth: torch.Tensor) -> Dict[str, torch.Tensor]:

        pred_traj = model_output["action_trajectory"]
        style_mu = model_output["style_mu"]
        style_logvar = model_output["style_logvar"]
        pred_trans_count = model_output["predicted_transition_count"]

        # --- 1. Sample-wise SDTW Reconstruction Loss ---
        recon_loss = self._calculate_samplewise_sdtw_loss(pred_traj, ground_truth)

        # --- 2. KL Divergence Loss ---
        kl_loss = self._calculate_kl_loss(style_mu, style_logvar)

        # --- 3. Auxiliary Transition Count Loss ---
        gt_trans_count = self._calculate_transition_count(ground_truth[..., 2], soft=False)
        aux_loss = nn.functional.mse_loss(pred_trans_count.squeeze(-1), gt_trans_count.float())

        # --- 4. Predicted Transition Count Regularizer ---
        pred_touch_trans_soft = self._calculate_transition_count(pred_traj[..., 2], soft=True)
        trans_reg_loss = nn.functional.mse_loss(pred_touch_trans_soft, gt_trans_count.float())

        # --- Total Loss ---
        total_loss = (self.w_recon * recon_loss +
                      self.w_kl * kl_loss +
                      self.w_aux * aux_loss +
                      self.w_trans * trans_reg_loss)

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "aux_transition_loss": aux_loss,
            "pred_transition_reg_loss": trans_reg_loss,
        }
