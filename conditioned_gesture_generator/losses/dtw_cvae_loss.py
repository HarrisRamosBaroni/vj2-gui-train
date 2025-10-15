
import torch
import torch.nn as nn
from typing import Dict

# Correcting the import path based on file hierarchy
from ..latent_action_decoder.soft_dtw_cuda import SoftDTW, focal_manhattan_focal_dist_func

class DTWCVAELoss(nn.Module):
    """
    Loss function for the DTW-CVAE model, combining a distribution-matching
    reconstruction loss (NAG), KL divergence, and auxiliary regularizers.
    """
    def __init__(self,
                 w_kl: float = 1.0,
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
        self.w_aux = w_aux
        self.w_trans = w_trans
        self.kl_free_bits = kl_free_bits
        self.dist_gamma = dist_gamma
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # Soft-DTW setup for Hausdorff distance
        # The proposal specifies a Sakoe-Chiba bandwidth of 150
        self.sdtw = SoftDTW(
            use_cuda=use_cuda,
            gamma=sdtw_gamma,
            bandwidth=150,
            dist_func=self.pointwise_dist_func
        )

    def pointwise_dist_func(self, x, y):
        """Point-wise distance for Soft-DTW as defined in the proposal."""
        return focal_manhattan_focal_dist_func(
            x, y,
            dist_gamma=self.dist_gamma,
            focal_gamma=self.focal_gamma,
            focal_alpha=self.focal_alpha
        )

    def _average_hausdorff_distance(self, gen_batch, real_batch):
        """
        Calculates the Average Hausdorff Distance between two batches of sequences
        using Soft-DTW as the underlying metric.
        """
        # Term 1: E_{x'~P_g}[min_{x~P_r} d(x', x)]
        term1 = self.sdtw(gen_batch, real_batch).mean()

        # Term 2: E_{x~P_r}[min_{x'~P_g} d(x, x')]
        term2 = self.sdtw(real_batch, gen_batch).mean()

        return term1 + term2

    def _calculate_variation_term(self, gen_batch, real_batch):
        """
        Calculates the variation term of the NAG loss.
        | E_{x1,x2~P_g}[d(x1,x2)] - E_{x1,x2~P_r}[d(x1,x2)] |
        """
        # To avoid redundant computations, we can shuffle the batches
        # and compute the distance to a different sample in the same batch.
        
        # Shuffle generated batch
        gen_batch_shuffled = gen_batch[torch.randperm(gen_batch.size(0))]
        # Shuffle real batch
        real_batch_shuffled = real_batch[torch.randperm(real_batch.size(0))]

        # Intra-batch distances
        d_gen = self.sdtw(gen_batch, gen_batch_shuffled).mean()
        d_real = self.sdtw(real_batch, real_batch_shuffled).mean()

        return torch.abs(d_gen - d_real)

    def _calculate_nag_loss(self, gen_batch, real_batch):
        """Non-adversarial generation loss."""
        similarity_term = self._average_hausdorff_distance(gen_batch, real_batch)
        variation_term = self._calculate_variation_term(gen_batch, real_batch)
        return similarity_term + variation_term

    def _calculate_kl_loss(self, mu, logvar):
        """KL divergence with free bits."""
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        # Apply free bits
        kl_loss = torch.mean(torch.clamp(kl_div - self.kl_free_bits, min=0.))
        return kl_loss

    def _calculate_transition_count(self, touch_signal, soft=True):
        """
        Calculates the number of touch state transitions.
        touch_signal: (B, T) tensor
        """
        if soft:
            # Differentiable approximation for predicted probabilities
            # Scale, sharpen with sigmoid, then sum absolute differences
            sharpened = torch.sigmoid((touch_signal - 0.5) * 10)
            return torch.sum(torch.abs(sharpened[:, 1:] - sharpened[:, :-1]), dim=1)
        else:
            # Hard count for ground truth
            return torch.sum(torch.abs(touch_signal[:, 1:] - touch_signal[:, :-1]), dim=1)

    def forward(self, model_output: Dict[str, torch.Tensor],
                ground_truth: torch.Tensor) -> Dict[str, torch.Tensor]:

        # Unpack model outputs
        pred_traj = model_output["action_trajectory"]
        style_mu = model_output["style_mu"]
        style_logvar = model_output["style_logvar"]
        pred_trans_count = model_output["predicted_transition_count"]

        # --- 1. NAG Reconstruction Loss ---
        nag_loss = self._calculate_nag_loss(pred_traj, ground_truth)

        # --- 2. KL Divergence Loss ---
        kl_loss = self._calculate_kl_loss(style_mu, style_logvar)

        # --- 3. Auxiliary Transition Count Loss ---
        gt_trans_count = self._calculate_transition_count(ground_truth[..., 2], soft=False)
        aux_loss = nn.functional.mse_loss(pred_trans_count.squeeze(-1), gt_trans_count.float())

        # --- 4. Predicted Transition Count Regularizer ---
        pred_touch_trans_soft = self._calculate_transition_count(pred_traj[..., 2], soft=True)
        trans_reg_loss = nn.functional.mse_loss(pred_touch_trans_soft, gt_trans_count.float())

        # --- Total Loss ---
        total_loss = (nag_loss +
                      self.w_kl * kl_loss +
                      self.w_aux * aux_loss +
                      self.w_trans * trans_reg_loss)

        return {
            "total_loss": total_loss,
            "nag_loss": nag_loss,
            "kl_loss": kl_loss,
            "aux_transition_loss": aux_loss,
            "pred_transition_reg_loss": trans_reg_loss,
        }
