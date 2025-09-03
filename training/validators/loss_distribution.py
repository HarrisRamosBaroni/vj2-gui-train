import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
import numpy as np

from .base import BaseValidator
from config import ROLLOUT_HORIZON

class LossDistributionValidator(BaseValidator):
    """
    Validator that computes and logs histograms of per-sample teacher-forcing (jloss)
    and rollout (sloss) losses over a subset of the validation set.
    """

    def __init__(self, frequency: int = 1, max_batches: int = 5, loss_exp: int = 1, **kwargs):
        """
        Args:
            frequency: Run every `frequency` validation cycles.
            max_batches: Number of validation batches to scan (to bound runtime).
            loss_exp: Exponent for loss (1 => L1, 2 => L2). Matches trainer.loss_exp.
        """
        super().__init__(frequency)
        self.max_batches = max_batches
        self.loss_exp = loss_exp
        # rollout horizon fetched from global config
        self.rollout_horizon = ROLLOUT_HORIZON

    @torch.no_grad()
    def run(self, model: torch.nn.Module, validation_loader: torch.utils.data.DataLoader, device: torch.device, global_step: int):
        model.eval()

        try:
            jloss_vals = []
            sloss_vals = []

            batches_processed = 0
            for sample in validation_loader:
                embeddings, actions = sample
                embeddings = embeddings.to(device)
                actions = actions.to(device, dtype=torch.float)

                # Layer-normalize embeddings (same as trainer)
                z_all = F.layer_norm(embeddings, (embeddings.size(-1),))
                h_all = z_all  # target

                # Format actions consistent with trainer
                if hasattr(model, 'actions_formatter'):
                    formatted_actions = model.actions_formatter(actions)
                else:
                    B, T_seq, _, _ = actions.shape
                    formatted_actions = actions.view(B, T_seq, -1)

                B = z_all.size(0)

                # --- Teacher forcing predictions (z_tf) ---
                # z_tf should predict h_all[:, 1:] from z_all[:, :-1] and formatted_actions
                try:
                    z_tf = model(z_all[:, :-1], formatted_actions)
                except Exception:
                    # If model fails on this batch (shape mismatch), skip it.
                    batches_processed += 1
                    if batches_processed >= self.max_batches:
                        break
                    continue

                if z_tf.shape[0] != B:
                    # Unexpected shape, skip
                    batches_processed += 1
                    if batches_processed >= self.max_batches:
                        break
                    continue

                # Per-transition teacher-forcing loss: mean over tokens, channels
                # z_tf and h_all[:,1:] shapes: [B, T-1, N, D]
                jloss_per_transition = torch.mean(torch.abs(z_tf - h_all[:, 1:]) ** self.loss_exp, dim=(-2,-1)) / float(self.loss_exp)
                # Reshape from [B, T-1] to a flat list of losses
                jloss_vals.extend(jloss_per_transition.flatten().cpu().numpy().tolist())

                # --- Rollout predictions (autoregressive) ---
                # Start from z_all[:, 0] and iteratively predict for rollout_horizon steps
                z_rollout = z_all[:, 0].unsqueeze(1)  # [B, 1, N, D]
                for i in range(self.rollout_horizon):
                    # Consistent with trainer: slice the action for the current timestep
                    # This works for both 3D [B,T,A] and 4D [B,T,L,3] action tensors
                    a = formatted_actions[:, i].unsqueeze(1)
                    z_rollout = model(z_rollout, a)

                # z_rollout is [B, 1, N, D] -> squeeze time dim
                z_rollout_final = z_rollout.squeeze(1)
                # target at rollout horizon
                if self.rollout_horizon < h_all.size(1):
                    target = h_all[:, self.rollout_horizon]
                    # Per-sample rollout loss: mean over tokens and channels
                    sloss_per_sample = torch.mean(torch.abs(z_rollout_final - target) ** self.loss_exp, dim=(1,2)) / float(self.loss_exp)
                    sloss_vals.extend(sloss_per_sample.cpu().numpy().tolist())

                batches_processed += 1
                if batches_processed >= self.max_batches:
                    break

            # If no values collected, skip logging
            if len(jloss_vals) == 0 and len(sloss_vals) == 0:
                return {}
        finally:
            model.train()

        # Plot histograms
        figs = {}
        if len(jloss_vals) > 0:
            fig_j, axj = plt.subplots(figsize=(8, 5))
            axj.hist(jloss_vals, bins=100, alpha=0.8)
            axj.set_title("Teacher-Forcing Loss (jloss) Distribution")
            axj.set_xlabel("Loss")
            axj.set_ylabel("Frequency")
            axj.grid(True)
            figs["validation/jloss_hist"] = fig_j

        if len(sloss_vals) > 0:
            fig_s, axs = plt.subplots(figsize=(8, 5))
            axs.hist(sloss_vals, bins=100, alpha=0.8, color='orange')
            axs.set_title("Rollout Loss (sloss) Distribution")
            axs.set_xlabel("Loss")
            axs.set_ylabel("Frequency")
            axs.grid(True)
            figs["validation/sloss_hist"] = fig_s

        # Log to wandb: both images and summary stats
        log_dict = {}
        if len(jloss_vals) > 0:
            log_dict["validation/jloss_mean"] = float(np.mean(jloss_vals))
            log_dict["validation/jloss_median"] = float(np.median(jloss_vals))
            log_dict["validation/jloss_std"] = float(np.std(jloss_vals))
            # attach image
            log_dict["validation/jloss_hist_image"] = wandb.Image(figs["validation/jloss_hist"])

        if len(sloss_vals) > 0:
            log_dict["validation/sloss_mean"] = float(np.mean(sloss_vals))
            log_dict["validation/sloss_median"] = float(np.median(sloss_vals))
            log_dict["validation/sloss_std"] = float(np.std(sloss_vals))
            log_dict["validation/sloss_hist_image"] = wandb.Image(figs["validation/sloss_hist"])

        # Close figures
        for f in figs.values():
            plt.close(f)
            
        return log_dict