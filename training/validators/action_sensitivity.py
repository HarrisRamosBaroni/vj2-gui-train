import torch
import torch.nn.functional as F
from logging import getLogger
import wandb
import matplotlib.pyplot as plt
import numpy as np
import random

logger = getLogger(__name__)

from .base import BaseValidator

def get_random_sample_from_loader(loader, device, keep_action_shape=True):
    """
    Gets a random, single (embedding, action) pair from the validation loader.
    If keep_action_shape is True, returns the action with its original shape.
    """
    dataset = loader.dataset
    embeddings, actions = dataset[random.randint(0, len(dataset) - 1)]
    
    # The dataloader doesn't apply the layer norm, so we do it here
    z_all = F.layer_norm(embeddings, (embeddings.size(-1),))
    
    # Select random state and action from the trajectory
    z = z_all[random.randint(0, z_all.shape[0] - 1)]
    a = actions[random.randint(0, actions.shape[0] - 1)]
    
    if not keep_action_shape:
        a = a.flatten()
        
    return z.to(device), a.to(device)


class ActionSensitivityValidator(BaseValidator):
    """
    Analyzes the model's sensitivity to changes in the action input while keeping the
    state input fixed. It helps understand if the model is appropriately conditioned on the action.
    """
    def __init__(self, frequency: int, num_initial_states: int = 5, num_perturbations: int = 100, **kwargs):
        super().__init__(frequency)
        self.num_initial_states = num_initial_states
        self.num_perturbations = num_perturbations

    @torch.no_grad()
    def run(self, model: torch.nn.Module, validation_loader: torch.utils.data.DataLoader, device: torch.device, global_step: int):
        model.eval()
        
        try:
            all_deltas = []

            for i in range(self.num_initial_states):
                z_i, _ = get_random_sample_from_loader(validation_loader, device, keep_action_shape=True)
                
                # 1. Get a fixed anchor prediction
                _, a_anchor = get_random_sample_from_loader(validation_loader, device, keep_action_shape=True)
                
                if hasattr(model, 'actions_formatter'):
                    a_anchor_formatted = model.actions_formatter(a_anchor.unsqueeze(0).unsqueeze(0))
                    z_pred_anchor = model(z_i.unsqueeze(0).unsqueeze(0), a_anchor_formatted).squeeze(0).squeeze(0)
                else:
                    a_anchor_flat = a_anchor.flatten().unsqueeze(0).unsqueeze(0)
                    z_pred_anchor = model(z_i.unsqueeze(0).unsqueeze(0), a_anchor_flat).squeeze(0).squeeze(0)


                # 2. Get predictions for a batch of other actions
                actions_to_batch = []
                for _ in range(self.num_perturbations):
                    _, a_j = get_random_sample_from_loader(validation_loader, device, keep_action_shape=True)
                    actions_to_batch.append(a_j)
                
                a_batch_unformatted = torch.stack(actions_to_batch).unsqueeze(1)

                if hasattr(model, 'actions_formatter'):
                    a_batch = model.actions_formatter(a_batch_unformatted)
                else:
                    B, T, L, D_a = a_batch_unformatted.shape
                    a_batch = a_batch_unformatted.view(B, T, -1)
                
                z_batch = z_i.unsqueeze(0).unsqueeze(0).repeat(self.num_perturbations, 1, 1, 1)

                preds = model(z_batch, a_batch)
                
                # 3. Calculate distance between each prediction and the anchor
                deltas = torch.mean(torch.abs(preds.squeeze(1) - z_pred_anchor.unsqueeze(0)), dim=(-2,-1))
                all_deltas.extend(deltas.cpu().numpy())

            # Create Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(all_deltas, bins=100, alpha=0.7)
            ax.set_xlabel("Mean L1 Distance from Anchor Prediction")
            ax.set_ylabel("Frequency")
            ax.set_title("Aggregate Prediction Cluster Spread (Action Sensitivity)")
            ax.grid(True)
            
            # Log to wandb
            logger.info(f"ActionSensitivityValidator: logging for {global_step=}")
            
            metrics = {
                "validation/action_sensitivity_plot": wandb.Image(fig),
                "validation/action_sensitivity_mean_delta": np.mean(all_deltas),
                "validation/action_sensitivity_median_delta": np.median(all_deltas),
                "validation/action_sensitivity_std_delta": np.std(all_deltas),
            }
            
            plt.close(fig)
            return metrics
        finally:
            model.train()