import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import random

from .base import BaseValidator

def get_random_sample_from_loader(loader, device):
    """Gets a random, single (embedding, action) pair from the validation loader."""
    # This is less efficient than getting a batch and picking one, but simple
    # to implement. For validation, this is acceptable.
    dataset = loader.dataset
    embeddings, actions = dataset[random.randint(0, len(dataset) - 1)]
    
    # The dataloader doesn't apply the layer norm, so we do it here
    z_all = F.layer_norm(embeddings, (embeddings.size(-1),))
    
    # Select random state and action from the trajectory
    z = z_all[random.randint(0, z_all.shape[0] - 1)]
    a = actions[random.randint(0, actions.shape[0] - 1)].flatten()
    return z.to(device), a.to(device)


class InputSensitivityValidator(BaseValidator):
    def __init__(self, frequency: int, num_initial_samples: int = 5, num_perturbations: int = 100, **kwargs):
        super().__init__(frequency)
        self.num_initial_samples = num_initial_samples
        self.num_perturbations = num_perturbations

    @torch.no_grad()
    def run(self, model: torch.nn.Module, validation_loader: torch.utils.data.DataLoader, device: torch.device, global_step: int):
        model.eval()
        
        results = []
        for i in range(self.num_initial_samples):
            # For each sample, we fix an action and vary the state
            _, a_i = get_random_sample_from_loader(validation_loader, device)
            
            # Prepare a_i for batching
            # Predictor expects [B, T, D_action], so we add T=1 dimension
            a_batch = a_i.unsqueeze(0).unsqueeze(0).repeat(self.num_perturbations * 2, 1, 1)

            states_to_batch = []
            for _ in range(self.num_perturbations):
                z1, _ = get_random_sample_from_loader(validation_loader, device)
                z2, _ = get_random_sample_from_loader(validation_loader, device)
                states_to_batch.extend([z1, z2])
            
            # Predictor expects [B, T, N, D], so we add T=1 dimension
            z_batch = torch.stack(states_to_batch).unsqueeze(1)
            
            preds = model(z_batch, a_batch)
            
            for k in range(0, len(states_to_batch), 2):
                z1 = states_to_batch[k]
                z2 = states_to_batch[k+1]
                pred1 = preds[k].squeeze(0)
                pred2 = preds[k+1].squeeze(0)

                delta_in = torch.mean(torch.abs(z2 - z1))
                delta_out = torch.mean(torch.abs(pred2 - pred1))
                results.append((delta_in.item(), delta_out.item()))

        delta_ins, delta_outs = zip(*results)

        # Create Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(delta_ins, delta_outs, alpha=0.5)
        ax.set_xlabel("Input State Delta (mean L1)")
        ax.set_ylabel("Output State Delta (mean L1)")
        ax.set_title("Predictor Sensitivity to State Perturbation")
        ax.grid(True)
        
        # Line of best fit
        slope, intercept, r_value, _, _ = stats.linregress(delta_ins, delta_outs)
        line = [slope * x + intercept for x in delta_ins]
        ax.plot(delta_ins, line, 'r', label=f'y={slope:.2f}x+{intercept:.2f}\n(R²={r_value**2:.2f})')
        
        # y=x line
        min_val = min(min(delta_ins), min(delta_outs))
        max_val = max(max(delta_ins), max(delta_outs))
        ax.plot([min_val, max_val], [min_val, max_val], 'g--', label='y=x')

        ax.legend()
        
        # Log to wandb
        wandb.log({
            "validation/state_sensitivity_plot": wandb.Image(fig),
            "validation/state_sensitivity_slope": slope,
            "validation/state_sensitivity_r_squared": r_value**2,
        }, step=global_step)
        
        plt.close(fig)
        model.train()