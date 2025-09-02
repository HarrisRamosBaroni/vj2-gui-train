import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np

from .base import BaseValidator
from gui_world_model.predictor_film import VJ2GUIPredictorFiLM

class FilmGammaValidator(BaseValidator):
    @torch.no_grad()
    def run(self, model: torch.nn.Module, validation_loader: torch.utils.data.DataLoader, device: torch.device, global_step: int):
        if not isinstance(model, VJ2GUIPredictorFiLM):
            return

        model.eval()
        
        all_gammas = []
        
        # film_heads is a ModuleList of nn.Linear layers
        for head in model.film_heads.film_generators:
            # Each head is a linear layer that outputs concatenated gammas and betas
            # The first half of the output dimension is gammas
            output_dim = head.out_features
            gamma_dim = output_dim // 2
            
            # Extract the weights corresponding to the gamma outputs
            gamma_weights = head.weight.data[:gamma_dim, :]
            all_gammas.append(gamma_weights.cpu().numpy().flatten())
            
        if not all_gammas:
            model.train()
            return
            
        all_gammas = np.concatenate(all_gammas)
        
        # Create Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_gammas, bins=100, alpha=0.75, density=True)
        ax.set_xlabel("Gamma Parameter Values")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of FiLM Gamma Parameters")
        ax.grid(True)
        
        # Log to wandb
        wandb.log({
            "validation/film_gamma_distribution": wandb.Image(fig)
        }, step=global_step)
        
        plt.close(fig)
        model.train()