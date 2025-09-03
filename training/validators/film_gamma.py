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
        all_betas = []
        
        try:
            # Get a sample batch to generate FiLM parameters
            try:
                sample_embeddings, sample_actions = next(iter(validation_loader))
                sample_embeddings = sample_embeddings.to(device)
                sample_actions = sample_actions.to(device, dtype=torch.float)
            except StopIteration:
                # Loader is empty, can't validate
                return {}

            # The FiLM predictor requires the original 4D action tensor
            formatted_actions = model.actions_formatter(sample_actions)
            
            # Pass through the action encoder and FiLM heads to get parameters
            action_context = model.action_encoder(formatted_actions)
            film_params_per_layer = model.film_heads(action_context, model.film_clamp_gamma_alpha)
            
            # film_params_per_layer is a DICTIONARY: {0: {'gamma_att':...}, 1: ...}
            for layer_index in film_params_per_layer:
                layer_params = film_params_per_layer[layer_index]
                all_gammas.extend(layer_params['gamma_att'].cpu().numpy().flatten())
                all_gammas.extend(layer_params['gamma_ff'].cpu().numpy().flatten())
                all_betas.extend(layer_params['beta_att'].cpu().numpy().flatten())
                all_betas.extend(layer_params['beta_ff'].cpu().numpy().flatten())

            if not all_gammas:
                return {}
                
            all_gammas = np.array(all_gammas)
            all_betas = np.array(all_betas)
            
            # Create Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            ax1.hist(all_gammas, bins=100, alpha=0.75, density=True)
            ax1.set_xlabel("Gamma Parameter Values")
            ax1.set_ylabel("Density")
            ax1.set_title("Distribution of FiLM Gamma Parameters")
            ax1.grid(True)
            
            ax2.hist(all_betas, bins=100, alpha=0.75, density=True, color='orange')
            ax2.set_xlabel("Beta Parameter Values")
            ax2.set_ylabel("Density")
            ax2.set_title("Distribution of FiLM Beta Parameters")
            ax2.grid(True)
            
            fig.tight_layout()
            
            # Log to wandb
            metrics = {
                "validation/film_param_distribution": wandb.Image(fig)
            }
            
            plt.close(fig)
            return metrics
        finally:
            # Crucially, ensure the model is always returned to train mode
            model.train()