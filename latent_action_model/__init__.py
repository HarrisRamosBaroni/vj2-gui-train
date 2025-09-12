"""
Latent Action Model (LAM) - A VAE for learning action representations from latent sequences.
"""

from .modules import LAMEncoder, LAMDecoder, TransformerBlock
from .vae import LatentActionVAE, load_model_from_config
from .dataloader import LAMDataset, create_dataloaders
from .training import LAMTrainer, train_with_variable_context

__all__ = [
    'LAMEncoder', 
    'LAMDecoder', 
    'TransformerBlock',
    'LatentActionVAE',
    'load_model_from_config',
    'LAMDataset', 
    'create_dataloaders',
    'LAMTrainer', 
    'train_with_variable_context'
]