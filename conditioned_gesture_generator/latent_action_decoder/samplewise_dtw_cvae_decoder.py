from typing import Any, Dict
import torch

from .dtw_cvae_decoder import DTWCVAEDecoder
from ..losses.samplewise_dtw_cvae_loss import SamplewiseDTWCVAELoss

class SamplewiseDTWCVAEDecoder(DTWCVAEDecoder):
    """
    A variant of the DTWCVAEDecoder that uses a sample-wise Soft-DTW
    reconstruction loss instead of the original batch-wise NAG loss.

    This is expected to provide a more stable training signal. The model
    architecture is identical to the parent DTWCVAEDecoder; only the
    loss function is different.
    """
    def get_loss_function(self):
        """
        Overrides the base method to return the sample-wise loss function.
        """
        # The loss_cfg is passed during __init__ of the parent class
        return SamplewiseDTWCVAELoss(**self.loss_cfg, use_cuda=torch.cuda.is_available())

    def get_config(self) -> Dict[str, Any]:
        """
        Ensure the config reflects this specific model variant.
        """
        config = super().get_config()
        config['model_class'] = 'SamplewiseDTWCVAEDecoder' # Add identifier
        return config
