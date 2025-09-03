from abc import ABC, abstractmethod
import torch

class BaseValidator(ABC):
    """
    Abstract base class for all validation modules.
    """
    def __init__(self, frequency: int, **kwargs):
        if frequency < 1:
            raise ValueError("Validator frequency must be 1 or greater.")
        self.frequency = frequency

    @abstractmethod
    def run(self, model: torch.nn.Module, validation_loader: torch.utils.data.DataLoader, device: torch.device, global_step: int) -> dict:
        """
        The main method to execute the validation check.

        Args:
            model (nn.Module): The predictor model (unwrapped from DDP).
            validation_loader (DataLoader): The validation data loader.
            device (torch.device): The device to run computations on.
            global_step (int): The current global training step for wandb logging.
        
        Returns:
            A dictionary of metrics to be logged to wandb.
        """
        return {}

    def should_run(self, validation_cycle_count: int) -> bool:
        """
        Determines if the validator should run based on its frequency.
        """
        return validation_cycle_count % self.frequency == 0