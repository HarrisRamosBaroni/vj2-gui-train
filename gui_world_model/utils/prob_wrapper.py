import torch
import torch.nn as nn

class ProbabilisticWrapper(nn.Module):
    """
    A wrapper for probabilistic models that standardizes their output.
    
    Allows switching between returning the full probabilistic distribution 
    (e.g., a tuple for a loss function) and only the deterministic prediction 
    (e.g., the mean `mu` for rollouts or standard validation).
    """
    def __init__(self, probabilistic_model: nn.Module):
        super().__init__()
        self.model = probabilistic_model
        self._return_full_distribution = True

    def forward(self, *args, **kwargs):
        pred = self.model(*args, **kwargs)
        
        if self._return_full_distribution:
            return pred
        else:
            if isinstance(pred, (list, tuple)):
                return pred[0]
            return pred

    def return_full(self):
        """Sets the wrapper to return the full probabilistic output."""
        self._return_full_distribution = True

    def return_mean(self):
        """Sets the wrapper to return only the deterministic mean."""
        self._return_full_distribution = False

    def get_config(self):
        return self.model.get_config()

    def actions_formatter(self, actions):
        if hasattr(self.model, 'actions_formatter'):
            return self.model.actions_formatter(actions)
        B, T_seq, _, _ = actions.shape
        return actions.view(B, T_seq, -1)

    def __getattr__(self, name):
        """Delegate other attributes to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)