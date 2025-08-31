import torch
import torch.nn as nn

def apply_film(z, gamma, beta):
    """
    Applies Feature-wise Linear Modulation (FiLM) to a tensor.

    Args:
        z (torch.Tensor): Input tensor of shape [B, T, N, D_m].
        gamma (torch.Tensor): Scale tensor of shape [B, T, 1, D_m].
        beta (torch.Tensor): Shift tensor of shape [B, T, 1, D_m].

    Returns:
        torch.Tensor: Modulated tensor of the same shape as z.
    """
    return z * gamma + beta

class ActionBiGRUEncoder(nn.Module):
    """
    Encodes a sequence of actions into a fixed-size context vector using a bidirectional GRU.
    Input: action sequence of shape [B, T, L, 3] where L is action sequence length
    Output: context vector of shape [B, T, D_ctx]
    """
    def __init__(self, action_dim, embedding_dim, context_dim):
        super().__init__()
        self.action_embedding = nn.Linear(action_dim, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            context_dim // 2,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, actions):
        """
        Args:
            actions (torch.Tensor): Shape [B, T, L, 3]
        """
        B, T, L, _ = actions.shape
        actions_reshaped = actions.view(B * T, L, -1)
        
        embedded_actions = self.action_embedding(actions_reshaped)
        _, h_n = self.gru(embedded_actions)
        
        # h_n shape is [2, B*T, D_ctx/2] (2 for bidirectional)
        # Concatenate the final hidden states of the forward and backward GRUs
        context = torch.cat([h_n[0], h_n[1]], dim=-1) # -> [B*T, D_ctx]
        
        # Reshape back to [B, T, D_ctx]
        context = context.view(B, T, -1)
        return context

class FiLMHeads(nn.Module):
    """
    A hypernetwork that generates FiLM parameters (gamma and beta) from a context vector.
    It creates separate linear heads for each modulated layer and sublayer (attention and MLP).
    """
    def __init__(self, context_dim, model_dim, num_modulated_layers):
        super().__init__()
        self.num_modulated_layers = num_modulated_layers
        
        # Create a list of ModuleLists, one for each modulated layer
        self.heads = nn.ModuleList()
        for _ in range(num_modulated_layers):
            # Each layer gets two heads for gamma (att, ff) and two for beta (att, ff)
            layer_heads = nn.ModuleDict({
                'gamma_att': nn.Linear(context_dim, model_dim),
                'beta_att': nn.Linear(context_dim, model_dim),
                'gamma_ff': nn.Linear(context_dim, model_dim),
                'beta_ff': nn.Linear(context_dim, model_dim),
            })
            self.heads.append(layer_heads)
        
        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights to produce gamma ~ 1 and beta ~ 0 at the beginning of training.
        """
        for layer_heads in self.heads:
            for head_name, head in layer_heads.items():
                nn.init.zeros_(head.weight)
                if 'gamma' in head_name:
                    nn.init.constant_(head.bias, 1.0)
                else: # beta
                    nn.init.zeros_(head.bias)

    def forward(self, context, clamp_gamma_alpha=None):
        """
        Generates FiLM parameters for all modulated layers.
        
        Args:
            context (torch.Tensor): The action context vector of shape [B, T, D_ctx].
            clamp_gamma_alpha (float, optional): If provided, clamps gamma values
                using the formula: 1 + alpha * tanh(gamma). Defaults to None.

        Returns:
            dict: A nested dictionary where keys are layer indices and values are
                  dictionaries of FiLM parameters {'gamma_att', 'beta_att', ...}.
        """
        film_params = {}
        for i, layer_heads in enumerate(self.heads):
            layer_params = {}
            for head_name, head in layer_heads.items():
                param = head(context)
                if 'gamma' in head_name and clamp_gamma_alpha is not None and clamp_gamma_alpha > 0:
                    param = 1 + clamp_gamma_alpha * torch.tanh(param)
                layer_params[head_name] = param.unsqueeze(2) # [B, T, 1, D_m] for broadcasting
            film_params[i] = layer_params
        return film_params