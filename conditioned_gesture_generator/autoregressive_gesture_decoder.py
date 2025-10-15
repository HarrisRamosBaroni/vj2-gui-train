import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer sequences."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape [B, L, D]
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]

class ActionEncoder(nn.Module):
    """Encodes action latent sequence using Transformer encoder."""
    def __init__(self, d_action: int, d_model: int, nhead: int, num_layers: int):
        super().__init__()
        self.d_action = d_action
        self.d_model = d_model

        # Project action latents to model dimension
        self.action_proj = nn.Linear(d_action, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, action_sequence: torch.Tensor) -> torch.Tensor:
        """Encode action sequence.

        Args:
            action_sequence: Shape [B, T, d_action]
        Returns:
            encoded_actions: Shape [B, T, d_model]
        """
        # Project and add positional encoding
        x = self.action_proj(action_sequence)  # [B, T, d_model]
        x = self.pos_encoder(x)

        # Encode with transformer
        encoded = self.transformer_encoder(x)  # [B, T, d_model]
        return encoded


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for conditioning."""

    def __init__(self, d_model: int, d_action: int):
        super().__init__()
        self.d_model = d_model
        self.d_action = d_action

        # Linear layers for scale and shift
        self.scale_layer = nn.Linear(d_action, d_model)
        self.shift_layer = nn.Linear(d_action, d_model)

        # Initialize scale and shift weights with small normal distribution, biases to zero
        nn.init.normal_(self.scale_layer.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.scale_layer.bias)
        nn.init.normal_(self.shift_layer.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.shift_layer.bias)

        # Initialize monitoring attributes
        self.last_scale_norm = None
        self.last_shift_norm = None
        self.last_scale_values = None
        self.last_shift_values = None

    def forward(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            x: Input features [B, L, d_model]
            actions: Action latents [B, L, d_action]
        Returns:
            Modulated features [B, L, d_model]
        """
        scale = self.scale_layer(actions)  # [B, L, d_model]
        shift = self.shift_layer(actions)  # [B, L, d_model]

        # Store for monitoring (detached to avoid affecting gradients)
        self.last_scale_norm = torch.norm(scale, dim=-1).detach()  # [B, L]
        self.last_shift_norm = torch.norm(shift, dim=-1).detach()  # [B, L]
        self.last_scale_values = scale.detach()  # [B, L, d_model]
        self.last_shift_values = shift.detach()  # [B, L, d_model]

        # FiLM: x * (1 + scale) + shift
        return x * (1 + scale) + shift


def expand_actions(action_sequence: torch.Tensor, upsample_factor: int = 250) -> torch.Tensor:
    """Expand action sequence from 1 action per timestep to upsample_factor gestures per action.

    Args:
        action_sequence: [B, T, d_action]
        upsample_factor: Number of gesture tokens per action (default 250)
    Returns:
        expanded_actions: [B, T*upsample_factor, d_action]
    """
    return action_sequence.repeat_interleave(upsample_factor, dim=1)


class FactorizedGestureDecoder(nn.Module):
    """Autoregressive gesture decoder with factorized outputs (x, y, touch)."""
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 x_classes: int = 3000, y_classes: int = 3000, touch_classes: int = 3,
                 dropout: float = 0.0, film_module: Optional[FiLM] = None):
        super().__init__()
        self.d_model = d_model
        self.x_classes = x_classes
        self.y_classes = y_classes
        self.touch_classes = touch_classes
        self.film_module = film_module

        # Separate token embeddings for each component
        # Split d_model evenly but handle remainder
        embed_dim_base = d_model // 3
        embed_dim_x = embed_dim_base + (1 if d_model % 3 > 0 else 0)
        embed_dim_y = embed_dim_base + (1 if d_model % 3 > 1 else 0)
        embed_dim_touch = embed_dim_base

        self.x_embed = nn.Embedding(x_classes, embed_dim_x)
        self.y_embed = nn.Embedding(y_classes, embed_dim_y)
        self.touch_embed = nn.Embedding(touch_classes, embed_dim_touch)

        # Projection to full model dimension (input dim = sum of embedding dims)
        concat_dim = embed_dim_x + embed_dim_y + embed_dim_touch
        self.embed_proj = nn.Linear(concat_dim, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer encoder (for GPT-style decoder-only architecture)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            activation='gelu',
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pre-output layer normalization
        self.output_norm = nn.LayerNorm(d_model)

        # Separate output heads for each component
        self.x_head = nn.Linear(d_model, x_classes)
        self.y_head = nn.Linear(d_model, y_classes)
        self.touch_head = nn.Linear(d_model, touch_classes)

    def forward(self,
                token_dict: Dict[str, torch.Tensor],
                src_mask: Optional[torch.Tensor] = None,
                expanded_actions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Decode gesture tokens autoregressively with FiLM conditioning.

        Args:
            token_dict: Dict with keys 'x', 'y', 'touch', each shape [B, L]
            src_mask: Causal mask for source sequence
            expanded_actions: Action latents expanded to match L, shape [B, L, d_action]
        Returns:
            Dict with 'x_logits', 'y_logits', 'touch_logits', each appropriate shape
        """
        # Embed each component separately
        x_emb = self.x_embed(token_dict['x'])      # [B, L, d_model//3]
        y_emb = self.y_embed(token_dict['y'])      # [B, L, d_model//3]
        touch_emb = self.touch_embed(token_dict['touch'])  # [B, L, d_model//3]

        # Concatenate embeddings and project to full dimension
        combined_emb = torch.cat([x_emb, y_emb, touch_emb], dim=-1)  # [B, L, d_model]
        x = self.embed_proj(combined_emb)
        x = self.embed_dropout(x)

        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.pos_dropout(x)

        # Apply FiLM conditioning if available
        if self.film_module is not None and expanded_actions is not None:
            x = self.film_module(x, expanded_actions)

        # Apply transformer encoder (GPT-style self-attention only)
        decoded = self.transformer_encoder(
            src=x,
            mask=src_mask
        )  # [B, L, d_model]

        # Apply layer normalization
        decoded = self.output_norm(decoded)

        # Apply separate output heads
        x_logits = self.x_head(decoded)        # [B, L, x_classes]
        y_logits = self.y_head(decoded)        # [B, L, y_classes]
        touch_logits = self.touch_head(decoded)  # [B, L, touch_classes]

        return {
            'x_logits': x_logits,
            'y_logits': y_logits,
            'touch_logits': touch_logits
        }


class FactorizedAutoregressiveGestureDecoder(nn.Module):
    """Conditional GPT-style factorized autoregressive gesture decoder with cross-attention."""

    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 x_classes: int = 3000,
                 y_classes: int = 3000,
                 touch_classes: int = 3,
                 tokenization_mode: str = "factorized",
                 max_seq_len: int = 2048,
                 d_action: int = 128,  # Action latent dimension
                 **kwargs):
        super().__init__()

        # Store config
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.x_classes = x_classes
        self.y_classes = y_classes
        self.touch_classes = touch_classes
        self.tokenization_mode = tokenization_mode
        self.max_seq_len = max_seq_len
        self.d_action = d_action

        # FiLM conditioning and upsample factor
        self.film = FiLM(d_model, d_action)
        self.upsample_factor = 250

        # Gesture decoder: autoregressive with self-attention only and FiLM conditioning
        self.gesture_decoder = FactorizedGestureDecoder(
            d_model, nhead, num_layers, x_classes, y_classes, touch_classes, dropout=0.0, film_module=self.film
        )

        # Special tokens
        self.bos_x_token_id = 0
        self.bos_y_token_id = 0
        self.bos_touch_token_id = 0

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for checkpointing."""
        return {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "x_classes": self.x_classes,
            "y_classes": self.y_classes,
            "touch_classes": self.touch_classes,
            "tokenization_mode": self.tokenization_mode,
            "max_seq_len": self.max_seq_len,
            "d_action": self.d_action,
        }

    def forward(self, action_sequence: torch.Tensor, gesture_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for training with FiLM conditioning.

        Args:
            action_sequence: Action latents, shape [B, T, d_action]
            gesture_tokens: Dict with 'x', 'y', 'touch' tokens, each shape [B, L]

        Returns:
            Dictionary containing logits for each component
        """
        B = gesture_tokens['x'].shape[0]
        L = gesture_tokens['x'].shape[1]

        # Expand actions from [B, T, d_action] to [B, T*250, d_action]
        expanded_actions = expand_actions(action_sequence, self.upsample_factor)  # [B, T*250, d_action]

        # Handle variable length: truncate or pad to match gesture sequence length L
        if expanded_actions.shape[1] > L:
            # Truncate if expanded actions are longer than gesture sequence
            expanded_actions = expanded_actions[:, :L, :]
        elif expanded_actions.shape[1] < L:
            # Pad by repeating the last action if gesture sequence is longer
            last_action = expanded_actions[:, -1:, :]  # [B, 1, d_action]
            padding_length = L - expanded_actions.shape[1]
            padding = last_action.repeat(1, padding_length, 1)  # [B, padding_length, d_action]
            expanded_actions = torch.cat([expanded_actions, padding], dim=1)  # [B, L, d_action]

        # Prepare input tokens (shift right for teacher forcing)
        input_tokens = {
            'x': torch.cat([
                torch.full((B, 1), self.bos_x_token_id, device=gesture_tokens['x'].device),
                gesture_tokens['x'][:, :-1]
            ], dim=1),
            'y': torch.cat([
                torch.full((B, 1), self.bos_y_token_id, device=gesture_tokens['y'].device),
                gesture_tokens['y'][:, :-1]
            ], dim=1),
            'touch': torch.cat([
                torch.full((B, 1), self.bos_touch_token_id, device=gesture_tokens['touch'].device),
                gesture_tokens['touch'][:, :-1]
            ], dim=1)
        }

        # Create causal mask
        src_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=gesture_tokens['x'].device),
            diagonal=1
        )

        # Autoregressive decoding with FiLM conditioning
        output_logits = self.gesture_decoder(input_tokens, src_mask=src_mask, expanded_actions=expanded_actions)

        # Add target tokens for loss computation
        output_logits.update({
            'target_x': gesture_tokens['x'],
            'target_y': gesture_tokens['y'],
            'target_touch': gesture_tokens['touch']
        })

        return output_logits

    def generate(self, action_sequence: torch.Tensor, max_length: Optional[int] = None, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """Generate gesture tokens autoregressively conditioned on actions.

        Args:
            action_sequence: Action latents, shape [B, T, d_action]
            max_length: Maximum sequence length (defaults to T*250)
            temperature: Sampling temperature

        Returns:
            Dict with generated 'x', 'y', 'touch' tokens
        """
        B, T, _ = action_sequence.shape
        device = action_sequence.device

        if max_length is None:
            max_length = T * self.upsample_factor  # 250 gestures per action (standard ratio)

        # Expand actions for FiLM conditioning
        expanded_actions = expand_actions(action_sequence, self.upsample_factor)  # [B, T*250, d_action]
        if expanded_actions.shape[1] < max_length:
            # Pad by repeating the last action if needed
            last_action = expanded_actions[:, -1:, :]
            padding_length = max_length - expanded_actions.shape[1]
            padding = last_action.repeat(1, padding_length, 1)
            expanded_actions = torch.cat([expanded_actions, padding], dim=1)

        # Initialize with BOS tokens
        generated_x = torch.full((B, 1), self.bos_x_token_id, device=device)
        generated_y = torch.full((B, 1), self.bos_y_token_id, device=device)
        generated_touch = torch.full((B, 1), self.bos_touch_token_id, device=device)

        for step in range(max_length):
            current_length = step + 1

            # Create causal mask for current length
            src_mask = torch.triu(
                torch.full((current_length, current_length), float('-inf'), device=device),
                diagonal=1
            )

            # Prepare current input
            current_tokens = {
                'x': generated_x,
                'y': generated_y,
                'touch': generated_touch
            }

            # Get logits for current step with FiLM conditioning
            current_expanded_actions = expanded_actions[:, :current_length, :]  # Slice to current length
            output_logits = self.gesture_decoder(
                current_tokens,
                src_mask=src_mask,
                expanded_actions=current_expanded_actions
            )

            # Forbid BOS for the next token
            x_next = output_logits['x_logits'][:, -1, :]
            y_next = output_logits['y_logits'][:, -1, :]
            t_next = output_logits['touch_logits'][:, -1, :]

            x_next[:, 0] = -float('inf')
            y_next[:, 0] = -float('inf')
            t_next[:, 0] = -float('inf')

            # Sample next tokens (last position only) with temperature scaling
            next_x = torch.multinomial(
                F.softmax(x_next / temperature, dim=-1), 1
            )
            next_y = torch.multinomial(
                F.softmax(y_next / temperature, dim=-1), 1
            )
            next_touch = torch.multinomial(
                F.softmax(t_next / temperature, dim=-1), 1
            )

            # Append to sequences
            generated_x = torch.cat([generated_x, next_x], dim=1)
            generated_y = torch.cat([generated_y, next_y], dim=1)
            generated_touch = torch.cat([generated_touch, next_touch], dim=1)

        # Remove BOS tokens
        return {
            'x': generated_x[:, 1:],
            'y': generated_y[:, 1:],
            'touch': generated_touch[:, 1:]
        }

    def generate_full_rollout(self, action_sequence: torch.Tensor, max_length: Optional[int] = None,
                             use_argmax: bool = True, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """Generate gesture tokens using full rollout (no teacher forcing) for visualization.

        Args:
            action_sequence: Action latents, shape [B, T, d_action]
            max_length: Maximum sequence length (defaults to T*250)
            use_argmax: If True, use argmax sampling; if False, use multinomial

        Returns:
            Dict with generated 'x', 'y', 'touch' tokens
        """
        B, T, _ = action_sequence.shape
        device = action_sequence.device

        if max_length is None:
            max_length = T * self.upsample_factor  # 250 gestures per action

        # Expand actions for FiLM conditioning
        expanded_actions = expand_actions(action_sequence, self.upsample_factor)  # [B, T*250, d_action]
        if expanded_actions.shape[1] < max_length:
            # Pad by repeating the last action if needed
            last_action = expanded_actions[:, -1:, :]
            padding_length = max_length - expanded_actions.shape[1]
            padding = last_action.repeat(1, padding_length, 1)
            expanded_actions = torch.cat([expanded_actions, padding], dim=1)

        # Initialize with BOS tokens only
        generated_x = torch.full((B, 1), self.bos_x_token_id, device=device)
        generated_y = torch.full((B, 1), self.bos_y_token_id, device=device)
        generated_touch = torch.full((B, 1), self.bos_touch_token_id, device=device)

        for step in range(max_length):
            current_length = step + 1

            # Create causal mask for current length
            src_mask = torch.triu(
                torch.full((current_length, current_length), float('-inf'), device=device),
                diagonal=1
            )

            # Prepare current input
            current_tokens = {
                'x': generated_x,
                'y': generated_y,
                'touch': generated_touch
            }

            # Get logits for current step with FiLM conditioning
            current_expanded_actions = expanded_actions[:, :current_length, :]  # Slice to current length
            output_logits = self.gesture_decoder(
                current_tokens,
                src_mask=src_mask,
                expanded_actions=current_expanded_actions
            )

            # Forbid BOS for the next token
            x_next = output_logits['x_logits'][:, -1, :]
            y_next = output_logits['y_logits'][:, -1, :]
            t_next = output_logits['touch_logits'][:, -1, :]

            x_next[:, 0] = -float('inf')
            y_next[:, 0] = -float('inf')
            t_next[:, 0] = -float('inf')

            # Sample next tokens (last position only) with temperature scaling
            if use_argmax:
                # For argmax, temperature doesn't affect the result
                next_x = torch.argmax(x_next, dim=-1, keepdim=True)
                next_y = torch.argmax(y_next, dim=-1, keepdim=True)
                next_touch = torch.argmax(t_next, dim=-1, keepdim=True)
            else:
                # Apply temperature scaling for multinomial sampling
                next_x = torch.multinomial(
                    F.softmax(x_next / temperature, dim=-1), 1
                )
                next_y = torch.multinomial(
                    F.softmax(y_next / temperature, dim=-1), 1
                )
                next_touch = torch.multinomial(
                    F.softmax(t_next / temperature, dim=-1), 1
                )

            # Append to sequences
            generated_x = torch.cat([generated_x, next_x], dim=1)
            generated_y = torch.cat([generated_y, next_y], dim=1)
            generated_touch = torch.cat([generated_touch, next_touch], dim=1)

        # Remove BOS tokens
        return {
            'x': generated_x[:, 1:],
            'y': generated_y[:, 1:],
            'touch': generated_touch[:, 1:]
        }
