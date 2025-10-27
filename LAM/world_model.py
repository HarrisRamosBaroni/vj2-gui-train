"""
World Model for Latent Action Modeling (LAM)

Architecture following description.md:
- Action Encoder: E_� : �_0^t � a_0^t
- World Encoder: E_� : �_0^T � h_world
- Dynamics Predictor: P_� : (�_0^t, a_0^t, h_world) � �_{t+1}

The WorldModel owns shared Tokenizer/Detokenizer and orchestrates all three components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from LAM.st_transformer import Tokenizer, Detokenizer, TransformerBlock


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ) with EMA codebook updates.

    Multi-level hierarchical quantization where residuals from each level
    are quantized by the next level.

    From description.md:
    - Uses straight-through estimator for gradients
    - EMA (Exponential Moving Average) for codebook updates
    - Codebook loss: ||sg[z_e] - e||^2 (sg = stop gradient)
    - Commitment loss: beta * ||z_e - sg[e]||^2

    Args:
        num_levels: Number of quantization levels (L_a or L_h)
        codebook_sizes: Tuple of codebook sizes per level, e.g., (256, 256, 256)
        d_code: Code dimension
        beta: Commitment loss weight (default 0.25)
        decay: EMA decay rate (default 0.99)
    """

    def __init__(self, num_levels, codebook_sizes, d_code, beta=0.25, decay=0.99):
        super().__init__()
        assert len(codebook_sizes) == num_levels, \
            f"codebook_sizes length {len(codebook_sizes)} must match num_levels {num_levels}"

        self.num_levels = num_levels
        self.codebook_sizes = codebook_sizes
        self.d_code = d_code
        self.beta = beta
        self.decay = decay

        # Create codebooks for each level as buffers (not trainable parameters)
        # to prevent optimizer interference with EMA updates
        for i in range(num_levels):
            self.register_buffer(
                f'codebook_{i}',
                torch.randn(codebook_sizes[i], d_code)
            )

        # EMA cluster size (for tracking codebook usage)
        # Initialize to small positive constant to avoid division by zero
        self.register_buffer('cluster_size', torch.ones(num_levels, max(codebook_sizes)) * 0.01)
        # EMA embedding sum (for codebook updates)
        # Initialize to zeros for proper EMA behavior
        self.register_buffer('embed_avg', torch.zeros(num_levels, max(codebook_sizes), d_code))
        # Cache recent encoder outputs for dead code reinitialization
        self.register_buffer('recent_inputs_cache', torch.zeros(1024, d_code))
        self.register_buffer('cache_ptr', torch.tensor(0, dtype=torch.long))

    def _get_codebook(self, level):
        """Helper to get codebook buffer for a given level."""
        return getattr(self, f'codebook_{level}')

    def _set_codebook(self, level, value):
        """Helper to set codebook buffer for a given level."""
        getattr(self, f'codebook_{level}').copy_(value)

    def reinitialize_dead_codes(self, recent_inputs=None, dead_threshold=0.01):
        """
        Reinitialize dead or unused codes with recent encoder outputs.

        Dead codes are those with cluster_size below dead_threshold. This prevents
        codebook collapse where some codes are never used.

        Args:
            recent_inputs: [N, d_code] - recent encoder outputs to use for reinitialization
                          If None, uses the cached recent inputs from training
            dead_threshold: Threshold below which codes are considered dead (default 0.01)

        Returns:
            dict: Statistics about reinitialization per level
                  {'level_0': num_reinitialized, 'level_1': num_reinitialized, ...}
        """
        # Use cached inputs if none provided
        if recent_inputs is None:
            recent_inputs = self.recent_inputs_cache

        if recent_inputs is None or recent_inputs.numel() == 0:
            return {f'level_{i}': 0 for i in range(self.num_levels)}

        stats = {}
        with torch.no_grad():
            for level in range(self.num_levels):
                codebook_size = self.codebook_sizes[level]
                codebook = self._get_codebook(level)

                # Identify dead codes (those with very low cluster size)
                cluster_counts = self.cluster_size[level, :codebook_size]
                dead_mask = cluster_counts < dead_threshold

                num_dead = dead_mask.sum().item()
                stats[f'level_{level}'] = 0

                if num_dead == 0:
                    continue

                # Randomly sample from recent inputs to reinitialize dead codes
                num_samples = min(num_dead, recent_inputs.shape[0])
                if num_samples > 0:
                    # Random indices from recent inputs
                    random_indices = torch.randperm(recent_inputs.shape[0], device=recent_inputs.device)[:num_samples]
                    sampled_inputs = recent_inputs[random_indices]

                    # Find dead code indices
                    dead_indices = torch.where(dead_mask)[0][:num_samples]

                    # Reinitialize dead codes
                    new_codebook = codebook.clone()
                    new_codebook[dead_indices] = sampled_inputs

                    # Update codebook and reset EMA statistics for reinitialized codes
                    self._set_codebook(level, new_codebook)
                    self.cluster_size[level, dead_indices] = 0.01  # Reset to initial value
                    self.embed_avg[level, dead_indices] = sampled_inputs * 0.01  # Initialize embed_avg

                    stats[f'level_{level}'] = num_samples

        return stats

    def forward(self, z):
        """
        Args:
            z: [*, d_code] - continuous embeddings

        Returns:
            z_q: [*, d_code] - quantized embeddings
            indices: [*, num_levels] - codebook indices for each level
            commitment_loss: scalar - commitment loss
            codebook_loss: scalar - codebook loss (for logging)
        """
        original_shape = z.shape
        z_flat = z.view(-1, self.d_code)  # [N, d_code]
        N = z_flat.shape[0]

        # Cache recent inputs for dead code reinitialization (during training only)
        if self.training:
            with torch.no_grad():
                cache_size = self.recent_inputs_cache.shape[0]
                # How many samples to add (limited by cache size)
                num_to_cache = min(N, cache_size)
                # Sample random subset if N > cache_size
                if N > num_to_cache:
                    sample_indices = torch.randperm(N, device=z_flat.device)[:num_to_cache]
                    samples_to_cache = z_flat[sample_indices]
                else:
                    samples_to_cache = z_flat

                # Circular buffer update
                ptr = self.cache_ptr.item()
                if ptr + num_to_cache <= cache_size:
                    self.recent_inputs_cache[ptr:ptr + num_to_cache] = samples_to_cache
                    self.cache_ptr.fill_((ptr + num_to_cache) % cache_size)
                else:
                    # Wrap around
                    first_part = cache_size - ptr
                    self.recent_inputs_cache[ptr:] = samples_to_cache[:first_part]
                    self.recent_inputs_cache[:num_to_cache - first_part] = samples_to_cache[first_part:]
                    self.cache_ptr.fill_(num_to_cache - first_part)

        z_q_total = torch.zeros_like(z_flat)
        indices_list = []
        commitment_loss = 0.0
        codebook_loss = 0.0

        residual = z_flat

        for level in range(self.num_levels):
            codebook = self._get_codebook(level)  # [codebook_size, d_code]
            codebook_size = self.codebook_sizes[level]

            # Compute distances: [N, codebook_size]
            distances = torch.cdist(residual, codebook)

            # Find nearest codebook entries
            indices = torch.argmin(distances, dim=1)  # [N]
            indices_list.append(indices)

            # Quantize
            z_q_level = F.embedding(indices, codebook)  # [N, d_code]

            # Compute losses (with stop gradient)
            # Codebook loss: ||sg[z_e] - sg[e]||^2 (for monitoring only with EMA updates)
            codebook_loss += F.mse_loss(z_q_level.detach(), residual.detach())
            # Commitment loss: beta * ||z_e - sg[e]||^2 (trains encoder to commit to codebook)
            commitment_loss += self.beta * F.mse_loss(residual, z_q_level.detach())

            # EMA codebook update (only during training)
            if self.training:
                with torch.no_grad():
                    # One-hot encoding of indices
                    encodings = F.one_hot(indices, codebook_size).float()  # [N, codebook_size]

                    # Compute per-batch statistics
                    batch_cluster_size = encodings.sum(0)  # [codebook_size]
                    embed_sum = encodings.t() @ residual  # [codebook_size, d_code]

                    # All-reduce for distributed training (synchronize across GPUs)
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        torch.distributed.all_reduce(batch_cluster_size)
                        torch.distributed.all_reduce(embed_sum)

                    # Update cluster size with EMA
                    self.cluster_size[level, :codebook_size] = \
                        self.cluster_size[level, :codebook_size] * self.decay + \
                        batch_cluster_size * (1 - self.decay)

                    # Update embedding average with EMA
                    self.embed_avg[level, :codebook_size] = \
                        self.embed_avg[level, :codebook_size] * self.decay + \
                        embed_sum * (1 - self.decay)

                    # Normalize and update codebook with explicit clamping
                    n = self.cluster_size[level, :codebook_size].unsqueeze(1)
                    # Clamp denominator to avoid division by near-zero values
                    n_clamped = torch.clamp(n, min=1e-5)
                    updated_codebook = self.embed_avg[level, :codebook_size] / n_clamped
                    self._set_codebook(level, updated_codebook)

            # Straight-through estimator: use quantized value in forward, but gradient flows through
            z_q_level = residual + (z_q_level - residual).detach()

            # Accumulate quantized values
            z_q_total += z_q_level

            # Update residual for next level
            residual = residual - z_q_level

        # Stack indices: [N, num_levels]
        indices_stacked = torch.stack(indices_list, dim=1)

        # Reshape back to original shape
        z_q = z_q_total.view(original_shape)
        indices_out = indices_stacked.view(*original_shape[:-1], self.num_levels)

        return z_q, indices_out, commitment_loss, codebook_loss


class ActionEncoder(nn.Module):
    """
    Action Encoder: Encodes transitions (z_t, z_{t+1}) → action code a_t

    From description.md:
    - Input: tokenized features [B*(T+1), d_model, H', W'] (receives from WorldModel)
    - ST-Transformer blocks with shifted_causal masking (diagonal=2)
    - Spatial mean-pooling
    - Linear projection: d_model → d_code_a
    - RVQ quantization
    - Output: [B, T, d_code_a] action codes

    The action encoder sees T+1 frames and produces T actions.
    With shifted causal masking, position t can see frames 0...t+1,
    allowing it to encode the transition from t to t+1 as action a_t.

    Args:
        d_model: Feature dimension after tokenization
        d_code: Action code dimension
        num_lvq_levels: Number of RVQ levels
        codebook_sizes: Tuple of codebook sizes per level
        patchsize: List of (width, height) tuples for multi-scale patching
        num_blocks: Number of ST-Transformer blocks (default 3)
        beta: RVQ commitment loss weight (default 0.25)
        decay: EMA decay rate for RVQ (default 0.99)
    """

    def __init__(
        self,
        d_model=256,
        d_code=128,
        num_lvq_levels=3,
        codebook_sizes=(256, 256, 256),
        patchsize=[(8, 8), (4, 4), (2, 2), (1, 1)],
        num_blocks=3,
        beta=0.25,
        decay=0.99
    ):
        super().__init__()
        self.d_model = d_model
        self.d_code = d_code

        # ST-Transformer blocks with shifted_causal masking (diagonal=2)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(patchsize=patchsize, hidden=d_model, temporal_causal_mask='shifted_causal')
            for _ in range(num_blocks)
        ])

        # Spatial mean-pooling
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # Linear projection: d_model → d_code
        self.action_proj = nn.Linear(d_model, d_code)

        # Residual Vector Quantizer
        self.rvq = ResidualVectorQuantizer(
            num_levels=num_lvq_levels,
            codebook_sizes=codebook_sizes,
            d_code=d_code,
            beta=beta,
            decay=decay
        )

    def forward(self, tokens, B, T, return_attention=False):
        """
        Args:
            tokens: [B*(T+1), d_model, H', W'] - tokenized features from WorldModel
            B: batch size
            T: temporal length (number of frames - 1, since we have T+1 input frames)
            return_attention: If True, return attention weights from all transformer blocks

        Returns:
            action_codes_quantized: [B, T, d_code] - quantized action codes
            indices: [B, T, num_lvq_levels] - codebook indices
            commitment_loss: scalar - RVQ commitment loss
            codebook_loss: scalar - RVQ codebook loss
            attention_weights: (optional) List of attention weights from each block
        """
        BT_plus_1, d_model, H_prime, W_prime = tokens.shape
        assert BT_plus_1 == B * (T + 1), f"Expected B*(T+1)={B*(T+1)}, got {BT_plus_1}"
        assert d_model == self.d_model, f"Expected d_model={self.d_model}, got {d_model}"

        # Create spatial mask (all zeros = nothing masked)
        mask = torch.zeros(BT_plus_1, 1, H_prime, W_prime, device=tokens.device)

        # Apply ST-Transformer blocks with shifted causal masking
        x = tokens
        all_attention_weights = []  # Collect attention from all blocks
        for block in self.transformer_blocks:
            # TransformerBlock expects dict format
            out = block({'x': x, 'm': mask, 'b': B, 'c': d_model}, return_attention=return_attention)
            x = out['x']  # [B*(T+1), d_model, H', W']

            # Collect attention weights if requested
            if return_attention:
                all_attention_weights.append(out['attention'])

        # Spatial mean-pooling: [B*(T+1), d_model, H', W'] → [B*(T+1), d_model]
        x = self.spatial_pool(x).squeeze(-1).squeeze(-1)  # [B*(T+1), d_model]

        # Reshape to [B, T+1, d_model]
        x = x.view(B, T + 1, self.d_model)

        # Drop the last frame: [B, T+1, d_model] → [B, T, d_model]
        # We have T+1 frames but only T actions (actions between frames)
        x = x[:, :-1, :]  # [B, T, d_model]

        # Project to action code dimension
        action_codes_continuous = self.action_proj(x)  # [B, T, d_code]

        # Quantize via RVQ
        action_codes_quantized, indices, commitment_loss, codebook_loss = \
            self.rvq(action_codes_continuous)

        if return_attention:
            return action_codes_quantized, indices, commitment_loss, codebook_loss, all_attention_weights
        return action_codes_quantized, indices, commitment_loss, codebook_loss


class WorldEncoder(nn.Module):
    """
    World Encoder: Encodes full sequence z_0^T → world hypothesis h_world

    From description.md:
    - Input: tokenized features [B*T, d_model, H', W'] (receives from WorldModel)
    - ST-Transformer blocks with NO causal masking (bidirectional)
    - Spatial + temporal mean-pooling
    - Linear projection: d_model → d_code_h
    - RVQ quantization
    - Output: [B, d_code_h] world hypothesis

    The world encoder sees all T frames and produces a single world hypothesis
    that captures global context.

    Args:
        d_model: Feature dimension after tokenization
        d_code: World hypothesis dimension
        num_lvq_levels: Number of RVQ levels
        codebook_sizes: Tuple of codebook sizes per level
        patchsize: List of (width, height) tuples for multi-scale patching
        num_blocks: Number of ST-Transformer blocks (default 3)
        beta: RVQ commitment loss weight (default 0.25)
        decay: EMA decay rate for RVQ (default 0.99)
    """

    def __init__(
        self,
        d_model=256,
        d_code=128,
        num_lvq_levels=3,
        codebook_sizes=(256, 256, 256),
        patchsize=[(8, 8), (4, 4), (2, 2), (1, 1)],
        num_blocks=3,
        beta=0.25,
        decay=0.99
    ):
        super().__init__()
        self.d_model = d_model
        self.d_code = d_code

        # ST-Transformer blocks with NO causal masking (bidirectional)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(patchsize=patchsize, hidden=d_model, temporal_causal_mask='none')
            for _ in range(num_blocks)
        ])

        # Spatial mean-pooling
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # Linear projection: d_model → d_code
        self.world_proj = nn.Linear(d_model, d_code)

        # Residual Vector Quantizer
        self.rvq = ResidualVectorQuantizer(
            num_levels=num_lvq_levels,
            codebook_sizes=codebook_sizes,
            d_code=d_code,
            beta=beta,
            decay=decay
        )

    def forward(self, tokens, B, T):
        """
        Args:
            tokens: [B*T, d_model, H', W'] - tokenized features from WorldModel
            B: batch size
            T: temporal length

        Returns:
            world_emb_quantized: [B, d_code] - quantized world hypothesis
            indices: [B, num_lvq_levels] - codebook indices
            commitment_loss: scalar - RVQ commitment loss
            codebook_loss: scalar - RVQ codebook loss
        """
        BT, d_model, H_prime, W_prime = tokens.shape
        assert BT == B * T, f"Expected B*T={B*T}, got {BT}"
        assert d_model == self.d_model, f"Expected d_model={self.d_model}, got {d_model}"

        # Create spatial mask (all zeros = nothing masked)
        mask = torch.zeros(BT, 1, H_prime, W_prime, device=tokens.device)

        # Apply ST-Transformer blocks with NO causal masking (bidirectional)
        x = tokens
        for block in self.transformer_blocks:
            # TransformerBlock expects dict format
            out = block({'x': x, 'm': mask, 'b': B, 'c': d_model})
            x = out['x']  # [B*T, d_model, H', W']

        # Spatial mean-pooling: [B*T, d_model, H', W'] → [B*T, d_model]
        x = self.spatial_pool(x).squeeze(-1).squeeze(-1)  # [B*T, d_model]

        # Reshape to [B, T, d_model]
        x = x.view(B, T, self.d_model)

        # Temporal mean-pooling: [B, T, d_model] → [B, d_model]
        x = x.mean(dim=1)  # [B, d_model]

        # Project to world hypothesis dimension
        world_emb_continuous = self.world_proj(x)  # [B, d_code]

        # Quantize via RVQ
        world_emb_quantized, indices, commitment_loss, codebook_loss = \
            self.rvq(world_emb_continuous)

        return world_emb_quantized, indices, commitment_loss, codebook_loss


class DynamicsPredictor(nn.Module):
    """
    Dynamics Predictor: Predicts next frame (z_0^t, a_0^t, h_world) → z_{t+1}

    From description.md:
    - Input: tokenized features [B*T, d_model, H', W'], action codes, world hypothesis
    - Linear projections: W_a^z (d_code_a → d_model), W_h^z (d_code_h → d_model)
    - Additive combination: frame_tokens + action_emb + world_emb (broadcasted)
    - ST-Transformer blocks with causal masking (diagonal=1)
    - Output: [B*T, d_model, H', W'] output tokens (WorldModel detokenizes)

    The predictor combines visual tokens with action and world information,
    then predicts future frames autoregressively.

    Args:
        d_model: Feature dimension after tokenization
        d_code_a: Action code dimension
        d_code_h: World hypothesis dimension
        patchsize: List of (width, height) tuples for multi-scale patching
        num_blocks: Number of ST-Transformer blocks (default 3)
    """

    def __init__(
        self,
        d_model=256,
        d_code_a=128,
        d_code_h=128,
        patchsize=[(8, 8), (4, 4), (2, 2), (1, 1)],
        num_blocks=3
    ):
        super().__init__()
        self.d_model = d_model
        self.d_code_a = d_code_a
        self.d_code_h = d_code_h

        # Linear projection: action codes → d_model
        self.action_proj = nn.Linear(d_code_a, d_model)

        # Linear projection: world hypothesis → d_model
        self.world_proj = nn.Linear(d_code_h, d_model)

        # ST-Transformer blocks with causal masking (diagonal=1)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(patchsize=patchsize, hidden=d_model, temporal_causal_mask='causal')
            for _ in range(num_blocks)
        ])

    def forward(self, tokens, action_codes, world_emb, B, T):
        """
        Per description.md: Predictor receives GT[0:T-1] frames and T-1 actions.
        Perfect alignment - no padding needed!

        Args:
            tokens: [B*T, d_model, H', W'] - tokenized features (T frames)
            action_codes: [B, T, d_code_a] - action codes (T actions, matching T frames)
            world_emb: [B, d_code_h] - world hypothesis (quantized)
            B: batch size
            T: temporal length (number of frames = number of actions)

        Returns:
            pred_tokens: [B*T, d_model, H', W'] - predicted output tokens (T predictions)
        """
        BT, d_model, H_prime, W_prime = tokens.shape
        assert BT == B * T, f"Expected B*T={B*T}, got {BT}"
        assert d_model == self.d_model, f"Expected d_model={self.d_model}, got {d_model}"

        # Verify action codes match frame count (perfect alignment per spec)
        assert action_codes.shape == (B, T, self.d_code_a), \
            f"Expected action_codes [{B}, {T}, {self.d_code_a}], got {action_codes.shape}"

        # =====================================================================
        # Step 1: Project action codes and world hypothesis to d_model
        # =====================================================================
        # Action: [B, T, d_code_a] → [B, T, d_model]
        # Perfect alignment! No padding needed per description.md
        action_emb = self.action_proj(action_codes)  # [B, T, d_model]

        # World: [B, d_code_h] → [B, d_model]
        world_emb_proj = self.world_proj(world_emb)  # [B, d_model]

        # =====================================================================
        # Step 2: Broadcast and add to frame tokens (additive combination)
        # =====================================================================
        # Reshape for broadcasting
        # Action: [B, T, d_model] → [B*T, d_model, 1, 1]
        action_emb_broadcast = action_emb.view(B * T, d_model, 1, 1)
        action_emb_broadcast = action_emb_broadcast.expand(-1, -1, H_prime, W_prime)

        # World: [B, d_model] → [B*T, d_model, 1, 1]
        world_emb_broadcast = world_emb_proj.unsqueeze(1).repeat(1, T, 1)  # [B, T, d_model]
        world_emb_broadcast = world_emb_broadcast.view(B * T, d_model, 1, 1)
        world_emb_broadcast = world_emb_broadcast.expand(-1, -1, H_prime, W_prime)

        # Additive combination: frame_tokens + action_emb + world_emb
        x = tokens + action_emb_broadcast + world_emb_broadcast  # [B*T, d_model, H', W']

        # =====================================================================
        # Step 3: Apply ST-Transformer blocks with causal masking
        # =====================================================================
        # Create spatial mask (all zeros = nothing masked)
        mask = torch.zeros(BT, 1, H_prime, W_prime, device=tokens.device)

        for block in self.transformer_blocks:
            # TransformerBlock expects dict format
            out = block({'x': x, 'm': mask, 'b': B, 'c': d_model})
            x = out['x']  # [B*T, d_model, H', W']

        return x

    def autoregressive_rollout(
        self,
        context: torch.Tensor,
        action_sequence: torch.Tensor,
        world_emb: torch.Tensor,
        tokenizer,
        detokenizer
    ) -> torch.Tensor:
        """
        Perform true autoregressive rollout where each prediction is immediately used as context.

        This is the real generation mode where the predictor must use its own predictions
        to generate the next frame, rather than relying on ground truth context.

        Args:
            context: [B, T_context, C=16, H=64, W=64] - initial context frames (raw frames)
            action_sequence: [B, T_rollout, d_code_a] - sequence of action codes to apply
            world_emb: [B, d_code_h] - fixed world hypothesis embedding
            tokenizer: Tokenizer module to convert frames to tokens
            detokenizer: Detokenizer module to convert tokens back to frames

        Returns:
            predictions: [B, T_rollout, C=16, H=64, W=64] - autoregressively generated frames
        """
        B, T_context, C, H, W = context.shape
        T_rollout = action_sequence.shape[1]
        device = context.device

        assert action_sequence.shape == (B, T_rollout, self.d_code_a), \
            f"Expected action_sequence shape [{B}, {T_rollout}, {self.d_code_a}], got {action_sequence.shape}"
        assert world_emb.shape == (B, self.d_code_h), \
            f"Expected world_emb shape [{B}, {self.d_code_h}], got {world_emb.shape}"
        assert C == 16 and H == 64 and W == 64, \
            f"Expected context with C=16, H=64, W=64, got C={C}, H={H}, W={W}"

        # List to store predicted frames
        predictions = []

        # Current context starts with the initial context
        current_context = context.clone()  # [B, T_context, C, H, W]

        # Autoregressive loop: predict T_rollout frames one by one
        for t in range(T_rollout):
            T_current = current_context.shape[1]

            # =====================================================================
            # Step 1: Tokenize current context
            # =====================================================================
            context_tokens, _, _ = tokenizer(current_context)  # [B*T_current, d_model, H', W']
            _, d_model, H_prime, W_prime = context_tokens.shape

            # =====================================================================
            # Step 2: Get actions for current context
            # =====================================================================
            # After our fix: predictor expects T frames + T actions (perfect alignment)
            # For T_current context frames, we need T_current actions
            # action_sequence: [B, T_rollout, d_code_a]
            # Extract first T_current actions: [B, T_current, d_code_a]
            current_actions = action_sequence[:, :T_current, :]  # [B, T_current, d_code_a]

            # =====================================================================
            # Step 3: Run dynamics predictor to get next frame prediction
            # =====================================================================
            # After fix: Pass T_current frames + T_current actions → get T_current predictions
            pred_tokens = self.forward(
                context_tokens,      # [B*T_current, d_model, H', W']
                current_actions,     # [B, T_current, d_code_a]
                world_emb,
                B,
                T_current
            )  # [B*T_current, d_model, H', W']

            # =====================================================================
            # Step 4: Extract prediction for the NEXT frame (frame T_current)
            # =====================================================================
            # Predictor outputs T_current predictions (for frames 1 to T_current)
            # The last prediction (at position T_current-1) predicts frame T_current
            # Extract the last frame's prediction
            last_frame_tokens = pred_tokens[(T_current - 1) * B : T_current * B, :, :, :]  # [B, d_model, H', W']

            # =====================================================================
            # Step 5: Detokenize to get the predicted frame
            # =====================================================================
            pred_frame = detokenizer(last_frame_tokens)  # [B, C=16, H=64, W=64]

            # Store prediction
            predictions.append(pred_frame.unsqueeze(1))  # [B, 1, C, H, W]

            # =====================================================================
            # Step 6: Append prediction to context for next step
            # =====================================================================
            current_context = torch.cat([current_context, pred_frame.unsqueeze(1)], dim=1)  # [B, T_current+1, C, H, W]

        # Concatenate all predictions along time dimension
        predictions = torch.cat(predictions, dim=1)  # [B, T_rollout, C, H, W]

        return predictions


class WorldModel(nn.Module):
    """
    Top-level World Model for Latent Action Modeling.

    Orchestrates three main components:
    1. ActionEncoder: Encodes transitions (z_t, z_{t+1}) � action code a_t
    2. WorldEncoder: Encodes full sequence z_0^T � world hypothesis h_world
    3. DynamicsPredictor: Predicts next frame (z_0^t, a_0^t, h_world) � z_{t+1}

    Architecture:
    - Tokenizer called ONCE (shared across all components)
    - Tokenized features distributed to ActionEncoder, WorldEncoder, DynamicsPredictor
    - Detokenizer called ONCE (at the end)

    Args:
        d_model: Feature dimension after tokenization (default 256)
        d_code_a: Action code dimension (default 128)
        d_code_h: World hypothesis dimension (default 128)
        num_lvq_levels_a: Number of RVQ levels for action encoder (default 3)
        num_lvq_levels_h: Number of RVQ levels for world encoder (default 6)
        codebook_sizes_a: Tuple of codebook sizes per level for actions (default (12, 64, 256))
        codebook_sizes_h: Tuple of codebook sizes per level for world (default (12, 24, 48, 256, 256, 256))
        patchsize: List of (width, height) tuples for multi-scale patching (default [(8,8), (4,4), (2,2), (1,1)])
        num_encoder_blocks: Number of ST-Transformer blocks for encoders (default 3)
        num_decoder_blocks: Number of ST-Transformer blocks for predictor (default 3)
        beta_a: RVQ commitment loss weight for action encoder (default 0.25)
        beta_h: RVQ commitment loss weight for world encoder (default 0.25)
        decay: EMA decay rate for RVQ codebook updates (default 0.99)
        use_random_temporal_pe: Enable random temporal PE offset during training for length extrapolation (default False)
        max_pe_offset: Maximum random offset for temporal PE when use_random_temporal_pe=True (default 120)
    """

    def __init__(
        self,
        d_model=256,
        d_code_a=128,
        d_code_h=128,
        num_lvq_levels_a=3,
        num_lvq_levels_h=6,
        codebook_sizes_a=(12, 64, 256),
        codebook_sizes_h=(12, 24, 48, 256, 256, 256),
        patchsize=[(8, 8), (4, 4), (2, 2), (1, 1)],
        num_encoder_blocks=3,
        num_decoder_blocks=3,
        beta_a=0.25,
        beta_h=0.25,
        decay=0.99,
        use_random_temporal_pe=False,
        max_pe_offset=120
    ):
        super().__init__()

        self.d_model = d_model
        self.d_code_a = d_code_a
        self.d_code_h = d_code_h

        # =====================================================================
        # Shared Tokenizer/Detokenizer (owned by WorldModel)
        # =====================================================================
        # Tokenizer: [B, T, 16, 64, 64] -> [B*T, d_model, 16, 16]
        # Includes positional embeddings (spatial + temporal)
        # With optional random temporal PE offset for length extrapolation
        self.tokenizer = Tokenizer(
            in_channels=16,
            channel=d_model,
            use_random_temporal_pe=use_random_temporal_pe,
            max_pe_offset=max_pe_offset
        )

        # Detokenizer: [B*T, d_model, 16, 16] -> [B*T, 16, 64, 64]
        self.detokenizer = Detokenizer(channel=d_model, out_channels=16)

        # =====================================================================
        # Three Main Components (to be implemented)
        # =====================================================================

        # ActionEncoder: Encodes transitions (z_t, z_{t+1}) → action code a_t
        self.action_encoder = ActionEncoder(
            d_model=d_model,
            d_code=d_code_a,
            num_lvq_levels=num_lvq_levels_a,
            codebook_sizes=codebook_sizes_a,
            patchsize=patchsize,
            num_blocks=num_encoder_blocks,
            beta=beta_a,
            decay=decay
        )

        # WorldEncoder: Encodes full sequence z_0^T → world hypothesis h_world
        self.world_encoder = WorldEncoder(
            d_model=d_model,
            d_code=d_code_h,
            num_lvq_levels=num_lvq_levels_h,
            codebook_sizes=codebook_sizes_h,
            patchsize=patchsize,
            num_blocks=num_encoder_blocks,
            beta=beta_h,
            decay=decay
        )

        # DynamicsPredictor: Predicts next frame (z_0^t, a_0^t, h_world) → z_{t+1}
        self.dynamics_predictor = DynamicsPredictor(
            d_model=d_model,
            d_code_a=d_code_a,
            d_code_h=d_code_h,
            patchsize=patchsize,
            num_blocks=num_decoder_blocks
        )

    def forward(self, h_sequence):
        """
        Forward pass through the World Model.
        Per description.md: GT[0:T-1] in → PRED[1:T] out

        Args:
            h_sequence: [B, T, 16, 64, 64] - VVAE latent sequence

        Returns:
            Dictionary containing:
                - pred_frames: [B, T-1, 16, 64, 64] - predicted frames (predicting frames 1 to T)
                - action_codes: [B, T-1, d_code_a] - quantized action codes
                - action_indices: [B, T-1, num_lvq_levels_a] - action codebook indices
                - world_emb: [B, d_code_h] - quantized world hypothesis
                - world_indices: [B, num_lvq_levels_h] - world codebook indices
                - losses: Dictionary of loss components
        """
        B, T, C, H, W = h_sequence.shape
        assert C == 16 and H == 64 and W == 64, \
            f"Expected input shape [B, T, 16, 64, 64], got [B, T, {C}, {H}, {W}]"

        # =====================================================================
        # Step 1: Tokenize ONCE (shared computation)
        # =====================================================================
        # Input: [B, T, 16, 64, 64]
        # Output: [B*T, d_model, 16, 16] with positional embeddings
        tokens, B_out, T_out = self.tokenizer(h_sequence)
        assert B_out == B and T_out == T, "Tokenizer changed batch/temporal dimensions"

        BT, d_model, H_prime, W_prime = tokens.shape
        assert BT == B * T, f"Expected B*T={B*T}, got {BT}"
        assert d_model == self.d_model, f"Expected d_model={self.d_model}, got {d_model}"

        # =====================================================================
        # Step 2: Distribute tokenized features to all three components
        # =====================================================================

        # Call ActionEncoder
        # Action encoder processes T frames to produce T-1 actions
        # (T frames = T-1 transitions)
        # ActionEncoder.forward expects tokens [B*T, d_model, H', W'] and produces [B, T-1, d_code_a]
        action_codes, action_indices, action_commit_loss, action_codebook_loss = \
            self.action_encoder(tokens, B, T - 1)  # Pass T-1 as the number of actions expected

        # Call WorldEncoder
        # World encoder processes all T frames to produce world hypothesis
        # world_emb: [B, d_code_h]
        # world_indices: [B, num_lvq_levels_h]
        # world_commit_loss, world_codebook_loss: scalars
        world_emb, world_indices, world_commit_loss, world_codebook_loss = \
            self.world_encoder(tokens, B, T)

        # Per description.md: Drop last frame, pass GT[0:T-1] to predictor
        # This gives perfect alignment: T-1 frames + T-1 actions → T-1 predictions
        tokens_input = tokens[:B * (T - 1)]  # [B*(T-1), d_model, H', W']

        # Call DynamicsPredictor
        # Dynamics predictor combines tokens, actions, and world hypothesis
        # to predict future frames
        # pred_tokens: [B*(T-1), d_model, H', W']
        pred_tokens = self.dynamics_predictor(tokens_input, action_codes, world_emb, B, T - 1)

        # =====================================================================
        # Step 3: Detokenize ONCE (at the end)
        # =====================================================================
        # Input: [B*(T-1), d_model, 16, 16]
        # Output: [B*(T-1), 16, 64, 64]
        pred_frames_flat = self.detokenizer(pred_tokens)

        # Reshape to [B, T-1, 16, 64, 64]
        pred_frames = pred_frames_flat.view(B, T - 1, 16, 64, 64)

        # =====================================================================
        # Step 4: Return all outputs and losses
        # =====================================================================
        return {
            'pred_frames': pred_frames,
            'action_codes': action_codes,
            'action_indices': action_indices,
            'world_emb': world_emb,
            'world_indices': world_indices,
            'losses': {
                'action_commit_loss': action_commit_loss,
                'action_codebook_loss': action_codebook_loss,
                'world_commit_loss': world_commit_loss,
                'world_codebook_loss': world_codebook_loss,
            }
        }

    def update_codebook_ema_decay(self, decay: float):
        """
        Update the EMA decay rate for all RVQ codebooks.

        Args:
            decay: New EMA decay rate (between 0 and 1)
        """
        self.action_encoder.rvq.decay = decay
        self.world_encoder.rvq.decay = decay
