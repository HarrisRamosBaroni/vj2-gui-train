import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BatchedVQLAMDecoder(nn.Module):
    """
    Batched VQ LAM Decoder using self-attention with causal masking.
    Processes all timesteps in parallel for teacher forcing.

    Architecture: Interleaved [state, codes, query] sequence with block-causal masking.
    """

    def __init__(
        self,
        patch_dim=1024,
        num_patches=256,
        codebook_dim=128,
        embed_dim=512,
        depth=3,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        init_std=0.02,
        max_seq_len=20
    ):
        super().__init__()

        self.patch_dim = patch_dim
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.codebook_dim = codebook_dim
        self.max_seq_len = max_seq_len

        # Input projections
        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.code_proj = nn.Linear(codebook_dim, embed_dim)

        # Learnable query embeddings for next frame patches
        self.query_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Positional embeddings for interleaved sequence
        # Max length: max_seq_len * (2*num_patches + 3)
        # Each timestep has: N frame patches + 3 code tokens + N query patches
        chunk_size = 2 * num_patches + 3
        max_seq_tokens = max_seq_len * chunk_size
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_tokens, embed_dim))

        # Self-attention transformer blocks (no cross-attention)
        from .modules import TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop_rate,
                drop=drop_rate
            )
            for _ in range(depth)
        ])

        # Final normalization and output projection
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, patch_dim)

        # Initialize weights
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        # Initialize embeddings
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        for layer_id, block in enumerate(self.blocks):
            layer_scale = math.sqrt(2.0 * (layer_id + 1))
            if hasattr(block.attn, 'out_proj'):
                block.attn.out_proj.weight.data.div_(layer_scale)
            if isinstance(block.mlp, nn.Sequential):
                block.mlp[-2].weight.data.div_(layer_scale)

    def _create_causal_mask(self, num_chunks: int, chunk_size: int, device: torch.device) -> torch.Tensor:
        """
        Create block-causal mask for interleaved sequence.

        Args:
            num_chunks: Number of timesteps (T-1)
            chunk_size: Tokens per chunk (2*N + 3)
            device: Device for mask tensor

        Returns:
            mask: [L, L] where L = num_chunks * chunk_size
        """
        # Create block-level causal mask: chunk t cannot see chunk t+1 onwards
        block_mask = torch.triu(
            torch.full((num_chunks, num_chunks), float('-inf'), device=device),
            diagonal=1
        )  # [num_chunks, num_chunks]

        # Expand to token level: each block becomes chunk_size x chunk_size
        mask = block_mask.repeat_interleave(chunk_size, dim=0).repeat_interleave(chunk_size, dim=1)
        # [L, L] where L = num_chunks * chunk_size

        return mask

    def forward(self, z_past, code_embeddings):
        """
        Batched decoding with causal masking for all timesteps in parallel.

        Args:
            z_past: Past patch sequence [B, T-1, N, D]
            code_embeddings: Code embeddings [B, T-1, 3, codebook_dim]

        Returns:
            z_next_pred: Predicted next frames [B, T-1, N, D]
        """
        B, T_minus_1, N, D = z_past.shape
        assert N == self.num_patches, f"Expected {self.num_patches} patches, got {N}"
        assert D == self.patch_dim, f"Expected patch dim {self.patch_dim}, got {D}"

        # Project to embedding space
        z_embed = self.patch_proj(z_past)  # [B, T-1, N, embed_dim]
        z_embed = F.layer_norm(z_embed, (self.embed_dim,))

        code_embed = self.code_proj(code_embeddings)  # [B, T-1, 3, embed_dim]
        code_embed = F.layer_norm(code_embed, (self.embed_dim,))

        # Build interleaved sequence: [frame, code, query] for each timestep
        # Chunk structure for timestep t:
        #   - N frame patches from z_t
        #   - 3 code tokens for transition t
        #   - N query patches (for predicting z_{t+1})
        tokens_list = []
        for t in range(T_minus_1):
            # Frame patches
            tokens_list.append(z_embed[:, t, :, :])  # [B, N, embed_dim]

            # Code tokens (all 3 at once)
            tokens_list.append(code_embed[:, t, :, :])  # [B, 3, embed_dim]

            # Query tokens
            query = self.query_embed.expand(B, -1, -1)  # [B, N, embed_dim]
            tokens_list.append(query)

        # Concatenate all tokens
        sequence = torch.cat(tokens_list, dim=1)  # [B, L, embed_dim] where L = T-1*(2N+3)
        L = sequence.shape[1]

        # Add positional encoding
        sequence = sequence + self.pos_embed[:, :L, :].to(sequence.device)

        # Create block-causal mask
        chunk_size = 2 * N + 3
        causal_mask = self._create_causal_mask(T_minus_1, chunk_size, sequence.device)

        # Apply self-attention blocks with causal mask
        for block in self.blocks:
            sequence = block(sequence, mask=causal_mask)

        # Normalize
        sequence = self.norm(sequence)

        # Extract query tokens (predictions) from each chunk
        predictions = []
        for t in range(T_minus_1):
            # Query tokens are at positions: t*chunk_size + N + 3 : (t+1)*chunk_size
            query_start = t * chunk_size + N + 3
            query_end = query_start + N
            pred_t = sequence[:, query_start:query_end, :]  # [B, N, embed_dim]
            predictions.append(pred_t)

        predictions = torch.stack(predictions, dim=1)  # [B, T-1, N, embed_dim]

        # Project back to patch space
        z_next_pred = self.output_proj(predictions)  # [B, T-1, N, D]

        return z_next_pred
