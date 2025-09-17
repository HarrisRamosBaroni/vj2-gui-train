import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and MLP."""
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, attn_drop=0.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        y = self.norm1(x)
        y, _ = self.attn(y, y, y)
        x = x + y
        
        # MLP with residual connection
        y = self.norm2(x)
        y = self.mlp(y)
        x = x + y
        
        return x


class CrossAttentionBlock(nn.Module):
    """
    Decoder block with self-attention on queries and cross-attention to memory.
    Architecture: SelfAttn -> CrossAttn -> MLP
    """
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, attn_drop=0.0, drop=0.0):
        super().__init__()
        # Self-attention for queries
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        
        # Cross-attention from queries to memory
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        
        # MLP
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, queries, memory):
        """
        Args:
            queries: Query tokens [B, N, D] where N is number of target patches
            memory: Memory tokens [B, M, D] where M = 1 + (T-1)*N (action + past patches)
        
        Returns:
            Updated queries [B, N, D]
        """
        # Self-attention among queries
        y = self.norm1(queries)
        y, _ = self.self_attn(y, y, y)
        queries = queries + y
        
        # Cross-attention from queries to memory
        y = self.norm2(queries)
        y, _ = self.cross_attn(y, memory, memory)  # Q from queries, K,V from memory
        queries = queries + y
        
        # MLP
        y = self.norm3(queries)
        y = self.mlp(y)
        queries = queries + y
        
        return queries


class LAMEncoder(nn.Module):
    """
    Latent Action Model Encoder (Genie-style)
    Encodes a sequence of patch token latents into per-transition action distributions.
    For each consecutive pair (z_t, z_{t+1}), predicts action parameters (mu_t, logvar_t).
    """
    
    def __init__(
        self, 
        patch_dim=1024,      # D: patch embedding dimension from V-JEPA
        num_patches=256,     # N: number of patches per frame (H*W)
        action_dim=128,
        embed_dim=512,
        depth=3,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        init_std=0.02,
        max_seq_len=20,      # Maximum temporal sequence length
        patch_size=16        # For spatial position calculation
    ):
        """
        Args:
            patch_dim: Dimension of each patch token from V-JEPA (D=1024)
            num_patches: Number of patches per frame (N=256 for 16x16 grid)
            action_dim: Dimension of action latent space
            embed_dim: Dimension of transformer embeddings  
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim multiplier
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            init_std: Standard deviation for weight initialization
            max_seq_len: Maximum temporal sequence length
            patch_size: Patch size for spatial position calculation
        """
        super().__init__()
        
        self.patch_dim = patch_dim
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Calculate spatial grid size (assuming square)
        self.grid_size = int(math.sqrt(num_patches))  # 16x16 = 256 patches
        
        # Patch embedding projection
        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        
        # No CLS token - we predict actions per timestep
        
        # Temporal positional embeddings (per frame)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(max_seq_len, embed_dim))
        
        # Spatial positional embeddings (per patch position in frame)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(num_patches, embed_dim))
        
        # Transformer blocks
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
        
        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Frame-level aggregation: pool patches to get frame representation
        self.frame_pool = nn.Linear(embed_dim, embed_dim)
        
        # Output heads for action distribution (per timestep)
        self.mu_head = nn.Linear(embed_dim, action_dim)
        self.logvar_head = nn.Linear(embed_dim, action_dim)
        
        # Initialize weights
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        
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

    def forward(self, z_sequence):
        # if dist.is_initialized():
        #     import logging
        #     logger = logging.getLogger(__name__)
        #     logger.info(f"Rank {dist.get_rank()}: Tensor device in LAMEncoder: {z_sequence.device}")
        """
        Args:
            z_sequence: Sequence of patch tokens [B, T, N, D] where:
                B = batch size
                T = temporal sequence length  
                N = number of patches per frame (256)
                D = patch embedding dimension (1024)
                
                IMPORTANT: Input should be pre-normalized using:
                F.layer_norm(z_sequence, (z_sequence.size(-1),))
                This ensures compatibility with VJ2 predictor models.
        
        Returns:
            mu: Action means for each transition, shape [B, T-1, A]
            logvar: Action log-variances for each transition, shape [B, T-1, A]
        """
        B, T, N, D = z_sequence.shape
        assert N == self.num_patches, f"Expected {self.num_patches} patches, got {N}"
        assert D == self.patch_dim, f"Expected patch dim {self.patch_dim}, got {D}"
        
        # Project patch tokens to embedding dimension
        # [B, T, N, D] -> [B, T, N, embed_dim]
        x = self.patch_proj(z_sequence)

        # Layer normalize after projection (helps with stability)
        x = F.layer_norm(x, (self.embed_dim,))

        # Add spatial positional embeddings (same for all frames)
        # spatial_pos_embed: [N, embed_dim] -> [1, 1, N, embed_dim]
        # Correctly unsqueeze to avoid implicit gathering in DDP
        spatial_pos = self.spatial_pos_embed.unsqueeze(0).unsqueeze(0)
        x = x + spatial_pos.to(x.device)   # [B, T, N, embed_dim]

        # Add temporal positional embeddings (same for all patches in a frame)
        # temporal_pos_embed: [max_seq_len, embed_dim] -> [1, T, 1, embed_dim] 
        # Correctly unsqueeze to avoid implicit gathering in DDP
        temporal_pos = self.temporal_pos_embed[:T].unsqueeze(0).unsqueeze(2)
        x = x + temporal_pos.to(x.device)   # [B, T, N, embed_dim]
        
        # Reshape to sequence: [B, T*N, embed_dim]
        x = x.reshape(B, T * N, self.embed_dim)
        
        # Apply transformer blocks (no CLS token - process full sequence)
        for block in self.blocks:
            x = block(x)
        
        # Normalize output
        x = self.norm(x)  # [B, T*N, embed_dim]
        
        # Reshape back to frame structure
        x = x.reshape(B, T, N, self.embed_dim)  # [B, T, N, embed_dim]
        
        # Pool patches within each frame to get frame-level representations
        # Mean pooling over spatial patches
        frame_features = x.mean(dim=2)  # [B, T, embed_dim]
        
        # Apply frame pooling layer for better representation
        frame_features = self.frame_pool(frame_features)  # [B, T, embed_dim]
        
        # For each transition (z_t, z_{t+1}), we use the representation at timestep t
        # This gives us T-1 transition representations
        transition_features = frame_features[:, :-1, :]  # [B, T-1, embed_dim]
        
        # Output action distribution parameters for each transition
        mu = self.mu_head(transition_features)       # [B, T-1, action_dim]
        logvar = self.logvar_head(transition_features)  # [B, T-1, action_dim]
        
        return mu, logvar


class LAMDecoder(nn.Module):
    """
    Latent Action Model Decoder with Cross-Attention
    Uses learnable query tokens that attend to both action and past patches to predict next frame.
    """
    
    def __init__(
        self,
        patch_dim=1024,
        num_patches=256, 
        action_dim=128,
        embed_dim=512,
        depth=3,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        init_std=0.02,
        max_seq_len=20
    ):
        """
        Args:
            patch_dim: Dimension of each patch token (D=1024)
            num_patches: Number of patches per frame (N=256)
            action_dim: Dimension of action latent space
            embed_dim: Dimension of transformer embeddings
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim multiplier
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            init_std: Standard deviation for weight initialization
            max_seq_len: Maximum temporal sequence length
        """
        super().__init__()
        
        self.patch_dim = patch_dim
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Input projections for memory
        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.action_proj = nn.Linear(action_dim, embed_dim)
        
        # Learnable query embeddings for next frame patches
        self.query_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Positional embeddings
        self.temporal_pos_embed = nn.Parameter(torch.zeros(max_seq_len, embed_dim))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(num_patches, embed_dim))
        # Separate spatial positional embeddings for queries (target frame)
        self.query_pos_embed = nn.Parameter(torch.zeros(num_patches, embed_dim))
        
        # Cross-attention decoder blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
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
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.query_pos_embed, std=0.02)
        
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
            # For CrossAttentionBlock, we have self_attn and cross_attn
            if hasattr(block, 'self_attn') and hasattr(block.self_attn, 'out_proj'):
                block.self_attn.out_proj.weight.data.div_(layer_scale)
            if hasattr(block, 'cross_attn') and hasattr(block.cross_attn, 'out_proj'):
                block.cross_attn.out_proj.weight.data.div_(layer_scale)
            if isinstance(block.mlp, nn.Sequential):
                block.mlp[-2].weight.data.div_(layer_scale)
    
    def forward(self, z_past, action_latent):
        """
        Args:
            z_past: Past patch sequence [B, T-1, N, D] where T-1 is past frames
                    IMPORTANT: Input should be pre-normalized using:
                    F.layer_norm(z_past, (z_past.size(-1),))
                    This ensures compatibility with VJ2 predictor models.
            action_latent: Action latent vector [B, A]
        
        Returns:
            z_next_pred: Predicted next frame patches [B, N, D]
        """
        B, T_past, N, D = z_past.shape
        assert N == self.num_patches, f"Expected {self.num_patches} patches, got {N}"
        assert D == self.patch_dim, f"Expected patch dim {self.patch_dim}, got {D}"
        
        # === Prepare Memory: Additive Action Conditioning (Genie-style) ===
        
        # Project past patches to embedding dimension
        # [B, T_past, N, D] -> [B, T_past, N, embed_dim]
        z_embed = self.patch_proj(z_past)

        # Layer normalize after projection (helps with L1 loss stability)
        z_embed = F.layer_norm(z_embed, (self.embed_dim,))

        # Add spatial positional embeddings to past patches
        # Correctly unsqueeze to avoid implicit gathering in DDP
        spatial_pos = self.spatial_pos_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, N, embed_dim]
        z_embed = z_embed + spatial_pos.to(z_embed.device)

        # Add temporal positional embeddings to past patches
        # Correctly unsqueeze to avoid implicit gathering in DDP
        temporal_pos = self.temporal_pos_embed[:T_past].unsqueeze(0).unsqueeze(2)  # [1, T_past, 1, embed_dim]
        z_embed = z_embed + temporal_pos.to(z_embed.device)
        
        # Project action latent and add to all patch tokens (additive conditioning)
        action_embed = self.action_proj(action_latent)  # [B, embed_dim]
        action_embed = F.layer_norm(action_embed, (self.embed_dim,))
        action_embed = action_embed.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, embed_dim]
        
        # Broadcast-add action to every patch token
        z_embed = z_embed + action_embed  # [B, T_past, N, embed_dim]
        
        # Flatten to sequence for transformer processing
        # Ensure we use the actual tensor dimensions, not assumptions
        current_B, current_T, current_N, current_embed = z_embed.shape
        memory = z_embed.reshape(current_B, current_T * current_N, current_embed)
        
        # === Prepare Queries: Learnable tokens for next frame patches ===
        
        # Initialize query tokens with learnable embeddings
        queries = self.query_embed.expand(B, -1, -1)  # [B, N, embed_dim]
        
        # Add positional embeddings for target frame patches
        query_pos = self.query_pos_embed.unsqueeze(0)  # [1, N, embed_dim]
        queries = queries + query_pos  # [B, N, embed_dim]
        
        # Also add action embedding to queries (same additive conditioning)
        queries = queries + action_embed.squeeze(1)  # [B, N, embed_dim] + [B, 1, embed_dim] -> [B, N, embed_dim]
        
        # === Cross-Attention Decoding ===
        
        # Apply cross-attention blocks
        # Each block: queries attend to memory (action + past patches)
        for block in self.blocks:
            queries = block(queries, memory)
        
        # === Output Prediction ===
        
        # Final normalization
        queries = self.norm(queries)  # [B, N, embed_dim]
        
        # Project each query to patch space
        z_next_pred = self.output_proj(queries)  # [B, N, D]
        
        return z_next_pred