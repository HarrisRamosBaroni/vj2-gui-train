import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import logging
logger = logging.getLogger(__name__)
import argparse
import math

import torch
import torch.nn as nn
import torchvision.transforms as T

from gui_world_model.utils.modules import ACBlock as Block
from gui_world_model.utils.modules import build_action_block_causal_attention_mask
from src.utils.tensors import trunc_normal_
from gui_world_model.encoder import VJEPA2Wrapper
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import (
    ACTION_DIM,
    ACTIONS_PER_BATCH,
    OBSERVATIONS_PER_WINDOW
)

# Import CNN Gesture Classifier for frozen action encoder
from conditioned_gesture_generator.cnn_gesture_classifier import CNNGestureClassifier

class FrozenCNNActionEncoder(nn.Module):
    """
    Frozen CNN-based action encoder that embeds CNN weights directly as non-trainable buffers.
    """
    
    def __init__(self, checkpoint_path, device='cuda'):
        super().__init__()
        
        # Load pretrained CNN gesture classifier
        print(f"Loading frozen CNN action encoder from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract config from checkpoint
        if 'config' not in checkpoint:
            raise ValueError(f"Checkpoint {checkpoint_path} missing config. Ensure it was saved with model config.")
        
        config = checkpoint['config']
        print(f"CNN Encoder config: {config}")
        
        # Create CNN gesture classifier with exact same architecture
        self.cnn_classifier = CNNGestureClassifier(
            input_dim=2,  # Only x, y coordinates
            sequence_length=config['sequence_length'],
            latent_dim=config['latent_dim'],
            num_classes=config['num_classes'],
            encoder_channels=config['encoder_channels'],
            decoder_channels=config['decoder_channels'],
        )
        
        # Load pretrained weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        self.cnn_classifier.load_state_dict(state_dict)
        
        # Store config for saving/loading
        self.cnn_config = config
        
        # Freeze all parameters by converting to non-trainable buffers
        # This ensures weights are saved with predictor but not trained
        for name, param in self.cnn_classifier.named_parameters():
            # Replace dots with underscores for valid buffer names
            buffer_name = f'frozen_cnn_{name.replace(".", "_")}'
            self.register_buffer(buffer_name, param.data.clone())
            param.requires_grad = False
        
        self.cnn_classifier.eval()
        
        # Store dimensions for reference
        self.latent_dim = config['latent_dim']
        self.sequence_length = config['sequence_length']
        
        print(f"✅ Frozen CNN action encoder loaded successfully")
        print(f"   Latent dim: {self.latent_dim}")
        print(f"   Sequence length: {self.sequence_length}")
        print(f"   Frozen parameters: {sum(p.numel() for p in self.cnn_classifier.parameters())}")
    
    def forward(self, gesture_coords):
        """
        Encode gesture coordinates to latent vectors using frozen CNN encoder.
        
        Args:
            gesture_coords: [B, T=250, 2] gesture coordinates (x, y only)
            
        Returns:
            latent: [B, latent_dim] encoded latent vector
        """
        with torch.no_grad():  # Ensure no gradients
            latent = self.cnn_classifier.encode(gesture_coords)
        return latent
    
    def get_config(self):
        """Return CNN configuration for saving."""
        return self.cnn_config


class VJ2GUIPredictorActionEncoded(nn.Module):
    """
    Action-Conditioned Predictor for GUI control using V-JEPA 2 architecture
    with frozen CNN-based action encoder.
    
    Takes a sequence of visual tokens (State) and a corresponding sequence of
    structured action sequences to predict the next sequence of visual tokens.
  
    Input Tensors:
        - z (State): A tensor of shape [B, T, N, D] representing the visual state.
            - B: Batch size of video clips.
            - T: Number of time steps (typically frames // tubelet_size).
            - N: Number of visual tokens per frame (e.g., 14x14 = 196).
            - D: Embedding dimension of each token (e.g., 1024).
        - actions (Action): A tensor of shape [B, T, 250, 3] representing structured actions.
            - T: Number of video frame chunks (e.g., 8).
            - 250: Gesture sequence length per chunk.
            - 3: x, y, pressure coordinates (pressure will be dropped).

    Output Tensor:
        - Predicted visual tokens of shape [B, T, N, D].
    """

    def __init__(
        self,
        img_size=(256, 256),
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        embed_dim=1024,
        predictor_embed_dim=1024,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        use_rope=True,
        frozen_action_encoder_checkpoint=None,  # Path to CNN gesture classifier checkpoint
        device=None
    ):
        super().__init__()

        # Capture all arguments except 'self' and '__class__'
        self._config = self.capture_init_args(locals())

        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Token embeddings
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        
        # Initialize frozen CNN action encoder
        if frozen_action_encoder_checkpoint is None:
            raise ValueError("frozen_action_encoder_checkpoint must be provided")
            
        self.frozen_action_encoder = FrozenCNNActionEncoder(frozen_action_encoder_checkpoint, self.device)
        
        # Action projection layer: CNN latent dim → predictor embed dim
        self.action_projection = nn.Linear(self.frozen_action_encoder.latent_dim, predictor_embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.predictor_blocks = nn.ModuleList([
            Block(
                use_rope=use_rope,
                grid_size=self.grid_height,
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.SiLU if use_silu else nn.GELU,
                wide_silu=wide_silu,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)

        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        # Build causal attention mask
        self.attn_mask = build_action_block_causal_attention_mask(
            self.num_frames // self.tubelet_size,
            self.grid_height,
            self.grid_width,
            1,                  # ← action_tokens = 1
        )
        logger.debug(f"{self.attn_mask.shape=}")
        print(f"{self.attn_mask.shape=}")  # [2056, 2056] where 2056 = T*(H*W+1) = 8*(16*16+1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    # ───────────────────────────────────────────────────────────────────────
    def forward(self, z, actions):
        """
        Processes the visual and action tokens to predict the next visual state.
        - z:       State tensor [B, T, N, D]
        - actions: Action tensor [B, T, 250, 3] - structured gesture sequences
        """
        B, T, N, D = z.shape
        H, W = self.grid_height, self.grid_width
        if N != H * W:
            raise ValueError(f"Mismatch in spatial token count: {N} != {H}*{W}")

        # Validate action tensor shape
        if len(actions.shape) != 4:
            raise ValueError(f"Expected actions shape [B, T, 250, 3], got {actions.shape}")
        
        B_act, T_act, seq_len, coord_dim = actions.shape
        if B_act != B or T_act != T:
            raise ValueError(f"Action batch/time mismatch: actions {actions.shape} vs visual {z.shape}")
        
        # Critical debugging: log actual input shapes
        logger.info(f"🔍 PREDICTOR FORWARD: z shape: {z.shape}, actions shape: {actions.shape}")
        logger.info(f"🔍 PREDICTOR FORWARD: B={B}, T={T}, N={N}, D={D}, H={H}, W={W}")
        
        z = self.predictor_embed(z)                        # [B, T, N, D]
        
        # Process actions through frozen CNN encoder
        # Drop pressure channel: [B, T, 250, 3] → [B, T, 250, 2]
        actions_xy = actions[:, :, :, :2]
        logger.debug(f"actions_xy.shape after dropping pressure: {actions_xy.shape}")
        
        # Process each timestep through CNN encoder
        action_latents = []
        for t in range(T):
            # Extract gesture sequence for this timestep: [B, 250, 2]
            timestep_gesture = actions_xy[:, t, :, :]
            logger.debug(f"timestep_gesture[{t}].shape: {timestep_gesture.shape}")
            
            # Ensure coordinates are in [0, 1] range (same preprocessing as training)
            timestep_gesture = torch.clamp(timestep_gesture, 0.0, 1.0)
            
            # Process through frozen CNN encoder: [B, 250, 2] → [B, latent_dim]
            timestep_latent = self.frozen_action_encoder(timestep_gesture)
            logger.debug(f"timestep_latent[{t}].shape: {timestep_latent.shape}")
            action_latents.append(timestep_latent)

        # Stack timestep latents: [B, T, latent_dim]
        action_latents = torch.stack(action_latents, dim=1)
        logger.debug(f"stacked action_latents.shape: {action_latents.shape}")
        
        # Project to predictor embedding dimension: [B, T, latent_dim] → [B, T, predictor_embed_dim]
        action_embeddings = self.action_projection(action_latents)
        logger.debug(f"action_embeddings.shape: {action_embeddings.shape}")
        
        # Add sequence dimension for concatenation: [B, T, predictor_embed_dim] → [B, T, 1, predictor_embed_dim]
        a = action_embeddings.unsqueeze(2)
        logger.debug(f"a.shape after unsqueeze: {a.shape}")

        # z = z.view(B, T, H * W, D)  # TODO: unnecessary? since N == H*W anyway...
        logger.debug(f"z.shape: {z.shape}")
        logger.debug(f"a.shape: {a.shape}")
        x = torch.cat([a, z], dim=2).flatten(1, 2)         # [B, T*(1+N), D]
        logger.debug(f"{x.shape=}")  # [B, T*(H*W+1), D] = [1, 2056, 1024]

        # Should be using mask with shape [:x.size(1), :x.size(1)] = [2056, 2056]
        attn_mask = self.attn_mask[:x.size(1), :x.size(1)].to(z.device, non_blocking=True)  # build_action_block_causal_attention_mask(T, H, W, 1).to(z.device)
        logger.debug(f"{attn_mask.shape=}")  # [2056, 2056] where 2056 = T*(H*W+1) = 8*(16*16+1)
        for blk in self.predictor_blocks:
            x = blk(x, mask=None, attn_mask=attn_mask,
                    T=T, H=H, W=W, action_tokens=1)        # ← 1 token

        x = x.view(B, T, 1 + H * W, D)
        x = x[:, :, 1:, :].reshape(B, T, H * W, D)         # strip action token for prediction

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        
        return x

    def actions_formatter(self, actions):
        """The FiLM predictor requires the original 4D action tensor [B, T, L, 3]."""
        return actions

    def get_config(self):
        return self._config

    def capture_init_args(self, local_vars):
        # Captures constructor arguments for saving/loading.
        # Exclude 'self' and '__class__'
        return {k: v for k, v in local_vars.items() if k not in ['self', '__class__']}

def load_predictor_a_encoded_model(model_path, device):
    """Loads the VJ2GUIPredictorActionEncoded model from a checkpoint."""
    print(f"Loading action-encoded predictor from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model_config = None
    if "predictor_config" in checkpoint:
        model_config = checkpoint["predictor_config"]
        # Ensure norm_layer is correctly referenced if it's a string
        if "norm_layer" in model_config and isinstance(model_config["norm_layer"], str):
            if model_config["norm_layer"] == "nn.LayerNorm":
                model_config["norm_layer"] = torch.nn.LayerNorm
        model = VJ2GUIPredictorActionEncoded(**model_config).to(device)
    else:
        # Fallback for older checkpoints without config
        raise ValueError("Action-encoded predictor requires checkpoint with config including frozen_action_encoder_checkpoint path")

    # Handle various checkpoint formats (DDP, simple, etc.)
    if "predictor" in checkpoint:
        state_dict = checkpoint["predictor"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Remove `module.` prefix if present from DDP training
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    model.eval()
    model.requires_grad_(False)
    print("✅ Action-encoded predictor loaded successfully.")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='info', choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('--cnn_checkpoint', type=str, required=True, 
                       help='Path to CNN gesture classifier checkpoint')
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))    

    encoder = VJEPA2Wrapper(num_frames=16)
    predictor = VJ2GUIPredictorActionEncoded(
        num_frames=16, 
        frozen_action_encoder_checkpoint=args.cnn_checkpoint
    ).to(encoder.device)

    # Test with structured action data
    from torchvision import transforms
    from device_control.screen_capture import capture_screen
    NUM_CONTEXT_FRAMES = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    buffer = []
    while len(buffer) < NUM_CONTEXT_FRAMES:
        frame = capture_screen()
        tensor = preprocess(frame).to(DEVICE)
        buffer.append(tensor)
    video_tensor = torch.stack(buffer, dim=0).unsqueeze(0)
    print(f"{video_tensor.shape=}")

    z_all = encoder(video_tensor)  # [1, 16, 256, 1024]
    B, T, N, D = z_all.shape # T = 8 (reduced from 16 to 8 by tubelet = 2)
    
    # Create structured action tensor: [B, T, 250, 3]
    actions = torch.rand(B, T, 250, 3).to(z_all.device)  # Random normalized coordinates

    print(f"Visual tokens shape      (z): {z_all.shape}")
    print(f"Structured actions shape (a): {actions.shape}")

    z_pred = predictor(z_all, actions)
    print(f"Predicted tokens shape: {z_pred.shape}")