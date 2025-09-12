import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
try:
    import pillow_avif  # Adds AVIF support to PIL
except ImportError:
    pass  # AVIF support not available
import wandb
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gui_world_model.encoder import VJEPA2Wrapper, DINOv3Wrapper
from gui_world_model.utils.patch_similarity_mapping import (
    compute_patch_similarity,
    apply_similarity_heatmap
)


def process_single_image_vjepa2(image_path: str, encoder: VJEPA2Wrapper):
    """
    Process a single image through VJEPA2 by duplicating it into a tublet of 2 frames.
    Uses the same preprocessing as stage2_generate_trajectories.py
    
    Args:
        image_path: Path to the input image
        encoder: VJEPA2Wrapper instance
    
    Returns:
        embeddings: Patch embeddings from VJEPA2 [N, D]
        transformed_image: Preprocessed PIL image
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Stage 1: Resize to 256x256 (matching video preprocessing in stage2)
    resize_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()  # Convert to tensor and scale to [0, 1]
    ])
    
    image_tensor = resize_transform(image)  # [C, H, W] in range [0, 1]
    
    # Duplicate image to create a tublet of 2 frames (VJEPA2 uses tubelet_size=2)
    # Shape: [B=1, T=2, C=3, H=256, W=256]
    batch = image_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1, 1)
    
    # Stage 2: Apply normalization (matching stage2_generate_trajectories.py)
    # ImageNet normalization values
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Reshape for normalization: [B*T, C, H, W]
    B, T, C, H, W = batch.shape
    batch_flat = batch.view(B * T, C, H, W)
    batch_normalized = normalize(batch_flat)
    
    # Reshape back: [B, T, C, H, W]
    batch_normalized = batch_normalized.view(B, T, C, H, W)
    
    # Process through VJEPA2
    with torch.no_grad():
        # encoder expects [B, T, C, H, W] and returns [B, T_actual, N, D]
        # T_actual = T // tubelet_size = 2 // 2 = 1
        embeddings = encoder(batch_normalized)  # [1, 1, N, D]
    
    # Extract embeddings for the single temporal step
    # Shape: [N, D] where N is number of patches
    embeddings = embeddings[0, 0]  # Remove batch and temporal dimensions
    
    # Convert the transformed tensor back to PIL Image for visualization
    # image_tensor is [C, H, W] in range [0, 1]
    image_array = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    transformed_image = Image.fromarray(image_array)
    
    return embeddings, transformed_image


def process_single_image_dinov3(image_path: str, encoder: DINOv3Wrapper):
    """
    Process a single image through DINOv3.
    Uses direct 224x224 reshape preprocessing (no center crop).
    
    Args:
        image_path: Path to the input image
        encoder: DINOv3Wrapper instance
    
    Returns:
        embeddings: Patch embeddings from DINOv3 [N, D] where N = 1 CLS + 4 register + 196 spatial
        transformed_image: Preprocessed PIL image (224x224 reshaped)
    """
    # Use the wrapper method for consistent preprocessing
    embeddings, transformed_image = encoder.process_single_image_direct(image_path)
    
    return embeddings, transformed_image


def process_single_image(image_path: str, encoder, encoder_type: str):
    """
    Process a single image through the specified encoder.
    
    Args:
        image_path: Path to the input image
        encoder: Encoder instance (VJEPA2Wrapper or DINOv3Wrapper)
        encoder_type: Type of encoder ('vjepa2' or 'dinov3')
    
    Returns:
        embeddings: Patch embeddings [N, D]
        transformed_image: Preprocessed PIL image
    """
    if encoder_type == 'vjepa2':
        return process_single_image_vjepa2(image_path, encoder)
    elif encoder_type == 'dinov3':
        return process_single_image_dinov3(image_path, encoder)
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}. Use 'vjepa2' or 'dinov3'")


def visualize_all_patches(image_path: str, encoder_type: str = 'vjepa2', wandb_project: str = "patch_visualization"):
    """
    Generate and log patch similarity visualizations for all patches in an image.
    
    Args:
        image_path: Path to the input image
        encoder_type: Type of encoder to use ('vjepa2' or 'dinov3')
        wandb_project: Name of the wandb project to log to
    """
    # Validate encoder type
    if encoder_type not in ['vjepa2', 'dinov3']:
        raise ValueError(f"Unsupported encoder type: {encoder_type}. Use 'vjepa2' or 'dinov3'")
    
    # Initialize wandb with encoder-specific config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if encoder_type == 'vjepa2':
        config = {
            "image_path": image_path,
            "model": "vjepa2_vit_large",
            "encoder_type": encoder_type,
            "image_size": 256,
            "tubelet_size": 2
        }
        model_name = "vjepa2_vit_large"
    else:  # dinov3
        config = {
            "image_path": image_path,
            "model": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "encoder_type": encoder_type,
            "image_size": 256
        }
        model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    
    run = wandb.init(
        project=wandb_project,
        config=config
    )
    
    # Initialize encoder based on type
    print(f"Loading {encoder_type.upper()} encoder...")
    
    if encoder_type == 'vjepa2':
        encoder = VJEPA2Wrapper(
            device=device,
            model_name='vjepa2_vit_large',
            num_frames=2,  # Set to 2 since we're duplicating a single image
            image_size=256
        )
    else:  # dinov3
        encoder = DINOv3Wrapper(
            device=device,
            model_name='facebook/dinov3-vitl16-pretrain-lvd1689m',
            num_frames=1,  # DINOv3 processes single frames
            image_size=256
        )
    
    # Process image through the selected encoder
    print(f"Processing image with {encoder_type.upper()}: {image_path}")
    embeddings, transformed_image = process_single_image(image_path, encoder, encoder_type)
    print(f"Embeddings shape: {embeddings.shape}")  # Should be [N, D]
    print(f"Transformed image size: {transformed_image.size}")  # Should be (256, 256)
    
    # Compute patch similarity matrix
    print("Computing patch similarity matrix...")
    similarity_matrix = compute_patch_similarity(embeddings, normalize=True)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")  # Should be [N, N]
    
    # Calculate grid dimensions and handle CLS token
    num_patches = similarity_matrix.shape[0]
    grid_size = int(np.sqrt(num_patches))
    
    # Check for DINOv3 case (CLS + register tokens + spatial patches)
    has_cls_and_registers = False
    if grid_size * grid_size != num_patches:
        # DINOv3 typically has: 1 CLS + 4 register + 196 spatial patches = 201 total
        if encoder_type == 'dinov3' and num_patches == 201:
            # 196 spatial patches = 14x14 grid
            spatial_patches = 196
            spatial_grid_size = 14
            has_cls_and_registers = True
            grid_size = spatial_grid_size
            print(f"Detected DINOv3 structure: 1 CLS + 4 register + {spatial_patches} spatial patches")
        else:
            # Try generic detection: N spatial patches + some special tokens
            for special_tokens in range(1, 6):  # Try 1-5 special tokens
                spatial_patches = num_patches - special_tokens
                spatial_grid_size = int(np.sqrt(spatial_patches))
                if spatial_grid_size * spatial_grid_size == spatial_patches:
                    has_cls_and_registers = True
                    grid_size = spatial_grid_size
                    print(f"Detected {spatial_patches} spatial patches + {special_tokens} special tokens")
                    break
            
            if not has_cls_and_registers:
                print(f"Warning: {num_patches} patches is not a perfect square, will pad visualization")
    
    # Determine which patches to visualize
    if has_cls_and_registers and encoder_type == 'dinov3':
        # For DINOv3: indices 0 = CLS, 1-4 = register tokens, 5-200 = spatial patches
        cls_idx = 0
        register_indices = list(range(1, 5))  # Indices 1, 2, 3, 4
        spatial_patch_indices = list(range(5, num_patches))  # Spatial patches start from index 5
        patch_indices_to_visualize = spatial_patch_indices  # Only spatial patches for grid
        
        print(f"Will generate {len(spatial_patch_indices)} spatial patch visualizations")
        print(f"CLS token (index {cls_idx}) and register tokens (indices {register_indices}) will be visualized separately")
    else:
        # For VJEPA2 or when no special tokens detected
        spatial_patch_indices = list(range(num_patches))
        patch_indices_to_visualize = spatial_patch_indices
        print(f"Will generate {len(patch_indices_to_visualize)} patch visualizations")
    
    # Generate special token visualizations first if present
    if has_cls_and_registers and encoder_type == 'dinov3':
        print("Generating special token heatmaps...")
        
        # CLS token visualization
        cls_heatmap_img = apply_similarity_heatmap(
            image=transformed_image,
            similarity_matrix=similarity_matrix,
            patch_idx=cls_idx,
            patch_size=None,
            alpha=0.6,
            colormap='hot'
        )
        
        cls_heatmap_np = np.array(cls_heatmap_img)
        wandb.log({
            "patch_cls_heatmap": wandb.Image(
                cls_heatmap_np,
                caption="CLS token similarity heatmap (global context)"
            )
        })
        print("  CLS token (global context) - logged")
        
        # Register token visualizations
        for i, reg_idx in enumerate(register_indices):
            reg_heatmap_img = apply_similarity_heatmap(
                image=transformed_image,
                similarity_matrix=similarity_matrix,
                patch_idx=reg_idx,
                patch_size=None,
                alpha=0.6,
                colormap='viridis'  # Different colormap for register tokens
            )
            
            reg_heatmap_np = np.array(reg_heatmap_img)
            wandb.log({
                f"patch_register_{i+1}_heatmap": wandb.Image(
                    reg_heatmap_np,
                    caption=f"Register token {i+1} similarity heatmap"
                )
            })
            print(f"  Register token {i+1} (index {reg_idx}) - logged")
    
    # Generate heatmap visualizations for spatial patches
    print(f"Generating spatial patch heatmaps...")
    
    # Create a grid of all spatial patch heatmaps
    heatmap_images = []
    
    for i, patch_idx in enumerate(patch_indices_to_visualize):
        # Generate heatmap for this patch
        heatmap_img = apply_similarity_heatmap(
            image=transformed_image,  # Use the 256x256 transformed image
            similarity_matrix=similarity_matrix,
            patch_idx=patch_idx,
            patch_size=None,  # Will be inferred
            alpha=0.6,
            colormap='hot'
        )
        
        # Convert PIL image to numpy for wandb
        heatmap_np = np.array(heatmap_img)
        heatmap_images.append(heatmap_np)
        
        # Log individual spatial patch heatmap
        caption = f"Spatial patch {patch_idx} similarity heatmap"
        log_key = f"patch_{patch_idx:03d}_heatmap"
        
        wandb.log({
            log_key: wandb.Image(
                heatmap_np,
                caption=caption
            )
        })
        
        # Calculate patch position in grid (for spatial patches)
        if has_cls_and_registers and encoder_type == 'dinov3':
            # For DINOv3: spatial patches start from index 5, so subtract 5 for grid position
            spatial_idx = patch_idx - 5
        else:
            spatial_idx = patch_idx
            
        row = spatial_idx // grid_size
        col = spatial_idx % grid_size
        print(f"  Spatial patch {patch_idx:3d} (grid row {row:2d}, col {col:2d}) - logged")
    
    # Create a combined visualization grid
    print("Creating combined visualization grid...")
    
    # Arrange heatmaps in a grid
    rows = []
    for i in range(0, len(heatmap_images), grid_size):
        row_images = heatmap_images[i:i+grid_size]
        if len(row_images) < grid_size:
            # Pad with black images if needed
            h, w, c = row_images[0].shape
            row_images.extend([np.zeros((h, w, c), dtype=np.uint8)] * (grid_size - len(row_images)))
        rows.append(np.hstack(row_images))
    
    combined_grid = np.vstack(rows)
    
    # Log the combined grid with appropriate caption
    if has_cls_and_registers and encoder_type == 'dinov3':
        caption = f"All {len(patch_indices_to_visualize)} spatial patch similarity heatmaps (14×14 grid, CLS + register tokens logged separately)"
    else:
        caption = f"All {len(patch_indices_to_visualize)} patch similarity heatmaps"
    
    wandb.log({
        "all_patches_grid": wandb.Image(
            combined_grid,
            caption=caption
        )
    })
    
    # Log the transformed image for reference
    wandb.log({
        "transformed_image": wandb.Image(
            np.array(transformed_image),
            caption="Transformed input image (256x256)"
        )
    })
    
    # Also log the original image if we want to see the full resolution
    original_image = Image.open(image_path).convert('RGB')
    wandb.log({
        "original_image": wandb.Image(
            np.array(original_image),
            caption="Original input image (full resolution)"
        )
    })
    
    # Log the raw similarity matrix as a heatmap
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(similarity_matrix.cpu().numpy(), cmap='hot', aspect='auto')
    ax.set_title(f"Patch Similarity Matrix ({num_patches} patches)")
    ax.set_xlabel("Patch Index")
    ax.set_ylabel("Patch Index")
    plt.colorbar(im, ax=ax)
    
    wandb.log({
        "similarity_matrix": wandb.Image(fig, caption="Raw patch similarity matrix")
    })
    plt.close(fig)
    
    print(f"✅ Visualization complete! Logged {num_patches} patch heatmaps to wandb project '{wandb_project}'")
    
    # Finish wandb run
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Visualize patch similarities using VJEPA2 or DINOv3")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=['vjepa2', 'dinov3'],
        default='vjepa2',
        help="Encoder to use: 'vjepa2' (default) or 'dinov3'"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="patch_visualization",
        help="Wandb project name (default: patch_visualization)"
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    # Run visualization
    visualize_all_patches(args.image_path, args.encoder, args.project)


if __name__ == "__main__":
    main()