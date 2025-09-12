import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union, Optional, Tuple


def compute_patch_similarity(embeddings: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Compute pairwise similarity matrix between patches in an embedding tensor.
    
    Args:
        embeddings: Tensor of shape [p, d] where p is number of patches and d is embedding dimension
        normalize: If True, normalize embeddings before computing similarity (cosine similarity)
                   If False, compute dot product similarity
    
    Returns:
        Similarity matrix of shape [p, p] where element [i, j] is the similarity between patch i and j
    """
    if embeddings.dim() != 2:
        raise ValueError(f"Expected 2D tensor [p, d], got shape {embeddings.shape}")
    
    if normalize:
        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute pairwise similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.t())
    
    return similarity_matrix


def compute_patch_similarity_batch(embeddings: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Compute pairwise similarity matrix for batched embeddings.
    
    Args:
        embeddings: Tensor of shape [b, p, d] where b is batch size, p is number of patches, d is embedding dimension
        normalize: If True, normalize embeddings before computing similarity (cosine similarity)
                   If False, compute dot product similarity
    
    Returns:
        Similarity matrix of shape [b, p, p] where element [i, j, k] is the similarity 
        between patch j and k in batch i
    """
    if embeddings.dim() != 3:
        raise ValueError(f"Expected 3D tensor [b, p, d], got shape {embeddings.shape}")
    
    if normalize:
        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=2)
    
    # Compute pairwise similarity matrix for each batch
    similarity_matrix = torch.bmm(embeddings, embeddings.transpose(1, 2))
    
    return similarity_matrix


def apply_similarity_heatmap(
    image: Union[str, Image.Image, np.ndarray],
    similarity_matrix: Union[torch.Tensor, np.ndarray],
    patch_idx: int,
    patch_size: Optional[int] = None,
    alpha: float = 0.5,
    colormap: str = 'hot'
) -> Image.Image:
    """
    Apply a heatmap overlay on an image based on patch similarity scores.
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        similarity_matrix: Similarity matrix of shape [p, p] where p is number of patches
        patch_idx: Index of the patch to visualize similarities for (0 to p-1)
        patch_size: Size of each patch. If None, will be inferred from image size and number of patches
        alpha: Transparency of the heatmap overlay (0=transparent, 1=opaque)
        colormap: Matplotlib colormap name for the heatmap
    
    Returns:
        PIL Image with heatmap overlay applied
    """
    # Load image if necessary
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Convert similarity matrix to numpy if needed
    if isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = similarity_matrix.detach().cpu().numpy()
    
    # Get similarities for the selected patch
    if patch_idx >= similarity_matrix.shape[0]:
        raise ValueError(f"patch_idx {patch_idx} out of range for {similarity_matrix.shape[0]} patches")
    
    patch_similarities = similarity_matrix[patch_idx]
    
    # Determine grid dimensions
    num_patches = len(patch_similarities)
    grid_size = int(np.sqrt(num_patches))
    
    # Handle non-perfect square patch counts (e.g., DINOv3 with CLS + register tokens)
    if grid_size * grid_size != num_patches:
        # If we have N spatial patches + some special tokens, try common patterns
        if num_patches > 1:
            # Try removing different numbers of special tokens (1-5)
            for special_tokens in range(1, 6):
                spatial_patches = num_patches - special_tokens
                spatial_grid_size = int(np.sqrt(spatial_patches))
                
                if spatial_grid_size * spatial_grid_size == spatial_patches:
                    print(f"Detected {spatial_patches} spatial patches + {special_tokens} special tokens")
                    grid_size = spatial_grid_size
                    
                    # Remove special tokens from the beginning (CLS + register tokens come first)
                    patch_similarities = patch_similarities[special_tokens:]
                    num_patches = len(patch_similarities)
                    break
            else:
                # No perfect square found, pad to next larger square
                grid_size = int(np.sqrt(num_patches)) + 1
                print(f"Warning: {num_patches} patches is not a perfect square, using {grid_size}x{grid_size} grid (padding with zeros)")
        else:
            raise ValueError(f"Invalid number of patches: {num_patches}")
    
    # Pad with zeros if needed for non-perfect square
    if grid_size * grid_size > num_patches:
        padding_needed = grid_size * grid_size - num_patches
        patch_similarities = np.pad(patch_similarities, (0, padding_needed), 'constant', constant_values=0)
    
    # Get image dimensions
    img_width, img_height = image.size
    
    # Infer patch size if not provided
    if patch_size is None:
        patch_size = img_width // grid_size
        if patch_size * grid_size != img_width or patch_size * grid_size != img_height:
            print(f"Warning: Image size {img_width}x{img_height} doesn't divide evenly by grid {grid_size}x{grid_size}")
    
    # Reshape similarities to 2D grid
    similarity_grid = patch_similarities.reshape(grid_size, grid_size)
    
    # Create heatmap at the resolution of the patches
    heatmap_low_res = similarity_grid
    
    # Resize heatmap to match image size exactly using scipy.ndimage.zoom
    from scipy.ndimage import zoom
    
    # Calculate exact zoom factors to match target dimensions
    zoom_factor_h = img_height / grid_size
    zoom_factor_w = img_width / grid_size
    
    heatmap_high_res = zoom(heatmap_low_res, (zoom_factor_h, zoom_factor_w), order=1)
    
    # Ensure exact dimensions (crop or pad if needed due to floating point precision)
    if heatmap_high_res.shape[0] != img_height or heatmap_high_res.shape[1] != img_width:
        # Crop or pad to exact size
        heatmap_temp = np.zeros((img_height, img_width))
        
        # Copy data, cropping or padding as needed
        h_end = min(heatmap_high_res.shape[0], img_height)
        w_end = min(heatmap_high_res.shape[1], img_width)
        heatmap_temp[:h_end, :w_end] = heatmap_high_res[:h_end, :w_end]
        
        heatmap_high_res = heatmap_temp
    
    # Normalize heatmap to [0, 1]
    heatmap_min = heatmap_high_res.min()
    heatmap_max = heatmap_high_res.max()
    if heatmap_max > heatmap_min:
        heatmap_normalized = (heatmap_high_res - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap_normalized = np.zeros_like(heatmap_high_res)
    
    # Apply colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_normalized)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Convert to PIL Image
    heatmap_img = Image.fromarray(heatmap_colored)
    
    # Blend with original image
    image_array = np.array(image)
    blended = (1 - alpha) * image_array + alpha * heatmap_colored
    blended = blended.astype(np.uint8)
    
    return Image.fromarray(blended)