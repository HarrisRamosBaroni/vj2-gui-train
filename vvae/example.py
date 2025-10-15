"""
Example script showing how to use VideoVAE+ encoder and decoder separately as frozen code.
"""

import torch
from omegaconf import OmegaConf
from utils.common_utils import instantiate_from_config


def load_model(config_path='configs/config_16z.yaml', device='cuda'):
    """
    Load the VideoVAE+ model from config.

    Args:
        config_path: Path to config yaml file (config_16z.yaml or config_4z.yaml)
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        model: The loaded VAE model
    """
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    model = model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


def encode_video(model, video_tensor, device='cuda'):
    """
    Encode video to latent space.

    Args:
        model: The VAE model
        video_tensor: Video tensor of shape [B, C, T, H, W]
                     Values should be in range [-1, 1]
        device: Device to run on

    Returns:
        latent: Encoded latent tensor
    """
    with torch.no_grad():
        video_tensor = video_tensor.to(device)

        # Get posterior distribution from encoder
        posterior = model.encode(video_tensor)

        # Get latent (mean of distribution, no sampling)
        latent = posterior.mode()

    return latent


def decode_latent(model, latent, device='cuda'):
    """
    Decode latent back to video space.

    Args:
        model: The VAE model
        latent: Latent tensor from encoder
        device: Device to run on

    Returns:
        video: Decoded video tensor of shape [B, C, T, H, W]
               Values will be in range [-1, 1]
    """
    with torch.no_grad():
        latent = latent.to(device)

        # Decode latent to video
        video = model.decode(latent)

    return video


def main():
    """Main example demonstrating encoder/decoder usage."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model (use config_16z.yaml for 16-channel or config_4z.yaml for 4-channel)
    print("Loading model...")
    model = load_model('configs/config_16z.yaml', device=device)
    print(f"Model loaded successfully!")
    print(f"Latent channels: {model.encoder.z_channels}")

    # Create example video tensor
    # Shape: [batch_size, channels, time, height, width]
    batch_size = 1
    channels = 3
    num_frames = 16  # Must be divisible by 4
    height = 256
    width = 256

    print(f"\nCreating example video tensor: [{batch_size}, {channels}, {num_frames}, {height}, {width}]")
    video = torch.randn(batch_size, channels, num_frames, height, width) * 0.5  # Random video in [-1, 1]

    # Encode video to latent space
    print("\n=== ENCODING ===")
    latent = encode_video(model, video, device=device)
    print(f"Input video shape: {video.shape}")
    print(f"Encoded latent shape: {latent.shape}")
    print(f"Compression ratio (spatial): {(height * width) / (latent.shape[-2] * latent.shape[-1]):.2f}x")
    print(f"Compression ratio (temporal): {num_frames / latent.shape[2]:.2f}x")

    # Decode latent back to video space
    print("\n=== DECODING ===")
    reconstructed = decode_latent(model, latent, device=device)
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed video shape: {reconstructed.shape}")

    # Calculate reconstruction error
    mse = torch.mean((video.to(device) - reconstructed) ** 2).item()
    print(f"\nReconstruction MSE: {mse:.6f}")

    print("\n=== USAGE NOTES ===")
    print("- Input videos should be normalized to [-1, 1]")
    print("- Number of frames must be divisible by 4")
    print("- For 16z model: latent has 16 channels, temporal compression 4x")
    print("- For 4z model: latent has 4 channels, temporal compression 4x")
    print("- All parameters are frozen (requires_grad=False)")

    # Example: Using encoder and decoder separately with different inputs
    print("\n=== SEPARATE ENCODER/DECODER USAGE ===")

    # You can access encoder and decoder directly
    encoder = model.encoder
    decoder = model.decoder

    # Encode with encoder only
    with torch.no_grad():
        posterior = encoder(video.to(device))
        latent2 = posterior.mode()

    # Decode with decoder only
    with torch.no_grad():
        decoded = decoder(latent2)

    print(f"Direct encoder output: {latent2.shape}")
    print(f"Direct decoder output: {decoded.shape}")
    print("\nYou can now use encoder and decoder independently in your pipeline!")


if __name__ == "__main__":
    main()
