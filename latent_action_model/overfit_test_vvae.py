"""
Overfit test for VVAE Latent Action Model.
Tests if the model can overfit to a small batch of VVAE data.
"""

import torch
import torch.optim as optim
from tqdm import tqdm
import argparse
from pathlib import Path

from latent_action_model.vqvae import VVAELatentActionVAE
from latent_action_model.dataloader_vvae import create_vvae_dataloaders


def overfit_test(
    data_dir: str,
    num_iterations: int = 1000,
    batch_size: int = 4,
    sequence_length: int = 8,
    learning_rate: float = 1e-3,
    action_dim: int = 128,
    embed_dim: int = 512,
    encoder_depth: int = 3,
    decoder_depth: int = 3,
    kl_weight: float = 0.0,  # Turn off KL for pure reconstruction test
    device: str = "cuda"
):
    """
    Overfit test: train model on a single batch to verify it can memorize.

    Args:
        data_dir: Path to VVAE h5 files
        num_iterations: Number of training iterations
        batch_size: Batch size (small for overfit test)
        sequence_length: Number of latent frames per sequence
        learning_rate: Learning rate
        action_dim: Action latent dimension
        embed_dim: Transformer embedding dimension
        encoder_depth: Encoder transformer depth
        decoder_depth: Decoder transformer depth
        kl_weight: KL divergence weight (0 for overfit test)
        device: Device to use
    """
    print("="*60)
    print("VVAE LAM Overfit Test")
    print("="*60)

    # Create dataloader
    print("\nCreating dataloaders...")
    train_loader, val_loader, _ = create_vvae_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        sequence_length=sequence_length,
        stride_train=1,
        num_workers=0,  # Single worker for simplicity
        train_ratio=0.8,
        val_ratio=0.1
    )

    # Get a single batch to overfit on
    print("Loading single batch for overfit test...")
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch['sequence'].shape}")  # [B, T, 16, 64, 64]

    # Create model
    print("\nCreating VVAELatentActionVAE...")
    model = VVAELatentActionVAE(
        action_dim=action_dim,
        embed_dim=embed_dim,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        encoder_heads=8,
        decoder_heads=8,
        kl_weight=kl_weight,
        reconstruction_weight=1.0
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))

    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}

    # Training loop
    print(f"\nTraining for {num_iterations} iterations...")
    print("Target: MSE and MAE should decrease to near zero\n")

    model.train()
    best_mse = float('inf')
    best_mae = float('inf')

    pbar = tqdm(range(num_iterations), desc="Overfitting")
    for iteration in pbar:
        # Forward pass
        losses = model.compute_loss(
            batch,
            beta_schedule=kl_weight,
            rollout_horizon=1,  # No rollout for overfit test
            rollout_weight=0.0,
            rollout_prob=0.0
        )

        loss = losses["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        mse_loss = losses["mse_loss"].item()
        mae_loss = losses["mae_loss"].item()
        kl_loss = losses["kl_loss"].item()

        best_mse = min(best_mse, mse_loss)
        best_mae = min(best_mae, mae_loss)

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.6f}",
            "mse": f"{mse_loss:.6f}",
            "mae": f"{mae_loss:.6f}",
            "kl": f"{kl_loss:.6f}"
        })

        # Print detailed stats every 100 iterations
        if (iteration + 1) % 100 == 0:
            print(f"\nIteration {iteration + 1}:")
            print(f"  Total Loss:  {loss.item():.6f}")
            print(f"  MSE Loss:    {mse_loss:.6f} (best: {best_mse:.6f})")
            print(f"  MAE Loss:    {mae_loss:.6f} (best: {best_mae:.6f})")
            print(f"  KL Loss:     {kl_loss:.6f}")

    # Final evaluation
    print("\n" + "="*60)
    print("OVERFIT TEST RESULTS")
    print("="*60)

    model.eval()
    with torch.no_grad():
        final_losses = model.compute_loss(
            batch,
            beta_schedule=kl_weight,
            rollout_horizon=1,
            rollout_weight=0.0,
            rollout_prob=0.0
        )

        print(f"\nFinal Metrics:")
        print(f"  Total Loss:  {final_losses['loss'].item():.6f}")
        print(f"  MSE Loss:    {final_losses['mse_loss'].item():.6f}")
        print(f"  MAE Loss:    {final_losses['mae_loss'].item():.6f}")
        print(f"  KL Loss:     {final_losses['kl_loss'].item():.6f}")

        # Success criteria
        mse_threshold = 0.01
        mae_threshold = 0.05

        print(f"\nSuccess Criteria:")
        print(f"  MSE < {mse_threshold}: {'✓ PASS' if final_losses['mse_loss'].item() < mse_threshold else '✗ FAIL'}")
        print(f"  MAE < {mae_threshold}: {'✓ PASS' if final_losses['mae_loss'].item() < mae_threshold else '✗ FAIL'}")

        if final_losses['mse_loss'].item() < mse_threshold and final_losses['mae_loss'].item() < mae_threshold:
            print("\n🎉 OVERFIT TEST PASSED! Model can learn from data.")
        else:
            print("\n⚠️  OVERFIT TEST FAILED. Model may have issues.")
            print("Possible causes:")
            print("  - Learning rate too low")
            print("  - Model capacity too small")
            print("  - Data format mismatch")
            print("  - Adapter layers not working correctly")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="VVAE LAM Overfit Test")
    parser.add_argument("--data_dir", type=str, default="/home/kevin/work/vj2-gui/output_h5",
                       help="Path to VVAE h5 files")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--sequence_length", type=int, default=8, help="Sequence length")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--action_dim", type=int, default=128, help="Action dimension")
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # Check if data directory exists
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory {args.data_dir} not found")
        return

    overfit_test(
        data_dir=args.data_dir,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        action_dim=args.action_dim,
        embed_dim=args.embed_dim,
        device=args.device
    )


if __name__ == "__main__":
    main()
