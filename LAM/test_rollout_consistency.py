"""
Test script to verify autoregressive rollout consistency.

This script tests if the autoregressive rollout produces the same result
when the same sequence is processed alone vs. in a batch.

Usage:
    python LAM/test_rollout_consistency.py --checkpoint_path <path_to_checkpoint>
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

from LAM.training import load_model
from LAM.generate_videos import autoregressive_rollout


def test_rollout_consistency(checkpoint_path: str, device: str = 'cuda'):
    """
    Test if autoregressive rollout is consistent across batch sizes.

    Strategy:
    1. Create a single test sequence
    2. Run autoregressive rollout alone (batch size 1)
    3. Run autoregressive rollout with it in a larger batch
    4. Compare rollout outputs - they should be identical
    """
    print("=" * 80)
    print("TESTING AUTOREGRESSIVE ROLLOUT CONSISTENCY")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model, _, checkpoint_info = load_model(checkpoint_path, device)
    model.eval()
    print(f"Model loaded (epoch {checkpoint_info['epoch']}, step {checkpoint_info['global_step']})")

    # Create test sequence
    B_test = 1
    T_test = 8
    C, H, W = 16, 64, 64

    print(f"\nCreating test sequence: [B={B_test}, T={T_test}, C={C}, H={H}, W={W}]")
    torch.manual_seed(42)
    test_sequence = torch.randn(B_test, T_test, C, H, W, device=device)

    # =========================================================================
    # Test 1: Rollout sequence alone (batch size 1)
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 1: Autoregressive rollout alone (batch size 1)")
    print("-" * 80)

    with torch.no_grad():
        pred_alone, mse_alone = autoregressive_rollout(model, test_sequence)

    print(f"Rollout output shape: {pred_alone.shape}")
    print(f"Per-step MSE: {[f'{m:.6f}' for m in mse_alone]}")

    # =========================================================================
    # Test 2: Rollout in larger batch
    # =========================================================================
    for batch_size in [2, 4, 8]:
        print(f"\n" + "-" * 80)
        print(f"TEST 2.{batch_size}: Autoregressive rollout in batch of size {batch_size}")
        print("-" * 80)

        # Create batch with test sequence as first item
        batch = torch.randn(batch_size, T_test, C, H, W, device=device)
        batch[0] = test_sequence[0]  # Place test sequence at position 0

        with torch.no_grad():
            pred_batch, mse_batch = autoregressive_rollout(model, batch)

        # Extract rollout for the first sequence (our test sequence)
        pred_batch_first = pred_batch[0:1]
        mse_batch_first = mse_batch  # MSE is computed across entire batch

        print(f"Rollout output shape: {pred_batch_first.shape}")
        print(f"Per-step MSE (entire batch): {[f'{m:.6f}' for m in mse_batch_first]}")

        # Compare rollout outputs
        rollout_diff = F.mse_loss(pred_alone, pred_batch_first).item()

        print(f"\nDifference (MSE) from batch-size-1 rollout:")
        print(f"  Rollout MSE: {rollout_diff:.10f}")

        # Compare per-step MSE for the test sequence specifically
        # We need to re-run rollout for just the first sequence to get per-step MSE
        with torch.no_grad():
            pred_first_only, mse_first_only = autoregressive_rollout(model, batch[0:1])

        print(f"  Per-step MSE (first sequence only): {[f'{m:.6f}' for m in mse_first_only]}")

        # Compare per-step MSE
        mse_diff = [abs(m1 - m2) for m1, m2 in zip(mse_alone, mse_first_only)]
        print(f"  Per-step MSE difference: {[f'{d:.10f}' for d in mse_diff]}")

        # Check if differences are negligible (< 1e-6)
        threshold = 1e-6
        if rollout_diff < threshold:
            print(f"  ✓ PASS: Rollout is consistent (difference < {threshold})")
        else:
            print(f"  ✗ FAIL: Rollout differs significantly!")
            print(f"  This suggests the autoregressive rollout is NOT batch-independent!")

            # Print frame-by-frame differences
            print(f"\n  Frame-by-frame MSE differences:")
            for t in range(T_test):
                frame_diff = F.mse_loss(pred_alone[0, t], pred_first_only[0, t]).item()
                print(f"    Frame {t}: {frame_diff:.10f}")

    # =========================================================================
    # Test 3: Check if rollout differs at different positions in batch
    # =========================================================================
    batch_size = 4
    print(f"\n" + "-" * 80)
    print(f"TEST 3: Autoregressive rollout at different positions (batch size {batch_size})")
    print("-" * 80)

    for position in range(batch_size):
        # Create batch with test sequence at different positions
        batch = torch.randn(batch_size, T_test, C, H, W, device=device)
        batch[position] = test_sequence[0]

        with torch.no_grad():
            pred_batch, _ = autoregressive_rollout(model, batch)

        # Extract rollout for our test sequence
        pred_at_position = pred_batch[position:position+1]

        # Compare to alone rollout
        rollout_diff = F.mse_loss(pred_alone, pred_at_position).item()

        print(f"\nPosition {position}:")
        print(f"  Rollout MSE difference: {rollout_diff:.10f}")

        threshold = 1e-6
        if rollout_diff < threshold:
            print(f"  ✓ PASS: Position {position} matches batch-size-1 rollout")
        else:
            print(f"  ✗ FAIL: Position {position} differs!")

            # Print frame-by-frame differences
            print(f"  Frame-by-frame MSE differences:")
            for t in range(T_test):
                frame_diff = F.mse_loss(pred_alone[0, t], pred_at_position[0, t]).item()
                print(f"    Frame {t}: {frame_diff:.10f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("If all tests pass, the autoregressive rollout is batch-independent.")
    print("If any test fails, the rollout is affected by batch size or position.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Test autoregressive rollout consistency')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to trained World Model checkpoint (.pt file)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    args = parser.parse_args()

    test_rollout_consistency(args.checkpoint_path, args.device)


if __name__ == '__main__':
    main()
