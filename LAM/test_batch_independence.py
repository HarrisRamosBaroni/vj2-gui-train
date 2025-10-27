"""
Test script to verify batch independence in WorldModel.

This script tests whether the model outputs for a given sequence are independent
of other sequences in the batch. If batch mixing is happening, the same sequence
processed in different batches will produce different outputs.

Usage:
    python LAM/test_batch_independence.py --checkpoint_path <path_to_checkpoint>
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

from LAM.training import load_model


def test_batch_independence(checkpoint_path: str, device: str = 'cuda'):
    """
    Test if model outputs are batch-independent.

    Strategy:
    1. Create a single test sequence
    2. Process it alone (batch size 1)
    3. Process it with dummy sequences in a larger batch
    4. Compare outputs - they should be identical if batch-independent
    """
    print("=" * 80)
    print("TESTING BATCH INDEPENDENCE")
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
    # Test 1: Process sequence alone (batch size 1)
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 1: Processing sequence alone (batch size 1)")
    print("-" * 80)

    with torch.no_grad():
        output_alone = model(test_sequence)

    pred_frames_alone = output_alone['pred_frames']
    action_codes_alone = output_alone['action_codes']
    world_emb_alone = output_alone['world_emb']

    print(f"Output shapes:")
    print(f"  pred_frames: {pred_frames_alone.shape}")
    print(f"  action_codes: {action_codes_alone.shape}")
    print(f"  world_emb: {world_emb_alone.shape}")

    # =========================================================================
    # Test 2: Process sequence in larger batch (as first item)
    # =========================================================================
    for batch_size in [2, 4, 8]:
        print(f"\n" + "-" * 80)
        print(f"TEST 2.{batch_size}: Processing in batch of size {batch_size} (as first item)")
        print("-" * 80)

        # Create batch with test sequence as first item
        batch = torch.randn(batch_size, T_test, C, H, W, device=device)
        batch[0] = test_sequence[0]  # Place test sequence at position 0

        with torch.no_grad():
            output_batch = model(batch)

        # Extract outputs for the first sequence (our test sequence)
        pred_frames_batch = output_batch['pred_frames'][0:1]
        action_codes_batch = output_batch['action_codes'][0:1]
        world_emb_batch = output_batch['world_emb'][0:1]

        print(f"Output shapes:")
        print(f"  pred_frames: {pred_frames_batch.shape}")
        print(f"  action_codes: {action_codes_batch.shape}")
        print(f"  world_emb: {world_emb_batch.shape}")

        # Compare outputs
        pred_diff = F.mse_loss(pred_frames_alone, pred_frames_batch).item()
        action_diff = F.mse_loss(action_codes_alone, action_codes_batch).item()
        world_diff = F.mse_loss(world_emb_alone, world_emb_batch).item()

        print(f"\nDifferences (MSE) from batch-size-1 output:")
        print(f"  pred_frames MSE: {pred_diff:.10f}")
        print(f"  action_codes MSE: {action_diff:.10f}")
        print(f"  world_emb MSE: {world_diff:.10f}")

        # Check if differences are negligible (< 1e-6)
        threshold = 1e-6
        if pred_diff < threshold and action_diff < threshold and world_diff < threshold:
            print(f"  ✓ PASS: Outputs are batch-independent (differences < {threshold})")
        else:
            print(f"  ✗ FAIL: Outputs differ significantly!")
            print(f"  This suggests BATCH MIXING is occurring!")

    # =========================================================================
    # Test 3: Process sequence in larger batch (as different positions)
    # =========================================================================
    batch_size = 4
    print(f"\n" + "-" * 80)
    print(f"TEST 3: Processing in batch of size {batch_size} at different positions")
    print("-" * 80)

    for position in range(batch_size):
        # Create batch with test sequence at different positions
        batch = torch.randn(batch_size, T_test, C, H, W, device=device)
        batch[position] = test_sequence[0]

        with torch.no_grad():
            output_batch = model(batch)

        # Extract outputs for our test sequence
        pred_frames_batch = output_batch['pred_frames'][position:position+1]
        action_codes_batch = output_batch['action_codes'][position:position+1]
        world_emb_batch = output_batch['world_emb'][position:position+1]

        # Compare to alone output
        pred_diff = F.mse_loss(pred_frames_alone, pred_frames_batch).item()
        action_diff = F.mse_loss(action_codes_alone, action_codes_batch).item()
        world_diff = F.mse_loss(world_emb_alone, world_emb_batch).item()

        print(f"\nPosition {position}:")
        print(f"  pred_frames MSE: {pred_diff:.10f}")
        print(f"  action_codes MSE: {action_diff:.10f}")
        print(f"  world_emb MSE: {world_diff:.10f}")

        threshold = 1e-6
        if pred_diff < threshold and action_diff < threshold and world_diff < threshold:
            print(f"  ✓ PASS: Position {position} is batch-independent")
        else:
            print(f"  ✗ FAIL: Position {position} shows batch mixing!")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("If all tests pass, the model is batch-independent.")
    print("If any test fails, there is batch mixing occurring somewhere in the model.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Test batch independence of WorldModel')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to trained World Model checkpoint (.pt file)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    args = parser.parse_args()

    test_batch_independence(args.checkpoint_path, args.device)


if __name__ == '__main__':
    main()
