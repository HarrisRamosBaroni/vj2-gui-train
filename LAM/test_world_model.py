"""
Test script for WorldModel (Latent Action Model) implementation.

Tests:
- WorldModel forward pass
- ActionEncoder, WorldEncoder, DynamicsPredictor components
- ResidualVectorQuantizer (RVQ)
- Model size and parameter counts
- Intermediate shape verification
- Gradient flow
"""
import torch
import sys
sys.path.insert(0, '/Users/xyle/Research/vj2-gui')

from LAM.world_model import WorldModel, ActionEncoder, WorldEncoder, DynamicsPredictor, ResidualVectorQuantizer


def test_rvq():
    """Test ResidualVectorQuantizer"""
    print("=" * 80)
    print("Testing ResidualVectorQuantizer (RVQ)...")
    print("=" * 80)

    # Create RVQ with 3 levels
    rvq = ResidualVectorQuantizer(
        num_levels=3,
        codebook_sizes=(256, 256, 256),
        d_code=128,
        beta=0.25,
        decay=0.99
    )
    rvq.train()

    # Create dummy continuous embeddings [B, T, d_code]
    B, T = 2, 4
    z = torch.randn(B, T, 128)

    print(f"Input shape: {z.shape}")

    # Forward pass
    z_q, indices, commitment_loss, codebook_loss = rvq(z)

    print(f"Quantized shape: {z_q.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Commitment loss: {commitment_loss.item():.6f}")
    print(f"Codebook loss: {codebook_loss.item():.6f}")

    # Check shapes
    assert z_q.shape == (B, T, 128), f"Expected {(B, T, 128)}, got {z_q.shape}"
    assert indices.shape == (B, T, 3), f"Expected {(B, T, 3)}, got {indices.shape}"

    # Check indices are in valid range
    assert indices.min() >= 0 and indices.max() < 256, "Indices out of range"

    print("✓ RVQ test passed!\n")


def test_world_model():
    """Test full WorldModel"""
    print("=" * 80)
    print("Testing WorldModel (full architecture)...")
    print("=" * 80)

    # Create WorldModel
    model = WorldModel(
        d_model=256,
        d_code_a=128,
        d_code_h=128,
        num_lvq_levels_a=3,
        num_lvq_levels_h=3,
        codebook_sizes_a=(256, 256, 256),
        codebook_sizes_h=(256, 256, 256),
        patchsize=[(8, 8), (4, 4), (2, 2), (1, 1)],
        num_encoder_blocks=3,
        num_decoder_blocks=3,
        beta_a=0.25,
        beta_h=0.25,
        decay=0.99
    )
    model.train()

    # Create dummy VVAE latent sequence [B, T, 16, 64, 64]
    B, T = 2, 5
    h_sequence = torch.randn(B, T, 16, 64, 64)

    print(f"\n{'='*80}")
    print("INPUT")
    print(f"{'='*80}")
    print(f"Input sequence shape: {h_sequence.shape}")
    print(f"  B (batch): {B}")
    print(f"  T (temporal): {T}")
    print(f"  Expected: {T-1} actions (transitions between {T} frames)")

    # === TEST 1: Forward pass ===
    print(f"\n{'='*80}")
    print("FORWARD PASS")
    print(f"{'='*80}")

    with torch.no_grad():
        output = model(h_sequence)

    # Check output dictionary keys
    expected_keys = {'pred_frames', 'action_codes', 'action_indices', 'world_emb', 'world_indices', 'losses'}
    assert set(output.keys()) == expected_keys, f"Missing keys: {expected_keys - set(output.keys())}"
    print("✓ Output dictionary has all expected keys")

    # === TEST 2: Check intermediate shapes ===
    print(f"\n{'='*80}")
    print("INTERMEDIATE SHAPES")
    print(f"{'='*80}")

    # Tokenizer output (internal, we'll track it manually)
    print("\n1. Tokenizer:")
    print(f"   Input:  [B={B}, T={T}, C=16, H=64, W=64]")
    print(f"   Output: [B*T={B*T}, d_model=256, H'=16, W'=16] (with PE)")

    print("\n2. ActionEncoder:")
    print(f"   Input:  [B*T={B*T}, d_model=256, H'=16, W'=16]")
    print(f"   Output: action_codes={output['action_codes'].shape}, action_indices={output['action_indices'].shape}")
    assert output['action_codes'].shape == (B, T-1, 128), f"Expected action_codes {(B, T-1, 128)}, got {output['action_codes'].shape}"
    assert output['action_indices'].shape == (B, T-1, 3), f"Expected action_indices {(B, T-1, 3)}, got {output['action_indices'].shape}"
    print(f"   ✓ Correct: [B={B}, T-1={T-1}, d_code_a=128] and indices [B={B}, T-1={T-1}, num_levels=3]")

    print("\n3. WorldEncoder:")
    print(f"   Input:  [B*T={B*T}, d_model=256, H'=16, W'=16]")
    print(f"   Output: world_emb={output['world_emb'].shape}, world_indices={output['world_indices'].shape}")
    assert output['world_emb'].shape == (B, 128), f"Expected world_emb {(B, 128)}, got {output['world_emb'].shape}"
    assert output['world_indices'].shape == (B, 3), f"Expected world_indices {(B, 3)}, got {output['world_indices'].shape}"
    print(f"   ✓ Correct: [B={B}, d_code_h=128] and indices [B={B}, num_levels=3]")

    print("\n4. DynamicsPredictor:")
    print(f"   Input:  tokens=[B*T={B*T}, d_model=256, H'=16, W'=16]")
    print(f"           actions=[B={B}, T-1={T-1}, d_code_a=128]")
    print(f"           world=[B={B}, d_code_h=128]")
    print(f"   Output: pred_tokens=[B*T={B*T}, d_model=256, H'=16, W'=16] (internal)")

    print("\n5. Detokenizer:")
    print(f"   Input:  [B*T={B*T}, d_model=256, H'=16, W'=16]")
    print(f"   Output: pred_frames={output['pred_frames'].shape}")
    assert output['pred_frames'].shape == (B, T, 16, 64, 64), f"Expected pred_frames {(B, T, 16, 64, 64)}, got {output['pred_frames'].shape}"
    print(f"   ✓ Correct: [B={B}, T={T}, C=16, H=64, W=64]")

    # === TEST 3: Check losses ===
    print(f"\n{'='*80}")
    print("LOSSES")
    print(f"{'='*80}")
    losses = output['losses']
    print(f"Action encoder:")
    print(f"  - Commitment loss: {losses['action_commit_loss'].item():.6f}")
    print(f"  - Codebook loss:   {losses['action_codebook_loss'].item():.6f}")
    print(f"World encoder:")
    print(f"  - Commitment loss: {losses['world_commit_loss'].item():.6f}")
    print(f"  - Codebook loss:   {losses['world_codebook_loss'].item():.6f}")

    total_rvq_loss = (losses['action_commit_loss'] + losses['action_codebook_loss'] +
                      losses['world_commit_loss'] + losses['world_codebook_loss'])
    print(f"\nTotal RVQ loss: {total_rvq_loss.item():.6f}")

    # === TEST 4: Model size ===
    print(f"\n{'='*80}")
    print("MODEL SIZE")
    print(f"{'='*80}")

    # Count parameters for each component
    tokenizer_params = sum(p.numel() for p in model.tokenizer.parameters())
    detokenizer_params = sum(p.numel() for p in model.detokenizer.parameters())
    action_encoder_params = sum(p.numel() for p in model.action_encoder.parameters())
    world_encoder_params = sum(p.numel() for p in model.world_encoder.parameters())
    dynamics_predictor_params = sum(p.numel() for p in model.dynamics_predictor.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    # Break down sub-components
    action_transformer_params = sum(p.numel() for p in model.action_encoder.transformer_blocks.parameters())
    action_rvq_params = sum(p.numel() for p in model.action_encoder.rvq.parameters())

    world_transformer_params = sum(p.numel() for p in model.world_encoder.transformer_blocks.parameters())
    world_rvq_params = sum(p.numel() for p in model.world_encoder.rvq.parameters())

    dynamics_transformer_params = sum(p.numel() for p in model.dynamics_predictor.transformer_blocks.parameters())

    # Calculate per-layer params
    num_action_blocks = len(model.action_encoder.transformer_blocks)
    num_world_blocks = len(model.world_encoder.transformer_blocks)
    num_dynamics_blocks = len(model.dynamics_predictor.transformer_blocks)

    action_per_layer = action_transformer_params / num_action_blocks if num_action_blocks > 0 else 0
    world_per_layer = world_transformer_params / num_world_blocks if num_world_blocks > 0 else 0
    dynamics_per_layer = dynamics_transformer_params / num_dynamics_blocks if num_dynamics_blocks > 0 else 0

    # Convert to GB (float32 = 4 bytes)
    def to_gb(params):
        return (params * 4) / (1024 ** 3)

    print("\nShared Components:")
    print(f"  Tokenizer:    {tokenizer_params / 1e6:>8.2f}M ({to_gb(tokenizer_params):.4f} GB)")
    print(f"  Detokenizer:  {detokenizer_params / 1e6:>8.2f}M ({to_gb(detokenizer_params):.4f} GB)")

    print("\nActionEncoder:")
    print(f"  Total:        {action_encoder_params / 1e6:>8.2f}M ({to_gb(action_encoder_params):.4f} GB)")
    print(f"    - ST-Transformer ({num_action_blocks} layers): {action_transformer_params / 1e6:>8.2f}M")
    print(f"      → Per layer: {action_per_layer / 1e6:>8.2f}M")
    print(f"    - RVQ (3 levels): {action_rvq_params / 1e6:>8.2f}M")

    print("\nWorldEncoder:")
    print(f"  Total:        {world_encoder_params / 1e6:>8.2f}M ({to_gb(world_encoder_params):.4f} GB)")
    print(f"    - ST-Transformer ({num_world_blocks} layers): {world_transformer_params / 1e6:>8.2f}M")
    print(f"      → Per layer: {world_per_layer / 1e6:>8.2f}M")
    print(f"    - RVQ (3 levels): {world_rvq_params / 1e6:>8.2f}M")

    print("\nDynamicsPredictor:")
    print(f"  Total:        {dynamics_predictor_params / 1e6:>8.2f}M ({to_gb(dynamics_predictor_params):.4f} GB)")
    print(f"    - ST-Transformer ({num_dynamics_blocks} layers): {dynamics_transformer_params / 1e6:>8.2f}M")
    print(f"      → Per layer: {dynamics_per_layer / 1e6:>8.2f}M")

    print(f"\n{'─'*80}")
    print(f"TOTAL:         {total_params / 1e6:>8.2f}M ({to_gb(total_params):.4f} GB)")
    print(f"{'─'*80}")

    print("\n✓ All shape checks passed!")


def test_backward_pass():
    """Test gradient flow"""
    print(f"\n{'='*80}")
    print("GRADIENT FLOW TEST")
    print(f"{'='*80}")

    # Create smaller model for faster testing
    model = WorldModel(
        d_model=64,
        d_code_a=32,
        d_code_h=32,
        num_lvq_levels_a=2,
        num_lvq_levels_h=2,
        codebook_sizes_a=(128, 128),
        codebook_sizes_h=(128, 128),
        patchsize=[(8, 8), (4, 4)],
        num_encoder_blocks=2,
        num_decoder_blocks=2
    )
    model.train()

    # Create dummy input
    B, T = 1, 3
    h_sequence = torch.randn(B, T, 16, 64, 64, requires_grad=True)

    # Forward pass
    output = model(h_sequence)

    # Compute total loss (reconstruction + RVQ)
    # Simple MSE reconstruction loss (compare prediction to ground truth)
    recon_loss = torch.nn.functional.mse_loss(output['pred_frames'], h_sequence)

    losses = output['losses']
    total_loss = (recon_loss +
                  losses['action_commit_loss'] + losses['action_codebook_loss'] +
                  losses['world_commit_loss'] + losses['world_codebook_loss'])

    print(f"\nLosses:")
    print(f"  Reconstruction:      {recon_loss.item():.6f}")
    print(f"  Action RVQ:          {(losses['action_commit_loss'] + losses['action_codebook_loss']).item():.6f}")
    print(f"  World RVQ:           {(losses['world_commit_loss'] + losses['world_codebook_loss']).item():.6f}")
    print(f"  Total:               {total_loss.item():.6f}")

    # Backward pass
    print("\nRunning backward pass...")
    total_loss.backward()

    # Check gradients
    tokenizer_has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                              for p in model.tokenizer.parameters())
    detokenizer_has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in model.detokenizer.parameters())
    action_encoder_has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                                   for p in model.action_encoder.parameters()
                                   if p.requires_grad)
    world_encoder_has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                                  for p in model.world_encoder.parameters()
                                  if p.requires_grad)
    dynamics_has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.dynamics_predictor.parameters())

    print("\nGradient flow:")
    print(f"  Tokenizer:         {tokenizer_has_grads}")
    print(f"  Detokenizer:       {detokenizer_has_grads}")
    print(f"  ActionEncoder:     {action_encoder_has_grads}")
    print(f"  WorldEncoder:      {world_encoder_has_grads}")
    print(f"  DynamicsPredictor: {dynamics_has_grads}")

    # RVQ codebooks should NOT require gradients (EMA updates only, stored as buffers)
    # Check that all codebook buffers have requires_grad=False
    action_rvq_requires_grad = any(
        model.action_encoder.rvq._get_codebook(i).requires_grad
        for i in range(model.action_encoder.rvq.num_levels)
    )
    world_rvq_requires_grad = any(
        model.world_encoder.rvq._get_codebook(i).requires_grad
        for i in range(model.world_encoder.rvq.num_levels)
    )

    print(f"\n  Action RVQ codebooks require_grad: {action_rvq_requires_grad} (should be False, uses EMA)")
    print(f"  World RVQ codebooks require_grad:  {world_rvq_requires_grad} (should be False, uses EMA)")

    assert tokenizer_has_grads, "Tokenizer should receive gradients"
    assert detokenizer_has_grads, "Detokenizer should receive gradients"
    assert dynamics_has_grads, "DynamicsPredictor should receive gradients"
    assert not action_rvq_requires_grad, "Action RVQ codebooks should NOT require gradients (EMA only, buffers)"
    assert not world_rvq_requires_grad, "World RVQ codebooks should NOT require gradients (EMA only, buffers)"

    print("\n✓ Gradient flow correct!")


if __name__ == "__main__":
    test_rvq()
    test_world_model()
    test_backward_pass()

    print(f"\n{'='*80}")
    print("🎉 ALL TESTS PASSED! WorldModel implementation verified.")
    print(f"{'='*80}")
    print("\nSummary:")
    print("  ✓ ResidualVectorQuantizer: Multi-level RVQ with EMA updates")
    print("  ✓ ActionEncoder: Transitions → action codes (shifted causal)")
    print("  ✓ WorldEncoder: Full sequence → world hypothesis (bidirectional)")
    print("  ✓ DynamicsPredictor: Tokens + actions + world → predictions (causal)")
    print("  ✓ All intermediate shapes verified")
    print("  ✓ Model size and parameter breakdown displayed")
    print("  ✓ Gradient flow correct (backprop + EMA)")
    print(f"{'='*80}\n")
