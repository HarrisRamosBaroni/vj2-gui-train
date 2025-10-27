# Test Script Implementation Guide

This guide explains the structure of `test_trained_world_model.py` and how to implement each evaluation metric one by one.

## Overall Structure

The test script is organized into:
1. **Model Loading** ✅ (Complete)
2. **Dataset Loading** ✅ (Complete)
3. **Wandb Setup** ✅ (Complete)
4. **Evaluation Loop** ✅ (Skeleton complete, metrics need implementation)
5. **Aggregate Statistics** ✅ (Complete)

## Evaluation Metrics

Each metric has its own function with detailed implementation notes. Implement them in this order:

### 1. Teacher Forcing MSE (Easiest)
**Function:** `compute_teacher_forcing_mse()`
**Reference:** `training.py` validation loop (lines ~710-720)

**Implementation steps:**
```python
def compute_teacher_forcing_mse(model: WorldModel, h_sequence: torch.Tensor):
    # 1. Run forward pass
    output = model(h_sequence)

    # 2. Extract predictions and losses
    pred_frames = output['pred_frames']
    rvq_losses = output['losses']

    # 3. Compute reconstruction MSE
    tf_mse = F.mse_loss(pred_frames, h_sequence)

    # 4. Return metrics
    return {
        'tf_mse': tf_mse.item(),
        'rvq_losses': rvq_losses  # Optional: for reference
    }
```

**Expected output:**
- Single scalar MSE value per batch
- Should match validation MSE from training

---

### 2. Action Sensitivity / dPSNR (Easiest - Already Implemented!)
**Function:** `compute_action_sensitivity()`
**Reference:** `utils.py` - `action_sensitivity_dsnr()` function

**Implementation steps:**
```python
def compute_action_sensitivity(model: WorldModel, h_sequence: torch.Tensor):
    # This is already implemented in utils.py!
    dsnr, psnr_seq, psnr_rand = action_sensitivity_dsnr(model, h_sequence)

    return {
        'dsnr': dsnr,
        'psnr_seq': psnr_seq,
        'psnr_rand': psnr_rand,
    }
```

**Expected output:**
- dsnr: Difference between PSNR with correct vs random actions
- psnr_seq: PSNR with correct actions
- psnr_rand: PSNR with random actions

---

### 3. Codebook Usage (Medium)
**Function:** `compute_codebook_usage()`
**Reference:** `training.py` lines ~755-763

**Implementation steps:**
```python
def compute_codebook_usage(model: WorldModel, h_sequence: torch.Tensor):
    # 1. Run forward pass to get indices
    output = model(h_sequence)

    # 2. Extract indices
    action_indices = output['action_indices']  # [B, T-1, num_lvq_levels]
    world_indices = output['world_indices']    # [B, num_lvq_levels]

    # 3. Use codebook_usage() from utils.py
    action_usage = codebook_usage(action_indices)  # [num_lvq_levels]
    world_usage = codebook_usage(world_indices.unsqueeze(1))  # Add T dim

    # 4. Convert to percentages (optional)
    # Get codebook sizes from model
    codebook_sizes_a = model.action_encoder.rvq.codebook_sizes
    codebook_sizes_h = model.world_encoder.rvq.codebook_sizes

    action_usage_pct = [action_usage[i] / codebook_sizes_a[i] * 100 for i in range(len(action_usage))]
    world_usage_pct = [world_usage[i] / codebook_sizes_h[i] * 100 for i in range(len(world_usage))]

    return {
        'action_usage': action_usage,
        'world_usage': world_usage,
        'action_usage_pct': action_usage_pct,
        'world_usage_pct': world_usage_pct,
    }
```

**Expected output:**
- Tensor of unique codes used per RVQ level for action and world encoders

---

### 4. Diagonal Attention (Medium-Hard)
**Function:** `compute_diagonal_attention()`
**Reference:** `training.py` lines ~769-800

**Implementation steps:**
```python
def compute_diagonal_attention(model: WorldModel, h_sequence: torch.Tensor):
    B, T, C, H, W = h_sequence.shape

    # 1. Tokenize sequence
    tokens, _, _ = model.tokenizer(h_sequence)

    # 2. Run action_encoder with return_attention=True
    _, _, _, _, attention_weights = model.action_encoder(
        tokens, B, T - 1, return_attention=True
    )

    # 3. Process attention weights for each block
    # attention_weights: list of lists [num_blocks][num_heads]
    block_diagonal_scores = []

    for block_attn in attention_weights:
        if len(block_attn) > 0:
            # Get first head's attention: [B, T*patches, T*patches]
            attn_first_head = block_attn[0]

            # Average over batch
            attn_avg = attn_first_head.mean(dim=0)  # [T*patches, T*patches]

            # Reshape to temporal structure
            num_patches = attn_avg.shape[0] // T
            attn_temporal = attn_avg.view(T, num_patches, T, num_patches)
            attn_temporal = attn_temporal.mean(dim=(1, 3))  # [T, T]

            # Compute diagonal score using utils function
            score = diagonal_attention_score(attn_temporal.unsqueeze(0))
            block_diagonal_scores.append(score)

    return {
        'diagonal_attention_per_block': block_diagonal_scores
    }
```

**Expected output:**
- List of scores (0-1) for each transformer block
- Should be close to 1.0 if model learned proper shifted causal attention

---

### 5. Real Rollout Error (HARDEST - SPECIAL ATTENTION NEEDED!)
**Function:** `compute_real_rollout_error()`
**No direct reference - this is a new evaluation method**

**Implementation steps:**

```python
def compute_real_rollout_error(model: WorldModel, h_sequence: torch.Tensor):
    B, T, C, H, W = h_sequence.shape
    device = h_sequence.device

    # =====================================================================
    # Step 1: Extract real actions and world embedding (run once on GT)
    # =====================================================================
    # Tokenize full sequence
    tokens, out_h, out_w = model.tokenizer(h_sequence)  # [B*T, d_model, out_h, out_w]

    # Get real actions from action encoder
    action_codes, _, _, _ = model.action_encoder(tokens, B, T - 1)  # [B, T-1, d_code_a]

    # Get real world embedding from world encoder
    world_emb, _, _, _ = model.world_encoder(tokens, B, T)  # [B, d_code_h]

    # =====================================================================
    # Step 2: Initialize autoregressive generation
    # =====================================================================
    # Start with ground truth frame 0 tokens
    # We need to maintain a sequence of tokens: [B*T, d_model, out_h, out_w]
    # Initially, only frame 0 is from GT, rest will be filled with predictions

    # Create a copy of tokens to modify during rollout
    rollout_tokens = tokens.clone()

    # Track MSE per step
    mse_per_step = []

    # =====================================================================
    # Step 3: Autoregressive generation loop
    # =====================================================================
    for step in range(1, T):  # Predict frames 1, 2, ..., T-1
        # Current action for this step (action from frame step-1 to frame step)
        action_for_step = action_codes[:, step - 1:step, :]  # [B, 1, d_code_a]

        # Predict next frame using dynamics predictor
        # Note: dynamics_predictor expects full sequence context
        # We pass rollout_tokens which has GT frame 0 + predicted frames 1..step-1
        pred_tokens = model.dynamics_predictor(
            rollout_tokens,
            action_for_step,  # Just the action for this step
            world_emb,
            B,
            step + 1  # Context length: frame 0 + predictions up to step
        )

        # Extract predicted tokens for frame 'step'
        # pred_tokens is [B*(step+1), d_model, out_h, out_w]
        # We want the last frame's tokens: indices [B*step:B*(step+1)]
        pred_frame_tokens = pred_tokens[B*step:B*(step+1), :, :, :]  # [B, d_model, out_h, out_w]

        # Decode to get predicted frame
        pred_frame = model.detokenizer(pred_frame_tokens)  # [B, C, H, W]

        # Compute MSE against ground truth frame at this step
        gt_frame = h_sequence[:, step, :, :, :]  # [B, C, H, W]
        step_mse = F.mse_loss(pred_frame, gt_frame)
        mse_per_step.append(step_mse.item())

        # CRITICAL: Replace tokens for frame 'step' with predicted tokens
        # This makes the next prediction use the predicted frame as context
        rollout_tokens[B*step:B*(step+1), :, :, :] = pred_frame_tokens.detach()

    # =====================================================================
    # Step 4: Return results
    # =====================================================================
    avg_mse = sum(mse_per_step) / len(mse_per_step)

    return {
        'rollout_mse_per_step': mse_per_step,  # List of MSE values
        'rollout_mse_avg': avg_mse,
    }
```

**CRITICAL NOTES FOR ROLLOUT:**
1. **Context management**: You need to carefully manage which tokens are GT vs predicted
2. **Action indexing**: action_codes[:, step-1] gives action from frame step-1 → step
3. **Token replacement**: After prediction, replace the tokens for the predicted frame
4. **Sequence length**: dynamics_predictor needs to know the current sequence length
5. **Potential issues**:
   - May need to re-tokenize predicted frames instead of using predicted tokens directly
   - Dynamics predictor might expect full T-length sequences (check implementation)
   - Action codes might need to be expanded/repeated properly

**This is the trickiest part and will likely need debugging!**

---

## Implementation Order

Recommend implementing in this order:

1. ✅ **Action Sensitivity** - Trivial (just call existing function)
2. ✅ **Teacher Forcing MSE** - Easy (similar to training validation)
3. ✅ **Codebook Usage** - Medium (straightforward, uses existing utility)
4. ✅ **Diagonal Attention** - Medium (requires attention weight processing)
5. ⚠️ **Real Rollout Error** - Hard (autoregressive generation, needs careful implementation)

## Testing Each Metric

After implementing each metric, test with:

```bash
python LAM/test_trained_world_model.py \
    --checkpoint_path /path/to/checkpoint.pt \
    --data_dir /path/to/data \
    --manifest_path /path/to/manifest.json \
    --split val \
    --split_pct 0.1 \
    --batch_size 4 \
    --no_wandb  # Test without wandb first
```

Check:
- No errors/exceptions
- Reasonable metric values (compare with training validation metrics)
- Proper logging output

## Wandb Logging Structure

Once all metrics work, wandb will log:

**Per-batch metrics:**
- `Test/tf_mse` - Teacher forcing MSE per batch

**Aggregate metrics:**
- `Test_Summary/tf_mse_mean/min/max` - TF MSE statistics
- `Test_Rollout/step_X_mse` - MSE at each rollout step
- `Test_Summary/rollout_mse_overall` - Average across all steps
- `Test_DiagonalAttention/block_X` - Score per transformer block
- `Test_CodebookUsage/action_level_X` - Action codebook usage per level
- `Test_CodebookUsage/world_level_X` - World codebook usage per level
- `Test_Summary/dsnr/psnr_seq/psnr_rand` - Action sensitivity metrics

---

## Next Steps

Start with the easiest metric (Action Sensitivity) and work your way up. Feel free to ask for help when implementing each one, especially the rollout error!
