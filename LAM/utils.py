"""
Utility functions and metrics for LAM.
"""
import torch
import torch.nn.functional as F


def codebook_usage(indices):
    """
    Count number of unique codebook entries used in a batch.

    Args:
        indices: [B, T, num_levels] - codebook indices for each RVQ level

    Returns:
        usage: [num_levels] - number of unique codes used for each codebook level
    """
    B, T, num_codes = indices.shape

    usage = []
    for i in range(num_codes):  # Dynamic: works with any number of RVQ levels
        indices_i = indices[:, :, i].flatten()  # [B*T]
        unique_codes = torch.unique(indices_i).numel()
        usage.append(unique_codes)

    return torch.tensor(usage)


def diagonal_attention_score(attention_weights):
    """
    Compute diagonalization score for Action Encoder attention.

    The Action Encoder uses shifted causal masking (diagonal=2) to see one step ahead.
    We measure how much attention is concentrated on:
    - Main diagonal (i, i): frame attending to itself
    - Upper diagonal (i, i+1): frame attending one step ahead

    Args:
        attention_weights: [num_heads, T, T] - attention weights from one transformer block

    Returns:
        score: scalar - diagonalization score (0 to 1)
                       1.0 = perfectly concentrated on main + upper diagonal
    """
    num_heads, T, _ = attention_weights.shape

    # Collect values from main and upper diagonals across all heads
    diagonal_values = []

    for head in range(num_heads):
        attn = attention_weights[head]  # [T, T]

        # Main diagonal: (i, i)
        for i in range(T):
            diagonal_values.append(attn[i, i].item())

        # Upper diagonal: (i, i+1) - one step ahead
        for i in range(T - 1):
            diagonal_values.append(attn[i, i + 1].item())

    # Average across all diagonal positions and all heads
    score = sum(diagonal_values) / len(diagonal_values)

    return score


def action_sensitivity_dsnr(world_model, h_sequence, eval_frame_idx=4):
    """
    Compute action sensitivity metric: dPSNR = PSNR_seq - PSNR_rand.

    Following the specification from description.md:
    - PSNR_seq: PSNR with correct actions (from encoder)
    - PSNR_rand: PSNR with random actions (sampled from categorical distribution)
    - dPSNR = (1/B) * sum_{b=1}^{B} [PSNR_seq_b - PSNR_rand_b]
    - PSNR = 10 * log10(1 / MSE)
    - Evaluated at a specific frame index (Genie uses t=4)

    Higher dPSNR means model is more sensitive to correct actions.

    Args:
        world_model: WorldModel instance
        h_sequence: [B, T, 16, 64, 64] - VVAE latent sequence (normalized to [-1, 1])
        eval_frame_idx: Frame index to evaluate (default: 4, following Genie)

    Returns:
        tuple: (dsnr, psnr_seq, psnr_rand) - all scalar values for logging
    """
    with torch.no_grad():
        B, T, C, H, W = h_sequence.shape
        device = h_sequence.device

        # Ensure eval_frame_idx is valid
        if eval_frame_idx >= T:
            eval_frame_idx = T - 1  # Use last frame if eval_frame_idx is too large

        # =========================================================================
        # Step 1: Tokenize once (shared for both seq and rand)
        # =========================================================================
        tokens, _, _ = world_model.tokenizer(h_sequence)  # [B*T, d_model, H', W']

        # =========================================================================
        # Step 2: Get action codes and world embedding
        # =========================================================================
        action_codes, _, _, _ = world_model.action_encoder(tokens, B, T - 1)  # [B, T-1, d_code_a]
        world_emb, _, _, _ = world_model.world_encoder(tokens, B, T)  # [B, d_code_h]

        # Per description.md: Drop last frame, pass GT[0:T-1] to predictor for perfect alignment
        tokens_input = tokens[:B * (T - 1)]  # [B*(T-1), d_model, H', W']

        # =========================================================================
        # Step 3: PSNR_seq - Predict with correct actions
        # =========================================================================
        pred_tokens_seq = world_model.dynamics_predictor(tokens_input, action_codes, world_emb, B, T - 1)
        pred_frames_seq = world_model.detokenizer(pred_tokens_seq).view(B, T - 1, C, H, W)

        # Extract frame at eval_frame_idx
        # pred_frames_seq predicts frames [1:T], so eval_frame_idx=4 corresponds to pred_frames_seq[:, 3]
        # Adjust index: eval_frame_idx points to GT frame, but predictions are for frames [1:T]
        pred_idx = eval_frame_idx - 1  # Frame at eval_frame_idx is predicted at position eval_frame_idx-1
        pred_frame_seq = pred_frames_seq[:, pred_idx, :, :, :]  # [B, C, H, W]
        gt_frame = h_sequence[:, eval_frame_idx, :, :, :]

        # Compute MSE per sample: [B]
        mse_seq_per_sample = ((pred_frame_seq - gt_frame) ** 2).view(B, -1).mean(dim=1)

        # =========================================================================
        # Step 4: PSNR_rand - Predict with random actions sampled from codebook
        # =========================================================================
        # Sample random actions from categorical distribution (codebook)
        # action_codes shape: [B, T-1, d_code_a]
        num_levels = world_model.action_encoder.rvq.num_levels

        # Sample random indices for each level
        action_codes_random = torch.zeros_like(action_codes)  # [B, T-1, d_code_a]

        for level_idx in range(num_levels):
            # Get codebook for this level: [codebook_size, d_code_a]
            codebook = world_model.action_encoder.rvq._get_codebook(level_idx)
            codebook_size = world_model.action_encoder.rvq.codebook_sizes[level_idx]

            # Sample random indices: [B, T-1]
            random_idx = torch.randint(0, codebook_size, (B, T - 1), device=device)

            # Look up embeddings: [B, T-1, d_code_a]
            level_codes = F.embedding(random_idx, codebook)

            # Add to total (RVQ sums across levels)
            action_codes_random += level_codes

        # Predict with random actions
        pred_tokens_rand = world_model.dynamics_predictor(tokens_input, action_codes_random, world_emb, B, T - 1)
        pred_frames_rand = world_model.detokenizer(pred_tokens_rand).view(B, T - 1, C, H, W)

        # Extract frame at eval_frame_idx: [B, C, H, W]
        pred_frame_rand = pred_frames_rand[:, pred_idx, :, :, :]

        # Compute MSE per sample: [B]
        mse_rand_per_sample = ((pred_frame_rand - gt_frame) ** 2).view(B, -1).mean(dim=1)

        # =========================================================================
        # Step 5: Compute PSNR per sample and average
        # =========================================================================
        # PSNR = 10 * log10(1 / MSE)
        # Data is normalized to [-1, 1], so max signal value is 1.0
        psnr_seq_per_sample = 10 * torch.log10(1.0 / (mse_seq_per_sample + 1e-8))  # [B]
        psnr_rand_per_sample = 10 * torch.log10(1.0 / (mse_rand_per_sample + 1e-8))  # [B]

        # Average across batch: dPSNR = (1/B) * sum[PSNR_seq - PSNR_rand]
        psnr_seq = psnr_seq_per_sample.mean()
        psnr_rand = psnr_rand_per_sample.mean()

        # Compute dPSNR
        dsnr = psnr_seq - psnr_rand

    return dsnr.item(), psnr_seq.item(), psnr_rand.item()


def world_sensitivity_dsnr(world_model, h_sequence, eval_frame_idx=4):
    """
    Compute world sensitivity metric: dPSNR = PSNR_seq - PSNR_rand.

    Following the specification from description.md:
    - PSNR_seq: PSNR with correct world embedding (from encoder)
    - PSNR_rand: PSNR with random world embedding (sampled from categorical distribution)
    - dPSNR = (1/B) * sum_{b=1}^{B} [PSNR_seq_b - PSNR_rand_b]
    - PSNR = 10 * log10(1 / MSE)
    - Evaluated at a specific frame index (Genie uses t=4)

    Higher dPSNR means model is more sensitive to correct world embedding.

    Args:
        world_model: WorldModel instance
        h_sequence: [B, T, 16, 64, 64] - VVAE latent sequence (normalized to [-1, 1])
        eval_frame_idx: Frame index to evaluate (default: 4, following Genie)

    Returns:
        tuple: (dsnr, psnr_seq, psnr_rand) - all scalar values for logging
    """
    with torch.no_grad():
        B, T, C, H, W = h_sequence.shape
        device = h_sequence.device

        # Ensure eval_frame_idx is valid
        if eval_frame_idx >= T:
            eval_frame_idx = T - 1  # Use last frame if eval_frame_idx is too large

        # =========================================================================
        # Step 1: Tokenize once (shared for both seq and rand)
        # =========================================================================
        tokens, _, _ = world_model.tokenizer(h_sequence)  # [B*T, d_model, H', W']

        # =========================================================================
        # Step 2: Get action codes and world embedding
        # =========================================================================
        action_codes, _, _, _ = world_model.action_encoder(tokens, B, T - 1)  # [B, T-1, d_code_a]
        world_emb, _, _, _ = world_model.world_encoder(tokens, B, T)  # [B, d_code_h]

        # Per description.md: Drop last frame, pass GT[0:T-1] to predictor for perfect alignment
        tokens_input = tokens[:B * (T - 1)]  # [B*(T-1), d_model, H', W']

        # =========================================================================
        # Step 3: PSNR_seq - Predict with correct world embedding
        # =========================================================================
        pred_tokens_seq = world_model.dynamics_predictor(tokens_input, action_codes, world_emb, B, T - 1)
        pred_frames_seq = world_model.detokenizer(pred_tokens_seq).view(B, T - 1, C, H, W)

        # Extract frame at eval_frame_idx
        # pred_frames_seq predicts frames [1:T], so eval_frame_idx=4 corresponds to pred_frames_seq[:, 3]
        # Adjust index: eval_frame_idx points to GT frame, but predictions are for frames [1:T]
        pred_idx = eval_frame_idx - 1  # Frame at eval_frame_idx is predicted at position eval_frame_idx-1
        pred_frame_seq = pred_frames_seq[:, pred_idx, :, :, :]  # [B, C, H, W]
        gt_frame = h_sequence[:, eval_frame_idx, :, :, :]

        # Compute MSE per sample: [B]
        mse_seq_per_sample = ((pred_frame_seq - gt_frame) ** 2).view(B, -1).mean(dim=1)

        # =========================================================================
        # Step 4: PSNR_rand - Predict with random world embedding sampled from codebook
        # =========================================================================
        # Sample random world embedding from categorical distribution (codebook)
        # world_emb shape: [B, d_code_h]
        num_levels = world_model.world_encoder.rvq.num_levels

        # Sample random indices for each level
        world_emb_random = torch.zeros_like(world_emb)  # [B, d_code_h]

        for level_idx in range(num_levels):
            # Get codebook for this level: [codebook_size, d_code_h]
            codebook = world_model.world_encoder.rvq._get_codebook(level_idx)
            codebook_size = world_model.world_encoder.rvq.codebook_sizes[level_idx]

            # Sample random indices: [B]
            random_idx = torch.randint(0, codebook_size, (B,), device=device)

            # Look up embeddings: [B, d_code_h]
            level_codes = F.embedding(random_idx, codebook)

            # Add to total (RVQ sums across levels)
            world_emb_random += level_codes

        # Predict with random world embedding
        pred_tokens_rand = world_model.dynamics_predictor(tokens_input, action_codes, world_emb_random, B, T - 1)
        pred_frames_rand = world_model.detokenizer(pred_tokens_rand).view(B, T - 1, C, H, W)

        # Extract frame at eval_frame_idx: [B, C, H, W]
        pred_frame_rand = pred_frames_rand[:, pred_idx, :, :, :]

        # Compute MSE per sample: [B]
        mse_rand_per_sample = ((pred_frame_rand - gt_frame) ** 2).view(B, -1).mean(dim=1)

        # =========================================================================
        # Step 5: Compute PSNR per sample and average
        # =========================================================================
        # PSNR = 10 * log10(1 / MSE)
        # Data is normalized to [-1, 1], so max signal value is 1.0
        psnr_seq_per_sample = 10 * torch.log10(1.0 / (mse_seq_per_sample + 1e-8))  # [B]
        psnr_rand_per_sample = 10 * torch.log10(1.0 / (mse_rand_per_sample + 1e-8))  # [B]

        # Average across batch: dPSNR = (1/B) * sum[PSNR_seq - PSNR_rand]
        psnr_seq = psnr_seq_per_sample.mean()
        psnr_rand = psnr_rand_per_sample.mean()

        # Compute dPSNR
        dsnr = psnr_seq - psnr_rand

    return dsnr.item(), psnr_seq.item(), psnr_rand.item()
