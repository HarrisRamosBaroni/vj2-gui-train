"""
Numerical Analysis of World Model Latent Action Space

This script performs systematic analyses of the World Model's action space and dynamics:

1. **Action-Conditioned Delta Consistency**: Measures if the same action produces
   consistent effects across different states by analyzing variance of deltas.

2. **Action Discriminability**: Compares within-action variance to between-action
   distances to see if different actions produce distinguishable effects.

3. **Rollout Error Accumulation**: Analyzes how prediction error grows over time
   in autoregressive rollout vs teacher forcing.

4. **Action Sensitivity Analysis**: Measures how sensitive the model is to changes
   in action codes (via finite differences).

5. **World Hypothesis Consistency**: Checks if the world encoder produces consistent
   embeddings for similar state sequences.

6. **Codebook Utilization**: Analyzes which codebook entries are used and their
   distribution across the dataset.

7. **Temporal Coherence**: Measures how smoothly predicted trajectories evolve
   compared to ground truth.
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import random

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from LAM.training import load_model
from LAM.world_model import WorldModel
from LAM.dataloader_vvae import create_vvae_dataloaders

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def get_sample_sequences(dataloader, num_samples: int, device: str):
    """Sample random sequences from the dataloader."""
    sequences = []
    dataset = dataloader.dataset
    rng = random.Random(42)

    # Sample random indices
    total_samples = len(dataset)
    indices = rng.sample(range(total_samples), min(num_samples, total_samples))

    for idx in indices:
        sample = dataset[idx]
        sequence = sample['sequence']  # [T, C, H, W]
        sequences.append(sequence.to(device))

    return sequences


# =============================================================================
# Analysis 1: Action-Conditioned Delta Consistency
# =============================================================================

@torch.no_grad()
def analyze_action_delta_consistency(
    model: WorldModel,
    sequences: list,
    output_dir: Path,
    wandb_run=None,
):
    """
    Measure consistency of action effects across different states.

    For each action code extracted from the data:
    - Apply it to multiple different states
    - Measure variance of resulting deltas (s_next - s_initial)
    - Low variance = consistent action effect
    """
    logger.info("=" * 80)
    logger.info("ANALYSIS 1: Action-Conditioned Delta Consistency")
    logger.info("=" * 80)

    device = next(model.parameters()).device

    # Extract states (first frame from each sequence)
    states = torch.stack([seq[0] for seq in sequences])  # [N, C, H, W]
    N = states.shape[0]

    # Extract action codes from sequences
    logger.info(f"Extracting action codes from {N} sequences...")
    all_action_codes = []

    for seq in sequences[:50]:  # Use first 50 sequences for actions
        # seq: [T, C, H, W], reshape to [1, T, C, H, W]
        seq_batch = seq.unsqueeze(0)

        # Tokenize
        tokens, _, _ = model.tokenizer(seq_batch)
        B, T = 1, seq_batch.shape[1]

        # Extract action codes
        action_codes, _, _, _ = model.action_encoder(tokens, B, T - 1)  # [1, T-1, d_code_a]
        all_action_codes.append(action_codes[0])  # [T-1, d_code_a]

    # Flatten to get all action codes
    all_action_codes = torch.cat(all_action_codes, dim=0)  # [K, d_code_a]
    K = min(all_action_codes.shape[0], 50)  # Use up to 50 actions
    action_samples = all_action_codes[:K]

    logger.info(f"Testing {K} action codes on {N} different states...")

    # For each action, apply to all states and measure delta consistency
    within_action_delta_norms = []
    all_delta_tensors = []

    for action_idx in range(K):
        action = action_samples[action_idx]  # [d_code_a]

        # Apply this action to all states
        deltas_for_action = []

        for state_idx in range(N):
            state = states[state_idx]  # [C, H, W]

            # Create a 2-frame sequence: [state, state] to apply action
            seq = state.unsqueeze(0).repeat(2, 1, 1, 1).unsqueeze(0)  # [1, 2, C, H, W]

            # Tokenize
            tokens, _, _ = model.tokenizer(seq)

            # Get world embedding from this state
            world_emb, _, _, _ = model.world_encoder(tokens, 1, 2)  # [1, d_code_h]

            # Predict next frame using this action
            action_batch = action.unsqueeze(0).unsqueeze(0)  # [1, 1, d_code_a]
            pred_tokens = model.dynamics_predictor(tokens, action_batch, world_emb, 1, 2)

            # Decode predicted frame 1
            pred_frame = model.detokenizer(pred_tokens[1:2])  # [1, C, H, W]

            # Compute delta
            delta = pred_frame[0] - state  # [C, H, W]
            deltas_for_action.append(delta)

        # Stack deltas for this action
        deltas_for_action = torch.stack(deltas_for_action)  # [N, C, H, W]
        all_delta_tensors.append(deltas_for_action)

        # Compute pairwise distances between deltas (consistency measure)
        deltas_flat = deltas_for_action.view(N, -1)
        pairwise_dists = torch.pdist(deltas_flat)
        within_action_delta_norms.extend(pairwise_dists.cpu().numpy())

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(within_action_delta_norms, bins=50, alpha=0.7, density=True)
    plt.title("Distribution of Within-Action Delta Distances")
    plt.xlabel("L2 Distance Between Deltas")
    plt.ylabel("Density")
    plt.grid(True)
    save_path = output_dir / "1_action_delta_consistency.png"
    plt.savefig(save_path)
    plt.close()

    if wandb_run:
        wandb_run.log({"1_action_delta_consistency": wandb.Image(str(save_path))})

    logger.info(f"Mean within-action delta distance: {np.mean(within_action_delta_norms):.6f}")
    logger.info(f"Std within-action delta distance: {np.std(within_action_delta_norms):.6f}")
    logger.info(f"Saved plot to {save_path}\n")

    return all_delta_tensors, action_samples


# =============================================================================
# Analysis 2: Action Discriminability
# =============================================================================

@torch.no_grad()
def analyze_action_discriminability(
    all_delta_tensors: list,
    output_dir: Path,
    wandb_run=None,
):
    """
    Compare within-action variance to between-action distances.
    Good discriminability = actions produce distinguishable effects.
    """
    logger.info("=" * 80)
    logger.info("ANALYSIS 2: Action Discriminability")
    logger.info("=" * 80)

    K = len(all_delta_tensors)
    N = all_delta_tensors[0].shape[0]

    # Compute mean delta for each action
    mean_deltas = torch.stack([deltas.mean(dim=0) for deltas in all_delta_tensors])  # [K, C, H, W]

    # Between-action distances (using mean deltas)
    mean_deltas_flat = mean_deltas.view(K, -1)
    between_action_dists = torch.pdist(mean_deltas_flat).cpu().numpy()

    # Within-action distances (all pairs within same action)
    within_action_dists = []
    for deltas in all_delta_tensors:
        deltas_flat = deltas.view(N, -1)
        dists = torch.pdist(deltas_flat).cpu().numpy()
        within_action_dists.extend(dists)

    # Plot density comparison
    plt.figure(figsize=(12, 6))

    if len(within_action_dists) > 0 and len(between_action_dists) > 0:
        kde_within = gaussian_kde(within_action_dists)
        kde_between = gaussian_kde(between_action_dists)

        max_dist = max(max(within_action_dists), max(between_action_dists))
        x_range = np.linspace(0, max_dist * 1.1, 500)

        plt.plot(x_range, kde_within(x_range), label="Within-Action (Same action, different states)", linewidth=2)
        plt.plot(x_range, kde_between(x_range), label="Between-Action (Different actions)", linewidth=2)
        plt.xlabel("L2 Distance")
        plt.ylabel("Density")
        plt.title("Action Discriminability: Within vs Between Action Distances")
        plt.legend()
        plt.grid(True)

    save_path = output_dir / "2_action_discriminability.png"
    plt.savefig(save_path)
    plt.close()

    if wandb_run:
        wandb_run.log({"2_action_discriminability": wandb.Image(str(save_path))})

    logger.info(f"Mean between-action distance: {np.mean(between_action_dists):.6f}")
    logger.info(f"Mean within-action distance: {np.mean(within_action_dists):.6f}")
    ratio = np.mean(between_action_dists) / (np.mean(within_action_dists) + 1e-8)
    logger.info(f"Discriminability ratio (between/within): {ratio:.4f}")
    logger.info(f"Saved plot to {save_path}\n")


# =============================================================================
# Analysis 3: Rollout Error Accumulation
# =============================================================================

@torch.no_grad()
def analyze_rollout_error_accumulation(
    model: WorldModel,
    sequences: list,
    max_steps: int,
    output_dir: Path,
    wandb_run=None,
):
    """
    Analyze how prediction error accumulates in autoregressive rollout.
    Compare to teacher forcing error at each step.
    """
    logger.info("=" * 80)
    logger.info("ANALYSIS 3: Rollout Error Accumulation")
    logger.info("=" * 80)

    device = next(model.parameters()).device

    # Select sequences with sufficient length
    valid_sequences = [seq for seq in sequences if seq.shape[0] >= max_steps]
    if len(valid_sequences) == 0:
        logger.warning("No sequences long enough for rollout analysis")
        return

    num_test = min(20, len(valid_sequences))
    test_sequences = valid_sequences[:num_test]

    logger.info(f"Testing rollout on {num_test} sequences for {max_steps} steps...")

    tf_errors_per_step = [[] for _ in range(max_steps - 1)]
    rollout_errors_per_step = [[] for _ in range(max_steps - 1)]

    for seq in test_sequences:
        seq_trimmed = seq[:max_steps].unsqueeze(0)  # [1, T, C, H, W]
        B, T = 1, max_steps

        # Teacher forcing: get predictions for all frames
        output_tf = model(seq_trimmed)
        pred_frames_tf = output_tf['pred_frames'][0]  # [T, C, H, W]

        # Compute TF error at each step
        for t in range(1, T):
            error = F.mse_loss(pred_frames_tf[t], seq_trimmed[0, t]).item()
            tf_errors_per_step[t - 1].append(error)

        # Autoregressive rollout with real actions and world embedding
        tokens_gt, _, _ = model.tokenizer(seq_trimmed)
        action_codes, _, _, _ = model.action_encoder(tokens_gt, B, T - 1)
        world_emb, _, _, _ = model.world_encoder(tokens_gt, B, T)

        rollout_tokens = tokens_gt.clone()

        for t in range(1, T):
            # Predict frame t
            pred_tokens = model.dynamics_predictor(rollout_tokens, action_codes, world_emb, B, T)

            # Extract predicted tokens for frame t
            pred_frame_tokens = pred_tokens[t:t+1]
            pred_frame = model.detokenizer(pred_frame_tokens)[0]  # [C, H, W]

            # Compute error
            error = F.mse_loss(pred_frame, seq_trimmed[0, t]).item()
            rollout_errors_per_step[t - 1].append(error)

            # Replace tokens
            rollout_tokens[t:t+1] = pred_frame_tokens.detach()

    # Compute means and stds
    tf_means = [np.mean(errors) for errors in tf_errors_per_step]
    tf_stds = [np.std(errors) for errors in tf_errors_per_step]
    rollout_means = [np.mean(errors) for errors in rollout_errors_per_step]
    rollout_stds = [np.std(errors) for errors in rollout_errors_per_step]

    # Plot error accumulation
    steps = np.arange(1, max_steps)

    plt.figure(figsize=(12, 6))
    plt.plot(steps, tf_means, 'o-', label='Teacher Forcing', linewidth=2)
    plt.fill_between(steps,
                     np.array(tf_means) - np.array(tf_stds),
                     np.array(tf_means) + np.array(tf_stds),
                     alpha=0.3)

    plt.plot(steps, rollout_means, 's-', label='Autoregressive Rollout', linewidth=2)
    plt.fill_between(steps,
                     np.array(rollout_means) - np.array(rollout_stds),
                     np.array(rollout_means) + np.array(rollout_stds),
                     alpha=0.3)

    plt.xlabel("Prediction Step")
    plt.ylabel("MSE")
    plt.title("Error Accumulation: Teacher Forcing vs Autoregressive Rollout")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    save_path = output_dir / "3_rollout_error_accumulation.png"
    plt.savefig(save_path)
    plt.close()

    if wandb_run:
        wandb_run.log({"3_rollout_error_accumulation": wandb.Image(str(save_path))})

    logger.info(f"Teacher forcing final step error: {tf_means[-1]:.6f}")
    logger.info(f"Rollout final step error: {rollout_means[-1]:.6f}")
    logger.info(f"Error growth factor (rollout/TF): {rollout_means[-1] / tf_means[-1]:.2f}x")
    logger.info(f"Saved plot to {save_path}\n")


# =============================================================================
# Analysis 4: Action Sensitivity (Finite Differences)
# =============================================================================

@torch.no_grad()
def analyze_action_sensitivity(
    model: WorldModel,
    sequences: list,
    output_dir: Path,
    epsilon: float = 0.1,
    wandb_run=None,
):
    """
    Measure sensitivity of predictions to action code perturbations.
    Use finite differences to approximate gradient.
    """
    logger.info("=" * 80)
    logger.info("ANALYSIS 4: Action Sensitivity")
    logger.info("=" * 80)

    device = next(model.parameters()).device

    # Use first sequence
    seq = sequences[0][:3].unsqueeze(0)  # [1, 3, C, H, W]
    B, T = 1, 3

    # Extract action codes and world embedding
    tokens_gt, _, _ = model.tokenizer(seq)
    action_codes, _, _, _ = model.action_encoder(tokens_gt, B, T - 1)  # [1, T-1, d_code_a]
    world_emb, _, _, _ = model.world_encoder(tokens_gt, B, T)

    # Get baseline prediction
    pred_tokens_base = model.dynamics_predictor(tokens_gt, action_codes, world_emb, B, T)
    pred_frame_base = model.detokenizer(pred_tokens_base[1:2])[0]  # [C, H, W]

    # Perturb each dimension of action code
    d_code_a = action_codes.shape[-1]
    sensitivities = []

    logger.info(f"Computing sensitivities for {d_code_a} action dimensions...")

    for dim in range(d_code_a):
        # Perturb this dimension
        action_codes_perturbed = action_codes.clone()
        action_codes_perturbed[0, 0, dim] += epsilon

        # Predict with perturbed action
        pred_tokens_perturbed = model.dynamics_predictor(tokens_gt, action_codes_perturbed, world_emb, B, T)
        pred_frame_perturbed = model.detokenizer(pred_tokens_perturbed[1:2])[0]  # [C, H, W]

        # Compute difference
        delta = pred_frame_perturbed - pred_frame_base
        sensitivity = torch.norm(delta).item() / epsilon
        sensitivities.append(sensitivity)

    # Plot sensitivities
    plt.figure(figsize=(12, 6))
    plt.bar(range(d_code_a), sensitivities)
    plt.xlabel("Action Code Dimension")
    plt.ylabel("Sensitivity (||Δframe|| / ε)")
    plt.title(f"Action Code Sensitivity (ε={epsilon})")
    plt.grid(True, axis='y')

    save_path = output_dir / "4_action_sensitivity.png"
    plt.savefig(save_path)
    plt.close()

    if wandb_run:
        wandb_run.log({"4_action_sensitivity": wandb.Image(str(save_path))})

    logger.info(f"Mean sensitivity: {np.mean(sensitivities):.6f}")
    logger.info(f"Max sensitivity: {np.max(sensitivities):.6f}")
    logger.info(f"Min sensitivity: {np.min(sensitivities):.6f}")
    logger.info(f"Saved plot to {save_path}\n")


# =============================================================================
# Analysis 5: World Hypothesis Consistency
# =============================================================================

@torch.no_grad()
def analyze_world_hypothesis_consistency(
    model: WorldModel,
    sequences: list,
    output_dir: Path,
    wandb_run=None,
):
    """
    Check if similar sequences produce similar world embeddings.
    """
    logger.info("=" * 80)
    logger.info("ANALYSIS 5: World Hypothesis Consistency")
    logger.info("=" * 80)

    device = next(model.parameters()).device

    num_test = min(50, len(sequences))
    test_sequences = sequences[:num_test]

    logger.info(f"Extracting world embeddings from {num_test} sequences...")

    world_embeddings = []

    for seq in test_sequences:
        seq_batch = seq[:8].unsqueeze(0)  # [1, T, C, H, W], use first 8 frames
        B, T = 1, seq_batch.shape[1]

        tokens, _, _ = model.tokenizer(seq_batch)
        world_emb, _, _, _ = model.world_encoder(tokens, B, T)  # [1, d_code_h]
        world_embeddings.append(world_emb[0])  # [d_code_h]

    world_embeddings = torch.stack(world_embeddings)  # [N, d_code_h]

    # Compute pairwise distances
    world_emb_flat = world_embeddings.view(num_test, -1)
    pairwise_dists = torch.pdist(world_emb_flat).cpu().numpy()

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(pairwise_dists, bins=50, alpha=0.7, density=True)
    plt.xlabel("L2 Distance Between World Embeddings")
    plt.ylabel("Density")
    plt.title("Distribution of Pairwise World Embedding Distances")
    plt.grid(True)

    save_path = output_dir / "5_world_hypothesis_consistency.png"
    plt.savefig(save_path)
    plt.close()

    if wandb_run:
        wandb_run.log({"5_world_hypothesis_consistency": wandb.Image(str(save_path))})

    logger.info(f"Mean world embedding distance: {np.mean(pairwise_dists):.6f}")
    logger.info(f"Std world embedding distance: {np.std(pairwise_dists):.6f}")
    logger.info(f"Saved plot to {save_path}\n")


# =============================================================================
# Analysis 6: Codebook Utilization
# =============================================================================

@torch.no_grad()
def analyze_codebook_utilization(
    model: WorldModel,
    sequences: list,
    output_dir: Path,
    wandb_run=None,
):
    """
    Analyze which codebook entries are used and their distribution.
    """
    logger.info("=" * 80)
    logger.info("ANALYSIS 6: Codebook Utilization")
    logger.info("=" * 80)

    device = next(model.parameters()).device

    num_test = min(100, len(sequences))
    test_sequences = sequences[:num_test]

    logger.info(f"Analyzing codebook usage on {num_test} sequences...")

    # Get codebook sizes
    codebook_sizes_a = model.action_encoder.rvq.codebook_sizes
    codebook_sizes_h = model.world_encoder.rvq.codebook_sizes
    num_levels_a = len(codebook_sizes_a)
    num_levels_h = len(codebook_sizes_h)

    # Track which codes are used
    action_code_usage = [set() for _ in range(num_levels_a)]
    world_code_usage = [set() for _ in range(num_levels_h)]

    for seq in test_sequences:
        seq_batch = seq[:8].unsqueeze(0)  # [1, T, C, H, W]

        output = model(seq_batch)

        # Action indices: [1, T-1, num_levels]
        action_indices = output['action_indices'][0]  # [T-1, num_levels]
        for level in range(num_levels_a):
            action_code_usage[level].update(action_indices[:, level].cpu().numpy().tolist())

        # World indices: [1, num_levels]
        world_indices = output['world_indices'][0]  # [num_levels]
        for level in range(num_levels_h):
            world_code_usage[level].add(world_indices[level].item())

    # Compute utilization percentages
    action_utilization = [len(used) / size * 100 for used, size in zip(action_code_usage, codebook_sizes_a)]
    world_utilization = [len(used) / size * 100 for used, size in zip(world_code_usage, codebook_sizes_h)]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(range(num_levels_a), action_utilization)
    ax1.set_xlabel("RVQ Level")
    ax1.set_ylabel("Utilization (%)")
    ax1.set_title("Action Encoder Codebook Utilization")
    ax1.set_ylim([0, 100])
    ax1.grid(True, axis='y')

    ax2.bar(range(num_levels_h), world_utilization)
    ax2.set_xlabel("RVQ Level")
    ax2.set_ylabel("Utilization (%)")
    ax2.set_title("World Encoder Codebook Utilization")
    ax2.set_ylim([0, 100])
    ax2.grid(True, axis='y')

    plt.tight_layout()
    save_path = output_dir / "6_codebook_utilization.png"
    plt.savefig(save_path)
    plt.close()

    if wandb_run:
        wandb_run.log({"6_codebook_utilization": wandb.Image(str(save_path))})

    logger.info("Action Encoder Codebook Utilization:")
    for level, util in enumerate(action_utilization):
        logger.info(f"  Level {level + 1}: {util:.1f}% ({len(action_code_usage[level])}/{codebook_sizes_a[level]})")

    logger.info("World Encoder Codebook Utilization:")
    for level, util in enumerate(world_utilization):
        logger.info(f"  Level {level + 1}: {util:.1f}% ({len(world_code_usage[level])}/{codebook_sizes_h[level]})")

    logger.info(f"Saved plot to {save_path}\n")


# =============================================================================
# Analysis 7: Temporal Coherence
# =============================================================================

@torch.no_grad()
def analyze_temporal_coherence(
    model: WorldModel,
    sequences: list,
    output_dir: Path,
    wandb_run=None,
):
    """
    Measure how smoothly predicted trajectories evolve compared to ground truth.
    """
    logger.info("=" * 80)
    logger.info("ANALYSIS 7: Temporal Coherence")
    logger.info("=" * 80)

    device = next(model.parameters()).device

    num_test = min(20, len(sequences))
    test_sequences = [seq for seq in sequences if seq.shape[0] >= 8][:num_test]

    logger.info(f"Analyzing temporal coherence on {num_test} sequences...")

    gt_frame_diffs = []
    tf_frame_diffs = []
    rollout_frame_diffs = []

    for seq in test_sequences:
        seq_batch = seq[:8].unsqueeze(0)  # [1, T, C, H, W]
        B, T = 1, 8

        # Teacher forcing predictions
        output_tf = model(seq_batch)
        pred_frames_tf = output_tf['pred_frames'][0]  # [T, C, H, W]

        # Rollout predictions
        tokens_gt, _, _ = model.tokenizer(seq_batch)
        action_codes, _, _, _ = model.action_encoder(tokens_gt, B, T - 1)
        world_emb, _, _, _ = model.world_encoder(tokens_gt, B, T)

        rollout_tokens = tokens_gt.clone()
        rollout_frames = [seq_batch[0, 0]]

        for t in range(1, T):
            pred_tokens = model.dynamics_predictor(rollout_tokens, action_codes, world_emb, B, T)
            pred_frame = model.detokenizer(pred_tokens[t:t+1])[0]
            rollout_frames.append(pred_frame)
            rollout_tokens[t:t+1] = pred_tokens[t:t+1].detach()

        rollout_frames = torch.stack(rollout_frames)  # [T, C, H, W]

        # Compute frame-to-frame differences
        for t in range(T - 1):
            gt_diff = F.mse_loss(seq_batch[0, t + 1], seq_batch[0, t]).item()
            tf_diff = F.mse_loss(pred_frames_tf[t + 1], pred_frames_tf[t]).item()
            rollout_diff = F.mse_loss(rollout_frames[t + 1], rollout_frames[t]).item()

            gt_frame_diffs.append(gt_diff)
            tf_frame_diffs.append(tf_diff)
            rollout_frame_diffs.append(rollout_diff)

    # Plot distributions
    plt.figure(figsize=(12, 6))
    plt.hist(gt_frame_diffs, bins=30, alpha=0.5, label='Ground Truth', density=True)
    plt.hist(tf_frame_diffs, bins=30, alpha=0.5, label='Teacher Forcing', density=True)
    plt.hist(rollout_frame_diffs, bins=30, alpha=0.5, label='Rollout', density=True)
    plt.xlabel("Frame-to-Frame MSE")
    plt.ylabel("Density")
    plt.title("Temporal Coherence: Frame-to-Frame Differences")
    plt.legend()
    plt.grid(True)

    save_path = output_dir / "7_temporal_coherence.png"
    plt.savefig(save_path)
    plt.close()

    if wandb_run:
        wandb_run.log({"7_temporal_coherence": wandb.Image(str(save_path))})

    logger.info(f"Mean GT frame-to-frame diff: {np.mean(gt_frame_diffs):.6f}")
    logger.info(f"Mean TF frame-to-frame diff: {np.mean(tf_frame_diffs):.6f}")
    logger.info(f"Mean rollout frame-to-frame diff: {np.mean(rollout_frame_diffs):.6f}")
    logger.info(f"Saved plot to {save_path}\n")


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze World Model Latent Action Space')

    # Model and data
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to trained World Model checkpoint (.pt file)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing VVAE HDF5 files')
    parser.add_argument('--manifest_path', type=str, required=True,
                        help='Path to manifest JSON file')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Which data split to use (default: val)')

    # Analysis parameters
    parser.add_argument('--num_sequences', type=int, default=100,
                        help='Number of sequences to sample for analysis (default: 100)')
    parser.add_argument('--sequence_length', type=int, default=8,
                        help='Sequence length to use (default: 8)')
    parser.add_argument('--rollout_steps', type=int, default=8,
                        help='Number of rollout steps for error accumulation analysis (default: 8)')

    # Output
    parser.add_argument('--output_dir', type=str, default='./output/world_model_analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')

    # Wandb
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='world_model_analysis',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (auto-generated if None)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity/team name')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("WORLD MODEL NUMERICAL ANALYSIS")
    logger.info("=" * 80)

    # Initialize wandb
    wandb_run = None
    if not args.no_wandb and WANDB_AVAILABLE:
        if args.wandb_run_name is None:
            checkpoint_name = Path(args.checkpoint_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.wandb_run_name = f"analysis_{checkpoint_name}_{timestamp}"

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args)
        )
        logger.info(f"Wandb initialized: {args.wandb_project}/{args.wandb_run_name}")

    # Load model
    logger.info("\nLoading model...")
    model, _, checkpoint_info = load_model(args.checkpoint_path, device=device)
    model.eval()
    logger.info(f"Loaded checkpoint from epoch {checkpoint_info['epoch']}, step {checkpoint_info['global_step']}")

    # Load data
    logger.info("\nLoading data...")
    train_loader, val_loader, test_loader = create_vvae_dataloaders(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        batch_size=1,  # Load one at a time
        sequence_length=args.sequence_length,
        stride_train=1,
        num_workers=4,
        ddp=False
    )

    if args.split == 'train':
        dataloader = train_loader
    elif args.split == 'val':
        dataloader = val_loader
    elif args.split == 'test':
        dataloader = test_loader
    else:
        raise ValueError(f"Invalid split: {args.split}")

    logger.info(f"Using {args.split} split with {len(dataloader)} batches")

    # Sample sequences
    logger.info(f"\nSampling {args.num_sequences} sequences...")
    sequences = get_sample_sequences(dataloader, args.num_sequences, device)
    logger.info(f"Sampled {len(sequences)} sequences\n")

    # Run analyses
    all_delta_tensors, action_samples = analyze_action_delta_consistency(model, sequences, output_dir, wandb_run)

    analyze_action_discriminability(all_delta_tensors, output_dir, wandb_run)

    analyze_rollout_error_accumulation(model, sequences, args.rollout_steps, output_dir, wandb_run)

    analyze_action_sensitivity(model, sequences, epsilon=0.1, output_dir=output_dir, wandb_run=wandb_run)

    analyze_world_hypothesis_consistency(model, sequences, output_dir, wandb_run)

    analyze_codebook_utilization(model, sequences, output_dir, wandb_run)

    analyze_temporal_coherence(model, sequences, output_dir, wandb_run)

    # Finish
    if wandb_run:
        wandb_run.finish()

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
