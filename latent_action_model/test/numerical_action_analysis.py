"""
Performs a systematic numerical analysis of the latent action space of a VAE model.

This script implements and visualizes the following analyses:
1.  **Action-Conditioned Delta Consistency**: Measures if a fixed action has a
    consistent effect across different states by analyzing the variance of the
    resulting state deltas (s_next - s_initial).

2.  **Action Discriminability**: Compares the variation of deltas within a single
    action to the variation between deltas from different actions.

3.  **Jacobian Sensitivity Analysis**: Approximates the Jacobian of the output
    state with respect to the action vector (ds_next/da) to understand which
    action dimensions are most influential.

4.  **Continuity and Composition**: Applies the same action repeatedly to a state
    to see if the effect accumulates consistently or collapses.

5.  **Nearest-Neighbor Retrieval**: Checks for memorization by generating states
    and finding their nearest neighbors in the training dataset.
"""
import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from logging import getLogger
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Project modules
from latent_action_model.vae import LatentActionVAE, load_lam_config
from training.dataloader import init_preprocessed_data_loader
from gui_world_model.utils.wandb_utils import init_wandb

logger = getLogger(__name__)

# --- Utility Functions ---

@torch.no_grad()
def layer_norm_patches(z: torch.Tensor) -> torch.Tensor:
    """Apply LayerNorm along the patch-feature dimension (D)."""
    return F.layer_norm(z, (z.size(-1),))

def load_vae_model(ckpt_path: Path, config_path: Path, device: torch.device) -> LatentActionVAE:
    """Load LatentActionVAE from checkpoint."""
    config_dict = load_lam_config(config_path)
    model = LatentActionVAE(**config_dict).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded VAE checkpoint from {ckpt_path}")
    return model

def get_sample_states(loader, num_states, device):
    """Sample states from the dataloader."""
    states = []
    dataset = loader.dataset
    rng = random.Random(42)
    indices = rng.sample(range(len(dataset)), num_states)
    
    for traj_idx in indices:
        traj_emb, _ = dataset[traj_idx]
        if traj_emb.shape[0] > 0:
            frame_idx = rng.randint(0, traj_emb.shape[0] - 1)
            state = layer_norm_patches(traj_emb[frame_idx]).to(device)
            states.append(state)
            
    return torch.stack(states)

# --- Analysis Functions ---

@torch.no_grad()
def run_delta_consistency_analysis(vae, states, actions, output_dir, run):
    """
    Analysis 1: Action-conditioned delta consistency.
    Computes deltas and variance statistics.
    """
    print("Running Analysis 1: Delta Consistency...")
    device = next(vae.parameters()).device
    num_states = states.shape[0]
    num_actions = actions.shape[0]
    
    all_deltas = torch.zeros(num_actions, num_states, *states.shape[1:])
    all_s_next = torch.zeros(num_actions, num_states, *states.shape[1:])

    for j in range(num_actions):
        action = actions[j].unsqueeze(0).expand(num_states, -1)
        s_batch = states.unsqueeze(1) # Add sequence dimension
        s_next = vae.decode(s_batch, action)
        deltas = s_next - states
        all_deltas[j] = deltas.cpu()
        all_s_next[j] = s_next.cpu()

    # Calculate pairwise distances for deltas within each action
    within_action_delta_distances = []
    for j in range(num_actions):
        deltas_flat = all_deltas[j].view(num_states, -1)
        dists = torch.pdist(deltas_flat)
        within_action_delta_distances.extend(dists.numpy())

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(within_action_delta_distances, bins=50, alpha=0.7, density=True)
    plt.title("Histogram of Within-Action Delta Distances")
    plt.xlabel("L2 Distance between Deltas")
    plt.ylabel("Density")
    plt.grid(True)
    save_path = output_dir / "1_delta_consistency_hist.png"
    plt.savefig(save_path)
    plt.close()
    if run:
        run.log({"1_delta_consistency": wandb.Image(str(save_path))})
    print(f"Saved delta consistency plot to {save_path}")
    
    return all_deltas, all_s_next


@torch.no_grad()
def run_discriminability_analysis(all_deltas, all_s_next, output_dir, run):
    """
    Analyses 2 & 3: Predictability and Discriminability.
    Compares within-action variance to between-action distances.
    """
    print("Running Analysis 2 & 3: Discriminability...")
    num_actions, num_states, _, _ = all_deltas.shape

    # Within-action variance
    s_next_variance = all_s_next.var(dim=1).mean(dim=(-1, -2)) # Var across states
    delta_variance = all_deltas.var(dim=1).mean(dim=(-1, -2)) # Var across states

    # Between-action distance
    mean_deltas = all_deltas.mean(dim=1)
    mean_deltas_flat = mean_deltas.view(num_actions, -1)
    between_action_distances = torch.pdist(mean_deltas_flat)

    # Plot: Within vs Between Variance/Distance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Density plot for distances
    within_action_distances_flat = torch.pdist(all_deltas.view(num_actions * num_states, -1))
    kde_within = gaussian_kde(within_action_distances_flat.numpy())
    kde_between = gaussian_kde(between_action_distances.numpy())
    
    # Determine the x-range based on the max of both distributions
    max_dist = max(within_action_distances_flat.max(), between_action_distances.max())
    x_range = np.linspace(0, max_dist * 1.05, 500)

    ax1.plot(x_range, kde_within(x_range), label="Within-Action Delta Distances")
    ax1.plot(x_range, kde_between(x_range), label="Between-Action Mean Delta Distances")
    ax1.set_title("Action Discriminability")
    ax1.set_xlabel("L2 Distance")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True)

    # Scatter plot of variances
    ax2.scatter(delta_variance.numpy(), s_next_variance.numpy(), alpha=0.6)
    ax2.set_title("Action-only vs (Action + State) Predictability")
    ax2.set_xlabel("Variance of Deltas (Effect Consistency)")
    ax2.set_ylabel("Variance of Next States (Memorization)")
    ax2.grid(True)
    
    save_path = output_dir / "2_discriminability_scatter.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    if run:
        run.log({"2_discriminability": wandb.Image(str(save_path))})
    print(f"Saved discriminability plot to {save_path}")


@torch.no_grad()
def run_jacobian_analysis(vae, state, action, output_dir, run, epsilon=1e-3):
    """
    Analysis 4: Jacobian / sensitivity analysis.
    """
    print("Running Analysis 4: Jacobian Sensitivity...")
    device = next(vae.parameters()).device
    action_dim = action.shape[0]
    state_dim = state.numel()
    
    state_unsqueezed = state.unsqueeze(0).unsqueeze(0) # Add batch and seq dims
    
    s_next_base = vae.decode(state_unsqueezed, action.unsqueeze(0)).flatten()
    
    jacobian = torch.zeros(state_dim, action_dim, device='cpu')

    for k in range(action_dim):
        action_plus = action.clone()
        action_plus[k] += epsilon
        s_next_plus = vae.decode(state_unsqueezed, action_plus.unsqueeze(0)).flatten()
        jacobian_col = (s_next_plus - s_next_base).cpu() / epsilon
        jacobian[:, k] = jacobian_col

    # Compute SVD
    _, S, _ = torch.svd(jacobian)
    
    # Plot scree plot
    plt.figure(figsize=(10, 6))
    plt.plot(S.numpy(), 'o-')
    plt.title("Scree Plot of Jacobian Singular Values")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Singular Value")
    plt.yscale('log')
    plt.grid(True)
    save_path = output_dir / "4_jacobian_scree.png"
    plt.savefig(save_path)
    plt.close()
    if run:
        run.log({"4_jacobian_scree": wandb.Image(str(save_path))})
    print(f"Saved Jacobian scree plot to {save_path}")


@torch.no_grad()
def run_composition_analysis(vae, state, action, num_steps, output_dir, run):
    """
    Analysis 5: Continuity and composition.
    """
    print("Running Analysis 5: Composition...")
    device = next(vae.parameters()).device
    s_current = state
    delta_norms = []

    for _ in range(num_steps):
        s_next = vae.decode(s_current.unsqueeze(0).unsqueeze(0), action.unsqueeze(0))
        delta = s_next - s_current
        delta_norms.append(torch.norm(delta.flatten()).cpu().item())
        s_current = s_next.squeeze(0)

    # Plot delta norms
    plt.figure(figsize=(10, 6))
    plt.plot(delta_norms, 'o-')
    plt.title("Delta Norm Under Repeated Action Application")
    plt.xlabel("Step")
    plt.ylabel("L2 Norm of Delta")
    plt.grid(True)
    save_path = output_dir / "5_composition_trajectory.png"
    plt.savefig(save_path)
    plt.close()
    if run:
        run.log({"5_composition_trajectory": wandb.Image(str(save_path))})
    print(f"Saved composition plot to {save_path}")


@torch.no_grad()
def run_retrieval_analysis(generated_states, dataset_states, output_dir, run):
    """
    Analysis 6: Nearest-neighbour retrieval in dataset.
    """
    print("Running Analysis 6: Nearest-Neighbor Retrieval...")
    
    num_generated = generated_states.shape[0]
    
    # Find nearest neighbors
    s_next_flat = generated_states.view(num_generated, -1).to('cpu')
    dataset_states_flat = dataset_states.view(dataset_states.shape[0], -1).to('cpu')
    
    dists = torch.cdist(s_next_flat, dataset_states_flat)
    nn_indices = torch.argmin(dists, dim=1)

    # Plot histogram of NN indices
    plt.figure(figsize=(10, 6))
    plt.hist(nn_indices.numpy(), bins=min(100, dataset_states.shape[0]), alpha=0.7)
    plt.title("Histogram of Nearest-Neighbor Indices in Dataset")
    plt.xlabel("Dataset State Index")
    plt.ylabel("Frequency")
    plt.grid(True)
    save_path = output_dir / "6_nn_retrieval_hist.png"
    plt.savefig(save_path)
    plt.close()
    if run:
        run.log({"6_nn_retrieval": wandb.Image(str(save_path))})
    print(f"Saved NN retrieval plot to {save_path}")


# --- Main Execution ---

def parse_args():
    p = argparse.ArgumentParser(description="Perform numerical analysis of the latent action space.")
    
    # Models & Data
    p.add_argument("--vae_ckpt", type=Path, required=True, help="Path to VAE checkpoint.")
    p.add_argument("--vae_config", type=Path, required=True, help="Path to VAE config.")
    p.add_argument("--data_dir", type=Path, required=True, help="Directory with .h5 trajectory files.")
    p.add_argument("--manifest", type=Path, default=None, help="Path to data manifest JSON.")
    p.add_argument("--split", type=str, default="val", help="Split name in manifest.")

    # Analysis Params
    p.add_argument("--num_actions", type=int, default=50, help="Number of random actions to sample (N).")
    p.add_argument("--num_states", type=int, default=100, help="Number of start states to sample (M).")
    p.add_argument("--num_composition_steps", type=int, default=20, help="Number of steps for composition analysis.")
    p.add_argument("--num_retrieval_db_size", type=int, default=1000, help="Number of dataset states for NN retrieval.")

    # Output & Logging
    p.add_argument("--output_dir", type=Path, default=Path("output/lam/numerical_action_analysis"), help="Directory to save results.")
    p.add_argument("--wandb_project", type=str, default="action-space-numerical-analysis", help="Wandb project name.")
    p.add_argument("--wandb_run_name", type=str, required=True, help="Wandb run name.")
    
    p.add_argument("--device", type=str, default="cuda", help="Device to use (cuda|cpu).")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Init W&B and Output Dir
    run = None
    if args.wandb_project:
        run = init_wandb(
            project_name=args.wandb_project,
            run_name=args.wandb_run_name,
            config=vars(args)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
        
    # 2. Load model
    vae = load_vae_model(args.vae_ckpt, args.vae_config, device)
    
    # 3. Load data
    loader, _ = init_preprocessed_data_loader(
        processed_data_dir=str(args.data_dir),
        batch_size=args.num_states, # Load many states at once
        num_workers=0,
        manifest_path=str(args.manifest) if args.manifest else None,
        split_name=args.split,
    )
    
    # 4. Sample data for analyses
    print(f"Sampling {args.num_states} states and {args.num_actions} actions...")
    sample_states = get_sample_states(loader, args.num_states, device)
    sample_actions = torch.randn(args.num_actions, vae.action_dim, device=device)
    
    # 5. Run analyses
    all_deltas, all_s_next = run_delta_consistency_analysis(vae, sample_states, sample_actions, args.output_dir, run)
    
    run_discriminability_analysis(all_deltas, all_s_next, args.output_dir, run)
    
    # For Jacobian and Composition, use the first sampled state/action
    run_jacobian_analysis(vae, sample_states[0], sample_actions[0], args.output_dir, run)
    
    run_composition_analysis(vae, sample_states[0], sample_actions[0], args.num_composition_steps, args.output_dir, run)
    
    # For retrieval, get a larger set of DB states
    print(f"Sampling {args.num_retrieval_db_size} states for NN retrieval database...")
    db_states = get_sample_states(loader, args.num_retrieval_db_size, device)
    
    # Use the generated states from the first analysis
    retrieval_generated_states = all_s_next.view(-1, *all_s_next.shape[2:])
    
    # Subsample if it's too large
    num_to_test = min(retrieval_generated_states.shape[0], 500)
    indices = torch.randperm(retrieval_generated_states.shape[0])[:num_to_test]
    retrieval_generated_states_sample = retrieval_generated_states[indices]
    
    run_retrieval_analysis(retrieval_generated_states_sample, 
                           db_states, 
                           args.output_dir, 
                           run)

    # 6. Finish
    if run:
        run.finish()
    print("Done.")

if __name__ == "__main__":
    main()
