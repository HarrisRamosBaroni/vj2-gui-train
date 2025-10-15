import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD=0.9995) -> torch.Tensor:
    """Spherical linear interpolation."""
    u0 = v0 / v0.norm()
    u1 = v1 / v1.norm()
    dot = (u0 * u1).sum()
    if dot.abs() > DOT_THRESHOLD:
        return (1 - t) * v0 + t * v1
    
    omega = dot.acos()
    sin_omega = omega.sin()
    
    s0 = ((1 - t) * omega).sin() / sin_omega
    s1 = (t * omega).sin() / sin_omega
    
    return s0 * v0 + s1 * v1

class BaseValidator:
    """Base class for validation modules so they all have the same interface."""
    def __init__(self, frequency: int):
        if frequency < 1:
            raise ValueError("Frequency must be at least 1.")
        self.frequency = frequency

    def should_run(self, validation_cycle_count: int) -> bool:
        """Determines if the validator should run based on the cycle count."""
        return validation_cycle_count > 0 and validation_cycle_count % self.frequency == 0

    def run(self, **kwargs):
        raise NotImplementedError

class DecoderVisualizer(BaseValidator):
    """
    A validator for visualizing the output of the action decoder models.
    It generates multi-sample trajectory overlay plots and latent space interpolations.
    """
    def __init__(self, frequency: int, num_overlay_samples: int = 4, num_interp_steps: int = 10):
        super().__init__(frequency)
        self.num_overlay_samples = num_overlay_samples
        self.num_interp_steps = num_interp_steps

    @torch.no_grad()
    def run(self, model: torch.nn.Module, validation_loader: torch.utils.data.DataLoader, lam_model: torch.nn.Module, device: torch.device, global_step: int) -> Dict:
        """
        Generates and logs visualizations to wandb.
        """
        print(f"Running DecoderVisualizer validator at global step {global_step}...")
        
        try:
            sample_batch = next(iter(validation_loader))
        except (StopIteration, TypeError):
            print("Visualizer skipped: Not enough data in validation loader.")
            return {"validation/visualizer_status": "skipped_no_data"}
        
        visual_embeddings, ground_truth_actions = sample_batch
        
        if visual_embeddings.shape[0] < self.num_overlay_samples:
            print(f"Visualizer skipped: Batch size ({visual_embeddings.shape[0]}) is smaller than required number of samples ({self.num_overlay_samples}).")
            return {"validation/visualizer_status": "skipped_small_batch"}

        visual_embeddings = visual_embeddings.to(device)
        ground_truth_actions = ground_truth_actions.to(device)

        # Slice to get the T-1 states corresponding to the T-1 actions
        num_actions = ground_truth_actions.size(1)
        state_embeddings = visual_embeddings[:, :num_actions, :, :]
        
        # Flatten ground truth and state embeddings to match the model's expected input shape
        flat_gt_actions = ground_truth_actions.reshape(-1, ground_truth_actions.size(2), ground_truth_actions.size(3))
        flat_s_batch = state_embeddings.reshape(-1, state_embeddings.size(2), state_embeddings.size(3))
        
        # The LAM requires T states to produce T-1 actions, so we pass the full visual_embeddings
        mu, logvar = lam_model.encode(visual_embeddings)
        z = lam_model.reparameterize(mu, logvar)
        z = z.reshape(-1, z.size(2))
        
        # Prepare kwargs for models that need them
        model_kwargs = {}
        if getattr(model, 'forward_requires_action', False):
            model_kwargs['A'] = flat_gt_actions
        if getattr(model, 'forward_requires_state', False):
            model_kwargs['s'] = flat_s_batch

        model_output = model(z, **model_kwargs)
        pred_traj = model_output["action_trajectory"]

        # --- 1. Trajectory Overlay Plots ---
        overlay_fig = self._plot_trajectory_overlays(pred_traj, flat_gt_actions, global_step)
        
        # --- 2. Latent Space Interpolation ---
        # For interpolation, we pass the flattened z and s batches
        interp_fig = self._plot_latent_interpolation(model, z, flat_s_batch, device)

        # --- 3. Flicker Rate Metric ---
        touch_binary = (pred_traj[:, :, 2].cpu().numpy() > 0.5).astype(int)
        flicker_rate = np.mean(np.abs(np.diff(touch_binary, axis=1)))

        log_dict = {
            "validation/trajectory_overlays": wandb.Image(overlay_fig),
            "validation/latent_interpolation": wandb.Image(interp_fig),
            "validation/touch_flicker_rate": flicker_rate
        }
        
        plt.close(overlay_fig)
        plt.close(interp_fig)
        
        return log_dict

    def _plot_trajectory_overlays(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor, global_step: int) -> plt.Figure:
        """Helper to create a grid of trajectory overlay plots."""
        pred_np = pred_traj.cpu().numpy()
        gt_np = gt_traj.cpu().numpy()
        
        indices = np.random.choice(pred_np.shape[0], size=self.num_overlay_samples, replace=False)
        
        fig, axes = plt.subplots(self.num_overlay_samples, 3, figsize=(15, 4 * self.num_overlay_samples), sharex=True)
        dims = ['x', 'y', 'touch_state']

        for row, sample_idx in enumerate(indices):
            for col in range(3):
                ax = axes[row, col]
                ax.plot(gt_np[sample_idx, :, col], label='Ground Truth', color='blue', alpha=0.8)
                ax.plot(pred_np[sample_idx, :, col], label='Prediction', color='red', linestyle='--')
                ax.grid(True, linestyle='--', alpha=0.6)
                
                if row == 0:
                    ax.set_title(dims[col])
                if col == 0:
                    ax.set_ylabel(f"Sample {sample_idx}")
        
        axes[-1, 1].set_xlabel("Timestep")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.suptitle(f"Trajectory Overlay Plots (Step: {global_step})", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        return fig

    def _plot_latent_interpolation(self, model: torch.nn.Module, z_batch: torch.Tensor, s_batch: torch.Tensor, device: torch.device) -> plt.Figure:
        """Helper to create and plot latent space interpolations within a batch."""
        indices = np.random.choice(z_batch.shape[0], size=2, replace=False)
        z1, z2 = z_batch[indices[0]], z_batch[indices[1]]
        s1, s2 = s_batch[indices[0]], s_batch[indices[1]] # Get corresponding states
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        for t in np.linspace(0, 1, self.num_interp_steps):
            z_interp = slerp(z1, z2, t).unsqueeze(0) # Add batch dimension for model

            # For generative tasks like interpolation, we prefer the .sample() method if it exists (like in ActionCVAE).
            # This avoids needing a ground-truth action `A`.
            # We must still provide the state `s` if required.
            if hasattr(model, 'sample') and callable(getattr(model, 'sample')):
                # We use s1 as the consistent state condition for the interpolation path
                model_output = model.sample(z=z_interp, s=s1.unsqueeze(0))
            else:
                # Fallback for older models or models without a dedicated sample method.
                model_kwargs = {}
                if getattr(model, 'forward_requires_state', False):
                    model_kwargs['s'] = s1.unsqueeze(0)
                model_output = model(z_interp, **model_kwargs)

            traj = model_output["action_trajectory"].cpu().numpy()[0]
            
            color = plt.cm.viridis(t)
            # Plot only where touch is active
            active_steps = traj[:, 2] > 0.5
            ax.plot(traj[active_steps, 0], traj[active_steps, 1], color=color, alpha=0.8)
        
        ax.set_title("Latent Space Interpolation (x vs y)")
        ax.set_xlabel("x coordinate")
        ax.set_ylabel("y coordinate")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axis('equal')

        return fig