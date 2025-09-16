import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt

def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD=0.9995) -> torch.Tensor:
    """Spherical linear interpolation."""
    u0 = v0 / v0.norm()
    u1 = v1 / v1.norm()
    dot = (u0 * u1).sum()
    if dot.abs() > DOT_THRESHOLD:
        return (1 - t) * v0 + t * v1 # Linear interpolation for close vectors
    
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
        return validation_cycle_count % self.frequency == 0

    def run(self, model: torch.nn.Module, validation_loader: torch.utils.data.DataLoader, device: torch.device, global_step: int):
        raise NotImplementedError

class DecoderVisualizer(BaseValidator):
    """
    A validator for visualizing the output of the action decoder models.
    It generates trajectory overlay plots and latent space interpolations.
    """
    def __init__(self, frequency: int):
        super().__init__(frequency)

    @torch.no_grad()
    def run(self, model: torch.nn.Module, validation_loader: torch.utils.data.DataLoader, lam_model: torch.nn.Module, device: torch.device, global_step: int):
        """
        Generates and logs visualizations to wandb.
        
        Note: This `run` signature is bespoke for the decoder trainer. It requires the LAM model.
        """
        print(f"Running DecoderVisualizer validator at global step {global_step}...")
        
        # --- 1. Trajectory Overlay Plot ---
        # Get one fixed batch from the validation loader
        # Get a real sample
        try:
            sample_iterator = iter(validation_loader)
            sample1 = next(sample_iterator)
            sample2 = next(sample_iterator)
        except (StopIteration, TypeError):
             return {"validation/visualizer_status": "skipped_not_enough_data"}

        # Correctly get embeddings and ground truth actions
        visual_embeddings1, ground_truth_actions1 = sample1
        visual_embeddings2, ground_truth_actions2 = sample2
        visual_embeddings1 = visual_embeddings1.to(device)
        ground_truth_actions1 = ground_truth_actions1.to(device)
        
        # Generate latent z and predicted trajectory
        mu, logvar = lam_model.encode(visual_embeddings1)
        z = lam_model.reparameterize(mu, logvar)[:, -1, :]
        pred_traj = model(z)

        # Create plot
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        dims = ['x', 'y', 'touch_state']
        
        pred_traj_np = pred_traj.cpu().numpy()[0]
        T_pred = pred_traj_np.shape[0]
        gt_traj_np = ground_truth_actions1.cpu().numpy()[0, :T_pred, :]

        for i in range(3):
            axes[i].plot(gt_traj_np[:, i], label='Ground Truth', color='blue', alpha=0.8)
            axes[i].plot(pred_traj_np[:, i], label='Prediction', color='red', linestyle='--')
            axes[i].set_ylabel(dims[i])
            axes[i].grid(True, linestyle='--', alpha=0.6)
            axes[i].legend()
            
        axes[2].set_xlabel("Timestep")
        fig.suptitle(f"Trajectory Overlay Plot (Step: {global_step})")
        fig.tight_layout()
        
        # Log to wandb
        overlay_plot = wandb.Image(fig)
        plt.close(fig)

        # --- 2. Latent Space Interpolation ---
        # --- 2. Latent Space Interpolation ---
        interp_fig = self._plot_latent_interpolation(model, lam_model, visual_embeddings1, visual_embeddings2, device)
        interp_plot = wandb.Image(interp_fig)
        plt.close(interp_fig)

        # --- 3. Flicker Rate Metric ---
        touch_binary = (pred_traj_np[:, 2] > 0.5).astype(int)
        flicker_rate = np.mean(np.abs(np.diff(touch_binary)))

        return {
            "validation/trajectory_overlay": overlay_plot,
            "validation/latent_interpolation": interp_plot,
            "validation/touch_flicker_rate": flicker_rate
        }

    def _plot_latent_interpolation(self, model, lam_model, embeds1, embeds2, device, n_steps=8):
        """Helper to create and plot latent space interpolations."""
        mu1, logvar1 = lam_model.encode(embeds1.to(device))
        z1 = lam_model.reparameterize(mu1, logvar1)[0, -1, :]
        
        mu2, logvar2 = lam_model.encode(embeds2.to(device))
        z2 = lam_model.reparameterize(mu2, logvar2)[0, -1, :]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Generate trajectories for interpolated latents
        for t in np.linspace(0, 1, n_steps):
            z_interp = slerp(z1, z2, t).unsqueeze(0)
            traj = model(z_interp).cpu().numpy()[0]
            
            color = plt.cm.viridis(t)
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.7, label=f"t={t:.2f}")

        ax.set_title("Latent Space Interpolation (x vs y)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linestyle='--', alpha=0.6)
        return fig