# The goal is to have a generator for basic actions


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import numpy as np
import time

# ============================
# Differentiable mapping z -> s
# ============================

class ZtoS(nn.Module):
    """Differentiable smooth step/ramp 2D curve generator."""
    def __init__(self, T=200, k=200):
        super().__init__()
        self.T = T
        self.k = k
        t = torch.linspace(0, 1, T)
        self.register_buffer("t", t)

    def forward(self, z):
        """
        z: [batch, 6] -> X0, Y0, R0, X1, Y1, R1
        Returns: curves: [batch, T, 2] where curves[:,:,0] is x and curves[:,:,1] is y
        """
        X0, Y0, R0, X1, Y1, R1 = z[..., 0], z[..., 1], z[..., 2], z[..., 3], z[..., 4], z[..., 5]
        t = self.t.view(1, self.T)

        # Broadcast params
        X0 = X0.view(-1, 1)
        Y0 = Y0.view(-1, 1)
        X1 = X1.view(-1, 1)
        Y1 = Y1.view(-1, 1)
        R0 = R0.view(-1, 1)
        R1 = R1.view(-1, 1)

        # Smooth masks (shared for both x and y)
        left_mask = torch.sigmoid(self.k * (t - R0))
        right_mask = torch.sigmoid(self.k * (R1 - t))
        middle_weight = left_mask * right_mask

        # Generate x coordinate
        ramp_x = X0 + (X1 - X0) * (t - R0) / (R1 - R0 + 1e-8)
        x_t = (1 - left_mask) * X0 + middle_weight * ramp_x + (1 - right_mask) * X1

        # Generate y coordinate
        ramp_y = Y0 + (Y1 - Y0) * (t - R0) / (R1 - R0 + 1e-8)
        y_t = (1 - left_mask) * Y0 + middle_weight * ramp_y + (1 - right_mask) * Y1

        # Stack x and y coordinates
        curves = torch.stack([x_t, y_t], dim=-1)  # [batch, T, 2]
        
        return curves


class ZtoSSharp:
    """Non-differentiable sharp step 2D curve generator using numpy operations.
    
    Creates 2D curves that:
    - Start at (0,0)
    - Jump to (X0,Y0) at time R0
    - Linearly transition from (X0,Y0) to (X1,Y1) between R0 and R1
    - Jump back to (0,0) after R1 (always (0,0) outside R0-R1 interval)
    """
    def __init__(self, T=200):
        self.T = T
        self.t = np.linspace(0, 1, T)
    
    def __call__(self, z):
        """
        z: [batch, 6] -> X0, Y0, R0, X1, Y1, R1 (numpy array or torch tensor)
        Returns: curves: [batch, T, 2] (torch tensor)
        
        Curve behavior:
        - t < R0: curve = (0,0)
        - R0 <= t <= R1: curve linearly transitions from (X0,Y0) to (X1,Y1)
        - t > R1: curve = (0,0) (always jumps back to origin)
        """
        if torch.is_tensor(z):
            z_np = z.cpu().detach().numpy()
            device = z.device
        else:
            z_np = z
            device = torch.device('cpu')
        
        batch_size = z_np.shape[0]
        curves = np.zeros((batch_size, self.T, 2))  # [batch, T, 2] for (x,y)
        
        for b in range(batch_size):
            X0, Y0, R0, X1, Y1, R1 = z_np[b]
            
            # Ensure R0 <= R1 for proper temporal ordering
            if R0 > R1:
                R0, R1 = R1, R0
                X0, Y0, X1, Y1 = X1, Y1, X0, Y0
            
            # Initialize curve to zeros (default state)
            curve = np.zeros((self.T, 2))
            
            # Find transition points (clamp to valid range)
            start_idx = max(0, min(self.T - 1, int(R0 * (self.T - 1))))
            end_idx = max(start_idx, min(self.T - 1, int(R1 * (self.T - 1))))
            
            # Only modify curve between R0 and R1
            if start_idx < self.T and start_idx <= end_idx:
                if start_idx == end_idx:
                    # Single point case
                    curve[start_idx] = [X0, Y0]
                else:
                    # Linear ramp from (X0,Y0) to (X1,Y1) between R0 and R1
                    ramp_length = end_idx - start_idx + 1
                    x_ramp = np.linspace(X0, X1, ramp_length)
                    y_ramp = np.linspace(Y0, Y1, ramp_length)
                    curve[start_idx:end_idx + 1, 0] = x_ramp
                    curve[start_idx:end_idx + 1, 1] = y_ramp
            
            # Curve remains (0,0) everywhere else (before R0 and after R1)
            
            curves[b] = curve
        
        return torch.tensor(curves, dtype=torch.float32, device=device)


# DTW loss function removed as requested


def compute_cem_loss(pred_curves, target_curves, loss_type='mse'):
    """
    Compute loss between predicted and target 2D curves using specified loss function.
    
    Args:
        pred_curves: [batch_size, T, 2] predicted curves (x,y coordinates)
        target_curves: [batch_size, T, 2] target curves (will be expanded to match batch_size)
        loss_type: str, one of ['mse', 'l1', 'l2', 'delta']
        
    Returns:
        losses: [batch_size] loss values (averaged over x and y coordinates)
    """
    # Expand target to match batch size if needed
    if target_curves.shape[0] == 1 and pred_curves.shape[0] > 1:
        target_curves = target_curves.expand_as(pred_curves)
    
    if loss_type == 'mse':
        # Mean Squared Error (L2 squared) - averaged over x,y and time
        losses = ((pred_curves - target_curves) ** 2).mean(dim=(1, 2))
        
    elif loss_type == 'l1':
        # L1 Loss (Mean Absolute Error) - averaged over x,y and time
        losses = torch.abs(pred_curves - target_curves).mean(dim=(1, 2))
        
    elif loss_type == 'l2':
        # L2 Loss (Root Mean Squared Error) - averaged over x,y and time
        losses = torch.sqrt(((pred_curves - target_curves) ** 2).mean(dim=(1, 2)))
        
    elif loss_type == 'delta':
        # Delta Loss: L1 of step-wise differences for both x and y
        pred_deltas = pred_curves[:, 1:, :] - pred_curves[:, :-1, :]  # [batch, T-1, 2]
        target_deltas = target_curves[:, 1:, :] - target_curves[:, :-1, :]  # [batch, T-1, 2]
        losses = torch.abs(pred_deltas - target_deltas).mean(dim=(1, 2))
        
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from ['mse', 'l1', 'l2', 'delta']")
    
    return losses


def CEM(
    model,
    target_s,
    pop_size=256,
    elite_frac=0.1,
    max_iters=50,
    init_mean=None,
    init_std=0.5,
    min_std=1e-6,
    verbose=False,
    loss_type='mse',
    device="cpu",
):
    """
    Cross Entropy Method for optimizing z given a differentiable model.

    Args:
        model: callable(z) -> curves, where z: [batch, 6], curves: [batch, T, 2]
        target_s: tensor of shape [1, T, 2], the target curve
        pop_size: total number of samples per iteration
        elite_frac: fraction of top-performing samples to keep
        max_iters: number of iterations
        init_mean: initial mean for z distribution, defaults to uniform [0,1]
        init_std: initial std deviation for sampling z
        device: torch device

    Returns:
        best_z: [1, 4], optimized latent vector
    """
    # Dimensionality of z (X0, Y0, R0, X1, Y1, R1)
    z_dim = 6
    target_s = target_s.to(device)

    # Initialize Gaussian sampling parameters
    if init_mean is None:
        mean = torch.rand(z_dim, device=device)
    else:
        mean = init_mean.clone().to(device)

    std = torch.ones(z_dim, device=device) * init_std
    n_elite = max(1, int(pop_size * elite_frac))

    best_z = None
    best_loss = float("inf")

    for it in range(max_iters):
        # Sample population
        z_samples = mean + std * torch.randn(pop_size, z_dim, device=device)
        z_samples = torch.sigmoid(z_samples)  # keep z in [0,1] range

        # Evaluate loss using specified loss function
        s_pred = model(z_samples)  # [pop_size, T]
        losses = compute_cem_loss(s_pred, target_s, loss_type)  # [pop_size]

        # Get elites
        elite_idx = torch.topk(-losses, k=n_elite).indices
        elites = z_samples[elite_idx]
        elite_losses = losses[elite_idx]

        # Update Gaussian parameters
        mean = elites.mean(dim=0)
        std = elites.std(dim=0) + 1e-6

        # Track best solution
        min_loss, min_idx = torch.min(elite_losses, dim=0)
        if min_loss.item() < best_loss:
            best_loss = min_loss.item()
            best_z = elites[min_idx].detach().clone()

        # Verbose output
        if verbose and (it % 10 == 0 or it == max_iters - 1):
            std_str = ', '.join([f'{s:.4f}' for s in std.cpu().numpy()])
            print(f"  CEM Iter {it:3d}: Best Loss = {best_loss:.6f}, Std = [{std_str}]")

        # Early stop if std is tiny
        if std.max() < min_std:
            if verbose:
                print(f"  Early stopping at iteration {it}: std.max() = {std.max().item():.2e} < {min_std:.2e}")
            break

    return best_z

# ==========================================
# Test script: recover z from target signal
# ==========================================
def test_differentiability():
    torch.manual_seed(42)

    # Create mapping model
    T = 200
    model = ZtoS(T=T, k=200)

    # Ground truth z
    true_z = torch.tensor([[0.2, 0.3, 0.8, 0.7]], dtype=torch.float32)  # [1, 4]

    # Generate target s'
    with torch.no_grad():
        target_s = model(true_z)

    # Initialize learnable z_hat
    z_hat = torch.randn_like(true_z, requires_grad=True)

    # Optimizer
    optimizer = optim.Adam([z_hat], lr=5e-2)

    # Loss function
    loss_fn = nn.MSELoss()

    losses = []
    for step in range(500):
        optimizer.zero_grad()
        pred_s = model(z_hat)
        loss = loss_fn(pred_s, target_s)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 50 == 0 or step == 499:
            print(f"Step {step:3d} | Loss: {loss.item():.6f} | z_hat = {z_hat.data.cpu().numpy()}")

    # ======================
    # Visualization
    # ======================
    pred_s = model(z_hat).detach()[0].cpu()
    target_s_plot = target_s[0].cpu()

    # Extract x and y components for 2D curves
    pred_x, pred_y = pred_s[:, 0].numpy(), pred_s[:, 1].numpy()
    target_x, target_y = target_s_plot[:, 0].numpy(), target_s_plot[:, 1].numpy()

    plt.figure(figsize=(15, 5))
    
    # Plot X trajectories
    plt.subplot(1, 3, 1)
    plt.plot(pred_x, label="Predicted X", linewidth=2)
    plt.plot(target_x, label="Target X", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("X Coordinate")
    plt.legend()
    plt.title("Recovered X Trajectory")
    plt.grid(True, alpha=0.3)
    
    # Plot Y trajectories
    plt.subplot(1, 3, 2)
    plt.plot(pred_y, label="Predicted Y", linewidth=2)
    plt.plot(target_y, label="Target Y", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.title("Recovered Y Trajectory")
    plt.grid(True, alpha=0.3)
    
    # Plot 2D curves (excluding (0,0) points)
    plt.subplot(1, 3, 3)
    # Filter out (0,0) points for target
    target_nonzero_mask = ~((target_x == 0) & (target_y == 0))
    if target_nonzero_mask.any():
        plt.plot(target_x[target_nonzero_mask], target_y[target_nonzero_mask], 
                label="Target", linestyle="--", linewidth=2, marker='s', markersize=3)
    
    # Filter out (0,0) points for prediction
    pred_nonzero_mask = ~((pred_x == 0) & (pred_y == 0))
    if pred_nonzero_mask.any():
        plt.plot(pred_x[pred_nonzero_mask], pred_y[pred_nonzero_mask], 
                label="Predicted", linewidth=2, marker='o', markersize=3)
    
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.title("2D Curve Comparison")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

    # Plot loss curve
    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Optimization Convergence")
    plt.show()

def generation_test(model, device="cpu", num_samples=10):
    """
    Generation Test: Sample 10 latents and run them through the decoder,
    plot all results on one graph.
    """
    print("\n=== Generation Test ===")
    torch.manual_seed(42)
    
    # Sample 10 random latent vectors [X0, Y0, R0, X1, Y1, R1]
    z_samples = torch.rand(num_samples, 6, device=device)
    
    # Generate 2D curves
    with torch.no_grad():
        curves_samples = model(z_samples)  # [num_samples, T, 2]
    
    # Plot all 2D trajectories
    plt.figure(figsize=(15, 5))
    
    # Plot 1: X trajectories over time
    plt.subplot(1, 3, 1)
    for i in range(num_samples):
        x_traj = curves_samples[i, :, 0].cpu().numpy()
        plt.plot(x_traj, label=f'Sample {i+1}', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('X Coordinate')
    plt.title('X Trajectories')
    plt.grid(True, alpha=0.3)
    if num_samples <= 5:
        plt.legend()
    
    # Plot 2: Y trajectories over time
    plt.subplot(1, 3, 2)
    for i in range(num_samples):
        y_traj = curves_samples[i, :, 1].cpu().numpy()
        plt.plot(y_traj, label=f'Sample {i+1}', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Y Coordinate')
    plt.title('Y Trajectories')
    plt.grid(True, alpha=0.3)
    if num_samples <= 5:
        plt.legend()
    
    # Plot 3: 2D curves in coordinate space (excluding (0,0) points)
    plt.subplot(1, 3, 3)
    for i in range(num_samples):
        x_traj = curves_samples[i, :, 0].cpu().numpy()
        y_traj = curves_samples[i, :, 1].cpu().numpy()
        
        # Filter out (0,0) points for cleaner 2D visualization
        nonzero_mask = ~((x_traj == 0) & (y_traj == 0))
        if nonzero_mask.any():
            x_filtered = x_traj[nonzero_mask]
            y_filtered = y_traj[nonzero_mask]
            plt.plot(x_filtered, y_filtered, label=f'Sample {i+1}', alpha=0.7, marker='o', markersize=2)
            # Mark start and end points of the filtered trajectory
            plt.plot(x_filtered[0], y_filtered[0], 'go', markersize=6, alpha=0.8)  # start
            plt.plot(x_filtered[-1], y_filtered[-1], 'ro', markersize=6, alpha=0.8)  # end
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('2D Curves (Green=Start, Red=End)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    if num_samples <= 5:
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Generated {num_samples} samples with latent vectors [X0, Y0, R0, X1, Y1, R1]:")
    for i, z in enumerate(z_samples):
        print(f"Sample {i+1}: z = {z.cpu().numpy()}")


def gradient_descent_comparison(model, target_s, device="cpu", num_steps=500):
    """
    Compare SGD vs Adam optimization for the same reconstruction goal.
    Shows both reconstruction results and loss curves.
    """
    print("\n=== Gradient Descent Comparison ===")
    torch.manual_seed(42)
    
    # Initialize two separate z_hat for fair comparison
    z_hat_sgd = torch.randn(1, 6, device=device, requires_grad=True)
    z_hat_adam = z_hat_sgd.clone().detach().requires_grad_(True)
    
    # Optimizers
    optimizer_sgd = optim.SGD([z_hat_sgd], lr=1e-1)
    optimizer_adam = optim.Adam([z_hat_adam], lr=5e-2)
    
    loss_fn = nn.MSELoss()
    losses_sgd = []
    losses_adam = []
    
    # Time SGD optimization
    print("Running SGD optimization...")
    sgd_start_time = time.time()
    for step in range(num_steps):
        # SGD step
        optimizer_sgd.zero_grad()
        pred_s_sgd = model(z_hat_sgd)
        loss_sgd = loss_fn(pred_s_sgd, target_s)
        loss_sgd.backward()
        optimizer_sgd.step()
        losses_sgd.append(loss_sgd.item())
        
        if step % 100 == 0:
            print(f"SGD Step {step:3d} | Loss: {loss_sgd.item():.6f}")
    sgd_time = time.time() - sgd_start_time
    
    # Time Adam optimization
    print("Running Adam optimization...")
    adam_start_time = time.time()
    for step in range(num_steps):
        # Adam step
        optimizer_adam.zero_grad()
        pred_s_adam = model(z_hat_adam)
        loss_adam = loss_fn(pred_s_adam, target_s)
        loss_adam.backward()
        optimizer_adam.step()
        losses_adam.append(loss_adam.item())
        
        if step % 100 == 0:
            print(f"Adam Step {step:3d} | Loss: {loss_adam.item():.6f}")
    adam_time = time.time() - adam_start_time
    
    # Generate final predictions
    with torch.no_grad():
        pred_s_sgd = model(z_hat_sgd)
        pred_s_adam = model(z_hat_adam)
    
    # Get x,y trajectories for 2D curves  
    target_x = target_s[0, :, 0].cpu().numpy()
    target_y = target_s[0, :, 1].cpu().numpy()
    pred_x_sgd = pred_s_sgd[0, :, 0].cpu().numpy()
    pred_y_sgd = pred_s_sgd[0, :, 1].cpu().numpy()
    pred_x_adam = pred_s_adam[0, :, 0].cpu().numpy()
    pred_y_adam = pred_s_adam[0, :, 1].cpu().numpy()
    
    # Plot reconstructions
    plt.figure(figsize=(20, 5))
    
    # X coordinate comparison
    plt.subplot(1, 4, 1)
    plt.plot(target_x, label="Target", linewidth=3, color='black')
    plt.plot(pred_x_sgd, label="SGD", linewidth=2, alpha=0.8)
    plt.plot(pred_x_adam, label="Adam", linewidth=2, alpha=0.8)
    plt.xlabel("Time")
    plt.ylabel("X Coordinate")
    plt.title("X Reconstruction Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Y coordinate comparison  
    plt.subplot(1, 4, 2)
    plt.plot(target_y, label="Target", linewidth=3, color='black')
    plt.plot(pred_y_sgd, label="SGD", linewidth=2, alpha=0.8)
    plt.plot(pred_y_adam, label="Adam", linewidth=2, alpha=0.8)
    plt.xlabel("Time")
    plt.ylabel("Y Coordinate") 
    plt.title("Y Reconstruction Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss curves comparison
    plt.subplot(1, 4, 3)
    plt.plot(losses_sgd, label="SGD", linewidth=2)
    plt.plot(losses_adam, label="Adam", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss (MSE)")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(1, 4, 4)
    plt.plot(losses_sgd[-100:], label="SGD (last 100)", linewidth=2)
    plt.plot(losses_adam[-100:], label="Adam (last 100)", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss (MSE)")
    plt.title("Loss Curves (Final 100 steps)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal Results:")
    print(f"SGD   - Final Loss: {losses_sgd[-1]:.6f}, Time: {sgd_time:.3f}s, z: {z_hat_sgd.detach().cpu().numpy()}")
    print(f"Adam  - Final Loss: {losses_adam[-1]:.6f}, Time: {adam_time:.3f}s, z: {z_hat_adam.detach().cpu().numpy()}")
    print(f"Speed comparison: Adam was {sgd_time/adam_time:.2f}x {'faster' if adam_time < sgd_time else 'slower'} than SGD")


def cem_optimization_test(model, target_s, device="cpu", pop_size=256, elite_frac=0.1, max_iters=50, 
                         init_std=0.5, init_mean=None, min_std=1e-6, verbose=False, loss_type='mse'):
    """
    CEM Optimization test for approaching a goal.
    """
    print("\n=== CEM Optimization Test ===")
    print(f"CEM Parameters:")
    print(f"  pop_size={pop_size}, elite_frac={elite_frac}, max_iters={max_iters}")
    print(f"  init_std={init_std}, min_std={min_std}, verbose={verbose}")
    print(f"  loss_type={loss_type}")
    if init_mean is not None:
        print(f"  init_mean={init_mean}")
    else:
        print(f"  init_mean=random")
    
    # Convert init_mean to torch tensor if provided
    if init_mean is not None:
        init_mean_tensor = torch.tensor(init_mean, dtype=torch.float32, device=device)
    else:
        init_mean_tensor = None
    
    # Time CEM optimization
    cem_start_time = time.time()
    best_z = CEM(
        model=model,
        target_s=target_s,
        pop_size=pop_size,
        elite_frac=elite_frac,
        max_iters=max_iters,
        init_std=init_std,
        init_mean=init_mean_tensor,
        min_std=min_std,
        verbose=verbose,
        loss_type=loss_type,
        device=device
    )
    cem_time = time.time() - cem_start_time
    
    # Generate prediction with best z
    with torch.no_grad():
        pred_curves = model(best_z.unsqueeze(0))
        final_loss = compute_cem_loss(pred_curves, target_s, loss_type).item()
    
    # Extract x and y trajectories
    target_x = target_s[0, :, 0].cpu().numpy()
    target_y = target_s[0, :, 1].cpu().numpy()
    pred_x = pred_curves[0, :, 0].cpu().numpy()
    pred_y = pred_curves[0, :, 1].cpu().numpy()
    
    # Plot results with separate x and y subplots
    plt.figure(figsize=(12, 5))
    
    # X coordinate comparison
    plt.subplot(1, 2, 1)
    plt.plot(target_x, label="Target X", linewidth=3, color='black')
    plt.plot(pred_x, label="CEM Result X", linewidth=2, alpha=0.8, color='red')
    plt.xlabel("Time")
    plt.ylabel("X Coordinate")
    plt.title("CEM Optimization: X Dimension")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Y coordinate comparison
    plt.subplot(1, 2, 2)
    plt.plot(target_y, label="Target Y", linewidth=3, color='black')
    plt.plot(pred_y, label="CEM Result Y", linewidth=2, alpha=0.8, color='blue')
    plt.xlabel("Time")
    plt.ylabel("Y Coordinate")
    plt.title("CEM Optimization: Y Dimension")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nCEM Results:")
    print(f"Best z: {best_z.cpu().numpy()}")
    print(f"Final Loss: {final_loss:.6f}")
    print(f"Optimization Time: {cem_time:.3f}s")


def create_target_signal(model, device="cpu", use_random=True):
    """
    Create a target 2D curve for optimization experiments.
    """
    if use_random:
        # Generate random target z: [X0, Y0, R0, X1, Y1, R1]
        true_z = torch.rand(1, 6, dtype=torch.float32, device=device)
        print(f"Random target created with z: {true_z.cpu().numpy()}")
    else:
        # Use fixed target z: [X0, Y0, R0, X1, Y1, R1]
        true_z = torch.tensor([[0.2, 0.3, 0.3, 0.8, 0.7, 0.7]], dtype=torch.float32, device=device)
        print(f"Fixed target created with z: {true_z.cpu().numpy()}")
    
    with torch.no_grad():
        target_s = model(true_z)
    
    return target_s, true_z


def main():
    parser = argparse.ArgumentParser(description="Decoder experiments")
    
    # Experiment selection flags
    parser.add_argument("--SAMPLE", action="store_true", help="Run generation test")
    parser.add_argument("--SGD", action="store_true", help="Run SGD optimization experiment")
    parser.add_argument("--ADAM", action="store_true", help="Run Adam optimization experiment")
    parser.add_argument("--CEM", action="store_true", help="Run CEM optimization experiment")
    
    # Decoder selection flags
    parser.add_argument("--smooth", action="store_true", help="Use smooth differentiable decoder (ZtoS)")
    parser.add_argument("--sharp", action="store_true", help="Use sharp non-differentiable decoder (ZtoSSharp)")
    
    # CEM parameters
    parser.add_argument("--cem-pop-size", type=int, default=256, help="CEM population size (default: 256)")
    parser.add_argument("--cem-elite-frac", type=float, default=0.1, help="CEM elite fraction (default: 0.1)")
    parser.add_argument("--cem-max-iters", type=int, default=50, help="CEM maximum iterations (default: 50)")
    parser.add_argument("--cem-init-std", type=float, default=0.5, help="CEM initial standard deviation (default: 0.5)")
    parser.add_argument("--cem-init-mean", type=float, nargs=6, default=None, help="CEM initial mean [X0, Y0, R0, X1, Y1, R1] (default: random)")
    parser.add_argument("--cem-min-std", type=float, default=1e-6, help="CEM minimum standard deviation for early stopping (default: 1e-6)")
    parser.add_argument("--cem-verbose", action="store_true", help="Show CEM progress every 10 iterations")
    parser.add_argument("--cem-loss", choices=['mse', 'l1', 'l2', 'delta'], default='mse', help="CEM loss function: mse (default), l1, l2, delta")
    
    # Other options
    parser.add_argument("--fixed-target", action="store_true", help="Use fixed target instead of random")
    
    args = parser.parse_args()
    
    # If no flags are provided, run the original test
    if not any([args.SAMPLE, args.SGD, args.ADAM, args.CEM]):
        print("No experiment flags provided. Running original test...")
        test_differentiability()
        return
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Decoder selection
    if args.sharp and args.smooth:
        print("Error: Cannot use both --sharp and --smooth flags simultaneously")
        return
    elif args.sharp:
        decoder_name = "Sharp (Non-differentiable)"
        model = ZtoSSharp(T=200)
        print(f"Using decoder: {decoder_name}")
    elif args.smooth:
        decoder_name = "Smooth (Differentiable)"
        model = ZtoS(T=200, k=200).to(device)
        print(f"Using decoder: {decoder_name}")
    else:
        # Default to smooth decoder
        decoder_name = "Smooth (Differentiable) [default]"
        model = ZtoS(T=200, k=200).to(device)
        print(f"Using decoder: {decoder_name}")
    
    # Create target signal for optimization experiments
    use_random_target = not args.fixed_target
    target_s, true_z = create_target_signal(model, device, use_random=use_random_target)
    
    # Run experiments based on flags
    if args.SAMPLE:
        generation_test(model, device)
    
    if args.SGD or args.ADAM:
        # Check if trying to use gradient-based optimization with non-differentiable decoder
        if args.sharp and (args.SGD or args.ADAM):
            print("\\nWarning: Cannot use gradient-based optimization (SGD/Adam) with sharp non-differentiable decoder.")
            print("Gradient-based optimization requires the smooth differentiable decoder.")
            print("Please use --smooth flag or remove --sharp flag.")
        elif args.SGD and args.ADAM:
            gradient_descent_comparison(model, target_s, device)
        elif args.SGD:
            print("\n=== SGD Only ===")
            # Run SGD only version
            torch.manual_seed(42)
            z_hat = torch.randn(1, 4, device=device, requires_grad=True)
            optimizer = optim.SGD([z_hat], lr=1e-1)
            loss_fn = nn.MSELoss()
            losses = []
            
            # Time SGD optimization
            sgd_start_time = time.time()
            for step in range(500):
                optimizer.zero_grad()
                pred_s = model(z_hat)
                loss = loss_fn(pred_s, target_s)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
                if step % 100 == 0:
                    print(f"Step {step:3d} | Loss: {loss.item():.6f}")
            sgd_time = time.time() - sgd_start_time
            
            # Plot results
            with torch.no_grad():
                pred_s = model(z_hat)
            
            # Extract x,y coordinates for 2D curves
            target_x = target_s[0, :, 0].cpu().numpy()
            target_y = target_s[0, :, 1].cpu().numpy()
            pred_x = pred_s[0, :, 0].cpu().numpy()
            pred_y = pred_s[0, :, 1].cpu().numpy()
            
            plt.figure(figsize=(15, 4))
            
            # X coordinate
            plt.subplot(1, 3, 1)
            plt.plot(target_x, label="Target X", linewidth=3)
            plt.plot(pred_x, label="SGD X", linewidth=2, alpha=0.8)
            plt.xlabel("Time")
            plt.ylabel("X Coordinate")
            plt.title("SGD X Reconstruction")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Y coordinate
            plt.subplot(1, 3, 2)
            plt.plot(target_y, label="Target Y", linewidth=3)
            plt.plot(pred_y, label="SGD Y", linewidth=2, alpha=0.8)
            plt.xlabel("Time")
            plt.ylabel("Y Coordinate")
            plt.title("SGD Y Reconstruction")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Loss curve
            plt.subplot(1, 3, 3)
            plt.plot(losses)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("SGD Loss Curve")
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            print(f"\nSGD Results:")
            print(f"Final Loss: {losses[-1]:.6f}")
            print(f"Optimization Time: {sgd_time:.3f}s")
            print(f"Final z: {z_hat.detach().cpu().numpy()}")
            
        elif args.ADAM:
            print("\n=== Adam Only ===")
            # Run Adam only version (similar to SGD but with Adam optimizer)
            torch.manual_seed(42)
            z_hat = torch.randn(1, 4, device=device, requires_grad=True)
            optimizer = optim.Adam([z_hat], lr=5e-2)
            loss_fn = nn.MSELoss()
            losses = []
            
            # Time Adam optimization
            adam_start_time = time.time()
            for step in range(500):
                optimizer.zero_grad()
                pred_s = model(z_hat)
                loss = loss_fn(pred_s, target_s)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
                if step % 100 == 0:
                    print(f"Step {step:3d} | Loss: {loss.item():.6f}")
            adam_time = time.time() - adam_start_time
            
            # Plot results
            with torch.no_grad():
                pred_s = model(z_hat)
            
            # Extract x,y coordinates for 2D curves
            target_x = target_s[0, :, 0].cpu().numpy()
            target_y = target_s[0, :, 1].cpu().numpy()
            pred_x = pred_s[0, :, 0].cpu().numpy()
            pred_y = pred_s[0, :, 1].cpu().numpy()
            
            plt.figure(figsize=(15, 4))
            
            # X coordinate
            plt.subplot(1, 3, 1)
            plt.plot(target_x, label="Target X", linewidth=3)
            plt.plot(pred_x, label="Adam X", linewidth=2, alpha=0.8)
            plt.xlabel("Time")
            plt.ylabel("X Coordinate")
            plt.title("Adam X Reconstruction")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Y coordinate
            plt.subplot(1, 3, 2)
            plt.plot(target_y, label="Target Y", linewidth=3)
            plt.plot(pred_y, label="Adam Y", linewidth=2, alpha=0.8)
            plt.xlabel("Time")
            plt.ylabel("Y Coordinate")
            plt.title("Adam Y Reconstruction")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Loss curve
            plt.subplot(1, 3, 3)
            plt.plot(losses)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Adam Loss Curve")
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            print(f"\nAdam Results:")
            print(f"Final Loss: {losses[-1]:.6f}")
            print(f"Optimization Time: {adam_time:.3f}s")
            print(f"Final z: {z_hat.detach().cpu().numpy()}")
    
    if args.CEM:
        cem_optimization_test(
            model, target_s, device,
            pop_size=args.cem_pop_size,
            elite_frac=args.cem_elite_frac,
            max_iters=args.cem_max_iters,
            init_std=args.cem_init_std,
            init_mean=args.cem_init_mean,
            min_std=args.cem_min_std,
            verbose=args.cem_verbose,
            loss_type=args.cem_loss
        )


if __name__ == "__main__":
    main()
