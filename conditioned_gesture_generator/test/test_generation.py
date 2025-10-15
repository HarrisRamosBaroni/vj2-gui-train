import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
import wandb

# Ensure the model class can be imported
import sys
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from conditioned_gesture_generator.autoregressive_gesture_decoder import FactorizedAutoregressiveGestureDecoder
from conditioned_gesture_generator.train_autoregressive import GestureTokenizer

def plot_temporal_gesture(values: dict, title: str = "Generated Gesture") -> plt.Figure:
    """
    Plots a gesture's x, y, and touch values over time.
    """
    # Ensure tensors are on CPU and converted to numpy
    x = values['x'].cpu().numpy().flatten()
    y = values['y'].cpu().numpy().flatten()
    touch = values['touch'].cpu().numpy().flatten()

    fig, ax = plt.subplots(figsize=(12, 6))
    timesteps = np.arange(len(x))

    # Plot x, y, and touch as time series
    ax.plot(timesteps, x, 'b-', alpha=0.8, label='X', linewidth=1.5)
    ax.plot(timesteps, y, 'g-', alpha=0.8, label='Y', linewidth=1.5)
    ax.plot(timesteps, touch, 'r-', alpha=0.8, label='Touch', linewidth=1.5)

    # Highlight touch regions
    touch_mask = touch > 0.5
    if np.any(touch_mask):
        ax.fill_between(timesteps, 0, 1, where=touch_mask, alpha=0.2, color='red')

    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='small')

    plt.tight_layout()
    return fig

def main():
    """Main function to run the generation and logging process."""
    parser = argparse.ArgumentParser(description="Generate and visualize gestures from a trained autoregressive model.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the JSON file with model configuration.")
    parser.add_argument("--wandb_project", type=str, default="action-decoding", help="W&B project name for logging.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name.")
    parser.add_argument("--num_actions", type=int, default=4, help="Number of latent actions in the input sequence.")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of unique gestures to generate and plot.")
    parser.add_argument("--use_argmax", action="store_true", help="Use deterministic argmax sampling instead of multinomial.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for multinomial sampling.")
    args = parser.parse_args()

    # Initialize W&B
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    print(f"Initialized W&B run in project '{args.wandb_project}'.")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    with open(args.config_path, 'r') as f:
        model_config = json.load(f)
    print(f"Loaded model configuration from {args.config_path}")

    # Instantiate model
    model = FactorizedAutoregressiveGestureDecoder(**model_config).to(device)

    # Load weights
    try:
        checkpoint = torch.load(args.weights_path, map_location=device)
        # If the checkpoint is a dictionary containing the model state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        if next(iter(state_dict)).startswith('module.'):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Successfully loaded model weights from {args.weights_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        wandb.finish()
        return

    # Instantiate tokenizer
    tokenizer = GestureTokenizer(n_classes=model_config.get('x_classes', 3000))
    print("Instantiated GestureTokenizer.")

    # Generation loop
    for i in range(args.num_samples):
        print(f"Generating sample {i + 1}/{args.num_samples}...")
        # Create a random action sequence
        d_action = model_config.get("d_action", 128)
        action_sequence = torch.randn(1, args.num_actions, d_action).to(device)

        with torch.no_grad():
            generated_tokens = model.generate_full_rollout(
                action_sequence,
                use_argmax=args.use_argmax,
                temperature=args.temperature
            )
        
        # Dequantize the generated tokens to continuous values
        continuous_values_tensor = tokenizer.dequantize(generated_tokens)  # Shape: [1, L, 3]

        # The plot function expects a dictionary of tensors, each of shape [L] but can handle [1, L] via flatten()
        continuous_values_dict = {
            'x': continuous_values_tensor[:, :, 0],
            'y': continuous_values_tensor[:, :, 1],
            'touch': continuous_values_tensor[:, :, 2]
        }
        
        # Plot the gesture
        fig = plot_temporal_gesture(continuous_values_dict, title=f"Generated Sample {i + 1}")
        
        # Log plot to W&B
        wandb.log({f"Generated Sample {i + 1}": wandb.Image(fig)})
        plt.close(fig) # Close figure to free memory

    print("\nAll samples generated and logged to W&B.")
    wandb.finish()

if __name__ == "__main__":
    main()