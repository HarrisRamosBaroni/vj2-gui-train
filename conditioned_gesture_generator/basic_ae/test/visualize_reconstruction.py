import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset

# project-specific modules
from config import ACTION_DIM, ACTIONS_PER_BATCH
from conditioned_gesture_generator.basic_ae.transformer_ae import TinyTransformerAutoencoder

class ActionBlockDataset(Dataset):
    """
    A PyTorch Dataset for loading action blocks from preprocessed .npy files.
    Copied from training/train_action_autoencoder.py for self-containment.
    """
    def __init__(self, data_dir, stride=ACTIONS_PER_BATCH):
        self.data_dir = Path(data_dir)
        self.stride = stride
        self.file_paths = list(self.data_dir.glob("*_actions.npy"))
        
        if not self.file_paths:
            raise ValueError(f"No '_actions.npy' files found in {self.data_dir}")

        self.indices = []
        for file_idx, file_path in enumerate(self.file_paths):
            try:
                data_shape = np.load(file_path, mmap_mode='r').shape
                total_actions = data_shape[0]
                
                max_start_index = total_actions - ACTIONS_PER_BATCH
                if max_start_index >= 0:
                    file_indices = [(file_idx, i) for i in range(0, max_start_index + 1, self.stride)]
                    self.indices.extend(file_indices)
            except Exception as e:
                print(f"Warning: Could not process file {file_path}: {e}")

        if not self.indices:
            raise ValueError("No valid action blocks could be found with the given stride "
                             f"({self.stride}) in the provided data directory.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, start_idx = self.indices[idx]
        file_path = self.file_paths[file_idx]

        data = np.load(file_path, mmap_mode='r')
        action_block_np = np.array(data[start_idx : start_idx + ACTIONS_PER_BATCH])
        
        action_block = torch.from_numpy(action_block_np).float()
        return action_block


def plot_reconstruction(original_block, reconstructed_block, sample_idx, output_dir):
    """
    Plots the original and reconstructed action blocks.
    Only plots reconstructed points where touch state >= 0.5.
    """
    plt.figure(figsize=(10, 5))
    
    # Original actions (only if touch state >= 0.5)
    original_x_active = []
    original_y_active = []
    for i in range(original_block.shape[0]):
        if original_block[i, 2] >= 0.5: # Check touch state
            original_x_active.append(original_block[i, 0])
            original_y_active.append(original_block[i, 1])
    
    plt.plot(original_x_active, original_y_active, 'b-', label='Original (x, y, touch >= 0.5)')
    
    # Reconstructed actions (only if touch state >= 0.5)
    reconstructed_x = []
    reconstructed_y = []
    for i in range(reconstructed_block.shape[0]):
        if reconstructed_block[i, 2] >= 0.5: # Check touch state
        # if True:
            reconstructed_x.append(reconstructed_block[i, 0])
            reconstructed_y.append(reconstructed_block[i, 1])
    
    if reconstructed_x: # Only plot if there are points to plot
        plt.plot(reconstructed_x, reconstructed_y, 'r--', label='Reconstructed (x, y, touch >= 0.5)')
    
    plt.title(f"Action Block Reconstruction - Sample {sample_idx}")
    plt.xlabel("Normalized X Coordinate")
    plt.ylabel("Normalized Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.xlim(-1, 2) # Assuming normalized coordinates
    plt.ylim(-1, 2) # Assuming normalized coordinates
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"reconstruction_sample_{sample_idx}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    test_dataset = ActionBlockDataset(args.test_dir)
    print(f"Found {len(test_dataset)} test samples.")

    # Initialize model and load checkpoint
    model = TinyTransformerAutoencoder(
        FEAT_DIM=ACTION_DIM,
        MODEL_DIM=args.model_dim,
        SEQ_LEN=ACTIONS_PER_BATCH,
        LATENT_DIM=args.latent_dim
    ).to(device)

    if not Path(args.checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint_path}")

    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval() # Set model to evaluation mode

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select random samples and plot
    sample_indices = random.sample(range(len(test_dataset)), min(args.num_samples, len(test_dataset)))
    
    print(f"Generating plots for {len(sample_indices)} random samples...")
    for i, idx in enumerate(sample_indices):
        original_block_tensor = test_dataset[idx].unsqueeze(0).to(device) # Add batch dim
        
        with torch.no_grad():
            _, reconstructed_block_tensor = model(original_block_tensor)
        
        original_block_np = original_block_tensor.squeeze(0).cpu().numpy()
        reconstructed_block_np = reconstructed_block_tensor.squeeze(0).cpu().numpy()
        
        plot_reconstruction(original_block_np, reconstructed_block_np, i + 1, output_dir)

    print("\nReconstruction visualization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize action block reconstructions from a trained autoencoder.")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained model checkpoint (e.g., 'checkpoints/best.pt').")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to the directory containing test _actions.npy files.")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of random samples to visualize.")
    parser.add_argument("--output_dir", type=str, default="reconstruction_plots",
                        help="Directory to save the output plots.")
    
    # Model hyperparameters (must match the trained model)
    parser.add_argument("--model_dim", type=int, default=64,
                        help="The internal dimension of the transformer model (must match trained model).")
    parser.add_argument("--latent_dim", type=int, default=10,
                        help="The dimension of the compressed latent representation (must match trained model).")

    args = parser.parse_args()
    main(args)