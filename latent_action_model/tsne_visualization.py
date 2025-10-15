"""
t-SNE Visualization for LAM Action Latents

This script extracts action latents from N trajectories using LAM encoder,
applies t-SNE to reduce them to 2D, and creates a wandb table with the
2D coordinates and corresponding 4 real frames for each action.
"""

import argparse
import os
import json
import yaml
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
from PIL import Image
import wandb
from typing import List, Tuple, Dict, Any
import random

from latent_action_model.vae import load_model_from_config, LatentActionVAE


class LAMTSNEVisualizer:
    def __init__(
        self,
        model_path: str,
        config_path: str,
        processed_data_dir: str,
        manifest_path: str,
        image_dir: str,
        device: str = "cuda"
    ):
        self.processed_data_dir = processed_data_dir
        self.manifest_path = manifest_path
        self.image_dir = image_dir
        self.device = device

        # Load model and config
        self.lam_model = self._load_model(model_path, config_path)
        self.lam_model.eval()

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

    def _load_model(self, model_path: str, config_path: str) -> LatentActionVAE:
        """Load LAM model with config"""
        config = {}

        # Load config if provided
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Extract parameters from wandb YAML format
            for key, value_dict in yaml_config.items():
                if isinstance(value_dict, dict) and 'value' in value_dict:
                    config[key] = value_dict['value']
                elif not key.startswith('_'):
                    config[key] = value_dict

            # Get encoder/decoder parameters from args in wandb config
            if '_wandb' in yaml_config and 'value' in yaml_config['_wandb']:
                wandb_config = yaml_config['_wandb']['value']
                if 'e' in wandb_config:
                    for run_id, run_data in wandb_config['e'].items():
                        if 'args' in run_data:
                            args = run_data['args']
                            for i in range(0, len(args), 2):
                                if i + 1 < len(args):
                                    arg_name = args[i].replace('--', '')
                                    arg_value = args[i + 1]
                                    try:
                                        config[arg_name] = int(arg_value)
                                    except ValueError:
                                        try:
                                            config[arg_name] = float(arg_value)
                                        except ValueError:
                                            config[arg_name] = arg_value

        # Set defaults if not in config
        config.setdefault('action_dim', 128)
        config.setdefault('latent_dim', 1024)

        print(f"Using config: {config}")

        # Create model directly
        model = LatentActionVAE(
            action_dim=config['action_dim'],
            latent_dim=config['latent_dim'],
            encoder_depth=config.get('encoder_depth', 8),
            decoder_depth=config.get('decoder_depth', 8),
            encoder_heads=config.get('encoder_heads', 8),
            decoder_heads=config.get('decoder_heads', 8),
            embed_dim=config.get('embed_dim', 512)
        )

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        return model

    def _load_trajectory_data(self, session_id: str, traj_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load JEPA embeddings and action for a single trajectory"""
        processed_file = os.path.join(self.processed_data_dir, "processed_data", f"{session_id}.h5")

        with h5py.File(processed_file, 'r') as f:
            # Load JEPA embeddings for this trajectory
            jepa_key = f"jepa_latent_{traj_idx:06d}"
            if jepa_key not in f:
                raise KeyError(f"JEPA key {jepa_key} not found in {processed_file}")

            jepa_embeddings = torch.tensor(f[jepa_key][:], dtype=torch.float32)  # [T, N, D]

            # Load action for this trajectory
            action_key = f"action_{traj_idx:06d}"
            if action_key not in f:
                raise KeyError(f"Action key {action_key} not found in {processed_file}")

            action = torch.tensor(f[action_key][:], dtype=torch.float32)  # [action_dim]

        return jepa_embeddings, action

    def _load_corresponding_images(self, session_id: str, traj_idx: int, T_jepa: int, action_step: int) -> List[Image.Image]:
        """
        Load 4 real frames corresponding to an action at given step.
        Each action corresponds to JEPA frames [action_step, action_step+1],
        which map to real frames [2*action_step, 2*action_step+1, 2*action_step+2, 2*action_step+3]
        """
        images = []

        # Calculate real frame indices (4 frames total)
        start_frame = 2 * action_step
        frame_indices = [start_frame + i for i in range(4)]

        for frame_idx in frame_indices:
            # Construct image path
            image_filename = f"image_{traj_idx:06d}_{frame_idx:06d}.jpg"
            image_path = os.path.join(self.image_dir, session_id, image_filename)

            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                images.append(image)
            else:
                print(f"Warning: Image not found: {image_path}")
                # Create a black placeholder image
                images.append(Image.new('RGB', (224, 224), (0, 0, 0)))

        return images

    def _load_corresponding_images_from_h5(self, h5_path, traj_idx: int, T_jepa: int, action_step: int) -> List[Image.Image]:
        """
        Load 4 real frames corresponding to an action at given step from H5 file.
        Each action corresponds to JEPA frames [action_step, action_step+1],
        which map to real frames [2*action_step, 2*action_step+1, 2*action_step+2, 2*action_step+3]
        """
        images = []

        try:
            with h5py.File(h5_path, 'r') as f:
                if 'images' not in f:
                    print(f"No 'images' key in {h5_path}")
                    return [Image.new('RGB', (224, 224), (0, 0, 0)) for _ in range(4)]

                # Calculate real frame indices (4 frames total)
                start_frame = 2 * action_step
                frame_indices = [start_frame + i for i in range(4)]

                for frame_idx in frame_indices:
                    if frame_idx < f['images'].shape[1]:  # Check if frame exists
                        img_data = f['images'][traj_idx, frame_idx]  # [H, W, C]
                        img = Image.fromarray(img_data.astype('uint8')).convert('RGB')
                        images.append(img)
                    else:
                        # Create a black placeholder image if frame doesn't exist
                        images.append(Image.new('RGB', (224, 224), (0, 0, 0)))

        except Exception as e:
            print(f"Error loading images from {h5_path}: {e}")
            images = [Image.new('RGB', (224, 224), (0, 0, 0)) for _ in range(4)]

        return images

    def _load_corresponding_images_from_images_h5(self, session_id: str, traj_idx: int, T_jepa: int, action_step: int) -> List[Image.Image]:
        """
        Load 4 real frames corresponding to an action at given step from images H5 file.
        Each action corresponds to JEPA frames [action_step, action_step+1],
        which map to real frames [2*action_step, 2*action_step+1, 2*action_step+2, 2*action_step+3]
        """
        images = []

        # Construct path to images H5 file
        images_h5_path = os.path.join(self.image_dir, f"{session_id}_images.h5")

        try:
            with h5py.File(images_h5_path, 'r') as f:
                if 'images' not in f:
                    print(f"No 'images' key in {images_h5_path}")
                    return [Image.new('RGB', (224, 224), (0, 0, 0)) for _ in range(4)]

                # Calculate real frame indices (4 frames total)
                start_frame = 2 * action_step
                frame_indices = [start_frame + i for i in range(4)]

                for frame_idx in frame_indices:
                    if traj_idx < f['images'].shape[0] and frame_idx < f['images'].shape[1]:
                        img_data = f['images'][traj_idx, frame_idx]  # [H, W, C]
                        img = Image.fromarray(img_data.astype('uint8')).convert('RGB')
                        images.append(img)
                    else:
                        # Create a black placeholder image if frame doesn't exist
                        images.append(Image.new('RGB', (224, 224), (0, 0, 0)))

        except Exception as e:
            print(f"Error loading images from {images_h5_path}: {e}")
            images = [Image.new('RGB', (224, 224), (0, 0, 0)) for _ in range(4)]

        return images

    def extract_action_latents(self, n_trajectories: int, split_name: str = "train") -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        """
        Extract action latents from N trajectories and their metadata

        Returns:
            action_latents: List of action latent tensors
            metadata: List of dicts with filename, traj_idx, action_step, images
        """
        from pathlib import Path

        action_latents = []
        metadata = []

        # Load manifest to get file list (same as attention_visualization.py)
        if split_name not in self.manifest['splits']:
            raise ValueError(f"Split '{split_name}' not found in manifest")

        file_list = self.manifest['splits'][split_name][:n_trajectories]
        processed_dir = Path(self.processed_data_dir)

        print(f"Processing {len(file_list)} files...")

        for filename in file_list:
            h5_path = processed_dir / filename
            if not h5_path.exists():
                print(f"Warning: File not found: {h5_path}")
                continue

            try:
                with h5py.File(h5_path, 'r') as f:
                    embeddings = f['embeddings'][:]  # [n_traj, T, N, D] or [T, N, D]

                    # Handle different shapes
                    if len(embeddings.shape) == 4:
                        n_traj, T, N, D = embeddings.shape
                    elif len(embeddings.shape) == 3:
                        T, N, D = embeddings.shape
                        n_traj = 1
                        embeddings = embeddings[None, ...]  # Add batch dimension: [1, T, N, D]
                    else:
                        print(f"Unexpected embeddings shape: {embeddings.shape}")
                        continue

                    print(f"Processing {filename}: {n_traj} trajectories, T={T}, N={N}, D={D}")

                    for traj_idx in range(n_traj):
                        try:
                            jepa_embeddings = torch.tensor(embeddings[traj_idx], dtype=torch.float32)  # [T, N, D]

                            # Move to device
                            jepa_embeddings = jepa_embeddings.to(self.device)

                            # Apply z-score normalization to JEPA embeddings
                            jepa_normalized = F.layer_norm(jepa_embeddings, jepa_embeddings.shape[-1:])

                            # Process entire trajectory at once to get all action latents
                            # Add batch dimension for LAM encoder: [1, T, N, D]
                            z_batch = jepa_normalized.unsqueeze(0)  # [1, T, N, D]

                            with torch.no_grad():
                                try:
                                    # Encode to get all action latents for this trajectory
                                    # LAM encoder returns action distributions for all T-1 transitions
                                    mu, logvar = self.lam_model.encoder(z_batch)  # [1, T-1, action_dim]

                                    # Extract each action latent (one per transition)
                                    for action_step in range(mu.shape[1]):  # T-1 actions
                                        action_latent = mu[0, action_step, :]  # [action_dim]

                                        # Load corresponding 4 real frames for this action
                                        # Extract session_id from filename (remove .h5 extension)
                                        session_id = filename.replace('.h5', '')
                                        try:
                                            images = self._load_corresponding_images_from_images_h5(session_id, traj_idx, T, action_step)
                                        except Exception as e:
                                            images = [Image.new('RGB', (224, 224), (0, 0, 0)) for _ in range(4)]

                                        # Store results
                                        action_latents.append(action_latent.cpu())
                                        metadata.append({
                                            'filename': filename,
                                            'traj_idx': traj_idx,
                                            'action_step': action_step,
                                            'images': images
                                        })

                                except Exception as e:
                                    import traceback
                                    print(f"Error processing trajectory {traj_idx}: {e}")
                                    traceback.print_exc()
                                    continue

                        except Exception as e:
                            print(f"Error processing trajectory {traj_idx} in {filename}: {e}")
                            continue

            except Exception as e:
                import traceback
                print(f"Error processing file {filename}: {e}")
                print("Full traceback:")
                traceback.print_exc()
                continue

        print(f"Extracted {len(action_latents)} action latents")
        return action_latents, metadata

    def create_tsne_visualization(
        self,
        n_trajectories: int,
        split_name: str = "train",
        perplexity: float = 30.0,
        max_iter: int = 1000,
        random_state: int = 42
    ):
        """Create t-SNE visualization and log to wandb"""

        # Initialize wandb
        wandb.init(
            project="lam-tsne-visualization",
            name=f"tsne_{n_trajectories}traj_{split_name}",
            config={
                "n_trajectories": n_trajectories,
                "split_name": split_name,
                "perplexity": perplexity,
                "max_iter": max_iter,
                "random_state": random_state
            }
        )

        # Extract action latents
        action_latents, metadata = self.extract_action_latents(n_trajectories, split_name)

        if len(action_latents) == 0:
            print("No action latents extracted. Exiting.")
            return

        # Convert to numpy array
        latent_matrix = torch.stack(action_latents).numpy()  # [n_samples, latent_dim]
        print(f"Action latents shape: {latent_matrix.shape}")

        # Apply t-SNE
        print("Applying t-SNE...")
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state,
            verbose=1
        )

        tsne_coords = tsne.fit_transform(latent_matrix)  # [n_samples, 2]
        print(f"t-SNE completed. Coordinates shape: {tsne_coords.shape}")

        # Create wandb table
        table_data = []

        for i, (coords, meta) in enumerate(zip(tsne_coords, metadata)):
            # Create a combined image from 4 frames
            combined_image = self._create_combined_image(meta['images'])

            table_data.append([
                coords[0],  # x coordinate
                coords[1],  # y coordinate
                meta['filename'],
                meta['traj_idx'],
                meta['action_step'],
                wandb.Image(combined_image, caption=f"{meta['filename']}_{meta['traj_idx']}_{meta['action_step']}")
            ])

        # Create wandb table
        table = wandb.Table(
            columns=[
                "tsne_x", "tsne_y", "filename", "traj_idx", "action_step",
                "combined_frames"
            ],
            data=table_data
        )

        # Log table
        wandb.log({
            "action_latent_tsne": table,
            "n_samples": len(action_latents),
            "latent_dim": latent_matrix.shape[1]
        })

        print(f"Logged {len(table_data)} samples to wandb")
        wandb.finish()

    def _create_combined_image(self, images: List[Image.Image]) -> Image.Image:
        """Combine 4 images into a 2x2 grid"""
        if len(images) != 4:
            images = images[:4] + [Image.new('RGB', (224, 224), (0, 0, 0))] * (4 - len(images))

        # Resize images to consistent size
        size = (112, 112)
        resized_images = [img.resize(size) for img in images]

        # Create 2x2 grid
        combined = Image.new('RGB', (size[0] * 2, size[1] * 2))
        combined.paste(resized_images[0], (0, 0))
        combined.paste(resized_images[1], (size[0], 0))
        combined.paste(resized_images[2], (0, size[1]))
        combined.paste(resized_images[3], (size[0], size[1]))

        return combined


def main():
    parser = argparse.ArgumentParser(description="t-SNE Visualization for LAM Action Latents")
    parser.add_argument("--model_path", required=True, help="Path to LAM model checkpoint")
    parser.add_argument("--config_path", required=True, help="Path to model config")
    parser.add_argument("--processed_data_dir", required=True, help="Path to processed data directory")
    parser.add_argument("--manifest_path", required=True, help="Path to manifest JSON file")
    parser.add_argument("--image_dir", required=True, help="Path to image directory")
    parser.add_argument("--n_trajectories", type=int, default=50, help="Number of trajectories to process")
    parser.add_argument("--split_name", default="train", help="Data split to use")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--max_iter", type=int, default=1000, help="t-SNE max iterations")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--device", default="cuda", help="Device to use")

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    # Create visualizer
    visualizer = LAMTSNEVisualizer(
        model_path=args.model_path,
        config_path=args.config_path,
        processed_data_dir=args.processed_data_dir,
        manifest_path=args.manifest_path,
        image_dir=args.image_dir,
        device=args.device
    )

    # Create visualization
    visualizer.create_tsne_visualization(
        n_trajectories=args.n_trajectories,
        split_name=args.split_name,
        perplexity=args.perplexity,
        max_iter=args.max_iter,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()