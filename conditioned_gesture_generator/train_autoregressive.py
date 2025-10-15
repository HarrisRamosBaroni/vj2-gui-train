import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter threading issues
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from latent_action_model.vae import LatentActionVAE, load_lam_config
from training.dataloader import init_preprocessed_data_loader
from src.utils.logging import get_logger


class GestureTokenizer:
    """Handles quantization of continuous gesture values to discrete tokens for factorized mode."""

    def __init__(self, n_classes: int = 3000, value_range: tuple = (0.0, 1.0),
                 tokenization_mode: str = "factorized"):
        self.n_classes = n_classes
        self.min_val, self.max_val = value_range
        self.tokenization_mode = "factorized"  # Only factorized mode supported

        # Factorized: separate tokens for x, y, touch
        self.x_bins = n_classes  # 3000 bins for x
        self.y_bins = n_classes  # 3000 bins for y
        self.touch_bins = 3  # 3 bins for touch: 0=BOS, 1=no touch, 2=touch

    def quantize(self, values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert continuous values to factorized token IDs.

        Args:
            values: Continuous values, shape [..., 3] (x, y, touch)
        Returns:
            Dict with keys 'x', 'y', 'touch', each shape [...]
        """
        # Clamp values to range
        values = torch.clamp(values, self.min_val, self.max_val)

        # Normalize to [0, 1]
        normalized = (values - self.min_val) / (self.max_val - self.min_val)

        # Factorized quantization - return separate tokens
        x_coords = normalized[..., 0]  # [...]
        y_coords = normalized[..., 1]  # [...]
        touch_vals = normalized[..., 2]  # [...]

        # Quantize each component separately
        # Reserve token ID 0 for BOS, so quantized tokens start from 1
        x_tokens = (x_coords * (self.x_bins - 2)).long() + 1  # Range [1, x_bins-1]
        y_tokens = (y_coords * (self.y_bins - 2)).long() + 1  # Range [1, y_bins-1]
        touch_tokens = (touch_vals > 0.5).long() + 1  # Range [1, 2] (1=no touch, 2=touch)

        # Clamp to valid ranges (BOS=0 reserved)
        x_tokens = torch.clamp(x_tokens, 1, self.x_bins - 1)
        y_tokens = torch.clamp(y_tokens, 1, self.y_bins - 1)
        touch_tokens = torch.clamp(touch_tokens, 1, 2)  # 1=no touch, 2=touch

        return {
            'x': x_tokens,
            'y': y_tokens,
            'touch': touch_tokens
        }

    def dequantize(self, token_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert factorized token IDs back to continuous values.

        Args:
            token_data: Dict with keys 'x', 'y', 'touch'
        Returns:
            values: Continuous values, shape [..., 3]
        """
        # Factorized dequantization - input is dict
        x_tokens = token_data['x']
        y_tokens = token_data['y']
        touch_tokens = token_data['touch']

        # Convert back to [0, 1] (shift back from [1, n_bins-1] to [0, n_bins-2])
        x_vals = (x_tokens.float() - 1) / (self.x_bins - 2)
        y_vals = (y_tokens.float() - 1) / (self.y_bins - 2)
        touch_vals = (touch_tokens.float() - 1)  # [1,2] -> [0,1]

        # Scale back to original range
        x_vals = x_vals * (self.max_val - self.min_val) + self.min_val
        y_vals = y_vals * (self.max_val - self.min_val) + self.min_val

        # Clamp all values to ensure they are within the valid range
        x_vals = torch.clamp(x_vals, 0.0, 1.0)
        y_vals = torch.clamp(y_vals, 0.0, 1.0)
        touch_vals = torch.clamp(touch_vals, 0.0, 1.0)

        values = torch.stack([x_vals, y_vals, touch_vals], dim=-1)
        return values





def visualize_action_reconstruction(original_actions, reconstructed_actions, title_prefix="", max_samples=6):
    """
    Create visualization comparing original vs reconstructed actions.

    Args:
        original_actions: Ground truth actions [B, T, Traj, 3] or [B, T*Traj, 3]
        reconstructed_actions: Reconstructed actions [B, T, Traj, 3] or [B, T*Traj, 3]
        title_prefix: Prefix for plot titles
        max_samples: Maximum number of action sequences to plot

    Returns:
        matplotlib figure
    """
    # Convert to numpy
    if isinstance(original_actions, torch.Tensor):
        original_actions = original_actions.detach().cpu().numpy()
    if isinstance(reconstructed_actions, torch.Tensor):
        reconstructed_actions = reconstructed_actions.detach().cpu().numpy()

    # Handle different input shapes
    if len(original_actions.shape) == 4:  # [B, T, Traj, 3]
        # Reshape to [B, T*Traj, 3] for plotting
        B, T, Traj, _ = original_actions.shape
        original_actions = original_actions.reshape(B, T * Traj, 3)
        reconstructed_actions = reconstructed_actions.reshape(B, T * Traj, 3)

    batch_size = min(max_samples, original_actions.shape[0])

    # Create subplots: 2 rows (original, reconstructed), batch_size columns
    fig, axes = plt.subplots(2, batch_size, figsize=(3 * batch_size, 6))
    if batch_size == 1:
        axes = axes.reshape(-1, 1)

    for i in range(batch_size):
        # Original actions (top row)
        ax_orig = axes[0, i]
        orig_data = original_actions[i]
        timesteps = np.arange(len(orig_data))

        ax_orig.plot(timesteps, orig_data[:, 0], 'b-', alpha=0.8, label='X', linewidth=1.5)
        ax_orig.plot(timesteps, orig_data[:, 1], 'g-', alpha=0.8, label='Y', linewidth=1.5)
        ax_orig.plot(timesteps, orig_data[:, 2], 'r-', alpha=0.8, label='Touch', linewidth=1.5)

        # Highlight touch regions
        touch_mask = orig_data[:, 2] > 0.5
        if np.any(touch_mask):
            ax_orig.fill_between(timesteps, 0, 1, where=touch_mask, alpha=0.2, color='red')

        ax_orig.set_title(f"{title_prefix} Original {i+1}")
        ax_orig.set_ylim(-0.1, 1.1)
        ax_orig.grid(True, alpha=0.3)
        if i == 0:
            ax_orig.legend(fontsize='small')
            ax_orig.set_ylabel("Value")

        # Reconstructed actions (bottom row)
        ax_recon = axes[1, i]
        recon_data = reconstructed_actions[i]

        ax_recon.plot(timesteps, recon_data[:, 0], 'b-', alpha=0.8, label='X', linewidth=1.5)
        ax_recon.plot(timesteps, recon_data[:, 1], 'g-', alpha=0.8, label='Y', linewidth=1.5)
        ax_recon.plot(timesteps, recon_data[:, 2], 'r-', alpha=0.8, label='Touch', linewidth=1.5)

        # Highlight touch regions
        touch_mask = recon_data[:, 2] > 0.5
        if np.any(touch_mask):
            ax_recon.fill_between(timesteps, 0, 1, where=touch_mask, alpha=0.2, color='red')

        ax_recon.set_title(f"{title_prefix} Reconstructed {i+1}")
        ax_recon.set_ylim(-0.1, 1.1)
        ax_recon.grid(True, alpha=0.3)
        ax_recon.set_xlabel("Timestep")
        if i == 0:
            ax_recon.legend(fontsize='small')
            ax_recon.set_ylabel("Value")

    plt.tight_layout()
    return fig


def visualize_generation_only(generated_actions, title_prefix="", max_samples=5):
    """
    Create visualization of generated action sequences only (no ground truth comparison).

    Args:
        generated_actions: Generated actions [B, L, 3]
        title_prefix: Prefix for plot titles
        max_samples: Maximum number of sequences to plot

    Returns:
        matplotlib figure
    """
    # Convert to numpy
    if isinstance(generated_actions, torch.Tensor):
        generated_actions = generated_actions.detach().cpu().numpy()

    batch_size = min(max_samples, generated_actions.shape[0])

    # Create subplots: 1 row, batch_size columns
    fig, axes = plt.subplots(1, batch_size, figsize=(4 * batch_size, 4))
    if batch_size == 1:
        axes = [axes]

    for i in range(batch_size):
        ax = axes[i]
        gen_data = generated_actions[i]
        timesteps = np.arange(len(gen_data))

        ax.plot(timesteps, gen_data[:, 0], 'b-', alpha=0.8, label='X', linewidth=1.5)
        ax.plot(timesteps, gen_data[:, 1], 'g-', alpha=0.8, label='Y', linewidth=1.5)
        ax.plot(timesteps, gen_data[:, 2], 'r-', alpha=0.8, label='Touch', linewidth=1.5)

        # Highlight touch regions
        touch_mask = gen_data[:, 2] > 0.5
        if np.any(touch_mask):
            ax.fill_between(timesteps, 0, 1, where=touch_mask, alpha=0.2, color='red')

        ax.set_title(f"{title_prefix} Sample {i+1}")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Timestep")
        if i == 0:
            ax.legend(fontsize='small')
            ax.set_ylabel("Value")

    plt.tight_layout()
    return fig


class AutoregressiveTrainer:
    """Trainer for autoregressive gesture generation model."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.logger = get_logger()

        # Set seeds for reproducibility
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Initialize tokenizer
        self.tokenizer = GestureTokenizer(
            n_classes=args.n_classes,
            tokenization_mode="factorized"
        )

        # --- Load LAM model for action conditioning ---
        self.lam_model = None
        if hasattr(args, 'lam_ckpt_path') and args.lam_ckpt_path:
            self.logger.info(f"Loading LAM model from {args.lam_ckpt_path}")
            lam_config = load_lam_config(args.lam_config)
            self.lam_model = LatentActionVAE(**lam_config)
            lam_checkpoint = torch.load(args.lam_ckpt_path, map_location=self.device)
            self.lam_model.load_state_dict(lam_checkpoint['model_state_dict'])
            self.lam_model.to(self.device)
            self.lam_model.eval()
            d_action = lam_config["action_dim"]
            self.logger.info(f"LAM model loaded successfully. d_action={d_action}")
        else:
            d_action = 128  # Default action dimension
            self.logger.info("No LAM model specified. Using dummy action sequences.")

        # --- Initialize conditional GPT-style factorized model ---
        from conditioned_gesture_generator.autoregressive_gesture_decoder import FactorizedAutoregressiveGestureDecoder
        model_args = {
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.nl_dec,
            "x_classes": args.n_classes,
            "y_classes": args.n_classes,
            "touch_classes": 3,  # 0=BOS, 1=no touch, 2=touch
            "tokenization_mode": "factorized",
            "max_seq_len": 2048,
            "d_action": d_action,  # Action latent dimension
            "dropout": args.dropout,  # Dropout for regularization
        }
        self.model = FactorizedAutoregressiveGestureDecoder(**model_args).to(self.device)
        print("Pure GPT-style factorized model instantiated successfully.")

        # --- Loss and Optimizer ---
        from conditioned_gesture_generator.autoregressive_loss import FactorizedAutoregressiveLoss
        self.loss_fn = FactorizedAutoregressiveLoss(
            delta_loss_weight=args.delta_loss_weight
        ).to(self.device)

        # --- Scheduled Sampling ---
        self.scheduled_sampling = args.scheduled_sampling
        self.sampling_decay_steps = args.sampling_decay_steps if hasattr(args, 'sampling_decay_steps') else args.num_epochs
        self.max_sampling_prob = args.max_sampling_prob
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.0
        )

        # Learning rate scheduler (per-step, not per-epoch)
        # Will set total_steps after dataloader is initialized
        self.scheduler = None
        self.total_steps = None

        print(f"Loss function {self.loss_fn.__class__.__name__} instantiated successfully.")

        # --- Dataloaders ---
        self.train_loader, _ = init_preprocessed_data_loader(
            processed_data_dir=args.processed_data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            manifest_path=args.manifest,
            split_name='train',
        )
        self.val_loader, _ = init_preprocessed_data_loader(
            processed_data_dir=args.processed_data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            manifest_path=args.manifest,
            split_name='validation',
        )

        # Initialize scheduler now that we know the dataloader size
        self.total_steps = len(self.train_loader) * args.num_epochs
        # Use constant learning rate (no scheduling)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)

        # --- Checkpointing and Logging ---
        self.run_dir = Path(args.output_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}_autoregressive"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.global_step = 0

        if args.wandb_project:
            run_name = args.wandb_run_name if args.wandb_run_name else self.run_dir.name
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
            wandb.watch(self.model, log="all", log_freq=100)
            self.logger.info(f"W&B initialized for run: {run_name}")

        # --- Resume from checkpoint if specified ---
        self.start_epoch = 0
        if args.resume_from:
            self.logger.info(f"Resuming training from checkpoint: {args.resume_from}")
            self._load_checkpoint(args.resume_from)

        # --- Overfitting Test Setup ---
        self.overfit_test = args.overfit_test
        self.fixed_batch = None

        if self.overfit_test:
            self.logger.info("=" * 60)
            self.logger.info("OVERFITTING TEST MODE ACTIVATED")
            self.logger.info(f"  - Using single batch for {args.overfit_epochs} epochs")
            self.logger.info("  - Validation disabled")
            self.logger.info("  - Detailed loss tracking enabled")
            self.logger.info("  - Delta loss disabled")
            self.logger.info("=" * 60)


            # Lower learning rate for stability (weight decay already 0.0, scheduler already constant)
            for g in self.optimizer.param_groups:
                g['lr'] = 3e-4  # safer for single-sample overfit

            # No regularizers to disable - FactorizedAutoregressiveLoss is already clean
            # Dropout already disabled by default

            # Get and store a single batch - use only first sample for true overfit test
            self.fixed_batch = next(iter(self.train_loader))
            vb, ga = self.fixed_batch
            self.fixed_batch = (vb[:1], ga[:1])  # keep only first sample
            action_seq, gesture_tokens = self.prepare_batch(self.fixed_batch)

            self.logger.info(f"Fixed batch info:")
            self.logger.info(f"  Action sequence shape: {action_seq.shape}")
            if isinstance(gesture_tokens, dict):
                self.logger.info(f"  Gesture tokens (factorized):")
                for key, tokens in gesture_tokens.items():
                    self.logger.info(f"    {key} shape: {tokens.shape}")
                    self.logger.info(f"    {key} unique tokens: {torch.unique(tokens).numel()}")
            else:
                self.logger.info(f"  Gesture tokens shape: {gesture_tokens.shape}")
                self.logger.info(f"  Sequence length: {gesture_tokens.shape[1]}")
                self.logger.info(f"  Unique tokens in batch: {torch.unique(gesture_tokens).numel()}")


    def prepare_batch(self, sample):
        """Prepare a training batch with action conditioning."""
        visual_embeddings, ground_truth_actions = sample
        visual_embeddings = visual_embeddings.to(self.device, non_blocking=True)  # [B, T, ...]
        ground_truth_actions = ground_truth_actions.to(self.device, non_blocking=True)  # [B, T-1, Traj, A]

        B, T_minus_1, Traj, A = ground_truth_actions.shape

        with torch.no_grad():
            # Extract action latents using LAM model
            if self.lam_model is not None:
                # Apply layer normalization to input sequences (same as LAM training)
                # This matches the preprocessing used by VJ2 predictor models and LAM training
                visual_embeddings_normalized = F.layer_norm(visual_embeddings, (visual_embeddings.size(-1),))

                # Use LAM to encode visual embeddings into action latents
                mu, logvar = self.lam_model.encode(visual_embeddings_normalized)  # Returns (mu, logvar) tuple
                # Use reparameterized samples: z = mu + eps * exp(0.5 * logvar)
                eps = torch.randn_like(mu)
                action_sequence = mu + eps * torch.exp(0.5 * logvar)  # [B, T-1, d_action]
            else:
                # Use dummy action sequences if no LAM model
                action_sequence = torch.randn(B, T_minus_1, 128, device=self.device)

            # Flatten all ground truth actions for tokenization
            gt_actions_reshaped = ground_truth_actions.reshape(B * T_minus_1, Traj, 3)  # [B*T_minus_1, Traj, 3]

            # Tokenize all at once
            tokens_flat = self.tokenizer.quantize(gt_actions_reshaped)

            # tokens_flat is a dict with keys 'x', 'y', 'touch', each [B*T_minus_1, Traj]
            # Reshape back to [B, T_minus_1, Traj] then flatten time/traj dimensions
            gesture_tokens = {}
            for key in ['x', 'y', 'touch']:
                tokens_reshaped = tokens_flat[key].reshape(B, T_minus_1, Traj)  # [B, T_minus_1, Traj]
                gesture_tokens[key] = tokens_reshaped.reshape(B, T_minus_1 * Traj)  # [B, T_minus_1 * Traj]

        return action_sequence, gesture_tokens

    def train_step(self, sample):
        """Single training step with scheduled sampling."""
        action_sequence, gesture_tokens = self.prepare_batch(sample)

        # Sanity check: assert targets never contain 0 after BOS
        if not hasattr(self, '_sanity_check_done'):
            for k in ['x', 'y', 'touch']:
                tgt = gesture_tokens[k]
                assert (tgt[:, 1:] != 0).all(), f"{k} has BOS id at t>0"
            self._sanity_check_done = True
            self.logger.info("✓ Sanity check passed: no BOS tokens in targets after t=0")

        self.optimizer.zero_grad()

        # Calculate scheduled sampling probability (linear decay to max_sampling_prob)
        if self.scheduled_sampling:
            current_epoch = self.global_step // len(self.train_loader)
            progress = min(1.0, current_epoch / self.sampling_decay_steps)
            sampling_prob = progress * self.max_sampling_prob
        else:
            sampling_prob = 0.0

        # Always apply scheduled sampling (per-position, not batch-level gating)
        if self.scheduled_sampling and sampling_prob > 0.0:
            # First forward pass to get logits for sampling
            with torch.no_grad():
                initial_output = self.model(action_sequence, gesture_tokens)

            # Apply per-position scheduled sampling with temperature
            mixed_tokens = self._apply_scheduled_sampling(
                gesture_tokens,
                initial_output,
                sampling_prob,
                temperature=getattr(self.args, 'sampling_temperature', 1.0)
            )

            # Second forward pass with mixed tokens for gradient computation
            model_output = self.model(action_sequence, mixed_tokens)
        else:
            # Standard teacher forcing
            model_output = self.model(action_sequence, gesture_tokens)

        # Compute loss
        loss_components = self.loss_fn(
            model_output,
            action_sequence=action_sequence,
            model=self.model
        )
        loss = loss_components["total_loss"]


        # Backward pass
        loss.backward()

        # Gradient clipping (skip during overfit test to allow large gradients)
        if not self.overfit_test:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

        self.optimizer.step()
        # No scheduler stepping needed (constant LR)

        # Log to wandb
        if self.args.wandb_project:
            log_dict = {f"train/{k}": v.item() if torch.is_tensor(v) else v
                       for k, v in loss_components.items()}
            log_dict["train/learning_rate"] = self.optimizer.param_groups[0]['lr']
            log_dict["train/sampling_probability"] = sampling_prob
            log_dict["global_step"] = self.global_step

            # Log FiLM scale/shift norms and magnitude distributions (only if FiLM was used in this forward pass)
            try:
                if (hasattr(self.model, 'film') and
                    hasattr(self.model.film, 'last_scale_norm') and
                    self.model.film.last_scale_norm is not None):

                    # Log norms (L2 norm per position)
                    scale_norms = self.model.film.last_scale_norm.flatten().cpu().numpy()
                    shift_norms = self.model.film.last_shift_norm.flatten().cpu().numpy()
                    log_dict["train/film_scale_norms"] = wandb.Histogram(scale_norms)
                    log_dict["train/film_shift_norms"] = wandb.Histogram(shift_norms)

                    # Log raw magnitude distributions
                    if (hasattr(self.model.film, 'last_scale_values') and
                        self.model.film.last_scale_values is not None):
                        scale_magnitudes = self.model.film.last_scale_values.flatten().cpu().numpy()
                        shift_magnitudes = self.model.film.last_shift_values.flatten().cpu().numpy()
                        log_dict["train/film_scale_magnitudes"] = wandb.Histogram(scale_magnitudes)
                        log_dict["train/film_shift_magnitudes"] = wandb.Histogram(shift_magnitudes)

            except Exception as e:
                # Silently skip if FiLM values aren't available
                pass

            wandb.log(log_dict)

        return loss_components

    def _apply_scheduled_sampling(self, ground_truth_tokens, initial_output, sampling_prob, temperature=1.0):
        """Apply per-position scheduled sampling to create mixed token sequences.

        Args:
            ground_truth_tokens: Dict with 'x', 'y', 'touch' tokens [B, L]
            initial_output: Model output with logits
            sampling_prob: Probability of sampling at each position t>0
            temperature: Temperature for sampling (default 1.0)

        Returns:
            mixed_tokens: Dict with mixed ground truth and sampled tokens
        """
        B, L = ground_truth_tokens['x'].shape
        device = ground_truth_tokens['x'].device

        # Extract logits from initial output
        x_logits = initial_output['x_logits']  # [B, L, x_classes]
        y_logits = initial_output['y_logits']  # [B, L, y_classes]
        touch_logits = initial_output['touch_logits']  # [B, L, touch_classes]

        # Create mixed token sequences
        mixed_tokens = {}

        # Always apply per-position sampling (not batch-level gating)
        for token_type, logits in [('x', x_logits), ('y', y_logits), ('touch', touch_logits)]:
            gt_tokens = ground_truth_tokens[token_type]  # [B, L]

            # Per-position Bernoulli mask for every position t>0
            sampling_mask = torch.zeros_like(gt_tokens, dtype=torch.bool)
            if L > 1:  # Only if we have positions beyond BOS
                sampling_mask[:, 1:] = torch.rand(B, L-1, device=device) < sampling_prob

            # Forbid BOS at t>0 to prevent context poisoning
            logits_masked = logits.clone()
            if L > 1:
                logits_masked[:, 1:, 0] = -float('inf')

            # Temperature-scaled sampling
            if temperature != 1.0:
                logits_masked = logits_masked / temperature

            # Sample from temperature-scaled logits
            probs = torch.softmax(logits_masked, dim=-1)
            sampled_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, L)

            # Mix ground truth and sampled tokens per position
            mixed_tokens[token_type] = torch.where(sampling_mask, sampled_tokens, gt_tokens)

        return mixed_tokens

    def validate(self):
        """Validation loop."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        all_loss_components = {}
        saved_action_sequences = []  # Store LAM latents for rollout generation

        with torch.no_grad():
            for batch_idx, sample in enumerate(self.val_loader):
                action_sequence, gesture_tokens = self.prepare_batch(sample)

                # Save action sequences for rollout generation (first few batches only)
                if len(saved_action_sequences) < 5:  # Collect enough for rollout
                    saved_action_sequences.append(action_sequence.detach())

                # Forward pass
                model_output = self.model(action_sequence, gesture_tokens)

                # Compute loss
                loss_components = self.loss_fn(
                    model_output,
                    action_sequence=action_sequence,
                    model=self.model
                )

                # Action sensitivity probe: run with random actions
                if batch_idx < 3:  # Only for first few batches to avoid overhead
                    random_action_sequence = torch.randn_like(action_sequence)
                    random_model_output = self.model(random_action_sequence, gesture_tokens)
                    random_loss_components = self.loss_fn(
                        random_model_output,
                        action_sequence=random_action_sequence,
                        model=self.model
                    )

                    # Store random action losses for sensitivity analysis
                    if 'x_ce_random' not in all_loss_components:
                        all_loss_components['x_ce_random'] = []
                        all_loss_components['y_ce_random'] = []
                        all_loss_components['touch_ce_random'] = []

                    all_loss_components['x_ce_random'].append(random_loss_components['loss_x'].item())
                    all_loss_components['y_ce_random'].append(random_loss_components['loss_y'].item())
                    all_loss_components['touch_ce_random'].append(random_loss_components['loss_touch'].item())


                total_loss += loss_components["total_loss"].item()
                num_batches += 1

                # Accumulate all loss components
                for k, v in loss_components.items():
                    if k not in all_loss_components:
                        all_loss_components[k] = []
                    all_loss_components[k].append(v.item() if torch.is_tensor(v) else v)

        # Average all components
        avg_loss_components = {k: np.mean(v) for k, v in all_loss_components.items()}
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # Log validation metrics
        if self.args.wandb_project:
            log_dict = {f"val/{k}": v for k, v in avg_loss_components.items()}
            log_dict["global_step"] = self.global_step

            # Add class histogram monitoring at step=1 for teacher-forcing
            if len(saved_action_sequences) > 0:
                with torch.no_grad():
                    # Get a sample for step=1 analysis
                    sample_action = saved_action_sequences[0][:1]  # Take first sample
                    tf_output = self.model(sample_action, self._get_sample_tokens_for_step1())

                    # Log step=1 class histograms for teacher-forcing
                    if 'x_logits' in tf_output:
                        x_step1 = torch.argmax(tf_output['x_logits'][:, 1, :], dim=-1).cpu().numpy()
                        y_step1 = torch.argmax(tf_output['y_logits'][:, 1, :], dim=-1).cpu().numpy()
                        touch_step1 = torch.argmax(tf_output['touch_logits'][:, 1, :], dim=-1).cpu().numpy()

                        log_dict["val/step1_x_tf_hist"] = wandb.Histogram(x_step1)
                        log_dict["val/step1_y_tf_hist"] = wandb.Histogram(y_step1)
                        log_dict["val/step1_touch_tf_hist"] = wandb.Histogram(touch_step1)

            wandb.log(log_dict)

        # Store collected action sequences for rollout generation
        self.saved_action_sequences = saved_action_sequences
        self.logger.info(f"Saved {len(saved_action_sequences)} action sequences for rollout generation")

        # Create reconstruction visualizations every validation run
        if self.args.wandb_project:
            self.logger.info("Creating reconstruction visualizations...")

            # Training set visualization (6 samples)
            train_result = self.create_reconstruction_visualization(
                self.train_loader, max_samples=6, dataset_name="Train"
            )
            if train_result is not None:
                log_dict = {
                    "reconstructions/train_tf": wandb.Image(train_result["tf"]),
                    "reconstructions/train_sampling": wandb.Image(train_result["mixed"]),
                    "global_step": self.global_step
                }

                # Add heatmap visualizations if available
                if "tf_heatmaps" in train_result and train_result["tf_heatmaps"]:
                    for component, fig in train_result["tf_heatmaps"].items():
                        component_name = component.replace('_logits', '')
                        log_dict[f"heatmaps/train_tf_{component_name}"] = wandb.Image(fig)
                        plt.close(fig)

                if "mixed_heatmaps" in train_result and train_result["mixed_heatmaps"]:
                    for component, fig in train_result["mixed_heatmaps"].items():
                        component_name = component.replace('_logits', '')
                        log_dict[f"heatmaps/train_mixed_{component_name}"] = wandb.Image(fig)
                        plt.close(fig)

                wandb.log(log_dict)
                plt.close(train_result["tf"])
                plt.close(train_result["mixed"])

            # Validation set visualization (6 samples)
            val_result = self.create_reconstruction_visualization(
                self.val_loader, max_samples=6, dataset_name="Val"
            )
            if val_result is not None:
                log_dict = {
                    "reconstructions/val_tf": wandb.Image(val_result["tf"]),
                    "reconstructions/val_sampling": wandb.Image(val_result["mixed"]),
                    "global_step": self.global_step
                }

                # Add heatmap visualizations if available
                if "tf_heatmaps" in val_result and val_result["tf_heatmaps"]:
                    for component, fig in val_result["tf_heatmaps"].items():
                        component_name = component.replace('_logits', '')
                        log_dict[f"heatmaps/val_tf_{component_name}"] = wandb.Image(fig)
                        plt.close(fig)

                if "mixed_heatmaps" in val_result and val_result["mixed_heatmaps"]:
                    for component, fig in val_result["mixed_heatmaps"].items():
                        component_name = component.replace('_logits', '')
                        log_dict[f"heatmaps/val_mixed_{component_name}"] = wandb.Image(fig)
                        plt.close(fig)

                wandb.log(log_dict)
                plt.close(val_result["tf"])
                plt.close(val_result["mixed"])

            # Generate 5 rollout samples from random LAM latents
            self.logger.info("Generating rollout samples from random LAM latents...")
            rollout_fig = self.create_rollout_visualization(num_samples=5, sequence_length=250)
            if rollout_fig is not None:
                wandb.log({
                    "reconstructions/rollout_generation": wandb.Image(rollout_fig),
                    "global_step": self.global_step
                })
                plt.close(rollout_fig)

            self.logger.info("Reconstruction visualizations completed.")

        self.model.train()
        return avg_loss, avg_loss_components

    def generate_samples(self, num_samples: int = 4):
        """Generate sample sequences for visualization."""
        self.model.eval()

        with torch.no_grad():
            # Get a validation batch
            sample = next(iter(self.val_loader))
            action_sequence, gesture_tokens = self.prepare_batch(sample)

            # Take only first few samples
            action_sequence = action_sequence[:num_samples]

            # Generate sequences conditioned on actions
            max_length = gesture_tokens['x'].shape[1]  # Use same length as training data
            generated_tokens = self.model.generate(
                action_sequence=action_sequence,
                max_length=max_length,
                temperature=getattr(self.args, 'sample_temperature', 1.0)
            )

            # Convert back to trajectories for visualization
            generated_trajectories = []
            for i in range(num_samples):
                # Extract tokens for sample i
                tokens = {
                    'x': generated_tokens['x'][i],      # [L]
                    'y': generated_tokens['y'][i],      # [L]
                    'touch': generated_tokens['touch'][i]  # [L]
                }
                trajectory = self.tokenizer.dequantize(tokens)  # [L, 3]
                generated_trajectories.append(trajectory)

            # Log to wandb as images/plots if desired
            # This would require additional visualization code

        self.model.train()
        return generated_trajectories

    def create_reconstruction_visualization(self, data_loader, max_samples=6, dataset_name=""):
        """Create reconstruction visualization comparing ground truth vs model output.
        Shows both pure teacher-forcing and mixed-context (scheduled sampling) reconstructions.
        """
        self.model.eval()

        tf_reconstructions = []  # Teacher-forcing reconstructions
        mixed_reconstructions = []  # Mixed-context reconstructions
        ground_truths = []
        tf_logits_list = []  # Store logits for heatmap visualization
        mixed_logits_list = []

        # Use current training sampling probability for mixed context
        current_epoch = self.global_step // len(self.train_loader) if hasattr(self, 'train_loader') else 0
        if self.scheduled_sampling and hasattr(self, 'sampling_decay_steps'):
            progress = min(1.0, current_epoch / self.sampling_decay_steps)
            sampling_prob = progress * self.max_sampling_prob
        else:
            sampling_prob = 0.0

        with torch.no_grad():
            for batch_idx, sample in enumerate(data_loader):
                if len(tf_reconstructions) >= max_samples:
                    break

                # Prepare batch
                action_sequence, gesture_tokens = self.prepare_batch(sample)

                # Get pure teacher-forcing predictions
                tf_output = self.model(action_sequence, gesture_tokens)

                # Get mixed-context predictions (if scheduled sampling is enabled)
                if sampling_prob > 0.0:
                    # First pass to get logits for sampling
                    initial_output = self.model(action_sequence, gesture_tokens)
                    # Apply scheduled sampling to create mixed tokens with same temperature as training
                    mixed_tokens = self._apply_scheduled_sampling(
                        gesture_tokens,
                        initial_output,
                        sampling_prob,
                        temperature=getattr(self.args, 'sampling_temperature', 1.0)
                    )
                    # Second pass with mixed tokens
                    mixed_output = self.model(action_sequence, mixed_tokens)
                else:
                    # If no scheduled sampling, mixed context is same as teacher forcing
                    mixed_output = tf_output

                # Process both teacher-forcing and mixed-context outputs
                def process_output(model_output, output_type="teacher-forcing"):
                    if isinstance(model_output, dict):  # Factorized mode
                        # Get the logits for each component - handle different key names
                        if 'x' in model_output:
                            x_logits = model_output['x']
                            y_logits = model_output['y']
                            touch_logits = model_output['touch']
                        elif 'x_logits' in model_output:
                            x_logits = model_output['x_logits']
                            y_logits = model_output['y_logits']
                            touch_logits = model_output['touch_logits']
                        else:
                            # Skip visualization if keys don't match expected format
                            self.logger.warning(f"Unknown model output format with keys: {list(model_output.keys())}")
                            return None, None

                        # Store logits for heatmap visualization
                        logits_dict = {
                            'x_logits': x_logits,
                            'y_logits': y_logits,
                            'touch_logits': touch_logits
                        }

                        # Convert logits to predictions
                        x_pred = torch.argmax(x_logits, dim=-1)  # [B, L]
                        y_pred = torch.argmax(y_logits, dim=-1)  # [B, L]
                        touch_pred = torch.argmax(touch_logits, dim=-1)  # [B, L]

                        # Create token dict for dequantization
                        pred_tokens = {
                            'x': x_pred,
                            'y': y_pred,
                            'touch': touch_pred
                        }

                        # Dequantize predictions
                        pred_actions = self.tokenizer.dequantize(pred_tokens)  # [B, L, 3]
                        return pred_actions, logits_dict
                    else:
                        # Non-factorized mode - implement if needed
                        self.logger.warning(f"Visualization not implemented for non-factorized mode ({output_type})")
                        return None, None

                # Process teacher-forcing output
                tf_pred_actions, tf_logits = process_output(tf_output, "teacher-forcing")
                if tf_pred_actions is None:
                    break

                # Process mixed-context output
                mixed_pred_actions, mixed_logits = process_output(mixed_output, "mixed-context")
                if mixed_pred_actions is None:
                    break

                # Get ground truth actions from the original batch
                _, ground_truth_actions = sample
                ground_truth_actions = ground_truth_actions.to(self.device)

                # Reshape ground truth to match prediction format [B, L, 3]
                B, T_minus_1, Traj, A = ground_truth_actions.shape
                gt_actions = ground_truth_actions.reshape(B, T_minus_1 * Traj, A)

                # Store both types of reconstructions
                tf_reconstructions.append(tf_pred_actions.cpu())
                mixed_reconstructions.append(mixed_pred_actions.cpu())
                ground_truths.append(gt_actions.cpu())

                # Store logits for heatmap visualization (only first batch to keep manageable)
                if len(tf_logits_list) == 0:
                    tf_logits_list.append(tf_logits)
                    mixed_logits_list.append(mixed_logits)

        if tf_reconstructions:
            # Concatenate all samples
            all_tf_reconstructions = torch.cat(tf_reconstructions, dim=0)  # [N, L, 3]
            all_mixed_reconstructions = torch.cat(mixed_reconstructions, dim=0)  # [N, L, 3]
            all_ground_truths = torch.cat(ground_truths, dim=0)  # [N, L, 3]

            # Create separate teacher-forcing and mixed-context visualizations
            tf_fig = visualize_action_reconstruction(
                all_ground_truths,
                all_tf_reconstructions,
                title_prefix=f"{dataset_name} Teacher Forcing",
                max_samples=max_samples
            )

            mixed_fig = visualize_action_reconstruction(
                all_ground_truths,
                all_mixed_reconstructions,
                title_prefix=f"{dataset_name} Mixed Context (p={sampling_prob:.2f})",
                max_samples=max_samples
            )

            # Create heatmap visualizations if we have logits
            tf_heatmaps = {}
            mixed_heatmaps = {}
            if tf_logits_list:
                tf_heatmaps = self.create_logit_heatmap(
                    tf_logits_list[0],
                    title_prefix=f"{dataset_name} TF"
                )
                mixed_heatmaps = self.create_logit_heatmap(
                    mixed_logits_list[0],
                    title_prefix=f"{dataset_name} Mixed"
                )

            self.model.train()
            return {
                "tf": tf_fig,
                "mixed": mixed_fig,
                "tf_heatmaps": tf_heatmaps,
                "mixed_heatmaps": mixed_heatmaps
            }

        self.model.train()
        return None

    def create_logit_heatmap(self, logits_dict, title_prefix="", max_samples=1):
        """Create heatmap visualization of logit distributions over time.

        Args:
            logits_dict: Dict with 'x_logits', 'y_logits', 'touch_logits' [B, L, num_classes]
            title_prefix: Prefix for plot titles
            max_samples: Number of samples to visualize (takes first N)

        Returns:
            Dict of matplotlib figures for each component
        """
        import matplotlib.pyplot as plt

        figs = {}

        for component in ['x_logits', 'y_logits', 'touch_logits']:
            if component not in logits_dict:
                continue

            logits = logits_dict[component]  # [B, L, num_classes]

            # Take first sample and convert to probabilities
            sample_logits = logits[0]  # [L, num_classes]
            probs = F.softmax(sample_logits, dim=-1)  # [L, num_classes]

            # Create heatmap: [num_classes, L] for proper orientation
            # Y-axis: token classes (0-2999), X-axis: time steps
            prob_matrix = probs.T.cpu().numpy()  # [num_classes, L]

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))

            # Create heatmap - limit to show reasonable range
            # Show only tokens 1-500 to make it readable (exclude BOS=0, limit range)
            display_range = min(500, prob_matrix.shape[0])
            display_probs = prob_matrix[1:display_range+1, :]  # Skip BOS token

            im = ax.imshow(display_probs,
                          aspect='auto',
                          cmap='hot',
                          interpolation='nearest',
                          origin='lower')

            # Labels and title
            component_name = component.replace('_logits', '').capitalize()
            ax.set_title(f"{title_prefix} {component_name} Probability Distribution")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Token Class")

            # Add colorbar
            plt.colorbar(im, ax=ax, label='Probability')

            # Set ticks
            ax.set_yticks(range(0, display_range, 50))
            ax.set_yticklabels(range(1, display_range+1, 50))  # Start from 1 (skip BOS)

            plt.tight_layout()
            figs[component] = fig

        return figs

    def _get_sample_tokens_for_step1(self):
        """Create minimal token sequence for step=1 analysis."""
        device = self.device
        # Create BOS tokens followed by one dummy token
        return {
            'x': torch.tensor([[0, 1]], device=device),  # [1, 2] - BOS then dummy
            'y': torch.tensor([[0, 1]], device=device),
            'touch': torch.tensor([[0, 1]], device=device)
        }

    def create_rollout_visualization(self, num_samples: int = 5, sequence_length: int = 250):
        """Create visualization of pure autoregressive generation from random LAM latents."""
        self.logger.info(f"Creating rollout visualization with {num_samples} samples, length {sequence_length}")
        self.model.eval()

        if self.lam_model is None:
            self.logger.warning("No LAM model available for rollout generation")
            return None

        with torch.no_grad():
            # Use saved action sequences from validation instead of re-encoding
            if not hasattr(self, 'saved_action_sequences') or len(self.saved_action_sequences) == 0:
                self.logger.warning("No saved action sequences available for rollout generation")
                self.model.train()
                return None

            # Collect action sequences to create rollout samples
            all_action_sequences = []
            for action_seq in self.saved_action_sequences:
                if len(all_action_sequences) >= num_samples:
                    break
                # Take first timestep from each sequence for single-action rollout
                if action_seq.shape[1] > 0:  # Ensure we have valid timesteps
                    single_action = action_seq[:1, :1, :]  # [1, 1, action_dim]
                    all_action_sequences.append(single_action)

            if len(all_action_sequences) == 0:
                self.logger.error("No valid action sequences found for rollout generation")
                self.model.train()
                return None

            # Stack into a single batch for rollout generation
            posterior_samples = torch.cat(all_action_sequences, dim=0)  # [num_samples, 1, action_dim]
            self.logger.info(f"Using saved action sequences for rollout: {posterior_samples.shape}")

            # Generate all sequences in one batch
            try:
                generated_tokens = self.model.generate_full_rollout(
                    action_sequence=posterior_samples,
                    max_length=sequence_length,
                    use_argmax=False,  # Use multinomial sampling to enable temperature
                    temperature=getattr(self.args, 'generation_temperature', 1.0)
                )

                # Log token histograms for quick diagnosis
                rollout_log_dict = {"global_step": self.global_step}
                for name, tok in generated_tokens.items():
                    rollout_log_dict[f"rollout/{name}_hist"] = wandb.Histogram(tok.flatten().cpu().numpy())
                    # Log step=1 class histograms for rollout generation
                    if tok.shape[1] > 1:  # Ensure we have step=1
                        step1_tokens = tok[:, 0].cpu().numpy()  # Step=1 tokens (after BOS)
                        rollout_log_dict[f"rollout/step1_{name}_hist"] = wandb.Histogram(step1_tokens)

                wandb.log(rollout_log_dict)

                # Convert tokens to gesture trajectories
                all_generated = self.tokenizer.dequantize(generated_tokens)  # [num_samples, sequence_length, 3]

            except Exception as e:
                self.logger.error(f"Failed to generate rollout samples: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                self.model.train()
                return None

        if all_generated is not None:

            # Create visualization (no ground truth comparison)
            fig = visualize_generation_only(
                all_generated,
                title_prefix="Pure Autoregressive Generation",
                max_samples=num_samples
            )

            self.model.train()
            return fig

        self.model.train()
        return None

    def overfitting_test_loop(self):
        """Run overfitting test on a single batch."""
        self.logger.info("Starting overfitting test...")

        # Prepare the fixed batch once
        action_sequence, gesture_tokens = self.prepare_batch(self.fixed_batch)

        self.model.train()
        prev_loss = None

        for epoch in range(self.args.overfit_epochs):
            self.optimizer.zero_grad()

            # Forward pass on fixed batch
            model_output = self.model(action_sequence, gesture_tokens)

            # Compute loss
            loss_components = self.loss_fn(
                model_output,
                action_sequence=action_sequence,
                model=self.model
            )
            loss = loss_components["total_loss"]


            # Backward pass
            loss.backward()
            # Skip gradient clipping in overfit test to allow large gradients for memorization
            if not self.overfit_test:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            # No scheduler stepping needed (constant LR)

            # Log detailed progress
            current_loss = loss.item()

            if epoch % max(1, self.args.overfit_epochs // 20) == 0 or epoch < 10:
                log_str = " | ".join([f"{k}: {v.item():.6f}" if torch.is_tensor(v) else f"{k}: {v:.6f}"
                                     for k, v in loss_components.items()])
                self.logger.info(f"[Overfit Epoch {epoch+1}/{self.args.overfit_epochs}] {log_str}")

                # Check if loss is decreasing
                if prev_loss is not None:
                    loss_change = current_loss - prev_loss
                    if loss_change < -1e-6:
                        change_indicator = "✅ DECREASING"
                    elif loss_change > 1e-6:
                        change_indicator = "❌ INCREASING"
                    else:
                        change_indicator = "⚠️ STABLE"

                    self.logger.info(f"  Loss change: {loss_change:+.6f} ({change_indicator})")

            # Log to wandb
            if self.args.wandb_project:
                log_dict = {f"overfit/{k}": v.item() if torch.is_tensor(v) else v
                           for k, v in loss_components.items()}
                log_dict["overfit/epoch"] = epoch
                log_dict["overfit/loss_change"] = current_loss - prev_loss if prev_loss is not None else 0
                wandb.log(log_dict)

            prev_loss = current_loss

            # Generate samples periodically
            if epoch > 0 and epoch % max(1, self.args.overfit_epochs // 5) == 0:
                self.logger.info(f"Generating sample at epoch {epoch+1}...")
                try:
                    with torch.no_grad():
                        max_length = gesture_tokens['x'].shape[1]

                        generated_tokens = self.model.generate(
                            action_sequence=action_sequence[:1],  # Just first sample
                            max_length=max_length,
                            temperature=0.8
                        )

                        # Log some token statistics
                        unique_generated_x = torch.unique(generated_tokens['x']).numel()
                        unique_generated_y = torch.unique(generated_tokens['y']).numel()
                        unique_generated_touch = torch.unique(generated_tokens['touch']).numel()
                        unique_generated = unique_generated_x + unique_generated_y + unique_generated_touch

                        unique_target_x = torch.unique(gesture_tokens['x'][:1]).numel()
                        unique_target_y = torch.unique(gesture_tokens['y'][:1]).numel()
                        unique_target_touch = torch.unique(gesture_tokens['touch'][:1]).numel()
                        unique_target = unique_target_x + unique_target_y + unique_target_touch

                        self.logger.info(f"  Generated {unique_generated} unique tokens ({unique_generated_x}x, {unique_generated_y}y, {unique_generated_touch}t) vs {unique_target} in target")

                        if self.args.wandb_project:
                            wandb.log({
                                "overfit/unique_generated_tokens": unique_generated,
                                "overfit/unique_target_tokens": unique_target,
                                "overfit/token_diversity_ratio": unique_generated / max(unique_target, 1),
                                "overfit/epoch": epoch
                            })

                except Exception as e:
                    self.logger.warning(f"Sample generation failed: {e}")

        # Final evaluation
        final_loss = prev_loss
        self.logger.info("=" * 60)
        self.logger.info("OVERFITTING TEST COMPLETE")
        self.logger.info(f"Final loss: {final_loss:.6f}")

        # Use perplexity for more realistic success criteria
        final_perplexity = torch.exp(torch.tensor(final_loss))

        if final_perplexity < 2.0:
            self.logger.info(f"✅ SUCCESS: Model successfully overfitted (PPL={final_perplexity:.2f} < 2.0)")
        elif final_perplexity < 5.0:
            self.logger.info(f"⚠️ PARTIAL: Model partially overfitted (PPL={final_perplexity:.2f} < 5.0)")
        else:
            self.logger.info(f"❌ FAILED: Model did not overfit well (PPL={final_perplexity:.2f} >= 5.0)")

        self.logger.info("=" * 60)

        # Save overfitting checkpoint
        self._save_checkpoint("overfit_final", final_loss)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model state
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info("✅ Model state loaded")

            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.logger.info("✅ Optimizer state loaded")

            # Load scheduler state
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.logger.info("✅ Scheduler state loaded")

            # Load training state
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            self.global_step = checkpoint.get("global_step", 0)

            # Calculate start epoch from global step (approximate)
            # Assuming roughly same number of batches per epoch
            batches_per_epoch = len(self.train_loader)
            self.start_epoch = self.global_step // batches_per_epoch

            self.logger.info(f"✅ Training state loaded:")
            self.logger.info(f"  - Best val loss: {self.best_val_loss:.4f}")
            self.logger.info(f"  - Global step: {self.global_step}")
            self.logger.info(f"  - Starting from epoch: {self.start_epoch}")

        except Exception as e:
            self.logger.error(f"❌ Failed to load checkpoint: {e}")
            raise

    def _save_checkpoint(self, tag: str, val_loss: float = None):
        """Save model checkpoint."""
        checkpoint_path = self.run_dir / f"autoregressive_{tag}.pt"

        checkpoint = {
            "config": self.model.get_config(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "epoch": self.start_epoch,  # Save current epoch
            "tokenizer_config": {
                "n_classes": self.tokenizer.n_classes,
                "value_range": (self.tokenizer.min_val, self.tokenizer.max_val)
            }
        }
        if val_loss is not None:
            checkpoint["val_loss"] = val_loss

        try:
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"💾 Checkpoint saved to {checkpoint_path}")

            # Create latest symlink
            latest_symlink = self.run_dir / "latest.pt"
            if latest_symlink.exists():
                latest_symlink.unlink()
            latest_symlink.symlink_to(checkpoint_path.name)

        except Exception as e:
            self.logger.error(f"❌ Failed to save checkpoint: {e}")

    def training_loop(self):
        """Main training loop."""
        if self.overfit_test:
            self.overfitting_test_loop()
            return

        self.logger.info("Starting autoregressive training loop...")

        # Run initial evaluation and visualization before training
        self.logger.info("=== Running initial evaluation before training ===")
        initial_val_loss, initial_val_components = self.validate()
        self.logger.info(f"Initial validation loss: {initial_val_loss:.4f}")

        # Log initial metrics
        log_str = " | ".join([f"{k}: {v:.4f}" for k, v in initial_val_components.items()])
        self.logger.info(f"Initial validation metrics: {log_str}")

        for epoch in range(self.start_epoch, self.args.num_epochs):
            self.model.train()

            epoch_loss = 0
            num_batches = 0

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")

            total_steps_in_epoch = len(self.train_loader)
            validation_interval = max(1, total_steps_in_epoch // 2)  # Every 50% of epoch

            for i, sample in enumerate(progress_bar):
                self.global_step += 1

                loss_components = self.train_step(sample)

                epoch_loss += loss_components["total_loss"].item()
                num_batches += 1

                # Update progress bar
                if i % 10 == 0:
                    postfix = {k: f"{v.item():.4f}" if torch.is_tensor(v) else f"{v:.4f}"
                              for k, v in loss_components.items() if k in ["total_loss", "accuracy", "perplexity"]}
                    progress_bar.set_postfix(postfix)

                # Log training step
                if i % 20 == 0:
                    log_str = " | ".join([f"{k}: {v.item():.4f}" if torch.is_tensor(v) else f"{k}: {v:.4f}"
                                         for k, v in loss_components.items()])
                    self.logger.info(f"[Epoch {epoch+1}/{self.args.num_epochs} | Step {self.global_step}] {log_str}")

                # Run validation every 50% of epoch
                if (i + 1) % validation_interval == 0:
                    percent_complete = int(((i + 1) / total_steps_in_epoch) * 100)
                    self.logger.info(f"Running validation at {percent_complete}% through epoch {epoch+1}")

                    val_loss, val_components = self.validate()
                    self.logger.info(f"Validation at {percent_complete}%: Loss = {val_loss:.4f}")

                    # Save best model if this is the best validation loss so far
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.logger.info(f"🏆 New best validation loss: {val_loss:.4f}. Saving checkpoint.")
                        self._save_checkpoint("best", val_loss)

            # Scheduler is now stepped per training step, not per epoch

            # Log epoch summary
            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            self.logger.info(f"--- Epoch {epoch+1} Complete ---")
            self.logger.info(f"Average Train Loss: {avg_train_loss:.4f}")
            self.logger.info(f"Best Validation Loss: {self.best_val_loss:.4f}")

            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(f"epoch_{epoch+1}", self.best_val_loss)

            # Generate samples periodically
            if (epoch + 1) % self.args.sample_freq == 0:
                self.logger.info("Generating sample sequences...")
                try:
                    self.generate_samples()
                except Exception as e:
                    self.logger.warning(f"Sample generation failed: {e}")

        self.logger.info("Training completed!")


def load_autoregressive_model(checkpoint_path: str, lam_ckpt_path: str, lam_config_path: str, device: str = "cuda"):
    """
    Load a trained autoregressive model from checkpoint.

    Args:
        checkpoint_path: Path to the autoregressive model checkpoint (.pt file)
        lam_ckpt_path: Path to the LAM checkpoint
        lam_config_path: Path to the LAM config
        device: Device to load model on

    Returns:
        tuple: (model, tokenizer, lam_model)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['config']
    tokenizer_config = checkpoint['tokenizer_config']

    print(f"Loading autoregressive model from: {checkpoint_path}")
    print(f"Model config: {model_config}")
    print(f"Best val loss: {checkpoint['best_val_loss']:.4f}")
    print(f"Global step: {checkpoint['global_step']}")

    # Load LAM model (same as training)
    from latent_action_model.vae import LatentActionVAE, load_lam_config
    lam_checkpoint = torch.load(lam_ckpt_path, map_location=device)
    lam_config = load_lam_config(lam_config_path)
    lam_model = LatentActionVAE(**lam_config).to(device)
    lam_model.load_state_dict(lam_checkpoint['model_state_dict'])
    lam_model.eval()
    # Freeze LAM
    for param in lam_model.parameters():
        param.requires_grad = False
    print("LAM model loaded and frozen.")

    # Create tokenizer
    tokenizer = GestureTokenizer(
        n_classes=tokenizer_config['n_classes'],
        value_range=tokenizer_config['value_range'],
        tokenization_mode="factorized"
    )

    # Create autoregressive model
    from conditioned_gesture_generator.autoregressive_gesture_decoder import FactorizedAutoregressiveGestureDecoder
    model = FactorizedAutoregressiveGestureDecoder(**model_config).to(device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Autoregressive model loaded successfully!")
    print(f"Model type: {model.__class__.__name__}")

    return model, tokenizer, lam_model


def get_args():
    parser = argparse.ArgumentParser(description="Train Autoregressive Gesture Decoder")

    # Data args
    parser.add_argument("--processed_data_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/autoregressive_decoder")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume training from")

    # Model args
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--nl_dec", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--n_classes", type=int, default=3000)

    # LAM conditioning args
    parser.add_argument("--lam_ckpt_path", type=str, default=None,
                       help="Path to LAM checkpoint for action conditioning")
    parser.add_argument("--lam_config", type=str, default=None,
                       help="Path to LAM config file")

    # Training args
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_actions", type=int, default=20,
                       help="Maximum number of actions (T) to use from each sequence for memory management")
    parser.add_argument("--delta_loss_weight", type=float, default=0.0,
                       help="Weight for delta consistency loss regularization (0.0 = disabled)")
    parser.add_argument("--scheduled_sampling", action="store_true",
                       help="Enable scheduled sampling to reduce exposure bias")
    parser.add_argument("--sampling_decay_steps", type=int, default=None,
                       help="Number of epochs for sampling probability to reach max_sampling_prob (default: num_epochs)")
    parser.add_argument("--max_sampling_prob", type=float, default=0.7,
                       help="Maximum sampling probability (0.7 = 70% model predictions, 30% teacher forcing)")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout probability for regularization")

    # Overfitting test args
    parser.add_argument("--overfit_test", action="store_true",
                       help="Run overfitting test on single batch instead of full training")
    parser.add_argument("--overfit_epochs", type=int, default=200,
                       help="Number of epochs for overfitting test")

    # Logging and sampling
    parser.add_argument("--wandb_project", type=str, default="autoregressive-gesture-decoder")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--sample_freq", type=int, default=10)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--generation_temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = get_args()
    try:
        trainer = AutoregressiveTrainer(args)
        trainer.training_loop()
        print("Training finished successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise


if __name__ == "__main__":
    main()
