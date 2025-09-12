import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime
from pathlib import Path

from .model_cnn_seededcnn import CNNSeededCNNVAE
from .data_loader import StreamingGestureDataLoader
from .loss_functions import VAELoss, BetaVAELoss, KLAnnealingVAELoss, create_loss_function
from .utils import (CoordinateQuantizer, CheckpointManager, ExperimentLogger, Visualizer,
                   summarize_model_architecture, generate_experiment_name, create_train_val_split,
                   parse_validation_interval, generate_reconstructed_trajectories, 
                   generate_sampled_trajectories)
from .monitoring import VAEMonitor, integrate_monitor_into_training


def train_epoch(model, train_loader, optimizer, loss_fn, quantizer, device, 
                checkpoint_manager, logger, visualizer, epoch, step_counter, 
                val_loader=None, val_steps=None, log_interval=50, vis_with_val=True,
                monitor=None):
    """Train for one epoch with integrated logging, monitoring and validation scheduling."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_recon_x_loss = 0
    total_recon_y_loss = 0
    total_kl_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, data in enumerate(pbar):
        data = data.to(device)  # [B, 250, 2]
        batch_size = data.shape[0]
        
        # Quantize targets for classification
        targets = quantizer.quantize(data)  # [B, 250, 2]
        
        # Forward pass
        optimizer.zero_grad()
        
        outputs = model(data)
        logits = outputs['logits']  # [B, 250, 2, k_classes] - separate x/y predictions
        mu = outputs['mu']
        log_sigma = outputs['log_sigma']
        
        # Compute loss
        if isinstance(loss_fn, BetaVAELoss):
            loss_fn.update_beta(step_counter[0])
        elif isinstance(loss_fn, KLAnnealingVAELoss):
            loss_fn.update_beta(epoch)
        
        loss_dict = loss_fn(logits, targets, mu, log_sigma)
        loss = loss_dict['total_loss']
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_recon_loss += loss_dict['recon_loss'].item()
        total_recon_x_loss += loss_dict['recon_x_loss'].item()
        total_recon_y_loss += loss_dict['recon_y_loss'].item()
        total_kl_loss += loss_dict['kl_loss'].item()
        num_batches += 1
        step_counter[0] += 1
        
        # Advanced monitoring if enabled
        if monitor is not None and step_counter[0] % (log_interval * 2) == 0:
            monitor_metrics = integrate_monitor_into_training(
                monitor, model, outputs, targets, loss_dict, data
            )
            logger.log_metrics(monitor_metrics, step=step_counter[0])
        
        # Log training metrics
        if step_counter[0] % log_interval == 0:
            metrics = {
                'train/loss': loss.item(),
                'train/recon_loss': loss_dict['recon_loss'].item() if torch.is_tensor(loss_dict['recon_loss']) else loss_dict['recon_loss'],
                'train/recon_x_loss': loss_dict['recon_x_loss'].item() if torch.is_tensor(loss_dict['recon_x_loss']) else loss_dict['recon_x_loss'],
                'train/recon_y_loss': loss_dict['recon_y_loss'].item() if torch.is_tensor(loss_dict['recon_y_loss']) else loss_dict['recon_y_loss'],
                'train/kl_loss': loss_dict['kl_loss'].item() if torch.is_tensor(loss_dict['kl_loss']) else loss_dict['kl_loss'],
                'train/beta': loss_dict.get('beta', 1.0),
                'train/epoch': epoch + batch_idx / len(train_loader)
            }
            logger.log_metrics(metrics, step=step_counter[0])
        
        # Register training state for emergency saving
        train_metrics = {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches
        }
        checkpoint_manager.register_training_state(
            model, optimizer, epoch, train_metrics
        )
        
        # Check if validation should be performed
        should_validate = (
            val_loader is not None and 
            val_steps is not None and 
            step_counter[0] % val_steps == 0 and
            step_counter[0] > 0  # Skip step 0
        )
        
        if should_validate:
            val_metrics = validate_epoch(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                quantizer=quantizer,
                device=device,
                logger=logger,
                visualizer=visualizer,
                epoch=epoch,
                step_counter=step_counter,
                create_visualizations=vis_with_val,
                monitor=monitor
            )
            
            # Log validation step info
            pbar.write(f"Step {step_counter[0]} Validation - "
                      f"Loss: {val_metrics['total_loss']:.4f}, "
                      f"Recon: {val_metrics['recon_loss']:.4f} "
                      f"(X: {val_metrics['recon_x_loss']:.4f}, Y: {val_metrics['recon_y_loss']:.4f}), "
                      f"KL: {val_metrics['kl_loss']:.4f}")
            
            # Return to training mode after validation
            model.train()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Recon': f'{loss_dict["recon_loss"].item():.4f}',
            'KL': f'{loss_dict["kl_loss"].item():.4f}',
            'Beta': f'{loss_dict.get("beta", 1.0):.3f}',
            'Step': step_counter[0]
        })
    
    return {
        'total_loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'recon_x_loss': total_recon_x_loss / num_batches,
        'recon_y_loss': total_recon_y_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches
    }


def validate_epoch(model, val_loader, loss_fn, quantizer, device, logger, 
                  visualizer, epoch, step_counter, create_visualizations=True,
                  monitor=None):
    """Validate for one epoch with comprehensive logging, visualization, and monitoring dashboard."""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_recon_x_loss = 0
    total_recon_y_loss = 0
    total_kl_loss = 0
    num_batches = 0
    
    # Collect losses for histogram visualization
    all_total_losses = []
    all_recon_losses = []
    all_recon_x_losses = []
    all_recon_y_losses = []
    all_kl_losses = []
    
    # Store data for visualization
    val_data_batch = None
    val_logits = None
    val_outputs = None
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_loader, desc='Validation')):
            data = data.to(device)
            
            # Quantize targets
            targets = quantizer.quantize(data)  # [B, 250, 2] - keep both x/y
            
            # Forward pass
            outputs = model(data)
            logits = outputs['logits']
            mu = outputs['mu']
            log_sigma = outputs['log_sigma']
            
            # Compute loss
            loss_dict = loss_fn(logits, targets, mu, log_sigma)
            
            # Update metrics
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_recon_x_loss += loss_dict['recon_x_loss'].item()
            total_recon_y_loss += loss_dict['recon_y_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            num_batches += 1
            
            # Collect losses for histograms
            all_total_losses.append(loss_dict['total_loss'].item())
            all_recon_losses.append(loss_dict['recon_loss'].item())
            all_recon_x_losses.append(loss_dict['recon_x_loss'].item())
            all_recon_y_losses.append(loss_dict['recon_y_loss'].item())
            all_kl_losses.append(loss_dict['kl_loss'].item())
            
            # Store first batch for visualization
            if batch_idx == 0 and create_visualizations:
                val_data_batch = data
                val_logits = logits
                val_outputs = outputs
            
            # Limit validation batches for speed
            if batch_idx >= 4:  # Only process first 5 batches
                break
    
    # Create visualizations
    if create_visualizations and val_data_batch is not None:
        # Generate reconstructed trajectories with 20 samples
        num_vis_samples = min(20, val_data_batch.shape[0])
        reconstructed_coords = generate_reconstructed_trajectories(
            model, val_data_batch[:num_vis_samples], quantizer, device
        )
        
        # Generate sampled trajectories from latent space
        sampled_coords = generate_sampled_trajectories(
            model, num_vis_samples, quantizer, device
        )
        
        # Log reconstructed and sampled trajectories (original will be in val_compare)
        logger.log_model_samples(reconstructed_coords, step=step_counter[0], 
                               sample_type="reconstructed")
        logger.log_model_samples(sampled_coords, step=step_counter[0], 
                               sample_type="sampled")
        
        # Log individual comparison plots under val_compare
        logger.log_val_compare_plots(
            val_data_batch[:num_vis_samples], reconstructed_coords, sampled_coords,
            step=step_counter[0]
        )
        
        # Create and log advanced visualizations
        if visualizer:
            try:
                # Logit heatmap visualization
                heatmap_fig = visualizer.visualize_logit_heatmap(
                    val_logits, val_data_batch, quantizer,
                    title_prefix=f"Epoch {epoch}", max_samples=2
                )
                logger.log_metrics({"val/logit_heatmap": heatmap_fig}, 
                                 step=step_counter[0])
                
                # Classification predictions
                pred_fig = visualizer.visualize_classification_predictions(
                    val_logits, val_data_batch, quantizer,
                    title_prefix=f"Epoch {epoch}", max_samples=4
                )
                logger.log_metrics({"val/predictions": pred_fig}, 
                                 step=step_counter[0])
                
                # Latent space visualization
                latent_fig = visualizer.visualize_latent_space(
                    model, val_loader, device, max_samples=200
                )
                logger.log_metrics({"val/latent_space": latent_fig}, 
                                 step=step_counter[0])
                
                # Create wandb-compatible heatmaps
                if logger.enabled:
                    heatmap_data = visualizer.create_wandb_logit_heatmaps(
                        val_logits, val_data_batch, quantizer, max_samples=2
                    )
                    for key, fig in heatmap_data.items():
                        logger.log_metrics({f"val/{key}": fig}, 
                                         step=step_counter[0])
                
            except Exception as e:
                print(f"Warning: Visualization failed: {e}")
    
    val_metrics = {
        'total_loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'recon_x_loss': total_recon_x_loss / num_batches,
        'recon_y_loss': total_recon_y_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches
    }
    
    # Log validation metrics
    logger.log_metrics({
        'val/loss': val_metrics['total_loss'],
        'val/recon_loss': val_metrics['recon_loss'],
        'val/recon_x_loss': val_metrics['recon_x_loss'],
        'val/recon_y_loss': val_metrics['recon_y_loss'],
        'val/kl_loss': val_metrics['kl_loss']
    }, step=step_counter[0])
    
    # Create monitoring dashboard during validation
    if monitor is not None:
        monitor_fig = monitor.create_monitoring_plots()
        logger.log_metrics({
            'monitor/dashboard': monitor_fig
        }, step=step_counter[0])
        # Plotly figures don't need explicit closing
        
        # Log summary metrics
        summary = monitor.get_summary_metrics()
        logger.log_metrics({
            f'monitor/summary_{k}': v for k, v in summary.items()
        }, step=step_counter[0])
    
    # Log loss histograms (like in no_int classifier)
    if create_visualizations and logger.enabled:
        logger.log_loss_histogram(all_total_losses, step=step_counter[0], loss_type="total")
        logger.log_loss_histogram(all_recon_losses, step=step_counter[0], loss_type="reconstruction")
        logger.log_loss_histogram(all_recon_x_losses, step=step_counter[0], loss_type="reconstruction_x")
        logger.log_loss_histogram(all_recon_y_losses, step=step_counter[0], loss_type="reconstruction_y")
        logger.log_loss_histogram(all_kl_losses, step=step_counter[0], loss_type="kl_divergence")
    
    return val_metrics


def main():
    parser = argparse.ArgumentParser(description='Train CNN-SeededCNN VAE')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Data directory path')
    parser.add_argument('--val_data', type=str, default=None,
                       help='Validation data directory (if provided, uses manual split)')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Auto validation split ratio when --val_data not provided (0.0-1.0, default: 0.2)')
    parser.add_argument('--no_val', action='store_true',
                       help='Skip validation entirely (faster training, no validation metrics)')
    
    # Model arguments
    parser.add_argument('--d_latent', type=int, default=124,
                       help='Latent dimension')
    parser.add_argument('--k_classes', type=int, default=5000,
                       help='Number of quantization classes')
    parser.add_argument('--encoder_hidden_dim', type=int, default=256,
                       help='Hidden dimension for encoder')
    parser.add_argument('--decoder_feature_dim', type=int, default=256,
                       help='Feature dimension for SeededCNN decoder')
    parser.add_argument('--decoder_num_layers', type=int, default=8,
                       help='Number of dilated conv layers in SeededCNN decoder')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta value for VAE loss')
    parser.add_argument('--beta_warmup', action='store_true',
                       help='Use beta warmup for VAE training')
    parser.add_argument('--beta_warmup_steps', type=int, default=1000,
                       help='Steps for beta warmup')
    parser.add_argument('--kl_anneal', type=int, default=None,
                       help='Enable KL annealing: epoch at which beta reaches maximum value (linear growth from 0)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loader workers (default=0 to avoid worker crashes)')
    
    # Logging and checkpoint arguments
    parser.add_argument('--project_name', type=str, default='vae_experiment',
                       help='wandb project name')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--disable_wandb', action='store_true',
                       help='Disable wandb logging')
    parser.add_argument('--checkpoint_base_dir', type=str, default='./checkpoints',
                       help='Base directory for checkpoints')
    parser.add_argument('--save_every', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--num_elite', type=int, default=3,
                       help='Number of elite models to keep based on validation score')
    parser.add_argument('--log_interval', type=int, default=50,
                       help='Log training metrics every N steps')
    parser.add_argument('--val_interval', type=str, default='1%',
                       help='Validation interval: N steps (int) or N%% of epoch (default: "1%%")')
    parser.add_argument('--vis_with_val', action='store_true', default=True,
                       help='Create visualizations with each validation (default: True)')
    parser.add_argument('--vis_interval', type=int, default=None,
                       help='Override: Create visualizations every N steps (overrides --vis_with_val)')
    
    # Monitoring arguments (enabled by default)
    parser.add_argument('--disable_monitoring', action='store_true',
                       help='Disable advanced VAE monitoring (enabled by default for better insights)')
    
    # Resume training arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from (e.g., ./checkpoints/exp/best_model.pt)')
    parser.add_argument('--auto_resume', action='store_true',
                       help='Automatically resume from the latest checkpoint in checkpoint_base_dir/experiment_name')
    parser.add_argument('--list_checkpoints', action='store_true',
                       help='List available checkpoints and exit')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Generate experiment name with encoder/decoder info
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    encoder_name = "cnn"  # Current implementation uses CNN encoder
    decoder_name = "seededcnn"  # Current implementation uses SeededCNN decoder
    
    # Generate unique experiment name
    experiment_name = args.experiment_name or generate_experiment_name(
        encoder_name=encoder_name,
        decoder_name=decoder_name, 
        args=args,
        timestamp=timestamp
    )
    
    # Create checkpoint directory based on experiment name
    checkpoint_dir = Path(args.checkpoint_base_dir) / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment: {experiment_name}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Handle checkpoint listing
    if args.list_checkpoints:
        temp_manager = CheckpointManager(str(checkpoint_dir), save_source=False, num_elite=args.num_elite)
        try:
            temp_manager.list_available_checkpoints()
            
            # Also show elite models
            elite_info = temp_manager.get_elite_models_info()
            if elite_info:
                print(f"\n🏆 Elite Models ({len(elite_info)} found):")
                for info in elite_info:
                    status = "✓" if info['exists'] else "✗"
                    print(f"  {status} {info['rank']}. {info['filename']} - Score: {info['score']:.6f} (Epoch {info['epoch']})")
            else:
                print("\n📝 No elite models found")
                
        except Exception as e:
            print(f"Error listing checkpoints: {e}")
        return
    
    # Create modular utilities
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        max_checkpoints=5,
        save_source=True,
        num_elite=args.num_elite
    )
    
    print(f"Elite model saving enabled: keeping top {args.num_elite} models")
    
    logger = ExperimentLogger(
        project=args.project_name,
        experiment_name=experiment_name,
        config=vars(args),
        enabled=not args.disable_wandb
    )
    
    visualizer = Visualizer(save_dir=str(checkpoint_dir / 'visualizations'))
    
    # Create data loaders with automatic train/val splitting
    print("Creating data loaders...")
    
    if args.no_val:
        # No validation - just training
        print("Validation disabled - training only mode")
        train_loader_manager = StreamingGestureDataLoader(
            train_path=args.data_path,
            val_path=None,
            batch_size=args.batch_size,
            sequence_length=250,
            normalize=True,
            num_workers=args.num_workers,
            shuffle=True
        )
        
        try:
            train_loader = train_loader_manager.get_train_loader()
            val_loader = None
        except ValueError:
            print(f"Error: Could not create training data loader from {args.data_path}")
            return
        
    elif args.val_data:
        # Manual validation data provided
        print(f"Using manual train/val split:")
        print(f"  Train data: {args.data_path}")
        print(f"  Val data: {args.val_data}")
        
        train_loader_manager = StreamingGestureDataLoader(
            train_path=args.data_path,
            val_path=None,
            batch_size=args.batch_size,
            sequence_length=250,
            normalize=True,
            num_workers=args.num_workers,
            shuffle=True
        )
        
        val_loader_manager = StreamingGestureDataLoader(
            train_path=None,
            val_path=args.val_data,
            batch_size=args.batch_size,
            sequence_length=250,
            normalize=True,
            num_workers=args.num_workers,
            shuffle=False
        )
        
        try:
            train_loader = train_loader_manager.get_train_loader()
        except ValueError:
            print(f"Error: Could not create training data loader from {args.data_path}")
            return
        
        try:
            val_loader = val_loader_manager.get_val_loader()
        except ValueError:
            print(f"Error: Could not create validation data loader from {args.val_data}")
            val_loader = None
            
    else:
        # Automatic train/val split from training data
        print(f"Creating automatic train/val split ({1-args.val_split:.1%}/{args.val_split:.1%})")
        print(f"Data source: {args.data_path}")
        
        # Create a single dataset manager to get the full dataset
        full_loader_manager = StreamingGestureDataLoader(
            train_path=args.data_path,
            val_path=None,
            batch_size=args.batch_size,
            sequence_length=250,
            normalize=True,
            num_workers=args.num_workers,
            shuffle=True
        )
        
        try:
            # Get the full dataset (before creating data loader)
            full_dataset = full_loader_manager.train_dataset
            if full_dataset is None:
                raise ValueError("Could not create dataset")
            
            # Split dataset
            train_dataset, val_dataset = create_train_val_split(
                full_dataset, val_split=args.val_split, seed=42
            )
            
            # Create data loaders from split datasets
            from torch.utils.data import DataLoader
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False
            )
            
        except ValueError as e:
            print(f"Error: Could not create automatic train/val split: {e}")
            return
    
    # Print data loader info
    if hasattr(locals().get('train_loader_manager'), 'get_data_info'):
        data_info = train_loader_manager.get_data_info()
        print(f"Data info: {data_info}")
    
    print(f"Train batches per epoch: {len(train_loader)}")
    if val_loader:
        print(f"Val batches per epoch: {len(val_loader)}")
    else:
        print("No validation data - training only")
    
    # Create model
    print("Creating model...")
    model = CNNSeededCNNVAE(
        d_latent=args.d_latent,
        k_classes=args.k_classes,
        encoder_hidden_dim=args.encoder_hidden_dim,
        decoder_feature_dim=args.decoder_feature_dim,
        decoder_num_layers=args.decoder_num_layers
    )
    
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Generate detailed model architecture summary
    model_summary = summarize_model_architecture(model, input_shape=(250, 2), device=device)
    print("\n" + model_summary)
    
    # Log model architecture to wandb
    logger.log_model_architecture(model, model_summary_text=model_summary)
    
    # Create loss function
    if args.kl_anneal is not None:
        loss_fn = KLAnnealingVAELoss(
            k_classes=args.k_classes,
            beta_max=args.beta,
            anneal_epochs=args.kl_anneal
        )
        print(f"Using KL annealing VAE: beta grows linearly from 0 to {args.beta} over {args.kl_anneal} epochs")
    elif args.beta_warmup:
        loss_fn = BetaVAELoss(
            k_classes=args.k_classes,
            beta_start=0.0,
            beta_end=args.beta,
            beta_warmup_steps=args.beta_warmup_steps
        )
        print(f"Using Beta-VAE with warmup: {args.beta_warmup_steps} steps")
    else:
        loss_fn = VAELoss(
            k_classes=args.k_classes,
            beta=args.beta
        )
        print(f"Using standard VAE with beta={args.beta}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create quantizer
    quantizer = CoordinateQuantizer(num_classes=args.k_classes)
    
    # Parse validation interval
    val_steps = None
    if val_loader is not None:
        val_steps = parse_validation_interval(args.val_interval, len(train_loader))
        print(f"Validation will run every {val_steps} steps ({args.val_interval})")
    
    # Initialize VAE monitor (enabled by default)
    monitor = None
    if not args.disable_monitoring:
        monitor = VAEMonitor(d_latent=args.d_latent, k_classes=args.k_classes)
        print("Advanced VAE monitoring enabled (use --disable_monitoring to turn off)")
        print("Monitoring dashboard will be created with each validation")
    else:
        print("VAE monitoring disabled")
    
    # Handle training resume from checkpoint
    start_epoch = 1
    best_val_loss = float('inf')
    step_counter = [0]  # Use list for mutable reference
    resume_info = None
    
    # Determine checkpoint to resume from
    resume_checkpoint_path = None
    if args.resume:
        resume_checkpoint_path = args.resume
        print(f"Resume requested from: {resume_checkpoint_path}")
    elif args.auto_resume:
        try:
            resume_checkpoint_path = checkpoint_manager.find_latest_checkpoint()
            print(f"Auto-resume found checkpoint: {resume_checkpoint_path}")
        except FileNotFoundError as e:
            print(f"Auto-resume failed: {e}")
            print("Starting fresh training instead")
    
    # Load checkpoint if specified
    if resume_checkpoint_path:
        try:
            resume_info = checkpoint_manager.load_checkpoint_for_training_resume(
                resume_checkpoint_path, model, optimizer, scheduler, logger
            )
            start_epoch = resume_info['start_epoch']
            best_val_loss = resume_info['best_val_loss']
            step_counter[0] = resume_info['step_counter']
            
            print(f"✅ Successfully resumed from checkpoint!")
            print(f"📊 Training will continue from epoch {start_epoch}")
            print(f"🎯 Best validation loss so far: {best_val_loss:.4f}")
            
            # Update logger with resume info if needed
            if logger.enabled:
                logger.log_metrics({
                    'resume/checkpoint_epoch': resume_info['checkpoint_epoch'],
                    'resume/start_epoch': start_epoch,
                    'resume/best_val_loss': best_val_loss,
                    'resume/step_counter': step_counter[0]
                }, step=step_counter[0])
                
        except Exception as e:
            print(f"❌ Failed to resume from checkpoint: {e}")
            print("Starting fresh training instead")
            start_epoch = 1
            best_val_loss = float('inf')
            step_counter = [0]
            resume_info = None
    
    # Training loop
    
    print("Starting training...")
    print(f"Logging enabled: {logger.enabled}")
    if args.vis_with_val and val_loader:
        print(f"Visualizations will be created with each validation")
    elif args.vis_interval:
        print(f"Visualization interval: every {args.vis_interval} steps")
    else:
        print("Visualizations disabled")
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train epoch
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            quantizer=quantizer,
            device=device,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            visualizer=visualizer,
            epoch=epoch,
            step_counter=step_counter,
            val_loader=val_loader,
            val_steps=val_steps,
            log_interval=args.log_interval,
            vis_with_val=args.vis_with_val
        )
        
        # Log epoch-level training metrics
        logger.log_metrics({
            'train_epoch/loss': train_metrics['total_loss'],
            'train_epoch/recon_loss': train_metrics['recon_loss'],
            'train_epoch/kl_loss': train_metrics['kl_loss'],
            'epoch': epoch
        }, step=step_counter[0])
        
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Train Loss: {train_metrics['total_loss']:.4f}, "
              f"Recon: {train_metrics['recon_loss']:.4f}, "
              f"KL: {train_metrics['kl_loss']:.4f}")
        
        # End-of-epoch validation (if not already done during training)
        val_metrics = None
        is_best = False
        
        if val_loader:
            # Only run end-of-epoch validation if we haven't done step-based validation
            # or for special epochs (first, last, etc.)
            run_epoch_val = (
                val_steps is None or  # No step-based validation configured
                epoch == 1 or  # First epoch
                epoch == args.epochs or  # Last epoch
                epoch % max(1, args.epochs // 10) == 0  # Every 10% of total epochs
            )
            
            if run_epoch_val:
                # Create visualizations for end-of-epoch validation
                create_vis = (epoch % max(1, args.epochs // 10) == 0) or epoch == args.epochs
                
                val_metrics = validate_epoch(
                    model=model,
                    val_loader=val_loader,
                    loss_fn=loss_fn,
                    quantizer=quantizer,
                    device=device,
                    logger=logger,
                    visualizer=visualizer,
                    epoch=epoch,
                    step_counter=step_counter,
                    create_visualizations=create_vis,
                    monitor=monitor
                )
                
                print(f"End-of-epoch Val Loss: {val_metrics['total_loss']:.4f}, "
                      f"Recon: {val_metrics['recon_loss']:.4f}, "
                      f"KL: {val_metrics['kl_loss']:.4f}")
                
                # Check for best model
                is_best = val_metrics['total_loss'] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics['total_loss']
                    print(f"🎉 New best validation loss: {best_val_loss:.4f}")
                
                # Save elite model if validation score qualifies (total loss)
                val_score = val_metrics['total_loss']
                elite_saved = checkpoint_manager.save_elite_model(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_score=val_score,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    scheduler=scheduler,
                    step_counter=step_counter[0],
                    steps_per_epoch=len(train_loader)
                )
                
                # Also save CE-elite model based on reconstruction loss only
                ce_score = val_metrics['recon_loss']
                ce_elite_saved = checkpoint_manager.save_elite_model_ce(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    ce_score=ce_score,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    scheduler=scheduler,
                    step_counter=step_counter[0],
                    steps_per_epoch=len(train_loader)
                )
        
        # Save checkpoints
        if epoch % args.save_every == 0 or epoch == args.epochs or is_best:
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                is_best=is_best,
                scheduler=scheduler,
                step_counter=step_counter[0],
                best_val_loss=best_val_loss,
                steps_per_epoch=len(train_loader)
            )
        
        # Update learning rate
        scheduler.step()
        
        # Log learning rate
        current_lr = scheduler.get_last_lr()[0]
        logger.log_metrics({'train/lr': current_lr}, step=step_counter[0])
    
    print("🎊 Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Display elite models summary
    elite_info = checkpoint_manager.get_elite_models_info()
    if elite_info:
        print(f"\n🏆 Elite Models Summary ({len(elite_info)} models saved):")
        for info in elite_info:
            print(f"  {info['rank']}. {info['filename']} - Score: {info['score']:.6f} (Epoch {info['epoch']})")
        
        best_elite = checkpoint_manager.get_best_elite_model()
        if best_elite:
            import os
            print(f"\n🥇 Best elite model: {os.path.basename(best_elite)}")
    else:
        print("\n📝 No elite models were saved during training")
    
    # Print resumption information
    if resume_info:
        print(f"\n📊 Training Summary:")
        print(f"  • Resumed from epoch {resume_info['checkpoint_epoch']}")
        print(f"  • Trained epochs {start_epoch} to {args.epochs}")
        print(f"  • Total epochs: {args.epochs}")
        print(f"  • Final step: {step_counter[0]}")
    else:
        print(f"\n📊 Training Summary:")
        print(f"  • Trained epochs 1 to {args.epochs}")
        print(f"  • Total epochs: {args.epochs}")
        print(f"  • Final step: {step_counter[0]}")
    
    print(f"\n💾 To resume training from this point, use:")
    print(f"  python -m conditioned_gesture_generator.gesture_vae.train_cnn_seededcnn \\")
    print(f"    --resume {checkpoint_dir}/best_model.pt \\")
    print(f"    --epochs {args.epochs + 50} \\")
    print(f"    [your other arguments]")
    
    # Show elite model resume options
    if elite_info:
        print(f"\n🏆 To resume from best elite model:")
        best_elite = checkpoint_manager.get_best_elite_model()
        if best_elite:
            print(f"  python -m conditioned_gesture_generator.gesture_vae.train_cnn_seededcnn \\")
            print(f"    --resume {best_elite} \\")
            print(f"    --epochs {args.epochs + 50} \\")
            print(f"    [your other arguments]")
    
    print(f"\n🔍 To list available checkpoints and elite models:")
    print(f"  python -m conditioned_gesture_generator.gesture_vae.train_cnn_seededcnn \\")
    print(f"    --list_checkpoints \\")
    print(f"    --checkpoint_base_dir {args.checkpoint_base_dir} \\")
    print(f"    --experiment_name {experiment_name}")
    
    # Finish logging
    logger.finish()


if __name__ == '__main__':
    main()