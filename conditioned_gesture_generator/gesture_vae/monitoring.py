"""
Advanced monitoring utilities for VAE training.
Provides detailed metrics for understanding VAE behavior and diagnosing issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
from collections import defaultdict, deque


class VAEMonitor:
    """
    Comprehensive monitoring for VAE training including:
    - KL divergence analysis per dimension
    - Posterior collapse detection
    - Reconstruction quality metrics
    - Latent space statistics
    - Gradient flow tracking
    """
    
    def __init__(self, d_latent: int = 128, k_classes: int = 3000, 
                 history_window: int = 100):
        """
        Args:
            d_latent: Latent dimension size
            k_classes: Number of quantization classes
            history_window: Window size for moving averages
        """
        self.d_latent = d_latent
        self.k_classes = k_classes
        self.history_window = history_window
        
        # Initialize tracking buffers
        self.reset()
        
    def reset(self):
        """Reset all tracking buffers."""
        # KL divergence tracking per dimension
        self.kl_per_dim_history = deque(maxlen=self.history_window)
        
        # Latent statistics
        self.mu_stats_history = deque(maxlen=self.history_window)
        self.sigma_stats_history = deque(maxlen=self.history_window)
        
        # Reconstruction metrics
        self.recon_accuracy_history = deque(maxlen=self.history_window)
        self.coord_accuracy_history = {'x': deque(maxlen=self.history_window),
                                       'y': deque(maxlen=self.history_window)}
        
        # Gradient tracking
        self.grad_norms = defaultdict(lambda: deque(maxlen=self.history_window))
        
        # Posterior collapse tracking
        self.active_dims_history = deque(maxlen=self.history_window)
        
        # Loss component ratios
        self.loss_ratio_history = deque(maxlen=self.history_window)
    
    def compute_kl_per_dimension(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> Dict:
        """
        Compute KL divergence for each latent dimension.
        
        Args:
            mu: Latent mean [B, d_latent]
            log_sigma: Latent log std [B, d_latent]
            
        Returns:
            Dict with per-dimension KL statistics
        """
        with torch.no_grad():
            # KL per dimension: -0.5 * (1 + 2*log_sigma - mu^2 - exp(2*log_sigma))
            kl_per_dim = -0.5 * (1 + 2*log_sigma - mu.pow(2) - (2*log_sigma).exp())
            kl_per_dim = kl_per_dim.mean(dim=0)  # Average over batch
            
            # Detect posterior collapse (dims with very low KL)
            collapse_threshold = 0.01
            collapsed_dims = (kl_per_dim < collapse_threshold).sum().item()
            active_dims = self.d_latent - collapsed_dims
            
            self.kl_per_dim_history.append(kl_per_dim.cpu().numpy())
            self.active_dims_history.append(active_dims)
            
            return {
                'kl_per_dim_mean': kl_per_dim.mean().item(),
                'kl_per_dim_std': kl_per_dim.std().item(),
                'kl_per_dim_max': kl_per_dim.max().item(),
                'kl_per_dim_min': kl_per_dim.min().item(),
                'active_dimensions': active_dims,
                'collapsed_dimensions': collapsed_dims,
                'collapse_ratio': collapsed_dims / self.d_latent
            }
    
    def compute_latent_statistics(self, mu: torch.Tensor, log_sigma: torch.Tensor, 
                                  z: Optional[torch.Tensor] = None) -> Dict:
        """
        Compute statistics about the latent space.
        
        Args:
            mu: Latent mean [B, d_latent]
            log_sigma: Latent log std [B, d_latent]
            z: Sampled latent (optional) [B, d_latent]
            
        Returns:
            Dict with latent space statistics
        """
        with torch.no_grad():
            sigma = log_sigma.exp()
            
            # Mu statistics
            mu_mean = mu.mean().item()
            mu_std = mu.std().item()
            mu_abs_mean = mu.abs().mean().item()
            
            # Sigma statistics
            sigma_mean = sigma.mean().item()
            sigma_std = sigma.std().item()
            sigma_min = sigma.min().item()
            sigma_max = sigma.max().item()
            
            # Latent space coverage (how spread out the encodings are)
            latent_norm = mu.norm(dim=1).mean().item()
            
            stats = {
                'mu_mean': mu_mean,
                'mu_std': mu_std,
                'mu_abs_mean': mu_abs_mean,
                'sigma_mean': sigma_mean,
                'sigma_std': sigma_std,
                'sigma_min': sigma_min,
                'sigma_max': sigma_max,
                'latent_norm': latent_norm,
            }
            
            # If sampled latent is provided, compute sampling statistics
            if z is not None:
                z_mean = z.mean().item()
                z_std = z.std().item()
                z_norm = z.norm(dim=1).mean().item()
                
                stats.update({
                    'z_mean': z_mean,
                    'z_std': z_std,
                    'z_norm': z_norm,
                })
            
            self.mu_stats_history.append({'mean': mu_mean, 'std': mu_std, 'norm': latent_norm})
            self.sigma_stats_history.append({'mean': sigma_mean, 'std': sigma_std})
            
            return stats
    
    def compute_reconstruction_metrics(self, logits: torch.Tensor, targets: torch.Tensor,
                                       original_coords: Optional[torch.Tensor] = None) -> Dict:
        """
        Compute detailed reconstruction quality metrics.
        
        Args:
            logits: Model predictions [B, T, 2, num_classes]
            targets: Target class indices [B, T, 2]
            original_coords: Original coordinates [B, T, 2] (optional)
            
        Returns:
            Dict with reconstruction metrics
        """
        with torch.no_grad():
            B, T = targets.shape[:2]
            
            # Compute accuracy per coordinate
            x_pred = torch.argmax(logits[:, :, 0, :], dim=-1)  # [B, T]
            y_pred = torch.argmax(logits[:, :, 1, :], dim=-1)  # [B, T]
            
            x_accuracy = (x_pred == targets[:, :, 0]).float().mean().item()
            y_accuracy = (y_pred == targets[:, :, 1]).float().mean().item()
            overall_accuracy = ((x_pred == targets[:, :, 0]) & 
                               (y_pred == targets[:, :, 1])).float().mean().item()
            
            # Top-5 accuracy
            x_top5 = torch.topk(logits[:, :, 0, :], k=min(5, self.k_classes), dim=-1).indices  # [B, T, 5]
            y_top5 = torch.topk(logits[:, :, 1, :], k=min(5, self.k_classes), dim=-1).indices  # [B, T, 5]
            
            x_target_expanded = targets[:, :, 0].unsqueeze(-1)  # [B, T, 1]
            y_target_expanded = targets[:, :, 1].unsqueeze(-1)  # [B, T, 1]
            
            x_top5_acc = (x_top5 == x_target_expanded).any(dim=-1).float().mean().item()
            y_top5_acc = (y_top5 == y_target_expanded).any(dim=-1).float().mean().item()
            
            # Confidence scores (max probability)
            x_confidence = F.softmax(logits[:, :, 0, :], dim=-1).max(dim=-1).values.mean().item()
            y_confidence = F.softmax(logits[:, :, 1, :], dim=-1).max(dim=-1).values.mean().item()
            
            # Entropy of predictions (uncertainty measure)
            x_probs = F.softmax(logits[:, :, 0, :], dim=-1)
            y_probs = F.softmax(logits[:, :, 1, :], dim=-1)
            
            x_entropy = -(x_probs * torch.log(x_probs + 1e-8)).sum(dim=-1).mean().item()
            y_entropy = -(y_probs * torch.log(y_probs + 1e-8)).sum(dim=-1).mean().item()
            
            metrics = {
                'x_accuracy': x_accuracy,
                'y_accuracy': y_accuracy,
                'overall_accuracy': overall_accuracy,
                'x_top5_accuracy': x_top5_acc,
                'y_top5_accuracy': y_top5_acc,
                'x_confidence': x_confidence,
                'y_confidence': y_confidence,
                'x_entropy': x_entropy,
                'y_entropy': y_entropy,
            }
            
            # If original coordinates provided, compute MSE
            if original_coords is not None:
                x_pred_coords = x_pred.float() / self.k_classes
                y_pred_coords = y_pred.float() / self.k_classes
                pred_coords = torch.stack([x_pred_coords, y_pred_coords], dim=-1)
                
                mse = F.mse_loss(pred_coords, original_coords).item()
                mae = F.l1_loss(pred_coords, original_coords).item()
                
                metrics.update({
                    'coord_mse': mse,
                    'coord_mae': mae,
                })
            
            self.recon_accuracy_history.append(overall_accuracy)
            self.coord_accuracy_history['x'].append(x_accuracy)
            self.coord_accuracy_history['y'].append(y_accuracy)
            
            return metrics
    
    def track_gradient_flow(self, model: nn.Module) -> Dict:
        """
        Track gradient flow through the model.
        
        Args:
            model: The VAE model
            
        Returns:
            Dict with gradient statistics
        """
        grad_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_mean = param.grad.data.abs().mean().item()
                
                # Track by layer type
                if 'encoder' in name:
                    layer_type = 'encoder'
                elif 'decoder' in name:
                    layer_type = 'decoder'
                else:
                    layer_type = 'other'
                
                self.grad_norms[layer_type].append(grad_norm)
                
                # Store detailed stats for key layers
                if any(key in name for key in ['fc_mu', 'fc_log_sigma', 'classifier']):
                    grad_stats[f'grad_norm_{name}'] = grad_norm
                    grad_stats[f'grad_mean_{name}'] = grad_mean
        
        # Compute aggregate statistics
        if self.grad_norms['encoder']:
            grad_stats['encoder_grad_norm'] = np.mean(list(self.grad_norms['encoder']))
        if self.grad_norms['decoder']:
            grad_stats['decoder_grad_norm'] = np.mean(list(self.grad_norms['decoder']))
        
        return grad_stats
    
    def compute_loss_balance(self, recon_loss: float, kl_loss: float, beta: float) -> Dict:
        """
        Track the balance between reconstruction and KL losses.
        
        Args:
            recon_loss: Reconstruction loss value
            kl_loss: KL divergence loss value
            beta: Current beta value
            
        Returns:
            Dict with loss balance metrics
        """
        total_loss = recon_loss + beta * kl_loss
        
        # Compute ratios
        recon_ratio = recon_loss / (total_loss + 1e-8)
        kl_ratio = (beta * kl_loss) / (total_loss + 1e-8)
        
        # Track if losses are balanced (ideally both contribute similarly)
        balance_score = 1.0 - abs(recon_ratio - kl_ratio)
        
        self.loss_ratio_history.append({
            'recon_ratio': recon_ratio,
            'kl_ratio': kl_ratio,
            'balance_score': balance_score
        })
        
        return {
            'recon_ratio': recon_ratio,
            'kl_ratio': kl_ratio,
            'balance_score': balance_score,
            'effective_kl': beta * kl_loss,
        }
    
    def get_summary_metrics(self) -> Dict:
        """
        Get summary of all tracked metrics.
        
        Returns:
            Dict with summary statistics
        """
        summary = {}
        
        # KL collapse summary
        if self.active_dims_history:
            summary['avg_active_dims'] = np.mean(list(self.active_dims_history))
            summary['min_active_dims'] = min(self.active_dims_history)
        
        # Reconstruction accuracy trend
        if self.recon_accuracy_history:
            recent_acc = list(self.recon_accuracy_history)[-10:]
            summary['recent_accuracy'] = np.mean(recent_acc)
            summary['accuracy_trend'] = recent_acc[-1] - recent_acc[0] if len(recent_acc) > 1 else 0
        
        # Loss balance
        if self.loss_ratio_history:
            recent_balance = [x['balance_score'] for x in list(self.loss_ratio_history)[-10:]]
            summary['avg_loss_balance'] = np.mean(recent_balance)
        
        return summary
    
    def create_monitoring_plots(self) -> go.Figure:
        """
        Create comprehensive monitoring visualization using Plotly.
        
        Returns:
            Plotly figure with monitoring plots
        """
        # Create 3x3 subplot grid
        subplot_titles = [
            'KL Divergence per Dimension', 'Active Latent Dimensions', 'Posterior Collapse Detection',
            'Reconstruction Quality', 'KL vs Reconstruction Trade-off', 'Latent Variance Evolution', 
            'Gradient Flow', 'Training Dynamics', 'Model Health'
        ]
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=subplot_titles,
            specs=[[{'type': 'xy'} if i != 0 else {'type': 'xy'} for i in range(3)] for _ in range(3)],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Plot 1: KL per dimension heatmap
        if self.kl_per_dim_history:
            kl_data = np.array(list(self.kl_per_dim_history)[-50:])  # Last 50 steps
            fig.add_trace(go.Heatmap(
                z=kl_data.T,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="KL Divergence")
            ), row=1, col=1)
        
        # Plot 2: Active dimensions over time
        if self.active_dims_history:
            steps = list(range(len(self.active_dims_history)))
            fig.add_trace(go.Scatter(
                x=steps, y=list(self.active_dims_history),
                mode='lines', name='Active Dims',
                line=dict(color='blue', width=2),
                showlegend=False
            ), row=1, col=2)
            
            fig.add_hline(y=self.d_latent, line_dash='dash', line_color='red',
                         opacity=0.5, row=1, col=2)
        
        # Plot 3: Coordinate accuracies
        if self.coord_accuracy_history['x']:
            steps = list(range(len(self.coord_accuracy_history['x'])))
            fig.add_trace(go.Scatter(
                x=steps, y=list(self.coord_accuracy_history['x']),
                mode='lines', name='X accuracy', line=dict(color='blue', width=2),
                showlegend=False
            ), row=1, col=3)
            
            fig.add_trace(go.Scatter(
                x=steps, y=list(self.coord_accuracy_history['y']),
                mode='lines', name='Y accuracy', line=dict(color='red', width=2),
                showlegend=False
            ), row=1, col=3)
        
        # Plot 4: Mu statistics
        if self.mu_stats_history:
            mu_data = list(self.mu_stats_history)
            steps = list(range(len(mu_data)))
            fig.add_trace(go.Scatter(
                x=steps, y=[x['mean'] for x in mu_data],
                mode='lines', name='μ Mean', line=dict(color='blue', width=2),
                showlegend=False
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=steps, y=[x['std'] for x in mu_data],
                mode='lines', name='μ Std', line=dict(color='red', width=2),
                showlegend=False
            ), row=2, col=1)
        
        # Plot 5: Sigma statistics
        if self.sigma_stats_history:
            sigma_data = list(self.sigma_stats_history)
            steps = list(range(len(sigma_data)))
            fig.add_trace(go.Scatter(
                x=steps, y=[x['mean'] for x in sigma_data],
                mode='lines', name='σ Mean', line=dict(color='blue', width=2),
                showlegend=False
            ), row=2, col=2)
            
            fig.add_trace(go.Scatter(
                x=steps, y=[x['std'] for x in sigma_data],
                mode='lines', name='σ Std', line=dict(color='red', width=2),
                showlegend=False
            ), row=2, col=2)
        
        # Plot 6: Loss balance
        if self.loss_ratio_history:
            loss_data = list(self.loss_ratio_history)
            steps = list(range(len(loss_data)))
            fig.add_trace(go.Scatter(
                x=steps, y=[x['recon_ratio'] for x in loss_data],
                mode='lines', name='Recon', line=dict(color='blue', width=2),
                showlegend=False
            ), row=2, col=3)
            
            fig.add_trace(go.Scatter(
                x=steps, y=[x['kl_ratio'] for x in loss_data],
                mode='lines', name='KL', line=dict(color='red', width=2),
                showlegend=False
            ), row=2, col=3)
            
            fig.add_trace(go.Scatter(
                x=steps, y=[x['balance_score'] for x in loss_data],
                mode='lines', name='Balance', line=dict(color='green', width=2, dash='dash'),
                showlegend=False
            ), row=2, col=3)
        
        # Plot 7: Gradient norms
        if self.grad_norms['encoder']:
            steps = list(range(len(self.grad_norms['encoder'])))
            fig.add_trace(go.Scatter(
                x=steps, y=list(self.grad_norms['encoder']),
                mode='lines', name='Encoder', line=dict(color='blue', width=2),
                showlegend=False
            ), row=3, col=1)
        
        if self.grad_norms['decoder']:
            steps = list(range(len(self.grad_norms['decoder'])))
            fig.add_trace(go.Scatter(
                x=steps, y=list(self.grad_norms['decoder']),
                mode='lines', name='Decoder', line=dict(color='red', width=2),
                showlegend=False
            ), row=3, col=1)
        
        # Plot 8: Overall accuracy trend
        if self.recon_accuracy_history:
            steps = list(range(len(self.recon_accuracy_history)))
            fig.add_trace(go.Scatter(
                x=steps, y=list(self.recon_accuracy_history),
                mode='lines', name='Accuracy', line=dict(color='blue', width=2),
                showlegend=False
            ), row=3, col=2)
            
            # Add moving average
            if len(self.recon_accuracy_history) > 10:
                window = min(10, len(self.recon_accuracy_history))
                moving_avg = np.convolve(list(self.recon_accuracy_history), 
                                        np.ones(window)/window, mode='valid')
                fig.add_trace(go.Scatter(
                    x=list(range(window-1, len(self.recon_accuracy_history))),
                    y=moving_avg,
                    mode='lines', name='Moving Avg', line=dict(color='red', width=2),
                    showlegend=False
                ), row=3, col=2)
        
        # Plot 9: Summary text
        summary = self.get_summary_metrics()
        summary_text = '<br>'.join([f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}' 
                                  for k, v in summary.items()])
        
        fig.add_annotation(
            text=summary_text,
            xref="x domain", yref="y domain",
            x=0.1, y=0.5,
            showarrow=False,
            font=dict(family="monospace", size=10),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            title="VAE Training Monitor",
            height=1200,
            width=2000,
            showlegend=False
        )
        
        # Update y-axis for log scale on gradient norms
        fig.update_yaxes(type="log", row=3, col=1)
        
        return fig
    
    def create_trajectory_time_series(self, original: torch.Tensor, 
                                     reconstructed: torch.Tensor,
                                     sampled: torch.Tensor,
                                     max_samples: int = 20) -> go.Figure:
        """
        Create detailed time series plots like in no_int classifier.
        Shows X/Y vs time plots for 20 samples.
        """
        n_samples = min(max_samples, original.shape[0])
        
        # Create time series layout: X vs time and Y vs time for each sample
        fig, axes = plt.subplots(n_samples, 2, figsize=(12, 2 * n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            orig = original[i].detach().cpu().numpy()
            recon = reconstructed[i].detach().cpu().numpy()
            samp = sampled[i].detach().cpu().numpy()
            
            # Find gesture start (ignore leading zeros)
            orig_start = self._find_gesture_start(orig)
            recon_start = self._find_gesture_start(recon)
            samp_start = self._find_gesture_start(samp)
            
            # Use earliest start across all three
            start_idx = min(orig_start, recon_start, samp_start)
            
            time_steps = np.arange(start_idx, len(orig))
            
            # Column 0: X coordinate time series
            axes[i, 0].plot(time_steps, orig[start_idx:, 0], 'b-', linewidth=1.5, 
                          alpha=0.8, label='Original')
            axes[i, 0].plot(time_steps, recon[start_idx:, 0], 'r--', linewidth=1.5, 
                          alpha=0.8, label='Recon')
            axes[i, 0].plot(time_steps, samp[start_idx:, 0], 'g:', linewidth=1.5, 
                          alpha=0.8, label='Sample')
            axes[i, 0].set_title(f'X vs Time - Sample {i+1}', fontsize=8)
            axes[i, 0].set_xlabel('Time Step', fontsize=8)
            axes[i, 0].set_ylabel('X Position', fontsize=8)
            axes[i, 0].set_ylim([0, 1])
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].legend(fontsize=6)
            axes[i, 0].tick_params(labelsize=6)
            
            # Column 1: Y coordinate time series
            axes[i, 1].plot(time_steps, orig[start_idx:, 1], 'b-', linewidth=1.5, 
                          alpha=0.8, label='Original')
            axes[i, 1].plot(time_steps, recon[start_idx:, 1], 'r--', linewidth=1.5, 
                          alpha=0.8, label='Recon')
            axes[i, 1].plot(time_steps, samp[start_idx:, 1], 'g:', linewidth=1.5, 
                          alpha=0.8, label='Sample')
            axes[i, 1].set_title(f'Y vs Time - Sample {i+1}', fontsize=8)
            axes[i, 1].set_xlabel('Time Step', fontsize=8)
            axes[i, 1].set_ylabel('Y Position', fontsize=8)
            axes[i, 1].set_ylim([0, 1])
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].legend(fontsize=6)
            axes[i, 1].tick_params(labelsize=6)
        
        plt.tight_layout()
        return fig
    
    def _find_gesture_start(self, coords):
        """Find the first non-zero point for both x and y coordinates."""
        x_nonzero = np.where(coords[:, 0] != 0)[0]
        y_nonzero = np.where(coords[:, 1] != 0)[0]
        
        if len(x_nonzero) == 0 and len(y_nonzero) == 0:
            return 0  # All zeros, start from beginning
        elif len(x_nonzero) == 0:
            return y_nonzero[0]
        elif len(y_nonzero) == 0:
            return x_nonzero[0]
        else:
            return min(x_nonzero[0], y_nonzero[0])


def integrate_monitor_into_training(monitor: VAEMonitor, model, outputs: Dict, 
                                   targets: torch.Tensor, loss_dict: Dict,
                                   original_coords: torch.Tensor) -> Dict:
    """
    Helper function to integrate monitor into training loop.
    
    Args:
        monitor: VAEMonitor instance
        model: VAE model
        outputs: Model output dictionary
        targets: Target classes [B, T, 2]
        loss_dict: Loss dictionary from loss function
        original_coords: Original coordinates [B, T, 2]
        
    Returns:
        Dict with all monitoring metrics
    """
    metrics = {}
    
    # KL analysis
    kl_metrics = monitor.compute_kl_per_dimension(outputs['mu'], outputs['log_sigma'])
    metrics.update({f'monitor/kl_{k}': v for k, v in kl_metrics.items()})
    
    # Latent statistics
    latent_stats = monitor.compute_latent_statistics(
        outputs['mu'], outputs['log_sigma'], outputs.get('z')
    )
    metrics.update({f'monitor/latent_{k}': v for k, v in latent_stats.items()})
    
    # Reconstruction metrics
    recon_metrics = monitor.compute_reconstruction_metrics(
        outputs['logits'], targets, original_coords
    )
    metrics.update({f'monitor/recon_{k}': v for k, v in recon_metrics.items()})
    
    # Gradient flow is now handled automatically by wandb.watch()
    
    # Loss balance
    balance_metrics = monitor.compute_loss_balance(
        loss_dict['recon_loss'].item() if torch.is_tensor(loss_dict['recon_loss']) else loss_dict['recon_loss'],
        loss_dict['kl_loss'].item() if torch.is_tensor(loss_dict['kl_loss']) else loss_dict['kl_loss'],
        loss_dict.get('beta', 1.0)
    )
    metrics.update({f'monitor/loss_{k}': v for k, v in balance_metrics.items()})
    
    return metrics