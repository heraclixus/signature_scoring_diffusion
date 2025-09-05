"""
Utilities for Signature Scoring Diffusion

This module contains visualization, logging, and helper functions used across
the signature scoring diffusion experiments.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


def get_betas(steps: int, beta_start: float = 1e-4, beta_end: float = 0.2, device: torch.device = None) -> torch.Tensor:
    """
    Generate beta schedule for diffusion process.
    
    Args:
        steps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value
        device: Target device
        
    Returns:
        Beta values for each diffusion step
    """
    if device is None:
        device = torch.device('cpu')
        
    diffusion_ind = torch.linspace(0, 1, steps).to(device)
    return beta_start * (1 - diffusion_ind) + beta_end * diffusion_ind


def setup_diffusion_schedule(steps: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Setup complete diffusion schedule.
    
    Args:
        steps: Number of diffusion steps
        device: Target device
        
    Returns:
        Tuple of (betas, alphas) where alphas are cumulative products
    """
    betas = get_betas(steps, device=device)
    alphas = torch.cumprod(1 - betas, dim=0)
    return betas, alphas


def generate_sinusoidal_data(N: int, T: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sinusoidal training data.
    
    Args:
        N: Number of samples
        T: Number of time points per sample
        device: Target device
        
    Returns:
        Tuple of (t, x) where t are time points and x are function values
    """
    t = torch.rand(N, T, 1, dtype=torch.float64).sort(1)[0].to(device)
    x = torch.sin(10 * t + 2 * np.pi * torch.rand(N, 1, 1, dtype=torch.float64).to(t))
    return t, x


def pregenerate_ou_noise(representative_t: torch.Tensor, num_samples: int, 
                        cache_size: int = 1000, theta: float = 2.0, sigma: float = 0.8) -> List[torch.Tensor]:
    """
    Pre-generate OU noise vectors for training efficiency.
    
    Args:
        representative_t: Representative time grid for generation
        num_samples: Number of samples per noise vector
        cache_size: Number of noise vectors to pre-generate
        theta: OU mean reversion parameter
        sigma: OU volatility parameter
        
    Returns:
        List of pre-generated OU noise vectors
    """
    from signature_models import generate_ou_noise
    
    print(f"Pre-generating {cache_size} OU noise vectors for training efficiency...")
    pregenerated_noise = []
    
    for _ in range(cache_size):
        z = generate_ou_noise(representative_t, num_samples, batch_size=1, theta=theta, sigma=sigma)
        pregenerated_noise.append(z)
    
    print(f"✅ Pre-generated {cache_size} OU noise vectors")
    return pregenerated_noise


def get_cached_ou_noise(pregenerated_noise: List[torch.Tensor], target_shape: Tuple[int, ...], 
                       num_samples: int, device: torch.device, theta: float = 2.0, sigma: float = 0.8) -> torch.Tensor:
    """
    Get OU noise from cache or generate new if shape mismatch.
    
    Args:
        pregenerated_noise: List of cached noise vectors
        target_shape: Target time grid shape [B, S, 1]
        num_samples: Number of samples needed
        device: Target device
        theta: OU mean reversion parameter
        sigma: OU volatility parameter
        
    Returns:
        OU noise tensor [B, num_samples, S, D]
    """
    from signature_models import generate_ou_noise
    
    if pregenerated_noise:
        # Try to use cached noise
        noise_idx = torch.randint(0, len(pregenerated_noise), (1,)).item()
        z_cached = pregenerated_noise[noise_idx]
        
        # Check if shapes match
        if z_cached.shape[2] == target_shape[1]:  # S dimension matches
            return z_cached
    
    # Generate new noise if no cache or shape mismatch
    t_grid = torch.linspace(0, 1, target_shape[1], dtype=torch.float64).view(1, -1, 1).to(device)
    return generate_ou_noise(t_grid, num_samples, batch_size=1, theta=theta, sigma=sigma)


class TrainingLogger:
    """Enhanced logger for training metrics and progress with early stopping"""
    
    def __init__(self, patience: int = 100):
        self.losses = []
        self.gradient_norms = []
        self.learning_rates = []
        self.patience = patience
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def log_step(self, loss: float, grad_norm: float, lr: float):
        """Log metrics for a training step"""
        self.losses.append(loss)
        self.gradient_norms.append(grad_norm)
        self.learning_rates.append(lr)
        
        # Early stopping logic
        if loss < self.best_loss:
            self.best_loss = loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            
    def should_early_stop(self) -> bool:
        """Check if training should stop early"""
        return self.epochs_without_improvement >= self.patience
        
    def print_progress(self, epoch: int, loss: float, grad_norm: float, lr: float):
        """Print training progress with early stopping info"""
        print(f"Epoch {epoch}: Loss = {loss:.6f}, Grad Norm = {grad_norm:.6f}, LR = {lr:.2e}")
        if self.epochs_without_improvement > 0:
            print(f"    No improvement for {self.epochs_without_improvement} epochs (patience: {self.patience})")
        
    def get_summary(self) -> dict:
        """Get training summary statistics"""
        return {
            'final_loss': self.losses[-1] if self.losses else 0,
            'final_grad_norm': self.gradient_norms[-1] if self.gradient_norms else 0,
            'total_epochs': len(self.losses),
            'best_loss': self.best_loss,
            'epochs_without_improvement': self.epochs_without_improvement
        }


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """
    Compute total gradient norm for a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total gradient norm
    """
    total_grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item() ** 2
    return total_grad_norm ** 0.5


def apply_gradient_clipping_with_scaling(model: torch.nn.Module, max_norm: float = 1.0, 
                                       adaptive: bool = True) -> float:
    """
    Apply adaptive gradient clipping with optional scaling.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        adaptive: Whether to use adaptive clipping based on gradient history
        
    Returns:
        Gradient norm before clipping
    """
    # Compute gradient norm before clipping
    grad_norm = compute_gradient_norm(model)
    
    if adaptive and grad_norm > max_norm * 2:
        # If gradient is very large, use more aggressive clipping
        effective_max_norm = max_norm * 0.5
    else:
        effective_max_norm = max_norm
    
    # Apply clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=effective_max_norm)
    
    return grad_norm


def check_model_health(model: torch.nn.Module) -> dict:
    """
    Check model health indicators for debugging training instability.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with health metrics
    """
    health_info = {
        'has_nan_params': False,
        'has_inf_params': False,
        'has_nan_grads': False,
        'has_inf_grads': False,
        'large_params': 0,
        'large_grads': 0,
        'total_params': 0
    }
    
    for name, param in model.named_parameters():
        health_info['total_params'] += param.numel()
        
        # Check parameters
        if torch.isnan(param).any():
            health_info['has_nan_params'] = True
            print(f"    ⚠️ NaN in parameter: {name}")
            
        if torch.isinf(param).any():
            health_info['has_inf_params'] = True
            print(f"    ⚠️ Inf in parameter: {name}")
            
        if (param.abs() > 10).any():
            health_info['large_params'] += 1
            
        # Check gradients
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                health_info['has_nan_grads'] = True
                print(f"    ⚠️ NaN in gradient: {name}")
                
            if torch.isinf(param.grad).any():
                health_info['has_inf_grads'] = True
                print(f"    ⚠️ Inf in gradient: {name}")
                
            if (param.grad.abs() > 10).any():
                health_info['large_grads'] += 1
    
    return health_info


def create_training_plots(logger: TrainingLogger, config_suffix: str = "", save_dir: str = ".") -> str:
    """
    Create comprehensive training visualization plots.
    
    Args:
        logger: Training logger with metrics
        config_suffix: Configuration suffix for filename
        save_dir: Directory to save plots
        
    Returns:
        Path to saved plot file
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Training loss
    ax = axes[0]
    ax.plot(logger.losses, color='C0', linewidth=2, label='Signature Loss')
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gradient norms and learning rate
    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(logger.gradient_norms, color='C1', linewidth=2, label='Grad Norm')
    ax2.plot(logger.learning_rates, color='C3', linewidth=2, linestyle='--', label='Learning Rate')
    ax.set_title('Training Dynamics')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm', color='C1')
    ax2.set_ylabel('Learning Rate', color='C3')
    ax.set_yscale('log')
    ax2.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save plot
    filename = f'training_metrics{config_suffix}.png'
    filepath = Path(save_dir) / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(filepath)


def create_sample_comparison_plot(samples: torch.Tensor, ground_truth: torch.Tensor, 
                                t_grid: torch.Tensor, t_gt: torch.Tensor, 
                                config_suffix: str = "", save_dir: str = ".") -> str:
    """
    Create side-by-side comparison of generated samples vs ground truth.
    
    Args:
        samples: Generated samples [N, S, D]
        ground_truth: Ground truth data [N, S, D] 
        t_grid: Time grid for samples [S] or [1, S, 1]
        t_gt: Time grid for ground truth [N, S, 1]
        config_suffix: Configuration suffix for filename
        save_dir: Directory to save plots
        
    Returns:
        Path to saved plot file
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Convert to numpy
    samples_np = samples.detach().cpu().numpy()
    ground_truth_np = ground_truth.detach().cpu().numpy()
    
    if len(t_grid.shape) == 3:
        t_grid_np = t_grid.squeeze().detach().cpu().numpy()
    else:
        t_grid_np = t_grid.detach().cpu().numpy()
    t_gt_np = t_gt.detach().cpu().numpy()
    
    # Plot 1: Generated samples
    ax = axes[0, 0]
    for i in range(min(10, len(samples_np))):
        ax.plot(t_grid_np, samples_np[i].squeeze(), color='C0', alpha=0.7, linewidth=1)
    ax.set_title('Generated Samples (First 10)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Ground truth
    ax = axes[0, 1]
    for i in range(min(10, len(ground_truth_np))):
        ax.plot(t_gt_np[i, :, 0], ground_truth_np[i, :, 0], color='C1', alpha=0.7, linewidth=1)
    ax.set_title('Ground Truth (First 10)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Direct comparison
    ax = axes[1, 0]
    for i in range(min(3, len(samples_np))):
        ax.plot(t_grid_np, samples_np[i].squeeze(), color='C0', alpha=0.8, 
               linewidth=2, label='Generated' if i == 0 else '')
    for i in range(min(3, len(ground_truth_np))):
        ax.plot(t_gt_np[i, :, 0], ground_truth_np[i, :, 0], color='C1', alpha=0.8, 
               linewidth=2, linestyle='--', label='Ground Truth' if i == 0 else '')
    ax.set_title('Direct Comparison (3 samples each)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Distribution comparison
    ax = axes[1, 1]
    ax.hist(samples_np.flatten(), bins=50, alpha=0.6, label='Generated', color='C0', density=True)
    ax.hist(ground_truth_np.flatten(), bins=50, alpha=0.6, label='Ground Truth', color='C1', density=True)
    ax.set_title('Value Distribution Comparison')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'sample_comparison{config_suffix}.png'
    filepath = Path(save_dir) / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(filepath)


def create_simple_sample_plot(samples: torch.Tensor, t_grid: torch.Tensor, 
                            title: str = "Generated Samples", config_suffix: str = "", 
                            save_dir: str = ".") -> str:
    """
    Create simple visualization of generated samples.
    
    Args:
        samples: Generated samples [N, S, D]
        t_grid: Time grid [S] or [1, S, 1]
        title: Plot title
        config_suffix: Configuration suffix for filename
        save_dir: Directory to save plots
        
    Returns:
        Path to saved plot file
    """
    plt.figure(figsize=(10, 6))
    
    samples_np = samples.detach().cpu().numpy()
    if len(t_grid.shape) == 3:
        t_grid_np = t_grid.squeeze().detach().cpu().numpy()
    else:
        t_grid_np = t_grid.detach().cpu().numpy()
    
    for i in range(min(10, len(samples_np))):
        alpha = 1.0 / (i + 1) if i < 10 else 0.1
        plt.plot(t_grid_np, samples_np[i].squeeze(), color='C0', alpha=alpha)
    
    plt.title(title)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    filename = f'samples{config_suffix}.png'
    filepath = Path(save_dir) / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(filepath)


def print_model_summary(model: torch.nn.Module, model_name: str = "Model"):
    """Print model parameter summary with initialization info"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 {model_name} Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print initialization info if available
    if hasattr(model, 'init_method'):
        print(f"Initialization: {model.init_method} (gain: {model.init_gain})")
    if hasattr(model, 'dropout'):
        print(f"Dropout rate: {model.dropout}")


def create_model_variants_for_testing():
    """
    Create different model variants for testing initialization methods.
    
    Returns:
        Dict of model configurations for stability testing
    """
    variants = {
        'xavier_uniform_conservative': {
            'init_method': 'xavier_uniform',
            'init_gain': 0.05,
            'dropout': 0.15
        },
        'xavier_normal_conservative': {
            'init_method': 'xavier_normal', 
            'init_gain': 0.05,
            'dropout': 0.15
        },
        'xavier_uniform_standard': {
            'init_method': 'xavier_uniform',
            'init_gain': 0.1,
            'dropout': 0.1
        },
        'xavier_normal_standard': {
            'init_method': 'xavier_normal',
            'init_gain': 0.1, 
            'dropout': 0.1
        },
        'kaiming_uniform': {
            'init_method': 'kaiming_uniform',
            'init_gain': 0.1,
            'dropout': 0.1
        },
        'orthogonal': {
            'init_method': 'orthogonal',
            'init_gain': 0.1,
            'dropout': 0.1
        }
    }
    
    return variants


def print_experiment_header(experiment_name: str, config: dict):
    """Print experiment header with configuration"""
    print("=" * 80)
    print(f"🚀 {experiment_name}")
    print("=" * 80)
    
    print("\n📋 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()


def print_training_complete(logger: TrainingLogger, experiment_name: str):
    """Print training completion summary"""
    summary = logger.get_summary()
    
    print("\n" + "="*50)
    print(f"✅ {experiment_name} Training Complete!")
    print("="*50)
    print(f"Final Loss: {summary['final_loss']:.6f}")
    print(f"Final Gradient Norm: {summary['final_grad_norm']:.6f}")
    print(f"Total Epochs: {summary['total_epochs']}")
    print("="*50)


def test_strict_properness_debug(model, x_clean, t_single, i_single, signature_loss_fn, 
                                device, num_test_samples=8, theta=2.0, sigma=0.8):
    """
    Debug function to test strict properness during training.
    
    Tests whether the signature score is minimized when prediction distribution
    equals the true distribution.
    
    Args:
        model: The signature scoring model
        x_clean: Clean target sample [S, D]
        t_single: Time points [S, 1] 
        i_single: Diffusion step [S, 1]
        signature_loss_fn: Signature loss function
        device: Torch device
        num_test_samples: Number of samples for testing
        
    Returns:
        dict: Debug information about strict properness
    """
    from signature_models import generate_ou_noise, add_noise
    
    with torch.no_grad():
        # Add noise to get x_noisy
        x_clean_batch = x_clean.unsqueeze(0)  # [1, S, D]
        t_batch = t_single.unsqueeze(0)       # [1, S, 1]
        i_batch = i_single.unsqueeze(0)       # [1, S, 1]
        
        x_noisy, _ = add_noise(x_clean_batch, t_batch, i_batch, 
                              torch.cumprod(1 - torch.linspace(1e-4, 0.2, 100).to(device), dim=0),
                              gp_sigma=0.05)
        x_noisy = x_noisy.squeeze(0)  # [S, D]
        
        # Test 1: Generate samples from model (predicted distribution)
        model_samples = []
        for _ in range(num_test_samples):
            z = generate_ou_noise(t_batch, 1, batch_size=1, theta=theta, sigma=sigma)[:, 0]  # [1, S, D]
            pred_sample = model(x_noisy.unsqueeze(0), t_batch, i_batch, z)  # [1, S, D]
            model_samples.append(pred_sample.squeeze(0))  # [S, D]
        
        model_samples = torch.stack(model_samples, dim=0)  # [num_test_samples, S, D]
        
        # Test 2: Generate samples from "true" distribution (add different noise to same clean sample)
        true_samples = []
        for _ in range(num_test_samples):
            # Generate different noise instances
            x_noisy_alt, _ = add_noise(x_clean_batch, t_batch, i_batch,
                                     torch.cumprod(1 - torch.linspace(1e-4, 0.2, 100).to(device), dim=0),
                                     gp_sigma=0.05)
            true_samples.append(x_clean)  # Use original clean sample as "true" sample
        
        true_samples = torch.stack(true_samples, dim=0)  # [num_test_samples, S, D]
        
        # Compute signature scores
        time_points = t_single.squeeze(-1)  # [S]
        
        # Score 1: Model samples vs target (what we're training)
        score_model = signature_loss_fn(model_samples, x_clean, time_points)
        
        # Score 2: True samples vs target (should be lower if strictly proper)
        score_true = signature_loss_fn(true_samples, x_clean, time_points)
        
        # Get detailed components if available
        try:
            model_score_detail, model_target_term, model_cross_term = signature_loss_fn.compute_components(
                model_samples, x_clean, time_points
            )
            true_score_detail, true_target_term, true_cross_term = signature_loss_fn.compute_components(
                true_samples, x_clean, time_points
            )
            
            has_components = True
        except:
            has_components = False
        
        # Strict properness check: score_true should be <= score_model
        is_strictly_proper = score_true.item() <= score_model.item()
        separation = score_model.item() - score_true.item()
        
        result = {
            'score_model': score_model.item(),
            'score_true': score_true.item(),
            'separation': separation,
            'is_strictly_proper': is_strictly_proper,
            'relative_difference': separation / abs(score_true.item()) if score_true.item() != 0 else float('inf')
        }
        
        if has_components:
            result.update({
                'model_target_term': model_target_term.item(),
                'model_cross_term': model_cross_term.item(),
                'true_target_term': true_target_term.item(),
                'true_cross_term': true_cross_term.item()
            })
        
        return result
