import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
from pathlib import Path
from kernels_score_loss import KernelScoreLoss, get_available_kernels
from signature_models import TransformerModel, generate_ou_noise, add_noise
from utils import (
    setup_diffusion_schedule, generate_sinusoidal_data, pregenerate_ou_noise,
    get_cached_ou_noise, TrainingLogger, 
    create_training_plots, create_sample_comparison_plot, create_simple_sample_plot,
    print_model_summary, print_experiment_header, print_training_complete,
    test_strict_properness_debug, apply_gradient_clipping_with_scaling, check_model_health
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(123)

print(f"Using device: {device}")

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description='Baseline Kernel Scoring Diffusion with configurable kernels')

# Model architecture
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
parser.add_argument('--num_layers', type=int, default=8, help='Number of transformer layers')
parser.add_argument('--num_samples', type=int, default=8, help='Number of samples for kernel scoring')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

# Training parameters
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--num_epochs', type=int, default=500, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--lr_decay', type=float, default=0.995, help='Learning rate decay factor')
parser.add_argument('--warmup_epochs', type=int, default=50, help='Learning rate warmup epochs')
parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')

# Initialization parameters
parser.add_argument('--init_method', type=str, default='xavier_uniform', 
                   choices=['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'normal', 'orthogonal'],
                   help='Weight initialization method')
parser.add_argument('--init_gain', type=float, default=0.1, help='Initialization gain')

# Kernel parameters
parser.add_argument('--kernel_type', type=str, default='rbf',
                   choices=['rbf', 'exponential', 'lp_norm'],
                   help='Kernel type for scoring')
parser.add_argument('--lambda_param', type=float, default=0.5, help='Lambda parameter for kernel score')
parser.add_argument('--gamma', type=float, default=1.0, help='Gamma parameter for RBF/exponential kernels')
parser.add_argument('--p_norm', type=float, default=2.0, help='P parameter for Lp norm kernel')

# Data parameters
parser.add_argument('--N', type=int, default=200, help='Number of training samples')
parser.add_argument('--T', type=int, default=100, help='Number of time points per sample')

# Experiment parameters
parser.add_argument('--experiment_name', type=str, default='baseline_kernels', help='Experiment name for file naming')
parser.add_argument('--save_logs', action='store_true', help='Save detailed training logs to JSON')

args = parser.parse_args()

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Prepare kernel parameters based on kernel type
kernel_params = {}
if args.kernel_type in ['rbf', 'exponential']:
    kernel_params['gamma'] = args.gamma
elif args.kernel_type == 'lp_norm':
    kernel_params['p'] = args.p_norm

config = {
    'N': args.N,
    'T': args.T,
    'diffusion_steps': 100,
    'hidden_dim': args.hidden_dim,
    'num_layers': args.num_layers,
    'num_samples': args.num_samples,
    'batch_size': args.batch_size,
    'num_epochs': args.num_epochs,
    'learning_rate': args.learning_rate,
    'weight_decay': args.weight_decay,
    'lr_decay': args.lr_decay,
    'warmup_epochs': args.warmup_epochs,
    'min_lr': args.min_lr,
    'init_method': args.init_method,
    'init_gain': args.init_gain,
    'dropout': args.dropout,
    'kernel_type': args.kernel_type,
    'kernel_params': kernel_params,
    'lambda_param': args.lambda_param,
    'experiment_name': args.experiment_name,
    'save_logs': args.save_logs
}

# Create descriptive suffix for file naming
kernel_param_str = "_".join([f"{k}{v}" for k, v in kernel_params.items()])
config_suffix = f"_{args.experiment_name}_{args.kernel_type}_{kernel_param_str}_init{args.init_method}_gain{args.init_gain}_drop{args.dropout}_lr{args.learning_rate}_lambda{args.lambda_param}_samples{args.num_samples}"

print_experiment_header(f"Baseline Kernel Scoring Diffusion ({args.kernel_type.upper()})", config)

# Print kernel information
available_kernels = get_available_kernels()
if args.kernel_type in available_kernels:
    print(f"\n🔧 Kernel Details:")
    print(f"  Type: {args.kernel_type}")
    print(f"  Description: {available_kernels[args.kernel_type]['description']}")
    print(f"  Parameters: {kernel_params}")

# ============================================================================
# DATA GENERATION AND DIFFUSION SETUP
# ============================================================================

N, T = config['N'], config['T']
t, x = generate_sinusoidal_data(N, T, device)

diffusion_steps = config['diffusion_steps']
betas, alphas = setup_diffusion_schedule(diffusion_steps, device)

gp_sigma = 0.05

# ============================================================================
# KERNEL SCORING SETUP
# ============================================================================

# Initialize kernel score loss module
kernel_loss_fn = KernelScoreLoss(
    kernel_type=config['kernel_type'],
    lambda_param=config['lambda_param'],
    num_samples=config['num_samples'],
    kernel_params=config['kernel_params'],
    clamp_range=(-50.0, 50.0)
)

# ============================================================================
# MODEL AND TRAINING SETUP
# ============================================================================

model = TransformerModel(
    dim=1, 
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers'],
    max_i=diffusion_steps, 
    num_samples=config['num_samples'],
    dropout=config['dropout'],
    init_method=config['init_method'],
    init_gain=config['init_gain']
).to(device).double()

print_model_summary(model, f"{args.kernel_type.upper()} Kernel Scoring Transformer")

# Optimizer with improved settings
optim = torch.optim.Adam(
    model.parameters(), 
    lr=config['learning_rate'], 
    weight_decay=config['weight_decay'],
    betas=(0.9, 0.999),
    eps=1e-8
)

# Learning rate scheduler with warmup
def get_lr_lambda(epoch):
    """Learning rate schedule with warmup"""
    if epoch < config['warmup_epochs']:
        return epoch / config['warmup_epochs']
    else:
        decay_epochs = epoch - config['warmup_epochs']
        lr_factor = config['lr_decay'] ** decay_epochs
        min_lr_factor = config['min_lr'] / config['learning_rate']
        return max(lr_factor, min_lr_factor)

scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=get_lr_lambda)

# Training logger
logger = TrainingLogger()

# Pre-generate OU noise for efficiency
representative_t = torch.linspace(0, 1, T, dtype=torch.float64).view(1, -1, 1).to(device)
pregenerated_ou_noise = pregenerate_ou_noise(representative_t, config['num_samples'])

def get_kernel_loss(x_batch, t_batch):
    """
    Compute kernel-based loss following Algorithm 1 from method.tex exactly
    
    Same structure as signature scoring but with different kernel functions
    """
    batch_size = x_batch.shape[0]  # n in algorithm
    
    # Step 1: Sample t_i ~ U([0,1]) for i ∈ [n]
    continuous_times = torch.rand(batch_size, dtype=torch.float64, device=device)
    diffusion_steps_sampled = (continuous_times * (diffusion_steps - 1)).long()
    diffusion_steps_tensor = diffusion_steps_sampled.view(-1, 1, 1).expand_as(x_batch[...,:1]).to(x_batch).double()
    
    # Step 2: X_0^i are already sampled (x_batch) 
    # Step 3: Sample X_{t_i}^i using forward diffusion process
    x_noisy, _ = add_noise(x_batch, t_batch, diffusion_steps_tensor, alphas, gp_sigma)
    
    # Steps 4-5: Sample ξ_ij ~ OU process and generate samples
    all_predictions = []
    for j in range(config['num_samples']):  # j ∈ [m] - population size
        # Step 4: Sample ξ_ij ~ OU process for i ∈ [n], j ∈ [m]
        z_batch = []
        for i in range(batch_size):  # i ∈ [n] - batch size
            t_single = t_batch[i:i+1]  # [1, S, 1]
            # Sample OU noise ξ_ij for this (i,j) pair
            z_ij = get_cached_ou_noise(pregenerated_ou_noise, t_single.shape, 1, device)[:, 0]  # [1, S, D]
            z_batch.append(z_ij)
        z_batch = torch.cat(z_batch, dim=0)  # [batch_size, S, D]
        
        # Step 5: Use generator P_θ to produce samples X̃_0^(i)
        pred_batch = model(x_noisy, t_batch, diffusion_steps_tensor, z_batch)  # [batch_size, S, D]
        all_predictions.append(pred_batch)
    
    # Stack predictions: [m, n, S, D] - m samples for each of n batch items
    predictions = torch.stack(all_predictions, dim=0)
    
    # Step 6: Compute L_kernel
    total_loss = 0.0
    for i in range(batch_size):  # For each batch item i ∈ [n]
        # Get m samples for batch item i
        pred_samples_i = predictions[:, i]  # [m, S, D] - X̃_0^(i) samples
        target_i = x_batch[i]               # [S, D] - X_0^i target
        time_i = t_batch[i, :, 0]           # [S] - time points for kernel
        
        # Compute kernel score
        loss_i = kernel_loss_fn(pred_samples_i, target_i, time_i)
        total_loss += loss_i
    
    return total_loss / batch_size


def debug_kernel_components(x_batch, t_batch, model, kernel_loss_fn, device):
    """Debug function to examine kernel loss components in detail"""
    with torch.no_grad():
        # Take first sample for detailed analysis
        x_clean = x_batch[0]  # [S, D]
        t_single = t_batch[0]  # [S, 1]
        
        # Random diffusion step
        i_test = torch.randint(0, diffusion_steps, (1,)).item()
        i_tensor = torch.tensor([i_test], dtype=torch.float64).expand_as(x_clean[...,:1]).to(device)
        
        # Add noise
        x_noisy, _ = add_noise(x_clean.unsqueeze(0), t_single.unsqueeze(0), i_tensor.unsqueeze(0), alphas, gp_sigma)
        x_noisy = x_noisy.squeeze(0)  # [S, D]
        
        # Generate samples from model
        model_samples = []
        for _ in range(config['num_samples']):
            z = get_cached_ou_noise(pregenerated_ou_noise, t_single.unsqueeze(0).shape, 1, device)[:, 0]  # [1, S, D]
            pred_sample = model(x_noisy.unsqueeze(0), t_single.unsqueeze(0), i_tensor.unsqueeze(0), z)
            model_samples.append(pred_sample.squeeze(0))
        
        model_samples = torch.stack(model_samples, dim=0)  # [num_samples, S, D]
        
        # Compute detailed components
        try:
            score, target_term, cross_term = kernel_loss_fn.compute_components(
                model_samples, x_clean, t_single.squeeze(-1)
            )
            
            return {
                'total_score': score.item(),
                'target_term': target_term.item(), 
                'cross_term': cross_term.item(),
                'num_samples': model_samples.shape[0],
                'diffusion_step': i_test,
                'kernel_type': kernel_loss_fn.kernel_type
            }
        except Exception as e:
            return {'error': str(e)}


# ============================================================================
# TRAINING LOOP
# ============================================================================

print(f"Starting kernel scoring diffusion training with {config['kernel_type']} kernel and {config['init_method']} initialization...")

print(f"🎯 Kernel: {config['kernel_type']} with params {config['kernel_params']}")
print(f"🎯 Initialization: {config['init_method']} with gain {config['init_gain']}")
print(f"🎯 Dropout rate: {config['dropout']}")
print(f"🎯 Learning rate: {config['learning_rate']} with warmup over {config['warmup_epochs']} epochs")

batch_size = config['batch_size']
num_epochs = config['num_epochs']

for epoch in tqdm(range(num_epochs)):  # k = 1:M training steps
    # Steps 1-2: Sample data X_0^i ~ P_0 for i ∈ [n]
    batch_indices = torch.randperm(N)[:batch_size]
    x_batch = x[batch_indices]  # X_0^i samples
    t_batch = t[batch_indices]  # Time grids for each sample
    
    optim.zero_grad()
    # Steps 3-6: Forward diffusion, OU sampling, generation, and loss computation
    loss = get_kernel_loss(x_batch, t_batch)  # Implements Algorithm 1
    
    # Check for NaN/Inf in loss
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"❌ Invalid loss at epoch {epoch}: {loss.item()}")
        print("🏥 Running emergency model health check...")
        health = check_model_health(model)
        break
    
    loss.backward()
    
    # Enhanced gradient clipping with adaptive scaling
    grad_norm = apply_gradient_clipping_with_scaling(model, max_norm=1.0, adaptive=True)
    
    optim.step()
    scheduler.step()
    
    # Log metrics
    current_lr = scheduler.get_last_lr()[0]
    logger.log_step(loss.item(), grad_norm, current_lr)
    
    # Early stopping check
    if logger.should_early_stop():
        print(f"🛑 Early stopping at epoch {epoch} - no improvement for {logger.patience} epochs")
        break
    
    if epoch % 50 == 0:
        logger.print_progress(epoch, loss.item(), grad_norm, current_lr)
        
        # Check if loss is decreasing
        if epoch >= 100:
            recent_losses = logger.losses[-20:]
            early_losses = logger.losses[-100:-80]
            
            if len(early_losses) > 0:
                recent_avg = np.mean(recent_losses)
                early_avg = np.mean(early_losses)
                
                if recent_avg >= early_avg:
                    print(f"    ⚠️  WARNING: Loss not decreasing! Recent: {recent_avg:.6f}, Earlier: {early_avg:.6f}")
                else:
                    improvement = ((early_avg - recent_avg) / early_avg) * 100
                    print(f"    ✅ Loss decreasing: {improvement:.2f}% improvement")
        
        # Debug kernel components and model health every 200 epochs
        if epoch % 200 == 0 and epoch > 0:
            print(f"🔍 Debugging {args.kernel_type.upper()} Kernel Loss Components...")
            
            component_info = debug_kernel_components(x_batch, t_batch, model, kernel_loss_fn, device)
            
            if 'error' not in component_info:
                print(f"    Total score: {component_info['total_score']:.6f}")
                print(f"    Target term: {component_info['target_term']:.6f}")
                print(f"    Cross term:  {component_info['cross_term']:.6f}")
                print(f"    Kernel type: {component_info['kernel_type']}")
                print(f"    Diff step:   {component_info['diffusion_step']}")
                print(f"    Samples:     {component_info['num_samples']}")
            else:
                print(f"    Error: {component_info['error']}")
            
            # Check model health
            print("🏥 Model Health Check...")
            health = check_model_health(model)
            if health['has_nan_params'] or health['has_inf_params'] or health['has_nan_grads'] or health['has_inf_grads']:
                print("    ❌ Model health issues detected!")
            else:
                print(f"    ✅ Model healthy - {health['large_params']} large params, {health['large_grads']} large grads")
            print()
        
        # Test strict properness for this kernel every 100 epochs
        if epoch % 100 == 0 and epoch > 0:
            print(f"🔍 Testing Strict Properness for {args.kernel_type.upper()} kernel...")
            
            # Create a simple strict properness test
            try:
                from kernels_score_loss import test_kernel_strict_properness
                
                properness_result = test_kernel_strict_properness(
                    kernel_type=args.kernel_type,
                    kernel_params=kernel_params,
                    lambda_param=args.lambda_param,
                    num_trials=5  # Quick test during training
                )
                
                print(f"    Same dist score: {properness_result['same_mean']:.6f}")
                print(f"    Diff dist score: {properness_result['diff_mean']:.6f}")
                print(f"    Separation: {properness_result['separation']:.6f}")
                print(f"    Strict proper: {'✅ YES' if properness_result['is_proper'] else '❌ NO'}")
                
            except Exception as e:
                print(f"    Error in properness test: {e}")
            print()

print_training_complete(logger, f"Baseline {args.kernel_type.upper()} Kernel Scoring")

# ============================================================================
# SAMPLING FUNCTION (Same as signature scoring)
# ============================================================================

@torch.no_grad()
def sample_single_trajectory(t_grid, eta=0.0):
    """Generate ONE sample following Algorithm 2 with OU processes"""
    if len(t_grid.shape) == 3:
        t_single_grid = t_grid[0:1]  # [1, S, 1]
    else:
        t_single_grid = t_grid.unsqueeze(0)  # [1, S, 1]
    
    # Step 1: Initialize with OU process
    x_current = generate_ou_noise(t_single_grid, 1, batch_size=1, theta=2.0, sigma=0.8)[:, 0]  # [1, S, D]
    
    # Step 2: Backward diffusion
    for diff_step in reversed(range(0, diffusion_steps)):
        alpha_current = alphas[diff_step]
        
        if diff_step > 0:
            alpha_prev = alphas[diff_step - 1]
        else:
            alpha_prev = torch.tensor(1.0, device=device)
        
        t_single = t_single_grid
        i_single = torch.tensor([diff_step], dtype=torch.float64).expand_as(x_current[...,:1]).to(device)
        
        # Sample OU processes for Z and ξ
        Z_ou = generate_ou_noise(t_single, 1, batch_size=1, theta=2.0, sigma=0.8)[:, 0]
        xi_ou = generate_ou_noise(t_single, 1, batch_size=1, theta=2.0, sigma=0.8)[:, 0]
        
        # Model prediction
        pred_x0 = model(x_current, t_single, i_single, xi_ou)
        
        # DDIM updates
        D_t = (x_current - alpha_current.sqrt() * pred_x0) / (1 - alpha_current).sqrt().clamp(min=1e-8)
        
        # Optional covariance term
        if diff_step > 0 and eta > 0:
            covariance_factor = eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_current) * 
                (1 - alpha_current / alpha_prev)
            ).clamp(min=1e-8)
            covariance_noise = covariance_factor * Z_ou
        else:
            covariance_noise = torch.zeros_like(x_current)
        
        # Update
        x_current = (alpha_prev.sqrt() * pred_x0 + 
                    (1 - alpha_prev).sqrt() * D_t + 
                    covariance_noise)
    
    return x_current


@torch.no_grad()
def sample_kernel_diffusion(t_grid, num_samples=20, eta=0.0):
    """Generate multiple samples by calling Algorithm 2 multiple times"""
    all_samples = []
    
    for sample_idx in range(num_samples):
        single_sample = sample_single_trajectory(t_grid, eta=eta)
        all_samples.append(single_sample)
    
    return torch.cat(all_samples, dim=0)

# ============================================================================
# EVALUATION AND VISUALIZATION
# ============================================================================

print("\nGenerating samples and creating visualizations...")

# Generate samples
t_grid = torch.linspace(0, 1, T, dtype=torch.float64).view(1, -1, 1).to(device)
num_generated_samples = 20
samples = sample_kernel_diffusion(t_grid, num_generated_samples)

# Create visualizations with descriptive suffix
training_plot_path = create_training_plots(logger, config_suffix)
comparison_plot_path = create_sample_comparison_plot(samples, x, t_grid, t, config_suffix)

# Create title with configuration info
title = f"Generated Samples - {args.experiment_name}\nKernel: {args.kernel_type}, λ={args.lambda_param}, Samples={args.num_samples}"
samples_plot_path = create_simple_sample_plot(
    samples, t_grid, title=title, config_suffix=config_suffix
)

# Save training logs if requested
if config['save_logs']:
    log_data = {
        'config': config,
        'training_metrics': {
            'losses': logger.losses,
            'gradient_norms': logger.gradient_norms,
            'learning_rates': logger.learning_rates
        },
        'final_summary': logger.get_summary()
    }
    
    log_filename = f"training_log{config_suffix}.json"
    with open(log_filename, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"📄 Training log saved: {log_filename}")

print("✅ Analysis complete!")
print(f"\nFiles generated:")
print(f"- {training_plot_path}")
print(f"- {comparison_plot_path}")  
print(f"- {samples_plot_path}")

# Print final summary with configuration
summary = logger.get_summary()
print(f"\n🏆 EXPERIMENT SUMMARY: {args.experiment_name}")
print(f"Kernel: {args.kernel_type} with params {config['kernel_params']}")
print(f"Initialization: {args.init_method} init, gain={args.init_gain}, dropout={args.dropout}")
print(f"Final Loss: {summary['final_loss']:.6f}")
print(f"Best Loss: {summary['best_loss']:.6f}")
print(f"Final Gradient Norm: {summary['final_grad_norm']:.6f}")
print(f"Total Epochs: {summary['total_epochs']}")
print(f"Early Stop: {'Yes' if summary['epochs_without_improvement'] >= 100 else 'No'}")
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
