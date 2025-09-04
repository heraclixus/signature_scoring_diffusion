import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from signature_score_loss import SignatureScoreLoss
from signature_models import TransformerModel, generate_ou_noise, add_noise
from utils import (
    setup_diffusion_schedule, generate_sinusoidal_data, pregenerate_ou_noise,
    get_cached_ou_noise, TrainingLogger, compute_gradient_norm, 
    create_training_plots, create_sample_comparison_plot, create_simple_sample_plot,
    print_model_summary, print_experiment_header, print_training_complete,
    test_strict_properness_debug
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(123)

print(f"Using device: {device}")

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

config = {
    'N': 200,
    'T': 100,  # Match baseline
    'diffusion_steps': 100,
    'hidden_dim': 64,
    'num_samples': 8,
    'batch_size': 16,  # Efficient batch processing like baseline
    'num_epochs': 500,  # Match baseline epoch count
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'lr_decay': 0.98
}

print_experiment_header("Baseline Signature Scoring Diffusion", config)

# ============================================================================
# DATA GENERATION AND DIFFUSION SETUP
# ============================================================================

N, T = config['N'], config['T']
t, x = generate_sinusoidal_data(N, T, device)

diffusion_steps = config['diffusion_steps']
betas, alphas = setup_diffusion_schedule(diffusion_steps, device)

gp_sigma = 0.05


# ============================================================================
# SIGNATURE SCORING SETUP
# ============================================================================

# Initialize signature score loss module
signature_loss_fn = SignatureScoreLoss(
    lambda_param=0.5,
    num_samples=config['num_samples'],
    dyadic_order=1,
    clamp_range=(-50.0, 50.0)
)

# ============================================================================
# MODEL AND TRAINING SETUP
# ============================================================================

model = TransformerModel(
    dim=1, 
    hidden_dim=config['hidden_dim'], 
    max_i=diffusion_steps, 
    num_samples=config['num_samples']
).to(device).double()

print_model_summary(model, "Signature Scoring Transformer")

# Optimizer and scheduler
optim = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=config['lr_decay'])

# Training logger
logger = TrainingLogger()

# Pre-generate OU noise for efficiency
representative_t = torch.linspace(0, 1, T, dtype=torch.float64).view(1, -1, 1).to(device)
pregenerated_ou_noise = pregenerate_ou_noise(representative_t, config['num_samples'])

def get_signature_loss(x_batch, t_batch):
    """
    Compute signature-based loss following baseline convention - process entire batch efficiently
    """
    batch_size = x_batch.shape[0]
    
    # Sample random diffusion steps (same as baseline)
    i = torch.randint(0, diffusion_steps, size=(batch_size,), dtype=torch.int64)
    i = i.view(-1, 1, 1).expand_as(x_batch[...,:1]).to(x_batch).double()
    
    # Add noise to entire batch (same as baseline)
    x_noisy, _ = add_noise(x_batch, t_batch, i, alphas, gp_sigma)
    
    # Generate multiple predictions for each batch item efficiently
    all_predictions = []
    for sample_idx in range(config['num_samples']):
        # Generate OU noise for entire batch
        z_batch = []
        for b in range(batch_size):
            t_single = t_batch[b:b+1]  # [1, S, 1]
            z_single = get_cached_ou_noise(pregenerated_ou_noise, t_single.shape, 1, device)[:, 0]  # [1, S, D]
            z_batch.append(z_single)
        z_batch = torch.cat(z_batch, dim=0)  # [batch_size, S, D]
        
        # Single forward pass for entire batch with this Z
        pred_batch = model(x_noisy, t_batch, i, z_batch)  # [batch_size, S, D]
        all_predictions.append(pred_batch)
    
    # Stack predictions: [num_samples, batch_size, S, D]
    predictions = torch.stack(all_predictions, dim=0)
    
    # Compute signature loss for each batch item
    total_loss = 0.0
    for b in range(batch_size):
        pred_samples_b = predictions[:, b]  # [num_samples, S, D]
        target_b = x_batch[b]               # [S, D]
        time_b = t_batch[b, :, 0]           # [S] - remove last dimension for signature loss
        
        loss_b = signature_loss_fn(pred_samples_b, target_b, time_b)
        total_loss += loss_b
    
    return total_loss / batch_size


def debug_signature_components(x_batch, t_batch, model, signature_loss_fn, device):
    """Debug function to examine signature loss components in detail"""
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
            score, target_term, cross_term = signature_loss_fn.compute_components(
                model_samples, x_clean, t_single.squeeze(-1)
            )
            
            return {
                'total_score': score.item(),
                'target_term': target_term.item(), 
                'cross_term': cross_term.item(),
                'num_samples': model_samples.shape[0],
                'diffusion_step': i_test
            }
        except Exception as e:
            return {'error': str(e)}


# ============================================================================
# TRAINING LOOP
# ============================================================================

print("Starting signature scoring diffusion training...")

# Better initialization
def init_weights(m):
    if hasattr(m, 'weight') and m.weight is not None:
        torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
    if hasattr(m, 'bias') and m.bias is not None:
        torch.nn.init.zeros_(m.bias)

model.apply(init_weights)

batch_size = config['batch_size']
num_epochs = config['num_epochs']

for epoch in tqdm(range(num_epochs)):
    # Sample random batch (like baseline)
    batch_indices = torch.randperm(N)[:batch_size]
    x_batch = x[batch_indices]
    t_batch = t[batch_indices]
    
    optim.zero_grad()
    loss = get_signature_loss(x_batch, t_batch)  # Now processes batch efficiently like baseline
    loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Compute gradient norm
    grad_norm = compute_gradient_norm(model)
    
    optim.step()
    scheduler.step()
    
    # Log metrics
    current_lr = scheduler.get_last_lr()[0]
    logger.log_step(loss.item(), grad_norm, current_lr)
    
    if epoch % 50 == 0:
        logger.print_progress(epoch, loss.item(), grad_norm, current_lr)
        
        # Check if loss is decreasing
        if epoch >= 100:
            recent_losses = logger.losses[-20:]  # Last 20 epochs
            early_losses = logger.losses[-100:-80]  # 80-100 epochs ago
            
            if len(early_losses) > 0:
                recent_avg = np.mean(recent_losses)
                early_avg = np.mean(early_losses)
                
                if recent_avg >= early_avg:
                    print(f"    ⚠️  WARNING: Loss not decreasing! Recent: {recent_avg:.6f}, Earlier: {early_avg:.6f}")
                else:
                    improvement = ((early_avg - recent_avg) / early_avg) * 100
                    print(f"    ✅ Loss decreasing: {improvement:.2f}% improvement")
        
        # Debug signature components every 200 epochs
        if epoch % 200 == 0 and epoch > 0:
            print("🔍 Debugging Signature Loss Components...")
            
            component_info = debug_signature_components(x_batch, t_batch, model, signature_loss_fn, device)
            
            if 'error' not in component_info:
                print(f"    Total score: {component_info['total_score']:.6f}")
                print(f"    Target term: {component_info['target_term']:.6f}")
                print(f"    Cross term:  {component_info['cross_term']:.6f}")
                print(f"    Diff step:   {component_info['diffusion_step']}")
                print(f"    Samples:     {component_info['num_samples']}")
            else:
                print(f"    Error: {component_info['error']}")
            print()
        
        # Debug strict properness every 100 epochs
        if epoch % 100 == 0 and epoch > 0:
            print("🔍 Testing Strict Properness...")
            
            # Use a random sample for testing
            test_idx = torch.randint(0, N, (1,)).item()
            x_test = x[test_idx]  # [S, D]
            t_test = t[test_idx]  # [S, 1]
            i_test = torch.randint(0, diffusion_steps, (1,)).item()
            i_test_tensor = torch.tensor([i_test], dtype=torch.float64).expand_as(x_test[...,:1]).to(device)
            
            debug_info = test_strict_properness_debug(
                model, x_test, t_test, i_test_tensor, signature_loss_fn, device
            )
            
            print(f"    Model score: {debug_info['score_model']:.6f}")
            print(f"    True score:  {debug_info['score_true']:.6f}")
            print(f"    Separation:  {debug_info['separation']:.6f}")
            print(f"    Strict proper: {'✅ YES' if debug_info['is_strictly_proper'] else '❌ NO'}")
            print(f"    Rel. diff: {debug_info['relative_difference']:.3f}")
            
            # Show detailed components if available
            if 'model_target_term' in debug_info:
                print(f"    Components breakdown:")
                print(f"      Model - Target: {debug_info['model_target_term']:.6f}, Cross: {debug_info['model_cross_term']:.6f}")
                print(f"      True  - Target: {debug_info['true_target_term']:.6f}, Cross: {debug_info['true_cross_term']:.6f}")
            print()

print_training_complete(logger, "Baseline Signature Scoring")

# ============================================================================
# SAMPLING FUNCTION
# ============================================================================

@torch.no_grad()
def sample_signature(t_grid, num_samples=20):
    """
    Generate samples using DDIM sampling with clean X_0 predictions
    Similar to baseline but uses model's clean predictions
    """
    # Ensure t_grid has the right shape for a single batch element
    if len(t_grid.shape) == 3:
        t_single_grid = t_grid[0:1]  # [1, S, 1]
    else:
        t_single_grid = t_grid.unsqueeze(0)  # [1, S, 1]
    
    from signature_models import get_gp_covariance
    cov = get_gp_covariance(t_single_grid, gp_sigma)
    L = torch.linalg.cholesky(cov)
    
    # Start with noise
    x = L @ torch.randn(num_samples, t_single_grid.shape[1], 1, dtype=torch.float64, device=device)
    
    for diff_step in reversed(range(0, diffusion_steps)):
        alpha = alphas[diff_step]
        beta = betas[diff_step]
        
        # Generate samples for each trajectory
        all_samples = []
        for sample_idx in range(num_samples):
            x_single = x[sample_idx:sample_idx+1]  # [1, S, 1]
            t_single = t_single_grid  # [1, S, 1] 
            i_single = torch.tensor([diff_step], dtype=torch.float64).expand_as(x_single[...,:1]).to(device)
            
            # Generate single OU noise for this sample
            z_single = generate_ou_noise(t_single, 1, batch_size=1, theta=2.0, sigma=0.8)[:, 0]  # [1, S, D]
            
            # Get clean X_0 prediction from model with OU noise input
            pred_x0 = model(x_single, t_single, i_single, z_single)  # [1, S, D]
            
            # DDIM update using predicted clean sample
            # x_t = sqrt(alpha) * x_0 + sqrt(1-alpha) * noise
            # So: noise = (x_t - sqrt(alpha) * x_0) / sqrt(1-alpha)
            pred_noise = (x_single - alpha.sqrt() * pred_x0) / (1 - alpha).sqrt().clamp(min=1e-8)
            
            # Standard DDIM update (same as baseline)
            x_single = (x_single - beta * pred_noise / (1 - alpha).sqrt().clamp(min=1e-8)) / (1 - beta).sqrt().clamp(min=1e-8)
            
            if diff_step > 0:
                z = L @ torch.randn_like(x_single)
                x_single = x_single + beta.sqrt() * z
                
            all_samples.append(x_single)
        
        x = torch.cat(all_samples, dim=0)
    
    return x

# ============================================================================
# EVALUATION AND VISUALIZATION
# ============================================================================

print("\nGenerating samples and creating visualizations...")

# Generate samples
t_grid = torch.linspace(0, 1, T, dtype=torch.float64).view(1, -1, 1).to(device)
num_generated_samples = 20
samples = sample_signature(t_grid, num_generated_samples)

# Create visualizations using utility functions
config_suffix = "_baseline_sigscore"

# Training metrics plot
training_plot_path = create_training_plots(logger, config_suffix)

# Sample comparison plot
comparison_plot_path = create_sample_comparison_plot(samples, x, t_grid, t, config_suffix)

# Simple samples plot
samples_plot_path = create_simple_sample_plot(
    samples, t_grid, 
    title="10 Generated Samples - Baseline Signature Scoring Diffusion", 
    config_suffix=config_suffix
)

print("✅ Analysis complete!")
print(f"\nFiles generated:")
print(f"- {training_plot_path}")
print(f"- {comparison_plot_path}")  
print(f"- {samples_plot_path}")

# Print final summary
summary = logger.get_summary()
print(f"\n🏆 BASELINE SIGNATURE SCORING SUMMARY:")
print(f"Final Loss: {summary['final_loss']:.6f}")
print(f"Final Gradient Norm: {summary['final_grad_norm']:.6f}")
print(f"Total Epochs: {summary['total_epochs']}")
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
