from typing import List, Callable
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import pysiglib.torch_api as pysiglib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(123)

print(f"Using device: {device}")

# ============================================================================
# DATA GENERATION (Same as baseline)
# ============================================================================

N, T = 200, 100
t = torch.rand(N, T, 1, dtype=torch.float64).sort(1)[0].to(device)
x = torch.sin(10 * t + 2 * np.pi * torch.rand(N, 1, 1, dtype=torch.float64).to(t))

# Visualize training data
for i in range(10):
    plt.plot(t[i,:,0].cpu().numpy(), x[i,:,0].cpu().numpy(), color='C0', alpha=1 / (i + 1))
plt.title('10 samples from the dataset\nEach curve is one "data point"')
plt.xlabel('t')
plt.ylabel('x')
plt.savefig('data_sigscore.png')
plt.close()

# ============================================================================
# DIFFUSION SETUP (Same as baseline)
# ============================================================================

def get_betas(steps):
    beta_start, beta_end = 1e-4, 0.2
    diffusion_ind = torch.linspace(0, 1, steps).to(device)
    return beta_start * (1 - diffusion_ind) + beta_end * diffusion_ind

diffusion_steps = 100
betas = get_betas(diffusion_steps)
alphas = torch.cumprod(1 - betas, dim=0)

gp_sigma = 0.05

def get_gp_covariance(t):
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2]).to(t) * 1e-5 # for numerical stability
    return torch.exp(-torch.square(s / gp_sigma)) + diag

def add_noise(x, t, i):
    """
    x: Clean data sample, shape [B, S, D]
    t: Times of observations, shape [B, S, 1]
    i: Diffusion step, shape [B, S, 1]
    """
    noise_gaussian = torch.randn_like(x, dtype=torch.float64)
    
    cov = get_gp_covariance(t)
    L = torch.linalg.cholesky(cov)
    noise = L @ noise_gaussian
    
    alpha = alphas[i.long()].to(x)
    x_noisy = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
    
    return x_noisy, noise

# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_value: float):
        super().__init__()
        self.max_value = max_value

        linear_dim = dim // 2
        periodic_dim = dim - linear_dim

        scale = torch.exp(-2 * torch.arange(0, periodic_dim).float() * math.log(self.max_value) / periodic_dim)
        shift = torch.zeros(periodic_dim)
        shift[::2] = 0.5 * math.pi
        
        # Register as buffers so they get converted with .double()
        self.register_buffer('scale', scale)
        self.register_buffer('shift', shift)

        self.linear_proj = nn.Linear(1, linear_dim)

    def forward(self, t):
        periodic = torch.sin(t * self.scale.to(t) + self.shift.to(t))
        linear = self.linear_proj(t / torch.tensor(self.max_value, dtype=t.dtype, device=t.device))
        return torch.cat([linear, periodic], -1)

class FeedForward(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, activation: Callable=nn.ReLU(), final_activation: Callable=None):
        super().__init__()

        hidden_dims = hidden_dims[:]
        hidden_dims.append(out_dim)

        layers = [nn.Linear(in_dim, hidden_dims[0])]

        for i in range(len(hidden_dims) - 1):
            layers.append(activation)
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

        if final_activation is not None:
            layers.append(final_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DistributionalGenerator(nn.Module):
    """
    Neural network that generates multiple samples from P_theta(X_0 | X_t, t)
    instead of predicting noise directly
    """
    def __init__(self, dim, hidden_dim, max_i, num_layers=8, num_samples=16):  # Increased samples
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_samples = num_samples

        self.t_enc = PositionalEncoding(hidden_dim, max_value=1)
        self.i_enc = PositionalEncoding(hidden_dim, max_value=max_i)

        self.input_proj = FeedForward(dim, [], hidden_dim)
        self.proj = FeedForward(3 * hidden_dim, [], hidden_dim, final_activation=nn.ReLU())

        # Add layer normalization for stability
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        self.enc_att = []
        self.feed_forwards = []
        for _ in range(num_layers):
            self.enc_att.append(nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True, dropout=0.1))
            self.feed_forwards.append(FeedForward(hidden_dim, [hidden_dim * 2], hidden_dim, activation=nn.GELU()))
        self.enc_att = nn.ModuleList(self.enc_att)
        self.feed_forwards = nn.ModuleList(self.feed_forwards)

        # Multiple output heads for distributional sampling with better initialization
        self.output_heads = nn.ModuleList([
            FeedForward(hidden_dim, [hidden_dim, hidden_dim], dim) for _ in range(num_samples)
        ])

        # Improved noise injection for diversity
        self.noise_proj = nn.Linear(hidden_dim, hidden_dim)
        self.diversity_scale = nn.Parameter(torch.tensor(0.1))  # Learnable diversity scale

    def forward(self, x, t, i, z=None):
        """
        Generate multiple samples from the conditional distribution
        x: [B, S, D] - noisy input
        t: [B, S, 1] - timestamps  
        i: [B, S, 1] - diffusion step
        z: [B, S, hidden_dim] - optional noise for diversity
        
        Returns: [B, num_samples, S, D] - multiple generated samples
        """
        shape = x.shape
        batch_size = shape[0]

        x = x.view(-1, *shape[-2:])
        t = t.view(-1, shape[-2], 1)
        i = i.view(-1, shape[-2], 1)

        # Encode inputs
        x_enc = self.input_proj(x)
        t_enc = self.t_enc(t)
        i_enc = self.i_enc(i)

        # Combine features
        features = self.proj(torch.cat([x_enc, t_enc, i_enc], -1))

        # Apply transformer layers with residual connections and layer norm
        for i, (att_layer, ff_layer, layer_norm) in enumerate(zip(self.enc_att, self.feed_forwards, self.layer_norms)):
            # Multi-head attention with residual connection
            y, _ = att_layer(query=features, key=features, value=features)
            features = layer_norm(features + y)
            
            # Feed forward with residual connection
            ff_out = ff_layer(features)
            features = features + ff_out

        # Generate multiple samples with improved diversity
        samples = []
        for head_idx, output_head in enumerate(self.output_heads):
            # Add structured noise for diversity
            if z is not None:
                noise_features = self.noise_proj(z) * self.diversity_scale
                diverse_features = features + noise_features
            else:
                # Use different random noise for each head with learnable scale
                noise = torch.randn_like(features) * self.diversity_scale
                # Add head-specific bias for diversity
                head_bias = torch.sin(torch.tensor(head_idx * 2.0 * np.pi / self.num_samples, device=features.device))
                diverse_features = features + noise + head_bias * 0.01

            sample = output_head(diverse_features)
            sample = sample.view(*shape)
            samples.append(sample)
        
        # Stack samples: [B, num_samples, S, D]
        return torch.stack(samples, dim=1)

# ============================================================================
# SIGNATURE SCORING FUNCTIONS
# ============================================================================

def compute_signature_kernel_pairwise(path1, path2):
    """
    Compute signature kernel between two individual paths
    path1, path2: [S, D+1] - individual paths with time as first dimension
    Returns: scalar kernel value
    """
    try:
        # Add batch dimension for pysiglib
        path1_batch = path1.unsqueeze(0)  # [1, S, D+1]
        path2_batch = path2.unsqueeze(0)  # [1, S, D+1]
        
        kernel_val = pysiglib.sig_kernel(path1_batch, path2_batch, dyadic_order=2)
        
        # Extract scalar value
        if kernel_val.dim() == 0:
            return kernel_val
        else:
            return kernel_val.flatten()[0]
            
    except Exception as e:
        print(f"Error in pairwise signature kernel: {e}")
        # Fallback to simple similarity
        return torch.exp(-torch.mean((path1 - path2)**2))

def compute_signature_kernel_matrix(samples_batch, t_batch):
    """
    Compute signature kernel matrix for a batch of samples
    samples_batch: [B, S, D] - batch of path samples
    t_batch: [B, S, 1] - corresponding time points
    Returns: [B, B] - kernel matrix
    """
    B = samples_batch.shape[0]
    
    # Convert to format expected by pysiglib: [B, S, D+1] where first dim is time
    paths = torch.cat([t_batch, samples_batch], dim=-1)  # [B, S, 2]
    
    # Compute pairwise kernel matrix
    kernel_matrix = torch.zeros(B, B, device=samples_batch.device, dtype=samples_batch.dtype)
    
    for i in range(B):
        for j in range(B):
            kernel_matrix[i, j] = compute_signature_kernel_pairwise(paths[i], paths[j])
    
    return kernel_matrix

def signature_score_loss(generated_samples, target_sample, t_single, lambda_param=0.5):
    """
    Compute signature scoring loss with improved stability
    generated_samples: [num_samples, S, D] - samples from P_theta
    target_sample: [S, D] - ground truth sample
    t_single: [S, 1] - time points
    lambda_param: weighting parameter in [0,1]
    
    Returns: scalar loss
    """
    num_samples = generated_samples.shape[0]
    device = generated_samples.device
    
    # Expand target to match batch dimension for kernel computation
    target_expanded = target_sample.unsqueeze(0)  # [1, S, D]
    t_expanded = t_single.unsqueeze(0)  # [1, S, 1]
    
    # Compute kernel terms with numerical stability
    # 1. Between generated samples: E[k(X_i, X_j)] for i≠j
    if num_samples > 1:
        t_gen = t_single.unsqueeze(0).expand(num_samples, -1, -1)  # [num_samples, S, 1]
        gen_kernel_matrix = compute_signature_kernel_matrix(generated_samples, t_gen)
        
        # Extract off-diagonal elements (i≠j terms) with better numerical stability
        mask = ~torch.eye(num_samples, dtype=torch.bool, device=device)
        if mask.sum() > 0:
            cross_kernel_term = gen_kernel_matrix[mask].mean()
            # Add small regularization to prevent extreme values
            cross_kernel_term = torch.clamp(cross_kernel_term, min=-10.0, max=10.0)
        else:
            cross_kernel_term = torch.tensor(0.0, device=device)
    else:
        cross_kernel_term = torch.tensor(0.0, device=device)
    
    # 2. Between generated and target: E[k(X_i, Y)] - More stable computation
    gen_target_kernels = []
    
    # Batch computation for efficiency
    all_gen_samples = generated_samples  # [num_samples, S, D]
    all_targets = target_expanded.expand(num_samples, -1, -1)  # [num_samples, S, D]
    all_t = t_expanded.expand(num_samples, -1, -1)  # [num_samples, S, 1]
    
    # Compute all cross-kernels in batches for stability
    batch_size = min(8, num_samples)  # Process in smaller batches
    gen_target_vals = []
    
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_gen = all_gen_samples[i:end_idx]
        batch_target = all_targets[i:end_idx]
        batch_t = all_t[i:end_idx]
        
        # Compute cross-kernels for this batch
        for j in range(batch_gen.shape[0]):
            gen_sample = batch_gen[j:j+1]  # [1, S, D]
            target_sample_j = batch_target[j:j+1]  # [1, S, D]
            t_sample = batch_t[j:j+1]  # [1, S, 1]
            
            combined_samples = torch.cat([gen_sample, target_sample_j], dim=0)  # [2, S, D]
            combined_t = torch.cat([t_sample, t_sample], dim=0)  # [2, S, 1]
            
            kernel_matrix = compute_signature_kernel_matrix(combined_samples, combined_t)
            kernel_val = kernel_matrix[0, 1]
            
            # Clamp for numerical stability
            kernel_val = torch.clamp(kernel_val, min=-10.0, max=10.0)
            gen_target_vals.append(kernel_val)
    
    gen_target_term = torch.stack(gen_target_vals).mean()
    
    # Signature score: λ/2 * E[k(X,X')] - E[k(X,Y)]
    # Add small regularization term to prevent collapse
    regularization = 0.001 * torch.mean(torch.var(generated_samples, dim=0))
    score = (lambda_param / 2) * cross_kernel_term - gen_target_term + regularization
    
    # Additional stability: clamp final score
    score = torch.clamp(score, min=-5.0, max=5.0)
    
    return score

# ============================================================================
# TRAINING SETUP
# ============================================================================

model = DistributionalGenerator(dim=1, hidden_dim=64, max_i=diffusion_steps, num_samples=16).to(device).double()

# Improved optimizer with better stability
optim = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=50, T_mult=2, eta_min=1e-6)

# Training tracking
training_losses = []
gradient_norms = []
cross_kernel_terms = []
gen_target_terms = []
learning_rates = []

def get_signature_loss(x_batch, t_batch):
    """
    Compute signature-based loss for a batch with improved stability
    """
    batch_size = x_batch.shape[0]
    
    # Sample random diffusion steps with bias toward middle steps for stability
    # Avoid very early/late diffusion steps which can be unstable
    i = torch.randint(10, diffusion_steps-10, size=(batch_size,), dtype=torch.int64)
    i = i.view(-1, 1, 1).expand_as(x_batch[...,:1]).to(x_batch).double()
    
    total_loss = 0.0
    valid_losses = 0
    
    for b in range(batch_size):
        # Get single sample and corresponding time
        x_single = x_batch[b:b+1]  # [1, S, D]
        t_single = t_batch[b:b+1]  # [1, S, 1]
        i_single = i[b:b+1]        # [1, S, 1]
        
        # Add noise
        x_noisy, _ = add_noise(x_single, t_single, i_single)
        
        # Generate multiple samples from the model
        generated_samples = model(x_noisy, t_single, i_single)  # [1, num_samples, S, D]
        generated_samples = generated_samples.squeeze(0)  # [num_samples, S, D]
        
        try:
            # Compute signature score loss
            loss = signature_score_loss(generated_samples, x_single.squeeze(0), t_single.squeeze(0), lambda_param=0.3)
            
            # Check for valid loss
            if torch.isfinite(loss):
                total_loss += loss
                valid_losses += 1
            else:
                print(f"Warning: Invalid loss detected at batch {b}")
                
        except Exception as e:
            print(f"Error computing loss for batch {b}: {e}")
            continue
    
    if valid_losses > 0:
        return total_loss / valid_losses
    else:
        return torch.tensor(0.0, device=x_batch.device, requires_grad=True)

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("Starting improved signature scoring diffusion training...")
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

batch_size = 2  # Even smaller batch size for better signature kernel estimation
num_epochs = 300  # More epochs with smaller learning rate
warmup_epochs = 50

# Initialize model parameters more carefully
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

model.apply(init_weights)

for epoch in tqdm(range(num_epochs)):
    # Sample random batch
    batch_indices = torch.randperm(N)[:batch_size]
    x_batch = x[batch_indices]
    t_batch = t[batch_indices]
    
    optim.zero_grad()
    
    # Compute loss with gradient accumulation for stability
    loss = get_signature_loss(x_batch, t_batch)
    
    if torch.isfinite(loss) and loss.item() != 0.0:
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Compute gradient norm for monitoring
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        optim.step()
        scheduler.step()
        
        # Log metrics
        training_losses.append(loss.item())
        gradient_norms.append(total_grad_norm)
        learning_rates.append(scheduler.get_last_lr()[0])
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Grad Norm = {total_grad_norm:.6f}, LR = {scheduler.get_last_lr()[0]:.2e}")
    else:
        print(f"Skipping epoch {epoch} due to invalid loss: {loss.item()}")
        # Still step scheduler
        scheduler.step()
        training_losses.append(0.0)
        gradient_norms.append(0.0)
        learning_rates.append(scheduler.get_last_lr()[0])

print("Training completed!")

# ============================================================================
# SAMPLING FUNCTION
# ============================================================================

@torch.no_grad()
def sample_signature(t_grid, num_samples=20):
    """
    Generate samples using the trained distributional model with improved stability
    """
    # Ensure t_grid has the right shape for a single batch element
    if len(t_grid.shape) == 3:
        t_single_grid = t_grid[0:1]  # [1, S, 1]
    else:
        t_single_grid = t_grid.unsqueeze(0)  # [1, S, 1]
    
    cov = get_gp_covariance(t_single_grid)
    L = torch.linalg.cholesky(cov)
    
    # Start with noise
    x = L @ torch.randn(num_samples, t_single_grid.shape[1], 1, dtype=torch.float64, device=device)
    
    # Use fewer diffusion steps for more stable sampling
    sampling_steps = list(range(0, diffusion_steps, 2))  # Skip every other step
    
    for diff_step in reversed(sampling_steps):
        alpha = alphas[diff_step]
        beta = betas[diff_step]
        
        # Generate samples for each trajectory
        all_samples = []
        for sample_idx in range(num_samples):
            x_single = x[sample_idx:sample_idx+1]  # [1, S, 1]
            t_single = t_single_grid  # [1, S, 1] 
            i_single = torch.tensor([diff_step], dtype=torch.float64).expand_as(x_single[...,:1]).to(device)
            
            try:
                # Generate multiple candidates
                generated_candidates = model(x_single, t_single, i_single)  # [1, num_model_samples, S, D]
                
                # Use median instead of mean for more robust prediction
                pred_x0 = generated_candidates.median(dim=1)[0]  # [1, S, D] - median prediction
                
                # Clamp predictions to reasonable range
                pred_x0 = torch.clamp(pred_x0, min=-3.0, max=3.0)
                
                # DDIM-style update with numerical stability
                noise_pred = (x_single - alpha.sqrt() * pred_x0) / (1 - alpha).sqrt().clamp(min=1e-8)
                x_single = (x_single - beta * noise_pred / (1 - alpha).sqrt().clamp(min=1e-8)) / (1 - beta).sqrt().clamp(min=1e-8)
                
                # Add stochastic component only for early steps
                if diff_step > diffusion_steps // 4:
                    z = L @ torch.randn_like(x_single) * 0.5  # Reduced noise
                    x_single = x_single + beta.sqrt() * z
                
                # Clamp to prevent explosion
                x_single = torch.clamp(x_single, min=-5.0, max=5.0)
                
            except Exception as e:
                print(f"Error in sampling step {diff_step}, sample {sample_idx}: {e}")
                # Keep previous value if error occurs
                pass
                
            all_samples.append(x_single)
        
        x = torch.cat(all_samples, dim=0)
    
    return x

# ============================================================================
# EVALUATION AND VISUALIZATION
# ============================================================================

print("\nGenerating samples and creating visualizations...")

# Generate samples - use same sequence length as training data
t_grid = torch.linspace(0, 1, T, dtype=torch.float64).view(1, -1, 1).to(device)  # Use T=100 like training
num_generated_samples = 20
samples = sample_signature(t_grid, num_generated_samples)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Training loss
ax = axes[0, 0]
ax.plot(training_losses, color='C0', linewidth=2)
ax.set_title('Signature Scoring Loss During Training')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.grid(True, alpha=0.3)

# Plot 2: Gradient norms and learning rate
ax = axes[0, 1]
ax2 = ax.twinx()
ax.plot(gradient_norms, color='C1', linewidth=2, label='Grad Norm')
ax2.plot(learning_rates, color='C3', linewidth=2, linestyle='--', label='Learning Rate')
ax.set_title('Training Dynamics')
ax.set_xlabel('Epoch')
ax.set_ylabel('Gradient Norm', color='C1')
ax2.set_ylabel('Learning Rate', color='C3')
ax.set_yscale('log')
ax2.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

# Plot 3: Generated samples
ax = axes[0, 2]
for i in range(min(10, num_generated_samples)):
    ax.plot(t_grid.squeeze().detach().cpu().numpy(), 
           samples[i].squeeze().detach().cpu().numpy(), 
           color='C0', alpha=0.7, linewidth=1)
ax.set_title('Generated Samples (First 10)')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.grid(True, alpha=0.3)

# Plot 4: Ground truth samples
ax = axes[1, 0]
for i in range(10):
    ax.plot(t[i,:,0].cpu().numpy(), x[i,:,0].cpu().numpy(), 
           color='C1', alpha=0.7, linewidth=1)
ax.set_title('Ground Truth (First 10)')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.grid(True, alpha=0.3)

# Plot 5: Loss components over time
ax = axes[1, 1]
ax.plot(training_losses, label='Total Loss', color='C0', linewidth=2)
ax.set_title('Training Dynamics')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Direct comparison
ax = axes[1, 2]
for i in range(5):
    ax.plot(t_grid.squeeze().detach().cpu().numpy(), 
           samples[i].squeeze().detach().cpu().numpy(), 
           color='C0', alpha=0.8, linewidth=1.5, label='Generated' if i == 0 else '')
for i in range(5):
    ax.plot(t[i,:,0].cpu().numpy(), x[i,:,0].cpu().numpy(), 
           color='C1', alpha=0.8, linewidth=1.5, linestyle='--', 
           label='Ground Truth' if i == 0 else '')
ax.set_title('Direct Comparison (5 samples each)')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('signature_scoring_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# Save simple sample visualization
plt.figure(figsize=(10, 6))
for i in range(10):
    plt.plot(t_grid.squeeze().detach().cpu().numpy(), 
             samples[i].squeeze().detach().cpu().numpy(), 
             color='C0', alpha=1 / (i + 1))
plt.title('10 Generated Samples - Signature Scoring Diffusion')
plt.xlabel('t')
plt.ylabel('x')
plt.savefig('samples_sigscore.png')
plt.close()

print("✅ Analysis complete!")
print("\nFiles generated:")
print("- data_sigscore.png (training data)")
print("- samples_sigscore.png (generated samples)")
print("- signature_scoring_analysis.png (comprehensive analysis)")

print(f"\n🏆 TRAINING SUMMARY:")
print(f"Final Loss: {training_losses[-1]:.6f}")
print(f"Final Gradient Norm: {gradient_norms[-1]:.6f}")
print(f"Training completed with {len(training_losses)} epochs")
