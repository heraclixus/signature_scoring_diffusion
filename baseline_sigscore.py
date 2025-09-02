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

class SignatureScoreModel(nn.Module):
    """
    More complex transformer model for better learning capacity
    """
    def __init__(self, dim, hidden_dim, max_i, num_layers=10, num_samples=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_samples = num_samples

        self.t_enc = PositionalEncoding(hidden_dim, max_value=1)
        self.i_enc = PositionalEncoding(hidden_dim, max_value=max_i)

        self.input_proj = FeedForward(dim, [], hidden_dim)
        self.proj = FeedForward(3 * hidden_dim, [hidden_dim], hidden_dim, final_activation=nn.ReLU())

        # More complex transformer layers
        self.enc_att = []
        self.feed_forward = []
        self.layer_norms1 = []
        self.layer_norms2 = []
        
        for _ in range(num_layers):
            self.enc_att.append(nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True, dropout=0.1))
            self.feed_forward.append(FeedForward(hidden_dim, [hidden_dim * 2], hidden_dim, activation=nn.GELU()))
            self.layer_norms1.append(nn.LayerNorm(hidden_dim))
            self.layer_norms2.append(nn.LayerNorm(hidden_dim))
            
        self.enc_att = nn.ModuleList(self.enc_att)
        self.feed_forward = nn.ModuleList(self.feed_forward)
        self.layer_norms1 = nn.ModuleList(self.layer_norms1)
        self.layer_norms2 = nn.ModuleList(self.layer_norms2)

        # Multiple output heads with more capacity
        self.output_heads = nn.ModuleList([
            FeedForward(hidden_dim, [hidden_dim], dim) for _ in range(num_samples)
        ])
        
        # Noise projection for better diversity
        self.noise_proj = FeedForward(dim, [], hidden_dim)

    def forward(self, x, t, i, z=None):
        """
        Generate multiple clean X_0 predictions from the conditional distribution
        
        x: [B, S, D] - noisy input
        t: [B, S, 1] - timestamps  
        i: [B, S, 1] - diffusion step
        z: [B, num_samples, S, D] - noise input for distributional sampling
        
        Returns: [B, num_samples, S, D] - multiple predicted clean samples
        """
        shape = x.shape
        batch_size = shape[0]

        x = x.view(-1, *shape[-2:])
        t = t.view(-1, shape[-2], 1)
        i = i.view(-1, shape[-2], 1)

        # Encode inputs (same as baseline)
        x_enc = self.input_proj(x)
        t_enc = self.t_enc(t)
        i_enc = self.i_enc(i)

        # Combine features
        features = self.proj(torch.cat([x_enc, t_enc, i_enc], -1))

        # Apply transformer layers with proper residual connections and layer norms
        for att_layer, ff_layer, ln1, ln2 in zip(self.enc_att, self.feed_forward, self.layer_norms1, self.layer_norms2):
            # Multi-head attention with residual connection
            residual = features
            features = ln1(features)
            y, _ = att_layer(query=features, key=features, value=features)
            features = residual + y
            
            # Feed forward with residual connection  
            residual = features
            features = ln2(features)
            ff_out = ff_layer(features)
            features = residual + ff_out

        # Generate multiple samples with noise input for diversity
        samples = []
        for head_idx, output_head in enumerate(self.output_heads):
            if z is not None:
                # Use provided noise for this sample
                noise_input = z[:, head_idx].view(-1, *shape[-2:])  # [B, S, D]
                # Project noise to feature space
                noise_features = self.noise_proj(noise_input) * 0.2
                diverse_features = features + noise_features
            else:
                # Generate random noise if not provided
                noise = torch.randn_like(features) * 0.1
                diverse_features = features + noise

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
        
        kernel_val = pysiglib.sig_kernel(path1_batch, path2_batch, dyadic_order=3)
        
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
    Compute signature scoring loss - CORRECTED VERSION
    
    The signature score S_λ(P,Y) = λ/2 * E[k(X,X')] - E[k(X,Y)] should be MINIMIZED
    Since we want the generated distribution P to match the target Y, we want:
    - High similarity between generated samples and target: maximize E[k(X,Y)]
    - Controlled diversity among generated samples: moderate E[k(X,X')]
    
    So the LOSS to minimize is: -S_λ(P,Y) = E[k(X,Y)] - λ/2 * E[k(X,X')]
    """
    num_samples = generated_samples.shape[0]
    device = generated_samples.device
    
    # 1. Compute E[k(X_i, X_j)] for i≠j (diversity among generated samples)
    cross_kernel_sum = 0.0
    cross_kernel_count = 0
    
    if num_samples > 1:
        for i in range(num_samples):
            for j in range(i+1, num_samples):
                # Get two different generated samples
                sample_i = generated_samples[i:i+1]  # [1, S, D]
                sample_j = generated_samples[j:j+1]  # [1, S, D]
                
                # Combine for kernel computation
                combined = torch.cat([sample_i, sample_j], dim=0)  # [2, S, D]
                t_combined = torch.cat([t_single.unsqueeze(0), t_single.unsqueeze(0)], dim=0)  # [2, S, 1]
                
                # Compute kernel
                kernel_matrix = compute_signature_kernel_matrix(combined, t_combined)
                cross_kernel_sum += kernel_matrix[0, 1]
                cross_kernel_count += 1
        
        cross_kernel_term = cross_kernel_sum / cross_kernel_count if cross_kernel_count > 0 else torch.tensor(0.0, device=device)
    else:
        cross_kernel_term = torch.tensor(0.0, device=device)
    
    # 2. Compute E[k(X_i, Y)] (similarity between generated samples and target)
    gen_target_sum = 0.0
    
    for i in range(num_samples):
        gen_sample = generated_samples[i:i+1]  # [1, S, D]
        target_expanded = target_sample.unsqueeze(0)  # [1, S, D]
        
        # Combine for kernel computation
        combined = torch.cat([gen_sample, target_expanded], dim=0)  # [2, S, D]
        t_combined = torch.cat([t_single.unsqueeze(0), t_single.unsqueeze(0)], dim=0)  # [2, S, 1]
        
        # Compute kernel
        kernel_matrix = compute_signature_kernel_matrix(combined, t_combined)
        gen_target_sum += kernel_matrix[0, 1]
    
    gen_target_term = gen_target_sum / num_samples
    
    # CORRECTED LOSS: We want to MAXIMIZE similarity to target and control diversity
    # Loss = -S_λ(P,Y) = E[k(X,Y)] - λ/2 * E[k(X,X')] 
    # But we minimize loss, so: Loss = -E[k(X,Y)] + λ/2 * E[k(X,X')]
    loss = -gen_target_term + (lambda_param / 2) * cross_kernel_term
    
    # Less aggressive clamping to allow more learning
    loss = torch.clamp(loss, min=-50.0, max=50.0)
    
    # Debug: print components occasionally
    if torch.rand(1).item() < 0.02:  # 2% of the time
        print(f"Debug - gen_target_term: {gen_target_term:.4f}, cross_kernel_term: {cross_kernel_term:.4f}, loss: {loss:.4f}")
    
    return loss

# ============================================================================
# TRAINING SETUP
# ============================================================================

model = SignatureScoreModel(dim=1, hidden_dim=64, max_i=diffusion_steps, num_samples=32).to(device).double()

# Optimizer with more aggressive learning rate decay
optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.98)  # More aggressive decay

# Training tracking
training_losses = []
gradient_norms = []
learning_rates = []

def get_signature_loss(x_batch, t_batch):
    """
    Compute signature-based loss with noise input for distributional sampling
    """
    batch_size = x_batch.shape[0]
    
    # Sample random diffusion steps
    i = torch.randint(0, diffusion_steps, size=(batch_size,), dtype=torch.int64)
    i = i.view(-1, 1, 1).expand_as(x_batch[...,:1]).to(x_batch).double()
    
    total_loss = 0.0
    
    for b in range(batch_size):
        # Get single sample and corresponding time
        x_clean = x_batch[b:b+1]  # [1, S, D] - clean target
        t_single = t_batch[b:b+1]  # [1, S, 1]
        i_single = i[b:b+1]        # [1, S, 1]
        
        # Add noise to get x_noisy
        x_noisy, _ = add_noise(x_clean, t_single, i_single)
        
        # Generate noise input for distributional sampling
        num_samples = model.num_samples
        z = torch.randn(1, num_samples, x_clean.shape[1], x_clean.shape[2], 
                       dtype=torch.float64, device=device) * 0.1
        
        # Model predicts clean X_0 from noisy input with noise for diversity
        predicted_clean_samples = model(x_noisy, t_single, i_single, z)  # [1, num_samples, S, D]
        predicted_clean_samples = predicted_clean_samples.squeeze(0)  # [num_samples, S, D]
        
        # Compute signature score loss between predicted clean samples and true clean sample
        loss = signature_score_loss(predicted_clean_samples, x_clean.squeeze(0), t_single.squeeze(0), lambda_param=0.3)
        total_loss += loss
    
    return total_loss / batch_size

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("Starting corrected signature scoring diffusion training...")
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

# Better initialization to prevent bias toward trending down
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

model.apply(init_weights)

batch_size = 2  # Smaller batch size due to 32 samples
num_epochs = 250  # More epochs with stronger decay

for epoch in tqdm(range(num_epochs)):
    # Sample random batch
    batch_indices = torch.randperm(N)[:batch_size]
    x_batch = x[batch_indices]
    t_batch = t[batch_indices]
    
    optim.zero_grad()
    loss = get_signature_loss(x_batch, t_batch)
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
    scheduler.step()  # Apply learning rate decay
    
    # Log metrics
    training_losses.append(loss.item())
    gradient_norms.append(total_grad_norm)
    learning_rates.append(scheduler.get_last_lr()[0])
    
    if epoch % 50 == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Grad Norm = {total_grad_norm:.6f}, LR = {current_lr:.2e}")

print("Training completed!")

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
    
    cov = get_gp_covariance(t_single_grid)
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
            
            # Generate noise for distributional sampling
            z = torch.randn(1, model.num_samples, t_single_grid.shape[1], 1, 
                           dtype=torch.float64, device=device) * 0.1
            
            # Get clean X_0 predictions from model with noise input
            predicted_clean_candidates = model(x_single, t_single, i_single, z)  # [1, num_model_samples, S, D]
            
            # Use first prediction (or could use mean/median)
            pred_x0 = predicted_clean_candidates[0, 0]  # [S, D] - first sample
            pred_x0 = pred_x0.unsqueeze(0)  # [1, S, D]
            
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
