from typing import List, Callable
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


torch.manual_seed(123)

N, T = 200, 100
t = torch.rand(N, T, 1, dtype=torch.float64).sort(1)[0].to(device)
x = torch.sin(10 * t + 2 * np.pi * torch.rand(N, 1, 1, dtype=torch.float64).to(t))

for i in range(10):
    plt.plot(t[i,:,0].cpu().numpy(), x[i,:,0].cpu().numpy(), color='C0', alpha=1 / (i + 1))
plt.title('10 samples from the dataset\nEach curve is one "data point"')
plt.xlabel('t')
plt.ylabel('x')
plt.savefig('data.png')
plt.close()


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

    
class TransformerModel(nn.Module):
    def __init__(self, dim, hidden_dim, max_i, num_layers=8, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.t_enc = PositionalEncoding(hidden_dim, max_value=1)
        self.i_enc = PositionalEncoding(hidden_dim, max_value=max_i)

        self.input_proj = FeedForward(dim, [], hidden_dim)

        self.proj = FeedForward(3 * hidden_dim, [], hidden_dim, final_activation=nn.ReLU())

        self.enc_att = []
        self.i_proj = []
        for _ in range(num_layers):
            self.enc_att.append(nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True))
            self.i_proj.append(nn.Linear(3 * hidden_dim, hidden_dim))
        self.enc_att = nn.ModuleList(self.enc_att)
        self.i_proj = nn.ModuleList(self.i_proj)

        self.output_proj = FeedForward(hidden_dim, [], dim)

    def forward(self, x, t, i):
        shape = x.shape

        x = x.view(-1, *shape[-2:])
        t = t.view(-1, shape[-2], 1)
        i = i.view(-1, shape[-2], 1)

        x = self.input_proj(x)
        t = self.t_enc(t)
        i = self.i_enc(i)

        x = self.proj(torch.cat([x, t, i], -1))

        for att_layer, i_proj in zip(self.enc_att, self.i_proj):
            y, _ = att_layer(query=x, key=x, value=x)
            x = x + torch.relu(y)

        x = self.output_proj(x)
        x = x.view(*shape)
        return x
    

model = TransformerModel(dim=1, hidden_dim=64, max_i=diffusion_steps).to(device).double()  # Convert to double precision
optim = torch.optim.Adam(model.parameters())



def get_loss(x, t):
    i = torch.randint(0, diffusion_steps, size=(x.shape[0],), dtype=torch.int64)
    i = i.view(-1, 1, 1).expand_as(x[...,:1]).to(x).double()  # Ensure double precision
    
    x_noisy, noise = add_noise(x, t, i)
    pred_noise = model(x_noisy, t, i)
    
    loss = (pred_noise - noise)**2
    return torch.mean(loss)


for i in tqdm(range(500)):
    optim.zero_grad()
    loss = get_loss(x, t)
    loss.backward()
    optim.step()
    if i % 1000 == 0:
        print(loss.item())


@torch.no_grad()
def sample(t):
    cov = get_gp_covariance(t)
    L = torch.linalg.cholesky(cov)

    x = L @ torch.randn_like(t)
    
    for diff_step in reversed(range(0, diffusion_steps)):
        alpha = alphas[diff_step]
        beta = betas[diff_step]

        z = L @ torch.randn_like(t)
        
        i = torch.tensor([diff_step], dtype=torch.float64).expand_as(x[...,:1]).to(device)
        pred_noise = model(x, t, i)
        
        x = (x - beta * pred_noise / (1 - alpha).sqrt()) / (1 - beta).sqrt() + beta.sqrt() * z
    return x


# Generate samples and compare with ground truth
print("\nGenerating samples and computing metrics...")

# Generate more samples for better statistical analysis
t_grid = torch.linspace(0, 1, 200, dtype=torch.float64).view(1, -1, 1).to(device) # Note that we can use different sequence length here without any issues
num_generated_samples = 100  # Generate more samples for robust statistics
samples = sample(t_grid.repeat(num_generated_samples, 1, 1))

# Import libraries for quantitative comparison
try:
    from scipy.stats import wasserstein_distance
    from sklearn.metrics import mean_squared_error
    import numpy as np
    METRICS_AVAILABLE = True
    print("✅ Quantitative metrics available")
except ImportError:
    print("⚠️ Warning: scipy or sklearn not available. Quantitative comparison will be limited.")
    METRICS_AVAILABLE = False

# Quantitative Comparison with Ground Truth
if METRICS_AVAILABLE:
    print("\n📊 QUANTITATIVE ANALYSIS")
    print("="*50)
    
    # Convert to numpy for analysis
    generated_samples = samples.detach().cpu().numpy()
    ground_truth_data = x.detach().cpu().numpy()  # Original training data
    
    # Flatten all samples for distributional comparison
    generated_flat = generated_samples.reshape(-1)  # All generated values
    ground_truth_flat = ground_truth_data.reshape(-1)  # All ground truth values
    
    # Compute Wasserstein distance
    wasserstein_dist = wasserstein_distance(generated_flat, ground_truth_flat)
    
    # Compute statistical metrics
    gen_mean, gen_std = np.mean(generated_flat), np.std(generated_flat)
    gt_mean, gt_std = np.mean(ground_truth_flat), np.std(ground_truth_flat)
    
    # Mean Squared Error of statistics
    mse_mean = (gen_mean - gt_mean)**2
    mse_std = (gen_std - gt_std)**2
    mse_stats = mse_mean + mse_std
    
    # Print results
    print(f"Wasserstein Distance:     {wasserstein_dist:.6f}")
    print(f"MSE (Statistics):         {mse_stats:.6f}")
    print(f"  - MSE (Mean):           {mse_mean:.6f}")
    print(f"  - MSE (Std):            {mse_std:.6f}")
    print()
    print("Statistical Comparison:")
    print(f"Generated - Mean: {gen_mean:.6f}, Std: {gen_std:.6f}")
    print(f"Ground Truth - Mean: {gt_mean:.6f}, Std: {gt_std:.6f}")
    
    # Compute trajectory-wise metrics
    print(f"\nTrajectory Analysis ({num_generated_samples} generated vs {len(ground_truth_data)} ground truth):")
    
    # Compare trajectory statistics
    gen_traj_means = np.mean(generated_samples, axis=(1, 2))  # Mean per trajectory
    gen_traj_stds = np.std(generated_samples, axis=(1, 2))    # Std per trajectory
    gt_traj_means = np.mean(ground_truth_data, axis=(1, 2))
    gt_traj_stds = np.std(ground_truth_data, axis=(1, 2))
    
    # Wasserstein distance between trajectory means and stds
    wd_means = wasserstein_distance(gen_traj_means, gt_traj_means)
    wd_stds = wasserstein_distance(gen_traj_stds, gt_traj_stds)
    
    print(f"Wasserstein Distance (Trajectory Means): {wd_means:.6f}")
    print(f"Wasserstein Distance (Trajectory Stds):  {wd_stds:.6f}")

# Create comprehensive visualization
print("\n🎨 Creating comprehensive visualization...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Sample trajectories (first 10)
ax = axes[0, 0]
for i in range(10):
    ax.plot(t_grid.squeeze().detach().cpu().numpy(), 
           samples[i].squeeze().detach().cpu().numpy(), 
           color='C0', alpha=0.7)
ax.set_title('Generated Samples (First 10)')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.grid(True, alpha=0.3)

# Plot 2: Ground truth trajectories (first 10)
ax = axes[0, 1]
for i in range(10):
    ax.plot(t[i,:,0].cpu().numpy(), x[i,:,0].cpu().numpy(), 
           color='C1', alpha=0.7)
ax.set_title('Ground Truth (First 10)')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.grid(True, alpha=0.3)

# Plot 3: Distribution comparison (histograms)
if METRICS_AVAILABLE:
    ax = axes[0, 2]
    ax.hist(generated_flat, bins=50, alpha=0.6, label='Generated', color='C0', density=True)
    ax.hist(ground_truth_flat, bins=50, alpha=0.6, label='Ground Truth', color='C1', density=True)
    ax.set_title('Value Distribution Comparison')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot 4: Trajectory means comparison
if METRICS_AVAILABLE:
    ax = axes[1, 0]
    ax.hist(gen_traj_means, bins=30, alpha=0.6, label='Generated', color='C0', density=True)
    ax.hist(gt_traj_means, bins=30, alpha=0.6, label='Ground Truth', color='C1', density=True)
    ax.set_title('Trajectory Means Distribution')
    ax.set_xlabel('Trajectory Mean')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot 5: Trajectory stds comparison
if METRICS_AVAILABLE:
    ax = axes[1, 1]
    ax.hist(gen_traj_stds, bins=30, alpha=0.6, label='Generated', color='C0', density=True)
    ax.hist(gt_traj_stds, bins=30, alpha=0.6, label='Ground Truth', color='C1', density=True)
    ax.set_title('Trajectory Stds Distribution')
    ax.set_xlabel('Trajectory Std')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot 6: Side-by-side sample comparison
ax = axes[1, 2]
# Plot a few generated samples
for i in range(5):
    ax.plot(t_grid.squeeze().detach().cpu().numpy(), 
           samples[i].squeeze().detach().cpu().numpy(), 
           color='C0', alpha=0.8, linewidth=1.5, label='Generated' if i == 0 else '')
# Plot a few ground truth samples  
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
plt.savefig('example_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# Also save the simple version for compatibility
plt.figure(figsize=(10, 6))
for i in range(10):
    plt.plot(t_grid.squeeze().detach().cpu().numpy(), samples[i].squeeze().detach().cpu().numpy(), color='C0', alpha=1 / (i + 1))
plt.title('10 new realizations')
plt.xlabel('t')
plt.ylabel('x')
plt.savefig('samples.png')
plt.close()

print("✅ Analysis complete!")
print("\nFiles generated:")
print("- samples.png (original format)")
print("- example_comprehensive_analysis.png (detailed comparison)")

if METRICS_AVAILABLE:
    print(f"\n🏆 FINAL RESULTS:")
    print(f"Wasserstein Distance to Ground Truth: {wasserstein_dist:.6f}")
    print(f"Lower values indicate better similarity to ground truth data")
else:
    print("\nInstall scipy and sklearn for quantitative metrics:")
    print("pip install scipy scikit-learn")