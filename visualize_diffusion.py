from typing import List, Callable
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(123)

# ============================================================================
# LOAD TRAINED MODEL AND DATA (Copy from baseline_tsdiff.py)
# ============================================================================

# Generate 1D Ornstein-Uhlenbeck process data
N, T = 200, 100

def generate_ou_process(n_paths, n_steps, dt=0.01, theta=1.0, sigma=0.5, x0=0.0):
    """
    Generate Ornstein-Uhlenbeck process paths
    dX_t = -theta * X_t * dt + sigma * dW_t
    """
    # Time grid for each path (like original baseline with random sampling)
    t_grid = torch.rand(n_paths, n_steps, 1, dtype=torch.float64).sort(1)[0].to(device)
    
    # Generate OU paths using exact solution for better accuracy
    x_paths = torch.zeros(n_paths, n_steps, 1, dtype=torch.float64, device=device)
    
    for path_idx in range(n_paths):
        x_path = torch.zeros(n_steps, 1, dtype=torch.float64, device=device)
        x_path[0, 0] = x0  # Initial condition
        
        times = t_grid[path_idx, :, 0]  # [n_steps]
        
        for i in range(1, n_steps):
            dt_actual = times[i] - times[i-1]
            
            # OU exact solution: X(t+dt) = X(t) * exp(-theta*dt) + noise
            x_prev = x_path[i-1, 0]
            
            # Mean and variance of OU process
            mean = x_prev * torch.exp(-theta * dt_actual)
            var = (sigma**2 / (2 * theta)) * (1 - torch.exp(-2 * theta * dt_actual))
            
            # Sample from conditional distribution
            noise = torch.randn(1, dtype=torch.float64, device=device) * torch.sqrt(var.clamp(min=1e-8))
            x_path[i, 0] = mean + noise
        
        x_paths[path_idx] = x_path
    
    return t_grid, x_paths

# Generate OU process data
# Generate sinusoidal data (as requested)
t = torch.rand(N, T, 1, dtype=torch.float64).sort(1)[0].to(device)
x = torch.sin(10 * t + 2 * np.pi * torch.rand(N, 1, 1, dtype=torch.float64).to(t))
# Diffusion setup
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
    diag = torch.eye(t.shape[-2]).to(t) * 1e-5
    return torch.exp(-torch.square(s / gp_sigma)) + diag

def add_noise(x, t, i):
    noise_gaussian = torch.randn_like(x, dtype=torch.float64)
    
    cov = get_gp_covariance(t)
    L = torch.linalg.cholesky(cov)
    noise = L @ noise_gaussian
    
    alpha = alphas[i.long()].to(x)
    x_noisy = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
    
    return x_noisy, noise

# Load model architecture (simplified version for visualization)
class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_value: float):
        super().__init__()
        self.max_value = max_value

        linear_dim = dim // 2
        periodic_dim = dim - linear_dim

        scale = torch.exp(-2 * torch.arange(0, periodic_dim).float() * math.log(self.max_value) / periodic_dim)
        shift = torch.zeros(periodic_dim)
        shift[::2] = 0.5 * math.pi
        
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

# Create and train model quickly for visualization
model = TransformerModel(dim=1, hidden_dim=64, max_i=diffusion_steps).to(device).double()
optim = torch.optim.Adam(model.parameters())

def get_loss(x, t):
    i = torch.randint(0, diffusion_steps, size=(x.shape[0],), dtype=torch.int64)
    i = i.view(-1, 1, 1).expand_as(x[...,:1]).to(x).double()
    
    x_noisy, noise = add_noise(x, t, i)
    pred_noise = model(x_noisy, t, i)
    
    loss = (pred_noise - noise)**2
    return torch.mean(loss)

# Quick training
print("Quick training for visualization...")
for i in tqdm(range(5000)):  # Fewer epochs for quick demo
    optim.zero_grad()
    loss = get_loss(x, t)
    loss.backward()
    optim.step()

print(f"Training completed. Final loss: {loss.item():.6f}")

# ============================================================================
# FORWARD DIFFUSION VISUALIZATION
# ============================================================================

def visualize_forward_diffusion():
    """
    Visualize the forward diffusion process as noise is gradually added
    """
    print("\nCreating forward diffusion visualization...")
    
    # Select a few representative trajectories
    num_viz_samples = 5
    viz_indices = [0, 1, 2, 3, 4]
    x_viz = x[viz_indices]  # [5, T, 1]
    t_viz = t[viz_indices]  # [5, T, 1]
    
    # Steps to visualize (every 10 steps)
    viz_steps = list(range(0, diffusion_steps, 2)) + [diffusion_steps-1]
    
    # Store noisy samples for each step
    noisy_samples = []
    
    for step in viz_steps:
        i_step = torch.tensor([step], dtype=torch.int64).expand(num_viz_samples, 1, 1).to(x_viz).double()
        x_noisy, _ = add_noise(x_viz, t_viz, i_step)
        noisy_samples.append(x_noisy.detach().cpu().numpy())
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def animate(frame):
        ax.clear()
        
        # Plot noisy samples
        for i in range(num_viz_samples):
            ax.plot(t_viz[i,:,0].cpu().numpy(), noisy_samples[frame][i,:,0], 
                   color=f'C{i}', alpha=0.8, linewidth=2, 
                   label=f'Sample {i+1}' if frame == 0 else '')
        
        # Plot original clean data as reference (faded)
        for i in range(num_viz_samples):
            ax.plot(t_viz[i,:,0].cpu().numpy(), x_viz[i,:,0].cpu().numpy(), 
                   color=f'C{i}', alpha=0.2, linewidth=1, linestyle='--')
        
        step = viz_steps[frame]
        noise_level = 1 - alphas[step].item()
        
        ax.set_title(f'Forward Diffusion: Sinusoidal → Noise\nStep {step}/{diffusion_steps-1}, Noise Level: {noise_level:.3f}')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2, 2)  # Adjusted for sinusoidal range
        
        if frame == 0:
            ax.legend()
    
    ani = animation.FuncAnimation(fig, animate, frames=len(viz_steps), 
                                 interval=500, repeat=True, blit=False)
    ani.save('forward_diffusion.gif', writer='pillow', fps=2)
    plt.close()
    
    print("✅ Forward diffusion GIF saved as 'forward_diffusion.gif'")

# ============================================================================
# BACKWARD DIFFUSION VISUALIZATION  
# ============================================================================

@torch.no_grad()
def visualize_backward_diffusion():
    """
    Visualize the backward diffusion (denoising) process during sampling
    """
    print("\nCreating backward diffusion visualization...")
    
    # Use same time grid for consistency
    t_grid = torch.linspace(0, 1, 100, dtype=torch.float64).view(1, -1, 1).to(device)
    num_viz_samples = 5
    
    # Start with noise
    cov = get_gp_covariance(t_grid)
    L = torch.linalg.cholesky(cov)
    x_samples = L @ torch.randn(num_viz_samples, t_grid.shape[1], 1, dtype=torch.float64, device=device)
    
    # Store samples for each denoising step
    denoising_samples = []
    viz_steps = list(range(diffusion_steps-1, -1, -2))  # Every 2 steps backwards (matching forward)
    
    # Store both the samples AND the corresponding time steps
    denoising_samples_with_time = []
    
    current_x = x_samples.clone()
    
    for step_idx, diff_step in enumerate(reversed(range(0, diffusion_steps))):
        alpha = alphas[diff_step]
        beta = betas[diff_step]

        z = L @ torch.randn(num_viz_samples, t_grid.shape[1], 1, dtype=torch.float64, device=device)
        
        # Apply model prediction for each sample
        all_updated = []
        for sample_idx in range(num_viz_samples):
            x_single = current_x[sample_idx:sample_idx+1]
            i_single = torch.tensor([diff_step], dtype=torch.float64).expand_as(x_single[...,:1]).to(device)
            pred_noise = model(x_single, t_grid, i_single)
            
            x_updated = (x_single - beta * pred_noise / (1 - alpha).sqrt()) / (1 - beta).sqrt() + beta.sqrt() * z[sample_idx:sample_idx+1]
            all_updated.append(x_updated)
        
        current_x = torch.cat(all_updated, dim=0)
        
        # Store samples at visualization steps with time information
        if diff_step in viz_steps:
            # Calculate backward time: starts at T and goes to 0
            backward_time_step = (diffusion_steps - 1 - diff_step) / (diffusion_steps - 1)
            denoising_samples_with_time.append({
                'samples': current_x.detach().cpu().numpy(),
                'diff_step': diff_step,
                'backward_time': backward_time_step,
                'forward_step': diffusion_steps - 1 - diff_step
            })
    
    # Sort by forward steps (so animation goes from noise to clean)
    denoising_samples_with_time.sort(key=lambda x: x['forward_step'])
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def animate_backward(frame):
        ax.clear()
        
        # Get current frame data
        frame_data = denoising_samples_with_time[frame]
        samples_frame = frame_data['samples']
        diff_step = frame_data['diff_step']
        backward_time = frame_data['backward_time']
        forward_step = frame_data['forward_step']
        
        # Plot current denoised samples
        for i in range(num_viz_samples):
            ax.plot(t_grid.squeeze().detach().cpu().numpy(), samples_frame[i,:,0], 
                   color=f'C{i}', alpha=0.8, linewidth=2,
                   label=f'Sample {i+1}' if frame == 0 else '')
        
        # Plot some ground truth samples as reference (faded)
        for i in range(3):
            ax.plot(t[i,:,0].cpu().numpy(), x[i,:,0].cpu().numpy(), 
                   color='gray', alpha=0.3, linewidth=1, linestyle=':')
        
        remaining_noise = 1 - alphas[diff_step].item() if diff_step < diffusion_steps else 0
        
        # Show both forward progress and backward diffusion time
        ax.set_title(f'Backward Diffusion: Noise → Sinusoidal\n'
                    f'Denoising Step {forward_step}/{diffusion_steps-1} (Diffusion Time: {1-backward_time:.3f})\n'
                    f'Remaining Noise: {remaining_noise:.3f}')
        ax.set_xlabel('t (spatial coordinate)')
        ax.set_ylabel('x')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2, 2)
        
        # Add text showing the backward time flow
        ax.text(0.02, 0.98, f'Backward Time: {backward_time:.3f} → 0.000', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if frame == 0:
            ax.legend()
    
    ani = animation.FuncAnimation(fig, animate_backward, frames=len(denoising_samples_with_time), 
                                 interval=300, repeat=True, blit=False)  # Faster animation
    ani.save('backward_diffusion.gif', writer='pillow', fps=3)
    plt.close()
    
    print("✅ Backward diffusion GIF saved as 'backward_diffusion.gif'")

# ============================================================================
# SIDE-BY-SIDE COMPARISON VISUALIZATION
# ============================================================================

def visualize_side_by_side():
    """
    Create a side-by-side comparison of forward and backward processes
    """
    print("\nCreating side-by-side comparison...")
    
    # Select one sample for detailed visualization
    sample_idx = 0
    x_single = x[sample_idx:sample_idx+1]  # [1, T, 1]
    t_single = t[sample_idx:sample_idx+1]  # [1, T, 1]
    
    # Forward process: gradually add noise
    forward_steps = list(range(0, diffusion_steps, 20))
    forward_samples = []
    
    for step in forward_steps:
        i_step = torch.tensor([step], dtype=torch.int64).expand_as(x_single[...,:1]).to(x_single).double()
        x_noisy, _ = add_noise(x_single, t_single, i_step)
        forward_samples.append(x_noisy[0,:,0].detach().cpu().numpy())
    
    # Backward process: start from noise and denoise
    t_grid = torch.linspace(0, 1, 100, dtype=torch.float64).view(1, -1, 1).to(device)
    cov = get_gp_covariance(t_grid)
    L = torch.linalg.cholesky(cov)
    
    x_backward = L @ torch.randn(1, t_grid.shape[1], 1, dtype=torch.float64, device=device)
    backward_samples = []
    backward_steps = list(range(diffusion_steps-1, -1, -20))
    
    current_x = x_backward.clone()
    step_counter = 0
    
    for diff_step in reversed(range(0, diffusion_steps)):
        alpha = alphas[diff_step]
        beta = betas[diff_step]
        
        z = L @ torch.randn_like(current_x)
        i_single = torch.tensor([diff_step], dtype=torch.float64).expand_as(current_x[...,:1]).to(device)
        pred_noise = model(current_x, t_grid, i_single)
        
        current_x = (current_x - beta * pred_noise / (1 - alpha).sqrt()) / (1 - beta).sqrt() + beta.sqrt() * z
        
        if diff_step in backward_steps:
            backward_samples.append(current_x[0,:,0].detach().cpu().numpy())
    
    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Forward process
    for i, (step, sample) in enumerate(zip(forward_steps, forward_samples)):
        alpha_val = 1 - alphas[step].item()
        ax1.plot(t_single[0,:,0].cpu().numpy(), sample, 
                alpha=0.6 + 0.4 * (i / len(forward_steps)), 
                color=plt.cm.Reds(i / len(forward_steps)),
                linewidth=2, label=f'Step {step} (α={alpha_val:.2f})')
    
    ax1.plot(t_single[0,:,0].cpu().numpy(), x_single[0,:,0].cpu().numpy(), 
            'k--', linewidth=3, alpha=0.8, label='Original Clean')
    ax1.set_title('Forward Diffusion: Sinusoidal → Noise\n(Adding GP-structured Noise)')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-2, 2)
    
    # Backward process
    for i, (step, sample) in enumerate(zip(backward_steps[::-1], backward_samples[::-1])):
        remaining_noise = 1 - alphas[step].item() if step >= 0 else 0
        ax2.plot(t_grid[0,:,0].cpu().numpy(), sample,
                alpha=0.6 + 0.4 * (i / len(backward_samples)),
                color=plt.cm.Blues(i / len(backward_samples)),
                linewidth=2, label=f'Step {diffusion_steps-1-step} (noise={remaining_noise:.2f})')
    
    ax2.set_title('Backward Diffusion: Noise → Sinusoidal\n(Denoising with Transformer)')
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-2, 2)
    
    plt.tight_layout()
    plt.savefig('diffusion_process_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Side-by-side comparison saved as 'diffusion_process_comparison.png'")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🎬 Creating diffusion process visualizations...")
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create all visualizations
    visualize_forward_diffusion()
    visualize_backward_diffusion() 
    visualize_side_by_side()
    
    print("\n✅ All sinusoidal diffusion visualizations completed!")
    print("\nFiles generated:")
    print("- forward_diffusion.gif (sinusoidal → noise process)")
    print("- backward_diffusion.gif (noise → sinusoidal denoising)")
    print("- diffusion_process_comparison.png (side-by-side comparison)")
