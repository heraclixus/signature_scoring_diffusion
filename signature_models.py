"""
Signature Scoring Diffusion Models

This module contains the transformer models and related components for signature scoring diffusion.
Implements proper reparameterization trick for path-space generative modeling.
"""

from typing import List, Callable
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
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
    """Multi-layer feedforward network"""
    
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, 
                 activation: Callable = nn.ReLU(), final_activation: Callable = None):
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
    """
    Transformer model with proper reparameterization trick for signature scoring diffusion.
    
    Uses the same architecture as the baseline diffusion model but implements proper
    reparameterization: deterministic base + stochastic component transform.
    
    This follows the VAE-style reparameterization where:
    - Transformer produces deterministic base prediction
    - Stochastic component Z is transformed and added
    - Final output: X₀ = base(Xₜ, t, i) + transform(Z)
    """
    
    def __init__(self, dim: int, hidden_dim: int, max_i: int, num_layers: int = 8, 
                 num_samples: int = 8, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_samples = num_samples

        # Positional encodings
        self.t_enc = PositionalEncoding(hidden_dim, max_value=1)
        self.i_enc = PositionalEncoding(hidden_dim, max_value=max_i)

        # Input projection
        self.input_proj = FeedForward(dim, [], hidden_dim)

        # Feature combination (3 inputs: x, t, i - deterministic)
        self.proj = FeedForward(3 * hidden_dim, [], hidden_dim, final_activation=nn.ReLU())

        # Transformer layers (same as baseline)
        self.enc_att = []
        self.i_proj = []
        for _ in range(num_layers):
            self.enc_att.append(nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True))
            self.i_proj.append(nn.Linear(3 * hidden_dim, hidden_dim))
        self.enc_att = nn.ModuleList(self.enc_att)
        self.i_proj = nn.ModuleList(self.i_proj)

        # Output heads for reparameterization
        self.base_output = FeedForward(hidden_dim, [], dim)  # Deterministic base
        self.stochastic_transform = FeedForward(dim, [hidden_dim], dim)  # Transform for Z

    def forward(self, x, t, i, z=None):
        """
        Forward pass with proper reparameterization trick.
        
        Args:
            x: [B, S, D] - noisy input (Xₜ)
            t: [B, S, 1] - timestamps  
            i: [B, S, 1] - diffusion step
            z: [B, S, D] - OU process noise (stochastic component) - optional
            
        Returns:
            [B, S, D] - predicted clean sample X₀ = base + transform(Z)
        """
        shape = x.shape

        # Reshape inputs
        x = x.view(-1, *shape[-2:])
        t = t.view(-1, shape[-2], 1)
        i = i.view(-1, shape[-2], 1)

        # Deterministic processing (same as baseline transformer)
        x_features = self.input_proj(x)
        t_features = self.t_enc(t)
        i_features = self.i_enc(i)

        # Combine deterministic features: x, t, i
        combined_features = self.proj(torch.cat([x_features, t_features, i_features], -1))

        # Apply transformer layers (deterministic)
        for att_layer, i_proj in zip(self.enc_att, self.i_proj):
            y, _ = att_layer(query=combined_features, key=combined_features, value=combined_features)
            combined_features = combined_features + torch.relu(y)

        # Deterministic base prediction
        base_output = self.base_output(combined_features)
        base_output = base_output.view(*shape)
        
        # Reparameterization trick: add transformed stochastic component
        if z is not None:
            z = z.view(-1, *shape[-2:])
            stochastic_component = self.stochastic_transform(z)
            stochastic_component = stochastic_component.view(*shape)
            
            # Final output: base + stochastic (like μ + σ⊙z in VAE)
            output = base_output + stochastic_component
        else:
            output = base_output
        
        return output


def generate_ou_noise(t_grid, num_samples, batch_size=1, theta=2.0, sigma=0.8):
    """
    Generate independent OU process noise vectors for reparameterization trick.
    
    This implements the stochastic component Z used in the reparameterization trick.
    Each Z is an independent Ornstein-Uhlenbeck process that respects the temporal
    structure of paths.
    
    Args:
        t_grid: [B, S, 1] - time grid
        num_samples: int - number of independent OU processes to generate
        batch_size: int - batch size 
        theta: float - OU mean reversion parameter
        sigma: float - OU volatility parameter
    
    Returns:
        z: [B, num_samples, S, D] - independent OU noise processes
    """
    B, S, D = t_grid.shape
    z = torch.zeros(batch_size, num_samples, S, D, dtype=torch.float64, device=t_grid.device)
    
    for b in range(batch_size):
        times = t_grid[b, :, 0]  # [S]
        
        for sample_idx in range(num_samples):
            # Generate OU process for this sample
            ou_path = torch.zeros(S, D, dtype=torch.float64, device=t_grid.device)
            ou_path[0, 0] = 0.0  # Start at zero
            
            for i in range(1, S):
                dt = times[i] - times[i-1]
                
                # OU exact solution: X(t+dt) = X(t) * exp(-theta*dt) + noise
                x_prev = ou_path[i-1, 0]
                
                # Mean and variance for OU transition
                mean = x_prev * torch.exp(-theta * dt)
                var = (sigma**2 / (2 * theta)) * (1 - torch.exp(-2 * theta * dt))
                
                # Sample from conditional distribution
                noise = torch.randn(1, dtype=torch.float64, device=t_grid.device) * torch.sqrt(var.clamp(min=1e-8))
                ou_path[i, 0] = mean + noise
            
            z[b, sample_idx] = ou_path
    
    return z


def get_gp_covariance(t, gp_sigma=0.05):
    """
    Get Gaussian process covariance matrix for noise generation.
    
    Args:
        t: [B, S, 1] - time points
        gp_sigma: float - GP length scale
        
    Returns:
        Covariance matrix [B, S, S]
    """
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2]).to(t) * 1e-5  # for numerical stability
    return torch.exp(-torch.square(s / gp_sigma)) + diag


def add_noise(x, t, i, alphas, gp_sigma=0.05):
    """
    Add noise to clean data sample using GP-structured noise.
    
    Args:
        x: Clean data sample, shape [B, S, D]
        t: Times of observations, shape [B, S, 1]
        i: Diffusion step, shape [B, S, 1]
        alphas: Cumulative alpha values for diffusion
        gp_sigma: GP length scale
        
    Returns:
        Tuple of (x_noisy, noise)
    """
    noise_gaussian = torch.randn_like(x, dtype=torch.float64)
    
    cov = get_gp_covariance(t, gp_sigma)
    L = torch.linalg.cholesky(cov)
    noise = L @ noise_gaussian
    
    alpha = alphas[i.long()].to(x)
    x_noisy = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
    
    return x_noisy, noise
