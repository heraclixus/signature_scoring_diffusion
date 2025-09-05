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
    Stabilized transformer model with proper reparameterization trick for signature scoring diffusion.
    
    Enhanced with layer normalization, dropout, and better initialization for training stability.
    
    This follows the VAE-style reparameterization where:
    - Transformer produces deterministic base prediction
    - Stochastic component Z is transformed and added
    - Final output: X₀ = base(Xₜ, t, i) + transform(Z)
    """
    
    def __init__(self, dim: int, hidden_dim: int, max_i: int, num_layers: int = 8, 
                 num_samples: int = 8, dropout: float = 0.1, 
                 init_method: str = "xavier_uniform", init_gain: float = 0.1, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_samples = num_samples
        self.dropout = dropout
        self.init_method = init_method
        self.init_gain = init_gain

        # Positional encodings
        self.t_enc = PositionalEncoding(hidden_dim, max_value=1)
        self.i_enc = PositionalEncoding(hidden_dim, max_value=max_i)

        # Input projection with layer norm
        self.input_proj = FeedForward(dim, [], hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Feature combination with layer norm
        self.proj = FeedForward(3 * hidden_dim, [], hidden_dim, final_activation=nn.ReLU())
        self.proj_norm = nn.LayerNorm(hidden_dim)

        # Transformer layers with proper normalization
        self.enc_att = []
        self.i_proj = []
        self.layer_norms1 = []  # Pre-attention layer norms
        self.layer_norms2 = []  # Pre-feedforward layer norms
        self.dropouts = []
        
        for _ in range(num_layers):
            self.enc_att.append(nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True, dropout=dropout))
            self.i_proj.append(nn.Linear(3 * hidden_dim, hidden_dim))
            self.layer_norms1.append(nn.LayerNorm(hidden_dim))
            self.layer_norms2.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
            
        self.enc_att = nn.ModuleList(self.enc_att)
        self.i_proj = nn.ModuleList(self.i_proj)
        self.layer_norms1 = nn.ModuleList(self.layer_norms1)
        self.layer_norms2 = nn.ModuleList(self.layer_norms2)
        self.dropouts = nn.ModuleList(self.dropouts)

        # Output heads for reparameterization with layer norms
        self.base_output = FeedForward(hidden_dim, [hidden_dim // 2], dim)  # More capacity
        self.base_norm = nn.LayerNorm(hidden_dim)
        
        self.stochastic_transform = FeedForward(dim, [hidden_dim, hidden_dim // 2], dim)  # More capacity
        self.stochastic_norm = nn.LayerNorm(dim)
        
        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """
        Configurable weight initialization for training stability.
        Supports Xavier uniform, Xavier normal, and other initialization methods.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Apply chosen initialization method
                if self.init_method == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight, gain=self.init_gain)
                elif self.init_method == "xavier_normal":
                    nn.init.xavier_normal_(module.weight, gain=self.init_gain)
                elif self.init_method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                elif self.init_method == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                elif self.init_method == "normal":
                    nn.init.normal_(module.weight, mean=0.0, std=self.init_gain)
                elif self.init_method == "orthogonal":
                    nn.init.orthogonal_(module.weight, gain=self.init_gain)
                else:
                    raise ValueError(f"Unknown initialization method: {self.init_method}")
                    
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize attention weights with same method
                for param in module.parameters():
                    if param.dim() > 1:
                        if self.init_method == "xavier_uniform":
                            nn.init.xavier_uniform_(param, gain=self.init_gain)
                        elif self.init_method == "xavier_normal":
                            nn.init.xavier_normal_(param, gain=self.init_gain)
                        else:
                            nn.init.xavier_uniform_(param, gain=self.init_gain)  # Default fallback
                    else:
                        nn.init.zeros_(param)
                        
            elif isinstance(module, nn.LayerNorm):
                # Standard layer norm initialization
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, t, i, z=None):
        """
        Forward pass with proper reparameterization trick and stabilization.
        
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

        # Deterministic processing with normalization
        x_features = self.input_norm(self.input_proj(x))
        t_features = self.t_enc(t)
        i_features = self.i_enc(i)

        # Combine deterministic features with normalization
        combined_features = self.proj_norm(self.proj(torch.cat([x_features, t_features, i_features], -1)))

        # Apply transformer layers with proper pre-norm architecture
        for att_layer, i_proj, ln1, ln2, dropout in zip(
            self.enc_att, self.i_proj, self.layer_norms1, self.layer_norms2, self.dropouts
        ):
            # Pre-norm attention with residual connection
            residual = combined_features
            normed_features = ln1(combined_features)
            y, _ = att_layer(query=normed_features, key=normed_features, value=normed_features)
            y = dropout(y)
            combined_features = residual + y
            
            # Pre-norm feedforward with residual connection
            residual = combined_features
            normed_features = ln2(combined_features)
            ff_out = torch.relu(normed_features)  # Simple activation instead of i_proj for now
            ff_out = dropout(ff_out)
            combined_features = residual + ff_out

        # Deterministic base prediction with normalization
        normed_features = self.base_norm(combined_features)
        base_output = self.base_output(normed_features)
        base_output = base_output.view(*shape)
        
        # Reparameterization trick: add transformed stochastic component
        if z is not None:
            z = z.view(-1, *shape[-2:])
            # Normalize stochastic input for stability
            z_normed = self.stochastic_norm(z)
            stochastic_component = self.stochastic_transform(z_normed)
            stochastic_component = stochastic_component.view(*shape)
            
            # Scale stochastic component for stability
            stochastic_component = stochastic_component * 0.1
            
            # Final output: base + scaled stochastic (like μ + σ⊙z in VAE)
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
