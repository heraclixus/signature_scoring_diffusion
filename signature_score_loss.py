import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import pysiglib.torch_api as pysiglib

class SignatureScoreLoss(nn.Module):
    """
    Signature Score Loss for training diffusion models on path/time series data.
    
    This loss function implements empirical signature score that is strictly proper.
    
    Args:
        lambda_param (float): Balance parameter between diversity and similarity (default: 0.5)
        num_samples (int): Number of samples to generate for expectation estimation (default: 16)
        dyadic_order (int): Dyadic order for signature kernel computation (default: 1)
        clamp_range (tuple): Range to clamp signature loss for numerical stability (default: (-10.0, 10.0))
        
    Forward Args:
        generated_samples (torch.Tensor): [m, S, D] - m samples from predicted distribution
        target_sample (torch.Tensor): [S, D] - target sample from true distribution  
        time_points (torch.Tensor): [S, 1] or [S] - time coordinates for the paths
        
    Returns:
        torch.Tensor: Scalar signature score loss (lower is better for same distribution)
    """
    
    def __init__(
        self, 
        lambda_param: float = 0.5,
        num_samples: int = 16,
        dyadic_order: int = 1,
        clamp_range: Tuple[float, float] = (-10.0, 10.0)
    ):
        super().__init__()
        self.lambda_param = lambda_param
        self.num_samples = num_samples
        self.dyadic_order = dyadic_order
        self.clamp_range = clamp_range
        
        print(f"SignatureScoreLoss initialized:")
        print(f"  λ = {lambda_param}")
        print(f"  m = {num_samples} samples")
        print(f"  Dyadic order = {dyadic_order}")
    
    def _compute_signature_kernel(self, path1: torch.Tensor, path2: torch.Tensor) -> torch.Tensor:
        """
        Compute signature kernel between two paths
        
        Args:
            path1, path2: [S, D+1] - paths with time as first dimension
            
        Returns:
            torch.Tensor: Scalar kernel value
        """
        # Add batch dimension for pysiglib
        path1_batch = path1.unsqueeze(0)  # [1, S, D+1]
        path2_batch = path2.unsqueeze(0)  # [1, S, D+1]
        
        kernel_val = pysiglib.sig_kernel(path1_batch, path2_batch, dyadic_order=self.dyadic_order)
        
        # Extract scalar value
        if kernel_val.dim() == 0:
            result = kernel_val
        else:
            result = kernel_val.flatten()[0]
        # Clamp for numerical stability
        result = torch.clamp(result, min=self.clamp_range[0], max=self.clamp_range[1])
        return result
    
    def _create_signature_path(self, time_points: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Create path for signature kernel computation
        
        Args:
            time_points: [S, 1] or [S] - time coordinates
            values: [S, D] - path values
            
        Returns:
            torch.Tensor: [S, D+1] - path with time as first dimension
        """
        # Ensure proper shapes
        if time_points.dim() == 1:
            time_points = time_points.unsqueeze(-1)  # [S] -> [S, 1]
        path = torch.cat([time_points, values], dim=-1)
        return path
    
    def forward(
        self, 
        generated_samples: torch.Tensor, 
        target_sample: torch.Tensor, 
        time_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute signature score loss using empirical formula
        
        Args:
            generated_samples: [m, S, D] - m samples from predicted distribution P_θ
            target_sample: [S, D] - target sample from true distribution
            time_points: [S, 1] or [S] - time coordinates
            
        Returns:
            torch.Tensor: Scalar signature score loss
        """
        m = generated_samples.shape[0]
        device = generated_samples.device
        # 1. Compute cross-kernel term: (λ/2m(m-1)) ∑_{i≠j} k_sig(X̃_i, X̃_j)
        cross_kernel_sum = 0.0
        
        for i in range(m):
            for j in range(m):
                if i != j:
                    # Create paths for signature kernel
                    path_i = self._create_signature_path(time_points, generated_samples[i])
                    path_j = self._create_signature_path(time_points, generated_samples[j])
                    
                    # Compute kernel
                    kernel_val = self._compute_signature_kernel(path_i, path_j)
                    cross_kernel_sum += kernel_val
        
        # Apply empirical formula coefficient: (λ/2m(m-1))
        cross_kernel_term = (self.lambda_param / 2) * cross_kernel_sum / (m * (m - 1)) if m > 1 else torch.tensor(0.0, device=device)   
        # 2. Compute target similarity term: (1/m) ∑_i k_sig(X̃_i, X_0)
        target_kernel_sum = 0.0
        
        for i in range(m):
            # Create paths
            gen_path = self._create_signature_path(time_points, generated_samples[i])
            target_path = self._create_signature_path(time_points, target_sample)
            
            # Compute kernel
            kernel_val = self._compute_signature_kernel(gen_path, target_path)
            target_kernel_sum += kernel_val
        
        target_kernel_term = target_kernel_sum / m
        
        # 3. Compute signature score using corrected empirical formula
        # cross_kernel_term already includes the (λ/2m(m-1)) coefficient
        signature_score = cross_kernel_term - target_kernel_term
        
        return signature_score
    
    def compute_components(
        self, 
        generated_samples: torch.Tensor, 
        target_sample: torch.Tensor, 
        time_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute signature score components for analysis
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                (signature_score, target_kernel_term, cross_kernel_term)
        """
        m = generated_samples.shape[0]
        device = generated_samples.device
        
        # Cross-kernel term: (λ/2m(m-1)) ∑_{i≠j} k_sig(X̃_i, X̃_j)
        cross_kernel_sum = 0.0
        
        for i in range(m):
            for j in range(m):
                if i != j:
                    path_i = self._create_signature_path(time_points, generated_samples[i])
                    path_j = self._create_signature_path(time_points, generated_samples[j])
                    kernel_val = self._compute_signature_kernel(path_i, path_j)
                    cross_kernel_sum += kernel_val
        
        # Apply empirical formula coefficient: (λ/2m(m-1))
        cross_kernel_term = (self.lambda_param / 2) * cross_kernel_sum / (m * (m - 1)) if m > 1 else torch.tensor(0.0, device=device)
        
        # Target similarity term
        target_kernel_sum = 0.0
        for i in range(m):
            gen_path = self._create_signature_path(time_points, generated_samples[i])
            target_path = self._create_signature_path(time_points, target_sample)
            kernel_val = self._compute_signature_kernel(gen_path, target_path)
            target_kernel_sum += kernel_val
        
        target_kernel_term = target_kernel_sum / m
        
        # Signature score: cross_kernel_term already includes the (λ/2m(m-1)) coefficient
        signature_score = cross_kernel_term - target_kernel_term
        
        return signature_score, target_kernel_term, cross_kernel_term


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_signature_score_loss(
    lambda_param: float = 0.5,
    num_samples: int = 16,
    dyadic_order: int = 1,
) -> SignatureScoreLoss:
    """
    Convenience function to create a signature score loss with common configurations
    
    Args:
        lambda_param: Balance between diversity and similarity (default: 0.5)
        num_samples: Number of samples for expectation estimation (default: 16)
        dyadic_order: Signature kernel dyadic order (default: 1)
        
    Returns:
        SignatureScoreLoss: Configured loss function
    """
    return SignatureScoreLoss(
        lambda_param=lambda_param,
        num_samples=num_samples,
        dyadic_order=dyadic_order,
    )


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the signature score loss
    """
    print("🔧 SignatureScoreLoss Usage Example")
    print("="*50)
    
    # Create loss function
    sig_loss_fn = create_signature_score_loss(
        lambda_param=0.5,
        num_samples=16,
        dyadic_order=1
    )
    
    # Example data
    device = torch.device('cpu')
    torch.manual_seed(42)
    
    # Time points
    n_points = 100
    t = torch.linspace(0, 1, n_points, dtype=torch.float64, device=device)
    
    # Generate example samples (sinusoidal with different phases)
    m = 16
    generated_samples = torch.zeros(m, n_points, 1, dtype=torch.float64, device=device)
    for i in range(m):
        phase = 2 * np.pi * torch.rand(1, device=device)
        amplitude = 0.8 + 0.4 * torch.rand(1, device=device)
        generated_samples[i, :, 0] = amplitude * torch.sin(10 * t + phase)
    
    # Target sample (also sinusoidal - same distribution)
    target_phase = 2 * np.pi * torch.rand(1, device=device)
    target_amplitude = 0.8 + 0.4 * torch.rand(1, device=device)
    target_sample = target_amplitude * torch.sin(10 * t + target_phase).unsqueeze(-1)
    
    # Compute loss
    loss = sig_loss_fn(generated_samples, target_sample, t)
    
    print(f"\nExample computation:")
    print(f"  Generated samples shape: {generated_samples.shape}")
    print(f"  Target sample shape: {target_sample.shape}")
    print(f"  Time points shape: {t.shape}")
    print(f"  Signature score loss: {loss.item():.6f}")
    
    # Compute components for analysis
    score, target_term, cross_term = sig_loss_fn.compute_components(generated_samples, target_sample, t)
    print(f"\nLoss components:")
    print(f"  E[k(X,Y)] (target similarity): {target_term.item():.6f}")
    print(f"  E[k(X,X')] (internal diversity): {cross_term.item():.6f}")
    print(f"  Signature score: {score.item():.6f}")
    print(f"  Formula: ({sig_loss_fn.lambda_param}/2) * {cross_term.item():.6f} - {target_term.item():.6f} = {score.item():.6f}")
    
    # Test with different distribution (exponential)
    print(f"\n🔬 Testing with different distribution:")
    exp_target = torch.exp(-3 * t).unsqueeze(-1)  # Different distribution
    loss_diff = sig_loss_fn(generated_samples, exp_target, t)
    
    print(f"  Same distribution loss: {loss.item():.6f}")
    print(f"  Different distribution loss: {loss_diff.item():.6f}")
    print(f"  Separation: {loss_diff.item() - loss.item():.6f}")
    print(f"  Strict properness: {'✅ PROPER' if loss.item() < loss_diff.item() else '❌ NOT PROPER'}")