"""
Kernel Score Loss Module

This module implements various kernel-based scoring functions for diffusion models.
Includes RBF, exponential, and Lp norm kernels with strict properness testing capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Callable
import math


class KernelScoreLoss(nn.Module):
    """
    General kernel score loss for training diffusion models on path/time series data.
    
    Implements the generalized kernel score:
    S_λ,k(P,Y) = (λ/2)E[k(X,X')] - E[k(X,Y)]
    
    where k is a user-specified kernel function.
    
    Args:
        kernel_type (str): Type of kernel ('rbf', 'exponential', 'lp_norm', 'polynomial')
        lambda_param (float): Balance parameter between diversity and similarity
        num_samples (int): Number of samples for expectation estimation
        kernel_params (dict): Kernel-specific parameters
        clamp_range (tuple): Range to clamp loss for numerical stability
    """
    
    def __init__(
        self, 
        kernel_type: str = 'rbf',
        lambda_param: float = 0.5,
        num_samples: int = 8,
        kernel_params: Optional[dict] = None,
        clamp_range: Tuple[float, float] = (-10.0, 10.0)
    ):
        super().__init__()
        self.kernel_type = kernel_type
        self.lambda_param = lambda_param
        self.num_samples = num_samples
        self.clamp_range = clamp_range
        
        # Set default kernel parameters
        if kernel_params is None:
            kernel_params = {}
            
        self.kernel_params = self._get_default_params(kernel_type, kernel_params)
        
        # Get kernel function
        self.kernel_fn = self._get_kernel_function(kernel_type)
        
        print(f"KernelScoreLoss initialized:")
        print(f"  Kernel type: {kernel_type}")
        print(f"  λ = {lambda_param}")
        print(f"  m = {num_samples} samples")
        print(f"  Kernel params: {self.kernel_params}")
    
    def _get_default_params(self, kernel_type: str, user_params: dict) -> dict:
        """Get default parameters for each kernel type"""
        defaults = {
            'rbf': {'gamma': 1.0},
            'exponential': {'gamma': 1.0},
            'lp_norm': {'p': 2.0}
        }
        
        if kernel_type not in defaults:
            raise ValueError(f"Unknown kernel type: {kernel_type}. Available: {list(defaults.keys())}")
            
        # Merge user parameters with defaults
        params = defaults[kernel_type].copy()
        params.update(user_params)
        return params
    
    def _get_kernel_function(self, kernel_type: str) -> Callable:
        """Get the kernel function based on type"""
        kernel_functions = {
            'rbf': self._rbf_kernel,
            'exponential': self._exponential_kernel,
            'lp_norm': self._lp_norm_kernel
        }
        
        if kernel_type not in kernel_functions:
            raise ValueError(f"Unknown kernel type: {kernel_type}. Available: {list(kernel_functions.keys())}")
            
        return kernel_functions[kernel_type]
    
    def _create_path(self, time_points: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Create path representation by concatenating time and values
        
        Args:
            time_points: [S] or [S, 1] - time coordinates
            values: [S, D] - function values
            
        Returns:
            path: [S, 1+D] - combined path
        """
        if time_points.dim() == 1:
            time_points = time_points.unsqueeze(-1)
        if values.dim() == 1:
            values = values.unsqueeze(-1)
            
        path = torch.cat([time_points, values], dim=-1)
        return path
    
    def _rbf_kernel(self, path1: torch.Tensor, path2: torch.Tensor) -> torch.Tensor:
        """
        RBF (Gaussian) kernel: k(x,y) = exp(-γ||x-y||²)
        """
        gamma = self.kernel_params['gamma']
        diff = path1 - path2
        squared_dist = torch.sum(diff * diff)
        return torch.exp(-gamma * squared_dist)
    
    def _exponential_kernel(self, path1: torch.Tensor, path2: torch.Tensor) -> torch.Tensor:
        """
        Exponential kernel: k(x,y) = exp(-γ||x-y||)
        """
        gamma = self.kernel_params['gamma']
        diff = path1 - path2
        l1_dist = torch.sum(torch.abs(diff))
        return torch.exp(-gamma * l1_dist)
    
    def _lp_norm_kernel(self, path1: torch.Tensor, path2: torch.Tensor) -> torch.Tensor:
        """
        Lp norm kernel: k(x,y) = exp(-||x-y||_p)
        """
        p = self.kernel_params['p']
        diff = path1 - path2
        
        if p == 1:
            lp_norm = torch.sum(torch.abs(diff))
        elif p == 2:
            lp_norm = torch.sqrt(torch.sum(diff * diff))
        elif p == float('inf'):
            lp_norm = torch.max(torch.abs(diff))
        else:
            lp_norm = torch.sum(torch.abs(diff) ** p) ** (1.0 / p)
            
        return torch.exp(-lp_norm)
    
    
    def forward(
        self, 
        generated_samples: torch.Tensor, 
        target_sample: torch.Tensor, 
        time_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute kernel score loss using the specified kernel
        
        Args:
            generated_samples: [m, S, D] - m samples from predicted distribution P_θ
            target_sample: [S, D] - target sample from true distribution
            time_points: [S, 1] or [S] - time coordinates
            
        Returns:
            torch.Tensor: Scalar kernel score loss
        """
        m = generated_samples.shape[0]
        device = generated_samples.device
        
        # 1. Compute cross-kernel term: (λ/2m(m-1)) ∑_{i≠j} k(X̃_i, X̃_j)
        cross_kernel_sum = 0.0
        
        for i in range(m):
            for j in range(m):
                if i != j:
                    # Create paths
                    path_i = self._create_path(time_points, generated_samples[i])
                    path_j = self._create_path(time_points, generated_samples[j])
                    
                    # Compute kernel
                    kernel_val = self.kernel_fn(path_i, path_j)
                    cross_kernel_sum += kernel_val
        
        # Apply coefficient: (λ/2m(m-1))
        cross_kernel_term = (self.lambda_param / 2) * cross_kernel_sum / (m * (m - 1)) if m > 1 else torch.tensor(0.0, device=device)
        
        # 2. Compute target similarity term: (1/m) ∑_i k(X̃_i, X_0)
        target_kernel_sum = 0.0
        
        for i in range(m):
            # Create paths
            gen_path = self._create_path(time_points, generated_samples[i])
            target_path = self._create_path(time_points, target_sample)
            
            # Compute kernel
            kernel_val = self.kernel_fn(gen_path, target_path)
            target_kernel_sum += kernel_val
        
        target_kernel_term = target_kernel_sum / m
        
        # 3. Compute final score: cross_term - target_term
        score = cross_kernel_term - target_kernel_term
        
        # Clamp for numerical stability
        if self.clamp_range is not None:
            score = torch.clamp(score, self.clamp_range[0], self.clamp_range[1])
        
        return score
    
    def compute_components(
        self, 
        generated_samples: torch.Tensor, 
        target_sample: torch.Tensor, 
        time_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute kernel score components separately for analysis
        
        Returns:
            Tuple of (total_score, target_term, cross_term)
        """
        m = generated_samples.shape[0]
        device = generated_samples.device
        
        # Compute cross-kernel term
        cross_kernel_sum = 0.0
        for i in range(m):
            for j in range(m):
                if i != j:
                    path_i = self._create_path(time_points, generated_samples[i])
                    path_j = self._create_path(time_points, generated_samples[j])
                    kernel_val = self.kernel_fn(path_i, path_j)
                    cross_kernel_sum += kernel_val
        
        cross_kernel_term = (self.lambda_param / 2) * cross_kernel_sum / (m * (m - 1)) if m > 1 else torch.tensor(0.0, device=device)
        
        # Compute target similarity term
        target_kernel_sum = 0.0
        for i in range(m):
            gen_path = self._create_path(time_points, generated_samples[i])
            target_path = self._create_path(time_points, target_sample)
            kernel_val = self.kernel_fn(gen_path, target_path)
            target_kernel_sum += kernel_val
        
        target_kernel_term = target_kernel_sum / m
        
        # Total score
        score = cross_kernel_term - target_kernel_term
        
        if self.clamp_range is not None:
            score = torch.clamp(score, self.clamp_range[0], self.clamp_range[1])
        
        return score, target_kernel_term, cross_kernel_term


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_rbf_kernel_loss(gamma: float = 1.0, lambda_param: float = 0.5, 
                          num_samples: int = 8) -> KernelScoreLoss:
    """Create RBF kernel score loss"""
    return KernelScoreLoss(
        kernel_type='rbf',
        lambda_param=lambda_param,
        num_samples=num_samples,
        kernel_params={'gamma': gamma}
    )


def create_exponential_kernel_loss(gamma: float = 1.0, lambda_param: float = 0.5,
                                  num_samples: int = 8) -> KernelScoreLoss:
    """Create exponential kernel score loss"""
    return KernelScoreLoss(
        kernel_type='exponential',
        lambda_param=lambda_param,
        num_samples=num_samples,
        kernel_params={'gamma': gamma}
    )


def create_lp_norm_kernel_loss(p: float = 2.0, lambda_param: float = 0.5,
                              num_samples: int = 8) -> KernelScoreLoss:
    """Create Lp norm kernel score loss"""
    return KernelScoreLoss(
        kernel_type='lp_norm',
        lambda_param=lambda_param,
        num_samples=num_samples,
        kernel_params={'p': p}
    )




def get_available_kernels() -> dict:
    """Get list of available kernel types and their parameters"""
    return {
        'rbf': {
            'description': 'RBF (Gaussian) kernel: exp(-γ||x-y||²)',
            'parameters': ['gamma'],
            'defaults': {'gamma': 1.0}
        },
        'exponential': {
            'description': 'Exponential kernel: exp(-γ||x-y||)',
            'parameters': ['gamma'], 
            'defaults': {'gamma': 1.0}
        },
        'lp_norm': {
            'description': 'Lp norm kernel: exp(-||x-y||_p)',
            'parameters': ['p'],
            'defaults': {'p': 2.0}
        }
    }


# ============================================================================
# KERNEL COMPARISON UTILITIES
# ============================================================================

def compare_kernels_on_data(data1: torch.Tensor, data2: torch.Tensor, 
                           time_points: torch.Tensor, lambda_param: float = 0.5) -> dict:
    """
    Compare different kernels on the same data for analysis
    
    Args:
        data1: [m, S, D] - first set of samples
        data2: [S, D] - target sample
        time_points: [S] - time points
        lambda_param: Lambda parameter for scoring
        
    Returns:
        Dictionary with scores for each kernel type
    """
    results = {}
    available_kernels = get_available_kernels()
    
    for kernel_type, kernel_info in available_kernels.items():
        try:
            # Create kernel loss with default parameters
            kernel_loss = KernelScoreLoss(
                kernel_type=kernel_type,
                lambda_param=lambda_param,
                num_samples=data1.shape[0],
                kernel_params=kernel_info['defaults']
            )
            
            # Compute score
            score = kernel_loss(data1, data2, time_points)
            
            # Get components
            total_score, target_term, cross_term = kernel_loss.compute_components(data1, data2, time_points)
            
            results[kernel_type] = {
                'score': score.item(),
                'target_term': target_term.item(),
                'cross_term': cross_term.item(),
                'description': kernel_info['description']
            }
            
        except Exception as e:
            results[kernel_type] = {
                'error': str(e),
                'score': float('nan')
            }
    
    return results


def test_kernel_strict_properness(kernel_type: str, kernel_params: dict = None,
                                 lambda_param: float = 0.5, num_trials: int = 20) -> dict:
    """
    Test strict properness for a specific kernel
    
    Args:
        kernel_type: Type of kernel to test
        kernel_params: Kernel-specific parameters
        lambda_param: Lambda parameter
        num_trials: Number of trials for statistical testing
        
    Returns:
        Dictionary with properness test results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    n_points = 100
    t = torch.linspace(0, 1, n_points, dtype=torch.float64, device=device)
    
    # Generate sinusoidal data (same distribution)
    sin_samples = []
    for i in range(16):
        phase = 2 * np.pi * torch.rand(1, device=device)
        sin_wave = torch.sin(10 * t + phase).unsqueeze(-1)
        sin_samples.append(sin_wave)
    sin_samples = torch.stack(sin_samples, dim=0)
    
    # Generate exponential data (different distribution)
    exp_samples = []
    for i in range(16):
        decay_rate = 2 + 3 * torch.rand(1, device=device)
        exp_wave = torch.exp(-decay_rate * t).unsqueeze(-1)
        exp_samples.append(exp_wave)
    exp_samples = torch.stack(exp_samples, dim=0)
    
    # Create kernel loss
    kernel_loss = KernelScoreLoss(
        kernel_type=kernel_type,
        lambda_param=lambda_param,
        num_samples=8,
        kernel_params=kernel_params
    )
    
    same_scores = []
    diff_scores = []
    
    for trial in range(num_trials):
        # Sample for this trial
        pred_samples = sin_samples[:8]
        same_target = sin_samples[8 + trial % 8]
        diff_target = exp_samples[trial % 8]
        
        # Compute scores
        score_same = kernel_loss(pred_samples, same_target, t)
        score_diff = kernel_loss(pred_samples, diff_target, t)
        
        same_scores.append(score_same.item())
        diff_scores.append(score_diff.item())
    
    # Analyze results
    same_mean = np.mean(same_scores)
    diff_mean = np.mean(diff_scores)
    separation = diff_mean - same_mean
    is_proper = same_mean < diff_mean
    
    return {
        'kernel_type': kernel_type,
        'kernel_params': kernel_params or {},
        'lambda_param': lambda_param,
        'same_scores': same_scores,
        'diff_scores': diff_scores,
        'same_mean': same_mean,
        'diff_mean': diff_mean,
        'separation': separation,
        'is_proper': is_proper,
        'same_std': np.std(same_scores),
        'diff_std': np.std(diff_scores)
    }
