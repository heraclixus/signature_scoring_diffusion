#!/usr/bin/env python3
"""
Test strict properness of different kernel scoring rules.

This script systematically tests various kernels (RBF, exponential, Lp norm, etc.)
to determine which ones satisfy the strict properness property.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from kernels_score_loss import (
    KernelScoreLoss, get_available_kernels, test_kernel_strict_properness,
    compare_kernels_on_data
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("🔬 Testing Strict Properness of Different Kernel Scoring Rules")
print("=" * 70)

# ============================================================================
# TEST DATA GENERATION
# ============================================================================

def create_test_distributions(n_samples=30, n_points=100):
    """Create diverse test distributions for kernel testing"""
    t = torch.linspace(0, 1, n_points, dtype=torch.float64, device=device)
    
    distributions = {}
    
    # Distribution 1: Sinusoidal (various frequencies and phases)
    sin_samples = []
    for i in range(n_samples):
        freq = 5 + 10 * torch.rand(1, device=device)
        phase = 2 * np.pi * torch.rand(1, device=device)
        amplitude = 0.5 + 0.5 * torch.rand(1, device=device)
        sin_wave = amplitude * torch.sin(freq * t + phase)
        sin_samples.append(sin_wave.unsqueeze(-1))
    distributions['sinusoidal'] = torch.stack(sin_samples, dim=0)
    
    # Distribution 2: Exponential decay
    exp_samples = []
    for i in range(n_samples):
        decay_rate = 1 + 4 * torch.rand(1, device=device)
        amplitude = 0.8 + 0.4 * torch.rand(1, device=device)
        exp_wave = amplitude * torch.exp(-decay_rate * t)
        exp_samples.append(exp_wave.unsqueeze(-1))
    distributions['exponential'] = torch.stack(exp_samples, dim=0)
    
    # Distribution 3: Polynomial
    poly_samples = []
    for i in range(n_samples):
        a = torch.randn(1, device=device) * 0.5
        b = torch.randn(1, device=device) * 0.3
        c = torch.randn(1, device=device) * 0.2
        poly_wave = a + b * t + c * t**2
        poly_samples.append(poly_wave.unsqueeze(-1))
    distributions['polynomial'] = torch.stack(poly_samples, dim=0)
    
    # Distribution 4: Step functions
    step_samples = []
    for i in range(n_samples):
        n_steps = torch.randint(3, 8, (1,)).item()
        step_points = torch.sort(torch.rand(n_steps-1, device=device))[0]
        step_values = torch.randn(n_steps, device=device) * 0.5
        
        step_wave = torch.zeros_like(t)
        step_points = torch.cat([torch.tensor([0.0], device=device), step_points, torch.tensor([1.0], device=device)])
        
        for j in range(len(step_points)-1):
            mask = (t >= step_points[j]) & (t < step_points[j+1])
            if j == len(step_points)-2:  # Last interval
                mask = t >= step_points[j]
            step_wave[mask] = step_values[j]
        
        step_samples.append(step_wave.unsqueeze(-1))
    distributions['step_function'] = torch.stack(step_samples, dim=0)
    
    return distributions, t


# ============================================================================
# KERNEL-SPECIFIC STRICT PROPERNESS TESTS
# ============================================================================

def test_all_kernels_properness(lambda_values=[0.1, 0.5, 1.0], num_trials=20):
    """
    Test strict properness for all available kernels
    """
    print("\n🧪 Testing Strict Properness for All Kernel Types")
    print("-" * 60)
    
    available_kernels = get_available_kernels()
    distributions, t = create_test_distributions(n_samples=50, n_points=100)
    
    results = {}
    
    for kernel_type, kernel_info in available_kernels.items():
        print(f"\n🔧 Testing {kernel_type.upper()} Kernel")
        print(f"   Description: {kernel_info['description']}")
        
        kernel_results = {}
        
        for lambda_param in lambda_values:
            print(f"\n   λ = {lambda_param}:")
            
            try:
                # Test with default parameters
                properness_result = test_kernel_strict_properness(
                    kernel_type=kernel_type,
                    kernel_params=kernel_info['defaults'],
                    lambda_param=lambda_param,
                    num_trials=num_trials
                )
                
                print(f"     Same dist: {properness_result['same_mean']:.6f} ± {properness_result['same_std']:.6f}")
                print(f"     Diff dist: {properness_result['diff_mean']:.6f} ± {properness_result['diff_std']:.6f}")
                print(f"     Separation: {properness_result['separation']:.6f}")
                print(f"     Strict proper: {'✅ YES' if properness_result['is_proper'] else '❌ NO'}")
                
                kernel_results[lambda_param] = properness_result
                
            except Exception as e:
                print(f"     ❌ Error: {e}")
                kernel_results[lambda_param] = {'error': str(e), 'is_proper': False}
        
        results[kernel_type] = kernel_results
    
    return results


def test_kernel_parameter_sensitivity():
    """
    Test how kernel parameters affect strict properness
    """
    print("\n🎛️ Testing Kernel Parameter Sensitivity")
    print("-" * 50)
    
    distributions, t = create_test_distributions(n_samples=30, n_points=100)
    sin_samples = distributions['sinusoidal'][:8]
    sin_target = distributions['sinusoidal'][8]
    exp_target = distributions['exponential'][0]
    
    # Test RBF gamma values
    print("\n🔴 RBF Kernel - Gamma Sensitivity:")
    gamma_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    rbf_results = {}
    
    for gamma in gamma_values:
        try:
            result = test_kernel_strict_properness(
                kernel_type='rbf',
                kernel_params={'gamma': gamma},
                lambda_param=0.5,
                num_trials=10
            )
            
            print(f"   γ={gamma}: Sep={result['separation']:.6f}, Proper={'✅' if result['is_proper'] else '❌'}")
            rbf_results[gamma] = result
            
        except Exception as e:
            print(f"   γ={gamma}: ❌ Error - {e}")
    
    # Test Lp norm p values
    print("\n📏 Lp Norm Kernel - p Sensitivity:")
    p_values = [1.0, 1.5, 2.0, 3.0, 5.0, float('inf')]
    lp_results = {}
    
    for p in p_values:
        try:
            result = test_kernel_strict_properness(
                kernel_type='lp_norm',
                kernel_params={'p': p},
                lambda_param=0.5,
                num_trials=10
            )
            
            p_str = "∞" if p == float('inf') else f"{p}"
            print(f"   p={p_str}: Sep={result['separation']:.6f}, Proper={'✅' if result['is_proper'] else '❌'}")
            lp_results[p] = result
            
        except Exception as e:
            p_str = "∞" if p == float('inf') else f"{p}"
            print(f"   p={p_str}: ❌ Error - {e}")
    
    return {
        'rbf_gamma': rbf_results,
        'lp_p': lp_results
    }


def visualize_kernel_comparison(all_results, param_results):
    """Create comprehensive visualization of kernel comparison results"""
    print("\n📊 Creating kernel comparison visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Kernel type comparison (separation)
    ax = axes[0, 0]
    kernel_names = []
    separations = []
    is_proper_list = []
    
    for kernel_type, lambda_results in all_results.items():
        if 0.5 in lambda_results and 'separation' in lambda_results[0.5]:
            kernel_names.append(kernel_type)
            separations.append(lambda_results[0.5]['separation'])
            is_proper_list.append(lambda_results[0.5]['is_proper'])
    
    colors = ['green' if proper else 'red' for proper in is_proper_list]
    bars = ax.bar(range(len(kernel_names)), separations, color=colors, alpha=0.7)
    ax.set_xticks(range(len(kernel_names)))
    ax.set_xticklabels(kernel_names, rotation=45)
    ax.set_ylabel('Score Separation')
    ax.set_title('Kernel Type Comparison (λ=0.5)\nGreen=Proper, Red=Not Proper')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, sep in zip(bars, separations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{sep:.3f}', ha='center', va='bottom')
    
    # Plot 2: Lambda parameter effects (RBF kernel)
    ax = axes[0, 1]
    if 'rbf' in all_results:
        rbf_results = all_results['rbf']
        lambdas = sorted(rbf_results.keys())
        rbf_seps = [rbf_results[lam]['separation'] if 'separation' in rbf_results[lam] else 0 for lam in lambdas]
        rbf_proper = [rbf_results[lam]['is_proper'] if 'is_proper' in rbf_results[lam] else False for lam in lambdas]
        
        colors = ['green' if proper else 'red' for proper in rbf_proper]
        ax.scatter(lambdas, rbf_seps, c=colors, s=100, alpha=0.7)
        ax.set_xlabel('λ parameter')
        ax.set_ylabel('Score Separation')
        ax.set_title('RBF Kernel: λ vs Separation')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 3: RBF gamma sensitivity
    ax = axes[0, 2]
    if 'rbf_gamma' in param_results:
        gamma_results = param_results['rbf_gamma']
        gammas = sorted(gamma_results.keys())
        gamma_seps = [gamma_results[g]['separation'] if 'separation' in gamma_results[g] else 0 for g in gammas]
        gamma_proper = [gamma_results[g]['is_proper'] if 'is_proper' in gamma_results[g] else False for g in gammas]
        
        colors = ['green' if proper else 'red' for proper in gamma_proper]
        ax.scatter(gammas, gamma_seps, c=colors, s=100, alpha=0.7)
        ax.set_xlabel('γ parameter')
        ax.set_ylabel('Score Separation')
        ax.set_title('RBF Kernel: γ vs Separation')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Lp norm p sensitivity
    ax = axes[1, 0]
    if 'lp_p' in param_results:
        p_results = param_results['lp_p']
        ps = [p for p in sorted(p_results.keys()) if p != float('inf')]
        p_seps = [p_results[p]['separation'] if 'separation' in p_results[p] else 0 for p in ps]
        p_proper = [p_results[p]['is_proper'] if 'is_proper' in p_results[p] else False for p in ps]
        
        colors = ['green' if proper else 'red' for proper in p_proper]
        ax.scatter(ps, p_seps, c=colors, s=100, alpha=0.7)
        ax.set_xlabel('p parameter')
        ax.set_ylabel('Score Separation')
        ax.set_title('Lp Norm Kernel: p vs Separation')
        ax.grid(True, alpha=0.3)
    
    # Plot 5: Kernel comparison summary
    ax = axes[1, 1]
    # Show comparison of all three kernels at optimal parameters
    kernel_types = ['rbf', 'exponential', 'lp_norm']
    kernel_separations = []
    kernel_colors = []
    
    for kernel_type in kernel_types:
        if kernel_type in all_results and 0.5 in all_results[kernel_type]:
            result = all_results[kernel_type][0.5]
            if 'separation' in result:
                kernel_separations.append(result['separation'])
                kernel_colors.append('green' if result['is_proper'] else 'red')
            else:
                kernel_separations.append(0)
                kernel_colors.append('gray')
        else:
            kernel_separations.append(0)
            kernel_colors.append('gray')
    
    bars = ax.bar(kernel_types, kernel_separations, color=kernel_colors, alpha=0.7)
    ax.set_xlabel('Kernel Type')
    ax.set_ylabel('Score Separation')
    ax.set_title('Kernel Performance Comparison (λ=0.5)')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, sep in zip(bars, kernel_separations):
        if sep > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{sep:.3f}', ha='center', va='bottom')
    
    # Plot 6: Summary table
    ax = axes[1, 2]
    ax.axis('off')
    
    # Create summary table
    summary_data = [['Kernel', 'Proper?', 'Best Sep.']]
    
    for kernel_type, lambda_results in all_results.items():
        if isinstance(lambda_results, dict) and 0.5 in lambda_results:
            result = lambda_results[0.5]
            if 'is_proper' in result:
                proper_str = '✅' if result['is_proper'] else '❌'
                sep_str = f"{result['separation']:.3f}"
                summary_data.append([kernel_type, proper_str, sep_str])
    
    if len(summary_data) > 1:
        table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
    ax.set_title('Kernel Properness Summary (λ=0.5)')
    
    plt.tight_layout()
    plt.savefig('kernel_properness_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("📈 Kernel comparison plots saved: kernel_properness_analysis.png")


def find_best_kernel_configurations():
    """
    Find the best kernel configurations for strict properness
    """
    print("\n🏆 Finding Best Kernel Configurations")
    print("-" * 50)
    
    # Test all kernels with default settings
    all_results = test_all_kernels_properness(lambda_values=[0.1, 0.3, 0.5, 0.7, 1.0])
    
    # Test parameter sensitivity
    param_results = test_kernel_parameter_sensitivity()
    
    # Analyze results
    best_configs = []
    
    for kernel_type, lambda_results in all_results.items():
        # Find best lambda for this kernel
        proper_lambdas = []
        for lam, result in lambda_results.items():
            if isinstance(result, dict) and result.get('is_proper', False):
                proper_lambdas.append((lam, result['separation']))
        
        if proper_lambdas:
            best_lambda, best_separation = max(proper_lambdas, key=lambda x: x[1])
            best_configs.append({
                'kernel_type': kernel_type,
                'lambda_param': best_lambda,
                'separation': best_separation,
                'description': get_available_kernels()[kernel_type]['description']
            })
    
    # Sort by separation (higher is better)
    best_configs.sort(key=lambda x: x['separation'], reverse=True)
    
    print("\n🥇 RANKING (by separation, higher = better discrimination):")
    for i, config in enumerate(best_configs, 1):
        print(f"{i}. {config['kernel_type'].upper():<12} λ={config['lambda_param']:<3} Sep={config['separation']:.6f}")
        print(f"   {config['description']}")
    
    if best_configs:
        print(f"\n🎯 RECOMMENDATION:")
        best = best_configs[0]
        print(f"Use '{best['kernel_type']}' kernel with λ={best['lambda_param']}")
        print(f"Expected separation: {best['separation']:.6f}")
    
    return best_configs, all_results, param_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main testing function"""
    print("🚀 Starting comprehensive kernel properness analysis...")
    
    # Find best configurations
    best_configs, all_results, param_results = find_best_kernel_configurations()
    
    # Create visualizations
    visualize_kernel_comparison(all_results, param_results)
    
    # Print final recommendations
    print("\n" + "="*80)
    print("📋 KERNEL SCORING INVESTIGATION SUMMARY")
    print("="*80)
    
    print(f"\n🔍 Key Findings:")
    
    # Count proper kernels
    proper_kernels = []
    for kernel_type, lambda_results in all_results.items():
        has_proper_lambda = any(
            result.get('is_proper', False) 
            for result in lambda_results.values() 
            if isinstance(result, dict)
        )
        if has_proper_lambda:
            proper_kernels.append(kernel_type)
    
    print(f"  Strictly proper kernels: {len(proper_kernels)}/{len(all_results)}")
    print(f"  Proper kernel types: {proper_kernels}")
    
    if best_configs:
        print(f"\n🏆 Top 3 Recommendations:")
        for i, config in enumerate(best_configs[:3], 1):
            print(f"  {i}. {config['kernel_type']} (λ={config['lambda_param']}, sep={config['separation']:.6f})")
    
    # Parameter sensitivity insights
    if 'rbf_gamma' in param_results:
        rbf_proper_gammas = [g for g, r in param_results['rbf_gamma'].items() if r.get('is_proper', False)]
        if rbf_proper_gammas:
            print(f"\n🔴 RBF Kernel: Proper γ values = {rbf_proper_gammas}")
    
    if 'lp_p' in param_results:
        lp_proper_ps = [p for p, r in param_results['lp_p'].items() if r.get('is_proper', False)]
        if lp_proper_ps:
            print(f"📏 Lp Norm: Proper p values = {lp_proper_ps}")
    
    print(f"\n💡 Usage Examples:")
    if best_configs:
        best = best_configs[0]
        print(f"# Best kernel configuration:")
        print(f"python baseline_kernels_score.py \\")
        print(f"    --kernel_type {best['kernel_type']} \\")
        print(f"    --lambda_param {best['lambda_param']} \\")
        
        if best['kernel_type'] == 'rbf':
            print(f"    --gamma 1.0 \\")
        elif best['kernel_type'] == 'lp_norm':
            print(f"    --p_norm 2.0 \\")
        
        print(f"    --experiment_name best_{best['kernel_type']}")
    
    print(f"\n📁 Files generated:")
    print(f"  - kernel_properness_analysis.png (comprehensive analysis)")


if __name__ == "__main__":
    main()
