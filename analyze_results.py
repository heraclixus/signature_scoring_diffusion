#!/usr/bin/env python3
"""
Analyze results from test_configurations.sh runs.

This script reads the JSON training logs and provides a comparative analysis
of different initialization methods.
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_training_logs():
    """Load all training log files"""
    log_files = glob.glob("training_log_init_*.json")
    
    if not log_files:
        print("❌ No training log files found. Run test_configurations.sh first.")
        return {}
    
    results = {}
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
                
            # Extract experiment name
            exp_name = data['config']['experiment_name']
            init_method = data['config']['init_method']
            
            results[init_method] = {
                'config': data['config'],
                'metrics': data['training_metrics'],
                'summary': data['final_summary'],
                'log_file': log_file
            }
            
        except Exception as e:
            print(f"⚠️ Error loading {log_file}: {e}")
    
    return results


def analyze_training_stability(results):
    """Analyze training stability across different initialization methods"""
    
    print("📊 TRAINING STABILITY ANALYSIS")
    print("=" * 60)
    
    stability_scores = {}
    
    for init_method, data in results.items():
        metrics = data['metrics']
        summary = data['summary']
        
        losses = metrics['losses']
        grad_norms = metrics['gradient_norms']
        
        # Calculate stability metrics
        if len(losses) > 50:
            # Loss stability (lower variance = more stable)
            recent_losses = losses[-50:]
            loss_variance = np.var(recent_losses)
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]  # Slope
            
            # Gradient stability
            recent_grads = grad_norms[-50:]
            grad_variance = np.var(recent_grads)
            
            # Final performance
            final_loss = summary['final_loss']
            best_loss = summary['best_loss']
            early_stopped = summary['epochs_without_improvement'] >= 100
            
            # Overall stability score (lower is better)
            stability_score = loss_variance + abs(loss_trend) * 10 + grad_variance * 0.1
            
            stability_scores[init_method] = {
                'stability_score': stability_score,
                'loss_variance': loss_variance,
                'loss_trend': loss_trend,
                'grad_variance': grad_variance,
                'final_loss': final_loss,
                'best_loss': best_loss,
                'early_stopped': early_stopped,
                'total_epochs': summary['total_epochs']
            }
            
            # Print individual results
            trend_desc = "decreasing" if loss_trend < 0 else "increasing" if loss_trend > 0 else "flat"
            stop_desc = "❌ Early stopped" if early_stopped else "✅ Completed"
            
            print(f"\n{init_method.upper()}:")
            print(f"  Final loss: {final_loss:.6f}")
            print(f"  Best loss:  {best_loss:.6f}")
            print(f"  Loss trend: {loss_trend:.8f} ({trend_desc})")
            print(f"  Loss variance: {loss_variance:.6f}")
            print(f"  Grad variance: {grad_variance:.6f}")
            print(f"  Epochs: {summary['total_epochs']}")
            print(f"  Status: {stop_desc}")
            print(f"  Stability score: {stability_score:.6f}")
    
    # Rank by stability
    if stability_scores:
        print(f"\n🏆 RANKING (by stability, lower score = more stable):")
        print("-" * 60)
        
        ranked = sorted(stability_scores.items(), key=lambda x: x[1]['stability_score'])
        
        for rank, (method, scores) in enumerate(ranked, 1):
            trend_emoji = "📉" if scores['loss_trend'] < 0 else "📈" if scores['loss_trend'] > 0 else "➡️"
            stop_emoji = "🛑" if scores['early_stopped'] else "✅"
            
            print(f"{rank}. {method.upper():<20} Score: {scores['stability_score']:.4f} "
                  f"{trend_emoji} Loss: {scores['final_loss']:.6f} {stop_emoji}")
        
        print(f"\n🎯 RECOMMENDATION:")
        best_method = ranked[0][0]
        best_scores = ranked[0][1]
        print(f"Use '{best_method}' initialization (most stable)")
        print(f"Best configuration achieved loss: {best_scores['best_loss']:.6f}")
        
        return best_method
    
    return None


def create_comparison_plots(results):
    """Create comparison plots for different initialization methods"""
    
    if not results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Loss curves
    ax = axes[0, 0]
    for init_method, data in results.items():
        losses = data['metrics']['losses']
        ax.plot(losses, label=init_method, alpha=0.8)
    ax.set_title('Training Loss Comparison')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gradient norms
    ax = axes[0, 1]
    for init_method, data in results.items():
        grad_norms = data['metrics']['gradient_norms']
        ax.plot(grad_norms, label=init_method, alpha=0.8)
    ax.set_title('Gradient Norm Comparison')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning rates
    ax = axes[1, 0]
    for init_method, data in results.items():
        lrs = data['metrics']['learning_rates']
        ax.plot(lrs, label=init_method, alpha=0.8)
    ax.set_title('Learning Rate Schedule')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final performance comparison
    ax = axes[1, 1]
    methods = list(results.keys())
    final_losses = [results[method]['summary']['final_loss'] for method in methods]
    best_losses = [results[method]['summary']['best_loss'] for method in methods]
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    ax.bar(x_pos - width/2, final_losses, width, label='Final Loss', alpha=0.8)
    ax.bar(x_pos + width/2, best_losses, width, label='Best Loss', alpha=0.8)
    
    ax.set_title('Final Performance Comparison')
    ax.set_xlabel('Initialization Method')
    ax.set_ylabel('Loss')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('initialization_comparison_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Comparison plots saved: initialization_comparison_analysis.png")


def main():
    """Main analysis function"""
    print("🔍 Analyzing Initialization Method Results")
    print("=" * 50)
    
    # Load results
    results = load_training_logs()
    
    if not results:
        return
    
    print(f"Found {len(results)} initialization method results:")
    for method in results.keys():
        print(f"  - {method}")
    
    # Analyze stability
    best_method = analyze_training_stability(results)
    
    # Create plots
    create_comparison_plots(results)
    
    if best_method:
        print(f"\n🎯 FINAL RECOMMENDATION: Use '{best_method}' initialization")
        print("\n💡 Next steps:")
        print(f"python baseline_sigscore.py --init_method {best_method} --init_gain 0.05 --num_epochs 1000")


if __name__ == "__main__":
    main()
