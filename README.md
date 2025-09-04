# Signature Scoring Diffusion for Time Series Generation

> This repository implements a novel approach to time series generation that combines diffusion models with signature-based scoring functions. The work extends traditional diffusion models by replacing standard noise prediction with distributional learning using signature kernels and proper reparameterization tricks in path space.

## Overview

### Baseline Implementation (`baseline_tsdiff.py`)

The baseline implements model is the [tsdiff](https://github.com/mbilos/tsdiff/) model.

#### Data Generation
- **Synthetic Dataset**: 200 sinusoidal time series with random phases: `x = sin(10t + 2π·rand)`
- **Irregular Time Sampling**: Each trajectory has 100 randomly sorted time points
- **Path-based Representation**: Treats entire time series as continuous paths

#### Forward Diffusion Process
- **Gaussian Process Noise Structure**: Uses exponential covariance `exp(-|t_i - t_j|²/σ²)` 
- **Temporal Correlation**: Noise respects temporal dependencies via GP covariance
- **Standard Schedule**: Linear β schedule from 1e-4 to 0.2 over 100 steps

#### Model Architecture
- **Transformer-based Denoiser**: 8-layer transformer with positional encoding
- **Multi-input Processing**: Handles noisy data `x`, timestamps `t`, and diffusion steps `i`
- **Positional Encoding**: Separate encoders for time and diffusion step information

#### Training & Sampling
- **Standard Denoising**: Learns to predict added GP-structured noise
- **DDPM Sampling**: Iterative denoising with GP-correlated random noise
- **Comprehensive Evaluation**: Wasserstein distance, statistical comparisons, visual analysis


## Proposed Extension: Signature Scoring Diffusion

### Mathematical Framework

The proposed method introduces a **distributional approach** to diffusion models using signature-based scoring functions:

#### Key Innovation: Distributional Learning
Instead of learning `E[X₀|Xₜ]` (point estimates), the model learns the **full conditional distribution** `P(X₀|Xₜ,t)`.

#### Forward Process (DSPD)
- **Path-Space Diffusion**: Treats time series as paths in function space
- **GP-Structured Noise**: Uses Ornstein-Uhlenbeck process covariance
- **Mathematical Formulation**:
  ```
  p(Xₜ|X₀) = N(αₜX₀, Σ(t))
  Σ(t)ᵢⱼ = σₜ exp(-γ|tᵢ - tⱼ|)
  ```

#### Signature Scoring Function
- **Signature Kernel**: `S_sig(P,Y) = ½E[k_sig(X,X')] - E[k_sig(X,Y)]`
- **Generalized Score**: `S_λ,sig(P,Y) = (λ/2)E[k_sig(X,X')] - E[k_sig(X,Y)]`
- **Path Signature**: Captures geometric properties of time series paths
- **Distributional Objective**: 
  ```
  L_sig = E[(1/m(m-1))Σᵢ≠ⱼ k_sig(X̃₀ⁱ,X̃₀ʲ) - (2/m)Σᵢ k_sig(X̃₀ⁱ,X₀)]
  ```

#### Training Algorithm
1. Sample diffusion time `t ~ U[0,1]` and data `X₀ ~ P₀`
2. Generate noisy samples `Xₜ` via forward process  
3. Use generator `P_θ` to produce `m` samples `{X̃₀ⁱ}`
4. Compute signature-based loss `L_sig`
5. Update parameters via gradient descent

#### Backward Sampling (DDIM-style)
- **Distributional Sampling**: Sample from learned distribution `P_θ(·|Xₜ,t)`
- **Deterministic Update**: `Xₛ = √αₛ X̃₀ + √(1-αₛ) Dₜ`
- **Flexible Scheduling**: Supports coarse-grained time discretization


## Key Differences

| Aspect | Baseline (Noise Prediction) | Proposed (Signature Scoring) |
|--------|---------------------------|------------------------------|
| **Learning Target** | Noise ε | Distribution P(X₀\|Xₜ,t) |
| **Loss Function** | MSE on noise | Signature kernel score |
| **Output** | Single denoised sample | Multiple samples from distribution |
| **Path Information** | Implicit in transformer | Explicit via signature kernels |
| **Sampling** | Deterministic denoising | Distributional sampling |

---

## Current Implementation Status

### **Reparameterization Trick in Path Space**

The signature scoring models uses VAE-style reparameterization:

#### **Mathematical Framework**
- **Deterministic Component**: `base = transformer(Xₜ, t, i)` 
- **Stochastic Component**: `Z ~ OU(θ=2.0, σ=0.8)` (Ornstein-Uhlenbeck process)
- **Reparameterization**: `X₀ = base + transform(Z)` (like μ + σ⊙z in VAEs)
- **Fair Comparison**: Same transformer architecture as baseline (8 layers, 1 head, same parameters)
- **OU Noise**: Independent OU processes for each sample (respects temporal structure)

### **Debugging and Monitoring**
#### **Training Diagnostics**
- **Loss Decrease Monitoring**: Warns if loss not improving over 20-epoch windows
- **Strict Properness Testing**: Verifies signature score behaves as proper scoring rule
- **Component Analysis**: Breaks down signature loss into target and cross terms
- **Real-time Validation**: Tests during training every 100 epochs

## Usage

### **Run Experiments**

#### **Baseline Diffusion**
```bash
python baseline_tsdiff.py
```
Generates standard diffusion model results for comparison.

#### **Baseline Signature Scoring**
```bash
python baseline_sigscore.py
```

## Current Challenges

### **Training Dynamics Issues**
- **Loss Stagnation**: Some configurations show non-decreasing loss
- **Strict Properness**: Needs verification during training
- **Component Balance**: Target vs cross terms may be imbalanced

## References
The method builds upon:
- **DSPD**: Biloš et al. (2023) - Modeling Temporal Data as Continuous Path
- **Distributional Diffusion**: De Bortoli et al. (2025) - Distributional Diffusion Models and Scoring
- **Neural SDEs**: Issa et al. (2023) - Non-adversarial Training of Neural SDEs
- **DDIM Sampling**: Song et al. (2022) - Denoising Diffusion Implicit Models
- **Signature Methods**: Chevyrev & Oberhauser (2022) - Signature methods in machine learning