#!/bin/bash

# Test different kernel configurations for signature scoring stability
# This script tests various kernels (RBF, exponential, Lp norm, etc.) with conservative settings

echo "🔧 Testing Different Kernel Scoring Configurations"
echo "=================================================="
echo "Testing 3 kernel types: RBF, Exponential, Lp Norm"
echo "Conservative settings: small LR, high dropout, long warmup"
echo ""

# Conservative base parameters for all tests
BASE_ARGS="--learning_rate 1e-5 --dropout 0.2 --warmup_epochs 100 --num_epochs 300 --batch_size 8 --lambda_param 0.5 --num_samples 8 --save_logs --init_method xavier_uniform --init_gain 0.05"

# Test 1: RBF Kernel (γ=1.0)
echo "📋 Test 1/3: RBF Kernel (Gaussian)"
python baseline_kernels_score.py \
    --kernel_type rbf \
    --gamma 1.0 \
    --experiment_name "kernel_rbf" \
    $BASE_ARGS

echo -e "\n" 

# Test 2: Exponential Kernel (γ=1.0)
echo "📋 Test 2/3: Exponential Kernel"
python baseline_kernels_score.py \
    --kernel_type exponential \
    --gamma 1.0 \
    --experiment_name "kernel_exponential" \
    $BASE_ARGS

echo -e "\n"

# Test 3: L2 Norm Kernel (p=2.0)
echo "📋 Test 3/3: L2 Norm Kernel"
python baseline_kernels_score.py \
    --kernel_type lp_norm \
    --p_norm 2.0 \
    --experiment_name "kernel_l2_norm" \
    $BASE_ARGS

echo -e "\n"


echo "✅ All 3 kernel types tested with conservative settings!"
echo ""
echo "📊 Results Summary:"
echo "- Each test used: LR=1e-5, dropout=0.2, warmup=100 epochs, λ=0.5"
echo "- Training logs: training_log_*kernel_*.json"
echo "- Plots: *kernel_*.png with full configuration in filename"
echo ""
echo "🔍 To analyze results:"
echo "1. Run: python test_kernel_properness.py (test strict properness)"
echo "2. Run: python analyze_results.py (compare training dynamics)"
echo "3. Check training logs for loss convergence patterns"
echo "4. Look for 'Strict proper: ✅ YES' in training output"
echo ""
echo "💡 Expected insights:"
echo "- RBF and exponential kernels should be most stable"
echo "- L2 norm should behave similarly to RBF"
echo "- Linear kernel may not be strictly proper"
echo "- Polynomial kernel behavior depends on degree"
