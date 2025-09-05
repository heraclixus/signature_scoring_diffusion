#!/bin/bash

# Test all 6 initialization methods with conservative hyperparameters for stability
# This script systematically tests each initialization method to find the most stable one

echo "🧪 Testing All 6 Initialization Methods with Conservative Settings"
echo "================================================================="
echo "Conservative settings: small LR, high dropout, long warmup, small gains"
echo ""

# Conservative base parameters for all tests
BASE_ARGS="--learning_rate 1e-5 --dropout 0.2 --warmup_epochs 100 --num_epochs 1000 --batch_size 32 --lambda_param 0.3 --num_samples 16 --save_logs"

# Test 1: Xavier Uniform
echo "📋 Test 1/6: Xavier Uniform Initialization"
python baseline_sigscore.py \
    --init_method xavier_uniform \
    --init_gain 0.05 \
    --experiment_name "init_xavier_uniform" \
    $BASE_ARGS

echo -e "\n" 

# Test 2: Xavier Normal
echo "📋 Test 2/6: Xavier Normal Initialization"
python baseline_sigscore.py \
    --init_method xavier_normal \
    --init_gain 0.05 \
    --experiment_name "init_xavier_normal" \
    $BASE_ARGS

echo -e "\n"

# Test 3: Kaiming Uniform
echo "📋 Test 3/6: Kaiming Uniform Initialization (ReLU optimized)"
python baseline_sigscore.py \
    --init_method kaiming_uniform \
    --init_gain 0.05 \
    --experiment_name "init_kaiming_uniform" \
    $BASE_ARGS

echo -e "\n"

# Test 4: Kaiming Normal
echo "📋 Test 4/6: Kaiming Normal Initialization (ReLU optimized)"
python baseline_sigscore.py \
    --init_method kaiming_normal \
    --init_gain 0.05 \
    --experiment_name "init_kaiming_normal" \
    $BASE_ARGS

echo -e "\n"

# Test 5: Normal Distribution
echo "📋 Test 5/6: Normal Distribution Initialization"
python baseline_sigscore.py \
    --init_method normal \
    --init_gain 0.02 \
    --experiment_name "init_normal" \
    $BASE_ARGS

echo -e "\n"

# Test 6: Orthogonal
echo "📋 Test 6/6: Orthogonal Initialization"
python baseline_sigscore.py \
    --init_method orthogonal \
    --init_gain 0.05 \
    --experiment_name "init_orthogonal" \
    $BASE_ARGS

echo -e "\n"

echo "✅ All 6 initialization methods tested with conservative settings!"
echo ""
echo "📊 Results Summary:"
echo "- Each test used: LR=1e-5, dropout=0.2, warmup=100 epochs, λ=0.3"
echo "- Training logs: training_log_init_*.json"
echo "- Plots: *_init_*.png with full configuration in filename"
echo ""
echo "🔍 To analyze results:"
echo "1. Check training logs for loss convergence"
echo "2. Look for 'Strict proper: ✅ YES' in output"
echo "3. Monitor 'Loss decreasing' vs 'WARNING: Loss not decreasing'"
echo "4. Compare final loss values across methods"
echo ""
echo "💡 Recommended next steps:"
echo "- Identify the most stable initialization method"
echo "- Run longer training (--num_epochs 1000) with best method"
echo "- Try slightly higher learning rates with best initialization"
