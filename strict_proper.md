# Strict Properness of Signature Scoring Function

## Overview

This document analyzes whether the signature scoring function proposed in `method.tex` is **strictly proper**, meaning it assigns the lowest expected score when the prediction distribution equals the true distribution.

## Mathematical Framework

### Definition: Strict Properness
A scoring rule S(P, y) is **strictly proper** if:
```
E_{Y~Q}[S(P, Y)] ≥ E_{Y~Q}[S(Q, Y)]
```
with equality if and only if P = Q, where:
- P is the predicted distribution
- Q is the true distribution  
- Y is a realization from the true distribution Q

### Our Signature Score Implementation

From `method.tex`, the empirical signature score is:
```
Ŝ_λ,sig(P_θ(·|X_t,t), X_0) = (1/m(m-1)) ∑_{i≠j} k_sig(X̃_0^(i), X̃_0^(j)) - (2/m) ∑_i k_sig(X̃_0^(i), X_0)
```

Where:
- `X̃_0^(i)` are samples from predicted distribution P_θ
- `X_0` is the true target sample
- `k_sig` is the signature kernel
- `m` is the number of generated samples

### Mathematical Analysis

#### 1. Population-Level Score
The population version of our score is:
```
S_λ,sig(P, Y) = (λ/2) E_{X,X'~P}[k_sig(X,X')] - E_{X~P}[k_sig(X,Y)]
```

#### 2. Expected Score Under True Distribution
When Y ~ Q (true distribution), the expected score is:
```
E_{Y~Q}[S_λ,sig(P, Y)] = (λ/2) E_{X,X'~P}[k_sig(X,X')] - E_{X~P, Y~Q}[k_sig(X,Y)]
```

#### 3. Optimal Prediction (P = Q)
When P = Q, the expected score becomes:
```
E_{Y~Q}[S_λ,sig(Q, Y)] = (λ/2) E_{X,X'~Q}[k_sig(X,X')] - E_{X,Y~Q}[k_sig(X,Y)]
                        = (λ/2) E_{X,X'~Q}[k_sig(X,X')] - E_{X,X'~Q}[k_sig(X,X')]
                        = ((λ/2) - 1) E_{X,X'~Q}[k_sig(X,X')]
```

#### 4. Strict Properness Condition
For strict properness, we need:
```
E_{Y~Q}[S_λ,sig(P, Y)] ≥ E_{Y~Q}[S_λ,sig(Q, Y)]
```

This translates to:
```
(λ/2) E_{X,X'~P}[k_sig(X,X')] - E_{X~P, Y~Q}[k_sig(X,Y)] ≥ ((λ/2) - 1) E_{X,X'~Q}[k_sig(X,X')]
```

Rearranging:
```
E_{X~P, Y~Q}[k_sig(X,Y)] - E_{X,X'~Q}[k_sig(X,X')] ≥ (λ/2)[E_{X,X'~P}[k_sig(X,X')] - E_{X,X'~Q}[k_sig(X,X')]]
```

## Critical Issues Identified

### ⚠️ Issue 1: Lambda Parameter Constraint
For strict properness, we typically need **λ ∈ [0, 2]** and the inequality above to hold. However, our analysis shows:

1. **When λ = 0**: Score becomes `-E[k_sig(X,Y)]`, which is the negative kernel mean
2. **When λ = 2**: Score becomes `E[k_sig(X,X')] - E[k_sig(X,Y)]`, which is the original MMD-style score
3. **Our choice λ = 0.1**: Very small weight on diversity term

### ⚠️ Issue 2: Empirical vs Population Discrepancy
Our implementation uses:
```
(1/m(m-1)) ∑_{i≠j} k_sig(X̃_i, X̃_j) - (2/m) ∑_i k_sig(X̃_i, X_0)
```

But the population score should be:
```
(λ/2) E[k_sig(X,X')] - E[k_sig(X,Y)]
```

**Key discrepancy**: The empirical version uses coefficient `1/m(m-1)` for cross-terms and `2/m` for target terms, which doesn't directly correspond to `λ/2` weighting.

### ⚠️ Issue 3: Kernel Properties Assumption
Strict properness requires the signature kernel to satisfy certain regularity conditions:
1. **Positive definiteness** ✅ (verified in our tests)
2. **Integrability** ✅ (finite kernel values)
3. **Characteristic property** ❓ (needs verification)

## Empirical Verification Needed

### Test 1: Self-Consistency
Does our score achieve minimum when P = Q?
- Generate samples from same distribution
- Compute signature score
- Verify it's lower than scores from different distributions

### Test 2: Monotonicity
As distributions become more different, does the score increase monotonically?

### Test 3: Lambda Sensitivity
How does the choice of λ affect strict properness?

## Implementation Analysis

### Our Current Implementation Issues

1. **Inconsistent Weighting**: 
   - Theory: `(λ/2) E[k(X,X')] - E[k(X,Y)]`
   - Implementation: Uses different empirical weights

2. **Lambda Choice**:
   - Theory: Suggests λ should balance diversity vs similarity
   - Practice: λ = 0.1 heavily emphasizes similarity term

3. **Empirical Estimation**:
   - Small sample sizes (m = 8) may not accurately estimate population expectations
   - Cross-validation needed to verify empirical score converges to population score

## Mathematical Concerns

### Question 1: Is Our Formula Correct?
The formula in `method.tex`:
```
(1/m(m-1)) ∑_{i≠j} k_sig(X̃_i, X̃_j) - (2/m) ∑_i k_sig(X̃_i, X_0)
```

Should this be:
```
(λ/(2m(m-1))) ∑_{i≠j} k_sig(X̃_i, X̃_j) - (1/m) ∑_i k_sig(X̃_i, X_0)
```

### Question 2: Characteristic Kernel
For strict properness, the signature kernel must be **characteristic** (i.e., k_sig(X,Y) uniquely determines the distribution). This needs mathematical verification.

### Question 3: Finite Sample Effects
With finite samples (m = 8), does the empirical score maintain strict properness? Or do we need bias corrections?

## Recommendations for Verification

### Immediate Tests
1. **Implement corrected empirical formula** with proper λ weighting
2. **Test with known distributions** (Gaussian, uniform, etc.)
3. **Verify minimum at P = Q** across different distribution pairs
4. **Study λ parameter effects** on strict properness

### Mathematical Investigation
1. **Prove/verify signature kernel is characteristic**
2. **Derive finite-sample corrections** for empirical score
3. **Establish λ parameter bounds** for strict properness
4. **Compare with other proper scoring rules** (energy score, MMD, etc.)

## Conclusion

**Current Status**: ⚠️ **Strict properness is NOT verified**

The implementation shows promising empirical behavior (better scores for similar distributions), but several theoretical gaps need addressing:

1. **Formula consistency** between theory and implementation
2. **Lambda parameter** theoretical justification  
3. **Finite sample** bias corrections
4. **Characteristic property** of signature kernel

**Next Steps**: Implement systematic tests to verify strict properness empirically and address the theoretical gaps identified above.

---

*This analysis reveals that while our signature scoring approach shows practical promise, the strict properness property requires further theoretical and empirical validation.*
