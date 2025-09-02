Awesome—let’s build you a tight, interview-ready problem set centered on **multiple uniform random variables**. I organized these by theme and difficulty so you can mix a few per prep session. I’m only giving statements (no solutions) so you can actually work them; each comes with a short nudge on what it’s testing.

---

# Core warm-ups (order stats, sums, min/max)

1. **Irwin–Hall tail:**
   Let $U_1,\dots,U_n\overset{iid}{\sim}\mathrm{Unif}(0,1)$. For a fixed $t\in[0,n]$, compute
   $\mathbb P\!\left(\sum_{i=1}^n U_i \ge t\right)$.
   *Concepts:* Irwin–Hall CDF / piecewise polynomials / symmetry at $n/2$.

2. **Range constraint:**
   With $U_{(1)}\le\cdots\le U_{(n)}$ the order statistics, find
   $\mathbb P\!\big(U_{(n)}-U_{(1)}\le \delta\big)$ for $\delta\in[0,1]$.
   *Concepts:* joint density of $(U_{(1)},U_{(n)})$, geometry.

3. **Largest gap:**
   Place $n$ i.i.d. $\mathrm{Unif}(0,1)$ points on $[0,1]$. Let $G_{\max}$ be the largest gap between consecutive ordered points (include edge gaps). Find $\mathbb E[G_{\max}]$ or tight bounds.
   *Concepts:* spacings $\sim$ Dirichlet; union bounds.

4. **Beta order stat:**
   Show $U_{(k)}\sim \mathrm{Beta}(k,n+1-k)$. Then compute $\mathbb E[U_{(k)}]$ and $\mathrm{Var}(U_{(k)})$.
   *Concepts:* order-stat densities; Beta moments.

5. **Sum stopping at 1 (classic):**
   Draw $U_1,U_2,\dots\stackrel{iid}{\sim}\mathrm{Unif}(0,1)$ until the partial sum first exceeds 1. Let $\tau=\min\{k:\sum_{i=1}^k U_i>1\}$. Find $\mathbb E[\tau]$.
   *Concepts:* combinatorics/geometry over the simplex; renewal flavor.

---

# Geometric probability in the unit square / triangle

6. **Triangle probability:**
   $U,V\sim\mathrm{Unif}(0,1)$ independent. Compute $\mathbb P(U+V\ge 1)$ and $\mathbb E[\max\{U,V\}]$.
   *Concepts:* unit square geometry; symmetry.

7. **Nonlinear boundary:**
   For $U,V$ as above, compute $\mathbb P\big(U\le V^2\big)$ and $\mathbb E[\mathbb 1\{U\le V^2\}]$.
   *Concepts:* integrate under a curve; conditioning on $V$.

8. **Distance in the square (lightweight version):**
   Pick two independent points $(U_1,V_1),(U_2,V_2)$ uniform on $[0,1]^2$. Derive an expression for $\mathbb E\big[\| (U_1,V_1)-(U_2,V_2)\|\big]$ (no need to fully simplify).
   *Concepts:* change of variables; symmetry; polar-ish tricks.

---

# Transformations & ratios

9. **Ratio of uniforms:**
   Let $U,V\stackrel{iid}{\sim}\mathrm{Unif}(0,1)$, independent. Find the density of $X=U/V$ on $[0,\infty)$ and compute $\mathbb P(X>c)$ for $c>0$.
   *Concepts:* Jacobians; piecewise integration.

10. **Product and min/max mix:**
    With iid uniforms as above, find the joint density of $(M,P)=(\max\{U,V\},\,UV)$.
    *Concepts:* mapping to triangular regions; support characterization.

---

# Spacings & Dirichlet structure

11. **Dirichlet spacings:**
    Let $U_{(0)}=0<U_{(1)}<\cdots<U_{(n)}<U_{(n+1)}=1$. Define spacings $S_i=U_{(i)}-U_{(i-1)}$.
    (a) Show $(S_1,\dots,S_{n+1})\sim\mathrm{Dirichlet}(\mathbf 1)$.
    (b) Compute $\mathbb P(\max_i S_i\le \delta)$ as an integral or bound it sharply.
    *Concepts:* symmetry; simplex volume; extreme of Dirichlet(1,…,1).

12. **Coverage by random arcs (intervals):**
    Place $m$ i.i.d. centers $C_i\sim\mathrm{Unif}([0,1])$; mark intervals $[C_i-\ell/2,C_i+\ell/2]\cap[0,1]$ for fixed $\ell\in(0,1]$. Give a formula (or bound) for expected covered length and the probability of full coverage when $\ell$ is small and $m$ grows.
    *Concepts:* inclusion–exclusion; Poissonization heuristics.

---

# Optimal stopping / decision problems (very “quanty”)

13. **One-shot accept/reject:**
    Observe $n$ i.i.d. $\mathrm{Unif}(0,1)$ sequentially. You may stop once to accept a value; payoff is the accepted value (or 0 if you never stop). Find the optimal threshold sequence $t_1,\dots,t_n$ and the optimal expected payoff.
    *Concepts:* dynamic programming / optimal stopping with known distribution.

14. **Maximizing the maximum under a budget:**
    You may sample up to $m$ i.i.d. $\mathrm{Unif}(0,1)$ draws at cost $c$ per draw. You can stop anytime and take the current max as payoff. Choose $m$ (possibly random) to maximize expected payoff minus cost. Characterize optimal policy vs $c$.
    *Concepts:* value of information; stopping with sampling cost.

15. **Adverse selection toy model:**
    A counterparty’s reservation price $V\sim \mathrm{Unif}(0,1)$. You post a take-it-or-leave-it quote $q\in[0,1]$. Trade occurs if $V\ge q$, and your P\&L is $q-V$.
    (a) Compute expected P\&L as a function of $q$.
    (b) Suppose you get $k$ i.i.d. noisy signals $S_i=V+\varepsilon_i$ with $\varepsilon_i\sim \mathrm{Unif}(-\eta,\eta)$. Find the Bayes-optimal $q$ given the signals.
    *Concepts:* conditioning; posterior from uniform noise; profit maximization.

---

# Estimation & inference with uniforms

16. **MLE and risk for $\mathrm{Unif}(0,\theta)$:**
    Given $X_1,\dots,X_n\overset{iid}{\sim}\mathrm{Unif}(0,\theta)$ with unknown $\theta>0$:
    (a) Derive the MLE $\hat\theta$.
    (b) Compute its bias and MSE.
    (c) Produce an unbiased estimator with smaller (or compare) MSE.
    *Concepts:* likelihood with bounded support; Lehmann–Scheffé via max.

17. **Goodness-of-fit via spacings:**
    Given $X_1,\dots,X_n\overset{iid}{\sim}F$ continuous, define $U_i=F(X_i)$ and order them. Show how a test based on scaled spacings $(n+1)S_i$ can detect non-uniformity; derive its asymptotic null distribution.
    *Concepts:* probability integral transform; Dirichlet to $\chi^2$-like tests.

18. **Two-sample test on $[0,1]$:**
    Samples $U_1,\dots,U_m$ and $V_1,\dots,V_n$ are i.i.d. $\mathrm{Unif}(0,1)$ under $H_0$. Under $H_1$, $V\sim \mathrm{Unif}(0,1+\delta)$ truncated to $[0,1]$ (density proportional to $1+\delta\mathbb 1$). Construct a distribution-free test using combined order stats and derive (exact or asymptotic) critical values.
    *Concepts:* rank tests; permutation logic.

---

# Dependence with uniform marginals (copulas)

19. **FGM copula dependence:**
    Construct $(U,V)$ with uniform marginals and joint CDF $C(u,v)=uv\!\left[1+\theta(1-u)(1-v)\right]$ for $\theta\in[-1,1]$.
    (a) Compute $\mathrm{Cov}(U,V)$ and Spearman’s $\rho_S$.
    (b) For $g(x)=\mathbb 1\{x>\tfrac12\}$, compute $\mathrm{Cov}(g(U),g(V))$.
    *Concepts:* copulas; dependence without changing marginals.

20. **Common-factor uniforms:**
    Let $Z\sim\mathrm{Unif}(0,1)$, and conditional on $Z$, draw $U_i\sim\mathrm{Unif}(\max\{0,Z-\alpha\},\min\{1,Z+\alpha\})$ independently for $i=1,\dots,k$.
    (a) Show marginals are uniform.
    (b) Compute $\mathrm{Corr}(U_i,U_j)$ as a function of $\alpha$.
    *Concepts:* conditional constructions; exchangeable dependence.

---

# High-dim / concentration flavored

21. **Max vs average (extremes):**
    For $U_1,\dots,U_n\overset{iid}{\sim}\mathrm{Unif}(0,1)$, find sharp bounds (or asymptotics) for
    $\mathbb P\!\left(\max_i U_i \le \frac{1}{2}+\varepsilon\right)$ and
    $\mathbb P\!\left(\frac{1}{n}\sum U_i \ge \frac12+\varepsilon\right)$ for fixed $\varepsilon>0$.
    *Concepts:* exact calc for max; Hoeffding/Chernoff for the mean.

22. **Empirical CDF sup-norm:**
    Let $F_n(x)=\frac{1}{n}\sum_{i=1}^n \mathbf 1\{U_i\le x\}$. Study $D_n=\sup_{x\in[0,1]} |F_n(x)-x|$:
    (a) Give distribution-free bounds for $\mathbb P(D_n>\delta)$.
    (b) Find the limiting distribution of $\sqrt n\,D_n$.
    *Concepts:* DKW inequality; Kolmogorov distribution.

---

# Simulation / MC estimators (interview coding-friendly)

23. **Hit-and-miss area via uniforms:**
    Use $(U,V)\sim\mathrm{Unif}([0,1]^2)$ to estimate $\int_0^1 f(x)\,dx$ for $f\in[0,1]$.
    (a) Construct an unbiased estimator using indicator $\mathbb 1\{V\le f(U)\}$.
    (b) Compute its variance and how control variates using $\mathbb 1\{V\le U\}$ change it.
    *Concepts:* unbiasedness; variance reduction.

24. **Acceptance–rejection with uniforms:**
    Target density $p(x)\propto e^{-x}\mathbf 1_{\{x\ge 0\}}$. Using proposal $Q=\mathrm{Unif}(0,b)$:
    (a) Find the minimal $M$ s.t. $p(x)\le M q(x)$ on $[0,b]$.
    (b) Optimize $b$ to minimize rejection rate.
    *Concepts:* bounding constants; optimizing acceptance probability.

---

# Puzzles / quick hitters

25. **Top-two exceedance:**
    Let $U_1,U_2,U_3\stackrel{iid}{\sim}\mathrm{Unif}(0,1)$. What is $\mathbb P(U_{(2)}+U_{(3)}\ge 1)$?
    *Concepts:* order-stat geometry; symmetry.

26. **At least one close pair:**
    For $n$ iid uniforms, bound $\mathbb P\big(\min_{i\ne j}|U_i-U_j|\le \delta\big)$ tightly as $n$ grows.
    *Concepts:* union bounds vs. Poisson heuristic on spacings.

27. **Condition on the max:**
    Given $M=U_{(n)}=m$, find the conditional joint law of the remaining $n-1$ points and compute $\mathbb E\!\left[\sum_{i=1}^{n-1} U_{(i)}\mid M=m\right]$.
    *Concepts:* self-similarity; conditional independence on scaled interval.

---

## How to use these

* Pick **2–3 Core**, **1 Decision**, and **1 Estimation/Inference** problem per practice block.
* For “quant flavor,” prioritize 13–15 and 23–24.
* For theory depth, lean on 11–12, 16–22.

If you want, I can turn this into a clean printable PDF or add brief “solution sketches” for a subset—just say which numbers you want first.
