# Rollout equivalence

------

### **Problem Statement**

Let $F_\theta$ denote a deterministic transformer with **strict causal masking**, such that for any finite sequence $x_{1:T}$,
$$
[F_\theta(x_{1:T})]_t = F_\theta(x_{<t}),
 \qquad
 x_{<t} = [x_1, \ldots, x_{t-1}].
$$
We define two algorithms that generate sequences using $F_\theta$.

------

#### **Algorithm A — Iterative Rollout**

Given an initial sequence $(x^{(0)}_{1:T})$ (e.g., the ground truth),
 Algorithm A repeatedly applies $F_\theta$ to its own previous output:
$$
\begin{cases}
 x^{(0)}_{1:T} \text{ (given)} \\
 x^{(r+1)}_t = F_\theta(x^{(r)}_{<t}), \quad \forall, r \ge 0,\ 1 \le t \le T.
 \end{cases}
$$
Thus, each rollout iteration $r \mapsto r+1$ updates all time indices causally using the preceding rollout.

------

#### **Algorithm B — Autoregressive Generation**

Starting from the first true token $x^{(0)}_1$, Algorithm B generates new tokens sequentially:
$$
\begin{cases}
 \hat{x}_1 = x^{(0)}_1,\\
 \hat{x}_t = F_\theta(\hat{x}_{<t}), \quad \forall, t \ge 2.
 \end{cases}
$$
Each new token is produced by applying $F_\theta$ once to the prefix of already-generated tokens.

------

### **Claim**

For every time index $t \in {1, \ldots, T}$,
 the two algorithms produce identical outputs **along the causal diagonal**:
$$
x^{(t)}_t = \hat{x}_t.
$$
Equivalently, the token at position (t) after (t) iterations of Algorithm A is equal to the token at position (t) generated after (t) autoregressive steps in Algorithm B.



