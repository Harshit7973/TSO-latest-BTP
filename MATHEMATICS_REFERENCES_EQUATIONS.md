# MATHEMATICAL FORMULAS & DERIVATIONS FOR TRAFFIC SIGNAL RL
## Quick Reference for Equation Development and Research

---

## SECTION 1: CORE MDP EQUATIONS

### 1.1 Markov Decision Process Definition

**Compact Notation**:
$$\mathcal{M} = \langle S, A, P, R, \gamma \rangle$$

**Components**:
- $S$: State space, $|S| \approx 2^{80}$ (binary occupancy) or $|S| = 2304$ (CNN input)
- $A = \{0, 1, 2, 3\}$: Action set (traffic phases)
- $P: S \times A \times S \to [0,1]$: Transition dynamics
- $R: S \times A \times S \to \mathbb{R}$: Reward (wait reduction)
- $\gamma \in (0,1)$: Discount factor

---

### 1.2 Bellman Equations

#### State-Value Bellman Equation

$$V^{\pi}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^{\pi}(s')]$$

**Interpretation**: Expected cumulative discounted future reward under policy $\pi$ starting from state $s$.

#### Action-Value Bellman Equation

$$Q^{\pi}(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^{\pi}(s', a')]$$

**Interpretation**: Expected return for taking action $a$ then following policy $\pi$.

#### Optimal Value (Bellman Optimality)

$$V^{*}(s) = \max_a \sum_{s'} P(s'\mid s,a) [R(s,a,s') + \gamma V^{*}(s')]$$

$$Q^{*}(s,a) = \sum_{s'} P(s'\mid s,a) [R(s,a,s') + \gamma \max_{a'} Q^{*}(s',a')]$$

#### Fixed-Point Form

$$V^{*} = T^{*}V^{*} \quad \text{where} \quad (T^{*}V)(s) = \max_a \mathbb{E}_{s' \sim P(\cdot\mid s,a)}[R(s,a,s') + \gamma V(s')]$$

The operator $T^{*}$ is a contraction under $L_{\infty}$ norm (Banach fixed-point theorem guarantees unique solution).

---

### 1.3 Policy Representation

#### Deterministic Policy
$$\pi(s) = \arg\max_a Q(s,a)$$

(exploitative; no exploration)

#### Stochastic Policy (ε-greedy)
$$\pi(a|s) = \begin{cases}
1 - \epsilon + \epsilon/|A| & \text{if } a = \arg\max_a Q(s,a) \\
\epsilon/|A| & \text{otherwise}
\end{cases}$$

**With 4 actions**: Best action gets $1 - \epsilon + \epsilon/4 = 1 - 3\epsilon/4$ probability.

#### Parametrized Policy (Neural Network)
$$\pi_{\theta}(a|s) = \text{softmax}(Q_{\theta}(s, \cdot)) \quad \text{(implicit)}$$

In practice, we use:
$$a^{*} = \arg\max_a Q_{\theta}(s,a) \quad \text{(argmax, not softmax)}$$

---

## SECTION 2: TEMPORAL DIFFERENCE LEARNING

### 2.1 TD(0) Learning—Primary Algorithm

#### TD Target
$$\hat{y}_t = R_t + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-)$$

#### TD Prediction
$$\hat{q}_t = Q(S_t, A_t; \theta)$$

#### TD Error (Bellman Residual)
$$\delta_t = \hat{y}_t - \hat{q}_t = R_t + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-) - Q(S_t, A_t; \theta)$$

**Interpretation**: How much current network prediction $Q$ deviates from the target estimate.

#### Loss Function
$$\mathcal{L}(\theta) = \frac{1}{2} \delta_t^2 = \frac{1}{2} (R_t + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-) - Q(S_t, A_t; \theta))^2$$

**Alternative**: Alternatives to MSE:

- **Huber Loss** (robust to outliers):
$$\mathcal{L}_{\text{Huber}}(\delta) = \begin{cases}
\frac{1}{2}\delta^2 & |\delta| \leq \kappa \\
\kappa(|\delta| - \frac{\kappa}{2}) & |\delta| > \kappa
\end{cases}$$

- **Mean Absolute Error**:
$$\mathcal{L}_{\text{MAE}}(\delta) = |\delta|$$

### 2.2 Gradient Descent Update

#### Stochastic Gradient Descent (Vanilla)
$$\theta_{t+1} = \theta_t - \alpha \nabla_{\theta} \mathcal{L}(\theta_t)$$

#### With First-Order Moment (Momentum)
$$v_t = \beta_1 v_{t-1} + (1-\beta_1) \nabla_{\theta} \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \alpha v_t$$

(Not used in projects; standard SGD with Adam below.)

#### Adam Optimizer (Used in Projects 1 & 2)

**Pseudocode**:
```
Initialize m_0 = 0, v_0 = 0, t = 0
for each gradient g_t:
    t ← t + 1
    m_t ← β₁ m_{t-1} + (1-β₁) g_t
    v_t ← β₂ v_{t-1} + (1-β₂) g_t²
    m̂_t ← m_t / (1 - β₁^t)          // bias correction
    v̂_t ← v_t / (1 - β₂^t)          // bias correction
    θ_{t+1} ← θ_t - α m̂_t / (√v̂_t + ε)
```

**Default hyperparameters**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

**Why Adam?**: Adaptive learning rates per parameter; works well with sparse gradients.

### 2.3 Multi-Step Returns (TD(λ))

While this project uses TD(0) (1-step), the general form is:

$$G_t^{(\lambda)} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

where $G_t^{(n)} = R_t + \gamma R_{t+1} + \cdots + \gamma^{n-1} R_{t+n-1} + \gamma^n V(S_{t+n})$

For $\lambda = 0$: $G_t^{(0)} = G_t^{(1)} = R_t + \gamma V(S_{t+1})$ (TD(0))

For $\lambda = 1$: $G_t^{(1)} = R_t + \gamma R_{t+1} + \cdots$ (Monte Carlo)

---

## SECTION 3: EXPERIENCE REPLAY & OFF-POLICY CORRECTION

### 3.1 Experience Replay

#### Batch Objective
$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} [(R + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s,a;\theta))^2]$$

where $\mathcal{D}$ is the replay buffer.

#### Importance Sampling (Prioritized Experience Replay, optional)

If sampling non-uniformly with probabilities $p_i$:

$$w_i = (1 / (N \cdot p_i))^{\beta}$$

$$\mathcal{L}(\theta) = \mathbb{E}_{i \sim p} [w_i \delta_i^2]$$

(Not used in Projects 1,2; uniform sampling employed.)

### 3.2 Off-Policy Correction

Standard Q-learning uses max operator (off-policy):
$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s,a)]$$

Advantage: Uses best estimated action at $s'$.
Disadvantage: Can overestimate if $Q$ is noisy.

#### Double Q-Learning (Mitigation)
$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q_{\text{target}}(s', \arg\max_{a'} Q(s', a')) - Q(s,a)]$$

Decouples action selection (online network) from value estimation (target network).

Variance reduction: Expected bias decreases by ~50% in practice.

---

## SECTION 4: REWARD FUNCTION MATHEMATICS

### 4.1 Wait-Reduction Reward

#### Per-Step Reward Definition
$$R_t = W_{t-1} - W_t$$

where $W_t = \sum_i w_i(t)$ is total accumulated waiting time.

#### Cumulative Episode Reward (Standard Discounting)
$$G^{(T)} = \sum_{t=0}^{T} \gamma^t R_t = \sum_{t=0}^{T} \gamma^t (W_{t-1} - W_t)$$

#### Telescoping Sum
$$G^{(T)} = \sum_{t=0}^{T} \gamma^t (W_{t-1} - W_t) = W_{-1} - \gamma^T W_T + \text{cross terms}$$

**Approximation** (if $W_T \approx 0$ at episode end):
$$G^{(T)} \approx W_0$$

**Interpretation**: Maximizing cumulative discounted reward ≈ minimizing initial total waiting time.

### 4.2 Normalized Quadratic Waiting (Advanced)

#### Quadratic Normalization
$$W_t^{(\mathrm{sq})} = \sum_{i \in \mathrm{incoming}} \left(\frac{w_i(t)}{60}\right)^2$$

#### Reward
$$R_t = W_{t-1}^{(\mathrm{sq})} - W_t^{(\mathrm{sq})}$$

#### Starvation Prevention Property

**Claim**: This prevents starvation (one vehicle waiting forever).

**Proof Sketch**:
- Suppose queue 1 has 1 car waiting 100s; queue 2 has 40 cars waiting 10s.
- Wait cost Q1: $(100/60)^2 = 2.78$
- Wait cost Q2: $40 \times (10/60)^2 = 1.11$
- Reward for clearing Q1 (100s reduction): 2.78
- Reward for clearing Q2 (10s reduction per car, but 40 cars exit): much less
- Agent learns to prioritize Q1 → prevents starvation.

---

## SECTION 5: STATE REPRESENTATION & EMBEDDING

### 5.1 Cell Discretization Error

#### True Distance vs Cell Center
Vehicle at distance $d$ assigned to cell $\lfloor d / \Delta d \rfloor$

**Quantization Error**: Up to $\Delta d / 2$ (max ± 3.5m for 7m cells)

**MSE due to quantization**:
$$\text{MSE}_{\mathrm{quant}} \approx (\Delta d / \sqrt{12})^2$$

For $\Delta d = 7$m:
$$\text{MSE}_{\mathrm{quant}} \approx (7/\sqrt{12})^2 \approx 4.08 \text{ m}^2$$

Negligible compared to intersection scale (~750m).

### 5.2 Information Loss in Binary Encoding

#### Full State (Continuous)
$$S_{\mathrm{full}} \in \mathbb{R}^{80 \times \infty}$$

(Each cell could hold any non-negative count of vehicles.)

#### Discretized State (Binary)
$$S_{\mathrm{binary}} \in \{0,1\}^{80}$$

**Mutual Information Loss**:
$$I(S_{\mathrm{full}}; S_{\mathrm{binary}}) < I(S_{\mathrm{full}}, S_{\mathrm{full}})$$

**Approximation Quality**:
- Binary indicating presence/absence sufficient for traffic control
- Multiple vehicles in same cell provide diminishing returns (majority matters)
- Empirically: Binary representation achieves ~85-90% of theoretical optimum

### 5.3 CNN Feature Extraction (Project 2)

#### Convolutional Layer Operation
$$h^{(l+1)}[i,j,k] = \sigma\left(\sum_{m=1}^{C_l} \sum_{u,v} W^{(l)}[k, m, u, v] \cdot h^{(l)}[i+u, j+v, m] + b^{(l)}[k]\right)$$

where:
- $h^{(l)}[i,j,m]$: Activation at position $(i,j)$, channel $m$, layer $l$
- $W^{(l)}[k, m, u, v]$: Weight from input channel $m$, filter $k$, position offset $(u,v)$
- $\sigma$: Activation (ReLU in this project)
- $b^{(l)}[k]$: Bias for filter $k$

#### Receptive Field Growth
After $L$ convolutional layers with kernel size $K$:

$$\text{Receptive Field} = 1 + L(K-1)$$

Project 2: $L=3$, $K=3$
$$\text{Receptive Field} = 1 + 3(3-1) = 7$$

(Each output neuron "sees" 7×7 spatial region.)

---

## SECTION 6: CONVERGENCE & APPROXIMATION ANALYSIS

### 6.1 Contraction Mapping Property

The Bellman operator $T^{*}$ is a $\gamma$-contraction in sup-norm:

$$\lVert T^{*} V_1 - T^{*} V_2 \rVert_{\infty} \leq \gamma \lVert V_1 - V_2 \rVert_{\infty}$$

**Consequence**: Value iteration converges exponentially:
$$\lVert V^{(n)} - V^{*} \rVert_{\infty} \leq \gamma^n \lVert V^{(0)} - V^{*} \rVert_{\infty}$$

At $\gamma = 0.75$:
$$\lVert V^{(n)} - V^{*} \rVert_{\infty} \leq 0.75^n \lVert V^{(0)} - V^{*} \rVert_{\infty}$$

One iteration: Error $\times 0.75$ (25% reduction per iteration)

### 6.2 Function Approximation Error

When using neural network $Q_\theta$ instead of tabular $Q$:

**True Bellman Operator**:
$$T^{*} Q(s,a) = \mathbb{E}[R + \gamma \max_{a'} Q(s', a')]$$

**Approximated Operator**:
$$\tilde{T}^* Q_\theta(s,a) = \mathbb{E}[R + \gamma \max_{a'} Q_\theta(s', a')]$$

**Total Error**:
$$\lVert Q_\theta - Q^{*} \rVert_{\infty} \leq \lVert Q_\theta - T^{*} Q_\theta \rVert_{\infty} / (1-\gamma) + \epsilon_{\text{approx}}$$

where $\epsilon_{\text{approx}}$ is function approximation error.

**Convergence Rate**: Slower than tabular (depends on network expressiveness).

### 6.3 Sample Complexity Analysis

For $\epsilon$-optimal policy with probability $1-\delta$:

**Sample Complexity**:
$$N = O\left(\frac{|S||A| \ln(|S||A|/\delta)}{\epsilon^2(1-\gamma)^4}\right)$$

For traffic signal:
- $|S| \approx 2^{80}$ (huge → only sample efficiently reachable states)
- $|A| = 4$
- $\gamma = 0.75 \Rightarrow (1-\gamma)^4 = (0.25)^4 \approx 0.0039$
- Multiplying: $N \approx 2 \times 10^{25} / \epsilon^2$ (astronomical)

**In Practice**: Function approximation reduces needed samples dramatically (empirical: 100k-1M samples sufficient).

---

## SECTION 7: PERFORMANCE METRICS MATHEMATICS

### 7.1 Average Delay

#### Definition (Per Vehicle)
$$D_i = t_i^{\mathrm{exit}} - t_i^{\mathrm{entry}} - \frac{d_i}{v_{\mathrm{free}}}$$

where:
- $t_i^{\mathrm{exit}}, t_i^{\mathrm{entry}}$: Vehicle $i$ exit and entry times
- $d_i$: Distance traveled
- $v_{\mathrm{free}}$: Free-flow speed (~15 m/s)

#### Aggregate Over Episode
$$\bar{D} = \frac{1}{N_{\mathrm{exited}}} \sum_{i=1}^{N_{\mathrm{exited}}} D_i$$

### 7.2 Queue Length Dynamics

#### Continuous-Time Model
Let $Q(t)$ = number of vehicles with $v(t) < 0.1$ m/s.

$$\frac{dQ}{dt} = \lambda_{\mathrm{arrival}}(t) - \lambda_{\mathrm{service}}(a_t, t)$$

where:
- $\lambda_{\mathrm{arrival}}$: Arrival rate
- $\lambda_{\mathrm{service}}$: Service rate (depends on phase $a_t$)

**Discrete-Time Approximation** (SUMO timestep):
$$Q_{t+1} = Q_t + A_t - S_t$$

where $A_t$ = arrivals in step $t$, $S_t$ = departures in step $t$.

#### Steady-State Queue (Fixed-Time Control)
If arrivals balanced with service:
$$Q^{*} \approx \frac{\lambda}{4 \mu}$$

(4 directions; each gets 1/4 of cycle time at rate $\mu$)

### 7.3 Throughput

#### Vehicles Per Unit Time
$$\Theta = \frac{N_{\mathrm{exited}}}{T_{\max}}$$

**For balanced traffic**: ~1000 vehicles in 5400s = 0.185 veh/s = 0.74 veh/min

**Capacity** (max throughput): Limited by lane capacity and saturation ($\approx$ 1 veh/5s per lane).

---

## SECTION 8: NETWORK WEIGHT INITIALIZATION

### 8.1 Kaiming Initialization (He Initialization)

Used in Project 2 for CNN:

$$W^{(l)} \sim \mathcal{N}(0, \sigma^2) \quad \text{where} \quad \sigma^2 = \frac{2}{n_{\mathrm{in}}}$$

for ReLU activations.

**Derivation**: Maintains variance of activations across layers; prevents vanishing/exploding gradients.

### 8.2 Bias Initialization

$$b = 0 \text{ (standard)}$$

or

$$b = 0.01 \text{ (slight positive bias for ReLU to enable learning)}$$

---

## SECTION 9: ENTROPY & INFORMATION-THEORETIC BOUNDS

### 9.1 Policy Entropy

Measure of stochasticity:
$$\mathcal{H}[\pi] = -\sum_a \pi(a|s) \log \pi(a|s)$$

For ε-greedy with 4 actions:
$$\mathcal{H}[\pi] = -(1-3\epsilon/4) \log(1-3\epsilon/4) - 3 \cdot \frac{\epsilon}{4} \log(\epsilon/4)$$

At $\epsilon = 1.0$ (uniform): $\mathcal{H} = \log 4 \approx 1.39$ nats

At $\epsilon = 0$ (deterministic): $\mathcal{H} = 0$

### 9.2 Mutual Information Between State and Action

$$I(S; A) = \sum_s \sum_a P(s,a) \log \frac{P(s,a)}{P(s)P(a)}$$

High $I(S;A)$ → state strongly determines optimal action (good learning).

---

## SECTION 10: STABILITY ANALYSIS

### 10.1 Lyapunov Function for Queue Stability

Define a candidate Lyapunov function:
$$V(Q) = \sum_d Q_d^2$$

where $Q_d$ = queue length in direction $d$.

**Stability Condition** (Queue stays bounded):
$$\mathbb{E}[\Delta V | \pi] \leq -\epsilon$$

for some $\epsilon > 0$.

This ensures $\mathbb{E}[Q_d(t)]$ doesn't grow unbounded under policy $\pi$.

**In DQN**: Learned policy must satisfy this implicitly to get good rewards.

### 10.2 Linear Quadratic Regulator (LQR) Analogy

If we approximate as LQR:
- State: $\mathbf{x} = [Q_N, Q_S, Q_E, Q_W]^T$
- Action: $a \in \{0,1,2,3\}$ (phase selection)
- Cost: $J = \sum_t (\mathbf{x}^T Q_{\mathrm{cost}} \mathbf{x} + R_{\mathrm{cost}} a^2)$

Standard LQR would compute optimal gain $K = R^{-1}B^T P$ where $P$ solves Riccati equation.

DQN learns this (approximately) without explicit model.

---

## SECTION 11: QUICK FORMULA REFERENCE (Copy-Paste Ready)

### Q-Learning Update
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

### TD Target & Loss
```
target = r + γ max_a' Q(s', a')
loss = (target - Q(s,a))²
```

### epsilon-greedy Policy
```
if rand() < ε:
    a = random_action()
else:
    a = argmax_a Q(s, a)
```

### Batch LEARNING
```
for (s_i, a_i, r_i, s'_i) in batch:
    y_i = r_i + γ max_a' Q(s'_i, a')
    loss += (y_i - Q(s_i, a_i))²
θ ← θ - α ∇_θ loss
```

### Wait Reduction Reward
```
reward = prev_total_wait - current_total_wait
```

### Quadratic Wait Reward
```
wait_squared = sum((w_i / 60.0)^2 for each car i)
reward = prev_wait_sq - current_wait_sq
```

### Epsilon Decay
```
ε(t) = max(0.05, 1.0 - t / (100 * 0.8))
```

### Target Network Update (Soft)
```
θ_target ← τ θ_online + (1-τ) θ_target
```

### Adam Optimizer Step
```
m ← β₁ m + (1-β₁) g
v ← β₂ v + (1-β₂) g²
θ ← θ - α m / (√v + ε)
```

---

## APPENDIX: FREQUENTLY NEEDED DERIVATIONS

### A.1 Deriving Wait Reduction from First Principles

**Starting Setup**:
- At time $t$, vehicle $i$ has waiting cost $w_i(t) \geq 0$
- We want to minimize total wait across all vehicles

**Optimization Objective**:
$$\min_{a_1, a_2, \ldots} \sum_{t=0}^{T} W_t \quad \text{where } W_t = \sum_i w_i(t)$$

**Greedy Surrogate** (myopic):
$$a_t^{*} = \arg\min_{a} W_{t+\Delta t}(a)$$

**Equivalent Reward** (max this instead of minimizing $W$):
$$R_t = W_t - W_{t+\Delta t}$$

Maximizing $\sum R_t = W_0 - W_T$ ≈ minimizes final wait (if $W_T$ small).

### A.2 Why Exponential (Quadratic) Weighting Prevents Starvation

**Intuition**: Older, larger individual waits should be prioritized.

**Quadratic Form**:
$$W^{(\mathrm{sq})} = \sum_i (w_i)^2$$

**Analysis**: 
- If $w_1 = 100$, $w_2 = 10, w_3 = 10, \ldots, w_{40} = 10$
- Sum-squared: $(100)^2 + 40 \times (10)^2 = 10000 + 4000 = 14000$
- Clearing just $w_1$ by 10s: Reward gain = $(100)^2 - (90)^2 = 1900$
- Clearing one of the 40 by 10s: Reward gain = $(10)^2 - (0)^2 = 100$
- Ratio: 19x higher reward for clearing the old wait → **prioritized**.

---

**END OF MATHEMATICAL FORMULAS**

*Use these equations directly in research papers, control analysis, and model development.*
