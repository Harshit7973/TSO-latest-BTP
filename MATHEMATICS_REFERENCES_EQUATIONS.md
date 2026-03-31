# MATHEMATICS REFERENCES: TSO-LATEST-BTP
## Equation Sheet for the Repository Implementation

This sheet contains equations aligned to the code currently present in `TSO-latest-BTP`.

---

## 1. MDP CORE

MDP tuple:

$$
\mathcal{M}=\langle S,A,P,R,\gamma\rangle
$$

State (default observation):

$$
s_t=[\phi_1,\ldots,\phi_{n_p},\ m_t,\ \rho_1,\ldots,\rho_m,\ q_1,\ldots,q_m]
$$

where:

1. $\phi_i$ = one-hot green-phase indicator.
2. $m_t$ = min-green flag.
3. $\rho_i$ = normalized lane density.
4. $q_i$ = normalized lane queue.

Observation size:

$$
|s_t|=n_p+1+2m
$$

Action space:

$$
a_t\in\{0,1,\ldots,n_p-1\}
$$

Transition:

$$
s_{t+1}\sim P(\cdot\mid s_t,a_t)
$$

---

## 2. REWARD EQUATIONS

Repository default (`diff-waiting-time`):

$$
w_t^{(ts)}=\frac{1}{100}\sum_{\ell \in L_{in}}W_{\ell}(t)
$$

$$
r_t=w_{t-1}^{(ts)}-w_t^{(ts)}
$$

Interpretation:

1. $r_t>0$ when waiting-time decreases.
2. $r_t<0$ when waiting-time increases.

---

## 3. DQN FORMULATION

Q approximation:

$$
Q_\theta(s,a)
$$

TD target:

$$
y_t=r_t+\gamma\max_{a'}Q_{\theta^-}(s_{t+1},a')
$$

TD error:

$$
\delta_t=y_t-Q_\theta(s_t,a_t)
$$

Loss:

$$
\mathcal{L}(\theta)=\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\big[\delta_t^2\big]
$$

Gradient step:

$$
\theta\leftarrow\theta-\alpha\nabla_\theta\mathcal{L}(\theta)
$$

Target-network update (SB3 config in this repo):

$$
\theta^-\leftarrow\theta\quad\text{every }N_{\text{target}}=1000\text{ steps}
$$

(Configured with `tau=1.0` and `target_update_interval=1000`.)

---

## 4. POLICY AND EXPLORATION

Greedy policy:

$$
\pi_{\text{greedy}}(s)=\arg\max_a Q_\theta(s,a)
$$

Epsilon-greedy execution:

$$
a_t=\begin{cases}
\text{random action} & \text{with prob. }\epsilon_t \\
\arg\max_a Q_\theta(s_t,a) & \text{with prob. }1-\epsilon_t
\end{cases}
$$

Exploration schedule in this repo:

1. $\epsilon_{\text{init}}=1.0$
2. $\epsilon_{\text{final}}=0.01$
3. exploration fraction $=0.15$

---

## 5. TRAINING HYPERPARAMETERS (btp/train_dqn.py)

1. $\alpha=10^{-3}$
2. $\gamma=0.99$
3. buffer size $=50{,}000$
4. learning starts $=1000$
5. batch size $=64$
6. train frequency $=4$
7. target update interval $=1000$
8. policy net architecture $=[256,256]$

---

## 6. BASELINE MODEL

Fixed-timing baseline (`fixed_ts=True`):

$$
\pi_{\text{fixed}}(a_t\mid s_t)=\text{deterministic timing program}
$$

Used to compare learned DQN performance.

---

## 7. METRICS USED IN EVALUATION

From evaluation script outputs:

1. Total reward:

$$
G=\sum_{t=1}^{T}r_t
$$

2. Mean waiting time:

$$
\bar{W}=\frac{1}{T}\sum_{t=1}^{T}W_t
$$

3. Mean queue length:

$$
\bar{Q}=\frac{1}{T}\sum_{t=1}^{T}Q_t
$$

4. Mean speed:

$$
\bar{V}=\frac{1}{T}\sum_{t=1}^{T}V_t
$$

---

## 8. QUICK COPY FORMULAS

Q-learning style target:

```text
y = r + gamma * max_a' Q_target(s', a')
```

DQN loss:

```text
loss = (y - Q_online(s, a))^2
```

Diff-wait reward:

```text
reward = prev_ts_wait - current_ts_wait
```

Epsilon-greedy:

```text
if rand < epsilon: random action
else: argmax Q(s, a)
```

---

## 9. FILE REFERENCE INDEX

1. Training hyperparameters: `btp/train_dqn.py`
2. Baseline logic: `btp/train_baseline.py`
3. Evaluation metrics use: `btp/test_model.py`
4. Observation definition: `sumo_rl/environment/observations.py`
5. Reward and signal logic: `sumo_rl/environment/traffic_signal.py`

---

**END OF REPOSITORY-SPECIFIC EQUATION REFERENCE**