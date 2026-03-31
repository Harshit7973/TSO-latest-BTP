# COMPLETE MATHEMATICAL MODEL: TSO-LATEST-BTP
## Repository-Specific RL Formulation and Implementation Notes

**Date**: March 31, 2026  
**Repository**: TSO-latest-BTP  
**Scope**: Only the code and configurations present in this repository

---

## SECTION 1: SYSTEM DEFINITION

### 1.1 Learning Setup

The repository trains a Deep Q-Network (DQN) agent using Stable-Baselines3 in single-agent SUMO environments.

Core training entrypoint:
- `btp/train_dqn.py`

Core environment:
- `sumo_rl/environment/env.py`
- `sumo_rl/environment/traffic_signal.py`
- `sumo_rl/environment/observations.py`

### 1.2 Supported Network Configurations

Defined in `btp/train_dqn.py`:

1. `single`
2. `2way`
3. `4x4`

Each configuration maps to specific `.net.xml` and `.rou.xml` files under `sumo_rl/nets/`.

---

## SECTION 2: MDP FORMULATION

### 2.1 MDP Tuple

$$
\mathcal{M}=\langle S, A, P, R, \gamma \rangle
$$

Where:

1. $S$ is the observation space returned by `DefaultObservationFunction`.
2. $A$ is a discrete phase-selection action set.
3. $P$ is induced by SUMO dynamics and selected traffic-light phase.
4. $R$ is the selected reward function (`diff-waiting-time` by default in environment logic).
5. $\gamma$ is the discount factor used by DQN.

### 2.2 State / Observation Space

Default observation in this repo:

$$
s_t = [\text{phase\_one\_hot},\ \text{min\_green},\ \rho_1,\ldots,\rho_m,\ q_1,\ldots,q_m]
$$

From `sumo_rl/environment/observations.py`:

1. `phase_one_hot`: one-hot indicator of active green phase.
2. `min_green`: binary flag for minimum phase-duration constraint.
3. $\rho_i$: normalized lane density for incoming lane $i$.
4. $q_i$: normalized queue (stopped vehicles) for incoming lane $i$.

Observation dimension:

$$
|s_t| = n_{\text{green\_phases}} + 1 + 2m
$$

where $m$ is the number of controlled incoming lanes.

### 2.3 Action Space

Action space is discrete with size equal to number of green phases:

$$
a_t \in \{0,1,\ldots,n_{\text{green\_phases}}-1\}
$$

From `TrafficSignal.set_next_phase(...)`:

1. If phase changes, yellow transition is enforced.
2. Minimum green duration is enforced.
3. Optional maximum-green enforcement can force phase rotation.

### 2.4 Transition Dynamics

Transition is simulator-driven:

$$
s_{t+1} \sim P(\cdot \mid s_t, a_t)
$$

SUMO determines vehicle movement, lane interactions, and queue evolution, while RL selects signal phases.

---

## SECTION 3: REWARD FUNCTION (REPO IMPLEMENTATION)

### 3.1 Default Reward Logic

In `sumo_rl/environment/traffic_signal.py`, default reward is differential waiting-time:

$$
\text{ts\_wait}_t = \frac{1}{100}\sum_{\ell \in \text{incoming lanes}} W_{\ell}(t)
$$

$$
r_t = \text{ts\_wait}_{t-1} - \text{ts\_wait}_t
$$

Interpretation:

1. Positive reward when cumulative waiting time decreases.
2. Negative reward when cumulative waiting time increases.

### 3.2 Alternative Reward Hooks

The repository exposes additional reward options in `TrafficSignal`:

1. Pressure-based reward.
2. Average-speed reward.
3. Queue-length reward.
4. CO2-related reward.

These can be selected by reward-function configuration in environment setup.

---

## SECTION 4: DQN OBJECTIVE AND UPDATE RULES

### 4.1 Q-Function Approximation

DQN approximates action-value function:

$$
Q_\theta(s,a) \approx Q^{*}(s,a)
$$

### 4.2 Temporal-Difference Target

For sampled transition $(s,a,r,s')$:

$$
y = r + \gamma \max_{a'} Q_{\theta^-}(s',a')
$$

### 4.3 Loss Function

$$
\mathcal{L}(\theta)=\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\left[(y-Q_\theta(s,a))^2\right]
$$

### 4.4 Parameter Update

$$
\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)
$$

with optimizer and schedule handled by Stable-Baselines3 DQN internals.

---

## SECTION 5: TRAINING HYPERPARAMETERS (btp/train_dqn.py)

Configured values:

1. Learning rate: $1\times10^{-3}$
2. Replay buffer size: $50{,}000$
3. Learning starts: $1000$ steps
4. Batch size: $64$
5. Discount factor: $\gamma=0.99$
6. Train frequency: every $4$ environment steps
7. Target update interval: $1000$ steps
8. Soft update coefficient: $\tau=1.0$
9. Exploration fraction: $0.15$
10. Initial epsilon: $1.0$
11. Final epsilon: $0.01$
12. Policy network architecture: `[256, 256]`

### 5.1 Network Architecture

The policy is `MlpPolicy` with two hidden layers:

$$
\text{Input} \rightarrow 256\ (\text{ReLU}) \rightarrow 256\ (\text{ReLU}) \rightarrow |A|
$$

### 5.2 Exploration Schedule

Epsilon-greedy exploration transitions from 1.0 to 0.01 over early training according to exploration fraction.

---

## SECTION 6: FIXED-TIMING BASELINE MODEL

Baseline runner: `btp/train_baseline.py`

The fixed-time baseline keeps signal logic non-learning (`fixed_ts=True`), providing a non-RL comparator.

Mathematically, baseline policy is constant/timing-rule driven:

$$
\pi_{\text{fixed}}(a_t\mid s_t) = \text{deterministic schedule}
$$

This serves as the control group against learned $\pi_\theta$ from DQN.

---

## SECTION 7: EVALUATION METRICS USED IN THIS REPO

Used in `btp/test_model.py` (`info` fields from environment):

1. System total waiting time.
2. System total stopped vehicles (queue proxy).
3. System mean speed.
4. Total episodic reward.

### 7.1 Representative Metric Definitions

Mean waiting time over horizon $T$:

$$
\bar{W}=\frac{1}{T}\sum_{t=1}^{T}W_t
$$

Mean queue length:

$$
\bar{Q}=\frac{1}{T}\sum_{t=1}^{T}Q_t
$$

Mean speed:

$$
\bar{V}=\frac{1}{T}\sum_{t=1}^{T}V_t
$$

Episodic return:

$$
G=\sum_{t=1}^{T}r_t
$$

---

## SECTION 8: CONSTRAINTS AND SAFETY LOGIC

From traffic-signal implementation:

1. Yellow phase insertion on green-phase transitions.
2. Minimum green-time enforcement.
3. Optional max-green enforcement.
4. Discrete action timing with simulator step coupling.

These constraints ensure physically valid signal operation during RL control.

---

## SECTION 9: CODE-TO-MATH MAPPING (THIS REPO ONLY)

1. MDP environment and step dynamics: `sumo_rl/environment/env.py`
2. Action constraints and reward internals: `sumo_rl/environment/traffic_signal.py`
3. Default observation construction: `sumo_rl/environment/observations.py`
4. DQN training config: `btp/train_dqn.py`
5. Fixed baseline evaluation: `btp/train_baseline.py`
6. Trained model evaluation: `btp/test_model.py`

---

## SECTION 10: SUMMARY

TSO-latest-BTP implements a repository-contained DQN traffic-signal optimization workflow with:

1. SUMO-backed Markov decision process.
2. Structured lane-density/queue observations.
3. Differential waiting-time reward.
4. Stable-Baselines3 DQN learning.
5. Fixed-time baseline for quantitative comparison.

This document intentionally includes only information verifiable from files in this repository.

---

**END OF REPOSITORY-SPECIFIC MATHEMATICAL MODEL**