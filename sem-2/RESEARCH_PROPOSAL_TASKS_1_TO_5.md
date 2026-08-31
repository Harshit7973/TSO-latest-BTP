# Semester 2 Research Proposal and Completed Work: Tasks 1–5

## Adaptive Deep Reinforcement Learning for Traffic Signal Optimisation in SUMO

**Project level:** Eighth-semester B.Tech project  
**Repository workspace:** `TSO-latest-BTP`  
**Semester 1 evidence:** Frozen single-intersection DQN under `btp/`  
**Semester 2 workspace:** Isolated experiments under `sem-2/`  
**Document status:** Consolidated implementation, methodology, results and proposal source document  
**Result status:** Tasks 1–5 have complete saved evaluations; all reported numbers below come from repository CSV/JSON artifacts

---

## 1. Executive summary

Urban traffic signals are commonly operated using fixed schedules that cannot
react to short-term changes in traffic demand. Reinforcement learning provides
a way to select signal phases from observed traffic conditions, but a credible
student project must answer more than whether one trained model produces one
good simulation. It must establish reproducibility, generalisation,
multi-objective behaviour, algorithmic comparison and scalability to multiple
intersections.

Semester 1 produced a Deep Q-Network (DQN) for one two-way intersection. The
Semester 2 work therefore followed five linked research questions:

1. Can the Semester 1 improvement be reproduced using paired random seeds?
2. Does the frozen Semester 1 DQN generalise to changing and unseen demand?
3. Does a richer observation and multi-objective reward improve robustness?
4. How does a policy-gradient method, Proximal Policy Optimisation (PPO),
   compare with DQN under the same scenarios?
5. Can one interaction-aware DQN control four connected intersections and
   beat fixed timing on unseen seeds?

The experimental progression is:

```text
Semester 1 single-intersection DQN
                │
                ▼
Task 1: paired-seed reproducibility
                │
                ▼
Task 2: dynamic-demand generalisation
                │
          ┌─────┴─────┐
          ▼           ▼
Task 3: DQN-v2    Task 4: PPO
          └─────┬─────┘
                ▼
Task 5: shared interaction-aware Dueling Double DQN on a 2×2 grid
```

The strongest final result is Task 5. On five unseen paired seeds, the shared
DQN reduced mean total waiting time by **89.60%**, reduced stopped vehicles by
**62.13%**, increased mean speed by **24.66%**, and increased completed trips
per hour by **0.49%** relative to fixed timing. It won all four metrics on all
five seeds, produced zero teleports, and passed all 15 structural evaluation
checks. Raw and shielded DQN produced the same aggregate results; the safety
shield intervened only once in 7,200 decisions.

The results also contain useful negative findings. DQN-v1 lost 2.90% throughput
under burst demand despite substantially reducing congestion. DQN-v2 improved
burst waiting relative to DQN-v1 but lost speed and throughput. PPO was better
than DQN-v1 under burst demand, mixed under direction switching, and worse on
the unseen-mixed scenario. These findings support the conclusion that no one
single-intersection configuration is best under every traffic distribution.

---

## 2. Background and motivation

### 2.1 Problem

A traffic signal controller must repeatedly choose a phase while vehicle
arrivals, queues and downstream conditions change. Fixed timing is predictable
and easy to deploy, but it does not use the current traffic state. A learned
controller can adapt its phase choice, although it introduces important
questions:

- Is the improvement reproducible or caused by a favourable random seed?
- Does the controller work when traffic differs from its training route?
- Does optimising delay create poor queues, unfair lane service or unnecessary
  emissions?
- Would a different reinforcement-learning algorithm behave better?
- Does a model trained for one signal scale to interacting intersections?
- Can training be resumed and independently audited on laptop hardware?

### 2.2 Research gap addressed by this project

The Semester 1 implementation demonstrated model training but did not provide
all of the evidence required for a final-year research contribution. In
particular, it used a single intersection, a single principal reward, limited
demand variation, and evaluation that did not enforce paired random seeds.

Semester 2 addresses that gap through a staged experimental design rather than
one oversized training run. Each task produces raw time-series CSV files,
aggregated summaries, validation JSON, plots and resumable checkpoints. This
makes both successful and unsuccessful findings auditable.

### 2.3 Main research aim

To design and evaluate adaptive deep-reinforcement-learning traffic signal
controllers that are reproducible, robust to demand changes, sensitive to
multiple traffic objectives and capable of interaction-aware control over a
small network of connected intersections.

### 2.4 Objectives

1. Establish a statistically fair fixed-timing versus DQN baseline.
2. Generate controlled dynamic-demand scenarios and measure generalisation.
3. Implement and evaluate an enhanced multi-objective DQN.
4. Implement PPO and compare it fairly with the frozen DQN.
5. Implement a shared Dueling Double DQN for four interacting signals.
6. Save sufficient raw evidence to reproduce every reported result.
7. Keep the Semester 1 model and outputs unchanged.

### 2.5 Research questions

- **RQ1:** Does the frozen Semester 1 DQN consistently outperform fixed timing
  when both methods receive identical traffic seeds?
- **RQ2:** How does DQN-v1 respond to balanced, directional, switching, burst
  and unseen demand?
- **RQ3:** Can enhanced traffic state and a multi-objective reward improve
  DQN-v1 on difficult dynamic scenarios?
- **RQ4:** Does PPO provide a better robustness/performance trade-off than DQN?
- **RQ5:** Can a parameter-sharing DQN coordinate a four-intersection 2×2 grid
  and outperform fixed timing on unseen seeds?

---

## 3. Platform, experimental controls and reproducibility

### 3.1 Software stack

The implementation uses:

- Eclipse SUMO for microscopic traffic simulation.
- The repository's `sumo_rl` environment for Gymnasium/PettingZoo-compatible
  traffic signal interaction.
- Stable-Baselines3 for Semester 1 DQN, DQN-v2 and PPO.
- PyTorch for the custom Task 5 Dueling Double DQN.
- NumPy and pandas for data handling.
- Matplotlib for automatically saved plots.

Tasks 1–4 were executed with Python 3.13.3, Stable-Baselines3 2.8.0a4 and
PyTorch 2.6.0+cu124; CUDA was available according to their saved metadata.
Task 5 used PyTorch 2.11.0+cpu and completed successfully on CPU. SUMO is often
the dominant computational cost, so CPU evaluation remains valid.

### 3.2 Isolation policy

The directory `btp/` is treated as read-only Semester 1 evidence. Task-specific
routes, checkpoints, logs, tables and figures are saved below the relevant
`sem-2/<task>/` folder. `sem-2/validate_workspace.py` checks required structure
and verifies SHA-256 hashes for protected Semester 1 files.

### 3.3 Paired-seed design

For a given scenario and seed, every compared controller receives the same SUMO
random seed and simulation horizon. If controller A and controller B produce
metric values `A_i` and `B_i` under seed `i`, percentage improvement is:

```text
lower-is-better metric:  improvement_i = 100 × (B_i - A_i) / |B_i|
higher-is-better metric: improvement_i = 100 × (A_i - B_i) / |B_i|
```

Here `B` is the baseline and `A` is the candidate. Pairing reduces the effect
of comparing different random traffic realisations.

### 3.4 Primary metrics

The same four report metrics are used across the tasks:

- **Mean system total waiting time:** lower is better.
- **Mean system total stopped vehicles:** lower is better.
- **Mean system mean speed:** higher is better.
- **Throughput:** final arrived vehicles divided by simulated seconds and
  multiplied by 3,600; higher is better.

Final departed, arrived and teleported vehicle counters are also retained.
Zero teleportation is preferred because teleports may indicate severe gridlock
or simulation repair behaviour.

### 3.5 Structural validation

Every evaluation CSV is automatically checked for:

- Required columns.
- Non-empty records.
- Finite metric values.
- Completion of the requested simulation horizon.
- Monotonic departed and arrived counters.
- Non-negative vehicle counts.

The evaluation scripts resume safely by retaining completed raw episode files.
Training tasks save numbered/latest/final checkpoints and state needed for
continuation.

---

## 4. Semester 1 baseline retained for Semester 2

Semester 1 trained a Stable-Baselines3 DQN on the repository's two-way,
single-intersection SUMO network. Its principal settings were:

| Parameter | Semester 1 DQN-v1 value |
|---|---:|
| Training steps | 100,000 |
| Hidden network | 256, 256 |
| Learning rate | 0.001 |
| Replay capacity | 50,000 |
| Learning starts | 1,000 |
| Batch size | 64 |
| Discount factor | 0.99 |
| Training frequency | Every 4 steps |
| Target update interval | 1,000 steps |
| Exploration | 1.0 to 0.01 over first 15% |
| Primary reward | Difference in waiting time |

The frozen model is `../btp/models/dqn_2way.zip`. It is not retrained in Tasks
1 or 2, ensuring those tasks measure reproducibility and generalisation rather
than additional learning.

---

## 5. Task 1 — Reproducible paired-seed benchmark

### 5.1 Why this task was required

A single favourable test episode is insufficient evidence for a reinforcement
learning claim. Randomised vehicle insertion can change congestion, so fixed
timing and DQN must be compared on identical seeds. Task 1 converts the
Semester 1 demonstration into a repeatable benchmark.

### 5.2 Research question

Does the frozen Semester 1 DQN consistently outperform fixed-time signal
control over ten paired 3,600-second simulations?

### 5.3 What was implemented

`01-reproducible-benchmark/run_benchmark.py`:

- Loads the frozen Semester 1 DQN without changing it.
- Runs fixed timing and DQN-v1 for seeds 101–110.
- Uses the same 3,600-second two-way route for both methods.
- Saves one step-level CSV for each method/seed pair.
- Skips already completed episodes on restart.
- Computes paired percentage improvements.
- Generates four plots with 95% confidence-interval error bars.
- Produces a structural validation report.

### 5.4 Experimental design

| Item | Value |
|---|---|
| Network | Two-way single intersection |
| Controllers | Fixed timing, frozen DQN-v1 |
| Seeds | 101–110 |
| Horizon | 3,600 seconds |
| Raw episodes | 20 |
| Decision mode | Deterministic model inference |

### 5.5 Results

| Controller | Waiting ↓ | Stopped ↓ | Speed ↑ | Throughput/h ↑ |
|---|---:|---:|---:|---:|
| Fixed timing | 1,297.31 | 41.70 | 2.81 | 2,091.20 |
| DQN-v1 | 256.42 | 18.27 | 4.03 | 2,435.40 |

Mean paired improvement of DQN-v1 over fixed timing:

| Metric | Improvement | Seed wins |
|---|---:|---:|
| Waiting time | **80.22% lower** | 10/10 |
| Stopped vehicles | **56.16% lower** | 10/10 |
| Mean speed | **43.42% higher** | 10/10 |
| Throughput | **16.46% higher** | 10/10 |

All 20 structural validation entries passed and no teleports were reported.

### 5.6 Interpretation

Task 1 confirms that the Semester 1 DQN result is not dependent on one selected
seed. The improvement is large and directionally consistent across all ten
pairs. This supports using DQN-v1 as a credible reference controller in Tasks
2–4.

The claim remains network-specific. It establishes reproducibility on the
single intersection and original route; it does not establish performance on
unseen demand or multiple intersections.

### 5.7 Evidence

- [Task 1 README](01-reproducible-benchmark/README.md)
- [Summary by seed](01-reproducible-benchmark/results/summary_by_seed_sec3600.csv)
- [Paired improvements](01-reproducible-benchmark/results/paired_improvements_sec3600.csv)
- [Validation report](01-reproducible-benchmark/results/validation_sec3600.json)
- [Waiting-time plot](01-reproducible-benchmark/plots/sec3600_mean_system_total_waiting_time.png)

---

## 6. Task 2 — Dynamic traffic and generalisation

### 6.1 Why this task was required

The original route's major direction change occurs after 25,000 seconds, while
the Semester 1 episode lasts only 3,600 seconds. Therefore, the trained agent
was not being meaningfully tested against within-episode traffic change. Task 2
compresses relevant demand changes into the evaluation horizon and adds unseen
traffic patterns.

### 6.2 Research question

Can the frozen DQN-v1 generalise without retraining when traffic becomes
directionally imbalanced, switches direction, arrives in bursts, or follows an
unseen mixed distribution?

### 6.3 What was implemented

Two scripts were added:

- `generate_routes.py` constructs task-owned SUMO route XML files from staged
  traffic rates.
- `evaluate_scenarios.py` evaluates fixed timing and frozen DQN-v1 using paired
  scenario/seed inputs, validates every CSV, aggregates results and creates
  grouped plots.

The route generator defines 12 movements: straight and turning movements from
north, south, east and west. Straight traffic receives 55% of an approach rate,
and each turning direction receives 22.5%.

### 6.4 Traffic scenarios

| Scenario | Definition | Approximate requested vehicles |
|---|---|---:|
| Balanced | North–south and east–west rates both 500 veh/h | 1,996 |
| NS peak | North–south 850, east–west 220 veh/h | 2,136 |
| EW peak | North–south 220, east–west 850 veh/h | 2,136 |
| Direction switch | Balanced → NS peak → EW peak → balanced | 2,066 |
| Burst | Low → high → low → high demand | 2,236 |
| Unseen mixed | North–south 650, east–west 380 veh/h | 2,056 |

For staged scenarios, changes occur at 900, 1,800 and 2,700 seconds in a
3,600-second episode.

### 6.5 Experimental design

| Item | Value |
|---|---|
| Network | Two-way single intersection |
| Controllers | Fixed timing, frozen DQN-v1 |
| Seeds | 201, 202, 203 |
| Scenarios | Six |
| Horizon | 3,600 seconds |
| Raw episodes | 36 |

### 6.6 Results by scenario

The table reports mean paired DQN-v1 improvement relative to fixed timing.

| Scenario | Waiting ↓ | Stopped ↓ | Speed ↑ | Throughput ↑ |
|---|---:|---:|---:|---:|
| Balanced | 89.17% | 73.17% | 75.47% | 1.09% |
| Burst | 53.81% | 39.67% | 42.47% | **−2.90%** |
| Direction switch | 91.96% | 83.60% | 125.14% | 3.53% |
| EW peak | 87.17% | 82.03% | 157.94% | 8.90% |
| NS peak | 92.03% | 81.05% | 126.58% | 8.55% |
| Unseen mixed | 92.32% | 80.28% | 118.02% | 4.12% |

Absolute mean values illustrate the difficult burst case:

| Burst controller | Waiting ↓ | Stopped ↓ | Speed ↑ | Throughput/h ↑ |
|---|---:|---:|---:|---:|
| Fixed timing | 1,659.35 | 51.35 | 3.18 | 2,071.33 |
| DQN-v1 | 766.39 | 30.98 | 4.54 | 2,011.33 |

All 36 structural checks passed, and no teleports were reported.

### 6.7 Interpretation

DQN-v1 generalised surprisingly well to five scenarios and greatly reduced
congestion in all six. Burst traffic was the main weakness: waiting and queues
improved, but throughput fell by 2.90%. This is a meaningful trade-off rather
than a failed simulation. The result motivates a richer state and reward that
can explicitly represent waiting, queues, outgoing congestion, emissions and
fairness.

Task 2 uses three seeds per scenario. The patterns are consistent, but a larger
seed set would strengthen statistical inference.

### 6.8 Evidence

- [Task 2 README](02-dynamic-traffic/README.md)
- [Route manifest](02-dynamic-traffic/generated-routes/sec3600/manifest.json)
- [Scenario summary](02-dynamic-traffic/results/summary_by_scenario_seed_sec3600.csv)
- [Paired improvements](02-dynamic-traffic/results/paired_improvements_sec3600.csv)
- [Validation report](02-dynamic-traffic/results/validation_sec3600.json)
- [Waiting-time plot](02-dynamic-traffic/plots/sec3600/scenario_mean_system_total_waiting_time.png)

---

## 7. Task 3 — Enhanced multi-objective DQN-v2

### 7.1 Why this task was required

DQN-v1 primarily optimises waiting-time change using the standard observation.
Traffic control is multi-objective: a controller should also limit queues,
avoid starving a lane, consider emissions and downstream congestion, and avoid
excessive switching. Task 3 tests whether additional state and reward terms
produce a more robust DQN.

### 7.2 Research question

Does an enhanced observation and multi-objective reward improve DQN-v1 on
direction-switch, burst and unseen-mixed demand without unacceptable throughput
loss?

### 7.3 Enhanced observation

`features.py` adds:

- Current phase one-hot vector.
- Minimum-green eligibility.
- Normalised elapsed phase time.
- Incoming-lane density.
- Incoming-lane queue.
- Per-lane accumulated waiting time, clipped after scaling by 300 seconds.
- Deterministically ordered outgoing-lane density.

Sorting outgoing lanes keeps feature positions stable across processes and
resume sessions. Maximum green is enforced at 60 seconds.

### 7.4 Multi-objective reward

For a traffic signal, the implemented reward is:

```text
r = waiting_improvement
    - 0.15 × queue_penalty
    - 0.02 × CO2_penalty
    - 0.05 × fairness_penalty
    - 0.03 × switching_penalty
```

where queue is scaled by 20 vehicles, CO2 is scaled by 100,000 and clipped,
fairness uses the maximum lane waiting time scaled by 300 seconds and clipped,
and switching is indicated by a yellow phase. The reward attempts to preserve
delay improvement while discouraging long queues, emissions, lane starvation
and unnecessary phase changes.

### 7.5 Model and training implementation

| Parameter | DQN-v2 value |
|---|---:|
| Training target | 60,000 steps |
| Training scenario | 1,800-second direction-switch route |
| Hidden network | 256, 256 |
| Learning rate | 0.0005 |
| Replay capacity | 50,000 |
| Learning starts | 2,000 |
| Batch size | 64 |
| Discount factor | 0.99 |
| Training frequency | Every 4 steps |
| Target update | Every 1,000 steps |
| Exploration | 1.0 to 0.03 over 25% of training |
| Resume chunk | 10,000 steps |

Successive episodes use deterministic but different SUMO seeds. The code saves
the model, replay buffer, training state, numbered checkpoints, monitor logs,
TensorBoard data and a training curve after each completed chunk.

### 7.6 Experimental design

| Item | Value |
|---|---|
| Controllers | Fixed timing, frozen DQN-v1, trained DQN-v2 |
| Scenarios | Direction switch, burst, unseen mixed |
| Seeds | 301–305 |
| Horizon | 3,600 seconds |
| Raw episodes | 45 |
| Completed training | 60,000/60,000 steps |

### 7.7 DQN-v2 versus DQN-v1

| Scenario | Waiting improvement | Stopped improvement | Speed change | Throughput change |
|---|---:|---:|---:|---:|
| Burst | **36.14%** | **13.88%** | −3.31% | −2.13% |
| Direction switch | **14.36%** | −4.06% | −2.80% | 0.02% |
| Unseen mixed | **4.69%** | **6.40%** | **3.38%** | −0.10% |

### 7.8 DQN-v2 versus fixed timing

| Scenario | Waiting improvement | Stopped improvement | Speed improvement | Throughput change |
|---|---:|---:|---:|---:|
| Burst | 70.55% | 46.53% | 34.26% | **−5.63%** |
| Direction switch | 93.16% | 82.55% | 118.15% | 3.46% |
| Unseen mixed | 93.03% | 82.14% | 130.08% | 4.08% |

All 45 structural checks passed, and no teleports were reported.

### 7.9 Interpretation

DQN-v2 achieved its clearest intended improvement on burst waiting and queues.
It also modestly improved unseen-mixed demand. However, the multi-objective
reward did not dominate DQN-v1 on every metric:

- Burst throughput fell 2.13% relative to DQN-v1 and 5.63% relative to fixed.
- Direction-switch stopped vehicles and speed were slightly worse than DQN-v1.
- The reward includes CO2 and fairness terms, but the final evaluation does not
  separately report emissions or per-lane fairness metrics. Therefore, the
  existing results must not claim a measured CO2 reduction.

The academically correct conclusion is that richer objectives changed the
trade-off and improved congestion in selected difficult conditions, not that
DQN-v2 universally replaced DQN-v1.

### 7.10 Evidence

- [Task 3 README](03-multiobjective-dqn/README.md)
- [Enhanced observation and reward](03-multiobjective-dqn/features.py)
- [Training metadata](03-multiobjective-dqn/results/training_metadata.json)
- [Comparison summary](03-multiobjective-dqn/results/comparison_summary_sec3600.csv)
- [DQN-v2 versus DQN-v1](03-multiobjective-dqn/results/improvement_v2_vs_v1_sec3600.csv)
- [Validation report](03-multiobjective-dqn/results/validation_sec3600.json)
- [Training curve](03-multiobjective-dqn/plots/training_curve.png)

---

## 8. Task 4 — PPO algorithm comparison

### 8.1 Why this task was required

Only testing variants of DQN cannot show whether the observed behaviour is
specific to value-based reinforcement learning. PPO is a widely used
policy-gradient algorithm and already had examples in the reference repository.
Task 4 adapts it to the same single-intersection environment and performs a
fair scenario/seed comparison.

### 8.2 Research question

How does PPO compare with the frozen DQN-v1 and fixed timing under burst,
direction-switch and unseen-mixed traffic?

### 8.3 What was implemented

The PPO task includes:

- Incrementing deterministic training seeds.
- Enforced maximum green.
- Resumable rollout chunks and numbered checkpoints.
- Monitor and TensorBoard logs.
- Training-curve generation.
- Three-controller paired evaluation.
- Raw CSV, aggregate tables, pairwise comparison CSVs and validation JSON.

### 8.4 PPO configuration

| Parameter | PPO value |
|---|---:|
| Requested training target | 60,000 steps |
| Actual completed steps | 60,416 |
| Hidden policy network | 128, 128 |
| Hidden value network | 128, 128 |
| Learning rate | 0.0003 |
| Rollout size | 512 |
| Batch size | 64 |
| Epochs per rollout | 10 |
| Discount factor | 0.99 |
| GAE lambda | 0.95 |
| PPO clip range | 0.2 |
| Entropy coefficient | 0.01 |

PPO slightly exceeds the requested target because it collects complete
rollouts. This is expected rather than an error.

### 8.5 Experimental design

| Item | Value |
|---|---|
| Controllers | Fixed timing, frozen DQN-v1, PPO |
| Scenarios | Direction switch, burst, unseen mixed |
| Seeds | 401–405 |
| Horizon | 3,600 seconds |
| Raw episodes | 45 |

### 8.6 PPO versus DQN-v1

| Scenario | Waiting improvement | Stopped improvement | Speed change | Throughput change |
|---|---:|---:|---:|---:|
| Burst | **45.98%** | **11.61%** | **4.14%** | −1.51% |
| Direction switch | **26.32%** | −1.82% | −1.71% | 0.02% |
| Unseen mixed | **−5.13%** | **−9.68%** | −2.16% | −0.04% |

### 8.7 PPO versus fixed timing

| Scenario | Waiting improvement | Stopped improvement | Speed improvement | Throughput change |
|---|---:|---:|---:|---:|
| Burst | 75.11% | 46.20% | 47.46% | **−4.72%** |
| Direction switch | 94.19% | 83.17% | 123.61% | 3.50% |
| Unseen mixed | 92.34% | 79.12% | 118.87% | 4.04% |

All 45 structural checks passed, and no teleports were reported.

### 8.8 Interpretation

PPO substantially improved burst waiting relative to DQN-v1, although it
reduced throughput by 1.51%. Under direction switching, PPO reduced waiting but
had slightly worse stopped vehicles and speed. Under unseen-mixed demand,
DQN-v1 was better on waiting, stopped vehicles and speed.

Therefore, PPO is a meaningful algorithmic comparison but not a universal
winner. The result supports a scenario-dependent interpretation:

- PPO adapted particularly well to burst-related congestion.
- DQN-v1 generalised better to the unseen mixed distribution.
- Both learned controllers strongly outperformed fixed timing on congestion.
- Training objective and traffic distribution matter as much as algorithm name.

### 8.9 Evidence

- [Task 4 README](04-ppo-comparison/README.md)
- [Training metadata](04-ppo-comparison/results/training_metadata.json)
- [Comparison summary](04-ppo-comparison/results/comparison_summary_sec3600.csv)
- [PPO versus DQN-v1](04-ppo-comparison/results/ppo_vs_dqn_sec3600.csv)
- [PPO versus fixed](04-ppo-comparison/results/ppo_vs_fixed_sec3600.csv)
- [Validation report](04-ppo-comparison/results/validation_sec3600.json)
- [Training curve](04-ppo-comparison/plots/training_curve.png)

---

## 9. Task 5 — Shared robust DQN for multi-intersection control

### 9.1 Why this task was required

The earlier tasks control one traffic light. Connected intersections introduce
interaction: releasing vehicles at one signal can create a queue at the next.
A tabular Q-learning design was initially explored for Task 5 but was removed
because its discretised state grew rapidly and its measured performance was not
sufficient. The replacement is a neural parameter-sharing architecture suited
to a four-signal 2×2 network.

### 9.2 Research question

Can one shared, interaction-aware DQN control four connected traffic signals
and outperform fixed timing on unseen paired traffic seeds while satisfying
safety and structural checks?

### 9.3 Parameter-sharing design

All four signals use the same neural network. At each environment step, the
network receives one state per signal and produces four phase values. Each
signal contributes a transition to one shared replay buffer. Parameter sharing:

- Reduces model count.
- Allows similar traffic patterns to update the same representation.
- Produces four replay transitions per environment step.
- Avoids a separate independently drifting network for every signal.

### 9.4 Interaction-aware 26-dimensional state

For each signal, the state contains:

| Component | Values |
|---|---:|
| Current phase one-hot | 4 |
| Minimum-green eligibility | 1 |
| Pressure per candidate phase | 4 |
| Incoming queue per phase | 4 |
| Outgoing queue per phase | 4 |
| Local queue and network mean/max/std queue plus progress | 5 |
| Intersection identity one-hot | 4 |
| **Total** | **26** |

Pressure is incoming queued vehicles minus outgoing queued vehicles for the
lanes served by a phase. Continuous traffic values use bounded `tanh`
normalisation. Network mean, maximum and standard deviation of queues form the
explicit interaction context.

### 9.5 Dueling Double DQN architecture

The shared network contains:

```text
26-dimensional state
        ↓
Linear 128 + LayerNorm + ReLU
        ↓
Linear 128 + ReLU
        ├──────────────┐
        ↓              ↓
Value stream       Advantage stream
128→64→1           128→64→4
        └──────┬───────┘
               ↓
Q(s,a) = V(s) + A(s,a) - mean(A)
```

Double-DQN targets use the online network to select the next action and the
target network to evaluate it. This reduces the direct maximisation bias of a
single Q-network.

### 9.6 Prioritised replay and optimisation

- Replay capacity: 50,000 transitions.
- Priority exponent: 0.6.
- Importance-sampling beta starts at 0.4 and anneals.
- Batch size: 128.
- Learning rate: 0.0003.
- Discount factor: 0.99.
- Target network update: every 750 gradient steps.
- Huber temporal-difference loss.
- Gradient norm clipping at 10.
- Expert cross-entropy weight: 0.15 for expert transitions.

The optimisation objective is:

```text
L = prioritised_Huber_TD_loss + 0.15 × expert_imitation_loss
```

### 9.7 Coordination reward

SUMO-RL's queue reward is used. Each agent combines local and team reward:

```text
effective reward = 0.75 × local reward + 0.25 × mean team reward
```

The result is divided by 20 and clipped to `[−5, 1]` before replay. This gives
each signal a local congestion objective while retaining a network-level
coordination incentive.

### 9.8 Expert warm-start and pressure shield

The first five training episodes use max-pressure actions. Their transitions
populate replay with useful behaviour and receive the auxiliary imitation loss.

After warm-start, the raw DQN action is restricted by a safety mask:

- Before minimum green is satisfied, the current phase is retained.
- Afterwards, phases within two vehicles of maximum pressure are eligible.
- Raw and executed actions are both logged.
- Shield interventions and expert agreement are reported.

Max pressure is not included as a controller in the main final plots or tables.
It remains an internal source of expert guidance, a basis for the action mask,
and an internal validation diagnostic. This dependence must be disclosed.

### 9.9 Training and model selection

| Parameter | Value |
|---|---:|
| Network | Four-signal 2×2 grid |
| Episodes | 40 |
| Seconds per training episode | 1,800 |
| Training seeds | 801–840 |
| Expert episodes | 5 |
| Epsilon | 0.15 to 0.01 over 30,000 steps |
| Validation frequency | Every 5 episodes |
| Validation seeds | 9601, 9602 |
| Validation horizon | 900 seconds |
| Final test seeds | 701–705 |

The validation score is:

```text
score = mean total waiting + 10 × mean stopped vehicles
```

A candidate qualifies when its validation score is no more than 85% of fixed
timing and its throughput is at least 97% of fixed timing. The best qualifying
checkpoint is retained rather than automatically using episode 40.

The best checkpoint occurred at episode 15:

| Validation metric | Fixed | Selected DQN |
|---|---:|---:|
| Waiting | 221.18 | 21.64 |
| Stopped | 10.54 | 3.90 |
| Speed | 6.41 | 8.01 |
| Throughput/h | 1,282 | 1,300 |
| Selection score | 326.57 | 60.67 |

Episode 40 degraded to a validation score of 129.19, expert agreement of
81.53% and shield intervention of 5.97%. Selecting episode 15 prevented this
late-training instability from contaminating the final evaluation.

### 9.10 Final experimental design

| Item | Value |
|---|---|
| Controllers in main comparison | Fixed, raw DQN, shielded DQN |
| Seeds | 701–705 |
| Horizon | 1,800 seconds |
| Raw episodes | 15 |
| Selected checkpoint | Episode 15 |
| Inference device | CPU |

### 9.11 Final results

| Controller | Waiting ↓ | Stopped ↓ | Speed ↑ | Throughput/h ↑ |
|---|---:|---:|---:|---:|
| Fixed timing | 225.37 | 10.79 | 6.38 | 1,377.60 |
| Raw DQN | 23.46 | 4.08 | 7.96 | 1,384.40 |
| Shielded DQN | 23.46 | 4.08 | 7.96 | 1,384.40 |

Shielded DQN versus fixed timing:

| Metric | Mean paired improvement | Wins |
|---|---:|---:|
| Waiting time | **89.60% lower** | 5/5 |
| Stopped vehicles | **62.13% lower** | 5/5 |
| Mean speed | **24.66% higher** | 5/5 |
| Throughput | **0.49% higher** | 5/5 |

Paired absolute DQN-minus-fixed differences with approximate 95% confidence
interval half-widths:

| Metric | Mean difference | 95% CI half-width |
|---|---:|---:|
| Waiting | −201.92 | 18.77 |
| Stopped vehicles | −6.71 | 0.66 |
| Mean speed | +1.57 | 0.10 |
| Throughput/h | +6.80 | 4.51 |

### 9.12 Safety and correctness results

- All 15 structural validation entries passed.
- Every episode reached 1,800 simulation seconds.
- Every raw episode contains 360 recorded steps.
- No vehicle teleportation occurred.
- Training, validation and final seeds do not overlap.
- The checkpoint SHA-256 is recorded in evaluation metadata.
- The automatic shielded-DQN success flag is `true`.
- The offline neural/replay/mask self-check passed all 11 checks.

### 9.13 Raw versus shielded interpretation

Raw and shielded DQN have identical aggregate metrics. Across the five final
seeds, the shield intervened only once in 7,200 decisions, giving a mean shield
rate of approximately 0.014%. Raw DQN agreed with the max-pressure expert on
99.61% of decisions; shielded DQN agreement was 99.63%.

This supports two conclusions:

1. The learned network generally selected an eligible action itself; frequent
   shield correction did not create the final improvement.
2. The network learned behaviour extremely close to the pressure expert. It is
   more accurate to describe it as an expert-guided learned policy than as a
   completely independent strategy discovered without domain knowledge.

### 9.14 Defensible Task 5 claim

> On five unseen paired traffic seeds in the SUMO 2×2 network, the proposed
> interaction-aware shared Dueling Double DQN reduced mean total waiting time
> by 89.60% and stopped vehicles by 62.13%, increased mean speed by 24.66%, and
> maintained slightly higher throughput than fixed-time control. All structural
> checks passed and no vehicle teleportation occurred.

This claim is limited to the tested network, routes, horizons and seeds. It is
not evidence of universal superiority or real-road deployment readiness.

### 9.15 Evidence

- [Task 5 README](05-multi-intersection/README.md)
- [Core DQN implementation](05-multi-intersection/dqn_core.py)
- [Training implementation](05-multi-intersection/train_dqn.py)
- [Final automatic analysis](05-multi-intersection/results/analysis_shared_dueling_ddqn_v1_sec1800.md)
- [Final evaluation table](05-multi-intersection/results/evaluation_shared_dueling_ddqn_v1_sec1800.csv)
- [Validation report](05-multi-intersection/results/validation_shared_dueling_ddqn_v1_sec1800.json)
- [Deployment gate](05-multi-intersection/results/deployment_gate_shared_dueling_ddqn_v1.json)
- [Waiting-time plot](05-multi-intersection/plots/shared_dueling_ddqn_v1_sec1800_mean_system_total_waiting_time.png)
- [Training plot](05-multi-intersection/plots/training_shared_dueling_ddqn_v1.png)

---

## 10. Cross-task result synthesis

### 10.1 What each task contributed

| Task | Scientific contribution | Main outcome |
|---|---|---|
| 1 | Reproducible paired-seed evidence | DQN-v1 beat fixed on all 10 seeds |
| 2 | Demand-shift/generalisation test | Strong generalisation; burst throughput exposed a weakness |
| 3 | Enhanced state and multi-objective DQN | Improved selected congestion metrics, but introduced trade-offs |
| 4 | Value-based versus policy-gradient comparison | PPO best on burst waiting; DQN better on unseen mixed |
| 5 | Interaction-aware multi-intersection learning | Shared DQN beat fixed on all five final seeds and all metrics |

### 10.2 Main findings

1. **The Semester 1 result is reproducible.** Task 1 converts a demonstration
   into ten paired comparisons with complete validation.
2. **DQN-v1 is more robust than initially expected.** It performs strongly on
   directional and switching demand without retraining.
3. **Burst demand is the consistent difficult case.** It creates a trade-off
   between congestion relief and completed throughput.
4. **More reward terms do not guarantee dominance.** DQN-v2 improves selected
   objectives but slightly worsens others.
5. **Algorithm performance is scenario-dependent.** PPO and DQN each have
   conditions where they are preferable.
6. **Parameter sharing makes four-signal control practical.** Task 5 uses one
   compact network and shared replay rather than four unrelated models.
7. **Checkpoint selection is important.** Task 5's episode-40 policy degraded,
   while held-out validation correctly retained episode 15.
8. **Expert guidance is highly influential.** Task 5's final model agrees with
   the pressure expert on more than 99.6% of decisions.

### 10.3 Overall project contribution

The project contribution is not merely “a DQN was trained.” It is a complete
experimental framework that:

- Re-evaluates a frozen prior model fairly.
- Generates controlled non-stationary traffic demand.
- Implements richer state and reward engineering.
- Compares value-based and policy-gradient learning.
- Scales to a four-intersection parameter-sharing controller.
- Adds expert warm-start, prioritised replay, Double DQN, dueling heads,
  interaction features, action constraints and held-out checkpoint selection.
- Produces raw, resumable and automatically validated evidence.

---

## 11. Proposed methodology section for a formal research proposal

The following text can be adapted directly into a proposal.

### 11.1 Proposed approach

The work adopts a staged simulation-based methodology. A frozen
single-intersection DQN is first benchmarked against fixed timing using paired
random seeds. Controlled dynamic-demand routes are then generated to evaluate
distribution shift. An enhanced DQN is trained using additional temporal,
queue, waiting and downstream features with a multi-objective reward. PPO is
trained as a policy-gradient comparator under the same environment. Finally, a
parameter-sharing Dueling Double DQN is trained on a four-intersection 2×2 grid
using local and network-level features, prioritised replay, expert warm-start
and a pressure-informed action mask.

All controllers are evaluated deterministically on held-out seeds. Raw
step-level data, aggregate metrics, confidence intervals, validation checks and
plots are retained. Training and evaluation seeds are separated in Task 5, and
the final checkpoint is selected using held-out validation rather than final
test performance.

### 11.2 Hypotheses

- **H1:** DQN-v1 produces lower waiting and queue metrics than fixed timing
  across paired seeds.
- **H2:** Dynamic demand reduces performance relative to balanced demand, with
  burst traffic being the most difficult scenario.
- **H3:** Richer state and reward design improves selected robustness metrics
  but may create speed/throughput trade-offs.
- **H4:** PPO and DQN exhibit scenario-dependent performance rather than one
  algorithm dominating universally.
- **H5:** A shared interaction-aware DQN reduces network congestion relative to
  fixed timing on unseen multi-intersection seeds.

### 11.3 Independent and dependent variables

Independent variables include controller type, traffic scenario, random seed,
single- versus multi-intersection topology, and raw versus shielded action
selection. Dependent variables include waiting, stopped vehicles, mean speed,
throughput, teleportation, shield intervention rate and expert agreement.

### 11.4 Evaluation protocol

1. Fix the network, route, horizon and seed for each paired comparison.
2. Evaluate every controller deterministically.
3. Save the full time series before aggregation.
4. Reject malformed or incomplete episodes using automatic validation.
5. Report every selected scenario and seed, including negative outcomes.
6. Use final held-out Task 5 seeds only after training and checkpoint selection.
7. Report uncertainty and avoid universal claims from small seed sets.

---

## 12. Limitations and threats to validity

### 12.1 Simulation-to-reality gap

All findings are produced in SUMO. Driver behaviour, detector errors,
pedestrians, weather, incidents and road-user non-compliance are simplified or
absent. Real-world deployment would require calibration and safety review.

### 12.2 Limited networks

Tasks 1–4 use one intersection, and Task 5 uses a four-signal 2×2 grid. Results
should not be generalised automatically to irregular city networks or large
corridors.

### 12.3 Seed counts

Task 1 uses ten seeds, Task 2 uses three per scenario, Tasks 3–4 use five per
scenario, and Task 5 uses five final seeds. These are reasonable for a laptop
B.Tech project but smaller than publication-scale evaluation.

### 12.4 Metric interpretation

“Mean system total waiting time” is the mean over recorded SUMO snapshots of
the system-level waiting measure; it is not necessarily the same as mean trip
delay per individual vehicle. Metric definitions must remain consistent when
comparing controllers.

### 12.5 Multi-objective evidence

DQN-v2 includes CO2 and fairness penalties, but the final tables do not contain
separate CO2 or lane-fairness outcomes. The project may state that these terms
were implemented, but should not claim measured emissions/fairness improvement
without additional evaluation.

### 12.6 Expert dependence in Task 5

Task 5 uses five max-pressure expert episodes and an expert imitation term. Its
final expert agreement above 99.6% indicates strong policy similarity. The
pressure mechanism is omitted from the main final controller table as requested
but must be disclosed as part of training and shielding.

### 12.7 Task 5 shield ablation

Only one shield intervention occurred in the final evaluation, so raw and
shielded aggregate results are identical. This shows that the selected model
rarely needed correction on these seeds; it does not prove the shield is
unnecessary under more severe or unseen disturbances.

### 12.8 Training instability

Task 5 validation degraded at episode 40. Held-out model selection handled the
issue, but the result shows that longer training is not automatically better.

---

## 13. Concluding extensions implemented as Tasks 6–9

The four previously proposed extensions are now implemented as controlled,
resumable studies:

1. **Task 6 — component sensitivity:** frozen-policy, paired feature-group
   occlusion with bootstrap uncertainty and explicit non-causal interpretation.
2. **Task 7 — incident and sensor robustness:** demand surge, temporary lane
   blockage, Gaussian noise, sensor dropout and delayed observations, including
   degradation and recovery analysis.
3. **Task 8 — transfer learning:** zero-shot, fine-tuned and scratch conditions
   under a directional demand shift with paired budgets, held-out selection and
   an unseen reverse-shift test.
4. **Task 9 — explainability:** gradients, distribution-preserving permutation,
   occlusion, action/Q-value behaviour, fixed representative cases and
   cross-seed explanation stability.

Their code is complete, but numerical claims must wait for final validation and
result artifacts. The complete protocol is documented in
[CONCLUDING_RESEARCH_TASKS_6_TO_9.md](CONCLUDING_RESEARCH_TASKS_6_TO_9.md).

An emissions extension should first add explicit CO2 evaluation columns so the
multi-objective reward can be supported by measured environmental results.

---

## 14. Suggested work distribution for three group members

| Member | Primary responsibility | Supporting responsibility |
|---|---|---|
| Member 1 | Reproducibility, dynamic routes and statistical tables | Proposal methodology and validation |
| Member 2 | DQN-v2 and PPO implementation/analysis | Reward and algorithm comparison |
| Member 3 | Multi-intersection shared DQN | Safety, interaction and checkpoint analysis |

All members should understand the paired-seed protocol and Task 5 limitations
because those affect the final claims and viva questions.

---

## 15. Reproduction commands

Run from the repository root after setting `SUMO_HOME` and activating the
Semester 2 environment.

```powershell
python sem-2/validate_workspace.py

python sem-2/01-reproducible-benchmark/run_benchmark.py --seconds 3600 --seeds 101 102 103 104 105 106 107 108 109 110

python sem-2/02-dynamic-traffic/evaluate_scenarios.py --seconds 3600 --seeds 201 202 203

python sem-2/03-multiobjective-dqn/train_dqn_v2.py --timesteps 60000 --chunk 10000 --seconds 1800 --device auto
python sem-2/03-multiobjective-dqn/evaluate_dqn_v2.py --seconds 3600 --seeds 301 302 303 304 305

python sem-2/04-ppo-comparison/train_ppo.py --timesteps 60000 --chunk 10240 --seconds 1800 --device auto
python sem-2/04-ppo-comparison/evaluate_ppo.py --seconds 3600 --seeds 401 402 403 404 405

python sem-2/05-multi-intersection/self_check.py
python sem-2/05-multi-intersection/train_dqn.py --episodes 40 --seconds 1800
python sem-2/05-multi-intersection/evaluate_dqn.py --seconds 1800 --seeds 701 702 703 704 705 --device cpu

python sem-2/06-component-ablation/run_ablation.py --seconds 1800 --seeds 1001 1002 1003 1004 1005 --device cpu

python sem-2/07-robustness-stress-test/run_stress_test.py --seconds 1800 --seeds 1101 1102 1103 1104 1105 --device cpu

python sem-2/08-transfer-learning/generate_target_routes.py
python sem-2/08-transfer-learning/train_transfer.py --episodes 12 --seconds 1800 --device auto
python sem-2/08-transfer-learning/evaluate_transfer.py --seconds 1800 --seeds 1301 1302 1303 1304 1305 --device cpu

python sem-2/09-explainability/run_explainability.py --seconds 1800 --seeds 1401 1402 1403 --max-analysis-samples 4000 --device cpu
```

Existing complete episode files are retained on rerun. Do not use `--force` or
`--fresh` unless intentionally starting a new experiment under a separately
archived result set.

---

## 16. Key repository evidence index

| Evidence | Location |
|---|---|
| Semester 2 overview | [README.md](README.md) |
| Workspace integrity report | [workspace_validation.json](workspace_validation.json) |
| Protected Semester 1 manifest | [sem1-readonly-manifest.json](sem1-readonly-manifest.json) |
| Common evaluation implementation | [common/experiment_utils.py](common/experiment_utils.py) |
| Task 1 results | [01-reproducible-benchmark/results](01-reproducible-benchmark/results) |
| Task 2 results | [02-dynamic-traffic/results](02-dynamic-traffic/results) |
| Task 3 results | [03-multiobjective-dqn/results](03-multiobjective-dqn/results) |
| Task 4 results | [04-ppo-comparison/results](04-ppo-comparison/results) |
| Task 5 results | [05-multi-intersection/results](05-multi-intersection/results) |
| Tasks 6–9 concluding protocol | [CONCLUDING_RESEARCH_TASKS_6_TO_9.md](CONCLUDING_RESEARCH_TASKS_6_TO_9.md) |
| Task 6 implementation | [06-component-ablation](06-component-ablation) |
| Task 7 implementation | [07-robustness-stress-test](07-robustness-stress-test) |
| Task 8 implementation | [08-transfer-learning](08-transfer-learning) |
| Task 9 implementation | [09-explainability](09-explainability) |

---

## 17. Recommended conclusion for the proposal/report

This project demonstrates a progression from a reproducible
single-intersection DQN to an interaction-aware multi-intersection controller.
The experiments show that learned controllers can substantially reduce
congestion relative to fixed timing, but also show that reward design,
algorithm choice and traffic distribution create meaningful trade-offs. The
final shared Dueling Double DQN provides the strongest evidence: it outperformed
fixed timing on every primary metric across five unseen paired seeds while
passing all structural checks. Its expert-guided training and close agreement
with pressure-based actions are explicitly acknowledged, preserving an honest
interpretation of the contribution.

The work is suitable for an eighth-semester B.Tech project because it combines
reinforcement-learning theory, traffic simulation, reproducible software,
controlled experiments, negative-result analysis and multi-agent scaling while
remaining executable on consumer laptop hardware.

---

## 18. Reference foundations for the written report

The final university proposal should format references according to the
required institutional style. The technical foundations used by this project
include:

1. Deep Q-Networks for value-based control.
2. Double DQN for reduced action-value overestimation.
3. Dueling network architectures for separate state-value and action-advantage
   estimation.
4. Prioritised experience replay.
5. Proximal Policy Optimisation.
6. SUMO microscopic traffic simulation.
7. SUMO-RL traffic signal reinforcement-learning environments.

Repository-specific mathematical descriptions are also available in
`../MATHEMATICAL_MODEL_COMPLETE.md`, `../MATHEMATICS_REFERENCES_EQUATIONS.md`
and `../TECHNICAL_SPECIFICATIONS_DETAILED.md`.
