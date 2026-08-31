# Task 5 — Shared robust DQN for multi-intersection control

## Current handoff context

Read this section before running or editing Task 5.

- Semester 1 demonstrated a DQN on one intersection.
- The former Task 5 used tabular Q-learning. Its code, checkpoints, plots and
  generated results were deliberately removed at the project owner's request.
  They remain recoverable from Git commit `3bcaf06` if historical comparison is
  ever required.
- This folder is a fresh implementation. It does not load a Semester 1 model or
  any former Task 5 checkpoint.
- No final result exists until the commands below have been run. Do not claim
  that DQN beats fixed timing merely because the code is present.
- All generated Task 5 artifacts stay inside this folder. Semester 1 and Tasks
  1–4 are not modified.

## Research question

Can one neural controller, shared by four interacting traffic signals, learn a
robust policy that beats fixed timing while remaining safe on unfamiliar
traffic seeds?

The final experiment compares five controllers on identical seeds:

1. Fixed timing.
2. Pure max-pressure control.
3. Raw shared DQN, retained as an honest ablation.
4. Shielded shared DQN, the principal learned controller.
5. The validation-gated deployed controller.

## Why this design is more appropriate than tabular Q-learning

The old Q-table treated each discretised state as unrelated to nearby states.
Multi-intersection state combinations therefore grew rapidly and the agents
changed one another's learning environment.

This implementation uses **parameter sharing**: all four signals train one
Dueling Double DQN. Every environment step contributes four transitions to a
common replay buffer. This provides substantially more experience per neural
network and transfers traffic patterns between geometrically similar signals.

The method is final-year B.Tech level: it includes deep reinforcement learning,
multi-agent parameter sharing, interaction features, expert warm-start,
prioritized replay, Double-DQN targets, a dueling network, safety constraints,
held-out model selection and paired-seed evaluation. It remains practical on
an RTX 3050 because the network has only two 128-unit hidden layers.

```text
four SUMO signals
       │ local pressure + network congestion context
       ▼
one shared Dueling Double DQN
       │ four Q-values per signal
       ▼
near-max-pressure safety mask ──► four executed actions
       │
       └── four transitions/step ──► shared prioritized replay
```

## Model and state

### Shared Dueling Double DQN

The network estimates four action values using separate value and advantage
heads:

```text
Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
```

Double-DQN targets use the online network to select the next action and the
target network to evaluate it. The loss combines prioritized Huber TD loss with
a small expert-imitation loss during replay of expert transitions.

### Interaction-aware 26-dimensional state

Each signal receives:

- Current phase: four-value one-hot vector.
- Minimum-green eligibility flag.
- Pressure for each of four possible phases.
- Incoming queue for each phase.
- Outgoing queue for each phase.
- Local total queue.
- Network mean, maximum and standard deviation of intersection queues.
- Simulation progress.
- Four-value intersection identity vector.

Continuous traffic features use bounded `tanh` normalisation. The network-level
features are the explicit interaction component: each decentralized action is
conditioned on congestion across the four-signal system.

## Robustness mechanisms

### 1. Max-pressure expert warm-start

The first five final-training episodes are controlled by max pressure. These
transitions fill replay with good behaviour and train the DQN using both TD and
imitation losses. This choice is evidence-based: the earlier diagnostic run
showed that max pressure was much stronger than fixed timing.

### 2. Pressure safety shield

After the expert stage, DQN selects actions only from phases whose pressure is
within two vehicles of the maximum-pressure phase. Before minimum green time is
satisfied, only the current phase is valid.

The shield does not hide intervention. Every raw DQN decision, executed action,
expert agreement and shield intervention is counted. `dqn_raw` is also
evaluated separately so the report can distinguish neural learning from the
safety layer.

### 3. Prioritized replay and target network

High-error transitions are sampled more frequently, importance weights are
annealed toward unbiased updates, gradients are clipped and the target network
is updated every 750 gradient steps.

### 4. Held-out checkpoint selection

Every five episodes, shielded DQN is evaluated without learning on seeds 9601
and 9602. The selection score is:

```text
score = mean total waiting measure + 10 × mean stopped vehicles
```

The lowest-score checkpoint is retained instead of assuming the last episode
is best. A second `best_qualified` checkpoint tracks the strongest validation
episode that also passes the deployment requirements, so a qualifying model is
not discarded merely because another checkpoint has slightly lower congestion
but unacceptable throughput.

### 5. Deployment gate

The selected DQN is deployed only if validation shows:

- Score at most 85% of fixed timing's score.
- Throughput at least 97% of fixed timing's throughput.

If it fails, `deployment_policy` becomes `max_pressure`. This does not convert a
failed DQN into a successful DQN claim. It prevents an unsafe or weak learned
model from being presented as the deployable system. Final evaluation still
reports raw and shielded DQN separately.

## One-time setup

Run from the repository root in PowerShell:

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
py -3.11 -m venv .venv-sem2
.\.venv-sem2\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r sem-2/requirements.txt
python sem-2/validate_workspace.py
```

For later sessions, set `SUMO_HOME` and reactivate `.venv-sem2` before running.

## Step 1 — Offline self-check

This checks state dimensions, action masking, replay serialization, finite
losses and a real optimizer update without starting SUMO:

```powershell
python sem-2/05-multi-intersection/self_check.py
```

Expected result: `"passed": true`. The machine-readable output is saved to
`results/self_check.json`.

## Step 2 — Short pipeline run

This verifies SUMO, training, validation, checkpointing and evaluation. These
are not report results:

```powershell
python sem-2/05-multi-intersection/train_dqn.py --episodes 6 --episodes-per-session 0 --seconds 600 --expert-episodes 2 --validation-every 3 --validation-seconds 600 --fresh
python sem-2/05-multi-intersection/evaluate_dqn.py --seconds 600 --seeds 9901
```

Both commands must complete without exceptions. Inspect the generated analysis
JSON and Markdown file, but do not mix the 600-second output with final results.

## Step 3 — Recommended final training

Start a new 40-episode, 1,800-second experiment:

```powershell
python sem-2/05-multi-intersection/train_dqn.py --episodes 40 --seconds 1800 --fresh
```

The default command runs five episodes, saves atomically and exits. Resume using
the identical command without `--fresh`:

```powershell
python sem-2/05-multi-intersection/train_dqn.py --episodes 40 --seconds 1800
```

Repeat the resume command until the terminal reports training complete and
prints the deployment policy. Do not change seconds, seed, pressure gap,
coordination weight or architecture while resuming.

Expected generous runtime on an RTX 3050 laptop is approximately **4–12 hours
total**, split over eight sessions. SUMO remains CPU-bound; the GPU accelerates
only the small neural updates. Actual time depends strongly on CPU and laptop
temperature.

Use `--device cpu` only if CUDA causes an installation problem. Do not increase
episodes merely to obtain a favourable result; held-out validation determines
the selected checkpoint.

## Step 4 — Final held-out evaluation

After training reaches 40/40:

```powershell
python sem-2/05-multi-intersection/evaluate_dqn.py --seconds 1800 --seeds 701 702 703 704 705
```

This runs 25 paired simulations: five controllers × five seeds. A generous
runtime allowance is **1–4 hours**. Completed raw episodes are retained on
rerun. The evaluator fingerprints the final checkpoint and refuses to mix raw
episodes from different models. Use `--force` only when an evaluation was
invalid or intentionally belongs to a newly trained checkpoint.

## Saved outputs

### Checkpoints

- `checkpoints/shared_dueling_ddqn_v1_latest.pt`: full resumable state including
  optimizer, prioritized replay and random-number states.
- `checkpoints/shared_dueling_ddqn_v1_best.pt`: best held-out DQN checkpoint.
- `checkpoints/shared_dueling_ddqn_v1_best_qualified.pt`: best DQN checkpoint
  that passes both deployment requirements.
- `checkpoints/shared_dueling_ddqn_v1_final.pt`: selected checkpoint plus the
  deployment-gate decision.
- `checkpoints/shared_dueling_ddqn_v1_episode_*.pt`: model-only milestones every
  five episodes.

### Training evidence

- `results/training_episodes_shared_dueling_ddqn_v1/`: raw episode CSVs.
- `results/training_summary_shared_dueling_ddqn_v1.csv`.
- `results/validation_history_shared_dueling_ddqn_v1.csv`.
- `results/validation_baselines_shared_dueling_ddqn_v1.json`.
- `results/deployment_gate_shared_dueling_ddqn_v1.json`.
- `plots/training_shared_dueling_ddqn_v1.png`.
- `plots/validation_shared_dueling_ddqn_v1.png`.

### Final evidence

- `results/episodes/shared_dueling_ddqn_v1_sec1800/`: raw paired episodes.
- `results/evaluation_shared_dueling_ddqn_v1_sec1800.csv`.
- `results/validation_shared_dueling_ddqn_v1_sec1800.json`.
- `results/analysis_shared_dueling_ddqn_v1_sec1800.json`.
- `results/analysis_shared_dueling_ddqn_v1_sec1800.md`.
- Paired comparison CSVs for fixed and max-pressure baselines.
- Four final metric plots under `plots/`.

## Correctness and acceptance checks

The evaluator stops rather than silently accepting incomplete data.

Required checks:

1. Training reaches 40/40 and the final checkpoint records the selected
   validation episode.
2. All expected raw episodes reach exactly 1,800 simulation seconds.
3. All 25 structural validation entries pass when raw DQN is included.
4. Teleported vehicles are zero or explicitly investigated.
5. The same traffic seed is used for every controller in each paired trial.
6. Final evaluation seeds must not overlap training or validation seeds; the
   evaluator checks this automatically.
7. Shield rate and expert-agreement rate are reported, not hidden.
8. DQN success requires lower waiting and queues on at least four of five seeds
   and mean throughput no worse than 3% below fixed timing.
9. `deployed` success and `dqn_shielded` success are reported separately.

The final automatic analysis contains Boolean success checks. A `true` value is
evidence for these five held-out seeds, not proof that every possible traffic
pattern will improve.

## How to interpret possible outcomes

### Shielded DQN passes

The main claim can be that parameter-sharing DQN, expert guidance and safety
constraints produced a learned multi-intersection controller that beat fixed
timing on held-out paired seeds. Report raw DQN and max pressure as ablations.

### Shielded DQN fails but deployed control passes

State clearly that the learned candidate did not satisfy deployment criteria
and the safety process selected max pressure. This remains a useful engineering
result, but it is not evidence that DQN beat fixed timing.

### Raw DQN fails while shielded DQN passes

Conclude that the neural value function was useful only with domain-informed
action constraints. Report the shield rate to quantify that dependence.

## Instructions for another coding agent

When diagnosing returned results, provide these files first:

- Training summary and validation history.
- Deployment-gate JSON.
- Final evaluation CSV.
- Final analysis JSON and Markdown.
- Structural validation JSON.
- Terminal output or error traceback.

Do not remove the max-pressure baseline, raw-DQN ablation, deployment gate or
paired seeds to make results appear stronger. Do not tune on final seeds
701–705. Hyperparameter changes must use training seeds and validation seeds
9601–9602, followed by a fresh final evaluation.
