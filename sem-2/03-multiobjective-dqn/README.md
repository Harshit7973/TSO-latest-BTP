# Task 3 — Enhanced multi-objective DQN-v2

## Purpose and contribution

This is the principal Semester 2 model contribution. It extends DQN-v1 with:

- Current-phase elapsed time.
- Per-lane accumulated waiting time.
- Outgoing-lane density.
- Enforced maximum green time.
- Queue, CO₂, fairness and switching penalties in addition to delay reduction.
- Training on traffic that changes direction within an episode.
- Deterministic but different SUMO seeds across successive training episodes.

The task performs an ablatable, academically meaningful improvement without
requiring research-scale hardware.

## Setup

Use Python 3.11 and complete the shared setup from Task 1. Verify CUDA with:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

MLP-based DQN and SUMO are frequently CPU-bound; using the RTX 3050 is allowed,
but a GPU does not guarantee a large speed-up.

## Training and resume

Pipeline check:

```powershell
python sem-2/03-multiobjective-dqn/train_dqn_v2.py --timesteps 5000 --chunk 2500 --seconds 600
```

Recommended run:

```powershell
python sem-2/03-multiobjective-dqn/train_dqn_v2.py --timesteps 60000 --chunk 10000 --seconds 1800 --device auto --fresh
```

Stronger final model if time permits:

```powershell
python sem-2/03-multiobjective-dqn/train_dqn_v2.py --timesteps 100000 --chunk 10000 --seconds 1800 --device auto
```

Use `--fresh` once when changing from the 600-second pipeline check to the real
1,800-second experiment. Omit `--fresh` on later resume commands. By default,
each command trains one 10,000-step chunk, saves the model/replay buffer/state,
and exits; rerun the same command until the 60,000-step target is reached. Use
`--chunks-per-session 0` only when you intentionally want one uninterrupted
run. The last fully saved chunk remains safe if interruption happens mid-chunk.
`--fresh` starts a new in-memory run but deliberately does not delete numbered
old checkpoints.

Generous estimates are 1–4 hours for 60,000 steps and 2–7 hours for 100,000
steps on a laptop; thermal throttling and SUMO CPU performance dominate.

## Evaluation

Quick:

```powershell
python sem-2/03-multiobjective-dqn/evaluate_dqn_v2.py --seconds 600 --seeds 301 --scenarios direction_switch
```

Recommended:

```powershell
python sem-2/03-multiobjective-dqn/evaluate_dqn_v2.py --seconds 3600 --seeds 301 302 303 304 305
```

Evaluation resumes by skipping completed CSVs. Expect approximately 2–6 hours
for the recommended 45 episodes, which can be split by scenario.

## Saved artifacts

- `checkpoints/dqn_v2_latest.zip`, plus `dqn_v2_final.zip` after the target is reached.
- `checkpoints/dqn_v2_replay_buffer.pkl` for resume.
- `checkpoints/training_state.json` and numbered checkpoints.
- `logs/training_session_*.monitor.csv` and TensorBoard logs.
- `plots/training_curve.png` and evaluation plots.
- `results/comparison_summary_sec<seconds>.csv` and three paired-comparison CSVs.
- Raw episode CSVs and `validation.json`.

## Correctness and research acceptance

1. The observation space reported by the new model must match the enhanced
   environment during both training and evaluation.
2. Training return need not rise monotonically, but its rolling curve should
   not become NaN or diverge permanently.
3. All evaluation validation entries must pass.
4. Compare DQN-v2 with DQN-v1 using identical scenario/seed pairs.
5. Report all three selected scenarios, including failures.
6. Teleportation should preferably be zero.
7. A useful DQN-v2 should improve at least two congestion metrics on average
   without materially reducing throughput or starving one direction. This is a
   project acceptance target, not permission to discard contrary results.

For an ablation study, train separate output copies after setting individual
reward weights in `features.py` to zero. Record each configuration rather than
overwriting the primary experiment.
