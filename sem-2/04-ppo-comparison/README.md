# Task 4 — PPO algorithm comparison

## Purpose

This task adapts the repository's previously unused PPO/Stable-Baselines3 idea
to the Semester 2 single-intersection experiment. PPO provides a meaningful
modern policy-gradient comparison without making the project excessively heavy.

It is an algorithm comparison, not a claim that PPO must outperform DQN.
Successive training episodes use deterministic but different SUMO seeds.

## Setup

Use the shared Python/SUMO installation from Task 1. `sem-2/requirements.txt`
already includes Stable-Baselines3 and TensorBoard support.

## Training and resume

Quick test:

```powershell
python sem-2/04-ppo-comparison/train_ppo.py --timesteps 5120 --chunk 2560 --seconds 600
```

Recommended:

```powershell
python sem-2/04-ppo-comparison/train_ppo.py --timesteps 60000 --chunk 10240 --seconds 1800 --device auto --fresh
```

Use `--fresh` once when changing from the quick 600-second check to the real
1,800-second run; omit it when resuming. By default, each invocation completes
one rollout chunk, saves, and exits. Rerun the same command until the target is
reached, or pass `--chunks-per-session 0` for one uninterrupted run. For a
stronger run, increase the cumulative target to 100,000. PPO resumes from
`checkpoints/ppo_latest.zip`. Because PPO collects full
512-step rollouts, its recorded timesteps can slightly exceed the requested
target. This is expected.

Generous laptop estimates are 1–4 hours for 60,000 steps and 3–8 hours for
100,000. CPU can be as fast as CUDA for this small MLP; report the actual device.

## Evaluation

```powershell
python sem-2/04-ppo-comparison/evaluate_ppo.py --seconds 3600 --seeds 401 402 403 404 405
```

The recommended evaluation runs fixed timing, DQN-v1 and PPO on three dynamic
scenarios and may require 2–6 hours. Split by `--scenarios` and resume later.

## Outputs

- Numbered, latest and final PPO checkpoints.
- Session-preserving monitor CSVs and TensorBoard logs.
- `plots/training_curve.png`.
- Raw paired evaluation episodes.
- `results/comparison_summary_sec<seconds>.csv`.
- `results/ppo_vs_fixed.csv` and `ppo_vs_dqn.csv`.
- Confidence-interval plots and validation JSON.

## Correctness checks

1. Training return and losses must remain finite.
2. Evaluation model and environment must both use the standard 21-value
   Semester 1 observation space.
3. Every controller must use identical scenario/seed pairs.
4. Validation JSON must pass, counters must be monotonic, and teleportation
   must be reported.
5. Report PPO even if it performs below DQN. A correct negative comparison is
   stronger than selecting only a favourable episode.
6. A useful model should exceed fixed timing on at least two primary congestion
   metrics without materially reducing final throughput, but this is an
   engineering target rather than a guaranteed output range.
