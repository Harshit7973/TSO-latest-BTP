# Task 2 — Dynamic traffic and generalisation

## Purpose

The original route changes direction only after 25,000 seconds, while Semester
1 episodes last 3,600 seconds. This task compresses meaningful demand changes
into a single episode and checks whether the frozen Semester 1 DQN generalises.

It creates six scenarios: balanced, north–south peak, east–west peak,
direction switching, sudden bursts, and an unseen mixed demand. All generated
route files remain inside this task and do not replace repository networks.

## Setup

Complete the shared setup in `sem-2/01-reproducible-benchmark/README.md`.

## Run

Generate/inspect route files only:

```powershell
python sem-2/02-dynamic-traffic/generate_routes.py --seconds 3600
```

Quick check:

```powershell
python sem-2/02-dynamic-traffic/evaluate_scenarios.py --seconds 600 --seeds 201 --scenarios balanced direction_switch burst
```

Recommended laptop experiment:

```powershell
python sem-2/02-dynamic-traffic/evaluate_scenarios.py --seconds 3600 --seeds 201 202 203
```

For final publication-quality evidence, use five seeds if laptop time permits.
Completed scenario/method/seed files are skipped on restart.

Expected generous runtime is 3–10 minutes per fixed/DQN pair and 2–6 hours for
the default 36 episodes. Split it across sessions by passing two or three
scenarios at a time; the final rerun will collect all existing results.

## Outputs

- `generated-routes/sec<seconds>/*.rou.xml`: task-owned SUMO demand definitions.
- `generated-routes/manifest.json`: requested traffic volumes.
- `results/episodes/sec<seconds>/`: raw time series.
- `results/summary_by_scenario_seed_sec<seconds>.csv`: episode statistics.
- `results/paired_improvements_sec<seconds>.csv`: paired generalisation result.
- `results/validation_sec<seconds>.json`: automated integrity checks.
- `plots/`: grouped report figures for all scenarios.

## Correctness checks

1. SUMO must load every generated route without route errors.
2. `validation.json` must pass for every episode.
3. Each selected scenario must have both fixed and DQN results for every seed.
4. Arrived/departed counters must be monotonic and teleportation should be zero
   or explicitly discussed.
5. Direction-switch plots should visibly respond near 900, 1,800 and 2,700
   simulation seconds in a 3,600-second run.
6. Do not expect DQN-v1 to win every scenario. Poor performance on east–west or
   burst traffic is a valid generalisation finding and motivates Task 3.

Expected ranges depend on scenario demand. Correctness is established by paired
inputs and valid counters—not by forcing an improvement.
