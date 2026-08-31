# Task 6 — Controlled component ablation and sensitivity study

## Research purpose

Task 5 performed well, but a final report should ask which information the
frozen policy actually uses. This task runs a controlled **post-training
occlusion study** on the selected Task 5 checkpoint. It evaluates the same
model and paired seeds after removing one state group at a time.

This is more rigorous than reading neural weights, but it is not identical to
retraining a new architecture without a feature. The automatic report states
that limitation explicitly.

## Compared methods

1. `fixed`: fixed signal timing.
2. `full_dqn`: complete shielded Task 5 controller.
3. `raw_dqn`: complete DQN without the action shield.
4. `no_network_context`: network mean/max/std queue features set to zero.
5. `no_identity`: four-value intersection identity set to zero.
6. `no_pressure`: four phase-pressure inputs set to zero.

The last three variants use the same frozen weights. The shield continues to
use the true simulator pressure, so its interventions are reported and raw DQN
remains the unshielded reference.

## Why this is a final-year contribution

- Uses controlled paired counterfactual inputs rather than anecdotal examples.
- Reports 10,000-resample paired bootstrap confidence intervals.
- Separates model sensitivity from shield dependence.
- Saves raw episodes and an explicit limitation statement.
- Prevents checkpoint/result mixing with a SHA-256 fingerprint.

## Setup and offline check

Complete the shared setup in `sem-2/README.md`, then run:

```powershell
python sem-2/validate_workspace.py
python sem-2/05-multi-intersection/self_check.py
```

Task 5 training and final checkpoint must already be complete.

## Quick pipeline check

```powershell
python sem-2/06-component-ablation/run_ablation.py --seconds 600 --seeds 1091 --device cpu
```

This is not a reportable experiment. Archive its `results/episodes/sec600`
folder before intentionally repeating a different 600-second configuration.

## Recommended final run

```powershell
python sem-2/06-component-ablation/run_ablation.py --seconds 1800 --seeds 1001 1002 1003 1004 1005 --device cpu
```

The final run contains 30 simulations. A generous CPU-laptop allowance is
**2–6 hours**. Completed method/seed CSVs are retained, so rerun the identical
command after interruption. Do not use `--force` unless the existing run was
invalid and has been archived.

## Saved evidence

- `results/episodes/sec1800/`: 30 raw time series.
- `results/ablation_summary_sec1800.csv`.
- `results/*_vs_fixed_sec1800.csv`.
- `results/*_vs_full_dqn_sec1800.csv`.
- `results/analysis_sec1800.json` and `.md`.
- `results/validation_sec1800.json`.
- `plots/sec1800_*.png`.
- `plots/component_degradation_sec1800.png`.

## Correctness and interpretation

The run is structurally complete only when all 30 validation entries pass,
every method has the same five seeds, all simulations reach 1,800 seconds and
teleportation is reported. A positive component-degradation bar means that
occluding the feature worsened the frozen controller on average.

Do not claim that a neutral occlusion proves a feature was useless during
training. Do not claim causal architectural importance without retraining
controlled variants. Negative or negligible component effects are valid
findings and must remain in the report.
