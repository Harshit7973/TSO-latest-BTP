# Tasks 6–9 implementation smoke-test record

Date: 31 August 2026

Environment used:

- Python 3.11.9
- PyTorch 2.11.0 CPU build
- pandas 3.0.1
- NumPy 2.4.3
- Eclipse SUMO 1.26.0
- Selected Task 5 checkpoint fingerprint:
  `b23826d00901aaa0cda41381e47d23f00a3790c2127e5a915ff953789d062e78`

## Important status

This record proves that every new code path executes and produces structurally
valid artifacts. It is **not** the final research result. The smoke runs use
short horizons and one evaluation seed; their positive or negative performance
flags must not be quoted in the dissertation. Final commands and predeclared
decision rules are in `CONCLUDING_RESEARCH_TASKS_6_TO_9.md`.

## Checks completed

| Check | Outcome |
|---|---|
| Python abstract-syntax parse | 22/22 Semester 2 Python files passed |
| Workspace/Semester 1 integrity | passed; all five protected hashes unchanged |
| Task 8 generated route XML | both domains accepted by SUMO |
| Shared feature contract | all 26 indices covered exactly once |
| Sensor fault semantics | dropout and two-decision delay assertions passed |
| Paired metric direction | all four synthetic improvement signs passed |
| Demand and incident hook | SUMO accepted 1.4 scale; lane applied and restored |
| Task 6 smoke experiment | 6/6 episode validations passed |
| Task 7 smoke experiment | 6/6 episode validations passed |
| Task 8 smoke training | fine-tuned and scratch runs completed with checkpoints |
| Task 8 smoke evaluation | 8/8 episode validations passed across two domains |
| Task 9 smoke audit | 480/480 decision rows; all correctness checks passed |
| Resume behaviour | Tasks 6–9 retained complete artifacts on identical reruns |
| Strict JSON | saved analyses re-aggregated with non-finite values rejected |

## Smoke commands

```powershell
python sem-2/06-component-ablation/run_ablation.py --seconds 600 --seeds 1091 --device cpu --bootstrap-resamples 1000

python sem-2/07-robustness-stress-test/run_stress_test.py --seconds 600 --seeds 1191 --scenarios nominal gaussian_noise --device cpu --bootstrap-resamples 1000

python sem-2/08-transfer-learning/train_transfer.py --run-name quick --episodes 2 --episodes-per-session 0 --seconds 600 --validation-every 1 --validation-seconds 600 --validation-seeds 1291 1292 --device cpu
python sem-2/08-transfer-learning/evaluate_transfer.py --run-name quick --seconds 600 --seeds 1293 --device cpu --bootstrap-resamples 1000

python sem-2/09-explainability/run_explainability.py --run-name quick --seconds 600 --seeds 1491 --max-analysis-samples 400 --device cpu
```

## Defects found and corrected during testing

1. Task 8 used attribute-style access for a column named `mode`; pandas 3
   resolves that name to the DataFrame method. It now uses explicit
   `frame["mode"]` access. The identical rerun completed from saved checkpoints
   without repeating training.
2. Task 9's one-seed stability standard deviation was undefined. It is now
   reported as zero for the pipeline check, while the final three-seed run
   computes genuine cross-seed dispersion.
3. Numeric-looking SUMO intersection IDs were inferred as integers after CSV
   reload. Q-trace filtering now preserves the inferred ID type.
4. Tasks 8 and 9 quick artifacts now live under `runs/quick/`, preventing any
   overwrite of final checkpoints or analyses.
5. Task 8 now has an immutable training manifest and rejects changes to seeds,
   routes or learning hyperparameters during resume.

These corrections were made before the implementation was marked ready for
the final laptop experiments.
