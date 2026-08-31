# Task 8 — Transfer learning and limited-budget adaptation

## Research purpose

Task 8 tests whether the completed Task 5 policy provides a useful starting
point when traffic demand changes. It compares **zero-shot reuse**, **fine-
tuning**, and **training from scratch** under the same target route, training
seeds, 12-episode budget, optimiser, replay design and safety shield.

The experiment measures sample efficiency, not only final performance.

## Domain shift

The source 2×2 route uses probability 0.10 for each of four straight flows.
Task 8 deterministically generates:

- `target_horizontal`: horizontal flows 0.16, vertical flows 0.06.
- `reverse_vertical`: horizontal flows 0.06, vertical flows 0.16.

Both domains have the same total flow probability, isolating directional shift
from a simple increase in total demand. Route hashes are recorded.

## Controlled learning comparison

| Method | Initial weights | Target training |
|---|---|---:|
| `zero_shot` | Selected Task 5 checkpoint | 0 episodes |
| `fine_tuned` | Selected Task 5 checkpoint | 12 episodes |
| `scratch` | Deterministic random initialisation | 12 episodes |

Fine-tuned and scratch models use identical seeds 1201–1212. Validation uses
1251–1252 every three episodes. Final evaluation uses unseen seeds 1301–1305.
No expert-imitation loss is used during target training, but both learning
conditions use the same pressure safety mask.

## Generated evidence

- Per-episode target-domain training time series.
- Resumable replay/model/optimizer checkpoints for both methods.
- Validation score at episodes 0, 3, 6, 9 and 12.
- Validation-score area under the curve; lower is better.
- Best validation checkpoint for each method.
- Final target-horizontal and reverse-vertical evaluation.
- Paired 10,000-resample bootstrap intervals.
- Separate performance checks; negative transfer is retained honestly.

## Step 1 — Generate and inspect routes

```powershell
python sem-2/08-transfer-learning/generate_target_routes.py
```

## Step 2 — Quick code-path check

Use the isolated `quick` run so this check cannot overwrite final checkpoints:

```powershell
python sem-2/08-transfer-learning/train_transfer.py --run-name quick --episodes 2 --episodes-per-session 0 --seconds 600 --validation-every 1 --validation-seconds 600 --validation-seeds 1291 1292 --device cpu
python sem-2/08-transfer-learning/evaluate_transfer.py --run-name quick --seconds 600 --seeds 1293 --device cpu --bootstrap-resamples 1000
```

Its artifacts are saved under `runs/quick/`; final artifacts remain untouched.

## Step 3 — Final resumable training

```powershell
python sem-2/08-transfer-learning/train_transfer.py --episodes 12 --seconds 1800 --device auto
```

The default completes two episodes for each learning condition and exits.
Rerun the identical command until both report complete. Allow **4–10 hours**
including validation. Do not change the route, episode count, seeds or
hyperparameters while resuming. The resolved CPU/CUDA device is also recorded;
use the same laptop and `--device` setting for every continuation.

## Step 4 — Final evaluation

```powershell
python sem-2/08-transfer-learning/evaluate_transfer.py --seconds 1800 --seeds 1301 1302 1303 1304 1305 --device cpu
```

This runs 40 simulations: four controllers × two domains × five seeds. Allow
**3–8 hours**. Completed CSVs are retained on rerun.

## Outputs

- `routes/*.rou.xml` and `routes/manifest.json`.
- `checkpoints/fine_tuned_latest.pt`, `fine_tuned_best.pt`.
- `checkpoints/scratch_latest.pt`, `scratch_best.pt`.
- `results/training_episodes/<mode>/`.
- `results/training_summary.csv` and `validation_history.csv`.
- `results/training_analysis.json`.
- `results/evaluation_summary_sec1800.csv`.
- `results/analysis_sec1800.json` and `.md`.
- `results/validation_sec1800.json`.
- `plots/sample_efficiency.png` and final metric plots.

## Honest interpretation

Fine-tuning is considered beneficial only when its target-domain score is
better than zero-shot and scratch, and its validation-score AUC is lower than
scratch. These are reported as separate Boolean checks. If fine-tuning fails
one check, report negative transfer or no sample-efficiency advantage. Do not
tune on final seeds 1301–1305.

Because the source model was expert-guided in Task 5 and both target learners
use a pressure shield, this experiment evaluates transfer of an expert-guided
representation—not knowledge learned without domain constraints.
