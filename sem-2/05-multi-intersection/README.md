# Task 5 — Compact-v2 cooperative multi-intersection control

## Current context: read this before running or editing

Semester 1 trained only a single-intersection DQN. Task 5 extends the project
to the unused four-signal 2×2 SUMO network and compares:

- Fixed-time control.
- Independent Q-learning: each signal uses its local reward.
- Cooperative Q-learning: local reward is mixed with a network team reward.

The first Task 5 implementation was trained for 15 episodes and executed
correctly, but it did **not** generalise to held-out seeds. Preserve those files
as the `v1` negative result; do not delete or overwrite them.

### Handoff status for the next coding agent

- The v1 training/evaluation results currently in this folder are real pulled
  results and are the evidence used in the diagnosis below.
- The `compact_v2` code is the corrective implementation. It has passed syntax
  and isolated policy-logic checks, but it has **not yet been trained in the
  repository**, so no performance claim should be made before the laptop run.
- The immediate job is to run the quick pipeline check, then complete both
  30-episode modes, then run the five held-out seeds in Step 3.
- Do not modify or delete unversioned v1 outputs. Compact-v2 uses distinct
  checkpoint, result, episode and plot names specifically to preserve them.

### What happened in the first run

Five held-out 1,800-second evaluations produced:

| Controller | Mean waiting measure | Mean stopped | Speed | Throughput/h |
|---|---:|---:|---:|---:|
| Fixed | 232.11 | 11.09 | 6.36 | 1396.8 |
| Cooperative v1 | 2429.86 | 36.19 | 4.62 | 1165.6 |
| Independent v1 | 3172.09 | 44.94 | 4.50 | 1161.6 |

Although the final training episodes looked good, exact unseen-state rates were
19–58% for cooperative control and 55–69% for independent control on most test
seeds. One independent seed had only 0.3% unseen states and performed extremely
well. This proves that the main problem was the original state representation,
not a failed SUMO run.

The old encoding discretised every lane density and queue into ten bins. The
Cartesian product was too large for a 15-episode Q-table. For an unseen state,
the old evaluator merely held the current phase, which could starve another
direction and produce very large queues.

## Compact-v2 correction

The new implementation is deliberately versioned as `compact_v2`. It never
loads the old checkpoints and writes to new filenames/directories.

### 1. Compact phase-pressure state

For each intersection:

```text
(current phase, minimum-green flag,
 pressure bin for action 0, pressure bin for action 1,
 pressure bin for action 2, pressure bin for action 3)
```

For each possible phase:

```text
pressure = queued vehicles on served incoming lanes
           - queued vehicles on served outgoing lanes
```

Pressure is mapped into five bins: non-positive, 1–2, 3–5, 6–10, and above 10.
For a four-action signal, the theoretical state count is approximately:

```text
4 current phases × 2 timing flags × 5^4 pressure combinations = 5,000
```

This is dramatically smaller and more transferable than ten bins for every
individual lane feature.

### 2. Queue-based learning reward

The default reward is negative local queue (`reward_fn="queue"`). It is a
stable immediate signal for short tabular training.

Independent mode uses:

```text
effective reward = local reward
```

Cooperative mode uses the default 0.5 coordination weight:

```text
team reward      = mean reward across all intersections
effective reward = 0.5 × local reward + 0.5 × team reward
```

### 3. Pressure-guided exploration

Exploratory decisions use max-pressure control 70% of the time and a random
phase 30% of the time. This teaches the Q-table from safer, congestion-aware
experience while retaining true exploration.

### 4. Safe unseen-state fallback

During evaluation, known compact states use the learned Q-table. A genuinely
unseen state uses deterministic max-pressure control rather than holding the
current phase. Every fallback is counted and reported, so this hybrid design
must be described honestly in the report.

For known states, the learned action is also protected by a conservative
starvation guard. It is overridden only when another phase has at least five
more pressure units. Override counts and rates are saved separately; this is a
safety layer, not an unreported replacement for Q-learning.

The final evaluator also runs pure max-pressure control on exactly the same
seeds. This is a meaningful adaptive baseline and shows whether learning adds
value beyond a standard traffic-control heuristic.

### 5. Held-out validation and best-checkpoint selection

Every five training episodes, the current policy is evaluated without learning
on seeds 9501 and 9502. The score is:

```text
validation score = mean waiting + 10 × mean stopped vehicles
```

The lowest-score checkpoint becomes `*_compact_v2_best.pkl`. When training
reaches the target, the validation-selected policy—not automatically the last
episode—is copied to `*_compact_v2_final.pkl`.

## Files that define compact-v2

- `multiagent_policy.py`: pressure state, max-pressure fallback and evaluation
  episode runner.
- `train_multiagent.py`: resumable Q-learning and held-out checkpoint selection.
- `evaluate_multiagent.py`: five-seed fixed, max-pressure and learned-policy
  comparison with fallback/override diagnostics.

An assisting coding agent must preserve the `compact_v2` tag and must not reuse
`independent_2x2_latest.pkl` or `cooperative_2x2_latest.pkl`; those are the
failed v1 checkpoints.

## Setup

Run from the repository root in PowerShell after completing the shared setup:

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
.\.venv-sem2\Scripts\Activate.ps1
python sem-2/validate_workspace.py
```

Training is mainly CPU/SUMO-bound. The RTX 3050 is not needed for tabular
learning.

## Step 1 — Quick pipeline check

These are non-final runs:

```powershell
python sem-2/05-multi-intersection/train_multiagent.py --mode independent --episodes 3 --seconds 600 --validation-every 3 --validation-seconds 600 --fresh
python sem-2/05-multi-intersection/train_multiagent.py --mode cooperative --episodes 3 --seconds 600 --validation-every 3 --validation-seconds 600 --fresh
python sem-2/05-multi-intersection/evaluate_multiagent.py --seconds 600 --seeds 9901 --allow-partial
```

Confirm that compact-v2 checkpoints, a training summary, validation history and
evaluation CSV are created without exceptions.

## Step 2 — Recommended final training

Start a clean 30-episode independent experiment:

```powershell
python sem-2/05-multi-intersection/train_multiagent.py --mode independent --episodes 30 --seconds 1800 --fresh
```

The command runs three episodes, saves, and exits. Continue with the same
command **without** `--fresh` until it reports episode 30:

```powershell
python sem-2/05-multi-intersection/train_multiagent.py --mode independent --episodes 30 --seconds 1800
```

Repeat the same process for cooperative control:

```powershell
python sem-2/05-multi-intersection/train_multiagent.py --mode cooperative --episodes 30 --seconds 1800 --fresh
python sem-2/05-multi-intersection/train_multiagent.py --mode cooperative --episodes 30 --seconds 1800
```

Rerun each resume command as many times as required. Do not change `--seconds`,
reward or experiment settings midway. The script rejects incompatible resume
arguments.

Generous runtime is approximately 2–8 hours per mode, including validation,
depending on CPU speed and laptop temperature. It is safe to split this over
ten short sessions per mode.

## Step 3 — Final held-out evaluation

After both modes reach episode 30:

```powershell
python sem-2/05-multi-intersection/evaluate_multiagent.py --network 2x2 --seconds 1800 --seeds 701 702 703 704 705
```

The evaluator loads validation-selected compact-v2 checkpoints. It does not
touch or reuse the old 601–605 evaluation files.

## Compact-v2 outputs

- `checkpoints/*_compact_v2_latest.pkl`: newest resumable training state.
- `checkpoints/*_compact_v2_best.pkl`: best held-out validation policy.
- `checkpoints/*_compact_v2_final.pkl`: selected final policy after episode 30.
- `results/training-episodes-compact_v2/`: raw training episodes.
- `results/training_summary_compact_v2.csv`: learning history and Q-state count.
- `results/validation_history_*_compact_v2.csv`: held-out checkpoint evidence.
- `results/episodes/2x2_compact_v2_sec1800/`: raw final evaluations.
- `results/evaluation_2x2_compact_v2_sec1800.csv`: controller summary including
  pressure-fallback rate.
- `results/validation_2x2_compact_v2_sec1800.json`: structural validation.
- `plots/*compact_v2*`: training and final comparison plots.

## Acceptance checks

1. Both modes must reach the complete 30-episode target.
2. The evaluator must load `*_compact_v2_final.pkl` and report five seeds.
3. All 20 structural validation entries must pass (four methods × five seeds).
4. Teleported vehicles should be zero or explicitly explained.
5. For each learned policy, mean pressure-fallback rate should preferably be
   below 20%. A larger value means state coverage remains insufficient, even
   if fallback performance is good. Pure max pressure intentionally reports a
   100% fallback rate because it has no Q-table.
6. Inspect override rate as a separate safety diagnostic. A high rate means the
   Q-policy is frequently being corrected by the pressure guard.
7. Compare every learned policy against fixed timing and pure max pressure on
   the same 701–705 seeds.
8. A strong result reduces waiting or stopped vehicles without materially
   reducing throughput. Do not hide a mode that performs worse.
9. Cooperative control is only claimed superior if its paired results beat
   independent control across multiple seeds, not merely one seed.

## If compact-v2 is still weak

Do not immediately increase to the RESCO 4×4 network. Give the coding agent:

- `training_summary_compact_v2.csv`
- Both validation-history CSVs
- `evaluation_2x2_compact_v2_sec1800.csv`
- The structural validation JSON
- Terminal output

Check fallback rate first. If it remains above 20%, extend the cumulative target
to 40 episodes using the same resume command. If fallback is low but performance
is weak, tune the cooperation weight or validation score rather than hiding the
result.

RESCO 4×4 remains optional and should only be attempted after compact-v2 works
on 2×2.
