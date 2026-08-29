# Task 5 — Cooperative multi-intersection control

## Purpose and reused repository components

Semester 1 trained only the single two-way intersection. This task uses the
previously unused multi-agent environment and 2×2/RESCO networks already in the
reference repository.

Four agents control the default 2×2 grid. Two scientifically useful variants
are compared:

- **Independent:** every intersection learns only from its local reward.
- **Cooperative:** local reward is mixed with the mean team reward, representing
  limited interaction between neighbouring controllers.

Tabular agents are intentionally used so this task remains achievable on a
laptop while demonstrating genuine multi-intersection coordination. The 16-
agent RESCO grid is provided only as an optional stronger experiment.

## Setup

Use the shared Python/SUMO environment from Task 1. Training is primarily CPU-
bound; the RTX 3050 is not required for tabular learning.

## Training and resume

First run a short API/pipeline test:

```powershell
python sem-2/05-multi-intersection/train_multiagent.py --mode independent --episodes 2 --seconds 600
python sem-2/05-multi-intersection/train_multiagent.py --mode cooperative --episodes 2 --seconds 600
```

Recommended final-year experiment:

```powershell
python sem-2/05-multi-intersection/train_multiagent.py --mode independent --episodes 15 --seconds 1800 --fresh
python sem-2/05-multi-intersection/train_multiagent.py --mode cooperative --episodes 15 --seconds 1800 --fresh
```

Use `--fresh` once for each mode when moving from the 600-second check to the
real 1,800-second run; omit it on later resume commands. `--episodes` is a
cumulative target. Each invocation runs three episodes by default, saves an
atomic latest checkpoint after every episode, and exits. Rerun the identical
command until episode 15, or use `--episodes-per-session 0` for an uninterrupted
run. Generous runtime is 1–4 hours per 15-episode mode on 2×2.

Optional 16-agent extension, only if the main results are already complete:

```powershell
python sem-2/05-multi-intersection/train_multiagent.py --network resco4x4 --mode cooperative --episodes 20 --seconds 1800
```

Allow roughly 6–20 hours for this optional run and split it across sessions.
Do not make 4×4 completion necessary for the final submission.

## Evaluation

```powershell
python sem-2/05-multi-intersection/evaluate_multiagent.py --network 2x2 --seconds 1800 --seeds 601 602 603 604 605
```

Missing modes are skipped, allowing intermediate evaluation. Completed files
are retained on rerun.

## Outputs

- Per-episode and final Q-table checkpoints for both modes.
- Raw training/evaluation CSV files.
- `results/training_summary.csv`.
- Fixed-versus-agent and cooperative-versus-independent paired CSVs.
- Training and evaluation plots.
- Validation JSON for every final episode.

## Correctness checks

1. The 2×2 environment must report four agent IDs.
2. Q-table state counts should become non-zero and generally grow across early
   episodes.
3. Epsilon must decay from 0.30 toward 0.02 and persist after resume.
4. Evaluation must be deterministic for known states and use the same SUMO seed
   for every controller.
5. Inspect `unseen_states` in raw evaluation CSVs. A high count means more
   training or better state generalisation is needed.
6. All validation entries must pass and teleportation must be reported.
7. Cooperative control is successful if it reduces network-level waiting or
   queues relative to independent learning without significantly reducing
   completed trips. If it does not, report the negative result and explain the
   state-space/exploration limitation.
