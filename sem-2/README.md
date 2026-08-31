# Semester 2 — Adaptive and multi-intersection traffic signal optimisation

This directory is an isolated Semester 8 B.Tech project workspace. It treats
the Semester 1 model and outputs under `btp/` as read-only evidence. Every new
model, route, checkpoint, CSV and plot is saved inside its owning task folder.

## Research narrative

Semester 1 established that a DQN could control one two-way SUMO intersection.
Semester 2 asks progressively stronger questions:

1. Is the reported improvement reproducible on paired traffic seeds?
2. Does the frozen model generalise when traffic demand changes?
3. Can enhanced state and a multi-objective reward improve robustness?
4. How does DQN compare with PPO under the same conditions?
5. Can a shared, interaction-aware DQN control four intersections robustly?

These tasks reuse previously unused parts of the reference repository—custom
observations/rewards, CO₂ metrics, PPO, PettingZoo-style multi-agent control,
2×2 networks and RESCO—without modifying the reference implementation.

## Task map

| Task | Deliverable | Typical recommended laptop budget |
|---|---|---:|
| `01-reproducible-benchmark` | Paired fixed vs frozen DQN-v1 evidence | 1–3 h |
| `02-dynamic-traffic` | Six dynamic/unseen demand scenarios | 2–6 h |
| `03-multiobjective-dqn` | New DQN-v2 plus v1/v2 ablation | 3–10 h total |
| `04-ppo-comparison` | PPO training and fair comparison | 3–10 h total |
| `05-multi-intersection` | Shared robust DQN on a four-signal 2×2 grid | 5–16 h total |

Times are deliberately generous estimates. SUMO is often CPU-bound and laptop
thermal limits matter more than the nominal RTX 3050 speed. Every long task is
resumable; no experiment requires keeping the laptop on for the entire budget.

## One-time Windows setup

Use Python 3.11 for broad compatibility:

```powershell
cd D:\TSO\TSO-latest-BTP
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
py -3.11 -m venv .venv-sem2
.\.venv-sem2\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r sem-2/requirements.txt
```

For subsequent sessions:

```powershell
cd D:\TSO\TSO-latest-BTP
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
.\.venv-sem2\Scripts\Activate.ps1
```

First verify isolation and required files:

```powershell
python sem-2/validate_workspace.py
```

## Recommended execution order

1. Run each task's quick command to detect setup errors.
2. Complete Task 1 before making any performance claim.
3. Complete Task 2 to establish DQN-v1's generalisation limitation.
4. Train and evaluate DQN-v2 as the principal contribution.
5. Train PPO as the algorithmic comparison.
6. Train and evaluate the shared shielded DQN in Task 5.
7. Attempt RESCO 4×4 only after all primary results are safely stored.

Each task README contains exact commands, runtime ranges, saved artifact names
and acceptance checks. Final runs should follow `results-manifest.md`.

## Instructions for the person running and testing the tasks

Run the tasks one by one in this order:

1. `01-reproducible-benchmark`
2. `02-dynamic-traffic`
3. `03-multiobjective-dqn`
4. `04-ppo-comparison`
5. `05-multi-intersection`

Before starting any experiment, activate the Semester 2 virtual environment,
set `SUMO_HOME`, and run:

```powershell
python sem-2/validate_workspace.py
```

For each task:

1. Open that task's `README.md`.
2. Run its quick/pipeline-check command first.
3. Check the terminal for errors and inspect its validation JSON.
4. Proceed to the recommended experiment only after the quick check succeeds.
5. Do not manually edit or delete raw CSV results to improve the reported
   outcome. Unexpected or negative results are valid findings.

Long-running work is deliberately divided into resumable sessions:

- DQN-v2 completes one training chunk per invocation by default.
- PPO completes one rollout chunk per invocation by default.
- Multi-intersection learning completes three episodes per invocation by
  default.
- Rerun the same command without `--fresh` to continue training.
- Evaluation scripts skip completed method/scenario/seed CSV files.
- Use `--force` only when a completed evaluation was invalid and must be
  intentionally replaced.

After completing each task, return or back up the complete task folder,
including:

- `results/`
- `plots/`
- `checkpoints/`
- `logs/`, where present
- Generated route files, where present
- The exact command used
- A copy or screenshot of terminal errors if the task failed

Do not begin the optional RESCO 4×4 experiment until Tasks 1–5 have completed
successfully on the smaller networks and their validation reports pass. The
2×2 multi-intersection experiment is the required deliverable; RESCO 4×4 is an
optional extension.

## Checkpoint and result policy

- DQN saves model plus replay buffer every chunk.
- PPO saves after each rollout chunk.
- Multi-intersection DQN saves model, optimizer and replay atomically after
  every episode.
- Evaluation skips completed method/scenario/seed CSVs.
- `--force` is required to replace evaluation files.
- `--fresh` never deletes old checkpoints; manually move old experiments to an
  archive folder before intentionally starting a new final run.

## Academic scope

The primary submission should include Tasks 1–3 and the 2×2 portion of Task 5.
Task 4 is a strong algorithm comparison. RESCO 4×4 is optional because a valid,
well-evaluated 2×2 coordination study is preferable to an incomplete large run.

Do not claim that a task is implemented merely because its code exists. Mark it
complete only after its validation JSON passes and its raw results, summary and
plots are present.
