# Task 1 — Reproducible paired-seed benchmark

## Purpose

Re-evaluate the frozen Semester 1 DQN against the fixed-time controller using
the same SUMO seed for each pair. This is the minimum evidence required before
claiming an improvement. It corrects the unpaired random-seed evaluation in the
Semester 1 scripts without changing `btp/models/dqn_2way.zip`.

## Setup

Run all commands from the repository root in PowerShell:

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
py -3.11 -m venv .venv-sem2
.\.venv-sem2\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r sem-2/requirements.txt
```

If PyTorch does not detect the RTX 3050, install the CUDA build using the
command provided by pytorch.org for the installed CUDA driver. SUMO simulation
is mostly CPU-bound, so CUDA has limited effect on evaluation speed.

## Run

Quick pipeline check (not reportable):

```powershell
python sem-2/01-reproducible-benchmark/run_benchmark.py --seconds 600 --seeds 101 102
```

Recommended laptop run:

```powershell
python sem-2/01-reproducible-benchmark/run_benchmark.py --seconds 3600 --seeds 101 102 103 104 105 106 107 108 109 110
```

Strong final run, if time permits, uses 20 seeds. Existing episode CSV files
are skipped automatically, so the command can be stopped and rerun. Use
`--force` only when intentionally replacing completed Task 1 results.

Expected generous runtime on a typical RTX 3050 laptop is 2–8 minutes per
3,600-second episode without GUI, or approximately 1–3 hours for 20 paired
episodes. Actual time depends more on CPU and SUMO than GPU. GUI runs can take
several times longer.

## Saved outputs

- `results/episodes/sec<seconds>/`: raw step-level CSV for every method and seed.
- `results/run_config.json`: exact configuration and library versions.
- `results/summary_by_seed_sec<seconds>.csv`: correct episode-level metrics.
- `results/paired_improvements_sec<seconds>.csv`: improvement for each paired seed.
- `results/validation_sec<seconds>.json`: automatic structural checks.
- `plots/`: report-ready comparisons with 95% confidence intervals.

## Correctness and acceptance checks

1. `validation.json` must show `passed: true` for every episode.
2. Fixed and DQN must have exactly the same seed set and simulation length.
3. Departed and arrived counters must be monotonic.
4. Teleported vehicles should normally be zero; investigate any non-zero value.
5. A 10-seed final experiment should contain 20 valid CSV files.
6. Compare paired seeds, not the best episode.
7. The previous saved data suggests DQN waiting time near 200–250 seconds and
   fixed timing near 1,200–1,400 seconds on this route. These are sanity ranges,
   not pass criteria. A materially different but valid result must be reported,
   not deleted.

The primary acceptance criterion is reproducibility and complete paired data,
not a predetermined improvement percentage.
