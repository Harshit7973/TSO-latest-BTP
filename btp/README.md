# BTP — RL-Based Traffic Signal Optimisation

## Quick Start (Run these in order)

> All commands should be run from the repo root: `sumo-rl-main/`

---

## Step 0 — Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

---

## Step 1 — Run Baseline (Fixed Timing)
```powershell
python btp/train_baseline.py --network 2way --seconds 3600
```

---

## Step 2 — Train Q-Learning Agent
```powershell
python btp/train_qlearning.py --network 2way --seconds 3600 --runs 3
```

---

## Step 3 — Train DQN Agent (uses RTX 3050 GPU 🚀)
```powershell
python btp/train_dqn.py --network 2way --seconds 3600 --timesteps 100000
```

---

## Step 4 — Test DQN Agent (with GUI)
```powershell
python btp/test_model.py --model btp/models/dqn_2way.zip --network 2way --gui
```

---

## Step 5 — Compare All Results
```powershell
python btp/compare_results.py --network 2way
```

---

## File Structure After Running
```
btp/
├── models/
│   ├── dqn_2way.zip          ← Trained DQN model
│   ├── qtable_2way.pkl       ← Trained Q-table
│   └── dqn_checkpoints/      ← Saved checkpoints every 10k steps
│
├── outputs/
│   ├── baseline/             ← Fixed timing CSV results
│   ├── qlearning/            ← Q-Learning CSV results
│   ├── dqn/                  ← DQN CSV results + TensorBoard logs
│   ├── test/                 ← Test run results
│   └── plots/                ← Comparison PNG plots (for report)
│
└── README.md
```

---

## Key Parameters to Tune (for BTP experiments)

| Parameter | Default | What it controls |
|---|---|---|
| `--seconds` | 3600 | Simulated traffic duration per episode |
| `--timesteps` | 100000 | Total DQN training steps |
| `--runs` | 3 | Q-Learning episodes |
| `--network` | 2way | Road network (`single`, `2way`, `4x4`) |

---

## Networks Available

| Network | Traffic Signals | Complexity |
|---|---|---|
| `single` | 1 | Simple (start here) |
| `2way` | 1 | Medium (2-directional flow) |
| `4x4` | 16 | Advanced (multi-agent) |

---

## Reward Functions (modify in train scripts)

| Name | Optimises |
|---|---|
| `diff-waiting-time` *(default)* | Total waiting time |
| `queue` | Queue length |
| `average-speed` | Vehicle speed |
| `pressure` | Flow balance |
| `co2` | Emissions |
