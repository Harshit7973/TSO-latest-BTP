# Task 7 — Incident and sensor-fault robustness

## Research purpose

Good nominal performance does not establish operational robustness. Task 7
tests the frozen Task 5 controller under physical disruption, excess demand and
imperfect observations. Fixed, raw DQN and shielded DQN receive identical
scenario/seed pairs.

## Stress scenarios

| Scenario | Controlled disturbance |
|---|---|
| `nominal` | Original Task 5 network and demand |
| `demand_surge` | SUMO demand scaled to 1.4× |
| `partial_lane_blockage` | Lane `-h11_0` reduced to 1 m/s from 600–1,200 s |
| `gaussian_noise` | Gaussian noise, σ=0.15, on normalised sensor features |
| `sensor_dropout` | 20% of sensor values independently replaced by zero |
| `delayed_observation` | Sensor features delayed by two decisions |

Sensor corruption changes the DQN input but not the simulator's underlying
traffic. The shield uses true simulator pressure. Consequently:

- `dqn_raw` measures the learned policy's direct sensor-fault sensitivity.
- `dqn_shielded` measures layered control with an ideal pressure safety channel.

This is an explicit limitation, not a hidden assumption.

## Analytical outputs

- Paired DQN-versus-fixed improvements with 10,000-resample bootstrap CIs.
- Degradation of every controller relative to its nominal result.
- Incident recovery time using a documented pre-incident queue threshold.
- Sensor corruption, shield intervention and expert-agreement rates.
- Per-scenario success decisions and an overall robustness target.
- Grouped plots with uncertainty bars.

## Quick pipeline check

Use nominal and one sensor scenario:

```powershell
python sem-2/07-robustness-stress-test/run_stress_test.py --seconds 600 --seeds 1191 --scenarios nominal gaussian_noise --device cpu
```

The lane-blockage scenario requires at least 1,500 seconds and is intentionally
excluded from this quick check.

## Recommended final run

```powershell
python sem-2/07-robustness-stress-test/run_stress_test.py --seconds 1800 --seeds 1101 1102 1103 1104 1105 --device cpu
```

This is 90 simulations. Allow **5–14 hours** on a CPU laptop. It is safe to
interrupt the complete command between episodes and rerun exactly the same
command: completed scenario/method/seed files are retained. The configuration
guard rejects a different scenario list or duration in the same results tree,
which prevents incompatible experiments from being mixed accidentally.

## Saved evidence

- `results/episodes/sec1800/`: 90 raw time series.
- `results/stress_summary_sec1800.csv`.
- `results/degradation_vs_nominal_sec1800.csv`.
- `results/incident_recovery_sec1800.csv`.
- `results/dqn_*_vs_fixed__<scenario>_sec1800.csv`.
- `results/analysis_sec1800.json` and `.md`.
- `results/validation_sec1800.json`.
- `plots/stress_sec1800_*.png`.

## Honest success criteria

A scenario passes when shielded DQN beats fixed waiting and stopped vehicles on
at least four of five seeds and loses no more than 5% mean throughput. The
overall target requires at least four of six scenarios and zero teleports. A
failed stress is a useful robustness boundary and must be reported rather than
deleted or retuned on the final seeds.
