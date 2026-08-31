# Task 6 automatic ablation analysis

This is a post-training occlusion study of one frozen Task 5 checkpoint.
It measures sensitivity, not the causal effect of retraining an architecture.

| Method | Waiting ↓ | Stopped ↓ | Speed ↑ | Throughput/h ↑ |
|---|---:|---:|---:|---:|
| fixed | 221.058 | 10.933 | 6.521 | 1314.000 |
| full_dqn | 24.883 | 4.325 | 7.900 | 1344.000 |
| raw_dqn | 24.883 | 4.325 | 7.900 | 1344.000 |
| no_network_context | 23.858 | 4.250 | 7.971 | 1350.000 |
| no_identity | 10.317 | 2.550 | 8.803 | 1344.000 |
| no_pressure | 14.683 | 3.375 | 8.413 | 1362.000 |

Structural validation: **6/6 passed**.

Full-DQN project success check: **True**.

Interpret a degraded occlusion as evidence that the frozen model uses that information.
A neutral result does not prove that the feature is unnecessary during training.
