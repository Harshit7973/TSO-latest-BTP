# Task 5 automatic result analysis

| Controller | Waiting ↓ | Stopped ↓ | Speed ↑ | Throughput/h ↑ |
|---|---:|---:|---:|---:|
| dqn_raw | 23.455 | 4.084 | 7.957 | 1384.400 |
| dqn_shielded | 23.455 | 4.084 | 7.957 | 1384.400 |
| fixed | 225.373 | 10.794 | 6.384 | 1377.600 |

Shielded DQN success check: **True**

A success flag is evidence from these held-out seeds, not a universal guarantee.
Raw DQN is an ablation; shielded DQN is the principal project controller.
Max pressure is used internally for expert guidance and the safety mask, but is not a final comparison row.
