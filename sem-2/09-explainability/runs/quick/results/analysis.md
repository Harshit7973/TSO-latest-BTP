# Task 9 automatic explainability report

Decision rows analysed: **480**.
Expert agreement: **99.38%**.
Shield intervention rate: **0.0000%**.

| Rank | Feature group | Pooled importance | Seed mean ± 95% CI | Permutation flip rate | Mean |gradient| |
|---:|---|---:|---:|---:|---:|
| 1 | phase | 0.33236 | 0.34079 ± 0.00000 | 0.38000 | 0.554496 |
| 2 | minimum_green | 0.21002 | 0.23137 ± 0.00000 | 0.18500 | 1.282679 |
| 3 | incoming_queue | 0.13901 | 0.11752 ± 0.00000 | 0.15750 | 1.017689 |
| 4 | pressure | 0.13785 | 0.12313 ± 0.00000 | 0.17500 | 0.824523 |
| 5 | outgoing_queue | 0.05375 | 0.05992 ± 0.00000 | 0.04000 | 0.597717 |
| 6 | local_queue | 0.04478 | 0.04487 ± 0.00000 | 0.00500 | 0.711895 |
| 7 | intersection_identity | 0.03727 | 0.03695 ± 0.00000 | 0.00750 | 0.370860 |
| 8 | network_context | 0.03024 | 0.03109 ± 0.00000 | 0.00250 | 0.466102 |
| 9 | simulation_progress | 0.01472 | 0.01438 ± 0.00000 | 0.00000 | 0.198620 |

Permutation preserves each feature group's empirical distribution while breaking its row-level association.
Gradient saliency measures local sensitivity. Agreement between the two strengthens an interpretation,
but neither proves causality because traffic features are correlated and the model was expert-guided.

All correctness checks passed: **True**.
