# Task 7 automatic robustness analysis

| Scenario | Shielded waiting improvement vs fixed | Queue improvement | Throughput improvement | Waiting wins |
|---|---:|---:|---:|---:|
| nominal | 86.33% | 58.59% | 4.00% | 1/1 |
| gaussian_noise | 87.34% | 59.89% | 2.67% | 1/1 |

Structural validation: **6/6 passed**.

Robustness target passed: **True**.

Sensor faults corrupt the learned observation. The shield still uses true simulator pressure;
therefore raw DQN is the pure perception-fault result and shielded DQN is the layered-safety result.
A failed scenario is a reportable robustness boundary, not permission to discard the seed.
