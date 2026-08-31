# Task 8 automatic transfer-learning analysis

| Domain | Method | Waiting ↓ | Stopped ↓ | Speed ↑ | Throughput/h ↑ | Score ↓ |
|---|---|---:|---:|---:|---:|---:|
| target_horizontal | fixed | 339.292 | 15.500 | 5.329 | 1266.000 | 494.292 |
| target_horizontal | zero_shot | 24.642 | 4.550 | 7.612 | 1392.000 | 70.142 |
| target_horizontal | fine_tuned | 24.642 | 4.550 | 7.612 | 1392.000 | 70.142 |
| target_horizontal | scratch | 111.075 | 15.083 | 4.363 | 1242.000 | 261.908 |
| reverse_vertical | fixed | 250.875 | 12.300 | 6.110 | 1308.000 | 373.875 |
| reverse_vertical | zero_shot | 19.608 | 3.750 | 8.275 | 1386.000 | 57.108 |
| reverse_vertical | fine_tuned | 19.608 | 3.750 | 8.275 | 1386.000 | 57.108 |
| reverse_vertical | scratch | 114.317 | 9.217 | 6.522 | 1350.000 | 206.483 |

Structural completion: **True**.
Fine-tuned target score better than zero-shot: **False**.
Fine-tuned target score better than scratch: **True**.
Fine-tuned validation AUC better than scratch: **True**.

A false performance check is an honest negative transfer result, not a failed pipeline.
The reverse-vertical domain measures retained generalisation after horizontal fine-tuning.
