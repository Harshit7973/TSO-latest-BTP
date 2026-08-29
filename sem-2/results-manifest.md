# Semester 2 results manifest

Generated files belong only below `sem-2`. The Semester 1 model and outputs in
`btp/` are treated as read-only inputs.

For every final experiment, record:

- Git commit ID.
- SUMO, Python, PyTorch and Stable-Baselines3 versions.
- GPU model and whether CUDA was used.
- Exact command.
- Traffic scenario and seeds.
- Training timesteps/episodes and checkpoint used.
- Raw per-episode CSV files.
- Aggregated summary CSV.
- Validation JSON.
- All report plots.

Do not manually edit raw CSV files. If a run is invalid, retain it with a note
and rerun using a new output name.
