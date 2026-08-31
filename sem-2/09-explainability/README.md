# Task 9 — Explainable DQN decisions and failure-case analysis

## Research purpose

Task 9 explains how the selected Task 5 neural controller responds to its 26
state features. It combines global sensitivity, behavioural statistics and
local decision examples instead of relying on one attractive heatmap.

## Explanation methods

1. **Gradient saliency:** absolute gradient of the selected Q-value with
   respect to each input, aggregated by feature group.
2. **Distribution-preserving permutation:** values for one group are shuffled
   between real decision rows. This preserves the empirical marginal
   distribution while breaking its association with the current traffic state.
3. **Zero occlusion:** one group is set to zero as a secondary sensitivity
   check.
4. **Decision behaviour:** Q-margin, action distribution, expert agreement and
   shield intervention are measured directly.
5. **Representative cases:** highest margin, lowest margin, first expert
   disagreement and first shield intervention are selected by fixed rules.
6. **Cross-seed stability:** importance is recomputed separately per seed;
   mean, 95% uncertainty and rank correlations expose unstable explanations.

The feature groups are phase, minimum-green flag, pressure, incoming queue,
outgoing queue, local queue, network context, simulation progress and
intersection identity.

## Why this is academically useful

- Saves every state, Q-value, action, mask and expert action for audit.
- Uses complementary model-gradient and perturbation methods.
- Does not select only successful or visually convenient decisions.
- Produces both global ranking and local counterfactual tables.
- Explicitly states that sensitivity is not causal explanation.

## Quick pipeline check

```powershell
python sem-2/09-explainability/run_explainability.py --run-name quick --seconds 600 --seeds 1491 --max-analysis-samples 400 --device cpu
```

Quick artifacts are isolated under `runs/quick/`; final outputs remain clean.

## Recommended final run

```powershell
python sem-2/09-explainability/run_explainability.py --seconds 1800 --seeds 1401 1402 1403 --max-analysis-samples 4000 --device cpu
```

This requires only three DQN simulations plus offline inference/gradients.
Allow **1–3 hours** on a CPU laptop. Completed trace files resume safely.

## Saved evidence

- `results/episodes/sec1800/`: episode-level raw simulation traces.
- `results/decisions/sec1800/`: state, Q-values, masks and actions.
- `results/global_feature_importance.csv`.
- `results/feature_importance_by_seed.csv` and `feature_importance_stability.csv`.
- `results/feature_rank_correlation.csv`.
- `results/representative_decisions.csv`.
- `results/local_occlusion_explanations.csv`.
- `results/analysis.json` and `analysis.md`.
- `results/validation_sec1800.json`.
- `plots/global_feature_importance.png`.
- `plots/action_distribution.png`.
- `plots/q_value_trace.png`.
- `plots/action_margin_distribution.png`.

## Correctness and honest interpretation

The default final run expects exactly 4,320 decisions: three seeds × 360
decision times × four intersections. All states/Q-values must be finite, all
actions valid, every feature group present and all three episode validations
passing.

High permutation or gradient importance means the model is sensitive to that
group on sampled decisions. It does not prove that changing a real traffic
variable causes the same effect because features are correlated. The model's
max-pressure expert-guided origin must be disclosed when explaining high
pressure/queue importance.
