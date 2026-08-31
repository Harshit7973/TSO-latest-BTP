"""Generate auditable global and local explanations for the Task 5 DQN."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


TASK_DIR = Path(__file__).resolve().parent
SEM2_ROOT = TASK_DIR.parent
sys.path.insert(0, str(SEM2_ROOT))
sys.path.insert(0, str(SEM2_ROOT / "05-multi-intersection"))

from common.experiment_utils import (  # noqa: E402
    collect_episode_summaries,
    ensure_output_tree,
    require_sumo,
    set_global_seed,
)
from common.multi_intersection_tools import (  # noqa: E402
    FEATURE_GROUPS,
    load_task5_model,
    run_extended_episode,
    write_json,
)
from dqn_core import checkpoint_path, resolve_device  # noqa: E402


def batched_q(model, states: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            tensor = torch.as_tensor(states[start : start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model(tensor).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def explanation_analysis(
    decisions: pd.DataFrame,
    *,
    model,
    device: torch.device,
    max_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if len(decisions) > max_samples:
        selected_indices = np.sort(rng.choice(len(decisions), size=max_samples, replace=False))
        sample = decisions.iloc[selected_indices].reset_index(drop=True)
    else:
        sample = decisions.reset_index(drop=True)
    state_columns = [f"state_{index}" for index in range(26)]
    states = sample[state_columns].to_numpy(dtype=np.float32)
    base_q = batched_q(model, states, device)
    base_actions = base_q.argmax(axis=1)
    base_selected_q = base_q[np.arange(len(states)), base_actions]

    tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
    tensor.requires_grad_(True)
    q_tensor = model(tensor)
    chosen = q_tensor.gather(1, torch.as_tensor(base_actions, device=device).unsqueeze(1)).sum()
    chosen.backward()
    gradients = tensor.grad.detach().abs().cpu().numpy()

    rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    for group, indices_tuple in FEATURE_GROUPS.items():
        indices = list(indices_tuple)
        occluded = states.copy()
        occluded[:, indices] = 0.0
        occluded_q = batched_q(model, occluded, device)
        occluded_actions = occluded_q.argmax(axis=1)
        permuted = states.copy()
        permutation = rng.permutation(len(states))
        permuted[:, indices] = states[permutation][:, indices]
        permuted_q = batched_q(model, permuted, device)
        permuted_actions = permuted_q.argmax(axis=1)
        rows.append(
            {
                "feature_group": group,
                "feature_count": len(indices),
                "mean_abs_gradient": float(gradients[:, indices].mean()),
                "occlusion_action_flip_rate": float(np.mean(occluded_actions != base_actions)),
                "occlusion_mean_abs_chosen_q_change": float(
                    np.mean(np.abs(occluded_q[np.arange(len(states)), base_actions] - base_selected_q))
                ),
                "permutation_action_flip_rate": float(np.mean(permuted_actions != base_actions)),
                "permutation_mean_abs_chosen_q_change": float(
                    np.mean(np.abs(permuted_q[np.arange(len(states)), base_actions] - base_selected_q))
                ),
            }
        )

    importance = pd.DataFrame(rows)
    for column in (
        "mean_abs_gradient",
        "permutation_action_flip_rate",
        "permutation_mean_abs_chosen_q_change",
    ):
        total = float(importance[column].sum())
        importance[f"normalised_{column}"] = importance[column] / total if total > 0 else 0.0
    importance["combined_importance"] = importance[
        [
            "normalised_mean_abs_gradient",
            "normalised_permutation_action_flip_rate",
            "normalised_permutation_mean_abs_chosen_q_change",
        ]
    ].mean(axis=1)
    importance = importance.sort_values("combined_importance", ascending=False).reset_index(drop=True)

    # Local explanations use a small, predeclared set of diagnostically useful
    # decisions rather than selecting cases that make the model look good.
    representative_indices: dict[str, int] = {
        "highest_margin": int(decisions.q_margin.astype(float).idxmax()),
        "lowest_margin": int(decisions.q_margin.astype(float).idxmin()),
    }
    disagreement = decisions[decisions.raw_action != decisions.expert_action]
    if not disagreement.empty:
        representative_indices["expert_disagreement"] = int(disagreement.index[0])
    intervention = decisions[decisions.shield_intervened == 1]
    if not intervention.empty:
        representative_indices["shield_intervention"] = int(intervention.index[0])
    representatives = decisions.loc[list(dict.fromkeys(representative_indices.values()))].copy()
    representatives.insert(
        0,
        "case_labels",
        [
            ",".join(label for label, index in representative_indices.items() if index == row_index)
            for row_index in representatives.index
        ],
    )
    for row_index, decision in representatives.iterrows():
        state = decision[state_columns].to_numpy(dtype=np.float32)[None, :]
        q_values = batched_q(model, state, device)[0]
        action = int(decision.raw_action)
        for group, indices_tuple in FEATURE_GROUPS.items():
            changed = state.copy()
            changed[:, list(indices_tuple)] = 0.0
            changed_q = batched_q(model, changed, device)[0]
            local_rows.append(
                {
                    "decision_index": int(row_index),
                    "case_labels": representatives.loc[row_index, "case_labels"],
                    "seed": int(decision.seed),
                    "step": float(decision.step),
                    "agent": str(decision.agent),
                    "raw_action": action,
                    "executed_action": int(decision.executed_action),
                    "expert_action": int(decision.expert_action),
                    "feature_group": group,
                    "chosen_q": float(q_values[action]),
                    "chosen_q_after_occlusion": float(changed_q[action]),
                    "chosen_q_change": float(changed_q[action] - q_values[action]),
                    "action_after_occlusion": int(np.argmax(changed_q)),
                    "action_flipped": int(np.argmax(changed_q) != action),
                }
            )
    return importance, representatives.reset_index(names="decision_index"), pd.DataFrame(local_rows)


def create_plots(
    importance: pd.DataFrame,
    decisions: pd.DataFrame,
    output_dir: Path,
) -> None:
    ranked = importance.sort_values("combined_importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    values = ranked.get("seed_mean_combined_importance", ranked.combined_importance)
    errors = ranked.get("seed_ci95_half_width", None)
    ax.barh(
        ranked.feature_group,
        values,
        xerr=errors,
        color="#4C78A8",
        capsize=3,
    )
    ax.set_xlabel("Mean combined normalised sensitivity across seeds")
    ax.set_title("Task 9 global DQN feature-group importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "global_feature_importance.png", dpi=180)
    plt.close(fig)

    action_counts = pd.crosstab(decisions.agent, decisions.executed_action, normalize="index")
    fig, ax = plt.subplots(figsize=(8, 5))
    action_counts.plot(kind="bar", stacked=True, ax=ax, colormap="tab20c")
    ax.set_ylabel("Fraction of decisions")
    ax.set_xlabel("Intersection")
    ax.set_title("Executed action distribution by intersection")
    ax.legend(title="Action", ncol=4)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "action_distribution.png", dpi=180)
    plt.close(fig)

    first_seed = int(decisions.seed.min())
    first_agent = sorted(decisions.agent.unique(), key=str)[0]
    trace = decisions[(decisions.seed == first_seed) & (decisions.agent == first_agent)].head(200)
    q_values = trace[[f"q_{index}" for index in range(4)]].to_numpy(dtype=float).T
    fig, ax = plt.subplots(figsize=(11, 4.5))
    image = ax.imshow(q_values, aspect="auto", cmap="coolwarm")
    ax.set_yticks(range(4), [f"Action {index}" for index in range(4)])
    ax.set_xlabel("Decision index")
    ax.set_title(f"Q-value trace: seed {first_seed}, intersection {first_agent}")
    fig.colorbar(image, ax=ax, label="Q-value")
    fig.tight_layout()
    fig.savefig(output_dir / "q_value_trace.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(decisions.q_margin.astype(float), bins=35, color="#F58518", alpha=0.85)
    ax.set_xlabel("Top-Q minus second-Q margin")
    ax.set_ylabel("Decisions")
    ax.set_title("DQN action confidence distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "action_margin_distribution.png", dpi=180)
    plt.close(fig)


def write_report(path: Path, analysis: dict[str, Any], importance: pd.DataFrame) -> None:
    lines = [
        "# Task 9 automatic explainability report",
        "",
        f"Decision rows analysed: **{analysis['decision_rows']}**.",
        f"Expert agreement: **{analysis['behaviour']['expert_agreement_rate'] * 100:.2f}%**.",
        f"Shield intervention rate: **{analysis['behaviour']['shield_intervention_rate'] * 100:.4f}%**.",
        "",
        "| Rank | Feature group | Pooled importance | Seed mean ± 95% CI | Permutation flip rate | Mean |gradient| |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in importance.reset_index(drop=True).iterrows():
        lines.append(
            f"| {rank + 1} | {row.feature_group} | {row.combined_importance:.5f} | "
            f"{row.seed_mean_combined_importance:.5f} ± {row.seed_ci95_half_width:.5f} | "
            f"{row.permutation_action_flip_rate:.5f} | {row.mean_abs_gradient:.6f} |"
        )
    lines.extend(
        [
            "",
            "Permutation preserves each feature group's empirical distribution while breaking its row-level association.",
            "Gradient saliency measures local sensitivity. Agreement between the two strengthens an interpretation,",
            "but neither proves causality because traffic features are correlated and the model was expert-guided.",
            "",
            f"All correctness checks passed: **{analysis['correctness']['passed']}**.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1401, 1402, 1403])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-analysis-samples", type=int, default=4000)
    parser.add_argument(
        "--run-name",
        default="final",
        help="Use 'final' for reportable outputs or an isolated label such as 'quick'",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    require_sumo()
    if args.seconds < 1 or args.seconds % 5 != 0:
        raise SystemExit("--seconds must be a positive multiple of the 5-second decision interval")
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("Explanation seeds must be unique")
    if args.max_analysis_samples < 1:
        raise SystemExit("--max-analysis-samples must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        raise SystemExit("--run-name may contain only letters, digits, '_' and '-'")
    device = resolve_device(args.device)
    experiment_root = TASK_DIR if args.run_name == "final" else TASK_DIR / "runs" / args.run_name
    paths = ensure_output_tree(experiment_root)
    checkpoint = checkpoint_path(SEM2_ROOT / "05-multi-intersection/checkpoints", "final")
    model, payload, fingerprint = load_task5_model(checkpoint, device)
    pressure_gap = float(payload["pressure_gap"])

    episodes_dir = paths["episodes"] / f"sec{args.seconds}"
    decisions_dir = paths["results"] / "decisions" / f"sec{args.seconds}"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    decisions_dir.mkdir(parents=True, exist_ok=True)
    config_path = paths["results"] / f"evaluation_config_sec{args.seconds}.json"
    config = {
        "run_name": args.run_name,
        "command_line": sys.argv,
        "seconds": args.seconds,
        "seeds": args.seeds,
        "device": str(device),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": fingerprint,
        "pressure_gap": pressure_gap,
        "max_analysis_samples": args.max_analysis_samples,
        "methods": ["gradient_saliency", "zero_occlusion", "distribution_preserving_permutation"],
    }
    if config_path.exists() and any(decisions_dir.glob("*.csv")) and not args.force:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        immutable = ("seconds", "seeds", "checkpoint_sha256", "pressure_gap", "max_analysis_samples")
        if any(previous.get(key) != config.get(key) for key in immutable):
            raise SystemExit("Existing Task 9 traces use a different configuration; archive them first")
    write_json(config_path, config | {"status": "in_progress"})

    for seed in args.seeds:
        episode_path = episodes_dir / f"dqn_shielded__seed{seed}.csv"
        decision_path = decisions_dir / f"decisions__seed{seed}.csv"
        if episode_path.exists() and decision_path.exists() and not args.force:
            print(f"[resume] keeping seed {seed}")
            continue
        set_global_seed(seed)
        print(f"[run] explanation trace seed={seed}")
        episode, decisions = run_extended_episode(
            seconds=args.seconds,
            seed=seed,
            controller="dqn_shielded",
            pressure_gap=pressure_gap,
            model=model,
            device=device,
            capture_decisions=True,
        )
        episode.to_csv(episode_path, index=False)
        decisions.to_csv(decision_path, index=False)

    summary, validations = collect_episode_summaries(episodes_dir, args.seconds)
    summary = summary[summary.seed.isin(set(args.seeds))].copy()
    expected_stems = {f"dqn_shielded__seed{seed}" for seed in args.seeds}
    validations = [
        row for row in validations if Path(str(row.get("file", ""))).stem in expected_stems
    ]
    decisions = pd.concat(
        [pd.read_csv(decisions_dir / f"decisions__seed{seed}.csv") for seed in args.seeds],
        ignore_index=True,
    )
    expected_decisions = len(args.seeds) * (args.seconds // 5) * 4
    structural = bool(
        len(summary) == len(args.seeds)
        and len(validations) == len(args.seeds)
        and all(bool(row.get("passed")) for row in validations)
        and len(decisions) == expected_decisions
    )
    if not structural:
        raise SystemExit(
            f"Task 9 incomplete: episodes={len(summary)}/{len(args.seeds)}, "
            f"decisions={len(decisions)}/{expected_decisions}"
        )
    state_values = decisions[[f"state_{index}" for index in range(26)]].to_numpy(dtype=float)
    q_values = decisions[[f"q_{index}" for index in range(4)]].to_numpy(dtype=float)
    finite = bool(np.isfinite(state_values).all() and np.isfinite(q_values).all())
    valid_actions = bool(
        decisions.raw_action.between(0, 3).all()
        and decisions.executed_action.between(0, 3).all()
        and decisions.expert_action.between(0, 3).all()
    )

    importance, representatives, local = explanation_analysis(
        decisions,
        model=model,
        device=device,
        max_samples=args.max_analysis_samples,
        seed=2026,
    )
    per_seed_frames: list[pd.DataFrame] = []
    per_seed_budget = max(1, args.max_analysis_samples // len(args.seeds))
    for seed in args.seeds:
        seed_decisions = decisions[decisions.seed == seed].reset_index(drop=True)
        seed_importance, _, _ = explanation_analysis(
            seed_decisions,
            model=model,
            device=device,
            max_samples=per_seed_budget,
            seed=2026 + seed,
        )
        seed_importance.insert(0, "seed", seed)
        seed_importance["rank"] = np.arange(1, len(seed_importance) + 1)
        per_seed_frames.append(seed_importance)
    per_seed_importance = pd.concat(per_seed_frames, ignore_index=True)
    stability = (
        per_seed_importance.groupby("feature_group")["combined_importance"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "seed_mean_combined_importance",
                "std": "seed_std_combined_importance",
            }
        )
    )
    stability["seed_ci95_half_width"] = (
        1.96 * stability.seed_std_combined_importance.fillna(0.0) / np.sqrt(len(args.seeds))
    )
    stability["seed_std_combined_importance"] = stability[
        "seed_std_combined_importance"
    ].fillna(0.0)
    importance = importance.merge(stability, on="feature_group", how="left")
    rank_pivot = per_seed_importance.pivot(
        index="feature_group", columns="seed", values="combined_importance"
    ).rank(axis=0, ascending=False)
    rank_correlation = rank_pivot.corr()
    importance.to_csv(paths["results"] / "global_feature_importance.csv", index=False)
    per_seed_importance.to_csv(paths["results"] / "feature_importance_by_seed.csv", index=False)
    stability.to_csv(paths["results"] / "feature_importance_stability.csv", index=False)
    rank_correlation.to_csv(paths["results"] / "feature_rank_correlation.csv")
    representatives.to_csv(paths["results"] / "representative_decisions.csv", index=False)
    local.to_csv(paths["results"] / "local_occlusion_explanations.csv", index=False)
    summary.to_csv(paths["results"] / f"episode_summary_sec{args.seconds}.csv", index=False)
    write_json(paths["results"] / f"validation_sec{args.seconds}.json", validations)

    correctness = {
        "structural_validation": structural,
        "expected_decision_rows": len(decisions) == expected_decisions,
        "finite_states_and_q_values": finite,
        "valid_actions": valid_actions,
        "all_feature_groups_present": set(importance.feature_group) == set(FEATURE_GROUPS),
        "importance_is_finite": bool(np.isfinite(importance.select_dtypes(include=[np.number])).all().all()),
        "per_seed_feature_groups_complete": bool(
            all(
                set(group.feature_group) == set(FEATURE_GROUPS)
                for _, group in per_seed_importance.groupby("seed")
            )
        ),
    }
    correctness["passed"] = all(correctness.values())
    analysis = {
        "run_name": args.run_name,
        "checkpoint_sha256": fingerprint,
        "seeds": args.seeds,
        "decision_rows": len(decisions),
        "analysis_sample_rows": min(len(decisions), args.max_analysis_samples),
        "behaviour": {
            "expert_agreement_rate": float((decisions.executed_action == decisions.expert_action).mean()),
            "raw_expert_agreement_rate": float((decisions.raw_action == decisions.expert_action).mean()),
            "shield_intervention_rate": float(decisions.shield_intervened.mean()),
            "mean_q_margin": float(decisions.q_margin.mean()),
            "median_q_margin": float(decisions.q_margin.median()),
        },
        "ranked_feature_groups": importance.to_dict("records"),
        "feature_rank_correlation_across_seeds": {
            str(row_seed): {
                str(column_seed): float(value)
                for column_seed, value in row.items()
            }
            for row_seed, row in rank_correlation.to_dict(orient="index").items()
        },
        "representative_case_labels": representatives.case_labels.tolist(),
        "correctness": correctness,
        "limitations": [
            "Gradient and permutation importance describe sensitivity, not causal traffic effects.",
            "Feature groups are correlated, so importance can be shared or displaced between groups.",
            "The explained model was trained with max-pressure expert guidance.",
            "Three explanation seeds are for behavioural coverage, not population-wide inference.",
        ],
    }
    analysis_path = paths["results"] / "analysis.json"
    report_path = paths["results"] / "analysis.md"
    write_json(analysis_path, analysis)
    write_report(report_path, analysis, importance)
    create_plots(importance, decisions, paths["plots"])
    write_json(config_path, config | {"status": "complete"})
    print(f"Task 9 complete: {len(decisions)} decisions, correctness={correctness['passed']}")
    print(f"Analysis: {analysis_path}")


if __name__ == "__main__":
    main()
