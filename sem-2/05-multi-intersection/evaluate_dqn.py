"""Evaluate the shared DQN, its safety shield, and traffic-control baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_DIR = Path(__file__).resolve().parent
SEM2_ROOT = TASK_DIR.parent
sys.path.insert(0, str(SEM2_ROOT))
sys.path.insert(0, str(TASK_DIR))

from common.experiment_utils import (  # noqa: E402
    REPO_ROOT,
    collect_episode_summaries,
    ensure_output_tree,
    mean_ci,
    paired_improvements,
    plot_method_comparison,
    require_sumo,
    set_global_seed,
    write_json,
)
from dqn_core import (  # noqa: E402
    ACTION_COUNT,
    EXPERIMENT_TAG,
    STATE_DIM,
    DuelingQNetwork,
    checkpoint_path,
    load_torch_checkpoint,
    resolve_device,
    run_controller_episode,
)


NETWORK = {
    "net": REPO_ROOT / "sumo_rl/nets/2x2grid/2x2.net.xml",
    "route": REPO_ROOT / "sumo_rl/nets/2x2grid/2x2.rou.xml",
}
PRIMARY_METRICS = {
    "mean_system_total_waiting_time": False,
    "mean_system_total_stopped": False,
    "mean_system_mean_speed": True,
    "throughput_veh_per_hour": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_diagnostics(summary: pd.DataFrame, episodes_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(episodes_dir.glob("*.csv")):
        parts = path.stem.split("__")
        method = parts[0]
        seed = int(next(part[4:] for part in parts if part.startswith("seed")))
        frame = pd.read_csv(path)
        rows.append(
            {
                "method": method,
                "seed": seed,
                "agent_decisions": int(frame.agent_decisions.iloc[-1]),
                "shield_interventions": int(frame.shield_interventions.iloc[-1]),
                "shield_rate": float(frame.shield_rate.iloc[-1]),
                "expert_agreements": int(frame.expert_agreements.iloc[-1]),
                "expert_agreement_rate": float(frame.expert_agreement_rate.iloc[-1]),
            }
        )
    return summary.merge(pd.DataFrame(rows), on=["method", "seed"], how="left")


def paired_analysis(summary: pd.DataFrame, baseline: str, candidate: str) -> dict[str, Any]:
    left = summary[summary.method == baseline].set_index("seed")
    right = summary[summary.method == candidate].set_index("seed")
    common = sorted(set(left.index) & set(right.index))
    output: dict[str, Any] = {"baseline": baseline, "candidate": candidate, "seeds": common}
    metrics: dict[str, Any] = {}
    for metric, higher_is_better in PRIMARY_METRICS.items():
        base = left.loc[common, metric].astype(float)
        cand = right.loc[common, metric].astype(float)
        improvement = (cand - base) / base.abs() * 100.0
        if not higher_is_better:
            improvement = -improvement
        wins = int((cand > base).sum()) if higher_is_better else int((cand < base).sum())
        metrics[metric] = {
            "mean_improvement_pct": float(improvement.mean()),
            "min_improvement_pct": float(improvement.min()),
            "max_improvement_pct": float(improvement.max()),
            "wins": wins,
            "trials": len(common),
        }
    output["metrics"] = metrics
    return output


def group_statistics(summary: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method, group in summary.groupby("method"):
        method_row: dict[str, Any] = {}
        for metric in PRIMARY_METRICS:
            mean, ci95 = mean_ci(group[metric])
            method_row[metric] = {"mean": mean, "ci95_half_width": ci95}
        method_row["shield_rate"] = float(group.shield_rate.mean())
        method_row["expert_agreement_rate"] = float(group.expert_agreement_rate.mean())
        output[str(method)] = method_row
    return output


def success_check(
    summary: pd.DataFrame,
    validations: list[dict[str, Any]],
    candidate: str,
) -> dict[str, Any]:
    paired = paired_analysis(summary, "fixed", candidate)
    metrics = paired["metrics"]
    congestion_wins = min(
        metrics["mean_system_total_waiting_time"]["wins"],
        metrics["mean_system_total_stopped"]["wins"],
    )
    throughput_change = metrics["throughput_veh_per_hour"]["mean_improvement_pct"]
    candidate_rows = summary[summary.method == candidate]
    required_wins = int(np.ceil(0.8 * len(candidate_rows)))
    zero_teleports = bool((candidate_rows.final_system_total_teleported == 0).all())
    structural_pass = all(bool(item.get("passed")) for item in validations)
    passed = bool(
        congestion_wins >= required_wins
        and throughput_change >= -3.0
        and zero_teleports
        and structural_pass
    )
    return {
        "passed": passed,
        "requirements": {
            "waiting_and_queue_wins_at_least": f"{required_wins}/{len(candidate_rows)}",
            "mean_throughput_change_at_least_pct": -3.0,
            "zero_teleports": True,
            "all_structural_checks": True,
        },
        "observed": {
            "minimum_congestion_wins": congestion_wins,
            "trials": len(candidate_rows),
            "mean_throughput_change_pct": throughput_change,
            "zero_teleports": zero_teleports,
            "structural_pass": structural_pass,
        },
    }


def write_markdown_report(path: Path, analysis: dict[str, Any]) -> None:
    groups = analysis["group_statistics"]
    metric_labels = [
        ("mean_system_total_waiting_time", "Waiting"),
        ("mean_system_total_stopped", "Stopped"),
        ("mean_system_mean_speed", "Speed"),
        ("throughput_veh_per_hour", "Throughput/h"),
    ]
    lines = [
        "# Task 5 automatic result analysis",
        "",
        f"Deployment policy: `{analysis['deployment_policy']}`",
        "",
        "| Controller | Waiting ↓ | Stopped ↓ | Speed ↑ | Throughput/h ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, values in groups.items():
        rendered = [f"{values[key]['mean']:.3f}" for key, _ in metric_labels]
        lines.append(f"| {method} | " + " | ".join(rendered) + " |")
    lines.extend(
        [
            "",
            f"Shielded DQN success check: **{analysis['success_checks']['dqn_shielded']['passed']}**",
            "",
            f"Deployed-controller success check: **{analysis['success_checks']['deployed']['passed']}**",
            "",
            "A success flag is evidence from these held-out seeds, not a universal guarantee.",
            "Raw DQN is an ablation; the shielded and deployed rows are the project controllers.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--seeds", nargs="+", type=int, default=[701, 702, 703, 704, 705])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pressure-gap", type=float, default=None)
    parser.add_argument("--skip-raw", action="store_true", help="Skip the unshielded DQN ablation")
    parser.add_argument("--allow-seed-overlap", action="store_true", help="Allow a diagnostic, non-final overlap")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    require_sumo()
    device = resolve_device(args.device)
    paths = ensure_output_tree(TASK_DIR)
    final_path = checkpoint_path(paths["checkpoints"], "final")
    checkpoint = load_torch_checkpoint(final_path, device)
    if checkpoint is None:
        raise SystemExit(f"Missing final checkpoint: {final_path}. Complete train_dqn.py first.")
    if checkpoint.get("experiment_tag") != EXPERIMENT_TAG:
        raise SystemExit(f"Incompatible checkpoint tag in {final_path}")
    if int(checkpoint.get("state_dim", -1)) != STATE_DIM or int(checkpoint.get("action_count", -1)) != ACTION_COUNT:
        raise SystemExit("Checkpoint architecture does not match this evaluator")
    completed = int(checkpoint.get("training_completed_episode", 0))
    target = int(checkpoint.get("target_episodes", 0))
    if completed < target:
        raise SystemExit(f"Training is incomplete: {completed}/{target}")
    if int(checkpoint.get("seconds", -1)) != args.seconds:
        raise SystemExit(
            f"Checkpoint used {checkpoint.get('seconds')} seconds, evaluation requested {args.seconds}."
        )
    training_seeds = {
        int(checkpoint["seed"]) + episode for episode in range(1, target + 1)
    }
    validation_seeds = {int(seed) for seed in checkpoint.get("validation_seeds", [])}
    overlap = set(args.seeds) & (training_seeds | validation_seeds)
    if overlap and not args.allow_seed_overlap:
        raise SystemExit(
            f"Evaluation seeds overlap training/validation seeds: {sorted(overlap)}. "
            "Choose genuinely held-out seeds."
        )
    pressure_gap = (
        float(checkpoint["pressure_gap"])
        if args.pressure_gap is None
        else float(args.pressure_gap)
    )
    if not np.isclose(pressure_gap, float(checkpoint["pressure_gap"])):
        raise SystemExit("Evaluation pressure gap must match training")

    model = DuelingQNetwork().to(device)
    model.load_state_dict(checkpoint["online_state_dict"])
    model.eval()
    deployment_policy = str(checkpoint.get("deployment_policy", "max_pressure"))
    checkpoint_sha256 = sha256(final_path)

    methods = ["fixed", "max_pressure"]
    if not args.skip_raw:
        methods.append("dqn_raw")
    methods.extend(["dqn_shielded", "deployed"])
    episodes_dir = paths["episodes"] / f"{EXPERIMENT_TAG}_sec{args.seconds}"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    config_path = paths["results"] / f"evaluation_config_{EXPERIMENT_TAG}.json"
    if config_path.exists() and any(episodes_dir.glob("*.csv")) and not args.force:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("checkpoint_sha256") != checkpoint_sha256:
            raise SystemExit(
                "Existing raw evaluations belong to a different final checkpoint. "
                "Archive them or rerun with --force."
            )
    evaluation_config = {
        "command_line": sys.argv,
        "seconds": args.seconds,
        "seeds": args.seeds,
        "device": str(device),
        "pressure_gap": pressure_gap,
        "methods": methods,
        "deployment_policy": deployment_policy,
        "checkpoint": str(final_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
    }
    write_json(config_path, evaluation_config | {"status": "in_progress"})
    for seed in args.seeds:
        for method in methods:
            output = episodes_dir / f"{method}__seed{seed}.csv"
            if output.exists() and not args.force:
                print(f"[resume] keeping {output.name}")
                continue
            set_global_seed(seed)
            print(f"[run] controller={method}, seed={seed}, device={device}")
            frame = run_controller_episode(
                network_config=NETWORK,
                seconds=args.seconds,
                seed=seed,
                controller=method,
                pressure_gap=pressure_gap,
                model=model,
                device=device,
                deployment_policy=deployment_policy,
            )
            frame.to_csv(output, index=False)

    summary, validations = collect_episode_summaries(episodes_dir, args.seconds)
    summary = add_diagnostics(summary, episodes_dir)
    summary = summary[
        summary.method.isin(methods) & summary.seed.isin(set(args.seeds))
    ].copy()
    expected_stems = {
        f"{method}__seed{seed}" for method in methods for seed in set(args.seeds)
    }
    validations = [
        item
        for item in validations
        if Path(str(item.get("file", ""))).stem in expected_stems
    ]
    expected_methods = set(methods)
    actual_methods = set(summary.method.unique())
    if actual_methods != expected_methods:
        raise SystemExit(f"Expected methods {sorted(expected_methods)}, found {sorted(actual_methods)}")
    expected_rows = len(methods) * len(set(args.seeds))
    if len(summary) != expected_rows:
        raise SystemExit(f"Expected {expected_rows} valid evaluation rows, found {len(summary)}")
    if len(validations) != expected_rows or not all(bool(item.get("passed")) for item in validations):
        raise SystemExit(
            f"Structural validation failed or was incomplete: "
            f"{sum(bool(item.get('passed')) for item in validations)}/{expected_rows} passed"
        )

    summary_path = paths["results"] / f"evaluation_{EXPERIMENT_TAG}_sec{args.seconds}.csv"
    validation_path = paths["results"] / f"validation_{EXPERIMENT_TAG}_sec{args.seconds}.json"
    summary.to_csv(summary_path, index=False)
    write_json(validation_path, validations)

    for baseline in ("fixed", "max_pressure"):
        for candidate in [method for method in methods if method != baseline]:
            paired_improvements(summary, baseline, candidate).to_csv(
                paths["results"]
                / f"{candidate}_vs_{baseline}_{EXPERIMENT_TAG}_sec{args.seconds}.csv",
                index=False,
            )

    comparisons = {
        f"{candidate}_vs_{baseline}": paired_analysis(summary, baseline, candidate)
        for baseline in ("fixed", "max_pressure")
        for candidate in methods
        if candidate != baseline
    }
    analysis = {
        "experiment_tag": EXPERIMENT_TAG,
        "checkpoint": str(final_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "policy_checkpoint_episode": int(checkpoint.get("completed_episode", 0)),
        "training_completed_episode": completed,
        "deployment_policy": deployment_policy,
        "deployment_gate": checkpoint.get("deployment_gate"),
        "evaluation_seeds": args.seeds,
        "group_statistics": group_statistics(summary),
        "paired_comparisons": comparisons,
        "success_checks": {
            "dqn_shielded": success_check(summary, validations, "dqn_shielded"),
            "deployed": success_check(summary, validations, "deployed"),
        },
        "structural_validation": {
            "passed": sum(bool(item.get("passed")) for item in validations),
            "total": len(validations),
        },
    }
    analysis_path = paths["results"] / f"analysis_{EXPERIMENT_TAG}_sec{args.seconds}.json"
    report_path = paths["results"] / f"analysis_{EXPERIMENT_TAG}_sec{args.seconds}.md"
    write_json(analysis_path, analysis)
    write_markdown_report(report_path, analysis)
    write_json(
        config_path,
        evaluation_config | {"status": "complete"},
    )
    plot_method_comparison(
        summary,
        paths["plots"],
        prefix=f"{EXPERIMENT_TAG}_sec{args.seconds}_",
    )

    columns = list(PRIMARY_METRICS)
    print("\nTask 5 evaluation means")
    print(summary.groupby("method")[columns].mean().round(3).to_string())
    print(f"\nShielded DQN success: {analysis['success_checks']['dqn_shielded']['passed']}")
    print(f"Deployed controller ({deployment_policy}) success: {analysis['success_checks']['deployed']['passed']}")
    print(f"Saved analysis: {analysis_path}")


if __name__ == "__main__":
    main()
