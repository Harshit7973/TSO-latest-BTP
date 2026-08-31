"""Run a paired post-training component ablation of the completed Task 5 DQN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_DIR = Path(__file__).resolve().parent
SEM2_ROOT = TASK_DIR.parent
sys.path.insert(0, str(SEM2_ROOT))

from common.experiment_utils import (  # noqa: E402
    collect_episode_summaries,
    ensure_output_tree,
    mean_ci,
    plot_method_comparison,
    require_sumo,
    set_global_seed,
)
from common.multi_intersection_tools import (  # noqa: E402
    PRIMARY_METRICS,
    load_task5_model,
    paired_bootstrap,
    paired_metric_table,
    run_extended_episode,
    write_json,
)

sys.path.insert(0, str(SEM2_ROOT / "05-multi-intersection"))
from dqn_core import checkpoint_path, resolve_device  # noqa: E402


METHODS: dict[str, dict[str, Any]] = {
    "fixed": {"controller": "fixed", "occluded": ()},
    "full_dqn": {"controller": "dqn_shielded", "occluded": ()},
    "raw_dqn": {"controller": "dqn_raw", "occluded": ()},
    "no_network_context": {
        "controller": "dqn_shielded",
        "occluded": ("network_context",),
    },
    "no_identity": {
        "controller": "dqn_shielded",
        "occluded": ("intersection_identity",),
    },
    "no_pressure": {
        "controller": "dqn_shielded",
        "occluded": ("pressure",),
    },
}


def add_diagnostics(summary: pd.DataFrame, episodes_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(episodes_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        method, seed_part = path.stem.split("__")
        rows.append(
            {
                "method": method,
                "seed": int(seed_part.removeprefix("seed")),
                "shield_rate": float(frame.shield_rate.iloc[-1]),
                "expert_agreement_rate": float(frame.expert_agreement_rate.iloc[-1]),
            }
        )
    return summary.merge(pd.DataFrame(rows), on=["method", "seed"], how="left")


def group_statistics(summary: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method, group in summary.groupby("method"):
        row: dict[str, Any] = {}
        for metric in PRIMARY_METRICS:
            mean, ci95 = mean_ci(group[metric])
            row[metric] = {"mean": mean, "ci95_half_width": ci95}
        row["shield_rate"] = float(group.shield_rate.fillna(0).mean())
        row["expert_agreement_rate"] = float(group.expert_agreement_rate.fillna(0).mean())
        output[str(method)] = row
    return output


def plot_degradation(summary: pd.DataFrame, output: Path) -> None:
    candidates = ["no_network_context", "no_identity", "no_pressure"]
    paired_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        paired = paired_metric_table(summary, "full_dqn", candidate)
        row: dict[str, Any] = {"ablation": candidate}
        for metric in PRIMARY_METRICS:
            # paired_metric_table is positive when candidate is better; negate
            # it so positive bars here consistently mean performance loss.
            row[metric] = -float(paired[f"improvement_pct__{metric}"].mean())
        paired_rows.append(row)
    frame = pd.DataFrame(paired_rows).set_index("ablation")
    labels = {
        "mean_system_total_waiting_time": "Waiting",
        "mean_system_total_stopped": "Stopped",
        "mean_system_mean_speed": "Speed",
        "throughput_veh_per_hour": "Throughput",
    }
    fig, ax = plt.subplots(figsize=(10, 5.5))
    frame.rename(columns=labels).plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean degradation relative to full DQN (%)")
    ax.set_xlabel("Occluded component")
    ax.set_title("Task 6 paired component sensitivity")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_report(path: Path, analysis: dict[str, Any]) -> None:
    groups = analysis["group_statistics"]
    lines = [
        "# Task 6 automatic ablation analysis",
        "",
        "This is a post-training occlusion study of one frozen Task 5 checkpoint.",
        "It measures sensitivity, not the causal effect of retraining an architecture.",
        "",
        "| Method | Waiting ↓ | Stopped ↓ | Speed ↑ | Throughput/h ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        values = groups[method]
        lines.append(
            f"| {method} | {values['mean_system_total_waiting_time']['mean']:.3f} | "
            f"{values['mean_system_total_stopped']['mean']:.3f} | "
            f"{values['mean_system_mean_speed']['mean']:.3f} | "
            f"{values['throughput_veh_per_hour']['mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Structural validation: **{analysis['structural_validation']['passed']}/"
            f"{analysis['structural_validation']['total']} passed**.",
            "",
            f"Full-DQN project success check: **{analysis['success_checks']['full_dqn_vs_fixed']}**.",
            "",
            "Interpret a degraded occlusion as evidence that the frozen model uses that information.",
            "A neutral result does not prove that the feature is unnecessary during training.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1001, 1002, 1003, 1004, 1005])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    require_sumo()
    if args.seconds < 1 or len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("Ablation duration must be positive and seeds must be unique")
    device = resolve_device(args.device)
    paths = ensure_output_tree(TASK_DIR)
    task5_checkpoints = SEM2_ROOT / "05-multi-intersection/checkpoints"
    final_checkpoint = checkpoint_path(task5_checkpoints, "final")
    model, checkpoint, fingerprint = load_task5_model(final_checkpoint, device)
    pressure_gap = float(checkpoint["pressure_gap"])

    episodes_dir = paths["episodes"] / f"sec{args.seconds}"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    config_path = paths["results"] / f"evaluation_config_sec{args.seconds}.json"
    config = {
        "command_line": sys.argv,
        "seconds": args.seconds,
        "seeds": args.seeds,
        "methods": list(METHODS),
        "device": str(device),
        "checkpoint": str(final_checkpoint.resolve()),
        "checkpoint_sha256": fingerprint,
        "pressure_gap": pressure_gap,
        "study_type": "post-training feature occlusion",
    }
    if config_path.exists() and any(episodes_dir.glob("*.csv")) and not args.force:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        immutable = ("seconds", "seeds", "methods", "checkpoint_sha256", "pressure_gap")
        if any(previous.get(key) != config.get(key) for key in immutable):
            raise SystemExit("Existing Task 6 episodes use a different configuration; archive them first")
    write_json(config_path, config | {"status": "in_progress"})

    for seed in args.seeds:
        for method, specification in METHODS.items():
            output = episodes_dir / f"{method}__seed{seed}.csv"
            if output.exists() and not args.force:
                print(f"[resume] keeping {output.name}")
                continue
            set_global_seed(seed)
            print(f"[run] method={method}, seed={seed}, device={device}")
            frame, _ = run_extended_episode(
                seconds=args.seconds,
                seed=seed,
                controller=specification["controller"],
                pressure_gap=pressure_gap,
                model=None if method == "fixed" else model,
                device=device,
                occluded_groups=specification["occluded"],
            )
            frame.to_csv(output, index=False)

    summary, validations = collect_episode_summaries(episodes_dir, args.seconds)
    summary = add_diagnostics(summary, episodes_dir)
    summary = summary[
        summary.method.isin(METHODS) & summary.seed.isin(set(args.seeds))
    ].copy()
    expected_stems = {f"{method}__seed{seed}" for method in METHODS for seed in args.seeds}
    validations = [
        row for row in validations if Path(str(row.get("file", ""))).stem in expected_stems
    ]
    expected_rows = len(METHODS) * len(args.seeds)
    if len(summary) != expected_rows:
        raise SystemExit(f"Expected {expected_rows} evaluation rows, found {len(summary)}")
    if len(validations) != expected_rows or not all(bool(row.get("passed")) for row in validations):
        raise SystemExit("Task 6 structural validation is incomplete or failed")
    if set(summary.method) != set(METHODS):
        raise SystemExit("Task 6 method set is incomplete")

    summary_path = paths["results"] / f"ablation_summary_sec{args.seconds}.csv"
    validation_path = paths["results"] / f"validation_sec{args.seconds}.json"
    summary.to_csv(summary_path, index=False)
    write_json(validation_path, validations)

    comparisons: dict[str, Any] = {}
    for candidate in [method for method in METHODS if method != "fixed"]:
        paired = paired_metric_table(summary, "fixed", candidate)
        paired.to_csv(
            paths["results"] / f"{candidate}_vs_fixed_sec{args.seconds}.csv",
            index=False,
        )
        comparisons[f"{candidate}_vs_fixed"] = paired_bootstrap(
            paired,
            resamples=args.bootstrap_resamples,
            seed=2026,
        )
    for candidate in ("raw_dqn", "no_network_context", "no_identity", "no_pressure"):
        paired = paired_metric_table(summary, "full_dqn", candidate)
        paired.to_csv(
            paths["results"] / f"{candidate}_vs_full_dqn_sec{args.seconds}.csv",
            index=False,
        )
        comparisons[f"{candidate}_vs_full_dqn"] = paired_bootstrap(
            paired,
            resamples=args.bootstrap_resamples,
            seed=2027,
        )

    full = comparisons["full_dqn_vs_fixed"]["metrics"]
    required_wins = int(np.ceil(0.8 * len(args.seeds)))
    success = bool(
        full["mean_system_total_waiting_time"]["wins"] >= required_wins
        and full["mean_system_total_stopped"]["wins"] >= required_wins
        and full["throughput_veh_per_hour"]["mean_improvement_pct"] >= -3.0
        and (summary.final_system_total_teleported == 0).all()
    )
    analysis = {
        "study_type": "post-training feature occlusion",
        "checkpoint_sha256": fingerprint,
        "seeds": args.seeds,
        "group_statistics": group_statistics(summary),
        "paired_bootstrap": comparisons,
        "success_checks": {"full_dqn_vs_fixed": success},
        "structural_validation": {"passed": expected_rows, "total": expected_rows},
        "limitations": [
            "Occlusion changes one frozen model's inputs and is not equivalent to retraining without a component.",
            "Zero-valued features may be outside some training-state combinations.",
            "The pressure shield still uses the simulator's unoccluded state.",
        ],
    }
    analysis_path = paths["results"] / f"analysis_sec{args.seconds}.json"
    report_path = paths["results"] / f"analysis_sec{args.seconds}.md"
    write_json(analysis_path, analysis)
    write_report(report_path, analysis)
    write_json(config_path, config | {"status": "complete"})
    plot_method_comparison(summary, paths["plots"], prefix=f"sec{args.seconds}_")
    plot_degradation(summary, paths["plots"] / f"component_degradation_sec{args.seconds}.png")
    print(f"Task 6 complete: {expected_rows}/{expected_rows} valid episodes")
    print(f"Full DQN vs fixed success: {success}")
    print(f"Analysis: {analysis_path}")


if __name__ == "__main__":
    main()
