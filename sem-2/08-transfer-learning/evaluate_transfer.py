"""Evaluate zero-shot, fine-tuned, and scratch DQN on shifted traffic domains."""

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
sys.path.insert(0, str(TASK_DIR))

from common.experiment_utils import (  # noqa: E402
    collect_episode_summaries,
    ensure_output_tree,
    mean_ci,
    require_sumo,
    set_global_seed,
)
from common.multi_intersection_tools import (  # noqa: E402
    PRIMARY_METRICS,
    load_task5_model,
    paired_bootstrap,
    paired_metric_table,
    run_extended_episode,
    sha256,
    write_json,
)
from dqn_core import DuelingQNetwork, resolve_device  # noqa: E402
from generate_target_routes import DOMAINS, generate  # noqa: E402


METHODS = ("fixed", "zero_shot", "fine_tuned", "scratch")
SCENARIOS = tuple(DOMAINS)


def load_selected(path: Path, device: torch.device) -> tuple[DuelingQNetwork, dict[str, Any], str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing selected Task 8 checkpoint: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    model = DuelingQNetwork().to(device)
    model.load_state_dict(payload["online_state_dict"])
    model.eval()
    return model, payload, sha256(path)


def add_diagnostics(summary: pd.DataFrame, episodes_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(episodes_dir.glob("*.csv")):
        method, scenario, seed_part = path.stem.split("__")
        frame = pd.read_csv(path)
        rows.append(
            {
                "method": method,
                "scenario": scenario,
                "seed": int(seed_part.removeprefix("seed")),
                "shield_rate": float(frame.shield_rate.iloc[-1]),
                "expert_agreement_rate": float(frame.expert_agreement_rate.iloc[-1]),
            }
        )
    return summary.merge(pd.DataFrame(rows), on=["method", "scenario", "seed"], how="left")


def plots(summary: pd.DataFrame, output_dir: Path, seconds: int) -> None:
    labels = {
        "mean_system_total_waiting_time": "Mean total waiting time (s)",
        "mean_system_total_stopped": "Mean stopped vehicles",
        "mean_system_mean_speed": "Mean speed (m/s)",
        "throughput_veh_per_hour": "Completed trips per hour",
    }
    x = np.arange(len(SCENARIOS))
    width = 0.20
    for metric, label in labels.items():
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for index, method in enumerate(METHODS):
            means, cis = [], []
            for scenario in SCENARIOS:
                values = summary[(summary.method == method) & (summary.scenario == scenario)][metric]
                mean, ci = mean_ci(values)
                means.append(mean)
                cis.append(ci)
            ax.bar(x + (index - 1.5) * width, means, width, yerr=cis, capsize=3, label=method)
        ax.set_xticks(x, [name.replace("_", "\n") for name in SCENARIOS])
        ax.set_ylabel(label)
        ax.set_title(f"Task 8 transfer evaluation: {label}")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"transfer_sec{seconds}_{metric}.png", dpi=180)
        plt.close(fig)


def group_statistics(summary: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for (scenario, method), group in summary.groupby(["scenario", "method"]):
        row: dict[str, Any] = {}
        for metric in PRIMARY_METRICS:
            mean, ci = mean_ci(group[metric])
            row[metric] = {"mean": mean, "ci95_half_width": ci}
        row["score"] = float(
            group.mean_system_total_waiting_time.mean()
            + 10.0 * group.mean_system_total_stopped.mean()
        )
        row["shield_rate"] = float(group.shield_rate.fillna(0).mean())
        output[f"{scenario}::{method}"] = row
    return output


def write_report(path: Path, analysis: dict[str, Any]) -> None:
    lines = [
        "# Task 8 automatic transfer-learning analysis",
        "",
        "| Domain | Method | Waiting ↓ | Stopped ↓ | Speed ↑ | Throughput/h ↑ | Score ↓ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for method in METHODS:
            row = analysis["group_statistics"][f"{scenario}::{method}"]
            lines.append(
                f"| {scenario} | {method} | "
                f"{row['mean_system_total_waiting_time']['mean']:.3f} | "
                f"{row['mean_system_total_stopped']['mean']:.3f} | "
                f"{row['mean_system_mean_speed']['mean']:.3f} | "
                f"{row['throughput_veh_per_hour']['mean']:.3f} | {row['score']:.3f} |"
            )
    checks = analysis["success_checks"]
    lines.extend(
        [
            "",
            f"Structural completion: **{checks['structural_complete']}**.",
            f"Fine-tuned target score better than zero-shot: **{checks['fine_tuned_beats_zero_shot_target']}**.",
            f"Fine-tuned target score better than scratch: **{checks['fine_tuned_beats_scratch_target']}**.",
            f"Fine-tuned validation AUC better than scratch: **{checks['fine_tuned_better_validation_auc']}**.",
            "",
            "A false performance check is an honest negative transfer result, not a failed pipeline.",
            "The reverse-vertical domain measures retained generalisation after horizontal fine-tuning.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1301, 1302, 1303, 1304, 1305])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument(
        "--run-name",
        default="final",
        help="Evaluate the final run or an isolated run such as 'quick'",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    require_sumo()
    if args.seconds < 1 or len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("Evaluation duration must be positive and seeds must be unique")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        raise SystemExit("--run-name may contain only letters, digits, '_' and '-'")
    device = resolve_device(args.device)
    experiment_root = TASK_DIR if args.run_name == "final" else TASK_DIR / "runs" / args.run_name
    paths = ensure_output_tree(experiment_root)
    training_config_path = paths["results"] / "training_config.json"
    if not training_config_path.exists():
        raise SystemExit(f"Missing Task 8 training manifest: {training_config_path}")
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    if training_config.get("status") != "complete":
        raise SystemExit(f"Task 8 {args.run_name!r} training manifest is not complete")
    training_seeds = {
        int(training_config["seed"]) + episode
        for episode in range(1, int(training_config["episodes"]) + 1)
    }
    reserved_seeds = training_seeds | set(map(int, training_config["validation_seeds"]))
    if reserved_seeds & set(args.seeds):
        raise SystemExit("Final evaluation seeds overlap Task 8 training or validation seeds")
    if int(training_config["seconds"]) != args.seconds:
        raise SystemExit("Evaluation horizon must match the selected Task 8 training run")
    manifest = generate(TASK_DIR / "routes")
    source_path = SEM2_ROOT / "05-multi-intersection/checkpoints/shared_dueling_ddqn_v1_final.pt"
    source, source_payload, source_sha = load_task5_model(source_path, device)
    pressure_gap = float(source_payload["pressure_gap"])
    fine, fine_payload, fine_sha = load_selected(paths["checkpoints"] / "fine_tuned_best.pt", device)
    scratch, scratch_payload, scratch_sha = load_selected(paths["checkpoints"] / "scratch_best.pt", device)
    for name, payload in (("fine_tuned", fine_payload), ("scratch", scratch_payload)):
        latest_path = paths["checkpoints"] / f"{name}_latest.pt"
        if not latest_path.exists():
            raise SystemExit(f"Missing resumable completion checkpoint for {name}")
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        if not bool(latest.get("training_complete")):
            raise SystemExit(f"{name} training is incomplete")
        if payload.get("source_checkpoint_sha256") != source_sha:
            raise SystemExit(f"{name} was trained from a different source experiment")
        expected_target_hash = manifest["domains"]["target_horizontal"]["sha256"]  # type: ignore[index]
        if payload.get("target_route_sha256") != expected_target_hash:
            raise SystemExit(f"{name} was trained on a different target route")
        if not np.isclose(float(payload.get("pressure_gap", np.nan)), pressure_gap):
            raise SystemExit(f"{name} uses an incompatible pressure shield")

    models = {"zero_shot": source, "fine_tuned": fine, "scratch": scratch}
    route_hashes = {
        scenario: manifest["domains"][scenario]["sha256"]  # type: ignore[index]
        for scenario in SCENARIOS
    }
    episodes_dir = paths["episodes"] / f"sec{args.seconds}"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    config_path = paths["results"] / f"evaluation_config_sec{args.seconds}.json"
    config = {
        "run_name": args.run_name,
        "command_line": sys.argv,
        "seconds": args.seconds,
        "seeds": args.seeds,
        "methods": list(METHODS),
        "scenarios": list(SCENARIOS),
        "device": str(device),
        "source_sha256": source_sha,
        "fine_tuned_checkpoint_sha256": fine_sha,
        "scratch_checkpoint_sha256": scratch_sha,
        "route_sha256": route_hashes,
        "selected_episodes": {
            "fine_tuned": fine_payload.get("best_episode"),
            "scratch": scratch_payload.get("best_episode"),
        },
    }
    if config_path.exists() and any(episodes_dir.glob("*.csv")) and not args.force:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        immutable = (
            "seconds", "seeds", "methods", "scenarios", "source_sha256",
            "fine_tuned_checkpoint_sha256", "scratch_checkpoint_sha256", "route_sha256",
        )
        if any(previous.get(key) != config.get(key) for key in immutable):
            raise SystemExit("Existing Task 8 evaluation uses different checkpoints or routes")
    write_json(config_path, config | {"status": "in_progress"})

    for scenario in SCENARIOS:
        route = TASK_DIR / "routes" / f"{scenario}.rou.xml"
        for seed in args.seeds:
            for method in METHODS:
                output = episodes_dir / f"{method}__{scenario}__seed{seed}.csv"
                if output.exists() and not args.force:
                    print(f"[resume] keeping {output.name}")
                    continue
                set_global_seed(seed)
                print(f"[run] domain={scenario}, method={method}, seed={seed}")
                frame, _ = run_extended_episode(
                    seconds=args.seconds,
                    seed=seed,
                    controller="fixed" if method == "fixed" else "dqn_shielded",
                    pressure_gap=pressure_gap,
                    model=models.get(method),
                    device=device,
                    route_file=route,
                )
                frame.to_csv(output, index=False)

    summary, validations = collect_episode_summaries(episodes_dir, args.seconds)
    summary = add_diagnostics(summary, episodes_dir)
    summary = summary[
        summary.method.isin(METHODS)
        & summary.scenario.isin(SCENARIOS)
        & summary.seed.isin(set(args.seeds))
    ].copy()
    expected_stems = {
        f"{method}__{scenario}__seed{seed}"
        for scenario in SCENARIOS for method in METHODS for seed in args.seeds
    }
    validations = [
        row for row in validations if Path(str(row.get("file", ""))).stem in expected_stems
    ]
    expected_rows = len(SCENARIOS) * len(METHODS) * len(args.seeds)
    structural = bool(
        len(summary) == expected_rows
        and len(validations) == expected_rows
        and all(bool(row.get("passed")) for row in validations)
    )
    if not structural:
        raise SystemExit(f"Task 8 structural validation failed; expected {expected_rows} complete rows")

    summary.to_csv(paths["results"] / f"evaluation_summary_sec{args.seconds}.csv", index=False)
    write_json(paths["results"] / f"validation_sec{args.seconds}.json", validations)
    comparisons: dict[str, Any] = {}
    for scenario in SCENARIOS:
        selected = summary[summary.scenario == scenario]
        for baseline, candidate in (
            ("fixed", "zero_shot"),
            ("fixed", "fine_tuned"),
            ("fixed", "scratch"),
            ("zero_shot", "fine_tuned"),
            ("scratch", "fine_tuned"),
        ):
            paired = paired_metric_table(selected, baseline, candidate)
            name = f"{scenario}__{candidate}_vs_{baseline}"
            paired.to_csv(paths["results"] / f"{name}_sec{args.seconds}.csv", index=False)
            comparisons[name] = paired_bootstrap(
                paired,
                resamples=args.bootstrap_resamples,
                seed=2026 + len(name),
            )

    statistics = group_statistics(summary)
    target = "target_horizontal"
    target_scores = {
        method: statistics[f"{target}::{method}"]["score"] for method in METHODS
    }
    training_analysis = json.loads((paths["results"] / "training_analysis.json").read_text(encoding="utf-8"))
    auc = training_analysis["validation_score_auc_lower_is_better"]
    zero_teleports = bool((summary.final_system_total_teleported == 0).all())
    checks = {
        "structural_complete": structural and zero_teleports,
        "fine_tuned_beats_zero_shot_target": target_scores["fine_tuned"] < target_scores["zero_shot"],
        "fine_tuned_beats_scratch_target": target_scores["fine_tuned"] < target_scores["scratch"],
        "fine_tuned_better_validation_auc": float(auc["fine_tuned"]) < float(auc["scratch"]),
        "zero_teleports": zero_teleports,
    }
    analysis = {
        "run_name": args.run_name,
        "checkpoint_sha256": {
            "source": source_sha,
            "fine_tuned": fine_sha,
            "scratch": scratch_sha,
        },
        "selected_episodes": config["selected_episodes"],
        "group_statistics": statistics,
        "paired_bootstrap": comparisons,
        "validation_score_auc_lower_is_better": auc,
        "success_checks": checks,
        "structural_validation": {"passed": expected_rows, "total": expected_rows},
        "limitations": training_analysis["limitations"],
    }
    analysis_path = paths["results"] / f"analysis_sec{args.seconds}.json"
    report_path = paths["results"] / f"analysis_sec{args.seconds}.md"
    write_json(analysis_path, analysis)
    write_report(report_path, analysis)
    write_json(config_path, config | {"status": "complete"})
    plots(summary, paths["plots"], args.seconds)
    print(f"Task 8 evaluation complete: {expected_rows}/{expected_rows} valid episodes")
    print(f"Analysis: {analysis_path}")


if __name__ == "__main__":
    main()
