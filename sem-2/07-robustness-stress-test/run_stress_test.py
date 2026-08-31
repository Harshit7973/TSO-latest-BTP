"""Evaluate Task 5 DQN under incidents, demand stress, and sensor faults."""

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
    require_sumo,
    set_global_seed,
)
from common.multi_intersection_tools import (  # noqa: E402
    LaneIncident,
    ObservationFault,
    PRIMARY_METRICS,
    load_task5_model,
    paired_bootstrap,
    paired_metric_table,
    run_extended_episode,
    write_json,
)

sys.path.insert(0, str(SEM2_ROOT / "05-multi-intersection"))
from dqn_core import checkpoint_path, resolve_device  # noqa: E402


METHODS = ("fixed", "dqn_raw", "dqn_shielded")
SCENARIOS: dict[str, dict[str, Any]] = {
    "nominal": {},
    "demand_surge": {"demand_scale": 1.4},
    "partial_lane_blockage": {"incident": True},
    "gaussian_noise": {"fault": "gaussian_noise"},
    "sensor_dropout": {"fault": "sensor_dropout"},
    "delayed_observation": {"fault": "delayed_observation"},
}


def diagnostic_summary(episodes_dir: Path) -> pd.DataFrame:
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
                "sensor_corruption_rate": float(frame.sensor_corruption_rate.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def incident_recovery(episodes_dir: Path, seconds: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start, end = 600.0, 1200.0
    for path in sorted(episodes_dir.glob("*__partial_lane_blockage__seed*.csv")):
        method, scenario, seed_part = path.stem.split("__")
        frame = pd.read_csv(path)
        before = frame[(frame.step >= start - 300) & (frame.step < start)]
        baseline = float(before.system_total_stopped.mean()) if not before.empty else 0.0
        threshold = max(baseline * 1.10, baseline + 1.0)
        after = frame[frame.step >= end].copy()
        rolling = after.system_total_stopped.rolling(3, min_periods=3).mean()
        recovered_rows = after.loc[rolling <= threshold]
        recovered = not recovered_rows.empty
        recovery_time = (
            float(recovered_rows.step.iloc[0] - end) if recovered else float(seconds - end)
        )
        rows.append(
            {
                "method": method,
                "scenario": scenario,
                "seed": int(seed_part.removeprefix("seed")),
                "preincident_stopped": baseline,
                "recovery_threshold": threshold,
                "recovered": recovered,
                "recovery_time_seconds": recovery_time,
                "right_censored": not recovered,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "method",
            "scenario",
            "seed",
            "preincident_stopped",
            "recovery_threshold",
            "recovered",
            "recovery_time_seconds",
            "right_censored",
        ],
    )


def scenario_plots(summary: pd.DataFrame, plots_dir: Path, seconds: int) -> None:
    labels = {
        "mean_system_total_waiting_time": "Mean total waiting time (s)",
        "mean_system_total_stopped": "Mean stopped vehicles",
        "mean_system_mean_speed": "Mean speed (m/s)",
        "throughput_veh_per_hour": "Completed trips per hour",
    }
    scenarios = [name for name in SCENARIOS if name in set(summary.scenario)]
    methods = list(METHODS)
    x = np.arange(len(scenarios))
    width = 0.25
    for metric, label in labels.items():
        fig, ax = plt.subplots(figsize=(12, 5.8))
        for index, method in enumerate(methods):
            means, cis = [], []
            for scenario in scenarios:
                values = summary[(summary.method == method) & (summary.scenario == scenario)][metric]
                mean, ci = mean_ci(values)
                means.append(mean)
                cis.append(ci)
            ax.bar(x + (index - 1) * width, means, width, yerr=cis, capsize=3, label=method)
        ax.set_xticks(x, [name.replace("_", "\n") for name in scenarios])
        ax.set_ylabel(label)
        ax.set_title(f"Task 7 robustness stress test: {label}")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / f"stress_sec{seconds}_{metric}.png", dpi=180)
        plt.close(fig)


def degradation_table(summary: pd.DataFrame) -> pd.DataFrame:
    nominal = summary[summary.scenario == "nominal"]
    rows: list[pd.DataFrame] = []
    for scenario in [
        name for name in SCENARIOS if name != "nominal" and name in set(summary.scenario)
    ]:
        stressed = summary[summary.scenario == scenario]
        for method in METHODS:
            left = nominal[nominal.method == method]
            right = stressed[stressed.method == method]
            paired = left.merge(right, on="seed", suffixes=("_nominal", "_stress"))
            output = paired[["seed"]].copy()
            output["method"] = method
            output["scenario"] = scenario
            for metric, higher_is_better in PRIMARY_METRICS.items():
                base = paired[f"{metric}_nominal"].astype(float)
                stress = paired[f"{metric}_stress"].astype(float)
                loss = base - stress if higher_is_better else stress - base
                output[f"degradation_pct__{metric}"] = np.where(
                    base != 0,
                    loss / base.abs() * 100.0,
                    np.nan,
                )
            rows.append(output)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(
        columns=[
            "seed",
            "method",
            "scenario",
            *[f"degradation_pct__{metric}" for metric in PRIMARY_METRICS],
        ]
    )


def write_report(path: Path, analysis: dict[str, Any]) -> None:
    lines = [
        "# Task 7 automatic robustness analysis",
        "",
        "| Scenario | Shielded waiting improvement vs fixed | Queue improvement | Throughput improvement | Waiting wins |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario in analysis["scenarios"]:
        metrics = analysis["scenario_comparisons"][scenario]["dqn_shielded_vs_fixed"]["metrics"]
        lines.append(
            f"| {scenario} | {metrics['mean_system_total_waiting_time']['mean_improvement_pct']:.2f}% | "
            f"{metrics['mean_system_total_stopped']['mean_improvement_pct']:.2f}% | "
            f"{metrics['throughput_veh_per_hour']['mean_improvement_pct']:.2f}% | "
            f"{metrics['mean_system_total_waiting_time']['wins']}/{metrics['mean_system_total_waiting_time']['trials']} |"
        )
    lines.extend(
        [
            "",
            f"Structural validation: **{analysis['structural_validation']['passed']}/"
            f"{analysis['structural_validation']['total']} passed**.",
            "",
            f"Robustness target passed: **{analysis['success_checks']['passed']}**.",
            "",
            "Sensor faults corrupt the learned observation. The shield still uses true simulator pressure;",
            "therefore raw DQN is the pure perception-fault result and shielded DQN is the layered-safety result.",
            "A failed scenario is a reportable robustness boundary, not permission to discard the seed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1101, 1102, 1103, 1104, 1105])
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--noise-std", type=float, default=0.15)
    parser.add_argument("--dropout-probability", type=float, default=0.20)
    parser.add_argument("--delay-decisions", type=int, default=2)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    require_sumo()
    if args.seconds < 1 or len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("Evaluation duration must be positive and seeds must be unique")
    if "nominal" not in args.scenarios:
        raise SystemExit("--scenarios must include nominal to compute controlled degradation")
    if args.seconds < 1500 and "partial_lane_blockage" in args.scenarios:
        raise SystemExit("Lane-blockage evaluation requires --seconds >= 1500")
    if not 0 <= args.dropout_probability <= 1:
        raise SystemExit("--dropout-probability must be in [0, 1]")
    if args.delay_decisions < 1:
        raise SystemExit("--delay-decisions must be positive")
    device = resolve_device(args.device)
    paths = ensure_output_tree(TASK_DIR)
    checkpoint = checkpoint_path(SEM2_ROOT / "05-multi-intersection/checkpoints", "final")
    model, payload, fingerprint = load_task5_model(checkpoint, device)
    pressure_gap = float(payload["pressure_gap"])

    episodes_dir = paths["episodes"] / f"sec{args.seconds}"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    config_path = paths["results"] / f"evaluation_config_sec{args.seconds}.json"
    config = {
        "command_line": sys.argv,
        "seconds": args.seconds,
        "seeds": args.seeds,
        "scenarios": args.scenarios,
        "methods": list(METHODS),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": fingerprint,
        "device": str(device),
        "pressure_gap": pressure_gap,
        "noise_std": args.noise_std,
        "dropout_probability": args.dropout_probability,
        "delay_decisions": args.delay_decisions,
        "incident": {"lane": "-h11_0", "start": 600, "end": 1200, "speed": 1.0},
        "demand_scale": 1.4,
    }
    if config_path.exists() and any(episodes_dir.glob("*.csv")) and not args.force:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        immutable = (
            "seconds", "seeds", "scenarios", "methods", "checkpoint_sha256",
            "noise_std", "dropout_probability", "delay_decisions",
        )
        if any(previous.get(key) != config.get(key) for key in immutable):
            raise SystemExit("Existing Task 7 episodes use a different configuration; archive them first")
    write_json(config_path, config | {"status": "in_progress"})

    scenario_offsets = {name: index * 100_000 for index, name in enumerate(SCENARIOS)}
    for scenario in args.scenarios:
        specification = SCENARIOS[scenario]
        for seed in args.seeds:
            for method in METHODS:
                output = episodes_dir / f"{method}__{scenario}__seed{seed}.csv"
                if output.exists() and not args.force:
                    print(f"[resume] keeping {output.name}")
                    continue
                set_global_seed(seed)
                fault = ObservationFault(
                    specification.get("fault", "none"),
                    seed=seed + scenario_offsets[scenario],
                    noise_std=args.noise_std,
                    dropout_probability=args.dropout_probability,
                    delay_decisions=args.delay_decisions,
                )
                incident = (
                    LaneIncident(start=600, end=1200, lane_id="-h11_0", reduced_speed=1.0)
                    if specification.get("incident")
                    else None
                )
                print(f"[run] scenario={scenario}, method={method}, seed={seed}")
                frame, _ = run_extended_episode(
                    seconds=args.seconds,
                    seed=seed,
                    controller=method,
                    pressure_gap=pressure_gap,
                    model=None if method == "fixed" else model,
                    device=device,
                    observation_fault=fault,
                    incident=incident,
                    demand_scale=float(specification.get("demand_scale", 1.0)),
                )
                frame.to_csv(output, index=False)

    summary, validations = collect_episode_summaries(episodes_dir, args.seconds)
    diagnostics = diagnostic_summary(episodes_dir)
    summary = summary.merge(diagnostics, on=["method", "scenario", "seed"], how="left")
    summary = summary[
        summary.method.isin(METHODS)
        & summary.scenario.isin(args.scenarios)
        & summary.seed.isin(set(args.seeds))
    ].copy()
    expected_stems = {
        f"{method}__{scenario}__seed{seed}"
        for scenario in args.scenarios
        for seed in args.seeds
        for method in METHODS
    }
    validations = [
        row for row in validations if Path(str(row.get("file", ""))).stem in expected_stems
    ]
    expected_rows = len(args.scenarios) * len(args.seeds) * len(METHODS)
    if len(summary) != expected_rows:
        raise SystemExit(f"Expected {expected_rows} result rows, found {len(summary)}")
    if len(validations) != expected_rows or not all(bool(row.get("passed")) for row in validations):
        raise SystemExit("Task 7 structural validation failed or is incomplete")

    summary_path = paths["results"] / f"stress_summary_sec{args.seconds}.csv"
    validation_path = paths["results"] / f"validation_sec{args.seconds}.json"
    summary.to_csv(summary_path, index=False)
    write_json(validation_path, validations)
    degradation = degradation_table(summary)
    degradation.to_csv(paths["results"] / f"degradation_vs_nominal_sec{args.seconds}.csv", index=False)
    recovery = incident_recovery(episodes_dir, args.seconds)
    recovery.to_csv(paths["results"] / f"incident_recovery_sec{args.seconds}.csv", index=False)

    scenario_comparisons: dict[str, Any] = {}
    scenario_passes = 0
    required_wins = int(np.ceil(0.8 * len(args.seeds)))
    for scenario in args.scenarios:
        selected = summary[summary.scenario == scenario]
        scenario_comparisons[scenario] = {}
        for candidate in ("dqn_raw", "dqn_shielded"):
            paired = paired_metric_table(selected, "fixed", candidate)
            paired.to_csv(
                paths["results"] / f"{candidate}_vs_fixed__{scenario}_sec{args.seconds}.csv",
                index=False,
            )
            scenario_comparisons[scenario][f"{candidate}_vs_fixed"] = paired_bootstrap(
                paired,
                resamples=args.bootstrap_resamples,
                seed=2026 + len(scenario),
            )
        shielded = scenario_comparisons[scenario]["dqn_shielded_vs_fixed"]["metrics"]
        scenario_pass = bool(
            shielded["mean_system_total_waiting_time"]["wins"] >= required_wins
            and shielded["mean_system_total_stopped"]["wins"] >= required_wins
            and shielded["throughput_veh_per_hour"]["mean_improvement_pct"] >= -5.0
        )
        scenario_passes += int(scenario_pass)

    zero_teleports = bool((summary.final_system_total_teleported == 0).all())
    target_scenarios = int(np.ceil(2 * len(args.scenarios) / 3))
    success = bool(scenario_passes >= target_scenarios and zero_teleports)
    analysis = {
        "checkpoint_sha256": fingerprint,
        "scenarios": args.scenarios,
        "scenario_comparisons": scenario_comparisons,
        "success_checks": {
            "passed": success,
            "required_scenarios": target_scenarios,
            "passing_scenarios": scenario_passes,
            "zero_teleports": zero_teleports,
            "per_scenario_rule": "waiting and stopped wins >=80% of seeds; mean throughput >=-5%",
        },
        "structural_validation": {"passed": expected_rows, "total": expected_rows},
        "limitations": [
            "Sensor corruption affects DQN observations, while the shield uses true simulator pressure.",
            "One partial blockage and one demand scale do not span all real incidents.",
            "SUMO results require real detector and traffic calibration before deployment claims.",
        ],
    }
    analysis_path = paths["results"] / f"analysis_sec{args.seconds}.json"
    report_path = paths["results"] / f"analysis_sec{args.seconds}.md"
    write_json(analysis_path, analysis)
    write_report(report_path, analysis)
    write_json(config_path, config | {"status": "complete"})
    scenario_plots(summary, paths["plots"], args.seconds)
    print(f"Task 7 complete: {expected_rows}/{expected_rows} valid episodes")
    print(f"Robustness target: {success} ({scenario_passes}/{len(args.scenarios)} scenarios)")
    print(f"Analysis: {analysis_path}")


if __name__ == "__main__":
    main()
