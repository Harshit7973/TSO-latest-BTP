"""Evaluate the Semester 1 DQN on unseen and time-varying traffic demand."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASK_DIR = Path(__file__).resolve().parent
SEM2_ROOT = TASK_DIR.parent
sys.path.insert(0, str(SEM2_ROOT))

from common.experiment_utils import (  # noqa: E402
    NETWORKS,
    REPO_ROOT,
    collect_episode_summaries,
    ensure_output_tree,
    paired_improvements,
    require_sumo,
    run_single_agent_episode,
    write_json,
)


SCENARIOS = ("balanced", "ns_peak", "ew_peak", "direction_switch", "burst", "unseen_mixed")


def plot_scenarios(summary, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "mean_system_total_waiting_time": "Mean total waiting time (s)",
        "mean_system_total_stopped": "Mean stopped vehicles",
        "mean_system_mean_speed": "Mean speed (m/s)",
        "throughput_veh_per_hour": "Completed trips per hour",
    }
    methods = [method for method in ("fixed", "dqn_v1") if method in set(summary.method)]
    scenarios = [scenario for scenario in SCENARIOS if scenario in set(summary.scenario)]
    x = np.arange(len(scenarios))
    width = 0.35
    for metric, label in specs.items():
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for index, method in enumerate(methods):
            values = [summary[(summary.method == method) & (summary.scenario == scenario)][metric].mean() for scenario in scenarios]
            errors = [summary[(summary.method == method) & (summary.scenario == scenario)][metric].std(ddof=1) for scenario in scenarios]
            errors = np.nan_to_num(errors)
            ax.bar(x + (index - 0.5) * width, values, width, yerr=errors, capsize=4, label=method)
        ax.set_xticks(x, scenarios, rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"Generalisation by traffic scenario: {label}")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / f"scenario_{metric}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "btp/models/dqn_2way.zip")
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[201, 202, 203])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    require_sumo()
    from stable_baselines3 import DQN

    paths = ensure_output_tree(TASK_DIR)
    episodes_dir = paths["episodes"] / f"sec{args.seconds}"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    route_dir = TASK_DIR / "generated-routes" / f"sec{args.seconds}"
    subprocess.run(
        [sys.executable, str(TASK_DIR / "generate_routes.py"), "--seconds", str(args.seconds), "--output-dir", str(route_dir)],
        check=True,
    )
    model = DQN.load(str(args.model), device="auto")
    net_file = NETWORKS["2way"]["net_file"]
    write_json(paths["results"] / "run_config.json", vars(args) | {"model": str(args.model.resolve())})

    for scenario in args.scenarios:
        route_file = route_dir / f"{scenario}.rou.xml"
        for seed in args.seeds:
            for method, fixed in (("fixed", True), ("dqn_v1", False)):
                output = episodes_dir / f"{method}__{scenario}__seed{seed}.csv"
                if output.exists() and not args.force:
                    print(f"[resume] keeping {output.name}")
                    continue
                print(f"[run] scenario={scenario} method={method} seed={seed}")
                run_single_agent_episode(
                    output_csv=output,
                    method=method,
                    seed=seed,
                    seconds=args.seconds,
                    net_file=net_file,
                    route_file=route_file,
                    model=None if fixed else model,
                    fixed=fixed,
                    use_gui=args.gui,
                )

    summary, validations = collect_episode_summaries(episodes_dir, args.seconds)
    summary.to_csv(paths["results"] / f"summary_by_scenario_seed_sec{args.seconds}.csv", index=False)
    write_json(paths["results"] / f"validation_sec{args.seconds}.json", validations)
    paired = paired_improvements(summary, baseline="fixed", candidate="dqn_v1")
    paired.to_csv(paths["results"] / f"paired_improvements_sec{args.seconds}.csv", index=False)
    plot_scenarios(summary, paths["plots"] / f"sec{args.seconds}")
    valid = sum(bool(item["passed"]) for item in validations)
    expected = len(args.scenarios) * len(args.seeds) * 2
    print(f"Valid episodes: {valid}/{expected}")
    if valid != expected:
        print("WARNING: results are incomplete; inspect validation.json and rerun.")


if __name__ == "__main__":
    main()
