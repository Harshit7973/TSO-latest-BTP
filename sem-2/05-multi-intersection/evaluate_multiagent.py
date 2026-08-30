"""Evaluate fixed timing and compact-v2 multi-agent policies on held-out seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


TASK_DIR = Path(__file__).resolve().parent
SEM2_ROOT = TASK_DIR.parent
sys.path.insert(0, str(SEM2_ROOT))
sys.path.insert(0, str(TASK_DIR))

from common.experiment_utils import (  # noqa: E402
    collect_episode_summaries,
    ensure_output_tree,
    paired_improvements,
    plot_method_comparison,
    require_sumo,
    set_global_seed,
    write_json,
)
from multiagent_policy import (  # noqa: E402
    EXPERIMENT_TAG,
    checkpoint_path,
    load_checkpoint,
    run_multiagent_episode,
)
from train_multiagent import NETWORKS  # noqa: E402


def add_policy_diagnostics(summary: pd.DataFrame, episodes_dir: Path) -> pd.DataFrame:
    diagnostics = []
    for path in sorted(episodes_dir.glob("*.csv")):
        parts = path.stem.split("__")
        method = parts[0]
        seed = int(next(part[4:] for part in parts if part.startswith("seed")))
        frame = pd.read_csv(path)
        diagnostics.append(
            {
                "method": method,
                "seed": seed,
                "pressure_fallback_actions": int(frame.pressure_fallback_actions.iloc[-1]),
                "pressure_override_actions": int(frame.pressure_override_actions.iloc[-1]),
                "total_agent_decisions": int(frame.total_agent_decisions.iloc[-1]),
                "fallback_rate": float(frame.fallback_rate.iloc[-1]),
                "override_rate": float(frame.override_rate.iloc[-1]),
            }
        )
    return summary.merge(pd.DataFrame(diagnostics), on=["method", "seed"], how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", choices=NETWORKS, default="2x2")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["independent", "cooperative"],
        default=["independent", "cooperative"],
    )
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--seeds", nargs="+", type=int, default=[701, 702, 703, 704, 705])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Permit non-final checkpoint evaluation")
    args = parser.parse_args()
    require_sumo()
    paths = ensure_output_tree(TASK_DIR)
    episodes_dir = paths["episodes"] / f"{args.network}_{EXPERIMENT_TAG}_sec{args.seconds}"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    controllers = {}
    controller_metadata = {}
    for mode in args.modes:
        final_path = checkpoint_path(paths["checkpoints"], mode, args.network, "final")
        latest_path = checkpoint_path(paths["checkpoints"], mode, args.network, "latest")
        best_path = checkpoint_path(paths["checkpoints"], mode, args.network, "best")
        final_checkpoint = load_checkpoint(final_path)
        latest_checkpoint = load_checkpoint(latest_path)

        # A short smoke run can leave a valid final file behind. If a different
        # full run is in progress, select latest so stale smoke-test weights are
        # never silently evaluated as the completed experiment.
        selected_path: Path | None = None
        checkpoint = None
        if final_checkpoint is not None:
            final_signature = (
                int(final_checkpoint.get("seconds", -1)),
                int(final_checkpoint.get("target_episodes", -1)),
            )
            latest_signature = None
            if latest_checkpoint is not None:
                latest_signature = (
                    int(latest_checkpoint.get("seconds", -1)),
                    int(latest_checkpoint.get("target_episodes", -1)),
                )
            if latest_signature is None or latest_signature == final_signature:
                selected_path, checkpoint = final_path, final_checkpoint
        if selected_path is None and latest_checkpoint is not None:
            selected_path, checkpoint = latest_path, latest_checkpoint
        if selected_path is None:
            selected_path, checkpoint = best_path, load_checkpoint(best_path)
        if checkpoint is None:
            print(f"[skip] missing compact-v2 checkpoint for {mode}; train it first")
            continue
        if checkpoint.get("experiment_tag") != EXPERIMENT_TAG:
            raise SystemExit(f"Refusing incompatible checkpoint: {selected_path}")
        if int(checkpoint.get("seconds", -1)) != args.seconds:
            raise SystemExit(
                f"{selected_path.name} was trained with --seconds {checkpoint.get('seconds')}, "
                f"but evaluation requested --seconds {args.seconds}. Use matching settings."
            )
        completed = int(checkpoint.get("training_completed_episode", checkpoint.get("completed_episode", 0)))
        target = int(checkpoint.get("target_episodes", completed))
        if completed < target and not args.allow_partial:
            raise SystemExit(
                f"{mode} compact-v2 training is incomplete ({completed}/{target}). Resume or pass --allow-partial."
            )
        controllers[mode] = checkpoint
        controller_metadata[mode] = {
            "checkpoint": str(selected_path.resolve()),
            "policy_checkpoint_episode": int(
                checkpoint.get("policy_checkpoint_episode", checkpoint.get("completed_episode", 0))
            ),
            "training_completed_episode": completed,
            "target_episodes": target,
            "best_validation_score": checkpoint.get("best_validation_score"),
            "q_states": sum(len(table) for table in checkpoint["tables"].values()),
        }

    methods = ["fixed", "max_pressure", *controllers.keys()]
    for seed in args.seeds:
        for method in methods:
            output = episodes_dir / f"{method}__seed{seed}.csv"
            if output.exists() and not args.force:
                print(f"[resume] keeping {output.name}")
                continue
            set_global_seed(seed)
            print(f"[run] method={method}, seed={seed}, network={args.network}, experiment={EXPERIMENT_TAG}")
            frame = run_multiagent_episode(
                network_config=NETWORKS[args.network],
                seconds=args.seconds,
                seed=seed,
                fixed=method == "fixed",
                tables=(
                    None
                    if method == "fixed"
                    else {} if method == "max_pressure" else controllers[method]["tables"]
                ),
                reward_fn="queue",
            )
            frame.to_csv(output, index=False)

    summary, validations = collect_episode_summaries(episodes_dir, args.seconds)
    summary = add_policy_diagnostics(summary, episodes_dir)
    summary_path = paths["results"] / f"evaluation_{args.network}_{EXPERIMENT_TAG}_sec{args.seconds}.csv"
    summary.to_csv(summary_path, index=False)
    write_json(
        paths["results"] / f"validation_{args.network}_{EXPERIMENT_TAG}_sec{args.seconds}.json",
        validations,
    )
    write_json(
        paths["results"] / f"evaluation_config_{args.network}_{EXPERIMENT_TAG}.json",
        {
            "network": args.network,
            "seconds": args.seconds,
            "seeds": args.seeds,
            "experiment_tag": EXPERIMENT_TAG,
            "controllers": controller_metadata,
        },
    )

    paired_improvements(summary, "fixed", "max_pressure").to_csv(
        paths["results"] / f"max_pressure_vs_fixed_{args.network}_{EXPERIMENT_TAG}_sec{args.seconds}.csv",
        index=False,
    )
    for method in controllers:
        paired_improvements(summary, "fixed", method).to_csv(
            paths["results"] / f"{method}_vs_fixed_{args.network}_{EXPERIMENT_TAG}_sec{args.seconds}.csv",
            index=False,
        )
        paired_improvements(summary, "max_pressure", method).to_csv(
            paths["results"]
            / f"{method}_vs_max_pressure_{args.network}_{EXPERIMENT_TAG}_sec{args.seconds}.csv",
            index=False,
        )
    if "independent" in controllers and "cooperative" in controllers:
        paired_improvements(summary, "independent", "cooperative").to_csv(
            paths["results"]
            / f"cooperative_vs_independent_{args.network}_{EXPERIMENT_TAG}_sec{args.seconds}.csv",
            index=False,
        )
    plot_method_comparison(
        summary,
        paths["plots"],
        prefix=f"{args.network}_{EXPERIMENT_TAG}_sec{args.seconds}_",
    )

    print("\nCompact-v2 evaluation summary")
    columns = [
        "mean_system_total_waiting_time",
        "mean_system_total_stopped",
        "mean_system_mean_speed",
        "throughput_veh_per_hour",
        "fallback_rate",
        "override_rate",
    ]
    print(summary.groupby("method")[columns].mean().round(3).to_string())
    for method in controllers:
        fallback_rate = float(summary.loc[summary.method == method, "fallback_rate"].mean())
        if fallback_rate > 0.20:
            print(f"WARNING: {method} fallback rate is {fallback_rate:.1%}; consider additional training.")
    print(f"Saved evaluation: {summary_path}")


if __name__ == "__main__":
    main()
