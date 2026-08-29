"""Evaluate fixed, independent and cooperative control on a multi-signal grid."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TASK_DIR = Path(__file__).resolve().parent
SEM2_ROOT = TASK_DIR.parent
sys.path.insert(0, str(SEM2_ROOT))

from common.experiment_utils import (  # noqa: E402
    collect_episode_summaries,
    ensure_output_tree,
    paired_improvements,
    plot_method_comparison,
    require_sumo,
    set_global_seed,
    write_json,
)
from train_multiagent import NETWORKS  # noqa: E402


def run_episode(method: str, checkpoint: dict | None, network: str, seconds: int, seed: int, output: Path) -> None:
    from sumo_rl import SumoEnvironment

    set_global_seed(seed)
    cfg = NETWORKS[network]
    fixed = method == "fixed"
    env = SumoEnvironment(
        net_file=str(cfg["net"]),
        route_file=str(cfg["route"]),
        use_gui=False,
        num_seconds=seconds,
        single_agent=False,
        fixed_ts=fixed,
        sumo_seed=seed,
        enforce_max_green=True,
        max_green=60,
        out_csv_name=None,
    )
    records = []
    unseen = 0
    try:
        observations = env.reset(seed=seed)
        done = {"__all__": False}
        while not done["__all__"]:
            if fixed:
                actions = {}
            else:
                actions = {}
                for agent, observation in observations.items():
                    state = env.encode(observation, agent)
                    values = checkpoint["tables"].get(agent, {}).get(state)
                    if values is None:
                        unseen += 1
                        actions[agent] = int(state[0])  # hold the current phase for an unseen state
                    else:
                        actions[agent] = int(np.argmax(values))
            observations, rewards, done, info = env.step(actions)
            records.append({"step": float(env.sim_step), "reward": float(np.mean(list(rewards.values()))) if rewards else 0.0, "unseen_states": unseen, **info})
    finally:
        env.close()
    pd.DataFrame(records).to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", choices=NETWORKS, default="2x2")
    parser.add_argument("--modes", nargs="+", choices=["independent", "cooperative"], default=["independent", "cooperative"])
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--seeds", nargs="+", type=int, default=[601, 602, 603, 604, 605])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Evaluate checkpoints before their episode target")
    args = parser.parse_args()
    require_sumo()
    paths = ensure_output_tree(TASK_DIR)
    episodes_dir = paths["episodes"] / f"{args.network}_sec{args.seconds}"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    controllers = {}
    for mode in args.modes:
        path = paths["checkpoints"] / f"{mode}_{args.network}_latest.pkl"
        if not path.exists():
            print(f"[skip] missing {path}; train {mode} mode first")
            continue
        with path.open("rb") as handle:
            controllers[mode] = pickle.load(handle)
        completed = int(controllers[mode].get("completed_episode", 0))
        target = int(controllers[mode].get("target_episodes", completed))
        if completed < target and not args.allow_partial:
            raise SystemExit(
                f"{mode} checkpoint is incomplete ({completed}/{target}). Resume training or pass --allow-partial."
            )
    methods = ["fixed", *controllers.keys()]
    for seed in args.seeds:
        for method in methods:
            output = episodes_dir / f"{method}__seed{seed}.csv"
            if output.exists() and not args.force:
                print(f"[resume] keeping {output.name}")
                continue
            print(f"[run] {method}, seed={seed}, network={args.network}")
            run_episode(method, controllers.get(method), args.network, args.seconds, seed, output)

    summary, validations = collect_episode_summaries(episodes_dir, args.seconds)
    summary.to_csv(paths["results"] / f"evaluation_{args.network}_sec{args.seconds}.csv", index=False)
    write_json(paths["results"] / f"validation_{args.network}_sec{args.seconds}.json", validations)
    for method in controllers:
        paired_improvements(summary, "fixed", method).to_csv(
            paths["results"] / f"{method}_vs_fixed_{args.network}_sec{args.seconds}.csv", index=False
        )
    if "independent" in controllers and "cooperative" in controllers:
        paired_improvements(summary, "independent", "cooperative").to_csv(
            paths["results"] / f"cooperative_vs_independent_{args.network}_sec{args.seconds}.csv", index=False
        )
    plot_method_comparison(summary, paths["plots"], prefix=f"{args.network}_sec{args.seconds}_")
    print("Multi-intersection evaluation complete.")


if __name__ == "__main__":
    main()
