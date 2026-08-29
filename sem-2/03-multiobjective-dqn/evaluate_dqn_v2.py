"""Compare fixed timing, DQN-v1 and enhanced DQN-v2 on dynamic scenarios."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
SEM2_ROOT = TASK_DIR.parent
sys.path.insert(0, str(SEM2_ROOT))
sys.path.insert(0, str(TASK_DIR))

from common.experiment_utils import (  # noqa: E402
    NETWORKS,
    REPO_ROOT,
    collect_episode_summaries,
    ensure_output_tree,
    paired_improvements,
    plot_method_comparison,
    read_json,
    require_sumo,
    run_single_agent_episode,
    write_json,
)
from features import EnhancedObservation, multi_objective_reward  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-model", type=Path, default=REPO_ROOT / "btp/models/dqn_2way.zip")
    parser.add_argument("--v2-model", type=Path, default=TASK_DIR / "checkpoints/dqn_v2_latest.zip")
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[301, 302, 303, 304, 305])
    parser.add_argument("--scenarios", nargs="+", default=["direction_switch", "burst", "unseen_mixed"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Permit evaluation before training target is reached")
    args = parser.parse_args()
    require_sumo()
    from stable_baselines3 import DQN

    if not args.v2_model.exists():
        raise SystemExit("DQN-v2 model is missing. Run train_dqn_v2.py first.")
    paths = ensure_output_tree(TASK_DIR)
    state = read_json(paths["checkpoints"] / "training_state.json", {})
    if state and int(state.get("completed_timesteps", 0)) < int(state.get("target_timesteps", 0)) and not args.allow_partial:
        raise SystemExit("DQN-v2 training target is incomplete. Resume training or pass --allow-partial for a non-final check.")
    episodes_dir = paths["episodes"] / f"sec{args.seconds}"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    route_dir = TASK_DIR / "generated-routes" / f"sec{args.seconds}"
    subprocess.run(
        [sys.executable, str(SEM2_ROOT / "02-dynamic-traffic/generate_routes.py"), "--seconds", str(args.seconds), "--output-dir", str(route_dir)],
        check=True,
    )
    models = {"dqn_v1": DQN.load(str(args.v1_model)), "dqn_v2": DQN.load(str(args.v2_model))}
    net_file = NETWORKS["2way"]["net_file"]
    write_json(paths["results"] / "evaluation_config.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})

    for scenario in args.scenarios:
        route_file = route_dir / f"{scenario}.rou.xml"
        for seed in args.seeds:
            for method in ("fixed", "dqn_v1", "dqn_v2"):
                output = episodes_dir / f"{method}__{scenario}__seed{seed}.csv"
                if output.exists() and not args.force:
                    print(f"[resume] keeping {output.name}")
                    continue
                is_v2 = method == "dqn_v2"
                run_single_agent_episode(
                    output_csv=output,
                    method=method,
                    seed=seed,
                    seconds=args.seconds,
                    net_file=net_file,
                    route_file=route_file,
                    model=models.get(method),
                    fixed=method == "fixed",
                    observation_class=EnhancedObservation if is_v2 else None,
                    reward_fn=multi_objective_reward if is_v2 else "diff-waiting-time",
                )

    summary, validations = collect_episode_summaries(episodes_dir, args.seconds)
    summary.to_csv(paths["results"] / f"comparison_summary_sec{args.seconds}.csv", index=False)
    write_json(paths["results"] / f"validation_sec{args.seconds}.json", validations)
    paired_improvements(summary, "fixed", "dqn_v1").to_csv(paths["results"] / f"improvement_v1_vs_fixed_sec{args.seconds}.csv", index=False)
    paired_improvements(summary, "fixed", "dqn_v2").to_csv(paths["results"] / f"improvement_v2_vs_fixed_sec{args.seconds}.csv", index=False)
    paired_improvements(summary, "dqn_v1", "dqn_v2").to_csv(paths["results"] / f"improvement_v2_vs_v1_sec{args.seconds}.csv", index=False)
    plot_method_comparison(summary, paths["plots"], prefix=f"v1_v2_sec{args.seconds}_")
    print("Evaluation complete. Use only rows whose validation entries passed.")


if __name__ == "__main__":
    main()
