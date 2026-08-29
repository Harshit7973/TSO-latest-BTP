"""Evaluate PPO, DQN-v1 and fixed timing on identical dynamic episodes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
SEM2_ROOT = TASK_DIR.parent
sys.path.insert(0, str(SEM2_ROOT))

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-model", type=Path, default=TASK_DIR / "checkpoints/ppo_latest.zip")
    parser.add_argument("--dqn-model", type=Path, default=REPO_ROOT / "btp/models/dqn_2way.zip")
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--seeds", nargs="+", type=int, default=[401, 402, 403, 404, 405])
    parser.add_argument("--scenarios", nargs="+", default=["direction_switch", "burst", "unseen_mixed"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Permit evaluation before training target is reached")
    args = parser.parse_args()
    require_sumo()
    from stable_baselines3 import DQN, PPO

    if not args.ppo_model.exists():
        raise SystemExit("PPO model missing. Run train_ppo.py first.")
    paths = ensure_output_tree(TASK_DIR)
    state = read_json(paths["checkpoints"] / "training_state.json", {})
    if state and int(state.get("completed_timesteps", 0)) < int(state.get("target_timesteps", 0)) and not args.allow_partial:
        raise SystemExit("PPO training target is incomplete. Resume training or pass --allow-partial for a non-final check.")
    episodes_dir = paths["episodes"] / f"sec{args.seconds}"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    route_dir = TASK_DIR / "generated-routes" / f"sec{args.seconds}"
    subprocess.run(
        [sys.executable, str(SEM2_ROOT / "02-dynamic-traffic/generate_routes.py"), "--seconds", str(args.seconds), "--output-dir", str(route_dir)],
        check=True,
    )
    models = {"dqn_v1": DQN.load(str(args.dqn_model)), "ppo": PPO.load(str(args.ppo_model))}
    net_file = NETWORKS["2way"]["net_file"]
    write_json(paths["results"] / "evaluation_config.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})

    for scenario in args.scenarios:
        route = route_dir / f"{scenario}.rou.xml"
        for seed in args.seeds:
            for method in ("fixed", "dqn_v1", "ppo"):
                output = episodes_dir / f"{method}__{scenario}__seed{seed}.csv"
                if output.exists() and not args.force:
                    print(f"[resume] keeping {output.name}")
                    continue
                run_single_agent_episode(
                    output_csv=output,
                    method=method,
                    seed=seed,
                    seconds=args.seconds,
                    net_file=net_file,
                    route_file=route,
                    model=models.get(method),
                    fixed=method == "fixed",
                )

    summary, validations = collect_episode_summaries(episodes_dir, args.seconds)
    summary.to_csv(paths["results"] / f"comparison_summary_sec{args.seconds}.csv", index=False)
    write_json(paths["results"] / f"validation_sec{args.seconds}.json", validations)
    paired_improvements(summary, "fixed", "ppo").to_csv(paths["results"] / f"ppo_vs_fixed_sec{args.seconds}.csv", index=False)
    paired_improvements(summary, "dqn_v1", "ppo").to_csv(paths["results"] / f"ppo_vs_dqn_sec{args.seconds}.csv", index=False)
    plot_method_comparison(summary, paths["plots"], prefix=f"ppo_comparison_sec{args.seconds}_")
    print("PPO evaluation complete.")


if __name__ == "__main__":
    main()
