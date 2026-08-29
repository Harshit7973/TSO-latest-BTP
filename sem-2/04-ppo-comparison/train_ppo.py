"""Train/resume PPO as a bounded-cost algorithmic comparison to DQN."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TASK_DIR = Path(__file__).resolve().parent
SEM2_ROOT = TASK_DIR.parent
sys.path.insert(0, str(SEM2_ROOT))

from common.experiment_utils import (  # noqa: E402
    NETWORKS,
    build_single_agent_env,
    ensure_output_tree,
    read_json,
    require_sumo,
    set_global_seed,
    wrap_with_incrementing_seeds,
    write_json,
)


def plot_curve(logs_dir: Path, output: Path) -> None:
    files = sorted(logs_dir.glob("training_session_*.monitor.csv"))
    if not files:
        return
    frame = pd.concat([pd.read_csv(path, comment="#") for path in files], ignore_index=True)
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frame.index + 1, frame.r, alpha=0.3, label="episode return")
    ax.plot(frame.index + 1, frame.r.rolling(10, min_periods=1).mean(), label="10-episode mean")
    ax.set(xlabel="Episode", ylabel="Return", title="PPO training curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=60_000)
    parser.add_argument("--chunk", type=int, default=10_240, help="Prefer multiples of PPO n_steps=512")
    parser.add_argument("--chunks-per-session", type=int, default=1, help="Use 0 to run all chunks without stopping")
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    require_sumo()
    set_global_seed(args.seed)

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    import stable_baselines3
    import torch

    paths = ensure_output_tree(TASK_DIR)
    route_dir = TASK_DIR / "generated-routes" / f"sec{args.seconds}"
    subprocess.run(
        [sys.executable, str(SEM2_ROOT / "02-dynamic-traffic/generate_routes.py"), "--seconds", str(args.seconds), "--output-dir", str(route_dir)],
        check=True,
    )
    route_file = route_dir / "direction_switch.rou.xml"
    latest = paths["checkpoints"] / "ppo_latest.zip"
    state_path = paths["checkpoints"] / "training_state.json"
    state = read_json(state_path, {})
    if latest.exists() and not args.fresh and state:
        if int(state.get("seconds_per_episode", args.seconds)) != args.seconds:
            raise SystemExit("Checkpoint episode length differs from --seconds. Use --fresh for the new experiment.")
        if int(state.get("seed", args.seed)) != args.seed:
            raise SystemExit("Checkpoint seed differs from --seed. Use the original seed or start with --fresh.")
    session_start = int(state.get("completed_timesteps", 0)) if latest.exists() and not args.fresh else 0

    cfg = NETWORKS["2way"]
    env = build_single_agent_env(
        net_file=cfg["net_file"],
        route_file=route_file,
        seconds=args.seconds,
        seed=args.seed,
        enforce_max_green=True,
    )
    approximate_completed_episodes = session_start // max(args.seconds // 5, 1)
    env = wrap_with_incrementing_seeds(env, args.seed, approximate_completed_episodes)
    env = Monitor(env, filename=str(paths["logs"] / f"training_session_{session_start}"))

    if latest.exists() and not args.fresh:
        print(f"[resume] loading {latest}")
        model = PPO.load(str(latest), env=env, device=args.device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
            tensorboard_log=str(paths["logs"] / "tensorboard"),
            seed=args.seed,
            device=args.device,
            verbose=1,
        )

    chunks_completed = 0
    while model.num_timesteps < args.timesteps and (
        args.chunks_per_session == 0 or chunks_completed < args.chunks_per_session
    ):
        requested = min(args.chunk, args.timesteps - model.num_timesteps)
        print(f"[train] current={model.num_timesteps:,}, requested additional={requested:,}")
        try:
            model.learn(total_timesteps=requested, reset_num_timesteps=False, progress_bar=True)
        except KeyboardInterrupt:
            env.close()
            print("Interrupted. The last fully completed chunk remains resumable; partial rollout was not promoted.")
            return
        model.save(str(latest))
        model.save(str(paths["checkpoints"] / f"ppo_step_{model.num_timesteps}.zip"))
        write_json(
            state_path,
            {
                "completed_timesteps": model.num_timesteps,
                "target_timesteps": args.timesteps,
                "seed": args.seed,
                "seconds_per_episode": args.seconds,
                "note": "PPO rollout collection can slightly exceed the requested target.",
            },
        )
        plot_curve(paths["logs"], paths["plots"] / "training_curve.png")
        chunks_completed += 1

    target_reached = model.num_timesteps >= args.timesteps
    if target_reached:
        model.save(str(paths["checkpoints"] / "ppo_final.zip"))
    env.close()
    write_json(
        paths["results"] / "training_metadata.json",
        {
            "python": platform.python_version(),
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "timesteps": model.num_timesteps,
            "target_timesteps": args.timesteps,
            "target_reached": target_reached,
            "route": str(route_file.resolve()),
        },
    )
    if target_reached:
        print(f"Final PPO saved to {paths['checkpoints'] / 'ppo_final.zip'}")
    else:
        print(f"Session chunk limit reached at {model.num_timesteps:,}. Rerun the same command to resume.")


if __name__ == "__main__":
    main()
