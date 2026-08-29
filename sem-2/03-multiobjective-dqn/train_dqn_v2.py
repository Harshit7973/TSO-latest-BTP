"""Train/resume an enhanced DQN in bounded chunks for laptop operation."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TASK_DIR = Path(__file__).resolve().parent
SEM2_ROOT = TASK_DIR.parent
sys.path.insert(0, str(SEM2_ROOT))
sys.path.insert(0, str(TASK_DIR))

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
from features import EnhancedObservation, multi_objective_reward  # noqa: E402


def plot_learning_curve(logs_dir: Path, output: Path) -> None:
    monitor_files = sorted(logs_dir.glob("training_session_*.monitor.csv"))
    if not monitor_files:
        return
    frames = [pd.read_csv(path, comment="#") for path in monitor_files]
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty or "r" not in frame:
        return
    smooth = frame["r"].rolling(10, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frame.index + 1, frame["r"], alpha=0.3, label="episode return")
    ax.plot(frame.index + 1, smooth, linewidth=2, label="10-episode mean")
    ax.set(xlabel="Episode", ylabel="Return", title="DQN-v2 training curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=60_000, help="Final cumulative target")
    parser.add_argument("--chunk", type=int, default=10_000, help="Checkpoint interval")
    parser.add_argument("--chunks-per-session", type=int, default=1, help="Use 0 to run all chunks without stopping")
    parser.add_argument("--seconds", type=int, default=1800, help="Seconds per training episode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fresh", action="store_true", help="Ignore (but do not delete) latest checkpoint")
    args = parser.parse_args()
    require_sumo()
    set_global_seed(args.seed)

    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    import stable_baselines3
    import torch

    paths = ensure_output_tree(TASK_DIR)
    route_dir = TASK_DIR / "generated-routes" / f"sec{args.seconds}"
    route_generator = SEM2_ROOT / "02-dynamic-traffic" / "generate_routes.py"
    subprocess.run(
        [sys.executable, str(route_generator), "--seconds", str(args.seconds), "--output-dir", str(route_dir)],
        check=True,
    )
    route_file = route_dir / "direction_switch.rou.xml"
    cfg = NETWORKS["2way"]
    latest_model = paths["checkpoints"] / "dqn_v2_latest.zip"
    replay_buffer = paths["checkpoints"] / "dqn_v2_replay_buffer.pkl"
    state_path = paths["checkpoints"] / "training_state.json"
    previous_state = read_json(state_path, {})
    if latest_model.exists() and not args.fresh and previous_state:
        if int(previous_state.get("seconds_per_episode", args.seconds)) != args.seconds:
            raise SystemExit("Checkpoint episode length differs from --seconds. Use --fresh for the new experiment.")
        if int(previous_state.get("seed", args.seed)) != args.seed:
            raise SystemExit("Checkpoint seed differs from --seed. Use the original seed or start with --fresh.")
    session_start = int(previous_state.get("completed_timesteps", 0)) if latest_model.exists() and not args.fresh else 0

    env = build_single_agent_env(
        net_file=cfg["net_file"],
        route_file=route_file,
        seconds=args.seconds,
        seed=args.seed,
        observation_class=EnhancedObservation,
        reward_fn=multi_objective_reward,
        enforce_max_green=True,
    )
    approximate_completed_episodes = session_start // max(args.seconds // 5, 1)
    env = wrap_with_incrementing_seeds(env, args.seed, approximate_completed_episodes)
    monitor_path = paths["logs"] / f"training_session_{session_start}"
    env = Monitor(env, filename=str(monitor_path), allow_early_resets=True)

    if latest_model.exists() and not args.fresh:
        print(f"[resume] loading {latest_model}")
        model = DQN.load(str(latest_model), env=env, device=args.device)
        if replay_buffer.exists():
            model.load_replay_buffer(str(replay_buffer))
    else:
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=5e-4,
            buffer_size=50_000,
            learning_starts=2_000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.25,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.03,
            policy_kwargs={"net_arch": [256, 256]},
            tensorboard_log=str(paths["logs"] / "tensorboard"),
            seed=args.seed,
            device=args.device,
            verbose=1,
        )

    chunks_completed = 0
    while model.num_timesteps < args.timesteps and (
        args.chunks_per_session == 0 or chunks_completed < args.chunks_per_session
    ):
        step_count = min(args.chunk, args.timesteps - model.num_timesteps)
        target = model.num_timesteps + step_count
        print(f"[train] {model.num_timesteps:,} -> {target:,}")
        try:
            model.learn(total_timesteps=step_count, reset_num_timesteps=False, progress_bar=True)
        except KeyboardInterrupt:
            env.close()
            print("Interrupted. The last fully completed chunk remains resumable; partial chunk data was not promoted.")
            return
        model.save(str(latest_model))
        model.save_replay_buffer(str(replay_buffer))
        model.save(str(paths["checkpoints"] / f"dqn_v2_step_{model.num_timesteps}.zip"))
        write_json(
            state_path,
            {
                "completed_timesteps": model.num_timesteps,
                "target_timesteps": args.timesteps,
                "seed": args.seed,
                "seconds_per_episode": args.seconds,
                "previous_state": previous_state,
            },
        )
        plot_learning_curve(paths["logs"], paths["plots"] / "training_curve.png")
        chunks_completed += 1

    target_reached = model.num_timesteps >= args.timesteps
    if target_reached:
        model.save(str(paths["checkpoints"] / "dqn_v2_final.zip"))
    env.close()
    metadata = {
        "python": platform.python_version(),
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_requested": args.device,
        "timesteps": model.num_timesteps,
        "target_timesteps": args.timesteps,
        "target_reached": target_reached,
        "route": str(route_file.resolve()),
        "observation": "phase + min-green + elapsed + density + queue + lane-wait + outgoing-density",
        "reward": "delay improvement - queue - CO2 - max-lane-wait - switching",
    }
    write_json(paths["results"] / "training_metadata.json", metadata)
    if target_reached:
        print(f"Training target reached. Final model: {paths['checkpoints'] / 'dqn_v2_final.zip'}")
    else:
        print(f"Session chunk limit reached at {model.num_timesteps:,}. Rerun the same command to resume.")


if __name__ == "__main__":
    main()
