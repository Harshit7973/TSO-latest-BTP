"""Train resumable independent/cooperative tabular agents on 2x2 or RESCO 4x4."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_DIR = Path(__file__).resolve().parent
SEM2_ROOT = TASK_DIR.parent
sys.path.insert(0, str(SEM2_ROOT))

from common.experiment_utils import (  # noqa: E402
    REPO_ROOT,
    ensure_output_tree,
    require_sumo,
    set_global_seed,
    summarize_episode,
    write_json,
)


NETWORKS = {
    "2x2": {
        "net": REPO_ROOT / "sumo_rl/nets/2x2grid/2x2.net.xml",
        "route": REPO_ROOT / "sumo_rl/nets/2x2grid/2x2.rou.xml",
    },
    "resco4x4": {
        "net": REPO_ROOT / "sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml",
        "route": REPO_ROOT / "sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml",
    },
}


def checkpoint_path(paths, mode: str, network: str, final: bool = False) -> Path:
    suffix = "final" if final else "latest"
    return paths["checkpoints"] / f"{mode}_{network}_{suffix}.pkl"


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def q_values(tables: dict, agent: str, state: tuple, action_count: int) -> np.ndarray:
    table = tables.setdefault(agent, {})
    if state not in table:
        table[state] = np.zeros(action_count, dtype=np.float32)
    return table[state]


def plot_training(summary_path: Path, plot_path: Path, mode: str, network: str) -> None:
    if not summary_path.exists():
        return
    frame = pd.read_csv(summary_path)
    selected = frame[(frame.method == mode) & (frame.network == network)]
    if selected.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(selected.episode, selected.mean_system_total_waiting_time, marker="o")
    axes[0].set_title("Mean system waiting time")
    axes[1].plot(selected.episode, selected.mean_system_total_stopped, marker="o")
    axes[1].set_title("Mean stopped vehicles")
    for ax in axes:
        ax.set_xlabel("Training episode")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{mode.title()} multi-agent learning")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["independent", "cooperative"], default="cooperative")
    parser.add_argument("--network", choices=NETWORKS, default="2x2")
    parser.add_argument("--episodes", type=int, default=15, help="Final cumulative episode target")
    parser.add_argument("--episodes-per-session", type=int, default=3, help="Use 0 to run to the target without stopping")
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--coordination", type=float, default=0.5)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    require_sumo()
    paths = ensure_output_tree(TASK_DIR)
    training_dir = paths["results"] / "training-episodes"
    training_dir.mkdir(exist_ok=True)
    latest = checkpoint_path(paths, args.mode, args.network)
    saved = None if args.fresh else load_checkpoint(latest)
    if saved:
        if saved["mode"] != args.mode or saved["network"] != args.network:
            raise SystemExit("Checkpoint configuration mismatch")
        if int(saved["seconds"]) != args.seconds:
            raise SystemExit(
                "Checkpoint episode length differs from --seconds. Use --fresh for a clean final run "
                "or rerun with the checkpoint's original value."
            )
        tables = saved["tables"]
        start_episode = int(saved["completed_episode"]) + 1
        epsilon = float(saved["epsilon"])
        print(f"[resume] episode {start_episode}, epsilon={epsilon:.4f}")
    else:
        tables = {}
        start_episode = 1
        epsilon = 0.30

    coordination = 0.0 if args.mode == "independent" else args.coordination
    cfg = NETWORKS[args.network]
    summary_path = paths["results"] / "training_summary.csv"
    prior_rows = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []

    from sumo_rl import SumoEnvironment

    session_end = args.episodes if args.episodes_per_session == 0 else min(
        args.episodes, start_episode + args.episodes_per_session - 1
    )
    last_completed = start_episode - 1
    for episode in range(start_episode, session_end + 1):
        episode_seed = args.seed + episode
        set_global_seed(episode_seed)
        env = SumoEnvironment(
            net_file=str(cfg["net"]),
            route_file=str(cfg["route"]),
            use_gui=False,
            num_seconds=args.seconds,
            single_agent=False,
            sumo_seed=episode_seed,
            reward_fn="diff-waiting-time",
            enforce_max_green=True,
            max_green=60,
            out_csv_name=None,
        )
        records = []
        try:
            observations = env.reset(seed=episode_seed)
            states = {agent: env.encode(observation, agent) for agent, observation in observations.items()}
            done = {"__all__": False}
            total_reward = 0.0
            while not done["__all__"]:
                actions = {}
                for agent, state in states.items():
                    values = q_values(tables, agent, state, env.action_spaces(agent).n)
                    actions[agent] = int(np.random.randint(env.action_spaces(agent).n)) if np.random.random() < epsilon else int(np.argmax(values))
                next_observations, rewards, done, info = env.step(actions)
                team_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
                next_states = {agent: env.encode(obs, agent) for agent, obs in next_observations.items()}
                for agent, action in actions.items():
                    local_reward = float(rewards.get(agent, 0.0))
                    effective_reward = (1.0 - coordination) * local_reward + coordination * team_reward
                    old_values = q_values(tables, agent, states[agent], env.action_spaces(agent).n)
                    if done["__all__"] or agent not in next_states:
                        target = effective_reward
                    else:
                        next_values = q_values(tables, agent, next_states[agent], env.action_spaces(agent).n)
                        target = effective_reward + args.gamma * float(np.max(next_values))
                    old_values[action] += args.alpha * (target - old_values[action])
                    total_reward += effective_reward
                states = next_states
                records.append({"step": float(env.sim_step), "reward": team_reward, **info})
        finally:
            env.close()

        frame = pd.DataFrame(records)
        raw_path = training_dir / f"{args.mode}_{args.network}_episode_{episode}.csv"
        frame.to_csv(raw_path, index=False)
        row = summarize_episode(frame, method=args.mode, seed=episode_seed, seconds=args.seconds, total_reward=total_reward)
        row["episode"] = episode
        row["network"] = args.network
        row["epsilon"] = epsilon
        row["coordination"] = coordination
        row["q_states"] = sum(len(table) for table in tables.values())
        prior_rows = [item for item in prior_rows if not (item.get("method") == args.mode and item.get("network") == args.network and int(item.get("episode", -1)) == episode)]
        prior_rows.append(row)
        pd.DataFrame(prior_rows).sort_values(["network", "method", "episode"]).to_csv(summary_path, index=False)

        epsilon = max(0.02, epsilon * 0.90)
        payload = {
            "mode": args.mode,
            "network": args.network,
            "completed_episode": episode,
            "target_episodes": args.episodes,
            "epsilon": epsilon,
            "tables": tables,
            "seconds": args.seconds,
            "seed": args.seed,
            "coordination": coordination,
            "alpha": args.alpha,
            "gamma": args.gamma,
        }
        save_checkpoint(latest, payload)
        save_checkpoint(paths["checkpoints"] / f"{args.mode}_{args.network}_episode_{episode}.pkl", payload)
        plot_training(summary_path, paths["plots"] / f"training_{args.mode}_{args.network}.png", args.mode, args.network)
        print(f"Episode {episode}/{args.episodes}: mean wait={row['mean_system_total_waiting_time']:.2f}, epsilon={epsilon:.3f}")
        last_completed = episode

    final_path = checkpoint_path(paths, args.mode, args.network, final=True)
    target_reached = last_completed >= args.episodes
    if latest.exists() and target_reached:
        save_checkpoint(final_path, load_checkpoint(latest))
    write_json(
        paths["results"] / f"config_{args.mode}_{args.network}.json",
        vars(args) | {"effective_coordination": coordination, "checkpoint": str(final_path.resolve())},
    )
    if target_reached:
        print(f"Final controller saved to {final_path}")
    else:
        print(f"Session episode limit reached at {last_completed}. Rerun the same command without --fresh to resume.")


if __name__ == "__main__":
    main()
