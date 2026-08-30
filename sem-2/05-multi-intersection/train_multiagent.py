"""Train compact-v2 independent/cooperative Q-agents with held-out validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_DIR = Path(__file__).resolve().parent
SEM2_ROOT = TASK_DIR.parent
sys.path.insert(0, str(SEM2_ROOT))
sys.path.insert(0, str(TASK_DIR))

from common.experiment_utils import (  # noqa: E402
    REPO_ROOT,
    ensure_output_tree,
    require_sumo,
    set_global_seed,
    summarize_episode,
    write_json,
)
from multiagent_policy import (  # noqa: E402
    EXPERIMENT_TAG,
    PRESSURE_OVERRIDE_GAP,
    checkpoint_path,
    compact_state,
    load_checkpoint,
    max_pressure_action,
    phase_pressure_scores,
    pressure_guarded_action,
    q_values,
    run_multiagent_episode,
    save_checkpoint,
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


def plot_training(summary_path: Path, plot_path: Path, mode: str, network: str) -> None:
    if not summary_path.exists():
        return
    frame = pd.read_csv(summary_path)
    selected = frame[(frame.method == mode) & (frame.network == network)].sort_values("episode")
    if selected.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(selected.episode, selected.mean_system_total_waiting_time, marker="o")
    axes[0].set_title("Mean system waiting time")
    axes[1].plot(selected.episode, selected.mean_system_total_stopped, marker="o")
    axes[1].set_title("Mean stopped vehicles")
    axes[2].plot(selected.episode, selected.q_states, marker="o", label="Q states")
    axes[2].set_title("Compact states learned")
    for axis in axes:
        axis.set_xlabel("Training episode")
        axis.grid(alpha=0.25)
    fig.suptitle(f"{mode.title()} compact-v2 multi-agent learning")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def validate_tables(
    *,
    tables: dict,
    network_config: dict[str, Path],
    seconds: int,
    seeds: list[int],
    reward_fn: str,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    """Evaluate without learning and return a congestion-balanced score."""
    rows: list[dict[str, float | int]] = []
    for seed in seeds:
        set_global_seed(seed)
        frame = run_multiagent_episode(
            network_config=network_config,
            seconds=seconds,
            seed=seed,
            fixed=False,
            tables=tables,
            reward_fn=reward_fn,
        )
        waiting = float(frame.system_total_waiting_time.mean())
        stopped = float(frame.system_total_stopped.mean())
        throughput = float(frame.system_total_arrived.iloc[-1]) / max(seconds, 1) * 3600.0
        fallback_rate = float(frame.fallback_rate.iloc[-1])
        rows.append(
            {
                "seed": seed,
                "mean_waiting": waiting,
                "mean_stopped": stopped,
                "throughput_veh_per_hour": throughput,
                "fallback_rate": fallback_rate,
            }
        )
    aggregate = {
        "validation_score": float(np.mean([row["mean_waiting"] + 10.0 * row["mean_stopped"] for row in rows])),
        "mean_waiting": float(np.mean([row["mean_waiting"] for row in rows])),
        "mean_stopped": float(np.mean([row["mean_stopped"] for row in rows])),
        "throughput_veh_per_hour": float(np.mean([row["throughput_veh_per_hour"] for row in rows])),
        "fallback_rate": float(np.mean([row["fallback_rate"] for row in rows])),
    }
    return aggregate, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["independent", "cooperative"], default="cooperative")
    parser.add_argument("--network", choices=NETWORKS, default="2x2")
    parser.add_argument("--episodes", type=int, default=30, help="Final cumulative episode target")
    parser.add_argument("--episodes-per-session", type=int, default=3, help="Use 0 to run to target")
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.12)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--coordination", type=float, default=0.5)
    parser.add_argument("--reward", choices=["queue", "diff-waiting-time", "pressure"], default="queue")
    parser.add_argument("--initial-epsilon", type=float, default=0.35)
    parser.add_argument("--min-epsilon", type=float, default=0.03)
    parser.add_argument("--epsilon-decay", type=float, default=0.92)
    parser.add_argument("--guided-exploration", type=float, default=0.70)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--validation-seconds", type=int, default=900)
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[9501, 9502])
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    require_sumo()
    if not 0.0 <= args.guided_exploration <= 1.0:
        raise SystemExit("--guided-exploration must be between 0 and 1")

    paths = ensure_output_tree(TASK_DIR)
    training_dir = paths["results"] / f"training-episodes-{EXPERIMENT_TAG}"
    training_dir.mkdir(exist_ok=True)
    latest = checkpoint_path(paths["checkpoints"], args.mode, args.network, "latest")
    best = checkpoint_path(paths["checkpoints"], args.mode, args.network, "best")
    saved = None if args.fresh else load_checkpoint(latest)
    if saved:
        expected = {
            "experiment_tag": EXPERIMENT_TAG,
            "mode": args.mode,
            "network": args.network,
            "seconds": args.seconds,
            "reward_fn": args.reward,
        }
        mismatches = [key for key, value in expected.items() if saved.get(key) != value]
        if mismatches:
            raise SystemExit(f"Checkpoint mismatch for {mismatches}. Use matching arguments or --fresh.")
        tables = saved["tables"]
        start_episode = int(saved["completed_episode"]) + 1
        epsilon = float(saved["epsilon"])
        best_score = float(saved.get("best_validation_score", np.inf))
        print(f"[resume] {EXPERIMENT_TAG} episode {start_episode}, epsilon={epsilon:.4f}")
    else:
        tables: dict[str, dict[tuple[int, ...], np.ndarray]] = {}
        start_episode = 1
        epsilon = args.initial_epsilon
        best_score = np.inf

    coordination = 0.0 if args.mode == "independent" else args.coordination
    config = NETWORKS[args.network]
    summary_path = paths["results"] / f"training_summary_{EXPERIMENT_TAG}.csv"
    validation_path = paths["results"] / f"validation_history_{args.mode}_{args.network}_{EXPERIMENT_TAG}.csv"
    prior_rows = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []
    if args.fresh:
        prior_rows = [
            row
            for row in prior_rows
            if not (row.get("method") == args.mode and row.get("network") == args.network)
        ]
        validation_rows: list[dict[str, Any]] = []
    else:
        validation_rows = pd.read_csv(validation_path).to_dict("records") if validation_path.exists() else []

    from sumo_rl import SumoEnvironment

    session_end = args.episodes if args.episodes_per_session == 0 else min(
        args.episodes, start_episode + args.episodes_per_session - 1
    )
    last_completed = start_episode - 1
    for episode in range(start_episode, session_end + 1):
        episode_seed = args.seed + episode
        set_global_seed(episode_seed)
        env = SumoEnvironment(
            net_file=str(config["net"]),
            route_file=str(config["route"]),
            use_gui=False,
            num_seconds=args.seconds,
            single_agent=False,
            sumo_seed=episode_seed,
            reward_fn=args.reward,
            enforce_max_green=True,
            max_green=60,
            out_csv_name=None,
        )
        records: list[dict[str, Any]] = []
        guided_actions = 0
        random_actions = 0
        greedy_actions = 0
        pressure_overrides = 0
        try:
            observations = env.reset(seed=episode_seed)
            states = {agent: compact_state(env, agent) for agent in observations}
            done = {"__all__": False}
            total_reward = 0.0
            while not done["__all__"]:
                actions: dict[str, int] = {}
                for agent, state in states.items():
                    scores = phase_pressure_scores(env, agent)
                    values = q_values(tables, agent, state, env.action_spaces(agent).n, scores)
                    if np.random.random() < epsilon:
                        if np.random.random() < args.guided_exploration:
                            actions[agent] = max_pressure_action(env, agent)
                            guided_actions += 1
                        else:
                            actions[agent] = int(np.random.randint(env.action_spaces(agent).n))
                            random_actions += 1
                    else:
                        actions[agent], used_override = pressure_guarded_action(
                            env,
                            agent,
                            values,
                            scores,
                        )
                        pressure_overrides += int(used_override)
                        greedy_actions += 1

                next_observations, rewards, done, info = env.step(actions)
                team_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
                next_states = {agent: compact_state(env, agent) for agent in next_observations}
                for agent, action in actions.items():
                    local_reward = float(rewards.get(agent, 0.0))
                    effective_reward = (1.0 - coordination) * local_reward + coordination * team_reward
                    old_values = q_values(tables, agent, states[agent], env.action_spaces(agent).n)
                    if done["__all__"] or agent not in next_states:
                        target = effective_reward
                    else:
                        next_scores = phase_pressure_scores(env, agent)
                        next_values = q_values(
                            tables,
                            agent,
                            next_states[agent],
                            env.action_spaces(agent).n,
                            next_scores,
                        )
                        target = effective_reward + args.gamma * float(np.max(next_values))
                    old_values[action] += args.alpha * (target - old_values[action])
                    total_reward += effective_reward
                states = next_states
                records.append({"step": float(env.sim_step), "reward": team_reward, **info})
        finally:
            env.close()

        frame = pd.DataFrame(records)
        raw_path = training_dir / f"{args.mode}_{args.network}_{EXPERIMENT_TAG}_episode_{episode}.csv"
        frame.to_csv(raw_path, index=False)
        row = summarize_episode(
            frame,
            method=args.mode,
            seed=episode_seed,
            seconds=args.seconds,
            total_reward=total_reward,
        )
        row.update(
            {
                "episode": episode,
                "network": args.network,
                "experiment_tag": EXPERIMENT_TAG,
                "epsilon": epsilon,
                "coordination": coordination,
                "q_states": sum(len(table) for table in tables.values()),
                "guided_actions": guided_actions,
                "random_actions": random_actions,
                "greedy_actions": greedy_actions,
                "pressure_overrides": pressure_overrides,
            }
        )
        prior_rows = [
            item
            for item in prior_rows
            if not (
                item.get("method") == args.mode
                and item.get("network") == args.network
                and int(item.get("episode", -1)) == episode
            )
        ]
        prior_rows.append(row)
        pd.DataFrame(prior_rows).sort_values(["network", "method", "episode"]).to_csv(summary_path, index=False)

        epsilon = max(args.min_epsilon, epsilon * args.epsilon_decay)
        payload: dict[str, Any] = {
            "experiment_tag": EXPERIMENT_TAG,
            "state_encoding": "current phase + min-green + one five-level pressure bin per action",
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
            "reward_fn": args.reward,
            "guided_exploration": args.guided_exploration,
            "pressure_override_gap": PRESSURE_OVERRIDE_GAP,
            "best_validation_score": best_score,
        }

        if args.validation_every > 0 and episode % args.validation_every == 0:
            aggregate, seed_rows = validate_tables(
                tables=tables,
                network_config=config,
                seconds=args.validation_seconds,
                seeds=args.validation_seeds,
                reward_fn=args.reward,
            )
            for seed_row in seed_rows:
                validation_rows.append(
                    {
                        "episode": episode,
                        "mode": args.mode,
                        "network": args.network,
                        **seed_row,
                    }
                )
            pd.DataFrame(validation_rows).to_csv(validation_path, index=False)
            improved = aggregate["validation_score"] < best_score
            if improved:
                best_score = aggregate["validation_score"]
                payload["best_validation_score"] = best_score
                payload["best_validation_metrics"] = aggregate
                save_checkpoint(best, payload)
                print(
                    f"[best] episode {episode}: score={best_score:.2f}, "
                    f"wait={aggregate['mean_waiting']:.2f}, queue={aggregate['mean_stopped']:.2f}, "
                    f"fallback={aggregate['fallback_rate']:.1%}"
                )

        payload["best_validation_score"] = best_score
        save_checkpoint(latest, payload)
        save_checkpoint(
            paths["checkpoints"] / f"{args.mode}_{args.network}_{EXPERIMENT_TAG}_episode_{episode}.pkl",
            payload,
        )
        plot_training(
            summary_path,
            paths["plots"] / f"training_{args.mode}_{args.network}_{EXPERIMENT_TAG}.png",
            args.mode,
            args.network,
        )
        print(
            f"Episode {episode}/{args.episodes}: wait={row['mean_system_total_waiting_time']:.2f}, "
            f"queue={row['mean_system_total_stopped']:.2f}, states={row['q_states']}, epsilon={epsilon:.3f}"
        )
        last_completed = episode

    final_path = checkpoint_path(paths["checkpoints"], args.mode, args.network, "final")
    target_reached = last_completed >= args.episodes
    if target_reached:
        selected = load_checkpoint(latest)
        selected_from = "latest"
        best_candidate = load_checkpoint(best)
        if best_candidate is not None:
            best_matches_run = (
                best_candidate.get("experiment_tag") == EXPERIMENT_TAG
                and best_candidate.get("mode") == args.mode
                and best_candidate.get("network") == args.network
                and int(best_candidate.get("seconds", -1)) == args.seconds
                and int(best_candidate.get("target_episodes", -1)) == args.episodes
                and best_candidate.get("reward_fn") == args.reward
            )
            if best_matches_run:
                selected = best_candidate
                selected_from = "best_validation"
        if selected is not None:
            selected["selected_for_final"] = selected_from
            selected["policy_checkpoint_episode"] = int(selected.get("completed_episode", last_completed))
            selected["training_completed_episode"] = last_completed
            selected["target_episodes"] = args.episodes
            save_checkpoint(final_path, selected)

    write_json(
        paths["results"] / f"config_{args.mode}_{args.network}_{EXPERIMENT_TAG}.json",
        vars(args)
        | {
            "experiment_tag": EXPERIMENT_TAG,
            "effective_coordination": coordination,
            "latest_checkpoint": str(latest.resolve()),
            "best_checkpoint": str(best.resolve()),
            "final_checkpoint": str(final_path.resolve()),
        },
    )
    if target_reached:
        print(f"Training target reached. Validation-selected controller: {final_path}")
    else:
        print(f"Session limit reached at episode {last_completed}. Rerun the same command without --fresh.")


if __name__ == "__main__":
    main()
