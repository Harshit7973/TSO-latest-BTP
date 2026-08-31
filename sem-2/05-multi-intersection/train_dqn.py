"""Train a resumable shared Dueling Double DQN on the four-signal 2x2 grid."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


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
from dqn_core import (  # noqa: E402
    ACTION_COUNT,
    EXPERIMENT_TAG,
    STATE_DIM,
    DuelingQNetwork,
    PrioritizedReplay,
    Transition,
    atomic_torch_save,
    batch_q_values,
    beta_at_step,
    build_state_bundle,
    capture_rng_state,
    checkpoint_path,
    epsilon_at_step,
    hard_update,
    load_torch_checkpoint,
    masked_argmax,
    optimize_double_dqn,
    resolve_device,
    restore_rng_state,
    run_controller_episode,
)


NETWORK = {
    "net": REPO_ROOT / "sumo_rl/nets/2x2grid/2x2.net.xml",
    "route": REPO_ROOT / "sumo_rl/nets/2x2grid/2x2.rou.xml",
}


def episode_metrics(frame: pd.DataFrame, seconds: int) -> dict[str, float]:
    return {
        "mean_waiting": float(frame.system_total_waiting_time.mean()),
        "mean_stopped": float(frame.system_total_stopped.mean()),
        "mean_speed": float(frame.system_mean_speed.mean()),
        "throughput_veh_per_hour": float(frame.system_total_arrived.iloc[-1])
        / max(seconds, 1)
        * 3600.0,
        "teleported": float(frame.system_total_teleported.iloc[-1]),
        "score": float(
            frame.system_total_waiting_time.mean()
            + 10.0 * frame.system_total_stopped.mean()
        ),
    }


def validation_baselines(
    *,
    cache_path: Path,
    seconds: int,
    seeds: list[int],
    pressure_gap: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    signature = {"seconds": seconds, "seeds": seeds, "pressure_gap": pressure_gap}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("signature") == signature:
            return cached["rows"]

    rows: list[dict[str, Any]] = []
    for controller in ("fixed", "max_pressure"):
        for seed in seeds:
            set_global_seed(seed)
            frame = run_controller_episode(
                network_config=NETWORK,
                seconds=seconds,
                seed=seed,
                controller=controller,
                pressure_gap=pressure_gap,
                model=None,
                device=device,
            )
            rows.append(
                {
                    "episode": 0,
                    "controller": controller,
                    "seed": seed,
                    **episode_metrics(frame, seconds),
                }
            )
    write_json(cache_path, {"signature": signature, "rows": rows})
    return rows


def validate_dqn(
    *,
    model: DuelingQNetwork,
    episode: int,
    seconds: int,
    seeds: list[int],
    pressure_gap: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        set_global_seed(seed)
        frame = run_controller_episode(
            network_config=NETWORK,
            seconds=seconds,
            seed=seed,
            controller="dqn_shielded",
            pressure_gap=pressure_gap,
            model=model,
            device=device,
        )
        rows.append(
            {
                "episode": episode,
                "controller": "dqn_shielded",
                "seed": seed,
                "shield_rate": float(frame.shield_rate.iloc[-1]),
                "expert_agreement_rate": float(frame.expert_agreement_rate.iloc[-1]),
                **episode_metrics(frame, seconds),
            }
        )
    model.train()
    return rows


def aggregate_validation(rows: list[dict[str, Any]], controller: str) -> dict[str, float]:
    selected = [row for row in rows if row["controller"] == controller]
    if not selected:
        raise ValueError(f"No validation rows for {controller}")
    keys = ("score", "mean_waiting", "mean_stopped", "mean_speed", "throughput_veh_per_hour")
    return {key: float(np.mean([float(row[key]) for row in selected])) for key in keys}


def checkpoint_payload(
    *,
    args: argparse.Namespace,
    online: DuelingQNetwork,
    target: DuelingQNetwork,
    optimizer: torch.optim.Optimizer,
    replay: PrioritizedReplay,
    episode: int,
    environment_step: int,
    gradient_step: int,
    learning_step: int,
    best_validation_score: float,
    best_validation: dict[str, float] | None,
    best_qualified_score: float,
    best_qualified_validation: dict[str, float] | None,
    rng: np.random.Generator,
    include_replay: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment_tag": EXPERIMENT_TAG,
        "algorithm": "parameter-sharing dueling double DQN with prioritized replay",
        "state_dim": STATE_DIM,
        "action_count": ACTION_COUNT,
        "completed_episode": episode,
        "target_episodes": args.episodes,
        "seconds": args.seconds,
        "seed": args.seed,
        "pressure_gap": args.pressure_gap,
        "coordination": args.coordination,
        "reward_scale": args.reward_scale,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "replay_capacity": args.replay_capacity,
        "replay_alpha": args.replay_alpha,
        "beta_start": args.beta_start,
        "learning_start": args.learning_start,
        "train_every": args.train_every,
        "target_update": args.target_update,
        "expert_episodes": args.expert_episodes,
        "expert_loss_weight": args.expert_loss_weight,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "validation_every": args.validation_every,
        "validation_seconds": args.validation_seconds,
        "validation_seeds": list(args.validation_seeds),
        "gate_score_fraction": args.gate_score_fraction,
        "gate_throughput_fraction": args.gate_throughput_fraction,
        "online_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "environment_step": environment_step,
        "gradient_step": gradient_step,
        "learning_step": learning_step,
        "best_validation_score": best_validation_score,
        "best_validation": best_validation,
        "best_qualified_score": best_qualified_score,
        "best_qualified_validation": best_qualified_validation,
        "rng_state": capture_rng_state(),
        "numpy_generator_state": copy.deepcopy(rng.bit_generator.state),
    }
    if include_replay:
        payload["replay_state"] = replay.state_dict()
    return payload


def plot_training(summary_path: Path, destination: Path) -> None:
    if not summary_path.exists():
        return
    frame = pd.read_csv(summary_path).sort_values("episode")
    if frame.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(frame.episode, frame.mean_system_total_waiting_time, marker="o")
    axes[0, 0].set_title("Training waiting measure")
    axes[0, 1].plot(frame.episode, frame.mean_system_total_stopped, marker="o")
    axes[0, 1].set_title("Training stopped vehicles")
    axes[1, 0].plot(frame.episode, frame.mean_loss, marker="o")
    axes[1, 0].set_title("Mean DQN loss")
    axes[1, 1].plot(frame.episode, frame.expert_agreement_rate, marker="o", label="Expert agreement")
    axes[1, 1].plot(frame.episode, frame.shield_rate, marker="o", label="Shield rate")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.25)
    fig.suptitle("Shared shielded Dueling Double DQN")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_validation(history_path: Path, destination: Path) -> None:
    if not history_path.exists():
        return
    frame = pd.read_csv(history_path)
    dqn = frame[frame.controller == "dqn_shielded"]
    baselines = frame[frame.episode == 0]
    if dqn.empty or baselines.empty:
        return
    grouped = dqn.groupby("episode").agg(score=("score", "mean"), throughput=("throughput_veh_per_hour", "mean"))
    fixed_score = baselines[baselines.controller == "fixed"].score.mean()
    pressure_score = baselines[baselines.controller == "max_pressure"].score.mean()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(grouped.index, grouped.score, marker="o", label="Shielded DQN")
    axes[0].axhline(fixed_score, color="tab:orange", linestyle="--", label="Fixed")
    axes[0].axhline(pressure_score, color="tab:green", linestyle="--", label="Max pressure")
    axes[0].set_ylabel("Waiting + 10 × stopped (lower is better)")
    axes[0].legend()
    axes[1].plot(grouped.index, grouped.throughput, marker="o")
    axes[1].set_ylabel("Throughput (vehicles/hour)")
    for axis in axes:
        axis.set_xlabel("Training episode")
        axis.grid(alpha=0.25)
    fig.suptitle("Held-out validation")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=40, help="Final cumulative target")
    parser.add_argument("--episodes-per-session", type=int, default=5, help="Use 0 for one uninterrupted run")
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=800)
    parser.add_argument("--expert-episodes", type=int, default=5)
    parser.add_argument("--pressure-gap", type=float, default=2.0)
    parser.add_argument("--coordination", type=float, default=0.25)
    parser.add_argument("--reward-scale", type=float, default=20.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--replay-alpha", type=float, default=0.6)
    parser.add_argument("--beta-start", type=float, default=0.4)
    parser.add_argument("--learning-start", type=int, default=1000)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--target-update", type=int, default=750)
    parser.add_argument("--expert-loss-weight", type=float, default=0.15)
    parser.add_argument("--epsilon-start", type=float, default=0.15)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay-steps", type=int, default=30000)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--validation-seconds", type=int, default=900)
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[9601, 9602])
    parser.add_argument("--gate-score-fraction", type=float, default=0.85)
    parser.add_argument("--gate-throughput-fraction", type=float, default=0.97)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    require_sumo()
    if args.episodes < args.expert_episodes:
        raise SystemExit("--episodes must be at least --expert-episodes")
    if not 0.0 <= args.coordination <= 1.0:
        raise SystemExit("--coordination must be between 0 and 1")
    if args.pressure_gap < 0:
        raise SystemExit("--pressure-gap cannot be negative")
    if args.replay_capacity < args.learning_start or args.learning_start < args.batch_size:
        raise SystemExit("Require replay-capacity >= learning-start >= batch-size")
    training_seeds = {args.seed + episode for episode in range(1, args.episodes + 1)}
    overlap = training_seeds & set(args.validation_seeds)
    if overlap:
        raise SystemExit(f"Training and validation seeds overlap: {sorted(overlap)}")

    device = resolve_device(args.device)
    set_global_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    paths = ensure_output_tree(TASK_DIR)
    latest_path = checkpoint_path(paths["checkpoints"], "latest")
    best_path = checkpoint_path(paths["checkpoints"], "best")
    qualified_path = checkpoint_path(paths["checkpoints"], "best_qualified")
    final_path = checkpoint_path(paths["checkpoints"], "final")
    summary_path = paths["results"] / f"training_summary_{EXPERIMENT_TAG}.csv"
    validation_path = paths["results"] / f"validation_history_{EXPERIMENT_TAG}.csv"
    baseline_cache = paths["results"] / f"validation_baselines_{EXPERIMENT_TAG}.json"

    online = DuelingQNetwork().to(device)
    target = DuelingQNetwork().to(device)
    hard_update(target, online)
    target.eval()
    optimizer = torch.optim.AdamW(online.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    replay = PrioritizedReplay(args.replay_capacity, args.replay_alpha)
    rng = np.random.default_rng(args.seed)
    start_episode = 1
    environment_step = 0
    gradient_step = 0
    learning_step = 0
    best_validation_score = float("inf")
    best_validation: dict[str, float] | None = None
    best_qualified_score = float("inf")
    best_qualified_validation: dict[str, float] | None = None

    saved = None if args.fresh else load_torch_checkpoint(latest_path, device)
    if saved is not None:
        expected = {
            "experiment_tag": EXPERIMENT_TAG,
            "seconds": args.seconds,
            "seed": args.seed,
            "pressure_gap": args.pressure_gap,
            "coordination": args.coordination,
            "reward_scale": args.reward_scale,
            "gamma": args.gamma,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "replay_capacity": args.replay_capacity,
            "replay_alpha": args.replay_alpha,
            "beta_start": args.beta_start,
            "learning_start": args.learning_start,
            "train_every": args.train_every,
            "target_update": args.target_update,
            "expert_episodes": args.expert_episodes,
            "expert_loss_weight": args.expert_loss_weight,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay_steps": args.epsilon_decay_steps,
            "validation_every": args.validation_every,
            "validation_seconds": args.validation_seconds,
            "validation_seeds": list(args.validation_seeds),
            "gate_score_fraction": args.gate_score_fraction,
            "gate_throughput_fraction": args.gate_throughput_fraction,
            "state_dim": STATE_DIM,
            "action_count": ACTION_COUNT,
        }
        mismatches = [key for key, value in expected.items() if saved.get(key) != value]
        if mismatches:
            raise SystemExit(f"Checkpoint mismatch for {mismatches}. Use matching arguments or --fresh.")
        online.load_state_dict(saved["online_state_dict"])
        target.load_state_dict(saved["target_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        replay = PrioritizedReplay.from_state_dict(saved["replay_state"])
        start_episode = int(saved["completed_episode"]) + 1
        environment_step = int(saved["environment_step"])
        gradient_step = int(saved["gradient_step"])
        learning_step = int(saved["learning_step"])
        best_validation_score = float(saved.get("best_validation_score", float("inf")))
        best_validation = saved.get("best_validation")
        best_qualified_score = float(saved.get("best_qualified_score", float("inf")))
        best_qualified_validation = saved.get("best_qualified_validation")
        restore_rng_state(saved.get("rng_state"))
        rng.bit_generator.state = saved["numpy_generator_state"]
        print(
            f"[resume] episode={start_episode}, replay={len(replay)}, "
            f"gradient_step={gradient_step}, device={device}"
        )

    prior_rows = [] if args.fresh or not summary_path.exists() else pd.read_csv(summary_path).to_dict("records")
    baseline_rows = validation_baselines(
        cache_path=baseline_cache,
        seconds=args.validation_seconds,
        seeds=args.validation_seeds,
        pressure_gap=args.pressure_gap,
        device=device,
    )
    fixed_validation = aggregate_validation(baseline_rows, "fixed")
    pressure_validation = aggregate_validation(baseline_rows, "max_pressure")
    if args.fresh or not validation_path.exists():
        validation_rows = baseline_rows.copy()
    else:
        validation_rows = pd.read_csv(validation_path).to_dict("records")

    from sumo_rl import SumoEnvironment

    session_end = args.episodes if args.episodes_per_session == 0 else min(
        args.episodes,
        start_episode + args.episodes_per_session - 1,
    )
    last_completed = start_episode - 1
    for episode in range(start_episode, session_end + 1):
        episode_seed = args.seed + episode
        set_global_seed(episode_seed)
        expert_episode = episode <= args.expert_episodes
        env = SumoEnvironment(
            net_file=str(NETWORK["net"]),
            route_file=str(NETWORK["route"]),
            use_gui=False,
            num_seconds=args.seconds,
            single_agent=False,
            sumo_seed=episode_seed,
            reward_fn="queue",
            enforce_max_green=True,
            max_green=60,
            out_csv_name=None,
        )
        records: list[dict[str, Any]] = []
        losses: list[dict[str, float]] = []
        decisions = 0
        interventions = 0
        agreements = 0
        exploratory_actions = 0
        total_scaled_reward = 0.0
        try:
            observations = env.reset(seed=episode_seed)
            agent_order = sorted(observations)
            done = {"__all__": False}
            while not done["__all__"]:
                states, masks, experts = build_state_bundle(
                    env,
                    agent_order,
                    args.seconds,
                    args.pressure_gap,
                )
                epsilon = 0.0 if expert_episode else epsilon_at_step(
                    learning_step,
                    args.epsilon_start,
                    args.epsilon_end,
                    args.epsilon_decay_steps,
                )
                actions: dict[str, int] = {}
                network_values = batch_q_values(online, states, agent_order, device)
                for agent in observations:
                    raw_values = network_values[agent]
                    raw_action = int(np.argmax(raw_values))
                    if expert_episode:
                        action = experts[agent]
                    elif rng.random() < epsilon:
                        valid_actions = np.flatnonzero(masks[agent])
                        action = int(rng.choice(valid_actions))
                        exploratory_actions += 1
                    else:
                        action = masked_argmax(raw_values, masks[agent])
                    actions[agent] = action
                    interventions += int(action != raw_action)
                    agreements += int(action == experts[agent])
                    decisions += 1

                next_observations, rewards, done, info = env.step(actions)
                environment_step += 1
                team_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
                if done["__all__"]:
                    next_states = {
                        agent: np.zeros(STATE_DIM, dtype=np.float32) for agent in actions
                    }
                    next_masks = {
                        agent: np.ones(ACTION_COUNT, dtype=np.bool_) for agent in actions
                    }
                else:
                    next_states, next_masks, _ = build_state_bundle(
                        env,
                        agent_order,
                        args.seconds,
                        args.pressure_gap,
                    )

                for agent, action in actions.items():
                    local_reward = float(rewards.get(agent, 0.0))
                    effective_reward = (
                        (1.0 - args.coordination) * local_reward
                        + args.coordination * team_reward
                    )
                    scaled_reward = float(np.clip(effective_reward / args.reward_scale, -5.0, 1.0))
                    total_scaled_reward += scaled_reward
                    replay.add(
                        Transition(
                            state=states[agent],
                            action=action,
                            reward=scaled_reward,
                            next_state=next_states[agent],
                            done=bool(done["__all__"]),
                            next_mask=next_masks[agent],
                            expert=expert_episode,
                        )
                    )

                if len(replay) >= args.learning_start and environment_step % args.train_every == 0:
                    beta = beta_at_step(gradient_step, args.beta_start, args.epsilon_decay_steps)
                    losses.append(
                        optimize_double_dqn(
                            online=online,
                            target=target,
                            optimizer=optimizer,
                            replay=replay,
                            batch_size=args.batch_size,
                            gamma=args.gamma,
                            beta=beta,
                            expert_loss_weight=args.expert_loss_weight,
                            rng=rng,
                            device=device,
                        )
                    )
                    gradient_step += 1
                    if gradient_step % args.target_update == 0:
                        hard_update(target, online)
                if not expert_episode:
                    learning_step += len(actions)
                observations = next_observations
                records.append(
                    {
                        "step": float(env.sim_step),
                        "reward": team_reward,
                        "scaled_training_reward": total_scaled_reward,
                        "agent_decisions": decisions,
                        "shield_interventions": interventions,
                        "shield_rate": interventions / max(decisions, 1),
                        "expert_agreements": agreements,
                        "expert_agreement_rate": agreements / max(decisions, 1),
                        **info,
                    }
                )
        finally:
            env.close()

        frame = pd.DataFrame(records)
        episode_dir = paths["results"] / f"training_episodes_{EXPERIMENT_TAG}"
        episode_dir.mkdir(exist_ok=True)
        frame.to_csv(episode_dir / f"episode_{episode:03d}.csv", index=False)
        row = summarize_episode(
            frame,
            method="shared_dqn_expert" if expert_episode else "shared_dqn",
            seed=episode_seed,
            seconds=args.seconds,
            total_reward=total_scaled_reward,
        )
        mean_loss = float(np.mean([item["loss"] for item in losses])) if losses else 0.0
        row.update(
            {
                "episode": episode,
                "expert_episode": expert_episode,
                "epsilon": epsilon,
                "replay_size": len(replay),
                "gradient_step": gradient_step,
                "mean_loss": mean_loss,
                "mean_td_loss": float(np.mean([item["td_loss"] for item in losses])) if losses else 0.0,
                "mean_imitation_loss": float(np.mean([item["imitation_loss"] for item in losses])) if losses else 0.0,
                "shield_rate": interventions / max(decisions, 1),
                "expert_agreement_rate": agreements / max(decisions, 1),
                "exploratory_actions": exploratory_actions,
                "device": str(device),
            }
        )
        prior_rows = [item for item in prior_rows if int(item.get("episode", -1)) != episode]
        prior_rows.append(row)
        pd.DataFrame(prior_rows).sort_values("episode").to_csv(summary_path, index=False)

        current_payload = checkpoint_payload(
            args=args,
            online=online,
            target=target,
            optimizer=optimizer,
            replay=replay,
            episode=episode,
            environment_step=environment_step,
            gradient_step=gradient_step,
            learning_step=learning_step,
            best_validation_score=best_validation_score,
            best_validation=best_validation,
            best_qualified_score=best_qualified_score,
            best_qualified_validation=best_qualified_validation,
            rng=rng,
            include_replay=True,
        )

        if args.validation_every > 0 and episode % args.validation_every == 0:
            new_rows = validate_dqn(
                model=online,
                episode=episode,
                seconds=args.validation_seconds,
                seeds=args.validation_seeds,
                pressure_gap=args.pressure_gap,
                device=device,
            )
            validation_rows = [
                item
                for item in validation_rows
                if not (item.get("controller") == "dqn_shielded" and int(item.get("episode", -1)) == episode)
            ]
            validation_rows.extend(new_rows)
            pd.DataFrame(validation_rows).sort_values(["episode", "controller", "seed"]).to_csv(
                validation_path,
                index=False,
            )
            aggregate = aggregate_validation(new_rows, "dqn_shielded")
            qualifies = bool(
                aggregate["score"] <= args.gate_score_fraction * fixed_validation["score"]
                and aggregate["throughput_veh_per_hour"]
                >= args.gate_throughput_fraction * fixed_validation["throughput_veh_per_hour"]
            )
            if aggregate["score"] < best_validation_score:
                best_validation_score = aggregate["score"]
                best_validation = aggregate | {"episode": float(episode)}
                current_payload["best_validation_score"] = best_validation_score
                current_payload["best_validation"] = best_validation
                best_payload = dict(current_payload)
                best_payload.pop("replay_state", None)
                atomic_torch_save(best_path, best_payload)
                print(
                    f"[best] episode={episode}, score={aggregate['score']:.2f}, "
                    f"wait={aggregate['mean_waiting']:.2f}, queue={aggregate['mean_stopped']:.2f}"
                )
            if qualifies and aggregate["score"] < best_qualified_score:
                best_qualified_score = aggregate["score"]
                best_qualified_validation = aggregate | {"episode": float(episode)}
                qualified_payload = dict(current_payload)
                qualified_payload["best_qualified_score"] = best_qualified_score
                qualified_payload["best_qualified_validation"] = best_qualified_validation
                qualified_payload.pop("replay_state", None)
                atomic_torch_save(qualified_path, qualified_payload)
                print(
                    f"[qualified] episode={episode}, score={aggregate['score']:.2f}, "
                    f"throughput={aggregate['throughput_veh_per_hour']:.1f}/h"
                )

        current_payload["best_validation_score"] = best_validation_score
        current_payload["best_validation"] = best_validation
        current_payload["best_qualified_score"] = best_qualified_score
        current_payload["best_qualified_validation"] = best_qualified_validation
        atomic_torch_save(latest_path, current_payload)
        if episode % 5 == 0:
            milestone = dict(current_payload)
            milestone.pop("replay_state", None)
            atomic_torch_save(
                paths["checkpoints"] / f"{EXPERIMENT_TAG}_episode_{episode:03d}.pt",
                milestone,
            )
        plot_training(summary_path, paths["plots"] / f"training_{EXPERIMENT_TAG}.png")
        plot_validation(validation_path, paths["plots"] / f"validation_{EXPERIMENT_TAG}.png")
        print(
            f"Episode {episode}/{args.episodes}: wait={row['mean_system_total_waiting_time']:.2f}, "
            f"queue={row['mean_system_total_stopped']:.2f}, epsilon={epsilon:.3f}, "
            f"replay={len(replay)}, device={device}"
        )
        last_completed = episode

    target_reached = last_completed >= args.episodes
    if target_reached:
        def compatible(candidate: dict[str, Any] | None) -> bool:
            return bool(
                candidate
                and candidate.get("experiment_tag") == EXPERIMENT_TAG
                and int(candidate.get("seconds", -1)) == args.seconds
                and int(candidate.get("target_episodes", -1)) == args.episodes
            )

        selected = load_torch_checkpoint(qualified_path, device)
        selected_kind = "best_qualified"
        if not compatible(selected):
            selected = None
        if selected is None:
            selected = load_torch_checkpoint(best_path, device)
            selected_kind = "best_validation"
            if not compatible(selected):
                selected = None
        if selected is None:
            selected = load_torch_checkpoint(latest_path, device)
            selected_kind = "latest"
            if not compatible(selected):
                selected = None
        selected_validation = (
            selected.get("best_qualified_validation")
            if selected_kind == "best_qualified"
            else selected.get("best_validation") if selected else None
        )
        qualifies = selected_kind == "best_qualified"
        deployment_policy = "dqn_shielded" if qualifies else "max_pressure"
        if selected is not None:
            selected["training_completed_episode"] = last_completed
            selected["target_episodes"] = args.episodes
            selected["deployment_policy"] = deployment_policy
            selected["selected_checkpoint_kind"] = selected_kind
            selected["deployment_gate"] = {
                "passed": qualifies,
                "requirements": {
                    "dqn_score_lte_fraction_of_fixed": args.gate_score_fraction,
                    "dqn_throughput_gte_fraction_of_fixed": args.gate_throughput_fraction,
                },
                "dqn_validation": selected_validation,
                "fixed_validation": fixed_validation,
                "max_pressure_validation": pressure_validation,
            }
            atomic_torch_save(final_path, selected)
        if selected is None:
            raise RuntimeError("Training reached its target but no checkpoint could be selected")
        write_json(paths["results"] / f"deployment_gate_{EXPERIMENT_TAG}.json", selected["deployment_gate"])
        print(f"Training complete. Deployment policy: {deployment_policy}. Final: {final_path}")
    else:
        print(f"Session stopped safely at episode {last_completed}. Rerun the same command without --fresh.")

    write_json(
        paths["results"] / f"training_config_{EXPERIMENT_TAG}.json",
        vars(args)
        | {
            "experiment_tag": EXPERIMENT_TAG,
            "command_line": sys.argv,
            "state_dim": STATE_DIM,
            "action_count": ACTION_COUNT,
            "resolved_device": str(device),
            "torch_version": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "trainable_parameters": sum(parameter.numel() for parameter in online.parameters()),
            "latest_checkpoint": str(latest_path.resolve()),
            "best_checkpoint": str(best_path.resolve()),
            "best_qualified_checkpoint": str(qualified_path.resolve()),
            "final_checkpoint": str(final_path.resolve()),
        },
    )


if __name__ == "__main__":
    main()
