"""Train paired fine-tuned and scratch DQNs on a shifted 2x2 traffic domain."""

from __future__ import annotations

import argparse
import copy
import json
import re
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
sys.path.insert(0, str(SEM2_ROOT / "05-multi-intersection"))
sys.path.insert(0, str(TASK_DIR))

from common.experiment_utils import (  # noqa: E402
    REPO_ROOT,
    ensure_output_tree,
    require_sumo,
    set_global_seed,
    summarize_episode,
)
from common.multi_intersection_tools import (  # noqa: E402
    load_task5_model,
    run_extended_episode,
    sha256,
    write_json,
)
from dqn_core import (  # noqa: E402
    ACTION_COUNT,
    STATE_DIM,
    DuelingQNetwork,
    PrioritizedReplay,
    Transition,
    atomic_torch_save,
    batch_q_values,
    beta_at_step,
    build_state_bundle,
    epsilon_at_step,
    hard_update,
    masked_argmax,
    optimize_double_dqn,
    resolve_device,
)
from generate_target_routes import generate  # noqa: E402


NETWORK_FILE = REPO_ROOT / "sumo_rl/nets/2x2grid/2x2.net.xml"
MODES = ("fine_tuned", "scratch")


def validation_metrics(frame: pd.DataFrame, seconds: int) -> dict[str, float]:
    waiting = float(frame.system_total_waiting_time.mean())
    stopped = float(frame.system_total_stopped.mean())
    return {
        "mean_waiting": waiting,
        "mean_stopped": stopped,
        "mean_speed": float(frame.system_mean_speed.mean()),
        "throughput_veh_per_hour": float(frame.system_total_arrived.iloc[-1]) / seconds * 3600.0,
        "teleported": float(frame.system_total_teleported.iloc[-1]),
        "score": waiting + 10.0 * stopped,
    }


def validate(
    model: DuelingQNetwork,
    *,
    route: Path,
    episode: int,
    mode: str,
    seconds: int,
    seeds: list[int],
    pressure_gap: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    for seed in seeds:
        frame, _ = run_extended_episode(
            seconds=seconds,
            seed=seed,
            controller="dqn_shielded",
            pressure_gap=pressure_gap,
            model=model,
            device=device,
            route_file=route,
        )
        rows.append({"mode": mode, "episode": episode, "seed": seed, **validation_metrics(frame, seconds)})
    model.train()
    return rows


def average_score(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([float(row["score"]) for row in rows]))


def initialise_mode(
    mode: str,
    *,
    source_model: DuelingQNetwork,
    source_sha256: str,
    args: argparse.Namespace,
    route_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    set_global_seed(args.seed)
    online = DuelingQNetwork().to(device)
    if mode == "fine_tuned":
        online.load_state_dict(copy.deepcopy(source_model.state_dict()))
    target = DuelingQNetwork().to(device)
    hard_update(target, online)
    optimizer = torch.optim.Adam(online.parameters(), lr=args.learning_rate)
    return {
        "format": 1,
        "mode": mode,
        "completed_episode": 0,
        "target_episodes": args.episodes,
        "seconds": args.seconds,
        "seed": args.seed,
        "source_checkpoint_sha256": source_sha256,
        "target_route_sha256": route_sha256,
        "pressure_gap": args.pressure_gap,
        "coordination": args.coordination,
        "reward_scale": args.reward_scale,
        "online_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "replay_state": PrioritizedReplay(args.replay_capacity, args.replay_alpha).state_dict(),
        "environment_step": 0,
        "gradient_step": 0,
        "learning_step": 0,
        # A shared exploration stream plus paired SUMO seeds makes the initial
        # weights the intended difference between fine-tuning and scratch.
        "rng_state": np.random.default_rng(args.seed).bit_generator.state,
        "best_score": float("inf"),
        "best_episode": None,
        "validation_history": [],
    }


def restore_mode(payload: dict[str, Any], args: argparse.Namespace, device: torch.device):
    expected = {
        "target_episodes": args.episodes,
        "seconds": args.seconds,
        "seed": args.seed,
        "pressure_gap": args.pressure_gap,
        "coordination": args.coordination,
        "reward_scale": args.reward_scale,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Resume mismatch for {key}: {payload.get(key)} != {value}")
    online = DuelingQNetwork().to(device)
    target = DuelingQNetwork().to(device)
    online.load_state_dict(payload["online_state_dict"])
    target.load_state_dict(payload["target_state_dict"])
    optimizer = torch.optim.Adam(online.parameters(), lr=args.learning_rate)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    replay = PrioritizedReplay.from_state_dict(payload["replay_state"])
    rng = np.random.default_rng()
    rng.bit_generator.state = payload["rng_state"]
    return online, target, optimizer, replay, rng


def save_payload(
    path: Path,
    payload: dict[str, Any],
    *,
    online: DuelingQNetwork,
    target: DuelingQNetwork,
    optimizer: torch.optim.Optimizer,
    replay: PrioritizedReplay,
    rng: np.random.Generator,
) -> None:
    payload["online_state_dict"] = online.state_dict()
    payload["target_state_dict"] = target.state_dict()
    payload["optimizer_state_dict"] = optimizer.state_dict()
    payload["replay_state"] = replay.state_dict()
    payload["rng_state"] = rng.bit_generator.state
    atomic_torch_save(path, payload)


def train_episode(
    *,
    mode: str,
    episode: int,
    route: Path,
    online: DuelingQNetwork,
    target: DuelingQNetwork,
    optimizer: torch.optim.Optimizer,
    replay: PrioritizedReplay,
    rng: np.random.Generator,
    payload: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sumo_rl import SumoEnvironment

    episode_seed = args.seed + episode
    set_global_seed(episode_seed)
    env = SumoEnvironment(
        net_file=str(NETWORK_FILE),
        route_file=str(route),
        use_gui=False,
        num_seconds=args.seconds,
        single_agent=False,
        fixed_ts=False,
        sumo_seed=episode_seed,
        reward_fn="queue",
        enforce_max_green=True,
        max_green=60,
        out_csv_name=None,
    )
    records: list[dict[str, Any]] = []
    losses: list[dict[str, float]] = []
    decisions = interventions = agreements = exploratory = 0
    total_scaled_reward = 0.0
    environment_step = int(payload["environment_step"])
    gradient_step = int(payload["gradient_step"])
    learning_step = int(payload["learning_step"])
    epsilon = args.epsilon_start
    try:
        observations = env.reset(seed=episode_seed)
        agent_order = sorted(observations)
        done = {"__all__": False}
        while not done["__all__"]:
            states, masks, experts = build_state_bundle(env, agent_order, args.seconds, args.pressure_gap)
            epsilon = epsilon_at_step(
                learning_step,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay_steps,
            )
            values = batch_q_values(online, states, agent_order, device)
            actions: dict[str, int] = {}
            for agent in observations:
                raw_action = int(np.argmax(values[agent]))
                if rng.random() < epsilon:
                    action = int(rng.choice(np.flatnonzero(masks[agent])))
                    exploratory += 1
                else:
                    action = masked_argmax(values[agent], masks[agent])
                actions[agent] = action
                decisions += 1
                interventions += int(action != raw_action)
                agreements += int(action == experts[agent])
            next_observations, rewards, done, info = env.step(actions)
            environment_step += 1
            team_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
            if done["__all__"]:
                next_states = {agent: np.zeros(STATE_DIM, dtype=np.float32) for agent in actions}
                next_masks = {agent: np.ones(ACTION_COUNT, dtype=np.bool_) for agent in actions}
            else:
                next_states, next_masks, _ = build_state_bundle(
                    env, agent_order, args.seconds, args.pressure_gap
                )
            for agent, action in actions.items():
                local_reward = float(rewards.get(agent, 0.0))
                effective = (1.0 - args.coordination) * local_reward + args.coordination * team_reward
                scaled = float(np.clip(effective / args.reward_scale, -5.0, 1.0))
                total_scaled_reward += scaled
                replay.add(
                    Transition(
                        state=states[agent],
                        action=action,
                        reward=scaled,
                        next_state=next_states[agent],
                        done=bool(done["__all__"]),
                        next_mask=next_masks[agent],
                        expert=False,
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
                        expert_loss_weight=0.0,
                        rng=rng,
                        device=device,
                    )
                )
                gradient_step += 1
                if gradient_step % args.target_update == 0:
                    hard_update(target, online)
            learning_step += len(actions)
            observations = next_observations
            records.append(
                {
                    "step": float(env.sim_step),
                    "reward": team_reward,
                    "scaled_training_reward": total_scaled_reward,
                    "shield_rate": interventions / max(decisions, 1),
                    "expert_agreement_rate": agreements / max(decisions, 1),
                    **info,
                }
            )
    finally:
        env.close()
    payload["environment_step"] = environment_step
    payload["gradient_step"] = gradient_step
    payload["learning_step"] = learning_step
    frame = pd.DataFrame(records)
    summary = summarize_episode(
        frame,
        method=mode,
        seed=episode_seed,
        seconds=args.seconds,
        total_reward=total_scaled_reward,
    )
    summary.update(
        {
            "mode": mode,
            "episode": episode,
            "epsilon": epsilon,
            "replay_size": len(replay),
            "gradient_step": gradient_step,
            "mean_loss": float(np.mean([row["loss"] for row in losses])) if losses else 0.0,
            "shield_rate": interventions / max(decisions, 1),
            "expert_agreement_rate": agreements / max(decisions, 1),
            "exploratory_actions": exploratory,
        }
    )
    return frame, summary


def plot_validation(path: Path, output: Path) -> None:
    if not path.exists():
        return
    frame = pd.read_csv(path)
    grouped = frame.groupby(["mode", "episode"], as_index=False).score.mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, group in grouped.groupby("mode"):
        ax.plot(group.episode, group.score, marker="o", label=mode)
    ax.set_xlabel("Target-domain training episode")
    ax.set_ylabel("Validation score (lower is better)")
    ax.set_title("Transfer learning sample efficiency")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--episodes-per-session", type=int, default=2)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--validation-every", type=int, default=3)
    parser.add_argument("--validation-seconds", type=int, default=900)
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[1251, 1252])
    parser.add_argument("--pressure-gap", type=float, default=2.0)
    parser.add_argument("--coordination", type=float, default=0.25)
    parser.add_argument("--reward-scale", type=float, default=20.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=30_000)
    parser.add_argument("--replay-alpha", type=float, default=0.6)
    parser.add_argument("--beta-start", type=float, default=0.4)
    parser.add_argument("--learning-start", type=int, default=500)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--target-update", type=int, default=750)
    parser.add_argument("--epsilon-start", type=float, default=0.10)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay-steps", type=int, default=12_000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--run-name",
        default="final",
        help="Use 'final' for reportable outputs or a safe label such as 'quick' for an isolated check",
    )
    args = parser.parse_args()

    require_sumo()
    if args.episodes < 1 or args.seconds < 1 or args.validation_seconds < 1:
        raise SystemExit("Episode counts and simulation durations must be positive")
    if args.validation_every < 1:
        raise SystemExit("--validation-every must be positive")
    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        raise SystemExit("Validation seeds must be unique")
    if args.learning_start < args.batch_size:
        raise SystemExit("--learning-start must be at least --batch-size")
    if args.replay_capacity < args.learning_start:
        raise SystemExit("--replay-capacity must be at least --learning-start")
    if not 0.0 <= args.gamma <= 1.0:
        raise SystemExit("--gamma must be in [0, 1]")
    if not 0.0 <= args.epsilon_end <= args.epsilon_start <= 1.0:
        raise SystemExit("Expected 0 <= epsilon-end <= epsilon-start <= 1")
    training_seeds = {args.seed + episode for episode in range(1, args.episodes + 1)}
    if training_seeds & set(args.validation_seeds):
        raise SystemExit("Training and validation seeds overlap")
    if args.episodes_per_session < 0:
        raise SystemExit("--episodes-per-session cannot be negative")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        raise SystemExit("--run-name may contain only letters, digits, '_' and '-'")
    device = resolve_device(args.device)
    experiment_root = TASK_DIR if args.run_name == "final" else TASK_DIR / "runs" / args.run_name
    paths = ensure_output_tree(experiment_root)
    route_manifest = generate(TASK_DIR / "routes")
    route = TASK_DIR / "routes/target_horizontal.rou.xml"
    route_sha = sha256(route)
    source_checkpoint = SEM2_ROOT / "05-multi-intersection/checkpoints/shared_dueling_ddqn_v1_final.pt"
    source_model, source_payload, source_sha = load_task5_model(source_checkpoint, device)
    if not np.isclose(args.pressure_gap, float(source_payload["pressure_gap"])):
        raise SystemExit("Transfer pressure gap must match the source checkpoint")

    training_config_path = paths["results"] / "training_config.json"
    training_config = {
        "run_name": args.run_name,
        "device": str(device),
        "modes": list(MODES),
        "episodes": args.episodes,
        "seconds": args.seconds,
        "seed": args.seed,
        "validation_every": args.validation_every,
        "validation_seconds": args.validation_seconds,
        "validation_seeds": args.validation_seeds,
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
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "source_checkpoint_sha256": source_sha,
        "target_route_sha256": route_sha,
    }
    existing_artifacts = any(paths["checkpoints"].glob("*.pt")) or any(paths["results"].glob("*.csv"))
    if training_config_path.exists():
        previous = json.loads(training_config_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in training_config.items()):
            raise SystemExit(
                "Existing Task 8 training artifacts use a different immutable configuration; "
                "use a new --run-name or archive the old run"
            )
    elif existing_artifacts:
        raise SystemExit("Task 8 artifacts exist without training_config.json; archive them before resuming")
    write_json(training_config_path, training_config | {"status": "in_progress"})

    validation_path = paths["results"] / "validation_history.csv"
    validation_rows = pd.read_csv(validation_path).to_dict("records") if validation_path.exists() else []
    summary_path = paths["results"] / "training_summary.csv"
    summary_rows = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []

    for mode in MODES:
        latest_path = paths["checkpoints"] / f"{mode}_latest.pt"
        best_path = paths["checkpoints"] / f"{mode}_best.pt"
        if latest_path.exists():
            payload = torch.load(latest_path, map_location=device, weights_only=False)
            if payload.get("source_checkpoint_sha256") != source_sha or payload.get("target_route_sha256") != route_sha:
                raise SystemExit(f"{mode} checkpoint belongs to a different source or target route")
        else:
            payload = initialise_mode(
                mode,
                source_model=source_model,
                source_sha256=source_sha,
                args=args,
                route_sha256=route_sha,
                device=device,
            )
        online, target, optimizer, replay, rng = restore_mode(payload, args, device)

        if not any(str(row.get("mode")) == mode and int(row.get("episode", -1)) == 0 for row in validation_rows):
            initial = validate(
                online,
                route=route,
                episode=0,
                mode=mode,
                seconds=args.validation_seconds,
                seeds=args.validation_seeds,
                pressure_gap=args.pressure_gap,
                device=device,
            )
            validation_rows.extend(initial)
            score = average_score(initial)
            payload["best_score"] = score
            payload["best_episode"] = 0
            atomic_torch_save(best_path, payload | {"online_state_dict": copy.deepcopy(online.state_dict())})

        start = int(payload["completed_episode"]) + 1
        session_limit = args.episodes if args.episodes_per_session == 0 else min(
            args.episodes, start + args.episodes_per_session - 1
        )
        for episode in range(start, session_limit + 1):
            frame, row = train_episode(
                mode=mode,
                episode=episode,
                route=route,
                online=online,
                target=target,
                optimizer=optimizer,
                replay=replay,
                rng=rng,
                payload=payload,
                args=args,
                device=device,
            )
            episode_dir = paths["results"] / "training_episodes" / mode
            episode_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(episode_dir / f"episode_{episode:03d}.csv", index=False)
            summary_rows = [
                old for old in summary_rows
                if not (str(old.get("mode")) == mode and int(old.get("episode", -1)) == episode)
            ]
            summary_rows.append(row)
            payload["completed_episode"] = episode
            if episode % args.validation_every == 0 or episode == args.episodes:
                current = validate(
                    online,
                    route=route,
                    episode=episode,
                    mode=mode,
                    seconds=args.validation_seconds,
                    seeds=args.validation_seeds,
                    pressure_gap=args.pressure_gap,
                    device=device,
                )
                validation_rows = [
                    old for old in validation_rows
                    if not (str(old.get("mode")) == mode and int(old.get("episode", -1)) == episode)
                ]
                validation_rows.extend(current)
                score = average_score(current)
                if score < float(payload["best_score"]):
                    payload["best_score"] = score
                    payload["best_episode"] = episode
                    atomic_torch_save(
                        best_path,
                        payload | {"online_state_dict": copy.deepcopy(online.state_dict())},
                    )
            save_payload(
                latest_path,
                payload,
                online=online,
                target=target,
                optimizer=optimizer,
                replay=replay,
                rng=rng,
            )
            pd.DataFrame(summary_rows).sort_values(["mode", "episode"]).to_csv(summary_path, index=False)
            pd.DataFrame(validation_rows).sort_values(["mode", "episode", "seed"]).to_csv(
                validation_path, index=False
            )
            print(
                f"{mode} episode {episode}/{args.episodes}: "
                f"waiting={row['mean_system_total_waiting_time']:.2f}, "
                f"stopped={row['mean_system_total_stopped']:.2f}, loss={row['mean_loss']:.4f}"
            )
        if int(payload["completed_episode"]) >= args.episodes:
            payload["training_complete"] = True
            save_payload(
                latest_path,
                payload,
                online=online,
                target=target,
                optimizer=optimizer,
                replay=replay,
                rng=rng,
            )

    plot_validation(validation_path, paths["plots"] / "sample_efficiency.png")
    complete = {}
    auc = {}
    if validation_path.exists():
        validation_frame = pd.read_csv(validation_path)
        for mode in MODES:
            grouped = validation_frame[validation_frame["mode"] == mode].groupby("episode").score.mean()
            if hasattr(np, "trapezoid"):
                area = np.trapezoid(grouped.to_numpy(), grouped.index.to_numpy())
            else:
                area = np.trapz(grouped.to_numpy(), grouped.index.to_numpy())
            auc[mode] = float(area)
            latest = torch.load(paths["checkpoints"] / f"{mode}_latest.pt", map_location="cpu", weights_only=False)
            complete[mode] = bool(latest.get("training_complete", False))
    write_json(
        paths["results"] / "training_analysis.json",
        {
            "run_name": args.run_name,
            "experiment_root": str(experiment_root.resolve()),
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_checkpoint_sha256": source_sha,
            "target_route": str(route.resolve()),
            "target_route_sha256": route_sha,
            "route_manifest": route_manifest,
            "training_complete": complete,
            "validation_score_auc_lower_is_better": auc,
            "paired_training_seeds": sorted(training_seeds),
            "validation_seeds": args.validation_seeds,
            "limitations": [
                "Fine-tuning inherits the expert-guided Task 5 source policy.",
                "Scratch and fine-tuned models use a pressure action shield during target training.",
                "Twelve episodes measure limited-budget adaptation, not asymptotic performance.",
            ],
        },
    )
    write_json(
        training_config_path,
        training_config | {"status": "complete" if all(complete.values()) else "in_progress"},
    )
    if all(complete.get(mode, False) for mode in MODES):
        print("Task 8 training complete for fine_tuned and scratch. Run evaluate_transfer.py.")
    else:
        print("Session complete. Rerun the identical command to continue both modes.")


if __name__ == "__main__":
    main()
