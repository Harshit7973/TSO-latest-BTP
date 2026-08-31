"""Shared robust-DQN components for the four-signal Task 5 experiment.

The primary controller is a parameter-sharing Dueling Double DQN.  Every
intersection uses the same network, receives local phase-pressure features and
small network-level congestion summaries, and acts through a max-pressure
safety shield.  The shield is intentionally explicit and measurable.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


EXPERIMENT_TAG = "shared_dueling_ddqn_v1"
ACTION_COUNT = 4
AGENT_COUNT = 4
STATE_DIM = 26
DEFAULT_SAFE_PRESSURE_GAP = 2.0


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    """Write a checkpoint atomically so interrupted laptop sessions can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return torch.load(path, map_location=device, weights_only=False)


def checkpoint_path(checkpoints_dir: Path, kind: str) -> Path:
    return checkpoints_dir / f"{EXPERIMENT_TAG}_{kind}.pt"


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def _served_lanes(env, agent: str, action: int) -> tuple[set[str], set[str]]:
    signal = env.traffic_signals[agent]
    phase_state = signal.green_phases[action].state
    controlled_links = env.sumo.trafficlight.getControlledLinks(agent)
    incoming: set[str] = set()
    outgoing: set[str] = set()
    for index, colour in enumerate(phase_state):
        if colour not in {"G", "g"} or index >= len(controlled_links):
            continue
        for link in controlled_links[index] or ():
            if len(link) >= 2:
                incoming.add(link[0])
                outgoing.add(link[1])
    return incoming, outgoing


@dataclass(frozen=True)
class PhaseStatistics:
    pressure: np.ndarray
    incoming_queue: np.ndarray
    outgoing_queue: np.ndarray
    local_queue: float


def phase_statistics(env, agent: str) -> PhaseStatistics:
    """Measure queue pressure for every action and unique local incoming lanes."""
    signal = env.traffic_signals[agent]
    if signal.num_green_phases != ACTION_COUNT:
        raise ValueError(
            f"Task 5 expects {ACTION_COUNT} green phases, but {agent} has "
            f"{signal.num_green_phases}."
        )
    pressure = np.zeros(ACTION_COUNT, dtype=np.float32)
    incoming_queue = np.zeros(ACTION_COUNT, dtype=np.float32)
    outgoing_queue = np.zeros(ACTION_COUNT, dtype=np.float32)
    all_incoming: set[str] = set()
    for action in range(ACTION_COUNT):
        incoming, outgoing = _served_lanes(env, agent, action)
        all_incoming.update(incoming)
        incoming_queue[action] = sum(
            env.sumo.lane.getLastStepHaltingNumber(lane) for lane in incoming
        )
        outgoing_queue[action] = sum(
            env.sumo.lane.getLastStepHaltingNumber(lane) for lane in outgoing
        )
        pressure[action] = incoming_queue[action] - outgoing_queue[action]
    local_queue = float(
        sum(env.sumo.lane.getLastStepHaltingNumber(lane) for lane in all_incoming)
    )
    return PhaseStatistics(pressure, incoming_queue, outgoing_queue, local_queue)


def max_pressure_action(scores: np.ndarray, current_phase: int) -> int:
    """Choose maximum pressure, retaining the current phase when it is tied."""
    maximum = float(np.max(scores))
    candidates = np.flatnonzero(np.isclose(scores, maximum)).tolist()
    if current_phase in candidates:
        return int(current_phase)
    return int(candidates[0])


def safe_action_mask(
    scores: np.ndarray,
    current_phase: int,
    min_green_satisfied: bool,
    pressure_gap: float,
) -> np.ndarray:
    """Allow only near-max-pressure actions once the signal may change phase."""
    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    if not min_green_satisfied:
        mask[current_phase] = True
        return mask
    mask = scores >= float(np.max(scores)) - pressure_gap
    if not bool(mask.any()):
        mask[max_pressure_action(scores, current_phase)] = True
    return mask


def build_state_bundle(
    env,
    agent_order: list[str],
    seconds: int,
    pressure_gap: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    """Build shared-policy states, safety masks and expert actions for all agents."""
    if len(agent_order) != AGENT_COUNT:
        raise ValueError(f"Task 5 expects {AGENT_COUNT} agents, found {agent_order}")
    statistics = {agent: phase_statistics(env, agent) for agent in agent_order}
    queues = np.asarray([statistics[agent].local_queue for agent in agent_order], dtype=np.float32)
    network_mean = float(np.mean(queues))
    network_max = float(np.max(queues))
    network_std = float(np.std(queues))
    progress = float(np.clip(float(env.sim_step) / max(seconds, 1), 0.0, 1.0))

    states: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    experts: dict[str, int] = {}
    for agent_index, agent in enumerate(agent_order):
        signal = env.traffic_signals[agent]
        current_phase = int(signal.green_phase)
        min_green_satisfied = bool(
            signal.time_since_last_phase_change >= signal.min_green + signal.yellow_time
        )
        stats = statistics[agent]
        phase_one_hot = np.eye(ACTION_COUNT, dtype=np.float32)[current_phase]
        identity = np.eye(AGENT_COUNT, dtype=np.float32)[agent_index]
        state = np.concatenate(
            [
                phase_one_hot,
                np.asarray([float(min_green_satisfied)], dtype=np.float32),
                np.tanh(stats.pressure / 10.0),
                np.tanh(stats.incoming_queue / 10.0),
                np.tanh(stats.outgoing_queue / 10.0),
                np.asarray(
                    [
                        math.tanh(stats.local_queue / 20.0),
                        math.tanh(network_mean / 20.0),
                        math.tanh(network_max / 20.0),
                        math.tanh(network_std / 20.0),
                        progress,
                    ],
                    dtype=np.float32,
                ),
                identity,
            ]
        ).astype(np.float32)
        if state.shape != (STATE_DIM,):
            raise RuntimeError(f"Unexpected state shape {state.shape}; expected {(STATE_DIM,)}")
        states[agent] = state
        masks[agent] = safe_action_mask(
            stats.pressure,
            current_phase,
            min_green_satisfied,
            pressure_gap,
        )
        experts[agent] = (
            current_phase
            if not min_green_satisfied
            else max_pressure_action(stats.pressure, current_phase)
        )
    return states, masks, experts


class DuelingQNetwork(nn.Module):
    """Small shared Dueling DQN suitable for the RTX 3050 and CPU inference."""

    def __init__(self, state_dim: int = STATE_DIM, action_count: int = ACTION_COUNT) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_count),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(states)
        value = self.value(encoded)
        advantage = self.advantage(encoded)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray
    expert: bool


class PrioritizedReplay:
    """Proportional prioritized replay with a serializable circular buffer."""

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.storage: list[Transition] = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.next_index = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: Transition) -> None:
        maximum = float(self.priorities[: len(self.storage)].max()) if self.storage else 1.0
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.next_index] = transition
        self.priorities[self.next_index] = max(maximum, 1e-4)
        self.next_index = (self.next_index + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        beta: float,
        rng: np.random.Generator,
    ) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        count = len(self.storage)
        if count < batch_size:
            raise ValueError(f"Replay has {count} items, fewer than batch size {batch_size}")
        scaled = np.power(self.priorities[:count].clip(min=1e-6), self.alpha)
        probabilities = scaled / scaled.sum()
        indices = rng.choice(count, size=batch_size, replace=False, p=probabilities)
        weights = np.power(count * probabilities[indices], -beta)
        weights /= weights.max()
        return [self.storage[int(index)] for index in indices], indices, weights.astype(np.float32)

    def update_priorities(self, indices: Iterable[int], priorities: Iterable[float]) -> None:
        for index, priority in zip(indices, priorities):
            self.priorities[int(index)] = max(float(priority), 1e-4)

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "storage": self.storage,
            "priorities": self.priorities,
            "next_index": self.next_index,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "PrioritizedReplay":
        replay = cls(int(state["capacity"]), float(state["alpha"]))
        replay.storage = state["storage"]
        replay.priorities = state["priorities"]
        replay.next_index = int(state["next_index"])
        return replay


def masked_argmax(q_values: np.ndarray, mask: np.ndarray) -> int:
    masked = np.where(mask, q_values, -np.inf)
    if not np.isfinite(masked).any():
        raise ValueError("Action mask contains no valid action")
    return int(np.argmax(masked))


@torch.no_grad()
def q_values(model: DuelingQNetwork, state: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    return model(tensor).squeeze(0).detach().cpu().numpy()


@torch.no_grad()
def batch_q_values(
    model: DuelingQNetwork,
    states: dict[str, np.ndarray],
    agent_order: list[str],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Evaluate all four agents in one small GPU/CPU batch."""
    tensor = torch.as_tensor(
        np.stack([states[agent] for agent in agent_order]),
        dtype=torch.float32,
        device=device,
    )
    values = model(tensor).detach().cpu().numpy()
    return {agent: values[index] for index, agent in enumerate(agent_order)}


def optimize_double_dqn(
    *,
    online: DuelingQNetwork,
    target: DuelingQNetwork,
    optimizer: torch.optim.Optimizer,
    replay: PrioritizedReplay,
    batch_size: int,
    gamma: float,
    beta: float,
    expert_loss_weight: float,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, float]:
    transitions, indices, weights = replay.sample(batch_size, beta, rng)
    states = torch.as_tensor(
        np.stack([item.state for item in transitions]), dtype=torch.float32, device=device
    )
    actions = torch.as_tensor(
        [item.action for item in transitions], dtype=torch.long, device=device
    )
    rewards = torch.as_tensor(
        [item.reward for item in transitions], dtype=torch.float32, device=device
    )
    next_states = torch.as_tensor(
        np.stack([item.next_state for item in transitions]), dtype=torch.float32, device=device
    )
    dones = torch.as_tensor(
        [item.done for item in transitions], dtype=torch.float32, device=device
    )
    next_masks = torch.as_tensor(
        np.stack([item.next_mask for item in transitions]), dtype=torch.bool, device=device
    )
    importance = torch.as_tensor(weights, dtype=torch.float32, device=device)
    expert_flags = torch.as_tensor(
        [item.expert for item in transitions], dtype=torch.bool, device=device
    )

    all_q = online(states)
    chosen_q = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        online_next = online(next_states).masked_fill(~next_masks, -1e9)
        next_actions = online_next.argmax(dim=1)
        target_next = target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards + gamma * (1.0 - dones) * target_next

    td_errors = targets - chosen_q
    td_loss = (
        F.smooth_l1_loss(chosen_q, targets, reduction="none") * importance
    ).mean()
    if bool(expert_flags.any()):
        imitation_loss = F.cross_entropy(all_q[expert_flags], actions[expert_flags])
    else:
        imitation_loss = torch.zeros((), device=device)
    loss = td_loss + expert_loss_weight * imitation_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
    optimizer.step()
    replay.update_priorities(indices, np.abs(td_errors.detach().cpu().numpy()) + 1e-4)
    return {
        "loss": float(loss.item()),
        "td_loss": float(td_loss.item()),
        "imitation_loss": float(imitation_loss.item()),
        "mean_abs_td_error": float(td_errors.abs().mean().item()),
    }


def hard_update(target: DuelingQNetwork, online: DuelingQNetwork) -> None:
    target.load_state_dict(online.state_dict())


def epsilon_at_step(
    learning_step: int,
    start: float,
    end: float,
    decay_steps: int,
) -> float:
    fraction = min(max(learning_step, 0) / max(decay_steps, 1), 1.0)
    return float(start + fraction * (end - start))


def beta_at_step(gradient_step: int, start: float, anneal_steps: int) -> float:
    fraction = min(max(gradient_step, 0) / max(anneal_steps, 1), 1.0)
    return float(start + fraction * (1.0 - start))


def run_controller_episode(
    *,
    network_config: dict[str, Path],
    seconds: int,
    seed: int,
    controller: str,
    pressure_gap: float,
    model: DuelingQNetwork | None,
    device: torch.device,
    deployment_policy: str | None = None,
) -> pd.DataFrame:
    """Run a deterministic evaluation episode for one controller."""
    from sumo_rl import SumoEnvironment

    effective_controller = controller
    if controller == "deployed":
        if deployment_policy not in {"dqn_shielded", "max_pressure"}:
            raise ValueError(f"Invalid deployment policy: {deployment_policy}")
        effective_controller = deployment_policy
    if effective_controller not in {"fixed", "max_pressure", "dqn_raw", "dqn_shielded"}:
        raise ValueError(f"Unknown controller: {effective_controller}")
    fixed = effective_controller == "fixed"
    if effective_controller in {"dqn_raw", "dqn_shielded"} and model is None:
        raise ValueError(f"{effective_controller} requires a DQN model")

    env = SumoEnvironment(
        net_file=str(network_config["net"]),
        route_file=str(network_config["route"]),
        use_gui=False,
        num_seconds=seconds,
        single_agent=False,
        fixed_ts=fixed,
        sumo_seed=seed,
        reward_fn="queue",
        enforce_max_green=True,
        max_green=60,
        out_csv_name=None,
    )
    records: list[dict[str, Any]] = []
    decisions = 0
    interventions = 0
    agreements = 0
    try:
        observations = env.reset(seed=seed)
        agent_order = sorted(observations)
        done = {"__all__": False}
        while not done["__all__"]:
            if fixed:
                actions: dict[str, int] = {}
            else:
                states, masks, experts = build_state_bundle(
                    env,
                    agent_order,
                    seconds,
                    pressure_gap,
                )
                actions = {}
                network_values = (
                    batch_q_values(model, states, agent_order, device)  # type: ignore[arg-type]
                    if effective_controller in {"dqn_raw", "dqn_shielded"}
                    else {}
                )
                for agent in observations:
                    expert_action = experts[agent]
                    if effective_controller == "max_pressure":
                        action = expert_action
                    else:
                        raw_values = network_values[agent]
                        raw_action = int(np.argmax(raw_values))
                        action = (
                            raw_action
                            if effective_controller == "dqn_raw"
                            else masked_argmax(raw_values, masks[agent])
                        )
                        interventions += int(action != raw_action)
                    agreements += int(action == expert_action)
                    decisions += 1
                    actions[agent] = action
            observations, rewards, done, info = env.step(actions)
            records.append(
                {
                    "step": float(env.sim_step),
                    "reward": float(np.mean(list(rewards.values()))) if rewards else 0.0,
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
    return pd.DataFrame(records)
