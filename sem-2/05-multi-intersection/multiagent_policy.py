"""Compact pressure-aware policy helpers for Task 5 compact-v2.

The original Task 5 used SUMO-RL's lane-by-lane 10-bin encoding.  That created
too many exact tabular states and produced 19-69% unseen decisions on held-out
seeds.  This module reduces the state to phase-level pressure bins and provides
a deterministic max-pressure fallback when a genuinely new state is observed.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT_TAG = "compact_v2"
PRESSURE_BIN_EDGES = (0.0, 2.0, 5.0, 10.0)
PRESSURE_OVERRIDE_GAP = 5.0


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a task-owned checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def checkpoint_path(checkpoints_dir: Path, mode: str, network: str, kind: str) -> Path:
    return checkpoints_dir / f"{mode}_{network}_{EXPERIMENT_TAG}_{kind}.pkl"


def _served_lanes(env, agent: str, action: int) -> tuple[set[str], set[str]]:
    """Return unique incoming/outgoing lanes receiving green for an action."""
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


def phase_pressure_scores(env, agent: str) -> np.ndarray:
    """Queue pressure for every possible green phase: incoming minus outgoing."""
    signal = env.traffic_signals[agent]
    scores = np.zeros(signal.num_green_phases, dtype=np.float32)
    for action in range(signal.num_green_phases):
        incoming, outgoing = _served_lanes(env, agent, action)
        incoming_queue = sum(env.sumo.lane.getLastStepHaltingNumber(lane) for lane in incoming)
        outgoing_queue = sum(env.sumo.lane.getLastStepHaltingNumber(lane) for lane in outgoing)
        scores[action] = float(incoming_queue - outgoing_queue)
    return scores


def pressure_bin(value: float) -> int:
    """Map pressure to one of five robust bins instead of ten bins per lane."""
    if value <= PRESSURE_BIN_EDGES[0]:
        return 0
    if value <= PRESSURE_BIN_EDGES[1]:
        return 1
    if value <= PRESSURE_BIN_EDGES[2]:
        return 2
    if value <= PRESSURE_BIN_EDGES[3]:
        return 3
    return 4


def compact_state(env, agent: str) -> tuple[int, ...]:
    """Encode current phase, timing eligibility and four phase pressures.

    For a four-action signal the theoretical state count is only
    ``4 * 2 * 5**4 = 5,000`` instead of a product of ten bins for every lane
    density and queue feature.
    """
    signal = env.traffic_signals[agent]
    min_green_satisfied = int(
        signal.time_since_last_phase_change >= signal.min_green + signal.yellow_time
    )
    bins = [pressure_bin(float(value)) for value in phase_pressure_scores(env, agent)]
    return tuple([int(signal.green_phase), min_green_satisfied, *bins])


def max_pressure_action(
    env,
    agent: str,
    pressure_scores: np.ndarray | None = None,
) -> int:
    """Select the largest-pressure phase, retaining current phase on a tie."""
    scores = phase_pressure_scores(env, agent) if pressure_scores is None else pressure_scores
    signal = env.traffic_signals[agent]
    maximum = float(np.max(scores))
    candidates = np.flatnonzero(np.isclose(scores, maximum)).tolist()
    current = int(signal.green_phase)
    if current in candidates:
        return current
    return int(candidates[0])


def pressure_guarded_action(
    env,
    agent: str,
    values: np.ndarray,
    pressure_scores: np.ndarray | None = None,
) -> tuple[int, bool]:
    """Apply the learned action unless it neglects a much higher-pressure phase."""
    scores = phase_pressure_scores(env, agent) if pressure_scores is None else pressure_scores
    learned_action = int(np.argmax(values))
    pressure_action = max_pressure_action(env, agent, scores)
    pressure_gap = float(scores[pressure_action] - scores[learned_action])
    if pressure_gap >= PRESSURE_OVERRIDE_GAP:
        return pressure_action, True
    return learned_action, False


def q_values(
    tables: dict[str, dict[tuple[int, ...], np.ndarray]],
    agent: str,
    state: tuple[int, ...],
    action_count: int,
    pressure_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Get Q-values and initialise new training states with a small pressure prior."""
    table = tables.setdefault(agent, {})
    if state not in table:
        values = np.zeros(action_count, dtype=np.float32)
        if pressure_scores is not None and len(pressure_scores) == action_count:
            shifted = pressure_scores.astype(np.float32) - float(np.min(pressure_scores))
            scale = float(np.max(shifted))
            if scale > 0:
                values = 0.05 * shifted / scale
        table[state] = values
    return table[state]


def deterministic_action(
    env,
    agent: str,
    tables: dict[str, dict[tuple[int, ...], np.ndarray]],
) -> tuple[int, bool, bool]:
    """Use learned Q-values with pressure fallback and a starvation guard."""
    state = compact_state(env, agent)
    values = tables.get(agent, {}).get(state)
    if values is None:
        return max_pressure_action(env, agent), True, False
    action, used_override = pressure_guarded_action(env, agent, values)
    return action, False, used_override


def run_multiagent_episode(
    *,
    network_config: dict[str, Path],
    seconds: int,
    seed: int,
    fixed: bool,
    tables: dict[str, dict[tuple[int, ...], np.ndarray]] | None = None,
    reward_fn: str = "queue",
) -> pd.DataFrame:
    """Run one deterministic fixed or compact-v2 evaluation episode."""
    from sumo_rl import SumoEnvironment

    env = SumoEnvironment(
        net_file=str(network_config["net"]),
        route_file=str(network_config["route"]),
        use_gui=False,
        num_seconds=seconds,
        single_agent=False,
        fixed_ts=fixed,
        sumo_seed=seed,
        reward_fn=reward_fn,
        enforce_max_green=True,
        max_green=60,
        out_csv_name=None,
    )
    records: list[dict[str, Any]] = []
    fallback_actions = 0
    override_actions = 0
    agent_decisions = 0
    try:
        observations = env.reset(seed=seed)
        done = {"__all__": False}
        while not done["__all__"]:
            if fixed:
                actions = {}
            else:
                actions = {}
                for agent in observations:
                    action, used_fallback, used_override = deterministic_action(env, agent, tables or {})
                    actions[agent] = action
                    fallback_actions += int(used_fallback)
                    override_actions += int(used_override)
                    agent_decisions += 1
            observations, rewards, done, info = env.step(actions)
            records.append(
                {
                    "step": float(env.sim_step),
                    "reward": float(np.mean(list(rewards.values()))) if rewards else 0.0,
                    "pressure_fallback_actions": fallback_actions,
                    "pressure_override_actions": override_actions,
                    "total_agent_decisions": agent_decisions,
                    "fallback_rate": fallback_actions / max(agent_decisions, 1),
                    "override_rate": override_actions / max(agent_decisions, 1),
                    **info,
                }
            )
    finally:
        env.close()
    return pd.DataFrame(records)
