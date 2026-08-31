"""Reusable, read-only extensions around the completed Task 5 controller.

The helpers in this file never modify Task 5 checkpoints or results.  They
support controlled state ablations, sensor faults, physical lane incidents,
decision tracing and paired bootstrap summaries for Tasks 6--9.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


SEM2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SEM2_ROOT.parent
TASK5_DIR = SEM2_ROOT / "05-multi-intersection"
if str(TASK5_DIR) not in sys.path:
    sys.path.insert(0, str(TASK5_DIR))

from dqn_core import (  # noqa: E402
    ACTION_COUNT,
    EXPERIMENT_TAG,
    STATE_DIM,
    DuelingQNetwork,
    batch_q_values,
    build_state_bundle,
    load_torch_checkpoint,
    masked_argmax,
)


TASK5_NETWORK = {
    "net": REPO_ROOT / "sumo_rl/nets/2x2grid/2x2.net.xml",
    "route": REPO_ROOT / "sumo_rl/nets/2x2grid/2x2.rou.xml",
}

# These slices are part of the Task 5 26-value state contract.
FEATURE_GROUPS: dict[str, tuple[int, ...]] = {
    "phase": tuple(range(0, 4)),
    "minimum_green": (4,),
    "pressure": tuple(range(5, 9)),
    "incoming_queue": tuple(range(9, 13)),
    "outgoing_queue": tuple(range(13, 17)),
    "local_queue": (17,),
    "network_context": tuple(range(18, 21)),
    "simulation_progress": (21,),
    "intersection_identity": tuple(range(22, 26)),
}
SENSOR_INDICES = tuple(range(5, 21))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_task5_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[DuelingQNetwork, dict[str, Any], str]:
    payload = load_torch_checkpoint(checkpoint, device)
    if payload is None:
        raise FileNotFoundError(f"Task 5 checkpoint not found: {checkpoint}")
    if payload.get("experiment_tag") != EXPERIMENT_TAG:
        raise ValueError(f"Incompatible Task 5 checkpoint tag: {payload.get('experiment_tag')}")
    if int(payload.get("state_dim", -1)) != STATE_DIM:
        raise ValueError("Task 5 checkpoint has a different state dimension")
    if int(payload.get("action_count", -1)) != ACTION_COUNT:
        raise ValueError("Task 5 checkpoint has a different action count")
    target = int(payload.get("target_episodes", 0))
    completed = int(payload.get("training_completed_episode", payload.get("completed_episode", 0)))
    if target <= 0 or completed < target:
        raise ValueError(f"Task 5 training is incomplete: {completed}/{target}")
    model = DuelingQNetwork().to(device)
    model.load_state_dict(payload["online_state_dict"])
    model.eval()
    return model, payload, sha256(checkpoint)


def occlude_state_groups(
    states: dict[str, np.ndarray],
    groups: Iterable[str],
) -> dict[str, np.ndarray]:
    indices: list[int] = []
    for group in groups:
        if group not in FEATURE_GROUPS:
            raise KeyError(f"Unknown feature group: {group}")
        indices.extend(FEATURE_GROUPS[group])
    output: dict[str, np.ndarray] = {}
    for agent, state in states.items():
        changed = np.asarray(state, dtype=np.float32).copy()
        changed[indices] = 0.0
        output[agent] = changed
    return output


@dataclass
class ObservationFault:
    """Deterministic state corruption applied only to DQN observations.

    The pressure safety mask remains based on the simulator's true state.  This
    makes raw DQN the pure sensor-fault result and shielded DQN a layered-safety
    result; reports must disclose that distinction.
    """

    kind: str = "none"
    seed: int = 0
    noise_std: float = 0.15
    dropout_probability: float = 0.20
    delay_decisions: int = 2
    rng: np.random.Generator = field(init=False)
    history: dict[str, deque[np.ndarray]] = field(default_factory=dict, init=False)
    corrupted_values: int = field(default=0, init=False)
    total_sensor_values: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        allowed = {"none", "gaussian_noise", "sensor_dropout", "delayed_observation"}
        if self.kind not in allowed:
            raise ValueError(f"Unknown fault kind {self.kind!r}; expected {sorted(allowed)}")
        self.rng = np.random.default_rng(self.seed)

    def apply(self, states: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for agent, state in states.items():
            current = np.asarray(state, dtype=np.float32).copy()
            self.total_sensor_values += len(SENSOR_INDICES)
            if self.kind == "gaussian_noise":
                noise = self.rng.normal(0.0, self.noise_std, len(SENSOR_INDICES)).astype(np.float32)
                current[list(SENSOR_INDICES)] = np.clip(
                    current[list(SENSOR_INDICES)] + noise,
                    -1.0,
                    1.0,
                )
                self.corrupted_values += int(np.count_nonzero(noise))
            elif self.kind == "sensor_dropout":
                drop = self.rng.random(len(SENSOR_INDICES)) < self.dropout_probability
                selected = np.asarray(SENSOR_INDICES, dtype=np.int64)[drop]
                current[selected] = 0.0
                self.corrupted_values += int(drop.sum())
            elif self.kind == "delayed_observation":
                history = self.history.setdefault(agent, deque(maxlen=self.delay_decisions + 1))
                history.append(current.copy())
                delayed = history[0]
                if len(history) > self.delay_decisions:
                    current[list(SENSOR_INDICES)] = delayed[list(SENSOR_INDICES)]
                    self.corrupted_values += len(SENSOR_INDICES)
            output[agent] = current
        return output

    @property
    def corruption_rate(self) -> float:
        return self.corrupted_values / max(self.total_sensor_values, 1)


@dataclass
class LaneIncident:
    start: float = 600.0
    end: float = 1200.0
    lane_id: str = "-h11_0"
    reduced_speed: float = 1.0
    original_speed: float | None = field(default=None, init=False)
    active: bool = field(default=False, init=False)
    applied: bool = field(default=False, init=False)
    restored: bool = field(default=False, init=False)

    def update(self, env: Any) -> None:
        step = float(env.sim_step)
        if not self.applied and step >= self.start:
            lane_ids = set(env.sumo.lane.getIDList())
            if self.lane_id not in lane_ids:
                raise ValueError(f"Incident lane {self.lane_id!r} is absent from the network")
            self.original_speed = float(env.sumo.lane.getMaxSpeed(self.lane_id))
            env.sumo.lane.setMaxSpeed(self.lane_id, self.reduced_speed)
            self.applied = True
            self.active = True
        if self.active and step >= self.end:
            self.restore(env)

    def restore(self, env: Any) -> None:
        if self.applied and not self.restored and self.original_speed is not None:
            env.sumo.lane.setMaxSpeed(self.lane_id, self.original_speed)
            self.restored = True
        self.active = False


def run_extended_episode(
    *,
    seconds: int,
    seed: int,
    controller: str,
    pressure_gap: float,
    model: DuelingQNetwork | None,
    device: torch.device,
    route_file: Path | None = None,
    occluded_groups: Iterable[str] = (),
    observation_fault: ObservationFault | None = None,
    incident: LaneIncident | None = None,
    demand_scale: float = 1.0,
    capture_decisions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one multi-intersection episode and optionally save decision evidence."""
    from sumo_rl import SumoEnvironment

    if controller not in {"fixed", "dqn_raw", "dqn_shielded"}:
        raise ValueError(f"Unsupported controller: {controller}")
    if controller != "fixed" and model is None:
        raise ValueError(f"{controller} requires a model")
    if demand_scale <= 0:
        raise ValueError("demand_scale must be positive")

    extra = None if math.isclose(demand_scale, 1.0) else f"--scale {demand_scale}"
    env = SumoEnvironment(
        net_file=str(TASK5_NETWORK["net"]),
        route_file=str(route_file or TASK5_NETWORK["route"]),
        use_gui=False,
        num_seconds=seconds,
        single_agent=False,
        fixed_ts=controller == "fixed",
        sumo_seed=seed,
        reward_fn="queue",
        enforce_max_green=True,
        max_green=60,
        out_csv_name=None,
        additional_sumo_cmd=extra,
    )
    records: list[dict[str, Any]] = []
    decisions_log: list[dict[str, Any]] = []
    decisions = interventions = agreements = 0
    fault = observation_fault or ObservationFault("none", seed)
    occlusions = tuple(occluded_groups)
    try:
        observations = env.reset(seed=seed)
        agent_order = sorted(observations)
        done = {"__all__": False}
        while not done["__all__"]:
            if incident is not None:
                incident.update(env)
            if controller == "fixed":
                actions: dict[str, int] = {}
            else:
                states, masks, experts = build_state_bundle(
                    env,
                    agent_order,
                    seconds,
                    pressure_gap,
                )
                observed = occlude_state_groups(states, occlusions) if occlusions else states
                observed = fault.apply(observed)
                values = batch_q_values(model, observed, agent_order, device)  # type: ignore[arg-type]
                actions = {}
                for agent in observations:
                    q_values = values[agent]
                    raw_action = int(np.argmax(q_values))
                    executed_action = (
                        raw_action
                        if controller == "dqn_raw"
                        else masked_argmax(q_values, masks[agent])
                    )
                    decisions += 1
                    interventions += int(executed_action != raw_action)
                    agreements += int(executed_action == experts[agent])
                    actions[agent] = executed_action
                    if capture_decisions:
                        row: dict[str, Any] = {
                            "step": float(env.sim_step),
                            "seed": seed,
                            "agent": agent,
                            "controller": controller,
                            "raw_action": raw_action,
                            "executed_action": executed_action,
                            "expert_action": int(experts[agent]),
                            "shield_intervened": int(executed_action != raw_action),
                            "q_margin": float(
                                np.partition(q_values, -2)[-1] - np.partition(q_values, -2)[-2]
                            ),
                        }
                        row.update({f"state_{i}": float(v) for i, v in enumerate(observed[agent])})
                        row.update({f"q_{i}": float(v) for i, v in enumerate(q_values)})
                        row.update({f"mask_{i}": int(v) for i, v in enumerate(masks[agent])})
                        decisions_log.append(row)
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
                    "sensor_corruption_rate": fault.corruption_rate,
                    "incident_active": int(incident.active) if incident is not None else 0,
                    **info,
                }
            )
    finally:
        if incident is not None:
            try:
                incident.restore(env)
            except Exception:
                pass
        env.close()
    return pd.DataFrame(records), pd.DataFrame(decisions_log)


PRIMARY_METRICS: dict[str, bool] = {
    "mean_system_total_waiting_time": False,
    "mean_system_total_stopped": False,
    "mean_system_mean_speed": True,
    "throughput_veh_per_hour": True,
}


def paired_metric_table(
    summary: pd.DataFrame,
    baseline: str,
    candidate: str,
    key_columns: tuple[str, ...] = ("seed",),
) -> pd.DataFrame:
    left = summary[summary.method == baseline]
    right = summary[summary.method == candidate]
    paired = left.merge(right, on=list(key_columns), suffixes=("_baseline", "_candidate"))
    output = paired[list(key_columns)].copy()
    for metric, higher_better in PRIMARY_METRICS.items():
        base = paired[f"{metric}_baseline"].astype(float)
        cand = paired[f"{metric}_candidate"].astype(float)
        signed = cand - base if higher_better else base - cand
        output[f"improvement_pct__{metric}"] = np.where(
            base != 0,
            signed / base.abs() * 100.0,
            np.nan,
        )
        output[f"difference__{metric}"] = cand - base
    return output


def paired_bootstrap(
    paired: pd.DataFrame,
    *,
    resamples: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Bootstrap paired rows and return honest finite-sample uncertainty."""
    if paired.empty:
        raise ValueError("Cannot bootstrap an empty paired table")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(paired), size=(resamples, len(paired)))
    output: dict[str, Any] = {"pairs": len(paired), "resamples": resamples, "metrics": {}}
    for metric in PRIMARY_METRICS:
        column = f"improvement_pct__{metric}"
        values = paired[column].to_numpy(dtype=float)
        means = values[indices].mean(axis=1)
        output["metrics"][metric] = {
            "mean_improvement_pct": float(values.mean()),
            "ci95_low_pct": float(np.quantile(means, 0.025)),
            "ci95_high_pct": float(np.quantile(means, 0.975)),
            "wins": int((values > 0).sum()),
            "trials": len(values),
        }
    return output


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
