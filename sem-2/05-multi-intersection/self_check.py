"""Fast offline checks for Task 5 state, shield, replay and DQN updates."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from dqn_core import (
    ACTION_COUNT,
    STATE_DIM,
    DuelingQNetwork,
    PrioritizedReplay,
    Transition,
    build_state_bundle,
    hard_update,
    masked_argmax,
    optimize_double_dqn,
    safe_action_mask,
)


TASK_DIR = Path(__file__).resolve().parent


class _Phase:
    def __init__(self, state: str) -> None:
        self.state = state


class _Signal:
    num_green_phases = 4
    green_phases = [_Phase("Grrr"), _Phase("rGrr"), _Phase("rrGr"), _Phase("rrrG")]
    green_phase = 1
    time_since_last_phase_change = 8
    min_green = 5
    yellow_time = 2


class _Lane:
    def getLastStepHaltingNumber(self, lane: str) -> int:
        values = {"in0": 12, "out0": 1, "in1": 6, "out1": 1, "in2": 3, "out2": 2, "in3": 0, "out3": 4}
        return values[lane.split("__", maxsplit=1)[1]]


class _TrafficLight:
    def getControlledLinks(self, agent: str):
        return [
            [(f"{agent}__in0", f"{agent}__out0", "")],
            [(f"{agent}__in1", f"{agent}__out1", "")],
            [(f"{agent}__in2", f"{agent}__out2", "")],
            [(f"{agent}__in3", f"{agent}__out3", "")],
        ]


class _Sumo:
    lane = _Lane()
    trafficlight = _TrafficLight()


class _Environment:
    sim_step = 100
    traffic_signals = {agent: _Signal() for agent in ["1", "2", "5", "6"]}
    sumo = _Sumo()


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    checks: dict[str, bool] = {}

    scores = np.asarray([10.0, 9.0, 4.0, -1.0], dtype=np.float32)
    eligible = safe_action_mask(scores, current_phase=1, min_green_satisfied=True, pressure_gap=2.0)
    locked = safe_action_mask(scores, current_phase=1, min_green_satisfied=False, pressure_gap=2.0)
    checks["eligible_mask"] = eligible.tolist() == [True, True, False, False]
    checks["minimum_green_lock"] = locked.tolist() == [False, True, False, False]
    checks["masked_argmax"] = masked_argmax(np.asarray([1.0, 2.0, 99.0, 100.0]), eligible) == 1

    env = _Environment()
    order = ["1", "2", "5", "6"]
    states, masks, experts = build_state_bundle(env, order, seconds=1800, pressure_gap=2.0)
    checks["four_agents"] = set(states) == set(order)
    checks["state_shape"] = all(state.shape == (STATE_DIM,) for state in states.values())
    checks["finite_state"] = all(bool(np.isfinite(state).all()) for state in states.values())
    checks["valid_masks"] = all(bool(mask.any()) and len(mask) == ACTION_COUNT for mask in masks.values())
    checks["expert_actions"] = all(action == 0 for action in experts.values())

    device = torch.device("cpu")
    online = DuelingQNetwork().to(device)
    target = DuelingQNetwork().to(device)
    hard_update(target, online)
    optimizer = torch.optim.AdamW(online.parameters(), lr=3e-4)
    replay = PrioritizedReplay(capacity=512)
    rng = np.random.default_rng(42)
    for index in range(256):
        state = rng.normal(size=STATE_DIM).astype(np.float32)
        next_state = rng.normal(size=STATE_DIM).astype(np.float32)
        replay.add(
            Transition(
                state=state,
                action=index % ACTION_COUNT,
                reward=float(-index % 7) / 10.0,
                next_state=next_state,
                done=index % 31 == 0,
                next_mask=np.ones(ACTION_COUNT, dtype=np.bool_),
                expert=index < 128,
            )
        )
    before = [parameter.detach().clone() for parameter in online.parameters()]
    metrics = optimize_double_dqn(
        online=online,
        target=target,
        optimizer=optimizer,
        replay=replay,
        batch_size=64,
        gamma=0.99,
        beta=0.4,
        expert_loss_weight=0.15,
        rng=rng,
        device=device,
    )
    checks["finite_loss"] = all(math.isfinite(value) for value in metrics.values())
    checks["parameters_updated"] = any(
        not torch.equal(old, new.detach()) for old, new in zip(before, online.parameters())
    )
    checks["replay_round_trip"] = len(PrioritizedReplay.from_state_dict(replay.state_dict())) == len(replay)

    passed = all(checks.values())
    output = {"passed": passed, "checks": checks, "optimizer_metrics": metrics}
    results_dir = TASK_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "self_check.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not passed:
        raise SystemExit("Task 5 self-check failed")


if __name__ == "__main__":
    main()
