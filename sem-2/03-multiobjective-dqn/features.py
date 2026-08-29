"""Enhanced observation and multi-objective reward used only by DQN-v2."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces


class EnhancedObservation:
    """Phase, timing, incoming density/queue/wait and outgoing density."""

    def __init__(self, ts):
        self.ts = ts

    def __call__(self) -> np.ndarray:
        phase = [float(self.ts.green_phase == index) for index in range(self.ts.num_green_phases)]
        min_green = [float(self.ts.time_since_last_phase_change >= self.ts.min_green + self.ts.yellow_time)]
        elapsed = [min(self.ts.time_since_last_phase_change / max(self.ts.max_green, 1), 1.0)]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        # Read current waiting directly to keep observation computation free of
        # the bookkeeping side effects used by the reward implementation.
        lane_wait = []
        for lane in self.ts.lanes:
            vehicles = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            current_wait = sum(self.ts.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles)
            lane_wait.append(min(current_wait / 300.0, 1.0))
        # Core SUMO-RL stores outgoing lanes through a set; sorting here keeps
        # feature positions stable across Python processes and resume sessions.
        outgoing_density = []
        for lane in sorted(self.ts.out_lanes):
            capacity = self.ts.lanes_length[lane] / (
                self.ts.MIN_GAP + self.ts.sumo.lane.getLastStepLength(lane)
            )
            outgoing_density.append(min(self.ts.sumo.lane.getLastStepVehicleNumber(lane) / capacity, 1.0))
        return np.asarray(phase + min_green + elapsed + density + queue + lane_wait + outgoing_density, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        size = self.ts.num_green_phases + 2 + 3 * len(self.ts.lanes) + len(self.ts.out_lanes)
        return spaces.Box(low=np.zeros(size, dtype=np.float32), high=np.ones(size, dtype=np.float32))


def multi_objective_reward(ts) -> float:
    """Balance delay reduction, queues, emissions, fairness and switching."""
    lane_waits = ts.get_accumulated_waiting_time_per_lane()
    scaled_wait = sum(lane_waits) / 100.0
    waiting_improvement = ts.last_ts_waiting_time - scaled_wait
    ts.last_ts_waiting_time = scaled_wait

    queue_penalty = ts.get_total_queued() / 20.0
    co2_penalty = min(ts.get_total_co2() / 100_000.0, 5.0)
    fairness_penalty = min(max(lane_waits, default=0.0) / 300.0, 5.0)
    switch_penalty = float(ts.is_yellow)
    return float(
        waiting_improvement
        - 0.15 * queue_penalty
        - 0.02 * co2_penalty
        - 0.05 * fairness_penalty
        - 0.03 * switch_penalty
    )
