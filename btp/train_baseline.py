"""
BTP - Fixed Timing Baseline
============================
Runs SUMO simulation with fixed traffic signal timing (no RL).
This is the baseline to compare all RL agents against.

Usage:
    python btp/train_baseline.py
    python btp/train_baseline.py --network 2way     # 2-way single intersection
    python btp/train_baseline.py --network 4x4      # 4x4 grid (multi-agent)
    python btp/train_baseline.py --seconds 3600     # 1 hour simulation
"""

import os
import sys
import argparse

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment

# ── Network configurations ──────────────────────────────────────────────────
NETWORKS = {
    "single": {
        "net_file": "sumo_rl/nets/single-intersection/single-intersection.net.xml",
        "route_file": "sumo_rl/nets/single-intersection/single-intersection.rou.xml",
        "single_agent": True,
    },
    "2way": {
        "net_file": "sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        "route_file": "sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        "single_agent": True,
    },
    "4x4": {
        "net_file": "sumo_rl/nets/4x4-Lucas/4x4.net.xml",
        "route_file": "sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
        "single_agent": False,
    },
}


def run_baseline(network: str, num_seconds: int, runs: int, use_gui: bool):
    cfg = NETWORKS[network]
    out_csv = f"btp/outputs/baseline/{network}"
    os.makedirs("btp/outputs/baseline", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BASELINE - Fixed Timing | Network: {network.upper()}")
    print(f"  Simulation seconds: {num_seconds} | Runs: {runs}")
    print(f"{'='*60}\n")

    for run in range(1, runs + 1):
        print(f"Run {run}/{runs} ...")
        env = SumoEnvironment(
            net_file=cfg["net_file"],
            route_file=cfg["route_file"],
            out_csv_name=out_csv,
            use_gui=use_gui,
            num_seconds=num_seconds,
            single_agent=cfg["single_agent"],
            fixed_ts=True,  # ← KEY: uses fixed timing, agent has no control
        )

        obs = env.reset()
        done = {"__all__": False} if not cfg["single_agent"] else False
        total_reward = 0

        if cfg["single_agent"]:
            obs, info = obs if isinstance(obs, tuple) else (obs, {})
            terminated, truncated = False, False
            while not (terminated or truncated):
                obs, reward, terminated, truncated, info = env.step(None)
                total_reward += reward
        else:
            while not done["__all__"]:
                _, _, done, info = env.step({})

        env.save_csv(out_csv, run)
        env.close()
        print(f"  ✓ Run {run} complete. Total reward: {total_reward:.2f}")

    print(f"\n✅ Baseline complete! CSV saved to: btp/outputs/baseline/{network}_*\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTP Fixed Timing Baseline")
    parser.add_argument("--network", choices=["single", "2way", "4x4"], default="2way",
                        help="Road network to use (default: 2way)")
    parser.add_argument("--seconds", type=int, default=3600,
                        help="Simulation seconds per run (default: 3600)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs (default: 1)")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Enable SUMO GUI visualisation")
    args = parser.parse_args()

    run_baseline(args.network, args.seconds, args.runs, args.gui)
