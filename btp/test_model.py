"""
BTP - Model Testing & Evaluation
==================================
Loads a trained DQN model and runs it in the SUMO environment.
Prints evaluation metrics: waiting time, queue, speed, throughput.

Usage:
    # Test DQN model with GUI
    python btp/test_model.py --model btp/models/dqn_2way.zip --network 2way --gui

    # Test DQN model without GUI (faster, just prints metrics)
    python btp/test_model.py --model btp/models/dqn_2way.zip --network 2way

    # Test on 4x4 grid
    python btp/test_model.py --model btp/models/dqn_4x4.zip --network 4x4
"""

import os
import sys
import argparse

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
from stable_baselines3 import DQN
from sumo_rl import SumoEnvironment

NETWORKS = {
    "single": {
        "net_file": "sumo_rl/nets/single-intersection/single-intersection.net.xml",
        "route_file": "sumo_rl/nets/single-intersection/single-intersection.rou.xml",
    },
    "2way": {
        "net_file": "sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        "route_file": "sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
    },
    "4x4": {
        "net_file": "sumo_rl/nets/4x4-grid/4x4.net.xml",
        "route_file": "sumo_rl/nets/4x4-grid/4x4c1c2c1c2.rou.xml",
    },
}


def test_model(model_path, network, num_seconds, use_gui):
    cfg = NETWORKS[network]
    os.makedirs("btp/outputs/test", exist_ok=True)
    out_csv = f"btp/outputs/test/{network}_dqn_test"

    print(f"\n{'='*60}")
    print(f"  TESTING MODEL | Network: {network.upper()}")
    print(f"  Model: {model_path}")
    print(f"{'='*60}\n")

    # ── Load Model ────────────────────────────────────────────────────────
    print(f"📂 Loading model from: {model_path}")
    model = DQN.load(model_path)
    print("✓ Model loaded successfully!\n")

    # ── Create Environment ────────────────────────────────────────────────
    env = SumoEnvironment(
        net_file=cfg["net_file"],
        route_file=cfg["route_file"],
        out_csv_name=out_csv,
        use_gui=use_gui,
        num_seconds=num_seconds,
        single_agent=True,
    )

    # ── Run Evaluation ────────────────────────────────────────────────────
    obs, info = env.reset()
    terminated, truncated = False, False

    total_reward = 0
    step = 0
    waiting_times = []
    queue_lengths = []
    speeds = []

    print("🚦 Running simulation..." + (" (SUMO GUI open)" if use_gui else ""))
    print("-" * 60)

    while not (terminated or truncated):
        # Deterministic=True → always picks the best action (no exploration)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step += 1

        # Collect metrics every step
        waiting_times.append(info.get("system_total_waiting_time", 0))
        queue_lengths.append(info.get("system_total_stopped", 0))
        speeds.append(info.get("system_mean_speed", 0))

        if step % 100 == 0:
            print(f"  Step {step:>6} | Reward: {reward:>8.2f} | "
                  f"Waiting: {info.get('system_total_waiting_time', 0):>8.1f}s | "
                  f"Queue: {info.get('system_total_stopped', 0):>4} veh")

    env.save_csv(out_csv, 1)
    env.close()

    # ── Print Summary ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  📊 EVALUATION RESULTS SUMMARY")
    print("="*60)
    print(f"  Total Steps          : {step:,}")
    print(f"  Total Reward         : {total_reward:.2f}")
    print(f"  Avg Waiting Time     : {np.mean(waiting_times):.2f} s")
    print(f"  Max Waiting Time     : {np.max(waiting_times):.2f} s")
    print(f"  Avg Queue Length     : {np.mean(queue_lengths):.2f} vehicles")
    print(f"  Max Queue Length     : {np.max(queue_lengths):.0f} vehicles")
    print(f"  Avg Speed            : {np.mean(speeds):.4f} m/s")
    print("="*60)
    print(f"  ✅ CSV saved to: {out_csv}_conn0_ep1.csv\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTP Model Testing")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved .zip model (e.g. btp/models/dqn_2way.zip)")
    parser.add_argument("--network", choices=["single", "2way", "4x4"], default="2way")
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Open SUMO GUI to visualise the agent controlling traffic")
    args = parser.parse_args()

    test_model(args.model, args.network, args.seconds, args.gui)
