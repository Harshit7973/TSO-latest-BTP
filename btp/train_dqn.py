"""
BTP - DQN Training (GPU Accelerated)
======================================
Trains a Deep Q-Network (DQN) agent using Stable-Baselines3.
Automatically uses your RTX 3050 GPU via CUDA.
Saves the trained model as a .zip file for later testing.

Usage:
    python btp/train_dqn.py
    python btp/train_dqn.py --network 2way --timesteps 200000
    python btp/train_dqn.py --network 4x4 --timesteps 500000
    python btp/train_dqn.py --gui    # Watch training in SUMO GUI (slower)
"""

import os
import sys
import argparse

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
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


def train_dqn(network, num_seconds, total_timesteps, use_gui):
    cfg = NETWORKS[network]
    os.makedirs("btp/outputs/dqn", exist_ok=True)
    os.makedirs("btp/models", exist_ok=True)
    os.makedirs("btp/models/dqn_checkpoints", exist_ok=True)
    out_csv = f"btp/outputs/dqn/{network}"
    model_path = f"btp/models/dqn_{network}"

    # ── Device detection ───────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n🚀 GPU detected: {gpu_name} — Training will be fast!")
    else:
        print("\n⚠️  No GPU detected, using CPU. Training will be slower.")

    print(f"\n{'='*60}")
    print(f"  DQN TRAINING | Network: {network.upper()} | Device: {device.upper()}")
    print(f"  Total timesteps: {total_timesteps:,} | Sim seconds: {num_seconds}")
    print(f"{'='*60}\n")

    # ── Create Environment ─────────────────────────────────────────────────
    env = SumoEnvironment(
        net_file=cfg["net_file"],
        route_file=cfg["route_file"],
        out_csv_name=out_csv,
        use_gui=use_gui,
        num_seconds=num_seconds,
        single_agent=True,  # DQN works in single-agent mode
    )
    env = Monitor(env, filename=f"btp/outputs/dqn/{network}_monitor")

    # ── DQN Model ─────────────────────────────────────────────────────────
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.15,           # Explore for 15% of training
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),  # 2-layer neural network
        verbose=1,
        device=device,                        # ← Uses RTX 3050 GPU
        tensorboard_log="btp/outputs/dqn/tensorboard/",
    )

    # ── Callbacks: Save checkpoints every 10k steps ────────────────────────
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="btp/models/dqn_checkpoints/",
        name_prefix=f"dqn_{network}",
        verbose=1,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print("⏳ Training started...\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # ── Save final model ───────────────────────────────────────────────────
    model.save(model_path)
    env.close()

    print(f"\n✅ Training complete!")
    print(f"✅ Model saved to: {model_path}.zip")
    print(f"✅ TensorBoard logs: btp/outputs/dqn/tensorboard/")
    print(f"\n   To view TensorBoard run:")
    print(f"   .\\venv\\Scripts\\python -m tensorboard.main --logdir btp/outputs/dqn/tensorboard/")
    print(f"\n   To test the model run:")
    print(f"   .\\venv\\Scripts\\python btp/test_model.py --model {model_path}.zip --network {network}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTP DQN Training")
    parser.add_argument("--network", choices=["single", "2way", "4x4"], default="2way")
    parser.add_argument("--seconds", type=int, default=3600,
                        help="Simulation seconds per episode (default: 3600)")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps (default: 100000)")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Enable SUMO GUI (much slower, use for debugging only)")
    args = parser.parse_args()

    train_dqn(args.network, args.seconds, args.timesteps, args.gui)
