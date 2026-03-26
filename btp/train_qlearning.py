"""
BTP - Q-Learning Training
==========================
Trains a tabular Q-learning agent for traffic signal control.
Saves the Q-table so it can be loaded and tested later.

Usage:
    python btp/train_qlearning.py
    python btp/train_qlearning.py --network 2way --runs 5
    python btp/train_qlearning.py --gui   # Watch training in SUMO GUI
"""

import os
import sys
import argparse
import pickle

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

NETWORKS = {
    "single": {
        "net_file": "sumo_rl/nets/single-intersection/single-intersection.net.xml",
        "route_file": "sumo_rl/nets/single-intersection/single-intersection.rou.xml",
    },
    "2way": {
        "net_file": "sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        "route_file": "sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
    },
}


def train_qlearning(network, num_seconds, runs, alpha, gamma, epsilon, use_gui):
    cfg = NETWORKS[network]
    os.makedirs("btp/outputs/qlearning", exist_ok=True)
    os.makedirs("btp/models", exist_ok=True)
    out_csv = f"btp/outputs/qlearning/{network}"
    model_path = f"btp/models/qtable_{network}.pkl"

    print(f"\n{'='*60}")
    print(f"  Q-LEARNING | Network: {network.upper()}")
    print(f"  Alpha={alpha} | Gamma={gamma} | Epsilon={epsilon}")
    print(f"  Simulation seconds: {num_seconds} | Runs: {runs}")
    print(f"{'='*60}\n")

    env = SumoEnvironment(
        net_file=cfg["net_file"],
        route_file=cfg["route_file"],
        out_csv_name=out_csv,
        use_gui=use_gui,
        num_seconds=num_seconds,
        single_agent=False,  # Q-learning handles multi-ts dict
    )

    # Initialise Q-agents (one per traffic signal — usually just 1 for single/2way)
    initial_obs = env.reset()
    ql_agents = {
        ts: QLAgent(
            starting_state=env.encode(initial_obs[ts], ts),
            state_space=env.observation_space,
            action_space=env.action_space,
            alpha=alpha,
            gamma=gamma,
            exploration_strategy=EpsilonGreedy(
                initial_epsilon=epsilon,
                min_epsilon=0.005,
                decay=1.0,
            ),
        )
        for ts in env.ts_ids
    }

    for run in range(1, runs + 1):
        if run > 1:
            obs = env.reset()
            for ts in env.ts_ids:
                ql_agents[ts].state = env.encode(obs[ts], ts)

        done = {"__all__": False}
        total_reward = 0
        step = 0

        print(f"Run {run}/{runs}")
        while not done["__all__"]:
            actions = {ts: ql_agents[ts].act() for ts in ql_agents}
            next_obs, rewards, done, info = env.step(actions)

            for ts in ql_agents:
                ql_agents[ts].learn(
                    next_state=env.encode(next_obs[ts], ts),
                    reward=rewards[ts],
                )
            total_reward += sum(rewards.values())
            step += 1

            if step % 500 == 0:
                avg_wait = info.get("system_total_waiting_time", "N/A")
                print(f"  Step {step} | Total reward: {total_reward:.2f} | "
                      f"System waiting time: {avg_wait}")

        env.save_csv(out_csv, run)
        print(f"  ✓ Run {run} done. Accumulated reward: {total_reward:.2f}\n")

    env.close()

    # ── Save Q-tables ──────────────────────────────────────────────────────
    q_tables = {ts: ql_agents[ts].q_table for ts in ql_agents}
    with open(model_path, "wb") as f:
        pickle.dump(q_tables, f)

    print(f"✅ Q-tables saved to: {model_path}")
    print(f"✅ Training CSVs saved to: btp/outputs/qlearning/{network}_*\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTP Q-Learning Training")
    parser.add_argument("--network", choices=["single", "2way"], default="2way")
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Exploration rate")
    parser.add_argument("--gui", action="store_true", default=False)
    args = parser.parse_args()

    train_qlearning(args.network, args.seconds, args.runs,
                    args.alpha, args.gamma, args.epsilon, args.gui)
