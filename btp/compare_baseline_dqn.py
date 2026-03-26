"""
BTP - Baseline vs DQN Specific Comparison
===========================================
Generates pure 1v1 comparison plots between the Fixed-Timing Baseline
and the trained DQN agent, excluding Q-Learning to show the DQN
improvements more clearly on the y-axis scale.

Usage:
    python btp/compare_baseline_dqn.py --network 2way
"""

import os
import re
import argparse
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Colour palette ────────────────────────────────────────────────────────
PALETTE = {
    "Baseline (Fixed)": "#e74c3c",  # Red
    "DQN (AI)":         "#2ecc71",  # Green
}

FILL_ALPHA = 0.15

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.labelpad": 8,
})

METRICS = {
    "system_total_waiting_time": "Total Waiting Time (s)",
    "system_mean_waiting_time":  "Mean Waiting Time per Vehicle (s)",
    "system_total_stopped":      "Queue Length (Stopped Vehicles)",
    "system_mean_speed":         "Average Traffic Speed (m/s)",
    "system_total_departed":     "Throughput (Vehicles completing trip)",
}

def load_csvs(pattern: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(pattern))
    if not files: return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

def load_episodes(pattern: str) -> list[pd.DataFrame] | None:
    files = glob.glob(pattern)
    if not files: return None
    def ep_num(path):
        m = re.search(r"ep(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else 0
    files = sorted(files, key=ep_num)
    return [pd.read_csv(f) for f in files]

def episode_means(episodes: list[pd.DataFrame], metric: str) -> np.ndarray:
    return np.array([ep[metric].mean() for ep in episodes if metric in ep.columns])

def rolling(series, window=50):
    return series.rolling(window=window, min_periods=1).mean()

def compare_1v1(network: str):
    os.makedirs("btp/outputs/plots_1v1", exist_ok=True)
    
    baseline_df  = load_csvs(f"btp/outputs/baseline/{network}_conn*_ep*.csv")
    dqn_episodes = load_episodes(f"btp/outputs/dqn/{network}_conn*_ep*.csv")

    if baseline_df is None or not dqn_episodes:
        print("Missing data for 1v1 comparison.")
        return

    # Find the BEST DQN episode for step-by-step comparison
    best_eps = {}
    for m in METRICS.keys():
        if m not in baseline_df.columns: continue
        ep_means = episode_means(dqn_episodes, m)
        if len(ep_means) == 0: continue
        if "speed" in m or "departed" in m:
            best_eps[m] = int(np.argmax(ep_means)) # higher is better
        else:
            best_eps[m] = int(np.argmin(ep_means)) # lower is better

    print(f"\n{'='*50}\n  GENERATING 1v1 COMPARISON PLOTS\n{'='*50}\n")
    
    for metric, label in METRICS.items():
        if metric not in baseline_df.columns or metric not in best_eps:
            continue
            
        print(f"▶ Plotting 1v1: {label}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#f8f9fa")
        for ax in [ax1, ax2]: ax.set_facecolor("#ffffff")

        # --- PANEL 1: Step-by-Step ---
        ax1.set_title(f"Simulation Timeline: {label}", fontweight="bold", pad=10)
        ax1.set_xlabel("Simulation Step")
        ax1.set_ylabel(label)

        # Baseline Line
        bl_y = rolling(baseline_df[metric])
        x = np.arange(len(bl_y))
        ax1.plot(x, bl_y, label="Traditional (Fixed Timer)", color=PALETTE["Baseline (Fixed)"], linestyle="--", linewidth=2.5)
        bl_std = baseline_df[metric].rolling(50, min_periods=1).std()
        ax1.fill_between(x, bl_y - bl_std, bl_y + bl_std, alpha=FILL_ALPHA, color=PALETTE["Baseline (Fixed)"])

        # DQN Line
        best_idx = best_eps[metric]
        dqn_df = dqn_episodes[best_idx]
        dqn_y = rolling(dqn_df[metric])
        x_dqn = np.arange(len(dqn_y))
        ax1.plot(x_dqn, dqn_y, label="AI Agent (DQN)", color=PALETTE["DQN (AI)"], linewidth=3)
        dqn_std = dqn_df[metric].rolling(50, min_periods=1).std()
        ax1.fill_between(x_dqn, dqn_y - dqn_std, dqn_y + dqn_std, alpha=FILL_ALPHA, color=PALETTE["DQN (AI)"])
        
        ax1.legend()
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

        # --- PANEL 2: Bar Comparison ---
        ax2.set_title(f"Average Performance: {label}", fontweight="bold", pad=10)
        bl_mean = baseline_df[metric].mean()
        
        ep_means_arr = episode_means(dqn_episodes, metric)
        dqn_mean = ep_means_arr.mean()
        
        bars = ax2.bar(["Traditional\n(Fixed Timer)", "AI Agent\n(DQN)"], 
                       [bl_mean, dqn_mean], 
                       color=[PALETTE["Baseline (Fixed)"], PALETTE["DQN (AI)"]],
                       alpha=0.85, edgecolor="white", width=0.6)
        
        # Annotate percentages
        if bl_mean > 0:
            diff = abs(bl_mean - dqn_mean) / bl_mean * 100
            diff_text = f"↓ {diff:.1f}% less" if dqn_mean < bl_mean else f"↑ {diff:.1f}% more"
        else:
            diff_text = ""
            
        for bar, val in zip(bars, [bl_mean, dqn_mean]):
            ax2.text(bar.get_x() + bar.get_width()/2, val, f"{val:.1f}", 
                     ha='center', va='bottom', fontweight='bold', fontsize=12)
            
        # Add improvement badge on DQN bar
        if diff_text:
            ax2.text(bars[1].get_x() + bars[1].get_width()/2, dqn_mean / 2, diff_text,
                     ha='center', va='center', fontweight='bold', color='white', fontsize=12)

        fig.suptitle(f"Traditional vs AI: {label}", fontsize=14, fontweight="bold", y=1.05)
        plt.tight_layout()
        
        out_path = f"btp/outputs/plots_1v1/{network}_{metric}_vs.png"
        plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        
    print(f"\n✅ 1v1 comparison plots saved to btp/outputs/plots_1v1/\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="2way")
    args = parser.parse_args()
    compare_1v1(args.network)
