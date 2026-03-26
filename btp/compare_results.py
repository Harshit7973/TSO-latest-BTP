"""
BTP - Results Comparison & Plotting
=====================================
Reads all CSV outputs from baseline, Q-Learning, and DQN experiments
and produces publication-quality comparison plots for the BTP report.

Plots generated:
  1. Step-by-step metric comparison (Baseline vs Q-Learning vs DQN best episode)
  2. DQN learning curve — episode-by-episode improvement over training
  3. Summary bar chart of mean metrics across all algorithms
  4. Distribution box plot of episode-level metrics

Usage:
    python btp/compare_results.py --network 2way
    python btp/compare_results.py --network 2way --metric system_total_waiting_time
"""

import os
import re
import argparse
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ── Colour palette (dark, premium) ────────────────────────────────────────────
PALETTE = {
    "Baseline (Fixed)": "#e74c3c",
    "Q-Learning":       "#f39c12",
    "DQN":              "#2ecc71",
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
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
})

METRICS = {
    "system_total_waiting_time": "Total Waiting Time (s)",
    "system_total_stopped":      "Total Stopped Vehicles",
    "system_mean_speed":         "Mean Speed (m/s)",
    "system_mean_waiting_time":  "Mean Waiting Time (s)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csvs(pattern: str) -> pd.DataFrame | None:
    """Load and concatenate all matching CSV files (treats them as one long run)."""
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def load_episodes(pattern: str) -> list[pd.DataFrame] | None:
    """Load each CSV as a separate episode. Returns list sorted by episode number."""
    files = glob.glob(pattern)
    if not files:
        return None

    def ep_num(path):
        m = re.search(r"ep(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else 0

    files = sorted(files, key=ep_num)
    return [pd.read_csv(f) for f in files]


def episode_means(episodes: list[pd.DataFrame], metric: str) -> np.ndarray:
    """Return the per-episode mean of a metric."""
    return np.array([ep[metric].mean() for ep in episodes if metric in ep.columns])


def rolling(series, window=50):
    return series.rolling(window=window, min_periods=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Step-by-step comparison (best episode each)
# ─────────────────────────────────────────────────────────────────────────────

def plot_step_comparison(axes, metric, baseline_df, ql_df, dqn_episodes):
    """Per-step rolling-mean comparison on a single axis."""
    ylabel = METRICS.get(metric, metric)
    axes.set_title(f"Step-by-Step: {ylabel}", fontsize=12, fontweight="bold", pad=10)
    axes.set_xlabel("Simulation Step")
    axes.set_ylabel(ylabel)

    datasets = {}
    if baseline_df is not None and metric in baseline_df.columns:
        datasets["Baseline (Fixed)"] = baseline_df[metric]
    if ql_df is not None and metric in ql_df.columns:
        datasets["Q-Learning"] = ql_df[metric]

    # For DQN use the episode with the best (lowest) mean metric
    if dqn_episodes:
        ep_means = episode_means(dqn_episodes, metric)
        if len(ep_means):
            # lower is better for waiting time / stopped; higher for speed
            if "speed" in metric:
                best_idx = int(np.argmax(ep_means))
            else:
                best_idx = int(np.argmin(ep_means))
            datasets["DQN"] = dqn_episodes[best_idx][metric]

    for label, series in datasets.items():
        color = PALETTE[label]
        y = rolling(series)
        x = np.arange(len(y))
        ls = "--" if label == "Baseline (Fixed)" else "-"
        lw = 2.5 if label == "DQN" else 2.0
        axes.plot(x, y, label=label, color=color, linestyle=ls, linewidth=lw)
        std = series.rolling(50, min_periods=1).std()
        axes.fill_between(x, y - std, y + std, alpha=FILL_ALPHA, color=color)

    axes.legend()
    axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — DQN learning curve (episode-level)
# ─────────────────────────────────────────────────────────────────────────────

def plot_dqn_learning_curve(axes, metric, dqn_episodes, baseline_df, ql_df):
    """Episode-by-episode learning curve for DQN with baseline/QL reference lines."""
    ylabel = METRICS.get(metric, metric)
    axes.set_title("DQN Learning Curve (per Episode)", fontsize=12, fontweight="bold", pad=10)
    axes.set_xlabel("Episode")
    axes.set_ylabel(ylabel)

    if not dqn_episodes:
        axes.text(0.5, 0.5, "No DQN data", ha="center", va="center",
                  transform=axes.transAxes, fontsize=13, color="gray")
        return

    ep_means_arr = episode_means(dqn_episodes, metric)
    n = len(ep_means_arr)
    x = np.arange(1, n + 1)

    # Raw scatter
    axes.scatter(x, ep_means_arr, color=PALETTE["DQN"], alpha=0.25, s=12, zorder=2)

    # Smoothed trend
    smooth = pd.Series(ep_means_arr).rolling(window=max(1, n // 20), min_periods=1).mean()
    axes.plot(x, smooth, color=PALETTE["DQN"], linewidth=2.5,
              label="DQN (smoothed)", zorder=3)

    # Reference lines
    if baseline_df is not None and metric in baseline_df.columns:
        bl_mean = baseline_df[metric].mean()
        axes.axhline(bl_mean, color=PALETTE["Baseline (Fixed)"], linestyle="--",
                     linewidth=1.8, label=f"Baseline mean ({bl_mean:.1f})")

    if ql_df is not None and metric in ql_df.columns:
        ql_mean = ql_df[metric].mean()
        axes.axhline(ql_mean, color=PALETTE["Q-Learning"], linestyle="-.",
                     linewidth=1.8, label=f"Q-Learning mean ({ql_mean:.1f})")

    axes.legend()
    axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Bar chart summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_bar_summary(axes, metric, baseline_df, ql_df, dqn_episodes):
    """Grouped bar chart of mean ± std for each algorithm."""
    ylabel = METRICS.get(metric, metric)
    axes.set_title(f"Mean {ylabel}", fontsize=12, fontweight="bold", pad=10)
    axes.set_ylabel(ylabel)

    labels, means, stds, colors = [], [], [], []

    for label, df in [("Baseline (Fixed)", baseline_df), ("Q-Learning", ql_df)]:
        if df is not None and metric in df.columns:
            labels.append(label)
            means.append(df[metric].mean())
            stds.append(df[metric].std())
            colors.append(PALETTE[label])

    if dqn_episodes:
        ep_means_arr = episode_means(dqn_episodes, metric)
        if len(ep_means_arr):
            labels.append("DQN")
            means.append(ep_means_arr.mean())
            stds.append(ep_means_arr.std())
            colors.append(PALETTE["DQN"])

    bars = axes.bar(labels, means, yerr=stds, capsize=6,
                    color=colors, alpha=0.82, edgecolor="white", linewidth=1.2,
                    error_kw={"elinewidth": 1.5, "ecolor": "dimgray"})

    # Value labels on bars
    for bar, mean_val in zip(bars, means):
        axes.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() * 0.98,
                  f"{mean_val:.1f}",
                  ha="center", va="top", fontsize=10, fontweight="bold",
                  color="white")

    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels, fontsize=10)
    axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Distribution box plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplot(axes, metric, baseline_df, ql_df, dqn_episodes):
    """Box plot of episode-level metric distributions."""
    ylabel = METRICS.get(metric, metric)
    axes.set_title(f"Distribution: {ylabel}", fontsize=12, fontweight="bold", pad=10)
    axes.set_ylabel(ylabel)

    plot_data, plot_labels, plot_colors = [], [], []

    if baseline_df is not None and metric in baseline_df.columns:
        plot_data.append(baseline_df[metric].values)
        plot_labels.append("Baseline\n(Fixed)")
        plot_colors.append(PALETTE["Baseline (Fixed)"])

    if ql_df is not None and metric in ql_df.columns:
        plot_data.append(ql_df[metric].values)
        plot_labels.append("Q-Learning")
        plot_colors.append(PALETTE["Q-Learning"])

    if dqn_episodes:
        ep_means_arr = episode_means(dqn_episodes, metric)
        if len(ep_means_arr):
            plot_data.append(ep_means_arr)
            plot_labels.append("DQN\n(per episode)")
            plot_colors.append(PALETTE["DQN"])

    if not plot_data:
        return

    bp = axes.boxplot(plot_data, labels=plot_labels, patch_artist=True,
                      medianprops={"linewidth": 2, "color": "white"},
                      whiskerprops={"linewidth": 1.5},
                      capprops={"linewidth": 1.5},
                      flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})

    for patch, color in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))


# ─────────────────────────────────────────────────────────────────────────────
# Main compare function
# ─────────────────────────────────────────────────────────────────────────────

def compare(network: str, metrics: list, window: int):
    os.makedirs("btp/outputs/plots", exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    baseline_df  = load_csvs(f"btp/outputs/baseline/{network}_conn*_ep*.csv")
    ql_df        = load_csvs(f"btp/outputs/qlearning/{network}_conn*_ep*.csv")
    dqn_episodes = load_episodes(f"btp/outputs/dqn/{network}_conn*_ep*.csv")

    if baseline_df is None and ql_df is None and dqn_episodes is None:
        print(f"\n❌ No CSV data found for network '{network}'.")
        print("   Please run train_baseline.py, train_qlearning.py, and train_dqn.py first.\n")
        return

    n_dqn_eps = len(dqn_episodes) if dqn_episodes else 0
    print(f"\n{'='*65}")
    print(f"  BTP RESULTS COMPARISON | Network: {network.upper()}")
    print(f"  Baseline rows : {len(baseline_df) if baseline_df is not None else 0:,}")
    print(f"  Q-Learning rows : {len(ql_df) if ql_df is not None else 0:,}")
    print(f"  DQN episodes   : {n_dqn_eps}")
    print(f"{'='*65}\n")

    # ── For each metric generate a 4-panel figure ──────────────────────────
    for metric in metrics:
        metric_label = METRICS.get(metric, metric)
        print(f"▶ Plotting: {metric_label}")

        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor("#f8f9fa")
        gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])  # Step comparison
        ax2 = fig.add_subplot(gs[0, 1])  # DQN learning curve
        ax3 = fig.add_subplot(gs[1, 0])  # Bar summary
        ax4 = fig.add_subplot(gs[1, 1])  # Box plot

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor("#ffffff")

        plot_step_comparison(ax1, metric, baseline_df, ql_df, dqn_episodes)
        plot_dqn_learning_curve(ax2, metric, dqn_episodes, baseline_df, ql_df)
        plot_bar_summary(ax3, metric, baseline_df, ql_df, dqn_episodes)
        plot_boxplot(ax4, metric, baseline_df, ql_df, dqn_episodes)

        fig.suptitle(
            f"BTP — RL Traffic Signal Optimisation\n"
            f"Network: {network.upper()}   |   Metric: {metric_label}",
            fontsize=15, fontweight="bold", y=1.01, color="#2c3e50"
        )

        out_path = f"btp/outputs/plots/{network}_{metric}.png"
        plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.show()
        print(f"  ✅ Saved → {out_path}")

    # ── Print summary table ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("📊  SUMMARY STATISTICS")
    print(f"{'='*65}")
    col_w = 22
    header = f"{'Algorithm':<{col_w}}" + "".join(
        f"{METRICS.get(m, m)[:18]:<{col_w}}" for m in metrics
    )
    print(header)
    print("-" * (col_w * (1 + len(metrics))))

    rows = [
        ("Baseline (Fixed)", baseline_df, None),
        ("Q-Learning",       ql_df,       None),
        ("DQN (all eps)",    None,         dqn_episodes),
    ]

    for label, df, episodes in rows:
        row = f"{label:<{col_w}}"
        for metric in metrics:
            if df is not None and metric in df.columns:
                row += f"{df[metric].mean():<{col_w}.2f}"
            elif episodes:
                ep_means_arr = episode_means(episodes, metric)
                if len(ep_means_arr):
                    row += f"{ep_means_arr.mean():<{col_w}.2f}"
                else:
                    row += f"{'N/A':<{col_w}}"
            else:
                row += f"{'N/A':<{col_w}}"
        print(row)

    print("-" * (col_w * (1 + len(metrics))))

    # % improvement DQN vs Baseline
    print(f"\n{'─'*65}")
    print("📈  IMPROVEMENT OVER BASELINE")
    print(f"{'─'*65}")
    for metric in metrics:
        ml = METRICS.get(metric, metric)
        if baseline_df is None or metric not in baseline_df.columns:
            continue
        bl_mean = baseline_df[metric].mean()
        if bl_mean == 0:
            continue

        improvements = {}
        if ql_df is not None and metric in ql_df.columns:
            improvements["Q-Learning"] = (bl_mean - ql_df[metric].mean()) / bl_mean * 100
        if dqn_episodes:
            ep_means_arr = episode_means(dqn_episodes, metric)
            if len(ep_means_arr):
                improvements["DQN"] = (bl_mean - ep_means_arr.mean()) / bl_mean * 100

        for alg, pct in improvements.items():
            direction = "improvement ✅" if ("speed" not in metric and pct > 0) \
                else ("worse ⚠️" if pct < 0 else "same")
            if "speed" in metric:
                direction = "improvement ✅" if pct < 0 else "worse ⚠️" if pct > 0 else "same"
            print(f"  {alg:<16} vs Baseline | {ml[:30]:<30}: {abs(pct):.1f}% {direction}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTP Results Comparison")
    parser.add_argument("--network", choices=["single", "2way", "4x4"], default="2way")
    parser.add_argument(
        "--metrics", nargs="+",
        default=["system_total_waiting_time", "system_total_stopped", "system_mean_speed"],
        help="Metrics to plot",
    )
    parser.add_argument("--window", type=int, default=50,
                        help="Rolling average window size (default: 50)")
    args = parser.parse_args()

    compare(args.network, args.metrics, args.window)
