# -*- coding: utf-8 -*-
"""
BTP - Fixed-Time vs DQN: Testing & Plotting
=============================================
Runs both Fixed-Time (baseline) and a trained DQN agent on the same SUMO
network, collects step-by-step metrics, and generates comparison plots
— all in one script, no pre-existing CSV data required.

Usage:
    # Quick test on 2way network (default)
    python btp/test_and_plot.py --model btp/models/dqn_2way.zip

    # Specify network and simulation length
    python btp/test_and_plot.py --model btp/models/dqn_2way.zip --network 2way --seconds 3600

    # Run multiple evaluation runs to average out stochasticity
    python btp/test_and_plot.py --model btp/models/dqn_2way.zip --runs 3

    # Watch in SUMO GUI (slower)
    python btp/test_and_plot.py --model btp/models/dqn_2way.zip --gui

Output:
    btp/outputs/test_results/  ← CSVs for each run
    btp/outputs/test_plots/    ← PNG comparison plots
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

# ── SUMO path setup ───────────────────────────────────────────────────────────
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("[ERROR] Please set the SUMO_HOME environment variable first.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from stable_baselines3 import DQN
from sumo_rl import SumoEnvironment

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

NETWORKS = {
    "single": {
        "net_file":    "sumo_rl/nets/single-intersection/single-intersection.net.xml",
        "route_file":  "sumo_rl/nets/single-intersection/single-intersection.rou.xml",
    },
    "2way": {
        "net_file":    "sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        "route_file":  "sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
    },
    "4x4": {
        "net_file":    "sumo_rl/nets/4x4-grid/4x4.net.xml",
        "route_file":  "sumo_rl/nets/4x4-grid/4x4c1c2c1c2.rou.xml",
    },
}

# Metrics to collect and plot
METRICS = {
    "system_total_waiting_time": "Total Waiting Time (s)",
    "system_mean_waiting_time":  "Mean Waiting Time / Vehicle (s)",
    "system_total_stopped":      "Queue Length (Stopped Vehicles)",
    "system_mean_speed":         "Average Speed (m/s)",
    "system_total_departed":     "Throughput (Vehicles)",
}

# Colour palette
COLOR_FIXED = "#e74c3c"   # Red  — Fixed-Time
COLOR_DQN   = "#2ecc71"   # Green — DQN Agent
FILL_ALPHA  = 0.15

plt.rcParams.update({
    "figure.dpi":         150,
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "axes.labelpad":      8,
    "legend.framealpha":  0.9,
    "legend.fontsize":    10,
})


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runners
# ─────────────────────────────────────────────────────────────────────────────

def run_fixed_time(cfg: dict, num_seconds: int, out_csv: str, use_gui: bool, run_id: int) -> dict:
    """
    Run one episode of Fixed-Time control (agent has no influence).
    Returns a dict of metric_name -> list[float] (one value per step).
    """
    env = SumoEnvironment(
        net_file=cfg["net_file"],
        route_file=cfg["route_file"],
        out_csv_name=out_csv,
        use_gui=use_gui,
        num_seconds=num_seconds,
        single_agent=True,
        fixed_ts=True,          # ← Fixed timing, no RL
    )

    obs = env.reset()
    obs, info = obs if isinstance(obs, tuple) else (obs, {})
    terminated = truncated = False

    records = {m: [] for m in METRICS}
    step = 0

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(None)
        step += 1
        for m in METRICS:
            records[m].append(info.get(m, 0.0))

        if step % 500 == 0:
            _print_step("Fixed-Time", step, info)

    env.save_csv(out_csv, run_id)
    env.close()
    return records


def run_dqn(cfg: dict, model_path: str, num_seconds: int, out_csv: str, use_gui: bool, run_id: int) -> dict:
    """
    Run one evaluation episode with a trained DQN model (deterministic=True).
    Returns a dict of metric_name -> list[float].
    """
    model = DQN.load(model_path)

    env = SumoEnvironment(
        net_file=cfg["net_file"],
        route_file=cfg["route_file"],
        out_csv_name=out_csv,
        use_gui=use_gui,
        num_seconds=num_seconds,
        single_agent=True,
    )

    obs, info = env.reset()
    terminated = truncated = False

    records  = {m: [] for m in METRICS}
    step     = 0
    total_reward = 0.0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        for m in METRICS:
            records[m].append(info.get(m, 0.0))

        if step % 500 == 0:
            _print_step("DQN Agent", step, info)

    env.save_csv(out_csv, run_id)
    env.close()
    return records, total_reward


def _print_step(label: str, step: int, info: dict):
    wt = info.get("system_total_waiting_time", 0)
    q  = info.get("system_total_stopped", 0)
    sp = info.get("system_mean_speed", 0)
    print(f"  [{label}] Step {step:>5} | Wait: {wt:>8.1f}s | Queue: {q:>4} | Speed: {sp:.3f} m/s")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def rolling_mean(data: list, window: int = 50) -> np.ndarray:
    return pd.Series(data).rolling(window=window, min_periods=1).mean().to_numpy()


def rolling_std(data: list, window: int = 50) -> np.ndarray:
    return pd.Series(data).rolling(window=window, min_periods=1).std().fillna(0).to_numpy()


def _avg_records(list_of_dicts: list[dict]) -> dict:
    """Average multiple run records step-by-step (pad shorter runs with NaN)."""
    merged = {}
    for m in METRICS:
        arrays = [r[m] for r in list_of_dicts]
        max_len = max(len(a) for a in arrays)
        padded = [np.pad(np.array(a, float), (0, max_len - len(a)),
                         constant_values=np.nan) for a in arrays]
        merged[m] = np.nanmean(padded, axis=0).tolist()
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Per-metric 4-panel figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric(metric: str, label: str,
                fixed_records: dict, dqn_records: dict,
                network: str, out_dir: str, window: int = 50):
    """
    For one metric, produce a 4-panel figure:
      [TL] Step-by-step timeline    [TR] Bar chart (mean ± std)
      [BL] Cumulative sum           [BR] Box plot
    """
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#f8f9fa")
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    ax_tl = fig.add_subplot(gs[0, 0])  # Timeline
    ax_tr = fig.add_subplot(gs[0, 1])  # Bar chart
    ax_bl = fig.add_subplot(gs[1, 0])  # Cumulative
    ax_br = fig.add_subplot(gs[1, 1])  # Box plot

    for ax in [ax_tl, ax_tr, ax_bl, ax_br]:
        ax.set_facecolor("#ffffff")

    fixed_data = np.array(fixed_records[metric], dtype=float)
    dqn_data   = np.array(dqn_records[metric],   dtype=float)
    x_fixed    = np.arange(len(fixed_data))
    x_dqn      = np.arange(len(dqn_data))

    # ── Top-Left: Step-by-step timeline ──────────────────────────────────────
    ax_tl.set_title(f"Simulation Timeline", fontweight="bold", pad=8)
    ax_tl.set_xlabel("Simulation Step")
    ax_tl.set_ylabel(label)

    for data, x, color, lbl, ls in [
        (fixed_data, x_fixed, COLOR_FIXED, "Fixed-Time (Traditional)", "--"),
        (dqn_data,   x_dqn,   COLOR_DQN,   "DQN Agent (AI)",           "-"),
    ]:
        y   = rolling_mean(data.tolist(), window)
        std = rolling_std(data.tolist(), window)
        ax_tl.plot(x, y, color=color, label=lbl, linestyle=ls, linewidth=2.5)
        ax_tl.fill_between(x, y - std, y + std, color=color, alpha=FILL_ALPHA)

    ax_tl.legend()
    ax_tl.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # ── Top-Right: Bar chart (mean ± std) ────────────────────────────────────
    ax_tr.set_title("Mean Performance ± Std Dev", fontweight="bold", pad=8)
    ax_tr.set_ylabel(label)

    fixed_mean, fixed_std = np.nanmean(fixed_data), np.nanstd(fixed_data)
    dqn_mean,   dqn_std   = np.nanmean(dqn_data),   np.nanstd(dqn_data)

    bars = ax_tr.bar(
        ["Fixed-Time\n(Traditional)", "DQN Agent\n(AI)"],
        [fixed_mean, dqn_mean],
        yerr=[fixed_std, dqn_std],
        color=[COLOR_FIXED, COLOR_DQN],
        capsize=7, alpha=0.85, edgecolor="white", linewidth=1.2,
        error_kw={"elinewidth": 1.8, "ecolor": "dimgray"},
    )

    # Value labels + improvement badge
    for bar, val in zip(bars, [fixed_mean, dqn_mean]):
        ax_tr.text(bar.get_x() + bar.get_width() / 2, val * 0.97,
                   f"{val:.2f}", ha="center", va="top",
                   fontweight="bold", fontsize=11, color="white")

    if fixed_mean != 0:
        pct = (fixed_mean - dqn_mean) / abs(fixed_mean) * 100
        higher_better = "speed" in metric or "departed" in metric
        if higher_better:
            badge = f"^ {abs(pct):.1f}% {'better' if pct < 0 else 'worse'}"
        else:
            badge = f"v {abs(pct):.1f}% {'less' if pct > 0 else 'more'}"
        ax_tr.text(bars[1].get_x() + bars[1].get_width() / 2,
                   dqn_mean / 2, badge,
                   ha="center", va="center",
                   fontweight="bold", color="white", fontsize=11)

    ax_tr.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.1f}"))

    # ── Bottom-Left: Cumulative sum ───────────────────────────────────────────
    ax_bl.set_title("Cumulative Metric Over Episode", fontweight="bold", pad=8)
    ax_bl.set_xlabel("Simulation Step")
    ax_bl.set_ylabel(f"Cumulative {label}")

    ax_bl.plot(x_fixed, np.cumsum(fixed_data), color=COLOR_FIXED,
               label="Fixed-Time", linestyle="--", linewidth=2.5)
    ax_bl.plot(x_dqn,   np.cumsum(dqn_data),   color=COLOR_DQN,
               label="DQN Agent",  linestyle="-",  linewidth=2.5)
    ax_bl.legend()
    ax_bl.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax_bl.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # ── Bottom-Right: Box plot ────────────────────────────────────────────────
    ax_br.set_title("Step-Level Distribution", fontweight="bold", pad=8)
    ax_br.set_ylabel(label)

    bp = ax_br.boxplot(
        [fixed_data[~np.isnan(fixed_data)], dqn_data[~np.isnan(dqn_data)]],
        labels=["Fixed-Time\n(Traditional)", "DQN Agent\n(AI)"],
        patch_artist=True,
        medianprops={"linewidth": 2.5, "color": "white"},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.35},
        widths=0.55,
    )
    for patch, color in zip(bp["boxes"], [COLOR_FIXED, COLOR_DQN]):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)

    ax_br.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.1f}"))

    # ── Supertitle & save ─────────────────────────────────────────────────────
    fig.suptitle(
        f"Fixed-Time  vs  DQN Agent  |  Network: {network.upper()}\n{label}",
        fontsize=15, fontweight="bold", y=1.02, color="#2c3e50",
    )

    out_path = os.path.join(out_dir, f"{network}_{metric}.png")
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_dashboard(fixed_records: dict, dqn_records: dict,
                           network: str, out_dir: str):
    """
    One-page summary: a bar-per-metric side-by-side with % improvement badges.
    """
    n = len(METRICS)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 6))
    fig.patch.set_facecolor("#1a1a2e")

    for ax, (metric, label) in zip(axes, METRICS.items()):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.grid(color="#444466", linestyle="--", alpha=0.4)

        fixed_mean = np.nanmean(fixed_records[metric])
        dqn_mean   = np.nanmean(dqn_records[metric])

        bars = ax.bar(["Fixed", "DQN"],
                      [fixed_mean, dqn_mean],
                      color=[COLOR_FIXED, COLOR_DQN],
                      alpha=0.88, edgecolor="#888888", width=0.55)

        ax.set_title(label, fontsize=9.5, fontweight="bold", pad=6)
        ax.set_ylabel("Mean Value", fontsize=9, color="white")

        for bar, val in zip(bars, [fixed_mean, dqn_mean]):
            ax.text(bar.get_x() + bar.get_width() / 2, val * 0.95,
                    f"{val:.1f}", ha="center", va="top",
                    fontweight="bold", fontsize=10, color="white")

        if fixed_mean != 0:
            pct = (fixed_mean - dqn_mean) / abs(fixed_mean) * 100
            higher_better = "speed" in metric or "departed" in metric
            improved = pct > 0 if not higher_better else pct < 0
            badge_color = "#2ecc71" if improved else "#e67e22"
            badge_txt = f"OK\n{abs(pct):.1f}%" if improved else f"!!\n{abs(pct):.1f}%"
            ax.text(bars[1].get_x() + bars[1].get_width() / 2,
                    dqn_mean * 0.5,
                    badge_txt,
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color=badge_color)

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.suptitle(
        f"[BTP] DQN vs Fixed-Time Summary  |  Network: {network.upper()}",
        fontsize=15, fontweight="bold", color="white", y=1.04,
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{network}_summary_dashboard.png")
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Dashboard saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(fixed_records: dict, dqn_records: dict, dqn_total_reward: float):
    col = 32
    print(f"\n{'='*65}")
    print(f"  [SUMMARY] EVALUATION RESULTS")
    print(f"{'='*65}")
    print(f"  {'Metric':<{col}} {'Fixed-Time':>12}  {'DQN Agent':>12}  {'Change':>12}")
    sep = '-' * (col + 42)
    print(f"  {sep}")

    for metric, label in METRICS.items():
        fm = np.nanmean(fixed_records[metric])
        dm = np.nanmean(dqn_records[metric])
        if fm != 0:
            pct = (fm - dm) / abs(fm) * 100
            higher_better = "speed" in metric or "departed" in metric
            arrow = ("+" if pct < 0 else "-") if higher_better else ("-" if pct > 0 else "+")
            tag = f"{arrow}{abs(pct):.1f}%"
        else:
            tag = "N/A"
        print(f"  {label:<{col}} {fm:>12.2f}  {dm:>12.2f}  {tag:>12}")

    print(f"  {sep}")
    print(f"  {'DQN Total Reward':<{col}} {'---':>12}  {dqn_total_reward:>12.2f}")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BTP — Fixed-Time vs DQN: Test & Plot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",   required=True,
                        help="Path to trained DQN .zip (e.g. btp/models/dqn_2way.zip)")
    parser.add_argument("--network", choices=["single", "2way", "4x4"], default="2way",
                        help="Road network (default: 2way)")
    parser.add_argument("--seconds", type=int, default=3600,
                        help="Simulation seconds per run (default: 3600)")
    parser.add_argument("--runs",    type=int, default=1,
                        help="Number of evaluation runs to average (default: 1)")
    parser.add_argument("--window",  type=int, default=50,
                        help="Rolling-average window for plots (default: 50)")
    parser.add_argument("--gui",     action="store_true", default=False,
                        help="Enable SUMO GUI (much slower)")
    args = parser.parse_args()

    cfg = NETWORKS[args.network]

    out_csv_dir  = "btp/outputs/test_results"
    out_plot_dir = "btp/outputs/test_plots"
    os.makedirs(out_csv_dir,  exist_ok=True)
    os.makedirs(out_plot_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  [BTP] Fixed-Time vs DQN  |  Network: {args.network.upper()}")
    print(f"  Model   : {args.model}")
    print(f"  Seconds : {args.seconds}   Runs: {args.runs}   Window: {args.window}")
    print(f"{'='*65}\n")

    # ── Validate model path ───────────────────────────────────────────────────
    if not os.path.isfile(args.model):
        sys.exit(f"[ERROR] Model not found: {args.model}")

    # ── Run both agents for `args.runs` episodes ──────────────────────────────
    fixed_run_records = []
    dqn_run_records   = []
    total_dqn_reward  = 0.0

    for run in range(1, args.runs + 1):
        print(f"--- Run {run}/{args.runs} " + "-"*50)

        # Fixed-Time
        print(f"\n[Fixed-Time] Run {run}/{args.runs}")
        fixed_csv = os.path.join(out_csv_dir, f"{args.network}_fixed")
        fixed_rec = run_fixed_time(cfg, args.seconds, fixed_csv, args.gui, run)
        fixed_run_records.append(fixed_rec)
        print(f"  >> Fixed-Time run {run} complete.\n")

        # DQN
        print(f"[DQN Agent] Run {run}/{args.runs}")
        dqn_csv = os.path.join(out_csv_dir, f"{args.network}_dqn")
        dqn_rec, reward = run_dqn(cfg, args.model, args.seconds, dqn_csv, args.gui, run)
        dqn_run_records.append(dqn_rec)
        total_dqn_reward += reward
        print(f"  >> DQN run {run} complete | Episode reward: {reward:.2f}\n")

    # ── Average across runs if multiple ──────────────────────────────────────
    fixed_records = _avg_records(fixed_run_records) if args.runs > 1 else fixed_run_records[0]
    dqn_records   = _avg_records(dqn_run_records)   if args.runs > 1 else dqn_run_records[0]
    avg_dqn_reward = total_dqn_reward / args.runs

    # ── Console summary ───────────────────────────────────────────────────────
    print_summary(fixed_records, dqn_records, avg_dqn_reward)

    # ── Generate plots ────────────────────────────────────────────────────────
    print(f"{'='*65}")
    print(f"  Generating plots -> {out_plot_dir}/")
    print(f"{'='*65}\n")

    for metric, label in METRICS.items():
        print(f"  >> Plotting: {label}")
        plot_metric(metric, label, fixed_records, dqn_records,
                    args.network, out_plot_dir, window=args.window)

    # Summary dashboard
    print(f"\n  >> Generating summary dashboard...")
    plot_summary_dashboard(fixed_records, dqn_records, args.network, out_plot_dir)

    print(f"\nDone! Results saved to:")
    print(f"    CSVs  -> {out_csv_dir}/")
    print(f"    Plots -> {out_plot_dir}/\n")


if __name__ == "__main__":
    main()
