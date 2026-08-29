"""Shared, read-only experiment helpers for the Semester 2 tasks.

All generated artifacts are written below ``sem-2``.  The helpers never write
to ``btp`` so the Semester 1 model and results remain unchanged.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEM2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SEM2_ROOT.parent

NETWORKS = {
    "2way": {
        "net_file": REPO_ROOT / "sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        "route_file": REPO_ROOT / "sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
    },
    "single": {
        "net_file": REPO_ROOT / "sumo_rl/nets/single-intersection/single-intersection.net.xml",
        "route_file": REPO_ROOT / "sumo_rl/nets/single-intersection/single-intersection.rou.xml",
    },
}

STEP_METRICS = (
    "system_total_waiting_time",
    "system_mean_waiting_time",
    "system_total_stopped",
    "system_mean_speed",
)
FINAL_METRICS = (
    "system_total_departed",
    "system_total_arrived",
    "system_total_teleported",
)


def require_sumo() -> None:
    """Validate SUMO_HOME before importing sumo_rl/traci."""
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise SystemExit(
            "SUMO_HOME is not set. On Windows run: "
            "$env:SUMO_HOME='C:\\Program Files (x86)\\Eclipse\\Sumo'"
        )
    tools = str(Path(sumo_home) / "tools")
    if tools not in sys.path:
        sys.path.append(tools)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_output_tree(task_dir: Path) -> dict[str, Path]:
    paths = {
        "results": task_dir / "results",
        "episodes": task_dir / "results" / "episodes",
        "plots": task_dir / "plots",
        "checkpoints": task_dir / "checkpoints",
        "logs": task_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def build_single_agent_env(
    *,
    net_file: Path,
    route_file: Path,
    seconds: int,
    seed: int,
    fixed: bool = False,
    use_gui: bool = False,
    observation_class: type | None = None,
    reward_fn: str | Callable = "diff-waiting-time",
    enforce_max_green: bool = True,
):
    require_sumo()
    from sumo_rl import SumoEnvironment

    kwargs: dict[str, Any] = {
        "net_file": str(net_file),
        "route_file": str(route_file),
        "use_gui": use_gui,
        "num_seconds": seconds,
        "single_agent": True,
        "fixed_ts": fixed,
        "sumo_seed": seed,
        "reward_fn": reward_fn,
        "enforce_max_green": enforce_max_green,
        "max_green": 60,
        "out_csv_name": None,
    }
    if observation_class is not None:
        kwargs["observation_class"] = observation_class
    return SumoEnvironment(**kwargs)


def wrap_with_incrementing_seeds(env: Any, base_seed: int, start_index: int = 0):
    """Give successive training episodes deterministic but different SUMO seeds."""
    import gymnasium as gym

    class IncrementingSeedWrapper(gym.Wrapper):
        def __init__(self, wrapped_env, first_seed: int, index: int):
            super().__init__(wrapped_env)
            self.first_seed = first_seed
            self.seed_index = index

        def reset(self, *, seed=None, options=None):
            selected_seed = self.first_seed + self.seed_index
            self.seed_index += 1
            return self.env.reset(seed=selected_seed, options=options)

    return IncrementingSeedWrapper(env, base_seed, start_index)


def run_single_agent_episode(
    *,
    output_csv: Path,
    method: str,
    seed: int,
    seconds: int,
    net_file: Path,
    route_file: Path,
    model: Any | None = None,
    fixed: bool = False,
    use_gui: bool = False,
    observation_class: type | None = None,
    reward_fn: str | Callable = "diff-waiting-time",
) -> dict[str, float | int | str]:
    """Run and persist one complete evaluation episode."""
    set_global_seed(seed)
    env = build_single_agent_env(
        net_file=net_file,
        route_file=route_file,
        seconds=seconds,
        seed=seed,
        fixed=fixed,
        use_gui=use_gui,
        observation_class=observation_class,
        reward_fn=reward_fn,
    )
    records: list[dict[str, Any]] = []
    total_reward = 0.0
    try:
        obs, info = env.reset(seed=seed)
        records.append({"step": float(env.sim_step), "reward": 0.0, **info})
        terminated = truncated = False
        while not (terminated or truncated):
            if fixed:
                action = None
            elif model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                raise ValueError("A learned controller requires a loaded model")
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            records.append({"step": float(env.sim_step), "reward": float(reward), **info})
    finally:
        env.close()

    frame = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    summary = summarize_episode(frame, method=method, seed=seed, seconds=seconds, total_reward=total_reward)
    return summary


def summarize_episode(
    frame: pd.DataFrame, *, method: str, seed: int, seconds: int, total_reward: float
) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {
        "method": method,
        "seed": seed,
        "requested_seconds": seconds,
        "recorded_steps": len(frame),
        "final_sim_time": float(frame["step"].iloc[-1]),
        "total_reward": float(total_reward),
    }
    for metric in STEP_METRICS:
        summary[f"mean_{metric}"] = float(frame[metric].mean())
        summary[f"max_{metric}"] = float(frame[metric].max())
    for metric in FINAL_METRICS:
        summary[f"final_{metric}"] = float(frame[metric].iloc[-1])
    summary["throughput_veh_per_hour"] = (
        float(frame["system_total_arrived"].iloc[-1]) / max(seconds, 1) * 3600.0
    )
    return summary


def validate_episode_csv(path: Path, seconds: int) -> dict[str, Any]:
    checks: dict[str, Any] = {"file": str(path), "passed": True, "checks": {}}
    try:
        frame = pd.read_csv(path)
        required = {"step", *STEP_METRICS, *FINAL_METRICS}
        checks["checks"]["required_columns"] = required.issubset(frame.columns)
        checks["checks"]["nonempty"] = len(frame) > 1
        checks["checks"]["finite_metrics"] = bool(
            np.isfinite(frame[list(STEP_METRICS) + list(FINAL_METRICS)].to_numpy(dtype=float)).all()
        )
        checks["checks"]["simulation_reached_horizon"] = float(frame["step"].iloc[-1]) >= seconds
        checks["checks"]["departed_monotonic"] = bool(frame["system_total_departed"].is_monotonic_increasing)
        checks["checks"]["arrived_monotonic"] = bool(frame["system_total_arrived"].is_monotonic_increasing)
        checks["checks"]["nonnegative_counts"] = bool(
            (frame[["system_total_stopped", *FINAL_METRICS]] >= 0).all().all()
        )
    except Exception as exc:  # validation report should survive one bad file
        checks["error"] = repr(exc)
        checks["passed"] = False
        return checks
    checks["passed"] = all(checks["checks"].values())
    return checks


def collect_episode_summaries(episodes_dir: Path, seconds: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for csv_path in sorted(episodes_dir.glob("*.csv")):
        validation = validate_episode_csv(csv_path, seconds)
        validations.append(validation)
        if not validation["passed"]:
            continue
        frame = pd.read_csv(csv_path)
        parts = csv_path.stem.split("__")
        method = parts[0]
        seed = int(next(part[4:] for part in parts if part.startswith("seed")))
        total_reward = float(frame["reward"].sum()) if "reward" in frame else 0.0
        row = summarize_episode(frame, method=method, seed=seed, seconds=seconds, total_reward=total_reward)
        if len(parts) > 2:
            row["scenario"] = parts[1]
        summaries.append(row)
    return pd.DataFrame(summaries), validations


def mean_ci(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return float("nan"), float("nan")
    mean = float(clean.mean())
    if len(clean) < 2:
        return mean, 0.0
    return mean, float(1.96 * clean.std(ddof=1) / np.sqrt(len(clean)))


def plot_method_comparison(summary: pd.DataFrame, plots_dir: Path, prefix: str = "") -> None:
    specs = {
        "mean_system_total_waiting_time": ("Mean total waiting time (s)", False),
        "mean_system_total_stopped": ("Mean stopped vehicles", False),
        "mean_system_mean_speed": ("Mean speed (m/s)", True),
        "throughput_veh_per_hour": ("Completed trips per hour", True),
    }
    methods = list(dict.fromkeys(summary["method"].tolist()))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(methods), 1)))
    for metric, (label, _higher_is_better) in specs.items():
        if metric not in summary:
            continue
        means, cis = zip(*(mean_ci(summary.loc[summary.method == method, metric]) for method in methods))
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(methods, means, yerr=cis, capsize=6, color=colors)
        ax.set_ylabel(label)
        ax.set_title(f"{label}: paired-seed evaluation")
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{prefix}{metric}.png", dpi=180)
        plt.close(fig)


def paired_improvements(summary: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    metrics = {
        "mean_system_total_waiting_time": False,
        "mean_system_total_stopped": False,
        "mean_system_mean_speed": True,
        "throughput_veh_per_hour": True,
    }
    key_columns = [column for column in ("scenario", "seed") if column in summary.columns]
    left = summary[summary.method == baseline]
    right = summary[summary.method == candidate]
    paired = left.merge(right, on=key_columns, suffixes=("_baseline", "_candidate"))
    output = paired[key_columns].copy()
    for metric, higher_better in metrics.items():
        base = paired[f"{metric}_baseline"]
        cand = paired[f"{metric}_candidate"]
        numerator = cand - base if higher_better else base - cand
        output[f"improvement_pct__{metric}"] = np.where(base != 0, numerator / base.abs() * 100.0, np.nan)
    return output
