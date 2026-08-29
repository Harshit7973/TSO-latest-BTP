"""Paired-seed benchmark of the frozen Semester 1 DQN and fixed timing."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
SEM2_ROOT = TASK_DIR.parent
sys.path.insert(0, str(SEM2_ROOT))

from common.experiment_utils import (  # noqa: E402
    NETWORKS,
    REPO_ROOT,
    collect_episode_summaries,
    ensure_output_tree,
    paired_improvements,
    plot_method_comparison,
    require_sumo,
    run_single_agent_episode,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "btp/models/dqn_2way.zip")
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(101, 111)))
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun completed seed/method CSV files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_sumo()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    from stable_baselines3 import DQN
    import stable_baselines3
    import torch

    paths = ensure_output_tree(TASK_DIR)
    episodes_dir = paths["episodes"] / f"sec{args.seconds}"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    model = DQN.load(str(args.model), device="auto")
    cfg = NETWORKS["2way"]

    write_json(
        paths["results"] / "run_config.json",
        {
            "model": str(args.model.resolve()),
            "seconds": args.seconds,
            "seeds": args.seeds,
            "python": platform.python_version(),
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "methods": ["fixed", "dqn_v1"],
        },
    )

    for seed in args.seeds:
        for method, fixed in (("fixed", True), ("dqn_v1", False)):
            output = episodes_dir / f"{method}__seed{seed}.csv"
            if output.exists() and not args.force:
                print(f"[resume] keeping {output.name}")
                continue
            print(f"[run] method={method} seed={seed} seconds={args.seconds}")
            run_single_agent_episode(
                output_csv=output,
                method=method,
                seed=seed,
                seconds=args.seconds,
                net_file=cfg["net_file"],
                route_file=cfg["route_file"],
                model=None if fixed else model,
                fixed=fixed,
                use_gui=args.gui,
            )

    summary, validations = collect_episode_summaries(episodes_dir, args.seconds)
    summary.to_csv(paths["results"] / f"summary_by_seed_sec{args.seconds}.csv", index=False)
    write_json(paths["results"] / f"validation_sec{args.seconds}.json", validations)
    if summary.empty:
        raise SystemExit("No valid episodes were produced; inspect validation.json")

    paired = paired_improvements(summary, baseline="fixed", candidate="dqn_v1")
    paired.to_csv(paths["results"] / f"paired_improvements_sec{args.seconds}.csv", index=False)
    plot_method_comparison(summary, paths["plots"], prefix=f"sec{args.seconds}_")

    required = len(args.seeds) * 2
    passed = sum(bool(item["passed"]) for item in validations)
    print(f"Completed: {passed}/{required} valid method-seed episodes")
    if passed != required:
        print("WARNING: validation is incomplete; do not report final percentages yet.")
    else:
        print("All structural checks passed. Review paired_improvements.csv before reporting results.")


if __name__ == "__main__":
    main()
