"""Validate task structure and verify key Semester 1 files remain unchanged."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


SEM2_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SEM2_ROOT.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = {
        "01-reproducible-benchmark": {
            "files": ["README.md", "run_benchmark.py"],
            "dirs": ["results", "plots", "checkpoints"],
        },
        "02-dynamic-traffic": {
            "files": ["README.md", "generate_routes.py", "evaluate_scenarios.py"],
            "dirs": ["results", "plots", "checkpoints"],
        },
        "03-multiobjective-dqn": {
            "files": ["README.md", "features.py", "train_dqn_v2.py", "evaluate_dqn_v2.py"],
            "dirs": ["results", "plots", "checkpoints"],
        },
        "04-ppo-comparison": {
            "files": ["README.md", "train_ppo.py", "evaluate_ppo.py"],
            "dirs": ["results", "plots", "checkpoints"],
        },
        "05-multi-intersection": {
            "files": ["README.md", "dqn_core.py", "self_check.py", "train_dqn.py", "evaluate_dqn.py"],
            "dirs": ["results", "plots", "checkpoints"],
        },
        "06-component-ablation": {
            "files": ["README.md", "run_ablation.py"],
            "dirs": ["results", "plots"],
        },
        "07-robustness-stress-test": {
            "files": ["README.md", "run_stress_test.py"],
            "dirs": ["results", "plots"],
        },
        "08-transfer-learning": {
            "files": ["README.md", "generate_target_routes.py", "train_transfer.py", "evaluate_transfer.py"],
            "dirs": ["results", "plots", "checkpoints", "routes"],
        },
        "09-explainability": {
            "files": ["README.md", "run_explainability.py"],
            "dirs": ["results", "plots"],
        },
    }
    report = {"structure": {}, "semester_1_hashes": {}, "passed": True}
    for folder, specification in required.items():
        missing = [
            name for name in specification["files"] if not (SEM2_ROOT / folder / name).exists()
        ]
        output_dirs = [
            name for name in specification["dirs"] if not (SEM2_ROOT / folder / name).is_dir()
        ]
        passed = not missing and not output_dirs
        report["structure"][folder] = {"passed": passed, "missing_files": missing, "missing_output_dirs": output_dirs}
        report["passed"] = report["passed"] and passed

    manifest = json.loads((SEM2_ROOT / "sem1-readonly-manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = REPO_ROOT / relative
        actual = sha256(path) if path.exists() else None
        passed = actual == expected
        report["semester_1_hashes"][relative] = {"passed": passed, "expected": expected, "actual": actual}
        report["passed"] = report["passed"] and passed

    output = SEM2_ROOT / "workspace_validation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Workspace validation failed")
    print(f"Workspace validation passed. Report: {output}")


if __name__ == "__main__":
    main()
