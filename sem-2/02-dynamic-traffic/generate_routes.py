"""Generate compact 3,600-second dynamic-demand SUMO route files."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
ROUTES = {
    "ns": "n_t t_s",
    "nw": "n_t t_w",
    "ne": "n_t t_e",
    "sn": "s_t t_n",
    "se": "s_t t_e",
    "sw": "s_t t_w",
    "ew": "e_t t_w",
    "en": "e_t t_n",
    "es": "e_t t_s",
    "we": "w_t t_e",
    "wn": "w_t t_n",
    "ws": "w_t t_s",
}


def directional(ns_rate: int, ew_rate: int) -> dict[str, int]:
    """Distribute an approach rate across straight and two turning routes."""
    values: dict[str, int] = {}
    for route in ROUTES:
        approach_rate = ns_rate if route[0] in {"n", "s"} else ew_rate
        values[route] = int(approach_rate * (0.55 if route[1] in {"s", "n", "w", "e"} and route in {"ns", "sn", "ew", "we"} else 0.225))
    return values


def scenario_stages(seconds: int) -> dict[str, list[tuple[int, int, dict[str, int]]]]:
    quarter = seconds // 4
    balanced = directional(500, 500)
    ns_peak = directional(850, 220)
    ew_peak = directional(220, 850)
    low = directional(220, 220)
    high = directional(900, 900)
    unseen = directional(650, 380)
    return {
        "balanced": [(0, seconds, balanced)],
        "ns_peak": [(0, seconds, ns_peak)],
        "ew_peak": [(0, seconds, ew_peak)],
        "direction_switch": [
            (0, quarter, balanced),
            (quarter, quarter * 2, ns_peak),
            (quarter * 2, quarter * 3, ew_peak),
            (quarter * 3, seconds, balanced),
        ],
        "burst": [
            (0, quarter, low),
            (quarter, quarter * 2, high),
            (quarter * 2, quarter * 3, low),
            (quarter * 3, seconds, high),
        ],
        "unseen_mixed": [(0, seconds, unseen)],
    }


def write_route_file(name: str, stages: list[tuple[int, int, dict[str, int]]], output_dir: Path) -> dict:
    root = ET.Element("routes")
    for route_id, edges in ROUTES.items():
        ET.SubElement(root, "route", id=f"route_{route_id}", edges=edges)
    total_expected = 0.0
    for stage_index, (begin, end, rates) in enumerate(stages, start=1):
        for route_id, rate in rates.items():
            if rate <= 0:
                continue
            total_expected += rate * (end - begin) / 3600.0
            ET.SubElement(
                root,
                "flow",
                id=f"{name}_{stage_index}_{route_id}",
                route=f"route_{route_id}",
                begin=str(begin),
                end=str(end),
                vehsPerHour=str(rate),
                departSpeed="max",
                departPos="base",
                departLane="best",
            )
    ET.indent(root, space="    ")
    output = output_dir / f"{name}.rou.xml"
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return {"scenario": name, "file": str(output.resolve()), "expected_insertions": round(total_expected, 1), "stages": len(stages)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.seconds < 600 or args.seconds % 4 != 0:
        raise SystemExit("--seconds must be at least 600 and divisible by 4")
    output_dir = args.output_dir or (TASK_DIR / "generated-routes" / f"sec{args.seconds}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = [write_route_file(name, stages, output_dir) for name, stages in scenario_stages(args.seconds).items()]
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for item in manifest:
        print(f"Generated {item['scenario']}: ~{item['expected_insertions']} requested vehicles")


if __name__ == "__main__":
    main()
