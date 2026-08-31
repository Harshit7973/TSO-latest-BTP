"""Generate deterministic target-domain routes for the Task 8 transfer study."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
SOURCE_ROUTE = TASK_DIR.parents[1] / "sumo_rl/nets/2x2grid/2x2.rou.xml"

DOMAINS = {
    "target_horizontal": {"flow_1": 0.16, "flow_2": 0.16, "flow_3": 0.06, "flow_4": 0.06},
    "reverse_vertical": {"flow_1": 0.06, "flow_2": 0.06, "flow_3": 0.16, "flow_4": 0.16},
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, object] = {}
    for domain, probabilities in DOMAINS.items():
        tree = ET.parse(SOURCE_ROUTE)
        root = tree.getroot()
        found: set[str] = set()
        for flow in root.findall("flow"):
            flow_id = str(flow.get("id"))
            if flow_id in probabilities:
                flow.set("probability", str(probabilities[flow_id]))
                found.add(flow_id)
        if found != set(probabilities):
            raise RuntimeError(f"Source route is missing expected flows: {sorted(set(probabilities) - found)}")
        ET.indent(root, space="    ")
        output = output_dir / f"{domain}.rou.xml"
        tree.write(output, encoding="utf-8", xml_declaration=True)
        generated[domain] = {
            "path": str(output.resolve()),
            "probabilities": probabilities,
            "sha256": file_sha256(output),
        }
    manifest = {
        "source": str(SOURCE_ROUTE.resolve()),
        "source_sha256": file_sha256(SOURCE_ROUTE),
        "domains": generated,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=TASK_DIR / "routes")
    args = parser.parse_args()
    manifest = generate(args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
