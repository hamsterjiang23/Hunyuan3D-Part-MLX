from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from split3d.hunyuan.metrics import instance_miou


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate projected, connectivity and full P3-SAM face labels")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target = np.load(args.target)
    stages = {
        "projected": args.result / "face_ids_projected.npy",
        "connectivity": args.result / "face_ids_connectivity.npy",
        "full_postprocess": args.result / "face_ids.npy",
    }
    metrics = {}
    for name, path in stages.items():
        labels = np.load(path)
        if labels.shape != target.shape:
            raise ValueError(
                f"{name} labels have shape {labels.shape}, but target has {target.shape}; "
                "rerun dataset evaluation with --no-clean-mesh so face indices remain aligned"
            )
        metrics[name] = {
            "instance_miou": instance_miou(labels, target),
            "part_count": int(np.sum(np.unique(labels) >= 0)),
            "unassigned_face_count": int(np.sum(labels < 0)),
        }
    payload = {
        "target_part_count": int(len(np.unique(target))),
        "face_count": int(len(target)),
        "stages": metrics,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
