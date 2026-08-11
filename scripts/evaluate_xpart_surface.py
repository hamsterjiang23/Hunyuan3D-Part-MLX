from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from split3d.hunyuan.metrics import best_rotated_surface_metrics


def _as_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected mesh or scene at {path}, got {type(loaded).__name__}")
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an X-Part scene with paper surface metrics")
    parser.add_argument("predicted", type=Path, help="Generated X-Part GLB scene")
    parser.add_argument("target", type=Path, help="Target object mesh")
    parser.add_argument("--points", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    predicted_scene = trimesh.load(args.predicted, force="scene")
    predicted = _as_mesh(args.predicted)
    target = _as_mesh(args.target)
    predicted_points = trimesh.sample.sample_surface(predicted, args.points, seed=args.seed)[0]
    target_points = trimesh.sample.sample_surface(target, args.points, seed=args.seed)[0]
    metrics = best_rotated_surface_metrics(predicted_points, target_points, axis=args.axis)
    geometries = list(predicted_scene.geometry.values())
    result: dict[str, Any] = {
        "protocol": {
            "sampled_surface_points": args.points,
            "normalization": "independent axis-aligned bounding cube to [-1, 1]",
            "rotations_degrees": [0, 90, 180, 270],
            "rotation_axis": args.axis,
            "seed": args.seed,
        },
        "geometry_count": len(geometries),
        "vertices": int(sum(len(geometry.vertices) for geometry in geometries)),
        "faces": int(sum(len(geometry.faces) for geometry in geometries)),
        "nonfinite_vertices": int(
            sum(np.size(geometry.vertices) - np.isfinite(geometry.vertices).sum() for geometry in geometries)
        ),
        "all_watertight": bool(all(geometry.is_watertight for geometry in geometries)),
        **metrics.to_dict(),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
