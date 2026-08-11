from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh

from split3d.hunyuan.p3sam_official_postprocess import clean_mesh_official, official_mesh_postprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay only the official P3-SAM mesh topology stages")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean-mesh", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postprocess-threshold", type=float, default=0.95)
    parser.add_argument(
        "--compare",
        type=Path,
        help="Existing trace directory whose three face-label stages must match",
    )
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    if args.clean_mesh:
        mesh = clean_mesh_official(mesh)
    with np.load(args.trace / "replay_manifest.npz", allow_pickle=False) as replay:
        face_indices = replay["face_indices"]
    point_ids = np.load(args.trace / "point_ids.npy")
    started = time.perf_counter()
    result = official_mesh_postprocess(
        mesh,
        face_indices,
        point_ids,
        threshold=args.postprocess_threshold,
        postprocess=args.postprocess,
    )
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    stages = {
        "face_ids_projected": result.projected_face_ids,
        "face_ids_connectivity": result.connectivity_face_ids,
        "face_ids_final": result.final_face_ids,
    }
    exact = {}
    for name, labels in stages.items():
        np.save(args.output / f"{name}.npy", labels)
        if args.compare is not None:
            expected = np.load(args.compare / f"{name}.npy")
            exact[name] = {
                "exact": bool(np.array_equal(labels, expected)),
                "different_faces": int(np.count_nonzero(labels != expected)),
            }
    payload = {"seconds": elapsed, "face_count": len(mesh.faces), "comparison": exact}
    (args.output / "runtime.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if exact and not all(stage["exact"] for stage in exact.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
