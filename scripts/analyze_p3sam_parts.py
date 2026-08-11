from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh

from split3d.hunyuan.p3sam_official_postprocess import build_adjacent_faces, connected_regions


def main() -> None:
    parser = argparse.ArgumentParser(description="Report P3-SAM part areas and topology from saved face labels")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loaded = trimesh.load(args.mesh, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected a Trimesh, got {type(loaded).__name__}")
    labels = np.asarray(np.load(args.labels), dtype=np.int64)
    if labels.shape != (len(loaded.faces),):
        raise ValueError(f"labels have shape {labels.shape}, expected {(len(loaded.faces),)}")

    adjacency = build_adjacent_faces(np.asarray(loaded.face_adjacency), len(loaded.faces))
    regions = connected_regions(labels, adjacency)
    component_counts = Counter(int(labels[region[0]]) for region in regions)
    face_areas = np.asarray(loaded.area_faces, dtype=np.float64)
    total_area = float(face_areas.sum())
    vertices = np.asarray(loaded.vertices)
    faces = np.asarray(loaded.faces)
    parts = []
    for part_id in np.unique(labels):
        if part_id < 0:
            continue
        mask = labels == part_id
        part_vertices = vertices[faces[mask]].reshape(-1, 3)
        area = float(face_areas[mask].sum())
        parts.append(
            {
                "part_id": int(part_id),
                "face_count": int(mask.sum()),
                "area": area,
                "area_fraction": area / total_area if total_area else 0.0,
                "connected_component_count": component_counts[int(part_id)],
                "bounds": [part_vertices.min(axis=0).tolist(), part_vertices.max(axis=0).tolist()],
            }
        )
    parts.sort(key=lambda part: part["area"], reverse=True)
    report = {
        "mesh": str(args.mesh),
        "labels": str(args.labels),
        "vertex_count": len(loaded.vertices),
        "face_count": len(loaded.faces),
        "total_area": total_area,
        "part_count": len(parts),
        "unassigned_face_count": int(np.sum(labels < 0)),
        "parts": parts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
