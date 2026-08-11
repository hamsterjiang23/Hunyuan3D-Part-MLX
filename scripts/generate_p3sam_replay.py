from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from split3d.hunyuan.p3sam_official_postprocess import clean_mesh_official
from split3d.hunyuan.p3sam_pipeline import farthest_point_indices, normalize_point_cloud
from split3d.hunyuan.sonata_data import official_fps_start_index


def mesh_geometry_hash(mesh: trimesh.Trimesh) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.vertices).view(np.uint8))
    digest.update(np.ascontiguousarray(mesh.faces).view(np.uint8))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a backend-independent replay manifest for synchronized P3-SAM inference"
    )
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=100_000)
    parser.add_argument("--prompts", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--official-fps-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-mesh", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    loaded = trimesh.load(args.mesh, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected a Trimesh, got {type(loaded).__name__}")
    mesh = clean_mesh_official(loaded) if args.clean_mesh else loaded.copy()
    sampled_points, face_indices = trimesh.sample.sample_surface(mesh, args.points, seed=args.seed)
    normalized_points = normalize_point_cloud(sampled_points)
    normals = np.asarray(mesh.face_normals)[face_indices]
    start_index = (
        official_fps_start_index(normalized_points, seed=args.seed) if args.official_fps_start else 0
    )
    prompt_indices = farthest_point_indices(normalized_points, args.prompts, start_index=start_index)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    geometry_hash = mesh_geometry_hash(mesh)
    np.savez_compressed(
        args.output,
        mesh_geometry_hash=np.asarray(geometry_hash),
        seed=np.asarray(args.seed, dtype=np.int64),
        sampled_points=np.asarray(sampled_points),
        normalized_points=np.asarray(normalized_points),
        normals=np.asarray(normals),
        face_indices=np.asarray(face_indices),
        prompt_indices=np.asarray(prompt_indices),
    )
    metadata = {
        "mesh": str(args.mesh),
        "mesh_geometry_hash": geometry_hash,
        "clean_mesh": args.clean_mesh,
        "point_count": args.points,
        "prompt_count": args.prompts,
        "seed": args.seed,
        "official_fps_start": args.official_fps_start,
        "fps_start_index": start_index,
        "face_count": len(mesh.faces),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
