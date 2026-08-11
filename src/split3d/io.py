from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .partition import geometric_face_adjacency

SUPPORTED_SUFFIXES = {".glb", ".gltf", ".obj"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mesh(path: Path, *, process: bool = False) -> trimesh.Trimesh:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        if path.suffix.lower() == ".fbx":
            raise ValueError("FBX conversion is not implemented yet; convert it to GLB with Blender first")
        raise ValueError(f"unsupported mesh format: {path.suffix}")

    loaded = trimesh.load(path, force="scene", process=process)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"no mesh geometry in {path}")
        mesh = loaded.to_geometry()
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"unsupported loaded object: {type(loaded)!r}")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"scene did not resolve to a triangle mesh: {type(mesh)!r}")
    if len(mesh.faces) == 0:
        raise ValueError(f"empty mesh: {path}")
    return mesh


def mesh_report(mesh: trimesh.Trimesh) -> dict[str, Any]:
    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    visual = mesh.visual
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(
            len(
                trimesh.graph.connected_components(
                    geometric_face_adjacency(mesh),
                    nodes=np.arange(len(mesh.faces), dtype=np.int64),
                    min_len=1,
                )
            )
        ),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "finite_vertices": bool(np.isfinite(mesh.vertices).all()),
        "degenerate_faces": int(np.count_nonzero(~np.isfinite(face_areas) | (face_areas <= 1e-15))),
        "has_uv": bool(getattr(visual, "uv", None) is not None),
        "visual_type": type(visual).__name__,
        "bounds": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
