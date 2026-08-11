from __future__ import annotations

import re
import uuid
from pathlib import Path

import numpy as np
import trimesh

from .contracts import PartRecord
from .partition import FaceGroup, compress_ranges
from .render import render_mesh_color

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    slug = _SLUG_PATTERN.sub("_", value.strip().lower()).strip("_")
    return slug or "part"


def _stable_part_id(source_hash: str, semantic_name: str, instance_index: int, face_indices: np.ndarray) -> str:
    face_hash = __import__("hashlib").sha256(np.asarray(face_indices, dtype=np.int64).tobytes()).hexdigest()
    value = f"{source_hash}:{semantic_name}:{instance_index}:{face_hash}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def _part_name(semantic_name: str, instance_index: int, instance_count: int) -> str:
    base = slugify(semantic_name)
    return base if instance_count == 1 else f"{base}_{instance_index:02d}"


def export_partition(
    mesh: trimesh.Trimesh,
    groups: list[FaceGroup],
    part_names: list[str],
    source_hash: str,
    output_dir: Path,
    *,
    individual: bool,
) -> tuple[list[PartRecord], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "parts"
    renders_dir = output_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    if individual:
        parts_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[int, int] = {}
    for group in groups:
        counts[group.semantic_index] = counts.get(group.semantic_index, 0) + 1

    scene = trimesh.Scene()
    records: list[PartRecord] = []
    for group in groups:
        semantic_name = part_names[group.semantic_index]
        name = _part_name(semantic_name, group.instance_index, counts[group.semantic_index])
        part_id = _stable_part_id(source_hash, semantic_name, group.instance_index, group.face_indices)
        part_mesh = mesh.submesh([group.face_indices], append=True, repair=False)
        if not isinstance(part_mesh, trimesh.Trimesh):
            raise TypeError("submesh extraction did not return a Trimesh")
        node_name = f"part_{name}_{part_id[:8]}"
        scene.add_geometry(part_mesh, node_name=node_name, geom_name=node_name)
        relative_file: str | None = None
        if individual:
            part_path = parts_dir / f"{name}.glb"
            part_mesh.export(part_path)
            relative_file = part_path.relative_to(output_dir).as_posix()
        render_path = renders_dir / f"{name}.png"
        render_mesh_color(part_mesh, render_path)
        records.append(
            PartRecord(
                part_id=part_id,
                name=name,
                semantic_name=semantic_name,
                instance_index=group.instance_index,
                confidence=1.0,
                face_count=int(len(group.face_indices)),
                source_face_ranges=compress_ranges(group.face_indices),
                bounds=np.asarray(part_mesh.bounds, dtype=np.float64).tolist(),
                node_name=node_name,
                file=relative_file,
                render=render_path.relative_to(output_dir).as_posix(),
            )
        )

    split_path = output_dir / "split.glb"
    scene.export(split_path)
    return records, split_path
