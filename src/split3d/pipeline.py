from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .contracts import SplitManifest
from .export import export_partition
from .io import atomic_json, load_mesh, mesh_report, sha256_file
from .partition import (
    connected_component_labels,
    fill_unlabeled,
    merge_small_label_regions,
    split_instances,
    validate_labels,
)
from .render import render_views


def inspect_asset(source: Path) -> dict[str, Any]:
    mesh = load_mesh(source, process=False)
    return {"source": str(source.resolve()), "sha256": sha256_file(source), **mesh_report(mesh)}


def render_asset(source: Path, output_dir: Path, *, view_count: int = 12, resolution: int = 512) -> dict[str, Any]:
    mesh = load_mesh(source, process=False)
    records = render_views(mesh, output_dir, view_count=view_count, resolution=resolution)
    return {
        "source": str(source.resolve()),
        "views": len(records),
        "resolution": resolution,
        "manifest": str((output_dir / "views.json").resolve()),
    }


def auto_split_asset(
    source: Path,
    output_dir: Path,
    *,
    part_names: list[str],
    detector_model: str | Path,
    segmenter_model: str | Path,
    view_count: int = 12,
    resolution: int = 512,
    device: str = "auto",
    local_files_only: bool = True,
    individual: bool = True,
) -> SplitManifest:
    from .vision import GroundingDinoDetector, Sam2BoxSegmenter, infer_face_labels

    mesh = load_mesh(source, process=False)
    views_dir = output_dir / "views"
    render_views(mesh, views_dir, view_count=view_count, resolution=resolution)
    detector = GroundingDinoDetector(detector_model, device=device, local_files_only=local_files_only)
    segmenter = Sam2BoxSegmenter(segmenter_model, device=device, local_files_only=local_files_only)
    labels, scores, detections = infer_face_labels(
        views_dir,
        len(mesh.faces),
        part_names,
        detector,
        segmenter,
    )
    labels = fill_unlabeled(mesh, labels)
    labels = merge_small_label_regions(mesh, labels, min_faces=8, min_semantic_fraction=0.1)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "face_labels.npy"
    np.save(labels_path, labels, allow_pickle=False)
    np.save(output_dir / "face_scores.npy", scores, allow_pickle=False)
    atomic_json(output_dir / "detections.json", {"schema_version": 1, "views": detections})
    return split_asset(
        source,
        output_dir,
        part_names=part_names,
        face_labels_path=labels_path,
        individual=individual,
    )


def split_asset(
    source: Path,
    output_dir: Path,
    *,
    part_names: list[str] | None = None,
    face_labels_path: Path | None = None,
    individual: bool = True,
) -> SplitManifest:
    mesh = load_mesh(source, process=False)
    warnings: list[str] = []
    if face_labels_path is None:
        labels = connected_component_labels(mesh)
        component_count = int(labels.max(initial=-1) + 1)
        part_names = [f"component_{index + 1:02d}" for index in range(component_count)]
        assignment = "connected_components"
        if component_count == 1:
            warnings.append("mesh is one connected component; semantic inference is required for meaningful parts")
    else:
        if not part_names:
            raise ValueError("--parts is required when --face-labels is provided")
        raw_labels = np.load(face_labels_path, allow_pickle=False)
        labels = validate_labels(raw_labels, part_names, len(mesh.faces))
        labels = fill_unlabeled(mesh, labels)
        assignment = "provided_face_labels"

    assert part_names is not None
    groups = split_instances(mesh, labels)
    source_hash = sha256_file(source)
    records, split_path = export_partition(
        mesh,
        groups,
        part_names,
        source_hash,
        output_dir,
        individual=individual,
    )
    (output_dir / "preview.png").unlink(missing_ok=True)
    manifest = SplitManifest(
        schema_version=1,
        source=str(source.resolve()),
        source_sha256=source_hash,
        source_faces=int(len(mesh.faces)),
        source_vertices=int(len(mesh.vertices)),
        assignment=assignment,
        split_glb=split_path.relative_to(output_dir).as_posix(),
        parts=records,
        warnings=warnings,
    )
    atomic_json(output_dir / "parts.json", manifest.to_dict())
    return manifest
