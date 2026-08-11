from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass(frozen=True)
class FaceGroup:
    semantic_index: int
    instance_index: int
    face_indices: np.ndarray


def geometric_face_adjacency(mesh: trimesh.Trimesh, decimals: int = 7) -> np.ndarray:
    """Build face adjacency after a temporary position-only weld.

    Attribute seams frequently duplicate vertices at identical positions. They
    must remain duplicated for UV and normal preservation, but should still be
    adjacent for segmentation and instance grouping.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("mesh bounding-box diagonal must be positive and finite")
    normalized = (vertices - bounds.mean(axis=0)) / diagonal
    _, inverse = np.unique(np.round(normalized, decimals=decimals), axis=0, return_inverse=True)
    welded_faces = inverse[np.asarray(mesh.faces, dtype=np.int64)]
    return np.asarray(trimesh.graph.face_adjacency(faces=welded_faces), dtype=np.int64)


def connected_component_labels(mesh: trimesh.Trimesh) -> np.ndarray:
    labels = np.full(len(mesh.faces), -1, dtype=np.int32)
    adjacency = geometric_face_adjacency(mesh)
    nodes = np.arange(len(mesh.faces), dtype=np.int64)
    for component_index, face_ids in enumerate(
        trimesh.graph.connected_components(adjacency, nodes=nodes, min_len=1)
    ):
        labels[np.asarray(face_ids, dtype=np.int64)] = component_index
    if np.any(labels < 0):
        raise RuntimeError("connected-component analysis did not assign every face")
    return labels


def fill_unlabeled(mesh: trimesh.Trimesh, labels: np.ndarray) -> np.ndarray:
    filled = np.asarray(labels, dtype=np.int32).copy()
    if filled.shape != (len(mesh.faces),):
        raise ValueError(f"face label shape must be ({len(mesh.faces)},), got {filled.shape}")
    if np.all(filled < 0):
        raise ValueError("at least one face must have a non-negative label")

    adjacency: list[list[int]] = [[] for _ in range(len(mesh.faces))]
    for left, right in geometric_face_adjacency(mesh):
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))

    queue = deque(int(index) for index in np.flatnonzero(filled >= 0))
    while queue:
        face_id = queue.popleft()
        for neighbor in adjacency[face_id]:
            if filled[neighbor] < 0:
                filled[neighbor] = filled[face_id]
                queue.append(neighbor)

    if np.any(filled < 0):
        fallback = int(np.bincount(filled[filled >= 0]).argmax())
        filled[filled < 0] = fallback
    return filled


def validate_labels(labels: np.ndarray, part_names: list[str], face_count: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.shape != (face_count,):
        raise ValueError(f"face label shape must be ({face_count},), got {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("face labels must use an integer dtype")
    values = values.astype(np.int32, copy=False)
    if values.size and values.max(initial=-1) >= len(part_names):
        raise ValueError("face labels contain an index not present in --parts")
    if values.min(initial=-1) < -1:
        raise ValueError("only -1 is accepted as the unlabeled sentinel")
    return values


def merge_small_label_regions(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    min_faces: int,
    *,
    min_semantic_fraction: float = 0.0,
) -> np.ndarray:
    """Merge minor semantic islands into the label sharing most of their boundary."""
    if min_faces < 1:
        raise ValueError("min_faces must be positive")
    if not 0.0 <= min_semantic_fraction < 1.0:
        raise ValueError("min_semantic_fraction must be in [0, 1)")
    merged = np.asarray(labels, dtype=np.int32).copy()
    if merged.shape != (len(mesh.faces),):
        raise ValueError(f"face label shape must be ({len(mesh.faces)},), got {merged.shape}")
    edges = geometric_face_adjacency(mesh)
    neighbors: list[list[int]] = [[] for _ in range(len(mesh.faces))]
    for left, right in edges:
        neighbors[int(left)].append(int(right))
        neighbors[int(right)].append(int(left))

    for _ in range(8):
        changed = False
        for semantic_index in np.unique(merged):
            selected = np.flatnonzero(merged == semantic_index)
            semantic_threshold = max(min_faces, int(np.ceil(len(selected) * min_semantic_fraction)))
            selected_mask = np.zeros(len(mesh.faces), dtype=bool)
            selected_mask[selected] = True
            selected_edges = edges[selected_mask[edges[:, 0]] & selected_mask[edges[:, 1]]]
            components = trimesh.graph.connected_components(selected_edges, nodes=selected, min_len=1)
            for component in components:
                face_ids = np.asarray(component, dtype=np.int64)
                if len(face_ids) >= semantic_threshold:
                    continue
                boundary_labels = [
                    int(merged[neighbor])
                    for face_id in face_ids
                    for neighbor in neighbors[int(face_id)]
                    if merged[neighbor] != semantic_index
                ]
                if not boundary_labels:
                    continue
                target = int(np.bincount(np.asarray(boundary_labels, dtype=np.int32)).argmax())
                merged[face_ids] = target
                changed = True
        if not changed:
            break
    return merged


def split_instances(mesh: trimesh.Trimesh, labels: np.ndarray) -> list[FaceGroup]:
    groups = [
        FaceGroup(
            semantic_index=semantic_index,
            instance_index=1,
            face_indices=np.flatnonzero(labels == semantic_index),
        )
        for semantic_index in sorted(int(value) for value in np.unique(labels))
    ]
    assigned = np.concatenate([group.face_indices for group in groups]) if groups else np.empty(0, dtype=np.int64)
    if len(assigned) != len(mesh.faces) or len(np.unique(assigned)) != len(mesh.faces):
        raise RuntimeError("partition must assign every source face exactly once")
    return groups


def compress_ranges(indices: np.ndarray) -> list[list[int]]:
    values = np.sort(np.asarray(indices, dtype=np.int64))
    if len(values) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(np.diff(values) != 1) + 1]
    ends = np.r_[starts[1:] - 1, len(values) - 1]
    return [[int(values[start]), int(values[end])] for start, end in zip(starts, ends, strict=True)]
