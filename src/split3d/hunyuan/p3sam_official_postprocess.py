"""Native port of the released P3-SAM ``auto_mask.py`` mesh semantics.

This file is a modified implementation derived from Tencent's Hunyuan3D-Part
release.  Neural-network execution remains in MLX; this module implements the
NumPy/trimesh topology path that follows point-mask prediction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OfficialPostprocessResult:
    """All observable face-label stages from the official automatic masker."""

    mesh: Any
    projected_face_ids: np.ndarray
    connectivity_face_ids: np.ndarray
    final_face_ids: np.ndarray


def clean_mesh_official(mesh: Any) -> Any:
    """Apply the released merge/process sequence and drop source visuals."""

    import trimesh

    cleaned = mesh.copy()
    cleaned.merge_vertices()
    cleaned.process(validate=True)
    return trimesh.Trimesh(vertices=cleaned.vertices, faces=cleaned.faces)


def build_adjacent_faces(face_adjacency: np.ndarray, face_count: int) -> np.ndarray:
    """Build the dense ``-1`` padded adjacency table used by the release."""

    adjacency = np.asarray(face_adjacency, dtype=np.int64).reshape(-1, 2)
    degrees = np.zeros(face_count, dtype=np.int32)
    for first, second in adjacency:
        degrees[first] += 1
        degrees[second] += 1
    max_degree = int(degrees.max(initial=0))
    result = np.full((face_count, max_degree), -1, dtype=np.int32)
    counts = np.zeros(face_count, dtype=np.int32)
    for first, second in adjacency:
        result[first, counts[first]] = second
        counts[first] += 1
        result[second, counts[second]] = first
        counts[second] += 1
    return result


def remove_outliers_iqr(data: np.ndarray, factor: float = 1.5) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    first = np.percentile(values, 25)
    third = np.percentile(values, 75)
    spread = third - first
    return values[(values >= first - factor * spread) & (values <= third + factor * spread)]


def better_aabb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    filtered = [remove_outliers_iqr(points[:, axis]) for axis in range(3)]
    return (
        np.asarray([axis.min() for axis in filtered]),
        np.asarray([axis.max() for axis in filtered]),
    )


def fix_labels(
    face_ids: np.ndarray,
    adjacent_faces: np.ndarray,
    *,
    use_aabb: bool = False,
    mesh: Any | None = None,
) -> np.ndarray:
    """Run the official at-most-50-pass neighbor flood fill in face order."""

    labels = np.asarray(face_ids, dtype=np.int64)
    aabb_face_masks: dict[int, np.ndarray] = {}
    if use_aabb:
        if mesh is None:
            raise ValueError("mesh is required when use_aabb=True")
        flat_points = np.asarray(mesh.vertices)[np.asarray(mesh.faces).reshape(-1)]
        for label in np.unique(labels):
            if label < 0:
                continue
            part_points = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[labels == label].reshape(-1)]
            minimum, maximum = better_aabb(part_points)
            inside = np.all((flat_points >= minimum) & (flat_points <= maximum), axis=1)
            aabb_face_masks[int(label)] = np.all(inside.reshape(-1, 3), axis=1)

    unlabelled = np.where(labels < 0)[0].tolist()
    face_limit = adjacent_faces.shape[0]
    for _ in range(50):
        changed = False
        remaining: list[int] = []
        for face in unlabelled:
            if not 0 <= face < face_limit:
                continue
            neighbors: list[int] = []
            for adjacent in adjacent_faces[face]:
                if adjacent == -1:
                    break
                label = int(labels[adjacent])
                if label >= 0 and (not use_aabb or aabb_face_masks[label][face]):
                    neighbors.append(label)
            if not neighbors:
                remaining.append(face)
                continue
            labels[face] = int(np.argmax(np.bincount(neighbors)))
            changed = True
        unlabelled = remaining
        if not changed:
            break
    return labels


def connected_regions(
    face_ids: np.ndarray,
    adjacent_faces: np.ndarray,
    *,
    return_face_region_ids: bool = False,
) -> list[list[int]] | tuple[list[list[int]], np.ndarray]:
    """Find edge-connected regions whose faces share the same label."""

    labels = np.asarray(face_ids)
    visited = np.zeros(len(labels), dtype=bool)
    regions: list[list[int]] = []
    face_region_ids = np.full(len(labels), -1, dtype=np.int64)
    for start in range(len(labels)):
        if visited[start]:
            continue
        region: list[int] = []
        queue: deque[int] = deque([start])
        while queue:
            face = queue.popleft()
            if visited[face]:
                continue
            visited[face] = True
            region.append(face)
            face_region_ids[face] = len(regions)
            if not 0 <= face < adjacent_faces.shape[0]:
                continue
            for adjacent in adjacent_faces[face]:
                if adjacent == -1:
                    break
                if not visited[adjacent] and labels[adjacent] == labels[face]:
                    queue.append(int(adjacent))
        regions.append(region)
    if return_face_region_ids:
        return regions, face_region_ids
    return regions


def aabb_distance(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]) -> float:
    minimum_1, maximum_1 = first
    minimum_2, maximum_2 = second
    # Preserve the released formula, including its non-standard overlap
    # behavior, because it controls fallback assignment for detached regions.
    delta = np.asarray(
        [
            max(0, maximum_2[axis] - minimum_1[axis], maximum_1[axis] - minimum_2[axis])
            for axis in range(3)
        ]
    )
    return float(np.linalg.norm(delta))


def aabb_volume(box: tuple[np.ndarray, np.ndarray]) -> float:
    return float(np.prod(box[1] - box[0]))


def _region_aabb(mesh: Any, region: list[int]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[region].reshape(-1)]
    return points.min(axis=0), points.max(axis=0)


def find_neighbor_regions(
    regions: list[list[int]],
    adjacent_faces: np.ndarray,
    *,
    region_aabbs: list[tuple[np.ndarray, np.ndarray]] | None = None,
    region_labels: list[int] | None = None,
) -> list[list[int]]:
    face_to_region = {face: index for index, region in enumerate(regions) for face in region}
    valid_fallback_indices: np.ndarray | None = None
    fallback_minimums: np.ndarray | None = None
    fallback_maximums: np.ndarray | None = None
    fallback_volumes: np.ndarray | None = None
    if region_aabbs is not None and region_labels is not None:
        valid_fallback_indices = np.asarray(
            [index for index, label in enumerate(region_labels) if label not in {-1, -2}],
            dtype=np.int64,
        )
        if len(valid_fallback_indices):
            fallback_minimums = np.stack([region_aabbs[index][0] for index in valid_fallback_indices])
            fallback_maximums = np.stack([region_aabbs[index][1] for index in valid_fallback_indices])
            fallback_volumes = np.prod(fallback_maximums - fallback_minimums, axis=1)
    result: list[list[int]] = []
    for index, region in enumerate(regions):
        neighbors: set[int] = set()
        for face in region:
            if not 0 <= face < adjacent_faces.shape[0]:
                continue
            for adjacent in adjacent_faces[face]:
                if adjacent == -1:
                    break
                target = face_to_region.get(int(adjacent))
                if target is None or target == index:
                    continue
                if region_labels is not None and region_labels[target] in {-1, -2}:
                    continue
                neighbors.add(target)
        ordered = list(neighbors)
        if (
            region_aabbs is not None
            and region_labels is not None
            and region_labels[index] in {-1, -2}
            and not ordered
            and valid_fallback_indices is not None
            and fallback_minimums is not None
            and fallback_maximums is not None
            and fallback_volumes is not None
            and len(valid_fallback_indices)
        ):
            current_minimum, current_maximum = region_aabbs[index]
            delta = np.maximum(
                0,
                np.maximum(fallback_maximums - current_minimum, current_maximum - fallback_minimums),
            )
            distances = np.linalg.norm(delta, axis=1)
            # Official tie order: distance, then smaller AABB volume, then the
            # first region encountered. lexsort preserves exactly that order.
            best_position = int(
                np.lexsort((valid_fallback_indices, fallback_volumes, distances))[0]
            )
            ordered = [int(valid_fallback_indices[best_position])]
        result.append(ordered)
    return result


def merge_small_regions(
    face_areas: np.ndarray,
    regions: list[list[int]],
    adjacent_faces: np.ndarray,
    face_ids: np.ndarray,
    *,
    threshold: float = 0.95,
) -> np.ndarray:
    """Port of the optional official cumulative-area postprocess."""

    total_area = float(np.sum(face_areas))
    relative_areas = [float(np.sum(face_areas[region]) / total_area) for region in regions]
    ordered = sorted(zip(relative_areas, regions, strict=True), key=lambda item: item[0], reverse=True)
    relative_areas = [item[0] for item in ordered]
    regions = [item[1] for item in ordered]
    cumulative = np.cumsum(relative_areas)
    neighbors = find_neighbor_regions(regions, adjacent_faces)
    result = np.asarray(face_ids).copy()
    for index, region in enumerate(regions):
        if cumulative[index] <= threshold or relative_areas[index] >= 0.01:
            continue
        best_area = 0.0
        best = -1
        for target in neighbors[index]:
            if cumulative[target] <= threshold and relative_areas[target] > best_area:
                best_area = relative_areas[target]
                best = target
        if best != -1:
            regions[best].extend(region)
            regions[index] = []
            result[region] = face_ids[regions[best][0]]
    return result


def promote_unlabelled_regions(regions: list[list[int]], face_ids: np.ndarray) -> np.ndarray:
    """Port the official helper, including its label-all-missing-region behavior."""

    labels = np.asarray(face_ids)
    result = labels.copy()
    max_id = int(np.max(np.unique(labels)))
    for region in regions:
        if labels[region[0]] in {-1, -2}:
            max_id += 1
            result[region] = max_id
    return result


def project_sample_labels_to_faces(
    face_indices: np.ndarray,
    point_ids: np.ndarray,
    face_count: int,
) -> np.ndarray:
    """Vote sampled labels directly onto their source faces like ``auto_mask.py``."""

    faces = np.asarray(face_indices, dtype=np.int64)
    labels = np.asarray(point_ids, dtype=np.int64)
    if faces.shape != labels.shape:
        raise ValueError(f"face_indices and point_ids must align: {faces.shape}, {labels.shape}")
    result = np.full(face_count, -2, dtype=np.int64)
    order = np.argsort(faces, kind="stable")
    sorted_faces = faces[order]
    sorted_labels = labels[order]
    starts = np.flatnonzero(np.r_[True, sorted_faces[1:] != sorted_faces[:-1]])
    stops = np.r_[starts[1:], len(sorted_faces)]
    for start, stop in zip(starts, stops, strict=True):
        face = int(sorted_faces[start])
        result[face] = int(np.argmax(np.bincount(sorted_labels[start:stop] + 2))) - 2
    return result


def _aabb_increase(
    target: tuple[np.ndarray, np.ndarray],
    addition: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    before_minimum, before_maximum = target
    after_minimum = np.minimum(before_minimum, addition[0])
    after_maximum = np.maximum(before_maximum, addition[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        minimum_increase = np.abs(after_minimum - before_minimum) / np.abs(before_minimum)
        maximum_increase = np.abs(after_maximum - before_maximum) / np.abs(before_maximum)
    return minimum_increase, maximum_increase


def official_mesh_postprocess(
    mesh: Any,
    face_indices: np.ndarray,
    point_ids: np.ndarray,
    *,
    threshold: float = 0.95,
    postprocess: bool = True,
) -> OfficialPostprocessResult:
    """Execute the complete released ``auto_mask.py`` topology pipeline."""

    adjacent = build_adjacent_faces(np.asarray(mesh.face_adjacency), len(mesh.faces))
    projected = project_sample_labels_to_faces(face_indices, point_ids, len(mesh.faces))

    # The first +1/-1 shift deliberately fills only faces never surface-sampled
    # (-2), while retaining model-uncovered faces (-1), exactly as released.
    first_filled = fix_labels(projected.copy() + 1, adjacent, mesh=mesh) - 1
    face_areas = np.asarray(mesh.area_faces)
    total_area = float(np.sum(face_areas))
    regions = connected_regions(first_filled, adjacent)
    connected_components, component_ids = connected_regions(
        np.ones_like(first_filled), adjacent, return_face_region_ids=True
    )
    component_areas = [float(np.sum(face_areas[region])) for region in connected_components]
    region_rows = [
        (region, float(np.sum(face_areas[region])), component_areas[int(component_ids[region[0]])])
        for region in regions
    ]
    region_rows.sort(key=lambda row: row[1], reverse=True)

    filtered = first_filled.copy()
    for region, area, component_area in region_rows:
        if area / (component_area + 1e-7) <= 0.001:
            filtered[region] = -1
    filled = fix_labels(filtered.copy(), adjacent, mesh=mesh)

    regions_2 = connected_regions(filled, adjacent)
    region_areas = [float(np.sum(face_areas[region])) for region in regions_2]
    region_labels = [int(filled[region[0]]) for region in regions_2]
    max_id = max(region_labels)
    for index, (area, label) in enumerate(zip(region_areas, region_labels, strict=True)):
        if area / total_area > 0.001 and label in {-1, -2}:
            max_id += 1
            region_labels[index] = max_id

    assigned = filled.copy()
    for region, label in zip(regions_2, region_labels, strict=True):
        assigned[region] = label

    id_aabbs = {
        int(label): _region_aabb(mesh, np.where(assigned == label)[0].tolist())
        for label in np.unique(assigned)
        if label >= 0
    }
    region_aabbs = [_region_aabb(mesh, region) for region in regions_2]
    neighbors = find_neighbor_regions(
        regions_2,
        adjacent,
        region_aabbs=region_aabbs,
        region_labels=region_labels,
    )
    for index, label in enumerate(region_labels):
        if label not in {-1, -2}:
            continue
        best_label = -1
        best_increase = 1e10
        for target in neighbors[index]:
            target_label = region_labels[target]
            if target_label in {-1, -2}:
                continue
            minimum, maximum = _aabb_increase(id_aabbs[target_label], region_aabbs[index])
            increase = float(max(np.max(minimum), np.max(maximum)))
            if increase < best_increase:
                best_increase = increase
                best_label = target_label
        if best_label >= 0:
            region_labels[index] = best_label

    connectivity = filled.copy()
    for region, label in zip(regions_2, region_labels, strict=True):
        connectivity[region] = label

    final = connectivity.copy()
    if postprocess:
        # Keep the released call ordering. ``promote_unlabelled_regions`` is
        # observable only through the helper API because the release then
        # overwrites it with ``do_post_process(face_ids_4)``.
        post_regions = connected_regions(connectivity, adjacent)
        promote_unlabelled_regions(post_regions, connectivity)
        final = merge_small_regions(
            face_areas,
            post_regions,
            adjacent,
            connectivity,
            threshold=threshold,
        )
    return OfficialPostprocessResult(
        mesh=mesh,
        projected_face_ids=projected,
        connectivity_face_ids=connectivity,
        final_face_ids=final,
    )
