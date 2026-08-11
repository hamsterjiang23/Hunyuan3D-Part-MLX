from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np
from scipy.spatial import cKDTree

from split3d.hunyuan.sonata_data import official_fps_start_index


class P3SAMBackend(Protocol):
    def extract_features(self, points: np.ndarray, normals: np.ndarray | None = None, *, seed: int = 42) -> Any: ...

    def predict_masks(
        self,
        features: Any,
        points: np.ndarray,
        prompts: np.ndarray,
        *,
        iterations: int = 1,
        prompt_batch_size: int = 32,
    ) -> Any: ...


@dataclass(frozen=True)
class SegmentationDiagnostics:
    point_count: int
    prompt_count: int
    initial_cluster_count: int
    stable_cluster_count: int
    final_part_count: int
    uncovered_fraction: float


@dataclass(frozen=True)
class MeshSegmentation:
    face_ids: np.ndarray
    bboxes: np.ndarray
    diagnostics: SegmentationDiagnostics
    stage_seconds: dict[str, float] = field(default_factory=dict)


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (maximum + minimum) / 2
    scale = float(np.max(np.abs((maximum - minimum) / 2)))
    return (points - center) / (scale + 1e-10)


def farthest_point_indices(points: np.ndarray, count: int, *, start_index: int = 0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    if not 0 < count <= len(points):
        raise ValueError(f"count must be in [1, {len(points)}], got {count}")
    if not 0 <= start_index < len(points):
        raise ValueError(f"start_index is out of range: {start_index}")
    selected = np.empty(count, dtype=np.int64)
    distances = np.full(len(points), np.inf, dtype=np.float32)
    current = start_index
    for index in range(count):
        selected[index] = current
        delta = points - points[current]
        distances = np.minimum(distances, np.einsum("nd,nd->n", delta, delta))
        current = int(np.argmax(distances))
    return selected


def _bbox_iou(points: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    first_points = points[first]
    second_points = points[second]
    if len(first_points) == 0 or len(second_points) == 0:
        return 0.0
    first_min, first_max = first_points.min(axis=0), first_points.max(axis=0)
    second_min, second_max = second_points.min(axis=0), second_points.max(axis=0)
    intersection = np.maximum(0, np.minimum(first_max, second_max) - np.maximum(first_min, second_min))
    intersection_volume = float(np.prod(intersection))
    first_volume = float(np.prod(first_max - first_min))
    second_volume = float(np.prod(second_max - second_min))
    union = first_volume + second_volume - intersection_volume
    return 0.0 if union <= 0 else intersection_volume / union


def select_automatic_masks(
    points: np.ndarray,
    candidate_masks: np.ndarray,
    predicted_iou: np.ndarray,
    *,
    nms_threshold: float = 0.9,
    minimum_cluster_prompts: int = 3,
    bbox_merge_threshold: float = 0.5,
    uncovered_acceptance: float = 0.7,
    nms_workers: int = 20,
) -> tuple[np.ndarray, SegmentationDiagnostics]:
    masks = np.asarray(candidate_masks, dtype=bool)
    scores = np.asarray(predicted_iou, dtype=np.float32)
    if masks.ndim != 2 or masks.shape[1] != len(scores):
        raise ValueError(f"candidate masks and scores do not align: {masks.shape}, {scores.shape}")
    sorted_indices = np.argsort(-scores, kind="stable")
    sorted_masks = masks[:, sorted_indices]

    # Pack point masks so each NMS comparison scans one bit per point instead
    # of allocating a pair of N-element boolean temporaries. This preserves
    # the official greedy representative order while avoiding pathological
    # runtimes when many masks survive early comparisons.
    del nms_workers  # Retained for API compatibility with older callers.
    packed_masks = np.packbits(sorted_masks, axis=0, bitorder="little")
    mask_areas = sorted_masks.sum(axis=0, dtype=np.int64)
    popcount = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)
    representatives: list[int] = []
    clusters: dict[int, list[int]] = {}
    for candidate in range(sorted_masks.shape[1]):
        if representatives:
            intersections = popcount[
                np.bitwise_and(packed_masks[:, candidate, None], packed_masks[:, representatives])
            ].sum(axis=0, dtype=np.int64)
            unions = mask_areas[candidate] + mask_areas[representatives] - intersections
            overlaps = np.divide(
                intersections,
                unions,
                out=np.ones(len(representatives), dtype=np.float64),
                where=unions != 0,
            )
            matches = np.flatnonzero(overlaps > nms_threshold)
            if len(matches):
                representative = representatives[int(matches[0])]
                clusters[representative].append(candidate)
                continue
        representatives.append(candidate)
        clusters[candidate] = [candidate]

    stable = [
        representative for representative in representatives if len(clusters[representative]) >= minimum_cluster_prompts
    ]
    merged: list[int] = []
    consumed: set[int] = set()
    for index, representative in enumerate(stable):
        if representative in consumed:
            continue
        merged.append(representative)
        for target in stable[index + 1 :]:
            if (
                target not in consumed
                and _bbox_iou(points, sorted_masks[:, target], sorted_masks[:, representative]) > bbox_merge_threshold
            ):
                consumed.add(target)

    uncovered = np.ones(len(points), dtype=bool)
    for representative in merged:
        uncovered[sorted_masks[:, representative]] = False
    for candidate in range(sorted_masks.shape[1]):
        if candidate in merged:
            continue
        candidate_area = int(sorted_masks[:, candidate].sum())
        if (
            candidate_area
            and np.logical_and(sorted_masks[:, candidate], uncovered).sum() / candidate_area > uncovered_acceptance
        ):
            merged.append(candidate)
            uncovered[sorted_masks[:, candidate]] = False

    merged.sort(key=lambda index: int(sorted_masks[:, index].sum()), reverse=True)
    point_ids = np.full(len(points), -1, dtype=np.int64)
    for part_id, representative in enumerate(merged):
        point_ids[sorted_masks[:, representative]] = part_id
    diagnostics = SegmentationDiagnostics(
        point_count=len(points),
        prompt_count=masks.shape[1],
        initial_cluster_count=len(representatives),
        stable_cluster_count=len(stable),
        final_part_count=len(merged),
        uncovered_fraction=float(np.mean(point_ids < 0)),
    )
    return point_ids, diagnostics


def _sample_each_face(mesh: Any, count: int, rng: np.random.Generator) -> np.ndarray:
    face_count = len(mesh.faces)
    u = np.sqrt(rng.random((face_count, count, 1)))
    v = rng.random((face_count, count, 1))
    weights = np.concatenate([1 - u, u * (1 - v), u * v], axis=-1)
    vertices = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    return np.sum(vertices[:, None, :, :] * weights[..., None], axis=2)


def point_labels_to_faces(
    mesh: Any,
    sampled_points: np.ndarray,
    point_ids: np.ndarray,
    *,
    samples_per_face: int = 10,
    seed: int = 42,
) -> np.ndarray:
    valid = point_ids >= 0
    if not np.any(valid):
        return np.full(len(mesh.faces), -1, dtype=np.int64)
    face_points = _sample_each_face(mesh, samples_per_face, np.random.default_rng(seed))
    nearest = cKDTree(sampled_points[valid]).query(face_points.reshape(-1, 3), workers=-1)[1]
    nearest_labels = point_ids[valid][nearest].reshape(len(mesh.faces), samples_per_face)
    return np.asarray([np.bincount(labels).argmax() for labels in nearest_labels], dtype=np.int64)


def face_bboxes(mesh: Any, face_ids: np.ndarray) -> np.ndarray:
    boxes = []
    for part_id in np.unique(face_ids):
        if part_id < 0:
            continue
        vertices = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[face_ids == part_id]].reshape(-1, 3)
        boxes.append([vertices.min(axis=0), vertices.max(axis=0)])
    return np.asarray(boxes, dtype=np.float32)


def segment_mesh(
    model: P3SAMBackend,
    mesh: Any,
    *,
    point_count: int = 100_000,
    prompt_count: int = 400,
    prompt_batch_size: int = 32,
    seed: int = 42,
    prompt_start_index: int | None = 0,
    progress: Callable[[str, float], None] | None = None,
) -> MeshSegmentation:
    try:
        import trimesh
    except ImportError as error:  # pragma: no cover - optional Mac runtime dependencies
        raise RuntimeError("segment_mesh requires trimesh") from error
    stage_seconds: dict[str, float] = {}

    def record_stage(name: str, started: float) -> float:
        elapsed = perf_counter() - started
        stage_seconds[name] = elapsed
        if progress is not None:
            progress(name, elapsed)
        return perf_counter()

    stage_started = perf_counter()
    sampled_points, face_indices = trimesh.sample.sample_surface(mesh, point_count, seed=seed)
    normalized_points = normalize_point_cloud(sampled_points)
    normals = np.asarray(mesh.face_normals)[face_indices]
    stage_started = record_stage("sample_surface", stage_started)
    features = model.extract_features(normalized_points, normals, seed=seed)
    stage_started = record_stage("extract_features", stage_started)
    if prompt_start_index is None:
        prompt_start_index = official_fps_start_index(normalized_points, seed=seed)
    prompt_indices = farthest_point_indices(
        normalized_points,
        prompt_count,
        start_index=prompt_start_index,
    )
    prompts = normalized_points[prompt_indices]
    stage_started = record_stage("sample_prompts", stage_started)
    predictions = model.predict_masks(
        features,
        normalized_points,
        prompts,
        prompt_batch_size=prompt_batch_size,
    )
    stage_started = record_stage("predict_masks", stage_started)
    probabilities = np.stack([np.asarray(mask).T for mask in predictions.masks], axis=-1)
    iou = np.asarray(predictions.predicted_iou)
    best_head = np.argmax(iou, axis=-1)
    candidate_masks = np.stack(
        [probabilities[:, prompt, best_head[prompt]] > 0.5 for prompt in range(prompt_count)],
        axis=1,
    )
    candidate_iou = iou[np.arange(prompt_count), best_head]
    point_ids, diagnostics = select_automatic_masks(normalized_points, candidate_masks, candidate_iou)
    stage_started = record_stage("select_masks", stage_started)
    face_ids = point_labels_to_faces(mesh, sampled_points, point_ids, seed=seed)
    bboxes = face_bboxes(mesh, face_ids)
    record_stage("project_faces", stage_started)
    return MeshSegmentation(
        face_ids=face_ids,
        bboxes=bboxes,
        diagnostics=diagnostics,
        stage_seconds=stage_seconds,
    )


segment_mesh_mlx = segment_mesh


def save_segmentation(
    mesh: Any,
    result: MeshSegmentation,
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    colors = rng.integers(32, 256, size=(max(1, result.diagnostics.final_part_count), 3), dtype=np.uint8)
    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    face_colors[:, 3] = 255
    valid = result.face_ids >= 0
    face_colors[valid, :3] = colors[result.face_ids[valid]]
    colored = mesh.copy()
    colored.visual.face_colors = face_colors
    colored.export(output_dir / "segmented.glb")
    np.save(output_dir / "face_ids.npy", result.face_ids)
    np.save(output_dir / "bboxes.npy", result.bboxes)
    (output_dir / "diagnostics.json").write_text(
        json.dumps(asdict(result.diagnostics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
