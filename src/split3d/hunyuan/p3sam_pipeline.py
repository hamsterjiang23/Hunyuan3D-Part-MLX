from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from split3d.hunyuan.p3sam_official_postprocess import clean_mesh_official, official_mesh_postprocess
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
        prompt_batch_size: int = 1,
    ) -> Any: ...


@dataclass(frozen=True)
class SegmentationDiagnostics:
    point_count: int
    prompt_count: int
    initial_cluster_count: int
    stable_cluster_count: int
    final_part_count: int
    uncovered_fraction: float
    projected_part_count: int = 0
    connectivity_part_count: int = 0
    postprocessed_part_count: int = 0


@dataclass(frozen=True)
class MeshSegmentation:
    face_ids: np.ndarray
    bboxes: np.ndarray
    diagnostics: SegmentationDiagnostics
    stage_seconds: dict[str, float] = field(default_factory=dict)
    mesh: Any | None = None
    projected_face_ids: np.ndarray | None = None
    connectivity_face_ids: np.ndarray | None = None


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    # Keep trimesh's float64 surface samples through normalization.  The
    # released P3-SAM demo computes its bounds in float64, runs Sonata's
    # CenterShift/GridSample in NumPy, and only casts to float32 in ToTensor.
    # Casting here changes voxel membership for boundary points and therefore
    # changes both Sonata representatives and the subsequent random FPS start.
    points = np.asarray(points)
    if not np.issubdtype(points.dtype, np.floating):
        points = points.astype(np.float64)
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
                # The released scalar implementation produces NaN for two
                # empty masks; NaN never passes the >0.9 NMS comparison.
                out=np.full(len(representatives), np.nan, dtype=np.float64),
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
    # Preserve the released mask IDs (indices after predicted-IoU sorting).
    # They are intentionally sparse and determine tie-breaking during face votes.
    for representative in merged:
        point_ids[sorted_masks[:, representative]] = representative
    diagnostics = SegmentationDiagnostics(
        point_count=len(points),
        prompt_count=masks.shape[1],
        initial_cluster_count=len(representatives),
        stable_cluster_count=len(stable),
        final_part_count=len(merged),
        uncovered_fraction=float(np.mean(point_ids < 0)),
    )
    return point_ids, diagnostics


def face_bboxes(mesh: Any, face_ids: np.ndarray) -> np.ndarray:
    boxes = []
    for part_id in np.unique(face_ids):
        if part_id < 0:
            continue
        vertices = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[face_ids == part_id]].reshape(-1, 3)
        boxes.append([vertices.min(axis=0), vertices.max(axis=0)])
    return np.asarray(boxes, dtype=np.float32)


def _mesh_geometry_hash(mesh: Any) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.vertices).view(np.uint8))
    digest.update(np.ascontiguousarray(mesh.faces).view(np.uint8))
    return digest.hexdigest()


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def segment_mesh(
    model: P3SAMBackend,
    mesh: Any,
    *,
    point_count: int = 100_000,
    prompt_count: int = 400,
    prompt_batch_size: int = 1,
    seed: int = 42,
    prompt_start_index: int | None = None,
    clean_mesh: bool = True,
    connectivity: bool = True,
    postprocess: bool = True,
    postprocess_threshold: float = 0.95,
    replay_manifest: str | Path | None = None,
    trace_dir: str | Path | None = None,
    trace_full_tensors: bool = False,
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
    processed_mesh = clean_mesh_official(mesh) if clean_mesh else mesh.copy()
    stage_started = record_stage("clean_mesh", stage_started)
    geometry_hash = _mesh_geometry_hash(processed_mesh)
    loaded_prompt_indices: np.ndarray | None = None
    if replay_manifest is None:
        sampled_points, face_indices = trimesh.sample.sample_surface(processed_mesh, point_count, seed=seed)
        normalized_points = normalize_point_cloud(sampled_points)
        normals = np.asarray(processed_mesh.face_normals)[face_indices]
    else:
        with np.load(replay_manifest, allow_pickle=False) as replay:
            replay_hash = str(replay["mesh_geometry_hash"].item())
            replay_seed = int(replay["seed"].item())
            if replay_hash != geometry_hash:
                raise ValueError("replay manifest mesh geometry does not match the cleaned input mesh")
            if replay_seed != seed:
                raise ValueError(f"replay manifest seed is {replay_seed}, requested seed is {seed}")
            sampled_points = replay["sampled_points"]
            normalized_points = replay["normalized_points"]
            normals = replay["normals"]
            face_indices = replay["face_indices"]
            loaded_prompt_indices = replay["prompt_indices"]
        if len(sampled_points) != point_count:
            raise ValueError(f"replay manifest has {len(sampled_points)} points, requested {point_count}")
        if len(loaded_prompt_indices) != prompt_count:
            raise ValueError(f"replay manifest has {len(loaded_prompt_indices)} prompts, requested {prompt_count}")
    stage_started = record_stage("sample_surface", stage_started)
    features = model.extract_features(normalized_points, normals, seed=seed)
    stage_started = record_stage("extract_features", stage_started)
    if loaded_prompt_indices is not None:
        prompt_indices = loaded_prompt_indices
    else:
        if prompt_start_index is None:
            prompt_start_index = official_fps_start_index(normalized_points, seed=seed)
        prompt_indices = farthest_point_indices(
            normalized_points,
            prompt_count,
            start_index=prompt_start_index,
        )
    prompts = normalized_points[prompt_indices]
    stage_started = record_stage("sample_prompts", stage_started)
    trace_path = Path(trace_dir) if trace_dir is not None else None
    if trace_path is not None:
        trace_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trace_path / "replay_manifest.npz",
            mesh_geometry_hash=np.asarray(geometry_hash),
            seed=np.asarray(seed, dtype=np.int64),
            sampled_points=np.asarray(sampled_points),
            normalized_points=np.asarray(normalized_points),
            normals=np.asarray(normals),
            face_indices=np.asarray(face_indices),
            prompt_indices=np.asarray(prompt_indices),
        )
        if trace_full_tensors:
            np.save(trace_path / "features.npy", _tensor_to_numpy(features))
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
    if trace_path is not None:
        np.save(trace_path / "predicted_iou.npy", iou)
        np.save(trace_path / "candidate_iou.npy", candidate_iou)
        np.save(trace_path / "candidate_masks.packbits.npy", np.packbits(candidate_masks, axis=0, bitorder="little"))
        (trace_path / "candidate_masks.json").write_text(
            json.dumps({"shape": list(candidate_masks.shape), "bitorder": "little"}, indent=2),
            encoding="utf-8",
        )
        if trace_full_tensors:
            np.save(trace_path / "mask_probabilities.npy", probabilities)
    point_ids, diagnostics = select_automatic_masks(normalized_points, candidate_masks, candidate_iou)
    stage_started = record_stage("select_masks", stage_started)
    topology = official_mesh_postprocess(
        processed_mesh,
        face_indices,
        point_ids,
        threshold=postprocess_threshold,
        postprocess=postprocess,
    )
    if not connectivity:
        face_ids = topology.projected_face_ids
    elif postprocess:
        face_ids = topology.final_face_ids
    else:
        face_ids = topology.connectivity_face_ids
    bboxes = face_bboxes(processed_mesh, face_ids)
    record_stage("official_mesh_postprocess", stage_started)
    projected_count = int(np.sum(np.unique(topology.projected_face_ids) >= 0))
    connectivity_count = int(np.sum(np.unique(topology.connectivity_face_ids) >= 0))
    postprocessed_count = int(np.sum(np.unique(topology.final_face_ids) >= 0))
    if trace_path is not None:
        np.save(trace_path / "point_ids.npy", point_ids)
        np.save(trace_path / "face_ids_projected.npy", topology.projected_face_ids)
        np.save(trace_path / "face_ids_connectivity.npy", topology.connectivity_face_ids)
        np.save(trace_path / "face_ids_final.npy", topology.final_face_ids)
    diagnostics = SegmentationDiagnostics(
        point_count=diagnostics.point_count,
        prompt_count=diagnostics.prompt_count,
        initial_cluster_count=diagnostics.initial_cluster_count,
        stable_cluster_count=diagnostics.stable_cluster_count,
        final_part_count=postprocessed_count if connectivity and postprocess else (
            connectivity_count if connectivity else projected_count
        ),
        uncovered_fraction=diagnostics.uncovered_fraction,
        projected_part_count=projected_count,
        connectivity_part_count=connectivity_count,
        postprocessed_part_count=postprocessed_count,
    )
    return MeshSegmentation(
        face_ids=face_ids,
        bboxes=bboxes,
        diagnostics=diagnostics,
        stage_seconds=stage_seconds,
        mesh=processed_mesh,
        projected_face_ids=topology.projected_face_ids,
        connectivity_face_ids=topology.connectivity_face_ids,
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
    import trimesh

    output_mesh = result.mesh if result.mesh is not None else mesh
    rng = np.random.default_rng(seed)
    label_arrays = [result.face_ids]
    if result.projected_face_ids is not None:
        label_arrays.append(result.projected_face_ids)
    if result.connectivity_face_ids is not None:
        label_arrays.append(result.connectivity_face_ids)
    labels = sorted(int(label) for label in np.unique(np.concatenate(label_arrays)) if label >= 0)
    colors = {label: rng.integers(32, 256, size=3, dtype=np.uint8) for label in labels}

    def save_colored(face_ids: np.ndarray, path: Path) -> None:
        face_colors = np.zeros((len(output_mesh.faces), 4), dtype=np.uint8)
        face_colors[:, 3] = 255
        for label, color in colors.items():
            face_colors[face_ids == label, :3] = color
        colored = trimesh.Trimesh(
            vertices=np.asarray(output_mesh.vertices),
            faces=np.asarray(output_mesh.faces),
            face_colors=face_colors,
            process=False,
        )
        colored.export(path)

    save_colored(result.face_ids, output_dir / "segmented.glb")
    save_colored(result.face_ids, output_dir / "segmented.ply")
    if result.projected_face_ids is not None:
        save_colored(result.projected_face_ids, output_dir / "segmented_projected.glb")
        np.save(output_dir / "face_ids_projected.npy", result.projected_face_ids)
    if result.connectivity_face_ids is not None:
        save_colored(result.connectivity_face_ids, output_dir / "segmented_connectivity.glb")
        np.save(output_dir / "face_ids_connectivity.npy", result.connectivity_face_ids)
    np.save(output_dir / "face_ids.npy", result.face_ids)
    np.save(output_dir / "bboxes.npy", result.bboxes)
    aabb_scene = trimesh.Scene()
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(exist_ok=True)
    part_manifest = []
    for index, label in enumerate(int(value) for value in np.unique(result.face_ids) if value >= 0):
        face_indices = np.where(result.face_ids == label)[0]
        part = output_mesh.submesh([face_indices], append=True, repair=False)
        part.visual.face_colors = np.tile(np.r_[colors[label], 255], (len(part.faces), 1))
        filename = f"part_{index:03d}_label_{label}.glb"
        part.export(parts_dir / filename)
        points = np.asarray(output_mesh.vertices)[np.asarray(output_mesh.faces)[face_indices].reshape(-1)]
        minimum, maximum = points.min(axis=0), points.max(axis=0)
        center, size = (minimum + maximum) / 2, maximum - minimum
        outline = trimesh.path.creation.box_outline()
        outline.vertices *= size
        outline.vertices += center
        outline.colors = np.tile(np.r_[colors[label], 255], (len(outline.entities), 1)).astype(np.uint8)
        aabb_scene.add_geometry(outline)
        part_manifest.append(
            {
                "part_index": index,
                "label": label,
                "face_count": int(len(face_indices)),
                "bbox": [minimum.tolist(), maximum.tolist()],
                "file": f"parts/{filename}",
            }
        )
    aabb_scene.export(output_dir / "segmented_aabb.glb")
    (output_dir / "parts.json").write_text(
        json.dumps(part_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "diagnostics.json").write_text(
        json.dumps(asdict(result.diagnostics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
