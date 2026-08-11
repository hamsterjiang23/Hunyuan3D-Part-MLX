from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SurfaceMetrics:
    chamfer_distance: float
    fscore_01: float
    fscore_005: float
    rotation_degrees: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def instance_miou(predicted_labels: np.ndarray, target_labels: np.ndarray, *, ignore_label: int = -1) -> float:
    """PartObjaverse-Tiny instance mIoU used by SAMPart3D/PartField.

    Each ground-truth part is matched independently to its best predicted mask;
    the returned value is a percentage in ``[0, 100]``.
    """

    predicted_labels = np.asarray(predicted_labels)
    target_labels = np.asarray(target_labels)
    if predicted_labels.shape != target_labels.shape:
        raise ValueError(f"label shapes differ: {predicted_labels.shape} != {target_labels.shape}")
    predicted_masks = [predicted_labels == label for label in np.unique(predicted_labels) if label != ignore_label]
    best_ious = []
    for label in np.unique(target_labels):
        if label == ignore_label:
            continue
        target_mask = target_labels == label
        best_ious.append(max((mask_iou(mask, target_mask) for mask in predicted_masks), default=0.0))
    return 0.0 if not best_ious else float(np.mean(best_ious) * 100)


def mask_iou(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if predicted.shape != target.shape:
        raise ValueError(f"mask shapes differ: {predicted.shape} != {target.shape}")
    union = np.logical_or(predicted, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(predicted, target).sum() / union)


def normalize_to_unit_cube(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"points must have shape [N, 3] with N > 0, got {points.shape}")
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2
    scale = float(np.max((maximum - minimum) / 2))
    if scale == 0:
        raise ValueError("cannot normalize a degenerate point cloud")
    return (points - center) / scale


def rotate_quarter_turns(
    points: np.ndarray,
    degrees: int,
    *,
    axis: Literal["x", "y", "z"] = "z",
) -> np.ndarray:
    if degrees not in (0, 90, 180, 270):
        raise ValueError(f"rotation must be a quarter turn, got {degrees}")
    radians = np.deg2rad(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    if axis == "x":
        rotation = np.asarray([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])
    elif axis == "y":
        rotation = np.asarray([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    elif axis == "z":
        rotation = np.asarray([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    else:
        raise ValueError(f"unsupported rotation axis: {axis}")
    return np.asarray(points) @ rotation.T


def surface_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    thresholds: tuple[float, float] = (0.1, 0.05),
    rotation_degrees: int = 0,
) -> SurfaceMetrics:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape[1] != 3 or len(predicted) == 0:
        raise ValueError(f"predicted must have shape [N, 3] with N > 0, got {predicted.shape}")
    if target.ndim != 2 or target.shape[1] != 3 or len(target) == 0:
        raise ValueError(f"target must have shape [M, 3] with M > 0, got {target.shape}")
    predicted_to_target = cKDTree(target).query(predicted, workers=-1)[0]
    target_to_predicted = cKDTree(predicted).query(target, workers=-1)[0]
    chamfer = float((predicted_to_target.mean() + target_to_predicted.mean()) / 2)
    fscores = []
    for threshold in thresholds:
        precision = float(np.mean(predicted_to_target <= threshold))
        recall = float(np.mean(target_to_predicted <= threshold))
        fscores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return SurfaceMetrics(
        chamfer_distance=chamfer,
        fscore_01=fscores[0],
        fscore_005=fscores[1],
        rotation_degrees=rotation_degrees,
    )


def best_rotated_surface_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    axis: Literal["x", "y", "z"] = "z",
) -> SurfaceMetrics:
    normalized_predicted = normalize_to_unit_cube(predicted)
    normalized_target = normalize_to_unit_cube(target)
    candidates = [
        surface_metrics(
            rotate_quarter_turns(normalized_predicted, degrees, axis=axis),
            normalized_target,
            rotation_degrees=degrees,
        )
        for degrees in (0, 90, 180, 270)
    ]
    return min(candidates, key=lambda metric: metric.chamfer_distance)
