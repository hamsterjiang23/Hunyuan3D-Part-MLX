from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from split3d.hunyuan.metrics import best_rotated_surface_metrics, instance_miou, mask_iou, surface_metrics


def test_mask_iou_handles_regular_and_empty_masks() -> None:
    assert mask_iou([True, True, False], [True, False, True]) == 1 / 3
    assert mask_iou([False, False], [False, False]) == 1.0


def test_instance_miou_matches_each_ground_truth_part_to_best_prediction() -> None:
    target = np.asarray([0, 0, 1, 1, 2, 2])
    predicted = np.asarray([4, 4, 5, 5, 5, 5])

    assert np.isclose(instance_miou(predicted, target), (100 + 50 + 50) / 3)


def test_instance_miou_matches_official_negative_label_semantics() -> None:
    target = np.asarray([0, 0, 1, 1, -1])
    predicted = np.asarray([-1, -1, 3, 3, 7])

    assert instance_miou(predicted, target) == 100


def test_instance_miou_matches_pinned_official_evaluator() -> None:
    path = Path(__file__).parents[1] / ".upstream/PartField/compute_metric.py"
    if not path.is_file():
        pytest.skip("pinned PartField evaluator is unavailable")
    spec = importlib.util.spec_from_file_location("official_partfield_metric", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = np.asarray([0, 0, 1, 1, 2, 2, -1])
    predicted = np.asarray([-1, -1, 4, 4, -2, 8, 8])
    predicted_masks = np.asarray([predicted == label for label in np.unique(predicted)])

    expected = module.eval_single_gt_shape(target, predicted_masks)
    assert np.isclose(instance_miou(predicted, target), expected)


def test_surface_metrics_are_perfect_for_identical_points() -> None:
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    metrics = surface_metrics(points, points)

    assert metrics.chamfer_distance == 0
    assert metrics.fscore_01 == 1
    assert metrics.fscore_005 == 1


def test_best_rotation_recovers_quarter_turn() -> None:
    target = np.asarray([[0, 0, 0], [2, 0, 0], [0, 1, 0], [0.2, 0.3, 1]], dtype=np.float64)
    predicted = target @ np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64).T
    metrics = best_rotated_surface_metrics(predicted, target)

    assert metrics.chamfer_distance < 1e-12
    assert metrics.fscore_01 == 1
    assert metrics.fscore_005 == 1
