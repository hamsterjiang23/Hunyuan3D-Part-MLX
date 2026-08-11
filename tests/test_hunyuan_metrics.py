from __future__ import annotations

import numpy as np

from split3d.hunyuan.metrics import best_rotated_surface_metrics, instance_miou, mask_iou, surface_metrics


def test_mask_iou_handles_regular_and_empty_masks() -> None:
    assert mask_iou([True, True, False], [True, False, True]) == 1 / 3
    assert mask_iou([False, False], [False, False]) == 1.0


def test_instance_miou_matches_each_ground_truth_part_to_best_prediction() -> None:
    target = np.asarray([0, 0, 1, 1, 2, 2])
    predicted = np.asarray([4, 4, 5, 5, 5, 5])

    assert np.isclose(instance_miou(predicted, target), (100 + 50 + 50) / 3)


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
