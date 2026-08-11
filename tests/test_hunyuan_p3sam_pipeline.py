from __future__ import annotations

import numpy as np

from split3d.hunyuan.p3sam_pipeline import (
    farthest_point_indices,
    normalize_point_cloud,
    select_automatic_masks,
)


def _reference_select(points: np.ndarray, masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    sorted_masks = masks[:, order]
    representatives: list[int] = []
    clusters: dict[int, list[int]] = {}
    for candidate in range(sorted_masks.shape[1]):
        for representative in representatives:
            union = np.logical_or(sorted_masks[:, candidate], sorted_masks[:, representative]).sum()
            # The released scalar implementation yields NaN for two empty
            # masks, so they remain distinct NMS representatives.
            overlap = np.nan if union == 0 else (
                np.logical_and(sorted_masks[:, candidate], sorted_masks[:, representative]).sum() / union
            )
            if overlap > 0.9:
                clusters[representative].append(candidate)
                break
        else:
            representatives.append(candidate)
            clusters[candidate] = [candidate]
    return np.asarray([len(clusters[index]) for index in representatives])


def test_packed_nms_matches_boolean_reference() -> None:
    rng = np.random.default_rng(7)
    points = rng.normal(size=(257, 3)).astype(np.float32)
    masks = rng.random((257, 23)) > 0.72
    masks[:, 1] = masks[:, 0]
    masks[:, 3:6] = False
    scores = rng.random(23).astype(np.float32)
    _, diagnostics = select_automatic_masks(points, masks, scores, minimum_cluster_prompts=1)
    reference_sizes = _reference_select(points, masks, scores)
    assert diagnostics.initial_cluster_count == len(reference_sizes)


def test_normalize_point_cloud_fits_minus_one_to_one() -> None:
    points = np.asarray([[0, 0, 0], [2, 1, 0]], dtype=np.float32)
    normalized = normalize_point_cloud(points)
    np.testing.assert_allclose(normalized, [[-1, -0.5, 0], [1, 0.5, 0]])


def test_normalize_point_cloud_preserves_float64_for_official_voxelization() -> None:
    points = np.asarray([[1e-10, 0, 0], [1.0, 0.5, 0]], dtype=np.float64)
    normalized = normalize_point_cloud(points)

    assert normalized.dtype == np.float64


def test_farthest_point_indices_start_at_zero_and_spread_out() -> None:
    points = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)

    np.testing.assert_array_equal(farthest_point_indices(points, 3), [0, 3, 1])


def test_automatic_mask_selection_deduplicates_and_assigns_parts() -> None:
    points = np.asarray([[0, 0, 0], [0.1, 0, 0], [1, 0, 0], [1.1, 0, 0]], dtype=np.float32)
    masks = np.asarray(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )
    point_ids, diagnostics = select_automatic_masks(points, masks, np.linspace(1, 0.5, 6))

    assert diagnostics.initial_cluster_count == 2
    assert diagnostics.final_part_count == 2
    assert diagnostics.uncovered_fraction == 0
    assert point_ids[0] == point_ids[1]
    assert point_ids[2] == point_ids[3]
    assert point_ids[0] != point_ids[2]
