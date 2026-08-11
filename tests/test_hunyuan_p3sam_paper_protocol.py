from __future__ import annotations

import numpy as np
import pytest

from split3d.hunyuan.p3sam_paper_protocol import (
    interactive_prompt_miou,
    sample_part_prompt_indices,
    select_joint_prompt_masks,
)


def test_sample_part_prompts_is_reproducible_and_covers_every_part() -> None:
    face_indices = np.asarray([0, 0, 1, 1, 2, 2, 2])
    target = np.asarray([10, 20, 30])

    labels, indices = sample_part_prompt_indices(face_indices, target, prompts_per_part=2, seed=42)
    repeated_labels, repeated_indices = sample_part_prompt_indices(
        face_indices, target, prompts_per_part=2, seed=42
    )

    np.testing.assert_array_equal(labels, [10, 20, 30])
    np.testing.assert_array_equal(labels, repeated_labels)
    np.testing.assert_array_equal(indices, repeated_indices)
    np.testing.assert_array_equal(
        target[face_indices[indices]],
        np.broadcast_to(labels[:, None], indices.shape),
    )


def test_sample_part_prompts_rejects_a_missed_surface_part() -> None:
    with pytest.raises(ValueError, match="missed ground-truth part 30"):
        sample_part_prompt_indices(
            np.asarray([0, 0, 1, 1]),
            np.asarray([10, 20, 30]),
            prompts_per_part=1,
            seed=42,
        )


def test_joint_mask_selection_expands_until_all_points_are_covered() -> None:
    probabilities = np.full((6, 2, 3), 0.1)
    probabilities[[0], 0, 0] = 0.9
    probabilities[[0, 1], 0, 1] = 0.9
    probabilities[[0, 1, 2], 0, 2] = 0.9
    probabilities[[5], 1, 0] = 0.9
    probabilities[[3, 4, 5], 1, 1] = 0.9
    probabilities[:, 1, 2] = 0.9

    result = select_joint_prompt_masks(probabilities)

    np.testing.assert_array_equal(result.selected_heads, [2, 1])
    np.testing.assert_array_equal(result.point_ids, [0, 0, 0, 1, 1, 1])
    assert result.coverage_fraction == 1.0
    assert result.overlap_fraction == 0.0


def test_interactive_prompt_miou_uses_the_predicted_iou_head() -> None:
    probabilities = np.full((4, 2, 3), 0.1)
    probabilities[[0, 1], 0, 1] = 0.9
    probabilities[[2, 3], 1, 2] = 0.9
    predicted_iou = np.asarray([[0.1, 0.9, 0.2], [0.1, 0.2, 0.9]])

    score = interactive_prompt_miou(
        probabilities,
        predicted_iou,
        np.asarray([4, 4, 8, 8]),
        np.asarray([4, 8]),
    )

    assert score == 100.0
