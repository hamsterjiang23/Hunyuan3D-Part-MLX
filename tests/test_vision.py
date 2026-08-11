from __future__ import annotations

import numpy as np

from split3d.vision import Detection, accumulate_face_scores


def test_accumulate_face_scores_uses_visibility_and_mask_pixels() -> None:
    face_ids = np.array([[0, 0, 1], [0, 2, -1]], dtype=np.int32)
    detections = [Detection(semantic_index=1, semantic_name="leg", score=0.5, box=[0, 0, 2, 2])]
    masks = np.array([[[True, False, True], [False, True, False]]])
    numerator = np.zeros((3, 2), dtype=np.float64)
    visibility = np.zeros(3, dtype=np.float64)
    accumulate_face_scores(face_ids, detections, masks, numerator, visibility)
    assert visibility.tolist() == [3.0, 1.0, 1.0]
    assert numerator[:, 1].tolist() == [0.5, 0.5, 0.5]
    assert np.all(numerator[:, 0] == 0)


def test_accumulate_tracks_visibility_per_detected_semantic() -> None:
    face_ids = np.array([[0, 1], [0, 1]], dtype=np.int32)
    detections = [Detection(semantic_index=1, semantic_name="leg", score=0.5, box=[0, 0, 1, 1])]
    masks = np.ones((1, 2, 2), dtype=bool)
    numerator = np.zeros((2, 2), dtype=np.float64)
    visibility = np.zeros(2, dtype=np.float64)
    semantic_visibility = np.zeros((2, 2), dtype=np.float64)
    accumulate_face_scores(face_ids, detections, masks, numerator, visibility, semantic_visibility)
    assert np.all(semantic_visibility[:, 0] == 0)
    assert semantic_visibility[:, 1].tolist() == [2.0, 2.0]
