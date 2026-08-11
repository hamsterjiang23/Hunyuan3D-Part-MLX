from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointMaskSelection:
    """Result of the paper-described multi-prompt mask expansion procedure."""

    selected_heads: np.ndarray
    point_ids: np.ndarray
    coverage_fraction: float
    overlap_fraction: float


def sample_part_prompt_indices(
    face_indices: np.ndarray,
    target_face_labels: np.ndarray,
    *,
    prompts_per_part: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample reproducible point prompts from every ground-truth part.

    This is deliberately an oracle benchmark helper: ``target_face_labels`` is
    used only to reconstruct the paper's connectivity and interactive protocols,
    never by the public automatic segmentation service.
    """

    sampled_faces = np.asarray(face_indices, dtype=np.int64)
    targets = np.asarray(target_face_labels)
    if sampled_faces.ndim != 1:
        raise ValueError(f"face_indices must be one-dimensional, got {sampled_faces.shape}")
    if targets.ndim != 1:
        raise ValueError(f"target_face_labels must be one-dimensional, got {targets.shape}")
    if len(sampled_faces) == 0:
        raise ValueError("face_indices cannot be empty")
    if sampled_faces.min() < 0 or sampled_faces.max() >= len(targets):
        raise ValueError("face_indices refer outside target_face_labels")
    if prompts_per_part < 1:
        raise ValueError("prompts_per_part must be positive")

    point_labels = targets[sampled_faces]
    part_labels = np.unique(targets)
    generator = np.random.RandomState(seed)
    prompt_indices = np.empty((len(part_labels), prompts_per_part), dtype=np.int64)
    for part_index, part_label in enumerate(part_labels):
        candidates = np.flatnonzero(point_labels == part_label)
        if len(candidates) == 0:
            raise ValueError(f"surface sampling missed ground-truth part {part_label.item()!r}")
        prompt_indices[part_index] = generator.choice(
            candidates,
            size=prompts_per_part,
            replace=len(candidates) < prompts_per_part,
        )
    return part_labels, prompt_indices


def select_joint_prompt_masks(
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> JointMaskSelection:
    """Reconstruct Appendix A.5's joint smallest-to-largest mask selection.

    Each prompt starts at its smallest predicted mask. At every step the next
    larger mask maximizing newly covered points minus added overlap is selected;
    ties prefer less overlap, then more coverage, less area growth, and the
    earlier prompt. The
    released repository does not contain the paper's evaluation implementation,
    so the deterministic tie-breaking here is explicitly auditable.
    """

    values = np.asarray(probabilities)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"probabilities must have shape [N, K, 3], got {values.shape}")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("probabilities require at least one point and one prompt")
    if not np.isfinite(values).all():
        raise ValueError("probabilities contain non-finite values")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")

    masks = values > threshold
    areas = masks.sum(axis=0, dtype=np.int64)
    head_order = np.argsort(areas, axis=1, kind="stable")
    prompt_count = values.shape[1]
    selected_rank = np.zeros(prompt_count, dtype=np.int64)
    selected_heads = head_order[:, 0].copy()
    selected_masks = masks[:, np.arange(prompt_count), selected_heads]
    coverage_counts = selected_masks.sum(axis=1, dtype=np.int64)

    while np.any(coverage_counts == 0) and np.any(selected_rank < 2):
        current_uncovered = int(np.count_nonzero(coverage_counts == 0))
        current_overlap = int(np.maximum(coverage_counts - 1, 0).sum())
        choices: list[tuple[int, int, int, int, int, np.ndarray]] = []
        for prompt in np.flatnonzero(selected_rank < 2):
            next_rank = selected_rank[prompt] + 1
            next_head = head_order[prompt, next_rank]
            next_counts = coverage_counts - selected_masks[:, prompt] + masks[:, prompt, next_head]
            coverage_gain = current_uncovered - int(np.count_nonzero(next_counts == 0))
            overlap_growth = int(np.maximum(next_counts - 1, 0).sum()) - current_overlap
            area_growth = int(areas[prompt, next_head] - areas[prompt, selected_heads[prompt]])
            net_gain = coverage_gain - overlap_growth
            choices.append((-net_gain, overlap_growth, -coverage_gain, area_growth, int(prompt), next_counts))
        _, _, _, _, prompt, coverage_counts = min(choices, key=lambda choice: choice[:5])
        selected_rank[prompt] += 1
        selected_heads[prompt] = head_order[prompt, selected_rank[prompt]]
        selected_masks[:, prompt] = masks[:, prompt, selected_heads[prompt]]

    selected_probabilities = values[:, np.arange(prompt_count), selected_heads]
    selected_probabilities = np.where(selected_masks, selected_probabilities, -np.inf)
    point_ids = np.argmax(selected_probabilities, axis=1).astype(np.int64)
    point_ids[coverage_counts == 0] = -1
    return JointMaskSelection(
        selected_heads=selected_heads,
        point_ids=point_ids,
        coverage_fraction=float(np.mean(coverage_counts > 0)),
        overlap_fraction=float(np.mean(coverage_counts > 1)),
    )


def interactive_prompt_miou(
    probabilities: np.ndarray,
    predicted_iou: np.ndarray,
    point_labels: np.ndarray,
    prompt_part_labels: np.ndarray,
    *,
    threshold: float = 0.5,
) -> float:
    """Compute the paper's single-prompt mask IoU averaged over prompts."""

    values = np.asarray(probabilities)
    scores = np.asarray(predicted_iou)
    targets = np.asarray(point_labels)
    prompt_labels = np.asarray(prompt_part_labels)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"probabilities must have shape [N, K, 3], got {values.shape}")
    if scores.shape != values.shape[1:]:
        raise ValueError(f"predicted_iou must have shape {values.shape[1:]}, got {scores.shape}")
    if targets.shape != (values.shape[0],):
        raise ValueError(f"point_labels must have shape {(values.shape[0],)}, got {targets.shape}")
    if prompt_labels.shape != (values.shape[1],):
        raise ValueError(f"prompt_part_labels must have shape {(values.shape[1],)}, got {prompt_labels.shape}")

    best_heads = np.argmax(scores, axis=1)
    predicted_masks = values[:, np.arange(values.shape[1]), best_heads] > threshold
    intersections = np.asarray(
        [np.count_nonzero(predicted_masks[:, index] & (targets == label)) for index, label in enumerate(prompt_labels)],
        dtype=np.float64,
    )
    unions = np.asarray(
        [np.count_nonzero(predicted_masks[:, index] | (targets == label)) for index, label in enumerate(prompt_labels)],
        dtype=np.float64,
    )
    ious = np.divide(intersections, unions, out=np.ones_like(intersections), where=unions != 0)
    return float(ious.mean() * 100.0)
