from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from split3d.hunyuan.p3sam_official_postprocess import (
    aabb_distance,
    build_adjacent_faces,
    connected_regions,
    find_neighbor_regions,
    fix_labels,
    merge_small_regions,
    project_sample_labels_to_faces,
)


class _SilentProgress:
    def update(self, _value: int) -> None:
        pass


class _SilentTimer:
    def __init__(self, _name: str) -> None:
        pass

    def __enter__(self) -> _SilentTimer:
        return self

    def __exit__(self, *_args: Any) -> None:
        pass


def _tqdm(iterable: Any = None, **_kwargs: Any) -> Any:
    return _SilentProgress() if iterable is None else iterable


def _official_functions() -> SimpleNamespace:
    """Load topology helpers directly from the pinned official source file."""

    path = Path(__file__).parents[1] / ".upstream/hunyuan3d-part/P3-SAM/demo/auto_mask.py"
    if not path.exists():
        pytest.skip("pinned upstream Hunyuan3D-Part checkout is unavailable")
    wanted = {
        "build_adjacent_faces_numba",
        "fix_label",
        "get_connected_region",
        "aabb_distance",
        "aabb_volume",
        "find_neighbor_part",
        "do_post_process",
        "do_no_mask_process",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions: list[ast.FunctionDef] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            node.decorator_list = []
            definitions.append(node)
    namespace: dict[str, Any] = {
        "np": np,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "Timer": _SilentTimer,
        "tqdm": _tqdm,
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in wanted})


def test_adjacency_and_flood_fill_match_pinned_official_source() -> None:
    official = _official_functions()
    pairs = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0], [1, 3]], dtype=np.int64)
    expected_adjacency = official.build_adjacent_faces_numba(pairs)
    actual_adjacency = build_adjacent_faces(pairs, 4)
    np.testing.assert_array_equal(actual_adjacency, expected_adjacency)

    labels = np.asarray([4, -1, -1, 7], dtype=np.int64)
    expected = official.fix_label(labels.copy(), expected_adjacency.copy(), show_info=False)
    actual = fix_labels(labels.copy(), actual_adjacency.copy())
    np.testing.assert_array_equal(actual, expected)


def test_connected_regions_match_pinned_official_source() -> None:
    official = _official_functions()
    pairs = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    adjacency = build_adjacent_faces(pairs, 5)
    labels = np.asarray([0, 0, 1, 1, 0], dtype=np.int64)
    expected_parts, expected_ids = official.get_connected_region(labels, adjacency, return_face_part_ids=True)
    actual_parts, actual_ids = connected_regions(labels, adjacency, return_face_region_ids=True)
    assert actual_parts == expected_parts
    np.testing.assert_array_equal(actual_ids, expected_ids)


def test_small_region_merge_matches_pinned_official_source() -> None:
    official = _official_functions()
    pairs = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    adjacency = build_adjacent_faces(pairs, 5)
    labels = np.asarray([10, 10, 10, 20, 30], dtype=np.int64)
    areas = np.asarray([40.0, 30.0, 25.0, 4.0, 1.0])
    regions = connected_regions(labels, adjacency)
    expected = official.do_post_process(
        areas,
        [part.copy() for part in regions],
        adjacency,
        labels,
        threshold=0.95,
        show_info=False,
    )
    actual = merge_small_regions(
        areas,
        [part.copy() for part in regions],
        adjacency,
        labels,
        threshold=0.95,
    )
    np.testing.assert_array_equal(actual, expected)


def test_source_face_projection_matches_released_vote_semantics() -> None:
    face_indices = np.asarray([2, 0, 2, 1, 0, 2, 1], dtype=np.int64)
    point_ids = np.asarray([8, 3, -1, 7, 3, 8, -1], dtype=np.int64)
    actual = project_sample_labels_to_faces(face_indices, point_ids, 4)

    expected = np.full(4, -2, dtype=np.int64)
    votes: dict[int, list[int]] = {}
    for face, label in zip(face_indices, point_ids, strict=True):
        votes.setdefault(int(face), []).append(int(label))
    for face, labels in votes.items():
        expected[face] = int(np.argmax(np.bincount(np.asarray(labels) + 2))) - 2
    np.testing.assert_array_equal(actual, expected)


def test_detached_region_aabb_fallback_matches_pinned_official_source() -> None:
    official = _official_functions()
    regions = [[0], [1], [2], [3]]
    adjacency = np.empty((4, 0), dtype=np.int32)
    labels = [-1, 4, 9, 12]
    boxes = [
        (np.asarray([0.0, 0.0, 0.0]), np.asarray([1.0, 1.0, 1.0])),
        (np.asarray([2.0, 0.0, 0.0]), np.asarray([3.0, 1.0, 1.0])),
        (np.asarray([-2.0, 0.0, 0.0]), np.asarray([-1.0, 1.0, 1.0])),
        (np.asarray([4.0, 0.0, 0.0]), np.asarray([5.0, 1.0, 1.0])),
    ]
    expected = official.find_neighbor_part(regions, adjacency, boxes, labels)
    actual = find_neighbor_regions(regions, adjacency, region_aabbs=boxes, region_labels=labels)
    assert actual == expected
    assert aabb_distance(boxes[0], boxes[1]) == official.aabb_distance(boxes[0], boxes[1])
