from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from split3d.hunyuan.sonata_data import (
    encode_serialized,
    fnv_hash,
    official_fps_start_index,
    prepare_sonata_input,
    serialize_points,
)


def test_official_fps_start_advances_past_grid_sample_draw() -> None:
    points = np.asarray([[0.001 * index, 0.0, 0.0] for index in range(40)], dtype=np.float32)
    shifted = points.copy()
    minimum, maximum = shifted.min(axis=0), shifted.max(axis=0)
    shifted -= [(minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2, minimum[2]]
    grid = np.floor(shifted / np.float32(0.005)).astype(np.int64)
    grid -= grid.min(axis=0)
    keys = fnv_hash(grid)
    _, counts = np.unique(keys[np.argsort(keys)], return_counts=True)
    rng = np.random.RandomState(42)
    rng.randint(0, int(counts.max()), size=len(counts))
    expected = int(rng.randint(0, len(points)))

    assert official_fps_start_index(points, seed=42) == expected


def test_prepare_sonata_input_is_seeded_and_maps_back_to_original_points() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.01, 0.0, 0.0]],
        dtype=np.float32,
    )
    first = prepare_sonata_input(points, grid_size=0.005, seed=2026)
    second = prepare_sonata_input(points, grid_size=0.005, seed=2026)

    assert first.coord.shape == (2, 3)
    assert first.feat.shape == (2, 9)
    assert first.inverse.shape == (4,)
    assert first.inverse.min() == 0
    assert first.inverse.max() == 1
    np.testing.assert_array_equal(first.coord, second.coord)
    np.testing.assert_array_equal(first.inverse, second.inverse)


def test_serialization_round_trip_indices() -> None:
    grid = np.asarray([[1, 0, 2], [0, 0, 0], [1, 1, 1], [0, 3, 2]], dtype=np.int64)
    serialized = serialize_points(grid, np.zeros(len(grid), dtype=np.int64))

    assert serialized.code.shape == (4, 4)
    for order, inverse in zip(serialized.order, serialized.inverse, strict=True):
        np.testing.assert_array_equal(order[inverse], np.arange(len(grid)))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch reference is optional")
def test_numpy_serialization_matches_released_torch_code() -> None:
    import torch

    root = Path(__file__).parents[1]
    package_root = root / ".upstream" / "hunyuan3d-part" / "XPart" / "partgen" / "models" / "sonata"
    sys.path.insert(0, str(package_root))
    try:
        from serialization.default import encode as torch_encode

        rng = np.random.default_rng(2026)
        grid = rng.integers(0, 64, size=(128, 3), dtype=np.int64)
        batch = rng.integers(0, 3, size=(128,), dtype=np.int64)
        for order in ("z", "z-trans", "hilbert", "hilbert-trans"):
            expected = torch_encode(torch.from_numpy(grid), torch.from_numpy(batch), 6, order).numpy()
            actual = encode_serialized(grid, batch, 6, order)
            np.testing.assert_array_equal(actual, expected)
    finally:
        sys.path.remove(str(package_root))
