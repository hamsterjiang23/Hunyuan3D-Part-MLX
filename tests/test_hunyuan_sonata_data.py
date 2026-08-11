from __future__ import annotations

import importlib.util
import sys
import types
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


def test_prepare_sonata_input_keeps_float64_until_released_tensor_boundary() -> None:
    # The third x coordinate is just below a 0.005 voxel boundary in float64,
    # but rounds onto the boundary if converted to float32 too early.
    points = np.asarray(
        [[-1.0, -1.0, 0.0], [1.0, 1.0, 1.0], [0.005 - 1e-10, 0.0, 0.5]],
        dtype=np.float64,
    )
    prepared = prepare_sonata_input(points, grid_size=0.005, seed=42)

    shifted = points.copy()
    minimum, maximum = shifted.min(axis=0), shifted.max(axis=0)
    shifted -= [(minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2, minimum[2]]
    expected_grid = np.floor(shifted / np.asarray(0.005)).astype(np.int64)
    expected_grid -= expected_grid.min(axis=0)

    np.testing.assert_array_equal(np.sort(prepared.grid_coord[:, 0]), np.sort(expected_grid[:, 0]))
    assert prepared.coord.dtype == np.float32
    assert prepared.feat.dtype == np.float32


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch reference is optional")
def test_prepare_sonata_input_matches_released_transform() -> None:
    root = Path(__file__).parents[1]
    sonata_root = root / ".upstream" / "hunyuan3d-part" / "XPart" / "partgen" / "models" / "sonata"
    if not sonata_root.exists():
        pytest.skip("pinned Hunyuan3D-Part source is unavailable")

    package_name = "split3d_test_sonata_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(sonata_root)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    try:
        for name in ("registry", "transform"):
            module_name = f"{package_name}.{name}"
            spec = importlib.util.spec_from_file_location(module_name, sonata_root / f"{name}.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        released = sys.modules[f"{package_name}.transform"]

        rng = np.random.default_rng(2026)
        points = rng.normal(size=(2048, 3)).astype(np.float64)
        points[0] = [-1.0, -1.0, 0.0]
        points[1] = [1.0, 1.0, 1.0]
        points[2] = [0.005 - 1e-10, 0.0, 0.5]
        normals = rng.normal(size=points.shape).astype(np.float64)
        data = {
            "coord": points.copy(),
            "normal": normals.copy(),
            "color": np.ones_like(points),
            "batch": np.zeros(len(points), dtype=np.int64),
        }
        np.random.seed(42)
        expected = released.default()(data)
        actual = prepare_sonata_input(points, normals, seed=42)

        for key in ("coord", "grid_coord", "color", "feat", "inverse"):
            np.testing.assert_array_equal(expected[key].numpy(), getattr(actual, key))
    finally:
        sys.modules.pop(f"{package_name}.transform", None)
        sys.modules.pop(f"{package_name}.registry", None)
        sys.modules.pop(package_name, None)


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
