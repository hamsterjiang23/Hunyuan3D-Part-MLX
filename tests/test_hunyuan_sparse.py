from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from split3d.hunyuan.sparse import build_subm_neighbor_map, subm_conv3d_numpy


def test_subm_conv_matches_known_spconv_kernel_orientation() -> None:
    coordinates = np.asarray(
        [[0, 2, 2, 1], [0, 2, 2, 2], [0, 2, 2, 3]],
        dtype=np.int32,
    )
    features = np.asarray([[1.0], [10.0], [100.0]], dtype=np.float32)
    weight = np.arange(27, dtype=np.float32).reshape(1, 3, 3, 3, 1)

    output = subm_conv3d_numpy(features, coordinates, weight)

    np.testing.assert_array_equal(output[:, 0], np.asarray([153.0, 1542.0, 1420.0]))


def test_neighbor_map_keeps_batch_boundaries_and_rejects_duplicates() -> None:
    coordinates = np.asarray(
        [[0, 1, 1, 1], [0, 1, 1, 2], [1, 1, 1, 1]],
        dtype=np.int32,
    )
    neighbors = build_subm_neighbor_map(coordinates)

    center = 13
    positive_axis2 = 14
    assert neighbors[:, center].tolist() == [0, 1, 2]
    assert neighbors[0, positive_axis2] == 1
    assert neighbors[2, positive_axis2] == -1
    with pytest.raises(ValueError, match="unique"):
        build_subm_neighbor_map(np.concatenate([coordinates, coordinates[:1]], axis=0))


@pytest.mark.skipif(
    importlib.util.find_spec("spconv") is None,
    reason="spconv CUDA reference is optional",
)
def test_numpy_reference_matches_spconv_cuda() -> None:
    import spconv.pytorch as spconv
    import torch
    from spconv.core import ConvAlgo

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(2026)
    coordinates = np.asarray(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 2],
            [0, 1, 2, 1],
            [0, 2, 1, 1],
            [0, 2, 2, 2],
        ],
        dtype=np.int32,
    )
    features = rng.standard_normal((len(coordinates), 2), dtype=np.float32)
    weight = rng.standard_normal((3, 3, 3, 3, 2), dtype=np.float32)
    bias = rng.standard_normal((3,), dtype=np.float32)

    conv = spconv.SubMConv3d(2, 3, 3, padding=1, bias=True, algo=ConvAlgo.Native).cuda().eval()
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(weight).cuda())
        assert conv.bias is not None
        conv.bias.copy_(torch.from_numpy(bias).cuda())
        sparse = spconv.SparseConvTensor(
            torch.from_numpy(features).cuda(),
            torch.from_numpy(coordinates).cuda(),
            [4, 4, 4],
            1,
        )
        expected = conv(sparse).features.cpu().numpy()

    actual = subm_conv3d_numpy(features, coordinates, weight, bias)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="MLX is only available on Apple silicon",
)
def test_mlx_subm_conv_matches_numpy_reference() -> None:
    import mlx.core as mx

    from split3d.hunyuan.sparse import SubMConv3dMLX

    rng = np.random.default_rng(2026)
    coordinates = np.asarray(
        [[0, 1, 1, 1], [0, 1, 1, 2], [0, 1, 2, 1], [0, 2, 2, 2]],
        dtype=np.int32,
    )
    features = rng.standard_normal((len(coordinates), 2), dtype=np.float32)
    weight = rng.standard_normal((3, 3, 3, 3, 2), dtype=np.float32)
    bias = rng.standard_normal((3,), dtype=np.float32)
    expected = subm_conv3d_numpy(features, coordinates, weight, bias)

    layer = SubMConv3dMLX(2, 3, bias=True, chunk_size=2)
    layer.weight = mx.array(weight)
    layer.bias = mx.array(bias)
    actual = np.asarray(layer(mx.array(features), coordinates))

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
