from __future__ import annotations

import json

import mlx.core as mx
import numpy as np

from split3d.hunyuan.sparse import SubMConv3dMLX, subm_conv3d_numpy


def main() -> int:
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
    absolute_error = np.abs(actual - expected)
    payload = {
        "device": str(mx.default_device()),
        "shape": list(actual.shape),
        "finite": bool(np.isfinite(actual).all()),
        "max_abs_error": float(absolute_error.max(initial=0.0)),
        "mean_abs_error": float(absolute_error.mean()),
    }
    print(json.dumps(payload, indent=2))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
