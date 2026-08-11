from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np


def kernel_offsets(kernel_size: int = 3, dilation: int = 1) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    if dilation <= 0:
        raise ValueError(f"dilation must be positive, got {dilation}")
    radius = kernel_size // 2
    return np.asarray(
        list(product(range(-radius * dilation, radius * dilation + 1, dilation), repeat=3)),
        dtype=np.int32,
    )


def build_subm_neighbor_map(
    coordinates: np.ndarray,
    *,
    kernel_size: int = 3,
    dilation: int = 1,
) -> np.ndarray:
    """Map every active output voxel and kernel position to an active input voxel.

    Coordinates use the same ``[batch, axis0, axis1, axis2]`` order passed to
    ``spconv.SparseConvTensor``. A missing neighbor is represented by ``-1``.
    Kernel positions follow C-order over the three spatial kernel axes, matching
    spconv's KRSC weight layout ``[out, k0, k1, k2, in]``.
    """

    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError(f"coordinates must have shape [N, 4], got {coordinates.shape}")
    if not np.issubdtype(coordinates.dtype, np.integer):
        raise TypeError(f"coordinates must be integral, got {coordinates.dtype}")
    coordinates = coordinates.astype(np.int64, copy=False)
    coordinate_tuples = [tuple(int(value) for value in row) for row in coordinates]
    lookup = {coordinate: index for index, coordinate in enumerate(coordinate_tuples)}
    if len(lookup) != len(coordinate_tuples):
        raise ValueError("SubMConv3d coordinates must be unique")

    offsets = kernel_offsets(kernel_size, dilation)
    neighbors = np.full((len(coordinates), len(offsets)), -1, dtype=np.int32)
    for output_index, coordinate in enumerate(coordinate_tuples):
        batch, axis0, axis1, axis2 = coordinate
        for kernel_index, (offset0, offset1, offset2) in enumerate(offsets):
            input_coordinate = (
                batch,
                axis0 + int(offset0),
                axis1 + int(offset1),
                axis2 + int(offset2),
            )
            neighbors[output_index, kernel_index] = lookup.get(input_coordinate, -1)
    return neighbors


def subm_conv3d_numpy(
    features: np.ndarray,
    coordinates: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    dilation: int = 1,
    neighbor_map: np.ndarray | None = None,
) -> np.ndarray:
    """Reference implementation matching spconv ``SubMConv3d`` cross-correlation."""

    features = np.asarray(features)
    weight = np.asarray(weight)
    if features.ndim != 2:
        raise ValueError(f"features must have shape [N, C_in], got {features.shape}")
    if weight.ndim != 5 or len(set(weight.shape[1:4])) != 1:
        raise ValueError(f"weight must have shape [C_out, K, K, K, C_in], got {weight.shape}")
    if features.shape[1] != weight.shape[-1]:
        raise ValueError(f"input channels do not match: features={features.shape}, weight={weight.shape}")
    kernel_size = weight.shape[1]
    if neighbor_map is None:
        neighbor_map = build_subm_neighbor_map(
            coordinates,
            kernel_size=kernel_size,
            dilation=dilation,
        )
    neighbor_map = np.asarray(neighbor_map)
    expected_map_shape = (features.shape[0], kernel_size**3)
    if neighbor_map.shape != expected_map_shape:
        raise ValueError(f"neighbor_map must have shape {expected_map_shape}, got {neighbor_map.shape}")

    output_dtype = np.result_type(features.dtype, weight.dtype, bias.dtype if bias is not None else features.dtype)
    output = np.zeros((features.shape[0], weight.shape[0]), dtype=output_dtype)
    flattened_weight = weight.reshape(weight.shape[0], kernel_size**3, weight.shape[-1])
    for kernel_index in range(kernel_size**3):
        input_indices = neighbor_map[:, kernel_index]
        valid = input_indices >= 0
        output[valid] += features[input_indices[valid]] @ flattened_weight[:, kernel_index, :].T
    if bias is not None:
        bias = np.asarray(bias)
        if bias.shape != (weight.shape[0],):
            raise ValueError(f"bias must have shape {(weight.shape[0],)}, got {bias.shape}")
        output += bias
    return output


try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # pragma: no cover - exercised by CUDA/CPU reference hosts
    mx = None
    nn = None


if nn is not None:

    class SubMConv3dMLX(nn.Module):
        """MLX implementation of spconv SubMConv3d with stable active coordinates."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            *,
            dilation: int = 1,
            bias: bool = False,
            chunk_size: int = 8192,
        ) -> None:
            super().__init__()
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
            if in_channels <= 0 or out_channels <= 0:
                raise ValueError("channel counts must be positive")
            if chunk_size <= 0:
                raise ValueError("chunk_size must be positive")
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.dilation = dilation
            self.chunk_size = chunk_size
            self.weight = mx.zeros((out_channels, kernel_size, kernel_size, kernel_size, in_channels))
            if bias:
                self.bias = mx.zeros((out_channels,))

        def __call__(
            self,
            features: Any,
            coordinates: np.ndarray,
            *,
            neighbor_map: np.ndarray | None = None,
        ) -> Any:
            if features.ndim != 2 or features.shape[1] != self.in_channels:
                raise ValueError(
                    f"features must have shape [N, {self.in_channels}], got {features.shape}"
                )
            if neighbor_map is None:
                neighbor_map = build_subm_neighbor_map(
                    coordinates,
                    kernel_size=self.kernel_size,
                    dilation=self.dilation,
                )
            expected_map_shape = (features.shape[0], self.kernel_size**3)
            if neighbor_map.shape != expected_map_shape:
                raise ValueError(f"neighbor_map must have shape {expected_map_shape}, got {neighbor_map.shape}")

            neighbor_indices = mx.array(neighbor_map, dtype=mx.int32)
            flattened_weight = self.weight.reshape(
                self.out_channels,
                self.kernel_size**3,
                self.in_channels,
            )
            chunks = []
            for start in range(0, features.shape[0], self.chunk_size):
                stop = min(start + self.chunk_size, features.shape[0])
                chunk_indices = neighbor_indices[start:stop]
                valid = chunk_indices >= 0
                safe_indices = mx.where(valid, chunk_indices, mx.zeros_like(chunk_indices))
                neighbor_features = mx.take(features, safe_indices, axis=0)
                neighbor_features = mx.where(valid[..., None], neighbor_features, 0)
                kernels = flattened_weight.transpose(1, 2, 0)
                chunk_output = mx.einsum("nki,kio->no", neighbor_features, kernels)
                if hasattr(self, "bias"):
                    chunk_output = chunk_output + self.bias
                mx.eval(chunk_output)
                chunks.append(chunk_output)
            return mx.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

else:

    class SubMConv3dMLX:  # pragma: no cover - defensive import error on non-Apple hosts
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("SubMConv3dMLX requires Apple MLX")
