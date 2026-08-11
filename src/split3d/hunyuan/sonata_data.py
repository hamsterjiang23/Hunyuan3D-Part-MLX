from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

SerializationOrder = Literal["z", "z-trans", "hilbert", "hilbert-trans"]


@dataclass(frozen=True)
class SonataInput:
    """Deterministic NumPy representation consumed by the MLX Sonata encoder."""

    coord: np.ndarray
    grid_coord: np.ndarray
    color: np.ndarray
    normal: np.ndarray
    feat: np.ndarray
    batch: np.ndarray
    offset: np.ndarray
    inverse: np.ndarray


@dataclass(frozen=True)
class SerializedPointOrder:
    depth: int
    code: np.ndarray
    order: np.ndarray
    inverse: np.ndarray


def _right_shift_bits(binary: np.ndarray, shift: int = 1) -> np.ndarray:
    if binary.shape[-1] <= shift:
        return np.zeros_like(binary)
    return np.pad(binary[..., :-shift], [(0, 0)] * (binary.ndim - 1) + [(shift, 0)])


def _gray_to_binary(gray: np.ndarray) -> np.ndarray:
    result = gray.astype(bool, copy=True)
    shift = 1 << (int(np.ceil(np.log2(result.shape[-1]))) - 1)
    while shift > 0:
        result = np.logical_xor(result, _right_shift_bits(result, shift))
        shift //= 2
    return result


def hilbert_encode(grid_coord: np.ndarray, depth: int) -> np.ndarray:
    """Encode 3-D integer coordinates exactly like Sonata's torch implementation."""

    locations = np.ascontiguousarray(grid_coord, dtype=np.int64)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError(f"grid_coord must have shape [N, 3], got {locations.shape}")
    if depth <= 0 or depth * 3 > 63:
        raise ValueError(f"depth must fit in a signed 64-bit code, got {depth}")

    bitpack_mask = np.left_shift(np.uint8(1), np.arange(8, dtype=np.uint8))
    bitpack_mask_rev = bitpack_mask[::-1]
    # The reference views little-endian int64 values as bytes, then reverses them.
    location_bytes = locations.astype("<i8", copy=False).view(np.uint8).reshape(-1, 3, 8)[..., ::-1]
    gray = np.bitwise_and(location_bytes[..., None], bitpack_mask_rev).astype(bool).reshape(-1, 3, 64)[..., -depth:]
    for bit in range(depth):
        for dim in range(3):
            mask = gray[:, dim, bit].copy()
            gray[:, 0, bit + 1 :] = np.logical_xor(gray[:, 0, bit + 1 :], mask[:, None])
            to_flip = np.logical_and(
                np.logical_not(mask[:, None]),
                np.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = np.logical_xor(gray[:, dim, bit + 1 :], to_flip)
            gray[:, 0, bit + 1 :] = np.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    binary = _gray_to_binary(gray.swapaxes(1, 2).reshape(-1, depth * 3))
    padded = np.pad(binary, ((0, 0), (64 - depth * 3, 0)))
    encoded_bytes = (
        (padded[:, ::-1].reshape(-1, 8, 8) * bitpack_mask[None, None, :]).sum(axis=2, dtype=np.uint16).astype(np.uint8)
    )
    return encoded_bytes.view("<i8").reshape(-1)


def z_order_encode(grid_coord: np.ndarray, depth: int) -> np.ndarray:
    coordinates = np.asarray(grid_coord, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"grid_coord must have shape [N, 3], got {coordinates.shape}")
    if depth <= 0 or depth > 16:
        raise ValueError(f"Sonata supports z-order depths from 1 to 16, got {depth}")
    result = np.zeros(len(coordinates), dtype=np.int64)
    for bit in range(depth):
        mask = np.int64(1 << bit)
        result |= (coordinates[:, 0] & mask) << (2 * bit + 2)
        result |= (coordinates[:, 1] & mask) << (2 * bit + 1)
        result |= (coordinates[:, 2] & mask) << (2 * bit)
    return result


def encode_serialized(
    grid_coord: np.ndarray,
    batch: np.ndarray,
    depth: int,
    order: SerializationOrder,
) -> np.ndarray:
    coordinates = np.asarray(grid_coord, dtype=np.int64)
    batch = np.asarray(batch, dtype=np.int64)
    if batch.shape != (len(coordinates),):
        raise ValueError(f"batch must have shape {(len(coordinates),)}, got {batch.shape}")
    if order.endswith("-trans"):
        coordinates = coordinates[:, [1, 0, 2]]
    if order.startswith("z"):
        code = z_order_encode(coordinates, depth)
    elif order.startswith("hilbert"):
        code = hilbert_encode(coordinates, depth)
    else:
        raise ValueError(f"unsupported serialization order: {order}")
    return (batch << (depth * 3)) | code


def serialize_points(
    grid_coord: np.ndarray,
    batch: np.ndarray,
    *,
    orders: Sequence[SerializationOrder] = ("z", "z-trans", "hilbert", "hilbert-trans"),
    depth: int | None = None,
    permutation: np.ndarray | None = None,
) -> SerializedPointOrder:
    coordinates = np.asarray(grid_coord, dtype=np.int64)
    if len(coordinates) == 0:
        raise ValueError("cannot serialize an empty point cloud")
    if np.any(coordinates < 0):
        raise ValueError("grid_coord must be non-negative")
    if depth is None:
        depth = int(int(coordinates.max()) + 1).bit_length()
    if depth > 16:
        raise ValueError(f"Sonata serialization depth exceeds 16: {depth}")
    codes = np.stack([encode_serialized(coordinates, batch, depth, order) for order in orders])
    point_orders = np.argsort(codes, axis=1, kind="stable")
    inverse = np.empty_like(point_orders)
    row = np.arange(point_orders.shape[0])[:, None]
    inverse[row, point_orders] = np.arange(point_orders.shape[1])[None, :]
    if permutation is not None:
        permutation = np.asarray(permutation)
        codes = codes[permutation]
        point_orders = point_orders[permutation]
        inverse = inverse[permutation]
    return SerializedPointOrder(depth=depth, code=codes, order=point_orders, inverse=inverse)


def fnv_hash(grid_coord: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(grid_coord).astype(np.uint64, copy=False)
    hashed = np.full(len(coordinates), np.uint64(14695981039346656037), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for axis in range(coordinates.shape[1]):
            hashed *= np.uint64(1099511628211)
            hashed = np.bitwise_xor(hashed, coordinates[:, axis])
    return hashed


def official_fps_start_index(
    points: np.ndarray,
    *,
    grid_size: float = 0.005,
    seed: int = 42,
) -> int:
    """Return the FPS start after the official GridSample RNG draw."""

    coord = np.asarray(points).copy()
    if not np.issubdtype(coord.dtype, np.floating):
        coord = coord.astype(np.float64)
    if coord.ndim != 2 or coord.shape[1] != 3 or len(coord) == 0:
        raise ValueError(f"points must have shape [N, 3] with N > 0, got {coord.shape}")
    minimum = coord.min(axis=0)
    maximum = coord.max(axis=0)
    coord -= [(minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2, minimum[2]]
    grid = np.floor(coord / np.asarray(grid_size)).astype(np.int64)
    grid -= grid.min(axis=0)
    keys = fnv_hash(grid)
    _, counts = np.unique(keys[np.argsort(keys)], return_counts=True)
    rng = np.random.RandomState(seed)
    rng.randint(0, int(counts.max()), size=len(counts))
    return int(rng.randint(0, len(coord)))


def prepare_sonata_input(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    *,
    colors: np.ndarray | None = None,
    grid_size: float = 0.005,
    seed: int = 42,
) -> SonataInput:
    """Apply the released CenterShift/GridSample/NormalizeColor pipeline.

    The upstream transform is configured in ``train`` mode even during inference,
    so this function exposes the random seed and makes voxel representative choice
    reproducible.
    """

    coord = np.asarray(points)
    if not np.issubdtype(coord.dtype, np.floating):
        coord = coord.astype(np.float64)
    if coord.ndim == 2:
        coord = coord[None, ...]
    if coord.ndim != 3 or coord.shape[-1] != 3:
        raise ValueError(f"points must have shape [N, 3] or [B, N, 3], got {coord.shape}")
    batch_size, points_per_batch, _ = coord.shape
    flat_coord = coord.reshape(-1, 3).copy()
    flat_normal = np.ones_like(flat_coord) if normals is None else np.asarray(normals).reshape(-1, 3).copy()
    flat_color = np.ones_like(flat_coord) if colors is None else np.asarray(colors).reshape(-1, 3).copy()
    if flat_normal.shape != flat_coord.shape or flat_color.shape != flat_coord.shape:
        raise ValueError("normals and colors must match the shape of points")

    rng = np.random.RandomState(seed)
    selected_parts: list[np.ndarray] = []
    inverse_parts: list[np.ndarray] = []
    grid_parts: list[np.ndarray] = []
    selected_base = 0
    for batch_index in range(batch_size):
        start = batch_index * points_per_batch
        stop = start + points_per_batch
        batch_coord = flat_coord[start:stop]
        minimum = batch_coord.min(axis=0)
        maximum = batch_coord.max(axis=0)
        batch_coord -= [
            (minimum[0] + maximum[0]) / 2,
            (minimum[1] + maximum[1]) / 2,
            minimum[2],
        ]
        grid = np.floor(batch_coord / np.asarray(grid_size)).astype(np.int64)
        grid -= grid.min(axis=0)
        key = fnv_hash(grid)
        # Match Sonata GridSample exactly: NumPy's default quicksort controls
        # the within-voxel order from which the seeded random representative is
        # chosen. A stable sort changes representatives and therefore features.
        sorted_indices = np.argsort(key)
        sorted_key = key[sorted_indices]
        _, inverse_sorted, counts = np.unique(sorted_key, return_inverse=True, return_counts=True)
        group_starts = np.cumsum(np.insert(counts, 0, 0)[:-1])
        offsets = rng.randint(0, int(counts.max()), size=len(counts)) % counts
        chosen_local = sorted_indices[group_starts + offsets]
        inverse_local = np.empty_like(inverse_sorted)
        inverse_local[sorted_indices] = inverse_sorted + selected_base
        selected_parts.append(chosen_local + start)
        inverse_parts.append(inverse_local)
        grid_parts.append(grid[chosen_local])
        selected_base += len(chosen_local)

    selected = np.concatenate(selected_parts)
    # Match the released ToTensor boundary: all NumPy geometry and voxel
    # selection above uses the incoming floating dtype; tensors become float32
    # only after representatives have been selected.
    selected_coord = flat_coord[selected].astype(np.float32, copy=False)
    selected_color = (flat_color[selected] / 255).astype(np.float32, copy=False)
    selected_normal = flat_normal[selected].astype(np.float32, copy=False)
    batch = np.repeat(
        np.arange(batch_size, dtype=np.int64),
        [len(part) for part in selected_parts],
    )
    offset = np.cumsum([len(part) for part in selected_parts], dtype=np.int64)
    return SonataInput(
        coord=selected_coord,
        grid_coord=np.concatenate(grid_parts).astype(np.int64, copy=False),
        color=selected_color,
        normal=selected_normal,
        feat=np.concatenate([selected_coord, selected_color, selected_normal], axis=1),
        batch=batch,
        offset=offset,
        inverse=np.concatenate(inverse_parts),
    )
