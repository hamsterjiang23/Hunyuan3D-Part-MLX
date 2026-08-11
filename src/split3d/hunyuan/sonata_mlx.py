from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from split3d.hunyuan.sonata_data import SerializedPointOrder, SonataInput, serialize_points
from split3d.hunyuan.sparse import SubMConv3dMLX, build_subm_neighbor_map

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - exercised by CUDA/CPU reference hosts
    mx = None


ENC_CHANNELS = (48, 96, 192, 384, 512)
ENC_DEPTHS = (3, 3, 3, 12, 3)
ENC_HEADS = (3, 6, 12, 24, 32)
SERIALIZATION_ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")
PAPER_SEED_42_ORDER_PERMUTATIONS = (
    np.asarray([2, 3, 0, 1]),
    np.asarray([2, 1, 3, 0]),
    np.asarray([0, 3, 2, 1]),
    np.asarray([1, 2, 0, 3]),
    np.asarray([2, 1, 3, 0]),
)


@dataclass
class SonataPointMLX:
    feat: Any
    coord: np.ndarray
    grid_coord: np.ndarray
    batch: np.ndarray
    offset: np.ndarray
    serialized: SerializedPointOrder
    neighbor_map: np.ndarray | None = None
    pooling_parent: SonataPointMLX | None = None
    pooling_inverse: np.ndarray | None = None


def attention_padding_maps(offset: np.ndarray, patch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Sonata-compatible pad, unpad, and sequence-boundary mappings."""

    offset = np.asarray(offset, dtype=np.int64)
    counts = np.diff(np.concatenate([np.zeros(1, dtype=np.int64), offset]))
    pad_parts: list[np.ndarray] = []
    unpad_parts: list[np.ndarray] = []
    boundaries = [0]
    original_start = 0
    padded_start = 0
    for count in counts:
        count_int = int(count)
        padded_count = (
            count_int if count_int <= patch_size else ((count_int + patch_size - 1) // patch_size) * patch_size
        )
        local_pad = np.arange(padded_count, dtype=np.int64)
        if padded_count != count_int:
            remainder = count_int % patch_size
            local_pad[padded_count - patch_size + remainder :] = local_pad[
                padded_count - 2 * patch_size + remainder : padded_count - patch_size
            ]
        pad_parts.append(local_pad + original_start)
        unpad_parts.append(np.arange(count_int, dtype=np.int64) + padded_start)
        if padded_count <= patch_size:
            boundaries.append(padded_start + padded_count)
        else:
            boundaries.extend(range(padded_start + patch_size, padded_start + padded_count + 1, patch_size))
        original_start += count_int
        padded_start += padded_count
    return np.concatenate(pad_parts), np.concatenate(unpad_parts), np.asarray(boundaries, dtype=np.int64)


def _require_mlx() -> None:
    if mx is None:
        raise RuntimeError("SonataMLX requires Apple MLX")


def _linear(x: Any, weight: Any, bias: Any | None = None) -> Any:
    result = x @ weight.T
    return result if bias is None else result + bias


def _layer_norm(x: Any, weight: Any, bias: Any, eps: float = 1e-5) -> Any:
    mean = mx.mean(x, axis=-1, keepdims=True)
    variance = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * mx.rsqrt(variance + eps) * weight + bias


def _gelu(x: Any) -> Any:
    return x * 0.5 * (1.0 + mx.erf(x / np.sqrt(2.0)))


class SerializedAttentionMLX:
    def __init__(
        self,
        weights: dict[str, Any],
        prefix: str,
        channels: int,
        heads: int,
        order_index: int,
        *,
        official_attention_precision: bool = False,
    ) -> None:
        self.channels = channels
        self.heads = heads
        self.order_index = order_index
        self.scale = (channels // heads) ** -0.5
        self.qkv_weight = weights[f"{prefix}.qkv.weight"]
        self.qkv_bias = weights[f"{prefix}.qkv.bias"]
        self.proj_weight = weights[f"{prefix}.proj.weight"]
        self.proj_bias = weights[f"{prefix}.proj.bias"]
        self.official_attention_precision = official_attention_precision

    def __call__(self, point: SonataPointMLX, patch_size: int = 1024) -> Any:
        pad, unpad, boundaries = attention_padding_maps(point.offset, patch_size)
        rank_order = point.serialized.order[self.order_index]
        ordered_point_indices = rank_order[pad]
        qkv = _linear(point.feat, self.qkv_weight, self.qkv_bias)
        qkv = mx.take(qkv, mx.array(ordered_point_indices, dtype=mx.int32), axis=0)
        output_dtype = qkv.dtype
        # Approximate the released Sonata FlashAttention dtype boundary by
        # quantizing packed QKV through FP16.  MLX 0.32.0 FP16 SDPA produces
        # all-NaN features for this network, so attention itself runs in FP32
        # after quantization and is cast to the surrounding model dtype before
        # the output projection.
        if self.official_attention_precision:
            qkv = qkv.astype(mx.float16).astype(mx.float32)
        qkv = qkv.reshape(-1, 3, self.heads, self.channels // self.heads)
        chunks = []
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            packed = qkv[int(start) : int(stop)]
            q = packed[:, 0].transpose(1, 0, 2)[None]
            k = packed[:, 1].transpose(1, 0, 2)[None]
            v = packed[:, 2].transpose(1, 0, 2)[None]
            attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
            chunks.append(attended[0].transpose(1, 0, 2).reshape(int(stop - start), self.channels))
        ordered_output = mx.concatenate(chunks, axis=0).astype(output_dtype)
        inverse = unpad[point.serialized.inverse[self.order_index]]
        output = mx.take(ordered_output, mx.array(inverse, dtype=mx.int32), axis=0)
        return _linear(output, self.proj_weight, self.proj_bias)


class SonataBlockMLX:
    def __init__(
        self,
        weights: dict[str, Any],
        prefix: str,
        channels: int,
        heads: int,
        order_index: int,
        *,
        official_attention_precision: bool = False,
    ) -> None:
        self.cpe_conv = SubMConv3dMLX(channels, channels, kernel_size=3, bias=True)
        self.cpe_conv.weight = weights[f"{prefix}.cpe.0.weight"]
        self.cpe_conv.bias = weights[f"{prefix}.cpe.0.bias"]
        self.cpe_linear_weight = weights[f"{prefix}.cpe.1.weight"]
        self.cpe_linear_bias = weights[f"{prefix}.cpe.1.bias"]
        self.cpe_norm_weight = weights[f"{prefix}.cpe.2.weight"]
        self.cpe_norm_bias = weights[f"{prefix}.cpe.2.bias"]
        self.norm1_weight = weights[f"{prefix}.norm1.0.weight"]
        self.norm1_bias = weights[f"{prefix}.norm1.0.bias"]
        self.norm2_weight = weights[f"{prefix}.norm2.0.weight"]
        self.norm2_bias = weights[f"{prefix}.norm2.0.bias"]
        self.attention = SerializedAttentionMLX(
            weights,
            f"{prefix}.attn",
            channels,
            heads,
            order_index,
            official_attention_precision=official_attention_precision,
        )
        self.mlp_fc1_weight = weights[f"{prefix}.mlp.0.fc1.weight"]
        self.mlp_fc1_bias = weights[f"{prefix}.mlp.0.fc1.bias"]
        self.mlp_fc2_weight = weights[f"{prefix}.mlp.0.fc2.weight"]
        self.mlp_fc2_bias = weights[f"{prefix}.mlp.0.fc2.bias"]

    def __call__(self, point: SonataPointMLX) -> SonataPointMLX:
        if point.neighbor_map is None:
            coordinates = np.concatenate([point.batch[:, None], point.grid_coord], axis=1)
            point.neighbor_map = build_subm_neighbor_map(coordinates)
        shortcut = point.feat
        cpe = self.cpe_conv(point.feat, np.empty((0, 4), dtype=np.int64), neighbor_map=point.neighbor_map)
        cpe = _linear(cpe, self.cpe_linear_weight, self.cpe_linear_bias)
        point.feat = shortcut + _layer_norm(cpe, self.cpe_norm_weight, self.cpe_norm_bias)

        shortcut = point.feat
        normalized = _layer_norm(point.feat, self.norm1_weight, self.norm1_bias)
        point.feat = normalized
        attended = self.attention(point)
        point.feat = shortcut + attended

        shortcut = point.feat
        normalized = _layer_norm(point.feat, self.norm2_weight, self.norm2_bias)
        hidden = _gelu(_linear(normalized, self.mlp_fc1_weight, self.mlp_fc1_bias))
        point.feat = shortcut + _linear(hidden, self.mlp_fc2_weight, self.mlp_fc2_bias)
        return point


class GridPoolingMLX:
    def __init__(self, weights: dict[str, Any], prefix: str, stride: int = 2) -> None:
        self.stride = stride
        self.proj_weight = weights[f"{prefix}.proj.weight"]
        self.proj_bias = weights[f"{prefix}.proj.bias"]
        self.norm_weight = weights[f"{prefix}.norm.0.weight"]
        self.norm_bias = weights[f"{prefix}.norm.0.bias"]

    def __call__(self, point: SonataPointMLX, *, permutation: np.ndarray | None = None) -> SonataPointMLX:
        pooled_grid = np.floor_divide(point.grid_coord, self.stride)
        keys = np.concatenate([point.batch[:, None], pooled_grid], axis=1)
        unique_keys, cluster, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        sorted_indices = np.argsort(cluster, kind="stable")
        starts = np.cumsum(np.insert(counts, 0, 0)[:-1])
        max_count = int(counts.max())
        members = np.full((len(counts), max_count), -1, dtype=np.int32)
        for group, (start, count) in enumerate(zip(starts, counts, strict=True)):
            members[group, : int(count)] = sorted_indices[int(start) : int(start + count)]
        valid = members >= 0
        safe_members = np.where(valid, members, 0)
        projected = _linear(point.feat, self.proj_weight, self.proj_bias)
        gathered = mx.take(projected, mx.array(safe_members, dtype=mx.int32), axis=0)
        gathered = mx.where(mx.array(valid)[..., None], gathered, -mx.inf)
        pooled_feat = mx.max(gathered, axis=1)
        pooled_feat = _gelu(_layer_norm(pooled_feat, self.norm_weight, self.norm_bias))
        pooled_coord = np.add.reduceat(point.coord[sorted_indices], starts, axis=0) / counts[:, None]
        pooled_batch = unique_keys[:, 0].astype(np.int64, copy=False)
        pooled_grid = unique_keys[:, 1:].astype(np.int64, copy=False)
        pooled_counts = np.bincount(pooled_batch, minlength=int(pooled_batch.max()) + 1)
        pooled_offset = np.cumsum(pooled_counts, dtype=np.int64)
        serialized = serialize_points(
            pooled_grid,
            pooled_batch,
            orders=SERIALIZATION_ORDERS,
            permutation=permutation,
        )
        return SonataPointMLX(
            feat=pooled_feat,
            coord=pooled_coord,
            grid_coord=pooled_grid,
            batch=pooled_batch,
            offset=pooled_offset,
            serialized=serialized,
            pooling_parent=point,
            pooling_inverse=cluster,
        )


class SonataFeatureExtractorMLX:
    """Native MLX port of the released encoder-only Sonata feature extractor."""

    def __init__(
        self,
        weights: dict[str, Any],
        prefix: str = "seg_feat_encoder",
        *,
        official_attention_precision: bool = False,
    ) -> None:
        _require_mlx()
        root = f"{prefix}." if prefix else ""
        sonata = f"{root}sonata"
        self.embedding_weight = weights[f"{sonata}.embedding.stem.linear.weight"]
        self.embedding_bias = weights[f"{sonata}.embedding.stem.linear.bias"]
        self.embedding_norm_weight = weights[f"{sonata}.embedding.stem.norm.weight"]
        self.embedding_norm_bias = weights[f"{sonata}.embedding.stem.norm.bias"]
        self.stages: list[tuple[GridPoolingMLX | None, list[SonataBlockMLX]]] = []
        for stage, (channels, depth, heads) in enumerate(zip(ENC_CHANNELS, ENC_DEPTHS, ENC_HEADS, strict=True)):
            stage_prefix = f"{sonata}.enc.enc{stage}"
            pooling = GridPoolingMLX(weights, f"{stage_prefix}.down") if stage else None
            blocks = [
                SonataBlockMLX(
                    weights,
                    f"{stage_prefix}.block{block}",
                    channels,
                    heads,
                    block % 4,
                    official_attention_precision=official_attention_precision,
                )
                for block in range(depth)
            ]
            self.stages.append((pooling, blocks))
        self.output_weights = [weights[f"{root}mlp.{index}.weight"] for index in (0, 2, 4)]
        self.output_biases = [weights[f"{root}mlp.{index}.bias"] for index in (0, 2, 4)]

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
        prefix: str = "seg_feat_encoder",
        *,
        official_attention_precision: bool = False,
    ) -> SonataFeatureExtractorMLX:
        _require_mlx()
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        weights = mx.load(str(path))
        selected = {key: value for key, value in weights.items() if key.startswith(f"{prefix}.")} if prefix else weights
        if not selected:
            raise ValueError(f"no tensors with prefix {prefix!r} in {path}")
        return cls(
            selected,
            prefix=prefix,
            official_attention_precision=official_attention_precision,
        )

    def __call__(
        self,
        data: SonataInput,
        *,
        order_permutations: tuple[np.ndarray, ...] = PAPER_SEED_42_ORDER_PERMUTATIONS,
    ) -> Any:
        if len(order_permutations) != len(self.stages):
            raise ValueError(f"expected {len(self.stages)} order permutations, got {len(order_permutations)}")
        feat = mx.array(data.feat, dtype=mx.float32)
        feat = _gelu(
            _layer_norm(
                _linear(feat, self.embedding_weight, self.embedding_bias),
                self.embedding_norm_weight,
                self.embedding_norm_bias,
            )
        )
        point = SonataPointMLX(
            feat=feat,
            coord=data.coord,
            grid_coord=data.grid_coord,
            batch=data.batch,
            offset=data.offset,
            serialized=serialize_points(
                data.grid_coord,
                data.batch,
                orders=SERIALIZATION_ORDERS,
                permutation=order_permutations[0],
            ),
        )
        for stage_index, (pooling, blocks) in enumerate(self.stages):
            if pooling is not None:
                point = pooling(point, permutation=order_permutations[stage_index])
            for block in blocks:
                point = block(point)
            mx.eval(point.feat)

        while point.pooling_parent is not None:
            parent = point.pooling_parent
            if point.pooling_inverse is None:
                raise RuntimeError("pooling hierarchy is missing its inverse map")
            upsampled = mx.take(point.feat, mx.array(point.pooling_inverse, dtype=mx.int32), axis=0)
            parent.feat = mx.concatenate([parent.feat, upsampled], axis=-1)
            point = parent
        output = point.feat
        for index, (weight, bias) in enumerate(zip(self.output_weights, self.output_biases, strict=True)):
            output = _linear(output, weight, bias)
            if index < 2:
                output = _gelu(output)
        output = mx.take(output, mx.array(data.inverse, dtype=mx.int32), axis=0)
        mx.eval(output)
        return output
