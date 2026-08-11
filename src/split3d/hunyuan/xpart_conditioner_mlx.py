from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from split3d.hunyuan.sonata_data import prepare_sonata_input
from split3d.hunyuan.sonata_mlx import SonataFeatureExtractorMLX
from split3d.hunyuan.xpart_shape_mlx import PointCrossAttentionEncoderMLX

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # pragma: no cover - MLX is available only on Apple silicon
    mx = None
    nn = None


def _require_mlx() -> None:
    if mx is None or nn is None:
        raise RuntimeError("X-Part Conditioner requires Apple silicon and MLX")


class XPartConditionerMLX:
    def __init__(self, weights: dict[str, Any]) -> None:
        _require_mlx()
        self.geo_encoder = PointCrossAttentionEncoderMLX.from_weights(
            weights,
            "geo_encoder.local_encoder.encoder",
        )
        self.obj_encoder = PointCrossAttentionEncoderMLX.from_weights(weights, "obj_encoder.encoder")
        self.seg_feat_encoder = SonataFeatureExtractorMLX(weights, prefix="seg_feat_encoder")
        self.geo_out_proj = nn.Linear(1536, 1024)
        self.obj_out_proj = nn.Linear(1536, 1024)
        self.geo_out_proj.load_weights(
            [
                ("weight", weights["geo_out_proj.weight"]),
                ("bias", weights["geo_out_proj.bias"]),
            ],
            strict=True,
        )
        self.obj_out_proj.load_weights(
            [
                ("weight", weights["obj_out_proj.weight"]),
                ("bias", weights["obj_out_proj.bias"]),
            ],
            strict=True,
        )
        mx.eval(self.geo_out_proj.parameters(), self.obj_out_proj.parameters())

    @classmethod
    def from_safetensors(cls, path: str | Path) -> XPartConditionerMLX:
        _require_mlx()
        return cls(mx.load(str(path)))

    def __call__(
        self,
        part_surface_inbbox: np.ndarray,
        object_surface: np.ndarray,
        *,
        seed: int = 42,
    ) -> dict[str, Any]:
        parts = np.asarray(part_surface_inbbox, dtype=np.float32)
        object_array = np.asarray(object_surface, dtype=np.float32)
        if parts.ndim != 3 or parts.shape[-1] != 7:
            raise ValueError(f"part_surface_inbbox must have shape [K, N, 7], got {parts.shape}")
        if object_array.ndim == 2:
            object_array = object_array[None]
        if object_array.ndim != 3 or object_array.shape[-1] != 7 or object_array.shape[0] != 1:
            raise ValueError(f"object_surface must have shape [1, N, 7], got {object_array.shape}")

        geo_features, local_query_points = self.geo_encoder(parts)
        object_features, global_query_points = self.obj_encoder(object_array)
        part_count = parts.shape[0]
        object_features = mx.broadcast_to(object_features, (part_count, *object_features.shape[1:]))

        prepared = prepare_sonata_input(object_array[0, :, :3], object_array[0, :, 3:6], seed=seed)
        point_features = self.seg_feat_encoder(prepared)
        point_features_cpu = np.array(point_features.astype(mx.float32))
        tree = cKDTree(object_array[0, :, :3])
        global_indices = tree.query(global_query_points.reshape(-1, 3), workers=-1)[1]
        global_semantics = point_features_cpu[global_indices].reshape(global_query_points.shape[:2] + (512,))
        global_semantics = np.broadcast_to(global_semantics, (part_count, *global_semantics.shape[1:]))
        local_indices = tree.query(local_query_points.reshape(-1, 3), workers=-1)[1]
        local_semantics = point_features_cpu[local_indices].reshape(local_query_points.shape[:2] + (512,))

        object_context = mx.concatenate(
            [object_features, mx.array(global_semantics, dtype=object_features.dtype)], axis=-1
        ).astype(self.obj_out_proj.weight.dtype)
        geo_context = mx.concatenate(
            [geo_features, mx.array(local_semantics, dtype=geo_features.dtype)], axis=-1
        ).astype(self.geo_out_proj.weight.dtype)
        contexts = {
            "obj_cond": self.obj_out_proj(object_context),
            "geo_cond": self.geo_out_proj(geo_context),
        }
        mx.eval(contexts)
        return contexts
