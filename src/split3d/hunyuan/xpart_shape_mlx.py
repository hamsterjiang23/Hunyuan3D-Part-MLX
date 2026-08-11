from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # pragma: no cover - MLX is available only on Apple silicon
    mx = None
    nn = None


def _require_mlx() -> None:
    if mx is None or nn is None:
        raise RuntimeError("X-Part MLX modules require Apple silicon and MLX")


class FourierEmbedder(nn.Module if nn is not None else object):
    def __init__(
        self,
        num_freqs: int = 8,
        *,
        input_dim: int = 3,
        include_input: bool = True,
        include_pi: bool = False,
    ) -> None:
        _require_mlx()
        super().__init__()
        frequencies = 2.0 ** np.arange(num_freqs, dtype=np.float32)
        self.frequencies = frequencies * np.pi if include_pi else frequencies
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.out_dim = input_dim * (num_freqs * 2 + int(include_input or num_freqs == 0))

    def __call__(self, x: Any) -> Any:
        if self.num_freqs == 0:
            return x
        frequencies = mx.array(self.frequencies, dtype=x.dtype)
        embedded = (x[..., None] * frequencies).reshape(*x.shape[:-1], -1)
        encoded = [mx.sin(embedded), mx.cos(embedded)]
        if self.include_input:
            encoded.insert(0, x)
        return mx.concatenate(encoded, axis=-1)


class _QKNormGroup(nn.Module if nn is not None else object):
    def __init__(self, head_dim: int) -> None:
        _require_mlx()
        super().__init__()
        self.q_norm = nn.LayerNorm(head_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)


class SelfAttention(nn.Module if nn is not None else object):
    def __init__(self, width: int, heads: int, *, qkv_bias: bool, qk_norm: bool) -> None:
        _require_mlx()
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim**-0.5
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = _QKNormGroup(self.head_dim) if qk_norm else None

    def __call__(self, x: Any) -> Any:
        batch, tokens, _ = x.shape
        qkv = self.c_qkv(x).reshape(batch, tokens, self.heads, 3 * self.head_dim)
        query, key, value = mx.split(qkv, 3, axis=-1)
        if self.attention is not None:
            query = self.attention.q_norm(query)
            key = self.attention.k_norm(key)
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        output = mx.fast.scaled_dot_product_attention(query, key, value, scale=self.scale)
        return self.c_proj(output.transpose(0, 2, 1, 3).reshape(batch, tokens, -1))


class VaeMLP(nn.Module if nn is not None else object):
    def __init__(self, width: int, expand_ratio: int = 4) -> None:
        _require_mlx()
        super().__init__()
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, width)

    def __call__(self, x: Any) -> Any:
        return self.c_proj(nn.gelu(self.c_fc(x)))


class ResidualAttentionBlock(nn.Module if nn is not None else object):
    def __init__(self, width: int, heads: int, *, qkv_bias: bool, qk_norm: bool) -> None:
        _require_mlx()
        super().__init__()
        self.attn = SelfAttention(width, heads, qkv_bias=qkv_bias, qk_norm=qk_norm)
        self.ln_1 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = VaeMLP(width)
        self.ln_2 = nn.LayerNorm(width, eps=1e-6)

    def __call__(self, x: Any) -> Any:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module if nn is not None else object):
    def __init__(self, width: int, layers: int, heads: int, *, qkv_bias: bool, qk_norm: bool) -> None:
        _require_mlx()
        super().__init__()
        self.resblocks = [
            ResidualAttentionBlock(width, heads, qkv_bias=qkv_bias, qk_norm=qk_norm) for _ in range(layers)
        ]

    def __call__(self, x: Any) -> Any:
        for block in self.resblocks:
            x = block(x)
        return x


class _CrossAttentionProjections(nn.Module if nn is not None else object):
    def __init__(self, width: int, data_width: int, heads: int, *, qkv_bias: bool, qk_norm: bool) -> None:
        _require_mlx()
        super().__init__()
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = _QKNormGroup(width // heads) if qk_norm else None


class ResidualCrossAttentionBlock(nn.Module if nn is not None else object):
    def __init__(
        self,
        width: int,
        heads: int,
        *,
        data_width: int | None = None,
        qkv_bias: bool,
        qk_norm: bool,
        mlp_expand_ratio: int = 4,
    ) -> None:
        _require_mlx()
        super().__init__()
        data_width = width if data_width is None else data_width
        self.heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim**-0.5
        self.attn = _CrossAttentionProjections(width, data_width, heads, qkv_bias=qkv_bias, qk_norm=qk_norm)
        self.ln_1 = nn.LayerNorm(width, eps=1e-6)
        self.ln_2 = nn.LayerNorm(data_width, eps=1e-6)
        self.ln_3 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = VaeMLP(width, mlp_expand_ratio)

    def __call__(self, x: Any, data: Any) -> Any:
        batch, query_count, _ = x.shape
        data_count = data.shape[1]
        query = self.attn.c_q(self.ln_1(x)).reshape(batch, query_count, self.heads, self.head_dim)
        key_value = self.attn.c_kv(self.ln_2(data)).reshape(
            batch, data_count, self.heads, 2 * self.head_dim
        )
        key, value = mx.split(key_value, 2, axis=-1)
        if self.attn.attention is not None:
            query = self.attn.attention.q_norm(query)
            key = self.attn.attention.k_norm(key)
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        output = mx.fast.scaled_dot_product_attention(query, key, value, scale=self.scale)
        x = x + self.attn.c_proj(output.transpose(0, 2, 1, 3).reshape(batch, query_count, -1))
        return x + self.mlp(self.ln_3(x))


class PointCrossAttentionEncoderMLX(nn.Module if nn is not None else object):
    def __init__(
        self,
        *,
        num_latents: int = 4096,
        width: int = 1024,
        heads: int = 16,
        layers: int = 8,
        point_feats: int = 4,
        num_freqs: int = 8,
        include_pi: bool = False,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        use_ln_post: bool = True,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.num_latents = num_latents
        self.point_feats = point_feats
        self.fourier_embedder = FourierEmbedder(num_freqs, include_pi=include_pi)
        self.input_proj = nn.Linear(self.fourier_embedder.out_dim + point_feats, width)
        self.cross_attn = ResidualCrossAttentionBlock(
            width,
            heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )
        self.self_attn = Transformer(width, layers, heads, qkv_bias=qkv_bias, qk_norm=qk_norm)
        self.ln_post = nn.LayerNorm(width) if use_ln_post else None

    @classmethod
    def from_weights(cls, weights: dict[str, Any], prefix: str, **config: Any) -> PointCrossAttentionEncoderMLX:
        model = cls(**config)
        root = f"{prefix}."
        selected = [(key[len(root) :], value) for key, value in weights.items() if key.startswith(root)]
        if not selected:
            raise ValueError(f"no encoder tensors found under {prefix!r}")
        model.load_weights(selected, strict=True)
        mx.eval(model.parameters())
        return model

    def sample_queries(self, surface: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import fpsample

        coordinates = np.asarray(surface[..., :3], dtype=np.float32)
        features = np.asarray(surface[..., 3 : 3 + self.point_feats], dtype=np.float32)
        query_indices = np.stack(
            [fpsample.fps_sampling(points, self.num_latents, start_idx=0) for points in coordinates]
        )
        query_coordinates = np.take_along_axis(coordinates, query_indices[..., None], axis=1)
        query_features = np.take_along_axis(features, query_indices[..., None], axis=1)
        return query_coordinates, query_features, query_indices

    def __call__(self, surface: np.ndarray) -> tuple[Any, np.ndarray]:
        coordinates = np.asarray(surface[..., :3], dtype=np.float32)
        features = np.asarray(surface[..., 3 : 3 + self.point_feats], dtype=np.float32)
        query_coordinates, query_features, _ = self.sample_queries(surface)
        query = mx.concatenate(
            [self.fourier_embedder(mx.array(query_coordinates)), mx.array(query_features)], axis=-1
        )
        data = mx.concatenate([self.fourier_embedder(mx.array(coordinates)), mx.array(features)], axis=-1)
        query = query.astype(self.input_proj.weight.dtype)
        data = data.astype(self.input_proj.weight.dtype)
        latents = self.cross_attn(self.input_proj(query), self.input_proj(data))
        latents = self.self_attn(latents)
        if self.ln_post is not None:
            latents = self.ln_post(latents)
        mx.eval(latents)
        return latents, query_coordinates


class CrossAttentionDecoder(nn.Module if nn is not None else object):
    def __init__(
        self,
        fourier_embedder: FourierEmbedder,
        *,
        width: int = 1024,
        heads: int = 16,
        qkv_bias: bool = False,
        qk_norm: bool = True,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.fourier_embedder = fourier_embedder
        self.query_proj = nn.Linear(fourier_embedder.out_dim, width)
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            width,
            heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )
        self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, 1)

    def __call__(self, queries: Any, latents: Any) -> Any:
        query_embeddings = self.query_proj(self.fourier_embedder(queries).astype(latents.dtype))
        output = self.cross_attn_decoder(query_embeddings, latents)
        return self.output_proj(self.ln_post(output))


class ShapeVAEDecoderMLX(nn.Module if nn is not None else object):
    def __init__(
        self,
        *,
        num_latents: int = 1024,
        embed_dim: int = 64,
        width: int = 1024,
        heads: int = 16,
        num_decoder_layers: int = 16,
        num_freqs: int = 8,
        include_pi: bool = False,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        scale_factor: float = 1.0039506158752403,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.scale_factor = scale_factor
        self.latent_shape = (num_latents, embed_dim)
        self.pre_kl = nn.Linear(width, embed_dim * 2)
        self.post_kl = nn.Linear(embed_dim, width)
        self.transformer = Transformer(
            width,
            num_decoder_layers,
            heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )
        fourier = FourierEmbedder(num_freqs, include_pi=include_pi)
        self.geo_decoder = CrossAttentionDecoder(
            fourier,
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )

    @classmethod
    def from_safetensors(cls, path: str | Path, **config: Any) -> ShapeVAEDecoderMLX:
        _require_mlx()
        model = cls(**config)
        weights = mx.load(str(path))
        selected = [(key, value) for key, value in weights.items() if not key.startswith("encoder.")]
        model.load_weights(selected, strict=True)
        mx.eval(model.parameters())
        return model

    def decode_features(self, latents: Any) -> Any:
        features = self.transformer(self.post_kl(latents / self.scale_factor))
        mx.eval(features)
        return features

    def query_sdf(self, queries: Any, features: Any) -> Any:
        return self.geo_decoder(queries, features)

    def _query_sdf_points(self, points: np.ndarray, features: Any, chunk_size: int) -> np.ndarray:
        outputs = []
        for start in range(0, len(points), chunk_size):
            queries = mx.array(points[start : start + chunk_size])[None].astype(features.dtype)
            logits = self.query_sdf(queries, features)
            mx.eval(logits)
            outputs.append(np.array(logits[0, :, 0].astype(mx.float32)))
        return np.concatenate(outputs)

    @staticmethod
    def _near_surface_mask(volume: np.ndarray, level: float) -> np.ndarray:
        shifted = volume + level
        valid = shifted > -9000
        sign = np.sign(shifted.astype(np.float32))
        same_sign = np.ones(volume.shape, dtype=bool)
        for axis in range(3):
            for shift in (1, -1):
                neighbor = np.roll(shifted, -shift, axis=axis).astype(np.float32)
                boundary = [slice(None)] * 3
                boundary[axis] = slice(-1, None) if shift > 0 else slice(0, 1)
                neighbor[tuple(boundary)] = shifted[tuple(boundary)]
                invalid = neighbor <= -9000
                neighbor[invalid] = shifted[invalid]
                same_sign &= np.sign(neighbor) == sign
        return np.logical_and(~same_sign, valid).astype(np.int32)

    @staticmethod
    def _dilate(volume: np.ndarray) -> np.ndarray:
        from scipy.ndimage import maximum_filter

        return maximum_filter(volume, size=3)

    def decode_to_mesh(
        self,
        latents: Any,
        *,
        bounds: float = 1.01,
        octree_resolution: int = 256,
        chunk_size: int = 100_000,
        mc_level: float = -1 / 512,
        min_resolution: int = 63,
    ) -> Any:
        import trimesh
        from skimage.measure import marching_cubes

        features = self.decode_features(latents)
        bbox_min = np.full(3, -bounds, dtype=np.float32)
        bbox_max = np.full(3, bounds, dtype=np.float32)
        bbox_size = bbox_max - bbox_min
        resolutions = []
        resolution = octree_resolution
        if resolution < min_resolution:
            resolutions.append(resolution)
        while resolution >= min_resolution:
            resolutions.append(resolution)
            resolution //= 2
        resolutions.reverse()

        coarse = resolutions[0]
        axes = [np.linspace(bbox_min[index], bbox_max[index], coarse + 1, dtype=np.float32) for index in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        grid_logits = self._query_sdf_points(grid.reshape(-1, 3), features, chunk_size).reshape(
            (coarse + 1,) * 3
        )
        final_mask = None
        for current in resolutions[1:]:
            next_shape = (current + 1,) * 3
            next_logits = np.full(next_shape, -10_000.0, dtype=np.float32)
            mask = self._near_surface_mask(grid_logits, mc_level)
            mask += (np.abs(grid_logits) < 0.95).astype(np.int32)
            expand = 0 if current == resolutions[-1] else 1
            for _ in range(expand):
                mask = self._dilate(mask)
            coarse_indices = np.where(mask > 0)
            next_index = np.zeros(next_shape, dtype=np.uint8)
            fine_indices = tuple(np.clip(indices * 2, 0, current) for indices in coarse_indices)
            next_index[fine_indices] = 1
            for _ in range(2 - expand):
                next_index = self._dilate(next_index)
            selected = np.where(next_index > 0)
            selected_grid = np.stack(selected, axis=1).astype(np.float32)
            points = selected_grid * (bbox_size / current) + bbox_min
            next_logits[selected] = self._query_sdf_points(points, features, chunk_size)
            grid_logits = next_logits
            final_mask = next_index > 0

        grid_logits[grid_logits == -10_000.0] = np.nan
        marching_volume = np.nan_to_num(
            grid_logits,
            nan=mc_level - 1.0,
            posinf=mc_level + 1.0,
            neginf=mc_level - 1.0,
        )
        vertices, faces, _, _ = marching_cubes(
            marching_volume,
            level=mc_level,
            method="lewiner",
            mask=final_mask,
        )
        grid_resolution = np.asarray(grid_logits.shape, dtype=np.float32) - 1
        vertices = vertices / grid_resolution * bbox_size + bbox_min
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
