from __future__ import annotations

import math
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
        raise RuntimeError("X-Part PartFormer requires Apple silicon and MLX")


class Timesteps(nn.Module if nn is not None else object):
    def __init__(self, num_channels: int, max_period: int = 10_000) -> None:
        _require_mlx()
        super().__init__()
        self.num_channels = num_channels
        self.max_period = max_period

    def __call__(self, timesteps: Any) -> Any:
        half = self.num_channels // 2
        exponent = -math.log(self.max_period) * mx.arange(half, dtype=mx.float32) / half
        angles = timesteps[:, None].astype(mx.float32) * mx.exp(exponent)[None, :]
        output = mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)
        if self.num_channels % 2:
            output = mx.pad(output, ((0, 0), (0, 1)))
        return output


class _Gelu(nn.Module if nn is not None else object):
    def __call__(self, x: Any) -> Any:
        return nn.gelu(x)


class TimestepEmbedder(nn.Module if nn is not None else object):
    def __init__(self, hidden_size: int, frequency_embedding_size: int) -> None:
        _require_mlx()
        super().__init__()
        self.time_embed = Timesteps(hidden_size)
        self.mlp = [
            nn.Linear(hidden_size, frequency_embedding_size),
            _Gelu(),
            nn.Linear(frequency_embedding_size, hidden_size),
        ]

    def __call__(self, timestep: Any) -> Any:
        output = self.time_embed(timestep).astype(self.mlp[0].weight.dtype)
        for layer in self.mlp:
            output = layer(output)
        return output[:, None, :]


class MLP(nn.Module if nn is not None else object):
    def __init__(self, width: int) -> None:
        _require_mlx()
        super().__init__()
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)

    def __call__(self, x: Any) -> Any:
        return self.fc2(nn.gelu(self.fc1(x)))


class Attention(nn.Module if nn is not None else object):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        use_global: bool = False,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_global = use_global
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x: Any) -> Any:
        old_batch, old_tokens, channels = x.shape
        if self.use_global:
            x = x.reshape(1, old_batch * old_tokens, channels)
        batch, tokens, _ = x.shape
        qkv = mx.concatenate([self.to_q(x), self.to_k(x), self.to_v(x)], axis=-1)
        qkv = qkv.reshape(batch, tokens, self.num_heads, 3 * self.head_dim)
        query, key, value = mx.split(qkv, 3, axis=-1)
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        if self.q_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        output = mx.fast.scaled_dot_product_attention(query, key, value, scale=self.scale)
        output = self.out_proj(output.transpose(0, 2, 1, 3).reshape(batch, tokens, -1))
        return output.reshape(old_batch, old_tokens, channels) if self.use_global else output


class CrossAttention(nn.Module if nn is not None else object):
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = False,
        qk_norm: bool = True,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, query_dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, query_dim, bias=qkv_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.out_proj = nn.Linear(query_dim, query_dim)

    def __call__(self, x: Any, context: Any) -> Any:
        batch, query_count, _ = x.shape
        context_count = context.shape[1]
        query = self.to_q(x).reshape(batch, query_count, self.num_heads, self.head_dim)
        key_value = mx.concatenate([self.to_k(context), self.to_v(context)], axis=-1)
        key_value = key_value.reshape(batch, context_count, self.num_heads, 2 * self.head_dim)
        key, value = mx.split(key_value, 2, axis=-1)
        if self.q_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        output = mx.fast.scaled_dot_product_attention(query, key, value, scale=self.scale)
        return self.out_proj(output.transpose(0, 2, 1, 3).reshape(batch, query_count, -1))


class _ExpertProjection(nn.Module if nn is not None else object):
    def __init__(self, width: int) -> None:
        _require_mlx()
        super().__init__()
        self.proj = nn.Linear(width, width * 4)

    def __call__(self, x: Any) -> Any:
        return nn.gelu(self.proj(x))


class _Identity(nn.Module if nn is not None else object):
    def __call__(self, x: Any) -> Any:
        return x


class Expert(nn.Module if nn is not None else object):
    def __init__(self, width: int) -> None:
        _require_mlx()
        super().__init__()
        self.net = [_ExpertProjection(width), _Identity(), nn.Linear(width * 4, width)]

    def __call__(self, x: Any) -> Any:
        for layer in self.net:
            x = layer(x)
        return x


class MoEGate(nn.Module if nn is not None else object):
    def __init__(self, width: int, num_experts: int) -> None:
        _require_mlx()
        super().__init__()
        self.weight = mx.zeros((num_experts, width))


class MoEBlock(nn.Module if nn is not None else object):
    def __init__(self, width: int, *, num_experts: int = 8, top_k: int = 2) -> None:
        _require_mlx()
        super().__init__()
        self.top_k = top_k
        self.experts = [Expert(width) for _ in range(num_experts)]
        self.gate = MoEGate(width, num_experts)
        self.shared_experts = Expert(width)

    def __call__(self, hidden_states: Any) -> Any:
        original_shape = hidden_states.shape
        flattened = hidden_states.reshape(-1, original_shape[-1])
        scores = mx.softmax(flattened @ self.gate.weight.T, axis=-1)
        indices = mx.argpartition(-scores, kth=self.top_k - 1, axis=-1)[:, : self.top_k]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        flat_indices = indices.reshape(-1)
        flat_weights = weights.reshape(-1)
        flat_indices_cpu = np.array(flat_indices, dtype=np.int32)
        output = mx.zeros_like(flattened)
        for expert_index, expert in enumerate(self.experts):
            selected_positions = mx.array(np.flatnonzero(flat_indices_cpu == expert_index), dtype=mx.int32)
            token_indices = selected_positions // self.top_k
            expert_input = mx.take(flattened, token_indices, axis=0)
            expert_output = expert(expert_input)
            expert_weight = mx.take(flat_weights, selected_positions, axis=0)[:, None]
            output = output.at[token_indices].add(expert_output * expert_weight)
        output = output + self.shared_experts(flattened)
        return output.reshape(original_shape)


class PartFormerBlock(nn.Module if nn is not None else object):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        global_attention: bool,
        skip_connection: bool,
        use_moe: bool,
        num_experts: int,
        moe_top_k: int,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn1 = Attention(hidden_size, num_heads, qk_norm=True, use_global=global_attention)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn2 = CrossAttention(hidden_size, 1024, num_heads, qk_norm=True)
        self.norm2_2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn2_2 = CrossAttention(hidden_size, 1024, num_heads, qk_norm=True)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.use_moe = use_moe
        if use_moe:
            self.moe = MoEBlock(hidden_size, num_experts=num_experts, top_k=moe_top_k)
        else:
            self.mlp = MLP(hidden_size)
        if skip_connection:
            self.skip_linear = nn.Linear(hidden_size * 2, hidden_size)
            self.skip_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        else:
            self.skip_linear = None

    def __call__(self, x: Any, object_context: Any, geo_context: Any, skip_value: Any | None = None) -> Any:
        if self.skip_linear is not None:
            x = self.skip_norm(self.skip_linear(mx.concatenate([skip_value, x], axis=-1)))
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), object_context) + self.attn2_2(self.norm2_2(x), geo_context)
        normalized = self.norm3(x)
        return x + (self.moe(normalized) if self.use_moe else self.mlp(normalized))


class FinalLayer(nn.Module if nn is not None else object):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        _require_mlx()
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)

    def __call__(self, x: Any) -> Any:
        return self.linear(self.norm_final(x)[:, 1:])


class PartFormerMLX(nn.Module if nn is not None else object):
    def __init__(
        self,
        *,
        input_size: int = 1024,
        in_channels: int = 64,
        hidden_size: int = 2048,
        depth: int = 21,
        num_heads: int = 16,
        valid_num: int = 50,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.input_size = input_size
        self.depth = depth
        self.valid_num = valid_num
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4)
        self.part_embed = mx.zeros((valid_num, hidden_size))
        self.blocks = [
            PartFormerBlock(
                hidden_size,
                num_heads,
                global_attention=(layer + 1) % 2 == 0,
                skip_connection=layer > depth // 2,
                use_moe=depth - layer <= num_moe_layers,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
            )
            for layer in range(depth)
        ]
        self.final_layer = FinalLayer(hidden_size, in_channels)

    @classmethod
    def from_safetensors(cls, path: str | Path) -> PartFormerMLX:
        _require_mlx()
        model = cls()
        weights = mx.load(str(path))
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
        return model

    def __call__(
        self,
        latents: Any,
        timestep: Any,
        contexts: dict[str, Any],
        *,
        part_indices: np.ndarray | None = None,
    ) -> Any:
        part_count = latents.shape[0]
        if part_count > self.valid_num:
            raise ValueError(f"X-Part supports at most {self.valid_num} parts, got {part_count}")
        if part_indices is None:
            part_indices = np.arange(part_count, dtype=np.int32)
        conditioning = self.t_embedder(timestep)
        x = self.x_embedder(latents)
        embeddings = mx.take(self.part_embed, mx.array(part_indices, dtype=mx.int32), axis=0)[:, None, :]
        x = mx.concatenate([conditioning, x + embeddings], axis=1)
        skip_values = []
        for layer, block in enumerate(self.blocks):
            skip_value = None if layer <= self.depth // 2 else skip_values.pop()
            x = block(x, contexts["obj_cond"], contexts["geo_cond"], skip_value)
            if layer < self.depth // 2:
                skip_values.append(x)
        output = self.final_layer(x)
        mx.eval(output)
        return output
