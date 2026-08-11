from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file


class _GeluProjection(nn.Module):
    def __init__(self, dim: int, inner_dim: int, *, bias: bool) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(self.proj(hidden_states))


class _FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        inner_dim: int | None = None,
        dropout: float = 0.0,
        final_dropout: bool = False,
        bias: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        inner_dim = inner_dim or dim * 4
        layers: list[nn.Module] = [
            _GeluProjection(dim, inner_dim, bias=bias),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=bias),
        ]
        if final_dropout:
            layers.append(nn.Dropout(dropout))
        self.net = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states


def _install_diffusers_feedforward_stub() -> None:
    diffusers = types.ModuleType("diffusers")
    models = types.ModuleType("diffusers.models")
    attention = types.ModuleType("diffusers.models.attention")
    attention.FeedForward = _FeedForward  # type: ignore[attr-defined]
    models.attention = attention  # type: ignore[attr-defined]
    diffusers.models = models  # type: ignore[attr-defined]
    sys.modules["diffusers"] = diffusers
    sys.modules["diffusers.models"] = models
    sys.modules["diffusers.models.attention"] = attention


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an exact small-token CUDA reference for X-Part PartFormer")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path(".upstream/hunyuan3d-part"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-tokens", type=int, default=8)
    parser.add_argument("--context-tokens", type=int, default=16)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _install_diffusers_feedforward_stub()
    sys.path.insert(0, str(args.upstream / "XPart" / "partgen"))
    from models.partformer_dit import PartFormerDITPlain

    model = PartFormerDITPlain(
        input_size=1024,
        in_channels=64,
        hidden_size=2048,
        encoder_hidden_dim=1024,
        encoder_hidden2_dim=1024,
        depth=21,
        num_heads=16,
        qk_norm=True,
        qkv_bias=False,
        qk_norm_type="rms",
        use_part_embed=True,
        valid_num=50,
        num_moe_layers=6,
        num_experts=8,
        moe_top_k=2,
    ).to(device="cuda", dtype=torch.bfloat16)
    model.load_state_dict(load_file(args.weights), strict=True)
    model.eval()

    rng = np.random.default_rng(123)
    latents_np = rng.standard_normal((args.parts, args.latent_tokens, 64), dtype=np.float32) * 0.1
    object_np = rng.standard_normal((args.parts, args.context_tokens, 1024), dtype=np.float32) * 0.1
    geo_np = rng.standard_normal((args.parts, args.context_tokens, 1024), dtype=np.float32) * 0.1
    torch.manual_seed(args.seed)
    part_indices = torch.randperm(50)[: args.parts].numpy()
    torch.manual_seed(args.seed)
    latents = torch.from_numpy(latents_np).to(device="cuda", dtype=torch.bfloat16)
    object_context = torch.from_numpy(object_np).to(device="cuda", dtype=torch.bfloat16)
    geo_context = torch.from_numpy(geo_np).to(device="cuda", dtype=torch.bfloat16)
    timestep = torch.zeros((args.parts,), device="cuda", dtype=torch.bfloat16)
    aabb = torch.zeros((1, args.parts, 2, 3), device="cuda", dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(
            latents,
            timestep,
            {"obj_cond": object_context, "geo_cond": geo_context},
            aabb=aabb,
            num_tokens=torch.full((1, args.parts), args.latent_tokens, device="cuda"),
        )
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "inputs.npz", latents=latents_np, obj_cond=object_np, geo_cond=geo_np)
    np.save(args.output / "part_indices.npy", part_indices)
    np.save(args.output / "output.npy", output.float().cpu().numpy())
    runtime = {
        "seconds": time.perf_counter() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "output_shape": list(output.shape),
        "part_indices": part_indices.tolist(),
    }
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
