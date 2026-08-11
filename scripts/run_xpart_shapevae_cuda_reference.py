from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CUDA ShapeVAE features and SDF queries for MLX comparison")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path(".upstream/hunyuan3d-part"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latents", type=Path)
    parser.add_argument("--query-count", type=int, default=64)
    args = parser.parse_args()

    pymeshlab = types.ModuleType("pymeshlab")
    pymeshlab.MeshSet = type("MeshSet", (), {})  # type: ignore[attr-defined]
    sys.modules.setdefault("pymeshlab", pymeshlab)
    sys.path.insert(0, str(args.upstream / "XPart"))
    from partgen.models.autoencoders.model import VolumeDecoderShapeVAE

    scale_factor = 1.0039506158752403
    model = VolumeDecoderShapeVAE(
        num_latents=1024,
        embed_dim=64,
        num_freqs=8,
        include_pi=False,
        heads=16,
        width=1024,
        num_encoder_layers=8,
        num_decoder_layers=16,
        qkv_bias=False,
        qk_norm=True,
        scale_factor=scale_factor,
        geo_decoder_mlp_expand_ratio=4,
        geo_decoder_downsample_ratio=1,
        geo_decoder_ln_post=True,
        point_feats=4,
        pc_size=81920,
        pc_sharpedge_size=0,
    ).to(device="cuda", dtype=torch.bfloat16)
    model.load_state_dict(load_file(args.weights), strict=True)
    model.eval()

    rng = np.random.default_rng(123)
    if args.latents is None:
        latents_np = rng.standard_normal((1, 1024, 64), dtype=np.float32) * 0.1
    else:
        latents_np = np.asarray(np.load(args.latents)[:1], dtype=np.float32)
    queries_np = rng.uniform(-1, 1, size=(1, args.query_count, 3)).astype(np.float32)
    latents = torch.from_numpy(latents_np).to(device="cuda", dtype=torch.bfloat16)
    queries = torch.from_numpy(queries_np).to(device="cuda", dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        features = model(latents / scale_factor)
        sdf = model.query_geometry(queries, features)
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "inputs.npz", latents=latents_np, queries=queries_np)
    np.save(args.output / "features.npy", features.float().cpu().numpy())
    np.save(args.output / "sdf.npy", sdf.float().cpu().numpy())
    runtime = {
        "seconds": time.perf_counter() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "feature_shape": list(features.shape),
        "sdf_shape": list(sdf.shape),
    }
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
