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
    parser = argparse.ArgumentParser(description="Run a deterministic CUDA X-Part point encoder comparison")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path(".upstream/hunyuan3d-part"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="obj_encoder.encoder")
    parser.add_argument("--query-tokens", type=int, default=8)
    parser.add_argument("--data-tokens", type=int, default=32)
    args = parser.parse_args()

    pymeshlab = types.ModuleType("pymeshlab")
    pymeshlab.MeshSet = type("MeshSet", (), {})  # type: ignore[attr-defined]
    sys.modules.setdefault("pymeshlab", pymeshlab)
    sys.path.insert(0, str(args.upstream / "XPart"))
    from partgen.models.autoencoders.attention_blocks import FourierEmbedder, PointCrossAttentionEncoder

    encoder = PointCrossAttentionEncoder(
        num_latents=args.query_tokens,
        downsample_ratio=4,
        pc_size=args.data_tokens,
        pc_sharpedge_size=0,
        fourier_embedder=FourierEmbedder(num_freqs=8, include_pi=False),
        point_feats=4,
        width=1024,
        heads=16,
        layers=8,
        qkv_bias=False,
        use_ln_post=True,
        qk_norm=True,
    ).to(device="cuda", dtype=torch.bfloat16)
    root = f"{args.prefix}."
    selected = {key[len(root) :]: value for key, value in load_file(args.weights).items() if key.startswith(root)}
    encoder.load_state_dict(selected, strict=True)
    encoder.eval()

    rng = np.random.default_rng(123)
    query_np = rng.standard_normal((1, args.query_tokens, 55), dtype=np.float32) * 0.1
    data_np = rng.standard_normal((1, args.data_tokens, 55), dtype=np.float32) * 0.1
    query = torch.from_numpy(query_np).to(device="cuda", dtype=torch.bfloat16)
    data = torch.from_numpy(data_np).to(device="cuda", dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = encoder.cross_attn(encoder.input_proj(query), encoder.input_proj(data))
        output = encoder.self_attn(output)
        output = encoder.ln_post(output)
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "inputs.npz", query=query_np, data=data_np)
    np.save(args.output / "output.npy", output.float().cpu().numpy())
    runtime = {
        "seconds": time.perf_counter() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "output_shape": list(output.shape),
    }
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
