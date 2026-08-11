from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from split3d.hunyuan.xpart_shape_mlx import ShapeVAEDecoderMLX


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLX ShapeVAE on saved CUDA comparison inputs")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = np.load(args.inputs)
    model = ShapeVAEDecoderMLX.from_safetensors(args.weights)
    latents = mx.array(inputs["latents"], dtype=mx.bfloat16)
    queries = mx.array(inputs["queries"], dtype=mx.bfloat16)
    mx.reset_peak_memory()
    started = time.perf_counter()
    features = model.decode_features(latents)
    sdf = model.query_sdf(queries, features)
    mx.eval(features, sdf)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "features.npy", np.array(features.astype(mx.float32)))
    np.save(args.output / "sdf.npy", np.array(sdf[..., 0].astype(mx.float32)))
    runtime = {
        "seconds": time.perf_counter() - started,
        "peak_memory_bytes": mx.get_peak_memory(),
        "feature_shape": list(features.shape),
        "sdf_shape": list(sdf.shape),
    }
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
