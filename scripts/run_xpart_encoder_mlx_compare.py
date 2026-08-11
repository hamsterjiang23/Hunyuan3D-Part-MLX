from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from split3d.hunyuan.xpart_shape_mlx import PointCrossAttentionEncoderMLX


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLX X-Part point encoder on saved CUDA inputs")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="obj_encoder.encoder")
    args = parser.parse_args()

    weights = mx.load(str(args.weights))
    encoder = PointCrossAttentionEncoderMLX.from_weights(weights, args.prefix, num_latents=8)
    inputs = np.load(args.inputs)
    query = mx.array(inputs["query"], dtype=mx.bfloat16)
    data = mx.array(inputs["data"], dtype=mx.bfloat16)
    mx.reset_peak_memory()
    started = time.perf_counter()
    output = encoder.cross_attn(encoder.input_proj(query), encoder.input_proj(data))
    output = encoder.self_attn(output)
    output = encoder.ln_post(output)
    mx.eval(output)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "output.npy", np.array(output.astype(mx.float32)))
    runtime = {
        "seconds": time.perf_counter() - started,
        "peak_memory_bytes": mx.get_peak_memory(),
        "output_shape": list(output.shape),
    }
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
