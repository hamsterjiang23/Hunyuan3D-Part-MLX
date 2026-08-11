from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from split3d.hunyuan.xpart_partformer_mlx import PartFormerMLX


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLX PartFormer with saved CUDA-reference inputs")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--part-indices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = np.load(args.inputs)
    model = PartFormerMLX.from_safetensors(args.weights)
    latents = mx.array(inputs["latents"], dtype=mx.bfloat16)
    timestep = mx.zeros((latents.shape[0],), dtype=mx.bfloat16)
    contexts = {
        "obj_cond": mx.array(inputs["obj_cond"], dtype=mx.bfloat16),
        "geo_cond": mx.array(inputs["geo_cond"], dtype=mx.bfloat16),
    }
    mx.reset_peak_memory()
    started = time.perf_counter()
    output = model(latents, timestep, contexts, part_indices=np.load(args.part_indices))
    mx.eval(output)
    runtime = {
        "seconds": time.perf_counter() - started,
        "peak_memory_bytes": mx.get_peak_memory(),
        "output_shape": list(output.shape),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "output.npy", np.array(output.astype(mx.float32)))
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
