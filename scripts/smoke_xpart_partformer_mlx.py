from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from split3d.hunyuan.xpart_partformer_mlx import PartFormerMLX


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict-load and forward-smoke X-Part PartFormer on MLX")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--latent-tokens", type=int, default=8)
    parser.add_argument("--context-tokens", type=int, default=16)
    args = parser.parse_args()

    model = PartFormerMLX.from_safetensors(args.weights)
    parameters = sum(int(value.size) for _, value in tree_flatten(model.parameters()))
    latents = mx.zeros((1, args.latent_tokens, 64), dtype=mx.bfloat16)
    timestep = mx.zeros((1,), dtype=mx.bfloat16)
    context = mx.zeros((1, args.context_tokens, 1024), dtype=mx.bfloat16)
    output = model(latents, timestep, {"obj_cond": context, "geo_cond": context})
    mx.eval(output)
    print(
        json.dumps(
            {
                "parameters": parameters,
                "output_shape": list(output.shape),
                "output_finite": bool(mx.all(mx.isfinite(output)).item()),
                "peak_memory_bytes": mx.get_peak_memory(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
