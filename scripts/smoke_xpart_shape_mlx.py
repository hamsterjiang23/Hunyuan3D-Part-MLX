from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from split3d.hunyuan.sonata_mlx import SonataFeatureExtractorMLX
from split3d.hunyuan.xpart_shape_mlx import PointCrossAttentionEncoderMLX, ShapeVAEDecoderMLX


def _parameter_count(model: object) -> int:
    return sum(int(value.size) for _, value in tree_flatten(model.parameters()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Load all X-Part MLX geometry modules with strict key checks")
    parser.add_argument("--conditioner", type=Path, required=True)
    parser.add_argument("--shapevae", type=Path, required=True)
    args = parser.parse_args()

    weights = mx.load(str(args.conditioner))
    geo_encoder = PointCrossAttentionEncoderMLX.from_weights(
        weights,
        "geo_encoder.local_encoder.encoder",
    )
    obj_encoder = PointCrossAttentionEncoderMLX.from_weights(weights, "obj_encoder.encoder")
    sonata = SonataFeatureExtractorMLX(weights, prefix="seg_feat_encoder")
    shapevae = ShapeVAEDecoderMLX.from_safetensors(args.shapevae)
    mx.eval(geo_encoder.parameters(), obj_encoder.parameters(), shapevae.parameters())
    print(
        json.dumps(
            {
                "geo_encoder_parameters": _parameter_count(geo_encoder),
                "obj_encoder_parameters": _parameter_count(obj_encoder),
                "sonata_loaded": sonata is not None,
                "shapevae_parameters": _parameter_count(shapevae),
                "peak_memory_bytes": mx.get_peak_memory(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
