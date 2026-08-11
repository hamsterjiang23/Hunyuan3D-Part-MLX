from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import trimesh

from split3d.hunyuan.xpart_conditioner_mlx import XPartConditionerMLX
from split3d.hunyuan.xpart_pipeline_mlx import normalize_mesh, sample_surface


def main() -> None:
    parser = argparse.ArgumentParser(description="Run X-Part Conditioner on a real sampled mesh")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--surface-points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh, force="mesh")
    mesh, _, _ = normalize_mesh(mesh)
    surface = sample_surface(mesh, args.surface_points, seed=args.seed)
    model = XPartConditionerMLX.from_safetensors(args.weights)
    started = time.perf_counter()
    context = model(surface[None], surface[None], seed=args.seed)
    mx.eval(context)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "seconds": elapsed,
                "obj_cond_shape": list(context["obj_cond"].shape),
                "geo_cond_shape": list(context["geo_cond"].shape),
                "finite": bool(
                    mx.all(mx.isfinite(context["obj_cond"])).item()
                    and mx.all(mx.isfinite(context["geo_cond"])).item()
                ),
                "peak_memory_bytes": mx.get_peak_memory(),
                "obj_cond_abs_max": float(mx.max(mx.abs(context["obj_cond"])).item()),
                "geo_cond_abs_max": float(mx.max(mx.abs(context["geo_cond"])).item()),
                "surface_checksum": float(np.sum(surface)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
