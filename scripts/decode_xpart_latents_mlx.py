from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import trimesh

from split3d.hunyuan.xpart_pipeline_mlx import normalize_mesh
from split3d.hunyuan.xpart_shape_mlx import ShapeVAEDecoderMLX


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode saved X-Part latents without rerunning diffusion")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = trimesh.load(args.mesh, force="mesh")
    _, center, scale = normalize_mesh(source)
    latents = mx.array(np.load(args.latents), dtype=mx.bfloat16)
    model = ShapeVAEDecoderMLX.from_safetensors(args.weights)
    rng = np.random.default_rng(args.seed)
    scene = trimesh.Scene()
    started = time.perf_counter()
    for index in range(latents.shape[0]):
        part = model.decode_to_mesh(
            latents[index : index + 1],
            octree_resolution=args.resolution,
            chunk_size=args.chunk_size,
        )
        part.vertices = np.asarray(part.vertices) * scale + center
        part.visual.face_colors = np.append(rng.integers(32, 256, size=3, dtype=np.uint8), 255)
        scene.add_geometry(part, node_name=f"part_{index:03d}")
    args.output.mkdir(parents=True, exist_ok=True)
    scene.export(args.output / "xpart_scene.glb")
    runtime = {
        "seconds": time.perf_counter() - started,
        "part_count": len(scene.geometry),
        "peak_memory_bytes": mx.get_peak_memory(),
    }
    (args.output / "decode_runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
