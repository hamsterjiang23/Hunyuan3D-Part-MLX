from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import trimesh

from split3d.hunyuan.xpart_pipeline_mlx import XPartPipelineMLX


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete native-MLX Hunyuan3D-Part X-Part pipeline")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--p3-weights", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=100_000)
    parser.add_argument("--prompts", type=int, default=400)
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--surface-points", type=int, default=81_920)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--sdf-chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--official-fps-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-mesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--connectivity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postprocess-threshold", type=float, default=0.95)
    parser.add_argument("--latents-only", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    pipeline = XPartPipelineMLX.from_pretrained(args.model_dir, p3_weights=args.p3_weights)
    load_seconds = time.perf_counter() - started
    mesh = trimesh.load(args.mesh, force="mesh")
    result = pipeline(
        mesh,
        point_count=args.points,
        prompt_count=args.prompts,
        prompt_batch_size=args.prompt_batch_size,
        surface_point_count=args.surface_points,
        num_inference_steps=args.steps,
        octree_resolution=args.resolution,
        sdf_chunk_size=args.sdf_chunk_size,
        seed=args.seed,
        official_fps_start=args.official_fps_start,
        clean_mesh=args.clean_mesh,
        connectivity=args.connectivity,
        postprocess=args.postprocess,
        postprocess_threshold=args.postprocess_threshold,
        output_latents=args.latents_only,
        progress=lambda stage, seconds: print(f"{stage}: {seconds:.3f}s", flush=True),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.latents_only:
        np.save(args.output / "latents.npy", result.scene)
    else:
        result.scene.export(args.output / "xpart_scene.glb")
        np.save(args.output / "latents.npy", result.latents)
    np.save(args.output / "bboxes.npy", result.bboxes)
    runtime = {
        "backend": "mlx",
        "load_seconds": load_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_memory_bytes": mx.get_peak_memory(),
        "part_count": int(len(result.bboxes)),
        "stage_seconds": result.stage_seconds,
    }
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
