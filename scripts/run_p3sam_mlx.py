from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import trimesh

from split3d.hunyuan.p3sam_mlx import P3SAMMLX
from split3d.hunyuan.p3sam_pipeline import save_segmentation, segment_mesh_mlx


def main() -> None:
    parser = argparse.ArgumentParser(description="Run native MLX P3-SAM automatic mesh segmentation")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=100_000)
    parser.add_argument("--prompts", type=int, default=400)
    parser.add_argument("--prompt-batch-size", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--official-fps-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-mesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--connectivity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postprocess-threshold", type=float, default=0.95)
    parser.add_argument("--replay-manifest", type=Path)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--trace-full-tensors", action="store_true")
    parser.add_argument(
        "--official-attention-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Experiment with FP16 QKV quantization and stable FP32 SDPA instead of the higher-scoring FP32 path",
    )
    args = parser.parse_args()

    mx.reset_peak_memory()
    started = time.perf_counter()
    model = P3SAMMLX.from_safetensors(
        args.weights,
        official_attention_precision=args.official_attention_precision,
    )
    load_seconds = time.perf_counter() - started
    mesh = trimesh.load(args.mesh, force="mesh")
    inference_started = time.perf_counter()
    result = segment_mesh_mlx(
        model,
        mesh,
        point_count=args.points,
        prompt_count=args.prompts,
        prompt_batch_size=args.prompt_batch_size,
        seed=args.seed,
        prompt_start_index=None if args.official_fps_start else 0,
        clean_mesh=args.clean_mesh,
        connectivity=args.connectivity,
        postprocess=args.postprocess,
        postprocess_threshold=args.postprocess_threshold,
        replay_manifest=args.replay_manifest,
        trace_dir=args.trace_dir,
        trace_full_tensors=args.trace_full_tensors,
        progress=lambda stage, seconds: print(f"{stage}: {seconds:.3f}s", flush=True),
    )
    inference_seconds = time.perf_counter() - inference_started
    save_segmentation(mesh, result, args.output, seed=args.seed)
    runtime = {
        "backend": "mlx",
        "device": str(mx.default_device()),
        "weights": str(args.weights),
        "mesh": str(args.mesh),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_memory_bytes": mx.get_peak_memory(),
        "stage_seconds": result.stage_seconds,
        "diagnostics": asdict(result.diagnostics),
    }
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
