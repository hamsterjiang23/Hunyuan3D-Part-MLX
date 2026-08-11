from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from split3d.hunyuan.p3sam_pipeline import normalize_point_cloud


def main() -> None:
    parser = argparse.ArgumentParser(description="Export P3-SAM Sonata features for cross-backend comparison")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--backend", choices=("cuda", "mlx"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path(".upstream/hunyuan3d-part"))
    parser.add_argument("--points", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--official-attention-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the released FP16 attention contract on CUDA or its stable quantization approximation on MLX",
    )
    args = parser.parse_args()

    model: Any
    runtime: Any
    if args.backend == "cuda":
        import torch as torch_runtime
        from run_p3sam_cuda_reference import P3SAMCUDA
        from safetensors.torch import load_file

        model = P3SAMCUDA(args.upstream, official_attention_precision=args.official_attention_precision)
        model.load_state_dict(load_file(args.weights), strict=True)
        model.cuda().eval()
        runtime = torch_runtime
    else:
        import mlx.core as mlx_runtime

        from split3d.hunyuan.p3sam_mlx import P3SAMMLX

        model = P3SAMMLX.from_safetensors(
            args.weights,
            official_attention_precision=args.official_attention_precision,
        )
        runtime = mlx_runtime

    mesh = trimesh.load(args.mesh, force="mesh")
    points, face_indices = trimesh.sample.sample_surface(mesh, args.points, seed=args.seed)
    normalized = normalize_point_cloud(points)
    normals = np.asarray(mesh.face_normals)[face_indices]
    started = time.perf_counter()
    if args.backend == "cuda":
        with runtime.inference_mode():
            features = model.extract_features(normalized, normals, seed=args.seed)
        runtime.cuda.synchronize()
        output = features.detach().cpu().numpy()
    else:
        output = np.array(model.extract_features(normalized, normals, seed=args.seed))
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "features.npy", output)
    np.savez_compressed(args.output / "input.npz", points=normalized, normals=normals)
    (args.output / "runtime.json").write_text(
        json.dumps({"backend": args.backend, "seconds": elapsed, "shape": list(output.shape)}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"backend": args.backend, "seconds": elapsed, "shape": list(output.shape)}))


if __name__ == "__main__":
    main()
