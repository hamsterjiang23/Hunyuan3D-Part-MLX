from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from split3d.hunyuan.metrics import instance_miou
from split3d.hunyuan.p3sam_pipeline import save_segmentation, segment_mesh


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["uid"]] = record
    return records


def _write_summary(path: Path, records: dict[str, dict[str, Any]]) -> None:
    values = list(records.values())
    summary = {
        "completed": len(values),
        "mean_instance_miou": float(np.mean([item["instance_miou"] for item in values])) if values else 0.0,
        "mean_inference_seconds": (
            float(np.mean([item["inference_seconds"] for item in values])) if values else 0.0
        ),
        "mean_peak_memory_bytes": (
            float(np.mean([item["peak_memory_bytes"] for item in values])) if values else 0.0
        ),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-safe P3-SAM benchmark on PartObjaverse-Tiny")
    parser.add_argument("--backend", choices=("cuda", "mlx"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path(".upstream/hunyuan3d-part"))
    parser.add_argument("--points", type=int, default=100_000)
    parser.add_argument("--prompts", type=int, default=400)
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--official-fps-start",
        action="store_true",
        help="Match the official demo's seeded random initial FPS point instead of deterministic index 0",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--uid", action="append", help="Evaluate only this UID; repeat for multiple shapes")
    parser.add_argument(
        "--exclude-uid",
        action="append",
        default=[],
        help="Skip this UID; repeat for multiple shapes",
    )
    parser.add_argument("--save-visuals", action="store_true")
    args = parser.parse_args()

    mesh_root = args.dataset / "PartObjaverse-Tiny_mesh"
    ground_truth_root = args.dataset / "PartObjaverse-Tiny_instance_gt"
    mesh_paths = (
        [mesh_root / f"{uid}.glb" for uid in args.uid]
        if args.uid
        else sorted(mesh_root.glob("*.glb"))
    )
    missing_meshes = [path for path in mesh_paths if not path.exists()]
    if missing_meshes:
        raise FileNotFoundError(f"missing requested meshes: {missing_meshes}")
    excluded_uids = set(args.exclude_uid)
    mesh_paths = [path for path in mesh_paths if path.stem not in excluded_uids]
    if args.limit is not None:
        mesh_paths = mesh_paths[: args.limit]

    runtime: Any
    model: Any
    if args.backend == "cuda":
        import torch as torch_runtime
        from run_p3sam_cuda_reference import P3SAMCUDA
        from safetensors.torch import load_file

        model = P3SAMCUDA(args.upstream)
        model.load_state_dict(load_file(args.weights), strict=True)
        model.cuda().eval()
        runtime = torch_runtime
    else:
        from split3d.hunyuan.p3sam_mlx import P3SAMMLX

        model = P3SAMMLX.from_safetensors(args.weights)
        runtime = importlib.import_module("mlx.core")

    args.output.mkdir(parents=True, exist_ok=True)
    records_path = args.output / "records.jsonl"
    completed = _load_completed(records_path)
    for mesh_path in mesh_paths:
        uid = mesh_path.stem
        if uid in completed:
            continue
        target_path = ground_truth_root / f"{uid}.npy"
        if not target_path.exists():
            raise FileNotFoundError(f"missing instance ground truth: {target_path}")
        if args.backend == "cuda":
            runtime.cuda.reset_peak_memory_stats()
        elif hasattr(runtime, "reset_peak_memory"):
            runtime.reset_peak_memory()

        mesh = trimesh.load(mesh_path, force="mesh")
        started = time.perf_counter()
        if args.backend == "cuda":
            with runtime.inference_mode():
                result = segment_mesh(
                    model,
                    mesh,
                    point_count=args.points,
                    prompt_count=args.prompts,
                    prompt_batch_size=args.prompt_batch_size,
                    seed=args.seed,
                    prompt_start_index=None if args.official_fps_start else 0,
                )
            runtime.cuda.synchronize()
            peak_memory = int(runtime.cuda.max_memory_allocated())
        else:
            result = segment_mesh(
                model,
                mesh,
                point_count=args.points,
                prompt_count=args.prompts,
                prompt_batch_size=args.prompt_batch_size,
                seed=args.seed,
                prompt_start_index=None if args.official_fps_start else 0,
            )
            peak_memory = int(runtime.get_peak_memory())
        inference_seconds = time.perf_counter() - started

        target = np.load(target_path)
        score = instance_miou(result.face_ids, target)
        sample_output = args.output / uid
        sample_output.mkdir(parents=True, exist_ok=True)
        np.save(sample_output / "face_ids.npy", result.face_ids)
        if args.save_visuals:
            save_segmentation(mesh, result, sample_output, seed=args.seed)
        record = {
            "uid": uid,
            "backend": args.backend,
            "instance_miou": score,
            "inference_seconds": inference_seconds,
            "peak_memory_bytes": peak_memory,
            "predicted_parts": result.diagnostics.final_part_count,
            "ground_truth_parts": int(len(np.unique(target))),
            "stage_seconds": result.stage_seconds,
            "diagnostics": asdict(result.diagnostics),
        }
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        completed[uid] = record
        _write_summary(args.output / "summary.json", completed)
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
