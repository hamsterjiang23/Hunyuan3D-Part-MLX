from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from split3d.hunyuan.metrics import instance_miou
from split3d.hunyuan.p3sam_official_postprocess import official_mesh_postprocess
from split3d.hunyuan.p3sam_paper_protocol import (
    interactive_prompt_miou,
    sample_part_prompt_indices,
    select_joint_prompt_masks,
)
from split3d.hunyuan.p3sam_pipeline import normalize_point_cloud


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        record["uid"]: record
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for record in [json.loads(line)]
    }


def _write_summary(path: Path, records: dict[str, dict[str, Any]]) -> None:
    values = list(records.values())

    def mean(key: str) -> float:
        return float(np.mean([record[key] for record in values])) if values else 0.0

    summary = {
        "completed": len(values),
        "protocol": "paper-described oracle reconstruction; released evaluation code unavailable",
        "mean_connectivity_instance_miou": mean("connectivity_instance_miou"),
        "mean_full_instance_miou": mean("full_instance_miou"),
        "mean_projected_instance_miou": mean("projected_instance_miou"),
        "mean_point_instance_miou": mean("point_instance_miou"),
        "mean_interactive_prompt_miou": mean("interactive_prompt_miou"),
        "mean_coverage_fraction": mean("coverage_fraction"),
        "mean_overlap_fraction": mean("overlap_fraction"),
        "mean_inference_seconds": mean("inference_seconds"),
        "mean_peak_memory_bytes": mean("peak_memory_bytes"),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    if args.backend == "cuda":
        import torch
        from run_p3sam_cuda_reference import P3SAMCUDA
        from safetensors.torch import load_file

        model = P3SAMCUDA(args.upstream, official_attention_precision=args.official_attention_precision)
        model.load_state_dict(load_file(args.weights), strict=True)
        return model.cuda().eval(), torch

    from split3d.hunyuan.p3sam_mlx import P3SAMMLX

    model = P3SAMMLX.from_safetensors(
        args.weights,
        official_attention_precision=args.official_attention_precision,
    )
    return model, importlib.import_module("mlx.core")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume-safe reconstruction of P3-SAM's GT-prompt paper protocols"
    )
    parser.add_argument("--backend", choices=("cuda", "mlx"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path(".upstream/hunyuan3d-part"))
    parser.add_argument("--points", type=int, default=100_000)
    parser.add_argument("--interactive-prompts-per-part", type=int, default=10)
    parser.add_argument("--prompt-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--postprocess-threshold", type=float, default=0.95)
    parser.add_argument(
        "--official-attention-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--uid", action="append")
    args = parser.parse_args()

    mesh_root = args.dataset / "PartObjaverse-Tiny_mesh"
    target_root = args.dataset / "PartObjaverse-Tiny_instance_gt"
    mesh_paths = [mesh_root / f"{uid}.glb" for uid in args.uid] if args.uid else sorted(mesh_root.glob("*.glb"))
    if args.limit is not None:
        mesh_paths = mesh_paths[: args.limit]
    missing = [path for path in mesh_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing requested meshes: {missing}")

    model, runtime = _load_model(args)
    args.output.mkdir(parents=True, exist_ok=True)
    records_path = args.output / "records.jsonl"
    completed = _load_completed(records_path)
    for mesh_path in mesh_paths:
        uid = mesh_path.stem
        if uid in completed:
            continue
        target = np.load(target_root / f"{uid}.npy")
        mesh = trimesh.load(mesh_path, force="mesh")
        if len(target) != len(mesh.faces):
            raise ValueError(f"{uid}: target has {len(target)} faces, mesh has {len(mesh.faces)}")
        if args.backend == "cuda":
            runtime.cuda.reset_peak_memory_stats()
        elif hasattr(runtime, "reset_peak_memory"):
            runtime.reset_peak_memory()

        started = time.perf_counter()
        sampled_points, face_indices = trimesh.sample.sample_surface(mesh, args.points, seed=args.seed)
        normalized_points = normalize_point_cloud(sampled_points)
        normals = np.asarray(mesh.face_normals)[face_indices]
        part_labels, prompt_indices = sample_part_prompt_indices(
            face_indices,
            target,
            prompts_per_part=args.interactive_prompts_per_part,
            seed=args.seed,
        )
        flattened_prompt_indices = prompt_indices.reshape(-1)

        if args.backend == "cuda":
            with runtime.inference_mode():
                features = model.extract_features(normalized_points, normals, seed=args.seed)
                predictions = model.predict_masks(
                    features,
                    normalized_points,
                    normalized_points[flattened_prompt_indices],
                    prompt_batch_size=args.prompt_batch_size,
                )
            runtime.cuda.synchronize()
            peak_memory = int(runtime.cuda.max_memory_allocated())
        else:
            features = model.extract_features(normalized_points, normals, seed=args.seed)
            predictions = model.predict_masks(
                features,
                normalized_points,
                normalized_points[flattened_prompt_indices],
                prompt_batch_size=args.prompt_batch_size,
            )
            peak_memory = int(runtime.get_peak_memory())

        probabilities = np.stack([np.asarray(mask).T for mask in predictions.masks], axis=-1)
        predicted_iou = np.asarray(predictions.predicted_iou)
        point_labels = target[face_indices]
        prompt_part_labels = np.repeat(part_labels, args.interactive_prompts_per_part)
        interactive_score = interactive_prompt_miou(
            probabilities,
            predicted_iou,
            point_labels,
            prompt_part_labels,
        )

        connectivity_prompt_columns = np.arange(len(part_labels)) * args.interactive_prompts_per_part
        joint = select_joint_prompt_masks(probabilities[:, connectivity_prompt_columns, :])
        topology = official_mesh_postprocess(
            mesh,
            face_indices,
            joint.point_ids,
            threshold=args.postprocess_threshold,
            postprocess=True,
        )
        inference_seconds = time.perf_counter() - started
        sample_output = args.output / uid
        sample_output.mkdir(parents=True, exist_ok=True)
        np.save(sample_output / "selected_heads.npy", joint.selected_heads)
        np.save(sample_output / "face_ids_projected.npy", topology.projected_face_ids)
        np.save(sample_output / "face_ids_connectivity.npy", topology.connectivity_face_ids)
        np.save(sample_output / "face_ids_final.npy", topology.final_face_ids)

        record = {
            "uid": uid,
            "backend": args.backend,
            "protocol": "paper-described oracle reconstruction",
            "ground_truth_parts": int(len(part_labels)),
            "prompt_count": int(len(flattened_prompt_indices)),
            "projected_instance_miou": instance_miou(topology.projected_face_ids, target),
            "connectivity_instance_miou": instance_miou(topology.connectivity_face_ids, target),
            "full_instance_miou": instance_miou(topology.final_face_ids, target),
            "point_instance_miou": instance_miou(joint.point_ids, point_labels),
            "interactive_prompt_miou": interactive_score,
            "coverage_fraction": joint.coverage_fraction,
            "overlap_fraction": joint.overlap_fraction,
            "inference_seconds": inference_seconds,
            "peak_memory_bytes": peak_memory,
        }
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        completed[uid] = record
        _write_summary(args.output / "summary.json", completed)
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
