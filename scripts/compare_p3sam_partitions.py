from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from split3d.hunyuan.metrics import mask_iou


def _best_mask_iou(source: np.ndarray, target: np.ndarray) -> float:
    source_labels = np.unique(source)
    target_labels = np.unique(target)
    best_ious = [
        max(mask_iou(source == source_label, target == target_label) for target_label in target_labels)
        for source_label in source_labels
    ]
    return float(np.mean(best_ious))


def compare_partitions(cuda_labels: np.ndarray, mlx_labels: np.ndarray) -> dict[str, float]:
    cuda_labels = np.asarray(cuda_labels)
    mlx_labels = np.asarray(mlx_labels)
    if cuda_labels.shape != mlx_labels.shape:
        raise ValueError(f"label shapes differ: {cuda_labels.shape} != {mlx_labels.shape}")
    cuda_to_mlx = _best_mask_iou(cuda_labels, mlx_labels)
    mlx_to_cuda = _best_mask_iou(mlx_labels, cuda_labels)
    return {
        "adjusted_rand_index": float(adjusted_rand_score(cuda_labels, mlx_labels)),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(cuda_labels, mlx_labels)
        ),
        "cuda_to_mlx_best_mask_iou": cuda_to_mlx,
        "mlx_to_cuda_best_mask_iou": mlx_to_cuda,
        "symmetric_best_mask_iou": (cuda_to_mlx + mlx_to_cuda) / 2,
    }


def _label_files(root: Path) -> dict[str, Path]:
    return {path.parent.name: path for path in root.glob("*/face_ids.npy")}


def _runtime_summary(root: Path, uids: list[str]) -> dict[str, float] | None:
    records_path = root / "records.jsonl"
    if not records_path.exists():
        return None
    selected = {
        record["uid"]: record
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for record in [json.loads(line)]
        if record["uid"] in uids
    }
    if selected.keys() != set(uids):
        return None
    records = [selected[uid] for uid in uids]
    return {
        "mean_instance_miou": float(np.mean([record["instance_miou"] for record in records])),
        "mean_inference_seconds": float(
            np.mean([record["inference_seconds"] for record in records])
        ),
        "mean_peak_memory_bytes": float(
            np.mean([record["peak_memory_bytes"] for record in records])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare P3-SAM CUDA and MLX face partitions")
    parser.add_argument("--cuda", type=Path, required=True, help="CUDA benchmark output directory")
    parser.add_argument("--mlx", type=Path, required=True, help="MLX benchmark output directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cuda_files = _label_files(args.cuda)
    mlx_files = _label_files(args.mlx)
    common_uids = sorted(cuda_files.keys() & mlx_files.keys())
    if not common_uids:
        raise FileNotFoundError("no matching CUDA/MLX face_ids.npy files")

    samples: dict[str, dict[str, float]] = {}
    for uid in common_uids:
        samples[uid] = compare_partitions(np.load(cuda_files[uid]), np.load(mlx_files[uid]))

    keys = list(next(iter(samples.values())))
    summary: dict[str, Any] = {
        "matched_samples": len(samples),
        **{key: float(np.mean([sample[key] for sample in samples.values()])) for key in keys},
        "cuda": _runtime_summary(args.cuda, common_uids),
        "mlx": _runtime_summary(args.mlx, common_uids),
        "samples": samples,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
