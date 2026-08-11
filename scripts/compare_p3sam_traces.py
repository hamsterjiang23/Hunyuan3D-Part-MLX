from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from split3d.hunyuan.metrics import mask_iou


def _numeric_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    return {
        "shape_equal": True,
        "exact": bool(np.array_equal(reference, candidate, equal_nan=True)),
        "max_abs": float(np.max(np.abs(difference), initial=0)),
        "mean_abs": float(np.mean(np.abs(difference))) if difference.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(difference)))) if difference.size else 0.0,
    }


def _partition_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    reference_labels = np.unique(reference)
    candidate_labels = np.unique(candidate)
    reference_to_candidate = np.mean(
        [
            max(mask_iou(reference == label, candidate == target) for target in candidate_labels)
            for label in reference_labels
        ]
    )
    candidate_to_reference = np.mean(
        [
            max(mask_iou(candidate == label, reference == target) for target in reference_labels)
            for label in candidate_labels
        ]
    )
    return {
        "shape_equal": True,
        "exact_labels": bool(np.array_equal(reference, candidate)),
        "adjusted_rand_index": float(adjusted_rand_score(reference, candidate)),
        "normalized_mutual_information": float(normalized_mutual_info_score(reference, candidate)),
        "symmetric_best_mask_iou": float((reference_to_candidate + candidate_to_reference) / 2),
    }


def _load_candidate_masks(root: Path) -> np.ndarray:
    metadata = json.loads((root / "candidate_masks.json").read_text(encoding="utf-8"))
    packed = np.load(root / "candidate_masks.packbits.npy")
    unpacked = np.unpackbits(packed, axis=0, bitorder=metadata["bitorder"])
    return unpacked[: metadata["shape"][0], : metadata["shape"][1]].astype(bool)


def compare_traces(reference: Path, candidate: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"reference": str(reference), "candidate": str(candidate)}
    with np.load(reference / "replay_manifest.npz", allow_pickle=False) as first, np.load(
        candidate / "replay_manifest.npz", allow_pickle=False
    ) as second:
        replay: dict[str, Any] = {}
        for key in first.files:
            if key not in second.files:
                replay[key] = {"missing_in_candidate": True}
            elif first[key].dtype.kind in "SUO":
                replay[key] = {"exact": bool(np.array_equal(first[key], second[key]))}
            else:
                replay[key] = _numeric_error(first[key], second[key])
        result["replay_manifest"] = replay

    for name in ("features", "predicted_iou", "candidate_iou", "mask_probabilities"):
        first_path = reference / f"{name}.npy"
        second_path = candidate / f"{name}.npy"
        if first_path.exists() and second_path.exists():
            result[name] = _numeric_error(np.load(first_path), np.load(second_path))
        else:
            result[name] = {"available": False}

    first_masks = _load_candidate_masks(reference)
    second_masks = _load_candidate_masks(candidate)
    result["candidate_masks"] = {
        **_numeric_error(first_masks, second_masks),
        "different_values": int(np.count_nonzero(first_masks != second_masks))
        if first_masks.shape == second_masks.shape
        else None,
    }
    for name in ("point_ids", "face_ids_projected", "face_ids_connectivity", "face_ids_final"):
        result[name] = _partition_metrics(
            np.load(reference / f"{name}.npy"),
            np.load(candidate / f"{name}.npy"),
        )
    result["replay_exact"] = all(
        value.get("exact", False) for value in result["replay_manifest"].values()
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare replay-synchronized CUDA and MLX P3-SAM stage traces")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-replay-exact", action="store_true")
    args = parser.parse_args()
    result = compare_traces(args.reference, args.candidate)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if args.require_replay_exact and not result["replay_exact"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
