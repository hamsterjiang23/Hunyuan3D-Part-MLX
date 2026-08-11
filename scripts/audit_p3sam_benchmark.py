from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from split3d.hunyuan.metrics import instance_miou

STAGE_FILES = {
    "projected_instance_miou": "face_ids_projected.npy",
    "connectivity_instance_miou": "face_ids_connectivity.npy",
    "instance_miou": "face_ids.npy",
}


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_benchmark(
    benchmark: Path,
    dataset: Path,
    metadata_path: Path,
    *,
    expected_backend: str | None = None,
) -> dict[str, Any]:
    records_path = benchmark / "records.jsonl"
    records = _load_records(records_path)
    record_uids = [str(record["uid"]) for record in records]
    if len(record_uids) != len(set(record_uids)):
        duplicates = sorted({uid for uid in record_uids if record_uids.count(uid) > 1})
        raise ValueError(f"records.jsonl contains duplicate UIDs: {duplicates}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_uids = {
        uid
        for category_items in metadata.values()
        for uid in category_items
    }
    actual_uids = set(record_uids)
    missing = sorted(expected_uids - actual_uids)
    unexpected = sorted(actual_uids - expected_uids)
    if missing or unexpected:
        raise ValueError(f"benchmark UID set differs from metadata; missing={missing}, unexpected={unexpected}")

    mesh_root = dataset / "PartObjaverse-Tiny_mesh"
    target_root = dataset / "PartObjaverse-Tiny_instance_gt"
    maximum_metric_delta = 0.0
    stage_file_count = 0
    for record in records:
        uid = str(record["uid"])
        if expected_backend is not None and record.get("backend") != expected_backend:
            raise ValueError(f"{uid}: backend is {record.get('backend')!r}, expected {expected_backend!r}")
        mesh = trimesh.load(mesh_root / f"{uid}.glb", force="mesh")
        target = np.load(target_root / f"{uid}.npy", allow_pickle=False)
        expected_shape = (len(mesh.faces),)
        if target.shape != expected_shape:
            raise ValueError(f"{uid}: target shape {target.shape} != mesh face shape {expected_shape}")
        for metric_name, filename in STAGE_FILES.items():
            labels_path = benchmark / uid / filename
            if not labels_path.exists():
                raise FileNotFoundError(f"{uid}: missing {filename}")
            labels = np.load(labels_path, allow_pickle=False)
            if labels.shape != expected_shape:
                raise ValueError(f"{uid}: {filename} shape {labels.shape} != {expected_shape}")
            if not np.issubdtype(labels.dtype, np.integer):
                raise TypeError(f"{uid}: {filename} dtype {labels.dtype} is not integer")
            recomputed = instance_miou(labels, target)
            delta = abs(recomputed - float(record[metric_name]))
            maximum_metric_delta = max(maximum_metric_delta, delta)
            if delta > 1e-10:
                raise ValueError(
                    f"{uid}: {metric_name} differs from saved labels by {delta}; "
                    f"record={record[metric_name]}, recomputed={recomputed}"
                )
            stage_file_count += 1

    category_counts = {
        category: sum(uid in actual_uids for uid in category_items)
        for category, category_items in metadata.items()
    }
    return {
        "status": "passed",
        "record_count": len(records),
        "unique_uid_count": len(actual_uids),
        "expected_uid_count": len(expected_uids),
        "stage_file_count": stage_file_count,
        "expected_stage_file_count": len(expected_uids) * len(STAGE_FILES),
        "maximum_metric_delta": maximum_metric_delta,
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "category_counts": category_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a completed P3-SAM paper-protocol benchmark")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--backend", choices=("cuda", "mlx"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit_benchmark(
        args.benchmark,
        args.dataset,
        args.metadata,
        expected_backend=args.backend,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
