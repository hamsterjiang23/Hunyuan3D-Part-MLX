from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from split3d.hunyuan.metrics import instance_miou


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute saved P3-SAM stage metrics using the pinned PartField label semantics"
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    records_path = args.benchmark / "records.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    uids = [record["uid"] for record in records]
    if len(uids) != len(set(uids)):
        raise ValueError("records.jsonl contains duplicate UIDs")

    stage_files = {
        "projected_instance_miou": "face_ids_projected.npy",
        "connectivity_instance_miou": "face_ids_connectivity.npy",
        "instance_miou": "face_ids.npy",
    }
    changed = {key: 0 for key in stage_files}
    maximum_delta = {key: 0.0 for key in stage_files}
    target_root = args.dataset / "PartObjaverse-Tiny_instance_gt"
    for record in records:
        uid = record["uid"]
        target = np.load(target_root / f"{uid}.npy")
        result_root = args.benchmark / uid
        for key, filename in stage_files.items():
            labels = np.load(result_root / filename)
            value = instance_miou(labels, target)
            delta = abs(value - float(record[key]))
            if delta > 1e-12:
                changed[key] += 1
                maximum_delta[key] = max(maximum_delta[key], delta)
            record[key] = value

    backup = args.benchmark / "records.before_official_metric_fix.jsonl"
    if not backup.exists():
        shutil.copy2(records_path, backup)
    temporary = records_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    os.replace(temporary, records_path)
    print(
        json.dumps(
            {
                "completed": len(records),
                "backup": str(backup),
                "changed_records": changed,
                "maximum_absolute_delta": maximum_delta,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
