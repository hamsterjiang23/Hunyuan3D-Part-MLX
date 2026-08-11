from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize P3-SAM using the paper's category-macro protocol")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = load_records(args.records)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    uid_to_category = {
        uid: category
        for category, category_items in metadata.items()
        for uid in category_items
    }
    per_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        category = uid_to_category.get(record["uid"])
        if category is None:
            raise KeyError(f"UID is absent from metadata: {record['uid']}")
        per_category[category].append(record)

    category_summary = {}
    for category in metadata:
        values = per_category.get(category, [])
        category_summary[category] = {
            "completed": len(values),
            "expected": len(metadata[category]),
            "mean_instance_miou": (
                float(np.mean([item["instance_miou"] for item in values])) if values else None
            ),
            "mean_projected_instance_miou": (
                float(np.mean([item["projected_instance_miou"] for item in values])) if values else None
            ),
            "mean_connectivity_instance_miou": (
                float(np.mean([item["connectivity_instance_miou"] for item in values])) if values else None
            ),
            "mean_inference_seconds": (
                float(np.mean([item["inference_seconds"] for item in values])) if values else None
            ),
        }
    complete_categories = [
        value
        for value in category_summary.values()
        if value["completed"] == value["expected"] and value["mean_instance_miou"] is not None
    ]
    summary = {
        "completed": len(records),
        "expected": sum(len(items) for items in metadata.values()),
        "paper_macro_instance_miou": (
            float(np.mean([item["mean_instance_miou"] for item in complete_categories]))
            if len(complete_categories) == len(metadata)
            else None
        ),
        "paper_macro_projected_instance_miou": (
            float(np.mean([item["mean_projected_instance_miou"] for item in complete_categories]))
            if len(complete_categories) == len(metadata)
            else None
        ),
        "paper_macro_connectivity_instance_miou": (
            float(np.mean([item["mean_connectivity_instance_miou"] for item in complete_categories]))
            if len(complete_categories) == len(metadata)
            else None
        ),
        "shape_micro_instance_miou": (
            float(np.mean([item["instance_miou"] for item in records])) if records else None
        ),
        "mean_inference_seconds": (
            float(np.mean([item["inference_seconds"] for item in records])) if records else None
        ),
        "mean_peak_memory_bytes": (
            float(np.mean([item["peak_memory_bytes"] for item in records])) if records else None
        ),
        "categories": category_summary,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
