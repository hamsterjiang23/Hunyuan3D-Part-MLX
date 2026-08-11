from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import trimesh

from split3d.hunyuan.metrics import instance_miou


def _load_auditor() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "audit_p3sam_benchmark.py"
    spec = importlib.util.spec_from_file_location("audit_p3sam_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset = tmp_path / "dataset"
    mesh_root = dataset / "PartObjaverse-Tiny_mesh"
    target_root = dataset / "PartObjaverse-Tiny_instance_gt"
    benchmark = tmp_path / "benchmark"
    mesh_root.mkdir(parents=True)
    target_root.mkdir(parents=True)
    benchmark.mkdir()
    metadata = {"Chair": {"shape-a": ["seat"]}, "Table": {"shape-b": ["top"]}}
    metadata_path = dataset / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    records = []
    for uid in ("shape-a", "shape-b"):
        mesh = trimesh.Trimesh(
            vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
            faces=np.asarray([[0, 1, 2]], dtype=np.int64),
            process=False,
        )
        mesh.export(mesh_root / f"{uid}.glb")
        target = np.asarray([0], dtype=np.int64)
        np.save(target_root / f"{uid}.npy", target)
        sample = benchmark / uid
        sample.mkdir()
        labels = {
            "face_ids_projected.npy": np.asarray([-1], dtype=np.int64),
            "face_ids_connectivity.npy": np.asarray([0], dtype=np.int64),
            "face_ids.npy": np.asarray([0], dtype=np.int64),
        }
        for filename, values in labels.items():
            np.save(sample / filename, values)
        records.append(
            {
                "uid": uid,
                "backend": "mlx",
                "projected_instance_miou": instance_miou(labels["face_ids_projected.npy"], target),
                "connectivity_instance_miou": instance_miou(labels["face_ids_connectivity.npy"], target),
                "instance_miou": instance_miou(labels["face_ids.npy"], target),
            }
        )
    (benchmark / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return benchmark, dataset, metadata_path


def test_audit_benchmark_checks_all_stage_files(tmp_path: Path) -> None:
    auditor = _load_auditor()
    benchmark, dataset, metadata = _fixture(tmp_path)

    result = auditor.audit_benchmark(benchmark, dataset, metadata, expected_backend="mlx")

    assert result["status"] == "passed"
    assert result["record_count"] == 2
    assert result["stage_file_count"] == 6
    assert result["maximum_metric_delta"] == 0


def test_audit_benchmark_rejects_duplicate_records(tmp_path: Path) -> None:
    auditor = _load_auditor()
    benchmark, dataset, metadata = _fixture(tmp_path)
    first = (benchmark / "records.jsonl").read_text(encoding="utf-8").splitlines()[0]
    with (benchmark / "records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(first + "\n")

    with pytest.raises(ValueError, match="duplicate UIDs"):
        auditor.audit_benchmark(benchmark, dataset, metadata, expected_backend="mlx")
