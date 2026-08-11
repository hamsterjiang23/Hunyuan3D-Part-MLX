from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_module():
    script = Path(__file__).parents[1] / "scripts/compare_p3sam_traces.py"
    spec = importlib.util.spec_from_file_location("compare_p3sam_traces", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_trace(path: Path, *, label_permutation: bool = False) -> None:
    path.mkdir()
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    np.savez_compressed(
        path / "replay_manifest.npz",
        mesh_geometry_hash=np.asarray("abc"),
        seed=np.asarray(42),
        sampled_points=points,
        normalized_points=points,
        normals=np.ones_like(points),
        face_indices=np.asarray([0, 1, 1]),
        prompt_indices=np.asarray([0, 1]),
    )
    masks = np.asarray([[True, False], [True, False], [False, True]])
    np.save(path / "candidate_masks.packbits.npy", np.packbits(masks, axis=0, bitorder="little"))
    (path / "candidate_masks.json").write_text(
        json.dumps({"shape": list(masks.shape), "bitorder": "little"}), encoding="utf-8"
    )
    np.save(path / "predicted_iou.npy", np.asarray([[0.9, 0.1], [0.8, 0.2]]))
    np.save(path / "candidate_iou.npy", np.asarray([0.9, 0.8]))
    labels = np.asarray([4, 4, 8]) if label_permutation else np.asarray([0, 0, 1])
    for name in ("point_ids", "face_ids_projected", "face_ids_connectivity", "face_ids_final"):
        np.save(path / f"{name}.npy", labels)


def test_compare_traces_requires_exact_inputs_but_accepts_label_permutations(tmp_path: Path) -> None:
    module = _load_module()
    reference, candidate = tmp_path / "cuda", tmp_path / "mlx"
    _write_trace(reference)
    _write_trace(candidate, label_permutation=True)
    result = module.compare_traces(reference, candidate)
    assert result["replay_exact"] is True
    assert result["candidate_masks"]["different_values"] == 0
    assert result["face_ids_final"]["exact_labels"] is False
    assert result["face_ids_final"]["adjusted_rand_index"] == 1.0

