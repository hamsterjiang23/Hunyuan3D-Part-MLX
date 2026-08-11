from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from split3d.pipeline import inspect_asset, split_asset


def test_split_asset_exports_manifest_and_glbs(tmp_path: Path) -> None:
    source = tmp_path / "box.glb"
    mesh = trimesh.creation.box()
    mesh.export(source)
    labels = np.where(mesh.triangles_center[:, 0] < 0, 0, 1).astype(np.int32)
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, labels)

    output = tmp_path / "result"
    manifest = split_asset(
        source,
        output,
        part_names=["left", "right"],
        face_labels_path=labels_path,
        individual=True,
    )

    assert manifest.source_faces == len(mesh.faces)
    assert sum(part.face_count for part in manifest.parts) == len(mesh.faces)
    assert (output / "split.glb").exists()
    assert (output / "parts.json").exists()
    assert all(part.file and (output / part.file).exists() for part in manifest.parts)
    assert all((output / part.render).exists() for part in manifest.parts)
    assert not (output / "preview.png").exists()
    for part in manifest.parts:
        image = np.asarray(Image.open(output / part.render).convert("RGB"))
        assert image.shape == (768, 768, 3)
        assert np.any(image != 245)
    saved = json.loads((output / "parts.json").read_text(encoding="utf-8"))
    assert saved["assignment"] == "provided_face_labels"


def test_inspect_asset(tmp_path: Path) -> None:
    source = tmp_path / "box.obj"
    trimesh.creation.box().export(source)
    report = inspect_asset(source)
    assert report["faces"] == 12
    assert report["finite_vertices"] is True
