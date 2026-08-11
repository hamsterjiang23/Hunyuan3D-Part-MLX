from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from split3d.render import render_views


def test_render_views_write_rgb_face_ids_and_manifest(tmp_path: Path) -> None:
    mesh = trimesh.creation.box()
    records = render_views(mesh, tmp_path, view_count=4, resolution=64)
    assert len(records) == 4
    for record in records:
        assert (tmp_path / record.rgb).exists()
        face_ids = np.load(tmp_path / record.face_ids, allow_pickle=False)
        assert face_ids.shape == (64, 64)
        assert np.any(face_ids >= 0)
        assert face_ids.max() < len(mesh.faces)
    manifest = json.loads((tmp_path / "views.json").read_text(encoding="utf-8"))
    assert manifest["view_count"] == 4
    assert manifest["resolution"] == 64
