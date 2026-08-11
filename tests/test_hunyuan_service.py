from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from split3d.hunyuan.service import PartJobService, PartTaskRequest, create_app


class FakeRunner:
    def __init__(self) -> None:
        self.unloaded = False

    @property
    def model_state(self) -> dict[str, bool]:
        return {"p3sam_loaded": True, "xpart_loaded": False}

    def run(
        self,
        request: PartTaskRequest,
        mesh_path: Path,
        output_dir: Path,
        progress: Any,
    ) -> Path:
        assert mesh_path.read_bytes() == b"fake-glb"
        progress("predict_masks", 1.25)
        output_dir.mkdir(parents=True, exist_ok=True)
        primary = output_dir / "segmented.glb"
        primary.write_bytes(b"result-glb")
        np.save(output_dir / "face_ids.npy", np.asarray([0, 1]))
        np.save(output_dir / "bboxes.npy", np.zeros((2, 2, 3)))
        parts_dir = output_dir / "parts"
        parts_dir.mkdir()
        (parts_dir / "part_000.glb").write_bytes(b"part-glb")
        (output_dir / "runtime.json").write_text("{}", encoding="utf-8")
        return primary

    def unload(self) -> bool:
        self.unloaded = True
        return True


def _wait_for_completion(client: TestClient, uid: str) -> dict[str, Any]:
    for _ in range(100):
        response = client.get(f"/status/{uid}")
        payload = response.json()
        if payload["status"] in {"completed", "error"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("task did not finish")


def test_part_worker_submit_status_download_and_bundle(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    mesh = input_root / "mesh.glb"
    mesh.write_bytes(b"fake-glb")
    runner = FakeRunner()
    service = PartJobService(tmp_path / "cache", input_root, runner)
    client = TestClient(create_app(service))
    try:
        submitted = client.post("/send", json={"mesh_path": str(mesh), "mode": "segment"})
        assert submitted.status_code == 200
        uid = submitted.json()["uid"]
        status = _wait_for_completion(client, uid)
        assert status["status"] == "completed"
        assert status["stage"] == "completed"
        assert client.get(f"/download/{uid}").content == b"result-glb"
        bundle = client.get(f"/bundle/{uid}")
        assert bundle.status_code == 200
        assert bundle.headers["content-type"] == "application/zip"
        with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
            assert archive.namelist() == [
                "bboxes.npy",
                "face_ids.npy",
                "parts/part_000.glb",
                "runtime.json",
                "segmented.glb",
            ]
            assert archive.read("parts/part_000.glb") == b"part-glb"
        assert client.get("/health").json()["p3sam_loaded"] is True
        assert client.post("/unload").json() == {"status": "unloaded", "was_loaded": True}
        assert runner.unloaded is True
    finally:
        service.shutdown()


def test_part_worker_rejects_paths_outside_input_root(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    outside = tmp_path / "outside.glb"
    outside.write_bytes(b"fake-glb")
    service = PartJobService(tmp_path / "cache", input_root, FakeRunner())
    client = TestClient(create_app(service))
    try:
        response = client.post("/send", json={"mesh_path": str(outside)})
        assert response.status_code == 400
        assert "input root" in response.json()["detail"]
    finally:
        service.shutdown()
