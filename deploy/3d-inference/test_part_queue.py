"""Protocol tests for the deployable Hunyuan3D-Part queue integration."""

import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import queue_server


class FakeResponse:
    status = 200

    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def make_manager(tmp_path: Path, *, max_mesh_bytes: int = 1024):
    return queue_server.QueueManager(
        str(tmp_path / "cache"),
        backend_urls={"hunyuan3d_part": "http://127.0.0.1:8083"},
        max_mesh_bytes=max_mesh_bytes,
        start_worker=False,
    )


def part_params(task_id=None, mode="segment"):
    return {
        "task_id": task_id,
        "filename": "chair.glb",
        "mode": mode,
        "points": 100_000,
        "prompts": 400,
        "prompt_batch_size": 8,
        "surface_points": 81_920,
        "steps": 50,
        "resolution": 128,
        "sdf_chunk_size": 100_000,
        "seed": 42,
        "official_fps_start": True,
        "clean_mesh": True,
        "connectivity": True,
        "postprocess": True,
        "postprocess_threshold": 0.95,
    }


def test_submit_part_is_durable_and_idempotent(tmp_path: Path):
    manager = make_manager(tmp_path)
    task_id = uuid4()
    uid, status, _ = manager.submit_part(b"binary-glb", part_params(task_id))
    assert (uid, status) == (str(task_id), "queued")
    task = manager.get_task(uid)
    assert task["model"] == "hunyuan3d_part"
    assert Path(task["input_path"]).suffix == ".glb"
    assert Path(task["input_path"]).read_bytes() == b"binary-glb"

    repeated = manager.submit_part(b"binary-glb", part_params(task_id))
    assert repeated[:2] == (uid, "queued")

    with pytest.raises(HTTPException) as conflict:
        manager.submit_part(b"binary-glb", part_params(task_id, mode="generate_parts"))
    assert conflict.value.status_code == 409


def test_forward_part_uses_shared_input_path(tmp_path: Path):
    manager = make_manager(tmp_path)
    uid, _, _ = manager.submit_part(b"binary-glb", part_params())
    task = manager.get_task(uid)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        return FakeResponse({"uid": uid})

    with patch("queue_server.urllib.request.urlopen", side_effect=fake_urlopen):
        assert manager._forward_to_backend(task) == uid

    assert captured["url"] == "http://127.0.0.1:8083/send"
    assert captured["body"]["mesh_path"] == task["input_path"]
    assert captured["body"]["mode"] == "segment"
    assert "image" not in captured["body"]


def test_submit_part_http_endpoint_and_size_limit(tmp_path: Path):
    manager = make_manager(tmp_path, max_mesh_bytes=10)
    previous = queue_server._manager
    queue_server._manager = manager
    client = TestClient(queue_server.app)
    try:
        response = client.post("/submit/part?filename=chair.glb", content=b"mesh")
        assert response.status_code == 200
        assert response.json()["model"] == "hunyuan3d_part"

        too_large = client.post("/submit/part?filename=chair.glb", content=b"01234567890")
        assert too_large.status_code == 413
    finally:
        queue_server._manager = previous
