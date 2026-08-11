"""Durable queue proxy — single, persistent, idempotent entry point for
TRELLIS (8081) and Hunyuan3D (8082) inference backends.

Clients generate a task_id UUID. Same task_id + same image + same params
is idempotent (returns existing task). Same task_id + different params →
409 Conflict.
"""

from __future__ import annotations

# Deployment snapshot: install beside task_store.py in the 3d-inference gateway.

import argparse
import base64
import binascii
import hashlib
import json
import logging
import os
import queue
import shutil
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Literal, Optional
from uuid import UUID

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from task_store import TaskStore, compute_fingerprint


logger = logging.getLogger("queue-server")

BACKENDS = {
    "trellis": "http://127.0.0.1:8081",
    "hunyuan3d": "http://127.0.0.1:8082",
    "hunyuan3d_part": "http://127.0.0.1:8083",
}
DEFAULT_MAX_IMAGE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_MESH_BYTES = 100 * 1024 * 1024
MAX_BASE64_CHARS = ((DEFAULT_MAX_IMAGE_BYTES + 2) // 3) * 4

# ── Quality presets ──────────────────────────────────────────────────────────
TRELLIS_QUALITY_PRESETS = {
    "low": {
        "pipeline_type": "512",
        "texture_size": 512,
        "target_faces": 50_000,
        "steps": 8,
        "no_texture": False,
    },
    "medium": {
        "pipeline_type": "1024",
        "texture_size": 1024,
        "target_faces": 200_000,
        "steps": 12,
        "no_texture": False,
    },
    "high": {
        "pipeline_type": "1024_cascade",
        "texture_size": 2048,
        "target_faces": 500_000,
        "steps": 16,
        "no_texture": False,
    },
    "ultra": {
        "pipeline_type": "1024_cascade",
        "texture_size": 2048,
        "target_faces": 2_000_000,
        "steps": 20,
        "no_texture": False,
    },
}

HUNYUAN_QUALITY_PRESETS = {
    "low": {
        "octree_resolution": 128,
        "face_count": 10_000,
        "texture": False,
        "num_inference_steps": 5,
        "guidance_scale": 5.0,
    },
    "medium": {
        "octree_resolution": 256,
        "face_count": 50_000,
        "texture": True,
        "num_inference_steps": 10,
        "guidance_scale": 7.5,
    },
    "high": {
        "octree_resolution": 384,
        "face_count": 100_000,
        "texture": True,
        "num_inference_steps": 15,
        "guidance_scale": 5.0,
    },
    "ultra": {
        "octree_resolution": 512,
        "face_count": 100_000,
        "texture": True,
        "num_inference_steps": 20,
        "guidance_scale": 5.0,
    },
}

QUALITY_LEVELS = Literal["low", "medium", "high", "ultra"]
MODEL_NAME = Literal["trellis", "hunyuan3d", "hunyuan3d_part"]


class SubmitRequest(BaseModel):
    task_id: Optional[UUID] = Field(
        None,
        description="Client-generated idempotency key. Same task_id + same params → "
        "returns existing task. Same task_id + different params → 409 Conflict.",
    )
    model: Literal["trellis", "hunyuan3d"]
    image: str = Field(..., min_length=4, max_length=MAX_BASE64_CHARS)
    quality: QUALITY_LEVELS = Field(
        "ultra",
        description="Quality preset. Maps all backend params.\n"
        "  trellis: low=512/512/50K, medium=1024/1024/200K, "
        "high=1024_cascade/2048/500K, ultra=1024_cascade/2048/2M.\n"
        "  hunyuan3d: low=128/10K/no-tex, medium=256/50K/tex, "
        "high=384/100K/tex, ultra=512/100K/tex/20step.",
    )
    target_faces: Optional[int] = Field(
        None,
        ge=10_000,
        le=2_000_000,
        description="Override: max triangle count (trellis only)",
    )
    # Legacy compat fields — kept as overrides
    pipeline_type: Optional[Literal["512", "1024", "1024_cascade"]] = Field(
        None, description="Deprecated: use quality instead"
    )
    texture_size: Optional[Literal[512, 1024, 2048]] = Field(
        None, description="Deprecated: use quality instead"
    )
    no_texture: Optional[bool] = Field(
        None, description="Deprecated: use quality instead"
    )
    texture: Optional[bool] = Field(
        None, description="Deprecated: use quality instead"
    )
    seed: int = Field(42, ge=0, le=2**32 - 1)
    num_inference_steps: Optional[int] = Field(
        None, ge=1, le=200, description="Override inference steps"
    )
    guidance_scale: Optional[float] = Field(
        None, ge=0.1, le=20.0, description="Override guidance scale (hunyuan3d)"
    )
    octree_resolution: Optional[int] = Field(
        None, ge=64, le=512, description="Override octree resolution (hunyuan3d)"
    )


class SubmitResponse(BaseModel):
    uid: str = Field(..., description="Persistent task identifier")
    model: MODEL_NAME
    status: Literal["queued", "dispatching", "processing", "completed"] = Field(
        "queued", description="'completed' when an identical task was already done"
    )


class TaskStatusResponse(BaseModel):
    status: Literal[
        "queued",
        "dispatching",
        "processing",
        "completed",
        "error",
        "expired",
        "cancelled",
    ]
    model: MODEL_NAME
    attempts: int
    position: Optional[int] = None
    download_url: Optional[str] = None
    bundle_url: Optional[str] = None
    message: Optional[str] = None


class RecentTaskItem(BaseModel):
    uid: str
    model: MODEL_NAME
    status: str
    quality: str
    submitted_at: float
    completed_at: Optional[float] = None
    download_url: Optional[str] = None
    bundle_url: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict, description="Params summary (image excluded)")


class RecentTasksResponse(BaseModel):
    tasks: list[RecentTaskItem]


class QueueItem(BaseModel):
    uid: str
    position: int
    model: MODEL_NAME


class QueueStatusResponse(BaseModel):
    queued: list[QueueItem]
    length: int
    counts: dict[str, int]


class CancelResponse(BaseModel):
    status: Literal["cancelled"]


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    database: bool
    backends: dict[str, bool]


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    database: bool
    backends: dict[str, bool]


class ImagePayloadError(ValueError):
    pass


class ImagePayloadTooLarge(ImagePayloadError):
    pass


class MeshPayloadError(ValueError):
    pass


class MeshPayloadTooLarge(MeshPayloadError):
    pass


class BackendTaskMissing(RuntimeError):
    pass


class QueueManager:
    """Serial dispatcher backed by SQLite and on-disk request payloads."""

    def __init__(
        self,
        save_dir: str,
        *,
        db_path: Optional[str] = None,
        backend_urls: Optional[dict[str, str]] = None,
        poll_interval: float = 5.0,
        backend_timeout: float = 3600.0,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        max_mesh_bytes: int = DEFAULT_MAX_MESH_BYTES,
        start_worker: bool = True,
    ):
        self.save_dir = os.path.abspath(save_dir)
        self.input_dir = os.path.join(self.save_dir, "inputs")
        self.result_dir = os.path.join(self.save_dir, "results")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        self.store = TaskStore(db_path or os.path.join(self.save_dir, "tasks.sqlite3"))
        self.backend_urls = dict(backend_urls or BACKENDS)
        self.poll_interval = poll_interval
        self.backend_timeout = backend_timeout
        self.max_image_bytes = max_image_bytes
        self.max_mesh_bytes = max_mesh_bytes
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker: Optional[threading.Thread] = None

        recovered = self.store.recoverable()
        for task in recovered:
            self._queue.put(task["uid"])
        if recovered:
            logger.info("Recovered %d unfinished task(s)", len(recovered))

        if start_worker:
            self._worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="durable-queue-worker",
            )
            self._worker.start()
            logger.info("Queue worker started")

    def _decode_image(self, encoded: str) -> bytes:
        if len(encoded) > ((self.max_image_bytes + 2) // 3) * 4:
            raise ImagePayloadTooLarge(
                f"Image exceeds the {self.max_image_bytes // (1024 * 1024)} MiB limit"
            )
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImagePayloadError("image must be valid base64") from exc
        if not raw:
            raise ImagePayloadError("image payload is empty")
        if len(raw) > self.max_image_bytes:
            raise ImagePayloadTooLarge(
                f"Image exceeds the {self.max_image_bytes // (1024 * 1024)} MiB limit"
            )
        return raw

    @staticmethod
    def _atomic_write(path: str, content: bytes) -> None:
        temp_path = f"{path}.{uuid.uuid4().hex}.tmp"
        try:
            with open(temp_path, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _resolve_quality_params(self, params: dict) -> dict:
        """Merge quality preset with individual overrides for backend forwarding."""
        model = params["model"]
        if model == "hunyuan3d_part":
            return {
                key: params[key]
                for key in (
                    "mode",
                    "points",
                    "prompts",
                    "prompt_batch_size",
                    "surface_points",
                    "steps",
                    "resolution",
                    "sdf_chunk_size",
                    "seed",
                    "official_fps_start",
                    "filename",
                )
                if key in params
            } | {"quality": params.get("mode", "segment")}
        quality = params.get("quality", "ultra")
        presets = TRELLIS_QUALITY_PRESETS if model == "trellis" else HUNYUAN_QUALITY_PRESETS
        preset = presets.get(quality, presets["ultra"])

        resolved = dict(preset)

        # Quality-agnostic fields
        resolved["seed"] = params.get("seed", 42)

        # Per-model overrides
        if model == "trellis":
            if params.get("pipeline_type"):
                resolved["pipeline_type"] = params["pipeline_type"]
            if params.get("texture_size"):
                resolved["texture_size"] = params["texture_size"]
            if params.get("target_faces"):
                resolved["target_faces"] = params["target_faces"]
            if params.get("no_texture") is not None:
                resolved["no_texture"] = params["no_texture"]
            if params.get("steps"):
                resolved["steps"] = params["steps"]
        else:  # hunyuan3d
            if params.get("octree_resolution"):
                resolved["octree_resolution"] = params["octree_resolution"]
            if params.get("num_inference_steps"):
                resolved["num_inference_steps"] = params["num_inference_steps"]
            if params.get("guidance_scale"):
                resolved["guidance_scale"] = params["guidance_scale"]
            if params.get("texture") is not None:
                resolved["texture"] = params["texture"]
            if params.get("face_count"):
                resolved["face_count"] = params["face_count"]

        resolved["quality"] = quality
        return resolved

    def submit(self, params: dict) -> tuple[str, str, Optional[str]]:
        """Submit a task. Returns (uid, status, existing_download_url).

        If task_id is provided and an identical task already exists, returns
        the existing task's UID with status='completed' and its download URL.
        If task_id is provided but params/image differ, raises HTTPException 409.
        """
        params = dict(params)
        raw_image = self._decode_image(params.pop("image"))
        image_sha256 = self._sha256(raw_image)
        model = params["model"]

        # Resolve quality presets for fingerprinting
        resolved = self._resolve_quality_params(params)

        client_task_id = params.get("task_id")
        uid = str(client_task_id) if client_task_id else str(uuid.uuid4())

        # Compute fingerprint for idempotency
        fp = compute_fingerprint(uid, model, resolved, image_sha256)

        # Check for existing identical task
        existing = self.store.find_by_fingerprint(fp)
        if existing:
            status = existing["status"]
            if status == "completed":
                result_path = existing.get("result_path")
                if result_path and os.path.exists(result_path):
                    logger.info(
                        "[%s] Idempotent hit: returning completed task",
                        uid[:8],
                    )
                    return uid, "completed", f"/download/{uid}"
            elif status in ("queued", "dispatching", "processing"):
                logger.info("[%s] Idempotent hit: task already in flight", uid[:8])
                return uid, status, None
            # Fall through for errored tasks — allow re-submit

        # Check for task_id conflict (same ID, different fingerprint)
        if client_task_id:
            existing_by_uid = self.store.get(uid)
            if existing_by_uid:
                existing_fp = existing_by_uid.get("request_fingerprint")
                if existing_fp and existing_fp != fp:
                    from fastapi import HTTPException as FastAPIHTTPException
                    raise FastAPIHTTPException(
                        status_code=409,
                        detail=(
                            f"Task ID {uid} already exists with different parameters. "
                            "Use a new task_id or submit the original request."
                        ),
                    )

        input_path = os.path.join(self.input_dir, f"{uid}.image")
        self._atomic_write(input_path, raw_image)
        try:
            self.store.create_task(
                uid=uid,
                model=model,
                params=resolved,
                input_path=input_path,
                request_fingerprint=fp,
                image_sha256=image_sha256,
            )
        except Exception:
            os.remove(input_path)
            raise
        self._queue.put(uid)
        logger.info("[%s] Queued (%s, quality=%s)", uid[:8], model, resolved.get("quality", "ultra"))
        return uid, "queued", None

    def submit_part(self, raw_mesh: bytes, params: dict) -> tuple[str, str, Optional[str]]:
        """Submit a binary GLB/OBJ to the Hunyuan3D-Part worker."""
        if not raw_mesh:
            raise MeshPayloadError("mesh payload is empty")
        if len(raw_mesh) > self.max_mesh_bytes:
            raise MeshPayloadTooLarge(
                f"Mesh exceeds the {self.max_mesh_bytes // (1024 * 1024)} MiB limit"
            )

        params = dict(params)
        filename = os.path.basename(str(params.get("filename", "mesh.glb")))
        suffix = os.path.splitext(filename)[1].lower()
        if suffix not in (".glb", ".obj"):
            raise MeshPayloadError("filename must end in .glb or .obj")
        params.update(model="hunyuan3d_part", filename=filename)
        mesh_sha256 = self._sha256(raw_mesh)
        resolved = self._resolve_quality_params(params)
        client_task_id = params.get("task_id")
        uid = str(client_task_id) if client_task_id else str(uuid.uuid4())
        fingerprint = compute_fingerprint(uid, "hunyuan3d_part", resolved, mesh_sha256)

        existing = self.store.find_by_fingerprint(fingerprint)
        if existing:
            status = existing["status"]
            if status == "completed":
                result_path = existing.get("result_path")
                if result_path and os.path.exists(result_path):
                    return uid, "completed", f"/download/{uid}"
            if status in ("queued", "dispatching", "processing"):
                return uid, status, None

        if client_task_id:
            existing_by_uid = self.store.get(uid)
            if existing_by_uid:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Task ID {uid} already exists. "
                        "Use a new task_id or submit the original request."
                    ),
                )

        input_path = os.path.join(self.input_dir, f"{uid}{suffix}")
        self._atomic_write(input_path, raw_mesh)
        try:
            self.store.create_task(
                uid=uid,
                model="hunyuan3d_part",
                params=resolved,
                input_path=input_path,
                request_fingerprint=fingerprint,
                image_sha256=mesh_sha256,
            )
        except Exception:
            os.remove(input_path)
            raise
        self._queue.put(uid)
        logger.info("[%s] Queued (hunyuan3d_part, mode=%s)", uid[:8], resolved["quality"])
        return uid, "queued", None

    def recent_tasks(self, limit: int = 50, model: Optional[str] = None, status_filter: Optional[str] = None) -> list[dict]:
        """Return recent tasks with image data excluded."""
        raw = self.store.list_recent(limit=limit, model=model, status=status_filter)
        result = []
        for task in raw:
            p = dict(task.get("params", {}))
            if "image" in p:
                p.pop("image", None)
            entry = {
                "uid": task["uid"],
                "model": task["model"],
                "status": task["status"],
                "quality": p.get("quality", "ultra"),
                "submitted_at": task["created_at"],
                "completed_at": task.get("completed_at"),
                "params": p,
            }
            if task["status"] == "completed":
                result_path = task.get("result_path")
                if result_path and os.path.exists(result_path):
                    entry["download_url"] = f"/download/{task['uid']}"
                    bundle_path = os.path.join(self.result_dir, f"{task['uid']}.zip")
                    if task["model"] == "hunyuan3d_part" and os.path.exists(bundle_path):
                        entry["bundle_url"] = f"/bundle/{task['uid']}"
            result.append(entry)
        return result

    def get_task(self, uid: str) -> Optional[dict]:
        return self.store.get(uid)

    def cancel(self, uid: str) -> bool:
        cancelled = self.store.cancel(uid)
        if cancelled:
            task = self.store.get(uid)
            if task:
                self._remove_file(task.get("input_path"))
        return cancelled

    def queue_list(self) -> list[dict]:
        return [
            {"uid": task["uid"], "position": index, "model": task["model"]}
            for index, task in enumerate(self.store.queued(), start=1)
        ]

    def _params_with_image(self, task: dict) -> dict:
        with open(task["input_path"], "rb") as stream:
            encoded_image = base64.b64encode(stream.read()).decode("ascii")
        return {**task["params"], "image": encoded_image}

    def _forward_to_backend(self, task: dict) -> str:
        model = task["model"]
        self._prepare_backend(model)
        params = task["params"]
        if model == "hunyuan3d_part":
            body = {
                "task_id": task["uid"],
                "mesh_path": task["input_path"],
                **{
                    key: params[key]
                    for key in (
                        "mode",
                        "points",
                        "prompts",
                        "prompt_batch_size",
                        "surface_points",
                        "steps",
                        "resolution",
                        "sdf_chunk_size",
                        "seed",
                        "official_fps_start",
                    )
                    if key in params
                },
            }
        else:
            params = self._params_with_image(task)
            body = {
                "task_id": task["uid"],
                "image": params["image"],
                "seed": params.get("seed", 42),
                "quality": params.get("quality", "ultra"),
            }
        if model == "trellis":
            body.update(
                pipeline_type=params.get("pipeline_type", "1024_cascade"),
                texture_size=params.get("texture_size", 2048),
                target_faces=params.get("target_faces", 2_000_000),
                no_texture=params.get("no_texture", False),
                steps=params.get("steps"),
            )
        elif model == "hunyuan3d":
            body.update(
                texture=params.get("texture", True),
                num_inference_steps=params.get("num_inference_steps", 20),
                guidance_scale=params.get("guidance_scale", 5.0),
                octree_resolution=params.get("octree_resolution", 512),
                face_count=params.get("face_count", 100_000),
            )

        request = urllib.request.Request(
            f"{self.backend_urls[model]}/send",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read())
        backend_uid = data.get("uid")
        if not backend_uid:
            raise RuntimeError(f"{model} backend did not return a task uid")
        logger.info(
            "[%s] Dispatched to %s, backend uid=%s",
            task["uid"][:8],
            model,
            backend_uid,
        )
        return backend_uid

    def _prepare_backend(self, model: str) -> None:
        """Unload the other model family before allocating unified memory."""
        other_models = [name for name in self.backend_urls if name != model]
        for other_model in other_models:
            request = urllib.request.Request(
                f"{self.backend_urls[other_model]}/unload",
                data=b"",
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"{other_model} backend unload returned {response.status}"
                        )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not unload {other_model} before {model} dispatch: {exc}"
                ) from exc

    def _download_result(self, task_uid: str, model: str, backend_uid: str) -> str:
        result_path = os.path.join(self.result_dir, f"{task_uid}.glb")
        temp_path = f"{result_path}.part"
        try:
            with urllib.request.urlopen(
                f"{self.backend_urls[model]}/download/{backend_uid}", timeout=120
            ) as response, open(temp_path, "wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp_path, result_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        if model == "hunyuan3d_part":
            bundle_path = os.path.join(self.result_dir, f"{task_uid}.zip")
            bundle_temp_path = f"{bundle_path}.part"
            try:
                with urllib.request.urlopen(
                    f"{self.backend_urls[model]}/bundle/{backend_uid}", timeout=120
                ) as response, open(bundle_temp_path, "wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(bundle_temp_path, bundle_path)
            except Exception:
                self._remove_file(result_path)
                raise
            finally:
                self._remove_file(bundle_temp_path)
        return result_path

    def _poll_backend(self, task_uid: str, model: str, backend_uid: str) -> str:
        deadline = time.monotonic() + self.backend_timeout
        last_connection_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"{self.backend_urls[model]}/status/{backend_uid}", timeout=10
                ) as response:
                    status = json.loads(response.read())
                last_connection_error = None
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise BackendTaskMissing(
                        f"{model} backend no longer knows task {backend_uid}"
                    ) from exc
                raise
            except (urllib.error.URLError, TimeoutError) as exc:
                last_connection_error = exc
                logger.warning("[%s] Backend poll failed: %s", task_uid[:8], exc)
                time.sleep(self.poll_interval)
                continue

            state = status.get("status")
            if state == "completed":
                return self._download_result(task_uid, model, backend_uid)
            if state == "error":
                raise RuntimeError(status.get("message", f"{model} backend error"))
            time.sleep(self.poll_interval)

        detail = f": {last_connection_error}" if last_connection_error else ""
        raise TimeoutError(
            f"{model} backend did not complete within {self.backend_timeout:.0f}s{detail}"
        )

    def _process_task(self, uid: str) -> None:
        task = self.store.claim(uid)
        if task is None:
            return

        backend_uid = task.get("backend_uid")
        if backend_uid:
            try:
                result_path = self._poll_backend(uid, task["model"], backend_uid)
            except BackendTaskMissing:
                logger.warning("[%s] Re-dispatching missing backend task", uid[:8])
                backend_uid = None
                task = self.store.retry(uid)
                if task is None:
                    raise RuntimeError(f"Task {uid} disappeared while retrying")
            else:
                self._complete(task, backend_uid, result_path)
                return

        backend_uid = self._forward_to_backend(task)
        self.store.update(uid, status="processing", backend_uid=backend_uid, error=None)
        result_path = self._poll_backend(uid, task["model"], backend_uid)
        self._complete(task, backend_uid, result_path)

    def _complete(self, task: dict, backend_uid: str, result_path: str) -> None:
        now = time.time()
        self.store.update(
            task["uid"],
            status="completed",
            backend_uid=backend_uid,
            result_path=result_path,
            completed_at=now,
            error=None,
        )
        self._remove_file(task.get("input_path"))
        started_at = task.get("started_at") or now
        logger.info("[%s] Completed in %.0fs", task["uid"][:8], now - started_at)

    def _worker_loop(self) -> None:
        while True:
            uid = self._queue.get()
            try:
                self._process_task(uid)
            except Exception as exc:
                logger.exception("[%s] Failed: %s", uid[:8], exc)
                self.store.update(
                    uid,
                    status="error",
                    error=str(exc),
                    completed_at=time.time(),
                )
                task = self.store.get(uid)
                if task:
                    self._remove_file(task.get("input_path"))
            finally:
                self._queue.task_done()

    @staticmethod
    def _remove_file(path: Optional[str]) -> None:
        if path:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def cleanup(self, max_age_seconds: int) -> int:
        cutoff = time.time() - max_age_seconds
        cleaned = 0
        for task in self.store.cleanup_candidates(cutoff):
            self._remove_file(task.get("input_path"))
            self._remove_file(task.get("result_path"))
            self._remove_file(os.path.join(self.result_dir, f"{task['uid']}.zip"))
            if self.store.delete(task["uid"]):
                cleaned += 1
        return cleaned

    def backend_health(self) -> dict[str, bool]:
        result = {}
        for model, base_url in self.backend_urls.items():
            try:
                with urllib.request.urlopen(
                    f"{base_url}/health", timeout=2
                ) as response:
                    result[model] = response.status == 200
            except Exception:
                result[model] = False
        return result


app = FastAPI(
    title="3D Inference Queue Proxy",
    version="3.0.0",
    description=(
        "Durable, single-queue API for TRELLIS and Hunyuan3D image-to-3D "
        "inference. Submit a task, poll its status, then download the GLB result."
    ),
    openapi_tags=[
        {"name": "tasks", "description": "Submit and inspect inference tasks."},
        {"name": "results", "description": "Download completed GLB files."},
        {"name": "operations", "description": "Queue and service health."},
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

_manager: Optional[QueueManager] = None
_api_key: Optional[str] = None


def get_manager() -> QueueManager:
    if _manager is None:
        raise HTTPException(status_code=503, detail="Queue manager is not initialized")
    return _manager


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if (
        _api_key
        and request.url.path not in ("/health", "/ready", "/docs", "/openapi.json")
        and request.headers.get("X-API-Key") != _api_key
    ):
        return JSONResponse({"detail": "Invalid API key"}, status_code=401)
    return await call_next(request)


@app.post(
    "/submit",
    response_model=SubmitResponse,
    tags=["tasks"],
    summary="Submit an image-to-3D task (idempotent with task_id)",
    responses={
        200: {"description": "Task submitted or existing task returned"},
        409: {"description": "task_id exists with different params"},
        413: {"description": "Image too large"},
    },
)
async def submit_task(request: SubmitRequest):
    try:
        uid, status, dl_url = get_manager().submit(request.model_dump())
    except HTTPException:
        raise
    except ImagePayloadTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ImagePayloadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"uid": uid, "model": request.model, "status": status}


@app.post(
    "/submit/part",
    response_model=SubmitResponse,
    tags=["tasks"],
    summary="Submit a GLB/OBJ for Hunyuan3D-Part MLX inference",
    responses={
        200: {"description": "Part task submitted or existing task returned"},
        409: {"description": "task_id already exists"},
        413: {"description": "Mesh too large"},
    },
)
async def submit_part_task(
    request: Request,
    filename: str = Query(..., min_length=1, max_length=255),
    task_id: Optional[UUID] = Query(None),
    mode: Literal["segment", "generate_parts"] = Query("segment"),
    points: int = Query(100_000, ge=1_000, le=200_000),
    prompts: int = Query(400, ge=1, le=1_000),
    prompt_batch_size: int = Query(8, ge=1, le=128),
    surface_points: int = Query(81_920, ge=4_096, le=200_000),
    steps: int = Query(50, ge=1, le=100),
    resolution: Literal[128, 256, 512] = Query(128),
    sdf_chunk_size: int = Query(100_000, ge=10_000, le=1_000_000),
    seed: int = Query(42, ge=0, le=2**32 - 1),
    official_fps_start: bool = Query(False),
):
    manager = get_manager()
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc
        if declared_size > manager.max_mesh_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Mesh exceeds the {manager.max_mesh_bytes // (1024 * 1024)} MiB limit",
            )
    raw_mesh = await request.body()
    params = {
        "task_id": task_id,
        "filename": filename,
        "mode": mode,
        "points": points,
        "prompts": prompts,
        "prompt_batch_size": prompt_batch_size,
        "surface_points": surface_points,
        "steps": steps,
        "resolution": resolution,
        "sdf_chunk_size": sdf_chunk_size,
        "seed": seed,
        "official_fps_start": official_fps_start,
    }
    try:
        uid, status, _ = manager.submit_part(raw_mesh, params)
    except HTTPException:
        raise
    except MeshPayloadTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except MeshPayloadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"uid": uid, "model": "hunyuan3d_part", "status": status}


@app.post(
    "/send",
    response_model=SubmitResponse,
    tags=["tasks"],
    summary="Submit using the legacy auto-detect format",
    deprecated=True,
)
async def send_task_compat(request: dict):
    model = "trellis" if "pipeline_type" in request else "hunyuan3d"
    validated = SubmitRequest.model_validate({"model": model, **request})
    return await submit_task(validated)


@app.get(
    "/status/{uid}",
    response_model=TaskStatusResponse,
    response_model_exclude_none=True,
    tags=["tasks"],
    summary="Get task status",
)
async def check_status(uid: str):
    manager = get_manager()
    task = manager.get_task(uid)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    response = {
        "status": task["status"],
        "model": task["model"],
        "attempts": task["attempts"],
    }
    if task["status"] == "completed":
        if not task.get("result_path") or not os.path.exists(task["result_path"]):
            response.update(status="expired", message="Result file has expired")
        else:
            response["download_url"] = f"/download/{uid}"
            bundle_path = os.path.join(manager.result_dir, f"{uid}.zip")
            if task["model"] == "hunyuan3d_part" and os.path.exists(bundle_path):
                response["bundle_url"] = f"/bundle/{uid}"
    elif task["status"] == "error":
        response["message"] = task["error"]
    elif task["status"] == "queued":
        for queued in manager.queue_list():
            if queued["uid"] == uid:
                response["position"] = queued["position"]
                break
    return response


@app.get(
    "/download/{uid}",
    tags=["results"],
    summary="Download a completed GLB",
    responses={
        200: {
            "description": "GLB model file",
            "content": {"model/gltf-binary": {}},
        },
        202: {"description": "Task has not completed"},
        404: {"description": "Task was not found or failed"},
        410: {"description": "Result retention period has expired"},
    },
)
async def download_model(uid: str):
    task = get_manager().get_task(uid)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] == "error":
        return JSONResponse({"error": task["error"]}, status_code=404)
    if task["status"] != "completed":
        return JSONResponse({"status": task["status"]}, status_code=202)
    result_path = task.get("result_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=410, detail="Result file has expired")
    return FileResponse(
        result_path,
        media_type="model/gltf-binary",
        filename=f"{uid}.glb",
    )


@app.get(
    "/bundle/{uid}",
    tags=["results"],
    summary="Download the complete Hunyuan3D-Part artifact bundle",
)
async def download_bundle(uid: str):
    manager = get_manager()
    task = manager.get_task(uid)
    if task is None or task["model"] != "hunyuan3d_part":
        raise HTTPException(status_code=404, detail="Part task not found")
    if task["status"] != "completed":
        return JSONResponse({"status": task["status"]}, status_code=202)
    bundle_path = os.path.join(manager.result_dir, f"{uid}.zip")
    if not os.path.exists(bundle_path):
        raise HTTPException(status_code=410, detail="Artifact bundle has expired")
    return FileResponse(
        bundle_path,
        media_type="application/zip",
        filename=f"{uid}.zip",
    )


@app.get(
    "/tasks/recent",
    response_model=RecentTasksResponse,
    response_model_exclude_none=True,
    tags=["tasks"],
    summary="List recent tasks (persistent across restarts)",
)
async def list_recent_tasks(
    limit: int = Query(20, ge=1, le=100),
    model: Optional[str] = Query(None, pattern="^(trellis|hunyuan3d|hunyuan3d_part)?$"),
    status: Optional[str] = Query(
        None, pattern="^(queued|dispatching|processing|completed|error|cancelled)?$"
    ),
):
    """List recent tasks, newest first.  Image data is never included."""
    tasks = get_manager().recent_tasks(limit=limit, model=model, status_filter=status)
    return {"tasks": tasks}


@app.get(
    "/queue",
    response_model=QueueStatusResponse,
    tags=["operations"],
    summary="Inspect the durable queue",
)
async def queue_status():
    manager = get_manager()
    queued = manager.queue_list()
    return {
        "queued": queued,
        "length": len(queued),
        "counts": manager.store.counts(),
    }


@app.post(
    "/cancel/{uid}",
    response_model=CancelResponse,
    tags=["tasks"],
    summary="Cancel a queued task",
)
async def cancel_task(uid: str):
    if get_manager().cancel(uid):
        return {"status": "cancelled"}
    raise HTTPException(
        status_code=400, detail="Cannot cancel: not queued or not found"
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["operations"],
    summary="Check queue process and database health",
)
async def health():
    manager = get_manager()
    database_ok = manager.store.ping()
    backends = manager.backend_health()
    return {
        "status": "healthy" if database_ok else "unhealthy",
        "database": database_ok,
        "backends": backends,
    }


@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["operations"],
    summary="Check whether all backends are ready",
    responses={503: {"description": "Database or a model backend is unavailable"}},
)
async def ready():
    manager = get_manager()
    database_ok = manager.store.ping()
    backends = manager.backend_health()
    payload = {
        "status": "ready" if database_ok and all(backends.values()) else "not_ready",
        "database": database_ok,
        "backends": backends,
    }
    if payload["status"] != "ready":
        return JSONResponse(payload, status_code=503)
    return payload


def _cleanup_loop(manager: QueueManager, retention_seconds: int) -> None:
    while True:
        cleaned = manager.cleanup(retention_seconds)
        if cleaned:
            logger.info("Cleaned up %d expired task(s)", cleaned)
        time.sleep(min(600, max(30, retention_seconds // 6)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Durable 3D inference queue")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--cache-path", default="./server_cache")
    parser.add_argument("--db-path")
    parser.add_argument("--backend-timeout", type=float, default=3600)
    parser.add_argument("--poll-interval", type=float, default=5)
    parser.add_argument("--max-image-mib", type=int, default=25)
    parser.add_argument("--max-mesh-mib", type=int, default=100)
    parser.add_argument("--retention-seconds", type=int, default=86400)
    parser.add_argument("--api-key", default=os.environ.get("INFERENCE_API_KEY"))
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    log_path = os.path.join(args.cache_path, "server.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    ))
    logging.getLogger().addHandler(file_handler)

    save_dir = os.path.abspath(args.cache_path)
    _manager = QueueManager(
        save_dir,
        db_path=args.db_path,
        poll_interval=args.poll_interval,
        backend_timeout=args.backend_timeout,
        max_image_bytes=args.max_image_mib * 1024 * 1024,
        max_mesh_bytes=args.max_mesh_mib * 1024 * 1024,
    )
    _api_key = args.api_key
    cleanup_thread = threading.Thread(
        target=_cleanup_loop,
        args=(_manager, args.retention_seconds),
        daemon=True,
        name="task-cleanup",
    )
    cleanup_thread.start()

    logger.info("Queue proxy starting on %s:%s", args.host, args.port)
    logger.info("Persistent store: %s", _manager.store.db_path)
    logger.info("Backends: %s", _manager.backend_urls)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_keep_alive=30,
    )
