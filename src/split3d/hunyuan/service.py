from __future__ import annotations

import gc
import json
import os
import shutil
import threading
import time
import uuid
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import UUID

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


class PartTaskRequest(BaseModel):
    task_id: UUID | None = None
    mesh_path: str = Field(..., min_length=1)
    mode: Literal["segment", "generate_parts"] = "segment"
    points: int = Field(100_000, ge=1_000, le=200_000)
    prompts: int = Field(400, ge=1, le=1_000)
    prompt_batch_size: int = Field(1, ge=1, le=8)
    surface_points: int = Field(81_920, ge=4_096, le=200_000)
    steps: int = Field(50, ge=1, le=100)
    resolution: Literal[128, 256, 512] = 128
    sdf_chunk_size: int = Field(100_000, ge=10_000, le=1_000_000)
    seed: int = Field(42, ge=0, le=2**32 - 1)
    official_fps_start: bool = True
    clean_mesh: bool = True
    connectivity: bool = True
    postprocess: bool = True
    postprocess_threshold: float = Field(0.95, ge=0.0, le=1.0)


class PartRunnerProtocol(Protocol):
    @property
    def model_state(self) -> dict[str, bool]: ...

    def run(
        self,
        request: PartTaskRequest,
        mesh_path: Path,
        output_dir: Path,
        progress: Callable[[str, float], None],
    ) -> Path: ...

    def unload(self) -> bool: ...


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class MLXPartRunner:
    def __init__(self, model_dir: Path, p3_weights: Path) -> None:
        self.model_dir = model_dir
        self.p3_weights = p3_weights
        self._p3sam: Any = None
        self._xpart: Any = None
        self._lock = threading.RLock()

    @property
    def model_state(self) -> dict[str, bool]:
        return {"p3sam_loaded": self._p3sam is not None, "xpart_loaded": self._xpart is not None}

    def _get_p3sam(self) -> Any:
        if self._xpart is not None:
            return self._xpart.p3sam
        if self._p3sam is None:
            from split3d.hunyuan.p3sam_mlx import P3SAMMLX

            self._p3sam = P3SAMMLX.from_safetensors(self.p3_weights)
        return self._p3sam

    def _get_xpart(self) -> Any:
        if self._xpart is None:
            from split3d.hunyuan.xpart_pipeline_mlx import XPartPipelineMLX

            self._p3sam = None
            gc.collect()
            self._xpart = XPartPipelineMLX.from_pretrained(
                self.model_dir,
                p3_weights=self.p3_weights,
            )
        return self._xpart

    def run(
        self,
        request: PartTaskRequest,
        mesh_path: Path,
        output_dir: Path,
        progress: Callable[[str, float], None],
    ) -> Path:
        import mlx.core as mx
        import trimesh

        with self._lock:
            started = time.perf_counter()
            mx.reset_peak_memory()
            mesh = trimesh.load(mesh_path, force="mesh")
            output_dir.mkdir(parents=True, exist_ok=True)
            if request.mode == "segment":
                from split3d.hunyuan.p3sam_pipeline import save_segmentation, segment_mesh_mlx

                result = segment_mesh_mlx(
                    self._get_p3sam(),
                    mesh,
                    point_count=request.points,
                    prompt_count=request.prompts,
                    prompt_batch_size=request.prompt_batch_size,
                    seed=request.seed,
                    prompt_start_index=None if request.official_fps_start else 0,
                    clean_mesh=request.clean_mesh,
                    connectivity=request.connectivity,
                    postprocess=request.postprocess,
                    postprocess_threshold=request.postprocess_threshold,
                    progress=progress,
                )
                save_segmentation(mesh, result, output_dir, seed=request.seed)
                primary = output_dir / "segmented.glb"
                runtime = {
                    "backend": "mlx",
                    "mode": request.mode,
                    "total_seconds": time.perf_counter() - started,
                    "peak_memory_bytes": mx.get_peak_memory(),
                    "stage_seconds": result.stage_seconds,
                    "diagnostics": asdict(result.diagnostics),
                }
            else:
                result = self._get_xpart()(
                    mesh,
                    point_count=request.points,
                    prompt_count=request.prompts,
                    prompt_batch_size=request.prompt_batch_size,
                    surface_point_count=request.surface_points,
                    num_inference_steps=request.steps,
                    octree_resolution=request.resolution,
                    sdf_chunk_size=request.sdf_chunk_size,
                    seed=request.seed,
                    official_fps_start=request.official_fps_start,
                    clean_mesh=request.clean_mesh,
                    connectivity=request.connectivity,
                    postprocess=request.postprocess,
                    postprocess_threshold=request.postprocess_threshold,
                    progress=progress,
                )
                primary = output_dir / "xpart_scene.glb"
                result.scene.export(primary)
                np.save(output_dir / "latents.npy", result.latents)
                np.save(output_dir / "bboxes.npy", result.bboxes)
                runtime = {
                    "backend": "mlx",
                    "mode": request.mode,
                    "total_seconds": time.perf_counter() - started,
                    "peak_memory_bytes": mx.get_peak_memory(),
                    "part_count": int(len(result.bboxes)),
                    "stage_seconds": result.stage_seconds,
                    "resolution": request.resolution,
                }
            _atomic_json(output_dir / "runtime.json", runtime)
            return primary

    def unload(self) -> bool:
        import mlx.core as mx

        with self._lock:
            loaded = self._p3sam is not None or self._xpart is not None
            self._p3sam = None
            self._xpart = None
            gc.collect()
            mx.clear_cache()
            return loaded


class PartJobService:
    def __init__(
        self,
        cache_path: Path,
        input_root: Path,
        runner: PartRunnerProtocol,
        *,
        retention_seconds: int = 86_400,
        max_mesh_bytes: int = 100 * 1024 * 1024,
    ) -> None:
        self.cache_path = cache_path.resolve()
        self.input_root = input_root.resolve()
        self.runner = runner
        self.retention_seconds = retention_seconds
        self.max_mesh_bytes = max_mesh_bytes
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hunyuan-part")
        self._lock = threading.RLock()
        self._processing: str | None = None
        self._stop_event = threading.Event()
        self._recover_interrupted_jobs()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="hunyuan-part-cleanup",
        )
        self._cleanup_thread.start()

    def _job_dir(self, uid: str) -> Path:
        return self.cache_path / uid

    def _status_path(self, uid: str) -> Path:
        return self._job_dir(uid) / "status.json"

    def _read_status(self, uid: str) -> dict[str, Any] | None:
        path = self._status_path(uid)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def _write_status(self, uid: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            current = self._read_status(uid) or {"uid": uid, "created_at": time.time()}
            current.update(updates, updated_at=time.time())
            _atomic_json(self._status_path(uid), current)
            return current

    def _recover_interrupted_jobs(self) -> None:
        for path in self.cache_path.glob("*/status.json"):
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("status") in {"queued", "processing"}:
                self._write_status(
                    state["uid"],
                    status="error",
                    message="Worker restarted before the task completed",
                )

    def _validated_source(self, mesh_path: str) -> Path:
        source = Path(mesh_path).resolve()
        try:
            source.relative_to(self.input_root)
        except ValueError as exc:
            raise ValueError("mesh_path must be inside the configured input root") from exc
        if source.suffix.lower() not in {".glb", ".obj"}:
            raise ValueError("only .glb and .obj inputs are supported")
        if not source.is_file():
            raise ValueError("mesh input does not exist")
        size = source.stat().st_size
        if size == 0:
            raise ValueError("mesh input is empty")
        if size > self.max_mesh_bytes:
            raise ValueError(f"mesh exceeds the {self.max_mesh_bytes // (1024 * 1024)} MiB limit")
        return source

    def submit(self, request: PartTaskRequest) -> tuple[str, str]:
        source = self._validated_source(request.mesh_path)
        uid = str(request.task_id or uuid.uuid4())
        params = request.model_dump(mode="json")
        params["task_id"] = uid
        existing = self._read_status(uid)
        if existing is not None:
            if existing.get("params") != params:
                raise ValueError("task_id already exists with different parameters")
            return uid, str(existing["status"])

        job_dir = self._job_dir(uid)
        job_dir.mkdir(parents=True, exist_ok=False)
        copied_input = job_dir / f"input{source.suffix.lower()}"
        shutil.copy2(source, copied_input)
        self._write_status(uid, status="queued", params=params, mode=request.mode)
        self._executor.submit(self._run, uid, request, copied_input)
        return uid, "queued"

    def _run(self, uid: str, request: PartTaskRequest, mesh_path: Path) -> None:
        with self._lock:
            self._processing = uid
        self._write_status(uid, status="processing", stage="load")

        def progress(stage: str, seconds: float) -> None:
            self._write_status(uid, status="processing", stage=stage, stage_seconds=seconds)

        try:
            output_dir = self._job_dir(uid) / "output"
            primary = self.runner.run(request, mesh_path, output_dir, progress)
            bundle = self._job_dir(uid) / "artifacts.zip"
            with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(output_dir.iterdir()):
                    archive.write(path, arcname=path.name)
            self._write_status(
                uid,
                status="completed",
                stage="completed",
                result_file=primary.name,
                bundle_file=bundle.name,
            )
        except Exception as exc:
            error_path = self._job_dir(uid) / "error.txt"
            error_path.write_text(str(exc), encoding="utf-8")
            self._write_status(uid, status="error", stage="error", message=str(exc))
        finally:
            with self._lock:
                self._processing = None

    def status(self, uid: str) -> dict[str, Any] | None:
        return self._read_status(uid)

    def primary_path(self, uid: str) -> Path | None:
        state = self._read_status(uid)
        if state is None or state.get("status") != "completed":
            return None
        path = self._job_dir(uid) / "output" / state["result_file"]
        return path if path.is_file() else None

    def bundle_path(self, uid: str) -> Path | None:
        state = self._read_status(uid)
        if state is None or state.get("status") != "completed":
            return None
        path = self._job_dir(uid) / state["bundle_file"]
        return path if path.is_file() else None

    def recent(self, limit: int) -> list[dict[str, Any]]:
        states = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in self.cache_path.glob("*/status.json")
        ]
        return sorted(states, key=lambda item: item["updated_at"], reverse=True)[:limit]

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._processing is not None

    def unload(self) -> bool:
        if self.busy:
            raise RuntimeError("inference worker is busy")
        return self.runner.unload()

    def cleanup(self) -> int:
        cutoff = time.time() - self.retention_seconds
        cleaned = 0
        for path in self.cache_path.iterdir():
            if not path.is_dir():
                continue
            state = self._read_status(path.name)
            if state and state.get("status") in {"completed", "error"} and state["updated_at"] < cutoff:
                shutil.rmtree(path)
                cleaned += 1
        return cleaned

    def _cleanup_loop(self) -> None:
        interval = min(600, max(30, self.retention_seconds // 6))
        while not self._stop_event.wait(interval):
            self.cleanup()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._cleanup_thread.join(timeout=5)
        self._executor.shutdown(wait=True, cancel_futures=True)


def create_app(service: PartJobService) -> FastAPI:
    app = FastAPI(
        title="Hunyuan3D-Part MLX Worker",
        version="0.1.0",
        description="Local async worker for P3-SAM segmentation and X-Part generation.",
    )

    @app.post("/send")
    async def send(request: PartTaskRequest) -> dict[str, str]:
        try:
            uid, status = service.submit(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"uid": uid, "status": status}

    @app.get("/status/{uid}")
    async def status(uid: str) -> dict[str, Any]:
        state = service.status(uid)
        if state is None:
            raise HTTPException(status_code=404, detail="Task not found")
        response = {
            key: value
            for key, value in state.items()
            if key not in {"params", "result_file", "bundle_file"}
        }
        if state["status"] == "completed":
            response.update(download_url=f"/download/{uid}", bundle_url=f"/bundle/{uid}")
        return response

    @app.get("/download/{uid}")
    async def download(uid: str) -> FileResponse:
        path = service.primary_path(uid)
        if path is None:
            raise HTTPException(status_code=404, detail="Completed result not found")
        return FileResponse(path, media_type="model/gltf-binary", filename=path.name)

    @app.get("/bundle/{uid}")
    async def bundle(uid: str) -> FileResponse:
        path = service.bundle_path(uid)
        if path is None:
            raise HTTPException(status_code=404, detail="Completed artifact bundle not found")
        return FileResponse(path, media_type="application/zip", filename=f"{uid}.zip")

    @app.get("/tasks/recent")
    async def recent(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
        return {"tasks": service.recent(limit)}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "busy": service.busy,
            **service.runner.model_state,
        }

    @app.post("/unload")
    async def unload() -> dict[str, Any]:
        try:
            unloaded = service.unload()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"status": "unloaded", "was_loaded": unloaded}

    return app
