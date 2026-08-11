from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from split3d.hunyuan.p3sam_mlx import P3SAMMLX
from split3d.hunyuan.p3sam_pipeline import segment_mesh
from split3d.hunyuan.xpart_conditioner_mlx import XPartConditionerMLX
from split3d.hunyuan.xpart_partformer_mlx import PartFormerMLX
from split3d.hunyuan.xpart_shape_mlx import ShapeVAEDecoderMLX

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - MLX is available only on Apple silicon
    mx = None


def _require_mlx() -> None:
    if mx is None:
        raise RuntimeError("X-Part pipeline requires Apple silicon and MLX")


@dataclass(frozen=True)
class XPartResult:
    scene: Any
    latents: np.ndarray
    bboxes: np.ndarray
    center: np.ndarray
    scale: float
    stage_seconds: dict[str, float] = field(default_factory=dict)


def normalize_mesh(mesh: Any) -> tuple[Any, np.ndarray, float]:
    normalized = mesh.copy()
    vertices = np.asarray(normalized.vertices, dtype=np.float64)
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = (minimum + maximum) / 2
    scale = float(np.max(maximum - minimum) / 2 / 0.8)
    normalized.vertices = (vertices - center) / scale
    return normalized, center, scale


def sample_surface(mesh: Any, count: int, *, seed: int) -> np.ndarray:
    import trimesh

    points, face_indices = trimesh.sample.sample_surface(mesh, count, seed=seed)
    normals = np.asarray(mesh.face_normals)[face_indices]
    labels = np.zeros((count, 1), dtype=np.float32)
    return np.concatenate([points.astype(np.float32), normals.astype(np.float32), labels], axis=-1)


def sample_part_surfaces(
    mesh: Any,
    bboxes: np.ndarray,
    count: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    surfaces = []
    valid_boxes = []
    for bbox in np.asarray(bboxes):
        inside = np.logical_and(vertices >= bbox[0], vertices <= bbox[1]).all(axis=-1)
        face_indices = np.flatnonzero(inside.any(axis=-1))
        if len(face_indices) == 0:
            continue
        submesh = mesh.submesh([face_indices], append=True, repair=False)
        surfaces.append(sample_surface(submesh, count, seed=seed))
        valid_boxes.append(bbox)
    if not surfaces:
        raise RuntimeError("P3-SAM did not produce any non-empty part bounding boxes")
    return np.stack(surfaces), np.asarray(valid_boxes, dtype=np.float32)


class XPartPipelineMLX:
    def __init__(
        self,
        p3sam: P3SAMMLX,
        conditioner: XPartConditionerMLX,
        partformer: PartFormerMLX,
        shapevae: ShapeVAEDecoderMLX,
    ) -> None:
        _require_mlx()
        self.p3sam = p3sam
        self.conditioner = conditioner
        self.partformer = partformer
        self.shapevae = shapevae

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, *, p3_weights: str | Path | None = None) -> XPartPipelineMLX:
        model_dir = Path(model_dir)
        p3_path = Path(p3_weights) if p3_weights is not None else model_dir / "p3sam" / "p3sam.safetensors"
        return cls(
            P3SAMMLX.from_safetensors(p3_path),
            XPartConditionerMLX.from_safetensors(model_dir / "conditioner" / "conditioner.safetensors"),
            PartFormerMLX.from_safetensors(model_dir / "model" / "model.safetensors"),
            ShapeVAEDecoderMLX.from_safetensors(model_dir / "shapevae" / "shapevae.safetensors"),
        )

    def __call__(
        self,
        mesh: Any,
        *,
        point_count: int = 100_000,
        prompt_count: int = 400,
        prompt_batch_size: int = 1,
        surface_point_count: int = 81_920,
        num_inference_steps: int = 50,
        octree_resolution: int = 256,
        sdf_chunk_size: int = 100_000,
        seed: int = 42,
        official_fps_start: bool = True,
        clean_mesh: bool = True,
        connectivity: bool = True,
        postprocess: bool = True,
        postprocess_threshold: float = 0.95,
        output_latents: bool = False,
        progress: Callable[[str, float], None] | None = None,
    ) -> XPartResult:
        import trimesh

        if surface_point_count < 4096:
            raise ValueError("surface_point_count must be at least 4096")
        timings: dict[str, float] = {}

        def finish(name: str, started: float) -> float:
            elapsed = time.perf_counter() - started
            timings[name] = elapsed
            if progress is not None:
                progress(name, elapsed)
            return time.perf_counter()

        stage = time.perf_counter()
        normalized, center, scale = normalize_mesh(mesh)
        object_surface = sample_surface(normalized, surface_point_count, seed=seed)[None]
        stage = finish("prepare_object", stage)

        segmentation = segment_mesh(
            self.p3sam,
            normalized,
            point_count=point_count,
            prompt_count=prompt_count,
            prompt_batch_size=prompt_batch_size,
            seed=seed,
            prompt_start_index=None if official_fps_start else 0,
            clean_mesh=clean_mesh,
            connectivity=connectivity,
            postprocess=postprocess,
            postprocess_threshold=postprocess_threshold,
        )
        part_surfaces, bboxes = sample_part_surfaces(
            normalized,
            segmentation.bboxes,
            surface_point_count,
            seed=seed,
        )
        stage = finish("predict_and_sample_parts", stage)

        contexts = self.conditioner(part_surfaces, object_surface, seed=seed)
        stage = finish("encode_condition", stage)

        rng = np.random.default_rng(seed)
        part_count = len(part_surfaces)
        latents_np = rng.standard_normal((part_count, *self.shapevae.latent_shape), dtype=np.float32)
        latents = mx.array(latents_np, dtype=mx.bfloat16)
        part_rng = np.random.default_rng(seed)
        sigmas = np.linspace(0, 1, num_inference_steps, dtype=np.float32)
        sigma_next = np.concatenate([sigmas[1:], np.ones(1, dtype=np.float32)])
        diffusion_started = time.perf_counter()
        for index, (sigma, following) in enumerate(zip(sigmas, sigma_next, strict=True)):
            timestep = mx.full((part_count,), sigma, dtype=mx.bfloat16)
            part_indices = part_rng.permutation(self.partformer.valid_num)[:part_count].astype(np.int32)
            velocity = self.partformer(latents, timestep, contexts, part_indices=part_indices)
            latents = (
                latents.astype(mx.float32) + np.float32(following - sigma) * velocity.astype(mx.float32)
            ).astype(mx.bfloat16)
            mx.eval(latents)
            if progress is not None:
                progress(f"diffusion_{index + 1}/{num_inference_steps}", time.perf_counter() - diffusion_started)
        stage = finish("diffusion", stage)

        scene = trimesh.Scene()
        if not output_latents:
            for part_index in range(part_count):
                part_mesh = self.shapevae.decode_to_mesh(
                    latents[part_index : part_index + 1],
                    octree_resolution=octree_resolution,
                    chunk_size=sdf_chunk_size,
                )
                part_mesh.vertices = np.asarray(part_mesh.vertices) * scale + center
                color = rng.integers(32, 256, size=3, dtype=np.uint8)
                part_mesh.visual.face_colors = np.append(color, 255)
                scene.add_geometry(part_mesh, node_name=f"part_{part_index:03d}")
        else:
            scene = np.array(latents.astype(mx.float32))
        finish("decode_parts", stage)
        return XPartResult(
            scene=scene,
            latents=np.array(latents.astype(mx.float32)),
            bboxes=bboxes,
            center=center,
            scale=scale,
            stage_seconds=timings,
        )
