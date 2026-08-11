from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class ViewRecord:
    index: int
    azimuth_degrees: float
    elevation_degrees: float
    rgb: str
    face_ids: str
    camera_basis: list[list[float]]
    center: list[float]
    half_extent: float


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError("camera basis vector has zero length")
    return vector / length


def _camera_basis(azimuth_degrees: float, elevation_degrees: float) -> np.ndarray:
    azimuth = np.deg2rad(azimuth_degrees)
    elevation = np.deg2rad(elevation_degrees)
    toward_camera = _normalize(
        np.array(
            [
                np.sin(azimuth) * np.cos(elevation),
                np.sin(elevation),
                np.cos(azimuth) * np.cos(elevation),
            ],
            dtype=np.float64,
        )
    )
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = _normalize(np.cross(world_up, toward_camera))
    up = _normalize(np.cross(toward_camera, right))
    return np.stack([right, up, toward_camera], axis=0)


def _base_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    visual = mesh.visual
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    texture = getattr(material, "baseColorTexture", None)
    if uv is None or texture is None:
        face_colors = getattr(visual, "face_colors", None)
        if face_colors is not None and len(face_colors) == len(faces):
            return np.asarray(face_colors, dtype=np.float64)[:, :3]
        return np.repeat(np.array([[184.0, 190.0, 198.0]], dtype=np.float64), len(faces), axis=0)

    image = np.asarray(texture.convert("RGB"), dtype=np.float64)
    centroid_uv = np.mean(np.asarray(uv, dtype=np.float64)[faces], axis=1)
    u = np.mod(centroid_uv[:, 0], 1.0)
    v = np.mod(centroid_uv[:, 1], 1.0)
    x = np.clip(np.rint(u * (image.shape[1] - 1)).astype(np.int64), 0, image.shape[1] - 1)
    y = np.clip(np.rint((1.0 - v) * (image.shape[0] - 1)).astype(np.int64), 0, image.shape[0] - 1)
    return image[y, x]


def _render_view(
    mesh: trimesh.Trimesh,
    basis: np.ndarray,
    center: np.ndarray,
    half_extent: float,
    resolution: int,
    face_colors: np.ndarray,
) -> tuple[Image.Image, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    viewed = (vertices - center) @ basis.T
    scale = (resolution - 1) / (2.0 * half_extent)
    screen = np.column_stack(
        (
            (viewed[:, 0] + half_extent) * scale,
            (half_extent - viewed[:, 1]) * scale,
            viewed[:, 2],
        )
    )
    triangles = screen[faces]
    visible = (
        (triangles[:, :, 0].max(axis=1) >= 0)
        & (triangles[:, :, 0].min(axis=1) < resolution)
        & (triangles[:, :, 1].max(axis=1) >= 0)
        & (triangles[:, :, 1].min(axis=1) < resolution)
    )

    face_vertices = viewed[faces]
    normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    shade = 0.58 + 0.42 * np.abs(normals @ np.array([-0.25, 0.35, 0.9]))
    colors = np.clip(face_colors * shade[:, None], 0.0, 255.0).astype(np.uint8)

    rgb = Image.new("RGB", (resolution, resolution), (245, 245, 245))
    face_image = Image.new("I", (resolution, resolution), 0)
    rgb_draw = ImageDraw.Draw(rgb)
    face_draw = ImageDraw.Draw(face_image)
    depth_order = np.argsort(np.mean(triangles[:, :, 2], axis=1))
    for face_id in depth_order[visible[depth_order]]:
        polygon = [tuple(point) for point in triangles[face_id, :, :2]]
        rgb_draw.polygon(polygon, fill=tuple(int(value) for value in colors[face_id]))
        face_draw.polygon(polygon, fill=int(face_id) + 1)
    face_ids = np.asarray(face_image, dtype=np.int32) - 1
    return rgb, face_ids


def render_views(
    mesh: trimesh.Trimesh,
    output_dir: Path,
    *,
    view_count: int = 12,
    resolution: int = 512,
    elevation_degrees: float = 18.0,
) -> list[ViewRecord]:
    if view_count < 1:
        raise ValueError("view_count must be positive")
    if resolution < 32:
        raise ValueError("resolution must be at least 32")
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    half_extent = float(np.max(bounds[1] - bounds[0]) * 0.56)
    if half_extent <= 0:
        raise ValueError("mesh extent must be positive")
    face_colors = _base_colors(mesh)

    records: list[ViewRecord] = []
    for index in range(view_count):
        azimuth = index * 360.0 / view_count
        elevation = elevation_degrees if index % 2 == 0 else -elevation_degrees * 0.5
        basis = _camera_basis(azimuth, elevation)
        rgb, face_ids = _render_view(mesh, basis, center, half_extent, resolution, face_colors)
        rgb_name = f"view_{index:02d}.png"
        face_name = f"view_{index:02d}_face_ids.npy"
        rgb.save(output_dir / rgb_name)
        np.save(output_dir / face_name, face_ids, allow_pickle=False)
        records.append(
            ViewRecord(
                index=index,
                azimuth_degrees=azimuth,
                elevation_degrees=elevation,
                rgb=rgb_name,
                face_ids=face_name,
                camera_basis=basis.tolist(),
                center=center.tolist(),
                half_extent=half_extent,
            )
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "resolution": resolution,
        "view_count": view_count,
        "views": [asdict(record) for record in records],
    }
    (output_dir / "views.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return records


def render_mesh_color(
    mesh: trimesh.Trimesh,
    output_path: Path,
    *,
    resolution: int = 768,
    azimuth_degrees: float = 25.0,
    elevation_degrees: float = 18.0,
) -> None:
    """Render one isolated mesh using its source texture or vertex/material color."""
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    half_extent = float(np.max(bounds[1] - bounds[0]) * 0.56)
    if half_extent <= 0:
        raise ValueError("mesh extent must be positive")
    basis = _camera_basis(azimuth_degrees, elevation_degrees)
    image, _ = _render_view(mesh, basis, center, half_extent, resolution, _base_colors(mesh))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
