"""Render a deterministic CPU-only point preview of a colored GLB.

This is a lightweight fallback for machines without Blender.  It samples the
mesh surface, projects the samples through an orthographic camera, and uses a
small z-buffered point splat to preserve the mesh's segmentation colors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def _face_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    visual = mesh.visual
    if not hasattr(visual, "face_colors"):
        visual = visual.to_color()
    colors = np.asarray(visual.face_colors, dtype=np.float32)[:, :3]
    return colors / 255.0


def render_preview(
    input_path: Path,
    output_path: Path,
    *,
    width: int = 1200,
    height: int = 800,
    sample_count: int = 500_000,
    seed: int = 42,
) -> None:
    loaded = trimesh.load(input_path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"no triangle mesh found in {input_path}")

    points, face_indices = trimesh.sample.sample_surface(
        loaded,
        sample_count,
        seed=seed,
    )
    colors = _face_colors(loaded)[face_indices]
    normals = np.asarray(loaded.face_normals, dtype=np.float64)[face_indices]

    view = np.asarray((1.45, -1.8, 1.15), dtype=np.float64)
    view /= np.linalg.norm(view)
    right = np.cross(np.asarray((0.0, 0.0, 1.0)), view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    projected_x = points @ right
    projected_y = points @ up
    depth = points @ view

    x_min, x_max = np.quantile(projected_x, (0.001, 0.999))
    y_min, y_max = np.quantile(projected_y, (0.001, 0.999))
    scale = min(
        width * 0.88 / max(x_max - x_min, 1e-9),
        height * 0.88 / max(y_max - y_min, 1e-9),
    )
    center_x = (x_min + x_max) * 0.5
    center_y = (y_min + y_max) * 0.5
    pixels_x = np.rint((projected_x - center_x) * scale + width * 0.5).astype(np.int32)
    pixels_y = np.rint(height * 0.5 - (projected_y - center_y) * scale).astype(np.int32)

    light = np.asarray((0.35, -0.55, 0.76), dtype=np.float64)
    light /= np.linalg.norm(light)
    intensity = 0.55 + 0.45 * np.abs(normals @ light)
    shaded = np.clip(colors * intensity[:, None], 0.0, 1.0)
    rgb = np.rint(shaded * 255.0).astype(np.uint8)

    offsets = np.asarray(
        [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)],
        dtype=np.int32,
    )
    splat_x = (pixels_x[:, None] + offsets[None, :, 0]).reshape(-1)
    splat_y = (pixels_y[:, None] + offsets[None, :, 1]).reshape(-1)
    splat_depth = np.repeat(depth, len(offsets))
    splat_rgb = np.repeat(rgb, len(offsets), axis=0)
    inside = (splat_x >= 0) & (splat_x < width) & (splat_y >= 0) & (splat_y < height)
    splat_x = splat_x[inside]
    splat_y = splat_y[inside]
    splat_depth = splat_depth[inside]
    splat_rgb = splat_rgb[inside]

    flat = splat_y.astype(np.int64) * width + splat_x
    z_buffer = np.full(width * height, -np.inf, dtype=np.float64)
    np.maximum.at(z_buffer, flat, splat_depth)
    visible = splat_depth >= z_buffer[flat] - 1e-12

    canvas = np.full((height * width, 3), 242, dtype=np.uint8)
    canvas[flat[visible]] = splat_rgb[visible]
    image = canvas.reshape(height, width, 3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--samples", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    render_preview(
        args.input,
        args.output,
        width=args.width,
        height=args.height,
        sample_count=args.samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
