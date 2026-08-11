from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Color a mesh from per-face integer labels")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh, force="mesh")
    labels = np.load(args.labels)
    if labels.shape != (len(mesh.faces),):
        raise ValueError(f"labels must have shape {(len(mesh.faces),)}, got {labels.shape}")
    unique = [int(label) for label in np.unique(labels) if label >= 0]
    rng = np.random.default_rng(args.seed)
    palette = {label: rng.integers(32, 256, size=3, dtype=np.uint8) for label in unique}
    face_colors = np.zeros((len(labels), 4), dtype=np.uint8)
    face_colors[:, 3] = 255
    for label, color in palette.items():
        face_colors[labels == label, :3] = color
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=face_colors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)


if __name__ == "__main__":
    main()
