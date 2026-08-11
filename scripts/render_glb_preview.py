"""Render a deterministic vertex-color GLB preview with Blender.

Usage:
    blender --background --python scripts/render_glb_preview.py -- input.glb output.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
from mathutils import Vector


def main() -> None:
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    if len(arguments) != 2:
        raise SystemExit("expected input.glb and output.png after --")
    input_path = Path(arguments[0]).resolve()
    output_path = Path(arguments[1]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(input_path))
    mesh_objects = [item for item in bpy.context.scene.objects if item.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"no mesh objects imported from {input_path}")

    corners = [item.matrix_world @ Vector(corner) for item in mesh_objects for corner in item.bound_box]
    minimum = Vector(min(point[index] for point in corners) for index in range(3))
    maximum = Vector(max(point[index] for point in corners) for index in range(3))
    center = (minimum + maximum) * 0.5
    radius = max((maximum - minimum).length * 0.5, 1e-3)

    camera_data = bpy.data.cameras.new("Preview Camera")
    camera = bpy.data.objects.new("Preview Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    direction = Vector((1.45, -1.8, 1.15)).normalized()
    camera.location = center + direction * radius * 2.8
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.lens = 55
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.studio_light = "paint.sl"
    scene.display.shading.color_type = "VERTEX"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "BOTH"
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    print(f"rendered {len(mesh_objects)} mesh object(s) to {output_path}")


if __name__ == "__main__":
    main()
