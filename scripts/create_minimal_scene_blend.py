"""
Create a minimal Blender scene compatible with pipeline/stage4_scene_render.py.

Run with Blender, for example:
  D:\\blender\\blender.exe -b -P scripts\\create_minimal_scene_blend.py -- --output assets\\scene\\4.blend

The generated scene contains:
- a ground mesh named "Plane" so scene_template.json can find it
- a camera named "Camera"
- soft area lighting and a muted world background
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Create a minimal ARIS scene blend")
    parser.add_argument("--output", default="assets/scene/4.blend")
    parser.add_argument("--plane-size", type=float, default=12.0)
    parser.add_argument("--camera-x", type=float, default=3.4)
    parser.add_argument("--camera-y", type=float, default=-5.2)
    parser.add_argument("--camera-z", type=float, default=2.4)
    return parser.parse_args(argv)


def look_at(obj, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def create_ground(size: float):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, 0.0))
    plane = bpy.context.object
    plane.name = "Plane"
    plane.data.name = "PlaneMesh"

    mat = bpy.data.materials.new("Muted Asphalt Ground")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.32, 0.30, 0.27, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.82
        bsdf.inputs["Metallic"].default_value = 0.0
    plane.data.materials.append(mat)
    return plane


def create_camera(x: float, y: float, z: float):
    bpy.ops.object.camera_add(location=(x, y, z))
    camera = bpy.context.object
    camera.name = "Camera"
    camera.data.name = "CameraData"
    camera.data.lens = 45.0
    camera.data.sensor_width = 32.0
    look_at(camera, Vector((0.0, 0.0, 0.45)))
    bpy.context.scene.camera = camera
    return camera


def create_lighting() -> None:
    bpy.ops.object.light_add(type="AREA", location=(-3.0, -4.0, 5.0))
    key = bpy.context.object
    key.name = "Soft_Key_Area"
    key.data.energy = 450.0
    key.data.size = 5.0
    look_at(key, Vector((0.0, 0.0, 0.0)))

    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 4.0))
    sun = bpy.context.object
    sun.name = "Soft_Overcast_Sun"
    sun.data.energy = 0.45
    sun.rotation_euler = (math.radians(45.0), 0.0, math.radians(-35.0))


def configure_world_and_render() -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 0.5

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.60, 0.62, 0.64, 1.0)
        bg.inputs["Strength"].default_value = 0.8


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = Path(str(output) + "@")
    if temp_output.exists():
        temp_output.unlink()

    clear_scene()
    create_ground(args.plane_size)
    create_camera(args.camera_x, args.camera_y, args.camera_z)
    create_lighting()
    configure_world_and_render()

    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    print(f"[minimal-scene] Wrote {output}")


if __name__ == "__main__":
    main()
