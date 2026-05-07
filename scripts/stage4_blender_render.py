"""
Stage 4: 3D-to-Multi-view — Blender EEVEE multi-angle rendering.

This script runs INSIDE Blender's Python interpreter.
Call via:
  /usr/bin/blender -b -P pipeline/stage4_blender_render.py -- \
    --input-dir pipeline/data/meshes/ \
    --output-dir pipeline/data/renders/ \
    --resolution 512 --engine EEVEE [--camera-distance 3.5]

Renders each .glb at azimuth=[0,45,...,315] x elevation=[0] = 8 views per object.
Outputs: renders/{obj_id}/az{N:03d}_el{M:+d}.png + renders/{obj_id}/metadata.json
"""

import bpy
import bmesh
import math
import json
import os
import sys
import argparse


# ── Rendering parameters ──────────────────────────────────────────────────────
AZIMUTHS = list(range(0, 360, 45))        # 8 angles: 0, 45, ..., 315
ELEVATIONS = [0]                          # Horizontal-only dataset
CAMERA_DISTANCE = 3.5                     # Sphere radius (normalized units)
RESOLUTION = 512                          # Square render resolution
ENGINE = "BLENDER_EEVEE"                  # or "CYCLES"
BG_COLOR = (1.0, 1.0, 1.0, 1.0)          # White background (RGBA)


def parse_args():
    """Parse args after '--' separator (Blender passes its own args before --)."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory with .glb files")
    parser.add_argument("--output-dir", required=True, help="Root output directory")
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES"])
    parser.add_argument("--obj-id", default=None, help="Process single object ID only")
    parser.add_argument("--camera-distance", type=float, default=CAMERA_DISTANCE,
                        help="Camera sphere radius in normalized units (default: 3.5)")
    return parser.parse_args(argv)


def reset_scene():
    """Remove all objects, lights, cameras from the current scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    # Remove orphaned meshes/materials
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.lights:
        bpy.data.lights.remove(block)


def setup_render_settings(resolution: int, engine: str):
    """Configure render engine, resolution, output format."""
    scene = bpy.context.scene
    scene.render.engine = f"BLENDER_{engine}" if not engine.startswith("BLENDER_") else engine
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    # White world background
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = BG_COLOR
        bg_node.inputs["Strength"].default_value = 1.0

    # EEVEE-specific settings
    if "EEVEE" in scene.render.engine:
        scene.eevee.taa_render_samples = 64
        scene.eevee.use_gtao = True
        scene.eevee.use_bloom = False
        scene.eevee.use_ssr = True


def import_glb(glb_path: str):
    """Import a .glb file and return the imported objects."""
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=glb_path)
    after = set(bpy.data.objects.keys())
    new_objs = [bpy.data.objects[n] for n in (after - before)]
    return new_objs


def normalize_object_v2(objects):
    """
    Simplified normalization: translate centroid to origin, scale to unit box.
    Works without depsgraph issues in Blender 3.0.1.
    """
    from mathutils import Vector

    if not objects:
        return None

    meshes = [o for o in objects if o.type == "MESH"]
    if not meshes:
        return objects[0]

    # World-space bounding box
    all_corners = []
    for obj in meshes:
        mat = obj.matrix_world
        for corner in obj.bound_box:
            all_corners.append(mat @ Vector(corner))

    min_x = min(v.x for v in all_corners)
    max_x = max(v.x for v in all_corners)
    min_y = min(v.y for v in all_corners)
    max_y = max(v.y for v in all_corners)
    min_z = min(v.z for v in all_corners)
    max_z = max(v.z for v in all_corners)

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    cz = (min_z + max_z) / 2
    extent = max(max_x - min_x, max_y - min_y, max_z - min_z)
    scale = 2.0 / extent if extent > 1e-6 else 1.0

    # Parent empty
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
    parent = bpy.context.active_object
    parent.name = "ModelRoot"

    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

    bpy.context.view_layer.objects.active = parent
    bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)

    # Move each mesh to compensate for centroid
    for obj in objects:
        obj.location.x -= cx
        obj.location.y -= cy
        obj.location.z -= cz

    parent.scale = (scale, scale, scale)
    bpy.ops.object.select_all(action="DESELECT")
    return parent


def add_lights(cs_lighting=None):
    """cs_lighting: dict with key_scale, fill_scale, rim_scale, world_ev, key_shadow_size, rim_shadow_size."""
    from mathutils import Vector
    cs = cs_lighting or {}
    lights_cfg = [
        ("KeyLight",  "SUN",  (4, -4, 6),  3.0 * cs.get("key_scale",  1.0)),
        ("FillLight", "AREA", (-3, 2, 4),  2.0 * cs.get("fill_scale", 1.0)),
        ("BackLight", "SPOT", (0,  5, -2), 1.5 * cs.get("rim_scale",  1.0)),
    ]
    created = []
    for name, ltype, loc, energy in lights_cfg:
        bpy.ops.object.light_add(type=ltype, location=loc)
        light = bpy.context.active_object
        light.name = name
        light.data.energy = energy
        direction = Vector((0, 0, 0)) - Vector(loc)
        rot = direction.to_track_quat("-Z", "Y")
        light.rotation_euler = rot.to_euler()
        created.append(light)
    # Apply world EV offset
    world_ev = cs.get("world_ev", 0.0)
    if world_ev != 0.0:
        scene = bpy.context.scene
        if scene.world and scene.world.node_tree:
            for node in scene.world.node_tree.nodes:
                if node.type == "BACKGROUND":
                    node.inputs[1].default_value *= (2.0 ** world_ev)
    # Apply per-light shadow soft size
    # created[0]=KeyLight, created[1]=FillLight, created[2]=BackLight(rim)
    if len(created) >= 3:
        key_light = created[0]
        rim_light = created[2]
        key_ss = cs.get("key_shadow_size", 1.0)
        rim_ss = cs.get("rim_shadow_size", 1.0)
        if key_ss != 1.0 and hasattr(key_light.data, "shadow_soft_size"):
            key_light.data.shadow_soft_size *= key_ss
        if rim_ss != 1.0 and hasattr(rim_light.data, "shadow_soft_size"):
            rim_light.data.shadow_soft_size *= rim_ss
    return created


def add_camera():
    """Add a camera object at origin (will be positioned per render)."""
    bpy.ops.object.camera_add(location=(0, 0, 0))
    cam = bpy.context.active_object
    cam.name = "RenderCamera"
    bpy.context.scene.camera = cam
    # Set focal length
    cam.data.lens = 50
    cam.data.clip_start = 0.1
    cam.data.clip_end = 100.0
    return cam


def position_camera(cam, azimuth_deg: float, elevation_deg: float, distance: float):
    """
    Place camera on a sphere at given azimuth and elevation, pointing at origin.
    azimuth: horizontal angle (0=front, 90=right, clockwise)
    elevation: vertical angle (0=equator, positive=above)
    """
    from mathutils import Vector

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    # Spherical to Cartesian
    x = distance * math.cos(el) * math.sin(az)
    y = -distance * math.cos(el) * math.cos(az)
    z = distance * math.sin(el)

    cam.location = (x, y, z)

    # Point camera at origin
    direction = Vector((0, 0, 0)) - Vector((x, y, z))
    rot = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot.to_euler()


def render_object(glb_path: str, obj_id: str, output_dir: str,
                  resolution: int, engine: str, camera_distance: float):
    """Full pipeline: reset scene, import mesh, normalize, render all angles."""
    reset_scene()
    setup_render_settings(resolution, engine)

    # Import model
    print(f"  Importing {glb_path} ...")
    imported = import_glb(glb_path)
    if not imported:
        print(f"  [WARN] No objects imported from {glb_path}")
        return []

    # Normalize
    normalize_object_v2(imported)

    # Lights and camera
    add_lights()
    cam = add_camera()

    # Output directory for this object
    obj_out = os.path.join(output_dir, obj_id)
    os.makedirs(obj_out, exist_ok=True)

    rendered_frames = []
    total = len(AZIMUTHS) * len(ELEVATIONS)
    count = 0

    for az in AZIMUTHS:
        for el in ELEVATIONS:
            count += 1
            filename = f"az{az:03d}_el{el:+03d}.png"
            out_path = os.path.join(obj_out, filename)

            if os.path.exists(out_path):
                rendered_frames.append({
                    "filename": filename,
                    "azimuth": az,
                    "elevation": el,
                    "path": out_path,
                })
                continue

            position_camera(cam, az, el, camera_distance)
            bpy.context.scene.render.filepath = out_path
            bpy.ops.render.render(write_still=True)

            print(f"    [{count}/{total}] az={az:3d} el={el:+3d} → {filename}")
            rendered_frames.append({
                "filename": filename,
                "azimuth": az,
                "elevation": el,
                "path": out_path,
            })

    # Save per-object metadata
    meta = {
        "obj_id": obj_id,
        "source_glb": glb_path,
        "resolution": resolution,
        "engine": engine,
        "camera_distance": camera_distance,
        "azimuths": AZIMUTHS,
        "elevations": ELEVATIONS,
        "total_renders": len(rendered_frames),
        "frames": rendered_frames,
    }
    meta_path = os.path.join(obj_out, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [Stage 4] {obj_id}: {len(rendered_frames)} renders → {obj_out}")
    return rendered_frames


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    resolution = args.resolution
    engine = args.engine
    camera_distance = args.camera_distance

    os.makedirs(output_dir, exist_ok=True)

    # Find all .glb files
    glb_files = sorted([
        f for f in os.listdir(input_dir) if f.endswith(".glb")
    ])

    if args.obj_id:
        glb_files = [f for f in glb_files if f.replace(".glb", "") == args.obj_id]

    if not glb_files:
        print(f"[Stage 4] No .glb files found in {input_dir}")
        sys.exit(1)

    print(f"[Stage 4] Rendering {len(glb_files)} objects, {len(AZIMUTHS)}×{len(ELEVATIONS)}=48 views each")
    print(f"[Stage 4] Resolution: {resolution}×{resolution}, Engine: {engine}")
    print(f"[Stage 4] Camera distance: {camera_distance}")

    all_meta = {}
    for fname in glb_files:
        obj_id = fname.replace(".glb", "")
        glb_path = os.path.join(input_dir, fname)
        print(f"\n[Stage 4] === Processing {obj_id} ===")
        frames = render_object(glb_path, obj_id, output_dir, resolution, engine, camera_distance)
        all_meta[obj_id] = {
            "glb": glb_path,
            "frames": len(frames),
        }

    # Summary
    total_renders = sum(v["frames"] for v in all_meta.values())
    summary = {
        "total_objects": len(all_meta),
        "total_renders": total_renders,
        "camera_distance": camera_distance,
        "azimuths": AZIMUTHS,
        "elevations": ELEVATIONS,
        "resolution": resolution,
        "objects": all_meta,
    }
    summary_path = os.path.join(output_dir, "render_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Stage 4] Done: {len(all_meta)} objects, {total_renders} renders total")
    print(f"[Stage 4] Summary → {summary_path}")


if __name__ == "__main__":
    main()
