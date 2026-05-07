"""
Minimal Blender background renderer for scene-only reference images.

Run with:
  blender -b /path/to/scene.blend -P scripts/render_scene_background.py -- \
    --output /abs/path/background.png --resolution 1024 --engine EEVEE
"""

import argparse
import json
import os
import sys

import bpy


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render scene background only.")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES"])
    parser.add_argument("--camera-name", default=None, help="Optional explicit camera name")
    parser.add_argument("--meta-path", default=None, help="Optional metadata JSON output path")
    return parser.parse_args(argv)


def resolve_camera(camera_name=None):
    scene = bpy.context.scene
    if camera_name:
        cam = bpy.data.objects.get(camera_name)
        if cam is None or cam.type != "CAMERA":
            raise RuntimeError(f"Camera '{camera_name}' not found or not a CAMERA object")
        scene.camera = cam
        return cam
    if scene.camera is not None:
        return scene.camera
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            scene.camera = obj
            return obj
    raise RuntimeError("No camera found in scene")


def configure_render(resolution, engine):
    scene = bpy.context.scene
    engine_map = {
        "EEVEE": "BLENDER_EEVEE_NEXT",
        "CYCLES": "CYCLES",
        "BLENDER_EEVEE": "BLENDER_EEVEE_NEXT",
        "BLENDER_EEVEE_NEXT": "BLENDER_EEVEE_NEXT",
        "BLENDER_WORKBENCH": "BLENDER_WORKBENCH",
    }
    resolved_engine = engine_map.get(engine, engine)
    scene.render.engine = resolved_engine
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if "EEVEE" in scene.render.engine:
        scene.eevee.taa_render_samples = 64
        scene.eevee.use_gtao = True
        scene.eevee.use_ssr = True


def main():
    args = parse_args()
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cam = resolve_camera(args.camera_name)
    configure_render(args.resolution, args.engine)

    scene = bpy.context.scene
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)

    meta_path = args.meta_path or os.path.splitext(out_path)[0] + "_meta.json"
    meta = {
        "blend_path": bpy.data.filepath,
        "scene_name": scene.name,
        "camera_name": cam.name,
        "engine": scene.render.engine,
        "resolution": args.resolution,
        "output": out_path,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[background] Rendered: {out_path}")
    print(f"[background] Meta: {meta_path}")


if __name__ == "__main__":
    main()
