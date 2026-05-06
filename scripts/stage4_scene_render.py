"""
Stage 4 Scene: insert reconstructed mesh into an existing Blender scene and
render fixed QC views for scene-aware review.
"""

import sys as _sys

if _sys.version_info[:2] == (3, 10):
    _np310_path = "/home/wuwenzhuo/.config/blender/3.0/scripts/addons/modules"
    if _np310_path not in _sys.path:
        _sys.path.insert(0, _np310_path)
    _sys.path = [p for p in _sys.path if "python3.11" not in p]

import numpy as _np_compat

for _alias in ("bool", "int", "float", "complex", "object", "str"):
    if not hasattr(_np_compat, _alias):
        setattr(_np_compat, _alias, eval(_alias))

del _np_compat, _alias

import argparse
import json
import math
import os
import sys
from pathlib import Path
from statistics import median

import bpy
import numpy as np
from mathutils import Matrix, Vector


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_BUILD_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_TEMPLATE_PATH = os.path.join(DATA_BUILD_ROOT, "configs", "scene_template.json")
DEFAULT_QC_VIEWS = [(0, 0), (90, 0), (180, 15), (45, 30)]
DEFAULT_RESOLUTION = 512
DEFAULT_ENGINE = "EEVEE"


def resolve_path(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(DATA_BUILD_ROOT, path_str)


def load_json(path: str, default=None):
    if not path or not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    p = argparse.ArgumentParser(description="Scene-aware Blender render")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--obj-id", default=None)
    p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    p.add_argument("--engine", default=DEFAULT_ENGINE, choices=["EEVEE", "CYCLES"])
    p.add_argument("--control-state", default=None)
    p.add_argument("--scene-template", default=DEFAULT_TEMPLATE_PATH)
    return p.parse_args(argv)


def load_scene_template(path: str) -> dict:
    data = load_json(path, default={}) or {}
    data.setdefault("blend_path", "/home/wuwenzhuo/blender/data/sence/4.blend")
    data.setdefault("use_existing_world", True)
    data.setdefault("disable_existing_lights", False)  # changed: default False
    data.setdefault("add_key_light", False)             # new: no extra key light by default
    data.setdefault("scale_existing_lights", 1.0)       # keep the authored scene look by default
    data.setdefault("scale_world_background", 0.24)
    data.setdefault("fallback_hdri_path", "assets/hdri/neutral_studio_01_2k.exr")
    data.setdefault("support_plane_mode", "ground_object_raycast")  # changed default
    data.setdefault("support_object_name", "ground")    # new: name hint for ground search
    data.setdefault("target_bbox_height_range", [0.3, 1.5])
    data.setdefault("qc_views", [[0, 0]])
    data.setdefault("camera_mode", "scene_camera_match")
    data.setdefault("render_engine", "CYCLES")
    data.setdefault("render_resolution", 1024)
    data.setdefault("cycles_samples", 512)
    data.setdefault("cycles_device", "GPU")
    data.setdefault("cycles_compute_device_type", "CUDA")
    data.setdefault("cycles_preview_samples", 32)
    data.setdefault("filmic_gamma", 0.5)
    data.setdefault("target_brightness", 0.36)
    data.setdefault("adaptive_brightness_damping", 0.45)
    data.setdefault("adaptive_brightness_min_gain", 0.82)
    data.setdefault("adaptive_brightness_max_gain", 1.18)
    data.setdefault("adaptive_world_gain_scale", 0.55)
    data.setdefault("camera_radius_ratio", 0.02)         # new
    data.setdefault("camera_height_ratio", 0.01)         # new
    data.setdefault("camera_object_span_multiplier", 2.8)
    data.setdefault("camera_object_diag_multiplier", 2.2)
    data.setdefault("camera_distance_min", 2.0)
    data.setdefault("camera_distance_max", 8.0)
    data.setdefault("camera_target_height_ratio", 0.45)
    data.setdefault("camera_anchor_u", 0.5)
    data.setdefault("camera_anchor_v", 0.22)
    data.setdefault("camera_distance", 3.5)
    data.setdefault("key_light_xyz", [3.5, -3.5, 5.0])
    data.setdefault("key_light_energy", 3.0)
    data.setdefault("key_light_yaw_deg", 0.0)
    data.setdefault("add_camera_fill_light", True)
    data.setdefault("camera_fill_energy", 420.0)
    data.setdefault("camera_fill_size_multiplier", 4.0)
    data.setdefault("camera_fill_forward_ratio", 0.08)
    data.setdefault("camera_fill_up_ratio", 0.3)
    data.setdefault("camera_fill_right_ratio", 0.18)
    data.setdefault("camera_fill_color", [1.0, 0.985, 0.97])
    data.setdefault("camera_fill_scene_match_kicker", True)
    data.setdefault("contact_shadow_distance", 0.04)
    data.setdefault("ground_contact_epsilon", 0.0014)
    data.setdefault("fallback_support_plane_size", 10.0)
    data.setdefault("adaptive_brightness", True)
    data.setdefault("object_scale_mode", "height_range")
    data.setdefault("preserve_blend_asset_materials", True)
    return data


def setup_render_settings(scene, resolution: int, engine: str, template: dict | None = None):
    template = template or {}
    eevee_name = "BLENDER_EEVEE_NEXT" if int(bpy.app.version[0]) >= 4 else "BLENDER_EEVEE"
    scene.render.engine = eevee_name if engine == "EEVEE" else "CYCLES"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    if "EEVEE" in scene.render.engine:
        scene.eevee.taa_render_samples = 64
        scene.eevee.use_gtao = True
        if hasattr(scene.eevee, "gtao_factor"):
            scene.eevee.gtao_factor = 1.0
        scene.eevee.use_bloom = False
        scene.eevee.use_ssr = True
        scene.eevee.shadow_cube_size = "1024"
        scene.eevee.shadow_cascade_size = "2048"

    if "CYCLES" in scene.render.engine:
        scene.cycles.samples = int(template.get("cycles_samples", 512))
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.01
        scene.cycles.use_denoising = True
        try:
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
        except TypeError:
            try:
                scene.cycles.denoiser = "NLM"
            except TypeError:
                pass
        cycles_device = str(template.get("cycles_device", "GPU")).upper()
        compute_device_type = str(template.get("cycles_compute_device_type", "CUDA")).upper()
        scene.cycles.device = "CPU" if cycles_device == "CPU" else "GPU"
        scene.cycles.tile_size = 512
        if hasattr(scene.cycles, "use_preview_denoising"):
            scene.cycles.use_preview_denoising = True
        if cycles_device != "CPU":
            try:
                prefs = bpy.context.preferences.addons['cycles'].preferences
                prefs.compute_device_type = compute_device_type
                prefs.get_devices()
                for d in prefs.devices:
                    d.use = (d.type == compute_device_type)
            except Exception:
                pass

    try:
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "None"
    except TypeError:
        scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = float(template.get("filmic_gamma", 0.5 if "CYCLES" in scene.render.engine else 1.0))


def import_glb(glb_path: str):
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=glb_path)
    after = set(bpy.data.objects.keys())
    return [bpy.data.objects[name] for name in (after - before)]


def import_blend_objects(blend_path: str):
    imported = []
    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects
    for obj in data_to.objects:
        if obj:
            bpy.context.collection.objects.link(obj)
            imported.append(obj)
    return imported


def choose_primary_imported_object(imported_objects):
    visible = [obj for obj in imported_objects if not obj.hide_get()]
    if not visible:
        return imported_objects[0] if imported_objects else None
    mesh_first = next((obj for obj in visible if obj.type == "MESH"), None)
    return mesh_first or visible[0]


def collect_imported_meshes(root_obj, imported_name_set):
    if root_obj is None:
        return []
    stack = [root_obj]
    visited = set()
    meshes = []
    while stack:
        obj = stack.pop()
        if obj.name in visited:
            continue
        visited.add(obj.name)
        if obj.type == "MESH" and not obj.hide_get():
            meshes.append(obj)
        for child in obj.children:
            if child.name in imported_name_set:
                stack.append(child)
    return meshes


def import_asset(asset_path: str):
    ext = Path(asset_path).suffix.lower()
    if ext == ".glb":
        imported = import_glb(asset_path)
        mesh_objects = [obj for obj in imported if obj.type == "MESH"]
        if not mesh_objects:
            raise RuntimeError(f"No mesh objects imported from {asset_path}")
        root = normalize_object_v2(imported)
        imported_names = {obj.name for obj in mesh_objects}
        return {
            "asset_kind": "glb",
            "imported": imported,
            "mesh_objects": mesh_objects,
            "root": root,
            "imported_names": imported_names,
            "primary_object": root,
        }

    if ext == ".blend":
        imported = import_blend_objects(asset_path)
        if not imported:
            raise RuntimeError(f"No objects imported from {asset_path}")
        imported_names = {obj.name for obj in imported}
        primary = choose_primary_imported_object(imported)
        mesh_objects = collect_imported_meshes(primary, imported_names)
        if not mesh_objects:
            mesh_objects = [obj for obj in imported if obj.type == "MESH" and not obj.hide_get()]
        if not mesh_objects:
            raise RuntimeError(f"No visible mesh objects imported from {asset_path}")
        root = primary or mesh_objects[0]
        return {
            "asset_kind": "blend",
            "imported": imported,
            "mesh_objects": mesh_objects,
            "root": root,
            "imported_names": imported_names,
            "primary_object": primary,
        }

    raise ValueError(f"Unsupported asset type: {asset_path}")


def percentile_bbox(objects, q_lo: float = 1.0, q_hi: float = 99.0):
    pts = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        mat = obj.matrix_world
        for v in obj.data.vertices:
            w = mat @ v.co
            pts.append([w.x, w.y, w.z])
    if not pts:
        return None
    pts = np.array(pts)
    lo = np.percentile(pts, q_lo, axis=0)
    hi = np.percentile(pts, q_hi, axis=0)
    center = (lo + hi) / 2.0
    size = hi - lo
    return {
        "min": lo.tolist(),
        "max": hi.tolist(),
        "center": center.tolist(),
        "size": size.tolist(),
    }


def compute_ground_contact_epsilon(objects, template: dict) -> float:
    base_epsilon = float(template.get("ground_contact_epsilon", 0.0014))
    bbox = percentile_bbox(objects)
    if bbox is None:
        return max(0.0005, min(0.0025, base_epsilon))
    bbox_height = max(float(bbox["size"][2]), 1e-3)
    adaptive_epsilon = bbox_height * 0.0018
    return max(0.0005, min(0.0025, max(base_epsilon * 0.55, adaptive_epsilon)))


def stabilize_ground_contact(root, objects, support_z: float, ground_contact_epsilon: float) -> dict:
    bbox = percentile_bbox(objects)
    if bbox is None:
        return {
            "applied": False,
            "correction_z": 0.0,
            "min_allowed_z": None,
            "max_allowed_z": None,
        }
    bbox_height = max(float(bbox["size"][2]), 1e-3)
    min_allowed_z = support_z - min(0.0008, bbox_height * 0.0015)
    max_allowed_z = support_z + max(ground_contact_epsilon * 1.15, min(0.0028, bbox_height * 0.0028))
    current_min_z = float(bbox["min"][2])
    correction_z = 0.0
    if current_min_z < min_allowed_z:
        correction_z = min_allowed_z - current_min_z
    elif current_min_z > max_allowed_z:
        correction_z = max_allowed_z - current_min_z
    if abs(correction_z) > 1e-6:
        root.location.z += correction_z
        bpy.context.view_layer.update()
    return {
        "applied": abs(correction_z) > 1e-6,
        "correction_z": round(float(correction_z), 6),
        "min_allowed_z": round(float(min_allowed_z), 6),
        "max_allowed_z": round(float(max_allowed_z), 6),
    }


def normalize_object_v2(objects):
    if not objects:
        return None
    mesh_objs = [o for o in objects if o.type == "MESH"]
    if not mesh_objs:
        return objects[0]

    bbox = percentile_bbox(mesh_objs, 1.0, 99.0)
    if bbox is None:
        return objects[0]

    cx, cy, cz = bbox["center"]
    max_ext = max(bbox["size"])
    scale = 2.0 / max_ext if max_ext > 1e-6 else 1.0

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
    parent = bpy.context.active_object
    parent.name = "SceneModelRoot"

    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
    bpy.context.view_layer.objects.active = parent
    bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)

    for obj in objects:
        obj.location.x -= cx
        obj.location.y -= cy
        obj.location.z -= cz

    parent.scale = (scale, scale, scale)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.update()
    return parent


# ---------------------------------------------------------------------------
# Ground detection helpers (mirrors render_all.py)
# ---------------------------------------------------------------------------

def find_ground_object(support_name_hint: str = "ground"):
    """Find ground object: first by name match, then lowest mostly-horizontal bbox.

    The fallback filters out near-vertical objects (size_z > 0.5 * max_xy)
    to avoid picking walls or underground geometry instead of the road surface.
    """
    candidates = [
        obj for obj in bpy.data.objects
        if obj.type == "MESH" and not obj.hide_get()
    ]
    # 1. Exact name match (case-insensitive)
    for obj in candidates:
        if obj.name.lower() == support_name_hint.lower():
            return obj, "name_match"
    # 2. Fallback: lowest bbox Z among mostly-horizontal meshes
    lowest_z = float("inf")
    ground_obj = None
    for obj in candidates:
        bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        xs = [v.x for v in bbox]
        ys = [v.y for v in bbox]
        zs = [v.z for v in bbox]
        size_x = max(xs) - min(xs)
        size_y = max(ys) - min(ys)
        size_z = max(zs) - min(zs)
        max_xy = max(size_x, size_y, 0.001)
        # Skip near-vertical objects (walls, fences, etc.)
        if size_z > 0.5 * max_xy:
            continue
        z = min(zs)
        if z < lowest_z:
            lowest_z = z
            ground_obj = obj
    if ground_obj is not None:
        return ground_obj, "lowest_horizontal_bbox"
    # 3. Last resort: truly lowest bbox (any orientation)
    for obj in candidates:
        bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        z = min(v.z for v in bbox)
        if z < lowest_z:
            lowest_z = z
            ground_obj = obj
    return ground_obj, "lowest_bbox"


def fix_ground_orientation(ground_obj):
    """Fix negative scale.z or abnormal X rotation (mirrors render_all.py:137-144)."""
    if ground_obj.scale.z < 0:
        ground_obj.scale.z = abs(ground_obj.scale.z)
        print("[scene] Fixed negative ground scale.z")
    rx = math.degrees(ground_obj.rotation_euler.x)
    if abs(rx) in [90, 180, 270]:
        ground_obj.rotation_euler.x = 0.0
        print("[scene] Fixed abnormal ground rotation.x")


def adjust_object_to_ground_with_ray(root, mesh_objects, ground_obj):
    """Precision-land root via multiple downward rays against the detected ground."""
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    pts = []
    for obj in mesh_objects:
        mat = obj.matrix_world
        for corner in obj.bound_box:
            pts.append(mat @ Vector(corner))
    if not pts:
        return {
            "success": False,
            "sample_count": 0,
            "hit_count": 0,
            "support_z": None,
        }

    min_z = min(v.z for v in pts)
    max_z = max(v.z for v in pts)
    center_x = sum(v.x for v in pts) / len(pts)
    center_y = sum(v.y for v in pts) / len(pts)

    xs = [v.x for v in pts]
    ys = [v.y for v in pts]
    x_lo = float(np.percentile(xs, 18.0))
    x_hi = float(np.percentile(xs, 82.0))
    y_lo = float(np.percentile(ys, 18.0))
    y_hi = float(np.percentile(ys, 82.0))
    sample_xy = [
        (center_x, center_y),
        (x_lo, y_lo),
        (x_lo, y_hi),
        (x_hi, y_lo),
        (x_hi, y_hi),
    ]
    ray_origin_z = min_z + max(0.01, (max_z - min_z) * 0.06)
    ray_dir = Vector((0, 0, -1))
    hit_locations = []

    for sx, sy in sample_xy:
        ray_origin = Vector((sx, sy, ray_origin_z))
        hit, location, _, _, hit_obj, _ = scene.ray_cast(depsgraph, ray_origin, ray_dir, distance=2.0)
        if hit and hit_obj == ground_obj:
            hit_locations.append(float(location.z))

    if not hit_locations:
        print("[scene] Ray cast missed ground for all samples, skipping")
        return {
            "success": False,
            "sample_count": len(sample_xy),
            "hit_count": 0,
            "support_z": None,
        }

    hit_locations.sort()
    pick_idx = min(len(hit_locations) - 1, max(0, math.ceil(len(hit_locations) * 0.75) - 1))
    support_z = hit_locations[pick_idx]
    delta_z = support_z - min_z
    root.location.z += delta_z
    bpy.context.view_layer.update()
    print(
        f"[scene] Ray hit ground with {len(hit_locations)}/{len(sample_xy)} samples, "
        f"support_z={support_z:.4f}, adjusted by {delta_z:.4f}"
    )
    return {
        "success": True,
        "sample_count": len(sample_xy),
        "hit_count": len(hit_locations),
        "support_z": float(support_z),
    }


def _configure_contact_shadow(light_obj, template: dict, control_scene: dict):
    """Make contact-shadow control actually affect real scene lights.

    The old code treated `contact_shadow_strength` like a distance multiplier,
    which often widened the contact shadow footprint and visually worsened
    grounding. Here we reinterpret larger strength as a *tighter* shadow
    search distance plus slightly lower bias.
    """
    if light_obj is None or getattr(light_obj, "type", None) != "LIGHT":
        return
    light_data = light_obj.data
    if light_data is None:
        return
    if hasattr(light_data, "use_contact_shadow"):
        light_data.use_contact_shadow = True
    strength_scale = max(0.6, float(control_scene.get("contact_shadow_strength", 1.0)))
    base_distance = float(template.get("contact_shadow_distance", 0.04))
    tuned_distance = base_distance / (strength_scale ** 0.85)
    if hasattr(light_data, "contact_shadow_distance"):
        light_data.contact_shadow_distance = max(0.002, min(base_distance * 1.05, tuned_distance))
    if hasattr(light_data, "contact_shadow_bias"):
        light_data.contact_shadow_bias = max(0.0005, min(0.003, 0.0022 / strength_scale))
    if hasattr(light_data, "contact_shadow_thickness"):
        light_data.contact_shadow_thickness = max(0.0008, min(0.018, 0.006 * strength_scale))


def configure_existing_scene_lights(template: dict, control_scene: dict):
    for obj in bpy.data.objects:
        if obj.type == "LIGHT" and not obj.hide_get():
            _configure_contact_shadow(obj, template, control_scene)


# ---------------------------------------------------------------------------
# Light helpers
# ---------------------------------------------------------------------------

def scale_existing_lights(factor: float):
    """Scale all scene light energies by factor (mirrors render_all.py:335-338)."""
    for light in bpy.data.lights:
        light.energy *= factor


def scale_world_background(scene, factor: float):
    """Scale world background strength by factor (mirrors render_all.py:337-340)."""
    if scene.world and scene.world.use_nodes:
        for node in scene.world.node_tree.nodes:
            if node.type == "BACKGROUND":
                node.inputs[1].default_value *= factor


def adjust_lighting_to_target_brightness(target_brightness: float = 0.36,
                                          preview_res: int = 128,
                                          preview_samples: int = 32,
                                          damping: float = 0.45,
                                          min_gain: float = 0.82,
                                          max_gain: float = 1.18,
                                          world_gain_scale: float = 0.55):
    """Adaptive brightness: render a quick preview, measure brightness, rescale lights.

    Ported from render_all.py:78-120. Requires PIL. Falls back silently if not available.
    """
    try:
        from PIL import Image
        import tempfile
    except ImportError:
        print("[scene] PIL not available, skipping adaptive brightness")
        return

    scene = bpy.context.scene
    # Save render settings
    orig_res_x = scene.render.resolution_x
    orig_res_y = scene.render.resolution_y
    orig_filepath = scene.render.filepath
    orig_samples = None
    if "CYCLES" in scene.render.engine:
        orig_samples = scene.cycles.samples

    # Quick low-res preview
    scene.render.resolution_x = preview_res
    scene.render.resolution_y = int(preview_res * 9 / 16) or preview_res
    if orig_samples is not None:
        scene.cycles.samples = preview_samples
    elif "EEVEE" in scene.render.engine:
        try:
            scene.eevee.taa_render_samples = preview_samples
        except Exception:
            pass

    with tempfile.TemporaryDirectory() as tmp_dir:
        preview_path = os.path.join(tmp_dir, "brightness_preview.png")
        scene.render.filepath = preview_path
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as e:
            print(f"[scene] Adaptive brightness preview render failed: {e}")
            # Restore
            scene.render.resolution_x = orig_res_x
            scene.render.resolution_y = orig_res_y
            scene.render.filepath = orig_filepath
            if orig_samples is not None:
                scene.cycles.samples = orig_samples
            return

        if os.path.exists(preview_path):
            try:
                img = Image.open(preview_path).convert("RGB")
                import numpy as _np_ab
                arr = _np_ab.array(img, dtype=float)
                actual = arr.mean() / 255.0
                if actual > 1e-6:
                    raw_ratio = target_brightness / actual
                    ratio = 1.0 + (raw_ratio - 1.0) * damping
                    ratio = max(min_gain, min(max_gain, ratio))
                    world_ratio = 1.0 + (ratio - 1.0) * world_gain_scale
                    print(f"[scene] Adaptive brightness: actual={actual:.3f}, "
                          f"target={target_brightness:.3f}, raw_ratio={raw_ratio:.3f}, "
                          f"ratio={ratio:.3f}, world_ratio={world_ratio:.3f}")
                    for light in bpy.data.lights:
                        light.energy *= ratio
                    world = scene.world
                    if world and world.use_nodes:
                        for node in world.node_tree.nodes:
                            if node.type == "BACKGROUND":
                                node.inputs[1].default_value *= world_ratio
                else:
                    print("[scene] Adaptive brightness: preview too dark, skipping")
            except Exception as e:
                print(f"[scene] Adaptive brightness measurement failed: {e}")

    # Restore render settings
    scene.render.resolution_x = orig_res_x
    scene.render.resolution_y = orig_res_y
    scene.render.filepath = orig_filepath
    if orig_samples is not None:
        scene.cycles.samples = orig_samples


def disable_existing_lights():
    """Legacy: hide all lights completely. Kept for backward compat."""
    disabled = []
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            obj.hide_render = True
            obj.hide_viewport = True
            disabled.append(obj.name)
    return disabled


# ---------------------------------------------------------------------------
# Legacy support plane detection (auto_detect_lowest_horizontal_mesh mode)
# ---------------------------------------------------------------------------

def detect_support_plane(ignore_names=None):
    ignore_names = set(ignore_names or [])
    candidates = []
    for obj in bpy.data.objects:
        if obj.name in ignore_names or obj.type != "MESH":
            continue
        if obj.hide_render:
            continue
        mesh = obj.data
        if not getattr(mesh, "polygons", None):
            continue

        mat = obj.matrix_world
        rot = mat.to_3x3().normalized()
        z_values = []
        xs = []
        ys = []
        for poly in mesh.polygons:
            normal_w = (rot @ poly.normal).normalized()
            if normal_w.z < 0.92:
                continue
            center = mat @ poly.center
            z_values.append(center.z)
            for vid in poly.vertices:
                wv = mat @ mesh.vertices[vid].co
                xs.append(wv.x)
                ys.append(wv.y)
        if not z_values or len(xs) < 3:
            continue
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        area = max(0.0, (x_max - x_min) * (y_max - y_min))
        if area <= 1e-6:
            continue
        candidates.append({
            "name": obj.name,
            "z": float(median(z_values)),
            "bounds": [float(x_min), float(x_max), float(y_min), float(y_max)],
            "area": float(area),
            "created": False,
        })

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item["area"], item["z"]))
    chosen = candidates[0]
    chosen["center_xy"] = [
        (chosen["bounds"][0] + chosen["bounds"][1]) / 2.0,
        (chosen["bounds"][2] + chosen["bounds"][3]) / 2.0,
    ]
    return chosen


def create_hidden_support_plane(size: float = 10.0):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "V7SupportPlane"
    plane.hide_render = True
    plane.hide_viewport = True
    half = size / 2.0
    return {
        "name": plane.name,
        "z": 0.0,
        "bounds": [-half, half, -half, half],
        "center_xy": [0.0, 0.0],
        "area": float(size * size),
        "created": True,
    }


def ensure_support_plane(template: dict, ignore_names=None):
    if template.get("support_plane_mode") == "auto_detect_lowest_horizontal_mesh":
        support = detect_support_plane(ignore_names=ignore_names)
        if support:
            return support
    return create_hidden_support_plane(float(template.get("fallback_support_plane_size", 10.0)))


# ---------------------------------------------------------------------------
# World environment
# ---------------------------------------------------------------------------

def ensure_world_environment(scene, template: dict, control_scene: dict):
    """Setup world environment.

    If use_existing_world=True (default for 4.blend), preserve the blend's own
    world nodes entirely — no new nodes, no new links, no HDRI load.
    This is the key fix: the old code always rebuilt the node tree, destroying
    the street-scene sky/environment that 4.blend carries.
    """
    if template.get("use_existing_world", True):
        world = scene.world
        env = None
        if world and world.use_nodes:
            env = next((n for n in world.node_tree.nodes if n.type == "TEX_ENVIRONMENT"), None)
        return {
            "has_environment_texture": env is not None,
            "fallback_hdri_used": False,
            "preserved_existing": True,
        }

    # use_existing_world=False: rebuild world nodes (fallback for non-4.blend scenes)
    if scene.world is None:
        scene.world = bpy.data.worlds.new("V7World")
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    bg = next((n for n in nodes if n.type == "BACKGROUND"), None)
    if bg is None:
        bg = nodes.new("ShaderNodeBackground")
    output = next((n for n in nodes if n.type == "OUTPUT_WORLD"), None)
    if output is None:
        output = nodes.new("ShaderNodeOutputWorld")
    if not bg.outputs["Background"].is_linked:
        links.new(bg.outputs["Background"], output.inputs["Surface"])

    env = next((n for n in nodes if n.type == "TEX_ENVIRONMENT"), None)
    fallback_used = False
    fallback_path = resolve_path(template.get("fallback_hdri_path", ""))
    if env is None and os.path.exists(fallback_path):
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(fallback_path, check_existing=True)
        fallback_used = True
    elif env is None:
        print(f"[scene] No environment texture found and fallback missing: {fallback_path}")

    mapping = None
    if env is not None:
        texcoord = next((n for n in nodes if n.type == "TEX_COORD"), None)
        if texcoord is None:
            texcoord = nodes.new("ShaderNodeTexCoord")
        if env.inputs["Vector"].is_linked and env.inputs["Vector"].links[0].from_node.type == "MAPPING":
            mapping = env.inputs["Vector"].links[0].from_node
        else:
            mapping = nodes.new("ShaderNodeMapping")
            links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
            if env.inputs["Vector"].is_linked:
                old_link = env.inputs["Vector"].links[0]
                links.remove(old_link)
            links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        if not env.outputs["Color"].is_linked:
            links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = float(control_scene.get("env_strength_scale", 1.0))
        yaw_deg = float(template.get("hdri_yaw_deg", 0.0)) + float(control_scene.get("hdri_yaw_deg", 0.0))
        mapping.inputs["Rotation"].default_value[2] = math.radians(yaw_deg)
    else:
        bg.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
        bg.inputs["Strength"].default_value = 0.6

    return {
        "has_environment_texture": env is not None,
        "fallback_hdri_used": fallback_used,
    }


# ---------------------------------------------------------------------------
# Material helpers
# ---------------------------------------------------------------------------

def _material_slots_for_objects(objects):
    seen = set()
    mats = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat and mat.name not in seen:
                seen.add(mat.name)
                mats.append(mat)
    return mats


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _object_has_image_texture(obj) -> bool:
    if obj.type != "MESH":
        return False
    if not getattr(obj.data, "uv_layers", None):
        return False
    if len(obj.data.uv_layers) == 0:
        return False
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
                return True
    return False


def _replace_all_materials(obj, mat):
    if obj.type != "MESH":
        return
    if len(obj.material_slots) == 0:
        obj.data.materials.append(mat)
        return
    for idx in range(len(obj.material_slots)):
        obj.material_slots[idx].material = mat


def _set_first_existing_input(node, candidate_names, value):
    for name in candidate_names:
        sock = node.inputs.get(name)
        if sock is not None:
            sock.default_value = value
            return True
    return False


def _set_socket_value_safe(sock, value):
    if sock is None:
        return False
    try:
        sock.default_value = value
        return True
    except (TypeError, ValueError):
        current = getattr(sock, "default_value", None)
        if isinstance(current, (tuple, list)):
            try:
                sock.default_value = tuple(value for _ in current)
                return True
            except (TypeError, ValueError):
                return False
        return False


def _build_reference_wood_material(name: str, reference_rgb, roughness_add: float = 0.0, specular_add: float = 0.0):
    base = [_clamp01(v) for v in reference_rgb[:3]]
    dark_rgb = tuple(_clamp01(v * 0.2 + 0.008) for v in base) + (1.0,)
    mid_rgb = tuple(_clamp01(v * 0.44 + 0.014) for v in base) + (1.0,)
    light_rgb = tuple(_clamp01(v * 0.82 + 0.022) for v in base) + (1.0,)

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    noise = nodes.new("ShaderNodeTexNoise")
    wave = nodes.new("ShaderNodeTexWave")
    mix_fac = nodes.new("ShaderNodeMixRGB")
    grain_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp = nodes.new("ShaderNodeValToRGB")
    bump = nodes.new("ShaderNodeBump")

    mapping.inputs["Scale"].default_value = (2.6, 1.0, 8.8)
    noise.inputs["Scale"].default_value = 5.5
    noise.inputs["Detail"].default_value = 12.0
    noise.inputs["Roughness"].default_value = 0.48
    wave.wave_type = "BANDS"
    wave.bands_direction = "Z"
    wave.inputs["Scale"].default_value = 9.0
    wave.inputs["Distortion"].default_value = 3.0
    wave.inputs["Detail"].default_value = 4.0
    mix_fac.blend_type = "ADD"
    mix_fac.inputs["Fac"].default_value = 0.5
    grain_ramp.color_ramp.elements[0].position = 0.3
    grain_ramp.color_ramp.elements[0].color = (0.08, 0.08, 0.08, 1.0)
    grain_ramp.color_ramp.elements[1].position = 0.77
    grain_ramp.color_ramp.elements[1].color = (0.92, 0.92, 0.92, 1.0)
    color_ramp.color_ramp.elements[0].position = 0.16
    color_ramp.color_ramp.elements[0].color = dark_rgb
    color_ramp.color_ramp.elements[1].position = 0.9
    color_ramp.color_ramp.elements[1].color = light_rgb
    mid_elem = color_ramp.color_ramp.elements.new(0.52)
    mid_elem.color = mid_rgb
    bump.inputs["Strength"].default_value = 0.09
    bump.inputs["Distance"].default_value = 0.08

    bsdf.inputs["Roughness"].default_value = min(max(0.72 + roughness_add, 0.34), 0.96)
    specular = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
    if specular is not None:
        specular.default_value = min(max(0.12 + specular_add, 0.0), 0.55)
    sheen = bsdf.inputs.get("Sheen Tint") or bsdf.inputs.get("Sheen")
    _set_socket_value_safe(sheen, 0.05)

    links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
    links.new(noise.outputs["Fac"], mix_fac.inputs["Color1"])
    links.new(wave.outputs["Color"], mix_fac.inputs["Color2"])
    links.new(mix_fac.outputs["Color"], grain_ramp.inputs["Fac"])
    links.new(grain_ramp.outputs["Color"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _build_reference_ceramic_material(name: str, reference_rgb, roughness_add: float = 0.0, specular_add: float = 0.0):
    base = [_clamp01(v) for v in reference_rgb[:3]]
    darker = tuple(_clamp01(v * 0.78 + 0.01) for v in base) + (1.0,)
    lighter = tuple(_clamp01(v * 1.05 + 0.015) for v in base) + (1.0,)
    speck = tuple(_clamp01(v * 0.92 + 0.03) for v in base) + (1.0,)

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    noise_large = nodes.new("ShaderNodeTexNoise")
    noise_small = nodes.new("ShaderNodeTexNoise")
    large_ramp = nodes.new("ShaderNodeValToRGB")
    speck_ramp = nodes.new("ShaderNodeValToRGB")
    mix_rgb = nodes.new("ShaderNodeMixRGB")
    bump = nodes.new("ShaderNodeBump")

    mapping.inputs["Scale"].default_value = (2.2, 2.2, 2.2)
    noise_large.inputs["Scale"].default_value = 5.0
    noise_large.inputs["Detail"].default_value = 6.0
    noise_large.inputs["Roughness"].default_value = 0.42
    noise_small.inputs["Scale"].default_value = 58.0
    noise_small.inputs["Detail"].default_value = 2.0
    noise_small.inputs["Roughness"].default_value = 0.25

    large_ramp.color_ramp.elements[0].position = 0.26
    large_ramp.color_ramp.elements[0].color = darker
    large_ramp.color_ramp.elements[1].position = 0.88
    large_ramp.color_ramp.elements[1].color = lighter

    speck_ramp.color_ramp.elements[0].position = 0.74
    speck_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    speck_ramp.color_ramp.elements[1].position = 0.93
    speck_ramp.color_ramp.elements[1].color = speck

    mix_rgb.blend_type = "ADD"
    mix_rgb.inputs["Fac"].default_value = 0.12
    bump.inputs["Strength"].default_value = 0.018
    bump.inputs["Distance"].default_value = 0.03

    bsdf.inputs["Roughness"].default_value = min(max(0.32 + roughness_add, 0.16), 0.62)
    _set_first_existing_input(bsdf, ["Specular IOR Level", "Specular"], min(max(0.34 + specular_add, 0.0), 0.7))
    _set_first_existing_input(bsdf, ["Coat Weight", "Clearcoat"], 0.03)
    _set_first_existing_input(bsdf, ["Coat Roughness", "Clearcoat Roughness"], 0.2)
    _set_first_existing_input(bsdf, ["IOR"], 1.46)

    links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise_large.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise_small.inputs["Vector"])
    links.new(noise_large.outputs["Fac"], large_ramp.inputs["Fac"])
    links.new(noise_small.outputs["Fac"], speck_ramp.inputs["Fac"])
    links.new(large_ramp.outputs["Color"], mix_rgb.inputs["Color1"])
    links.new(speck_ramp.outputs["Color"], mix_rgb.inputs["Color2"])
    links.new(mix_rgb.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise_small.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _build_reference_plastic_material(name: str, reference_rgb, roughness_add: float = 0.0, specular_add: float = 0.0):
    base = [_clamp01(v) for v in reference_rgb[:3]]
    dark_rgb = tuple(_clamp01(v * 0.68 + 0.015) for v in base) + (1.0,)
    mid_rgb = tuple(_clamp01(v * 0.92 + 0.02) for v in base) + (1.0,)
    bright_rgb = tuple(_clamp01(v * 1.08 + 0.03) for v in base) + (1.0,)

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    noise_large = nodes.new("ShaderNodeTexNoise")
    noise_small = nodes.new("ShaderNodeTexNoise")
    grime_mix = nodes.new("ShaderNodeMixRGB")
    color_ramp = nodes.new("ShaderNodeValToRGB")
    speck_ramp = nodes.new("ShaderNodeValToRGB")
    bump = nodes.new("ShaderNodeBump")

    mapping.inputs["Scale"].default_value = (1.8, 1.8, 1.8)
    noise_large.inputs["Scale"].default_value = 3.2
    noise_large.inputs["Detail"].default_value = 7.0
    noise_large.inputs["Roughness"].default_value = 0.44
    noise_small.inputs["Scale"].default_value = 42.0
    noise_small.inputs["Detail"].default_value = 3.0
    noise_small.inputs["Roughness"].default_value = 0.34

    color_ramp.color_ramp.elements[0].position = 0.24
    color_ramp.color_ramp.elements[0].color = dark_rgb
    color_ramp.color_ramp.elements[1].position = 0.88
    color_ramp.color_ramp.elements[1].color = bright_rgb
    mid_elem = color_ramp.color_ramp.elements.new(0.56)
    mid_elem.color = mid_rgb

    speck_ramp.color_ramp.elements[0].position = 0.72
    speck_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    speck_ramp.color_ramp.elements[1].position = 0.94
    speck_ramp.color_ramp.elements[1].color = (0.26, 0.22, 0.18, 1.0)

    grime_mix.blend_type = "MULTIPLY"
    grime_mix.inputs["Fac"].default_value = 0.18
    bump.inputs["Strength"].default_value = 0.02
    bump.inputs["Distance"].default_value = 0.02

    bsdf.inputs["Roughness"].default_value = min(max(0.48 + roughness_add, 0.18), 0.82)
    _set_first_existing_input(bsdf, ["Specular IOR Level", "Specular"], min(max(0.28 + specular_add, 0.0), 0.62))
    _set_first_existing_input(bsdf, ["Coat Weight", "Clearcoat"], 0.02)
    _set_first_existing_input(bsdf, ["Coat Roughness", "Clearcoat Roughness"], 0.28)
    _set_first_existing_input(bsdf, ["IOR"], 1.48)

    links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise_large.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise_small.inputs["Vector"])
    links.new(noise_large.outputs["Fac"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], grime_mix.inputs["Color1"])
    links.new(noise_small.outputs["Fac"], speck_ramp.inputs["Fac"])
    links.new(speck_ramp.outputs["Color"], grime_mix.inputs["Color2"])
    links.new(grime_mix.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise_small.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _should_force_reference_material(control_material: dict, had_image_texture: bool) -> bool:
    if not had_image_texture:
        return False
    if "force_reference_material" in control_material:
        return bool(control_material.get("force_reference_material"))
    return any((
        abs(float(control_material.get("saturation_scale", 1.0)) - 1.0) >= 0.15,
        abs(float(control_material.get("value_scale", 1.0)) - 1.0) >= 0.2,
        abs(float(control_material.get("hue_offset", 0.0))) >= 0.02,
        abs(float(control_material.get("roughness_add", 0.0))) >= 0.12,
        abs(float(control_material.get("specular_add", 0.0))) >= 0.08,
    ))


def ensure_reference_material(
    objects,
    reference_rgb=None,
    roughness_add: float = 0.0,
    specular_add: float = 0.0,
    force_reference_material: bool = False,
    reference_material_mode: str = "auto",
):
    if not reference_rgb:
        return {
            "applied": False,
            "used_mode": None,
            "had_image_texture": False,
            "forced_override": False,
            "reason": "missing_reference_rgb",
        }
    had_image_texture = any(_object_has_image_texture(obj) for obj in objects)
    if had_image_texture and not force_reference_material:
        return {
            "applied": False,
            "used_mode": None,
            "had_image_texture": True,
            "forced_override": False,
            "reason": "textured_mesh_preserved",
        }

    mode = (reference_material_mode or "auto").strip().lower()
    if mode == "auto":
        mode = "wood"

    if mode == "ceramic":
        builder = _build_reference_ceramic_material
    elif mode == "plastic":
        builder = _build_reference_plastic_material
    else:
        builder = _build_reference_wood_material
    fallback = builder(
        f"__ARISRef_{mode.title()}__",
        reference_rgb=reference_rgb,
        roughness_add=roughness_add,
        specular_add=specular_add,
    )
    for obj in objects:
        _replace_all_materials(obj, fallback)
    return {
        "applied": True,
        "used_mode": mode,
        "had_image_texture": had_image_texture,
        "forced_override": bool(had_image_texture and force_reference_material),
        "reason": "reference_override_applied",
    }


def sanitize_materials(objects, roughness_add: float = 0.0, specular_add: float = 0.0):
    for mat in _material_slots_for_objects(objects):
        if not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            metallic = node.inputs.get("Metallic")
            roughness = node.inputs.get("Roughness")
            specular = node.inputs.get("Specular IOR Level") or node.inputs.get("Specular")
            emission_strength = node.inputs.get("Emission Strength")
            if metallic is not None:
                metallic.default_value = min(metallic.default_value, 0.35)
            if roughness is not None:
                roughness.default_value = min(max(roughness.default_value + roughness_add, 0.28), 0.96)
            if specular is not None:
                specular.default_value = min(max(specular.default_value + specular_add, 0.0), 0.55)
            clearcoat = node.inputs.get("Coat Weight") or node.inputs.get("Clearcoat")
            if clearcoat is not None:
                clearcoat.default_value = min(clearcoat.default_value, 0.08)
            if emission_strength is not None:
                emission_strength.default_value = 0.0


def adjust_material_hsv(
    objects,
    saturation_scale: float = 1.0,
    value_scale: float = 1.0,
    hue_offset: float = 0.0,
):
    if (
        abs(saturation_scale - 1.0) < 1e-6
        and abs(value_scale - 1.0) < 1e-6
        and abs(hue_offset) < 1e-6
    ):
        return
    for mat in _material_slots_for_objects(objects):
        if not mat.use_nodes:
            continue
        tree = mat.node_tree
        principled = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if principled is None:
            continue
        base_color_input = principled.inputs.get("Base Color")
        if base_color_input is None:
            continue

        existing_link = None
        for link in list(tree.links):
            if link.to_node == principled and link.to_socket == base_color_input:
                existing_link = link
                break

        hsv_node = tree.nodes.new("ShaderNodeHueSaturation")
        hsv_node.inputs["Hue"].default_value = 0.5 + hue_offset
        hsv_node.inputs["Saturation"].default_value = saturation_scale
        hsv_node.inputs["Value"].default_value = value_scale

        if existing_link is not None:
            src = existing_link.from_socket
            tree.links.remove(existing_link)
            tree.links.new(src, hsv_node.inputs["Color"])
        else:
            hsv_node.inputs["Color"].default_value = list(base_color_input.default_value)
        tree.links.new(hsv_node.outputs["Color"], base_color_input)


# ---------------------------------------------------------------------------
# Lights and camera
# ---------------------------------------------------------------------------

def create_key_light(target: Vector, template: dict, control_lighting: dict, control_scene: dict):
    base_loc = Vector(template.get("key_light_xyz", [3.5, -3.5, 5.0]))
    yaw_deg = float(template.get("key_light_yaw_deg", 0.0)) + float(control_lighting.get("key_yaw_deg", 0.0))
    rot = Matrix.Rotation(math.radians(yaw_deg), 3, "Z")
    loc = rot @ base_loc
    bpy.ops.object.light_add(type="SUN", location=loc)
    light = bpy.context.active_object
    light.name = "V7KeyLight"
    light.data.energy = float(template.get("key_light_energy", 3.0)) * float(control_lighting.get("key_scale", 1.0))
    direction = target - loc
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    _configure_contact_shadow(light, template, control_scene)
    return light


def create_camera_fill_light(camera, target: Vector, target_bbox: dict | None, template: dict, control_scene: dict | None = None):
    cam_loc = camera.matrix_world.translation.copy()
    direction = target - cam_loc
    dist = max(direction.length, 1e-4)
    direction.normalize()

    quat = camera.matrix_world.to_quaternion()
    right = quat @ Vector((1.0, 0.0, 0.0))
    up = quat @ Vector((0.0, 1.0, 0.0))

    if target_bbox:
        bbox_diag = (Vector(target_bbox["max"]) - Vector(target_bbox["min"])).length
    else:
        bbox_diag = 1.0
    bbox_diag = max(bbox_diag, 0.75)

    forward_ratio = float(template.get("camera_fill_forward_ratio", 0.08))
    up_ratio = float(template.get("camera_fill_up_ratio", 0.30))
    right_ratio = float(template.get("camera_fill_right_ratio", 0.18))
    fill_energy = float(template.get("camera_fill_energy", 420.0))
    if template.get("camera_mode") == "scene_camera_match" and template.get("camera_fill_scene_match_kicker", True):
        forward_ratio *= 0.4
        up_ratio *= 1.1
        right_ratio = max(right_ratio, 0.26)
        fill_energy *= 0.6

    loc = (
        cam_loc
        + direction * (dist * forward_ratio)
        + up * (bbox_diag * up_ratio)
        + right * (bbox_diag * right_ratio)
    )

    bpy.ops.object.light_add(type="AREA", location=loc)
    light = bpy.context.active_object
    light.name = "V7CameraFill"
    light.rotation_euler = (target - loc).to_track_quat("-Z", "Y").to_euler()
    light.data.energy = fill_energy
    color = template.get("camera_fill_color", [1.0, 0.985, 0.97])
    if hasattr(light.data, "color") and isinstance(color, (list, tuple)) and len(color) >= 3:
        light.data.color = tuple(float(v) for v in color[:3])
    size = max(2.5, bbox_diag * float(template.get("camera_fill_size_multiplier", 4.0)))
    if hasattr(light.data, "shape"):
        light.data.shape = "RECTANGLE"
    if hasattr(light.data, "size"):
        light.data.size = size
    if hasattr(light.data, "size_y"):
        light.data.size_y = size * 0.8
    _configure_contact_shadow(light, template, control_scene or {})
    return light


def create_qc_camera(scene):
    bpy.ops.object.camera_add(location=(0.0, 0.0, 0.0))
    cam = bpy.context.active_object
    cam.name = "V7QCCamera"
    cam.data.lens = 50
    cam.data.clip_start = 0.1
    cam.data.clip_end = 100.0
    scene.camera = cam
    return cam


def resolve_scene_camera(scene):
    if scene.camera is not None:
        return scene.camera
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            scene.camera = obj
            return obj
    return create_qc_camera(scene)


def camera_screen_ground_anchor(scene, camera, ground_z: float, u: float = 0.5, v: float = 0.22):
    """Intersect a screen-space ray with the horizontal ground plane.

    `u` and `v` are normalized screen coordinates where (0,0) is bottom-left
    and (1,1) is top-right. Using a lower-mid screen anchor matches the
    pseudo-reference setup much better than aiming through the image center.
    """
    frame_local = list(camera.data.view_frame(scene=scene))
    if len(frame_local) != 4:
        return None

    bottom = sorted(frame_local, key=lambda p: p.y)[:2]
    top = sorted(frame_local, key=lambda p: p.y)[-2:]
    bl, br = sorted(bottom, key=lambda p: p.x)
    tl, tr = sorted(top, key=lambda p: p.x)

    local_point = (
        (1 - u) * (1 - v) * bl
        + u * (1 - v) * br
        + (1 - u) * v * tl
        + u * v * tr
    )

    origin = camera.matrix_world.translation.copy()
    world_point = camera.matrix_world @ local_point
    direction = (world_point - origin).normalized()
    if abs(direction.z) < 1e-6:
        return None
    t = (ground_z - origin.z) / direction.z
    if t <= 0:
        return None
    return origin + direction * t


def position_camera(cam, target: Vector, azimuth_deg: float, elevation_deg: float, distance: float):
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = target.x + distance * math.cos(el) * math.sin(az)
    y = target.y - distance * math.cos(el) * math.cos(az)
    z = target.z + distance * math.sin(el)
    cam.location = (x, y, z)
    direction = target - Vector((x, y, z))
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


# ---------------------------------------------------------------------------
# Scene bounds and camera distance (mirrors render_all.py:55-64, 259-291)
# ---------------------------------------------------------------------------

def get_scene_bounds_all(ignore_names=None):
    """World-space bbox of all visible meshes, optionally excluding named objects."""
    ignore = set(ignore_names or [])
    min_v = [float("inf")] * 3
    max_v = [float("-inf")] * 3
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.hide_get() or obj.name in ignore:
            continue
        for corner in obj.bound_box:
            w = obj.matrix_world @ Vector(corner)
            for i in range(3):
                min_v[i] = min(min_v[i], w[i])
                max_v[i] = max(max_v[i], w[i])
    if min_v[0] == float("inf"):
        return Vector((0, 0, 0)), Vector((10, 10, 5))
    return Vector(min_v), Vector(max_v)


def compute_scene_camera_params(scene_bounds_min, scene_bounds_max, target_center, template):
    """Camera distance relative to scene bounds (mirrors render_all.py ring ratios)."""
    size = scene_bounds_max - scene_bounds_min
    max_span = max(size.x, size.y, 0.1)
    scene_height = max(size.z, 0.1)
    distance = max_span * float(template.get("camera_radius_ratio", 0.02))
    height_offset = scene_height * float(template.get("camera_height_ratio", 0.01))
    return distance, height_offset


def compute_object_camera_params(target_bbox: dict, template: dict):
    """Camera distance driven by the inserted object's own size.

    Scene-relative distance caused many views to land inside or too close to the
    object because the scene is much larger than the inserted asset. For review
    renders we want the subject readable first, so the default camera now scales
    from the object bbox itself.
    """
    size = Vector(target_bbox["size"])
    diag = max(size.length, 1e-3)
    max_span = max(size.x, size.y, size.z, 1e-3)
    distance = max(
        diag * float(template.get("camera_object_diag_multiplier", 2.2)),
        max_span * float(template.get("camera_object_span_multiplier", 2.8)),
        float(template.get("camera_distance_min", 2.0)),
    )
    distance = min(distance, float(template.get("camera_distance_max", 8.0)))
    target = Vector(target_bbox["center"])
    target[2] = float(target_bbox["min"][2]) + size.z * float(template.get("camera_target_height_ratio", 0.45))
    return distance, target


# ---------------------------------------------------------------------------
# Object placement helpers
# ---------------------------------------------------------------------------

def auto_adjust_object_size_scene_relative(root, mesh_objects, scene_bounds_min, scene_bounds_max):
    """Scale object to ~10% of avg scene dimension (matches render_all.py).

    This makes motorbikes/gazebos appear at realistic proportions relative
    to the scene (street/park) rather than being tiny boxes.
    """
    size = scene_bounds_max - scene_bounds_min
    avg_dim = (size.x + size.y + size.z) / 3.0
    if avg_dim < 1e-6:
        return
    target_size = avg_dim * 0.1
    bbox = percentile_bbox(mesh_objects)
    if not bbox:
        return
    current_size = max(bbox["size"])
    if current_size < 1e-6:
        return
    scale_factor = target_size / current_size
    root.scale = tuple(v * scale_factor for v in root.scale)
    bpy.context.view_layer.update()
    print(f"[scene] Scene-relative scale: avg_dim={avg_dim:.2f}, "
          f"target={target_size:.2f}, current={current_size:.2f}, factor={scale_factor:.3f}")


def target_bbox_height(objects):
    bbox = percentile_bbox(objects)
    return float(bbox["size"][2]) if bbox else 0.0


def fit_object_height(root, objects, target_range):
    current = target_bbox_height(objects)
    if current <= 1e-6:
        return current, 1.0
    lo, hi = float(target_range[0]), float(target_range[1])
    if lo <= current <= hi:
        return current, 1.0
    desired = (lo + hi) / 2.0
    factor = desired / current
    root.scale = tuple(v * factor for v in root.scale)
    bpy.context.view_layer.update()
    return target_bbox_height(objects), factor


def align_object_to_support(root, objects, support, control_obj, target_range):
    support_center = support.get("center_xy", [0.0, 0.0])
    root.location.x = support_center[0]
    root.location.y = support_center[1]
    root.rotation_euler.z += math.radians(float(control_obj.get("yaw_deg", 0.0)))

    _, auto_scale = fit_object_height(root, objects, target_range)
    manual_scale = float(control_obj.get("scale", 1.0))
    if abs(manual_scale - 1.0) > 1e-6:
        root.scale = tuple(v * manual_scale for v in root.scale)
        bpy.context.view_layer.update()

    bbox = percentile_bbox(objects)
    bottom_z = float(bbox["min"][2]) if bbox else 0.0
    support_z = float(support.get("z", 0.0))
    root.location.z += support_z - bottom_z
    root.location.z += float(control_obj.get("offset_z", 0.0))
    bpy.context.view_layer.update()
    return auto_scale


# ---------------------------------------------------------------------------
# Physics metrics
# ---------------------------------------------------------------------------

def compute_physics_metrics(objects, support):
    bbox = percentile_bbox(objects)
    if bbox is None:
        return {
            "bbox_height": 0.0,
            "contact_gap": 0.0,
            "penetration_depth": 0.0,
            "is_floating": False,
            "is_intersecting_support": False,
            "is_out_of_support_bounds": False,
        }

    min_z = float(bbox["min"][2])
    max_z = float(bbox["max"][2])
    support_z = float(support.get("z", 0.0))
    bbox_height = max(max_z - min_z, 0.0)
    gap = max(0.0, min_z - support_z)
    penetration = max(0.0, support_z - min_z)
    floating_tol = max(0.0012, min(0.004, bbox_height * 0.005))
    intersect_tol = max(0.0008, min(0.003, bbox_height * 0.004))
    bounds = support.get("bounds")
    out_of_bounds = False
    if bounds:
        out_of_bounds = (
            bbox["min"][0] < bounds[0] or bbox["max"][0] > bounds[1] or
            bbox["min"][1] < bounds[2] or bbox["max"][1] > bounds[3]
        )
    return {
        "bbox_height": round(max_z - min_z, 6),
        "contact_gap": round(gap, 6),
        "penetration_depth": round(penetration, 6),
        "floating_tolerance": round(floating_tol, 6),
        "intersect_tolerance": round(intersect_tol, 6),
        "is_floating": gap > floating_tol,
        "is_intersecting_support": penetration > intersect_tol,
        "is_out_of_support_bounds": bool(out_of_bounds),
    }


# ---------------------------------------------------------------------------
# Mask rendering (mirrors render_all.py render_mask_for_main strategy)
# ---------------------------------------------------------------------------

def render_mask(scene, out_path: str, imported_mesh_names):
    """Render an object-only RGBA mask image.

    The downstream CV code can directly consume the alpha channel, so the most
    reliable mask is to render the inserted object alone with a transparent
    background. The RGB content is irrelevant; alpha carries the silhouette.
    """
    mesh_objs = [
        bpy.data.objects[n] for n in imported_mesh_names
        if n in bpy.data.objects and bpy.data.objects[n].type == "MESH"
    ]

    # --- Record original state ---
    orig_light_energies = {lt: lt.energy for lt in bpy.data.lights}
    orig_view_transform = scene.view_settings.view_transform
    try:
        orig_look = scene.view_settings.look
    except AttributeError:
        orig_look = "None"
    orig_gamma = scene.view_settings.gamma
    orig_color_mode = scene.render.image_settings.color_mode
    orig_film_transparent = scene.render.film_transparent

    world = scene.world
    # --- Hide all non-imported mesh objects ---
    hidden_objs = []
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name not in imported_mesh_names and not obj.hide_render:
            obj.hide_render = True
            hidden_objs.append(obj)

    # --- Replace materials with white emission ---
    mask_mat = bpy.data.materials.new("__V7MaskTmp__")
    mask_mat.use_nodes = True
    mnodes = mask_mat.node_tree.nodes
    mnodes.clear()
    out_node = mnodes.new("ShaderNodeOutputMaterial")
    emit_node = mnodes.new("ShaderNodeEmission")
    emit_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emit_node.inputs["Strength"].default_value = 1.0
    mask_mat.node_tree.links.new(emit_node.outputs["Emission"], out_node.inputs["Surface"])

    orig_materials = []
    for mesh in mesh_objs:
        if len(mesh.material_slots) == 0:
            mesh.data.materials.append(mask_mat)
        orig_materials.append([slot.material for slot in mesh.material_slots])
        for i in range(len(mesh.material_slots)):
            mesh.material_slots[i].material = mask_mat

    # --- Render imported meshes only over transparent background ---
    for lt in bpy.data.lights:
        lt.energy = 0.0

    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except TypeError:
        pass
    scene.view_settings.gamma = 1.0
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"

    # --- Render ---
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)

    # --- Restore materials ---
    for mesh, mats in zip(mesh_objs, orig_materials):
        for i, mat in enumerate(mats):
            mesh.material_slots[i].material = mat
    bpy.data.materials.remove(mask_mat)

    # --- Restore visibility ---
    for obj in hidden_objs:
        obj.hide_render = False

    # --- Restore lights ---
    for lt, energy in orig_light_energies.items():
        lt.energy = energy

    # --- Restore color management ---
    scene.view_settings.view_transform = orig_view_transform
    try:
        scene.view_settings.look = orig_look
    except TypeError:
        pass
    scene.view_settings.gamma = orig_gamma
    scene.render.image_settings.color_mode = orig_color_mode
    scene.render.film_transparent = orig_film_transparent


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_object(asset_path: str, obj_id: str, output_dir: str,
                  resolution: int, engine: str, template: dict, control_state: dict):
    blend_path = template["blend_path"]
    if not os.path.exists(blend_path):
        raise FileNotFoundError(f"Scene blend not found: {blend_path}")

    bpy.ops.wm.open_mainfile(filepath=blend_path)
    scene = bpy.context.scene
    setup_render_settings(scene, resolution, engine, template)

    # --- Light handling ---
    if template.get("disable_existing_lights", False):
        # Legacy path: hide lights entirely
        disable_existing_lights()
    else:
        # New default: scale lights down, preserve scene look (mirrors render_all.py:335-340)
        scale_existing_lights(float(template.get("scale_existing_lights", 0.3)))
        scale_world_background(scene, float(template.get("scale_world_background", 0.3)))

    imported_info = import_asset(asset_path)
    asset_kind = imported_info["asset_kind"]
    imported = imported_info["imported"]
    mesh_objects = imported_info["mesh_objects"]
    root = imported_info["root"]
    imported_names = imported_info["imported_names"]

    control_object = control_state.get("object", {})
    control_scene = control_state.get("scene", {})
    control_lighting = control_state.get("lighting", {})
    control_material = control_state.get("material", {})
    camera_mode = template.get("camera_mode", "scene_bounds_relative")
    configure_existing_scene_lights(template, control_scene)

    support_plane_mode = template.get("support_plane_mode", "ground_object_raycast")
    extra_meta = {}

    if support_plane_mode == "ground_object_raycast":
        # New mode: name-based ground detection + ray-cast precise landing
        support_name_hint = template.get("support_object_name", "ground")
        ground_obj, ground_detection_mode = find_ground_object(support_name_hint)

        if ground_obj:
            fix_ground_orientation(ground_obj)

            # Scale to target height range or use scene-relative sizing
            object_scale_mode = template.get("object_scale_mode", "height_range")
            if object_scale_mode == "scene_relative":
                bounds_min, bounds_max = get_scene_bounds_all(ignore_names=imported_names)
                auto_adjust_object_size_scene_relative(root, mesh_objects, bounds_min, bounds_max)
                auto_scale = 1.0
            else:
                _, auto_scale = fit_object_height(
                    root, mesh_objects, template.get("target_bbox_height_range", [0.3, 1.5])
                )

            # Apply manual scale from controller
            manual_scale = float(control_object.get("scale", 1.0))
            if abs(manual_scale - 1.0) > 1e-6:
                root.scale = tuple(v * manual_scale for v in root.scale)
                bpy.context.view_layer.update()

            # Position at ground center XY or at the authored scene-camera focus point.
            ground_bbox = [ground_obj.matrix_world @ Vector(c) for c in ground_obj.bound_box]
            ground_cx = sum(v.x for v in ground_bbox) / 8.0
            ground_cy = sum(v.y for v in ground_bbox) / 8.0
            ground_z_rough = min(v.z for v in ground_bbox)
            anchor_mode = "ground_center"
            anchor_point = None
            if camera_mode == "scene_camera_match":
                scene_cam = resolve_scene_camera(scene)
                anchor_point = camera_screen_ground_anchor(
                    scene,
                    scene_cam,
                    ground_z_rough,
                    u=float(template.get("camera_anchor_u", 0.5)),
                    v=float(template.get("camera_anchor_v", 0.22)),
                )
                if anchor_point is not None:
                    anchor_mode = "scene_camera_screen_ray"

            if anchor_point is not None:
                root.location.x = anchor_point.x
                root.location.y = anchor_point.y
            else:
                root.location.x = ground_cx
                root.location.y = ground_cy

            # Apply yaw from controller
            root.rotation_euler.z += math.radians(float(control_object.get("yaw_deg", 0.0)))
            bpy.context.view_layer.update()

            # Rough Z: align object bbox bottom to ground bbox bottom
            bbox = percentile_bbox(mesh_objects)
            if bbox:
                bottom_z = float(bbox["min"][2])
                root.location.z += ground_z_rough - bottom_z
                bpy.context.view_layer.update()

            # Precision landing via ray cast
            ground_hit = adjust_object_to_ground_with_ray(root, mesh_objects, ground_obj)

            # Apply Z offset from controller, then add a small size-aware epsilon
            # to avoid visible asphalt intersection without making the object float.
            root.location.z += float(control_object.get("offset_z", 0.0))
            ground_contact_epsilon = compute_ground_contact_epsilon(mesh_objects, template)
            root.location.z += ground_contact_epsilon
            bpy.context.view_layer.update()

            # Build support dict for physics metrics
            ground_z = (
                float(ground_hit.get("support_z"))
                if ground_hit.get("support_z") is not None
                else min(v.z for v in ground_bbox)
            )
            grounding_stabilization = stabilize_ground_contact(
                root,
                mesh_objects,
                support_z=ground_z,
                ground_contact_epsilon=ground_contact_epsilon,
            )
            ground_bounds = [
                min(v.x for v in ground_bbox), max(v.x for v in ground_bbox),
                min(v.y for v in ground_bbox), max(v.y for v in ground_bbox),
            ]
            support = {
                "name": ground_obj.name,
                "z": ground_z,
                "bounds": ground_bounds,
                "center_xy": [ground_cx, ground_cy],
                "area": (ground_bounds[1] - ground_bounds[0]) * (ground_bounds[3] - ground_bounds[2]),
                "created": False,
            }
            extra_meta = {
                "ground_object_name": ground_obj.name,
                "ground_detection_mode": ground_detection_mode,
                "ground_hit_success": bool(ground_hit.get("success")),
                "ground_hit_sample_count": int(ground_hit.get("sample_count", 0)),
                "ground_hit_count": int(ground_hit.get("hit_count", 0)),
                "ground_contact_epsilon": round(float(ground_contact_epsilon), 6),
                "grounding_stabilization": grounding_stabilization,
                "placement_anchor_mode": anchor_mode,
                "placement_anchor_xy": [
                    round(float(root.location.x), 6),
                    round(float(root.location.y), 6),
                ],
            }
        else:
            # No ground found: fallback to hidden support plane
            print("[scene] No ground object found, falling back to hidden support plane")
            support = create_hidden_support_plane(float(template.get("fallback_support_plane_size", 10.0)))
            auto_scale = align_object_to_support(
                root, mesh_objects, support, control_object,
                template.get("target_bbox_height_range", [0.3, 1.5])
            )
            extra_meta = {
                "ground_object_name": None,
                "ground_detection_mode": "fallback_plane",
                "ground_hit_success": False,
            }
    else:
        # Legacy mode: auto_detect_lowest_horizontal_mesh
        support = ensure_support_plane(template, ignore_names=imported_names)
        auto_scale = align_object_to_support(
            root, mesh_objects, support, control_object,
            template.get("target_bbox_height_range", [0.3, 1.5])
        )

    preserve_blend_materials = bool(template.get("preserve_blend_asset_materials", True))
    should_preserve_materials = asset_kind == "blend" and preserve_blend_materials
    material_adaptation = {
        "applied": False,
        "used_mode": None,
        "had_image_texture": any(_object_has_image_texture(obj) for obj in mesh_objects),
        "forced_override": False,
        "reason": "blend_asset_materials_preserved" if should_preserve_materials else "not_requested",
    }
    if not should_preserve_materials:
        force_reference_material = _should_force_reference_material(
            control_material,
            material_adaptation["had_image_texture"],
        )
        material_adaptation = ensure_reference_material(
            mesh_objects,
            reference_rgb=control_material.get("reference_rgb"),
            roughness_add=float(control_material.get("roughness_add", 0.0)),
            specular_add=float(control_material.get("specular_add", 0.0)),
            force_reference_material=force_reference_material,
            reference_material_mode=str(control_material.get("reference_material_mode", "auto")),
        )
        sanitize_materials(
            mesh_objects,
            roughness_add=float(control_material.get("roughness_add", 0.0)),
            specular_add=float(control_material.get("specular_add", 0.0)),
        )
        adjust_material_hsv(
            mesh_objects,
            saturation_scale=float(control_material.get("saturation_scale", 1.0)),
            value_scale=float(control_material.get("value_scale", 1.0)),
            hue_offset=float(control_material.get("hue_offset", 0.0)),
        )

    env_meta = ensure_world_environment(scene, template, control_scene)

    target_bbox = percentile_bbox(mesh_objects)
    target = Vector(target_bbox["center"]) if target_bbox else Vector((0.0, 0.0, 0.5))

    # Conditionally add key light (default: False — preserve 4.blend lights)
    if template.get("add_key_light", False):
        create_key_light(target, template, control_lighting, control_scene)

    # Camera mode: teacher/background match or generated QC camera
    camera_target = target
    camera_name = None
    if camera_mode == "scene_camera_match":
        cam = resolve_scene_camera(scene)
        camera_name = cam.name
        camera_dist = None
        camera_target = None
    else:
        cam = create_qc_camera(scene)
        if camera_mode == "object_bbox_relative" and target_bbox:
            camera_dist, camera_target = compute_object_camera_params(target_bbox, template)
        elif camera_mode == "scene_bounds_relative":
            bounds_min, bounds_max = get_scene_bounds_all(ignore_names=imported_names)
            camera_dist, _ = compute_scene_camera_params(bounds_min, bounds_max, target, template)
        else:
            camera_dist = float(template.get("camera_distance", 3.5))
        camera_name = cam.name

    if template.get("add_camera_fill_light", False):
        create_camera_fill_light(cam, target, target_bbox, template, control_scene)

    obj_out = os.path.join(output_dir, obj_id)
    os.makedirs(obj_out, exist_ok=True)

    qc_views = [(int(v[0]), int(v[1])) for v in template.get("qc_views", DEFAULT_QC_VIEWS)]

    if template.get("adaptive_brightness", False):
        if camera_mode != "scene_camera_match":
            preview_az, preview_el = qc_views[0] if qc_views else DEFAULT_QC_VIEWS[0]
            target_bbox = percentile_bbox(mesh_objects)
            preview_target = Vector(target_bbox["center"]) if target_bbox else Vector((0.0, 0.0, 0.5))
            if camera_mode == "object_bbox_relative" and target_bbox:
                preview_dist, preview_target = compute_object_camera_params(target_bbox, template)
            elif camera_mode == "scene_bounds_relative":
                bounds_min, bounds_max = get_scene_bounds_all(ignore_names=imported_names)
                preview_dist, _ = compute_scene_camera_params(bounds_min, bounds_max, preview_target, template)
            else:
                preview_dist = float(template.get("camera_distance", 3.5))
            position_camera(cam, preview_target, preview_az, preview_el, preview_dist)
        adjust_lighting_to_target_brightness(
            target_brightness=float(template.get("target_brightness", 0.4)),
            preview_samples=int(template.get("cycles_preview_samples", 32)),
            damping=float(template.get("adaptive_brightness_damping", 0.45)),
            min_gain=float(template.get("adaptive_brightness_min_gain", 0.82)),
            max_gain=float(template.get("adaptive_brightness_max_gain", 1.18)),
            world_gain_scale=float(template.get("adaptive_world_gain_scale", 0.55)),
        )

    rendered_frames = []
    for az, el in qc_views:
        el_str = f"{el:+03d}"
        rgb_name = f"az{az:03d}_el{el_str}.png"
        mask_name = f"az{az:03d}_el{el_str}_mask.png"
        rgb_path = os.path.join(obj_out, rgb_name)
        mask_path = os.path.join(obj_out, mask_name)

        target_bbox = percentile_bbox(mesh_objects)
        target = Vector(target_bbox["center"]) if target_bbox else Vector((0.0, 0.0, 0.5))
        if camera_mode == "scene_camera_match":
            pass
        else:
            if camera_mode == "object_bbox_relative" and target_bbox:
                camera_dist, camera_target = compute_object_camera_params(target_bbox, template)
            else:
                camera_target = target
            position_camera(cam, camera_target, az, el, camera_dist)

        scene.render.filepath = rgb_path
        bpy.ops.render.render(write_still=True)
        render_mask(scene, mask_path, imported_names)

        rendered_frames.append({
            "filename": rgb_name,
            "mask_filename": mask_name,
            "azimuth": az,
            "elevation": el,
            "path": rgb_path,
            "mask_path": mask_path,
        })

    physics = compute_physics_metrics(mesh_objects, support)
    metadata = {
        "obj_id": obj_id,
        "source_asset": asset_path,
        "source_asset_kind": asset_kind,
        "blend_path": blend_path,
        "resolution": resolution,
        "engine": engine,
        "control_state": control_state,
        "preserved_blend_materials": bool(should_preserve_materials),
        "material_adaptation": material_adaptation,
        "support_plane_name": support.get("name"),
        "support_plane_z": support.get("z"),
        "support_bounds_xy": support.get("bounds"),
        "support_plane_mode": support_plane_mode,
        "camera_mode": camera_mode,
        "camera_name": camera_name,
        "camera_distance_used": round(camera_dist, 4) if camera_dist is not None else None,
        "camera_target_used": [round(float(v), 6) for v in camera_target] if camera_target is not None else None,
        "qc_views": qc_views,
        "auto_scale_factor": round(auto_scale, 6),
        "environment": env_meta,
        **extra_meta,
        **physics,
        "frames": rendered_frames,
    }
    with open(os.path.join(obj_out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def main():
    args = parse_args()
    template = load_scene_template(args.scene_template)
    control_state = load_json(args.control_state, default={}) or {}

    os.makedirs(args.output_dir, exist_ok=True)
    asset_files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.endswith(".glb") or f.endswith(".blend")
    )
    if args.obj_id:
        asset_files = [f for f in asset_files if Path(f).stem == args.obj_id]
    if not asset_files:
        print(f"[scene-render] No .glb or .blend files found in {args.input_dir}")
        sys.exit(1)

    summary = {
        "scene_template": resolve_path(args.scene_template),
        "total_objects": 0,
        "objects": {},
    }

    for fname in asset_files:
        obj_id = Path(fname).stem
        asset_path = os.path.join(args.input_dir, fname)
        print(f"[scene-render] Rendering {obj_id} from {asset_path}")
        metadata = render_object(
            asset_path=asset_path,
            obj_id=obj_id,
            output_dir=args.output_dir,
            resolution=args.resolution,
            engine=args.engine,
            template=template,
            control_state=control_state,
        )
        summary["objects"][obj_id] = {
            "contact_gap": metadata["contact_gap"],
            "penetration_depth": metadata["penetration_depth"],
            "bbox_height": metadata["bbox_height"],
            "support_plane_name": metadata["support_plane_name"],
            "frames": len(metadata["frames"]),
        }

    summary["total_objects"] = len(summary["objects"])
    with open(os.path.join(args.output_dir, "render_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[scene-render] Done: {summary['total_objects']} objects")


if __name__ == "__main__":
    main()
