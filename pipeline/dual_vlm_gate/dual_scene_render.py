"""Blender entrypoint for dual-object scene renders with gate metadata.

Run inside Blender:
  blender scene.blend --background --python pipeline/dual_vlm_gate/dual_scene_render.py -- \
    --obj-a mesh_a.glb --obj-b mesh_b.glb --output-dir /tmp/dual_round \
    --sample-id pair_001 --control-state state.json
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import demo_dual_object_placement as dual  # noqa: E402


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser(description="Render a dual-object VLM-gate scene")
    parser.add_argument("--obj-a", required=True)
    parser.add_argument("--obj-b", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--control-state", default=None)
    parser.add_argument("--placement-state-in", default=None)
    parser.add_argument("--placement-state-out", default=None)
    parser.add_argument("--template", choices=list(dual.COMPOSITION_TEMPLATES.keys()), default=None)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--engine", default="CYCLES", choices=["EEVEE", "CYCLES"])
    parser.add_argument("--max-object-ratio", type=float, default=dual.MAX_OBJECT_RATIO)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def load_json(path, default=None):
    if not path:
        return default
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def vec_to_list(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [round(float(v), 6) for v in value]


def bbox_to_json(bbox):
    if not bbox:
        return None
    return {
        "min": vec_to_list(bbox.get("min")),
        "max": vec_to_list(bbox.get("max")),
        "center": vec_to_list(bbox.get("center")),
        "size": vec_to_list(bbox.get("size")),
    }


def placement_to_state(placement: dict, seed: int, source: str) -> dict:
    return {
        "schema": "dual_placement_state_v1",
        "source": source,
        "seed": int(seed),
        "template": placement.get("template"),
        "obj1_position": vec_to_list(placement.get("obj1_position")),
        "obj2_position": vec_to_list(placement.get("obj2_position")),
        "separation": round(float(placement.get("separation", 0.0)), 6),
        "height_offset": round(float(placement.get("height_offset", 0.0)), 6),
        "side": int(placement.get("side", 0) or 0),
        "camera_azimuth": round(float(placement.get("camera_azimuth", 0.0)), 10),
    }


def object_has_image_texture(meshes) -> bool:
    for obj in meshes:
        for slot in getattr(obj, "material_slots", []):
            mat = slot.material
            if mat is None or not getattr(mat, "use_nodes", False) or not mat.node_tree:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
                    return True
    return False


def apply_object_control(root, control: dict, apply_location: bool = True):
    scale = float(control.get("scale", 1.0))
    yaw = math.radians(float(control.get("yaw_deg", 0.0)))
    offset_xy = control.get("offset_xy") or [0.0, 0.0]
    offset_z = float(control.get("offset_z", 0.0))
    root.scale = tuple(float(v) * scale for v in root.scale)
    root.rotation_euler.z += yaw
    if apply_location:
        root.location.x += float(offset_xy[0])
        root.location.y += float(offset_xy[1])
        root.location.z += offset_z
    bpy.context.view_layer.update()


def raycast_ground_z(scene_bvh, x: float, y: float, scene_min, scene_max, ground_info=None):
    z_top = float(scene_max[2]) + 2.0
    direction = Vector((0, 0, -1))
    hit_loc, hit_normal, _, _ = scene_bvh.ray_cast(Vector((float(x), float(y), z_top)), direction)
    if hit_loc is None:
        return None
    if ground_info and hit_loc.z > float(ground_info.get("z", 0.0)) + 1.0:
        retry_origin = Vector((float(x), float(y), hit_loc.z - 0.01))
        hit_loc2, hit_normal2, _, _ = scene_bvh.ray_cast(retry_origin, direction)
        if hit_loc2 is not None and hit_normal2.z > 0.6:
            hit_loc = hit_loc2
            hit_normal = hit_normal2
    if hit_normal is None or hit_normal.z < 0.45:
        return None
    if ground_info and abs(hit_loc.z - float(ground_info.get("z", hit_loc.z))) > 8.0:
        return None
    return float(hit_loc.z)


def apply_post_placement_control(root, meshes, control: dict, scene_bvh, scene_min, scene_max, ground_info, label: str):
    adjustments = []
    bbox = dual.get_object_bbox(meshes)
    if bbox is None:
        return adjustments
    ground_snap = float(control.get("ground_snap", 0.0) or 0.0)
    ground_snap_enabled = bool(control.get("ground_snap_enabled", True))
    offset_xy = control.get("offset_xy") or [0.0, 0.0]
    offset_z = float(control.get("offset_z", 0.0))

    root.location.x += float(offset_xy[0])
    root.location.y += float(offset_xy[1])
    bpy.context.view_layer.update()

    if ground_snap > 0.0 and not ground_snap_enabled:
        adjustments.append({
            "object": label,
            "action": "ground_snap",
            "snap_count": round(ground_snap, 6),
            "applied": False,
            "reason": "ground_snap_disabled",
        })
    elif ground_snap > 0.0:
        bbox = dual.get_object_bbox(meshes)
        bottom_x = float((bbox["min"][0] + bbox["max"][0]) / 2.0)
        bottom_y = float((bbox["min"][1] + bbox["max"][1]) / 2.0)
        ground_z = raycast_ground_z(scene_bvh, bottom_x, bottom_y, scene_min, scene_max, ground_info)
        if ground_z is not None:
            before = float(bbox["min"][2])
            dual.place_object_at(root, meshes, bottom_x, bottom_y, ground_z)
            after_bbox = dual.get_object_bbox(meshes)
            adjustments.append({
                "object": label,
                "action": "ground_snap",
                "snap_count": round(ground_snap, 6),
                "xy": [round(bottom_x, 6), round(bottom_y, 6)],
                "ground_z": round(ground_z, 6),
                "bottom_z_before": round(before, 6),
                "bottom_z_after": round(float(after_bbox["min"][2]), 6) if after_bbox else None,
            })
        else:
            adjustments.append({
                "object": label,
                "action": "ground_snap",
                "snap_count": round(ground_snap, 6),
                "applied": False,
                "reason": "ground_raycast_failed",
            })

    if abs(offset_z) > 1e-6:
        before = float(root.location.z)
        root.location.z += offset_z
        adjustments.append({
            "object": label,
            "action": "offset_z",
            "old_root_z": round(before, 6),
            "new_root_z": round(float(root.location.z), 6),
            "offset_z": round(offset_z, 6),
        })
    bpy.context.view_layer.update()
    return adjustments


def apply_pair_control(root1, root2, placement: dict, state: dict):
    pair = state.get("pair") or {}
    distance_scale = float(pair.get("distance_scale", 1.0))
    height_offset_scale = float(pair.get("height_offset_scale", 1.0))
    side_sign = int(pair.get("side_sign", 0) or 0)
    if abs(distance_scale - 1.0) < 1e-6 and abs(height_offset_scale - 1.0) < 1e-6 and side_sign == 0:
        return

    p1 = np.array(placement["obj1_position"], dtype=float)
    p2 = np.array(placement["obj2_position"], dtype=float)
    delta = p2 - p1
    if side_sign < 0:
        delta[:2] *= -1.0
    delta[:2] *= distance_scale
    delta[2] *= height_offset_scale
    desired_p2 = p1 + delta
    current_p2 = np.array(root2.location[:], dtype=float)
    bottom_delta = desired_p2 - p2
    root2.location = Vector(current_p2 + bottom_delta)
    placement["obj2_position"] = desired_p2.tolist()
    placement["separation"] = float(placement.get("separation", 0.0)) * distance_scale
    placement["height_offset"] = float(placement.get("height_offset", 0.0)) * height_offset_scale
    bpy.context.view_layer.update()


def setup_camera_with_control(placement, scene_min, scene_max, state):
    camera = state.get("camera") or {}
    if float(camera.get("azimuth_offset_deg", 0.0)):
        placement["camera_azimuth"] = (
            float(placement["camera_azimuth"]) + math.radians(float(camera.get("azimuth_offset_deg", 0.0)))
        ) % (2 * math.pi)
    elevation = float(camera.get("elevation_deg", dual.CAMERA_ELEVATION_DEG))
    cam = dual.setup_camera_for_dual(placement, scene_min, scene_max, elevation_deg=elevation)
    cam.data.lens = float(camera.get("focal_length_mm", cam.data.lens))
    distance_scale = float(camera.get("distance_scale", 1.0))
    distance_scale *= float(camera.get("framing_margin", 1.0))
    if abs(distance_scale - 1.0) > 1e-6:
        combined = dual.compute_combined_bbox(placement["bbox1"], placement["bbox2"])
        center = Vector(combined["center"].tolist())
        direction = cam.location - center
        cam.location = center + direction * distance_scale
        cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
    return cam


def _rgb_to_adjusted_rgba(value, value_scale: float, saturation_scale: float, contrast_scale: float):
    rgba = list(value)
    if len(rgba) < 4:
        rgba += [1.0] * (4 - len(rgba))
    r, g, b, a = [float(v) for v in rgba[:4]]
    h, s, v = colorsys.rgb_to_hsv(max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b)))
    s = max(0.0, min(1.0, s * saturation_scale))
    v = max(0.0, min(1.0, v * value_scale))
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    if abs(contrast_scale - 1.0) > 1e-6:
        r = max(0.0, min(1.0, 0.5 + (r - 0.5) * contrast_scale))
        g = max(0.0, min(1.0, 0.5 + (g - 0.5) * contrast_scale))
        b = max(0.0, min(1.0, 0.5 + (b - 0.5) * contrast_scale))
    return (r, g, b, a)


def apply_material_control(meshes, material_state: dict, prefix: str) -> list[dict]:
    value_scale = float(material_state.get(f"{prefix}_value_scale", 1.0))
    saturation_scale = float(material_state.get(f"{prefix}_saturation_scale", 1.0))
    contrast_scale = float(material_state.get(f"{prefix}_contrast_scale", 1.0))
    if (
        abs(value_scale - 1.0) < 1e-6
        and abs(saturation_scale - 1.0) < 1e-6
        and abs(contrast_scale - 1.0) < 1e-6
    ):
        return []

    adjusted = []
    seen = set()
    for obj in meshes:
        for slot in getattr(obj, "material_slots", []):
            mat = slot.material
            if mat is None or mat.name in seen:
                continue
            seen.add(mat.name)
            if not getattr(mat, "use_nodes", False) or not mat.node_tree:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    base = node.inputs.get("Base Color")
                    if base is not None and hasattr(base, "default_value"):
                        old = tuple(float(v) for v in base.default_value)
                        base.default_value = _rgb_to_adjusted_rgba(old, value_scale, saturation_scale, contrast_scale)
                        adjusted.append({
                            "material": mat.name,
                            "node": node.name,
                            "input": "Base Color",
                            "old": [round(v, 6) for v in old],
                            "new": [round(float(v), 6) for v in base.default_value],
                        })
                    roughness = node.inputs.get("Roughness")
                    if roughness is not None and hasattr(roughness, "default_value") and contrast_scale > 1.0:
                        old_r = float(roughness.default_value)
                        roughness.default_value = max(0.15, min(1.0, old_r / math.sqrt(contrast_scale)))
                        adjusted.append({
                            "material": mat.name,
                            "node": node.name,
                            "input": "Roughness",
                            "old": round(old_r, 6),
                            "new": round(float(roughness.default_value), 6),
                        })
    bpy.context.view_layer.update()
    return adjusted


def scale_world_environment(state):
    lighting = state.get("lighting") or {}
    env_scale = float(lighting.get("env_strength_scale", 1.0))
    if abs(env_scale - 1.0) < 1e-6:
        return []
    world = bpy.context.scene.world
    if not world or not getattr(world, "use_nodes", False) or not world.node_tree:
        return []
    adjusted = []
    for node in world.node_tree.nodes:
        for input_name in ("Strength", "EV"):
            inp = node.inputs.get(input_name) if hasattr(node, "inputs") else None
            if inp is None or not hasattr(inp, "default_value"):
                continue
            try:
                old = float(inp.default_value)
            except Exception:
                continue
            inp.default_value = old * env_scale
            adjusted.append({"node": node.name, "input": input_name, "old": round(old, 6), "new": round(float(inp.default_value), 6)})
    return adjusted


def scale_existing_lights(state):
    lighting = state.get("lighting") or {}
    fill_scale = float(lighting.get("fill_energy_scale", 1.0))
    for obj in bpy.data.objects:
        if obj.type != "LIGHT":
            continue
        name = obj.name.lower()
        if "spot" in name:
            continue
        if "fill" in name or obj.data.type == "SUN":
            obj.data.energy *= fill_scale


def normalize_scene_lights(state) -> list[dict]:
    lighting = state.get("lighting") or {}
    global_scale = float(lighting.get("scene_light_scale", 1.0))
    type_scales = {
        "AREA": float(lighting.get("scene_area_light_scale", 1.0)),
        "POINT": float(lighting.get("scene_point_light_scale", 1.0)),
        "SPOT": float(lighting.get("scene_spot_light_scale", 1.0)),
        "SUN": float(lighting.get("scene_sun_light_scale", 1.0)),
    }
    max_energy = float(lighting.get("scene_light_max_energy", 500.0))
    adjusted = []
    for obj in bpy.data.objects:
        if obj.type != "LIGHT":
            continue
        if obj.name.startswith("DualPipeline_"):
            continue
        light_type = obj.data.type
        old = float(getattr(obj.data, "energy", 0.0))
        new = old * global_scale * type_scales.get(light_type, 1.0)
        if light_type == "SUN":
            new = min(new, max_energy / 100.0)
        else:
            new = min(new, max_energy)
        if abs(new - old) > 1e-6:
            obj.data.energy = new
            adjusted.append({
                "name": obj.name,
                "type": light_type,
                "old": round(old, 6),
                "new": round(float(new), 6),
            })
    bpy.context.view_layer.update()
    return adjusted


def configure_contact_shadows(state) -> dict:
    shadow = state.get("shadow") or {}
    contact_strength = float(shadow.get("contact_shadow_strength", 1.0))
    ao_strength = float(shadow.get("ambient_occlusion_strength", 1.0))
    scene = bpy.context.scene
    eevee = getattr(scene, "eevee", None)
    if eevee is not None:
        for attr, value in (
            ("use_gtao", True),
            ("gtao_distance", 3.0 * ao_strength),
            ("gtao_factor", 1.5 * ao_strength),
        ):
            if hasattr(eevee, attr):
                try:
                    setattr(eevee, attr, value)
                except Exception:
                    pass

    light_updates = []
    for obj in bpy.data.objects:
        if obj.type != "LIGHT":
            continue
        data = obj.data
        for attr, value in (
            ("use_shadow", True),
            ("use_contact_shadow", True),
            ("contact_shadow_bias", 0.001),
            ("contact_shadow_distance", 2.0 * contact_strength),
            ("shadow_soft_size", max(float(getattr(data, "shadow_soft_size", 0.25)), 0.25 / max(contact_strength, 0.001))),
        ):
            if hasattr(data, attr):
                try:
                    setattr(data, attr, value)
                except Exception:
                    pass
        light_updates.append(obj.name)
    return {
        "contact_shadow_strength": round(contact_strength, 6),
        "ambient_occlusion_strength": round(ao_strength, 6),
        "lights_touched": light_updates,
    }


def configure_cycles_quality(state) -> dict:
    render_state = state.get("render") or {}
    scene = bpy.context.scene
    updates = {}
    if scene.render.engine != "CYCLES" or not hasattr(scene, "cycles"):
        return updates
    samples = int(render_state.get("cycles_samples", 256))
    old_samples = int(getattr(scene.cycles, "samples", samples))
    if old_samples != samples:
        scene.cycles.samples = samples
        updates["samples"] = {"old": old_samples, "new": samples}
    for attr, value in (
        ("use_denoising", bool(render_state.get("use_denoising", True))),
        ("use_fast_gi", bool(render_state.get("use_fast_gi", False))),
    ):
        if hasattr(scene.cycles, attr):
            old = getattr(scene.cycles, attr)
            try:
                setattr(scene.cycles, attr, value)
                if old != value:
                    updates[attr] = {"old": old, "new": value}
            except Exception:
                pass
    if hasattr(scene.cycles, "max_bounces"):
        old_bounces = int(scene.cycles.max_bounces)
        new_bounces = int(render_state.get("max_bounces", max(old_bounces, 6)))
        scene.cycles.max_bounces = new_bounces
        if old_bounces != new_bounces:
            updates["max_bounces"] = {"old": old_bounces, "new": new_bounces}
    return updates


def main():
    args = parse_args()
    state = load_json(args.control_state, default={}) or {}
    seed = args.seed
    if seed is None:
        seed = 0
    seed += int((state.get("scene") or {}).get("seed_offset", 0) or 0)
    random.seed(seed)
    np.random.seed(seed)

    dual.MAX_OBJECT_RATIO = float(args.max_object_ratio)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = out_dir / f"{args.sample_id}.png"

    ground_info = dual.detect_ground_plane()
    scene_min, scene_max = dual.get_scene_bounds()
    scene_size = scene_max - scene_min
    if ground_info is None:
        ground_info = {
            "name": "_fallback",
            "z": 0.0,
            "bounds": [float(scene_min[0]), float(scene_max[0]), float(scene_min[1]), float(scene_max[1])],
            "area": float(scene_size[0] * scene_size[1]),
            "center_xy": [float((scene_min[0] + scene_max[0]) / 2), float((scene_min[1] + scene_max[1]) / 2)],
        }

    scene_bvh = dual.build_scene_bvh()
    root1, meshes1 = dual.import_asset(args.obj_a)
    root2, meshes2 = dual.import_asset(args.obj_b)
    dual.scale_object_to_scene(root1, meshes1, scene_min, scene_max)
    dual.scale_object_to_scene(root2, meshes2, scene_min, scene_max)
    apply_object_control(root1, state.get("object_a") or {}, apply_location=False)
    apply_object_control(root2, state.get("object_b") or {}, apply_location=False)
    material_state = state.get("material") or {}
    material_adjustments = {
        "a": apply_material_control(meshes1, material_state, "a"),
        "b": apply_material_control(meshes2, material_state, "b"),
    }

    placement_state = load_json(args.placement_state_in, default=None)
    if isinstance(placement_state, dict):
        placement = dual.restore_two_object_placement(root1, meshes1, root2, meshes2, placement_state)
    else:
        cam_az = dual.find_hdri_best_azimuth()
        if cam_az is not None:
            cam_az = (cam_az + math.pi) % (2 * math.pi)
        placement = dual.place_two_objects(
            root1,
            meshes1,
            root2,
            meshes2,
            scene_bvh,
            scene_min,
            scene_max,
            ground_info=ground_info,
            template_name=args.template,
            camera_azimuth=cam_az,
        )
    if placement is None:
        save_json(out_dir / "metadata.json", {"success": False, "error": "placement_failed"})
        raise SystemExit(2)
    if args.placement_state_out:
        save_json(args.placement_state_out, placement_to_state(placement, seed=seed, source="generated"))

    apply_pair_control(root1, root2, placement, state)
    post_placement_adjustments = {
        "a": apply_post_placement_control(
            root1,
            meshes1,
            state.get("object_a") or {},
            scene_bvh,
            scene_min,
            scene_max,
            ground_info,
            "a",
        ),
        "b": apply_post_placement_control(
            root2,
            meshes2,
            state.get("object_b") or {},
            scene_bvh,
            scene_min,
            scene_max,
            ground_info,
            "b",
        ),
    }
    placement["bbox1"] = dual.get_object_bbox(meshes1)
    placement["bbox2"] = dual.get_object_bbox(meshes2)
    cam = setup_camera_with_control(placement, scene_min, scene_max, state)
    dual.setup_render(resolution=args.resolution, engine=args.engine)
    render_quality_adjustments = configure_cycles_quality(state)
    scene_light_adjustments = normalize_scene_lights(state)
    shadow_adjustments = configure_contact_shadows(state)
    dual.ensure_minimum_lighting()
    lighting_state = state.get("lighting") or {}
    dual.add_object_spotlight(
        placement,
        intensity=50.0 * float(lighting_state.get("spot_energy_scale", 1.0)),
        spot_size_deg=float(lighting_state.get("spot_size_deg", 45.0)),
        height=float(lighting_state.get("spot_height", 5.0)),
    )
    world_adjustments = scale_world_environment(state)
    scale_existing_lights(state)
    dual.render_to_file(str(rgb_path))

    bbox1 = placement["bbox1"]
    bbox2 = placement["bbox2"]
    center1 = np.array(bbox1["center"], dtype=float)
    center2 = np.array(bbox2["center"], dtype=float)
    metadata = {
        "success": True,
        "sample_id": args.sample_id,
        "rgb_path": str(rgb_path),
        "scene_path": bpy.data.filepath,
        "control_state_path": args.control_state,
        "placement_state_in": args.placement_state_in,
        "placement_state_out": args.placement_state_out,
        "placement_source": placement.get("placement_source", "generated"),
        "control_state": state,
        "ground": ground_info,
        "scene_bounds": {"min": vec_to_list(scene_min), "max": vec_to_list(scene_max), "size": vec_to_list(scene_size)},
        "objects": {
            "a": {
                "asset_path": args.obj_a,
                "root_name": root1.name,
                "mesh_count": len(meshes1),
                "has_image_texture": object_has_image_texture(meshes1),
                "bbox": bbox_to_json(bbox1),
                "material_adjustments": material_adjustments["a"],
                "post_placement_adjustments": post_placement_adjustments["a"],
            },
            "b": {
                "asset_path": args.obj_b,
                "root_name": root2.name,
                "mesh_count": len(meshes2),
                "has_image_texture": object_has_image_texture(meshes2),
                "bbox": bbox_to_json(bbox2),
                "material_adjustments": material_adjustments["b"],
                "post_placement_adjustments": post_placement_adjustments["b"],
            },
        },
        "pair": {
            "template": placement.get("template"),
            "obj1_position": vec_to_list(placement.get("obj1_position")),
            "obj2_position": vec_to_list(placement.get("obj2_position")),
            "separation": round(float(placement.get("separation", 0.0)), 6),
            "height_offset": round(float(placement.get("height_offset", 0.0)), 6),
            "side": int(placement.get("side", 0) or 0),
            "camera_azimuth": round(float(placement.get("camera_azimuth", 0.0)), 10),
            "center_distance": round(float(np.linalg.norm(center2 - center1)), 6),
            "xy_center_distance": round(float(np.linalg.norm((center2 - center1)[:2])), 6),
        },
        "camera": {
            "name": cam.name,
            "location": vec_to_list(cam.location),
            "lens": float(cam.data.lens),
        },
        "lighting": {
            "control": lighting_state,
            "world_adjustments": world_adjustments,
            "scene_light_adjustments": scene_light_adjustments,
            "shadow_adjustments": shadow_adjustments,
            "render_quality_adjustments": render_quality_adjustments,
            "lights": [
                {
                    "name": obj.name,
                    "type": obj.data.type,
                    "energy": round(float(getattr(obj.data, "energy", 0.0)), 6),
                    "location": vec_to_list(obj.location),
                    "spot_size": round(float(getattr(obj.data, "spot_size", 0.0)), 6) if obj.data.type == "SPOT" else None,
                }
                for obj in bpy.data.objects
                if obj.type == "LIGHT"
            ],
        },
    }
    save_json(out_dir / "metadata.json", metadata)
    save_json(out_dir / "render_summary.json", {"success": True, "rgb_path": str(rgb_path), "metadata_path": str(out_dir / "metadata.json")})


if __name__ == "__main__":
    main()
