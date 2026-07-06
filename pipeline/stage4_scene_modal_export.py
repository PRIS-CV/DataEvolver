"""
Stage 4 scene modal export: compute geometry metadata and optional depth/normal
passes for a single already-selected scene control state.

This script is designed for derived datasets and must write only to a new
output root. It never updates the original rotation datasets in place.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage4_scene_render as s4


DEFAULT_TEMPLATE_PATH = SCRIPT_DIR.parent / "configs" / "scene_template.json"


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    p = argparse.ArgumentParser(description="Export geometry/depth/normal for a single scene render state")
    p.add_argument("--asset-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--obj-id", required=True)
    p.add_argument("--output-stem", required=True, help="e.g. yaw000")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--engine", choices=["EEVEE", "CYCLES"], default="CYCLES")
    p.add_argument("--control-state", required=True)
    p.add_argument("--scene-template", default=str(DEFAULT_TEMPLATE_PATH))
    p.add_argument("--write-depth-normal", action="store_true")
    p.add_argument("--depth-normal-samples", type=int, default=1)
    return p.parse_args(argv)


def round_float(value, ndigits=6):
    return round(float(value), ndigits)


def vec_to_list(vec, ndigits=6):
    return [round_float(v, ndigits) for v in vec]


def matrix_to_list(mat, ndigits=6):
    return [[round_float(v, ndigits) for v in row] for row in mat]


def collect_world_points(objects):
    points = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        mat = obj.matrix_world
        for v in obj.data.vertices:
            points.append(mat @ v.co)
    return points


def compute_bbox_from_points(points):
    if not points:
        return None
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    min_v = Vector((min(xs), min(ys), min(zs)))
    max_v = Vector((max(xs), max(ys), max(zs)))
    center = (min_v + max_v) / 2.0
    size = max_v - min_v
    return {
        "min": vec_to_list(min_v),
        "max": vec_to_list(max_v),
        "center": vec_to_list(center),
        "size": vec_to_list(size),
    }


def bbox_corners(min_xyz, max_xyz):
    min_v = Vector(min_xyz)
    max_v = Vector(max_xyz)
    return [
        Vector((x, y, z))
        for x in (min_v.x, max_v.x)
        for y in (min_v.y, max_v.y)
        for z in (min_v.z, max_v.z)
    ]


def compute_camera_intrinsics(scene, cam):
    camd = cam.data
    render = scene.render
    scale = render.resolution_percentage / 100.0
    width = render.resolution_x * scale
    height = render.resolution_y * scale
    pixel_aspect = render.pixel_aspect_x / render.pixel_aspect_y

    if camd.type != "PERSP":
        return {
            "camera_model": "unsupported_non_pinhole",
            "image_width": int(width),
            "image_height": int(height),
            "fx": None,
            "fy": None,
            "cx": width / 2.0,
            "cy": height / 2.0,
        }

    sensor_fit = camd.sensor_fit
    if sensor_fit == "AUTO":
        sensor_fit = "HORIZONTAL" if width >= height else "VERTICAL"

    if sensor_fit == "VERTICAL":
        su = width / camd.sensor_width / pixel_aspect
        sv = height / camd.sensor_height
    else:
        su = width / camd.sensor_width
        sv = height * pixel_aspect / camd.sensor_height

    fx = camd.lens * su
    fy = camd.lens * sv
    cx = width / 2.0 - camd.shift_x * width
    cy = height / 2.0 + camd.shift_y * height
    return {
        "camera_model": "pinhole",
        "image_width": int(width),
        "image_height": int(height),
        "fx": round_float(fx),
        "fy": round_float(fy),
        "cx": round_float(cx),
        "cy": round_float(cy),
    }


def project_bbox_to_image(scene, cam, bbox_world, image_width, image_height):
    if bbox_world is None:
        return None
    corners = bbox_corners(bbox_world["min"], bbox_world["max"])
    pixels = []
    for corner in corners:
        co = world_to_camera_view(scene, cam, corner)
        if co.z <= 0.0:
            continue
        x = co.x * image_width
        y = (1.0 - co.y) * image_height
        pixels.append((x, y))
    if not pixels:
        return None
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    return [
        round_float(max(0.0, min(xs))),
        round_float(max(0.0, min(ys))),
        round_float(min(float(image_width), max(xs))),
        round_float(min(float(image_height), max(ys))),
    ]


def configure_pass_nodes(scene, tmp_dir: Path, stem: str):
    scene.use_nodes = True
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True

    tree = scene.node_tree
    tree.nodes.clear()

    rlayers = tree.nodes.new("CompositorNodeRLayers")
    rlayers.location = (0, 0)

    depth_exr = tree.nodes.new("CompositorNodeOutputFile")
    depth_exr.base_path = str(tmp_dir)
    depth_exr.format.file_format = "OPEN_EXR"
    # Blender 4.x File Output node does not accept BW for OpenEXR here.
    # Depth is still linked from the single-channel Z pass; Blender writes it
    # into an EXR-compatible output without requiring a BW enum.
    depth_exr.format.color_mode = "RGB"
    depth_exr.format.color_depth = "32"
    depth_exr.file_slots[0].path = f"{stem}_depth_"
    tree.links.new(rlayers.outputs["Depth"], depth_exr.inputs[0])

    normalize = tree.nodes.new("CompositorNodeNormalize")
    normalize.location = (200, 120)
    tree.links.new(rlayers.outputs["Depth"], normalize.inputs[0])

    depth_png = tree.nodes.new("CompositorNodeOutputFile")
    depth_png.base_path = str(tmp_dir)
    depth_png.format.file_format = "PNG"
    depth_png.format.color_mode = "BW"
    depth_png.file_slots[0].path = f"{stem}_depth_vis_"
    depth_png.location = (420, 120)
    tree.links.new(normalize.outputs[0], depth_png.inputs[0])

    normal_exr = tree.nodes.new("CompositorNodeOutputFile")
    normal_exr.base_path = str(tmp_dir)
    normal_exr.format.file_format = "OPEN_EXR"
    normal_exr.format.color_mode = "RGB"
    normal_exr.format.color_depth = "32"
    normal_exr.file_slots[0].path = f"{stem}_normal_"
    normal_exr.location = (420, -80)
    tree.links.new(rlayers.outputs["Normal"], normal_exr.inputs[0])

    mul = tree.nodes.new("CompositorNodeMixRGB")
    mul.blend_type = "MULTIPLY"
    mul.inputs[0].default_value = 1.0
    mul.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
    mul.location = (200, -220)
    tree.links.new(rlayers.outputs["Normal"], mul.inputs[1])

    add = tree.nodes.new("CompositorNodeMixRGB")
    add.blend_type = "ADD"
    add.inputs[0].default_value = 1.0
    add.inputs[2].default_value = (0.5, 0.5, 0.5, 0.0)
    add.location = (420, -220)
    tree.links.new(mul.outputs[0], add.inputs[1])

    normal_png = tree.nodes.new("CompositorNodeOutputFile")
    normal_png.base_path = str(tmp_dir)
    normal_png.format.file_format = "PNG"
    normal_png.format.color_mode = "RGB"
    normal_png.file_slots[0].path = f"{stem}_normal_vis_"
    normal_png.location = (650, -220)
    tree.links.new(add.outputs[0], normal_png.inputs[0])


def move_render_output(tmp_dir: Path, prefix: str, suffix: str, dst_path: Path):
    matches = sorted(tmp_dir.glob(f"{prefix}*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"Missing compositor output for {prefix}*{suffix}")
    src = matches[0]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()
    src.replace(dst_path)
    return dst_path


def render_depth_and_normal(scene, out_dir: Path, stem: str, sample_count: int):
    tmp_dir = out_dir / f"__tmp_modal_{stem}"
    if tmp_dir.exists():
        for child in tmp_dir.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if "CYCLES" in scene.render.engine:
        scene.cycles.samples = int(sample_count)
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_denoising = False

    configure_pass_nodes(scene, tmp_dir, stem)
    scene.render.filepath = str(tmp_dir / f"{stem}_combined.png")
    bpy.ops.render.render(write_still=False)

    depth_exr = move_render_output(tmp_dir, f"{stem}_depth_", ".exr", out_dir / f"{stem}_depth.exr")
    depth_vis = move_render_output(tmp_dir, f"{stem}_depth_vis_", ".png", out_dir / f"{stem}_depth_vis.png")
    normal_exr = move_render_output(tmp_dir, f"{stem}_normal_", ".exr", out_dir / f"{stem}_normal.exr")
    normal_png = move_render_output(tmp_dir, f"{stem}_normal_vis_", ".png", out_dir / f"{stem}_normal.png")

    for child in list(tmp_dir.iterdir()):
        if child.is_file() or child.is_symlink():
            child.unlink()
    tmp_dir.rmdir()
    return {
        "depth_exr": str(depth_exr),
        "depth_vis": str(depth_vis),
        "normal_exr": str(normal_exr),
        "normal_png": str(normal_png),
    }


def render_modal(asset_path: str, obj_id: str, output_dir: str, output_stem: str,
                 resolution: int, engine: str, template: dict, control_state: dict,
                 write_depth_normal: bool, depth_normal_samples: int,
                 control_state_path: str):
    blend_path = template["blend_path"]
    if not os.path.exists(blend_path):
        raise FileNotFoundError(f"Scene blend not found: {blend_path}")

    bpy.ops.wm.open_mainfile(filepath=blend_path)
    scene = bpy.context.scene
    s4.setup_render_settings(scene, resolution, engine, template)

    if template.get("disable_existing_lights", False):
        s4.disable_existing_lights()
    else:
        s4.scale_existing_lights(float(template.get("scale_existing_lights", 0.3)))
        s4.scale_world_background(scene, float(template.get("scale_world_background", 0.3)))

    imported_info = s4.import_asset(asset_path)
    asset_kind = imported_info["asset_kind"]
    mesh_objects = imported_info["mesh_objects"]
    root = imported_info["root"]
    imported_names = imported_info["imported_names"]

    control_object = control_state.get("object", {})
    control_scene = control_state.get("scene", {})
    control_lighting = control_state.get("lighting", {})
    control_material = control_state.get("material", {})
    camera_mode = template.get("camera_mode", "scene_bounds_relative")
    s4.configure_existing_scene_lights(template, control_scene)

    support_plane_mode = template.get("support_plane_mode", "ground_object_raycast")
    extra_meta = {}

    if support_plane_mode == "ground_object_raycast":
        support_name_hint = template.get("support_object_name", "ground")
        ground_obj, ground_detection_mode = s4.find_ground_object(support_name_hint)
        if ground_obj:
            s4.fix_ground_orientation(ground_obj)
            object_scale_mode = template.get("object_scale_mode", "height_range")
            if object_scale_mode == "scene_relative":
                bounds_min, bounds_max = s4.get_scene_bounds_all(ignore_names=imported_names)
                s4.auto_adjust_object_size_scene_relative(root, mesh_objects, bounds_min, bounds_max)
                auto_scale = 1.0
            else:
                _, auto_scale = s4.fit_object_height(
                    root, mesh_objects, template.get("target_bbox_height_range", [0.3, 1.5])
                )

            manual_scale = float(control_object.get("scale", 1.0))
            if abs(manual_scale - 1.0) > 1e-6:
                root.scale = tuple(v * manual_scale for v in root.scale)
                bpy.context.view_layer.update()

            ground_bbox = [ground_obj.matrix_world @ Vector(c) for c in ground_obj.bound_box]
            ground_cx = sum(v.x for v in ground_bbox) / 8.0
            ground_cy = sum(v.y for v in ground_bbox) / 8.0
            ground_z_rough = min(v.z for v in ground_bbox)
            anchor_mode = "ground_center"
            anchor_point = None
            if camera_mode == "scene_camera_match":
                scene_cam = s4.resolve_scene_camera(scene)
                anchor_point = s4.camera_screen_ground_anchor(
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

            root.rotation_euler.z += math.radians(float(control_object.get("yaw_deg", 0.0)))
            bpy.context.view_layer.update()

            bbox = s4.percentile_bbox(mesh_objects)
            if bbox:
                bottom_z = float(bbox["min"][2])
                root.location.z += ground_z_rough - bottom_z
                bpy.context.view_layer.update()

            ground_hit = s4.adjust_object_to_ground_with_ray(root, mesh_objects, ground_obj)
            root.location.z += float(control_object.get("offset_z", 0.0))
            ground_contact_epsilon = s4.compute_ground_contact_epsilon(mesh_objects, template)
            root.location.z += ground_contact_epsilon
            bpy.context.view_layer.update()

            ground_z = float(ground_hit.get("support_z")) if ground_hit.get("support_z") is not None else min(v.z for v in ground_bbox)
            grounding_stabilization = s4.stabilize_ground_contact(
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
                "ground_contact_epsilon": round_float(ground_contact_epsilon),
                "grounding_stabilization": grounding_stabilization,
                "placement_anchor_mode": anchor_mode,
                "placement_anchor_xy": [round_float(root.location.x), round_float(root.location.y)],
            }
        else:
            support = s4.create_hidden_support_plane(float(template.get("fallback_support_plane_size", 10.0)))
            auto_scale = s4.align_object_to_support(
                root, mesh_objects, support, control_object,
                template.get("target_bbox_height_range", [0.3, 1.5])
            )
            extra_meta = {
                "ground_object_name": None,
                "ground_detection_mode": "fallback_plane",
                "ground_hit_success": False,
            }
    else:
        support = s4.ensure_support_plane(template, ignore_names=imported_names)
        auto_scale = s4.align_object_to_support(
            root, mesh_objects, support, control_object,
            template.get("target_bbox_height_range", [0.3, 1.5])
        )

    preserve_blend_materials = bool(template.get("preserve_blend_asset_materials", True))
    should_preserve_materials = asset_kind == "blend" and preserve_blend_materials
    material_adaptation = {
        "applied": False,
        "used_mode": None,
        "had_image_texture": any(s4._object_has_image_texture(obj) for obj in mesh_objects),
        "forced_override": False,
        "reason": "blend_asset_materials_preserved" if should_preserve_materials else "not_requested",
    }
    if not should_preserve_materials:
        force_reference_material = s4._should_force_reference_material(
            control_material,
            material_adaptation["had_image_texture"],
        )
        material_adaptation = s4.ensure_reference_material(
            mesh_objects,
            reference_rgb=control_material.get("reference_rgb"),
            roughness_add=float(control_material.get("roughness_add", 0.0)),
            specular_add=float(control_material.get("specular_add", 0.0)),
            force_reference_material=force_reference_material,
            reference_material_mode=str(control_material.get("reference_material_mode", "auto")),
        )
        s4.sanitize_materials(
            mesh_objects,
            roughness_add=float(control_material.get("roughness_add", 0.0)),
            specular_add=float(control_material.get("specular_add", 0.0)),
        )
        s4.adjust_material_hsv(
            mesh_objects,
            saturation_scale=float(control_material.get("saturation_scale", 1.0)),
            value_scale=float(control_material.get("value_scale", 1.0)),
            hue_offset=float(control_material.get("hue_offset", 0.0)),
        )

    env_meta = s4.ensure_world_environment(scene, template, control_scene)
    target_bbox = s4.percentile_bbox(mesh_objects)
    target = Vector(target_bbox["center"]) if target_bbox else Vector((0.0, 0.0, 0.5))

    if template.get("add_key_light", False):
        s4.create_key_light(target, template, control_lighting, control_scene)

    camera_target = target
    camera_name = None
    if camera_mode == "scene_camera_match":
        cam = s4.resolve_scene_camera(scene)
        camera_name = cam.name
        camera_dist = None
        camera_target = None
    else:
        cam = s4.create_qc_camera(scene)
        if camera_mode == "object_bbox_relative" and target_bbox:
            camera_dist, camera_target = s4.compute_object_camera_params(target_bbox, template)
        elif camera_mode == "scene_bounds_relative":
            bounds_min, bounds_max = s4.get_scene_bounds_all(ignore_names=imported_names)
            camera_dist, _ = s4.compute_scene_camera_params(bounds_min, bounds_max, target, template)
        else:
            camera_dist = float(template.get("camera_distance", 3.5))
        camera_name = cam.name

    if template.get("add_camera_fill_light", False):
        s4.create_camera_fill_light(cam, target, target_bbox, template, control_scene)

    qc_views = [(int(v[0]), int(v[1])) for v in template.get("qc_views", s4.DEFAULT_QC_VIEWS)]
    if template.get("adaptive_brightness", False):
        if camera_mode != "scene_camera_match":
            preview_az, preview_el = qc_views[0] if qc_views else s4.DEFAULT_QC_VIEWS[0]
            target_bbox = s4.percentile_bbox(mesh_objects)
            preview_target = Vector(target_bbox["center"]) if target_bbox else Vector((0.0, 0.0, 0.5))
            if camera_mode == "object_bbox_relative" and target_bbox:
                preview_dist, preview_target = s4.compute_object_camera_params(target_bbox, template)
            elif camera_mode == "scene_bounds_relative":
                bounds_min, bounds_max = s4.get_scene_bounds_all(ignore_names=imported_names)
                preview_dist, _ = s4.compute_scene_camera_params(bounds_min, bounds_max, preview_target, template)
            else:
                preview_dist = float(template.get("camera_distance", 3.5))
            s4.position_camera(cam, preview_target, preview_az, preview_el, preview_dist)
        s4.adjust_lighting_to_target_brightness(
            target_brightness=float(template.get("target_brightness", 0.4)),
            preview_samples=int(template.get("cycles_preview_samples", 32)),
            damping=float(template.get("adaptive_brightness_damping", 0.45)),
            min_gain=float(template.get("adaptive_brightness_min_gain", 0.82)),
            max_gain=float(template.get("adaptive_brightness_max_gain", 1.18)),
            world_gain_scale=float(template.get("adaptive_world_gain_scale", 0.55)),
        )

    az, el = qc_views[0] if qc_views else (0, 0)
    target_bbox = s4.percentile_bbox(mesh_objects)
    target = Vector(target_bbox["center"]) if target_bbox else Vector((0.0, 0.0, 0.5))
    if camera_mode != "scene_camera_match":
        if camera_mode == "object_bbox_relative" and target_bbox:
            camera_dist, camera_target = s4.compute_object_camera_params(target_bbox, template)
        else:
            camera_target = target
        s4.position_camera(cam, camera_target, az, el, camera_dist)

    obj_out = Path(output_dir).resolve()
    obj_out.mkdir(parents=True, exist_ok=True)

    intr = compute_camera_intrinsics(scene, cam)
    cam_to_world = cam.matrix_world.copy()
    world_to_cam = cam.matrix_world.inverted()
    cam_rot = cam.matrix_world.to_quaternion()
    camera_forward = cam_rot @ Vector((0.0, 0.0, -1.0))
    camera_up = cam_rot @ Vector((0.0, 1.0, 0.0))

    world_points = collect_world_points(mesh_objects)
    world_bbox = compute_bbox_from_points(world_points)
    world_to_object = root.matrix_world.inverted()
    object_points = [world_to_object @ p for p in world_points]
    object_bbox = compute_bbox_from_points(object_points)
    bbox2d = project_bbox_to_image(scene, cam, world_bbox, intr["image_width"], intr["image_height"])
    center_world = Vector(world_bbox["center"]) if world_bbox else Vector((0.0, 0.0, 0.0))
    center_camera = cam.matrix_world.inverted() @ center_world.to_4d()
    object_rot = root.matrix_world.to_euler("XYZ")

    geometry_metadata = {
        "obj_id": obj_id,
        "output_stem": output_stem,
        "camera_model": intr["camera_model"],
        "image_width": intr["image_width"],
        "image_height": intr["image_height"],
        "fx": intr["fx"],
        "fy": intr["fy"],
        "cx": intr["cx"],
        "cy": intr["cy"],
        "camera_to_world_4x4": matrix_to_list(cam_to_world),
        "world_to_camera_4x4": matrix_to_list(cam.matrix_world.inverted()),
        "camera_position_xyz": vec_to_list(cam.matrix_world.translation),
        "camera_forward_xyz": vec_to_list(camera_forward),
        "camera_up_xyz": vec_to_list(camera_up),
        "object_to_world_4x4": matrix_to_list(root.matrix_world),
        "world_to_object_4x4": matrix_to_list(root.matrix_world.inverted()),
        "object_position_xyz": vec_to_list(root.matrix_world.translation),
        "object_rotation_euler_xyz": [round_float(object_rot.x), round_float(object_rot.y), round_float(object_rot.z)],
        "object_scale_xyz": [round_float(v) for v in root.scale],
        "object_yaw_deg": round_float(float(control_object.get("yaw_deg", 0.0))),
        "support_plane_z": support.get("z"),
        "camera_mode": camera_mode,
        "placement_anchor_xy": extra_meta.get("placement_anchor_xy"),
        "ground_contact_epsilon": extra_meta.get("ground_contact_epsilon"),
        "bbox_3d_world": world_bbox,
        "bbox_3d_object": object_bbox,
        "bbox_2d_xyxy": bbox2d,
        "object_center_world": world_bbox["center"] if world_bbox else None,
        "object_center_camera": [
            round_float(center_camera.x),
            round_float(center_camera.y),
            round_float(center_camera.z),
        ] if world_bbox else None,
        "source_asset": asset_path,
        "camera_name": camera_name,
        "base_render_context": {
            "blend_path": blend_path,
            "resolution": resolution,
            "engine": engine,
            "control_state_path": str(Path(control_state_path).resolve()),
        },
    }

    geom_path = obj_out / f"{output_stem}_geometry_metadata.json"
    with geom_path.open("w", encoding="utf-8") as f:
        json.dump(geometry_metadata, f, indent=2, ensure_ascii=False)

    modal_outputs = None
    if write_depth_normal:
        modal_outputs = render_depth_and_normal(scene, obj_out, output_stem, depth_normal_samples)

    summary = {
        "obj_id": obj_id,
        "output_stem": output_stem,
        "geometry_metadata_path": str(geom_path),
        "wrote_depth_normal": bool(write_depth_normal),
        "modal_outputs": modal_outputs,
        "physics": s4.compute_physics_metrics(mesh_objects, support),
        "environment": env_meta,
        "material_adaptation": material_adaptation,
        **extra_meta,
    }
    with (obj_out / f"{output_stem}_modal_export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    global args
    args = parse_args()
    template = s4.load_scene_template(args.scene_template)
    control_state = s4.load_json(args.control_state, default={}) or {}
    summary = render_modal(
        asset_path=args.asset_path,
        obj_id=args.obj_id,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        resolution=args.resolution,
        engine=args.engine,
        template=template,
        control_state=control_state,
        write_depth_normal=args.write_depth_normal,
        depth_normal_samples=args.depth_normal_samples,
        control_state_path=args.control_state,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
