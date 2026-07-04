"""Create a Blender scene from a validated HYWorld mesh contract.

Run with Blender: blender -b -P scripts/create_hyworld_scene_blend.py -- ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from hyworld_scene_contract import load_scene_contract, selected_support_surface


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Create contract-backed HYWorld Blender scene")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--blend-output", required=True)
    parser.add_argument("--template-output", required=True)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--render-width", type=int, default=832)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--engine", choices=["EEVEE", "CYCLES"], default="EEVEE")
    parser.add_argument("--eevee-samples", type=int, default=32)
    parser.add_argument(
        "--support-only",
        action="store_true",
        help="Keep only the contract support surface from the HYWorld mesh to avoid shell occlusion.",
    )
    parser.add_argument(
        "--support-proxy-surface",
        action="store_true",
        help="Create a clean renderable support surface from the HYWorld contract bounds/height.",
    )
    parser.add_argument("--support-crop-tolerance", type=float, default=0.12)
    parser.add_argument("--support-crop-padding", type=float, default=0.08)
    parser.add_argument(
        "--camera-background",
        choices=["panorama", "dark"],
        default="panorama",
        help="Use HYWorld panorama for camera rays, or keep it lighting-only with a dark camera background.",
    )
    parser.add_argument(
        "--camera-source",
        choices=["worldmirror", "support_anchor"],
        default="worldmirror",
        help="Use the calibrated WorldMirror camera; support_anchor is diagnostic legacy behavior.",
    )
    parser.add_argument("--worldmirror-camera-index", type=int, default=0)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_mesh(path: str):
    before = set(bpy.data.objects.keys())
    suffix = Path(path).suffix.lower()
    if suffix == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=path)
        else:
            bpy.ops.import_mesh.ply(filepath=path)
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        raise ValueError(f"Unsupported HYWorld scene mesh: {path}")
    imported = [bpy.data.objects[name] for name in set(bpy.data.objects.keys()) - before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("HYWorld scene import produced no mesh")
    return meshes


def crop_mesh_to_support(obj, support: dict, tolerance: float, padding: float):
    import bmesh

    bounds = support.get("bounds") or support.get("bounds_xy")
    height = float(support.get("z", support.get("height", 0.0)))
    if not bounds or len(bounds) != 4:
        raise ValueError("Support-only scene creation requires support bounds_xy")
    min_x, max_x, min_y, max_y = [float(v) for v in bounds]
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    matrix = obj.matrix_world.copy()
    delete_faces = []
    kept = 0
    for face in bm.faces:
        center = matrix @ face.calc_center_median()
        keep = (
            min_x <= float(center.x) <= max_x
            and min_y <= float(center.y) <= max_y
            and abs(float(center.z) - height) <= tolerance
        )
        if keep:
            kept += 1
        else:
            delete_faces.append(face)
    if kept < 20:
        bm.free()
        raise RuntimeError(f"Support-only crop kept too few faces: {kept}")
    bmesh.ops.delete(bm, geom=delete_faces, context="FACES")
    loose_verts = [vert for vert in bm.verts if not vert.link_faces]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return kept


def create_support_proxy_surface(support: dict, subdivisions: int = 24):
    bounds = support.get("bounds") or support.get("bounds_xy")
    height = float(support.get("z", support.get("height", 0.0)))
    if not bounds or len(bounds) != 4:
        raise ValueError("Support proxy surface requires support bounds_xy")
    min_x, max_x, min_y, max_y = [float(v) for v in bounds]
    pad_x = max((max_x - min_x) * 0.12, 0.08)
    pad_y = max((max_y - min_y) * 0.12, 0.08)
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y
    subdivisions = max(2, int(subdivisions))
    vertices = []
    for iy in range(subdivisions + 1):
        y = min_y + (max_y - min_y) * iy / subdivisions
        for ix in range(subdivisions + 1):
            x = min_x + (max_x - min_x) * ix / subdivisions
            vertices.append((x, y, height))
    faces = []
    row = subdivisions + 1
    for iy in range(subdivisions):
        for ix in range(subdivisions):
            a = iy * row + ix
            faces.append((a, a + 1, a + row + 1, a + row))
    mesh = bpy.data.meshes.new("HYWorldSupportSurfaceMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(str(support.get("object_name") or "HYWorldReconstruction"), mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def apply_support_proxy_material(obj):
    material = bpy.data.materials.new(f"{obj.name}_support_proxy_material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    noise = nodes.new("ShaderNodeTexNoise")
    ramp = nodes.new("ShaderNodeValToRGB")
    bump = nodes.new("ShaderNodeBump")
    noise.inputs["Scale"].default_value = 18.0
    noise.inputs["Detail"].default_value = 7.0
    noise.inputs["Roughness"].default_value = 0.52
    ramp.color_ramp.elements[0].position = 0.18
    ramp.color_ramp.elements[0].color = (0.16, 0.19, 0.18, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.34, 0.38, 0.36, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.84
    if bsdf.inputs.get("Metallic"):
        bsdf.inputs["Metallic"].default_value = 0.0
    bump.inputs["Strength"].default_value = 0.018
    bump.inputs["Distance"].default_value = 0.035
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(material)


def apply_vertex_color_material(obj):
    material = bpy.data.materials.new(f"{obj.name}_hyworld_material")
    material.use_nodes = True
    material.use_backface_culling = False
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    color_name = None
    if getattr(obj.data, "color_attributes", None):
        color_name = obj.data.color_attributes[0].name
    elif getattr(obj.data, "vertex_colors", None):
        color_name = obj.data.vertex_colors[0].name
    if bsdf:
        bsdf.inputs["Roughness"].default_value = 1.0
        metallic = bsdf.inputs.get("Metallic")
        if metallic:
            metallic.default_value = 0.0
        specular = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
        if specular:
            specular.default_value = 0.0
        if color_name:
            attribute = nodes.new("ShaderNodeAttribute")
            attribute.attribute_name = color_name
            emission_input = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
            if emission_input:
                bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
                links.new(attribute.outputs["Color"], emission_input)
                emission_strength = bsdf.inputs.get("Emission Strength")
                if emission_strength:
                    emission_strength.default_value = 1.0
            else:
                links.new(attribute.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.clear()
    obj.data.materials.append(material)


def setup_panorama_world(path: str | None, camera_background: str = "panorama"):
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("HYWorldLighting")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    environment_background = nodes.new("ShaderNodeBackground")
    environment_background.inputs["Strength"].default_value = 0.65
    if path:
        environment = nodes.new("ShaderNodeTexEnvironment")
        environment.image = bpy.data.images.load(path, check_existing=True)
        links.new(environment.outputs["Color"], environment_background.inputs["Color"])
    if camera_background == "panorama":
        links.new(environment_background.outputs["Background"], output.inputs["Surface"])
        return

    dark_background = nodes.new("ShaderNodeBackground")
    dark_background.inputs["Color"].default_value = (0.03, 0.035, 0.04, 1.0)
    dark_background.inputs["Strength"].default_value = 0.2
    light_path = nodes.new("ShaderNodeLightPath")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(light_path.outputs["Is Camera Ray"], mix.inputs[0])
    links.new(environment_background.outputs["Background"], mix.inputs[1])
    links.new(dark_background.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_anchor_camera(anchor):
    target = Vector(anchor)
    bpy.ops.object.camera_add(location=(target.x, target.y - 4.2, target.z + 1.55))
    camera = bpy.context.object
    camera.name = "Camera"
    camera.data.lens = 45
    look_at(camera, (target.x, target.y, target.z + 0.55))
    bpy.context.scene.camera = camera
    bpy.ops.object.light_add(type="AREA", location=(target.x - 1.8, target.y - 2.4, target.z + 3.2))
    light = bpy.context.object
    light.name = "SceneMatchedKey"
    light.data.energy = 450
    light.data.size = 4.0
    look_at(light, (target.x, target.y, target.z + 0.5))
    return camera, {
        "source": "support_anchor",
        "location": [float(value) for value in camera.location],
    }


def rotation_between(source: Vector, target: Vector) -> Matrix:
    source = source.normalized()
    target = target.normalized()
    rotation = source.rotation_difference(target)
    return rotation.to_matrix()


def load_camera_matrix(entry):
    return Matrix(entry["matrix"] if isinstance(entry, dict) else entry)


def setup_worldmirror_camera(contract: dict, camera_index: int, image_width: int, image_height: int):
    scene_asset = contract.get("scene_asset") or {}
    calibration_path = scene_asset.get("calibration_path")
    camera_path = contract.get("camera_path")
    if not calibration_path or not camera_path:
        raise RuntimeError("WorldMirror camera setup requires calibration_path and camera_path")
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    cameras = json.loads(Path(camera_path).read_text(encoding="utf-8"))
    extrinsics = cameras.get("extrinsics") or []
    intrinsics = cameras.get("intrinsics") or []
    if not extrinsics or len(extrinsics) != len(intrinsics):
        raise RuntimeError("WorldMirror camera_params has missing or unaligned matrices")
    camera_index = int(camera_index)
    if not 0 <= camera_index < len(extrinsics):
        raise IndexError(f"WorldMirror camera index out of range: {camera_index}")

    output_w2c = load_camera_matrix(extrinsics[camera_index])
    output_c2w = output_w2c.inverted()
    hyworld_first_w2c = Matrix(calibration["hyworld_first_w2c"])
    world_alignment = hyworld_first_w2c.inverted()
    ground_normal = Vector(calibration["ground_normal_before_alignment"])
    ground_rotation = rotation_between(ground_normal, Vector((0.0, 0.0, 1.0)))
    ground_height = -float(calibration["ground_plane_offset_before_alignment"])
    metric_scale = float(calibration["metric_scale"])

    output_center = output_c2w.translation
    aligned_center = world_alignment @ output_center
    rotated_center = ground_rotation @ aligned_center
    metric_center = Vector((
        rotated_center.x * metric_scale,
        rotated_center.y * metric_scale,
        (rotated_center.z - ground_height) * metric_scale,
    ))

    # WorldMirror uses the common CV camera frame (+X right, +Y down,
    # +Z forward). Blender camera local axes are +X right, +Y up, -Z view.
    cv_to_blender = Matrix.Diagonal((1.0, -1.0, -1.0))
    metric_camera_rotation = (
        ground_rotation
        @ world_alignment.to_3x3()
        @ output_c2w.to_3x3()
        @ cv_to_blender
    )
    camera_world = metric_camera_rotation.to_4x4()
    camera_world.translation = metric_center

    bpy.ops.object.camera_add(location=metric_center)
    camera = bpy.context.object
    camera.name = "WorldMirrorCamera"
    camera.matrix_world = camera_world
    intrinsic = load_camera_matrix(intrinsics[camera_index]).to_3x3()
    fx = float(intrinsic[0][0])
    fy = float(intrinsic[1][1])
    cx = float(intrinsic[0][2])
    cy = float(intrinsic[1][2])
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36.0
    camera.data.lens = fx * camera.data.sensor_width / float(image_width)
    camera.data.shift_x = (float(image_width) * 0.5 - cx) / float(image_width)
    camera.data.shift_y = (cy - float(image_height) * 0.5) / float(image_width)
    camera.data.clip_start = 0.03
    camera.data.clip_end = 200.0
    bpy.context.scene.camera = camera

    forward = -(metric_camera_rotation @ Vector((0.0, 0.0, 1.0)))
    bpy.ops.object.light_add(
        type="AREA",
        location=metric_center + Vector((-0.6, -0.4, 1.4)),
    )
    light = bpy.context.object
    light.name = "WorldMirrorCameraKey"
    light.data.energy = 180
    light.data.size = 3.0
    look_at(light, metric_center + forward * 3.0)
    return camera, {
        "source": "worldmirror",
        "camera_index": camera_index,
        "camera_path": str(Path(camera_path).resolve()),
        "calibration_path": str(Path(calibration_path).resolve()),
        "location": [round(float(value), 6) for value in metric_center],
        "rotation_matrix": [
            [round(float(metric_camera_rotation[row][col]), 8) for col in range(3)]
            for row in range(3)
        ],
        "intrinsic": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "image_width": int(image_width),
            "image_height": int(image_height),
        },
        "metric_scale": metric_scale,
    }


def main() -> int:
    args = parse_args()
    contract_path = Path(args.contract).resolve()
    contract = load_scene_contract(contract_path)
    support = selected_support_surface(contract)
    clear_scene()
    support_crop_faces = None
    if args.support_proxy_surface:
        meshes = [create_support_proxy_surface(support)]
        apply_support_proxy_material(meshes[0])
    else:
        meshes = import_mesh(contract["scene_asset"]["mesh_path"])
        meshes[0].name = support["object_name"]
        if args.support_only:
            support_crop_faces = crop_mesh_to_support(
                meshes[0],
                support,
                tolerance=float(args.support_crop_tolerance),
                padding=float(args.support_crop_padding),
            )
    for index, obj in enumerate(meshes):
        if index:
            obj.name = f"HYWorldScenePart_{index:03d}"
        if not args.support_proxy_surface:
            apply_vertex_color_material(obj)
    setup_panorama_world((contract.get("environment") or {}).get("panorama_path"), args.camera_background)
    if args.camera_source == "worldmirror":
        _, camera_metadata = setup_worldmirror_camera(
            contract,
            args.worldmirror_camera_index,
            args.render_width,
            args.render_height,
        )
    else:
        _, camera_metadata = setup_anchor_camera(support["anchor_xyz"])
    scene = bpy.context.scene
    scene.render.resolution_x = args.render_width if args.camera_source == "worldmirror" else args.resolution
    scene.render.resolution_y = args.render_height if args.camera_source == "worldmirror" else args.resolution
    scene.render.resolution_percentage = 100
    scene.render.engine = "BLENDER_EEVEE_NEXT" if args.engine == "EEVEE" and bpy.app.version[0] >= 4 else (
        "BLENDER_EEVEE" if args.engine == "EEVEE" else "CYCLES"
    )
    if args.engine == "EEVEE" and hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.eevee_samples
    blend_output = Path(args.blend_output).resolve()
    blend_output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_output), compress=True)
    template = {
        "blend_path": str(blend_output),
        "require_scene_contract": True,
        "scene_contract_path": str(contract_path),
        "support_plane_mode": "ground_object_raycast",
        "support_object_name": support["object_name"],
        "use_existing_world": True,
        "disable_existing_lights": False,
        "scale_existing_lights": 1.0,
        "scale_world_background": 1.0,
        "camera_mode": "scene_camera_match",
        "qc_views": [[0, 0]],
        "render_engine": args.engine,
        "render_resolution": args.resolution,
        "eevee_render_samples": args.eevee_samples,
        "view_transform": "Standard",
        "render_exposure": 0.0,
        "filmic_gamma": 1.0,
        "adaptive_brightness": False,
        "preserve_blend_asset_materials": True,
        "render_depth_normal": True,
        "depth_output_format": "PNG",
        "fallback_support_plane_size": 0,
        "panorama_role": "camera_background" if args.camera_background == "panorama" else "lighting_only",
        "hyworld_mesh_role": (
            "support_proxy_surface"
            if args.support_proxy_surface
            else ("support_only" if args.support_only else "full_reconstruction")
        ),
        "support_crop_faces": support_crop_faces,
        "support_proxy_surface": bool(args.support_proxy_surface),
        "calibrated_camera": camera_metadata,
        "render_width": int(scene.render.resolution_x),
        "render_height": int(scene.render.resolution_y),
    }
    template_output = Path(args.template_output).resolve()
    template_output.parent.mkdir(parents=True, exist_ok=True)
    template_output.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"success": True, "blend": str(blend_output), "template": str(template_output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
