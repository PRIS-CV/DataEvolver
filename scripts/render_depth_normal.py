"""
Blender script: generate depth and normal maps for all 8 yaw views of one object.

Usage:
  blender --background --python render_depth_normal.py -- \
    --obj-id obj_1471 \
    --runtime-root <repo-root>/runtime/new_asset_force_rot8_20260429_203842 \
    --output-root <repo-root>/runtime/new_asset_force_rot8_20260429_203842/trainready

Per-object flow:
  1. Open blend file, import GLB, set up scene (once)
  2. For each yaw in [0,45,90,135,180,225,270,315]:
     a. Apply yaw rotation
     b. Re-ground the object
     c. Render depth + normal passes
     d. Save depth.png, normal.png to trainready/views/<obj_id>/
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from statistics import median

import bpy
from mathutils import Matrix, Vector

# ── Paths ───────────────────────────────────────────────────────────
SCRIPT_DIR = "<repo-root>/scripts"
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

YAW_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

# ══════════════════════════════════════════════════════════════════════
# Minimal helpers from stage4_scene_render (inlined for independence)
# ══════════════════════════════════════════════════════════════════════

def load_json(path: str, default=None):
    if not path or not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _percentile(values, q):
    """Pure-Python percentile (linear interpolation)."""
    if not values:
        return 0.0
    sv = sorted(values)
    n = len(sv)
    if n == 1:
        return sv[0]
    k = (q / 100.0) * (n - 1)
    f = int(k)
    c = k - f
    if f + 1 < n:
        return sv[f] + c * (sv[f + 1] - sv[f])
    return sv[f]


def percentile_bbox(objects, q_lo: float = 1.0, q_hi: float = 99.0):
    xs, ys, zs = [], [], []
    for obj in objects:
        if obj.type != "MESH":
            continue
        mat = obj.matrix_world
        for v in obj.data.vertices:
            w = mat @ v.co
            xs.append(w.x)
            ys.append(w.y)
            zs.append(w.z)
    if not xs:
        return None
    lo = [_percentile(xs, q_lo), _percentile(ys, q_lo), _percentile(zs, q_lo)]
    hi = [_percentile(xs, q_hi), _percentile(ys, q_hi), _percentile(zs, q_hi)]
    center = [(lo[i] + hi[i]) / 2.0 for i in range(3)]
    size = [hi[i] - lo[i] for i in range(3)]
    return {"min": lo, "max": hi, "center": center, "size": size}


def import_glb(glb_path: str):
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=glb_path)
    after = set(bpy.data.objects.keys())
    return [bpy.data.objects[name] for name in (after - before)]


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


# ── Ground helpers ───────────────────────────────────────────────────

def find_ground_object(support_name_hint: str = "ground"):
    candidates = [obj for obj in bpy.data.objects if obj.type == "MESH" and not obj.hide_get()]
    for obj in candidates:
        if obj.name.lower() == support_name_hint.lower():
            return obj, "name_match"
    lowest_z = float("inf")
    ground_obj = None
    for obj in candidates:
        bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        xs = [v.x for v in bbox]; ys = [v.y for v in bbox]; zs = [v.z for v in bbox]
        size_x = max(xs) - min(xs); size_y = max(ys) - min(ys); size_z = max(zs) - min(zs)
        max_xy = max(size_x, size_y, 0.001)
        if size_z > 0.5 * max_xy:
            continue
        z = min(zs)
        if z < lowest_z:
            lowest_z = z
            ground_obj = obj
    if ground_obj is not None:
        return ground_obj, "lowest_horizontal_bbox"
    for obj in candidates:
        bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        z = min(v.z for v in bbox)
        if z < lowest_z:
            lowest_z = z
            ground_obj = obj
    return ground_obj, "lowest_bbox"


def fix_ground_orientation(ground_obj):
    if ground_obj.scale.z < 0:
        ground_obj.scale.z = abs(ground_obj.scale.z)
    rx = math.degrees(ground_obj.rotation_euler.x)
    if abs(rx) in [90, 180, 270]:
        ground_obj.rotation_euler.x = 0.0


def adjust_object_to_ground_with_ray(root, mesh_objects, ground_obj):
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    pts = []
    for obj in mesh_objects:
        mat = obj.matrix_world
        for corner in obj.bound_box:
            pts.append(mat @ Vector(corner))
    if not pts:
        return {"success": False, "sample_count": 0, "hit_count": 0, "support_z": None}
    min_z = min(v.z for v in pts)
    max_z = max(v.z for v in pts)
    center_x = sum(v.x for v in pts) / len(pts)
    center_y = sum(v.y for v in pts) / len(pts)
    xs = [v.x for v in pts]; ys = [v.y for v in pts]
    x_lo = float(_percentile(xs, 18.0)); x_hi = float(_percentile(xs, 82.0))
    y_lo = float(_percentile(ys, 18.0)); y_hi = float(_percentile(ys, 82.0))
    sample_xy = [(center_x, center_y), (x_lo, y_lo), (x_lo, y_hi), (x_hi, y_lo), (x_hi, y_hi)]
    ray_origin_z = min_z + max(0.01, (max_z - min_z) * 0.06)
    ray_dir = Vector((0, 0, -1))
    hit_locations = []
    for sx, sy in sample_xy:
        ray_origin = Vector((sx, sy, ray_origin_z))
        hit, location, _, _, hit_obj, _ = scene.ray_cast(depsgraph, ray_origin, ray_dir, distance=2.0)
        if hit and hit_obj == ground_obj:
            hit_locations.append(float(location.z))
    if not hit_locations:
        return {"success": False, "sample_count": len(sample_xy), "hit_count": 0, "support_z": None}
    hit_locations.sort()
    pick_idx = min(len(hit_locations) - 1, max(0, math.ceil(len(hit_locations) * 0.75) - 1))
    support_z = hit_locations[pick_idx]
    delta_z = support_z - min_z
    root.location.z += delta_z
    bpy.context.view_layer.update()
    return {"success": True, "sample_count": len(sample_xy), "hit_count": len(hit_locations), "support_z": float(support_z)}


def compute_ground_contact_epsilon(objects, base_epsilon: float = 0.0014) -> float:
    bbox = percentile_bbox(objects)
    if bbox is None:
        return max(0.0005, min(0.0025, base_epsilon))
    bbox_height = max(float(bbox["size"][2]), 1e-3)
    adaptive_epsilon = bbox_height * 0.0018
    return max(0.0005, min(0.0025, max(base_epsilon * 0.55, adaptive_epsilon)))


# ── Material helpers ─────────────────────────────────────────────────

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


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


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


def _set_first_existing_input(node, candidate_names, value):
    for name in candidate_names:
        sock = node.inputs.get(name)
        if sock is not None:
            sock.default_value = value
            return True
    return False


def _build_reference_wood_material(name: str, reference_rgb, roughness_add=0.0, specular_add=0.0):
    base = [_clamp01(v) for v in reference_rgb[:3]]
    dark_rgb = tuple(_clamp01(v * 0.2 + 0.008) for v in base) + (1.0,)
    mid_rgb = tuple(_clamp01(v * 0.44 + 0.014) for v in base) + (1.0,)
    light_rgb = tuple(_clamp01(v * 0.82 + 0.022) for v in base) + (1.0,)
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes; links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial"); bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    texcoord = nodes.new("ShaderNodeTexCoord"); mapping = nodes.new("ShaderNodeMapping")
    noise = nodes.new("ShaderNodeTexNoise"); wave = nodes.new("ShaderNodeTexWave")
    mix_fac = nodes.new("ShaderNodeMixRGB"); grain_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp = nodes.new("ShaderNodeValToRGB"); bump = nodes.new("ShaderNodeBump")
    mapping.inputs["Scale"].default_value = (2.6, 1.0, 8.8)
    noise.inputs["Scale"].default_value = 5.5; noise.inputs["Detail"].default_value = 12.0
    noise.inputs["Roughness"].default_value = 0.48
    wave.wave_type = "BANDS"; wave.bands_direction = "Z"
    wave.inputs["Scale"].default_value = 9.0; wave.inputs["Distortion"].default_value = 3.0
    wave.inputs["Detail"].default_value = 4.0
    mix_fac.blend_type = "ADD"; mix_fac.inputs["Fac"].default_value = 0.5
    grain_ramp.color_ramp.elements[0].position = 0.3; grain_ramp.color_ramp.elements[0].color = (0.08, 0.08, 0.08, 1.0)
    grain_ramp.color_ramp.elements[1].position = 0.77; grain_ramp.color_ramp.elements[1].color = (0.92, 0.92, 0.92, 1.0)
    color_ramp.color_ramp.elements[0].position = 0.16; color_ramp.color_ramp.elements[0].color = dark_rgb
    color_ramp.color_ramp.elements[1].position = 0.9; color_ramp.color_ramp.elements[1].color = light_rgb
    mid_elem = color_ramp.color_ramp.elements.new(0.52); mid_elem.color = mid_rgb
    bump.inputs["Strength"].default_value = 0.09; bump.inputs["Distance"].default_value = 0.08
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


def _build_reference_ceramic_material(name: str, reference_rgb, roughness_add=0.0, specular_add=0.0):
    base = [_clamp01(v) for v in reference_rgb[:3]]
    darker = tuple(_clamp01(v * 0.78 + 0.01) for v in base) + (1.0,)
    lighter = tuple(_clamp01(v * 1.05 + 0.015) for v in base) + (1.0,)
    speck = tuple(_clamp01(v * 0.92 + 0.03) for v in base) + (1.0,)
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes; links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial"); bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    texcoord = nodes.new("ShaderNodeTexCoord"); mapping = nodes.new("ShaderNodeMapping")
    noise_large = nodes.new("ShaderNodeTexNoise"); noise_small = nodes.new("ShaderNodeTexNoise")
    large_ramp = nodes.new("ShaderNodeValToRGB"); speck_ramp = nodes.new("ShaderNodeValToRGB")
    mix_rgb = nodes.new("ShaderNodeMixRGB"); bump = nodes.new("ShaderNodeBump")
    mapping.inputs["Scale"].default_value = (2.2, 2.2, 2.2)
    noise_large.inputs["Scale"].default_value = 5.0; noise_large.inputs["Detail"].default_value = 6.0
    noise_large.inputs["Roughness"].default_value = 0.42
    noise_small.inputs["Scale"].default_value = 58.0; noise_small.inputs["Detail"].default_value = 2.0
    noise_small.inputs["Roughness"].default_value = 0.25
    large_ramp.color_ramp.elements[0].position = 0.26; large_ramp.color_ramp.elements[0].color = darker
    large_ramp.color_ramp.elements[1].position = 0.88; large_ramp.color_ramp.elements[1].color = lighter
    speck_ramp.color_ramp.elements[0].position = 0.74; speck_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    speck_ramp.color_ramp.elements[1].position = 0.93; speck_ramp.color_ramp.elements[1].color = speck
    mix_rgb.blend_type = "ADD"; mix_rgb.inputs["Fac"].default_value = 0.12
    bump.inputs["Strength"].default_value = 0.018; bump.inputs["Distance"].default_value = 0.03
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


def _build_reference_plastic_material(name: str, reference_rgb, roughness_add=0.0, specular_add=0.0):
    base = [_clamp01(v) for v in reference_rgb[:3]]
    dark_rgb = tuple(_clamp01(v * 0.68 + 0.015) for v in base) + (1.0,)
    mid_rgb = tuple(_clamp01(v * 0.92 + 0.02) for v in base) + (1.0,)
    bright_rgb = tuple(_clamp01(v * 1.08 + 0.03) for v in base) + (1.0,)
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes; links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial"); bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    texcoord = nodes.new("ShaderNodeTexCoord"); mapping = nodes.new("ShaderNodeMapping")
    noise_large = nodes.new("ShaderNodeTexNoise"); noise_small = nodes.new("ShaderNodeTexNoise")
    grime_mix = nodes.new("ShaderNodeMixRGB"); color_ramp = nodes.new("ShaderNodeValToRGB")
    speck_ramp = nodes.new("ShaderNodeValToRGB"); bump = nodes.new("ShaderNodeBump")
    mapping.inputs["Scale"].default_value = (1.8, 1.8, 1.8)
    noise_large.inputs["Scale"].default_value = 3.2; noise_large.inputs["Detail"].default_value = 7.0
    noise_large.inputs["Roughness"].default_value = 0.44
    noise_small.inputs["Scale"].default_value = 42.0; noise_small.inputs["Detail"].default_value = 3.0
    noise_small.inputs["Roughness"].default_value = 0.34
    color_ramp.color_ramp.elements[0].position = 0.24; color_ramp.color_ramp.elements[0].color = dark_rgb
    color_ramp.color_ramp.elements[1].position = 0.88; color_ramp.color_ramp.elements[1].color = bright_rgb
    mid_elem = color_ramp.color_ramp.elements.new(0.56); mid_elem.color = mid_rgb
    speck_ramp.color_ramp.elements[0].position = 0.72; speck_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    speck_ramp.color_ramp.elements[1].position = 0.94; speck_ramp.color_ramp.elements[1].color = (0.26, 0.22, 0.18, 1.0)
    grime_mix.blend_type = "MULTIPLY"; grime_mix.inputs["Fac"].default_value = 0.18
    bump.inputs["Strength"].default_value = 0.02; bump.inputs["Distance"].default_value = 0.02
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


def ensure_reference_material(objects, reference_rgb=None, roughness_add=0.0, specular_add=0.0,
                              force_reference_material=False, reference_material_mode="auto"):
    if not reference_rgb:
        return {"applied": False, "used_mode": None, "had_image_texture": False, "forced_override": False, "reason": "missing_reference_rgb"}
    had_image_texture = any(_object_has_image_texture(obj) for obj in objects)
    if had_image_texture and not force_reference_material:
        return {"applied": False, "used_mode": None, "had_image_texture": True, "forced_override": False, "reason": "textured_mesh_preserved"}
    mode = (reference_material_mode or "auto").strip().lower()
    if mode == "auto":
        mode = "wood"
    if mode == "ceramic":
        builder = _build_reference_ceramic_material
    elif mode == "plastic":
        builder = _build_reference_plastic_material
    else:
        builder = _build_reference_wood_material
    fallback = builder(f"__ARISRef_{mode.title()}__", reference_rgb=reference_rgb,
                       roughness_add=roughness_add, specular_add=specular_add)
    for obj in objects:
        if obj.type != "MESH":
            continue
        if len(obj.material_slots) == 0:
            obj.data.materials.append(fallback)
        for idx in range(len(obj.material_slots)):
            obj.material_slots[idx].material = fallback
    return {"applied": True, "used_mode": mode, "had_image_texture": had_image_texture,
            "forced_override": bool(had_image_texture and force_reference_material), "reason": "reference_override_applied"}


def adjust_material_hsv(objects, saturation_scale=1.0, value_scale=1.0, hue_offset=0.0):
    if abs(saturation_scale - 1.0) < 1e-6 and abs(value_scale - 1.0) < 1e-6 and abs(hue_offset) < 1e-6:
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


# ── Camera helpers ───────────────────────────────────────────────────

def resolve_scene_camera(scene):
    if scene.camera is not None:
        return scene.camera
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            scene.camera = obj
            return obj
    bpy.ops.object.camera_add(location=(0, 0, 0))
    cam = bpy.context.active_object
    cam.name = "V7QCCamera"
    cam.data.lens = 50
    cam.data.clip_start = 0.1
    cam.data.clip_end = 100.0
    scene.camera = cam
    return cam


def camera_screen_ground_anchor(scene, camera, ground_z: float, u: float = 0.5, v: float = 0.22):
    frame_local = list(camera.data.view_frame(scene=scene))
    if len(frame_local) != 4:
        return None
    bottom = sorted(frame_local, key=lambda p: p.y)[:2]
    top = sorted(frame_local, key=lambda p: p.y)[-2:]
    bl, br = sorted(bottom, key=lambda p: p.x)
    tl, tr = sorted(top, key=lambda p: p.x)
    local_point = (1 - u) * (1 - v) * bl + u * (1 - v) * br + (1 - u) * v * tl + u * v * tr
    origin = camera.matrix_world.translation.copy()
    world_point = camera.matrix_world @ local_point
    direction = (world_point - origin).normalized()
    if abs(direction.z) < 1e-6:
        return None
    t = (ground_z - origin.z) / direction.z
    if t <= 0:
        return None
    return origin + direction * t


# ── Light helpers ────────────────────────────────────────────────────

def _configure_contact_shadow(light_obj, contact_shadow_strength: float = 1.0):
    if light_obj is None or getattr(light_obj, "type", None) != "LIGHT":
        return
    light_data = light_obj.data
    if light_data is None:
        return
    if hasattr(light_data, "use_contact_shadow"):
        light_data.use_contact_shadow = True
    strength_scale = max(0.6, float(contact_shadow_strength))
    base_distance = 0.04
    tuned_distance = base_distance / (strength_scale ** 0.85)
    if hasattr(light_data, "contact_shadow_distance"):
        light_data.contact_shadow_distance = max(0.002, min(base_distance * 1.05, tuned_distance))
    if hasattr(light_data, "contact_shadow_bias"):
        light_data.contact_shadow_bias = max(0.0005, min(0.003, 0.0022 / strength_scale))
    if hasattr(light_data, "contact_shadow_thickness"):
        light_data.contact_shadow_thickness = max(0.0008, min(0.018, 0.006 * strength_scale))


def configure_existing_scene_lights(contact_shadow_strength: float = 1.0):
    for obj in bpy.data.objects:
        if obj.type == "LIGHT" and not obj.hide_get():
            _configure_contact_shadow(obj, contact_shadow_strength)


def scale_existing_lights(factor: float):
    for light in bpy.data.lights:
        light.energy *= factor


def scale_world_background(scene, factor: float):
    if scene.world and scene.world.use_nodes:
        for node in scene.world.node_tree.nodes:
            if node.type == "BACKGROUND":
                node.inputs[1].default_value *= factor


def fit_object_height(root, objects, target_range):
    bbox = percentile_bbox(objects)
    if bbox is None:
        return 0.0, 1.0
    current = float(bbox["size"][2])
    if current <= 1e-6:
        return 0.0, 1.0
    lo, hi = float(target_range[0]), float(target_range[1])
    if lo <= current <= hi:
        return current, 1.0
    desired = (lo + hi) / 2.0
    factor = desired / current
    root.scale = tuple(v * factor for v in root.scale)
    bpy.context.view_layer.update()
    return float(percentile_bbox(objects)["size"][2]) if percentile_bbox(objects) else 0.0, factor


# ══════════════════════════════════════════════════════════════════════
# Render settings with depth + normal passes
# ══════════════════════════════════════════════════════════════════════

def setup_render_with_passes(resolution: int = 1024, engine: str = "CYCLES",
                               cycles_samples: int = 8, filmic_gamma: float = 0.54):
    scene = bpy.context.scene
    eevee_name = "BLENDER_EEVEE_NEXT" if int(bpy.app.version[0]) >= 4 else "BLENDER_EEVEE"
    scene.render.engine = eevee_name if engine == "EEVEE" else "CYCLES"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    if "CYCLES" in scene.render.engine:
        scene.cycles.samples = cycles_samples
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_denoising = False
        scene.cycles.device = "GPU"
        try:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            prefs.compute_device_type = "CUDA"
            prefs.get_devices()
            for d in prefs.devices:
                d.use = (d.type == "CUDA")
        except Exception:
            pass

    try:
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "None"
    except TypeError:
        scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = filmic_gamma

    # ── Enable depth + normal passes ──
    vl = bpy.context.view_layer
    vl.use_pass_z = True
    vl.use_pass_normal = True


def save_depth_pass(output_path: str):
    """Extract and save the depth pass as a 16-bit PNG."""
    render_result = bpy.context.scene.node_tree.nodes.get("Render Layers")
    if render_result is None:
        scene = bpy.context.scene
        scene.use_nodes = True
        nodes = scene.node_tree.nodes
        nodes.clear()
        render_result = nodes.new("ShaderNodeRenderLayer")

    scene = bpy.context.scene
    scene.use_nodes = True
    nodes = scene.node_tree.nodes

    # Get the rendered image
    depth_pixels = bpy.data.images["Viewer Node"].pixels if "Viewer Node" in bpy.data.images else None

    # Alternative approach: get render result directly
    import tempfile
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "16"
    tmp_exr = os.path.join(tempfile.gettempdir(), "_tmp_depth.exr")
    scene.render.filepath = tmp_exr

    # Use compositor to output depth
    nodes.clear()
    rl = nodes.new("ShaderNodeRenderLayer")
    output = nodes.new("ShaderNodeOutputFile")
    output.format.file_format = "OPEN_EXR"
    output.base_path = os.path.dirname(output_path)
    output.file_slots.clear()
    output.file_slots.new("depth")
    output.file_slots[0].path = os.path.splitext(os.path.basename(output_path))[0] + "_"
    nodes.link(rl.outputs["Depth"], output.inputs[0])

    # For PNG output, we need a normalize node
    nodes.clear()
    rl = nodes.new("ShaderNodeRenderLayer")
    normalize = nodes.new("ShaderNodeNormalize")
    output_node = nodes.new("ShaderNodeOutputFile")
    output_node.format.file_format = "PNG"
    output_node.format.color_mode = "BW"
    output_node.format.color_depth = "16"
    output_node.base_path = os.path.dirname(output_path)
    output_node.file_slots.clear()
    slot = output_node.file_slots.new(os.path.splitext(os.path.basename(output_path))[0])
    nodes.link(rl.outputs["Depth"], normalize.inputs["Value"])
    nodes.link(normalize.outputs["Value"], output_node.inputs[0])


def save_passes_direct(depth_path: str, normal_path: str):
    """Save depth and normal passes by compositing to PNG files.

    Depth: normalized Z depth (16-bit grayscale PNG).
    Normal: world-space normals remapped from [-1,1] to [0,1] (RGB PNG).

    Compositor nodes write files with a frame-number suffix (e.g. _tmp0001.png).
    We rename them to the final desired paths after rendering.
    """
    scene = bpy.context.scene
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    nodes.clear()

    rl = nodes.new("CompositorNodeRLayers")

    out_dir = os.path.dirname(depth_path)
    depth_stem = os.path.splitext(os.path.basename(depth_path))[0]
    normal_stem = os.path.splitext(os.path.basename(normal_path))[0]
    depth_tmp_slot = "_tmp_d_" + depth_stem
    normal_tmp_slot = "_tmp_n_" + normal_stem

    # ── Depth: normalize → 16-bit grayscale PNG ──
    normalize_z = nodes.new("CompositorNodeNormalize")
    depth_out = nodes.new("CompositorNodeOutputFile")
    depth_out.format.file_format = "PNG"
    depth_out.format.color_mode = "BW"
    depth_out.format.color_depth = "16"
    depth_out.base_path = out_dir + "/"
    depth_out.file_slots.clear()
    depth_out.file_slots.new(depth_tmp_slot)
    links.new(rl.outputs["Depth"], normalize_z.inputs["Value"])
    links.new(normalize_z.outputs["Value"], depth_out.inputs[0])

    # ── Normal: remap [-1,1] → [0,1] via (N+1)*0.5 ──
    # MixRGB ADD:  result = Color1 + Color2  (with Fac=1.0)
    add_rgb = nodes.new("CompositorNodeMixRGB")
    add_rgb.blend_type = "ADD"
    add_rgb.inputs[0].default_value = 1.0  # Fac
    add_rgb.inputs[2].default_value = (1.0, 1.0, 1.0, 1.0)  # Color2 = offset

    # MixRGB MULTIPLY:  result = Color1 * Color2  (with Fac=1.0)
    mul_rgb = nodes.new("CompositorNodeMixRGB")
    mul_rgb.blend_type = "MULTIPLY"
    mul_rgb.inputs[0].default_value = 1.0  # Fac
    mul_rgb.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)  # Color2 = scale

    normal_out = nodes.new("CompositorNodeOutputFile")
    normal_out.format.file_format = "PNG"
    normal_out.format.color_mode = "RGB"
    normal_out.format.color_depth = "16"
    normal_out.base_path = out_dir + "/"
    normal_out.file_slots.clear()
    normal_out.file_slots.new(normal_tmp_slot)

    links.new(rl.outputs["Normal"], add_rgb.inputs[1])     # Normal → ADD.Color1
    links.new(add_rgb.outputs["Image"], mul_rgb.inputs[1])  # ADD out → MUL.Color1
    links.new(mul_rgb.outputs["Image"], normal_out.inputs[0])

    # ── Render (compositor saves pass files automatically) ──
    bpy.ops.render.render(write_still=False)

    # ── Rename compositor output files to final names ──
    depth_tmp_file = os.path.join(out_dir, depth_tmp_slot + "0001.png")
    normal_tmp_file = os.path.join(out_dir, normal_tmp_slot + "0001.png")

    if os.path.exists(depth_tmp_file):
        os.rename(depth_tmp_file, depth_path)
    else:
        print(f"  WARNING: depth temp file not found: {depth_tmp_file}")

    if os.path.exists(normal_tmp_file):
        os.rename(normal_tmp_file, normal_path)
    else:
        print(f"  WARNING: normal temp file not found: {normal_tmp_file}")


# ══════════════════════════════════════════════════════════════════════
# Main: render one object across 8 yaws with depth + normal
# ══════════════════════════════════════════════════════════════════════

def render_object_depth_normal(obj_id: str, runtime_root: str, output_root: str):
    """Render depth + normal for all 8 yaw views of one object."""

    # Read base metadata from yaw000
    rotation8_dir = os.path.join(runtime_root, "rotation8_consistent", "objects", obj_id)
    yaw000_meta_path = os.path.join(rotation8_dir, "yaw000_render_metadata.json")
    meta = load_json(yaw000_meta_path)
    if meta is None:
        print(f"[ERROR] No metadata found for {obj_id}")
        return False

    blend_path = meta["blend_path"]
    asset_path = meta["source_asset"]
    resolution = meta.get("resolution", 1024)
    engine = meta.get("engine", "CYCLES")
    control_state = meta["control_state"]

    if not os.path.exists(blend_path):
        print(f"[ERROR] Blend file not found: {blend_path}")
        return False

    # ── Open blend & setup ──
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    scene = bpy.context.scene

    # Read scene template for render settings
    template_path = os.path.join(runtime_root, "rotation8_consistent", "scene_template_fixed_camera.json")
    template = load_json(template_path, default={}) or {}

    setup_render_with_passes(
        resolution=resolution,
        engine=engine,
        cycles_samples=8,
        filmic_gamma=float(template.get("filmic_gamma", 0.54)),
    )

    # ── Scale existing lights & world ──
    scale_existing_lights(float(template.get("scale_existing_lights", 1.5)))
    scale_world_background(scene, float(template.get("scale_world_background", 1.2)))

    # ── Import asset ──
    imported = import_glb(asset_path)
    mesh_objects = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objects:
        print(f"[ERROR] No mesh objects imported from {asset_path}")
        return False
    root = normalize_object_v2(imported)

    # ── Sanitize materials ──
    sanitize_materials(mesh_objects, roughness_add=float(meta.get("material_adaptation", {}).get("effective_roughness_add", 0.0)),
                       specular_add=float(meta.get("material_adaptation", {}).get("effective_specular_add", 0.0)))

    control_material = control_state.get("material", {})
    force_ref = any([
        abs(float(control_material.get("saturation_scale", 1.0)) - 1.0) >= 0.15,
        abs(float(control_material.get("value_scale", 1.0)) - 1.0) >= 0.2,
        abs(float(control_material.get("hue_offset", 0.0))) >= 0.02,
        abs(float(control_material.get("roughness_add", 0.0))) >= 0.12,
        abs(float(control_material.get("specular_add", 0.0))) >= 0.08,
    ])
    ensure_reference_material(
        mesh_objects,
        reference_rgb=control_material.get("reference_rgb"),
        roughness_add=float(control_material.get("roughness_add", 0.0)),
        specular_add=float(control_material.get("specular_add", 0.0)),
        force_reference_material=force_ref,
    )
    adjust_material_hsv(
        mesh_objects,
        saturation_scale=float(control_material.get("saturation_scale", 1.0)),
        value_scale=float(control_material.get("value_scale", 1.0)),
        hue_offset=float(control_material.get("hue_offset", 0.0)),
    )

    # ── Ground detection ──
    support_name = template.get("support_object_name", "BézierCurve")
    ground_obj, _ = find_ground_object(support_name)
    if ground_obj:
        fix_ground_orientation(ground_obj)
    else:
        print(f"[ERROR] No ground object found (support_name={support_name})")
        return False

    # Scale to target height
    fit_object_height(root, mesh_objects, template.get("target_bbox_height_range", [0.5, 2.0]))

    # Get ground bounds for XY placement
    ground_bbox = [ground_obj.matrix_world @ Vector(c) for c in ground_obj.bound_box]
    ground_cx = sum(v.x for v in ground_bbox) / 8.0
    ground_cy = sum(v.y for v in ground_bbox) / 8.0
    ground_z_rough = min(v.z for v in ground_bbox)

    # ── Camera ──
    cam = resolve_scene_camera(scene)
    anchor_point = camera_screen_ground_anchor(scene, cam, ground_z_rough,
                                                u=float(template.get("camera_anchor_u", 0.5)),
                                                v=float(template.get("camera_anchor_v", 0.22)))

    # ── Configure lights ──
    control_scene = control_state.get("scene", {})
    configure_existing_scene_lights(float(control_scene.get("contact_shadow_strength", 1.0)))

    # Add key light if template says so
    if template.get("add_key_light", False):
        control_lighting = control_state.get("lighting", {})
        key_yaw = float(template.get("key_light_yaw_deg", 0.0)) + float(control_lighting.get("key_yaw_deg", 0.0))
        base_loc = Vector(template.get("key_light_xyz", [3.5, -3.5, 5.0]))
        rot = Matrix.Rotation(math.radians(key_yaw), 3, "Z")
        loc = rot @ base_loc
        bpy.ops.object.light_add(type="SUN", location=loc)
        key_light = bpy.context.active_object
        key_light.name = "V7KeyLight"
        key_light.data.energy = float(template.get("key_light_energy", 30.0)) * float(control_lighting.get("key_scale", 1.0))
        _configure_contact_shadow(key_light, float(control_scene.get("contact_shadow_strength", 1.0)))

    # ── Output dir ──
    views_out = os.path.join(output_root, "views", obj_id)
    os.makedirs(views_out, exist_ok=True)

    # ── Per-yaw loop ──
    control_object = control_state.get("object", {})
    base_scale = float(control_object.get("scale", 1.0))
    base_offset_z = float(control_object.get("offset_z", 0.0))

    for yaw in YAW_ANGLES:
        yaw_str = f"yaw{yaw:03d}"
        print(f"  [{obj_id}] Rendering {yaw_str} depth+normal...")

        # Reset root position to ground center for each yaw
        if anchor_point is not None:
            root.location.x = anchor_point.x
            root.location.y = anchor_point.y
        else:
            root.location.x = ground_cx
            root.location.y = ground_cy

        # Apply (or reset) yaw rotation
        root.rotation_euler.z = math.radians(float(yaw))
        bpy.context.view_layer.update()

        # Reset Z before re-grounding (avoid accumulation across yaws)
        root.location.z = 0.0

        # Z alignment
        bbox = percentile_bbox(mesh_objects)
        if bbox:
            bottom_z = float(bbox["min"][2])
            root.location.z += ground_z_rough - bottom_z
            bpy.context.view_layer.update()

        # Precision landing
        adjust_object_to_ground_with_ray(root, mesh_objects, ground_obj)

        # Z offset + epsilon
        root.location.z += base_offset_z
        ground_contact_epsilon = compute_ground_contact_epsilon(mesh_objects)
        root.location.z += ground_contact_epsilon
        bpy.context.view_layer.update()

        # ── Render depth + normal ──
        depth_path = os.path.join(views_out, f"{yaw_str}_depth.png")
        normal_path = os.path.join(views_out, f"{yaw_str}_normal.png")

        save_passes_direct(depth_path, normal_path)

        print(f"    -> {depth_path}")
        print(f"    -> {normal_path}")

    return True


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(description="Render depth + normal maps for one object")
    p.add_argument("--obj-id", required=True)
    p.add_argument("--runtime-root", required=True)
    p.add_argument("--output-root", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(f"[depth-normal] Processing {args.obj_id}")
    print(f"  runtime_root: {args.runtime_root}")
    print(f"  output_root: {args.output_root}")
    success = render_object_depth_normal(args.obj_id, args.runtime_root, args.output_root)
    if success:
        print(f"[depth-normal] {args.obj_id} DONE")
    else:
        print(f"[depth-normal] {args.obj_id} FAILED")
        sys.exit(1)
