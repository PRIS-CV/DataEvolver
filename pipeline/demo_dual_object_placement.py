"""
Demo: place two objects into an arbitrary Blender scene with proper camera framing.

Usage (run inside Blender):
    blender scene.blend --python pipeline/demo_dual_object_placement.py -- \
        --obj1 /path/to/asset1.glb \
        --obj2 /path/to/asset2.glb \
        --output /path/to/render.png

Logic:
    1. Build unified BVH from all scene geometry
    2. Detect support plane (ground)
    3. Import & scale both objects to fit scene proportions
    4. Place object A at a valid sampled position (raycast + collision)
    5. Place object B relative to A using a composition template
    6. Position camera to frame both objects with background visible
    7. Render
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COMPOSITION_TEMPLATES = {
    "side_by_side": {
        "angle_range": (-30, 30),
        "distance_factor": (1.2, 1.8),
    },
    "diagonal": {
        "angle_range": (30, 60),
        "distance_factor": (1.4, 2.2),
    },
    "foreground_background": {
        "angle_range": (-15, 15),
        "distance_factor": (2.0, 3.0),
    },
}

# B object floats slightly above A
B_HEIGHT_OFFSET_FACTOR = (0.225, 0.525)  # fraction of B's height to lift (1.5x original)
# Minimum gap between A and B (fraction of smaller object's size)
MIN_GAP_FACTOR = 0.2

MAX_OBJECT_RATIO = 0.08
MIN_OBJECT_SIZE = 0.2
MAX_PLACEMENT_ATTEMPTS = 50
CAMERA_ELEVATION_DEG = 20.0
CAMERA_MARGIN_FACTOR = 4.0
RENDER_RESOLUTION = 512


# ---------------------------------------------------------------------------
# Scene BVH
# ---------------------------------------------------------------------------

def is_scene_relevant_object(obj):
    """Filter out placeholder/scatter objects and tiny background planes."""
    if obj.type != "MESH" or obj.hide_render:
        return False
    name = obj.name.lower()
    if "fog" in name or "sky" in name:
        return False
    # Skip scatter system placeholders
    if "scatter5_placeholder" in name or "scatter_obj" in name:
        return False
    # Skip objects with 0 vertices (empty scatter containers)
    if not obj.data.vertices:
        return False
    # Skip tiny 4-vert planes that are far from origin (background cards)
    if len(obj.data.vertices) <= 4:
        loc = obj.matrix_world.translation
        if abs(loc.z) > 20 or loc.length > 50:
            return False
    return True


def build_scene_bvh(exclude_names=None):
    """Merge all relevant scene mesh objects into a single BVH tree."""
    exclude = set(exclude_names or [])
    bm = bmesh.new()
    for obj in bpy.data.objects:
        if not is_scene_relevant_object(obj):
            continue
        if obj.name in exclude:
            continue
        try:
            me = obj.to_mesh()
        except RuntimeError:
            continue
        me.transform(obj.matrix_world)
        bm.from_mesh(me)
        obj.to_mesh_clear()
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    return bvh


def detect_ground_plane():
    """Find the largest roughly-horizontal surface as the ground/support plane.

    Returns dict with keys: name, z, bounds [x_min, x_max, y_min, y_max], area, center_xy
    or None if no suitable ground found.
    """
    candidates = []
    for obj in bpy.data.objects:
        if not is_scene_relevant_object(obj):
            continue
        mesh = obj.data
        if not getattr(mesh, "polygons", None):
            continue

        mat = obj.matrix_world
        rot = mat.to_3x3().normalized()
        z_values = []
        xs, ys = [], []
        for poly in mesh.polygons:
            normal_w = (rot @ poly.normal).normalized()
            if normal_w.z < 0.85:
                continue
            center = mat @ poly.center
            z_values.append(center.z)
            for vid in poly.vertices:
                wv = mat @ mesh.vertices[vid].co
                xs.append(wv.x)
                ys.append(wv.y)
        if not z_values or len(xs) < 3:
            continue
        from statistics import median
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        area = max(0.0, (x_max - x_min) * (y_max - y_min))
        if area < 0.5:
            continue
        candidates.append({
            "name": obj.name,
            "z": float(median(z_values)),
            "bounds": [float(x_min), float(x_max), float(y_min), float(y_max)],
            "area": float(area),
        })

    if not candidates:
        return None
    # Prefer lowest large horizontal surface as ground
    # Sort by: area (descending) but penalize high-z surfaces
    # Strategy: among candidates with area > 50% of max area, pick the lowest z
    max_area = max(c["area"] for c in candidates)
    large_enough = [c for c in candidates if c["area"] > max_area * 0.3]
    if large_enough:
        large_enough.sort(key=lambda c: c["z"])
        chosen = large_enough[0]
    else:
        candidates.sort(key=lambda c: -c["area"])
        chosen = candidates[0]
    chosen["center_xy"] = [
        (chosen["bounds"][0] + chosen["bounds"][1]) / 2.0,
        (chosen["bounds"][2] + chosen["bounds"][3]) / 2.0,
    ]
    print(f"[dual] Detected ground: '{chosen['name']}', z={chosen['z']:.3f}, "
          f"area={chosen['area']:.1f}, bounds={chosen['bounds']}")
    return chosen


def get_scene_bounds(exclude_names=None):
    """World-space AABB of relevant scene meshes (filters out background/placeholders)."""
    exclude = set(exclude_names or [])
    min_v = np.array([np.inf, np.inf, np.inf])
    max_v = np.array([-np.inf, -np.inf, -np.inf])
    for obj in bpy.data.objects:
        if not is_scene_relevant_object(obj):
            continue
        if obj.name in exclude:
            continue
        for corner in obj.bound_box:
            w = obj.matrix_world @ Vector(corner)
            for i in range(3):
                min_v[i] = min(min_v[i], w[i])
                max_v[i] = max(max_v[i], w[i])
    if min_v[0] == np.inf:
        return np.zeros(3), np.array([10.0, 10.0, 5.0])
    return min_v, max_v


# ---------------------------------------------------------------------------
# Object import & scaling
# ---------------------------------------------------------------------------

def import_asset(path: str):
    """Import GLB/OBJ/FBX and return root + mesh objects."""
    before = set(bpy.data.objects)
    ext = Path(path).suffix.lower()
    if ext == ".glb" or ext == ".gltf":
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".blend":
        with bpy.data.libraries.load(path) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:
            if obj:
                bpy.context.collection.objects.link(obj)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    imported = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in imported if o.type == "MESH"]

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
    root = bpy.context.active_object
    root.name = f"DualPlacement_{Path(path).stem}"
    for obj in imported:
        if obj.parent is None:
            obj.parent = root
    bpy.context.view_layer.update()
    return root, meshes


def get_object_bbox(meshes):
    """Compute world-space AABB of mesh list."""
    pts = []
    for obj in meshes:
        mat = obj.matrix_world
        for v in obj.data.vertices:
            w = mat @ v.co
            pts.append([w.x, w.y, w.z])
    if not pts:
        return None
    pts = np.array(pts)
    return {
        "min": pts.min(axis=0),
        "max": pts.max(axis=0),
        "center": (pts.min(axis=0) + pts.max(axis=0)) / 2,
        "size": pts.max(axis=0) - pts.min(axis=0),
    }


def scale_object_to_scene(root, meshes, scene_min, scene_max):
    """Scale object so its max dimension is at most MAX_OBJECT_RATIO of scene avg span.

    Also enforces a minimum size so tiny objects aren't invisible.
    """
    scene_size = scene_max - scene_min
    avg_scene_dim = float(np.mean(scene_size[:2]))  # XY average
    target_max_size = avg_scene_dim * MAX_OBJECT_RATIO

    bbox = get_object_bbox(meshes)
    if bbox is None:
        return 1.0
    current_max = float(max(bbox["size"]))
    if current_max < 1e-6:
        return 1.0

    if current_max > target_max_size:
        factor = target_max_size / current_max
    elif current_max < MIN_OBJECT_SIZE:
        factor = MIN_OBJECT_SIZE / current_max
    else:
        factor = 1.0

    # Halve all objects after scene-based scaling
    factor *= 0.5

    root.scale = tuple(v * factor for v in root.scale)
    bpy.context.view_layer.update()
    print(f"[dual] Scaled {root.name}: {current_max:.3f} -> {current_max*factor:.3f} "
          f"(scene avg XY={avg_scene_dim:.2f}, max allowed={target_max_size:.2f})")
    return factor


# ---------------------------------------------------------------------------
# Placement: find valid ground position
# ---------------------------------------------------------------------------

def find_ground_position(scene_bvh, scene_min, scene_max, ground_info=None,
                         exclude_bvh=None, center_bias=None, radius=None,
                         max_attempts=MAX_PLACEMENT_ATTEMPTS):
    """Sample a valid ground position via downward raycasting."""
    if ground_info is not None:
        x_min, x_max, y_min, y_max = ground_info["bounds"]
        ground_z = ground_info["z"]
    else:
        x_min, x_max = float(scene_min[0]), float(scene_max[0])
        y_min, y_max = float(scene_min[1]), float(scene_max[1])
        ground_z = float(scene_min[2])

    z_top = float(scene_max[2]) + 2.0

    for _ in range(max_attempts):
        if center_bias is not None and radius is not None:
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(radius * 0.4, radius)
            x = center_bias[0] + r * math.cos(angle)
            y = center_bias[1] + r * math.sin(angle)
            # Clamp to ground bounds
            x = max(x_min, min(x_max, x))
            y = max(y_min, min(y_max, y))
        else:
            # Sample within inner 80% of ground bounds
            margin_x = (x_max - x_min) * 0.1
            margin_y = (y_max - y_min) * 0.1
            x = random.uniform(x_min + margin_x, x_max - margin_x)
            y = random.uniform(y_min + margin_y, y_max - margin_y)

        origin = Vector((x, y, z_top))
        direction = Vector((0, 0, -1))

        # Cast multiple rays - we want to find the ground, not rooftops
        # Strategy: cast ray, if hit is too high above ground, cast again from below that hit
        hit_loc, hit_normal, _, _ = scene_bvh.ray_cast(origin, direction)

        if hit_loc is None:
            continue

        # If hit is far above ground level, try to find ground below it
        if ground_info and hit_loc.z > ground_info["z"] + 1.0:
            # Cast again from just below the hit
            retry_origin = Vector((x, y, hit_loc.z - 0.01))
            hit_loc2, hit_normal2, _, _ = scene_bvh.ray_cast(retry_origin, direction)
            if hit_loc2 is not None and hit_normal2.z > 0.6:
                hit_loc = hit_loc2
                hit_normal = hit_normal2
            else:
                continue

        # Must be roughly horizontal (ground)
        if hit_normal.z < 0.6:
            continue
        # Should be near the detected ground level for flat ground, while still
        # allowing terrain scenes to use the actual raycast height.
        if ground_info and abs(hit_loc.z - ground_z) > 8.0:
            continue

        return (float(hit_loc.x), float(hit_loc.y), float(hit_loc.z), hit_normal)

    return None


def check_placement_collision(obj_root, obj_meshes, scene_bvh):
    """Check if placed object overlaps with scene geometry using BVH overlap."""
    bm = bmesh.new()
    for obj in obj_meshes:
        try:
            me = obj.to_mesh()
        except RuntimeError:
            continue
        me.transform(obj.matrix_world)
        bm.from_mesh(me)
        obj.to_mesh_clear()
    if len(bm.verts) == 0:
        bm.free()
        return False
    obj_bvh = BVHTree.FromBMesh(bm)
    bm.free()
    overlaps = obj_bvh.overlap(scene_bvh)
    return len(overlaps) > 0


def place_object_at(root, meshes, x, y, z):
    """Move object root so its bottom center sits at (x, y, z + small offset)."""
    bbox = get_object_bbox(meshes)
    if bbox is None:
        root.location = Vector((x, y, z))
        return
    bottom_center = np.array([(bbox["min"][0] + bbox["max"][0]) / 2,
                              (bbox["min"][1] + bbox["max"][1]) / 2,
                              bbox["min"][2]])
    current_root = np.array(root.location[:])
    # Lift slightly above ground to avoid BVH overlap with ground surface
    offset = np.array([x, y, z + 0.005]) - bottom_center
    root.location = Vector(current_root + offset)
    bpy.context.view_layer.update()


def place_object_bottom_at(root, meshes, position):
    """Move object root so its current bottom center sits at a saved position."""
    place_object_at(root, meshes, float(position[0]), float(position[1]), float(position[2]))


def restore_two_object_placement(root1, meshes1, root2, meshes2, placement_state):
    """Restore a frozen dual placement generated by an initial search pass."""
    obj1_position = placement_state.get("obj1_position")
    obj2_position = placement_state.get("obj2_position")
    if not obj1_position or not obj2_position:
        return None
    place_object_bottom_at(root1, meshes1, obj1_position)
    place_object_bottom_at(root2, meshes2, obj2_position)
    bbox1 = get_object_bbox(meshes1)
    bbox2 = get_object_bbox(meshes2)
    return {
        "template": placement_state.get("template", "fixed"),
        "obj1_position": [float(v) for v in obj1_position],
        "obj2_position": [float(v) for v in obj2_position],
        "separation": float(placement_state.get("separation", 0.0)),
        "height_offset": float(placement_state.get("height_offset", float(obj2_position[2]) - float(obj1_position[2]))),
        "side": int(placement_state.get("side", 0) or 0),
        "camera_azimuth": float(placement_state.get("camera_azimuth", 0.0)),
        "bbox1": bbox1,
        "bbox2": bbox2,
        "placement_source": "frozen",
    }


# ---------------------------------------------------------------------------
# Dual placement: camera-first approach
# ---------------------------------------------------------------------------

def find_hdri_best_azimuth():
    """Find world-space azimuth facing the most interesting part of the HDRI.

    Samples the environment texture at the horizon band, scores each azimuth
    sector by brightness + variance, returns the world azimuth of the best one.
    Returns None if no HDRI environment is found.
    """
    world = bpy.context.scene.world
    if not world or not world.node_tree:
        return None

    env_node = None
    mapping_z_rot = 0.0
    for node in world.node_tree.nodes:
        if node.type == 'TEX_ENVIRONMENT':
            env_node = node
        elif node.type == 'MAPPING':
            try:
                mapping_z_rot = node.inputs['Rotation'].default_value[2]
            except Exception:
                pass

    if not env_node or not env_node.image:
        return None

    img = env_node.image
    w, h = img.size[0], img.size[1]
    ch = img.channels
    pixels = np.array(img.pixels[:]).reshape(h, w, ch)
    # Blender stores pixels bottom-to-top, flip to top-to-bottom
    pixels = pixels[::-1]

    # Horizon band: 30%-70% of height (includes slightly above and below horizon)
    y0 = int(h * 0.3)
    y1 = int(h * 0.7)
    band = pixels[y0:y1, :, :3]

    # Luminance
    lum = 0.299 * band[:, :, 0] + 0.587 * band[:, :, 1] + 0.114 * band[:, :, 2]

    # Score each azimuth sector
    n_sectors = 36
    best_score = -1.0
    best_u = 0.5

    for i in range(n_sectors):
        x0 = int(i * w / n_sectors)
        x1 = int((i + 1) * w / n_sectors)
        sector_lum = lum[:, x0:x1]
        score = float(np.mean(sector_lum) + 0.5 * np.std(sector_lum))
        if score > best_score:
            best_score = score
            best_u = (i + 0.5) / n_sectors

    # Convert HDRI u-coordinate to world-space azimuth.
    # Blender equirectangular: u = atan2(dx, -dy)/(2π) + 0.5
    # So for a given u:  α = (u - 0.5)*2π,  direction = (sin(α), -cos(α), 0)
    # World azimuth θ = atan2(-cos(α), sin(α))
    alpha = (best_u - 0.5) * 2 * math.pi
    hdri_azimuth = math.atan2(-math.cos(alpha), math.sin(alpha))

    # Account for Mapping node Z rotation
    world_azimuth = (hdri_azimuth + mapping_z_rot) % (2 * math.pi)

    print(f"[dual] HDRI best azimuth: {math.degrees(world_azimuth):.1f}° "
          f"(u={best_u:.3f}, score={best_score:.3f}, mapping_rot={math.degrees(mapping_z_rot):.1f}°)")
    return world_azimuth

def place_two_objects(root1, meshes1, root2, meshes2, scene_bvh, scene_min, scene_max,
                      ground_info=None, template_name=None, camera_azimuth=None):
    """Place two objects with AB line perpendicular to camera view.

    Strategy:
    1. Pick a random camera azimuth direction
    2. Place object A on the ground
    3. Place object B to the side of A (perpendicular to camera direction)
       at a slightly higher elevation (floating effect)
    4. Ensure no contact between A and B

    Returns:
        dict with placement info or None on failure
    """
    if template_name is None:
        template_name = random.choice(list(COMPOSITION_TEMPLATES.keys()))
    print(f"[dual] Using composition template: {template_name}")

    # 1. Choose camera azimuth (use provided or random)
    if camera_azimuth is None:
        camera_azimuth = random.uniform(0, 2 * math.pi)
    # Camera looks in -azimuth direction (toward objects)
    cam_forward = np.array([math.cos(camera_azimuth), math.sin(camera_azimuth)])
    # Perpendicular to camera view (the direction AB should align with)
    cam_right = np.array([-cam_forward[1], cam_forward[0]])

    # 2. Place object A on ground (with slight float)
    pos1 = find_ground_position(scene_bvh, scene_min, scene_max, ground_info=ground_info)
    if pos1 is None:
        print("[dual] ERROR: could not find valid position for object 1")
        return None
    x1, y1, z1, _ = pos1
    # Slight float for object A
    bbox1_pre = get_object_bbox(meshes1)
    a_float = float(bbox1_pre["size"][2]) * 0.05 if bbox1_pre is not None else 0.05
    z1 += a_float
    place_object_at(root1, meshes1, x1, y1, z1)

    # 3. Compute separation: A and B must not touch
    bbox1 = get_object_bbox(meshes1)
    bbox2 = get_object_bbox(meshes2)
    size1_xy = float(max(bbox1["size"][:2])) if bbox1 is not None else 1.0
    size2_xy = float(max(bbox2["size"][:2])) if bbox2 is not None else 1.0
    size2_z = float(bbox2["size"][2]) if bbox2 is not None else 1.0
    min_size = min(size1_xy, size2_xy)

    # Separation = half-widths + gap
    half_extent_1 = size1_xy / 2.0
    half_extent_2 = size2_xy / 2.0
    gap = min_size * MIN_GAP_FACTOR
    separation = half_extent_1 + half_extent_2 + gap
    separation *= 2.0

    # 4. B position: along cam_right from A (perpendicular to camera view)
    # Add small random sign to place B left or right
    side = random.choice([-1, 1])
    bx = x1 + side * separation * cam_right[0]
    by = y1 + side * separation * cam_right[1]

    # B floats above A's ground level
    height_offset = size2_z * random.uniform(*B_HEIGHT_OFFSET_FACTOR)
    bz = z1 + height_offset

    # 5. Place B (floating, no ground raycast needed since it's airborne)
    place_object_at(root2, meshes2, bx, by, bz)

    # 6. Verify no collision between B and scene geometry
    if check_placement_collision(root2, meshes2, scene_bvh):
        # Try the other side
        side = -side
        bx = x1 + side * separation * cam_right[0]
        by = y1 + side * separation * cam_right[1]
        place_object_at(root2, meshes2, bx, by, bz)
        if check_placement_collision(root2, meshes2, scene_bvh):
            # Increase separation
            separation *= 1.5
            bx = x1 + side * separation * cam_right[0]
            by = y1 + side * separation * cam_right[1]
            place_object_at(root2, meshes2, bx, by, bz)
            if check_placement_collision(root2, meshes2, scene_bvh):
                print("[dual] ERROR: could not find valid position for object 2")
                return None

    # 7. Verify A doesn't collide with scene
    if check_placement_collision(root1, meshes1, scene_bvh):
        root1.scale = tuple(v * 0.8 for v in root1.scale)
        bpy.context.view_layer.update()
        place_object_at(root1, meshes1, x1, y1, z1)

    # 8. Verify A and B don't touch each other
    final_bbox1 = get_object_bbox(meshes1)
    final_bbox2 = get_object_bbox(meshes2)

    return {
        "template": template_name,
        "obj1_position": [x1, y1, z1],
        "obj2_position": [bx, by, bz],
        "separation": separation,
        "height_offset": height_offset,
        "side": side,
        "camera_azimuth": camera_azimuth,
        "bbox1": final_bbox1,
        "bbox2": final_bbox2,
    }


# ---------------------------------------------------------------------------
# Camera framing
# ---------------------------------------------------------------------------

def compute_combined_bbox(bbox1, bbox2):
    """Compute AABB encompassing both objects."""
    combined_min = np.minimum(bbox1["min"], bbox2["min"])
    combined_max = np.maximum(bbox1["max"], bbox2["max"])
    return {
        "min": combined_min,
        "max": combined_max,
        "center": (combined_min + combined_max) / 2,
        "size": combined_max - combined_min,
    }


def setup_camera_for_dual(placement_info, scene_min, scene_max, elevation_deg=CAMERA_ELEVATION_DEG):
    """Position camera using the pre-determined azimuth from placement.

    The camera looks from the azimuth direction toward the objects.
    AB line is perpendicular to camera view by construction.
    """
    bbox1 = placement_info["bbox1"]
    bbox2 = placement_info["bbox2"]
    combined = compute_combined_bbox(bbox1, bbox2)

    # Look-at target: combined center, at ~40% height
    target = Vector(combined["center"].tolist())
    target.z = float(combined["min"][2]) + float(combined["size"][2]) * 0.4

    # Camera direction from the pre-computed azimuth
    camera_azimuth = placement_info["camera_azimuth"]
    cam_dir_xy = np.array([math.cos(camera_azimuth), math.sin(camera_azimuth)])

    # Distance: fit combined diagonal in FOV with margin
    combined_diag = float(np.linalg.norm(combined["size"]))
    fov_rad = math.radians(50)  # ~50mm lens
    distance = (combined_diag * CAMERA_MARGIN_FACTOR) / (2 * math.tan(fov_rad / 2))
    distance *= 1.875
    distance *= 0.8  # pull camera closer
    distance = max(distance, 2.0)
    distance = min(distance, 100.0)

    # Camera position
    el_rad = math.radians(elevation_deg)
    cam_x = target.x + distance * math.cos(el_rad) * cam_dir_xy[0]
    cam_y = target.y + distance * math.cos(el_rad) * cam_dir_xy[1]
    cam_z = target.z + distance * math.sin(el_rad)

    # Create or reuse camera
    cam = None
    for obj in bpy.data.objects:
        if obj.type == "CAMERA" and obj.name == "DualPlacementCam":
            cam = obj
            break
    if cam is None:
        bpy.ops.object.camera_add(location=(cam_x, cam_y, cam_z))
        cam = bpy.context.active_object
        cam.name = "DualPlacementCam"
    else:
        cam.location = (cam_x, cam_y, cam_z)

    cam.data.lens = 50
    cam.data.clip_start = 0.1
    cam.data.clip_end = 200.0

    # Point camera at target
    direction = target - Vector((cam_x, cam_y, cam_z))
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    bpy.context.scene.camera = cam
    bpy.context.view_layer.update()

    print(f"[dual] Camera at ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}), "
          f"distance={distance:.2f}, looking at ({target.x:.2f}, {target.y:.2f}, {target.z:.2f})")
    return cam


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def setup_render(resolution=RENDER_RESOLUTION, engine="EEVEE"):
    scene = bpy.context.scene
    scene.render.resolution_x = int(resolution * 16 / 9)
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    if engine.upper() == "CYCLES":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 128
        scene.cycles.device = "GPU"
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT" if bpy.app.version >= (4, 0, 0) else "BLENDER_EEVEE"


def render_to_file(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"[dual] Rendered to {output_path}")


def ensure_minimum_lighting():
    """Add a fill sun light if scene ambient is too dark."""
    scene = bpy.context.scene
    world = scene.world
    needs_fill = False

    if world and world.use_nodes:
        for node in world.node_tree.nodes:
            if node.type == 'GROUP':
                for inp in node.inputs:
                    if 'strength' in inp.name.lower() or 'ev' in inp.name.lower():
                        if hasattr(inp, 'default_value') and inp.default_value < 1.0:
                            needs_fill = True
                            break
            if node.type == 'BACKGROUND':
                strength_input = node.inputs.get('Strength')
                if strength_input and strength_input.default_value < 0.5:
                    needs_fill = True
    elif not world:
        needs_fill = True

    if not needs_fill:
        total_energy = sum(obj.data.energy for obj in bpy.data.objects if obj.type == 'LIGHT')
        if total_energy < 5000:
            needs_fill = True

    if needs_fill:
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        sun = bpy.context.active_object
        sun.name = 'DualPipeline_FillLight'
        sun.data.energy = 3.0
        sun.data.angle = 0.5
        sun.rotation_euler = (0.7, 0.2, -0.5)
        print("[dual] Added fill sun light (scene was too dark)")


def add_object_spotlight(placement_info, intensity=50.0, spot_size_deg=45.0, height=5.0):
    """Add a spot light above the objects to brighten them without affecting the whole scene."""
    bbox1 = placement_info["bbox1"]
    bbox2 = placement_info["bbox2"]
    combined_min = np.minimum(bbox1["min"], bbox2["min"])
    combined_max = np.maximum(bbox1["max"], bbox2["max"])
    center = (combined_min + combined_max) / 2.0

    # Place light above objects, slightly offset toward camera
    light_pos = (float(center[0]), float(center[1]), float(combined_max[2]) + height)

    bpy.ops.object.light_add(type='SPOT', location=light_pos)
    spot = bpy.context.active_object
    spot.name = "DualPipeline_ObjectSpot"
    spot.data.energy = intensity
    spot.data.spot_size = math.radians(spot_size_deg)
    spot.data.spot_blend = 0.5  # soft edge
    spot.data.shadow_soft_size = 1.0

    # Point straight down at objects
    spot.rotation_euler = (0, 0, 0)  # default points -Z = downward
    bpy.context.view_layer.update()

    print(f"[dual] Added spot light above objects: energy={intensity}, "
          f"pos=({light_pos[0]:.1f}, {light_pos[1]:.1f}, {light_pos[2]:.1f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Dual object placement demo")
    parser.add_argument("--obj1", required=True, help="Path to first asset (GLB/OBJ/FBX)")
    parser.add_argument("--obj2", required=True, help="Path to second asset (GLB/OBJ/FBX)")
    parser.add_argument("--output", default="/tmp/dual_placement_render.png",
                        help="Output render path")
    parser.add_argument("--template", choices=list(COMPOSITION_TEMPLATES.keys()),
                        default=None, help="Composition template (random if not set)")
    parser.add_argument("--resolution", type=int, default=RENDER_RESOLUTION)
    parser.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES"])
    parser.add_argument("--max-object-ratio", type=float, default=MAX_OBJECT_RATIO,
                        help="Max object size as fraction of scene (default 0.08)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args(argv)


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    global MAX_OBJECT_RATIO
    MAX_OBJECT_RATIO = args.max_object_ratio

    print(f"[dual] === Dual Object Placement Demo ===")
    print(f"[dual] Scene: {bpy.data.filepath}")
    print(f"[dual] Obj1: {args.obj1}")
    print(f"[dual] Obj2: {args.obj2}")

    # 1. Analyze scene
    ground_info = detect_ground_plane()
    scene_min, scene_max = get_scene_bounds()
    scene_size = scene_max - scene_min
    print(f"[dual] Scene bounds: {scene_min} -> {scene_max}")
    print(f"[dual] Scene size: {scene_size}")

    if ground_info is None:
        print("[dual] WARNING: no ground plane detected, using Z=0 fallback")
        ground_info = {
            "name": "_fallback",
            "z": 0.0,
            "bounds": [float(scene_min[0]), float(scene_max[0]),
                       float(scene_min[1]), float(scene_max[1])],
            "area": float(scene_size[0] * scene_size[1]),
            "center_xy": [float((scene_min[0] + scene_max[0]) / 2),
                          float((scene_min[1] + scene_max[1]) / 2)],
        }

    # 2. Build scene BVH (before importing objects)
    scene_bvh = build_scene_bvh()

    # 3. Import objects
    root1, meshes1 = import_asset(args.obj1)
    root2, meshes2 = import_asset(args.obj2)

    # 4. Scale objects to fit scene
    obj_names = [root1.name, root2.name] + [m.name for m in meshes1 + meshes2]
    scale_object_to_scene(root1, meshes1, scene_min, scene_max)
    scale_object_to_scene(root2, meshes2, scene_min, scene_max)

    # 4.5. Find HDRI best direction for camera
    hdri_azimuth = find_hdri_best_azimuth()
    if hdri_azimuth is not None:
        # camera_azimuth = direction from target to camera
        # Camera looks at target, so background is at azimuth camera_azimuth - π
        # We want background = hdri_azimuth, so camera_azimuth = hdri_azimuth + π
        cam_az = (hdri_azimuth + math.pi) % (2 * math.pi)
        print(f"[dual] Camera azimuth set from HDRI: {math.degrees(cam_az):.1f}° "
              f"(facing HDRI at {math.degrees(hdri_azimuth):.1f}°)")
    else:
        cam_az = None
        print("[dual] No HDRI found, using random camera azimuth")

    # 5. Place both objects
    placement = place_two_objects(root1, meshes1, root2, meshes2,
                                  scene_bvh, scene_min, scene_max,
                                  ground_info=ground_info,
                                  template_name=args.template,
                                  camera_azimuth=cam_az)
    if placement is None:
        print("[dual] FAILED: could not place objects. Scene may be too constrained.")
        sys.exit(1)

    print(f"[dual] Placement successful: template={placement['template']}, "
          f"separation={placement['separation']:.2f}")

    # 6. Setup camera
    cam = setup_camera_for_dual(placement, scene_min, scene_max)

    # 7. Render
    setup_render(resolution=args.resolution, engine=args.engine)
    ensure_minimum_lighting()
    add_object_spotlight(placement)
    render_to_file(args.output)

    print(f"[dual] Done!")


if __name__ == "__main__":
    main()
