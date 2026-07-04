#!/usr/bin/env python3
"""Find HYWorld support anchors visible from the authored Blender camera.

Run inside Blender:
  blender -b scene.blend -P scripts/find_hyworld_visible_support_anchor.py -- \
    --scene-contract contract.json --output candidates.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-contract", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid-size", type=int, default=17)
    parser.add_argument("--target-height", type=float, default=0.3)
    parser.add_argument("--ground-ray-height", type=float, default=0.8)
    parser.add_argument("--ground-height-tolerance", type=float, default=0.35)
    parser.add_argument("--vertex-height-tolerance", type=float, default=0.18)
    parser.add_argument("--vertex-stride", type=int, default=2)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    contract_path = Path(args.scene_contract).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    support_id = contract["selected_support_surface_id"]
    support = next(item for item in contract["support_surfaces"] if item["id"] == support_id)
    bounds = [float(value) for value in support["bounds_xy"]]
    support_height = float(support["height"])
    support_obj = bpy.data.objects.get(support["object_name"])
    if support_obj is None:
        raise RuntimeError(f"Support object not found: {support['object_name']}")

    scene = bpy.context.scene
    camera = scene.camera or next(
        (obj for obj in bpy.data.objects if obj.type == "CAMERA"),
        None,
    )
    if camera is None:
        raise RuntimeError("No authored scene camera found")
    scene.camera = camera
    depsgraph = bpy.context.evaluated_depsgraph_get()
    camera_location = camera.matrix_world.translation.copy()

    grid_size = max(3, int(args.grid_size))
    margin_x = (bounds[1] - bounds[0]) * 0.04
    margin_y = (bounds[3] - bounds[2]) * 0.04
    candidates = []
    rejected = {"ground": 0, "visibility": 0, "frame": 0}
    support_points = []
    for ix in range(grid_size):
        x = bounds[0] + margin_x + (
            (bounds[1] - bounds[0] - 2 * margin_x) * ix / (grid_size - 1)
        )
        for iy in range(grid_size):
            y = bounds[2] + margin_y + (
                (bounds[3] - bounds[2] - 2 * margin_y) * iy / (grid_size - 1)
            )
            ray_origin = Vector((x, y, support_height + args.ground_ray_height))
            hit, location, normal, _, hit_obj, _ = scene.ray_cast(
                depsgraph,
                ray_origin,
                Vector((0.0, 0.0, -1.0)),
                distance=args.ground_ray_height + args.ground_height_tolerance + 0.5,
            )
            if (
                not hit
                or hit_obj != support_obj
                or abs(float(location.z) - support_height) > args.ground_height_tolerance
                or float(normal.z) < 0.45
            ):
                rejected["ground"] += 1
                continue
            support_points.append((float(x), float(y), float(location.z), "grid_raycast"))

    # Sparse grid rays miss fragmented HYWorld support patches. Sample actual
    # reconstructed vertices near the contract height as a second source.
    world_matrix = support_obj.matrix_world
    normal_matrix = world_matrix.to_3x3().inverted().transposed()
    vertex_stride = max(1, int(args.vertex_stride))
    seen_cells = {
        (round(point[0] / 0.12), round(point[1] / 0.12))
        for point in support_points
    }
    for vertex_index, vertex in enumerate(support_obj.data.vertices):
        if vertex_index % vertex_stride:
            continue
        world = world_matrix @ vertex.co
        if (
            world.x < bounds[0] + margin_x
            or world.x > bounds[1] - margin_x
            or world.y < bounds[2] + margin_y
            or world.y > bounds[3] - margin_y
            or abs(float(world.z) - support_height) > args.vertex_height_tolerance
        ):
            continue
        world_normal = (normal_matrix @ vertex.normal).normalized()
        if float(world_normal.z) < 0.35:
            continue
        cell = (round(float(world.x) / 0.12), round(float(world.y) / 0.12))
        if cell in seen_cells:
            continue
        seen_cells.add(cell)
        support_points.append((
            float(world.x),
            float(world.y),
            float(world.z),
            "mesh_vertex",
        ))

    for x, y, ground_z, source in support_points:
        target = Vector((x, y, ground_z + args.target_height))
        projection = world_to_camera_view(scene, camera, target)
        if not (
            projection.z > 0
            and 0.04 <= projection.x <= 0.96
            and 0.04 <= projection.y <= 0.96
        ):
            rejected["frame"] += 1
            continue

        ray = target - camera_location
        distance = ray.length
        blocked, _, _, _, _, _ = scene.ray_cast(
            depsgraph,
            camera_location,
            ray.normalized(),
            distance=max(0.0, distance - 0.06),
        )
        if blocked:
            rejected["visibility"] += 1
            continue

        frame_center_distance = math.hypot(
            float(projection.x) - 0.5,
            float(projection.y) - 0.42,
        )
        score = 1.0 - frame_center_distance
        candidates.append({
            "score": round(score, 6),
            "anchor_xyz": [
                round(float(x), 6),
                round(float(y), 6),
                round(float(ground_z), 6),
            ],
            "source": source,
            "camera_projection": [
                round(float(projection.x), 6),
                round(float(projection.y), 6),
                round(float(projection.z), 6),
            ],
            "camera_distance": round(float(distance), 6),
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    result = {
        "schema": "dataevolver.hyworld_visible_support_anchors.v1",
        "scene_contract": str(contract_path),
        "camera": {
            "name": camera.name,
            "location": [round(float(value), 6) for value in camera_location],
        },
        "grid_size": grid_size,
        "support_point_count": len(support_points),
        "candidate_count": len(candidates),
        "rejected": rejected,
        "candidates": candidates,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "camera": result["camera"],
        "candidate_count": len(candidates),
        "best": candidates[:5],
        "rejected": rejected,
    }, indent=2))


if __name__ == "__main__":
    main()
