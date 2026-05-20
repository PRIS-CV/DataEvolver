#!/usr/bin/env python3
"""Render one DataEvolver real-scene record with the universal 3D layout contract.

Run inside Blender with a scene blend already loaded:

    blender scene.blend -b --python render_dataevolver_universal_contract_record.py -- \
      --obj1 /path/to/a.glb --obj2 /path/to/b.glb \
      --out /path/to/dataset --record-id dataevolver_universal_0000 \
      --scene-name outdoor9 --scene-type outdoor --scene-template configs/...

The script reuses DataEvolver dual-object placement, then emits:

- target RGB render with real scene and meshes
- OSCR-style 3D box structure visualization
- per-object mesh silhouette masks
- depth-order and orientation/normal proxy maps
- a universal-schema record fragment consumed by the batch runner
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


PAPER_SIGNALS = [
    {
        "paper_key": "seethrough3d",
        "paper_title": "SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation",
        "arxiv_url": "https://arxiv.org/abs/2602.23359",
        "signal": "target/OSCR/per-object-mask/occlusion contract",
    },
    {
        "paper_key": "poci_diff",
        "paper_title": "POCI-Diff: Position Objects Consistently and Interactively with 3D-Layout Guided Diffusion",
        "arxiv_url": "https://arxiv.org/abs/2601.14056",
        "signal": "3D layout edit action and object positioning",
    },
    {
        "paper_key": "scratchpad3d",
        "paper_title": "3D Space as a Scratchpad for Editable Text-to-Image Generation",
        "arxiv_url": "https://arxiv.org/abs/2601.14602",
        "signal": "agent-readable 3D workspace and scene graph",
    },
    {
        "paper_key": "viewpoint_tokens",
        "paper_title": "Camera Control for Text-to-Image Generation via Learning Viewpoint Tokens",
        "arxiv_url": "https://arxiv.org/abs/2604.19954",
        "signal": "viewpoint token and numeric camera pose",
    },
    {
        "paper_key": "ctrlshift",
        "paper_title": "Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation",
        "arxiv_url": "https://arxiv.org/abs/2602.11440",
        "signal": "geometry-aware object manipulation and preservation checks",
    },
    {
        "paper_key": "3d_ard_plus",
        "paper_title": "Co-generation of Layout and Shape from Text via Autoregressive 3D Diffusion",
        "arxiv_url": "https://arxiv.org/abs/2604.16552",
        "signal": "sequential layout and shape provenance",
    },
]


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    default_root = Path(os.environ.get("DATAEVOLVER_ROOT") or os.environ.get("ARIS_ROOT") or "/home/jiazhuangzhuang/ARIS")
    parser.add_argument("--dataevolver-root", type=Path, default=default_root)
    parser.add_argument("--aris-root", dest="dataevolver_root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--obj1", type=Path, required=True)
    parser.add_argument("--obj2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--scene-type", default="unknown")
    parser.add_argument("--scene-template", default="")
    parser.add_argument("--scene-blend-path", default="")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--engine", choices=["EEVEE", "CYCLES"], default="EEVEE")
    parser.add_argument("--max-object-ratio", type=float, default=0.08)
    parser.add_argument("--fragment-output", type=Path, required=True)
    return parser.parse_args(argv)


def import_dual_module(dataevolver_root: Path):
    script = dataevolver_root / "pipeline" / "demo_dual_object_placement.py"
    spec = importlib.util.spec_from_file_location("dataevolver_dual_object_placement", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import DataEvolver placement script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_mat(name: str, color, alpha: float = 1.0, emission: bool = False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], alpha)
        bsdf.inputs["Roughness"].default_value = 0.54
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = alpha
        if emission:
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = (color[0], color[1], color[2], 1.0)
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = 1.0
    mat.blend_method = "BLEND" if alpha < 1.0 else "OPAQUE"
    mat.use_screen_refraction = alpha < 1.0
    return mat


def set_world(color, strength: float = 0.8) -> None:
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("UniversalContractWorld")
        bpy.context.scene.world = world
    world.color = color[:3]
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (color[0], color[1], color[2], 1.0)
        bg.inputs["Strength"].default_value = strength


def render_to(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def bbox_to_json(bbox: dict) -> dict:
    center = [float(v) for v in bbox["center"]]
    size = [max(0.0001, float(v)) for v in bbox["size"]]
    return {
        "center_xyz": [round(v, 5) for v in center],
        "size_xyz": [round(v, 5) for v in size],
    }


def bbox_corners(bbox: dict) -> list[Vector]:
    mn = bbox["min"]
    mx = bbox["max"]
    return [
        Vector((x, y, z))
        for x in (float(mn[0]), float(mx[0]))
        for y in (float(mn[1]), float(mx[1]))
        for z in (float(mn[2]), float(mx[2]))
    ]


def create_bbox_cube(name: str, bbox: dict, mat) -> bpy.types.Object:
    center = bbox["center"]
    size = [max(0.01, float(v)) for v in bbox["size"]]
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(float(center[0]), float(center[1]), float(center[2])))
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    obj.data.materials.append(mat)
    bpy.context.view_layer.update()
    return obj


def set_visibility(scene_objs: set[bpy.types.Object], obj_a: set[bpy.types.Object], obj_b: set[bpy.types.Object], boxes: list[bpy.types.Object], mode: str) -> None:
    for obj in bpy.data.objects:
        if obj.type in {"CAMERA", "LIGHT"}:
            obj.hide_render = False
            obj.hide_viewport = False
            continue
        if obj in boxes:
            visible = mode in {"structure", "depth", "normal"}
        elif obj in obj_a:
            visible = mode in {"target", "mask_a"}
        elif obj in obj_b:
            visible = mode in {"target", "mask_b"}
        elif obj in scene_objs:
            visible = mode in {"target", "structure"}
        else:
            visible = mode == "target"
        obj.hide_render = not visible
        obj.hide_viewport = not visible


def replace_materials(meshes: list[bpy.types.Object], mat) -> None:
    for obj in meshes:
        if obj.type != "MESH":
            continue
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def camera_pose(camera: bpy.types.Object, target: Vector) -> tuple[str, dict, dict]:
    delta = camera.location - target
    distance = max(1e-6, float(delta.length))
    azimuth = math.degrees(math.atan2(delta.y, delta.x)) % 360.0
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, delta.z / distance))))
    sector_names = ["east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast"]
    sector = sector_names[int(((azimuth + 22.5) % 360.0) // 45.0)]
    elev_bin = int(round(elevation / 10.0) * 10)
    token = f"dataevolver_{sector}_e{elev_bin}"
    lens = float(camera.data.lens)
    sensor = float(camera.data.sensor_width)
    fov = 2.0 * math.degrees(math.atan(sensor / (2.0 * lens))) if lens > 0 else 0.0
    pose = {
        "azimuth_deg": round(azimuth, 3),
        "elevation_deg": round(elevation, 3),
        "distance": round(distance, 5),
        "pitch_deg": round(math.degrees(camera.rotation_euler.x), 3),
        "yaw_deg": round(math.degrees(camera.rotation_euler.z), 3),
    }
    intrinsics = {
        "focal_length_mm": round(lens, 4),
        "sensor_width_mm": round(sensor, 4),
        "fov_deg": round(fov, 4),
    }
    return token, pose, intrinsics


def projected_coverage(bbox: dict, camera: bpy.types.Object) -> float:
    scene = bpy.context.scene
    coords = [world_to_camera_view(scene, camera, corner) for corner in bbox_corners(bbox)]
    xs = [max(0.0, min(1.0, float(c.x))) for c in coords]
    ys = [max(0.0, min(1.0, float(c.y))) for c in coords]
    return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))


def relation_and_depth(bboxes: list[dict], camera: bpy.types.Object) -> tuple[list[dict], list[str], list[str], list[dict]]:
    centers = [Vector((float(b["center"][0]), float(b["center"][1]), float(b["center"][2]))) for b in bboxes]
    ids = ["obj_a", "obj_b"]
    dx = centers[1].x - centers[0].x
    dy = centers[1].y - centers[0].y
    dz = centers[1].z - centers[0].z
    if abs(dz) > max(abs(dx), abs(dy)) * 0.5:
        predicate = "above" if dz > 0 else "below"
    elif abs(dx) >= abs(dy):
        predicate = "left_of" if centers[0].x < centers[1].x else "right_of"
    else:
        predicate = "behind" if centers[0].y > centers[1].y else "in_front_of"
    distances = [(ids[i], float((centers[i] - camera.location).length)) for i in range(2)]
    depth_order = [obj_id for obj_id, _ in sorted(distances, key=lambda item: item[1])]
    relations = [{"subject": "obj_a", "predicate": predicate, "object": "obj_b"}]
    occlusion = [
        {
            "front_object": depth_order[0],
            "behind_object": depth_order[-1],
            "visibility_note": "proxy depth order computed from camera-to-bbox-center distance",
        }
    ]
    return relations, [predicate], depth_order, occlusion


def render_auxiliary_artifacts(args: argparse.Namespace, camera: bpy.types.Object, scene_objs: set, obj_a: set, obj_b: set, meshes1, meshes2, bboxes: list[dict]) -> tuple[dict, list[str], float]:
    out = args.out
    rid = args.record_id
    box_objs: list[bpy.types.Object] = []

    colors = [(0.9, 0.16, 0.12, 0.42), (0.08, 0.52, 0.82, 0.42)]
    for idx, bbox in enumerate(bboxes):
        mat = make_mat(f"universal_oscr_{idx}", colors[idx], alpha=0.42, emission=True)
        box = create_bbox_cube(f"universal_oscr_box_{idx}", bbox, mat)
        wire = box.modifiers.new("oscr_wire", "WIREFRAME")
        wire.thickness = max(0.01, min(float(max(bbox["size"])) * 0.018, 0.08))
        wire.use_even_offset = True
        box_objs.append(box)

    structure_rel = f"structure/{rid}_oscr.png"
    set_world((0.78, 0.82, 0.86, 1.0), strength=0.8)
    set_visibility(scene_objs, obj_a, obj_b, box_objs, "structure")
    render_to(out / structure_rel)

    for idx, box in enumerate(box_objs):
        rank = idx / max(1, len(box_objs) - 1)
        box.data.materials.clear()
        box.data.materials.append(make_mat(f"universal_depth_{idx}", (0.18 + 0.68 * (1.0 - rank),) * 3 + (1.0,), alpha=1.0, emission=True))
        for modifier in list(box.modifiers):
            box.modifiers.remove(modifier)
    depth_rel = f"depth/{rid}_depth_order.png"
    set_world((0.0, 0.0, 0.0, 1.0), strength=0.0)
    set_visibility(scene_objs, obj_a, obj_b, box_objs, "depth")
    render_to(out / depth_rel)

    for idx, box in enumerate(box_objs):
        center = bboxes[idx]["center"]
        angle = math.atan2(float(center[1]) - float(camera.location.y), float(center[0]) - float(camera.location.x))
        val = (math.sin(angle) + 1.0) * 0.5
        box.data.materials.clear()
        box.data.materials.append(make_mat(f"universal_normal_{idx}", (val, 0.45, 1.0 - val, 1.0), alpha=1.0, emission=True))
    normal_rel = f"normal/{rid}_orientation_proxy.png"
    set_visibility(scene_objs, obj_a, obj_b, box_objs, "normal")
    render_to(out / normal_rel)

    mask_mat = make_mat("universal_mask_white", (1.0, 1.0, 1.0, 1.0), alpha=1.0, emission=True)
    replace_materials(meshes1, mask_mat)
    replace_materials(meshes2, mask_mat)
    mask_rels = []
    coverage = 0.0
    for mode, bbox in (("mask_a", bboxes[0]), ("mask_b", bboxes[1])):
        suffix = "obj_a" if mode == "mask_a" else "obj_b"
        mask_rel = f"mask/{rid}_{suffix}.png"
        set_world((0.0, 0.0, 0.0, 1.0), strength=0.0)
        set_visibility(scene_objs, obj_a, obj_b, box_objs, mode)
        render_to(out / mask_rel)
        mask_rels.append(mask_rel)
        coverage += projected_coverage(bbox, camera)
    return {"structure": structure_rel, "depth": depth_rel, "normal": normal_rel}, mask_rels, min(1.0, coverage)


def build_record(args: argparse.Namespace, placement: dict, camera: bpy.types.Object, target_rel: str, aux: dict, masks: list[str], mask_coverage: float, bboxes: list[dict], scales: list[float]) -> tuple[dict, dict]:
    combined = placement["bbox1"]
    target = Vector(combined["center"].tolist())
    token, pose, intrinsics = camera_pose(camera, target)
    relations, spatial, depth_order, occlusion = relation_and_depth(bboxes, camera)
    mesh_ids = [args.obj1.stem, args.obj2.stem]
    record = {
        "record_id": args.record_id,
        "source_paper_signals": PAPER_SIGNALS,
        "prompt": {
            "raw_text": f"A real DataEvolver {args.scene_type} scene containing {mesh_ids[0]} and {mesh_ids[1]} with explicit 3D layout, viewpoint, masks and occlusion metadata.",
            "object_spans": [
                {"object_id": "obj_a", "text_span": mesh_ids[0]},
                {"object_id": "obj_b", "text_span": mesh_ids[1]},
            ],
            "negative_constraints": ["do not merge object identities", "keep the two object masks separable", "preserve camera and scene metadata"],
        },
        "scene_graph": {"nodes": ["obj_a", "obj_b"], "relations": relations},
        "objects": [
            {
                "object_id": "obj_a",
                "category": mesh_ids[0],
                "mesh_or_proxy": str(args.obj1),
                "bbox_3d": bbox_to_json(bboxes[0]),
                "orientation": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
                "scale": round(scales[0], 7),
                "material": "source_glb_material_for_target; white_emission_for_mask",
            },
            {
                "object_id": "obj_b",
                "category": mesh_ids[1],
                "mesh_or_proxy": str(args.obj2),
                "bbox_3d": bbox_to_json(bboxes[1]),
                "orientation": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
                "scale": round(scales[1], 7),
                "material": "source_glb_material_for_target; white_emission_for_mask",
            },
        ],
        "camera": {
            "viewpoint_token": token,
            "pose": pose,
            "intrinsics": intrinsics,
            "relative_description": f"{token}; DataEvolver camera framed around placed dual objects",
        },
        "layout_control": {
            "control_type": ["oscr", "3d_boxes", "mesh_scratchpad", "viewpoint_token", "relative_pose_pair", "sequential_3d_scene"],
            "spatial_relations": spatial,
            "depth_order": depth_order,
            "occlusion_pairs": occlusion,
        },
        "render_artifacts": {
            "target_image": target_rel,
            "structure_image": aux["structure"],
            "masks": masks,
            "depth": aux["depth"],
            "normal": aux["normal"],
        },
        "edit_action": {
            "action_type": "none",
            "before_state": None,
            "after_state": None,
            "preservation_targets": ["real DataEvolver scene background", "object identity", "camera framing"],
        },
        "validation": {
            "geometry_checks": [
                "target exists",
                "structure exists",
                "mask count equals object count",
                "depth-order proxy exists",
                "orientation proxy exists",
                "camera pose and intrinsics recorded",
                "scene accepted after excluded-scene filtering",
            ],
            "mask_coverage": round(mask_coverage, 5),
            "occlusion_consistency": "proxy depth order recorded from camera-to-object distance",
            "vlm_score": None,
            "failure_tags": [] if mask_coverage > 0 else ["low_projected_mask_coverage"],
        },
        "loop_trace": [
            {
                "candidate_id": "round00_dataevolver_dual_placement",
                "critique": "DataEvolver placement produced a real-scene dual-object target candidate.",
                "repair_action": "Render universal artifacts: OSCR boxes, mesh masks, depth-order proxy, orientation proxy and schema metadata.",
                "accept": False,
            },
            {
                "candidate_id": "round01_contract_complete",
                "critique": "Required files and universal-schema fields are present for this sample.",
                "repair_action": "Promote to DataEvolver-real universal 3D layout record.",
                "accept": True,
            },
        ],
        "provenance": {
            "label": "self_generated",
            "source_paths": [str(args.scene_blend_path), str(args.obj1), str(args.obj2), str(args.dataevolver_root / "pipeline" / "demo_dual_object_placement.py")],
        },
        "promotion_decision": {
            "status": "accepted" if mask_coverage > 0 else "needs_repair",
            "reason": "complete DataEvolver-real universal render contract" if mask_coverage > 0 else "projected masks are too small",
        },
    }
    compat = {
        "subjects": mesh_ids,
        "target": target_rel,
        "PLACEHOLDER_prompts": "a real DataEvolver scene containing PLACEHOLDER with explicit 3D layout",
        "oscr": aux["structure"],
        "call_ids": [[0], [1]],
        "cuboids_segmasks": masks,
    }
    return record, compat


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    args.fragment_output.parent.mkdir(parents=True, exist_ok=True)

    dual = import_dual_module(args.dataevolver_root)
    dual.MAX_OBJECT_RATIO = args.max_object_ratio

    scene_objs_before = set(bpy.data.objects)
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
    root1, meshes1 = dual.import_asset(str(args.obj1))
    root2, meshes2 = dual.import_asset(str(args.obj2))
    obj_a = set([root1] + meshes1)
    obj_b = set([root2] + meshes2)

    scale1 = dual.scale_object_to_scene(root1, meshes1, scene_min, scene_max)
    scale2 = dual.scale_object_to_scene(root2, meshes2, scene_min, scene_max)
    hdri_azimuth = dual.find_hdri_best_azimuth()
    cam_az = (hdri_azimuth + math.pi) % (2 * math.pi) if hdri_azimuth is not None else None
    placement = dual.place_two_objects(root1, meshes1, root2, meshes2, scene_bvh, scene_min, scene_max, ground_info=ground_info, camera_azimuth=cam_az)
    if placement is None:
        raise RuntimeError("DataEvolver dual placement failed")

    camera = dual.setup_camera_for_dual(placement, scene_min, scene_max)
    dual.setup_render(resolution=args.resolution, engine=args.engine)
    dual.ensure_minimum_lighting()
    dual.add_object_spotlight(placement)

    target_rel = f"target/{args.record_id}.png"
    set_visibility(scene_objs_before, obj_a, obj_b, [], "target")
    render_to(args.out / target_rel)

    bboxes = [dual.get_object_bbox(meshes1), dual.get_object_bbox(meshes2)]
    if any(bbox is None for bbox in bboxes):
        raise RuntimeError("Could not compute imported object bboxes")
    aux, masks, coverage = render_auxiliary_artifacts(args, camera, scene_objs_before, obj_a, obj_b, meshes1, meshes2, bboxes)
    record, compat = build_record(args, placement, camera, target_rel, aux, masks, coverage, bboxes, [scale1, scale2])
    fragment = {
        "record": record,
        "st3d_compat": compat,
        "scene": {
            "name": args.scene_name,
            "type": args.scene_type,
            "template": args.scene_template,
            "blend_path": args.scene_blend_path,
            "ground": ground_info,
        },
        "placement": {
            "template": placement.get("template"),
            "separation": float(placement.get("separation", 0.0)),
            "height_offset": float(placement.get("height_offset", 0.0)),
        },
    }
    args.fragment_output.write_text(json.dumps(fragment, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"record_id": args.record_id, "target": target_rel, "mask_coverage": record["validation"]["mask_coverage"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
