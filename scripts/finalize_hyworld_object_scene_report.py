#!/usr/bin/env python3
"""Finalize and validate the HYWorld object-scene rebuild report.

The script is intentionally evidence-first: it records file hashes, image
statistics, object metadata gates, contact sheets, per-scene lineage, and a
root validation report.  It does not call VLM services and does not treat VLM
output as authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any


DEFAULT_SCENES = ("scene_001", "scene_002")
DEFAULT_OBJECTS = ("obj_001", "obj_009")
DEFAULT_YAWS = (0, 45, 90, 135, 180, 225, 270, 315)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize HYWorld object-scene report")
    parser.add_argument("--dataset-base", required=True)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--scene-dir", default=None)
    parser.add_argument("--template-dir", default=None)
    parser.add_argument("--scenes", type=parse_csv, default=list(DEFAULT_SCENES))
    parser.add_argument("--objects", type=parse_csv, default=list(DEFAULT_OBJECTS))
    parser.add_argument("--qwen-image-model-path", default="/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/Qwen-Image-2512")
    parser.add_argument("--qwen-image-edit-model-path", default="/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/Qwen-Image-Edit-2511")
    parser.add_argument("--hyworld-model-path", default="/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/HYWorld")
    parser.add_argument("--worldmirror-model-path", default="/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/WorldMirror")
    parser.add_argument("--strict-scene-views", action="store_true", help="Fail if scene_views/view_*.png are missing")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "relative_path": rel(path, root),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256(path),
    }


def image_stats(path: Path) -> dict[str, Any]:
    from PIL import Image, ImageStat

    if not path.exists():
        return {"exists": False}
    image = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(image)
    luma = image.convert("L")
    luma_stat = ImageStat.Stat(luma)
    luma_hist = luma.histogram()
    total = max(image.width * image.height, 1)
    nonblack = sum(luma_hist[9:]) / float(total)
    nonwhite = sum(luma_hist[:247]) / float(total)
    extrema = stat.extrema
    return {
        "exists": True,
        "width": image.width,
        "height": image.height,
        "mean": round(float(sum(stat.mean) / len(stat.mean)), 4),
        "std": round(float(sum(stat.stddev) / len(stat.stddev)), 4),
        "luma_std": round(float(luma_stat.stddev[0]), 4),
        "min": int(min(channel_min for channel_min, _ in extrema)),
        "max": int(max(channel_max for _, channel_max in extrema)),
        "nonblack_fraction": round(nonblack, 6),
        "nonwhite_fraction": round(nonwhite, 6),
        "nonflat_pass": bool((sum(stat.stddev) / len(stat.stddev)) > 8.0 and luma_stat.stddev[0] > 6.0),
    }


def mask_stats(path: Path) -> dict[str, Any]:
    from PIL import Image

    if not path.exists():
        return {"exists": False, "nonempty": False}
    image = Image.open(path).convert("L")
    binary = image.point(lambda value: 255 if value > 8 else 0).convert("L")
    bbox = binary.getbbox()
    hist = binary.histogram()
    total = max(image.width * image.height, 1)
    area_fraction = hist[255] / float(total)
    result: dict[str, Any] = {
        "exists": True,
        "width": image.width,
        "height": image.height,
        "nonempty": bool(bbox),
        "area_fraction": round(area_fraction, 6),
    }
    if bbox:
        left, top, right, bottom = bbox
        result["bbox_xyxy"] = [int(left), int(top), int(right - 1), int(bottom - 1)]
        result["bbox_area_fraction"] = round(float((right - left) * (bottom - top) / total), 6)
    return result


def contact_sheet(images: list[tuple[str, Path]], output: Path, *, columns: int = 4, thumb_width: int = 320) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    existing = [(label, path) for label, path in images if path.exists()]
    if not existing:
        return {"created": False, "reason": "no input images", "output": str(output)}
    thumbs: list[tuple[str, Any]] = []
    label_height = 28
    for label, path in existing:
        image = Image.open(path).convert("RGB")
        scale = thumb_width / float(image.width)
        thumb_height = max(1, int(round(image.height * scale)))
        image = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        thumbs.append((label, image))
    rows = int(math.ceil(len(thumbs) / columns))
    cell_height = max(image.height for _, image in thumbs) + label_height
    sheet = Image.new("RGB", (columns * thumb_width, rows * cell_height), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for index, (label, image) in enumerate(thumbs):
        row, col = divmod(index, columns)
        x = col * thumb_width
        y = row * cell_height
        sheet.paste(image, (x, y + label_height))
        draw.text((x + 6, y + 7), label, fill=(235, 235, 235), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return {"created": True, "output": str(output), "image_count": len(existing), "sha256": sha256(output)}


def validate_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "pass": False}
    payload = read_json(path)
    required = {
        "ground_hit_success": True,
        "contract_support_plane_used": False,
        "joint_3d_render": True,
        "is_floating": False,
        "is_intersecting_support": False,
    }
    checks = {key: payload.get(key) == value for key, value in required.items()}
    camera_mode_ok = payload.get("camera_mode") == "scene_camera_match"
    ground_source_ok = payload.get("ground_hit_source") == "mesh_raycast"
    checks["camera_mode_scene_camera_match"] = camera_mode_ok
    checks["ground_hit_source_mesh_raycast"] = ground_source_ok
    return {
        "exists": True,
        "pass": all(checks.values()),
        "checks": checks,
        "selected_fields": {key: payload.get(key) for key in [
            "ground_hit_success",
            "ground_hit_source",
            "contract_support_plane_used",
            "joint_3d_render",
            "is_floating",
            "is_intersecting_support",
            "is_out_of_support_bounds",
            "camera_mode",
        ]},
    }


def object_label(obj_id: str) -> str:
    if obj_id == "obj_001":
        return "04_obj_001"
    if obj_id == "obj_009":
        return "05_obj_009"
    return obj_id


def finalize_scene(
    scene_id: str,
    objects: list[str],
    dataset_base: Path,
    report_dir: Path,
    scene_dir: Path,
    template_dir: Path,
    strict_scene_views: bool,
) -> dict[str, Any]:
    scene_report = report_dir / scene_id
    scene_asset_dir = scene_dir / scene_id
    scene_result: dict[str, Any] = {
        "scene_id": scene_id,
        "scene_report": str(scene_report),
        "files": {},
        "scene_views": {},
        "objects": {},
        "checks": {},
    }
    for name, path in {
        "hyworld_input": scene_report / "01_hyworld_input.png",
        "pano": scene_report / "02_pano.png",
        "pure_3d_scene_render": scene_report / "03_3d_scene_render.png",
        "scene_mesh": scene_asset_dir / "worldmirror_frame0_scene_mesh.ply",
        "scene_calibration": scene_asset_dir / "worldmirror_frame0_scene_mesh.calibration.json",
        "scene_contract": scene_asset_dir / "scene_contract.json",
        "objects_manifest": scene_report / "objects_manifest.json",
        "scene_object_insert_template": scene_report / "scene_object_insert_template.json",
        "summary": scene_report / "summary.json",
        "blender_batch_result": scene_report / "blender_batch_result.json",
        "scene_contract_object_insert": scene_report / "scene_contract_object_insert.json",
    }.items():
        scene_result["files"][name] = file_record(path, dataset_base)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            scene_result["files"][name]["image_stats"] = image_stats(path)

    scene_view_dir = scene_report / "scene_views"
    view_images = sorted(scene_view_dir.glob("view_*.png"))
    scene_result["scene_views"]["count"] = len(view_images)
    scene_result["scene_views"]["images"] = [
        {**file_record(path, dataset_base), "image_stats": image_stats(path)} for path in view_images
    ]
    scene_result["scene_views"]["contact_sheet"] = contact_sheet(
        [(path.stem, path) for path in view_images],
        scene_view_dir / "contact_sheet.png",
        columns=4,
    )

    scene_summary_path = scene_report / "summary.json"
    if scene_summary_path.exists():
        summary = read_json(scene_summary_path)
        scene_result["checks"]["batch_summary_success"] = bool(summary.get("success"))
        scene_result["checks"]["batch_joint_3d_render"] = bool(summary.get("joint_3d_render"))
        scene_result["checks"]["batch_total_renders_16"] = summary.get("total_renders") == 16

    for obj_id in objects:
        obj_dir = scene_report / "objects" / obj_id
        obj_result: dict[str, Any] = {
            "object_dir": str(obj_dir),
            "rotations": {},
            "contact_sheet": {},
            "representative": {},
        }
        yaw_images: list[tuple[str, Path]] = []
        for yaw in DEFAULT_YAWS:
            slug = f"yaw{yaw:03d}"
            rgb = obj_dir / f"{slug}.png"
            mask = obj_dir / f"{slug}_mask.png"
            depth = obj_dir / f"{slug}_depth.png"
            normal = obj_dir / f"{slug}_normal.png"
            metadata = obj_dir / f"{slug}_render_metadata.json"
            yaw_images.append((slug, rgb))
            obj_result["rotations"][slug] = {
                "rgb": {**file_record(rgb, dataset_base), "image_stats": image_stats(rgb)},
                "mask": {**file_record(mask, dataset_base), "mask_stats": mask_stats(mask)},
                "depth": {**file_record(depth, dataset_base), "image_stats": image_stats(depth)},
                "normal": {**file_record(normal, dataset_base), "image_stats": image_stats(normal)},
                "metadata": {**file_record(metadata, dataset_base), "metadata_checks": validate_metadata(metadata)},
            }
        label = object_label(obj_id)
        obj_result["contact_sheet"] = contact_sheet(
            yaw_images,
            scene_report / f"{label}_rotation8_contact.png",
            columns=4,
        )
        representative = scene_report / f"{label}_insert_render.png"
        source = obj_dir / "yaw000.png"
        if source.exists():
            shutil.copy2(source, representative)
        obj_result["representative"] = file_record(representative, dataset_base)
        scene_result["objects"][obj_id] = obj_result

    scene_result["checks"]["required_three_stage_images"] = all(
        scene_result["files"][key]["exists"] for key in ["hyworld_input", "pano", "pure_3d_scene_render"]
    )
    scene_result["checks"]["required_scene_middle_state"] = all(
        scene_result["files"][key]["exists"] for key in ["scene_mesh", "scene_calibration", "scene_contract"]
    )
    scene_result["checks"]["scene_views_present"] = len(view_images) >= 8
    scene_result["checks"]["scene_views_required_pass"] = (len(view_images) >= 8) if strict_scene_views else True
    scene_result["checks"]["objects_rotation8_complete"] = all(
        all(
            obj_data["rotations"][f"yaw{yaw:03d}"]["rgb"]["exists"]
            and obj_data["rotations"][f"yaw{yaw:03d}"]["mask"]["exists"]
            and obj_data["rotations"][f"yaw{yaw:03d}"]["depth"]["exists"]
            and obj_data["rotations"][f"yaw{yaw:03d}"]["normal"]["exists"]
            and obj_data["rotations"][f"yaw{yaw:03d}"]["metadata"]["metadata_checks"]["pass"]
            for yaw in DEFAULT_YAWS
        )
        for obj_data in scene_result["objects"].values()
    )
    scene_result["pass"] = all(scene_result["checks"].values())
    write_json(scene_report / "lineage.json", scene_result)
    return scene_result


def main() -> int:
    args = parse_args()
    dataset_base = Path(args.dataset_base).resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else dataset_base / "reports" / "hyworld_qwen2512_object_insert_20260704_v1"
    scene_dir = Path(args.scene_dir).resolve() if args.scene_dir else dataset_base / "scenes" / "hyworld_qwen2512_depthmesh_camera_rebuild_20260704_v1"
    template_dir = Path(args.template_dir).resolve() if args.template_dir else dataset_base / "templates" / "hyworld_qwen2512_depthmesh_camera_rebuild_20260704_v1"

    scenes = [
        finalize_scene(scene_id, args.objects, dataset_base, report_dir, scene_dir, template_dir, args.strict_scene_views)
        for scene_id in args.scenes
    ]
    manifest = {
        "schema": "dataevolver.hyworld_qwen2512_object_scene_manifest.v1",
        "dataset_base": str(dataset_base),
        "report_dir": str(report_dir),
        "scene_dir": str(scene_dir),
        "template_dir": str(template_dir),
        "models": {
            "qwen_image_2512": args.qwen_image_model_path,
            "qwen_image_edit_2511": args.qwen_image_edit_model_path,
            "hyworld": args.hyworld_model_path,
            "worldmirror": args.worldmirror_model_path,
        },
        "resource_policy": {
            "available_gpus": "0-6",
            "avoid_duplicate_model_weight_loads": True,
            "vlm_pass_authoritative": False,
            "blender_scene_object_rendering_can_be_sharded": True,
        },
        "scene_reconstruction_policy": {
            "current_frame0_mesh": "single_camera_reprojection_baseline",
            "multi_view_gate_required": True,
            "forbidden_fallbacks": [
                "fixed_radius_shell",
                "blurred_stretched_panorama",
                "procedural_support_plane_as_scene",
                "unconditional_multitrajectory_fusion",
            ],
        },
        "scenes": scenes,
    }
    validation = {
        "schema": "dataevolver.hyworld_object_scene_validation.v1",
        "pass": all(scene.get("pass") for scene in scenes),
        "strict_scene_views": bool(args.strict_scene_views),
        "vlm_pass_authoritative": False,
        "scene_count": len(scenes),
        "scenes": {
            scene["scene_id"]: {
                "pass": scene.get("pass"),
                "checks": scene.get("checks"),
                "scene_view_count": scene.get("scene_views", {}).get("count"),
                "objects": {
                    obj_id: {
                        "rotation_count": len(obj_data.get("rotations", {})),
                        "contact_sheet": obj_data.get("contact_sheet", {}),
                        "representative": obj_data.get("representative", {}),
                    }
                    for obj_id, obj_data in scene.get("objects", {}).items()
                },
            }
            for scene in scenes
        },
    }
    write_json(report_dir / "manifest.json", manifest)
    write_json(report_dir / "validation_report.json", validation)
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    return 0 if validation["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
