"""
Build a geometry-complete train-ready rotation dataset from a geometry-only
rotation8 dataset root.

Outputs a new parallel dataset root containing:
- views/obj_xxx/yawYYY.(png|mask|render_metadata|control|geometry_metadata|depth|normal)
- objects/obj_xxx/object_manifest.json
- pairs/train_pairs.csv
- pairs/train_pairs.jsonl
- summary.json
- manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
MODAL_SCRIPT = REPO_ROOT / "pipeline" / "stage4_scene_modal_export.py"

VIEW_NAMES = {
    0: "front view",
    45: "front-right view",
    90: "right side view",
    135: "back-right view",
    180: "back view",
    225: "back-left view",
    270: "left side view",
    315: "front-left view",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build geomodal train-ready rotation8 dataset")
    parser.add_argument("--source-root", required=True, help="Geometry-only rotation8 root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--blender", dest="blender_bin", required=True)
    parser.add_argument("--scene-template", default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--engine", choices=["EEVEE", "CYCLES"], default=None)
    parser.add_argument("--asset-mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--source-rotation-deg", type=int, default=0)
    parser.add_argument("--target-rotations", default="45,90,135,180,225,270,315")
    parser.add_argument("--depth-normal-samples", type=int, default=1)
    parser.add_argument("--launch-stagger-seconds", type=float, default=1.0)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-gpu", default=None)
    parser.add_argument("--worker-manifest-path", default=None)
    parser.add_argument("--worker-log-path", default=None)
    return parser.parse_args()


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def copy_or_link(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)
    return dst


def parse_gpus(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_rotations(raw: str) -> List[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def audit_root(path: Path) -> dict:
    files = [p for p in path.rglob("*") if p.is_file()]
    latest = max((p.stat().st_mtime for p in files), default=0.0)
    return {
        "root": str(path),
        "file_count": len(files),
        "latest_mtime": latest,
    }


def view_name(rotation_deg: int) -> str:
    return VIEW_NAMES.get(int(rotation_deg), f"{int(rotation_deg)} degree view")


def prepare_tasks(source_root: Path, output_root: Path, asset_mode: str) -> tuple[list[dict], dict]:
    manifest = load_json(source_root / "manifest.json")
    if not manifest:
        raise FileNotFoundError(f"manifest.json not found in {source_root}")

    objects_payload = manifest.get("objects") or {}
    tasks = []
    prepared_objects = {}

    for obj_id in sorted(objects_payload.keys()):
        source_obj_manifest = load_json(source_root / "objects" / obj_id / "object_manifest.json", default={}) or {}
        obj_out_dir = output_root / "views" / obj_id
        obj_out_dir.mkdir(parents=True, exist_ok=True)

        rotations_out = []
        for record in source_obj_manifest.get("rotations", []):
            rotation_deg = int(record["rotation_deg"])
            rot_slug = f"yaw{rotation_deg:03d}"

            rgb_src = source_root / record["rgb_path"]
            mask_src = source_root / record["mask_path"]
            meta_src = source_root / record["render_metadata_path"]
            control_src = source_root / record["control_state_path"]
            geom_src = source_root / record["geometry_metadata_path"]

            rgb_out = copy_or_link(rgb_src, obj_out_dir / f"{rot_slug}.png", asset_mode)
            mask_out = copy_or_link(mask_src, obj_out_dir / f"{rot_slug}_mask.png", asset_mode)
            meta_out = copy_or_link(meta_src, obj_out_dir / f"{rot_slug}_render_metadata.json", asset_mode)
            control_out = copy_or_link(control_src, obj_out_dir / f"{rot_slug}_control.json", asset_mode)
            geom_out = copy_or_link(geom_src, obj_out_dir / f"{rot_slug}_geometry_metadata.json", asset_mode)

            render_meta = load_json(meta_src, default={}) or {}
            asset_path = str(render_meta.get("source_asset") or "")
            if not asset_path:
                raise ValueError(f"Missing source_asset in {meta_src}")
            if not os.path.isabs(asset_path):
                asset_path = str((REPO_ROOT / asset_path).resolve())

            task = {
                "obj_id": obj_id,
                "rotation_deg": rotation_deg,
                "output_stem": rot_slug,
                "asset_path": asset_path,
                "output_dir": str(obj_out_dir),
                "control_state_path": str(control_out),
                "rgb_path": rel(rgb_out, output_root),
                "mask_path": rel(mask_out, output_root),
                "render_metadata_path": rel(meta_out, output_root),
                "control_path": rel(control_out, output_root),
                "geometry_metadata_path": rel(geom_out, output_root),
                "depth_exr_path": rel(obj_out_dir / f"{rot_slug}_depth.exr", output_root),
                "depth_vis_path": rel(obj_out_dir / f"{rot_slug}_depth_vis.png", output_root),
                "normal_exr_path": rel(obj_out_dir / f"{rot_slug}_normal.exr", output_root),
                "normal_png_path": rel(obj_out_dir / f"{rot_slug}_normal.png", output_root),
                "base_pair_name": record.get("pair_name"),
                "base_best_round_idx": record.get("base_best_round_idx"),
                "base_hybrid_score": record.get("base_hybrid_score"),
            }
            tasks.append(task)
            rotations_out.append({
                "rotation_deg": rotation_deg,
                "view_name": view_name(rotation_deg),
                "rgb_path": rel(rgb_out, output_root),
                "mask_path": rel(mask_out, output_root),
                "render_metadata_path": rel(meta_out, output_root),
                "control_state_path": rel(control_out, output_root),
                "geometry_metadata_path": rel(geom_out, output_root),
                "depth_exr_path": task["depth_exr_path"],
                "depth_vis_path": task["depth_vis_path"],
                "normal_exr_path": task["normal_exr_path"],
                "normal_png_path": task["normal_png_path"],
                "base_pair_name": record.get("pair_name"),
                "base_best_round_idx": record.get("base_best_round_idx"),
                "base_hybrid_score": record.get("base_hybrid_score"),
            })

        prepared_objects[obj_id] = {
            "obj_id": obj_id,
            "base_selection": source_obj_manifest.get("base_selection"),
            "rotations": rotations_out,
        }

    return tasks, prepared_objects


def shard_tasks(tasks: List[dict], gpus: List[str]) -> List[dict]:
    buckets = [{"gpu": gpu, "items": []} for gpu in gpus]
    for idx, task in enumerate(tasks):
        buckets[idx % len(gpus)]["items"].append(task)
    return [b for b in buckets if b["items"]]


def run_worker(args) -> int:
    items_manifest = load_json(Path(args.worker_manifest_path), default={}) or {}
    items = items_manifest.get("items") or []
    worker_log_path = Path(args.worker_log_path)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.worker_gpu)

    status = {
        "status": "running",
        "worker_gpu": args.worker_gpu,
        "completed_items": 0,
        "total_items": len(items),
        "current_item": None,
        "failures": [],
    }
    save_json(worker_log_path, status)

    for item in items:
        status["current_item"] = f"{item['obj_id']}:{item['output_stem']}"
        save_json(worker_log_path, status)
        cmd = [
            args.blender_bin,
            "-b",
            "-P",
            str(MODAL_SCRIPT),
            "--",
            "--asset-path",
            item["asset_path"],
            "--output-dir",
            item["output_dir"],
            "--obj-id",
            item["obj_id"],
            "--output-stem",
            item["output_stem"],
            "--resolution",
            str(args.resolution),
            "--engine",
            args.engine,
            "--control-state",
            item["control_state_path"],
            "--scene-template",
            args.scene_template,
            "--write-depth-normal",
            "--depth-normal-samples",
            str(args.depth_normal_samples),
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.returncode != 0:
            status["failures"].append({
                "item": status["current_item"],
                "return_code": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
            })
        else:
            status["completed_items"] += 1
        save_json(worker_log_path, status)

    status["status"] = "completed" if not status["failures"] else "completed_with_failures"
    status["current_item"] = None
    save_json(worker_log_path, status)
    return 0 if not status["failures"] else 1


def orchestrate(args) -> int:
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "_logs").mkdir(parents=True, exist_ok=True)

    source_summary = load_json(source_root / "summary.json", default={}) or {}
    scene_template = Path(args.scene_template).resolve() if args.scene_template else Path(
        source_summary.get("scene_template") or REPO_ROOT / "configs" / "scene_template.json"
    ).resolve()
    resolution = int(args.resolution or source_summary.get("resolution") or 1024)
    engine = str(args.engine or source_summary.get("engine") or "CYCLES")
    source_audit_before = audit_root(source_root)

    tasks, prepared_objects = prepare_tasks(source_root, output_root, args.asset_mode)
    gpus = parse_gpus(args.gpus)
    shards = shard_tasks(tasks, gpus)

    processes = []
    for shard in shards:
        gpu = shard["gpu"]
        slug = str(gpu).replace(":", "_")
        worker_manifest_path = output_root / "_logs" / f"worker_gpu_{slug}_manifest.json"
        worker_log_path = output_root / "_logs" / f"worker_gpu_{slug}.json"
        save_json(worker_manifest_path, {"items": shard["items"]})
        cmd = [
            args.python_bin,
            str(SCRIPT_PATH),
            "--source-root",
            str(source_root),
            "--output-dir",
            str(output_root),
            "--worker-mode",
            "--worker-gpu",
            str(gpu),
            "--worker-manifest-path",
            str(worker_manifest_path),
            "--worker-log-path",
            str(worker_log_path),
            "--blender",
            args.blender_bin,
            "--scene-template",
            str(scene_template),
            "--resolution",
            str(resolution),
            "--engine",
            engine,
            "--depth-normal-samples",
            str(args.depth_normal_samples),
        ]
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
        processes.append(proc)
        time.sleep(float(args.launch_stagger_seconds))

    return_codes = [proc.wait() for proc in processes]
    source_audit_after = audit_root(source_root)

    target_rotations = parse_rotations(args.target_rotations)
    source_rotation_deg = int(args.source_rotation_deg)
    source_slug = f"yaw{source_rotation_deg:03d}"
    object_manifest_map = {}
    pair_rows = []
    rendered_depth_count = 0
    rendered_normal_count = 0

    for obj_id, payload in prepared_objects.items():
        rotation_map = {f"yaw{int(item['rotation_deg']):03d}": item for item in payload["rotations"]}
        for item in payload["rotations"]:
            if (output_root / item["depth_exr_path"]).exists():
                rendered_depth_count += 1
            if (output_root / item["normal_exr_path"]).exists():
                rendered_normal_count += 1

        source_view = rotation_map[source_slug]
        training_pairs = []
        for target_rotation in target_rotations:
            target_slug = f"yaw{int(target_rotation):03d}"
            target_view = rotation_map[target_slug]
            pair_id = f"{obj_id}_{source_slug}_to_{target_slug}"
            pair_payload = {
                "pair_id": pair_id,
                "obj_id": obj_id,
                "task_type": "rotation_edit",
                "split": "train",
                "source_rotation_deg": source_rotation_deg,
                "target_rotation_deg": int(target_rotation),
                "source_view_name": view_name(source_rotation_deg),
                "target_view_name": view_name(target_rotation),
                "instruction": f"Rotate this object from front view to {view_name(target_rotation)}.",
                "source_image": source_view["rgb_path"],
                "target_image": target_view["rgb_path"],
                "source_mask": source_view["mask_path"],
                "target_mask": target_view["mask_path"],
                "source_render_metadata": source_view["render_metadata_path"],
                "target_render_metadata": target_view["render_metadata_path"],
                "source_control_state": source_view["control_state_path"],
                "target_control_state": target_view["control_state_path"],
                "source_geometry_metadata": source_view["geometry_metadata_path"],
                "target_geometry_metadata": target_view["geometry_metadata_path"],
                "source_depth": source_view["depth_exr_path"],
                "target_depth": target_view["depth_exr_path"],
                "source_normal": source_view["normal_exr_path"],
                "target_normal": target_view["normal_exr_path"],
                "source_normal_vis": source_view["normal_png_path"],
                "target_normal_vis": target_view["normal_png_path"],
                "source_depth_vis": source_view["depth_vis_path"],
                "target_depth_vis": target_view["depth_vis_path"],
            }
            pair_rows.append(pair_payload)
            training_pairs.append(pair_payload)

        object_manifest = {
            "obj_id": obj_id,
            "source_rotation_deg": source_rotation_deg,
            "source_view_name": view_name(source_rotation_deg),
            "target_rotations": target_rotations,
            "base_selection": payload.get("base_selection"),
            "views": rotation_map,
            "training_pairs": training_pairs,
        }
        save_json(output_root / "objects" / obj_id / "object_manifest.json", object_manifest)
        object_manifest_map[obj_id] = object_manifest

    pair_rows = sorted(pair_rows, key=lambda item: item["pair_id"])
    write_jsonl(output_root / "pairs" / "train_pairs.jsonl", pair_rows)
    write_csv(output_root / "pairs" / "train_pairs.csv", pair_rows)

    summary = {
        "success": all(code == 0 for code in return_codes),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_type": "rotation8_geomodal_trainready",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "asset_mode": args.asset_mode,
        "resolution": resolution,
        "engine": engine,
        "scene_template": str(scene_template),
        "source_rotation_deg": source_rotation_deg,
        "source_view_name": view_name(source_rotation_deg),
        "target_rotations": target_rotations,
        "target_view_names": [view_name(rot) for rot in target_rotations],
        "total_objects": len(prepared_objects),
        "rendered_views": len(tasks),
        "rendered_masks": len(tasks),
        "geometry_metadata_count": len(tasks),
        "depth_exr_count": rendered_depth_count,
        "normal_exr_count": rendered_normal_count,
        "training_pairs": len(pair_rows),
        "pairs_per_object": len(target_rotations),
        "source_summary": source_summary,
        "source_root_readonly_audit_before": source_audit_before,
        "source_root_readonly_audit_after": source_audit_after,
    }
    save_json(output_root / "summary.json", summary)
    save_json(output_root / "manifest.json", {"summary": summary, "objects": object_manifest_map, "pairs": pair_rows})
    return 0 if summary["success"] else 1


def main():
    args = parse_args()
    if args.worker_mode:
        raise SystemExit(run_worker(args))
    raise SystemExit(orchestrate(args))


if __name__ == "__main__":
    main()
