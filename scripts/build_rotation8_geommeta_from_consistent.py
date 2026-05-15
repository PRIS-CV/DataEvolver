"""
Build a geometry-only dataset from the consistent rotation8 dataset without
modifying the original source root.

Outputs a new parallel directory containing:
- views/obj_xxx/yawYYY.{png,_mask.png,_render_metadata.json,_control.json,_geometry_metadata.json}
- objects/obj_xxx/object_manifest.json
- summary.json
- manifest.json
"""

from __future__ import annotations

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Build geometry-only rotation8 dataset from consistent source")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--blender", dest="blender_bin", required=True)
    parser.add_argument("--scene-template", default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--engine", choices=["EEVEE", "CYCLES"], default=None)
    parser.add_argument("--asset-mode", choices=["copy", "symlink"], default="symlink")
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

            rgb_out = copy_or_link(rgb_src, obj_out_dir / f"{rot_slug}.png", asset_mode)
            mask_out = copy_or_link(mask_src, obj_out_dir / f"{rot_slug}_mask.png", asset_mode)
            meta_out = copy_or_link(meta_src, obj_out_dir / f"{rot_slug}_render_metadata.json", asset_mode)
            control_out = copy_or_link(control_src, obj_out_dir / f"{rot_slug}_control.json", asset_mode)

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
                "base_pair_name": record.get("pair_name"),
                "base_best_round_idx": record.get("base_best_round_idx"),
                "base_hybrid_score": record.get("base_hybrid_score"),
            }
            tasks.append(task)
            rotations_out.append({
                **record,
                "rgb_path": rel(rgb_out, output_root),
                "mask_path": rel(mask_out, output_root),
                "render_metadata_path": rel(meta_out, output_root),
                "control_state_path": rel(control_out, output_root),
                "geometry_metadata_path": rel(obj_out_dir / f"{rot_slug}_geometry_metadata.json", output_root),
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
    source_audit_before = audit_root(source_root)

    scene_template = Path(args.scene_template).resolve() if args.scene_template else (source_root / "scene_template_fixed_camera.json").resolve()
    resolution = int(args.resolution or source_summary.get("resolution") or 1024)
    engine = str(args.engine or source_summary.get("engine") or "CYCLES")

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
        ]
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
        processes.append(proc)
        time.sleep(float(args.launch_stagger_seconds))

    return_codes = [proc.wait() for proc in processes]
    source_audit_after = audit_root(source_root)

    object_manifest_map = {}
    success_items = 0
    for obj_id, payload in prepared_objects.items():
        for record in payload["rotations"]:
            geom_path = output_root / record["geometry_metadata_path"]
            summary_path = output_root / "views" / obj_id / f"yaw{int(record['rotation_deg']):03d}_modal_export_summary.json"
            record["geometry_metadata_exists"] = geom_path.exists()
            record["modal_export_summary_path"] = rel(summary_path, output_root)
            if geom_path.exists():
                success_items += 1
        object_manifest_map[obj_id] = payload
        save_json(output_root / "objects" / obj_id / "object_manifest.json", payload)

    summary = {
        "success": all(code == 0 for code in return_codes),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "dataset_type": "rotation8_geometry_only",
        "asset_mode": args.asset_mode,
        "total_objects": len(prepared_objects),
        "total_views": len(tasks),
        "geometry_metadata_count": success_items,
        "resolution": resolution,
        "engine": engine,
        "scene_template": str(scene_template),
        "source_summary": source_summary,
        "source_root_readonly_audit_before": source_audit_before,
        "source_root_readonly_audit_after": source_audit_after,
    }
    save_json(output_root / "summary.json", summary)
    save_json(output_root / "manifest.json", {"summary": summary, "objects": object_manifest_map})
    return 0 if summary["success"] else 1


def main():
    args = parse_args()
    if args.worker_mode:
        raise SystemExit(run_worker(args))
    raise SystemExit(orchestrate(args))


if __name__ == "__main__":
    main()
