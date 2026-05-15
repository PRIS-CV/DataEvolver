"""
Export scene-aware multi-view renders from an existing evolution result folder.

This reuses the same stage4_scene_render.py pipeline, scene template, and
per-object control states from a previous evolution run, but expands the output
to a larger QC view set for dataset construction.
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
from typing import Dict, List, Optional


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
SCENE_RENDER_SCRIPT = REPO_ROOT / "pipeline" / "stage4_scene_render.py"
DEFAULT_BLENDER = os.environ.get("BLENDER_BIN", "blender")
DEFAULT_TEMPLATE = REPO_ROOT / "configs" / "scene_template.json"
DEFAULT_MESHES_DIR = REPO_ROOT / "pipeline" / "data" / "meshes"
FULL48_QC_VIEWS = [[az, el] for az in range(0, 360, 30) for el in (-15, 0, 15, 30)]


def parse_args():
    parser = argparse.ArgumentParser(description="Export scene-aware multiview renders from a prior evolution run")
    parser.add_argument("--source-evolution-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--obj-ids", nargs="+", required=True)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--blender", dest="blender_bin", default=DEFAULT_BLENDER)
    parser.add_argument("--meshes-dir", default=str(DEFAULT_MESHES_DIR))
    parser.add_argument("--scene-template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--camera-mode", default="object_bbox_relative",
                        choices=["object_bbox_relative", "scene_bounds_relative", "scene_camera_match"])
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--engine", choices=["EEVEE", "CYCLES"], default=None)
    parser.add_argument("--launch-stagger-seconds", type=float, default=2.0)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-gpu", default=None)
    parser.add_argument("--worker-shard-dir", default=None)
    parser.add_argument("--worker-log-path", default=None)
    parser.add_argument("--export-template-path", default=None)
    return parser.parse_args()


def parse_gpus(raw: str) -> List[str]:
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise ValueError("No GPUs specified")
    return gpus


def shard_obj_ids(obj_ids: List[str], gpus: List[str]) -> List[Dict[str, object]]:
    buckets = [{"gpu": gpu, "obj_ids": []} for gpu in gpus]
    for index, obj_id in enumerate(obj_ids):
        buckets[index % len(gpus)]["obj_ids"].append(obj_id)
    return [bucket for bucket in buckets if bucket["obj_ids"]]


def slugify_gpu(gpu: str) -> str:
    return gpu.replace(":", "_").replace("/", "_").replace("\\", "_")


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def candidate_control_paths(source_root: Path, obj_id: str) -> List[Path]:
    obj_root = source_root / obj_id
    candidates: List[Path] = []

    for pattern in ["attempt*_renders", "probe*_renders"]:
        render_dirs = sorted(obj_root.glob(pattern), reverse=True)
        for render_dir in render_dirs:
            candidates.append(render_dir / f"_control_{obj_id}.json")

    candidates.extend([
        obj_root / "states" / "control_state_best.json",
        obj_root / "states" / "cs_attempt_99.json",
        obj_root / "states" / "cs_attempt_01.json",
        obj_root / "states" / "cs_attempt_00.json",
    ])
    return candidates


def resolve_control_state(source_root: Path, obj_id: str) -> Optional[Path]:
    for candidate in candidate_control_paths(source_root, obj_id):
        if candidate.exists():
            return candidate
    return None


def build_export_template(args, output_root: Path) -> Path:
    base_template = load_json(Path(args.scene_template), default={}) or {}
    if args.engine:
        base_template["render_engine"] = args.engine
    if args.resolution:
        base_template["render_resolution"] = args.resolution
    base_template["qc_views"] = FULL48_QC_VIEWS
    base_template["camera_mode"] = args.camera_mode

    export_template_path = output_root / "scene_template_multiview.json"
    save_json(export_template_path, base_template)
    return export_template_path


def run_worker(args) -> int:
    assert args.worker_gpu is not None
    assert args.worker_shard_dir is not None
    assert args.export_template_path is not None

    source_root = Path(args.source_evolution_dir).resolve()
    shard_dir = Path(args.worker_shard_dir).resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.worker_gpu

    worker_summary = {
        "gpu": args.worker_gpu,
        "obj_ids": args.obj_ids,
        "source_evolution_dir": str(source_root),
        "export_template_path": args.export_template_path,
        "started_at": time.time(),
        "objects": {},
        "success": True,
    }

    for obj_id in args.obj_ids:
        control_state_path = resolve_control_state(source_root, obj_id)
        if control_state_path is None:
            worker_summary["objects"][obj_id] = {
                "return_code": -1,
                "error": "control_state_not_found",
            }
            worker_summary["success"] = False
            worker_summary["failed_obj_id"] = obj_id
            break

        cmd = [
            args.blender_bin,
            "-b",
            "-P",
            str(SCENE_RENDER_SCRIPT),
            "--",
            "--input-dir",
            args.meshes_dir,
            "--output-dir",
            str(shard_dir),
            "--obj-id",
            obj_id,
            "--resolution",
            str(args.resolution or 1024),
            "--engine",
            args.engine or "CYCLES",
            "--control-state",
            str(control_state_path),
            "--scene-template",
            args.export_template_path,
        ]
        started = time.time()
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        object_dir = shard_dir / obj_id
        metadata_path = object_dir / "metadata.json"
        metadata = load_json(metadata_path, default={}) or {}
        worker_summary["objects"][obj_id] = {
            "return_code": completed.returncode,
            "elapsed_seconds": round(time.time() - started, 3),
            "control_state_path": str(control_state_path),
            "object_dir": str(object_dir),
            "metadata_path": str(metadata_path) if metadata_path.exists() else None,
            "total_frames": len(metadata.get("frames", [])),
            "camera_mode": metadata.get("camera_mode"),
            "stdout_tail": completed.stdout[-4000:],
        }
        if completed.returncode != 0:
            worker_summary["success"] = False
            worker_summary["failed_obj_id"] = obj_id
            break

    worker_summary["elapsed_seconds"] = round(time.time() - worker_summary["started_at"], 3)
    save_json(Path(args.worker_log_path), worker_summary)
    return 0 if worker_summary["success"] else 1


def move_object_dirs(output_root: Path, shard_dir: Path, obj_ids: List[str]) -> Dict[str, str]:
    moved = {}
    for obj_id in obj_ids:
        src = shard_dir / obj_id
        if not src.exists():
            continue
        dst = output_root / obj_id
        if dst.exists():
            raise FileExistsError(f"Destination already exists: {dst}")
        shutil.move(str(src), str(dst))
        moved[obj_id] = str(dst)
    return moved


def run_orchestrator(args) -> int:
    source_root = Path(args.source_evolution_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    shard_root = output_root / "_shards"
    log_root = output_root / "_logs"
    shard_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    export_template_path = build_export_template(args, output_root)
    sample_template = load_json(export_template_path, default={}) or {}
    effective_resolution = int(args.resolution or sample_template.get("render_resolution", 1024))
    effective_engine = str(args.engine or sample_template.get("render_engine", "CYCLES"))

    request = {
        "source_evolution_dir": str(source_root),
        "output_dir": str(output_root),
        "obj_ids": args.obj_ids,
        "gpus": parse_gpus(args.gpus),
        "meshes_dir": os.path.abspath(args.meshes_dir),
        "scene_template": str(export_template_path),
        "camera_mode": args.camera_mode,
        "resolution": effective_resolution,
        "engine": effective_engine,
        "qc_views": FULL48_QC_VIEWS,
        "requested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(output_root / "export_request.json", request)

    launched = []
    for index, shard in enumerate(shard_obj_ids(args.obj_ids, parse_gpus(args.gpus))):
        gpu = str(shard["gpu"])
        obj_ids = list(shard["obj_ids"])
        gpu_slug = slugify_gpu(gpu)
        shard_dir = shard_root / f"shard_gpu_{gpu_slug}"
        log_path = log_root / f"worker_gpu_{gpu_slug}.json"
        shard_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python_bin,
            str(SCRIPT_PATH),
            "--worker-mode",
            "--source-evolution-dir",
            str(source_root),
            "--output-dir",
            str(output_root),
            "--obj-ids",
            *obj_ids,
            "--meshes-dir",
            args.meshes_dir,
            "--blender",
            args.blender_bin,
            "--gpus",
            gpu,
            "--camera-mode",
            args.camera_mode,
            "--resolution",
            str(effective_resolution),
            "--engine",
            effective_engine,
            "--worker-gpu",
            gpu,
            "--worker-shard-dir",
            str(shard_dir),
            "--worker-log-path",
            str(log_path),
            "--export-template-path",
            str(export_template_path),
        ]
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        launched.append({
            "gpu": gpu,
            "obj_ids": obj_ids,
            "process": process,
            "shard_dir": shard_dir,
            "log_path": log_path,
        })
        if index < len(args.obj_ids) - 1 and args.launch_stagger_seconds > 0:
            time.sleep(args.launch_stagger_seconds)

    overall_success = True
    merged_object_dirs = {}
    merged_objects = {}

    for shard in launched:
        return_code = shard["process"].wait()
        log_json = load_json(shard["log_path"], default={}) or {}
        if return_code != 0 or not log_json.get("success", False):
            overall_success = False
        merged_objects.update(log_json.get("objects", {}))
        merged_object_dirs.update(move_object_dirs(output_root, shard["shard_dir"], shard["obj_ids"]))

    total_frames = 0
    for obj_id, object_dir in merged_object_dirs.items():
        metadata = load_json(Path(object_dir) / "metadata.json", default={}) or {}
        total_frames += len(metadata.get("frames", []))

    summary = {
        "multi_gpu": True,
        "success": overall_success,
        "source_evolution_dir": str(source_root),
        "gpus": parse_gpus(args.gpus),
        "total_objects": len(merged_object_dirs),
        "total_frames": total_frames,
        "camera_mode": args.camera_mode,
        "resolution": effective_resolution,
        "engine": effective_engine,
        "object_dirs": merged_object_dirs,
        "objects": merged_objects,
    }
    save_json(output_root / "export_summary.json", summary)
    return 0 if overall_success else 1


def main():
    args = parse_args()
    if args.worker_mode:
        raise SystemExit(run_worker(args))
    raise SystemExit(run_orchestrator(args))


if __name__ == "__main__":
    main()
