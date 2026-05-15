"""
Multi-GPU wrapper for pipeline/stage4_blender_render.py.

This script shards object ids across multiple GPUs and launches one worker
process per GPU. Each worker renders its assigned objects sequentially using
Blender, writing results into a shard directory. The wrapper consolidates the
per-object render folders back into a shared output root and writes a merged
summary manifest.
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
BLENDER_SCRIPT = REPO_ROOT / "pipeline" / "stage4_blender_render.py"
DEFAULT_BLENDER = os.environ.get("BLENDER_BIN", "blender")


def parse_args():
    parser = argparse.ArgumentParser(description="Run stage4 multiview rendering across multiple GPUs")
    parser.add_argument("--input-dir", required=True, help="Directory containing .glb meshes")
    parser.add_argument("--output-dir", required=True, help="Merged output root")
    parser.add_argument("--obj-ids", nargs="+", required=True, help="Object ids to render")
    parser.add_argument("--gpus", default="0,1,2", help="Comma-separated GPU ids, e.g. 0,1,2")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--blender", dest="blender_bin", default=DEFAULT_BLENDER)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES"])
    parser.add_argument("--camera-distance", type=float, default=3.5)
    parser.add_argument("--launch-stagger-seconds", type=float, default=2.0)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-gpu", default=None)
    parser.add_argument("--worker-shard-dir", default=None)
    parser.add_argument("--worker-log-path", default=None)
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


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_worker(args) -> int:
    assert args.worker_gpu is not None
    assert args.worker_shard_dir is not None
    shard_dir = Path(args.worker_shard_dir).resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.worker_gpu

    worker_summary = {
        "gpu": args.worker_gpu,
        "obj_ids": args.obj_ids,
        "resolution": args.resolution,
        "engine": args.engine,
        "camera_distance": args.camera_distance,
        "started_at": time.time(),
        "objects": {},
        "success": True,
    }

    for obj_id in args.obj_ids:
        cmd = [
            args.blender_bin,
            "-b",
            "-P",
            str(BLENDER_SCRIPT),
            "--",
            "--input-dir",
            args.input_dir,
            "--output-dir",
            str(shard_dir),
            "--resolution",
            str(args.resolution),
            "--engine",
            args.engine,
            "--camera-distance",
            str(args.camera_distance),
            "--obj-id",
            obj_id,
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
            "object_dir": str(object_dir),
            "metadata_path": str(metadata_path) if metadata_path.exists() else None,
            "total_renders": metadata.get("total_renders"),
            "stdout_tail": completed.stdout[-4000:],
        }
        if completed.returncode != 0:
            worker_summary["success"] = False
            worker_summary["failed_obj_id"] = obj_id
            break

    worker_summary["elapsed_seconds"] = round(time.time() - worker_summary["started_at"], 3)
    if args.worker_log_path:
        save_json(Path(args.worker_log_path), worker_summary)
    else:
        print(json.dumps(worker_summary, indent=2, ensure_ascii=False))
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
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    shard_root = output_root / "_shards"
    log_root = output_root / "_logs"
    shard_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    gpus = parse_gpus(args.gpus)
    shards = shard_obj_ids(args.obj_ids, gpus)
    request = {
        "input_dir": os.path.abspath(args.input_dir),
        "output_dir": str(output_root),
        "obj_ids": args.obj_ids,
        "gpus": gpus,
        "resolution": args.resolution,
        "engine": args.engine,
        "camera_distance": args.camera_distance,
        "requested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(output_root / "render_request.json", request)

    launched = []
    for index, shard in enumerate(shards):
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
            "--input-dir",
            args.input_dir,
            "--output-dir",
            str(output_root),
            "--obj-ids",
            *obj_ids,
            "--resolution",
            str(args.resolution),
            "--engine",
            args.engine,
            "--camera-distance",
            str(args.camera_distance),
            "--blender",
            args.blender_bin,
            "--worker-gpu",
            gpu,
            "--worker-shard-dir",
            str(shard_dir),
            "--worker-log-path",
            str(log_path),
        ]

        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        launched.append(
            {
                "gpu": gpu,
                "gpu_slug": gpu_slug,
                "obj_ids": obj_ids,
                "process": process,
                "shard_dir": shard_dir,
                "log_path": log_path,
                "started_at": time.time(),
            }
        )
        if index < len(shards) - 1 and args.launch_stagger_seconds > 0:
            time.sleep(args.launch_stagger_seconds)

    merged_object_dirs = {}
    merged_objects = {}
    overall_success = True

    for shard in launched:
        return_code = shard["process"].wait()
        log_json = load_json(shard["log_path"], default={}) or {}
        if return_code != 0 or not log_json.get("success", False):
            overall_success = False
        merged_objects.update(log_json.get("objects", {}))
        merged_object_dirs.update(move_object_dirs(output_root, shard["shard_dir"], shard["obj_ids"]))

    total_renders = 0
    for obj_id, object_dir in merged_object_dirs.items():
        metadata = load_json(Path(object_dir) / "metadata.json", default={}) or {}
        total_renders += int(metadata.get("total_renders", 0) or 0)

    summary = {
        "multi_gpu": True,
        "success": overall_success,
        "gpus": gpus,
        "total_objects": len(merged_object_dirs),
        "total_renders": total_renders,
        "resolution": args.resolution,
        "engine": args.engine,
        "camera_distance": args.camera_distance,
        "object_dirs": merged_object_dirs,
        "objects": merged_objects,
    }
    save_json(output_root / "render_summary_multigpu.json", summary)
    return 0 if overall_success else 1


def main():
    args = parse_args()
    if args.worker_mode:
        raise SystemExit(run_worker(args))
    raise SystemExit(run_orchestrator(args))


if __name__ == "__main__":
    main()
