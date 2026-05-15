"""
Multi-GPU wrapper for run_scene_evolution_loop.py.

This keeps the original single-GPU scene evolution loop unchanged and
parallelizes work at the object level by launching one process per GPU shard.
Each shard writes to its own temporary output directory; the wrapper then
merges summaries and moves object result directories back to the shared root.
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
SINGLE_GPU_ENTRY = REPO_ROOT / "run_scene_evolution_loop.py"
DEFAULT_BLENDER = os.environ.get("BLENDER_BIN", "blender")
DEFAULT_SCENE_TEMPLATE = REPO_ROOT / "configs" / "scene_template.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Run scene evolution across multiple GPUs by object sharding")
    parser.add_argument("--meshes-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--obj-ids", nargs="+", default=None)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--blender", default=DEFAULT_BLENDER, dest="blender_bin")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--scene-template", default=str(DEFAULT_SCENE_TEMPLATE))
    parser.add_argument("--launch-stagger-seconds", type=float, default=3.0)
    parser.add_argument(
        "--keep-shard-object-dirs",
        action="store_true",
        help="Keep per-object result directories under shard outputs instead of moving them to the merged root",
    )
    return parser.parse_args()


def discover_obj_ids(meshes_dir: str) -> List[str]:
    return sorted(path.stem for path in Path(meshes_dir).glob("*.glb"))


def parse_devices(raw: str) -> List[str]:
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if not devices:
        raise ValueError("No devices specified")
    return devices


def shard_obj_ids(obj_ids: List[str], devices: List[str]) -> List[Dict[str, object]]:
    buckets = [{"device": device, "obj_ids": []} for device in devices]
    for index, obj_id in enumerate(obj_ids):
        buckets[index % len(devices)]["obj_ids"].append(obj_id)
    return [bucket for bucket in buckets if bucket["obj_ids"]]


def slugify_device(device: str) -> str:
    return device.replace(":", "_").replace("/", "_").replace("\\", "_").replace(",", "_")


def device_to_visible_gpu(device: str) -> str:
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    raise ValueError(f"Unsupported device format for GPU sharding: {device}")


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def launch_shards(args, shards: List[Dict[str, object]], output_root: Path):
    shard_root = output_root / "_shards"
    log_root = output_root / "_logs"
    shard_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    launched = []
    for shard_index, shard in enumerate(shards):
        device = str(shard["device"])
        obj_ids = list(shard["obj_ids"])
        device_slug = slugify_device(device)
        visible_gpu = device_to_visible_gpu(device)
        shard_dir = shard_root / f"shard_{device_slug}"
        shard_log = log_root / f"{device_slug}.log"
        shard_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python_bin,
            str(SINGLE_GPU_ENTRY),
            "--meshes-dir",
            args.meshes_dir,
            "--output-dir",
            str(shard_dir),
            "--device",
            "cuda:0",
            "--blender",
            args.blender_bin,
            "--scene-template",
            args.scene_template,
        ]
        if args.profile:
            cmd.extend(["--profile", args.profile])
        if obj_ids:
            cmd.append("--obj-ids")
            cmd.extend(obj_ids)

        log_file = shard_log.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible_gpu
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        launched.append(
            {
                "device": device,
                "device_slug": device_slug,
                "obj_ids": obj_ids,
                "cmd": cmd,
                "process": process,
                "log_file": log_file,
                "log_path": str(shard_log),
                "shard_dir": str(shard_dir),
                "started_at": time.time(),
            }
        )
        if shard_index < len(shards) - 1 and args.launch_stagger_seconds > 0:
            time.sleep(args.launch_stagger_seconds)
    return launched


def move_or_reference_object_dirs(output_root: Path, shard_dir: Path, obj_ids: List[str], keep_in_place: bool):
    object_dirs = {}
    for obj_id in obj_ids:
        src = shard_dir / obj_id
        if not src.exists():
            continue
        if keep_in_place:
            object_dirs[obj_id] = str(src)
            continue
        dst = output_root / obj_id
        if dst.exists():
            raise FileExistsError(f"Destination already exists while consolidating object dir: {dst}")
        shutil.move(str(src), str(dst))
        object_dirs[obj_id] = str(dst)
    return object_dirs


def merge_results(args, launched: List[Dict[str, object]], output_root: Path):
    merged_results = {}
    merged_scores = {}
    merged_object_dirs = {}
    accepted = 0
    shard_summaries = []

    for shard in launched:
        process = shard["process"]
        return_code = process.wait()
        elapsed = round(time.time() - float(shard["started_at"]), 3)
        shard["log_file"].close()

        shard_summary_path = Path(shard["shard_dir"]) / "scene_validation_summary.json"
        shard_summary = load_json(shard_summary_path, default={}) or {}
        shard_info = {
            "device": shard["device"],
            "obj_ids": shard["obj_ids"],
            "return_code": return_code,
            "elapsed_seconds": elapsed,
            "log_path": shard["log_path"],
            "summary_path": str(shard_summary_path),
        }
        shard_summaries.append(shard_info)

        if return_code != 0:
            for obj_id in shard["obj_ids"]:
                merged_results[obj_id] = {
                    "obj_id": obj_id,
                    "accepted": False,
                    "baseline_zone": "low",
                    "error": f"shard_failed_on_{shard['device']}",
                }
            continue

        shard_results = shard_summary.get("results", {})
        for obj_id, result in shard_results.items():
            merged_results[obj_id] = result
            merged_scores[obj_id] = result.get("final_hybrid", 0.0)
            if result.get("accepted"):
                accepted += 1

        merged_object_dirs.update(
            move_or_reference_object_dirs(
                output_root=output_root,
                shard_dir=Path(shard["shard_dir"]),
                obj_ids=shard["obj_ids"],
                keep_in_place=args.keep_shard_object_dirs,
            )
        )

    ordered_results = {obj_id: merged_results[obj_id] for obj_id in (args.obj_ids or merged_results.keys()) if obj_id in merged_results}
    ordered_scores = {obj_id: merged_scores.get(obj_id, 0.0) for obj_id in ordered_results}
    ordered_object_dirs = {obj_id: merged_object_dirs.get(obj_id) for obj_id in ordered_results if obj_id in merged_object_dirs}

    summary = {
        "multi_gpu": True,
        "devices": parse_devices(args.devices),
        "total": len(ordered_results),
        "accepted": accepted,
        "acceptance_rate": accepted / max(1, len(ordered_results)),
        "final_scores": ordered_scores,
        "results": ordered_results,
        "object_result_dirs": ordered_object_dirs,
        "shards": shard_summaries,
    }
    save_json(output_root / "scene_validation_summary.json", summary)
    save_json(output_root / "multi_gpu_runtime.json", summary)
    return summary


def main():
    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    obj_ids = args.obj_ids or discover_obj_ids(args.meshes_dir)
    if not obj_ids:
        raise SystemExit("No object ids found to process")
    args.obj_ids = obj_ids

    devices = parse_devices(args.devices)
    shards = shard_obj_ids(obj_ids, devices)

    request = {
        "meshes_dir": os.path.abspath(args.meshes_dir),
        "output_dir": str(output_root),
        "obj_ids": obj_ids,
        "devices": devices,
        "python_bin": args.python_bin,
        "blender_bin": args.blender_bin,
        "profile": args.profile,
        "scene_template": args.scene_template,
        "launch_stagger_seconds": args.launch_stagger_seconds,
        "keep_shard_object_dirs": args.keep_shard_object_dirs,
    }
    save_json(output_root / "multi_gpu_request.json", request)

    launched = launch_shards(args, shards, output_root)
    summary = merge_results(args, launched, output_root)
    print(
        f"\n=== Multi-GPU Scene Summary: {summary['accepted']}/{summary['total']} accepted, "
        f"rate={summary['acceptance_rate']:.1%} ==="
    )


if __name__ == "__main__":
    main()
