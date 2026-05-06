"""
Run scene-aware freeform evolution for a fixed set of object rotations.

This keeps the same scene-aware rendering pipeline, the same Qwen3.5 reviewer,
freeform feedback, and thinking mode, while locking object yaw to one of the
requested rotations (for example 0/90/180/270).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
LOOP_ENTRY = REPO_ROOT / "run_scene_evolution_loop.py"
DEFAULT_BLENDER = os.environ.get("BLENDER_BIN", "/home/wuwenzhuo/blender-4.24/blender")
DEFAULT_PROFILE = REPO_ROOT / "configs" / "dataset_profiles" / "scene_v7.json"
DEFAULT_SCENE_TEMPLATE = REPO_ROOT / "configs" / "scene_template.json"
DEFAULT_MESHES_DIR = REPO_ROOT / "pipeline" / "data" / "meshes"


def parse_args():
    parser = argparse.ArgumentParser(description="Run scene evolution for fixed object rotations across multiple GPUs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--obj-ids", nargs="+", required=True)
    parser.add_argument("--rotations", nargs="+", type=int, default=[0, 90, 180, 270])
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--blender", dest="blender_bin", default=DEFAULT_BLENDER)
    parser.add_argument("--meshes-dir", default=str(DEFAULT_MESHES_DIR))
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--scene-template", default=str(DEFAULT_SCENE_TEMPLATE))
    parser.add_argument("--launch-stagger-seconds", type=float, default=3.0)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-gpu", default=None)
    parser.add_argument("--worker-shard-dir", default=None)
    parser.add_argument("--worker-log-path", default=None)
    parser.add_argument("--pair-manifest", default=None)
    return parser.parse_args()


def parse_gpus(raw: str) -> List[str]:
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise ValueError("No GPUs specified")
    return gpus


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_pairs(obj_ids: List[str], rotations: List[int]) -> List[dict]:
    pairs = []
    for obj_id in obj_ids:
        for rotation in rotations:
            pairs.append({
                "obj_id": obj_id,
                "rotation_deg": int(rotation),
                "pair_id": f"{obj_id}_yaw{int(rotation):03d}",
            })
    return pairs


def shard_pairs(pairs: List[dict], gpus: List[str]) -> List[Dict[str, object]]:
    buckets = [{"gpu": gpu, "pairs": []} for gpu in gpus]
    for index, pair in enumerate(pairs):
        buckets[index % len(gpus)]["pairs"].append(pair)
    return [bucket for bucket in buckets if bucket["pairs"]]


def slugify_gpu(gpu: str) -> str:
    return gpu.replace(":", "_").replace("/", "_").replace("\\", "_")


def build_pair_profile(base_profile_path: Path, rotation_deg: int, output_path: Path):
    profile = load_json(base_profile_path, default={}) or {}
    profile["use_freeform_vlm_feedback"] = True
    profile["enable_vlm_thinking"] = True
    profile["initial_control_state_overrides"] = {
        "object": {
            "yaw_deg": float(rotation_deg),
        }
    }
    profile["locked_control_state_overrides"] = {
        "object": {
            "yaw_deg": float(rotation_deg),
        }
    }
    save_json(output_path, profile)


def run_worker(args) -> int:
    assert args.worker_gpu is not None
    assert args.worker_shard_dir is not None
    assert args.pair_manifest is not None

    shard_dir = Path(args.worker_shard_dir).resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = shard_dir / "_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_json(Path(args.pair_manifest), default=[]) or []

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.worker_gpu

    worker_summary = {
        "gpu": args.worker_gpu,
        "pairs": pairs,
        "started_at": time.time(),
        "results": {},
        "success": True,
    }

    for pair in pairs:
        obj_id = pair["obj_id"]
        rotation_deg = int(pair["rotation_deg"])
        pair_id = pair["pair_id"]
        pair_dir = shard_dir / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        profile_path = profiles_dir / f"{pair_id}.json"
        build_pair_profile(Path(args.profile), rotation_deg, profile_path)

        cmd = [
            args.python_bin,
            str(LOOP_ENTRY),
            "--meshes-dir",
            args.meshes_dir,
            "--output-dir",
            str(pair_dir),
            "--obj-ids",
            obj_id,
            "--device",
            "cuda:0",
            "--blender",
            args.blender_bin,
            "--profile",
            str(profile_path),
            "--scene-template",
            args.scene_template,
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
        result_json = pair_dir / "scene_validation_summary.json"
        pair_summary = load_json(result_json, default={}) or {}
        worker_summary["results"][pair_id] = {
            "obj_id": obj_id,
            "rotation_deg": rotation_deg,
            "return_code": completed.returncode,
            "elapsed_seconds": round(time.time() - started, 3),
            "profile_path": str(profile_path),
            "pair_dir": str(pair_dir),
            "result_json": str(result_json) if result_json.exists() else None,
            "summary": pair_summary,
            "stdout_tail": completed.stdout[-4000:],
        }
        if completed.returncode != 0:
            worker_summary["success"] = False
            worker_summary["failed_pair_id"] = pair_id
            break

    worker_summary["elapsed_seconds"] = round(time.time() - worker_summary["started_at"], 3)
    save_json(Path(args.worker_log_path), worker_summary)
    return 0 if worker_summary["success"] else 1


def run_orchestrator(args) -> int:
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    shard_root = output_root / "_shards"
    log_root = output_root / "_logs"
    meta_root = output_root / "_meta"
    shard_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(args.obj_ids, args.rotations)
    gpus = parse_gpus(args.gpus)
    request = {
        "meshes_dir": os.path.abspath(args.meshes_dir),
        "output_dir": str(output_root),
        "obj_ids": args.obj_ids,
        "rotations": args.rotations,
        "gpus": gpus,
        "profile": os.path.abspath(args.profile),
        "scene_template": os.path.abspath(args.scene_template),
        "requested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(output_root / "rotation_dataset_request.json", request)

    launched = []
    shards = shard_pairs(pairs, gpus)
    for index, shard in enumerate(shards):
        gpu = str(shard["gpu"])
        gpu_slug = slugify_gpu(gpu)
        shard_dir = shard_root / f"shard_gpu_{gpu_slug}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"worker_gpu_{gpu_slug}.json"
        pair_manifest = meta_root / f"pairs_gpu_{gpu_slug}.json"
        save_json(pair_manifest, shard["pairs"])

        cmd = [
            args.python_bin,
            str(SCRIPT_PATH),
            "--worker-mode",
            "--output-dir",
            str(output_root),
            "--obj-ids",
            *args.obj_ids,
            "--rotations",
            *[str(v) for v in args.rotations],
            "--gpus",
            gpu,
            "--python",
            args.python_bin,
            "--blender",
            args.blender_bin,
            "--meshes-dir",
            args.meshes_dir,
            "--profile",
            args.profile,
            "--scene-template",
            args.scene_template,
            "--worker-gpu",
            gpu,
            "--worker-shard-dir",
            str(shard_dir),
            "--worker-log-path",
            str(log_path),
            "--pair-manifest",
            str(pair_manifest),
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
            "process": process,
            "log_path": log_path,
        })
        if index < len(shards) - 1 and args.launch_stagger_seconds > 0:
            time.sleep(args.launch_stagger_seconds)

    overall_success = True
    merged_results = {}
    for shard in launched:
        return_code = shard["process"].wait()
        log_json = load_json(Path(shard["log_path"]), default={}) or {}
        if return_code != 0 or not log_json.get("success", False):
            overall_success = False
        merged_results.update(log_json.get("results", {}))

    summary = {
        "multi_gpu": True,
        "success": overall_success,
        "total_pairs": len(merged_results),
        "pairs": merged_results,
    }
    save_json(output_root / "rotation_dataset_summary.json", summary)
    return 0 if overall_success else 1


def main():
    args = parse_args()
    if args.worker_mode:
        raise SystemExit(run_worker(args))
    raise SystemExit(run_orchestrator(args))


if __name__ == "__main__":
    main()
