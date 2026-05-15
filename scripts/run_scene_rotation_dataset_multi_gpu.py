"""
Run agent-guided scene render/review steps for a fixed set of object rotations.

Unlike the previous automatic controller loop, this wrapper only performs
explicitly requested rounds. It renders, collects Qwen3.5 freeform feedback
with thinking enabled, and stores the traces. The next action is expected to be
chosen by the outer agent/human, not by the script itself.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_BLENDER = os.environ.get("BLENDER_BIN", "blender")
DEFAULT_PROFILE = REPO_ROOT / "configs" / "dataset_profiles" / "scene_v7.json"
DEFAULT_SCENE_TEMPLATE = REPO_ROOT / "configs" / "scene_template.json"
DEFAULT_MESHES_DIR = REPO_ROOT / "pipeline" / "data" / "meshes"


def parse_args():
    parser = argparse.ArgumentParser(description="Run scene evolution for fixed object rotations across multiple GPUs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--obj-ids", nargs="+", required=True)
    parser.add_argument("--rotations", nargs="+", type=int, default=[0, 90, 180, 270])
    parser.add_argument(
        "--gpus",
        default="0,1,2",
        help="Worker GPU groups. Use ',' inside one review group and ';' between workers, e.g. '0,1;2,3'.",
    )
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--blender", dest="blender_bin", default=DEFAULT_BLENDER)
    parser.add_argument("--meshes-dir", default=str(DEFAULT_MESHES_DIR))
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--scene-template", default=str(DEFAULT_SCENE_TEMPLATE))
    parser.add_argument("--launch-stagger-seconds", type=float, default=3.0)
    parser.add_argument("--round-idx", type=int, default=0)
    parser.add_argument("--actions-manifest", default=None)
    parser.add_argument("--pairs-file", default=None)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-gpu", default=None)
    parser.add_argument("--worker-shard-dir", default=None)
    parser.add_argument("--worker-log-path", default=None)
    parser.add_argument("--pair-manifest", default=None)
    parser.add_argument("--allow-unsafe-single-gpu-review", action="store_true")
    return parser.parse_args()


def parse_gpus(raw: str) -> List[str]:
    raw = str(raw or "").strip()
    if not raw:
        raise ValueError("No GPUs specified")
    gpus = [item.strip() for item in raw.split(";") if item.strip()] if ";" in raw else [raw]
    if not gpus:
        raise ValueError("No GPUs specified")
    return gpus


def parse_visible_gpu_ids(raw: str) -> List[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


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


def run_worker(args) -> int:
    assert args.worker_gpu is not None
    assert args.worker_shard_dir is not None
    assert args.pair_manifest is not None

    shard_dir = Path(args.worker_shard_dir).resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_json(Path(args.pair_manifest), default=[]) or []
    actions_manifest = load_json(Path(args.actions_manifest), default={}) if args.actions_manifest else {}

    visible_gpu_ids = parse_visible_gpu_ids(args.worker_gpu)
    if not visible_gpu_ids:
        raise ValueError("worker_gpu must specify at least one visible GPU id")
    if len(visible_gpu_ids) < 2 and not args.allow_unsafe_single_gpu_review:
        raise ValueError(
            "single-GPU VLM review is disabled; pass grouped GPUs like --gpus 0,1;2,3 "
            "or explicitly opt into --allow-unsafe-single-gpu-review"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpu_ids)
    if len(visible_gpu_ids) > 1:
        os.environ["VLM_ALLOW_BALANCED"] = "1"
        os.environ.pop("VLM_FORCE_SINGLE_GPU", None)
    else:
        os.environ.setdefault("VLM_FORCE_SINGLE_GPU", "1")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import run_scene_agent_step as agent_step

    worker_summary = {
        "gpu": args.worker_gpu,
        "review_visible_gpus": visible_gpu_ids,
        "review_mode": "balanced" if len(visible_gpu_ids) > 1 else "single_gpu",
        "pairs": pairs,
        "round_idx": args.round_idx,
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
        started = time.time()
        try:
            pair_actions = actions_manifest.get(pair_id, [])
            summary = agent_step.run_agent_step(
                output_dir=str(pair_dir),
                obj_id=obj_id,
                rotation_deg=rotation_deg,
                round_idx=args.round_idx,
                actions=pair_actions,
                step_scale=1.0,
                agent_note=f"multi_gpu_batch round={args.round_idx}",
                control_state_in=None,
                prev_renders_dir=None,
                device="cuda:0",
                blender_bin=args.blender_bin,
                meshes_dir=args.meshes_dir,
                profile_path=args.profile,
                scene_template_path=args.scene_template,
            )
            worker_summary["results"][pair_id] = {
                "obj_id": obj_id,
                "rotation_deg": rotation_deg,
                "return_code": 0,
                "elapsed_seconds": round(time.time() - started, 3),
                "pair_dir": str(pair_dir),
                "result_json": summary.get("review_agg_path"),
                "summary": summary,
                "actions": pair_actions,
            }
        except Exception:
            worker_summary["results"][pair_id] = {
                "obj_id": obj_id,
                "rotation_deg": rotation_deg,
                "return_code": 1,
                "elapsed_seconds": round(time.time() - started, 3),
                "pair_dir": str(pair_dir),
                "result_json": None,
                "summary": {},
                "stdout_tail": traceback.format_exc()[-4000:],
            }
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

    if args.pairs_file:
        pairs = load_json(Path(args.pairs_file), default=[]) or []
    else:
        pairs = build_pairs(args.obj_ids, args.rotations)
    gpus = parse_gpus(args.gpus)
    gpu_groups = [parse_visible_gpu_ids(gpu) for gpu in gpus]
    if any(not group for group in gpu_groups):
        raise ValueError(f"Invalid --gpus specification: {args.gpus}")
    if not args.allow_unsafe_single_gpu_review and any(len(group) < 2 for group in gpu_groups):
        raise ValueError(
            "Single-GPU reviewer workers are disabled. Use grouped GPUs such as --gpus 0,1;2,3 "
            "or explicitly opt into --allow-unsafe-single-gpu-review."
        )
    seen_gpu_ids = set()
    for group in gpu_groups:
        overlap = seen_gpu_ids.intersection(group)
        if overlap:
            overlap_text = ",".join(sorted(overlap))
            raise ValueError(f"Overlapping reviewer GPU groups are not allowed: {overlap_text}")
        seen_gpu_ids.update(group)

    request = {
        "meshes_dir": os.path.abspath(args.meshes_dir),
        "output_dir": str(output_root),
        "obj_ids": args.obj_ids,
        "rotations": args.rotations,
        "pairs_file": os.path.abspath(args.pairs_file) if args.pairs_file else None,
        "gpus": gpus,
        "gpu_groups": gpu_groups,
        "round_idx": args.round_idx,
        "profile": os.path.abspath(args.profile),
        "scene_template": os.path.abspath(args.scene_template),
        "actions_manifest": os.path.abspath(args.actions_manifest) if args.actions_manifest else None,
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
            "--round-idx",
            str(args.round_idx),
            "--worker-gpu",
            gpu,
            "--worker-shard-dir",
            str(shard_dir),
            "--worker-log-path",
            str(log_path),
            "--pair-manifest",
            str(pair_manifest),
        ]
        if args.actions_manifest:
            cmd.extend(["--actions-manifest", args.actions_manifest])
        if args.allow_unsafe_single_gpu_review:
            cmd.append("--allow-unsafe-single-gpu-review")
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=None,
            stderr=None,
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
