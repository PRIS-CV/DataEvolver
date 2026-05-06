"""
Bootstrap canonical yaw000 scene states for a batch of objects.

Each object gets exactly one pair root: obj_xxx_yaw000.
The loop stops when:
- reviewer says keep
- the score plateaus for two consecutive rounds
- max_rounds is reached
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
PIPELINE_ROOT = REPO_ROOT / "pipeline"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import run_scene_agent_monitor as monitor
from export_scene_multiview_from_pair_evolution import collect_round_records, select_best_round


DEFAULT_OBJECTS_FILE = REPO_ROOT / "configs" / "seed_concepts" / "scene_obj021_050_objects.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "pipeline" / "data" / "evolution_scene_v7_obj021_obj050_yaw000_bootstrap_20260408"
DEFAULT_PROFILE = REPO_ROOT / "configs" / "dataset_profiles" / "scene_v7_full50_loop.json"
DEFAULT_SCENE_TEMPLATE = REPO_ROOT / "configs" / "scene_template.json"
DEFAULT_PYTHON = monitor.DEFAULT_STEP_PYTHON
DEFAULT_BLENDER = monitor.DEFAULT_BLENDER


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap yaw000 scene states")
    parser.add_argument("--objects-file", default=str(DEFAULT_OBJECTS_FILE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--python", dest="python_bin", default=DEFAULT_PYTHON)
    parser.add_argument("--blender", dest="blender_bin", default=DEFAULT_BLENDER)
    parser.add_argument("--meshes-dir", required=True)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--scene-template", default=str(DEFAULT_SCENE_TEMPLATE))
    parser.add_argument("--render-prior-library", default=None)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--plateau-window", type=int, default=2)
    parser.add_argument("--plateau-eps", type=float, default=0.01)
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


def parse_gpus(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def slugify_gpu(gpu: str) -> str:
    return gpu.replace(":", "_").replace("/", "_").replace("\\", "_")


def load_object_ids(objects_file: Path) -> List[str]:
    payload = load_json(objects_file, default=None)
    if not isinstance(payload, list):
        raise FileNotFoundError(f"Invalid objects file: {objects_file}")
    obj_ids = [str(item.get("id")).strip() for item in payload if (item or {}).get("id")]
    if not obj_ids:
        raise ValueError(f"No objects found in {objects_file}")
    return obj_ids


def load_object_categories(objects_file: Path) -> Dict[str, str]:
    payload = load_json(objects_file, default=[]) or []
    if not isinstance(payload, list):
        return {}
    return {
        str(item.get("id")): str(item.get("category"))
        for item in payload
        if isinstance(item, dict) and item.get("id") and item.get("category")
    }


def deep_merge_dict(base: dict, override: Optional[dict]) -> dict:
    result = copy.deepcopy(base)
    if not isinstance(override, dict):
        return result
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def default_control_state() -> dict:
    action_space = load_json(REPO_ROOT / "configs" / "scene_action_space.json", default={}) or {}
    return copy.deepcopy(action_space.get("default_control_state") or {})


def select_render_prior(render_prior_library: Optional[dict], category: Optional[str]) -> tuple[Optional[dict], str]:
    if not render_prior_library:
        return None, "none"
    categories = render_prior_library.get("categories") or {}
    if category and category in categories:
        return categories[category].get("control_state"), f"category:{category}"
    return (render_prior_library.get("global") or {}).get("control_state"), "global"


def materialize_prior_control_state(
    *,
    pair_dir: Path,
    obj_id: str,
    category: Optional[str],
    render_prior_library: Optional[dict],
) -> Optional[Path]:
    prior_state, prior_source = select_render_prior(render_prior_library, category)
    if not prior_state:
        return None
    state = deep_merge_dict(default_control_state(), prior_state)
    state.setdefault("object", {})["yaw_deg"] = 0.0
    prior_path = pair_dir / "states" / "round00_render_prior_input.json"
    save_json(prior_path, state)
    save_json(
        pair_dir / "render_prior_selection.json",
        {
            "obj_id": obj_id,
            "category": category,
            "prior_source": prior_source,
            "render_prior_schema": render_prior_library.get("schema_version") if render_prior_library else None,
            "render_prior_record_count": render_prior_library.get("record_count") if render_prior_library else None,
        },
    )
    return prior_path


def shard_objects(obj_ids: List[str], gpus: List[str]) -> List[dict]:
    buckets = [{"gpu": gpu, "objects": []} for gpu in gpus]
    for idx, obj_id in enumerate(obj_ids):
        buckets[idx % len(gpus)]["objects"].append(obj_id)
    return [bucket for bucket in buckets if bucket["objects"]]


def collect_hybrid_scores(pair_dir: Path, obj_id: str) -> List[float]:
    scores: List[float] = []
    for record in collect_round_records(pair_dir, obj_id):
        value = record.get("hybrid_score")
        try:
            scores.append(float(value))
        except (TypeError, ValueError):
            continue
    return scores


def detect_plateau(scores: List[float], window: int, eps: float) -> bool:
    if len(scores) < window + 1:
        return False
    best_progress: List[float] = []
    best_so_far: Optional[float] = None
    for score in scores:
        best_so_far = score if best_so_far is None else max(best_so_far, score)
        best_progress.append(best_so_far)
    recent = best_progress[-(window + 1) :]
    deltas = [recent[idx] - recent[idx - 1] for idx in range(1, len(recent))]
    return all(delta < eps for delta in deltas)


def run_round0(
    *,
    pair_dir: Path,
    obj_id: str,
    python_bin: str,
    blender_bin: str,
    meshes_dir: str,
    profile: str,
    scene_template: str,
    control_state_in: Optional[Path] = None,
) -> int:
    cmd = [
        python_bin,
        str(REPO_ROOT / "scripts" / "run_scene_agent_step.py"),
        "--output-dir",
        str(pair_dir),
        "--obj-id",
        obj_id,
        "--rotation-deg",
        "0",
        "--round-idx",
        "0",
        "--agent-note",
        "yaw000_bootstrap_round00",
        "--device",
        "cuda:0",
        "--blender",
        blender_bin,
        "--meshes-dir",
        meshes_dir,
        "--profile",
        profile,
        "--scene-template",
        scene_template,
    ]
    if control_state_in is not None:
        cmd.extend(["--control-state-in", str(control_state_in)])
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def best_selection(pair_dir: Path, obj_id: str) -> Optional[dict]:
    chosen = select_best_round(collect_round_records(pair_dir, obj_id))
    if not chosen:
        return None
    return {
        "best_round_idx": int(chosen["round_idx"]),
        "best_hybrid_score": chosen.get("hybrid_score"),
        "best_control_state_path": chosen.get("control_state_path"),
        "selection_reason": chosen.get("selection_reason"),
        "asset_viability": chosen.get("asset_viability"),
        "issue_tags": chosen.get("issue_tags") or [],
    }


def bootstrap_object(
    *,
    obj_id: str,
    output_root: Path,
    python_bin: str,
    blender_bin: str,
    meshes_dir: str,
    profile: str,
    scene_template: str,
    render_prior_library: Optional[dict],
    object_category: Optional[str],
    max_rounds: int,
    plateau_window: int,
    plateau_eps: float,
) -> dict:
    pair_dir = output_root / f"{obj_id}_yaw000"
    pair_dir.mkdir(parents=True, exist_ok=True)
    decisions_dir = pair_dir / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)

    launched_round00 = False
    while True:
        latest_agg, latest_trace, round_idx = monitor.find_round_files(pair_dir)
        if latest_agg is None:
            control_state_in = materialize_prior_control_state(
                pair_dir=pair_dir,
                obj_id=obj_id,
                category=object_category,
                render_prior_library=render_prior_library,
            )
            rc = run_round0(
                pair_dir=pair_dir,
                obj_id=obj_id,
                python_bin=python_bin,
                blender_bin=blender_bin,
                meshes_dir=meshes_dir,
                profile=profile,
                scene_template=scene_template,
                control_state_in=control_state_in,
            )
            launched_round00 = True
            if rc != 0:
                return {
                    "obj_id": obj_id,
                    "pair_name": pair_dir.name,
                    "final_status": "launch_failed",
                    "latest_round_idx": -1,
                    "best_selection": best_selection(pair_dir, obj_id),
                    "launched_round00": launched_round00,
                }
            continue

        agg = monitor.load_json(latest_agg, default={}) or {}
        trace_payload = monitor.load_json(latest_trace, default={}) if latest_trace else {}
        trace_text = monitor.extract_trace_text(trace_payload or {}, agg)
        decision = monitor.build_decision_payload(pair_dir, latest_agg, latest_trace, round_idx, agg, trace_text)
        decision_path = decisions_dir / f"round{round_idx:02d}_decision.json"
        save_json(decision_path, decision)

        score_history = collect_hybrid_scores(pair_dir, obj_id)
        chosen_best = best_selection(pair_dir, obj_id)
        if decision["detected_verdict"] == "keep":
            return {
                "obj_id": obj_id,
                "pair_name": pair_dir.name,
                "final_status": "kept",
                "latest_round_idx": round_idx,
                "best_selection": chosen_best,
                "latest_decision_path": str(decision_path),
                "launched_round00": launched_round00,
            }
        if detect_plateau(score_history, plateau_window, plateau_eps):
            return {
                "obj_id": obj_id,
                "pair_name": pair_dir.name,
                "final_status": "plateau",
                "latest_round_idx": round_idx,
                "best_selection": chosen_best,
                "latest_decision_path": str(decision_path),
                "launched_round00": launched_round00,
            }
        if round_idx >= max_rounds:
            return {
                "obj_id": obj_id,
                "pair_name": pair_dir.name,
                "final_status": "manual_review_required",
                "latest_round_idx": round_idx,
                "best_selection": chosen_best,
                "latest_decision_path": str(decision_path),
                "launched_round00": launched_round00,
            }

        rc = monitor.run_one_round(
            pair_dir=pair_dir,
            obj_id=obj_id,
            rotation_deg=0,
            next_round_idx=round_idx + 1,
            actions=decision["chosen_actions"],
            python_bin=python_bin,
            blender_bin=blender_bin,
            meshes_dir=meshes_dir,
            profile=profile,
            scene_template=scene_template,
        )
        if rc != 0:
            return {
                "obj_id": obj_id,
                "pair_name": pair_dir.name,
                "final_status": "launch_failed",
                "latest_round_idx": round_idx,
                "best_selection": chosen_best,
                "latest_decision_path": str(decision_path),
                "launched_round00": launched_round00,
            }


def run_worker(args) -> int:
    manifest = load_json(Path(args.worker_manifest_path), default={}) or {}
    obj_ids = manifest.get("objects") or []
    output_root = Path(args.output_root).resolve()
    worker_log_path = Path(args.worker_log_path).resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.worker_gpu)
    render_prior_library = (
        load_json(Path(args.render_prior_library), default=None)
        if args.render_prior_library
        else None
    )
    object_categories = load_object_categories(Path(args.objects_file).resolve())

    summary = {
        "status": "running",
        "worker_gpu": args.worker_gpu,
        "objects_total": len(obj_ids),
        "objects_completed": 0,
        "results": {},
    }
    save_json(worker_log_path, summary)

    for obj_id in obj_ids:
        result = bootstrap_object(
            obj_id=obj_id,
            output_root=output_root,
            python_bin=args.python_bin,
            blender_bin=args.blender_bin,
            meshes_dir=args.meshes_dir,
            profile=args.profile,
            scene_template=args.scene_template,
            render_prior_library=render_prior_library,
            object_category=object_categories.get(obj_id),
            max_rounds=args.max_rounds,
            plateau_window=args.plateau_window,
            plateau_eps=args.plateau_eps,
        )
        summary["results"][obj_id] = result
        summary["objects_completed"] += 1
        save_json(worker_log_path, summary)

    summary["status"] = "completed"
    summary["status_counts"] = dict(Counter(result["final_status"] for result in summary["results"].values()))
    save_json(worker_log_path, summary)
    return 0 if summary["status_counts"].get("launch_failed", 0) == 0 else 1


def run_orchestrator(args) -> int:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root = output_root / "_logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    obj_ids = load_object_ids(Path(args.objects_file).resolve())
    gpus = parse_gpus(args.gpus)
    shards = shard_objects(obj_ids, gpus)
    request = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "objects_file": str(Path(args.objects_file).resolve()),
        "output_root": str(output_root),
        "meshes_dir": str(Path(args.meshes_dir).resolve()),
        "render_prior_library": str(Path(args.render_prior_library).resolve()) if args.render_prior_library else None,
        "gpus": gpus,
        "objects": obj_ids,
        "max_rounds": args.max_rounds,
        "plateau_window": args.plateau_window,
        "plateau_eps": args.plateau_eps,
    }
    save_json(output_root / "bootstrap_request.json", request)

    launched = []
    for index, shard in enumerate(shards):
        gpu = str(shard["gpu"])
        gpu_slug = slugify_gpu(gpu)
        manifest_path = logs_root / f"worker_gpu_{gpu_slug}_manifest.json"
        log_path = logs_root / f"worker_gpu_{gpu_slug}.json"
        save_json(manifest_path, {"objects": shard["objects"]})
        cmd = [
            args.python_bin,
            str(SCRIPT_PATH),
            "--worker-mode",
            "--objects-file",
            args.objects_file,
            "--output-root",
            str(output_root),
            "--worker-gpu",
            gpu,
            "--worker-manifest-path",
            str(manifest_path),
            "--worker-log-path",
            str(log_path),
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
            *(
                ["--render-prior-library", args.render_prior_library]
                if args.render_prior_library
                else []
            ),
            "--max-rounds",
            str(args.max_rounds),
            "--plateau-window",
            str(args.plateau_window),
            "--plateau-eps",
            str(args.plateau_eps),
        ]
        process = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
        launched.append({"gpu": gpu, "process": process, "log_path": log_path})
        if index < len(shards) - 1 and args.launch_stagger_seconds > 0:
            time.sleep(float(args.launch_stagger_seconds))

    results: Dict[str, dict] = {}
    for item in launched:
        item["process"].wait()
        payload = load_json(item["log_path"], default={}) or {}
        results.update(payload.get("results") or {})

    status_counts = dict(Counter(result["final_status"] for result in results.values()))
    summary = {
        "success": status_counts.get("launch_failed", 0) == 0,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_root": str(output_root),
        "meshes_dir": str(Path(args.meshes_dir).resolve()),
        "total_objects": len(results),
        "status_counts": status_counts,
        "max_rounds": args.max_rounds,
        "plateau_window": args.plateau_window,
        "plateau_eps": args.plateau_eps,
    }
    save_json(output_root / "summary.json", summary)
    save_json(output_root / "manifest.json", {"summary": summary, "objects": {k: results[k] for k in sorted(results)}})
    return 0 if summary["success"] else 1


def main():
    args = parse_args()
    if args.worker_mode:
        raise SystemExit(run_worker(args))
    raise SystemExit(run_orchestrator(args))


if __name__ == "__main__":
    main()
