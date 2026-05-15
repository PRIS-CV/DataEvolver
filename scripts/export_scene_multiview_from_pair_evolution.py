"""
Export scene-aware multi-view renders from pair-level evolution results.

This is the pair-aware companion to export_scene_multiview_from_evolution.py.
It reads pair folders such as obj_001_yaw090, selects the best available round
for each pair, and re-renders a configurable final multiview set for dataset
export.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
DEFAULT_EXPORT_AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]
DEFAULT_EXPORT_ELEVATIONS = [0]


def parse_args():
    parser = argparse.ArgumentParser(description="Export multiview renders from pair-level scene evolution roots")
    parser.add_argument("--source-root", required=True, help="Evolution root containing obj_xxx_yawYYY pair dirs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pair-names", nargs="*", default=None, help="Optional subset of pair dir names")
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--blender", dest="blender_bin", default=DEFAULT_BLENDER)
    parser.add_argument("--meshes-dir", default=str(DEFAULT_MESHES_DIR))
    parser.add_argument("--scene-template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument(
        "--camera-mode",
        default="object_bbox_relative",
        choices=["object_bbox_relative", "scene_bounds_relative", "scene_camera_match"],
    )
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--engine", choices=["EEVEE", "CYCLES"], default=None)
    parser.add_argument("--launch-stagger-seconds", type=float, default=2.0)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-gpu", default=None)
    parser.add_argument("--worker-manifest-path", default=None)
    parser.add_argument("--worker-shard-dir", default=None)
    parser.add_argument("--worker-log-path", default=None)
    parser.add_argument("--export-template-path", default=None)
    parser.add_argument("--azimuths", default="0,45,90,135,180,225,270,315")
    parser.add_argument("--elevations", default="0")
    return parser.parse_args()


def parse_gpus(raw: str) -> List[str]:
    if not raw:
        raise ValueError("No GPUs specified")
    if ";" in raw:
        gpus = [item.strip() for item in raw.split(";") if item.strip()]
    else:
        gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise ValueError("No GPUs specified")
    return gpus


def slugify_gpu(gpu: str) -> str:
    return gpu.replace(":", "_").replace("/", "_").replace("\\", "_").replace(",", "_")


def parse_int_csv(raw: str) -> List[int]:
    values: List[int] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_pair_name(name: str) -> Optional[dict]:
    match = re.fullmatch(r"(obj_\d+)_yaw(\d+)", name)
    if not match:
        return None
    return {
        "pair_name": name,
        "obj_id": match.group(1),
        "rotation_deg": int(match.group(2)),
    }


def _pair_priority(pair_dir: Path) -> tuple:
    best_round = -1
    for agent_path in pair_dir.glob("agent_round*.json"):
        round_idx = _round_from_name(agent_path, "agent_round", ".json")
        if round_idx is not None and round_idx > best_round:
            best_round = round_idx
    latest_mtime = pair_dir.stat().st_mtime if pair_dir.exists() else 0.0
    return (best_round, latest_mtime)


def discover_pair_dirs(source_root: Path, requested_pair_names: Optional[List[str]]) -> List[dict]:
    requested = set(requested_pair_names or [])
    pair_map: Dict[str, dict] = {}
    for pair_dir in sorted(source_root.rglob("obj_*_yaw*")):
        if not pair_dir.is_dir():
            continue
        parsed = parse_pair_name(pair_dir.name)
        if not parsed:
            continue
        if requested and parsed["pair_name"] not in requested:
            continue
        record = dict(parsed)
        record["pair_dir"] = str(pair_dir.resolve())
        existing = pair_map.get(parsed["pair_name"])
        if existing is None or _pair_priority(pair_dir) > _pair_priority(Path(existing["pair_dir"])):
            pair_map[parsed["pair_name"]] = record
    pairs = sorted(pair_map.values(), key=lambda item: item["pair_name"])
    if requested:
        found = {item["pair_name"] for item in pairs}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"Missing pair dirs: {missing}")
    return pairs


def shard_pairs(pair_records: List[dict], gpus: List[str]) -> List[Dict[str, object]]:
    buckets = [{"gpu": gpu, "pairs": []} for gpu in gpus]
    for index, pair in enumerate(pair_records):
        buckets[index % len(gpus)]["pairs"].append(pair)
    return [bucket for bucket in buckets if bucket["pairs"]]


def build_export_template(args, output_root: Path) -> Path:
    base_template = load_json(Path(args.scene_template), default={}) or {}
    if args.engine:
        base_template["render_engine"] = args.engine
    if args.resolution:
        base_template["render_resolution"] = args.resolution
    azimuths = parse_int_csv(args.azimuths) or list(DEFAULT_EXPORT_AZIMUTHS)
    elevations = parse_int_csv(args.elevations) or list(DEFAULT_EXPORT_ELEVATIONS)
    base_template["qc_views"] = [[az, el] for az in azimuths for el in elevations]
    base_template["camera_mode"] = args.camera_mode
    base_template["export_azimuths"] = azimuths
    base_template["export_elevations"] = elevations

    export_template_path = output_root / "scene_template_multiview.json"
    save_json(export_template_path, base_template)
    return export_template_path


def _round_from_name(path: Path, prefix: str, suffix: str) -> Optional[int]:
    match = re.fullmatch(rf"{re.escape(prefix)}(\d+){re.escape(suffix)}", path.name)
    return int(match.group(1)) if match else None


def collect_round_records(pair_dir: Path, obj_id: str) -> List[dict]:
    records: List[dict] = []
    for agent_path in sorted(pair_dir.glob("agent_round*.json")):
        round_idx = _round_from_name(agent_path, "agent_round", ".json")
        if round_idx is None:
            continue
        agent = load_json(agent_path, default={}) or {}
        agg_path = Path(agent.get("review_agg_path")) if agent.get("review_agg_path") else (pair_dir / "reviews" / f"{obj_id}_r{round_idx:02d}_agg.json")
        agg = load_json(agg_path, default={}) or {}
        decision_path = pair_dir / "decisions" / f"round{round_idx:02d}_decision.json"
        decision = load_json(decision_path, default={}) or {}
        control_path = pair_dir / f"round{round_idx:02d}_renders" / f"_control_{obj_id}.json"
        score = agg.get("hybrid_score", agent.get("hybrid_score"))
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = None
        record = {
            "round_idx": round_idx,
            "agent_path": str(agent_path),
            "agg_path": str(agg_path) if agg_path.exists() else None,
            "decision_path": str(decision_path) if decision_path.exists() else None,
            "control_state_path": str(control_path) if control_path.exists() else None,
            "hybrid_score": score,
            "hybrid_route": agg.get("hybrid_route"),
            "asset_viability": agg.get("asset_viability"),
            "structure_consistency": agg.get("structure_consistency"),
            "issue_tags": agg.get("issue_tags") or [],
            "suggested_actions": agg.get("suggested_actions") or [],
            "detected_verdict": decision.get("detected_verdict"),
        }
        if record["control_state_path"]:
            records.append(record)
    return records


def select_best_round(records: List[dict]) -> Optional[dict]:
    if not records:
        return None

    keep_records = [r for r in records if r.get("detected_verdict") == "keep"]
    ranked = keep_records if keep_records else records

    def key_fn(item: dict):
        score = item.get("hybrid_score")
        return (score if score is not None else -1.0, -(item.get("round_idx") or 0))

    chosen = max(ranked, key=key_fn)
    chosen = dict(chosen)
    chosen["selection_reason"] = "best_keep_score" if keep_records else "best_hybrid_score"
    return chosen


def render_one_pair(
    *,
    pair_record: dict,
    best_record: dict,
    shard_dir: Path,
    env: dict,
    blender_bin: str,
    meshes_dir: str,
    export_template_path: str,
    resolution: int,
    engine: str,
) -> dict:
    pair_name = str(pair_record["pair_name"])
    obj_id = str(pair_record["obj_id"])
    rotation_deg = int(pair_record["rotation_deg"])
    temp_root = shard_dir / f"_tmp_{pair_name}"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        blender_bin,
        "-b",
        "-P",
        str(SCENE_RENDER_SCRIPT),
        "--",
        "--input-dir",
        meshes_dir,
        "--output-dir",
        str(temp_root),
        "--obj-id",
        obj_id,
        "--resolution",
        str(resolution),
        "--engine",
        engine,
        "--control-state",
        str(best_record["control_state_path"]),
        "--scene-template",
        export_template_path,
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

    rendered_dir = temp_root / obj_id
    final_dir = shard_dir / pair_name
    if final_dir.exists():
        raise FileExistsError(f"Destination already exists: {final_dir}")

    result = {
        "pair_name": pair_name,
        "obj_id": obj_id,
        "rotation_deg": rotation_deg,
        "return_code": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "best_round_idx": best_record["round_idx"],
        "hybrid_score": best_record.get("hybrid_score"),
        "detected_verdict": best_record.get("detected_verdict"),
        "asset_viability": best_record.get("asset_viability"),
        "structure_consistency": best_record.get("structure_consistency"),
        "issue_tags": best_record.get("issue_tags") or [],
        "suggested_actions": best_record.get("suggested_actions") or [],
        "control_state_path": best_record["control_state_path"],
        "selection_reason": best_record["selection_reason"],
        "stdout_tail": completed.stdout[-4000:],
    }

    if completed.returncode != 0:
        shutil.rmtree(temp_root, ignore_errors=True)
        return result

    if not rendered_dir.exists():
        result["return_code"] = -1
        result["error"] = "rendered_dir_missing"
        shutil.rmtree(temp_root, ignore_errors=True)
        return result

    shutil.move(str(rendered_dir), str(final_dir))
    metadata = load_json(final_dir / "metadata.json", default={}) or {}
    result["final_dir"] = str(final_dir)
    result["metadata_path"] = str(final_dir / "metadata.json") if (final_dir / "metadata.json").exists() else None
    result["total_frames"] = len(metadata.get("frames", []))
    result["camera_mode"] = metadata.get("camera_mode")
    save_json(
        final_dir / "pair_export_meta.json",
        {
            "pair_name": pair_name,
            "obj_id": obj_id,
            "rotation_deg": rotation_deg,
            "source_pair_dir": pair_record["pair_dir"],
            "best_round": best_record,
            "export_result": result,
        },
    )
    shutil.rmtree(temp_root, ignore_errors=True)
    return result


def run_worker(args) -> int:
    assert args.worker_gpu is not None
    assert args.worker_manifest_path is not None
    assert args.worker_shard_dir is not None
    assert args.worker_log_path is not None
    assert args.export_template_path is not None

    manifest_path = Path(args.worker_manifest_path).resolve()
    shard_dir = Path(args.worker_shard_dir).resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(manifest_path, default={}) or {}
    pairs = manifest.get("pairs") or []

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.worker_gpu)

    worker_summary = {
        "gpu": args.worker_gpu,
        "pairs_total": len(pairs),
        "started_at": time.time(),
        "success": True,
        "pairs": {},
    }

    for pair_record in pairs:
        pair_dir = Path(pair_record["pair_dir"])
        obj_id = str(pair_record["obj_id"])
        records = collect_round_records(pair_dir, obj_id)
        best_record = select_best_round(records)
        if best_record is None:
            worker_summary["pairs"][pair_record["pair_name"]] = {
                "return_code": -1,
                "error": "best_round_not_found",
                "source_pair_dir": str(pair_dir),
            }
            worker_summary["success"] = False
            continue

        result = render_one_pair(
            pair_record=pair_record,
            best_record=best_record,
            shard_dir=shard_dir,
            env=env,
            blender_bin=args.blender_bin,
            meshes_dir=args.meshes_dir,
            export_template_path=args.export_template_path,
            resolution=int(args.resolution or 1024),
            engine=str(args.engine or "CYCLES"),
        )
        worker_summary["pairs"][pair_record["pair_name"]] = result
        if result.get("return_code") != 0:
            worker_summary["success"] = False

    worker_summary["elapsed_seconds"] = round(time.time() - worker_summary["started_at"], 3)
    save_json(Path(args.worker_log_path), worker_summary)
    return 0 if worker_summary["success"] else 1


def move_pair_dirs(output_root: Path, shard_dir: Path, pair_names: List[str]) -> Dict[str, str]:
    moved = {}
    for pair_name in pair_names:
        src = shard_dir / pair_name
        if not src.exists():
            continue
        dst = output_root / pair_name
        if dst.exists():
            raise FileExistsError(f"Destination already exists: {dst}")
        shutil.move(str(src), str(dst))
        moved[pair_name] = str(dst)
    return moved


def run_orchestrator(args) -> int:
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    shard_root = output_root / "_shards"
    log_root = output_root / "_logs"
    shard_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    pair_records = discover_pair_dirs(source_root, args.pair_names)
    if not pair_records:
        raise FileNotFoundError(f"No pair dirs found under {source_root}")

    export_template_path = build_export_template(args, output_root)
    sample_template = load_json(export_template_path, default={}) or {}
    effective_resolution = int(args.resolution or sample_template.get("render_resolution", 1024))
    effective_engine = str(args.engine or sample_template.get("render_engine", "CYCLES"))

    request = {
        "source_root": str(source_root),
        "output_dir": str(output_root),
        "pair_names": [item["pair_name"] for item in pair_records],
        "gpus": parse_gpus(args.gpus),
        "meshes_dir": os.path.abspath(args.meshes_dir),
        "scene_template": str(export_template_path),
        "camera_mode": args.camera_mode,
        "resolution": effective_resolution,
        "engine": effective_engine,
        "qc_views": sample_template.get("qc_views", []),
        "requested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(output_root / "export_request.json", request)

    launched = []
    shard_plan = shard_pairs(pair_records, parse_gpus(args.gpus))
    for index, shard in enumerate(shard_plan):
        gpu = str(shard["gpu"])
        pair_subset = list(shard["pairs"])
        gpu_slug = slugify_gpu(gpu)
        shard_dir = shard_root / f"shard_gpu_{gpu_slug}"
        log_path = log_root / f"worker_gpu_{gpu_slug}.json"
        manifest_path = log_root / f"worker_gpu_{gpu_slug}_manifest.json"
        shard_dir.mkdir(parents=True, exist_ok=True)
        save_json(manifest_path, {"pairs": pair_subset})

        cmd = [
            args.python_bin,
            str(SCRIPT_PATH),
            "--worker-mode",
            "--source-root",
            str(source_root),
            "--output-dir",
            str(output_root),
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
            "--worker-manifest-path",
            str(manifest_path),
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
        launched.append(
            {
                "gpu": gpu,
                "pair_names": [item["pair_name"] for item in pair_subset],
                "process": process,
                "shard_dir": shard_dir,
                "log_path": log_path,
            }
        )
        if index < len(shard_plan) - 1 and args.launch_stagger_seconds > 0:
            time.sleep(args.launch_stagger_seconds)

    overall_success = True
    merged_pair_dirs = {}
    merged_pairs = {}

    for shard in launched:
        return_code = shard["process"].wait()
        log_json = load_json(shard["log_path"], default={}) or {}
        if return_code != 0 or not log_json.get("success", False):
            overall_success = False
        merged_pairs.update(log_json.get("pairs", {}))
        merged_pair_dirs.update(move_pair_dirs(output_root, shard["shard_dir"], shard["pair_names"]))

    total_frames = 0
    for pair_name, pair_dir in merged_pair_dirs.items():
        metadata = load_json(Path(pair_dir) / "metadata.json", default={}) or {}
        total_frames += len(metadata.get("frames", []))
        merged_pairs.setdefault(pair_name, {})["final_dir"] = pair_dir
        merged_pairs[pair_name]["total_frames"] = len(metadata.get("frames", []))

    summary = {
        "multi_gpu": True,
        "success": overall_success,
        "source_root": str(source_root),
        "gpus": parse_gpus(args.gpus),
        "total_pairs": len(merged_pair_dirs),
        "total_frames": total_frames,
        "camera_mode": args.camera_mode,
        "resolution": effective_resolution,
        "engine": effective_engine,
        "pair_dirs": merged_pair_dirs,
        "pairs": merged_pairs,
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
