"""
Export a rotation8 dataset by rotating the object, not the camera.

For each object:
1. scan the pair-evolution root (obj_xxx_yaw000/090/180/270)
2. select the single best pair-level result by best available hybrid score
3. take that pair's best control state as the base scene configuration
4. override only object.yaw_deg to the requested horizontal yaws
5. render one fixed-camera image per yaw

Outputs a standalone dataset directory with flattened per-object files plus
summary / object manifests.
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

from export_scene_multiview_from_pair_evolution import (
    collect_round_records,
    discover_pair_dirs,
    load_json,
    save_json,
    select_best_round,
)


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
SCENE_RENDER_SCRIPT = REPO_ROOT / "pipeline" / "stage4_scene_render.py"
DEFAULT_BLENDER = os.environ.get("BLENDER_BIN", "blender")
DEFAULT_TEMPLATE = REPO_ROOT / "configs" / "scene_template.json"
DEFAULT_SCENE_POOL = REPO_ROOT / "configs" / "scene_pool.json"
DEFAULT_MESHES_DIR = REPO_ROOT / "pipeline" / "data" / "meshes"
DEFAULT_ROTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]


def load_scene_pool(pool_path: str | Path) -> list[dict]:
    pool_path = Path(pool_path)
    if not pool_path.exists():
        raise FileNotFoundError(f"Scene pool not found: {pool_path}")
    data = json.loads(pool_path.read_text(encoding="utf-8"))
    scenes = data.get("scenes", [])
    if not scenes:
        raise ValueError(f"Scene pool is empty: {pool_path}")
    return scenes


def resolve_scene_template_for_obj(obj_id: str, scene_pool: list[dict] | None, default_template: str) -> str:
    if not scene_pool:
        return default_template
    import re as _re
    _m = _re.search(r"(\d+)", obj_id)
    idx = (int(_m.group(1)) if _m else hash(obj_id)) % len(scene_pool)
    scene = scene_pool[idx]
    template = scene.get("template", default_template)
    if not Path(template).is_absolute():
        template = str(REPO_ROOT / template)
    return template


def parse_args():
    parser = argparse.ArgumentParser(description="Export rotation8 dataset from best per-object pair state")
    parser.add_argument("--source-root", required=True, help="Pair evolution root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--blender", dest="blender_bin", default=DEFAULT_BLENDER)
    parser.add_argument("--meshes-dir", default=str(DEFAULT_MESHES_DIR))
    parser.add_argument("--scene-template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--scene-pool", default=None,
                        help="Path to scene_pool.json. When set, each object gets a deterministic scene from the pool.")
    parser.add_argument("--scene-assignments", default=None,
                        help="Path to scene_assignments.json from gate loop. Reuses same object→scene mapping.")
    parser.add_argument("--engine", choices=["EEVEE", "CYCLES"], default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--rotations", default="0,45,90,135,180,225,270,315")
    parser.add_argument(
        "--base-rotation-deg",
        type=int,
        default=0,
        help="Canonical source pair rotation. By default, use yaw000 best state as the base for all target rotations.",
    )
    parser.add_argument(
        "--fallback-to-best-any-angle",
        action="store_true",
        help="If canonical base rotation is missing for an object, fall back to the object's best pair from any available angle.",
    )
    parser.add_argument("--launch-stagger-seconds", type=float, default=2.0)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-gpu", default=None)
    parser.add_argument("--worker-manifest-path", default=None)
    parser.add_argument("--worker-shard-dir", default=None)
    parser.add_argument("--worker-log-path", default=None)
    return parser.parse_args()


def parse_gpus(raw: str) -> List[str]:
    items = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not items:
        raise ValueError("No GPUs specified")
    return items


def parse_rotations(raw: str) -> List[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        return list(DEFAULT_ROTATIONS)
    return values


def slugify_gpu(gpu: str) -> str:
    return gpu.replace(":", "_").replace("/", "_").replace("\\", "_")


def _relative_str(path: Optional[Path], base: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _copy_if_exists(src: Optional[Path], dst: Path) -> Optional[str]:
    if src is None or not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def select_best_pair_per_object(
    source_root: Path,
    *,
    base_rotation_deg: int,
    fallback_to_best_any_angle: bool,
) -> List[dict]:
    pair_records = discover_pair_dirs(source_root, requested_pair_names=None)
    by_object_any: Dict[str, dict] = {}
    by_object_canonical: Dict[str, dict] = {}

    for pair_record in pair_records:
        pair_dir = Path(pair_record["pair_dir"])
        obj_id = str(pair_record["obj_id"])
        round_records = collect_round_records(pair_dir, obj_id)
        best_round = select_best_round(round_records)
        if best_round is None:
            continue

        score = best_round.get("hybrid_score")
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = -1.0

        candidate = {
            "obj_id": obj_id,
            "pair_name": pair_record["pair_name"],
            "rotation_deg": int(pair_record["rotation_deg"]),
            "pair_dir": str(pair_dir),
            "best_round_idx": int(best_round["round_idx"]),
            "hybrid_score": score,
            "control_state_path": best_round.get("control_state_path"),
            "selection_reason": best_round.get("selection_reason"),
            "issue_tags": best_round.get("issue_tags") or [],
            "asset_viability": best_round.get("asset_viability"),
            "structure_consistency": best_round.get("structure_consistency"),
        }

        existing = by_object_any.get(obj_id)
        if existing is None or (candidate["hybrid_score"], -candidate["best_round_idx"]) > (
            existing["hybrid_score"],
            -existing["best_round_idx"],
        ):
            by_object_any[obj_id] = candidate

        if candidate["rotation_deg"] == int(base_rotation_deg):
            existing_canonical = by_object_canonical.get(obj_id)
            if existing_canonical is None or (candidate["hybrid_score"], -candidate["best_round_idx"]) > (
                existing_canonical["hybrid_score"],
                -existing_canonical["best_round_idx"],
            ):
                by_object_canonical[obj_id] = candidate

    selected: Dict[str, dict] = {}
    all_obj_ids = sorted(by_object_any)
    for obj_id in all_obj_ids:
        if obj_id in by_object_canonical:
            chosen = dict(by_object_canonical[obj_id])
            chosen["base_selection_mode"] = f"canonical_yaw{int(base_rotation_deg):03d}"
            selected[obj_id] = chosen
        elif fallback_to_best_any_angle:
            chosen = dict(by_object_any[obj_id])
            chosen["base_selection_mode"] = "fallback_best_any_angle"
            selected[obj_id] = chosen
        else:
            raise FileNotFoundError(
                f"Missing canonical base pair for {obj_id}: yaw{int(base_rotation_deg):03d}"
            )

    return [selected[obj_id] for obj_id in sorted(selected)]


def shard_objects(records: List[dict], gpus: List[str]) -> List[Dict[str, object]]:
    buckets = [{"gpu": gpu, "objects": []} for gpu in gpus]
    for index, record in enumerate(records):
        buckets[index % len(gpus)]["objects"].append(record)
    return [bucket for bucket in buckets if bucket["objects"]]


def build_scene_template(base_template_path: Path, engine: Optional[str], resolution: Optional[int], output_root: Path, suffix: str = "") -> Path:
    template = load_json(base_template_path, default={}) or {}
    if engine:
        template["render_engine"] = engine
    if resolution:
        template["render_resolution"] = resolution
    template["qc_views"] = [[0, 0]]
    name = f"scene_template_fixed_camera{suffix}.json"
    out_path = output_root / name
    save_json(out_path, template)
    return out_path


def render_one_rotation(
    *,
    record: dict,
    rotation_deg: int,
    shard_dir: Path,
    env: dict,
    blender_bin: str,
    meshes_dir: str,
    scene_template_path: str,
    resolution: int,
    engine: str,
    output_root: Path,
) -> dict:
    obj_id = str(record["obj_id"])
    pair_name = str(record["pair_name"])
    obj_dir = output_root / "objects" / obj_id
    obj_dir.mkdir(parents=True, exist_ok=True)

    base_control_path = Path(str(record["control_state_path"])).resolve()
    base_control = load_json(base_control_path, default={}) or {}
    base_control.setdefault("object", {})
    base_control["object"]["yaw_deg"] = float(rotation_deg)

    rot_slug = f"yaw{rotation_deg:03d}"
    temp_root = shard_dir / f"_tmp_{obj_id}_{rot_slug}"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    temp_control_path = temp_root / f"{rot_slug}_control.json"
    save_json(temp_control_path, base_control)

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
        str(temp_control_path),
        "--scene-template",
        scene_template_path,
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

    render_obj_dir = temp_root / obj_id
    rgb_src = render_obj_dir / "az000_el+00.png"
    mask_src = render_obj_dir / "az000_el+00_mask.png"
    meta_src = render_obj_dir / "metadata.json"
    if completed.returncode != 0 or not rgb_src.exists():
        return {
            "return_code": completed.returncode if completed.returncode != 0 else -1,
            "elapsed_seconds": round(time.time() - started, 3),
            "error": "render_failed_or_missing_rgb",
            "stdout_tail": completed.stdout[-4000:],
        }

    rgb_out = Path(_copy_if_exists(rgb_src, obj_dir / f"{rot_slug}.png"))
    mask_out = Path(_copy_if_exists(mask_src, obj_dir / f"{rot_slug}_mask.png")) if mask_src.exists() else None
    meta_out = Path(_copy_if_exists(meta_src, obj_dir / f"{rot_slug}_render_metadata.json")) if meta_src.exists() else None
    control_out = Path(_copy_if_exists(temp_control_path, obj_dir / f"{rot_slug}_control.json"))

    # Copy depth and normal maps if they were rendered
    depth_out = None
    normal_out = None
    for f in render_obj_dir.iterdir():
        if "depth" in f.name and f.suffix in (".exr", ".png"):
            depth_out = Path(_copy_if_exists(f, obj_dir / f"{rot_slug}_depth{f.suffix}"))
        elif "normal" in f.name and f.suffix == ".png":
            normal_out = Path(_copy_if_exists(f, obj_dir / f"{rot_slug}_normal.png"))

    render_meta = load_json(meta_src, default={}) or {}
    result = {
        "obj_id": obj_id,
        "rotation_deg": rotation_deg,
        "pair_name": pair_name,
        "base_pair_rotation_deg": int(record["rotation_deg"]),
        "base_best_round_idx": int(record["best_round_idx"]),
        "base_hybrid_score": record.get("hybrid_score"),
        "base_control_state_path": str(base_control_path),
        "return_code": 0,
        "elapsed_seconds": round(time.time() - started, 3),
        "rgb_path": _relative_str(rgb_out, output_root),
        "mask_path": _relative_str(mask_out, output_root),
        "depth_path": _relative_str(depth_out, output_root) if depth_out else None,
        "normal_path": _relative_str(normal_out, output_root) if normal_out else None,
        "render_metadata_path": _relative_str(meta_out, output_root),
        "control_state_path": _relative_str(control_out, output_root),
        "camera_mode": render_meta.get("camera_mode"),
        "frames": len(render_meta.get("frames", [])),
    }
    shutil.rmtree(temp_root, ignore_errors=True)
    return result


def run_worker(args) -> int:
    assert args.worker_gpu is not None
    assert args.worker_manifest_path is not None
    assert args.worker_shard_dir is not None
    assert args.worker_log_path is not None

    output_root = Path(args.output_dir).resolve()
    shard_dir = Path(args.worker_shard_dir).resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(Path(args.worker_manifest_path), default={}) or {}
    rotations = manifest.get("rotations") or list(DEFAULT_ROTATIONS)
    records = manifest.get("objects") or []
    scene_assignments = manifest.get("scene_assignments") or {}
    default_scene_template = str((output_root / "scene_template_fixed_camera.json").resolve())

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.worker_gpu)

    worker_summary = {
        "gpu": args.worker_gpu,
        "rotations": rotations,
        "objects_total": len(records),
        "started_at": time.time(),
        "success": True,
        "objects": {},
    }

    for record in records:
        obj_id = str(record["obj_id"])
        obj_template = scene_assignments.get(obj_id, default_scene_template)
        obj_results = []
        for rotation_deg in rotations:
            result = render_one_rotation(
                record=record,
                rotation_deg=int(rotation_deg),
                shard_dir=shard_dir,
                env=env,
                blender_bin=args.blender_bin,
                meshes_dir=args.meshes_dir,
                scene_template_path=obj_template,
                resolution=int(args.resolution or 1024),
                engine=str(args.engine or "CYCLES"),
                output_root=output_root,
            )
            obj_results.append(result)
            if result.get("return_code") != 0:
                worker_summary["success"] = False
        worker_summary["objects"][obj_id] = {
            "base_selection": record,
            "rotations": obj_results,
        }

    worker_summary["elapsed_seconds"] = round(time.time() - worker_summary["started_at"], 3)
    save_json(Path(args.worker_log_path), worker_summary)
    return 0 if worker_summary["success"] else 1


def run_orchestrator(args) -> int:
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    shard_root = output_root / "_shards"
    log_root = output_root / "_logs"
    shard_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    rotations = parse_rotations(args.rotations)
    selected_objects = select_best_pair_per_object(
        source_root,
        base_rotation_deg=int(args.base_rotation_deg),
        fallback_to_best_any_angle=bool(args.fallback_to_best_any_angle),
    )
    if not selected_objects:
        raise FileNotFoundError(f"No objects discovered under {source_root}")

    template_path = build_scene_template(Path(args.scene_template), args.engine, args.resolution, output_root)
    template = load_json(template_path, default={}) or {}
    effective_resolution = int(args.resolution or template.get("render_resolution", 1024))
    effective_engine = str(args.engine or template.get("render_engine", "CYCLES"))

    scene_assignments: dict[str, str] = {}
    _built_templates: dict[str, str] = {}
    if args.scene_assignments:
        raw = load_json(Path(args.scene_assignments), default={}) or {}
        for oid, tmpl_path in raw.items():
            if tmpl_path not in _built_templates:
                suffix = f"_{Path(tmpl_path).stem}" if tmpl_path != args.scene_template else ""
                built = build_scene_template(Path(tmpl_path), args.engine, args.resolution, output_root, suffix=suffix)
                _built_templates[tmpl_path] = str(built.resolve())
            scene_assignments[oid] = _built_templates[tmpl_path]
        print(f"[export] Loaded {len(scene_assignments)} scene assignments from {args.scene_assignments}")
    elif args.scene_pool:
        pool = load_scene_pool(args.scene_pool)
        obj_ids = [str(r["obj_id"]) for r in selected_objects]
        for oid in obj_ids:
            raw_tmpl = resolve_scene_template_for_obj(oid, pool, args.scene_template)
            if raw_tmpl not in _built_templates:
                suffix = f"_{Path(raw_tmpl).stem}" if raw_tmpl != args.scene_template else ""
                built = build_scene_template(Path(raw_tmpl), args.engine, args.resolution, output_root, suffix=suffix)
                _built_templates[raw_tmpl] = str(built.resolve())
            scene_assignments[oid] = _built_templates[raw_tmpl]
        save_json(output_root / "scene_assignments.json", scene_assignments)
        for oid, tmpl in scene_assignments.items():
            print(f"[scene-pool] {oid} → {Path(tmpl).name}")

    request = {
        "source_root": str(source_root),
        "output_dir": str(output_root),
        "rotations": rotations,
        "gpus": parse_gpus(args.gpus),
        "meshes_dir": os.path.abspath(args.meshes_dir),
        "scene_template": str(template_path),
        "resolution": effective_resolution,
        "engine": effective_engine,
        "selected_objects": selected_objects,
        "base_rotation_deg": int(args.base_rotation_deg),
        "fallback_to_best_any_angle": bool(args.fallback_to_best_any_angle),
        "requested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(output_root / "export_request.json", request)

    launched = []
    for index, shard in enumerate(shard_objects(selected_objects, parse_gpus(args.gpus))):
        gpu = str(shard["gpu"])
        gpu_slug = slugify_gpu(gpu)
        manifest_path = log_root / f"worker_gpu_{gpu_slug}_manifest.json"
        log_path = log_root / f"worker_gpu_{gpu_slug}.json"
        shard_dir = shard_root / f"shard_gpu_{gpu_slug}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        save_json(manifest_path, {"objects": shard["objects"], "rotations": rotations, "scene_assignments": scene_assignments})

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
            "--resolution",
            str(effective_resolution),
            "--engine",
            effective_engine,
            "--rotations",
            ",".join(str(r) for r in rotations),
            "--base-rotation-deg",
            str(int(args.base_rotation_deg)),
            *(
                ["--fallback-to-best-any-angle"]
                if args.fallback_to_best_any_angle
                else []
            ),
            "--worker-gpu",
            gpu,
            "--worker-manifest-path",
            str(manifest_path),
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
                "process": process,
                "manifest_path": manifest_path,
                "log_path": log_path,
            }
        )
        if index < len(parse_gpus(args.gpus)) - 1 and args.launch_stagger_seconds > 0:
            time.sleep(args.launch_stagger_seconds)

    overall_success = True
    merged_objects: Dict[str, dict] = {}
    for launched_worker in launched:
        return_code = launched_worker["process"].wait()
        log_json = load_json(launched_worker["log_path"], default={}) or {}
        if return_code != 0 or not log_json.get("success", False):
            overall_success = False
        merged_objects.update(log_json.get("objects", {}))

    total_renders = sum(len(obj_payload.get("rotations", [])) for obj_payload in merged_objects.values())
    summary = {
        "success": overall_success,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "total_objects": len(merged_objects),
        "rotations_per_object": len(rotations),
        "total_renders": total_renders,
        "gpus": parse_gpus(args.gpus),
        "resolution": effective_resolution,
        "engine": effective_engine,
        "rotations": rotations,
    }
    save_json(output_root / "summary.json", summary)
    save_json(output_root / "manifest.json", {"summary": summary, "objects": merged_objects})

    for obj_id, payload in merged_objects.items():
        object_manifest = {
            "obj_id": obj_id,
            "base_selection": payload.get("base_selection"),
            "rotations": payload.get("rotations", []),
        }
        save_json(output_root / "objects" / obj_id / "object_manifest.json", object_manifest)

    return 0 if overall_success else 1


def main():
    args = parse_args()
    if args.worker_mode:
        raise SystemExit(run_worker(args))
    raise SystemExit(run_orchestrator(args))


if __name__ == "__main__":
    main()
