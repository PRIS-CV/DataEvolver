#!/usr/bin/env python3
"""Run a VLM score gate over scene render candidates.

For each object/yaw pair, this script repeatedly calls run_scene_agent_step.py.
After each render/review round it checks a scheduled threshold:

    threshold(round) = min(threshold_max, threshold_start + round * threshold_step)

If the selected VLM score reaches the current threshold, the pair is accepted
into the gate manifest. Otherwise the script chooses the next actions from the
freeform VLM review and loops again. Pairs that do not pass within max rounds
are rejected and should not be exported into the next training dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parent.parent
PIPELINE_ROOT = REPO_ROOT / "pipeline"
for path in (str(SCRIPT_DIR), str(REPO_ROOT), str(PIPELINE_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
sys.path[:0] = [str(SCRIPT_DIR), str(REPO_ROOT), str(PIPELINE_ROOT)]

from asset_lifecycle import detect_asset_viability  # noqa: E402
from run_scene_agent_step import run_agent_step  # noqa: E402
from run_scene_agent_monitor import decide_actions, extract_trace_text  # noqa: E402


DEFAULT_BLENDER = "blender"
DEFAULT_PROFILE = REPO_ROOT / "configs" / "dataset_profiles" / "scene_v7.json"
DEFAULT_SCENE_TEMPLATE = REPO_ROOT / "configs" / "scene_template.json"
DEFAULT_SCENE_POOL = REPO_ROOT / "configs" / "scene_pool.json"
DEFAULT_MESHES_DIR = REPO_ROOT / "pipeline" / "data" / "meshes"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM quality gate loop for scene assets")
    parser.add_argument("--root-dir", required=True, help="Gate output root")
    parser.add_argument("--objects-file", default=None, help="Seed concept JSON list")
    parser.add_argument("--ids", default=None, help="Comma-separated obj ids")
    parser.add_argument("--rotations", default="0", help="Comma-separated yaw degrees to gate")
    parser.add_argument("--threshold-start", type=float, default=0.58)
    parser.add_argument("--threshold-step", type=float, default=0.03)
    parser.add_argument("--threshold-max", type=float, default=0.72)
    parser.add_argument("--score-field", choices=["hybrid_score", "vlm_only_score"], default="hybrid_score")
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--min-vlm-only-score", type=float, default=None)
    parser.add_argument("--step-scale", type=float, default=0.25)
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early-stop after this many consecutive rounds without setting a new best score. Use <=0 to disable.",
    )
    parser.add_argument(
        "--rollback-score-drop-threshold",
        type=float,
        default=0.05,
        help="If the current score is worse than the historical best by at least this amount, continue from best state next round.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU indices for multi-GPU parallel (e.g. 0,1,2). "
                             "Overrides --device. Objects are sharded across GPUs.")
    parser.add_argument("--blender", dest="blender_bin", default=DEFAULT_BLENDER)
    parser.add_argument("--meshes-dir", default=str(DEFAULT_MESHES_DIR))
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--scene-template", default=str(DEFAULT_SCENE_TEMPLATE))
    parser.add_argument("--scene-pool", default=None,
                        help="Path to scene_pool.json. When set, each object gets a deterministic scene from the pool, "
                             "overriding --scene-template.")
    parser.add_argument("--reject-abandon", action="store_true", help="Reject immediately when asset_viability=abandon")
    parser.add_argument("--disable-material-gate", action="store_true", help="Do not block acceptance on Stage4 mesh/material gate metadata")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_ids(args: argparse.Namespace) -> list[str]:
    if args.ids:
        return [item.strip() for item in args.ids.split(",") if item.strip()]
    if not args.objects_file:
        raise ValueError("Either --ids or --objects-file is required")
    payload = load_json(Path(args.objects_file).resolve(), default=[])
    if not isinstance(payload, list):
        raise ValueError("--objects-file must be a JSON list")
    ids = []
    for item in payload:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]).strip())
        elif isinstance(item, (list, tuple)) and item:
            ids.append(str(item[0]).strip())
    return [obj_id for obj_id in ids if obj_id]


def parse_rotations(raw: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        return [0]
    return values


def threshold_for_round(round_idx: int, start: float, step: float, max_value: float) -> float:
    return min(float(max_value), float(start) + int(round_idx) * float(step))


def coerce_score(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_asset_viability(value) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"continue", "abandon", "unclear"}:
        return raw
    return "unclear"


def read_trace_for_agg(agg_path: Optional[Path]) -> str:
    if not agg_path or not agg_path.exists():
        return ""
    reviews_dir = agg_path.parent
    trace_candidates = sorted(reviews_dir.glob(agg_path.name.replace("_agg.json", "*_trace.json")))
    if not trace_candidates:
        trace_candidates = sorted(reviews_dir.glob("*_trace.json"))
    trace_payload = load_json(trace_candidates[-1], default={}) if trace_candidates else {}
    agg = load_json(agg_path, default={}) or {}
    return extract_trace_text(trace_payload or {}, agg)


def resolve_asset_viability(result: dict) -> tuple[str, Optional[str], Optional[str]]:
    direct_viability = normalize_asset_viability(result.get("asset_viability"))
    direct_reason = str(result.get("abandon_reason") or "").strip() or None
    if direct_viability == "abandon":
        return direct_viability, direct_reason, None

    agg_path_raw = result.get("review_agg_path")
    agg_path = Path(agg_path_raw) if agg_path_raw else None
    agg = load_json(agg_path, default={}) if agg_path and agg_path.exists() else {}
    if not isinstance(agg, dict):
        agg = {}
    trace_text = read_trace_for_agg(agg_path)
    detected_viability, detected_reason, detected_confidence = detect_asset_viability(trace_text, agg)
    detected_viability = normalize_asset_viability(detected_viability)
    if detected_viability != "unclear":
        return detected_viability, detected_reason, detected_confidence
    return direct_viability, direct_reason, None


def resolve_material_gate(result: dict, enabled: bool = True) -> tuple[bool, Optional[str], dict]:
    quality = result.get("mesh_material_quality")
    if not enabled:
        return True, None, quality if isinstance(quality, dict) else {}
    if not isinstance(quality, dict) or not quality:
        return False, "missing_material_gate_metadata", quality if isinstance(quality, dict) else {}
    has_image_texture = quality.get("has_image_texture")
    if has_image_texture is not True:
        if has_image_texture is False:
            return False, "missing_image_texture", quality
        return False, "missing_has_image_texture_flag", quality
    passed = quality.get("material_gate_passed")
    if passed is True:
        return True, None, quality
    if passed is False:
        return False, str(quality.get("material_gate_reason") or "material_gate_failed"), quality
    return False, "missing_material_gate_pass_flag", quality


def choose_next_actions(result: dict, round_idx: int) -> tuple[list[str], list[str]]:
    agg_path_raw = result.get("review_agg_path")
    agg_path = Path(agg_path_raw) if agg_path_raw else None
    agg = load_json(agg_path, default={}) if agg_path else {}
    if not isinstance(agg, dict):
        agg = {}
    trace_text = read_trace_for_agg(agg_path)
    actions, reasons = decide_actions(agg, trace_text, round_idx)
    if not actions:
        actions = [str(v) for v in (result.get("suggested_actions") or []) if v and str(v) != "NO_OP"]
        reasons = ["fallback to suggested_actions"] * len(actions)
    return actions[:4], reasons[:4]


def pair_name(obj_id: str, rotation_deg: int) -> str:
    return f"{obj_id}_yaw{int(rotation_deg):03d}"


def existing_gate_status(pair_dir: Path) -> Optional[dict]:
    status_path = pair_dir / "gate_status.json"
    data = load_json(status_path, default=None)
    return data if isinstance(data, dict) else None


def score_improved(score: Optional[float], best_score: Optional[float], eps: float = 1e-6) -> bool:
    if score is None:
        return False
    if best_score is None:
        return True
    return float(score) > float(best_score) + float(eps)


def run_pair(args: argparse.Namespace, obj_id: str, rotation_deg: int, scene_pool: list[dict] | None = None) -> dict:
    root_dir = Path(args.root_dir).resolve()
    pair_dir = root_dir / pair_name(obj_id, rotation_deg)
    scene_template = resolve_scene_template_for_obj(obj_id, scene_pool, args.scene_template)
    if args.resume:
        status = existing_gate_status(pair_dir)
        if status and status.get("status") in {"accepted", "rejected"}:
            return status

    actions: list[str] = []
    action_reasons: list[str] = []
    rounds = []
    started = time.time()
    best_score: Optional[float] = None
    best_round_idx: Optional[int] = None
    best_state_out: Optional[str] = None
    best_render_dir: Optional[str] = None
    best_review_agg_path: Optional[str] = None
    best_vlm_only_score: Optional[float] = None
    next_control_state_in: Optional[str] = None
    next_prev_renders_dir: Optional[str] = None
    non_improve_streak = 0

    for round_idx in range(max(1, int(args.max_rounds))):
        threshold = threshold_for_round(
            round_idx,
            args.threshold_start,
            args.threshold_step,
            args.threshold_max,
        )
        result = run_agent_step(
            output_dir=str(pair_dir),
            obj_id=obj_id,
            rotation_deg=rotation_deg,
            round_idx=round_idx,
            actions=actions,
            step_scale=args.step_scale,
            agent_note=f"quality_gate threshold={threshold:.3f}; reasons={' | '.join(action_reasons)}",
            control_state_in=next_control_state_in,
            prev_renders_dir=next_prev_renders_dir,
            device=args.device,
            blender_bin=args.blender_bin,
            meshes_dir=args.meshes_dir,
            profile_path=args.profile,
            scene_template_path=scene_template,
        )

        score = coerce_score(result.get(args.score_field))
        vlm_only = coerce_score(result.get("vlm_only_score"))
        asset_viability, abandon_reason, abandon_confidence = resolve_asset_viability(result)
        material_gate_passed, material_gate_reason, material_quality = resolve_material_gate(
            result,
            enabled=not args.disable_material_gate,
        )
        passes_score = score is not None and score >= threshold
        passes_vlm_only = args.min_vlm_only_score is None or (
            vlm_only is not None and vlm_only >= float(args.min_vlm_only_score)
        )
        abandon_block = args.reject_abandon and asset_viability == "abandon"
        material_block = not material_gate_passed
        improved = score_improved(score, best_score)
        if improved:
            best_score = score
            best_round_idx = round_idx
            best_state_out = result.get("state_out")
            best_render_dir = result.get("render_dir")
            best_review_agg_path = result.get("review_agg_path")
            best_vlm_only_score = vlm_only
            non_improve_streak = 0
        else:
            non_improve_streak += 1

        rollback_applied = False
        rollback_reason = None
        rollback_target_round_idx = None
        rollback_threshold = float(args.rollback_score_drop_threshold)
        if (
            best_score is not None
            and best_state_out
            and best_render_dir
            and score is not None
            and rollback_threshold > 0
            and float(best_score) - float(score) >= rollback_threshold
        ):
            next_control_state_in = best_state_out
            next_prev_renders_dir = best_render_dir
            rollback_applied = True
            rollback_target_round_idx = best_round_idx
            rollback_reason = f"score_drop_from_best>={rollback_threshold:.3f}"
        else:
            next_control_state_in = result.get("state_out") or best_state_out
            next_prev_renders_dir = result.get("render_dir") or best_render_dir

        round_record = {
            "round_idx": round_idx,
            "threshold": threshold,
            "score_field": args.score_field,
            "score": score,
            "vlm_only_score": vlm_only,
            "passes_score": passes_score,
            "passes_vlm_only": passes_vlm_only,
            "passes_material_gate": material_gate_passed,
            "material_gate_reason": material_gate_reason,
            "mesh_material_quality": material_quality,
            "asset_viability": asset_viability,
            "abandon_reason": abandon_reason,
            "abandon_confidence": abandon_confidence,
            "issue_tags": result.get("issue_tags"),
            "actions_in": actions,
            "state_in": result.get("state_in"),
            "state_out": result.get("state_out"),
            "best_score_after_round": best_score,
            "best_round_idx_after_round": best_round_idx,
            "best_vlm_only_score_after_round": best_vlm_only_score,
            "non_improve_streak_after_round": non_improve_streak,
            "rollback_applied_for_next_round": rollback_applied,
            "rollback_target_round_idx": rollback_target_round_idx,
            "rollback_reason": rollback_reason,
            "next_control_state_in": next_control_state_in,
            "result_path": str(pair_dir / f"agent_round{round_idx:02d}.json"),
        }
        rounds.append(round_record)

        if result.get("success") and passes_score and passes_vlm_only and not abandon_block and not material_block:
            status = {
                "status": "accepted",
                "obj_id": obj_id,
                "rotation_deg": rotation_deg,
                "pair_dir": str(pair_dir),
                "accepted_round_idx": round_idx,
                "accepted_score": score,
                "accepted_threshold": threshold,
                "score_field": args.score_field,
                "asset_viability": asset_viability,
                "abandon_reason": abandon_reason,
                "mesh_material_quality": material_quality,
                "best_score": best_score,
                "best_round_idx": best_round_idx,
                "best_state_out": best_state_out,
                "best_render_dir": best_render_dir,
                "best_review_agg_path": best_review_agg_path,
                "state_out": result.get("state_out"),
                "render_dir": result.get("render_dir"),
                "review_agg_path": result.get("review_agg_path"),
                "rounds": rounds,
                "elapsed_seconds": round(time.time() - started, 3),
            }
            save_json(pair_dir / "gate_status.json", status)
            return status

        if abandon_block:
            status = {
                "status": "rejected",
                "reason": "asset_viability_abandon",
                "obj_id": obj_id,
                "rotation_deg": rotation_deg,
                "pair_dir": str(pair_dir),
                "asset_viability": asset_viability,
                "abandon_reason": abandon_reason,
                "abandon_confidence": abandon_confidence,
                "best_score": best_score,
                "best_round_idx": best_round_idx,
                "best_state_out": best_state_out,
                "best_render_dir": best_render_dir,
                "best_review_agg_path": best_review_agg_path,
                "rounds": rounds,
                "elapsed_seconds": round(time.time() - started, 3),
            }
            save_json(pair_dir / "gate_status.json", status)
            return status

        actions, action_reasons = choose_next_actions(result, round_idx)
        save_json(
            pair_dir / f"gate_decision_round{round_idx:02d}.json",
            {
                "round_idx": round_idx,
                "threshold": threshold,
                "score": score,
                "passed": bool(passes_score and passes_vlm_only and not material_block),
                "material_gate_passed": material_gate_passed,
                "material_gate_reason": material_gate_reason,
                "asset_viability": asset_viability,
                "abandon_reason": abandon_reason,
                "best_score_after_round": best_score,
                "best_round_idx_after_round": best_round_idx,
                "non_improve_streak_after_round": non_improve_streak,
                "rollback_applied_for_next_round": rollback_applied,
                "rollback_target_round_idx": rollback_target_round_idx,
                "rollback_reason": rollback_reason,
                "next_control_state_in": next_control_state_in,
                "next_actions": actions,
                "next_action_reasons": action_reasons,
            },
        )

        if int(args.patience) > 0 and non_improve_streak >= int(args.patience):
            status = {
                "status": "rejected",
                "reason": "patience_exhausted_without_improvement",
                "obj_id": obj_id,
                "rotation_deg": rotation_deg,
                "pair_dir": str(pair_dir),
                "best_score": best_score,
                "best_round_idx": best_round_idx,
                "best_state_out": best_state_out,
                "best_render_dir": best_render_dir,
                "best_review_agg_path": best_review_agg_path,
                "last_score": score,
                "last_threshold": threshold,
                "patience": int(args.patience),
                "non_improve_streak": non_improve_streak,
                "rounds": rounds,
                "elapsed_seconds": round(time.time() - started, 3),
            }
            save_json(pair_dir / "gate_status.json", status)
            return status

    last = rounds[-1] if rounds else {}
    status = {
        "status": "rejected",
        "reason": "max_rounds_without_passing_gate",
        "obj_id": obj_id,
        "rotation_deg": rotation_deg,
        "pair_dir": str(pair_dir),
        "best_score": best_score,
        "best_round_idx": best_round_idx,
        "best_state_out": best_state_out,
        "best_render_dir": best_render_dir,
        "best_review_agg_path": best_review_agg_path,
        "last_score": last.get("score"),
        "last_threshold": last.get("threshold"),
        "last_material_gate_passed": last.get("passes_material_gate"),
        "last_material_gate_reason": last.get("material_gate_reason"),
        "rounds": rounds,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    save_json(pair_dir / "gate_status.json", status)
    return status


def summarize(statuses: Iterable[dict]) -> dict:
    statuses = list(statuses)
    accepted = [item for item in statuses if item.get("status") == "accepted"]
    rejected = [item for item in statuses if item.get("status") == "rejected"]
    return {
        "success": True,
        "total": len(statuses),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted": accepted,
        "rejected": rejected,
    }


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    obj_ids = parse_ids(args)

    scene_pool = None
    if args.scene_pool:
        scene_pool = load_scene_pool(args.scene_pool)
        assignments = {
            oid: resolve_scene_template_for_obj(oid, scene_pool, args.scene_template)
            for oid in obj_ids
        }
        assign_path = root_dir / "scene_assignments.json"
        if not assign_path.exists() or args.gpus:
            save_json(assign_path, assignments)
        for oid, tmpl in assignments.items():
            print(f"[scene-pool] {oid} → {Path(tmpl).name}")

    if args.gpus and len(args.gpus.split(",")) > 1:
        run_multigpu(args, obj_ids, root_dir)
        return

    rotations = parse_rotations(args.rotations)
    statuses = []
    for obj_id in obj_ids:
        for rotation_deg in rotations:
            print(f"[gate] {obj_id} yaw{rotation_deg:03d}")
            status = run_pair(args, obj_id, rotation_deg, scene_pool=scene_pool)
            print(
                f"  -> {status.get('status')} score={status.get('accepted_score', status.get('last_score'))} "
                f"threshold={status.get('accepted_threshold', status.get('last_threshold'))}"
            )
            statuses.append(status)
            save_json(root_dir / "gate_manifest_partial.json", summarize(statuses))

    manifest = summarize(statuses)
    save_json(root_dir / "gate_manifest.json", manifest)
    save_json(root_dir / "accepted_manifest.json", {"objects": manifest["accepted"]})
    save_json(root_dir / "rejected_manifest.json", {"objects": manifest["rejected"]})
    print(f"[gate] accepted={manifest['accepted_count']} rejected={manifest['rejected_count']} -> {root_dir / 'gate_manifest.json'}")


def run_multigpu(args: argparse.Namespace, obj_ids: list[str], root_dir: Path) -> None:
    """Shard objects across GPUs, launch parallel workers, merge results."""
    import subprocess as _sp

    gpu_indices = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_gpus = len(gpu_indices)

    shards: list[list[str]] = [[] for _ in range(n_gpus)]
    for i, obj_id in enumerate(obj_ids):
        shards[i % n_gpus].append(obj_id)

    print(f"[multigpu] {len(obj_ids)} objects → {n_gpus} GPUs: {[len(s) for s in shards]}")

    VLM_LOAD_STAGGER_SECONDS = 90  # stagger worker starts to avoid simultaneous VLM loading OOM

    workers: list[tuple[_sp.Popen, int, Path]] = []
    for idx, (gpu, shard) in enumerate(zip(gpu_indices, shards)):
        if not shard:
            continue
        if idx > 0:
            print(f"  [multigpu] waiting {VLM_LOAD_STAGGER_SECONDS}s before launching worker {idx} (stagger VLM load)...")
            time.sleep(VLM_LOAD_STAGGER_SECONDS)
        worker_dir = root_dir / f"_worker_{gpu}"
        worker_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(SCRIPT_PATH),
            "--root-dir", str(root_dir),
            "--ids", ",".join(shard),
            "--rotations", args.rotations,
            "--threshold-start", str(args.threshold_start),
            "--threshold-step", str(args.threshold_step),
            "--threshold-max", str(args.threshold_max),
            "--score-field", args.score_field,
            "--max-rounds", str(args.max_rounds),
            "--step-scale", str(args.step_scale),
            "--patience", str(args.patience),
            "--rollback-score-drop-threshold", str(args.rollback_score_drop_threshold),
            "--device", "cuda:0",
            "--blender", args.blender_bin,
            "--meshes-dir", args.meshes_dir,
            "--profile", args.profile,
            "--scene-template", args.scene_template,
        ]
        if args.scene_pool:
            cmd += ["--scene-pool", args.scene_pool]
        if args.min_vlm_only_score is not None:
            cmd += ["--min-vlm-only-score", str(args.min_vlm_only_score)]
        if args.reject_abandon:
            cmd.append("--reject-abandon")
        if args.disable_material_gate:
            cmd.append("--disable-material-gate")
        if args.resume:
            cmd.append("--resume")

        log_path = worker_dir / "worker.log"
        fh = open(log_path, "w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["VLM_FORCE_SINGLE_GPU"] = "1"
        proc = _sp.Popen(cmd, stdout=fh, stderr=_sp.STDOUT, env=env)
        workers.append((proc, int(gpu), log_path))
        print(f"  [worker GPU {gpu}] PID {proc.pid}, {len(shard)} objects: {shard[:3]}{'...' if len(shard) > 3 else ''}")

    all_statuses = []
    for proc, gpu, log_path in workers:
        ret = proc.wait()
        print(f"  [worker GPU {gpu}] exited {ret}")
        if ret != 0:
            print(f"  [worker GPU {gpu}] FAILED — see {log_path}")

    for worker_manifest in root_dir.glob("gate_manifest_partial.json"):
        pass

    for obj_id in obj_ids:
        rotations = parse_rotations(args.rotations)
        for rotation_deg in rotations:
            pair_key = f"{obj_id}_yaw{rotation_deg:03d}"
            status_path = root_dir / pair_key / "gate_status.json"
            if status_path.exists():
                all_statuses.append(json.loads(status_path.read_text()))

    manifest = summarize(all_statuses)
    save_json(root_dir / "gate_manifest.json", manifest)
    save_json(root_dir / "accepted_manifest.json", {"objects": manifest["accepted"]})
    save_json(root_dir / "rejected_manifest.json", {"objects": manifest["rejected"]})
    print(f"[multigpu] accepted={manifest['accepted_count']} rejected={manifest['rejected_count']} -> {root_dir / 'gate_manifest.json'}")


if __name__ == "__main__":
    main()
