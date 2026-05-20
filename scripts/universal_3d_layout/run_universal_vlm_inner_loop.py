#!/usr/bin/env python3
"""Run the dual-object VLM inner loop on a selected subset of DataEvolver universal records.

This runner is intentionally persistent per GPU worker: it imports the dual
agent step once, so the Qwen VLM remains loaded across rounds and samples.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any


DEFAULT_DATASET_ROOT = Path(os.environ.get("DATAEVOLVER_UNIVERSAL_DATASET_ROOT") or "/aaaidata/jiazhuangzhuang/ARIS_runtime/universal_contract_batch_200_20260518")
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("DATAEVOLVER_UNIVERSAL_VLM_OUTPUT_ROOT") or "/aaaidata/jiazhuangzhuang/ARIS_runtime/universal_contract_selected108_vlm_scorecalib_t075_r5_20260519")
DEFAULT_DATAEVOLVER_ROOT = Path(os.environ.get("DATAEVOLVER_ROOT") or os.environ.get("ARIS_ROOT") or "/home/jiazhuangzhuang/ARIS")
DEFAULT_BLENDER = Path(os.environ.get("DATAEVOLVER_BLENDER") or os.environ.get("ARIS_BLENDER") or "/aaaidata/zhangqisong/blender-4.24/blender")
DEFAULT_VLM_MODEL_PATH = Path(os.environ.get("VLM_MODEL_PATH") or "/huggingface/model_hub/Qwen3.6-27B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataevolver-root", type=Path, default=DEFAULT_DATAEVOLVER_ROOT)
    parser.add_argument("--aris-root", dest="dataevolver_root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--vlm-model-path", type=Path, default=DEFAULT_VLM_MODEL_PATH)
    parser.add_argument("--selection-manifest", type=Path, default=None)
    parser.add_argument("--total", type=int, default=108)
    parser.add_argument("--per-scene", type=int, default=27)
    parser.add_argument("--needs-per-scene", type=int, default=18)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--engine", choices=["EEVEE", "CYCLES"], default="EEVEE")
    parser.add_argument("--step-scale", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--rollback-score-drop-threshold", type=float, default=0.05)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    return parser.parse_args()


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists() or path.is_dir():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_jsonl(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_index(record: dict) -> int:
    rid = record.get("record_id", "")
    try:
        return int(rid.split("_")[1])
    except Exception:
        return 10**9


def scene_key(record: dict) -> str:
    source_paths = record.get("provenance", {}).get("source_paths") or []
    if source_paths:
        return Path(source_paths[0]).stem
    return str(record.get("record_id", "")).rsplit("__", 1)[-1]


def mesh_rank(record: dict) -> int:
    total = 0
    for obj in record.get("objects") or []:
        text = str(obj.get("category") or obj.get("mesh_or_proxy") or "")
        digits = "".join(ch if ch.isdigit() else " " for ch in text).split()
        if digits:
            total += int(digits[-1])
    return total


def review_priority(record: dict, review: dict | None) -> tuple:
    review = review or {}
    needs = review.get("status") == "needs_vggt_inference"
    failures = review.get("failure_tags") or []
    mask_cov = float(record.get("validation", {}).get("mask_coverage") or 0.0)
    return (0 if needs else 1, -len(failures), -mesh_rank(record), mask_cov, record_index(record))


def load_reviews(dataset_root: Path) -> dict[str, dict]:
    review_path = dataset_root / "metadata" / "geometry_review_vggt_omega_proxy.jsonl"
    return {item.get("record_id"): item for item in load_jsonl(review_path) if item.get("record_id")}


def select_records(records: list[dict], reviews: dict[str, dict], args: argparse.Namespace) -> list[dict]:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_scene[scene_key(record)].append(record)

    selected: list[dict] = []
    for scene in sorted(by_scene):
        scene_records = sorted(by_scene[scene], key=lambda rec: review_priority(rec, reviews.get(rec.get("record_id"))))
        needs = [rec for rec in scene_records if (reviews.get(rec.get("record_id")) or {}).get("status") == "needs_vggt_inference"]
        accepted = [rec for rec in scene_records if (reviews.get(rec.get("record_id")) or {}).get("status") != "needs_vggt_inference"]

        chunk: list[dict] = []
        chunk.extend(needs[: min(args.needs_per_scene, args.per_scene)])
        accepted_target = max(0, args.per_scene - len(chunk))
        chunk.extend(accepted[:accepted_target])
        if len(chunk) < args.per_scene:
            already = {rec["record_id"] for rec in chunk}
            for rec in scene_records:
                if rec["record_id"] not in already:
                    chunk.append(rec)
                if len(chunk) >= args.per_scene:
                    break
        selected.extend(chunk[: args.per_scene])

    if len(selected) < args.total:
        already = {rec["record_id"] for rec in selected}
        for rec in sorted(records, key=lambda rec: review_priority(rec, reviews.get(rec.get("record_id")))):
            if rec["record_id"] not in already:
                selected.append(rec)
            if len(selected) >= args.total:
                break

    return sorted(selected[: args.total], key=record_index)


def write_selection_manifest(records: list[dict], reviews: dict[str, dict], args: argparse.Namespace) -> Path:
    out_root = args.output_root
    manifest_path = args.selection_manifest or (out_root / "metadata" / "selected_universal_manifest.json")
    jsonl_path = manifest_path.with_suffix(".jsonl")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    selected = select_records(records, reviews, args)
    rows = []
    for order, record in enumerate(selected):
        rid = record["record_id"]
        review = reviews.get(rid) or {}
        objects = record.get("objects") or []
        rows.append(
            {
                "order": order,
                "record_id": rid,
                "scene": scene_key(record),
                "source_scene": (record.get("provenance", {}).get("source_paths") or [None])[0],
                "mesh_a": objects[0].get("mesh_or_proxy") if len(objects) > 0 else None,
                "mesh_b": objects[1].get("mesh_or_proxy") if len(objects) > 1 else None,
                "target_image": str(args.dataset_root / record.get("render_artifacts", {}).get("target_image", "")),
                "mask_coverage": record.get("validation", {}).get("mask_coverage"),
                "proxy_review_status": review.get("status", "missing"),
                "proxy_failure_tags": review.get("failure_tags") or [],
            }
        )

    status_counts: dict[str, int] = defaultdict(int)
    scene_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[row["proxy_review_status"]] += 1
        scene_counts[row["scene"]] += 1

    manifest = {
        "schema": "selected_universal_dual_vlm_inner_loop_manifest_v1",
        "created_at_unix": time.time(),
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "selection_strategy": {
            "total": args.total,
            "per_scene": args.per_scene,
            "needs_vggt_proxy_cases_per_scene_target": args.needs_per_scene,
            "notes": [
                "Stratified by DataEvolver scene.",
                "Within each scene, prioritizes proxy geometry-review failures and higher-numbered meshes.",
                "The VLM inner loop itself uses Qwen3.6-27B via DataEvolver stage5_5_vlm_review.py.",
            ],
        },
        "runtime": {
            "dataevolver_root": str(args.dataevolver_root),
            "blender": str(args.blender),
            "vlm_model_path": str(args.vlm_model_path),
            "max_rounds": args.max_rounds,
            "threshold": args.threshold,
            "engine": args.engine,
            "resolution": args.resolution,
        },
        "counts": {
            "selected": len(rows),
            "by_scene": dict(sorted(scene_counts.items())),
            "by_proxy_review_status": dict(sorted(status_counts.items())),
        },
        "records": rows,
        "jsonl": str(jsonl_path),
    }
    save_json(manifest_path, manifest)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return manifest_path


def ensure_selection(args: argparse.Namespace) -> Path:
    manifest_path = args.selection_manifest or (args.output_root / "metadata" / "selected_universal_manifest.json")
    if manifest_path.exists():
        return manifest_path
    records = load_jsonl(args.dataset_root / "metadata" / "records.jsonl")
    reviews = load_reviews(args.dataset_root)
    return write_selection_manifest(records, reviews, args)


def bottom_position(obj: dict) -> list[float]:
    bbox = obj.get("bbox_3d") or {}
    center = [float(v) for v in bbox.get("center_xyz", [0.0, 0.0, 0.0])]
    size = [float(v) for v in bbox.get("size_xyz", [0.0, 0.0, 0.0])]
    return [center[0], center[1], center[2] - size[2] / 2.0 - 0.005]


def make_placement_state(record: dict) -> dict:
    objects = record.get("objects") or []
    pos_a = bottom_position(objects[0])
    pos_b = bottom_position(objects[1])
    dx = pos_b[0] - pos_a[0]
    dy = pos_b[1] - pos_a[1]
    camera_azimuth = math.radians(float(record.get("camera", {}).get("pose", {}).get("azimuth_deg", 0.0)))
    return {
        "schema": "dual_placement_state_v1",
        "source": "universal_contract_record_bbox",
        "source_record_id": record["record_id"],
        "seed": record_index(record),
        "template": "fixed_from_universal_contract",
        "obj1_position": [round(v, 6) for v in pos_a],
        "obj2_position": [round(v, 6) for v in pos_b],
        "separation": round(float(math.sqrt(dx * dx + dy * dy)), 6),
        "height_offset": round(float(pos_b[2] - pos_a[2]), 6),
        "side": 0,
        "camera_azimuth": round(float(camera_azimuth), 10),
    }


def make_initial_state(record: dict, default_control_state) -> dict:
    state = deepcopy(default_control_state())
    camera = record.get("camera") or {}
    pose = camera.get("pose") or {}
    intrinsics = camera.get("intrinsics") or {}
    state.setdefault("camera", {})
    state["camera"]["elevation_deg"] = float(pose.get("elevation_deg", state["camera"].get("elevation_deg", 20.0)))
    state["camera"]["focal_length_mm"] = float(intrinsics.get("focal_length_mm", state["camera"].get("focal_length_mm", 50.0)))
    state["camera"]["distance_scale"] = 1.0
    state["camera"]["framing_margin"] = 1.0
    state["camera"]["azimuth_offset_deg"] = 0.0
    return state


def coerce_score(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def has_scene_hard_fail(result: dict, hard_fail_issues: set[str]) -> bool:
    tags = {str(tag) for tag in result.get("issue_tags") or []}
    return bool(tags & hard_fail_issues)


def should_stop_stagnant(rounds: list[dict], actions: list[str], best_score: float | None) -> bool:
    if len(rounds) < 3 or best_score is None:
        return False
    recent_scores = [coerce_score(r.get("score")) for r in rounds[-3:]]
    if any(score is None for score in recent_scores):
        return False
    if max(recent_scores) < best_score + 1e-6 and best_score - recent_scores[-1] >= 0.04:
        return True
    contact_loop = {"CONTACT_SHADOW_UP", "AO_UP", "A_LOWER_TINY", "B_LOWER_TINY"}
    return set(actions or []).issubset(contact_loop) and len(set(actions or [])) >= 2


def write_decision(pair_dir: Path, round_idx: int, result: dict, threshold: float, passed: bool, best_score: float | None) -> None:
    save_json(
        pair_dir / f"gate_decision_round{round_idx:02d}.json",
        {
            "round_idx": round_idx,
            "passed": bool(passed),
            "score": result.get("hybrid_score"),
            "threshold": threshold,
            "best_score_after_round": best_score,
            "issue_tags": result.get("issue_tags"),
            "suggested_actions": result.get("suggested_actions"),
            "state_out": result.get("state_out"),
            "render_dir": result.get("render_dir"),
            "review_agg_path": result.get("review_agg_path"),
        },
    )


def run_one_record(record: dict, args: argparse.Namespace, modules: dict[str, Any], shard_tag: str) -> dict:
    actions_mod = modules["actions"]
    step_mod = modules["step"]
    pair_dir = args.output_root / record["record_id"]
    pair_dir.mkdir(parents=True, exist_ok=True)
    status_path = pair_dir / "gate_status.json"
    if args.resume:
        existing = load_json(status_path, default=None)
        if isinstance(existing, dict) and existing.get("status") in {"accepted", "rejected", "error"}:
            return {**existing, "resume_skip": True}

    objects = record.get("objects") or []
    if len(objects) != 2:
        status = {"status": "error", "reason": "record_does_not_have_two_objects", "sample_id": record.get("record_id")}
        save_json(status_path, status)
        return status
    source_paths = record.get("provenance", {}).get("source_paths") or []
    scene = source_paths[0] if source_paths else None
    mesh_a = objects[0].get("mesh_or_proxy")
    mesh_b = objects[1].get("mesh_or_proxy")
    if not scene or not mesh_a or not mesh_b:
        status = {"status": "error", "reason": "missing_scene_or_mesh_path", "sample_id": record.get("record_id")}
        save_json(status_path, status)
        return status

    save_json(pair_dir / "source_record.json", record)
    placement_state_path = pair_dir / "placement_state.json"
    initial_state_path = pair_dir / "states" / "round00_universal_input.json"
    save_json(placement_state_path, make_placement_state(record))
    save_json(initial_state_path, make_initial_state(record, actions_mod.default_control_state))

    best_score = None
    best_round_idx = None
    best_result = None
    no_improve = 0
    rounds: list[dict] = []
    actions: list[str] = []
    state_in: str | None = str(initial_state_path)
    prev_rgb: str | None = None
    started = time.time()

    try:
        for round_idx in range(int(args.max_rounds)):
            step_args = SimpleNamespace(
                output_dir=str(pair_dir),
                sample_id=record["record_id"],
                obj_a_id=objects[0].get("category", "obj_a"),
                obj_b_id=objects[1].get("category", "obj_b"),
                mesh_a=mesh_a,
                mesh_b=mesh_b,
                ref_a=None,
                ref_b=None,
                scene=scene,
                round_idx=round_idx,
                actions=actions,
                step_scale=args.step_scale,
                control_state_in=state_in,
                placement_state=str(placement_state_path),
                placement_state_out=None,
                prev_rgb=prev_rgb,
                device=args.device,
                blender=str(args.blender),
                resolution=args.resolution,
                engine=args.engine,
                template=None,
                seed=record_index(record),
            )
            log_path = pair_dir / "logs" / f"{shard_tag}_round{round_idx:02d}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result = step_mod.run_step(step_args)

            score = coerce_score(result.get("hybrid_score"))
            improved = score is not None and (best_score is None or score > best_score + 1e-6)
            if improved:
                best_score = score
                best_round_idx = round_idx
                best_result = result
                no_improve = 0
            else:
                no_improve += 1
            passed = bool(result.get("success")) and score is not None and score >= float(args.threshold)
            rounds.append(
                {
                    "round_idx": round_idx,
                    "score": score,
                    "threshold": float(args.threshold),
                    "passed": passed,
                    "issue_tags": result.get("issue_tags"),
                    "suggested_actions": result.get("suggested_actions"),
                    "actions_in": actions,
                    "state_out": result.get("state_out"),
                    "render_dir": result.get("render_dir"),
                    "round_log": str(log_path),
                }
            )
            write_decision(pair_dir, round_idx, result, float(args.threshold), passed, best_score)

            if passed:
                status = {
                    "status": "accepted",
                    "sample_id": record["record_id"],
                    "accepted_round_idx": round_idx,
                    "accepted_score": score,
                    "accepted_threshold": float(args.threshold),
                    "accepted_render_dir": result.get("render_dir"),
                    "accepted_state_out": result.get("state_out"),
                    "best_score": best_score,
                    "best_round_idx": best_round_idx,
                    "rounds": rounds,
                    "placement_state": str(placement_state_path),
                    "source_dataset_record": str(pair_dir / "source_record.json"),
                    "elapsed_seconds": round(time.time() - started, 3),
                }
                save_json(status_path, status)
                return status

            if has_scene_hard_fail(result, actions_mod.SCENE_HARD_FAIL_ISSUES):
                status = {
                    "status": "rejected",
                    "reason": "scene_hard_fail",
                    "sample_id": record["record_id"],
                    "best_score": best_score,
                    "best_round_idx": best_round_idx,
                    "hard_fail_tags": result.get("issue_tags"),
                    "rounds": rounds,
                    "placement_state": str(placement_state_path),
                    "source_dataset_record": str(pair_dir / "source_record.json"),
                    "elapsed_seconds": round(time.time() - started, 3),
                }
                save_json(status_path, status)
                return status

            review = load_json(result.get("review_agg_path") or "", default={}) or {}
            actions = actions_mod.choose_actions(review)
            state_in = result.get("state_out")
            prev_rgb = result.get("rgb_path")
            if best_result and score is not None and best_score is not None:
                if best_score - score >= float(args.rollback_score_drop_threshold):
                    state_in = best_result.get("state_out")
                    prev_rgb = best_result.get("rgb_path")
            if should_stop_stagnant(rounds, actions, best_score):
                status = {
                    "status": "rejected",
                    "reason": "stagnant_after_best_round",
                    "sample_id": record["record_id"],
                    "best_score": best_score,
                    "best_round_idx": best_round_idx,
                    "next_actions": actions,
                    "rounds": rounds,
                    "placement_state": str(placement_state_path),
                    "source_dataset_record": str(pair_dir / "source_record.json"),
                    "elapsed_seconds": round(time.time() - started, 3),
                }
                save_json(status_path, status)
                return status
            if args.patience > 0 and no_improve >= int(args.patience):
                break

        status = {
            "status": "rejected",
            "reason": "max_rounds_without_passing_gate",
            "sample_id": record["record_id"],
            "best_score": best_score,
            "best_round_idx": best_round_idx,
            "best_render_dir": best_result.get("render_dir") if best_result else None,
            "best_state_out": best_result.get("state_out") if best_result else None,
            "last_score": rounds[-1].get("score") if rounds else None,
            "last_threshold": float(args.threshold),
            "rounds": rounds,
            "placement_state": str(placement_state_path),
            "source_dataset_record": str(pair_dir / "source_record.json"),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        save_json(status_path, status)
        return status
    except Exception as exc:
        status = {
            "status": "error",
            "reason": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "sample_id": record.get("record_id"),
            "rounds": rounds,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        save_json(status_path, status)
        return status


def import_dataevolver_modules(dataevolver_root: Path) -> dict[str, Any]:
    if str(dataevolver_root) not in sys.path:
        sys.path.insert(0, str(dataevolver_root))
    from pipeline.dual_vlm_gate import dual_actions, dual_agent_step

    return {"actions": dual_actions, "step": dual_agent_step}


def write_worker_progress(args: argparse.Namespace, shard_records: list[dict], completed_statuses: list[dict], started_at: float) -> None:
    counts: dict[str, int] = defaultdict(int)
    for item in completed_statuses:
        counts[item.get("status", "unknown")] += 1
    payload = {
        "schema": "selected_universal_vlm_inner_loop_progress_v1",
        "updated_at_unix": time.time(),
        "output_root": str(args.output_root),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "pid": os.getpid(),
        "device": args.device,
        "engine": args.engine,
        "resolution": args.resolution,
        "max_rounds": args.max_rounds,
        "assigned": len(shard_records),
        "completed": len(completed_statuses),
        "remaining": max(0, len(shard_records) - len(completed_statuses)),
        "counts": dict(sorted(counts.items())),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    save_json(args.output_root / "metadata" / f"progress_shard{args.shard_index}.json", payload)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, num_shards)")

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "metadata").mkdir(parents=True, exist_ok=True)

    manifest_path = ensure_selection(args)
    manifest = load_json(manifest_path, default={}) or {}
    rows = manifest.get("records") or []
    record_ids = [row["record_id"] for row in rows]
    records_by_id = {record["record_id"]: record for record in load_jsonl(args.dataset_root / "metadata" / "records.jsonl")}
    selected_records = [records_by_id[rid] for rid in record_ids if rid in records_by_id]
    shard_records = [rec for idx, rec in enumerate(selected_records) if idx % args.num_shards == args.shard_index]

    runtime_manifest = {
        "schema": "selected_universal_dual_vlm_inner_loop_runtime_v1",
        "selection_manifest": str(manifest_path),
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "dataevolver_root": str(args.dataevolver_root),
        "blender": str(args.blender),
        "vlm_model_path": str(args.vlm_model_path),
        "vlm_system": "DataEvolver stage5_5_vlm_review.py / Qwen3.6-27B",
        "max_rounds": args.max_rounds,
        "threshold": args.threshold,
        "engine": args.engine,
        "resolution": args.resolution,
        "num_shards": args.num_shards,
    }
    save_json(args.output_root / "metadata" / "runtime_manifest.json", runtime_manifest)

    if args.prepare_only:
        print(json.dumps({"prepared": True, "manifest": str(manifest_path), "selected": len(selected_records)}, ensure_ascii=False))
        return

    modules = import_dataevolver_modules(args.dataevolver_root)
    started_at = time.time()
    completed_statuses: list[dict] = []
    shard_tag = f"shard{args.shard_index}"
    print(
        json.dumps(
            {
                "event": "worker_start",
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "assigned": len(shard_records),
                "device": args.device,
                "pid": os.getpid(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    write_worker_progress(args, shard_records, completed_statuses, started_at)

    for local_idx, record in enumerate(shard_records):
        sample_started = time.time()
        print(
            json.dumps(
                {
                    "event": "sample_start",
                    "shard_index": args.shard_index,
                    "local_index": local_idx,
                    "assigned": len(shard_records),
                    "record_id": record["record_id"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        status = run_one_record(record, args, modules, shard_tag)
        status_summary = {
            "record_id": record["record_id"],
            "status": status.get("status"),
            "reason": status.get("reason"),
            "best_score": status.get("best_score"),
            "accepted_score": status.get("accepted_score"),
            "accepted_round_idx": status.get("accepted_round_idx"),
            "elapsed_seconds": status.get("elapsed_seconds"),
            "resume_skip": status.get("resume_skip", False),
        }
        completed_statuses.append(status)
        append_jsonl(args.output_root / "metadata" / f"worker_shard{args.shard_index}_events.jsonl", status_summary)
        write_worker_progress(args, shard_records, completed_statuses, started_at)
        print(
            json.dumps(
                {
                    "event": "sample_done",
                    "shard_index": args.shard_index,
                    "record_id": record["record_id"],
                    "status": status.get("status"),
                    "best_score": status.get("best_score"),
                    "elapsed_seconds": round(time.time() - sample_started, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_worker_progress(args, shard_records, completed_statuses, started_at)
    print(
        json.dumps(
            {
                "event": "worker_done",
                "shard_index": args.shard_index,
                "completed": len(completed_statuses),
                "elapsed_seconds": round(time.time() - started_at, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
