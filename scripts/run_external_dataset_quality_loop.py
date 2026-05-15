#!/usr/bin/env python3
"""Run the external dataset-quality loop end-to-end.

Loop stages:
1. Generate new Stage1 seed concepts (optional AI)
2. Build Stage1->Stage3 assets into an isolated sandbox root
3. Run VLM quality gate on candidate scene pairs
4. Keep only accepted pairs
5. Export rotation8 renders from accepted base states
6. Build train-ready dataset
7. Build object-disjoint split

This script focuses on data quality before training. It does not launch LoRA
training directly; the output is a higher-quality dataset root ready for the
next training round.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent

SEED_GEN_SCRIPT = REPO_ROOT / "scripts" / "stage1_generate_ai_seed_concepts.py"
ASSET_BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_scene_assets_from_stage1.py"
GATE_SCRIPT = REPO_ROOT / "scripts" / "run_vlm_quality_gate_loop.py"
ACCEPTED_ROOT_SCRIPT = REPO_ROOT / "scripts" / "build_accepted_pair_root_from_gate.py"
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export_rotation8_from_best_object_state.py"
TRAINREADY_SCRIPT = REPO_ROOT / "scripts" / "build_rotation8_trainready_dataset.py"
SPLIT_SCRIPT = REPO_ROOT / "scripts" / "build_object_split_for_rotation_dataset.py"
FEEDBACK_SCRIPT_DIR = REPO_ROOT / "scripts" / "feedback_loop"
COMPARE_SCRIPT = FEEDBACK_SCRIPT_DIR / "compare.py"
ANALYZE_FEEDBACK_SCRIPT = FEEDBACK_SCRIPT_DIR / "analyze_feedback_for_dataset.py"
PINNED_SPLIT_SCRIPT = FEEDBACK_SCRIPT_DIR / "build_pinned_split.py"
AUGMENT_DATASET_SCRIPT = FEEDBACK_SCRIPT_DIR / "build_augmented_rotation_dataset.py"
VALIDATE_DATASET_SCRIPT = FEEDBACK_SCRIPT_DIR / "validate_dataset.py"
DEFAULT_GATE_PROFILE = REPO_ROOT / "configs" / "dataset_profiles" / "scene_v7.json"
STAGE3_SCRIPT = REPO_ROOT / "pipeline" / "stage3_image_to_3d.py"
TEXTURE_REPAIR_REASONS = {
    "missing_image_texture",
    "missing_material_gate_metadata",
    "missing_has_image_texture_flag",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external dataset quality loop")
    parser.add_argument("--loop-root", required=True)
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--blender", dest="blender_bin", default="blender")
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--seed-start-index", type=int, default=51)
    parser.add_argument("--seed-model", default="claude-haiku-4-5")
    parser.add_argument("--seed-api-provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--seed-api-base-url", default=None)
    parser.add_argument("--seed-api-group", default=None)
    parser.add_argument("--seed-api-timeout", type=float, default=300)
    parser.add_argument("--seed-template-only", action="store_true")
    parser.add_argument(
        "--research-prior-path",
        default=None,
        help="Optional research_prior.json on this machine. Stage1 treats it as soft guidance and warns/continues on failure.",
    )
    parser.add_argument("--t2i-device", default="cuda:0")
    parser.add_argument("--sam-device", default="cuda:0")
    parser.add_argument("--i23d-devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--gate-rotations", default="0")
    parser.add_argument("--threshold-start", type=float, default=0.58)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--threshold-max", type=float, default=0.68)
    parser.add_argument("--score-field", choices=["hybrid_score", "vlm_only_score"], default="hybrid_score")
    parser.add_argument("--min-vlm-only-score", type=float, default=0.50)
    parser.add_argument("--gate-max-rounds", type=int, default=6)
    parser.add_argument("--gate-step-scale", type=float, default=0.25)
    parser.add_argument("--gate-patience", type=int, default=3)
    parser.add_argument("--gate-rollback-score-drop-threshold", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--texture-repair-attempts",
        type=int,
        default=1,
        help="How many times to retry Stage3 + re-gate for rejects caused by missing texture metadata or missing image texture.",
    )
    parser.add_argument("--train-objects", type=int, default=7)
    parser.add_argument("--val-objects", type=int, default=1)
    parser.add_argument("--test-objects", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--asset-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--stage1-mode", choices=["api", "template-only", "dry-run"], default=None)
    parser.add_argument("--baseline-split-root", default=None, help="Existing split dataset to augment; original split is pinned")
    parser.add_argument("--augment-baseline", action="store_true", help="Merge accepted new train pairs into --baseline-split-root")
    parser.add_argument("--feedback-plan", default=None, help="Existing dataset_feedback_plan.json with weak_target_rotations")
    parser.add_argument("--baseline-eval-dir", default=None, help="Baseline eval dir for compare.py")
    parser.add_argument("--current-eval-dir", default=None, help="Current eval dir for compare.py; enables feedback-plan generation")
    parser.add_argument("--baseline-mode", default=None)
    parser.add_argument("--current-mode", default=None)
    parser.add_argument("--new-object-budget", type=int, default=None)
    parser.add_argument("--max-weak-angles", type=int, default=4)
    parser.add_argument("--cost-cap-hours", type=float, default=12.0)
    parser.add_argument("--use-plan-seed-count", action="store_true")
    parser.add_argument("--export-rotations", default=None, help="Override exported rotations; default uses 0 plus weak rotations when available")
    parser.add_argument("--target-rotations", default=None, help="Override train target rotations; default uses weak rotations when available")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_loop_gate_profile(output_path: Path, assets_root: Path, base_profile_path: Path) -> dict:
    base_profile = load_json(base_profile_path, default=None)
    if not isinstance(base_profile, dict):
        raise FileNotFoundError(f"Invalid gate profile: {base_profile_path}")

    profile = dict(base_profile)
    profile["reference_images_dir"] = str((assets_root / "images").resolve())
    save_json(output_path, profile)
    return profile


def comma_ints(values) -> str:
    out = []
    for value in values or []:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item not in out:
            out.append(item)
    return ",".join(str(item) for item in out)


def parse_cuda_devices(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def shard_ids(obj_ids: list[str], devices: list[str]) -> list[dict]:
    buckets = [{"device": device, "obj_ids": []} for device in devices]
    for idx, obj_id in enumerate(obj_ids):
        buckets[idx % len(devices)]["obj_ids"].append(obj_id)
    return [bucket for bucket in buckets if bucket["obj_ids"]]


def run_step(cmd: list[str], name: str, cwd: Path, log_dir: Path, step_index: int) -> dict:
    started = time.time()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{step_index:02d}_{name}.log"
    tail = deque(maxlen=400)
    print(f"\n=== [{step_index:02d}] {name} ===")
    print(" ".join(str(item) for item in cmd))
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            tail.append(line)
        return_code = proc.wait()
    return {
        "name": name,
        "command": cmd,
        "return_code": return_code,
        "elapsed_seconds": round(time.time() - started, 3),
        "log_path": str(log_path),
        "stdout_tail": "".join(tail)[-8000:],
    }


def append_step(steps: list[dict], cmd: list[str], name: str, cwd: Path, log_dir: Path) -> dict:
    record = run_step(cmd, name, cwd, log_dir, len(steps) + 1)
    steps.append(record)
    return record


def clear_mesh_outputs(assets_root: Path, obj_ids: list[str]) -> None:
    for subdir_name in ("meshes_raw", "meshes"):
        subdir = assets_root / subdir_name
        if not subdir.exists():
            continue
        for obj_id in obj_ids:
            path = subdir / f"{obj_id}.glb"
            if path.exists() or path.is_symlink():
                path.unlink()


def copy_repaired_meshes(assets_root: Path, obj_ids: list[str]) -> None:
    meshes_raw_dir = assets_root / "meshes_raw"
    meshes_dir = assets_root / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    for obj_id in obj_ids:
        src = meshes_raw_dir / f"{obj_id}.glb"
        if src.exists():
            shutil.copy2(src, meshes_dir / src.name)


def collect_texture_rejects(gate_root: Path) -> tuple[list[str], list[dict]]:
    manifest = load_json(gate_root / "gate_manifest.json", default={}) or {}
    rejected = manifest.get("rejected") or []
    obj_ids: list[str] = []
    details: list[dict] = []
    for item in rejected:
        if not isinstance(item, dict):
            continue
        obj_id = str(item.get("obj_id") or "").strip()
        if not obj_id:
            continue
        material_reason = (
            item.get("last_material_gate_reason")
            or item.get("material_gate_reason")
            or (((item.get("rounds") or [{}])[-1] or {}).get("material_gate_reason"))
        )
        quality = item.get("mesh_material_quality")
        if not isinstance(quality, dict):
            quality = (((item.get("rounds") or [{}])[-1] or {}).get("mesh_material_quality")) or {}
        has_image_texture = quality.get("has_image_texture") if isinstance(quality, dict) else None
        if material_reason in TEXTURE_REPAIR_REASONS or has_image_texture is False:
            if obj_id not in obj_ids:
                obj_ids.append(obj_id)
            details.append(
                {
                    "obj_id": obj_id,
                    "pair_dir": item.get("pair_dir"),
                    "material_gate_reason": material_reason,
                    "has_image_texture": has_image_texture,
                    "status": item.get("status"),
                    "reject_reason": item.get("reason"),
                }
            )
    return sorted(obj_ids), details


def run_texture_repair_attempt(
    *,
    args: argparse.Namespace,
    assets_root: Path,
    steps: list[dict],
    logs_dir: Path,
    obj_ids: list[str],
    attempt_idx: int,
) -> list[dict]:
    if not obj_ids:
        return []
    clear_mesh_outputs(assets_root, obj_ids)
    devices = parse_cuda_devices(args.i23d_devices) or ["cuda:0"]
    records = []
    for shard in shard_ids(obj_ids, devices):
        cmd = [
            args.python_bin,
            str(STAGE3_SCRIPT),
            "--device",
            shard["device"],
            "--no-skip",
            "--prompts-path",
            str(assets_root / "prompts.json"),
            "--images-dir",
            str(assets_root / "images_rgba"),
            "--rgb-images-dir",
            str(assets_root / "images"),
            "--output-dir",
            str(assets_root / "meshes_raw"),
            "--ids",
            ",".join(shard["obj_ids"]),
        ]
        record = append_step(
            steps,
            cmd,
            f"repair_stage3_textures_attempt{attempt_idx:02d}_{str(shard['device']).replace(':', '_')}",
            REPO_ROOT,
            logs_dir,
        )
        records.append(record)
    if all(record["return_code"] == 0 for record in records):
        copy_repaired_meshes(assets_root, obj_ids)
    return records


def fail(summary_path: Path, state_path: Path, payload: dict) -> None:
    save_json(summary_path, payload)
    save_json(state_path, {**payload, "schema_version": "external_dataset_quality_loop_v3", "status": "failed"})
    raise SystemExit(1)


def generate_feedback_plan(args: argparse.Namespace, loop_root: Path, steps: list[dict], logs_dir: Path) -> dict:
    feedback_dir = loop_root / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)

    if args.feedback_plan:
        plan_path = Path(args.feedback_plan).resolve()
        plan = load_json(plan_path, default={}) or {}
        plan["_plan_path"] = str(plan_path)
        return plan

    if not (args.current_eval_dir and args.baseline_eval_dir):
        return {}

    compare_report = feedback_dir / "compare_report.json"
    plan_path = feedback_dir / "dataset_feedback_plan.json"
    compare_cmd = [
        args.python_bin,
        str(COMPARE_SCRIPT),
        "--current",
        str(Path(args.current_eval_dir).resolve()),
        "--baseline",
        str(Path(args.baseline_eval_dir).resolve()),
        "--output",
        str(compare_report),
    ]
    if args.current_mode:
        compare_cmd.extend(["--current-mode", args.current_mode])
    if args.baseline_mode:
        compare_cmd.extend(["--baseline-mode", args.baseline_mode])
    record = append_step(steps, compare_cmd, "compare_eval_feedback", REPO_ROOT, logs_dir)
    if record["return_code"] != 0:
        raise RuntimeError(f"compare.py failed; see {record['log_path']}")

    analyze_cmd = [
        args.python_bin,
        str(ANALYZE_FEEDBACK_SCRIPT),
        "--compare-report",
        str(compare_report),
        "--output",
        str(plan_path),
        "--new-object-budget",
        str(args.new_object_budget or args.seed_count),
        "--max-weak-angles",
        str(args.max_weak_angles),
        "--cost-cap-hours",
        str(args.cost_cap_hours),
    ]
    record = append_step(steps, analyze_cmd, "analyze_dataset_feedback", REPO_ROOT, logs_dir)
    if record["return_code"] != 0:
        raise RuntimeError(f"analyze_feedback_for_dataset.py failed; see {record['log_path']}")

    plan = load_json(plan_path, default={}) or {}
    plan["_plan_path"] = str(plan_path)
    plan["_compare_report_path"] = str(compare_report)
    return plan


def run_checked(steps: list[dict], cmd: list[str], name: str, summary_path: Path, state_path: Path, payload_base: dict, logs_dir: Path) -> None:
    record = append_step(steps, cmd, name, REPO_ROOT, logs_dir)
    if record["return_code"] != 0:
        fail(summary_path, state_path, {**payload_base, "success": False, "failed_step": name, "steps": steps})


def main() -> None:
    args = parse_args()
    loop_root = Path(args.loop_root).resolve()
    loop_root.mkdir(parents=True, exist_ok=True)

    seeds_file = loop_root / "seed_concepts.json"
    assets_root = loop_root / "stage1_assets"
    gate_root = loop_root / "gate_loop"
    accepted_root = loop_root / "accepted_pairs"
    export_root = loop_root / "rotation8_consistent"
    trainready_root = loop_root / "trainready"
    split_root = loop_root / "trainready_split"
    augmented_root = loop_root / "augmented_trainready_split"
    pinned_split_path = loop_root / "pinned_split.json"
    validation_report = loop_root / "validation_report.json"
    summary_path = loop_root / "external_quality_loop_summary.json"
    state_path = loop_root / "LOOP_STATE.json"
    logs_dir = loop_root / "logs"
    gate_profile_path = loop_root / "scene_v7_gate_profile.json"

    steps = []
    payload_base = {
        "schema_version": "external_dataset_quality_loop_v3",
        "loop_root": str(loop_root),
        "seed_concepts_file": str(seeds_file),
        "research_prior_path": args.research_prior_path,
        "stage1_assets_root": str(assets_root),
        "gate_root": str(gate_root),
        "accepted_pair_root": str(accepted_root),
        "rotation8_export_root": str(export_root),
        "trainready_root": str(trainready_root),
        "split_root": str(split_root),
        "augmented_root": str(augmented_root),
        "pinned_split_path": str(pinned_split_path),
        "validation_report": str(validation_report),
        "gate_profile_path": str(gate_profile_path),
        "gate_step_scale": float(args.gate_step_scale),
        "gate_patience": int(args.gate_patience),
        "gate_rollback_score_drop_threshold": float(args.gate_rollback_score_drop_threshold),
        "texture_repair_attempts": int(args.texture_repair_attempts),
    }
    texture_repair_history: list[dict] = []

    try:
        feedback_plan = generate_feedback_plan(args, loop_root, steps, logs_dir)
    except RuntimeError as exc:
        fail(summary_path, state_path, {**payload_base, "success": False, "error": str(exc), "steps": steps})

    weak_target_rotations = [
        int(value)
        for value in (feedback_plan.get("weak_target_rotations") or [])
        if int(value) != 0
    ]
    if args.use_plan_seed_count and feedback_plan.get("new_object_budget"):
        args.seed_count = int(feedback_plan["new_object_budget"])

    export_rotations = args.export_rotations
    if not export_rotations and weak_target_rotations:
        export_rotations = comma_ints([0, *weak_target_rotations])
    target_rotations = args.target_rotations
    if not target_rotations and weak_target_rotations:
        target_rotations = comma_ints(weak_target_rotations)

    augment_enabled = bool(args.augment_baseline or args.baseline_split_root)
    if augment_enabled and not args.baseline_split_root:
        fail(
            summary_path,
            state_path,
            {
                **payload_base,
                "success": False,
                "error": "--augment-baseline requires --baseline-split-root",
                "steps": steps,
            },
        )

    save_json(
        state_path,
        {
            **payload_base,
            "status": "running",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "feedback_plan": feedback_plan,
            "weak_target_rotations": weak_target_rotations,
            "export_rotations": export_rotations,
            "target_rotations": target_rotations,
            "augment_baseline": augment_enabled,
            "baseline_split_root": str(Path(args.baseline_split_root).resolve()) if args.baseline_split_root else None,
            "texture_repair_history": texture_repair_history,
            "steps": steps,
        },
    )

    seed_cmd = [
        args.python_bin,
        str(SEED_GEN_SCRIPT),
        "--output-file",
        str(seeds_file),
        "--count",
        str(args.seed_count),
        "--start-index",
        str(args.seed_start_index),
        "--model",
        args.seed_model,
    ]
    if args.seed_api_provider:
        seed_cmd.extend(["--api-provider", args.seed_api_provider])
    if args.seed_api_base_url:
        seed_cmd.extend(["--api-base-url", args.seed_api_base_url])
    if args.seed_api_group:
        seed_cmd.extend(["--api-group", args.seed_api_group])
    seed_cmd.extend(["--api-timeout", str(args.seed_api_timeout)])
    if args.seed_template_only:
        seed_cmd.append("--template-only")
    if args.research_prior_path:
        seed_cmd.extend(["--research-prior-path", args.research_prior_path])
    run_checked(steps, seed_cmd, "generate_seed_concepts", summary_path, state_path, payload_base, logs_dir)

    stage1_mode = args.stage1_mode or ("template-only" if args.seed_template_only else "api")
    asset_cmd = [
        args.python_bin,
        str(ASSET_BUILD_SCRIPT),
        "--objects-file",
        str(seeds_file),
        "--output-root",
        str(assets_root),
        "--python",
        args.python_bin,
        "--stage1-mode",
        stage1_mode,
        "--stage1-model",
        args.seed_model,
        "--t2i-device",
        args.t2i_device,
        "--sam-device",
        args.sam_device,
        "--i23d-devices",
        args.i23d_devices,
        "--no-skip",
    ]
    if args.research_prior_path:
        asset_cmd.extend(["--research-prior-path", args.research_prior_path])
    if stage1_mode == "api" and args.seed_api_provider == "anthropic":
        asset_cmd.extend(["--stage1-api-provider", "anthropic"])
    if stage1_mode == "api" and args.seed_api_base_url:
        asset_cmd.extend(["--stage1-api-base-url", args.seed_api_base_url])
        asset_cmd.extend(["--stage1-api-timeout", str(args.seed_api_timeout)])
    run_checked(steps, asset_cmd, "build_stage1_assets", summary_path, state_path, payload_base, logs_dir)

    build_loop_gate_profile(
        output_path=gate_profile_path,
        assets_root=assets_root,
        base_profile_path=DEFAULT_GATE_PROFILE,
    )

    gate_cmd = [
        args.python_bin,
        str(GATE_SCRIPT),
        "--root-dir",
        str(gate_root),
        "--objects-file",
        str(seeds_file),
        "--rotations",
        args.gate_rotations,
        "--threshold-start",
        str(args.threshold_start),
        "--threshold-step",
        str(args.threshold_step),
        "--threshold-max",
        str(args.threshold_max),
        "--score-field",
        args.score_field,
        "--max-rounds",
        str(args.gate_max_rounds),
        "--min-vlm-only-score",
        str(args.min_vlm_only_score),
        "--step-scale",
        str(args.gate_step_scale),
        "--patience",
        str(args.gate_patience),
        "--rollback-score-drop-threshold",
        str(args.gate_rollback_score_drop_threshold),
        "--device",
        args.device,
        "--blender",
        args.blender_bin,
        "--meshes-dir",
        str(assets_root / "meshes"),
        "--profile",
        str(gate_profile_path),
        "--reject-abandon",
    ]
    if args.resume:
        gate_cmd.append("--resume")
    run_checked(steps, gate_cmd, "run_vlm_quality_gate", summary_path, state_path, payload_base, logs_dir)

    for attempt_idx in range(1, max(0, int(args.texture_repair_attempts)) + 1):
        texture_reject_ids, texture_reject_details = collect_texture_rejects(gate_root)
        if not texture_reject_ids:
            break
        repair_records = run_texture_repair_attempt(
            args=args,
            assets_root=assets_root,
            steps=steps,
            logs_dir=logs_dir,
            obj_ids=texture_reject_ids,
            attempt_idx=attempt_idx,
        )
        if not repair_records or not all(record["return_code"] == 0 for record in repair_records):
            fail(
                summary_path,
                state_path,
                {
                    **payload_base,
                    "success": False,
                    "failed_step": f"repair_stage3_textures_attempt{attempt_idx:02d}",
                    "texture_repair_history": texture_repair_history
                    + [
                        {
                            "attempt_idx": attempt_idx,
                            "obj_ids": texture_reject_ids,
                            "rejected_details": texture_reject_details,
                            "step_records": repair_records,
                        }
                    ],
                    "steps": steps,
                },
            )
        if gate_root.exists():
            shutil.rmtree(gate_root)
        rerun_name = f"rerun_vlm_quality_gate_after_texture_repair_{attempt_idx:02d}"
        run_checked(steps, gate_cmd, rerun_name, summary_path, state_path, payload_base, logs_dir)
        remaining_ids, remaining_details = collect_texture_rejects(gate_root)
        texture_repair_history.append(
            {
                "attempt_idx": attempt_idx,
                "obj_ids": texture_reject_ids,
                "rejected_details": texture_reject_details,
                "step_records": repair_records,
                "remaining_texture_reject_ids": remaining_ids,
                "remaining_texture_reject_details": remaining_details,
            }
        )

    accepted_cmd = [
        args.python_bin,
        str(ACCEPTED_ROOT_SCRIPT),
        "--gate-root",
        str(gate_root),
        "--output-dir",
        str(accepted_root),
        "--asset-mode",
        args.asset_mode,
        "--overwrite",
    ]
    run_checked(steps, accepted_cmd, "build_accepted_pair_root", summary_path, state_path, payload_base, logs_dir)

    export_cmd = [
        args.python_bin,
        str(EXPORT_SCRIPT),
        "--source-root",
        str(accepted_root),
        "--output-dir",
        str(export_root),
        "--python",
        args.python_bin,
        "--blender",
        args.blender_bin,
        "--meshes-dir",
        str(assets_root / "meshes"),
        "--gpus",
        "0",
        "--fallback-to-best-any-angle",
    ]
    if export_rotations:
        export_cmd.extend(["--rotations", export_rotations])
    run_checked(steps, export_cmd, "export_rotation8_from_accepted", summary_path, state_path, payload_base, logs_dir)

    trainready_cmd = [
        args.python_bin,
        str(TRAINREADY_SCRIPT),
        "--source-root",
        str(export_root),
        "--output-dir",
        str(trainready_root),
        "--copy-mode",
        args.asset_mode,
    ]
    if target_rotations:
        trainready_cmd.extend(["--target-rotations", target_rotations])
    run_checked(steps, trainready_cmd, "build_trainready", summary_path, state_path, payload_base, logs_dir)

    final_dataset_root = split_root
    if augment_enabled:
        baseline_split_root = Path(args.baseline_split_root).resolve()
        pinned_cmd = [
            args.python_bin,
            str(PINNED_SPLIT_SCRIPT),
            "--dataset-root",
            str(baseline_split_root),
            "--output",
            str(pinned_split_path),
            "--source-note",
            "external_dataset_quality_loop_v3 baseline split",
        ]
        run_checked(steps, pinned_cmd, "build_pinned_split", summary_path, state_path, payload_base, logs_dir)

        augment_cmd = [
            args.python_bin,
            str(AUGMENT_DATASET_SCRIPT),
            "--baseline-split-root",
            str(baseline_split_root),
            "--new-trainready-root",
            str(trainready_root),
            "--output-dir",
            str(augmented_root),
            "--pinned-split-path",
            str(pinned_split_path),
            "--asset-mode",
            args.asset_mode,
        ]
        if target_rotations:
            augment_cmd.extend(["--target-rotations", target_rotations])
        run_checked(steps, augment_cmd, "build_augmented_dataset", summary_path, state_path, payload_base, logs_dir)
        final_dataset_root = augmented_root
    else:
        split_cmd = [
            args.python_bin,
            str(SPLIT_SCRIPT),
            "--source-root",
            str(trainready_root),
            "--output-dir",
            str(split_root),
            "--train-objects",
            str(args.train_objects),
            "--val-objects",
            str(args.val_objects),
            "--test-objects",
            str(args.test_objects),
            "--seed",
            str(args.split_seed),
            "--asset-mode",
            args.asset_mode,
        ]
        run_checked(steps, split_cmd, "build_split", summary_path, state_path, payload_base, logs_dir)

    if not args.skip_validation:
        validate_cmd = [
            args.python_bin,
            str(VALIDATE_DATASET_SCRIPT),
            str(final_dataset_root),
            "--output",
            str(validation_report),
        ]
        if augment_enabled:
            validate_cmd.extend([
                "--pinned-split-path",
                str(pinned_split_path),
                "--expected-manifest",
                str(augmented_root / "augmented_manifest.json"),
            ])
        run_checked(steps, validate_cmd, "validate_final_dataset", summary_path, state_path, payload_base, logs_dir)

    payload = {
        **payload_base,
        "success": True,
        "status": "completed",
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feedback_plan": feedback_plan,
        "weak_target_rotations": weak_target_rotations,
        "export_rotations": export_rotations,
        "target_rotations": target_rotations,
        "augment_baseline": augment_enabled,
        "baseline_split_root": str(Path(args.baseline_split_root).resolve()) if args.baseline_split_root else None,
        "texture_repair_history": texture_repair_history,
        "final_dataset_root": str(final_dataset_root),
        "steps": steps,
    }
    save_json(summary_path, payload)
    save_json(state_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
