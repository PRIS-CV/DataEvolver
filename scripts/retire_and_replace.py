#!/usr/bin/env python3
"""
Retire underperforming objects and optionally generate replacements.

Two modes:
  1. analyze: Read compare_report → identify objects to retire → write retirement manifest
  2. generate: Read retirement manifest → generate replacement assets via full pipeline

Usage:
    # Analyze only (called by run_full_pipeline.py in feedback phase):
    python scripts/feedback_loop/retire_and_replace.py \
        --compare-report feedback/compare_report.json \
        --output-dir feedback/retirement \
        --psnr-floor 14.0 \
        --max-retire 5

    # Generate replacements (called by run_full_pipeline.py in improvement phase):
    python scripts/feedback_loop/retire_and_replace.py \
        --mode generate \
        --retire-manifest feedback/retirement/retirement_manifest.json \
        --output-dir improvements/replacements \
        --python /path/to/python \
        --blender /path/to/blender
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="Retire bad objects and generate replacements")
    p.add_argument("--mode", choices=["analyze", "generate"], default="analyze")

    g = p.add_argument_group("Analyze mode")
    g.add_argument("--compare-report", default=None)
    g.add_argument("--psnr-floor", type=float, default=14.0,
                   help="Objects with overall PSNR below this are retired unconditionally")
    g.add_argument("--max-retire", type=int, default=5)
    g.add_argument("--feedback-state", default=None,
                   help="FEEDBACK_STATE.json for cross-round tracking")
    g.add_argument("--min-rounds-as-bad", type=int, default=1,
                   help="Consecutive rounds as bottom-20%% before retirement")

    g = p.add_argument_group("Generate mode")
    g.add_argument("--retire-manifest", default=None)
    g.add_argument("--python", default=sys.executable)
    g.add_argument("--blender", default="/aaaidata/zhangqisong/blender-4.24/blender")
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--i23d-devices", default="cuda:0,cuda:1,cuda:2")
    g.add_argument("--seed-model", default="claude-opus-4-6")
    g.add_argument("--api-provider", default="anthropic")
    g.add_argument("--api-base-url", default="https://dragoncode.codes/v1")
    g.add_argument("--template-only", action="store_true")

    g = p.add_argument_group("Common")
    g.add_argument("--output-dir", required=True)
    return p.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Analyze Mode ─────────────────────────────────────────────────────────────

def analyze_retirement(args) -> dict:
    """Identify objects to retire based on eval metrics."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.compare_report or not Path(args.compare_report).exists():
        print("[retire] No compare report, nothing to analyze")
        manifest = {"retired_objects": [], "reason": "no_compare_report"}
        save_json(output_dir / "retirement_manifest.json", manifest)
        return manifest

    report = load_json(Path(args.compare_report))

    sources = report.get("sources", {})
    source_key = "spatialedit" if "spatialedit" in sources else next(iter(sources), None)
    if not source_key:
        print("[retire] No metric source in compare report")
        manifest = {"retired_objects": [], "reason": "no_metric_source"}
        save_json(output_dir / "retirement_manifest.json", manifest)
        return manifest

    source = sources[source_key]

    # Support both source-level (compare.py) and diagnostics-level layouts
    all_angle_bad_raw = source.get("all_angle_bad_objects",
                                   source.get("diagnostics", {}).get("all_angle_bad_objects", []))
    all_angle_bad = []
    for item in all_angle_bad_raw:
        if isinstance(item, dict):
            all_angle_bad.append(item.get("object_id", item.get("obj_id", str(item))))
        else:
            all_angle_bad.append(str(item))

    per_object = source.get("per_object",
                            source.get("baseline", {}).get("per_object", {}))

    # Cross-round tracking
    history = {}
    if args.feedback_state and Path(args.feedback_state).exists():
        fs = load_json(Path(args.feedback_state))
        history = fs.get("bad_object_history", {})

    candidates = []

    for obj_id in all_angle_bad:
        obj_metrics = per_object.get(obj_id, {})
        obj_psnr = obj_metrics.get("psnr", 999)

        prev = history.get(obj_id, {})
        rounds_as_bad = len(prev.get("rounds_as_bad", [])) + 1

        reason = None
        if obj_psnr < args.psnr_floor:
            reason = "psnr_floor"
        elif rounds_as_bad >= args.min_rounds_as_bad:
            reason = "persistent_bad"

        if reason:
            candidates.append({
                "obj_id": obj_id,
                "reason": reason,
                "psnr": round(obj_psnr, 4),
                "rounds_as_bad": rounds_as_bad,
            })

    candidates.sort(key=lambda x: x["psnr"])
    retired = candidates[:args.max_retire]
    skipped = candidates[args.max_retire:]

    manifest = {
        "schema": "retirement_manifest_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "psnr_floor": args.psnr_floor,
        "min_rounds_as_bad": args.min_rounds_as_bad,
        "max_retire": args.max_retire,
        "all_angle_bad_count": len(all_angle_bad),
        "candidates_count": len(candidates),
        "retired_objects": retired,
        "skipped_objects": skipped,
    }

    save_json(output_dir / "retirement_manifest.json", manifest)

    for obj in retired:
        print(f"  [RETIRE] {obj['obj_id']}: psnr={obj['psnr']}, reason={obj['reason']}")
    if skipped:
        for obj in skipped:
            print(f"  [SKIP]   {obj['obj_id']}: psnr={obj['psnr']} (over max_retire)")

    print(f"[retire] {len(retired)} objects retired, {len(skipped)} skipped (cap={args.max_retire})")
    return manifest


# ── Generate Mode ────────────────────────────────────────────────────────────

def generate_replacements(args) -> dict:
    """Generate replacement assets for retired objects."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.retire_manifest or not Path(args.retire_manifest).exists():
        print("[replace] No retirement manifest")
        return {"replacements": []}

    manifest = load_json(Path(args.retire_manifest))
    retired = manifest.get("retired_objects", [])
    if not retired:
        print("[replace] No objects to replace")
        return {"replacements": []}

    retire_ids = [obj["obj_id"] for obj in retired]
    count = len(retire_ids)

    # Determine next available obj_id
    existing_ids = set()
    for obj in retired:
        oid = obj["obj_id"]
        try:
            num = int(oid.replace("obj_", ""))
            existing_ids.add(num)
        except ValueError:
            pass
    start_index = max(existing_ids, default=1100) + 100

    # Generate seed concepts
    seed_file = output_dir / "replacement_seeds.json"
    seed_cmd = [
        args.python,
        str(REPO_ROOT / "scripts" / "stage1_generate_ai_seed_concepts.py"),
        "--output-file", str(seed_file),
        "--count", str(count),
        "--start-index", str(start_index),
    ]
    if args.template_only:
        seed_cmd += ["--template-only"]
    else:
        seed_cmd += [
            "--model", args.seed_model,
            "--api-provider", args.api_provider,
            "--api-base-url", args.api_base_url,
        ]

    print(f"[replace] Generating {count} replacement concepts starting at obj_{start_index}")
    rc = subprocess.run(seed_cmd).returncode
    if rc != 0:
        raise RuntimeError(f"Seed concept generation failed (rc={rc})")

    # Build assets (T2I → SAM2 → 3D)
    assets_root = output_dir / "assets"
    build_cmd = [
        args.python,
        str(REPO_ROOT / "scripts" / "build_scene_assets_from_stage1.py"),
        "--objects-file", str(seed_file),
        "--output-root", str(assets_root),
        "--python", args.python,
        "--t2i-device", args.device,
        "--sam-device", args.device,
        "--i23d-devices", args.i23d_devices,
        "--no-skip",
    ]
    if args.template_only:
        build_cmd += ["--stage1-mode", "template-only"]
    else:
        build_cmd += [
            "--stage1-mode", "api",
            "--stage1-model", args.seed_model,
            "--stage1-api-provider", args.api_provider,
            "--stage1-api-base-url", args.api_base_url,
        ]

    print(f"[replace] Building replacement assets...")
    rc = subprocess.run(build_cmd).returncode
    if rc != 0:
        raise RuntimeError(f"Asset build failed (rc={rc})")

    # VLM gate (quick, 2 rounds max)
    gate_root = output_dir / "gate"
    gate_profile = REPO_ROOT / "configs" / "dataset_profiles" / "scene_v7.json"
    gate_cmd = [
        args.python,
        str(REPO_ROOT / "scripts" / "run_vlm_quality_gate_loop.py"),
        "--root-dir", str(gate_root),
        "--objects-file", str(seed_file),
        "--rotations", "0",
        "--threshold-start", "0.50",
        "--threshold-max", "0.60",
        "--max-rounds", "3",
        "--patience", "0",
        "--device", args.device,
        "--blender", args.blender,
        "--meshes-dir", str(assets_root / "meshes"),
        "--profile", str(gate_profile),
    ]

    print(f"[replace] Running VLM quality gate on replacements...")
    subprocess.run(gate_cmd)

    # Export rotation8
    accepted_root = output_dir / "accepted"
    subprocess.run([
        args.python,
        str(REPO_ROOT / "scripts" / "build_accepted_pair_root_from_gate.py"),
        "--gate-root", str(gate_root),
        "--output-dir", str(accepted_root),
        "--asset-mode", "symlink",
        "--overwrite",
    ])

    rot8_root = output_dir / "rotation8"
    subprocess.run([
        args.python,
        str(REPO_ROOT / "scripts" / "export_rotation8_from_best_object_state.py"),
        "--source-root", str(accepted_root),
        "--output-dir", str(rot8_root),
        "--python", args.python,
        "--blender", args.blender,
        "--meshes-dir", str(assets_root / "meshes"),
        "--gpus", "0",
        "--fallback-to-best-any-angle",
    ])

    # Build trainready
    trainready_root = output_dir / "trainready"
    subprocess.run([
        args.python,
        str(REPO_ROOT / "scripts" / "build_rotation8_trainready_dataset.py"),
        "--source-root", str(rot8_root),
        "--output-dir", str(trainready_root),
        "--copy-mode", "symlink",
        "--target-rotations", "45,90,135,180,225,270,315",
    ])

    # Build replacement map
    replacement_map = {}
    seeds = load_json(seed_file) if seed_file.exists() else []
    for i, obj in enumerate(retired):
        if i < len(seeds):
            replacement_map[obj["obj_id"]] = seeds[i]["id"]

    result = {
        "schema": "replacement_result_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "retired_objects": retire_ids,
        "replacement_map": replacement_map,
        "replacement_trainready": str(trainready_root),
        "replacement_assets": str(assets_root),
    }
    save_json(output_dir / "replacement_result.json", result)

    print(f"[replace] Generated {len(replacement_map)} replacements")
    for old_id, new_id in replacement_map.items():
        print(f"  {old_id} → {new_id}")

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.mode == "analyze":
        analyze_retirement(args)
    elif args.mode == "generate":
        generate_replacements(args)


if __name__ == "__main__":
    main()
