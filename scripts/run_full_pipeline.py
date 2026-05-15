#!/usr/bin/env python3
"""
全自动 pipeline：资产生成 → VLM 评测 → 反馈改进

将整条数据集构建流水线串成单个脚本，在 tmux 里启动后无需人工干预。

Usage:
    tmux new -s full-pipeline
    export ANTHROPIC_API_KEY="sk-ant-..."
    python ~/ARIS/scripts/run_full_pipeline.py \
        --seed-count 18 \
        --max-feedback-rounds 1 \
        2>&1 | tee ~/full_pipeline.log

    # 从已有数据集开始（跳过资产生成）：
    python ~/ARIS/scripts/run_full_pipeline.py \
        --existing-dataset /path/to/split_dataset \
        --max-feedback-rounds 1

Steps:
    Phase A-D: seed concepts → T2I → SAM2 → 3D → VLM gate → rotation8 → trainready → split
    Phase E:   VLM evaluation of dataset target renders
    Phase F:   analyze feedback → rerender weak angles → retire bad objects → augmented dataset
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "python": "python",
    "blender": "blender",
    "runtime_root": "<runtime-root>",
    "baseline_split": str(REPO_ROOT / "pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410"),
    "gate_profile": str(REPO_ROOT / "configs/dataset_profiles/scene_v7.json"),
    "scene_template": str(REPO_ROOT / "configs/scene_template.json"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Full automated dataset pipeline")

    g = p.add_argument_group("Global")
    g.add_argument("--runtime-root", default=DEFAULTS["runtime_root"])
    g.add_argument("--python", default=DEFAULTS["python"])
    g.add_argument("--blender", default=DEFAULTS["blender"])
    g.add_argument("--dry-run", action="store_true")

    g = p.add_argument_group("Phase A-D: Asset Generation")
    g.add_argument("--existing-dataset", default=None,
                   help="Skip asset generation, use this split dataset root")
    g.add_argument("--seed-count", type=int, default=18)
    g.add_argument("--seed-start-index", type=int, default=None)
    g.add_argument("--seed-model", default="claude-opus-4-6")
    g.add_argument("--api-provider", default="anthropic")
    g.add_argument("--api-base-url", default="https://dragoncode.codes/v1")
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--i23d-devices", default="cuda:0,cuda:1,cuda:2")
    g.add_argument("--export-gpus", default="0,1,2")
    g.add_argument("--gate-max-rounds", type=int, default=5)
    g.add_argument("--gate-threshold", type=float, default=0.78)
    g.add_argument("--template-only", action="store_true")

    g = p.add_argument_group("Phase E: VLM Evaluation")
    g.add_argument("--vlm-eval-split", default="train",
                   help="Which split to VLM-evaluate (default: train)")
    g.add_argument("--vlm-eval-device", default=None,
                   help="Device for VLM eval (defaults to --device)")
    g.add_argument("--vlm-review-mode", default="scene_insert",
                   choices=["scene_insert", "studio"])
    g.add_argument("--skip-vlm-eval", action="store_true",
                   help="Skip VLM evaluation, go straight to done")

    g = p.add_argument_group("Phase F: Feedback")
    g.add_argument("--max-feedback-rounds", type=int, default=1)
    g.add_argument("--retire-score-floor", type=float, default=0.40,
                   help="VLM score floor for retirement (default: 0.40)")
    g.add_argument("--retire-max-per-round", type=int, default=5)
    g.add_argument("--rerender-weak-angles", action="store_true", default=True)
    g.add_argument("--no-rerender-weak-angles", dest="rerender_weak_angles",
                   action="store_false")

    g = p.add_argument_group("Augmentation")
    g.add_argument("--baseline-split", default=DEFAULTS["baseline_split"])
    g.add_argument("--augment-baseline", action="store_true", default=True)
    g.add_argument("--no-augment-baseline", dest="augment_baseline",
                   action="store_false")

    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_step(name: str, cmd: list, log_dir: Path, env=None, dry_run=False,
             timeout=None) -> int:
    log(f"=== {name} ===")
    log(f"CMD: {' '.join(str(c) for c in cmd)}")
    log_file = log_dir / f"{name}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        log(f"[dry-run] skipped")
        return 0

    t0 = time.time()
    with log_file.open("w", encoding="utf-8") as lf:
        proc = subprocess.run(
            [str(c) for c in cmd],
            stdout=lf, stderr=subprocess.STDOUT,
            env=env, timeout=timeout,
        )
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    log(f"  {status} ({elapsed:.0f}s) → {log_file}")

    if proc.returncode != 0:
        with log_file.open("r") as f:
            tail = f.readlines()[-30:]
        log(f"  Last 30 lines:\n{''.join(tail)}")

    return proc.returncode


def run_checked(name: str, cmd: list, log_dir: Path, env=None,
                dry_run=False, timeout=None):
    rc = run_step(name, cmd, log_dir, env=env, dry_run=dry_run, timeout=timeout)
    if rc != 0 and not dry_run:
        raise RuntimeError(f"Step {name} failed with rc={rc}")


# ── Phase A-D: Asset Generation ─────────────────────────────────────────────

def run_phase_ad(args, state: dict) -> str:
    """Run asset pipeline, return trainready split dataset root."""
    run_root = Path(state["run_root"])
    log_dir = run_root / "logs"

    launch_script = REPO_ROOT / "scripts" / "launch_new_asset_force_rot8.sh"

    env = os.environ.copy()
    seed_start = args.seed_start_index or ""

    cmd = [
        "bash", str(launch_script),
        "--device", args.device,
        "--i23d-devices", args.i23d_devices,
        "--export-gpus", args.export_gpus,
        "--seed-count", str(args.seed_count),
        "--seed-model", args.seed_model,
        "--api-provider", args.api_provider,
        "--api-base-url", args.api_base_url,
        "--gate-max-rounds", str(args.gate_max_rounds),
        "--gate-threshold", str(args.gate_threshold),
        "--no-augment",
    ]
    if seed_start:
        cmd += ["--seed-start-index", str(seed_start)]
    if args.template_only:
        cmd += ["--template-only"]
    if args.dry_run:
        cmd += ["--dry-run"]

    run_checked("phase_ad_asset_pipeline", cmd, log_dir, env=env,
                dry_run=args.dry_run, timeout=86400)

    asset_run = _find_latest_asset_run(Path(args.runtime_root))
    if not asset_run:
        raise RuntimeError("Cannot find asset run output")

    loop_state = load_json(asset_run / "LOOP_STATE.json")
    trainready_root = loop_state.get("trainready_root", "")
    if not trainready_root or not Path(trainready_root).exists():
        raise RuntimeError(f"trainready_root not found: {trainready_root}")

    state["asset_run_root"] = str(asset_run)
    state["asset_trainready"] = trainready_root
    state["asset_meshes_dir"] = str(
        Path(loop_state.get("stage1_assets_root", "")) / "meshes")
    state["asset_rotation8_root"] = loop_state.get(
        "rotation8_enhanced_root",
        loop_state.get("rotation8_export_root", ""))

    split_root = run_root / "initial_split"
    trainready_manifest = Path(trainready_root) / "manifest.json"
    if trainready_manifest.exists():
        n_objects = load_json(trainready_manifest).get("total_objects", 0)
    else:
        n_objects = len(list((Path(trainready_root) / "objects").iterdir()))
    n_val = max(1, round(n_objects * 0.15))
    n_test = max(1, round(n_objects * 0.15))
    n_train = max(1, n_objects - n_val - n_test)

    run_checked("phase_ad_build_split", [
        args.python,
        str(REPO_ROOT / "scripts" / "build_object_split_for_rotation_dataset.py"),
        "--source-root", trainready_root,
        "--output-dir", str(split_root),
        "--train-objects", str(n_train),
        "--val-objects", str(n_val),
        "--test-objects", str(n_test),
        "--seed", "42",
        "--asset-mode", "symlink",
    ], log_dir, dry_run=args.dry_run)

    if args.augment_baseline and Path(args.baseline_split).exists():
        augmented_root = run_root / "augmented_split"
        pinned_split = run_root / "pinned_split.json"

        run_checked("phase_ad_pinned_split", [
            args.python,
            str(REPO_ROOT / "scripts/feedback_loop/build_pinned_split.py"),
            "--dataset-root", args.baseline_split,
            "--output", str(pinned_split),
        ], log_dir, dry_run=args.dry_run)

        run_checked("phase_ad_augment", [
            args.python,
            str(REPO_ROOT / "scripts/feedback_loop/build_augmented_rotation_dataset.py"),
            "--baseline-split-root", args.baseline_split,
            "--new-trainready-root", trainready_root,
            "--output-dir", str(augmented_root),
            "--pinned-split-path", str(pinned_split),
            "--asset-mode", "symlink",
        ], log_dir, dry_run=args.dry_run)

        state["initial_dataset"] = str(augmented_root)
        return str(augmented_root)

    state["initial_dataset"] = str(split_root)
    return str(split_root)


def _find_latest_asset_run(runtime_root: Path) -> Path | None:
    candidates = sorted(
        runtime_root.glob("new_asset_force_rot8_*/LOOP_STATE.json"),
        key=lambda p: p.parent.name,
    )
    if candidates:
        return candidates[-1].parent
    return None


# ── Phase E: VLM Evaluation ─────────────────────────────────────────────────

def run_vlm_evaluation(args, state: dict, dataset_root: str,
                       round_idx: int = 0) -> str:
    """Run VLM review on dataset target renders, return report path."""
    run_root = Path(state["run_root"])
    log_dir = run_root / "logs"
    vlm_eval_dir = run_root / f"round_{round_idx}" / "vlm_eval"
    vlm_report = vlm_eval_dir / "vlm_eval_report.json"

    eval_device = args.vlm_eval_device or args.device

    cmd = [
        args.python,
        str(REPO_ROOT / "scripts/feedback_loop/eval_dataset_vlm.py"),
        "--dataset-root", dataset_root,
        "--output", str(vlm_report),
        "--device", eval_device,
        "--split", args.vlm_eval_split,
        "--review-mode", args.vlm_review_mode,
    ]

    gate_profile = DEFAULTS.get("gate_profile")
    if gate_profile and Path(gate_profile).exists():
        cmd += ["--profile", gate_profile]

    run_checked(f"vlm_evaluation_round{round_idx}", cmd, log_dir,
                dry_run=args.dry_run, timeout=43200)

    state["vlm_eval_report"] = str(vlm_report)
    return str(vlm_report)


# ── Phase F: Feedback Analysis ───────────────────────────────────────────────

def run_feedback_analysis(args, state: dict, vlm_report_path: str,
                          round_idx: int) -> dict:
    """Run feedback analysis on VLM eval report, return feedback plan."""
    run_root = Path(state["run_root"])
    log_dir = run_root / "logs"
    feedback_dir = run_root / f"round_{round_idx}" / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)

    feedback_plan_path = feedback_dir / "dataset_feedback_plan.json"
    analyze_cmd = [
        args.python,
        str(REPO_ROOT / "scripts/feedback_loop/analyze_feedback_for_dataset.py"),
        "--compare-report", vlm_report_path,
        "--output", str(feedback_plan_path),
    ]

    run_checked(f"feedback_analyze_round{round_idx}", analyze_cmd,
                log_dir, dry_run=args.dry_run)

    plan = load_json(feedback_plan_path) if feedback_plan_path.exists() else {}

    retire_cmd = [
        args.python,
        str(REPO_ROOT / "scripts/feedback_loop/retire_and_replace.py"),
        "--compare-report", vlm_report_path,
        "--output-dir", str(feedback_dir / "retirement"),
        "--psnr-floor", str(args.retire_score_floor),
        "--max-retire", str(args.retire_max_per_round),
    ]

    run_checked(f"feedback_retire_round{round_idx}", retire_cmd,
                log_dir, dry_run=args.dry_run)

    retirement_manifest = feedback_dir / "retirement" / "retirement_manifest.json"
    if retirement_manifest.exists():
        plan["retirement"] = load_json(retirement_manifest)

    state.setdefault("feedback_rounds", []).append({
        "round": round_idx,
        "feedback_dir": str(feedback_dir),
        "vlm_report": vlm_report_path,
        "feedback_plan": str(feedback_plan_path),
    })

    return plan


# ── Phase F: Improvements ───────────────────────────────────────────────────

def run_improvements(args, state: dict, feedback: dict, current_dataset: str,
                     round_idx: int) -> str:
    """Execute improvements, return improved dataset root."""
    run_root = Path(state["run_root"])
    log_dir = run_root / "logs"
    improvement_dir = run_root / f"round_{round_idx}" / "improvements"
    improvement_dir.mkdir(parents=True, exist_ok=True)

    rot8_root = state.get("asset_rotation8_root", "")
    meshes_dir = state.get("asset_meshes_dir", "")
    new_trainready = None

    # 1. Rerender weak angles
    if args.rerender_weak_angles and rot8_root and meshes_dir:
        weak_angles = feedback.get("weak_target_rotations", [])
        if weak_angles:
            relit_root = improvement_dir / "rotation8_relit"
            rerender_cmd = [
                args.python,
                str(REPO_ROOT / "scripts" / "rerender_weak_angles.py"),
                "--source-root", rot8_root,
                "--output-dir", str(relit_root),
                "--meshes-dir", meshes_dir,
                "--blender", args.blender,
                "--scene-template", DEFAULTS["scene_template"],
                "--gpus", args.export_gpus,
            ]
            run_checked(f"rerender_round{round_idx}", rerender_cmd,
                        log_dir, dry_run=args.dry_run)

            relit_trainready = improvement_dir / "relit_trainready"
            target_rots = ",".join(
                str(a) for a in [45, 90, 135, 180, 225, 270, 315])
            run_checked(f"relit_trainready_round{round_idx}", [
                args.python,
                str(REPO_ROOT / "scripts/build_rotation8_trainready_dataset.py"),
                "--source-root", str(relit_root),
                "--output-dir", str(relit_trainready),
                "--copy-mode", "symlink",
                "--target-rotations", target_rots,
            ], log_dir, dry_run=args.dry_run)

            new_trainready = str(relit_trainready)
            state["relit_trainready"] = new_trainready

    # 2. Retire bad objects
    retirement = feedback.get("retirement", {})
    retired_objects = retirement.get("retired_objects", [])
    retire_ids = [obj["obj_id"] for obj in retired_objects] if retired_objects else []

    # 3. Generate replacement assets (if any retired)
    if retire_ids:
        log(f"Generating replacements for {len(retire_ids)} retired objects: "
            f"{retire_ids}")
        replacement_dir = improvement_dir / "replacements"
        retire_manifest = (
            run_root / f"round_{round_idx}" / "feedback"
            / "retirement" / "retirement_manifest.json"
        )
        replace_cmd = [
            args.python,
            str(REPO_ROOT / "scripts/feedback_loop/retire_and_replace.py"),
            "--mode", "generate",
            "--retire-manifest", str(retire_manifest),
            "--output-dir", str(replacement_dir),
            "--python", args.python,
            "--blender", args.blender,
            "--device", args.device,
            "--i23d-devices", args.i23d_devices,
            "--seed-model", args.seed_model,
            "--api-provider", args.api_provider,
            "--api-base-url", args.api_base_url,
        ]
        if args.template_only:
            replace_cmd += ["--template-only"]

        rc = run_step(f"generate_replacements_round{round_idx}", replace_cmd,
                      log_dir, dry_run=args.dry_run, timeout=43200)
        if rc != 0:
            log("[WARN] Replacement generation failed, "
                "proceeding without replacements")
            retire_ids = []

    # 4. Build improved augmented dataset
    improved_root = run_root / f"round_{round_idx + 1}" / "dataset"
    pinned_split = run_root / "pinned_split.json"

    if not pinned_split.exists():
        run_checked(f"pinned_split_round{round_idx}", [
            args.python,
            str(REPO_ROOT / "scripts/feedback_loop/build_pinned_split.py"),
            "--dataset-root", current_dataset,
            "--output", str(pinned_split),
        ], log_dir, dry_run=args.dry_run)

    augment_cmd = [
        args.python,
        str(REPO_ROOT / "scripts/feedback_loop/build_augmented_rotation_dataset.py"),
        "--baseline-split-root", current_dataset,
        "--output-dir", str(improved_root),
        "--pinned-split-path", str(pinned_split),
        "--asset-mode", "symlink",
    ]
    if new_trainready:
        augment_cmd += ["--new-trainready-root", new_trainready]
    if retire_ids:
        retire_file = improvement_dir / "retire_ids.json"
        save_json(retire_file, retire_ids)
        augment_cmd += ["--retire-objects", str(retire_file)]

    run_checked(f"augment_round{round_idx}", augment_cmd,
                log_dir, dry_run=args.dry_run)

    run_checked(f"validate_round{round_idx}", [
        args.python,
        str(REPO_ROOT / "scripts/feedback_loop/validate_dataset.py"),
        str(improved_root),
        "--output", str(improved_root / "validation_report.json"),
        "--pinned-split-path", str(pinned_split),
    ], log_dir, dry_run=args.dry_run)

    state["improved_dataset"] = str(improved_root)
    return str(improved_root)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.runtime_root) / f"full_pipeline_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    state = {
        "schema": "full_pipeline_v2",
        "run_root": str(run_root),
        "timestamp": timestamp,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
    }
    state_path = run_root / "PIPELINE_STATE.json"
    save_json(state_path, state)

    log("=" * 70)
    log("Full Automated Dataset Pipeline (VLM eval, no training)")
    log(f"  Run root:   {run_root}")
    log(f"  Dry run:    {args.dry_run}")
    log(f"  Feedback:   {args.max_feedback_rounds} round(s)")
    log(f"  VLM eval:   split={args.vlm_eval_split}, "
        f"mode={args.vlm_review_mode}")
    log("=" * 70)

    t0 = time.time()

    try:
        # Phase A-D: Asset generation
        if args.existing_dataset:
            dataset_root = args.existing_dataset
            log(f"Using existing dataset: {dataset_root}")
            state["initial_dataset"] = dataset_root
        else:
            log("Phase A-D: Building assets and dataset...")
            dataset_root = run_phase_ad(args, state)
            log(f"Phase A-D complete: {dataset_root}")

        save_json(state_path, state)

        # Phase E+F: VLM evaluation → feedback → improve (each round)
        if args.skip_vlm_eval:
            log("Skipping VLM evaluation (--skip-vlm-eval)")
        else:
            for fb_round in range(args.max_feedback_rounds):
                log(f"Round {fb_round}: VLM evaluation of dataset...")
                vlm_report = run_vlm_evaluation(args, state, dataset_root,
                                                fb_round)
                log(f"Round {fb_round}: VLM eval complete: {vlm_report}")
                save_json(state_path, state)

                log(f"Round {fb_round}: Feedback analysis...")
                feedback = run_feedback_analysis(args, state, vlm_report,
                                                 fb_round)
                save_json(state_path, state)

                log(f"Round {fb_round}: Applying improvements...")
                improved_dataset = run_improvements(args, state, feedback,
                                                    dataset_root, fb_round)
                log(f"Round {fb_round}: Improvements complete: "
                    f"{improved_dataset}")
                save_json(state_path, state)

                dataset_root = improved_dataset

        # Done
        elapsed = time.time() - t0
        state["status"] = "completed"
        state["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state["elapsed_seconds"] = round(elapsed, 1)
        state["elapsed_hours"] = round(elapsed / 3600, 1)
        state["final_dataset"] = dataset_root
        save_json(state_path, state)

        log("")
        log("=" * 70)
        log("Pipeline completed!")
        log(f"  Total time:       {elapsed/3600:.1f} hours")
        log(f"  Final dataset:    {dataset_root}")
        log(f"  State:            {state_path}")
        log(f"  Feedback rounds:  {len(state.get('feedback_rounds', []))}")
        log("=" * 70)

    except Exception as exc:
        elapsed = time.time() - t0
        state["status"] = "failed"
        state["error"] = str(exc)
        state["elapsed_seconds"] = round(elapsed, 1)
        save_json(state_path, state)
        log(f"Pipeline FAILED after {elapsed/3600:.1f}h: {exc}")
        raise


if __name__ == "__main__":
    main()
