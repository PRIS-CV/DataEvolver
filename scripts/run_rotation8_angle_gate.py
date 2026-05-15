"""
Per-angle quality gate for rotation8 exports.

After rotation8 export, evaluates each angle of each object with VLM.
Weak angles get per-angle enhancement: VLM-guided control_state adjustments
+ Blender re-render, keeping whichever version scores higher.

Inserted between rotation8 export (Step 5) and trainready build (Step 6).

Usage:
    python scripts/run_rotation8_angle_gate.py \
        --source-root /path/to/rotation8_consistent \
        --output-dir /path/to/rotation8_enhanced \
        --meshes-dir /path/to/meshes \
        --blender /path/to/blender \
        --profile scene_v7.json \
        --device cuda:0 \
        --angle-threshold 0.50 \
        --max-enhance-rounds 2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess as _sp
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

VLM_WEIGHTS = {
    "lighting": 0.25,
    "object_integrity": 0.30,
    "composition": 0.20,
    "render_quality_semantic": 0.10,
    "overall": 0.15,
}

YAW_ACTIONS_BLACKLIST = {"O_YAW_CW", "O_YAW_CCW", "O_YAW_CW_BIG", "O_YAW_CCW_BIG"}


def parse_args():
    p = argparse.ArgumentParser(description="Rotation8 per-angle quality gate")
    p.add_argument("--source-root", required=True, help="rotation8_consistent root")
    p.add_argument("--output-dir", required=True, help="Enhanced rotation8 output root")
    p.add_argument("--meshes-dir", required=True, help="Stage1 meshes directory")
    p.add_argument("--blender", required=True, help="Blender binary path")
    p.add_argument("--profile", required=True, help="Gate profile JSON path")
    p.add_argument("--scene-template", default=None,
                   help="Scene template (default: source-root/scene_template_fixed_camera.json)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpus", default=None,
                   help="Comma-separated GPU indices for multi-GPU parallel (e.g. 0,1,2)")
    p.add_argument("--ids", default=None,
                   help="Comma-separated object IDs (worker mode, used internally)")
    p.add_argument("--angle-threshold", type=float, default=0.50,
                   help="Minimum VLM score per angle to pass")
    p.add_argument("--max-enhance-rounds", type=int, default=2,
                   help="Max enhancement rounds per weak angle")
    p.add_argument("--step-scale", type=float, default=0.25)
    p.add_argument("--copy-mode", choices=["copy", "symlink"], default="symlink",
                   help="How to materialize unchanged files")
    return p.parse_args()


def _compute_vlm_score(scores: dict) -> float:
    return sum(VLM_WEIGHTS[k] * (scores.get(k, 3) - 1) / 4.0 for k in VLM_WEIGHTS)


def _rot_slug(deg: int) -> str:
    return f"yaw{deg:03d}"


def _copy_or_link(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


# ── Phase 1: Evaluate ────────────────────────────────────────────────────────

def evaluate_all_angles(
    source_root: Path,
    objects: dict,
    device: str,
    profile: dict,
    angle_threshold: float,
) -> Tuple[Dict[str, Dict[int, dict]], List[tuple]]:
    from stage5_5_vlm_review import run_vlm_review

    angle_scores: Dict[str, Dict[int, dict]] = {}
    weak_angles: List[tuple] = []

    obj_ids = sorted(objects.keys())
    total_angles = 0

    for obj_idx, obj_id in enumerate(obj_ids, 1):
        obj_dir = source_root / "objects" / obj_id
        obj_manifest = json.loads((obj_dir / "object_manifest.json").read_text())
        rotations = obj_manifest.get("rotations", [])
        obj_scores: Dict[int, dict] = {}

        for rot_info in rotations:
            deg = rot_info["rotation_deg"]
            rgb_path = source_root / rot_info["rgb_path"]
            control_path = source_root / rot_info["control_state_path"]
            total_angles += 1

            if not rgb_path.exists():
                print(f"  [WARN] {obj_id} {_rot_slug(deg)}: RGB missing, skip")
                continue

            review = run_vlm_review(
                rgb_path=str(rgb_path),
                sample_id=f"{obj_id}_{_rot_slug(deg)}_anglegate",
                round_idx=0,
                active_group="object",
                az=deg, el=0,
                device=device,
                review_mode=profile.get("review_mode", "scene_insert"),
                enable_thinking=profile.get("enable_vlm_thinking", True),
                prompt_appendix=profile.get("prompt_appendix", ""),
            )

            vlm_score = _compute_vlm_score(review.get("scores", {}))
            info = {
                "vlm_score": vlm_score,
                "issue_tags": review.get("issue_tags", []),
                "suggested_actions": review.get("suggested_actions", []),
                "lighting_diagnosis": review.get("lighting_diagnosis", "unknown"),
            }
            obj_scores[deg] = info

            tag = "WEAK" if vlm_score < angle_threshold else "ok"
            print(f"  [{obj_idx}/{len(obj_ids)}] {obj_id} {_rot_slug(deg)}: "
                  f"vlm={vlm_score:.3f} [{tag}]  issues={info['issue_tags']}")

            if vlm_score < angle_threshold:
                weak_angles.append((
                    obj_id, deg, vlm_score,
                    info["suggested_actions"],
                    str(control_path),
                ))

        angle_scores[obj_id] = obj_scores

    print(f"\n[angle_gate] Evaluated {total_angles} angles across {len(obj_ids)} objects")
    print(f"[angle_gate] Weak angles: {len(weak_angles)} / {total_angles} "
          f"(threshold={angle_threshold})")

    return angle_scores, weak_angles


# ── Phase 2: Enhance ─────────────────────────────────────────────────────────

def enhance_weak_angles(
    weak_angles: List[tuple],
    work_root: Path,
    meshes_dir: Path,
    blender: str,
    profile_path: Path,
    scene_template: str,
    device: str,
    step_scale: float,
    max_rounds: int,
    angle_threshold: float,
) -> Dict[Tuple[str, int], dict]:
    from run_scene_agent_step import run_agent_step

    enhanced: Dict[Tuple[str, int], dict] = {}

    for idx, (obj_id, deg, orig_score, actions, control_path) in enumerate(weak_angles, 1):
        print(f"\n  [{idx}/{len(weak_angles)}] Enhancing {obj_id} {_rot_slug(deg)} "
              f"(orig={orig_score:.3f})")

        best_score = orig_score
        best_result = None
        current_control = control_path
        current_actions = [a for a in actions if a not in YAW_ACTIONS_BLACKLIST]

        for round_idx in range(max_rounds):
            if not current_actions or current_actions == ["NO_OP"]:
                current_actions = ["NO_OP"]

            pair_dir = work_root / obj_id / _rot_slug(deg)
            pair_dir.mkdir(parents=True, exist_ok=True)

            result = run_agent_step(
                output_dir=str(pair_dir),
                obj_id=obj_id,
                rotation_deg=deg,
                round_idx=round_idx,
                actions=current_actions,
                step_scale=step_scale,
                agent_note=f"angle_gate_r{round_idx}_{_rot_slug(deg)}",
                control_state_in=current_control,
                prev_renders_dir=None,
                device=device,
                blender_bin=blender,
                meshes_dir=str(meshes_dir),
                profile_path=str(profile_path),
                scene_template_path=scene_template,
            )

            new_score = result.get("hybrid_score", 0)
            print(f"    round {round_idx}: hybrid={new_score:.3f} "
                  f"(best={best_score:.3f}) actions={current_actions}")

            if new_score > best_score:
                best_score = new_score
                best_result = result
                current_control = result.get("state_out", current_control)

            current_actions = result.get("suggested_actions", ["NO_OP"])
            current_actions = [a for a in current_actions if a not in YAW_ACTIONS_BLACKLIST]

            if new_score >= angle_threshold:
                print(f"    → passed threshold ({angle_threshold}), stop")
                break

        if best_result and best_score > orig_score:
            enhanced[(obj_id, deg)] = {
                "result": best_result,
                "original_score": orig_score,
                "enhanced_score": best_score,
            }
            print(f"    → enhanced: {orig_score:.3f} → {best_score:.3f}")
        else:
            print(f"    → no improvement, keeping original")

    return enhanced


# ── Phase 3: Build output ────────────────────────────────────────────────────

def build_enhanced_output(
    source_root: Path,
    output_root: Path,
    objects: dict,
    angle_scores: Dict[str, Dict[int, dict]],
    enhanced: Dict[Tuple[str, int], dict],
    copy_mode: str,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    total_enhanced = 0
    out_objects = {}

    for obj_id in sorted(objects.keys()):
        obj_src = source_root / "objects" / obj_id
        obj_dst = output_root / "objects" / obj_id
        obj_dst.mkdir(parents=True, exist_ok=True)

        obj_manifest = json.loads((obj_src / "object_manifest.json").read_text())
        rotations = obj_manifest.get("rotations", [])
        new_rotations = []

        for rot_info in rotations:
            deg = rot_info["rotation_deg"]
            slug = _rot_slug(deg)
            new_rot = dict(rot_info)

            if (obj_id, deg) in enhanced:
                enh = enhanced[(obj_id, deg)]
                result = enh["result"]
                render_dir = Path(result["render_dir"]) / obj_id

                rgb_src = _find_render(render_dir, mask=False)
                mask_src = _find_render(render_dir, mask=True)
                meta_src = render_dir / "metadata.json"
                ctrl_src = Path(result.get("state_out", ""))

                if rgb_src and rgb_src.exists():
                    shutil.copy2(rgb_src, obj_dst / f"{slug}.png")
                if mask_src and mask_src.exists():
                    shutil.copy2(mask_src, obj_dst / f"{slug}_mask.png")
                if meta_src.exists():
                    shutil.copy2(meta_src, obj_dst / f"{slug}_render_metadata.json")
                if ctrl_src.exists():
                    shutil.copy2(ctrl_src, obj_dst / f"{slug}_control.json")

                new_rot["enhanced"] = True
                new_rot["original_vlm_score"] = enh["original_score"]
                new_rot["enhanced_hybrid_score"] = enh["enhanced_score"]
                new_rot["rgb_path"] = f"objects/{obj_id}/{slug}.png"
                new_rot["mask_path"] = f"objects/{obj_id}/{slug}_mask.png"
                new_rot["render_metadata_path"] = f"objects/{obj_id}/{slug}_render_metadata.json"
                new_rot["control_state_path"] = f"objects/{obj_id}/{slug}_control.json"
                total_enhanced += 1
            else:
                for suffix in [".png", "_mask.png", "_control.json", "_render_metadata.json"]:
                    src_file = obj_src / f"{slug}{suffix}"
                    dst_file = obj_dst / f"{slug}{suffix}"
                    if src_file.exists():
                        _copy_or_link(src_file, dst_file, copy_mode)

                new_rot["enhanced"] = False
                score_info = angle_scores.get(obj_id, {}).get(deg, {})
                new_rot["vlm_score"] = score_info.get("vlm_score")
                new_rot["rgb_path"] = f"objects/{obj_id}/{slug}.png"
                new_rot["mask_path"] = f"objects/{obj_id}/{slug}_mask.png"
                new_rot["render_metadata_path"] = f"objects/{obj_id}/{slug}_render_metadata.json"
                new_rot["control_state_path"] = f"objects/{obj_id}/{slug}_control.json"

            new_rotations.append(new_rot)

        obj_manifest["rotations"] = new_rotations
        obj_manifest["angle_gate_applied"] = True
        with (obj_dst / "object_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(obj_manifest, f, indent=2, ensure_ascii=False)

        out_objects[obj_id] = obj_manifest

    return out_objects, total_enhanced


def _find_render(render_dir: Path, mask: bool = False) -> Optional[Path]:
    if not render_dir.exists():
        return None
    suffix = "_mask.png" if mask else ".png"
    for f in sorted(render_dir.glob(f"az*_el*{suffix}")):
        if mask and "_mask" in f.name:
            return f
        if not mask and "_mask" not in f.name:
            return f
    return None


# ── Multi-GPU dispatcher ─────────────────────────────────────────────────────

SCRIPT_PATH = Path(__file__).resolve()


def run_multigpu(args, obj_ids: List[str]) -> dict:
    """Shard objects across GPUs, launch parallel workers, merge results."""
    gpu_indices = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_gpus = len(gpu_indices)
    output_root = Path(args.output_dir).resolve()

    shards: list[list[str]] = [[] for _ in range(n_gpus)]
    for i, obj_id in enumerate(obj_ids):
        shards[i % n_gpus].append(obj_id)

    print(f"[multigpu] {len(obj_ids)} objects → {n_gpus} GPUs: {[len(s) for s in shards]}")

    workers: list[tuple[_sp.Popen, str, Path]] = []
    for gpu, shard in zip(gpu_indices, shards):
        if not shard:
            continue
        worker_dir = output_root / f"_worker_{gpu}"
        worker_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(SCRIPT_PATH),
            "--source-root", args.source_root,
            "--output-dir", str(worker_dir),
            "--meshes-dir", args.meshes_dir,
            "--blender", args.blender,
            "--profile", args.profile,
            "--device", "cuda:0",
            "--ids", ",".join(shard),
            "--angle-threshold", str(args.angle_threshold),
            "--max-enhance-rounds", str(args.max_enhance_rounds),
            "--step-scale", str(args.step_scale),
            "--copy-mode", args.copy_mode,
        ]
        if args.scene_template:
            cmd += ["--scene-template", args.scene_template]

        log_path = worker_dir / "worker.log"
        fh = open(log_path, "w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["VLM_FORCE_SINGLE_GPU"] = "1"
        proc = _sp.Popen(cmd, stdout=fh, stderr=_sp.STDOUT, env=env)
        workers.append((proc, gpu, log_path))
        print(f"  [worker GPU {gpu}] PID {proc.pid}, {len(shard)} objects: {shard}")

    for proc, gpu, log_path in workers:
        ret = proc.wait()
        status = "OK" if ret == 0 else "FAILED"
        print(f"  [worker GPU {gpu}] exited {ret} [{status}]")
        if ret != 0:
            print(f"  [worker GPU {gpu}] see {log_path}")

    # Merge partial results
    merged_angle_scores: Dict[str, Dict[int, dict]] = {}
    merged_enhanced: Dict[Tuple[str, int], dict] = {}
    total_weak = 0
    total_eval_time = 0.0

    for gpu, shard in zip(gpu_indices, shards):
        if not shard:
            continue
        worker_dir = output_root / f"_worker_{gpu}"
        partial = worker_dir / "_partial_results.json"
        if not partial.exists():
            print(f"  [WARN] GPU {gpu}: no partial results")
            continue
        data = json.loads(partial.read_text())
        for obj_id, scores in data.get("angle_scores", {}).items():
            merged_angle_scores[obj_id] = {int(k): v for k, v in scores.items()}
        for key_str, enh in data.get("enhanced", {}).items():
            obj_id, deg_str = key_str.rsplit("_", 1)
            merged_enhanced[(obj_id, int(deg_str))] = enh
        total_weak += data.get("weak_count", 0)
        total_eval_time = max(total_eval_time, data.get("eval_seconds", 0))

    return {
        "angle_scores": merged_angle_scores,
        "enhanced": merged_enhanced,
        "weak_count": total_weak,
        "eval_seconds": total_eval_time,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_dir).resolve()
    meshes_dir = Path(args.meshes_dir).resolve()
    profile_path = Path(args.profile).resolve()
    profile = json.loads(profile_path.read_text())

    scene_template = args.scene_template
    if not scene_template:
        fixed_cam = source_root / "scene_template_fixed_camera.json"
        if fixed_cam.exists():
            scene_template = str(fixed_cam)
        else:
            scene_template = str(REPO_ROOT / "configs" / "scene_template.json")

    source_manifest = json.loads((source_root / "manifest.json").read_text())
    all_objects = source_manifest.get("objects", {})
    if not all_objects:
        raise ValueError(f"No objects in {source_root}/manifest.json")

    # Filter to --ids subset if in worker mode
    if args.ids:
        ids_set = set(args.ids.split(","))
        objects = {k: v for k, v in all_objects.items() if k in ids_set}
    else:
        objects = all_objects

    obj_ids = sorted(objects.keys())

    # Multi-GPU dispatcher mode
    if args.gpus and not args.ids:
        gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
        if len(gpu_list) > 1:
            total_angles = sum(
                len(obj.get("rotations", [])) for obj in objects.values()
            )
            print("=" * 60)
            print("[angle_gate] Multi-GPU Rotation8 Per-Angle Quality Gate")
            print(f"  Source:       {source_root}")
            print(f"  Output:       {output_root}")
            print(f"  Objects:      {len(objects)}")
            print(f"  Angles:       {total_angles}")
            print(f"  GPUs:         {gpu_list}")
            print(f"  Threshold:    {args.angle_threshold}")
            print(f"  Max rounds:   {args.max_enhance_rounds}")
            print("=" * 60)

            t0 = time.time()
            merged = run_multigpu(args, obj_ids)
            elapsed = time.time() - t0

            # Phase 3: build output from merged results
            print(f"\n=== Phase 3: Build enhanced output (merged from {len(gpu_list)} GPUs) ===")
            out_objects, total_enhanced = build_enhanced_output(
                source_root, output_root, all_objects,
                merged["angle_scores"], merged["enhanced"], args.copy_mode,
            )

            summary = {
                "success": True,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_root": str(source_root),
                "output_root": str(output_root),
                "angle_threshold": args.angle_threshold,
                "max_enhance_rounds": args.max_enhance_rounds,
                "total_objects": len(all_objects),
                "total_angles": total_angles,
                "weak_angles_found": merged["weak_count"],
                "angles_enhanced": total_enhanced,
                "angles_kept_original": total_angles - total_enhanced,
                "evaluation_seconds": round(merged["eval_seconds"], 1),
                "gpus": gpu_list,
                "per_angle_scores": {
                    obj_id: {
                        str(deg): round(info.get("vlm_score", 0), 3)
                        for deg, info in scores.items()
                    }
                    for obj_id, scores in merged["angle_scores"].items()
                },
            }

            with (output_root / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            manifest = {"summary": summary, "objects": out_objects}
            with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            print(f"\n{'=' * 60}")
            print(f"[angle_gate] Done (multi-GPU, {elapsed:.0f}s)")
            print(f"  Enhanced:  {total_enhanced} / {merged['weak_count']} weak angles")
            print(f"  Unchanged: {total_angles - total_enhanced} angles")
            print(f"  Output:    {output_root}")
            print(f"{'=' * 60}")
            return

    # Single-GPU / worker mode
    total_angles = sum(
        len(obj.get("rotations", [])) for obj in objects.values()
    )

    print("=" * 60)
    print("[angle_gate] Rotation8 Per-Angle Quality Gate")
    print(f"  Source:       {source_root}")
    print(f"  Output:       {output_root}")
    print(f"  Objects:      {len(objects)}")
    print(f"  Angles:       {total_angles}")
    print(f"  Threshold:    {args.angle_threshold}")
    print(f"  Max rounds:   {args.max_enhance_rounds}")
    print(f"  Device:       {args.device}")
    print("=" * 60)

    # Phase 1
    print("\n=== Phase 1: Evaluate all angles with VLM ===")
    t0 = time.time()
    angle_scores, weak_angles = evaluate_all_angles(
        source_root, objects, args.device, profile, args.angle_threshold,
    )
    eval_elapsed = time.time() - t0
    print(f"[angle_gate] Evaluation took {eval_elapsed:.0f}s")

    # Phase 2
    enhanced = {}
    if weak_angles:
        print(f"\n=== Phase 2: Enhance {len(weak_angles)} weak angles ===")
        t1 = time.time()
        work_root = output_root / "_enhance_work"
        enhanced = enhance_weak_angles(
            weak_angles, work_root, meshes_dir, args.blender,
            profile_path, scene_template, args.device,
            args.step_scale, args.max_enhance_rounds, args.angle_threshold,
        )
        enhance_elapsed = time.time() - t1
        print(f"\n[angle_gate] Enhancement took {enhance_elapsed:.0f}s")
    else:
        print("\n[angle_gate] No weak angles found, skipping enhancement")

    # Worker mode: write partial results for merger and return
    if args.ids:
        partial = {
            "angle_scores": {
                obj_id: {str(deg): info for deg, info in scores.items()}
                for obj_id, scores in angle_scores.items()
            },
            "enhanced": {
                f"{obj_id}_{deg}": enh
                for (obj_id, deg), enh in enhanced.items()
            },
            "weak_count": len(weak_angles),
            "eval_seconds": eval_elapsed,
        }
        with (output_root / "_partial_results.json").open("w", encoding="utf-8") as f:
            json.dump(partial, f, indent=2, ensure_ascii=False)
        print(f"[worker] Partial results written to {output_root / '_partial_results.json'}")
        return

    # Phase 3 (single-GPU full run)
    print(f"\n=== Phase 3: Build enhanced output ===")
    out_objects, total_enhanced = build_enhanced_output(
        source_root, output_root, objects,
        angle_scores, enhanced, args.copy_mode,
    )

    # Write summary & manifest
    summary = {
        "success": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "angle_threshold": args.angle_threshold,
        "max_enhance_rounds": args.max_enhance_rounds,
        "total_objects": len(objects),
        "total_angles": total_angles,
        "weak_angles_found": len(weak_angles),
        "angles_enhanced": total_enhanced,
        "angles_kept_original": total_angles - total_enhanced,
        "evaluation_seconds": round(eval_elapsed, 1),
        "per_angle_scores": {
            obj_id: {
                str(deg): round(info.get("vlm_score", 0), 3)
                for deg, info in scores.items()
            }
            for obj_id, scores in angle_scores.items()
        },
    }

    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    manifest = {"summary": summary, "objects": out_objects}
    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"[angle_gate] Done")
    print(f"  Enhanced:  {total_enhanced} / {len(weak_angles)} weak angles")
    print(f"  Unchanged: {total_angles - total_enhanced} angles")
    print(f"  Output:    {output_root}")
    print(f"{'=' * 60}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
