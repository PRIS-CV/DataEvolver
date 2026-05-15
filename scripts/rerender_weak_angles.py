"""
Re-render weak angles (90/180/270) with adjusted lighting for all objects.

The key light is fixed at yaw=0 (front). When the object rotates to 180/270,
the visible face gets less direct light. This script:
  1. Partially rotates key_yaw_deg toward the object's facing direction
  2. Increases env_strength_scale for more ambient fill
  3. Slightly increases material value_scale for brightness

Usage:
    python rerender_weak_angles.py \
        --source-root /path/to/rotation8_consistent \
        --output-dir /path/to/rotation8_relit \
        --meshes-dir /path/to/meshes \
        --blender /path/to/blender \
        --scene-template /path/to/scene_template_fixed_camera.json \
        --gpus 0,1,2
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

WEAK_ANGLE_ADJUSTMENTS = {
    90: {
        "lighting": {"key_yaw_deg": 30.0},
        "scene": {"env_strength_scale": 1.15},
        "material": {"value_scale": 1.05},
    },
    180: {
        "lighting": {"key_yaw_deg": 60.0},
        "scene": {"env_strength_scale": 1.25},
        "material": {"value_scale": 1.08},
    },
    270: {
        "lighting": {"key_yaw_deg": 90.0},
        "scene": {"env_strength_scale": 1.30},
        "material": {"value_scale": 1.10},
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Re-render weak angles with adjusted lighting")
    p.add_argument("--source-root", required=True, help="rotation8_consistent root")
    p.add_argument("--output-dir", required=True, help="Output directory for relit renders")
    p.add_argument("--meshes-dir", required=True, help="Stage1 meshes directory")
    p.add_argument("--blender", required=True, help="Blender binary path")
    p.add_argument("--scene-template", required=True, help="Scene template JSON")
    p.add_argument("--render-script", default=None, help="Path to stage4_scene_render.py")
    p.add_argument("--gpus", default="0,1,2", help="Comma-separated GPU IDs")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def apply_adjustments(control_state, adjustments):
    cs = json.loads(json.dumps(control_state))
    for group, params in adjustments.items():
        if group not in cs:
            cs[group] = {}
        for key, value in params.items():
            cs[group][key] = value
    return cs


def find_render_output(render_dir):
    rgb, mask = None, None
    for png in sorted(Path(render_dir).rglob("*.png")):
        if "_mask" in png.name:
            mask = png
        elif "az" in png.name or "yaw" in png.name:
            if rgb is None:
                rgb = png
    return rgb, mask


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_dir).resolve()
    meshes_dir = Path(args.meshes_dir).resolve()
    gpus = [int(g) for g in args.gpus.split(",")]

    render_script = args.render_script
    if not render_script:
        for candidate in [
            Path(__file__).resolve().parent.parent / "pipeline" / "stage4_scene_render.py",
            Path("<repo-root>/pipeline/stage4_scene_render.py"),
        ]:
            if candidate.exists():
                render_script = str(candidate)
                break
    if not render_script or not Path(render_script).exists():
        print(f"[ERROR] Cannot find stage4_scene_render.py")
        sys.exit(1)

    manifest = load_json(source_root / "manifest.json")
    objects = manifest.get("objects", {})
    obj_ids = sorted(objects.keys())

    print("=" * 60)
    print("[rerender] Weak-Angle Re-lighting")
    print(f"  Source:  {source_root}")
    print(f"  Output:  {output_root}")
    print(f"  Objects: {len(obj_ids)}")
    print(f"  Angles:  {sorted(WEAK_ANGLE_ADJUSTMENTS.keys())}")
    print(f"  GPUs:    {gpus}")
    print("=" * 60)

    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = output_root / "_rerender_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for obj_id in obj_ids:
        obj_src = source_root / "objects" / obj_id
        for deg, adjustments in sorted(WEAK_ANGLE_ADJUSTMENTS.items()):
            slug = f"yaw{deg:03d}"
            ctrl_path = obj_src / f"{slug}_control.json"
            if not ctrl_path.exists():
                print(f"  [SKIP] {obj_id} {slug}: no control state")
                continue
            tasks.append((obj_id, deg, slug, str(ctrl_path), adjustments))

    print(f"[rerender] Total render tasks: {len(tasks)}")

    if args.dry_run:
        for obj_id, deg, slug, ctrl_path, adj in tasks:
            print(f"  [dry-run] {obj_id} {slug}: {json.dumps(adj)}")
        return

    results = {"success": [], "failed": []}
    t0 = time.time()

    for idx, (obj_id, deg, slug, ctrl_path, adjustments) in enumerate(tasks, 1):
        gpu_id = gpus[idx % len(gpus)]
        control = load_json(ctrl_path)
        adjusted = apply_adjustments(control, adjustments)

        adj_ctrl_path = work_dir / obj_id / f"{slug}_adjusted_control.json"
        save_json(adj_ctrl_path, adjusted)

        glb_path = meshes_dir / f"{obj_id}.glb"
        if not glb_path.exists():
            print(f"  [{idx}/{len(tasks)}] {obj_id} {slug}: SKIP (no mesh at {glb_path})")
            results["failed"].append({"obj_id": obj_id, "angle": deg, "reason": "no_mesh"})
            continue

        render_out = work_dir / obj_id / slug
        render_out.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            args.blender, "--background", "--python", render_script,
            "--",
            "--input-dir", str(meshes_dir),
            "--output-dir", str(render_out),
            "--obj-id", obj_id,
            "--control-state", str(adj_ctrl_path),
            "--scene-template", args.scene_template,
        ]

        print(f"  [{idx}/{len(tasks)}] {obj_id} {slug} (gpu={gpu_id}) ...", end=" ", flush=True)

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
            rgb, mask = find_render_output(render_out)
            if proc.returncode == 0 and rgb:
                print(f"OK")
                results["success"].append({
                    "obj_id": obj_id, "angle": deg,
                    "rgb_path": str(rgb),
                    "mask_path": str(mask) if mask else None,
                    "adjusted_control": str(adj_ctrl_path),
                })
            else:
                print(f"FAIL (rc={proc.returncode})")
                results["failed"].append({
                    "obj_id": obj_id, "angle": deg,
                    "reason": f"rc={proc.returncode}",
                })
                with open(log_dir / f"{obj_id}_{slug}.log", "w") as lf:
                    lf.write(proc.stdout or "")
                    lf.write("\n--- STDERR ---\n")
                    lf.write(proc.stderr or "")
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            results["failed"].append({"obj_id": obj_id, "angle": deg, "reason": "timeout"})

    elapsed = time.time() - t0
    print(f"\n[rerender] Renders done in {elapsed:.0f}s: "
          f"{len(results['success'])} ok, {len(results['failed'])} failed")

    # Build output: symlink unchanged files, copy re-rendered weak angles
    print("[rerender] Building output directory...")
    for obj_id in obj_ids:
        obj_src = source_root / "objects" / obj_id
        obj_dst = output_root / "objects" / obj_id
        obj_dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(obj_src.iterdir()):
            dst_f = obj_dst / f.name
            if not (dst_f.exists() or dst_f.is_symlink()):
                os.symlink(f.resolve(), dst_f)

    replaced = 0
    for item in results["success"]:
        obj_id = item["obj_id"]
        deg = item["angle"]
        slug = f"yaw{deg:03d}"
        obj_dst = output_root / "objects" / obj_id

        rgb_src = Path(item["rgb_path"])
        dst_rgb = obj_dst / f"{slug}.png"
        if dst_rgb.exists() or dst_rgb.is_symlink():
            dst_rgb.unlink()
        shutil.copy2(rgb_src, dst_rgb)

        if item.get("mask_path"):
            mask_src = Path(item["mask_path"])
            dst_mask = obj_dst / f"{slug}_mask.png"
            if dst_mask.exists() or dst_mask.is_symlink():
                dst_mask.unlink()
            shutil.copy2(mask_src, dst_mask)

        adj_ctrl = Path(item["adjusted_control"])
        dst_ctrl = obj_dst / f"{slug}_control.json"
        if dst_ctrl.exists() or dst_ctrl.is_symlink():
            dst_ctrl.unlink()
        shutil.copy2(adj_ctrl, dst_ctrl)
        replaced += 1

    # Update manifest
    manifest_out = dict(manifest)
    manifest_out["rerender_info"] = {
        "weak_angles": sorted(WEAK_ANGLE_ADJUSTMENTS.keys()),
        "adjustments": {str(k): v for k, v in WEAK_ANGLE_ADJUSTMENTS.items()},
        "success_count": len(results["success"]),
        "failed_count": len(results["failed"]),
        "replaced_count": replaced,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(output_root / "manifest.json", manifest_out)
    save_json(output_root / "rerender_report.json", results)

    if (source_root / "scene_template_fixed_camera.json").exists():
        shutil.copy2(source_root / "scene_template_fixed_camera.json",
                      output_root / "scene_template_fixed_camera.json")

    print(f"\n{'=' * 60}")
    print(f"[rerender] Done.")
    print(f"  Replaced: {replaced} / {len(tasks)} weak-angle renders")
    print(f"  Output:   {output_root}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
