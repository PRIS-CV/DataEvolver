"""
Render all 8 rotations (yaw000~315) for all objects using the bright template.
Reuses yaw000 bright results already rendered, only renders yaw045~315.
Outputs a complete consistent rotation8 dataset matching the structure of
dataset_scene_v7_full50_rotation8_consistent_yaw000_final_20260410.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path("<run-root>")
SCENE_RENDER_SCRIPT = REPO_ROOT / "pipeline" / "stage4_scene_render.py"
BLENDER_BIN = "blender"
MESHES_DIR = REPO_ROOT / "pipeline" / "data" / "meshes"

# Source: original consistent dataset (for control states)
SOURCE_DATASET = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full50_rotation8_consistent_yaw000_final_20260410"

# Bright yaw000 already rendered
BRIGHT_YAW000 = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full50_rotation8_bright_yaw000_20260416"

# Bright template
BRIGHT_TEMPLATE = REPO_ROOT / "pipeline" / "data" / "scene_template_bright_camera.json"

# Output: full bright rotation8 dataset
OUTPUT_ROOT = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full50_rotation8_consistent_bright_20260416"

ROTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]
RESOLUTION = 1024
ENGINE = "CYCLES"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def render_one_rotation(obj_id: str, control: dict, rotation_deg: int, gpu_id: str) -> dict:
    """Render a single rotation for one object."""
    obj_out = OUTPUT_ROOT / "objects" / obj_id
    obj_out.mkdir(parents=True, exist_ok=True)
    rot_slug = f"yaw{rotation_deg:03d}"

    # Modify control state for this rotation
    ctrl = json.loads(json.dumps(control))  # deep copy
    ctrl.setdefault("object", {})
    ctrl["object"]["yaw_deg"] = float(rotation_deg)

    # Write temp control
    temp_dir = OUTPUT_ROOT / "_tmp" / f"{obj_id}_{rot_slug}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    control_path = temp_dir / f"{rot_slug}_control.json"
    save_json(control_path, ctrl)

    cmd = [
        BLENDER_BIN, "-b", "-P", str(SCENE_RENDER_SCRIPT), "--",
        "--input-dir", str(MESHES_DIR),
        "--output-dir", str(temp_dir),
        "--obj-id", obj_id,
        "--resolution", str(RESOLUTION),
        "--engine", ENGINE,
        "--control-state", str(control_path),
        "--scene-template", str(BRIGHT_TEMPLATE),
    ]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    started = time.time()
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = round(time.time() - started, 1)

    render_dir = temp_dir / obj_id
    rgb_src = render_dir / "az000_el+00.png"
    mask_src = render_dir / "az000_el+00_mask.png"
    meta_src = render_dir / "metadata.json"

    if completed.returncode != 0 or not rgb_src.exists():
        print(f"    [FAIL] {obj_id} {rot_slug} (rc={completed.returncode}, {elapsed}s)")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {"status": "failed", "rotation_deg": rotation_deg, "elapsed": elapsed}

    # Copy outputs
    shutil.copy2(rgb_src, obj_out / f"{rot_slug}.png")
    if mask_src.exists():
        shutil.copy2(mask_src, obj_out / f"{rot_slug}_mask.png")
    if meta_src.exists():
        shutil.copy2(meta_src, obj_out / f"{rot_slug}_render_metadata.json")
    shutil.copy2(control_path, obj_out / f"{rot_slug}_control.json")

    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"    [OK]   {obj_id} {rot_slug} ({elapsed}s)")
    return {"status": "success", "rotation_deg": rotation_deg, "elapsed": elapsed}


def main():
    gpu_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    print(f"[render_bright_all_rotations] GPU={gpu_id}")
    print(f"  Source: {SOURCE_DATASET}")
    print(f"  Bright yaw000: {BRIGHT_YAW000}")
    print(f"  Output: {OUTPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Copy templates
    shutil.copy2(BRIGHT_TEMPLATE, OUTPUT_ROOT / "scene_template_fixed_camera.json")

    # Discover objects
    objects_dir = SOURCE_DATASET / "objects"
    obj_ids = sorted([d.name for d in objects_dir.iterdir() if d.is_dir() and d.name.startswith("obj_")])
    print(f"  Objects: {len(obj_ids)}, Rotations: {ROTATIONS}")

    total_success = 0
    total_fail = 0

    for idx, obj_id in enumerate(obj_ids, 1):
        print(f"\n[{idx}/{len(obj_ids)}] {obj_id}")

        # Read yaw000 control from source (base orientation)
        control_path = objects_dir / obj_id / "yaw000_control.json"
        if not control_path.exists():
            print(f"  [SKIP] No yaw000_control.json")
            continue

        control = load_json(control_path)
        obj_out = OUTPUT_ROOT / "objects" / obj_id
        obj_out.mkdir(parents=True, exist_ok=True)

        # Copy yaw000 from bright preview
        bright_yaw000_dir = BRIGHT_YAW000 / "objects" / obj_id
        if (bright_yaw000_dir / "yaw000_bright.png").exists():
            shutil.copy2(bright_yaw000_dir / "yaw000_bright.png", obj_out / "yaw000.png")
            if (bright_yaw000_dir / "yaw000_bright_mask.png").exists():
                shutil.copy2(bright_yaw000_dir / "yaw000_bright_mask.png", obj_out / "yaw000_mask.png")
            if (bright_yaw000_dir / "yaw000_bright_render_metadata.json").exists():
                shutil.copy2(bright_yaw000_dir / "yaw000_bright_render_metadata.json", obj_out / "yaw000_render_metadata.json")
            if (bright_yaw000_dir / "yaw000_bright_control.json").exists():
                shutil.copy2(bright_yaw000_dir / "yaw000_bright_control.json", obj_out / "yaw000_control.json")
            print(f"    [COPY] yaw000 from bright preview")
            total_success += 1
        else:
            # Re-render yaw000
            r = render_one_rotation(obj_id, control, 0, gpu_id)
            total_success += 1 if r["status"] == "success" else 0
            total_fail += 1 if r["status"] == "failed" else 0

        # Render yaw045~315
        for rot in ROTATIONS:
            if rot == 0:
                continue
            r = render_one_rotation(obj_id, control, rot, gpu_id)
            total_success += 1 if r["status"] == "success" else 0
            total_fail += 1 if r["status"] == "failed" else 0

        # Copy object_manifest from source if exists
        src_manifest = objects_dir / obj_id / "object_manifest.json"
        if src_manifest.exists():
            shutil.copy2(src_manifest, obj_out / "object_manifest.json")

    # Write summary
    summary = {
        "success": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_type": "rotation8_consistent_bright",
        "source_dataset": str(SOURCE_DATASET),
        "bright_template": str(BRIGHT_TEMPLATE),
        "gpu_id": gpu_id,
        "total_objects": len(obj_ids),
        "rotations_per_object": 8,
        "total_renders": total_success,
        "failed_renders": total_fail,
        "rotations": ROTATIONS,
        "resolution": RESOLUTION,
        "engine": ENGINE,
    }
    save_json(OUTPUT_ROOT / "summary.json", summary)
    save_json(OUTPUT_ROOT / "manifest.json", {"summary": summary})

    print(f"\n[render_bright_all_rotations] Done: {total_success} ok, {total_fail} fail")
    print(f"  Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
