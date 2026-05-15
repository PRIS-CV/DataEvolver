"""
Render brightened yaw000 views for all objects using a new scene template.

Reuses existing yaw000_control.json from the consistent yaw000 dataset,
only changing the scene template to increase brightness.
Does NOT modify the original dataset.
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

# Source: existing consistent yaw000 dataset (read-only)
SOURCE_DATASET = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full50_rotation8_consistent_yaw000_final_20260410"

# New bright template
BRIGHT_TEMPLATE = REPO_ROOT / "pipeline" / "data" / "scene_template_bright_camera.json"

# Output: new brightened dataset
OUTPUT_ROOT = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full50_rotation8_bright_yaw000_20260416"

RESOLUTION = 1024
ENGINE = "CYCLES"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def render_one_object(obj_id: str, control_json: dict, gpu_id: str = "0") -> dict:
    """Render yaw000 for a single object using the bright template."""
    obj_out = OUTPUT_ROOT / "objects" / obj_id
    obj_out.mkdir(parents=True, exist_ok=True)

    # Write control state to temp location
    temp_dir = OUTPUT_ROOT / "_tmp" / obj_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    control_path = temp_dir / "yaw000_control.json"
    save_json(control_path, control_json)

    # Build Blender command
    cmd = [
        BLENDER_BIN,
        "-b",
        "-P", str(SCENE_RENDER_SCRIPT),
        "--",
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
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = round(time.time() - started, 1)

    # Collect results
    render_dir = temp_dir / obj_id
    rgb_src = render_dir / "az000_el+00.png"
    mask_src = render_dir / "az000_el+00_mask.png"
    meta_src = render_dir / "metadata.json"

    if completed.returncode != 0 or not rgb_src.exists():
        print(f"  [FAIL] {obj_id} (rc={completed.returncode}, {elapsed}s)")
        if completed.stdout:
            print(completed.stdout[-2000:])
        return {
            "obj_id": obj_id,
            "status": "failed",
            "return_code": completed.returncode,
            "elapsed_seconds": elapsed,
        }

    # Copy results to output
    shutil.copy2(rgb_src, obj_out / "yaw000_bright.png")
    if mask_src.exists():
        shutil.copy2(mask_src, obj_out / "yaw000_bright_mask.png")
    if meta_src.exists():
        shutil.copy2(meta_src, obj_out / "yaw000_bright_render_metadata.json")
    shutil.copy2(control_path, obj_out / "yaw000_bright_control.json")

    # Also copy the original yaw000 for comparison
    src_obj_dir = SOURCE_DATASET / "objects" / obj_id
    orig_rgb = src_obj_dir / "yaw000.png"
    if orig_rgb.exists():
        shutil.copy2(orig_rgb, obj_out / "yaw000_original.png")

    # Cleanup temp
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"  [OK]   {obj_id} ({elapsed}s)")
    return {
        "obj_id": obj_id,
        "status": "success",
        "return_code": 0,
        "elapsed_seconds": elapsed,
        "bright_path": f"objects/{obj_id}/yaw000_bright.png",
        "original_path": f"objects/{obj_id}/yaw000_original.png",
    }


def main():
    gpu_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    print(f"[render_bright_yaw000] GPU={gpu_id}")
    print(f"  Source: {SOURCE_DATASET}")
    print(f"  Template: {BRIGHT_TEMPLATE}")
    print(f"  Output: {OUTPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Copy bright template to output for reference
    shutil.copy2(BRIGHT_TEMPLATE, OUTPUT_ROOT / "scene_template_bright_camera.json")

    # Also copy original template for comparison
    orig_template = SOURCE_DATASET / "scene_template_fixed_camera.json"
    if orig_template.exists():
        shutil.copy2(orig_template, OUTPUT_ROOT / "scene_template_original.json")

    # Discover objects
    objects_dir = SOURCE_DATASET / "objects"
    obj_ids = sorted([d.name for d in objects_dir.iterdir() if d.is_dir() and d.name.startswith("obj_")])
    print(f"  Objects: {len(obj_ids)}")

    results = []
    for idx, obj_id in enumerate(obj_ids, 1):
        print(f"\n[{idx}/{len(obj_ids)}] Rendering {obj_id} ...")

        # Read existing control state
        control_path = objects_dir / obj_id / "yaw000_control.json"
        if not control_path.exists():
            print(f"  [SKIP] No yaw000_control.json for {obj_id}")
            results.append({"obj_id": obj_id, "status": "skipped"})
            continue

        control = load_json(control_path)
        result = render_one_object(obj_id, control, gpu_id)
        results.append(result)

    # Summary
    success = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")
    skipped = sum(1 for r in results if r.get("status") == "skipped")

    summary = {
        "success": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_type": "bright_yaw000_preview",
        "source_dataset": str(SOURCE_DATASET),
        "bright_template": str(BRIGHT_TEMPLATE),
        "gpu_id": gpu_id,
        "total_objects": len(obj_ids),
        "rendered": success,
        "failed": failed,
        "skipped": skipped,
        "results": results,
    }
    save_json(OUTPUT_ROOT / "summary.json", summary)

    print(f"\n[render_bright_yaw000] Done: {success} ok, {failed} fail, {skipped} skip")
    print(f"  Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
