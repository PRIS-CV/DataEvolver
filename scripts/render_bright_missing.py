"""Re-render missing objects (obj_021~050) for the bright dataset."""
from __future__ import annotations
import json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path("<run-root>")
SCENE_RENDER_SCRIPT = REPO_ROOT / "pipeline" / "stage4_scene_render.py"
BLENDER_BIN = "blender"
MESHES_DIR = REPO_ROOT / "pipeline" / "data" / "meshes"
SOURCE_DATASET = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full50_rotation8_consistent_yaw000_final_20260410"
BRIGHT_TEMPLATE = REPO_ROOT / "pipeline" / "data" / "scene_template_bright_camera.json"
OUTPUT_ROOT = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full50_rotation8_consistent_bright_20260416"
ROTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]

def load_json(p):
    with open(p) as f: return json.load(f)
def save_json(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f: json.dump(d, f, indent=2, ensure_ascii=False)

def render_one(obj_id, control, rotation_deg, gpu_id):
    obj_out = OUTPUT_ROOT / "objects" / obj_id
    obj_out.mkdir(parents=True, exist_ok=True)
    rot_slug = f"yaw{rotation_deg:03d}"
    ctrl = json.loads(json.dumps(control))
    ctrl.setdefault("object", {})
    ctrl["object"]["yaw_deg"] = float(rotation_deg)
    temp_dir = OUTPUT_ROOT / "_tmp" / f"{obj_id}_{rot_slug}"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    cp = temp_dir / f"{rot_slug}_control.json"
    save_json(cp, ctrl)
    cmd = [BLENDER_BIN, "-b", "-P", str(SCENE_RENDER_SCRIPT), "--",
           "--input-dir", str(MESHES_DIR), "--output-dir", str(temp_dir),
           "--obj-id", obj_id, "--resolution", "1024", "--engine", "CYCLES",
           "--control-state", str(cp), "--scene-template", str(BRIGHT_TEMPLATE)]
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = gpu_id
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = round(time.time() - t0, 1)
    rgb = temp_dir / obj_id / "az000_el+00.png"
    mask = temp_dir / obj_id / "az000_el+00_mask.png"
    meta = temp_dir / obj_id / "metadata.json"
    if r.returncode != 0 or not rgb.exists():
        print(f"    [FAIL] {obj_id} {rot_slug} ({elapsed}s)")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False
    shutil.copy2(rgb, obj_out / f"{rot_slug}.png")
    if mask.exists(): shutil.copy2(mask, obj_out / f"{rot_slug}_mask.png")
    if meta.exists(): shutil.copy2(meta, obj_out / f"{rot_slug}_render_metadata.json")
    shutil.copy2(cp, obj_out / f"{rot_slug}_control.json")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"    [OK]   {obj_id} {rot_slug} ({elapsed}s)")
    return True

def main():
    gpu_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    objects_dir = SOURCE_DATASET / "objects"
    obj_ids = sorted([d.name for d in objects_dir.iterdir() if d.is_dir() and d.name.startswith("obj_")])
    # Find missing objects
    missing = []
    for obj_id in obj_ids:
        bright_dir = OUTPUT_ROOT / "objects" / obj_id
        if not (bright_dir / "yaw000.png").exists():
            missing.append(obj_id)
    print(f"Missing objects: {len(missing)} — {missing[:5]}...")
    ok, fail = 0, 0
    for idx, obj_id in enumerate(missing, 1):
        print(f"\n[{idx}/{len(missing)}] {obj_id}")
        control = load_json(objects_dir / obj_id / "yaw000_control.json")
        for rot in ROTATIONS:
            if render_one(obj_id, control, rot, gpu_id): ok += 1
            else: fail += 1
        src_manifest = objects_dir / obj_id / "object_manifest.json"
        if src_manifest.exists():
            shutil.copy2(src_manifest, OUTPUT_ROOT / "objects" / obj_id / "object_manifest.json")
    print(f"\nDone: {ok} ok, {fail} fail")

if __name__ == "__main__":
    main()
