"""
Build the full50 scene dataset expansion in parallel roots.

Pipeline:
1. metadata precheck for obj021-050
2. stage1->3 sandbox asset generation for obj021-050
3. mesh precheck for obj021-050
4. yaw000 bootstrap for obj021-050
5. export obj021-050 consistent rotation4/rotation8
6. merge old20 + new30 into full50 consistent roots
7. build full50 train-ready / split
8. build full50 geommeta / geomodal / geomodal split
9. audit old roots before and after
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent

PRECHECK_SCRIPT = REPO_ROOT / "scripts" / "precheck_scene_full50_assets.py"
STAGE1_ASSETS_SCRIPT = REPO_ROOT / "scripts" / "build_scene_assets_from_stage1.py"
BOOTSTRAP_SCRIPT = REPO_ROOT / "scripts" / "bootstrap_scene_yaw000_objects.py"
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export_rotation8_from_best_object_state.py"
MERGE_SCRIPT = REPO_ROOT / "scripts" / "merge_rotation_consistent_roots.py"
TRAINREADY_SCRIPT = REPO_ROOT / "scripts" / "build_rotation8_trainready_dataset.py"
SPLIT_SCRIPT = REPO_ROOT / "scripts" / "build_object_split_for_rotation_dataset.py"
GEOMMETA_SCRIPT = REPO_ROOT / "scripts" / "build_rotation8_geommeta_from_consistent.py"
GEOMODAL_SCRIPT = REPO_ROOT / "scripts" / "build_rotation8_geomodal_trainready_dataset.py"

DEFAULT_OBJECTS_FILE = REPO_ROOT / "configs" / "seed_concepts" / "scene_obj021_050_objects.json"
DEFAULT_FULL_OBJECTS_FILE = REPO_ROOT / "configs" / "seed_concepts" / "scene_full50_objects.json"
DEFAULT_PROFILE = REPO_ROOT / "configs" / "dataset_profiles" / "scene_v7_full50_loop.json"
DEFAULT_SCENE_TEMPLATE = REPO_ROOT / "configs" / "scene_template.json"

OLD_ROT4 = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full20_rotation4_consistent_yaw000_20260407"
OLD_ROT8 = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full20_rotation8_consistent_yaw000_20260407"
OLD_TRAINREADY = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full20_rotation8_trainready_front2others_20260407"
OLD_SPLIT = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full20_rotation8_trainready_front2others_splitobj_seed42_20260407"
OLD_GEOMMETA = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full20_rotation8_geommeta_from_consistent_20260407"
OLD_GEOMODAL = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full20_rotation8_geomodal_trainready_front2others_20260407"
OLD_GEOMODAL_SPLIT = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full20_rotation8_geomodal_trainready_front2others_splitobj_seed42_20260407"

STAGE1_ASSET_ROOT = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_obj021_obj050_stage1_assets_20260408"
NEW_BOOTSTRAP = REPO_ROOT / "pipeline" / "data" / "evolution_scene_v7_obj021_obj050_yaw000_bootstrap_20260408"
NEW_ROT4 = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_obj021_obj050_rotation4_consistent_yaw000_20260408"
NEW_ROT8 = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_obj021_obj050_rotation8_consistent_yaw000_20260408"
FULL50_ROT4 = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full50_rotation4_consistent_yaw000_20260408"
FULL50_ROT8 = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full50_rotation8_consistent_yaw000_20260408"
FULL50_TRAINREADY = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full50_rotation8_trainready_front2others_20260408"
FULL50_SPLIT = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_20260408"
FULL50_GEOMMETA = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full50_rotation8_geommeta_from_consistent_20260408"
FULL50_GEOMODAL = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full50_rotation8_geomodal_trainready_front2others_20260408"
FULL50_GEOMODAL_SPLIT = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_full50_rotation8_geomodal_trainready_front2others_splitobj_seed42_20260408"
DEFAULT_REPORT_PATH = REPO_ROOT / "pipeline" / "data" / "scene_v7_full50_expansion_report_20260408.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Build the full50 scene expansion pipeline")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--blender", dest="blender_bin", required=True)
    parser.add_argument("--objects-file", default=str(DEFAULT_OBJECTS_FILE))
    parser.add_argument("--full-objects-file", default=str(DEFAULT_FULL_OBJECTS_FILE))
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--scene-template", default=str(DEFAULT_SCENE_TEMPLATE))
    parser.add_argument("--stage1-mode", choices=["api", "template-only", "dry-run"], default="template-only")
    parser.add_argument("--stage1-model", default="claude-haiku-4-5")
    parser.add_argument("--t2i-device", default="cuda:0")
    parser.add_argument("--sam-device", default="cuda:0")
    parser.add_argument("--i23d-devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--bootstrap-gpus", default="0,1,2")
    parser.add_argument("--export-gpus", default="0,1,2")
    parser.add_argument("--asset-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--train-objects", type=int, default=35)
    parser.add_argument("--val-objects", type=int, default=7)
    parser.add_argument("--test-objects", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    return parser.parse_args()


def audit_root(path: Path) -> dict:
    files = [p for p in path.rglob("*") if p.is_file()]
    latest = max((p.stat().st_mtime for p in files), default=0.0)
    return {"root": str(path), "file_count": len(files), "latest_mtime": latest}


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_step(cmd: list[str], name: str) -> dict:
    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "name": name,
        "command": cmd,
        "return_code": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "stdout_tail": completed.stdout[-8000:],
    }


def write_failure_report(path: Path, steps: list[dict], audits_before: dict):
    save_json(
        path,
        {
            "success": False,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "steps": steps,
            "old_root_audits_before": audits_before,
        },
    )


def main():
    args = parse_args()
    report_path = Path(args.report_path).resolve()
    old_roots = [
        OLD_ROT4,
        OLD_ROT8,
        OLD_TRAINREADY,
        OLD_SPLIT,
        OLD_GEOMMETA,
        OLD_GEOMODAL,
        OLD_GEOMODAL_SPLIT,
    ]
    old_audits_before = {root.name: audit_root(root) for root in old_roots if root.exists()}
    steps = []

    metadata_precheck_path = report_path.with_name("scene_v7_full50_metadata_precheck_20260408.json")
    steps.append(
        run_step(
            [
                args.python_bin,
                str(PRECHECK_SCRIPT),
                "--objects-file",
                args.objects_file,
                "--full-objects-file",
                args.full_objects_file,
                "--report-path",
                str(metadata_precheck_path),
            ],
            "metadata_precheck",
        )
    )
    if steps[-1]["return_code"] != 0:
        write_failure_report(report_path, steps, old_audits_before)
        raise SystemExit(1)

    steps.append(
        run_step(
            [
                args.python_bin,
                str(STAGE1_ASSETS_SCRIPT),
                "--objects-file",
                args.objects_file,
                "--output-root",
                str(STAGE1_ASSET_ROOT),
                "--python",
                args.python_bin,
                "--stage1-mode",
                args.stage1_mode,
                "--stage1-model",
                args.stage1_model,
                "--t2i-device",
                args.t2i_device,
                "--sam-device",
                args.sam_device,
                "--i23d-devices",
                args.i23d_devices,
            ],
            "build_stage1_assets",
        )
    )
    if steps[-1]["return_code"] != 0:
        write_failure_report(report_path, steps, old_audits_before)
        raise SystemExit(1)

    mesh_precheck_path = report_path.with_name("scene_v7_full50_mesh_precheck_20260408.json")
    steps.append(
        run_step(
            [
                args.python_bin,
                str(PRECHECK_SCRIPT),
                "--objects-file",
                args.objects_file,
                "--full-objects-file",
                args.full_objects_file,
                "--meshes-dir",
                str(STAGE1_ASSET_ROOT / "meshes"),
                "--require-meshes",
                "--report-path",
                str(mesh_precheck_path),
            ],
            "mesh_precheck",
        )
    )
    if steps[-1]["return_code"] != 0:
        write_failure_report(report_path, steps, old_audits_before)
        raise SystemExit(1)

    steps.append(
        run_step(
            [
                args.python_bin,
                str(BOOTSTRAP_SCRIPT),
                "--objects-file",
                args.objects_file,
                "--output-root",
                str(NEW_BOOTSTRAP),
                "--python",
                args.python_bin,
                "--blender",
                args.blender_bin,
                "--meshes-dir",
                str(STAGE1_ASSET_ROOT / "meshes"),
                "--profile",
                args.profile,
                "--scene-template",
                args.scene_template,
                "--gpus",
                args.bootstrap_gpus,
                "--max-rounds",
                "10",
            ],
            "bootstrap_yaw000",
        )
    )
    if steps[-1]["return_code"] != 0:
        write_failure_report(report_path, steps, old_audits_before)
        raise SystemExit(1)

    steps.append(
        run_step(
            [
                args.python_bin,
                str(EXPORT_SCRIPT),
                "--source-root",
                str(NEW_BOOTSTRAP),
                "--output-dir",
                str(NEW_ROT4),
                "--python",
                args.python_bin,
                "--blender",
                args.blender_bin,
                "--gpus",
                args.export_gpus,
                "--meshes-dir",
                str(STAGE1_ASSET_ROOT / "meshes"),
                "--scene-template",
                args.scene_template,
                "--base-rotation-deg",
                "0",
                "--rotations",
                "0,90,180,270",
            ],
            "export_obj021_050_rotation4",
        )
    )
    if steps[-1]["return_code"] != 0:
        write_failure_report(report_path, steps, old_audits_before)
        raise SystemExit(1)

    steps.append(
        run_step(
            [
                args.python_bin,
                str(EXPORT_SCRIPT),
                "--source-root",
                str(NEW_BOOTSTRAP),
                "--output-dir",
                str(NEW_ROT8),
                "--python",
                args.python_bin,
                "--blender",
                args.blender_bin,
                "--gpus",
                args.export_gpus,
                "--meshes-dir",
                str(STAGE1_ASSET_ROOT / "meshes"),
                "--scene-template",
                args.scene_template,
                "--base-rotation-deg",
                "0",
                "--rotations",
                "0,45,90,135,180,225,270,315",
            ],
            "export_obj021_050_rotation8",
        )
    )
    if steps[-1]["return_code"] != 0:
        write_failure_report(report_path, steps, old_audits_before)
        raise SystemExit(1)

    steps.append(
        run_step(
            [
                args.python_bin,
                str(MERGE_SCRIPT),
                "--source-roots",
                str(OLD_ROT4),
                str(NEW_ROT4),
                "--output-dir",
                str(FULL50_ROT4),
                "--asset-mode",
                args.asset_mode,
                "--expected-total-objects",
                "50",
            ],
            "merge_full50_rotation4",
        )
    )
    if steps[-1]["return_code"] != 0:
        write_failure_report(report_path, steps, old_audits_before)
        raise SystemExit(1)

    steps.append(
        run_step(
            [
                args.python_bin,
                str(MERGE_SCRIPT),
                "--source-roots",
                str(OLD_ROT8),
                str(NEW_ROT8),
                "--output-dir",
                str(FULL50_ROT8),
                "--asset-mode",
                args.asset_mode,
                "--expected-total-objects",
                "50",
            ],
            "merge_full50_rotation8",
        )
    )
    if steps[-1]["return_code"] != 0:
        write_failure_report(report_path, steps, old_audits_before)
        raise SystemExit(1)

    steps.append(
        run_step(
            [
                args.python_bin,
                str(TRAINREADY_SCRIPT),
                "--source-root",
                str(FULL50_ROT8),
                "--output-dir",
                str(FULL50_TRAINREADY),
                "--copy-mode",
                args.asset_mode,
            ],
            "build_full50_trainready",
        )
    )
    if steps[-1]["return_code"] != 0:
        write_failure_report(report_path, steps, old_audits_before)
        raise SystemExit(1)

    steps.append(
        run_step(
            [
                args.python_bin,
                str(SPLIT_SCRIPT),
                "--source-root",
                str(FULL50_TRAINREADY),
                "--output-dir",
                str(FULL50_SPLIT),
                "--train-objects",
                str(args.train_objects),
                "--val-objects",
                str(args.val_objects),
                "--test-objects",
                str(args.test_objects),
                "--seed",
                str(args.seed),
                "--asset-mode",
                args.asset_mode,
            ],
            "build_full50_split",
        )
    )
    if steps[-1]["return_code"] != 0:
        write_failure_report(report_path, steps, old_audits_before)
        raise SystemExit(1)

    steps.append(
        run_step(
            [
                args.python_bin,
                str(GEOMMETA_SCRIPT),
                "--source-root",
                str(FULL50_ROT8),
                "--output-dir",
                str(FULL50_GEOMMETA),
                "--gpus",
                args.export_gpus,
                "--python",
                args.python_bin,
                "--blender",
                args.blender_bin,
                "--scene-template",
                args.scene_template,
                "--asset-mode",
                args.asset_mode,
            ],
            "build_full50_geommeta",
        )
    )
    if steps[-1]["return_code"] != 0:
        write_failure_report(report_path, steps, old_audits_before)
        raise SystemExit(1)

    steps.append(
        run_step(
            [
                args.python_bin,
                str(GEOMODAL_SCRIPT),
                "--source-root",
                str(FULL50_GEOMMETA),
                "--output-dir",
                str(FULL50_GEOMODAL),
                "--gpus",
                args.export_gpus,
                "--python",
                args.python_bin,
                "--blender",
                args.blender_bin,
                "--asset-mode",
                args.asset_mode,
            ],
            "build_full50_geomodal_trainready",
        )
    )
    if steps[-1]["return_code"] != 0:
        write_failure_report(report_path, steps, old_audits_before)
        raise SystemExit(1)

    steps.append(
        run_step(
            [
                args.python_bin,
                str(SPLIT_SCRIPT),
                "--source-root",
                str(FULL50_GEOMODAL),
                "--output-dir",
                str(FULL50_GEOMODAL_SPLIT),
                "--train-objects",
                str(args.train_objects),
                "--val-objects",
                str(args.val_objects),
                "--test-objects",
                str(args.test_objects),
                "--seed",
                str(args.seed),
                "--asset-mode",
                args.asset_mode,
            ],
            "build_full50_geomodal_split",
        )
    )

    old_audits_after = {root.name: audit_root(root) for root in old_roots if root.exists()}
    report = {
        "success": all(step["return_code"] == 0 for step in steps) and (old_audits_before == old_audits_after),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": steps,
        "old_root_audits_before": old_audits_before,
        "old_root_audits_after": old_audits_after,
        "old_roots_unchanged": old_audits_before == old_audits_after,
        "outputs": {
            "stage1_asset_root": str(STAGE1_ASSET_ROOT),
            "bootstrap_root": str(NEW_BOOTSTRAP),
            "obj021_050_rotation4": str(NEW_ROT4),
            "obj021_050_rotation8": str(NEW_ROT8),
            "full50_rotation4": str(FULL50_ROT4),
            "full50_rotation8": str(FULL50_ROT8),
            "full50_trainready": str(FULL50_TRAINREADY),
            "full50_split": str(FULL50_SPLIT),
            "full50_geommeta": str(FULL50_GEOMMETA),
            "full50_geomodal": str(FULL50_GEOMODAL),
            "full50_geomodal_split": str(FULL50_GEOMODAL_SPLIT),
        },
    }
    save_json(report_path, report)
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
