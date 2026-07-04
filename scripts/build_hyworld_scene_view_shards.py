#!/usr/bin/env python3
"""Build and render per-view HYWorld/WorldMirror scene shards.

This intentionally avoids the old unconditional multi-frame mesh fusion.  Each
view is reconstructed from the matching WorldMirror frame prefix, then rendered
with that frame's calibrated WorldMirror camera.  It is meant as a deterministic
scene-quality gate before object insertion is treated as complete.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_prefixes(value: str) -> list[str]:
    prefixes = [item.strip() for item in value.split(",") if item.strip()]
    if not prefixes:
        raise argparse.ArgumentTypeError("prefix list is empty")
    return prefixes


def parse_view_map(value: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError("view map entries must be label:prefix")
        label, prefix = [part.strip() for part in item.split(":", 1)]
        if not label or not prefix:
            raise argparse.ArgumentTypeError("view map entries must have non-empty label and prefix")
        pairs.append((label, prefix))
    if not pairs:
        raise argparse.ArgumentTypeError("view map is empty")
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/render per-view HYWorld scene shards")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument(
        "--hyworld-scene-dir",
        default=None,
        help="HYWorld scene directory; used to infer WorldMirror/images/camera/panorama paths when explicit paths are omitted.",
    )
    parser.add_argument("--worldmirror-dir", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--hyworld-camera-path", default=None)
    parser.add_argument("--panorama-path", default=None)
    parser.add_argument("--scene-output-dir", required=True)
    parser.add_argument("--template-output-dir", required=True)
    parser.add_argument("--report-scene-dir", required=True)
    parser.add_argument("--blender-bin", default=os.environ.get("BLENDER_BIN", "blender"))
    parser.add_argument("--prefixes", type=parse_prefixes, default=parse_prefixes(
        "pano-0000,pano-0045,pano-0090,pano-0135,pano-0180,pano-0225,pano-0270,pano-0315"
    ))
    parser.add_argument(
        "--view-map",
        type=parse_view_map,
        default=None,
        help="Comma-separated output_label:frame_prefix mapping, e.g. 000:pano-0000,045:pano-0034.",
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--render-width", type=int, default=832)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--engine", choices=["EEVEE", "CYCLES"], default="EEVEE")
    parser.add_argument("--xvfb", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def infer_inputs(args: argparse.Namespace) -> dict[str, Path]:
    scene_dir = Path(args.hyworld_scene_dir).resolve() if args.hyworld_scene_dir else None
    worldmirror_dir = Path(args.worldmirror_dir).resolve() if args.worldmirror_dir else None
    images_dir = Path(args.images_dir).resolve() if args.images_dir else None
    hyworld_camera_path = Path(args.hyworld_camera_path).resolve() if args.hyworld_camera_path else None
    panorama_path = Path(args.panorama_path).resolve() if args.panorama_path else None

    if scene_dir:
        if worldmirror_dir is None:
            candidates = [
                candidate.parent
                for candidate in scene_dir.rglob("camera_params.json")
                if (candidate.parent / "depth").is_dir()
            ]
            worldmirror_dir = first_existing(candidates)
        if images_dir is None and worldmirror_dir is not None:
            generated_image_dirs = (
                sorted((scene_dir / "render_results").glob("generation_bank_*/images"))
                if (scene_dir / "render_results").exists()
                else []
            )
            images_dir = first_existing([
                worldmirror_dir / "images",
                worldmirror_dir.parent / "images",
                scene_dir / "render_results" / "pano_bank" / "images",
                *generated_image_dirs,
            ])
        if hyworld_camera_path is None:
            trajectory_camera_files = (
                sorted((scene_dir / "render_results").glob("*/*/camera.json"))
                if (scene_dir / "render_results").exists()
                else []
            )
            hyworld_camera_path = first_existing([
                scene_dir / "render_results" / "pano_bank" / "cameras.json",
                scene_dir / "render_results" / "pano_bank" / "camera.json",
                *trajectory_camera_files,
            ])
        if panorama_path is None:
            panorama_path = first_existing([
                scene_dir / "panorama.png",
                scene_dir / "pano.png",
            ])

    missing = {
        "worldmirror_dir": worldmirror_dir,
        "images_dir": images_dir,
        "hyworld_camera_path": hyworld_camera_path,
        "panorama_path": panorama_path,
    }
    unresolved = [name for name, value in missing.items() if value is None]
    if unresolved:
        raise SystemExit(
            "Missing required inputs: "
            + ", ".join(unresolved)
            + ". Pass explicit paths or --hyworld-scene-dir with standard HYWorld outputs."
        )
    assert worldmirror_dir is not None and images_dir is not None and hyworld_camera_path is not None and panorama_path is not None
    return {
        "worldmirror_dir": worldmirror_dir,
        "images_dir": images_dir,
        "hyworld_camera_path": hyworld_camera_path,
        "panorama_path": panorama_path,
    }


def run_command(command: list[str], *, cwd: Path, dry_run: bool) -> dict[str, object]:
    if dry_run:
        return {"returncode": 0, "command": command, "stdout_tail": "", "dry_run": True}
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "command": command,
        "stdout_tail": completed.stdout[-8000:],
    }


def maybe_xvfb(command: list[str], mode: str) -> list[str]:
    if mode == "on" or (mode == "auto" and not os.environ.get("DISPLAY")):
        return ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24", *command]
    return command


def load_selected_camera_index(calibration_path: Path) -> int:
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    indices = payload.get("selected_frame_indices") or []
    if not indices:
        raise RuntimeError(f"Calibration has no selected_frame_indices: {calibration_path}")
    return int(indices[0])


def view_suffix(prefix: str) -> str:
    raw = prefix.rsplit("-", 1)[-1]
    try:
        return f"{int(raw):03d}"
    except ValueError:
        return raw


def main() -> int:
    args = parse_args()
    inputs = infer_inputs(args)
    views = args.view_map or [(view_suffix(prefix), prefix) for prefix in args.prefixes]
    scene_output_dir = Path(args.scene_output_dir).resolve()
    template_output_dir = Path(args.template_output_dir).resolve()
    report_scene_dir = Path(args.report_scene_dir).resolve()
    scene_views_dir = report_scene_dir / "scene_views"
    scene_views_dir.mkdir(parents=True, exist_ok=True)
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    template_output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for suffix, prefix in views:
        shard_dir = scene_output_dir / "view_shards" / prefix
        mesh_path = shard_dir / f"{prefix}_scene_mesh.ply"
        contract_path = shard_dir / f"{prefix}_scene_contract.json"
        calibration_path = shard_dir / f"{prefix}_scene_mesh.calibration.json"
        blend_path = template_output_dir / f"{args.scene_id}_{prefix}.blend"
        template_path = template_output_dir / f"{args.scene_id}_{prefix}.template.json"
        render_path = scene_views_dir / f"view_{suffix}.png"
        log_path = scene_views_dir / f"view_{suffix}.log"

        item: dict[str, object] = {
            "prefix": prefix,
            "view": suffix,
            "mesh_path": str(mesh_path),
            "contract_path": str(contract_path),
            "calibration_path": str(calibration_path),
            "blend_path": str(blend_path),
            "template_path": str(template_path),
            "render_path": str(render_path),
        }
        if args.skip_existing and render_path.exists() and contract_path.exists() and calibration_path.exists():
            item["status"] = "skipped_existing"
            results.append(item)
            continue

        build_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_worldmirror_scene_mesh.py"),
            "--worldmirror-dir", str(inputs["worldmirror_dir"]),
            "--images-dir", str(inputs["images_dir"]),
            "--hyworld-camera-path", str(inputs["hyworld_camera_path"]),
            "--panorama-path", str(inputs["panorama_path"]),
            "--output-mesh", str(mesh_path),
            "--output-contract", str(contract_path),
            "--output-calibration", str(calibration_path),
            "--scene-id", f"{args.scene_id}_{prefix}",
            "--stride", str(args.stride),
            "--minimum-frame-count", "1",
            "--frame-name-prefix", prefix,
        ]
        build_result = run_command(build_cmd, cwd=REPO_ROOT, dry_run=args.dry_run)
        item["build"] = build_result
        if build_result["returncode"] != 0:
            item["status"] = "build_failed"
            results.append(item)
            break

        camera_index = 0 if args.dry_run else load_selected_camera_index(calibration_path)
        item["worldmirror_camera_index"] = camera_index
        create_cmd = maybe_xvfb([
            args.blender_bin,
            "-b",
            "-P",
            str(REPO_ROOT / "scripts" / "create_hyworld_scene_blend.py"),
            "--",
            "--contract", str(contract_path),
            "--blend-output", str(blend_path),
            "--template-output", str(template_path),
            "--render-width", str(args.render_width),
            "--render-height", str(args.render_height),
            "--engine", args.engine,
            "--eevee-samples", str(args.samples),
            "--camera-source", "worldmirror",
            "--worldmirror-camera-index", str(camera_index),
        ], args.xvfb)
        create_result = run_command(create_cmd, cwd=REPO_ROOT, dry_run=args.dry_run)
        item["create_blend"] = create_result
        if create_result["returncode"] != 0:
            item["status"] = "create_blend_failed"
            results.append(item)
            break

        render_cmd = maybe_xvfb([
            args.blender_bin,
            "-b",
            str(blend_path),
            "-P",
            str(REPO_ROOT / "scripts" / "render_hyworld_blend.py"),
            "--",
            "--output", str(render_path),
            "--samples", str(args.samples),
            "--emission-strength", "1.0",
        ], args.xvfb)
        render_result = run_command(render_cmd, cwd=REPO_ROOT, dry_run=args.dry_run)
        item["render"] = render_result
        if not args.dry_run:
            log_path.write_text(render_result.get("stdout_tail", ""), encoding="utf-8")
        if render_result["returncode"] != 0:
            item["status"] = "render_failed"
            results.append(item)
            break
        item["status"] = "success"
        results.append(item)

    success = all(item.get("status") in {"success", "skipped_existing"} for item in results) and len(results) == len(views)
    summary = {
        "schema": "dataevolver.hyworld_scene_view_shards.v1",
        "scene_id": args.scene_id,
        "success": success,
        "views_requested": [{"label": label, "prefix": prefix} for label, prefix in views],
        "inputs": {key: str(value) for key, value in inputs.items()},
        "results": results,
        "scene_views_dir": str(scene_views_dir),
    }
    summary_path = scene_views_dir / "scene_view_shards_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
