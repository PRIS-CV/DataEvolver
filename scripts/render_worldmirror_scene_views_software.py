#!/usr/bin/env python3
"""Render HYWorld/WorldMirror scene views with a lightweight software rasterizer.

This is a no-model-load validation renderer.  It builds one WorldMirror
variable-depth mesh per selected frame, then projects that mesh through the
matching calibrated WorldMirror camera.  The goal is to verify the
Pano/depth/WorldMirror mesh/camera path without relying on VLM pass/fail or a
missing Blender executable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_view_map(value: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw in value.split(","):
        item = raw.strip()
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
    parser = argparse.ArgumentParser(description="Software render selected WorldMirror scene views")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--worldmirror-results-dir", required=True, help="Directory containing camera_params.json and depth/")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--hyworld-first-calibration", required=True, help="Calibration JSON containing hyworld_first_w2c")
    parser.add_argument("--panorama-path", required=True)
    parser.add_argument("--scene-output-dir", required=True)
    parser.add_argument("--report-scene-dir", required=True)
    parser.add_argument("--view-map", type=parse_view_map, required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--splat-radius", type=int, default=3)
    parser.add_argument("--background", choices=["source", "dark"], default="source")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-9:
        return np.eye(3) if dot > 0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (np.linalg.norm(cross) ** 2))


def load_camera_matrix(entry) -> np.ndarray:
    return np.asarray(entry["matrix"] if isinstance(entry, dict) else entry, dtype=np.float64)


def metric_vertices_to_output_camera(
    vertices_metric: np.ndarray,
    calibration: dict,
    output_w2c: np.ndarray,
) -> np.ndarray:
    first_hyworld_w2c = np.asarray(calibration["hyworld_first_w2c"], dtype=np.float64)
    normal = np.asarray(calibration["ground_normal_before_alignment"], dtype=np.float64)
    rotation = rotation_between(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    ground_height = -float(calibration["ground_plane_offset_before_alignment"])
    scale = float(calibration["metric_scale"])

    rotated = vertices_metric / scale
    rotated[:, 2] += ground_height
    aligned = rotated @ rotation
    output_world = transform_points(aligned, first_hyworld_w2c)
    return transform_points(output_world, output_w2c)


def load_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.uint8)
    if len(colors) != len(vertices):
        colors = np.full((len(vertices), 3), 180, dtype=np.uint8)
    return vertices, colors


def z_splat_render(
    vertices_metric: np.ndarray,
    colors: np.ndarray,
    calibration: dict,
    output_w2c: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    background: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, dict[str, float]]:
    camera = metric_vertices_to_output_camera(vertices_metric.copy(), calibration, output_w2c)
    z = camera[:, 2]
    valid = np.isfinite(z) & (z > 1e-4)
    camera = camera[valid]
    colors = colors[valid]
    z = z[valid]
    u = intrinsic[0, 0] * camera[:, 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera[:, 1] / z + intrinsic[1, 2]
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    in_frame = (ui >= -radius) & (ui < width + radius) & (vi >= -radius) & (vi < height + radius)
    ui = ui[in_frame]
    vi = vi[in_frame]
    z = z[in_frame]
    colors = colors[in_frame]
    order = np.argsort(z)[::-1]  # far to near; near overwrites.
    canvas = background.copy()
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    covered = np.zeros((height, width), dtype=bool)
    r = max(1, int(radius))
    for idx in order:
        x = int(ui[idx])
        y = int(vi[idx])
        zval = float(z[idx])
        x0 = max(0, x - r)
        x1 = min(width, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(height, y + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        patch = zbuf[y0:y1, x0:x1]
        mask = zval < patch
        if not mask.any():
            continue
        patch[mask] = zval
        canvas_patch = canvas[y0:y1, x0:x1]
        canvas_patch[mask] = colors[idx]
        covered[y0:y1, x0:x1][mask] = True
    stats = {
        "projected_vertex_count": int(len(ui)),
        "coverage_fraction": float(covered.mean()),
        "depth_min": float(np.nanmin(z)) if len(z) else None,
        "depth_max": float(np.nanmax(z)) if len(z) else None,
    }
    return canvas, stats


def index_for_prefix(images: list[Path], prefix: str) -> int:
    matches = [index for index, path in enumerate(images) if path.stem.startswith(prefix)]
    if not matches:
        raise RuntimeError(f"No image matches prefix {prefix}")
    return matches[0]


def run_mesh_build(args: argparse.Namespace, prefix: str, scene_id: str, shard_dir: Path) -> dict:
    mesh_path = shard_dir / f"{prefix}_scene_mesh.ply"
    contract_path = shard_dir / f"{prefix}_scene_contract.json"
    calibration_path = shard_dir / f"{prefix}_scene_mesh.calibration.json"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_worldmirror_scene_mesh.py"),
        "--worldmirror-dir", str(Path(args.worldmirror_results_dir).resolve()),
        "--images-dir", str(Path(args.images_dir).resolve()),
        "--hyworld-camera-path", str(Path(args.hyworld_first_calibration).resolve()),
        "--reuse-calibration-path", str(Path(args.hyworld_first_calibration).resolve()),
        "--panorama-path", str(Path(args.panorama_path).resolve()),
        "--output-mesh", str(mesh_path),
        "--output-contract", str(contract_path),
        "--output-calibration", str(calibration_path),
        "--allow-contract-failure",
        "--scene-id", scene_id,
        "--stride", str(args.stride),
        "--minimum-frame-count", "1",
        "--frame-name-prefix", prefix,
    ]
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "command": command,
        "stdout_tail": completed.stdout[-8000:],
        "mesh_path": str(mesh_path),
        "contract_path": str(contract_path),
        "calibration_path": str(calibration_path),
    }


def main() -> int:
    args = parse_args()
    worldmirror_results = Path(args.worldmirror_results_dir).resolve()
    images_dir = Path(args.images_dir).resolve()
    scene_output_dir = Path(args.scene_output_dir).resolve()
    report_scene_dir = Path(args.report_scene_dir).resolve()
    scene_views_dir = report_scene_dir / "scene_views"
    scene_views_dir.mkdir(parents=True, exist_ok=True)
    (scene_output_dir / "view_shards").mkdir(parents=True, exist_ok=True)

    cameras = read_json(worldmirror_results / "camera_params.json")
    extrinsics = cameras["extrinsics"]
    intrinsics = cameras["intrinsics"]
    images = sorted(images_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"No PNG images found in {images_dir}")

    results: list[dict] = []
    for label, prefix in args.view_map:
        shard_dir = scene_output_dir / "view_shards" / prefix
        render_path = scene_views_dir / f"view_{label}.png"
        if args.skip_existing and render_path.exists() and (shard_dir / f"{prefix}_scene_mesh.ply").exists():
            results.append({"label": label, "prefix": prefix, "render_path": str(render_path), "status": "skipped_existing"})
            continue
        shard_dir.mkdir(parents=True, exist_ok=True)
        build = run_mesh_build(args, prefix, f"{args.scene_id}_{label}_{prefix}", shard_dir)
        item = {"label": label, "prefix": prefix, "render_path": str(render_path), "build": build}
        if build["returncode"] != 0:
            item["status"] = "build_failed"
            results.append(item)
            break
        calibration = read_json(Path(build["calibration_path"]))
        selected = int(calibration["selected_frame_indices"][0])
        image_index = index_for_prefix(images, prefix)
        if selected != image_index:
            item["index_warning"] = {"selected_frame_index": selected, "image_prefix_index": image_index}
        source = Image.open(images[selected]).convert("RGB")
        width, height = source.size
        background = np.asarray(source if args.background == "source" else Image.new("RGB", source.size, (8, 8, 8))).copy()
        vertices, colors = load_mesh(Path(build["mesh_path"]))
        rendered, stats = z_splat_render(
            vertices,
            colors,
            calibration,
            load_camera_matrix(extrinsics[selected]),
            load_camera_matrix(intrinsics[selected])[:3, :3],
            width,
            height,
            background,
            args.splat_radius,
        )
        Image.fromarray(rendered).save(render_path)
        item.update({
            "status": "success",
            "selected_frame_index": selected,
            "source_image": str(images[selected]),
            "render_stats": stats,
            "render_width": width,
            "render_height": height,
            "renderer": "software_z_splat_worldmirror_mesh",
            "background_mode": args.background,
        })
        results.append(item)

    success = len(results) == len(args.view_map) and all(item.get("status") in {"success", "skipped_existing"} for item in results)
    summary = {
        "schema": "dataevolver.worldmirror_software_scene_views.v1",
        "scene_id": args.scene_id,
        "success": success,
        "renderer": "software_z_splat_worldmirror_mesh",
        "vlm_pass_authoritative": False,
        "views_requested": [{"label": label, "prefix": prefix} for label, prefix in args.view_map],
        "worldmirror_results_dir": str(worldmirror_results),
        "images_dir": str(images_dir),
        "results": results,
    }
    write_json(scene_views_dir / "scene_view_shards_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
