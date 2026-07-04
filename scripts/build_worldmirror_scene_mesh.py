#!/usr/bin/env python3
"""Build a Blender-ready colored mesh from WorldMirror depth and cameras."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
except ModuleNotFoundError:
    class _NoopSpan:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def set_attribute(self, *args, **kwargs):
            return None

        def set_attributes(self, *args, **kwargs):
            return None

        def record_exception(self, *args, **kwargs):
            return None

        def set_status(self, *args, **kwargs):
            return None

    class _NoopTracer:
        def start_as_current_span(self, *args, **kwargs):
            return _NoopSpan()

    class _NoopCounter:
        def add(self, *args, **kwargs):
            return None

    class _NoopHistogram:
        def record(self, *args, **kwargs):
            return None

    class _NoopMeter:
        def create_counter(self, *args, **kwargs):
            return _NoopCounter()

        def create_histogram(self, *args, **kwargs):
            return _NoopHistogram()

    class _NoopTraceModule:
        @staticmethod
        def get_tracer(*args, **kwargs):
            return _NoopTracer()

    class _NoopMetricsModule:
        @staticmethod
        def get_meter(*args, **kwargs):
            return _NoopMeter()

    class StatusCode:
        ERROR = "ERROR"

    class Status:
        def __init__(self, *args, **kwargs):
            pass

    metrics = _NoopMetricsModule()
    trace = _NoopTraceModule()
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from hyworld_scene_contract import build_scene_contract, write_contract
try:
    from observability import init_observability, shutdown_observability
except ModuleNotFoundError:
    def init_observability(*args, **kwargs):
        return None

    def shutdown_observability(*args, **kwargs):
        return None


logger = logging.getLogger("dataevolver.worldmirror_mesh")
tracer = trace.get_tracer("dataevolver.worldmirror_mesh")
meter = metrics.get_meter("dataevolver.worldmirror_mesh")
mesh_builds = meter.create_counter("dataevolver.worldmirror_mesh.builds", unit="1")
mesh_duration = meter.create_histogram("dataevolver.worldmirror_mesh.duration", unit="s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert WorldMirror depth output into a real 3D scene mesh")
    parser.add_argument("--worldmirror-dir", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--hyworld-camera-path", required=True)
    parser.add_argument("--panorama-path", default=None)
    parser.add_argument("--output-mesh", required=True)
    parser.add_argument("--output-contract", required=True)
    parser.add_argument("--output-calibration", default=None)
    parser.add_argument(
        "--allow-contract-failure",
        action="store_true",
        help="For scene-view diagnostics, keep the mesh/calibration even if support-surface contract extraction fails.",
    )
    parser.add_argument(
        "--reuse-calibration-path",
        default=None,
        help="Reuse an existing WorldMirror alignment calibration instead of fitting a new ground plane.",
    )
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--support-object-name", default="HYWorldReconstruction")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--depth-edge-ratio", type=float, default=0.12)
    parser.add_argument("--camera-height-m", type=float, default=1.6)
    parser.add_argument("--ground-quantile", type=float, default=0.10)
    parser.add_argument("--ground-distance-threshold", type=float, default=0.02)
    parser.add_argument("--support-plane-tolerance-m", type=float, default=0.10)
    parser.add_argument(
        "--frame-name-prefix",
        action="append",
        default=[],
        help="Only fuse frames whose image stem starts with this prefix; may be repeated.",
    )
    parser.add_argument("--minimum-frame-count", type=int, default=3)
    return parser.parse_args()


def load_camera_matrices(path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    extrinsics = [np.asarray(item["matrix"], dtype=np.float64) for item in payload["extrinsics"]]
    intrinsics = [np.asarray(item["matrix"], dtype=np.float64) for item in payload["intrinsics"]]
    return extrinsics, intrinsics


def load_hyworld_first_w2c(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "hyworld_first_w2c" in payload:
        return np.asarray(payload["hyworld_first_w2c"], dtype=np.float64)
    if "extrinsics" in payload:
        return np.asarray(payload["extrinsics"][0]["matrix"], dtype=np.float64)
    if "extrinsic" in payload:
        extrinsic = payload["extrinsic"]
        if isinstance(extrinsic, list) and extrinsic and isinstance(extrinsic[0], list):
            first = extrinsic[0]
            if first and isinstance(first[0], list):
                return np.asarray(first, dtype=np.float64)
        return np.asarray(extrinsic, dtype=np.float64)
    first_key = sorted(payload)[0]
    return np.asarray(payload[first_key]["extrinsic"], dtype=np.float64)


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


def fit_ground_plane(vertices: np.ndarray, quantile: float, distance_threshold: float) -> tuple[np.ndarray, float, int]:
    rng = np.random.default_rng(42)
    sample = vertices
    if len(sample) > 500_000:
        sample = sample[rng.choice(len(sample), 500_000, replace=False)]
    cutoff = float(np.quantile(sample[:, 2], quantile))
    low = sample[sample[:, 2] <= cutoff]
    if len(low) > 50_000:
        low = low[rng.choice(len(low), 50_000, replace=False)]
    minimum_up_alignment = math.cos(math.radians(20.0))
    best_indices = np.empty(0, dtype=np.int64)
    for _ in range(750):
        points = low[rng.choice(len(low), 3, replace=False)]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        if normal[2] < 0:
            normal = -normal
        if normal[2] < minimum_up_alignment:
            continue
        offset = -float(np.dot(normal, points[0]))
        indices = np.flatnonzero(np.abs(low @ normal + offset) <= distance_threshold)
        if len(indices) > len(best_indices):
            best_indices = indices
    if len(best_indices) < 500:
        raise RuntimeError(f"WorldMirror ground plane has too few inliers: {len(best_indices)}")

    inliers = low[best_indices]
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))
    if normal[2] < math.cos(math.radians(20.0)):
        raise RuntimeError(f"WorldMirror low surface is not a valid ground plane: normal={normal.tolist()}")
    return normal, offset, len(best_indices)


def make_frame_mesh(
    depth: np.ndarray,
    image: Image.Image,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
    stride: int,
    edge_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape
    rows = np.arange(0, height, stride)
    cols = np.arange(0, width, stride)
    sampled = depth[np.ix_(rows, cols)]
    uu, vv = np.meshgrid(cols, rows)
    z = sampled
    x = (uu - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (vv - intrinsic[1, 2]) * z / intrinsic[1, 1]
    camera_points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    points = transform_points(camera_points, np.linalg.inv(w2c))

    rgb = np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC))
    colors = rgb[np.ix_(rows, cols)].reshape(-1, 3)
    valid = np.isfinite(sampled) & (sampled > 0)
    grid = np.arange(sampled.size, dtype=np.int64).reshape(sampled.shape)
    quad_depth = np.stack(
        [sampled[:-1, :-1], sampled[:-1, 1:], sampled[1:, :-1], sampled[1:, 1:]],
        axis=-1,
    )
    quad_valid = (
        valid[:-1, :-1]
        & valid[:-1, 1:]
        & valid[1:, :-1]
        & valid[1:, 1:]
    )
    continuity = (quad_depth.max(axis=-1) - quad_depth.min(axis=-1)) <= (
        edge_ratio * np.maximum(np.median(quad_depth, axis=-1), 1e-6)
    )
    keep = quad_valid & continuity
    top_left = grid[:-1, :-1][keep]
    top_right = grid[:-1, 1:][keep]
    bottom_left = grid[1:, :-1][keep]
    bottom_right = grid[1:, 1:][keep]
    faces = np.concatenate(
        [
            np.stack([top_left, bottom_left, top_right], axis=1),
            np.stack([top_right, bottom_left, bottom_right], axis=1),
        ],
        axis=0,
    )
    return points, faces, colors


def build_mesh(args: argparse.Namespace) -> dict[str, object]:
    import trimesh

    worldmirror_dir = Path(args.worldmirror_dir).resolve()
    images_dir = Path(args.images_dir).resolve()
    depth_files = sorted((worldmirror_dir / "depth").glob("*.npy"))
    image_files = sorted(images_dir.glob("*.png"))
    output_extrinsics, output_intrinsics = load_camera_matrices(worldmirror_dir / "camera_params.json")
    aligned_count = min(len(depth_files), len(image_files), len(output_extrinsics), len(output_intrinsics))
    prefixes = tuple(str(value) for value in args.frame_name_prefix if str(value))
    selected_indices = [
        index
        for index in range(aligned_count)
        if not prefixes or image_files[index].stem.startswith(prefixes)
    ]
    frame_count = len(selected_indices)
    if frame_count < max(1, int(args.minimum_frame_count)):
        raise RuntimeError(f"WorldMirror reconstruction has only {frame_count} aligned frames")

    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    colors_parts: list[np.ndarray] = []
    vertex_offset = 0
    depth_values: list[np.ndarray] = []
    for index in selected_indices:
        depth = np.load(depth_files[index])
        depth_values.append(depth.reshape(-1))
        vertices, faces, colors = make_frame_mesh(
            depth,
            Image.open(image_files[index]),
            output_extrinsics[index],
            output_intrinsics[index],
            args.stride,
            args.depth_edge_ratio,
        )
        vertices_parts.append(vertices)
        faces_parts.append(faces + vertex_offset)
        colors_parts.append(colors)
        vertex_offset += len(vertices)

    vertices = np.concatenate(vertices_parts)
    faces = np.concatenate(faces_parts)
    colors = np.concatenate(colors_parts)
    first_hyworld_w2c = load_hyworld_first_w2c(Path(args.hyworld_camera_path).resolve())
    vertices = transform_points(vertices, np.linalg.inv(first_hyworld_w2c))

    reused_alignment = None
    if args.reuse_calibration_path:
        reused_alignment = json.loads(Path(args.reuse_calibration_path).read_text(encoding="utf-8"))
        normal = np.asarray(reused_alignment["ground_normal_before_alignment"], dtype=np.float64)
        offset = float(reused_alignment["ground_plane_offset_before_alignment"])
        ground_inliers = int(reused_alignment.get("ground_inlier_count", 0))
        camera_height = float(reused_alignment["camera_height_before_scale"])
        scale = float(reused_alignment["metric_scale"])
    else:
        normal, offset, ground_inliers = fit_ground_plane(
            vertices,
            args.ground_quantile,
            args.ground_distance_threshold,
        )
        camera_height = abs(offset)
        if camera_height < 0.05:
            raise RuntimeError(f"WorldMirror ground plane is too close to camera origin: {camera_height}")
        scale = args.camera_height_m / camera_height
    rotation = rotation_between(normal, np.array([0.0, 0.0, 1.0]))
    vertices = vertices @ rotation.T
    ground_height = -offset
    vertices[:, 2] -= ground_height
    vertices *= scale

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=colors,
        process=False,
    )
    mesh.remove_unreferenced_vertices()
    output_mesh = Path(args.output_mesh).resolve()
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_mesh)

    all_depth = np.concatenate(depth_values)
    calibration = {
        "schema": "dataevolver.worldmirror_calibration.v1",
        "scene_id": args.scene_id,
        "frame_count": frame_count,
        "aligned_input_count": aligned_count,
        "frame_name_prefixes": list(prefixes),
        "selected_frame_indices": selected_indices,
        "selected_frame_names": [image_files[index].stem for index in selected_indices],
        "depth_coefficient_of_variation": float(all_depth.std() / max(abs(all_depth.mean()), 1e-9)),
        "hyworld_first_w2c": first_hyworld_w2c.round(8).tolist(),
        "ground_normal_before_alignment": normal.round(8).tolist(),
        "ground_plane_offset_before_alignment": offset,
        "ground_inlier_count": ground_inliers,
        "camera_height_before_scale": camera_height,
        "target_camera_height_m": args.camera_height_m,
        "metric_scale": scale,
        "reused_alignment_calibration_path": str(Path(args.reuse_calibration_path).resolve()) if args.reuse_calibration_path else None,
        "mesh_vertex_count": int(len(mesh.vertices)),
        "mesh_face_count": int(len(mesh.faces)),
    }
    calibration_path = Path(args.output_calibration).resolve() if args.output_calibration else output_mesh.with_suffix(".calibration.json")
    calibration_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")

    contract_error = None
    try:
        contract = build_scene_contract(
            output_mesh,
            args.support_object_name,
            scene_id=args.scene_id,
            depth_path=worldmirror_dir / "depth",
            panorama_path=args.panorama_path,
            camera_path=worldmirror_dir / "camera_params.json",
            support_height_hint=0.0,
            support_plane_tolerance=args.support_plane_tolerance_m,
        )
        contract["scene_asset"]["calibration_path"] = str(calibration_path)
        write_contract(args.output_contract, contract)
    except Exception as exc:
        if not args.allow_contract_failure:
            raise
        contract_error = str(exc)
        contract = {
            "schema": "dataevolver.hyworld_scene_contract_failed.v1",
            "scene_id": args.scene_id,
            "error": contract_error,
            "scene_asset": {
                "mesh_path": str(output_mesh),
                "calibration_path": str(calibration_path),
            },
            "support_surfaces": [],
            "camera_path": str(worldmirror_dir / "camera_params.json"),
            "environment": {"panorama_path": args.panorama_path},
        }
        Path(args.output_contract).resolve().write_text(
            json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return {
        "success": True,
        "mesh_path": str(output_mesh),
        "contract_path": str(Path(args.output_contract).resolve()),
        "calibration_path": str(calibration_path),
        "frame_count": frame_count,
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "support_surface": contract["support_surfaces"][0] if contract.get("support_surfaces") else None,
        "contract_error": contract_error,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_observability("dataevolver.worldmirror_scene_mesh")
    started = time.perf_counter()
    outcome = "failure"
    response: dict[str, object]
    with tracer.start_as_current_span("hyworld.worldmirror_mesh.build") as span:
        span.set_attribute("hyworld.scene_id", args.scene_id)
        try:
            response = build_mesh(args)
            outcome = "success"
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            logger.exception("WorldMirror scene mesh build failed")
            response = {"success": False, "error": str(exc)}
        elapsed = time.perf_counter() - started
        span.set_attributes({"outcome": outcome, "hyworld.duration_seconds": elapsed})
        mesh_builds.add(1, {"outcome": outcome})
        mesh_duration.record(elapsed, {"outcome": outcome})
        response["elapsed_seconds"] = round(elapsed, 3)
    print(json.dumps(response, indent=2, ensure_ascii=False))
    shutdown_observability()
    return 0 if response.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
