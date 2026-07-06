"""Validation and contracts for HYWorld-to-Blender 3D scene integration."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - HYWorld is an optional extra.
    np = None  # type: ignore[assignment]


SCHEMA_VERSION = "dataevolver.hyworld_scene.v1"
OBJECT_SCHEMA_VERSION = "dataevolver.object3d.v1"


class SceneContractError(RuntimeError):
    pass


def _require_numpy():
    if np is None:
        raise ImportError('HYWorld scene contracts require numpy; install with python -m pip install -e ".[hyworld]"')
    return np


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _axis_index(axis: str) -> int:
    try:
        return {"x": 0, "y": 1, "z": 2}[axis.lower()]
    except KeyError as exc:
        raise ValueError("up_axis must be x, y, or z") from exc


def radial_statistics(vertices: np.ndarray, max_samples: int = 250_000) -> dict[str, float]:
    np = _require_numpy()
    if len(vertices) > max_samples:
        indices = np.linspace(0, len(vertices) - 1, max_samples, dtype=np.int64)
        vertices = vertices[indices]
    radii = np.linalg.norm(vertices, axis=1)
    mean = float(radii.mean())
    std = float(radii.std())
    return {
        "min": float(radii.min()),
        "p05": float(np.quantile(radii, 0.05)),
        "median": float(np.median(radii)),
        "p95": float(np.quantile(radii, 0.95)),
        "max": float(radii.max()),
        "mean": mean,
        "std": std,
        "coefficient_of_variation": std / max(abs(mean), 1e-9),
        "relative_range": (float(radii.max()) - float(radii.min())) / max(abs(mean), 1e-9),
    }


def depth_statistics(distance: np.ndarray) -> dict[str, float]:
    np = _require_numpy()
    values = np.asarray(distance, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        raise SceneContractError("Depth payload has no finite positive values")
    mean = float(values.mean())
    std = float(values.std())
    return {
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "mean": mean,
        "std": std,
        "coefficient_of_variation": std / max(abs(mean), 1e-9),
        "relative_range": (float(values.max()) - float(values.min())) / max(abs(mean), 1e-9),
    }


def validate_non_shell_geometry(
    radial: dict[str, float],
    depth: dict[str, float] | None = None,
    minimum_variation: float = 0.01,
) -> None:
    if radial["coefficient_of_variation"] < minimum_variation or radial["relative_range"] < minimum_variation:
        raise SceneContractError(
            "HYWorld geometry is a constant-radius panorama shell, not a placeable 3D scene"
        )
    if depth is not None and (
        depth["coefficient_of_variation"] < minimum_variation
        or depth["relative_range"] < minimum_variation
    ):
        raise SceneContractError(
            "HYWorld depth is effectively constant; regenerate metric depth instead of using a fixed-radius fallback"
        )


def load_depth_payload(path: str | os.PathLike[str]) -> tuple[np.ndarray, dict[str, float]]:
    np = _require_numpy()
    depth_path = Path(path)
    if depth_path.is_dir():
        files = sorted(depth_path.glob("*.npy"))
        if not files:
            raise SceneContractError(f"Depth directory has no .npy files: {depth_path}")
        values = np.concatenate([np.load(file).reshape(-1) for file in files])
        stats = depth_statistics(values)
        stats["source_file_count"] = len(files)
        return values, stats
    if depth_path.suffix.lower() == ".npy":
        values = np.load(depth_path)
        return values, depth_statistics(values)

    import torch

    payload = torch.load(depth_path, map_location="cpu", weights_only=False)
    distance = payload.get("distance") if isinstance(payload, dict) else payload
    if distance is None:
        raise SceneContractError("Depth payload is missing distance")
    if hasattr(distance, "detach"):
        distance = distance.detach().cpu().numpy()
    values = np.asarray(distance)
    return values, depth_statistics(values)


def _load_mesh(path: Path):
    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    if hasattr(loaded, "geometry"):
        geometries = [geom for geom in loaded.geometry.values() if hasattr(geom, "faces") and len(geom.faces)]
        if not geometries:
            raise SceneContractError(f"No triangle geometry found in {path}")
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded
    if not hasattr(mesh, "faces") or not len(mesh.faces):
        raise SceneContractError(f"No triangle faces found in {path}")
    return mesh


@dataclass
class SupportSurface:
    object_name: str
    anchor_xyz: list[float]
    bounds_xy: list[float]
    height: float
    area: float
    face_count: int
    normal_xyz: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": "support_000",
            "source": "hyworld_reconstruction",
            "artificial": False,
            "object_name": self.object_name,
            "anchor_xyz": self.anchor_xyz,
            "bounds_xy": self.bounds_xy,
            "height": self.height,
            "area": self.area,
            "face_count": self.face_count,
            "normal_xyz": self.normal_xyz,
        }


def extract_support_surface(
    mesh,
    object_name: str,
    up_axis: str = "z",
    maximum_slope_degrees: float = 20.0,
    plane_tolerance: float | None = None,
    minimum_area: float = 0.5,
    minimum_span: float = 0.5,
    max_faces: int = 500_000,
    preferred_height: float | None = None,
) -> SupportSurface:
    np = _require_numpy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    area_scale = 1.0
    if len(faces) > max_faces:
        original_face_count = len(faces)
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        faces = faces[indices]
        normals = normals[indices]
        areas = areas[indices]
        area_scale = original_face_count / len(indices)

    up_idx = _axis_index(up_axis)
    horizontal = np.abs(normals[:, up_idx]) >= math.cos(math.radians(maximum_slope_degrees))
    finite = np.isfinite(areas) & (areas > 0)
    candidate = horizontal & finite
    if not candidate.any():
        raise SceneContractError("HYWorld mesh has no sufficiently horizontal reconstructed faces")

    face_vertices = vertices[faces[candidate]]
    centers = face_vertices.mean(axis=1)
    candidate_areas = areas[candidate] * area_scale
    scene_extent = float(np.ptp(vertices[:, up_idx]))
    tolerance = plane_tolerance or max(0.02, scene_extent * 0.005)
    bins = np.round(centers[:, up_idx] / tolerance).astype(np.int64)
    unique_bins, inverse = np.unique(bins, return_inverse=True)
    area_by_bin = np.bincount(inverse, weights=candidate_areas)
    bin_heights = unique_bins.astype(np.float64) * tolerance
    lower_limit = float(vertices[:, up_idx].min())
    upper_limit = float(vertices[:, up_idx].max())
    support_ceiling = lower_limit + 0.45 * max(upper_limit - lower_limit, tolerance)
    lower_bins = np.flatnonzero(bin_heights <= support_ceiling)
    if preferred_height is None:
        order = lower_bins[np.argsort(area_by_bin[lower_bins])[::-1]]
    else:
        order = lower_bins[
            np.lexsort(
                (
                    -area_by_bin[lower_bins],
                    np.abs(bin_heights[lower_bins] - preferred_height),
                )
            )
        ]

    horizontal_axes = [idx for idx in range(3) if idx != up_idx]
    for selected in order:
        mask = inverse == selected
        selected_faces = face_vertices[mask]
        selected_areas = candidate_areas[mask]
        points = selected_faces.reshape(-1, 3)
        first = points[:, horizontal_axes[0]]
        second = points[:, horizontal_axes[1]]
        bounds = [float(first.min()), float(first.max()), float(second.min()), float(second.max())]
        spans = [bounds[1] - bounds[0], bounds[3] - bounds[2]]
        area = float(selected_areas.sum())
        if area < minimum_area or min(spans) < minimum_span:
            continue
        weights = np.repeat(selected_areas, 3)
        anchor = np.average(points, axis=0, weights=weights)
        height = float(np.average(centers[mask, up_idx], weights=selected_areas))
        anchor[up_idx] = height
        normal = np.zeros(3, dtype=np.float64)
        normal[up_idx] = 1.0
        return SupportSurface(
            object_name=object_name,
            anchor_xyz=[round(float(value), 6) for value in anchor],
            bounds_xy=[round(value, 6) for value in bounds],
            height=round(height, 6),
            area=round(area, 6),
            face_count=int(mask.sum()),
            normal_xyz=normal.tolist(),
        )
    raise SceneContractError("HYWorld mesh has no support surface large enough for object placement")


def build_scene_contract(
    mesh_path: str | os.PathLike[str],
    support_object_name: str,
    *,
    scene_id: str,
    up_axis: str = "z",
    depth_path: str | os.PathLike[str] | None = None,
    panorama_path: str | os.PathLike[str] | None = None,
    camera_path: str | os.PathLike[str] | None = None,
    support_height_hint: float | None = None,
    support_plane_tolerance: float | None = None,
) -> dict[str, Any]:
    np = _require_numpy()
    mesh_file = Path(mesh_path).resolve()
    if not mesh_file.is_file():
        raise FileNotFoundError(mesh_file)
    if not support_object_name or support_object_name.lower() in {"plane", "ground", "supportplane"}:
        raise SceneContractError("Production support_object_name must identify reconstructed HYWorld geometry")

    mesh = _load_mesh(mesh_file)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    radial = radial_statistics(vertices)
    depth_info = None
    if depth_path:
        _, depth_info = load_depth_payload(depth_path)
    validate_non_shell_geometry(radial, depth_info)
    support = extract_support_surface(
        mesh,
        support_object_name,
        up_axis=up_axis,
        preferred_height=support_height_hint,
        plane_tolerance=support_plane_tolerance,
    )

    contract = {
        "schema": SCHEMA_VERSION,
        "status": "ready",
        "scene_id": scene_id,
        "coordinate_frame": {
            "units": "meters",
            "handedness": "right",
            "up_axis": up_axis.lower(),
            "origin": "hyworld_reconstruction",
        },
        "scene_asset": {
            "mesh_path": str(mesh_file),
            "sha256": _sha256(mesh_file),
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "bounds": np.asarray(mesh.bounds, dtype=float).round(6).tolist(),
            "radial_statistics": radial,
            "depth_path": str(Path(depth_path).resolve()) if depth_path else None,
            "depth_statistics": depth_info,
        },
        "support_surfaces": [support.to_dict()],
        "selected_support_surface_id": "support_000",
        "camera_path": str(Path(camera_path).resolve()) if camera_path else None,
        "environment": {
            "panorama_path": str(Path(panorama_path).resolve()) if panorama_path else None,
            "panorama_role": "lighting_only",
        },
        "render_policy": {
            "artificial_support_allowed": False,
            "require_shared_renderer": True,
            "require_depth_normal_same_camera": True,
        },
    }
    return contract


def build_object_contract(
    asset_path: str | os.PathLike[str],
    *,
    object_id: str,
    up_axis: str = "z",
) -> dict[str, Any]:
    np = _require_numpy()
    asset = Path(asset_path).resolve()
    mesh = _load_mesh(asset)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    up_idx = _axis_index(up_axis)
    extents = bounds[1] - bounds[0]
    horizontal_axes = [idx for idx in range(3) if idx != up_idx]
    return {
        "schema": OBJECT_SCHEMA_VERSION,
        "status": "ready",
        "object_id": object_id,
        "asset_path": str(asset),
        "sha256": _sha256(asset),
        "coordinate_frame": {
            "units": "meters",
            "handedness": "right",
            "up_axis": up_axis.lower(),
        },
        "bounds": bounds.round(6).tolist(),
        "dimensions_m": extents.round(6).tolist(),
        "contact_plane": {
            "height": round(float(bounds[0, up_idx]), 6),
            "footprint": [
                round(float(bounds[0, horizontal_axes[0]]), 6),
                round(float(bounds[1, horizontal_axes[0]]), 6),
                round(float(bounds[0, horizontal_axes[1]]), 6),
                round(float(bounds[1, horizontal_axes[1]]), 6),
            ],
        },
        "canonical_forward_axis": "-y",
    }


def write_contract(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, output)


def load_scene_contract(path: str | os.PathLike[str]) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA_VERSION or payload.get("status") != "ready":
        raise SceneContractError(f"Invalid or unready scene contract: {path}")
    policy = payload.get("render_policy") or {}
    if policy.get("artificial_support_allowed") is not False:
        raise SceneContractError("Production scene contract must forbid artificial support geometry")
    selected_id = payload.get("selected_support_surface_id")
    surfaces = payload.get("support_surfaces") or []
    selected = next((surface for surface in surfaces if surface.get("id") == selected_id), None)
    if not selected or selected.get("artificial") is not False:
        raise SceneContractError("Scene contract lacks a reconstructed support surface")
    return payload


def selected_support_surface(contract: dict[str, Any]) -> dict[str, Any]:
    selected_id = contract["selected_support_surface_id"]
    return next(surface for surface in contract["support_surfaces"] if surface["id"] == selected_id)
