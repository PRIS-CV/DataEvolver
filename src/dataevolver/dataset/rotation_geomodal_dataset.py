"""
Utilities for loading the derived rotation8 geomodal train-ready datasets.

This module is intentionally read-only: it never mutates the dataset root.
It supports both:
- train-ready root
- object-disjoint split root
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from torch.utils.data import Dataset as _TorchDatasetBase
except ImportError:  # pragma: no cover
    class _TorchDatasetBase:  # type: ignore[override]
        pass


try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


def _load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _require_pil():
    if Image is None:
        raise ImportError("Pillow is required for image loading but is not installed")


def _require_numpy():
    if np is None:
        raise ImportError("numpy is required for array loading but is not installed")


def load_image(path: Path, *, as_numpy: bool = False):
    _require_pil()
    image = Image.open(path)
    if as_numpy:
        _require_numpy()
        return np.array(image)
    return image


def load_exr(path: Path):
    """
    Best-effort EXR loading.

    Tries imageio first, then OpenCV with EXR support. Raises an informative
    error if no backend is available.
    """
    _require_numpy()

    try:
        import imageio.v3 as iio  # type: ignore

        return iio.imread(path)
    except Exception:
        pass

    try:
        import cv2  # type: ignore

        data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if data is None:
            raise ValueError(f"cv2 failed to read EXR: {path}")
        return data
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load EXR '{path}'. Install imageio or OpenCV with EXR support."
        ) from exc


@dataclass(frozen=True)
class RotationGeomodalPaths:
    source_image: Path
    target_image: Path
    source_mask: Path
    target_mask: Path
    source_render_metadata: Path
    target_render_metadata: Path
    source_control_state: Path
    target_control_state: Path
    source_geometry_metadata: Path
    target_geometry_metadata: Path
    source_depth: Path
    target_depth: Path
    source_normal: Path
    target_normal: Path
    source_normal_vis: Path
    target_normal_vis: Path
    source_depth_vis: Path
    target_depth_vis: Path


def _resolve_paths(root: Path, row: dict) -> RotationGeomodalPaths:
    return RotationGeomodalPaths(
        source_image=(root / row["source_image"]).resolve(),
        target_image=(root / row["target_image"]).resolve(),
        source_mask=(root / row["source_mask"]).resolve(),
        target_mask=(root / row["target_mask"]).resolve(),
        source_render_metadata=(root / row["source_render_metadata"]).resolve(),
        target_render_metadata=(root / row["target_render_metadata"]).resolve(),
        source_control_state=(root / row["source_control_state"]).resolve(),
        target_control_state=(root / row["target_control_state"]).resolve(),
        source_geometry_metadata=(root / row["source_geometry_metadata"]).resolve(),
        target_geometry_metadata=(root / row["target_geometry_metadata"]).resolve(),
        source_depth=(root / row["source_depth"]).resolve(),
        target_depth=(root / row["target_depth"]).resolve(),
        source_normal=(root / row["source_normal"]).resolve(),
        target_normal=(root / row["target_normal"]).resolve(),
        source_normal_vis=(root / row["source_normal_vis"]).resolve(),
        target_normal_vis=(root / row["target_normal_vis"]).resolve(),
        source_depth_vis=(root / row["source_depth_vis"]).resolve(),
        target_depth_vis=(root / row["target_depth_vis"]).resolve(),
    )


class RotationGeomodalDataset(_TorchDatasetBase):
    """
    Read-only dataset wrapper for:
    - dataset_scene_v7_full20_rotation8_geomodal_trainready_front2others_20260407
    - dataset_scene_v7_full20_rotation8_geomodal_trainready_front2others_splitobj_seed42_20260407

    By default it returns:
    - pair-level fields from train_pairs.jsonl / val_pairs.jsonl / test_pairs.jsonl
    - absolute resolved paths
    - parsed geometry metadata

    Heavy modalities are opt-in.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        load_geometry: bool = True,
        load_render_metadata: bool = False,
        load_control_state: bool = False,
        load_rgb: bool = False,
        load_mask: bool = False,
        load_depth_exr: bool = False,
        load_normal_exr: bool = False,
        load_depth_vis: bool = False,
        load_normal_vis: bool = False,
        image_as_numpy: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.summary = _load_json(self.root / "summary.json", default={}) or {}
        self.manifest = _load_json(self.root / "manifest.json", default={}) or {}
        self.split = split
        self.load_geometry = load_geometry
        self.load_render_metadata = load_render_metadata
        self.load_control_state = load_control_state
        self.load_rgb = load_rgb
        self.load_mask = load_mask
        self.load_depth_exr = load_depth_exr
        self.load_normal_exr = load_normal_exr
        self.load_depth_vis = load_depth_vis
        self.load_normal_vis = load_normal_vis
        self.image_as_numpy = image_as_numpy

        self._pair_file = self._resolve_pair_file(split)
        self.rows = _load_jsonl(self._pair_file)

    def _resolve_pair_file(self, split: str) -> Path:
        pairs_root = self.root / "pairs"
        mapping = {
            "train": pairs_root / "train_pairs.jsonl",
            "val": pairs_root / "val_pairs.jsonl",
            "test": pairs_root / "test_pairs.jsonl",
            "all": pairs_root / "all_pairs.jsonl",
        }
        pair_file = mapping.get(split)
        if pair_file is None:
            raise ValueError(f"Unsupported split '{split}'. Expected train/val/test/all.")
        if pair_file.exists():
            return pair_file
        if split == "all":
            train_file = pairs_root / "train_pairs.jsonl"
            if train_file.exists():
                return train_file
        raise FileNotFoundError(pair_file)

    def available_splits(self) -> List[str]:
        pairs_root = self.root / "pairs"
        splits = []
        for split_name, file_name in [
            ("train", "train_pairs.jsonl"),
            ("val", "val_pairs.jsonl"),
            ("test", "test_pairs.jsonl"),
            ("all", "all_pairs.jsonl"),
        ]:
            if (pairs_root / file_name).exists():
                splits.append(split_name)
        return splits

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = dict(self.rows[index])
        paths = _resolve_paths(self.root, row)

        sample = {
            **row,
            "dataset_root": str(self.root),
            "paths": paths,
            "source_image_path_abs": str(paths.source_image),
            "target_image_path_abs": str(paths.target_image),
            "source_mask_path_abs": str(paths.source_mask),
            "target_mask_path_abs": str(paths.target_mask),
            "source_geometry_metadata_path_abs": str(paths.source_geometry_metadata),
            "target_geometry_metadata_path_abs": str(paths.target_geometry_metadata),
            "source_depth_path_abs": str(paths.source_depth),
            "target_depth_path_abs": str(paths.target_depth),
            "source_normal_path_abs": str(paths.source_normal),
            "target_normal_path_abs": str(paths.target_normal),
        }

        if self.load_geometry:
            sample["source_geometry"] = _load_json(paths.source_geometry_metadata, default={}) or {}
            sample["target_geometry"] = _load_json(paths.target_geometry_metadata, default={}) or {}

        if self.load_render_metadata:
            sample["source_render_metadata_json"] = _load_json(paths.source_render_metadata, default={}) or {}
            sample["target_render_metadata_json"] = _load_json(paths.target_render_metadata, default={}) or {}

        if self.load_control_state:
            sample["source_control_state_json"] = _load_json(paths.source_control_state, default={}) or {}
            sample["target_control_state_json"] = _load_json(paths.target_control_state, default={}) or {}

        if self.load_rgb:
            sample["source_image_data"] = load_image(paths.source_image, as_numpy=self.image_as_numpy)
            sample["target_image_data"] = load_image(paths.target_image, as_numpy=self.image_as_numpy)

        if self.load_mask:
            sample["source_mask_data"] = load_image(paths.source_mask, as_numpy=self.image_as_numpy)
            sample["target_mask_data"] = load_image(paths.target_mask, as_numpy=self.image_as_numpy)

        if self.load_depth_exr:
            sample["source_depth_data"] = load_exr(paths.source_depth)
            sample["target_depth_data"] = load_exr(paths.target_depth)

        if self.load_normal_exr:
            sample["source_normal_data"] = load_exr(paths.source_normal)
            sample["target_normal_data"] = load_exr(paths.target_normal)

        if self.load_depth_vis:
            sample["source_depth_vis_data"] = load_image(paths.source_depth_vis, as_numpy=self.image_as_numpy)
            sample["target_depth_vis_data"] = load_image(paths.target_depth_vis, as_numpy=self.image_as_numpy)

        if self.load_normal_vis:
            sample["source_normal_vis_data"] = load_image(paths.source_normal_vis, as_numpy=self.image_as_numpy)
            sample["target_normal_vis_data"] = load_image(paths.target_normal_vis, as_numpy=self.image_as_numpy)

        return sample

    def get_pair_ids(self) -> List[str]:
        return [str(row["pair_id"]) for row in self.rows]

    def iter_rows(self) -> Iterable[dict]:
        for idx in range(len(self)):
            yield self[idx]

    def describe(self) -> dict:
        return {
            "root": str(self.root),
            "split": self.split,
            "available_splits": self.available_splits(),
            "num_pairs": len(self.rows),
            "dataset_type": self.summary.get("dataset_type"),
            "summary": self.summary,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RotationGeomodalDataset(root={self.root!s}, split={self.split!r}, "
            f"num_pairs={len(self.rows)})"
        )


def summarize_geomodal_dataset(root: str | Path) -> dict:
    root_path = Path(root).resolve()
    summary = _load_json(root_path / "summary.json", default={}) or {}
    manifest = _load_json(root_path / "manifest.json", default={}) or {}
    return {
        "root": str(root_path),
        "summary": summary,
        "num_objects": len((manifest.get("objects") or {})),
        "num_pairs": len((manifest.get("pairs") or [])),
        "available_splits": RotationGeomodalDataset(root_path).available_splits(),
    }
