from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.rotation_geomodal_dataset import RotationGeomodalDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect rotation geomodal dataset loader")
    parser.add_argument("--root", required=True, help="Geomodal train-ready or split dataset root")
    parser.add_argument("--split", default="train", help="train / val / test / all")
    parser.add_argument("--index", type=int, default=0, help="Sample index to inspect")
    parser.add_argument("--load-rgb", action="store_true")
    parser.add_argument("--load-depth-exr", action="store_true")
    parser.add_argument("--load-normal-exr", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ds = RotationGeomodalDataset(
        args.root,
        split=args.split,
        load_geometry=True,
        load_render_metadata=False,
        load_control_state=False,
        load_rgb=args.load_rgb,
        load_depth_exr=args.load_depth_exr,
        load_normal_exr=args.load_normal_exr,
        image_as_numpy=True,
    )
    sample = ds[args.index]

    preview = {
        "dataset": ds.describe(),
        "pair_id": sample["pair_id"],
        "obj_id": sample["obj_id"],
        "instruction": sample["instruction"],
        "source_rotation_deg": sample["source_rotation_deg"],
        "target_rotation_deg": sample["target_rotation_deg"],
        "source_image_path_abs": sample["source_image_path_abs"],
        "target_image_path_abs": sample["target_image_path_abs"],
        "source_geometry_keys": sorted(sample["source_geometry"].keys()),
        "target_geometry_keys": sorted(sample["target_geometry"].keys()),
        "source_bbox_2d": sample["source_geometry"].get("bbox_2d_xyxy"),
        "target_bbox_2d": sample["target_geometry"].get("bbox_2d_xyxy"),
    }

    if args.load_rgb:
        preview["source_image_shape"] = list(sample["source_image_data"].shape)
        preview["target_image_shape"] = list(sample["target_image_data"].shape)
    if args.load_depth_exr:
        preview["source_depth_shape"] = list(sample["source_depth_data"].shape)
        preview["target_depth_shape"] = list(sample["target_depth_data"].shape)
    if args.load_normal_exr:
        preview["source_normal_shape"] = list(sample["source_normal_data"].shape)
        preview["target_normal_shape"] = list(sample["target_normal_data"].shape)

    print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
