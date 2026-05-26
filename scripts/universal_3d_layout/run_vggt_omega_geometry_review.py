#!/usr/bin/env python3
"""VGGT-Omega geometry review adapter for universal 3D layout records.

The server currently has VGGT-Omega weights under:

    /huggingface/model_hub/VGGT-Omega

but not the official inference code. This adapter therefore provides:

- manifest mode: record integration status and expected paths
- proxy mode: run deterministic geometry checks from existing metadata/masks

Once facebookresearch/vggt-omega code is installed, the inference section can be
filled in without changing the JSONL contract consumed by the rest of the demo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MODEL_DIR = Path("/huggingface/model_hub/VGGT-Omega")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--mode", choices=["manifest", "proxy"], default="proxy")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_records(dataset_root: Path) -> list[dict]:
    path = dataset_root / "metadata" / "records.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_exists(dataset_root: Path, rel_path: str | None) -> bool:
    return bool(rel_path) and (dataset_root / rel_path).exists() and (dataset_root / rel_path).stat().st_size > 0


def model_status(model_dir: Path) -> dict:
    files = {}
    if model_dir.exists():
        for path in sorted(model_dir.glob("*")):
            if path.is_file() and path.name != ".hfd":
                files[path.name] = path.stat().st_size
    return {
        "model_dir": str(model_dir),
        "exists": model_dir.exists(),
        "has_weights": any(name.endswith(".pt") for name in files),
        "files": files,
        "official_code_status": "not_found_in_model_cache",
        "license_note": "cc-by-nc-4.0; do not publish local .hfd metadata",
    }


def bbox_depth_order(record: dict) -> list[str]:
    objects = record.get("objects", [])
    centers = []
    for obj in objects:
        center = obj.get("bbox_3d", {}).get("center_xyz", [0, 0, 0])
        camera_distance = record.get("camera", {}).get("pose", {}).get("distance")
        # In proxy mode we do not have a predicted depth map. Use bbox z/y cues
        # only as a deterministic sanity fallback and mark the source clearly.
        centers.append((obj.get("object_id"), center[1], center[2], camera_distance))
    return [item[0] for item in sorted(centers, key=lambda item: (item[1], -item[2])) if item[0]]


def proxy_review_record(dataset_root: Path, record: dict, status: dict) -> dict:
    artifacts = record["render_artifacts"]
    mask_paths = artifacts.get("masks", [])
    missing = []
    for key in ["target_image", "structure_image", "depth", "normal"]:
        if not file_exists(dataset_root, artifacts.get(key)):
            missing.append(key)
    for i, rel_path in enumerate(mask_paths):
        if not file_exists(dataset_root, rel_path):
            missing.append(f"mask_{i}")

    recorded_order = record.get("layout_control", {}).get("depth_order", [])
    proxy_order = bbox_depth_order(record)
    order_match = proxy_order == recorded_order if proxy_order and recorded_order else None
    mask_cov = record.get("validation", {}).get("mask_coverage", 0.0)
    failure_tags = []
    if missing:
        failure_tags.append("missing_artifact")
    if mask_cov < 0.004:
        failure_tags.append("very_low_mask_coverage")
    if order_match is False:
        failure_tags.append("proxy_depth_order_mismatch")

    checks = {
        "artifact_completeness": "pass" if not missing else "fail",
        "mask_depth_order_consistency": "proxy_pass" if order_match is True else ("proxy_unknown" if order_match is None else "proxy_fail"),
        "occlusion_pair_consistency": "pending_vggt_inference",
        "camera_pose_consistency": "pending_vggt_inference",
        "scale_visibility": "pass" if mask_cov >= 0.004 else "warn",
    }
    suggestions = []
    if "very_low_mask_coverage" in failure_tags:
        suggestions.extend(["CAMERA_TIGHTER", "A_SCALE_UP_10_OR_B_SCALE_UP_10"])
    if "proxy_depth_order_mismatch" in failure_tags:
        suggestions.append("VERIFY_OCCLUSION_PAIR_WITH_VGGT_DEPTH")

    return {
        "record_id": record["record_id"],
        "reviewer": "VGGT-Omega geometry reviewer",
        "mode": "proxy",
        "model_path": str(status["model_dir"]),
        "model_status": status,
        "status": "needs_vggt_inference" if failure_tags else "proxy_accepted",
        "predicted_artifacts": {"depth": None, "camera": None, "point_cloud": None},
        "checks": checks,
        "metrics": {
            "recorded_mask_coverage": mask_cov,
            "recorded_depth_order": recorded_order,
            "proxy_bbox_depth_order": proxy_order,
            "proxy_depth_order_agreement": order_match,
            "missing_artifact_count": len(missing),
        },
        "failure_tags": failure_tags,
        "repair_suggestions": suggestions,
    }


def write_manifest(dataset_root: Path, out_path: Path, status: dict) -> None:
    manifest = {
        "schema": "vggt_omega_geometry_review_manifest_v1",
        "dataset_root": str(dataset_root),
        "model": status,
        "intended_output": "metadata/geometry_review_vggt_omega.jsonl",
        "current_output": str(out_path),
        "review_role": "geometry/depth/camera consistency reviewer for VLM loop",
        "pipeline_position": [
            "render artifacts",
            "schema completeness checks",
            "VLM semantic review",
            "VGGT-Omega geometry review",
            "merged validation and repair action selection",
        ],
        "notes": [
            "The model cache contains weights but no official inference code.",
            "Proxy mode is deterministic and does not claim VGGT-Omega inference.",
            "Install facebookresearch/vggt-omega code before enabling full inference.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.out
    if out is None:
        suffix = "manifest.json" if args.mode == "manifest" else "geometry_review_vggt_omega_proxy.jsonl"
        out = args.dataset_root / "metadata" / suffix
    status = model_status(args.model_dir)

    if args.mode == "manifest":
        write_manifest(args.dataset_root, out, status)
        print(out)
        return

    records = load_records(args.dataset_root)
    if args.limit > 0:
        records = records[: args.limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(proxy_review_record(args.dataset_root, record, status), ensure_ascii=False) + "\n")
    print(out)


if __name__ == "__main__":
    main()
