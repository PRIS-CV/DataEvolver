"""
Merge multiple consistent rotation dataset roots into one new parallel root.

The merge is read-only with respect to source roots. Per-object directories are
materialized into the new output root by symlink or copy.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Merge consistent rotation roots")
    parser.add_argument("--source-roots", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--expected-total-objects", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def audit_root(path: Path) -> dict:
    files = [p for p in path.rglob("*") if p.is_file()]
    latest = max((p.stat().st_mtime for p in files), default=0.0)
    return {"root": str(path), "file_count": len(files), "latest_mtime": latest}


def materialize_dir(src: Path, dst: Path, mode: str):
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=True)
    else:
        shutil.copytree(src, dst)


def main():
    args = parse_args()
    source_roots = [Path(root).resolve() for root in args.source_roots]
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "objects").mkdir(parents=True, exist_ok=True)

    source_audits_before = [audit_root(root) for root in source_roots]
    source_summaries: List[dict] = []
    merged_objects: Dict[str, dict] = {}
    total_renders = 0

    for root in source_roots:
        manifest = load_json(root / "manifest.json", default=None)
        if not manifest:
            raise FileNotFoundError(f"manifest.json not found in {root}")
        source_summaries.append(load_json(root / "summary.json", default={}) or {})
        for obj_id, payload in sorted((manifest.get("objects") or {}).items()):
            if obj_id in merged_objects:
                raise ValueError(f"Duplicate object id across merged roots: {obj_id}")
            object_dir = root / "objects" / obj_id
            if not object_dir.exists():
                raise FileNotFoundError(f"Missing object dir: {object_dir}")
            materialize_dir(object_dir, output_root / "objects" / obj_id, args.asset_mode)
            merged_objects[obj_id] = payload
            total_renders += len(payload.get("rotations") or [])

    template_src = source_roots[0] / "scene_template_fixed_camera.json"
    if template_src.exists():
        template_dst = output_root / "scene_template_fixed_camera.json"
        if template_dst.exists() or template_dst.is_symlink():
            template_dst.unlink()
        shutil.copy2(template_src, template_dst)

    if args.expected_total_objects is not None and len(merged_objects) != int(args.expected_total_objects):
        raise ValueError(f"Expected {args.expected_total_objects} objects, got {len(merged_objects)}")

    source_audits_after = [audit_root(root) for root in source_roots]
    summary = {
        "success": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_type": "merged_rotation_consistent",
        "source_roots": [str(root) for root in source_roots],
        "output_root": str(output_root),
        "asset_mode": args.asset_mode,
        "total_objects": len(merged_objects),
        "total_renders": total_renders,
        "source_summaries": source_summaries,
        "source_root_readonly_audit_before": source_audits_before,
        "source_root_readonly_audit_after": source_audits_after,
    }
    save_json(output_root / "summary.json", summary)
    save_json(output_root / "manifest.json", {"summary": summary, "objects": merged_objects})


if __name__ == "__main__":
    main()
