#!/usr/bin/env python3
"""Materialize only accepted gate pairs into a clean pair-evolution root.

This is the bridge between:
  run_vlm_quality_gate_loop.py
and:
  export_rotation8_from_best_object_state.py

The output root contains only accepted `obj_xxx_yawYYY` directories (as symlink
or copy), so downstream export scripts cannot accidentally pick rejected pairs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build accepted-only pair root from gate manifest")
    parser.add_argument("--gate-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def materialize(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    if mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=True)
    else:
        shutil.copytree(src, dst)


def main() -> None:
    args = parse_args()
    gate_root = Path(args.gate_root).resolve()
    output_root = Path(args.output_dir).resolve()
    manifest = load_json(gate_root / "gate_manifest.json", default={}) or {}
    accepted = list(manifest.get("accepted") or [])
    if not accepted:
        raise SystemExit(f"No accepted pairs found in {gate_root / 'gate_manifest.json'}")

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output dir is not empty, pass --overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    entries = []
    for item in accepted:
        pair_dir = Path(item["pair_dir"]).resolve()
        dst = output_root / pair_dir.name
        materialize(pair_dir, dst, args.asset_mode)
        entry = {
            "obj_id": item.get("obj_id"),
            "rotation_deg": item.get("rotation_deg"),
            "pair_dir": str(dst),
            "source_pair_dir": str(pair_dir),
            "accepted_round_idx": item.get("accepted_round_idx"),
            "accepted_score": item.get("accepted_score"),
            "accepted_threshold": item.get("accepted_threshold"),
        }
        entries.append(entry)

    save_json(
        output_root / "accepted_pairs_manifest.json",
        {
            "success": True,
            "gate_root": str(gate_root),
            "output_root": str(output_root),
            "asset_mode": args.asset_mode,
            "accepted_count": len(entries),
            "pairs": entries,
        },
    )
    print(json.dumps({"accepted_count": len(entries), "output_root": str(output_root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
