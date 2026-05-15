"""
Build a curated 4-angle dataset snapshot from pair-level evolution results.

This script selects the best available round for each obj_xxx_yawYYY pair,
copies the chosen RGB/mask/metadata/control files into a compact dataset
directory, and writes root/object/pair manifests for downstream use.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from export_scene_multiview_from_pair_evolution import (
    collect_round_records,
    discover_pair_dirs,
    select_best_round,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build curated rotation4 dataset snapshot")
    parser.add_argument("--source-root", required=True, help="Pair-evolution root")
    parser.add_argument("--output-dir", required=True, help="Dataset output directory")
    parser.add_argument(
        "--pair-names",
        nargs="*",
        default=None,
        help="Optional subset of pair names like obj_001_yaw000",
    )
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


def _copy_if_exists(src: Optional[Path], dst: Path) -> Optional[str]:
    if src is None or not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _format_rot(rotation_deg: int) -> str:
    return f"yaw{int(rotation_deg):03d}"


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _relative_str(path: Optional[Path], base: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def build_dataset(source_root: Path, output_root: Path, pair_names: Optional[List[str]]) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    objects_root = output_root / "objects"
    pairs_root = output_root / "pairs"
    objects_root.mkdir(parents=True, exist_ok=True)
    pairs_root.mkdir(parents=True, exist_ok=True)

    pair_records = discover_pair_dirs(source_root, pair_names)
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    objects_manifest: Dict[str, dict] = {}
    pairs_manifest: Dict[str, dict] = {}
    csv_rows: List[dict] = []
    skipped_pairs: Dict[str, dict] = {}

    for pair_record in pair_records:
        pair_dir = Path(pair_record["pair_dir"])
        obj_id = str(pair_record["obj_id"])
        rotation_deg = int(pair_record["rotation_deg"])
        pair_name = str(pair_record["pair_name"])

        records = collect_round_records(pair_dir, obj_id)
        best_record = select_best_round(records)
        if best_record is None:
            skipped_pairs[pair_name] = {
                "reason": "best_round_not_found",
                "pair_dir": str(pair_dir),
            }
            continue

        round_idx = int(best_record["round_idx"])
        render_dir = pair_dir / f"round{round_idx:02d}_renders" / obj_id
        image_src = render_dir / "az000_el+00.png"
        mask_src = render_dir / "az000_el+00_mask.png"
        render_meta_src = render_dir / "metadata.json"
        control_src = Path(best_record["control_state_path"]) if best_record.get("control_state_path") else None
        agg_src = Path(best_record["agg_path"]) if best_record.get("agg_path") else None
        agent_src = Path(best_record["agent_path"]) if best_record.get("agent_path") else None
        decision_src = Path(best_record["decision_path"]) if best_record.get("decision_path") else None

        if not image_src.exists():
            skipped_pairs[pair_name] = {
                "reason": "selected_image_missing",
                "pair_dir": str(pair_dir),
                "selected_round_idx": round_idx,
            }
            continue

        obj_dir = objects_root / obj_id
        obj_dir.mkdir(parents=True, exist_ok=True)
        rot_slug = _format_rot(rotation_deg)
        pair_dir_out = pairs_root / pair_name
        pair_dir_out.mkdir(parents=True, exist_ok=True)

        rgb_out = Path(_copy_if_exists(image_src, obj_dir / f"{rot_slug}.png"))
        mask_out = Path(_copy_if_exists(mask_src, obj_dir / f"{rot_slug}_mask.png")) if mask_src.exists() else None
        render_meta_out = Path(_copy_if_exists(render_meta_src, obj_dir / f"{rot_slug}_render_metadata.json")) if render_meta_src.exists() else None
        control_out = Path(_copy_if_exists(control_src, obj_dir / f"{rot_slug}_control.json")) if control_src and control_src.exists() else None
        agg_out = Path(_copy_if_exists(agg_src, pair_dir_out / "review_agg.json")) if agg_src and agg_src.exists() else None
        agent_out = Path(_copy_if_exists(agent_src, pair_dir_out / "agent_round.json")) if agent_src and agent_src.exists() else None
        decision_out = Path(_copy_if_exists(decision_src, pair_dir_out / "decision.json")) if decision_src and decision_src.exists() else None

        render_meta = load_json(render_meta_src, default={}) or {}
        pair_payload = {
            "pair_name": pair_name,
            "obj_id": obj_id,
            "rotation_deg": rotation_deg,
            "source_pair_dir": str(pair_dir),
            "selected_round": best_record,
            "selection": {
                "best_round_idx": round_idx,
                "hybrid_score": _safe_float(best_record.get("hybrid_score")),
                "detected_verdict": best_record.get("detected_verdict"),
                "selection_reason": best_record.get("selection_reason"),
                "asset_viability": best_record.get("asset_viability"),
                "structure_consistency": best_record.get("structure_consistency"),
                "issue_tags": best_record.get("issue_tags") or [],
                "suggested_actions": best_record.get("suggested_actions") or [],
            },
            "dataset_files": {
                "rgb": _relative_str(rgb_out, output_root),
                "mask": _relative_str(mask_out, output_root),
                "render_metadata": _relative_str(render_meta_out, output_root),
                "control_state": _relative_str(control_out, output_root),
                "review_agg": _relative_str(agg_out, output_root),
                "agent_round": _relative_str(agent_out, output_root),
                "decision": _relative_str(decision_out, output_root),
            },
            "source_files": {
                "rgb": str(image_src),
                "mask": str(mask_src) if mask_src.exists() else None,
                "render_metadata": str(render_meta_src) if render_meta_src.exists() else None,
                "control_state": str(control_src) if control_src and control_src.exists() else None,
                "review_agg": str(agg_src) if agg_src and agg_src.exists() else None,
                "agent_round": str(agent_src) if agent_src and agent_src.exists() else None,
                "decision": str(decision_src) if decision_src and decision_src.exists() else None,
            },
            "render_metadata_summary": {
                "camera_mode": render_meta.get("camera_mode"),
                "render_engine": render_meta.get("render_engine"),
                "resolution": render_meta.get("resolution"),
                "frames": len(render_meta.get("frames", [])),
            },
        }
        save_json(pair_dir_out / "pair_meta.json", pair_payload)
        pairs_manifest[pair_name] = pair_payload

        obj_entry = objects_manifest.setdefault(
            obj_id,
            {
                "obj_id": obj_id,
                "angles_present": [],
                "complete_rotation4": False,
                "rotations": {},
            },
        )
        obj_entry["rotations"][rot_slug] = {
            "pair_name": pair_name,
            "rotation_deg": rotation_deg,
            "rgb": _relative_str(rgb_out, output_root),
            "mask": _relative_str(mask_out, output_root),
            "render_metadata": _relative_str(render_meta_out, output_root),
            "control_state": _relative_str(control_out, output_root),
            "hybrid_score": _safe_float(best_record.get("hybrid_score")),
            "detected_verdict": best_record.get("detected_verdict"),
            "selection_reason": best_record.get("selection_reason"),
        }
        obj_entry["angles_present"] = sorted(obj_entry["rotations"].keys())
        obj_entry["complete_rotation4"] = len(obj_entry["rotations"]) == 4

        csv_rows.append(
            {
                "pair_name": pair_name,
                "obj_id": obj_id,
                "rotation_deg": rotation_deg,
                "best_round_idx": round_idx,
                "hybrid_score": _safe_float(best_record.get("hybrid_score")),
                "detected_verdict": best_record.get("detected_verdict") or "",
                "selection_reason": best_record.get("selection_reason") or "",
                "asset_viability": best_record.get("asset_viability") or "",
                "structure_consistency": best_record.get("structure_consistency") or "",
                "issue_tags": "|".join(best_record.get("issue_tags") or []),
                "rgb_path": _relative_str(rgb_out, output_root),
                "mask_path": _relative_str(mask_out, output_root),
                "control_state_path": _relative_str(control_out, output_root),
                "source_pair_dir": str(pair_dir),
            }
        )

    for obj_id, payload in objects_manifest.items():
        save_json(objects_root / obj_id / "object_manifest.json", payload)

    csv_rows = sorted(csv_rows, key=lambda item: item["pair_name"])
    csv_path = output_root / "pairs_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_name",
                "obj_id",
                "rotation_deg",
                "best_round_idx",
                "hybrid_score",
                "detected_verdict",
                "selection_reason",
                "asset_viability",
                "structure_consistency",
                "issue_tags",
                "rgb_path",
                "mask_path",
                "control_state_path",
                "source_pair_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    total_pairs = len(pair_records)
    selected_pairs = len(pairs_manifest)
    keep_pairs = sum(1 for item in pairs_manifest.values() if item["selection"]["detected_verdict"] == "keep")
    complete_objects = sum(1 for item in objects_manifest.values() if item["complete_rotation4"])
    score_values = [
        item["selection"]["hybrid_score"]
        for item in pairs_manifest.values()
        if item["selection"]["hybrid_score"] is not None
    ]

    summary = {
        "dataset_name": output_root.name,
        "created_at": created_at,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "total_pairs_requested": total_pairs,
        "total_pairs_selected": selected_pairs,
        "total_pairs_skipped": len(skipped_pairs),
        "keep_pairs": keep_pairs,
        "non_keep_pairs": selected_pairs - keep_pairs,
        "total_objects": len(objects_manifest),
        "complete_objects": complete_objects,
        "incomplete_objects": len(objects_manifest) - complete_objects,
        "avg_best_hybrid_score": round(sum(score_values) / len(score_values), 4) if score_values else None,
        "expected_rotations": [0, 90, 180, 270],
        "skipped_pairs": skipped_pairs,
    }
    save_json(output_root / "summary.json", summary)

    manifest = {
        "summary": summary,
        "objects": dict(sorted(objects_manifest.items())),
        "pairs": dict(sorted(pairs_manifest.items())),
    }
    save_json(output_root / "manifest.json", manifest)
    return manifest


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_dir).resolve()
    manifest = build_dataset(source_root, output_root, args.pair_names)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
