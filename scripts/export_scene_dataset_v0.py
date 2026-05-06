#!/usr/bin/env python3
"""
export_scene_dataset_v0.py

Export scene_dataset_v0 from full pilot results.

Filter rules (all must pass):
  1. exit_reason in {accepted_after_try, mid_no_improve}
  2. obj_id != obj_003
  3. contact_gap <= contact_gap_threshold (default 0.005)
  4. penetration_depth == 0
  5. is_out_of_support_bounds == false

Usage:
  python3 scripts/export_scene_dataset_v0.py \
    --pilot-dir pipeline/data/evolution_scene_v7_full_pilot \
    --output-dir pipeline/data/scene_dataset_v0 \
    --contact-gap-threshold 0.005
"""

import argparse
import json
import os
import shutil

VALID_EXIT_REASONS = {"accepted_after_try", "mid_no_improve", "improved_after_try"}
PRIORITY_OBJECTS = {"obj_001", "obj_005", "obj_007", "obj_008", "obj_009", "obj_010"}
CAUTIOUS_OBJECTS = {"obj_002", "obj_004", "obj_006"}
EXCLUDED_OBJECTS = {"obj_003"}


def quality_tier(score: float) -> str:
    if score >= 0.70:
        return "A"
    elif score >= 0.64:
        return "B"
    elif score >= 0.60:
        return "C"
    return "D"


def obj_group(obj_id: str) -> str:
    if obj_id in PRIORITY_OBJECTS:
        return "priority"
    if obj_id in CAUTIOUS_OBJECTS:
        return "cautious"
    return "excluded"


def get_best_render_dir(obj_dir: str, result: dict) -> str:
    """Return the render dir with the best final renders (attempt1 if exists, else last probe)."""
    exit_reason = result.get("exit_reason", "")
    if exit_reason in ("accepted_after_try", "improved_after_try"):
        attempt_dir = os.path.join(obj_dir, "attempt1_renders")
        if os.path.isdir(attempt_dir):
            return attempt_dir
    for probe_label in ["probe2_renders", "probe1_renders", "probe0_renders"]:
        d = os.path.join(obj_dir, probe_label)
        if os.path.isdir(d):
            return d
    return None


def export_object(obj_id: str, obj_dir: str, result: dict, output_dir: str,
                  reference_images_dir: str, gap_thresh: float):
    prog = result.get("diagnostics", {}).get("programmatic_physics", {})
    contact_gap = float(prog.get("contact_gap", 0.0) or 0.0)
    penetration = float(prog.get("penetration_depth", 0.0) or 0.0)
    out_of_bounds = bool(prog.get("is_out_of_support_bounds", False))

    # Apply filter rules
    exit_reason = result.get("exit_reason", "")
    if exit_reason not in VALID_EXIT_REASONS:
        return None, f"skip: exit_reason={exit_reason}"
    if obj_id in EXCLUDED_OBJECTS:
        return None, "skip: excluded object"
    if contact_gap > gap_thresh:
        return None, f"skip: contact_gap={contact_gap:.4f} > {gap_thresh}"
    if penetration > 0:
        return None, f"skip: penetration={penetration:.4f}"
    if out_of_bounds:
        return None, "skip: out_of_support_bounds"

    # Find best render dir
    render_src = get_best_render_dir(obj_dir, result)
    if render_src is None:
        return None, "skip: no render dir found"

    obj_render_dir = os.path.join(render_src, obj_id)
    if not os.path.isdir(obj_render_dir):
        return None, f"skip: obj render dir missing {obj_render_dir}"

    all_pngs = sorted(f for f in os.listdir(obj_render_dir) if f.endswith(".png"))
    rgb_files = [f for f in all_pngs if not f.endswith("_mask.png")]
    mask_files = [f for f in all_pngs if f.endswith("_mask.png")]

    if not rgb_files:
        return None, "skip: no RGB renders found"

    # Create output dirs
    out_obj = os.path.join(output_dir, obj_id)
    out_renders = os.path.join(out_obj, "renders")
    out_masks = os.path.join(out_obj, "masks")
    os.makedirs(out_renders, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    # Copy renders and masks
    for fname in rgb_files:
        shutil.copy2(os.path.join(obj_render_dir, fname), os.path.join(out_renders, fname))
    for fname in mask_files:
        shutil.copy2(os.path.join(obj_render_dir, fname), os.path.join(out_masks, fname))

    # Copy reference image
    ref_src = os.path.join(reference_images_dir, f"{obj_id}.png")
    if os.path.exists(ref_src):
        shutil.copy2(ref_src, os.path.join(out_obj, "reference.png"))
    else:
        print(f"  [warn] reference image not found: {ref_src}")

    # Copy result json
    shutil.copy2(
        os.path.join(obj_dir, "scene_evolution_result.json"),
        os.path.join(out_obj, "scene_evolution_result.json"),
    )

    # Write metadata.json
    metadata = {
        "obj_id": obj_id,
        "contact_gap": contact_gap,
        "penetration_depth": penetration,
        "is_out_of_support_bounds": out_of_bounds,
        "is_floating": bool(prog.get("is_floating", False)),
        "bbox_height": prog.get("bbox_height"),
        "rgb_count": len(rgb_files),
        "mask_count": len(mask_files),
        "render_source": os.path.basename(render_src),
    }
    with open(os.path.join(out_obj, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    final_score = result.get("final_hybrid", 0.0)
    confirmed_score = result.get("confirmed_score", 0.0)
    state_log = result.get("state_log", [])
    action_taken = state_log[0].get("action_taken", "None") if state_log else "None"
    lighting_diag = result.get("diagnostics", {}).get("lighting_diagnosis", "unknown")

    entry = {
        "obj_id": obj_id,
        "confirmed_score": round(confirmed_score, 4),
        "final_score": round(final_score, 4),
        "delta": round(final_score - confirmed_score, 4),
        "action_taken": action_taken,
        "exit_reason": exit_reason,
        "lighting_diagnosis": lighting_diag,
        "contact_gap": contact_gap,
        "is_floating": bool(prog.get("is_floating", False)),
        "quality_tier": quality_tier(final_score),
        "group": obj_group(obj_id),
    }
    return entry, "ok"


def main():
    parser = argparse.ArgumentParser(description="Export scene_dataset_v0 from full pilot")
    parser.add_argument("--pilot-dir", required=True, help="Path to evolution full pilot directory")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory")
    parser.add_argument(
        "--contact-gap-threshold", type=float, default=0.005,
        help="Max allowed contact_gap (default 0.005)",
    )
    parser.add_argument("--reference-images-dir", default=None,
                        help="Reference images dir (default: <pilot-dir>/../../images)")
    args = parser.parse_args()

    pilot_dir = os.path.abspath(args.pilot_dir)
    output_dir = os.path.abspath(args.output_dir)
    gap_thresh = args.contact_gap_threshold

    if args.reference_images_dir:
        ref_dir = os.path.abspath(args.reference_images_dir)
    else:
        ref_dir = os.path.join(os.path.dirname(os.path.dirname(pilot_dir)), "images")

    print(f"Pilot dir:     {pilot_dir}")
    print(f"Output dir:    {output_dir}")
    print(f"Ref images:    {ref_dir}")
    print(f"Gap threshold: {gap_thresh}")
    os.makedirs(output_dir, exist_ok=True)

    index_entries = []
    obj_ids = sorted(
        d for d in os.listdir(pilot_dir)
        if os.path.isdir(os.path.join(pilot_dir, d)) and d.startswith("obj_")
    )

    for obj_id in obj_ids:
        obj_dir = os.path.join(pilot_dir, obj_id)
        result_path = os.path.join(obj_dir, "scene_evolution_result.json")
        if not os.path.exists(result_path):
            print(f"[{obj_id}] SKIP: no scene_evolution_result.json")
            continue

        with open(result_path) as f:
            result = json.load(f)

        entry, status = export_object(obj_id, obj_dir, result, output_dir, ref_dir, gap_thresh)
        if entry is None:
            print(f"[{obj_id}] FILTERED ({status})")
        else:
            index_entries.append(entry)
            tier = entry["quality_tier"]
            print(
                f"[{obj_id}] EXPORTED  tier={tier} final={entry['final_score']:.4f}"
                f" delta={entry['delta']:+.4f} action={entry['action_taken']}"
            )

    tier_counts = {}
    for e in index_entries:
        t = e["quality_tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1

    index = {
        "objects": index_entries,
        "summary": {
            "total": len(index_entries),
            "tier_A": tier_counts.get("A", 0),
            "tier_B": tier_counts.get("B", 0),
            "tier_C": tier_counts.get("C", 0),
            "contact_gap_threshold": gap_thresh,
        },
    }

    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n=== Export complete: {len(index_entries)} objects ===")
    print(f"  Tier A (>=0.70):    {tier_counts.get('A', 0)}")
    print(f"  Tier B (0.64-0.70): {tier_counts.get('B', 0)}")
    print(f"  Tier C (0.60-0.64): {tier_counts.get('C', 0)}")
    print(f"  index.json -> {index_path}")


if __name__ == "__main__":
    main()
