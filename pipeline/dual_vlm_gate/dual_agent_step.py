from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
for p in (REPO_ROOT, REPO_ROOT / "pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pipeline.stage5_5_vlm_review import compute_hybrid_score, run_vlm_review  # noqa: E402
from pipeline.dual_vlm_gate.dual_actions import (  # noqa: E402
    apply_actions,
    default_control_state,
    load_json,
    save_json,
)


DEFAULT_BLENDER = os.environ.get("ARIS_BLENDER", "blender")
DEFAULT_SCENE = os.environ.get("ARIS_DUAL_SCENE", "<large-file-root>")
DUAL_ACTION_SPACE = THIS_DIR / "dual_action_space.json"
DUAL_RENDER_SCRIPT = THIS_DIR / "dual_scene_render.py"


DUAL_ISSUE_TAGS = [
    "none",
    "object_a_too_large",
    "object_a_too_small",
    "object_b_too_large",
    "object_b_too_small",
    "object_too_large",
    "object_too_small",
    "objects_too_close",
    "objects_too_far",
    "object_a_floating",
    "object_b_floating",
    "floating_object",
    "ground_intersection",
    "ground_intersection_visible",
    "object_a_intersecting",
    "object_b_intersecting",
    "scale_implausible",
    "bad_composition",
    "occlusion",
    "underexposed",
    "overexposed",
    "shadow_missing",
    "weak_contact_shadow",
    "flat_lighting",
    "scene_light_mismatch",
    "washed_out",
    "color_too_pale",
    "flat_material",
    "low_material_contrast",
    "object_cropped",
    "relation_unclear",
    "too_top_down",
    "geometry_distortion",
    "texture_mismatch",
    "scene_bad_geometry",
    "scene_bad_lighting",
    "ground_unreliable",
    "object_scene_scale_unrecoverable",
    "bad_scene_color_cast",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run one dual-object VLM gate step")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--obj-a-id", default="object_a")
    parser.add_argument("--obj-b-id", default="object_b")
    parser.add_argument("--mesh-a", required=True)
    parser.add_argument("--mesh-b", required=True)
    parser.add_argument("--ref-a", default=None)
    parser.add_argument("--ref-b", default=None)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--round-idx", type=int, default=0)
    parser.add_argument("--actions", nargs="*", default=[])
    parser.add_argument("--step-scale", type=float, default=1.0)
    parser.add_argument("--control-state-in", default=None)
    parser.add_argument("--placement-state", default=None)
    parser.add_argument("--placement-state-out", default=None)
    parser.add_argument("--prev-rgb", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blender", default=DEFAULT_BLENDER)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--engine", default="CYCLES", choices=["EEVEE", "CYCLES"])
    parser.add_argument("--template", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def compact_json(payload, max_chars: int = 2200) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[:max_chars] + "...<truncated>"
    return text


def build_prompt_appendix(
    *,
    obj_a_id: str,
    obj_b_id: str,
    ref_a: Optional[str],
    ref_b: Optional[str],
    metadata: dict,
    state: dict,
    prev_review: dict | None,
) -> str:
    allowed_actions = ", ".join(json.loads(DUAL_ACTION_SPACE.read_text())["groups"]["dual_scene"]["actions"].keys())
    parts = [
        "Dual-object review context:",
        "Images are ordered as: current render, reference image for object A if present, reference image for object B if present, previous render if present.",
        f"Object A id: {obj_a_id}; reference A: {ref_a or 'missing'}",
        f"Object B id: {obj_b_id}; reference B: {ref_b or 'missing'}",
        "Judge both object identities, relative scale, pair spacing, occlusion, grounding/contact, lighting, and final composition.",
        "Prefer camera scale/framing before changing mesh scale when objects are too small.",
        "Use scene-light normalization actions when the scene is overlit, washed out, shadowless, or has flat lighting.",
        "Material saturation/value/contrast actions are allowed when source textures are present but appear pale, gray, or washed out in the render.",
        "For floating/contact/intersection problems, prefer GROUND_SNAP actions before contact-shadow or AO actions.",
        "For scale_implausible/object_too_large, prefer object scale-down actions before camera or light changes.",
        "If the scene itself is unusable for local refinement, tag scene_bad_geometry, scene_bad_lighting, ground_unreliable, or object_scene_scale_unrecoverable.",
        "Use these dual issue tags when relevant: " + ", ".join(DUAL_ISSUE_TAGS),
        "Use only these suggested_actions: " + allowed_actions,
        "Current dual control state:",
        compact_json(state, max_chars=1600),
        "Current render metadata:",
        compact_json(
            {
                "objects": metadata.get("objects"),
                "pair": metadata.get("pair"),
                "camera": metadata.get("camera"),
                "scene_bounds": metadata.get("scene_bounds"),
            },
            max_chars=2400,
        ),
    ]
    if prev_review:
        parts.extend(["Previous round review summary:", compact_json(prev_review, max_chars=1400)])
    parts.append(
        "Return the normal VLM JSON schema. For issue_tags and suggested_actions, use the dual tags/actions above."
    )
    return "\n".join(parts)


def run_blender_render(args, round_dir: Path, control_state_path: Path) -> tuple[bool, dict]:
    cmd = [
        args.blender,
        args.scene,
        "--background",
        "--python",
        str(DUAL_RENDER_SCRIPT),
        "--",
        "--obj-a",
        args.mesh_a,
        "--obj-b",
        args.mesh_b,
        "--sample-id",
        args.sample_id,
        "--output-dir",
        str(round_dir),
        "--control-state",
        str(control_state_path),
        "--resolution",
        str(args.resolution),
        "--engine",
        args.engine,
        "--seed",
        str(args.seed),
    ]
    if args.placement_state:
        cmd += ["--placement-state-in", args.placement_state]
    if args.placement_state_out:
        cmd += ["--placement-state-out", args.placement_state_out]
    if args.template:
        cmd += ["--template", args.template]
    log_path = round_dir / "blender.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    metadata = load_json(round_dir / "metadata.json", default={}) or {}
    return result.returncode == 0 and metadata.get("success") is True, metadata


def run_step(args) -> dict:
    started = time.time()
    pair_dir = Path(args.output_dir).resolve()
    states_dir = pair_dir / "states"
    reviews_dir = pair_dir / "reviews"
    round_tag = f"round{args.round_idx:02d}"
    round_dir = pair_dir / f"{round_tag}_renders"
    states_dir.mkdir(parents=True, exist_ok=True)
    reviews_dir.mkdir(parents=True, exist_ok=True)
    round_dir.mkdir(parents=True, exist_ok=True)

    if args.control_state_in:
        base_state = load_json(args.control_state_in, default=default_control_state()) or default_control_state()
    else:
        auto_prev = states_dir / f"round{args.round_idx - 1:02d}.json"
        base_state = load_json(auto_prev, default=default_control_state()) if args.round_idx > 0 else default_control_state()
    input_state_path = states_dir / f"{round_tag}_input.json"
    state_path = states_dir / f"{round_tag}.json"
    save_json(input_state_path, base_state)
    state, action_results = apply_actions(args.actions, base_state, step_scale=args.step_scale)
    save_json(state_path, state)

    ok, metadata = run_blender_render(args, round_dir, state_path)
    rgb_path = round_dir / f"{args.sample_id}.png"
    result_path = pair_dir / f"agent_{round_tag}.json"
    if not ok or not rgb_path.exists():
        result = {
            "success": False,
            "sample_id": args.sample_id,
            "round_idx": args.round_idx,
            "render_dir": str(round_dir),
            "state_out": str(state_path),
            "placement_state": args.placement_state,
            "placement_state_out": args.placement_state_out,
            "actions": action_results,
            "metadata": metadata,
            "error": "dual_render_failed",
            "elapsed_seconds": round(time.time() - started, 3),
        }
        save_json(result_path, result)
        return result

    prev_review = None
    if args.round_idx > 0:
        prev_review = load_json(reviews_dir / f"{args.sample_id}_r{args.round_idx - 1:02d}_agg.json", default=None)
    prompt_appendix = build_prompt_appendix(
        obj_a_id=args.obj_a_id,
        obj_b_id=args.obj_b_id,
        ref_a=args.ref_a,
        ref_b=args.ref_b,
        metadata=metadata,
        state=state,
        prev_review=prev_review,
    )
    review = run_vlm_review(
        rgb_path=str(rgb_path),
        sample_id=args.sample_id,
        round_idx=args.round_idx,
        active_group="dual_scene",
        az=0,
        el=0,
        prev_rgb_path=args.prev_rgb,
        device=args.device,
        prompt_appendix=prompt_appendix,
        issue_tags_whitelist=DUAL_ISSUE_TAGS,
        reference_image_path=args.ref_a,
        pseudo_reference_path=args.ref_b,
        action_space_path=str(DUAL_ACTION_SPACE),
        review_mode="scene_insert",
        use_freeform_first=True,
        enable_thinking=True,
    )
    cv_metrics = {
        "cv_score": 0.5,
        "exposure_score": 0.5,
        "sharpness_score": 0.5,
        "framing_score": 0.5,
        "mask_available": False,
    }
    hybrid_score, route = compute_hybrid_score(review.get("scores", {}), cv_metrics, review_mode="scene_insert")
    review["hybrid_score"] = round(float(hybrid_score), 4)
    review["hybrid_route"] = route
    review["dual_metadata_path"] = str(round_dir / "metadata.json")
    review_path = reviews_dir / f"{args.sample_id}_r{args.round_idx:02d}_agg.json"
    trace_path = reviews_dir / f"{args.sample_id}_r{args.round_idx:02d}_trace.json"
    trace = review.pop("vlm_dialogue", None)
    save_json(review_path, review)
    if trace is not None:
        save_json(trace_path, trace)

    result = {
        "success": True,
        "sample_id": args.sample_id,
        "obj_a_id": args.obj_a_id,
        "obj_b_id": args.obj_b_id,
        "round_idx": args.round_idx,
        "pair_dir": str(pair_dir),
        "render_dir": str(round_dir),
        "rgb_path": str(rgb_path),
        "prev_rgb": args.prev_rgb,
        "state_in": args.control_state_in,
        "state_out": str(state_path),
        "placement_state": args.placement_state,
        "placement_state_out": args.placement_state_out,
        "review_agg_path": str(review_path),
        "hybrid_score": review.get("hybrid_score"),
        "vlm_route": review.get("vlm_route"),
        "issue_tags": review.get("issue_tags"),
        "suggested_actions": review.get("suggested_actions"),
        "actions": action_results,
        "metadata_path": str(round_dir / "metadata.json"),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    save_json(result_path, result)
    return result


def main():
    args = parse_args()
    result = run_step(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
