from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
for p in (REPO_ROOT, REPO_ROOT / "pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RENDER_SCRIPT = THIS_DIR / "dual_scene_render.py"
DUAL_ACTION_SPACE = THIS_DIR / "dual_action_space.json"


INIT_ISSUE_TAGS = [
    "none",
    "object_too_small",
    "object_too_large",
    "object_cutoff_left",
    "object_cutoff_right",
    "object_cutoff_top",
    "object_cutoff_bottom",
    "floating_object",
    "ground_intersection",
    "geometry_distortion",
    "partial_occlusion",
    "background_distracting",
    "underexposed",
    "overexposed",
    "shadow_missing",
    "scene_light_mismatch",
    "scale_implausible",
    "physical_implausibility",
]

INIT_HARD_FAIL_TAGS = {
    "geometry_distortion",
    "object_cutoff_left",
    "object_cutoff_right",
    "object_cutoff_top",
    "object_cutoff_bottom",
    "partial_occlusion",
}


def load_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and select an initial dual-object scene placement")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--mesh-a", required=True)
    parser.add_argument("--mesh-b", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--control-state", required=True)
    parser.add_argument("--placement-state-out", required=True)
    parser.add_argument("--blender", default=os.environ.get("ARIS_BLENDER", "blender"))
    parser.add_argument("--engine", default="CYCLES", choices=["EEVEE", "CYCLES"])
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--template", default=None)
    parser.add_argument("--use-vlm-init-review", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vlm-weight", type=float, default=0.55)
    parser.add_argument("--vlm-min-score", type=float, default=0.35)
    parser.add_argument("--ref-a", default=None)
    parser.add_argument("--ref-b", default=None)
    return parser.parse_args()


def nested_get(payload: dict, *keys, default=None):
    cur = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def score_candidate(metadata: dict) -> tuple[float, list[str]]:
    reasons = []
    if metadata.get("success") is not True:
        return -1.0, ["render_failed"]
    score = 0.0
    ground = metadata.get("ground") or {}
    pair = metadata.get("pair") or {}
    scene_size = nested_get(metadata, "scene_bounds", "size", default=[0, 0, 0]) or [0, 0, 0]
    obj_a_size = nested_get(metadata, "objects", "a", "bbox", "size", default=[0, 0, 0]) or [0, 0, 0]
    obj_b_size = nested_get(metadata, "objects", "b", "bbox", "size", default=[0, 0, 0]) or [0, 0, 0]

    if ground.get("name") and ground.get("bounds"):
        score += 0.20
    if float(ground.get("area") or 0.0) > 1.0:
        score += 0.10
    if pair.get("obj1_position") and pair.get("obj2_position"):
        score += 0.15
    if float(pair.get("xy_center_distance") or 0.0) > 0.2:
        score += 0.10

    max_scene_xy = max(float(scene_size[0] or 0.0), float(scene_size[1] or 0.0), 1e-6)
    max_obj = max([float(v or 0.0) for v in obj_a_size + obj_b_size] or [0.0])
    ratio = max_obj / max_scene_xy
    if 0.005 <= ratio <= 0.12:
        score += 0.15
    else:
        reasons.append(f"size_ratio={ratio:.4f}")

    height_offset = abs(float(pair.get("height_offset") or 0.0))
    if height_offset < max(max(float(obj_b_size[2] or 0.0), 1e-6) * 0.6, 0.8):
        score += 0.10
    else:
        reasons.append(f"height_offset={height_offset:.3f}")

    ground_name = str(ground.get("name") or "").lower()
    if "sky" in ground_name or "fog" in ground_name:
        score -= 0.40
        reasons.append("bad_ground_name")
    if "dome" in ground_name and "ground" not in ground_name:
        score -= 0.20
        reasons.append("dome_non_ground")

    cam_loc = nested_get(metadata, "camera", "location", default=None)
    if cam_loc:
        score += 0.10

    return round(score, 4), reasons


def build_init_prompt_appendix(metadata: dict, rule_score: float, rule_reasons: list[str]) -> str:
    context = {
        "rule_score": rule_score,
        "rule_reasons": rule_reasons,
        "ground": metadata.get("ground"),
        "pair": metadata.get("pair"),
        "camera": metadata.get("camera"),
        "scene_bounds": metadata.get("scene_bounds"),
        "object_bboxes": {
            "a": nested_get(metadata, "objects", "a", "bbox", default=None),
            "b": nested_get(metadata, "objects", "b", "bbox", default=None),
        },
    }
    return (
        "Initial-candidate selection task:\n"
        "You are NOT refining the image and you do NOT need to suggest actions. "
        "Select whether this render is a usable starting point for later local VLM refinement.\n"
        "Prioritize these checks: both objects are visible, recognizable, not clipped, not severely occluded; "
        "objects are plausibly on or near the support surface; there is no severe floating, burial, or ground loss; "
        "camera angle is natural; scene/dome/background edges are not badly distorted; lighting is sufficient for later refinement; "
        "scale is roughly plausible.\n"
        "Use the standard review JSON schema. Encode initial-candidate quality mainly in scores.overall, scores.composition, "
        "scores.object_integrity, and issue_tags. Use suggested_actions ['NO_OP'] unless there is one obvious local action.\n"
        "Score guidance: 5 means excellent starting point; 4 usable; 3 borderline but refinable; "
        "2 poor initial candidate; 1 unusable.\n"
        "Current candidate metadata:\n"
        + json.dumps(context, ensure_ascii=False, sort_keys=True)
    )


def run_vlm_init_review(args, candidate: dict, metadata: dict) -> dict:
    rgb_path = candidate.get("rgb_path")
    if not rgb_path or not Path(rgb_path).exists():
        return {
            "usable_for_refinement": False,
            "vlm_init_score": 0.0,
            "failure_tags": ["missing_rgb"],
            "error": "missing_rgb",
        }
    from pipeline.stage5_5_vlm_review import compute_hybrid_score, compute_vlm_score, run_vlm_review

    prompt_appendix = build_init_prompt_appendix(
        metadata,
        float(candidate.get("rule_score", candidate.get("score", 0.0)) or 0.0),
        list(candidate.get("reasons") or []),
    )
    review = run_vlm_review(
        rgb_path=str(rgb_path),
        sample_id=f"{args.sample_id}_init{candidate.get('candidate_idx', 0):02d}",
        round_idx=0,
        active_group="dual_scene",
        az=0,
        el=0,
        prev_rgb_path=None,
        device=args.device,
        prompt_appendix=prompt_appendix,
        issue_tags_whitelist=INIT_ISSUE_TAGS,
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
    hybrid_score, hybrid_route = compute_hybrid_score(review.get("scores", {}), cv_metrics, review_mode="scene_insert")
    vlm_only_score = round(float(compute_vlm_score(review.get("scores", {}))), 4)
    tags = [str(t) for t in review.get("issue_tags") or []]
    usable = bool(hybrid_score >= float(args.vlm_min_score) and not (INIT_HARD_FAIL_TAGS & set(tags)))
    review["hybrid_score"] = hybrid_score
    review["hybrid_route"] = hybrid_route
    review["vlm_only_score"] = vlm_only_score
    review["usable_for_refinement"] = usable
    review["vlm_init_score"] = hybrid_score
    review["failure_tags"] = [t for t in tags if t != "none"]
    return review


def run_candidate(args, idx: int) -> dict:
    out_dir = Path(args.output_dir) / f"candidate_{idx:02d}"
    placement_path = out_dir / "placement_state.json"
    cmd = [
        args.blender,
        args.scene,
        "--background",
        "--python",
        str(RENDER_SCRIPT),
        "--",
        "--obj-a",
        args.mesh_a,
        "--obj-b",
        args.mesh_b,
        "--sample-id",
        f"{args.sample_id}_init{idx:02d}",
        "--output-dir",
        str(out_dir),
        "--control-state",
        args.control_state,
        "--placement-state-out",
        str(placement_path),
        "--resolution",
        str(args.resolution),
        "--engine",
        args.engine,
        "--seed",
        str(args.seed + idx),
    ]
    if args.template:
        cmd += ["--template", args.template]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "blender.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    metadata = load_json(out_dir / "metadata.json", default={}) or {}
    score, reasons = score_candidate(metadata)
    return {
        "candidate_idx": idx,
        "returncode": result.returncode,
        "rule_score": score,
        "score": score,
        "reasons": reasons,
        "render_dir": str(out_dir),
        "rgb_path": metadata.get("rgb_path"),
        "metadata_path": str(out_dir / "metadata.json"),
        "placement_state_path": str(placement_path),
        "success": result.returncode == 0 and metadata.get("success") is True and placement_path.exists(),
    }


def main():
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results = [run_candidate(args, idx) for idx in range(max(1, int(args.candidates)))]
    if args.use_vlm_init_review:
        for candidate in results:
            if not candidate.get("success"):
                candidate["vlm_review"] = {
                    "usable_for_refinement": False,
                    "vlm_init_score": 0.0,
                    "failure_tags": ["render_failed"],
                }
                candidate["final_score"] = -1.0
                continue
            metadata = load_json(candidate["metadata_path"], default={}) or {}
            review = run_vlm_init_review(args, candidate, metadata)
            review_path = Path(candidate["render_dir"]) / "init_vlm_review.json"
            trace = review.pop("vlm_dialogue", None)
            save_json(review_path, review)
            if trace is not None:
                save_json(Path(candidate["render_dir"]) / "init_vlm_trace.json", trace)
            vlm_score = float(review.get("vlm_init_score", 0.0) or 0.0)
            rule_score = float(candidate.get("rule_score", 0.0) or 0.0)
            weight = max(0.0, min(1.0, float(args.vlm_weight)))
            candidate["vlm_review_path"] = str(review_path)
            candidate["vlm_init_score"] = round(vlm_score, 4)
            candidate["usable_for_refinement"] = bool(review.get("usable_for_refinement"))
            candidate["failure_tags"] = review.get("failure_tags") or []
            candidate["final_score"] = round((1.0 - weight) * rule_score + weight * vlm_score, 4) if candidate["usable_for_refinement"] else -1.0
            candidate["score"] = candidate["final_score"]
    else:
        for candidate in results:
            candidate["final_score"] = candidate.get("rule_score", candidate.get("score", -1.0))
            candidate["usable_for_refinement"] = bool(candidate.get("success"))
    viable = [r for r in results if r.get("success")]
    if args.use_vlm_init_review:
        viable = [r for r in viable if r.get("usable_for_refinement")]
    if not viable:
        save_json(root / "init_search_summary.json", {"status": "failed", "results": results})
        raise SystemExit(2)
    viable.sort(key=lambda r: (float(r.get("final_score", r.get("score", -1)) or -1), -int(r.get("candidate_idx") or 0)), reverse=True)
    best = viable[0]
    placement = load_json(best["placement_state_path"], default={}) or {}
    placement["init_search"] = {
        "selected_candidate": best["candidate_idx"],
        "score": best.get("final_score", best.get("score")),
        "rule_score": best.get("rule_score"),
        "vlm_init_score": best.get("vlm_init_score"),
        "use_vlm_init_review": bool(args.use_vlm_init_review),
        "reasons": best["reasons"],
        "failure_tags": best.get("failure_tags", []),
        "render_dir": best["render_dir"],
        "rgb_path": best.get("rgb_path"),
        "vlm_review_path": best.get("vlm_review_path"),
    }
    save_json(args.placement_state_out, placement)
    summary = {"status": "selected", "best": best, "results": results, "placement_state_out": args.placement_state_out}
    save_json(root / "init_search_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
