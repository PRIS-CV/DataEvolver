"""
run_evolution_loop.py - VLM-feedback-driven render self-evolution.
"""

import os
import sys
import json
import copy
import shutil
import argparse
import subprocess
from collections import Counter
from pathlib import Path
from typing import Optional, List

# --- v5/v6: PIPELINE_STATE.json persistence ---------------------------------
import datetime as _dt

_PIPELINE_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PIPELINE_STATE.json")


def _load_pipeline_state():
    if os.path.exists(_PIPELINE_STATE_PATH):
        with open(_PIPELINE_STATE_PATH) as f:
            return json.load(f)
    return None


def _save_pipeline_state(state):
    state["timestamp"] = _dt.datetime.now().isoformat()
    with open(_PIPELINE_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _init_pipeline_state(obj_ids, output_dir):
    return {
        "task": f"evolution_{os.path.basename(output_dir)}",
        "phase": "running",
        "status": "in_progress",
        "codex_review_threadId": None,
        "objects": {oid: {"status": "pending"} for oid in obj_ids},
        "timestamp": _dt.datetime.now().isoformat(),
    }


def _update_obj_state(pstate, obj_id, result):
    pstate["objects"][obj_id] = {
        "status": "done",
        "final_score": result.get("final_hybrid", 0),
        "exit_reason": result.get("exit_reason", "unknown"),
        "accepted": result.get("accepted", False),
        "probes_run": result.get("probes_run", 0),
    }
    done = sum(1 for v in pstate["objects"].values() if v.get("status") == "done")
    total = len(pstate["objects"])
    if done >= total:
        pstate["status"] = "completed"
    _save_pipeline_state(pstate)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "pipeline"))

from stage5_5_vlm_review import review_object
from stage5_6_feedback_apply import (  # noqa: E402
    apply_feedback, load_action_space, default_control_state,
    save_control_state, ActionResult,
)

# Mesh-level reject evidence tags
MESH_EVIDENCE_TAGS = {
    "mesh_interpenetration",
    "geometry_distortion",
    "ground_intersection",
}

# diagnosis -> issue_tag mapping
DIAGNOSIS_TO_TAG = {
    "flat_no_rim": "flat_lighting",
    "flat_low_contrast": "weak_subject_separation",
    "underexposed_global": "underexposed",
    "underexposed_shadow": "harsh_shadow",
    "harsh_shadow_key": "harsh_shadow",
}

GROUP_ORDER = ["lighting", "camera", "object", "scene", "material"]

BLENDER_BIN = os.environ.get("BLENDER_BIN", "/home/wuwenzhuo/blender-4.24/blender")
RENDER_SCRIPT = os.path.join(SCRIPT_DIR, "pipeline", "stage4_blender_render.py")

PROFILE_DEFAULTS = {
    "dataset_name": "default",
    "task_type": "generic",
    "accept_threshold": 0.80,
    "reject_threshold": 0.40,
    "preserve_score_threshold": 0.77,
    "explore_threshold": 0.68,
    "stability_threshold": 0.03,
    "unstable_variance_limit": 0.05,
    "mid_budget": 1,
    "low_budget": 2,
    "improve_eps": 0.01,
    "max_rounds": 5,
    "patience": 4,
    "review_view_policy": "canonical_4",
    "action_whitelist": None,
    "issue_tags_whitelist": None,
    "prompt_appendix": "",
    "reference_images_dir": None,
}


def load_profile(path):
    if path is None:
        return dict(PROFILE_DEFAULTS)
    with open(path) as f:
        data = json.load(f)
    return {**PROFILE_DEFAULTS, **data}


def _resolve_reference_image(obj_id: str, profile_cfg: dict, renders_dir: str) -> Optional[str]:
    """Resolve T2I reference image path. Returns absolute path or None."""
    ref_dir = profile_cfg.get("reference_images_dir", None)
    if ref_dir is None:
        ref_dir = os.path.join(os.path.dirname(os.path.abspath(renders_dir)), "images")
    if not os.path.isabs(ref_dir):
        ref_dir = os.path.join(SCRIPT_DIR, ref_dir)
    ref_path = os.path.join(ref_dir, f"{obj_id}.png")
    return ref_path if os.path.exists(ref_path) else None


def _aggregate_baseline_diagnostics(diag_list: list) -> dict:
    """
    Aggregate diagnostics across 2-3 baseline probes.

    Rules:
    - structure_consistency: worst-case
    - physics_consistency: worst-case
    - color_consistency: majority vote
    - lighting_diagnosis: majority vote
    """
    if not diag_list:
        return {}

    struct_vals = [d.get("structure_consistency", "good") for d in diag_list]
    if "major_mismatch" in struct_vals:
        agg_structure = "major_mismatch"
    elif "minor_mismatch" in struct_vals:
        agg_structure = "minor_mismatch"
    else:
        agg_structure = "good"

    phys_vals = [d.get("physics_consistency", "good") for d in diag_list]
    if "major_issue" in phys_vals:
        agg_physics = "major_issue"
    elif "minor_issue" in phys_vals:
        agg_physics = "minor_issue"
    else:
        agg_physics = "good"

    color_vals = [d.get("color_consistency", "good") for d in diag_list]
    agg_color = Counter(color_vals).most_common(1)[0][0]

    light_vals = [d.get("lighting_diagnosis", "good") for d in diag_list]
    agg_lighting = Counter(light_vals).most_common(1)[0][0]

    return {
        "structure_consistency": agg_structure,
        "physics_consistency": agg_physics,
        "color_consistency": agg_color,
        "lighting_diagnosis": agg_lighting,
    }


def _determine_active_group(review: dict) -> str:
    """
    Deterministically pick active group from diagnostics.
    Priority: lighting > material > object > scene
    """
    if review.get("lighting_diagnosis", "good") != "good":
        return "lighting"
    if review.get("color_consistency", "good") != "good":
        return "material"
    if review.get("physics_consistency", "good") == "minor_issue":
        return "object"
    return "lighting"


def classify_score_zone(score, preserve_threshold, explore_threshold):
    if score >= preserve_threshold:
        return "high"
    if score >= explore_threshold:
        return "mid"
    return "low"


def blacklist_failed_action(action_blacklist: set, aspace: dict, action_name: str,
                            sub_results=None) -> list:
    """
    Blacklist a failed action and its applied sub-actions.
    Returns a sorted list of newly blocked action names.
    """
    if not action_name or action_name == "NO_OP":
        return []
    blocked = {action_name}
    compound = aspace.get("compound_actions", {}).get(action_name)
    if compound:
        if sub_results:
            for sr in sub_results:
                if getattr(sr, "applied", False):
                    blocked.add(getattr(sr, "action", ""))
        else:
            blocked.update(compound.get("sub_actions", []) or [])
    newly_blocked = sorted(a for a in blocked if a and a not in action_blacklist)
    action_blacklist.update({a for a in blocked if a})
    return newly_blocked


def pick_next_group(issue_tags, suggested_actions, group_tried, aspace):
    fallback_map = aspace.get("issue_to_action_fallback", {})
    candidate_groups = []
    for tag in issue_tags:
        for action in fallback_map.get(tag, []):
            for grp, gdata in aspace["groups"].items():
                if action in gdata["actions"] and grp not in group_tried:
                    candidate_groups.append(grp)
    for action in suggested_actions:
        for grp, gdata in aspace["groups"].items():
            if action in gdata["actions"] and grp not in group_tried:
                candidate_groups.append(grp)
    if candidate_groups:
        return Counter(candidate_groups).most_common(1)[0][0]
    for grp in GROUP_ORDER:
        if grp not in group_tried:
            return grp
    return GROUP_ORDER[0]


def rerender_object(obj_id, meshes_dir, output_dir, control_state,
                    blender_bin=BLENDER_BIN, resolution=512, engine="EEVEE"):
    glb_path = os.path.join(meshes_dir, f"{obj_id}.glb")
    if not os.path.exists(glb_path):
        print(f"  [rerender] GLB not found: {glb_path}")
        return False
    cs_tmp = os.path.join(output_dir, f"_cs_{obj_id}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(cs_tmp, "w") as f:
        json.dump(control_state, f, indent=2)
    cmd = [
        blender_bin, "-b", "-P", RENDER_SCRIPT, "--",
        "--input-dir", meshes_dir,
        "--output-dir", output_dir,
        "--resolution", str(resolution),
        "--engine", engine,
        "--obj-id", obj_id,
        "--control-state", cs_tmp,
    ]
    print(f"  [rerender] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  [rerender] Blender failed:\n{result.stderr[-1000:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  [rerender] Blender timed out")
        return False
    except Exception as e:
        print(f"  [rerender] Error: {e}")
        return False


def evolve_object(obj_id, renders_dir, evolution_dir, meshes_dir=None,
                  device="cuda:0", max_rounds=None, blender_bin=BLENDER_BIN,
                  profile=None, reference_image_path=None):
    import statistics as _stats

    prof = profile or PROFILE_DEFAULTS

    _ACCEPT_TH = float(prof.get("accept_threshold", PROFILE_DEFAULTS["accept_threshold"]))
    _REJECT_TH = float(prof.get("reject_threshold", PROFILE_DEFAULTS["reject_threshold"]))
    _PRESERVE_TH = float(prof.get("preserve_score_threshold", PROFILE_DEFAULTS["preserve_score_threshold"]))
    _EXPLORE_TH = float(prof.get("explore_threshold", PROFILE_DEFAULTS["explore_threshold"]))
    _STABILITY_TH = float(prof.get("stability_threshold", PROFILE_DEFAULTS["stability_threshold"]))
    _UNSTABLE_VAR = float(prof.get("unstable_variance_limit", PROFILE_DEFAULTS["unstable_variance_limit"]))
    _MID_BUDGET = int(prof.get("mid_budget", PROFILE_DEFAULTS["mid_budget"]))
    _LOW_BUDGET = int(prof.get("low_budget", PROFILE_DEFAULTS["low_budget"]))
    _IMPROVE_EPS = float(prof.get("improve_eps", PROFILE_DEFAULTS["improve_eps"]))
    _MAX_ROUNDS = int(prof.get("max_rounds", max_rounds or PROFILE_DEFAULTS["max_rounds"]))
    _PATIENCE = int(prof.get("patience", PROFILE_DEFAULTS["patience"]))
    _PROMPT_APPENDIX = prof.get("prompt_appendix", "")
    _ISSUE_TAGS_WL = prof.get("issue_tags_whitelist", None)

    obj_dir = os.path.join(evolution_dir, obj_id)
    reviews_dir = os.path.join(obj_dir, "reviews")
    states_dir = os.path.join(obj_dir, "states")
    history_json = os.path.join(states_dir, "history.json")
    sharp_hist = os.path.join(obj_dir, "sharpness_history.json")
    os.makedirs(reviews_dir, exist_ok=True)
    os.makedirs(states_dir, exist_ok=True)

    aspace = load_action_space()
    control_state = default_control_state(aspace)
    action_blacklist: set = set()
    state_log: list = []

    baseline_probes = []

    def _run_clean_probe(probe_idx_offset: int) -> Optional[tuple]:
        cs_path = os.path.join(states_dir, f"cs_baseline_p{probe_idx_offset:02d}.json")
        save_control_state(control_state, cs_path)
        rev = review_object(
            obj_id=obj_id,
            renders_dir=renders_dir,
            output_dir=reviews_dir,
            round_idx=probe_idx_offset,
            active_group="lighting",
            prev_renders_dir=None,
            device=device,
            history_file=None,
            prompt_appendix=_PROMPT_APPENDIX,
            issue_tags_whitelist=_ISSUE_TAGS_WL,
            reference_image_path=reference_image_path,
            profile_cfg=prof,
        )
        if not rev:
            return None
        return rev.get("hybrid_score", 0.0), rev

    p0 = _run_clean_probe(0)
    if p0 is None:
        print(f"  [{obj_id}] Baseline probe 0 failed - render_hard_fail")
        return {
            "obj_id": obj_id,
            "final_hybrid": 0.0,
            "probes_run": 0,
            "updates_run": 0,
            "accepted": False,
            "exit_reason": "rejected_render_hard_fail",
            "state_log": [],
            "best_state": control_state,
            "best_state_path": "",
        }
    baseline_probes.append(p0)
    print(f"  [{obj_id}] Baseline p0: {p0[0]:.4f}")

    p1 = _run_clean_probe(1)
    if p1 is None:
        print(f"  [{obj_id}] Baseline probe 1 failed - render_hard_fail")
        return {
            "obj_id": obj_id,
            "final_hybrid": p0[0],
            "probes_run": 1,
            "updates_run": 0,
            "accepted": False,
            "exit_reason": "rejected_render_hard_fail",
            "state_log": [],
            "best_state": control_state,
            "best_state_path": "",
        }
    baseline_probes.append(p1)
    print(f"  [{obj_id}] Baseline p1: {p1[0]:.4f}")

    scores_so_far = [p0[0], p1[0]]
    score_repeats = 2
    score_span = max(scores_so_far) - min(scores_so_far)

    if abs(p0[0] - p1[0]) > _STABILITY_TH:
        p2 = _run_clean_probe(2)
        if p2 is not None:
            baseline_probes.append(p2)
            scores_so_far.append(p2[0])
            score_repeats = 3
            score_span = max(scores_so_far) - min(scores_so_far)
            print(f"  [{obj_id}] Baseline p2: {p2[0]:.4f}")

    if score_repeats == 3:
        confirmed_score = _stats.median(scores_so_far)
        if score_span > _UNSTABLE_VAR:
            print(f"  [{obj_id}] UNSTABLE: span={score_span:.4f} > {_UNSTABLE_VAR}")
            return {
                "obj_id": obj_id,
                "final_hybrid": confirmed_score,
                "probes_run": score_repeats,
                "updates_run": 0,
                "accepted": False,
                "exit_reason": "rejected_unstable_score",
                "state_log": [],
                "best_state": control_state,
                "best_state_path": "",
                "confirmed_score": confirmed_score,
            }
    else:
        confirmed_score = sum(scores_so_far) / len(scores_so_far)

    baseline_review_dicts = [bp[1] for bp in baseline_probes]
    agg_baseline_diag = _aggregate_baseline_diagnostics(baseline_review_dicts)
    confirmed_structure = agg_baseline_diag.get("structure_consistency", "good")
    confirmed_physics = agg_baseline_diag.get("physics_consistency", "good")
    confirmed_color = agg_baseline_diag.get("color_consistency", "good")
    confirmed_lighting = agg_baseline_diag.get("lighting_diagnosis", "good")
    print(
        f"  [{obj_id}] Confirmed baseline: {confirmed_score:.4f} "
        f"(repeats={score_repeats}, span={score_span:.4f})"
    )
    print(
        f"  [{obj_id}] Diagnostics: struct={confirmed_structure} "
        f"phys={confirmed_physics} color={confirmed_color} light={confirmed_lighting}"
    )

    best_score = confirmed_score
    best_state = copy.deepcopy(control_state)
    baseline_state = copy.deepcopy(control_state)

    def _make_result(exit_reason: str, final_score: float, attempts_used: int, updates_used: int) -> dict:
        best_state_path = os.path.join(states_dir, "control_state_best.json")
        save_control_state(best_state, best_state_path)
        result = {
            "obj_id": obj_id,
            "final_hybrid": final_score,
            "probes_run": score_repeats + attempts_used,
            "updates_run": updates_used,
            "accepted": final_score >= _ACCEPT_TH,
            "exit_reason": exit_reason,
            "state_log": state_log,
            "best_state": best_state,
            "best_state_path": best_state_path,
            "confirmed_score": confirmed_score,
            "score_repeats": score_repeats,
            "score_span": round(score_span, 6),
            "diagnostics": agg_baseline_diag,
        }
        result_path = os.path.join(obj_dir, "evolution_result.json")
        with open(result_path, "w") as f:
            json.dump({k: v for k, v in result.items() if k != "best_state"}, f, indent=2)
        print(
            f"\n[{obj_id}] Exit: {exit_reason} | final={final_score:.4f} | "
            f"accepted={result['accepted']} | attempts={attempts_used}"
        )
        return result

    if confirmed_structure == "major_mismatch":
        return _make_result("rejected_mesh", confirmed_score, 0, 0)
    if confirmed_physics == "major_issue":
        return _make_result("rejected_physics_major", confirmed_score, 0, 0)
    if confirmed_score >= _ACCEPT_TH:
        return _make_result("accepted_baseline", confirmed_score, 0, 0)
    if confirmed_score >= _PRESERVE_TH:
        return _make_result("preserved_high_zone", confirmed_score, 0, 0)
    if confirmed_score < _REJECT_TH:
        return _make_result("rejected_low_quality", confirmed_score, 0, 0)

    zone = "mid" if confirmed_score >= _EXPLORE_TH else "low"
    budget = _MID_BUDGET if zone == "mid" else _LOW_BUDGET
    print(f"  [{obj_id}] Zone={zone}, budget={budget}, exploring with preset_mode")

    active_group = _determine_active_group(agg_baseline_diag)
    combined_review = dict(baseline_review_dicts[-1])
    combined_review.update(agg_baseline_diag)
    combined_review["active_group"] = active_group
    combined_review["confirmed_score"] = confirmed_score
    combined_review["zone"] = zone

    agg_review_path = os.path.join(reviews_dir, f"{obj_id}_baseline_agg.json")
    with open(agg_review_path, "w") as f:
        json.dump(combined_review, f, indent=2)

    attempts_made = 0
    updates_made = 0

    for attempt_idx in range(min(budget, _MAX_ROUNDS)):
        print(f"\n  [{obj_id}] Search attempt {attempt_idx+1}/{budget} | group={active_group} | zone={zone}")

        attempt_state = copy.deepcopy(baseline_state)
        cs_attempt_path = os.path.join(states_dir, f"cs_attempt_{attempt_idx:02d}.json")
        save_control_state(attempt_state, cs_attempt_path)

        next_cs_path = os.path.join(states_dir, f"cs_attempt_{attempt_idx:02d}_next.json")
        feedback = apply_feedback(
            review_json_path=agg_review_path,
            control_state_path=cs_attempt_path,
            output_control_state_path=next_cs_path,
            round_idx=attempt_idx,
            history_json_path=history_json,
            hybrid_score=confirmed_score,
            action_blacklist=action_blacklist,
            preset_mode=True,
            active_group_override=active_group,
        )

        action = feedback.get("action_taken", "NO_OP")
        sub_results = feedback.get("sub_results")
        attempt_control_state = feedback.get("control_state", attempt_state)
        attempts_made += 1

        if action == "__MESH_REJECT__":
            state_log.append({
                "attempt": attempt_idx,
                "zone": zone,
                "action_taken": action,
                "hybrid_score": confirmed_score,
                "confirmed_score": confirmed_score,
                "score_repeats": score_repeats,
                "score_delta": 0.0,
                "active_group": active_group,
                "tried_presets": sorted(action_blacklist),
                "action_blacklist": sorted(action_blacklist),
                "sub_actions_applied": None,
                "exit_reason": "rejected_mesh",
            })
            return _make_result("rejected_mesh", confirmed_score, attempts_made, updates_made)

        if action == "NO_OP":
            state_log.append({
                "attempt": attempt_idx,
                "zone": zone,
                "action_taken": action,
                "hybrid_score": confirmed_score,
                "confirmed_score": confirmed_score,
                "score_repeats": score_repeats,
                "score_delta": 0.0,
                "active_group": active_group,
                "tried_presets": sorted(action_blacklist),
                "action_blacklist": sorted(action_blacklist),
                "sub_actions_applied": None,
                "exit_reason": "mid_no_improve" if zone == "mid" else "low_exhausted",
            })
            return _make_result("mid_no_improve" if zone == "mid" else "low_exhausted", confirmed_score, attempts_made, updates_made)

        updates_made += 1

        if meshes_dir:
            new_render_base = os.path.join(obj_dir, f"renders_u{attempt_idx + 1:02d}")
            attempt_renders_dir = os.path.join(new_render_base, obj_id)
            success = rerender_object(
                obj_id=obj_id,
                meshes_dir=meshes_dir,
                output_dir=new_render_base,
                control_state=attempt_control_state,
                blender_bin=blender_bin,
            )
            has_renders = (
                os.path.isdir(attempt_renders_dir)
                and len([f for f in os.listdir(attempt_renders_dir) if f.endswith(".png")]) > 0
            )
            if not (success and has_renders):
                newly_blocked = blacklist_failed_action(action_blacklist, aspace, action, sub_results)
                state_log.append({
                    "attempt": attempt_idx,
                    "zone": zone,
                    "action_taken": action,
                    "hybrid_score": confirmed_score,
                    "confirmed_score": confirmed_score,
                    "score_repeats": score_repeats,
                    "score_delta": 0.0,
                    "active_group": active_group,
                    "tried_presets": sorted(newly_blocked),
                    "action_blacklist": sorted(action_blacklist),
                    "sub_actions_applied": (
                        [getattr(sr, "action", "") for sr in (sub_results or []) if getattr(sr, "applied", False)]
                        if sub_results else None
                    ),
                    "exit_reason": "rejected_render_hard_fail",
                })
                if zone == "mid":
                    return _make_result("rejected_render_hard_fail", confirmed_score, attempts_made, updates_made)
                continue
        else:
            print(f"  [{obj_id}] No meshes-dir, dry run attempt {attempt_idx+1}")
            attempt_renders_dir = renders_dir

        attempt_review = review_object(
            obj_id=obj_id,
            renders_dir=os.path.dirname(attempt_renders_dir) if meshes_dir else attempt_renders_dir,
            output_dir=reviews_dir,
            round_idx=score_repeats + attempt_idx,
            active_group=active_group,
            prev_renders_dir=renders_dir,
            device=device,
            history_file=sharp_hist,
            prompt_appendix=_PROMPT_APPENDIX,
            issue_tags_whitelist=_ISSUE_TAGS_WL,
            reference_image_path=reference_image_path,
            profile_cfg=prof,
        )

        if not attempt_review:
            newly_blocked = blacklist_failed_action(action_blacklist, aspace, action, sub_results)
            state_log.append({
                "attempt": attempt_idx,
                "zone": zone,
                "action_taken": action,
                "hybrid_score": confirmed_score,
                "confirmed_score": confirmed_score,
                "score_repeats": score_repeats,
                "score_delta": 0.0,
                "active_group": active_group,
                "tried_presets": sorted(newly_blocked),
                "action_blacklist": sorted(action_blacklist),
                "sub_actions_applied": (
                    [getattr(sr, "action", "") for sr in (sub_results or []) if getattr(sr, "applied", False)]
                    if sub_results else None
                ),
                "exit_reason": "rejected_render_hard_fail",
            })
            if zone == "mid":
                return _make_result("rejected_render_hard_fail", confirmed_score, attempts_made, updates_made)
            continue

        attempt_score = attempt_review.get("hybrid_score", 0.0)
        delta = attempt_score - confirmed_score
        log_entry = {
            "attempt": attempt_idx,
            "zone": zone,
            "action_taken": action,
            "hybrid_score": attempt_score,
            "confirmed_score": confirmed_score,
            "score_repeats": score_repeats,
            "score_delta": delta,
            "active_group": active_group,
            "tried_presets": sorted(action_blacklist),
            "action_blacklist": sorted(action_blacklist),
            "sub_actions_applied": (
                [getattr(sr, "action", "") for sr in (sub_results or []) if getattr(sr, "applied", False)]
                if sub_results else None
            ),
            "exit_reason": None,
        }

        print(
            f"  [{obj_id}] Attempt {attempt_idx+1}: action={action} -> "
            f"score={attempt_score:.4f} (baseline={confirmed_score:.4f}, delta={delta:+.4f})"
        )

        if attempt_score > confirmed_score + _IMPROVE_EPS:
            best_score = attempt_score
            best_state = copy.deepcopy(attempt_control_state)
            log_entry["exit_reason"] = "accepted_after_try"
            state_log.append(log_entry)
            return _make_result("accepted_after_try", best_score, attempts_made, updates_made)

        newly_blocked = blacklist_failed_action(action_blacklist, aspace, action, sub_results)
        log_entry["tried_presets"] = sorted(newly_blocked)
        log_entry["action_blacklist"] = sorted(action_blacklist)
        log_entry["exit_reason"] = "attempt_failed"
        state_log.append(log_entry)
        print(f"  [{obj_id}] Attempt {attempt_idx+1} did not improve, blacklisted '{action}'")

        if zone == "mid":
            return _make_result("mid_no_improve", confirmed_score, attempts_made, updates_made)

    return _make_result("low_exhausted" if zone == "low" else "mid_no_improve", confirmed_score, attempts_made, updates_made)


def discover_obj_ids(renders_dir):
    return sorted(
        d for d in os.listdir(renders_dir)
        if os.path.isdir(os.path.join(renders_dir, d))
    )


def run_evolution(renders_dir, output_dir, obj_ids=None, meshes_dir=None,
                  device="cuda:0", max_rounds=None, blender_bin=BLENDER_BIN, profile=None):
    os.makedirs(output_dir, exist_ok=True)
    if obj_ids is None:
        obj_ids = discover_obj_ids(renders_dir)
    prof = profile or PROFILE_DEFAULTS
    print(f"[Evolution] Processing {len(obj_ids)} objects: {obj_ids}")
    all_results = {}
    _pstate = _init_pipeline_state(obj_ids, output_dir)
    _save_pipeline_state(_pstate)

    for obj_id in obj_ids:
        try:
            ref_img = _resolve_reference_image(obj_id, prof, renders_dir)
            if ref_img:
                print(f"[Evolution] Reference image for {obj_id}: {ref_img}")
            else:
                print(f"[Evolution] No reference image found for {obj_id}")

            result = evolve_object(
                obj_id=obj_id,
                renders_dir=renders_dir,
                evolution_dir=output_dir,
                meshes_dir=meshes_dir,
                device=device,
                max_rounds=max_rounds,
                blender_bin=blender_bin,
                profile=profile,
                reference_image_path=ref_img,
            )
            all_results[obj_id] = result
            _update_obj_state(_pstate, obj_id, result)
        except Exception as e:
            print(f"[Evolution] ERROR on {obj_id}: {e}")
            import traceback
            traceback.print_exc()
            all_results[obj_id] = {"obj_id": obj_id, "error": str(e)}
            _update_obj_state(_pstate, obj_id, {"final_hybrid": 0, "exit_reason": "error",
                                                "accepted": False, "probes_run": 0})

    accepted = sum(1 for r in all_results.values() if r.get("accepted", False))
    final_scores = {k: v.get("final_hybrid", 0.0) for k, v in all_results.items()}
    summary = {
        "total": len(all_results),
        "accepted": accepted,
        "acceptance_rate": accepted / max(1, len(all_results)),
        "final_scores": final_scores,
        "results": all_results,
    }
    device_tag = device.replace(":", "").replace(",", "_")
    partial_path = os.path.join(output_dir, f"_partial_{device_tag}.json")
    partial = {
        "device": device,
        "obj_ids": list(all_results.keys()),
        "accepted": accepted,
        "total": len(all_results),
        "final_scores": final_scores,
    }
    with open(partial_path, "w") as f:
        json.dump(partial, f, indent=2)

    avg_score = sum(final_scores.values()) / max(1, len(final_scores))
    print(f"\n[Evolution] Done: {accepted}/{len(all_results)} accepted, avg_score={avg_score:.3f}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="VLM-feedback render self-evolution loop")
    p.add_argument("--renders-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--obj-ids", nargs="+", default=None)
    p.add_argument("--meshes-dir", default=None)
    p.add_argument("--max-rounds", type=int, default=PROFILE_DEFAULTS["max_rounds"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--blender", default=BLENDER_BIN, dest="blender_bin")
    p.add_argument("--profile", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profile = load_profile(args.profile)
    if args.profile:
        print(f"[Evolution] Profile: {args.profile}")
    summary = run_evolution(
        renders_dir=args.renders_dir,
        output_dir=args.output_dir,
        obj_ids=args.obj_ids,
        meshes_dir=args.meshes_dir,
        device=args.device,
        max_rounds=args.max_rounds,
        blender_bin=args.blender_bin,
        profile=profile,
    )
    print(f"\n=== Summary: {summary['accepted']}/{summary['total']} accepted, "
          f"rate={summary['acceptance_rate']:.1%} ===")
