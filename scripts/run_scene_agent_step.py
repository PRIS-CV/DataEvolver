"""
Single agent-guided scene step.

This script does exactly one round for one (object, yaw) pair:
1. build / load control state
2. optionally apply explicitly provided actions
3. render with the normal scene-aware pipeline
4. run Qwen3.5 freeform review with thinking enabled

It does NOT auto-select actions from the review. The next action is expected to
be chosen by the human/agent after reading the freeform review output.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
PIPELINE_ROOT = REPO_ROOT / "pipeline"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import run_scene_evolution_loop as scene_loop
from stage5_6_feedback_apply import (
    apply_action,
    default_control_state,
    load_action_space,
    load_control_state,
    save_control_state,
)


DEFAULT_BLENDER = os.environ.get("BLENDER_BIN", "blender")
DEFAULT_PROFILE = REPO_ROOT / "configs" / "dataset_profiles" / "scene_v7.json"
DEFAULT_SCENE_TEMPLATE = REPO_ROOT / "configs" / "scene_template.json"
DEFAULT_MESHES_DIR = REPO_ROOT / "pipeline" / "data" / "meshes"


def parse_args():
    parser = argparse.ArgumentParser(description="Run one agent-guided scene render/review step")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--obj-id", required=True)
    parser.add_argument("--rotation-deg", type=int, required=True)
    parser.add_argument("--round-idx", type=int, default=0)
    parser.add_argument("--actions", nargs="*", default=[])
    parser.add_argument("--step-scale", type=float, default=1.0)
    parser.add_argument("--agent-note", default="")
    parser.add_argument("--control-state-in", default=None)
    parser.add_argument("--prev-renders-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blender", dest="blender_bin", default=DEFAULT_BLENDER)
    parser.add_argument("--meshes-dir", default=str(DEFAULT_MESHES_DIR))
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--scene-template", default=str(DEFAULT_SCENE_TEMPLATE))
    return parser.parse_args()


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compact_json(payload, max_chars: int = 1800) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = str(payload)
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[:max_chars] + "...<truncated>"
    return text


def _latest_trace_text(reviews_dir: Path, prev_round_idx: int) -> str:
    candidates = sorted(reviews_dir.glob(f"*_r{prev_round_idx:02d}_*_trace.json"))
    if not candidates:
        candidates = sorted(reviews_dir.glob("*_trace.json"))
    if not candidates:
        return ""
    trace = load_json(candidates[-1], default={}) or {}
    attempts = trace.get("attempts") or []
    if attempts:
        text = attempts[-1].get("assistant_text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    for key in ("assistant_text", "raw_text", "response_text", "content", "answer", "model_output"):
        value = trace.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_review_context_appendix(
    *,
    pair_dir: Path,
    reviews_dir: Path,
    obj_id: str,
    round_idx: int,
    agent_note: str,
    current_state: dict,
    render_metadata: dict,
    reference_image_path: Optional[str],
) -> str:
    """Context injected into the VLM review prompt so the freeform reviewer can diagnose, not just score."""
    sections = [
        "Additional review context for this iterative scene loop:",
        "- Treat this as a local optimization loop for one object at one yaw.",
        "- Prefer concrete visual diagnosis and concrete next actions over generic scoring comments.",
    ]
    if reference_image_path:
        sections.append(f"- Original reference image path: {reference_image_path}")
    if agent_note:
        sections.append(f"- Agent / gate note from previous decision: {agent_note}")

    if round_idx > 0:
        prev_agg_path = reviews_dir / f"{obj_id}_r{round_idx - 1:02d}_agg.json"
        prev_agg = load_json(prev_agg_path, default={}) or {}
        if isinstance(prev_agg, dict) and prev_agg:
            prev_summary = {
                "hybrid_score": prev_agg.get("hybrid_score"),
                "vlm_only_score": prev_agg.get("vlm_only_score"),
                "issue_tags": prev_agg.get("issue_tags"),
                "suggested_actions": prev_agg.get("suggested_actions"),
                "advisor_actions": prev_agg.get("advisor_actions"),
                "lighting_diagnosis": prev_agg.get("lighting_diagnosis"),
                "structure_consistency": prev_agg.get("structure_consistency"),
                "color_consistency": prev_agg.get("color_consistency"),
                "physics_consistency": prev_agg.get("physics_consistency"),
                "asset_viability": prev_agg.get("asset_viability"),
                "abandon_reason": prev_agg.get("abandon_reason"),
            }
            sections.append("- Previous round aggregate review JSON summary:")
            sections.append(_compact_json(prev_summary, max_chars=1400))
        prev_trace = _latest_trace_text(reviews_dir, round_idx - 1)
        if prev_trace:
            sections.append("- Previous round freeform reviewer feedback excerpt:")
            sections.append(" ".join(prev_trace.split())[:1800])

    state_summary = {
        "object": current_state.get("object"),
        "lighting": current_state.get("lighting"),
        "camera": current_state.get("camera"),
        "scene": current_state.get("scene"),
        "material": current_state.get("material"),
    }
    sections.append("- Current control_state summary:")
    sections.append(_compact_json(state_summary, max_chars=1800))

    metadata_summary = {}
    if isinstance(render_metadata, dict):
        for key in (
            "mesh_material_quality",
            "material_adaptation",
            "object_bbox",
            "programmatic_physics",
            "control_state_path",
        ):
            if key in render_metadata:
                metadata_summary[key] = render_metadata.get(key)
    if metadata_summary:
        sections.append("- Current render metadata summary:")
        sections.append(_compact_json(metadata_summary, max_chars=1600))

    sections.append(
        "Required reviewer behavior: name the primary failure, compare against the reference, "
        "judge scene integration, and propose up to three concrete next actions."
    )
    return "\n".join(sections)


def _rotation_override(rotation_deg: int) -> dict:
    return {
        "object": {
            "yaw_deg": float(rotation_deg),
        }
    }


def _detect_previous_pair_artifacts(pair_dir: Path, round_idx: int) -> tuple[Optional[Path], Optional[Path]]:
    if round_idx <= 0:
        return None, None
    prev_state = pair_dir / "states" / f"round{round_idx - 1:02d}.json"
    prev_renders = pair_dir / f"round{round_idx - 1:02d}_renders"
    return (prev_state if prev_state.exists() else None,
            prev_renders if prev_renders.exists() else None)


def build_pair_profile(base_profile_path: Path, rotation_deg: int) -> dict:
    profile = scene_loop.load_profile(str(base_profile_path))
    profile["use_freeform_vlm_feedback"] = True
    profile["enable_vlm_thinking"] = True
    profile["auto_retry_with_vlm_only"] = False
    profile["initial_control_state_overrides"] = scene_loop._deep_merge_dict(
        profile.get("initial_control_state_overrides") or {},
        _rotation_override(rotation_deg),
    )
    profile["locked_control_state_overrides"] = scene_loop._deep_merge_dict(
        profile.get("locked_control_state_overrides") or {},
        _rotation_override(rotation_deg),
    )
    return profile


def run_agent_step(
    *,
    output_dir: str,
    obj_id: str,
    rotation_deg: int,
    round_idx: int,
    actions: List[str],
    step_scale: float,
    agent_note: str,
    control_state_in: Optional[str],
    prev_renders_dir: Optional[str],
    device: str,
    blender_bin: str,
    meshes_dir: str,
    profile_path: str,
    scene_template_path: str,
) -> dict:
    pair_dir = Path(output_dir).resolve()
    pair_dir.mkdir(parents=True, exist_ok=True)
    states_dir = pair_dir / "states"
    reviews_dir = pair_dir / "reviews"
    states_dir.mkdir(parents=True, exist_ok=True)
    reviews_dir.mkdir(parents=True, exist_ok=True)

    aspace = load_action_space(scene_loop.SCENE_ACTION_SPACE_PATH)
    profile = build_pair_profile(Path(profile_path), rotation_deg)
    scene_template = scene_loop.load_scene_template(scene_template_path)
    scene_template["__path__"] = scene_template_path

    auto_prev_state, auto_prev_renders = _detect_previous_pair_artifacts(pair_dir, round_idx)
    control_state_path = Path(control_state_in).resolve() if control_state_in else auto_prev_state
    effective_prev_renders = prev_renders_dir or (str(auto_prev_renders) if auto_prev_renders else None)

    if control_state_path is not None and control_state_path.exists():
        state = load_control_state(str(control_state_path), aspace)
    else:
        state = default_control_state(aspace)

    state = scene_loop._apply_control_overrides(state, profile.get("initial_control_state_overrides"))
    state = scene_loop._apply_control_overrides(state, profile.get("locked_control_state_overrides"))

    round_tag = f"round{round_idx:02d}"
    round_input_state_path = states_dir / f"{round_tag}_input.json"
    round_state_path = states_dir / f"{round_tag}.json"
    save_control_state(state, str(round_input_state_path))

    action_results = []
    current_state = copy.deepcopy(state)
    for action in actions:
        result = apply_action(action, current_state, aspace, step_scale=step_scale)
        current_state = result.state
        action_results.append({
            "action": action,
            "applied": bool(result.applied),
            "target": result.target,
            "delta_sign": result.delta_sign,
        })

    current_state = scene_loop._apply_control_overrides(current_state, profile.get("locked_control_state_overrides"))
    save_control_state(current_state, str(round_state_path))

    render_resolution = int(scene_template.get("render_resolution", 512))
    render_engine = str(scene_template.get("render_engine", "EEVEE")).upper()
    render_dir = pair_dir / f"{round_tag}_renders"
    reference_image_path = scene_loop._resolve_reference_image(obj_id, profile)
    started = time.time()
    ok = scene_loop.render_scene_state(
        obj_id=obj_id,
        meshes_dir=meshes_dir,
        output_dir=str(render_dir),
        control_state=current_state,
        scene_template_path=scene_template["__path__"],
        blender_bin=blender_bin,
        resolution=render_resolution,
        engine=render_engine,
        reference_image_path=reference_image_path,
    )
    obj_render_dir = render_dir / obj_id
    has_renders = obj_render_dir.is_dir() and any(
        p.name.endswith(".png") and not p.name.endswith("_mask.png")
        for p in obj_render_dir.iterdir()
    )
    render_metadata = load_json(obj_render_dir / "metadata.json", default={}) if obj_render_dir.is_dir() else {}
    if not isinstance(render_metadata, dict):
        render_metadata = {}
    if not (ok and has_renders):
        result = {
            "success": False,
            "obj_id": obj_id,
            "rotation_deg": rotation_deg,
            "round_idx": round_idx,
            "pair_dir": str(pair_dir),
            "render_dir": str(render_dir),
            "review_agg_path": None,
            "actions": action_results,
            "material_adaptation": render_metadata.get("material_adaptation"),
            "mesh_material_quality": render_metadata.get("mesh_material_quality"),
            "agent_note": agent_note,
            "elapsed_seconds": round(time.time() - started, 3),
            "error": "render_failed",
        }
        save_json(pair_dir / f"agent_{round_tag}.json", result)
        return result

    profile_prompt_appendix = profile.get("prompt_appendix", "")
    dynamic_review_appendix = _build_review_context_appendix(
        pair_dir=pair_dir,
        reviews_dir=reviews_dir,
        obj_id=obj_id,
        round_idx=round_idx,
        agent_note=agent_note,
        current_state=current_state,
        render_metadata=render_metadata,
        reference_image_path=reference_image_path,
    )
    prompt_appendix_parts = [part for part in (profile_prompt_appendix, dynamic_review_appendix) if part]
    review_prompt_appendix = "\n\n".join(prompt_appendix_parts)

    review = scene_loop.review_object(
        obj_id=obj_id,
        renders_dir=str(render_dir),
        output_dir=str(reviews_dir),
        round_idx=round_idx,
        active_group="scene",
        prev_renders_dir=effective_prev_renders,
        device=device,
        history_file=str(pair_dir / "sharpness_history.json"),
        prompt_appendix=review_prompt_appendix,
        issue_tags_whitelist=profile.get("issue_tags_whitelist"),
        reference_image_path=reference_image_path,
        pseudo_reference_path=scene_loop._resolve_pseudo_reference_image(obj_id, profile),
        profile_cfg=profile,
        review_mode=profile.get("review_mode", "scene_insert"),
        action_space_path=scene_loop.SCENE_ACTION_SPACE_PATH,
        rep_views=scene_template.get("qc_views"),
    )

    review_agg_path = reviews_dir / f"{obj_id}_r{round_idx:02d}_agg.json"
    result = {
        "success": True,
        "obj_id": obj_id,
        "rotation_deg": rotation_deg,
        "round_idx": round_idx,
        "pair_dir": str(pair_dir),
        "render_dir": str(render_dir),
        "prev_renders_dir": effective_prev_renders,
        "state_in": str(control_state_path) if control_state_path else None,
        "state_out": str(round_state_path),
        "review_agg_path": str(review_agg_path) if review_agg_path.exists() else None,
        "hybrid_score": review.get("hybrid_score"),
        "vlm_only_score": review.get("vlm_only_score"),
        "lighting_diagnosis": review.get("lighting_diagnosis"),
        "asset_viability": review.get("asset_viability"),
        "abandon_reason": review.get("abandon_reason"),
        "material_adaptation": render_metadata.get("material_adaptation"),
        "mesh_material_quality": render_metadata.get("mesh_material_quality"),
        "issue_tags": review.get("issue_tags"),
        "suggested_actions": review.get("suggested_actions"),
        "review_prompt_context": {
            "has_agent_note": bool(agent_note),
            "has_previous_review": round_idx > 0,
            "has_render_metadata": bool(render_metadata),
            "prompt_appendix_chars": len(review_prompt_appendix),
        },
        "actions": action_results,
        "agent_note": agent_note,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    save_json(pair_dir / f"agent_{round_tag}.json", result)
    return result


def main():
    args = parse_args()
    result = run_agent_step(
        output_dir=args.output_dir,
        obj_id=args.obj_id,
        rotation_deg=args.rotation_deg,
        round_idx=args.round_idx,
        actions=args.actions,
        step_scale=args.step_scale,
        agent_note=args.agent_note,
        control_state_in=args.control_state_in,
        prev_renders_dir=args.prev_renders_dir,
        device=args.device,
        blender_bin=args.blender_bin,
        meshes_dir=args.meshes_dir,
        profile_path=args.profile,
        scene_template_path=args.scene_template,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
