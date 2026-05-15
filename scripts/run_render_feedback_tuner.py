#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent

DEFAULT_ROOT_DIR = REPO_ROOT / "pipeline" / "data" / "evolution_scene_v7_full20_rotation4_agent_round0_nostage35_20260404"
DEFAULT_TEMPLATE_PATH = REPO_ROOT / "configs" / "scene_template.json"
DEFAULT_PROFILE = REPO_ROOT / "configs" / "dataset_profiles" / "scene_v7_full20_loop.json"
DEFAULT_MESHES_DIR = REPO_ROOT / "pipeline" / "data" / "meshes"
DEFAULT_PYTHON = "python"
DEFAULT_BLENDER = "blender"
DEFAULT_RUNTIME_DIR = REPO_ROOT / "runtime" / "full20_scene_20260404" / "render_tuner"

CRITICAL_ISSUES = {
    "flat_lighting",
    "weak_subject_separation",
    "color_shift",
    "floating_visible",
    "ground_intersection_visible",
    "ground_intersection",
    "object_too_large",
    "object_too_small",
}

ACCEPTANCE_HINTS = (
    "acceptable",
    "good enough",
    "looks good",
    "looks solid",
    "can be accepted",
    "ready to use",
    "可以了",
    "够好了",
    "可接受",
)

REJECTION_HINTS = (
    "not acceptable",
    "not good enough",
    "still not acceptable",
    "still needs work",
    "needs more work",
    "requires revision",
    "should be revised",
    "cannot be accepted",
    "还不行",
    "还需要",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Codex render feedback tuner loop")
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--template-path", default=str(DEFAULT_TEMPLATE_PATH))
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--meshes-dir", default=str(DEFAULT_MESHES_DIR))
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--blender-bin", default=DEFAULT_BLENDER)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_DIR))
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--benchmark-limit", type=int, default=4)
    parser.add_argument("--target-mean-hybrid", type=float, default=0.60)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    return parser.parse_args()


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def extract_trace_text(trace_payload: dict) -> str:
    attempts = trace_payload.get("attempts") or []
    if attempts:
        text = attempts[-1].get("assistant_text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    for key in ("assistant_text", "raw_text", "response_text", "content", "answer", "model_output"):
        value = trace_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_reviewer_verdict(trace_text: str) -> Optional[str]:
    if not trace_text:
        return None
    matches = re.findall(r"^\s*verdict\s*:\s*([a-z_]+)", trace_text, flags=re.IGNORECASE | re.MULTILINE)
    if not matches:
        return None
    return matches[-1].strip().lower()


def reviewer_accepts(trace_text: str) -> bool:
    if not trace_text:
        return False
    normalized = re.sub(r"\s+", " ", trace_text.strip().lower())
    verdict = extract_reviewer_verdict(trace_text)
    if verdict == "keep":
        return True
    if verdict in {"reject", "revise"}:
        return False
    if any(hint in normalized for hint in REJECTION_HINTS):
        return False
    return any(hint in normalized for hint in ACCEPTANCE_HINTS)


def is_asset_blocked_trace(trace_text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (trace_text or "").strip().lower())
    if not normalized:
        return False
    hard_hints = (
        "fundamentally wrong",
        "different model",
        "object identity mismatch",
        "geometry is completely off",
        "wrong object geometry",
        "asset viability: abandon",
        "should be regenerated",
        "completely wrong shape",
        "not the same object as the reference",
        "object itself is fundamentally wrong",
        "merged mesh",
        "glitchy mesh",
        "physically impossible",
        "two skateboards",
        "double truck",
        "6 wheels",
        "six wheels",
        "structurally broken",
        "merged incorrectly",
        "two boards merged",
        "misaligned trucks",
        "misaligned wheels",
        "wrong base asset",
        "wrong source asset",
    )
    return any(hint in normalized for hint in hard_hints)


def is_asset_blocked_result(result: Optional[dict]) -> bool:
    if not result:
        return False
    viability = str(result.get("asset_viability") or "").strip().lower()
    if viability == "abandon":
        return True
    abandon_reason = str(result.get("abandon_reason") or "").strip()
    if abandon_reason and is_asset_blocked_trace(abandon_reason):
        return True
    return False


def parse_pair_name(pair_dir: Path) -> tuple[str, int]:
    stem = pair_dir.name
    left, yaw_text = stem.split("_yaw", 1)
    return left, int(yaw_text)


def find_latest_round_artifacts(pair_dir: Path) -> Optional[dict]:
    reviews_dir = pair_dir / "reviews"
    aggs = sorted(reviews_dir.glob("*_agg.json"))
    if not aggs:
        return None
    agg_path = aggs[-1]
    round_idx = -1
    stem = agg_path.name
    if "_r" in stem and stem.endswith("_agg.json"):
        try:
            round_idx = int(stem.split("_r", 1)[1].split("_agg.json", 1)[0])
        except Exception:
            round_idx = -1
    state_path = pair_dir / "states" / f"round{round_idx:02d}.json"
    if not state_path.exists():
        return None
    traces = sorted(reviews_dir.glob(f"*_r{round_idx:02d}_*_trace.json"))
    trace_path = traces[-1] if traces else None
    agg = load_json(agg_path, default={}) or {}
    trace_payload = load_json(trace_path, default={}) if trace_path else {}
    return {
        "agg_path": agg_path,
        "trace_path": trace_path,
        "agg": agg,
        "trace_text": extract_trace_text(trace_payload or {}),
        "round_idx": round_idx,
        "state_path": state_path,
    }


def collect_candidates(root_dir: Path) -> list[dict]:
    candidates = []
    for pair_dir in sorted(root_dir.glob("_shards/*/obj_*_yaw*")):
        if not pair_dir.is_dir():
            continue
        try:
            obj_id, rotation_deg = parse_pair_name(pair_dir)
        except Exception:
            continue
        artifacts = find_latest_round_artifacts(pair_dir)
        if not artifacts:
            continue
        agg = artifacts["agg"]
        issue_tags = [str(tag) for tag in (agg.get("issue_tags") or []) if str(tag)]
        hybrid = agg.get("hybrid_score")
        hybrid_score = float(hybrid) if hybrid is not None else 0.0
        priority = sum(1 for tag in issue_tags if tag in CRITICAL_ISSUES)
        text = artifacts["trace_text"].lower()
        if "plastic" in text or "cgi-like" in text:
            priority += 1
        if "floating" in text or "hovering" in text:
            priority += 1
        candidates.append(
            {
                "pair_dir": pair_dir,
                "pair_name": pair_dir.name,
                "obj_id": obj_id,
                "rotation_deg": rotation_deg,
                "round_idx": artifacts["round_idx"],
                "state_path": artifacts["state_path"],
                "agg_path": artifacts["agg_path"],
                "trace_path": artifacts["trace_path"],
                "hybrid_score": hybrid_score,
                "issue_tags": issue_tags,
                "trace_text": artifacts["trace_text"],
                "asset_blocked": is_asset_blocked_trace(artifacts["trace_text"]),
                "priority": priority,
            }
        )
    return candidates


def select_benchmarks(candidates: list[dict], limit: int) -> list[dict]:
    ranked = sorted(
        candidates,
        key=lambda item: (item.get("asset_blocked", False), -item["priority"], item["hybrid_score"], -item["round_idx"], item["pair_name"]),
    )
    chosen = []
    seen_obj_ids = set()
    for item in ranked:
        if item["obj_id"] in seen_obj_ids and len(chosen) < max(2, limit // 2):
            continue
        chosen.append(item)
        seen_obj_ids.add(item["obj_id"])
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        for item in ranked:
            if item in chosen:
                continue
            chosen.append(item)
            if len(chosen) >= limit:
                break
    return chosen


def _run(cmd: list[str]) -> int:
    print("[tuner]", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _find_previous_pair_state(runtime_dir: Path, pair_name: str, iteration_idx: int) -> Optional[Path]:
    for prev_idx in range(iteration_idx - 1, 0, -1):
        state_path = runtime_dir / f"iter_{prev_idx:02d}" / pair_name / "states" / "round00.json"
        if state_path.exists():
            return state_path
    return None


def _find_previous_pair_trace(runtime_dir: Path, pair_name: str, iteration_idx: int) -> str:
    for prev_idx in range(iteration_idx - 1, 0, -1):
        reviews_dir = runtime_dir / f"iter_{prev_idx:02d}" / pair_name / "reviews"
        traces = sorted(reviews_dir.glob("*_trace.json"))
        if not traces:
            continue
        trace_payload = load_json(traces[-1], default={}) or {}
        text = extract_trace_text(trace_payload)
        if text:
            return text
    return ""


def _find_latest_runtime_pair_trace(runtime_dir: Path, pair_name: str) -> str:
    traces = sorted((runtime_dir.glob(f"iter_*/{pair_name}/reviews/*_trace.json")))
    for trace_path in reversed(traces):
        trace_payload = load_json(trace_path, default={}) or {}
        text = extract_trace_text(trace_payload)
        if text:
            return text
    return ""


def _find_latest_pair_trace_across_tuner_runs(runtime_dir: Path, pair_name: str) -> str:
    trace_roots = [runtime_dir]
    trace_roots.extend(
        sorted(
            runtime_dir.parent.glob("render_tuner_archive_*"),
            key=lambda path: path.name,
            reverse=True,
        )
    )
    for root in trace_roots:
        text = _find_latest_runtime_pair_trace(root, pair_name)
        if text:
            return text
    return ""


def _find_latest_pair_result_across_tuner_runs(runtime_dir: Path, pair_name: str) -> dict:
    trace_roots = [runtime_dir]
    trace_roots.extend(
        sorted(
            runtime_dir.parent.glob("render_tuner_archive_*"),
            key=lambda path: path.name,
            reverse=True,
        )
    )
    for root in trace_roots:
        result_paths = sorted(root.glob(f"iter_*/{pair_name}/agent_round00.json"), reverse=True)
        for result_path in result_paths:
            payload = load_json(result_path, default={}) or {}
            if payload:
                return payload
    return {}


def _ensure_section(state: dict, key: str, default: dict) -> dict:
    value = state.get(key)
    if not isinstance(value, dict):
        value = copy.deepcopy(default)
        state[key] = value
    return value


def _state_mul(section: dict, key: str, factor: float, lo: float, hi: float):
    section[key] = round(_clamp(float(section.get(key, 1.0)) * factor, lo, hi), 4)


def _state_add(section: dict, key: str, delta: float, lo: float, hi: float):
    section[key] = round(_clamp(float(section.get(key, 0.0)) + delta, lo, hi), 4)


def build_pair_control_state(base_state: dict, trace_text: str) -> tuple[dict, list[str]]:
    state = copy.deepcopy(base_state)
    lighting = _ensure_section(state, "lighting", {"key_scale": 1.0, "key_yaw_deg": 0.0})
    obj = _ensure_section(state, "object", {"offset_z": 0.0, "yaw_deg": 0.0, "scale": 1.0})
    scene = _ensure_section(state, "scene", {"hdri_yaw_deg": 0.0, "env_strength_scale": 1.0, "contact_shadow_strength": 1.0})
    material = _ensure_section(
        state,
        "material",
        {
            "saturation_scale": 1.0,
            "value_scale": 1.0,
            "hue_offset": 0.0,
            "specular_add": 0.0,
            "roughness_add": 0.0,
        },
    )
    text = re.sub(r"\s+", " ", (trace_text or "").strip().lower())
    changes: list[str] = []
    if not text:
        return state, changes

    too_dark = any(
        hint in text
        for hint in (
            "too dark",
            "significantly darker",
            "underexposed",
            "very dim",
            "gloomy",
            "almost silhouetted",
            "almost a silhouette",
            "crushed blacks",
            "dark maroon",
            "almost black",
            "pitch black",
            "black blob",
            "failed insertion",
        )
    )
    flat = any(
        hint in text
        for hint in (
            "flat lighting",
            "lighting is flat",
            "too flat",
            "low contrast",
            "contrast is too low",
            "weak subject separation",
            "looks like a silhouette",
            "generic dark shape",
            "losing a lot of detail",
        )
    )
    god_rays_overdone = any(
        hint in text
        for hint in (
            "god ray",
            "god rays",
            "rays are too strong",
            "too strong and distracting",
            "look a bit artificial",
            "artificial or \"god ray\" heavy",
            "artificial or 'god ray' heavy",
        )
    )
    too_cool = any(
        hint in text
        for hint in (
            "too cool",
            "cooler",
            "blue cast",
            "bluer cast",
            "cool, blue",
            "warmer",
        )
    )
    weak_contact = any(
        hint in text
        for hint in (
            "weak shadow",
            "weak shadows",
            "contact shadow is weak",
            "contact shadow is missing",
            "shadow is weak",
            "poorly grounded",
            "disconnected from the ground",
            "floating",
            "floaty",
            "hovering",
        )
    )
    intersect = any(
        hint in text
        for hint in (
            "intersect",
            "penetrat",
            "sinks into",
            "buried",
            "embedded in the road",
        )
    )
    too_large = any(
        hint in text
        for hint in (
            "too large",
            "looks huge",
            "giant prop",
            "front wheel is very large",
            "front wheel looks disproportionately large",
            "disproportionately large",
            "bit large for",
        )
    )
    implausible_placement = any(
        hint in text
        for hint in (
            "middle of a road lane",
            "middle of the road",
            "directly on the white lane",
            "white lane marking",
            "in the lane",
            "physically implausible",
            "belongs on the sidewalk",
            "belongs on the curb",
        )
    )
    too_small = any(hint in text for hint in ("too small", "looks tiny", "undersized"))
    harsh_shadow = any(
        hint in text
        for hint in (
            "harsh shadow",
            "harsh shadows",
            "too harsh",
            "silhouette effect",
            "deep shadow",
            "deeply shadowed",
            "high-contrast lighting",
        )
    )
    too_matte = any(
        hint in text
        for hint in (
            "too matte",
            "lacks highlights",
            "missing metallic sheen",
            "reflection is weak",
            "reflections are weak",
            "wet road reflection is missing",
            "sheen",
            "muddy",
            "dull",
        )
    )
    muddy = any(hint in text for hint in ("muddy", "muted", "desaturated", "washed out"))
    metallic_goal = any(hint in text for hint in ("metallic", "glossy", "sheen", "wetter look", "painted metal"))
    plastic = any(
        hint in text
        for hint in (
            "plastic",
            "plastic-like",
            "too smooth",
            "toy",
            "strange specular highlight",
            "specular highlight",
            "too glossy",
            "too sharp",
        )
    )
    glossy_goal = any(
        hint in text
        for hint in (
            "glossy ceramic",
            "wet-looking glaze",
            "wet look",
            "glaze",
            "metallic sheen",
            "painted metal",
        )
    )
    giant_scale = any(
        hint in text
        for hint in (
            "way too big",
            "gigantic",
            "giant prop",
            "dominates the frame",
            "huge on the road",
        )
    )

    if too_dark:
        _state_mul(scene, "env_strength_scale", 1.26, 0.5, 2.0)
        _state_mul(material, "value_scale", 1.16, 0.5, 1.5)
        _state_mul(material, "saturation_scale", 1.08, 0.5, 1.5)
        _state_mul(lighting, "key_scale", 1.15, 0.5, 2.0)
        changes.append("pair 控制：更强地提亮环境/材质值/key light，并轻微提升饱和度，回应 underexposed/too dark")

    if too_dark and any(hint in text for hint in ("almost black", "pitch black", "black blob", "silhouette")):
        _state_mul(scene, "env_strength_scale", 1.12, 0.5, 2.0)
        _state_mul(material, "value_scale", 1.08, 0.5, 1.5)
        _state_add(material, "specular_add", 0.03, -0.3, 0.5)
        changes.append("pair 联合控制：对 almost-black/silhouette 情况再额外提亮，并补少量高光恢复体积")

    if flat:
        _state_mul(lighting, "key_scale", 1.18, 0.5, 2.0)
        _state_mul(scene, "env_strength_scale", 1.06, 0.5, 2.0)
        if not plastic:
            _state_add(material, "specular_add", 0.03, -0.3, 0.5)
        changes.append("pair 控制：加强 key light，并轻微抬升环境亮度，回应 flat lighting / low contrast")

    if god_rays_overdone:
        _state_mul(scene, "env_strength_scale", 0.88, 0.5, 2.0)
        _state_mul(lighting, "key_scale", 0.94, 0.5, 2.0)
        changes.append("pair 控制：更明显地下压环境和主光，减轻过强 god rays / 人工体积光")

    if harsh_shadow:
        _state_mul(lighting, "key_scale", 0.9, 0.5, 2.0)
        _state_mul(scene, "env_strength_scale", 1.1, 0.5, 2.0)
        changes.append("pair 控制：压一点 key、抬一点环境光，减轻 harsh shadow / silhouette effect")

    if too_cool:
        _state_add(material, "hue_offset", 0.03, -0.15, 0.15)
        changes.append("pair 控制：向暖色偏移 hue，回应 cooler/bluer cast")

    if weak_contact:
        _state_mul(scene, "contact_shadow_strength", 1.25, 0.5, 2.0)
        if not intersect:
            _state_add(obj, "offset_z", -0.008, -0.1, 0.1)
        changes.append("pair 控制：增强 contact shadow，并轻微下压落地，回应 floating/weak shadow")
    elif intersect:
        _state_add(obj, "offset_z", 0.008, -0.1, 0.1)
        changes.append("pair 控制：轻微抬高物体，避免埋地")

    if too_large and not too_small:
        _state_mul(obj, "scale", 0.82 if giant_scale else 0.9, 0.7, 1.4)
        changes.append("pair 控制：缩小 object.scale，回应 giant/too large/perspective wheel oversized")
    elif too_small and not too_large:
        _state_mul(obj, "scale", 1.07, 0.7, 1.4)
        changes.append("pair 控制：放大 object.scale，回应 undersized")

    if implausible_placement:
        _state_mul(obj, "scale", 0.9, 0.7, 1.4)
        _state_mul(scene, "contact_shadow_strength", 1.12, 0.5, 2.0)
        changes.append("pair 控制：对明显不合理落位先轻微缩小并加强接地，减少 lane-center 违和感")

    if plastic:
        _state_add(material, "specular_add", -0.05, -0.3, 0.5)
        _state_add(material, "roughness_add", 0.07, -0.3, 0.6)
        _state_mul(material, "saturation_scale", 0.94, 0.5, 1.5)
        changes.append("pair 控制：减弱过 sharp specular，并提高 roughness，减轻 plastic-like 高光")
    elif too_matte or muddy or metallic_goal:
        _state_mul(material, "saturation_scale", 1.06, 0.5, 1.5)
        _state_add(material, "specular_add", 0.08, 0.0, 0.5)
        _state_add(material, "roughness_add", -0.06, -0.3, 0.6)
        changes.append("pair 控制：提高饱和度/specular 并降低 roughness，回应 matte/muddy/metallic reflection")

    if any(hint in text for hint in ("brownish-grey", "taupe", "brownish grey", "metallic brownish-grey", "metallic taupe")):
        _state_add(material, "hue_offset", 0.025, -0.15, 0.15)
        _state_mul(material, "saturation_scale", 0.92, 0.5, 1.5)
        changes.append("pair 控制：向 taupe/brownish-grey 拉回 hue，并轻微降饱和，回应车漆颜色偏黑")

    if too_dark and flat:
        _state_mul(scene, "env_strength_scale", 1.08, 0.5, 2.0)
        _state_mul(material, "value_scale", 1.06, 0.5, 1.5)
        _state_mul(lighting, "key_scale", 1.08, 0.5, 2.0)
        _state_mul(scene, "contact_shadow_strength", 1.12, 0.5, 2.0)
        changes.append("pair 联合控制：对 underexposed + flat 组合进一步提亮并加强主体分离")

    if too_dark and glossy_goal and not plastic:
        _state_add(material, "specular_add", 0.04, -0.3, 0.5)
        _state_add(material, "roughness_add", -0.04, -0.3, 0.6)
        changes.append("pair 联合控制：对深色釉面/金属材质补一点 sheen，避免发黑发闷")

    return state, changes


def run_benchmark_pair(args, benchmark: dict, runtime_dir: Path, iteration_idx: int, iteration_dir: Path) -> dict:
    pair_runtime = iteration_dir / benchmark["pair_name"]
    pair_runtime.mkdir(parents=True, exist_ok=True)
    base_state_path = _find_previous_pair_state(runtime_dir, benchmark["pair_name"], iteration_idx) or Path(benchmark["state_path"])
    feedback_text = _find_previous_pair_trace(runtime_dir, benchmark["pair_name"], iteration_idx) or str(benchmark.get("trace_text") or "")
    base_state = load_json(base_state_path, default={}) or {}
    tuned_state, control_changes = build_pair_control_state(base_state, feedback_text)
    tuned_state_path = pair_runtime / "control_state.tuned.json"
    save_json(tuned_state_path, tuned_state)
    save_json(
        pair_runtime / "control_adjustments.json",
        {
            "pair_name": benchmark["pair_name"],
            "iteration_idx": iteration_idx,
            "base_state_path": str(base_state_path),
            "feedback_trace_excerpt": feedback_text[:3000],
            "control_changes": control_changes,
        },
    )
    agent_note = " | ".join(control_changes[:4])
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "run_scene_agent_step.py"),
        "--output-dir",
        str(pair_runtime),
        "--obj-id",
        benchmark["obj_id"],
        "--rotation-deg",
        str(benchmark["rotation_deg"]),
        "--round-idx",
        "0",
        "--agent-note",
        agent_note,
        "--control-state-in",
        str(tuned_state_path),
        "--device",
        args.device,
        "--blender",
        args.blender_bin,
        "--meshes-dir",
        args.meshes_dir,
        "--profile",
        args.profile,
        "--scene-template",
        args.template_path,
    ]
    rc = _run(cmd)
    result_path = pair_runtime / "agent_round00.json"
    result = load_json(result_path, default={}) or {}
    result["return_code"] = rc
    result["pair_name"] = benchmark["pair_name"]
    result["obj_id"] = benchmark["obj_id"]
    result["rotation_deg"] = benchmark["rotation_deg"]
    result["base_state_path"] = str(base_state_path)
    result["tuned_state_path"] = str(tuned_state_path)
    result["control_changes"] = control_changes
    result["feedback_trace_excerpt"] = feedback_text[:3000]
    review_agg_path = pair_runtime / "reviews" / f"{benchmark['obj_id']}_r00_agg.json"
    result["review_agg_path"] = str(review_agg_path) if review_agg_path.exists() else None
    if review_agg_path.exists():
        review_agg = load_json(review_agg_path, default={}) or {}
        result["issue_tags"] = review_agg.get("issue_tags") or []
        result["lighting_diagnosis"] = review_agg.get("lighting_diagnosis")
        result["structure_consistency"] = review_agg.get("structure_consistency")
        result["hybrid_score"] = review_agg.get("hybrid_score", result.get("hybrid_score"))
    traces = sorted((pair_runtime / "reviews").glob("*_trace.json"))
    if traces:
        trace_text = extract_trace_text(load_json(traces[-1], default={}) or {})
        result["trace_text"] = trace_text
        result["reviewer_verdict"] = extract_reviewer_verdict(trace_text)
        result["reviewer_accepts"] = reviewer_accepts(trace_text)
    else:
        result["trace_text"] = ""
        result["reviewer_verdict"] = None
        result["reviewer_accepts"] = False
    result["asset_blocked"] = is_asset_blocked_trace(result["trace_text"]) or is_asset_blocked_result(result)
    return result


def summarize_iteration(results: list[dict]) -> dict:
    scorable_results = [item for item in results if not item.get("asset_blocked")]
    scores = [float(item.get("hybrid_score") or 0.0) for item in scorable_results if item.get("hybrid_score") is not None]
    mean_hybrid = statistics.fmean(scores) if scores else 0.0
    issue_counter = Counter()
    text_counter = Counter()
    accepted_pairs: list[str] = []
    asset_blocked_pairs = [str(item.get("pair_name") or "") for item in results if item.get("asset_blocked")]
    for item in scorable_results:
        for tag in item.get("issue_tags") or []:
            issue_counter[str(tag)] += 1
        text = str(item.get("trace_text") or "").lower()
        if item.get("reviewer_accepts"):
            accepted_pairs.append(str(item.get("pair_name") or ""))
        if "plastic" in text or "cgi-like" in text:
            text_counter["plastic"] += 1
        if "floating" in text or "hovering" in text:
            text_counter["floating_visible"] += 1
        if "too bright" in text or "washed out" in text or "too pale" in text:
            text_counter["too_bright"] += 1
        if "too dark" in text or "too dim" in text or "gloomy" in text:
            text_counter["too_dark"] += 1
        if "too matte" in text or "lacks highlights" in text:
            text_counter["too_matte"] += 1
        if "god ray" in text or "god rays" in text or "rays are too strong" in text or "artificial" in text:
            text_counter["god_rays_overdone"] += 1
    return {
        "mean_hybrid_score": round(mean_hybrid, 4),
        "issue_counts": dict(issue_counter),
        "text_counts": dict(text_counter),
        "asset_blocked_pairs": [pair for pair in asset_blocked_pairs if pair],
        "accepted_pairs": [pair for pair in accepted_pairs if pair],
        "accepted_count": len([pair for pair in accepted_pairs if pair]),
    }


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def apply_feedback_to_template(template: dict, summary: dict) -> tuple[dict, list[str]]:
    template = dict(template)
    issues = summary["issue_counts"]
    texts = summary["text_counts"]
    changes: list[str] = []

    flat = int(issues.get("flat_lighting", 0))
    separation = int(issues.get("weak_subject_separation", 0))
    plastic = int(texts.get("plastic", 0))
    too_bright = int(texts.get("too_bright", 0)) + int(issues.get("overexposed", 0))
    too_dark = int(texts.get("too_dark", 0)) + int(issues.get("underexposed", 0))
    floating = int(texts.get("floating_visible", 0)) + int(issues.get("floating_visible", 0))
    intersect = int(issues.get("ground_intersection_visible", 0)) + int(issues.get("ground_intersection", 0))
    too_large = int(issues.get("object_too_large", 0))
    too_small = int(issues.get("object_too_small", 0))
    color_shift = int(issues.get("color_shift", 0))
    too_matte = int(texts.get("too_matte", 0))
    god_rays_overdone = int(texts.get("god_rays_overdone", 0))

    if flat or separation:
        template["add_key_light"] = True
        template["key_light_energy"] = round(_clamp(float(template.get("key_light_energy", 3.0)) + 0.7, 2.5, 7.0), 3)
        fill_mul = 1.03 if too_dark > too_bright else 0.92
        template["camera_fill_energy"] = round(_clamp(float(template.get("camera_fill_energy", 420.0)) * fill_mul, 180.0, 560.0), 3)
        template["scale_existing_lights"] = round(_clamp(float(template.get("scale_existing_lights", 0.3)) * 0.92, 0.16, 0.34), 4)
        template["scale_world_background"] = round(_clamp(float(template.get("scale_world_background", 0.24)) * 0.94, 0.14, 0.26), 4)
        changes.append("加强主体分离：开启/增强 key light，并按明暗情况微调 fill/world 强度")

    if (plastic or too_bright) and too_bright >= too_dark:
        template["target_brightness"] = round(_clamp(float(template.get("target_brightness", 0.36)) - 0.015, 0.28, 0.38), 4)
        template["camera_fill_energy"] = round(_clamp(float(template.get("camera_fill_energy", 420.0)) * 0.9, 160.0, 520.0), 3)
        template["scale_existing_lights"] = round(_clamp(float(template.get("scale_existing_lights", 0.3)) * 0.93, 0.16, 0.34), 4)
        template["scale_world_background"] = round(_clamp(float(template.get("scale_world_background", 0.24)) * 0.9, 0.12, 0.25), 4)
        changes.append("压低过曝/塑料感：降低亮度目标和环境/填充光")

    if too_dark > too_bright:
        template["target_brightness"] = round(_clamp(float(template.get("target_brightness", 0.36)) + 0.018, 0.28, 0.42), 4)
        template["camera_fill_energy"] = round(_clamp(float(template.get("camera_fill_energy", 420.0)) * 1.12, 180.0, 580.0), 3)
        template["scale_world_background"] = round(_clamp(float(template.get("scale_world_background", 0.24)) * 1.04, 0.12, 0.28), 4)
        template["key_light_energy"] = round(_clamp(float(template.get("key_light_energy", 3.0)) + 0.35, 2.5, 7.0), 3)
        changes.append("回补偏暗：优先提高目标亮度、fill light 和 world 背景")

    if god_rays_overdone:
        template["scale_world_background"] = round(_clamp(float(template.get("scale_world_background", 0.24)) * 0.86, 0.10, 0.28), 4)
        template["scale_existing_lights"] = round(_clamp(float(template.get("scale_existing_lights", 0.3)) * 0.9, 0.16, 0.34), 4)
        changes.append("压低过强 god rays：更明显地收 world 背景和现有灯光强度")

    if color_shift:
        template["camera_fill_color"] = [1.0, 0.992, 0.985]
        changes.append("减轻色偏：把 camera fill 调回更中性的暖白")

    if floating and not intersect:
        template["ground_contact_epsilon"] = round(_clamp(float(template.get("ground_contact_epsilon", 0.0014)) * 0.86, 0.0006, 0.0022), 6)
        template["contact_shadow_distance"] = round(_clamp(float(template.get("contact_shadow_distance", 0.04)) * 0.88, 0.015, 0.08), 4)
        changes.append("降低悬浮感：下调 ground_contact_epsilon 并收紧 contact shadow 距离")
    elif intersect:
        template["ground_contact_epsilon"] = round(_clamp(float(template.get("ground_contact_epsilon", 0.0014)) * 1.12, 0.0006, 0.0024), 6)
        changes.append("减轻埋地：上调 ground_contact_epsilon")

    if too_large >= 1 and too_large > too_small:
        lo, hi = template.get("target_bbox_height_range", [0.3, 1.5])
        scale = 0.9 if too_large >= 2 else 0.96
        template["target_bbox_height_range"] = [round(_clamp(float(lo) * scale, 0.22, 0.8), 4), round(_clamp(float(hi) * scale, 0.9, 1.6), 4)]
        changes.append("缩小物体默认落地尺寸范围")
    elif too_small >= 2 and too_small > too_large:
        lo, hi = template.get("target_bbox_height_range", [0.3, 1.5])
        scale = 1.06
        template["target_bbox_height_range"] = [round(_clamp(float(lo) * scale, 0.22, 0.8), 4), round(_clamp(float(hi) * scale, 0.9, 1.7), 4)]
        changes.append("放大物体默认落地尺寸范围")

    if too_matte and not plastic:
        template["camera_fill_energy"] = round(_clamp(float(template.get("camera_fill_energy", 420.0)) * 0.96, 180.0, 540.0), 3)
        changes.append("避免过 matte：略降 fill，让高光更能站出来")

    return template, changes


def apply_fallback_exploration(template: dict, summary: dict, iteration_idx: int) -> tuple[dict, list[str]]:
    template = dict(template)
    issues = summary["issue_counts"]
    texts = summary["text_counts"]
    changes: list[str] = []

    too_dark = int(texts.get("too_dark", 0)) + int(issues.get("underexposed", 0))
    too_bright = int(texts.get("too_bright", 0)) + int(issues.get("overexposed", 0))
    flat = int(issues.get("flat_lighting", 0))
    separation = int(issues.get("weak_subject_separation", 0))
    intersect = int(issues.get("ground_intersection_visible", 0)) + int(issues.get("ground_intersection", 0))
    floating = int(texts.get("floating_visible", 0)) + int(issues.get("floating_visible", 0))

    if too_dark > too_bright:
        before = float(template.get("target_brightness", 0.36))
        after = round(_clamp(before + 0.006, 0.28, 0.42), 4)
        if after != before:
            template["target_brightness"] = after
            changes.append("fallback 探索：继续轻微提亮目标亮度")
            return template, changes

    if too_bright >= too_dark:
        before = float(template.get("camera_fill_energy", 420.0))
        factor = 0.95 if iteration_idx % 2 else 0.92
        after = round(_clamp(before * factor, 160.0, 620.0), 3)
        if after != before:
            template["camera_fill_energy"] = after
            changes.append("fallback 探索：继续下压 fill light，避免发灰发白")
            return template, changes

    if flat or separation:
        before = float(template.get("key_light_energy", 3.0))
        after = round(_clamp(before + 0.25, 2.5, 7.0), 3)
        if after != before:
            template["key_light_energy"] = after
            changes.append("fallback 探索：继续加重 key light，强化主体分离")
            return template, changes

    if intersect or floating:
        before = float(template.get("ground_contact_epsilon", 0.0014))
        factor = 1.08 if intersect else 0.92
        after = round(_clamp(before * factor, 0.0005, 0.0026), 6)
        if after != before:
            template["ground_contact_epsilon"] = after
            if not intersect:
                shadow_before = float(template.get("contact_shadow_distance", 0.04))
                template["contact_shadow_distance"] = round(_clamp(shadow_before * 0.94, 0.015, 0.08), 4)
            changes.append("fallback 探索：继续微调 ground contact / contact shadow")
            return template, changes

    before = float(template.get("scale_world_background", 0.24))
    delta = -0.01 if iteration_idx % 2 else 0.01
    after = round(_clamp(before + delta, 0.10, 0.28), 4)
    if after != before:
        template["scale_world_background"] = after
        changes.append("fallback 探索：交替扰动 world background 强度")
        return template, changes

    return template, changes


def template_diff(before: dict, after: dict) -> dict:
    diff = {}
    keys = sorted(set(before) | set(after))
    for key in keys:
        if before.get(key) != after.get(key):
            diff[key] = {"before": before.get(key), "after": after.get(key)}
    return diff


def main():
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    template_path = Path(args.template_path).resolve()
    runtime_dir = Path(args.runtime_dir).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    history_path = runtime_dir / "tuning_history.json"
    done_path = runtime_dir / "render_tuner_done.json"
    active_path = runtime_dir / "render_tuner_active.json"
    backup_path = runtime_dir / "scene_template.before_tuning.json"

    base_template = load_json(template_path, default={}) or {}
    if not backup_path.exists():
        save_json(backup_path, base_template)

    candidates = collect_candidates(root_dir)
    candidate_map = {item["pair_name"]: item for item in candidates}
    history = load_json(history_path, default={}) or {}
    benchmark_records = []
    history_needs_reset = False
    if isinstance(history.get("benchmarks"), list) and history.get("benchmarks"):
        for record in history["benchmarks"]:
            pair_name = str(record.get("pair_name") or "")
            candidate = candidate_map.get(pair_name)
            latest_runtime_trace = _find_latest_pair_trace_across_tuner_runs(runtime_dir, pair_name)
            latest_runtime_result = _find_latest_pair_result_across_tuner_runs(runtime_dir, pair_name)
            blocked = (
                is_asset_blocked_trace(latest_runtime_trace or (candidate.get("trace_text") if candidate else ""))
                or is_asset_blocked_result(latest_runtime_result)
            )
            if candidate and not blocked:
                benchmark_records.append(candidate)
    if not benchmark_records:
        benchmark_records = select_benchmarks(candidates, args.benchmark_limit)
        history_needs_reset = True
    elif len(benchmark_records) < args.benchmark_limit:
        existing = {item["pair_name"] for item in benchmark_records}
        for candidate in select_benchmarks(candidates, args.benchmark_limit * 2):
            if candidate["pair_name"] in existing or candidate.get("asset_blocked"):
                continue
            benchmark_records.append(candidate)
            existing.add(candidate["pair_name"])
            if len(benchmark_records) >= args.benchmark_limit:
                break
        history_needs_reset = True
    if history_needs_reset:
        history = {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "root_dir": str(root_dir),
            "template_path": str(template_path),
            "benchmarks": [
                {
                    "pair_name": item["pair_name"],
                    "obj_id": item["obj_id"],
                    "rotation_deg": item["rotation_deg"],
                    "source_round_idx": item["round_idx"],
                    "source_hybrid_score": item["hybrid_score"],
                    "source_issue_tags": item["issue_tags"],
                }
                for item in benchmark_records
            ],
            "iterations": [],
        }
    benchmarks = benchmark_records
    if not benchmarks:
        save_json(done_path, {"status": "no_benchmarks", "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        return 0

    completed_iterations = len(history.get("iterations") or [])
    prior_scores = [float(item.get("mean_hybrid_score") or 0.0) for item in history.get("iterations") or []]
    best_mean_hybrid = max(prior_scores) if prior_scores else -math.inf
    start_iteration = completed_iterations + 1
    for iteration_idx in range(start_iteration, args.max_iterations + 1):
        save_json(
            active_path,
            {
                "status": "running",
                "iteration_idx": iteration_idx,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        iteration_dir = runtime_dir / f"iter_{iteration_idx:02d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        results = [run_benchmark_pair(args, bench, runtime_dir, iteration_idx, iteration_dir) for bench in benchmarks]
        summary = summarize_iteration(results)
        mean_hybrid = float(summary["mean_hybrid_score"])
        improvement = 0.0 if best_mean_hybrid == -math.inf else mean_hybrid - best_mean_hybrid

        template_before = load_json(template_path, default={}) or {}
        template_after, changes = apply_feedback_to_template(template_before, summary)
        diff = template_diff(template_before, template_after)
        if not diff:
            template_after, fallback_changes = apply_fallback_exploration(template_before, summary, iteration_idx)
            diff = template_diff(template_before, template_after)
            changes.extend(fallback_changes)

        iteration_record = {
            "iteration_idx": iteration_idx,
            "mean_hybrid_score": summary["mean_hybrid_score"],
            "improvement_vs_best": round(improvement, 4),
            "issue_counts": summary["issue_counts"],
            "text_counts": summary["text_counts"],
            "asset_blocked_pairs": summary["asset_blocked_pairs"],
            "accepted_pairs": summary["accepted_pairs"],
            "accepted_count": summary["accepted_count"],
            "changes": changes,
            "template_diff": diff,
            "results": results,
        }
        history["iterations"].append(iteration_record)
        save_json(history_path, history)

        result_map = {str(item.get("pair_name") or ""): item for item in results}
        for benchmark in benchmarks:
            latest = result_map.get(benchmark["pair_name"])
            if not latest:
                continue
            state_out = latest.get("state_out")
            if state_out and Path(state_out).exists():
                benchmark["state_path"] = Path(state_out)
            trace_text = str(latest.get("trace_text") or "")
            if trace_text:
                benchmark["trace_text"] = trace_text

        scorable_pair_count = len([item for item in results if not item.get("asset_blocked")])
        good_enough = scorable_pair_count > 0 and summary["accepted_count"] >= scorable_pair_count
        if mean_hybrid > best_mean_hybrid:
            best_mean_hybrid = mean_hybrid

        if good_enough:
            save_json(
                done_path,
                {
                    "status": "good_enough",
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "iteration_idx": iteration_idx,
                    "mean_hybrid_score": summary["mean_hybrid_score"],
                    "accepted_pairs": summary["accepted_pairs"],
                },
            )
            if active_path.exists():
                active_path.unlink()
            return 0

        if diff:
            save_json(template_path, template_after)

    save_json(
        done_path,
        {
            "status": "max_iterations_reached",
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "iteration_idx": args.max_iterations,
            "best_mean_hybrid_score": round(best_mean_hybrid, 4) if best_mean_hybrid > -math.inf else None,
            "last_accepted_pairs": history["iterations"][-1]["accepted_pairs"] if history["iterations"] else [],
        },
    )
    if active_path.exists():
        active_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
