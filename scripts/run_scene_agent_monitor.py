"""
Persistent monitor for the scene freeform-VLM loop.

This script watches an existing rotation4 agent root, reads the latest
freeform Qwen3.5 review text for each (obj, yaw) pair, chooses the next
explicit actions, and launches the next round with run_scene_agent_step.py.

Unlike the older automatic controller loop:
- it reads the freeform review text (`assistant_text`) first
- it records an explicit decision file for every round
- it stops a pair only when the reviewer says `Verdict: keep`

There is still a high-round safety cap. The loop should stop at "good enough",
not chase marginal gains forever, so a hard cap is enforced even if a larger
value is passed on the command line.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
PIPELINE_ROOT = REPO_ROOT / "pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from asset_lifecycle import (
    asset_status_blocks_scene_loop,
    detect_asset_viability,
    detect_verdict_from_review,
    ensure_deprecated_and_enqueue,
    evaluate_deprecation_for_pair,
    get_registry_entry,
)

DEFAULT_BLENDER = os.environ.get("BLENDER_BIN", "/home/wuwenzhuo/blender-4.24/blender")
DEFAULT_PROFILE = REPO_ROOT / "configs" / "dataset_profiles" / "scene_v7.json"
DEFAULT_SCENE_TEMPLATE = REPO_ROOT / "configs" / "scene_template.json"
DEFAULT_MESHES_DIR = REPO_ROOT / "pipeline" / "data" / "meshes"
DEFAULT_STEP_PYTHON = "/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python"
HARD_MAX_ROUNDS = 15


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor scene freeform loop and keep iterating until VLM says keep")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument(
        "--gpus",
        default="0,1,2",
        help="Worker GPU groups. Use ',' for GPUs inside one review group and ';' between workers, e.g. '0,1;2,3'.",
    )
    parser.add_argument("--python", dest="python_bin", default=DEFAULT_STEP_PYTHON)
    parser.add_argument("--blender", dest="blender_bin", default=DEFAULT_BLENDER)
    parser.add_argument("--meshes-dir", default=str(DEFAULT_MESHES_DIR))
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--scene-template", default=str(DEFAULT_SCENE_TEMPLATE))
    parser.add_argument("--max-rounds", type=int, default=HARD_MAX_ROUNDS)
    parser.add_argument("--pass-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--launch-stagger-seconds", type=float, default=2.0)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-gpu", default=None)
    parser.add_argument("--worker-shard-dir", default=None)
    parser.add_argument("--worker-log-path", default=None)
    parser.add_argument("--allow-unsafe-single-gpu-review", action="store_true")
    parser.add_argument("--disable-asset-deprecation", action="store_true")
    parser.add_argument("--ignore-registry-blocks", action="store_true")
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


def parse_gpus(raw: str) -> List[str]:
    raw = str(raw or "").strip()
    if not raw:
        raise ValueError("No GPUs specified")
    return [item.strip() for item in raw.split(";") if item.strip()] if ";" in raw else [raw]


def slugify_gpu(gpu: str) -> str:
    return gpu.replace(":", "_").replace("/", "_").replace("\\", "_").replace(",", "_")


def parse_visible_gpu_ids(raw: str) -> List[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def prepare_worker_shard_dir(shard_root: Path, log_root: Path, gpu_slug: str) -> Optional[Path]:
    source_dirs = [
        shard_root / f"shard_gpu_{gpu_slug}",
        shard_root / f"resume_gpu_{gpu_slug}",
    ]
    pair_dirs: List[Path] = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        pair_dirs.extend(
            sorted(
                p for p in source_dir.iterdir()
                if p.is_dir() and p.name.startswith("obj_")
            )
        )
    if not pair_dirs:
        return None

    merged_root = log_root / "merged_worker_shards"
    merged_root.mkdir(parents=True, exist_ok=True)
    worker_dir = merged_root / f"gpu_{gpu_slug}"
    worker_dir.mkdir(parents=True, exist_ok=True)

    desired = {pair_dir.name: pair_dir.resolve() for pair_dir in pair_dirs}
    for existing in list(worker_dir.iterdir()):
        if existing.name not in desired:
            if existing.is_symlink() or existing.is_file():
                existing.unlink()
            elif existing.is_dir():
                # Only symlinked pair dirs are expected here.
                continue
            continue
        if not existing.is_symlink():
            continue
        try:
            if existing.resolve() == desired[existing.name]:
                desired.pop(existing.name, None)
                continue
        except FileNotFoundError:
            pass
        existing.unlink()

    for name, target in desired.items():
        link_path = worker_dir / name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(target, target_is_directory=True)

    return worker_dir


def parse_pair_name(pair_dir: Path) -> Tuple[str, int]:
    name = pair_dir.name
    match = re.fullmatch(r"(obj_\d+)_yaw(\d+)", name)
    if not match:
        raise ValueError(f"Invalid pair dir name: {name}")
    return match.group(1), int(match.group(2))


def find_round_files(pair_dir: Path) -> Tuple[Optional[Path], Optional[Path], int]:
    reviews_dir = pair_dir / "reviews"
    aggs = sorted(reviews_dir.glob("*_agg.json"))
    traces = sorted(reviews_dir.glob("*_trace.json"))
    if not aggs:
        return None, None, -1
    latest_agg = aggs[-1]
    round_match = re.search(r"_r(\d+)_agg\.json$", latest_agg.name)
    round_idx = int(round_match.group(1)) if round_match else -1
    latest_trace = None
    if traces:
        expected = reviews_dir / latest_agg.name.replace("_agg.json", "_az000_el+00_trace.json")
        latest_trace = expected if expected.exists() else traces[-1]
    return latest_agg, latest_trace, round_idx


def find_latest_agent_result(pair_dir: Path) -> Tuple[Optional[Path], Optional[dict], int]:
    agent_files = sorted(pair_dir.glob("agent_round*.json"))
    if not agent_files:
        return None, None, -1
    latest = agent_files[-1]
    match = re.search(r"agent_round(\d+)\.json$", latest.name)
    round_idx = int(match.group(1)) if match else -1
    payload = load_json(latest, default={}) or {}
    return latest, payload, round_idx


def get_pair_max_round(pair_dir: Path) -> int:
    rounds = [
        int(d.name[5:7])
        for d in pair_dir.iterdir()
        if d.is_dir()
        and d.name.startswith("round")
        and d.name.endswith("_renders")
        and d.name[5:7].isdigit()
    ]
    return max(rounds) if rounds else -1


def _extract_feedback_excerpt_text(agg: dict) -> str:
    excerpts = agg.get("freeform_feedback_excerpt") or []
    chunks: List[str] = []
    for item in excerpts:
        if not isinstance(item, dict):
            continue
        feedback = item.get("feedback")
        if isinstance(feedback, str) and feedback.strip():
            chunks.append(feedback.strip())
    return "\n\n".join(chunks)


def extract_trace_text(trace_payload: dict, agg: Optional[dict] = None) -> str:
    agg = agg or {}
    attempts = trace_payload.get("attempts") or []
    if attempts:
        attempt = attempts[-1]
        text = attempt.get("assistant_text")
        if isinstance(text, str) and text.strip():
            trace_text = text.strip()
            excerpt_text = _extract_feedback_excerpt_text(agg)
            if excerpt_text:
                trace_norm = " ".join(trace_text.split())
                excerpt_norm = " ".join(excerpt_text.split())
                if excerpt_norm and excerpt_norm not in trace_norm:
                    return f"{trace_text}\n\n{excerpt_text}"
            return trace_text
    for key in ("assistant_text", "raw_text", "response_text", "content", "answer", "model_output"):
        value = trace_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _extract_feedback_excerpt_text(agg)


def detect_verdict(text: str, agg: dict) -> str:
    return detect_verdict_from_review(text, agg)


def _contains_any(text: str, needles: List[str]) -> bool:
    return any(needle in text for needle in needles)


def _suggested_actions(agg: dict) -> List[str]:
    return [str(v) for v in (agg.get("suggested_actions") or []) if v]


def _prune_conflicting_actions(actions: List[str], reasons: List[str]) -> Tuple[List[str], List[str]]:
    kept_actions: List[str] = []
    kept_reasons: List[str] = []
    seen = set()

    conflicts = {
        "O_SCALE_UP_10": {"O_SCALE_DOWN_10"},
        "O_SCALE_DOWN_10": {"O_SCALE_UP_10"},
        "O_LIFT_SMALL": {"O_LOWER_SMALL"},
        "O_LOWER_SMALL": {"O_LIFT_SMALL"},
        "M_VALUE_UP": {"M_VALUE_DOWN", "M_VALUE_DOWN_STRONG"},
        "M_VALUE_DOWN": {"M_VALUE_UP"},
        "M_VALUE_DOWN_STRONG": {"M_VALUE_UP"},
        "M_HUE_WARM_POS": {"M_HUE_WARM_NEG"},
        "M_HUE_WARM_NEG": {"M_HUE_WARM_POS"},
        "M_ROUGHNESS_UP": {"M_ROUGHNESS_DOWN"},
        "M_ROUGHNESS_DOWN": {"M_ROUGHNESS_UP"},
    }

    for action, reason in zip(actions, reasons):
        if action in seen:
            continue
        if any(conflict in seen for conflict in conflicts.get(action, set())):
            continue
        kept_actions.append(action)
        kept_reasons.append(reason)
        seen.add(action)

    return kept_actions, kept_reasons


def decide_actions(agg: dict, trace_text: str, round_idx: int) -> Tuple[List[str], List[str]]:
    text = (trace_text or "").lower()
    issues = {str(v).lower() for v in (agg.get("issue_tags") or [])}
    suggestions = _suggested_actions(agg)
    physics = agg.get("programmatic_physics") or {}
    lighting = str(agg.get("lighting_diagnosis") or "").lower()
    color_consistency = str(agg.get("color_consistency") or "").lower()

    chosen: List[str] = []
    reasons: List[str] = []
    blocked: set[str] = set()

    def add(action: str, reason: str):
        if action and action not in blocked and action not in chosen:
            chosen.append(action)
            reasons.append(reason)

    too_small = "object_too_small" in issues or _contains_any(text, ["too small", "miniature", "toy", "undersized"])
    too_large = "object_too_large" in issues or _contains_any(text, ["too large", "oversized", "giant", "huge"])
    detail_obscured = _contains_any(
        text,
        [
            "silhouetted",
            "silhouette",
            "almost black",
            "hard to see details",
            "details are hard to see",
            "details are lost",
            "details in the shadows",
            "hard to read",
        ],
    )
    strange_texture = _contains_any(
        text,
        [
            "cracked",
            "veiny",
            "weirdly solid",
            "solid wheels",
            "wheel texture",
            "wheels look weird",
            "wheels have a strange",
            "strange texture",
        ],
    )
    too_dark = "underexposed" in issues or detail_obscured or _contains_any(text, ["too dark", "too black", "pitch black", "much darker"])
    too_bright = "overexposed" in issues or _contains_any(text, ["too bright", "too light", "too pale", "washed out"])
    pink_or_purple = _contains_any(text, ["pinkish", "too pink", "pink/purple", "pink", "purple", "magenta", "violet"])
    too_warm = _contains_any(text, ["too orange", "too reddish", "too red", "too yellow", "too warm"])
    too_cool = pink_or_purple or _contains_any(text, ["too blue", "too cool", "bluish", "purple cast", "magenta cast"])
    too_sat = _contains_any(text, ["too saturated", "too vibrant"])
    explicit_plastic = _contains_any(text, ["plastic", "cgi-like", "sticker", "pasted on", "too smooth", "plastic-like"])
    explicit_matte = _contains_any(text, ["too matte", "lacks highlights", "lacking the subtle highlights", "flat white stripes"])
    direct_roughness_down = "m_roughness_down" in text or _contains_any(text, ["lower roughness", "roughness down"])
    direct_sheen_up = "m_sheen_up" in text
    wants_shinier = _contains_any(
        text,
        [
            "shinier",
            "more reflective",
            "highly reflective",
            "needs highlights",
            "lacks highlights",
            "wetter",
            "glossier",
            "specular",
            "reflective strips",
            "reflective tape",
            "too matte",
        ],
    )
    wants_warmer = (
        too_cool
        or "color_shift" in issues
        or color_consistency == "major_shift"
        or _contains_any(text, ["dull orange", "muddy orange", "desaturated and cool", "cool compared to the reference"])
    )
    too_plastic = explicit_plastic
    too_matte = explicit_matte
    ground_intersection = (
        "ground_intersection_visible" in issues
        or "ground_intersection" in issues
        or _contains_any(text, ["intersecting the ground", "sinking into", "cuts into the road", "ground intersection", "pressed into the ground"])
    )
    explicit_floating = _contains_any(text, ["floating", "hovering", "not contacting", "floating slightly", "sitting on top without"])
    flat_base_against_ground = _contains_any(
        text,
        ["flat against the ground", "base is very flat", "base looks flat", "pressed flat", "very flat and wide"],
    )
    shadow_only_floating = explicit_floating and flat_base_against_ground and _contains_any(
        text,
        ["poorly lit", "shadow is very soft", "shadow is weak", "contact shadow"],
    )
    should_lower_for_ground = (
        bool(physics.get("is_floating"))
        or float(physics.get("contact_gap") or 0.0) > 0.002
        or (explicit_floating and not flat_base_against_ground and not shadow_only_floating)
    )
    weak_ground = ground_intersection or explicit_floating or _contains_any(
        text,
        ["contact shadow", "weak grounding", "shadow is too weak", "anchored poorly", "needs stronger shadow"],
    )
    weak_separation = "weak_subject_separation" in issues or _contains_any(
        text,
        ["weak subject separation", "doesn't stand out", "subject separation", "does not pop", "doesn't pop", "blends into the scene"],
    )
    flat_lighting = "flat_lighting" in issues or lighting == "flat_low_contrast" or _contains_any(text, ["flat lighting", "flat", "low contrast"])
    strong_god_rays = _contains_any(
        text,
        [
            "god rays",
            "sunbeams",
            "crepuscular rays",
            "volumetric light",
            "rays are too strong",
            "strong rays",
            "too foggy",
            "very foggy",
            "very hazy",
        ],
    )
    needs_more_light = too_dark or _contains_any(text, ["gloomy", "needs more light", "too dim", "not enough light"])
    needs_less_light = too_bright or _contains_any(text, ["washed out", "too pale"])

    # If text heuristics fire both ways, trust explicit issue tags first.
    if too_small and too_large:
        if "object_too_small" in issues and "object_too_large" not in issues:
            too_large = False
        elif "object_too_large" in issues and "object_too_small" not in issues:
            too_small = False
        else:
            too_small = False
            too_large = False
    if too_dark and too_bright:
        if "underexposed" in issues and "overexposed" not in issues:
            too_bright = False
        elif "overexposed" in issues and "underexposed" not in issues:
            too_dark = False
        else:
            too_dark = False
            too_bright = False
    if too_warm and too_cool:
        if pink_or_purple:
            too_warm = False
        else:
            too_cool = False
    if too_plastic and too_matte:
        if explicit_plastic and not explicit_matte:
            too_matte = False
        elif explicit_matte and not explicit_plastic:
            too_plastic = False
        else:
            too_matte = False

    if ground_intersection:
        blocked.add("O_LOWER_SMALL")
    if explicit_floating or float(physics.get("contact_gap") or 0.0) > 0.002:
        blocked.add("O_LIFT_SMALL")
    if flat_base_against_ground:
        blocked.add("O_LOWER_SMALL")
    if wants_shinier or too_matte:
        blocked.add("M_ROUGHNESS_UP")
    if too_plastic and not wants_shinier and not too_matte:
        blocked.add("M_SHEEN_UP")
    if flat_lighting and not needs_more_light:
        blocked.add("ENV_STRENGTH_UP")
    if strong_god_rays:
        blocked.add("ENV_STRENGTH_UP")
        blocked.add("L_KEY_UP")
    if needs_less_light:
        blocked.add("M_VALUE_UP")
    if needs_more_light:
        blocked.add("M_VALUE_DOWN")
    if too_warm:
        blocked.add("M_HUE_WARM_POS")
    if wants_warmer:
        blocked.add("M_HUE_WARM_NEG")

    if too_small:
        add("O_SCALE_UP_10", "VLM 明确说物体太小/像玩具，需要放大")
    if too_large:
        add("O_SCALE_DOWN_10", "VLM 明确说物体过大，需要缩小")
    if weak_ground:
        add("S_CONTACT_SHADOW_UP", "VLM 反复指出接地差/阴影弱，先加强接触阴影")
        if ground_intersection:
            add("O_LIFT_SMALL", "VLM 明确说物体压地/切地，先轻微抬起")
        elif should_lower_for_ground:
            add("O_LOWER_SMALL", "程序物理显示有接触间隙，轻微下压")

    for action in suggestions:
        add(action, "补充使用 reviewer 建议动作")
        if len(chosen) >= 4:
            break

    if needs_more_light:
        add("ENV_STRENGTH_UP", "VLM 说整体过黑/像剪影/细节看不清，先抬环境光把暗部细节救回来")
        add("M_VALUE_UP", "VLM 说过黑/过暗或不够亮，先提亮材质值")
        if detail_obscured or strange_texture:
            add("L_KEY_UP", "VLM 说细节看不清或轮组纹理怪，补一点主光让轮廓与局部结构更清楚")
    if needs_less_light:
        add("M_VALUE_DOWN", "VLM 说过亮/过浅，先压低材质亮度")
    if too_warm:
        add("M_HUE_WARM_NEG", "VLM 说偏橙/偏红/偏黄，往更冷更中性的方向拉")
        add("M_SATURATION_DOWN", "偏暖通常伴随过饱和，顺手压一点饱和度")
    if wants_warmer:
        add("M_HUE_WARM_POS", "VLM 明确说偏冷/偏紫/偏蓝或整体偏 dull，往更中性偏暖的方向拉回")
    elif too_sat:
        add("M_SATURATION_DOWN", "VLM 说过饱和/太鲜艳，先降饱和")
    if direct_roughness_down:
        add("M_ROUGHNESS_DOWN", "VLM 自由文本直接点名降低粗糙度，让材质更湿更亮")
    elif direct_sheen_up or wants_shinier or too_matte:
        add("M_SHEEN_UP", "VLM 明确要求更亮的高光/更湿更反光，优先补 specular")
    elif too_plastic:
        add("M_ROUGHNESS_UP", "VLM 说塑料感/过于平滑，先加粗糙度压掉廉价塑料感")
    if flat_lighting or weak_separation:
        add("L_KEY_UP", "VLM 说主体分离弱/光照过平，优先加强主光让主体更出挑")
        if needs_more_light:
            add("L_KEY_UP", "整体偏暗时补主光，比直接抬环境更稳")
    if strong_god_rays:
        add("ENV_STRENGTH_DOWN", "VLM 说 god rays / sunbeams / 雾太重，先压低环境强度减少体积光感")
        add("L_KEY_DOWN", "VLM 说光柱过强/太雾，先轻压主光避免继续把高光和雾打爆")

    if not chosen:
        if round_idx == 0:
            add("S_CONTACT_SHADOW_UP", "默认先提升接地阴影")
            add("L_KEY_UP", "默认先提升主体与背景的明暗分离")
        else:
            add("S_CONTACT_SHADOW_UP", "没有明确信号时优先继续加强接地")
            add("L_KEY_UP", "没有明确信号时优先继续拉开主体与背景")

    chosen, reasons = _prune_conflicting_actions(chosen, reasons)
    return chosen[:4], reasons[:4]


def build_decision_payload(pair_dir: Path, latest_agg: Path, latest_trace: Optional[Path], round_idx: int, agg: dict, trace_text: str) -> dict:
    verdict = detect_verdict(trace_text, agg)
    actions, reasons = decide_actions(agg, trace_text, round_idx)
    asset_viability, abandon_reason, abandon_confidence = detect_asset_viability(trace_text, agg)
    return {
        "pair_dir": str(pair_dir),
        "source_round_idx": round_idx,
        "source_agg_path": str(latest_agg),
        "source_trace_path": str(latest_trace) if latest_trace else None,
        "detected_verdict": verdict,
        "asset_viability": asset_viability,
        "abandon_reason": abandon_reason,
        "abandon_confidence": abandon_confidence,
        "hybrid_score": agg.get("hybrid_score"),
        "issue_tags": agg.get("issue_tags"),
        "lighting_diagnosis": agg.get("lighting_diagnosis"),
        "structure_consistency": agg.get("structure_consistency"),
        "programmatic_physics": agg.get("programmatic_physics"),
        "reviewer_excerpt": " ".join(trace_text.split())[:500],
        "chosen_actions": actions,
        "decision_reasons": reasons,
    }


def run_one_round(
    *,
    pair_dir: Path,
    obj_id: str,
    rotation_deg: int,
    next_round_idx: int,
    actions: List[str],
    python_bin: str,
    blender_bin: str,
    meshes_dir: str,
    profile: str,
    scene_template: str,
) -> int:
    prev_round = next_round_idx - 1
    state_in = pair_dir / "states" / f"round{prev_round:02d}.json"
    prev_renders = pair_dir / f"round{prev_round:02d}_renders"
    cmd = [
        python_bin,
        str(REPO_ROOT / "scripts" / "run_scene_agent_step.py"),
        "--output-dir",
        str(pair_dir),
        "--obj-id",
        obj_id,
        "--rotation-deg",
        str(rotation_deg),
        "--round-idx",
        str(next_round_idx),
        "--actions",
        *actions,
        "--agent-note",
        "monitor_loop_continue",
        "--control-state-in",
        str(state_in),
        "--prev-renders-dir",
        str(prev_renders),
        "--device",
        "cuda:0",
        "--blender",
        blender_bin,
        "--meshes-dir",
        meshes_dir,
        "--profile",
        profile,
        "--scene-template",
        scene_template,
    ]
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _handle_failed_launch(*, pair_dir: Path, obj_id: str, round_idx: int, decision: dict) -> dict:
    latest_agent_path, latest_agent, agent_round_idx = find_latest_agent_result(pair_dir)
    if latest_agent and latest_agent.get("error") == "render_failed":
        deprecation_result = ensure_deprecated_and_enqueue(
            obj_id,
            reason="render_failed_after_launch",
            trigger_source=f"{pair_dir.name}:round{round_idx:02d}:launch_failed",
            evidence={
                "agent_result_path": str(latest_agent_path) if latest_agent_path else None,
                "agent_result": latest_agent,
            },
        )
        decision = dict(decision)
        decision["chosen_actions"] = []
        decision["decision_reasons"] = ["asset deprecated because the launched round failed to render"]
        decision["deprecation_check"] = {
            "should_deprecate": True,
            "deprecation_kind": "hard",
            "reason": "render_failed_after_launch",
        }
        decision["deprecation_result"] = deprecation_result
        decision_name = f"round{agent_round_idx:02d}_decision.json" if agent_round_idx >= 0 else "asset_deprecation.json"
        save_json(pair_dir / "decisions" / decision_name, decision)
        return {
            "pair": pair_dir.name,
            "status": "deprecated_enqueued",
            "done": True,
            "latest_round_idx": agent_round_idx,
            "decision": decision,
            "return_code": 1,
        }
    return {
        "pair": pair_dir.name,
        "status": "launch_failed",
        "done": False,
        "latest_round_idx": round_idx - 1,
        "next_round_idx": round_idx,
        "decision": decision,
        "return_code": 1,
    }


def process_pair(
    *,
    pair_dir: Path,
    python_bin: str,
    blender_bin: str,
    meshes_dir: str,
    profile: str,
    scene_template: str,
    max_rounds: int,
    disable_asset_deprecation: bool,
    ignore_registry_blocks: bool,
) -> dict:
    obj_id, rotation_deg = parse_pair_name(pair_dir)
    decisions_dir = pair_dir / "decisions"

    registry_entry = get_registry_entry(obj_id)
    if (
        registry_entry
        and asset_status_blocks_scene_loop(registry_entry.get("status"))
        and not ignore_registry_blocks
    ):
        return {
            "pair": pair_dir.name,
            "status": "asset_waiting_replacement",
            "done": True,
            "asset_status": registry_entry.get("status"),
            "active_version": registry_entry.get("active_version"),
        }

    mesh_path = Path(meshes_dir) / f"{obj_id}.glb"
    if not mesh_path.exists() or mesh_path.stat().st_size <= 0:
        deprecation_result = ensure_deprecated_and_enqueue(
            obj_id,
            reason=f"mesh_missing_or_empty:{mesh_path}",
            trigger_source=f"{pair_dir.name}:mesh_check",
            evidence={"mesh_path": str(mesh_path)},
        )
        decision = {
            "pair_dir": str(pair_dir),
            "source_round_idx": -1,
            "source_agg_path": None,
            "source_trace_path": None,
            "detected_verdict": "reject",
            "asset_viability": "abandon",
            "abandon_reason": "mesh_missing_or_empty",
            "abandon_confidence": "high",
            "hybrid_score": None,
            "issue_tags": ["none"],
            "lighting_diagnosis": None,
            "structure_consistency": None,
            "programmatic_physics": None,
            "reviewer_excerpt": "",
            "chosen_actions": [],
            "decision_reasons": ["asset deprecated because mesh is missing or empty"],
            "deprecation_check": {
                "should_deprecate": True,
                "deprecation_kind": "hard",
                "reason": "mesh_missing_or_empty",
                "evidence": {"mesh_path": str(mesh_path)},
            },
            "deprecation_result": deprecation_result,
        }
        save_json(decisions_dir / "asset_deprecation.json", decision)
        return {
            "pair": pair_dir.name,
            "status": "deprecated_enqueued",
            "done": True,
            "latest_round_idx": -1,
            "decision": decision,
        }

    latest_agg, latest_trace, round_idx = find_round_files(pair_dir)
    if latest_agg is None:
        latest_agent_path, latest_agent, agent_round_idx = find_latest_agent_result(pair_dir)
        if latest_agent and latest_agent.get("error") == "render_failed":
            deprecation_result = ensure_deprecated_and_enqueue(
                obj_id,
                reason="render_failed_before_review",
                trigger_source=f"{pair_dir.name}:render_failed",
                evidence={
                    "agent_result_path": str(latest_agent_path) if latest_agent_path else None,
                    "agent_result": latest_agent,
                },
            )
            decision = {
                "pair_dir": str(pair_dir),
                "source_round_idx": agent_round_idx,
                "source_agg_path": None,
                "source_trace_path": None,
                "detected_verdict": "reject",
                "asset_viability": "abandon",
                "abandon_reason": "render_failed_before_review",
                "abandon_confidence": "high",
                "hybrid_score": None,
                "issue_tags": ["none"],
                "lighting_diagnosis": None,
                "structure_consistency": None,
                "programmatic_physics": None,
                "reviewer_excerpt": "",
                "chosen_actions": [],
                "decision_reasons": ["asset deprecated because render failed before review"],
                "deprecation_check": {
                    "should_deprecate": True,
                    "deprecation_kind": "hard",
                    "reason": "render_failed_before_review",
                },
                "deprecation_result": deprecation_result,
            }
            decision_name = f"round{agent_round_idx:02d}_decision.json" if agent_round_idx >= 0 else "asset_deprecation.json"
            save_json(decisions_dir / decision_name, decision)
            return {
                "pair": pair_dir.name,
                "status": "deprecated_enqueued",
                "done": True,
                "latest_round_idx": agent_round_idx,
                "decision": decision,
            }
        return {
            "pair": pair_dir.name,
            "status": "no_review_yet",
            "done": False,
            "message": "pair has no review yet",
        }

    agg = load_json(latest_agg, default={}) or {}
    trace_payload = load_json(latest_trace, default={}) if latest_trace else {}
    trace_text = extract_trace_text(trace_payload or {}, agg)
    decision = build_decision_payload(pair_dir, latest_agg, latest_trace, round_idx, agg, trace_text)
    verdict = decision["detected_verdict"]

    deprecation_check = evaluate_deprecation_for_pair(
        pair_dir=pair_dir,
        latest_round_idx=round_idx,
        latest_agg=agg,
        latest_trace_text=trace_text,
        latest_verdict=verdict,
    )
    decision["deprecation_check"] = deprecation_check

    if deprecation_check.get("should_deprecate") and not disable_asset_deprecation:
        deprecation_result = ensure_deprecated_and_enqueue(
            obj_id,
            reason=str(deprecation_check.get("reason") or "asset_deprecated"),
            trigger_source=f"{pair_dir.name}:round{round_idx:02d}",
            evidence=deprecation_check.get("evidence"),
        )
        decision["chosen_actions"] = []
        decision["decision_reasons"] = [f"asset deprecated: {deprecation_check.get('reason')}"]
        decision["deprecation_result"] = deprecation_result
    elif deprecation_check.get("should_deprecate") and disable_asset_deprecation:
        decision["deprecation_ignored"] = True
        decision["decision_reasons"] = list(decision.get("decision_reasons") or []) + [
            f"ignore asset deprecation and continue iterating: {deprecation_check.get('reason')}"
        ]

    save_json(decisions_dir / f"round{round_idx:02d}_decision.json", decision)

    if deprecation_check.get("should_deprecate") and not disable_asset_deprecation:
        return {
            "pair": pair_dir.name,
            "status": "deprecated_enqueued",
            "done": True,
            "latest_round_idx": round_idx,
            "decision": decision,
        }

    if verdict == "keep":
        return {
            "pair": pair_dir.name,
            "status": "kept",
            "done": True,
            "latest_round_idx": round_idx,
            "decision": decision,
        }

    if round_idx >= max_rounds:
        return {
            "pair": pair_dir.name,
            "status": "max_rounds_reached",
            "done": True,
            "latest_round_idx": round_idx,
            "decision": decision,
        }

    next_round_idx = round_idx + 1
    rc = run_one_round(
        pair_dir=pair_dir,
        obj_id=obj_id,
        rotation_deg=rotation_deg,
        next_round_idx=next_round_idx,
        actions=decision["chosen_actions"],
        python_bin=python_bin,
        blender_bin=blender_bin,
        meshes_dir=meshes_dir,
        profile=profile,
        scene_template=scene_template,
    )

    return {
        "pair": pair_dir.name,
        "status": "launched_next_round" if rc == 0 else "launch_failed",
        "done": False,
        "latest_round_idx": round_idx,
        "next_round_idx": next_round_idx,
        "decision": decision,
        "return_code": rc,
    } if rc == 0 else _handle_failed_launch(
        pair_dir=pair_dir,
        obj_id=obj_id,
        round_idx=next_round_idx,
        decision=decision,
    )


def run_worker(args) -> int:
    shard_dir = Path(args.worker_shard_dir).resolve()
    log_path = Path(args.worker_log_path).resolve()
    pair_dirs = sorted([p for p in shard_dir.iterdir() if p.is_dir() and p.name.startswith("obj_")])
    visible_gpu_ids = parse_visible_gpu_ids(args.worker_gpu)
    if not visible_gpu_ids:
        raise ValueError("worker_gpu must specify at least one visible GPU id")

    summary = {
        "gpu": args.worker_gpu,
        "review_visible_gpus": visible_gpu_ids,
        "review_mode": "balanced" if len(visible_gpu_ids) > 1 else "single_gpu",
        "shard_dir": str(shard_dir),
        "started_at": time.time(),
        "passes": [],
        "success": True,
        "status": "running",
        "current_pair": None,
        "failed_pair": None,
        "message": None,
        "last_round_launched": None,
    }
    if len(visible_gpu_ids) < 2 and not args.allow_unsafe_single_gpu_review:
        summary["success"] = False
        summary["status"] = "invalid_reviewer_gpu_config"
        summary["message"] = (
            "single-GPU VLM review is disabled; pass multi-GPU groups like --gpus 0,1;2,3 "
            "or explicitly opt into --allow-unsafe-single-gpu-review"
        )
        summary["ended_at"] = time.time()
        save_json(log_path, summary)
        return 2

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpu_ids)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if len(visible_gpu_ids) > 1:
        os.environ["VLM_ALLOW_BALANCED"] = "1"
        os.environ.pop("VLM_FORCE_SINGLE_GPU", None)
    else:
        os.environ.setdefault("VLM_FORCE_SINGLE_GPU", "1")

    save_json(log_path, summary)

    while True:
        pair_dirs = sorted(
            [p for p in shard_dir.iterdir() if p.is_dir() and p.name.startswith("obj_")],
            key=lambda p: (get_pair_max_round(p), p.name),
        )
        pass_result = {
            "started_at": time.time(),
            "pairs": [],
        }
        all_keep = True
        for pair_dir in pair_dirs:
            summary["current_pair"] = pair_dir.name
            summary["status"] = "running"
            summary["message"] = "processing_pair"
            save_json(log_path, summary)
            result = process_pair(
                pair_dir=pair_dir,
                python_bin=args.python_bin,
                blender_bin=args.blender_bin,
                meshes_dir=args.meshes_dir,
                profile=args.profile,
                scene_template=args.scene_template,
                max_rounds=args.max_rounds,
                disable_asset_deprecation=args.disable_asset_deprecation,
                ignore_registry_blocks=args.ignore_registry_blocks,
            )
            pass_result["pairs"].append(result)
            summary["message"] = result.get("status")
            if result.get("status") == "launched_next_round":
                summary["last_round_launched"] = result.get("next_round_idx")
            if not result.get("done"):
                all_keep = False
            if result.get("status") == "launch_failed":
                summary["status"] = "launch_failed_retrying"
                summary["failed_pair"] = result["pair"]
                summary["message"] = "launch_failed_retrying"
                save_json(log_path, summary)
        pass_result["ended_at"] = time.time()
        summary["passes"].append(pass_result)
        summary["last_pass_at"] = pass_result["ended_at"]
        save_json(log_path, summary)
        if all_keep:
            summary["status"] = "completed"
            summary["message"] = "all_pairs_kept"
            summary["current_pair"] = None
            summary["ended_at"] = time.time()
            save_json(log_path, summary)
            return 0
        time.sleep(args.pass_sleep_seconds)


def run_orchestrator(args) -> int:
    root_dir = Path(args.root_dir).resolve()
    shard_root = root_dir / "_shards"
    log_root = root_dir / "_monitor_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    gpus = parse_gpus(args.gpus)
    gpu_groups = [parse_visible_gpu_ids(gpu) for gpu in gpus]
    if any(not group for group in gpu_groups):
        raise ValueError(f"Invalid --gpus specification: {args.gpus}")
    if not args.allow_unsafe_single_gpu_review and any(len(group) < 2 for group in gpu_groups):
        raise ValueError(
            "Single-GPU reviewer workers are disabled. Use grouped GPUs such as --gpus 0,1;2,3 "
            "or explicitly opt into --allow-unsafe-single-gpu-review."
        )
    seen_gpu_ids = set()
    for group in gpu_groups:
        overlap = seen_gpu_ids.intersection(group)
        if overlap:
            overlap_text = ",".join(sorted(overlap))
            raise ValueError(f"Overlapping reviewer GPU groups are not allowed: {overlap_text}")
        seen_gpu_ids.update(group)

    launched = []
    for gpu in gpus:
        gpu_slug = slugify_gpu(gpu)
        shard_dir = prepare_worker_shard_dir(shard_root, log_root, gpu_slug)
        if shard_dir is None:
            continue
        log_path = log_root / f"monitor_gpu_{gpu_slug}.json"
        cmd = [
            args.python_bin,
            str(SCRIPT_PATH),
            "--worker-mode",
            "--root-dir",
            str(root_dir),
            "--gpus",
            gpu,
            "--python",
            args.python_bin,
            "--blender",
            args.blender_bin,
            "--meshes-dir",
            args.meshes_dir,
            "--profile",
            args.profile,
            "--scene-template",
            args.scene_template,
            "--max-rounds",
            str(args.max_rounds),
            "--pass-sleep-seconds",
            str(args.pass_sleep_seconds),
            "--worker-gpu",
            gpu,
            "--worker-shard-dir",
            str(shard_dir),
            "--worker-log-path",
            str(log_path),
        ]
        if args.allow_unsafe_single_gpu_review:
            cmd.append("--allow-unsafe-single-gpu-review")
        if args.disable_asset_deprecation:
            cmd.append("--disable-asset-deprecation")
        if args.ignore_registry_blocks:
            cmd.append("--ignore-registry-blocks")
        process = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
        launched.append({"gpu": gpu, "process": process, "log_path": log_path})
        if args.launch_stagger_seconds > 0:
            time.sleep(args.launch_stagger_seconds)

    overall_success = True
    worker_logs = {}
    for item in launched:
        rc = item["process"].wait()
        log_json = load_json(Path(item["log_path"]), default={}) or {}
        worker_logs[f"gpu_{slugify_gpu(item['gpu'])}"] = log_json
        if rc != 0 or not log_json.get("success", False):
            overall_success = False

    summary = {
        "success": overall_success,
        "root_dir": str(root_dir),
        "worker_logs": worker_logs,
    }
    save_json(log_root / "monitor_summary.json", summary)
    return 0 if overall_success else 1


def main():
    args = parse_args()
    requested_max_rounds = args.max_rounds
    args.max_rounds = min(args.max_rounds, HARD_MAX_ROUNDS)
    if requested_max_rounds != args.max_rounds:
        print(
            f"[scene-agent-monitor] Requested --max-rounds={requested_max_rounds} exceeds hard cap "
            f"{HARD_MAX_ROUNDS}; clamping.",
            file=sys.stderr,
        )
    if args.worker_mode:
        raise SystemExit(run_worker(args))
    raise SystemExit(run_orchestrator(args))


if __name__ == "__main__":
    main()
