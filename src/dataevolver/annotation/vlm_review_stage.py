"""
Stage 5.5: VLM-based render review + CV metrics.

Reviews rendered images for one object using Qwen3.5-35B-A3B and
traditional CV metrics. Outputs a JSON review with hybrid quality score.

v6: Added structure_consistency, color_consistency, physics_consistency
    diagnostics with reference image comparison support.

Usage:
  python -m dataevolver.annotation.vlm_review_stage \
    --renders-dir .dataevolver/runtime/renders \
    --obj-id obj_001 \
    --output-dir .dataevolver/runtime/vlm_reviews \
    [--prev-rgb path/to/prev.png] \
    [--round-idx 0] \
    [--active-group lighting] \
    [--device cuda:0]
"""

import os
import sys
import json
import base64
import math
import re
import argparse
import copy
import traceback
from collections import Counter as _Counter
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from dataevolver.paths import CONFIGS_ROOT, DATAEVOLVER_ROOT, REPO_ROOT
from dataevolver.runtime.asset_lifecycle import (
    ABANDON_CONFIDENCE_VALUES,
    ASSET_VIABILITY_VALUES,
    detect_asset_viability,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_MODEL_ROOT = Path(
    os.environ.get("DATAEVOLVER_MODEL_ROOT", os.fspath(DATAEVOLVER_ROOT / "local" / "model_hub"))
)
VLM_MODEL_PATH = os.environ.get("VLM_MODEL_PATH", os.fspath(LOCAL_MODEL_ROOT / "Qwen3.5-35B-A3B"))
VLM_MODEL_NAME = "Qwen3.5-35B-A3B"
DATA_BUILD_ROOT = os.fspath(REPO_ROOT)
ACTION_SPACE_PATH = os.fspath(CONFIGS_ROOT / "action_space.json")
SCENE_ACTION_SPACE_PATH = os.fspath(CONFIGS_ROOT / "scene_action_space.json")

# Representative canonical views: (azimuth, elevation)
REP_VIEWS = [(0, 0), (90, 0), (180, 0), (270, 0)]
SCENE_REP_VIEWS = [(0, 0), (90, 0), (180, 15), (45, 30)]

ISSUE_TAGS = [
    "none", "underexposed", "overexposed", "flat_lighting", "harsh_shadow",
    "weak_subject_separation", "background_too_bright", "background_too_dark",
    "object_too_small", "object_too_large", "off_center_left", "off_center_right",
    "off_center_up", "off_center_down", "object_cutoff_left", "object_cutoff_right",
    "object_cutoff_top", "object_cutoff_bottom", "floating_object",
    "ground_intersection", "geometry_distortion", "partial_occlusion",
    "mask_boundary_error", "mask_hole", "mask_spill", "background_distracting",
    "depth_of_field_too_strong", "depth_of_field_too_weak",
    "mesh_interpenetration", "color_shift", "physical_implausibility",
    "scene_light_mismatch", "shadow_missing", "scale_implausible",
    "floating_visible", "ground_intersection_visible",
]

LIGHTING_DIAGNOSIS_VALUES = {
    "flat_no_rim", "flat_low_contrast", "underexposed_global",
    "underexposed_shadow", "harsh_shadow_key",
    "scene_light_mismatch", "shadow_missing", "good"
}

# v6: New diagnostic field valid values
STRUCTURE_VALUES = {"good", "minor_mismatch", "major_mismatch"}
COLOR_VALUES = {"good", "minor_shift", "major_shift"}
PHYSICS_VALUES = {"good", "minor_issue", "major_issue"}

GROUPS = ["lighting", "camera", "object", "scene", "material"]

# Hybrid score weights
VLM_W = {"lighting": 0.25, "object_integrity": 0.30, "composition": 0.20,
          "render_quality_semantic": 0.10, "overall": 0.15}
# Full 4-metric weights (used when mask is available)
CV_W  = {"mask_score": 0.30, "exposure_score": 0.25,
          "sharpness_score": 0.20, "framing_score": 0.25}
# 2-metric weights when no mask (exposure 0.556, sharpness 0.444)
CV_W_NOMASK = {"exposure_score": 0.556, "sharpness_score": 0.444}
CV_W_SCENE = {"exposure_score": 0.45, "sharpness_score": 0.35, "framing_score": 0.20}


def _load_action_space_data(action_space_path: Optional[str] = None) -> dict:
    path = action_space_path or ACTION_SPACE_PATH
    with open(path) as f:
        return json.load(f)


def _resolve_rep_views(review_mode: str = "studio", rep_views=None):
    if rep_views:
        return rep_views
    return list(SCENE_REP_VIEWS if review_mode == "scene_insert" else REP_VIEWS)

# ─────────────────────────────────────────────────────────────────────────────
# CV Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_mask_score(mask_path: str) -> float:
    """Boundary F1 (3px band) + silhouette sanity (holes/fragments)."""
    try:
        import cv2
        mask = np.array(Image.open(mask_path).convert("L"))
        binary = (mask > 127).astype(np.uint8)
        if binary.sum() == 0:
            return 0.0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        boundary_band = cv2.dilate(binary, kernel) - cv2.erode(binary, kernel)
        edges = cv2.Canny(binary * 255, 50, 150)
        edges_b = (edges > 0).astype(np.uint8)

        tp = float(np.sum(edges_b & (boundary_band > 0)))
        fp = float(np.sum(edges_b & (boundary_band == 0)))
        fn = float(np.sum((edges_b == 0) & (boundary_band > 0)))
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)

        # Fragment penalty
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        total_fg = int(binary.sum())
        if n_labels > 1:
            main_area = int(stats[1:, cv2.CC_STAT_AREA].max())
            frag_penalty = 1.0 - max(0.0, (total_fg - main_area) / (total_fg + 1e-8))
        else:
            frag_penalty = 1.0

        # Hole penalty (large holes inside the object)
        inv = 1 - binary
        n_holes, _, h_stats, _ = cv2.connectedComponentsWithStats(inv)
        large_holes = sum(1 for a in (h_stats[1:, cv2.CC_STAT_AREA] if n_holes > 1 else []) if a > 200)
        hole_penalty = max(0.0, 1.0 - large_holes * 0.1)

        sanity = (frag_penalty + hole_penalty) / 2.0
        return float(np.clip(0.7 * f1 + 0.3 * sanity, 0.0, 1.0))
    except Exception as e:
        print(f"  [mask_score] fallback (0.5): {e}")
        return 0.5


def compute_exposure_score(rgb_path: str, mask_path: Optional[str] = None) -> float:
    """Luminance clipping + median deviation for foreground pixels."""
    try:
        img = np.array(Image.open(rgb_path).convert("RGB")).astype(float) / 255.0
        if mask_path and os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
            fg = mask > 127
        else:
            h, w = img.shape[:2]
            fg = np.zeros((h, w), bool)
            fg[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True

        if fg.sum() < 100:
            return 0.5

        lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        fg_lum = lum[fg]

        bright_clip = float((fg_lum > 0.95).mean())
        dark_clip   = float((fg_lum < 0.05).mean())
        median_lum  = float(np.median(fg_lum))

        clip_pen   = min(1.0, (bright_clip + dark_clip) * 2.0)
        median_pen = min(1.0, abs(median_lum - 0.45) * 1.5)
        return float(np.clip(1.0 - clip_pen * 0.6 - median_pen * 0.4, 0.0, 1.0))
    except Exception as e:
        print(f"  [exposure_score] fallback (0.5): {e}")
        return 0.5


def compute_sharpness_score(rgb_path: str, history_file: Optional[str] = None) -> float:
    """Laplacian variance, log-mapped with bootstrap percentile calibration.
    Falls back to fixed range lo=40, hi=180 (CG render typical) until 10+ samples.
    """
    try:
        import cv2
        img = np.array(Image.open(rgb_path).convert("L"))
        lap_var = float(cv2.Laplacian(img.astype(np.float32), cv2.CV_32F).var())

        history = []
        if history_file and os.path.exists(history_file):
            with open(history_file) as f:
                history = json.load(f)
        history.append(lap_var)
        if len(history) > 500:
            history = history[-500:]
        if history_file:
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            with open(history_file, "w") as f:
                json.dump(history, f)

        if len(history) >= 10:
            lo = float(np.percentile(history, 10))
            hi = float(np.percentile(history, 90))
        else:
            lo, hi = 40.0, 180.0  # CG render typical range

        score = (math.log1p(lap_var) - math.log1p(lo)) / (math.log1p(hi) - math.log1p(lo) + 1e-6)
        return float(np.clip(score, 0.0, 1.0))
    except Exception as e:
        print(f"  [sharpness_score] fallback (0.5): {e}")
        return 0.5


def compute_framing_score(rgb_path: str, mask_path: Optional[str] = None,
                          weak_centering: bool = False) -> float:
    """Frame quality with a weaker center prior for scene-mode renders."""
    try:
        import cv2
        img = np.array(Image.open(rgb_path).convert("RGB"))
        h, w = img.shape[:2]

        if mask_path and os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
            binary = (mask > 127).astype(np.uint8)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 30, 1, cv2.THRESH_BINARY)

        if binary.sum() == 0:
            return 0.5

        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        row_idx = np.where(rows)[0]
        col_idx = np.where(cols)[0]
        rmin, rmax = int(row_idx[0]), int(row_idx[-1])
        cmin, cmax = int(col_idx[0]), int(col_idx[-1])

        obj_cx = (cmin + cmax) / 2.0
        obj_cy = (rmin + rmax) / 2.0
        d = math.hypot(obj_cx - w / 2.0, obj_cy - h / 2.0)
        if weak_centering:
            center_score = 1.0 - d / (math.hypot(w, h) + 1e-8)
        else:
            center_score = 1.0 - d / (math.hypot(w, h) / 2.0 + 1e-8)

        area_frac = float(binary.sum()) / (h * w)
        if weak_centering:
            if area_frac < 0.08:
                size_score = area_frac / 0.08
            elif area_frac > 0.80:
                size_score = 1.0 - (area_frac - 0.80) / 0.20
            else:
                size_score = 1.0
        else:
            if area_frac < 0.15:
                size_score = area_frac / 0.15
            elif area_frac > 0.65:
                size_score = 1.0 - (area_frac - 0.65) / 0.35
            else:
                size_score = 1.0

        border = 5
        cutoff = (0.25 * int(rmin < border) + 0.25 * int(rmax > h - border) +
                  0.25 * int(cmin < border) + 0.25 * int(cmax > w - border))

        if weak_centering:
            score = 0.20 * center_score + 0.45 * max(0.0, size_score) + 0.35 * (1.0 - cutoff)
        else:
            score = 0.45 * center_score + 0.35 * max(0.0, size_score) + 0.20 * (1.0 - cutoff)
        return float(np.clip(score, 0.0, 1.0))
    except Exception as e:
        print(f"  [framing_score] fallback (0.5): {e}")
        return 0.5


def compute_cv_metrics(rgb_path: str, mask_path: Optional[str],
                       history_file: Optional[str] = None,
                       review_mode: str = "studio") -> dict:
    """Compute CV quality metrics. When mask is unavailable, only use exposure+sharpness."""
    mask_available = bool(mask_path and os.path.exists(mask_path))

    es = compute_exposure_score(rgb_path, mask_path if mask_available else None)
    ss = compute_sharpness_score(rgb_path, history_file)

    if review_mode == "scene_insert":
        # Scene mode is currently judged primarily by the final RGB result.
        # The Blender mask path is too fragile to be trusted as a hard signal,
        # so we deliberately ignore it here and use only weak RGB framing.
        mask_available = False
        ms = None
        fs = compute_framing_score(
            rgb_path,
            None,
            weak_centering=True,
        )
        cv = (CV_W_SCENE["exposure_score"] * es
              + CV_W_SCENE["sharpness_score"] * ss
              + CV_W_SCENE["framing_score"] * fs)
        return {
            "mask_score":      round(ms, 4) if ms is not None else None,
            "exposure_score":  round(es, 4),
            "sharpness_score": round(ss, 4),
            "framing_score":   round(fs, 4),
            "cv_score":        round(cv, 4),
            "mask_available":  mask_available,
        }

    if mask_available:
        ms = compute_mask_score(mask_path)
        fs = compute_framing_score(rgb_path, mask_path)
        cv = (CV_W["mask_score"] * ms + CV_W["exposure_score"] * es
              + CV_W["sharpness_score"] * ss + CV_W["framing_score"] * fs)
        return {
            "mask_score":      round(ms, 4),
            "exposure_score":  round(es, 4),
            "sharpness_score": round(ss, 4),
            "framing_score":   round(fs, 4),
            "cv_score":        round(cv, 4),
            "mask_available":  True,
        }
    else:
        # No mask: exposure(0.556) + sharpness(0.444), framing/mask unreliable
        cv = CV_W_NOMASK["exposure_score"] * es + CV_W_NOMASK["sharpness_score"] * ss
        return {
            "mask_score":      None,
            "exposure_score":  round(es, 4),
            "sharpness_score": round(ss, 4),
            "framing_score":   None,
            "cv_score":        round(cv, 4),
            "mask_available":  False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_vlm_score(vlm_scores: dict) -> float:
    return float(sum(VLM_W[k] * (vlm_scores.get(k, 3) - 1) / 4.0 for k in VLM_W))


def compute_hybrid_score(vlm_scores: dict, cv_metrics: dict,
                         review_mode: str = "studio") -> tuple:
    """Returns (hybrid_score, route)."""
    s_vlm = compute_vlm_score(vlm_scores)
    s_cv  = cv_metrics["cv_score"]
    hybrid = 0.70 * s_vlm + 0.30 * s_cv if review_mode == "scene_insert" else 0.50 * s_vlm + 0.50 * s_cv

    # Hard gates
    if vlm_scores.get("object_integrity", 3) <= 2:
        hybrid = min(hybrid, 0.69)
    if review_mode != "scene_insert":
        if cv_metrics.get("mask_available") and cv_metrics.get("mask_score") is not None:
            if cv_metrics["mask_score"] < 0.60:
                hybrid = min(hybrid, 0.74)
        if cv_metrics["exposure_score"] < 0.55:
            hybrid = min(hybrid, 0.74)
    else:
        if cv_metrics["exposure_score"] < 0.45:
            hybrid = min(hybrid, 0.74)

    if hybrid >= 0.80:
        route = "pass"
    elif hybrid >= 0.55:
        route = "needs_fix"
    else:
        route = "reject"

    return round(hybrid, 4), route


# ─────────────────────────────────────────────────────────────────────────────
# v6: Reference image resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_reference_image(obj_id, profile_cfg, renders_dir):
    """Resolve reference image path, return absolute path or None."""
    ref_dir = profile_cfg.get("reference_images_dir", None) if profile_cfg else None

    if ref_dir is None:
        # Default: infer from renders_dir sibling directory
        # renders_dir = .dataevolver/runtime/renders
        # -> ref_dir = .dataevolver/runtime/images
        ref_dir = os.path.join(os.path.dirname(renders_dir), "images")

    if not os.path.isabs(ref_dir):
        # Relative path: resolve relative to DATA_BUILD_ROOT
        ref_dir = os.path.join(DATA_BUILD_ROOT, ref_dir)

    ref_path = os.path.join(ref_dir, f"{obj_id}.png")
    return ref_path if os.path.exists(ref_path) else None


def _resolve_pseudo_reference_image(obj_id, profile_cfg, renders_dir):
    """Resolve pseudo-reference image path, return absolute path or None."""
    if profile_cfg and not profile_cfg.get("use_pseudo_reference", False):
        return None

    ref_dir = profile_cfg.get("pseudo_reference_dir", None) if profile_cfg else None
    if ref_dir is None:
        ref_dir = os.path.join(os.path.dirname(renders_dir), "pseudo_references")

    if not os.path.isabs(ref_dir):
        ref_dir = os.path.join(DATA_BUILD_ROOT, ref_dir)

    candidates = [
        os.path.join(ref_dir, obj_id, "pseudo_reference.png"),
        os.path.join(ref_dir, f"{obj_id}.png"),
    ]
    for ref_path in candidates:
        if os.path.exists(ref_path):
            return ref_path
    return None


# ─────────────────────────────────────────────────────────────────────────────
# VLM inference
# ─────────────────────────────────────────────────────────────────────────────

_vlm_model = None
_vlm_processor = None


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _resolve_vlm_load_kwargs(device: str) -> dict:
    import torch

    force_single = os.environ.get("VLM_FORCE_SINGLE_GPU", "0").strip().lower() in {"1", "true", "yes", "on"}
    allow_balanced = os.environ.get("VLM_ALLOW_BALANCED", "0").strip().lower() in {"1", "true", "yes", "on"}
    visible_devices_env = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    num_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if force_single:
        return {
            "device_map": device,
            "low_cpu_mem_usage": True,
        }
    if device.startswith("cuda:") and not allow_balanced:
        return {
            "device_map": device,
            "low_cpu_mem_usage": True,
        }
    if len(visible_devices_env) == 1:
        return {
            "device_map": device,
            "low_cpu_mem_usage": True,
        }
    if num_visible > 1:
        max_mem_gib = os.environ.get("VLM_MAX_MEMORY_GIB", "75")
        max_memory = {idx: f"{max_mem_gib}GiB" for idx in range(num_visible)}
        return {
            "device_map": "balanced",
            "max_memory": max_memory,
            "low_cpu_mem_usage": True,
        }
    return {
        "device_map": device,
        "low_cpu_mem_usage": True,
    }


def load_vlm(device: str = "cuda:0"):
    global _vlm_model, _vlm_processor
    if _vlm_model is not None:
        return _vlm_model, _vlm_processor

    print(f"[VLM] Loading {VLM_MODEL_NAME} from {VLM_MODEL_PATH} -> {device}")
    from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
    import transformers.modeling_utils as _mu
    import torch
    import gc

    # Free residual GPU memory from prior steps (Blender CYCLES, 3D recon, etc.)
    gc.collect()
    torch.cuda.empty_cache()

    # Disable caching_allocator_warmup: it estimates pre-allocation by total
    # param count (~65GB for 35B MoE) instead of active params (~7GB), causing OOM.
    _mu.caching_allocator_warmup = lambda *a, **kw: None

    _vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_PATH, trust_remote_code=True)
    load_kwargs = _resolve_vlm_load_kwargs(device)
    print(f"[VLM] load kwargs: {load_kwargs}")
    _vlm_model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        VLM_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        **load_kwargs,
    )
    _vlm_model.eval()
    print("[VLM] Model loaded.")
    return _vlm_model, _vlm_processor


def _img_to_b64(path: str, max_size: int = 512) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _allowed_actions_for_group(active_group: str, action_space_path: Optional[str]) -> list:
    aspace = _load_action_space_data(action_space_path)
    allowed = list(aspace.get("groups", {}).get(active_group, {}).get("actions", {}).keys())
    for cname, cdef in aspace.get("compound_actions", {}).items():
        if cdef.get("group") == active_group:
            allowed.append(cname)
    return ["NO_OP"] + allowed if allowed else ["NO_OP"]


def _allowed_actions_all(action_space_path: Optional[str]) -> list:
    aspace = _load_action_space_data(action_space_path)
    allowed: List[str] = ["NO_OP"]
    for _, gdef in aspace.get("groups", {}).items():
        allowed.extend(gdef.get("actions", {}).keys())
    allowed.extend(aspace.get("compound_actions", {}).keys())
    return _dedupe_keep_order(allowed)


def _build_common_prompt(sample_id: str, round_idx: int, active_group: str,
                         has_prev: bool, az: int, el: int,
                         prompt_appendix: str = "",
                         issue_tags_whitelist=None,
                         has_reference: bool = False,
                         has_pseudo_reference: bool = False,
                         action_space_path: Optional[str] = None,
                         review_mode: str = "studio") -> tuple:
    effective_tags = issue_tags_whitelist if issue_tags_whitelist else ISSUE_TAGS
    allowed_issues = ", ".join(effective_tags)
    allowed_actions_str = ", ".join(_allowed_actions_for_group(active_group, action_space_path))

    system_msg = (
        "You are a practical render reviewer helping improve iterative 3D scene insertion results. "
        "Your primary goal is to judge whether the current image looks convincing and usable to a human viewer. "
        "Use the numeric scores only as coarse bookkeeping, not as the main objective. "
        "Return ONLY a valid JSON object, no other text, no markdown fences. "
        "Do not explain your reasoning. The first character of your reply must be '{' and the last character must be '}'. "
        "Fill every field using only the allowed enums or numeric ranges."
    )

    image_desc_parts = [f"Image 1 is the CURRENT render (view az={az}, el={el})."]
    next_idx = 2
    if has_reference:
        image_desc_parts.append(
            f"Image {next_idx} is the ORIGINAL REFERENCE image (object identity, texture, color reference)."
        )
        next_idx += 1
    if has_pseudo_reference:
        image_desc_parts.append(
            f"Image {next_idx} is the PSEUDO-REFERENCE image (rough target scene integration example, not strict ground truth)."
        )
        next_idx += 1
    if has_prev:
        image_desc_parts.append(f"Image {next_idx} is the PREVIOUS render (round {round_idx-1}).")
    elif not has_reference and not has_pseudo_reference:
        image_desc_parts.append("Only one image is provided (no previous render or reference for comparison).")
    pairwise_instr = " ".join(image_desc_parts)

    if review_mode == "scene_insert":
        decision_priority = (
            "Decision priority for scene insertion:\n"
            "  1. Judge the CURRENT render primarily by final visual effect: does it look like a believable, keepable image to a human reviewer?\n"
            "  2. Prefer overall realism and natural integration over metric-like precision.\n"
            "  3. Do NOT over-penalize small imperfections if the image already looks good enough overall.\n"
            "  4. Reserve very low scores for obvious failures that materially hurt the final image.\n"
            "  5. If the current image looks clearly better than the previous one to a human, reflect that in pairwise_vs_prev even if the improvement is modest."
        )
        score_guide = (
            "Scoring guide (1=very bad, 2=poor, 3=acceptable, 4=good, 5=excellent):\n"
            "  lighting: whether object lighting matches scene illumination, direction and shadow behavior\n"
            "    If lighting < 4, also report lighting_diagnosis:\n"
            "      scene_light_mismatch: object shadow direction is clearly OPPOSITE to scene shadows, OR object has visibly incompatible time-of-day lighting (e.g., bright-daylit object in a nighttime scene). Minor tonal differences or soft shadow variations do NOT qualify.\n"
            "      shadow_missing: object lacks convincing grounding/contact shadow\n"
            "      underexposed_global: object is globally too dark\n"
            "      flat_low_contrast: object appears washed out with weak tonal separation\n"
            "      harsh_shadow_key: key light shadow is harsh or distracting\n"
            "      good: lighting >= 4\n"
            "  object_integrity: object geometry remains coherent and matches the same object identity\n"
            "  composition: object size/framing is usable in the scene; penalize only clearly bad scale, cutoff, or placement that a human would immediately notice\n"
            "  render_quality_semantic: texture/material quality and overall scene integration\n"
            "  overall: human-judged final image quality; would you keep this image as a good final result?"
        )
        ref_diag_guide = (
            "\n\nAdditional diagnostics:\n"
            "  structure_consistency: compare ONLY the object's 3D shape and geometry with the reference, ignoring background, lighting, scale, or viewing angle differences. Report major_mismatch ONLY if the object appears to be a fundamentally different item type (e.g., a chair rendered as a table). Minor rendering or perspective differences on the same object type are 'good'.\n"
            "  color_consistency: compare texture/color palette with the reference image\n"
            "  physics_consistency: ONLY flag if the object is VISIBLY hovering in mid-air above the ground, or VISIBLY clipping/embedded into the ground surface. Do NOT penalize for shadow quality, lighting inconsistency, soft contact appearance, or mild perspective ambiguity — those belong to lighting_diagnosis or composition. Be very conservative; programmatic checks already handle precise measurements."
        )
        if has_pseudo_reference:
            ref_diag_guide += (
                "\n"
                "  pseudo-reference usage: if a PSEUDO-REFERENCE image is provided, use it ONLY as a rough target for scene feel, plausible placement, overall composition, grounding feel, and lighting compatibility. "
                "Do NOT treat the pseudo-reference as exact ground truth for absolute scale, exact pixel position, exact framing, or exact shadow shape. "
                "Do NOT use the pseudo-reference to downgrade structure_consistency or color_consistency. Structure and color must be judged primarily against the ORIGINAL REFERENCE image. "
                "If the CURRENT render looks more natural or more believable than the pseudo-reference in some local detail, prefer the CURRENT render."
            )
        json_template = (
            "{"
            "\"schema_version\": \"vlm_review_v1\","
            "\"sample_id\": \"" + sample_id + "\","
            "\"round_idx\": " + str(round_idx) + ","
            "\"vlm_route\": \"<pass|needs_fix|reject>\","
            "\"scores\": {\"lighting\": <1-5>, \"object_integrity\": <1-5>, \"composition\": <1-5>, \"render_quality_semantic\": <1-5>, \"overall\": <1-5>},"
            "\"confidence\": {\"lighting\": \"<low|medium|high>\", \"object_integrity\": \"<low|medium|high>\", \"composition\": \"<low|medium|high>\", \"render_quality_semantic\": \"<low|medium|high>\", \"overall\": \"<low|medium|high>\"},"
            "\"issue_tags\": [\"<scene_light_mismatch|shadow_missing|scale_implausible|floating_visible|ground_intersection_visible|color_shift|none>\"],"
            "\"suggested_actions\": [\"<action1>\"],"
            "\"lighting_diagnosis\": \"<scene_light_mismatch|shadow_missing|underexposed_global|flat_low_contrast|harsh_shadow_key|good>\","
            "\"structure_consistency\": \"<good|minor_mismatch|major_mismatch>\","
            "\"color_consistency\": \"<good|minor_shift|major_shift>\","
            "\"physics_consistency\": \"<good|minor_issue|major_issue>\","
            "\"asset_viability\": \"<continue|abandon|unclear>\","
            "\"abandon_reason\": \"<short reason or null>\","
            "\"abandon_confidence\": \"<low|medium|high|null>\","
            "\"pairwise_vs_prev\": {\"available\": " + ("true" if has_prev else "false") + ", \"winner\": \"<current|previous|tie|none>\", \"lighting\": \"<better|same|worse|na>\", \"object_integrity\": \"<better|same|worse|na>\", \"composition\": \"<better|same|worse|na>\", \"render_quality_semantic\": \"<better|same|worse|na>\"}"
            "}"
        )
    else:
        score_guide = (
            "Scoring guide (1=very bad, 2=poor, 3=acceptable, 4=good, 5=excellent):\n"
            "  lighting: overall illumination quality, no harsh shadows, good contrast\n"
            "    If lighting < 4, also report lighting_diagnosis:\n"
            "      flat_no_rim: no rim/edge separation from background\n"
            "      flat_low_contrast: low tonal range, washed out\n"
            "      underexposed_global: entire image too dark\n"
            "      underexposed_shadow: shadow areas too deep/crushed\n"
            "      harsh_shadow_key: hard, distracting key light shadows\n"
            "      good: lighting >= 4\n"
            "  object_integrity: object is complete, no clipping, correct geometry\n"
            "  composition: object centering, appropriate scale, not cut off\n"
            "  render_quality_semantic: no artifacts, blurriness, or texture issues\n"
            "  overall: holistic dataset usability score"
        )
        if has_reference:
            ref_diag_guide = (
                "\n\nAdditional diagnostics (compare current render with the REFERENCE image):\n"
                "  structure_consistency: Does the rendered 3D object structure match the reference? "
                "(good=match, minor_mismatch=small differences, major_mismatch=significant structural deviation)\n"
                "  color_consistency: Is the color palette (hue, saturation, brightness) consistent with reference? "
                "(good=match, minor_shift=slight color difference, major_shift=significant color deviation)\n"
                "  physics_consistency: Is the object physically plausible (no floating, no ground intersection, "
                "correct shadow/contact)? (good=plausible, minor_issue=slight implausibility, major_issue=severe problem)"
            )
        else:
            ref_diag_guide = (
                "\n\nAdditional diagnostics (assess based on the render alone):\n"
                "  structure_consistency: Is the 3D object structurally coherent? "
                "(good=coherent, minor_mismatch=small issues, major_mismatch=significant problems)\n"
                "  color_consistency: Are colors natural and balanced? "
                "(good=balanced, minor_shift=slight issue, major_shift=significant issue)\n"
                "  physics_consistency: Is the object physically plausible? "
                "(good=plausible, minor_issue=slight issue, major_issue=severe problem)"
            )
        json_template = (
            "{"
            "\"schema_version\": \"vlm_review_v1\","
            "\"sample_id\": \"" + sample_id + "\","
            "\"round_idx\": " + str(round_idx) + ","
            "\"vlm_route\": \"<pass|needs_fix|reject>\","
            "\"scores\": {\"lighting\": <1-5>, \"object_integrity\": <1-5>, \"composition\": <1-5>, \"render_quality_semantic\": <1-5>, \"overall\": <1-5>},"
            "\"confidence\": {\"lighting\": \"<low|medium|high>\", \"object_integrity\": \"<low|medium|high>\", \"composition\": \"<low|medium|high>\", \"render_quality_semantic\": \"<low|medium|high>\", \"overall\": \"<low|medium|high>\"},"
            "\"issue_tags\": [\"<tag1>\"],"
            "\"suggested_actions\": [\"<action1>\"],"
            "\"lighting_diagnosis\": \"<flat_no_rim|flat_low_contrast|underexposed_global|underexposed_shadow|harsh_shadow_key|good>\","
            "\"structure_consistency\": \"<good|minor_mismatch|major_mismatch>\","
            "\"color_consistency\": \"<good|minor_shift|major_shift>\","
            "\"physics_consistency\": \"<good|minor_issue|major_issue>\","
            "\"asset_viability\": \"<continue|abandon|unclear>\","
            "\"abandon_reason\": \"<short reason or null>\","
            "\"abandon_confidence\": \"<low|medium|high|null>\","
            "\"pairwise_vs_prev\": {\"available\": " + ("true" if has_prev else "false") + ", \"winner\": \"<current|previous|tie|none>\", \"lighting\": \"<better|same|worse|na>\", \"object_integrity\": \"<better|same|worse|na>\", \"composition\": \"<better|same|worse|na>\", \"render_quality_semantic\": \"<better|same|worse|na>\"}"
            "}"
        )

    scene_priority_block = f"{decision_priority}\n\n" if review_mode == "scene_insert" else ""
    user_msg = (
        f"Analyze this rendered 3D object image.\n"
        f"sample_id={sample_id}, round={round_idx}, view=az{az:03d}_el{el:+03d}, "
        f"active_fix_group={active_group}\n"
        f"{pairwise_instr}\n\n"
        f"{scene_priority_block}"
        f"{score_guide}"
        f"{ref_diag_guide}\n\n"
        f"Allowed issue_tags (choose up to 3 that clearly and materially hurt the final image): {allowed_issues}\n"
        f"Allowed suggested_actions for group '{active_group}' (choose up to 2): {allowed_actions_str}\n\n"
        "Output rules:\n"
        "  1. Output exactly one JSON object and nothing else.\n"
        "  2. Do not include analysis, prose, bullet points, or markdown fences.\n"
        "  3. Replace every placeholder with a concrete value.\n"
        "  4. If uncertain, choose the closest allowed enum and keep the JSON valid.\n\n"
        f"Output ONLY this JSON object (replace <...> with your values):\n"
        f"{json_template}"
    )
    if prompt_appendix:
        user_msg = user_msg + "\n\n" + prompt_appendix
    return system_msg, user_msg


def _build_prompt(sample_id: str, round_idx: int, active_group: str,
                  has_prev: bool, az: int, el: int,
                  prompt_appendix: str = "",
                  issue_tags_whitelist=None,
                  has_reference: bool = False,
                  has_pseudo_reference: bool = False,
                  action_space_path: Optional[str] = None,
                  review_mode: str = "studio") -> tuple:
    return _build_common_prompt(
        sample_id=sample_id,
        round_idx=round_idx,
        active_group=active_group,
        has_prev=has_prev,
        az=az,
        el=el,
        prompt_appendix=prompt_appendix,
        issue_tags_whitelist=issue_tags_whitelist,
        has_reference=has_reference,
        has_pseudo_reference=has_pseudo_reference,
        action_space_path=action_space_path,
        review_mode=review_mode,
    )


def _build_freeform_prompt(sample_id: str, round_idx: int, active_group: str,
                           has_prev: bool, az: int, el: int,
                           prompt_appendix: str = "",
                           has_reference: bool = False,
                           has_pseudo_reference: bool = False,
                           action_space_path: Optional[str] = None,
                           review_mode: str = "studio") -> tuple:
    allowed_actions_str = ", ".join(_allowed_actions_all(action_space_path))
    image_desc_parts = [f"Image 1 is the CURRENT render (view az={az}, el={el})."]
    next_idx = 2
    if has_reference:
        image_desc_parts.append(
            f"Image {next_idx} is the ORIGINAL REFERENCE image (object identity, texture, color reference)."
        )
        next_idx += 1
    if has_pseudo_reference:
        image_desc_parts.append(
            f"Image {next_idx} is the PSEUDO-REFERENCE image (rough target scene integration example, not strict ground truth)."
        )
        next_idx += 1
    if has_prev:
        image_desc_parts.append(f"Image {next_idx} is the PREVIOUS render (round {round_idx-1}).")

    system_msg = (
        "You are a practical visual reviewer helping improve 3D scene-insertion renders. "
        "Think carefully, then give a concise final answer in plain text. "
        "Focus on what a human viewer would immediately notice: size, color mismatch, unnatural lighting, fake material, "
        "weak grounding, or obviously implausible placement. "
        "Do not force a score-first answer. Prioritize actionable critique."
    )

    if review_mode == "scene_insert":
        task_block = (
            "Judge the final image effect first. If the image is already believable overall, say so. "
            "If not, identify the most important problems in natural language. "
            "Use the pseudo-reference only as a rough scene feel / placement hint. "
            "Do not treat it as exact ground truth for absolute size, exact framing, or exact shadow shape."
        )
    else:
        task_block = (
            "Judge the render primarily by visual usability and whether the object still looks like the same object."
        )

    user_msg = (
        f"Review this render.\n"
        f"sample_id={sample_id}, round={round_idx}, view=az{az:03d}_el{el:+03d}, active_fix_group={active_group}\n"
        f"{' '.join(image_desc_parts)}\n\n"
        f"{task_block}\n\n"
        "Please respond in plain text with exactly these sections:\n"
        "Verdict: keep / revise / reject\n"
        "Asset viability: continue / abandon / unclear\n"
        "Abandon reason: ... (write 'none' if not abandoning)\n"
        "Abandon confidence: low / medium / high / none\n"
        "Major issues:\n"
        "- ...\n"
        "Suggested fixes:\n"
        "- ...\n"
        "If helpful, mention concrete action names from this action space, even if they belong to a different fix group than the current active group. Natural language is preferred:\n"
        f"{allowed_actions_str}\n\n"
        "Be specific if something is too dark, too black, too large, too small, has obvious color difference, "
        "has fake lighting, fake material, weak grounding, or bad shadow integration."
    )
    if prompt_appendix:
        user_msg = user_msg + "\n\n" + prompt_appendix
    return system_msg, user_msg


def _iter_json_candidates(text: str) -> List[str]:
    """Yield plausible JSON object substrings from a noisy model response."""
    candidates: List[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    fenced_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates.extend(fenced_matches)

    start = None
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:idx + 1].strip())
                    start = None

    seen = set()
    unique: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _extract_json(text: str) -> Optional[dict]:
    """Extract a valid JSON object from model output, tolerating noisy wrappers."""
    for candidate in _iter_json_candidates(text):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _build_json_repair_prompt(raw_text: str, sample_id: str, round_idx: int,
                              active_group: str,
                              action_space_path: Optional[str] = None) -> tuple:
    allowed_actions = ", ".join(_allowed_actions_all(action_space_path))
    repair_system = (
        "You convert an image-review analysis into a strict JSON object. "
        "Return ONLY JSON, no explanation, no markdown fences. "
        "The first character must be '{' and the last character must be '}'."
    )
    repair_user = (
        f"The previous model output was not valid JSON. Convert it into a valid JSON object for sample_id={sample_id}, "
        f"round_idx={round_idx}, active_group={active_group}.\n\n"
        "Use only these enums when filling fields:\n"
        f"- issue_tags: {', '.join(ISSUE_TAGS)}\n"
        f"- suggested_actions: {allowed_actions}\n"
        f"- lighting_diagnosis: {', '.join(sorted(LIGHTING_DIAGNOSIS_VALUES))}\n"
        f"- structure_consistency: {', '.join(sorted(STRUCTURE_VALUES))}\n"
        f"- color_consistency: {', '.join(sorted(COLOR_VALUES))}\n"
        f"- physics_consistency: {', '.join(sorted(PHYSICS_VALUES))}\n"
        f"- asset_viability: {', '.join(sorted(ASSET_VIABILITY_VALUES))}\n"
        f"- abandon_confidence: {', '.join(sorted(ABANDON_CONFIDENCE_VALUES))}, null\n"
        "- pairwise_vs_prev.winner: current, previous, tie, none\n"
        "- pairwise_vs_prev subfields: better, same, worse, na\n"
        "- confidence values: low, medium, high\n"
        "- score values: integers 1 to 5\n\n"
        "Required JSON schema keys:\n"
        "schema_version, sample_id, round_idx, vlm_route, scores, confidence, issue_tags, suggested_actions, "
        "lighting_diagnosis, structure_consistency, color_consistency, physics_consistency, "
        "asset_viability, abandon_reason, abandon_confidence, pairwise_vs_prev\n\n"
        "Previous output to convert:\n"
        f"{raw_text}"
    )
    return repair_system, repair_user


def _coerce_freeform_to_json(model, processor, raw_text: str, sample_id: str,
                             round_idx: int, active_group: str, device: str,
                             action_space_path: Optional[str] = None) -> Optional[dict]:
    import torch

    repair_system, repair_user = _build_json_repair_prompt(
        raw_text,
        sample_id=sample_id,
        round_idx=round_idx,
        active_group=active_group,
        action_space_path=action_space_path,
    )
    repair_messages = [
        {"role": "system", "content": [{"type": "text", "text": repair_system}]},
        {"role": "user", "content": [{"type": "text", "text": repair_user}]},
    ]
    try:
        text_input = processor.apply_chat_template(
            repair_messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        text_input = text_input + "{"
        inputs = processor(text=[text_input], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        repair_text = "{" + processor.decode(generated, skip_special_tokens=True)
        return _extract_json(repair_text)
    except Exception as exc:
        print(f"  [VLM] JSON repair error: {exc}")
        return None


def _heuristic_review_from_freeform(raw_text: str, sample_id: str, round_idx: int) -> dict:
    text = raw_text.lower()
    issue_tags = []
    suggested_actions = []
    scores = {
        "lighting": 3,
        "object_integrity": 3,
        "composition": 3,
        "render_quality_semantic": 3,
        "overall": 3,
    }
    lighting_diagnosis = "good"
    structure_consistency = "good"
    color_consistency = "good"
    physics_consistency = "good"
    vlm_route = "needs_fix"
    asset_viability, abandon_reason, abandon_confidence = detect_asset_viability(raw_text)

    def contains(*needles):
        return any(n in text for n in needles)

    if contains("too large", "too big", "oversized", "过大", "太大", "偏大"):
        issue_tags.append("scale_implausible")
        suggested_actions.append("O_SCALE_DOWN_10")
        scores["composition"] = min(scores["composition"], 2)

    if contains("too small", "undersized", "过小", "太小", "偏小"):
        issue_tags.append("scale_implausible")
        suggested_actions.append("O_SCALE_UP_10")
        scores["composition"] = min(scores["composition"], 2)

    if contains("too dark", "too black", "underexposed", "darker than reference", "太暗", "太黑", "偏黑"):
        lighting_diagnosis = "underexposed_global"
        issue_tags.append("underexposed")
        suggested_actions.extend(["ENV_STRENGTH_UP", "M_VALUE_UP"])
        scores["lighting"] = min(scores["lighting"], 2)
        color_consistency = "minor_shift" if color_consistency == "good" else color_consistency

    if contains("color difference", "color mismatch", "color shift", "色差", "偏色", "颜色不一致", "颜色太假"):
        issue_tags.append("color_shift")
        suggested_actions.extend(["M_VALUE_UP", "M_SATURATION_DOWN"])
        color_consistency = "major_shift" if contains("major", "severe", "明显", "很大") else "minor_shift"
        scores["render_quality_semantic"] = min(scores["render_quality_semantic"], 2)

    if contains("lighting mismatch", "light mismatch", "fake lighting", "光影不一致", "光照不一致", "阴影方向不一致"):
        issue_tags.append("scene_light_mismatch")
        suggested_actions.append("ENV_ROTATE_30")
        lighting_diagnosis = "scene_light_mismatch"
        scores["lighting"] = min(scores["lighting"], 2)

    if contains("shadow missing", "missing shadow", "grounding weak", "接触阴影", "阴影缺失", "缺少阴影"):
        issue_tags.append("shadow_missing")
        suggested_actions.append("S_CONTACT_SHADOW_UP")
        if lighting_diagnosis == "good":
            lighting_diagnosis = "shadow_missing"
        scores["lighting"] = min(scores["lighting"], 2)

    if contains("material fake", "material looks fake", "too glossy", "plastic", "材质假", "塑料感", "反光假"):
        suggested_actions.append("M_ROUGHNESS_UP")
        scores["render_quality_semantic"] = min(scores["render_quality_semantic"], 2)

    if contains("floating", "hovering", "漂浮", "浮空"):
        issue_tags.append("floating_visible")
        suggested_actions.append("O_LOWER_SMALL")
        physics_consistency = "minor_issue"
        scores["composition"] = min(scores["composition"], 2)

    if contains("intersecting ground", "embedded in ground", "穿地", "穿插地面"):
        issue_tags.append("ground_intersection_visible")
        suggested_actions.append("O_LIFT_SMALL")
        physics_consistency = "minor_issue"
        scores["composition"] = min(scores["composition"], 2)

    if contains("different object", "wrong object", "不是同一个", "结构不一致"):
        structure_consistency = "major_mismatch"
        scores["object_integrity"] = min(scores["object_integrity"], 2)
        if asset_viability != "abandon":
            asset_viability = "abandon"
            abandon_reason = abandon_reason or "wrong object geometry"
            abandon_confidence = abandon_confidence or "high"

    if contains("keep", "acceptable", "good enough", "可以接受", "可接受", "能接受") and not issue_tags:
        vlm_route = "pass"
        scores["overall"] = 4

    if contains("reject", "unusable", "不可用", "不行") and len(issue_tags) >= 2:
        vlm_route = "reject"
        scores["overall"] = 2

    if asset_viability == "abandon" and vlm_route != "pass":
        vlm_route = "reject"
        scores["overall"] = min(scores["overall"], 2)

    issue_tags = _dedupe_keep_order(issue_tags)[:3] or ["none"]
    suggested_actions = _dedupe_keep_order(suggested_actions)[:2] or ["NO_OP"]

    return {
        "schema_version": "vlm_review_v1",
        "sample_id": sample_id,
        "round_idx": round_idx,
        "vlm_route": vlm_route,
        "scores": scores,
        "confidence": {k: "medium" for k in scores},
        "issue_tags": issue_tags,
        "suggested_actions": suggested_actions,
        "lighting_diagnosis": lighting_diagnosis,
        "structure_consistency": structure_consistency,
        "color_consistency": color_consistency,
        "physics_consistency": physics_consistency,
        "asset_viability": asset_viability,
        "abandon_reason": abandon_reason,
        "abandon_confidence": abandon_confidence,
        "pairwise_vs_prev": {
            "available": False,
            "winner": "none",
            "lighting": "na",
            "object_integrity": "na",
            "composition": "na",
            "render_quality_semantic": "na",
        },
    }


def _validate_review(obj: dict, sample_id: str, round_idx: int,
                     action_space_path: Optional[str] = None) -> dict:
    """Coerce and fill missing fields with safe defaults."""
    if not isinstance(obj, dict):
        obj = {}
    obj.setdefault("schema_version", "vlm_review_v1")
    obj.setdefault("sample_id", sample_id)
    obj.setdefault("round_idx", round_idx)
    obj.setdefault("vlm_route", "needs_fix")

    scores_default = {"lighting": 3, "object_integrity": 3, "composition": 3,
                      "render_quality_semantic": 3, "overall": 3}
    scores = obj.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}
    for k, v in scores_default.items():
        if k not in scores or not isinstance(scores[k], int):
            try:
                scores[k] = max(1, min(5, int(scores.get(k, v))))
            except Exception:
                scores[k] = v
        else:
            scores[k] = max(1, min(5, int(scores[k])))
    obj["scores"] = scores

    conf_default = {k: "medium" for k in scores_default}
    conf = obj.get("confidence", {})
    if not isinstance(conf, dict):
        conf = {}
    for k in conf_default:
        if k not in conf or conf[k] not in ("low", "medium", "high"):
            conf[k] = "medium"
    obj["confidence"] = conf

    tags = obj.get("issue_tags", [])
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        tags = []
    tags = [t for t in tags if t in ISSUE_TAGS][:3]
    if not tags:
        tags = ["none"]
    obj["issue_tags"] = tags

    actions = obj.get("suggested_actions", ["NO_OP"])
    if isinstance(actions, str):
        actions = [actions]
    if not isinstance(actions, list):
        actions = ["NO_OP"]
    actions = [a for a in actions if isinstance(a, str)][:2]
    # Fix 1: Hard-validate actions against action_space (filter stale/illegal names)
    try:
        _aspace = _load_action_space_data(action_space_path)
        valid_actions = set()
        for grp_data in _aspace["groups"].values():
            valid_actions.update(grp_data["actions"].keys())
        valid_actions.update(_aspace.get("compound_actions", {}).keys())
        valid_actions.add("NO_OP")
        actions = [a for a in actions if a in valid_actions]
    except Exception:
        pass  # If action_space unavailable, skip validation
    if not actions:
        actions = ["NO_OP"]
    obj["suggested_actions"] = actions

    # Validate lighting_diagnosis (optional field, default "good")
    diagnosis = obj.get("lighting_diagnosis", "good")
    if diagnosis not in LIGHTING_DIAGNOSIS_VALUES:
        diagnosis = "good"
    obj["lighting_diagnosis"] = diagnosis

    # v6: Validate new diagnostic fields (optional, default "good")
    sc = obj.get("structure_consistency", "good")
    if sc not in STRUCTURE_VALUES:
        sc = "good"
    obj["structure_consistency"] = sc

    cc = obj.get("color_consistency", "good")
    if cc not in COLOR_VALUES:
        cc = "good"
    obj["color_consistency"] = cc

    pc = obj.get("physics_consistency", "good")
    if pc not in PHYSICS_VALUES:
        pc = "good"
    obj["physics_consistency"] = pc

    asset_viability = str(obj.get("asset_viability", "unclear") or "unclear").lower()
    if asset_viability not in ASSET_VIABILITY_VALUES:
        asset_viability = "unclear"
    obj["asset_viability"] = asset_viability

    abandon_reason = obj.get("abandon_reason")
    if abandon_reason is not None:
        abandon_reason = str(abandon_reason).strip() or None
    if str(abandon_reason or "").lower() in {"none", "null", "n/a", "na"}:
        abandon_reason = None
    obj["abandon_reason"] = abandon_reason

    abandon_confidence = str(obj.get("abandon_confidence") or "").lower() or None
    if abandon_confidence in {"none", "null", "n/a", "na"}:
        abandon_confidence = None
    if abandon_confidence not in ABANDON_CONFIDENCE_VALUES:
        abandon_confidence = None
    obj["abandon_confidence"] = abandon_confidence

    ppv = obj.get("pairwise_vs_prev", {})
    if not isinstance(ppv, dict):
        ppv = {}
    ppv.setdefault("available", False)
    ppv.setdefault("winner", "none")
    for k in ("lighting", "object_integrity", "composition", "render_quality_semantic"):
        ppv.setdefault(k, "na")
    obj["pairwise_vs_prev"] = ppv

    return obj


def run_vlm_review(rgb_path: str, sample_id: str, round_idx: int,
                   active_group: str, az: int, el: int,
                   prev_rgb_path: Optional[str] = None,
                   device: str = "cuda:0", max_retries: int = 3,
                   prompt_appendix: str = "",
                   issue_tags_whitelist=None,
                   reference_image_path: Optional[str] = None,
                   pseudo_reference_path: Optional[str] = None,
                   action_space_path: Optional[str] = None,
                   review_mode: str = "studio",
                   use_freeform_first: bool = False,
                   enable_thinking: bool = False) -> dict:
    """Run VLM inference for one view. Returns validated review dict."""
    import torch

    model, processor = load_vlm(device)
    has_reference = reference_image_path is not None and os.path.exists(reference_image_path)
    has_pseudo_reference = pseudo_reference_path is not None and os.path.exists(pseudo_reference_path)
    has_prev = prev_rgb_path is not None
    dialogue_trace = {
        "sample_id": sample_id,
        "round_idx": round_idx,
        "review_mode": review_mode,
        "use_freeform_first": bool(use_freeform_first),
        "enable_thinking": bool(enable_thinking),
        "attempts": [],
    }

    # Build message content: current render first, then original reference,
    # then pseudo-reference, then previous
    content = [{"type": "image", "image": rgb_path}]
    if has_reference:
        content.append({"type": "image", "image": reference_image_path})
    if has_pseudo_reference:
        content.append({"type": "image", "image": pseudo_reference_path})
    if prev_rgb_path and os.path.exists(prev_rgb_path):
        content.append({"type": "image", "image": prev_rgb_path})
    for attempt in range(max_retries):
        try:
            if use_freeform_first:
                freeform_system, freeform_user = _build_freeform_prompt(
                    sample_id, round_idx, active_group, has_prev, az, el,
                    prompt_appendix=prompt_appendix,
                    has_reference=has_reference,
                    has_pseudo_reference=has_pseudo_reference,
                    action_space_path=action_space_path,
                    review_mode=review_mode,
                )
                freeform_content = list(content)
                freeform_content.append({"type": "text", "text": freeform_user})
                freeform_messages = [
                    {"role": "system", "content": [{"type": "text", "text": freeform_system}]},
                    {"role": "user", "content": freeform_content},
                ]
                text_input = processor.apply_chat_template(
                    freeform_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": bool(enable_thinking)},
                )
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(freeform_messages)
                inputs = processor(
                    text=[text_input],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                    )
                generated = output_ids[0][inputs.input_ids.shape[1]:]
                freeform_text = processor.decode(generated, skip_special_tokens=True).strip()

                repaired = _coerce_freeform_to_json(
                    model,
                    processor,
                    freeform_text,
                    sample_id=sample_id,
                    round_idx=round_idx,
                    active_group=active_group,
                    device=device,
                    action_space_path=action_space_path,
                )
                parse_mode = "repaired_from_freeform"
                if repaired is None:
                    repaired = _heuristic_review_from_freeform(freeform_text, sample_id, round_idx)
                    parse_mode = "heuristic_from_freeform"
                validated = _validate_review(repaired, sample_id, round_idx, action_space_path=action_space_path)
                validated["freeform_feedback"] = freeform_text
                validated["freeform_action_hints"] = list(validated.get("suggested_actions", []))
                validated["review_backend"] = "qwen35_freeform"
                validated["parse_mode"] = parse_mode
                validated["used_thinking"] = bool(enable_thinking)
                dialogue_trace["attempts"].append({
                    "attempt": attempt + 1,
                    "mode": "freeform",
                    "system_prompt": freeform_system,
                    "user_prompt": freeform_user,
                    "assistant_text": freeform_text,
                    "parse_mode": parse_mode,
                })
                validated["vlm_dialogue"] = dialogue_trace
                return validated

            system_msg, user_msg = _build_prompt(
                sample_id, round_idx, active_group, has_prev, az, el,
                prompt_appendix=prompt_appendix,
                issue_tags_whitelist=issue_tags_whitelist,
                has_reference=has_reference,
                has_pseudo_reference=has_pseudo_reference,
                action_space_path=action_space_path,
                review_mode=review_mode,
            )
            json_content = list(content)
            json_content.append({"type": "text", "text": user_msg})
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_msg}]},
                {"role": "user", "content": json_content},
            ]
            text_input = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            text_input = text_input + "{"
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            generated = output_ids[0][inputs.input_ids.shape[1]:]
            text_out = "{" + processor.decode(generated, skip_special_tokens=True)
            parsed = _extract_json(text_out)

            if parsed is not None:
                validated = _validate_review(parsed, sample_id, round_idx, action_space_path=action_space_path)
                validated["review_backend"] = "qwen35_json"
                validated["parse_mode"] = "direct_json"
                validated["used_thinking"] = False
                dialogue_trace["attempts"].append({
                    "attempt": attempt + 1,
                    "mode": "json",
                    "system_prompt": system_msg,
                    "user_prompt": user_msg,
                    "assistant_text": text_out,
                    "parse_mode": "direct_json",
                })
                validated["vlm_dialogue"] = dialogue_trace
                return validated

            repaired = _coerce_freeform_to_json(
                model,
                processor,
                text_out,
                sample_id=sample_id,
                round_idx=round_idx,
                active_group=active_group,
                device=device,
                action_space_path=action_space_path,
            )
            if repaired is not None:
                print(f"  [VLM] attempt {attempt+1}: JSON repair fallback succeeded")
                validated = _validate_review(repaired, sample_id, round_idx, action_space_path=action_space_path)
                validated["freeform_feedback"] = text_out
                validated["freeform_action_hints"] = list(validated.get("suggested_actions", []))
                validated["review_backend"] = "qwen35_json"
                validated["parse_mode"] = "repaired_from_json_failure"
                validated["used_thinking"] = False
                dialogue_trace["attempts"].append({
                    "attempt": attempt + 1,
                    "mode": "json",
                    "system_prompt": system_msg,
                    "user_prompt": user_msg,
                    "assistant_text": text_out,
                    "parse_mode": "repaired_from_json_failure",
                })
                validated["vlm_dialogue"] = dialogue_trace
                return validated

            dialogue_trace["attempts"].append({
                "attempt": attempt + 1,
                "mode": "json",
                "assistant_text": text_out,
                "parse_mode": "failed",
            })
            print(f"  [VLM] attempt {attempt+1}: JSON parse failed, raw='{text_out[:120]}'")
        except Exception as e:
            print(f"  [VLM] attempt {attempt+1} error: {e}")
            print(traceback.format_exc())

    # Fallback: return neutral review
    print(f"  [VLM] all {max_retries} attempts failed, returning neutral review")
    validated = _validate_review({}, sample_id, round_idx, action_space_path=action_space_path)
    validated["review_backend"] = "qwen35_fallback"
    validated["parse_mode"] = "neutral_fallback"
    validated["used_thinking"] = bool(enable_thinking and use_freeform_first)
    validated["vlm_dialogue"] = dialogue_trace
    return validated


# ─────────────────────────────────────────────────────────────────────────────
# Per-object review (aggregates multiple views)
# ─────────────────────────────────────────────────────────────────────────────

def _review_object_legacy(obj_id: str, renders_dir: str, output_dir: str,
                  round_idx: int = 0, active_group: str = "lighting",
                  prev_renders_dir: Optional[str] = None,
                  device: str = "cuda:0",
                  history_file: Optional[str] = None,
                  prompt_appendix: str = "",
                  issue_tags_whitelist=None,
                  reference_image_path: Optional[str] = None,
                  pseudo_reference_path: Optional[str] = None,
                  profile_cfg: dict = None) -> dict:
    """
    Review all representative views for obj_id. Returns aggregated result.
    Saves per-view reviews to output_dir/{obj_id}_round{round_idx}_{az}_{el}.json
    and aggregated to output_dir/{obj_id}_round{round_idx}_agg.json.
    """
    obj_renders = os.path.join(renders_dir, obj_id)
    os.makedirs(output_dir, exist_ok=True)

    # v6: Resolve reference image if not explicitly provided
    if reference_image_path is None:
        reference_image_path = _resolve_reference_image(obj_id, profile_cfg, renders_dir)
    if reference_image_path:
        print(f"  [{obj_id}] Reference image: {reference_image_path}")
    if pseudo_reference_path is None:
        pseudo_reference_path = _resolve_pseudo_reference_image(obj_id, profile_cfg, renders_dir)
    if pseudo_reference_path:
        print(f"  [{obj_id}] Pseudo-reference image: {pseudo_reference_path}")

    per_view_results = []

    for az, el in REP_VIEWS:
        el_str = f"{el:+03d}"
        fname  = f"az{az:03d}_el{el_str}.png"
        rgb_path = os.path.join(obj_renders, fname)
        if not os.path.exists(rgb_path):
            print(f"  [review] View not found, skip: {rgb_path}")
            continue

        prev_rgb = None
        if prev_renders_dir:
            prev_rgb = os.path.join(prev_renders_dir, obj_id, fname)
            if not os.path.exists(prev_rgb):
                prev_rgb = None

        # CV metrics (no mask in pipeline renders)
        cv = compute_cv_metrics(rgb_path, mask_path=None, history_file=history_file)

        # VLM review
        sample_id = f"{obj_id}_az{az:03d}_el{el_str}"
        vlm_review = run_vlm_review(
            rgb_path, sample_id, round_idx, active_group, az, el,
            prev_rgb_path=prev_rgb, device=device,
            prompt_appendix=prompt_appendix,
            issue_tags_whitelist=issue_tags_whitelist,
            reference_image_path=reference_image_path,
            pseudo_reference_path=pseudo_reference_path,
        )

        hybrid, route = compute_hybrid_score(vlm_review["scores"], cv)
        vlm_review["cv_metrics"]    = cv
        vlm_review["hybrid_score"]  = hybrid
        vlm_review["hybrid_route"]  = route

        # Save per-view
        view_out = os.path.join(output_dir, f"{obj_id}_r{round_idx:02d}_az{az:03d}_el{el_str}.json")
        with open(view_out, "w") as f:
            json.dump(vlm_review, f, indent=2)

        per_view_results.append(vlm_review)
        print(f"  [{obj_id}] az={az:3d} el={el:+3d} | hybrid={hybrid:.3f} route={route}")

    if not per_view_results:
        print(f"  [review] No views found for {obj_id}")
        return {}

    # Aggregate: mean scores, union issue_tags, majority route
    agg_vlm_scores = {}
    for k in ("lighting", "object_integrity", "composition", "render_quality_semantic", "overall"):
        agg_vlm_scores[k] = round(float(np.mean([r["scores"][k] for r in per_view_results])), 2)

    agg_cv = {}
    for k in ("exposure_score", "sharpness_score", "cv_score"):
        agg_cv[k] = round(float(np.mean([r["cv_metrics"][k] for r in per_view_results])), 4)
    # mask_score / framing_score only when available
    for k in ("mask_score", "framing_score"):
        vals = [r["cv_metrics"][k] for r in per_view_results if r["cv_metrics"].get(k) is not None]
        agg_cv[k] = round(float(np.mean(vals)), 4) if vals else None
    agg_cv["mask_available"] = any(r["cv_metrics"].get("mask_available", False) for r in per_view_results)

    hybrid_scores = [r["hybrid_score"] for r in per_view_results]
    agg_hybrid = round(float(np.mean(hybrid_scores)), 4)
    worst_hybrid = round(float(np.min(hybrid_scores)), 4)

    # Aggregate hybrid with penalty for worst view (dataset quality = min quality view)
    final_hybrid = round(0.7 * agg_hybrid + 0.3 * worst_hybrid, 4)

    routes = [r["hybrid_route"] for r in per_view_results]
    route_counts = {r: routes.count(r) for r in set(routes)}
    final_route = max(route_counts, key=route_counts.get)

    # Collect most frequent issue_tags
    all_tags = []
    for r in per_view_results:
        all_tags.extend(r.get("issue_tags", []))
    all_tags = [t for t in all_tags if t != "none"]
    tag_counts = {}
    for t in all_tags:
        tag_counts[t] = tag_counts.get(t, 0) + 1
    top_tags = sorted(tag_counts, key=tag_counts.get, reverse=True)[:3] or ["none"]

    # Collect suggested actions
    all_actions = []
    for r in per_view_results:
        all_actions.extend(r.get("suggested_actions", []))
    action_counts = {}
    for a in all_actions:
        action_counts[a] = action_counts.get(a, 0) + 1
    top_actions = sorted(action_counts, key=action_counts.get, reverse=True)[:2] or ["NO_OP"]

    # Aggregate lighting_diagnosis (majority vote)
    diag_counts = _Counter(r.get("lighting_diagnosis", "good") for r in per_view_results)
    agg_diagnosis = diag_counts.most_common(1)[0][0]

    # v6: Aggregate structure_consistency (worst-case / any-view)
    struct_values = [r.get("structure_consistency", "good") for r in per_view_results]
    if "major_mismatch" in struct_values:
        agg_structure = "major_mismatch"
    elif "minor_mismatch" in struct_values:
        agg_structure = "minor_mismatch"
    else:
        agg_structure = "good"

    # v6: Aggregate physics_consistency (worst-case / any-view)
    physics_values = [r.get("physics_consistency", "good") for r in per_view_results]
    if "major_issue" in physics_values:
        agg_physics = "major_issue"
    elif "minor_issue" in physics_values:
        agg_physics = "minor_issue"
    else:
        agg_physics = "good"

    # v6: Aggregate color_consistency (majority vote)
    color_counts = _Counter(r.get("color_consistency", "good") for r in per_view_results)
    agg_color = color_counts.most_common(1)[0][0]

    viability_values = [r.get("asset_viability", "unclear") for r in per_view_results]
    if "abandon" in viability_values:
        agg_asset_viability = "abandon"
    elif "continue" in viability_values:
        agg_asset_viability = "continue"
    else:
        agg_asset_viability = "unclear"

    confidence_rank = {"low": 1, "medium": 2, "high": 3}
    agg_abandon_reason = None
    agg_abandon_confidence = None
    for result in per_view_results:
        if result.get("asset_viability") == "abandon" and result.get("abandon_reason"):
            agg_abandon_reason = result.get("abandon_reason")
            break
    abandon_confidences = [
        result.get("abandon_confidence")
        for result in per_view_results
        if result.get("asset_viability") == "abandon" and result.get("abandon_confidence") in confidence_rank
    ]
    if abandon_confidences:
        agg_abandon_confidence = max(abandon_confidences, key=lambda item: confidence_rank[item])

    # Per-view CV summary for downstream checks
    per_view_cv = [
        {
            "az": az,
            "el": el,
            "exposure_score": r["cv_metrics"]["exposure_score"],
            "sharpness_score": r["cv_metrics"]["sharpness_score"],
        }
        for r, (az, el) in zip(per_view_results, REP_VIEWS[:len(per_view_results)])
    ]

    aggregated = {
        "schema_version":    "vlm_review_v1",
        "obj_id":            obj_id,
        "round_idx":         round_idx,
        "active_group":      active_group,
        "num_views_reviewed": len(per_view_results),
        "agg_vlm_scores":    agg_vlm_scores,
        "agg_cv_metrics":    agg_cv,
        "hybrid_score":      final_hybrid,
        "worst_view_hybrid": worst_hybrid,
        "hybrid_route":      final_route,
        "issue_tags":        top_tags,
        "suggested_actions": top_actions,
        "lighting_diagnosis": agg_diagnosis,
        "lighting_diagnosis_stats": dict(diag_counts),
        "structure_consistency": agg_structure,
        "color_consistency": agg_color,
        "physics_consistency": agg_physics,
        "asset_viability": agg_asset_viability,
        "abandon_reason": agg_abandon_reason,
        "abandon_confidence": agg_abandon_confidence,
        "per_view_cv":       per_view_cv,
        "per_view":          [
            {"az": r["sample_id"].split("_az")[1].split("_")[0],
             "el": r["sample_id"].split("_el")[1] if "_el" in r["sample_id"] else "0",
             "hybrid_score": r["hybrid_score"],
             "route": r["hybrid_route"]}
            for r in per_view_results
        ],
    }

    agg_out = os.path.join(output_dir, f"{obj_id}_r{round_idx:02d}_agg.json")
    with open(agg_out, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"  [{obj_id}] round={round_idx} | final_hybrid={final_hybrid:.3f} | route={final_route} | tags={top_tags} | diagnosis={agg_diagnosis} | struct={agg_structure} | color={agg_color} | physics={agg_physics}")
    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _load_scene_metadata(obj_renders: str) -> dict:
    meta_path = os.path.join(obj_renders, "metadata.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [scene-meta] failed to read {meta_path}: {e}")
        return {}


def _merge_programmatic_physics(scene_meta: dict, physics_consistency: str) -> tuple:
    if not scene_meta:
        return physics_consistency, {}
    gap = float(scene_meta.get("contact_gap", 0.0) or 0.0)
    penetration = float(scene_meta.get("penetration_depth", 0.0) or 0.0)
    out_of_bounds = bool(scene_meta.get("is_out_of_support_bounds", False))
    merged = physics_consistency
    if gap > 0.05 or penetration > 0.05 or out_of_bounds:
        merged = "major_issue"
    elif (gap > 0.01 or penetration > 0.01) and merged == "good":
        merged = "minor_issue"
    elif gap < 0.01 and penetration < 0.01 and not out_of_bounds:
        # Programmatic physics is clean: downgrade VLM false positives
        if merged == "major_issue":
            merged = "minor_issue"
    return merged, {
        "contact_gap": round(gap, 6),
        "penetration_depth": round(penetration, 6),
        "is_floating": bool(scene_meta.get("is_floating", False)),
        "is_intersecting_support": bool(scene_meta.get("is_intersecting_support", False)),
        "is_out_of_support_bounds": out_of_bounds,
        "bbox_height": scene_meta.get("bbox_height"),
    }


def review_object(obj_id: str, renders_dir: str, output_dir: str,
                  round_idx: int = 0, active_group: str = "lighting",
                  prev_renders_dir: Optional[str] = None,
                  device: str = "cuda:0",
                  history_file: Optional[str] = None,
                  prompt_appendix: str = "",
                  issue_tags_whitelist=None,
                  reference_image_path: Optional[str] = None,
                  pseudo_reference_path: Optional[str] = None,
                  profile_cfg: dict = None,
                  review_mode: str = "studio",
                  action_space_path: Optional[str] = None,
                  rep_views=None) -> dict:
    obj_renders = os.path.join(renders_dir, obj_id)
    os.makedirs(output_dir, exist_ok=True)

    if reference_image_path is None:
        reference_image_path = _resolve_reference_image(obj_id, profile_cfg, renders_dir)
    if reference_image_path:
        print(f"  [{obj_id}] Reference image: {reference_image_path}")
    if pseudo_reference_path is None:
        pseudo_reference_path = _resolve_pseudo_reference_image(obj_id, profile_cfg, renders_dir)
    if pseudo_reference_path:
        print(f"  [{obj_id}] Pseudo-reference image: {pseudo_reference_path}")

    resolved_action_space = action_space_path or (
        SCENE_ACTION_SPACE_PATH if review_mode == "scene_insert" else ACTION_SPACE_PATH
    )
    views = _resolve_rep_views(review_mode=review_mode, rep_views=rep_views)
    scene_meta = _load_scene_metadata(obj_renders) if review_mode == "scene_insert" else {}
    use_freeform_vlm_feedback = bool((profile_cfg or {}).get("use_freeform_vlm_feedback", review_mode == "scene_insert"))
    enable_vlm_thinking = bool((profile_cfg or {}).get("enable_vlm_thinking", review_mode == "scene_insert"))

    per_view_results = []
    view_labels = []

    for az, el in views:
        el_str = f"{el:+03d}"
        fname = f"az{az:03d}_el{el_str}.png"
        rgb_path = os.path.join(obj_renders, fname)
        if not os.path.exists(rgb_path):
            print(f"  [review] View not found, skip: {rgb_path}")
            continue

        prev_rgb = None
        if prev_renders_dir:
            prev_rgb = os.path.join(prev_renders_dir, obj_id, fname)
            if not os.path.exists(prev_rgb):
                prev_rgb = None

        mask_path = None
        if review_mode == "scene_insert":
            candidate_mask = os.path.join(obj_renders, f"az{az:03d}_el{el_str}_mask.png")
            if os.path.exists(candidate_mask):
                mask_path = candidate_mask

        cv = compute_cv_metrics(
            rgb_path,
            mask_path=mask_path,
            history_file=history_file,
            review_mode=review_mode,
        )

        sample_id = f"{obj_id}_az{az:03d}_el{el_str}"
        vlm_review = run_vlm_review(
            rgb_path, sample_id, round_idx, active_group, az, el,
            prev_rgb_path=prev_rgb, device=device,
            prompt_appendix=prompt_appendix,
            issue_tags_whitelist=issue_tags_whitelist,
            reference_image_path=reference_image_path,
            pseudo_reference_path=pseudo_reference_path,
            action_space_path=resolved_action_space,
            review_mode=review_mode,
            use_freeform_first=use_freeform_vlm_feedback,
            enable_thinking=enable_vlm_thinking,
        )

        hybrid, route = compute_hybrid_score(vlm_review["scores"], cv, review_mode=review_mode)
        vlm_only = round(compute_vlm_score(vlm_review["scores"]), 4)
        vlm_review["cv_metrics"] = cv
        vlm_review["hybrid_score"] = hybrid
        vlm_review["hybrid_route"] = route
        vlm_review["vlm_only_score"] = vlm_only
        vlm_review["cv_only_score"] = cv["cv_score"]

        trace_path = None
        dialogue_trace = vlm_review.pop("vlm_dialogue", None)
        if dialogue_trace is not None:
            trace_path = os.path.join(output_dir, f"{obj_id}_r{round_idx:02d}_az{az:03d}_el{el_str}_trace.json")
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(dialogue_trace, f, indent=2, ensure_ascii=False)
            vlm_review["vlm_dialogue_path"] = trace_path

        view_out = os.path.join(output_dir, f"{obj_id}_r{round_idx:02d}_az{az:03d}_el{el_str}.json")
        with open(view_out, "w") as f:
            json.dump(vlm_review, f, indent=2)

        per_view_results.append(vlm_review)
        view_labels.append((az, el))
        print(f"  [{obj_id}] az={az:3d} el={el:+3d} | hybrid={hybrid:.3f} route={route}")

    if not per_view_results:
        print(f"  [review] No views found for {obj_id}")
        return {}

    agg_vlm_scores = {}
    for k in ("lighting", "object_integrity", "composition", "render_quality_semantic", "overall"):
        agg_vlm_scores[k] = round(float(np.mean([r["scores"][k] for r in per_view_results])), 2)

    agg_cv = {}
    for k in ("exposure_score", "sharpness_score", "cv_score"):
        agg_cv[k] = round(float(np.mean([r["cv_metrics"][k] for r in per_view_results])), 4)
    for k in ("mask_score", "framing_score"):
        vals = [r["cv_metrics"][k] for r in per_view_results if r["cv_metrics"].get(k) is not None]
        agg_cv[k] = round(float(np.mean(vals)), 4) if vals else None
    agg_cv["mask_available"] = any(r["cv_metrics"].get("mask_available", False) for r in per_view_results)

    def _aggregate_quality(scores):
        mean_v = float(np.mean(scores))
        worst_v = float(np.min(scores))
        return round(0.7 * mean_v + 0.3 * worst_v, 4), round(worst_v, 4)

    final_hybrid, worst_hybrid = _aggregate_quality([r["hybrid_score"] for r in per_view_results])
    final_vlm_only, worst_vlm_only = _aggregate_quality([r["vlm_only_score"] for r in per_view_results])
    final_cv_only, worst_cv_only = _aggregate_quality([r["cv_only_score"] for r in per_view_results])

    routes = [r["hybrid_route"] for r in per_view_results]
    route_counts = {r: routes.count(r) for r in set(routes)}
    final_route = max(route_counts, key=route_counts.get)

    all_tags = []
    for r in per_view_results:
        all_tags.extend(r.get("issue_tags", []))
    all_tags = [t for t in all_tags if t != "none"]
    tag_counts = {}
    for t in all_tags:
        tag_counts[t] = tag_counts.get(t, 0) + 1
    top_tags = sorted(tag_counts, key=tag_counts.get, reverse=True)[:3] or ["none"]

    all_actions = []
    for r in per_view_results:
        all_actions.extend(r.get("suggested_actions", []))
    action_counts = {}
    for a in all_actions:
        action_counts[a] = action_counts.get(a, 0) + 1
    top_actions = sorted(action_counts, key=action_counts.get, reverse=True)[:2] or ["NO_OP"]

    advisor_actions = []
    for r in per_view_results:
        advisor_actions.extend(r.get("freeform_action_hints", []))
        advisor_actions.extend(r.get("suggested_actions", []))
    advisor_counts = {}
    for action in advisor_actions:
        advisor_counts[action] = advisor_counts.get(action, 0) + 1
    advisor_actions = sorted(advisor_counts, key=advisor_counts.get, reverse=True)[:4]

    diag_counts = _Counter(r.get("lighting_diagnosis", "good") for r in per_view_results)
    agg_diagnosis = diag_counts.most_common(1)[0][0]

    struct_values = [r.get("structure_consistency", "good") for r in per_view_results]
    if "major_mismatch" in struct_values:
        agg_structure = "major_mismatch"
    elif "minor_mismatch" in struct_values:
        agg_structure = "minor_mismatch"
    else:
        agg_structure = "good"

    physics_values = [r.get("physics_consistency", "good") for r in per_view_results]
    if "major_issue" in physics_values:
        agg_physics = "major_issue"
    elif "minor_issue" in physics_values:
        agg_physics = "minor_issue"
    else:
        agg_physics = "good"

    color_counts = _Counter(r.get("color_consistency", "good") for r in per_view_results)
    agg_color = color_counts.most_common(1)[0][0]

    viability_values = [r.get("asset_viability", "unclear") for r in per_view_results]
    if "abandon" in viability_values:
        agg_asset_viability = "abandon"
    elif "continue" in viability_values:
        agg_asset_viability = "continue"
    else:
        agg_asset_viability = "unclear"

    confidence_rank = {"low": 1, "medium": 2, "high": 3}
    agg_abandon_reason = None
    agg_abandon_confidence = None
    for result in per_view_results:
        if result.get("asset_viability") == "abandon" and result.get("abandon_reason"):
            agg_abandon_reason = result.get("abandon_reason")
            break
    abandon_confidences = [
        result.get("abandon_confidence")
        for result in per_view_results
        if result.get("asset_viability") == "abandon" and result.get("abandon_confidence") in confidence_rank
    ]
    if abandon_confidences:
        agg_abandon_confidence = max(abandon_confidences, key=lambda item: confidence_rank[item])

    programmatic = {}
    if review_mode == "scene_insert":
        agg_physics, programmatic = _merge_programmatic_physics(scene_meta, agg_physics)
        if programmatic.get("contact_gap", 0.0) > 0.01 and "floating_visible" not in top_tags:
            top_tags = ["floating_visible"] + [t for t in top_tags if t != "none"]
        if programmatic.get("penetration_depth", 0.0) > 0.01 and "ground_intersection_visible" not in top_tags:
            top_tags = ["ground_intersection_visible"] + [t for t in top_tags if t != "none"]
        tgt = profile_cfg.get("target_bbox_height_range") if profile_cfg else None
        if tgt and programmatic.get("bbox_height") is not None:
            bbox_height = float(programmatic["bbox_height"])
            if bbox_height < float(tgt[0]) or bbox_height > float(tgt[1]):
                if "scale_implausible" not in top_tags:
                    top_tags = ["scale_implausible"] + [t for t in top_tags if t != "none"]
        top_tags = (top_tags or ["none"])[:3]
        if agg_diagnosis == "good" and programmatic.get("contact_gap", 0.0) > 0.01:
            agg_diagnosis = "shadow_missing"

    per_view_cv = [
        {
            "az": az,
            "el": el,
            "exposure_score": r["cv_metrics"]["exposure_score"],
            "sharpness_score": r["cv_metrics"]["sharpness_score"],
            "framing_score": r["cv_metrics"].get("framing_score"),
        }
        for r, (az, el) in zip(per_view_results, view_labels)
    ]

    aggregated = {
        "schema_version": "vlm_review_v1",
        "obj_id": obj_id,
        "round_idx": round_idx,
        "active_group": active_group,
        "review_mode": review_mode,
        "reference_image_path": reference_image_path,
        "pseudo_reference_path": pseudo_reference_path,
        "num_views_reviewed": len(per_view_results),
        "agg_vlm_scores": agg_vlm_scores,
        "agg_cv_metrics": agg_cv,
        "hybrid_score": final_hybrid,
        "vlm_only_score": final_vlm_only,
        "cv_only_score": final_cv_only,
        "worst_view_hybrid": worst_hybrid,
        "worst_view_vlm_only": worst_vlm_only,
        "worst_view_cv_only": worst_cv_only,
        "hybrid_route": final_route,
        "issue_tags": top_tags,
        "suggested_actions": top_actions,
        "advisor_actions": advisor_actions,
        "lighting_diagnosis": agg_diagnosis,
        "lighting_diagnosis_stats": dict(diag_counts),
        "structure_consistency": agg_structure,
        "color_consistency": agg_color,
        "physics_consistency": agg_physics,
        "asset_viability": agg_asset_viability,
        "abandon_reason": agg_abandon_reason,
        "abandon_confidence": agg_abandon_confidence,
        "programmatic_physics": programmatic if review_mode == "scene_insert" else None,
        "freeform_feedback_excerpt": [
            {
                "az": az,
                "el": el,
                "dialogue_path": r.get("vlm_dialogue_path"),
                "feedback": (r.get("freeform_feedback", "")[:400] if r.get("freeform_feedback") else None),
            }
            for r, (az, el) in zip(per_view_results, view_labels)
        ],
        "per_view_cv": per_view_cv,
        "per_view": [
            {
                "az": az,
                "el": el,
                "hybrid_score": r["hybrid_score"],
                "vlm_only_score": r["vlm_only_score"],
                "cv_only_score": r["cv_only_score"],
                "route": r["hybrid_route"],
            }
            for r, (az, el) in zip(per_view_results, view_labels)
        ],
    }

    agg_out = os.path.join(output_dir, f"{obj_id}_r{round_idx:02d}_agg.json")
    with open(agg_out, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(
        f"  [{obj_id}] round={round_idx} | final_hybrid={final_hybrid:.3f} | "
        f"route={final_route} | tags={top_tags} | diagnosis={agg_diagnosis} | "
        f"struct={agg_structure} | color={agg_color} | physics={agg_physics} | "
        f"asset_viability={agg_asset_viability}"
    )
    return aggregated


def parse_args():
    p = argparse.ArgumentParser(description="Stage 5.5: VLM render review + CV metrics")
    p.add_argument("--renders-dir",  required=True, help="Root dir with {obj_id}/az*.png")
    p.add_argument("--obj-id",       required=True, help="Object ID to review (e.g. obj_001)")
    p.add_argument("--output-dir",   required=True, help="Directory to save review JSONs")
    p.add_argument("--round-idx",    type=int, default=0)
    p.add_argument("--active-group", default="lighting",
                   choices=["lighting", "camera", "object", "scene", "material"])
    p.add_argument("--prev-renders-dir", default=None,
                   help="Previous-round renders dir for pairwise comparison")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--history-file", default=None,
                   help="JSON file for sharpness rolling percentile history")
    p.add_argument("--review-mode", default="studio", choices=["studio", "scene_insert"])
    p.add_argument("--action-space-path", default=None)
    p.add_argument("--reference-image-path", default=None)
    p.add_argument("--pseudo-reference-path", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = review_object(
        obj_id=args.obj_id,
        renders_dir=args.renders_dir,
        output_dir=args.output_dir,
        round_idx=args.round_idx,
        active_group=args.active_group,
        prev_renders_dir=args.prev_renders_dir,
        device=args.device,
        history_file=args.history_file,
        reference_image_path=args.reference_image_path,
        pseudo_reference_path=args.pseudo_reference_path,
        review_mode=args.review_mode,
        action_space_path=args.action_space_path,
    )
    print(f"\n=== Review summary: hybrid={result.get('hybrid_score','N/A')}, route={result.get('hybrid_route','N/A')} ===")
