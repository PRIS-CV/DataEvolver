from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONTROL_STATE = {
    "object_a": {
        "scale": 1.0,
        "yaw_deg": 0.0,
        "offset_xy": [0.0, 0.0],
        "offset_z": 0.0,
        "ground_snap": 0.0,
    },
    "object_b": {
        "scale": 1.0,
        "yaw_deg": 0.0,
        "offset_xy": [0.0, 0.0],
        "offset_z": 0.0,
        "ground_snap": 0.0,
    },
    "pair": {
        "distance_scale": 1.0,
        "height_offset_scale": 1.0,
        "side_sign": 0,
    },
    "camera": {
        "distance_scale": 1.0,
        "azimuth_offset_deg": 0.0,
        "elevation_deg": 20.0,
        "focal_length_mm": 50.0,
        "framing_margin": 1.0,
    },
    "lighting": {
        "spot_energy_scale": 1.0,
        "fill_energy_scale": 1.0,
        "spot_height": 5.0,
        "spot_size_deg": 45.0,
        "env_strength_scale": 1.0,
        "scene_light_scale": 1.0,
        "scene_area_light_scale": 1.0,
        "scene_point_light_scale": 1.0,
        "scene_spot_light_scale": 1.0,
        "scene_sun_light_scale": 1.0,
        "scene_light_max_energy": 500.0,
    },
    "material": {
        "a_value_scale": 1.0,
        "b_value_scale": 1.0,
        "a_saturation_scale": 1.0,
        "b_saturation_scale": 1.0,
        "a_contrast_scale": 1.0,
        "b_contrast_scale": 1.0,
    },
    "shadow": {
        "contact_shadow_strength": 1.0,
        "ambient_occlusion_strength": 1.0,
        "ground_shadow_receiver": True,
    },
    "scene": {
        "seed_offset": 0,
    },
}


ACTION_SPECS = {
    "A_SCALE_UP_10": ("object_a.scale", "mul", 1.1, 0.4, 2.0),
    "A_SCALE_DOWN_10": ("object_a.scale", "mul", 0.9, 0.4, 2.0),
    "B_SCALE_UP_10": ("object_b.scale", "mul", 1.1, 0.4, 2.0),
    "B_SCALE_DOWN_10": ("object_b.scale", "mul", 0.9, 0.4, 2.0),
    "A_LIFT_SMALL": ("object_a.offset_z", "add", 0.04, -0.3, 0.5),
    "A_LOWER_SMALL": ("object_a.offset_z", "add", -0.04, -0.3, 0.5),
    "B_LIFT_SMALL": ("object_b.offset_z", "add", 0.04, -0.3, 0.8),
    "B_LOWER_SMALL": ("object_b.offset_z", "add", -0.04, -0.3, 0.8),
    "A_LIFT_TINY": ("object_a.offset_z", "add", 0.015, -0.3, 0.5),
    "A_LOWER_TINY": ("object_a.offset_z", "add", -0.015, -0.3, 0.5),
    "B_LIFT_TINY": ("object_b.offset_z", "add", 0.015, -0.3, 0.8),
    "B_LOWER_TINY": ("object_b.offset_z", "add", -0.015, -0.3, 0.8),
    "A_GROUND_SNAP": ("object_a.ground_snap", "add", 1.0, 0.0, 99.0),
    "B_GROUND_SNAP": ("object_b.ground_snap", "add", 1.0, 0.0, 99.0),
    "BOTH_GROUND_SNAP": (None, "compound_ground_snap", 0.0, 0.0, 0.0),
    "A_ROTATE_Z_POS_15": ("object_a.yaw_deg", "add", 15.0, -180.0, 180.0),
    "A_ROTATE_Z_NEG_15": ("object_a.yaw_deg", "add", -15.0, -180.0, 180.0),
    "B_ROTATE_Z_POS_15": ("object_b.yaw_deg", "add", 15.0, -180.0, 180.0),
    "B_ROTATE_Z_NEG_15": ("object_b.yaw_deg", "add", -15.0, -180.0, 180.0),
    "PAIR_INCREASE_DISTANCE": ("pair.distance_scale", "mul", 1.15, 0.5, 2.5),
    "PAIR_DECREASE_DISTANCE": ("pair.distance_scale", "mul", 0.85, 0.5, 2.5),
    "PAIR_B_HIGHER": ("pair.height_offset_scale", "mul", 1.15, 0.4, 2.2),
    "PAIR_B_LOWER": ("pair.height_offset_scale", "mul", 0.85, 0.4, 2.2),
    "CAMERA_WIDER": ("camera.distance_scale", "mul", 1.15, 0.6, 2.0),
    "CAMERA_CLOSER": ("camera.distance_scale", "mul", 0.9, 0.6, 2.0),
    "CAMERA_AZ_POS_15": ("camera.azimuth_offset_deg", "add", 15.0, -90.0, 90.0),
    "CAMERA_AZ_NEG_15": ("camera.azimuth_offset_deg", "add", -15.0, -90.0, 90.0),
    "CAMERA_ELEVATION_UP": ("camera.elevation_deg", "add", 8.0, 5.0, 60.0),
    "CAMERA_ELEVATION_DOWN": ("camera.elevation_deg", "add", -8.0, 5.0, 60.0),
    "CAMERA_FOCAL_WIDER": ("camera.focal_length_mm", "mul", 0.85, 24.0, 85.0),
    "CAMERA_FOCAL_TIGHTER": ("camera.focal_length_mm", "mul", 1.15, 24.0, 85.0),
    "FRAMING_MARGIN_DOWN": ("camera.framing_margin", "mul", 0.9, 0.6, 1.6),
    "FRAMING_MARGIN_UP": ("camera.framing_margin", "mul", 1.1, 0.6, 1.6),
    "L_SPOT_UP": ("lighting.spot_energy_scale", "mul", 1.25, 0.4, 3.0),
    "L_SPOT_DOWN": ("lighting.spot_energy_scale", "mul", 0.8, 0.4, 3.0),
    "L_FILL_UP": ("lighting.fill_energy_scale", "mul", 1.25, 0.4, 3.0),
    "L_FILL_DOWN": ("lighting.fill_energy_scale", "mul", 0.8, 0.4, 3.0),
    "L_SPOT_HIGHER": ("lighting.spot_height", "mul", 1.2, 1.5, 12.0),
    "L_SPOT_LOWER": ("lighting.spot_height", "mul", 0.85, 1.5, 12.0),
    "L_SPOT_WIDER": ("lighting.spot_size_deg", "mul", 1.2, 15.0, 90.0),
    "L_SPOT_NARROWER": ("lighting.spot_size_deg", "mul", 0.85, 15.0, 90.0),
    "ENV_STRENGTH_UP": ("lighting.env_strength_scale", "mul", 1.25, 0.3, 3.0),
    "ENV_STRENGTH_DOWN": ("lighting.env_strength_scale", "mul", 0.8, 0.3, 3.0),
    "SCENE_LIGHTS_DOWN": ("lighting.scene_light_scale", "mul", 0.7, 0.05, 2.0),
    "SCENE_LIGHTS_UP": ("lighting.scene_light_scale", "mul", 1.2, 0.05, 2.0),
    "SCENE_AREA_LIGHTS_DOWN": ("lighting.scene_area_light_scale", "mul", 0.6, 0.05, 2.0),
    "SCENE_POINT_LIGHTS_DOWN": ("lighting.scene_point_light_scale", "mul", 0.6, 0.05, 2.0),
    "SCENE_SPOT_LIGHTS_DOWN": ("lighting.scene_spot_light_scale", "mul", 0.7, 0.05, 2.0),
    "SCENE_SUN_LIGHTS_DOWN": ("lighting.scene_sun_light_scale", "mul", 0.7, 0.05, 2.0),
    "SCENE_LIGHTS_CLAMP_STRONG": ("lighting.scene_light_max_energy", "mul", 0.5, 30.0, 2000.0),
    "CONTACT_SHADOW_UP": ("shadow.contact_shadow_strength", "mul", 1.35, 0.5, 4.0),
    "AO_UP": ("shadow.ambient_occlusion_strength", "mul", 1.35, 0.5, 4.0),
    "AO_DOWN": ("shadow.ambient_occlusion_strength", "mul", 0.75, 0.2, 4.0),
    "A_SATURATION_UP": ("material.a_saturation_scale", "mul", 1.15, 0.4, 2.5),
    "B_SATURATION_UP": ("material.b_saturation_scale", "mul", 1.15, 0.4, 2.5),
    "A_SATURATION_DOWN": ("material.a_saturation_scale", "mul", 0.85, 0.4, 2.5),
    "B_SATURATION_DOWN": ("material.b_saturation_scale", "mul", 0.85, 0.4, 2.5),
    "A_VALUE_DOWN": ("material.a_value_scale", "mul", 0.9, 0.4, 1.8),
    "B_VALUE_DOWN": ("material.b_value_scale", "mul", 0.9, 0.4, 1.8),
    "A_VALUE_UP": ("material.a_value_scale", "mul", 1.1, 0.4, 1.8),
    "B_VALUE_UP": ("material.b_value_scale", "mul", 1.1, 0.4, 1.8),
    "A_CONTRAST_UP": ("material.a_contrast_scale", "mul", 1.15, 0.4, 2.5),
    "B_CONTRAST_UP": ("material.b_contrast_scale", "mul", 1.15, 0.4, 2.5),
    "NO_OP": (None, "noop", 0.0, 0.0, 0.0),
}


ISSUE_TO_ACTIONS = {
    "object_a_too_large": ["CAMERA_FOCAL_WIDER", "A_SCALE_DOWN_10", "CAMERA_WIDER"],
    "object_a_too_small": ["CAMERA_FOCAL_TIGHTER", "CAMERA_CLOSER", "FRAMING_MARGIN_DOWN", "A_SCALE_UP_10"],
    "object_b_too_large": ["CAMERA_FOCAL_WIDER", "B_SCALE_DOWN_10", "CAMERA_WIDER"],
    "object_b_too_small": ["CAMERA_FOCAL_TIGHTER", "CAMERA_CLOSER", "FRAMING_MARGIN_DOWN", "B_SCALE_UP_10"],
    "object_too_large": ["A_SCALE_DOWN_10", "B_SCALE_DOWN_10", "CAMERA_FOCAL_WIDER", "CAMERA_WIDER"],
    "object_too_small": ["CAMERA_FOCAL_TIGHTER", "CAMERA_CLOSER", "FRAMING_MARGIN_DOWN", "A_SCALE_UP_10", "B_SCALE_UP_10"],
    "scale_implausible": ["B_SCALE_DOWN_10", "A_SCALE_DOWN_10", "CAMERA_FOCAL_WIDER", "FRAMING_MARGIN_UP"],
    "object_scene_scale_unrecoverable": ["NO_OP"],
    "objects_too_close": ["PAIR_INCREASE_DISTANCE", "CAMERA_WIDER"],
    "objects_too_far": ["PAIR_DECREASE_DISTANCE", "CAMERA_CLOSER"],
    "object_a_floating": ["A_GROUND_SNAP", "CONTACT_SHADOW_UP", "AO_UP", "CAMERA_ELEVATION_UP"],
    "object_b_floating": ["B_GROUND_SNAP", "CONTACT_SHADOW_UP", "AO_UP", "PAIR_B_LOWER"],
    "floating_object": ["BOTH_GROUND_SNAP", "CONTACT_SHADOW_UP", "AO_UP", "CAMERA_ELEVATION_UP"],
    "ground_intersection": ["BOTH_GROUND_SNAP", "A_LIFT_SMALL", "B_LIFT_SMALL", "CAMERA_ELEVATION_UP"],
    "ground_intersection_visible": ["BOTH_GROUND_SNAP", "A_LIFT_SMALL", "B_LIFT_SMALL", "CAMERA_ELEVATION_UP"],
    "object_a_intersecting": ["A_GROUND_SNAP", "A_LIFT_SMALL"],
    "object_b_intersecting": ["B_GROUND_SNAP", "B_LIFT_SMALL", "PAIR_B_HIGHER"],
    "bad_composition": ["CAMERA_FOCAL_WIDER", "CAMERA_WIDER", "CAMERA_AZ_POS_15"],
    "occlusion": ["PAIR_INCREASE_DISTANCE", "CAMERA_AZ_POS_15", "CAMERA_FOCAL_WIDER", "CAMERA_WIDER"],
    "underexposed": ["ENV_STRENGTH_UP", "L_SPOT_UP", "L_FILL_UP", "L_SPOT_LOWER"],
    "overexposed": ["SCENE_LIGHTS_DOWN", "SCENE_LIGHTS_CLAMP_STRONG", "ENV_STRENGTH_DOWN", "L_SPOT_DOWN"],
    "shadow_missing": ["CONTACT_SHADOW_UP", "AO_UP", "L_SPOT_NARROWER", "SCENE_LIGHTS_DOWN"],
    "flat_lighting": ["SCENE_LIGHTS_DOWN", "SCENE_LIGHTS_CLAMP_STRONG", "ENV_STRENGTH_DOWN", "CONTACT_SHADOW_UP"],
    "scene_light_mismatch": ["SCENE_LIGHTS_DOWN", "SCENE_AREA_LIGHTS_DOWN", "SCENE_LIGHTS_CLAMP_STRONG", "L_SPOT_NARROWER"],
    "washed_out": ["SCENE_LIGHTS_DOWN", "SCENE_AREA_LIGHTS_DOWN", "A_VALUE_DOWN", "B_VALUE_DOWN"],
    "color_too_pale": ["A_SATURATION_UP", "B_SATURATION_UP", "A_CONTRAST_UP", "B_CONTRAST_UP"],
    "flat_material": ["A_CONTRAST_UP", "B_CONTRAST_UP", "A_SATURATION_UP", "B_SATURATION_UP"],
    "low_material_contrast": ["A_CONTRAST_UP", "B_CONTRAST_UP", "A_VALUE_DOWN", "B_VALUE_DOWN"],
    "weak_contact_shadow": ["CONTACT_SHADOW_UP", "AO_UP", "L_SPOT_NARROWER"],
    "object_cropped": ["CAMERA_WIDER", "CAMERA_FOCAL_WIDER", "FRAMING_MARGIN_UP"],
    "relation_unclear": ["CAMERA_ELEVATION_UP", "CAMERA_WIDER", "PAIR_INCREASE_DISTANCE"],
    "too_top_down": ["CAMERA_ELEVATION_DOWN"],
    "scene_bad_geometry": ["NO_OP"],
    "scene_bad_lighting": ["NO_OP"],
    "ground_unreliable": ["NO_OP"],
    "bad_scene_color_cast": ["SCENE_LIGHTS_DOWN", "SCENE_LIGHTS_CLAMP_STRONG", "A_SATURATION_UP", "B_SATURATION_UP"],
    "none": ["NO_OP"],
}


HIGH_PRIORITY_ISSUES = {
    "overexposed",
    "washed_out",
    "color_too_pale",
    "flat_material",
    "low_material_contrast",
    "flat_lighting",
    "scene_light_mismatch",
    "shadow_missing",
    "weak_contact_shadow",
    "ground_intersection",
    "ground_intersection_visible",
    "floating_object",
    "object_a_floating",
    "object_b_floating",
    "scale_implausible",
    "object_too_large",
    "object_a_too_large",
    "object_b_too_large",
    "underexposed",
    "object_cropped",
    "relation_unclear",
}


GEOMETRY_CONTACT_ISSUES = {
    "floating_object",
    "object_a_floating",
    "object_b_floating",
    "ground_intersection",
    "ground_intersection_visible",
    "object_a_intersecting",
    "object_b_intersecting",
}

SCALE_ISSUES = {
    "scale_implausible",
    "object_too_large",
    "object_a_too_large",
    "object_b_too_large",
    "object_scene_scale_unrecoverable",
}

SCENE_HARD_FAIL_ISSUES = {
    "scene_bad_geometry",
    "scene_bad_lighting",
    "ground_unreliable",
    "object_scene_scale_unrecoverable",
}


def load_json(path: str | Path, default: Any = None) -> Any:
    if path is None or str(path) == "":
        return default
    path = Path(path)
    if not path.exists() or path.is_dir():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def default_control_state() -> dict:
    return copy.deepcopy(DEFAULT_CONTROL_STATE)


def get_nested(payload: dict, dotted: str):
    cur = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    return cur, parts[-1]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def apply_action(action: str, state: dict, step_scale: float = 1.0) -> dict:
    action = str(action or "NO_OP").strip()
    spec = ACTION_SPECS.get(action)
    if spec is None:
        return {"action": action, "applied": False, "reason": "unknown_action"}
    target, op, delta, lo, hi = spec
    if op == "noop":
        return {"action": action, "applied": True, "target": None, "value": None}
    if op == "compound_ground_snap":
        a_state = state.setdefault("object_a", {})
        b_state = state.setdefault("object_b", {})
        old_a = float(a_state.get("ground_snap", 0.0))
        old_b = float(b_state.get("ground_snap", 0.0))
        a_state["ground_snap"] = old_a + 1.0
        b_state["ground_snap"] = old_b + 1.0
        a_state["offset_z"] = 0.0
        b_state["offset_z"] = 0.0
        return {
            "action": action,
            "applied": True,
            "target": "object_a.ground_snap,object_b.ground_snap",
            "old_value": [round(old_a, 6), round(old_b, 6)],
            "new_value": [round(float(a_state["ground_snap"]), 6), round(float(b_state["ground_snap"]), 6)],
            "reset_offset_z": True,
        }
    parent, key = get_nested(state, target)
    old = float(parent.get(key, 0.0))
    if op == "mul":
        scaled_delta = 1.0 + (float(delta) - 1.0) * float(step_scale)
        new = old * scaled_delta
    elif op == "add":
        new = old + float(delta) * float(step_scale)
    else:
        return {"action": action, "applied": False, "target": target, "reason": f"unsupported_op:{op}"}
    new = clamp(new, float(lo), float(hi))
    parent[key] = new
    if action == "A_GROUND_SNAP":
        state.setdefault("object_a", {})["offset_z"] = 0.0
    elif action == "B_GROUND_SNAP":
        state.setdefault("object_b", {})["offset_z"] = 0.0
    return {
        "action": action,
        "applied": True,
        "target": target,
        "old_value": round(old, 6),
        "new_value": round(new, 6),
    }


def apply_actions(actions: list[str], state: dict, step_scale: float = 1.0) -> tuple[dict, list[dict]]:
    next_state = copy.deepcopy(state)
    results = []
    for action in actions:
        results.append(apply_action(action, next_state, step_scale=step_scale))
    return next_state, results


def append_issue_actions(actions: list[str], tags: list[str], max_actions: int) -> list[str]:
    for tag in tags:
        for action in ISSUE_TO_ACTIONS.get(str(tag), []):
            if action != "NO_OP" and action not in actions:
                actions.append(action)
            if len(actions) >= max_actions:
                return actions
    return actions


def normalize_action_conflicts(actions: list[str]) -> list[str]:
    cleaned: list[str] = []
    for action in actions:
        if action in {"A_SCALE_UP_10", "B_SCALE_UP_10"}:
            opposite = action.replace("_UP_", "_DOWN_")
            if opposite in cleaned:
                continue
        if action in {"A_SCALE_DOWN_10", "B_SCALE_DOWN_10"}:
            opposite = action.replace("_DOWN_", "_UP_")
            if opposite in cleaned:
                cleaned.remove(opposite)
        if action not in cleaned:
            cleaned.append(action)
    return cleaned


def choose_actions(review: dict, max_actions: int = 4) -> list[str]:
    tags = [str(tag) for tag in review.get("issue_tags") or []]
    direct = [str(v) for v in review.get("suggested_actions") or [] if str(v) in ACTION_SPECS and str(v) != "NO_OP"]
    actions: list[str] = []

    if SCENE_HARD_FAIL_ISSUES & set(tags):
        return ["NO_OP"]

    if "underexposed" in tags:
        append_issue_actions(actions, ["underexposed"], max_actions)
        if len(actions) >= max_actions:
            return normalize_action_conflicts(actions)[:max_actions]

    if GEOMETRY_CONTACT_ISSUES & set(tags):
        for action in ("BOTH_GROUND_SNAP", "A_GROUND_SNAP", "B_GROUND_SNAP"):
            if action in ISSUE_TO_ACTIONS.get("floating_object", []) or action in direct:
                if action not in actions:
                    actions.append(action)
            if len(actions) >= max_actions:
                return normalize_action_conflicts(actions)[:max_actions]
        if not actions:
            actions.append("BOTH_GROUND_SNAP")

    if SCALE_ISSUES & set(tags):
        for action in direct:
            if "SCALE" in action and action not in actions:
                actions.append(action)
            if len(actions) >= max_actions:
                return normalize_action_conflicts(actions)[:max_actions]
        append_issue_actions(actions, [tag for tag in tags if tag in SCALE_ISSUES], max_actions)
        if len(actions) >= max_actions:
            return normalize_action_conflicts(actions)[:max_actions]

    priority_tags = [tag for tag in tags if tag in HIGH_PRIORITY_ISSUES]
    append_issue_actions(actions, priority_tags, max_actions)
    if len(actions) >= max_actions:
        return normalize_action_conflicts(actions)[:max_actions]

    for action in direct:
        if action not in actions:
            actions.append(action)
        if len(actions) >= max_actions:
            return normalize_action_conflicts(actions)[:max_actions]

    append_issue_actions(actions, [tag for tag in tags if tag not in priority_tags], max_actions)
    return normalize_action_conflicts(actions)[:max_actions]
