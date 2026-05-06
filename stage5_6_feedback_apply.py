"""
Stage 5.6: Feedback executor — applies discrete action to control_state.

Takes an aggregated VLM review JSON and the current control_state,
selects the best action from the active group (with anti-oscillation logic),
applies it with step_scale, and writes the new control_state.

v6: Added preset_mode for deterministic cross-group action selection
    based on lighting_diagnosis, color_consistency, physics_consistency.

Usage:
  python pipeline/stage5_6_feedback_apply.py \
    --review-json  pipeline/data/vlm_reviews/obj_001_r00_agg.json \
    --control-state pipeline/data/control_states/obj_001.json \
    --output-control-state pipeline/data/control_states/obj_001_r01.json \
    [--round-idx 1] \
    [--history-json pipeline/data/control_states/obj_001_history.json]
"""

import os
import sys
import json
import copy
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DATA_BUILD_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTION_SPACE_PATH = os.path.join(DATA_BUILD_ROOT, "configs", "action_space.json")

GROUP_ORDER = ["lighting", "camera", "object", "scene", "material"]


# ─────────────────────────────────────────────────────────────────────────────
# ActionResult: unified return type for apply_action
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionResult:
    state: dict
    target: Optional[str]
    delta_sign: int
    applied: bool
    action: str
    sub_results: list = field(default=None)  # List[ActionResult] for compound; None for atomic


# ─────────────────────────────────────────────────────────────────────────────
# Control state helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_action_space(path: Optional[str] = None) -> dict:
    with open(path or ACTION_SPACE_PATH) as f:
        return json.load(f)


def default_control_state(aspace: dict) -> dict:
    return copy.deepcopy(aspace["default_control_state"])


def load_control_state(path: str, aspace: dict) -> dict:
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default_control_state(aspace)


def save_control_state(state: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Anti-oscillation state
# ─────────────────────────────────────────────────────────────────────────────

def load_history(history_path: Optional[str]) -> dict:
    """Load per-target anti-oscillation history."""
    if history_path and os.path.exists(history_path):
        with open(history_path) as f:
            return json.load(f)
    return {}


def save_history(history: dict, history_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(history_path)), exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def update_history(history: dict, target: str, delta_sign: int) -> dict:
    """
    Track sign of last delta per target for oscillation detection.
    delta_sign: +1 or -1
    Freezes target after 3 sign flips.
    """
    hist = copy.deepcopy(history)
    if target not in hist:
        hist[target] = {"last_sign": 0, "flip_count": 0, "frozen": False}
    entry = hist[target]
    if entry["last_sign"] != 0 and entry["last_sign"] != delta_sign:
        entry["flip_count"] += 1
        if entry["flip_count"] >= 3:
            entry["frozen"] = True
            print(f"  [anti-osc] Freezing target '{target}' after {entry['flip_count']} sign flips")
    else:
        entry["flip_count"] = 0
    entry["last_sign"] = delta_sign
    hist[target] = entry
    return hist


def is_frozen(history: dict, target: str) -> bool:
    return history.get(target, {}).get("frozen", False)


# ─────────────────────────────────────────────────────────────────────────────
# Dead zone check
# ─────────────────────────────────────────────────────────────────────────────

def is_in_dead_zone(state: dict, group: str, target: str, aspace: dict) -> bool:
    """
    Skip action if the target param is already at/near its bound
    or if the relevant dimension score is already satisfactory (>=4/5).
    """
    actions = aspace["groups"][group]["actions"]
    bounds = None
    for aname, adef in actions.items():
        if adef["target"] == target:
            bounds = adef["bounds"]
            break
    if bounds is None:
        return False

    current = state.get(group, {}).get(target, 1.0 if target.endswith("scale") else 0.0)
    lo, hi = bounds
    margin = (hi - lo) * 0.05
    if current <= lo + margin or current >= hi - margin:
        print(f"  [dead-zone] Target '{target}' at bound ({current:.3f} in [{lo},{hi}]), skip")
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Compound action helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_action(action_name: str, aspace: dict):
    """Return (group_name, action_def) or (None, None)."""
    for grp, grp_data in aspace["groups"].items():
        if action_name in grp_data["actions"]:
            return grp, grp_data["actions"][action_name]
    return None, None


def _apply_compound(compound_name: str, control_state: dict, aspace: dict,
                    history: dict, step_scale: float = 1.0) -> ActionResult:
    """
    ANY_SUCCESS strategy: >=1 sub-action applied -> compound succeeds.
    Each sub-action independently checks frozen + dead-zone.
    State chains (sub2 sees sub1 result).
    """
    cdef = aspace["compound_actions"][compound_name]
    state = copy.deepcopy(control_state)
    sub_results = []
    any_applied = False

    for sub_name in cdef["sub_actions"]:
        grp, adef = _find_action(sub_name, aspace)
        if adef is None:
            sub_results.append(ActionResult(
                state=state, target=None, delta_sign=0, applied=False,
                action=sub_name, sub_results=None))
            continue
        target = adef["target"]
        # frozen -> skip
        if is_frozen(history, target):
            sub_results.append(ActionResult(
                state=state, target=target, delta_sign=0, applied=False,
                action=sub_name, sub_results=None))
            continue
        # dead-zone -> skip
        if is_in_dead_zone(state, grp, target, aspace):
            sub_results.append(ActionResult(
                state=state, target=target, delta_sign=0, applied=False,
                action=sub_name, sub_results=None))
            continue
        # apply (chained state)
        sr = apply_action(sub_name, state, aspace, step_scale)
        state = sr.state
        any_applied = any_applied or sr.applied
        sub_results.append(sr)

    primary = next(
        (sr for sr in sub_results if sr.applied),
        sub_results[0] if sub_results else
        ActionResult(state=state, target=None, delta_sign=0, applied=False, action=compound_name),
    )
    return ActionResult(
        state=state,
        target=primary.target,
        delta_sign=primary.delta_sign,
        applied=any_applied,
        action=compound_name,
        sub_results=sub_results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Action selection
# ─────────────────────────────────────────────────────────────────────────────



# Fix 3: Conditional compound promotion
PROMOTE_CONTEXT = {
    "L_RIM_UP":  {"weak_subject_separation"},
    "L_KEY_UP":  {"underexposed"},
    "L_FILL_UP": {"underexposed"},
}


def _promote_to_compound(action, active_group, state, history, aspace, issue_tags):
    """Conditionally promote an atomic action to compound when issue_tags match."""
    context_tags = PROMOTE_CONTEXT.get(action, set())
    if not (context_tags & set(issue_tags)):
        return action  # Not in context, no promotion
    for cname, cdef in aspace.get("compound_actions", {}).items():
        if cdef.get("group") != active_group:
            continue
        if action not in cdef["sub_actions"]:
            continue
        # Check at least 1 sub-action is viable
        viable = any(
            _find_action(s, aspace)[1] is not None
            and not is_frozen(history, _find_action(s, aspace)[1]["target"])
            for s in cdef["sub_actions"]
            if _find_action(s, aspace)[1] is not None
        )
        if viable:
            return cname
    return action



def _compound_allowed(action: str, control_state: dict, aspace: dict) -> bool:
    """Check compound_guards: block compound if any state value exceeds max_state."""
    guards = aspace.get("compound_guards", {})
    guard = guards.get(action)
    if guard is None:
        return True  # No guard defined -> allow
    max_state = guard.get("max_state", {})
    lighting = control_state.get("lighting", {})
    for param, threshold in max_state.items():
        current_val = lighting.get(param, 1.0)
        if current_val > threshold:
            print(f"  [compound_guard] BLOCKED {action}: {param}={current_val:.3f} > {threshold}")
            return False
    return True


def select_action(review: dict, active_group: str, control_state: dict,
                  history: dict, aspace: dict,
                  action_blacklist: set = None, issue_tags: list = None,
                  preset_mode: bool = False,
                  action_whitelist: list = None) -> Optional[str]:
    """
    Select the best single action from active_group.

    When preset_mode=True (v6):
      Deterministic selection based on diagnostic fields, skipping VLM suggestions.
      Priority: lighting_diagnosis -> color_consistency -> physics_consistency -> issue_tags fallback.

    When preset_mode=False (legacy):
      Priority:
      1. VLM suggested_actions that belong to active_group (atomic or compound)
      2. issue_to_action_fallback based on issue_tags (atomic or compound)
      3. NO_OP
    """
    if action_blacklist is None:
        action_blacklist = set()

    if issue_tags is None:
        issue_tags = review.get("issue_tags", [])

    whitelist = set(action_whitelist or [])

    def _is_allowed(action: str) -> bool:
        if action == "NO_OP":
            return True
        if not whitelist:
            return True
        return action in whitelist

    # ─── v6: preset_mode — deterministic cross-group selection ───────────
    if preset_mode:
        # 1. lighting / scene lighting (primary optimization dimension)
        if active_group in ("lighting", "scene"):
            diag = review.get("lighting_diagnosis", "good")
            diag_map = aspace.get("lighting_diagnosis_to_action", {})
            diag_actions = diag_map.get(diag, [])
            for action in diag_actions:
                if not _is_allowed(action):
                    continue
                if action == "NO_OP":
                    return "NO_OP"
                if action in action_blacklist:
                    print(f"  [preset] Diagnosis '{diag}' -> '{action}' blacklisted, skip")
                    continue
                # Check if it's a compound action
                if action in aspace.get("compound_actions", {}):
                    if not _compound_allowed(action, control_state, aspace):
                        continue
                    cdef = aspace["compound_actions"][action]
                    viable = any(
                        _find_action(s, aspace)[1] is not None
                        and not is_frozen(history, _find_action(s, aspace)[1]["target"])
                        for s in cdef["sub_actions"]
                        if _find_action(s, aspace)[1] is not None
                    )
                    if viable:
                        print(f"  [preset] Diagnosis '{diag}' -> compound '{action}'")
                        return action
                else:
                    # Atomic action — check group membership, frozen, dead-zone
                    grp, adef = _find_action(action, aspace)
                    if adef is None:
                        continue
                    if is_frozen(history, adef["target"]):
                        continue
                    if grp and is_in_dead_zone(control_state, grp, adef["target"], aspace):
                        continue
                    print(f"  [preset] Diagnosis '{diag}' -> '{action}'")
                    return action

        # 2. color shift -> material preset (max 1 attempt)
        color_diag = review.get("color_consistency", "good")
        if color_diag in ("minor_shift", "major_shift"):
            color_map = aspace.get("color_issue_to_action", {})
            color_actions = color_map.get(color_diag, [])
            for action in color_actions:
                if not _is_allowed(action):
                    continue
                if action in action_blacklist:
                    continue
                grp, adef = _find_action(action, aspace)
                if adef is None:
                    continue
                if is_frozen(history, adef["target"]):
                    continue
                if grp and is_in_dead_zone(control_state, grp, adef["target"], aspace):
                    continue
                print(f"  [preset] Color '{color_diag}' -> '{action}'")
                return action

        # 3. physics minor -> object/scene preset (only minor, major is rejected upstream)
        physics_diag = review.get("physics_consistency", "good")
        if physics_diag != "good":
            physics_map = aspace.get("physics_issue_to_action", {})
            physics_actions = physics_map.get(physics_diag, []) or physics_map.get("minor_issue", [])
            for action in physics_actions:
                if not _is_allowed(action):
                    continue
                if action in action_blacklist:
                    continue
                grp, adef = _find_action(action, aspace)
                if adef is None:
                    continue
                if is_frozen(history, adef["target"]):
                    continue
                if grp and is_in_dead_zone(control_state, grp, adef["target"], aspace):
                    continue
                print(f"  [preset] Physics minor_issue -> '{action}'")
                return action

        # 4. issue_tags fallback (same as legacy but without VLM suggestions)
        fallback_map = aspace.get("issue_to_action_fallback", {})
        for tag in issue_tags:
            candidates = fallback_map.get(tag, [])
            if "__MESH_REJECT__" in candidates:
                return "__MESH_REJECT__"
            for action in candidates:
                if not _is_allowed(action):
                    continue
                if action in action_blacklist:
                    continue
                # Check compound
                if action in aspace.get("compound_actions", {}):
                    cdef = aspace["compound_actions"][action]
                    if not _compound_allowed(action, control_state, aspace):
                        continue
                    viable = any(
                        _find_action(s, aspace)[1] is not None
                        and not is_frozen(history, _find_action(s, aspace)[1]["target"])
                        for s in cdef["sub_actions"]
                        if _find_action(s, aspace)[1] is not None
                    )
                    if viable:
                        print(f"  [preset] Fallback tag '{tag}' -> compound '{action}'")
                        return action
                    continue
                # Atomic
                grp, adef = _find_action(action, aspace)
                if adef is None:
                    continue
                if is_frozen(history, adef["target"]):
                    continue
                if grp and is_in_dead_zone(control_state, grp, adef["target"], aspace):
                    continue
                print(f"  [preset] Fallback tag '{tag}' -> '{action}'")
                return action

        print(f"  [preset] No applicable preset action, returning NO_OP")
        return "NO_OP"

    # ─── Legacy mode (preset_mode=False) ─────────────────────────────────
    group_actions = set(aspace["groups"].get(active_group, {}).get("actions", {}).keys())
    compound_actions = aspace.get("compound_actions", {})

    # Check VLM suggestions (atomic first, then compound)
    for action in review.get("suggested_actions", []):
        if not _is_allowed(action):
            continue
        if action in action_blacklist:
            print(f"  [select] Suggested '{action}' blacklisted, skip")
            continue
        if action in group_actions:
            adef = aspace["groups"][active_group]["actions"][action]
            target = adef["target"]
            if is_frozen(history, target):
                print(f"  [select] Suggested '{action}' -> target frozen, skip")
                continue
            if is_in_dead_zone(control_state, active_group, target, aspace):
                continue
            # Fix 3: Conditionally promote to compound
            promoted = _promote_to_compound(action, active_group, control_state, history, aspace, issue_tags)
            if promoted != action and promoted not in action_blacklist and _compound_allowed(promoted, control_state, aspace):
                print(f"  [select] Promoted '{action}' -> '{promoted}'")
                return promoted
            print(f"  [select] Using VLM suggestion: {action}")
            return action
        if action in compound_actions:
            cdef = compound_actions[action]
            if cdef.get("group") == active_group:
                viable = any(
                    _find_action(s, aspace)[1] is not None
                    and not is_frozen(history, _find_action(s, aspace)[1]["target"])
                    for s in cdef["sub_actions"]
                    if _find_action(s, aspace)[1] is not None
                )
                if viable and _compound_allowed(action, control_state, aspace):
                    print(f"  [select] Using VLM compound suggestion: {action}")
                    return action

    # Fallback via issue_tags
    fallback_map = aspace.get("issue_to_action_fallback", {})
    for tag in review.get("issue_tags", []):
        candidates = fallback_map.get(tag, [])
        if "__MESH_REJECT__" in candidates:
            print(f"  [select] Tag '{tag}' -> __MESH_REJECT__ sentinel triggered")
            return "__MESH_REJECT__"
        for action in candidates:
            if not _is_allowed(action):
                continue
            if action in action_blacklist:
                continue
            if action in group_actions:
                adef = aspace["groups"][active_group]["actions"][action]
                target = adef["target"]
                if is_frozen(history, target):
                    continue
                if is_in_dead_zone(control_state, active_group, target, aspace):
                    continue
                # Fix 3: Conditionally promote to compound
                promoted = _promote_to_compound(action, active_group, control_state, history, aspace, issue_tags)
                if promoted != action and promoted not in action_blacklist and _compound_allowed(promoted, control_state, aspace):
                    print(f"  [select] Fallback promoted '{action}' -> '{promoted}'")
                    return promoted
                print(f"  [select] Fallback from tag '{tag}': {action}")
                return action
            if action in compound_actions:
                cdef = compound_actions[action]
                if cdef.get("group") != active_group:
                    continue
                viable = any(
                    _find_action(s, aspace)[1] is not None
                    and not is_frozen(history, _find_action(s, aspace)[1]["target"])
                    and not is_in_dead_zone(
                        control_state,
                        _find_action(s, aspace)[0] or active_group,
                        _find_action(s, aspace)[1]["target"],
                        aspace,
                    )
                    for s in cdef["sub_actions"]
                    if _find_action(s, aspace)[1] is not None
                )
                if viable and _compound_allowed(action, control_state, aspace):
                    print(f"  [select] Fallback compound from tag '{tag}': {action}")
                    return action

    print(f"  [select] No applicable action for group '{active_group}', returning NO_OP")
    return "NO_OP"


# ─────────────────────────────────────────────────────────────────────────────
# Apply action — always returns ActionResult
# ─────────────────────────────────────────────────────────────────────────────

def apply_action(action: str, control_state: dict, aspace: dict,
                 step_scale: float = 1.0) -> ActionResult:
    """
    Apply action to control_state with step_scale attenuation.
    Always returns ActionResult.
    """
    state = copy.deepcopy(control_state)

    if action == "NO_OP":
        return ActionResult(state=state, target=None, delta_sign=0, applied=False, action=action)

    # Find group and action def
    group_found = None
    adef = None
    for grp, grp_data in aspace["groups"].items():
        if action in grp_data["actions"]:
            group_found = grp
            adef = grp_data["actions"][action]
            break

    if adef is None:
        print(f"  [apply] Unknown action '{action}', no-op")
        return ActionResult(state=state, target=None, delta_sign=0, applied=False, action=action)

    group = group_found
    target = adef["target"]
    delta_type = adef["delta_type"]
    raw_delta = adef["delta"]
    lo, hi = adef["bounds"]

    # Ensure group dict exists
    if group not in state:
        state[group] = {}

    # Default initial value
    default = aspace["default_control_state"].get(group, {}).get(target)
    if default is None:
        default = 1.0 if delta_type == "mul" else 0.0

    current = state[group].get(target, default)

    # Attenuate delta
    if delta_type == "mul":
        effective_delta = (raw_delta - 1.0) * step_scale + 1.0
        new_val = current * effective_delta
    else:  # add
        effective_delta = raw_delta * step_scale
        new_val = current + effective_delta

    # Clamp to bounds
    new_val = max(lo, min(hi, new_val))
    state[group][target] = round(new_val, 6)

    delta_sign = 1 if (new_val - current) > 0 else (-1 if (new_val - current) < 0 else 0)
    print(f"  [apply] {action}: {group}.{target}: {current:.4f} -> {new_val:.4f} "
          f"(step_scale={step_scale:.2f}, sign={delta_sign:+d})")

    return ActionResult(state=state, target=target, delta_sign=delta_sign, applied=True, action=action)


# ─────────────────────────────────────────────────────────────────────────────
# Step scale schedule (score-adaptive)
# ─────────────────────────────────────────────────────────────────────────────

def get_step_scale(round_idx: int, hybrid_score: float = 0.0) -> float:
    """Round 0: 100%, Round 1: 70%, Round 2+: 50%. Score-adaptive: if score<0.65 boost by 1.2x."""
    base = {0: 1.0, 1: 0.7, 2: 0.5}.get(round_idx, 0.4)
    if 0 < hybrid_score < 0.65:
        base *= 1.2
    return min(base, 1.2)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def apply_feedback(review_json_path: str, control_state_path: str,
                   output_control_state_path: str, round_idx: int = 0,
                   history_json_path: Optional[str] = None,
                   hybrid_score: float = 0.0,
                   action_blacklist: set = None,
                   preset_mode: bool = False,
                   active_group_override: str = None,
                   action_space_path: Optional[str] = None,
                   action_whitelist: list = None) -> dict:
    """
    Load review + control_state, select action, apply, save new control_state.
    Returns dict with action taken and new control_state.

    v6: Added preset_mode and active_group_override parameters.
    """
    aspace = load_action_space(action_space_path)

    with open(review_json_path) as f:
        review = json.load(f)

    control_state = load_control_state(control_state_path, aspace)
    history = load_history(history_json_path)

    # v6: Allow caller to override active_group (used in bounded search)
    if active_group_override:
        active_group = active_group_override
    else:
        active_group = review.get("active_group", GROUP_ORDER[round_idx % len(GROUP_ORDER)])
    step_scale   = get_step_scale(round_idx, hybrid_score)

    action = select_action(review, active_group, control_state, history, aspace,
                          action_blacklist=action_blacklist,
                          preset_mode=preset_mode,
                          action_whitelist=action_whitelist)

    is_compound = action in aspace.get("compound_actions", {})
    if is_compound:
        result = _apply_compound(action, control_state, aspace, history, step_scale)
        new_state = result.state
        for sr in (result.sub_results or []):
            if sr.target and sr.delta_sign != 0 and sr.applied:
                history = update_history(history, sr.target, sr.delta_sign)
        if history_json_path:
            save_history(history, history_json_path)
    else:
        result = apply_action(action, control_state, aspace, step_scale)
        new_state  = result.state
        target     = result.target
        delta_sign = result.delta_sign
        if target is not None and delta_sign != 0:
            history = update_history(history, target, delta_sign)
            if history_json_path:
                save_history(history, history_json_path)

    save_control_state(new_state, output_control_state_path)
    print(f"  [feedback] action={action}, new_state saved to {output_control_state_path}")

    return {
        "action_taken":      action,
        "active_group":      active_group,
        "round_idx":         round_idx,
        "step_scale":        step_scale,
        "action_whitelist":  action_whitelist,
        "control_state":     new_state,
        "sub_results":       result.sub_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stage 5.6: Apply VLM feedback to control_state")
    p.add_argument("--review-json",           required=True)
    p.add_argument("--control-state",         required=True)
    p.add_argument("--output-control-state",  required=True)
    p.add_argument("--round-idx",   type=int, default=0)
    p.add_argument("--history-json", default=None)
    p.add_argument("--action-space-path", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = apply_feedback(
        review_json_path=args.review_json,
        control_state_path=args.control_state,
        output_control_state_path=args.output_control_state,
        round_idx=args.round_idx,
        history_json_path=args.history_json,
        action_space_path=args.action_space_path,
    )
    print(f"\n=== Feedback applied: action={result['action_taken']}, "
          f"group={result['active_group']}, step_scale={result['step_scale']} ===")
