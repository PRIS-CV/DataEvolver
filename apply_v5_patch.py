#!/usr/bin/env python3
"""
v5 patch script — applies all 4 changes to the data_build codebase.
Run on server: python3 apply_v5_patch.py
"""
import json
import os
import re
import shutil
from datetime import datetime

ROOT = "/aaaidata/zhangqisong/data_build"
BACKUP_SUFFIX = f".bak_v4b_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def backup(path):
    if os.path.exists(path):
        shutil.copy2(path, path + BACKUP_SUFFIX)
        print(f"  backup: {path}{BACKUP_SUFFIX}")

# ═══════════════════════════════════════════════════════════════════════
# PATCH 1: configs/action_space.json
# ═══════════════════════════════════════════════════════════════════════
def patch_action_space():
    path = os.path.join(ROOT, "configs", "action_space.json")
    backup(path)
    with open(path) as f:
        data = json.load(f)

    # 1a. Change flat_lighting fallback: remove L_RIM_FILL_UP, add L_KEY_SHADOW_SOFT
    fb = data.get("issue_to_action_fallback", {})
    fb["flat_lighting"] = ["L_RIM_UP", "L_FILL_DOWN", "L_KEY_SHADOW_SOFT"]
    print("  [1a] flat_lighting fallback -> ['L_RIM_UP', 'L_FILL_DOWN', 'L_KEY_SHADOW_SOFT']")

    # 1b. Add compound_guards with numerical max_state
    data["compound_guards"] = {
        "L_RIM_FILL_UP": {
            "max_state": {
                "rim_scale": 1.25,
                "fill_scale": 1.15
            }
        },
        "L_KEY_FILL_UP": {
            "max_state": {
                "key_scale": 1.44,
                "fill_scale": 1.15
            }
        }
    }
    print("  [1b] Added compound_guards (rim_scale<=1.25, fill_scale<=1.15)")

    # 1c. Add preserve_score_threshold
    data["preserve_score_threshold"] = 0.76
    print("  [1c] Added preserve_score_threshold = 0.76")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  DONE: {path}\n")

# ═══════════════════════════════════════════════════════════════════════
# PATCH 2: pipeline/stage5_6_feedback_apply.py
# ═══════════════════════════════════════════════════════════════════════
def patch_feedback_apply():
    path = os.path.join(ROOT, "pipeline", "stage5_6_feedback_apply.py")
    backup(path)
    with open(path) as f:
        content = f.read()

    # 2a. Narrow PROMOTE_CONTEXT: remove flat_lighting from L_RIM_UP
    old_promote = '''PROMOTE_CONTEXT = {
    "L_RIM_UP":  {"flat_lighting", "weak_subject_separation"},
    "L_KEY_UP":  {"underexposed"},
    "L_FILL_UP": {"flat_lighting", "underexposed"},
}'''
    new_promote = '''PROMOTE_CONTEXT = {
    "L_RIM_UP":  {"weak_subject_separation"},
    "L_KEY_UP":  {"underexposed"},
    "L_FILL_UP": {"underexposed"},
}'''
    if old_promote in content:
        content = content.replace(old_promote, new_promote)
        print("  [2a] PROMOTE_CONTEXT: removed flat_lighting from L_RIM_UP and L_FILL_UP")
    else:
        print("  [2a] WARNING: PROMOTE_CONTEXT pattern not found, trying relaxed match")
        content = content.replace(
            '"L_RIM_UP":  {"flat_lighting", "weak_subject_separation"}',
            '"L_RIM_UP":  {"weak_subject_separation"}'
        )
        content = content.replace(
            '"L_FILL_UP": {"flat_lighting", "underexposed"}',
            '"L_FILL_UP": {"underexposed"}'
        )

    # 2b. Add _compound_allowed() function right before select_action
    compound_allowed_func = '''
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


'''
    # Insert before select_action definition
    select_anchor = "def select_action(review: dict, active_group: str, control_state: dict,"
    if select_anchor in content:
        content = content.replace(select_anchor, compound_allowed_func + select_anchor)
        print("  [2b] Added _compound_allowed() function")
    else:
        print("  [2b] ERROR: select_action anchor not found!")

    # 2c. Add preserve_score_threshold at top of select_action body
    old_blacklist_init = '''    if action_blacklist is None:
        action_blacklist = set()'''
    new_blacklist_init = '''    if action_blacklist is None:
        action_blacklist = set()

    # v5: High-score preservation — don't risk degrading a good state
    _preserve_th = float(aspace.get("preserve_score_threshold", 0.76))
    _current_score = review.get("hybrid_score", 0.0)
    if _current_score >= _preserve_th:
        print(f"  [select] Score {_current_score:.4f} >= {_preserve_th} → NO_OP (preserve)")
        return "NO_OP"'''
    if old_blacklist_init in content:
        content = content.replace(old_blacklist_init, new_blacklist_init, 1)
        print("  [2c] Added preserve_score_threshold logic at top of select_action")
    else:
        print("  [2c] ERROR: blacklist init anchor not found!")

    # 2d. Add compound_guard checks at all 4 compound return points
    # Point 1: VLM promoted compound
    old_promoted_vlm = '''            promoted = _promote_to_compound(action, active_group, control_state, history, aspace, issue_tags)
            if promoted != action and promoted not in action_blacklist:
                print(f"  [select] Promoted \'{action}\' -> \'{promoted}\'")
                return promoted
            print(f"  [select] Using VLM suggestion: {action}")
            return action'''
    new_promoted_vlm = '''            promoted = _promote_to_compound(action, active_group, control_state, history, aspace, issue_tags)
            if promoted != action and promoted not in action_blacklist and _compound_allowed(promoted, control_state, aspace):
                print(f"  [select] Promoted \'{action}\' -> \'{promoted}\'")
                return promoted
            print(f"  [select] Using VLM suggestion: {action}")
            return action'''
    content = content.replace(old_promoted_vlm, new_promoted_vlm, 1)

    # Point 2: VLM compound suggestion
    old_vlm_compound = '''                if viable:
                    print(f"  [select] Using VLM compound suggestion: {action}")
                    return action'''
    new_vlm_compound = '''                if viable and _compound_allowed(action, control_state, aspace):
                    print(f"  [select] Using VLM compound suggestion: {action}")
                    return action'''
    content = content.replace(old_vlm_compound, new_vlm_compound, 1)

    # Point 3: Fallback promoted compound
    old_promoted_fb = '''                promoted = _promote_to_compound(action, active_group, control_state, history, aspace, issue_tags)
                if promoted != action and promoted not in action_blacklist:
                    print(f"  [select] Fallback promoted \'{action}\' -> \'{promoted}\'")
                    return promoted'''
    new_promoted_fb = '''                promoted = _promote_to_compound(action, active_group, control_state, history, aspace, issue_tags)
                if promoted != action and promoted not in action_blacklist and _compound_allowed(promoted, control_state, aspace):
                    print(f"  [select] Fallback promoted \'{action}\' -> \'{promoted}\'")
                    return promoted'''
    content = content.replace(old_promoted_fb, new_promoted_fb, 1)

    # Point 4: Fallback compound from tag
    old_fb_compound = '''                if viable:
                    print(f"  [select] Fallback compound from tag \'{tag}\': {action}")
                    return action'''
    new_fb_compound = '''                if viable and _compound_allowed(action, control_state, aspace):
                    print(f"  [select] Fallback compound from tag \'{tag}\': {action}")
                    return action'''
    content = content.replace(old_fb_compound, new_fb_compound, 1)

    print("  [2d] Added _compound_allowed guards at all 4 compound return paths")

    with open(path, "w") as f:
        f.write(content)
    print(f"  DONE: {path}\n")

# ═══════════════════════════════════════════════════════════════════════
# PATCH 3: run_evolution_loop.py — add PIPELINE_STATE.json persistence
# ═══════════════════════════════════════════════════════════════════════
def patch_evolution_loop():
    path = os.path.join(ROOT, "run_evolution_loop.py")
    backup(path)
    with open(path) as f:
        content = f.read()

    # Add PIPELINE_STATE helper functions after imports
    state_helpers = '''
# ─── v5: PIPELINE_STATE.json persistence ────────────────────────────────────
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

# ─── end v5 persistence ────────────────────────────────────────────────────

'''

    # Insert after "import json" line (or near top after imports)
    # Find the last import block
    anchor = "from typing import"
    if anchor not in content:
        anchor = "import json"
    idx = content.find(anchor)
    if idx >= 0:
        # Find end of that line
        eol = content.index("\n", idx)
        content = content[:eol+1] + state_helpers + content[eol+1:]
        print("  [3a] Added PIPELINE_STATE helper functions")
    else:
        print("  [3a] ERROR: import anchor not found!")

    # Now we need to add _init + _update calls in the main loop.
    # Find where the main object loop starts and where results are saved
    # We'll add init before the loop and update after each object's result is saved.

    # Look for the pattern where evolution_result.json is written
    result_save_pattern = 'json.dump(result, f'
    if result_save_pattern in content:
        # Add _update_obj_state call right after the result JSON dump block
        old_dump = '''        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)'''
        new_dump = '''        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        # v5: Update PIPELINE_STATE.json
        if "_pstate" in dir():
            _update_obj_state(_pstate, obj_id, result)'''
        if old_dump in content:
            content = content.replace(old_dump, new_dump, 1)
            print("  [3b] Added _update_obj_state call after result save")
        else:
            print("  [3b] WARNING: result dump pattern not found exactly, trying alternate")

    # Add _init call near the start of main/run function
    # Look for where obj_ids iteration begins
    for_obj_pattern = "for obj_id in obj_ids:"
    if for_obj_pattern in content:
        content = content.replace(
            for_obj_pattern,
            "# v5: Initialize pipeline state\n    _pstate = _init_pipeline_state(obj_ids, output_dir)\n    _save_pipeline_state(_pstate)\n\n    " + for_obj_pattern,
            1
        )
        print("  [3c] Added _init_pipeline_state before main loop")
    else:
        print("  [3c] WARNING: for obj_id loop not found")

    with open(path, "w") as f:
        f.write(content)
    print(f"  DONE: {path}\n")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("v5 Patch — Lighting Sprint v5")
    print("=" * 60)
    print()

    print("[PATCH 1] configs/action_space.json")
    patch_action_space()

    print("[PATCH 2] pipeline/stage5_6_feedback_apply.py")
    patch_feedback_apply()

    print("[PATCH 3] run_evolution_loop.py")
    patch_evolution_loop()

    print("=" * 60)
    print("All patches applied. Run verification:")
    print("  python3 -c \"from pipeline.stage5_6_feedback_apply import select_action; print('import OK')\"")
    print("  python3 -c \"import run_evolution_loop; print('import OK')\"")
    print("=" * 60)
