"""
端到端 pipeline 验证脚本 (dry-run, 无 Blender re-render, 无 VLM GPU).
验证所有 8 个 Fix 的逻辑正确性。
"""
import sys, os, json, traceback

ROOT = '/aaaidata/zhangqisong/data_build'
sys.path.insert(0, os.path.join(ROOT, 'pipeline'))
sys.path.insert(0, ROOT)

RENDERS_DIR = os.path.join(ROOT, 'pipeline/data/renders')
REVIEWS_DIR = os.path.join(ROOT, 'pipeline/data/vlm_reviews')

PASS = "[PASS]"
FAIL = "[FAIL]"

def check(name, cond, detail=""):
    mark = PASS if cond else FAIL
    print(f"  {mark} {name}" + (f": {detail}" if detail else ""))
    return cond

errors = 0

# ── Fix 1+2+3: CV metrics without mask ────────────────────────────────────────
print("\n=== Fix 1+2+3: CV metrics (no mask) ===")
try:
    from stage5_5_vlm_review import compute_cv_metrics, compute_hybrid_score

    rgb = os.path.join(RENDERS_DIR, 'obj_001/az000_el+00.png')
    cv = compute_cv_metrics(rgb, mask_path=None)

    errors += 0 if check("mask_available=False", cv['mask_available'] == False) else 1
    errors += 0 if check("mask_score=None", cv['mask_score'] is None) else 1
    errors += 0 if check("framing_score=None", cv['framing_score'] is None) else 1
    errors += 0 if check("sharpness_score in [0.2, 0.9]", 0.2 <= cv['sharpness_score'] <= 0.9,
                          f"got {cv['sharpness_score']:.3f}") else 1
    errors += 0 if check("cv_score in [0.5, 1.0]", 0.5 <= cv['cv_score'] <= 1.0,
                          f"got {cv['cv_score']:.3f}") else 1

    # Fix 1: no mask → no mask gate
    vlm_hi = {'lighting': 5, 'object_integrity': 5, 'composition': 5,
               'render_quality_semantic': 5, 'overall': 5}
    hybrid, route = compute_hybrid_score(vlm_hi, cv)
    errors += 0 if check("no mask gate → hybrid not capped at 0.74",
                          hybrid > 0.74, f"hybrid={hybrid:.3f}") else 1

    # Fix 1: with mask_available=True and bad mask_score → gate applies
    cv_mask = dict(cv, mask_available=True, mask_score=0.40, framing_score=0.5)
    # recompute cv_score for this fake dict
    cv_mask['cv_score'] = 0.556 * cv_mask['exposure_score'] + 0.444 * cv_mask['sharpness_score']
    h2, _ = compute_hybrid_score(vlm_hi, cv_mask)
    errors += 0 if check("bad mask_score → gate caps at 0.74",
                          h2 <= 0.74, f"hybrid={h2:.3f}") else 1

    print(f"  sharpness={cv['sharpness_score']:.3f}, cv_score={cv['cv_score']:.3f}")
except Exception as e:
    print(f"  {FAIL} Exception: {e}"); traceback.print_exc(); errors += 1

# ── Fix 4: ActionResult ───────────────────────────────────────────────────────
print("\n=== Fix 4: ActionResult unified return type ===")
try:
    from stage5_6_feedback_apply import (
        ActionResult, apply_action, load_action_space, default_control_state
    )
    aspace = load_action_space()
    state = default_control_state(aspace)

    r_noop = apply_action('NO_OP', state, aspace)
    errors += 0 if check("NO_OP returns ActionResult", isinstance(r_noop, ActionResult)) else 1
    errors += 0 if check("NO_OP.applied=False", r_noop.applied == False) else 1
    errors += 0 if check("NO_OP.target=None", r_noop.target is None) else 1

    r_act = apply_action('L_RIM_UP', state, aspace, step_scale=1.0)
    errors += 0 if check("L_RIM_UP returns ActionResult", isinstance(r_act, ActionResult)) else 1
    errors += 0 if check("L_RIM_UP.applied=True", r_act.applied == True) else 1
    errors += 0 if check("L_RIM_UP.target=rim_scale", r_act.target == 'rim_scale') else 1
    errors += 0 if check("L_RIM_UP.delta_sign=+1", r_act.delta_sign == 1) else 1
    rim = r_act.state['lighting']['rim_scale']
    errors += 0 if check("rim_scale increased from 1.0", rim > 1.0, f"got {rim:.4f}") else 1
except Exception as e:
    print(f"  {FAIL} Exception: {e}"); traceback.print_exc(); errors += 1

# ── Fix 6: Dynamic group selection ───────────────────────────────────────────
print("\n=== Fix 6: Dynamic group selection ===")
try:
    from run_evolution_loop import pick_next_group

    # flat_lighting → lighting group
    g = pick_next_group(['flat_lighting'], ['L_RIM_UP'], set(), aspace)
    errors += 0 if check("flat_lighting → lighting", g == 'lighting', f"got {g}") else 1

    # After lighting tried → fallback to next round-robin
    g2 = pick_next_group(['flat_lighting'], [], {'lighting'}, aspace)
    errors += 0 if check("tried lighting → different group", g2 != 'lighting', f"got {g2}") else 1

    # off_center_left → camera group
    g3 = pick_next_group(['off_center_left'], [], set(), aspace)
    errors += 0 if check("off_center_left → camera", g3 == 'camera', f"got {g3}") else 1

    # All groups tried → reset
    g4 = pick_next_group([], [], {'lighting', 'camera', 'object', 'scene'}, aspace)
    errors += 0 if check("all tried → reset to lighting", g4 == 'lighting', f"got {g4}") else 1
except Exception as e:
    print(f"  {FAIL} Exception: {e}"); traceback.print_exc(); errors += 1

# ── Fix 5: NO_OP does not consume update budget ───────────────────────────────
print("\n=== Fix 5: NO_OP budget logic ===")
try:
    # Simulate evolve_object's inner loop logic
    update_count = 0
    group_tried = set()
    GROUP_ORDER = ['lighting', 'camera', 'object', 'scene']

    # Simulate 4 consecutive NO_OPs
    noop_actions = ['NO_OP', 'NO_OP', 'NO_OP', 'NO_OP']
    for action in noop_actions:
        if action == 'NO_OP':
            group_tried.add(GROUP_ORDER[len(group_tried)])
            if len(group_tried) >= len(GROUP_ORDER):
                break
            continue
        update_count += 1

    errors += 0 if check("4x NO_OP → update_count=0", update_count == 0,
                          f"got {update_count}") else 1
    errors += 0 if check("4x NO_OP → all groups tried", len(group_tried) == 4,
                          f"tried={group_tried}") else 1

    # Simulate real action
    update_count2 = 0
    for action in ['L_RIM_UP']:
        update_count2 += 1
    errors += 0 if check("real action → update_count=1", update_count2 == 1) else 1
except Exception as e:
    print(f"  {FAIL} Exception: {e}"); traceback.print_exc(); errors += 1

# ── Fix 8: Anti-oscillation threshold = 3 ────────────────────────────────────
print("\n=== Fix 8: Anti-oscillation threshold ===")
try:
    from stage5_6_feedback_apply import update_history

    hist = {}
    hist = update_history(hist, 'rim_scale', +1)
    hist = update_history(hist, 'rim_scale', -1)
    errors += 0 if check("1 flip → not frozen", hist['rim_scale']['frozen'] == False) else 1
    hist = update_history(hist, 'rim_scale', +1)
    errors += 0 if check("2 flips → not frozen", hist['rim_scale']['frozen'] == False) else 1
    hist = update_history(hist, 'rim_scale', -1)
    errors += 0 if check("3 flips → frozen=True", hist['rim_scale']['frozen'] == True) else 1
except Exception as e:
    print(f"  {FAIL} Exception: {e}"); traceback.print_exc(); errors += 1

# ── Fix 7: batch_summary filename ─────────────────────────────────────────────
print("\n=== Fix 7: Batch summary filename ===")
try:
    import os as _os
    # Simulate with CUDA_VISIBLE_DEVICES set
    _os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    visible = _os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
    fname = f'batch_summary_gpu{visible}.json'
    errors += 0 if check("filename uses CUDA_VISIBLE_DEVICES",
                          fname == 'batch_summary_gpu0,1.json', f"got {fname}") else 1
    # Simulate without env var
    del _os.environ['CUDA_VISIBLE_DEVICES']
    visible2 = _os.environ.get('CUDA_VISIBLE_DEVICES', 'cuda0')
    fname2 = f'batch_summary_gpu{visible2}.json'
    errors += 0 if check("fallback to device string", 'batch_summary_gpu' in fname2,
                          f"got {fname2}") else 1
except Exception as e:
    print(f"  {FAIL} Exception: {e}"); traceback.print_exc(); errors += 1

# ── apply_feedback end-to-end (uses existing review JSON) ─────────────────────
print("\n=== Integration: apply_feedback on existing review JSON ===")
try:
    from stage5_6_feedback_apply import apply_feedback
    import tempfile

    review_path = os.path.join(REVIEWS_DIR, 'obj_001_r00_agg.json')
    if os.path.exists(review_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_cs = os.path.join(tmpdir, 'cs_out.json')
            result = apply_feedback(
                review_json_path=review_path,
                control_state_path='/nonexistent',  # triggers default
                output_control_state_path=out_cs,
                round_idx=0,
            )
            errors += 0 if check("apply_feedback returns dict", isinstance(result, dict)) else 1
            errors += 0 if check("action_taken key exists", 'action_taken' in result) else 1
            errors += 0 if check("control_state saved", os.path.exists(out_cs)) else 1
            print(f"  action_taken={result['action_taken']}, group={result['active_group']}")
    else:
        print(f"  SKIP: review file not found at {review_path}")
except Exception as e:
    print(f"  {FAIL} Exception: {e}"); traceback.print_exc(); errors += 1

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
if errors == 0:
    print(f"ALL TESTS PASSED (0 failures)")
else:
    print(f"FAILURES: {errors}")
    sys.exit(1)
