#!/usr/bin/env python3
"""
Aggregate v4b_full results vs v2 baseline.
Prints per-object table + summary stats (excl hard_fail).
"""
import json, os

ROOT     = "/aaaidata/zhangqisong/data_build/pipeline/data/evolution_v4b_full"
V2_ROOT  = "/aaaidata/zhangqisong/data_build/pipeline/data/evolution_v2"
V4_ROOT  = "/aaaidata/zhangqisong/data_build/pipeline/data/evolution_v4"
ALL_OBJS = [f"obj_{i:03d}" for i in range(1, 11)]

rows = []
for obj in ALL_OBJS:
    for label, root in [("v4b", ROOT), ("v4", V4_ROOT), ("v2", V2_ROOT)]:
        fpath = os.path.join(root, obj, "evolution_result.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            r = json.load(f)
        if label == "v4b":
            rows.append({
                "obj": obj,
                "v4b": r.get("final_hybrid", 0),
                "accepted": r.get("accepted", False),
                "exit": r.get("exit_reason", "?"),
                "probes": r.get("probes_run", 0),
                "actions": [e.get("action_taken") for e in r.get("state_log", [])],
                "scores": [e.get("hybrid_score", 0) for e in r.get("state_log", [])],
            })

# Load v2 and v4 baselines
v2 = {}
v4 = {}
for obj in ALL_OBJS:
    for target, root in [(v2, V2_ROOT), (v4, V4_ROOT)]:
        fpath = os.path.join(root, obj, "evolution_result.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                r = json.load(f)
            target[obj] = r.get("final_hybrid", 0)

# Table header
print(f"=== evolution_v4b_full  ({len(rows)}/10 done) ===\n")
print(f"{'obj':<10} {'v4b':>7} {'v4':>7} {'v2':>7}  {'Δv4b-v2':>9}  {'acc':<5} {'exit':<18} actions")
print("-" * 85)

hard_fails_v4b = []
excl_rows = []
for r in rows:
    obj = r["obj"]
    fh  = r["v4b"]
    fv4 = v4.get(obj, 0)
    fv2 = v2.get(obj, 0)
    d   = fh - fv2
    sign = "+" if d >= 0 else ""
    acc_str = "YES" if r["accepted"] else "no"
    # Deduplicate action list
    actions_short = []
    prev = None
    for a in r["actions"]:
        if a != prev:
            actions_short.append(a or "NO_OP")
            prev = a
    print(f"{obj:<10} {fh:>7.4f} {fv4:>7.4f} {fv2:>7.4f}  {sign+f'{d:.4f}':>9}  {acc_str:<5} {r['exit']:<18} {actions_short}")
    if r["exit"] == "render_hard_fail" or fh == 0:
        hard_fails_v4b.append(obj)
    else:
        excl_rows.append((obj, fh, fv4, fv2))

# Summary stats (excluding hard fails)
if excl_rows:
    v4b_vals  = [x[1] for x in excl_rows]
    v4_vals   = [x[2] for x in excl_rows if x[2] > 0]
    v2_vals   = [x[3] for x in excl_rows if x[3] > 0]
    deltas    = [x[1] - x[3] for x in excl_rows if x[3] > 0]
    n_better  = sum(1 for d in deltas if d > 0)
    n_worse   = sum(1 for d in deltas if d < -0.01)
    n_flat    = len(deltas) - n_better - n_worse

    print(f"\n--- Summary (excl {hard_fails_v4b}) ---")
    print(f"avg v4b  = {sum(v4b_vals)/len(v4b_vals):.4f}")
    if v4_vals:
        print(f"avg v4   = {sum(v4_vals)/len(v4_vals):.4f}")
    if v2_vals:
        print(f"avg v2   = {sum(v2_vals)/len(v2_vals):.4f}")
    print(f"Δ(v4b-v2): {n_better} better / {n_flat} flat / {n_worse} worse")
    if deltas:
        print(f"avg delta = {sum(deltas)/len(deltas):+.4f}")

    accepted_v4b = sum(1 for r in rows if r["accepted"])
    print(f"Accepted @0.80: {accepted_v4b}/{len(rows)}")
    accepted_078 = sum(1 for r in rows if r["v4b"] >= 0.78 and r["exit"] != "render_hard_fail")
    print(f"Accepted @0.78: {accepted_078}/{len(rows)}")
