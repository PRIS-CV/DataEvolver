#!/usr/bin/env python3
"""Aggregate v5 full results vs v4b and v2"""
import json, os

ROOT = "pipeline/data/evolution_v5"
V2   = "pipeline/data/evolution_v2"
V4B  = "pipeline/data/evolution_v4b_full"

header = f"{'obj':<10} {'v5':>7} {'v4b':>7} {'v2':>7} {'dv5-v2':>9} {'dv5-v4b':>9}  {'exit':<14} actions"
print(header)
print("-" * 90)

v5_scores = []
v2_scores = []
rows = []

for i in range(1, 11):
    obj = f"obj_{i:03d}"
    v5s = v4bs = v2s = 0.0
    exit_r = "?"
    acts = []

    for label, root in [("v5", ROOT), ("v4b", V4B), ("v2", V2)]:
        fp = os.path.join(root, obj, "evolution_result.json")
        if os.path.exists(fp):
            with open(fp) as f:
                r = json.load(f)
            val = r.get("final_hybrid", 0)
            if label == "v5":
                v5s = val
                exit_r = r.get("exit_reason", "?")
                acts = [e.get("action_taken") for e in r.get("state_log", [])]
            elif label == "v4b":
                v4bs = val
            else:
                v2s = val

    d1 = v5s - v2s
    d2 = v5s - v4bs
    s1 = "+" if d1 >= 0 else ""
    s2 = "+" if d2 >= 0 else ""

    # dedupe actions
    short = []
    prev = None
    for a in acts:
        if a != prev:
            short.append(a or "None")
            prev = a

    print(f"{obj:<10} {v5s:>7.4f} {v4bs:>7.4f} {v2s:>7.4f} {s1}{d1:.4f}{'':>{8-len(s1+f'{d1:.4f}')}} {s2}{d2:.4f}{'':>{8-len(s2+f'{d2:.4f}')}}  {exit_r:<14} {short}")

    if v5s > 0:
        v5_scores.append(v5s)
    if v2s > 0:
        v2_scores.append(v2s)

print()
if v5_scores:
    avg5 = sum(v5_scores) / len(v5_scores)
    print(f"avg v5  (excl hard_fail) = {avg5:.4f}  ({len(v5_scores)} objects)")
if v2_scores:
    avg2 = sum(v2_scores) / len(v2_scores)
    print(f"avg v2  (excl hard_fail) = {avg2:.4f}  ({len(v2_scores)} objects)")
if v5_scores and v2_scores:
    print(f"delta avg = {avg5 - avg2:+.4f}")

total = len(v5_scores) + (10 - len(v5_scores))  # include hard_fail
n80 = sum(1 for s in v5_scores if s >= 0.80)
n78 = sum(1 for s in v5_scores if s >= 0.78)
n76 = sum(1 for s in v5_scores if s >= 0.76)
print(f"Accepted @0.80: {n80}/10")
print(f"Accepted @0.78: {n78}/10")
print(f"Accepted @0.76: {n76}/10")

# Count preserved exits
preserved = 0
for i in range(1, 11):
    obj = f"obj_{i:03d}"
    fp = os.path.join(ROOT, obj, "evolution_result.json")
    if os.path.exists(fp):
        with open(fp) as f:
            r = json.load(f)
        if r.get("exit_reason") == "preserved":
            preserved += 1
print(f"Preserved exits: {preserved}/10")
