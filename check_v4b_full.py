#!/usr/bin/env python3
"""Check evolution_v4b_full status vs v2 baseline"""
import json, os

ROOT = "/aaaidata/zhangqisong/data_build/pipeline/data/evolution_v4b_full"
V2_ROOT = "/aaaidata/zhangqisong/data_build/pipeline/data/evolution_v2"

all_objs = [f"obj_{i:03d}" for i in range(1, 11)]
done, not_done = [], []

for obj in all_objs:
    fpath = os.path.join(ROOT, obj, "evolution_result.json")
    if not os.path.exists(fpath):
        not_done.append(obj)
        continue
    with open(fpath) as f:
        r = json.load(f)

    v2path = os.path.join(V2_ROOT, obj, "evolution_result.json")
    v2s = 0
    if os.path.exists(v2path):
        with open(v2path) as f2:
            v2s = json.load(f2).get("final_hybrid", 0)

    fh = r["final_hybrid"]
    delta = fh - v2s
    sign = "+" if delta >= 0 else ""
    done.append((obj, fh, v2s, delta, r["accepted"], r["exit_reason"], r["probes_run"]))

print(f"=== evolution_v4b_full progress: {len(done)}/10 done ===\n")

if done:
    print(f"{'obj':<10} {'v4b':>7} {'v2':>7} {'delta':>8}  {'acc':<6} {'exit':<14} {'pr'}")
    print("-" * 62)
    for obj, fh, v2s, delta, acc, ex, pr in done:
        sign = "+" if delta >= 0 else ""
        acc_str = "YES" if acc else "no"
        print(f"{obj:<10} {fh:>7.4f} {v2s:>7.4f} {sign+f'{delta:.4f}':>8}  {acc_str:<6} {ex:<14} {pr}")

    finals = [x[1] for x in done]
    v2s_vals = [x[2] for x in done if x[2] > 0]
    n_accepted = sum(1 for x in done if x[3])
    print(f"\nDone so far: avg_v4b={sum(finals)/len(finals):.4f}  accepted={n_accepted}/{len(done)}")

if not_done:
    print(f"\nStill running: {', '.join(not_done)}")

    # Show partial progress
    print("\nPartial (renders dir count):")
    for obj in not_done:
        obj_dir = os.path.join(ROOT, obj)
        if os.path.exists(obj_dir):
            rounds = [d for d in os.listdir(obj_dir) if d.startswith("renders_u")]
            reviews = []
            rev_dir = os.path.join(obj_dir, "reviews")
            if os.path.exists(rev_dir):
                reviews = os.listdir(rev_dir)
            # Try to get latest score
            latest_score = "?"
            states_dir = os.path.join(obj_dir, "states")
            if os.path.exists(states_dir):
                state_files = sorted(os.listdir(states_dir))
                if state_files:
                    latest_score = f"{len(rounds)} renders, {len(reviews)} reviews"
            print(f"  {obj}: {latest_score}")
        else:
            print(f"  {obj}: not started")
