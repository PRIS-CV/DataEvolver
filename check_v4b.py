#!/usr/bin/env python3
"""Check v4b A/B test results"""
import json, os

ROOT = "/aaaidata/zhangqisong/data_build/pipeline/data/evolution_v4b"
V2_ROOT = "/aaaidata/zhangqisong/data_build/pipeline/data/evolution_v2"

for obj in ["obj_007", "obj_008", "obj_010"]:
    fpath = os.path.join(ROOT, obj, "evolution_result.json")
    if not os.path.exists(fpath):
        print(f"{obj}: NOT DONE YET")
        continue
    with open(fpath) as f:
        r = json.load(f)

    # Load v2 baseline
    v2path = os.path.join(V2_ROOT, obj, "evolution_result.json")
    v2s = 0
    if os.path.exists(v2path):
        with open(v2path) as f:
            v2s = json.load(f).get("final_hybrid", 0)

    fh = r["final_hybrid"]
    delta = fh - v2s
    sign = "+" if delta >= 0 else ""

    print(f"{obj}: final={fh:.4f} (v2={v2s:.4f}, {sign}{delta:.4f}) "
          f"accepted={r['accepted']} exit={r['exit_reason']} "
          f"probes={r['probes_run']} updates={r['updates_run']}")

    # Show score trajectory
    scores = [e["hybrid_score"] for e in r["state_log"]]
    actions = [e.get("action_taken", "?") for e in r["state_log"]]
    tags = [e.get("issue_tags", [])[:1] for e in r["state_log"]]
    print(f"  scores: {' -> '.join(f'{s:.3f}' for s in scores)}")
    print(f"  actions: {actions}")
    print(f"  tags: {tags}")
    print()
