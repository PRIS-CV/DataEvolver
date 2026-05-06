"""
Step 2: Batch VLM review for all objects + consistency test.

1. Reviews all obj_* that haven't been reviewed yet.
2. Runs obj_001 az=0 image 3 times to compute score std dev (consistency check).
3. Prints analysis summary.
"""

import os, sys, json, time
import numpy as np

sys.path.insert(0, "/aaaidata/zhangqisong/data_build/pipeline")
from stage5_5_vlm_review import review_object, run_vlm_review, compute_cv_metrics, REP_VIEWS

RENDERS_DIR = "/aaaidata/zhangqisong/data_build/pipeline/data/renders"
OUTPUT_DIR  = "/aaaidata/zhangqisong/data_build/pipeline/data/vlm_reviews"
DEVICE      = "cuda:0"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Discover all available obj_ids
all_objs = sorted(
    d for d in os.listdir(RENDERS_DIR)
    if os.path.isdir(os.path.join(RENDERS_DIR, d))
)
print(f"[batch] Found {len(all_objs)} objects: {all_objs}")

# ── 1. Review remaining objects ───────────────────────────────────────────
results = {}
for obj_id in all_objs:
    agg_path = os.path.join(OUTPUT_DIR, f"{obj_id}_r00_agg.json")
    if os.path.exists(agg_path):
        with open(agg_path) as f:
            results[obj_id] = json.load(f)
        print(f"[batch] {obj_id}: cached (hybrid={results[obj_id].get('hybrid_score','?')})")
        continue

    print(f"\n[batch] === Reviewing {obj_id} ===")
    t0 = time.time()
    r = review_object(
        obj_id=obj_id,
        renders_dir=RENDERS_DIR,
        output_dir=OUTPUT_DIR,
        round_idx=0,
        active_group="lighting",
        device=DEVICE,
    )
    results[obj_id] = r
    print(f"[batch] {obj_id} done in {time.time()-t0:.1f}s")

# ── 2. Distribution analysis ──────────────────────────────────────────────
print("\n=== Score Distribution Analysis ===")
hybrids = [r.get("hybrid_score", 0) for r in results.values() if r]
routes  = [r.get("hybrid_route", "?") for r in results.values() if r]
all_vlm = []
for r in results.values():
    if r and "agg_vlm_scores" in r:
        s = r["agg_vlm_scores"]
        all_vlm.append([s.get(k, 3) for k in ("lighting","object_integrity","composition","render_quality_semantic","overall")])

print(f"Objects reviewed: {len(hybrids)}")
print(f"Hybrid scores: min={min(hybrids):.3f} max={max(hybrids):.3f} mean={np.mean(hybrids):.3f} std={np.std(hybrids):.3f}")
print(f"Routes: { {r: routes.count(r) for r in set(routes)} }")
if all_vlm:
    arr = np.array(all_vlm)
    for i, dim in enumerate(("lighting","obj_integrity","composition","rq_semantic","overall")):
        print(f"  {dim}: mean={arr[:,i].mean():.2f} std={arr[:,i].std():.2f}")

# Issue tag frequency
tag_freq = {}
for r in results.values():
    if r:
        for t in r.get("issue_tags", []):
            tag_freq[t] = tag_freq.get(t, 0) + 1
print(f"Issue tag frequency: {dict(sorted(tag_freq.items(), key=lambda x: -x[1]))}")

# ── 3. Consistency test: obj_001 az=0, 3 repeated inferences ─────────────
print("\n=== Consistency Test (obj_001 az=0, N=3) ===")
from stage5_5_vlm_review import run_vlm_review
rgb_path = os.path.join(RENDERS_DIR, "obj_001", "az000_el+00.png")
consistency_scores = []
for trial in range(3):
    print(f"  Trial {trial+1}/3 ...")
    t0 = time.time()
    rev = run_vlm_review(rgb_path, "obj_001_az000_el+00", 0, "lighting", 0, 0, device=DEVICE)
    consistency_scores.append(rev["scores"])
    print(f"  scores: {rev['scores']} tags: {rev['issue_tags']} ({time.time()-t0:.1f}s)")

dims = ("lighting","object_integrity","composition","render_quality_semantic","overall")
print("\nConsistency std dev per dimension (target <= 0.5):")
for d in dims:
    vals = [s[d] for s in consistency_scores]
    std  = float(np.std(vals))
    ok   = "PASS" if std <= 0.5 else "FAIL"
    print(f"  {d}: values={vals} std={std:.3f} [{ok}]")

# Save summary
summary = {
    "batch_results": {k: {
        "hybrid_score": v.get("hybrid_score"),
        "hybrid_route": v.get("hybrid_route"),
        "issue_tags":   v.get("issue_tags"),
        "suggested_actions": v.get("suggested_actions"),
    } for k, v in results.items()},
    "distribution": {
        "hybrids": hybrids,
        "mean": float(np.mean(hybrids)),
        "std":  float(np.std(hybrids)),
        "routes": {r: routes.count(r) for r in set(routes)},
    },
    "consistency_test": {
        "scores_per_trial": consistency_scores,
        "std_per_dim": {d: float(np.std([s[d] for s in consistency_scores])) for d in dims},
    }
}
with open(os.path.join(OUTPUT_DIR, "step2_analysis.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nStep 2 analysis saved to {OUTPUT_DIR}/step2_analysis.json")
