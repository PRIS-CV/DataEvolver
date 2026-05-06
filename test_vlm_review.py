"""
快速测试：对所有 obj 的代表视角（az=0/90/180/270, el=0）跑 VLM review
验证标准：
  1. JSON parse success rate >= 95%
  2. 输出所有字段完整
  3. 评分分布合理
"""
import sys, os, json
sys.path.insert(0, '/aaaidata/zhangqisong/data_build/pipeline')
sys.path.insert(0, '/aaaidata/zhangqisong/data_build')

from stage5_5_vlm_review import review_object

RENDERS_DIR = '/aaaidata/zhangqisong/data_build/pipeline/data/renders'
OUTPUT_DIR  = '/aaaidata/zhangqisong/data_build/pipeline/data/vlm_reviews_test'
DEVICE = 'cuda:0'

import os
obj_ids = sorted(d for d in os.listdir(RENDERS_DIR) if os.path.isdir(os.path.join(RENDERS_DIR, d)))
print(f"Found {len(obj_ids)} objects: {obj_ids}")

results = {}
parse_ok = 0
total = 0

for obj_id in obj_ids:
    print(f"\n{'='*50}")
    try:
        agg = review_object(
            obj_id=obj_id,
            renders_dir=RENDERS_DIR,
            output_dir=OUTPUT_DIR,
            round_idx=0,
            active_group='lighting',
            device=DEVICE,
        )
        results[obj_id] = agg
        if agg:
            parse_ok += agg.get('num_views_reviewed', 0)
            total    += agg.get('num_views_reviewed', 0)
    except Exception as e:
        import traceback
        print(f"ERROR on {obj_id}: {e}")
        traceback.print_exc()

print(f"\n{'='*60}")
print(f"=== SUMMARY: {len(results)} objects reviewed ===")
for oid, r in results.items():
    print(f"  {oid}: hybrid={r.get('hybrid_score','N/A'):.3f}, route={r.get('hybrid_route','?')}, tags={r.get('issue_tags','?')}")

# Save summary
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, 'test_summary.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUTPUT_DIR}/test_summary.json")
