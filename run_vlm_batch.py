"""
Batch VLM review for a subset of objects on a given device.

Usage:
  python run_vlm_batch.py --obj-ids obj_001 obj_002 obj_004 --device cuda:0
"""
import sys, os, json, argparse
sys.path.insert(0, '/aaaidata/zhangqisong/data_build/pipeline')
sys.path.insert(0, '/aaaidata/zhangqisong/data_build')

from stage5_5_vlm_review import review_object

RENDERS_DIR = '/aaaidata/zhangqisong/data_build/pipeline/data/renders'
OUTPUT_DIR  = '/aaaidata/zhangqisong/data_build/pipeline/data/vlm_reviews'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--obj-ids', nargs='+', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--active-group', default='lighting')
    p.add_argument('--round-idx', type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    print(f'[batch] device={args.device}, objs={args.obj_ids}, group={args.active_group}')

    results = {}
    for obj_id in args.obj_ids:
        print(f'\n{"="*60}')
        print(f'[batch] Processing {obj_id} on {args.device}')
        try:
            agg = review_object(
                obj_id=obj_id,
                renders_dir=RENDERS_DIR,
                output_dir=OUTPUT_DIR,
                round_idx=args.round_idx,
                active_group=args.active_group,
                device=args.device,
            )
            results[obj_id] = agg
        except Exception as e:
            import traceback
            print(f'ERROR on {obj_id}: {e}')
            traceback.print_exc()
            results[obj_id] = {'error': str(e)}

    print(f'\n{"="*60}')
    print(f'=== BATCH SUMMARY ({args.device}) ===')
    for oid, r in results.items():
        if 'error' in r:
            print(f'  {oid}: ERROR {r["error"]}')
        else:
            print(f'  {oid}: hybrid={r.get("hybrid_score","N/A")}, route={r.get("hybrid_route","?")}, tags={r.get("issue_tags","?")}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Fix 7: use CUDA_VISIBLE_DEVICES env var to avoid filename collision across GPUs
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", args.device.replace(":", ""))
    out_path = os.path.join(OUTPUT_DIR, f'batch_summary_gpu{visible}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'Results saved to {out_path}')

if __name__ == '__main__':
    main()
