"""
aggregate_results.py — Merge per-object evolution results into evolution_summary.json.

Scans evolution_dir/obj_*/evolution_result.json and aggregates into a single
summary file via atomic os.replace(), safe for concurrent use.

Usage:
  python aggregate_results.py --evolution-dir pipeline/data/evolution_full
"""

import os
import glob
import json
import argparse


def aggregate_evolution(evolution_dir: str, output_path: str = None) -> dict:
    """
    Scan evolution_dir/obj_*/evolution_result.json and merge into evolution_summary.json.
    Returns the summary dict.
    Uses atomic os.replace() to avoid partial writes.
    """
    if output_path is None:
        output_path = os.path.join(evolution_dir, "evolution_summary.json")

    pattern = os.path.join(evolution_dir, "obj_*", "evolution_result.json")
    result_files = sorted(glob.glob(pattern))

    if not result_files:
        print(f"[aggregate] No evolution_result.json found under {evolution_dir}")
        return {"total": 0, "accepted": 0, "acceptance_rate": 0.0,
                "final_scores": {}, "results": {}}

    all_results = {}
    for rf in result_files:
        try:
            with open(rf) as f:
                r = json.load(f)
            obj_id = r.get("obj_id", os.path.basename(os.path.dirname(rf)))
            all_results[obj_id] = r
        except Exception as e:
            print(f"[aggregate] WARN: could not read {rf}: {e}")

    accepted = sum(1 for r in all_results.values() if r.get("accepted", False))
    final_scores = {k: v.get("final_hybrid", 0.0) for k, v in all_results.items()}

    summary = {
        "total":           len(all_results),
        "accepted":        accepted,
        "acceptance_rate": accepted / max(1, len(all_results)),
        "final_scores":    final_scores,
        "results":         all_results,
    }

    # Atomic write: write to .tmp then rename
    tmp_path = output_path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp_path, output_path)

    avg_score = sum(final_scores.values()) / max(1, len(final_scores))
    print(f"[aggregate] {len(all_results)} objects: {accepted} accepted "
          f"({accepted/max(1,len(all_results)):.1%}), avg_score={avg_score:.3f}")
    print(f"[aggregate] Summary → {output_path}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate per-object evolution_result.json into evolution_summary.json"
    )
    p.add_argument("--evolution-dir", required=True,
                   help="Root evolution dir containing obj_*/evolution_result.json")
    p.add_argument("--output", default=None,
                   help="Output path (default: {evolution-dir}/evolution_summary.json)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = aggregate_evolution(args.evolution_dir, args.output)
    print(f"\n=== Aggregation complete: {summary['accepted']}/{summary['total']} accepted, "
          f"rate={summary['acceptance_rate']:.1%} ===")
