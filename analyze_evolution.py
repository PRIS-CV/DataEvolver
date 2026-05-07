"""
analyze_evolution.py — 三表结果分析脚本

读取 evolution_summary.json (由 aggregate_results.py 生成) 及各 obj 的
evolution_result.json，输出：
  Table 1: Per-object 指标汇总
  Table 2: Issue tag 分布（次数 + 占比）
  Table 3: Exit reason 分布
"""

import json
import glob
import sys
import os
from collections import Counter


def load_summary(evolution_dir: str) -> dict:
    path = os.path.join(evolution_dir, "evolution_summary.json")
    if not os.path.exists(path):
        print(f"[analyze] evolution_summary.json not found at {path}")
        print("[analyze] Tip: run aggregate_results.py first")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_all_results(evolution_dir: str) -> list:
    results = []
    for fpath in sorted(glob.glob(os.path.join(evolution_dir, "obj_*", "evolution_result.json"))):
        with open(fpath) as f:
            results.append(json.load(f))
    return results


def table1_per_object(results: list):
    print("\n" + "=" * 70)
    print("TABLE 1: Per-Object Summary")
    print("=" * 70)
    header = f"{'obj_id':12s} {'final_hybrid':>12s} {'probes':>7s} {'updates':>8s} {'accepted':>9s} {'exit_reason':>14s}"
    print(header)
    print("-" * 70)
    for r in results:
        obj_id = r.get("obj_id", "?")
        score = r.get("final_hybrid", 0.0)
        probes = r.get("probes_run", "?")
        updates = r.get("updates_run", "?")
        accepted = "YES" if r.get("accepted", False) else "no"
        # exit_reason from last state_log entry if present
        state_log = r.get("state_log", [])
        exit_reason = "?"
        if state_log:
            last = state_log[-1]
            exit_reason = last.get("exit_reason", "max_rounds")
        print(f"{obj_id:12s} {score:>12.4f} {str(probes):>7s} {str(updates):>8s} {accepted:>9s} {exit_reason:>14s}")
    print("-" * 70)
    accepted_total = sum(1 for r in results if r.get("accepted", False))
    avg_score = sum(r.get("final_hybrid", 0.0) for r in results) / max(1, len(results))
    print(f"{'TOTAL':12s} {avg_score:>12.4f} {'':>7s} {'':>8s} {accepted_total}/{len(results):>6d}")


def table2_tag_distribution(results: list):
    print("\n" + "=" * 70)
    print("TABLE 2: Issue Tag Distribution")
    print("=" * 70)
    all_tags = []
    for r in results:
        for entry in r.get("state_log", []):
            tags = entry.get("issue_tags", [])
            all_tags.extend([t for t in tags if t != "none"])
    total = max(1, len(all_tags))
    counter = Counter(all_tags)
    print(f"{'tag':30s} {'count':>8s} {'pct':>8s}")
    print("-" * 50)
    for tag, cnt in counter.most_common():
        print(f"{tag:30s} {cnt:>8d} {cnt/total*100:>7.1f}%")
    print(f"\nTotal tag occurrences: {total}")


def table3_exit_reasons(results: list):
    print("\n" + "=" * 70)
    print("TABLE 3: Exit Reason Distribution")
    print("=" * 70)
    reasons = []
    for r in results:
        state_log = r.get("state_log", [])
        if state_log:
            last = state_log[-1]
            reasons.append(last.get("exit_reason", "max_rounds"))
        else:
            reasons.append("no_data")
    counter = Counter(reasons)
    total = max(1, len(reasons))
    print(f"{'exit_reason':20s} {'count':>8s} {'pct':>8s}")
    print("-" * 40)
    for reason, cnt in counter.most_common():
        print(f"{reason:20s} {cnt:>8d} {cnt/total*100:>7.1f}%")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Analyze evolution results (3 tables)")
    p.add_argument("--evolution-dir", required=True,
                   help="Evolution output dir (contains evolution_summary.json + obj_*/)")
    args = p.parse_args()

    summary = load_summary(args.evolution_dir)
    results = load_all_results(args.evolution_dir)

    if not results:
        print("[analyze] No evolution_result.json files found.")
        sys.exit(1)

    print(f"\n[analyze] Loaded {len(results)} objects from {args.evolution_dir}")
    print(f"[analyze] Summary: accepted={summary.get('accepted', '?')}/{summary.get('total', '?')}, "
          f"rate={summary.get('acceptance_rate', 0):.1%}")

    table1_per_object(results)
    table2_tag_distribution(results)
    table3_exit_reasons(results)
    print()


if __name__ == "__main__":
    main()
