#!/usr/bin/env python3
"""Force-accept all rejected objects in gate_manifest.json."""
import json
import sys
from pathlib import Path

gate_root = Path(sys.argv[1])
manifest_path = gate_root / "gate_manifest.json"
if not manifest_path.exists():
    print(f"[force_accept] ERROR: {manifest_path} not found")
    sys.exit(1)

manifest = json.loads(manifest_path.read_text())
rejected = manifest.get("rejected", [])
accepted = manifest.get("accepted", [])
force_promoted = []

for item in rejected:
    obj_id = item.get("obj_id", "unknown")
    pair_dir = item.get("pair_dir", "")
    best_score = item.get("best_score", 0)
    best_round_idx = item.get("best_round_idx", 0)

    promoted = {
        "status": "accepted",
        "obj_id": obj_id,
        "rotation_deg": item.get("rotation_deg", 0),
        "pair_dir": pair_dir,
        "accepted_round_idx": best_round_idx,
        "accepted_score": best_score,
        "accepted_threshold": 0.0,
        "score_field": item.get("score_field", "hybrid_score"),
        "force_accepted": True,
        "original_reject_reason": item.get("reason", "unknown"),
        "best_score": best_score,
        "best_round_idx": best_round_idx,
        "best_state_out": item.get("best_state_out"),
        "best_render_dir": item.get("best_render_dir"),
        "best_review_agg_path": item.get("best_review_agg_path"),
        "state_out": item.get("best_state_out"),
        "render_dir": item.get("best_render_dir"),
        "review_agg_path": item.get("best_review_agg_path"),
        "rounds": item.get("rounds", []),
        "elapsed_seconds": item.get("elapsed_seconds", 0),
    }
    accepted.append(promoted)
    force_promoted.append(obj_id)
    print(f"  [force_accept] {obj_id}: rejected -> accepted (best_score={best_score:.3f}, round={best_round_idx})")

    gate_status_path = Path(pair_dir) / "gate_status.json"
    if gate_status_path.exists():
        gs = json.loads(gate_status_path.read_text())
        gs["status"] = "accepted"
        gs["force_accepted"] = True
        gs["accepted_round_idx"] = best_round_idx
        gs["accepted_score"] = best_score
        gs["accepted_threshold"] = 0.0
        gate_status_path.write_text(json.dumps(gs, indent=2, ensure_ascii=False))

manifest["accepted"] = accepted
manifest["rejected"] = []
manifest["accepted_count"] = len(accepted)
manifest["rejected_count"] = 0
manifest["force_promoted"] = force_promoted

manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(f"[force_accept] done: {len(force_promoted)} force-accepted, {len(accepted)} total accepted")
