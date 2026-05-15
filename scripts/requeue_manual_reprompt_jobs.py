#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
PIPELINE_ROOT = REPO_ROOT / "pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from asset_lifecycle import enqueue_stage1_restart, load_queue, load_registry


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-requeue manual_reprompt_required assets back into stage1_restart jobs."
    )
    parser.add_argument("--max-jobs", type=int, default=2)
    parser.add_argument("--max-stage1-restarts", type=int, default=999)
    return parser.parse_args()


def _attempt_idx_from_version(version_id: str | None) -> int:
    raw = str(version_id or "")
    if not raw.startswith("attempt_"):
        return 0
    try:
        return int(raw.split("_", 1)[1])
    except Exception:
        return 0


def _latest_known_attempt_idx(entry: dict) -> int:
    best = int(entry.get("replacement_attempts_used") or 0)
    for version in entry.get("versions") or []:
        best = max(best, _attempt_idx_from_version(version.get("version_id")))
    return best


def _has_active_queue_job(canonical_id: str) -> bool:
    for job in load_queue():
        if job.get("canonical_id") != canonical_id:
            continue
        if str(job.get("status") or "") in {"queued", "running"}:
            return True
    return False


def main():
    args = parse_args()
    registry = load_registry()
    queued = []

    for canonical_id in sorted(registry):
        if len(queued) >= args.max_jobs:
            break
        entry = registry.get(canonical_id) or {}
        if str(entry.get("status") or "") != "manual_reprompt_required":
            continue
        if _has_active_queue_job(canonical_id):
            continue

        after_attempt_idx = _latest_known_attempt_idx(entry)
        deprecated_version = str(entry.get("active_version") or f"attempt_{after_attempt_idx:02d}")
        reason = str(entry.get("deprecation_reason") or "auto_requeue_manual_reprompt")
        job = enqueue_stage1_restart(
            canonical_id,
            after_attempt_idx=after_attempt_idx,
            deprecated_version=deprecated_version,
            reason=reason,
            trigger_source="manual_reprompt:auto_requeue",
            max_stage1_restarts=args.max_stage1_restarts,
        )
        if job is not None:
            queued.append(
                {
                    "canonical_id": canonical_id,
                    "attempt_idx": int(job.get("attempt_idx") or 0),
                    "stage1_restart_idx": int(job.get("stage1_restart_idx") or 0),
                    "repair_from_attempt_idx": int(job.get("repair_from_attempt_idx") or after_attempt_idx),
                }
            )

    print(json.dumps({"queued_jobs": queued, "queued_count": len(queued)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
