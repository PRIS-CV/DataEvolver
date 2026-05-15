from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import socket
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows local editing path
    fcntl = None


PIPELINE_DIR = Path(__file__).resolve().parent
DATA_DIR = PIPELINE_DIR / "data"
PROMPTS_PATH = DATA_DIR / "prompts.json"
IMAGES_DIR = DATA_DIR / "images"
IMAGES_RGBA_DIR = DATA_DIR / "images_rgba"
MESHES_RAW_DIR = DATA_DIR / "meshes_raw"
MESHES_DIR = DATA_DIR / "meshes"
ASSET_VERSIONS_DIR = DATA_DIR / "asset_versions"
ASSET_REGISTRY_PATH = DATA_DIR / "asset_registry.json"
REGEN_QUEUE_PATH = DATA_DIR / "regeneration_queue.json"
STATE_LOCK_PATH = DATA_DIR / ".asset_state.lock"

ASSET_VIABILITY_VALUES = {"continue", "abandon", "unclear"}
ABANDON_CONFIDENCE_VALUES = {"low", "medium", "high"}
REGISTRY_STATUSES = {
    "active",
    "deprecated_pending_replacement",
    "deprecated",
    "replacement_failed",
    "manual_reprompt_required",
}
QUEUE_STATUSES = {"queued", "running", "smoke_failed", "promoted", "exhausted"}
BLOCKING_ASSET_STATUSES = {
    "deprecated_pending_replacement",
    "manual_reprompt_required",
}
GEOMETRY_QUALITY_SUFFIX = "high quality 3D geometry, detailed surface, clear edges, non-flat"

ABANDON_HINTS = [
    "should be regenerated",
    "needs regeneration",
    "regenerate this asset",
    "regenerate the asset",
    "unusable asset",
    "not salvageable",
    "cannot be salvaged",
    "wrong object geometry",
    "fundamentally wrong geometry",
    "render effect not salvageable",
    "不可补救",
    "建议重生",
    "建议重新生成",
    "需要重新生成",
    "渲染效果不可补救",
    "几何完全不对",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default):
    if not path.exists():
        return copy.deepcopy(default)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp_path = Path(f.name)
    os.replace(tmp_path, path)


@contextmanager
def asset_state_lock():
    STATE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_LOCK_PATH.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _running_owner_tag() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _parse_iso_datetime(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None


def load_registry() -> dict:
    data = _read_json(ASSET_REGISTRY_PATH, {})
    return data if isinstance(data, dict) else {}


def save_registry(registry: dict):
    _write_json(ASSET_REGISTRY_PATH, registry)


def load_queue() -> list[dict]:
    data = _read_json(REGEN_QUEUE_PATH, [])
    return data if isinstance(data, list) else []


def save_queue(queue: list[dict]):
    _write_json(REGEN_QUEUE_PATH, queue)


def load_prompts() -> list[dict]:
    data = _read_json(PROMPTS_PATH, [])
    return data if isinstance(data, list) else []


def save_prompts(prompts: list[dict]):
    _write_json(PROMPTS_PATH, prompts)


def attempt_version_id(attempt_idx: int) -> str:
    return f"attempt_{int(attempt_idx):02d}"


def version_dir(canonical_id: str, version_id: str) -> Path:
    return ASSET_VERSIONS_DIR / canonical_id / version_id


def find_prompt_entry(prompts: list[dict], canonical_id: str) -> Optional[dict]:
    for item in prompts:
        if item.get("id") == canonical_id:
            return item
    return None


def _find_version_entry(entry: dict, version_id: str) -> Optional[dict]:
    for version in entry.get("versions", []):
        if version.get("version_id") == version_id:
            return version
    return None


def ensure_registry_entry(canonical_id: str) -> tuple[dict, dict]:
    registry = load_registry()
    entry = registry.get(canonical_id)
    changed = False
    if not isinstance(entry, dict):
        entry = {
            "canonical_id": canonical_id,
            "active_version": attempt_version_id(0),
            "status": "active",
            "replacement_attempts_used": 0,
            "stage1_restart_count": 0,
            "deprecation_reason": None,
            "versions": [],
            "created_at": _now_iso(),
        }
        registry[canonical_id] = entry
        changed = True

    if entry.get("status") not in REGISTRY_STATUSES:
        entry["status"] = "active"
        changed = True
    if not isinstance(entry.get("replacement_attempts_used"), int):
        entry["replacement_attempts_used"] = int(entry.get("replacement_attempts_used") or 0)
        changed = True
    if not isinstance(entry.get("stage1_restart_count"), int):
        entry["stage1_restart_count"] = int(entry.get("stage1_restart_count") or 0)
        changed = True
    if not isinstance(entry.get("versions"), list):
        entry["versions"] = []
        changed = True

    active_version = str(entry.get("active_version") or attempt_version_id(0))
    entry["active_version"] = active_version
    version = _find_version_entry(entry, active_version)
    if version is None:
        version = {
            "version_id": active_version,
            "status": "active",
            "path": str(version_dir(canonical_id, active_version)),
            "created_at": entry.get("created_at") or _now_iso(),
        }
        entry["versions"].append(version)
        changed = True
    elif version.get("status") not in REGISTRY_STATUSES:
        version["status"] = "active"
        changed = True

    if changed:
        save_registry(registry)
    return registry, entry


def get_registry_entry(canonical_id: str) -> Optional[dict]:
    registry = load_registry()
    entry = registry.get(canonical_id)
    return entry if isinstance(entry, dict) else None


def asset_status_blocks_scene_loop(status: Optional[str]) -> bool:
    return str(status or "") in BLOCKING_ASSET_STATUSES


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def snapshot_current_active_version(canonical_id: str) -> tuple[dict, dict, dict]:
    registry, entry = ensure_registry_entry(canonical_id)
    active_version = str(entry.get("active_version") or attempt_version_id(0))
    attempt_dir = version_dir(canonical_id, active_version)
    attempt_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts()
    prompt_entry = copy.deepcopy(find_prompt_entry(prompts, canonical_id) or {"id": canonical_id, "prompt": ""})
    _write_json(attempt_dir / "prompt.json", prompt_entry)
    _write_json(attempt_dir / "prompts.json", [prompt_entry])

    _copy_if_exists(IMAGES_DIR / f"{canonical_id}.png", attempt_dir / "images" / f"{canonical_id}.png")
    _copy_if_exists(IMAGES_RGBA_DIR / f"{canonical_id}.png", attempt_dir / "images_rgba" / f"{canonical_id}.png")
    _copy_if_exists(MESHES_RAW_DIR / f"{canonical_id}.glb", attempt_dir / "meshes_raw" / f"{canonical_id}.glb")
    _copy_if_exists(MESHES_DIR / f"{canonical_id}.glb", attempt_dir / "meshes" / f"{canonical_id}.glb")

    version_entry = _find_version_entry(entry, active_version)
    if version_entry is None:
        version_entry = {
            "version_id": active_version,
            "status": entry.get("status") if entry.get("status") in REGISTRY_STATUSES else "active",
            "created_at": _now_iso(),
        }
        entry.setdefault("versions", []).append(version_entry)
    version_entry["path"] = str(attempt_dir)
    version_entry["prompt_path"] = str(attempt_dir / "prompt.json")
    version_entry["snapshot_updated_at"] = _now_iso()
    save_registry(registry)
    return registry, entry, version_entry


def compute_seed_bases(canonical_id: str, attempt_idx: int) -> tuple[int, int]:
    digest = hashlib.sha256(f"{canonical_id}:{attempt_idx}".encode("utf-8")).hexdigest()
    seed_anchor = int(digest[:10], 16)
    seed_base_stage2 = 100000 + (seed_anchor % 700000)
    seed_base_stage3 = 200000 + ((seed_anchor // 7) % 700000)
    return seed_base_stage2, seed_base_stage3


def append_geometry_quality_suffix(prompt_text: str) -> str:
    prompt_text = str(prompt_text or "").strip()
    if not prompt_text:
        return GEOMETRY_QUALITY_SUFFIX
    if GEOMETRY_QUALITY_SUFFIX.lower() in prompt_text.lower():
        return prompt_text
    return f"{prompt_text}, {GEOMETRY_QUALITY_SUFFIX}"


def create_attempt_prompt_snapshot(
    canonical_id: str,
    attempt_idx: int,
    *,
    prompt_entry: Optional[dict] = None,
    add_geometry_suffix: bool = False,
) -> tuple[Path, dict]:
    attempt_dir = version_dir(canonical_id, attempt_version_id(attempt_idx))
    attempt_dir.mkdir(parents=True, exist_ok=True)
    if prompt_entry is None:
        prompts = load_prompts()
        prompt_entry = copy.deepcopy(find_prompt_entry(prompts, canonical_id) or {"id": canonical_id, "prompt": ""})
    else:
        prompt_entry = copy.deepcopy(prompt_entry)
    if add_geometry_suffix:
        prompt_entry["prompt"] = append_geometry_quality_suffix(prompt_entry.get("prompt", ""))
    _write_json(attempt_dir / "prompt.json", prompt_entry)
    prompts_snapshot_path = attempt_dir / "prompts.json"
    _write_json(prompts_snapshot_path, [prompt_entry])
    return prompts_snapshot_path, prompt_entry


def _last_structured_match(text: str, pattern: str) -> Optional[str]:
    matches = re.findall(pattern, str(text or ""), flags=re.IGNORECASE | re.MULTILINE)
    if not matches:
        return None
    last = matches[-1]
    if isinstance(last, tuple):
        return str(last[0]).strip()
    return str(last).strip()


def detect_asset_viability(trace_text: str = "", agg: Optional[dict] = None) -> tuple[str, Optional[str], Optional[str]]:
    agg = agg or {}
    text = str(trace_text or "")
    viability = _last_structured_match(text, r"^\s*Asset viability:\s*(continue|abandon|unclear)\b")
    reason = _last_structured_match(text, r"^\s*Abandon reason:\s*(.+?)\s*$")
    confidence = _last_structured_match(text, r"^\s*Abandon confidence:\s*(low|medium|high|none)\b")
    if viability:
        viability = viability.lower()
        if str(reason or "").strip().lower() in {"none", "null", "n/a", "na"}:
            reason = None
        confidence = str(confidence or "").lower() or None
        if confidence not in ABANDON_CONFIDENCE_VALUES:
            confidence = None
        return viability, reason, confidence

    agg_viability = str(agg.get("asset_viability") or "").lower()
    agg_reason = agg.get("abandon_reason")
    agg_confidence = str(agg.get("abandon_confidence") or "").lower() or None
    if agg_viability in ASSET_VIABILITY_VALUES:
        if agg_confidence not in ABANDON_CONFIDENCE_VALUES:
            agg_confidence = None
        return agg_viability, agg_reason, agg_confidence

    lowered = text.lower()
    for phrase in ABANDON_HINTS:
        if phrase.lower() in lowered:
            return "abandon", phrase, "high"
    return "unclear", agg_reason, agg_confidence if agg_confidence in ABANDON_CONFIDENCE_VALUES else None


def detect_verdict_from_review(trace_text: str = "", agg: Optional[dict] = None) -> str:
    verdict = _last_structured_match(str(trace_text or ""), r"^\s*Verdict:\s*(keep|revise|reject)\b")
    if verdict:
        return verdict.lower()
    route = str((agg or {}).get("hybrid_route") or "").lower()
    if route in {"accept", "accepted", "keep"}:
        return "keep"
    if route in {"reject", "rejected"}:
        return "reject"
    return "revise"


def _load_round_agg(pair_dir: Path, round_idx: int) -> Optional[dict]:
    reviews_dir = pair_dir / "reviews"
    matches = sorted(reviews_dir.glob(f"*_r{round_idx:02d}_agg.json"))
    if not matches:
        return None
    return _read_json(matches[-1], None)


def _load_round_trace_text(pair_dir: Path, round_idx: int) -> str:
    reviews_dir = pair_dir / "reviews"
    matches = sorted(reviews_dir.glob(f"*_r{round_idx:02d}_*_trace.json"))
    if not matches:
        return ""
    trace_payload = _read_json(matches[0], {})
    attempts = trace_payload.get("attempts") or []
    if attempts:
        assistant_text = attempts[-1].get("assistant_text")
        if isinstance(assistant_text, str):
            return assistant_text
    for key in ("assistant_text", "raw_text", "response_text", "content", "answer", "model_output"):
        value = trace_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _load_round_decision(pair_dir: Path, round_idx: int) -> Optional[dict]:
    decision_path = pair_dir / "decisions" / f"round{round_idx:02d}_decision.json"
    if not decision_path.exists():
        return None
    return _read_json(decision_path, None)


def evaluate_deprecation_for_pair(
    pair_dir: Path,
    latest_round_idx: int,
    latest_agg: Optional[dict],
    latest_trace_text: str,
    latest_verdict: str,
    soft_threshold: float = 0.58,
) -> dict:
    latest_agg = latest_agg or {}
    asset_viability, abandon_reason, abandon_confidence = detect_asset_viability(latest_trace_text, latest_agg)

    if str(latest_agg.get("structure_consistency") or "") == "major_mismatch":
        return {
            "should_deprecate": True,
            "deprecation_kind": "hard",
            "reason": "structure_consistency=major_mismatch",
            "asset_viability": asset_viability,
            "abandon_reason": abandon_reason,
            "abandon_confidence": abandon_confidence,
            "evidence": {"structure_consistency": "major_mismatch"},
        }

    if asset_viability == "abandon":
        return {
            "should_deprecate": True,
            "deprecation_kind": "hard",
            "reason": abandon_reason or "asset_viability=abandon",
            "asset_viability": asset_viability,
            "abandon_reason": abandon_reason,
            "abandon_confidence": abandon_confidence,
            "evidence": {
                "asset_viability": asset_viability,
                "abandon_reason": abandon_reason,
                "abandon_confidence": abandon_confidence,
            },
        }

    if latest_round_idx < 3:
        return {
            "should_deprecate": False,
            "asset_viability": asset_viability,
            "abandon_reason": abandon_reason,
            "abandon_confidence": abandon_confidence,
        }

    baseline = _load_round_agg(pair_dir, 0)
    round_aggs = [_load_round_agg(pair_dir, idx) for idx in (1, 2, 3)]
    if baseline is None or any(item is None for item in round_aggs):
        return {
            "should_deprecate": False,
            "asset_viability": asset_viability,
            "abandon_reason": abandon_reason,
            "abandon_confidence": abandon_confidence,
        }

    round_verdicts = []
    round_scores = []
    round_tags = []
    for idx, agg in zip((1, 2, 3), round_aggs):
        decision = _load_round_decision(pair_dir, idx) or {}
        verdict = str(decision.get("detected_verdict") or "").lower()
        if verdict not in {"keep", "revise", "reject"}:
            verdict = detect_verdict_from_review(_load_round_trace_text(pair_dir, idx), agg)
        round_verdicts.append(verdict)
        round_scores.append(float(agg.get("hybrid_score") or 0.0))
        round_tags.append([tag for tag in (agg.get("issue_tags") or []) if tag and tag != "none"])

    if any(verdict == "keep" for verdict in round_verdicts):
        return {
            "should_deprecate": False,
            "asset_viability": asset_viability,
            "abandon_reason": abandon_reason,
            "abandon_confidence": abandon_confidence,
        }

    tag_counter = Counter()
    for tags in round_tags:
        tag_counter.update(set(tags))
    recurring_issue_count = sum(1 for _, count in tag_counter.items() if count >= 2)

    baseline_score = float(baseline.get("hybrid_score") or 0.0)
    best_score = max(round_scores)
    improvement = best_score - baseline_score

    if best_score < soft_threshold and improvement < 0.02 and recurring_issue_count >= 2:
        return {
            "should_deprecate": True,
            "deprecation_kind": "soft",
            "reason": (
                f"round1-3 stalled: best_hybrid={best_score:.4f} < {soft_threshold:.2f}, "
                f"improvement={improvement:.4f} < 0.02, recurring_issue_count={recurring_issue_count}"
            ),
            "asset_viability": asset_viability,
            "abandon_reason": abandon_reason,
            "abandon_confidence": abandon_confidence,
            "evidence": {
                "baseline_hybrid": round(baseline_score, 4),
                "round_scores": [round(score, 4) for score in round_scores],
                "round_verdicts": round_verdicts,
                "recurring_issue_tags": sorted([tag for tag, count in tag_counter.items() if count >= 2]),
                "recurring_issue_count": recurring_issue_count,
                "best_hybrid": round(best_score, 4),
                "improvement_vs_baseline": round(improvement, 4),
            },
        }

    return {
        "should_deprecate": False,
        "asset_viability": asset_viability,
        "abandon_reason": abandon_reason,
        "abandon_confidence": abandon_confidence,
    }


def ensure_deprecated_and_enqueue(
    canonical_id: str,
    *,
    reason: str,
    trigger_source: str,
    evidence: Optional[dict] = None,
    max_attempts: int = 2,
) -> dict:
    with asset_state_lock():
        registry, entry, version_entry = snapshot_current_active_version(canonical_id)
        queue = load_queue()
        active_version = str(entry.get("active_version") or attempt_version_id(0))

        version_entry["status"] = "deprecated_pending_replacement"
        version_entry["deprecated_at"] = _now_iso()
        version_entry["deprecation_reason"] = reason
        if evidence:
            version_entry["deprecation_evidence"] = evidence

        entry["status"] = "deprecated_pending_replacement"
        entry["deprecation_reason"] = reason
        entry["deprecated_at"] = _now_iso()

        existing_job = next(
            (
                job
                for job in queue
                if job.get("canonical_id") == canonical_id
                and job.get("status") in {"queued", "running"}
            ),
            None,
        )
        if existing_job is not None:
            save_registry(registry)
            return {
                "canonical_id": canonical_id,
                "active_version": active_version,
                "status": entry["status"],
                "enqueued": False,
                "job": copy.deepcopy(existing_job),
            }

        next_attempt = int(entry.get("replacement_attempts_used") or 0) + 1
        if next_attempt > max_attempts:
            entry["status"] = "manual_reprompt_required"
            version_entry["status"] = "replacement_failed"
            save_registry(registry)
            return {
                "canonical_id": canonical_id,
                "active_version": active_version,
                "status": entry["status"],
                "enqueued": False,
                "job": None,
            }

        seed_base_stage2, seed_base_stage3 = compute_seed_bases(canonical_id, next_attempt)
        job = {
            "canonical_id": canonical_id,
            "deprecated_version": active_version,
            "reason": reason,
            "trigger_source": trigger_source,
            "attempt_idx": next_attempt,
            "status": "queued",
            "job_kind": "replacement",
            "cycle_attempt_idx": 1,
            "stage1_restart_idx": int(entry.get("stage1_restart_count") or 0),
            "use_geometry_suffix": False,
            "seed_base_stage2": seed_base_stage2,
            "seed_base_stage3": seed_base_stage3,
            "enqueued_at": _now_iso(),
        }
        if evidence:
            job["evidence"] = evidence
        queue.append(job)
        save_registry(registry)
        save_queue(queue)
        return {
            "canonical_id": canonical_id,
            "active_version": active_version,
            "status": entry["status"],
            "enqueued": True,
            "job": copy.deepcopy(job),
        }


def enqueue_replacement_attempt(
    canonical_id: str,
    *,
    attempt_idx: int,
    deprecated_version: str,
    reason: str,
    trigger_source: str,
    cycle_attempt_idx: int,
    stage1_restart_idx: int,
    use_geometry_suffix: bool,
    job_kind: str = "replacement",
    prompt_source_attempt_idx: Optional[int] = None,
) -> Optional[dict]:
    with asset_state_lock():
        queue = load_queue()
        if any(
            job.get("canonical_id") == canonical_id and job.get("status") in {"queued", "running"}
            for job in queue
        ):
            return None
        seed_base_stage2, seed_base_stage3 = compute_seed_bases(canonical_id, attempt_idx)
        job = {
            "canonical_id": canonical_id,
            "deprecated_version": deprecated_version,
            "reason": reason,
            "trigger_source": trigger_source,
            "attempt_idx": attempt_idx,
            "status": "queued",
            "job_kind": job_kind,
            "cycle_attempt_idx": cycle_attempt_idx,
            "stage1_restart_idx": stage1_restart_idx,
            "use_geometry_suffix": use_geometry_suffix,
            "prompt_source_attempt_idx": prompt_source_attempt_idx,
            "seed_base_stage2": seed_base_stage2,
            "seed_base_stage3": seed_base_stage3,
            "enqueued_at": _now_iso(),
        }
        queue.append(job)
        save_queue(queue)
        return copy.deepcopy(job)


def claim_next_queued_job() -> Optional[dict]:
    with asset_state_lock():
        queue = load_queue()
        for job in queue:
            if job.get("status") != "queued":
                continue
            job["status"] = "running"
            job["started_at"] = _now_iso()
            job["claimed_by"] = _running_owner_tag()
            save_queue(queue)
            return copy.deepcopy(job)
    return None


def recover_stale_running_jobs(*, stale_after_seconds: int = 7200) -> list[dict]:
    recovered = []
    now = datetime.now(timezone.utc)
    with asset_state_lock():
        queue = load_queue()
        changed = False
        for job in queue:
            if job.get("status") != "running":
                continue
            started_at = _parse_iso_datetime(job.get("started_at"))
            if started_at is None or (now - started_at).total_seconds() >= stale_after_seconds:
                job["status"] = "queued"
                job["recovered_at"] = _now_iso()
                job["recovered_reason"] = "missing_started_at" if started_at is None else "stale_running_timeout"
                recovered.append(copy.deepcopy(job))
                changed = True
        if changed:
            save_queue(queue)
    return recovered


def set_queue_job_status(canonical_id: str, attempt_idx: int, status: str, **extra) -> Optional[dict]:
    if status not in QUEUE_STATUSES:
        raise ValueError(f"Invalid queue status: {status}")
    with asset_state_lock():
        queue = load_queue()
        target = None
        for job in queue:
            if job.get("canonical_id") == canonical_id and int(job.get("attempt_idx") or -1) == int(attempt_idx):
                job["status"] = status
                job.update(extra)
                job["updated_at"] = _now_iso()
                target = copy.deepcopy(job)
                break
        if target is not None:
            save_queue(queue)
        return target


def promote_attempt_to_active(canonical_id: str, attempt_idx: int, prompt_entry: dict) -> dict:
    with asset_state_lock():
        registry, entry = ensure_registry_entry(canonical_id)
        new_version_id = attempt_version_id(attempt_idx)
        attempt_dir = version_dir(canonical_id, new_version_id)

        if not attempt_dir.exists():
            raise FileNotFoundError(f"Attempt directory not found: {attempt_dir}")

        prompt_entry = copy.deepcopy(prompt_entry)
        prompts = load_prompts()
        existing_prompt = find_prompt_entry(prompts, canonical_id)
        if existing_prompt is None:
            prompts.append(prompt_entry)
        else:
            existing_prompt.update(prompt_entry)
        save_prompts(prompts)

        copy_map = [
            (attempt_dir / "images" / f"{canonical_id}.png", IMAGES_DIR / f"{canonical_id}.png"),
            (attempt_dir / "images_rgba" / f"{canonical_id}.png", IMAGES_RGBA_DIR / f"{canonical_id}.png"),
            (attempt_dir / "meshes_raw" / f"{canonical_id}.glb", MESHES_RAW_DIR / f"{canonical_id}.glb"),
            (attempt_dir / "meshes" / f"{canonical_id}.glb", MESHES_DIR / f"{canonical_id}.glb"),
        ]
        for src, dst in copy_map:
            _copy_if_exists(src, dst)

        old_active = str(entry.get("active_version") or attempt_version_id(0))
        old_version_entry = _find_version_entry(entry, old_active)
        if old_version_entry is not None:
            old_version_entry["status"] = "deprecated"

        new_version_entry = _find_version_entry(entry, new_version_id)
        if new_version_entry is None:
            new_version_entry = {
                "version_id": new_version_id,
                "created_at": _now_iso(),
                "path": str(attempt_dir),
            }
            entry.setdefault("versions", []).append(new_version_entry)
        new_version_entry["status"] = "active"
        new_version_entry["path"] = str(attempt_dir)
        new_version_entry["promoted_at"] = _now_iso()
        new_version_entry["prompt_path"] = str(attempt_dir / "prompt.json")

        entry["active_version"] = new_version_id
        entry["status"] = "active"
        entry["deprecation_reason"] = None
        entry["stage1_restart_count"] = 0
        entry["replacement_attempts_used"] = max(int(entry.get("replacement_attempts_used") or 0), attempt_idx)
        save_registry(registry)
        return copy.deepcopy(entry)


def record_failed_attempt(
    canonical_id: str,
    attempt_idx: int,
    *,
    reason: str,
    final_failure: bool = False,
) -> dict:
    with asset_state_lock():
        registry, entry = ensure_registry_entry(canonical_id)
        version_id = attempt_version_id(attempt_idx)
        attempt_entry = _find_version_entry(entry, version_id)
        if attempt_entry is None:
            attempt_entry = {
                "version_id": version_id,
                "path": str(version_dir(canonical_id, version_id)),
                "created_at": _now_iso(),
            }
            entry.setdefault("versions", []).append(attempt_entry)
        attempt_entry["status"] = "replacement_failed"
        attempt_entry["failure_reason"] = reason
        attempt_entry["failed_at"] = _now_iso()

        entry["replacement_attempts_used"] = max(int(entry.get("replacement_attempts_used") or 0), attempt_idx)
        if final_failure:
            entry["status"] = "manual_reprompt_required"
        else:
            entry["status"] = "deprecated_pending_replacement"
        entry["deprecation_reason"] = reason
        save_registry(registry)
        return copy.deepcopy(entry)


def enqueue_stage1_restart(
    canonical_id: str,
    *,
    after_attempt_idx: int,
    deprecated_version: str,
    reason: str,
    trigger_source: str,
    max_stage1_restarts: int = 2,
) -> Optional[dict]:
    with asset_state_lock():
        registry, entry = ensure_registry_entry(canonical_id)
        queue = load_queue()
        if any(
            job.get("canonical_id") == canonical_id and job.get("status") in {"queued", "running"}
            for job in queue
        ):
            return None

        restart_count = int(entry.get("stage1_restart_count") or 0)
        if restart_count >= max_stage1_restarts:
            entry["status"] = "manual_reprompt_required"
            entry["deprecation_reason"] = reason
            save_registry(registry)
            return None

        next_restart_idx = restart_count + 1
        next_attempt_idx = after_attempt_idx + 1
        seed_base_stage2, seed_base_stage3 = compute_seed_bases(canonical_id, next_attempt_idx)
        job = {
            "canonical_id": canonical_id,
            "deprecated_version": deprecated_version,
            "reason": reason,
            "trigger_source": trigger_source,
            "attempt_idx": next_attempt_idx,
            "status": "queued",
            "job_kind": "stage1_restart",
            "repair_from_attempt_idx": after_attempt_idx,
            "cycle_attempt_idx": 1,
            "stage1_restart_idx": next_restart_idx,
            "use_geometry_suffix": False,
            "seed_base_stage2": seed_base_stage2,
            "seed_base_stage3": seed_base_stage3,
            "enqueued_at": _now_iso(),
        }
        queue.append(job)
        entry["stage1_restart_count"] = next_restart_idx
        entry["status"] = "deprecated_pending_replacement"
        entry["deprecation_reason"] = f"stage1_restart_pending:{reason}"
        save_registry(registry)
        save_queue(queue)
        return copy.deepcopy(job)
