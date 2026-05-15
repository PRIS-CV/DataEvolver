from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
PIPELINE_ROOT = REPO_ROOT / "pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from asset_lifecycle import (
    DATA_DIR,
    MESHES_DIR,
    attempt_version_id,
    claim_next_queued_job,
    compute_seed_bases,
    create_attempt_prompt_snapshot,
    detect_asset_viability,
    detect_verdict_from_review,
    enqueue_replacement_attempt,
    enqueue_stage1_restart,
    ensure_registry_entry,
    find_prompt_entry,
    load_queue,
    load_prompts,
    promote_attempt_to_active,
    record_failed_attempt,
    recover_stale_running_jobs,
    save_queue,
    set_queue_job_status,
    version_dir,
)


DEFAULT_BLENDER = os.environ.get("BLENDER_BIN", "blender")
DEFAULT_PROFILE = REPO_ROOT / "configs" / "dataset_profiles" / "scene_v7.json"
DEFAULT_SCENE_TEMPLATE = REPO_ROOT / "configs" / "scene_template.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Process regeneration_queue.json and build replacement assets")
    parser.add_argument("--max-jobs", type=int, default=1, help="Maximum queued jobs to process in this run")
    parser.add_argument("--max-attempts-per-object", type=int, default=2)
    parser.add_argument("--max-stage1-restarts-per-object", type=int, default=8)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--generation-device", default="cuda:0")
    parser.add_argument("--smoke-device", default="cuda:2")
    parser.add_argument(
        "--smoke-visible-gpus",
        default=None,
        help="Comma-separated physical GPU ids exposed to the smoke-gate reviewer. Defaults to the smoke GPU plus a sibling GPU when available.",
    )
    parser.add_argument("--blender-bin", default=DEFAULT_BLENDER)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--scene-template", default=str(DEFAULT_SCENE_TEMPLATE))
    parser.add_argument("--idle-memory-mb", type=int, default=10000)
    parser.add_argument("--stale-running-seconds", type=int, default=7200)
    parser.add_argument("--skip-gpu-idle-check", action="store_true")
    parser.add_argument("--allow-unsafe-single-gpu-review", action="store_true")
    parser.add_argument(
        "--stage1-mode",
        choices=("auto", "api", "template"),
        default="auto",
        help="How stage1 restart should regenerate prompts. 'auto' uses Anthropic API when available, otherwise scene-conditioned templates.",
    )
    parser.add_argument("--stage1-model", default="claude-haiku-4-5")
    return parser.parse_args()


def _read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _run_command(cmd: list[str], *, env: Optional[dict] = None) -> int:
    print("[regen]", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)


def _extract_gpu_index(device: str) -> Optional[int]:
    if not device.startswith("cuda:"):
        return None
    try:
        return int(device.split(":", 1)[1])
    except Exception:
        return None


def _parse_visible_gpu_ids(raw: Optional[str]) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _query_gpu_memory_used() -> dict[int, int]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return {}
    usage = {}
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        usage[int(parts[0])] = int(parts[1])
    return usage


def _gpu_ids_are_idle(gpu_ids: list[str], max_used_mb: int) -> bool:
    if not gpu_ids:
        return True
    usage = _query_gpu_memory_used()
    if not usage:
        return False
    for gpu_id in gpu_ids:
        try:
            gpu_index = int(gpu_id)
        except Exception:
            return False
        if usage.get(gpu_index, max_used_mb + 1) >= max_used_mb:
            return False
    return True


def _resolve_smoke_visible_gpus(args) -> list[str]:
    requested = _parse_visible_gpu_ids(args.smoke_visible_gpus)
    if requested:
        return requested

    primary_index = _extract_gpu_index(args.smoke_device)
    if primary_index is None:
        return []

    usage = _query_gpu_memory_used()
    if not usage:
        return [str(primary_index)]

    available = sorted(usage)
    if primary_index not in available:
        return [str(primary_index)]

    sibling = None
    for gpu_index in available:
        if gpu_index > primary_index:
            sibling = gpu_index
            break
    if sibling is None:
        for gpu_index in reversed(available):
            if gpu_index != primary_index:
                sibling = gpu_index
                break

    visible = [str(primary_index)]
    if sibling is not None:
        visible.append(str(sibling))
    return visible


def _resolve_generation_gpu_ids(args) -> list[str]:
    logical_index = _extract_gpu_index(args.generation_device)
    if logical_index is None:
        return []

    visible = _parse_visible_gpu_ids(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if visible:
        if 0 <= logical_index < len(visible):
            return [visible[logical_index]]
        return []
    return [str(logical_index)]


def _ensure_attempt_dirs(attempt_dir: Path):
    for subdir in (
        "images",
        "images_rgba",
        "meshes_raw",
        "meshes",
        "mesh_qc",
        "smoke",
    ):
        (attempt_dir / subdir).mkdir(parents=True, exist_ok=True)


def _write_smoke_scene_template(base_template_path: Path, target_path: Path):
    template = _read_json(base_template_path, {})
    template["qc_views"] = [[0, 0]]
    _write_json(target_path, template)


def _find_queue_job(canonical_id: str, attempt_idx: int) -> Optional[dict]:
    for job in load_queue():
        if job.get("canonical_id") == canonical_id and int(job.get("attempt_idx") or -1) == int(attempt_idx):
            return job
    return None


def _load_current_prompt_entry(canonical_id: str) -> dict:
    prompt_entry = find_prompt_entry(load_prompts(), canonical_id)
    if isinstance(prompt_entry, dict):
        return copy.deepcopy(prompt_entry)
    return {
        "id": canonical_id,
        "name": canonical_id,
        "category": "unknown",
        "prompt": "",
        "features": {},
    }


def _load_latest_trace_text(reviews_dir: Path, canonical_id: str) -> str:
    if not reviews_dir.exists():
        return ""
    trace_candidates = sorted(reviews_dir.glob(f"{canonical_id}_r00_*_trace.json"))
    if not trace_candidates:
        return ""
    trace_payload = _read_json(trace_candidates[-1], {})
    attempts = trace_payload.get("attempts") or []
    if attempts:
        text = attempts[-1].get("assistant_text")
        if isinstance(text, str):
            return text
    for key in ("assistant_text", "raw_text", "response_text", "content", "answer", "model_output"):
        value = trace_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_bullet_section(text: str, heading: str) -> list[str]:
    lines = str(text or "").splitlines()
    target = f"{heading.strip().lower()}:"
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == target:
            # Prefer the last matching heading because freeform traces may include
            # multiple drafts before the final structured answer.
            start_idx = idx + 1
    if start_idx is None:
        return []

    items = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            if items:
                break
            continue
        if re.match(r"^[A-Za-z][A-Za-z _-]+:\s*$", stripped):
            break
        if stripped.startswith(("-", "*")):
            stripped = stripped[1:].strip()
        items.append(stripped)
    return [item for item in items if item]


def _derive_suggested_fixes(trace_text: str, smoke_agg: dict) -> list[str]:
    direct = _extract_bullet_section(trace_text, "Suggested fixes")
    if direct:
        return direct

    actions = [str(item).strip() for item in (smoke_agg.get("suggested_actions") or []) if str(item).strip()]
    if actions:
        return [f"Consider action {action}" for action in actions[:4]]

    fallback_sentences = []
    sentence_candidates = re.split(r"(?<=[.!?])\s+", str(trace_text or ""))
    for sentence in sentence_candidates:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in ("suggest", "recommend", "should", "could", "needs to", "need to")):
            cleaned = sentence.strip(" -*")
            if cleaned:
                fallback_sentences.append(cleaned)
        if len(fallback_sentences) >= 4:
            break
    return fallback_sentences


def _build_stage1_repair_spec(canonical_id: str, job: dict, attempt_dir: Path) -> Path:
    raw_repair_from_attempt_idx = job.get("repair_from_attempt_idx")
    if raw_repair_from_attempt_idx is None:
        repair_from_attempt_idx = int(job.get("attempt_idx") or 1) - 1
    else:
        repair_from_attempt_idx = int(raw_repair_from_attempt_idx)
    source_attempt_dir = version_dir(canonical_id, attempt_version_id(repair_from_attempt_idx))

    prompt_entry = _read_json(source_attempt_dir / "prompt.json", None)
    if not isinstance(prompt_entry, dict):
        prompt_entry = _load_current_prompt_entry(canonical_id)

    smoke_result = _read_json(source_attempt_dir / "smoke_result.json", {})
    smoke_reviews_dir = source_attempt_dir / "smoke" / "reviews"
    smoke_agg = _read_json(smoke_reviews_dir / f"{canonical_id}_r00_agg.json", {})
    trace_text = _load_latest_trace_text(smoke_reviews_dir, canonical_id)
    suggested_fixes = _derive_suggested_fixes(trace_text, smoke_agg)

    failure_summary = {
        "reason": str(job.get("reason") or smoke_result.get("reason") or ""),
        "detected_verdict": smoke_result.get("detected_verdict") or smoke_agg.get("hybrid_route"),
        "asset_viability": smoke_result.get("asset_viability") or smoke_agg.get("asset_viability"),
        "hybrid_score": smoke_result.get("hybrid_score") if smoke_result.get("hybrid_score") is not None else smoke_agg.get("hybrid_score"),
        "structure_consistency": smoke_result.get("structure_consistency") or smoke_agg.get("structure_consistency"),
        "issue_tags": smoke_agg.get("issue_tags") or [],
        "lighting_diagnosis": smoke_agg.get("lighting_diagnosis"),
        "abandon_reason": smoke_result.get("abandon_reason") or smoke_agg.get("abandon_reason"),
        "major_issues": _extract_bullet_section(trace_text, "Major issues"),
        "suggested_fixes": suggested_fixes,
        "trace_text_excerpt": " ".join(str(trace_text or "").split())[:1200],
        "source_attempt_idx": repair_from_attempt_idx,
    }
    repair_spec = {
        "id": prompt_entry.get("id") or canonical_id,
        "name": prompt_entry.get("name") or canonical_id,
        "category": prompt_entry.get("category") or "unknown",
        "previous_prompt": prompt_entry.get("prompt") or "",
        "previous_features": prompt_entry.get("features") or {},
        "failure_summary": failure_summary,
    }
    repair_spec_path = attempt_dir / "stage1_repair_spec.json"
    _write_json(repair_spec_path, repair_spec)
    return repair_spec_path


def _enqueue_followup_attempt(
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
):
    enqueue_replacement_attempt(
        canonical_id,
        attempt_idx=attempt_idx,
        deprecated_version=deprecated_version,
        reason=reason,
        trigger_source=trigger_source,
        cycle_attempt_idx=cycle_attempt_idx,
        stage1_restart_idx=stage1_restart_idx,
        use_geometry_suffix=use_geometry_suffix,
        job_kind=job_kind,
        prompt_source_attempt_idx=prompt_source_attempt_idx,
    )


def _run_stage1_restart(args, canonical_id: str, attempt_dir: Path, job: dict) -> dict:
    stage1_output = attempt_dir / "stage1_prompts.json"
    repair_spec_path = _build_stage1_repair_spec(canonical_id, job, attempt_dir)
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "pipeline" / "stage1_text_expansion.py"),
        "--repair-spec-file",
        str(repair_spec_path),
        "--output-file",
        str(stage1_output),
    ]
    use_api = args.stage1_mode == "api" or (
        args.stage1_mode == "auto"
        and bool(os.environ.get("ANTHROPIC_API_KEY"))
    )
    if use_api:
        cmd.extend(["--model", args.stage1_model])
    else:
        cmd.extend(["--scene-conditioned", "--template-only"])
    rc = _run_command(cmd)
    if rc != 0:
        raise RuntimeError(f"stage1_restart_failed:rc={rc}")
    payload = _read_json(stage1_output, [])
    if not payload:
        raise RuntimeError("stage1_restart_failed:empty_prompt_output")
    return payload[0]


def _prepare_attempt_inputs(args, canonical_id: str, attempt_idx: int, job: dict) -> tuple[Path, dict, Path]:
    attempt_dir = version_dir(canonical_id, attempt_version_id(attempt_idx))
    _ensure_attempt_dirs(attempt_dir)
    job_kind = str(job.get("job_kind") or "replacement")
    use_geometry_suffix = bool(job.get("use_geometry_suffix"))
    prompt_entry = None
    if job_kind == "stage1_restart":
        prompt_entry = _run_stage1_restart(args, canonical_id, attempt_dir, job)
    else:
        prompt_source_attempt_idx = job.get("prompt_source_attempt_idx")
        if prompt_source_attempt_idx is not None:
            source_prompt_path = (
                version_dir(canonical_id, attempt_version_id(int(prompt_source_attempt_idx))) / "prompt.json"
            )
            if source_prompt_path.exists():
                prompt_entry = _read_json(source_prompt_path, None)
    prompts_path, prompt_entry = create_attempt_prompt_snapshot(
        canonical_id,
        attempt_idx,
        prompt_entry=prompt_entry,
        add_geometry_suffix=use_geometry_suffix,
    )
    _write_json(
        attempt_dir / "attempt_metadata.json",
        {
            "canonical_id": canonical_id,
            "attempt_idx": attempt_idx,
            "job_kind": job_kind,
            "cycle_attempt_idx": int(job.get("cycle_attempt_idx") or 1),
            "stage1_restart_idx": int(job.get("stage1_restart_idx") or 0),
            "repair_from_attempt_idx": job.get("repair_from_attempt_idx"),
            "use_geometry_suffix": use_geometry_suffix,
            "prompt_source_attempt_idx": job.get("prompt_source_attempt_idx"),
            "prompt": prompt_entry.get("prompt"),
        },
    )
    return prompts_path, prompt_entry, attempt_dir


def _run_replacement_pipeline(args, canonical_id: str, attempt_idx: int, job: dict, attempt_dir: Path, prompts_path: Path) -> int:
    seed_base_stage2, seed_base_stage3 = compute_seed_bases(canonical_id, attempt_idx)
    seed_base_stage2 = int(job.get("seed_base_stage2") or seed_base_stage2)
    seed_base_stage3 = int(job.get("seed_base_stage3") or seed_base_stage3)
    stage2_rc = _run_command(
        [
            args.python_bin,
            str(REPO_ROOT / "pipeline" / "stage2_t2i_generate.py"),
            "--device",
            args.generation_device,
            "--prompts-path",
            str(prompts_path),
            "--output-dir",
            str(attempt_dir / "images"),
            "--ids",
            canonical_id,
            "--seed-base",
            str(seed_base_stage2),
            "--no-skip",
        ]
    )
    if stage2_rc != 0:
        return stage2_rc

    stage25_rc = _run_command(
        [
            args.python_bin,
            str(REPO_ROOT / "pipeline" / "stage2_5_sam2_segment.py"),
            "--device",
            args.generation_device,
            "--ids",
            canonical_id,
            "--input-dir",
            str(attempt_dir / "images"),
            "--output-dir",
            str(attempt_dir / "images_rgba"),
            "--no-skip",
        ]
    )
    if stage25_rc != 0:
        return stage25_rc

    stage3_rc = _run_command(
        [
            args.python_bin,
            str(REPO_ROOT / "pipeline" / "stage3_image_to_3d.py"),
            "--device",
            args.generation_device,
            "--ids",
            canonical_id,
            "--images-dir",
            str(attempt_dir / "images_rgba"),
            "--output-dir",
            str(attempt_dir / "meshes_raw"),
            "--seed-base",
            str(seed_base_stage3),
            "--no-skip",
        ]
    )
    if stage3_rc != 0:
        return stage3_rc

    raw_mesh_path = attempt_dir / "meshes_raw" / f"{canonical_id}.glb"
    mesh_path = attempt_dir / "meshes" / f"{canonical_id}.glb"
    if not raw_mesh_path.exists():
        return 1

    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_mesh_path, mesh_path)
    _write_json(
        attempt_dir / "mesh_qc" / f"{canonical_id}_qc.json",
        {
            "canonical_id": canonical_id,
            "status": "skipped_stage3_5",
            "stage3_5_removed": True,
            "source_mesh": str(raw_mesh_path),
            "materialized_mesh": str(mesh_path),
        },
    )
    return 0


def _evaluate_smoke_gate(smoke_root: Path, canonical_id: str) -> tuple[bool, dict]:
    agg_path = smoke_root / "reviews" / f"{canonical_id}_r00_agg.json"
    agent_path = smoke_root / "agent_round00.json"
    if not agg_path.exists():
        agent_result = _read_json(agent_path, {}) if agent_path.exists() else {}
        return False, {
            "reason": "missing_smoke_review_agg",
            "agent_result": agent_result,
        }

    agg = _read_json(agg_path, {})
    trace_candidates = sorted((smoke_root / "reviews").glob(f"{canonical_id}_r00_*_trace.json"))
    trace_text = ""
    if trace_candidates:
        trace_payload = _read_json(trace_candidates[0], {})
        attempts = trace_payload.get("attempts") or []
        if attempts:
            trace_text = str(attempts[-1].get("assistant_text") or "")
    verdict = detect_verdict_from_review(trace_text, agg)
    asset_viability, abandon_reason, abandon_confidence = detect_asset_viability(trace_text, agg)
    hybrid_score = float(agg.get("hybrid_score") or 0.0)
    structure_consistency = str(agg.get("structure_consistency") or "good")

    passed = (
        structure_consistency != "major_mismatch"
        and asset_viability != "abandon"
        and (verdict != "reject" or hybrid_score >= 0.55)
    )
    return passed, {
        "agg_path": str(agg_path),
        "trace_text_excerpt": " ".join(trace_text.split())[:500],
        "detected_verdict": verdict,
        "asset_viability": asset_viability,
        "abandon_reason": abandon_reason,
        "abandon_confidence": abandon_confidence,
        "hybrid_score": hybrid_score,
        "structure_consistency": structure_consistency,
    }


def _run_smoke_gate(args, canonical_id: str, attempt_dir: Path, smoke_visible_gpus: list[str]) -> tuple[bool, dict]:
    smoke_dir = attempt_dir / "smoke"
    smoke_scene_template = attempt_dir / "smoke_scene_template.json"
    _write_smoke_scene_template(Path(args.scene_template), smoke_scene_template)

    env = os.environ.copy()
    if smoke_visible_gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(smoke_visible_gpus)
    if len(smoke_visible_gpus) > 1:
        env["VLM_ALLOW_BALANCED"] = "1"
        env.pop("VLM_FORCE_SINGLE_GPU", None)
    elif args.allow_unsafe_single_gpu_review:
        env["VLM_FORCE_SINGLE_GPU"] = "1"
    smoke_rc = _run_command(
        [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "run_scene_agent_step.py"),
            "--output-dir",
            str(smoke_dir),
            "--obj-id",
            canonical_id,
            "--rotation-deg",
            "0",
            "--round-idx",
            "0",
            "--device",
            "cuda:0",
            "--blender",
            args.blender_bin,
            "--meshes-dir",
            str(attempt_dir / "meshes"),
            "--profile",
            args.profile,
            "--scene-template",
            str(smoke_scene_template),
        ],
        env=env,
    )
    passed, smoke_payload = _evaluate_smoke_gate(smoke_dir, canonical_id)
    smoke_payload["return_code"] = smoke_rc
    smoke_payload["smoke_visible_gpus"] = smoke_visible_gpus
    return passed, smoke_payload


def _mark_attempt_promoted(canonical_id: str, attempt_idx: int, smoke_payload: dict):
    set_queue_job_status(
        canonical_id,
        attempt_idx,
        "promoted",
        promoted_review_agg=smoke_payload.get("agg_path"),
        smoke_result=smoke_payload,
    )


def _mark_attempt_failed(
    args,
    job: dict,
    canonical_id: str,
    attempt_idx: int,
    *,
    deprecated_version: str,
    reason: str,
    max_attempts: int,
):
    cycle_attempt_idx = int(job.get("cycle_attempt_idx") or 1)
    stage1_restart_idx = int(job.get("stage1_restart_idx") or 0)
    exhausted_cycle = cycle_attempt_idx >= max_attempts
    should_restart_from_stage1 = exhausted_cycle and stage1_restart_idx < args.max_stage1_restarts_per_object

    entry = record_failed_attempt(
        canonical_id,
        attempt_idx,
        reason=reason,
        final_failure=exhausted_cycle and not should_restart_from_stage1,
    )
    if not exhausted_cycle:
        set_queue_job_status(canonical_id, attempt_idx, "smoke_failed", failure_reason=reason)
        _enqueue_followup_attempt(
            canonical_id,
            attempt_idx=attempt_idx + 1,
            deprecated_version=deprecated_version,
            reason=reason,
            trigger_source=f"{canonical_id}:attempt_{attempt_idx:02d}_failed",
            cycle_attempt_idx=cycle_attempt_idx + 1,
            stage1_restart_idx=stage1_restart_idx,
            use_geometry_suffix=True,
            prompt_source_attempt_idx=attempt_idx if str(job.get("job_kind") or "replacement") == "stage1_restart" else job.get("prompt_source_attempt_idx"),
        )
    elif should_restart_from_stage1:
        set_queue_job_status(canonical_id, attempt_idx, "exhausted", failure_reason=reason)
        enqueue_stage1_restart(
            canonical_id,
            after_attempt_idx=attempt_idx,
            deprecated_version=deprecated_version,
            reason=reason,
            trigger_source=f"{canonical_id}:attempt_{attempt_idx:02d}_failed",
            max_stage1_restarts=args.max_stage1_restarts_per_object,
        )
    else:
        set_queue_job_status(canonical_id, attempt_idx, "exhausted", failure_reason=reason)
    return entry


def _process_job(args, job: dict):
    canonical_id = str(job["canonical_id"])
    attempt_idx = int(job["attempt_idx"])
    deprecated_version = str(job.get("deprecated_version") or attempt_version_id(0))
    try:
        smoke_visible_gpus = _resolve_smoke_visible_gpus(args)
        if len(smoke_visible_gpus) < 2 and not args.allow_unsafe_single_gpu_review:
            print(
                f"[regen] smoke gate for {canonical_id} requires at least 2 visible GPUs; "
                f"resolved={smoke_visible_gpus or ['<none>']}. Leaving job queued."
            )
            set_queue_job_status(
                canonical_id,
                attempt_idx,
                "queued",
                requeued_reason="insufficient_smoke_visible_gpus",
            )
            return
        if not args.skip_gpu_idle_check and not _gpu_ids_are_idle(smoke_visible_gpus, args.idle_memory_mb):
            print(f"[regen] smoke gate GPUs busy for {canonical_id}, leaving job queued: {smoke_visible_gpus}")
            set_queue_job_status(
                canonical_id,
                attempt_idx,
                "queued",
                requeued_reason="smoke_gpus_busy",
            )
            return

        set_queue_job_status(
            canonical_id,
            attempt_idx,
            "running",
            runner_pid=os.getpid(),
            runner_host=os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME"),
        )
        prompts_path, prompt_entry, attempt_dir = _prepare_attempt_inputs(args, canonical_id, attempt_idx, job)

        pipeline_rc = _run_replacement_pipeline(args, canonical_id, attempt_idx, job, attempt_dir, prompts_path)
        if pipeline_rc != 0:
            _mark_attempt_failed(
                args,
                job,
                canonical_id,
                attempt_idx,
                deprecated_version=deprecated_version,
                reason=f"replacement_pipeline_failed:rc={pipeline_rc}",
                max_attempts=args.max_attempts_per_object,
            )
            return

        passed, smoke_payload = _run_smoke_gate(args, canonical_id, attempt_dir, smoke_visible_gpus)
        _write_json(attempt_dir / "smoke_result.json", smoke_payload)
        if not passed:
            _mark_attempt_failed(
                args,
                job,
                canonical_id,
                attempt_idx,
                deprecated_version=deprecated_version,
                reason=f"smoke_gate_failed:{smoke_payload.get('reason') or smoke_payload.get('detected_verdict')}",
                max_attempts=args.max_attempts_per_object,
            )
            return

        promote_attempt_to_active(canonical_id, attempt_idx, prompt_entry)
        _mark_attempt_promoted(canonical_id, attempt_idx, smoke_payload)
    except Exception as exc:
        _mark_attempt_failed(
            args,
            job,
            canonical_id,
            attempt_idx,
            deprecated_version=deprecated_version,
            reason=f"runner_exception:{exc}",
            max_attempts=args.max_attempts_per_object,
        )


def main():
    args = parse_args()
    recovered = recover_stale_running_jobs(stale_after_seconds=args.stale_running_seconds)
    if recovered:
        print(
            "[regen] recovered stale running jobs:",
            ", ".join(f"{job.get('canonical_id')}:{job.get('attempt_idx')}" for job in recovered),
        )

    smoke_visible_gpus = _resolve_smoke_visible_gpus(args)
    required_gpu_ids = set(smoke_visible_gpus)
    required_gpu_ids.update(_resolve_generation_gpu_ids(args))

    processed = 0
    while processed < args.max_jobs:
        if len(smoke_visible_gpus) < 2 and not args.allow_unsafe_single_gpu_review:
            print(f"[regen] insufficient smoke-visible GPUs: {smoke_visible_gpus or ['<none>']}")
            break
        if not args.skip_gpu_idle_check and not _gpu_ids_are_idle(sorted(required_gpu_ids), args.idle_memory_mb):
            print(f"[regen] required GPUs busy, skipping claim: {sorted(required_gpu_ids)}")
            break

        job = claim_next_queued_job()
        if not job:
            break

        _process_job(args, job)
        processed += 1


if __name__ == "__main__":
    main()
