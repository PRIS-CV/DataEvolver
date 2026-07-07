"""GPU lease planning for long-running DataEvolver jobs.

The planner is intentionally small and deterministic.  It does not try to be a
cluster scheduler; it turns the current GPU state plus stage profiles into a
safe launch plan, so HYWorld-style experiments do not keep launching expensive
jobs onto unsuitable cards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import shlex
from pathlib import Path


@dataclass(frozen=True)
class GPUInfo:
    """One row from `nvidia-smi --query-gpu`."""

    index: int
    memory_used_mib: int
    memory_total_mib: int
    utilization_gpu: int
    uuid: str = ""

    @property
    def free_mib(self) -> int:
        return max(0, self.memory_total_mib - self.memory_used_mib)


@dataclass(frozen=True)
class StageProfile:
    """Resource requirements and retry policy for a pipeline stage."""

    name: str
    gpus_per_job: int = 1
    min_free_mib_per_gpu: int = 0
    max_utilization: int = 15
    max_parallel: int | None = None
    exclusive: bool = True
    notes: str = ""


@dataclass(frozen=True)
class JobRequest:
    """A pending stage shard that needs GPU allocation."""

    job_id: str
    profile_name: str
    priority: int = 100
    preferred_gpus: tuple[int, ...] = ()


@dataclass(frozen=True)
class GPULease:
    """A scheduler decision assigning a job to one or more GPU ids."""

    job_id: str
    profile_name: str
    gpu_ids: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class RejectedJob:
    """A pending job that could not be placed."""

    job_id: str
    profile_name: str
    reason: str


@dataclass(frozen=True)
class GPUPlan:
    leases: tuple[GPULease, ...] = ()
    rejected: tuple[RejectedJob, ...] = ()
    idle_gpu_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class GPUReservation:
    """A GPU excluded from new work for a concrete reason."""

    gpu_id: int
    reason: str


@dataclass(frozen=True)
class ShardSpec:
    """Executable independent shard for dynamic backfill scheduling."""

    shard_id: str
    profile_name: str
    command: tuple[str, ...]
    priority: int = 100
    preferred_gpus: tuple[int, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BackfillPlan:
    """One scheduling cycle: reservations plus leases for runnable shards."""

    reservations: tuple[GPUReservation, ...] = ()
    leases: tuple[GPULease, ...] = ()
    rejected: tuple[RejectedJob, ...] = ()
    skipped_completed: tuple[str, ...] = ()
    idle_gpu_ids: tuple[int, ...] = ()


DEFAULT_STAGE_PROFILES: dict[str, StageProfile] = {
    "hyworld_pano_h100_80gb": StageProfile(
        name="hyworld_pano_h100_80gb",
        gpus_per_job=1,
        min_free_mib_per_gpu=83_000,
        max_utilization=5,
        max_parallel=1,
        notes=(
            "HY-Pano at 1536x768 can peak beyond one 80GB H100 during VAE decode. "
            "If it OOMs near 76-77GB allocated, do not fan out more seeds at the "
            "same resolution; reduce resolution/steps or change stage."
        ),
    ),
    "hyworld_pano_lowres_h100_80gb": StageProfile(
        name="hyworld_pano_lowres_h100_80gb",
        gpus_per_job=1,
        min_free_mib_per_gpu=72_000,
        max_utilization=5,
        max_parallel=1,
        notes=(
            "Use only after a high-resolution HY-Pano OOM. Lower height/width first, "
            "then run one candidate while other GPUs process validation/render shards."
        ),
    ),
    "hyworld_worldgen_2gpu": StageProfile(
        name="hyworld_worldgen_2gpu",
        gpus_per_job=2,
        min_free_mib_per_gpu=72_000,
        max_utilization=10,
        max_parallel=1,
        notes="Stable HYWorld video/worldgen route normally uses a 2-GPU shard.",
    ),
    "blender_render": StageProfile(
        name="blender_render",
        gpus_per_job=1,
        min_free_mib_per_gpu=4_000,
        max_utilization=80,
        max_parallel=None,
        notes="Blender/Xvfb render shards are good fillers for otherwise idle GPUs.",
    ),
    "scene_view_gate": StageProfile(
        name="scene_view_gate",
        gpus_per_job=1,
        min_free_mib_per_gpu=4_000,
        max_utilization=80,
        max_parallel=None,
        notes="Independent HYWorld scene-view gate render shard.",
    ),
    "rotation8_blender": StageProfile(
        name="rotation8_blender",
        gpus_per_job=1,
        min_free_mib_per_gpu=4_000,
        max_utilization=80,
        max_parallel=None,
        notes="Independent object-scene rotation8 Blender shard.",
    ),
    "sr_batch": StageProfile(
        name="sr_batch",
        gpus_per_job=1,
        min_free_mib_per_gpu=12_000,
        max_utilization=80,
        max_parallel=None,
        notes="RealESRGAN/contact-sheet batch shard.",
    ),
    "vlm_service_request": StageProfile(
        name="vlm_service_request",
        gpus_per_job=0,
        min_free_mib_per_gpu=0,
        max_utilization=100,
        max_parallel=None,
        exclusive=False,
        notes="Uses an already-running VLM/vLLM service; no new GPU lease.",
    ),
    "quality_audit_cpu": StageProfile(
        name="quality_audit_cpu",
        gpus_per_job=0,
        min_free_mib_per_gpu=0,
        max_utilization=100,
        max_parallel=None,
        exclusive=False,
        notes="CPU/file-system audit shard; keep it in the unified ledger.",
    ),
    "vlm_reserved": StageProfile(
        name="vlm_reserved",
        gpus_per_job=1,
        min_free_mib_per_gpu=0,
        max_utilization=100,
        max_parallel=1,
        notes="Reserved vLLM/VLM service; do not allocate unless explicitly allowed.",
    ),
}


def _parse_int_token(value: str) -> int:
    match = re.search(r"-?\d+", value)
    if not match:
        raise ValueError(f"expected integer token, got {value!r}")
    return int(match.group(0))


def parse_gpu_spec(spec: str, *, total_gpus: int | None = None) -> tuple[set[int], set[int]]:
    """Parse GPU include/exclude specs.

    Examples:
    - ``"0-6"`` includes GPUs 0..6.
    - ``"0-6,!5"`` includes 0..6 except 5.
    - ``"all,!7"`` requires ``total_gpus`` and excludes GPU 7.
    """

    includes: set[int] = set()
    excludes: set[int] = set()
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        is_exclude = token.startswith("!") or token.startswith("^")
        if is_exclude:
            token = token[1:].strip()
        if token == "all":
            if total_gpus is None:
                raise ValueError("'all' requires total_gpus")
            values = set(range(total_gpus))
        elif "-" in token:
            start_text, end_text = token.split("-", 1)
            start = _parse_int_token(start_text)
            end = _parse_int_token(end_text)
            if end < start:
                raise ValueError(f"invalid GPU range {raw!r}")
            values = set(range(start, end + 1))
        else:
            values = {_parse_int_token(token)}
        if is_exclude:
            excludes.update(values)
        else:
            includes.update(values)
    return includes, excludes


def parse_nvidia_smi_gpu_csv(text: str) -> list[GPUInfo]:
    """Parse common `nvidia-smi --query-gpu` CSV rows.

    Accepts rows like ``0, 42383 MiB, 81559 MiB, 78 %`` or
    ``0, NVIDIA H100, 42383 MiB, 81559 MiB, 78 %, P0``.
    """

    gpus: list[GPUInfo] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("index"):
            continue
        parts = [part.strip() for part in stripped.split(",")]
        numbers = [_parse_int_token(part) for part in parts if re.search(r"-?\d+", part) and not part.startswith("GPU-")]
        if len(numbers) < 4:
            continue
        index = numbers[0]
        uuid = next((part for part in parts if part.startswith("GPU-")), "")
        # Query variants may include GPU name with no number, or pstate P0 at the end.
        memory_used = numbers[-3]
        memory_total = numbers[-2]
        utilization = numbers[-1]
        if parts[-1].upper().startswith("P") and len(numbers) >= 5:
            memory_used = numbers[-4]
            memory_total = numbers[-3]
            utilization = numbers[-2]
        gpus.append(
            GPUInfo(
                index=index,
                memory_used_mib=memory_used,
                memory_total_mib=memory_total,
                utilization_gpu=utilization,
                uuid=uuid,
            )
        )
    return gpus


def available_gpu_ids(
    gpus: list[GPUInfo],
    profile: StageProfile,
    *,
    include: set[int] | None = None,
    exclude: set[int] | None = None,
    already_allocated: set[int] | None = None,
) -> list[int]:
    include = include if include is not None else {gpu.index for gpu in gpus}
    exclude = exclude or set()
    already_allocated = already_allocated or set()
    ids = []
    for gpu in sorted(gpus, key=lambda item: item.index):
        if gpu.index not in include or gpu.index in exclude or gpu.index in already_allocated:
            continue
        if gpu.free_mib < profile.min_free_mib_per_gpu:
            continue
        if gpu.utilization_gpu > profile.max_utilization:
            continue
        ids.append(gpu.index)
    return ids


def plan_gpu_leases(
    gpus: list[GPUInfo],
    jobs: list[JobRequest],
    *,
    include_spec: str = "all",
    exclude_spec: str = "",
    profiles: dict[str, StageProfile] | None = None,
) -> GPUPlan:
    """Allocate jobs to GPUs according to profile limits.

    Jobs are placed by ascending ``priority``.  A GPU is never leased twice in
    the same plan.  Profile ``max_parallel`` prevents the common HYWorld failure
    mode where many high-memory Pano candidates are launched just to all OOM at
    decode.
    """

    profiles = profiles or DEFAULT_STAGE_PROFILES
    all_gpu_ids = {gpu.index for gpu in gpus}
    includes, inline_excludes = parse_gpu_spec(include_spec, total_gpus=max(all_gpu_ids) + 1 if all_gpu_ids else 0)
    explicit_excludes = parse_gpu_spec(exclude_spec, total_gpus=max(all_gpu_ids) + 1 if all_gpu_ids else 0)[0] if exclude_spec else set()
    include = includes or all_gpu_ids
    exclude = inline_excludes | explicit_excludes
    allocated: set[int] = set()
    profile_counts: dict[str, int] = {}
    leases: list[GPULease] = []
    rejected: list[RejectedJob] = []

    for job in sorted(jobs, key=lambda item: (item.priority, item.job_id)):
        profile = profiles.get(job.profile_name)
        if profile is None:
            rejected.append(RejectedJob(job.job_id, job.profile_name, "unknown profile"))
            continue
        count = profile_counts.get(profile.name, 0)
        if profile.max_parallel is not None and count >= profile.max_parallel:
            rejected.append(RejectedJob(job.job_id, profile.name, f"profile max_parallel={profile.max_parallel} reached"))
            continue
        candidate_ids = available_gpu_ids(
            gpus,
            profile,
            include=set(job.preferred_gpus) if job.preferred_gpus else include,
            exclude=exclude,
            already_allocated=allocated if profile.exclusive else set(),
        )
        if len(candidate_ids) < profile.gpus_per_job:
            rejected.append(
                RejectedJob(
                    job.job_id,
                    profile.name,
                    f"need {profile.gpus_per_job} GPU(s) with >= {profile.min_free_mib_per_gpu} MiB free",
                )
            )
            continue
        gpu_ids = tuple(candidate_ids[: profile.gpus_per_job])
        allocated.update(gpu_ids)
        profile_counts[profile.name] = count + 1
        leases.append(GPULease(job.job_id, profile.name, gpu_ids, profile.notes))

    idle = tuple(sorted((include & all_gpu_ids) - exclude - allocated))
    return GPUPlan(tuple(leases), tuple(rejected), idle)


def infer_gpu_reservations(
    gpus: list[GPUInfo],
    *,
    include_spec: str = "all",
    exclude_spec: str = "",
    reserve_spec: str = "",
    busy_memory_mib: int = 60_000,
    busy_utilization: int = 90,
) -> tuple[GPUReservation, ...]:
    """Infer GPUs that should not receive new shards in this cycle.

    This intentionally uses GPU-level state instead of process names as the
    source of truth.  `nvidia-smi --query-compute-apps` can report UUIDs without
    stable GPU indices on some systems; memory/utilization thresholds are enough
    to protect vLLM and unknown busy jobs from accidental reuse.
    """

    all_gpu_ids = {gpu.index for gpu in gpus}
    total = max(all_gpu_ids) + 1 if all_gpu_ids else 0
    includes, inline_excludes = parse_gpu_spec(include_spec, total_gpus=total)
    explicit_excludes = parse_gpu_spec(exclude_spec, total_gpus=total)[0] if exclude_spec else set()
    explicit_reserve = parse_gpu_spec(reserve_spec, total_gpus=total)[0] if reserve_spec else set()
    include = includes or all_gpu_ids
    reservations: dict[int, GPUReservation] = {}

    for gpu in gpus:
        if gpu.index not in include:
            continue
        if gpu.index in inline_excludes | explicit_excludes:
            reservations[gpu.index] = GPUReservation(gpu.index, "excluded by gpu spec")
            continue
        if gpu.index in explicit_reserve:
            reservations[gpu.index] = GPUReservation(gpu.index, "explicitly reserved")
            continue
        if gpu.memory_used_mib >= busy_memory_mib:
            reservations[gpu.index] = GPUReservation(
                gpu.index,
                f"busy memory {gpu.memory_used_mib} MiB >= {busy_memory_mib} MiB",
            )
            continue
        if gpu.utilization_gpu >= busy_utilization:
            reservations[gpu.index] = GPUReservation(
                gpu.index,
                f"busy utilization {gpu.utilization_gpu}% >= {busy_utilization}%",
            )
    return tuple(sorted(reservations.values(), key=lambda item: item.gpu_id))


def completed_shard_ids(shards: list[ShardSpec], *, base_dir: str | Path | None = None) -> set[str]:
    """Return shards whose declared expected outputs already exist."""

    root = Path(base_dir) if base_dir else None
    completed: set[str] = set()
    for shard in shards:
        if not shard.expected_outputs:
            continue
        paths = [Path(output) if root is None else root / output for output in shard.expected_outputs]
        if all(path.exists() for path in paths):
            completed.add(shard.shard_id)
    return completed


def plan_backfill_cycle(
    gpus: list[GPUInfo],
    shards: list[ShardSpec],
    *,
    include_spec: str = "all",
    exclude_spec: str = "",
    reserve_spec: str = "",
    busy_memory_mib: int = 60_000,
    busy_utilization: int = 90,
    profiles: dict[str, StageProfile] | None = None,
    base_dir: str | Path | None = None,
) -> BackfillPlan:
    """Plan one dynamic backfill cycle for independent shards."""

    reservations = infer_gpu_reservations(
        gpus,
        include_spec=include_spec,
        exclude_spec=exclude_spec,
        reserve_spec=reserve_spec,
        busy_memory_mib=busy_memory_mib,
        busy_utilization=busy_utilization,
    )
    completed = completed_shard_ids(shards, base_dir=base_dir)
    pending_jobs = [
        JobRequest(
            job_id=shard.shard_id,
            profile_name=shard.profile_name,
            priority=shard.priority,
            preferred_gpus=shard.preferred_gpus,
        )
        for shard in shards
        if shard.shard_id not in completed
    ]
    reservation_exclude = ",".join(str(item.gpu_id) for item in reservations)
    combined_exclude = ",".join(part for part in [exclude_spec, reservation_exclude] if part)
    gpu_plan = plan_gpu_leases(
        gpus,
        pending_jobs,
        include_spec=include_spec,
        exclude_spec=combined_exclude,
        profiles=profiles,
    )
    return BackfillPlan(
        reservations=reservations,
        leases=gpu_plan.leases,
        rejected=gpu_plan.rejected,
        skipped_completed=tuple(sorted(completed)),
        idle_gpu_ids=gpu_plan.idle_gpu_ids,
    )


def shard_command_display(shard: ShardSpec) -> str:
    """Shell-safe display string for logs and dry-runs."""

    return " ".join(shlex.quote(part) for part in shard.command)


def shard_from_mapping(payload: dict[str, object]) -> ShardSpec:
    """Parse a shard JSON object into a typed spec."""

    raw_command = payload.get("command")
    if not isinstance(raw_command, list) or not raw_command:
        raise ValueError("shard command must be a non-empty list")
    env_payload = payload.get("env") or {}
    if not isinstance(env_payload, dict):
        raise ValueError("shard env must be an object")
    return ShardSpec(
        shard_id=str(payload.get("id") or payload.get("shard_id") or ""),
        profile_name=str(payload.get("profile") or payload.get("profile_name") or ""),
        command=tuple(str(item) for item in raw_command),
        priority=int(payload.get("priority", 100)),
        preferred_gpus=tuple(int(item) for item in payload.get("preferred_gpus", []) or []),
        expected_outputs=tuple(str(item) for item in payload.get("expected_outputs", []) or []),
        cwd=str(payload.get("cwd") or ""),
        env={str(key): str(value) for key, value in env_payload.items()},
    )


def classify_failure(stdout_tail: str) -> str:
    """Classify common long-run failures into scheduler actions."""

    text = stdout_tail.lower()
    if "cuda out of memory" in text or "torch.outofmemoryerror" in text:
        if "panogen" in text or "autoencoder_kl_3d" in text:
            return "hyworld_pano_oom"
        return "cuda_oom"
    if "blender: command not found" in text:
        return "blender_missing"
    if "worldmirror reconstruction has only 0 aligned frames" in text:
        return "worldmirror_viewmap_mismatch"
    if "nccl" in text and ("all-gather" in text or "all_gather" in text or "invalid usage" in text):
        return "nccl_collective_failure"
    return "unknown"


def recommended_action(failure_class: str) -> str:
    actions = {
        "hyworld_pano_oom": (
            "Do not launch more same-resolution HY-Pano seed shards. Retry one shard after lowering "
            "height/width or using a lower-memory decode path, then fill idle GPUs with Blender, "
            "validation, or object insertion shards."
        ),
        "cuda_oom": "Release the GPU lease, lower memory pressure, and retry only after a fresh nvidia-smi check.",
        "blender_missing": "Resolve Blender from production profile or BLENDER_BIN before starting scene gates.",
        "worldmirror_viewmap_mismatch": "Rebuild scene gate with frame prefixes that match the actual pano_bank image names.",
        "nccl_collective_failure": (
            "Do not increase nproc blindly. Use one process per GPU, set torchrun env cleanly, and "
            "fall back to the smallest stable shard before scaling."
        ),
        "unknown": "Preserve logs as failure evidence and require manual classification before relaunch.",
    }
    return actions.get(failure_class, actions["unknown"])
