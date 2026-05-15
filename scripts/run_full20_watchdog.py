#!/usr/bin/env python3
"""
Persistent watchdog for the full20 scene loop.

It keeps three long-running pieces alive without relying on the interactive
session:
1. scene workers on idle scene GPUs
2. regeneration daemon on the regen GPU
3. export watcher after scene + regen work drains
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_ROOT_DIR = REPO_ROOT / "pipeline" / "data" / "evolution_scene_v7_full20_rotation4_agent_round0_nostage35_20260404"
DEFAULT_EXPORT_DIR = REPO_ROOT / "pipeline" / "data" / "export_scene_v7_full20_rotation8_best_multiview_20260405_horizontal"
DEFAULT_LOG_DIR = REPO_ROOT / "runtime" / "full20_scene_20260404" / "logs"
DEFAULT_PYTHON = "python"
DEFAULT_BLENDER = os.environ.get("BLENDER_BIN", "blender")
DEFAULT_PROFILE = REPO_ROOT / "configs" / "dataset_profiles" / "scene_v7_full20_loop.json"
DEFAULT_SCENE_TEMPLATE = REPO_ROOT / "configs" / "scene_template.json"
SCENE_WORKER_PATTERN = "run_scene_agent_monitor.py --worker-mode"
REGEN_PATTERN = "run_full20_regen_daemon.sh|run_asset_regeneration_queue.py"
EXPORT_PATTERN = "run_full20_export_watch.sh"
TUNER_WATCHDOG_PATTERN = "run_render_feedback_tuner_watchdog.sh"
TERMINAL_SCENE_STATUSES = {"kept", "max_rounds_reached", "deprecated"}
BLOCKED_OBJECT_STATUSES = {
    "deprecated_pending_replacement",
    "manual_reprompt_required",
    "deprecated",
    "replacement_failed",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Keep the full20 scene loop alive until export can finish")
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--python", dest="python_bin", default=DEFAULT_PYTHON)
    parser.add_argument("--blender", dest="blender_bin", default=DEFAULT_BLENDER)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--scene-template", default=str(DEFAULT_SCENE_TEMPLATE))
    parser.add_argument("--scene-gpus", default="0,2", help="Comma-separated GPUs reserved for scene workers")
    parser.add_argument("--regen-gpu", default="1")
    parser.add_argument("--sleep-seconds", type=float, default=45.0)
    parser.add_argument("--gpu-idle-mb", type=int, default=12000)
    parser.add_argument("--max-rounds", type=int, default=15)
    return parser.parse_args()


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_cmd(cmd: List[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=check,
    )


def pgrep_lines(pattern: str) -> List[str]:
    try:
        completed = run_cmd(["pgrep", "-af", pattern], check=False)
    except FileNotFoundError:
        return []
    if completed.returncode not in (0, 1):
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def query_gpu_memory() -> Dict[str, int]:
    completed = run_cmd(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=False,
    )
    usage: Dict[str, int] = {}
    if completed.returncode != 0:
        return usage
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        idx, used = [part.strip() for part in line.split(",", 1)]
        usage[idx] = int(used)
    return usage


def parse_pair_name(name: str) -> Optional[Tuple[str, int]]:
    match = re.fullmatch(r"(obj_\d+)_yaw(\d+)", name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def latest_decision_status(pair_dir: Path) -> Optional[str]:
    decision_files = sorted((pair_dir / "decisions").glob("round*_decision.json"))
    if not decision_files:
        return None
    payload = load_json(decision_files[-1], default={}) or {}
    return payload.get("status")


def latest_agent_round_idx(pair_dir: Path) -> int:
    best = -1
    for agent_path in pair_dir.glob("agent_round*.json"):
        match = re.fullmatch(r"agent_round(\d+)\.json", agent_path.name)
        if not match:
            continue
        best = max(best, int(match.group(1)))
    return best


def latest_scene_status(pair_dir: Path, max_rounds: int) -> Optional[str]:
    status = latest_decision_status(pair_dir)
    if status:
        return status
    agent_round_idx = latest_agent_round_idx(pair_dir)
    if agent_round_idx >= max_rounds:
        return "max_rounds_reached"
    return None


def load_registry_status_map(root_dir: Path) -> Dict[str, str]:
    registry_path = root_dir.parent / "asset_registry.json"
    registry = load_json(registry_path, default={}) or {}
    return {obj_id: str((entry or {}).get("status") or "") for obj_id, entry in registry.items()}


def scan_pending_worker_dirs(root_dir: Path, max_rounds: int) -> List[dict]:
    registry_status = load_registry_status_map(root_dir)
    shard_root = root_dir / "_shards"
    worker_dirs: List[dict] = []
    for worker_dir in sorted(shard_root.iterdir() if shard_root.exists() else []):
        if not worker_dir.is_dir():
            continue
        if not (worker_dir.name.startswith("shard_gpu_") or worker_dir.name.startswith("resume_gpu_")):
            continue
        pending_pairs: List[str] = []
        blocked_pairs = 0
        done_pairs = 0
        for pair_dir in sorted(worker_dir.iterdir()):
            if not pair_dir.is_dir():
                continue
            parsed = parse_pair_name(pair_dir.name)
            if not parsed:
                continue
            obj_id, _ = parsed
            if registry_status.get(obj_id) in BLOCKED_OBJECT_STATUSES:
                blocked_pairs += 1
                continue
            status = latest_scene_status(pair_dir, max_rounds=max_rounds)
            if status in TERMINAL_SCENE_STATUSES:
                done_pairs += 1
                continue
            pending_pairs.append(pair_dir.name)
        worker_dirs.append(
            {
                "name": worker_dir.name,
                "path": str(worker_dir),
                "pending_count": len(pending_pairs),
                "pending_pairs": pending_pairs[:20],
                "done_pairs": done_pairs,
                "blocked_pairs": blocked_pairs,
            }
        )
    worker_dirs.sort(key=lambda item: (-int(item["pending_count"]), str(item["name"])))
    return worker_dirs


def queue_status_counts(root_dir: Path) -> Dict[str, int]:
    queue_path = root_dir.parent / "regeneration_queue.json"
    queue = load_json(queue_path, default=[]) or []
    return dict(Counter(str(item.get("status") or "unknown") for item in queue))


def registry_status_counts(root_dir: Path) -> Dict[str, int]:
    registry_path = root_dir.parent / "asset_registry.json"
    registry = load_json(registry_path, default={}) or {}
    return dict(Counter(str((entry or {}).get("status") or "unknown") for entry in registry.values()))


def start_background(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )
    log_handle.close()
    return process.pid


def ensure_regen_daemon(args, events: List[str]):
    if pgrep_lines(REGEN_PATTERN):
        return
    cmd = [
        "bash",
        str(REPO_ROOT / "scripts" / "run_full20_regen_daemon.sh"),
        str(REPO_ROOT),
    ]
    pid = start_background(cmd, Path(args.log_dir) / "regen_daemon.log")
    events.append(f"started_regen_daemon:{pid}")


def ensure_export_watch(args, events: List[str]):
    if pgrep_lines(EXPORT_PATTERN):
        return
    cmd = [
        "bash",
        str(REPO_ROOT / "scripts" / "run_full20_export_watch.sh"),
        str(REPO_ROOT),
        args.root_dir,
        args.export_dir,
    ]
    pid = start_background(cmd, Path(args.log_dir) / "export_watch.log")
    events.append(f"started_export_watch:{pid}")


def ensure_tuner_watchdog(args, events: List[str]):
    if pgrep_lines(TUNER_WATCHDOG_PATTERN):
        return
    cmd = [
        "bash",
        str(REPO_ROOT / "scripts" / "run_render_feedback_tuner_watchdog.sh"),
        str(REPO_ROOT),
    ]
    pid = start_background(cmd, Path(args.log_dir) / "render_tuner_watchdog.log")
    events.append(f"started_tuner_watchdog:{pid}")


def parse_active_scene_workers(lines: List[str]) -> Tuple[set[str], set[str]]:
    active_gpus: set[str] = set()
    active_dirs: set[str] = set()
    for line in lines:
        gpu_match = re.search(r"--worker-gpu\s+([0-9]+)", line)
        dir_match = re.search(r"--worker-shard-dir\s+(\S+)", line)
        if gpu_match:
            active_gpus.add(gpu_match.group(1))
        if dir_match:
            active_dirs.add(dir_match.group(1))
    return active_gpus, active_dirs


def maybe_launch_scene_workers(args, worker_dirs: List[dict], gpu_usage: Dict[str, int], events: List[str]):
    active_lines = pgrep_lines(SCENE_WORKER_PATTERN)
    active_gpus, active_dirs = parse_active_scene_workers(active_lines)
    available_dirs = [item for item in worker_dirs if int(item["pending_count"]) > 0 and item["path"] not in active_dirs]
    for gpu in [item.strip() for item in str(args.scene_gpus).split(",") if item.strip()]:
        if gpu in active_gpus:
            continue
        if gpu_usage.get(gpu, 10**9) >= int(args.gpu_idle_mb):
            continue
        if not available_dirs:
            break
        chosen = available_dirs.pop(0)
        log_name = f"watchdog_scene_{chosen['name']}_gpu{gpu}.log"
        summary_name = f"watchdog_{chosen['name']}_gpu{gpu}.json"
        cmd = [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "run_scene_agent_monitor.py"),
            "--worker-mode",
            "--root-dir",
            args.root_dir,
            "--gpus",
            gpu,
            "--python",
            args.python_bin,
            "--blender",
            args.blender_bin,
            "--meshes-dir",
            str(REPO_ROOT / "pipeline" / "data" / "meshes"),
            "--profile",
            args.profile,
            "--scene-template",
            args.scene_template,
            "--max-rounds",
            str(args.max_rounds),
            "--pass-sleep-seconds",
            "2",
            "--worker-gpu",
            gpu,
            "--worker-shard-dir",
            str(chosen["path"]),
            "--worker-log-path",
            str(Path(args.root_dir) / "_monitor_logs" / summary_name),
            "--allow-unsafe-single-gpu-review",
        ]
        pid = start_background(cmd, Path(args.log_dir) / log_name)
        active_gpus.add(gpu)
        active_dirs.add(str(chosen["path"]))
        events.append(f"started_scene_worker:gpu{gpu}:{chosen['name']}:{pid}")


def main():
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = log_dir / "watchdog_heartbeat.json"

    while True:
        tick_started = time.time()
        events: List[str] = []
        ensure_regen_daemon(args, events)
        ensure_tuner_watchdog(args, events)
        ensure_export_watch(args, events)

        worker_dirs = scan_pending_worker_dirs(root_dir, max_rounds=args.max_rounds)
        gpu_usage = query_gpu_memory()
        maybe_launch_scene_workers(args, worker_dirs, gpu_usage, events)

        payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "root_dir": str(root_dir),
            "export_dir": str(Path(args.export_dir).resolve()),
            "gpu_usage_mb": gpu_usage,
            "queue_counts": queue_status_counts(root_dir),
            "registry_counts": registry_status_counts(root_dir),
            "scene_worker_dirs": worker_dirs,
            "active_scene_workers": pgrep_lines(SCENE_WORKER_PATTERN),
            "active_regen": pgrep_lines(REGEN_PATTERN),
            "active_tuner_watchdog": pgrep_lines(TUNER_WATCHDOG_PATTERN),
            "active_export_watch": pgrep_lines(EXPORT_PATTERN),
            "events": events,
            "elapsed_seconds": round(time.time() - tick_started, 3),
        }
        save_json(heartbeat_path, payload)
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
