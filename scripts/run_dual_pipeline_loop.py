#!/usr/bin/env python3
"""
Dual-pipeline auto-loop runner
===============================

Automatically chains multiple rounds of ``run_dual_pipeline.py``:

    Round N completes → read merged_trainready from state
    → feed as --existing-baseline to Round N+1 → repeat

Stops when:
  - ``--target-objects`` reached, OR
  - ``--max-rounds`` reached, OR
  - consecutive stalls (0 new objects), OR
  - a round fails (non-zero exit)

Usage
-----
    python scripts/run_dual_pipeline_loop.py \
        --target-objects 60 \
        --initial-baseline <path> \
        --full-objects 3 --weak-objects 3 --weak-angles 225 \
        --gate-threshold 0.51 --gate-max-rounds 3 \
        --angle-gate-threshold 0.51 --angle-gate-max-rounds 3 \
        --no-force-accept --skip-training
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_ROOT = Path(os.environ.get(
    "ARIS_RUNTIME_ROOT",
    "<runtime-root>",
))
PYTHON = os.environ.get(
    "ARIS_PYTHON",
    "python",
)

LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
LOOP_STATE_FILE = "DUAL_LOOP_STATE.json"
# ═══════════════════════════════════════════════════════════════════════════════
# Loop state persistence
# ═══════════════════════════════════════════════════════════════════════════════

class LoopState:
    """Tracks multi-round loop progress, supports resume."""

    def __init__(self, loop_root: Path):
        self.path = loop_root / LOOP_STATE_FILE
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {
                "schema": "dual_loop_v1",
                "created": datetime.now().isoformat(),
                "rounds": [],
                "total_objects": 0,
                "total_pairs": 0,
                "current_baseline": None,
                "stop_reason": None,
            }

    def save(self):
        self.data["updated"] = datetime.now().isoformat()
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False)
        )

    def add_round(self, info: dict):
        self.data["rounds"].append(info)
        self.data["total_objects"] = info["cumulative_objects"]
        self.data["total_pairs"] = info["cumulative_pairs"]
        self.data["current_baseline"] = info["merged_root"]
        self.save()

    @property
    def completed_rounds(self) -> int:
        return len(self.data["rounds"])

    @property
    def current_baseline(self) -> str | None:
        return self.data.get("current_baseline")

    @property
    def total_objects(self) -> int:
        return self.data.get("total_objects", 0)

    @property
    def total_pairs(self) -> int:
        return self.data.get("total_pairs", 0)

    def finish(self, reason: str):
        self.data["stop_reason"] = reason
        self.data["finished"] = datetime.now().isoformat()
        self.save()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def count_objects_in_baseline(baseline_path: str) -> int:
    """Read manifest.json from a trainready root and return object count."""
    manifest = Path(baseline_path) / "manifest.json"
    if not manifest.exists():
        return 0
    data = json.loads(manifest.read_text())
    return len(data.get("objects", {}))


def count_pairs_in_baseline(baseline_path: str) -> int:
    manifest = Path(baseline_path) / "manifest.json"
    if not manifest.exists():
        return 0
    data = json.loads(manifest.read_text())
    return len(data.get("pairs", []))


def read_round_result(run_root: str) -> dict | None:
    """Read DUAL_PIPELINE_STATE.json from a completed round."""
    state_path = Path(run_root) / "DUAL_PIPELINE_STATE.json"
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text())


def find_latest_run_root(pattern: str = "dual_pipeline_*") -> Path | None:
    """Find the most recently created dual_pipeline run directory."""
    candidates = sorted(RUNTIME_ROOT.glob(pattern), reverse=True)
    for c in candidates:
        if c.is_dir() and (c / "DUAL_PIPELINE_STATE.json").exists():
            return c
    return None


def build_cmd(args, baseline: str | None) -> list[str]:
    """Build the command to invoke run_dual_pipeline.py."""
    cmd = [
        PYTHON,
        str(REPO_ROOT / "scripts" / "run_dual_pipeline.py"),
        "--full-objects", str(args.full_objects),
        "--weak-objects", str(args.weak_objects),
    ]
    if args.group_ratio:
        cmd += ["--group-ratio", args.group_ratio]
    if args.weak_angles:
        cmd += ["--weak-angles", args.weak_angles]
    if args.eval_metrics:
        cmd += ["--eval-metrics", args.eval_metrics]
    if args.weak_threshold:
        cmd += ["--weak-threshold", str(args.weak_threshold)]
    if baseline:
        cmd += ["--existing-baseline", baseline]

    # gate params
    cmd += ["--gate-max-rounds", str(args.gate_max_rounds)]
    cmd += ["--gate-threshold", str(args.gate_threshold)]
    cmd += ["--angle-gate-threshold", str(args.angle_gate_threshold)]
    cmd += ["--angle-gate-max-rounds", str(args.angle_gate_max_rounds)]
    cmd += ["--weak-gate-reject-threshold", str(args.weak_gate_reject_threshold)]
    if args.no_force_accept:
        cmd.append("--no-force-accept")

    # training
    if args.skip_training:
        cmd.append("--skip-training")
    else:
        cmd += ["--epochs", str(args.epochs)]
        cmd += ["--lr", args.lr]
        cmd += ["--lora-rank", str(args.lora_rank)]

    # misc
    cmd += ["--gpus", args.gpus]
    cmd += ["--device", args.device]
    cmd += ["--seed", str(args.seed)]

    return cmd


# ═══════════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_loop(args):
    """Main loop: repeatedly call run_dual_pipeline.py until stopping condition."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.resume:
        loop_root = Path(args.resume)
        if not loop_root.exists():
            print(f"[ERROR] Resume path does not exist: {loop_root}")
            sys.exit(1)
    else:
        loop_root = RUNTIME_ROOT / f"dual_loop_{timestamp}"
        loop_root.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = loop_root / "loop.log"
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FMT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("dual_loop")

    state = LoopState(loop_root)

    # Initial baseline
    if state.completed_rounds == 0:
        if not args.initial_baseline:
            log.error("--initial-baseline required for first round")
            sys.exit(1)
        baseline = args.initial_baseline
        baseline_objs = count_objects_in_baseline(baseline)
        baseline_pairs = count_pairs_in_baseline(baseline)
        log.info(f"Initial baseline: {baseline}")
        log.info(f"  Objects: {baseline_objs}, Pairs: {baseline_pairs}")
        state.data["initial_baseline"] = baseline
        state.data["initial_objects"] = baseline_objs
        state.data["initial_pairs"] = baseline_pairs
        state.save()
    else:
        baseline = state.current_baseline
        baseline_objs = state.total_objects
        baseline_pairs = state.total_pairs
        if state.data.get("stop_reason"):
            log.info(f"  Previous stop reason: {state.data['stop_reason']} — clearing for resume")
            state.data["stop_reason"] = None
            state.data.pop("finished", None)
            state.save()
        log.info(f"Resuming from round {state.completed_rounds}")
        log.info(f"  Current baseline: {baseline}")
        log.info(f"  Cumulative: {baseline_objs} objects, {baseline_pairs} pairs")

    log.info(f"\n{'='*70}")
    log.info(f"  Dual Pipeline Auto-Loop")
    log.info(f"  Loop root: {loop_root}")
    log.info(f"  Target: {args.target_objects} objects")
    log.info(f"  Max rounds: {args.max_rounds}")
    log.info(f"  Max stall: {args.max_stall}")
    log.info(f"{'='*70}\n")

    if args.dry_run:
        log.info("[DRY-RUN] Would execute the following rounds:")
        for i in range(1, args.max_rounds + 1):
            cmd = build_cmd(args, baseline)
            log.info(f"\n[Round {i}] CMD: {' '.join(cmd)}")
            if baseline_objs >= args.target_objects:
                log.info(f"[Round {i}] Target reached, would stop")
                break
            baseline_objs += 4  # simulate
        return

    stall_count = 0
    round_num = state.completed_rounds + 1

    while round_num <= args.max_rounds:
        log.info(f"\n{'='*70}")
        log.info(f"  Round {round_num} / {args.max_rounds}")
        log.info(f"  Current: {baseline_objs} objects, {baseline_pairs} pairs")
        log.info(f"  Baseline: {baseline}")
        log.info(f"{'='*70}\n")

        # Check target
        if baseline_objs >= args.target_objects:
            log.info(f"[STOP] Target reached: {baseline_objs} >= {args.target_objects}")
            state.finish("target_reached")
            break

        # Build and run command
        cmd = build_cmd(args, baseline)
        log.info(f"[Round {round_num}] CMD: {' '.join(cmd)}")

        t0 = time.time()
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=os.environ.copy())
        elapsed = time.time() - t0

        if result.returncode != 0:
            log.error(f"[Round {round_num}] FAILED with exit code {result.returncode}")
            state.finish("round_failed")
            sys.exit(1)

        # Find the run directory (most recent dual_pipeline_*)
        run_root = find_latest_run_root()
        if not run_root:
            log.error(f"[Round {round_num}] Cannot find run directory")
            state.finish("run_dir_not_found")
            sys.exit(1)

        log.info(f"[Round {round_num}] Run directory: {run_root}")

        # Read result
        round_state = read_round_result(str(run_root))
        if not round_state:
            log.error(f"[Round {round_num}] Cannot read DUAL_PIPELINE_STATE.json")
            state.finish("state_read_failed")
            sys.exit(1)

        merged_root = round_state.get("merged_root")
        if not merged_root:
            log.error(f"[Round {round_num}] No merged_root in state")
            state.finish("no_merged_root")
            sys.exit(1)

        new_objs = count_objects_in_baseline(merged_root)
        new_pairs = count_pairs_in_baseline(merged_root)
        added_objs = new_objs - baseline_objs
        added_pairs = new_pairs - baseline_pairs

        log.info(f"[Round {round_num}] Completed in {elapsed:.1f}s")
        log.info(f"  New total: {new_objs} objects (+{added_objs}), {new_pairs} pairs (+{added_pairs})")
        log.info(f"  Merged: {merged_root}")

        # Check stall
        if added_objs == 0:
            stall_count += 1
            log.warning(f"[Round {round_num}] STALL: 0 new objects (stall count: {stall_count}/{args.max_stall})")
            if stall_count >= args.max_stall:
                log.error(f"[STOP] Max stall reached: {stall_count} consecutive rounds with 0 new objects")
                state.finish("max_stall")
                break
        else:
            stall_count = 0

        # Record round
        round_info = {
            "round": round_num,
            "run_root": str(run_root),
            "baseline": baseline,
            "merged_root": merged_root,
            "added_objects": added_objs,
            "added_pairs": added_pairs,
            "cumulative_objects": new_objs,
            "cumulative_pairs": new_pairs,
            "elapsed_seconds": elapsed,
            "group_a": round_state.get("group_a_ids", []),
            "group_b": round_state.get("group_b_ids", []),
        }
        state.add_round(round_info)

        # Check stop signal file
        stop_file = loop_root / "STOP_AFTER_ROUND"
        if stop_file.exists():
            log.info(f"[STOP] Stop signal file detected: {stop_file}")
            stop_file.unlink()
            state.finish("stop_signal")
            break

        # Update for next round
        baseline = merged_root
        baseline_objs = new_objs
        baseline_pairs = new_pairs
        round_num += 1

        # Cooldown
        if args.cooldown > 0 and round_num <= args.max_rounds:
            log.info(f"[Cooldown] Sleeping {args.cooldown}s before next round...")
            time.sleep(args.cooldown)

    # Final summary
    if round_num > args.max_rounds:
        log.info(f"[STOP] Max rounds reached: {args.max_rounds}")
        state.finish("max_rounds")

    log.info(f"\n{'='*70}")
    log.info(f"  Loop Complete")
    log.info(f"  Total rounds: {state.completed_rounds}")
    log.info(f"  Final: {state.total_objects} objects, {state.total_pairs} pairs")
    log.info(f"  Final baseline: {state.current_baseline}")
    log.info(f"  Stop reason: {state.data.get('stop_reason')}")
    log.info(f"  State file: {state.path}")
    log.info(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Dual pipeline auto-loop: chain multiple rounds until target reached",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g_loop = p.add_argument_group("Loop control")
    g_loop.add_argument("--target-objects", type=int, required=True,
                        help="Stop when cumulative objects >= this")
    g_loop.add_argument("--initial-baseline", type=str, default=None,
                        help="Initial baseline trainready root (required for first run)")
    g_loop.add_argument("--max-rounds", type=int, default=20,
                        help="Safety cap on number of rounds (default: 20)")
    g_loop.add_argument("--max-stall", type=int, default=2,
                        help="Stop after N consecutive rounds with 0 new objects (default: 2)")
    g_loop.add_argument("--cooldown", type=int, default=30,
                        help="Seconds to sleep between rounds (default: 30)")
    g_loop.add_argument("--resume", type=str, default=None,
                        help="Resume from existing loop directory")
    g_loop.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")

    g_data = p.add_argument_group("Data generation (passed to run_dual_pipeline.py)")
    g_data.add_argument("--full-objects", type=int, default=3)
    g_data.add_argument("--weak-objects", type=int, default=3)
    g_data.add_argument("--group-ratio", type=str, default=None,
                        help="A:B 分组比例（如 3:2），基于 accepted 数量动态分组")
    g_data.add_argument("--eval-metrics", type=str, default=None)
    g_data.add_argument("--weak-angles", type=str, default="225")
    g_data.add_argument("--weak-threshold", type=float, default=0.98)

    g_gate = p.add_argument_group("VLM gate (passed to run_dual_pipeline.py)")
    g_gate.add_argument("--gate-max-rounds", type=int, default=3)
    g_gate.add_argument("--gate-threshold", type=float, default=0.51)
    g_gate.add_argument("--angle-gate-threshold", type=float, default=0.51)
    g_gate.add_argument("--angle-gate-max-rounds", type=int, default=3)
    g_gate.add_argument("--weak-gate-reject-threshold", type=float, default=0.45)
    g_gate.add_argument("--no-force-accept", action="store_true")

    g_train = p.add_argument_group("Training (passed to run_dual_pipeline.py)")
    g_train.add_argument("--epochs", type=int, default=30)
    g_train.add_argument("--lr", type=str, default="1e-4")
    g_train.add_argument("--lora-rank", type=int, default=32)
    g_train.add_argument("--skip-training", action="store_true")

    g_misc = p.add_argument_group("Misc (passed to run_dual_pipeline.py)")
    g_misc.add_argument("--gpus", type=str, default="0,1,2")
    g_misc.add_argument("--device", type=str, default="cuda:0")
    g_misc.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    run_loop(args)


if __name__ == "__main__":
    main()
