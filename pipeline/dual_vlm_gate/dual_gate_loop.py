from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.dual_vlm_gate.dual_actions import (  # noqa: E402
    SCENE_HARD_FAIL_ISSUES,
    choose_actions,
    default_control_state,
    load_json,
    save_json,
)


DEFAULT_CONFIG = THIS_DIR / "default_config.json"
STEP_SCRIPT = THIS_DIR / "dual_agent_step.py"
INIT_SEARCH_SCRIPT = THIS_DIR / "dual_init_search.py"


def parse_args():
    cfg = load_json(DEFAULT_CONFIG, default={}) or {}
    gate_cfg = cfg.get("gate") or {}
    review_cfg = cfg.get("review") or {}
    render_cfg = cfg.get("render") or {}
    parser = argparse.ArgumentParser(description="Dual-object VLM quality gate loop")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--obj-a-id", default="object_a")
    parser.add_argument("--obj-b-id", default="object_b")
    parser.add_argument("--mesh-a", required=True)
    parser.add_argument("--mesh-b", required=True)
    parser.add_argument("--ref-a", default=None)
    parser.add_argument("--ref-b", default=None)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--threshold", type=float, default=float(gate_cfg.get("threshold", 0.72)))
    parser.add_argument("--max-rounds", type=int, default=int(gate_cfg.get("max_rounds", 5)))
    parser.add_argument("--step-scale", type=float, default=float(gate_cfg.get("step_scale", 1.0)))
    parser.add_argument("--patience", type=int, default=int(gate_cfg.get("patience", 0)))
    parser.add_argument("--rollback-score-drop-threshold", type=float, default=float(gate_cfg.get("rollback_score_drop_threshold", 0.05)))
    parser.add_argument("--device", default=str(review_cfg.get("device", "cuda:0")))
    parser.add_argument("--blender", default=os.environ.get("ARIS_BLENDER", "blender"))
    parser.add_argument("--resolution", type=int, default=int(render_cfg.get("resolution", 768)))
    parser.add_argument("--engine", default=str(render_cfg.get("engine", "CYCLES")), choices=["EEVEE", "CYCLES"])
    parser.add_argument("--template", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-candidates", type=int, default=int(gate_cfg.get("init_candidates", 8)))
    parser.add_argument("--use-vlm-init-review", action="store_true", default=bool(gate_cfg.get("use_vlm_init_review", False)))
    parser.add_argument("--vlm-init-weight", type=float, default=float(gate_cfg.get("vlm_init_weight", 0.55)))
    parser.add_argument("--vlm-init-min-score", type=float, default=float(gate_cfg.get("vlm_init_min_score", 0.35)))
    parser.add_argument("--placement-state", default=None, help="Use a preselected placement_state.json instead of running init search.")
    parser.add_argument("--initial-control-state", default=None, help="Optional control-state JSON used as the round00 baseline.")
    parser.add_argument("--skip-init-search", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def coerce_score(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_init_search(args, pair_dir: Path, initial_state_path: Path, placement_state_path: Path) -> dict:
    cmd = [
        sys.executable,
        str(INIT_SEARCH_SCRIPT),
        "--output-dir",
        str(pair_dir / "init_search"),
        "--sample-id",
        args.sample_id,
        "--mesh-a",
        args.mesh_a,
        "--mesh-b",
        args.mesh_b,
        "--scene",
        args.scene,
        "--control-state",
        str(initial_state_path),
        "--placement-state-out",
        str(placement_state_path),
        "--blender",
        args.blender,
        "--engine",
        args.engine,
        "--resolution",
        str(args.resolution),
        "--seed",
        str(args.seed),
        "--candidates",
        str(args.init_candidates),
    ]
    if args.use_vlm_init_review:
        cmd += [
            "--use-vlm-init-review",
            "--device",
            args.device,
            "--vlm-weight",
            str(args.vlm_init_weight),
            "--vlm-min-score",
            str(args.vlm_init_min_score),
        ]
    if args.ref_a:
        cmd += ["--ref-a", args.ref_a]
    if args.ref_b:
        cmd += ["--ref-b", args.ref_b]
    if args.template:
        cmd += ["--template", args.template]
    logs_dir = pair_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "init_search.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    summary = load_json(pair_dir / "init_search" / "init_search_summary.json", default={}) or {}
    summary["returncode"] = result.returncode
    summary["log_path"] = str(log_path)
    return summary


def run_step(
    args,
    pair_dir: Path,
    round_idx: int,
    actions: list[str],
    state_in: str | None,
    prev_rgb: str | None,
    placement_state_in: str | None,
    placement_state_out: str | None,
) -> dict:
    cmd = [
        sys.executable,
        str(STEP_SCRIPT),
        "--output-dir",
        str(pair_dir),
        "--sample-id",
        args.sample_id,
        "--obj-a-id",
        args.obj_a_id,
        "--obj-b-id",
        args.obj_b_id,
        "--mesh-a",
        args.mesh_a,
        "--mesh-b",
        args.mesh_b,
        "--scene",
        args.scene,
        "--round-idx",
        str(round_idx),
        "--step-scale",
        str(args.step_scale),
        "--device",
        args.device,
        "--blender",
        args.blender,
        "--resolution",
        str(args.resolution),
        "--engine",
        args.engine,
        "--seed",
        str(args.seed),
    ]
    if args.ref_a:
        cmd += ["--ref-a", args.ref_a]
    if args.ref_b:
        cmd += ["--ref-b", args.ref_b]
    if args.template:
        cmd += ["--template", args.template]
    if state_in:
        cmd += ["--control-state-in", state_in]
    if prev_rgb:
        cmd += ["--prev-rgb", prev_rgb]
    if placement_state_in:
        cmd += ["--placement-state", placement_state_in]
    if placement_state_out:
        cmd += ["--placement-state-out", placement_state_out]
    if actions:
        cmd += ["--actions", *actions]

    logs_dir = pair_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"round{round_idx:02d}.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    payload = load_json(pair_dir / f"agent_round{round_idx:02d}.json", default={})
    if result.returncode != 0 and isinstance(payload, dict):
        payload.setdefault("success", False)
        payload.setdefault("error", f"step_exit_{result.returncode}")
    return payload if isinstance(payload, dict) else {"success": False, "error": "missing_step_payload"}


def write_decision(pair_dir: Path, round_idx: int, result: dict, threshold: float, passed: bool, best_score: Optional[float]):
    save_json(
        pair_dir / f"gate_decision_round{round_idx:02d}.json",
        {
            "round_idx": round_idx,
            "passed": bool(passed),
            "score": result.get("hybrid_score"),
            "threshold": threshold,
            "best_score_after_round": best_score,
            "issue_tags": result.get("issue_tags"),
            "suggested_actions": result.get("suggested_actions"),
            "state_out": result.get("state_out"),
            "render_dir": result.get("render_dir"),
            "review_agg_path": result.get("review_agg_path"),
        },
    )


def has_scene_hard_fail(result: dict) -> bool:
    tags = {str(tag) for tag in result.get("issue_tags") or []}
    return bool(tags & SCENE_HARD_FAIL_ISSUES)


def should_stop_stagnant(rounds: list[dict], actions: list[str], best_score: Optional[float]) -> bool:
    if len(rounds) < 3 or best_score is None:
        return False
    recent = rounds[-3:]
    recent_scores = [coerce_score(r.get("score")) for r in recent]
    if any(score is None for score in recent_scores):
        return False
    if max(recent_scores) < best_score + 1e-6 and best_score - recent_scores[-1] >= 0.04:
        return True
    contact_loop = {"CONTACT_SHADOW_UP", "AO_UP", "A_LOWER_TINY", "B_LOWER_TINY"}
    if set(actions or []).issubset(contact_loop) and len(set(actions or [])) >= 2:
        return True
    return False


def run_gate(args) -> dict:
    root_dir = Path(args.root_dir).resolve()
    pair_dir = root_dir / args.sample_id
    pair_dir.mkdir(parents=True, exist_ok=True)
    status_path = pair_dir / "gate_status.json"
    if args.resume:
        existing = load_json(status_path, default=None)
        if isinstance(existing, dict) and existing.get("status") in {"accepted", "rejected"}:
            return existing

    actions: list[str] = []
    state_in = None
    prev_rgb = None
    best_score = None
    best_round_idx = None
    best_result = None
    no_improve = 0
    rounds = []
    placement_state_path = pair_dir / "placement_state.json"
    initial_state_path = pair_dir / "states" / "round00_input.json"
    initial_state_path.parent.mkdir(parents=True, exist_ok=True)
    if not initial_state_path.exists():
        initial_state = default_control_state()
        override_state = load_json(args.initial_control_state, default=None) if args.initial_control_state else None
        if isinstance(override_state, dict):
            from copy import deepcopy

            def merge(base, override):
                out = deepcopy(base)
                for key, value in override.items():
                    if isinstance(value, dict) and isinstance(out.get(key), dict):
                        out[key] = merge(out[key], value)
                    else:
                        out[key] = deepcopy(value)
                return out

            initial_state = merge(initial_state, override_state)
        save_json(initial_state_path, initial_state)
    if args.placement_state and not placement_state_path.exists():
        placement_state = load_json(args.placement_state, default=None)
        if not isinstance(placement_state, dict):
            status = {
                "status": "rejected",
                "reason": "preselected_placement_state_missing_or_invalid",
                "sample_id": args.sample_id,
                "placement_state_source": args.placement_state,
                "rounds": [],
            }
            save_json(status_path, status)
            return status
        save_json(placement_state_path, placement_state)
    init_summary = None
    if not args.skip_init_search and not placement_state_path.exists():
        init_summary = run_init_search(args, pair_dir, initial_state_path, placement_state_path)
        if init_summary.get("returncode") != 0 or not placement_state_path.exists():
            status = {
                "status": "rejected",
                "reason": "init_search_failed",
                "sample_id": args.sample_id,
                "init_search": init_summary,
                "rounds": [],
            }
            save_json(status_path, status)
            return status

    for round_idx in range(int(args.max_rounds)):
        placement_state_exists = placement_state_path.exists()
        result = run_step(
            args,
            pair_dir,
            round_idx,
            actions,
            state_in,
            prev_rgb,
            str(placement_state_path) if placement_state_exists else None,
            str(placement_state_path) if round_idx == 0 and not placement_state_exists else None,
        )
        score = coerce_score(result.get("hybrid_score"))
        improved = score is not None and (best_score is None or score > best_score + 1e-6)
        if improved:
            best_score = score
            best_round_idx = round_idx
            best_result = result
            no_improve = 0
        else:
            no_improve += 1
        passed = bool(result.get("success")) and score is not None and score >= float(args.threshold)
        rounds.append(
            {
                "round_idx": round_idx,
                "score": score,
                "threshold": float(args.threshold),
                "passed": passed,
                "issue_tags": result.get("issue_tags"),
                "suggested_actions": result.get("suggested_actions"),
                "actions_in": actions,
                "state_out": result.get("state_out"),
                "render_dir": result.get("render_dir"),
            }
        )
        write_decision(pair_dir, round_idx, result, float(args.threshold), passed, best_score)
        if passed:
            status = {
                "status": "accepted",
                "sample_id": args.sample_id,
                "accepted_round_idx": round_idx,
                "accepted_score": score,
                "accepted_threshold": float(args.threshold),
                "accepted_render_dir": result.get("render_dir"),
                "accepted_state_out": result.get("state_out"),
                "best_score": best_score,
                "best_round_idx": best_round_idx,
                "rounds": rounds,
                "placement_state": str(placement_state_path) if placement_state_path.exists() else None,
                "init_search": init_summary,
            }
            save_json(status_path, status)
            return status

        if has_scene_hard_fail(result):
            status = {
                "status": "rejected",
                "reason": "scene_hard_fail",
                "sample_id": args.sample_id,
                "best_score": best_score,
                "best_round_idx": best_round_idx,
                "hard_fail_tags": result.get("issue_tags"),
                "rounds": rounds,
                "placement_state": str(placement_state_path) if placement_state_path.exists() else None,
                "init_search": init_summary,
            }
            save_json(status_path, status)
            return status

        review = load_json(result.get("review_agg_path") or "", default={}) or {}
        actions = choose_actions(review)
        state_in = result.get("state_out")
        prev_rgb = result.get("rgb_path")
        if best_result and score is not None and best_score is not None:
            if best_score - score >= float(args.rollback_score_drop_threshold):
                state_in = best_result.get("state_out")
                prev_rgb = best_result.get("rgb_path")
        if should_stop_stagnant(rounds, actions, best_score):
            status = {
                "status": "rejected",
                "reason": "stagnant_after_best_round",
                "sample_id": args.sample_id,
                "best_score": best_score,
                "best_round_idx": best_round_idx,
                "next_actions": actions,
                "rounds": rounds,
                "placement_state": str(placement_state_path) if placement_state_path.exists() else None,
                "init_search": init_summary,
            }
            save_json(status_path, status)
            return status
        if args.patience > 0 and no_improve >= int(args.patience):
            break

    status = {
        "status": "rejected",
        "reason": "max_rounds_without_passing_gate",
        "sample_id": args.sample_id,
        "best_score": best_score,
        "best_round_idx": best_round_idx,
        "best_render_dir": best_result.get("render_dir") if best_result else None,
        "best_state_out": best_result.get("state_out") if best_result else None,
        "last_score": rounds[-1].get("score") if rounds else None,
        "last_threshold": float(args.threshold),
        "rounds": rounds,
        "placement_state": str(placement_state_path) if placement_state_path.exists() else None,
        "init_search": init_summary,
    }
    save_json(status_path, status)
    return status


def main():
    args = parse_args()
    status = run_gate(args)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
