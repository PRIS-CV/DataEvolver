"""
Dry-run a human-gated child-agent loop for the scene optimization pipeline.

This simulation does not call Blender, VLM, training code, or MCP agents. It
models the control policy only:
1. The child gate agent inspects the latest review state.
2. If a human command exists for this round, it obeys that command.
3. If no human command exists, it follows deterministic fallback rules.
4. It records a mock main-agent call and advances the loop.

Human command JSONL examples:
  {"round": 1, "command": "actions", "actions": ["O_SCALE_UP_10"], "note": "try larger"}
  {"round": 2, "command": "keep", "note": "good enough"}
  {"round": 3, "command": "stop", "note": "pause for manual review"}
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_OUTPUT_DIR = Path("_tmp") / "human_gate_loop_sim"

ISSUE_TO_ACTIONS = {
    "underexposed": ["ENV_STRENGTH_UP", "L_KEY_UP"],
    "weak_subject_separation": ["ENV_STRENGTH_UP"],
    "shadow_missing": ["S_CONTACT_SHADOW_UP", "L_KEY_UP"],
    "floating_visible": ["O_LOWER_SMALL", "S_CONTACT_SHADOW_UP"],
    "ground_intersection_visible": ["O_LIFT_SMALL"],
    "object_too_small": ["O_SCALE_UP_10"],
    "object_too_large": ["O_SCALE_DOWN_10"],
    "color_shift": ["M_VALUE_DOWN", "M_SATURATION_DOWN"],
    "scene_light_mismatch": ["ENV_ROTATE_30", "L_KEY_YAW_POS_15"],
}

ACTION_EFFECTS = {
    "ENV_STRENGTH_UP": {"underexposed": -0.45, "weak_subject_separation": -0.20},
    "L_KEY_UP": {"underexposed": -0.30, "shadow_missing": -0.10},
    "S_CONTACT_SHADOW_UP": {"shadow_missing": -0.45, "floating_visible": -0.20},
    "O_LOWER_SMALL": {"floating_visible": -0.50, "ground_intersection_visible": 0.15},
    "O_LIFT_SMALL": {"ground_intersection_visible": -0.50, "floating_visible": 0.10},
    "O_SCALE_UP_10": {"object_too_small": -0.55, "object_too_large": 0.10},
    "O_SCALE_DOWN_10": {"object_too_large": -0.55, "object_too_small": 0.10},
    "M_VALUE_DOWN": {"color_shift": -0.25},
    "M_SATURATION_DOWN": {"color_shift": -0.25},
    "ENV_ROTATE_30": {"scene_light_mismatch": -0.35},
    "L_KEY_YAW_POS_15": {"scene_light_mismatch": -0.25},
}


@dataclass
class ReviewState:
    round_idx: int
    score: float
    issue_weights: Dict[str, float]
    verdict: str = "needs_fix"


@dataclass
class GateDecision:
    round_idx: int
    mode: str
    command: Optional[str]
    actions: List[str]
    should_continue: bool
    should_keep: bool
    reason: str
    human_note: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a human-gated child-agent loop")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--human-commands", default=None, help="Optional JSONL command inbox")
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--clean", action="store_true", help="Clear output dir before simulation")
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_human_commands(path: Optional[str]) -> Dict[int, dict]:
    if not path:
        return {}
    inbox = Path(path)
    if not inbox.exists():
        raise FileNotFoundError(f"human command inbox not found: {inbox}")

    commands: Dict[int, dict] = {}
    with inbox.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            payload = json.loads(line)
            if "round" not in payload:
                raise ValueError(f"{inbox}:{line_no} missing 'round'")
            commands[int(payload["round"])] = payload
    return commands


def active_issues(review: ReviewState, threshold: float = 0.28) -> List[str]:
    return [
        issue
        for issue, weight in sorted(review.issue_weights.items(), key=lambda item: item[1], reverse=True)
        if weight >= threshold
    ]


def decide_without_human(review: ReviewState) -> GateDecision:
    if review.score >= 0.86 and not active_issues(review, threshold=0.35):
        return GateDecision(
            round_idx=review.round_idx,
            mode="auto",
            command=None,
            actions=[],
            should_continue=False,
            should_keep=True,
            reason="score and issue weights reached the keep threshold",
        )

    actions: List[str] = []
    reasons: List[str] = []
    for issue in active_issues(review):
        for action in ISSUE_TO_ACTIONS.get(issue, []):
            if action not in actions:
                actions.append(action)
                reasons.append(f"{issue}->{action}")
            if len(actions) >= 3:
                break
        if len(actions) >= 3:
            break

    if not actions:
        actions = ["ENV_STRENGTH_UP"]
        reasons = ["fallback: keep improving subject separation"]

    return GateDecision(
        round_idx=review.round_idx,
        mode="auto",
        command=None,
        actions=actions,
        should_continue=True,
        should_keep=False,
        reason="; ".join(reasons),
    )


def decide_with_human(review: ReviewState, command_payload: dict) -> GateDecision:
    command = str(command_payload.get("command", "")).strip().lower()
    note = command_payload.get("note")

    if command == "keep":
        return GateDecision(
            round_idx=review.round_idx,
            mode="human",
            command=command,
            actions=[],
            should_continue=False,
            should_keep=True,
            reason="human accepted current result",
            human_note=note,
        )

    if command == "stop":
        return GateDecision(
            round_idx=review.round_idx,
            mode="human",
            command=command,
            actions=[],
            should_continue=False,
            should_keep=False,
            reason="human paused or stopped the loop",
            human_note=note,
        )

    if command in {"actions", "retry"}:
        actions = [str(v) for v in command_payload.get("actions", []) if str(v).strip()]
        if not actions:
            fallback = decide_without_human(review)
            actions = fallback.actions
        return GateDecision(
            round_idx=review.round_idx,
            mode="human",
            command=command,
            actions=actions,
            should_continue=True,
            should_keep=False,
            reason="human supplied explicit next actions",
            human_note=note,
        )

    raise ValueError(f"unsupported human command at round {review.round_idx}: {command!r}")


def score_from_issues(issue_weights: Dict[str, float], rng: random.Random) -> float:
    total_penalty = sum(issue_weights.values()) / max(1, len(issue_weights))
    noise = rng.uniform(-0.015, 0.015)
    return round(max(0.0, min(0.98, 0.94 - total_penalty * 0.55 + noise)), 4)


def apply_mock_main_agent(review: ReviewState, actions: List[str], rng: random.Random) -> ReviewState:
    next_weights = dict(review.issue_weights)
    for action in actions:
        for issue, delta in ACTION_EFFECTS.get(action, {}).items():
            current = next_weights.get(issue, 0.0)
            next_weights[issue] = max(0.0, min(1.0, current + delta))

    # Natural small drift: reviews are noisy and unrelated issues may improve a little.
    for issue, value in list(next_weights.items()):
        drift = rng.uniform(-0.06, 0.025)
        next_weights[issue] = round(max(0.0, min(1.0, value + drift)), 4)

    next_score = score_from_issues(next_weights, rng)
    verdict = "keep" if next_score >= 0.86 and not active_issues(
        ReviewState(review.round_idx + 1, next_score, next_weights),
        threshold=0.35,
    ) else "needs_fix"

    return ReviewState(
        round_idx=review.round_idx + 1,
        score=next_score,
        issue_weights=next_weights,
        verdict=verdict,
    )


def initial_review_state(rng: random.Random) -> ReviewState:
    issue_weights = {
        "underexposed": 0.62,
        "weak_subject_separation": 0.48,
        "shadow_missing": 0.55,
        "floating_visible": 0.38,
        "color_shift": 0.22,
        "scene_light_mismatch": 0.34,
        "object_too_small": 0.12,
        "object_too_large": 0.05,
        "ground_intersection_visible": 0.03,
    }
    return ReviewState(
        round_idx=0,
        score=score_from_issues(issue_weights, rng),
        issue_weights=issue_weights,
    )


def run_simulation(output_dir: Path, commands: Dict[int, dict], max_rounds: int, seed: int) -> dict:
    rng = random.Random(seed)
    review = initial_review_state(rng)
    history: List[dict] = []
    calls_dir = output_dir / "main_agent_calls"
    reviews_dir = output_dir / "mock_reviews"
    decisions_dir = output_dir / "gate_decisions"

    final_status = "max_rounds_reached"
    for _ in range(max_rounds):
        save_json(reviews_dir / f"round{review.round_idx:02d}.json", asdict(review))

        command_payload = commands.get(review.round_idx)
        if command_payload is not None:
            decision = decide_with_human(review, command_payload)
        else:
            decision = decide_without_human(review)

        save_json(decisions_dir / f"round{review.round_idx:02d}_decision.json", asdict(decision))

        history.append({
            "round": review.round_idx,
            "score": review.score,
            "verdict": review.verdict,
            "active_issues": active_issues(review),
            "decision_mode": decision.mode,
            "command": decision.command,
            "actions": decision.actions,
            "reason": decision.reason,
        })

        if decision.should_keep:
            final_status = "kept"
            break
        if not decision.should_continue:
            final_status = "stopped"
            break

        mock_call = {
            "round": review.round_idx,
            "called_agent": "main_agent_mock",
            "actions": decision.actions,
            "note": decision.human_note or decision.reason,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_json(calls_dir / f"round{review.round_idx:02d}_main_agent_call.json", mock_call)
        review = apply_mock_main_agent(review, decision.actions, rng)

    summary = {
        "status": final_status,
        "seed": seed,
        "max_rounds": max_rounds,
        "rounds_observed": len(history),
        "human_command_rounds": sorted(commands.keys()),
        "history": history,
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    commands = load_human_commands(args.human_commands)
    summary = run_simulation(
        output_dir=output_dir,
        commands=commands,
        max_rounds=args.max_rounds,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
