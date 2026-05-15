"""
Dry-run a two-agent stage loop for the scene-aware rotation pipeline.

This simulation models the architecture the project needs:
- Main agent: executes the stage chain.
- Review agent: only wakes up after Stage 5 and routes the next iteration.

Routing rule after Stage 5:
- route_to = "stage2": regenerate image, then continue Stage 2.5 -> 3 -> 4 -> 5
- route_to = "stage3": regenerate 3D asset, then continue Stage 4 -> 5
- route_to = "stage4": rerender/reposition in Blender, then continue Stage 5
- route_to = "end": accept and stop

It does not call real T2I, SAM, Hunyuan3D, Blender, VLM, MCP, or training.
It only tests whether the two-agent control flow behaves as expected.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_OUTPUT_DIR = Path("_tmp") / "dual_agent_stage_loop_sim"

STAGE_SEQUENCE = ["stage1", "stage2", "stage2_5", "stage3", "stage4", "stage5"]
ROUTE_START_INDEX = {
    "stage2": STAGE_SEQUENCE.index("stage2"),
    "stage3": STAGE_SEQUENCE.index("stage3"),
    "stage4": STAGE_SEQUENCE.index("stage4"),
}

ISSUE_TO_ROUTE = {
    "image_not_usable": "stage2",
    "wrong_object": "stage2",
    "bad_cutout": "stage2",
    "mesh_broken": "stage3",
    "texture_missing": "stage3",
    "bad_geometry": "stage3",
    "bad_lighting": "stage4",
    "bad_composition": "stage4",
    "floating": "stage4",
    "weak_shadow": "stage4",
}

ROUTE_PRIORITY = {
    "stage2": 0,
    "stage3": 1,
    "stage4": 2,
    "end": 3,
}

STAGE_EFFECTS = {
    "stage2": {
        "image_not_usable": -0.55,
        "wrong_object": -0.50,
        "bad_cutout": 0.12,
        "mesh_broken": 0.08,
        "bad_geometry": 0.08,
    },
    "stage2_5": {
        "bad_cutout": -0.55,
        "mesh_broken": -0.08,
    },
    "stage3": {
        "mesh_broken": -0.58,
        "texture_missing": -0.52,
        "bad_geometry": -0.50,
        "bad_lighting": 0.06,
    },
    "stage4": {
        "bad_lighting": -0.55,
        "bad_composition": -0.50,
        "floating": -0.48,
        "weak_shadow": -0.46,
    },
}


@dataclass
class PipelineState:
    iteration: int
    issue_weights: Dict[str, float]
    score: float


@dataclass
class StageEvent:
    iteration: int
    agent: str
    stage: str
    action: str
    score_before: float
    score_after: float
    note: str


@dataclass
class ReviewDecision:
    iteration: int
    agent: str
    route_to: str
    should_end: bool
    active_issues: List[str]
    reason: str
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a dual-agent stage routing loop")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--scenario",
        choices=["render_first", "mesh_first", "image_first", "mixed"],
        default="mixed",
        help="Initial failure profile for the mock review agent",
    )
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_from_issues(issue_weights: Dict[str, float], rng: random.Random) -> float:
    sorted_weights = sorted(issue_weights.values(), reverse=True)
    major = sum(sorted_weights[:3]) / 3.0
    minor = sum(sorted_weights[3:]) / max(1, len(sorted_weights[3:]))
    noise = rng.uniform(-0.012, 0.012)
    return round(clamp01(0.96 - major * 0.52 - minor * 0.18 + noise), 4)


def initial_issues(scenario: str) -> Dict[str, float]:
    base = {
        "image_not_usable": 0.08,
        "wrong_object": 0.06,
        "bad_cutout": 0.12,
        "mesh_broken": 0.12,
        "texture_missing": 0.10,
        "bad_geometry": 0.16,
        "bad_lighting": 0.18,
        "bad_composition": 0.16,
        "floating": 0.12,
        "weak_shadow": 0.14,
    }
    if scenario == "image_first":
        base.update({"image_not_usable": 0.96, "wrong_object": 0.82, "bad_cutout": 0.78})
    elif scenario == "mesh_first":
        base.update({"mesh_broken": 0.96, "texture_missing": 0.84, "bad_geometry": 0.82})
    elif scenario == "render_first":
        base.update({"bad_lighting": 0.96, "bad_composition": 0.86, "floating": 0.74, "weak_shadow": 0.82})
    else:
        base.update({"mesh_broken": 0.80, "bad_geometry": 0.72, "bad_lighting": 0.88, "weak_shadow": 0.76})
    return base


def active_issues(state: PipelineState, threshold: float = 0.30) -> List[str]:
    return [
        issue
        for issue, weight in sorted(state.issue_weights.items(), key=lambda item: item[1], reverse=True)
        if weight >= threshold
    ]


def execute_stage(state: PipelineState, stage: str, rng: random.Random) -> PipelineState:
    next_weights = dict(state.issue_weights)
    for issue, delta in STAGE_EFFECTS.get(stage, {}).items():
        if issue in next_weights:
            next_weights[issue] = clamp01(next_weights[issue] + delta)

    # Small uncertainty: a real stage can fix one thing while slightly disturbing another.
    for issue, value in list(next_weights.items()):
        if stage == "stage1":
            drift = 0.0
        else:
            drift = rng.uniform(-0.025, 0.018)
        next_weights[issue] = round(clamp01(value + drift), 4)

    return PipelineState(
        iteration=state.iteration,
        issue_weights=next_weights,
        score=score_from_issues(next_weights, rng),
    )


def review_agent_decide(state: PipelineState) -> ReviewDecision:
    issues = active_issues(state)
    if state.score >= 0.86 and not issues:
        return ReviewDecision(
            iteration=state.iteration,
            agent="review_agent",
            route_to="end",
            should_end=True,
            active_issues=[],
            reason="Stage 5 review passed; no active issue above threshold",
            score=state.score,
        )

    candidate_routes = []
    for issue in issues:
        route = ISSUE_TO_ROUTE.get(issue)
        if route:
            candidate_routes.append((ROUTE_PRIORITY[route], route, issue))

    if not candidate_routes:
        return ReviewDecision(
            iteration=state.iteration,
            agent="review_agent",
            route_to="stage4",
            should_end=False,
            active_issues=issues,
            reason="No upstream issue matched; default to Stage 4 render refinement",
            score=state.score,
        )

    _, route_to, reason_issue = sorted(candidate_routes)[0]
    return ReviewDecision(
        iteration=state.iteration,
        agent="review_agent",
        route_to=route_to,
        should_end=False,
        active_issues=issues,
        reason=f"highest-priority active issue '{reason_issue}' requires returning to {route_to}",
        score=state.score,
    )


def run_iteration(
    *,
    state: PipelineState,
    start_stage: str,
    rng: random.Random,
) -> tuple[PipelineState, List[StageEvent], ReviewDecision]:
    events: List[StageEvent] = []
    start_idx = ROUTE_START_INDEX.get(start_stage, 0)
    current = state

    for stage in STAGE_SEQUENCE[start_idx:]:
        before = current.score
        current = execute_stage(current, stage, rng)
        events.append(
            StageEvent(
                iteration=state.iteration,
                agent="main_agent",
                stage=stage,
                action="execute",
                score_before=before,
                score_after=current.score,
                note="main agent executed pipeline stage",
            )
        )
        if stage == "stage5":
            break

    decision = review_agent_decide(current)
    return current, events, decision


def run_simulation(output_dir: Path, max_iterations: int, seed: int, scenario: str) -> dict:
    rng = random.Random(seed)
    state = PipelineState(
        iteration=0,
        issue_weights=initial_issues(scenario),
        score=score_from_issues(initial_issues(scenario), rng),
    )
    next_start_stage = "stage1"
    all_events: List[dict] = []
    decisions: List[dict] = []

    final_status = "max_iterations_reached"
    for iteration in range(max_iterations):
        state.iteration = iteration
        save_json(output_dir / "states" / f"iteration{iteration:02d}_input.json", asdict(state))

        state, events, decision = run_iteration(
            state=state,
            start_stage=next_start_stage,
            rng=rng,
        )
        save_json(output_dir / "states" / f"iteration{iteration:02d}_after_stage5.json", asdict(state))
        save_json(output_dir / "review_agent" / f"iteration{iteration:02d}_decision.json", asdict(decision))

        all_events.extend(asdict(event) for event in events)
        decisions.append(asdict(decision))

        if decision.should_end:
            final_status = "completed"
            break

        next_start_stage = decision.route_to
        state = PipelineState(
            iteration=iteration + 1,
            issue_weights=state.issue_weights,
            score=state.score,
        )

    summary = {
        "status": final_status,
        "scenario": scenario,
        "seed": seed,
        "max_iterations": max_iterations,
        "agents": {
            "main_agent": "executes stage chain",
            "review_agent": "decides after Stage 5 whether to end or route back to Stage 2/3/4",
        },
        "events": all_events,
        "decisions": decisions,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_simulation(
        output_dir=output_dir,
        max_iterations=args.max_iterations,
        seed=args.seed,
        scenario=args.scenario,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
