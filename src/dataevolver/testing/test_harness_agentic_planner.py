import json
import hashlib
from pathlib import Path

import pytest

from dataevolver.harness.models import (
    ExperimentKind,
    FailureSignature,
    GoalSpec,
    PlannerContext,
    PlannerDecision,
    PlannerDecisionKind,
    StrategyPatch,
)
from dataevolver.harness.planner import AgenticRSIPlanner
from dataevolver.harness.strategy import ExperimentCompiler


ROOT = Path(__file__).resolve().parents[3]


class SequenceBackend:
    backend_id = "fake-openai-compatible"
    model_id = "fake-rsi-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, messages, *, schema):
        self.calls.append({"messages": messages, "schema": schema})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def context(*, attempted=(), remaining_experiments=2):
    return PlannerContext(
        goal={"target": "verified 3D asset"},
        remaining_budget={"rounds": 2, "experiments": remaining_experiments, "operator_calls": 4},
        asset_id="mug",
        incumbent={"bundle_manifest": "incumbent/bundle_manifest.json"},
        evaluation={"target_metrics": {"geometry_structure_consistency": 0.3, "render_quality": 0.4}},
        failure_signature=FailureSignature("GEOMETRY_MISMATCH"),
        evidence={"issue_tags": ["geometry_distortion"]},
        experiment_history=(),
        attempted_strategy_hashes=tuple(attempted),
        capability_memories=(),
        allowed_experiment_kinds=tuple(ExperimentKind),
        allowed_scene_actions=("L_KEY_UP", "ENV_ROTATE_30"),
        registered_operators=({"operator_id": "dataevolver.3d_asset_evolution", "version": "1"},),
        compatibility_hash="contract-a",
    )


def decision_payload(**changes):
    payload = {
        "decision_id": "d1",
        "decision": "RUN_EXPERIMENT",
        "asset_id": "mug",
        "failure_code": "GEOMETRY_MISMATCH",
        "experiment_kind": "PROMPT_REGENERATE",
        "hypothesis": "A cleaner single-object prompt will improve geometry.",
        "strategy_patch": {
            "prompt_text": "A single red ceramic mug, isolated, complete silhouette.",
            "negative_prompt_append": ["multiple objects"],
            "t2i_seed": 23,
            "reconstruction_seed": None,
            "force_resegment": False,
            "scene_actions": [],
            "restart_stage": "t2i",
        },
        "expected_effect": {"geometry_structure_consistency": "improve"},
        "required_evidence": ["geometry_structure_consistency"],
        "confidence": "medium",
        "rationale_summary": "The current failure is upstream of reconstruction.",
    }
    payload.update(changes)
    return payload


def test_strategy_patch_and_decision_reject_unknown_or_executable_fields():
    with pytest.raises(ValueError, match="unknown StrategyPatch fields"):
        StrategyPatch.from_dict({"prompt_text": "mug", "command": "rm -rf /"})

    with pytest.raises(ValueError, match="unknown PlannerDecision fields"):
        PlannerDecision.from_dict({**decision_payload(), "python": "print('unsafe')"})

    with pytest.raises(ValueError, match="at most 2"):
        StrategyPatch.from_dict({"scene_actions": ["a", "b", "c"]})


def test_agentic_planner_repairs_invalid_json_once_and_returns_valid_decision():
    backend = SequenceBackend(["not-json", json.dumps(decision_payload())])
    planner = AgenticRSIPlanner(backend=backend, config={"timeout_seconds": 120})

    decision = planner.plan(context())

    assert decision.decision is PlannerDecisionKind.RUN_EXPERIMENT
    assert decision.strategy_patch.restart_stage == "t2i"
    assert len(backend.calls) == 2
    assert "validation_error" in backend.calls[1]["messages"][-1]["content"]
    trace = planner.last_trace()
    assert trace["raw_responses"] == ["not-json", json.dumps(decision_payload())]
    assert trace["repair_attempted"] is True
    assert trace["outcome"] == "VALID_DECISION"


def test_agentic_planner_abstains_without_deterministic_fallback():
    backend = SequenceBackend([RuntimeError("endpoint unavailable")])
    planner = AgenticRSIPlanner(backend=backend, config={})

    decision = planner.plan(context())

    assert decision.decision is PlannerDecisionKind.ABSTAIN
    assert "endpoint unavailable" in decision.rationale_summary
    assert len(backend.calls) == 1
    assert planner.last_trace()["outcome"] == "BACKEND_ERROR"


@pytest.mark.parametrize(
    ("kind", "patch", "restart"),
    [
        ("PROMPT_REGENERATE", {"prompt_text": "single mug", "restart_stage": "t2i"}, "t2i"),
        ("RESEGMENT_RECONSTRUCT", {"force_resegment": True, "restart_stage": "segment"}, "segment"),
        ("RECONSTRUCT_RETRY", {"reconstruction_seed": 41, "restart_stage": "reconstruct"}, "reconstruct"),
        ("SCENE_REFINE", {"scene_actions": ["L_KEY_UP"], "restart_stage": "render"}, "render"),
    ],
)
def test_experiment_compiler_maps_only_registered_experiment_shapes(kind, patch, restart):
    evidence = "render_quality" if kind == "SCENE_REFINE" else "geometry_structure_consistency"
    payload = decision_payload(
        experiment_kind=kind,
        strategy_patch=patch,
        expected_effect={evidence: "improve"},
        required_evidence=[evidence],
    )
    compiler = ExperimentCompiler(
        operator_id="dataevolver.3d_asset_evolution",
        operator_version="1",
        allowed_scene_actions={"L_KEY_UP", "ENV_ROTATE_30"},
    )

    compiled = compiler.compile(PlannerDecision.from_dict(payload), context())

    assert compiled.restart_stage == restart
    assert compiled.operator_id == "dataevolver.3d_asset_evolution"
    assert compiled.operator_parameters["strategy_patch"] == StrategyPatch.from_dict(patch).to_dict()
    assert compiled.idempotency_key.endswith(compiled.novelty_key)


def test_experiment_compiler_rejects_duplicate_or_out_of_budget_strategy():
    decision = PlannerDecision.from_dict(decision_payload())
    compiler = ExperimentCompiler(
        operator_id="dataevolver.3d_asset_evolution",
        operator_version="1",
        allowed_scene_actions={"L_KEY_UP"},
    )
    compiled = compiler.compile(decision, context())

    with pytest.raises(ValueError, match="already attempted"):
        compiler.compile(decision, context(attempted=(compiled.novelty_key,)))
    with pytest.raises(ValueError, match="budget"):
        compiler.compile(decision, context(remaining_experiments=0))


def test_experiment_compiler_rejects_cross_kind_patch_and_unknown_scene_action():
    compiler = ExperimentCompiler(
        operator_id="dataevolver.3d_asset_evolution",
        operator_version="1",
        allowed_scene_actions={"L_KEY_UP"},
    )
    mixed = decision_payload(strategy_patch={"prompt_text": "mug", "scene_actions": ["L_KEY_UP"], "restart_stage": "t2i"})
    with pytest.raises(ValueError, match="not allowed for PROMPT_REGENERATE"):
        compiler.compile(PlannerDecision.from_dict(mixed), context())

    unknown = decision_payload(
        experiment_kind="SCENE_REFINE",
        strategy_patch={"scene_actions": ["UNREGISTERED"], "restart_stage": "render"},
        expected_effect={"render_quality": "improve"},
        required_evidence=["render_quality"],
    )
    with pytest.raises(ValueError, match="scene action is not registered"):
        compiler.compile(PlannerDecision.from_dict(unknown), context())

    unknown_evidence = decision_payload(required_evidence=["unregistered_metric"])
    with pytest.raises(ValueError, match="required evidence is not a target metric"):
        compiler.compile(PlannerDecision.from_dict(unknown_evidence), context())


def test_agentic_example_configs_and_schemas_are_self_consistent():
    goal_path = ROOT / "configs" / "harness" / "agentic_3d_rsi_canary.json"
    evaluator_path = ROOT / "configs" / "harness" / "agentic_3d_rsi_evaluator.json"
    goal = GoalSpec.from_json(goal_path.read_text(encoding="utf-8"))
    evaluator_hash = hashlib.sha256(evaluator_path.read_bytes()).hexdigest()

    assert goal.planner["kind"] == "agentic_rsi"
    assert goal.frozen_evaluator["sha256"] == evaluator_hash
    assert goal.objective["initial_trial_key"]["evaluator_hash"] == evaluator_hash
    for name in ("harness_strategy_patch.schema.json", "harness_planner_decision.schema.json"):
        json.loads((ROOT / "configs" / "schemas" / name).read_text(encoding="utf-8"))
    real_goal = GoalSpec.from_json(
        (ROOT / "configs" / "harness" / "agentic_3d_rsi_real_canary.json").read_text(encoding="utf-8")
    )
    assert real_goal.objective["real_canary"] is True
    assert len(real_goal.objective["asset_ids"]) == 1
    assert real_goal.budgets.max_experiments == 3
    assert real_goal.budgets.max_planner_calls == 4
    assert real_goal.frozen_evaluator["sha256"] == evaluator_hash


def test_agentic_planner_runtime_fingerprint_changes_without_recording_secret(monkeypatch):
    from dataevolver.harness.planner import OpenAICompatiblePlannerBackend

    monkeypatch.setenv("DATAEVOLVER_AGENT_BASE_URL", "http://planner-a/v1")
    monkeypatch.setenv("DATAEVOLVER_AGENT_API_KEY", "secret-value")
    monkeypatch.setenv("DATAEVOLVER_AGENT_MODEL", "model-a")
    planner = AgenticRSIPlanner(backend=OpenAICompatiblePlannerBackend({}), config={})
    first = planner.config()
    monkeypatch.setenv("DATAEVOLVER_AGENT_MODEL", "model-b")
    second = planner.config()

    assert first["model_id"] == "model-a"
    assert second["model_id"] == "model-b"
    assert first["base_url_hash"]
    assert "secret-value" not in json.dumps(first)
