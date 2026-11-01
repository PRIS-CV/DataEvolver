import hashlib
import json
from pathlib import Path

import pytest

from dataevolver.harness.controller import EvolutionController
from dataevolver.harness.evaluator import EvaluatorContract
from dataevolver.harness.models import (
    BudgetSpec,
    ControllerState,
    ExpectedOutputSpec,
    GoalSpec,
    MatchedTrialKey,
    OperatorSpec,
    StageStatus,
)
from dataevolver.harness.operators import FakeOperator, InjectedOperatorCrash, OperatorRegistry
from dataevolver.harness.planner import DeterministicPlanner


TERMINAL = {ControllerState.COMPLETE, ControllerState.ABSTAIN}


def evaluator_payload():
    return {
        "schema_version": "dataevolver.harness.evaluator.v1",
        "evaluator_id": "fake-quality-v1",
        "revision": "1",
        "hard_constraints": {"artifact_valid": {"required": True}},
        "target_metrics": {"quality": {"required": True, "minimum": 0.8, "minimum_improvement": 0.2}},
        "protected_metrics": {"safety": {"required": True, "max_regression": 0.05}},
    }


def trial_key(evaluator_hash, *, seed=17):
    return MatchedTrialKey(
        input_artifact_hash="input-a",
        parent_artifact_hash="parent-a",
        seed=seed,
        operator_id="fake.baseline",
        operator_version="1",
        model_config_hash="model-a",
        preprocessing_hash="pre-a",
        evaluator_hash=evaluator_hash,
        environment_contract_hash="env-a",
    )


def build_controller(
    tmp_path,
    *,
    campaign_id="demo",
    baseline_failure="LOW_QUALITY",
    candidate_metrics=None,
    max_operator_calls=3,
    crash_baseline=False,
    canary_seed=17,
):
    evaluator_path = tmp_path / "evaluator.json"
    evaluator_path.write_text(json.dumps(evaluator_payload(), sort_keys=True), encoding="utf-8")
    evaluator_hash = hashlib.sha256(evaluator_path.read_bytes()).hexdigest()
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"schema_version":"test.schema.v1"}\n', encoding="utf-8")
    operator_source = tmp_path / "operator.py"
    operator_source.write_text("VERSION = 1\n", encoding="utf-8")
    initial_key = trial_key(evaluator_hash, seed=17)
    canary_key = trial_key(evaluator_hash, seed=canary_seed)
    interventions = (
        {
            "intervention_id": "fix-low-quality",
            "failure_code": "LOW_QUALITY",
            "operator_id": "fake.candidate",
            "level": "L0",
            "parameters": {"quality_patch": "improve"},
            "candidate_delta": {"quality_patch": "improve"},
            "rollback_plan": {"kind": "retain_baseline"},
        },
    )
    goal = GoalSpec(
        schema_version="dataevolver.harness.goal.v1",
        campaign_id=campaign_id,
        objective={
            "kind": "fake_quality",
            "initial_trial_key": initial_key.to_dict(),
            "canary_trial_key": canary_key.to_dict(),
        },
        initial_operator_id="fake.baseline",
        route_candidates=("fake",),
        budgets=BudgetSpec(2, max_operator_calls, 1, 0, 60),
        resource_policy={
            "allow_gpu": False,
            "allowed_operators": ["fake.baseline", "fake.candidate"],
        },
        frozen_evaluator={"path": str(evaluator_path), "sha256": evaluator_hash},
        interventions=interventions,
        lease_seconds=10,
    )

    def spec(operator_id):
        return OperatorSpec(
            operator_id=operator_id,
            version="1",
            retry_safe=True,
            expected_output_specs=(ExpectedOutputSpec("effect.json"),),
            model_config_hash="model-a",
            code_revision="code-a",
            source_path=str(operator_source),
        )

    registry = OperatorRegistry()
    registry.register(
        FakeOperator(
            spec("fake.baseline"),
            status=StageStatus.FAILED,
            metrics={"quality": 0.4, "safety": 1.0},
            hard_constraints={"artifact_valid": True},
            failure_code=baseline_failure,
            crash_after_effect=crash_baseline,
        )
    )
    registry.register(
        FakeOperator(
            spec("fake.candidate"),
            status=StageStatus.SUCCESS,
            metrics=candidate_metrics if candidate_metrics is not None else {"quality": 0.9, "safety": 1.0},
            hard_constraints={"artifact_valid": True},
        )
    )
    planner = DeterministicPlanner(goal.interventions)
    controller = EvolutionController.create(
        repo_root=tmp_path,
        campaigns_root=tmp_path / "campaigns",
        goal=goal,
        registry=registry,
        planner=planner,
        evaluator_contract=EvaluatorContract.from_path(evaluator_path),
        schema_paths=(schema_path,),
        execution_paths=(operator_source,),
        idempotency_key=f"request-{campaign_id}",
    )
    return controller, operator_source


def read_events(controller):
    return controller.store.events.read()


def test_tick_is_bounded_and_full_fake_loop_creates_terminal_evidence(tmp_path):
    controller, _ = build_controller(tmp_path)

    receipt = controller.tick(max_actions=1)
    assert receipt.previous_state is ControllerState.INIT
    assert receipt.new_state is ControllerState.PLAN
    assert receipt.action_count == 1
    assert receipt.receipt_hash

    final = controller.run()
    event_types = [event.event_type for event in read_events(controller)]
    campaign_dir = controller.layout.campaign_dir
    assert final.state is ControllerState.COMPLETE
    assert "INTERVENTION_ADOPTED" in event_types
    assert "CAPABILITY_CANDIDATE_CREATED" in event_types
    assert (campaign_dir / "goal.json").is_file()
    assert (campaign_dir / "frozen_contract.json").is_file()
    assert (campaign_dir / "events.jsonl").is_file()
    assert (campaign_dir / "state.json").is_file()
    assert (campaign_dir / "rounds" / "r000" / "stage_result.json").is_file()
    assert (campaign_dir / "experiments" / "e000" / "baseline" / "evaluation.json").is_file()
    assert (campaign_dir / "experiments" / "e000" / "candidate" / "evaluation.json").is_file()
    assert (campaign_dir / "final_report.json").is_file()


def test_budget_exhaustion_abstains_before_candidate_side_effect(tmp_path):
    controller, _ = build_controller(tmp_path, max_operator_calls=1)

    final = controller.run()

    assert final.state is ControllerState.ABSTAIN
    assert final.terminal_reason == "BUDGET_EXHAUSTED"
    candidate_effects = list((controller.layout.campaign_dir / "experiments").glob("**/candidate/effect.json"))
    assert candidate_effects == []


def test_unknown_failure_abstains_without_unregistered_action(tmp_path):
    controller, _ = build_controller(tmp_path, baseline_failure="UNKNOWN_FAILURE")

    final = controller.run()

    assert final.state is ControllerState.ABSTAIN
    assert final.terminal_reason == "NO_SAFE_ACTION"


def test_protected_regression_rolls_back_and_never_adopts(tmp_path):
    controller, _ = build_controller(tmp_path, candidate_metrics={"quality": 0.9, "safety": 0.8})

    final = controller.run()
    event_types = [event.event_type for event in read_events(controller)]

    assert final.state is ControllerState.ABSTAIN
    assert "CANDIDATE_ROLLED_BACK" in event_types
    assert "INTERVENTION_ADOPTED" not in event_types


def test_crash_after_effect_recovers_one_logical_commit_and_one_effect(tmp_path):
    controller, _ = build_controller(tmp_path, crash_baseline=True)
    controller.tick()
    controller.tick()
    with pytest.raises(InjectedOperatorCrash):
        controller.tick()

    final = controller.run()
    effect = json.loads(
        (controller.layout.campaign_dir / "rounds" / "r000" / "execute" / "effect.json").read_text(encoding="utf-8")
    )
    committed = [
        event
        for event in read_events(controller)
        if event.event_type == "ACTION_COMMITTED" and event.payload.get("call_id") == "demo:r000:initial"
    ]
    assert final.state is ControllerState.COMPLETE
    assert effect["execution_count"] == 1
    assert len(committed) == 1


def test_resume_after_logical_commit_before_state_transition_keeps_call_context(tmp_path):
    controller, _ = build_controller(tmp_path)
    controller.tick()
    controller.tick()
    state = controller.status()
    call = controller._make_call(
        call_id="demo:r000:initial",
        operator_id="fake.baseline",
        work_dir=controller.layout.campaign_dir / "rounds" / "r000" / "execute",
        key=trial_key(controller.evaluator.contract.sha256),
    )

    committed_state, _, _ = controller._execute_call(state, call)

    assert committed_state.data["pending_call"]["call_id"] == "demo:r000:initial"
    final = controller.run()
    assert final.state is ControllerState.COMPLETE


def test_mismatched_initial_provenance_rematerializes_baseline(tmp_path):
    controller, _ = build_controller(tmp_path, canary_seed=18)

    final = controller.run()
    event_types = [event.event_type for event in read_events(controller)]

    assert final.state is ControllerState.COMPLETE
    assert "BASELINE_REMATERIALIZED" in event_types
    assert (controller.layout.campaign_dir / "experiments" / "e000" / "baseline" / "effect.json").is_file()


def test_contract_drift_pauses_before_next_effect(tmp_path):
    controller, operator_source = build_controller(tmp_path)
    controller.tick()
    operator_source.write_text("VERSION = 2\n", encoding="utf-8")

    receipt = controller.tick()

    assert receipt.new_state is ControllerState.PAUSED_CONTRACT_DRIFT
    assert controller.status().data["paused_from"] == "PLAN"
    assert not (controller.layout.campaign_dir / "rounds" / "r000" / "execute" / "effect.json").exists()
