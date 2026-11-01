import hashlib

from dataevolver.harness.evaluator import (
    DeterministicEvaluator,
    EvaluatorContract,
    FrozenExecutionContract,
)
from dataevolver.harness.experiments import CandidateDecision, CandidateGate, can_reuse_baseline
from dataevolver.harness.models import (
    BudgetSpec,
    EvaluationReport,
    EvidenceStatus,
    EvidenceValue,
    GoalSpec,
    MatchedTrialKey,
    OperatorSpec,
    StageResult,
    StageStatus,
)
from dataevolver.harness.operators import FakeOperator, OperatorRegistry


def write_evaluator(path):
    path.write_text(
        """{
  "schema_version": "dataevolver.harness.evaluator.v1",
  "evaluator_id": "fake-quality-v1",
  "revision": "1",
  "hard_constraints": {"artifact_valid": {"required": true}},
  "target_metrics": {"quality": {"required": true, "minimum": 0.8, "minimum_improvement": 0.2}},
  "protected_metrics": {"safety": {"required": true, "max_regression": 0.05}}
}
""",
        encoding="utf-8",
    )


def matched_key(seed=17):
    return MatchedTrialKey(
        input_artifact_hash="input-a",
        parent_artifact_hash="parent-a",
        seed=seed,
        operator_id="fake.baseline",
        operator_version="1",
        model_config_hash="model-a",
        preprocessing_hash="pre-a",
        evaluator_hash="eval-a",
        environment_contract_hash="env-a",
    )


def evidence(metric, value, status=EvidenceStatus.VERIFIED):
    return EvidenceValue(metric=metric, status=status, value=value, required=True)


def report(*, quality, safety=1.0, quality_status=EvidenceStatus.VERIFIED, key=None):
    return EvaluationReport(
        evaluator_id="fake-quality-v1",
        evaluator_revision="1",
        evaluator_hash="eval-a",
        hard_constraints={"artifact_valid": evidence("artifact_valid", True)},
        target_metrics={"quality": evidence("quality", quality, quality_status)},
        protected_metrics={"safety": evidence("safety", safety)},
        matched_trial_key=key or matched_key(),
    )


def test_unobserved_required_metric_blocks_candidate_adoption(tmp_path):
    evaluator_path = tmp_path / "evaluator.json"
    write_evaluator(evaluator_path)
    contract = EvaluatorContract.from_path(evaluator_path)

    decision = CandidateGate(contract).decide(
        report(quality=0.4),
        report(quality=None, quality_status=EvidenceStatus.UNOBSERVED),
    )

    assert decision.decision is CandidateDecision.ABSTAIN
    assert "quality" in decision.reasons[0]


def test_protected_metric_regression_rolls_back_candidate(tmp_path):
    evaluator_path = tmp_path / "evaluator.json"
    write_evaluator(evaluator_path)
    contract = EvaluatorContract.from_path(evaluator_path)

    decision = CandidateGate(contract).decide(report(quality=0.4), report(quality=0.9, safety=0.8))

    assert decision.decision is CandidateDecision.ROLLBACK
    assert any("safety" in reason for reason in decision.reasons)


def test_baseline_reuse_requires_exact_key_and_verified_evidence():
    baseline = report(quality=0.4)

    assert can_reuse_baseline(baseline, matched_key(), matched_key()) is True
    assert can_reuse_baseline(baseline, matched_key(), matched_key(seed=18)) is False
    observed = report(quality=0.4, quality_status=EvidenceStatus.OBSERVED)
    assert can_reuse_baseline(observed, matched_key(), matched_key()) is False


def test_deterministic_evaluator_marks_missing_required_metric_unobserved(tmp_path):
    evaluator_path = tmp_path / "evaluator.json"
    write_evaluator(evaluator_path)
    evaluator = DeterministicEvaluator(EvaluatorContract.from_path(evaluator_path))
    result = StageResult(
        operator_id="fake.baseline",
        operator_version="1",
        status=StageStatus.FAILED,
        metrics={"safety": 1.0},
        hard_constraints={"artifact_valid": True},
        matched_trial_key=matched_key(),
    )

    evaluated = evaluator.evaluate(result, effect_status=EvidenceStatus.VERIFIED)

    assert evaluated.target_metrics["quality"].status is EvidenceStatus.UNOBSERVED
    assert evaluated.protected_metrics["safety"].status is EvidenceStatus.VERIFIED


def test_frozen_execution_contract_detects_operator_source_drift(tmp_path):
    evaluator_path = tmp_path / "evaluator.json"
    schema_path = tmp_path / "schema.json"
    operator_source = tmp_path / "operator.py"
    write_evaluator(evaluator_path)
    schema_path.write_text('{"schema_version":"test.v1"}\n', encoding="utf-8")
    operator_source.write_text("VERSION = 1\n", encoding="utf-8")
    evaluator_hash = hashlib.sha256(evaluator_path.read_bytes()).hexdigest()
    goal = GoalSpec(
        schema_version="dataevolver.harness.goal.v1",
        campaign_id="demo",
        objective={"kind": "quality"},
        initial_operator_id="fake.baseline",
        route_candidates=("fake",),
        budgets=BudgetSpec(2, 3, 1, 0, 60),
        resource_policy={"allow_gpu": False, "allowed_operators": ["fake.baseline"]},
        frozen_evaluator={"path": str(evaluator_path), "sha256": evaluator_hash},
    )
    spec = OperatorSpec(
        operator_id="fake.baseline",
        version="1",
        source_path=str(operator_source),
        code_revision="code-a",
    )
    registry = OperatorRegistry()
    registry.register(
        FakeOperator(
            spec,
            status=StageStatus.FAILED,
            metrics={"quality": 0.4, "safety": 1.0},
            hard_constraints={"artifact_valid": True},
        )
    )
    guard = FrozenExecutionContract(
        repo_root=tmp_path,
        goal=goal,
        registry=registry,
        planner_id="deterministic-v1",
        planner_config={"interventions": []},
        evaluator_path=evaluator_path,
        schema_paths=(schema_path,),
        execution_paths=(operator_source,),
    )
    frozen = guard.freeze()

    assert guard.verify(frozen).valid is True
    operator_source.write_text("VERSION = 2\n", encoding="utf-8")
    drift = guard.verify(frozen)
    assert drift.valid is False
    assert "harness_code_inventory_hash" in drift.changed_fields
