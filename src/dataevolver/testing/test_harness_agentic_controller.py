import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from dataevolver.harness.capabilities import CapabilityStore
from dataevolver.harness.controller import EvolutionController
from dataevolver.harness.evaluator import DeterministicEvaluator, EvaluatorContract
from dataevolver.harness.experiments import CandidateDecision, CandidateGate
from dataevolver.harness.models import (
    BudgetSpec,
    Capability,
    CapabilityStatus,
    EffectVerification,
    EvidenceStatus,
    ExpectedOutputSpec,
    ExperimentKind,
    FailureSignature,
    GoalSpec,
    Intervention,
    InterventionLevel,
    MatchedTrialKey,
    OperatorCall,
    OperatorReceipt,
    OperatorSpec,
    PlannerDecisionKind,
    RecoveryResult,
    RecoveryStatus,
    StageResult,
    StageStatus,
    StrategyPatch,
)
from dataevolver.harness.operators import OperatorRegistry, _load_receipt, _receipt_path, _sha256_file, _write_receipt
from dataevolver.harness.planner import AgenticRSIPlanner
from dataevolver.harness.state_store import atomic_write_json
from dataevolver.harness.strategy import ExperimentCompiler


def evaluator_payload():
    return {
        "schema_version": "dataevolver.harness.evaluator.v1",
        "evaluator_id": "agentic-3d-fake-v1",
        "revision": "1",
        "hard_constraints": {"artifact_valid": {"required": True}},
        "target_metrics": {
            "geometry_structure_consistency": {
                "required": True, "goal_minimum": 0.8, "minimum_improvement": 0.2, "max_regression": 0.05,
            },
            "render_quality": {
                "required": True, "goal_minimum": 0.8, "minimum_improvement": 0.2, "max_regression": 0.05,
            },
        },
        "protected_metrics": {"identity_alignment": {"required": True, "max_regression": 0.05}},
    }


def trial_key(evaluator_hash):
    return MatchedTrialKey(
        input_artifact_hash="asset-request-a",
        parent_artifact_hash="incumbent-a",
        seed=17,
        operator_id="dataevolver.3d_asset_evolution",
        operator_version="1",
        model_config_hash="models-a",
        preprocessing_hash="pre-a",
        evaluator_hash=evaluator_hash,
        environment_contract_hash="env-a",
    )


class SequenceBackend:
    backend_id = "fake-agent"
    model_id = "fake-agent-v1"

    def __init__(self, payloads):
        self.payloads = list(payloads)

    def complete_json(self, messages, *, schema):
        return json.dumps(self.payloads.pop(0))


def decision(decision_id, failure, kind, patch, evidence):
    return {
        "decision_id": decision_id,
        "decision": PlannerDecisionKind.RUN_EXPERIMENT.value,
        "asset_id": "mug",
        "failure_code": failure,
        "experiment_kind": kind,
        "hypothesis": f"Address {failure} with {kind}.",
        "strategy_patch": patch,
        "expected_effect": {evidence: "improve"},
        "required_evidence": [evidence],
        "confidence": "high",
        "rationale_summary": "Bounded fake decision for controller integration.",
    }


def accept_decision(asset_id="mug"):
    return {
        "decision_id": f"accept-{asset_id}",
        "decision": "ACCEPT_ASSET",
        "asset_id": asset_id,
        "failure_code": "GOAL_SATISFIED",
        "experiment_kind": None,
        "hypothesis": "All frozen objectives are verified.",
        "strategy_patch": None,
        "expected_effect": {},
        "required_evidence": [],
        "confidence": "high",
        "rationale_summary": "Accept only after the deterministic final gate.",
    }


class Scripted3DOperator:
    def __init__(self, source_path):
        self.spec = OperatorSpec(
            operator_id="dataevolver.3d_asset_evolution",
            version="1",
            retry_safe=True,
            expected_output_specs=(ExpectedOutputSpec("effect.json"),),
            model_config_hash="models-a",
            code_revision="code-a",
            source_path=str(source_path),
        )
        self.executions = []

    @staticmethod
    def _result(kind, asset_id=None):
        if kind == "INITIAL":
            return {
                "status": "FAILED",
                "metrics": {"geometry_structure_consistency": 0.3, "render_quality": 0.4, "identity_alignment": 1.0},
                "hard_constraints": {"artifact_valid": True},
                "failure_code": "GEOMETRY_MISMATCH",
            }
        if kind == "PROMPT_REGENERATE":
            return {
                "status": "SUCCESS",
                "metrics": {"geometry_structure_consistency": 0.85, "render_quality": 0.4, "identity_alignment": 1.0},
                "hard_constraints": {"artifact_valid": True},
                "failure_code": "LIGHTING_BAD",
            }
        return {
            "status": "SUCCESS",
            "metrics": {"geometry_structure_consistency": 0.85, "render_quality": 0.9, "identity_alignment": 1.0},
            "hard_constraints": {"artifact_valid": True},
            "failure_code": "",
        }

    def _receipt(self, call, reused):
        effect = Path(call.work_dir) / "effect.json"
        payload = json.loads(effect.read_text(encoding="utf-8"))
        now = datetime.now(timezone.utc).isoformat()
        return OperatorReceipt(
            call_id=call.call_id,
            operator_id=self.spec.operator_id,
            operator_version="1",
            status=StageStatus(payload["stage_result"]["status"]),
            started_at=now,
            finished_at=now,
            return_code=0,
            output_artifacts=({"path": str(effect), "sha256": _sha256_file(effect)},),
            effect_hash=_sha256_file(effect),
            reused=reused,
            stage_result={
                "operator_id": self.spec.operator_id,
                "operator_version": "1",
                **payload["stage_result"],
            },
        )

    def execute(self, call: OperatorCall):
        prior = _load_receipt(_receipt_path(call))
        if prior:
            return prior.with_updates(reused=True)
        effect = Path(call.work_dir) / "effect.json"
        if not effect.exists():
            kind = str(call.parameters.get("experiment_kind") or "INITIAL")
            self.executions.append(kind)
            atomic_write_json(effect, {"stage_result": self._result(kind, call.parameters.get("asset_id"))})
        return _write_receipt(_receipt_path(call), self._receipt(call, reused=False))

    def recover(self, call):
        receipt = _load_receipt(_receipt_path(call))
        if receipt:
            return RecoveryResult(RecoveryStatus.RECOVERED, receipt, "receipt found")
        effect = Path(call.work_dir) / "effect.json"
        if effect.exists():
            recovered = _write_receipt(_receipt_path(call), self._receipt(call, reused=True))
            return RecoveryResult(RecoveryStatus.RECOVERED, recovered, "effect recovered")
        return RecoveryResult(RecoveryStatus.RETRY_SAFE, None, "no effect")

    def verify_effect(self, call, receipt):
        effect = Path(call.work_dir) / "effect.json"
        if not effect.exists():
            return EffectVerification(EvidenceStatus.UNOBSERVED, "effect missing")
        actual = _sha256_file(effect)
        if receipt and receipt.effect_hash != actual:
            return EffectVerification(EvidenceStatus.CONTRADICTED, "hash mismatch")
        return EffectVerification(EvidenceStatus.VERIFIED, "effect verified", {str(effect): actual})


def write_evaluator(tmp_path):
    path = tmp_path / "evaluator.json"
    path.write_text(json.dumps(evaluator_payload(), sort_keys=True), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def report(evaluator, key, geometry, render, identity=1.0):
    return evaluator.evaluate(
        StageResult(
            operator_id="fake",
            operator_version="1",
            status=StageStatus.SUCCESS,
            metrics={
                "geometry_structure_consistency": geometry,
                "render_quality": render,
                "identity_alignment": identity,
            },
            hard_constraints={"artifact_valid": True},
            matched_trial_key=key,
        ),
        effect_status=EvidenceStatus.VERIFIED,
    )


def test_progressive_gate_adopts_targeted_improvement_before_final_goal(tmp_path):
    path, digest = write_evaluator(tmp_path)
    contract = EvaluatorContract.from_path(path)
    evaluator = DeterministicEvaluator(contract)
    key = trial_key(digest)

    result = CandidateGate(contract).decide(
        report(evaluator, key, 0.3, 0.4),
        report(evaluator, key, 0.85, 0.4),
        targeted_metrics=("geometry_structure_consistency",),
        progressive=True,
    )

    assert result.decision is CandidateDecision.ADOPT_CANDIDATE


def capability(novelty="novel-a", compatibility="contract-a"):
    patch = StrategyPatch.from_dict({"scene_actions": ["L_KEY_UP"], "restart_stage": "render"})
    intervention = Intervention(
        intervention_id="scene-fix",
        failure_code="LIGHTING_BAD",
        operator_id="dataevolver.3d_asset_evolution",
        level=InterventionLevel.L0,
        parameters={"strategy_patch": patch.to_dict()},
    )
    return Capability(
        capability_id=f"cap-{novelty}",
        version="1",
        status=CapabilityStatus.CANDIDATE,
        failure_signature=FailureSignature("LIGHTING_BAD"),
        intervention=intervention,
        strategy_patch=patch,
        experiment_kind=ExperimentKind.SCENE_REFINE,
        metric_delta={"render_quality": 0.3},
        protected_metric_delta={"identity_alignment": 0.0},
        context_signature="scene-lighting",
        compatibility_hash=compatibility,
        source_asset_id="mug",
        novelty_key=novelty,
    )


def test_capability_store_queries_compatible_novel_strategies_and_marks_failures(tmp_path):
    store = CapabilityStore(tmp_path / "capabilities.jsonl")
    assert store.append_candidate(capability()) is True
    assert store.append_candidate(capability()) is False
    store.append_candidate(capability("novel-b", "other-contract"))

    matches = store.query("scene-lighting", "LIGHTING_BAD", "contract-a", attempted=(), limit=5)
    assert [item.novelty_key for item in matches] == ["novel-a"]

    store.mark_known_failure("novel-a", asset_id="chair", compatibility_hash="contract-a", reason="regressed")
    assert store.query("scene-lighting", "LIGHTING_BAD", "contract-a", attempted=(), limit=5) == ()


def build_agentic_controller(
    tmp_path,
    decisions,
    *,
    asset_ids=("mug",),
    accepted_asset_quota=1,
    max_operator_calls=3,
    max_planner_calls=4,
):
    evaluator_path, evaluator_hash = write_evaluator(tmp_path)
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"schema_version":"test.schema.v1"}\n', encoding="utf-8")
    operator_source = tmp_path / "operator.py"
    operator_source.write_text("VERSION = 1\n", encoding="utf-8")
    key = trial_key(evaluator_hash)
    goal = GoalSpec(
        schema_version="dataevolver.harness.goal.v1",
        campaign_id="agentic-loop",
        objective={
            "kind": "agentic_3d_rsi",
            "asset_ids": list(asset_ids),
            "accepted_asset_quota": accepted_asset_quota,
            "initial_trial_key": key.to_dict(),
            "allowed_experiment_kinds": [item.value for item in ExperimentKind],
            "allowed_scene_actions": ["L_KEY_UP"],
        },
        initial_operator_id="dataevolver.3d_asset_evolution",
        route_candidates=("agentic_3d",),
        budgets=BudgetSpec(4, max_operator_calls, 2, 0, 60, max_planner_calls=max_planner_calls),
        resource_policy={"allow_gpu": False, "allowed_operators": ["dataevolver.3d_asset_evolution"]},
        frozen_evaluator={"path": str(evaluator_path), "sha256": evaluator_hash},
        planner={"kind": "agentic_rsi"},
        interventions=(),
        lease_seconds=10,
    )
    operator = Scripted3DOperator(operator_source)
    registry = OperatorRegistry()
    registry.register(operator)
    planner = AgenticRSIPlanner(backend=SequenceBackend(decisions), config={})
    compiler = ExperimentCompiler(
        operator_id="dataevolver.3d_asset_evolution",
        operator_version="1",
        allowed_scene_actions={"L_KEY_UP"},
    )
    controller = EvolutionController.create(
        repo_root=tmp_path,
        campaigns_root=tmp_path / "campaigns",
        goal=goal,
        registry=registry,
        planner=planner,
        evaluator_contract=EvaluatorContract.from_path(evaluator_path),
        schema_paths=(schema_path,),
        execution_paths=(operator_source,),
        experiment_compiler=compiler,
        idempotency_key="agentic-request",
    )
    return controller, operator


def test_agentic_controller_adopts_twice_reuses_memory_and_completes(tmp_path):
    decisions = [
        decision(
            "d1", "GEOMETRY_MISMATCH", "PROMPT_REGENERATE",
            {"prompt_text": "single red mug", "t2i_seed": 23, "restart_stage": "t2i"},
            "geometry_structure_consistency",
        ),
        decision(
            "d2", "LIGHTING_BAD", "SCENE_REFINE",
            {"scene_actions": ["L_KEY_UP"], "restart_stage": "render"},
            "render_quality",
        ),
        accept_decision(),
    ]
    controller, operator = build_agentic_controller(tmp_path, decisions)

    step = controller.tick(max_actions=1)
    assert step.previous_state.value == "INIT"
    assert step.new_state.value == "PLAN"
    assert step.action_count == 1
    final = controller.run()
    events = controller.store.events.read()
    event_types = [event.event_type for event in events]

    assert final.state.value == "COMPLETE"
    assert final.round_index >= 2
    assert final.experiments_used == 2
    assert operator.executions == ["INITIAL", "PROMPT_REGENERATE", "SCENE_REFINE"]
    assert event_types.count("INTERVENTION_ADOPTED") == 2
    assert "CAPABILITY_MEMORY_RETRIEVED" in event_types
    retrieved = [event for event in events if event.event_type == "CAPABILITY_MEMORY_RETRIEVED"]
    assert any(event.payload["count"] == 1 for event in retrieved)
    assert event_types.count("PLANNER_DECISION_RECORDED") == 3
    assert final.planner_calls == 3
    assert len(controller.capabilities.list_capabilities()) == 2
    traces = sorted((controller.layout.campaign_dir / "planner").glob("p*/trace.json"))
    assert len(traces) == 3
    assert all(json.loads(path.read_text(encoding="utf-8"))["raw_responses"] for path in traces)
    final_report = json.loads(controller.layout.final_report_path.read_text(encoding="utf-8"))
    assert final_report["capability_level"] == "agentic_3d_mock_verified"
    assert final_report["capabilities_learned"] == 2
    assert controller.replay().to_dict() == controller.status().to_dict()
    assert controller.explain(limit=2)["state"]["state"] == "COMPLETE"
    before_events = len(controller.store.events.read())
    before_commits = sum(event.event_type == "ACTION_COMMITTED" for event in controller.store.events.read())
    effect_files = sorted(controller.layout.campaign_dir.glob("rounds/**/effect.json"))
    before_effects = [(path, path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()) for path in effect_files]

    repeated = controller.run()

    assert repeated.state.value == "COMPLETE"
    assert len(controller.store.events.read()) == before_events
    assert sum(event.event_type == "ACTION_COMMITTED" for event in controller.store.events.read()) == before_commits
    assert [(path, path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()) for path in effect_files] == before_effects


def test_agentic_controller_rolls_back_then_plans_different_novelty(tmp_path):
    first = decision(
        "d1", "GEOMETRY_MISMATCH", "PROMPT_REGENERATE",
        {"prompt_text": "single red mug", "t2i_seed": 23, "restart_stage": "t2i"},
        "geometry_structure_consistency",
    )
    second = decision(
        "d2", "GEOMETRY_MISMATCH", "PROMPT_REGENERATE",
        {"prompt_text": "centered ceramic cup", "t2i_seed": 29, "restart_stage": "t2i"},
        "geometry_structure_consistency",
    )
    controller, operator = build_agentic_controller(tmp_path, [first, second, accept_decision()])
    original_result = operator._result
    attempts = {"prompt": 0}

    def result(kind, asset_id=None):
        if kind == "PROMPT_REGENERATE":
            attempts["prompt"] += 1
            if attempts["prompt"] == 1:
                return {
                    "status": "SUCCESS",
                    "metrics": {"geometry_structure_consistency": 0.85, "render_quality": 0.4, "identity_alignment": 0.7},
                    "hard_constraints": {"artifact_valid": True},
                    "failure_code": "GEOMETRY_MISMATCH",
                }
            return {
                "status": "SUCCESS",
                "metrics": {"geometry_structure_consistency": 0.9, "render_quality": 0.85, "identity_alignment": 1.0},
                "hard_constraints": {"artifact_valid": True},
                "failure_code": "",
            }
        return original_result(kind, asset_id)

    operator._result = result

    final = controller.run()
    events = controller.store.events.read()

    assert final.state.value == "COMPLETE"
    assert any(event.event_type == "CANDIDATE_ROLLED_BACK" for event in events)
    assert sum(event.event_type == "PLANNER_DECISION_RECORDED" for event in events) == 3
    assert final.planner_calls == 3
    assert len(set(final.data["attempted_strategy_hashes"])) == 2


def test_agentic_controller_exhausts_independent_planner_budget_after_adopt(tmp_path):
    first = decision(
        "d1", "GEOMETRY_MISMATCH", "PROMPT_REGENERATE",
        {"prompt_text": "single red mug", "t2i_seed": 23, "restart_stage": "t2i"},
        "geometry_structure_consistency",
    )
    controller, operator = build_agentic_controller(
        tmp_path,
        [first],
        max_planner_calls=1,
        max_operator_calls=2,
    )

    final = controller.run()
    report = json.loads(controller.layout.final_report_path.read_text(encoding="utf-8"))

    assert final.state.value == "ABSTAIN"
    assert final.terminal_reason == "BUDGET_EXHAUSTED"
    assert final.planner_calls == 1
    assert report["terminal_state"] == "BUDGET_EXHAUSTED"
    assert operator.executions == ["INITIAL", "PROMPT_REGENERATE"]


def test_agentic_controller_processes_asset_queue_without_hard_coding_one_asset(tmp_path):
    controller, operator = build_agentic_controller(
        tmp_path,
        [],
        asset_ids=("mug", "chair"),
        accepted_asset_quota=2,
        max_operator_calls=2,
    )

    def successful_initial(kind, asset_id=None):
        assert kind == "INITIAL"
        return {
            "status": "SUCCESS",
            "metrics": {"geometry_structure_consistency": 0.9, "render_quality": 0.9, "identity_alignment": 1.0},
            "hard_constraints": {"artifact_valid": True},
            "failure_code": "",
        }

    operator._result = successful_initial

    final = controller.run()

    assert final.state.value == "COMPLETE"
    assert final.data["accepted_assets"] == ["mug", "chair"]
    assert final.operator_calls == 2
    assert operator.executions == ["INITIAL", "INITIAL"]


def test_verified_abandon_decision_advances_to_next_asset(tmp_path):
    abandon = {
        "decision_id": "abandon-mug",
        "decision": "ABANDON_ASSET",
        "asset_id": "mug",
        "failure_code": "ASSET_ABANDON_RECOMMENDED",
        "experiment_kind": None,
        "hypothesis": "The current asset is not viable.",
        "strategy_patch": None,
        "expected_effect": {},
        "required_evidence": [],
        "confidence": "high",
        "rationale_summary": "Verified viability evidence recommends abandonment.",
    }
    controller, operator = build_agentic_controller(
        tmp_path,
        [abandon],
        asset_ids=("mug", "chair"),
        accepted_asset_quota=1,
        max_operator_calls=2,
    )

    def by_asset(kind, asset_id=None):
        if asset_id == "mug":
            return {
                "status": "FAILED",
                "metrics": {"geometry_structure_consistency": 0.1, "render_quality": 0.1, "identity_alignment": 1.0},
                "hard_constraints": {"artifact_valid": True},
                "failure_code": "ASSET_ABANDON_RECOMMENDED",
            }
        return {
            "status": "SUCCESS",
            "metrics": {"geometry_structure_consistency": 0.9, "render_quality": 0.9, "identity_alignment": 1.0},
            "hard_constraints": {"artifact_valid": True},
            "failure_code": "",
        }

    operator._result = by_asset

    final = controller.run()

    assert final.state.value == "COMPLETE"
    assert final.data["abandoned_assets"] == ["mug"]
    assert final.data["accepted_assets"] == ["chair"]
    assert operator.executions == ["INITIAL", "INITIAL"]
