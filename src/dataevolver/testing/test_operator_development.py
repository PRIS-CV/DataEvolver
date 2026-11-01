"""CPU contract/recovery evidence only; no production quality qualification."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, file_sha256, payload_sha256
from dataevolver.harness.operator_development import (
    USAGE_KEYS, CandidatePackage, ControllerQualificationAuthority, ControllerRequestAuthority,
    DevelopmentBudget, OperatorDevelopmentPlan, QualificationProtocol, assess_candidate, inspect_candidate,
)
from dataevolver.harness.adapters.operator_development import CandidateInspectionOperator
from dataevolver.harness.controller import EvolutionController
from dataevolver.harness.evaluator import EvaluatorContract
from dataevolver.harness.state_store import atomic_write_json
from dataevolver.harness.models import ControllerState, EvidenceStatus, OperatorCall, OperatorSpec, StageStatus
from dataevolver.harness.operators import FakeOperator, OperatorRegistry, _receipt_path
from dataevolver.testing.test_harness_controller import build_controller
from dataevolver.testing.test_operator_catalog import declaration
from dataevolver.testing.test_runtime_capability_requests import decision, request
from dataevolver.testing.test_runtime_rsi import make_runtime


def write_ref(root, store, path, value, *, raw=False):
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(value if raw else json.dumps(value, sort_keys=True))
    return ArtifactRef(store, path, file_sha256(file)).to_dict()


def fixture(tmp_path):
    parent_root = tmp_path / "origin"
    parent_root.mkdir()
    c, backend = make_runtime(parent_root, [decision()])
    original = c.planner.domain.context
    c.planner.domain.context = lambda s: {**original(s), "capability_requests_allowed": True}
    c.planner.domain.compile_capability_request = lambda d, s: {
        "request": request(), "proposal_actor": "runtime_agent", "decision": d,
        "production_effects": 0, "status": "IMPLEMENTATION_REQUIRED"}
    assert c.run().terminal_reason == "CAPABILITY_REQUESTED"
    request_file = Path(c.status().data["rsi_capability_request"]["path"])
    roots = {"origin": parent_root, "protocol": tmp_path / "protocol", "candidate": tmp_path / "candidate"}
    for root in roots.values():
        root.mkdir(exist_ok=True)
    store = ArtifactStore(roots)
    candidate = roots["candidate"]
    # Importing this file would have an externally visible effect. Inspection
    # must only hash it, including when its filename looks like executable code.
    implementation = write_ref(candidate, "candidate", "operator.py",
        "from pathlib import Path\nPath(" + repr(str(tmp_path / "EXECUTED")) + ").touch()\n", raw=True)
    spec = OperatorSpec("candidate.reusable", "1", source_path="operator.py", code_revision=implementation["sha256"], retry_safe=True)
    contract = declaration(spec)
    spec_ref = write_ref(candidate, "candidate", "spec.json", spec.to_dict())
    contract_ref = write_ref(candidate, "candidate", "contract.json", contract.to_dict())
    deps = write_ref(candidate, "candidate", "dependencies.json",
                     {"schema_version": "dataevolver.candidate_dependencies.v1", "dependencies": []})
    changes = write_ref(candidate, "candidate", "changes.txt", "Fixture candidate, never activated.", raw=True)
    sources = []
    for name, split in [("dev", "development"), ("regression", "regression"),
                        ("independent-a", "independent"), ("independent-b", "independent"), ("hold", "heldout")]:
        ref = write_ref(roots["protocol"], "protocol", name + ".json", {"family": name})
        sources.append({"family_id": name, "split": split, "artifact": ref})
    template_root = tmp_path / "template"
    template_root.mkdir()
    template, _ = build_controller(template_root)
    validators = {stage: OperatorSpec("qualified." + stage, "1", retry_safe=True,
                         source_path=str(template_root / "operator.py"), code_revision="trusted-fixture-validator")
                  for stage in ("engineering", "independent")}
    protocol = QualificationProtocol("frozen-fixture-protocol", tuple(sources), {
        stage: {"validator_operator_id": spec.operator_id, "validator_spec_sha256": spec.sha256(),
                "evaluator_sha256": template.evaluator.contract.sha256,
                "check_ids": ["preserve", "matched-comparison", "negative-cases"]}
        for stage, spec in validators.items()}, "a" * 64)
    protocol_ref = write_ref(roots["protocol"], "protocol", "protocol.json", protocol.to_dict())
    budget = DevelopmentBudget({key: 1000000 for key in USAGE_KEYS}, {key: 0 for key in USAGE_KEYS}, 1792328957.7055008)
    plan = OperatorDevelopmentPlan("candidate-development", ArtifactRef("origin",
        request_file.relative_to(parent_root).as_posix(), file_sha256(request_file)).to_dict(),
        c.goal.campaign_id, file_sha256(c.layout.frozen_contract_path), protocol_ref,
        contract.compute_hash(), "candidate", ("origin", "protocol"), budget.to_dict())
    package = CandidatePackage("candidate-v1", plan.compute_hash(),
        (implementation, spec_ref, contract_ref, deps, changes), implementation, spec_ref, contract_ref, deps, changes)
    return c, backend, plan, package, store, protocol, validators


def inspect(f):
    c, _, plan, package, store, _, _ = f
    return inspect_candidate(plan, package, store, ControllerRequestAuthority(c))


def test_inert_candidate_is_hash_closed_without_qualification_or_model_calls(tmp_path):
    f = fixture(tmp_path)
    c, backend, plan, package, store, _, _ = f
    before = c.layout.events_path.read_bytes()
    result = inspect(f)
    assert result["status"] == "STRUCTURE_VERIFIED" and len(result["files"]) == 5
    assert result["engineering_qualification"] == result["independent_qualification"] == "NOT_ASSESSED"
    assert not result["promoted"] and not result["builder_verified"] and not result["candidate_code_executed"]
    assert not (tmp_path / "EXECUTED").exists() and len(backend.calls) == 1
    assert c.layout.events_path.read_bytes() == before
    assert OperatorDevelopmentPlan.from_dict(plan.to_dict()) == plan
    assert CandidatePackage.from_dict(package.to_dict()) == package
    assert assess_candidate(plan, package, store, ControllerRequestAuthority(c), {})["qualification_evidence"] == {}


@pytest.mark.parametrize("which", ["extra", "modified", "symlink", "hardlink", "overlap", "request", "plan", "snapshot"])
def test_candidate_and_origin_tampering_fail_closed(tmp_path, which):
    f = list(fixture(tmp_path))
    c, _, plan, package, store, _, _ = f
    root = store.roots["candidate"]
    if which == "extra":
        (root / "undeclared.py").write_text("bad")
    elif which == "modified":
        (root / "operator.py").write_text("changed")
    elif which == "symlink":
        (root / "operator.py").unlink()
        (root / "operator.py").symlink_to(store.resolve(ArtifactRef.from_dict(plan.qualification_protocol)))
    elif which == "hardlink":
        import os
        os.link(root / "operator.py", tmp_path / "alias.py")
    elif which == "overlap":
        f[4] = ArtifactStore({**store.roots, "overlap": root / "inside"})
    elif which == "request":
        store.resolve(ArtifactRef.from_dict(plan.capability_request)).write_text('{}')
    elif which == "plan":
        f[2] = replace(plan, target_contract_sha256="e" * 64)
    else:
        state = json.loads(c.layout.state_path.read_text())
        state["planner_calls"] += 1
        c.layout.state_path.write_text(json.dumps(state))
    with pytest.raises((ValueError, KeyError)):
        inspect(tuple(f))
    assert not (tmp_path / "EXECUTED").exists()


@pytest.mark.parametrize("field", ["promoted", "verified", "engineering_qualification", "shell"])
def test_package_cannot_self_report_authority(tmp_path, field):
    f = fixture(tmp_path)
    with pytest.raises(ValueError, match="unknown CandidatePackage"):
        CandidatePackage.from_dict({**f[3].to_dict(), field: True})


def test_registered_inspection_recovers_without_candidate_execution_and_detects_drift(tmp_path):
    c, _, plan, package, store, _, _ = fixture(tmp_path)
    operator = CandidateInspectionOperator(plan=plan, package=package, store=store,
        request_authority=ControllerRequestAuthority(c), output_root=tmp_path / "inspection")
    registry = OperatorRegistry()
    registry.register(operator)
    assert registry.catalog()["operators"][0]["contract"]["category"] == "evaluation"
    call = OperatorCall("inspection", "inspect-1", "inspect-1", operator.spec.operator_id, "1", 0,
                        str(tmp_path / "inspection" / "call-1"))
    receipt = operator.execute(call)
    assert operator.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    _receipt_path(call).unlink()  # Crash after durable effect, before receipt commit.
    recovered = operator.recover(call).receipt
    assert recovered.reused and recovered.effect_hash == receipt.effect_hash
    assert not (tmp_path / "EXECUTED").exists()
    (store.roots["candidate"] / "operator.py").write_text("changed")
    assert operator.verify_effect(call, recovered).status is EvidenceStatus.CONTRADICTED
    with pytest.raises(ValueError):
        operator.execute(call)


class QualificationFixture(FakeOperator):
    def __init__(self, spec, payload):
        super().__init__(spec, status=StageStatus.SUCCESS, metrics={"quality": .9, "safety": 1.0},
                         hard_constraints={"artifact_valid": True})
        self.payload = payload

    def _receipt_from_effect(self, call, *, reused):
        receipt = super()._receipt_from_effect(call, reused=reused)
        return receipt.with_updates(stage_result={**receipt.stage_result, **self.payload})

    def verify_effect(self, call, receipt):
        if receipt is not None:
            for key, value in self.payload.items():
                if receipt.stage_result.get(key) != value:
                    raise ValueError("trusted fixture validator rejects modified result")
        return super().verify_effect(call, receipt)


def qualify(tmp_path, f, stage, *, changes=None):
    _, _, plan, package, _, protocol, validators = f
    root = tmp_path / stage
    root.mkdir()
    old, _ = build_controller(root)
    binding = {"plan_sha256": plan.compute_hash(), "package_sha256": package.compute_hash(),
               "protocol_sha256": protocol.compute_hash(), "stage": stage,
               "matched_context_sha256": protocol.matched_context_sha256}
    payload = {"qualification_binding": binding, "source_family_ids": protocol.families(stage),
               "check_results": {key: True for key in protocol.stages[stage]["check_ids"]},
               "lifecycle_usage": {key: 1 for key in USAGE_KEYS}}
    payload.update(changes or {})
    registry = OperatorRegistry()
    registry.register(QualificationFixture(validators[stage], payload))
    goal = old.goal.with_updates(campaign_id=stage,
        initial_operator_id=validators[stage].operator_id,
        resource_policy={"allow_gpu": False, "allowed_operators": [validators[stage].operator_id]},
        objective={**old.goal.objective, "operator_qualification": binding}, interventions=())
    c = EvolutionController.create(repo_root=root, campaigns_root=root / "campaigns", goal=goal,
        registry=registry, planner=old.planner, evaluator_contract=old.evaluator.contract,
        idempotency_key="qualification-" + stage)
    assert c.run().state is ControllerState.COMPLETE
    event = next(e for e in c.store.events.read() if e.event_type == "ACTION_COMMITTED")
    return ControllerQualificationAuthority(c, event.payload["call_id"])


def test_qualification_reads_actual_controller_effects_and_frozen_evaluator(tmp_path):
    f = fixture(tmp_path)
    c, _, plan, package, store, _, _ = f
    authorities = {stage: qualify(tmp_path, f, stage) for stage in ("engineering", "independent")}
    before = {stage: a.controller.layout.events_path.read_bytes() for stage, a in authorities.items()}
    result = assess_candidate(plan, package, store, ControllerRequestAuthority(c), authorities)
    assert result["engineering_qualification"] == result["independent_qualification"] == "VERIFIED"
    assert result["lifecycle_usage"] == {key: 2 + (result["total_bytes"] if key == "output_bytes" else 0)
                                         for key in USAGE_KEYS}
    assert not result["promoted"] and not result["reused"] and not result["method_improvement_verified"]
    assert before == {stage: a.controller.layout.events_path.read_bytes() for stage, a in authorities.items()}


@pytest.mark.parametrize("changes", [
    {"check_results": {"preserve": True}}, {"source_family_ids": ["hold"]},
    {"qualification_binding": {"verified": True}},
    {"lifecycle_usage": {key: 1000001 for key in USAGE_KEYS}},
])
def test_self_reported_or_mismatched_qualification_cannot_pass(tmp_path, changes):
    f = fixture(tmp_path)
    c, _, plan, package, store, _, _ = f
    authority = qualify(tmp_path, f, "engineering", changes=changes)
    with pytest.raises(ValueError):
        assess_candidate(plan, package, store, ControllerRequestAuthority(c), {"engineering": authority})


def test_protocol_rejects_independent_family_or_content_leakage(tmp_path):
    f = fixture(tmp_path)
    protocol = f[5]
    sources = [dict(s) for s in protocol.sources]
    sources[-1]["artifact"] = sources[0]["artifact"]
    with pytest.raises(ValueError, match="isolated"):
        replace(protocol, sources=sources)
    with pytest.raises(ValueError, match="total budget"):
        DevelopmentBudget({key: 1 for key in USAGE_KEYS}, {key: 2 for key in USAGE_KEYS}, 123)
    with pytest.raises(ValueError, match="finite"):
        DevelopmentBudget({key: 1 for key in USAGE_KEYS}, {key: 0 for key in USAGE_KEYS}, float('nan'))


def test_registered_inspection_executes_under_the_existing_controller_and_replays(tmp_path):
    f = fixture(tmp_path)
    origin, _, plan, package, store, _, _ = f
    root = tmp_path / "inspection-campaign"
    root.mkdir()
    template, _ = build_controller(root)
    operator = CandidateInspectionOperator(plan=plan, package=package, store=store,
        request_authority=ControllerRequestAuthority(origin), output_root=root / "outputs")
    registry = OperatorRegistry()
    registry.register(operator)
    evaluator = root / "inspection-evaluator.json"
    atomic_write_json(evaluator, {"schema_version": "dataevolver.harness.evaluator.v1",
        "evaluator_id": "structure-only", "revision": "1",
        "hard_constraints": {"artifact_valid": {"required": True}},
        "target_metrics": {"structure_verified": {"required": True, "minimum": 1}}, "protected_metrics": {}})
    goal = template.goal.with_updates(campaign_id="registered-inspection", initial_operator_id=operator.spec.operator_id,
        resource_policy={"allow_gpu": False, "allowed_operators": [operator.spec.operator_id]},
        frozen_evaluator={"path": str(evaluator), "sha256": file_sha256(evaluator)}, interventions=())
    c = EvolutionController.create(repo_root=root, campaigns_root=root / "outputs", goal=goal,
        registry=registry, planner=template.planner, evaluator_contract=EvaluatorContract.from_path(evaluator),
        idempotency_key="registered-inspection")
    end = c.run()
    assert end.state is ControllerState.COMPLETE and end.operator_calls == 1 and end.experiments_used == 0
    assert c.replay().to_dict() == end.to_dict()
    assert not c.capabilities.list_capabilities()
    before = c.layout.events_path.read_bytes()
    c.run()
    assert c.layout.events_path.read_bytes() == before and not (tmp_path / "EXECUTED").exists()


def test_failed_or_drifted_validator_does_not_qualify(tmp_path):
    f = fixture(tmp_path)
    c, _, plan, package, store, _, _ = f
    authority = qualify(tmp_path, f, "engineering")
    authority.controller.evaluator.contract.payload["target_metrics"]["quality"]["minimum"] = .99
    with pytest.raises(ValueError, match="rejected"):
        assess_candidate(plan, package, store, ControllerRequestAuthority(c), {"engineering": authority})


def test_output_receipt_symlink_cannot_escape_the_inspection_root(tmp_path):
    c, _, plan, package, store, _, _ = fixture(tmp_path)
    operator = CandidateInspectionOperator(plan=plan, package=package, store=store,
        request_authority=ControllerRequestAuthority(c), output_root=tmp_path / "out")
    call = OperatorCall("inspect", "call", "call", operator.spec.operator_id, "1", 0, str(tmp_path / "out/call"))
    work = Path(call.work_dir)
    work.mkdir(parents=True)
    (work / ".receipts").symlink_to(tmp_path / "origin", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        operator.execute(call)
