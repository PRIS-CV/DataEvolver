"""Version, scope, revocation, committed authority and recovery invariants."""
import copy
import json

import pytest

from dataevolver.harness.operator_lifecycle import (OperatorVersionStore, EMPTY_HEAD, SYNTHETIC_SCOPE,
    qualification_snapshot, engineering_origin_matches)
from dataevolver.dataset.contracts import payload_sha256
from dataevolver.harness.state_store import atomic_write_json


def snapshot():
    return {"operator_id": "data.test_authority", "version": "1", "scope": SYNTHETIC_SCOPE,
            "package_sha256": "a" * 64}


def adopt(store, owner="fixture-adopter"):
    with store.lease():
        return store.append(action="ADOPTED", snapshot=snapshot(), owner={"fixture": owner}, expected_head=store.head())


def test_catalog_immutable_version_and_revocation_preserve_history(tmp_path):
    store = OperatorVersionStore(tmp_path / "versions")
    event = adopt(store)
    assert store.active(event, scope=SYNTHETIC_SCOPE, head=store.head()) == snapshot()
    with store.lease():
        revocation = store.append(action="REVOKED", snapshot=snapshot(), owner={"fixture": "revoker"},
                                  expected_head=store.head(), reason="qualification scope withdrawn")
    assert revocation["previous_sha256"] == event["sha256"] and store.included(event) == event
    with pytest.raises(ValueError):
        store.active(event, scope=SYNTHETIC_SCOPE, head=store.head())
    with pytest.raises(ValueError, match="already exists"):
        adopt(store, "another-adopter")


@pytest.mark.parametrize("change", ["scope", "head", "snapshot", "action"])
def test_scope_or_snapshot_cannot_silently_escalate(tmp_path, change):
    store = OperatorVersionStore(tmp_path)
    event = adopt(store)
    args = {"scope": SYNTHETIC_SCOPE, "head": store.head()}
    if change == "scope":
        args["scope"] = "production_images"
    elif change == "head":
        args["head"] = EMPTY_HEAD
    else:
        event = copy.deepcopy(event)
        if change == "snapshot":
            event["snapshot"]["version"] = "2"
        else:
            event["action"] = "REVOKED"
    with pytest.raises(ValueError):
        store.active(event, **args)


def test_catalog_same_owner_recovery_is_exact_and_nonduplicating(tmp_path):
    store = OperatorVersionStore(tmp_path)
    event = adopt(store)
    with store.lease():
        assert store.append(action="ADOPTED", snapshot=snapshot(), owner=event["owner"], expected_head=EMPTY_HEAD) == event
        with pytest.raises(ValueError, match="cannot change"):
            store.append(action="REVOKED", snapshot=snapshot(), owner=event["owner"], expected_head=store.head(), reason="different")
    assert len(store.events()) == 1


@pytest.mark.parametrize("change", ["gap", "hash", "symlink", "duplicate_version"])
def test_catalog_corruption_refuses_resolution(tmp_path, change):
    store = OperatorVersionStore(tmp_path / "versions")
    event = adopt(store)
    path = store.root / "00000001.json"
    if change == "gap":
        path.rename(store.root / "00000002.json")
    elif change == "symlink":
        target = tmp_path / "event.json"
        path.rename(target)
        path.symlink_to(target)
    elif change == "hash":
        event["snapshot"]["version"] = "changed"
        atomic_write_json(path, event)
    else:
        body = {**{k: v for k, v in event.items() if k != "sha256"}, "sequence": 2,
                "previous_sha256": event["sha256"], "owner": {"fixture": "forged"}}
        atomic_write_json(store.root / "00000002.json", {**body, "sha256": payload_sha256(body)})
    with pytest.raises(ValueError):
        store.events()


def test_summary_json_cannot_masquerade_as_historical_qualification():
    with pytest.raises(ValueError, match="historical Controller readers"):
        qualification_snapshot(package=None, store=None, engineering={"PASSED": True}, independent={"PASSED": True}, runtime={})


@pytest.mark.parametrize('change', ['none', 'wrong_historical_reader', 'wrong_descriptor', 'wrong_report'])
def test_qualification_origin_preserves_original_reader_without_ignoring_other_identity(change):
    current = {'reader_sha256': 'new-reader', 'descriptor': {'config': 'exact-config'}, 'frozen_sha256': 'a'*64}
    origin = {'authority': {**copy.deepcopy(current), 'reader_sha256': 'original-reader'},
              'report_sha256': 'b'*64, 'package_sha256': 'c'*64}
    if change == 'wrong_historical_reader':
        origin['authority']['reader_sha256'] = 'unverified-reader'
    elif change == 'wrong_descriptor':
        origin['authority']['descriptor']['config'] = 'other-config'
    elif change == 'wrong_report':
        origin['report_sha256'] = 'd'*64
    assert engineering_origin_matches(origin, current, 'original-reader', 'b'*64, 'c'*64) is (change == 'none')


@pytest.mark.parametrize('change', ['none', 'wrong_commit', 'unbound_env'])
def test_legacy_environment_commit_is_derived_only_from_original_bound_source(change):
    original = {'reader_sha256': 'old', 'descriptor': {'env_path': '/host/config/env.sh'},
                'source_commits': {'/host/config/source': 'a'*40}}
    current = {**copy.deepcopy(original), 'reader_sha256': 'new', 'env_source_commit': 'a'*40}
    if change == 'wrong_commit':
        current['env_source_commit'] = 'b'*40
    elif change == 'unbound_env':
        original['source_commits'] = {}
        current['source_commits'] = {}
    origin = {'authority': original, 'report_sha256': 'c'*64, 'package_sha256': 'd'*64}
    assert engineering_origin_matches(origin, current, 'old', 'c'*64, 'd'*64) is (change == 'none')


def mutation_fixture(tmp_path, monkeypatch):
    from dataevolver.harness.adapters import operator_version_lifecycle as module
    from dataevolver.harness.operator_development import DevelopmentBudget
    prior = dict(operator_calls=9, planner_calls=2, model_calls=4, image_experiments=0, output_bytes=100, wall_seconds=1)
    value = {**snapshot(), "lifecycle_total": prior, "resource_deadline_epoch": 1}
    monkeypatch.setattr(module, "qualification_snapshot", lambda **kwargs: copy.deepcopy(value))
    class Store:
        roots = {"candidate": tmp_path / "candidate"}
    budget = DevelopmentBudget({**prior, "operator_calls": 10, "output_bytes": 100000, "wall_seconds": 100}, prior, 1)
    op = module.OperatorVersionMutation(catalog=OperatorVersionStore(tmp_path / "versions"), expected_head=EMPTY_HEAD,
        budget=budget, output_root=tmp_path / "output", package=None, store=Store())
    return op, value


def test_adoption_needs_controller_commit_and_finished_effect_is_never_rewritten(tmp_path, monkeypatch):
    from dataevolver.harness.models import OperatorCall, EvidenceStatus, RecoveryStatus
    from dataevolver.harness.operator_lifecycle import CommittedOperatorAdoptionAuthority
    op, _ = mutation_fixture(tmp_path, monkeypatch)
    call = OperatorCall("test-adoption", "adopt-1", "adopt-1", op.spec.operator_id, "1", 0, str(op.output_root / "call"))
    receipt = op.execute(call)
    assert op.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    assert op.execute(call).reused and len(op.catalog.events()) == 1
    path = op._path(call)
    value = json.loads(path.read_text())
    value["event"]["snapshot"]["version"] = "forged"
    atomic_write_json(path, value)
    assert op.recover(call).status is RecoveryStatus.AMBIGUOUS
    with pytest.raises(ValueError):
        op.execute(call)


def test_intent_without_durable_effect_is_ambiguous_and_never_repeated(tmp_path, monkeypatch):
    from dataevolver.harness.models import OperatorCall, RecoveryStatus
    op, _ = mutation_fixture(tmp_path, monkeypatch)
    call = OperatorCall("test-adoption", "adopt-1", "adopt-1", op.spec.operator_id, "1", 0, str(op.output_root / "call"))
    atomic_write_json(op._path(call).parent / "intent.json", {"owner": op._owner(call), "at": "fixture"})
    assert op.recover(call).status is RecoveryStatus.AMBIGUOUS
    assert op.catalog.events() == []


def test_controller_evaluator_rejection_does_not_authorize_catalog_entry(tmp_path, monkeypatch):
    from dataevolver.testing.test_harness_controller import build_controller
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.operators import OperatorRegistry
    from dataevolver.harness.planner import DeterministicPlanner
    from dataevolver.harness.evaluator import EvaluatorContract
    from dataevolver.harness.operator_lifecycle import CommittedOperatorAdoptionAuthority
    from dataevolver.dataset.contracts import file_sha256
    op, _ = mutation_fixture(tmp_path, monkeypatch)
    (tmp_path / "template").mkdir()
    template, _ = build_controller(tmp_path / "template")
    evaluator = tmp_path / "evaluator.json"
    atomic_write_json(evaluator, {"schema_version": "dataevolver.harness.evaluator.v1", "evaluator_id": "reject-adoption", "revision": "1",
        "hard_constraints": {"artifact_valid": {"required": True}}, "target_metrics": {"operator_adopted": {"required": True, "minimum": 2}}, "protected_metrics": {}})
    goal = template.goal.with_updates(campaign_id="rejected-adoption", initial_operator_id=op.spec.operator_id, interventions=(),
        resource_policy={"allow_gpu": False, "allowed_operators": [op.spec.operator_id]},
        frozen_evaluator={"path": str(evaluator), "sha256": file_sha256(evaluator)})
    registry = OperatorRegistry(); registry.register(op)
    c = EvolutionController.create(repo_root=tmp_path, campaigns_root=op.output_root / "campaigns", goal=goal,
        registry=registry, planner=DeterministicPlanner(()), evaluator_contract=EvaluatorContract.from_path(evaluator), idempotency_key=goal.campaign_id)
    c.run()
    call = next(e.payload["call_id"] for e in c.store.events.read() if e.event_type == "ACTION_COMMITTED")
    assert len(op.catalog.events()) == 1
    with pytest.raises(ValueError, match="complete"):
        CommittedOperatorAdoptionAuthority(c, op, call).read()


def test_subsequent_request_is_a_new_authenticated_family_and_geometry(tmp_path):
    from dataevolver.testing.independent_authority_vectors import make_subsequent_request
    from dataevolver.harness.candidate_trust import CandidateTrustContext
    from dataevolver.harness.authority_qualification import AuthorityFixtureExpectation, validate_fixture
    v = make_subsequent_request(tmp_path / "subsequent", package_sha256="a"*64,
        implementation_sha256="b"*64, request_sha256="c"*64)
    assert len(v["cases"]) == 1 and v["family_ids"] == ["subsequent-angled-aperture"]
    read = lambda ref: json.loads((tmp_path / ref["path"]).read_text())
    case = v["cases"][0]
    payload = read(case["input"])
    context = CandidateTrustContext.from_dict(read(v["trust_contexts"][case["case_id"]]))
    expected = AuthorityFixtureExpectation.from_dict(read(v["expectations"][case["case_id"]]))
    validate_fixture(payload, context, expected)
    assert payload["evidence"]["source"]["manifest"]["width"] == 1280
    assert expected.status == "executable" and len(expected.resolved_pixels) == 12
