"""Contract/recovery engineering evidence only, never production qualification."""
import json
from dataclasses import replace

import pytest
from jsonschema import ValidationError

from dataevolver.harness.operator_contracts import OperatorContract
from dataevolver.harness.operators import OperatorRegistry
from dataevolver.harness.models import ControllerState, OperatorCall
from dataevolver.testing.test_harness_controller import build_controller


def declaration(spec, **changes):
    values = dict(operator_id=spec.operator_id, version=spec.version,
        spec_sha256=spec.sha256(), category="basic", input_schemas={},
        output_schemas={"effect": {"type": "object"}},
        parameter_schema={"type": "object", "additionalProperties": False,
                          "properties": {"allowed": {"type": "integer"}}},
        modalities=("json",), effect_semantics="pure_data",
        recovery_semantics="verify_then_retry", limitations=("Fixture only.",))
    return OperatorContract(**{**values, **changes})


def registry_with_contract(controller, **changes):
    registry = OperatorRegistry()
    for spec in controller.registry.specs():
        operator = controller.registry.get(spec.operator_id)
        registry.register(operator, contract=declaration(spec, **changes)
                          if spec.operator_id == controller.goal.initial_operator_id else None)
    return registry


def test_catalog_reads_live_registry_without_effects_or_readiness_claim(tmp_path):
    c, _ = build_controller(tmp_path)
    registry = registry_with_contract(c)
    before = c.layout.events_path.read_bytes() if c.layout.events_path.exists() else b""
    first = registry.catalog()
    declared = next(x for x in first["operators"] if x["declaration_status"] == "DECLARED")
    assert all(x["production_qualification"] == "NOT_ASSESSED" for x in first["operators"])
    assert not first["inference_performed"] and not first["qualification_performed"]
    declared["contract"]["parameter_schema"]["properties"]["unsafe"] = {}
    assert registry.catalog() != first
    assert not c.layout.events_path.exists() or c.layout.events_path.read_bytes() == before
    assert not list(c.layout.campaign_dir.rglob("effect.json"))


def test_closed_parameters_are_checked_before_controller_commits_or_runs(tmp_path):
    c, _ = build_controller(tmp_path)
    c.registry = registry_with_contract(c)
    spec = c.registry.get(c.goal.initial_operator_id).spec
    call = OperatorCall(campaign_id=c.goal.campaign_id, call_id="bad-input", idempotency_key="bad-input",
        operator_id=spec.operator_id, operator_version=spec.version, round_index=0,
        work_dir=str(tmp_path / "unexecuted"), parameters={"model": "substitute"})
    before = c.status().to_dict()
    with pytest.raises(ValidationError):
        c._execute_call(c.status(), call)
    assert c.status().to_dict() == before
    assert not (tmp_path / "unexecuted").exists()


def test_new_declaration_is_drift_even_if_old_snapshot_had_no_contract_key(tmp_path):
    c, _ = build_controller(tmp_path)
    c.registry = registry_with_contract(c)
    c.contract_guard.registry = c.registry
    end = c.tick()
    assert end.new_state is ControllerState.PAUSED_CONTRACT_DRIFT
    assert "operator_contracts_hash" in c.status().data["contract_drift"]
    assert not list(c.layout.campaign_dir.rglob("effect.json"))


def test_declared_contract_change_is_detected_before_next_effect(tmp_path):
    c, _ = build_controller(tmp_path)
    c.registry = registry_with_contract(c)
    c.contract_guard.registry = c.registry
    frozen = c.contract_guard.freeze()
    c.layout.frozen_contract_path.write_text(json.dumps(frozen))
    c.registry = registry_with_contract(c, limitations=("Changed scope.",))
    c.contract_guard.registry = c.registry
    assert c.tick().new_state is ControllerState.PAUSED_CONTRACT_DRIFT
    assert not list(c.layout.campaign_dir.rglob("effect.json"))


def test_unknown_frozen_contract_key_is_not_treated_as_missing_null(tmp_path):
    c, _ = build_controller(tmp_path)
    frozen = c.contract_guard.freeze()
    frozen['unrecognized_authority'] = None
    drift = c.contract_guard.verify(frozen)
    assert not drift.valid and 'unrecognized_authority' in drift.changed_fields


def test_declaration_cannot_change_version_implementation_or_retry_authority(tmp_path):
    c, _ = build_controller(tmp_path)
    spec = c.registry.specs()[0]
    declared = declaration(spec)
    for changed in [replace(spec, version="2"), replace(spec, code_revision="different")]:
        with pytest.raises(ValueError, match="registered implementation"):
            declared.bound_to(changed)
    with pytest.raises(ValueError, match="retry authority"):
        declaration(replace(spec, retry_safe=False)).bound_to(replace(spec, retry_safe=False))


@pytest.mark.parametrize("ref", ["https://invalid.example/schema", "file:///etc/passwd", "other.json"])
def test_operator_schema_rejects_external_resolution(tmp_path, ref):
    c, _ = build_controller(tmp_path)
    with pytest.raises(ValueError, match="external references"):
        declaration(c.registry.specs()[0], input_schemas={"data": {"$ref": ref}})


def test_contract_roundtrip_cannot_smuggle_readiness_or_runtime_authority(tmp_path):
    c, _ = build_controller(tmp_path)
    value = declaration(c.registry.specs()[0]).to_dict()
    assert OperatorContract.from_dict(value).to_dict() == value
    with pytest.raises(ValueError, match="unknown OperatorContract"):
        OperatorContract.from_dict({**value, "production_ready": True})
    with pytest.raises(ValueError, match="closed object"):
        declaration(c.registry.specs()[0], parameter_schema={"type": "object"})


def test_public_catalog_command_does_not_tick_or_mutate_campaign(tmp_path, monkeypatch, capsys):
    from dataevolver.harness import cli
    c, _ = build_controller(tmp_path)
    before = c.status().to_dict()
    monkeypatch.setattr(cli, "_existing_controller", lambda *args: c)
    assert cli.main(["operators", "--campaign", "demo", "--campaign-root", str(tmp_path)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["frozen_contract_matches"] and value["campaign_id"] == "demo"
    assert not value["inference_performed"]
    assert c.status().to_dict() == before


def test_real_training_registry_declarations_bind_outer_protocol_without_inference(tmp_path):
    from dataevolver.testing.test_training_rsi_domain import fixture_domain
    from dataevolver.harness.adapters.training_rsi import TrainingRSITrialOperator, TrainingRSIObservationOperator
    domain, _, _ = fixture_domain(tmp_path)
    registry = OperatorRegistry()
    for cls in [TrainingRSITrialOperator, TrainingRSIObservationOperator]:
        operator = cls(domain)
        registry.register(operator)
    rows = registry.catalog()["operators"]
    assert {r["contract"]["category"] for r in rows} == {"basic", "evaluation"}
    assert all(r["declaration_status"] == "DECLARED" for r in rows)
    assert all(r["production_qualification"] == "NOT_ASSESSED" for r in rows)


def test_runtime_planner_discovers_only_authorized_contracts_and_no_shell_material(tmp_path):
    from dataevolver.testing.test_runtime_rsi import make_runtime, action
    c, backend = make_runtime(tmp_path, [action('ABSTAIN')])
    allowed = (c.goal.initial_operator_id,)
    c.planner.bind_operator_registry(c.registry, allowed)
    description = c.planner.operator_catalog()
    assert [x['operator_id'] for x in description] == list(allowed)
    assert not any(k in json.dumps(description) for k in ['argv', 'allowed_write_roots', 'executable_realpath'])
    assert c.planner.config()['registered_operator_contracts'] == description
    # The regular planner call records the same authorized view; discovery has
    # not added an executable action, a new operator or a qualification claim.
    c.planner.plan(c.status())
    trace = json.loads((c.planner.trace_root / 'p000' / 'trace.json').read_text())
    assert trace['context']['registered_operator_contracts'] == description
    assert all(x['production_qualification'] == 'NOT_ASSESSED' for x in description)
    assert len(backend.calls) == 1
