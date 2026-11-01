"""CPU fixtures for production evidence plumbing, never production qualification."""
import copy
import json
from dataclasses import replace

import pytest

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from dataevolver.harness.controller import EvolutionController
from dataevolver.harness.models import ControllerState
from dataevolver.harness.operator_development import DevelopmentBudget, USAGE_KEYS
from dataevolver.harness.operator_lifecycle_authority import FrozenOperatorLifecycleAuthority
from dataevolver.harness.operators import OperatorRegistry
from dataevolver.harness.production_qualification import (
    EVIDENCE_ROLES, RUNTIME_KEYS, ProductionQualificationExecution,
    ControllerProductionQualificationAuthority, assess_production_qualification,
    inspect_validator_bindings,
)
from dataevolver.testing.test_harness_controller import build_controller
from dataevolver.testing.test_operator_catalog import declaration
from dataevolver.testing.test_operator_development import fixture, write_ref, QualificationFixture


class FixtureLifecycle(FrozenOperatorLifecycleAuthority):
    """Explicit fixture authority; no historical/production result is claimed."""
    def __init__(self, summary):
        self.summary = summary

    def identity(self):
        return {"fixture": "host-bound-lineage"}

    def read(self):
        return copy.deepcopy(self.summary)


def setup(tmp_path):
    f = fixture(tmp_path)
    _, _, plan, package, store, design, validators = f
    matched = write_ref(store.roots["protocol"], "protocol", "matched.json", {"fixture": "original-matched-context"})
    design = replace(design, matched_context_sha256=matched["sha256"])
    template = tmp_path/'runtime-template'; template.mkdir()
    c, _ = build_controller(template)
    declarations = {s: declaration(spec, category="evaluation") for s, spec in validators.items()}
    stages = {s: {**rule, "validator_contract_sha256": declarations[s].compute_hash()}
              for s, rule in design.stages.items()}
    usage = {k: 0 for k in USAGE_KEYS}; usage.update(model_calls=4, operator_calls=12, wall_seconds=253)
    parent = FixtureLifecycle({"production_requirements": {"design_protocol_sha256": design.compute_hash(),
        "matched_context_sha256": design.matched_context_sha256}, "lifecycle_total": usage,
        "resource_deadline_epoch": 1792328957.7055008, "current_status": "REVOKED",
        "operator_id": "candidate.reusable", "version": "0"})
    execution = ProductionQualificationExecution("production-fixture-execution", payload_sha256(parent.identity()),
        payload_sha256(parent.read()), "candidate.reusable", "1", package.compute_hash(),
        design.to_dict(), matched, {**{k: payload_sha256({"runtime": k}) for k in RUNTIME_KEYS},
                                    'trust_namespace': 'registered_production_runtime'}, stages,
        DevelopmentBudget({k: 1000000 for k in USAGE_KEYS}, usage, 1792328957.7055008).to_dict())
    return f, execution, parent, declarations, c.evaluator.contract


class ProductionFixtureValidator(QualificationFixture):
    def __init__(self, spec, payload, store, *, omit_closure=False):
        super().__init__(spec, payload)
        self.store, self.omit_closure = store, omit_closure

    def verify_effect(self, call, receipt):
        result = super().verify_effect(call, receipt)
        hashes = dict(result.artifact_hashes)
        if not self.omit_closure:
            for ref in self.payload["fixture_verified_refs"]:
                path = self.store.resolve(ArtifactRef.from_dict(ref))
                hashes[str(path)] = file_sha256(path)
        return replace(result, artifact_hashes=hashes)


def stage_controller(tmp_path, f, execution, parent, declarations, stage, *, abi_change=None,
                     omit_closure=False, bind_goal=True, usage=None, extra_refs=()):
    _, _, _, package, store, _, validators = f
    protocol = execution.execution_protocol()
    root = tmp_path/('production-' + stage); root.mkdir()
    template, _ = build_controller(root)
    sources = [s["artifact"] for s in protocol.sources if s["family_id"] in protocol.families(stage)]
    artifacts = {role: write_ref(store.roots["protocol"], "protocol", stage + '/' + role + '.json',
                                  {"role": role, "stage": stage}) for role in EVIDENCE_ROLES}
    abi = {"schema_version": "dataevolver.registered_production_evidence.v1",
           "execution_contract_sha256": execution.compute_hash(), "stage": stage,
           "runtime": execution.runtime, "source_artifacts": sources, "artifacts": artifacts}
    if abi_change:
        abi_change(abi)
    binding = {"plan_sha256": execution.compute_hash(), "package_sha256": package.compute_hash(),
               "protocol_sha256": protocol.compute_hash(), "stage": stage,
               "matched_context_sha256": protocol.matched_context_sha256}
    payload = {"qualification_binding": binding, "source_family_ids": protocol.families(stage),
               "check_results": {key: True for key in protocol.stages[stage]["check_ids"]},
               "lifecycle_usage": usage or {k: 1 for k in USAGE_KEYS}, "production_evidence": abi,
               "fixture_verified_refs": sources + list(artifacts.values()) + [execution.matched_context] + list(extra_refs)}
    registry = OperatorRegistry()
    op = ProductionFixtureValidator(validators[stage], payload, store, omit_closure=omit_closure)
    registry.register(op, contract=declarations[stage])
    objective = {**template.goal.objective, "operator_qualification": binding}
    if bind_goal:
        objective['production_qualification_execution'] = execution.compute_hash()
    goal = template.goal.with_updates(campaign_id='production-' + stage, initial_operator_id=op.spec.operator_id,
        resource_policy={"allow_gpu": False, "allowed_operators": [op.spec.operator_id]},
        objective=objective, interventions=())
    c = EvolutionController.create(repo_root=root, campaigns_root=root/'campaigns', goal=goal,
        registry=registry, planner=template.planner, evaluator_contract=template.evaluator.contract,
        idempotency_key='production-' + stage)
    assert c.run().state is ControllerState.COMPLETE
    event = next(e for e in c.store.events.read() if e.event_type == 'ACTION_COMMITTED')
    def forbidden(*a, **kw):
        raise AssertionError('evidence reading must not execute or recover')
    op.execute = op.recover = forbidden
    authority = ControllerProductionQualificationAuthority(c, event.payload['call_id'],
        execution=execution, package=package, store=store, predecessor_authority=parent)
    return authority


def test_new_execution_contract_preserves_design_and_requires_host_bindings(tmp_path):
    f, execution, parent, declarations, evaluator = setup(tmp_path)
    assert ProductionQualificationExecution.from_dict(execution.to_dict()) == execution
    registry = OperatorRegistry()
    for s, spec in f[-1].items():
        op = ProductionFixtureValidator(spec, {}, f[4])
        registry.register(op, contract=declarations[s])
    before = f[4].resolve(ArtifactRef.from_dict(f[2].qualification_protocol)).read_bytes()
    result = execution.inspect(authority=parent, registry=registry,
        evaluators={s: evaluator for s in execution.stages}, store=f[4], now=1792328956)
    assert result['status'] == 'BOUND'
    assert not result['production_execution_allowed'] and not result['new_execution_or_budget_authorization']
    assert f[4].resolve(ArtifactRef.from_dict(f[2].qualification_protocol)).read_bytes() == before
    expired = execution.inspect(authority=parent, registry=registry,
        evaluators={s: evaluator for s in execution.stages}, store=f[4], now=1792328958)
    assert expired['gap_codes'] == ['RESOURCE_DEADLINE_PASSED']


@pytest.mark.parametrize('change', ['placeholder', 'removed_check', 'matched', 'runtime', 'unknown', 'synthetic_trust'])
def test_design_or_incomplete_runtime_cannot_be_relabelled_as_execution(tmp_path, change):
    _, execution, _, _, _ = setup(tmp_path)
    value = execution.to_dict()
    if change == 'placeholder': value['stages']['engineering']['validator_operator_id'] = 'qualification.pending_engineering'
    elif change == 'removed_check': value['stages']['independent']['check_ids'].pop()
    elif change == 'matched': value['matched_context']['sha256'] = 'f'*64
    elif change == 'runtime': value['runtime'].pop('k_snapshot_sha256')
    elif change == 'synthetic_trust': value['runtime']['trust_namespace'] = 'synthetic_authority_contract_independent'
    else: value['production_adoption_allowed'] = True
    with pytest.raises(ValueError): ProductionQualificationExecution.from_dict(value)


@pytest.mark.parametrize('change', ['revoked_version', 'renamed', 'deadline', 'prior_cost', 'summary', 'raw_json'])
def test_original_revoke_deadline_and_failed_costs_cannot_be_reset(tmp_path, change):
    _, execution, parent, _, _ = setup(tmp_path)
    value = execution.to_dict()
    if change == 'revoked_version': value['operator_version'] = '0'
    elif change == 'renamed': value['operator_id'] = 'candidate.different-name'
    elif change == 'deadline': value['budget']['resource_deadline_epoch'] += 1
    elif change == 'prior_cost': value['budget']['prior_usage']['model_calls'] = 0
    elif change == 'summary': value['predecessor_summary_sha256'] = 'f'*64
    else: parent = parent.read()
    with pytest.raises(ValueError): ProductionQualificationExecution.from_dict(value).verify_predecessor(parent)


def test_missing_or_changed_real_registry_and_evaluator_fail_without_verification(tmp_path):
    f, execution, _, declarations, evaluator = setup(tmp_path)
    registry = OperatorRegistry()
    first = inspect_validator_bindings(execution.stages, registry, {})
    assert all('VALIDATOR_NOT_REGISTERED' in v['gap_codes'] for v in first['stages'].values())
    for spec in f[-1].values(): registry.register(ProductionFixtureValidator(spec, {}, f[4]))
    missing = inspect_validator_bindings(execution.stages, registry, {s: evaluator for s in execution.stages})
    assert all('VALIDATOR_EVALUATION_CONTRACT_MISSING' in v['gap_codes'] for v in missing['stages'].values())
    value = json.loads(evaluator.path.read_text()); value['revision'] = 'changed'
    evaluator.path.write_text(json.dumps(value))
    assert 'EVALUATOR_MISMATCH' in inspect_validator_bindings(execution.stages, registry,
        {s: evaluator for s in execution.stages})['stages']['engineering']['gap_codes']


def test_two_actual_committed_controller_stages_verify_readonly_and_aggregate_costs(tmp_path):
    f, execution, parent, declarations, _ = setup(tmp_path)
    authorities = {s: stage_controller(tmp_path, f, execution, parent, declarations, s) for s in execution.stages}
    before = {s: a.base.controller.layout.events_path.read_bytes() for s, a in authorities.items()}
    result = assess_production_qualification(execution, authorities)
    assert result['production_engineering_qualification'] == result['production_independent_qualification'] == 'VERIFIED'
    assert not result['production_adoption_allowed'] and not result['method_improvement_verified']
    assert result['lifecycle_total']['model_calls'] == 6  # Four prior + two fixture reported, no real model calls.
    assert before == {s: a.base.controller.layout.events_path.read_bytes() for s, a in authorities.items()}


@pytest.mark.parametrize('change', ['runtime', 'heldout', 'role', 'extra', 'unbound_hashes', 'unbound_goal'])
def test_copied_checks_do_not_waive_runtime_sources_or_committed_closure(tmp_path, change):
    f, execution, parent, declarations, _ = setup(tmp_path)
    def mutate(abi):
        if change == 'runtime': abi['runtime'] = {**abi['runtime'], 'r_state_sha256': 'f'*64}
        elif change == 'heldout': abi['source_artifacts'] = [execution.design_protocol['sources'][-1]['artifact']]
        elif change == 'role': abi['artifacts'].pop('native_audit')
        elif change == 'extra': abi['passed'] = True
    authority = stage_controller(tmp_path, f, execution, parent, declarations, 'engineering',
        abi_change=mutate, omit_closure=change == 'unbound_hashes', bind_goal=change != 'unbound_goal')
    with pytest.raises(ValueError): authority.verify('engineering')


def test_combined_costs_cannot_fit_each_stage_by_resetting_between_them(tmp_path):
    f, execution, parent, declarations, _ = setup(tmp_path)
    value = execution.to_dict(); value['budget']['limits']['model_calls'] = 5
    execution = ProductionQualificationExecution.from_dict(value)
    authorities = {s: stage_controller(tmp_path, f, execution, parent, declarations, s) for s in execution.stages}
    for s, a in authorities.items(): assert a.verify(s)['usage']['model_calls'] == 1
    with pytest.raises(ValueError, match='budget exceeded'):
        assess_production_qualification(execution, authorities)


def test_changed_candidate_bytes_cannot_keep_qualified_metadata(tmp_path):
    f, execution, parent, declarations, _ = setup(tmp_path)
    authority = stage_controller(tmp_path, f, execution, parent, declarations, 'engineering')
    f[4].resolve(ArtifactRef.from_dict(f[3].implementation)).write_text('changed candidate bytes')
    with pytest.raises(ValueError, match='hash changed'):
        authority.verify('engineering')
    assert not (tmp_path/'EXECUTED').exists()
