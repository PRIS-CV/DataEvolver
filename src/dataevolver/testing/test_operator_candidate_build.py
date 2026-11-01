"""ModelGateway fixture tests; generated source stays inert and unqualified."""
import json
from dataclasses import replace

import pytest

from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile
from dataevolver.adapters.model_gateway import ModelGateway, ModelError, MockModelProvider
from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from dataevolver.harness.adapters.operator_candidate_build import CandidateBuildOperator
from dataevolver.harness.operator_development import (CandidatePackage, ControllerRequestAuthority,
    DevelopmentBudget, OperatorDevelopmentPlanV2, OperatorInterface, inspect_candidate, read_development_plan)
from dataevolver.harness.models import EvidenceStatus, OperatorCall, RecoveryStatus, StageStatus
from dataevolver.testing.test_operator_development import fixture, write_ref


def make_builder(tmp_path, *, output=None, output_bytes=100000, provider=None):
    c, _, old, old_package, store, _, _ = fixture(tmp_path)
    candidate = store.roots['candidate']
    for path in candidate.iterdir():
        path.unlink()  # Fixture setup; this is not performed by the builder.
    declaration = {
        'schema_version': 'dataevolver.operator_contract.v1', 'operator_id': 'candidate.fixture', 'version': '1',
        'category': 'basic', 'input_schemas': {'input': {'type': 'object'}},
        'output_schemas': {'output': {'type': 'object'}},
        'parameter_schema': {'type': 'object', 'additionalProperties': False, 'properties': {}},
        'modalities': ['json'], 'effect_semantics': 'pure_data',
        'recovery_semantics': 'verify_then_retry', 'limitations': ['Fixture candidate only.']}
    interface = OperatorInterface(declaration, True, 30)
    target = write_ref(store.roots['protocol'], 'protocol', 'target-interface.json', interface.to_dict())
    budget = DevelopmentBudget({**old.budget['limits'], 'model_calls': 1, 'output_bytes': output_bytes},
                               old.budget['prior_usage'], old.budget['resource_deadline_epoch'])
    plan = OperatorDevelopmentPlanV2(old.plan_id, old.capability_request, old.request_campaign_id,
        old.request_frozen_sha256, old.qualification_protocol, target, old.candidate_store,
        old.protected_stores, budget.to_dict())
    output = output or {'files': [{'path': 'operator.py', 'content':
        'raise RuntimeError("must remain inert")\ndef resolve(payload):\n    return {"status": "unsupported"}\n'}],
        'entrypoint': 'operator.py', 'dependencies': [], 'change_description': 'A fixture candidate, not qualified.'}
    profile = ModelProfile('builder-fixture', 'mock', 'fixture-only', '1', mock_output=output)
    root = tmp_path/'builder-output'
    gateway = ModelGateway(profiles=(profile,), policy=ModelPolicy({'agent.runtime_rsi': profile.profile_id}, 1),
        receipt_root=root/'model-receipts', providers={profile.profile_id: provider} if provider else None)
    builder = CandidateBuildOperator(plan=plan, store=store, request_authority=ControllerRequestAuthority(c),
        gateway=gateway, role='agent.runtime_rsi', output_root=root,
        native_roots=(gateway.root/'effects'/profile.profile_id,), max_tokens=2048, timeout_seconds=60)
    call = OperatorCall('build-fixture', 'build-1', 'build-1', builder.spec.operator_id, '1', 0, str(root/'call-1'))
    return builder, call, c, profile


def test_build_compiles_exact_interface_after_generation_without_executing_source(tmp_path):
    builder, call, c, _ = make_builder(tmp_path)
    request = builder._request(call)
    text = json.dumps(request.to_dict())
    assert not request.artifacts and 'independent-a' not in text and 'hold' not in text
    receipt = builder.execute(call)
    assert receipt.status is StageStatus.SUCCESS
    assert builder.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    package = CandidatePackage.from_dict(json.loads((tmp_path/'builder-output/call-1/candidate-package.json').read_text()))
    assert package.implementation['media_type'] == 'text/x-python'
    result = inspect_candidate(builder.plan, package, builder.store, builder.authority)
    assert result['engineering_qualification'] == result['independent_qualification'] == 'NOT_ASSESSED'
    assert not result['candidate_code_executed'] and not result['promoted']
    assert read_development_plan(builder.plan.to_dict()) == builder.plan
    assert builder.gateway.audit_usage()['model_calls'] == 1
    assert builder.recover(call).receipt.reused
    assert builder.gateway.audit_usage()['model_calls'] == 1


def test_completed_native_effect_recovers_without_new_call_or_new_budget(tmp_path, monkeypatch):
    builder, call, _, _ = make_builder(tmp_path)
    finish = builder._finish
    monkeypatch.setattr(builder, '_finish', lambda *args: (_ for _ in ()).throw(RuntimeError('process interrupted')))
    with pytest.raises(RuntimeError, match='interrupted'):
        builder.execute(call)
    assert builder.gateway.audit_usage()['model_calls'] == 1
    monkeypatch.setattr(builder, '_finish', finish)
    recovered = builder.recover(call)
    assert recovered.status is RecoveryStatus.RECOVERED
    assert builder.gateway.audit_usage()['model_calls'] == 1
    assert builder.verify_effect(call, recovered.receipt).status is EvidenceStatus.VERIFIED


@pytest.mark.parametrize('path', ['../escape.py', '/absolute.py', 'operator-spec.json', 'install.sh'])
def test_invalid_candidate_output_is_terminal_and_cost_is_retained(tmp_path, path):
    output = {'files': [{'path': path, 'content': 'pass'}], 'entrypoint': path,
              'dependencies': [], 'change_description': 'Invalid fixture.'}
    builder, call, _, _ = make_builder(tmp_path, output=output)
    receipt = builder.execute(call)
    assert receipt.status is StageStatus.FAILED and receipt.stage_result['failure_code'] == 'INVALID_CANDIDATE'
    assert builder.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    assert builder.execute(call).status is StageStatus.FAILED
    assert builder.gateway.audit_usage()['model_calls'] == 1
    assert not list(builder.store.roots['candidate'].iterdir())


def test_output_quota_is_checked_before_any_candidate_file_is_written(tmp_path):
    builder, call, _, _ = make_builder(tmp_path, output_bytes=1)
    receipt = builder.execute(call)
    assert receipt.status is StageStatus.FAILED
    assert not list(builder.store.roots['candidate'].iterdir())
    assert builder.gateway.audit_usage()['model_calls'] == 1


def test_new_call_id_cannot_reset_single_builder_budget_or_change_receipts(tmp_path):
    builder, call, _, _ = make_builder(tmp_path)
    builder.execute(call)
    other = replace(call, call_id='build-2', idempotency_key='build-2', work_dir=str(tmp_path/'builder-output/call-2'))
    with pytest.raises(ValueError, match='single builder call'):
        builder.execute(other)
    assert builder.gateway.audit_usage()['model_calls'] == 1


def test_native_or_candidate_drift_and_forged_stage_result_prevent_recovery(tmp_path):
    builder, call, _, _ = make_builder(tmp_path)
    receipt = builder.execute(call)
    forged = receipt.with_updates(stage_result={**receipt.stage_result, 'metrics': {'quality': 1.0}})
    assert builder.verify_effect(call, forged).status is EvidenceStatus.CONTRADICTED
    (builder.store.roots['candidate']/'operator.py').write_text('changed')
    assert builder.recover(call).status is RecoveryStatus.AMBIGUOUS
    assert builder.gateway.audit_usage()['model_calls'] == 1


def test_inert_interface_cannot_prebind_implementation_or_grant_execution_authority(tmp_path):
    builder, _, _, _ = make_builder(tmp_path)
    interface = builder.interface
    with pytest.raises(ValueError, match='unknown implementation'):
        replace(interface, declaration={**interface.declaration, 'spec_sha256': 'a'*64})
    from dataevolver.harness.models import OperatorSpec
    d = interface.declaration
    spec = OperatorSpec(d['operator_id'], d['version'], retry_safe=True, timeout_seconds=30)
    with pytest.raises(ValueError, match='execution authority'):
        interface.bind(replace(spec, argv=('python', 'candidate.py')))


def test_unknown_provider_effect_is_never_resent(tmp_path):
    class UnknownProvider:
        def invoke(self, request, request_hash):
            raise ModelError('AMBIGUOUS_EFFECT', 'fixture lost response')

        def lookup(self, request, request_hash):
            return None

    builder, call, _, _ = make_builder(tmp_path, provider=lambda: UnknownProvider())
    with pytest.raises(ModelError):
        builder.execute(call)
    assert builder.recover(call).status is RecoveryStatus.AMBIGUOUS
    with pytest.raises(ValueError, match='no recoverable completion'):
        builder.execute(call)
    assert builder.gateway.audit_usage()['model_calls'] == 1


def test_registered_builder_runs_through_existing_controller_without_promoting_code(tmp_path):
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.evaluator import EvaluatorContract
    from dataevolver.harness.models import ControllerState
    from dataevolver.harness.operators import OperatorRegistry
    from dataevolver.harness.state_store import atomic_write_json
    from dataevolver.testing.test_harness_controller import build_controller
    builder, _, _, _ = make_builder(tmp_path)
    template, _ = build_controller(tmp_path/'template')
    registry = OperatorRegistry()
    registry.register(builder)
    evaluator = tmp_path/'builder-output/evaluator.json'
    atomic_write_json(evaluator, {'schema_version': 'dataevolver.harness.evaluator.v1',
        'evaluator_id': 'candidate-generation-only', 'revision': '1',
        'hard_constraints': {'artifact_valid': {'required': True}},
        'target_metrics': {'candidate_built': {'required': True, 'minimum': 1}}, 'protected_metrics': {}})
    goal = template.goal.with_updates(campaign_id='registered-builder', initial_operator_id=builder.spec.operator_id,
        resource_policy={'allow_gpu': False, 'allowed_operators': [builder.spec.operator_id]},
        frozen_evaluator={'path': str(evaluator), 'sha256': file_sha256(evaluator)}, interventions=())
    c = EvolutionController.create(repo_root=tmp_path/'template', campaigns_root=tmp_path/'builder-output/campaigns',
        goal=goal, registry=registry, planner=template.planner,
        evaluator_contract=EvaluatorContract.from_path(evaluator), idempotency_key='registered-builder')
    end = c.run()
    assert end.state is ControllerState.COMPLETE and end.operator_calls == 1 and end.experiments_used == 0
    assert builder.gateway.audit_usage()['model_calls'] == 1
    assert c.replay().to_dict() == end.to_dict() and not c.capabilities.list_capabilities()
    events = c.layout.events_path.read_bytes()
    c.run()
    assert events == c.layout.events_path.read_bytes() and builder.gateway.audit_usage()['model_calls'] == 1
    from dataevolver.dataset.contracts import ArtifactStore
    from dataevolver.harness.adapters.operator_candidate_recovery import ControllerBuildOutputAuthority,CandidateBuildRecoveryOperator
    import math
    authority=ControllerBuildOutputAuthority(c,builder,next(e.payload['call_id'] for e in c.store.events.read() if e.event_type=='ACTION_COMMITTED'))
    response,proof=authority.read()
    old=builder.budget
    prior={**old.prior_usage,'model_calls':1,'operator_calls':1,'wall_seconds':math.ceil(proof['usage']['observed_wall_seconds'])}
    budget=DevelopmentBudget({**old.limits,'operator_calls':2,'model_calls':1,'planner_calls':0,'image_experiments':0},prior,old.resource_deadline_epoch)
    plan=replace(builder.plan,plan_id='cpu-recovery',budget=budget.to_dict())
    store=ArtifactStore({**builder.store.roots,'candidate':tmp_path/'recovered-candidate'})
    recovery=CandidateBuildRecoveryOperator(plan=plan,store=store,request_authority=builder.authority,
        build_authority=authority,output_root=tmp_path/'recovery-output')
    from dataevolver.harness.models import OperatorCall
    call=OperatorCall('cpu-recovery','recover','recover',recovery.spec.operator_id,'1',0,str(tmp_path/'recovery-output/call'))
    result=recovery.execute(call)
    assert recovery.verify_effect(call,result).status is EvidenceStatus.VERIFIED
    assert builder.gateway.audit_usage()['model_calls']==1
    assert (store.roots['candidate']/'operator.py').read_bytes()==(builder.store.roots['candidate']/'operator.py').read_bytes()
    (store.roots['candidate']/'operator.py').unlink()
    assert recovery.verify_effect(call,result).status is EvidenceStatus.CONTRADICTED
    assert not (store.roots['candidate']/'operator.py').exists()


def test_callable_entrypoint_notation_is_losslessly_normalized_without_source_changes(tmp_path):
    output={'files':[{'path':'operator.py','content':'def resolve(payload):\n    return payload\n'}],
            'entrypoint':'operator.py:resolve','dependencies':[],'change_description':'Interface notation fixture.'}
    builder,call,_,_=make_builder(tmp_path,output=output)
    receipt=builder.execute(call)
    assert receipt.status is StageStatus.SUCCESS
    package=CandidatePackage.from_dict(json.loads((tmp_path/'builder-output/call-1/candidate-package.json').read_text()))
    assert package.implementation['path']=='operator.py'
    assert (builder.store.roots['candidate']/'operator.py').read_text()==output['files'][0]['content']
    assert builder.gateway.audit_usage()['model_calls']==1
