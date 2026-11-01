"""Registered check recovery/budget invariants with actual confined children."""
import json
import os
import platform
from dataclasses import replace

import pytest

from dataevolver.dataset.contracts import payload_sha256
from dataevolver.harness.adapters.operator_candidate_checks import CandidateChecksOperator
from dataevolver.harness.candidate_sandbox import LinuxPureDataRunner, SandboxLimits
from dataevolver.harness.operator_development import CandidatePackage, ControllerRequestAuthority, DevelopmentBudget
from dataevolver.harness.models import EvidenceStatus, OperatorCall, RecoveryStatus
from dataevolver.testing.test_operator_candidate_build import make_builder
from dataevolver.testing.test_operator_development import write_ref

pytestmark = pytest.mark.skipif(platform.system() != 'Linux' or os.geteuid() != 0,
                                reason='explicit Linux root bootstrap required')


def setup(tmp_path, source=None):
    source = source or 'def resolve(p):\n return {"result":p["value"]} if type(p.get("value")) is int else {"status":"unsupported"}\n'
    output = {'files': [{'path':'operator.py','content':source}], 'entrypoint':'operator.py',
              'dependencies':[], 'change_description':'Generic engineering fixture.'}
    builder, build_call, origin, _ = make_builder(tmp_path, output=output)
    builder.execute(build_call)
    package = CandidatePackage.from_dict(json.loads((tmp_path/'builder-output/call-1/candidate-package.json').read_text()))
    cases = []
    for kind, value, expected in [('positive',3,{'result':3}),('negative',None,{'status':'unsupported'})]:
        ref = write_ref(builder.store.roots['protocol'],'protocol','behavior-'+kind+'.json',{'value':value})
        cases.append({'case_id':kind,'kind':kind,'input':ref,'expected_subset':expected})
    runner = LinuxPureDataRunner(limits=SandboxLimits(output_bytes=512))
    prior = dict(builder.plan.budget['prior_usage']);prior.update(model_calls=1,operator_calls=1)
    budget = DevelopmentBudget({**builder.plan.budget['limits'], 'model_calls':1,'planner_calls':0,
                               'image_experiments':0}, prior, builder.plan.budget['resource_deadline_epoch'])
    protocol = {'schema_version':'dataevolver.candidate_behavior_checks.v1',
        'plan_sha256':builder.plan.compute_hash(),'package_sha256':package.compute_hash(),
        'runtime_sha256':payload_sha256(runner.identity()),'cases':cases,'budget':budget.to_dict()}
    operator = CandidateChecksOperator(plan=builder.plan,package=package,store=builder.store,
        request_authority=ControllerRequestAuthority(origin),runner=runner,check_protocol=protocol,
        output_root=tmp_path/'checks')
    call = OperatorCall('candidate-checks-fixture','checks-1','checks-1',operator.spec.operator_id,'1',0,
                        str(tmp_path/'checks/call-1'))
    return operator,call


def test_real_registered_checks_recover_without_executing_again(tmp_path,monkeypatch):
    op,call=setup(tmp_path);receipt=op.execute(call)
    assert op.verify_effect(call,receipt).status is EvidenceStatus.VERIFIED
    report=json.loads(op._path(call).read_text())
    assert report['all_checks_passed'] and report['positive_behavior_verified']
    assert report['engineering_qualification']==report['independent_qualification']=='NOT_ASSESSED'
    assert not report['promoted'] and report['lifecycle_usage']['model_calls']==0
    monkeypatch.setattr(op.runner,'run',lambda *a:pytest.fail('completed source was executed again'))
    assert op.execute(call).reused
    op._path(call).unlink()
    assert op.recover(call).status is RecoveryStatus.AMBIGUOUS  # receipt without effect


def test_negative_only_behavior_does_not_pass_positive_capability(tmp_path):
    op,call=setup(tmp_path,'def resolve(p):return {"status":"unsupported"}\n')
    receipt=op.execute(call)
    assert op.verify_effect(call,receipt).status is EvidenceStatus.VERIFIED
    assert receipt.stage_result['metrics']=={'behavior_checks_passed':0.0,'positive_behavior_verified':0.0}
    assert json.loads(op._path(call).read_text())['cases'][1]['passed']


def test_unknown_child_effect_is_not_retried(tmp_path,monkeypatch):
    op,call=setup(tmp_path)
    monkeypatch.setattr(op.runner,'run',lambda *a:(_ for _ in ()).throw(RuntimeError('interrupted')))
    with pytest.raises(RuntimeError):op.execute(call)
    monkeypatch.setattr(op.runner,'run',lambda *a:pytest.fail('unknown effect resent'))
    assert op.recover(call).status is RecoveryStatus.AMBIGUOUS
    with pytest.raises(ValueError,match='no resend'):op.execute(call)


def test_completed_outputs_without_receipt_recover_without_run(tmp_path,monkeypatch):
    from dataevolver.harness.operators import _receipt_path
    op,call=setup(tmp_path);op.execute(call)
    op._path(call).unlink();_receipt_path(call).unlink()
    monkeypatch.setattr(op.runner,'run',lambda *a:pytest.fail('completed output rerun'))
    assert op.recover(call).status is RecoveryStatus.RECOVERED


def test_case_and_stream_tampering_rejected(tmp_path):
    op,call=setup(tmp_path);receipt=op.execute(call)
    record=op._path(call).parent/'executions/0.json'
    value=json.loads(record.read_text());value['output']={'result':99};record.write_text(json.dumps(value))
    assert op.verify_effect(call,receipt).status is EvidenceStatus.CONTRADICTED


def test_new_call_identity_cannot_reset_cpu_budget(tmp_path):
    op,call=setup(tmp_path);op.execute(call)
    other=replace(call,call_id='checks-2',idempotency_key='checks-2',work_dir=str(tmp_path/'checks/call-2'))
    with pytest.raises(ValueError,match='reset execution allowance'):op.execute(other)


def test_new_model_authority_and_deadline_changes_rejected(tmp_path):
    op,_=setup(tmp_path)
    op.protocol['budget']['limits']['model_calls']+=1
    with pytest.raises(ValueError,match='no model authority'):op._check_protocol()
    op.protocol['budget']['limits']['model_calls']-=1
    op.protocol['budget']['resource_deadline_epoch']+=1
    with pytest.raises(ValueError,match='deadline'):op._check_protocol()


def test_output_capture_reserved_before_execution(tmp_path):
    op,_=setup(tmp_path)
    op.protocol['budget']['limits']['output_bytes']=1
    with pytest.raises(ValueError,match='budget exceeded'):op._check_protocol()
