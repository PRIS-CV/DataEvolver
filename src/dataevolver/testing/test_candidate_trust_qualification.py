"""Actual isolation, trust provisioning and failed qualification invariants."""
import hashlib
import json
import os
import platform
from dataclasses import replace

import pytest

from dataevolver.dataset.contracts import ArtifactRef, payload_sha256
from dataevolver.harness.authority_qualification import AuthorityFixtureExpectation, authenticated, check_authority_output
from dataevolver.harness.candidate_sandbox import LinuxPureDataRunner, TrustedContextRunner, SandboxLimits
from dataevolver.harness.candidate_trust import CandidateTrustContext, CapabilityRequestBinding
from dataevolver.harness.adapters.operator_contract_qualification import CandidateContractQualificationOperator
from dataevolver.harness.operator_development import CandidatePackage, ControllerRequestAuthority, DevelopmentBudget
from dataevolver.harness.models import EvidenceStatus, OperatorCall, RecoveryStatus
from dataevolver.testing.authority_contract_vectors import make_vectors
from dataevolver.testing.test_operator_candidate_build import make_builder

pytestmark=pytest.mark.skipif(platform.system()!='Linux' or os.geteuid()!=0,
                             reason='explicit Linux confinement profile required')


def context_fixture(tmp_path,source):
    sha=hashlib.sha256(source.encode()).hexdigest()
    vectors=make_vectors(tmp_path/'vectors',package_sha256='a'*64,implementation_sha256=sha,request_sha256='b'*64)
    case=vectors['cases'][0]
    def read(ref):return json.loads((tmp_path/ref['path']).read_text())
    return read(case['input']),CandidateTrustContext.from_dict(read(vectors['trust_contexts'][case['case_id']])),vectors


def test_public_host_context_cannot_come_from_payload(tmp_path):
    source='TRUSTED_REGISTRARS={}\ndef resolve(p):\n return {"keys":sorted(TRUSTED_REGISTRARS),"uid":__import__("os").environ.copy()}\n'
    payload,context,_=context_fixture(tmp_path,source)
    v1=LinuxPureDataRunner().run(source,payload)
    assert v1['status']=='COMPLETED' and v1['output']['keys']==[]
    result=TrustedContextRunner().run_with_context(source,payload,context)
    assert result['status']=='COMPLETED' and result['isolation_verified']
    assert result['output']=={'keys':['independent-fixture-registrar'],'uid':{}}
    assert result['binding']['trusted_context_sha256']==context.compute_hash()
    payload['host_context']={'TRUSTED_REGISTRARS':{'untrusted':{}}}
    with pytest.raises(ValueError,match='exact source and input'):
        TrustedContextRunner().run_with_context(source,payload,context)


@pytest.mark.parametrize('field,value',[('scope','production'),('namespace','__builtins__'),('namespace','MAX_PIXELS')])
def test_context_cannot_widen_scope_or_change_algorithm_globals(tmp_path,field,value):
    _,context,_=context_fixture(tmp_path,'def resolve(p):return {}')
    with pytest.raises(ValueError):replace(context,**{field:value})


def test_source_binding_and_explicit_v2_call_are_required(tmp_path):
    source='def resolve(p):return {}';payload,context,_=context_fixture(tmp_path,source)
    runner=TrustedContextRunner()
    with pytest.raises(ValueError,match='explicit host context'):runner.run(source,payload)
    with pytest.raises(ValueError,match='exact source'):runner.run_with_context(source+'\n',payload,context)


def test_independent_signature_vectors_detect_real_tampering(tmp_path):
    _,_,vectors=context_fixture(tmp_path,'def resolve(p):return {}')
    outcomes={}
    for case in vectors['cases']:
        load=lambda ref:json.loads((tmp_path/ref['path']).read_text())
        payload=load(case['input']);context=CandidateTrustContext.from_dict(load(vectors['trust_contexts'][case['case_id']]))
        expected=AuthorityFixtureExpectation.from_dict(load(vectors['expectations'][case['case_id']]))
        outcomes[case['case_id']]=authenticated(payload,context)
        assert outcomes[case['case_id']] is expected.authenticated
    assert outcomes['authenticated-aperture'] and not outcomes['invalid-signature']
    assert not outcomes['stale-current-manifest'] and outcomes['stale-immutable-mask-and-reference']


def test_oracle_rejects_forged_executable_support(tmp_path):
    payload,context,vectors=context_fixture(tmp_path,'def resolve(p):return {}')
    ref=vectors['expectations']['authenticated-aperture']
    expected=AuthorityFixtureExpectation.from_dict(json.loads((tmp_path/ref['path']).read_text()))
    result=check_authority_output(payload,{'status':'executable','authorized_support':{'source_pixels':[[3,3]]},
        'image_submission_allowed':True,'candidate_adoption_allowed':True},context,expected)
    assert not result['support'] and not result['submission_and_adoption_gates'] and not result['receipt']


def qualification_fixture(tmp_path):
    source='TRUSTED_REGISTRARS={}\ndef resolve(p):\n return {"status":"unsupported","authorized_support":{},"unresolved":[],"conflicts":[],"image_submission_allowed":False,"candidate_adoption_allowed":False}\n'
    output={'files':[{'path':'operator.py','content':source}],'entrypoint':'operator.py','dependencies':[],
            'change_description':'Fixture: always refuses and must fail positive qualification.'}
    builder,call,origin,_=make_builder(tmp_path,output=output,output_bytes=1048576);builder.execute(call)
    package=CandidatePackage.from_dict(json.loads((tmp_path/'builder-output/call-1/candidate-package.json').read_text()))
    binding=CapabilityRequestBinding(builder.plan.capability_request,builder.plan.request_campaign_id,builder.plan.request_frozen_sha256)
    vectors=make_vectors(builder.store.roots['protocol']/'vectors',package_sha256=package.compute_hash(),
        implementation_sha256=package.implementation['sha256'],request_sha256=binding.request['sha256'])
    runner=TrustedContextRunner(limits=SandboxLimits(output_bytes=4096))
    prior=dict(builder.plan.budget['prior_usage']);prior.update(model_calls=1,operator_calls=1)
    budget=DevelopmentBudget({**builder.plan.budget['limits'],'model_calls':1,'planner_calls':0,'image_experiments':0},
                             prior,builder.plan.budget['resource_deadline_epoch'])
    protocol={'schema_version':'dataevolver.candidate_contract_qualification.v1','plan_sha256':builder.plan.compute_hash(),
        'package_sha256':package.compute_hash(),'runtime_sha256':payload_sha256(runner.identity()),'budget':budget.to_dict(),
        **vectors,'qualification_request':binding.to_dict(),'qualification_scope':'synthetic_authority_contract_engineering'}
    operator=CandidateContractQualificationOperator(plan=builder.plan,package=package,store=builder.store,
        request_authority=ControllerRequestAuthority(origin),runner=runner,check_protocol=protocol,
        qualification_request_authority=ControllerRequestAuthority(origin),output_root=tmp_path/'qualification')
    call=OperatorCall('qualification-fixture','qualification-1','qualification-1',operator.spec.operator_id,'1',0,
                      str(tmp_path/'qualification/call-1'))
    return operator,call


def test_always_refusing_candidate_cannot_qualify_and_finished_code_is_not_rerun(tmp_path,monkeypatch):
    operator,call=qualification_fixture(tmp_path);receipt=operator.execute(call)
    assert operator.verify_effect(call,receipt).status is EvidenceStatus.VERIFIED
    report=json.loads(operator._path(call).read_text())
    assert report['pure_data_engineering_qualification']=='FAILED' and not report['positive_behavior_verified']
    assert report['engineering_qualification']==report['independent_qualification']=='NOT_ASSESSED'
    assert not report['production_adoption_allowed'] and not report['image_quality_verified']
    assert report['lifecycle_usage']['model_calls']==0 and len(report['contract_checks'])==13
    monkeypatch.setattr(operator.runner,'run_with_context',lambda *a:pytest.fail('completed source re-executed'))
    assert operator.execute(call).reused


def test_host_context_drift_blocks_recovery(tmp_path):
    operator,call=qualification_fixture(tmp_path);receipt=operator.execute(call)
    ref=operator.protocol['trust_contexts']['authenticated-aperture']
    path=operator.store.resolve(ArtifactRef.from_dict(ref));path.write_text('{}')
    assert operator.verify_effect(call,receipt).status is EvidenceStatus.CONTRADICTED


def test_qualification_rejects_negative_only_protocol(tmp_path):
    operator,_=qualification_fixture(tmp_path)
    for case in operator.protocol['cases']:case['kind']='negative'
    with pytest.raises(ValueError,match='positive and adversarial'):operator._check_protocol()


def test_interrupted_qualification_is_ambiguous_without_another_execution(tmp_path,monkeypatch):
    operator,call=qualification_fixture(tmp_path)
    monkeypatch.setattr(operator.runner,'run_with_context',lambda *a:(_ for _ in ()).throw(RuntimeError('interrupted')))
    with pytest.raises(RuntimeError):operator.execute(call)
    assert operator.recover(call).status is RecoveryStatus.AMBIGUOUS


def test_trust_profile_still_denies_native_network_syscalls(tmp_path):
    source='import ctypes\ndef resolve(p):\n ctypes.CDLL(None).syscall(41,2,1,0);return {}\n'
    payload,context,_=context_fixture(tmp_path,source)
    result=TrustedContextRunner().run_with_context(source,payload,context)
    assert result['isolation_verified'] and result['return_code']==-31
