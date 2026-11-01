"""Independent-family isolation, frozen reader bindings and refusal gates."""
import json
import os
import platform

import pytest

from dataevolver.dataset.contracts import payload_sha256
from dataevolver.harness.authority_qualification import AuthorityFixtureExpectation, authenticated
from dataevolver.harness.candidate_sandbox import TrustedContextRunner, SandboxLimits
from dataevolver.harness.candidate_trust import CandidateTrustContext
from dataevolver.harness.contract_execution_authority import FrozenContractExecutionAuthority
from dataevolver.harness.adapters.operator_independent_contract import IndependentContractQualificationOperator
from dataevolver.harness.operator_development import DevelopmentBudget
from dataevolver.harness.models import EvidenceStatus, OperatorCall
from dataevolver.testing.independent_authority_vectors import make_independent_vectors
from dataevolver.testing.test_candidate_trust_qualification import qualification_fixture


def test_independent_corpus_declares_two_new_nontrivial_geometry_families(tmp_path):
    v=make_independent_vectors(tmp_path/'independent',package_sha256='a'*64,
        implementation_sha256='b'*64,request_sha256='c'*64)
    assert len(v['family_ids'])==2 and len(v['cases'])==6
    widths=set();subjects=set();outcomes=[]
    for case in v['cases']:
        read=lambda ref:json.loads((tmp_path/ref['path']).read_text())
        payload=read(case['input']);ctx=CandidateTrustContext.from_dict(read(v['trust_contexts'][case['case_id']]))
        gold=AuthorityFixtureExpectation.from_dict(read(v['expectations'][case['case_id']]))
        assert authenticated(payload,ctx) is gold.authenticated
        widths.add(payload['evidence']['source']['manifest']['width']);subjects.add(len(payload['evidence']['instances']))
        outcomes.append(gold.status)
    assert widths=={1536,2048} and subjects=={1,2} and outcomes.count('executable')==2
    with pytest.raises(FileExistsError):
        make_independent_vectors(tmp_path/'independent',package_sha256='a'*64,implementation_sha256='b'*64,request_sha256='c'*64)


def test_historical_reader_rejects_candidate_import_roots_before_starting_a_process(tmp_path):
    candidate=tmp_path/'candidate';source=candidate/'source';source.mkdir(parents=True)
    (candidate/'source-commit').write_text('a'*40)
    with pytest.raises(ValueError,match='module root'):
        FrozenContractExecutionAuthority(config=tmp_path,module_root=source,plan_path=tmp_path/'plan',
            package_path=tmp_path/'package',stores={'candidate':candidate},request_locations={},
            execution_roots=(source,),sandbox_limits={'output_bytes':8192},env_path=candidate/'env.sh',python='/usr/bin/python3')


def test_reader_pins_a_separate_host_environment_and_rejects_candidate_owned_environment(tmp_path):
    import sys
    cfg=tmp_path/'config';source=cfg/'source';source.mkdir(parents=True)
    (cfg/'source-commit').write_text('a'*40);(cfg/'protected').mkdir()
    (cfg/'protected/qualification-goal.json').write_text(json.dumps({'campaign_id':'fixture'}))
    frozen=cfg/'output/campaigns/fixture';frozen.mkdir(parents=True);(frozen/'frozen_contract.json').write_text('{}')
    plan=cfg/'protected/plan.json';plan.write_text(json.dumps({'candidate_store':'candidate'}))
    package=cfg/'protected/package.json';package.write_text('{}')
    envcfg=tmp_path/'historical-builder';envcfg.mkdir();(envcfg/'source-commit').write_text('b'*40)
    env=envcfg/'env.sh';env.write_text('export FIXTURE_ONLY=1\n')
    candidate=tmp_path/'candidate';candidate.mkdir()
    kwargs=dict(config=cfg,module_root=source,plan_path=plan,package_path=package,stores={'candidate':candidate},
        request_locations={},execution_roots=(source,),sandbox_limits={'output_bytes':8192},env_path=env,python=sys.executable)
    reader=FrozenContractExecutionAuthority(**kwargs);assert reader.identity()['env_source_commit']=='b'*40
    env.write_text('export FIXTURE_ONLY=2\n')
    with pytest.raises(ValueError,match='binding changed'):reader.identity()
    (candidate/'source-commit').write_text('c'*40);bad=candidate/'env.sh';bad.write_text('exit 0\n')
    with pytest.raises(ValueError,match='outside candidate'):FrozenContractExecutionAuthority(**{**kwargs,'env_path':bad})


@pytest.mark.parametrize('change', ['added', 'modified', 'deleted'])
def test_verified_reader_cache_needs_identical_complete_dependency_inventory(tmp_path, monkeypatch, change):
    import sys
    from types import SimpleNamespace
    cfg=tmp_path/'config';source=cfg/'source';source.mkdir(parents=True)
    (cfg/'source-commit').write_text('a'*40);(cfg/'protected').mkdir()
    (cfg/'protected/qualification-goal.json').write_text(json.dumps({'campaign_id':'fixture'}))
    frozen=cfg/'output/campaigns/fixture';frozen.mkdir(parents=True);(frozen/'frozen_contract.json').write_text('{}')
    plan=cfg/'protected/plan.json';plan.write_text(json.dumps({'candidate_store':'candidate'}))
    package=cfg/'protected/package.json';package.write_text('{}')
    env=cfg/'env.sh';env.write_text('export FIXTURE_ONLY=1\n')
    candidate=tmp_path/'candidate';candidate.mkdir();artifact=candidate/'source.py';artifact.write_text('inert')
    reader=FrozenContractExecutionAuthority(config=cfg,module_root=source,plan_path=plan,package_path=package,
        stores={'candidate':candidate},request_locations={},execution_roots=(source,),sandbox_limits={'output_bytes':8192},env_path=env,python=sys.executable)
    calls=[]
    def run(*args,**kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout=json.dumps({'origin':{'campaign_id':'fixture',
            'frozen_sha256':reader.identity()['frozen_sha256']},'sealed':[1]}))
    monkeypatch.setattr('dataevolver.harness.contract_execution_authority.subprocess.run',run)
    first=reader.read();first['sealed'].append(2)
    assert reader.read()['sealed']==[1] and len(calls)==1
    if change=='added':(candidate/'extra.py').write_text('inert-extra')
    elif change=='modified':artifact.write_text('changed')
    else:artifact.unlink()
    with pytest.raises(ValueError,match='evidence changed'):reader.read()
    assert len(calls)==1


class FixtureEngineeringAuthority:
    """A test host authority; never used in the real qualification launcher."""
    def __init__(self,package,prior,deadline):
        self.value={'scope':'synthetic_authority_contract_engineering','pure_data_engineering_qualification':'PASSED',
            'engineering_evaluator_accepted':True,
            'origin':{'package_sha256':package.compute_hash()},'production_adoption_allowed':False,
            'fixture_family_ids':['synthetic-aperture-plane-engineering'],'input_sha256s':['d'*64],
            'report_sha256':'e'*64,'lifecycle_total':prior,'resource_deadline_epoch':deadline}
    def identity(self):return {'fixture_host_authority':'not-production'}
    def read(self):return self.value


def fixture(tmp_path):
    previous,_=qualification_fixture(tmp_path)
    vectors=make_independent_vectors(previous.store.roots['protocol']/'independent',
        package_sha256=previous.package.compute_hash(),implementation_sha256=previous.package.implementation['sha256'],
        request_sha256=previous.protocol['qualification_request']['request']['sha256'])
    budget=DevelopmentBudget.from_dict(previous.protocol['budget'])
    parent=FixtureEngineeringAuthority(previous.package,budget.prior_usage,budget.resource_deadline_epoch)
    runner=TrustedContextRunner(limits=SandboxLimits(output_bytes=8192))
    protocol={**previous.protocol,**vectors,'schema_version':'dataevolver.independent_contract_qualification.v1',
        'qualification_scope':'synthetic_authority_contract_independent','runtime_sha256':payload_sha256(runner.identity()),
        'engineering_origin':{'authority':parent.identity(),'report_sha256':parent.read()['report_sha256'],
            'package_sha256':previous.package.compute_hash()}}
    kwargs=dict(plan=previous.plan,package=previous.package,store=previous.store,request_authority=previous.request_authority,
        runner=runner,check_protocol=protocol,qualification_request_authority=previous.qualification_request_authority,
        output_root=tmp_path/'independent-output',engineering_authority=parent)
    return kwargs,parent


@pytest.mark.skipif(platform.system()!='Linux' or os.geteuid()!=0,reason='actual Linux sandbox required')
def test_independent_stage_cannot_qualify_a_refusing_candidate_or_reexecute_finished_calls(tmp_path,monkeypatch):
    kwargs,parent=fixture(tmp_path);op=IndependentContractQualificationOperator(**kwargs)
    call=OperatorCall('independent-fixture','independent-1','independent-1',op.spec.operator_id,'1',0,str(op.output_root/'call'))
    receipt=op.execute(call);report=json.loads(op._path(call).read_text())
    assert report['pure_data_independent_qualification']=='FAILED'
    assert receipt.stage_result['metrics']['independent_contract_checks_passed']==0
    assert report['pure_data_engineering_qualification']=='PASSED' and report['independent_qualification']=='NOT_ASSESSED'
    assert not report['production_adoption_allowed'] and op.verify_effect(call,receipt).status is EvidenceStatus.VERIFIED
    monkeypatch.setattr(op.runner,'run_with_context',lambda *args:pytest.fail('finished candidate rerun'))
    assert op.execute(call).reused


@pytest.mark.parametrize('change',['failed_engineering','evaluator_rejected','different_package','reused_family','reused_payload','lost_cost','later_deadline'])
@pytest.mark.skipif(platform.system()!='Linux' or os.geteuid()!=0,reason='actual Linux sandbox required')
def test_independent_admission_rejects_weak_or_leaked_evidence_before_execution(tmp_path,change):
    kwargs,parent=fixture(tmp_path)
    if change=='failed_engineering':parent.value['pure_data_engineering_qualification']='FAILED'
    elif change=='evaluator_rejected':parent.value['engineering_evaluator_accepted']=False
    elif change=='different_package':parent.value['origin']['package_sha256']='f'*64
    elif change=='reused_family':parent.value['fixture_family_ids']=list(kwargs['check_protocol']['family_ids'])
    elif change=='reused_payload':parent.value['input_sha256s']=[kwargs['check_protocol']['cases'][0]['input']['sha256']]
    elif change=='lost_cost':parent.value['lifecycle_total']={**parent.value['lifecycle_total'],'model_calls':99}
    else:parent.value['resource_deadline_epoch']+=1
    with pytest.raises(ValueError):IndependentContractQualificationOperator(**kwargs)
    assert not (tmp_path/'independent-output').exists()


@pytest.mark.skipif(platform.system()!='Linux' or os.geteuid()!=0,reason='actual Linux sandbox required')
def test_committed_independent_summary_retains_refusal_and_seals_payloads(tmp_path):
    from dataevolver.harness.contract_execution_authority import read_independent_controller
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.evaluator import EvaluatorContract
    from dataevolver.harness.operators import OperatorRegistry
    from dataevolver.harness.planner import DeterministicPlanner
    from dataevolver.harness.state_store import atomic_write_json
    from dataevolver.dataset.contracts import file_sha256
    from dataevolver.testing.test_harness_controller import build_controller
    kwargs,_=fixture(tmp_path);op=IndependentContractQualificationOperator(**kwargs)
    template,_=build_controller(tmp_path/'template');evaluator=tmp_path/'independent-evaluator.json'
    atomic_write_json(evaluator,{'schema_version':'dataevolver.harness.evaluator.v1','evaluator_id':'independent-fixture','revision':'1',
        'hard_constraints':{'artifact_valid':{'required':True}},
        'target_metrics':{'independent_contract_checks_passed':{'required':True,'minimum':1}},'protected_metrics':{}})
    goal=template.goal.with_updates(campaign_id='committed-independent-fixture',initial_operator_id=op.spec.operator_id,
        objective={**template.goal.objective,'candidate_package_sha256':op.package.compute_hash(),
            'contract_qualification_protocol_sha256':payload_sha256(op.protocol)},interventions=(),
        resource_policy={'allow_gpu':False,'allowed_operators':[op.spec.operator_id]},
        frozen_evaluator={'path':str(evaluator),'sha256':file_sha256(evaluator)})
    registry=OperatorRegistry();registry.register(op)
    c=EvolutionController.create(repo_root=tmp_path,campaigns_root=op.output_root/'campaigns',goal=goal,registry=registry,
        planner=DeterministicPlanner(()),evaluator_contract=EvaluatorContract.from_path(evaluator),idempotency_key=goal.campaign_id)
    c.run();call=next(e.payload['call_id'] for e in c.store.events.read() if e.event_type=='ACTION_COMMITTED')
    value=read_independent_controller(c,op,call)
    assert value['pure_data_independent_qualification']=='FAILED'
    assert not value['production_adoption_allowed'] and all('input' not in case and 'actual' not in case for case in value['cases'])
    with pytest.raises(ValueError,match='exact committed'):read_independent_controller(c,op,'uncommitted')
