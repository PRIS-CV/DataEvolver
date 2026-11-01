"""Engineering fixtures for dispatch/recovery; no real production quality claim."""
import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from dataevolver.dataset.contracts import ArtifactStore, file_sha256, payload_sha256
from dataevolver.harness.adapters.production_qualification import ProductionQualificationOperator
from dataevolver.harness.controller import EvolutionController
from dataevolver.harness.models import (ControllerState, EvidenceStatus, OperatorCall, OperatorReceipt, OperatorSpec,
                                       RecoveryStatus, StageStatus)
from dataevolver.harness.evaluator import EvaluatorContract
from dataevolver.harness.state_store import atomic_write_json
from dataevolver.harness.operators import FakeOperator, OperatorRegistry, InjectedOperatorCrash
from dataevolver.harness.production_qualification import (EVIDENCE_ROLES,
    ControllerProductionQualificationAuthority)
from dataevolver.testing.test_harness_controller import build_controller
from dataevolver.testing.test_operator_catalog import declaration
from dataevolver.testing.test_operator_development import write_ref
from dataevolver.testing.test_production_qualification import setup


class AdmissionFixture:
    """Engineering observation only; no real resource readiness claim."""
    def __init__(self, root):
        self.root, self.available, self.calls = root, True, 0

    def identity(self):
        return {'fixture':True, 'output_root':str(self.root), 'max_output_bytes':8192}

    def preflight(self, call, deadline):
        self.calls += 1
        p = self.root/(call.call_id+'.json')
        value = {'admitted':self.available, 'call_sha256':payload_sha256(call.to_dict()),
                 'resource_deadline_epoch':deadline, 'per_request_gateway_admission_required':True,
                 'reason':'fixture observation only'}
        atomic_write_json(p, value)
        return {**value, 'artifact_hashes':{str(p):file_sha256(p)}}

    def verify(self, call, proof):
        from dataevolver.harness.models import EffectVerification
        p = self.root/(call.call_id+'.json')
        expected = {**json.loads(p.read_text()), 'artifact_hashes':{str(p):file_sha256(p)}}
        return EffectVerification(EvidenceStatus.VERIFIED if expected==proof else EvidenceStatus.CONTRADICTED,
                                  'fixture-only resource receipt', artifact_hashes=proof['artifact_hashes'])


class WorkloadFixture(FakeOperator):
    def __init__(self, store, package):
        spec=OperatorSpec('fixture.production_workload','1',source_path=str(Path(__file__)),
                          code_revision='fixture-host-verifier-only',retry_safe=False)
        super().__init__(spec,status=StageStatus.SUCCESS,metrics={'quality':1.,'safety':1.},
                         hard_constraints={'artifact_valid':True})
        self.store,self.package=store,package
        self.execution=None;self.calls=0;self.cases={};self.reject_use=False;self.fail_check=False

    def execute(self,call):
        self.calls+=1
        family=call.parameters['family_id'];root=Path(call.work_dir)
        runtime={'profile':'fixture-only-confined-runtime','context_scope':'registered_production_runtime'}
        raw={'schema_version':'dataevolver.isolated_candidate_execution.v2','status':'COMPLETED',
             'isolation_verified':True,'return_code':0,'output':{'fixture':True},'stdout':'{"fixture":true}',
             'runtime':runtime,'binding':{'source_sha256':self.package.implementation['sha256'],
                                        'trusted_context_sha256':'d'*64}}
        execution=write_ref(self.store.roots['evidence'],'evidence',
            (root/'capture.json').relative_to(self.store.roots['evidence']).as_posix(),raw)
        artifacts={k:write_ref(self.store.roots['evidence'],'evidence',
            (root/(k+'.json')).relative_to(self.store.roots['evidence']).as_posix(),
            {'role':k,'fixture_only':True,'execution':execution}) for k in EVIDENCE_ROLES}
        checks={k:not self.fail_check for k in self.execution.stages['engineering']['check_ids']}
        self.cases[call.call_id]={'binding':{'execution_contract_sha256':self.execution.compute_hash(),
            'package_sha256':self.package.compute_hash(),'stage':'engineering','family_id':family,
            'source':call.input_artifact_refs[0],'matched_context_sha256':self.execution.execution_protocol().matched_context_sha256,
            'runtime':self.execution.runtime},'checks':checks,'artifacts':artifacts,
            'usage':{'operator_calls':1,'planner_calls':0,'model_calls':0,'image_experiments':0,
                     'output_bytes':10000,'wall_seconds':1},
            'candidate_invocation':{'package_sha256':self.package.compute_hash(),
                'implementation_sha256':self.package.implementation['sha256'],
                'actual_execution':not self.reject_use,'execution_artifact':execution}}
        return super().execute(call)

    def _receipt_from_effect(self,call,*,reused):
        r=super()._receipt_from_effect(call,reused=reused)
        return replace(r,stage_result={**r.stage_result,'production_case':self.cases[call.call_id]})

    def verify_effect(self,call,receipt):
        from dataevolver.dataset.supervision import artifact_closure
        result=super().verify_effect(call,receipt)
        case=self.cases[call.call_id]
        if receipt.stage_result.get('production_case')!=case:raise ValueError('host fixture rejects altered case')
        refs=list(case['artifacts'].values())+[call.input_artifact_refs[0],self.execution.matched_context]
        hashes=dict(result.artifact_hashes)
        for ref in artifact_closure(refs,self.store):
            path=self.store.resolve(ref);hashes[str(path)]=ref.sha256
        return replace(result,artifact_hashes=hashes)


def producer_fixture(tmp_path, *, preflight=False, admission_bound=True, probe_root=None):
    f,execution,parent,_,evaluator=setup(tmp_path)
    output=tmp_path/'producer'
    store=ArtifactStore({**f[4].roots,'evidence':output})
    parent.summary['production_requirements']['stages']=copy.deepcopy(execution.design_protocol['stages'])
    parent.summary['resource_deadline_epoch']=time.time()+100000
    budget=copy.deepcopy(execution.budget);budget['resource_deadline_epoch']=parent.summary['resource_deadline_epoch']
    budget['limits']['output_bytes']=4000000
    runtime={**execution.runtime,'runner_sha256':payload_sha256(
        {'profile':'fixture-only-confined-runtime','context_scope':'registered_production_runtime'})}
    execution=replace(execution,predecessor_summary_sha256=payload_sha256(parent.read()),budget=budget,runtime=runtime)
    worker=WorkloadFixture(store,f[3]);registry=OperatorRegistry()
    registry.register(worker,contract=declaration(worker.spec,category='evaluation',recovery_semantics='reconcile_only',
        parameter_schema={'type':'object','additionalProperties':False,'required':['family_id'],
                          'properties':{'family_id':{'type':'string'}}},output_schemas={'production_case':{'type':'object'}}))
    workloads=[]
    for i,s in enumerate(execution.execution_protocol().sources):
        if s['family_id'] not in execution.execution_protocol().families('engineering'):continue
        call=OperatorCall('fixture-workload','case-'+str(i),'case-'+str(i),worker.spec.operator_id,'1',i,
            str(output/'workloads'/('case-'+str(i))),parameters={'family_id':s['family_id']},input_artifact_refs=(s['artifact'],))
        workloads.append({'family_id':s['family_id'],'input':s['artifact'],'call':call.to_dict(),
            'upper_bound':{'operator_calls':1,'planner_calls':0,'model_calls':0,'image_experiments':0,
                           'output_bytes':50000,'wall_seconds':5}})
    kwargs=dict(stage='engineering',package=f[3],store=store,predecessor_authority=parent,
                workload_registry=registry,output_root=output,evidence_store='evidence',
                execution=None if preflight else execution,workloads=() if preflight else workloads,
                admission_probes={worker.spec.operator_id:AdmissionFixture(probe_root or output/'resources')} if admission_bound else {})
    op=ProductionQualificationOperator(**kwargs)
    if not preflight:
        stages=copy.deepcopy(execution.stages)
        stages['engineering'].update(validator_operator_id=op.spec.operator_id,
            validator_spec_sha256=op.spec.sha256(),validator_contract_sha256=op.operator_contract.compute_hash())
        execution=replace(execution,stages=stages)
        op=ProductionQualificationOperator(**{**kwargs,'execution':execution})
        assert op.spec.sha256()==stages['engineering']['validator_spec_sha256']
        worker.execution=execution
    call=OperatorCall('producer','producer-call','producer-call',op.spec.operator_id,'1',0,
                     str(output/'calls/first'),parameters=op.parameters())
    return op,worker,call,parent,store,f[3],execution


def test_registered_producer_preflight_writes_real_gap_result_without_dispatch(tmp_path):
    op,worker,call,*_=producer_fixture(tmp_path,preflight=True)
    receipt=op.execute(call)
    assert worker.calls==0 and receipt.stage_result['metrics']['production_workloads_admitted']==0
    assert op.verify_effect(call,receipt).status is EvidenceStatus.VERIFIED
    recovered=op.recover(call)
    assert recovered.status is RecoveryStatus.RECOVERED and recovered.receipt.reused
    report=json.loads((Path(call.work_dir)/'production_qualification.json').read_text())
    assert not report['qualified'] and report['admission']['reason']=='EXECUTION_CONTRACT_NOT_BOUND'
    assert report['lifecycle_usage']['output_bytes']>0


def test_composite_drives_registered_workloads_and_produces_existing_controller_abi(tmp_path):
    op,worker,_,parent,store,package,execution=producer_fixture(tmp_path)
    root=tmp_path/'controller';root.mkdir();template,_=build_controller(root)
    evaluator=root/'producer-evaluator.json'
    atomic_write_json(evaluator,{'schema_version':'dataevolver.harness.evaluator.v1','evaluator_id':'fixture-production',
        'revision':'1','hard_constraints':{'artifact_valid':{'required':True}},
        'target_metrics':{'production_qualification_passed':{'required':True,'minimum':1}},'protected_metrics':{}})
    stages=copy.deepcopy(execution.stages);stages['engineering']['evaluator_sha256']=file_sha256(evaluator)
    execution=replace(execution,stages=stages);op.execution=worker.execution=execution
    protocol=execution.execution_protocol()
    binding={'plan_sha256':execution.compute_hash(),'package_sha256':package.compute_hash(),
             'protocol_sha256':protocol.compute_hash(),'stage':'engineering',
             'matched_context_sha256':protocol.matched_context_sha256}
    registry=OperatorRegistry();registry.register(op)
    # Use the fixture's original Evaluator, whose hash is already in the design.
    goal=template.goal.with_updates(campaign_id='producer-controller',initial_operator_id=op.spec.operator_id,
        initial_operator_parameters=op.parameters(),interventions=(),
        objective={**template.goal.objective,'operator_qualification':binding,
                   'production_qualification_execution':execution.compute_hash()},
        frozen_evaluator={'path':str(evaluator),'sha256':file_sha256(evaluator)},
        resource_policy={'allow_gpu':False,'allowed_operators':[op.spec.operator_id]})
    c=EvolutionController.create(repo_root=root,campaigns_root=op.output_root/'calls',goal=goal,registry=registry,
        planner=template.planner,evaluator_contract=EvaluatorContract.from_path(evaluator),idempotency_key='producer-fixture')
    assert c.run().state is ControllerState.COMPLETE
    committed=next(e for e in c.store.events.read() if e.event_type=='ACTION_COMMITTED')
    receipt=OperatorReceipt.from_dict(committed.payload['receipt'])
    assert worker.calls==2 and receipt.stage_result['check_results']=={k:True for k in execution.stages['engineering']['check_ids']}
    assert op.verify_effect(OperatorCall.from_dict(next(e for e in c.store.events.read()
        if e.event_type=='ACTION_PLANNED').payload['operator_call']),receipt).status is EvidenceStatus.VERIFIED
    assert set(receipt.stage_result['production_evidence']['artifacts'])==EVIDENCE_ROLES
    assert c._frozen()[0] and c.replay().to_dict()==c.status().to_dict()
    authority=ControllerProductionQualificationAuthority(c,committed.payload['call_id'],execution=execution,
        package=package,store=store,predecessor_authority=parent)
    assert authority.verify('engineering')['production_qualification_evidence_verified']


def test_no_resend_after_incomplete_workload_receipt(tmp_path):
    op,worker,call,*_=producer_fixture(tmp_path)
    worker.crash_after_effect=True
    with pytest.raises(InjectedOperatorCrash):op.execute(call)
    assert op.recover(call).status is RecoveryStatus.AMBIGUOUS
    with pytest.raises(ValueError):op.execute(call)
    assert worker.calls==1


def test_completed_workloads_recover_after_parent_report_write_crash(tmp_path):
    op,worker,call,*_=producer_fixture(tmp_path)
    original=op._save_report
    op._save_report=lambda *a: (_ for _ in ()).throw(InjectedOperatorCrash('parent report'))
    with pytest.raises(InjectedOperatorCrash):op.execute(call)
    assert worker.calls==2
    op._save_report=original
    recovered=op.recover(call)
    assert recovered.status is RecoveryStatus.RECOVERED and recovered.receipt.reused
    assert worker.calls==2


@pytest.mark.parametrize('change',['no_actual_use','budget','stale_manifest'])
def test_producer_never_turns_missing_use_or_budget_into_qualification(tmp_path,change):
    op,worker,call,*_=producer_fixture(tmp_path)
    if change=='no_actual_use':worker.reject_use=True
    elif change=='budget':
        value=copy.deepcopy(op.execution.budget);value['limits']['operator_calls']=value['prior_usage']['operator_calls']+1
        op.execution=replace(op.execution,budget=value);call=replace(call,parameters=op.parameters())
    else:op.workloads[0]['family_id']='changed'
    with pytest.raises(ValueError):op.execute(call)
    assert worker.calls==(1 if change=='no_actual_use' else 0)


def test_completed_producer_cannot_be_given_another_owner(tmp_path):
    op,worker,call,*_=producer_fixture(tmp_path)
    op.execute(call)
    with pytest.raises(ValueError,match='reset'):
        op.execute(replace(call,call_id='other',idempotency_key='other',work_dir=str(op.output_root/'calls/other')))
    assert worker.calls==2


def test_expired_deadline_delegates_readiness_without_extending_it(tmp_path):
    op, worker, call, parent, *_ = producer_fixture(tmp_path)
    deadline = 1
    parent.summary['resource_deadline_epoch'] = deadline
    budget = {**op.execution.budget, 'resource_deadline_epoch':deadline}
    op.execution = worker.execution = replace(op.execution, budget=budget,
        predecessor_summary_sha256=payload_sha256(parent.read()))
    call = replace(call, parameters=op.parameters())
    receipt = op.execute(call)
    assert worker.calls==2 and receipt.stage_result['metrics']['production_workloads_admitted']==1
    report=json.loads((Path(call.work_dir)/'production_qualification.json').read_text())
    assert all(r['proof']['resource_deadline_epoch']==deadline for r in report['admission']['resources'])
    assert op.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    assert op.recover(call).status is RecoveryStatus.RECOVERED
    assert op.admission_probes[worker.spec.operator_id].calls==2


@pytest.mark.parametrize('bound', [True, False])
def test_missing_or_unavailable_resource_admission_never_dispatches(tmp_path,bound):
    op, worker, call, *_ = producer_fixture(tmp_path, admission_bound=bound)
    if bound:
        op.admission_probes[worker.spec.operator_id].available=False
    receipt=op.execute(call)
    assert worker.calls==0 and receipt.stage_result['metrics']['production_workloads_admitted']==0
    report=json.loads((Path(call.work_dir)/'production_qualification.json').read_text())
    assert report['admission']['reason']==('WORKLOAD_RESOURCES_UNAVAILABLE' if bound else 'RESOURCE_ADMISSION_NOT_BOUND')


def test_resource_adapter_cannot_expand_the_producer_write_roots(tmp_path):
    root=tmp_path/'outside'
    with pytest.raises(ValueError,match='producer write roots'):
        producer_fixture(tmp_path,probe_root=root)
    assert not root.exists()
