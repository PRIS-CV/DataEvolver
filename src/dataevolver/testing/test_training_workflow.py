"""Fixture engineering checks only; no real model quality or RSI benefit claims."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataevolver.dataset.contracts import payload_sha256
from dataevolver.dataset.training_pipeline import QUALITY_CHECKS
from dataevolver.harness.state_store import atomic_write_json
from dataevolver.harness.training_workflow import (TrainingWorkflowDomain,task_scope,
    validate_policy,verify_runtime_pair)
from dataevolver.harness.training_workflow_delivery import merge_usage,capture_execution_audit
from dataevolver.harness.runtime_rsi import validate_decision
from dataevolver.testing.test_runtime_rsi import action,make_runtime
from dataevolver.harness.models import BudgetSpec,ControllerState


def pair(tmp):
    review={'status':'pass','criteria':{k:{'status':'pass'} for k in QUALITY_CHECKS}}
    source={'sha256':'a'*64};old={'sha256':'b'*64};new={'sha256':'c'*64}
    evidence={'source':source,'candidate':old,'proposal_actor':'runtime_agent'}
    path=tmp/'probe.json';atomic_write_json(path,evidence)
    patch={'proposal_actor':'runtime_agent','layer':'R','source':source,
           'evidence':[{'path':str(path),'sha256':payload_sha256(evidence)}],
           'experience':{'capability_id':'fixture-K'}}
    base={'qualified':False,'source':source,'candidate':old,'source_family':'fixture-a0',
          'inner_work_dir':str(tmp/'baseline'),'media':[{'artifact':old}]*6}
    candidate={**base,'qualified':True,'candidate':new,'inner_work_dir':str(tmp/'candidate'),
        'runtime_visual_actor':'runtime_agent','model_review':review,'runtime_visual_review':review,
        'patch':patch,'experience_id':'fixture-K','media':[{'artifact':new}]*6}
    rows=[{'request':{'request_id':'fixture-'+p,'role':'fixture','seed':7,'response_schema':{}},
           'receipt_hash':'fixture-'+p} for p in ('refine-generate','generate','resolve','segment','shadow-probe','background')]
    for trial in (base,candidate):
        actual=copy.deepcopy(rows)
        if trial is candidate:actual[-1]['request']['condition']='changed-by-runtime-policy'
        atomic_write_json(Path(trial['inner_work_dir'])/'families/fixture-a0.json',{'model_requests':actual})
    history=[{'decision':{'decision':'ADOPT_CANDIDATE'},'intervention':{'candidate_delta':patch}}]
    return base,candidate,history


def test_candidate_is_not_eligible_outside_explicit_independent_validation():
    value=action('RUN_EXPERIMENT',layer='R',patch='effect_support',experience='fixture-K')
    context={'available_evidence_ids':['baseline'],'allowed_patches':['effect_support'],
             'testable_experience_ids':['fixture-K'],'compatible_experience_ids':[]}
    with pytest.raises(ValueError,match='ineligible'):validate_decision(value,context)
    context['knowledge_validation_only']=True
    assert validate_decision(value,context)['experience_id']=='fixture-K'


def test_real_reuse_contract_requires_adoption_current_source_and_changed_request(tmp_path):
    baseline,candidate,history=pair(tmp_path)
    assert verify_runtime_pair(baseline,candidate,history,experience_id='fixture-K')['verified']
    bad=copy.deepcopy(candidate);bad['experience_id']=''
    with pytest.raises(ValueError,match='actual compiled'):verify_runtime_pair(baseline,bad,history,experience_id='fixture-K')
    bad=copy.deepcopy(candidate);bad['source']={'sha256':'different-source'}
    with pytest.raises(ValueError,match='same-source'):verify_runtime_pair(baseline,bad,history)
    with pytest.raises(ValueError,match='adopted'):verify_runtime_pair(baseline,candidate,[])
    baseline['qualified']=True
    with pytest.raises(ValueError,match='failed baseline'):verify_runtime_pair(baseline,candidate,history)


def test_policy_cannot_copy_source_roi_or_task_answer():
    policy={k:'Inspect current evidence before a conditional repair.' for k in
            ['failure_mode','applicability','probe_instruction','repair_instruction']}
    policy['precautions']=['Do not expand over preserved objects.']
    assert validate_policy(policy)==policy
    for text in ['Use box_xyxy_normalized [0.1,0.2,0.3,0.4]','Repeat main-train-004','Read /unavailable/private/file.png']:
        with pytest.raises(ValueError):validate_policy({**policy,'repair_instruction':text})


def test_empirical_scope_does_not_extrapolate_selection_count_or_model_version():
    task={'selection_mode':'all','subjects':[{},{}],'selected_subject_ids':['s1','s2']}
    scope=task_scope(task,{'revision':'fixture-v1'})
    assert scope!=task_scope({**task,'selection_mode':'selected'}, {'revision':'fixture-v1'})
    assert scope!=task_scope({**task,'subjects':[{}, {},{}]}, {'revision':'fixture-v1'})
    assert scope!=task_scope(task, {'revision':'fixture-v2'})


def test_one_controller_automatically_advances_distinct_queue_calls_and_replays(tmp_path):
    c,backend=make_runtime(tmp_path,[action('COMPLETE'),action('COMPLETE')])
    original=c.planner.domain
    class QueueDomain:
        def config(self):return {'kind':'fixture-two-task-queue'}
        def context(self,state):return original.context(state)
        def initial_parameters(self,state):return {'fixture_stage':state.data.get('queue',0)}
        def complete_stage(self,controller,state):
            if state.data.get('queue',0)==1:return controller._complete(state),'QUEUE_COMPLETE'
            data={**state.data,'queue':1}
            return controller._commit(state,ControllerState.PLAN,'QUEUE_ADVANCE','NEXT_TASK',data=data,
                counters={'round_index':state.round_index+1}),'NEXT_TASK'
    domain=QueueDomain();c.planner.domain=domain
    # New controller registration freezes the queue hooks rather than mutating a live contract.
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.runtime_rsi import RuntimeRSIPlanner
    from dataevolver.harness.operators import FakeOperator
    old=c.registry.get('fake.baseline')
    c.registry._operators['fake.baseline']=FakeOperator(old.spec,status=old.status,
        metrics={'quality':.95,'safety':1.0},hard_constraints={'artifact_valid':True})
    goal=c.goal.with_updates(campaign_id='fixture-queue',budgets=BudgetSpec(5,8,2,0,60,4))
    planner=RuntimeRSIPlanner(backend=backend,domain=domain,trace_root=tmp_path/'campaigns/fixture-queue/planner')
    q=EvolutionController.create(repo_root=tmp_path,campaigns_root=tmp_path/'campaigns',goal=goal,
        registry=c.registry,planner=planner,evaluator_contract=c.evaluator.contract,
        execution_paths=(Path(old.spec.source_path),),idempotency_key='fixture-queue')
    result=q.run()
    assert result.state is ControllerState.COMPLETE and result.operator_calls==2
    assert q.replay().to_dict()==result.to_dict()
    assert len(backend.calls)==2
    assert sorted(p.name for p in (q.layout.campaign_dir/'rounds').iterdir())==['r000','r001']
    before=q.layout.events_path.read_bytes();q.run();assert q.layout.events_path.read_bytes()==before


def test_task_budget_allows_consuming_its_last_recorded_decision():
    domain=TrainingWorkflowDomain.__new__(TrainingWorkflowDomain)
    domain.value={'limits':{'planner_per_task':3}}
    state=SimpleNamespace(planner_calls=3,data={'training_workflow':{'planner_start':0}})
    assert not domain.decision_budget_available(state)
    state.data['rsi_pending_decision']={'action':'COMPLETE'}
    assert domain.decision_budget_available(state)


def test_held_out_quality_repair_does_not_create_new_knowledge(tmp_path):
    from dataevolver.testing.test_runtime_rsi import decisions
    c,_=make_runtime(tmp_path,decisions())
    c.planner.domain.allow_knowledge_learning=lambda state:False
    # The hook is part of this fixture's frozen registration from creation;
    # adding a bound method does not change its declared config.
    result=c.run()
    assert result.state is ControllerState.COMPLETE
    assert result.data['rsi_history'][0]['decision']['decision']=='ADOPT_CANDIDATE'
    assert result.data['rsi_history'][0]['knowledge_status']=='HELD_OUT_REPAIR_ONLY'
    assert not c.capabilities.list_capabilities()


def test_usage_union_retains_failures_without_double_counting_or_inventing_cost():
    row={'request_id':'fixture-r','request_hash':'same','attempts':1,'wall_seconds':2,'monetary_cost':None,'state':'FAILED'}
    usage={'attempts':[row]}
    assert merge_usage(usage,usage)['model_calls']==1
    assert merge_usage(usage)['monetary_cost'] is None
    with pytest.raises(ValueError):merge_usage(usage,{'attempts':[{**row,'request_hash':'changed'}]})


def test_audit_copies_exact_failed_native_and_rejects_unknown_effect(tmp_path):
    root=tmp_path/'run';ledger=root/'model_receipts/fixture-r.intent.json'
    value={'request_hash':'fixture','attempts':1,'wall_seconds':2,'state':'FAILED','role':'fixture'}
    atomic_write_json(ledger,value)
    native=root/'native/review/fixture-r.native.failure.json';atomic_write_json(native,{'fixture':True})
    result=capture_execution_audit([root],tmp_path/'audit')
    assert result['usage']['model_calls']==1 and result['usage']['observed_wall_seconds']==2
    assert result['model_calls_submitted_by_audit']==0
    assert (tmp_path/'audit/runs/run000/native/review/fixture-r.native.failure.json').read_bytes()==native.read_bytes()
    atomic_write_json(ledger,{**value,'state':'STARTED'})
    with pytest.raises(ValueError,match='unknown model effect'):capture_execution_audit([root],tmp_path/'other-audit')


def test_declared_preflight_reservation_keeps_cost_without_inventing_native_inference(tmp_path):
    root=tmp_path/'run'
    atomic_write_json(root/'model_receipts/fixture-admission.intent.json',{'request_hash':'fixture','attempts':1,
        'wall_seconds':.002,'state':'RETRY_SAFE','error_code':'INVALID_INPUT','role':'fixture'})
    result=capture_execution_audit([root],tmp_path/'audit')
    assert result['usage']['model_calls']==1 and result['usage']['observed_wall_seconds']==.002
    assert result['preflight_rejections_without_native_forward']['fixture-admission']['classification']=='provider_preflight_rejected'


def test_runtime_audit_follows_embedded_preceding_production_contract(tmp_path):
    old=tmp_path/'old';root=tmp_path/'current'
    atomic_write_json(old/'config.json',{'production':{'preceding_runs':[]}})
    atomic_write_json(old/'model_receipts/fixture-old.intent.json',{'request_hash':'fixture','attempts':1,
        'wall_seconds':2,'state':'FAILED','role':'fixture'})
    atomic_write_json(old/'native/fixture-old.native.failure.json',{'fixture':True})
    atomic_write_json(root/'goal.json',{'planner':{'runtime':{'domain':{'production_contract':{'production':{
        'preceding_runs':[{'config_dir':str(old)}]}}}}}})
    result=capture_execution_audit([root],tmp_path/'audit')
    assert result['usage']['model_calls']==1 and 'fixture-old' in result['requests']


def test_audit_resolves_legacy_effect_origin_by_declared_config_hash(tmp_path):
    from dataevolver.dataset.contracts import file_sha256
    old=tmp_path/'experiments/old';root=tmp_path/'experiments/current'
    atomic_write_json(old/'config.json',{'production':{'fixture':True}})
    atomic_write_json(old/'model_receipts/fixture-origin.intent.json',{'request_hash':'fixture','attempts':1,
        'wall_seconds':2,'state':'FAILED','role':'fixture'})
    atomic_write_json(old/'native/fixture-origin.native.failure.json',{'fixture':True})
    atomic_write_json(root/'config.json',{'production':{'generation_reuse':{'fixture':{
        'source_config_sha256':file_sha256(old/'config.json')}}}})
    result=capture_execution_audit([root],tmp_path/'audit',config_roots=[tmp_path/'experiments'])
    assert result['usage']['model_calls']==1 and 'fixture-origin' in result['requests']


def workflow_goal():
    from dataevolver.harness.training_workflow import SCHEMA
    from dataevolver.harness.training_workflow_run import workflow_budgets
    from dataevolver.harness.models import GoalSpec
    task={'slot_id':'fixture','split':'train','seed':7,'selection_mode':'all',
        'subjects':[{'description':'fixture item'}]*2,'selected_subject_ids':['s1','s2']}
    stages=[{'id':'distill','kind':'distill'},
        {'id':'independent','kind':'validation','task':{**task,'slot_id':'independent'}},
        {'id':'promote','kind':'promote'}]
    for split,count in [('train',12),('validation',6),('test',6)]:
        for i in range(count):
            stages.append({'id':f'{split}-{i}','kind':'main','task':{**task,'slot_id':f'{split}-{i}','split':split}})
    stages.append({'id':'export','kind':'export'})
    d={'schema_version':SCHEMA,'stages':stages,'production_contract':{'production':{'evidence_kind':'real_execution'},'models':{}},
        'experience_scope':task_scope(task,{}),'limits':{'experiments_per_task':2,'planner_per_task':8,'max_experiments':50},
        'gpu0_uuid':'fixture-gpu0','gpu_uuids':['fixture-gpu0'],'expected_counts':{'train':18,'validation':6,'test':6},
        'accepted_counts':{'train':6,'validation':0,'test':0},'accepted_proofs':['fixture-proof']*6}
    return GoalSpec.from_dict({'schema_version':'dataevolver.harness.goal.v1','campaign_id':'fixture-workflow',
        'objective':{'real_data_execution':True},'initial_operator_id':'data.training_rsi_trial',
        'budgets':workflow_budgets(d,60),'resource_policy':{'physical_gpu_indices':[0]},
        'frozen_evaluator':{'path':'fixture-evaluator.json','sha256':'f'*64},
        'planner':{'kind':'runtime_rsi','runtime':{'domain':d}}})


def test_workflow_authority_binds_original_split_gpu0_and_finite_queue():
    from dataevolver.harness.training_workflow_run import validate_workflow_contract
    goal=workflow_goal();validate_workflow_contract(goal)
    with pytest.raises(ValueError,match='GPU1'):
        validate_workflow_contract(goal.with_updates(resource_policy={'physical_gpu_indices':[1]}))
    d=copy.deepcopy(goal.planner['runtime']['domain']);d['stages'][1]['task']['split']='test'
    with pytest.raises(ValueError,match='validation/test'):
        validate_workflow_contract(goal.with_updates(planner={'runtime':{'domain':d}}))
    d=copy.deepcopy(goal.planner['runtime']['domain']);d['production_contract']['production']['evidence_kind']='fixture'
    with pytest.raises(ValueError,match='fixture'):
        validate_workflow_contract(goal.with_updates(planner={'runtime':{'domain':d}}))


def test_observation_protocol_handoff_never_restarts_an_unknown_effect_or_budget():
    from dataevolver.harness.training_rsi_run import inherited_planner_usage
    goal=SimpleNamespace(objective={'source_family':'fixture-source'},
        planner={'runtime':{'domain':{'prior_runtime':{'planner_calls':2}}}})
    state={'state':'ABSTAIN','terminal_reason':'RSI_AGENT_ABSTAINED','experiments_used':0,'planner_calls':4}
    assert inherited_planner_usage(state,goal,'fixture-source')==6
    for change in [{'terminal_reason':'UNKNOWN_EFFECT'},{'experiments_used':1},{'state':'PROPOSE'},{'planner_calls':8}]:
        with pytest.raises(ValueError):inherited_planner_usage({**state,**change},goal,'fixture-source')


def test_audit_includes_implemented_capability_request_and_its_original_native(tmp_path):
    from dataevolver.dataset.contracts import file_sha256
    diagnostic=tmp_path/'diagnostic';current=tmp_path/'current'
    request=diagnostic/'campaigns/diagnostic/capability-requests/p000.json'
    atomic_write_json(request,{'fixture':'original runtime request, not a repair answer'})
    atomic_write_json(diagnostic/'model_receipts/fixture-capability.intent.json',{
        'request_hash':'fixture','attempts':1,'wall_seconds':3,'state':'FAILED','role':'agent.runtime_rsi'})
    native=diagnostic/'native/fixture-capability.native.failure.json';atomic_write_json(native,{'fixture':'exact failure bytes'})
    atomic_write_json(current/'goal.json',{'planner':{'runtime':{'domain':{'surface_capability':{
        'diagnostic_config':str(diagnostic),'request_path':str(request),'request_sha256':file_sha256(request)}}}}})
    result=capture_execution_audit([current],tmp_path/'audit')
    assert 'fixture-capability' in result['requests'] and str(request) in result['source_hashes']
    assert str(native) in result['source_hashes'] and result['source_journals_unchanged']
    assert result['model_calls_submitted_by_audit']==0
    atomic_write_json(request,{'fixture':'changed request'})
    with pytest.raises(ValueError,match='capability request changed'):
        capture_execution_audit([current],tmp_path/'audit-invalid')
