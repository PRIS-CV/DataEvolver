"""Reject ambiguous/partial recovery; retain original decisions and failed costs."""
import copy,json
from types import SimpleNamespace
import pytest
from dataevolver.harness.boundary_recovery import check_failed_boundary_stage, BoundaryProbeRecoveryDomain
from dataevolver.dataset.contracts import payload_sha256
from dataevolver.testing.test_boundary_evidence import query, fixture_inputs


def failure():
    q=query();request={'request_id':'failed-native','messages':[{'role':'user','content':'native'}]}
    return {'decision':{'action':'OBSERVE','observation':{'tool':'probe_instance_boundaries','parameters_json':json.dumps(q)}},
        'pending_call':{'operator_id':'data.probe_instance_boundaries','parameters':q},'request':request,
        'native_job':{'request':copy.deepcopy(request),'request_hash':'a'*64,'binding':'b'*64},
        'native_failure':{'code':'KeyError','message':"'loading_info'",'binding':'b'*64,'resident_pid':123},
        'gateway_intent':{'state':'FAILED','error_code':'RECORDED_FAILURE','wall_seconds':2.5,'request_hash':'a'*64},
        'old_resident_pid':123,'successful_output_present':False,'new_dataset_files':[],'sam_forward_count':0,
        'visual_review_calls':0,'unknown_duration_attempts':0,'missing_component_before_forward_verified':True}


def test_known_failed_stage_retains_exact_original_runtime_arguments_and_spent_cost():
    value=failure();before=copy.deepcopy(value);decision=check_failed_boundary_stage(value)
    assert decision==before['decision'] and value==before
    decision['observation']['parameters_json']='{}'
    assert value==before and value['gateway_intent']['wall_seconds']==2.5


@pytest.mark.parametrize('mutate',[
 lambda v:v.update(successful_output_present=True),lambda v:v.update(new_dataset_files=['partial-mask']),
 lambda v:v.update(sam_forward_count=1),lambda v:v.update(visual_review_calls=1),
 lambda v:v.update(unknown_duration_attempts=1),lambda v:v.update(missing_component_before_forward_verified=False),
 lambda v:v['native_failure'].update(code='TimeoutError'),lambda v:v['native_failure'].update(message='other'),
 lambda v:v['gateway_intent'].update(state='STARTED'),lambda v:v['gateway_intent'].update(wall_seconds=None),
 lambda v:v['gateway_intent'].update(wall_seconds=float('nan')),lambda v:v['gateway_intent'].update(wall_seconds=-1),
 lambda v:v['pending_call']['parameters'].update(question='developer answer'),
 lambda v:v['native_job'].update(request_hash='changed'),lambda v:v['native_failure'].update(binding='other'),
 lambda v:v['native_failure'].update(resident_pid=456),lambda v:v['request'].update(request_id='different'),
])
def test_ambiguous_partial_changed_or_other_failures_cannot_be_reexecuted(mutate):
    value=failure();mutate(value)
    with pytest.raises(ValueError):check_failed_boundary_stage(value)


def domain_fixture():
    value=failure();value.update(feedback_sha256=payload_sha256({'feedback':1}),original_deadline_epoch=1000,
        trace_path='old/trace.json',lifecycle_total={'model_calls':7,'planner_calls':4,'operator_calls':17,
        'wall_seconds':394,'output_bytes':1000,'image_experiments':0})
    class Failure:
        def identity(self):return {'old':'frozen'}
        def read(self):return copy.deepcopy(value)
    class Feedback:
        def identity(self):return {'feedback':'frozen'}
        def read(self):return {'feedback':1},{}
        def probe_inputs(self):return fixture_inputs()
    prior=copy.deepcopy(value['lifecycle_total']);inc={'model_calls':2,'planner_calls':0,'operator_calls':2,
        'wall_seconds':3600,'output_bytes':67108864,'image_experiments':0}
    admission={'schema_version':'dataevolver.boundary_probe_recovery_admission.v1',
        'failed_authority_sha256':payload_sha256(Failure().identity()),'new_planner_decisions':0,
        'max_segmentation_calls':1,'max_visual_review_calls':1,'resource_deadline_epoch':1000,
        'budget':{'schema_version':'dataevolver.operator_development_budget.v1',
                  'limits':{k:prior[k]+inc[k] for k in prior},'prior_usage':prior,'resource_deadline_epoch':1000}}
    return Failure(),Feedback(),admission


def test_recovery_uses_original_decision_without_a_new_planner_and_refuses_modified_parameters():
    failure_authority,feedback,admission=domain_fixture()
    domain=BoundaryProbeRecoveryDomain(feedback_authority=feedback,failure_authority=failure_authority,
        observation_operator_id='data.probe_instance_boundaries',admission=admission)
    decision,trace,counters=domain.recover_preflight_decision(SimpleNamespace(data={}))
    assert counters=={'planner_calls':1} and trace=='old/trace.json'
    assert domain.compile_observation(decision,None)[1]==query()
    assert domain.recover_preflight_decision(SimpleNamespace(data={'preflight_proposal_recovered':True})) is None
    changed=copy.deepcopy(decision);changed['observation']['parameters_json']='{}'
    with pytest.raises(ValueError):domain.compile_observation(changed,None)


@pytest.mark.parametrize('change',['refund','new-planner','deadline','extra-sam'])
def test_spent_failures_and_deadline_cannot_be_reset(change):
    f,feedback,a=domain_fixture()
    if change=='refund':a['budget']['prior_usage']['model_calls']-=1
    if change=='new-planner':a['new_planner_decisions']=1
    if change=='deadline':a['budget']['resource_deadline_epoch']+=1
    if change=='extra-sam':a['max_segmentation_calls']=2
    with pytest.raises(ValueError):BoundaryProbeRecoveryDomain(feedback_authority=feedback,failure_authority=f,
        observation_operator_id='data.probe_instance_boundaries',admission=a)
