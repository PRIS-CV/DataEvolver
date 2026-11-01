"""Negative dispatch and reservation fixtures for exact-decision handoff."""
import json
from types import SimpleNamespace
import pytest
from dataevolver.adapters.model_contracts import ModelReceipt
from dataevolver.dataset.contracts import payload_sha256,file_sha256
from dataevolver.harness.state_store import atomic_write_json
from dataevolver.harness import training_protocol_recovery as module


def fixture(tmp_path,monkeypatch):
    cfg=tmp_path/'config';campaign=cfg/'campaigns/fixture';work=campaign/'rounds/r000/execute/production'
    prior=tmp_path/'prior';requestsha='c'*64
    source={'sha256':'a'*64};candidate={'sha256':'b'*64}
    obs=tmp_path/'observed.json';atomic_write_json(obs,{'source':source,'candidate':candidate})
    decision={'action':'OBSERVE','observation':{'tool':'preflight_reconstruction',
        'parameters_json':json.dumps({'observation_id':'observation','context_padding_fraction':.1})}}
    receipt=ModelReceipt('fixture-runtime','c'*64,'d'*64,'gpt-6.1-sol','fixture','local_python',decision,{},'stop')
    atomic_write_json(campaign/'planner/model_receipts/fixture-runtime.receipt.json',
        {'receipt':receipt.to_dict(),'sha256':payload_sha256(receipt.to_dict())})
    atomic_write_json(campaign/'planner/model_receipts/fixture-runtime.intent.json',{'request_hash':receipt.request_hash})
    inherited=campaign/'planner/model_receipts/ancestor.intent.json'
    atomic_write_json(inherited,{'fixture':'exact inherited intent'})
    domain={'prior_runtime':{'config_dir':str(prior),'planner_calls':30},
        'surface_capability':{'request_sha256':requestsha,'file_hashes':{str(inherited):file_sha256(inherited)}},
        'seed':{'file_hashes':{}},'prior_observations':[{'path':str(obs),'sha256':file_sha256(obs),'item':{'call_id':'observation'}}]}
    goal=SimpleNamespace(planner={'runtime':{'domain':domain}},campaign_id='fixture',budgets=SimpleNamespace(max_planner_calls=4))
    atomic_write_json(cfg/'goal.json',{'fixture':'goal'});atomic_write_json(cfg/'frozen.json',{'fixture':'frozen'})
    atomic_write_json(campaign/'state.json',{'state':'ABSTAIN','terminal_reason':'INVALID_RSI_OBSERVATION',
        'planner_calls':1,'operator_calls':1,'experiments_used':0,'data':{'rsi_pending_decision':decision}})
    atomic_write_json(campaign/'rounds/r000/execute/trial.json',{'inner_work_dir':str(work)})
    atomic_write_json(work/'production.json',{'fixture':'cached'})
    atomic_write_json(campaign/'planner/decisions/p000/trace.json',{'outcome':'VALID_DECISION','decision':decision,
        'proposal_actor':'runtime_agent','context':{'current_trial':{'source':source,'candidate':candidate}},
        'model_call':{'receipt':receipt.to_dict(),'receipt_hash':receipt.without_reuse_hash()}})
    monkeypatch.setattr(module.GoalSpec,'from_json',lambda raw:goal)
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,
        stdout=json.dumps({'frozen_contract_valid':True,'replay_matches_snapshot':True})))
    return cfg,campaign,prior,requestsha


def test_exact_recovery_counts_old_slot_and_refuses_unknown_or_submitted_effects(tmp_path,monkeypatch):
    cfg,campaign,prior,requestsha=fixture(tmp_path,monkeypatch)
    def verify():return module.verified_observation_handoff(cfg,prior_config=prior,capability_request_sha256=requestsha)
    value=verify()
    assert value['prior_runtime_planner_calls']==30 and value['remaining_decisions']==3
    assert value['decision_slots_consumed']==1 and value['new_model_calls_for_recovery']==0
    extra=campaign/'planner/model_receipts/new.intent.json';atomic_write_json(extra,{'state':'STARTED'})
    with pytest.raises(ValueError,match='unexpected'):verify()
    extra.unlink()
    job=campaign/'observations/p000/native/job.json';atomic_write_json(job,{'state':'STARTED'})
    with pytest.raises(ValueError,match='already have been dispatched'):verify()


def schema_fixture(tmp_path,monkeypatch):
    from copy import deepcopy
    from dataevolver.adapters.model_contracts import ModelProfile,ModelRequest
    from dataevolver.dataset.contracts import ArtifactRef
    from dataevolver.dataset.surface_preflight import TYPED_SURFACE_SCHEMA
    cfg,campaign,prior,requestsha=fixture(tmp_path,monkeypatch)
    # Use the same receipt construction as the genuine Planner, followed by a
    # separate pre-inference schema rejection for its registered observation.
    profile=ModelProfile('fixture-planner','local_python','gpt-6.1-sol','fixture',('json','vision'))
    artifact=ArtifactRef('dataset','source.png','a'*64,'image/png').to_dict()
    task={'selected_subject_ids':['s1']};question='Inspect this source using typed uncertainty'
    trial={'source':{'sha256':'a'*64},'candidate':{'sha256':'b'*64},
        'media':[{'label':name,'artifact':artifact} for name in ['source','candidate','raw','support']]}
    trialpath=campaign/'continuation/rejected-view.json';atomic_write_json(trialpath,trial)
    decision={'action':'OBSERVE','observation':{'tool':'inspect_surfaces','parameters_json':json.dumps({'question':question})}}
    receipt=ModelReceipt('fixture-second','e'*64,'d'*64,'gpt-6.1-sol','fixture','local_python',decision,{},'stop')
    atomic_write_json(campaign/'planner/model_receipts/fixture-second.receipt.json',
        {'receipt':receipt.to_dict(),'sha256':payload_sha256(receipt.to_dict())})
    atomic_write_json(campaign/'planner/model_receipts/fixture-second.intent.json',{'request_hash':receipt.request_hash})
    atomic_write_json(campaign/'planner/decisions/p001/trace.json',{'outcome':'VALID_DECISION','decision':decision,
        'proposal_actor':'runtime_agent','context':{'current_trial':trial},
        'model_call':{'receipt':receipt.to_dict(),'receipt_hash':receipt.without_reuse_hash()}})
    pending={'operator_id':'data.training_rsi_observe','call_id':'fixture:p001:observe',
        'parameters':{'tool':'inspect_surfaces','parameters':{'question':question},
            'trial_path':str(trialpath),'trial_sha256':payload_sha256(trial)}}
    atomic_write_json(campaign/'state.json',{'state':'PROPOSE','experiments_used':0,'planner_calls':2,'operator_calls':2,
        'data':{'rsi_pending_decision':decision,'pending_call':pending}})
    run=cfg; (run/'exit_code').write_text('1')
    prompt={'question':question,'task':task,'media_order':[x['label'] for x in trial['media']],
        'typed_contract':{'existing_authorized_support':artifact}}
    content=json.dumps(prompt);messages=[{'role':'user','content':content}]
    old=deepcopy(TYPED_SURFACE_SCHEMA);old['properties']['schema_version']={'const':'dataevolver.surface_observation.v2'}
    old['properties']['uncertain_regions']['items']['properties']['subject_ids']['uniqueItems']=True
    refs=tuple(ArtifactRef.from_dict(x['artifact']) for x in trial['media'])
    binding=payload_sha256({'messages':messages,'schema':old,'artifacts':[r.to_dict() for r in refs]})
    request=ModelRequest('rsi-'+binding[:28],'agent.runtime_rsi',tuple(messages),old,artifacts=refs,
        max_tokens=4096,timeout_seconds=900,max_attempts=1,seed=0)
    expected=payload_sha256({'request':request.to_dict(),'profile_hash':payload_sha256({'profile':profile.to_dict(),'model':profile.model,'endpoint':''})})
    native=campaign/'planner/native'/request.request_id;native.mkdir(parents=True)
    (native/'prompt.txt').write_text('USER:\n'+content)
    atomic_write_json(native/'output-schema.json',old)
    atomic_write_json(native/'native.intent.json',{'request_hash':expected,'profile_hash':profile.compute_hash()})
    atomic_write_json(native/'native.failure.json',{'request_hash':expected,'error_code':'PLANNER_FAILED','wall_seconds':2})
    atomic_write_json(campaign/'planner/model_receipts'/ (request.request_id+'.intent.json'),
        {'state':'FAILED','error_code':'PLANNER_FAILED','wall_seconds':3,'request_hash':expected})
    error={'status':400,'error':{'code':'invalid_json_schema',
        'message':'uncertain_regions.items.properties.subject_ids.uniqueItems rejected'}}
    (native/'events.jsonl').write_text('\n'.join(json.dumps(x) for x in [
        {'type':'thread.started'},{'type':'turn.started'},{'type':'error','message':json.dumps(error)},
        {'type':'turn.failed'}]))
    # Existing ancestor receipts are hash-frozen before the additional failure.
    known={str(p):file_sha256(p) for p in (campaign/'planner/model_receipts').glob('*.json')
        if not p.name.startswith(request.request_id)}
    domain={'prior_runtime':{'config_dir':str(prior),'planner_calls':30},'surface_capability':{
        'request_sha256':requestsha,'file_hashes':known},'seed':{'file_hashes':{}},
        'production_contract':{'production':{'tasks':[task]}},
        'protocol_observation_recovery':{'trace_path':str(campaign/'planner/decisions/p000/trace.json')}}
    goal=SimpleNamespace(planner={'runtime':{'domain':domain,'planner_profile':profile.to_dict(),'planner_timeout_seconds':900}},
        campaign_id='fixture',budgets=SimpleNamespace(max_planner_calls=4))
    monkeypatch.setattr(module.GoalSpec,'from_json',lambda raw:goal)
    return cfg,campaign,prior,requestsha,native


def test_schema_repair_handoff_preserves_decisions_slots_and_failed_cost(tmp_path,monkeypatch):
    cfg,campaign,prior,requestsha,native=schema_fixture(tmp_path,monkeypatch)
    proof=module.verified_schema_observation_handoff(cfg,prior_config=prior,capability_request_sha256=requestsha)
    assert proof['decision_slots_consumed']==2 and proof['remaining_decisions']==2
    assert proof['new_model_calls_for_recovery']==0 and proof['failed_gateway_wall_seconds']==3
    assert proof['old_schema_sha256']!=proof['new_schema_sha256']
    assert not proof['recovered_decision_edited'] and len(proof['recovered_decision_trace_paths'])==2
    (native/'events.jsonl').write_text(json.dumps({'type':'item.completed','item':{'text':'possible inference'}}))
    with pytest.raises(ValueError,match='inference output|ambiguous'):module.verified_schema_observation_handoff(
        cfg,prior_config=prior,capability_request_sha256=requestsha)


@pytest.mark.parametrize('change',['completed','wrong_schema_error','unknown_reservation','prompt_changed','semantic_change','image_job'])
def test_schema_repair_refuses_unknown_or_non_transport_changes(tmp_path,monkeypatch,change):
    cfg,campaign,prior,requestsha,native=schema_fixture(tmp_path,monkeypatch)
    if change=='completed':atomic_write_json(native/'final-response.json',{'status':'unknown'})
    elif change=='wrong_schema_error':
        (native/'events.jsonl').write_text('\n'.join(json.dumps(x) for x in [
            {'type':'error','message':json.dumps({'status':503,'error':{'code':'unavailable'}})}, {'type':'turn.failed'}]))
    elif change=='unknown_reservation':atomic_write_json(campaign/'planner/model_receipts/orphan.intent.json',{'state':'STARTED'})
    elif change=='prompt_changed':(native/'prompt.txt').write_text('USER:\n{}')
    elif change=='semantic_change':
        schema=json.loads((native/'output-schema.json').read_text());schema['properties']['status']['enum'].append('accepted')
        atomic_write_json(native/'output-schema.json',schema)
    else:atomic_write_json(campaign/'experiments/e000/native/job.json',{'state':'STARTED'})
    with pytest.raises((ValueError,KeyError)):
        module.verified_schema_observation_handoff(cfg,prior_config=prior,capability_request_sha256=requestsha)


def test_observation_recovery_retains_preceding_verified_cpu_recheck(tmp_path,monkeypatch):
    cfg,campaign,prior,requestsha=fixture(tmp_path,monkeypatch)
    path=campaign/'observations/p000/observation.json'
    atomic_write_json(path,{'source':{'sha256':'a'*64},'candidate':{'sha256':'b'*64},'tool':'recheck_surface_support'})
    atomic_write_json(path.parent/'runtime-effect.json',{'files':{str(path):file_sha256(path)}})
    state=json.loads((campaign/'state.json').read_text());state.update(planner_calls=2,operator_calls=2)
    state['data']['rsi_observations']=[{'call_id':'verified-cpu','verification':{'status':'VERIFIED'},
        'stage_result':{'evidence_refs':[str(path)]}}]
    state['data']['rsi_pending_decision']['observation']['parameters_json']=json.dumps({'observation_id':'verified-cpu','context_padding_fraction':.03})
    atomic_write_json(campaign/'state.json',state)
    trace=json.loads((campaign/'planner/decisions/p000/trace.json').read_text())
    receipt=ModelReceipt('fixture-second','f'*64,'d'*64,'gpt-6.1-sol','fixture','local_python',state['data']['rsi_pending_decision'],{},'stop')
    trace.update(decision=receipt.output,model_call={'receipt':receipt.to_dict(),'receipt_hash':receipt.without_reuse_hash()})
    atomic_write_json(campaign/'planner/decisions/p001/trace.json',trace)
    atomic_write_json(campaign/'planner/model_receipts/fixture-second.receipt.json',{'receipt':receipt.to_dict(),'sha256':payload_sha256(receipt.to_dict())})
    atomic_write_json(campaign/'planner/model_receipts/fixture-second.intent.json',{'request_hash':receipt.request_hash})
    proof=module.verified_observation_handoff(cfg,prior_config=prior,capability_request_sha256=requestsha)
    assert proof['decision_slots_consumed']==2 and proof['remaining_decisions']==2
    assert proof['decision']['observation']['parameters_json']==state['data']['rsi_pending_decision']['observation']['parameters_json']
    assert len(proof['registered_observations'])==1 and len(proof['recovered_decision_trace_paths'])==2
    assert proof['new_model_calls_for_recovery']==0
