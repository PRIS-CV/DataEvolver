"""Fixture replay of verified pre-submission failures; no real model claims."""

import pytest

from dataevolver.adapters.model_contracts import ModelReceipt
from dataevolver.dataset.contracts import payload_sha256
from dataevolver.harness.state_store import atomic_write_json
from dataevolver.harness.training_preflight_recovery import verified_preflight_proposal
from dataevolver.testing.test_runtime_rsi import make_runtime,action,decisions


def fixture_preflight(root):
    baseline=root/'rounds/r000/execute/trial.json';work=root/'experiments/e000/candidate/production'
    source={'sha256':'a'*64};candidate={'sha256':'b'*64}
    atomic_write_json(baseline,{'source':source,'candidate':candidate,'inner_work_dir':str(root/'baseline-work')})
    atomic_write_json(work/'production.json',{'failures':[{'error_code':'KeyError','error':"'status_path'"}]})
    trial=work.parent/'trial.json';atomic_write_json(trial,{'failure_kind':'incomplete_media_pipeline','inner_work_dir':str(work)})
    proposal=action('RUN_EXPERIMENT',layer='R',patch='effect_support')
    receipt=ModelReceipt('fixture-runtime','c'*64,'d'*64,'gpt-6.1-sol','fixture','local_python',proposal,{},'stop')
    atomic_write_json(root/'planner/model_receipts/fixture-runtime.receipt.json',
        {'receipt':receipt.to_dict(),'sha256':payload_sha256(receipt.to_dict())})
    atomic_write_json(root/'planner/decisions/p001/trace.json',{'decision':proposal,'outcome':'VALID_DECISION',
        'proposal_actor':'runtime_agent','context':{'source_sha256':source['sha256'],'current_trial':{'candidate':candidate}},
        'model_call':{'receipt':receipt.to_dict(),'receipt_hash':receipt.without_reuse_hash()}})
    return {'state':'ABSTAIN','terminal_reason':'RSI_AGENT_ABSTAINED','experiments_used':1,
        'data':{'rsi_history':[{'decision':{'decision':'ROLLBACK'},'proposal':proposal}],
                'experiment_candidate_stage_result':{'evidence_refs':[str(trial)]}}},work


def test_preflight_replay_proves_no_new_reservation_and_preserves_actual_decision(tmp_path):
    state,work=fixture_preflight(tmp_path)
    proof=verified_preflight_proposal(tmp_path,state)
    assert proof['new_model_reservations_before_failure']==0 and proof['decision_edited'] is False
    assert proof['decision']==state['data']['rsi_history'][0]['proposal']
    atomic_write_json(work/'pe_model_receipts/new.intent.json',{'fixture':'possibly submitted'})
    with pytest.raises(ValueError,match='reservation'):verified_preflight_proposal(tmp_path,state)


def test_unknown_native_job_cannot_use_preflight_replay(tmp_path):
    state,work=fixture_preflight(tmp_path)
    atomic_write_json(work/'native/pe/new.job.json',{'fixture':'possibly submitted'})
    with pytest.raises(ValueError,match='native model job'):verified_preflight_proposal(tmp_path,state)


def test_controller_replays_verified_proposal_without_another_planner_call(tmp_path):
    c,backend=make_runtime(tmp_path,[action('OBSERVE',tool='inspect'),action('COMPLETE')])
    prior=tmp_path/'prior-proposal.json';atomic_write_json(prior,{'fixture':'verified old proposal'})
    def recover(state):
        if state.data.get('rsi_observations') and not state.data.get('preflight_proposal_recovered'):
            return decisions()[1],str(prior)
        return None
    c.planner.domain.recover_preflight_decision=recover
    result=c.run()
    assert result.state.value=='COMPLETE' and result.planner_calls==2 and len(backend.calls)==2
    assert result.experiments_used==1
    assert any(e.event_type=='RSI_PREFLIGHT_PROPOSAL_RECOVERED' for e in c.store.events.read())
    assert str(prior) in c.capabilities.list_capabilities()[0].validation_evidence


def test_pe_binding_rejects_missing_status_metadata_before_pipeline(tmp_path):
    from dataevolver.harness.training_rsi_run import bind_resident_deployment
    media=tmp_path/'media.json';prompt=tmp_path/'prompt.json'
    atomic_write_json(media,{'gpu_uuids':['fixture-gpu0'],'pid':7})
    atomic_write_json(prompt,{'pid':7,'models':{}})
    with pytest.raises(ValueError,match='status_path'):
        bind_resident_deployment({'models':{'workers':{'pe':{'kind':'pe_i2i'}}}}, {},
            {'media_identity':str(media),'prompt_identity':str(prompt),'gpu_uuids':['fixture-gpu0']})


def test_resident_wait_deadline_does_not_expire_an_available_service(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import dataevolver.harness.adapters.training_supervision as module
    from dataevolver.adapters.model_gateway import ModelError
    operator=module.TrainingSupervisionOperator.__new__(module.TrainingSupervisionOperator)
    operator.custom_wait=None;operator.prompt_policy=SimpleNamespace(bindings={'training.refine_i2i':'pe'})
    operator.contract={'models':{'workers':{'pe':{'kind':'pe_i2i'}}},'production':{'resource_deadline_epoch':0}}
    sleeps=[];calls=[]
    monkeypatch.setattr(module,'pe_observation',lambda config:calls.append('ready') or {'fixture':'ready'})
    monkeypatch.setattr(module.time,'sleep',sleeps.append)
    operator._wait(SimpleNamespace(work_dir=str(tmp_path/'ready')),'training.refine_i2i')
    assert calls==['ready','ready'] and sleeps==[30]
    def busy(config):raise ModelError('RESOURCE_BUSY','fixture busy',retry_safe=True)
    monkeypatch.setattr(module,'pe_observation',busy)
    with pytest.raises(TimeoutError):operator._wait(SimpleNamespace(work_dir=str(tmp_path/'busy')),'training.refine_i2i')


def test_recovered_observation_consumes_original_decision_slot_without_inference(tmp_path):
    c,backend=make_runtime(tmp_path,[decisions()[1],action('COMPLETE')])
    proof=tmp_path/'prior-observe.json';atomic_write_json(proof,{'fixture':'original completion'})
    def recover(state):
        if not state.data.get('preflight_proposal_recovered'):
            return action('OBSERVE',tool='inspect'),str(proof),{'planner_calls':1}
        return None
    c.planner.domain.recover_preflight_decision=recover
    final=c.run()
    assert final.state.value=='COMPLETE' and final.planner_calls==3 and len(backend.calls)==2
    assert final.operator_calls==3 and final.experiments_used==1
    recovered=[e for e in c.store.events.read() if e.event_type=='RSI_PREFLIGHT_PROPOSAL_RECOVERED']
    assert len(recovered)==1 and c.replay().to_dict()==final.to_dict()
