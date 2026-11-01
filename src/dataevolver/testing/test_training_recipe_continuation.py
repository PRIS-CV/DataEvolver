"""Engineering fixtures for authority, evidence and compilation; no quality claims."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataevolver.harness.training_recipe_space import apply_recipe_patch,validate_recipe_patch
from dataevolver.harness.training_continuation import verified_rejected_continuation,bind_rejected_view
from dataevolver.harness.state_store import atomic_write_json
from dataevolver.dataset.contracts import file_sha256
from dataevolver.testing.test_training_rsi_domain import fixture_domain,observe,proposal
from dataevolver.testing.test_training_workflow import pair
from dataevolver.harness.training_workflow import verify_runtime_pair
from dataevolver.harness.adapters.training_rsi import inspect_committed_requests


def contract():
    return {'planning':{'goal':{'user_constraint':'preserve nonselected subjects'}},
        'models':{'fixed':'fixture weights'},'production':{'recipe':{
            'true_cfg_scale':2,'background_instruction':'Original fixed requirements.',
            'background_conditioning':'masked_region','background_conditioning_shape':'bounding_rectangle'},
            'tasks':[{'selected_subject_ids':['s1'],'seed':7,'selected_edge_margin_pixels':8,
                      'scene_prompt':'original scene'}]}}


def test_recipe_patch_changes_only_declared_execution_fields():
    original=contract();saved=copy.deepcopy(original)
    new=apply_recipe_patch(original,{'recipe':{'background_true_cfg_scale':3,
        'background_conditioning_shape':'region','background_addendum':'fixture detail'}})
    assert original==saved
    assert new['planning']==original['planning'] and new['models']==original['models']
    task=new['production']['tasks'][0]
    assert task['seed']==7 and task['scene_prompt']=='original scene' and task['background_true_cfg_scale']==3
    assert new['production']['recipe']['true_cfg_scale']==2
    assert new['production']['recipe']['background_instruction'].startswith('Original fixed requirements.')


def test_source_image_switch_canonicalizes_only_unused_mask_shape():
    original=contract()
    updated=apply_recipe_patch(original,{'recipe':{'background_conditioning':'source_image'}})
    assert updated['production']['recipe']['background_conditioning']=='source_image'
    assert updated['production']['recipe']['background_conditioning_shape']=='region'
    assert updated['production']['tasks']==original['production']['tasks']
    assert updated['production']['recipe']['background_instruction']==original['production']['recipe']['background_instruction']
    assert original['production']['recipe']['background_conditioning_shape']=='bounding_rectangle'


def test_request_inspection_exposes_exact_downstream_prompt_without_calls(tmp_path):
    from dataevolver.adapters.model_contracts import ModelReceipt,ModelRequest
    request=ModelRequest('fixture-background','training.background',
        ({'role':'user','content':'{"prompt":"actual PE result, fixture only"}'},),{'type':'object'},seed=7)
    receipt=ModelReceipt(request.request_id,'a'*64,'b'*64,'fixture-model','fixture-revision',
        'local_python',{'fixture':'actual output'},{'wall_seconds':1},'stop')
    row={'request':request.to_dict(),'receipt':receipt.to_dict(),'receipt_hash':receipt.without_reuse_hash()}
    atomic_write_json(tmp_path/'production.json',{'fixture':True})
    family=tmp_path/'families/fixture-a0.json';atomic_write_json(family,{'model_requests':[row]})
    trial={'inner_work_dir':str(tmp_path),'source_family':'fixture-a0',
        'production_sha256':file_sha256(tmp_path/'production.json')}
    old=file_sha256(family);result=inspect_committed_requests(trial)
    assert result['requests'][0]['request']['messages']==request.to_dict()['messages']
    assert result['new_model_requests']==0 and file_sha256(family)==old
    value=json.loads(family.read_text());value['model_requests'][0]['receipt_hash']='changed';atomic_write_json(family,value)
    with pytest.raises(ValueError,match='identity changed'):inspect_committed_requests(trial)


def test_wording_patch_needs_actual_request_inspection_not_only_image_evidence(tmp_path):
    domain,state,trial=fixture_domain(tmp_path)
    domain.contract['production']['recipe']=contract()['production']['recipe'];observe(domain,state,trial)
    p=proposal();p['patch']={'kind':'background_recipe','parameters_json':json.dumps({'recipe':{'background_addendum':'fixture extra wording'}})}
    with pytest.raises(ValueError,match='actual PE'):domain.compile_intervention(p,state)
    path=Path(state.data['rsi_observations'][0]['stage_result']['evidence_refs'][0])
    value=json.loads(path.read_text());value['tool']='inspect_requests';atomic_write_json(path,value)
    assert domain.compile_intervention(p,state).candidate_delta['parameters']['recipe']['background_addendum']=='fixture extra wording'
    _,params=domain.compile_observation({'observation':{'tool':'inspect_requests','parameters_json':'{}'}},state)
    assert params['trial_sha256']


@pytest.mark.parametrize('values',[{'seed':9},{'model':'other'},{'quality_threshold':0},
    {'background_true_cfg_scale':float('nan')},{'mask_dilation_pixels':True},
    {'selected_edge_margin_pixels':9},{'background_feather_pixels':33},
    {'background_conditioning':'unknown'},{'background_addendum':''}])
def test_protected_and_unbounded_recipe_changes_rejected(values):
    with pytest.raises(ValueError):validate_recipe_patch({'recipe':values},['s1'])


def test_runtime_selects_recipe_with_current_evidence_and_prevents_duplicate(tmp_path):
    domain,state,trial=fixture_domain(tmp_path);domain.contract['production']['recipe']=contract()['production']['recipe']
    p=proposal();p['patch']={'kind':'background_recipe','parameters_json':json.dumps({'recipe':{'background_true_cfg_scale':3}})}
    with pytest.raises(ValueError,match='registered probe'):domain.compile_intervention(p,state)
    observe(domain,state,trial);result=domain.compile_intervention(p,state)
    assert result.candidate_delta['kind']=='background_recipe'
    assert domain.context(state)['patch_contract']['background_recipe']['compilation_base']
    domain.value['prior_runtime']={'comparisons':[{'intervention':{'candidate_delta':result.candidate_delta}}]}
    with pytest.raises(ValueError,match='completed recipe'):domain.compile_intervention(p,state)


def predecessor(tmp_path):
    root=tmp_path/'old';work=root/'experiments/e000/candidate';path=work/'trial.json'
    trial={'scope':'candidate-000','qualified':False,'source':{'sha256':'a'*64},'candidate':{'sha256':'b'*64}}
    atomic_write_json(path,trial)
    atomic_write_json(work/'runtime-effect.json',{'files':{str(path):file_sha256(path)}})
    goal=SimpleNamespace(objective={'source_family':'fixture'},
        planner={'runtime':{'domain':{'prior_runtime':{'planner_calls':9}}}},
        budgets=SimpleNamespace(to_dict=lambda:{'max_planner_calls':1}))
    state={'state':'ABSTAIN','terminal_reason':'BUDGET_EXHAUSTED','planner_calls':1,'experiments_used':1,
        'data':{'pending_call':None,'rsi_history':[{'decision':{'decision':'ROLLBACK'}}],
            'experiment_candidate_stage_result':{'evidence_refs':[str(path)]}}}
    return root,state,goal,path


def test_continuation_retains_usage_and_rejected_media_without_replaying_effect(tmp_path):
    root,state,goal,path=predecessor(tmp_path)
    proof=verified_rejected_continuation(root,state,goal,'fixture')
    assert proof['planner_calls']==10 and proof['experiments_used']==1
    view,scope=bind_rejected_view(proof,tmp_path/'new',{'candidate-000':{'dataset':'old-read-only'}})
    assert json.loads(Path(view['path']).read_text())['candidate']==json.loads(path.read_text())['candidate']
    assert scope['dataset']=='old-read-only' and view['original_sha256']==file_sha256(path)
    atomic_write_json(path,{'qualified':False,'failure_kind':'incomplete_media_pipeline'})
    with pytest.raises(ValueError,match='unknown/incomplete'):verified_rejected_continuation(root,state,goal,'fixture')


def test_continuation_rejects_pending_and_tampered_effects(tmp_path):
    root,state,goal,path=predecessor(tmp_path)
    state['data']['pending_call']={'possible':'submitted'}
    with pytest.raises(ValueError,match='reconciled'):verified_rejected_continuation(root,state,goal,'fixture')
    state['data']['pending_call']=None
    value=json.loads(path.read_text());value['unexpected']='edit';atomic_write_json(path,value)
    with pytest.raises(ValueError,match='evidence changed'):verified_rejected_continuation(root,state,goal,'fixture')


def test_terminal_controller_does_not_hide_orphan_model_reservation(tmp_path):
    root,state,goal,path=predecessor(tmp_path)
    intent=root/'observations/orphan/model_receipts/request.intent.json'
    atomic_write_json(intent,{'state':'STARTED','request_hash':'fixture'})
    with pytest.raises(ValueError,match='unreconciled'):verified_rejected_continuation(root,state,goal,'fixture')
    atomic_write_json(intent,{'state':'AMBIGUOUS','request_hash':'fixture','wall_seconds':3})
    with pytest.raises(ValueError,match='unreconciled'):verified_rejected_continuation(root,state,goal,'fixture')
    atomic_write_json(intent,{'state':'INVALID_OUTPUT','request_hash':'fixture','wall_seconds':3})
    assert verified_rejected_continuation(root,state,goal,'fixture')['planner_calls']==10


def test_knowledge_requires_real_payload_change_not_new_request_id(tmp_path):
    old,new,history=pair(tmp_path)
    new['media'][5]=old['media'][5]
    assert verify_runtime_pair(old,new,history)['verified']
    path=Path(new['inner_work_dir'])/'families/fixture-a0.json';value=json.loads(path.read_text())
    value['model_requests'][-1]['request'].pop('condition')
    value['model_requests'][-1]['request']['request_id']='another-attempt-background'
    atomic_write_json(path,value)
    with pytest.raises(ValueError,match='payload'):verify_runtime_pair(old,new,history)
