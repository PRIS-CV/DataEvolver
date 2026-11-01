"""Engineering fixtures only; actual PE quality requires production receipts."""
import copy
import json
from pathlib import Path

import pytest

from test_training_pipeline import pipeline
from dataevolver.adapters.model_contracts import ModelProfile,ModelPolicy,ModelRequest
from dataevolver.adapters.model_gateway import ModelGateway,ModelError
from dataevolver.adapters import training_prompt_enhancement as pe
from dataevolver.dataset.contracts import file_sha256,payload_sha256,ArtifactRef
from dataevolver.dataset.supervision import load_dataset


def configure_pe(ops,tmp):
    profiles=tuple(ops.gateway.profiles.values())+(
        ModelProfile('pe-t2i','mock','fixture-pe-t2i','fixture-1',('json',),mock_output={
            'rewritten_prompt':'fixture refined scene','wh_ratio':'1:1'}),
        ModelProfile('pe-i2i','mock','fixture-pe-i2i','fixture-1',('json','vision'),mock_output={
            'rewritten_prompt':'fixture refined background','wh_ratio':'','ratio_follow':'<image1>'}))
    bindings={**ops.gateway.policy.bindings,'training.refine_t2i':'pe-t2i','training.refine_i2i':'pe-i2i'}
    ops.gateway=ModelGateway(profiles=profiles,policy=ModelPolicy(bindings,None,usage_mode='accounting'),
        receipt_root=tmp/'gateway-pe',artifact_store=ops.store)
    ops.recipe.update(refine_scene_prompt=True,refine_background_prompt=True)


def test_refinements_reach_both_actual_image_requests_and_portable_trace(pipeline):
    ops,task,waits,tmp=pipeline;configure_pe(ops,tmp)
    task.update(scene_refinement_request='fixture vague scene',background_refinement_request='fixture vague edit',
        refinement_request_origin='fixture_task_configuration')
    goal=copy.deepcopy(ops.config['planning']['goal']);row=ops.run_family(task,0)
    image=next(m for m in row['model_receipts'] if m['request']['role']=='training.generate')
    edit=next(m for m in row['model_receipts'] if m['request']['role']=='training.background')
    assert json.loads(image['request']['messages'][0]['content'])['prompt']=='fixture refined scene'
    assert json.loads(edit['request']['messages'][0]['content'])['prompt']=='fixture refined background'
    refiners=[m for m in row['model_receipts'] if m['purpose']=='prompt_refinement'];assert len(refiners)==2
    assert 'fixture vague scene' in refiners[0]['request']['messages'][0]['content']
    assert 'fixture vague edit' in refiners[1]['request']['messages'][0]['content']
    assert 'Fixed execution requirements' in refiners[1]['request']['messages'][0]['content']
    assert not refiners[0]['request']['artifacts']
    assert refiners[1]['request']['artifacts']==edit['request']['artifacts']
    assert ops.config['planning']['goal']==goal and set(row['input'])=={'background','subjects','instruction'}
    evidence=row['annotations']['background_completion']['prompt_refinement']
    value=json.loads(ops.store.resolve(ArtifactRef.from_dict(evidence)).read_text())
    assert value['pe_receipt_hash']==refiners[1]['receipt_hash'] and value['refined_prompt']=='fixture refined background'
    outputs=copy.deepcopy(ops.last_outputs);ops.export([row],tmp/'bundle',[outputs])
    assert load_dataset(tmp/'bundle',allow_fixture=True)[0][0]==row
    waits.clear();assert ops.run_family(task,0)==row and waits==[]
    assert ops.gateway.audit_usage()['model_calls']==7


def test_refinement_cannot_reference_absent_input_images(pipeline):
    from dataclasses import replace
    from dataevolver.dataset.training_pipeline import QualityRejected
    ops,task,_,tmp=pipeline;configure_pe(ops,tmp)
    profile=ops.gateway.profiles['pe-t2i']
    ops.gateway.profiles['pe-t2i']=replace(profile,mock_output={'rewritten_prompt':'use <image1>','wh_ratio':'1:1'})
    with pytest.raises(QualityRejected,match='not supplied'):ops.run_family(task,0)
    assert ops.gateway.audit_usage()['model_calls']==1  # No image generation after invalid PE output.


def test_pe_parser_uses_final_answer_not_thinking_and_rejects_partial_or_conflicting_ratio():
    raw='<think>not an execution prompt</think>{"rewritten_prompt":"remove it","wh_ratio":"","ratio_follow":"<image1>"}'
    assert pe.parse_refinement(raw,'pe_i2i')['rewritten_prompt']=='remove it'
    with pytest.raises(ValueError):pe.parse_refinement(raw[:-2],'pe_i2i')
    with pytest.raises(ValueError,match='exclusive'):
        pe.parse_refinement('{"rewritten_prompt":"edit","wh_ratio":"1:1","ratio_follow":"<image1>"}','pe_i2i')


def test_completed_pe_native_lookup_does_not_contact_or_load_resident(tmp_path,monkeypatch):
    config={'kind':'pe_t2i','deployment_env':'FIXTURE_PE','deployment_sha256':'a'*64,
        'file_hashes':{'fixture-model':'b'*64},'implementation_sha256':file_sha256(Path(pe.__file__)),
        'activation_reserve_mib':4096,'max_input_tokens':8192}
    profile=ModelProfile('pe','local_python',pe.PE_MODELS['pe_t2i'],'sha256:'+payload_sha256(config['file_hashes']),('json',))
    from dataevolver.dataset.contracts import ArtifactStore
    provider=pe.PromptEnhancementProvider(profile,config,ArtifactStore({'data':tmp_path/'data'}),'data',tmp_path/'native')
    request=ModelRequest('fixture-pe','training.refine_t2i',({'role':'user','content':'{}'},),pe.PE_T2I_SCHEMA)
    response={'model':profile.model,'output':{'rewritten_prompt':'fixture','wh_ratio':'1:1'},'usage':{},'finish_reason':'stop'}
    provider.root.mkdir();digest='c'*64
    provider.path(request).write_text(json.dumps({'binding':provider.binding(digest),'response':response,'sha256':payload_sha256(response)}))
    monkeypatch.setattr(pe,'pe_identity',lambda c:pytest.fail('completed request touched PE deployment'))
    assert provider.invoke(request,digest)==response
    changed={**config,'implementation_sha256':'0'*64}
    with pytest.raises(ValueError,match='implementation changed'):pe.validate_pe_worker(changed)


def test_auxiliary_gateway_keeps_legacy_contract_and_all_failed_costs(tmp_path):
    from dataevolver.adapters.model_gateway import RoutedModelGateway
    base=ModelProfile('base','mock','fixture-base','fixture-1',('json',),mock_output={'ok':True})
    extra=ModelProfile('pe','mock','fixture-pe','fixture-1',('json',),mock_output={'ok':True})
    policy=ModelPolicy({'production':'base'},None,usage_mode='accounting')
    schema={'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}
    old=ModelGateway(profiles=(base,),policy=policy,receipt_root=tmp_path/'model_receipts')
    request=ModelRequest('fixture-old','production',({'role':'user','content':'{}'},),schema)
    original=old.invoke(request);before={p.name:p.read_bytes() for p in old.root.glob('*.json')}
    expanded=ModelGateway(profiles=(base,extra),policy=policy,receipt_root=old.root)
    aux=ModelGateway(profiles=(base,extra),policy=ModelPolicy({'refinement':'pe'},None,usage_mode='accounting'),
        receipt_root=tmp_path/'pe_model_receipts')
    routed=RoutedModelGateway(expanded,aux)
    assert expanded.contract_hash()==old.contract_hash()
    assert routed.recover(request).without_reuse_hash()==original.without_reuse_hash()
    assert {p.name:p.read_bytes() for p in old.root.glob('*.json')}==before
    refined=ModelRequest('fixture-refinement','refinement',request.messages,schema)
    routed.invoke(refined)
    invalid=ModelRequest('fixture-failed-refinement','refinement',request.messages,{'type':'object','required':['missing_fixture_field']})
    with pytest.raises(ModelError):routed.invoke(invalid)
    usage=routed.audit_usage()
    assert usage['model_calls']==3 and len(usage['attempts'])==3
    assert any(row['request_id']=='fixture-failed-refinement' and row['state']=='INVALID_OUTPUT' for row in usage['attempts'])
    assert usage['monetary_cost'] is None
    with pytest.raises(ValueError,match='ambiguous'):RoutedModelGateway(expanded,expanded)


def test_controller_operator_uses_two_ledgers_and_replays_complete_refined_bundle(pipeline):
    from dataclasses import replace
    from test_training_pipeline import _batch
    from dataevolver.harness.adapters.training_supervision import TrainingSupervisionOperator
    from dataevolver.harness.models import StageStatus,EvidenceStatus
    old,call,waits,tmp=_batch(pipeline)
    config=copy.deepcopy(old.contract)
    config['models']['profiles'] += [
        ModelProfile('pe-t2i','mock','fixture-pe-t2i','fixture-1',('json',),mock_output={
            'rewritten_prompt':'fixture refined scene','wh_ratio':'1:1'}).to_dict(),
        ModelProfile('pe-i2i','mock','fixture-pe-i2i','fixture-1',('json','vision'),mock_output={
            'rewritten_prompt':'fixture refined background','wh_ratio':'','ratio_follow':'<image1>'}).to_dict()]
    config['models']['prompt_policy']=ModelPolicy({'training.refine_t2i':'pe-t2i','training.refine_i2i':'pe-i2i'},
        None,usage_mode='accounting').to_dict()
    config['production']['recipe'].update(refine_scene_prompt=True,refine_background_prompt=True)
    op=TrainingSupervisionOperator(allowed_write_root=tmp/'campaign',contract=config,
        roots={k:str(v) for k,v in old.store.roots.items()},wait_for_role=waits.append)
    call=replace(call,parameters=config)
    receipt=op.execute(call)
    assert receipt.status is StageStatus.SUCCESS
    assert op.verify_effect(call,receipt).status is EvidenceStatus.VERIFIED
    assert len(list((tmp/'campaign/execute/model_receipts').glob('*.intent.json')))==5
    assert len(list((tmp/'campaign/execute/pe_model_receipts').glob('*.intent.json')))==2
    assert op.gateway(call).audit_usage()['model_calls']==7
    waits.clear();assert op.execute(call).reused and not waits


def test_loaded_device_map_uses_materialized_parameters_without_optional_hf_attribute():
    from types import SimpleNamespace
    class Loaded:
        def parameters(self):return iter([SimpleNamespace(device='cuda:0')])
    assert pe.loaded_device_map(Loaded())=={'':'cuda:0'}
    class NotLoaded:
        def parameters(self):return iter([SimpleNamespace(device='meta')])
    with pytest.raises(ValueError,match='not materialized'):pe.loaded_device_map(NotLoaded())


def test_component_loading_does_not_reserve_already_resident_weights_twice():
    config={'models':{'pe_t2i':{'total_weight_bytes':18819721168},'pe_i2i':{'total_weight_bytes':18819721168}},
        'placement':{'pe_t2i':'cuda:0','pe_i2i':'cuda:0'}}
    assert pe.remaining_load_mib(config,{})==46136
    assert pe.remaining_load_mib(config,{'pe_t2i':object()})==28188
    config['placement']['pe_t2i']='cpu'
    assert pe.remaining_load_mib(config,{})==28188


def test_explicit_thinking_mode_is_part_of_the_costed_request_and_trace(pipeline):
    ops,task,_,tmp=pipeline;configure_pe(ops,tmp)
    ops.recipe.update(pe_t2i_enable_thinking=False,pe_i2i_enable_thinking=True)
    row=ops.run_family(task,0)
    requests={m['request']['role']:json.loads(m['request']['messages'][0]['content'])
        for m in row['model_receipts'] if m['purpose']=='prompt_refinement'}
    assert requests['training.refine_t2i']['enable_thinking'] is False
    assert requests['training.refine_i2i']['enable_thinking'] is True
    ref=ArtifactRef.from_dict(row['annotations']['background_completion']['prompt_refinement'])
    assert json.loads(ops.store.resolve(ref).read_text())['enable_thinking'] is True
