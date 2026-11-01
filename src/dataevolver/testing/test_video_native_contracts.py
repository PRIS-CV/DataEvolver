"""CPU fixtures for video pricing, bounded forward counters and Review consistency."""
import contextlib
import copy
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from dataevolver.adapters.model_contracts import ModelProfile,ModelPolicy,ModelRequest
from dataevolver.adapters.model_gateway import ModelGateway,ModelError
from dataevolver.adapters.training_media_worker import TrainingMediaProvider
from dataevolver.dataset.contracts import ArtifactRef,ArtifactStore,canonical_json,file_sha256
from dataevolver.harness.attempt_limits import ModelAttemptLimits

SUBJECTS=[{'subject_id':n,'target_instance_id':n+'-1','description':n} for n in ('cube','sphere')]

def fake_av(monkeypatch,count=4):
    class Container:
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def decode(self,*,video):return iter([object() for _ in range(count)])
    monkeypatch.setitem(sys.modules,'av',SimpleNamespace(open=lambda _:Container()))

def video_request(tmp_path,*,request_id='video',subjects=SUBJECTS,extra=None):
    root=tmp_path/'data';root.mkdir(exist_ok=True);(root/'video.mp4').write_bytes(b'CPU fixture encoded bytes')
    store=ArtifactStore({'data':root});ref=ArtifactRef('data','video.mp4',file_sha256(root/'video.mp4'),'video/mp4')
    q=ModelRequest(request_id,'visual.segment.video',({'role':'user','content':canonical_json({'subjects':subjects,**(extra or {})})},),
        {'type':'object'},(ref,))
    return store,q

def limits(quota=10,roots=()):
    return ModelAttemptLimits({'sam':quota},{'visual.segment.image':'sam','visual.segment.video':'sam'},
        ledger_roots=tuple(map(str,roots)),unit_rules={'visual.segment.image':'image_subject','visual.segment.video':'video_frame_subject'})

def test_video_reservation_uses_decoded_frames_ignoring_claimed_header_count(tmp_path,monkeypatch):
    fake_av(monkeypatch);store,q=video_request(tmp_path,extra={'declared_frames':1})
    assert limits().units(q,artifact_store=store)==8
    with pytest.raises(ValueError,match='verified artifact store'):limits().units(q)
    wrong=ModelAttemptLimits({'sam':10},{q.role:'sam'},unit_rules={q.role:'image_subject'})
    with pytest.raises(ValueError,match='cannot price video'):wrong.units(q,artifact_store=store)

@pytest.mark.parametrize('count',[0,257])
def test_undecodable_or_unbounded_video_cannot_obtain_units(tmp_path,monkeypatch,count):
    fake_av(monkeypatch,count);store,q=video_request(tmp_path)
    with pytest.raises(ValueError):limits().units(q,artifact_store=store)

def test_video_source_hash_and_subject_uniqueness_checked_before_reservation(tmp_path,monkeypatch):
    fake_av(monkeypatch);store,q=video_request(tmp_path,subjects=[SUBJECTS[0],SUBJECTS[0]])
    with pytest.raises(ValueError,match='distinct'):limits().units(q,artifact_store=store)
    store,q=video_request(tmp_path);store.roots['data'].joinpath('video.mp4').write_bytes(b'changed')
    with pytest.raises(ValueError):limits().units(q,artifact_store=store)

def test_old_image_units_remain_charged_and_cached_video_recovery_is_free(tmp_path,monkeypatch):
    fake_av(monkeypatch);store,q=video_request(tmp_path);old=tmp_path/'old';old.mkdir()
    (old/'old.intent.json').write_text(json.dumps({'role':'visual.segment.image','attempts':1,'started_at':time.time(),
        'native_units_per_attempt':2}))
    profile=ModelProfile('sam','mock','fixture-sam','1',('json','vision'),mock_output={})
    gateway=ModelGateway(profiles=(profile,),policy=ModelPolicy({q.role:'sam'},2),receipt_root=tmp_path/'gateway',
        artifact_store=store,attempt_limits=limits(10,[old]))
    result=gateway.invoke(q);assert not result.reused
    assert gateway.audit_usage()['native_units_reserved']==8
    monkeypatch.setitem(sys.modules,'av',SimpleNamespace(open=lambda _:pytest.fail('cache decoded video again')))
    assert gateway.invoke(q).reused and gateway.audit_usage()['model_calls']==1
    fake_av(monkeypatch);_,next_q=video_request(tmp_path,request_id='next')
    with pytest.raises(ModelError) as error:gateway.invoke(next_q)
    assert error.value.code=='BUDGET_EXHAUSTED'
    assert not (gateway.root/'next.intent.json').exists()
    assert json.loads((old/'old.intent.json').read_text())['native_units_per_attempt']==2

def test_native_video_audit_keeps_frame_forwards_and_subject_units_distinct(tmp_path,monkeypatch):
    fake_av(monkeypatch);store,q=video_request(tmp_path)
    profile=ModelProfile('sam','mock','fixture-sam','1',('json','vision'),mock_output={})
    gateway=ModelGateway(profiles=(profile,),policy=ModelPolicy({q.role:'sam'},1),receipt_root=tmp_path/'gateway',
        artifact_store=store,attempt_limits=limits())
    gateway.invoke(q)
    provider=object.__new__(TrainingMediaProvider)
    provider.root=tmp_path/'native';provider.root.mkdir();provider.profile=profile;provider.config={'fixture':True}
    provider.dataset_store='data';provider.store=store
    gateway.factories['sam']=lambda:provider
    path=provider.root/'video.native.sam-usage.json'
    value={'binding':provider.binding(gateway._prepare(q)[4]),'gateway_requests':1,'input_videos':1,
        'decoded_frames':4,'subjects':2,'frame_forward_attempts':4,'frame_subject_units_attempted':8}
    path.write_text(json.dumps(value));audit=gateway.audit_usage()
    assert audit['sam_video_frame_forwards_observed']==4 and audit['sam_video_frame_subject_units_observed']==8
    assert audit['sam_image_subject_forwards_observed']==0 and audit['sam_requests_without_native_accounting']==0
    value['frame_subject_units_attempted']=7;path.write_text(json.dumps(value))
    with pytest.raises(ModelError) as error:gateway.audit_usage()
    assert error.value.code=='INVALID_RECEIPT'

def sam_fixture(tmp_path,monkeypatch,*,extra_frame=False,empty_mask=False):
    from dataevolver.adapters import local_visual_worker
    frames=[Image.new('RGB',(16,16),'red') for _ in range(4)]
    monkeypatch.setattr(local_visual_worker,'_video_frames',lambda *a,**k:frames)
    class Param:
        dtype='torch.float32'
        def numel(self):return 1
        def is_floating_point(self):return True
    class Model:
        def __init__(self):self.calls=[]
        def parameters(self):return iter([Param()])
        def to(self,*_):return self
        def eval(self):return self
        def forward(self,*,frame_idx,**kwargs):self.calls.append(frame_idx);return SimpleNamespace(frame_idx=frame_idx)
        def propagate_in_video_iterator(self,session,*,max_frame_num_to_track):
            assert max_frame_num_to_track==4
            for i in range(4+int(extra_frame)):yield self.forward(frame_idx=i,inference_session=session,reverse=False)
    model=Model()
    class Mask:
        def __init__(self,value):self.value=value
        def detach(self):return self
        def cpu(self):return self
        def numpy(self):return self.value
    class Processor:
        def init_video_session(self,**kwargs):return 'session'
        def add_text_prompt(self,session,*,text):return session
        def postprocess_outputs(self,session,result):
            mask=np.zeros((16,16),dtype=bool) if empty_mask else np.ones((16,16),dtype=bool)
            return {'object_ids':SimpleNamespace(tolist=lambda:[11,22]),'prompt_to_obj_ids':{'cube':[11],'sphere':[22]},
                'masks':[Mask(mask),Mask(mask)]}
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(float32='torch.float32',inference_mode=contextlib.nullcontext))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        Sam3VideoModel=SimpleNamespace(from_pretrained=lambda *a,**k:(model,{'missing_keys':[],'mismatched_keys':[]})),
        Sam3VideoProcessor=SimpleNamespace(from_pretrained=lambda *a,**k:Processor())))
    native=tmp_path/'native/request.native.json'
    payload={'inputs':[{'path':'fixture'}],'config':{'max_frames':4,'image_side':32},'model_path':'fixture',
        'dataset_root':str(tmp_path/'data'),'dataset_store':'data','output_prefix':'masks',
        'native_output':str(native),'binding':'fixture-bound'}
    return local_visual_worker,model,payload,native.with_suffix('.sam-usage.json')

def test_video_worker_persists_each_real_frame_forward_before_effect(tmp_path,monkeypatch):
    worker,model,payload,journal=sam_fixture(tmp_path,monkeypatch)
    runtime={};output,usage,finish=worker._sam_video(payload,{'subjects':SUBJECTS},runtime)
    assert model.calls==[0,1,2,3] and output['frame_count']==4 and finish=='stop'
    assert runtime['sam_video_counts']['frame_subject_units_attempted']==8
    assert json.loads(journal.read_text())['frame_forward_attempts']==4
    assert runtime['precision']['storage_elements_by_dtype']=={'torch.float32':1}
    assert len(list((tmp_path/'data').rglob('*.png')))==8

def test_video_worker_blocks_extra_frame_before_unreserved_forward(tmp_path,monkeypatch):
    worker,model,payload,journal=sam_fixture(tmp_path,monkeypatch,extra_frame=True)
    with pytest.raises(ModelError) as error:worker._sam_video(payload,{'subjects':SUBJECTS},{})
    assert error.value.code=='ACCOUNTING_MISMATCH' and model.calls==[0,1,2,3]
    assert json.loads(journal.read_text())['frame_subject_units_attempted']==8

def test_video_worker_retains_partial_cost_when_mask_quality_fails(tmp_path,monkeypatch):
    worker,model,payload,journal=sam_fixture(tmp_path,monkeypatch,empty_mask=True)
    with pytest.raises(ModelError) as error:worker._sam_video(payload,{'subjects':SUBJECTS},{})
    assert error.value.code=='QUALITY_REJECTED' and model.calls==[0]
    assert json.loads(journal.read_text())['frame_subject_units_attempted']==2

def test_review_instructs_same_overall_rule_without_breaking_bound_context(tmp_path,monkeypatch):
    from dataevolver.dataset import visual_steps
    root=tmp_path/'data';root.mkdir();Image.new('RGB',(16,16),'red').save(root/'image.png')
    ref=ArtifactRef('data','image.png',file_sha256(root/'image.png'),'image/png');store=ArtifactStore({'data':root})
    row={'id':'fixture','image':ref.to_dict(),'prompt':'a red cube'}
    output={'status':'uncertain','summary':'Source details unobserved','criteria':{
        'visible':{'status':'pass','reason':'visible'},'original_preserved':{'status':'uncertain','reason':'source not shown'}}}
    captured=[]
    def invoke(ctx,role,parameters,artifacts,schema):
        captured.append(parameters['prompt'])
        q=ModelRequest('fixture-review',role,({'role':'user','content':canonical_json(parameters)},),schema,tuple(artifacts))
        return {'request':q.to_dict(),'receipt_hash':'0'*64,'output':copy.deepcopy(output)}
    def write(name,value):
        path=root/name;path.write_text(canonical_json(value));return ArtifactRef('data',name,file_sha256(path))
    monkeypatch.setattr(visual_steps,'_invoke',invoke)
    ctx=SimpleNamespace(store=store,parameters={'training_objective':'Match actual visible content',
        'dimensions':['visible','original_preserved']},node=SimpleNamespace(operator='visual.review.image'),
        read=lambda _:row,write_json=write)
    visual_steps.quality_review(ctx)
    assert 'pass only when every criterion passes' in captured[0]
    context=json.loads(captured[0].split('\nREVIEW_CONTEXT=',1)[1])
    assert context['training_objective']==ctx.parameters['training_objective']
    before={p:p.read_bytes() for p in root.iterdir()}
    output['status']='pass'
    with pytest.raises(ValueError,match='contradicts'):visual_steps.quality_review(ctx)
    assert before=={p:p.read_bytes() for p in root.iterdir()}
