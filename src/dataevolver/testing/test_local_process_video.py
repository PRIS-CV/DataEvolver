"""CPU wire/effect tests with decoded-frame fixtures; no model quality claims."""
import base64
import copy
from io import BytesIO
import json
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from dataevolver.adapters.local_process_endpoint import LocalProcessEndpointProvider
from dataevolver.adapters.model_contracts import ModelPolicy,ModelProfile,ModelRequest
from dataevolver.adapters.model_gateway import ModelGateway,ModelError
from dataevolver.dataset.contracts import ArtifactRef,ArtifactStore,file_sha256,payload_sha256

SCHEMA={'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}
SAMPLING={'max_frames':8,'image_side':32}

def decoded_frames(monkeypatch,colors=('red','green','blue','yellow','purple'),sizes=None):
    frames=[Image.new('RGB',sizes[i] if sizes else (16,16),color) for i,color in enumerate(colors)]
    class Container:
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def decode(self,*,video):
            assert video==0
            for image in frames:
                yield SimpleNamespace(width=image.width,height=image.height,to_image=lambda im=image:im.copy())
    monkeypatch.setitem(sys.modules,'av',SimpleNamespace(open=lambda _:Container()))
    return frames

def fixture(tmp_path,monkeypatch,sampling=SAMPLING,media_type='video/mp4'):
    data=tmp_path/'data';data.mkdir();path=data/'input';path.write_bytes(b'CPU encoded-source fixture')
    store=ArtifactStore({'data':data});ref=ArtifactRef('data','input',file_sha256(path),media_type)
    if media_type=='image/png':
        Image.new('RGB',(16,16),'cyan').save(path,format='PNG');ref=ArtifactRef('data','input',file_sha256(path),media_type)
    deployment={'origin':'http://127.0.0.1:1','model':'fixture-qwen','served_model':'fixture-wire'}
    profile=ModelProfile('review','local_python',deployment['model'],
        LocalProcessEndpointProvider.profile_revision(deployment,sampling),('json','vision'))
    provider=LocalProcessEndpointProvider(profile,deployment,store,tmp_path/'native',video_sampling=sampling)
    monkeypatch.setattr(provider,'verify_deployment',lambda:None)
    wire=[]
    def send(suffix,*,payload=None,timeout=10):
        assert suffix=='/chat/completions';wire.append(copy.deepcopy(payload))
        return {'model':'fixture-wire','choices':[{'message':{'content':'{"ok":true}'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':10,'completion_tokens':4}}
    monkeypatch.setattr(provider,'_send',send)
    role='visual.review.video' if media_type.startswith('video/') else 'visual.review.image'
    request=ModelRequest('sampled-video',role,({'role':'user','content':'Inspect actual media; preserve uncertainty.'},),SCHEMA,(ref,))
    gateway=ModelGateway(profiles=(profile,),policy=ModelPolicy({role:'review'},1),
        receipt_root=tmp_path/'gateway',artifact_store=store,providers={'review':lambda:provider})
    return gateway,provider,request,wire

def test_video_transport_requires_explicit_binding_and_keeps_default_revision(tmp_path,monkeypatch):
    gateway,provider,request,wire=fixture(tmp_path,monkeypatch,sampling=None)
    assert provider.profile.revision=='sha256:'+payload_sha256(provider.deployment)
    with pytest.raises(ModelError,match='bound sampled-frame'):gateway.invoke(request)
    assert not wire and gateway.audit_usage()['model_calls']==1

def test_one_wire_image_contains_exact_first_middle_last_pixels_and_original_request(tmp_path,monkeypatch):
    frames=decoded_frames(monkeypatch);gateway,provider,request,wire=fixture(tmp_path,monkeypatch)
    original=copy.deepcopy(request.to_dict());receipt=gateway.invoke(request)
    assert receipt.output=={'ok':True} and receipt.usage['images']==1 and len(wire)==1
    assert request.to_dict()==original and wire[0]['messages'][1]==original['messages'][0]
    content=wire[0]['messages'][-1]['content'];assert len(content)==2 and '[0, 2, 4]' in content[0]['text']
    with Image.open(BytesIO(base64.b64decode(content[1]['image_url']['url'].split(',',1)[1]))) as sheet:
        assert sheet.size==(48,16)
        for position,index in enumerate([0,2,4]):
            assert sheet.crop((position*16,0,(position+1)*16,16)).tobytes()==frames[index].tobytes()
    native=json.loads((provider.root/'sampled-video.native.json').read_text())
    observation=native['response']['runtime']['video_observation']
    assert observation['source']==request.artifacts[0].to_dict() and observation['frame_indices']==[0,2,4]
    assert observation['decoded_frame_count']==5 and observation['scope']=='first_middle_last_frames_only'
    hashes={p:p.read_bytes() for p in provider.root.rglob('*') if p.is_file()}
    monkeypatch.setitem(sys.modules,'av',SimpleNamespace(open=lambda _:pytest.fail('cached effect decoded again')))
    assert provider.lookup(request,gateway._prepare(request)[4])['output']=={'ok':True}
    monkeypatch.setattr(gateway,'_provider',lambda *_:pytest.fail('cached effect created provider'))
    assert gateway.invoke(request).reused and len(wire)==1
    assert hashes=={p:p.read_bytes() for p in provider.root.rglob('*') if p.is_file()}

def test_sampling_options_cannot_reuse_a_profile_for_different_frames(tmp_path,monkeypatch):
    _,provider,_,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='exact loopback'):
        LocalProcessEndpointProvider(provider.profile,provider.deployment,provider.store,tmp_path/'other',
            video_sampling={'max_frames':16,'image_side':32})

@pytest.mark.parametrize('change',['sampling','deployment','invalid_sampling'])
def test_post_construction_binding_drift_rejected_before_native_intent(tmp_path,monkeypatch,change):
    gateway,provider,request,wire=fixture(tmp_path,monkeypatch)
    if change=='sampling':provider.video_sampling['max_frames']=16
    elif change=='deployment':provider.deployment['max_model_len']=1000000
    else:provider.video_sampling={'max_frames':True,'image_side':32}
    with pytest.raises(ModelError) as error:gateway.invoke(request)
    assert error.value.code=='DEPLOYMENT_MISMATCH'
    assert not wire and not provider.root.exists()
    assert gateway.audit_usage()['model_calls']==1

def test_native_lookup_rejects_later_sampling_drift(tmp_path,monkeypatch):
    decoded_frames(monkeypatch);gateway,provider,request,wire=fixture(tmp_path,monkeypatch);gateway.invoke(request)
    provider.video_sampling['max_frames']=16
    with pytest.raises(ModelError) as error:provider.lookup(request,gateway._prepare(request)[4])
    assert error.value.code=='DEPLOYMENT_MISMATCH' and len(wire)==1

@pytest.mark.parametrize('sampling',[{'max_frames':True,'image_side':32},{'max_frames':257,'image_side':32},
    {'max_frames':8,'image_side':4096},{'max_frames':8,'image_side':32,'extra':1}])
def test_invalid_sampling_bounds_rejected_before_effects(sampling):
    with pytest.raises(ValueError,match='explicit frame'):LocalProcessEndpointProvider.profile_revision({},sampling)

@pytest.mark.parametrize('kind',['frames','dimensions','inconsistent'])
def test_decode_limits_and_changing_dimensions_fail_before_http(tmp_path,monkeypatch,kind):
    colors=('red','blue','green','yellow') if kind=='frames' else ('red','blue','green')
    sizes=[(48,48)]*3 if kind=='dimensions' else [(16,16),(16,16),(24,16)] if kind=='inconsistent' else None
    decoded_frames(monkeypatch,colors,sizes)
    gateway,provider,request,wire=fixture(tmp_path,monkeypatch,sampling={'max_frames':3,'image_side':32})
    with pytest.raises((ValueError,ModelError)):gateway.invoke(request)
    assert not wire and not list(provider.root.rglob('contact-sheet.png'))
    assert gateway.audit_usage()['model_calls']==1

def test_native_lookup_rejects_changed_contact_sheet(tmp_path,monkeypatch):
    decoded_frames(monkeypatch);gateway,provider,request,wire=fixture(tmp_path,monkeypatch);gateway.invoke(request)
    sheet=next(provider.root.rglob('contact-sheet.png'));sheet.write_bytes(b'changed')
    with pytest.raises(ModelError) as error:provider.lookup(request,gateway._prepare(request)[4])
    assert error.value.code=='INVALID_RECEIPT'
    assert len(wire)==1

def test_video_sampling_binding_preserves_image_wire_without_video_metadata(tmp_path,monkeypatch):
    gateway,provider,request,wire=fixture(tmp_path,monkeypatch,media_type='image/png');receipt=gateway.invoke(request)
    assert receipt.output=={'ok':True} and len(wire[0]['messages'][-1]['content'])==1
    assert 'runtime' not in json.loads((provider.root/'sampled-video.native.json').read_text())['response']

def test_interrupted_video_http_is_never_resent_or_redecoded(tmp_path,monkeypatch):
    decoded_frames(monkeypatch);gateway,provider,request,wire=fixture(tmp_path,monkeypatch)
    def interrupted(*args,**kwargs):
        wire.append('one failed HTTP');raise ModelError('AMBIGUOUS_EFFECT','fixture disconnect')
    monkeypatch.setattr(provider,'_send',interrupted)
    with pytest.raises(ModelError):gateway.invoke(request)
    before={p:p.read_bytes() for p in provider.root.rglob('*') if p.is_file()}
    monkeypatch.setitem(sys.modules,'av',SimpleNamespace(open=lambda _:pytest.fail('uncertain effect decoded again')))
    with pytest.raises(ModelError,match='no committed native completion'):gateway.invoke(request)
    assert len(wire)==1 and gateway.audit_usage()['model_calls']==1
    assert before=={p:p.read_bytes() for p in provider.root.rglob('*') if p.is_file()}
