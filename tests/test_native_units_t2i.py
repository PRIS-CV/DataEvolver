import copy
import json
import pytest

from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile, ModelRequest
from dataevolver.adapters.model_gateway import ModelGateway, ModelError
from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, canonical_json, file_sha256
from dataevolver.dataset.supervision import export_dataset, validate_record
from dataevolver.dataset.t2i_supervision import from_visual_receipts, export_imagefolder, audit_imagefolder
from dataevolver.harness.attempt_limits import ModelAttemptLimits


def image(tmp_path):
    from PIL import Image
    store = ArtifactStore({'data': tmp_path/'assets'})
    store.roots['data'].mkdir()
    Image.new('RGB', (16, 16), 'red').save(store.roots['data']/'image.png')
    return store, ArtifactRef('data', 'image.png', file_sha256(store.roots['data']/'image.png'), 'image/png')


def test_subject_units_enforced_before_second_effect_and_resume_is_free(tmp_path):
    store, ref = image(tmp_path)
    profile = ModelProfile('sam', 'mock', 'fixture-sam', '1', capabilities=('json','vision'), mock_output={})
    limits = ModelAttemptLimits({'sam': 3}, {'segment': 'sam'}, unit_rules={'segment': 'image_subject'})
    gateway = ModelGateway(profiles=(profile,), policy=ModelPolicy({'segment':'sam'}, 9),
        receipt_root=tmp_path/'receipts', artifact_store=store, attempt_limits=limits)
    def request(name):
        return ModelRequest(name,'segment',({'role':'user','content':canonical_json({
            'subjects':[{'subject_id':'a'},{'subject_id':'b'}]})},),{'type':'object'},(ref,))
    gateway.invoke(request('one'))
    gateway.invoke(request('one'))
    assert gateway.audit_usage()['native_units_reserved'] == 2
    with pytest.raises(ModelError, match='exhausted'): gateway.invoke(request('two'))
    assert not (gateway.root/'two.intent.json').exists()


def pair(tmp_path):
    store, ref = image(tmp_path)
    params = {'prompt':'a red ball left of a blue cube','width':16,'height':16,'steps':1,'true_cfg_scale':1}
    output = {'artifact':ref.to_dict(),'actual_parameters':params,'image_size':[16,16]}
    review = {'status':'pass','criteria':{'spatial':{'status':'pass','evidence':'fixture'}}}
    profiles = (ModelProfile('g','mock','fixture','1',capabilities=('json','vision'),mock_output=output),
                ModelProfile('r','mock','fixture','1',capabilities=('json','vision'),mock_output=review))
    gateway = ModelGateway(profiles=profiles,policy=ModelPolicy({
        'visual.image.generate':'g','visual.review.image':'r'},2),receipt_root=tmp_path/'receipts',artifact_store=store)
    envelopes=[]
    for role,content,artifacts in [('visual.image.generate',params,()),('visual.review.image',{'prompt':'review'},(ref,))]:
        q=ModelRequest('generate' if not artifacts else 'review',role,
            ({'role':'user','content':canonical_json(content)},),{'type':'object'},artifacts)
        r=gateway.invoke(q)
        envelopes.append({'request':q.to_dict(),'receipt_hash':r.without_reuse_hash(),'output':r.output})
    row=from_visual_receipts(gateway,*envelopes,record_id='sample',family_id='family',split='train',provenance={'source':'fixture'})
    row['evidence_kind']='fixture'
    return store,row


def test_export_caption_matches_actual_generation_and_media_bytes(tmp_path):
    store,row=pair(tmp_path)
    validate_record(row,store,allow_fixture=True)
    with pytest.raises(ValueError):validate_record(row,store)
    export_dataset([row],store,tmp_path/'bundle',expected_counts={'train':1},allow_fixture=True)
    exported=export_imagefolder(tmp_path/'bundle',tmp_path/'training',allow_fixture=True)
    assert exported['records'][0]['caption']==row['input']['prompt']
    assert exported['records'][0]['media_sha256']==row['target']['image']['sha256']
    assert audit_imagefolder(tmp_path/'training')['unique_samples']==1
    bad=copy.deepcopy(row);bad['input']['prompt']='wrong condition'
    with pytest.raises(ValueError,match='binding'):validate_record(bad,store,allow_fixture=True)
    (tmp_path/'training'/'train'/'sample.png').write_bytes(b'changed')
    with pytest.raises(ValueError):audit_imagefolder(tmp_path/'training')
