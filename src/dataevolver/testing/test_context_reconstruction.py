"""Coordinate and receipt fixtures; no image-quality or method-benefit claims."""
import json
import shutil
import numpy as np
import pytest
from PIL import Image

from dataevolver.dataset.contracts import ArtifactStore,ArtifactRef
from dataevolver.dataset.supervision import write_image,artifact_closure,complete_background_composite
from dataevolver.dataset.context_reconstruction import context_crop_box,prepare_context_condition,project_context_output
from dataevolver.dataset.surface_effects import UnsupportedSurfaceRepair,audit_surface_composition


def test_context_transform_inverse_preserves_canvas_and_closed_artifact_provenance(tmp_path):
    store=ArtifactStore({'data':str(tmp_path/'data')})
    source=write_image(store,'data','source.png',Image.new('RGB',(32,32),(10,20,30)))
    mask=np.zeros((32,32),dtype='uint8');mask[8:15,9:18]=255
    support=write_image(store,'data','support.png',Image.fromarray(mask))
    condition=write_image(store,'data','condition.png',Image.fromarray(np.where(mask[:,:,None]>0,127,np.array([10,20,30])).astype('uint8')))
    image,transform=prepare_context_condition(store,'data','fixture',source,condition,support,
        {'context_padding_fraction':.1},model_size=(64,64))
    raw=write_image(store,'data','raw-model.png',Image.new('RGB',(64,64),(200,210,220)))
    projected,proof=project_context_output(store,'data','fixture',source,raw,transform)
    value=json.loads(store.resolve(proof).read_text())
    assert value['source_coordinate_projection'] and not value['canvas_expanded'] and not value['quality_verified']
    assert value['raw_model_output']==raw.to_dict() and value['projected_model_output']==projected.to_dict()
    assert Image.open(store.resolve(image)).size==(64,64) and Image.open(store.resolve(projected)).size==(32,32)
    out=complete_background_composite(store,'data','fixture',source,projected,support)
    pixels=np.asarray(Image.open(store.resolve(out)))
    assert np.all(pixels[mask==0]==[10,20,30]) and np.all(pixels[mask>0]==[200,210,220])
    assert prepare_context_condition(store,'data','fixture',source,condition,support,{'context_padding_fraction':.1},model_size=(64,64))==(image,transform)
    shutil.copytree(store.roots['data'],tmp_path/'copied')
    copied=ArtifactStore({'data':str(tmp_path/'copied')})
    refs=list(artifact_closure({'proof':proof.to_dict()},copied))
    assert raw in refs and transform in refs and condition in refs and support in refs
    for ref in refs:
        if ref.media_type=='image/png':
            with Image.open(copied.resolve(ref)) as im:im.load()
    wrong=write_image(store,'data','wrong-size.png',Image.new('RGB',(32,32)))
    with pytest.raises(ValueError,match='dimensions changed'):project_context_output(store,'data','other',source,wrong,transform)


@pytest.mark.parametrize('padding',[-.1,.26,float('nan'),True,'0.1'])
def test_context_crop_rejects_unbounded_or_non_numeric_padding(padding):
    mask=Image.new('L',(32,32));mask.paste(255,(10,10,15,15))
    with pytest.raises(ValueError):context_crop_box(mask,padding)


def test_context_crop_refuses_duplicate_full_frame_and_retains_edge_support():
    with pytest.raises(UnsupportedSurfaceRepair,match='full-frame'):context_crop_box(Image.new('L',(32,32),255),0)
    with pytest.raises(UnsupportedSurfaceRepair,match='empty'):context_crop_box(Image.new('L',(32,32)),.1)
    with pytest.raises(UnsupportedSurfaceRepair,match='square'):context_crop_box(Image.new('L',(32,16)),.1)
    mask=Image.new('L',(32,32));mask.paste(255,(27,15,32,20))
    box=context_crop_box(mask,.2)
    assert box[2]==32 and box[0]<=27 and box[1]<=15 and box[3]>=20
    assert box[2]-box[0]==box[3]-box[1]
