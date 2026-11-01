"""A source-coordinate crop for the fixed resident editor, not a new model.

Runtime evidence supplies the authorized support and runtime proposals choose the
context padding. The crop is square and inverse-mapped without canvas expansion.
Quality remains the responsibility of the unchanged frozen evaluators.
"""
from __future__ import annotations

import math
from .contracts import ArtifactRef,canonical_json
from .supervision import write_image,write_artifact
from .surface_effects import UnsupportedSurfaceRepair

OPERATOR_ID='data.surface_context_reconstruction'
OPERATOR_VERSION='1'


def context_crop_box(support, padding_fraction):
    if type(padding_fraction) not in (int,float) or not math.isfinite(padding_fraction) or not 0<=padding_fraction<=.25:
        raise ValueError('runtime context padding must be a finite fraction between zero and one quarter')
    width,height=support.size
    if width!=height:raise UnsupportedSurfaceRepair('context reconstruction currently requires a square source')
    box=support.getbbox()
    if box is None:raise UnsupportedSurfaceRepair('context reconstruction has empty authorized support')
    x0,y0,x1,y1=box
    side=math.ceil(max(x1-x0,y1-y0)*(1+2*padding_fraction))
    if side>=width:
        raise UnsupportedSurfaceRepair('context crop would repeat full-frame conditioning; no distinct reconstruction is available')
    left=max(0,min(width-side,math.floor((x0+x1-side)/2)))
    top=max(0,min(height-side,math.floor((y0+y1-side)/2)))
    result=(left,top,left+side,top+side)
    if not (result[0]<=x0<x1<=result[2] and result[1]<=y0<y1<=result[3]):
        raise ValueError('context crop excludes authorized support')
    return result


def prepare_context_condition(store,store_name,family,target,condition,support,parameters,*,model_size):
    from PIL import Image
    if set(parameters)!={'context_padding_fraction'}:raise ValueError('context reconstruction parameters changed')
    source=Image.open(store.resolve(target)).convert('RGB')
    image=Image.open(store.resolve(condition)).convert('RGB')
    mask=Image.open(store.resolve(support)).convert('L')
    if source.size!=image.size or source.size!=mask.size or model_size[0]!=model_size[1]:
        raise ValueError('context reconstruction image frames differ')
    box=context_crop_box(mask,parameters['context_padding_fraction'])
    crop=image.crop(box).resize(tuple(model_size),Image.Resampling.LANCZOS)
    prefix=family+'/surface-reconstruction/'
    ref=write_image(store,store_name,prefix+'model-condition.png',crop)
    value={'schema_version':'dataevolver.context_reconstruction.v1',
        'operator_id':OPERATOR_ID,'operator_version':OPERATOR_VERSION,
        'source':target.to_dict(),'source_condition':condition.to_dict(),'authorized_support':support.to_dict(),
        'model_condition':ref.to_dict(),'source_size':list(source.size),'model_size':list(model_size),
        'source_crop_xyxy':list(box),'parameters':parameters,'image_resampling':'lanczos',
        'canvas_expanded':False,'quality_verified':False,'proposal_actor':'runtime_agent'}
    contract=write_artifact(store,store_name,prefix+'condition-transform.json',canonical_json(value).encode(),'application/json')
    return ref,contract


def project_context_output(store,store_name,family,target,raw_model,transform):
    from PIL import Image
    import json
    value=json.loads(store.resolve(transform).read_text())
    source=Image.open(store.resolve(target)).convert('RGB')
    raw=Image.open(store.resolve(raw_model)).convert('RGB')
    if value['source']!=target.to_dict() or raw.size!=tuple(value['model_size']) or source.size!=tuple(value['source_size']):
        raise ValueError('context reconstruction output or source dimensions changed')
    box=value['source_crop_xyxy'];side=box[2]-box[0]
    if side!=box[3]-box[1] or side<=0:raise ValueError('context inverse requires its square source crop')
    projected=source.copy();projected.paste(raw.resize((side,side),Image.Resampling.LANCZOS),(box[0],box[1]))
    prefix=family+'/surface-reconstruction/'
    ref=write_image(store,store_name,prefix+'projected-edit.png',projected)
    proof={**value,'condition_transform':transform.to_dict(),'raw_model_output':raw_model.to_dict(),
        'projected_model_output':ref.to_dict(),'source_coordinate_projection':True,
        'composition_support_unchanged':True,'quality_verified':False}
    evidence=write_artifact(store,store_name,prefix+'projection.json',canonical_json(proof).encode(),'application/json')
    return ref,evidence
