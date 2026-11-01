"""Evidence-bound surface/effect support, shared by conditioning and composition.

This module supplies no image-specific geometry. Registered runtime observations
supply all polygons. Pixel differences are diagnostic, never attribution labels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import ArtifactRef, canonical_json
from .supervision import write_artifact, write_image

OPERATOR_ID = 'data.surface_effect_support'
OPERATOR_VERSION = '2'

def _object(properties):
    return {'type':'object','additionalProperties':False,'required':list(properties),'properties':properties}

def _array(item, maximum=32):
    return {'type':'array','maxItems':maximum,'items':item}

_TEXT={'type':'string','minLength':1,'maxLength':2000}
_POLYGON={'type':'array','minItems':3,'maxItems':64,'items':{
    'type':'array','minItems':2,'maxItems':2,'items':{'type':'number','minimum':0,'maximum':1}}}
SURFACE_OBSERVATION_SCHEMA=_object({
    'status':{'type':'string','enum':['supported','unsupported']},
    'reason':_TEXT,
    'surfaces':_array(_object({'surface_id':_TEXT,'description':_TEXT,'polygon':_POLYGON})),
    'subjects':_array(_object({'subject_id':_TEXT,'surface_ids':_array(_TEXT),
        'effect_assessment':{'type':'string','enum':['attributed','none_observed','uncertain']},'reason':_TEXT})),
    'effects':_array(_object({'subject_id':_TEXT,'surface_id':_TEXT,
        'kind':{'type':'string','enum':['cast_shadow','contact_shadow','reflection','color_spill','vapor']},
        'polygon':_POLYGON,'reason':_TEXT})),
    'preservation_regions':_array(_object({'polygon':_POLYGON,'reason':_TEXT})),
    'uncertain_regions':_array(_object({'polygon':_POLYGON,'reason':_TEXT})),
})

class UnsupportedSurfaceRepair(ValueError):
    """Known unsupported attribution or geometry; never a quality acceptance."""


def _polygon(points):
    if len({tuple(p) for p in points})!=len(points):
        raise ValueError('surface polygon repeats vertices')
    def cross(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    area=sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(points,points[1:]+points[:1]))
    if not math.isfinite(area) or abs(area)<1e-8:
        raise ValueError('surface polygon has no finite area')
    edges=list(zip(points,points[1:]+points[:1]))
    for i,(a,b) in enumerate(edges):
        for j,(c,d) in enumerate(edges):
            if j<=i or j==i+1 or (i==0 and j==len(edges)-1):continue
            def on(a,b,c):
                return abs(cross(a,b,c))<1e-12 and min(a[0],b[0])<=c[0]<=max(a[0],b[0]) and min(a[1],b[1])<=c[1]<=max(a[1],b[1])
            if ((cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0)
                    or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b)):
                raise ValueError('self-intersecting surface polygon')


def validate_surface_observation(value, selected, *, structural_only=False):
    if value.get('schema_version') == 'dataevolver.removal_coverage_observation.v1':
        from .surface_composition import validate_coverage_observation
        return validate_coverage_observation(value, selected, structural_only=structural_only)
    if value.get('schema_version'):
        from .surface_preflight import validate_typed_observation
        return validate_typed_observation(value,selected,structural_only=structural_only)
    from jsonschema import validate
    validate(value,SURFACE_OBSERVATION_SCHEMA)
    # JSON schema permits arbitrary Python float values in some validators.
    canonical_json(value)
    for group in ('surfaces','effects','preservation_regions','uncertain_regions'):
        for row in value[group]:_polygon(row['polygon'])
    surfaces={x['surface_id']:x for x in value['surfaces']}
    if len(surfaces)!=len(value['surfaces']):raise ValueError('duplicate receiving surface')
    subjects={x['subject_id']:x for x in value['subjects']}
    if len(subjects)!=len(value['subjects']) or set(subjects)!=set(selected):
        raise ValueError('surface observation must cover exactly selected instances')
    for row in subjects.values():
        if set(row['surface_ids'])-surfaces.keys() or len(set(row['surface_ids']))!=len(row['surface_ids']):
            raise ValueError('subject receiving surface is unbound')
    for row in value['effects']:
        if row['subject_id'] not in subjects or row['surface_id'] not in subjects[row['subject_id']]['surface_ids']:
            raise ValueError('effect must bind a selected instance and its receiving surface')
        if subjects[row['subject_id']]['effect_assessment']!='attributed':
            raise ValueError('effect conflicts with its attribution assessment')
    for row in subjects.values():
        if row['effect_assessment']=='attributed' and not any(e['subject_id']==row['subject_id'] for e in value['effects']):
            raise ValueError('attribution assessment has no observed effect')
    if not structural_only and value['status']!='supported':
        raise UnsupportedSurfaceRepair('registered surface attribution is unsupported: '+value['reason'])
    if not structural_only and any(x['effect_assessment']=='uncertain' for x in subjects.values()) and not value['uncertain_regions']:
        raise UnsupportedSurfaceRepair('partial attribution requires explicit uncertain regions to preserve')
    return value


@dataclass(frozen=True)
class SurfaceSupport:
    """One raster support; conditioning and composition consume this same object."""
    support: object
    protected: object
    diagnostics: dict


def rasterize_surface_support(observation, selected, kept, *, size, load_mask, edge_margin=0, existing_support=None, bindings=None, padding_fraction=None, model_size=None, authority_separation=False, surface_composition=False):
    if surface_composition:
        from .surface_composition import coverage_surface_support
        return coverage_surface_support(observation, selected, kept, size=size, load_mask=load_mask,
            existing_support=existing_support, bindings=bindings, edge_margin=edge_margin,
            padding_fraction=padding_fraction, model_size=model_size)
    if authority_separation:
        from .surface_authority import conservative_surface_support
        return conservative_surface_support(observation,selected,kept,size=size,load_mask=load_mask,
            edge_margin=edge_margin,existing_support=existing_support,bindings=bindings,
            padding_fraction=padding_fraction,model_size=model_size)
    if observation.get('schema_version'):
        from .surface_preflight import typed_surface_support
        return typed_surface_support(observation,selected,kept,size=size,load_mask=load_mask,
            edge_margin=edge_margin,existing_support=existing_support,bindings=bindings,
            padding_fraction=padding_fraction,model_size=model_size)
    import numpy as np
    from PIL import Image,ImageDraw,ImageFilter
    validate_surface_observation(observation,[x['subject_id'] for x in selected])
    width,height=size
    if type(edge_margin) is not int or not 0<=edge_margin<=8:
        raise ValueError('surface support preserves the frozen selected edge margin')
    def polygon(points):
        im=Image.new('L',size);ImageDraw.Draw(im).polygon(
            [(round(x*(width-1)),round(y*(height-1))) for x,y in points],fill=255)
        return np.asarray(im)>0
    def masks(instances):
        result=np.zeros((height,width),dtype=bool)
        for row in instances:
            mask=load_mask(row)
            if mask.size!=size:raise ValueError('surface instance mask dimensions changed')
            result|=np.asarray(mask.convert('L'))>127
        return result
    selected_mask=masks(selected);kept_mask=masks(kept)
    if np.any(selected_mask&kept_mask):
        raise UnsupportedSurfaceRepair('selected and preserved segmentation overlap')
    objects=selected_mask
    if edge_margin:
        objects=np.asarray(Image.fromarray(selected_mask.astype('uint8')*255).filter(
            ImageFilter.MaxFilter(2*edge_margin+1)))>0
    protected=kept_mask.copy()
    for row in observation['preservation_regions']+observation['uncertain_regions']:
        # Observed surface exclusions protect visible source content. The known
        # selected object silhouettes are removed under the original task; no
        # underlying surface pixel is asserted to have been observed there.
        protected|=polygon(row['polygon'])&~selected_mask
    surfaces={x['surface_id']:polygon(x['polygon']) for x in observation['surfaces']}
    effects=np.zeros((height,width),dtype=bool)
    for row in observation['effects']:
        effect=polygon(row['polygon'])
        if np.any(effect&~surfaces[row['surface_id']]):
            raise UnsupportedSurfaceRepair('attributed effect crosses its observed receiving surface; split evidence is required')
        if np.any(effect&protected):
            raise UnsupportedSurfaceRepair('attributed effect overlaps protected or uncertain content')
        effects|=effect
    support=(objects|effects)&~protected
    if not support.any() or np.any(selected_mask&~support):
        raise UnsupportedSurfaceRepair('surface support does not cover selected objects')
    return SurfaceSupport(Image.fromarray(support.astype('uint8')*255),
        Image.fromarray(protected.astype('uint8')*255),
        {'editable_pixels':int(support.sum()),'attributed_effect_pixels':int(effects.sum()),
         'protected_pixels':int(protected.sum()),'selected_pixels':int(selected_mask.sum()),
         'conditioning_and_composition_identical':True,
         'partial_attribution':any(x['effect_assessment']=='uncertain' for x in observation['subjects']),
         'unresolved_subject_ids':[x['subject_id'] for x in observation['subjects'] if x['effect_assessment']=='uncertain'],
         'complete_goal_established':False,'pixel_differences_establish_attribution':False})


def compile_surface_artifacts(store,store_name,family,target,selected,kept,contract,*,edge_margin=0,model_size=None):
    from PIL import Image
    source=Image.open(store.resolve(target)).convert('RGB')
    if contract['source_image']!=target.to_dict() or contract['proposal_actor']!='runtime_agent':
        raise ValueError('surface contract must bind its actual source and runtime observation')
    store.resolve(ArtifactRef.from_dict(contract['rejected_candidate']))
    existing=contract.get('existing_authorized_support')
    if contract['observation'].get('schema_version') and not existing:
        raise ValueError('typed contract has no bound existing support')
    compiled=rasterize_surface_support(contract['observation'],selected,kept,size=source.size,
        existing_support=Image.open(store.resolve(ArtifactRef.from_dict(existing))) if existing else None,
        authority_separation=bool(contract.get('implementation',{}).get('authority_separation')),
        surface_composition=bool(contract.get('implementation',{}).get('composition')),
        bindings={'source':target.to_dict(),'candidate':contract['rejected_candidate'],
            'existing_authorized_support':existing,'instance_masks':[x['mask'] for x in selected+kept]},
        padding_fraction=contract.get('reconstruction',{}).get('context_padding_fraction'),model_size=model_size,
        load_mask=lambda x:Image.open(store.resolve(ArtifactRef.from_dict(x['mask']))),edge_margin=edge_margin)
    prefix=family+'/surface-repair/'
    region=write_image(store,store_name,prefix+'support.png',compiled.support)
    protected=write_image(store,store_name,prefix+'protected.png',compiled.protected)
    # This exact support becomes BOTH the missing input and the composite alpha.
    condition=Image.composite(Image.new('RGB',source.size,(127,127,127)),source,compiled.support)
    conditioning=write_image(store,store_name,prefix+'condition.png',condition)
    visible=source.convert('RGBA');visible.putalpha(Image.eval(compiled.support,lambda x:255-x))
    visible_ref=write_image(store,store_name,prefix+'visible.png',visible)
    document={**contract,'operator_id':OPERATOR_ID,'operator_version':OPERATOR_VERSION,
        'selected_instances':selected,'preserved_instances':kept,
        'support':region.to_dict(),'conditioning_region':region.to_dict(),'composition_alpha':region.to_dict(),
        'conditioning_image':conditioning.to_dict(),'protected':protected.to_dict(),
        'diagnostics':compiled.diagnostics,'edge_margin_pixels':edge_margin}
    ref=write_artifact(store,store_name,prefix+'contract.json',canonical_json(document).encode(),'application/json')
    return visible_ref,region,conditioning,ref


def audit_surface_composition(store,target,raw,completed,region,protected):
    import numpy as np
    from PIL import Image
    def read(ref,mode):return np.asarray(Image.open(store.resolve(ref)).convert(mode))
    s,r,c=[read(ref,'RGB') for ref in (target,raw,completed)]
    a=read(region,'L')>0;p=read(protected,'L')>0
    if s.shape!=r.shape or s.shape!=c.shape:raise ValueError('surface composition changed dimensions')
    if np.any(c[a]!=r[a]) or np.any(c[~a]!=s[~a]) or np.any(c[p]!=s[p]):
        raise ValueError('surface composition violates the shared support or preservation contract')
    return {'supported_pixels_equal_raw':True,'outside_pixels_equal_source':True,'protected_pixels_equal_source':True,
        'restored_supported_source_effect_pixels':0,'quality_or_attribution_established':False,
        'support_sha256':region.sha256,'raw_sha256':raw.sha256,'completed_sha256':completed.sha256}
