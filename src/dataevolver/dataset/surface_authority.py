"""Conservative executable pixels derived from runtime evidence and prior authority.

Unresolved pixels are preserved. Partial execution eligibility is not a claim
that the full removal goal can be completed or that an effect is ground truth.
"""
from __future__ import annotations
import hashlib
from .contracts import payload_sha256

OPERATOR_ID='data.conservative_surface_authority'
OPERATOR_VERSION='1'


def conservative_surface_support(observation,selected,kept,*,size,load_mask,existing_support,
                                 edge_margin=0,bindings=None,padding_fraction=None,model_size=None):
    import numpy as np
    from PIL import Image,ImageDraw
    from .surface_preflight import validate_typed_observation,UnsupportedReconstruction
    from .surface_effects import SurfaceSupport
    from .context_reconstruction import context_crop_box
    ids=[x['subject_id'] for x in selected];validate_typed_observation(observation,ids)
    if type(edge_margin) is not int or not 0<=edge_margin<=8:raise ValueError('frozen edge margin changed')
    if existing_support is None or existing_support.size!=size:raise ValueError('actual prior support required')
    pixels=np.asarray(existing_support.convert('L'))
    if not np.isin(pixels,[0,255]).all():raise ValueError('prior authority must be binary')
    old=pixels>0;width,height=size
    def polygon(points):
        im=Image.new('L',size);ImageDraw.Draw(im).polygon(
            [(round(x*(width-1)),round(y*(height-1))) for x,y in points],fill=255)
        return np.asarray(im)>0
    def masks(rows):
        total=np.zeros_like(old);each={}
        for row in rows:
            im=load_mask(row)
            if im.size!=size:raise ValueError('original instance mask dimensions changed')
            each[row['subject_id']]=np.asarray(im.convert('L'))>127;total|=each[row['subject_id']]
        return total,each
    silhouettes,by_id=masks(selected);kept_mask,_=masks(kept)
    visible=kept_mask.copy()
    for row in observation['preservation_regions']:visible|=polygon(row['polygon'])
    uncertain=np.zeros_like(old);reconstruction=np.zeros_like(old)
    for row in observation['uncertain_regions']:
        if row['kind']=='effect_attribution':uncertain|=polygon(row['polygon'])
        else:reconstruction|=polygon(row['polygon'])
    surfaces={x['surface_id']:polygon(x['polygon']) for x in observation['surfaces']}
    receiving=np.zeros_like(old)
    for mask in surfaces.values():receiving|=mask
    crossing=np.zeros_like(old);effects=np.zeros_like(old);per_subject={x:np.zeros_like(old) for x in ids}
    for row in observation['effects']:
        mask=polygon(row['polygon']);effects|=mask;per_subject[row['subject_id']]|=mask
        crossing|=mask & ~surfaces[row['surface_id']]
    exclusions=visible|uncertain|crossing|~receiving
    effect_authority=old & ~silhouettes & ~exclusions
    support=silhouettes | effect_authority
    conflicts=[]
    def conflict(category,mask):
        if mask.any():conflicts.append({'category':category,'pixels':int(mask.sum()),'subject_ids':ids})
    conflict('selected_preserved_mask_overlap',silhouettes & kept_mask)
    conflict('selected_silhouette_protected_visible',silhouettes & visible)
    conflict('selected_silhouette_outside_prior_authority',silhouettes & ~old)
    if not silhouettes.any():conflicts.append({'category':'empty_selected_authority','pixels':0,'subject_ids':ids})
    coverage=[]
    for sid in ids:
        rows=[x for x in observation['uncertain_regions'] if sid in x['subject_ids']]
        ambiguous=np.zeros_like(old)
        for row in rows:
            if row['kind']=='effect_attribution':ambiguous|=polygon(row['polygon'])
        coverage.append({'subject_id':sid,'selected_silhouette_pixels':int(by_id[sid].sum()),
            'executable_silhouette_pixels':int((by_id[sid]&support).sum()),
            'observed_effect_pixels_outside_execution':int((per_subject[sid]&~support).sum()),
            'uncertain_attribution_pixels_outside_silhouette':int((ambiguous&~silhouettes).sum()),
            'completion_established':False})
    excluded=old & ~support
    protected=(visible | uncertain | crossing | excluded | (reconstruction & ~old)) & ~silhouettes
    receipt={'schema_version':'dataevolver.surface_authority.v1','operator_id':OPERATOR_ID,
        'operator_version':OPERATOR_VERSION,'bindings':bindings or {},
        'observation_sha256':payload_sha256(observation),
        'existing_support_pixel_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
        'source_size':list(size),'coverage':coverage,'conflicts':conflicts,
        'silhouette_authority_pixels':int(silhouettes.sum()),'prior_effect_authority_pixels':int(effect_authority.sum()),
        'editable_pixels':int(support.sum()),'protected_pixels':int(protected.sum()),
        'existing_authorized_pixels':int(old.sum()),'excluded_prior_authority_pixels':int(excluded.sum()),
        'uncertain_prior_effect_pixels_preserved':int((old&uncertain&~silhouettes).sum()),
        'new_effect_pixels_outside_prior_authority_preserved':int((effects&~old).sum()),
        'reconstruction_uncertainty_inside_execution_pixels':int((reconstruction&support).sum()),
        'attributed_effect_pixels':int(effects.sum()),'effect_expansion_pixels':0,
        'uncertainty_added_editable_pixels':0,'context_added_editable_pixels':0,
        'support_subset_of_prior_authority':not bool((support&~old).any()),
        'conditioning_and_composition_identical':True,'partial_execution':bool(excluded.any() or (effects&~support).any()),
        'unresolved_subject_ids':[x['subject_id'] for x in coverage if x['observed_effect_pixels_outside_execution'] or x['uncertain_attribution_pixels_outside_silhouette']],
        'complete_goal_established':False,'full_removal_verified':False,'image_model_calls':0,'quality_verified':False,
        'supported':not conflicts}
    receipt['execution_status']='unsupported' if conflicts else ('partial_supported' if receipt['partial_execution'] else 'supported')
    image=Image.fromarray(support.astype('uint8')*255)
    if padding_fraction is not None and not conflicts:
        try:
            box=context_crop_box(image,padding_fraction);dimensions=list(model_size or size);side=box[2]-box[0]
            if len(dimensions)!=2 or dimensions[0]!=dimensions[1] or any(type(x) is not int or x<=0 for x in dimensions):
                raise ValueError('positive square model dimensions required')
            receipt['crop']={'source_crop_xyxy':list(box),'model_size':dimensions,'padding_fraction':padding_fraction,
                'source_origin_xy':list(box[:2]),'source_to_model_scale':[dimensions[0]/side,dimensions[1]/side],
                'padding_convention':'ceil(max(bbox width,height)*(1+2*fraction)); clamped square centered on bbox',
                'canvas_expanded':False}
        except ValueError as exc:
            conflicts.append({'category':'unsupported_context_crop','reason':str(exc),'pixels':0,'subject_ids':ids})
            receipt['supported']=False;receipt['execution_status']='unsupported'
    receipt['receipt_sha256']=payload_sha256(receipt)
    if conflicts:raise UnsupportedReconstruction(receipt)
    # No uncertain attribution pixel outside a selected task silhouette executes;
    # neither old rectangles, reconstruction uncertainty nor context add pixels.
    if (support&~old).any() or (support&visible).any() or (effect_authority&uncertain).any():
        raise ValueError('conservative authority invariant violated')
    return SurfaceSupport(image,Image.fromarray(protected.astype('uint8')*255),receipt)
