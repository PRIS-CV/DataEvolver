"""Typed runtime evidence and CPU checks; uncertainty never grants edit authority."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from .contracts import canonical_json, payload_sha256
from .surface_effects import SURFACE_OBSERVATION_SCHEMA, SurfaceSupport, UnsupportedSurfaceRepair

CONTRACT_ID = 'data.typed_surface_reconstruction_preflight'
CONTRACT_VERSION = '1'
OBSERVATION_VERSION = 'dataevolver.surface_observation.v2'
TYPED_SURFACE_SCHEMA = deepcopy(SURFACE_OBSERVATION_SCHEMA)
TYPED_SURFACE_SCHEMA['properties']['schema_version'] = {'type':'string','enum':[OBSERVATION_VERSION]}
TYPED_SURFACE_SCHEMA['required'].append('schema_version')
_uncertainty = TYPED_SURFACE_SCHEMA['properties']['uncertain_regions']['items']
_uncertainty['properties'].update({
    'kind': {'type':'string', 'enum':['effect_attribution','reconstruction']},
    'subject_ids': {'type':'array','minItems':1,'items':{'type':'string','minLength':1}}})
_uncertainty['required'] += ['kind','subject_ids']


class UnsupportedReconstruction(UnsupportedSurfaceRepair):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__('; '.join(x['category'] for x in receipt['conflicts']))


def validate_typed_observation(value, selected, *, structural_only=False):
    from jsonschema import validate
    from .surface_effects import validate_surface_observation
    validate(value, TYPED_SURFACE_SCHEMA)
    canonical_json(value)
    if any(len(set(x['subject_ids']))!=len(x['subject_ids']) for x in value['uncertain_regions']):
        raise ValueError('typed uncertainty repeats selected instance IDs')
    if any(set(x['subject_ids']) - set(selected) for x in value['uncertain_regions']):
        raise ValueError('typed uncertainty names an unbound selected instance')
    legacy = deepcopy(value)
    legacy.pop('schema_version')
    for row in legacy['uncertain_regions']:
        row.pop('kind'); row.pop('subject_ids')
    # Structural/geometry validation only; no semantic relabeling of old output.
    validate_surface_observation(legacy, selected, structural_only=structural_only)
    return value


def typed_surface_support(observation, selected, kept, *, size, load_mask, existing_support,
                          edge_margin=0, bindings=None, padding_fraction=None, model_size=None):
    import numpy as np
    from PIL import Image, ImageDraw
    from .context_reconstruction import context_crop_box
    validate_typed_observation(observation, [x['subject_id'] for x in selected])
    if type(edge_margin) is not int or not 0 <= edge_margin <= 8:
        raise ValueError('typed support preserves the frozen selected edge margin')
    if existing_support is None or existing_support.size != size:
        raise ValueError('typed reconstruction requires the actual existing authorized support')
    old_values = np.asarray(existing_support.convert('L'))
    if not np.isin(old_values, [0,255]).all():
        raise ValueError('existing authorized support must be binary, not an inferred alpha threshold')
    old = old_values > 0
    width, height = size
    def polygon(points):
        image = Image.new('L',size)
        ImageDraw.Draw(image).polygon([(round(x*(width-1)),round(y*(height-1))) for x,y in points],fill=255)
        return np.asarray(image)>0
    def instances(rows):
        result = np.zeros((height,width),dtype=bool)
        for row in rows:
            image = load_mask(row)
            if image.size != size: raise ValueError('typed instance mask dimensions changed')
            result |= np.asarray(image.convert('L'))>127
        return result
    selected_mask, kept_mask = instances(selected), instances(kept)
    visible = kept_mask.copy()
    for row in observation['preservation_regions']: visible |= polygon(row['polygon'])
    attribution = np.zeros_like(old)
    reconstruction = np.zeros_like(old)
    uncertainty_counts=[]
    for row in observation['uncertain_regions']:
        region = polygon(row['polygon'])
        uncertainty_counts.append({'kind':row['kind'],'subject_ids':row['subject_ids'],
            'pixels':int(region.sum()),'inside_existing_authorization_pixels':int((region & old).sum()),
            'outside_existing_authorization_pixels':int((region & ~old).sum()),
            'protected_visible_overlap_pixels':int((region & visible).sum())})
        if row['kind']=='effect_attribution': attribution |= region
        else: reconstruction |= region
    conflicts = []
    def conflict(category, pixels, subject_ids=()):
        count = int(pixels.sum())
        if count: conflicts.append({'category':category,'pixels':count,'subject_ids':list(subject_ids)})
    ids = [x['subject_id'] for x in selected]
    conflict('selected_preserved_mask_overlap',selected_mask & kept_mask,ids)
    conflict('existing_support_missing_selected',selected_mask & ~old,ids)
    conflict('existing_support_overrides_protected_visible',old & visible,ids)
    # Original task masks authorize removal of the selected silhouette; ambiguous
    # light outside those silhouettes is never licensed by that task alone.
    conflict('existing_support_uncertain_effect_attribution',old & attribution & ~selected_mask,ids)
    surfaces = {x['surface_id']:polygon(x['polygon']) for x in observation['surfaces']}
    effects = np.zeros_like(old)
    for row in observation['effects']:
        region = polygon(row['polygon']); subject = [row['subject_id']]
        conflict('effect_crosses_receiving_surface',region & ~surfaces[row['surface_id']],subject)
        conflict('effect_overrides_protected_visible',region & visible,subject)
        conflict('effect_uncertain_attribution',region & attribution & ~selected_mask,subject)
        conflict('effect_expansion_into_reconstruction_uncertainty',region & reconstruction & ~old,subject)
        effects |= region
    support = old | effects
    if not old.any(): conflicts.append({'category':'empty_existing_support','pixels':0,'subject_ids':ids})
    protected = visible | (attribution & ~selected_mask) | (reconstruction & ~old)
    receipt = {'schema_version':'dataevolver.reconstruction_preflight.v1',
        'contract_id':CONTRACT_ID,'contract_version':CONTRACT_VERSION,
        'bindings':bindings or {},'observation_sha256':payload_sha256(observation),
        'source_size':list(size),'existing_support_pixel_sha256':hashlib.sha256(old_values.tobytes()).hexdigest(),
        'uncertainty_regions':uncertainty_counts,'conflicts':conflicts,'editable_pixels':int(support.sum()),
        'existing_authorized_pixels':int(old.sum()),'attributed_effect_pixels':int(effects.sum()),
        'effect_expansion_pixels':int((effects & ~old).sum()),'protected_pixels':int(protected.sum()),
        'reconstruction_uncertainty_inside_authorization_pixels':int((reconstruction & old).sum()),
        'reconstruction_uncertainty_outside_authorization_pixels':int((reconstruction & ~old).sum()),
        'uncertainty_added_editable_pixels':0,'context_added_editable_pixels':0,
        'partial_attribution':any(x['effect_assessment']=='uncertain' for x in observation['subjects']),
        'unresolved_subject_ids':[x['subject_id'] for x in observation['subjects'] if x['effect_assessment']=='uncertain'],
        'conditioning_and_composition_identical':True,'complete_goal_established':False,
        'image_model_calls':0,'quality_verified':False,'supported':not conflicts}
    image = Image.fromarray(support.astype('uint8')*255)
    if padding_fraction is not None and not conflicts:
        try:
            box = context_crop_box(image,padding_fraction)
            dimensions = list(model_size or size)
            if len(dimensions)!=2 or dimensions[0]!=dimensions[1] or any(type(x) is not int or x<=0 for x in dimensions):
                raise ValueError('preflight model dimensions must be positive and square')
            side = box[2]-box[0]
            receipt['crop'] = {'source_crop_xyxy':list(box),'model_size':dimensions,
                'padding_fraction':padding_fraction,'padding_convention':'ceil(max(bbox width,height)*(1+2*fraction)); clamped square centered on bbox',
                'source_to_model_scale':[dimensions[0]/side,dimensions[1]/side],
                'source_origin_xy':list(box[:2]),'canvas_expanded':False}
        except ValueError as exc:
            conflicts.append({'category':'unsupported_context_crop','pixels':0,'subject_ids':ids,'reason':str(exc)})
            receipt['supported']=False
    receipt['receipt_sha256']=payload_sha256(receipt)
    if not receipt['supported']: raise UnsupportedReconstruction(receipt)
    return SurfaceSupport(image,Image.fromarray(protected.astype('uint8')*255),receipt)
