"""Bounded background execution expressions; user goals and checks stay frozen."""
from __future__ import annotations

import copy
import math


RECIPE_SPACE = {
    'background_true_cfg_scale': {'minimum': 1, 'maximum': 8},
    'background_conditioning': ['source_image', 'masked_region'],
    'background_conditioning_shape': ['region', 'bounding_rectangle'],
    'mask_dilation_pixels': {'minimum': 0, 'maximum': 128, 'type': 'integer'},
    'background_feather_pixels': {'minimum': 0, 'maximum': 32, 'type': 'integer'},
    'selected_edge_margin_pixels': {'minimum': 0, 'maximum': 8, 'type': 'integer'},
    'background_addendum': {'type': 'string', 'max_length': 2000,
        'role': 'additional execution wording; original removal/preservation requirements and evaluation remain mandatory'},
}


def validate_recipe_patch(parameters, selected):
    from .adapters.training_rsi import validate_regions
    if not isinstance(parameters, dict) or set(parameters)-{'recipe', 'regions'}:
        raise ValueError('background recipe supports only registered recipe fields and optional observed regions')
    recipe=parameters.get('recipe')
    if not isinstance(recipe, dict) or not recipe or set(recipe)-set(RECIPE_SPACE):
        raise ValueError('background recipe contains an unknown or protected field')
    for key,value in recipe.items():
        spec=RECIPE_SPACE[key]
        if isinstance(spec,list):
            if value not in spec:raise ValueError('unknown background execution mode')
        elif spec.get('type')=='string':
            if not isinstance(value,str) or not value.strip() or len(value)>spec['max_length']:
                raise ValueError('background addendum must be bounded nonempty execution text')
        elif (type(value) not in (int,float) or not math.isfinite(value) or
              not spec['minimum']<=value<=spec['maximum'] or
              (spec.get('type')=='integer' and type(value) is not int)):
            raise ValueError('background execution value is outside its registered range')
    if 'regions' in parameters:validate_regions({'regions':parameters['regions']},selected)
    return parameters


def apply_recipe_patch(contract, parameters):
    result=copy.deepcopy(contract)
    validate_recipe_patch(parameters,result['production']['tasks'][0]['selected_subject_ids'])
    recipe=result['production']['recipe'];task=result['production']['tasks'][0]
    for key,value in parameters['recipe'].items():
        if key in {'background_true_cfg_scale','selected_edge_margin_pixels'}:task[key]=value
        elif key=='background_addendum':recipe['background_instruction']+='\nRuntime execution detail: '+value
        else:recipe[key]=value
    # Shape describes a missing-region transformation. It has no operation
    # under source-image conditioning, so canonicalize the unused field rather
    # than requiring the Planner to propose a second, inert change.
    if recipe.get('background_conditioning','source_image')=='source_image':
        recipe['background_conditioning_shape']='region'
    return result


def recipe_view(contract):
    recipe=contract['production'].get('recipe',{});task=contract['production']['tasks'][0]
    result={k:recipe.get(k) for k in RECIPE_SPACE if k not in
            {'background_true_cfg_scale','selected_edge_margin_pixels','background_addendum'}}
    result.update(background_true_cfg_scale=task.get('background_true_cfg_scale',recipe.get('true_cfg_scale')),
                  selected_edge_margin_pixels=task.get('selected_edge_margin_pixels',0),
                  original_background_instruction=recipe.get('background_instruction'))
    return result
