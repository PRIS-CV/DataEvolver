"""Startup component contracts for already selected model implementations.

These helpers do not load, move, mutate or repair a live model. Loader metadata
must be retained at construction time; an absent loading report is not replaced
with an invented empty success report.
"""
from collections.abc import Mapping
import json
import math


COMPONENT_SCHEMA = 'dataevolver.resident_sam_components.v1'


def validate_sam_components(components):
    if not isinstance(components, Mapping):
        raise ValueError('SAM resident components must be an explicit mapping')
    required = {'processor', 'video_model', 'model', 'loading_info'}
    if not required <= set(components):
        raise ValueError('SAM resident component contract is missing: '+','.join(sorted(required-set(components))))
    info = components['loading_info']
    if (not isinstance(info, Mapping) or not {'missing_keys', 'mismatched_keys'} <= set(info)
            or any(not isinstance(info[k], (list, tuple, set)) for k in ('missing_keys', 'mismatched_keys'))):
        raise ValueError('SAM requires the actual loader report with missing/mismatched key fields')
    if info['missing_keys'] or info['mismatched_keys'] or info.get('error_msgs'):
        raise ValueError('SAM checkpoint loading was incomplete')
    if (components['processor'] is None or components['video_model'] is None
            or getattr(components['video_model'], 'detector_model', None) is not components['model']):
        raise ValueError('resident SAM detector must be the exact detector owned by the loaded video model')
    return {'schema_version': COMPONENT_SCHEMA, 'loader_report_present': True,
            'detector_object_identity_verified': True, 'new_model_loads': 0,
            'model_movement': False, 'live_component_mutation': False}


def sam_components(*, processor, video_model, loading_info):
    """Retain real loader output alongside the same already loaded objects."""
    components = {'processor': processor, 'video_model': video_model,
                  'model': video_model.detector_model, 'loading_info': loading_info}
    validate_sam_components(components)
    return components


def loading_report_json(value):
    """Serialize the real loader report without altering the live component map.

    Transformers may return sets for key inventories. Retain their contents as
    sorted arrays for status JSON; never replace absent metadata with success.
    """
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError('loader report keys must be strings')
        return {k: loading_report_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [loading_report_json(v) for v in value]
        return sorted(items, key=lambda v: json.dumps(v, sort_keys=True)) if isinstance(value, (set, frozenset)) else items
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError('loader report contains an unsupported non-JSON value')
