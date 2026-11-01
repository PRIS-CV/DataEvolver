"""No models or tensors load in these startup-contract engineering fixtures."""
from types import SimpleNamespace

import pytest

from dataevolver.adapters.resident_components import sam_components, validate_sam_components, loading_report_json


def fixture():
    detector = object(); video = SimpleNamespace(detector_model=detector); processor = object()
    info = {'missing_keys': [], 'mismatched_keys': [], 'unexpected_keys': [], 'error_msgs': []}
    return processor, video, info


def test_loader_report_and_detector_references_are_preserved_without_loading_or_copying_models():
    processor, video, info = fixture()
    components = sam_components(processor=processor, video_model=video, loading_info=info)
    assert components['processor'] is processor and components['video_model'] is video
    assert components['model'] is video.detector_model and components['loading_info'] is info
    result = validate_sam_components(components)
    assert result['loader_report_present'] and result['new_model_loads'] == 0
    assert not result['live_component_mutation'] and not result['model_movement']


def test_exact_legacy_dictionary_missing_loader_report_is_rejected_before_inference():
    processor, video, _ = fixture()
    legacy = {'processor': processor, 'video_model': video, 'model': video.detector_model}
    with pytest.raises(ValueError, match='loading_info'): validate_sam_components(legacy)
    assert set(legacy) == {'processor', 'video_model', 'model'}


@pytest.mark.parametrize('info', [None, {}, {'missing_keys': []},
    {'missing_keys': ['weight'], 'mismatched_keys': []}, {'missing_keys': [], 'mismatched_keys': ['shape']},
    {'missing_keys': [], 'mismatched_keys': [], 'error_msgs': ['load error']}])
def test_missing_or_incomplete_loader_provenance_cannot_be_fabricated(info):
    processor, video, _ = fixture()
    with pytest.raises(ValueError): sam_components(processor=processor, video_model=video, loading_info=info)


def test_separate_detector_object_cannot_impersonate_the_loaded_model():
    processor, video, info = fixture()
    with pytest.raises(ValueError, match='exact detector'):
        validate_sam_components({'processor': processor, 'video_model': video, 'model': object(), 'loading_info': info})


def test_real_loader_set_inventories_are_json_serializable_without_fabricating_or_mutating_them():
    import json
    processor, video, _ = fixture()
    info = {'missing_keys': set(), 'mismatched_keys': [], 'unexpected_keys': {'second', 'first'}, 'details': (1, 2)}
    components = sam_components(processor=processor, video_model=video, loading_info=info)
    serialized = loading_report_json(info)
    assert json.loads(json.dumps(serialized)) == {'missing_keys': [], 'mismatched_keys': [],
        'unexpected_keys': ['first', 'second'], 'details': [1, 2]}
    assert components['loading_info'] is info and isinstance(info['missing_keys'], set)
    assert info['unexpected_keys'] == {'first', 'second'}


@pytest.mark.parametrize('value', [object(), float('nan'), {1: 'non-string key'}])
def test_unknown_loader_report_values_are_not_stringified_or_invented(value):
    with pytest.raises(ValueError): loading_report_json(value)
