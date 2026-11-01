"""Frozen evaluator selection contracts; fixtures never load a real model."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataevolver.adapters import benchmark_worker as worker
from dataevolver.adapters.model_contracts import ModelProfile
from dataevolver.dataset.contracts import ArtifactStore, file_sha256, payload_sha256


def configuration(kind='spatial', model=None):
    value = {'schema_version': worker.SCHEMA, 'kind': kind, 'precision': 'bf16',
        'execution_device': 'cuda', 'implementation_sha256': file_sha256(Path(worker.__file__)),
        'file_hashes': {'config.json': '0'*64}, 'runtime_versions': {'python': 'fixture'},
        'max_images': 1, 'resource_profile': {'cpu_offload': False}, 'gpu_uuid': 'GPU-fixture'}
    if model is not None:
        value['model'] = model
    return value


@pytest.mark.parametrize('kind,selected', [
    ('spatial', 'Qwen/Qwen2.5-VL-72B-Instruct'),
    ('geneval2', 'Qwen/Qwen3-VL-8B-Instruct')])
def test_historical_default_evaluator_and_config_bytes_are_preserved(kind, selected):
    config = configuration(kind); before = copy.deepcopy(config)
    worker.validate_worker(config)
    assert worker.benchmark_model(config) == selected
    assert config == before and 'model' not in config


def test_spatial_explicit_qwen3vl8b_binds_exact_profile_and_new_config_hash(tmp_path):
    config = configuration(model='Qwen/Qwen3-VL-8B-Instruct')
    old = configuration(); worker.validate_worker(config)
    assert payload_sha256(config) != payload_sha256(old)
    store = ArtifactStore({'data': tmp_path})
    profile = ModelProfile('spatial', 'local_python', config['model'], 'sha256:'+payload_sha256(config))
    provider = worker.LocalBenchmarkProvider(profile, config, store, 'data', tmp_path/'native')
    assert provider.profile.model == 'Qwen/Qwen3-VL-8B-Instruct'
    wrong = ModelProfile('spatial', 'local_python', worker.MODELS['spatial'], profile.revision)
    with pytest.raises(ValueError, match='identity mismatch'):
        worker.LocalBenchmarkProvider(wrong, config, store, 'data', tmp_path/'native')
    wrong = ModelProfile('spatial', 'local_python', config['model'], 'sha256:'+payload_sha256(old))
    with pytest.raises(ValueError, match='identity mismatch'):
        worker.LocalBenchmarkProvider(wrong, config, store, 'data', tmp_path/'native')


@pytest.mark.parametrize('kind,selected', [
    ('geneval2', 'Qwen/Qwen2.5-VL-72B-Instruct'),
    ('spatial', 'Qwen/Qwen3-VL-8B-Thinking'),
    ('spatial', 'Qwen/Qwen3-VL-8B-Instruct-FP8'),
    ('spatial', 'unregistered/model'),
    ('unknown', 'Qwen/Qwen3-VL-8B-Instruct')])
def test_unapproved_model_or_task_change_is_rejected_before_loading(kind, selected):
    with pytest.raises(ValueError):
        worker.validate_worker(configuration(kind, selected))


@pytest.mark.parametrize('change', ['precision', 'offload'])
def test_smaller_model_keeps_bf16_and_no_offload_requirements(change):
    config = configuration(model='Qwen/Qwen3-VL-8B-Instruct')
    if change == 'precision': config['precision'] = 'fp8'
    else: config['resource_profile']['cpu_offload'] = True
    with pytest.raises(ValueError, match='invalid frozen benchmark'):
        worker.validate_worker(config)


@pytest.mark.parametrize('selected,model_type,expected_class', [
    ('Qwen/Qwen2.5-VL-72B-Instruct', 'qwen2_5_vl', 'qwen25'),
    ('Qwen/Qwen3-VL-8B-Instruct', 'qwen3_vl', 'qwen3')])
def test_loader_selects_architecture_from_frozen_model_not_benchmark_kind(
        tmp_path, monkeypatch, selected, model_type, expected_class):
    (tmp_path/'config.json').write_text(json.dumps({'model_type': model_type}))
    calls = []
    class Fake:
        hf_device_map = {'': 0}
        def eval(self): return self
    def loader(name):
        def load(root, **kwargs):
            calls.append((name, root, kwargs))
            return Fake(), {'missing_keys': [], 'mismatched_keys': [], 'error_msgs': []}
        return SimpleNamespace(from_pretrained=load)
    fake_transformers = SimpleNamespace(Qwen2_5_VLForConditionalGeneration=loader('qwen25'),
        Qwen3VLForConditionalGeneration=loader('qwen3'),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: 'processor'))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(bfloat16='BF16'))
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)
    result = worker.load_components(configuration(model=selected), str(tmp_path))
    assert len(calls) == 1 and calls[0][0] == expected_class
    assert calls[0][2]['dtype'] == 'BF16' and calls[0][2]['device_map'] == {'': 0}
    assert result['processor'] == 'processor'


@pytest.mark.parametrize('checkpoint', [
    {'model_type': 'qwen2_5_vl'},
    {'model_type': 'qwen3_vl', 'quantization_config': {'quant_method': 'fp8'}}])
def test_wrong_checkpoint_architecture_or_quantization_stops_before_model_import(tmp_path, checkpoint):
    (tmp_path/'config.json').write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match='checkpoint architecture or quantization'):
        worker.load_components(configuration(model='Qwen/Qwen3-VL-8B-Instruct'), str(tmp_path))


@pytest.mark.parametrize('selected', [None, 'Qwen/Qwen3-VL-8B-Instruct'])
def test_resident_identity_uses_exact_configured_spatial_model(tmp_path, monkeypatch, selected):
    from dataevolver.adapters import training_media_resident as resident
    from dataevolver.adapters.model_gateway import ModelError
    config = configuration(model=selected)
    config['model_path_env'] = 'FIXTURE_SPATIAL_MODEL'
    handler = Path(worker.__file__).resolve()
    document = {'models': {'spatial': {'model': worker.benchmark_model(config),
        'path': str(tmp_path), 'assets_sha256': payload_sha256(config['file_hashes'])}},
        'socket_path': str(tmp_path/'fixture.sock'), 'handler_path': str(handler),
        'handler_sha256': file_sha256(handler),
        'resident_implementation_sha256': file_sha256(Path(resident.__file__))}
    path = tmp_path/'identity.json'; path.write_text(json.dumps(document))
    config['resident'] = {'identity_env': 'FIXTURE_SPATIAL_IDENTITY',
        'socket_env': 'FIXTURE_SPATIAL_SOCKET', 'identity_sha256': file_sha256(path),
        'implementation_sha256': document['resident_implementation_sha256'],
        'activation_reserve_mib': [8192]}
    monkeypatch.setenv('FIXTURE_SPATIAL_MODEL', str(tmp_path))
    monkeypatch.setenv('FIXTURE_SPATIAL_IDENTITY', str(path))
    monkeypatch.setenv('FIXTURE_SPATIAL_SOCKET', document['socket_path'])
    worker.validate_worker(config)
    assert resident.identity(config) == document
    document['models']['spatial']['model'] = (
        worker.MODELS['spatial'] if selected else worker.MODELS['geneval2'])
    path.write_text(json.dumps(document))
    config['resident']['identity_sha256'] = file_sha256(path)
    with pytest.raises(ModelError, match='loaded model differs'):
        resident.identity(config)
