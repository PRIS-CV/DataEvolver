"""CPU evidence for binding/recovery invariants; these are not model quality tests."""
import copy
import json
from pathlib import Path

import pytest

from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile, ModelRequest
from dataevolver.adapters.model_gateway import ModelGateway, ModelError
from dataevolver.adapters.visual_models import (
    MODELS, PRECISION, SCHEMA, LocalVisualProvider, inspect_assets, validate_visual_worker)
from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, canonical_json, file_sha256, payload_sha256
from dataevolver.dataset.visual_steps import media_source, segmentation_subjects
from dataevolver.harness.data_contracts import DataInput, DataNodeGraphV3, DataNodeSpecV3, DataRunBudget
from dataevolver.harness.models import EvidenceStatus, OperatorCall, StageStatus
from dataevolver.harness.operator_sdk import builtin_step_registry
from dataevolver.harness.plan_compiler import CompileContext, CompiledRun, builtin_domains, compile_plan
from dataevolver.harness.task_contracts import DatasetIntent, QualityPolicy, PROTECTED_GATES


def fixture_assets(root):
    root.mkdir()
    (root/'config.json').write_text(json.dumps({'architectures': ['Qwen3_5ForConditionalGeneration']}))
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'model-1.safetensors'}}))
    (root/'model-1.safetensors').write_bytes(b'fixture-not-a-real-model')
    (root/'system_prompt.txt').write_text('Released prompt fixture')
    # A PE directory can also contain an unrelated image model index.
    (root/'model_index.json').write_text('{"_class_name":"QwenImage21Pipeline"}')
    return root


def worker_config(tmp_path, kind='pe_t2i'):
    from dataevolver.adapters import local_visual_worker, training_media_worker, visual_models
    return {'schema_version': SCHEMA, 'kind': kind, 'model_path_env': 'VISUAL_MODEL', 'python_env': 'VISUAL_PYTHON',
        'file_hashes': {'config.json': '0'*64}, 'runtime_versions': {'torch': 'fixture', 'transformers': 'fixture', 'torchao': 'fixture'},
        'implementation_sha256': file_sha256(Path(local_visual_worker.__file__)),
        'binding_sha256': file_sha256(Path(visual_models.__file__)),
        'transport_sha256': file_sha256(Path(training_media_worker.__file__)),
        'threads': 1, 'max_images': 1, 'image_side': 1024, 'max_input_tokens': 1024, 'max_frames': 124,
        'execution_device': 'cuda', 'precision': PRECISION[kind], 'dependencies': {},
        'gpu_uuid': 'GPU-fixture',
        'resource_profile': {'min_free_mib': 8192, 'max_utilization': 25, 'cpu_offload': False, 'estimate_basis': 'fixture-only'}}


@pytest.mark.parametrize('role,kind', [('visual.image.edit', 'qwen21'), ('visual.prompt.i2i', 'pe_i2i')])
@pytest.mark.parametrize('count', [0, 1, 10, 11])
def test_multi_reference_request_boundaries(tmp_path, role, kind, count):
    from dataevolver.adapters.visual_models import validate_visual_request
    config = worker_config(tmp_path, kind)
    config['max_images'] = 10
    parameters = {'prompt': 'fixture'}
    if kind == 'qwen21':
        parameters.update(width=512, height=512, steps=1, true_cfg_scale=1.0)
    request = ModelRequest('multi-ref', role, ({'role': 'user', 'content': canonical_json(parameters)},),
        {'type': 'object'}, artifacts=tuple(ArtifactRef('source', f'{i}.png', 'a'*64, 'image/png') for i in range(count)))
    if count in (1, 10):
        validate_visual_request(request, config)
    else:
        with pytest.raises(ValueError):
            validate_visual_request(request, config)


def test_pe_i2i_forwards_all_reference_images_in_order(tmp_path, monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    from PIL import Image
    from dataevolver.adapters.local_visual_worker import _pe
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext))
    (tmp_path / 'system_prompt.txt').write_text('fixture system prompt')
    images = []
    for i in range(2):
        path = tmp_path / f'{i}.png'
        Image.new('RGB', (4, 4), (i, 0, 0)).save(path)
        images.append({'path': str(path)})
    captured = {}
    class Inputs(dict):
        def to(self, device):
            return self
    class Generated:
        def __getitem__(self, key):
            return [7]
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            captured['content'] = messages[1]['content']
            return Inputs(input_ids=SimpleNamespace(shape=(1, 2)), attention_mask=SimpleNamespace(sum=lambda: 2))
        def decode(self, *args, **kwargs):
            return '{"rewritten_prompt":"fixture","ratio_follow":"<image1>","wh_ratio":""}'
    components = {'processor': Processor(), 'model': SimpleNamespace(generate=lambda **kw: Generated(),
                   generation_config=SimpleNamespace(eos_token_id=7))}
    payload = {'config': {'kind': 'pe_i2i', 'max_input_tokens': 64}, 'inputs': images,
               'model_path': str(tmp_path), 'request': {'max_tokens': 32}, 'native_output': str(tmp_path / 'native.json')}
    _, usage, finish = _pe(payload, {'prompt': 'fixture edit'}, {}, components)
    assert [item['image'].getpixel((0, 0)) for item in captured['content'] if item['type'] == 'image'] == [(0, 0, 0), (1, 0, 0)]
    assert usage['images'] == 2 and finish == 'stop'


def test_pe_inspection_uses_its_language_config_even_with_image_pipeline_files(tmp_path):
    root = fixture_assets(tmp_path/'pe')
    value = inspect_assets('pe_t2i', root)
    assert value['model'] == MODELS['pe_t2i'] and value['weight_files'] == ['model-1.safetensors']
    assert not value['inference_performed'] and not value['weights_hashed'] and not value['execution_ready']
    frozen = inspect_assets('pe_t2i', root, hash_weights=True)
    assert frozen['file_hashes']['model-1.safetensors'] == file_sha256(root/'model-1.safetensors')
    assert not frozen['execution_ready']


@pytest.mark.parametrize('shard', ['missing.safetensors', '../outside.safetensors'])
def test_incomplete_or_escaped_shards_are_rejected(tmp_path, shard):
    root = fixture_assets(tmp_path/'pe')
    (tmp_path/'outside.safetensors').write_bytes(b'outside')
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': shard}}))
    with pytest.raises(ValueError, match='missing or escaped'): inspect_assets('pe_t2i', root)


def test_wrong_precision_role_and_changed_worker_rejected_before_model_loading(tmp_path):
    config = worker_config(tmp_path); validate_visual_worker(config)
    with pytest.raises(ValueError, match='precision'): validate_visual_worker({**config, 'precision': 'bf16'})
    with pytest.raises(ValueError, match='implementation'): validate_visual_worker({**config, 'implementation_sha256': '0'*64})
    with pytest.raises(ValueError, match='transport'): validate_visual_worker({**config, 'transport_sha256': '0'*64})
    profile = ModelProfile('pe', 'local_python', MODELS['pe_t2i'], 'sha256:'+payload_sha256(config))
    provider = LocalVisualProvider(profile, config, ArtifactStore({'dataset': tmp_path/'dataset'}), 'dataset', tmp_path/'native')
    wrong = ModelRequest('wrong', 'visual.prompt.i2i', ({'role': 'user', 'content': '{"prompt":"x"}'},), {'type': 'object'})
    with pytest.raises(ValueError, match='wrong model'): provider.invoke(wrong, '0'*64)
    assert not provider.root.exists()


def gateway_fixture(tmp_path, *, observation_status='pass', wrong_image=False, wrong_prompt=False):
    store = ArtifactStore({'dataset': tmp_path/'data'})
    root = store.roots['dataset']; root.mkdir()
    from PIL import Image
    Image.new('RGB', (16, 16), 'red').save(root/'image.png')
    ref = ArtifactRef('dataset', 'image.png', file_sha256(root/'image.png'), 'image/png')
    generation = {'prompt': 'one red ball', 'width': 16, 'height': 16, 'steps': 1, 'true_cfg_scale': 1}
    generated = {'artifact': ref.to_dict(), 'actual_parameters': generation, 'image_size': [16, 16]}
    observed = {'status': observation_status, 'reason': 'CPU fixture', 'coordinate_system': 'normalized_0_1',
        'subjects': [{'subject_id': 's1', 'target_instance_id': 'object-1', 'description': 'red ball',
                      'box_xyxy_normalized': [.1, .1, .9, .9]}]}
    profiles = (ModelProfile('generate', 'mock', 'fixture-image', '1', capabilities=('json', 'vision'), mock_output=generated),
        ModelProfile('observe', 'mock', 'fixture-vision', '1', capabilities=('json', 'vision'), mock_output=observed))
    gateway = ModelGateway(profiles=profiles, policy=ModelPolicy({'visual.image.generate': 'generate', 'visual.inspect': 'observe'}, 2),
                           receipt_root=tmp_path/'receipts', artifact_store=store)
    gen = ModelRequest('gen', 'visual.image.generate', ({'role': 'user', 'content': canonical_json(generation)},), {'type': 'object'})
    rec = gateway.invoke(gen)
    media = {'request': gen.to_dict(), 'receipt_hash': rec.without_reuse_hash(), 'output': rec.output}
    context = {'generation_prompt': 'wrong' if wrong_prompt else generation['prompt'],
               'generation_request_sha256': gen.compute_hash(), 'subject_ids': ['s1']}
    image = ref
    if wrong_image:
        Image.new('RGB', (16, 16), 'blue').save(root/'other.png')
        image = ArtifactRef('dataset', 'other.png', file_sha256(root/'other.png'), 'image/png')
    obs = ModelRequest('obs', 'visual.inspect', ({'role': 'user', 'content': canonical_json({
        'prompt': 'Observe actual media.\nMEDIA_CONTEXT='+canonical_json(context)})},), {'type': 'object'}, (image,))
    rec = gateway.invoke(obs)
    observation = {'request': obs.to_dict(), 'receipt_hash': rec.without_reuse_hash(), 'output': rec.output}
    return gateway, media, observation, ref


def test_prompt_linkage_reuses_confirmed_effects_with_no_new_request(tmp_path):
    gateway, media, observation, ref = gateway_fixture(tmp_path)
    before = sorted((p.name, p.read_bytes()) for p in gateway.root.glob('*.json'))
    assert segmentation_subjects(gateway, media, observation)[0] == ref
    assert media_source(gateway, media) == (ref, 'one red ball')
    assert sorted((p.name, p.read_bytes()) for p in gateway.root.glob('*.json')) == before


def source_pe_graph(tmp_path,*,with_source=True):
    source,media,_,image=gateway_fixture(tmp_path)
    raw=tmp_path/'raw';raw.mkdir()
    (raw/'media.json').write_text(json.dumps(media));(raw/'prompt.json').write_text('{"prompt":"Change the red ball to blue."}')
    refs={n:ArtifactRef('raw',n+'.json',file_sha256(raw/(n+'.json'))) for n in ('media','prompt')}
    profile=ModelProfile('pe','mock','fixture-pe','1',capabilities=('json','vision'),
        mock_output={'rewritten_prompt':'A blue ball.','wh_ratio':'','ratio_follow':'1'})
    node=DataNodeSpecV3('refine','visual.prompt.i2i','1','model',
        {k:DataInput(artifact=v) for k,v in refs.items()},
        {'seed':0,'max_tokens':128,'timeout_seconds':2},'visual.prompt.i2i',1,1)
    contract={'intent':DatasetIntent('reuse','curate','visual_generation','prompt.v1','CPU source-role fixture',tuple(refs.values())).to_dict(),
        'quality_policy':QualityPolicy('reuse',tuple(sorted(PROTECTED_GATES)),(node.operator,),{},1).to_dict(),
        'graph':DataNodeGraphV3((node,)).to_dict(),'budget':DataRunBudget(1,1,1,1,2**20,30).to_dict(),
        'stores':{'raw':'RAW','dataset':'SOURCE_DATA','output':'OUTPUT_DATA'},'dataset_store':'output',
        'models':{'profiles':[profile.to_dict()],'policy':ModelPolicy({'visual.prompt.i2i':'pe'},1).to_dict(),
                  'workers':{},'ledger_prefix':'gateway'}}
    roots={'raw':raw,'dataset':source.store.roots['dataset'],'output':tmp_path/'output'}
    registry=builtin_step_registry()
    compiled=compile_plan({'domain':'visual_generation','contract':contract},CompileContext(
        registry,builtin_domains(),(node.operator,),roots=roots))
    assert isinstance(compiled,CompiledRun)
    op=compiled.bind(registry,allowed_write_root=tmp_path/'runtime',roots=roots,
        source_gateways={'visual.image.generate':source} if with_source else None)
    call=OperatorCall('reuse','refine','refine','data.graph','1',0,str(tmp_path/'runtime/call'),parameters=contract)
    return source,media,op,call


def test_new_graph_reads_old_confirmed_generation_without_old_effect_or_cost(tmp_path,monkeypatch):
    source,media,op,call=source_pe_graph(tmp_path)
    before={p:p.read_bytes() for p in source.root.glob('*.json')}
    monkeypatch.setattr(source,'_provider',lambda *_:pytest.fail('source reader attempted repair/load'))
    receipt=op.execute(call)
    assert receipt.status is StageStatus.SUCCESS and op.verify_effect(call,receipt).status is EvidenceStatus.VERIFIED
    assert op.model_usage(call)['model_calls']==1 and source.audit_usage()['model_calls']==2
    assert op.gateway(call).audit_usage()['model_calls']==3
    assert op.recover(call).receipt.reused and op.model_usage(call)['model_calls']==1
    assert before=={p:p.read_bytes() for p in source.root.glob('*.json')}
    assert op.gateway(call).health('visual.image.generate')['status']=='READ_ONLY_CACHE'
    with pytest.raises(ModelError,match='cannot dispatch'):
        op.gateway(call).invoke(ModelRequest.from_dict(media['request']))
    assert before=={p:p.read_bytes() for p in source.root.glob('*.json')}


def test_new_graph_without_source_gateway_still_refuses_old_role(tmp_path):
    source,_,op,call=source_pe_graph(tmp_path,with_source=False)
    receipt=op.execute(call)
    assert receipt.status is StageStatus.FAILED and 'role is not registered' in receipt.stage_result['failure_detail']
    assert op.model_usage(call)['model_calls']==0 and source.audit_usage()['model_calls']==2


@pytest.mark.parametrize('suffix',['.intent.json','.receipt.json'])
def test_source_role_cannot_repair_missing_local_completion(tmp_path,monkeypatch,suffix):
    source,media,op,call=source_pe_graph(tmp_path)
    request=ModelRequest.from_dict(media['request']);path=source.root/(request.request_id+suffix);path.unlink()
    monkeypatch.setattr(source,'_provider',lambda *_:pytest.fail('source cache repair attempted'))
    with pytest.raises(ModelError,match='no repair'):
        op.gateway(call).recover(request)
    assert not path.exists() and op.model_usage(call)['model_calls']==0


@pytest.mark.parametrize('change',['profile','root'])
def test_source_gateway_contract_or_ledger_drift_rejected_before_effect(tmp_path,change):
    source,_,op,call=source_pe_graph(tmp_path)
    original=op.spec.model_config_hash
    if change=='profile':
        from dataclasses import replace
        source.profiles['generate']=replace(source.profiles['generate'],revision='changed')
    else:source.root=tmp_path/'different-ledger'
    with pytest.raises(ValueError,match='frozen source gateway binding changed'):
        op.gateway(call)
    assert op.spec.model_config_hash==original and op.model_usage(call)['model_calls']==0


@pytest.mark.parametrize('kwargs,reason', [({'wrong_image': True}, 'another media'),
    ({'wrong_prompt': True}, 'prompt binding'), ({'observation_status': 'uncertain'}, 'uncertain')])
def test_wrong_image_prompt_or_uncertain_observation_rejected(tmp_path, kwargs, reason):
    gateway, media, observation, _ = gateway_fixture(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=reason): segmentation_subjects(gateway, media, observation)


def test_forged_observation_and_changed_media_rejected(tmp_path):
    gateway, media, observation, ref = gateway_fixture(tmp_path)
    forged = copy.deepcopy(observation); forged['output']['subjects'][0]['description'] = 'forged'
    with pytest.raises(ValueError, match='native effect'): segmentation_subjects(gateway, media, forged)
    gateway.store.resolve(ref).write_bytes(b'changed')
    with pytest.raises((ValueError, ModelError)): segmentation_subjects(gateway, media, observation)


def test_visual_generation_compiles_executes_and_recovers_through_existing_graph(tmp_path):
    raw, dataset = tmp_path/'raw', tmp_path/'dataset'; raw.mkdir(); dataset.mkdir()
    prompt = raw/'prompt.json'; prompt.write_text('{"prompt":"one red ball"}')
    source = ArtifactRef('raw', prompt.name, file_sha256(prompt))
    from PIL import Image
    Image.new('RGB', (16, 16), 'red').save(dataset/'image.png')
    image = ArtifactRef('dataset', 'image.png', file_sha256(dataset/'image.png'), 'image/png')
    sampling = {'width': 16, 'height': 16, 'steps': 1, 'true_cfg_scale': 1}
    profile = ModelProfile('image', 'mock', 'fixture-image', '1', mock_output={
        'artifact': image.to_dict(), 'actual_parameters': {**sampling, 'prompt': 'one red ball'}, 'image_size': [16, 16]})
    node = DataNodeSpecV3('generate', 'visual.image.generate', '1', 'model', {'prompt': DataInput(artifact=source)},
        {'sampling': sampling, 'seed': 1, 'max_tokens': 64, 'timeout_seconds': 2}, 'visual.image.generate', 1, 0)
    contract = {'intent': DatasetIntent('visual', 'generate', 'image', 'image.v1', 'CPU binding fixture', (source,)).to_dict(),
        'quality_policy': QualityPolicy('visual', tuple(sorted(PROTECTED_GATES)), (node.operator,), {}, 1).to_dict(),
        'graph': DataNodeGraphV3((node,)).to_dict(), 'budget': DataRunBudget(1, 1, 0, 1, 1000000, 20).to_dict(),
        'stores': {'raw': 'RAW', 'dataset': 'DATASET'}, 'dataset_store': 'dataset',
        'models': {'profiles': [profile.to_dict()], 'policy': ModelPolicy({'visual.image.generate': 'image'}, 1).to_dict(), 'workers': {}}}
    roots = {'raw': str(raw), 'dataset': str(dataset)}
    registry = builtin_step_registry()
    context = CompileContext(registry, builtin_domains(), (node.operator,), roots=roots)
    compiled = compile_plan({'domain': 'visual_generation', 'contract': contract}, context)
    assert isinstance(compiled, CompiledRun)
    operator = compiled.bind(registry, allowed_write_root=tmp_path/'runtime', roots=roots)
    call = OperatorCall('visual-fixture', 'build', 'visual-key', operator.spec.operator_id, operator.spec.version,
                        0, str(tmp_path/'runtime/build'), parameters=compiled.contract)
    receipt = operator.execute(call)
    assert operator.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    before = operator.model_usage(call)
    reused = operator.recover(call)
    assert reused.receipt.reused and operator.model_usage(call) == before
    assert before['model_calls'] == 1


def test_observation_and_segmentation_graph_recovers_after_native_effect_before_receipt(tmp_path, monkeypatch):
    raw, dataset = tmp_path/'raw', tmp_path/'dataset'; raw.mkdir(); dataset.mkdir()
    prompt = raw/'prompt.json'; prompt.write_text('{"prompt":"one red ball"}')
    source = ArtifactRef('raw', prompt.name, file_sha256(prompt))
    from PIL import Image
    Image.new('RGB', (16, 16), 'red').save(dataset/'image.png')
    Image.new('L', (16, 16), 255).save(dataset/'mask.png')
    image = ArtifactRef('dataset', 'image.png', file_sha256(dataset/'image.png'), 'image/png')
    mask = ArtifactRef('dataset', 'mask.png', file_sha256(dataset/'mask.png'), 'image/png')
    sampling = {'width': 16, 'height': 16, 'steps': 1, 'true_cfg_scale': 1}
    outputs = {
        'generate': {'artifact': image.to_dict(), 'actual_parameters': {**sampling, 'prompt': 'one red ball'}, 'image_size': [16, 16]},
        'observe': {'status': 'pass', 'reason': 'fixture', 'coordinate_system': 'normalized_0_1', 'subjects': [
            {'subject_id': 's1', 'target_instance_id': 'object-1', 'description': 'red ball', 'box_xyxy_normalized': [.1, .1, .9, .9]}]},
        'segment': {'instances': [{'subject_id': 's1', 'target_instance_id': 'object-1', 'mask': mask.to_dict()}]},
    }
    profiles = [ModelProfile(name, 'mock', 'fixture-'+name, '1', capabilities=('json', 'vision'), mock_output=value)
                for name, value in outputs.items()]
    common = {'seed': 1, 'max_tokens': 128, 'timeout_seconds': 2}
    nodes = (
        DataNodeSpecV3('generate', 'visual.image.generate', '1', 'model', {'prompt': DataInput(artifact=source)},
            {**common, 'sampling': sampling}, 'visual.image.generate', 1, 0),
        DataNodeSpecV3('observe', 'visual.observe.image', '1', 'model', {'media': DataInput(node='generate', output='media')},
            {**common, 'subject_ids': ['s1']}, 'visual.inspect', 1, 1),
        DataNodeSpecV3('segment', 'visual.segment.image', '1', 'model',
            {'media': DataInput(node='generate', output='media'), 'observation': DataInput(node='observe', output='observation')},
            common, 'visual.segment.image', 1, 1))
    contract = {'intent': DatasetIntent('visual', 'generate', 'image', 'image.v1', 'CPU lifecycle fixture', (source,)).to_dict(),
        'quality_policy': QualityPolicy('visual', tuple(sorted(PROTECTED_GATES)), tuple(n.operator for n in nodes), {}, 3).to_dict(),
        'graph': DataNodeGraphV3(nodes).to_dict(), 'budget': DataRunBudget(3, 3, 2, 1, 1000000, 30).to_dict(),
        'stores': {'raw': 'RAW', 'dataset': 'DATASET'}, 'dataset_store': 'dataset',
        'models': {'profiles': [p.to_dict() for p in profiles], 'policy': ModelPolicy({
            'visual.image.generate': 'generate', 'visual.inspect': 'observe', 'visual.segment.image': 'segment'}, 3).to_dict(), 'workers': {}}}
    registry = builtin_step_registry(); roots = {'raw': str(raw), 'dataset': str(dataset)}
    compiled = compile_plan({'domain': 'visual_generation', 'contract': contract},
        CompileContext(registry, builtin_domains(), tuple(n.operator for n in nodes), roots=roots))
    assert isinstance(compiled, CompiledRun)
    operator = compiled.bind(registry, allowed_write_root=tmp_path/'runtime', roots=roots)
    call = OperatorCall('visual-fixture', 'build', 'key', 'data.graph', '1', 0,
                        str(tmp_path/'runtime/build'), parameters=compiled.contract)
    from dataevolver.harness.adapters import data_graph
    original, crashes = data_graph._write_receipt, []
    def interrupt(path, receipt):
        if receipt.operator_id == 'visual.observe.image' and not crashes:
            crashes.append(True); raise RuntimeError('after native effect before receipt')
        return original(path, receipt)
    monkeypatch.setattr(data_graph, '_write_receipt', interrupt)
    with pytest.raises(RuntimeError, match='after native effect'): operator.execute(call)
    reopened = compiled.bind(registry, allowed_write_root=tmp_path/'runtime', roots=roots)
    receipt = reopened.execute(call)
    assert reopened.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    assert reopened.model_usage(call)['model_calls'] == 3
    assert reopened.execute(call).reused
    assert reopened.model_usage(call)['model_calls'] == 3


def test_precision_inspects_inner_storage_and_rejects_bf16_label(tmp_path):
    import torch
    from dataevolver.adapters.local_visual_worker import _precision
    model = torch.nn.Linear(16, 16, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='actual FP8'): _precision(model, {}, 'fixture')
    model.weight = torch.nn.Parameter(torch.zeros(16, 16, dtype=torch.float8_e4m3fn), requires_grad=False)
    runtime = {}; _precision(model, runtime, 'CPU storage fixture, not inference')
    assert runtime['precision']['storage_elements_by_dtype']['torch.float8_e4m3fn'] == 256


@pytest.mark.parametrize('status,criterion', [('pass', 'pass'), ('fail', 'fail'), ('uncertain', 'uncertain'), ('pass', 'fail')])
def test_quality_feedback_binds_generated_media_and_keeps_uncalibrated_judgments(tmp_path, status, criterion):
    from dataevolver.harness.feedback import FeedbackSignal
    raw, dataset = tmp_path/'raw', tmp_path/'dataset'; raw.mkdir(); dataset.mkdir()
    prompt = raw/'prompt.json'; prompt.write_text('{"prompt":"one red ball"}')
    source = ArtifactRef('raw', prompt.name, file_sha256(prompt))
    from PIL import Image
    Image.new('RGB', (16, 16), 'red').save(dataset/'image.png')
    image = ArtifactRef('dataset', 'image.png', file_sha256(dataset/'image.png'), 'image/png')
    sampling = {'width': 16, 'height': 16, 'steps': 1, 'true_cfg_scale': 1}
    profiles = [ModelProfile('image', 'mock', 'fixture-image', '1', mock_output={
        'artifact': image.to_dict(), 'actual_parameters': {**sampling, 'prompt': 'one red ball'}, 'image_size': [16, 16]}),
        ModelProfile('judge', 'mock', 'fixture-judge', '1', capabilities=('json', 'vision'), mock_output={
            'status': status, 'summary': 'fixture judgment', 'criteria': {'fidelity': {'status': criterion, 'reason': 'fixture'}}})]
    common = {'seed': 1, 'max_tokens': 128, 'timeout_seconds': 2}
    nodes = (DataNodeSpecV3('generate', 'visual.image.generate', '1', 'model', {'prompt': DataInput(artifact=source)},
        {**common, 'sampling': sampling}, 'visual.image.generate', 1, 0),
        DataNodeSpecV3('review', 'visual.review.image', '1', 'model', {'media': DataInput(node='generate', output='media')},
        {**common, 'dimensions': ['fidelity']}, 'visual.review.image', 1, 1))
    contract = {'intent': DatasetIntent('visual', 'generate', 'image', 'image.v1', 'CPU quality feedback fixture', (source,)).to_dict(),
        'quality_policy': QualityPolicy('visual', tuple(sorted(PROTECTED_GATES)), tuple(n.operator for n in nodes), {}, 2).to_dict(),
        'graph': DataNodeGraphV3(nodes).to_dict(), 'budget': DataRunBudget(2, 2, 1, 1, 1000000, 20).to_dict(),
        'stores': {'raw': 'RAW', 'dataset': 'DATASET'}, 'dataset_store': 'dataset',
        'models': {'profiles': [p.to_dict() for p in profiles], 'policy': ModelPolicy({
            'visual.image.generate': 'image', 'visual.review.image': 'judge'}, 2).to_dict(), 'workers': {}}}
    registry = builtin_step_registry(); roots = {'raw': str(raw), 'dataset': str(dataset)}
    compiled = compile_plan({'domain': 'visual_generation', 'contract': contract},
        CompileContext(registry, builtin_domains(), tuple(n.operator for n in nodes), roots=roots))
    assert isinstance(compiled, CompiledRun)
    operator = compiled.bind(registry, allowed_write_root=tmp_path/'runtime', roots=roots)
    call = OperatorCall('quality-fixture', 'build', 'key', 'data.graph', '1', 0,
                        str(tmp_path/'runtime/build'), parameters=compiled.contract)
    if status == 'pass' and criterion == 'fail':
        failed = operator.execute(call)
        assert failed.status is StageStatus.FAILED and failed.return_code == 1
        assert (failed.operator_id, failed.operator_version) == (operator.spec.operator_id, operator.spec.version)
        assert failed.stage_result['failure_code'] == 'INVALID_DATA_EFFECT'
        assert 'contradicts' in failed.stage_result['failure_detail']
        assert failed.stage_result['hard_constraints']['artifact_valid'] is False
        verification = operator.verify_effect(call, failed)
        assert verification.status is EvidenceStatus.VERIFIED
        assert 'candidate validation failed' in verification.reason
        assert not (Path(call.work_dir)/'effect.json').exists()
        assert not (Path(call.work_dir)/'node_runs/review/effect.json').exists()
        recovered = operator.recover(call)
        assert recovered.receipt.status is StageStatus.FAILED and recovered.receipt.reused
        assert operator.execute(call).reused
        assert operator.model_usage(call)['model_calls'] == 2
        return
    receipt = operator.execute(call)
    assert operator.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    effect = json.loads((Path(call.work_dir)/'node_runs/review/effect.json').read_text())
    signal_ref = ArtifactRef.from_dict(effect['outputs']['feedback'])
    signal = FeedbackSignal.from_dict(json.loads(operator.store.resolve(signal_ref).read_text()))
    assert signal.validate_evidence(operator.store)
    assert signal.kind == 'sample_quality' and signal.scope['data_sha256'] == image.sha256
    assert signal.uncertainty == 'unknown' and signal.scope['calibration_verified'] is False
    assert signal.metrics['status'] == status and signal.metrics['criteria']['fidelity']['status'] == criterion
    assert operator.model_usage(call)['model_calls'] == 2
