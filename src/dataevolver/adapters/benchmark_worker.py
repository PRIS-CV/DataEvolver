"""Local native scoring ABI; retains benchmark prompts and token probabilities."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import platform
from importlib import metadata

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from dataevolver.harness.state_store import atomic_write_json
from .model_contracts import ModelRequest
from .training_media_worker import TrainingMediaProvider, deployment_observation

SCHEMA = 'dataevolver.generation_benchmark_worker.v1'
MODELS = {'spatial': 'Qwen/Qwen2.5-VL-72B-Instruct', 'geneval2': 'Qwen/Qwen3-VL-8B-Instruct'}


def benchmark_model(config):
    """Select an explicit frozen evaluator while retaining historical defaults."""
    kind = config.get('kind')
    if kind not in MODELS:
        raise ValueError('unsupported benchmark kind')
    selected = config.get('model', MODELS[kind])
    allowed = (set(MODELS.values()) if kind == 'spatial' else {MODELS['geneval2']})
    if not isinstance(selected, str) or selected not in allowed:
        raise ValueError('unsupported frozen benchmark model')
    return selected


def answer_probabilities(logits, ids):
    """The official token-variant sum, with stable normalization and raw audit."""
    import math
    import numpy as np
    values=np.asarray(logits,dtype=np.float64)
    if values.ndim!=1 or not ids or len(set(ids))!=len(ids) or any(type(i) is not int or not 0<=i<len(values) for i in ids):
        raise ValueError('answer variants need distinct, valid first-token IDs')
    if np.isnan(values).any() or np.isposinf(values).any() or not np.isfinite(values).any():
        raise ValueError('answer logits are invalid')
    weights=np.exp(values-np.max(values));denominator=math.fsum(weights.tolist())
    probabilities=[float(weights[i]/denominator) for i in ids]
    raw=math.fsum(probabilities)
    if raw>1+8*len(ids)*np.finfo(np.float64).eps:
        raise ValueError('answer probability exceeds normalized token mass')
    return min(1.,raw),probabilities,raw


def validate_worker(config):
    if (config.get('schema_version') != SCHEMA or config.get('kind') not in MODELS
            or config.get('precision') != 'bf16' or config.get('execution_device') != 'cuda'
            or config.get('implementation_sha256') != file_sha256(Path(__file__))
            or not config.get('file_hashes') or not config.get('runtime_versions')
            or config.get('max_images') != 1 or config['resource_profile']['cpu_offload']):
        raise ValueError('invalid frozen benchmark deployment')
    benchmark_model(config)
    devices = config.get('gpu_uuid', '').split(',')
    if len(devices) not in {1, 2} or len(set(devices)) != len(devices) or any(not d.startswith('GPU-') for d in devices):
        raise ValueError('benchmark deployment must freeze its physical GPU')
    loading = config.get('loading', {})
    if loading:
        if (set(loading) != {'max_memory_mib', 'min_free_mib'} or len(devices) != 2
                or any(len(loading[k]) != 2 for k in loading)
                or any(type(v) is not int or v <= 0 for values in loading.values() for v in values)
                or any(a >= b for a, b in zip(loading['max_memory_mib'], loading['min_free_mib']))):
            raise ValueError('benchmark sharding must retain activation memory on each GPU')
    if config.get('resident'):
        from .training_media_resident import validate_resident_config
        validate_resident_config(config)


def verify_assets(config):
    root = Path(os.environ[config['model_path_env']]).resolve()
    files = {p.relative_to(root).as_posix() for p in root.rglob('*.safetensors')}
    files |= {p.relative_to(root).as_posix() for p in root.iterdir()
              if p.is_file() and p.suffix in {'.json','.txt','.jinja','.model'}}
    if not files or not any(p.endswith('.safetensors') for p in files) or not files <= set(config['file_hashes']):
        raise ValueError('benchmark assets are not completely frozen')
    for relative, digest in config['file_hashes'].items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != digest:
            raise ValueError('benchmark model/processor bytes changed')
    script = Path(os.environ[config['benchmark_script_env']])
    if file_sha256(script) != config['benchmark_script_sha256']:
        raise ValueError('official benchmark source changed')
    return str(root)


def load_components(config, root):
    selected = benchmark_model(config)
    checkpoint = json.loads((Path(root)/'config.json').read_text())
    expected_type = 'qwen2_5_vl' if selected == MODELS['spatial'] else 'qwen3_vl'
    if checkpoint.get('model_type') != expected_type or checkpoint.get('quantization_config') is not None:
        raise ValueError('benchmark checkpoint architecture or quantization differs from its frozen BF16 model')
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration
    cls = Qwen2_5_VLForConditionalGeneration if expected_type == 'qwen2_5_vl' else Qwen3VLForConditionalGeneration
    placement = ({'device_map': 'balanced', 'max_memory': {i: str(v)+'MiB'
                  for i, v in enumerate(config['loading']['max_memory_mib'])}}
                 if config.get('loading') else {'device_map': {'': 0}})
    model, info = cls.from_pretrained(root, dtype=torch.bfloat16, local_files_only=True,
        **placement, attn_implementation='sdpa', output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('benchmark checkpoint loading was incomplete')
    if any(str(v) in {'cpu', 'disk', 'meta'} for v in getattr(model, 'hf_device_map', {}).values()):
        raise ValueError('benchmark weights cannot be offloaded')
    return {'model':model.eval(), 'processor':AutoProcessor.from_pretrained(root, local_files_only=True)}


def bound_input_image(payload, request):
    """Read only the exact hashed input from a host-registered artifact root."""
    if len(payload['inputs']) != 1 or payload['inputs'][0]['ref'] != request.artifacts[0].to_dict():
        raise ValueError('benchmark input inventory differs from the request')
    ref = ArtifactRef.from_dict(payload['inputs'][0]['ref'])
    path = Path(payload['inputs'][0]['path']).resolve()
    roots = {payload['dataset_store']: Path(payload['dataset_root']).resolve()}
    roots.update({name: Path(os.environ[env]).resolve()
                  for name, env in payload['config'].get('input_store_envs', {}).items()})
    if payload['config'].get('resident'):
        from .training_media_resident import identity
        frozen = identity(payload['config'])['input_store_roots']
        if any(str(roots[name]) != value for name, value in frozen.items()):
            raise ValueError('native benchmark input stores differ from the loaded host identity')
    if ref.store not in roots or path != (roots[ref.store]/ref.path).resolve() or not path.is_relative_to(roots[ref.store]):
        raise ValueError('benchmark input escaped the registered store')
    if file_sha256(path) != ref.sha256:
        raise ValueError('benchmark input image bytes changed')
    return path


def spatial_prompt(script, questions):
    if len(questions) != 10:
        raise ValueError('SpatialGenEval requires all ten ordered questions')
    for node in ast.parse(Path(script).read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='vlm_content_template' for t in node.targets):
            template = ast.literal_eval(node.value)
            return template.replace('##Multiple-Choice Questions##','\n'.join(q.strip() for q in questions))
    raise ValueError('pinned official SpatialGenEval prompt template is absent')


class LocalBenchmarkProvider(TrainingMediaProvider):
    worker_module = 'dataevolver.adapters.benchmark_worker'

    def __init__(self, profile, config, store, dataset_store, receipt_root):
        validate_worker(config)
        if (profile.backend != 'local_python' or profile.model != benchmark_model(config)
                or profile.revision != 'sha256:'+payload_sha256(config)):
            raise ValueError('benchmark model/profile identity mismatch')
        self.profile, self.config, self.store = profile, config, store
        self.dataset_store, self.root = dataset_store, Path(receipt_root)

    def invoke(self, request, request_hash):
        if request.role != 'benchmark.'+self.config['kind'] or len(request.artifacts) != 1:
            raise ValueError('benchmark role/image binding mismatch')
        if not request.artifacts[0].media_type.startswith('image/'):
            raise ValueError('benchmark input is not an image')
        return super().invoke(request, request_hash)


def run_payload(path, *, resident_models=None, resident_status=None):
    import torch
    from PIL import Image
    payload = json.loads(Path(path).read_text())
    config = payload['config']; validate_worker(config)
    request = ModelRequest.from_dict(payload['request'])
    if request.role != 'benchmark.'+config['kind'] or len(request.artifacts) != 1:
        raise ValueError('benchmark request role/image mismatch')
    if (payload['binding'] != payload_sha256([payload['request_hash'],payload['profile'],config,payload['dataset_store']])
            or payload['profile']['model'] != benchmark_model(config)
            or payload['profile']['revision'] != 'sha256:'+payload_sha256(config)
            or os.environ.get('CUDA_VISIBLE_DEVICES') != config['gpu_uuid']):
        raise ValueError('native benchmark payload/deployment binding changed')
    for package, version in config['runtime_versions'].items():
        actual = platform.python_version() if package == 'python' else metadata.version(package)
        if actual != version: raise ValueError('benchmark runtime changed')
    entry = resident_models[config['kind']] if resident_models is not None else None
    root = entry['model_path'] if entry is not None else verify_assets(config)
    components = entry['components'] if entry is not None else load_components(config, root)
    model, processor = components['model'], components['processor']
    image_path = bound_input_image(payload, request)
    parameters = json.loads(request.messages[0]['content'])
    if config['kind'] == 'spatial':
        script = Path(os.environ[config['benchmark_script_env']])
        if file_sha256(script) != config['benchmark_script_sha256']:
            raise ValueError('official spatial scoring prompt changed')
        prompt = spatial_prompt(script, parameters['questions'])
    else:
        prompt = parameters['question']+' Answer in one word.'
    with Image.open(image_path) as opened:
        image = opened.convert('RGB')
    messages = [{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
    if config['kind']=='spatial': messages.insert(0,{'role':'system','content':'You are a professional image critic.'})
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt').to(model.device)
    torch.manual_seed(request.seed)
    with torch.inference_mode():
        if config['kind']=='spatial':
            tokens = model.generate(**inputs, do_sample=True, temperature=1., max_new_tokens=request.max_tokens)
            raw = processor.decode(tokens[0,inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
            output = {'raw_response':raw}
            output_tokens = tokens.shape[-1]-inputs['input_ids'].shape[-1]
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else (eos or [])
            finish_reason = 'stop' if int(tokens[0,-1]) in eos else 'length'
        else:
            result = model.generate(**inputs,max_new_tokens=1,do_sample=False,output_scores=True,return_dict_in_generate=True)
            logits=result.scores[0][0].detach().float().cpu().numpy()
            answer = parameters['answer']
            numbers = dict(zip(('one','two','three','four','five','six','seven','eight','nine','ten'),map(str,range(1,11))))
            numeric = numbers.get(answer,'other')
            variants = ([answer,answer.capitalize(),' '+answer,' '+answer.capitalize(),numeric,' '+numeric]
                        if parameters['question'].startswith('How many') else ['Yes','yes',' yes',' Yes'])
            ids = [processor.tokenizer.encode(v)[0] for v in variants]
            import numpy as np
            logits_path=Path(payload['native_output']).with_suffix('.logits.npy')
            np.save(logits_path,logits,allow_pickle=False)
            probability,probabilities,raw_sum=answer_probabilities(logits,ids)
            logits_evidence={'path':str(logits_path),'sha256':file_sha256(logits_path),'shape':list(logits.shape),
                'source_dtype':str(result.scores[0].dtype),'normalization':'float64_softmax','raw_variant_sum':raw_sum}
            output = {'atom_probability':probability,'answer_token_ids':ids,
                      'answer_token_probabilities':probabilities}
            output_tokens = 1
            # One-token probability extraction is complete after one forward.
            # Keep token-generation termination separately from task completion.
            finish_reason = 'stop'
    response = {'model':benchmark_model(config), 'revision':payload['profile']['revision'], 'output':output,
        'usage':{'images':1,'input_tokens':int(inputs['input_ids'].numel()),'output_tokens':int(output_tokens),
                 'cost':None},'finish_reason':finish_reason,'provider_request_id':request.request_id,
        'runtime':{'precision':'bf16','deployment':deployment_observation(config, resident_status=resident_status), 'resident':entry is not None,
                   'benchmark_script_sha256':config['benchmark_script_sha256'],
                   'input_image': request.artifacts[0].to_dict(),
                   'generation_finish_reason': finish_reason if config['kind']=='spatial' else 'length',
                   'native_forwards': 1,
                   **({'answer_logits':logits_evidence} if config['kind']=='geneval2' else {}),
                   'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated()}}
    atomic_write_json(Path(payload['native_output']),{'binding':payload['binding'],
                     'sha256':payload_sha256(response),'response':response})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--payload',required=True)
    run_payload(parser.parse_args().payload)
