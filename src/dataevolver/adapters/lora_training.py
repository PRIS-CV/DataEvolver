"""Training exposure and proof checks for a real external LoRA runner."""
from __future__ import annotations

import json
import math
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256


SCHEMA = 'dataevolver.lora_training.v1'


def validate_training_config(config):
    if config.get('schema_version') != SCHEMA:
        raise ValueError('unregistered LoRA training input contract')
    integers = ('unique_samples', 'repeats', 'num_train_epochs', 'num_processes',
                'train_batch_size', 'gradient_accumulation_steps', 'max_train_steps', 'resolution', 'rank', 'lora_alpha')
    if any(type(config.get(k)) is not int or config[k] < 1 for k in integers):
        raise ValueError('training exposure fields must be positive integers')
    if (config['num_processes'] != 1 or config.get('mixed_precision') != 'bf16'
            or config.get('image_column') != 'image' or config.get('caption_column') != 'caption'
            or config.get('random_flip') is not False or config.get('caption_dropout') != 0
            or config.get('destructive_crop') is not False or config.get('dataloader_num_workers') != 0):
        raise ValueError('training conditioning/geometry/process contract changed')
    exposure = config['unique_samples']*config['repeats']*config['num_train_epochs']
    batch = config['num_processes']*config['train_batch_size']*config['gradient_accumulation_steps']
    if exposure % batch or config['max_train_steps'] != exposure//batch:
        raise ValueError('optimizer steps and effective global batch do not match exposure')
    if (type(config.get('learning_rate')) not in (int,float) or not math.isfinite(config['learning_rate'])
            or config['learning_rate'] <= 0):
        raise ValueError('training learning rate must be finite and positive')
    return {'unique_samples':config['unique_samples'], 'logical_exposures':exposure,
            'effective_global_batch':batch, 'optimizer_steps':exposure//batch}


def checkpoint_inventory(path):
    """Read the actual safetensors header, independent of torch/model availability."""
    path = Path(path)
    with path.open('rb') as stream:
        length = int.from_bytes(stream.read(8), 'little')
        if not 2 <= length <= 16*1024*1024 or length+8 > path.stat().st_size:
            raise ValueError('checkpoint has an invalid safetensors header')
        header = json.loads(stream.read(length))
    tensors = {k:v for k,v in header.items() if k != '__metadata__'}
    sizes = {'F32':4,'F16':2,'BF16':2,'F64':8}
    if (not tensors or not any('lora_A' in k or 'lora_B' in k for k in tensors)
            or any('lora_' not in k for k in tensors)):
        raise ValueError('checkpoint is not an identifiable LoRA adapter')
    intervals = []
    for name, item in tensors.items():
        shape, dtype, offsets = item['shape'], item['dtype'], item['data_offsets']
        if (dtype not in sizes or not shape or any(type(n) is not int or n < 1 for n in shape)
                or len(offsets) != 2 or offsets[1]-offsets[0] != math.prod(shape)*sizes[dtype]):
            raise ValueError('checkpoint tensor shape/dtype/offset mismatch')
        intervals.append(tuple(offsets))
    end = 0
    for start, stop in sorted(intervals):
        if start != end or stop <= start:
            raise ValueError('checkpoint tensor data has a gap or overlap')
        end = stop
    if end != path.stat().st_size-8-length:
        raise ValueError('checkpoint tensor payload is incomplete')
    return {'sha256':file_sha256(path), 'tensors':tensors}


def validate_training_proof(job, raw, store):
    config = json.loads(store.resolve(job.training_config).read_text())
    expected = validate_training_config(config)
    proof_ref = ArtifactRef.from_dict(raw['training_proof'])
    proof = json.loads(store.resolve(proof_ref).read_text())
    dataset = json.loads(store.resolve(job.dataset).read_text())
    train_rows = [r for r in dataset['records'] if r['split'] == 'train']
    records = {r['id']:r for r in train_rows}
    if (len(records) != len(train_rows) or len(records) != expected['unique_samples']
            or set(proof['consumed']) != set(records)):
        raise ValueError('training did not consume the complete exact dataset')
    required_count = config['repeats']*config['num_train_epochs']
    for key, row in records.items():
        actual = proof['consumed'][key]
        if (type(actual['logical_count']) is not int or type(actual['physical_count']) is not int
                or actual['logical_count'] != required_count or actual['physical_count'] != required_count
                or actual['caption_sha256'] != payload_sha256(row['input']['prompt'])
                or actual['media_sha256'] != row['target']['image']['sha256']):
            raise ValueError('actual training image/caption/exposure differs from the frozen dataset')
    if (proof.get('job_sha256') != job.compute_hash() or proof.get('evidence_kind') != 'real_execution'
            or proof.get('optimizer_steps') != expected['optimizer_steps']
            or proof.get('effective_global_batch') != expected['effective_global_batch']
            or proof.get('finite_gradients') is not True or proof.get('reload_verified') is not True
            or proof.get('trainable_before_sha256') == proof.get('trainable_after_sha256')):
        raise ValueError('optimizer/update/reload evidence is incomplete')
    losses = proof.get('losses', [])
    if (len(losses) != expected['optimizer_steps'] * config['gradient_accumulation_steps']
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in losses)
            or type(proof.get('updated_parameter_count')) is not int or proof['updated_parameter_count'] < 1):
        raise ValueError('finite loss and actual parameter-update evidence is incomplete')
    for name in ('trainable_before_sha256','trainable_after_sha256'):
        value = proof.get(name)
        if not isinstance(value,str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('training parameter hash is invalid')
    checkpoint = checkpoint_inventory(store.resolve(ArtifactRef.from_dict(raw['checkpoint'])))
    if checkpoint['sha256'] != proof.get('reloaded_checkpoint_sha256'):
        raise ValueError('reloaded checkpoint differs from the delivered adapter')
    from .model_contracts import ModelRequest, ModelReceipt
    control = proof['reload_inference']
    request, receipt = ModelRequest.from_dict(control['request']), ModelReceipt.from_dict(control['receipt'])
    if (request.role != 'training.reload' or receipt.backend == 'mock' or receipt.model != job.model
            or receipt.request_id != request.request_id or receipt.finish_reason != 'stop'
            or receipt.request_hash != payload_sha256({'request': request.to_dict(), 'profile_hash': receipt.profile_hash})
            or receipt.without_reuse_hash() != control['receipt_hash']
            or receipt.output['reloaded_checkpoint_sha256'] != checkpoint['sha256']):
        raise ValueError('controlled reload inference is missing or bound to another checkpoint')
    image = store.resolve(ArtifactRef.from_dict(receipt.output['artifact']))
    native = json.loads(store.resolve(ArtifactRef.from_dict(control['native_receipt'])).read_text())
    if (control.get('fresh_process') is not True or native['request_hash'] != receipt.request_hash
            or native['response']['output'] != receipt.output
            or native['response']['revision'] != receipt.revision
            or native['response']['runtime'].get('fresh_process') is not True
            or native['response']['runtime']['pid'] == proof.get('training_pid')):
        raise ValueError('checkpoint reload was not verified in a fresh native process')
    from PIL import Image
    with Image.open(image) as opened:
        opened.load()
        if opened.size != (config['resolution'], config['resolution']):
            raise ValueError('reload inference output dimensions changed')
    return {**expected, 'proof_sha256':proof_ref.sha256, 'checkpoint_sha256':checkpoint['sha256']}
