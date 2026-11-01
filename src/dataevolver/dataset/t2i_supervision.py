"""Text-to-image supervision and standalone training-input export."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from .contracts import ArtifactRef, canonical_json, file_sha256, payload_sha256


def supervision_split(value):
    """Translate catalog aliases without rewriting a persistent family split."""
    result={'dev':'validation','eval':'test'}.get(value,value)
    if result not in {'train','validation','test','pilot'}:
        raise ValueError('unsupported catalog split for supervision')
    return result


def validate_t2i_binding(row, store, *, require_accepted=True):
    inputs, target = row['input'], row['target']
    if set(inputs) != {'prompt'} or set(target) != {'image'}:
        raise ValueError('T2I supervision needs prompt -> image')
    prompt = inputs['prompt']
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('T2I caption must be the nonempty generation condition')
    image = ArtifactRef.from_dict(target['image'])
    if not image.media_type.startswith('image/'):
        raise ValueError('T2I target must be an image')
    condition = row['annotations']['conditioning']
    if condition != {'prompt_sha256': payload_sha256(prompt), 'image_sha256': image.sha256}:
        raise ValueError('training image/caption binding changed')
    generated, reviewed = False, False
    for entry in row['model_receipts']:
        request, receipt = entry['request'], entry['receipt']
        output = receipt['output']
        artifacts = [ArtifactRef.from_dict(a) for a in request['artifacts']]
        if request['role'] == 'visual.image.generate':
            actual = json.loads(request['messages'][0]['content'])
            if (ArtifactRef.from_dict(output['artifact']) == image and actual['prompt'] == prompt
                    and output['actual_parameters'] == actual):
                generated = True
        if request['role'] == 'visual.review.image' and artifacts == [image]:
            if output == row['annotations']['review'] and output.get('status') == 'pass':
                reviewed = True
    if not generated or (require_accepted and not reviewed):
        raise ValueError('T2I generation/review receipt is not bound to the training pair')
    store.resolve(image)


def from_visual_receipts(gateway, media, review, *, record_id, family_id, split, provenance):
    from .visual_steps import confirmed, media_source
    image, prompt = media_source(gateway, media)
    generation_request, generation = confirmed(gateway, media, {'visual.image.generate'})
    review_request, reviewed = confirmed(gateway, review, {'visual.review.image'})
    if review_request.artifacts != (image,) or reviewed.output.get('status') != 'pass':
        raise ValueError('T2I pair lacks a passing review of its actual image')
    return {'schema_version': 'dataevolver.supervision.v1', 'evidence_kind': 'real_execution',
        'task_type': 'text_to_image_generation', 'id': record_id, 'family_id': family_id,
        'split': split, 'status': 'pass', 'input': {'prompt': prompt}, 'target': {'image': image.to_dict()},
        'annotations': {'provenance': provenance, 'review': reviewed.output,
            'conditioning': {'prompt_sha256': payload_sha256(prompt), 'image_sha256': image.sha256}},
        'model_receipts': [{'request': q.to_dict(), 'receipt': r.to_dict(),
                            'receipt_hash': r.without_reuse_hash()}
                           for q, r in ((generation_request, generation), (review_request, reviewed))]}


def export_imagefolder(bundle, destination, *, split='train', allow_fixture=False):
    """Export actual bytes plus captions; preserve sample IDs for consumption audit."""
    from .supervision import load_dataset
    loaded = load_dataset(bundle, split=split, allow_fixture=allow_fixture)
    if not loaded or any(r['task_type'] != 'text_to_image_generation' for r, _ in loaded):
        raise ValueError('imagefolder export requires nonempty T2I supervision')
    destination = Path(destination)
    expected_source = file_sha256(Path(bundle)/'manifest.json')
    if destination.exists():
        result = audit_imagefolder(destination)
        if result['source_manifest_sha256'] != expected_source:
            raise ValueError('imagefolder belongs to a different supervision release')
        return result
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name+'.partial-', dir=destination.parent))
    directory = temporary/'train'
    directory.mkdir()
    records = []
    for row, store in loaded:
        image = ArtifactRef.from_dict(row['target']['image'])
        suffix = '.png' if image.media_type == 'image/png' else '.jpg' if image.media_type == 'image/jpeg' else None
        if suffix is None:
            raise ValueError('training imagefolder only supports PNG/JPEG bytes')
        name = row['id']+suffix
        shutil.copyfile(store.resolve(image), directory/name)
        records.append({'file_name': name, 'caption': row['input']['prompt'],
                        'sample_id': row['id'], 'media_sha256': image.sha256})
    (directory/'metadata.jsonl').write_text(''.join(canonical_json(r)+'\n' for r in records))
    (temporary/'training_manifest.json').write_text(canonical_json({
        'schema_version': 'dataevolver.t2i_training_input.v1', 'source_manifest_sha256': expected_source,
        'image_column': 'image', 'caption_column': 'caption', 'records': records}))
    audit_imagefolder(temporary)
    temporary.rename(destination)
    return audit_imagefolder(destination)


def audit_imagefolder(root):
    from PIL import Image
    root = Path(root).resolve()
    manifest = json.loads((root/'training_manifest.json').read_text())
    if manifest['schema_version'] != 'dataevolver.t2i_training_input.v1':
        raise ValueError('unknown training input contract')
    records = [json.loads(l) for l in (root/'train'/'metadata.jsonl').read_text().splitlines()]
    if records != manifest['records'] or len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('imagefolder metadata or sample membership changed')
    for row in records:
        path = (root/'train'/row['file_name']).resolve()
        if (not path.is_relative_to(root/'train') or file_sha256(path) != row['media_sha256']
                or not isinstance(row['caption'], str) or not row['caption'].strip()):
            raise ValueError('training image/caption integrity failed')
        with Image.open(path) as image:
            image.load()
    return {**manifest, 'training_manifest_sha256': file_sha256(root/'training_manifest.json'),
            'unique_samples': len(records)}
