"""Registered data steps around existing native visual envelopes."""
from .contracts import ArtifactRef, file_sha256


def catalog_from_media(ctx):
    from PIL import Image
    from .families import image_fingerprint
    from .visual_steps import media_source
    image, prompt = media_source(ctx.graph.gateway(ctx.parent), ctx.read('media'))
    with Image.open(ctx.store.resolve(image)) as opened:
        opened.load()
        fingerprint = image_fingerprint(opened)
        size = opened.size
    row = {'id':ctx.parameters['record_id'], 'image':image.to_dict(), 'prompt':prompt,
        'source_dataset':ctx.parameters['source_dataset'], 'source_family':ctx.parameters['source_family'],
        'width':size[0], 'height':size[1], **fingerprint}
    return {'catalog':ctx.write_json('catalog.json', {'rows':[row]})}


def supervision_value(ctx):
    from .t2i_supervision import from_visual_receipts, supervision_split
    from .visual_steps import media_source
    gateway = ctx.graph.gateway(ctx.parent)
    image, _ = media_source(gateway, ctx.read('media'))
    rows = [r for r in ctx.read('catalog')['rows'] if ArtifactRef.from_dict(r['image']) == image]
    if len(rows) != 1 or rows[0].get('family_status') != 'clean':
        raise ValueError('supervision requires exactly one clean persistent-family binding')
    row = rows[0]
    record = from_visual_receipts(gateway, ctx.read('media'), ctx.read('review'), record_id=row['id'],
        family_id=row['family_id'], split=supervision_split(row['split']), provenance={
            'source_dataset':row['source_dataset'], 'source_family':row['source_family'],
            'pixel_sha256':row['pixel_sha256'],'family_ledger_split':row['split']})
    from .supervision import validate_record
    validate_record(record, ctx.store)
    return record


def supervision_record(ctx):
    return {'record':ctx.write_json('supervision.json', supervision_value(ctx))}


def verified_supervision_record(ctx):
    from .visual_steps import confirmed, segmentation_subjects
    row = supervision_value(ctx)
    gateway = ctx.graph.gateway(ctx.parent)
    media, observation, masks = ctx.read('media'), ctx.read('observation'), ctx.read('masks')
    image, subjects = segmentation_subjects(gateway, media, observation)
    oq, observed = confirmed(gateway, observation, {'visual.inspect'})
    sq, segmented = confirmed(gateway, masks, {'visual.segment.image'})
    import json
    if (sq.artifacts != (image,) or json.loads(sq.messages[0]['content'])['subjects'] != subjects
            or {r['subject_id'] for r in segmented.output['instances']} != {s['subject_id'] for s in subjects}):
        raise ValueError('supervision segmentation is bound to another observation or image')
    usage = next(r for r in gateway.audit_usage()['attempts'] if r['request_id'] == sq.request_id)
    if (usage.get('sam_native_usage') or {}).get('forward_attempts') != len(subjects):
        raise ValueError('supervision requires actual SAM image-subject accounting')
    row['annotations'].update(observation=observed.output, segmentation=segmented.output,
                              sam_native_usage=usage['sam_native_usage'])
    row['model_receipts'].extend({'request': q.to_dict(), 'receipt': r.to_dict(),
        'receipt_hash': r.without_reuse_hash()} for q, r in ((oq, observed), (sq, segmented)))
    from .supervision import validate_record
    validate_record(row, ctx.store)
    return {'record': ctx.write_json('verified-supervision.json', row)}


def append_record(ctx):
    from .supervision import validate_record
    value, record = ctx.read('records'), ctx.read('record')
    rows = value['records'] + [record]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('supervision collection contains a duplicate record ID')
    for row in rows:
        validate_record(row, ctx.store)
    return {'records': ctx.write_json('records.json', {'records': rows})}


def export_t2i(ctx):
    from .supervision import export_dataset, load_dataset
    from .t2i_supervision import export_imagefolder
    from .family_ledger import ledger_for_context
    rows = ctx.read('records')['records']
    if any(r['task_type'] != 'text_to_image_generation' for r in rows):
        raise ValueError('T2I delivery cannot accept another supervision task')
    ledger = ledger_for_context(ctx)
    ids = [r['id'] for r in rows]
    ledger.register_release(ctx.parameters['release_id'], ids)
    family = ledger.require_release(ctx.parameters['release_id'], ids, unique_families=True)
    destination = ctx.graph.dataset_root / ctx.prefix / 'bundle'
    export_dataset(rows, ctx.store, destination, expected_counts=ctx.parameters['expected_counts'],
                   metadata={'family_release': ctx.parameters['release_id'], **family})
    load_dataset(destination)
    training = ctx.graph.dataset_root / ctx.prefix / 'imagefolder'
    exported = export_imagefolder(destination, training)
    files = []
    for directory in (destination, training):
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                relative = path.relative_to(ctx.graph.dataset_root).as_posix()
                digest = file_sha256(path)
                ctx.budget.reserve_file(relative + ':' + digest, path.stat().st_size)
                files.append(ArtifactRef(ctx.graph.dataset_store, relative, digest,
                                        'application/json' if path.suffix == '.json' else 'application/octet-stream').to_dict())
    return {'delivery': ctx.write_json('delivery.json', {'release_id': ctx.parameters['release_id'],
        'bundle_relative': destination.relative_to(ctx.graph.dataset_root).as_posix(),
        'training_relative': training.relative_to(ctx.graph.dataset_root).as_posix(),
        'manifest_sha256': exported['source_manifest_sha256'], 'training_manifest_sha256': exported['training_manifest_sha256'],
        'unique_samples': exported['unique_samples'], 'family': family, 'files': files,
        'evidence_level': 'real_execution_verified', 'training_benefit_verified': False})}


def register(registry):
    from dataevolver.harness.operator_sdk import declare_step
    registry.register_step(declare_step('images.catalog_from_media', catalog_from_media,
        inputs={'media':{'type':'object'}}, outputs={'catalog':{'type':'object'}},
        parameters={'type':'object','additionalProperties':False,'properties':{
            k:{'type':'string','minLength':1} for k in ['record_id','source_dataset','source_family']},
            'required':['record_id','source_dataset','source_family']}, kind='data', modalities=('json','image')))
    registry.register_step(declare_step('supervision.text_to_image', supervision_record,
        inputs={k:{'type':'object'} for k in ['media','review','catalog']},
        outputs={'record':{'type':'object'}}, parameters={'type':'object','properties':{},'additionalProperties':False},
        kind='data', modalities=('json','image')))
    registry.register_step(declare_step('supervision.text_to_image_verified', verified_supervision_record,
        inputs={k: {'type': 'object'} for k in ['media', 'review', 'catalog', 'observation', 'masks']},
        outputs={'record': {'type': 'object'}}, parameters={'type': 'object', 'properties': {}, 'additionalProperties': False},
        modalities=('json', 'image')))
    registry.register_step(declare_step('supervision.append', append_record,
        inputs={'records': {'type': 'object', 'required': ['records']}, 'record': {'type': 'object'}},
        outputs={'records': {'type': 'object', 'required': ['records']}},
        parameters={'type': 'object', 'properties': {}, 'additionalProperties': False}, modalities=('json', 'image')))
    registry.register_step(declare_step('supervision.export_t2i', export_t2i,
        inputs={'records': {'type': 'object', 'required': ['records']}}, outputs={'delivery': {'type': 'object'}},
        parameters={'type': 'object', 'additionalProperties': False, 'properties': {
            'ledger_path': {'type': 'string', 'minLength': 1}, 'namespace': {'type': 'string', 'minLength': 1},
            'hamming_distance': {'type': 'integer', 'minimum': 0, 'maximum': 64},
            'release_id': {'type': 'string', 'minLength': 1},
            'expected_counts': {'type': 'object', 'minProperties': 1, 'additionalProperties': {'type': 'integer', 'minimum': 0}}},
            'required': ['ledger_path', 'namespace', 'hamming_distance', 'release_id', 'expected_counts']}, modalities=('json', 'image')))
