"""Composable media steps; every model effect remains owned by ModelGateway."""
from __future__ import annotations

import json

from dataevolver.adapters.model_contracts import ModelRequest
from dataevolver.adapters.training_prompt_enhancement import PE_I2I_SCHEMA, PE_T2I_SCHEMA
from .contracts import ArtifactRef, canonical_json, identifier, payload_sha256
from .training_pipeline import IMAGE_SCHEMA, MASK_SCHEMA, RESOLVE_SCHEMA, normalize_instance_observation
from .supervision import refs


ENVELOPE = {"type": "object", "required": ["request", "receipt_hash", "output"],
            "properties": {"request": {"type": "object"}, "receipt_hash": {"type": "string"},
                           "output": {"type": "object"}}, "additionalProperties": False}
VIDEO_SCHEMA = {"type": "object", "required": ["artifact", "actual_parameters", "frame_count", "fps"],
    "properties": {"artifact": {"type": "object"}, "actual_parameters": {"type": "object"},
        "frame_count": {"type": "integer"}, "fps": {"type": "number"}}, "additionalProperties": False}


def confirmed(gateway, envelope, roles):
    """Recover an already confirmed effect; this function never invokes a model."""
    request = ModelRequest.from_dict(envelope["request"])
    if request.role not in roles:
        raise ValueError("visual provenance has the wrong model role")
    receipt = gateway.recover(request)
    if (receipt is None or receipt.finish_reason != "stop" or receipt.without_reuse_hash() != envelope["receipt_hash"] or
            receipt.output != envelope["output"]):
        raise ValueError("visual provenance differs from the confirmed native effect")
    return request, receipt


def media_source(gateway, envelope):
    request, receipt = confirmed(gateway, envelope, {"visual.image.generate", "visual.image.edit", "visual.video.generate"})
    ref = ArtifactRef.from_dict(receipt.output["artifact"])
    parameters = json.loads(request.messages[0]["content"])
    if receipt.output["actual_parameters"] != parameters:
        raise ValueError("actual generation parameters differ from the frozen request")
    if not ref.media_type.startswith("video/" if request.role == "visual.video.generate" else "image/"):
        raise ValueError("generated media type differs from its confirmed role")
    return ref, parameters["prompt"]


def segmentation_subjects(gateway, media, observation):
    ref, prompt = media_source(gateway, media)
    _, receipt = confirmed(gateway, observation, {"visual.inspect", "visual.inspect.video"})
    request = ModelRequest.from_dict(observation["request"])
    if request.artifacts != (ref,):
        raise ValueError("segmentation observation is bound to another media artifact")
    text = json.loads(request.messages[0]['content'])['prompt']
    if '\nMEDIA_CONTEXT=' not in text:
        raise ValueError('observation is missing its generation prompt binding')
    context = json.loads(text.rsplit('\nMEDIA_CONTEXT=', 1)[1])
    if (context.get('generation_prompt') != prompt or
            context.get('generation_request_sha256') != ModelRequest.from_dict(media['request']).compute_hash()):
        raise ValueError('observation generation prompt binding changed')
    result = normalize_instance_observation(receipt.output)
    if result["status"] != "pass" or not result["subjects"]:
        raise ValueError("uncertain or failed observation cannot authorize segmentation")
    ids = [s["subject_id"] for s in result["subjects"]]
    instances = [s["target_instance_id"] for s in result["subjects"]]
    if len(set(ids)) != len(ids) or len(set(instances)) != len(instances):
        raise ValueError("observed subject and instance IDs must be unique")
    if sorted(ids) != sorted(context['subject_ids']):
        raise ValueError('observation omitted or added a requested subject')
    for subject in result["subjects"]:
        identifier(subject["subject_id"]); identifier(subject["target_instance_id"])
        if not subject["description"].strip(): raise ValueError("observed subject description is empty")
    return ref, result["subjects"]


def _gateway(ctx): return ctx.graph.gateway(ctx.parent)


def _invoke(ctx, role, parameters, artifacts, schema):
    if ctx.node.role != role or ctx.node.max_model_calls != 1:
        raise ValueError("visual step needs exactly one declared call for its exact model role")
    request = ModelRequest("visual-"+payload_sha256([ctx.prefix, role])[:32], role,
        ({"role": "user", "content": canonical_json(parameters)},), schema, tuple(artifacts),
        max_tokens=ctx.parameters["max_tokens"], timeout_seconds=ctx.parameters["timeout_seconds"],
        seed=ctx.parameters["seed"])
    receipt = ctx.invoke(request)
    if receipt.finish_reason != "stop": raise ValueError("truncated model output cannot advance the graph")
    for ref in refs(receipt.output):
        if ref.store != ctx.graph.dataset_store:
            raise ValueError('generated outputs must stay in the declared writable dataset store')
        path = ctx.store.resolve(ref)
        ctx.budget.reserve_file(ref.path+':'+ref.sha256, path.stat().st_size)
    return {"request": request.to_dict(), "receipt_hash": receipt.without_reuse_hash(), "output": receipt.output}


def _prompt(ctx):
    value = ctx.read("prompt")
    if "request" in value:
        request, receipt = confirmed(_gateway(ctx), value, {"visual.prompt.t2i", "visual.prompt.i2i"})
        if request.role == 'visual.prompt.i2i':
            if 'media' not in ctx.inputs:
                raise ValueError('image-edit refinement requires its original media')
            ref, _ = media_source(_gateway(ctx), ctx.read('media'))
            if request.artifacts != (ref,):
                raise ValueError('image-edit refinement is bound to another image')
        return receipt.output["rewritten_prompt"]
    if set(value) != {"prompt"} or not isinstance(value["prompt"], str) or not value["prompt"].strip():
        raise ValueError("prompt input must be explicit text or a confirmed PE result")
    return value["prompt"]


def generate(ctx):
    role = ctx.node.operator
    inputs = []
    if role in {"visual.image.edit", "visual.video.generate"}:
        ref, _ = media_source(_gateway(ctx), ctx.read("media"))
        if not ref.media_type.startswith("image/"): raise ValueError("editing and I2AV require an initial image")
        inputs.append(ref)
    parameters = {**ctx.parameters["sampling"], "prompt": _prompt(ctx)}
    result = _invoke(ctx, role, parameters, inputs, VIDEO_SCHEMA if role == "visual.video.generate" else IMAGE_SCHEMA)
    return {"media": ctx.write_json("media.json", result)}


def refine(ctx):
    role = ctx.node.operator
    inputs = []
    if role == "visual.prompt.i2i":
        ref, _ = media_source(_gateway(ctx), ctx.read("media")); inputs.append(ref)
    result = _invoke(ctx, role, {"prompt": _prompt(ctx)}, inputs,
                     PE_T2I_SCHEMA if role == "visual.prompt.t2i" else PE_I2I_SCHEMA)
    return {"prompt": ctx.write_json("prompt.json", result)}


def observe(ctx):
    media = ctx.read('media')
    ref, prompt = media_source(_gateway(ctx), media)
    video = ref.media_type.startswith("video/")
    if video != (ctx.node.operator == "visual.observe.video"):
        raise ValueError("observation operator does not match the actual media type")
    prompt = ("Observe the actual media. The generation prompt is context, not evidence of visible objects. "
              "Identify exactly these subject IDs, with unique target_instance_id, visible descriptions and boxes. "
              "Use uncertain for missing or ambiguous objects. " +
              ("Video observation covers first, middle and last frames only. " if video else "") +
              '\nMEDIA_CONTEXT=' + canonical_json({"generation_prompt": prompt,
                  'generation_request_sha256': ModelRequest.from_dict(media['request']).compute_hash(),
                  "subject_ids": ctx.parameters["subject_ids"]}))
    result = _invoke(ctx, "visual.inspect.video" if video else "visual.inspect", {"prompt": prompt}, [ref], RESOLVE_SCHEMA)
    if sorted(s["subject_id"] for s in result["output"]["subjects"]) != sorted(ctx.parameters["subject_ids"]):
        raise ValueError("observed subject IDs differ from the requested IDs")
    return {"observation": ctx.write_json("observation.json", result)}


def segment(ctx):
    ref, subjects = segmentation_subjects(_gateway(ctx), ctx.read("media"), ctx.read("observation"))
    video = ref.media_type.startswith("video/")
    if video != (ctx.node.operator == "visual.segment.video"):
        raise ValueError("segmentation operator does not match the observed media type")
    parameters = {"subjects": subjects}
    if not video: parameters.update(threshold=.5, mask_threshold=.5, min_box_iou=.3)
    schema = {"type": "object", "required": ["frames", "frame_count"]} if video else MASK_SCHEMA
    result = _invoke(ctx, ctx.node.operator, parameters, [ref], schema)
    return {"masks": ctx.write_json("masks.json", result)}


def _review_source(ctx):
    media = ctx.read('media')
    objective = ctx.parameters.get('training_objective')
    if objective is not None and (not isinstance(objective, str) or not objective.strip()):
        raise ValueError('training objective must be nonempty text')
    if any(key in media for key in ('request', 'receipt_hash', 'output')):
        ref, prompt = media_source(_gateway(ctx), media)
        return ref, prompt, None
    if (not {'id', 'image', 'prompt'} <= media.keys()
            or not isinstance(media['id'], str) or not media['id'].strip()
            or not isinstance(media['prompt'], str) or not media['prompt'].strip()):
        raise ValueError('imported review requires an identified image/caption record')
    if not objective:
        raise ValueError('imported review requires an explicit training objective')
    ref = ArtifactRef.from_dict(media['image'])
    if not ref.media_type.startswith('image/'):
        raise ValueError('imported caption review requires an image')
    ctx.store.resolve(ref)
    return ref, media['prompt'], media


def quality_review(ctx):
    from dataevolver.harness.feedback import FeedbackSignal
    ref, prompt, imported = _review_source(ctx)
    video = ref.media_type.startswith('video/')
    if video != (ctx.node.operator == 'visual.review.video'):
        raise ValueError('quality reviewer does not match the generated media type')
    dimensions = ctx.parameters['dimensions']
    status = {'type': 'string', 'enum': ['pass', 'fail', 'uncertain']}
    criterion = {'type': 'object', 'properties': {'status': status, 'reason': {'type': 'string'}},
                 'required': ['status', 'reason'], 'additionalProperties': False}
    schema = {'type': 'object', 'properties': {'status': status, 'summary': {'type': 'string'},
        'criteria': {'type': 'object', 'properties': {k: criterion for k in dimensions},
                     'required': dimensions, 'additionalProperties': False}},
        'required': ['status', 'summary', 'criteria'], 'additionalProperties': False}
    context = {'dimensions': dimensions, 'generation_prompt': prompt}
    scope = {}
    if imported is not None:
        context = {'dimensions': dimensions, 'source_caption': prompt,
                   'source_record_sha256': payload_sha256(imported)}
        scope['source_record_sha256'] = context['source_record_sha256']
    objective = ctx.parameters.get('training_objective')
    if objective is not None:
        context['training_objective'] = objective
        scope['training_objective_sha256'] = payload_sha256(objective)
    instruction = ('Inspect the actual media against these explicit dimensions. Keep failures and uncertainty. '
        'The declared caption and requirements are not evidence that an object exists. '
        + ('Only first/middle/last video frames are observed; unobserved temporal criteria must be uncertain. ' if video else ''))
    status_rule = ('The overall status must be pass only when every criterion passes. '
        'If any criterion fails, use fail. Otherwise, if any criterion is uncertain, use uncertain. ')
    if imported is None and objective is None:
        instruction = ('Inspect the actual media against these explicit dimensions. Keep failures and uncertainty. '
            'The generation prompt is intended content, not evidence that an object exists. '
            + ('Only first/middle/last video frames are observed; unobserved temporal criteria must be uncertain. ' if video else ''))
        bound_prompt = instruction + status_rule + canonical_json(context)
    else:
        bound_prompt = instruction + status_rule + '\nREVIEW_CONTEXT=' + canonical_json(context)
    result = _invoke(ctx, ctx.node.operator,
        {'prompt': bound_prompt}, [ref], schema)
    criteria = result['output']['criteria']
    if (result['output']['status'] == 'pass' and any(v['status'] != 'pass' for v in criteria.values())):
        raise ValueError('a passing quality summary contradicts its per-dimension judgments')
    evidence = ctx.write_json('quality-review.json', result)
    signal = FeedbackSignal('sample_quality', (evidence, ref),
        {'data_sha256': ref.sha256, 'review_request_sha256': ModelRequest.from_dict(result['request']).compute_hash(),
         'dimensions': dimensions, 'observation_scope': 'sampled_video_frames' if video else 'image',
         'calibration_verified': False, **scope},
        {'status': result['output']['status'], 'criteria': criteria}, 'unknown')
    return {'review': evidence, 'feedback': ctx.write_json('feedback.json', signal.to_dict())}


def refine_caption(ctx):
    ref, _, source = _review_source(ctx)
    if source is None:
        raise ValueError('caption correction requires an imported record, not generated supervision')
    request, receipt = confirmed(_gateway(ctx), ctx.read('review'), {'visual.review.image'})
    bound_prompt = json.loads(request.messages[0]['content'])['prompt']
    context = json.loads(bound_prompt.rsplit('\nREVIEW_CONTEXT=', 1)[1])
    if (request.artifacts != (ref,)
            or context.get('source_record_sha256') != payload_sha256(source)
            or context.get('source_caption') != source['prompt']
            or context.get('training_objective') != ctx.parameters['training_objective']
            or context.get('dimensions') != ctx.parameters['dimensions']):
        raise ValueError('caption correction review belongs to another record or objective')
    if receipt.output['status'] not in {'fail', 'uncertain'}:
        raise ValueError('a passing review does not authorize caption correction')
    schema = {'type': 'object', 'properties': {
        'status': {'type': 'string', 'enum': ['pass', 'uncertain']},
        'prompt': {'type': 'string'}, 'reason': {'type': 'string'}},
        'required': ['status', 'prompt', 'reason'], 'additionalProperties': False}
    prompt = ('Correct this imported training caption using the actual image and bound rejection reasons. '
        'Keep the original training objective and review dimensions. Do not change the image or relax '
        'requirements. Return uncertain if the image cannot support a faithful caption.\n'
        + canonical_json({'training_objective': ctx.parameters['training_objective'],
            'dimensions': ctx.parameters['dimensions'], 'source_caption': source['prompt'],
            'review': receipt.output}))
    result = _invoke(ctx, ctx.node.operator, {'prompt': prompt}, [ref], schema)
    if result['output']['status'] != 'pass' or not result['output']['prompt'].strip():
        raise ValueError('caption correction remained uncertain')
    corrected = {**source, 'prompt': result['output']['prompt'], 'caption_correction': {
        'source_record_sha256': payload_sha256(source),
        'review_request_sha256': request.compute_hash(),
        'correction_request_sha256': ModelRequest.from_dict(result['request']).compute_hash()}}
    return {'media': ctx.write_json('corrected-record.json', corrected),
            'correction': ctx.write_json('caption-correction.json', result)}


def repair_context(ctx):
    """Read the same confirmed rejection for fixed and confined repair methods."""
    ref, caption, imported = _review_source(ctx)
    if not ref.media_type.startswith('image/'):
        raise ValueError('repair prompt requires an image')
    ctx.store.resolve(ref)
    request, receipt = confirmed(_gateway(ctx), ctx.read('review'), {'visual.review.image'})
    context = json.loads(json.loads(request.messages[0]['content'])['prompt'].rsplit('\nREVIEW_CONTEXT=', 1)[1])
    expected = {'dimensions': ctx.parameters['dimensions'],
                'training_objective': ctx.parameters['training_objective']}
    expected.update({'source_caption': caption, 'source_record_sha256': payload_sha256(imported)}
                    if imported is not None else {'generation_prompt': caption})
    if request.artifacts != (ref,) or context != expected:
        raise ValueError('repair review belongs to another image, caption, objective or dimensions')
    if receipt.output['status'] not in {'fail', 'uncertain'}:
        raise ValueError('a passing review does not authorize image repair')
    criteria = receipt.output['criteria']
    if sorted(criteria) != sorted(ctx.parameters['dimensions']):
        raise ValueError('repair review criteria differ from the frozen dimensions')
    reasons = {k: v['reason'] for k, v in criteria.items() if v['status'] != 'pass'}
    if not reasons:
        raise ValueError('repair needs an explicit failed or uncertain criterion')
    return {'image': ref.to_dict(), 'caption': caption, 'review': receipt.output,
            'source_media_sha256': payload_sha256(ctx.read('media')),
            'review_request_sha256': request.compute_hash(),
            'review_receipt_hash': receipt.without_reuse_hash(),
            'training_objective': ctx.parameters['training_objective'],
            'dimensions': list(ctx.parameters['dimensions']), 'reasons': reasons}


def write_repair_prompt(ctx, context, instruction):
    """The host owns the unchanged objective and confirmed-source trace."""
    if context != repair_context(ctx):
        raise ValueError('repair context changed while preparing its instruction')
    if not instruction.strip():
        raise ValueError('repair instruction must be nonempty')
    prompt = (instruction + '\nOriginal unchanged goal: ' + context['training_objective']
              + '\nRecorded failed criteria: ' + json.dumps(context['reasons'], sort_keys=True))
    trace = {'schema_version': 'dataevolver.repair_prompt.v1', 'source_image': context['image'],
             'source_media_sha256': context['source_media_sha256'],
             'review_request_sha256': context['review_request_sha256'], 'review_receipt_hash': context['review_receipt_hash'],
             'training_objective_sha256': payload_sha256(context['training_objective']),
             'dimensions': context['dimensions'], 'instruction_sha256': payload_sha256(instruction),
             'prompt_sha256': payload_sha256(prompt), 'production_qualification': False}
    return {'prompt': ctx.write_json('repair-prompt.json', {'prompt': prompt}),
            'repair': ctx.write_json('repair-binding.json', trace)}


def repair_prompt(ctx):
    """Prepare a fixed repair recipe from confirmed rejection; no model call."""
    return write_repair_prompt(ctx, repair_context(ctx), ctx.parameters['repair_instruction'])


def register(registry, *, repair_step=None):
    from dataevolver.harness.operator_sdk import declare_step
    if repair_step is not None:
        from dataevolver.harness.operator_sdk import StepDefinition
        if (not isinstance(repair_step, StepDefinition) or repair_step.kind != 'data'
                or repair_step.spec.operator_id != 'visual.prompt.repair'
                or set(repair_step.contract.input_schemas) != {'media', 'review'}
                or set(repair_step.contract.output_schemas) != {'prompt', 'repair'}):
            raise ValueError('repair step override must retain the pure-data media/review and prompt/repair interface')
    common = {"seed": {"type": "integer", "minimum": 0}, "max_tokens": {"type": "integer", "minimum": 1},
              "timeout_seconds": {"type": "integer", "minimum": 1}}
    declarations = [
        ("visual.image.generate", generate, ["prompt"], "media", {"sampling": {"type": "object"}}),
        ("visual.image.edit", generate, ["prompt", "media"], "media", {"sampling": {"type": "object"}}),
        ("visual.video.generate", generate, ["prompt", "media"], "media", {"sampling": {"type": "object"}}),
        ("visual.prompt.t2i", refine, ["prompt"], "prompt", {}),
        ("visual.prompt.i2i", refine, ["prompt", "media"], "prompt", {}),
        ("visual.observe.image", observe, ["media"], "observation",
            {"subject_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 16, "uniqueItems": True}}),
        ("visual.observe.video", observe, ["media"], "observation",
            {"subject_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 16, "uniqueItems": True}}),
        ("visual.segment.image", segment, ["media", "observation"], "masks", {}),
        ("visual.segment.video", segment, ["media", "observation"], "masks", {}),
    ]
    for name, function, inputs, output, extra in declarations:
        props = {**common, **extra}
        registry.register_step(declare_step(name, function, kind="model",
            inputs={port: {"type": "object"} for port in inputs}, outputs={output: ENVELOPE},
            parameters={"type": "object", "properties": props, "required": list(props), "additionalProperties": False},
            modalities=("json", "image", "video")))
    for name in ('visual.review.image', 'visual.review.video'):
        props = {**common, 'dimensions': {'type': 'array', 'items': {'type': 'string', 'minLength': 1},
                                         'minItems': 1, 'maxItems': 16, 'uniqueItems': True}}
        required = list(props)
        props['training_objective'] = {'type': 'string', 'minLength': 1}
        registry.register_step(declare_step(name, quality_review, kind='model', inputs={'media': {'type': 'object'}},
            outputs={'review': ENVELOPE, 'feedback': {'type': 'object', 'required': ['schema_version', 'kind', 'evidence', 'scope']}},
            parameters={'type': 'object', 'properties': props, 'required': required, 'additionalProperties': False},
            category='evaluation', modalities=('json', 'image', 'video')))
    props = {**common, 'training_objective': {'type': 'string', 'minLength': 1},
             'dimensions': {'type': 'array', 'items': {'type': 'string', 'minLength': 1},
                            'minItems': 1, 'maxItems': 16, 'uniqueItems': True}}
    registry.register_step(declare_step('visual.caption.refine', refine_caption, kind='model',
        inputs={'media': {'type': 'object'}, 'review': ENVELOPE},
        outputs={'media': {'type': 'object'}, 'correction': ENVELOPE},
        parameters={'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False},
        modalities=('json', 'image')))
    props = {k: props[k] for k in ('training_objective', 'dimensions')}
    props['repair_instruction'] = {'type': 'string', 'minLength': 1, 'maxLength': 16384}
    registry.register_step(repair_step if repair_step is not None else declare_step('visual.prompt.repair', repair_prompt,
        inputs={'media': {'type': 'object'}, 'review': ENVELOPE},
        outputs={'prompt': {'type': 'object', 'required': ['prompt'],
                            'properties': {'prompt': {'type': 'string'}}, 'additionalProperties': False},
                 'repair': {'type': 'object'}},
        parameters={'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False},
        modalities=('json', 'image')))
