"""Confined instruction generation for an unqualified repair-method candidate.

This is an isolated engineering entry in the existing data graph. It grants no
model, image, qualification, adoption or production authority. The host retains
source validation, objective, review dimensions, rendering and model requests.
"""
import copy
import hashlib
import json
from pathlib import Path

from dataevolver.dataset import visual_steps
from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from dataevolver.dataset.visual_steps import repair_context, write_repair_prompt
from .candidate_build_authority import CommittedCandidateBuildAuthority
from .candidate_sandbox import LinuxPureDataRunner
from .models import OperatorSpec
from .operator_contracts import OperatorContract
from .operator_development import inspect_candidate
from .operator_sdk import StepDefinition

SCOPE = 'image_repair_instruction.v1'
OPERATOR_ID = 'visual.prompt.repair'


def instruction_payload(context):
    """Pass JSON task feedback only; no store, path, score, tool or endpoint."""
    return {'source_image_sha256': context['image']['sha256'], **copy.deepcopy({key: context[key] for key in (
        'caption', 'training_objective', 'dimensions', 'review')})}


def checked_instruction(execution, *, source, payload, runtime):
    """Only the isolated program's bound instruction can cross back to the host."""
    expected = {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'input_sha256': hashlib.sha256(json.dumps(payload, allow_nan=False, separators=(',', ':')).encode()).hexdigest(),
        'runtime_sha256': payload_sha256(runtime)}
    if (execution.get('status') != 'COMPLETED' or execution.get('isolation_verified') is not True
            or execution.get('runtime') != runtime or execution.get('binding') != expected):
        raise ValueError('repair candidate execution lost its source, input or isolation binding')
    output = execution.get('output')
    if (not isinstance(output, dict) or set(output) != {'repair_instruction'}
            or not isinstance(output['repair_instruction'], str)
            or not output['repair_instruction'].strip() or len(output['repair_instruction']) > 4096):
        raise ValueError('repair candidate may return only one bounded nonempty instruction')
    return output['repair_instruction']


def candidate_repair_step(*, plan, package, store, request_authority, build_authority, runner):
    """Host-bound prequalification step; callers cannot supply arbitrary source."""
    if (not isinstance(build_authority, CommittedCandidateBuildAuthority)
            or not isinstance(runner, LinuxPureDataRunner)):
        raise TypeError('repair candidate needs the committed real build and confined runner')
    inspect_candidate(plan, package, store, request_authority)
    if build_authority.read()['package'] != package.to_dict():
        raise ValueError('repair candidate differs from its committed model build')
    source_ref = ArtifactRef.from_dict(package.implementation)
    target = OperatorSpec.from_dict(json.loads(store.resolve(ArtifactRef.from_dict(package.operator_spec)).read_text()))
    target_contract = OperatorContract.from_dict(json.loads(
        store.resolve(ArtifactRef.from_dict(package.operator_contract)).read_text())).bound_to(target)
    if ((target.operator_id, target.version) != (OPERATOR_ID, '2')
            or target_contract.effect_semantics != 'pure_data'):
        raise ValueError('repair scope cannot execute another operator or a model-bearing candidate')
    runtime = runner.identity()
    origin = build_authority.identity()
    source = store.resolve(source_ref).read_text()

    def execute(ctx):
        inspect_candidate(plan, package, store, request_authority)
        if (build_authority.identity() != origin or build_authority.read()['package'] != package.to_dict()
                or runner.identity() != runtime or store.resolve(source_ref).read_text() != source):
            raise ValueError('repair candidate package, build or runtime changed')
        context = repair_context(ctx)
        payload = instruction_payload(context)
        execution = runner.run(source, payload)
        instruction = checked_instruction(execution, source=source, payload=payload, runtime=runtime)
        outputs = write_repair_prompt(ctx, context, instruction)
        # Evidence is a separate data artifact, not an adoption/qualification.
        ctx.write_json('candidate-repair-execution.json', {'scope': SCOPE,
            'package_sha256': package.compute_hash(), 'build_origin': origin,
            'execution': execution, 'production_qualification': False,
            'model_authority': False, 'image_authority': False})
        return outputs

    spec = OperatorSpec(OPERATOR_ID, '2', retry_safe=True, source_path=str(Path(__file__)),
        code_revision=payload_sha256({'source': source_ref.to_dict(), 'package': package.compute_hash(),
            'runner': runtime, 'build_origin': origin, 'adapter': file_sha256(Path(__file__)),
            'host_renderer': file_sha256(Path(visual_steps.__file__))}))
    parameters = {'training_objective': {'type': 'string', 'minLength': 1},
        'dimensions': {'type': 'array', 'items': {'type': 'string', 'minLength': 1},
            'minItems': 1, 'maxItems': 16, 'uniqueItems': True}}
    contract = OperatorContract(OPERATOR_ID, '2', spec.sha256(), 'basic',
        {'media': {'type': 'object'}, 'review': {'type': 'object'}},
        {'prompt': {'type': 'object', 'required': ['prompt'],
            'properties': {'prompt': {'type': 'string'}}, 'additionalProperties': False},
         'repair': {'type': 'object'}},
        {'type': 'object', 'properties': parameters, 'required': list(parameters), 'additionalProperties': False},
        ('json', 'image'), 'pure_data', 'verify_then_reuse',
        ('Unqualified candidate engineering only; no adoption or method benefit.',
         'Host retains original image, objective, dimensions, renderer and all model authority.'))
    return StepDefinition(spec, contract, 'data', execute)
