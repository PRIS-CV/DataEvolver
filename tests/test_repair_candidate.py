"""CPU confinement and host binding only; build provenance is a unit fixture."""
import copy
import json
from pathlib import Path

import pytest

from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, canonical_json, file_sha256, payload_sha256
from dataevolver.harness import repair_candidate as module
from dataevolver.harness.candidate_build_authority import CommittedCandidateBuildAuthority
from dataevolver.harness.candidate_sandbox import LinuxPureDataRunner
from dataevolver.harness.models import OperatorSpec
from dataevolver.harness.operator_contracts import OperatorContract
from dataevolver.harness.operator_development import CandidatePackage

SOURCE = "def resolve(payload):\n    return {'repair_instruction': 'Separate the rejected objects; preserve the original counts and positions.'}\n"


def context():
    return {'image': {'sha256': 'a'*64, 'store': 'private_store', 'path': 'private/source.png'},
        'caption': 'Three red cubes and two blue spheres.', 'training_objective': 'Keep counts and spatial relations.',
        'dimensions': ['counts', 'spacing'], 'review': {'status': 'fail', 'criteria': {
            'counts': {'status': 'pass', 'reason': 'Correct.'}, 'spacing': {'status': 'fail', 'reason': 'Spheres touch.'}}},
        'source_media_sha256': 'b'*64, 'review_request_sha256': 'c'*64, 'review_receipt_hash': 'd'*64,
        'reasons': {'spacing': 'Spheres touch.'}}


@pytest.fixture
def execution():
    runner = LinuxPureDataRunner()
    payload = module.instruction_payload(context())
    result = runner.run(SOURCE, payload)
    assert result['isolation_verified'] and result['status'] == 'COMPLETED'
    return runner, payload, result


def test_real_confined_program_receives_feedback_without_host_paths_or_mutation(execution):
    runner, payload, result = execution
    original = context(); saved = copy.deepcopy(original)
    instruction = module.checked_instruction(result, source=SOURCE, payload=payload, runtime=runner.identity())
    assert instruction.startswith('Separate')
    assert 'private_store' not in json.dumps(payload) and 'private/source.png' not in json.dumps(payload)
    payload['review']['criteria']['spacing']['reason'] = 'changed inside caller'
    assert original == saved


@pytest.mark.parametrize('change', ['source', 'input', 'runtime', 'isolation', 'status'])
def test_output_cannot_be_rebound_to_another_candidate_or_feedback(execution, change):
    runner, payload, result = execution
    value = copy.deepcopy(result); source = SOURCE
    if change == 'source': source += '# another candidate\n'
    if change == 'input': payload['training_objective'] = 'Another goal.'
    if change == 'runtime': value['runtime']['uid'] = 0
    if change == 'isolation': value['isolation_verified'] = False
    if change == 'status': value['status'] = 'FAILED'
    with pytest.raises(ValueError, match='binding'):
        module.checked_instruction(value, source=source, payload=payload, runtime=runner.identity())


@pytest.mark.parametrize('output', [
    {'repair_instruction': 'Change it.', 'training_objective': 'Replace the goal.'},
    {'repair_instruction': 'Change it.', 'image': {'sha256': '0'*64}},
    {'repair_instruction': 'Change it.', 'model_authority': True},
    {'repair_instruction': ' '}, {'repair_instruction': 'x'*4097},
])
def test_candidate_cannot_return_goal_media_authority_or_unbounded_text(execution, output):
    runner, payload, result = execution
    value = copy.deepcopy(result); value['output'] = output
    with pytest.raises(ValueError, match='only one bounded'):
        module.checked_instruction(value, source=SOURCE, payload=payload, runtime=runner.identity())


def test_arbitrary_candidate_source_has_no_engineering_entry():
    with pytest.raises(TypeError, match='committed real build'):
        module.candidate_repair_step(plan=None, package=None, store=None,
            request_authority=None, build_authority=None, runner=None)


@pytest.mark.parametrize('change', ['operator', 'ports', 'model'])
def test_registry_repair_override_retains_its_pure_data_interface(change):
    from dataevolver.harness.operator_sdk import builtin_step_registry, declare_step
    inputs = {'media': {'type': 'object'}, 'review': {'type': 'object'}}
    if change == 'ports': del inputs['review']
    definition = declare_step('another.operator' if change == 'operator' else 'visual.prompt.repair',
        context, inputs=inputs, outputs={'prompt': {'type': 'object'}, 'repair': {'type': 'object'}},
        parameters={'type': 'object', 'properties': {}, 'additionalProperties': False},
        kind='model' if change == 'model' else 'data')
    with pytest.raises(ValueError, match='pure-data media/review'):
        builtin_step_registry(repair_step=definition)


def package_fixture(tmp_path, monkeypatch):
    """Mock only model-build provenance, never confinement or graph execution."""
    root = tmp_path/'candidate'; root.mkdir()
    store = ArtifactStore({'candidate': root})
    inventory = []
    def write(name, value, media='application/json'):
        path = root/name; path.write_text(value if isinstance(value, str) else canonical_json(value))
        ref = ArtifactRef('candidate', name, file_sha256(path), media).to_dict(); inventory.append(ref); return ref
    source = write('candidate.py', SOURCE, 'text/x-python')
    spec = OperatorSpec('visual.prompt.repair', '2', retry_safe=True, source_path='candidate.py', code_revision=source['sha256'])
    contract = OperatorContract(spec.operator_id, '2', spec.sha256(), 'basic', {}, {'repair_instruction': {'type': 'string'}},
        {'type': 'object', 'properties': {}, 'additionalProperties': False}, ('json',), 'pure_data', 'verify_then_reuse', ('Unit fixture only.',))
    refs = [write('operator-spec.json', spec.to_dict()), write('operator-contract.json', contract.to_dict()),
        write('dependencies.json', {'schema_version': 'dataevolver.candidate_dependencies.v1', 'dependencies': []}),
        write('changes.txt', 'Unit fixture; not a model-built or qualified candidate.', 'text/plain')]
    package = CandidatePackage('unit-only-repair', '0'*64, tuple(inventory), source, *refs)
    authority = object.__new__(CommittedCandidateBuildAuthority)
    authority.read = lambda: {'package': package.to_dict()}
    authority.identity = lambda: {'fixture': 'unit-build-provenance-only'}
    monkeypatch.setattr(module, 'inspect_candidate', lambda *args: None)
    return store, package, authority


def test_unqualified_step_uses_original_goal_and_one_fixture_review_then_replays(tmp_path, monkeypatch):
    from test_engineering_graph import repair_graph
    from dataevolver.harness.adapters.data_graph import DataGraphOperator
    from dataevolver.harness.data_contracts import DataNodeGraphV3
    from dataevolver.harness.operator_sdk import builtin_step_registry
    store, package, authority = package_fixture(tmp_path, monkeypatch)
    runner = LinuxPureDataRunner()
    definition = module.candidate_repair_step(plan=None, package=package, store=store,
        request_authority=None, build_authority=authority, runner=runner)
    (tmp_path/'graph').mkdir()
    original, old_call, media = repair_graph(tmp_path/'graph')
    contract = copy.deepcopy(original.contract); graph = DataNodeGraphV3.from_dict(contract['graph'])
    nodes = [n.to_dict() for n in graph.nodes]
    nodes[-1]['operator_version'] = '2'; del nodes[-1]['parameters']['repair_instruction']
    contract['graph'] = {'schema_version': graph.schema, 'nodes': nodes}
    registry = builtin_step_registry(repair_step=definition)
    operator = DataGraphOperator(allowed_write_root=tmp_path/'candidate-runtime', contract=contract,
        roots=original.store.roots, registry=registry)
    call = old_call.with_updates(work_dir=str(tmp_path/'candidate-runtime/run'), parameters=contract)
    receipt = operator.execute(call)
    assert receipt.status.value == 'SUCCESS'
    assert operator.verify_effect(call, receipt).status.value == 'VERIFIED'
    effect = json.loads((Path(call.work_dir)/'node_runs/repair/effect.json').read_text())
    prompt = json.loads(operator.store.resolve(ArtifactRef.from_dict(effect['outputs']['prompt'])).read_text())['prompt']
    trace = json.loads(operator.store.resolve(ArtifactRef.from_dict(effect['outputs']['repair'])).read_text())
    assert '\nOriginal unchanged goal: Faithful captions for text-to-image training.' in prompt
    assert trace['source_image'] == media['image'] and trace['production_qualification'] is False
    evidence = next(operator.dataset_root.rglob('candidate-repair-execution.json'))
    value = json.loads(evidence.read_text())
    assert value['execution']['isolation_verified'] and value['package_sha256'] == package.compute_hash()
    assert not value['model_authority'] and not value['image_authority'] and not value['production_qualification']
    assert operator.model_usage(call)['model_calls'] == 1  # Mock Review only, not a real model.
    before = {p: file_sha256(p) for p in operator.ledger_root(call).glob('*.json')}
    assert operator.execute(call).reused and before == {p: file_sha256(p) for p in operator.ledger_root(call).glob('*.json')}
