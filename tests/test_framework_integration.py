"""Real artifact/Controller paths with CPU fixtures; no model quality claim."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, file_sha256, payload_sha256
from dataevolver.dataset.families import assign_families
from dataevolver.harness.data_contracts import DataInput, DataNodeSpec, DataNodeSpecV3, DataNodeGraphV3, DataRunBudget
from dataevolver.harness.models import EvidenceStatus, OperatorCall, RecoveryStatus
from dataevolver.harness.operator_sdk import builtin_step_registry, declare_step
from dataevolver.harness.plan_compiler import CompileContext, CompiledRun, CompilationGap, builtin_domains, compile_plan
from dataevolver.harness.task_contracts import DatasetIntent, QualityPolicy, PROTECTED_GATES


def catalog_contract(tmp_path):
    raw = tmp_path/'raw'; raw.mkdir()
    rows = [{'id': f'r{i}', 'source_id': 'a', 'source_dataset': 'fixture', 'source_family': '',
             'pixel_sha256': f'{i:064x}', 'dhash': f'{i:016x}', 'split': 'train',
             'protected_gates': {}, 'quality': {'clarity_proxy': i/10}}
            for i in range(6)]
    rows[-1]['pixel_sha256'] = rows[0]['pixel_sha256']
    path = raw/'source.json'; path.write_text(json.dumps({'rows': rows, 'sources': [{'rows_inspected': len(rows)}]}))
    ref = ArtifactRef('raw', path.name, file_sha256(path))
    nodes = [DataNodeSpecV3('load', 'catalog.load', '1', 'data', {'source': DataInput(artifact=ref)}),
        DataNodeSpecV3('families', 'images.families', '1', 'data', {'catalog': DataInput(node='load', output='catalog')}, {'hamming_distance': 0}),
        DataNodeSpecV3('select', 'images.select', '1', 'data', {'catalog': DataInput(node='families', output='catalog')}, {'per_source': 3, 'seed': 7}),
        DataNodeSpecV3('export', 'catalog.export', '1', 'data', {'catalog': DataInput(node='select', output='catalog')})]
    contract = {'intent': DatasetIntent('fixture', 'curate', 'catalog', 'catalog.v1', 'CPU catalog fixture', (ref,)).to_dict(),
        'quality_policy': QualityPolicy('fixture', tuple(sorted(PROTECTED_GATES)), tuple(n.operator for n in nodes), {}, 1).to_dict(),
        'graph': DataNodeGraphV3(tuple(nodes)).to_dict(), 'budget': DataRunBudget(4, 0, 0, 6, 1000000, 60).to_dict(),
        'stores': {'raw': 'FIXTURE_RAW', 'dataset': 'FIXTURE_DATASET'}, 'dataset_store': 'dataset', 'models': {}}
    return contract, {'raw': raw, 'dataset': tmp_path/'dataset'}, rows


def compile_fixture(tmp_path):
    contract, roots, rows = catalog_contract(tmp_path)
    registry = builtin_step_registry()
    context = CompileContext(registry, builtin_domains(), tuple(n['operator'] for n in contract['graph']['nodes']), roots=roots)
    return compile_plan({'domain': 'catalog', 'contract': contract}, context), context, rows


def test_compiled_catalog_uses_shared_family_implementation_and_recovers(tmp_path, monkeypatch):
    compiled, context, rows = compile_fixture(tmp_path)
    assert isinstance(compiled, CompiledRun)
    assert context.registry.step('images.families').implementation.__module__ == 'dataevolver.dataset.curation'
    compiled = CompiledRun.from_dict(compiled.to_dict())
    root = tmp_path/'campaign'
    op = compiled.bind(context.registry, allowed_write_root=root, roots=context.roots)
    call = OperatorCall('fixture', 'catalog-call', 'catalog-key', 'data.graph', '1', 0, str(root/'call'), parameters=compiled.contract)
    from dataevolver.harness.adapters import data_graph
    original = data_graph._write_receipt
    crashed = []
    def once(path, receipt):
        if receipt.operator_id == 'images.families' and not crashed:
            crashed.append(True)
            raise RuntimeError('after family effect, before receipt')
        return original(path, receipt)
    monkeypatch.setattr(data_graph, '_write_receipt', once)
    with pytest.raises(RuntimeError, match='before receipt'): op.execute(call)
    reopened = compiled.bind(context.registry, allowed_write_root=root, roots=context.roots)
    receipt = reopened.execute(call)
    assert reopened.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    journal = json.loads((Path(call.work_dir)/'data_budget.json').read_text())
    assert sum(journal['attempts'].values()) == 4
    assert reopened.execute(call).reused
    assert json.loads((Path(call.work_dir)/'data_budget.json').read_text()) == journal
    effect = json.loads((Path(call.work_dir)/'node_runs/families/effect.json').read_text())
    grouped = json.loads(reopened.store.resolve(ArtifactRef.from_dict(effect['outputs']['catalog'])).read_text())
    assert grouped['rows'] == assign_families(rows, 0)
    assert grouped['duplicate_method']['semantic_duplicate_detection'] == 'not_performed'


@pytest.mark.parametrize('change', ['version', 'port', 'parameter', 'source'])
def test_compile_rejects_bad_bindings_before_creating_effects(tmp_path, change):
    contract, roots, _ = catalog_contract(tmp_path)
    if change == 'version': contract['graph']['nodes'][1]['operator_version'] = '999'
    if change == 'port': contract['graph']['nodes'][1]['inputs']['catalog']['output'] = 'missing'
    if change == 'parameter': contract['graph']['nodes'][1]['parameters']['extra'] = True
    if change == 'source': contract['graph']['nodes'][0]['inputs']['source']['artifact']['sha256'] = 'f'*64
    context = CompileContext(builtin_step_registry(), builtin_domains(), tuple(n['operator'] for n in contract['graph']['nodes']), roots=roots)
    assert isinstance(compile_plan({'domain': 'catalog', 'contract': contract}, context), CompilationGap)
    assert not roots['dataset'].exists()


def test_text_contract_is_explicit_v3_and_factory_version_is_exact(tmp_path):
    args = dict(node_id='text', operator='text.describe', operator_version='2', kind='model', inputs={}, role='planner', max_model_calls=1)
    with pytest.raises(ValueError): DataNodeSpec(**args)
    assert DataNodeSpecV3(**args).max_model_images == 0
    registry = builtin_step_registry()
    with pytest.raises(ValueError): registry.step('images.families', '2')


def test_registry_planning_view_exposes_same_specs(tmp_path):
    from dataevolver.workflows.training_goal.catalog import operator_catalog
    registry = builtin_step_registry()
    view = operator_catalog(Path(__file__).parent, registry=registry)
    rows = {r['operator']: r for r in view['operators']}
    for spec in registry.step_specs():
        assert rows[spec.operator_id]['spec_sha256'] == spec.sha256()
        assert rows[spec.operator_id]['production_qualification'] == 'NOT_ASSESSED'


def text_step(ctx):
    from dataevolver.adapters.model_contracts import ModelRequest
    request = ModelRequest('text-node', ctx.node.role, ({'role': 'user', 'content': 'CPU protocol fixture'},),
        {'type': 'object', 'properties': {'answer': {'type': 'string'}}, 'required': ['answer']})
    return {'result': ctx.write_json('answer.json', ctx.invoke(request).output)}


def test_text_only_graph_runs_through_gateway_and_keeps_exact_version(tmp_path):
    from dataevolver.adapters.model_contracts import ModelProfile, ModelPolicy
    registry = builtin_step_registry()
    definition = declare_step('text.describe', text_step, inputs={}, outputs={'result': {'type': 'object'}},
        parameters={'type': 'object', 'additionalProperties': False, 'properties': {}}, version='2', kind='model')
    registry.register_step(definition)
    profile = ModelProfile('fixture', 'mock', 'fixture', 'r1', mock_output={'answer': 'fixture'})
    node = DataNodeSpecV3('text', 'text.describe', '2', 'model', {}, {}, 'planner', 1, 0)
    contract = {'intent': DatasetIntent('fixture', 'generate', 'catalog', 'json', 'Text node fixture').to_dict(),
        'quality_policy': QualityPolicy('fixture', tuple(sorted(PROTECTED_GATES)), ('text.describe',), {}, 1).to_dict(),
        'graph': DataNodeGraphV3((node,)).to_dict(), 'budget': DataRunBudget(1, 1, 0, 1, 100000, 60).to_dict(),
        'stores': {'dataset': 'FIXTURE_DATASET'}, 'dataset_store': 'dataset',
        'models': {'profiles': [profile.to_dict()], 'policy': ModelPolicy({'planner': 'fixture'}, 1).to_dict(), 'workers': {}}}
    context = CompileContext(registry, builtin_domains(), ('text.describe',), roots={'dataset': tmp_path/'dataset'})
    compiled = compile_plan({'domain': 'catalog', 'contract': contract}, context)
    assert isinstance(compiled, CompiledRun)
    op = compiled.bind(registry, allowed_write_root=tmp_path/'runtime', roots=context.roots)
    call = OperatorCall('fixture', 'text-call', 'text-key', 'data.graph', '1', 0, str(tmp_path/'runtime/call'), parameters=compiled.contract)
    receipt = op.execute(call)
    assert op.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    effect = json.loads((Path(call.work_dir)/'node_runs/text/effect.json').read_text())
    assert effect['model_requests'][0]['request']['artifacts'] == []
    assert json.loads((Path(call.work_dir)/'node_runs/text/node_run.json').read_text())['operator_revision'] == definition.spec.code_revision
    assert op.model_usage(call)['model_calls'] == 1


def test_family_definition_is_transitive_and_preserves_source_membership():
    rows = [{'id': str(i), 'pixel_sha256': str(i), 'dhash': f'{i*65535:016x}',
             'source_dataset': 'd', 'source_family': ''} for i in range(4)]
    rows[1]['pixel_sha256'] = rows[0]['pixel_sha256']
    rows[1]['source_family'] = rows[2]['source_family'] = 'source-family'
    before = copy.deepcopy(rows)
    result = assign_families(rows, 0)
    assert len({r['family_id'] for r in result[:3]}) == 1
    assert result[3]['family_id'] != result[0]['family_id']
    assert rows == before


def test_transition_proposal_rejects_stale_state_and_counter_changes():
    from types import SimpleNamespace
    from dataevolver.harness.domains import TransitionProposal
    state = SimpleNamespace(to_dict=lambda: {'seq': 1}, round_index=2)
    proposal = TransitionProposal.from_state(state, {'completed': ['a']}, completion_evidence={'receipt': 'bound'})
    proposal.validate(state)
    with pytest.raises(ValueError): replace(proposal, counters={'planner_calls': 0}).validate(state)
    other = SimpleNamespace(to_dict=lambda: {'seq': 2}, round_index=2)
    with pytest.raises(ValueError): proposal.validate(other)


def test_catalog_trial_runs_in_the_real_controller_and_replays(tmp_path):
    from dataevolver.harness.catalog_rsi import CatalogRSIDomain, CatalogTrialOperator
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.evaluator import EvaluatorContract
    from dataevolver.harness.models import GoalSpec, BudgetSpec
    from dataevolver.harness.operators import OperatorRegistry
    from dataevolver.harness.planner import DeterministicPlanner
    contract, roots, _ = catalog_contract(tmp_path)
    node = contract['graph']['nodes'][2]
    node['operator'] = 'images.select_balanced'
    node['parameters'] = {'quotas': {'a': {'fixture': 3}}, 'ranking': 'hash', 'seed': 1, 'min_clarity': 0.0}
    contract['quality_policy']['allowed_operations'] = [n['operator'] for n in contract['graph']['nodes']]
    source = roots['raw']/'source.json'
    value = json.loads(source.read_text())
    for row in value['rows']: row['label_id'] = 'fixture'
    source.write_text(json.dumps(value))
    ref = contract['graph']['nodes'][0]['inputs']['source']['artifact']; ref['sha256'] = file_sha256(source)
    contract['intent']['sources'][0] = dict(ref)
    registry = builtin_step_registry()
    compiled = compile_plan({'domain': 'catalog', 'contract': contract}, CompileContext(
        registry, builtin_domains(), tuple(n['operator'] for n in contract['graph']['nodes']), roots=roots))
    assert isinstance(compiled, CompiledRun)
    domain = CatalogRSIDomain(configuration={'compiled_run': compiled.to_dict(), 'source': ref,
        'baseline_ranking': 'clarity', 'allowed_rankings': ['hash', 'clarity'], 'quotas': {'a': {'fixture': 3}}, 'target_quality': 0.0}, registry=registry, roots=roots)
    op = CatalogTrialOperator(domain=domain, output_root=tmp_path/'campaigns')
    runtime = OperatorRegistry(); runtime.register(op)
    evaluator = tmp_path/'evaluator.json'
    evaluator.write_text(json.dumps({'schema_version': 'dataevolver.harness.evaluator.v1',
        'evaluator_id': 'fixture', 'revision': '1', 'hard_constraints': {'artifact_valid': {'required': True}},
        'target_metrics': {'completion_ratio': {'minimum': 1}}, 'protected_metrics': {}}))
    key = {'input_artifact_hash': ref['sha256'], 'parent_artifact_hash': ref['sha256'], 'seed': 1,
           'operator_id': op.spec.operator_id, 'operator_version': '1', 'model_config_hash': 'none',
           'preprocessing_hash': compiled.compute_hash(), 'evaluator_hash': file_sha256(evaluator), 'environment_contract_hash': 'fixture'}
    goal = GoalSpec('dataevolver.harness.goal.v1', 'catalog-fixture',
        {'required_hard_constraints': ['artifact_valid'], 'target_metrics': {'completion_ratio': 1}, 'initial_trial_key': key},
        op.spec.operator_id, ('fixture',), BudgetSpec(1, 1, 0, 0, 60), {'allowed_operators': [op.spec.operator_id]},
        {'path': str(evaluator), 'sha256': file_sha256(evaluator)},
        initial_operator_parameters={'ranking': 'clarity', 'experience_id': ''})
    controller = EvolutionController.create(repo_root=tmp_path, campaigns_root=tmp_path/'campaigns', goal=goal,
        registry=runtime, planner=DeterministicPlanner(()), evaluator_contract=EvaluatorContract.from_path(evaluator), idempotency_key='catalog-fixture')
    assert controller.run().state.value == 'COMPLETE'
    assert controller.replay().to_dict() == controller.status().to_dict()
    before = controller.layout.events_path.read_bytes()
    controller.run()
    assert controller.layout.events_path.read_bytes() == before


def test_restricted_catalog_scope_cannot_be_activated_by_an_event_label(tmp_path):
    from dataevolver.harness.operator_lifecycle import OperatorVersionStore
    store = OperatorVersionStore(tmp_path/'versions')
    snapshot = {'operator_id': 'images.families', 'version': '2', 'scope': 'catalog_family.v1'}
    with store.lease():
        event = store.append(action='ADOPTED', snapshot=snapshot, owner={'test': 'fixture'}, expected_head=store.head())
    with pytest.raises(ValueError, match='authority'):
        store.active(event, scope='catalog_family.v1', head=store.head())


def test_catalog_trial_preserves_failed_graph_without_reading_absent_export(tmp_path):
    from dataevolver.harness.catalog_rsi import CatalogRSIDomain, CatalogTrialOperator
    from dataevolver.harness.models import StageStatus
    contract, roots, _ = catalog_contract(tmp_path)
    node = contract['graph']['nodes'][2]
    node['operator'] = 'images.select_balanced'
    node['parameters'] = {'quotas': {'a': {'missing-label': 1}}, 'ranking': 'hash', 'seed': 0, 'min_clarity': 0.}
    contract['quality_policy']['allowed_operations'] = [n['operator'] for n in contract['graph']['nodes']]
    source = roots['raw']/'source.json'
    value = json.loads(source.read_text())
    for row in value['rows']: row['label_id'] = 'present-label'
    source.write_text(json.dumps(value))
    source_ref = contract['graph']['nodes'][0]['inputs']['source']['artifact']
    source_ref['sha256'] = file_sha256(source)
    contract['intent']['sources'][0] = dict(source_ref)
    registry = builtin_step_registry()
    compiled = compile_plan({'domain':'catalog','contract':contract}, CompileContext(
        registry,builtin_domains(),tuple(n['operator'] for n in contract['graph']['nodes']),roots=roots))
    assert isinstance(compiled, CompiledRun)
    domain = CatalogRSIDomain(configuration={'compiled_run':compiled.to_dict(),
        'source':contract['intent']['sources'][0], 'baseline_ranking':'hash', 'allowed_rankings':['hash','clarity'],
        'quotas':node['parameters']['quotas'], 'target_quality':0.},registry=registry,roots=roots)
    op = CatalogTrialOperator(domain=domain,output_root=tmp_path/'campaign')
    call = OperatorCall('fixture','catalog-failure','catalog-failure-key','data.catalog_trial','1',0,
        str(tmp_path/'campaign/call'),parameters={'ranking':'hash','experience_id':''})
    receipt = op.execute(call)
    assert receipt.status is StageStatus.FAILED and receipt.return_code != 0
    assert receipt.stage_result['failure_code'] == 'COMPILED_DATA_FAILED'
    assert op.verify_effect(call,receipt).status is EvidenceStatus.VERIFIED
    budget = Path(call.work_dir)/'graph/call/data_budget.json'
    before = budget.read_bytes()
    assert op.execute(call).reused
    assert budget.read_bytes() == before


def test_public_compile_and_evolve_entry_consumes_the_bound_registry(tmp_path, monkeypatch, capsys):
    from dataevolver.harness import cli
    contract, roots, _ = catalog_contract(tmp_path)
    profile = tmp_path/'profile.json'
    profile.write_text(json.dumps({'allowed_operators': [n['operator'] for n in contract['graph']['nodes']],
                                  'roots': {k: str(v) for k, v in roots.items()}}))
    plan = tmp_path/'plan.json'; plan.write_text(json.dumps({'domain': 'catalog', 'contract': contract}))
    evaluator = tmp_path/'evaluator.json'
    evaluator.write_text(json.dumps({'schema_version': 'dataevolver.harness.evaluator.v1', 'evaluator_id': 'fixture', 'revision': '1',
        'hard_constraints': {'artifact_valid': {'required': True}}, 'target_metrics': {'completion_ratio': {'minimum': 1}}, 'protected_metrics': {}}))
    assert cli.main(['compile-plan', '--plan', str(plan), '--profile', str(profile), '--output', str(tmp_path/'compiled'),
                     '--campaign', 'compile-cli', '--evaluator', str(evaluator)]) == 0
    monkeypatch.setenv('FIXTURE_RAW', str(roots['raw'])); monkeypatch.setenv('FIXTURE_DATASET', str(roots['dataset']))
    assert cli.main(['evolve', '--goal', str(tmp_path/'compiled/goal.json'), '--campaign-root', str(tmp_path/'campaigns'), '--idempotency-key', 'compile-cli']) == 0
    state = json.loads((tmp_path/'campaigns/compile-cli/state.json').read_text())
    assert state['state'] == 'COMPLETE'


def test_inference_bridge_roles_are_explicit_and_preserve_request_binding():
    from dataevolver.adapters.cli_planner_service import registered_roles, bound_request
    from dataevolver.adapters.model_contracts import ModelProfile, ModelRequest
    profile = ModelProfile('fixture', 'local_python', 'gpt-6.1-sol', 'fixture-revision')
    request = ModelRequest('review-fixture', 'main.review', ({'role': 'user', 'content': 'Review'},),
                           {'type': 'object'})
    def envelope(value):
        return {'request': value.to_dict(), 'profile_hash': profile.compute_hash(),
                'request_hash': payload_sha256({'request': value.to_dict(), 'profile_hash': payload_sha256({
                    'profile': profile.to_dict(), 'model': profile.model, 'endpoint': ''})})}
    with pytest.raises(ValueError, match='unbound'):
        bound_request(envelope(request), profile, registered_roles({}))
    roles = registered_roles({'allowed_roles': ['main.review', 'build.operator']})
    assert bound_request(envelope(request), profile, roles).to_dict() == request.to_dict()
    changed = envelope(request); changed['request']['messages'][0]['content'] = 'Changed'
    with pytest.raises(ValueError, match='unbound'):
        bound_request(changed, profile, roles)
    with pytest.raises(ValueError, match='unbound'):
        bound_request(envelope(replace(request, max_attempts=2)), profile, roles)
    with pytest.raises(ValueError): registered_roles({'allowed_roles': []})


def test_catalog_knowledge_rejects_new_manifest_with_reused_source_family(monkeypatch):
    from types import SimpleNamespace
    from dataevolver.harness import catalog_knowledge as knowledge
    baseline = {'source_sha256': 'new-manifest', 'source_families': ['family-a'],
                'counts': {'a': {'label': 1}}, 'rows': 1, 'unique_families': 1,
                'quality': 0.1, 'ranking': 'hash', 'reviews': []}
    improved = {**baseline, 'quality': 0.9, 'ranking': 'clarity'}
    op = object.__new__(knowledge.CatalogKnowledgeOperator)
    op.source = [{'report': {'source_sha256': 'old-manifest', 'source_families': ['family-a']}}]
    op.validation = [('baseline', None), ('improved', None)]
    op.candidate = SimpleNamespace(intervention=SimpleNamespace(parameters={'ranking': 'clarity'}))
    monkeypatch.setattr(knowledge, 'read_trials', lambda c, _: [{'report': baseline if c == 'baseline' else improved}])
    with pytest.raises(ValueError, match='independent'):
        op._validation()
    baseline['source_families'] = improved['source_families'] = ['family-b']
    op.source[0]['report']['source_media'] = [{'pixel_sha256':'old-pixels','image_sha256':'old-image','dhash':'0001'}]
    baseline['source_media'] = improved['source_media'] = [{'pixel_sha256':'new-pixels','image_sha256':'new-image','dhash':'0002'}]
    assert len(op._validation()) == 2
    improved['source_media'][0]['pixel_sha256'] = 'old-pixels'
    with pytest.raises(ValueError, match='independent'):
        op._validation()


def test_catalog_trial_remeasures_nested_image_bytes(tmp_path):
    from types import SimpleNamespace
    from PIL import Image
    from dataevolver.dataset.families import image_fingerprint
    from dataevolver.dataset.quality import clarity_features
    from dataevolver.harness.catalog_rsi import CatalogTrialOperator
    image = Image.new('RGB', (32, 32), 'red')
    path = tmp_path/'source.png'; image.save(path)
    row = {'id':'native-row','image':ArtifactRef('raw',path.name,file_sha256(path),'image/png').to_dict(),
           **image_fingerprint(image),'quality':clarity_features(image)}
    source = tmp_path/'source.json'; source.write_text(json.dumps({'rows':[row]}))
    op = object.__new__(CatalogTrialOperator)
    op.domain = SimpleNamespace(roots={'raw':tmp_path},catalog_ref=ArtifactRef('raw',source.name,file_sha256(source)))
    assert op.source_media()[0]['pixel_sha256'] == row['pixel_sha256']
    row['quality']['clarity_proxy'] += .1
    source.write_text(json.dumps({'rows':[row]}))
    op.domain.catalog_ref = ArtifactRef('raw',source.name,file_sha256(source))
    with pytest.raises(ValueError, match='measured clarity'):
        op.source_media()


def test_catalog_planner_context_exposes_the_same_strict_patch_contract(tmp_path):
    from types import SimpleNamespace
    from jsonschema import validate, ValidationError
    from dataevolver.harness.catalog_rsi import CatalogRSIDomain
    from dataevolver.harness.runtime_rsi import validate_decision
    compiled, context, _ = compile_fixture(tmp_path)
    source = compiled.contract['graph']['nodes'][0]['inputs']['source']['artifact']
    config = {'compiled_run': compiled.to_dict(), 'source': source, 'baseline_ranking': 'hash',
              'allowed_rankings': ['hash', 'clarity'], 'quotas': {}, 'target_quality': .5,
              'source_measurements_allowed':True,'source_measurement_scope':'development'}
    domain = CatalogRSIDomain(configuration=config, registry=context.registry, roots=context.roots)
    state = SimpleNamespace(data={'last_stage_result': {'status':'SUCCESS','evidence_refs': ['bound-trial']}})
    view = domain.context(state)
    assert view['source_observations']['current_ranking'] == 'hash'
    assert view['source_observations']['rows'][1]['clarity_proxy'] == .1
    assert view['source_observations']['source_sha256'] == source['sha256']
    parameters = json.loads(view['action_serialization']['RUN_EXPERIMENT']['parameters_json_example'])
    validate(parameters, view['patch_parameter_schemas']['sampling_recipe'])
    decision = {'action': 'RUN_EXPERIMENT', 'layer': 'R', 'hypothesis': 'Measure allowed ranking',
                'evidence_ids': ['bound-trial'], 'experience_id': '', 'reason': 'Bound source comparison',
                'observation': {'tool': '', 'parameters_json': '{}'},
                'patch': {'kind': 'sampling_recipe', 'parameters_json': json.dumps(parameters)}}
    validate_decision(decision, view)
    assert domain.compile_intervention(decision, state).parameters['ranking'] == 'clarity'
    with pytest.raises(ValidationError):
        validate({**parameters, 'experience_id': ''}, view['patch_parameter_schemas']['sampling_recipe'])
    config['capability_request_only'] = True
    request_domain = CatalogRSIDomain(configuration=config, registry=context.registry, roots=context.roots)
    request_view = request_domain.context(state)
    payload = {'problem': 'Bound implementation gap', 'proposed_capability': 'Reusable family grouping',
               'implementation_kind': 'registered_operator', 'required_inputs': ['rows'], 'required_outputs': ['rows'],
               'acceptance_checks': ['Exact relation'], 'validation_design': 'New source family'}
    validate(payload, request_view['patch_parameter_schemas']['capability_gap'])
    decision.update(action='REQUEST_CAPABILITY', layer='M',
                    patch={'kind': 'capability_gap', 'parameters_json': json.dumps(payload)})
    validate_decision(decision, request_view)
    assert request_domain.compile_capability_request(decision, state)['request'] == payload
    config['source_measurement_scope'] = 'sealed'
    with pytest.raises(ValueError, match='independent or sealed'):
        CatalogRSIDomain(configuration=config,registry=context.registry,roots=context.roots).context(state)


@pytest.mark.parametrize('invalid', ['missing_label','missing_clarity','nan_clarity'])
def test_balanced_selection_reports_bad_rows_as_data_errors(invalid):
    from dataevolver.dataset.quality import select_balanced
    row = {'id':'row','source_id':'source','label_id':'label','family_id':'family','quality':{'clarity_proxy':.5}}
    if invalid == 'missing_label':row.pop('label_id')
    elif invalid == 'missing_clarity':row['quality'] = {}
    else:row['quality']['clarity_proxy'] = float('nan')
    with pytest.raises(ValueError):
        select_balanced([row],{'source':{'label':1}},ranking='clarity',seed=0)


def test_builder_development_context_binds_public_contract_without_reading_qualification(tmp_path):
    from dataevolver.harness.operator_sdk import attach_development_context
    from dataevolver.harness.operator_development import OperatorInterface, QualificationProtocol
    step = builtin_step_registry().step('images.families', '1')
    declaration = step.contract.to_dict(); declaration.pop('spec_sha256'); declaration['version'] = '2'
    interface = OperatorInterface(declaration, True, 30)
    store = ArtifactStore({'protocol': tmp_path})
    cases = []
    for index, split in enumerate(['development', 'regression', 'independent', 'independent', 'heldout']):
        path = tmp_path/f'{index}.json'; path.write_text(json.dumps({'example': index}))
        ref = ArtifactRef('protocol', path.name, file_sha256(path))
        cases.append({'family_id': f'family-{index}', 'split': split, 'artifact': ref.to_dict()})
    stage = {'validator_operator_id': 'validate', 'validator_spec_sha256': 'a'*64,
             'evaluator_sha256': 'b'*64, 'check_ids': ['exact_relation']}
    protocol = QualificationProtocol('examples', tuple(cases),
        {'engineering': stage, 'independent': stage}, 'c'*64)
    accessed = []
    class DevelopmentOnlyStore:
        def resolve(self, ref):
            accessed.append(ref.path)
            assert ref == ArtifactRef.from_dict(cases[0]['artifact'])
            return store.resolve(ref)
    enriched = attach_development_context(interface, public_step=step, protocol=protocol,
        store=DevelopmentOnlyStore(), input_port='catalog')
    context = enriched.declaration['input_schemas']['catalog']['x-dataevolver-development-context']
    assert accessed == ['0.json']
    assert context['public_operator_contract'] == step.contract.to_dict()
    assert context['public_operator_spec_sha256'] == step.spec.sha256()
    assert context['development_examples'][0]['input'] == {'example': 0}
    assert len(context['development_examples']) == 1
    assert 'x-dataevolver-development-context' not in interface.declaration['input_schemas']['catalog']
    assert OperatorInterface.from_dict(enriched.to_dict()).to_dict() == enriched.to_dict()
    with pytest.raises(ValueError, match='byte limit'):
        attach_development_context(interface, public_step=step, protocol=protocol,
            store=store, input_port='catalog', max_example_bytes=1)
    with pytest.raises(ValueError, match='overwritten'):
        attach_development_context(enriched, public_step=step, protocol=protocol,
            store=store, input_port='catalog')
