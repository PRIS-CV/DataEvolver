"""Evidence-boundary fixtures; these never establish native M qualification."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, file_sha256, payload_sha256
from dataevolver.harness import repair_qualification as module


def test_repair_scope_refuses_caller_reported_build_and_scores(tmp_path):
    with pytest.raises(ValueError, match='real build'):
        module.RepairCandidateChecks(plan=None, package=None, store=None, request_authority=None,
            build_authority=None, runner=None, cases=[], phase='independent', run_pairs={},
            outcome_authorities=[], output_root=tmp_path)
    with pytest.raises(TypeError, match='committed data graphs'):
        module.CommittedRepairRunAuthority(repair={}, edited={}, repair_node='repair', edit_node='edit',
            review_node='review', plan=None, package=None, store=None, request_authority=None,
            build_authority=None, runner=None, candidate=True)
    with pytest.raises(TypeError, match='actual committed'):
        module.RepairQualificationAuthority(engineering={}, independent={}, package=None,
            runner=None, lifecycle_total={}, resource_deadline_epoch=1)


@pytest.fixture
def comparison(tmp_path):
    """Only unit authority answers; no native models, Controller or model build."""
    operator = object.__new__(module.RepairCandidateChecks)
    operator.plan = SimpleNamespace(plan_id='unit-repair')
    operator.package = SimpleNamespace(compute_hash=lambda: 'candidate')
    operator.runner = SimpleNamespace(identity=lambda: {'fixture': 'runtime'})
    operator.build = SimpleNamespace(read=lambda: {'origin': {'fixture': 'unit-only'}})
    operator.phase = 'independent'
    operator.store = ArtifactStore({'protected': tmp_path})
    profile_hash = payload_sha256({'fixture': '8B-config'})
    operator.cases, operator.pairs, operator.outcomes = [], {}, []
    for i in range(2):
        descriptor = {'training_objective': 'Counts and spatial relations.', 'dimensions': ['count', 'spacing'],
            'generation_prompt': f'Frozen source family {i}.', 'baseline_instruction': 'Fixed repair.',
            'questions': [f'Question {k}' for k in range(10)], 'answers': ['yes']*10,
            'atoms': [{'question': 'Correct count?', 'answer': 'yes'}],
            'benchmark_profiles_sha256': dict.fromkeys(('spatial', 'geneval2'), profile_hash)}
        path = tmp_path/f'case{i}.json'; path.write_text(json.dumps(descriptor))
        case = {'family_id': f'family{i}', 'split': 'independent',
            'artifact': ArtifactRef('protected', path.name, file_sha256(path)).to_dict()}
        context = {'image': {'sha256': f'original{i}'}, 'training_objective': descriptor['training_objective'],
            'dimensions': descriptor['dimensions'], 'caption': descriptor['generation_prompt'],
            'review': {'criteria': {'count': {'status': 'pass'}, 'spacing': {'status': 'fail'}}}}
        descriptor.update(source_context_sha256=payload_sha256(context), matching={'seed': 0, 'model': 'fixture-only'})
        path.write_text(json.dumps(descriptor))
        case['artifact']['sha256'] = file_sha256(path)
        def run(label):
            return {'origin': {'fixture': 'unit-only'}, 'context': copy.deepcopy(context),
                'image': {'sha256': f'{label}{i}'}, 'prompt_sha256': label,
                'instruction_sha256': payload_sha256(descriptor['baseline_instruction']) if label == 'baseline' else 'candidate',
                'review': {'status': 'pass', 'criteria': {'count': {'status': 'pass'}, 'spacing': {'status': 'pass'}}},
                'matching': {'seed': 0, 'model': 'fixture-only'},
                'package_sha256': 'candidate' if label == 'candidate' else None}
        values = [run('baseline'), run('candidate')]
        operator.cases.append(case)
        operator.pairs[case['family_id']] = [SimpleNamespace(read=lambda v=v: copy.deepcopy(v)) for v in values]
        for kind in ('spatial', 'geneval2'):
            items, results = [], {}
            for value, score in zip(values, (.5, .6)):
                identity = value['image']['sha256']
                items.append({'id': identity, 'image_sha256': identity,
                    **{k: descriptor[k] for k in (('questions', 'answers') if kind == 'spatial' else ('atoms',))}})
                results[identity] = {'result': {'score': {'avg_acc': score}} if kind == 'spatial'
                    else {'soft_tifa_gm': score*100}}
            value = {'report': {'manifest': {'qualification_plan_id': 'unit-repair', 'kind': kind, 'items': items},
                'results': results}, 'origin': {'fixture': 'unit-only'}, 'receipt_sha256': 'fixture'}
            authority = SimpleNamespace(read=lambda v=value: copy.deepcopy(v),
                operator=SimpleNamespace(gateway=SimpleNamespace(profiles={
                    'fixture': SimpleNamespace(to_dict=lambda: {'fixture': '8B-config'})})))
            authority.operator.gateway.identity = lambda role, a=authority: a.operator.gateway.profiles['fixture'].to_dict()
            operator.outcomes.append(authority)
    return operator


def test_complete_unit_pair_computes_joint_quality_without_running_candidates(comparison):
    result = comparison._report()
    assert result['mean_delta'] == pytest.approx(.1) and result['passed']
    assert not result['training_benefit_verified'] and not result['production_adoption_allowed']
    assert all(r['protected_checks_passed'] for r in result['runs'])


@pytest.mark.parametrize('change', ['source', 'goal', 'dimensions', 'sampling', 'package', 'baseline_recipe',
    'question', 'gold', 'atom', 'evaluator', 'missing_atomic', 'duplicate', 'nonfinite', 'another_plan'])
def test_comparison_refuses_unmatched_or_rebound_evidence(comparison, change):
    pair = comparison.pairs['family0']
    value = pair[1].read()
    if change == 'source': value['context']['image']['sha256'] = 'another'
    if change == 'goal': value['context']['training_objective'] = 'another'
    if change == 'dimensions': value['context']['dimensions'] = ['relaxed']
    if change == 'sampling': value['matching']['seed'] = 1
    if change == 'package': value['package_sha256'] = 'another'
    pair[1].read = lambda: copy.deepcopy(value)
    if change == 'baseline_recipe':
        baseline = pair[0].read(); baseline['instruction_sha256'] = 'another'
        pair[0].read = lambda: baseline
    if change in {'question', 'gold', 'atom', 'nonfinite', 'another_plan'}:
        authority = comparison.outcomes[1 if change == 'atom' else 0]
        data = authority.read(); manifest = data['report']['manifest']
        if change == 'question': manifest['items'][0]['questions'][0] = 'easier'
        if change == 'gold': manifest['items'][0]['answers'][0] = 'no'
        if change == 'atom': manifest['items'][0]['atoms'][0]['answer'] = 'no'
        if change == 'nonfinite': data['report']['results']['baseline0']['result']['score']['avg_acc'] = float('nan')
        if change == 'another_plan': manifest['qualification_plan_id'] = 'another'
        authority.read = lambda: data
    if change == 'evaluator':
        comparison.outcomes[0].operator.gateway.profiles['fixture'].to_dict = lambda: {'fixture': '72B-changed'}
    if change == 'missing_atomic': comparison.outcomes.pop(1)
    if change == 'duplicate': comparison.outcomes.append(comparison.outcomes[0])
    with pytest.raises((ValueError, KeyError)):
        comparison._report()


def test_original_five_point_gate_is_preserved(comparison):
    for a in comparison.outcomes:
        value = a.read()
        for identity, row in value['report']['results'].items():
            score = .549 if identity.startswith('candidate') else .5
            row['result'] = {'score': {'avg_acc': score}} if value['report']['manifest']['kind'] == 'spatial' else {'soft_tifa_gm': 100*score}
        a.read = lambda v=value: v
    result = comparison._report()
    assert result['mean_delta'] == pytest.approx(.049) and not result['passed']


@pytest.mark.parametrize('change', ['uncertain', 'lost_protected_count', 'same_instruction', 'regression_loss'])
def test_quality_gain_cannot_override_protection_or_regression(comparison, change):
    pairs = [comparison.pairs[c['family_id']][1] for c in comparison.cases]
    for pair in pairs:
        value = pair.read()
        if change == 'uncertain': value['review']['status'] = 'uncertain'
        if change == 'lost_protected_count': value['review']['criteria']['count']['status'] = 'fail'
        if change == 'same_instruction': value['prompt_sha256'] = 'baseline'
        pair.read = lambda v=value: v
    if change == 'regression_loss':
        comparison.phase = 'engineering'; comparison.cases[0]['split'] = 'regression'
        for a in comparison.outcomes[:2]:
            value = a.read()
            value['report']['results']['candidate0']['result'] = ({'score': {'avg_acc': .4}}
                if value['report']['manifest']['kind'] == 'spatial' else {'soft_tifa_gm': 40})
            a.read = lambda v=value: v
    assert not comparison._report()['passed']


def test_benchmark_authorities_do_not_cross_selector_and_repair_scopes(monkeypatch):
    from dataevolver.harness.artifact_operator import CommittedWorkloadAuthority
    from dataevolver.harness.adapters.generation_benchmark import CommittedBenchmarkOutcomeAuthority
    reader = object.__new__(module.CommittedRepairBenchmarkAuthority)
    selector = object.__new__(CommittedBenchmarkOutcomeAuthority)
    value = {'report': {'manifest': {'purpose': 'selection_qualification', 'qualification_plan_id': 'plan',
        'split': 'development'}}}
    monkeypatch.setattr(CommittedWorkloadAuthority, 'read', lambda self: value)
    with pytest.raises(ValueError, match='own frozen purpose'): reader.read()
    value['report']['manifest']['purpose'] = 'repair_qualification'
    assert reader.read() == value
    with pytest.raises(ValueError, match='cannot qualify a selector'): selector.read()


def test_repair_workload_rechecks_evidence_at_receipt_verification(comparison, monkeypatch):
    monkeypatch.setattr(module, 'inspect_candidate', lambda *args: None)
    comparison.request = None
    report = comparison._report()
    comparison.validate_report(None, report)
    report['passed'] = False
    with pytest.raises(ValueError, match='verdict changed'):
        comparison.validate_report(None, report)


@pytest.mark.parametrize('change', ['both_source', 'both_sampling', 'extra_image'])
def test_frozen_descriptor_blocks_changing_both_arms_together(comparison, change):
    if change == 'extra_image':
        authority = comparison.outcomes[0]
        value = authority.read()
        item = copy.deepcopy(value['report']['manifest']['items'][0])
        item.update(id='unplanned', image_sha256='unplanned')
        value['report']['manifest']['items'].append(item)
        value['report']['results']['unplanned'] = {'result': {'score': {'avg_acc': 1.}}}
        authority.read = lambda: value
    else:
        for arm in comparison.pairs['family0']:
            value = arm.read()
            if change == 'both_source': value['context']['image']['sha256'] = 'same-prompt-another-seed'
            else: value['matching']['seed'] = 99
            arm.read = lambda v=value: v
    with pytest.raises(ValueError): comparison._report()


def committed_workload(tmp_path, operator, parameters, metric, campaign):
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.evaluator import EvaluatorContract
    from dataevolver.harness.models import GoalSpec, BudgetSpec
    from dataevolver.harness.operators import OperatorRegistry
    from dataevolver.harness.planner import DeterministicPlanner
    path = tmp_path/(campaign+'-evaluator.json')
    path.write_text(json.dumps({'schema_version': 'dataevolver.harness.evaluator.v1',
        'evaluator_id': campaign, 'revision': '1', 'hard_constraints': {'artifact_valid': {'required': True}},
        'target_metrics': {metric: {'minimum': 1}}, 'protected_metrics': {}}))
    evaluator = EvaluatorContract.from_path(path)
    key = {'input_artifact_hash': payload_sha256(parameters), 'parent_artifact_hash': payload_sha256(parameters),
        'seed': 0, 'operator_id': operator.spec.operator_id, 'operator_version': operator.spec.version,
        'model_config_hash': operator.spec.model_config_hash or 'fixture',
        'preprocessing_hash': operator.spec.code_revision, 'evaluator_hash': evaluator.sha256,
        'environment_contract_hash': 'fixture'}
    goal = GoalSpec('dataevolver.harness.goal.v1', campaign, {'kind': 'fixture', 'initial_trial_key': key}, operator.spec.operator_id,
        (), BudgetSpec(1, 1, 0, 0, 120), {'allowed_operators': [operator.spec.operator_id]},
        {'path': str(path), 'sha256': evaluator.sha256}, initial_operator_parameters=parameters)
    registry = OperatorRegistry(); registry.register(operator)
    controller = EvolutionController.create(repo_root=tmp_path, campaigns_root=operator.output_root/'campaigns'
        if hasattr(operator, 'output_root') else operator.root/'campaigns', goal=goal, registry=registry,
        planner=DeterministicPlanner(()), evaluator_contract=evaluator, idempotency_key=campaign)
    assert controller.run().state.value == 'COMPLETE'
    assert controller.replay().to_dict() == controller.status().to_dict()
    return controller


def test_real_committed_graph_is_read_only_and_mock_review_cannot_qualify(tmp_path):
    from test_engineering_graph import repair_graph
    from dataevolver.harness.artifact_operator import CommittedEffectAuthority
    operator, _, _ = repair_graph(tmp_path)
    controller = committed_workload(tmp_path, operator, operator.contract, 'completion_ratio', 'repair-graph-fixture')
    authority = CommittedEffectAuthority(controller, operator)
    step, effect = module.CommittedRepairRunAuthority._node(authority, 'repair')
    assert step.spec.operator_id == 'visual.prompt.repair' and not effect['model_requests']
    before = {p: file_sha256(p) for p in operator.root.rglob('*.json')}
    reader = module.CommittedRepairRunAuthority(repair=authority, edited=authority,
        repair_node='repair', edit_node='unused-edit', review_node='unused-review', plan=None,
        package=SimpleNamespace(compute_hash=lambda: 'unit-package'), store=operator.store,
        request_authority=None, build_authority=None, runner=None, candidate=False)
    with pytest.raises(ValueError, match='native original image and source Review'): reader.read()
    assert before == {p: file_sha256(p) for p in operator.root.rglob('*.json')}
    controller.layout.state_path.write_text(controller.layout.state_path.read_text().replace('COMPLETE', 'ABSTAIN'))
    with pytest.raises(ValueError, match='complete'): reader.read()


def test_typed_repair_adoption_replay_and_revocation_guard_actual_graph(tmp_path, monkeypatch):
    """Synthetic qualification/build only; actual catalog, Controllers, graph and sandbox."""
    from test_engineering_graph import repair_graph
    from test_repair_candidate import package_fixture
    from dataevolver.harness.adapters.data_graph import DataGraphOperator
    from dataevolver.harness.adapters.operator_version_lifecycle import OperatorVersionMutation
    from dataevolver.harness.artifact_operator import CommittedEffectAuthority
    from dataevolver.harness.candidate_sandbox import LinuxPureDataRunner
    from dataevolver.harness.data_contracts import DataNodeGraphV3
    from dataevolver.harness.operator_development import DevelopmentBudget
    from dataevolver.harness.operator_lifecycle import OperatorVersionStore, CommittedOperatorAdoptionAuthority
    from dataevolver.harness.operator_sdk import builtin_step_registry
    from dataevolver.harness.models import OperatorCall
    store, package, build = package_fixture(tmp_path, monkeypatch)
    runner = LinuxPureDataRunner()
    prior = dict(operator_calls=9, planner_calls=2, model_calls=4, image_experiments=0, output_bytes=100, wall_seconds=1)
    snapshot = {'operator_id': module.OPERATOR_ID, 'version': '2', 'scope': module.SCOPE,
        'package_sha256': package.compute_hash(), 'runtime': runner.identity(),
        'lifecycle_total': prior, 'resource_deadline_epoch': 1}
    qualification = object.__new__(module.RepairQualificationAuthority)
    qualification.read = lambda: copy.deepcopy(snapshot)
    catalog = OperatorVersionStore(tmp_path/'versions')
    budget = DevelopmentBudget({**prior, 'operator_calls': 12, 'output_bytes': 200000, 'wall_seconds': 100}, prior, 1)
    mutation = OperatorVersionMutation(catalog=catalog, expected_head=catalog.head(), budget=budget,
        output_root=tmp_path/'adoption', store=store, qualification_authority=qualification)
    controller = committed_workload(tmp_path, mutation, {}, 'operator_adopted', 'repair-adoption-fixture')
    call_id = next(e.payload['call_id'] for e in controller.store.events.read() if e.event_type == 'ACTION_COMMITTED')
    adoption = CommittedOperatorAdoptionAuthority(controller, mutation, call_id)
    kwargs = dict(catalog=catalog, adoption=adoption, qualification=qualification, expected_head=catalog.head(),
        plan=None, package=package, store=store, request_authority=None, build_authority=build, runner=runner)
    definition = module.qualified_repair_step(**kwargs)
    (tmp_path/'graph').mkdir()
    original, old_call, _ = repair_graph(tmp_path/'graph')
    contract = copy.deepcopy(original.contract)
    nodes = [n.to_dict() for n in DataNodeGraphV3.from_dict(contract['graph']).nodes]
    nodes[-1]['operator_version'] = '2'; del nodes[-1]['parameters']['repair_instruction']
    contract['graph']['nodes'] = nodes
    operator = DataGraphOperator(allowed_write_root=tmp_path/'adopted-runtime', contract=contract,
        roots=original.store.roots, registry=builtin_step_registry(repair_step=definition))
    graph_controller = committed_workload(tmp_path, operator, contract, 'completion_ratio', 'adopted-repair-graph')
    call = OperatorCall.from_dict(next(e.payload['operator_call'] for e in graph_controller.store.events.read()
        if e.event_type == 'ACTION_PLANNED'))
    assert operator.execute(call).reused and len(catalog.events()) == 1
    graph_authority = CommittedEffectAuthority(graph_controller, operator)
    reader = module.CommittedRepairRunAuthority(repair=graph_authority, edited=graph_authority,
        repair_node='repair', edit_node='unused-edit', review_node='unused-review', plan=None,
        package=package, store=store, request_authority=None, build_authority=build, runner=runner,
        candidate=True, adoption=adoption, qualification=qualification)
    # The adopted implementation hash is recognized, but synthetic Review still grants no native qualification.
    with pytest.raises(ValueError, match='native original image and source Review'): reader.read()
    revocation = OperatorVersionMutation(catalog=catalog, expected_head=catalog.head(), budget=budget,
        output_root=tmp_path/'revocation', adoption=adoption, reason='Unit revocation exercise.')
    committed_workload(tmp_path, revocation, {}, 'operator_revoked', 'repair-revocation-fixture')
    with pytest.raises(ValueError): module.qualified_repair_step(**kwargs)
    with pytest.raises(ValueError, match='native original image and source Review'): reader.read()
    # The previously resolved closure also checks the current catalog inside the execution lease.
    outputs = {}
    for node in operator.graph.ordered():
        child, step = operator._step(call, node, outputs, operator.journal(call))
        effect = json.loads((Path(child.work_dir)/'effect.json').read_text())
        if node.node_id == 'repair':
            with pytest.raises(ValueError): definition.implementation(step.ctx)
        outputs[node.node_id] = {k: ArtifactRef.from_dict(v) for k, v in effect['outputs'].items()}
    assert len(catalog.events()) == 2


def test_complete_candidate_graph_binds_edit_and_review_native_lookup(tmp_path, monkeypatch):
    """All model outputs/build origin are simulated; graph, receipts and sandbox are real CPU effects."""
    from PIL import Image
    from test_repair_candidate import package_fixture
    from dataevolver.adapters.model_contracts import ModelProfile, ModelPolicy
    from dataevolver.harness.adapters.data_graph import DataGraphOperator
    from dataevolver.harness.artifact_operator import CommittedEffectAuthority
    from dataevolver.harness.candidate_sandbox import LinuxPureDataRunner
    from dataevolver.harness.data_contracts import DataInput, DataNodeSpecV3, DataNodeGraphV3, DataRunBudget
    from dataevolver.harness.operator_sdk import builtin_step_registry
    from dataevolver.harness.task_contracts import DatasetIntent, QualityPolicy, PROTECTED_GATES
    store, package, build = package_fixture(tmp_path, monkeypatch)
    runner = LinuxPureDataRunner()
    definition = module.candidate_repair_step(plan=None, package=package, store=store,
        request_authority=None, build_authority=build, runner=runner)
    inputs, data = tmp_path/'inputs', tmp_path/'data'
    inputs.mkdir(); data.mkdir()
    source_path = inputs/'prompt.json'; source_path.write_text(json.dumps({'prompt': 'Frozen fixture objects.'}))
    source = ArtifactRef('inputs', 'prompt.json', file_sha256(source_path))
    images = {}
    for name, color in [('original', 'red'), ('edited', 'blue')]:
        path = data/(name+'.png'); Image.new('RGB', (16, 16), color).save(path)
        images[name] = ArtifactRef('data', path.name, file_sha256(path), 'image/png')
    class Provider:
        def __init__(self, model): self.model, self.records, self.invocations = model, {}, 0
        def invoke(self, request, digest):
            self.invocations += 1
            if request.role == 'visual.review.image':
                status = 'fail' if request.artifacts[0] == images['original'] else 'pass'
                output = {'status': status, 'summary': status,
                    'criteria': {'spacing': {'status': status, 'reason': status}}}
            else:
                output = {'artifact': images['original' if request.role == 'visual.image.generate' else 'edited'].to_dict(),
                    'actual_parameters': json.loads(request.messages[0]['content']), 'image_size': [16, 16]}
            value = {'model': self.model, 'output': output, 'usage': {'input_tokens': 1, 'output_tokens': 1,
                'images': 1 if request.role != 'visual.image.generate' else 0}, 'finish_reason': 'stop'}
            self.records[digest] = value
            return copy.deepcopy(value)
        def lookup(self, request, digest): return copy.deepcopy(self.records.get(digest))
    providers = {key: Provider('fixture-'+key) for key in ('image', 'review')}
    profiles = [ModelProfile(key, 'local_python', provider.model, 'unit-only', ('json', 'vision'))
        for key, provider in providers.items()]
    common = {'seed': 0, 'max_tokens': 128, 'timeout_seconds': 2}
    review_parameters = {**common, 'training_objective': 'Frozen count and spacing.', 'dimensions': ['spacing']}
    nodes = (
        DataNodeSpecV3('generate', 'visual.image.generate', '1', 'model', {'prompt': DataInput(artifact=source)},
            {**common, 'sampling': {}}, 'visual.image.generate', 1, 0),
        DataNodeSpecV3('before', 'visual.review.image', '1', 'model', {'media': DataInput(node='generate', output='media')},
            review_parameters, 'visual.review.image', 1, 1),
        DataNodeSpecV3('repair', module.OPERATOR_ID, '2', 'data',
            {'media': DataInput(node='generate', output='media'), 'review': DataInput(node='before', output='review')},
            {k: review_parameters[k] for k in ('training_objective', 'dimensions')}),
        DataNodeSpecV3('edit', 'visual.image.edit', '1', 'model',
            {'media': DataInput(node='generate', output='media'), 'prompt': DataInput(node='repair', output='prompt')},
            {**common, 'sampling': {}}, 'visual.image.edit', 1, 1),
        DataNodeSpecV3('after', 'visual.review.image', '1', 'model', {'media': DataInput(node='edit', output='media')},
            review_parameters, 'visual.review.image', 1, 1))
    contract = {'intent': DatasetIntent('fixture-repair', 'curate', 'visual_generation', 'fixture',
            'Unit-only complete repair graph.', (source,)).to_dict(),
        'quality_policy': QualityPolicy('fixture-repair', tuple(sorted(PROTECTED_GATES)),
            tuple(dict.fromkeys(n.operator for n in nodes)), {}, 5).to_dict(),
        'graph': DataNodeGraphV3(nodes).to_dict(), 'budget': DataRunBudget(5, 4, 3, 64, 2**24, 100).to_dict(),
        'stores': {'inputs': 'FIXTURE_INPUTS', 'data': 'FIXTURE_DATA'}, 'dataset_store': 'data',
        'models': {'profiles': [p.to_dict() for p in profiles], 'workers': {},
            'policy': ModelPolicy({'visual.image.generate': 'image', 'visual.image.edit': 'image',
                                  'visual.review.image': 'review'}, 4).to_dict()}}
    operator = DataGraphOperator(allowed_write_root=tmp_path/'graph-runtime', contract=contract,
        roots={'inputs': inputs, 'data': data}, registry=builtin_step_registry(repair_step=definition),
        providers={key: (lambda p=p: p) for key, p in providers.items()})
    controller = committed_workload(tmp_path, operator, contract, 'completion_ratio', 'complete-repair-graph')
    authority = CommittedEffectAuthority(controller, operator)
    reader = module.CommittedRepairRunAuthority(repair=authority, edited=authority,
        repair_node='repair', edit_node='edit', review_node='after', plan=None, package=package, store=store,
        request_authority=None, build_authority=build, runner=runner, candidate=True)
    value = reader.read()
    assert value['image'] == images['edited'].to_dict() and value['review']['status'] == 'pass'
    assert value['context']['image'] == images['original'].to_dict()
    assert reader.read() == value and sum(p.invocations for p in providers.values()) == 4
    for record in providers['image'].records.values(): record['output']['actual_parameters']['prompt'] = 'tampered-native'
    with pytest.raises(ValueError, match='original native effect'): reader.read()
    assert sum(p.invocations for p in providers.values()) == 4
