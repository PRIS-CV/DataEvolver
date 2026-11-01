"""Exact repair/edit/Review/benchmark evidence under existing host authorities.

Reading these authorities never executes a candidate or invokes a model. A
qualification needs complete matched runs, including native regression cases.
"""
import copy
import json
import math
from dataclasses import replace
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from dataevolver.dataset.visual_steps import confirmed, repair_context
from .artifact_operator import ArtifactWorkloadOperator, CommittedEffectAuthority, CommittedWorkloadAuthority
from .adapters.data_graph import DataGraphOperator
from .adapters.generation_benchmark import GenerationBenchmarkOperator
from .candidate_build_authority import CommittedCandidateBuildAuthority
from .candidate_sandbox import LinuxPureDataRunner
from .operator_development import QualificationProtocol, inspect_candidate
from .operator_sdk import builtin_step_registry
from .repair_candidate import SCOPE, OPERATOR_ID, candidate_repair_step, checked_instruction, instruction_payload


def native_visual_receipt(ctx, envelope, roles):
    gateway = ctx.graph.gateway(ctx.parent)
    request, receipt = confirmed(gateway, envelope, roles)
    if receipt.backend == 'mock':
        raise ValueError('repair qualification requires a native original image and source Review, edit and post Review')
    owner = gateway._gateway(request.role) if hasattr(gateway, '_gateway') else gateway
    profile, model, endpoint, profile_hash, digest = owner._prepare(request)
    native = owner._provider(profile, model, endpoint).lookup(request, digest)
    if (native is None or native['model'] != receipt.model or profile_hash != receipt.profile_hash
            or native['output'] != receipt.output
            or any(native['usage'].get(k) != receipt.usage.get(k)
                   for k in ('input_tokens', 'output_tokens', 'images'))):
        raise ValueError('repair model receipt differs from its original native effect')
    return request, receipt


class CommittedRepairBenchmarkAuthority(CommittedWorkloadAuthority):
    def __init__(self, controller, operator):
        if not isinstance(operator, GenerationBenchmarkOperator):
            raise TypeError('repair outcomes require the real benchmark workload')
        super().__init__(controller, operator)

    def read(self):
        value = super().read()
        manifest = value['report']['manifest']
        if (manifest.get('purpose') != 'repair_qualification' or not manifest.get('qualification_plan_id')
                or manifest.get('split') != 'development'):
            raise ValueError('repair outcomes need their own frozen purpose and plan')
        return value


class CommittedRepairRunAuthority:
    """Bind a pure repair step to the exact native image edit and goal Review.

    The repair and edit may belong to separate committed graphs, allowing an
    existing fixed baseline to be reused without resending its model requests.
    """
    def __init__(self, *, repair, edited, repair_node, edit_node, review_node,
                 plan, package, store, request_authority, build_authority, runner, candidate,
                 adoption=None, qualification=None):
        if (any(not isinstance(a, CommittedEffectAuthority) or not isinstance(a.operator, DataGraphOperator)
                for a in (repair, edited)) or type(candidate) is not bool):
            raise TypeError('repair evidence requires committed data graphs')
        self.repair, self.edited = repair, edited
        self.nodes = repair_node, edit_node, review_node
        self.plan, self.package, self.store = plan, package, store
        self.request, self.build, self.runner, self.candidate = request_authority, build_authority, runner, candidate
        self.adoption, self.qualification = adoption, qualification
        if adoption is not None or qualification is not None:
            from .operator_lifecycle import CommittedOperatorAdoptionAuthority
            if (not candidate or not isinstance(adoption, CommittedOperatorAdoptionAuthority)
                    or not isinstance(qualification, RepairQualificationAuthority)):
                raise TypeError('adopted repair evidence requires committed adoption and exact qualification')

    def identity(self):
        return {'repair': self.repair.identity(), 'edited': self.edited.identity(),
                'nodes': list(self.nodes), 'candidate': self.candidate, 'package_sha256': self.package.compute_hash(),
                'adoption': self.adoption.identity() if self.adoption is not None else None}

    @staticmethod
    def _node(authority, name):
        call, _ = authority.read_effect()
        operator = authority.operator
        outputs = {}
        for node in operator.graph.ordered():
            child, step = operator._step(call, node, outputs, operator.journal(call))
            effect = json.loads((Path(child.work_dir)/'effect.json').read_text())
            if node.node_id == name:
                return step, effect
            outputs[node.node_id] = {k: ArtifactRef.from_dict(v) for k, v in effect['outputs'].items()}
        raise ValueError('required repair evidence node is absent')

    @staticmethod
    def _json(step, ref):
        return json.loads(step.ctx.store.resolve(ArtifactRef.from_dict(ref)).read_text())

    def read(self):
        repair, effect = self._node(self.repair, self.nodes[0])
        expected = (candidate_repair_step(plan=self.plan, package=self.package, store=self.store,
            request_authority=self.request, build_authority=self.build, runner=self.runner) if self.candidate
            else builtin_step_registry().step(OPERATOR_ID, '1'))
        expected_spec = expected.spec
        if self.adoption is not None:
            event = self.adoption.read()
            snapshot = event['snapshot']
            self.qualification.verify(snapshot)
            if (snapshot['scope'] != SCOPE or snapshot['package_sha256'] != self.package.compute_hash()
                    or snapshot['runtime'] != self.runner.identity()):
                raise ValueError('adopted repair run changed its qualified package or runtime')
            expected_spec = adopted_repair_spec(expected, event['sha256'], snapshot)
        if repair.spec.sha256() != expected_spec.sha256() or effect['model_requests']:
            raise ValueError('repair run did not use the exact pure host-bound implementation')
        context = repair_context(repair.ctx)
        native_visual_receipt(repair.ctx, repair.ctx.read('review'), {'visual.review.image'})
        native_visual_receipt(repair.ctx, repair.ctx.read('media'),
            {'visual.image.generate', 'visual.image.edit'})
        prompt = self._json(repair, effect['outputs']['prompt'])['prompt']
        trace = self._json(repair, effect['outputs']['repair'])
        if self.candidate:
            evidence = [r for r in effect['artifacts'] if r['path'].endswith('/candidate-repair-execution.json')]
            if len(evidence) != 1:
                raise ValueError('repair run lacks its exact sandbox execution')
            execution = self._json(repair, evidence[0])
            if (execution['scope'] != SCOPE or execution['package_sha256'] != self.package.compute_hash()
                    or execution['build_origin'] != self.build.identity()):
                raise ValueError('repair execution belongs to another package or build')
            instruction = checked_instruction(execution['execution'],
                source=self.store.resolve(ArtifactRef.from_dict(self.package.implementation)).read_text(),
                payload=instruction_payload(context), runtime=self.runner.identity())
        else:
            instruction = repair.ctx.parameters['repair_instruction']
        expected_prompt = (instruction+'\nOriginal unchanged goal: '+context['training_objective']
                           +'\nRecorded failed criteria: '+json.dumps(context['reasons'], sort_keys=True))
        expected_trace = {'schema_version': 'dataevolver.repair_prompt.v1', 'source_image': context['image'],
            'source_media_sha256': context['source_media_sha256'],
            'review_request_sha256': context['review_request_sha256'], 'review_receipt_hash': context['review_receipt_hash'],
            'training_objective_sha256': payload_sha256(context['training_objective']),
            'dimensions': context['dimensions'], 'instruction_sha256': payload_sha256(instruction),
            'prompt_sha256': payload_sha256(prompt), 'production_qualification': False}
        if prompt != expected_prompt or trace != expected_trace:
            raise ValueError('repair prompt or host source binding changed')
        edit, image_effect = self._node(self.edited, self.nodes[1])
        review, review_effect = self._node(self.edited, self.nodes[2])
        registry = builtin_step_registry()
        for step, name in [(edit, 'visual.image.edit'), (review, 'visual.review.image')]:
            if step.spec.sha256() != registry.step(name, '1').spec.sha256():
                raise ValueError('repair quality evidence used a different host image or Review step')
        if edit.ctx.read('prompt') != {'prompt': prompt} or edit.ctx.read('media') != repair.ctx.read('media'):
            raise ValueError('native edit did not consume the exact repair prompt and original media')
        envelope = self._json(edit, image_effect['outputs']['media'])
        request, receipt = native_visual_receipt(edit.ctx, envelope, {'visual.image.edit'})
        parameters = json.loads(request.messages[0]['content'])
        image = receipt.output['artifact']
        if (receipt.backend == 'mock' or request.artifacts != (ArtifactRef.from_dict(context['image']),)
                or parameters['prompt'] != prompt or receipt.output['actual_parameters'] != parameters):
            raise ValueError('repair image is not its bound actual native edit')
        after = self._json(review, review_effect['outputs']['review'])
        review_request, review_receipt = native_visual_receipt(review.ctx, after, {'visual.review.image'})
        reviewed_context = json.loads(json.loads(review_request.messages[0]['content'])['prompt'].rsplit('\nREVIEW_CONTEXT=', 1)[1])
        if (review_receipt.backend == 'mock' or review.ctx.read('media') != envelope
                or review_request.artifacts != (ArtifactRef.from_dict(image),)
                or reviewed_context != {'training_objective': context['training_objective'],
                    'dimensions': context['dimensions'], 'generation_prompt': prompt}
                or sorted(review_receipt.output['criteria']) != sorted(context['dimensions'])):
            raise ValueError('post-edit Review changed the image, objective or dimensions')
        return {'origin': self.identity(), 'context': context, 'image': image, 'prompt_sha256': payload_sha256(prompt),
            'instruction_sha256': payload_sha256(instruction),
            'review': review_receipt.output, 'matching': {'edit_seed': request.seed,
                'sampling': {k: v for k, v in parameters.items() if k != 'prompt'},
                'edit_model': receipt.model, 'edit_profile_hash': receipt.profile_hash,
                'review_model': review_receipt.model, 'review_profile_hash': review_receipt.profile_hash},
            'package_sha256': self.package.compute_hash() if self.candidate else None}


def repair_check_spec(phase, output_root, runner):
    from .models import OperatorSpec, ExpectedOutputSpec
    return OperatorSpec('evolution.repair_'+phase, '1', retry_safe=True,
        source_path=str(Path(__file__)), code_revision=file_sha256(Path(__file__)),
        model_config_hash=payload_sha256(runner.identity()), allowed_write_roots=(str(Path(output_root).resolve()),),
        expected_output_specs=(ExpectedOutputSpec('workload.json'),))


def repair_protocol_context(descriptor):
    return {'scope': SCOPE, **{key: descriptor[key] for key in
        ('training_objective', 'dimensions', 'benchmark_profiles_sha256')}}


class RepairCandidateChecks(ArtifactWorkloadOperator):
    def __init__(self, *, plan, package, store, request_authority, build_authority, runner,
                 cases, phase, run_pairs, outcome_authorities, output_root, minimum_delta=.05):
        if (phase not in {'engineering', 'independent'} or minimum_delta != .05
                or not isinstance(build_authority, CommittedCandidateBuildAuthority)
                or not isinstance(runner, LinuxPureDataRunner) or not outcome_authorities
                or any(not isinstance(a, CommittedRepairBenchmarkAuthority) for a in outcome_authorities)):
            raise ValueError('repair qualification requires real build, sandbox and repair benchmark authorities')
        inspect_candidate(plan, package, store, request_authority)
        if build_authority.read()['package'] != package.to_dict():
            raise ValueError('repair qualification package differs from its model build')
        self.plan, self.package, self.store, self.request = plan, package, store, request_authority
        self.build, self.runner, self.phase = build_authority, runner, phase
        self.protocol = QualificationProtocol.from_dict(json.loads(store.resolve(ArtifactRef.from_dict(plan.qualification_protocol)).read_text()))
        models = []
        for source in self.protocol.sources:
            ref = ArtifactRef.from_dict(source['artifact'])
            if ref.store not in plan.protected_stores:
                raise ValueError('repair qualification sources must remain protected')
            descriptor = json.loads(store.resolve(ref).read_text())
            if payload_sha256(repair_protocol_context(descriptor)) != self.protocol.matched_context_sha256:
                raise ValueError('repair sources differ from the frozen method and evaluator context')
            models.append(descriptor['benchmark_profiles_sha256'])
        if any(m != models[0] or set(m) != {'spatial', 'geneval2'} for m in models):
            raise ValueError('repair stages must freeze the same two evaluator configurations')
        splits = {'development', 'regression'} if phase == 'engineering' else {'independent'}
        expected = [s for s in self.protocol.sources if s['split'] in splits]
        if cases != expected or not cases or set(run_pairs) != {s['family_id'] for s in cases}:
            raise ValueError('repair qualification must cover every exact predeclared source')
        self.cases, self.pairs, self.outcomes = copy.deepcopy(cases), dict(run_pairs), tuple(outcome_authorities)
        for pair in self.pairs.values():
            if (len(pair) != 2 or any(not isinstance(a, CommittedRepairRunAuthority) for a in pair)
                    or pair[0].candidate or not pair[1].candidate or pair[1].adoption is not None):
                raise TypeError('repair qualification requires fixed and exact candidate native runs')
        super().__init__(operator_id='evolution.repair_'+phase, output_root=output_root,
            configuration={'plan': plan.compute_hash(), 'package': package.compute_hash(), 'phase': phase,
                'scope': SCOPE, 'runtime': runner.identity(), 'cases': cases, 'minimum_delta': .05,
                'pairs': {k: [a.identity() for a in v] for k, v in self.pairs.items()},
                'outcomes': [a.identity() for a in self.outcomes]})
        self.spec = repair_check_spec(phase, output_root, runner); self._declare_contract()
        if self.spec.sha256() != self.protocol.stages[phase]['validator_spec_sha256']:
            raise ValueError('repair validator differs from the frozen protocol')

    def _report(self):
        scores, origins, evaluator = {}, [], {}
        for authority in self.outcomes:
            value = authority.read(); report = value['report']; manifest = report['manifest']; kind = manifest['kind']
            if manifest['qualification_plan_id'] != self.plan.plan_id:
                raise ValueError('repair benchmark belongs to another plan')
            # Model contracts include actual evaluator revision and dtype configuration.
            identity = payload_sha256(authority.operator.gateway.identity('benchmark.'+kind))
            if kind in evaluator and evaluator[kind] != identity:
                raise ValueError('fixed and candidate benchmarks used different evaluator versions')
            evaluator[kind] = identity; origins.append({'origin': value['origin'], 'receipt_sha256': value['receipt_sha256']})
            for item in manifest['items']:
                key = item['image_sha256'], kind
                result = report['results'][item['id']]['result']
                quality = result['score']['avg_acc'] if kind == 'spatial' else result['soft_tifa_gm']/100
                if key in scores or not math.isfinite(quality) or not 0 <= quality <= 1:
                    raise ValueError('repair benchmark has duplicate or invalid image scores')
                scores[key] = (quality, item)
        runs, images = [], set()
        for case in self.cases:
            descriptor = json.loads(self.store.resolve(ArtifactRef.from_dict(case['artifact'])).read_text())
            baseline, candidate = [a.read() for a in self.pairs[case['family_id']]]
            context = baseline['context']
            if (candidate['context'] != context or baseline['matching'] != candidate['matching']
                    or candidate['package_sha256'] != self.package.compute_hash()
                    or context['training_objective'] != descriptor['training_objective']
                    or context['dimensions'] != descriptor['dimensions'] or context['caption'] != descriptor['generation_prompt']
                    or payload_sha256(context) != descriptor['source_context_sha256']
                    or baseline['matching'] != descriptor['matching']
                    or baseline['instruction_sha256'] != payload_sha256(descriptor['baseline_instruction'])
                    or evaluator != descriptor['benchmark_profiles_sha256']):
                raise ValueError('repair comparison changed its source, model, goal, dimensions or sampling')
            if context['image']['sha256'] in images:
                raise ValueError('repair cases reuse actual source images')
            images.add(context['image']['sha256'])
            quality = []
            for run in (baseline, candidate):
                observed = {}
                for kind in ('spatial', 'geneval2'):
                    score, item = scores[(run['image']['sha256'], kind)]
                    keys = ('questions', 'answers') if kind == 'spatial' else ('atoms',)
                    if any(item[k] != descriptor[k] for k in keys):
                        raise ValueError('repair benchmark changed the predeclared questions or answers')
                    observed[kind] = score
                quality.append(math.sqrt(observed['spatial']*observed['geneval2']))
            protected = all(candidate['review']['criteria'][key]['status'] == 'pass'
                for key, value in context['review']['criteria'].items() if value['status'] == 'pass')
            runs.append({'case': case, 'baseline': baseline, 'candidate': candidate,
                'baseline_quality': quality[0], 'candidate_quality': quality[1], 'delta': quality[1]-quality[0],
                'protected_checks_passed': protected and candidate['review']['status'] == 'pass',
                'behavior_changed': baseline['prompt_sha256'] != candidate['prompt_sha256']})
        expected_scores = {(run[side]['image']['sha256'], kind) for run in runs
            for side in ('baseline', 'candidate') for kind in ('spatial', 'geneval2')}
        if set(scores) != expected_scores:
            raise ValueError('repair benchmark includes images outside the frozen comparisons')
        delta = sum(r['delta'] for r in runs)/len(runs)
        protected = all(r['protected_checks_passed'] for r in runs)
        regression = all(r['delta'] >= -1e-12 for r in runs if r['case']['split'] == 'regression')
        changed = any(r['behavior_changed'] for r in runs)
        return {'scope': SCOPE, 'phase': self.phase, 'package_sha256': self.package.compute_hash(),
            'runtime': self.runner.identity(), 'build': self.build.read(), 'runs': runs, 'outcome_origins': origins,
            'mean_delta': delta, 'minimum_delta': .05, 'protected_checks_passed': protected,
            'regression_preserved': regression, 'behavior_changed': changed,
            'passed': protected and regression and changed and (self.phase != 'independent' or delta+1e-12 >= .05),
            'quality_scope': 'joint_spatial_and_atomic_native_benchmark_proxy',
            'production_adoption_allowed': False, 'training_benefit_verified': False}

    def workload(self, call):
        return self._report()

    def validate_report(self, call, report):
        inspect_candidate(self.plan, self.package, self.store, self.request)
        if report != self._report():
            raise ValueError('repair qualification evidence or verdict changed')

    def result(self, call, report):
        value = super().result(call, report)
        value['metrics'].update(qualification_passed=float(report['passed']), quality_delta=report['mean_delta'])
        return value


class RepairQualificationAuthority:
    def __init__(self, *, engineering, independent, package, runner, lifecycle_total, resource_deadline_epoch):
        if (not isinstance(runner, LinuxPureDataRunner) or any(
               not isinstance(a, CommittedWorkloadAuthority) or not isinstance(a.operator, RepairCandidateChecks)
               for a in (engineering, independent))):
            raise TypeError('repair adoption requires actual committed qualification workloads')
        self.engineering, self.independent, self.package, self.runner = engineering, independent, package, runner
        self.total, self.deadline = dict(lifecycle_total), resource_deadline_epoch

    def read(self):
        values = [self.engineering.read(), self.independent.read()]
        plans = [a.operator.plan.compute_hash() for a in (self.engineering, self.independent)]
        if plans[0] != plans[1]:
            raise ValueError('repair qualification stages belong to different frozen plans')
        for authority, value, phase in zip((self.engineering, self.independent), values, ('engineering', 'independent')):
            report = value['report']
            if (report['scope'] != SCOPE or report['phase'] != phase or report['passed'] is not True
                    or report['package_sha256'] != self.package.compute_hash() or report['runtime'] != self.runner.identity()
                    or authority.controller.evaluator.contract.sha256 != authority.operator.protocol.stages[phase]['evaluator_sha256']):
                raise ValueError('repair candidate did not pass its exact frozen qualification')
        cases = [r['case'] for v in values for r in v['report']['runs']]
        if (values[0]['origin']['campaign_id'] == values[1]['origin']['campaign_id']
                or len({c['family_id'] for c in cases}) != len(cases)
                or len({c['artifact']['sha256'] for c in cases}) != len(cases)):
            raise ValueError('repair qualification campaigns or source families overlap')
        return {'operator_id': OPERATOR_ID, 'version': '2', 'scope': SCOPE,
            'package_sha256': self.package.compute_hash(), 'implementation': self.package.implementation,
            'runtime': self.runner.identity(), 'engineering': values[0]['origin'], 'independent': values[1]['origin'],
            'qualification_receipts': [v['receipt_sha256'] for v in values], 'build_origin': values[0]['report']['build']['origin'],
            'lifecycle_total': self.total, 'resource_deadline_epoch': self.deadline,
            'production_adoption_allowed': False, 'model_authority': False, 'image_authority': False,
            'training_benefit_verified': False}

    def verify(self, snapshot):
        if snapshot != self.read():
            raise ValueError('repair qualification snapshot changed')


def adopted_repair_spec(step, head, snapshot):
    return step.spec.with_updates(code_revision=payload_sha256([step.spec.code_revision, head, snapshot]))


def qualified_repair_step(*, catalog, adoption, qualification, expected_head, plan, package, store,
                          request_authority, build_authority, runner):
    from .operator_sdk import StepDefinition
    snapshot = catalog.active(adoption.read(), scope=SCOPE, head=expected_head, qualification_authority=qualification)
    if (snapshot['package_sha256'] != package.compute_hash() or snapshot['runtime'] != runner.identity()
            or adoption.read()['sha256'] != expected_head):
        raise ValueError('adopted repair package differs from the requested implementation')
    step = candidate_repair_step(plan=plan, package=package, store=store,
        request_authority=request_authority, build_authority=build_authority, runner=runner)
    def execute(ctx):
        with catalog.lease():
            catalog.active(adoption.read(), scope=SCOPE, head=expected_head, qualification_authority=qualification)
            return step.implementation(ctx)
    spec = adopted_repair_spec(step, expected_head, snapshot)
    return StepDefinition(spec, replace(step.contract, spec_sha256=spec.sha256()), step.kind, execute)
