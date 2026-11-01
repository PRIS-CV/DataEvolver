"""Model-built, confined catalog selection with fixed gates and real outcomes."""
import copy
import hashlib
import json
import math
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from dataevolver.dataset.quality import counts, select_balanced
from .task_contracts import PROTECTED_GATES
from .artifact_operator import ArtifactWorkloadOperator, CommittedWorkloadAuthority
from .candidate_build_authority import CommittedCandidateBuildAuthority
from .candidate_sandbox import LinuxPureDataRunner
from .operator_development import inspect_candidate, QualificationProtocol
from .adapters.generation_benchmark import CommittedBenchmarkOutcomeAuthority

SCOPE = 'catalog_selection.v1'
OPERATOR_ID = 'images.select_balanced'


def eligible_rows(payload):
    """The host owns eligibility; a candidate can only order eligible IDs."""
    requirements = payload['requirements']
    if (set(requirements) != {'required_gates', 'review_required', 'required_categories'}
            or requirements['review_required'] is not True or not PROTECTED_GATES.issubset(requirements['required_gates'])
            or len(set(requirements['required_gates'])) != len(requirements['required_gates'])):
        raise ValueError('selection requires frozen gates and all required reviews')
    rows = payload['catalog']['rows']
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('selection catalog IDs must be unique')
    return [r for r in rows if r.get('review_status') == 'pass'
            and all(r.get('protected_gates', {}).get(g) is True for g in requirements['required_gates'])
            and r.get('quality', {}).get('clarity_proxy', 0) >= payload['parameters']['min_clarity']]


def validate_selection(payload, output):
    if not isinstance(output, dict) or set(output) != {'selected_ids'}:
        raise ValueError('selection output must contain only immutable row IDs')
    ids = output['selected_ids']
    eligible = {r['id']: r for r in eligible_rows(payload)}
    if not isinstance(ids, list) or len(set(ids)) != len(ids) or any(i not in eligible for i in ids):
        raise ValueError('candidate added, duplicated or selected an ineligible row')
    rows = [eligible[i] for i in ids]
    if (counts(rows) != payload['parameters']['quotas']
            or len({r.get('canonical_family_id',r['family_id']) for r in rows}) != len(rows)
            or not set(payload['requirements']['required_categories']).issubset({r['category'] for r in rows})):
        raise ValueError('candidate changed quotas, unique families or protected category coverage')
    return rows


def baseline_selection(payload):
    parameters = payload['parameters']
    rows = select_balanced(eligible_rows(payload), parameters['quotas'], ranking=parameters['ranking'],
                           seed=parameters['seed'], min_clarity=parameters['min_clarity'])
    result = {'selected_ids': [r['id'] for r in rows]}
    validate_selection(payload, result)
    return result


def selection_check_spec(phase, output_root, runner):
    from .models import OperatorSpec, ExpectedOutputSpec
    return OperatorSpec('evolution.selection_'+phase, '1', retry_safe=True,
        source_path=str(Path(__file__)), code_revision=file_sha256(Path(__file__)),
        model_config_hash=payload_sha256(runner.identity()), allowed_write_roots=(str(Path(output_root).resolve()),),
        expected_output_specs=(ExpectedOutputSpec('workload.json'),))


class SelectionCandidateChecks(ArtifactWorkloadOperator):
    def __init__(self, *, plan, package, store, request_authority, build_authority, runner,
                 cases, phase, outcome_authorities, output_root, minimum_delta=.05):
        if (phase not in {'engineering', 'independent'} or not isinstance(runner, LinuxPureDataRunner)
                or not isinstance(build_authority, CommittedCandidateBuildAuthority)
                or minimum_delta != .05 or not outcome_authorities
                or any(not isinstance(a, CommittedBenchmarkOutcomeAuthority) for a in outcome_authorities)):
            raise ValueError('selection qualification needs real build, benchmark and sandbox authorities')
        inspect_candidate(plan, package, store, request_authority)
        from .models import OperatorSpec
        spec=OperatorSpec.from_dict(json.loads(store.resolve(ArtifactRef.from_dict(package.operator_spec)).read_text()))
        if (spec.operator_id,spec.version)!=(OPERATOR_ID,'2'):
            raise ValueError('selection scope cannot qualify another operator or version')
        if build_authority.read()['package'] != package.to_dict():
            raise ValueError('selection package differs from the committed model build')
        self.plan, self.package, self.store = plan, package, store
        self.request_authority, self.build = request_authority, build_authority
        self.runner, self.cases, self.phase = runner, copy.deepcopy(cases), phase
        self.outcomes = tuple(outcome_authorities)
        self.protocol = QualificationProtocol.from_dict(json.loads(store.resolve(ArtifactRef.from_dict(plan.qualification_protocol)).read_text()))
        families, images, ids = {}, {}, set()
        for original in self.protocol.sources:
            raw = json.loads(store.resolve(ArtifactRef.from_dict(original['artifact'])).read_text())
            baseline_selection(raw)
            for row in raw['catalog']['rows']:
                ref = ArtifactRef.from_dict(row['image']); store.resolve(ref)
                if ref.store not in plan.protected_stores or row['id'] in ids:
                    raise ValueError('qualification IDs and media must have unique protected provenance')
                ids.add(row['id'])
                for mapping, key in [(families, row['family_id']), (images, ref.sha256)]:
                    owner = original['family_id']
                    if key in mapping and mapping[key] != owner:
                        raise ValueError('qualification cases share actual image/family provenance')
                    mapping[key] = owner
        splits = {'development', 'regression'} if phase == 'engineering' else {'independent'}
        expected = [s for s in self.protocol.sources if s['split'] in splits]
        if (cases != expected or not cases or (phase == 'independent' and len(cases) < 2)
                or (phase == 'engineering' and {c['split'] for c in cases} != splits)):
            raise ValueError('qualification must cover the exact predeclared isolated sources')
        self.source = ArtifactRef.from_dict(package.implementation)
        self._outcome_values()  # Refuse unavailable evidence before candidate execution.
        super().__init__(operator_id='evolution.selection_'+phase, output_root=output_root,
            configuration={'plan': plan.compute_hash(), 'package': package.compute_hash(),
                'build': build_authority.identity(), 'runtime': runner.identity(), 'cases': cases,
                'outcomes': [a.identity() for a in self.outcomes], 'phase': phase, 'scope': SCOPE, 'minimum_delta': .05})
        self.spec = selection_check_spec(phase, output_root, runner); self._declare_contract()
        if self.protocol.stages[phase]['validator_spec_sha256'] != self.spec.sha256():
            raise ValueError('selection validator differs from the frozen qualification protocol')

    def _outcome_values(self):
        by_image, origins = {}, []
        for authority in self.outcomes:
            value = authority.read(); report = value['report']; manifest = report['manifest']
            if manifest['qualification_plan_id'] != self.plan.plan_id:
                raise ValueError('independent outcomes cannot be reused for another candidate plan')
            origins.append({'origin': value['origin'], 'receipt_sha256': value['receipt_sha256']})
            for item in manifest['items']:
                result = report['results'][item['id']]['result']
                quality = result['score']['avg_acc'] if manifest['kind']=='spatial' else result['soft_tifa_gm']/100
                image = item['image_sha256']; values = by_image.setdefault(image, {})
                if manifest['kind'] in values:
                    raise ValueError('qualification benchmark image is duplicated')
                if not math.isfinite(quality) or not 0 <= quality <= 1:
                    raise ValueError('qualification requires finite observed benchmark outcomes')
                values[manifest['kind']] = quality
        result = {}
        for case in self.cases:
            payload = json.loads(self.store.resolve(ArtifactRef.from_dict(case['artifact'])).read_text())
            for row in payload['catalog']['rows']:
                image = row['image']['sha256']
                if image not in by_image or set(by_image[image]) != {'spatial', 'geneval2'}:
                    raise ValueError('all pool images need independent spatial and atomic benchmark evidence')
                result[row['id']] = math.sqrt(by_image[image]['spatial']*by_image[image]['geneval2'])
        return result, origins

    def _assess(self, case, execution, quality):
        payload = json.loads(self.store.resolve(ArtifactRef.from_dict(case['artifact'])).read_text())
        baseline = baseline_selection(payload)
        valid, reason, selected = False, '', []
        try:
            if execution['status'] != 'COMPLETED' or execution['isolation_verified'] is not True:
                raise ValueError('candidate did not complete in the confined runner')
            selected = validate_selection(payload, execution['output']); valid = True
        except (ValueError, KeyError, TypeError) as exc:
            reason = str(exc)
        ids = [r['id'] for r in selected]
        baseline_quality = sum(quality[i] for i in baseline['selected_ids'])/len(baseline['selected_ids'])
        measured = sum(quality[i] for i in ids)/len(ids) if ids else 0.
        return {'case': case, 'execution': execution, 'valid': valid, 'reason': reason,
                'baseline_ids': baseline['selected_ids'], 'selected_ids': ids,
                'behavior_changed': valid and set(ids) != set(baseline['selected_ids']),
                'baseline_quality': baseline_quality, 'candidate_quality': measured,
                'delta': measured-baseline_quality}

    def _report(self, runs, origins):
        valid = all(r['valid'] for r in runs)
        delta = sum(r['delta'] for r in runs)/len(runs)
        changed = any(r['behavior_changed'] for r in runs)
        regression_ok = all(r['delta'] >= -1e-12 for r in runs if r['case']['split']=='regression')
        passed = valid and changed and regression_ok and (self.phase!='independent' or delta+1e-12 >= .05)
        return {'scope': SCOPE, 'phase': self.phase, 'package_sha256': self.package.compute_hash(),
            'runtime': self.runner.identity(), 'build': self.build.read(), 'outcome_origins': origins,
            'runs': runs, 'mean_delta': delta, 'minimum_delta': .05, 'behavior_changed': changed,
            'protected_checks_passed': valid, 'regression_preserved': regression_ok, 'passed': passed,
            'quality_scope': 'joint_spatial_and_atomic_native_benchmark_proxy',
            'production_adoption_allowed': False, 'training_benefit_verified': False,
            'method_improvement_verified': False}

    def workload(self, call):
        quality, origins = self._outcome_values(); runs = []
        source = self.store.resolve(self.source).read_text()
        for case in self.cases:
            payload = json.loads(self.store.resolve(ArtifactRef.from_dict(case['artifact'])).read_text())
            execution = self.runner.run(source, payload)
            runs.append(self._assess(case, execution, quality))
        return self._report(runs, origins)

    def validate_report(self, call, report):
        inspect_candidate(self.plan, self.package, self.store, self.request_authority)
        quality, origins = self._outcome_values()
        if [r['case'] for r in report['runs']] != self.cases:
            raise ValueError('selection qualification source membership changed')
        runs = []
        for saved, case in zip(report['runs'], self.cases):
            execution = saved['execution']
            payload = json.loads(self.store.resolve(ArtifactRef.from_dict(case['artifact'])).read_text())
            if (execution['binding']['source_sha256'] != self.source.sha256
                    or execution['binding']['input_sha256'] != hashlib.sha256(json.dumps(payload,allow_nan=False,separators=(',',':')).encode()).hexdigest()
                    or execution['runtime'] != self.runner.identity()):
                raise ValueError('selection invocation source, input or isolation changed')
            runs.append(self._assess(case, execution, quality))
        if report != self._report(runs, origins):
            raise ValueError('selection qualification verdict or evidence changed')

    def result(self, call, report):
        value = super().result(call, report)
        value['metrics'].update(qualification_passed=float(report['passed']), quality_delta=report['mean_delta'])
        return value


class SelectionQualificationAuthority:
    def __init__(self, *, engineering, independent, package, runner, lifecycle_total, resource_deadline_epoch):
        if any(not isinstance(a, CommittedWorkloadAuthority) or not isinstance(a.operator, SelectionCandidateChecks)
               for a in (engineering, independent)):
            raise TypeError('selection adoption needs actual committed qualification workloads')
        self.engineering, self.independent = engineering, independent
        self.package, self.runner = package, runner
        self.total, self.deadline = dict(lifecycle_total), resource_deadline_epoch

    def read(self):
        values = [self.engineering.read(), self.independent.read()]
        for value, phase in zip(values, ['engineering', 'independent']):
            report = value['report']
            if (report['scope'] != SCOPE or report['phase'] != phase or report['passed'] is not True
                    or report['package_sha256'] != self.package.compute_hash() or report['runtime'] != self.runner.identity()):
                raise ValueError('selection candidate did not pass its frozen real qualification')
        for authority,phase in [(self.engineering,'engineering'),(self.independent,'independent')]:
            if authority.controller.evaluator.contract.sha256!=authority.operator.protocol.stages[phase]['evaluator_sha256']:
                raise ValueError('selection qualification evaluator differs from its predeclared protocol')
        if values[0]['origin']['campaign_id'] == values[1]['origin']['campaign_id']:
            raise ValueError('selection qualification campaigns must be independent')
        all_cases = [r['case'] for value in values for r in value['report']['runs']]
        if (len({c['family_id'] for c in all_cases}) != len(all_cases)
                or len({c['artifact']['sha256'] for c in all_cases}) != len(all_cases)):
            raise ValueError('selection qualification source families or contents overlap')
        return {'operator_id': OPERATOR_ID, 'version': '2', 'scope': SCOPE,
            'package_sha256': self.package.compute_hash(), 'implementation': self.package.implementation,
            'runtime': self.runner.identity(), 'engineering': values[0]['origin'], 'independent': values[1]['origin'],
            'qualification_receipts': [v['receipt_sha256'] for v in values],
            'build_origin': values[0]['report']['build']['origin'],
            'lifecycle_total': self.total, 'resource_deadline_epoch': self.deadline,
            'production_adoption_allowed': False, 'model_authority': False, 'image_authority': False,
            'training_benefit_verified': False}

    def verify(self, snapshot):
        if snapshot != self.read():
            raise ValueError('selection qualification snapshot changed')


def qualified_selection_step(*, catalog, adoption, qualification, expected_head, store, runner, requirements):
    from .operator_sdk import StepDefinition
    from .models import OperatorSpec
    from .operator_contracts import OperatorContract
    snapshot = catalog.active(adoption.read(), scope=SCOPE, head=expected_head, qualification_authority=qualification)
    ref = ArtifactRef.from_dict(snapshot['implementation']); source = store.resolve(ref).read_text()
    required = copy.deepcopy(requirements)
    def execute(ctx):
        with catalog.lease():
            catalog.active(adoption.read(), scope=SCOPE, head=expected_head, qualification_authority=qualification)
            value = ctx.read('catalog')
            if not set(ctx.policy.required_gates).issubset(required['required_gates']):
                raise ValueError('qualified selector cannot weaken current protected gates')
            payload = {'catalog': value, 'parameters': dict(ctx.parameters), 'requirements': required}
            execution = runner.run(source, payload)
            if execution['status'] != 'COMPLETED' or execution['isolation_verified'] is not True:
                raise ValueError('qualified selector did not complete in the confined runner')
            selected = validate_selection(payload, execution['output'])
            value.update(rows=selected, selection_policy=dict(ctx.parameters))
            ctx.write_json('selection-execution.json', execution)
            return {'catalog': ctx.write_json('selected.json', value)}
    spec = OperatorSpec(OPERATOR_ID, '2', retry_safe=True, source_path=str(Path(__file__)),
        code_revision=payload_sha256([ref.to_dict(), runner.identity(), required, file_sha256(Path(__file__))]))
    contract = OperatorContract(OPERATOR_ID, '2', spec.sha256(), 'basic', {'catalog': {'type': 'object'}},
        {'catalog': {'type': 'object'}}, {'type': 'object', 'additionalProperties': False,
            'properties': {k: {} for k in ['quotas', 'ranking', 'seed', 'min_clarity']},
            'required': ['quotas', 'ranking', 'seed', 'min_clarity']}, ('json', 'image'),
        'pure_data', 'verify_then_reuse', ('Confined selection only; all immutable host gates and quotas remain required.',))
    return StepDefinition(spec, contract, 'data', execute)
