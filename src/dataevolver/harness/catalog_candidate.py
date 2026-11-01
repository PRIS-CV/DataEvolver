"""Qualified catalog-family candidates in the existing confined runner/lifecycle."""
import copy
import json
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from dataevolver.dataset.families import assign_families
from .artifact_operator import ArtifactWorkloadOperator, CommittedWorkloadAuthority
from .candidate_sandbox import LinuxPureDataRunner
from .operator_development import inspect_candidate

SCOPE = 'catalog_family.v1'


def catalog_check_spec(phase, output_root, runner):
    from .models import OperatorSpec, ExpectedOutputSpec
    return OperatorSpec('evolution.catalog_'+phase, '1', retry_safe=True,
        source_path=str(Path(__file__)), code_revision=file_sha256(Path(__file__)),
        model_config_hash=payload_sha256(runner.identity()), allowed_write_roots=(str(Path(output_root).resolve()),),
        expected_output_specs=(ExpectedOutputSpec('workload.json'),))


class CatalogCandidateChecks(ArtifactWorkloadOperator):
    def __init__(self, *, plan, package, store, request_authority, runner, cases, phase, output_root):
        if phase not in {'engineering', 'independent', 'invocation'} or not isinstance(runner, LinuxPureDataRunner):
            raise ValueError('catalog candidate needs an explicit isolated runner/phase')
        inspect_candidate(plan, package, store, request_authority)
        self.plan, self.package, self.store, self.authority = plan, package, store, request_authority
        self.runner, self.cases, self.phase = runner, copy.deepcopy(cases), phase
        self.source_ref = ArtifactRef.from_dict(package.implementation)
        if not cases or len({c['family_id'] for c in cases}) != len(cases):
            raise ValueError('qualification cases need unique source families')
        for case in cases: store.resolve(ArtifactRef.from_dict(case['artifact']))
        super().__init__(operator_id='evolution.catalog_'+phase, output_root=output_root,
            configuration={'plan': plan.compute_hash(), 'package': package.compute_hash(),
                'runtime': runner.identity(), 'cases': cases, 'phase': phase, 'scope': SCOPE},
            parameter_schema={'type':'object','additionalProperties':False,
                'properties':{k:{'type':'string','pattern':'^[a-f0-9]{64}$'} for k in
                    ['plan_sha256','package_sha256','cases_sha256','runtime_sha256']},
                'required':['plan_sha256','package_sha256','cases_sha256','runtime_sha256']})
        self.spec = catalog_check_spec(phase, output_root, runner)
        self._declare_contract()
        if phase != 'invocation':
            from .operator_development import QualificationProtocol
            protocol = QualificationProtocol.from_dict(json.loads(store.resolve(ArtifactRef.from_dict(plan.qualification_protocol)).read_text()))
            if protocol.stages[phase]['validator_spec_sha256'] != self.spec.sha256():
                raise ValueError('qualification validator differs from the predeclared protocol')

    def parameters(self):
        return {'plan_sha256': self.plan.compute_hash(), 'package_sha256': self.package.compute_hash(),
                'cases_sha256': payload_sha256(self.cases), 'runtime_sha256': payload_sha256(self.runner.identity())}

    def path(self, call):
        if call.parameters != self.parameters():
            raise ValueError('catalog candidate call differs from the frozen workload binding')
        return super().path(call)

    def workload(self, call):
        source = self.store.resolve(self.source_ref).read_text()
        runs = []
        for case in self.cases:
            payload = json.loads(self.store.resolve(ArtifactRef.from_dict(case['artifact'])).read_text())
            expected = {'rows': assign_families(payload['rows'], payload['hamming_distance'])}
            result = self.runner.run(source, payload)
            passed = result['status'] == 'COMPLETED' and result['isolation_verified'] is True and result['output'] == expected
            runs.append({'case': case, 'execution': result, 'expected_sha256': payload_sha256(expected), 'passed': passed})
        return {'scope': SCOPE, 'phase': self.phase, 'package_sha256': self.package.compute_hash(),
                'source_sha256': self.source_ref.sha256, 'runtime': self.runner.identity(), 'runs': runs,
                'passed': all(r['passed'] for r in runs), 'production_adoption_allowed': False,
                'method_improvement_verified': False}

    def validate_report(self, call, report):
        inspect_candidate(self.plan, self.package, self.store, self.authority)
        if (report['scope'] != SCOPE or report['phase'] != self.phase
                or report['package_sha256'] != self.package.compute_hash()
                or report['source_sha256'] != self.source_ref.sha256 or report['runtime'] != self.runner.identity()
                or [r['case'] for r in report['runs']] != self.cases):
            raise ValueError('catalog candidate qualification binding changed')
        for run in report['runs']:
            ref = ArtifactRef.from_dict(run['case']['artifact'])
            payload = json.loads(self.store.resolve(ref).read_text())
            expected = {'rows': assign_families(payload['rows'], payload['hamming_distance'])}
            execution = run['execution']
            if (execution['binding']['source_sha256'] != self.source_ref.sha256
                    or execution['binding']['input_sha256'] != __import__('hashlib').sha256(json.dumps(payload, allow_nan=False, separators=(',', ':')).encode()).hexdigest()
                    or execution['runtime'] != report['runtime'] or run['expected_sha256'] != payload_sha256(expected)
                    or run['passed'] != (execution['status'] == 'COMPLETED' and execution['isolation_verified'] is True and execution['output'] == expected)):
                raise ValueError('catalog candidate workload evidence changed')
        if report['passed'] != all(r['passed'] for r in report['runs']): raise ValueError('qualification verdict changed')

    def result(self, call, report):
        result = super().result(call, report)
        result['metrics']['qualification_passed'] = float(report['passed'])
        return result


class CatalogQualificationAuthority:
    """Two real frozen Controller workloads qualify only this confined scope."""
    def __init__(self, *, engineering, independent, package, runner, lifecycle_total, resource_deadline_epoch):
        if not all(isinstance(a, CommittedWorkloadAuthority) for a in (engineering, independent)):
            raise ValueError('qualification needs actual committed workload readers')
        self.engineering, self.independent, self.package, self.runner = engineering, independent, package, runner
        self.total, self.deadline = dict(lifecycle_total), resource_deadline_epoch

    def read(self):
        eng, ind = self.engineering.read(), self.independent.read()
        for entry, phase in [(eng, 'engineering'), (ind, 'independent')]:
            report = entry['report']
            if (report['scope'] != SCOPE or report['phase'] != phase or report['passed'] is not True
                    or report['package_sha256'] != self.package.compute_hash() or report['runtime'] != self.runner.identity()):
                raise ValueError('catalog qualification did not pass its real workload')
        ef = {r['case']['family_id'] for r in eng['report']['runs']}
        inf = {r['case']['family_id'] for r in ind['report']['runs']}
        eh = {r['case']['artifact']['sha256'] for r in eng['report']['runs']}
        ih = {r['case']['artifact']['sha256'] for r in ind['report']['runs']}
        if ef & inf or eh & ih or len(inf) < 2 or eng['origin']['campaign_id'] == ind['origin']['campaign_id']:
            raise ValueError('engineering/independent catalog workloads are not isolated')
        return {'operator_id': 'images.families', 'version': '2', 'scope': SCOPE,
                'package_sha256': self.package.compute_hash(), 'implementation': self.package.implementation,
                'runtime': self.runner.identity(), 'engineering': eng['origin'], 'independent': ind['origin'],
                'qualification_receipts': [eng['receipt_sha256'], ind['receipt_sha256']],
                'lifecycle_total': self.total, 'resource_deadline_epoch': self.deadline,
                'production_adoption_allowed': False}

    def verify(self, snapshot):
        if snapshot != self.read(): raise ValueError('catalog qualification snapshot changed')


def qualified_family_step(*, catalog, adoption, qualification, expected_head, store, runner):
    """The same images.families ID; candidate source runs only outside Python host."""
    from .operator_sdk import StepDefinition
    from .models import OperatorSpec
    from .operator_contracts import OperatorContract
    snapshot = catalog.active(adoption.read(), scope=SCOPE, head=expected_head, qualification_authority=qualification)
    ref = ArtifactRef.from_dict(snapshot['implementation'])
    source = store.resolve(ref).read_text()
    def execute(ctx):
        with catalog.lease():
            catalog.active(adoption.read(), scope=SCOPE, head=expected_head, qualification_authority=qualification)
            value = ctx.read('catalog')
            payload = {'rows': value['rows'], 'hamming_distance': ctx.parameters['hamming_distance']}
            result = runner.run(source, payload)
            expected = assign_families(value['rows'], payload['hamming_distance'], check=ctx.check)
            if result['status'] != 'COMPLETED' or not result['isolation_verified'] or result['output'] != {'rows': expected}:
                raise ValueError('qualified family candidate violated its workload contract')
            from dataevolver.dataset.families import duplicate_method
            value['rows'] = result['output']['rows']; value['duplicate_method'] = duplicate_method(payload['hamming_distance'])
            ctx.write_json('candidate-execution.json', result)
            return {'catalog': ctx.write_json('families.json', value)}
    spec = OperatorSpec('images.families', '2', retry_safe=True, source_path=str(Path(__file__)),
        code_revision=payload_sha256([ref.to_dict(), runner.identity(), file_sha256(Path(__file__))]))
    contract = OperatorContract('images.families', '2', spec.sha256(), 'basic',
        {'catalog': {'type': 'object'}}, {'catalog': {'type': 'object'}},
        {'type': 'object', 'additionalProperties': False, 'properties': {'hamming_distance': {'type': 'integer', 'minimum': 0, 'maximum': 64}}, 'required': ['hamming_distance']},
        ('json', 'image'), 'pure_data', 'verify_then_reuse', ('Confined catalog-family scope only; no semantic deduplication or production-code promotion.',))
    return StepDefinition(spec, contract, 'data', execute)
