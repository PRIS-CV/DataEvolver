"""A benchmark workload under the existing Controller, with native readback."""
import json

from dataevolver.adapters.model_contracts import ModelRequest, ModelReceipt
from dataevolver.dataset.benchmark_evaluation import GenerationBenchmarkEvaluation, BenchmarkProtocolIncomplete
from dataevolver.dataset.generation_evaluation import spatial_answers
from dataevolver.dataset.contracts import file_sha256
from dataevolver.dataset.contracts import payload_sha256
from dataevolver.adapters.model_gateway import ModelError
from ..artifact_operator import ArtifactWorkloadOperator, CommittedWorkloadAuthority


class GenerationBenchmarkOperator(ArtifactWorkloadOperator):
    def __init__(self, *, gateway, store, manifest, journal_root, output_root):
        self.evaluation = GenerationBenchmarkEvaluation(gateway=gateway, store=store, root=journal_root, manifest=manifest)
        self.gateway = gateway
        super().__init__(operator_id='data.generation_benchmark', output_root=output_root,
            configuration={'manifest': manifest, 'model_contract': gateway.contract_hash(),
                'journal_root': str(self.evaluation.journal.root),
                'stores': {k: str(v) for k, v in store.roots.items()}})
        roots = {str(self.output_root), str(gateway.root), str(self.evaluation.journal.root.resolve())}
        for role in gateway.policy.bindings:
            profile, model, endpoint, _ = gateway._binding(role)
            roots.add(str(gateway._provider(profile, model, endpoint).root.resolve()))
        self.spec = self.spec.with_updates(allowed_write_roots=tuple(sorted(roots)))
        self._declare_contract()

    def _readback(self, rows=None):
        rows = self.evaluation.journal.complete() if rows is None else rows
        inventory = {}
        for row in rows.values():
            self.evaluation.verify_item(row)
            evidence = row['evidence']
            if len(evidence['requests']) != len(evidence['receipts']):
                raise ValueError('benchmark request/receipt cardinality changed')
            for raw, saved in zip(evidence['requests'], evidence['receipts']):
                request, receipt = ModelRequest.from_dict(raw), ModelReceipt.from_dict(saved)
                profile, model, endpoint, profile_hash, digest = self.gateway._prepare(request)
                provider = self.gateway._provider(profile, model, endpoint)
                native = provider.lookup(request, digest)
                if (receipt.backend == 'mock' or receipt.model != model or receipt.profile_hash != profile_hash
                        or receipt.request_hash != digest or receipt.output != native['output']
                        or native['runtime']['input_image'] != request.artifacts[0].to_dict()
                        or any(receipt.usage.get(k) != native['usage'].get(k) for k in ('input_tokens', 'output_tokens', 'images'))):
                    raise ValueError('benchmark native result differs from the confirmed request')
                if self.evaluation.manifest['kind']=='geneval2':
                    import numpy as np
                    from dataevolver.adapters.benchmark_worker import answer_probabilities
                    evidence=native['runtime']['answer_logits'];path=__import__('pathlib').Path(evidence['path']).resolve()
                    if not path.is_relative_to(provider.root.resolve()) or file_sha256(path)!=evidence['sha256']:
                        raise ValueError('native answer logits escaped their bound journal')
                    logits=np.load(path,allow_pickle=False)
                    mass,probabilities,raw_sum=answer_probabilities(logits,receipt.output['answer_token_ids'])
                    if (list(logits.shape)!=evidence['shape'] or mass!=receipt.output['atom_probability']
                            or probabilities!=receipt.output['answer_token_probabilities'] or raw_sum!=evidence['raw_variant_sum']):
                        raise ValueError('answer-token probabilities differ from the actual native logits')
                    inventory[str(path)]=evidence['sha256']
                for path in (provider.path(request), self.gateway.root/(request.request_id+'.receipt.json'),
                             self.gateway.root/(request.request_id+'.intent.json')):
                    inventory[str(path)] = file_sha256(path)
        return rows, inventory

    def _failure_report(self):
        expected=set()
        for item in self.evaluation.manifest['items']:
            maximum=25 if self.evaluation.manifest['kind']=='spatial' else len(item['atoms'])
            expected.update('benchmark-'+payload_sha256([self.evaluation.journal.binding,item['id'],i])[:28] for i in range(maximum))
        attempts=[a for a in self.gateway.audit_usage()['attempts'] if a['request_id'] in expected]
        closed={'FAILED','RETRY_SAFE','INVALID_OUTPUT'}
        native_failed=any(a['state'] in closed for a in attempts)
        protocol_failed=False
        if self.evaluation.manifest['kind']=='spatial':
            for item in self.evaluation.manifest['items']:
                ids={'benchmark-'+payload_sha256([self.evaluation.journal.binding,item['id'],i])[:28] for i in range(25)}
                completed=[a for a in attempts if a['request_id'] in ids and a['state']=='COMPLETED']
                if len(completed)==25:
                    valid=0
                    for row in completed:
                        receipt=json.loads((self.gateway.root/(row['request_id']+'.receipt.json')).read_text())['receipt']
                        try:spatial_answers(receipt['output']['raw_response']);valid+=1
                        except ValueError:pass
                    protocol_failed=protocol_failed or valid<5
        if not (native_failed or protocol_failed) or any(a['state'] not in closed|{'COMPLETED'} for a in attempts):
            raise ModelError('AMBIGUOUS_EFFECT','benchmark failure is not fully reconciled; no terminal failure or resend')
        inventory={}
        for row in attempts:
            request_id=row['request_id']
            for path in self.gateway.root.glob(request_id+'.*.json'):inventory[str(path)]=file_sha256(path)
            profile,model,endpoint,_=self.gateway._binding(row['role'])
            provider=self.gateway._provider(profile,model,endpoint)
            for path in provider.root.glob(request_id+'.*.json'):inventory[str(path)]=file_sha256(path)
        partial={}
        for path in self.evaluation.journal.root.glob('*.result.json'):
            value=json.loads(path.read_text())
            if value['binding']!=self.evaluation.journal.binding:raise ValueError('partial benchmark manifest changed')
            partial[value['id']]=value
        self._readback(partial)
        return {'status':'FAILED','failure_reason':'native_request_failed' if native_failed else 'rollout_budget_exhausted',
            'manifest':self.evaluation.manifest,'attempts':attempts,
            'expected_images':len(self.evaluation.manifest['items']),'scored_images':len(partial),
            'results':partial,'native_inventory':inventory,'training_benefit_verified':False}

    def workload(self, call):
        try:
            summary = self.evaluation.run()
        except (ModelError,BenchmarkProtocolIncomplete):
            return self._failure_report()
        rows, inventory = self._readback()
        return {'status':'COMPLETE','summary': summary, 'manifest': self.evaluation.manifest,
                'results': rows, 'native_inventory': inventory,
                'training_benefit_verified': False}

    def validate_report(self, call, report):
        if report.get('status')=='FAILED':
            if report!=self._failure_report():raise ValueError('benchmark failure accounting changed')
            return
        rows, inventory = self._readback()
        if (report['manifest'] != self.evaluation.manifest or report['results'] != rows
                or report['native_inventory'] != inventory or report['training_benefit_verified'] is not False
                or report['summary'] != self.evaluation.summary()):
            raise ValueError('benchmark workload binding changed')

    def result(self,call,report):
        result=super().result(call,report)
        if report['status']=='FAILED':
            result.update(status='FAILED',failure_code='BENCHMARK_NATIVE_FAILED')
            result['metrics']['completion_ratio']=0.
        return result


class CommittedBenchmarkOutcomeAuthority(CommittedWorkloadAuthority):
    def __init__(self, controller, operator):
        if not isinstance(operator, GenerationBenchmarkOperator):
            raise TypeError('selection outcomes require the registered real benchmark workload')
        super().__init__(controller, operator)

    def read(self):
        value = super().read()
        manifest = value['report']['manifest']
        if manifest.get('purpose') != 'selection_qualification' or not manifest.get('qualification_plan_id'):
            raise ValueError('development integration or final sealed benchmarks cannot qualify a selector')
        return value
