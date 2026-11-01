"""Bounded benchmark requests over the existing Gateway and expected-ID journal."""
import json

from .contracts import ArtifactRef, payload_sha256
from .generation_evaluation import EvaluationJournal, spatial_answers, spatial_score, soft_tifa
from dataevolver.adapters.model_contracts import ModelRequest


class BenchmarkProtocolIncomplete(ValueError):
    pass


class GenerationBenchmarkEvaluation:
    """Recover the same requests; missing images/answers never change membership."""
    def __init__(self, *, gateway, store, root, manifest):
        self.gateway, self.store = gateway, store
        self.journal = EvaluationJournal(root, manifest)
        self.manifest = manifest
        if manifest['kind'] not in {'spatial', 'geneval2'}:
            raise ValueError('unsupported generation benchmark')
        if manifest.get('split') not in {'development', 'sealed'}:
            raise ValueError('benchmark feedback boundary must be explicit')
        for item in manifest['items']:
            image = ArtifactRef.from_dict(item['image'])
            if image.sha256 != item['image_sha256']:
                raise ValueError('benchmark ID/image digest differs')
            store.resolve(image)
            if manifest['kind'] == 'spatial':
                if len(item['questions']) != 10 or len(item['answers']) != 10:
                    raise ValueError('SpatialGenEval needs all ten original questions/answers')
            elif not item.get('atoms') or any(set(a) != {'question', 'answer'} for a in item['atoms']):
                raise ValueError('GenEval2 requires the complete frozen atom list')

    def _request(self, item, index, parameters, schema, max_tokens):
        kind = self.manifest['kind']
        request = ModelRequest('benchmark-'+payload_sha256([self.journal.binding, item['id'], index])[:28],
            'benchmark.'+kind, ({'role': 'user', 'content': json.dumps(parameters)},), schema,
            artifacts=(ArtifactRef.from_dict(item['image']),), seed=index,
            max_tokens=max_tokens, timeout_seconds=self.manifest.get('timeout_seconds', 1200), max_attempts=1)
        receipt = self.gateway.invoke(request)
        if receipt.backend == 'mock':
            raise ValueError('benchmark acceptance requires a real native model')
        return request, receipt

    @staticmethod
    def _receipt_value(receipt):
        value = receipt.to_dict(); value['reused'] = False
        return value

    def evaluate_item(self, item):
        receipts, requests = [], []
        if self.manifest['kind'] == 'spatial':
            valid = []
            schema = {'type': 'object', 'additionalProperties': False,
                      'required': ['raw_response'], 'properties': {'raw_response': {'type': 'string'}}}
            for attempt in range(25):
                request, receipt = self._request(item, attempt, {'questions': item['questions']}, schema, 2048)
                requests.append(request.to_dict()); receipts.append(self._receipt_value(receipt))
                try:
                    valid.append(spatial_answers(receipt.output['raw_response']))
                except ValueError:
                    continue
                if len(valid) == 5:
                    break
            if len(valid) != 5:
                raise BenchmarkProtocolIncomplete('SpatialGenEval lacks five valid rollouts within 25 attempts; ID remains incomplete')
            result = {'score': spatial_score(valid, item['answers']), 'rollouts': valid,
                      'attempts': len(receipts), 'valid_rollouts': 5}
        else:
            probabilities = []
            schema = {'type': 'object', 'additionalProperties': False,
                'required': ['atom_probability', 'answer_token_ids', 'answer_token_probabilities'],
                'properties': {'atom_probability': {'type': 'number', 'minimum': 0, 'maximum': 1},
                    'answer_token_ids': {'type': 'array', 'minItems': 1, 'items': {'type': 'integer', 'minimum': 0}},
                    'answer_token_probabilities': {'type': 'array', 'minItems': 1,
                        'items': {'type': 'number', 'minimum': 0, 'maximum': 1}}}}
            for index, atom in enumerate(item['atoms']):
                request, receipt = self._request(item, index, atom, schema, 1)
                value = receipt.output
                if (len(value['answer_token_ids']) != len(value['answer_token_probabilities'])
                        or abs(sum(value['answer_token_probabilities'])-value['atom_probability']) > 1e-7):
                    raise ValueError('GenEval2 answer probability evidence changed')
                probabilities.append(value['atom_probability'])
                requests.append(request.to_dict()); receipts.append(self._receipt_value(receipt))
            result = {'atom_probabilities': probabilities, **soft_tifa([probabilities])}
        evidence = {'status': 'VERIFIED', 'requests': requests, 'receipts': receipts,
                    'model_receipt_sha256s': [payload_sha256(r) for r in receipts]}
        return self.journal.record(item['id'], image_path=self.store.resolve(ArtifactRef.from_dict(item['image'])),
            image_sha256=item['image_sha256'], result=result, evidence=evidence)

    def run(self):
        for item in self.manifest['items']:
            self.evaluate_item(item)
        return self.summary()

    def summary(self):
        rows = self.journal.complete()
        for value in rows.values():
            self.verify_item(value)
        if self.manifest['kind'] == 'geneval2':
            result = soft_tifa([rows[item['id']]['result']['atom_probabilities'] for item in self.manifest['items']])
        else:
            result = {key: sum(row['result']['score'][key] for row in rows.values())/len(rows)
                      for key in ('avg_acc', 'basic_acc', 'spatial_acc', 'Position', 'Occlusion')}
        return {'manifest_sha256': self.journal.binding, 'expected_images': len(self.manifest['items']),
                'scored_images': len(rows), 'metrics': result, 'split': self.manifest['split'],
                'training_benefit_verified': False}

    def verify_item(self, value):
        items={item['id']:item for item in self.manifest['items']}
        if (value['id'] not in items or value['binding']!=self.journal.binding
                or value['image_sha256']!=items[value['id']]['image_sha256']):
            raise ValueError('benchmark result membership or image binding changed')
        item=items[value['id']];evidence=value['evidence']
        requests,receipts=evidence['requests'],evidence['receipts'];kind=self.manifest['kind']
        if (evidence['status']!='VERIFIED' or not requests or len(requests)!=len(receipts)
                or evidence['model_receipt_sha256s']!=[payload_sha256(r) for r in receipts]
                or len(requests)>(25 if kind=='spatial' else len(item['atoms']))):
            raise ValueError('benchmark request/receipt evidence changed')
        valid,probabilities=[],[]
        for index,(raw,receipt) in enumerate(zip(requests,receipts)):
            request=ModelRequest.from_dict(raw)
            parameters={'questions':item['questions']} if kind=='spatial' else item['atoms'][index]
            if (request.request_id!='benchmark-'+payload_sha256([self.journal.binding,item['id'],index])[:28]
                    or request.role!='benchmark.'+kind or request.artifacts!=(ArtifactRef.from_dict(item['image']),)
                    or request.seed!=index or request.max_attempts!=1
                    or request.max_tokens!=(2048 if kind=='spatial' else 1)
                    or json.loads(request.messages[0]['content'])!=parameters):
                raise ValueError('benchmark attempt ordering, parameters or image association changed')
            if kind=='spatial':
                if len(valid)==5:raise ValueError('benchmark issued a redundant sixth valid rollout')
                try:valid.append(spatial_answers(receipt['output']['raw_response']))
                except ValueError:pass
            else:
                output=receipt['output']
                if (len(output['answer_token_ids'])!=len(output['answer_token_probabilities'])
                        or abs(sum(output['answer_token_probabilities'])-output['atom_probability'])>1e-7):
                    raise ValueError('benchmark atom probabilities differ from token evidence')
                probabilities.append(output['atom_probability'])
        if kind=='spatial':
            expected={'score':spatial_score(valid,item['answers']),'rollouts':valid,'attempts':len(receipts),'valid_rollouts':5}
        else:
            if len(probabilities)!=len(item['atoms']):raise ValueError('benchmark atom denominator changed')
            expected={'atom_probabilities':probabilities,**soft_tifa([probabilities])}
        if value['result']!=expected:raise ValueError('benchmark score differs from its actual native answer evidence')
