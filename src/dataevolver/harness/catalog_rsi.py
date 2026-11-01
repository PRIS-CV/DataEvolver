"""Small catalog quality workflow using the existing graph, Controller and RSI."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from dataevolver.adapters.model_contracts import ModelRequest
from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, payload_sha256, file_sha256
from dataevolver.dataset.quality import counts
from .artifact_operator import ArtifactWorkloadOperator
from .models import Intervention, InterventionLevel
from .plan_compiler import CompiledRun


class CatalogRSIDomain:
    def __init__(self, *, configuration, registry, roots, training_feedback=None):
        self.value = copy.deepcopy(configuration)
        if 'training_feedback' in self.value:
            raise ValueError('training feedback must come from the registered adapter')
        self.registry, self.roots = registry, roots
        self.compiled = CompiledRun.from_dict(self.value['compiled_run'])
        self.catalog_ref = ArtifactRef.from_dict(self.value['source'])
        self.training_feedback = training_feedback
        if training_feedback is not None:
            from .training_feedback import ExternalTrainingRunAdapter, TrainingJobSpec
            adapter,job,receipt=training_feedback
            if (not isinstance(adapter,ExternalTrainingRunAdapter) or not isinstance(job,TrainingJobSpec)
                    or not isinstance(receipt,ArtifactRef)):
                raise TypeError('catalog feedback requires the registered training adapter/job/receipt')
            context=self._training_context()
            self.value['training_feedback']={'job':job.to_dict(),'receipt':receipt.to_dict(),
                'stores':{k:str(v) for k,v in adapter.store.roots.items()},'registry_root':str(adapter.root),
                'context':context}

    def _training_context(self):
        adapter,job,receipt=self.training_feedback
        path=adapter._path(job)
        registered=json.loads(path.read_text()) if path.is_file() else {}
        if (registered.get('job')!=job.to_dict() or registered.get('receipt')!=receipt.to_dict()
                or registered.get('job_sha256')!=job.compute_hash()):
            raise ValueError('catalog feedback needs an already collected exact training result')
        return adapter.next_data_plan_context(job,receipt)

    def config(self): return self.value

    def source_families(self):
        catalog = json.loads(ArtifactStore(self.roots).resolve(self.catalog_ref).read_text())
        rows = catalog['rows']
        if any(not r.get('source_dataset') or not r.get('source_family') for r in rows):
            return []
        return sorted({payload_sha256([r['source_dataset'], r['source_family']]) for r in rows})

    def initial_parameters(self, state): return {'ranking': self.value['baseline_ranking'], 'experience_id': ''}

    def asset_id(self, state): return self.catalog_ref.sha256

    def trial_key(self, name, state):
        return copy.deepcopy(self.value['matched_trial_key'])

    def context(self, state):
        result = state.data.get('last_stage_result', {})
        allowed = self.value.get('source_measurements_allowed') is True
        scope = self.value.get('source_measurement_scope')
        if allowed and scope not in {'development','reuse'}:
            raise ValueError('source measurements cannot expose independent or sealed inputs')
        catalog = (json.loads(ArtifactStore(self.roots).resolve(self.catalog_ref).read_text())
                   if allowed and result.get('status') == 'SUCCESS' else None)
        observations = [{'id':row.get('id'),'family_id':row.get('canonical_family_id',row.get('family_id')),
            'source_id':row.get('source_id'),'label_id':row.get('label_id'),
            'clarity_proxy':row.get('quality',{}).get('clarity_proxy')} for row in catalog['rows'][:64]] if catalog else []
        from .capability_requests import CAPABILITY_REQUEST_SCHEMA
        request_only = self.value.get('capability_request_only', False)
        context={'task': 'catalog_sampling_and_family_deduplication',
            'quality_scope': 'controlled_clarity_and_fixed_quota_only',
            'allowed_actions': ['REQUEST_CAPABILITY', 'ABSTAIN'] if self.value.get('capability_request_only') else ['RUN_EXPERIMENT', 'COMPLETE', 'ABSTAIN'],
            'capability_requests_allowed': self.value.get('capability_request_only', False),
            'method_gap': self.value.get('method_gap'),
            'allowed_observations': [], 'allowed_patches': ['capability_gap'] if request_only else ['sampling_recipe'],
            'patch_parameter_schemas': {'capability_gap': CAPABILITY_REQUEST_SCHEMA} if request_only else {
                'sampling_recipe': {'type': 'object', 'additionalProperties': False,
                    'properties': {'ranking': {'enum': self.value['allowed_rankings']}}, 'required': ['ranking']}},
            'action_serialization': {
                'RUN_EXPERIMENT': {'layer': 'R', 'patch_kind': 'sampling_recipe',
                    'parameters_json_example': '{"ranking":"clarity"}'},
                'REQUEST_CAPABILITY': {'layer': 'M', 'patch_kind': 'capability_gap',
                    'parameters_json': 'JSON object matching patch_parameter_schemas.capability_gap'},
                'experience_id': 'Top-level action field only; never add it to patch.parameters_json.',
                'terminal_actions': 'COMPLETE/ABSTAIN have empty patch and observation tools, with parameters_json set to {}.'},
            'available_evidence_ids': result.get('evidence_refs', []),
            'current_metrics': result.get('metrics', {}), 'frozen_quotas': self.value['quotas'],
            'source_observations': {'source_sha256':self.catalog_ref.sha256,'rows':observations,
            'total_rows':len(catalog['rows']) if catalog else None,'truncated':len(catalog['rows'])>64 if catalog else False,
                'status':'AVAILABLE' if catalog else 'NOT_AUTHORIZED_OR_UNVERIFIED',
                'current_ranking':state.data.get('rsi_incumbent_patch',{}).get('ranking',self.value['baseline_ranking']),
                'ranking_definitions':{'hash':'Stable hash of the frozen seed and row ID.',
                    'clarity':'Descending existing measured clarity_proxy; same hash tie break and fixed quotas/families.'},
                'scope':'Source measurements only. They support a bounded comparison, not a prediction of semantic or training benefit.'},
            'permitted_rankings': self.value['allowed_rankings'],
            'compatible_experience_ids': [e['id'] for e in self.value.get('experiences', [])],
            'compatible_experiences': self.value.get('experiences', []),
            'target_quality': self.value['target_quality'],
            'comparison_contract': self.value.get('comparison_contract'),
            'maximum_candidates': self.value.get('maximum_candidates', 1),
            'note': 'Keep source, family relation, quotas, evaluator, models and budget fixed. Quality scores are measured proxies, not training benefit.'}
        if self.value.get('maximum_candidates',1)==0 and not request_only:
            context['allowed_actions']=['COMPLETE','ABSTAIN']
        if self.training_feedback is not None:
            feedback=self._training_context()
            if feedback!=self.value['training_feedback']['context']:
                raise ValueError('training feedback changed after this data run was frozen')
            context['feedback_signals']=feedback['feedback']
            context['training_feedback_scope']={k:feedback[k] for k in
                ('allowed_changes','requires_new_frozen_run','training_benefit_verified')}
            evidence_id='training-feedback:'+self.training_feedback[2].sha256
            context['training_feedback_scope']['evidence_id']=evidence_id
            context['available_evidence_ids']=[*context['available_evidence_ids'],evidence_id]
        return context

    def compile_observation(self, decision, state):
        raise ValueError('catalog profile has no extra observation authority')

    def compile_capability_request(self, decision, state):
        from .capability_requests import validate_capability_request
        if not self.value.get('capability_request_only'):
            raise ValueError('catalog profile does not permit method development requests')
        request = validate_capability_request(json.loads(decision['patch']['parameters_json']))
        return {'request': request, 'decision': decision, 'proposal_actor': 'runtime_agent',
                'production_effects': 0, 'evidence': state.data.get('last_stage_result', {}).get('evidence_refs', []),
                'source_sha256': self.catalog_ref.sha256}

    def compile_intervention(self, decision, state):
        patch = json.loads(decision['patch']['parameters_json'])
        if decision['layer'] != 'R' or set(patch) != {'ranking'} or patch['ranking'] not in self.value['allowed_rankings']:
            raise ValueError('catalog intervention must be a permitted R ranking')
        experience_id = decision['experience_id']
        if experience_id:
            matching = [e for e in self.value.get('experiences', []) if e['id'] == experience_id]
            if len(matching) != 1 or matching[0]['ranking'] != patch['ranking']:
                raise ValueError('experience did not bind the actual ranking')
        delta = {'ranking': patch['ranking'], 'experience_id': experience_id,
                 'source_sha256': self.catalog_ref.sha256, 'proposal_actor': 'runtime_agent',
                 'evidence_ids': decision['evidence_ids']}
        return Intervention(intervention_id='catalog-'+payload_sha256(delta)[:20], level=InterventionLevel.L1,
            failure_code='CATALOG_QUALITY_GAP', operator_id='data.catalog_trial',
            parameters={'ranking': patch['ranking'], 'experience_id': experience_id},
            candidate_delta=delta, expected_effect={'hypothesis': decision['hypothesis']})


class CatalogTrialOperator(ArtifactWorkloadOperator):
    """Every trial runs the compiled graph; optional reviews use one shared Gateway."""
    def __init__(self, *, domain, output_root, gateway=None, review_roles=()):
        self.domain, self.gateway, self.review_roles = domain, gateway, tuple(review_roles)
        super().__init__(operator_id='data.catalog_trial', output_root=output_root,
            configuration={'domain': domain.config(), 'reviews': list(review_roles),
                           'model_contract': gateway.contract_hash() if gateway else None},
            parameter_schema={'type':'object','additionalProperties':False,
                'properties':{'ranking':{'enum':domain.value['allowed_rankings']},'experience_id':{'type':'string'}},
                'required':['ranking','experience_id']})

    def _compiled(self, call):
        if set(call.parameters) != {'ranking', 'experience_id'}:
            raise ValueError('unexpected catalog recipe fields')
        ranking = call.parameters['ranking']
        if ranking not in self.domain.value['allowed_rankings']:
            raise ValueError('ranking is outside frozen profile')
        contract = copy.deepcopy(self.domain.compiled.contract)
        selected = [n for n in contract['graph']['nodes'] if n['operator'] == 'images.select_balanced']
        if len(selected) != 1: raise ValueError('catalog trial requires one registered balanced selector')
        selected[0]['parameters']['ranking'] = ranking
        return CompiledRun(self.domain.compiled.domain, contract, self.domain.compiled.operator_refs,
                           self.domain.compiled.node_calls, self.domain.compiled.policies)

    def source_media(self):
        """Bind nested image bytes and measured features, not only catalog labels."""
        from PIL import Image
        from dataevolver.dataset.families import image_fingerprint
        from dataevolver.dataset.quality import clarity_features
        store = ArtifactStore(self.domain.roots)
        rows = json.loads(store.resolve(self.domain.catalog_ref).read_text())['rows']
        if not any('image' in row for row in rows):
            return []  # A catalog-only CPU fixture has no native media evidence.
        if any('image' not in row for row in rows):
            raise ValueError('catalog trial mixes image and non-image source rows')
        result = []
        for row in rows:
            image = ArtifactRef.from_dict(row['image'])
            with Image.open(store.resolve(image)) as opened:
                opened.load()
                fingerprint = image_fingerprint(opened)
                quality = clarity_features(opened)
            if (any(row.get(k) != v for k, v in fingerprint.items())
                    or any(row.get('quality', {}).get(k) != v for k, v in quality.items())):
                raise ValueError('catalog source image or measured clarity differs from its bytes')
            result.append({'id': row['id'], 'image_sha256': image.sha256, **fingerprint})
        return result

    def workload(self, call):
        source_media = self.source_media()
        compiled = self._compiled(call)
        graph = compiled.bind(self.domain.registry, allowed_write_root=Path(call.work_dir)/'graph', roots=self.domain.roots)
        inner = call.with_updates(operator_id=graph.spec.operator_id, operator_version=graph.spec.version,
            call_id=call.call_id+'-graph', idempotency_key=call.idempotency_key+'-graph',
            work_dir=str(Path(call.work_dir)/'graph/call'), parameters=compiled.contract)
        receipt = graph.execute(inner)
        if graph.verify_effect(inner, receipt).status.value != 'VERIFIED': raise ValueError('catalog graph effect unverified')
        if receipt.status.value != 'SUCCESS':
            return {'compiled_run': compiled.to_dict(), 'inner_call': inner.to_dict(), 'graph_receipt': receipt.to_dict(),
                'ranking': call.parameters['ranking'], 'experience_id': call.parameters['experience_id'],
                'source_sha256': self.domain.catalog_ref.sha256, 'source_families': self.domain.source_families(),
                'source_media': source_media, 'graph_failed': True,
                'graph_failure_code': receipt.stage_result.get('failure_code'),
                'graph_failure_detail': receipt.stage_result.get('failure_detail'), 'training_benefit_verified': False}
        effect = json.loads((Path(inner.work_dir)/'node_runs/export/effect.json').read_text())
        output = ArtifactRef.from_dict(effect['outputs']['catalog'])
        catalog = json.loads(graph.store.resolve(output).read_text())
        rows = catalog['rows']
        measured = sum(r['quality']['clarity_proxy'] for r in rows)/len(rows)
        reviews = []
        for index, role in enumerate(self.review_roles):
            artifacts = tuple(ArtifactRef.from_dict(r['image']) for r in rows[:4] if 'image' in r)
            if not artifacts:
                raise ValueError('a visual catalog review requires actual image artifacts')
            request = ModelRequest('catalog-review-'+payload_sha256([call.to_dict(), role, index])[:24], role,
                ({'role': 'user', 'content': json.dumps({'prompt':
                    'Review the actual selected images. Report pass/fail/uncertain for usable visible detail; do not claim semantic truth or training benefit.\n'
                    + json.dumps({'selected': [r['id'] for r in rows], 'measured_clarity': measured})})},),
                {'type': 'object', 'additionalProperties': False, 'properties': {'status': {'enum': ['pass', 'fail', 'uncertain']}, 'reason': {'type': 'string'}}, 'required': ['status', 'reason']},
                artifacts=artifacts, max_tokens=512, timeout_seconds=self.domain.value.get('model_timeout_seconds', 300))
            result = self.gateway.invoke(request)
            reviews.append({'request': request.to_dict(), 'receipt_hash': result.without_reuse_hash(), 'output': result.output})
        return {'compiled_run': compiled.to_dict(), 'inner_call': inner.to_dict(), 'graph_receipt': receipt.to_dict(),
            'output': output.to_dict(), 'ranking': call.parameters['ranking'], 'experience_id': call.parameters['experience_id'],
            'source_sha256': self.domain.catalog_ref.sha256, 'quality': measured, 'rows': len(rows),
            'source_families': self.domain.source_families(), 'source_media': source_media,
            'unique_families': len({r['family_id'] for r in rows}), 'counts': counts(rows),
            'reviews': reviews, 'training_benefit_verified': False, 'semantic_duplicate_detection': 'not_performed'}

    def validate_report(self, call, report):
        compiled = self._compiled(call)
        if report['compiled_run'] != compiled.to_dict() or report['source_sha256'] != self.domain.catalog_ref.sha256:
            raise ValueError('catalog trial source/recipe changed')
        if report['source_families'] != self.domain.source_families():
            raise ValueError('catalog source-family evidence changed')
        if report['source_media'] != self.source_media():
            raise ValueError('catalog source-media evidence changed')
        graph = compiled.bind(self.domain.registry, allowed_write_root=Path(call.work_dir)/'graph', roots=self.domain.roots)
        from .models import OperatorCall, OperatorReceipt
        inner = OperatorCall.from_dict(report['inner_call'])
        if graph.verify_effect(inner, OperatorReceipt.from_dict(report['graph_receipt'])).status.value != 'VERIFIED':
            raise ValueError('catalog trial lost its graph evidence')
        receipt = OperatorReceipt.from_dict(report['graph_receipt'])
        if report.get('graph_failed'):
            if (receipt.status.value != 'FAILED' or receipt.return_code == 0
                    or report['graph_failure_code'] != receipt.stage_result.get('failure_code')
                    or report['graph_failure_detail'] != receipt.stage_result.get('failure_detail')):
                raise ValueError('catalog graph failure differs from its receipt')
            return
        if receipt.status.value != 'SUCCESS' or receipt.return_code != 0:
            raise ValueError('failed catalog graph cannot become a successful trial')
        rows = json.loads(graph.store.resolve(ArtifactRef.from_dict(report['output'])).read_text())['rows']
        if (report['counts'] != counts(rows) or report['counts'] != self.domain.value['quotas']
                or report['rows'] != len(rows) or report['unique_families'] != len(rows)
                or report['quality'] != sum(r['quality']['clarity_proxy'] for r in rows)/len(rows)):
            raise ValueError('catalog quality/quota/family measurement changed')
        if len(report['reviews']) != len(self.review_roles): raise ValueError('catalog review membership changed')
        for row, role in zip(report['reviews'], self.review_roles):
            request = ModelRequest.from_dict(row['request'])
            result = self.gateway.recover(request)
            if request.role != role or result is None or result.without_reuse_hash() != row['receipt_hash'] or result.output != row['output']:
                raise ValueError('catalog model review lost its native receipt')

    def result(self, call, report):
        value = super().result(call, report)
        if report.get('graph_failed'):
            value.update(status='FAILED', failure_code='COMPILED_DATA_FAILED',
                failure_detail=report['graph_failure_detail'])
            value['metrics'].update(completion_ratio=0., quality=0., coverage=0.)
            value['hard_constraints'].update(quota_preserved=False, family_isolation=False, quality_assessed=False)
            return value
        value['metrics'].update(quality=report['quality'], coverage=1.0)
        value['hard_constraints'].update(quota_preserved=True, family_isolation=True,
            quality_assessed=all(r['output']['status'] == 'pass' for r in report['reviews']))
        value['failure_code'] = 'CATALOG_QUALITY_GAP' if report['quality'] < self.domain.value['target_quality'] else None
        return value

    def failure_signature(self, report):
        return {'code': 'CATALOG_QUALITY_GAP', 'source': report['source_sha256']}
