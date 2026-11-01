"""One runtime-chosen native segmentation and a separately recorded visual probe.

Masks are proposals. They cannot rewrite immutable masks, become edit support,
or reactivate a candidate. Controller/Gateway own dispatch and no-resend recovery.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from dataevolver.adapters.model_contracts import ModelRequest
from dataevolver.dataset.training_pipeline import MASK_SCHEMA
from .. import boundary_evidence as boundary
from ..models import (OperatorSpec, ExpectedOutputSpec, OperatorReceipt, StageStatus,
    EffectVerification, EvidenceStatus, RecoveryResult, RecoveryStatus)
from ..operator_contracts import OperatorContract
from ..operators import _load_receipt, _receipt_path, _write_receipt
from ..state_store import CampaignLease, atomic_write_json


class BoundaryEvidenceProbeOperator:
    def __init__(self, *, authority, segment_gateway, review_backend, resource_operator, output_root):
        from ..training_candidate_feedback import FrozenTrainingCandidateFeedbackAuthority
        from .training_supervision import TrainingSupervisionOperator
        if type(authority) is not FrozenTrainingCandidateFeedbackAuthority:
            raise ValueError('boundary probes require registered original training evidence')
        if (type(resource_operator) is not TrainingSupervisionOperator or resource_operator.custom_wait is not None
                or getattr(resource_operator._wait, '__func__', None) is not TrainingSupervisionOperator._wait):
            raise ValueError('boundary probes require the original registered resource admission')
        if (segment_gateway.policy.bindings != {'training.segment': 'segmentation'}
                or segment_gateway.policy.max_calls != 1 or segment_gateway.policy.usage_mode != 'bounded'):
            raise ValueError('boundary probe needs a single-call SAM3 Gateway')
        self.authority = authority; self.origin = authority.identity(); self.inputs = authority.probe_inputs()
        self.segment_gateway, self.backend, self.resource_operator = segment_gateway, review_backend, resource_operator
        from dataevolver.adapters.training_media_worker import TrainingMediaProvider
        from dataevolver.adapters.codex_planner import RecoveringPlannerBackend
        provider = segment_gateway.factories['segmentation']()
        if (type(provider) is not TrainingMediaProvider
                or provider.config != resource_operator.contract['models']['workers']['segmentation']
                or type(review_backend) is not RecoveringPlannerBackend
                or review_backend.gateway.identity('agent.runtime_rsi')['model'] != 'gpt-6.1-sol'):
            raise ValueError('boundary probe providers differ from the registered resident/Planner models')
        self.segment_contract_hash = segment_gateway.contract_hash()
        self.review_contract_hash = review_backend.gateway.contract_hash()
        self.resource_spec_hash = resource_operator.spec.sha256()
        self.output_root = Path(output_root).resolve()
        self.dataset_root = segment_gateway.store.roots[provider.dataset_store]
        self.native_roots = (provider.root.resolve(), segment_gateway.root, review_backend.gateway.root)
        if (any(not root.is_relative_to(self.output_root) for root in self.native_roots)
                or self.dataset_root.is_relative_to(self.output_root)
                or self.output_root.is_relative_to(self.dataset_root)):
            raise ValueError('boundary native journals and writable dataset need disjoint approved roots')
        if review_backend.gateway.policy.max_calls != 2:
            raise ValueError('runtime decision and visual review require an explicit two-call bound')
        _, proof = authority.read()
        protected = [Path(self.origin['config']), *[Path(p) for p in self.inputs['store_roots'].values()]]
        protected.extend(Path(p) for p in proof['artifact_hashes'])
        if any(root.is_relative_to(p) or p.is_relative_to(root)
               for root in (self.output_root, self.dataset_root) for p in protected):
            raise ValueError('boundary probe outputs overlap original evidence')
        if segment_gateway.identity('training.segment')['model'] != 'SAM3':
            raise ValueError('boundary probe cannot substitute the selected segmentation model')
        self.spec = OperatorSpec('data.probe_instance_boundaries', '1', retry_safe=False,
            source_path=str(Path(__file__).resolve()), code_revision=payload_sha256({
                'adapter': file_sha256(Path(__file__)), 'contracts': file_sha256(Path(boundary.__file__)),
                'authority': self.origin, 'inputs': payload_sha256(self.inputs),
                'segmentation': segment_gateway.contract_hash(), 'review': review_backend.gateway.contract_hash(),
                'native_roots': [str(p) for p in self.native_roots],
                'resources': resource_operator.spec.sha256()}),
            allowed_write_roots=(str(self.output_root), str(self.dataset_root)),
            expected_output_specs=(ExpectedOutputSpec('boundary-probe.json'),))
        self.operator_contract = OperatorContract(self.spec.operator_id, '1', self.spec.sha256(), 'basic',
            {'registered_instance_evidence': {'type': 'object'}}, {'boundary_proposals': {'type': 'object'}},
            boundary.QUERY_SCHEMA, ('image', 'json'), 'model', 'verify_then_reuse', (
                'The runtime chooses subject IDs, text and boxes; original correspondence and thresholds stay fixed.',
                'Native masks and visual assessments are unqualified proposals, never edit or effect authority.',
                'No candidate, immutable mask, quality gate or accepted dataset record is changed.',
                'Interrupted stages reconcile exact existing receipts; an unknown effect is never resubmitted.'))

    def _path(self, call):
        if (call.operator_id, call.operator_version) != (self.spec.operator_id, '1'):
            raise ValueError('boundary probe operator identity changed')
        self.operator_contract.validate_parameters(call.parameters)
        path = Path(call.work_dir)/'boundary-probe.json'
        if not path.resolve().is_relative_to(self.output_root) or path.parent.resolve() == self.output_root:
            raise ValueError('boundary probe escaped its write root')
        if any(p.is_symlink() for p in (path, path.parent, *path.parent.parents, path.parent/'probe-intent.json',
                                        _receipt_path(call))):
            raise ValueError('boundary probe paths cannot contain symlinks')
        return path

    def _binding(self, call):
        if self.authority.identity() != self.origin: raise ValueError('boundary probe origin changed')
        from .training_supervision import TrainingSupervisionOperator
        if (self.segment_gateway.contract_hash() != self.segment_contract_hash
                or self.backend.gateway.contract_hash() != self.review_contract_hash
                or self.resource_operator.spec.sha256() != self.resource_spec_hash
                or self.resource_operator.custom_wait is not None
                or getattr(self.resource_operator._wait, '__func__', None) is not TrainingSupervisionOperator._wait):
            raise ValueError('frozen probe model/resource admission changed')
        return {'call_sha256': payload_sha256(call.to_dict()), 'operator_spec_sha256': self.spec.sha256(),
                'origin_sha256': payload_sha256(self.origin)}

    def _segment_request(self, call):
        parameters = boundary.compile_boundary_query(call.parameters, self.inputs)
        recipe = self.inputs['contract']['production']['recipe']
        return ModelRequest('boundary-'+payload_sha256(self._binding(call))[:28], 'training.segment',
            ({'role': 'user', 'content': json.dumps(parameters, sort_keys=True, separators=(',', ':'))},),
            MASK_SCHEMA, artifacts=(ArtifactRef.from_dict(self.inputs['trial']['source']),),
            max_tokens=1, timeout_seconds=min(900, recipe['model_timeout_seconds']), max_attempts=1,
            seed=self.inputs['original_segment_record']['request']['seed'])

    def _review_request(self, call, segmented):
        trial = self.inputs['trial']; selected = {x['subject_id'] for x in segmented.output['instances']}
        media = [*trial['media'][:3], *[x for x in trial['media'][3:] if x['label'] in {
            *('reference '+sid for sid in selected), *('instance mask '+sid for sid in selected)}],
            *[{'label': 'native boundary proposal '+x['subject_id'], 'artifact': x['mask']}
              for x in segmented.output['instances']]]
        self.backend.set_artifacts([ArtifactRef.from_dict(x['artifact']) for x in media])
        prompt = {'purpose': 'Independent runtime visual assessment of actual native boundary proposals',
            'question': call.parameters['question'], 'media_order': [x['label'] for x in media],
            'immutable_correspondence': [{'subject_id': x['subject_id'], 'target_instance_id': x['target_instance_id']}
                                         for x in segmented.output['instances']],
            'rules': 'Inspect actual source/reference, immutable mask and native proposal separately. '
                     'Assess identity, fine boundary/foliage coverage and preservation of visible holes or '
                     'background gaps. Use uncertain whenever display or evidence cannot establish this. '
                     'The prior immutable mask is not ground truth for newly proposed pixels. Do not infer '
                     'effect ownership from mask differences. Do not invent new geometry or repair instructions. '
                     'This observation grants no edit authority, qualification, adoption or quality acceptance.'}
        messages = [{'role': 'user', 'content': json.dumps(prompt, ensure_ascii=False, sort_keys=True)}]
        binding = payload_sha256({'messages': messages, 'schema': boundary.REVIEW_SCHEMA,
                                  'artifacts': [x.to_dict() for x in self.backend.artifacts]})
        request = ModelRequest('rsi-'+binding[:28], 'agent.runtime_rsi', tuple(messages), boundary.REVIEW_SCHEMA,
            artifacts=self.backend.artifacts, max_tokens=4096, timeout_seconds=900, max_attempts=1, seed=0)
        return request, media

    @staticmethod
    def _existing(gateway, request):
        # Verification must not materialize a missing receipt or invoke a provider.
        if not (gateway.root/(request.request_id+'.receipt.json')).is_file(): return None
        return gateway._recover(request, gateway._prepare(request))

    def _finish(self, call, *, submit):
        path = self._path(call); binding = self._binding(call)
        intent = json.loads((path.parent/'probe-intent.json').read_text())
        if intent['binding'] != binding: raise ValueError('boundary probe intent changed')
        request = self._segment_request(call)
        receipt = self.segment_gateway.recover(request) if submit else self._existing(self.segment_gateway, request)
        if receipt is None:
            if not submit: raise ValueError('boundary segmentation has no committed receipt; no resend')
            self.resource_operator._wait(call, 'training.segment')
            receipt = self.segment_gateway.invoke(request)
        # The native provider verifies exact response, masks and decode bytes.
        provider = self.segment_gateway.factories['segmentation']()
        if provider.lookup(request, receipt.request_hash)['output'] != receipt.output:
            raise ValueError('native boundary proposal differs from Gateway receipt')
        diagnostics = boundary.boundary_mask_diagnostics(self.inputs, call.parameters, receipt.output,
                                                        self.segment_gateway.store)
        review_request, media = self._review_request(call, receipt)
        visual = (self.backend.gateway.recover(review_request) if submit
                  else self._existing(self.backend.gateway, review_request))
        if visual is None:
            if not submit: raise ValueError('visual boundary assessment has no committed receipt; no resend')
            # Use the same authenticated ModelGateway backend as the Planner.
            self.backend.complete_json(list(review_request.messages), schema=boundary.REVIEW_SCHEMA)
            visual = self._existing(self.backend.gateway, review_request)
        if visual is None: raise ValueError('visual boundary assessment did not commit')
        expected = {x['subject_id']: x['target_instance_id'] for x in receipt.output['instances']}
        assessments = visual.output['assessments']
        if len(assessments) != len(expected) or {x['subject_id']: x['target_instance_id'] for x in assessments} != expected:
            raise ValueError('runtime visual assessment changed selected-instance correspondence')
        native = self.backend.gateway.root.parent/'native'/review_request.request_id
        native_value = json.loads((native/'native.json').read_text())
        if (native_value['output_sha256'] != file_sha256(native/'final-response.json')
                or native_value['events_sha256'] != file_sha256(native/'events.jsonl')
                or native_value['request_hash'] != visual.request_hash
                or native_value['response']['output'] != visual.output):
            raise ValueError('visual boundary native output/receipt changed')
        files = {str(p): file_sha256(p) for p in provider.root.rglob('*') if p.is_file()}
        files.update({str(p): file_sha256(p) for p in native.rglob('*') if p.is_file()})
        for ref in [ArtifactRef.from_dict(x['mask']) for x in receipt.output['instances']]:
            asset = self.segment_gateway.store.resolve(ref); files[str(asset)] = file_sha256(asset)
        return {'schema_version': 'dataevolver.boundary_evidence_probe.v1', 'binding': binding, 'at': intent['at'],
            'source': self.inputs['trial']['source'], 'candidate': self.inputs['trial']['candidate'],
            'runtime_query': call.parameters, 'segmentation': {'request': request.to_dict(), 'receipt': receipt.to_dict(),
                'receipt_hash': receipt.without_reuse_hash()}, 'mask_diagnostics': diagnostics,
            'visual_review': {'request': review_request.to_dict(), 'receipt': visual.to_dict(),
                'receipt_hash': visual.without_reuse_hash()}, 'media_order': media, 'files': files,
            'proposal_status': 'OBSERVED_UNQUALIFIED', 'boundary_witness_qualification': 'NOT_ASSESSED',
            'effect_attribution': 'NOT_ASSESSED', 'image_submission_allowed': False,
            'candidate_adoption_allowed': False, 'new_accepted_records': 0,
            'immutable_masks_modified': False, 'generation_model_calls': 0, 'new_permissions': False}

    def verify_effect(self, call, receipt):
        try:
            path = self._path(call)
            if not path.exists(): return EffectVerification(EvidenceStatus.UNOBSERVED, 'boundary probe report absent')
            value = json.loads(path.read_text()); expected = self._finish(call, submit=False)
            # A Gateway readback may report reused=True; it does not change the
            # recorded output or semantic receipt identity.
            for name in ('segmentation', 'visual_review'):
                expected[name]['receipt']['reused'] = value[name]['receipt']['reused']
            if value != expected: raise ValueError('boundary proposal or actual visual evidence changed')
            if receipt is not None and (receipt.effect_hash != file_sha256(path)
                    or receipt.call_id != call.call_id or receipt.stage_result != self._receipt(call, path, False).stage_result):
                raise ValueError('boundary probe Controller receipt changed')
            return EffectVerification(EvidenceStatus.VERIFIED, 'actual SAM and visual probe verified; no edit authority',
                artifact_hashes={str(path): file_sha256(path), **value['files']})
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            return EffectVerification(EvidenceStatus.CONTRADICTED, str(exc))

    def _receipt(self, call, path, reused):
        value = json.loads(path.read_text())
        return OperatorReceipt(call.call_id, self.spec.operator_id, '1', StageStatus.SUCCESS, 0,
            value['at'], value['at'], reused=reused, effect_hash=file_sha256(path),
            output_artifacts=({'path': str(path), 'sha256': file_sha256(path)},),
            stage_result={'operator_id': self.spec.operator_id, 'operator_version': '1', 'status': 'SUCCESS',
                'metrics': {'boundary_probe_captured': 1.0, 'production_qualification_passed': 0.0},
                'hard_constraints': {'artifact_valid': True}, 'evidence_refs': [str(path)]})

    def recover(self, call):
        path = self._path(call); prior = _load_receipt(_receipt_path(call))
        verified = self.verify_effect(call, prior)
        if verified.status is EvidenceStatus.VERIFIED:
            return RecoveryResult(RecoveryStatus.RECOVERED, _write_receipt(_receipt_path(call), self._receipt(call, path, True)))
        if verified.status is EvidenceStatus.UNOBSERVED and prior is None:
            if not (path.parent/'probe-intent.json').exists(): return RecoveryResult(RecoveryStatus.RETRY_SAFE, reason='no probe intent')
            try:
                value = self._finish(call, submit=False); atomic_write_json(path, value)
                return RecoveryResult(RecoveryStatus.RECOVERED, _write_receipt(_receipt_path(call), self._receipt(call, path, True)))
            except (OSError, ValueError, KeyError, TypeError, RuntimeError): pass
        return RecoveryResult(RecoveryStatus.AMBIGUOUS, reason='incomplete or contradicted probe; inspect native, never resend')

    def execute(self, call):
        path = self._path(call)
        self.output_root.mkdir(parents=True, exist_ok=True)
        with CampaignLease(self.output_root/'.boundary-probe.lease', ttl_seconds=2400):
            recovered = self.recover(call)
            if recovered.status is RecoveryStatus.RECOVERED: return recovered.receipt
            if recovered.status is not RecoveryStatus.RETRY_SAFE: raise ValueError(recovered.reason)
            journal = self.output_root/'single-boundary-probe.json'; binding = self._binding(call)
            if journal.exists() and json.loads(journal.read_text()) != binding:
                raise ValueError('a new call cannot reset the single probe allowance')
            atomic_write_json(journal, binding)
            atomic_write_json(path.parent/'probe-intent.json', {'binding': binding, 'at': datetime.now(timezone.utc).isoformat()})
            atomic_write_json(path, self._finish(call, submit=True))
            return _write_receipt(_receipt_path(call), self._receipt(call, path, False))
