"""Conditional catalog experience using CapabilityStore and real trial evidence."""
import json
from pathlib import Path

from dataevolver.adapters.model_contracts import ModelRequest
from dataevolver.dataset.contracts import file_sha256, payload_sha256
from .artifact_operator import ArtifactWorkloadOperator
from .models import (Capability, CapabilityStatus, FailureSignature, Intervention, InterventionLevel,
                     OperatorCall, OperatorReceipt)


def read_trials(controller, operator, *, include_receipts=False):
    state = controller.status()
    if (state.state.value not in {'COMPLETE', 'ABSTAIN'} or state.data.get('pending_call')
            or not controller._frozen()[0] or controller.replay().to_dict() != state.to_dict()):
        raise ValueError('catalog experience needs a terminal frozen replayable source')
    events = controller.store.events.read()
    result = []
    for event in events:
        if event.event_type != 'ACTION_COMMITTED' or event.payload['receipt']['operator_id'] != operator.spec.operator_id:
            continue
        receipt = OperatorReceipt.from_dict(event.payload['receipt'])
        planned = [e for e in events if e.event_type == 'ACTION_PLANNED' and e.payload.get('call_id') == receipt.call_id]
        if len(planned) != 1: raise ValueError('catalog trial lost its intent')
        call = OperatorCall.from_dict(planned[0].payload['operator_call'])
        check = operator.verify_effect(call, receipt)
        if check.status.value != 'VERIFIED' or check.artifact_hashes != event.payload['effect_verification']['artifact_hashes']:
            raise ValueError('catalog experience trial effect changed')
        report = json.loads(operator.path(call).read_text())['report']
        result.append({'report': report, 'path': str(operator.path(call)), 'sha256': file_sha256(operator.path(call)),
                       'campaign_id': controller.goal.campaign_id, 'frozen_sha256': file_sha256(controller.layout.frozen_contract_path)})
        if include_receipts:
            result[-1].update(call=call.to_dict(), receipt=receipt.to_dict())
    if not result: raise ValueError('catalog experience has no executed source trial')
    return result


class CatalogKnowledgeOperator(ArtifactWorkloadOperator):
    def __init__(self, *, phase, output_root, capability_store, scope, source_controller, source_operator,
                 gateway=None, role='build.distill', validation=None, candidate=None):
        if phase not in {'distill', 'promote'}: raise ValueError('unknown experience stage')
        self.phase, self.capabilities, self.scope = phase, capability_store, dict(scope)
        self.source_controller, self.source_operator = source_controller, source_operator
        self.gateway, self.role, self.validation, self.candidate = gateway, role, validation, candidate
        source = read_trials(source_controller, source_operator)
        if not any(h['decision']['decision'] == 'ADOPT_CANDIDATE' for h in source_controller.status().data.get('rsi_history', [])):
            raise ValueError('experience requires actual Controller adoption')
        self.source = source
        super().__init__(operator_id='evolution.catalog_knowledge_'+phase, output_root=output_root,
            configuration={'phase': phase, 'scope': scope, 'source': source,
                'model': gateway.contract_hash() if gateway else None, 'candidate': candidate.to_dict() if candidate else None,
                'validation_origins': [c.goal.campaign_id for c, op in validation] if validation else []})

    def _validation(self):
        if not self.validation or not self.candidate: raise ValueError('promotion needs independent comparison and candidate')
        baseline, improved = [read_trials(c, op)[-1] for c, op in self.validation]
        a, b = baseline['report'], improved['report']
        sources = {s['report']['source_sha256'] for s in self.source}
        families = {f for s in self.source for f in s['report'].get('source_families', [])}
        source_media = [m for s in self.source for m in s['report'].get('source_media', [])]
        validation_media = b.get('source_media', [])
        source_pixels = {m['pixel_sha256'] for m in source_media}
        source_hashes = {m['image_sha256'] for m in source_media}
        source_dhashes = {m['dhash'] for m in source_media}
        if (a['source_sha256'] != b['source_sha256'] or b['source_sha256'] in sources
                or not families or not b.get('source_families')
                or a.get('source_families') != b['source_families']
                or families.intersection(b['source_families'])
                or a['counts'] != b['counts'] or a['rows'] != b['rows']
                or b['unique_families'] != b['rows'] or b['quality'] <= a['quality']
                or b['ranking'] != self.candidate.intervention.parameters['ranking']
                or a['ranking'] == b['ranking'] or any(r['output']['status'] != 'pass' for r in b['reviews'])
                or not source_media or not validation_media or a.get('source_media') != validation_media
                or any(m['pixel_sha256'] in source_pixels or m['image_sha256'] in source_hashes
                       or m['dhash'] in source_dhashes for m in validation_media)):
            raise ValueError('experience failed independent source/quality/protection validation')
        return [baseline, improved]

    def workload(self, call):
        if self.phase == 'distill':
            a, b = self.source[0]['report'], self.source[-1]['report']
            request = ModelRequest('distill-'+payload_sha256([call.to_dict(), self.scope])[:24], self.role,
                ({'role': 'system', 'content': 'Distil only the actual successful sampling ranking into a conditional reusable policy. No source IDs, labels, coordinates or answers. Report the exact successful ranking.'},
                 {'role': 'user', 'content': json.dumps({'scope': self.scope, 'before': a['ranking'], 'before_quality': a['quality'], 'after': b['ranking'], 'after_quality': b['quality']})}),
                {'type': 'object', 'additionalProperties': False, 'properties': {'ranking': {'enum': ['hash', 'clarity']}, 'applicability': {'type': 'string'}, 'precaution': {'type': 'string'}}, 'required': ['ranking', 'applicability', 'precaution']},
                max_tokens=512, timeout_seconds=300)
            receipt = self.gateway.invoke(request)
            if receipt.output['ranking'] != b['ranking'] or b['quality'] <= a['quality']:
                raise ValueError('distilled policy differs from actual improvement')
            policy = receipt.output
            capability = Capability('catalog-k-'+payload_sha256([self.scope, b['ranking']])[:20], '1', CapabilityStatus.CANDIDATE,
                FailureSignature('CATALOG_QUALITY_GAP'),
                Intervention('catalog-experience', 'CATALOG_QUALITY_GAP', 'data.catalog_trial', InterventionLevel.L1,
                    parameters={'ranking': b['ranking'], 'policy': policy, 'scope': self.scope}),
                validation_evidence=tuple(s['path'] for s in self.source), context_signature='catalog_quality',
                compatibility_hash=payload_sha256(self.scope), source_asset_id=a['source_sha256'])
            trace = {'request': request.to_dict(), 'receipt_hash': receipt.without_reuse_hash()}
            proof = []
        else:
            proof = self._validation()
            capability = self.candidate.with_updates(status=CapabilityStatus.PROMOTED,
                validation_evidence=(*self.candidate.validation_evidence, *(s['path'] for s in proof)))
            trace = None
        self.capabilities.append_candidate(capability)
        return {'capability': capability.to_dict(), 'scope': self.scope, 'source': self.source,
                'independent_validation': proof, 'model': trace, 'training_benefit_verified': False}

    def validate_report(self, call, report):
        if report['scope'] != self.scope or report['source'] != read_trials(self.source_controller, self.source_operator):
            raise ValueError('experience source or scope changed')
        capability = Capability.from_dict(report['capability'])
        if capability not in self.capabilities.list_capabilities(): raise ValueError('experience absent from CapabilityStore')
        if self.phase == 'promote':
            if report['independent_validation'] != self._validation() or capability.status is not CapabilityStatus.PROMOTED:
                raise ValueError('experience independent qualification changed')
        else:
            trace = report['model']; receipt = self.gateway.recover(ModelRequest.from_dict(trace['request']))
            if receipt is None or receipt.without_reuse_hash() != trace['receipt_hash']:
                raise ValueError('experience distillation native receipt changed')


def compatible_catalog_experiences(store, *, scope, qualification=None):
    from .artifact_operator import CommittedWorkloadAuthority
    qualified = None
    if qualification is not None:
        if not isinstance(qualification, CommittedWorkloadAuthority):
            raise TypeError('catalog experience requires committed qualification evidence')
        report = qualification.read()['report']
        if report['scope'] != scope or not report['independent_validation']:
            raise ValueError('catalog experience qualification scope changed')
        qualified = Capability.from_dict(report['capability'])
    accepted, excluded = [], []
    for capability in store.list_capabilities():
        if (qualified == capability and capability.status is CapabilityStatus.PROMOTED and capability.compatibility_hash == payload_sha256(scope)
                and capability.intervention.parameters.get('scope') == scope and capability.validation_evidence
                and all(Path(p).is_file() for p in capability.validation_evidence)):
            accepted.append({'id': capability.capability_id, 'ranking': capability.intervention.parameters['ranking'],
                             'version': capability.version, 'scope': scope})
        else: excluded.append({'id': capability.capability_id, 'reason': 'unqualified_or_incompatible'})
    return {'records': accepted, 'excluded': excluded, 'empty_reason': None if accepted else 'NO_ENTRY'}
