"""Registered import of independent provider receipts into the frozen qualifier."""
from dataclasses import replace
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from .. import evidence_acquisition as acquisition
from ..boundary_witness_readback import FrozenBoundaryWitnessFeedbackAuthority
from ..production_feedback_request import FrozenProductionFeedbackRequestAuthority
from ..models import ExpectedOutputSpec, EvidenceStatus
from ..operator_contracts import OperatorContract
from .boundary_qualification_feedback import BoundaryQualificationFeedbackOperator


class IndependentEvidenceAcquisitionOperator(BoundaryQualificationFeedbackOperator):
    def __init__(self, *, authority, witness_authority, request_authority, provider, output_root):
        if (type(witness_authority) is not FrozenBoundaryWitnessFeedbackAuthority
                or type(request_authority) is not FrozenProductionFeedbackRequestAuthority
                or type(provider) is not acquisition.RegisteredIndependentEvidenceProvider):
            raise ValueError('evidence acquisition needs the original registered witness, request and provider')
        super().__init__(authority=authority, output_root=output_root)
        self.witness_authority = witness_authority; self.witness_binding = witness_authority.identity()
        self.witness_summary, self.witness_proof = witness_authority.read()
        self.request_authority = request_authority; self.request_binding = request_authority.identity()
        self.request = request_authority.read(); self.provider = provider; self.provider_binding = provider.identity()
        if (payload_sha256(self.witness_summary['witness']) != self.summary['qualification']['witness_sha256']
                or self.request['feedback_sha256'] != self.summary_hash
                or self.request['request']['request']['implementation_kind'] != 'registered_operator'
                or any(self.request.get(k) is not True for k in (
                    'frozen', 'replay', 'registered_effect_verified', 'native_and_receipt_verified'))):
            raise ValueError('evidence acquisition does not match the actual runtime request and prior qualification')
        self.acquired = provider.acquire(self.witness_summary['witness'])
        protected = [Path(self.request_binding['config']), witness_authority.config,
            *(Path(p) for p in self.request['artifact_hashes']),
            *(Path(p) for p in self.acquired['artifact_hashes'])]
        if any(p.resolve().is_relative_to(self.output_root) or self.output_root.is_relative_to(p.resolve())
               for p in protected):
            raise ValueError('evidence acquisition overlaps original evidence')
        self.spec = replace(self.spec, operator_id='evolution.acquire_independent_evidence',
            source_path=str(Path(__file__).resolve()), code_revision=payload_sha256({
                'feedback': self.spec.code_revision, 'adapter': file_sha256(Path(__file__)),
                'port': file_sha256(Path(acquisition.__file__)), 'witness': self.witness_binding,
                'request': self.request_binding, 'provider': self.provider_binding,
                'acquired': payload_sha256(self.acquired)}),
            expected_output_specs=(ExpectedOutputSpec('evidence_acquisition.json'),))
        self.operator_contract = OperatorContract(self.spec.operator_id, '1', self.spec.sha256(), 'evaluation',
            {'registered_witness': {'type': 'object'}, 'prior_qualification': {'type': 'object'},
             'provider_provenance': acquisition.PROVIDER_SCHEMA}, {'acquisition': {'type': 'object'}},
            {'type': 'object', 'additionalProperties': False, 'properties': {}}, ('json',),
            'pure_data', 'verify_then_reuse', (
                'Import only completed evidence from the existing frozen production Controller authorities.',
                'Native-audit provenance binds provider trust, independent assessment, claim scope and actual receipts.',
                'Missing providers, authority, scope or independent conflict resolution remain unmet prerequisites.',
                'Use the original qualifier; no inference, probes, task pixel answers, submission or adoption.'))

    def _validate(self, call):
        path = super()._validate(call).with_name('evidence_acquisition.json')
        if path.is_symlink():
            raise ValueError('evidence acquisition cannot be a symlink')
        return path

    def _result(self):
        super()._result()
        if (self.witness_authority.identity() != self.witness_binding
                or self.witness_authority.read() != (self.witness_summary, self.witness_proof)
                or self.request_authority.identity() != self.request_binding
                or self.request_authority.read() != self.request
                or self.provider.identity() != self.provider_binding
                or self.provider.acquire(self.witness_summary['witness']) != self.acquired):
            raise ValueError('evidence acquisition original request, witness or provider changed')
        return {'schema_version': 'dataevolver.registered_independent_evidence_acquisition.v1',
            'acquisition': self.acquired, 'qualification': self.acquired['qualification'],
            'request_sha256': self.request['request_sha256'], 'feedback_sha256': self.summary_hash,
            'new_model_calls': 0, 'new_probe_executions': 0, 'new_accepted_records': 0,
            'image_submission_allowed': False, 'candidate_adoption_allowed': False,
            'production_qualification_passed': False}

    def _receipt(self, call, path, reused):
        receipt = super()._receipt(call, path, reused)
        return replace(receipt, stage_result={**receipt.stage_result, 'metrics': {
            'independent_evidence_acquisition_captured': 1.0, 'production_qualification_passed': 0.0}})

    def verify_effect(self, call, receipt):
        result = super().verify_effect(call, receipt)
        if result.status is EvidenceStatus.VERIFIED:
            result = replace(result, artifact_hashes={**self.authority.read()[1]['artifact_hashes'],
                **self.witness_proof['artifact_hashes'],
                **self.request['artifact_hashes'], **self.acquired['artifact_hashes'], **result.artifact_hashes})
        return result
