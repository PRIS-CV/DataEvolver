"""Runtime-requested CPU operational qualifier over registered evidence."""
from dataclasses import replace
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from .. import boundary_qualification as qualification
from ..models import ExpectedOutputSpec, EvidenceStatus
from ..operator_contracts import OperatorContract
from ..production_feedback_request import FrozenProductionFeedbackRequestAuthority
from .boundary_witness_feedback import BoundaryWitnessFeedbackOperator


class BoundaryOperationalQualificationOperator(BoundaryWitnessFeedbackOperator):
    def __init__(self, *, authority, request_authority, evidence_authority, output_root):
        if (type(request_authority) is not FrozenProductionFeedbackRequestAuthority
                or type(evidence_authority) is not qualification.RegisteredBoundaryQualificationEvidence):
            raise ValueError('operational qualification needs registered runtime and production evidence authorities')
        self.request_authority = request_authority; self.request_binding = request_authority.identity()
        self.request = request_authority.read(); self.evidence_authority = evidence_authority
        self.evidence_binding = evidence_authority.identity()
        super().__init__(authority=authority, output_root=output_root)
        if (self.request['feedback_sha256'] != self.summary_hash
                or self.request['request']['request']['implementation_kind'] != 'registered_operator'
                or any(self.request.get(k) is not True for k in (
                    'frozen', 'replay', 'registered_effect_verified', 'native_and_receipt_verified'))):
            raise ValueError('operational qualification was not requested from this exact registered witness')
        self.consumed = evidence_authority.read(self.summary['witness'])
        protected = [Path(self.request_binding['config']),
                     *(Path(p) for p in self.request['artifact_hashes']),
                     *(Path(p) for p in self.consumed['artifact_hashes'])]
        protected.extend(a.base.controller.layout.campaign_dir for a in evidence_authority.authorities.values())
        if any(p.resolve().is_relative_to(self.output_root) or self.output_root.is_relative_to(p.resolve())
               for p in protected):
            raise ValueError('operational qualification outputs overlap original evidence')
        self.spec = replace(self.spec, operator_id='evolution.qualify_boundary_evidence',
            source_path=str(Path(__file__).resolve()), code_revision=payload_sha256({
                'feedback': self.spec.code_revision, 'adapter': file_sha256(Path(__file__)),
                'contract': file_sha256(Path(qualification.__file__)), 'request': self.request_binding,
                'evidence': self.evidence_binding, 'consumed': payload_sha256(self.consumed)}),
            expected_output_specs=(ExpectedOutputSpec('boundary_qualification.json'),))
        self.operator_contract = OperatorContract(self.spec.operator_id, '1', self.spec.sha256(), 'evaluation',
            {'registered_boundary_witness': {'type': 'object'},
             'qualified_independent_evidence': qualification.EVIDENCE_SCHEMA},
            {'operational_qualification': {'type': 'object'}},
            {'type': 'object', 'additionalProperties': False, 'properties': {}}, ('json',),
            'pure_data', 'verify_then_reuse', (
                'Actual engineering and independent production Controller authorities qualify any supplied evidence.',
                'Missing qualified evidence remains insufficient; mask differences and local visual support are not pixel truth.',
                'Full required scope and explicit independent conflict resolution are checked per original instance and claim.',
                'No models, probes, image submission, adoption, promotion or full task completion are authorized.'))

    def _validate(self, call):
        path = super()._validate(call).with_name('boundary_qualification.json')
        if path.is_symlink():
            raise ValueError('boundary qualification cannot be a symlink')
        return path

    def _result(self):
        super()._result()
        if (self.request_authority.identity() != self.request_binding
                or self.request_authority.read() != self.request
                or self.evidence_authority.identity() != self.evidence_binding
                or self.evidence_authority.read(self.summary['witness']) != self.consumed):
            raise ValueError('operational qualification request or independent evidence changed')
        value = qualification.qualify_boundary_witness(self.summary['witness'], self.consumed)
        return {'schema_version': 'dataevolver.registered_boundary_operational_qualification.v1',
            'qualification': value, 'request_sha256': self.request['request_sha256'],
            'feedback_sha256': self.summary_hash, 'evidence_authority_sha256': payload_sha256(self.evidence_binding),
            'new_model_calls': 0, 'new_probe_executions': 0, 'production_qualification_passed': False,
            'image_submission_allowed': False, 'candidate_adoption_allowed': False, 'new_accepted_records': 0}

    def _receipt(self, call, path, reused):
        receipt = super()._receipt(call, path, reused)
        return replace(receipt, stage_result={**receipt.stage_result, 'metrics': {
            'boundary_qualification_diagnostic_captured': 1.0, 'production_qualification_passed': 0.0}})

    def verify_effect(self, call, receipt):
        verified = super().verify_effect(call, receipt)
        if verified.status is EvidenceStatus.VERIFIED:
            _, proof = self.authority.read()
            verified = replace(verified, artifact_hashes={**proof['artifact_hashes'],
                **self.request['artifact_hashes'], **self.consumed['artifact_hashes'], **verified.artifact_hashes})
        return verified
